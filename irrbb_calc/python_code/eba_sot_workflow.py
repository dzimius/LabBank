"""EBA IRRBB Standardized SOT Workflow
========================================
Run order: nii_calc_workflow OR eve_calc_workflow → THIS FILE
(requires irrbb.curves to be populated with shocked discount curves)

Computes the EVE and NII Supervisory Outlier Test from the cash-flow results
(irrbb.eve_results / irrbb.nii_results); the 50% haircut on a gain is applied to each
currency's TOTAL delta (Reg. (EU) 2024/856), not per time bucket.

Produces output/<bank>/eba_sot_results.xlsx:
  EVE_SOT_detail  — per (scenario, currency, tenor bucket of the cash-flow date): PV base,
                    PV shocked, delta_pv -- from the cf.eve_* tables, so the buckets add up
                    exactly to delta_eve of the summary (no haircut at bucket level)
  EVE_SOT_summary — per (scenario, currency): ΔEVE, ΔEVE_reg, sot_eve_pct,
                    sot_eve_pct_reg
  NII_SOT_detail  — per (scenario, currency, tenor bucket of the payment date): NII base,
                    NII shocked, delta_nii -- from irrbb.nii_results_bucket (written by
                    nii_calc_workflow), adding up exactly to delta_nii of the summary
  NII_SOT_summary — per (scenario, currency): ΔNII, ΔNII_reg, sot_nii_pct,
                    sot_nii_pct_reg

Currencies: every summary amount is in the bank's HOME currency (foreign-currency NII / EVE
converted at the report-date FX rate, labbank_common/fx.py), so % of Tier 1 is comparable.
A bank with more than one currency also gets a currency = 'ALL' row per scenario: plain sum
for base / shocked / delta, and for the regulatory delta the EBA aggregation across
currencies -- losses in full, gains at 50% (sum of the per-currency *_reg values).
The detail sheets are in the home currency too (same FX).

Why bucket-level 50% differs from total-level 50%:
  Applying the haircut per bucket prevents positive contributions in some
  buckets from fully offsetting losses in others.  A bank with a positive
  gap in short tenors and negative in long tenors will show a more
  conservative (larger) adverse impact under the regulatory calculation
  than under a simple total-level haircut.
"""
import os
import pandas as pd

import config
import sql_setup
from labbank_common.bank_config import get_bank   # sql_setup puts the project root on sys.path
import eba_shock_curves as esc

import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from labbank_common import irrbb_behaviour as beh_scen
from labbank_common import fx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
OUT_DIR = get_bank().output_dir('irrbb_calc')

config.report_date = pd.Timestamp(get_bank().report_date)

HORIZON_YF    = 1.0
CURRENCY      = get_bank().market.ccy
OWN_SHOCK_BPS = -100.0

DISC_CURVE_MAP = {
    "PLN": "PLN_disc_curve",
    "EUR": "EUR_disc_curve",
    "USD": "USD_disc_curve", "GBP": "GBP_disc_curve",
}

# ── 0. Load inputs ─────────────────────────────────────────────────────────────
TIER1_CAPITAL = sql_setup.load_tier1_capital(config.report_date)
print(f"Tier 1 capital: {TIER1_CAPITAL:,.0f}")

mkt_df = sql_setup.load_mkt_curves(config.report_date)
if mkt_df.empty:
    raise RuntimeError("No market curves found. Run cash_flow_calc / nii_calc_workflow first.")


# CF-based NII totals per scenario (base + all shocked) — for accurate delta_nii
nii_cf = sql_setup.load_nii_summary(config.report_date)
if nii_cf.empty:
    raise RuntimeError("No NII results found. Run nii_calc_workflow first.")
nii_cf["nii_total"] = nii_cf["nii_total"] * fx.to_home_series(nii_cf["currency"], get_bank(), config.report_date)
nii_base_map = (
    nii_cf[nii_cf["scenario_id"] == "base"]
    .set_index("currency")["nii_total"]
    .to_dict()
)

# EVE uses run-off CFs across all maturities — load pre-computed PV from eve_results
eve_cf = sql_setup.load_eve_summary(config.report_date)
if eve_cf.empty:
    raise RuntimeError("No EVE results found. Run eve_calc_workflow first.")
eve_cf["pv_total"] = eve_cf["pv_total"] * fx.to_home_series(eve_cf["currency"], get_bank(), config.report_date)
eve_base_map = (
    eve_cf[eve_cf["scenario_id"] == "base"]
    .set_index("currency")["pv_total"]
    .to_dict()
)

# EVE uses run-off CFs across all maturities
beh_df = sql_setup.load_all_beh_schedules(config.report_date)
try:
    swap_df = sql_setup.load_all_swap_beh_schedules(config.report_date)
except Exception:
    swap_df = pd.DataFrame()
all_beh = pd.concat([beh_df, swap_df], ignore_index=True) if not swap_df.empty else beh_df
print(f"Loaded {len(all_beh):,} CF rows ({all_beh['schedule_id'].nunique()} schedules)")
scen_beh = sql_setup.load_beh_variants(config.report_date)   # EBA CPR/TDRR scenario variants

# ── 1. Run SOT for all 7 stressed scenarios ────────────────────────────────────
print("\nComputing EBA SOT (bucket-level 50% haircut)...")
print(f"{'Scenario':<10} {'dEVE':>14} {'dEVE_reg':>14} {'dNII':>14} {'dNII_reg':>14}")
print("-" * 60)

eve_summary_parts = []
nii_summary_parts = []

for scenario_id in esc.ALL_SCENARIO_IDS:
    if scenario_id == "base":
        continue

    disc_parts = []
    for ccy, curve_name in DISC_CURVE_MAP.items():
        try:
            disc_parts.append(
                esc.load_irrbb_disc_curve(
                    sql_setup.engine, curve_name, scenario_id, config.report_date
                )
            )
        except Exception:
            pass

    if not disc_parts:
        print(f"  Skipping '{scenario_id}': no disc curves in irrbb.curves.")
        continue

    shocked_disc = pd.concat(disc_parts).sort_index()


    # EVE summary — CF-based delta (shocked absolute EVE minus base), per currency
    eve_rows = []
    shocked_eve_rows = eve_cf[eve_cf["scenario_id"] == scenario_id]
    for _, row in shocked_eve_rows.iterrows():
        ccy      = row["currency"]
        base_eve = eve_base_map.get(ccy, 0.0)
        delta    = float(row["pv_total"]) - base_eve
        delta_reg = delta if delta <= 0 else 0.5 * delta
        eve_rows.append({
            "scenario_id":   scenario_id,
            "currency":      ccy,
            "eve_base":      base_eve,
            "eve_shocked":   float(row["pv_total"]),
            "delta_eve":     delta,
            "delta_eve_reg": delta_reg,
        })
    eve_summ = pd.DataFrame(eve_rows)
    eve_summary_parts.append(eve_summ)


    # NII summary — CF-based delta (shocked absolute NII minus base), per currency
    nii_rows = []
    shocked_nii_rows = nii_cf[nii_cf["scenario_id"] == scenario_id]
    for _, row in shocked_nii_rows.iterrows():
        ccy       = row["currency"]
        base_nii  = nii_base_map.get(ccy, 0.0)
        delta     = float(row["nii_total"]) - base_nii
        delta_reg = delta if delta <= 0 else 0.5 * delta
        nii_rows.append({
            "scenario_id":   scenario_id,
            "currency":      ccy,
            "nii_base":      base_nii,
            "nii_shocked":   float(row["nii_total"]),
            "delta_nii":     delta,
            "delta_nii_reg": delta_reg,
        })
    nii_summ = pd.DataFrame(nii_rows)
    nii_summary_parts.append(nii_summ)

    dv  = eve_summ["delta_eve"].sum()
    dvr = eve_summ["delta_eve_reg"].sum()
    dn  = nii_summ["delta_nii"].sum()
    dnr = nii_summ["delta_nii_reg"].sum()
    print(f"{scenario_id:<10} {dv:>14,.0f} {dvr:>14,.0f} {dn:>14,.0f} {dnr:>14,.0f}")

# ── 2. Combine and add SOT % ───────────────────────────────────────────────────
def _bucket_detail(df: pd.DataFrame, value: str, prefix: str) -> pd.DataFrame:
    """scenario x currency x tenor bucket: base, shocked and delta of `value` (home currency),
    from per-cash-flow results -- sums exactly to the summary delta."""
    if df.empty:
        return pd.DataFrame()
    df = df.assign(**{value: df[value] * fx.to_home_series(df["currency"], get_bank(),
                                                           config.report_date).to_numpy()})
    base = (df[df["scenario_id"] == "base"].groupby(["currency", "tenor_bucket"])[value].sum()
            .rename(f"{prefix}_base"))
    sh = df[df["scenario_id"] != "base"].set_index(["currency", "tenor_bucket"])
    out = sh.join(base, how="left").reset_index()
    out[f"{prefix}_base"] = out[f"{prefix}_base"].fillna(0.0)
    out = out.rename(columns={value: f"{prefix}_shocked"})
    out[f"delta_{prefix}"] = out[f"{prefix}_shocked"] - out[f"{prefix}_base"]
    _order = {lbl: i for i, (_c, lbl) in enumerate(sql_setup._TENOR_BUCKET_CUTOFFS)}
    out["_s"] = out["tenor_bucket"].map(lambda b: _order.get(b, 999))
    return (out.sort_values(["scenario_id", "currency", "_s"]).drop(columns="_s")
            [["scenario_id", "currency", "tenor_bucket", f"{prefix}_base", f"{prefix}_shocked",
              f"delta_{prefix}"]].reset_index(drop=True))


from sqlalchemy import text as _text
_eve_cf_rows = pd.concat([pd.read_sql_query(_text(
    f"SELECT scenario_id, currency, tenor_bucket, SUM(pv_capital + pv_interest) AS pv "
    f"FROM cf.{t} WHERE report_date = :rd GROUP BY scenario_id, currency, tenor_bucket"),
    sql_setup.engine, params={"rd": config.report_date})
    for t in ("eve_base_scenario", "eve_par_scenarios", "eve_short_scenarios",
              "eve_step_flat_scenarios", "eve_own_scenarios")], ignore_index=True)
_eve_cf_rows = _eve_cf_rows[_eve_cf_rows["scenario_id"].isin(esc.ALL_SCENARIO_IDS)].drop_duplicates()
eve_detail = _bucket_detail(_eve_cf_rows, "pv", "pv")
try:
    _nii_b = pd.read_sql_query(_text(
        "SELECT scenario_id, currency, tenor_bucket, nii_total FROM irrbb.nii_results_bucket "
        "WHERE report_date = :rd"), sql_setup.engine, params={"rd": config.report_date})
except Exception as _e:                     # nii_calc_workflow not re-run since this change
    print(f"  NII detail skipped: irrbb.nii_results_bucket unavailable ({type(_e).__name__})")
    _nii_b = pd.DataFrame()
nii_detail = _bucket_detail(_nii_b, "nii_total", "nii")
eve_summary = pd.concat(eve_summary_parts, ignore_index=True) if eve_summary_parts else pd.DataFrame()
nii_summary = pd.concat(nii_summary_parts, ignore_index=True) if nii_summary_parts else pd.DataFrame()

def _add_all_ccy(summary: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Append a currency='ALL' row per scenario (multi-currency banks only): sum of the
    home-currency amounts; *_reg columns sum the per-currency regulatory deltas (losses in
    full, gains at 50%) = the EBA aggregation across currencies."""
    if summary.empty or summary["currency"].nunique() < 2:
        return summary
    tot = summary.groupby("scenario_id", sort=False)[cols].sum().reset_index()
    tot["currency"] = "ALL"
    return pd.concat([summary, tot[summary.columns]], ignore_index=True)


eve_summary = _add_all_ccy(eve_summary, ["eve_base", "eve_shocked", "delta_eve", "delta_eve_reg"])
nii_summary = _add_all_ccy(nii_summary, ["nii_base", "nii_shocked", "delta_nii", "delta_nii_reg"])

if not eve_summary.empty:
    eve_summary["sot_eve_pct"]     = eve_summary["delta_eve"]     / TIER1_CAPITAL * 100.0
    eve_summary["sot_eve_pct_reg"] = eve_summary["delta_eve_reg"] / TIER1_CAPITAL * 100.0
    eve_summary["sot_breach"]      = eve_summary["sot_eve_pct_reg"] < -15.0

if not nii_summary.empty:
    nii_summary["sot_nii_pct"]     = nii_summary["delta_nii"]     / TIER1_CAPITAL * 100.0
    nii_summary["sot_nii_pct_reg"] = nii_summary["delta_nii_reg"] / TIER1_CAPITAL * 100.0
    # The supervisory NII SOT uses the two PARALLEL scenarios only (Reg. (EU) 2024/856);
    # the other scenarios stay as internal information, never flagged as an SOT breach.
    nii_summary["regulatory_test"] = nii_summary["scenario_id"].isin(["par_up", "par_dn"])
    nii_summary["sot_breach"]      = nii_summary["regulatory_test"] & (nii_summary["sot_nii_pct_reg"] < -5.0)

# ── 3. Write final SOT values to irrbb.irrbb_report ───────────────────────────
# This is the sole writer to irrbb.irrbb_report.
# NII delta comes from CF-based nii_results (includes IRS, per-product rates).
# EVE delta comes from CF-based eve_results (full repricing with cap/floor/coeff).
# EVE_SOT_detail / NII_SOT_detail bucket the same cash-flow results (they sum to these deltas).
sql_setup.reset_irrbb_report()

if not nii_summary.empty:
    nii_report = nii_summary[["scenario_id", "currency",
                               "delta_nii", "delta_nii_reg",
                               "sot_nii_pct", "sot_nii_pct_reg"]].copy()
    sql_setup.upsert_irrbb_report(nii_report, config.report_date, mode="nii")
    print("  NII SOT written to irrbb.irrbb_report")

if not eve_summary.empty:
    eve_report = eve_summary[["scenario_id", "currency",
                               "delta_eve", "delta_eve_reg",
                               "sot_eve_pct", "sot_eve_pct_reg"]].copy()
    sql_setup.upsert_irrbb_report(eve_report, config.report_date, mode="eve")
    print("  EVE SOT written to irrbb.irrbb_report")

# ── 4. Write results.irrbb_report (base + shocked absolute values) ────────────
# Merges nii_summary and eve_summary into the combined results schema table so
# the user does NOT need to run lcr_nsfr_workflow just to populate irrbb_report.
print("\nWriting results.irrbb_report...")
sql_setup.reset_results_irrbb_report()

if not nii_summary.empty and not eve_summary.empty:
    # Build a per-(scenario,currency) frame with both NII and EVE columns
    nii_cols = nii_summary[["scenario_id", "currency", "nii_base", "nii_shocked",
                             "delta_nii", "sot_nii_pct"]].copy()
    eve_cols = eve_summary[["scenario_id", "currency", "eve_base", "eve_shocked",
                             "delta_eve", "sot_eve_pct"]].copy()
    results_df = nii_cols.merge(eve_cols, on=["scenario_id", "currency"], how="outer")
    results_df["report_date"] = config.report_date

    # Add base row (delta = 0 by definition) for each currency
    currencies = results_df["currency"].dropna().unique().tolist()
    base_rows = []
    for ccy in currencies:
        nii_b = sum(nii_base_map.values()) if ccy == "ALL" else nii_base_map.get(ccy, 0.0)
        eve_b = sum(eve_base_map.values()) if ccy == "ALL" else eve_base_map.get(ccy, 0.0)
        base_rows.append({
            "report_date": config.report_date,
            "currency":    ccy,
            "scenario_id": "base",
            "nii_base":    nii_b,
            "nii_shocked": nii_b,
            "delta_nii":   0.0,
            "sot_nii_pct": 0.0,
            "eve_base":    eve_b,
            "eve_shocked": eve_b,
            "delta_eve":   0.0,
            "sot_eve_pct": 0.0,
        })
    results_df = (
        pd.concat([pd.DataFrame(base_rows), results_df], ignore_index=True)
        .sort_values(["currency", "scenario_id"])
        .reset_index(drop=True)
    )

    sql_setup.write_results_irrbb_report(results_df)
    print(f"  Written {len(results_df)} rows to results.irrbb_report.")

    # Also write irrbb_report.xlsx to output/<bank>/ for direct inspection
    irrbb_report_path = str(OUT_DIR / "irrbb_report.xlsx")
    with pd.ExcelWriter(irrbb_report_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="IRRBB_all", index=False)
        (results_df[["currency", "scenario_id", "nii_base", "nii_shocked", "delta_nii", "sot_nii_pct"]]
         .sort_values(["currency", "scenario_id"])
         .to_excel(writer, sheet_name="Delta_NII", index=False))
        (results_df[["currency", "scenario_id", "eve_base", "eve_shocked", "delta_eve", "sot_eve_pct"]]
         .sort_values(["currency", "scenario_id"])
         .to_excel(writer, sheet_name="Delta_EVE", index=False))
    print(f"  IRRBB report Excel written to {irrbb_report_path}")
else:
    print("  Skipped: nii_summary or eve_summary is empty.")

# ── 5. Write EBA SOT detail Excel ──────────────────────────────────────────────
output_path = str(OUT_DIR / "eba_sot_results.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    if not eve_detail.empty:
        (eve_detail
         .drop(columns=["bucket_order"], errors="ignore")
         .to_excel(writer, sheet_name="EVE_SOT_detail", index=False))

    if not eve_summary.empty:
        eve_summary.to_excel(writer, sheet_name="EVE_SOT_summary", index=False)

    if not nii_detail.empty:
        (nii_detail
         .drop(columns=["bucket_order"], errors="ignore")
         .to_excel(writer, sheet_name="NII_SOT_detail", index=False))

    if not nii_summary.empty:
        nii_summary.to_excel(writer, sheet_name="NII_SOT_summary", index=False)

print(f"\nEBA SOT results written to {output_path}")

# ── 6. Print SOT breach summary ────────────────────────────────────────────────
print("\n-- EVE SOT summary (threshold: -15% of Tier 1) --")
if not eve_summary.empty:
    print(eve_summary[["scenario_id", "currency", "eve_base", "eve_shocked",
                        "delta_eve", "delta_eve_reg",
                        "sot_eve_pct", "sot_eve_pct_reg", "sot_breach"]]
          .to_string(index=False))

print("\n-- NII SOT summary (threshold: -5% of Tier 1) --")
if not nii_summary.empty:
    print(nii_summary[["scenario_id", "currency", "nii_base", "nii_shocked",
                        "delta_nii", "delta_nii_reg",
                        "sot_nii_pct", "sot_nii_pct_reg", "regulatory_test", "sot_breach"]]
          .to_string(index=False))
