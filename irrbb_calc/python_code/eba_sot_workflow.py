"""EBA IRRBB Standardized SOT Workflow
========================================
Run order: nii_calc_workflow OR eve_calc_workflow → THIS FILE
(requires irrbb.curves to be populated with shocked discount curves)

Computes EVE and NII SOT per EBA/GL/2018/02 Annex III with the 50%
supervisory haircut applied at the TIME-BUCKET level — NOT at the total.

Produces output/eba_sot_results.xlsx:
  EVE_SOT_detail  — per (scenario, currency, EBA bucket): net_cf, d_f_base,
                    d_f_shocked, delta_pv, delta_pv_reg
  EVE_SOT_summary — per (scenario, currency): ΔEVE, ΔEVE_reg, sot_eve_pct,
                    sot_eve_pct_reg
  NII_SOT_detail  — per (scenario, currency, tenor bucket): gap_cf, Δr,
                    remain_yf, delta_nii, delta_nii_reg
  NII_SOT_summary — per (scenario, currency): ΔNII, ΔNII_reg, sot_nii_pct,
                    sot_nii_pct_reg

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
import eba_shock_curves as esc
import eba_sot_objects as sot

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/irrbb_calc"
os.chdir(BASE_DIR)

config.report_date = pd.to_datetime("2024-12-31")

HORIZON_YF    = 1.0
CURRENCY      = "PLN"
OWN_SHOCK_BPS = -100.0

DISC_CURVE_MAP = {
    "PLN": "PLN_disc_curve",
    "EUR": "EUR_disc_curve",
    "USD": "USD_disc_curve",
}

# ── 0. Load inputs ─────────────────────────────────────────────────────────────
TIER1_CAPITAL = sql_setup.load_tier1_capital(config.report_date)
print(f"Tier 1 capital: {TIER1_CAPITAL:,.0f}")

mkt_df = sql_setup.load_mkt_curves(config.report_date)
if mkt_df.empty:
    raise RuntimeError("No market curves found. Run cash_flow_calc / nii_calc_workflow first.")

ir_gap = sql_setup.load_ir_gap_beh()

# CF-based NII totals per scenario (base + all shocked) — for accurate delta_nii
nii_cf = sql_setup.load_nii_summary(config.report_date)
if nii_cf.empty:
    raise RuntimeError("No NII results found. Run nii_calc_workflow first.")
nii_base_map = (
    nii_cf[nii_cf["scenario_id"] == "base"]
    .set_index("currency")["nii_total"]
    .to_dict()
)

# EVE uses run-off CFs across all maturities — load pre-computed PV from eve_results
eve_cf = sql_setup.load_eve_summary(config.report_date)
if eve_cf.empty:
    raise RuntimeError("No EVE results found. Run eve_calc_workflow first.")
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

# ── 1. Run SOT for all 7 stressed scenarios ────────────────────────────────────
print("\nComputing EBA SOT (bucket-level 50% haircut)...")
print(f"{'Scenario':<10} {'ΔEVE':>14} {'ΔEVE_reg':>14} {'ΔNII':>14} {'ΔNII_reg':>14}")
print("-" * 60)

eve_detail_parts  = []
eve_summary_parts = []
nii_detail_parts  = []
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

    # EVE SOT (run-off, all maturities) — bucket detail only (EBA standardised method)
    eve_det, _ = sot.compute_eve_sot(
        all_beh, mkt_df, shocked_disc, DISC_CURVE_MAP,
        config.report_date, scenario_id,
    )
    eve_detail_parts.append(eve_det)

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

    # NII SOT — detail bucketing via gap-based method (for bucket breakdown)
    nii_det, _ = sot.compute_nii_sot(
        ir_gap, mkt_df, shocked_disc, DISC_CURVE_MAP,
        config.report_date, scenario_id, HORIZON_YF,
    )
    nii_detail_parts.append(nii_det)

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
eve_detail  = pd.concat(eve_detail_parts,  ignore_index=True) if eve_detail_parts  else pd.DataFrame()
eve_summary = pd.concat(eve_summary_parts, ignore_index=True) if eve_summary_parts else pd.DataFrame()
nii_detail  = pd.concat(nii_detail_parts,  ignore_index=True) if nii_detail_parts  else pd.DataFrame()
nii_summary = pd.concat(nii_summary_parts, ignore_index=True) if nii_summary_parts else pd.DataFrame()

if not eve_summary.empty:
    eve_summary["sot_eve_pct"]     = eve_summary["delta_eve"]     / TIER1_CAPITAL * 100.0
    eve_summary["sot_eve_pct_reg"] = eve_summary["delta_eve_reg"] / TIER1_CAPITAL * 100.0
    eve_summary["sot_breach"]      = eve_summary["sot_eve_pct_reg"] < -15.0

if not nii_summary.empty:
    nii_summary["sot_nii_pct"]     = nii_summary["delta_nii"]     / TIER1_CAPITAL * 100.0
    nii_summary["sot_nii_pct_reg"] = nii_summary["delta_nii_reg"] / TIER1_CAPITAL * 100.0
    nii_summary["sot_breach"]      = nii_summary["sot_nii_pct_reg"] < -5.0

# ── 3. Write final SOT values to irrbb.irrbb_report ───────────────────────────
# This is the sole writer to irrbb.irrbb_report.
# NII delta comes from CF-based nii_results (includes IRS, per-product rates).
# EVE delta comes from CF-based eve_results (full repricing with cap/floor/coeff).
# EVE_SOT_detail uses the EBA bucket method (compute_eve_sot) for bucket breakdown.
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
        nii_b = nii_base_map.get(ccy, 0.0)
        eve_b = eve_base_map.get(ccy, 0.0)
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

    # Also write irrbb_report.xlsx to output/ for direct inspection
    irrbb_report_path = "output/irrbb_report.xlsx"
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
output_path = "output/eba_sot_results.xlsx"

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
print("\n── EVE SOT summary (threshold: −15% of Tier 1) ──")
if not eve_summary.empty:
    print(eve_summary[["scenario_id", "currency", "eve_base", "eve_shocked",
                        "delta_eve", "delta_eve_reg",
                        "sot_eve_pct", "sot_eve_pct_reg", "sot_breach"]]
          .to_string(index=False))

print("\n── NII SOT summary (threshold: −5% of Tier 1) ──")
if not nii_summary.empty:
    print(nii_summary[["scenario_id", "currency", "nii_base", "nii_shocked",
                        "delta_nii", "delta_nii_reg",
                        "sot_nii_pct", "sot_nii_pct_reg", "sot_breach"]]
          .to_string(index=False))
