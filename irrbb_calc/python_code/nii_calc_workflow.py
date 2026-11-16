"""NII Calculation Workflow
===========================
Run order: balance_generate → balance_gen_add_data → cash_flow_calc
           → ir_derivatives → THIS FILE

Produces
--------
1. irrbb.curves (SQL)         — base + 6 EBA SOT + own shocked discount curves
2. output/<bank>/irrbb_shock_curves.xlsx — day-by-day shocked curve shapes (audit/review)
3. output/<bank>/nii_results.xlsx    — all NII outputs:
     NII_simple_detail / NII_simple_summary  — gap-based simple NII (±100 bps)
     NII_base_detail   / NII_base_summary    — CF-based base NII (absolute income)
     NII_shocked_detail / NII_shocked_summary— cash-flow ΔNII per scenario; the detail per
                                               tenor bucket of the payment date (own currency)
                                               sums exactly to the summary
"""
import os
import pandas as pd

import config
import sql_setup
from labbank_common.bank_config import get_bank   # sql_setup puts the project root on sys.path
import nii_calc_objects as nii_obj
import eve_calc_objects as eve_obj
import eba_shock_curves as esc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
OUT_DIR = get_bank().output_dir('irrbb_calc')

config.report_date = pd.Timestamp(get_bank().report_date)

HORIZON_YF    = 1.0
CURRENCY      = get_bank().market.ccy   # home currency of the bank: EBA shock calibration
OWN_SHOCK_BPS = -100.0          # own scenario: -100 bps parallel
CURVE_FLOOR   = None            # post-shock curve floor: EBA lower bound (Reg. 2024/856 Art. 3(7))
NII_FLOOR     = float('-inf')   # no floor on PROJECTED index rates: product floors/caps come from
                                # the rate model (bs.models_rate), the curve floor from CURVE_FLOOR

horizon_end = config.report_date + pd.Timedelta(days=round(HORIZON_YF * 365.25))

DISC_CURVE_MAP = {
    "PLN": "PLN_disc_curve",
    "EUR": "EUR_disc_curve",
    "USD": "USD_disc_curve", "GBP": "GBP_disc_curve"
}

# Per-product client-rate model (beta, spread, floors, caps) from bs.models_rate --
# same source as the CF engine and the NII/EVE fast layer (labbank_common.rate_models)
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from labbank_common import rate_models
from labbank_common import irrbb_behaviour as beh_scen
CAPS_MAP, FLOORS_MAP, COEFF_A_MAP, COEFF_B_MAP = rate_models.rate_maps(
    rate_models.load_rate_models(
        sql_setup.engine, config.report_date,
        required_codes=rate_models.deposit_product_codes(sql_setup.engine),
    )
)

# ── 0. Load Tier 1 capital ────────────────────────────────────────────────────
TIER1_CAPITAL = sql_setup.load_tier1_capital(config.report_date)
print(f"Tier 1 capital loaded from schemat.equity: {TIER1_CAPITAL:,.0f}")

# ── 1. Build and store shocked yield curves ───────────────────────────────────
print(config.report_date, "Building EBA SOT shocked curves...")
mkt_df = sql_setup.load_mkt_curves(config.report_date)
if mkt_df.empty:
    raise RuntimeError("No market curves found in mkt.curves for report_date. "
                       "Run cash_flow_calc workflow first.")

shocked_df = esc.build_all_shocked_curves(
    mkt_df, config.report_date, CURRENCY, OWN_SHOCK_BPS, CURVE_FLOOR
)
esc.write_irrbb_curves(sql_setup.engine, shocked_df, config.report_date)
print(f"  Written {len(shocked_df):,} rows to irrbb.curves "
      f"({len(esc.ALL_SCENARIO_IDS)} scenarios × "
      f"{mkt_df.groupby('curve_name').ngroups} curves)")

# Append own scenarios (±100 bps, ±1 bps) to irrbb.curves
_own_curves_df = esc.build_own_shocked_curves(mkt_df, config.report_date, nii_floor_rate=CURVE_FLOOR)
esc.write_irrbb_curves(sql_setup.engine, _own_curves_df, config.report_date, replace=False)
print(f"  Appended {len(_own_curves_df):,} own-scenario rows to irrbb.curves "
      f"({esc.OWN_SCENARIO_IDS})")

esc.generate_shock_excel(
    mkt_df, config.report_date, CURRENCY,
    output_path=str(OUT_DIR / "irrbb_shock_curves.xlsx"),
    own_shock_bps=OWN_SHOCK_BPS,
    nii_floor_rate=CURVE_FLOOR,
    horizon_days=round(HORIZON_YF * 365),
)

# Multi-curve bank (OIS discounting + IBOR projection curves, labbank_common/curves.py): NII
# discounts nothing -- every forward and renewal rate below comes from the currency's
# projection curve (base and shocked alike, so a zero shock reproduces the base).
from labbank_common.curves import projection_curve_map
if mkt_df["curve_name"].str.contains("_fwd_curve_").any():
    DISC_CURVE_MAP = {c: n for c, n in projection_curve_map(DISC_CURVE_MAP).items()
                      if n in set(mkt_df["curve_name"])}
    print(f"  Multi-curve: NII forwards / renewals on {DISC_CURVE_MAP}")

# ── 2. Simple NII (repricing gap based, ±100 bps shocks) ──────────────────────
print("Computing simple (gap-based) NII...")
ir_gap_beh = sql_setup.load_ir_gap_beh()


def _beta_adjusted_gap_by_bucket() -> pd.DataFrame:
    """currency x tenor_bucket beta-adjusted gap (balance sheet / IRS) from irrbb.ir_gap_beh_a,
    with the same rule as the ALM report chapter 3c (labbank_common.rate_models)."""
    import numpy as _np
    from labbank_common import rate_models as _rm
    from sqlalchemy import text as _t
    ga = pd.read_sql_query(_t("""
        SELECT g.currency, CAST(g.product_code AS VARCHAR(4)) AS product_code, g.bs_side,
               g.tenor_bucket, SUM(g.gap_cf) AS gap_cf
        FROM irrbb.ir_gap_beh_a g
        GROUP BY g.currency, g.product_code, g.bs_side, g.tenor_bucket"""), sql_setup.engine)
    names = pd.read_sql_query(_t("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code, MAX(product_name) AS product_name
        FROM (SELECT product_code, product_name FROM schemat.loans WHERE report_date = :rd
              UNION ALL SELECT product_code, product_name FROM schemat.deposits WHERE report_date = :rd
              UNION ALL SELECT product_code, product_name FROM schemat.financial_instruments
                        WHERE report_date = :rd) u GROUP BY product_code"""),
        sql_setup.engine, params={"rd": config.report_date})
    ga = ga.merge(names, on="product_code", how="left")
    # today's 1M index per currency from the base curve (continuous zero, ~30 days)
    idx = {}
    for ccy in ga["currency"].unique():
        cn = DISC_CURVE_MAP.get(ccy)
        c = mkt_df[mkt_df["curve_name"] == cn] if cn else pd.DataFrame()
        if not c.empty:
            r = c.iloc[(c["n_days"] - 30).abs().argmin()]
            idx[ccy] = float(-_np.log(r["d_f"]) / (r["n_days"] / 365.0)) if r["n_days"] > 0 else 0.0
    ccy_of = ga.drop_duplicates("product_code").set_index("product_code")["currency"].to_dict()
    betas = _rm.effective_betas(_rm.load_rate_models(sql_setup.engine, config.report_date),
                                {pc: idx.get(c) for pc, c in ccy_of.items() if c in idx})
    ga["_s"] = ga["tenor_bucket"].map(nii_obj._tenor_sort_key)
    ga = ga.sort_values(["currency", "product_code", "bs_side", "_s"])
    ga["gap_beta"] = 0.0
    for (_c, pc, _b), ix in ga.groupby(["currency", "product_code", "bs_side"]).groups.items():
        g = ga.loc[ix]
        ga.loc[ix, "gap_beta"] = _rm.beta_adjusted_gap(g["gap_cf"].to_numpy(), betas.get(pc, 1.0))
    ga["kind"] = _np.where(ga["product_code"] == "0000", "gap_beta_irs", "gap_beta_bs")
    return (ga.pivot_table(index=["currency", "tenor_bucket"], columns="kind", values="gap_beta",
                           aggfunc="sum").reset_index())


simple_detail, simple_summary = nii_obj.compute_nii(
    ir_gap_beh, horizon_yf=HORIZON_YF, gap_beta=_beta_adjusted_gap_by_bucket(),
)

# ── 3. Base NII (CF-based, no shock) — absolute income level ──────────────────
print("Computing base NII (CF-based)...")
beh_df = sql_setup.load_beh_schedules(config.report_date, horizon_end)

try:
    swap_df = sql_setup.load_swap_beh_schedules(config.report_date, horizon_end)
except Exception:
    swap_df = pd.DataFrame()
all_beh = pd.concat([beh_df, swap_df], ignore_index=True) if not swap_df.empty else beh_df

# Scenario variants (EBA RTS 2024/857 Art. 9-10: CPR / TDRR x gamma), swapped in per scenario
scen_beh = sql_setup.load_beh_variants(config.report_date, horizon_end)
print(f"  Loaded {len(scen_beh):,} scenario-variant CF rows (cf.products_scen); "
      f"EBA CPR/TDRR multipliers {'ON' if beh_scen.APPLY_EBA_BEHAVIOUR else 'OFF (baseline behaviour)'}")

# Load the base discount curve once — used for renewal rate derivation in
# compute_nii_base_schedule (same formula as shocked scenario).
_base_disc_parts_step3 = []
for _ccy, _curve_name in DISC_CURVE_MAP.items():
    try:
        _base_disc_parts_step3.append(
            esc.load_irrbb_disc_curve(
                sql_setup.engine, _curve_name, "base", config.report_date
            )
        )
    except Exception:
        pass
_base_disc_step3 = (
    pd.concat(_base_disc_parts_step3).sort_index()
    if _base_disc_parts_step3 else None
)

base_detail, base_summary = nii_obj.compute_nii_base(
    all_beh, config.report_date, horizon_yf=HORIZON_YF,
    caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
    coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
    base_disc_df=_base_disc_step3, disc_curve_map=DISC_CURVE_MAP,
)
print(f"  Base NII: {base_summary['nii_total'].sum():+,.0f}")

# Per-cash-flow NII rows -> schedule rows (irrbb.nii_results) and tenor buckets of the payment
# date (irrbb.nii_results_bucket: the SOT detail sheet, which then sums to the headline)
_NII_COLS = ["nii_interest", "nii_renewal", "nii_total"]
_bucket_parts = []


def _schedule_and_buckets(rows: pd.DataFrame, sid: str) -> pd.DataFrame:
    rows = rows.assign(tenor_bucket=sql_setup._assign_tenor_bucket(rows["cf_end_dt"], config.report_date))
    _bucket_parts.append(rows.groupby(["currency", "tenor_bucket"])[_NII_COLS].sum()
                         .reset_index().assign(scenario_id=sid))
    gc = [c for c in nii_obj._SCHED_GROUP_COLS if c in rows.columns]
    return rows.groupby(gc)[_NII_COLS].sum().reset_index().assign(scenario_id=sid)


# Write base schedule-level NII to SQL
sql_setup.reset_nii_results()
base_sched = _schedule_and_buckets(nii_obj.compute_nii_shocked_schedule(
    all_beh, _base_disc_step3, DISC_CURVE_MAP, config.report_date, "base", HORIZON_YF, NII_FLOOR,
    caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
    coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
    base_disc_df=_base_disc_step3, return_rows=True,
), "base")
sql_setup.write_nii_results(base_sched, config.report_date)
print(f"  {'base':8s}: NII = {base_sched['nii_total'].sum():+,.0f}  "
      f"({len(base_sched)} schedule rows written to irrbb.nii_results)")

# ── 4. Shocked NII schedules for all EBA scenarios (CF-based, absolute) ───────
print("Computing shocked NII schedules for all scenarios (CF-based)...")

for scenario_id in esc.ALL_SCENARIO_IDS:
    if scenario_id == "base":
        continue

    disc_series_parts = []
    for ccy, curve_name in DISC_CURVE_MAP.items():
        try:
            s = esc.load_irrbb_disc_curve(
                sql_setup.engine, curve_name, scenario_id, config.report_date
            )
            disc_series_parts.append(s)
        except Exception:
            pass

    if not disc_series_parts:
        print(f"  Skipping scenario '{scenario_id}': no disc curves found.")
        continue

    shocked_disc = pd.concat(disc_series_parts).sort_index()

    # CF-based absolute NII for this scenario → irrbb.nii_results
    shocked_sched = _schedule_and_buckets(nii_obj.compute_nii_shocked_schedule(
        beh_scen.nmd_migration(all_beh, beh_scen.apply_variant(all_beh, scen_beh, scenario_id)),
        shocked_disc, DISC_CURVE_MAP,
        config.report_date, scenario_id, HORIZON_YF, NII_FLOOR,
        caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
        coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
        base_disc_df=_base_disc_step3, return_rows=True,
    ), scenario_id)
    sql_setup.write_nii_results(shocked_sched, config.report_date)

sql_setup.run_store.replace_run(sql_setup.engine, pd.concat(_bucket_parts, ignore_index=True),
                                "nii_results_bucket", "irrbb")      # history per (report_date, run_id)
print(f"  irrbb.nii_results_bucket: NII per scenario x currency x tenor bucket")

# NII_shocked_detail: per scenario x currency x tenor bucket, shocked - base (own currency)
_bk = pd.concat(_bucket_parts, ignore_index=True)
_bk_base = _bk[_bk["scenario_id"] == "base"].set_index(["currency", "tenor_bucket"])["nii_total"]
shocked_detail = _bk[_bk["scenario_id"] != "base"].rename(columns={"nii_total": "nii_shocked"})
shocked_detail["nii_base"] = [_bk_base.get((c, b), 0.0) for c, b in
                              zip(shocked_detail["currency"], shocked_detail["tenor_bucket"])]
shocked_detail["delta_nii"] = shocked_detail["nii_shocked"] - shocked_detail["nii_base"]
shocked_detail = shocked_detail[["scenario_id", "currency", "tenor_bucket", "nii_base",
                                 "nii_shocked", "delta_nii"]]
_ord = {lbl: i for i, (_c, lbl) in enumerate(sql_setup._TENOR_BUCKET_CUTOFFS)}
shocked_detail = (shocked_detail.assign(_s=shocked_detail["tenor_bucket"].map(lambda b: _ord.get(b, 999)))
                  .sort_values(["scenario_id", "currency", "_s"]).drop(columns="_s").reset_index(drop=True))

# ── 4b. NII analytical CF tables ──────────────────────────────────────────────
# nii_base_scenario  — base rates, horizon-cut CFs
# nii_par_scenarios  — par_up / par_dn shocked CFs
# nii_own_scenarios  — own_100_up/dn, own_1_up/dn shocked CFs
print("Writing NII analytical CF tables to SQL...")

for _t in ("nii_base_scenario", "nii_par_scenarios", "nii_own_scenarios"):
    sql_setup.reset_shocked_cf_products(_t, table_type="nii")

# Base
_base_disc_parts = []
for ccy, curve_name in DISC_CURVE_MAP.items():
    try:
        _base_disc_parts.append(
            esc.load_irrbb_disc_curve(sql_setup.engine, curve_name, "base", config.report_date)
        )
    except Exception:
        pass
_base_disc = pd.concat(_base_disc_parts).sort_index() if _base_disc_parts else None

def _nii_consistent_rates(cf_detail: pd.DataFrame, beh: pd.DataFrame, disc: pd.Series, sid: str) -> pd.DataFrame:
    """Interest rate of every row from the SAME NII function as irrbb.nii_results (anchoring to
    the base-curve forward), so the cf.nii_* tables cancel at a 0 bp shock exactly like the SOT."""
    rows = nii_obj.compute_nii_shocked_schedule(
        beh, disc, DISC_CURVE_MAP, config.report_date, sid, HORIZON_YF, NII_FLOOR,
        caps_map=CAPS_MAP, floors_map=FLOORS_MAP, coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
        base_disc_df=_base_disc, return_rows=True,
    )
    cf_detail = cf_detail.copy()
    cf_detail.loc[rows.index, "eff_rate_shocked"] = rows["eff_rate_shocked"].to_numpy()
    cf_detail["nii_rate_consistent"] = True
    return cf_detail

_base_cf = _nii_consistent_rates(eve_obj.compute_base_cf_detail(all_beh, scenario_id="base"),
                                 all_beh, _base_disc, "base")
sql_setup.write_shocked_cf_products(
    _base_cf, config.report_date, "nii_base_scenario",
    shocked_disc_df=_base_disc, disc_curve_map=DISC_CURVE_MAP, table_type="nii",
    coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
    caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
)
print(f"  nii_base_scenario: {len(_base_cf):,} CF rows (base)")

# par_up / par_dn
for _sid in ("par_up", "par_dn"):
    _disc_parts = []
    for ccy, curve_name in DISC_CURVE_MAP.items():
        try:
            _disc_parts.append(
                esc.load_irrbb_disc_curve(sql_setup.engine, curve_name, _sid, config.report_date)
            )
        except Exception:
            pass
    if not _disc_parts:
        continue
    _disc = pd.concat(_disc_parts).sort_index()
    _beh_s = beh_scen.apply_variant(all_beh, scen_beh, _sid)
    _cf = _nii_consistent_rates(eve_obj.compute_shocked_cf_detail(
        _beh_s, shocked_disc_df=_disc, disc_curve_map=DISC_CURVE_MAP,
        scenario_id=_sid, eve_floor_rate=NII_FLOOR,
        caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
        coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
    ), _beh_s, _disc, _sid)
    sql_setup.write_shocked_cf_products(
        _cf, config.report_date, "nii_par_scenarios",
        shocked_disc_df=_disc, disc_curve_map=DISC_CURVE_MAP, table_type="nii",
        coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
        caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
    )
    print(f"  nii_par_scenarios: {len(_cf):,} CF rows (scenario: {_sid})")

# Own scenarios — curves already appended to irrbb.curves in step 1
for _sid in esc.OWN_SCENARIO_IDS:
    _disc_parts = []
    for ccy, curve_name in DISC_CURVE_MAP.items():
        try:
            _disc_parts.append(
                esc.load_irrbb_disc_curve(sql_setup.engine, curve_name, _sid, config.report_date)
            )
        except Exception:
            pass
    if not _disc_parts:
        continue
    _disc = pd.concat(_disc_parts).sort_index()
    _beh_s = beh_scen.apply_variant(all_beh, scen_beh, _sid)
    _cf = _nii_consistent_rates(eve_obj.compute_shocked_cf_detail(
        _beh_s, shocked_disc_df=_disc, disc_curve_map=DISC_CURVE_MAP,
        scenario_id=_sid, eve_floor_rate=NII_FLOOR,
        caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
        coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
    ), _beh_s, _disc, _sid)
    sql_setup.write_shocked_cf_products(
        _cf, config.report_date, "nii_own_scenarios",
        shocked_disc_df=_disc, disc_curve_map=DISC_CURVE_MAP, table_type="nii",
        coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
        caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
    )
    print(f"  nii_own_scenarios: {len(_cf):,} CF rows "
          f"(scenario: {_sid}  [{esc.OWN_SCENARIO_LABELS[_sid]}])")

# ── 5. CF-based delta_nii summary from irrbb.nii_results ──────────────────────
# delta_nii = nii_shocked - nii_base  (consistent with results.irrbb_report view)
nii_cf = sql_setup.load_nii_summary(config.report_date)
base_nii_map = (
    nii_cf[nii_cf["scenario_id"] == "base"]
    .set_index("currency")["nii_total"]
    .to_dict()
)

cf_summary_rows = []
for scenario_id in esc.ALL_SCENARIO_IDS:
    if scenario_id == "base":
        continue
    for _, row in nii_cf[nii_cf["scenario_id"] == scenario_id].iterrows():
        ccy       = row["currency"]
        delta     = float(row["nii_total"]) - base_nii_map.get(ccy, 0.0)
        delta_reg = delta if delta <= 0 else 0.5 * delta
        cf_summary_rows.append({
            "scenario_id":       scenario_id,
            "currency":          ccy,
            "delta_nii":         delta,
            "delta_nii_reg":     delta_reg,
            "sot_nii_pct":       delta / TIER1_CAPITAL * 100.0,
            "sot_nii_pct_reg":   delta_reg / TIER1_CAPITAL * 100.0,
        })
        print(f"  {scenario_id:8s}: delta_NII = {delta:+,.0f}")

shocked_summary = pd.DataFrame(cf_summary_rows)

# NOTE: irrbb.irrbb_report is written exclusively by eba_sot_workflow (final step).
# nii_results (schedule-level) is the intermediate output consumed by eba_sot_workflow.

# ── 6. Write to Excel ──────────────────────────────────────────────────────────
output_path = str(OUT_DIR / "nii_results.xlsx")
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    simple_detail.to_excel(writer,  sheet_name="NII_simple_detail",   index=False)
    simple_summary.to_excel(writer, sheet_name="NII_simple_summary",  index=False)

    base_detail.to_excel(writer,    sheet_name="NII_base_detail",     index=False)
    base_summary.to_excel(writer,   sheet_name="NII_base_summary",    index=False)

    if not shocked_detail.empty:
        shocked_detail.to_excel(writer,  sheet_name="NII_shocked_detail",  index=False)
    if not shocked_summary.empty:
        if "sot_nii_pct" not in shocked_summary.columns:
            shocked_summary["sot_nii_pct"] = None
        if "sot_nii_pct_reg" not in shocked_summary.columns:
            shocked_summary["sot_nii_pct_reg"] = None
        shocked_summary.to_excel(writer, sheet_name="NII_shocked_summary", index=False)

print(f"NII results written to {output_path}")
