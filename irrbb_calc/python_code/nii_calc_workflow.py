"""NII Calculation Workflow
===========================
Run order: balance_generate → balance_gen_add_data → cash_flow_calc
           → ir_derivatives → THIS FILE

Produces
--------
1. irrbb.curves (SQL)         — base + 6 EBA SOT + own shocked discount curves
2. output/irrbb_shock_curves.xlsx — day-by-day shocked curve shapes (audit/review)
3. output/nii_results.xlsx    — all NII outputs:
     NII_simple_detail / NII_simple_summary  — gap-based simple NII
     NII_base_detail   / NII_base_summary    — EBA base NII (CF-based, no shock)
     NII_shocked_detail / NII_shocked_summary— ΔNii under each EBA SOT scenario
"""
import os
import pandas as pd

import config
import sql_setup
import nii_calc_objects as nii_obj
import eba_shock_curves as esc

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/irrbb_calc"
os.chdir(BASE_DIR)

config.report_date = pd.to_datetime("2024-12-31")

HORIZON_YF    = 1.0
CURRENCY      = "PLN"           # primary currency for EBA shock calibration
OWN_SHOCK_BPS = -100.0          # own scenario: -100 bps parallel
NII_FLOOR     = 0.0             # 0 % floor for all products (EBA NII)

horizon_end = config.report_date + pd.Timedelta(days=round(HORIZON_YF * 365.25))

# Map currency → discount curve name (used for shocked fwd_rt derivation).
# Add entries here for each currency in your portfolio.
DISC_CURVE_MAP = {
    "PLN": "PLN_disc_curve",    # adjust to match your mkt.curves curve_name
    "EUR": "EUR_disc_curve",
    "USD": "USD_disc_curve"
}

# Per-product rate caps and floors for NII calculation.
# Loaded from the same interest_rt.xlsx used by the CF calc workflow.
# Values must be in decimal (same unit as fwd_rt: 0.001 = 0.1%, 0.01 = 1%).
_ir_df = pd.read_excel("../balance_generate/input_data/interest_rt.xlsx")
CAPS_MAP: dict[str, float] = {
    str(int(r["product_code"])): float(r["client_cap"])
    for _, r in _ir_df.iterrows()
    if "client_cap" in _ir_df.columns and not pd.isna(r.get("client_cap"))
}
FLOORS_MAP: dict[str, float] = {
    str(int(r["product_code"])): float(r["client_floor"])
    for _, r in _ir_df.iterrows()
    if "client_floor" in _ir_df.columns and not pd.isna(r.get("client_floor"))
}
# Linear transform coefficients: client_rt = a * index_rt + b/100
# b column in interest_rt.xlsx is in percentage points (e.g. 10.0 = 10% = 0.10 decimal)
COEFF_A_MAP: dict[str, float] = {
    str(int(r["product_code"])): float(r["a"])
    for _, r in _ir_df.iterrows()
    if "a" in _ir_df.columns and not pd.isna(r.get("a")) and float(r["a"]) != 1.0
}
COEFF_B_MAP: dict[str, float] = {
    str(int(r["product_code"])): float(r["b"]) / 100.0   # % points → decimal
    for _, r in _ir_df.iterrows()
    if "b" in _ir_df.columns and not pd.isna(r.get("b")) and float(r["b"]) != 0.0
}

# ── 0. Load Tier 1 capital from schemat.equity ────────────────────────────────
TIER1_CAPITAL = sql_setup.load_tier1_capital(config.report_date)
print(f"Tier 1 capital loaded from schemat.equity: {TIER1_CAPITAL:,.0f}")

# ── 1. Build and store shocked yield curves ───────────────────────────────────
print(config.report_date, "Building EBA SOT shocked curves...")
mkt_df = sql_setup.load_mkt_curves(config.report_date)
if mkt_df.empty:
    raise RuntimeError("No market curves found in mkt.curves for report_date. "
                       "Run cash_flow_calc workflow first.")

shocked_df = esc.build_all_shocked_curves(
    mkt_df, config.report_date, CURRENCY, OWN_SHOCK_BPS, NII_FLOOR
)
esc.write_irrbb_curves(sql_setup.engine, shocked_df, config.report_date)
print(f"  Written {len(shocked_df):,} rows to irrbb.curves "
      f"({len(esc.ALL_SCENARIO_IDS)} scenarios × "
      f"{mkt_df.groupby('curve_name').ngroups} curves)")

esc.generate_shock_excel(
    mkt_df, config.report_date, CURRENCY,
    output_path="output/irrbb_shock_curves.xlsx",
    own_shock_bps=OWN_SHOCK_BPS,
    nii_floor_rate=NII_FLOOR,
    horizon_days=round(HORIZON_YF * 365),
)

# ── 2. Simple NII (repricing gap based, with ±100 bps shocks) ─────────────────
print("Computing simple (gap-based) NII...")
ir_gap_beh = sql_setup.load_ir_gap_beh()
simple_detail, simple_summary = nii_obj.compute_nii(
    ir_gap_beh, horizon_yf=HORIZON_YF,
)

# ── 3. Base NII (EBA CF-based, no shock) ──────────────────────────────────────
print("Computing base NII (CF-based)...")
beh_df = sql_setup.load_beh_schedules(config.report_date, horizon_end)

# Load swap schedules (product_type='S') and append — empty if IRS workflow not run
try:
    swap_df = sql_setup.load_swap_beh_schedules(config.report_date, horizon_end)
except Exception:
    swap_df = pd.DataFrame()
all_beh = pd.concat([beh_df, swap_df], ignore_index=True) if not swap_df.empty else beh_df

base_detail, base_summary = nii_obj.compute_nii_base(
    beh_df, config.report_date, horizon_yf=HORIZON_YF
)

# ── 4. Shocked NII for all 8 scenarios ────────────────────────────────────────
print("Computing shocked NII for all scenarios...")
shocked_detail_parts  = []
shocked_summary_parts = []

# Reset NII results table once before writing
sql_setup.reset_nii_results()

# Write base schedule-level NII
base_sched = nii_obj.compute_nii_base_schedule(
    all_beh, config.report_date, HORIZON_YF,
    caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
    coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
)
sql_setup.write_nii_results(base_sched, config.report_date)
base_nii_total = base_sched["nii_total"].sum()
print(f"  {'base':8s}: NII = {base_nii_total:+,.0f}  ({len(base_sched)} schedule rows written to irrbb.nii_results)")

for scenario_id in esc.ALL_SCENARIO_IDS:
    if scenario_id == "base":
        continue  # base already written by compute_nii_base_schedule above
    # Load shocked disc curve (combined across all curves needed)
    disc_series_parts = []
    for ccy, curve_name in DISC_CURVE_MAP.items():
        try:
            s = esc.load_irrbb_disc_curve(
                sql_setup.engine, curve_name, scenario_id, config.report_date
            )
            disc_series_parts.append(s)
        except Exception:
            pass   # curve not present for this scenario/currency — skip

    if not disc_series_parts:
        print(f"  Skipping scenario '{scenario_id}': no disc curves found.")
        continue

    shocked_disc = pd.concat(disc_series_parts).sort_index()

    det, summ = nii_obj.compute_nii_shocked(
        beh_df,
        shocked_disc_df = shocked_disc,
        disc_curve_map  = DISC_CURVE_MAP,
        report_date     = config.report_date,
        scenario_id     = scenario_id,
        horizon_yf      = HORIZON_YF,
        nii_floor_rate  = NII_FLOOR,
        caps_map        = CAPS_MAP,
        floors_map      = FLOORS_MAP,
    )
    shocked_detail_parts.append(det)
    shocked_summary_parts.append(summ)

    # Write shocked schedule-level NII
    shocked_sched = nii_obj.compute_nii_shocked_schedule(
        all_beh,
        shocked_disc_df = shocked_disc,
        disc_curve_map  = DISC_CURVE_MAP,
        report_date     = config.report_date,
        scenario_id     = scenario_id,
        horizon_yf      = HORIZON_YF,
        nii_floor_rate  = NII_FLOOR,
        caps_map        = CAPS_MAP,
        floors_map      = FLOORS_MAP,
        coeff_a_map     = COEFF_A_MAP,
        coeff_b_map     = COEFF_B_MAP,
    )
    sql_setup.write_nii_results(shocked_sched, config.report_date)
    print(f"  {scenario_id:8s}: NII = {shocked_sched['nii_total'].sum():+,.0f}")

shocked_detail  = pd.concat(shocked_detail_parts,  ignore_index=True) \
                  if shocked_detail_parts  else pd.DataFrame()
shocked_summary = pd.concat(shocked_summary_parts, ignore_index=True) \
                  if shocked_summary_parts else pd.DataFrame()

# ── 5. ΔNII vs base + SOT ─────────────────────────────────────────────────────
if not shocked_summary.empty and not base_summary.empty:
    base_nii_by_ccy = base_summary.set_index("currency")["nii_total"]
    shocked_summary["delta_nii"] = (
        shocked_summary["nii_total"]
        - shocked_summary["currency"].map(base_nii_by_ccy).fillna(0.0)
    )
    shocked_detail["delta_nii"] = shocked_detail["nii_total"]  # absolute for detail

    # SOT: delta_NII / Tier1 (%). Breach if < -5%.
    shocked_summary["sot_nii_pct"] = shocked_summary["delta_nii"] / TIER1_CAPITAL * 100.0

    # Regulatory delta NII: positive changes discounted at 50% (EBA/RTS/2022/10 Art.6(3))
    d = shocked_summary["delta_nii"]
    shocked_summary["delta_nii_reg"]   = d.where(d < 0, d * 0.5)
    shocked_summary["sot_nii_pct_reg"] = shocked_summary["delta_nii_reg"] / TIER1_CAPITAL * 100.0

# ── 5b. Write SOT summary to irrbb.irrbb_report ───────────────────────────────
if not shocked_summary.empty:
    sql_setup.reset_irrbb_report()
    sql_setup.upsert_irrbb_report(shocked_summary, config.report_date, mode="nii")

# ── 6. Write to Excel ──────────────────────────────────────────────────────────
output_path = "output/nii_results.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    # Simple NII (gap-based)
    simple_detail.to_excel(writer,  sheet_name="NII_simple_detail",   index=False)
    simple_summary.to_excel(writer, sheet_name="NII_simple_summary",  index=False)

    # EBA base NII (CF-based, no shock)
    base_detail.to_excel(writer,    sheet_name="NII_base_detail",     index=False)
    base_summary.to_excel(writer,   sheet_name="NII_base_summary",    index=False)

    # Shocked NII (all scenarios stacked, scenario_id column distinguishes them)
    if not shocked_detail.empty:
        shocked_detail.to_excel(writer,  sheet_name="NII_shocked_detail",  index=False)
    if not shocked_summary.empty:
        # Ensure SOT column is present even if block above was skipped
        if "sot_nii_pct" not in shocked_summary.columns:
            shocked_summary["sot_nii_pct"] = None
        if "sot_nii_pct_reg" not in shocked_summary.columns:
            shocked_summary["sot_nii_pct_reg"] = None
        shocked_summary.to_excel(writer, sheet_name="NII_shocked_summary", index=False)

print(f"NII results written to {output_path}")
