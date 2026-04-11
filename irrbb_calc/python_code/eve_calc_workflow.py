"""EVE Calculation Workflow
===========================
Run order: balance_generate → balance_gen_add_data → cash_flow_calc
           → ir_derivatives (optional) → THIS FILE

EVE (Economic Value of Equity) is a run-off measure: the balance sheet is
wound down without renewal.  Only existing cash flows are discounted — there
is no constant-balance / reinvestment assumption as in NII.

  EVE(base)    = Σ CF_i × d_f_base(t_i) × sign
  EVE(shocked) = Σ CF_i_shocked × d_f_shocked(t_i) × sign
  ΔEVE         = EVE(shocked) − EVE(base)

Produces
--------
1. irrbb.curves (SQL)         — base + 6 EBA SOT + own shocked discount curves
                                (same table as NII; overwritten if NII was run first)
2. irrbb.eve_results (SQL)    — schedule-level PV for base + all shocked scenarios
3. output/eve_results.xlsx    — EVE outputs:
     EVE_base_detail   / EVE_base_summary    — base EVE by (currency, source)
     EVE_shocked_detail / EVE_shocked_summary — ΔEVE under each EBA SOT scenario
"""
import os
import pandas as pd

import config
import sql_setup
import eve_calc_objects as eve_obj
import eba_shock_curves as esc

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/irrbb_calc"
os.chdir(BASE_DIR)

config.report_date = pd.to_datetime("2024-12-31")

CURRENCY      = "PLN"           # primary currency for EBA shock calibration
OWN_SHOCK_BPS = -100.0          # own scenario: -100 bps parallel
EVE_FLOOR     = 0.0             # 0 % floor on forward rates (EBA standard)

# Map currency → discount curve name
DISC_CURVE_MAP = {
    "PLN": "PLN_disc_curve",
    "EUR": "EUR_disc_curve",
    "USD": "USD_disc_curve"
}

# Per-product rate caps and floors (same source as NII)
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
COEFF_A_MAP: dict[str, float] = {
    str(int(r["product_code"])): float(r["a"])
    for _, r in _ir_df.iterrows()
    if "a" in _ir_df.columns and not pd.isna(r.get("a")) and float(r["a"]) != 1.0
}
COEFF_B_MAP: dict[str, float] = {
    str(int(r["product_code"])): float(r["b"]) / 100.0
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
    mkt_df, config.report_date, CURRENCY, OWN_SHOCK_BPS, EVE_FLOOR
)
esc.write_irrbb_curves(sql_setup.engine, shocked_df, config.report_date)
print(f"  Written {len(shocked_df):,} rows to irrbb.curves "
      f"({len(esc.ALL_SCENARIO_IDS)} scenarios × "
      f"{mkt_df.groupby('curve_name').ngroups} curves)")

# ── 2. Load run-off CF schedules (all maturities, no horizon filter) ──────────
print("Loading run-off cash flow schedules (all maturities)...")
beh_df = sql_setup.load_all_beh_schedules(config.report_date)

# Load swap schedules and append — empty if IRS workflow not run
try:
    swap_df = sql_setup.load_all_swap_beh_schedules(config.report_date)
except Exception:
    swap_df = pd.DataFrame()
all_beh = pd.concat([beh_df, swap_df], ignore_index=True) if not swap_df.empty else beh_df

print(f"  Loaded {len(all_beh):,} CF rows across "
      f"{all_beh['schedule_id'].nunique()} schedules")

# ── 3. Base EVE ───────────────────────────────────────────────────────────────
print("Computing base EVE...")
base_detail, _ = eve_obj.compute_eve_base(all_beh)

# ── 4. Reset EVE results table and write base scenario ────────────────────────
sql_setup.reset_eve_results()

base_sched = eve_obj.compute_eve_base_schedule(all_beh, scenario_id="base")
sql_setup.write_eve_results(base_sched, config.report_date)

# Derive base summary from the schedule rows written to SQL (single source of truth)
base_summary = (
    base_sched.groupby("currency")[["pv_capital", "pv_interest", "pv_total"]]
    .sum().reset_index()
)
print(f"  EVE base: {base_summary['pv_total'].sum():+,.0f}  "
      f"({len(base_sched)} schedule rows written to irrbb.eve_results)")

# ── 5. Shocked EVE for all 8 EBA SOT scenarios ────────────────────────────────
print("Computing shocked EVE for all scenarios...")
shocked_detail_parts  = []
shocked_summary_parts = []

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

    det, _ = eve_obj.compute_eve_shocked(
        all_beh,
        shocked_disc_df = shocked_disc,
        disc_curve_map  = DISC_CURVE_MAP,
        scenario_id     = scenario_id,
        eve_floor_rate  = EVE_FLOOR,
        caps_map        = CAPS_MAP,
        floors_map      = FLOORS_MAP,
        coeff_a_map     = COEFF_A_MAP,
        coeff_b_map     = COEFF_B_MAP,
    )
    shocked_detail_parts.append(det)

    shocked_sched = eve_obj.compute_eve_shocked_schedule(
        all_beh,
        shocked_disc_df = shocked_disc,
        disc_curve_map  = DISC_CURVE_MAP,
        scenario_id     = scenario_id,
        eve_floor_rate  = EVE_FLOOR,
        caps_map        = CAPS_MAP,
        floors_map      = FLOORS_MAP,
        coeff_a_map     = COEFF_A_MAP,
        coeff_b_map     = COEFF_B_MAP,
    )
    sql_setup.write_eve_results(shocked_sched, config.report_date)

    # Derive summary from schedule rows written to SQL (single source of truth)
    summ = (
        shocked_sched.groupby("currency")[["pv_capital", "pv_interest", "pv_total"]]
        .sum().reset_index()
        .assign(scenario_id=scenario_id)
    )
    shocked_summary_parts.append(summ)
    print(f"  {scenario_id:8s}: EVE = {shocked_sched['pv_total'].sum():+,.0f}")

shocked_detail  = pd.concat(shocked_detail_parts,  ignore_index=True) \
                  if shocked_detail_parts  else pd.DataFrame()
shocked_summary = pd.concat(shocked_summary_parts, ignore_index=True) \
                  if shocked_summary_parts else pd.DataFrame()

# ── 6. ΔEVE vs base + SOT ─────────────────────────────────────────────────────
if not shocked_summary.empty and not base_summary.empty:
    base_eve_by_ccy = base_summary.set_index("currency")["pv_total"]
    shocked_summary["delta_eve"] = (
        shocked_summary["pv_total"]
        - shocked_summary["currency"].map(base_eve_by_ccy).fillna(0.0)
    )
    # For detail: absolute shocked PV (delta can be computed vs base_detail)
    shocked_detail["delta_eve"] = shocked_detail["pv_total"]

    # SOT: delta_EVE / Tier1 (%). Breach if < -15%.
    shocked_summary["sot_eve_pct"] = shocked_summary["delta_eve"] / TIER1_CAPITAL * 100.0

    # Regulatory delta EVE: positive changes discounted at 50% (EBA/RTS/2022/10 Art.5(3))
    d = shocked_summary["delta_eve"]
    shocked_summary["delta_eve_reg"]   = d.where(d < 0, d * 0.5)
    shocked_summary["sot_eve_pct_reg"] = shocked_summary["delta_eve_reg"] / TIER1_CAPITAL * 100.0

# NOTE: irrbb.irrbb_report is written exclusively by eba_sot_workflow (final step).
# eve_results (schedule-level) is the intermediate output consumed by eba_sot_workflow.

# ── 6b. Shocked CF tables: cf.products_par_dn + cf.products_worst_eve ─────────
print("Writing shocked CF schedules to SQL...")

# Identify worst EVE scenario (largest negative delta_eve across all currencies)
worst_eve_sid = "par_up"   # sensible fallback
if not shocked_summary.empty and "delta_eve" in shocked_summary.columns:
    worst_row     = shocked_summary.loc[shocked_summary["delta_eve"].idxmin()]
    worst_eve_sid = str(worst_row["scenario_id"])
    print(f"  Worst EVE scenario: {worst_eve_sid} "
          f"(delta_eve = {worst_row['delta_eve']:+,.0f})")

sql_setup.reset_shocked_cf_products("products_par_dn")
sql_setup.reset_shocked_cf_products("products_worst_eve")

# Write base scenario first so both tables contain base + shocked rows for direct comparison
_base_cf = eve_obj.compute_base_cf_detail(all_beh, scenario_id="base")
_base_disc_parts = []
for ccy, curve_name in DISC_CURVE_MAP.items():
    try:
        _base_disc_parts.append(
            esc.load_irrbb_disc_curve(sql_setup.engine, curve_name, "base", config.report_date)
        )
    except Exception:
        pass
_base_disc = pd.concat(_base_disc_parts).sort_index() if _base_disc_parts else None
for _table_name in ("products_par_dn", "products_worst_eve"):
    sql_setup.write_shocked_cf_products(
        _base_cf, config.report_date, _table_name,
        shocked_disc_df=_base_disc,
        disc_curve_map=DISC_CURVE_MAP,
    )
print(f"  base     : {len(_base_cf):,} base CF rows written to both tables")

_scenarios_to_write = [("par_dn", "products_par_dn"),
                       (worst_eve_sid, "products_worst_eve")]

for _scenario_id, _table_name in _scenarios_to_write:
    _disc_parts = []
    for ccy, curve_name in DISC_CURVE_MAP.items():
        try:
            _disc_parts.append(
                esc.load_irrbb_disc_curve(
                    sql_setup.engine, curve_name, _scenario_id, config.report_date
                )
            )
        except Exception:
            pass
    if not _disc_parts:
        print(f"  Skipping {_table_name}: no disc curves for '{_scenario_id}'.")
        continue

    _shocked_disc = pd.concat(_disc_parts).sort_index()
    _cf_detail = eve_obj.compute_shocked_cf_detail(
        all_beh,
        shocked_disc_df = _shocked_disc,
        disc_curve_map  = DISC_CURVE_MAP,
        scenario_id     = _scenario_id,
        eve_floor_rate  = EVE_FLOOR,
        caps_map        = CAPS_MAP,
        floors_map      = FLOORS_MAP,
        coeff_a_map     = COEFF_A_MAP,
        coeff_b_map     = COEFF_B_MAP,
    )
    sql_setup.write_shocked_cf_products(
        _cf_detail, config.report_date, _table_name,
        shocked_disc_df=_shocked_disc,
        disc_curve_map=DISC_CURVE_MAP,
    )
    print(f"  {_table_name}: {len(_cf_detail):,} CF rows written "
          f"(scenario: {_scenario_id})")

# ── 7. Write to Excel ──────────────────────────────────────────────────────────
output_path = "output/eve_results.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    base_detail.to_excel(writer,  sheet_name="EVE_base_detail",    index=False)
    base_summary.to_excel(writer, sheet_name="EVE_base_summary",   index=False)

    if not shocked_detail.empty:
        shocked_detail.to_excel(writer,  sheet_name="EVE_shocked_detail",  index=False)
    if not shocked_summary.empty:
        # Ensure SOT column is present even if block above was skipped
        if "sot_eve_pct" not in shocked_summary.columns:
            shocked_summary["sot_eve_pct"] = None
        if "sot_eve_pct_reg" not in shocked_summary.columns:
            shocked_summary["sot_eve_pct_reg"] = None
        shocked_summary.to_excel(writer, sheet_name="EVE_shocked_summary", index=False)

print(f"EVE results written to {output_path}")
