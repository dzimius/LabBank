"""NII Calculation Workflow
===========================
Run order: balance_generate → balance_gen_add_data → cash_flow_calc
           → ir_derivatives → THIS FILE

Produces
--------
1. irrbb.curves (SQL)         — base + 6 EBA SOT + own shocked discount curves
2. output/irrbb_shock_curves.xlsx — day-by-day shocked curve shapes (audit/review)
3. output/nii_results.xlsx    — all NII outputs:
     NII_simple_detail / NII_simple_summary  — gap-based simple NII (±100 bps)
     NII_base_detail   / NII_base_summary    — CF-based base NII (absolute income)
     NII_shocked_detail / NII_shocked_summary— gap-based ΔNII per EBA SOT scenario

Shocked NII uses the same gap-based method as eba_sot_workflow (compute_nii_sot):
    delta_nii = gap × Δr × remain_yf   (per tenor bucket)
This avoids the fragility of reconstructing absolute NII from shocked discount curves
and reconciles with the EBA SOT results by construction.
"""
import os
import pandas as pd

import config
import sql_setup
import nii_calc_objects as nii_obj
import eve_calc_objects as eve_obj
import eba_shock_curves as esc
import eba_sot_objects as sot

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

config.report_date = pd.to_datetime("2026-06-30")

HORIZON_YF    = 1.0
CURRENCY      = "PLN"           # primary currency for EBA shock calibration
OWN_SHOCK_BPS = -100.0          # own scenario: -100 bps parallel
NII_FLOOR     = 0.0             # 0 % floor for all products (EBA NII)

horizon_end = config.report_date + pd.Timedelta(days=round(HORIZON_YF * 365.25))

DISC_CURVE_MAP = {
    "PLN": "PLN_disc_curve",
    "EUR": "EUR_disc_curve",
    "USD": "USD_disc_curve"
}

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
    mkt_df, config.report_date, CURRENCY, OWN_SHOCK_BPS, NII_FLOOR
)
esc.write_irrbb_curves(sql_setup.engine, shocked_df, config.report_date)
print(f"  Written {len(shocked_df):,} rows to irrbb.curves "
      f"({len(esc.ALL_SCENARIO_IDS)} scenarios × "
      f"{mkt_df.groupby('curve_name').ngroups} curves)")

# Append own scenarios (±100 bps, ±1 bps) to irrbb.curves
_own_curves_df = esc.build_own_shocked_curves(mkt_df, config.report_date, nii_floor_rate=NII_FLOOR)
esc.write_irrbb_curves(sql_setup.engine, _own_curves_df, config.report_date, replace=False)
print(f"  Appended {len(_own_curves_df):,} own-scenario rows to irrbb.curves "
      f"({esc.OWN_SCENARIO_IDS})")

esc.generate_shock_excel(
    mkt_df, config.report_date, CURRENCY,
    output_path="output/irrbb_shock_curves.xlsx",
    own_shock_bps=OWN_SHOCK_BPS,
    nii_floor_rate=NII_FLOOR,
    horizon_days=round(HORIZON_YF * 365),
)

# ── 2. Simple NII (repricing gap based, ±100 bps shocks) ──────────────────────
print("Computing simple (gap-based) NII...")
ir_gap_beh = sql_setup.load_ir_gap_beh()
simple_detail, simple_summary = nii_obj.compute_nii(
    ir_gap_beh, horizon_yf=HORIZON_YF,
)

# ── 3. Base NII (CF-based, no shock) — absolute income level ──────────────────
print("Computing base NII (CF-based)...")
beh_df = sql_setup.load_beh_schedules(config.report_date, horizon_end)

try:
    swap_df = sql_setup.load_swap_beh_schedules(config.report_date, horizon_end)
except Exception:
    swap_df = pd.DataFrame()
all_beh = pd.concat([beh_df, swap_df], ignore_index=True) if not swap_df.empty else beh_df

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
)
print(f"  Base NII: {base_summary['nii_total'].sum():+,.0f}")

# Write base schedule-level NII to SQL
sql_setup.reset_nii_results()
base_sched = nii_obj.compute_nii_base_schedule(
    all_beh, config.report_date, HORIZON_YF,
    caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
    coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
    base_disc_df=_base_disc_step3, disc_curve_map=DISC_CURVE_MAP,
)
sql_setup.write_nii_results(base_sched, config.report_date)
print(f"  {'base':8s}: NII = {base_sched['nii_total'].sum():+,.0f}  "
      f"({len(base_sched)} schedule rows written to irrbb.nii_results)")

# ── 4. Shocked NII schedules for all EBA scenarios (CF-based, absolute) ───────
print("Computing shocked NII schedules for all scenarios (CF-based)...")
shocked_detail_parts = []

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

    # Gap-based detail kept for bucket breakdown in Excel
    det, _ = sot.compute_nii_sot(
        ir_gap_beh, mkt_df, shocked_disc, DISC_CURVE_MAP,
        config.report_date, scenario_id, HORIZON_YF,
    )
    shocked_detail_parts.append(det)

    # CF-based absolute NII for this scenario → irrbb.nii_results
    shocked_sched = nii_obj.compute_nii_shocked_schedule(
        all_beh, shocked_disc, DISC_CURVE_MAP,
        config.report_date, scenario_id, HORIZON_YF, NII_FLOOR,
        caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
        coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
    )
    sql_setup.write_nii_results(shocked_sched, config.report_date)

shocked_detail = pd.concat(shocked_detail_parts, ignore_index=True) \
                 if shocked_detail_parts else pd.DataFrame()

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
_base_cf = eve_obj.compute_base_cf_detail(all_beh, scenario_id="base")
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
    _cf = eve_obj.compute_shocked_cf_detail(
        all_beh, shocked_disc_df=_disc, disc_curve_map=DISC_CURVE_MAP,
        scenario_id=_sid, eve_floor_rate=NII_FLOOR,
        caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
        coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
    )
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
    _cf = eve_obj.compute_shocked_cf_detail(
        all_beh, shocked_disc_df=_disc, disc_curve_map=DISC_CURVE_MAP,
        scenario_id=_sid, eve_floor_rate=NII_FLOOR,
        caps_map=CAPS_MAP, floors_map=FLOORS_MAP,
        coeff_a_map=COEFF_A_MAP, coeff_b_map=COEFF_B_MAP,
    )
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
output_path = "output/nii_results.xlsx"
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
