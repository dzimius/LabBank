"""IRS Workflow
=============
Reads ir_derivatives/input/irs_input.xlsx, loads transactions to SQL,
generates CF schedules for fixed and floating legs, computes the IRS
repricing gap contribution and writes all results to the database.

Run order (after cf_calc_workflow.py has populated mkt.curves, mkt.fixings):
  1. balance_generate
  2. balance_gen_add_data
  3. cash_flow_calc
  4. THIS FILE  ← ir_derivatives
  5. irrbb_calc
"""
import datetime
import os

import pandas as pd

import config
import sql_setup
import irs_objects as irs_obj

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

config.report_date = pd.to_datetime("2026-06-30")
print(datetime.datetime.now(), "Starting IRS workflow...")

# ── 1. Load input Excel ────────────────────────────────────────────────────────
INPUT_FILE = "input/irs_input.xlsx"
df_raw = pd.read_excel(INPUT_FILE)


# Derived / default columns
if "float_b_day_conv" not in df_raw.columns:
    df_raw["float_b_day_conv"] = df_raw["fixed_b_day_conv"]
if "float_spread" not in df_raw.columns:
    df_raw["float_spread"] = 0.0
if "pay_fixed" not in df_raw.columns:
    df_raw["pay_fixed"] = 1     # default: bank pays fixed
if "swap_type" not in df_raw.columns:
    df_raw["swap_type"] = "IRS"

df_raw["start_date"]    = pd.to_datetime(df_raw["start_date"])
df_raw["maturity_date"] = pd.to_datetime(df_raw["maturity_date"])
df_raw["report_date"]   = config.report_date

# Filter to current report date
df_swaps = df_raw[df_raw["report_date"] == config.report_date].copy()
print(f"  Loaded {len(df_swaps)} IRS transaction(s) from {INPUT_FILE}")

# ── 2. Reset and load to SQL ───────────────────────────────────────────────────
sql_setup.reset_data(mode=0)

# Expands each swap to two leg rows, writes schemat.ir_swaps + sched.ir_swaps
legs_df = sql_setup.load_and_insert_ir_swaps(df_swaps)
print(f"  Loaded {len(legs_df)} legs ({len(df_swaps)} swap(s)) -> schemat/sched.ir_swaps")

# ── 3. Load market data ────────────────────────────────────────────────────────
sched_df = legs_df   # alias: sched_df used in CF generation and gap computation
disc_curves_needed = sched_df["disc_curve"].dropna().unique()
fwd_curves_needed  = sched_df.loc[sched_df["leg_type"] == "FLOAT",
                                   ["fwd_curve", "fixing_freq", "currency"]].drop_duplicates()

disc_series_list = []
for cn in disc_curves_needed:
    disc_series_list.append(sql_setup.load_disc_curve(cn, config.report_date))
disc_df = pd.concat(disc_series_list).sort_index() if disc_series_list else pd.Series(dtype=float)

fwd_series_list = []
for _, r in fwd_curves_needed.iterrows():
    fwd_series_list.append(
        sql_setup.load_fwd_curve(r["fwd_curve"], r["fixing_freq"],
                                 r["currency"], config.report_date)
    )
fwd_df = pd.concat(fwd_series_list).sort_index() if fwd_series_list else pd.Series(dtype=float)

fix_df = sql_setup.load_fixings()
print("  Market data loaded.")

# ── 4. Generate CF schedules ───────────────────────────────────────────────────
cf_parts = []
for row in sched_df.itertuples(index=False, name="Row"):
    if row.leg_type == "FIXED":
        part = irs_obj.gen_fixed_leg_schedule(row, config.report_date, disc_df)
    else:
        part = irs_obj.gen_float_leg_schedule(row, config.report_date, disc_df, fwd_df, fix_df)
    if part is not None and len(part) > 0:
        cf_parts.append(part)

if not cf_parts:
    raise RuntimeError("No CF rows generated — check input data and market curves.")

cf_df = pd.concat(cf_parts, ignore_index=True)
print(f"  Generated {len(cf_df)} CF rows across {sched_df['leg_type'].value_counts().to_dict()} legs")

# ── 5. Write CF to SQL (orig == beh for swaps) ────────────────────────────────
sql_setup.write_df(cf_df, "ir_swap_orig", schema="cf")
sql_setup.write_df(cf_df, "ir_swap_beh",  schema="cf")
print("  CF schedules written to cf.ir_swap_orig and cf.ir_swap_beh")

# ── 6. Compute and store IRS repricing gap ────────────────────────────────────
swap_gap = irs_obj.compute_ir_swap_gap(cf_df, sched_df, config.report_date)
sql_setup.write_df(swap_gap, "ir_swap_gap_beh", schema="irrbb")
print(f"  IRS gap written to irrbb.ir_swap_gap_beh ({len(swap_gap)} rows)")

# ── 7. Merge IRS gap into the main BS gap tables ──────────────────────────────
# Adds gap_cf_irs to irrbb.ir_gap_beh and appends product_code='0000' rows to
# irrbb.ir_gap_beh_a so NII can be computed from a single unified gap table.
# Guarded: irrbb.ir_gap_beh is written by cash_flow_calc; if that hasn't run
# yet (e.g. standalone balance-sheet job), the merge is deferred gracefully.
try:
    sql_setup.append_irs_to_ir_gap_tables(swap_gap, config.report_date)
    print("  IRS gap merged into irrbb.ir_gap_beh and irrbb.ir_gap_beh_a")
except Exception as _gap_err:
    print(f"  Note: IRS gap merge deferred — irrbb gap tables not yet populated.")
    print(f"  Run cash_flow_calc first, then re-run irs_workflow to merge the gap.")
    print(f"  ({type(_gap_err).__name__}: {_gap_err})")

print(datetime.datetime.now(), "IRS workflow complete.")
