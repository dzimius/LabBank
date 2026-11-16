"""Swap book + FX deals workflow
==============================
Reads inputs/<bank>/derivatives/derivatives.xlsx (labbank_common/derivatives.py):
  sheet 'swap'    -- the swap book (IRS priced; CIRS / OIS: layout ready, not priced yet)
  sheet 'fx_spot' -- FX spots / forwards / FX swaps, one leg per deal and currency
loads the legs to SQL, generates their cash-flow schedules (each currency on its own
curve), computes the repricing-gap contribution and writes all results to the database.

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
from labbank_common.bank_config import get_bank   # sql_setup puts the project root on sys.path
import irs_objects as irs_obj
from labbank_common import derivatives as deriv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

config.report_date = pd.Timestamp(get_bank().report_date)
print(datetime.datetime.now(), "Starting swap book / FX deals workflow...")

# ── 1. Load input Excel ────────────────────────────────────────────────────────
INPUT_FILE = deriv.input_path(get_bank())
df_raw = deriv.read_swaps(get_bank())
df_raw["report_date"] = config.report_date
df_swaps = df_raw.copy()
print(f"  Loaded {len(df_swaps)} swap(s) from {INPUT_FILE.name} [swap]")

# Asset swaps (sheet asset_swap): one pay-fixed swap per hedged bond, built from the bonds
# balance generation just produced; saved next to the outputs so the sandbox sees the same
# book (labbank_common.derivatives.read_swap_book).
_asw_rules = deriv.read_asset_swap_rules(get_bank())
_asw_path  = deriv.asset_swap_output_path(get_bank())
if not _asw_rules.empty:
    _bonds = pd.read_sql_query(
        sql_setup.text("SELECT * FROM schemat.financial_instruments WHERE report_date = :rd"),
        sql_setup.engine, params={"rd": config.report_date.date()})
    _asw = deriv.build_asset_swaps(_asw_rules, _bonds, config.report_date, get_bank().generation.seed)
    _asw_path.parent.mkdir(parents=True, exist_ok=True)
    _asw.to_excel(_asw_path, index=False)
    df_swaps = pd.concat([df_swaps, _asw[deriv.SWAP_COLUMNS]], ignore_index=True)
    for _pc, _g in _asw.groupby(_asw["swap_id"].str[3:7]):
        _tot = float(_bonds.loc[_bonds["product_code"].astype(str) == _pc, "balance_amt"].sum())
        print(f"  Asset swaps on {_pc}: {len(_g)} bond(s), {_g['notional'].sum()/1e6:,.0f}M of "
              f"{_tot/1e6:,.0f}M nominal ({_g['notional'].sum()/_tot:.0%}) -> {_asw_path.name}")
elif _asw_path.exists():
    _asw_path.unlink()                                   # rules removed: drop the stale book

fx_deals = deriv.read_fx_deals(get_bank())
fx_legs  = deriv.fx_legs(fx_deals, config.report_date) if not fx_deals.empty else pd.DataFrame()
print(f"  Loaded {len(fx_deals)} FX deal(s) [fx_spot] -> {len(fx_legs)} outstanding leg(s)")

# ── 2. Reset and load to SQL ───────────────────────────────────────────────────
sql_setup.reset_data(mode=0)

# Expands each swap to two leg rows, writes schemat.ir_swaps + sched.ir_swaps
legs_df = sql_setup.load_and_insert_ir_swaps(df_swaps, fx_legs, config.report_date)
sql_setup.write_fx_deals(fx_deals)
print(f"  Loaded {len(legs_df)} legs ({len(df_swaps)} swap(s), {len(fx_legs)} FX leg(s)) -> schemat/sched.ir_swaps")

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
    elif row.leg_type == "FX":
        part = irs_obj.gen_fx_leg_schedule(row, config.report_date, disc_df)
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
# swap book and FX deals kept apart: IRS -> gap_cf_irs / product 0000, FX -> gap_cf_fx / 0100
_fx_ids  = set(sched_df.loc[sched_df["leg_type"] == "FX", "schedule_id"].astype(str))
_is_fx   = cf_df["schedule_id"].astype(str).isin(_fx_ids)
swap_gap = irs_obj.compute_ir_swap_gap(cf_df[~_is_fx], sched_df, config.report_date)
sql_setup.write_df(swap_gap, "ir_swap_gap_beh", schema="irrbb")
print(f"  IRS gap written to irrbb.ir_swap_gap_beh ({len(swap_gap)} rows)")
fx_gap = irs_obj.compute_ir_swap_gap(cf_df[_is_fx], sched_df, config.report_date) if _is_fx.any() else pd.DataFrame()
with sql_setup.engine.begin() as _c:
    _c.exec_driver_sql("IF OBJECT_ID('irrbb.fx_swap_gap_beh', 'U') IS NOT NULL DROP TABLE irrbb.fx_swap_gap_beh;")
if not fx_gap.empty:
    sql_setup.write_df(fx_gap, "fx_swap_gap_beh", schema="irrbb")
    print(f"  FX deal gap written to irrbb.fx_swap_gap_beh ({len(fx_gap)} rows)")

# ── 7. Merge IRS gap into the main BS gap tables ──────────────────────────────
# Adds gap_cf_irs to irrbb.ir_gap_beh and appends product_code='0000' rows to
# irrbb.ir_gap_beh_a so NII can be computed from a single unified gap table.
# Guarded: irrbb.ir_gap_beh is written by cash_flow_calc; if that hasn't run
# yet (e.g. standalone balance-sheet job), the merge is deferred gracefully.
try:
    sql_setup.append_irs_to_ir_gap_tables(swap_gap, config.report_date)
    print("  IRS gap merged into irrbb.ir_gap_beh and irrbb.ir_gap_beh_a")
    if not fx_gap.empty:
        sql_setup.append_irs_to_ir_gap_tables(fx_gap, config.report_date,
                                              gap_col="gap_cf_fx", product_code="0100")
        print("  FX deal gap merged into irrbb.ir_gap_beh (gap_cf_fx) and ir_gap_beh_a (0100)")
except Exception as _gap_err:
    print(f"  Note: IRS gap merge deferred — irrbb gap tables not yet populated.")
    print(f"  Run cash_flow_calc first, then re-run swap_workflow to merge the gap.")
    print(f"  ({type(_gap_err).__name__}: {_gap_err})")

print(datetime.datetime.now(), "Swap book / FX deals workflow complete.")
