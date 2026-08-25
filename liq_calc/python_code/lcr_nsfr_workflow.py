"""LCR / NSFR Workflow
=====================
Run order: balance_generate → balance_gen_add_data → cash_flow_calc → THIS FILE

Produces
--------
1. results.lcr_nsfr  (SQL) — one row per (report_date, currency):
     hqla, outflows_30d, inflows_30d, net_outflows_30d, lcr, asf, rsf, nsfr

2. output/lcr_nsfr_report.xlsx  — LCR and NSFR

Note: results.irrbb_report and output/irrbb_report.xlsx are now written by
eba_sot_workflow.py (irrbb_calc) — run that after nii_calc_workflow + eve_calc_workflow.
"""
import os
import pandas as pd

import sql_setup
import lcr_nsfr_objects as liq_obj

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

REPORT_DATE    = pd.to_datetime("2026-06-30")
TOTAL_ASSETS   = 10_000_000_000

BS_STRUCT_PATH = "../balance_generate/input_data/bank_data.xlsx"

# ── 1. Load balance-sheet structure ──────────────────────────────────────────
print(f"{REPORT_DATE.date()}  Loading bs_structure...")
bs_struct = pd.read_excel(BS_STRUCT_PATH, sheet_name="bs_structure")

# ── 2. Load 30-day asset inflows from cf.products ────────────────────────────
print("Loading 30-day asset inflows from cf.products...")
try:
    inflows_30d = sql_setup.load_30d_asset_inflows(REPORT_DATE)
    print(f"  30-day inflows by currency: {inflows_30d}")
except Exception as e:
    print(f"  Warning: could not load 30-day inflows ({e}). Using 0.")
    inflows_30d = {}

# ── 2b. Load deposit maturity split (within 30d → 100% run-off) ───────────────
print("Loading deposit maturity split from schemat.deposits...")
try:
    td_split = sql_setup.load_td_maturity_split(REPORT_DATE)
    print(f"  Deposit maturity split:\n{td_split.to_string(index=False)}")
except Exception as e:
    print(f"  Warning: could not load deposit maturity split ({e}). Using bs_struct weights only.")
    td_split = None

# ── 3. Compute LCR ───────────────────────────────────────────────────────────
print("Computing LCR...")
lcr_df = liq_obj.compute_lcr(bs_struct, TOTAL_ASSETS, inflows_30d, td_split)
for _, r in lcr_df.iterrows():
    print(
        f"  {r['currency']}  HQLA={r['hqla']:,.0f}  "
        f"Outflows={r['outflows_30d']:,.0f}  "
        f"Inflows={r['inflows_30d']:,.0f}  "
        f"Net={r['net_outflows_30d']:,.0f}  "
        + (f"LCR={r['lcr']:.2%}" if pd.notna(r['lcr']) else "LCR=N/A")
    )

# ── 4. Compute NSFR ──────────────────────────────────────────────────────────
print("Computing NSFR...")
nsfr_df = liq_obj.compute_nsfr(bs_struct, TOTAL_ASSETS)
for _, r in nsfr_df.iterrows():
    print(
        f"  {r['currency']}  ASF={r['asf']:,.0f}  RSF={r['rsf']:,.0f}  "
        + (f"NSFR={r['nsfr']:.2%}" if pd.notna(r['nsfr']) else "NSFR=N/A")
    )

# ── 5. Write LCR / NSFR to SQL ───────────────────────────────────────────────
print("Writing results.lcr_nsfr to SQL...")
liq_summary = lcr_df.merge(nsfr_df, on="currency", how="outer")
liq_summary.insert(0, "report_date", REPORT_DATE)

sql_setup.reset_lcr_nsfr()
sql_setup.write_lcr_nsfr(liq_summary)
print(f"  Written {len(liq_summary)} rows to results.lcr_nsfr.")

# ── 6. Write LCR / NSFR to Excel ─────────────────────────────────────────────
lcr_nsfr_path = "output/lcr_nsfr_report.xlsx"
print(f"Writing LCR/NSFR Excel to {lcr_nsfr_path}...")
with pd.ExcelWriter(lcr_nsfr_path, engine="openpyxl") as writer:
    liq_summary.to_excel(writer, sheet_name="LCR_NSFR", index=False)
print(f"  Written to {lcr_nsfr_path}")

print("Done.")
