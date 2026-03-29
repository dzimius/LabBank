"""LCR / NSFR + IRRBB Report Workflow
=====================================
Run order: balance_generate → balance_gen_add_data → cash_flow_calc
           → irrbb_calc (nii_calc_workflow + eve_calc_workflow) → THIS FILE

Produces
--------
1. results.lcr_nsfr  (SQL) — one row per (report_date, currency):
     hqla, outflows_30d, inflows_30d, net_outflows_30d, lcr, asf, rsf, nsfr

2. results.irrbb_report (SQL) — one row per (report_date, currency, scenario_id):
     nii_base, nii_shocked, delta_nii, eve_base, eve_shocked, delta_eve
   + extra rows: scenario_id='worst_nii' and 'worst_eve' per currency

3. output/lcr_nsfr_report.xlsx  — LCR and NSFR only
4. output/irrbb_report.xlsx     — all EBA scenarios + worst_nii + worst_eve rows
"""
import os
import pandas as pd

import sql_setup
import lcr_nsfr_objects as liq_obj

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/liq_calc"
os.chdir(BASE_DIR)

REPORT_DATE    = pd.to_datetime("2024-12-31")
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

# ── 7. Load NII and EVE results from irrbb schema ────────────────────────────
print("Loading NII results from irrbb.nii_results...")
try:
    nii_df = sql_setup.load_nii_by_scenario(REPORT_DATE)
    print(f"  Loaded {len(nii_df)} (currency, scenario) NII rows.")
except Exception as e:
    raise RuntimeError(
        f"Failed to load irrbb.nii_results: {e}\n"
        "Run nii_calc_workflow.py before this script."
    ) from e

print("Loading EVE results from irrbb.eve_results...")
try:
    eve_df = sql_setup.load_eve_by_scenario(REPORT_DATE)
    print(f"  Loaded {len(eve_df)} (currency, scenario) EVE rows.")
except Exception as e:
    raise RuntimeError(
        f"Failed to load irrbb.eve_results: {e}\n"
        "Run eve_calc_workflow.py before this script."
    ) from e

# ── 8. Build IRRBB report (NII + EVE only, with worst-scenario rows) ──────────
print("Building IRRBB report...")
tier1_capital = sql_setup.load_tier1_capital(REPORT_DATE)
print(f"  Tier 1 capital loaded from schemat.equity: {tier1_capital:,.0f}")

irrbb_df = liq_obj.build_irrbb_report(
    report_date     = REPORT_DATE,
    nii_by_scenario = nii_df,
    eve_by_scenario = eve_df,
    tier1_capital   = tier1_capital,
)
print(f"  Report rows: {len(irrbb_df)}  "
      f"(currencies: {irrbb_df['currency'].nunique()}, "
      f"scenarios: {irrbb_df['scenario_id'].nunique()})")

# ── 9. Write IRRBB report to SQL ──────────────────────────────────────────────
print("Writing results.irrbb_report to SQL...")
sql_setup.reset_irrbb_report()
sql_setup.write_irrbb_report(irrbb_df)
print(f"  Written {len(irrbb_df)} rows to results.irrbb_report.")

# ── 10. Write IRRBB report to Excel ───────────────────────────────────────────
irrbb_path = "output/irrbb_report.xlsx"
print(f"Writing IRRBB Excel to {irrbb_path}...")

# Separate regular scenarios from worst rows for clarity
regular_rows = irrbb_df[~irrbb_df["scenario_id"].isin(["worst_nii", "worst_eve"])]
worst_rows   = irrbb_df[irrbb_df["scenario_id"].isin(["worst_nii", "worst_eve"])]

delta_nii = (
    irrbb_df[["currency", "scenario_id", "nii_base", "nii_shocked", "delta_nii"]]
    .sort_values(["currency", "scenario_id"])
    .reset_index(drop=True)
)
delta_eve = (
    irrbb_df[["currency", "scenario_id", "eve_base", "eve_shocked", "delta_eve"]]
    .sort_values(["currency", "scenario_id"])
    .reset_index(drop=True)
)

with pd.ExcelWriter(irrbb_path, engine="openpyxl") as writer:
    irrbb_df.to_excel(   writer, sheet_name="IRRBB_all",   index=False)
    worst_rows.to_excel( writer, sheet_name="Worst_scenarios", index=False)
    delta_nii.to_excel(  writer, sheet_name="Delta_NII",    index=False)
    delta_eve.to_excel(  writer, sheet_name="Delta_EVE",    index=False)

print(f"  Written to {irrbb_path}")
print("Done.")
