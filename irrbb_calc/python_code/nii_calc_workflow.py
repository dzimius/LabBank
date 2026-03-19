import os
import pandas as pd

import config
import sql_setup
import nii_calc_objects as nii_obj

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/irrbb_calc"
os.chdir(BASE_DIR)

config.report_date = pd.to_datetime("2024-12-31")
HORIZON_YF = 1.0
horizon_end = config.report_date + pd.Timedelta(days=round(HORIZON_YF * 365.25))

# ── Simple NII (repricing gap based) ─────────────────────────────────────────
ir_gap_beh = sql_setup.load_ir_gap_beh()
# Current accounts have 0% rate — floor prevents -100bps shock from applying
ca_gap = sql_setup.load_ir_gap_ca()
simple_detail, simple_summary = nii_obj.compute_nii(ir_gap_beh, horizon_yf=HORIZON_YF, ca_gap=ca_gap)

# ── EBA Constant Balance NII (behavioral CF schedules) ───────────────────────
beh_df = sql_setup.load_beh_schedules(config.report_date, horizon_end)
base_detail, base_summary = nii_obj.compute_nii_base(beh_df, config.report_date, horizon_yf=HORIZON_YF)

# ── Write to Excel ────────────────────────────────────────────────────────────
output_path = "output/nii_results.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    # Simple NII (gap-based, with shocks)
    simple_detail.to_excel(writer, sheet_name="NII_simple_detail",  index=False)
    simple_summary.to_excel(writer, sheet_name="NII_simple_summary", index=False)
    # EBA NII base (CF-based, no shock)
    base_detail.to_excel(writer, sheet_name="NII_base_detail",   index=False)
    base_summary.to_excel(writer, sheet_name="NII_base_summary",  index=False)

print(f"NII results written to {output_path}")
