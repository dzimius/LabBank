import pandas as pd
import os
import sql_setup
import config
import datetime as dt
import QuantLib as ql
import cf_calc_objects as cf_obj
import datetime

print(datetime.datetime.now())
BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/cash_flow_calc"
os.chdir(BASE_DIR)


config.report_date = pd.to_datetime('2024-12-31')

mode = 0
## musi byc zaimplementowany mode aby przy jedynce dorzucac tylko harmonogramy ktore nie sa w juz isntiejacych tabelach

sql_setup.reset_data(mode)


#####################
####### crerate forward interpolated curves
#################################

uniq_curves = cf_obj.get_unique_curves()

disc_curves = pd.DataFrame(columns=["curve_name", "n_days", "node_date", "d_f"])
fwd_curves = pd.DataFrame(columns=["curve_name", "n_days", "node_date", "fwd_rt"])

for i, row in uniq_curves.loc[uniq_curves['curve_type'] == 'disc_curve'].iterrows():
    new_df = cf_obj.get_interpolated_curves(row)
    disc_curves = pd.concat([disc_curves, new_df])
for i, row in uniq_curves.loc[uniq_curves['curve_type'] == 'fwd_curve'].iterrows():
    new_df = cf_obj.get_interpolated_curves(row)
    fwd_curves = pd.concat([fwd_curves, new_df])

fixing_history = sql_setup.sql_select_fixings()

disc_df = (
    disc_curves
    .set_index(["curve_name", "node_date"])["d_f"]
    .sort_index()
)

# forward curve: (curve_name, fixing_freq, node_date) -> fwd_rt
fwd_df = (
    fwd_curves
    .set_index(["curve_name", "fixing_freq", "node_date"])["fwd_rt"]
    .sort_index()
)

# fixingi: (fixing_date, rate_index) -> rate
fix_df = (
    fixing_history
    .set_index(["fixing_date", "rate_index"])["rate"]
    .sort_index()
)

#####
### GENERATE SCHED DATES
#############
BUFFER_ROWS = 200_000  # celujesz w ~200k wierszy per write (zależnie od szerokości)
buffer = []
buffer_count = 0
for table_name in cf_obj.dict_cols_loan_fin_inst.keys():
    target_table = f"{cf_obj.dict_nms_loan_fin_inst[table_name]}"
    buffer.clear()
    buffer_count = 0

    chunks = sql_setup.load_sched_params(table_name, cf_obj.dict_cols_loan_fin_inst[table_name])

    for df_in in chunks:
        parts = [
            cf_obj.gen_orgin_sched_loan_fin_inst(row, config.report_date, disc_df, fwd_df, fix_df)
            for row in df_in.itertuples(index=False, name="Row")
        ]
        parts = [p for p in parts if p is not None and len(p) > 0]
        if not parts:
            continue

        batch_df = pd.concat(parts, ignore_index=True)

        # Compute outstanding_bal, int_pmt, capital_pmt for every instrument type.
        # df_in carries balance_amt and amort_type per schedule_id — the full batch
        # is processed vectorized in one call (no per-schedule Python loop).
        batch_df = cf_obj.compute_amort_schedule_vectorized(batch_df, df_in)

        add_n = len(batch_df)
        buffer.append(batch_df)
        buffer_count += add_n

        if buffer_count >= BUFFER_ROWS:
            out_df = pd.concat(buffer, ignore_index=True)
            sql_setup.write_df(out_df, target_table, chunksize=50000)
            buffer.clear()
            buffer_count = 0

    # flush na koniec tabeli
    if buffer:
        out_df = pd.concat(buffer, ignore_index=True)
        sql_setup.write_df(out_df, target_table, chunksize=50000)

print(datetime.datetime.now())


