import pandas as pd
import os
import sys
import b_s_add_data_objects as bs_objs
import sql_setup

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common import rate_models

if __name__ == "__main__":
    report_date = pd.to_datetime('2026-06-30')
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE_DIR)
    curve_file_name    = 'input/curve_input.xlsx'
    fixing_file_name   = 'input/fixing_input.xlsx'
    depo_ir_file_name  = 'input/dep_beh_models_ir.xlsx'
    depo_liq_file_name = 'input/dep_beh_models_liq.xlsx'
    loan_file_name     = 'input/loan_beh_models.xlsx'
    tables = ['loans']

    mode = 0
    sql_setup._ensure_schemas()
    sql_setup.reset_data_models(mode, report_date, ['models_loan', 'models_deposit_ir', 'models_deposit_liq', 'models_rate'])

    sql_setup.reset_data_remove_always(["mkt.curves", "mkt.fixings", "sched.loans", "sched.fin_inst", "sched.deposits"])

###curve generation
###################
    df_curves = bs_objs.curve_generation_job(curve_file_name, report_date, mode, min_date='2026-01-01')
    sql_setup.append_df_to_table(df_curves, 'curves')

#####
## load fixing history
    df_fixing = bs_objs.load_historical_fixings(fixing_file_name)
    sql_setup.append_df_to_table(df_fixing, 'fixings')

###depo beh models (IR and LIQ)
######################
    df_depo_beh_ir = bs_objs.depo_beh_models_job(depo_ir_file_name)
    sql_setup.append_df_to_table(df_depo_beh_ir, 'models_deposit_ir')

    df_depo_beh_liq = bs_objs.depo_beh_models_job(depo_liq_file_name)
    sql_setup.append_df_to_table(df_depo_beh_liq, 'models_deposit_liq')

    df_loan_beh = bs_objs.loan_beh_models_job(loan_file_name)
    sql_setup.append_df_to_table(df_loan_beh, 'models_loan')

###client-rate model (beta / margin / floors / caps per product) -> bs.models_rate
######################
    df_rate = rate_models.read_rate_excel(path=rate_models.EXCEL_PATH)
    sql_setup.append_df_to_table(df_rate, 'models_rate')


#### create sched id tables
    for table in bs_objs.dict_tbl_sched_id.keys():
        sql_setup.create_sched_id_tbl_sql(
            sql_setup.engine,
            source_table=table,
            target_table=bs_objs.dict_tbl_sched_id[table],
            columns=bs_objs.dict_tbl_sched_id_cols[table],
            sum_cols=bs_objs.dict_tbl_sched_sum_cols[table],
            avg_cols=bs_objs.dict_tbl_sched_avg_cols.get(table, []),
            null_cols=bs_objs.dict_tbl_sched_null_cols.get(table, []),
            source_schema="schemat",
            target_schema="sched",
        )
        sql_setup.update_schedule_id_sql(
            sql_setup.engine,
            table_name=table,
            sched_table_name=bs_objs.dict_tbl_sched_id[table],
            columns=bs_objs.dict_tbl_sched_id_cols[table],
            source_schema="schemat",
            sched_schema="sched",
        )


### dalej wrzucanie modeli pod product code, że jest jeszcze nazwa modelu (SQL lub python)


