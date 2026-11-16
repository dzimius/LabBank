import pandas as pd
import os
import sys
import b_s_add_data_objects as bs_objs
import sql_setup

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common import rate_models
from labbank_common.bank_config import get_bank
from labbank_common import fx
from labbank_common import catalogue

if __name__ == "__main__":
    report_date = pd.Timestamp(get_bank().report_date)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE_DIR)
    input_dir = get_bank().models_dir()
    depo_ir_file_name  = input_dir / 'dep_beh_models_ir.xlsx'
    depo_liq_file_name = input_dir / 'dep_beh_models_liq.xlsx'
    loan_file_name     = input_dir / 'loan_beh_models.xlsx'
    depo_tdrr_file_name = input_dir / 'dep_redemption_models.xlsx'
    tables = ['loans']

    mode = 0
    sql_setup._ensure_schemas()
    sql_setup.reset_data_models(mode, report_date, ['models_loan', 'models_deposit_ir', 'models_deposit_liq', 'models_rate', 'models_deposit_tdrr'])

    sql_setup.reset_data_remove_always(["mkt.curves", "mkt.fixings", "mkt.fx_rates",
                                        "sched.loans", "sched.fin_inst", "sched.deposits"])

###curve generation + fixing history: home currency + every foreign currency the bank holds
###################
    for curve_file in fx.market_input_files('curve_input.xlsx'):
        df_curves = bs_objs.curve_generation_job(curve_file, report_date, mode, min_date='2026-01-01')
        basis_file = curve_file.parent / 'curve_basis.csv'          # multi-curve (OIS + tenors)
        if basis_file.exists():
            df_curves = bs_objs.apply_curve_basis(df_curves, basis_file)
            print(f"  {curve_file.parent.name}: multi-curve from {basis_file.name}: "
                  f"{sorted(df_curves['curve_name'].unique())}")
        sql_setup.append_df_to_table(df_curves, 'curves')

    for fixing_file in fx.market_input_files('fixing_input.xlsx'):
        df_fixing = bs_objs.load_historical_fixings(fixing_file)
        sql_setup.append_df_to_table(df_fixing, 'fixings')

### FX rates (units of currency per 1 home-currency unit) -- only banks with foreign currencies
    if get_bank().market.foreign_ccys:
        df_fx = pd.read_excel(fx.fx_input_path())
        df_fx = df_fx[df_fx['currency'].isin(get_bank().market.foreign_ccys)]
        sql_setup.append_df_to_table(df_fx, 'fx_rates')

###depo beh models (IR and LIQ)
######################
    # catalogue variants (bank_data sheet catalogue) get their parent's models
    _pmap = catalogue.parent_map(get_bank()) if catalogue.bank_data_path(get_bank()).exists() else {}
    df_depo_beh_ir = bs_objs.depo_beh_models_job(depo_ir_file_name, _pmap)
    sql_setup.append_df_to_table(df_depo_beh_ir, 'models_deposit_ir')

    df_depo_beh_liq = bs_objs.depo_beh_models_job(depo_liq_file_name, _pmap)
    sql_setup.append_df_to_table(df_depo_beh_liq, 'models_deposit_liq')
    bs_objs.check_core_within_stable(df_depo_beh_ir, df_depo_beh_liq)

    df_loan_beh = bs_objs.loan_beh_models_job(loan_file_name, _pmap)
    sql_setup.append_df_to_table(df_loan_beh, 'models_loan')

    # term-deposit early-redemption model (EBA RTS 2024/857 Art. 10)
    df_depo_tdrr = bs_objs.depo_tdrr_models_job(depo_tdrr_file_name, _pmap)
    sql_setup.append_df_to_table(df_depo_tdrr, 'models_deposit_tdrr')

###client-rate model (beta / margin / floors / caps per product) -> bs.models_rate
######################
    df_rate = rate_models.read_rate_excel()
    sql_setup.append_df_to_table(df_rate, 'models_rate')


#### create sched id tables
    for table, cols in bs_objs.optional_source_cols.items():
        added = sql_setup.ensure_optional_columns(sql_setup.engine, "schemat", table, cols)
        if added:
            print(f"  schemat.{table}: added optional NULL columns {added}")
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


