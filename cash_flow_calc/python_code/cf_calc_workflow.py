import pandas as pd
import os
import sql_setup
import config
import datetime as dt
import QuantLib as ql
import cf_calc_objects as cf_obj
import datetime

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/cash_flow_calc"
os.chdir(BASE_DIR)

print(datetime.datetime.now(), "Starting cash flow calculation workflow...")
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
models_loan = sql_setup.sql_select_models_loan()
models_deposit = sql_setup.sql_select_models_deposit()

# Load floor parameters per product from interest_rt.xlsx
_ir_df = pd.read_excel('../balance_generate/input_data/interest_rt.xlsx')
ir_params = {}
for _, _r in _ir_df.iterrows():
    pc = int(_r['product_code'])
    ir_params[pc] = {
        'index_floor': float(_r['index_floor']) if 'index_floor' in _ir_df.columns and not pd.isna(_r['index_floor']) else None,
        'client_floor': float(_r['client_floor']) if 'client_floor' in _ir_df.columns and not pd.isna(_r['client_floor']) else None,
    }

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
buffer_beh = []
buffer_beh_count = 0

for table_name in cf_obj.dict_cols_loan_fin_inst.keys():
    target_table = cf_obj.dict_nms_loan_fin_inst[table_name]
    is_loans    = table_name == 'loans'
    is_fin_inst = table_name == 'fin_inst'
    buffer.clear()
    buffer_count = 0
    buffer_beh.clear()
    buffer_beh_count = 0

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
        batch_df = cf_obj.compute_amort_schedule_vectorized(batch_df, df_in, ir_params=ir_params, exact=True)

        if is_loans:
            # Join start_date and product_code from df_in to compute loan age tenor
            batch_df = batch_df.merge(
                df_in[['schedule_id', 'start_date', 'product_code']],
                on='schedule_id', how='left'
            )
            batch_df['start_date'] = pd.to_datetime(batch_df['start_date'])
            batch_df['_months'] = (
                (batch_df['cf_end_dt'].dt.year - batch_df['start_date'].dt.year) * 12
                + (batch_df['cf_end_dt'].dt.month - batch_df['start_date'].dt.month)
            ).astype(int)
            batch_df['tenor'] = batch_df['_months'].astype(str) + 'M'

            # Join prepayment rate from models_loan on product_code + tenor
            batch_df = batch_df.merge(
                models_loan[['product_code', 'tenor', 'prep_rate']],
                on=['product_code', 'tenor'], how='left'
            )
            batch_df['cpr_rate'] = batch_df['prep_rate'].fillna(0.0)

            batch_df = cf_obj.compute_adjusted_schedule(batch_df, exact=True)

            # Split into origin (contractual) and behavioural tables
            BEH_COLS = ['schedule_id', 'bs_side', 'rate_index', 'fixing_dt', 'cf_start_dt', 'cf_end_dt',
                        'cf_yf', 'd_f', 'fwd_rt', 'margin', 'client_rt', 'outstanding_adj', 'capital_adj',
                        'prepayment_pmt', 'int_adj']
            beh_df = batch_df[BEH_COLS].copy().rename(columns={
                'outstanding_adj': 'outstanding_bal',
                'capital_adj':     'capital_pmt',
                'int_adj':         'int_pmt',
            })
            # Drop periods where loan is already fully repaid (prepayment shortens schedule)
            beh_df = beh_df[beh_df['outstanding_bal'] > 0]
            beh_df['total_pmt'] = beh_df['capital_pmt'] + beh_df['prepayment_pmt'] + beh_df['int_pmt']

            batch_df = batch_df.drop(
                columns=['start_date', 'product_code', '_months', 'tenor', 'prep_rate',
                         'cpr_rate', 'outstanding_adj', 'capital_adj', 'prepayment_pmt', 'int_adj',
                         '_is_annuity']
            )

            add_beh = len(beh_df)
            buffer_beh.append(beh_df)
            buffer_beh_count += add_beh
            if buffer_beh_count >= BUFFER_ROWS:
                out_beh = pd.concat(buffer_beh, ignore_index=True)
                sql_setup.write_df(out_beh, 'loan_beh', schema='cf', chunksize=50000)
                buffer_beh.clear()
                buffer_beh_count = 0

        if not is_loans:
            batch_df = batch_df.drop(columns=['_is_annuity', 'margin', 'client_rt'], errors='ignore')

        add_n = len(batch_df)
        buffer.append(batch_df)
        buffer_count += add_n

        if buffer_count >= BUFFER_ROWS:
            out_df = pd.concat(buffer, ignore_index=True)
            sql_setup.write_df(out_df, target_table, schema='cf', chunksize=50000)
            if is_fin_inst:
                sql_setup.write_df(out_df, 'fin_inst_beh', schema='cf', chunksize=50000)
            buffer.clear()
            buffer_count = 0

    # flush na koniec tabeli
    if buffer:
        out_df = pd.concat(buffer, ignore_index=True)
        sql_setup.write_df(out_df, target_table, schema='cf', chunksize=50000)
        if is_fin_inst:
            sql_setup.write_df(out_df, 'fin_inst_beh', schema='cf', chunksize=50000)
    if is_loans and buffer_beh:
        out_beh = pd.concat(buffer_beh, ignore_index=True)
        sql_setup.write_df(out_beh, 'loan_beh', schema='cf', chunksize=50000)

### DEPOSITS (term + non-maturity)
for table_name in cf_obj.dict_cols_deposits.keys():
    orig_table = cf_obj.dict_nms_deposits[table_name]
    beh_table = cf_obj.dict_nms_deposits_beh[table_name]
    buffer.clear()
    buffer_count = 0
    buffer_beh.clear()
    buffer_beh_count = 0

    chunks = sql_setup.load_sched_params(table_name, cf_obj.dict_cols_deposits[table_name])

    for df_in in chunks:
        # --- origin schedule ---
        orig_parts = [
            cf_obj.gen_deposit_sched(row, config.report_date, disc_df)
            for row in df_in.itertuples(index=False, name="Row")
        ]
        orig_parts = [p for p in orig_parts if p is not None and len(p) > 0]
        if orig_parts:
            batch_df = pd.concat(orig_parts, ignore_index=True)
            batch_df['total_pmt'] = batch_df['capital_pmt'] + batch_df['int_pmt']
            buffer.append(batch_df)
            buffer_count += len(batch_df)
            if buffer_count >= BUFFER_ROWS:
                sql_setup.write_df(pd.concat(buffer, ignore_index=True), orig_table, schema='cf', chunksize=50000)
                buffer.clear()
                buffer_count = 0

        # --- behavioural schedule ---
        beh_parts = []
        for row in df_in.itertuples(index=False, name="Row"):
            product_models = models_deposit[models_deposit['product_code'] == row.product_code]
            part = cf_obj.compute_deposit_beh_schedule(
                row, config.report_date, disc_df, product_models
            )
            if part is not None and len(part) > 0:
                beh_parts.append(part)

        if beh_parts:
            beh_df = pd.concat(beh_parts, ignore_index=True)
            beh_df['total_pmt'] = beh_df['capital_pmt'] + beh_df['int_pmt']
            buffer_beh.append(beh_df)
            buffer_beh_count += len(beh_df)
            if buffer_beh_count >= BUFFER_ROWS:
                sql_setup.write_df(pd.concat(buffer_beh, ignore_index=True), beh_table, schema='cf', chunksize=50000)
                buffer_beh.clear()
                buffer_beh_count = 0

    if buffer:
        sql_setup.write_df(pd.concat(buffer, ignore_index=True), orig_table, schema='cf', chunksize=50000)
    if buffer_beh:
        sql_setup.write_df(pd.concat(buffer_beh, ignore_index=True), beh_table, schema='cf', chunksize=50000)

#####################
####### interest rate gap
#################################

liq_orig, liq_beh = sql_setup.compute_liq_gap(config.report_date)
sql_setup.write_df(liq_orig, 'liq_gap_orig', schema='irrbb')
sql_setup.write_df(liq_beh,  'liq_gap_beh',  schema='irrbb')

ir_orig, ir_beh, ir_beh_a = sql_setup.compute_ir_gap(config.report_date)
sql_setup.write_df(ir_orig,  'ir_gap_orig',  schema='irrbb')
sql_setup.write_df(ir_beh,   'ir_gap_beh',   schema='irrbb')
sql_setup.write_df(ir_beh_a, 'ir_gap_beh_a', schema='irrbb')

print(datetime.datetime.now(), "Finished cash flow calculation workflow.")
