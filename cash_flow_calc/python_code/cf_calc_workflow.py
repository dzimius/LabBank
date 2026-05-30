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

fixing_history     = sql_setup.sql_select_fixings()
models_loan        = sql_setup.sql_select_models_loan()
models_deposit_ir  = sql_setup.sql_select_models_deposit_ir()
models_deposit_liq = sql_setup.sql_select_models_deposit_liq()

# Load floor parameters per product from interest_rt.xlsx
_ir_df = pd.read_excel('../balance_generate/input_data/interest_rt.xlsx')
ir_params = {}
for _, _r in _ir_df.iterrows():
    pc = int(_r['product_code'])
    ir_params[pc] = {
        'index_floor': float(_r['index_floor']) if 'index_floor' in _ir_df.columns and not pd.isna(_r['index_floor']) else None,
        'client_floor': float(_r['client_floor']) if 'client_floor' in _ir_df.columns and not pd.isna(_r['client_floor']) else None,
        'delay': int(_r['delay']) if 'delay' in _ir_df.columns and not pd.isna(_r['delay']) else 0,
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
BUFFER_ROWS = 200_000
buffer: list[pd.DataFrame] = []
buffer_count = 0

def _flush(buf: list, schema: str = 'cf') -> None:
    if buf:
        sql_setup.write_df(pd.concat(buf, ignore_index=True), 'products', schema=schema, chunksize=50000)
        buf.clear()

def _add_sched_meta(df: pd.DataFrame, df_in: pd.DataFrame, product_type: str) -> pd.DataFrame:
    """Join currency + product_code from the sched chunk and stamp product_type."""
    meta = df_in[['schedule_id', 'currency', 'product_code']].drop_duplicates()
    df = df.merge(meta, on='schedule_id', how='left')
    df['product_type'] = product_type
    return df

### LOANS + FIN_INST
for table_name in cf_obj.dict_cols_loan_fin_inst.keys():
    is_loans    = table_name == 'loans'
    product_type = 'L' if is_loans else 'F'
    buffer.clear()
    buffer_count = 0

    chunks = sql_setup.load_sched_params(table_name, cf_obj.dict_cols_loan_fin_inst[table_name])

    for df_in in chunks:
        parts = [
            cf_obj.gen_orgin_sched_loan_fin_inst(row, config.report_date, disc_df, fwd_df, fix_df, ir_params=ir_params)
            for row in df_in.itertuples(index=False, name="Row")
        ]
        parts = [p for p in parts if p is not None and len(p) > 0]
        if not parts:
            continue

        batch_df = pd.concat(parts, ignore_index=True)
        batch_df = cf_obj.compute_amort_schedule_vectorized(batch_df, df_in, ir_params=ir_params, exact=True)

        if is_loans:
            # ── contractual (orig) side ──────────────────────────────────────
            orig_df = batch_df[['schedule_id', 'bs_side', 'rate_index', 'fixing_dt',
                                 'cf_start_dt', 'cf_start_dt_delay', 'cf_end_dt', 'cf_yf', 'd_f', 'fwd_rt',
                                 'margin', 'client_rt', 'outstanding_bal',
                                 'capital_pmt', 'int_pmt', 'total_pmt']].copy()

            # ── behavioural (beh) side ───────────────────────────────────────
            batch_df = batch_df.merge(
                df_in[['schedule_id', 'start_date', 'product_code']],
                on='schedule_id', how='left'
            )
            batch_df['start_date'] = pd.to_datetime(batch_df['start_date'])
            batch_df['_months'] = (
                (batch_df['cf_end_dt'].dt.year  - batch_df['start_date'].dt.year)  * 12
                + (batch_df['cf_end_dt'].dt.month - batch_df['start_date'].dt.month)
            ).astype(int)
            batch_df['tenor'] = batch_df['_months'].astype(str) + 'M'
            batch_df = batch_df.merge(
                models_loan[['product_code', 'tenor', 'prep_rate']],
                on=['product_code', 'tenor'], how='left'
            )
            batch_df['cpr_rate'] = batch_df['prep_rate'].fillna(0.0)
            batch_df = cf_obj.compute_adjusted_schedule(batch_df, exact=True)

            beh_df = (
                batch_df[['schedule_id', 'bs_side', 'rate_index', 'fixing_dt',
                           'cf_start_dt', 'cf_start_dt_delay', 'cf_end_dt', 'cf_yf', 'd_f', 'fwd_rt',
                           'margin', 'client_rt', 'outstanding_adj', 'capital_adj',
                           'prepayment_pmt', 'int_adj']]
                .copy()
                .rename(columns={'outstanding_adj': 'outstanding_bal',
                                 'capital_adj':     'capital_pmt',
                                 'int_adj':         'int_pmt'})
            )
            beh_df = beh_df[beh_df['outstanding_bal'] > 0]
            beh_df['total_pmt'] = (beh_df['capital_pmt']
                                   + beh_df['prepayment_pmt']
                                   + beh_df['int_pmt'])

            # Merge orig + beh: left join keeps all contractual periods;
            # fully-prepaid periods (absent from beh) get beh_* = 0, comp_* = -con_*.
            merged = cf_obj.merge_cf_orig_beh(orig_df, beh_df, how='left')

        else:
            # fin_inst: beh = orig (no behavioural adjustment)
            fin_inst_df = batch_df.drop(columns=['_is_annuity', 'margin', 'client_rt'],
                                        errors='ignore')
            merged = cf_obj.merge_cf_orig_beh(fin_inst_df, fin_inst_df, how='inner')

        merged = _add_sched_meta(merged, df_in, product_type)

        buffer.append(merged)
        buffer_count += len(merged)
        if buffer_count >= BUFFER_ROWS:
            _flush(buffer)
            buffer_count = 0

    _flush(buffer)
    buffer_count = 0

### DEPOSITS (term + non-maturity)
buffer.clear()
buffer_count = 0
liq_buffer: list[pd.DataFrame] = []
liq_buffer_count = 0

def _flush_liq(buf: list) -> None:
    if buf:
        sql_setup.write_df(pd.concat(buf, ignore_index=True), 'products_liq', schema='cf', chunksize=50000)
        buf.clear()

for table_name in cf_obj.dict_cols_deposits.keys():
    chunks = sql_setup.load_sched_params(table_name, cf_obj.dict_cols_deposits[table_name])

    for df_in in chunks:
        orig_parts = [
            cf_obj.gen_deposit_sched(row, config.report_date, disc_df)
            for row in df_in.itertuples(index=False, name="Row")
        ]
        beh_parts     = []
        liq_beh_parts = []
        for row in df_in.itertuples(index=False, name="Row"):
            # IR behavioral model (used for NII, EVE, IR gap)
            product_models_ir = models_deposit_ir[models_deposit_ir['product_code'] == row.product_code]
            part = cf_obj.compute_deposit_beh_schedule(
                row, config.report_date, disc_df, product_models_ir
            )
            if part is not None and len(part) > 0:
                beh_parts.append(part)

            # LIQ behavioral model (used for LCR, NSFR, LIQ gap)
            product_models_liq = models_deposit_liq[models_deposit_liq['product_code'] == row.product_code]
            liq_part = cf_obj.compute_deposit_beh_schedule(
                row, config.report_date, disc_df, product_models_liq
            )
            if liq_part is not None and len(liq_part) > 0:
                liq_beh_parts.append(liq_part)

        orig_parts = [p for p in orig_parts if p is not None and len(p) > 0]
        if not orig_parts and not beh_parts:
            continue

        orig_batch = pd.concat(orig_parts, ignore_index=True) if orig_parts else pd.DataFrame()
        beh_batch  = pd.concat(beh_parts,  ignore_index=True) if beh_parts  else pd.DataFrame()

        if not orig_batch.empty:
            orig_batch['total_pmt'] = orig_batch['capital_pmt'] + orig_batch['int_pmt']
        if not beh_batch.empty:
            beh_batch['total_pmt'] = beh_batch['capital_pmt'] + beh_batch['int_pmt']

        # No IR beh model for this chunk: treat beh = orig (no behavioural adjustment)
        if beh_batch.empty:
            beh_batch = orig_batch.copy()
        # No orig (already matured): treat orig = beh
        if orig_batch.empty:
            orig_batch = beh_batch.copy()

        # Full outer join: orig bullet vs N IR beh tenor-bucket rows per schedule_id
        merged = cf_obj.merge_cf_orig_beh(orig_batch, beh_batch, how='outer')
        merged = _add_sched_meta(merged, df_in, 'D')

        buffer.append(merged)
        buffer_count += len(merged)
        if buffer_count >= BUFFER_ROWS:
            _flush(buffer)
            buffer_count = 0

        # LIQ behavioral schedule → cf.products_liq
        if liq_beh_parts:
            liq_beh_batch = pd.concat(liq_beh_parts, ignore_index=True)
            liq_beh_batch['total_pmt'] = liq_beh_batch['capital_pmt'] + liq_beh_batch['int_pmt']
            liq_beh_batch = liq_beh_batch.rename(columns={
                'outstanding_bal': 'beh_outstanding',
                'capital_pmt':     'beh_capital_pmt',
                'int_pmt':         'beh_interest_pmt',
                'total_pmt':       'beh_total_pmt',
            })
            liq_beh_batch = _add_sched_meta(liq_beh_batch, df_in, 'D')
            liq_buffer.append(liq_beh_batch)
            liq_buffer_count += len(liq_beh_batch)
            if liq_buffer_count >= BUFFER_ROWS:
                _flush_liq(liq_buffer)
                liq_buffer_count = 0

    _flush(buffer)
    _flush_liq(liq_buffer)
    buffer_count = 0
    liq_buffer_count = 0

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
