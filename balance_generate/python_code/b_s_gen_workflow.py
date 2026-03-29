import pandas as pd
import os
from b_s_gen_objects import ProductFactory, compute_deposit_client_rt, compute_asset_rates
from sqlalchemy import create_engine
import config
import sql_setup
from sqlalchemy import text

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/balance_generate"
os.chdir(BASE_DIR)


config.report_date = pd.to_datetime('2024-12-31')  # Set the report date
config.balance_start_date = pd.to_datetime('2015-01-01')
total_assets = 10_000_000_000
full_balance_amt = total_assets * 2 #assets + liabilities+equity = 2*assets

mode = 0 # 0 -- create new tables and remove old , 1 -- only delete rows without creating new schema of table
sql_setup.reset_data(mode, report_date=config.report_date)

df_bs_struct = pd.read_excel('input_data/bank_data.xlsx', sheet_name='bs_structure')

df_client_t = pd.read_excel('dictionaries/client_type.xlsx')
df_bs_struct['balance_amt'] = full_balance_amt * df_bs_struct['bs_percentage'] / 100
df_bs_struct['amort_type'] = df_bs_struct['amort_type'].astype('Int64')
df_result = df_bs_struct.merge(df_client_t, on='client_type_id', how='left')

# Load interest rate formula file and historical fixings
# interest_rt.xlsx columns: product_code, a, b (% points), delay, index_floor, client_floor
# rate_index per product is taken from bank_data_only_dep.xlsx (already loaded as df_bs_struct)
interest_rt = pd.read_excel('input_data/interest_rt.xlsx')
fixing_file = '../balance_gen_add_data/input/fixing_input.xlsx'
fixings_raw = pd.read_excel(fixing_file)
fixings_raw = fixings_raw.rename(columns={'date': 'fixing_date'})

product_objects = {}


parts = []
for _, row in df_result.iterrows():
    p = ProductFactory.create(row)
    df = p.build_result_df()
    parts.append(df)                       # NIE zapisujemy jeszcze do transactions

# nadaj client_id i zrób parowania na CAŁOŚCI
all_tx = pd.concat(parts, ignore_index=True)
all_tx = sql_setup.add_client_id(all_tx, seed=42, pct=0.4)

# Inject own_name, amort_type, and regulatory LCR/NSFR weights from bs_structure.
# Keyed by (product_code, bs_side) so dual-sided products (e.g. 7900) resolve correctly.
_reg_cols = ['own_name', 'amort_type', 'hqla_class', 'haircut', 'LCR', 'ASF', 'RSF']
_avail_reg = [c for c in _reg_cols if c in df_bs_struct.columns]
_meta = (
    df_bs_struct[['product_code', 'bs_side'] + _avail_reg]
    .drop_duplicates(['product_code', 'bs_side'])
    .set_index(['product_code', 'bs_side'])
)
_idx = pd.MultiIndex.from_arrays(
    [all_tx['product_code'].astype(int), all_tx['bs_side']],
    names=['product_code', 'bs_side'],
)
all_tx['own_name']   = _meta['own_name'].reindex(_idx).values

# Regulatory weights — map bs_struct column names to target column names
_weight_map = {'hqla_class': 'hqla_class', 'haircut': 'haircut',
               'LCR': 'lcr_weight', 'ASF': 'asf_weight', 'RSF': 'rsf_weight'}
for src, dst in _weight_map.items():
    if src in _meta.columns:
        all_tx[dst] = _meta[src].reindex(_idx).values
# ensure amort_type is object dtype so mixed int/None assignment works
if 'amort_type' not in all_tx.columns:
    all_tx['amort_type'] = None
all_tx['amort_type'] = all_tx['amort_type'].astype(object)
_amort_from_struct   = _meta['amort_type'].reindex(_idx).astype(object).values
# only fill where the product class did not set amort_type itself
_missing = all_tx['amort_type'].isna()
all_tx.loc[_missing, 'amort_type'] = _amort_from_struct[_missing]

# Compute curr_rt and client_rt for deposit products (formula from interest_rt.xlsx)
all_tx = compute_deposit_client_rt(all_tx, interest_rt, df_bs_struct, fixings_raw, config.report_date)

# Compute curr_rt and client_rt for loan and fin_inst products (current fixing + margin)
all_tx = compute_asset_rates(all_tx, interest_rt, df_bs_struct, fixings_raw, config.report_date)

# teraz dopiero:
sql_setup.append_df_to_table(all_tx, "transactions")   # parent
# potem dzieci:
for cls, table in ProductFactory.table_registry.items():
    pnames = [k for k,v in ProductFactory.class_registry.items() if v is cls]
    subset = all_tx[all_tx["product_name"].isin(pnames)]
    if not subset.empty:
        sql_setup.append_df_to_table(subset, table)
