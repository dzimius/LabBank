import pandas as pd
import os
from b_s_gen_objects import ProductFactory
from sqlalchemy import create_engine
import config
import sql_setup
from sqlalchemy import text

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/balance_generate"
os.chdir(BASE_DIR)


config.report_date = pd.to_datetime('2024-12-31')  # Set the report date
config.balance_start_date = pd.to_datetime('2015-01-01')
full_balance_amt = 100_000_000_000 #assets + liabilities+equity = 2*assets

mode = 0 # 0 -- create new tables and remove old , 1 -- only delete rows without creating new schema of table
sql_setup.reset_data(mode, config.report_date)

df_bs_struct = pd.read_excel('input_data/bank_data_only_dep.xlsx', sheet_name='bs_structure')

df_client_t = pd.read_excel('dictionaries/client_type.xlsx')
df_bs_struct['balance_amt'] = full_balance_amt * df_bs_struct['bs_percentage'] / 100
df_bs_struct['amort_type'] = df_bs_struct['amort_type'].astype('Int64')
df_result = df_bs_struct.merge(df_client_t, on='client_type_id', how='left')

product_objects = {}


parts = []
for _, row in df_result.iterrows():
    p = ProductFactory.create(row)
    df = p.build_result_df()
    parts.append(df)                       # NIE zapisujemy jeszcze do transactions

# nadaj client_id i zrób parowania na CAŁOŚCI
all_tx = pd.concat(parts, ignore_index=True)
all_tx = sql_setup.add_client_id(all_tx, seed=42, pct=0.4)

# teraz dopiero:
sql_setup.append_df_to_table(all_tx, "transactions")   # parent
# potem dzieci:
for cls, table in ProductFactory.table_registry.items():
    pnames = [k for k,v in ProductFactory.class_registry.items() if v is cls]
    subset = all_tx[all_tx["product_name"].isin(pnames)]
    if not subset.empty:
        sql_setup.append_df_to_table(subset, table)
