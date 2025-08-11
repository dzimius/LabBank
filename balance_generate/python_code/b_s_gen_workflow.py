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
full_balance_amt = 50_000_000_000

with sql_setup.engine.begin() as conn:
    for t in ["transactions", "loans", "deposits", "financial_instruments", "equity"]:
        conn.execute(text(f"IF OBJECT_ID('dbo.{t}', 'U') IS NOT NULL DROP TABLE dbo.{t};"))
    # tu kluczowe:
    sql_setup.metadata.create_all(bind=conn, checkfirst=False)

df_bs_struct = pd.read_excel('input_data/bank_data_only_dep.xlsx', sheet_name='bs_structure')

df_client_t = pd.read_excel('dictionaries/client_type.xlsx')
df_bs_struct['balance_amt'] = full_balance_amt * df_bs_struct['bs_percentage'] / 100
df_bs_struct['amort_type'] = df_bs_struct['amort_type'].astype('Int64')
df_result = df_bs_struct.merge(df_client_t, on='client_type_id', how='left')

product_objects = {}

# for i, row in df_result.iterrows():
#     product = ProductFactory.create(row)
#     df = product.build_result_df()
#     product_objects[row['product_name']] = df
#     code = str(row['product_code'])
#     #df.to_excel(f'output_data/{code}_transactions.xlsx', index=False)

for _, row in df_result.iterrows():
    product = ProductFactory.create(row)
    df = product.build_result_df()

    # 1) transactions
    sql_setup.append_transactions(df)

    # 2) tabela specyficzna
    dest = ProductFactory.table_registry.get(type(product))
    if dest:
        sql_setup.append_df_to_table(df, dest)
