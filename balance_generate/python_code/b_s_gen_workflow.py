import pandas as pd
import os
from b_s_gen_objects import ProductFactory
import config

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/balance_generate"
os.chdir(BASE_DIR)

config.report_date = pd.to_datetime('2024-12-31')  # Set the report date
full_balance_amt = 10_000_000_000

df_bs_struct = pd.read_excel('input_data/bank_data_only_dep.xlsx', sheet_name='bs_structure')

df_client_t = pd.read_excel('dictionaries/client_type.xlsx')
df_bs_struct['balance_amt'] = full_balance_amt * df_bs_struct['bs_percentage'] / 100
df_bs_struct['amort_type'] = df_bs_struct['amort_type'].astype('Int64')
df_result = df_bs_struct.merge(df_client_t, on='client_type_id', how='left')

product_objects = {}

for i, row in df_result.iterrows():
    product = ProductFactory.create(row)
    df = product.build_result_df()
    product_objects[row['product_name']] = df
    code = str(row['product_code'])
    df.to_excel(f'output_data/{code}_transactions.xlsx', index=False)

# for i, row in df_result.iterrows():
#     try:
#         product = ProductFactory.create(row)
#         product_objects[i] = product
#     except Exception as e:
#         print(f"Błąd w wierszu {i}: {e}")