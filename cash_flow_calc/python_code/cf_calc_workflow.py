import pandas as pd
import os
import sql_setup
import datetime as dt

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/cash_flow_calc"
os.chdir(BASE_DIR)


report_date = '2024-12-31'
tables = ['loans']

for table_name in tables:
    chunks_loans = sql_setup.sql_get_params(sql_setup.engine, table_name,
                             ['transaction_id', 'start_date', 'maturity_date', 'balance_amt'],
                             report_date)
    for df_loans in chunks_loans:
        print(df_loans.head())
