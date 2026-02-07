import pandas as pd
import os
import sql_setup
import datetime as dt
import QuantLib as ql
import cf_calc_objects as cf_obj

BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/cash_flow_calc"
os.chdir(BASE_DIR)


report_date = '2024-12-31'
tables = ['loans']
## zastanowic sie czy mozna napisac for table
# in tables i miec gdzies w slowniku jakie zmienne w zaleznosci od tabeli zeby byla jedna glowna petla

for table_name in tables:
    chunks = sql_setup.load_sched_params(table_name, cf_obj.dict_sched_col_per_table[table_name], report_date)
    for df_loans in chunks:
        df_loans.apply(lambda row: generate_schedule(row))

