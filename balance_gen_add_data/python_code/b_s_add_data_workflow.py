import pandas as pd
import os
import curve_generator, insert_beh_models
import sql_setup
import matplotlib.pyplot as plt

if __name__ == "__main__":
    report_date = pd.to_datetime('2024-12-31')
    BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/balance_gen_add_data"
    os.chdir(BASE_DIR)
    curve_file_name = 'input/curve_input.xlsx'
    depo_file_name = 'input/dep_beh_models.xlsx'

###curve generation
###################
    curves_df = curve_generator.curve_generation_job(curve_file_name, report_date, min_date='2024-01-01')
    sql_setup.append_df_to_table(curves_df, 'curves')
###depo beh models
######################
    df_depo_beh = insert_beh_models.depo_beh_models_job(depo_file_name, report_date)
    sql_setup.append_df_to_table(df_depo_beh, 'models_deposit')




### !!!!! zrob curves job jako funkcja
### dalej zrob behaviorals models jobs gdzie wrzucasz te excele do baz danych
#### dalej podpipanie krzywych discount i forward pod każdy produkt ( w SQL lub python chyba musi byc krzywa doscunt i forward w danych wejściowych)
### dalej wrzucanie modeli pod product code, że jest jeszcze nazwa modelu (SQL lub python)


