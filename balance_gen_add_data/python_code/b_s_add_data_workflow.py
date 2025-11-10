import pandas as pd
import os
import curve_generator
import sql_setup
import matplotlib.pyplot as plt

if __name__ == "__main__":
    BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/balance_gen_add_data"
    os.chdir(BASE_DIR)
    curves_df = pd.read_excel('input/curve_input.xlsx')
    report_date = pd.to_datetime('2024-12-31')
    curves_df = curve_generator.curve_generation_job(curves_df, report_date, min_date='2024-01-01')
    sql_setup.append_df_to_table(curves_df, 'curves')


### !!!!! zrob curves job jako funkcja
### dalej zrob behaviorals models jobs gdzie wrzucasz te excele do baz danych
#### dalej podpipanie krzywych discount i forward pod każdy produkt ( w SQL lub python chyba musi byc krzywa doscunt i forward w danych wejściowych)
### dalej wrzucanie modeli pod product code, że jest jeszcze nazwa modelu (SQL lub python)


