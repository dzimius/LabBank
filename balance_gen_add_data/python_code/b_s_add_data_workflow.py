import pandas as pd
import os
import curve_generator, insert_beh_models
import sql_setup

if __name__ == "__main__":
    report_date = pd.to_datetime('2024-12-31')
    BASE_DIR = "C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/balance_gen_add_data"
    os.chdir(BASE_DIR)
    curve_file_name = 'input/curve_input.xlsx'
    depo_file_name = 'input/dep_beh_models.xlsx'
    loan_file_name = 'input/loan_beh_models.xlsx'

    mode = 0
    sql_setup.reset_data(mode, report_date)

###curve generation
###################
    curves_df = curve_generator.curve_generation_job(curve_file_name, report_date, mode, min_date='2024-01-01')
    sql_setup.append_df_to_table(curves_df, 'curves')


###depo beh models
######################
    df_depo_beh = insert_beh_models.depo_beh_models_job(depo_file_name, report_date)
    sql_setup.append_df_to_table(df_depo_beh, 'models_deposit')

    df_loan_beh = insert_beh_models.loan_beh_models_job(loan_file_name, report_date)
    sql_setup.append_df_to_table(df_loan_beh, 'models_loan')




### !!!!! zrob curves job jako funkcja
### dalej zrob behaviorals models jobs gdzie wrzucasz te excele do baz danych
#### dalej podpipanie krzywych discount i forward pod każdy produkt ( w SQL lub python chyba musi byc krzywa doscunt i forward w danych wejściowych)
### dalej wrzucanie modeli pod product code, że jest jeszcze nazwa modelu (SQL lub python)


