import pandas as pd


def depo_beh_models_job(file_name, report_date):
    df_depo_beh = pd.read_excel(file_name)
    df_depo_beh = df_depo_beh.melt(id_vars=['report_date', 'tenor'],  var_name='product_code', value_name='outstanding')
    df_depo_beh = df_depo_beh.dropna(subset=['outstanding'])
    df_depo_beh = df_depo_beh[['report_date', 'product_code', 'tenor', 'outstanding']]
    return df_depo_beh

def loan_beh_models_job(file_name, report_date):
    df_loan_beh = pd.read_excel(file_name)
    df_loan_beh = df_loan_beh.melt(id_vars=['report_date', 'tenor'],  var_name='product_code', value_name='prep_rate')
    df_loan_beh = df_loan_beh.dropna(subset=['prep_rate'])
    df_loan_beh = df_loan_beh[['report_date', 'product_code', 'tenor', 'prep_rate']]
    return df_loan_beh

