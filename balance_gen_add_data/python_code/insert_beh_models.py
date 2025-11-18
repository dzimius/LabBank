import pandas as pd


def depo_beh_models_job(file_name, report_date):
    df_depo_beh = pd.read_excel(file_name)
    df_depo_beh = df_depo_beh.melt(id_vars=['report_date', 'tenor'],  var_name='product_code', value_name='outstanding')
    df_depo_beh = df_depo_beh.dropna(subset=['outstanding'])
    df_depo_beh = df_depo_beh[['report_date', 'product_code', 'tenor', 'outstanding']]
    print('a')
    return df_depo_beh

