import pandas as pd
from scipy.stats import truncnorm
import QuantLib as ql
import numpy as np

dict_sched_col_per_table = {
    'loans': ['transaction_id', 'start_date', 'maturity_date', 'payment_freq', 'fixing_freq', 'basis', 'b_day_conv'],
    'deposits': ['transaction_id', 'start_date', 'maturity_date', 'payment_freq', 'fixing_freq', 'basis', 'b_day_conv']
}


def get_calendar_from_currency(curr:str) -> object:
    currency_to_calendar = {
        'PLN': ql.Poland(),
        'EUR': ql.TARGET(),
        'USD': ql.UnitedStates(ql.UnitedStates.NYSE)
    }
    return currency_to_calendar.get(curr, ql.TARGET())
