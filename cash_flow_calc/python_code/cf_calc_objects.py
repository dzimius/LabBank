import pandas as pd
import QuantLib as ql
import numpy as np
import datetime
from bisect import bisect_left
import sql_setup
import config

dict_cols_loan_fin_inst = {
    'loans_sched_id': ['schedule_id', 'currency', 'start_date', 'maturity_date', 'payment_freq',
                       'fixing_freq', 'dc_conv', 'b_day_conv', "rate_index", 'disc_curve', 'fwd_curve'],
    'fin_inst_sched_id': ['schedule_id', 'currency', 'start_date', 'maturity_date', 'payment_freq',
                          'fixing_freq', 'dc_conv', 'b_day_conv', "rate_index", 'disc_curve', 'fwd_curve'],
}

dict_nms_loan_fin_inst = {
    'loans_sched_id': 'loan_sched_dates',
    'fin_inst_sched_id': 'fin_inst_sched_dates',
}

sched_tables = ['loans_sched_id', 'fin_inst_sched_id']


def get_calendar_from_currency(curr:str) -> object:
    currency_to_calendar = {
        'PLN': ql.Poland(),
        'EUR': ql.TARGET(),
        'USD': ql.UnitedStates(ql.UnitedStates.NYSE)
    }
    return currency_to_calendar.get(curr, ql.TARGET())

def get_dc_conv_from_str(b:str) -> object:
    dc_conv_from_str = {
        'ACT/365': ql.Actual365Fixed(),
        'ACT/ACT': ql.ActualActual(ql.ActualActual.ISDA),
        '30/360': ql.Thirty360(ql.Thirty360.ISDA)
    }
    return dc_conv_from_str.get(b, ql.ActualActual(ql.ActualActual.ISDA))

def get_dc_from_currency(curr:str) -> object:
    currency_to_dc_conv = {
        'PLN': ql.Actual365Fixed(),
        'USD': ql.ActualActual(ql.ActualActual.ISDA),
        'EUR': ql.Thirty360(ql.Thirty360.ISDA),
    }
    return currency_to_dc_conv.get(curr, ql.Actual365Fixed())

def tenor_to_months(tenor: str) -> int:
    tenor = tenor.strip().upper()

    if tenor.endswith("M"):
        return int(tenor[:-1])

    if tenor.endswith("Y"):
        return int(tenor[:-1]) * 12

    raise ValueError(f"Unsupported tenor: {tenor}")

def freeze_fixing_dates(fixing_dates, payment_freq: str, fixing_freq: str):
    pay_m = tenor_to_months(payment_freq)
    fix_m = tenor_to_months(fixing_freq)

    if fix_m < pay_m:
        raise ValueError("Assumption violated: payment_freq must be <= fixing_freq")

    if fix_m % pay_m != 0:
        raise ValueError(f"Fixing freq {fixing_freq} not divisible by payment freq {payment_freq}")

    block = fix_m // pay_m
    n = len(fixing_dates)

    out = []
    for i in range(n):
        anchor_i = (i // block) * block
        out.append(fixing_dates[anchor_i])
    return out

# def gen_orgin_sched_loan_fin_inst(row, report_date, disc_curves, fwd_curves, fixing_history):
#     r_dt_ql = ql.Date.from_date(report_date)
#
#     cal = get_calendar_from_currency(row.currency)
#     bdc = getattr(ql, row.b_day_conv)
#     dc_conv = get_dc_conv_from_str(row.dc_conv)
#
#     start_dt_ql = ql.Date.from_date(row.start_date)
#     end_dt_ql = ql.Date.from_date(row.maturity_date)
#
#     # --- payment schedule ---
#     p_freq = ql.Period(row.payment_freq)
#     pay_dates = payment_dates_advance_n(start_dt_ql, end_dt_ql, p_freq, cal, bdc)
#
#     # znajdź pierwszy okres po report_date
#     k = bisect_left(pay_dates, r_dt_ql)
#     start_idx = max(k - 1, 0)
#     pay_dates_filt = pay_dates[start_idx:]
#
#     if not pay_dates_filt:
#         return None
#
#     # --- start/end okresu odsetkowego ---
#     if start_idx > 0:
#         first_prev = pay_dates[start_idx - 1]
#     else:
#         first_prev = start_dt_ql
#
#     cf_start_dates = [first_prev] + pay_dates_filt[:-1]
#     cf_end_dates = pay_dates_filt
#
#     fix_shift = ql.Period(-2, ql.Days)
#     fixing_dates = [cal.advance(dt, fix_shift, bdc) for dt in cf_start_dates]
#
#     accrual_yf = [
#         dc_conv.yearFraction(d0, d1)
#         for d0, d1 in zip(cf_start_dates, cf_end_dates)
#     ]
#     if row.fixing_freq is None:
#         fixing_dates = [fixing_dates[0]] * len(fixing_dates)
#     else:
#         fixing_dates = freeze_fixing_dates(
#             fixing_dates,
#             payment_freq=row.payment_freq,
#             fixing_freq=row.fixing_freq,
#         )
#
#     result_df = pd.DataFrame(
#         {
#             "schedule_id": [row.schedule_id] * len(cf_end_dates),
#             "rate_index": [row.rate_index] * len(cf_end_dates),
#             "fixing_dt": fixing_dates,
#             "cf_start_dt": cf_start_dates,
#             "cf_end_dt": cf_end_dates,
#             "cf_yf": accrual_yf,
#         }
#     )
#
#     result_df["cf_start_dt"] = ql_column_to_datetime(result_df["cf_start_dt"])
#     result_df["cf_end_dt"] = ql_column_to_datetime(result_df["cf_end_dt"])
#     result_df["fixing_dt"] = ql_column_to_datetime(result_df["fixing_dt"])
#     result_df['join_dt'] = (
#         result_df.groupby('fixing_dt')['cf_start_dt']
#         .transform('min')
#     )
#
#     result_df = result_df.merge(
#         disc_curves.loc[
#             disc_curves['curve_name'] == row.disc_curve,
#             ['curve_name', 'node_date', 'd_f']
#         ],
#         how='left',
#         left_on='cf_end_dt',
#         right_on='node_date'
#     )
#     result_df = result_df.drop(columns=['node_date', 'curve_name'])
#     result_df = result_df.merge(
#         fwd_curves.loc[
#             (fwd_curves['curve_name'] == row.fwd_curve) & (fwd_curves['fixing_freq'] == row.fixing_freq),
#             ['curve_name', 'node_date', 'fwd_rt']
#         ],
#         how='left',
#         left_on='join_dt',
#         right_on='node_date'
#     )
#     mask = result_df["join_dt"] <= config.report_date
#     tmp = result_df.loc[mask, ["join_dt", "rate_index"]].copy()
#     tmp = tmp.merge(
#         fixing_history[["fixing_date", "rate_index", "rate"]],
#         left_on=["join_dt", "rate_index"],
#         right_on=["fixing_date", "rate_index"],
#         how="left",
#     )
#     result_df.loc[mask, "fwd_rt"] = tmp["rate"].to_numpy()/100
#     result_df = result_df.drop(columns=['node_date', 'join_dt', 'curve_name'])
#
#
#     return result_df

def gen_orgin_sched_loan_fin_inst(
    row,
    report_date,
    disc_df,   # pd.Series MultiIndex: (curve_name, node_date) -> d_f
    fwd_df,    # pd.Series MultiIndex: (curve_name, fixing_freq, node_date) -> fwd_rt
    fix_df,    # pd.Series MultiIndex: (fixing_date, rate_index) -> rate
):
    r_dt_ql = ql.Date.from_date(report_date)

    cal = get_calendar_from_currency(row.currency)
    bdc = getattr(ql, row.b_day_conv)
    dc_conv = get_dc_conv_from_str(row.dc_conv)

    start_dt_ql = ql.Date.from_date(row.start_date)
    end_dt_ql = ql.Date.from_date(row.maturity_date)

    # --- payment schedule ---
    p_freq = ql.Period(row.payment_freq)
    pay_dates = payment_dates_advance_n(start_dt_ql, end_dt_ql, p_freq, cal, bdc)

    # znajdź pierwszy okres po report_date
    k = bisect_left(pay_dates, r_dt_ql)
    start_idx = max(k - 1, 0)
    pay_dates_filt = pay_dates[start_idx:]

    if not pay_dates_filt:
        return None

    # --- start/end okresu odsetkowego ---
    if start_idx > 0:
        first_prev = pay_dates[start_idx - 1]
    else:
        first_prev = start_dt_ql

    cf_start_dates = [first_prev] + pay_dates_filt[:-1]
    cf_end_dates = pay_dates_filt

    fix_shift = ql.Period(-2, ql.Days)
    fixing_dates = [cal.advance(dt, fix_shift, bdc) for dt in cf_start_dates]

    accrual_yf = [dc_conv.yearFraction(d0, d1) for d0, d1 in zip(cf_start_dates, cf_end_dates)]

    if row.fixing_freq is None:
        fixing_dates = [fixing_dates[0]] * len(fixing_dates)
    else:
        fixing_dates = freeze_fixing_dates(
            fixing_dates,
            payment_freq=row.payment_freq,
            fixing_freq=row.fixing_freq,
        )

    result_df = pd.DataFrame(
        {
            "schedule_id": [row.schedule_id] * len(cf_end_dates),
            "rate_index": [row.rate_index] * len(cf_end_dates),
            "fixing_dt": fixing_dates,
            "cf_start_dt": cf_start_dates,
            "cf_end_dt": cf_end_dates,
            "cf_yf": accrual_yf,
        }
    )

    # QuantLib.Date -> datetime
    result_df["cf_start_dt"] = ql_column_to_datetime(result_df["cf_start_dt"])
    result_df["cf_end_dt"] = ql_column_to_datetime(result_df["cf_end_dt"])
    result_df["fixing_dt"] = ql_column_to_datetime(result_df["fixing_dt"])

    # join_dt = min(cf_start_dt) dla danego fixing_dt
    result_df["join_dt"] = (
        result_df.groupby("fixing_dt")["cf_start_dt"]
        .transform("min")
    )

    n = len(result_df)

    # --- discount factors lookup (curve_name, node_date) -> d_f
    disc_idx = pd.MultiIndex.from_arrays(
        [[row.disc_curve] * n, result_df["cf_end_dt"].to_numpy()],
        names=["curve_name", "node_date"],
    )
    result_df["d_f"] = disc_df.reindex(disc_idx).to_numpy()

    # --- forward rates lookup (curve_name, fixing_freq, node_date) -> fwd_rt
    fwd_idx = pd.MultiIndex.from_arrays(
        [[row.fwd_curve] * n, [row.fixing_freq] * n, result_df["join_dt"].to_numpy()],
        names=["curve_name", "fixing_freq", "node_date"],
    )
    result_df["fwd_rt"] = fwd_df.reindex(fwd_idx).to_numpy()

    # --- overwrite z fixing_history dla join_dt <= report_date
    mask = result_df["fixing_dt"] <= report_date  # albo config.report_date
    if mask.any():
        fix_idx = pd.MultiIndex.from_arrays(
            [
                result_df.loc[mask, "fixing_dt"].to_numpy(),  # <-- tu zmiana
                result_df.loc[mask, "rate_index"].to_numpy(),
            ],
            names=["fixing_date", "rate_index"],
        )
        result_df.loc[mask, "fwd_rt"] = fix_df.reindex(fix_idx).to_numpy() / 100.0

    result_df = result_df.drop(columns=["join_dt"])
    return result_df

def payment_dates_advance_n(start_date, end_date, freq, calendar, bdc, eom=False):
    freq_in_months = freq.length() * (freq.units() == ql.Years and 12 or 1)

    diff_in_months = (
        (end_date.year() - start_date.year()) * 12
        + (end_date.month() - start_date.month())
    )

    n = diff_in_months // freq_in_months

    return [
        calendar.advance(start_date, i * freq, bdc, eom)
        for i in range(1, n + 1)
    ]

def ql_column_to_datetime(series: pd.Series) -> pd.Series:
    """Convert pandas Series of QuantLib.Date -> pandas datetime."""
    return pd.to_datetime(
        [
            datetime.date(d.year(), d.month(), d.dayOfMonth())
            if isinstance(d, ql.Date) and d != ql.Date()
            else None
            for d in series.to_numpy()
        ],
        errors="coerce",
    )

def get_unique_curves():
    uniq_discount_curves = sql_setup.sql_get_uniq_curves(['loans_sched_id', 'fin_inst_sched_id'], 'disc_curve')
    uniq_forward_curves = sql_setup.sql_get_uniq_curves(['loans_sched_id', 'fin_inst_sched_id'], 'fwd_curve')
    result_df = pd.concat([uniq_discount_curves, uniq_forward_curves])
    return result_df

def get_interpolated_curves(row):
    if row['curve_type'] == 'fwd_curve':
        curve_df = get_daily_intrpl_curve(row['curve_name'], row['fixing_freq'])
    elif row['curve_type'] == 'disc_curve':
        curve_df = get_daily_intrpl_curve(row['curve_name'])
    return curve_df

def get_daily_intrpl_curve(curve_name, fixing_freq=None):
    df_sql = sql_setup.sql_select_specific_curve(curve_name)
    x_s = df_sql['n_days']
    y_s = df_sql['d_f']
    x = x_s.to_numpy(dtype=float)
    y = y_s.to_numpy(dtype=float)
    df_intpl = interpolate_d_f(x, y, config.report_date)
    df_intpl.insert(0, "curve_name", [curve_name] * len(df_intpl))

    ## if forward rate additionaly take fixing_freq into account
    if fixing_freq:
        df_intpl["fixing_freq"] =  [fixing_freq] * len(df_intpl)
        ccy = curve_name[:3]
        cal = get_calendar_from_currency(ccy)
        dc_conv = get_dc_from_currency(ccy)
        df_intpl['end_dt'] = [cal.advance(ql.Date.from_date(dt), ql.Period(fixing_freq), ql.ModifiedFollowing)
                  for dt in df_intpl['node_date'].dt.to_pydatetime()]
        df_intpl['year_frac'] = [dc_conv.yearFraction(ql.Date.from_date(s_dt), e_dt) for s_dt, e_dt
                                 in zip(df_intpl['node_date'], df_intpl['end_dt'])]
        df_helper = df_intpl.copy()[['node_date', 'd_f']].rename(columns={'node_date': 'date', 'd_f': 'd_f_2'})
        df_intpl["end_dt"] = pd.to_datetime([d.to_date() for d in df_intpl["end_dt"]])
        df_intpl = df_intpl.merge(df_helper, how= 'left',
                                          left_on= 'end_dt', right_on= 'date')
        df_intpl['fwd_rt'] = (df_intpl['d_f']/df_intpl['d_f_2'] - 1)/df_intpl['year_frac']
        df_intpl = df_intpl.drop(columns= ['date', 'end_dt', 'year_frac', 'd_f', 'd_f_2'])
        df_intpl = df_intpl.dropna(subset=['fwd_rt'])
    return df_intpl

def interpolate_d_f(x, y, report_date):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if np.any(x == 0):
        y[x == 0] = 1.0
    else:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 1.0)

    grid_days = np.arange(0, int(x.max()) + 1, dtype=float)

    ln_y = np.log(y)
    ln_y_interp = np.interp(grid_days, x, ln_y)
    d_f_interp = np.exp(ln_y_interp)

    return pd.DataFrame({
        "n_days": grid_days.astype(int),
        "node_date": report_date + pd.to_timedelta(grid_days, unit="D"),
        "d_f": d_f_interp
    })


