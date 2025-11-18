import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import QuantLib as ql
import os
from concurrent.futures import ProcessPoolExecutor
# import matplotlib.pyplot as plt


def build_curve_ql(valuation_date: ql.Date, quotes: dict) -> ql.YieldTermStructureHandle:
    ql.Settings.instance().evaluationDate = valuation_date
    calendar = ql.Poland()
    day_count = ql.Actual365Fixed()
    fixing_days = 2
    bdc = ql.ModifiedFollowing
    eom = False

    def helper(rate, months):
        return ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(rate)),
            ql.Period(months, ql.Months),
            fixing_days,
            calendar,
            bdc,
            eom,
            day_count
        )

    helpers = [
        helper(quotes["1M"], 1),
        helper(quotes["3M"], 3),
        helper(quotes["6M"], 6),
        helper(quotes["1Y"], 12),
    ]

    settlement_date = calendar.advance(valuation_date, fixing_days, ql.Days)
    curve = ql.PiecewiseLogLinearDiscount(settlement_date, helpers, day_count)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)

def sample_hw_state_at(horizon_time: float, a: float, sigma: float, n_paths: int, seed: int = 42):
    """
    W modelu HW1F r(t) = x(t) + phi(t), gdzie x(t) spełnia OU:
       dx = -a x dt + sigma dW, x(0)=0  =>  x(t) ~ N(0,  sigma^2/(2a)*(1-e^{-2at}))
    Zwraca wektor x(t_h).
    """
    if a <= 0 or sigma <= 0:
        raise ValueError("Parametry a i sigma muszą być dodatnie.")
    var = (sigma**2) * (1.0 - np.exp(-2.0*a*horizon_time)) / (2.0*a)
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=np.sqrt(var), size=n_paths)

def ten2period(s: str) -> ql.Period:
    s = s.upper()
    return ql.Period(int(s[:-1]), ql.Months) if s.endswith("M") else ql.Period(int(s[:-1]), ql.Years)


def simulate_hw_curves_quantlib(
    subdf,
    a: float = 0.10,
    sigma: float = 0.01,
    n_paths: int = 100,
    horizon: ql.Period = ql.Period(1, ql.Years),
    output_tenors = tuple([f"{i}M" for i in range(1, 12*30+1)]),
    rng_seed: int = 123
):
    curve_date = pd.to_datetime(subdf['date'].values[0])
    ql_valuation = ql.Date.from_date(curve_date)
    cal = ql.Poland()
    dc = ql.Actual365Fixed()

    quotes = {
        '1M': subdf['WIB1M'].values[0] / 100.0,
        '3M': subdf['WIB3M'].values[0] / 100.0,
        '6M': subdf['WIB6M'].values[0] / 100.0,
        '1Y': subdf['WIB1Y'].values[0] / 100.0
    }

    yts = build_curve_ql(ql_valuation, quotes)
    model = ql.HullWhite(yts, a, sigma)

    horizon_date = cal.advance(ql_valuation, horizon)
    t_h = dc.yearFraction(ql_valuation, horizon_date)

    x_h = sample_hw_state_at(horizon_time=t_h, a=a, sigma=sigma,
                             n_paths=n_paths, seed=rng_seed)

    targets = [cal.advance(ql_valuation, ten2period(tk)) for tk in output_tenors]
    t_targets = [dc.yearFraction(ql_valuation, dT) for dT in targets]
    taus_today = [dc.yearFraction(ql_valuation, dT) for dT in targets]
    taus_forward = [dc.yearFraction(horizon_date, dT) for dT in targets]

    cols = [tk.upper() for tk in output_tenors]

    # Liczymy od razu średnią dla każdego tenora, bez przechowywania wszystkich ścieżek
    avg_rates = np.empty(len(cols))
    for j in range(len(cols)):
        if targets[j] <= horizon_date:
            P0T = yts.discount(t_targets[j])
            avg_rates[j] = -np.log(P0T) / max(taus_today[j], 1e-12)
        else:
            # policz y dla każdej ścieżki i zrób średnią
            y_samples = np.empty(n_paths)
            for i, x in enumerate(x_h):
                P = model.discountBond(t_h, t_targets[j], float(x))
                y_samples[i] = -np.log(P) / max(taus_forward[j], 1e-12)
            avg_rates[j] = y_samples.mean()

    result_curves = pd.DataFrame({
        "curve_date": curve_date.date(),
        "tenor": cols,
        "year_frac": [dc.yearFraction(ql_valuation, tgt) for tgt in targets],
        "mat_date": [pd.Timestamp(tgt.to_date()) for tgt in targets],
        "int_rate": avg_rates
    })

    return result_curves

# def curve_generation_job(df, report_date):
#     unique_dates = df['date'].unique()
#     unique_dates = unique_dates[(unique_dates<=report_date) & (unique_dates>'2020-01-01')]
#     count = 1
#     for dt in list(unique_dates):
#         sub_df = df[df['date'] == dt]
#         out = simulate_hw_curves_quantlib(sub_df)
#         if count == 1:
#             output_df = out
#         else:
#             output_df = pd.concat([output_df, out])
#         count = 0
#     return output_df

def _worker_simulate(record: dict) -> pd.DataFrame:
    subdf = pd.DataFrame([record])
    return simulate_hw_curves_quantlib(subdf)

def curve_generation_job(file_name, report_date, min_date='2020-01-01', max_workers=None):
    df = pd.read_excel(file_name)
    report_date = pd.to_datetime(report_date)
    min_date = pd.to_datetime(min_date)
    dates = pd.to_datetime(df['date'].unique())
    mask = (dates <= report_date) & (dates > min_date)
    dates = dates[mask]
    records = [df.loc[df['date'] == dt].iloc[0].to_dict() for dt in dates]
    if len(records) == 0:
        return pd.DataFrame(columns=["curve_date","tenor","year_frac","mat_date","int_rate"])
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_worker_simulate, records))
    return pd.concat(results, ignore_index=True)


