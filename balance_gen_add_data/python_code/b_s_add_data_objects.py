from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import QuantLib as ql

dict_tbl_sched_id = {
    'loans': 'loans',
    'financial_instruments': 'fin_inst',
    'deposits': 'deposits',
}
dict_tbl_sched_id_cols = {
    # amort_type included so each schedule_id maps to a single amortization type
    # product_code included for loans so prepayment model can be joined in CF calc
    # bs_side included to carry balance sheet sign (A/L) into cash flow tables
    'loans': ["currency", "bs_side", "rate_type", "start_date", "maturity_date", "payment_freq", "fixing_freq",
              "dc_conv", "b_day_conv", "rate_index", "disc_curve", "fwd_curve", "amort_type", "product_code"],
    'financial_instruments': ["currency", "bs_side", "rate_type", "product_code", "start_date", "maturity_date", "payment_freq", "fixing_freq",
                              "dc_conv", "b_day_conv", "rate_index", "disc_curve", "fwd_curve", "amort_type"],
    # deposits: grouped by contract params + product_code for separate behavioural models
    # NULL maturity_date = non-maturity (overnight)
    # rate_type: F=fixed, V=variable/float, A=administrative (bank-managed, e.g. CA/saving)
    'deposits': ["product_code", "currency", "bs_side", "rate_type", "start_date", "maturity_date", "dc_conv", "b_day_conv", "disc_curve", "fwd_curve"],
}
dict_tbl_sched_sum_cols = {
    'loans': ["init_balance_amt", "balance_amt"],
    'financial_instruments': ["balance_amt"],
    'deposits': ["balance_amt"],
}
# Columns aggregated with AVG() into the sched table (e.g. per-product margin/client rate)
dict_tbl_sched_avg_cols = {
    'loans': ["margin", "index_rt", "client_rt"],
    'financial_instruments': ["index_rt", "client_rt"],
    'deposits': ["index_rt", "client_rt"],
}

def get_calendar_from_currency(curr: str) -> ql.Calendar:
    currency_to_calendar = {
        'PLN': ql.Poland(),
        'EUR': ql.TARGET(),
        'USD': ql.UnitedStates(ql.UnitedStates.NYSE)
    }
    return currency_to_calendar.get(curr, ql.TARGET())

def get_dc_from_currency(curr: str) -> ql.DayCounter:
    currency_to_dc_conv = {
        'PLN': ql.Actual365Fixed(),
        'USD': ql.ActualActual(ql.ActualActual.ISDA),
        'EUR': ql.Thirty360(ql.Thirty360.ISDA),
    }
    return currency_to_dc_conv.get(curr, ql.Actual365Fixed())

def depo_beh_models_job(file_name: str) -> pd.DataFrame:
    df_depo_beh = pd.read_excel(file_name)
    df_depo_beh = df_depo_beh.melt(id_vars=['report_date', 'tenor'],  var_name='product_code', value_name='outstanding')
    df_depo_beh = df_depo_beh.dropna(subset=['outstanding'])
    df_depo_beh = df_depo_beh[['report_date', 'product_code', 'tenor', 'outstanding']]
    return df_depo_beh

def loan_beh_models_job(file_name: str) -> pd.DataFrame:
    df_loan_beh = pd.read_excel(file_name)
    df_loan_beh = df_loan_beh.melt(id_vars=['report_date', 'tenor'],  var_name='product_code', value_name='prep_rate')
    df_loan_beh = df_loan_beh.dropna(subset=['prep_rate'])
    df_loan_beh = df_loan_beh[['report_date', 'product_code', 'tenor', 'prep_rate']]
    return df_loan_beh

def load_historical_fixings(curve_file_name: str) -> pd.DataFrame:
    fixing_hst = pd.read_excel(curve_file_name)
    df_fixing = fixing_hst[['date', 'rate_index', 'tenor', 'currency', 'rate']].rename(columns={'date': 'fixing_date'})
    df_fixing = df_fixing.sort_values(by=['fixing_date'])
    return df_fixing


def build_curve_ql(valuation_date: ql.Date, quotes: dict[str, float]) -> ql.YieldTermStructureHandle:
    ql.Settings.instance().evaluationDate = valuation_date
    calendar = ql.Poland()
    day_count = ql.Actual365Fixed()
    fixing_days = 2
    bdc = ql.ModifiedFollowing
    eom = False

    def helper(rate, period):
        return ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(rate)),
            period,
            fixing_days,
            calendar,
            bdc,
            eom,
            day_count
        )

    helpers = [
        helper(quotes["1D"], ql.Period(1, ql.Days)),
        helper(quotes["1M"], ql.Period(1, ql.Months)),
        helper(quotes["3M"], ql.Period(3, ql.Months)),
        helper(quotes["6M"], ql.Period(6, ql.Months)),
        helper(quotes["1Y"], ql.Period(12, ql.Months)),
    ]

    settlement_date = calendar.advance(valuation_date, fixing_days, ql.Days)
    curve = ql.PiecewiseLogLinearDiscount(settlement_date, helpers, day_count)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)

def sample_hw_state_at(horizon_time: float, a: float, sigma: float, n_paths: int, seed: int = 42) -> np.ndarray:
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
    n = int(s[:-1])
    u = s[-1]
    if u == "D":
        return ql.Period(n, ql.Days)
    if u == "W":
        return ql.Period(n, ql.Weeks)
    if u == "M":
        return ql.Period(n, ql.Months)
    if u == "Y":
        return ql.Period(n, ql.Years)
    raise ValueError(f"Nieznany tenor: {s}")


def simulate_hw_curves_quantlib(
    subdf: pd.DataFrame,
    a: float = 0.10,
    sigma: float = 0.01,
    n_paths: int = 100,
    horizon: ql.Period = ql.Period(1, ql.Years),
    output_tenors: tuple[str, ...] = tuple([f"{i}M" for i in range(0, 12*30+1)]),
    rng_seed: int = 123,
) -> pd.DataFrame:
    curve_date = pd.to_datetime(subdf['date'].values[0])

    ql_valuation = ql.Date.from_date(curve_date)

    quotes = {
        '1D': subdf.loc[subdf['tenor'] == '1D', 'rate'].values[0] / 100.0,
        '1M': subdf.loc[subdf['tenor'] == '1M', 'rate'].values[0] / 100.0,
        '3M': subdf.loc[subdf['tenor'] == '3M', 'rate'].values[0] / 100.0,
        '6M': subdf.loc[subdf['tenor'] == '6M', 'rate'].values[0] / 100.0,
        '1Y': subdf.loc[subdf['tenor'] == '1Y', 'rate'].values[0] / 100.0
    }
    ccy = subdf['currency'].values[0]
    cal = get_calendar_from_currency(ccy)
    dc_conv =get_dc_from_currency(ccy)

    yts = build_curve_ql(ql_valuation, quotes)
    model = ql.HullWhite(yts, a, sigma)

    horizon_date = cal.advance(ql_valuation, horizon)
    t_h = dc_conv.yearFraction(ql_valuation, horizon_date)

    # 1D data / year_frac (do outputu)
    spot_1d_date = cal.advance(ql_valuation, ql.Period(1, ql.Days))
    spot_1d_year_frac = dc_conv.yearFraction(ql_valuation, spot_1d_date)

    # --- użyj HullWhiteProcess zamiast własnej OU-ki ---
    time_steps = 1
    hw_process = ql.HullWhiteProcess(yts, a, sigma)
    rng = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(time_steps, ql.UniformRandomGenerator())
    )
    path_generator = ql.GaussianPathGenerator(hw_process, t_h, time_steps, rng, False)

    x_h = np.empty(n_paths)
    for i in range(n_paths):
        path = path_generator.next().value()
        x_h[i] = path.back()
    # ---------------------------------------------------

    targets = [cal.advance(ql_valuation, ten2period(tk)) for tk in output_tenors]
    t_targets = [dc_conv.yearFraction(ql_valuation, dT) for dT in targets]
    taus_today = [dc_conv.yearFraction(ql_valuation, dT) for dT in targets]
    taus_forward = [dc_conv.yearFraction(horizon_date, dT) for dT in targets]

    # oryginalne tenory (żeby znaleźć pozycję 0M)
    orig_cols = [tk.upper() for tk in output_tenors]
    idx0 = orig_cols.index("0M") if "0M" in orig_cols else None

    # output: zamieniamy nazwę 0M -> 1D
    cols = ["1D" if tk.upper() == "0M" else tk.upper() for tk in output_tenors]

    avg_rates = np.empty(len(cols))
    for j in range(len(cols)):
        if targets[j] <= horizon_date:
            P0T = yts.discount(t_targets[j])
            avg_rates[j] = -np.log(P0T) / max(taus_today[j], 1e-12)
        else:
            y_samples = np.empty(n_paths)
            for i, x in enumerate(x_h):
                P = model.discountBond(t_h, t_targets[j], float(x))
                y_samples[i] = -np.log(P) / max(taus_forward[j], 1e-12)
            avg_rates[j] = y_samples.mean()

    # nadpisz "dawne 0M" stopą z 1D (teraz będzie w output jako tenor=1D)
    if idx0 is not None:
        avg_rates[idx0] = quotes["1D"]

        # i popraw daty targetów dla tej pozycji na +1D, żeby year_frac/mat_date nie były 0
        targets[idx0] = spot_1d_date

    year_fracs = [dc_conv.yearFraction(ql_valuation, tgt) for tgt in targets]
    disc_factors = [math.exp(- r * yf) for r, yf in zip(avg_rates, year_fracs)]
    day_n_list = [int(tgt - ql_valuation) for tgt in targets]

    result_curves_disc = pd.DataFrame({
        "curve_date": curve_date.date(),
        "tenor": cols,
        "year_frac": year_fracs,
        "n_days": day_n_list,
        "mat_date": [tgt.to_date() for tgt in targets],
        "zero_rate": avg_rates,
        "d_f": disc_factors,
        "currency": ccy,
        "curve_name": ccy + '_disc_curve'
    })
    result_curves_fwd = pd.DataFrame({
        "curve_date": curve_date.date(),
        "tenor": cols,
        "year_frac": year_fracs,
        "n_days": day_n_list,
        "mat_date": [tgt.to_date() for tgt in targets],
        "zero_rate": avg_rates,
        "d_f": disc_factors,
        "currency": ccy,
        "curve_name": ccy + '_fwd_curve'
    })

    result_curves = pd.concat([result_curves_disc, result_curves_fwd], ignore_index=True)

    # finalnie wymuś spójne year_frac/mat_date dla tenor=1D (na wszelki wypadek)
    mask_1d = result_curves["tenor"] == "1D"
    result_curves.loc[mask_1d, "mat_date"] = pd.Timestamp(spot_1d_date.to_date())
    result_curves.loc[mask_1d, "year_frac"] = spot_1d_year_frac

    return result_curves


def _worker_simulate(record: pd.DataFrame) -> pd.DataFrame:
    #subdf = pd.DataFrame([record])
    return simulate_hw_curves_quantlib(record)

def curve_generation_job(
    file_name: str,
    report_date: str | pd.Timestamp,
    mode: int,
    min_date: str = '2020-01-01',
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_excel(file_name)
    report_date = pd.to_datetime(report_date)
    min_date = pd.to_datetime(min_date)
    dates = pd.to_datetime(df['date'].unique())
    if mode == 1:
        min_date = report_date
    mask = (dates <= report_date) & (dates >= min_date)
    dates = dates[mask]

    for ccy in df['currency'].unique():
        records = [df.loc[(df['date'] == dt) & (df['currency'] == ccy)] for dt in dates]
        #records = [df.loc[(df['date'] == dt) & (df['currency'] == ccy)].iloc[0].to_dict() for dt in dates]
        if len(records) == 0:
            return pd.DataFrame(columns=["curve_date","tenor","year_frac","mat_date","int_rate"])
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_worker_simulate, records))
    return pd.concat(results, ignore_index=True)
