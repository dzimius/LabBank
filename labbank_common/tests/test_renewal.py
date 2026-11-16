"""Constant-balance-sheet renewal rate (labbank_common.curves.renewal_market_rate)."""
import numpy as np
import pandas as pd

from labbank_common.curves import renew_months, renewal_market_rate

RD = pd.Timestamp("2026-09-30")
HOR = RD + pd.Timedelta(days=365)


def _flat(rate):
    """Continuously compounded flat curve: DF(t) = exp(-r t), t in ACT/365 years."""
    return lambda d: np.exp(-rate * ((pd.to_datetime(pd.Series(d)) - RD).dt.days.to_numpy() / 365.0))


def test_renew_months():
    m = renew_months(["10Y", "3M", None, None, "1D"], ["F", "V", "F", "A", "V"], [300, 360, 6, np.nan, 1])
    assert m[0] == 120 and m[1] == 3 and m[2] == 6
    assert np.isnan(m[3]) and np.isnan(m[4])


def test_short_or_no_period_is_horizon_forward():
    end = pd.Series([RD + pd.Timedelta(days=90)] * 2)
    df_at = _flat(0.03)
    r = renewal_market_rate(end, HOR, np.array([3.0, np.nan]), df_at)
    rem = (HOR - end[0]).days / 365.0
    expected = (np.exp(0.03 * rem) - 1.0) / rem
    assert np.allclose(r, expected)


def test_long_fixing_is_par_rate():
    end = pd.Series([RD + pd.Timedelta(days=90)])
    r = renewal_market_rate(end, HOR, np.array([120.0]), _flat(0.03))
    assert abs(r[0] - (np.exp(0.03) - 1.0)) < 2e-4          # annual par rate on a flat curve
    r_h = renewal_market_rate(end, HOR, np.array([120.0]), _flat(0.03), mode="horizon")
    assert abs(r_h[0] - r[0]) > 1e-4                       # old convention differs


def test_after_horizon_is_zero():
    r = renewal_market_rate(pd.Series([HOR + pd.Timedelta(days=1)]), HOR, np.array([12.0]), _flat(0.03))
    assert r[0] == 0.0
