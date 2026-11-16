"""Invariants of the supervisory shocks and of the shared fixing rules (fixing_engine)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "python_code")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))
import eba_shock_curves as esc  # noqa: E402
import fixing_engine as fe  # noqa: E402

EUR = {"par": 200, "short": 250, "long": 100}


def test_short_shock_decays_with_four_year_constant():
    t = np.array([0.0, 1.0, 4.0])
    sr = esc.shock_bps_at_tenors(t, "sr_up", EUR)
    np.testing.assert_allclose(sr, 250 * np.exp(-t / 4))
    assert sr[2] == pytest.approx(91.97, abs=0.01)          # review: 91.97bp at 4Y, not 4.58bp


def test_flattener_and_steepener_coefficients():
    t = np.array([10.0])
    fs, fl = np.exp(-t / 4), 1 - np.exp(-t / 4)
    np.testing.assert_allclose(esc.shock_bps_at_tenors(t, "flat", EUR), 0.8 * 250 * fs - 0.6 * 100 * fl)
    np.testing.assert_allclose(esc.shock_bps_at_tenors(t, "steep", EUR), -0.65 * 250 * fs + 0.9 * 100 * fl)


def _curve(rate, days=(0, 365, 3650, 18250)):
    d = np.array(days, dtype=float)
    return pd.DataFrame({"n_days": d, "d_f": np.exp(-rate * d / 365.25)})


def test_base_curve_is_never_floored_and_down_shock_uses_eba_bound():
    neg = esc.apply_shock_to_disc_curve(_curve(-0.005), "base", EUR)
    assert neg["zero_rt"].iloc[1] == pytest.approx(-0.005, abs=1e-9)    # -0.5% stays -0.5%
    dn = esc.apply_shock_to_disc_curve(_curve(0.01), "par_dn", EUR)
    # 1% - 2% = -1% at 1Y: above the -147bp bound -> not floored at 0% any more
    assert dn["zero_rt"].iloc[1] == pytest.approx(-0.01, abs=1e-6)
    # an already lower observed rate is never lifted by the bound
    low = esc.apply_shock_to_disc_curve(_curve(-0.02), "par_dn", EUR)
    assert low["zero_rt"].iloc[1] == pytest.approx(-0.02, abs=1e-6)
    np.testing.assert_allclose(esc.eba_lower_bound_bps([0, 1, 50, 60]), [-150, -147, 0, 0])


def _rows():
    # monthly coupons, 3M fixing: one block Oct-Dec (fixed 2 days before 1 Oct), one Jan-Mar
    starts = pd.to_datetime(["2026-10-01", "2026-11-01", "2026-12-01", "2027-01-01", "2027-02-01", "2027-03-01"])
    ends = starts + pd.offsets.MonthBegin(1)
    fix = pd.to_datetime(["2026-09-29"] * 3 + ["2026-12-30"] * 3)
    return pd.DataFrame({"schedule_id": 1, "currency": "EUR", "rate_type": "V", "fixing_dt": fix,
                         "cf_start_dt": starts, "cf_end_dt": ends,
                         "cf_yf": (ends - starts).days / 360.0})


def test_one_forward_period_per_fixing_block_on_index_basis():
    s, e, yf = fe.block_periods(_rows())
    assert s.nunique() == 2 and e.nunique() == 2                    # one period per block
    assert s.iloc[3] == pd.Timestamp("2027-01-01")                   # accrual start, not fixing date
    assert yf.iloc[0] == pytest.approx(92 / 360)                     # EURIBOR ACT/360


def test_post_reset_coupons_of_fixed_products_float():
    df = _rows().assign(rate_type="F")
    rt = fe.effective_rate_type(df, pd.Timestamp("2026-09-30"))
    assert list(rt) == ["F"] * 3 + ["V"] * 3                         # fixed until the reset


def test_nmd_outstanding_keeps_todays_rate():
    # IR profile = repricing profile: NMD rows (deposits only) lock; others untouched
    df = pd.DataFrame({"product_code": ["6000", "8000", "7900", "8000"],
                       "product_type": ["D", "D", "D", "L"],
                       "rate_type":    ["A", "V", "V", "V"]})
    rt = fe.effective_rate_type(df, pd.Timestamp("2026-09-30"), nmd_codes={"6000", "8000"})
    assert list(rt) == ["F", "F", "V", "V"]


def test_horizon_accrual_counts_income_earned_in_the_year():
    rd, he = pd.Timestamp("2026-09-30"), pd.Timestamp("2027-09-30")
    df = pd.DataFrame({"cf_start_dt": pd.to_datetime(["2026-10-01", "2026-04-01", "2026-10-02"]),
                       "cf_end_dt": pd.to_datetime(["2027-10-01", "2026-10-31", "2027-09-30"])})
    f = fe.horizon_accrual(df, rd, he)
    assert f[0] == pytest.approx(364 / 365)        # annual coupon paid 1 day after: ~full year
    assert f[1] == pytest.approx(31 / 213)         # only the part after the report date
    assert f[2] == pytest.approx(1.0)


def test_forward_rate_simple_to_one_year_compounded_beyond():
    from labbank_common.curves import forward_rate
    z, z2 = 0.03, 0.05
    # 3M: money-market simple rate
    f3m = forward_rate(1.0, np.exp(-z * 0.25), 0.25)
    assert f3m == pytest.approx((np.exp(z * 0.25) - 1) / 0.25)
    # 10Y block: annually compounded -> e^z - 1, and a 2pp shock moves it by ~2pp (not ~2.9pp)
    f10 = forward_rate(1.0, np.exp(-z * 10), 10.0)
    f10s = forward_rate(1.0, np.exp(-z2 * 10), 10.0)
    assert f10 == pytest.approx(np.exp(z) - 1)
    assert f10s - f10 == pytest.approx(0.0208, abs=0.001)


def test_block_runs_over_full_index_tenor_even_when_cut_at_the_horizon():
    # only the first two coupons of a 3M block are loaded (the third starts after the horizon)
    df = _rows().iloc[:2].assign(fixing_freq="3M")
    s, e, yf = fe.block_periods(df)
    # start + 3M, not 1 Dec; 1 Jan 2027 is a TARGET holiday -> Modified Following to Mon 4 Jan
    assert e.iloc[0] == pd.Timestamp("2027-01-04")
    assert yf.iloc[0] == pytest.approx(95 / 360)


def test_modified_following_matches_quantlib_and_stays_in_the_month():
    import QuantLib as ql
    d = pd.to_datetime(["2027-01-01", "2026-10-31", "2027-04-02", "2026-12-26"])  # holiday, Sat month end, Good Friday, Boxing Day
    got = fe.modified_following(d, ["EUR"] * 4)
    want = [pd.Timestamp(ql.TARGET().adjust(ql.Date.from_date(x), ql.ModifiedFollowing).to_date()) for x in d]
    assert list(pd.to_datetime(got)) == want
    assert pd.Timestamp(got[1]) == pd.Timestamp("2026-10-30")       # rolls back, not into November
