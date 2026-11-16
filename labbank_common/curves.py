"""curves.py
=========
Curve names of the multi-curve setup (see inputs/market/eur/build_curve_basis.py).

  <ccy>_disc_curve        discounting (EUR: STR OIS; a currency without a basis file: its one curve)
  <ccy>_fwd_curve         generic projection curve = the bootstrapped IBOR curve (EUR: EURIBOR 6M);
                          also used by IRRBB for renewals and forward-rate moves
  <ccy>_fwd_curve_<T>     projection curve of IBOR tenor T (1M, 3M, 1Y) where a basis exists

A floating coupon fixing every F months is projected on the tenor curve nearest to F
(the generic curve stands for 6M); fixed-rate periods longer than a year use the generic curve.
"""
from __future__ import annotations

import os

import numpy as np

GENERIC_TENOR_MONTHS = 6          # the bootstrapped curve is EURIBOR-6M-like

# Day basis of the money-market INDEX a forward is quoted on (EURIBOR / USD term rates
# ACT/360, WIBOR / SONIA ACT/365). A projected index rate must use it -- not the curve's
# zero-rate quoting basis -- since coupons accrue that same index rate on ACT/360 (EUR).
INDEX_DAY_BASIS = {"EUR": 360, "USD": 360, "PLN": 365, "GBP": 365}


def index_day_basis(ccy: str) -> int:
    return INDEX_DAY_BASIS.get(str(ccy)[:3].upper(), 365)


def _months(tenor: str) -> float | None:
    t = str(tenor).strip().upper()
    try:
        n = float(t[:-1])
    except ValueError:
        return None
    return {"D": n / 30.4375, "W": n / 4.345, "M": n, "Y": n * 12}.get(t[-1])


def projection_curve(fwd_curve: str, fixing_freq: str | None, available: set[str] | frozenset) -> str:
    """Curve to project a coupon that fixes every ``fixing_freq`` on (a name in ``available``)."""
    if not fixing_freq:
        return fwd_curve
    m = _months(fixing_freq)
    if m is None or m > 12:
        return fwd_curve
    cands = {GENERIC_TENOR_MONTHS: fwd_curve}
    prefix = fwd_curve + "_"
    for name in available:
        if name.startswith(prefix):
            tm = _months(name[len(prefix):])
            if tm is not None:
                cands[tm] = name
    best = min(cands, key=lambda k: (abs(k - m), k))
    return cands[best]


def projection_curve_map(currencies) -> dict[str, str]:
    """currency -> generic projection curve (IRRBB forwards / renewals)."""
    return {c: f"{c}_fwd_curve" for c in currencies}


def forward_rate(df_start, df_end, yf):
    """Index forward rate implied by two discount factors over a period of yf years.

    Up to 1 year: simple (money-market) rate (DF_s / DF_e - 1) / yf -- EURIBOR / WIBOR style.
    Longer periods (a 5Y or 10Y fixing block of a reset-fixed loan): annually compounded
    (DF_s / DF_e)^(1 / yf) - 1, the convention of the swap rates such fixings are set on.
    A simple rate over 10 years overstates both the level (~+0.5pp at 3%) and the
    sensitivity (~x1.46 at 3%) -- a 2pp shock moved the reset coupon by ~2.9pp.
    """
    ds = np.asarray(df_start, dtype=float)
    de = np.asarray(df_end, dtype=float)
    t = np.asarray(yf, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ds / de
        simple = (ratio - 1.0) / t
        comp = np.power(ratio, 1.0 / t) - 1.0
    return np.where(t > 1.0 + 1e-9, comp, simple)


# Constant-balance-sheet renewal (IRRBB NII): "tenor" = capital repaid inside the year renews
# like for like on the product's own repricing period -- a 10Y-fixed mortgage at the 10Y
# forward-starting par rate, a 6M term deposit at the 6M forward -- as new sales are priced;
# "horizon" = the simpler convention used before 2026-10-28 (forward from the repayment date
# to the horizon end, whatever the product). LABBANK_RENEWAL=horizon switches back.
RENEWAL_MODE = os.environ.get("LABBANK_RENEWAL", "tenor").strip().lower()


def renew_months(fixing_freq, rate_type, term_months) -> np.ndarray:
    """Repricing period (months) a renewed contract keeps its rate for: the fixing frequency
    ('10Y', '3M'; '1D' and shorter = 0), else for a fixed-rate contract without one (term
    deposits) its original term; NaN = reprices continuously (NMDs, overnight)."""
    ff = np.asarray(fixing_freq, dtype=object)
    rt = np.asarray(rate_type, dtype=object)
    tm = np.asarray(term_months, dtype=float) if term_months is not None else np.full(len(ff), np.nan)
    out = np.full(len(ff), np.nan)
    for i, f in enumerate(ff):
        m = _months(f) if f is not None and str(f).strip() not in ("", "None", "nan") else None
        if m is not None:
            out[i] = m if m >= 1 else np.nan
        elif str(rt[i]) == "F" and np.isfinite(tm[i]) and tm[i] >= 1:
            out[i] = tm[i]
    return out


def renewal_market_rate(cf_end, horizon_end, months, df_at, floor_rate: float = 0.0,
                        mode: str | None = None) -> np.ndarray:
    """Market rate earned until the horizon end by capital repaid at cf_end and renewed.

    A renewed contract that reprices again before the horizon (months/12 <= time left, or no
    repricing period) earns the chain of resets ~ the forward from cf_end to the horizon end.
    One that keeps its rate past the horizon earns the rate fixed at renewal: the forward
    over its period up to a year, the annual par rate of a forward-starting swap beyond
    (sum of annual DFs, the convention long fixings are set on). df_at(dates) -> DF array.
    Rows repaid at or after the horizon get 0."""
    import pandas as pd
    mode = (mode or RENEWAL_MODE)
    end = pd.to_datetime(pd.Series(np.asarray(cf_end))).reset_index(drop=True)
    hor = pd.Timestamp(horizon_end)
    remain = ((hor - end).dt.days / 365.0).to_numpy(dtype=float)
    need = remain > 0
    out = np.zeros(len(end))
    if not need.any():
        return out
    d_end = np.asarray(df_at(end), dtype=float)
    d_hor = np.asarray(df_at(pd.Series([hor] * len(end))), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_hor = np.where(need & (d_end > 0) & (d_hor > 0), (d_end / d_hor - 1.0) / remain, np.nan)
    rate = f_hor
    if mode == "tenor" and months is not None:
        mo = np.asarray(months, dtype=float)
        fix = need & np.isfinite(mo) & (mo / 12.0 > remain + 1e-9)
        for m in np.unique(mo[fix]):
            sel = fix & (mo == m)
            s = end[sel]
            if m <= 12:
                e = s + pd.DateOffset(months=int(m))
                d_e = np.asarray(df_at(e), dtype=float)
                yf = ((e - s).dt.days / 365.0).to_numpy(dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = (d_end[sel] / d_e - 1.0) / yf
            else:
                n = int(round(m / 12.0))
                ann = np.zeros(int(sel.sum()))
                for k in range(1, n + 1):
                    ann += np.asarray(df_at(s + pd.DateOffset(years=k)), dtype=float)
                d_n = np.asarray(df_at(s + pd.DateOffset(years=n)), dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = (d_end[sel] - d_n) / ann
            rate = rate.copy()
            rate[sel] = r
    bad = need & ~np.isfinite(rate)
    if bad.any():
        raise ValueError(f"NII renewal: {int(bad.sum())} renewing cash flow(s) without a discount factor "
                         "-- missing market data would be a silent 0% renewal rate")
    out[need] = np.maximum(floor_rate, rate[need])
    return out


def fixing_asof(fix_df, fixing_dates, rate_indices, max_lag_days: int = 10) -> np.ndarray:
    """Historical fixing of each (fixing_date, rate_index): the last published quote on or before
    the date. fix_df: Series MultiIndex (fixing_date, rate_index) -> rate (in % points).

    A fixing date can be a TARGET business day without a quote in the source (24 Dec, Ascension
    Day): an exact-date lookup gave NaN, the CF engine then priced the contract at no rate and
    wrote zero cash flows, so the contract dropped out of NII / EVE. A quote older than
    max_lag_days, or an index with no history at all, stays NaN."""
    import pandas as pd
    dates = pd.to_datetime(pd.Series(np.asarray(fixing_dates))).to_numpy()
    idxs = np.asarray(rate_indices, dtype=object)
    out = np.full(len(dates), np.nan)
    src = fix_df.reset_index()
    src.columns = ["fixing_date", "rate_index", "rate"]
    src["fixing_date"] = pd.to_datetime(src["fixing_date"])
    for idx_name, g in src.dropna(subset=["rate"]).groupby("rate_index"):
        m = idxs == idx_name
        if not m.any():
            continue
        g = g.sort_values("fixing_date")
        known = g["fixing_date"].to_numpy()
        pos = np.searchsorted(known, dates[m], side="right") - 1
        ok = pos >= 0
        val = np.full(m.sum(), np.nan)
        val[ok] = g["rate"].to_numpy(dtype=float)[pos[ok]]
        lag = np.full(m.sum(), np.inf)
        lag[ok] = (dates[m][ok] - known[pos[ok]]) / np.timedelta64(1, "D")
        val[lag > max_lag_days] = np.nan
        out[m] = val
    return out
