"""Contract-level market values at any date of a market path (the monthly accounting loop).

The IRRBB EVE tables price the book at its report date only. The monthly P&L needs the fair value
of securities and swaps at every month-end of the roll-forward window, on that month's curve
(mkt.curves of the rolled bank holds a bootstrapped curve for every business day of the path).
This module prices from the contract terms directly:

  fixed securities   coupons client_rt x nominal x yf every payment_freq from the start date,
                     nominal at maturity; dirty = sum CF x DF_d(T) x exp(-s (T - d)/365);
                     clean = dirty - accrued coupon
  floating notes     clean = nominal x (1 + (margin - s) x annuity_s(d)), margin = client_rt - index_rt
                     (coupon resets at the index on the discount curve: a par floater plus the
                     margin over the market spread)
  IRS                fixed leg: periods every fixed_pay_freq back from maturity, yf by fixed_dc_conv;
                     float leg: every float_pay_freq; a period already started pays the forward fixed
                     on the curve of its start date (the path's realised fixing), later periods the
                     forward on the projection curve (fwd_curve by fixing tenor) of date d; both legs
                     discounted on disc_curve. MV = receive leg - pay leg, clean of accrued coupons.

Dates are unadjusted (no business-day roll): the check against the EVE-based values at the report
date (acct_valuation_workflow) measures what that costs (pnl_rollforward.valuation_check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

import acct_common as ac


def tenor_months(t) -> int:
    s = str(t).strip().upper()
    n = float(s[:-1])
    m = {"M": n, "Y": 12 * n}.get(s[-1])
    if m is None:
        if s[-1] == "D":
            return 0
        raise ValueError(f"tenor {t!r}")
    return int(round(m))


def year_frac(a: pd.Timestamp, b: pd.Timestamp, dc: str) -> float:
    dc = str(dc).upper()
    if dc.startswith("30"):
        return ((b.year - a.year) * 360 + (b.month - a.month) * 30 + (min(b.day, 30) - min(a.day, 30))) / 360.0
    days = (b - a).days
    return days / 360.0 if "360" in dc else days / 365.0


class PathCurves:
    """Bootstrapped curves of a database (mkt.curves), any curve date up to the last one."""

    def __init__(self, engine, names, start, end):
        q = text("""SELECT curve_date, curve_name, n_days, d_f FROM mkt.curves
                    WHERE curve_date >= :s AND curve_date <= :e""")
        c = pd.read_sql(q, engine, params={"s": pd.Timestamp(start).date(), "e": pd.Timestamp(end).date()})
        c = c[c["curve_name"].isin(set(names))]
        c["curve_date"] = pd.to_datetime(c["curve_date"])
        self.names = set(c["curve_name"])
        self.dates = np.array(sorted(c["curve_date"].unique()), dtype="datetime64[ns]")
        self._g = {k: (g["n_days"].to_numpy(float), g["d_f"].to_numpy(float))
                   for k, g in c.sort_values("n_days").groupby(["curve_name", "curve_date"])}

    def asof(self, d) -> pd.Timestamp:
        d = np.datetime64(pd.Timestamp(d))
        i = np.searchsorted(self.dates, d, side="right") - 1
        if i < 0:
            raise ValueError(f"no curve on or before {pd.Timestamp(d).date()}")
        return pd.Timestamp(self.dates[i])

    def df(self, name: str, d, targets) -> np.ndarray:
        """DF of the curve dated asof(d), at target dates, measured from d."""
        cd = self.asof(d)
        x, y = self._g[(name, cd)]
        t = (pd.to_datetime(pd.Series(targets)) - cd).dt.days.to_numpy(float)
        t0 = float((pd.Timestamp(d) - cd).days)
        return ac.interp_df(x, y, t) / ac.interp_df(x, y, np.array([t0]))[0]

    def fwd(self, name: str, d, a, b, dc="ACT/360") -> float:
        """Simple forward rate a -> b on the curve of date d."""
        da, db = self.df(name, d, [a, b])
        yf = year_frac(pd.Timestamp(a), pd.Timestamp(b), dc)
        return (da / db - 1.0) / yf if yf > 0 else 0.0


def _periods_forward(start, maturity, months: int) -> list[tuple]:
    start, maturity = pd.Timestamp(start), pd.Timestamp(maturity)
    if months <= 0:
        return [(start, maturity)]
    out, a, k = [], start, 1
    while True:
        b = start + pd.DateOffset(months=months * k)
        if b >= maturity - pd.Timedelta(days=5):
            out.append((a, maturity))
            return out
        out.append((a, b))
        a, k = b, k + 1


def _periods_back(start, maturity, months: int) -> list[tuple]:
    start, maturity = pd.Timestamp(start), pd.Timestamp(maturity)
    ends, k = [], 0
    while True:
        e = maturity - pd.DateOffset(months=months * k)
        if e <= start + pd.Timedelta(days=5):
            break
        ends.append(e)
        k += 1
    ends = sorted(ends)
    starts = [start] + ends[:-1]
    return list(zip(starts, ends))


def security_value(row, d, curves: PathCurves, spread_bp: float) -> tuple[float, float]:
    """(clean, accrued) of one security contract at date d (0, 0 once matured)."""
    d = pd.Timestamp(d)
    mat = pd.Timestamp(row.maturity_date)
    if mat <= d or pd.Timestamp(row.start_date) > d:
        return 0.0, 0.0
    n = float(row.balance_amt)
    s = spread_bp / 1e4
    per = _periods_forward(row.start_date, mat, tenor_months(row.payment_freq))
    per = [(a, b) for a, b in per if b > d]
    ends = [b for _, b in per]
    dfs = curves.df(row.disc_curve, d, ends) * np.exp(-s * (pd.to_datetime(pd.Series(ends)) - d).dt.days.to_numpy() / 365.0)
    yfs = np.array([year_frac(a, b, row.dc_conv) for a, b in per])
    a0, b0 = per[0]
    frac = max(0.0, (d - a0).days) / max(1, (b0 - a0).days)
    if str(row.rate_type).upper() == "V":
        margin = float(row.client_rt) - float(row.index_rt if pd.notna(row.index_rt) else 0.0)
        return n * (1.0 + (margin - s) * float(np.sum(yfs * dfs))), 0.0
    cpn = float(row.client_rt) * n * yfs
    dirty = float(np.sum(cpn * dfs) + n * dfs[-1])
    acc = float(cpn[0] * frac)
    return dirty - acc, acc


def swap_value(sw, d, curves: PathCurves) -> tuple[float, float]:
    """(clean MV, accrued net coupon) of one single-currency swap at d, from the bank's side."""
    d = pd.Timestamp(d)
    mat = pd.Timestamp(sw.maturity_date)
    if mat <= d:
        return 0.0, 0.0
    n = float(sw.notional)
    disc = sw.disc_curve
    proj = ac_projection(sw.fwd_curve, sw.float_fixing_freq, curves.names)
    # fixed leg
    fp = [(a, b) for a, b in _periods_back(sw.start_date, mat, tenor_months(sw.fixed_pay_freq)) if b > d]
    fyf = np.array([year_frac(a, b, sw.fixed_dc_conv) for a, b in fp])
    fdf = curves.df(disc, d, [b for _, b in fp])
    fix_pv = float(np.sum(n * float(sw.fixed_rate) * fyf * fdf))
    a0, b0 = fp[0]
    fix_acc = n * float(sw.fixed_rate) * fyf[0] * max(0.0, (d - a0).days) / max(1, (b0 - a0).days) if a0 < d else 0.0
    # float leg
    lp = [(a, b) for a, b in _periods_back(sw.start_date, mat, tenor_months(sw.float_pay_freq)) if b > d]
    spread = float(sw.float_spread) if pd.notna(sw.float_spread) else 0.0
    rates, yfs = [], []
    for a, b in lp:
        on = a if a <= d else d                 # started: fixed on its start date's curve
        rates.append(curves.fwd(proj, on, a, b, sw.float_dc_conv) + spread)
        yfs.append(year_frac(a, b, sw.float_dc_conv))
    rates, yfs = np.array(rates), np.array(yfs)
    ldf = curves.df(disc, d, [b for _, b in lp])
    flt_pv = float(np.sum(n * rates * yfs * ldf))
    a1, b1 = lp[0]
    flt_acc = n * rates[0] * yfs[0] * max(0.0, (d - a1).days) / max(1, (b1 - a1).days) if a1 < d else 0.0
    sign = 1.0 if int(sw.pay_fixed) == 1 else -1.0       # pay fixed: receive float
    dirty = sign * (flt_pv - fix_pv)
    acc = sign * (flt_acc - fix_acc)
    return dirty - acc, acc


def ac_projection(fwd_curve, fixing_freq, names):
    from labbank_common.curves import projection_curve
    return projection_curve(fwd_curve, fixing_freq, frozenset(names))
