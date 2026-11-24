"""dated_swaps.py
================
Swaps of a roll-forward year on their own trade dates.

The optimiser runs on the plan book look at one date, the plan date (the rolled bank's report date). New
swaps are not traded on that date but spread over the year:

  "plan" dates     a new swap (one tenor, one floating leg) is bought on the dates of the plan's own new swaps,
                   split like their notionals. One optimiser variable = the whole strip; it is valued at the
                   plan date with every part already seasoned (remaining life, its own fixed rate, the floating
                   coupon in progress already fixed).
  monthly dates    one variable per trade month and tenor (monthly_swap_plan.py).

Rates come from the rolled market data of the bank: the fixed rate is the par quote "IRS <ccy> <n>Y" of the
trade date (+ a cost in bp), the floating coupon in progress the EURIBOR quote of its fixing date. Valuation is
multi-curve like the cash-flow engine: discounting on the OIS curve, EURIBOR 3M / 6M forwards on their own
curve = OIS zero rate + the tenor spread of curve_basis.csv; a rate shock moves both. Fixed leg annual 30/360,
floating leg ACT/360 (the coupon DF_start / DF_end - 1 on the projection curve). Months are days / 30.4375 from
the report date; discount factors log-linear between the monthly nodes of the curve tensors.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(_PY, "..", "..")))
from labbank_common.bank_config import get_bank, home_ccy  # noqa: E402

BANK = get_bank()
RD = pd.Timestamp(BANK.report_date)
CCY = home_ccy()
DPM = 30.4375


# ── market data ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _curve_quotes() -> pd.DataFrame:
    d = pd.read_excel(BANK.market_dir() / "curve_input.xlsx")
    return d.assign(date=pd.to_datetime(d["date"])).sort_values("date")


@lru_cache(maxsize=None)
def _fixings() -> pd.DataFrame:
    d = pd.read_excel(BANK.market_dir() / "fixing_input.xlsx")
    return d.assign(date=pd.to_datetime(d["date"])).sort_values("date")


@lru_cache(maxsize=None)
def par_quote(years: float, date) -> float:
    q = _curve_quotes()
    q = q[(q["ir_type"] == f"IRS {CCY} {years:g}Y") & (q["date"] <= pd.Timestamp(date))]
    if q.empty:
        raise ValueError(f"no quote IRS {CCY} {years:g}Y on {date}")
    return float(q["rate"].iloc[-1]) / 100.0


@lru_cache(maxsize=None)
def fixing(freq_m: int, date) -> float:
    q = _fixings()
    q = q[(q["rate_index"] == f"{CCY}_ASK_{freq_m}M") & (q["date"] <= pd.Timestamp(date))]
    if q.empty:
        raise ValueError(f"no fixing {CCY}_ASK_{freq_m}M on {date}")
    return float(q["rate"].iloc[-1]) / 100.0


@lru_cache(maxsize=None)
def _basis():
    p = BANK.market_dir() / "curve_basis.csv"
    if not p.exists():
        return None
    b = pd.read_csv(p)
    return {c: (g["n_days"].to_numpy(float), g["spread"].to_numpy(float)) for c, g in b.groupby("curve")}


def tenor_spread(freq_m: int, months) -> np.ndarray:
    """Zero-rate spread of the EURIBOR freq_m curve over the OIS curve (6M = the generic curve)."""
    b = _basis()
    m = np.asarray(months, float)
    if b is None:
        return np.zeros_like(m)
    key = f"{freq_m}M"
    own = np.interp(m * DPM, *b[key]) if (freq_m != 6 and key in b) else 0.0
    return own - np.interp(m * DPM, *b["disc"])


# ── a swap as cash-flow arrays seen from a valuation date ────────────────────
def _roll(start, n_months: int) -> pd.Timestamp:
    d = pd.Timestamp(start) + pd.DateOffset(months=int(n_months))
    b = d + pd.offsets.BusinessDay(0)                       # modified following (weekends)
    return b if b.month == d.month else d - pd.offsets.BusinessDay(1)


def make(start, maturity, freq_m: int, fixed_rate: float, weight: float = 1.0, float_ratio: float = 1.0,
         asof=None) -> dict | None:
    """Pay-fixed swap: remaining cash flows at `asof` (default report date). None = matured or not started.
    float_ratio = floating / fixed notional (1 for an IRS)."""
    asof = RD if asof is None else pd.Timestamp(asof)
    start, maturity = pd.Timestamp(start), pd.Timestamp(maturity)
    if maturity <= asof or start > asof:
        return None
    life = (maturity.year - start.year) * 12 + maturity.month - start.month

    def mon(d):
        return (d - asof).days / DPM

    fl = [start] + [maturity if k == life else _roll(start, k) for k in range(freq_m, life + 1, freq_m)]
    fx = [start] + [maturity if k == life else _roll(start, k) for k in range(12, life + 1, 12)]
    i = next(k for k in range(1, len(fl)) if fl[k] > asof)     # coupon in progress
    return dict(w=float(weight), k=float(fixed_rate), f=int(freq_m), ratio=float(float_ratio),
                start=str(start.date()), maturity=str(maturity.date()),
                fixed_m=[mon(d) for d in fx[1:] if d > asof],
                fix=fixing(freq_m, fl[i - 1]), fix_a=(fl[i] - fl[i - 1]).days / 360.0,
                reset_m=[mon(d) for d in fl[i:]])          # end of the coupon in progress, then every reset


def strip(trades: list, years: float, freq_m: int, cost_bp: float, asof=None) -> list:
    """Pay-fixed swap of `years` bought on several dates. trades: [(date, share)]."""
    out = []
    for d, w in trades:
        c = make(d, _roll(d, int(round(years * 12))), freq_m, par_quote(years, pd.Timestamp(d)) + cost_bp / 1e4, w,
                 asof=asof)
        if c is not None:
            out.append(c)
    return out


# ── pricing on a monthly discount curve ──────────────────────────────────────
def _df(disc: np.ndarray, months) -> np.ndarray:
    ln = np.concatenate([[0.0], np.log(np.maximum(disc, 1e-12))])
    return np.exp(np.interp(np.asarray(months, float), np.arange(len(ln), dtype=float), ln))


def _proj(disc, months, f):
    m = np.asarray(months, float)
    return _df(disc, m) * np.exp(-tenor_spread(f, m) * m / 12.0)


def eve(disc: np.ndarray, comps: list) -> float:
    """PV per unit notional of the strip (receive floating - pay fixed)."""
    tot = 0.0
    for c in comps:
        rs = np.array(c["reset_m"])
        d = _df(disc, rs)
        p = _proj(disc, rs, c["f"])
        fl = c["fix"] * c["fix_a"] * d[0] + float(np.sum((p[:-1] / p[1:] - 1.0) * d[1:]))
        tot += c["w"] * (c["ratio"] * fl - c["k"] * float(_df(disc, c["fixed_m"]).sum()))
    return tot


def float_nii(disc: np.ndarray, comps: list, horizon: float = 12.0) -> float:
    """Floating-leg interest accrued from the valuation date over `horizon` months, per unit notional."""
    tot = 0.0
    for c in comps:
        rs = np.array(c["reset_m"])
        x = c["fix"] * min(rs[0], horizon) * DPM / 360.0
        p = _proj(disc, rs, c["f"])
        for a, b, pa, pb in zip(rs[:-1], rs[1:], p[:-1], p[1:]):
            if a >= horizon:
                break
            x += (pa / pb - 1.0) * (min(b, horizon) - a) / (b - a)
        tot += c["w"] * c["ratio"] * x
    return tot


def fixed_nii(comps: list, horizon: float = 12.0) -> float:
    return sum(c["w"] * c["k"] * min(c["reset_m"][-1], horizon) / 12.0 for c in comps)


# ── interest inside the plan year (trade date -> plan date) ──────────────────
def window_interest(start, freq_m: int, fixed_rate: float, months, base_rd, float_ratio: float = 1.0) -> np.ndarray:
    """Net interest (receive floating - pay fixed) per unit notional in each month of the plan year
    (month ends `months`, the first month starts at base_rd), on the rolled fixings."""
    start = pd.Timestamp(start)
    edges = [pd.Timestamp(base_rd)] + [pd.Timestamp(d) for d in months]
    out = np.zeros(len(edges) - 1)
    k = 0
    while True:
        a, b = (_roll(start, k) if k else start), _roll(start, k + freq_m)
        if a >= edges[-1]:
            return out
        r = fixing(freq_m, a)
        for j in range(len(out)):
            days = (min(b, edges[j + 1]) - max(a, edges[j])).days
            if days > 0:
                out[j] += float_ratio * r * days / 360.0 - fixed_rate * days / 365.0
        k += freq_m


def planned_trades(engine, base_rd) -> list:
    """The plan's own new pay-fixed swaps (traded after base_rd), one dict per swap."""
    from sqlalchemy import text
    d = pd.read_sql_query(text(
        "SELECT swap_id, MIN(start_date) s, MAX(maturity_date) m, "
        "MAX(CASE WHEN leg_type='FIXED' THEN CAST(notional AS FLOAT) END) n, "
        "MAX(CASE WHEN leg_type='FLOAT' THEN CAST(notional AS FLOAT) END) nf, "
        "MAX(CASE WHEN leg_type='FIXED' THEN CAST(fixed_rate AS FLOAT) END) k, "
        "MAX(CASE WHEN leg_type='FLOAT' THEN fixing_freq END) f "
        "FROM schemat.ir_swaps WHERE report_date=:rd AND swap_type='IRS' AND start_date > :b "
        "GROUP BY swap_id ORDER BY MIN(start_date)"), engine, params={"rd": RD, "b": pd.Timestamp(base_rd)})
    return [dict(swap_id=r.swap_id, start=pd.Timestamp(r.s), maturity=pd.Timestamp(r.m), n=float(r.n),
                 nf=float(r.nf), k=float(r.k), f=int(str(r.f).rstrip("M"))) for r in d.itertuples()]
