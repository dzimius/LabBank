"""Shared pieces of the per-currency market-input builders (build_usd_inputs.py,
build_gbp_inputs.py): quote conventions, the output files and the validation against
the clearing-house settlement curve.

Quote convention of curve_input.xlsx / fixing_input.xlsx (same as PLN and EUR, read by
b_s_add_data_objects.build_curve_ql): percent; up to 1Y a simple ACT/365 rate; from 2Y an
annually compounded rate (the bootstrap turns it into the discount factor (1 + r)^-T).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SETTLEMENT = os.path.join(HERE, "clearing_curves", "settlement_eur_usd_gbp_pln_20261007.csv")
CURVE_TENORS = ["1D", "1M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y"]
YEARS = {"1D": 1 / 365, "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
         "4Y": 4.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0}


def quote_from_zero_cc(z_pct: pd.Series | float, tenor: str):
    """Continuously compounded zero (percent) -> project quote (percent)."""
    z, t = np.asarray(z_pct, dtype=float) / 100.0, YEARS[tenor]
    q = (np.exp(z * t) - 1.0) / t if t <= 1.0 else np.exp(z) - 1.0
    return q * 100.0


def settlement_quotes(curve_id: str, date: str = "2026-10-07") -> pd.Series:
    """Project-convention quotes (percent) at CURVE_TENORS from a settlement discount curve."""
    s = pd.read_csv(SETTLEMENT)
    s = s[(s["Curve ID"] == curve_id) & (s["Value Type"] == "S")]
    if s.empty:
        raise ValueError(f"{curve_id} not in {os.path.basename(SETTLEMENT)}")
    d0 = pd.Timestamp(date)
    days, df = s["Maturity Offset"].to_numpy(dtype=float), s["Value"].to_numpy(dtype=float)
    out = {}
    for t in CURVE_TENORS:
        n = 1 if t == "1D" else ((d0 + pd.DateOffset(months=int(round(YEARS[t] * 12)))) - d0).days
        dfi = np.exp(np.interp(n, days, np.log(df)))
        tau = n / 365.0
        out[t] = ((1.0 / dfi - 1.0) / tau if YEARS[t] <= 1.0 else dfi ** (-1.0 / tau) - 1.0) * 100.0
    return pd.Series(out)


def write_outputs(panel: pd.DataFrame, ccy: str, names: dict, fix_tenors: list[str], out_dir: str) -> None:
    """panel: daily, columns = CURVE_TENORS (percent). Same layouts as input/eur/*.xlsx."""
    fx = panel[fix_tenors].reset_index().melt(id_vars="date", var_name="tenor", value_name="rate")
    fx = pd.concat([fx.assign(rate_index=f"{ccy}_ASK_" + fx.tenor), fx.assign(rate_index=f"{ccy}_BID_" + fx.tenor)])
    fx["currency"] = ccy
    fx = fx[["date", "currency", "rate_index", "rate", "tenor"]].sort_values(["date", "rate_index"])
    fx.to_excel(os.path.join(out_dir, "fixing_input.xlsx"), index=False)
    cv = panel[CURVE_TENORS].reset_index().melt(id_vars="date", var_name="tenor", value_name="rate")
    cv["ir_type"] = cv.tenor.map(names)
    cv["currency"] = ccy
    cv = cv[["date", "currency", "ir_type", "rate", "tenor"]].sort_values(["date", "tenor"])
    cv.to_excel(os.path.join(out_dir, "curve_input.xlsx"), index=False)
    print(f"  fixing_input.xlsx {len(fx):,} rows | curve_input.xlsx {len(cv):,} rows")


def validate(panel: pd.DataFrame, curve_id: str, date: str = "2026-10-07") -> pd.DataFrame:
    ours = panel.loc[pd.Timestamp(date), CURVE_TENORS]
    ref = settlement_quotes(curve_id, date)
    v = pd.DataFrame({"ours_pct": ours, f"{curve_id}_pct": ref, "diff_bp": (ours - ref) * 100.0})
    print(f"  validation vs {curve_id} on {date}:")
    print(v.round(3).to_string())
    return v
