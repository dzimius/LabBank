"""hedge_export.py
==================
Turn an optimizer's chosen swap-ladder notional into an ACTIONABLE deal list
-- one row per active bucket, in the exact same schema as
ir_derivatives/input/irs_input.xlsx, so it can be handed to a trading desk
or fed straight back into the IRS booking pipeline.

Restores what bs_optimization/output/optimal_hedge_swap_input.xlsx used to
give (a concrete "here's what to trade" file) -- that file's generator
script no longer exists in the codebase and its numbers predate this
session's fixes (IRS_MARGIN_BPS, seasoned_notional_cap, carry removed from
the objective), so it was stale. This rebuilds the same idea from the
CURRENT (fixed) optimizer result instead of a single stale row.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import swap_ladder as sl

_HERE    = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.normpath(os.path.join(_HERE, "..", "output"))

# Booking conventions -- fixed for every deal in this book (not something the
# optimizer chooses), copied from the schema the old optimal_hedge_swap_input.xlsx
# used (itself matching ir_derivatives/input/irs_input.xlsx's columns).
_FIXED_PAY_FREQ    = "3M"
_FIXED_DC_CONV     = "ACT/ACT"
_FIXED_B_DAY_CONV  = "ModifiedFollowing"
_FLOAT_PAY_FREQ    = "3M"
_FLOAT_FIXING_FREQ = "3M"
_FLOAT_SPREAD      = 0.0
_FLOAT_DC_CONV     = "ACT/365"
_FLOAT_B_DAY_CONV  = "ModifiedFollowing"

_COLUMNS = [
    "report_date", "swap_id", "swap_type", "pay_fixed", "notional", "currency",
    "start_date", "maturity_date", "fixed_rate", "fixed_pay_freq", "fixed_dc_conv",
    "fixed_b_day_conv", "float_rate_index", "float_pay_freq", "float_fixing_freq",
    "float_spread", "float_dc_conv", "float_b_day_conv", "disc_curve", "fwd_curve",
]


def build_hedge_deals(
    swap_notional: dict[str, float],
    ladder:        dict,
    direction:     float,
    currency:      str = sl.CURRENCY,
    report_date:   pd.Timestamp = sl.REPORT_DATE,
    min_notional:  float = 1.0,
) -> pd.DataFrame:
    """One row per active bucket, in ir_derivatives/input/irs_input.xlsx's schema.

    Parameters
    ----------
    swap_notional : bucket_id -> chosen notional (PLN), e.g. JointResult.swap_notional
    ladder         : the ladder dict with bucket_ids/elapsed_m/tenor_years/fixed_rate
                     aligned by position (e.g. JointResult.ladder)
    direction      : +1.0 = pay-fixed/receive-float, -1.0 = receive-fixed/pay-float
    """
    bucket_idx = {b: i for i, b in enumerate(ladder["bucket_ids"])}
    pay_fixed  = 1 if direction > 0 else 0

    rows = []
    for bucket_id, notional in swap_notional.items():
        if notional <= min_notional:
            continue
        i = bucket_idx[bucket_id]
        elapsed_m   = int(ladder["elapsed_m"][i])
        tenor_years = float(ladder["tenor_years"][i])
        fixed_rate  = float(ladder["fixed_rate"][i])

        start_date    = report_date - pd.DateOffset(months=elapsed_m)
        maturity_date = start_date + pd.DateOffset(years=int(round(tenor_years)))

        rows.append({
            "report_date":       report_date,
            "swap_id":           f"IRS_HEDGE_OPT_{bucket_id}",
            "swap_type":         "IRS",
            "pay_fixed":         pay_fixed,
            "notional":          round(float(notional), 2),
            "currency":          currency,
            "start_date":        start_date,
            "maturity_date":     maturity_date,
            "fixed_rate":        fixed_rate,
            "fixed_pay_freq":    _FIXED_PAY_FREQ,
            "fixed_dc_conv":     _FIXED_DC_CONV,
            "fixed_b_day_conv":  _FIXED_B_DAY_CONV,
            "float_rate_index":  f"{currency}_ASK_3M",
            "float_pay_freq":    _FLOAT_PAY_FREQ,
            "float_fixing_freq": _FLOAT_FIXING_FREQ,
            "float_spread":      _FLOAT_SPREAD,
            "float_dc_conv":     _FLOAT_DC_CONV,
            "float_b_day_conv":  _FLOAT_B_DAY_CONV,
            "disc_curve":        f"{currency}_disc_curve",
            "fwd_curve":         f"{currency}_fwd_curve",
        })

    return pd.DataFrame(rows, columns=_COLUMNS).sort_values("notional", ascending=False).reset_index(drop=True)


def save_hedge_deals(deals: pd.DataFrame, path: str | None = None) -> str:
    if path is None:
        path = os.path.join(_OUT_DIR, "optimal_hedge_swap_input.xlsx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        deals.to_excel(writer, sheet_name="Sheet1", index=False)
    print(f"Optimal hedge deal list saved: {path}  ({len(deals)} swap(s), "
          f"total notional {deals['notional'].sum()/1e6:,.1f}M)" if len(deals) > 0
          else f"Optimal hedge deal list saved: {path}  (0 swaps -- no new hedge needed)")
    return path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, _HERE)
    _OPTPREP = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "python_code"))
    if _OPTPREP not in sys.path:
        sys.path.insert(0, _OPTPREP)

    from bs_vector import BalanceSheetParams, CohortRates
    from bs_optimizer import OptimizationConfig
    from optimizer_io import load_optimizer_config
    from joint_optimizer import optimize_bs_and_ladder

    _PARAMS = os.path.normpath(os.path.join(_OPTPREP, "..", "output", "product_params.npz"))
    _XL     = os.path.normpath(os.path.join(_HERE, "..", "input", "optimizer_config.xlsx"))

    params = BalanceSheetParams.load(_PARAMS)
    cr     = CohortRates.load(_PARAMS)
    cfg    = load_optimizer_config(_XL, params)

    result = optimize_bs_and_ladder(cfg, params, cr)
    deals  = build_hedge_deals(result.swap_notional, result.ladder, result.swap_direction)
    save_hedge_deals(deals)
