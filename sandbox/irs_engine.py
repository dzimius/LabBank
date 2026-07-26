"""irs_engine.py
===============
Analytical NII / EVE metrics for the user-edited IRS book.

Conventions match the optimize_prep fast-metric framework:
  - sign = +1 for receive-fixed (pay_fixed=False)
  - sign = -1 for pay-fixed    (pay_fixed=True)
  - NII: accrual basis, 12-month horizon, not discounted
  - EVE: PV of all remaining net cash flows at market rates

Payment schedule:
  - Fixed leg : annual coupon at fixed_rate  (default freq 12M)
  - Float leg : quarterly coupon at fwd rate (default freq 3M)

Both schedules are projected on the monthly grid of CurveTensors.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

REPORT_DATE_DEFAULT = date(2024, 12, 31)
EBA_SCENARIOS       = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn", "own"]
_DAY_FRAC           = 30.4375          # average days per month


def _maturity_months(maturity: object, report_date: date) -> float:
    """Remaining maturity in fractional months (≥ 0)."""
    if maturity is None:
        return 0.0
    if isinstance(maturity, pd.Timestamp):
        mat = maturity.date()
    elif isinstance(maturity, date):
        mat = maturity
    else:
        try:
            mat = pd.Timestamp(maturity).date()
        except Exception:
            return 0.0
    return max(0.0, (mat - report_date).days / _DAY_FRAC)


def _fwd_rate_period(disc: np.ndarray, t_start: int, t_end: int) -> float:
    """Annualised forward rate for the period [t_start, t_end] months.

    t_start and t_end are month indices (1-based: month 1 = first month).
    disc[m-1] = discount factor at end of month m.
    """
    period_yf = (t_end - t_start) / 12.0
    if period_yf <= 0:
        return 0.0
    d_start = disc[t_start - 1] if t_start > 0 else 1.0
    d_end   = disc[min(t_end - 1, len(disc) - 1)]
    return (d_start / max(d_end, 1e-15) - 1.0) / period_yf


def _nii_one(
    notional: float,
    fixed_rate: float,
    sign: float,
    T_months: float,
    base_fwd: np.ndarray,
    float_freq_m: int = 3,
) -> float:
    """NII over min(T, 12) months on an accrual basis.

    Fixed leg  : accrues continuously at fixed_rate.
    Float leg  : reprices every float_freq_m months using average fwd rates.
    Both are computed for the 12-month horizon only.
    """
    horizon = min(int(round(T_months)), 12)
    if horizon <= 0:
        return 0.0

    # fixed leg accrual
    fixed_income = notional * fixed_rate * horizon / 12.0

    # float leg: sum each repricing period within [0, horizon]
    float_cost = 0.0
    t = 0
    while t < horizon:
        t_end = min(t + float_freq_m, horizon, int(round(T_months)))
        period_m  = t_end - t
        if period_m <= 0:
            break
        avg_fwd = float(base_fwd[t:t_end].mean()) if t_end > t else 0.0
        float_cost += notional * avg_fwd * period_m / 12.0
        t = t_end

    return sign * (fixed_income - float_cost)


def _eve_one(
    notional: float,
    fixed_rate: float,
    sign: float,
    T_months: float,
    disc: np.ndarray,
    base_disc: np.ndarray,    # used only for float fwd rate if same as disc
    fixed_freq_m: int = 12,
    float_freq_m: int = 3,
) -> float:
    """Mark-to-market EVE using the proper annual/quarterly payment schedule.

    Fixed leg  : annual coupon payments at fixed_rate, discounted at disc curve.
    Float leg  : quarterly coupon payments at the forward rate implied by disc,
                 discounted at disc curve.
    """
    T = int(round(T_months))
    if T <= 0:
        return 0.0

    pv_fixed = 0.0
    # annual fixed payments (at months fixed_freq_m, 2*fixed_freq_m, …, T)
    t = fixed_freq_m
    while t <= T:
        t_idx    = min(t - 1, len(disc) - 1)
        # coupon covers last fixed_freq_m months (or partial at final period)
        prev_t   = t - fixed_freq_m
        period_m = min(fixed_freq_m, T - prev_t)
        year_frac = period_m / 12.0
        pv_fixed += notional * fixed_rate * year_frac * disc[t_idx]
        t += fixed_freq_m

    pv_float = 0.0
    # quarterly float payments
    t = float_freq_m
    while t <= T:
        t_idx    = min(t - 1, len(disc) - 1)
        t_start  = t - float_freq_m
        period_m = min(float_freq_m, T - t_start)
        year_frac = period_m / 12.0
        fwd = _fwd_rate_period(disc, t_start, t)
        pv_float += notional * fwd * year_frac * disc[t_idx]
        t += float_freq_m

    return sign * (pv_fixed - pv_float)


def compute_irs_metrics(
    irs_df: pd.DataFrame,
    curves,                         # CurveTensors from optimize_prep
    report_date: date | None = None,
    scenarios: list[str] | None = None,
    currency: str = "PLN",
) -> dict:
    """Aggregate NII / EVE metrics for every row in `irs_df`.

    Parameters
    ----------
    irs_df      : user-edited swap table. Required columns:
                  notional, pay_fixed, fixed_rate, maturity_date.
                  Expired swaps (maturity ≤ report_date) are skipped.
    curves      : CurveTensors loaded from curve_tensors.npz
    report_date : valuation date (default 2024-12-31)
    scenarios   : EBA scenario IDs to compute; default = all 7
    currency    : used for disc/fwd curve lookup

    Returns
    -------
    dict:
        "nii_base"  : float
        "eve_base"  : float
        "delta_nii" : dict[str, float]   scenario_id → ΔNII PLN
        "delta_eve" : dict[str, float]   scenario_id → ΔEVE PLN
    """
    if report_date is None:
        report_date = REPORT_DATE_DEFAULT
    if scenarios is None:
        scenarios = EBA_SCENARIOS

    base_disc = curves.get_disc_curve("base", currency)
    base_fwd  = curves.get_fwd_curve("base",  currency)

    shocked: dict[str, np.ndarray] = {
        s: curves.get_disc_curve(s, currency)
        for s in scenarios
    }
    shocked_fwd: dict[str, np.ndarray] = {
        s: curves.get_fwd_curve(s, currency)
        for s in scenarios
    }

    total_nii = 0.0
    total_eve = 0.0
    d_nii     = {s: 0.0 for s in scenarios}
    d_eve     = {s: 0.0 for s in scenarios}

    for _, row in irs_df.iterrows():
        notional   = float(row.get("notional") or 0)
        fixed_rate = float(row.get("fixed_rate") or 0)
        pay_fixed  = bool(row.get("pay_fixed", False))

        if notional <= 0:
            continue

        T_frac = _maturity_months(row.get("maturity_date"), report_date)
        if T_frac <= 0:
            continue

        sign = -1.0 if pay_fixed else 1.0

        nii_base = _nii_one(notional, fixed_rate, sign, T_frac, base_fwd)
        eve_base = _eve_one(notional, fixed_rate, sign, T_frac, base_disc, base_disc)

        total_nii += nii_base
        total_eve += eve_base

        for s in scenarios:
            nii_s = _nii_one(notional, fixed_rate, sign, T_frac, shocked_fwd[s])
            eve_s = _eve_one(notional, fixed_rate, sign, T_frac, shocked[s], shocked[s])
            d_nii[s] += nii_s - nii_base
            d_eve[s] += eve_s - eve_base

    return {
        "nii_base":  total_nii,
        "eve_base":  total_eve,
        "delta_nii": d_nii,
        "delta_eve": d_eve,
    }


def extract_npz_irs_metrics(params) -> dict:
    """Pull the IRS contribution already embedded in the npz.

    Product code '0000' rows represent the two legs of the baseline swap book.
    These are the numbers that compute_all_metrics currently includes.
    We extract them so we can substitute the user-recomputed values.

    Returns same dict structure as compute_irs_metrics.
    """
    mask  = np.array([str(pc) == "0000" for pc in params.product_code])
    bal   = params.balance_arr[mask]
    scens = [str(s) for s in params.scenario_ids]

    nii_base = float(np.dot(bal, params.nii_unit_rate[mask]))
    eve_base = float(np.dot(bal, params.eve_pv_factor[mask]))

    d_nii = {}
    d_eve = {}
    for s_idx, s in enumerate(scens):
        d_nii[s] = float(np.dot(bal, params.delta_nii_unit[mask, s_idx]))
        d_eve[s] = float(np.dot(bal, params.delta_eve_unit[mask, s_idx]))

    return {"nii_base": nii_base, "eve_base": eve_base,
            "delta_nii": d_nii, "delta_eve": d_eve}
