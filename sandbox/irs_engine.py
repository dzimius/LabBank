"""irs_engine.py
===============
Analytical NII / EVE metrics for the user-edited IRS book.

Conventions match the optimize_prep fast-metric framework:
  - sign = +1 for receive-fixed (pay_fixed=False)
  - sign = -1 for pay-fixed    (pay_fixed=True)
  - NII: accrual basis, 12-month horizon, not discounted
  - EVE: PV of all remaining net cash flows at market rates

Float-leg rate lock (current fixing)
------------------------------------
A swap that has already started has its current float coupon fixed at the
last reset. That coupon is locked -- identical in base and every shocked
scenario -- until the next reset date, and only then does the float leg
reprice to the (shocked) forward curve. The next reset comes from start_date
and the reset frequency (= index tenor: WIBOR 1M resets monthly, 6M
half-yearly), so a 1M leg passes a shock through to NII much sooner than a 6M
leg. This mirrors the full pipeline (ir_derivatives/irs_objects.py locks the
current period from historical fixings).

The locked rate itself is proxied by the base-curve forward over the locked
period: it is the same number in base and shocked runs, so it cancels out of
every ΔNII / ΔEVE -- only the lock's length matters for the SOT.

A forward-starting swap (start_date after the report date) has no flows before
its start; its first fixing is in the future, so it is not locked.

Schedules (monthly curve grid, fractional months):
  - NII : 12M horizon, accrual basis, not discounted
  - EVE : fixed leg annual coupons from the start (final stub at maturity);
          float leg = locked coupon paid at the next reset + par-floater
          N·(DF(next reset) − DF(T)) after it
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

REPORT_DATE_DEFAULT = date(2026, 6, 30)
EBA_SCENARIOS       = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn", "own"]
_DAY_FRAC           = 30.4375          # average days per month


def _freq_months(freq) -> float | None:
    """'3M' -> 3.0, '1Y' -> 12.0; None when unparseable."""
    if freq is None or (isinstance(freq, float) and np.isnan(freq)):
        return None
    f = str(freq).strip().upper()
    try:
        if f.endswith("M"):
            return float(f[:-1])
        if f.endswith("Y"):
            return float(f[:-1]) * 12.0
    except ValueError:
        return None
    return None


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


def _df_at(disc: np.ndarray, t: float) -> float:
    """Discount factor at fractional month t (log-linear between month nodes;
    disc[m-1] = DF at the end of month m, DF(0) = 1)."""
    if t <= 0:
        return 1.0
    grid = np.arange(0, len(disc) + 1, dtype=float)
    logd = np.concatenate(([0.0], np.log(np.maximum(disc, 1e-15))))
    return float(np.exp(np.interp(t, grid, logd)))


def _period_rate(disc: np.ndarray, t0: float, t1: float) -> float:
    """Simple annualised forward rate over [t0, t1] months."""
    yf = (t1 - t0) / 12.0
    if yf <= 0:
        return 0.0
    return (_df_at(disc, t0) / max(_df_at(disc, t1), 1e-15) - 1.0) / yf


def _float_lock(start_date, freq_m: float, T_months: float,
                report_date: date) -> tuple[float, float]:
    """(t_start, t_lock) in months from the report date.

    t_start : when the swap starts accruing (0 if it already started).
    t_lock  : end of the locked current float period = next reset date, capped
              at maturity. t_lock == t_start means no locked period (forward
              start, or no start date known -> legacy behaviour).
    """
    if start_date is None or (isinstance(start_date, float) and np.isnan(start_date)):
        return 0.0, 0.0
    try:
        sd = pd.Timestamp(start_date).date()
    except Exception:
        return 0.0, 0.0
    months_since = (report_date - sd).days / _DAY_FRAC
    if months_since < 0:                       # forward start
        t_start = min(-months_since, T_months)
        return t_start, t_start
    if freq_m <= 0:
        return 0.0, 0.0
    k = int(np.floor(months_since / freq_m + 1e-9))
    t_next = (k + 1) * freq_m - months_since
    return 0.0, float(min(t_next, T_months))


def _nii_one(
    notional: float,
    fixed_rate: float,
    sign: float,
    T_months: float,
    fwd: np.ndarray,
    t_start: float = 0.0,
    t_lock: float = 0.0,
    lock_rate: float = 0.0,
) -> float:
    """NII over the 12M horizon (or to maturity), accrual basis.

    Fixed leg  : accrues at fixed_rate from t_start.
    Float leg  : accrues at lock_rate over [t_start, t_lock] (the current,
                 already-fixed period), then at the monthly forwards `fwd`.
    """
    horizon = min(T_months, 12.0)
    if horizon <= t_start:
        return 0.0
    fixed_income = notional * fixed_rate * (horizon - t_start) / 12.0
    float_cost = 0.0
    for m in range(int(np.floor(t_start)), int(np.ceil(horizon))):
        a, b = max(float(m), t_start), min(float(m + 1), horizon)
        if b <= a:
            continue
        locked = max(0.0, min(b, t_lock) - a)
        float_cost += notional * (locked * lock_rate + (b - a - locked) * float(fwd[m])) / 12.0
    return sign * (fixed_income - float_cost)


def _eve_one(
    notional: float,
    fixed_rate: float,
    sign: float,
    T_months: float,
    disc: np.ndarray,
    t_start: float = 0.0,
    t_lock: float = 0.0,
    lock_rate: float = 0.0,
    fixed_freq_m: float = 12.0,
) -> float:
    """Mark-to-market value of the remaining flows.

    Fixed leg : annual coupons from t_start (final stub at maturity).
    Float leg : locked coupon N·lock_rate·τ paid at t_lock, then a par floater
                worth N·(DF(t_lock) − DF(T)) (resets to market thereafter).
    """
    T = float(T_months)
    if T <= t_start:
        return 0.0
    pv_fixed, t_prev = 0.0, t_start
    while t_prev < T - 1e-9:
        t_pay = min(t_prev + fixed_freq_m, T)
        pv_fixed += notional * fixed_rate * (t_pay - t_prev) / 12.0 * _df_at(disc, t_pay)
        t_prev = t_pay
    t_l = min(max(t_lock, t_start), T)
    pv_float = notional * lock_rate * (t_l - t_start) / 12.0 * _df_at(disc, t_l)
    pv_float += notional * (_df_at(disc, t_l) - _df_at(disc, T))
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
    report_date : valuation date (default: curves.report_date)
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
        _rd = getattr(curves, "report_date", None)
        report_date = pd.Timestamp(_rd).date() if _rd else REPORT_DATE_DEFAULT
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

        # reset frequency = index tenor (float_fixing_freq, else from the index name)
        freq_m = _freq_months(row.get("float_fixing_freq"))
        if freq_m is None:
            freq_m = _freq_months(str(row.get("float_rate_index") or "").rsplit("_", 1)[-1])
        if freq_m is None:
            freq_m = 3.0
        t_start, t_lock = _float_lock(row.get("start_date"), freq_m, T_frac, report_date)
        # current fixing: scenario-independent (base-curve proxy, cancels in Δ)
        lock = (t_start, t_lock, _period_rate(base_disc, t_start, t_lock))

        nii_base = _nii_one(notional, fixed_rate, sign, T_frac, base_fwd, *lock)
        eve_base = _eve_one(notional, fixed_rate, sign, T_frac, base_disc, *lock)

        total_nii += nii_base
        total_eve += eve_base

        for s in scenarios:
            nii_s = _nii_one(notional, fixed_rate, sign, T_frac, shocked_fwd[s], *lock)
            eve_s = _eve_one(notional, fixed_rate, sign, T_frac, shocked[s], *lock)
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
