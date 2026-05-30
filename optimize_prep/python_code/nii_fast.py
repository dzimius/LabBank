"""nii_fast.py
=============
Fast NII computation for the optimization hot path.

Per-product NII model
---------------------
Base NII:
    NII = amounts @ nii_unit_rate
    Calibrated from the full IRRBB pipeline — exact at calibration point.

Delta NII under a shocked scenario:
    Cohort products (1000-5000, all rate types):
        Uses the aggregated monthly CF schedule stored in cohort_outstanding_m.
        For each calendar month m (0..11):
            locked months (m < t_first_int):  delta = 0 (rate already fixed)
            floating months (m >= t_first_int):
                fixing event k covers months [t_first + k*F, t_first + (k+1)*F)
                rate_k = F-month forward rate at event time from disc curve
                delta_m = outstanding_frac[m] × clip(base_k + coeff_a×Δrate_k, floor, cap)
                          − outstanding_frac[m] × base_k
        delta NII = amounts × sign × Σ_m delta_m / 12

    Single-row / behavioural products (is_cohort=False):
        amounts @ delta_nii_unit[:, s]
        Calibrated from the exact IRRBB pipeline per scenario.

Time: O(12 × n_groups) per scenario — microseconds.
"""
from __future__ import annotations

import numpy as np
from bs_vector import BalanceSheetParams, CurveTensors

HORIZON_YF = 1.0
_HORIZON_M = 12.0


# ─────────────────────────────────────────────────────────────────────────────
# Schedule-based delta NII helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f_fwd_rate(disc: np.ndarray, event_m: float, F: float, disc_len: int) -> float:
    """Annualised F-month forward rate at event_m months from disc curve.

    disc[i] = discount factor at month i+1 (0-indexed).
    """
    _EPS = 1e-15
    idx_s = max(0, min(int(round(event_m)) - 1, disc_len - 1))
    idx_e = max(0, min(int(round(event_m + F)) - 1, disc_len - 1))
    return (disc[idx_s] / max(disc[idx_e], _EPS) - 1.0) * 12.0 / F


def _cohort_delta_nii_sched(
    amounts: np.ndarray,         # (m,) PLN balance
    outstanding_m: np.ndarray,   # (m, 12) normalised outstanding per month
    t_first_int: int,            # months to first repricing (locked months: 0..t_first_int-1)
    F: float,                    # fixing frequency in months
    coeff_a: np.ndarray,         # (m,)
    coeff_b: np.ndarray,         # (m,) spread / margin added to forward rate
    floor_: np.ndarray,          # (m,)
    cap_: np.ndarray,            # (m,)
    sign: np.ndarray,            # (m,)
    disc_shocked: np.ndarray,    # (disc_len,)
    disc_base: np.ndarray,       # (disc_len,)
) -> np.ndarray:
    """Delta NII (PLN) for cohorts sharing the same (F, t_first_int).

    Locked months (0 .. t_first_int-1): rate is fixed → delta = 0.
    Floating months (t_first_int .. 11): stable-margin formula applied to both
        base and shocked curves so floor/cap clips at the correct client-rate level.
        base_client  = clip(ca * fwd_base  + cb, floor, cap)
        shock_client = clip(ca * fwd_shock + cb, floor, cap)
        delta        = shock_client - base_client
    """
    disc_len = len(disc_base)
    F = max(F, 1.0)
    delta_w = np.zeros(len(amounts))  # weighted rate delta per cohort

    for m in range(t_first_int, 12):
        k = int((m - t_first_int) / F)
        event_t = t_first_int + k * F

        base_fwd    = _f_fwd_rate(disc_base,    event_t, F, disc_len)
        shocked_fwd = _f_fwd_rate(disc_shocked, event_t, F, disc_len)

        base_client    = np.clip(coeff_a * base_fwd    + coeff_b, floor_, cap_)
        shocked_client = np.clip(coeff_a * shocked_fwd + coeff_b, floor_, cap_)
        delta_r = shocked_client - base_client

        delta_w += outstanding_m[:, m] * delta_r / 12.0

    return amounts * delta_w * sign


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_nii(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    curves: CurveTensors,
    horizon_yf: float = HORIZON_YF,
    currency: str = "PLN",
) -> float:
    """NII for a single base scenario (base curves == shocked curves).

    Parameters
    ----------
    amounts  : float64 (n,) -- PLN balance per product
    params   : BalanceSheetParams
    curves   : CurveTensors (base disc curves are used for both base and shocked)
    horizon_yf : NII horizon in year fractions (default 1Y)

    Returns scalar NII in PLN.
    """
    return float(np.dot(amounts, params.nii_unit_rate))


def compute_nii_scenarios(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    curves: CurveTensors,
    horizon_yf: float = HORIZON_YF,
    currency: str = "PLN",
) -> dict[str, float]:
    """NII for base and all shocked scenarios.

    All cohort products (fixed and floating) use the schedule-based delta NII:
        - months < t_first_int : locked rate, delta = 0
        - months >= t_first_int: F-month forward rate from disc curve

    Single-row and behavioural products (is_cohort=False) use calibrated
    delta_nii_unit from the IRRBB pipeline.

    Returns dict: 'base' -> NII in PLN, scenario_id -> delta_NII in PLN.
    """
    horizon_m = int(round(horizon_yf * _HORIZON_M))   # 12
    base_disc = curves.get_disc_curve("base", currency)

    base_nii  = float(np.dot(amounts, params.nii_unit_rate))

    # Single-row products: calibrated delta (no schedule)
    mask_sng = ~params.is_cohort

    # Floor / cap arrays (−∞ / +∞ where not specified)
    floor_arr = np.where(np.isfinite(params.client_floor), params.client_floor, -np.inf)
    cap_arr   = np.where(np.isfinite(params.client_cap),   params.client_cap,    np.inf)

    # ── Pre-group cohort products by (F, t_first_int) — scenario-independent ─
    cohort_groups: dict[tuple, np.ndarray] = {}   # (F, t1_int) -> global gidx

    # Cohort products skipped by the schedule model (t1 >= horizon) fall back
    # to calibrated delta_nii_unit — same as single-row products.  This covers
    # fixed-coupon bonds (3100) and NMD deposits (7060/5000) where t_first=999.
    mask_locked = np.zeros(len(amounts), dtype=bool)

    if params.is_cohort.any():
        # t_first_int clamped to [1, 999]; groups with t1 >= horizon_m are
        # entirely locked within the horizon — use calibrated fallback.
        t1_int_arr = np.clip(
            np.round(params.cohort_t_first_m).astype(int), 1, 999
        )
        mask_locked = params.is_cohort & (t1_int_arr >= horizon_m)
        for F in np.unique(params.repricing_tenor_m[params.is_cohort]):
            mask_F  = params.is_cohort & (params.repricing_tenor_m == F)
            gidx_F  = np.where(mask_F)[0]
            t1_F    = t1_int_arr[mask_F]
            for t1 in np.unique(t1_F):
                if t1 >= horizon_m:
                    continue          # handled by calibrated fallback
                sub  = t1_F == t1
                gidx = gidx_F[sub]
                cohort_groups[(F, int(t1))] = gidx

    # Combined mask: single-row/behavioural + cohorts locked past horizon
    mask_calib = mask_sng | mask_locked

    result: dict[str, float] = {"base": base_nii}

    for s_idx, scen in enumerate(params.scenario_ids):
        scen = str(scen)
        shocked_disc = curves.get_disc_curve(scen, currency)

        # Calibrated delta for single-row/behavioural and locked-out cohorts
        delta_sng = float(
            np.dot(amounts[mask_calib], params.delta_nii_unit[mask_calib, s_idx])
        )

        # Schedule-based delta for all cohort products
        delta_coh = 0.0
        for (F, t1), gidx in cohort_groups.items():
            d = _cohort_delta_nii_sched(
                amounts[gidx],
                params.cohort_outstanding_m[gidx, :],
                t1, F,
                params.coeff_a[gidx],
                params.coeff_b[gidx],
                floor_arr[gidx],
                cap_arr[gidx],
                params.sign[gidx],
                shocked_disc, base_disc,
            )
            delta_coh += float(d.sum())

        result[scen] = delta_sng + delta_coh

    return result


def compute_sot_nii_pct(delta_nii: float, tier1_capital: float) -> float:
    """EBA SOT NII metric: delta_NII / Tier1 capital x 100.  Threshold: > -5%."""
    return delta_nii / tier1_capital * 100.0 if tier1_capital > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def compute_nii_base_fast(amounts: np.ndarray, params) -> float:
    """Base NII = amounts @ nii_unit_rate."""
    return float(np.dot(amounts, params.nii_unit_rate))


def compute_delta_nii_fast(amounts: np.ndarray, params, scenario_id: str) -> float:
    """delta_NII for one calibrated scenario (fixed/single-row only, no curves)."""
    s = params.scenario_index(scenario_id)
    return float(np.dot(amounts, params.delta_nii_unit[:, s]))


def compute_nii_all_scenarios(
    amounts: np.ndarray,
    params,
    curves=None,
    horizon_yf: float = HORIZON_YF,
    currency: str = "PLN",
) -> dict[str, float]:
    """NII for all scenarios.

    Uses schedule-based delta NII for cohort products when curves is provided.
    Falls back to calibrated delta_nii_unit for all products when curves is None.
    """
    if curves is None or not params.is_cohort.any():
        base = float(np.dot(amounts, params.nii_unit_rate))
        result: dict[str, float] = {"base": base}
        for s_idx, scen in enumerate(params.scenario_ids):
            result[str(scen)] = float(np.dot(amounts, params.delta_nii_unit[:, s_idx]))
        return result
    return compute_nii_scenarios(amounts, params, curves, horizon_yf, currency)


def compute_nii_schedule_base_by_product(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    curves: CurveTensors,
    currency: str = "PLN",
) -> dict[tuple, float]:
    """Base NII per (product_code, bs_side, currency) using the schedule formula.

    Cohort products: computes Σ_m outstanding_m × clip(ca×fwd_b+cb, fl, cp) × sign / 12
    using the same stable-margin formula as the delta computation.  The result
    will differ from the calibrated nii_unit_rate for products where the locked
    rate history diverges from the current forward curve.

    Single-row / behavioural products: falls back to calibrated nii_unit_rate
    (no CF schedule available for these products).

    Returns dict: (product_code, bs_side, currency) → base NII in PLN.
    The difference  exact_base − schedule_base  is the per-product bias that can
    be applied to shift shocked-scenario NII estimates.
    """
    base_disc = curves.get_disc_curve("base", currency)
    disc_len  = len(base_disc)
    floor_arr = np.where(np.isfinite(params.client_floor), params.client_floor, -np.inf)
    cap_arr   = np.where(np.isfinite(params.client_cap),   params.client_cap,    np.inf)

    result: dict[tuple, float] = {}

    for i in range(len(amounts)):
        k   = (str(params.product_code[i]), str(params.bs_side[i]), str(params.currency[i]))
        bal = float(amounts[i])
        if bal == 0.0:
            result.setdefault(k, 0.0)
            continue

        if not params.is_cohort[i]:
            result[k] = result.get(k, 0.0) + bal * float(params.nii_unit_rate[i])
            continue

        sign_i = float(params.sign[i])
        t1     = int(np.clip(round(params.cohort_t_first_m[i]), 1, 999))
        F      = max(float(params.repricing_tenor_m[i]), 1.0)
        ca     = float(params.coeff_a[i])
        cb     = float(params.coeff_b[i])
        fl     = float(floor_arr[i])
        cp     = float(cap_arr[i])
        lk_rt  = float(params.cohort_locked_rate[i])

        w = 0.0
        for m in range(12):
            out_m = float(params.cohort_outstanding_m[i, m])
            if out_m == 0.0:
                continue
            if m < t1:
                rate = lk_rt
            else:
                kk   = int((m - t1) / F)
                ev_t = int(t1 + kk * F)
                fwd  = _f_fwd_rate(base_disc, ev_t, F, disc_len)
                rate = float(np.clip(ca * fwd + cb, fl, cp))
            w += out_m * rate / 12.0

        result[k] = result.get(k, 0.0) + bal * sign_i * w

    return result
