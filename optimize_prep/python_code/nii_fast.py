"""nii_fast.py
=============
Fast NII computation for the optimization hot path.

Two levels of approximation
---------------------------

Level 1 — unit-rate linear model (default, fastest):
    NII(w) = Σ_i amounts[i] × nii_unit_rate[i]

    Where nii_unit_rate[i] = (nii_interest + nii_renewal) / balance at the
    current market conditions, extracted from irrbb.nii_results.

    This is exact at the current balance sheet and accurate near it.
    It treats each product's NII per PLN of balance as a constant (linear
    in amounts).  Time: O(n) dot product — microseconds.

Level 2 — repricing gap + renewal model (more accurate for large shifts):
    Approximates NII by separating the book into:
    a) Existing CFs within the horizon (earning eff_rate on average balance)
    b) Renewal of maturing/amortising capital at the shocked forward rate

    Activated by passing fwd_curve argument (360-element array from CurveTensors).

    For each product i:
        nii_interest[i] = amounts[i] × eff_rate × min(repricing_tenor/12, horizon) × (1 - amort/2)
        nii_renewal[i]  = amounts[i] × amort_frac × renewal_rate × avg_remain_yf

    where:
        eff_rate      = nii_unit_rate[i] / horizon_yf  (effective rate from base)
        renewal_rate  = max(floor, min(cap, a × fwd_curve[repricing_m - 1] + b))
        avg_remain_yf = (horizon_yf - repricing_tenor / 12) / 2, clipped ≥ 0

All functions accept and return plain floats (or per-currency dicts).
No pandas, no SQL in the hot path.
"""
from __future__ import annotations

import numpy as np
from bs_vector import BalanceSheetParams, CurveTensors

HORIZON_YF = 1.0   # 12-month NII horizon


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — linear unit-rate model
# ─────────────────────────────────────────────────────────────────────────────

def compute_nii_base_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> float:
    """Base NII at current market conditions (level 1).

    NII = Σ amounts[i] × nii_unit_rate[i]

    Parameters
    ----------
    amounts : float64 (n,) — PLN balance per product
    params  : BalanceSheetParams

    Returns
    -------
    Total NII in PLN.
    """
    return float(np.dot(amounts, params.nii_unit_rate))


def compute_delta_nii_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    scenario_id: str,
) -> float:
    """Delta NII for a shocked scenario (level 1).

    delta_NII = Σ amounts[i] × delta_nii_unit[i, s]
    where s = scenario index for scenario_id.

    Returns
    -------
    delta_NII in PLN (negative = loss).
    """
    s = params.scenario_index(scenario_id)
    return float(np.dot(amounts, params.delta_nii_unit[:, s]))


def compute_nii_shocked_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    scenario_id: str,
) -> float:
    """Absolute NII under shocked scenario (level 1).

    = NII_base + delta_NII(scenario)
    """
    base  = compute_nii_base_fast(amounts, params)
    delta = compute_delta_nii_fast(amounts, params, scenario_id)
    return base + delta


def compute_nii_all_scenarios(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> dict[str, float]:
    """Compute base NII and delta_NII for all scenarios.

    Returns dict:
        'base'    → base NII
        scenario_id → delta_NII for each shocked scenario
    """
    base = compute_nii_base_fast(amounts, params)
    # delta_nii_unit shape: (n, S)
    delta_vec = params.delta_nii_unit.T @ amounts   # (S,)
    result = {"base": base}
    for s_idx, scen in enumerate(params.scenario_ids):
        result[str(scen)] = float(delta_vec[s_idx])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — repricing gap + renewal model
# ─────────────────────────────────────────────────────────────────────────────

def _renewal_rate(
    repricing_m: np.ndarray,   # (n,) months to repricing
    coeff_a: np.ndarray,       # (n,)
    coeff_b: np.ndarray,       # (n,)
    client_floor: np.ndarray,  # (n,)
    client_cap: np.ndarray,    # (n,)
    fwd_curve: np.ndarray,     # (360,) annualised fwd rates
) -> np.ndarray:
    """Compute per-product client renewal rate using the fwd curve.

    renewal_rate[i] = clip(a[i] × fwd_curve[month[i] - 1] + b[i], floor, cap)

    For products with repricing_m > 360 (ultra-long fixed): uses last fwd rate.
    """
    m_idx = np.clip(repricing_m.astype(int) - 1, 0, len(fwd_curve) - 1)
    fwd   = fwd_curve[m_idx]
    rate  = coeff_a * fwd + coeff_b
    rate  = np.maximum(client_floor, rate)
    rate  = np.minimum(client_cap,   rate)
    return rate


def compute_nii_repricing_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    fwd_curve: np.ndarray,        # (360,) annualised fwd rates for this scenario/currency
    horizon_yf: float = HORIZON_YF,
    currency_filter: str | None = None,
) -> float:
    """NII using repricing gap + renewal model (level 2).

    Separates the book into:
    1. Pre-repricing interest: existing rate earned until repricing (or horizon, whichever first)
    2. Post-repricing renewal: maturing/repriced capital × market renewal rate × remaining horizon

    This is more accurate than the unit-rate model when the balance sheet
    weights shift significantly away from the current structure.

    Parameters
    ----------
    amounts        : float64 (n,) — PLN balance per product
    params         : BalanceSheetParams
    fwd_curve      : (360,) annualised fwd rate grid for the scenario and currency
    horizon_yf     : NII horizon in year fractions (default 1Y)
    currency_filter: if set, only includes products for that currency

    Returns
    -------
    Scalar NII in PLN.
    """
    mask = np.ones(len(amounts), dtype=bool)
    if currency_filter is not None:
        mask = params.currency == currency_filter

    a      = amounts * mask
    sign   = params.sign
    repric = params.repricing_tenor_m             # months to repricing
    repric_yf = np.clip(repric / 12.0, 0.0, horizon_yf)  # yf to repricing (within horizon)
    remain_yf = np.clip(horizon_yf - repric_yf, 0.0, None)  # remaining yf after repricing

    # eff_rate per unit from the base unit-rate (= NII/balance = eff_rate × horizon_yf approximately)
    eff_rate = np.where(horizon_yf > 0, params.nii_unit_rate / horizon_yf, 0.0)

    # Average balance during pre-repricing period (account for 50% amortisation simple approx)
    avg_balance_factor = 1.0 - params.amort_frac_1y * 0.5

    # Component 1: interest from existing positions (pre-repricing)
    nii_interest = a * eff_rate * repric_yf * avg_balance_factor * sign

    # Component 2: renewal interest on amortised/repriced capital
    renewal_rt = _renewal_rate(
        repric,
        params.coeff_a,
        params.coeff_b,
        params.client_floor,
        params.client_cap,
        fwd_curve,
    )
    # Amount that renews at market rate
    renewing_amount = a * params.amort_frac_1y
    # Average remain_yf for renewal (simple: from repricing to horizon end)
    avg_remain = remain_yf * 0.5   # assumes evenly distributed repayments in remain period
    nii_renewal = renewing_amount * renewal_rt * avg_remain * sign

    return float(np.sum(nii_interest + nii_renewal))


# ─────────────────────────────────────────────────────────────────────────────
# SOT delta-NII regulatory metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_sot_nii_pct(
    delta_nii: float,
    tier1_capital: float,
) -> float:
    """EBA SOT NII metric: delta_NII / Tier1 capital × 100.

    EBA threshold: must not be < -5%.
    """
    return delta_nii / tier1_capital * 100.0 if tier1_capital > 0 else float("nan")
