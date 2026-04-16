"""eve_fast.py
=============
Fast EVE and delta-EVE computation for the optimization hot path.

Two levels of approximation
---------------------------

Level 1 — unit-rate linear model (default, fastest):
    EVE_base(w) = Σ amounts[i] × eve_pv_factor[i]
    delta_EVE(w, scenario) = Σ amounts[i] × delta_eve_unit[i, s]

    eve_pv_factor[i] and delta_eve_unit[i, s] are extracted from the exact
    EVE pipeline (irrbb.eve_results).  This makes the approximation exact
    at the current balance sheet and linearly accurate near it.

    The approximation assumes each product's EVE per PLN is independent
    of the total allocation — valid when the product mix within a
    product_code class stays constant.

Level 2 — modified duration model:
    delta_EVE[i] ≈ -D_mod[i] × amounts[i] × delta_r_parallel

    D_mod is extracted from the exact EVE par_up/par_dn results:
        D_mod[i] = -(PV_par_dn[i] - PV_par_up[i]) / (2 × Δr_par × balance[i])

    Provides a smooth, differentiable delta_EVE for the optimizer's
    gradient computation without requiring a new scenario lookup.
    Suitable as a cheap inner-loop gradient; use level 1 for objective
    evaluations to stay close to the exact pipeline.

All functions accept pure numpy arrays — no pandas in the hot path.
"""
from __future__ import annotations

import numpy as np
from bs_vector import BalanceSheetParams, CurveTensors


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — unit-rate linear model
# ─────────────────────────────────────────────────────────────────────────────

def compute_eve_base_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> float:
    """Base EVE (PV of existing book) — level 1.

    EVE_base = Σ amounts[i] × eve_pv_factor[i]

    Parameters
    ----------
    amounts : float64 (n,)
    params  : BalanceSheetParams

    Returns
    -------
    EVE in PLN.
    """
    return float(np.dot(amounts, params.eve_pv_factor))


def compute_delta_eve_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    scenario_id: str,
) -> float:
    """delta_EVE for a shocked scenario — level 1.

    delta_EVE = Σ amounts[i] × delta_eve_unit[i, s]

    Returns
    -------
    delta_EVE in PLN (negative = loss under shock).
    """
    s = params.scenario_index(scenario_id)
    return float(np.dot(amounts, params.delta_eve_unit[:, s]))


def compute_eve_shocked_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    scenario_id: str,
) -> float:
    """Absolute EVE under shocked scenario — level 1."""
    return (compute_eve_base_fast(amounts, params)
            + compute_delta_eve_fast(amounts, params, scenario_id))


def compute_eve_all_scenarios(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> dict[str, float]:
    """Compute base EVE and delta_EVE for all scenarios.

    Returns dict:
        'base'        → base EVE
        scenario_id   → delta_EVE for each shocked scenario
    """
    base = compute_eve_base_fast(amounts, params)
    delta_vec = params.delta_eve_unit.T @ amounts   # (S,)
    result = {"base": base}
    for s_idx, scen in enumerate(params.scenario_ids):
        result[str(scen)] = float(delta_vec[s_idx])
    return result


def compute_worst_delta_eve(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> tuple[float, str]:
    """Return (worst delta_EVE, scenario_id) across all scenarios.

    worst = minimum (most negative) delta_EVE.
    """
    delta_vec = params.delta_eve_unit.T @ amounts   # (S,)
    worst_idx  = int(np.argmin(delta_vec))
    return float(delta_vec[worst_idx]), str(params.scenario_ids[worst_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — modified duration model
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_eve_duration(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    delta_r_parallel: float,
) -> float:
    """delta_EVE via modified duration for a parallel shift — level 2.

    delta_EVE[i] ≈ -D_mod[i] × amounts[i] × delta_r_parallel

    This is a smooth, differentiable approximation of delta_EVE.
    Best used for gradient computation inside the optimizer; use level 1
    for function evaluations to stay close to the exact model.

    Parameters
    ----------
    delta_r_parallel : rate shock in decimal (e.g. +0.02 for +200 bps par_up)
    """
    return float(-np.dot(params.d_mod * amounts, delta_r_parallel))


def compute_delta_eve_gradient(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    delta_r_parallel: float,
) -> np.ndarray:
    """Gradient of delta_EVE w.r.t. amounts — level 2.

    d(delta_EVE) / d(amounts[i]) = -D_mod[i] × delta_r_parallel

    Useful for gradient-based optimisers (SLSQP, L-BFGS-B).
    Returns (n,) array.
    """
    return -params.d_mod * delta_r_parallel


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 — analytical PV using pre-interpolated discount curves
# ─────────────────────────────────────────────────────────────────────────────

def _annuity_pv(
    coupon_rate: np.ndarray,    # (n,) annual coupon rate (decimal)
    disc_curve: np.ndarray,     # (360,) monthly discount factors
    repricing_m: np.ndarray,    # (n,) average maturity in months
    sign: np.ndarray,           # (n,) +1 for assets, -1 for liabilities
) -> np.ndarray:
    """PV of a flat-coupon annuity per unit balance, using shocked disc curve.

    Approximation: treats each product as a bullet bond with:
    - monthly coupon = coupon_rate / 12
    - maturity at repricing_tenor_m months from today
    - principal repaid at maturity

    This gives a closed-form-like PV using pre-interpolated d_f grids.
    Returns (n,) PV per unit balance.
    """
    n = len(coupon_rate)
    m_idx = np.clip(repricing_m.astype(int) - 1, 0, len(disc_curve) - 1)

    pv_per_unit = np.zeros(n, dtype=float)
    for i in range(n):
        T = m_idx[i]
        if T < 0:
            continue
        # PV of coupons: Σ_{t=1}^{T} (rate/12) × d_f[t]
        pv_coupon = np.dot(coupon_rate[i] / 12.0, disc_curve[:T + 1].sum())
        # PV of principal repayment at T
        pv_principal = disc_curve[T]
        pv_per_unit[i] = (pv_coupon + pv_principal) * sign[i]
    return pv_per_unit


def compute_eve_analytical_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    curves: CurveTensors,
    scenario_id: str,
    currency: str = "PLN",
) -> float:
    """EVE using analytical PV of annuities — level 3.

    More accurate for large rate shifts or new balance sheet structures
    that deviate significantly from the current book.

    Uses the repricing_tenor_m as average remaining maturity and
    nii_unit_rate / horizon_yf as the effective coupon rate per product.

    Parameters
    ----------
    amounts     : float64 (n,)
    params      : BalanceSheetParams
    curves      : CurveTensors
    scenario_id : e.g. 'par_up', 'par_dn', 'base'
    currency    : currency for which to look up the disc curve
    """
    disc_curve  = curves.get_disc_curve(scenario_id, currency)
    coupon_rate = np.abs(params.nii_unit_rate)   # effective rate per year

    pv_per_unit = _annuity_pv(
        coupon_rate,
        disc_curve,
        params.repricing_tenor_m,
        params.sign,
    )
    mask = params.currency == currency
    return float(np.dot(amounts * mask, pv_per_unit))


# ─────────────────────────────────────────────────────────────────────────────
# SOT EVE regulatory metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_sot_eve_pct(
    delta_eve: float,
    tier1_capital: float,
) -> float:
    """EBA SOT EVE metric: delta_EVE / Tier1 capital × 100.

    EBA threshold: must not be < -15%.
    """
    return delta_eve / tier1_capital * 100.0 if tier1_capital > 0 else float("nan")
