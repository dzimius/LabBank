"""hyp_engine.py
================
Compute IRRBB metrics for a hypothetical PLN rate environment.

The user selects a stylised curve shape + level.  We replace the PLN base
scenario in CurveTensors with the hypothetical curve and shift ALL shocked
PLN scenarios by the same delta, preserving EBA shock shapes relative to the
new base.

NII base + delta_NII — full CF computation: rate_matrix rebuilt from hyp
    curves, then einsum over outstanding profiles + calibration corrections.
EVE base  — CF re-discounting with new disc factors (compute_eve_cf_all).
delta_EVE — CF re-discounting with new shocked disc factors.
"""
from __future__ import annotations

import numpy as np

from bs_vector import BalanceSheetParams, CurveTensors, CohortRates


# ── Hypothetical CurveTensors ─────────────────────────────────────────────────

def build_hyp_curve_tensors(
    curves: CurveTensors,
    fwd_hyp_pln: np.ndarray,          # (360,) annualised fwd rates, decimal
) -> CurveTensors:
    """Return a new CurveTensors where the PLN base is replaced by fwd_hyp_pln.

    All PLN scenarios are shifted by (fwd_hyp - fwd_base) so EBA shock shapes
    are preserved relative to the new base.  EUR/USD scenarios are unchanged.
    """
    n_m     = curves.n_months
    pln_idx = int(np.where(curves.currencies == "PLN")[0][0])

    fwd_real_base = curves.fwd_rates[0, pln_idx, :n_m]        # (n_m,)
    delta          = fwd_hyp_pln[:n_m] - fwd_real_base        # (n_m,)

    new_fwd  = curves.fwd_rates.copy()                         # (S, C, 360)
    new_fwd[:, pln_idx, :n_m] += delta[None, :]
    new_fwd  = np.maximum(new_fwd, 0.0)

    # Recompute disc factors from shifted fwd
    new_disc = np.exp(-np.cumsum(new_fwd / 12.0, axis=2))

    return CurveTensors(
        disc_factors = new_disc,
        fwd_rates    = new_fwd,
        scenario_ids = curves.scenario_ids.copy(),
        currencies   = curves.currencies.copy(),
        curve_names  = curves.curve_names.copy(),
        report_date  = curves.report_date,
        n_months     = curves.n_months,
    )


# ── Rate matrix for hypothetical curves ──────────────────────────────────────

def build_hyp_rate_matrix(
    params:     BalanceSheetParams,
    cr:         CohortRates,
    hyp_curves: CurveTensors,
) -> tuple[np.ndarray, np.ndarray]:
    """Build rate_matrix and renewal_rate_matrix for hypothetical CurveTensors.

    Returns (rate_matrix, renewal_rate_matrix), both (n, 12, n_rate_scen).

    Fixed cohorts: coupon_rate, constant across months and scenarios.
    Float locked period (month < cohort_t_first_m): cohort_locked_rate.
    Float repricing period: clip(coeff_a * hyp_fwd[repricing_tenor] + coeff_b, floor, cap).
    Renewal rate: clip(coeff_a * hyp_fwd[m] + coeff_b, floor, cap) — 1-month forward.
    """
    n       = len(params.product_code)
    n_m     = hyp_curves.n_months
    n_scen  = cr.n_rate_scen
    pln_idx = int(np.where(hyp_curves.currencies == "PLN")[0][0])

    tenor_idx = np.clip(params.repricing_tenor_m.astype(int) - 1, 0, n_m - 1)  # (n,)
    t_first   = np.clip(params.cohort_t_first_m.astype(int), 0, 12)             # (n,)
    is_fixed  = params.rate_type == "F"                                          # (n,)

    safe_a     = np.nan_to_num(params.coeff_a,      nan=0.0)
    safe_b     = np.nan_to_num(params.coeff_b,      nan=0.0)
    safe_floor = np.nan_to_num(params.client_floor, nan=-1.0)
    safe_cap   = np.nan_to_num(params.client_cap,   nan=1.0)

    rate_matrix    = np.zeros((n, 12, n_scen), dtype=float)
    renewal_matrix = np.zeros((n, 12, n_scen), dtype=float)

    for rs in range(n_scen):
        # hyp_curves.fwd_rates is ordered per hyp_curves.scenario_ids (the full
        # curve-tensor scenario set, alphabetical) -- NOT per cr.rate_scenario_ids
        # (the calibrated EBA-scenario set, definition order). Only index 0
        # ("base") coincides between the two orderings; every other position
        # must be looked up by name or every non-base scenario silently pulls
        # the wrong curve (2026-08-15 fix -- this previously scrambled every
        # shock's label, e.g. "own" was actually built from the par_dn curve).
        curve_idx   = hyp_curves.scenario_index(str(cr.rate_scenario_ids[rs]))
        fwd_s       = hyp_curves.fwd_rates[curve_idx, pln_idx, :n_m]                 # (n_m,)
        client_rate = np.clip(safe_a * fwd_s[tenor_idx] + safe_b, safe_floor, safe_cap)  # (n,)

        for m in range(12):
            rate_matrix[:, m, rs] = np.where(
                is_fixed,
                params.coupon_rate,
                np.where(m < t_first, params.cohort_locked_rate, client_rate),
            )
            ren_rate = np.clip(safe_a * fwd_s[m] + safe_b, safe_floor, safe_cap)
            renewal_matrix[:, m, rs] = np.where(is_fixed, 0.0, ren_rate)

    return rate_matrix, renewal_matrix


# ── NII in hypothetical environment ──────────────────────────────────────────

def compute_nii_hyp(
    balance:     np.ndarray,
    params:      BalanceSheetParams,
    cr:          CohortRates,
    curves:      CurveTensors,
    fwd_hyp_pln: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Return (nii_hyp_base, delta_nii_dict).

    Base NII
    --------
    Rebuild the client-rate matrix from the hypothetical forward curve (floors /
    caps applied at the *hypothetical* rate level), run the CF einsum, then anchor
    to the exact-pipeline base with a **curve-independent** offset:

        bias[i] = exact_nii_base[i] - CF_fast_nii_base_on_MARKET_curve[i]

    so the hypothetical curve's genuine effect on base NII flows through instead
    of being cancelled by re-anchoring against the hypothetical CF base.

    delta_NII (per EBA shock)
    -------------------------
    The base supervisory curve is the calibration point, so start from the exact
    ``params.delta_nii_unit`` and add only the second-order correction the CF
    model sees from moving the base curve — i.e. the difference in how the
    0%-floor / cap bites at the hypothetical level vs the market level:

        d[i,s] = market_delta[i,s]
               + ( (nii_hyp[i,s]-nii_hyp[i,base]) - (nii_mkt[i,s]-nii_mkt[i,base]) )

    Both CF terms come from the *same* code path, so any model quirk cancels in
    the delta-of-deltas.  When hyp == market this reproduces ``delta_nii_unit``
    exactly.
    """
    pln_idx      = int(np.where(curves.currencies == "PLN")[0][0])
    fwd_mkt_pln  = curves.fwd_rates[0, pln_idx, :curves.n_months]

    # Both re-priced through the identical code path so the delta-of-deltas below
    # isolates purely the effect of swapping the base curve.
    market_curves = build_hyp_curve_tensors(curves, fwd_mkt_pln)
    hyp_curves    = build_hyp_curve_tensors(curves, fwd_hyp_pln)

    hyp_rm,  hyp_rnm   = build_hyp_rate_matrix(params, cr, hyp_curves)
    mkt_rm,  mkt_rnm   = build_hyp_rate_matrix(params, cr, market_curves)

    amount_x_out  = (balance * params.sign)[:, None] * params.cohort_interest_yf_m   # (n, 12)
    cap_x_remain  = (balance * params.sign)[:, None] * params.cohort_capital_remain_m  # (n, 12)

    nii_hyp = (np.einsum("im,ims->is", amount_x_out, hyp_rm)
               + np.einsum("im,ims->is", cap_x_remain, hyp_rnm))                     # (n, n_scen)
    nii_mkt = (np.einsum("im,ims->is", amount_x_out, mkt_rm)
               + np.einsum("im,ims->is", cap_x_remain, mkt_rnm))

    base_idx = cr.scenario_index("base")

    # ── base level: hyp CF base + curve-independent calibration offset ────────
    bias      = (balance * params.nii_unit_rate) - nii_mkt[:, base_idx]
    nii_base  = float((nii_hyp[:, base_idx] + bias).sum())

    # ── shocked deltas: exact market delta + floor/cap re-binding correction ──
    p_scen = list(params.scenario_ids)
    delta_nii: dict[str, float] = {}
    for rs, scen in enumerate(cr.rate_scenario_ids):
        scen = str(scen)
        if scen == "base":
            continue
        try:
            market_delta = float(np.dot(balance, params.delta_nii_unit[:, p_scen.index(scen)]))
        except ValueError:
            market_delta = float(nii_mkt[:, rs].sum() - nii_mkt[:, base_idx].sum())
        cf_adj = float((nii_hyp[:, rs] - nii_hyp[:, base_idx]).sum()
                       - (nii_mkt[:, rs] - nii_mkt[:, base_idx]).sum())
        delta_nii[scen] = market_delta + cf_adj

    return nii_base, delta_nii


# ── EVE in hypothetical environment ──────────────────────────────────────────

def compute_eve_hyp(
    balance:     np.ndarray,
    params:      BalanceSheetParams,
    curves:      CurveTensors,
    fwd_hyp_pln: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Return (eve_hyp_base, delta_eve_dict) using a duration-based approach.

    Base EVE
    --------
    The pre-stored `eve_pv_frac` in CohortRates is scenario-specific but was
    built for the original curves — it overrides CF discounting for 490/502
    cohorts and makes the CF matrix insensitive to a new base curve.

    Instead we use `delta_eve_unit` (calibrated per-bp sensitivity) together
    with the per-product modified duration tenor to project the EVE level at
    the hypothetical base:

        scale_i = (fwd_hyp(d_mod_tenor_i) − fwd_base(d_mod_tenor_i))
                  / (fwd_par_up(d_mod_tenor_i) − fwd_base(d_mod_tenor_i))

        EVE_hyp = EVE_calib + Σ_i balance_i × delta_eve_unit[i, par_up] × scale_i

    This weights each product's EVE sensitivity by how much the hypothetical
    curve shifts at that product's rate-sensitive tenor.

    delta_EVE (shocked)
    -------------------
    Reuse the calibrated ``delta_eve_unit`` from the exact pipeline.  EVE shock
    sensitivity does shift with the base level (convexity, and the 0%-floor on
    down-shocks), but re-deriving it needs full CF re-discounting under every
    (hyp-curve × shock) pair; that is a documented first-order approximation
    here — the shock shape itself is unchanged by ``build_hyp_curve_tensors``,
    so to first order the sensitivity carries over.  The NII deltas *do* pick up
    the floor re-binding (see ``compute_nii_hyp``).
    """
    n_m      = curves.n_months
    pln_idx  = int(np.where(curves.currencies == "PLN")[0][0])
    fwd_base = curves.fwd_rates[0, pln_idx, :n_m]

    par_up_scen_idx = curves.scenario_index("par_up")
    fwd_par_up      = curves.fwd_rates[par_up_scen_idx, pln_idx, :n_m]

    # Map each product's modified duration (years) → month index in curve arrays
    d_mod_yrs  = np.abs(params.d_mod)                                     # (n,)
    tenor_idx  = np.clip(np.round(d_mod_yrs * 12).astype(int) - 1, 0, n_m - 1)

    # Rate shift at each product's duration tenor
    hyp_shift = fwd_hyp_pln[tenor_idx] - fwd_base[tenor_idx]             # (n,)
    pup_shift = fwd_par_up[tenor_idx]  - fwd_base[tenor_idx]             # (n,)

    # Scale par_up sensitivity; floor avoids zero-division for zero-duration products
    safe_pup  = np.where(np.abs(pup_shift) > 1e-7, pup_shift, 1e-7)
    scale     = hyp_shift / safe_pup                                      # (n,)

    par_up_p_idx = list(params.scenario_ids).index("par_up")
    d_eve_par_up = params.delta_eve_unit[:, par_up_p_idx]                 # (n,)

    eve_calib  = float(np.dot(balance, params.eve_pv_factor))
    eve_shift  = float(np.dot(balance * d_eve_par_up, scale))
    eve_base   = eve_calib + eve_shift

    # delta_EVE: calibrated unit sensitivities (level-independent, first order)
    delta_eve: dict[str, float] = {}
    for s_idx, scen in enumerate(params.scenario_ids):
        delta_eve[str(scen)] = float(np.dot(balance, params.delta_eve_unit[:, s_idx]))

    return eve_base, delta_eve


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_hyp_metrics(
    balance:     np.ndarray,
    params:      BalanceSheetParams,
    cr:          CohortRates,
    curves:      CurveTensors,
    fwd_hyp_pln: np.ndarray,
) -> dict:
    """Full IRRBB metrics for a hypothetical PLN rate environment.

    Returns dict with:
        nii_base   float      12M NII in PLN
        delta_nii  dict[str, float]  delta_NII per scenario
        eve_base   float      EVE in PLN
        delta_eve  dict[str, float]  delta_EVE per scenario
    """
    nii_base, delta_nii = compute_nii_hyp(balance, params, cr, curves, fwd_hyp_pln)
    eve_base, delta_eve = compute_eve_hyp(balance, params, curves, fwd_hyp_pln)

    return {
        "nii_base":  nii_base,
        "delta_nii": delta_nii,
        "eve_base":  eve_base,
        "delta_eve": delta_eve,
    }
