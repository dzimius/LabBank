"""nii_eve_cf_fast.py
====================
CF-based fast NII and EVE for the optimizer hot path.

NII model
---------
Uses pre-computed rate_matrix[n, 12, n_scen] where each cell is the client
interest rate for cohort i in calendar month m under scenario s.

    NII[i, s] = balance[i] * sign[i]
                * dot(outstanding_m[i,:], rate_matrix[i,:,s]) / 12

Runtime: single einsum — O(n*12*n_scen) in numpy, microseconds for n<2000.

Bias correction
---------------
The CF model has a systematic error at the base point (approximated forward
rates vs exact pipeline).  Compute once at calibration:

    bias_nii[product] = exact_nii_base[product] - fast_nii_cf_base[product]

Apply to every scenario:

    NII_corrected[product, s] = NII_fast_cf[product, s] + bias_nii[product]

At the base point this is exact; in shocked scenarios the bias is assumed
constant (first-order correction).

EVE delta — two approaches
---------------------------
1. Duration (fast, smooth):
       ΔEVE[i,s] = balance[i] * delta_eve_dur_unit[i,s]
   where delta_eve_dur_unit was pre-computed as:
       -sign[i] * D_mod[i] * (fwd_shocked[D_mod_tenor] - fwd_base[D_mod_tenor])

2. CF-based (slower at calibration, more accurate for fixed cohorts):
       EVE_CF[i,s] = balance[i] * sum_q(cf_frac[i,q] * disc[s,ccy_i,q_month])
       ΔEVE_CF[i,s] = EVE_CF[i,s] - EVE_CF[i,base]
   Uses cf_fixed_int_frac for fixed cohorts (capital+interest) and
   cf_capital_frac for float cohorts (capital only, reprices near par).

Both approaches include a bias correction:
    bias_eve[product] = exact_eve_base[product] - fast_eve_cf_base[product]
"""
from __future__ import annotations

import numpy as np
from bs_vector import BalanceSheetParams, CohortRates, CurveTensors


# ─────────────────────────────────────────────────────────────────────────────
# NII
# ─────────────────────────────────────────────────────────────────────────────

def _nii_cohort_matrix(
    balance: np.ndarray,          # (n,)
    params:  BalanceSheetParams,
    cr:      CohortRates,
) -> np.ndarray:
    """Interest NII per cohort per scenario — shape (n, n_scen).

    Uses rate_matrix[n, 12, n_scen] and cohort_interest_yf_m[n, 12].
    """
    # weighted outstanding-year-fraction: (n, 12)
    amount_x_out = (balance * params.sign)[:, None] * params.cohort_interest_yf_m
    # dot over 12 months for each scenario: (n, n_scen)
    return np.einsum("im,ims->is", amount_x_out, cr.rate_matrix)


# remain_yf[m] = fraction of horizon remaining after renewal in month m
# renewal happens ~mid-month, so remaining = (12 - m - 0.5) / 12
_REMAIN_YF = np.array([(12 - m - 0.5) / 12.0 for m in range(12)], dtype=float)


def _nii_renewal_matrix(
    balance: np.ndarray,
    params:  BalanceSheetParams,
    cr:      CohortRates,
) -> np.ndarray:
    """Renewal NII per cohort per scenario — shape (n, n_scen).

    Renewal = capital repaid in month m reinvested at new-business rate
    for the remaining (12 - m - 0.5) / 12 year-fraction of the horizon.

        NII_renewal[i,s] = sign[i] * Σ_m (capital_m[i] * balance[i]
                           * renewal_rate[i,m,s] * remain_yf[m])
    """
    # weighted capital repayment: (n, 12)
    amount_x_cap = (balance * params.sign)[:, None] * params.cohort_capital_remain_m
    # scalar remain_yf per month: (1, 12)
    remain = _REMAIN_YF[None, :]
    # weighted capital × remain_yf: (n, 12)
    cap_x_remain = amount_x_cap
    # dot over 12 months for each scenario: (n, n_scen)
    return np.einsum("im,ims->is", cap_x_remain, cr.renewal_rate_matrix)


def _apply_fixed_rate_calibration(
    nii_total: np.ndarray,      # (n, n_scen) — modified in place
    balance:   np.ndarray,
    params:    BalanceSheetParams,
    cr:        CohortRates,
) -> None:
    """Override shocked NII for F-type cohort products with calibrated delta_nii_unit.

    F-type products (fixed coupon) have a constant rate_matrix across all scenarios,
    meaning their run-off NII is scenario-independent (zero run-off delta).  All
    scenario sensitivity therefore flows through the renewal stream, which uses the
    contractual repricing schedule.  For NMD products like 7060 (fixed-term deposits
    that roll over monthly), the contractual schedule gives 100% of balance repricing
    within 12M while the IRRBB behavioral model gives only ~13% effective repricing.
    This produces a 8-10x overestimate for sr_dn/sr_up (wrong sign on the total).

    Fix: for rows where rate_matrix is constant across scenarios, replace shocked NII
    with base_NII + calibrated_delta_nii_unit.  This gives exact IRRBB sensitivity
    for F-type products without implementing the behavioral model explicitly.

    V-type (floating) products are unaffected — they keep the rate_matrix approach.
    """
    base_idx = cr.scenario_index("base")

    # Detect fixed-rate rows: rate_matrix has no cross-scenario variation
    rm = cr.rate_matrix  # (n, 12, n_scen)
    is_fixed = np.abs(rm - rm[:, :, base_idx : base_idx + 1]).max(axis=(1, 2)) < 1e-8
    if not is_fixed.any():
        return

    # Build scenario mapping: cr.rate_scenario_ids index -> params.scenario_ids index
    p_scen_list = list(params.scenario_ids)
    scen_map: dict[int, int] = {}
    for rs, scen in enumerate(cr.rate_scenario_ids):
        if str(scen) == "base":
            continue
        try:
            scen_map[rs] = p_scen_list.index(str(scen))
        except ValueError:
            pass

    # For each shocked scenario: total = base_NII + calibrated_delta
    base_nii_fixed = nii_total[is_fixed, base_idx : base_idx + 1]  # (n_fixed, 1)
    calib_mat = np.zeros((int(is_fixed.sum()), cr.n_rate_scen), dtype=float)
    for rs, p_idx in scen_map.items():
        calib_mat[:, rs] = balance[is_fixed] * params.delta_nii_unit[is_fixed, p_idx]
    nii_total[is_fixed, :] = base_nii_fixed + calib_mat


def _fallback_mask(params: BalanceSheetParams) -> np.ndarray:
    """Rows where method B has no real monthly CF schedule to drive off of.

    Single-row behavioural products (non-cohort) and cohort entries using a
    synthetic constant-balance profile (cf.products has no complete monthly
    schedule for them) -- for these, the CF einsum is structurally zero
    regardless of curve/scenario, so any base or delta NII must come from the
    calibrated per-unit rates instead.
    """
    out = params.cohort_outstanding_m
    cap = params.cohort_capital_m
    synthetic_constant = (
        params.is_cohort
        & (np.abs(cap).sum(axis=1) < 1e-10)
        & (np.abs(out - 1.0).max(axis=1) < 1e-8)
    )
    return (~params.is_cohort) | synthetic_constant


def _apply_calibrated_delta_fallback(
    nii_total: np.ndarray,      # (n, n_scen) - modified in place
    balance:   np.ndarray,
    params:    BalanceSheetParams,
    cr:        CohortRates,
) -> None:
    """Use calibrated deltas where method B has no real monthly CF schedule.

    Some entries are single-row behavioural products, and some cohort entries use
    a synthetic constant-balance profile because cf.products does not provide a
    complete monthly schedule.  For those rows a CF-based delta is not a better
    approximation of the cashflow engine; it is an invented repricing profile.
    Keep the B base level, but replace shocked columns with base + calibrated
    delta_NII per PLN from the exact engine.
    """
    base_idx = cr.scenario_index("base")
    p_scen_list = list(params.scenario_ids)
    scen_map: dict[int, int] = {}
    for rs, scen in enumerate(cr.rate_scenario_ids):
        if str(scen) == "base":
            continue
        try:
            scen_map[rs] = p_scen_list.index(str(scen))
        except ValueError:
            pass

    fallback = _fallback_mask(params)
    if not fallback.any():
        return

    base_nii = nii_total[fallback, base_idx : base_idx + 1]
    calib_mat = np.zeros((int(fallback.sum()), cr.n_rate_scen), dtype=float)
    for rs, p_idx in scen_map.items():
        calib_mat[:, rs] = balance[fallback] * params.delta_nii_unit[fallback, p_idx]
    nii_total[fallback, :] = base_nii + calib_mat


def _apply_base_nii_bias(
    nii_total: np.ndarray,      # (n, n_scen) - modified in place
    balance:   np.ndarray,
    params:    BalanceSheetParams,
    cr:        CohortRates,
) -> None:
    """Anchor method B base NII to the exact calibrated base level.

    This applies a constant row-level offset to all scenarios, so shocked deltas
    stay purely driven by the CF/rate_matrix approximation.
    """
    base_idx = cr.scenario_index("base")
    exact_base = balance * params.nii_unit_rate
    bias = exact_base - nii_total[:, base_idx]
    nii_total += bias[:, None]


def compute_nii_cf_all(
    balance: np.ndarray,
    params:  BalanceSheetParams,
    cr:      CohortRates,
) -> dict[str, float]:
    """NII for all scenarios (portfolio total).

    CF-based for rows with a usable monthly schedule:
    outstanding_m * rate_matrix / 12 + capital_m * renewal_rate * remain_yf.
    Rows without a real schedule use calibrated delta_NII_unit on shocked columns.

    Returns dict: scenario_id -> NII in PLN.
    """
    nii_total = (
        _nii_cohort_matrix(balance, params, cr)
        + _nii_renewal_matrix(balance, params, cr)
    )                           # (n, n_scen)
    _apply_fixed_rate_calibration(nii_total, balance, params, cr)
    _apply_calibrated_delta_fallback(nii_total, balance, params, cr)
    _apply_base_nii_bias(nii_total, balance, params, cr)
    return {str(s): float(nii_total[:, i].sum()) for i, s in enumerate(cr.rate_scenario_ids)}


def compute_nii_cf_by_product(
    balance:  np.ndarray,
    params:   BalanceSheetParams,
    cr:       CohortRates,
) -> dict[tuple, np.ndarray]:
    """NII per (product_code, bs_side, currency) per scenario — shape (n_scen,).

    CF-based for rows with a usable monthly schedule:
    outstanding_m * rate_matrix / 12 + capital_m * renewal_rate * remain_yf.
    Returns dict keyed by (product_code, bs_side, currency).
    """
    nii_cohort  = _nii_cohort_matrix(balance, params, cr)   # (n, n_scen)
    nii_renewal = _nii_renewal_matrix(balance, params, cr)  # (n, n_scen)
    nii_total   = nii_cohort + nii_renewal
    _apply_fixed_rate_calibration(nii_total, balance, params, cr)
    _apply_calibrated_delta_fallback(nii_total, balance, params, cr)
    _apply_base_nii_bias(nii_total, balance, params, cr)

    agg: dict[tuple, np.ndarray] = {}
    for i, (pc, sid, ccy) in enumerate(
        zip(params.product_code, params.bs_side, params.currency)
    ):
        k = (str(pc), str(sid), str(ccy))
        if k not in agg:
            agg[k] = np.zeros(cr.n_rate_scen, dtype=float)
        agg[k] += nii_total[i]
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# EVE — duration approach
# ─────────────────────────────────────────────────────────────────────────────

def compute_eve_dur_all(
    balance: np.ndarray,
    params:  BalanceSheetParams,
    cr:      CohortRates,
) -> dict[str, float]:
    """Base EVE + delta EVE (duration approach) for all scenarios.

    Base: balance @ eve_pv_factor (calibrated, exact).
    Delta: balance @ delta_eve_dur_unit[:, s] (pre-computed from D_mod + curve shifts).

    Returns dict: 'base' -> EVE, scenario_id -> delta_EVE.
    """
    eve_base = float(np.dot(balance, params.eve_pv_factor))
    result: dict[str, float] = {}
    for i, s in enumerate(cr.rate_scenario_ids):
        s = str(s)
        if s == "base":
            result["base"] = eve_base
        else:
            result[s] = float(np.dot(balance, cr.delta_eve_dur_unit[:, i]))
    return result


def compute_eve_dur_by_product(
    balance: np.ndarray,
    params:  BalanceSheetParams,
    cr:      CohortRates,
) -> dict[tuple, np.ndarray]:
    """delta EVE per product per scenario — duration approach.

    Returns dict: (product_code, bs_side, currency) -> (n_scen,) array.
    Index 0 = base (EVE level), indices 1+ = delta_EVE per shocked scenario.
    """
    # base EVE per cohort
    eve_base_cohort = balance * params.eve_pv_factor               # (n,)
    # delta EVE per cohort per scenario: (n, n_scen)
    delta_cohort = balance[:, None] * cr.delta_eve_dur_unit        # (n, n_scen)

    agg: dict[tuple, np.ndarray] = {}
    for i, (pc, sid, ccy) in enumerate(
        zip(params.product_code, params.bs_side, params.currency)
    ):
        k = (str(pc), str(sid), str(ccy))
        if k not in agg:
            agg[k] = np.zeros(cr.n_rate_scen, dtype=float)
        agg[k][0] += eve_base_cohort[i]
        agg[k][1:] += delta_cohort[i, 1:]
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# EVE — CF approach
# ─────────────────────────────────────────────────────────────────────────────

def _eve_cf_cohort_matrix(
    balance: np.ndarray,
    params:  BalanceSheetParams,
    cr:      CohortRates,
    curves:  CurveTensors,
    use_precomputed_frac: bool = True,
) -> np.ndarray:
    """EVE (PV of CFs) per cohort per scenario — shape (n, n_scen).

    Fixed cohorts: sum_q (capital_cf + interest_cf) * disc(q_yf, scenario).
    Float cohorts: sum_q  capital_cf               * disc(q_yf, scenario).

    Index 0 = base EVE level, indices 1+ = shocked EVE level
    (caller computes delta = shocked - base).

    use_precomputed_frac: the pre-stored cr.eve_pv_frac shortcut (below) is
    only valid when `curves` IS the curve cr.eve_pv_frac was calibrated
    against -- set False when `curves` is a different/hypothetical curve
    (e.g. sandbox/build_scenario_curves.py's per-shape precompute), or this
    silently discards the CF-based recomputation above and returns the
    real-curve EVE regardless of what was passed in (2026-08-15 fix -- this
    previously made EVE bit-identical across every hypothetical curve for
    the 522/535 cohorts that have a stored eve_pv_frac).
    """
    n       = len(balance)
    n_scen  = cr.n_rate_scen
    max_q   = cr.cf_fixed_int_frac.shape[1]
    eve_mat = np.zeros((n, n_scen), dtype=float)

    # Convert year fractions to 0-indexed month positions in disc tensor (0-based)
    # disc_tensor[:, :, m] = disc factor at end of month m+1
    # yf -> month index = round(yf * 12) - 1, clipped to [0, N_M-1]
    N_M     = curves.n_months
    cf_midx = np.clip(np.round(cr.cf_yf * 12).astype(int) - 1, 0, N_M - 1)  # (n, max_q)

    # Valid CF mask (n, max_q)
    valid = np.arange(max_q)[None, :] < cr.cf_n_q[:, None]        # (n, max_q)

    for rs, scen_label in enumerate(cr.rate_scenario_ids):
        scen_label = str(scen_label)
        try:
            s_idx = curves.scenario_index(scen_label)
        except KeyError:
            s_idx = curves.scenario_index("base")

        for c_idx, ccy in enumerate(curves.currencies):
            mask_ccy = cr.ccy_idx == c_idx
            if not mask_ccy.any():
                continue

            disc_arr = curves.disc_factors[s_idx, c_idx]   # (N_M,)
            # disc factors at each CF quarter midpoint for this currency group
            disc_q = disc_arr[cf_midx[mask_ccy, :]]        # (n_ccy, max_q)
            cf_frac = cr.cf_fixed_int_frac[mask_ccy, :]    # (n_ccy, max_q)
            val_ccy = valid[mask_ccy, :]                    # (n_ccy, max_q)
            # PV fraction per cohort — apply sign so liabilities are negative
            pv_frac = (cf_frac * disc_q * val_ccy).sum(axis=1)   # (n_ccy,)
            eve_mat[mask_ccy, rs] = balance[mask_ccy] * params.sign[mask_ccy] * pv_frac

    if use_precomputed_frac:
        use_eff = np.any(np.abs(cr.eve_pv_frac) > 1e-18, axis=(1, 2))
        if use_eff.any():
            eve_mat[use_eff, :] = (
                balance[use_eff, None]
                * params.sign[use_eff, None]
                * cr.eve_pv_frac[use_eff, :, :].sum(axis=1)
            )

    return eve_mat


def compute_eve_cf_all(
    balance: np.ndarray,
    params:  BalanceSheetParams,
    cr:      CohortRates,
    curves:  CurveTensors,
) -> dict[str, float]:
    """Base EVE + delta EVE (CF approach) for all scenarios.

    Returns dict: 'base' -> EVE, scenario_id -> delta_EVE.
    """
    eve_mat  = _eve_cf_cohort_matrix(balance, params, cr, curves)  # (n, n_scen)
    base_idx = cr.scenario_index("base")
    eve_base_raw = float(eve_mat[:, base_idx].sum())
    eve_base = float(np.dot(balance, params.eve_pv_factor))

    result: dict[str, float] = {"base": eve_base}
    for rs, scen in enumerate(cr.rate_scenario_ids):
        scen = str(scen)
        if scen == "base":
            continue
        result[scen] = float(eve_mat[:, rs].sum()) - eve_base_raw
    return result


def compute_eve_cf_by_product(
    balance: np.ndarray,
    params:  BalanceSheetParams,
    cr:      CohortRates,
    curves:  CurveTensors,
) -> dict[tuple, np.ndarray]:
    """delta EVE per product per scenario — CF approach.

    Returns dict: (product_code, bs_side, currency) -> (n_scen,) array.
    Index 0 = base EVE level, indices 1+ = delta_EVE.
    """
    eve_mat  = _eve_cf_cohort_matrix(balance, params, cr, curves)  # (n, n_scen)
    base_idx = cr.scenario_index("base")

    agg: dict[tuple, np.ndarray] = {}
    for i, (pc, sid, ccy) in enumerate(
        zip(params.product_code, params.bs_side, params.currency)
    ):
        k = (str(pc), str(sid), str(ccy))
        if k not in agg:
            agg[k] = np.zeros(cr.n_rate_scen, dtype=float)
        agg[k][0] += balance[i] * params.eve_pv_factor[i]
        for rs in range(cr.n_rate_scen):
            if rs != base_idx:
                agg[k][rs] += eve_mat[i, rs] - eve_mat[i, base_idx]
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Bias correction helpers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# EVE — fast CF approach using pre-indexed cohort_disc_q (no CurveTensors)
# ─────────────────────────────────────────────────────────────────────────────

def compute_eve_cf_fast_all(
    balance: np.ndarray,
    params:  BalanceSheetParams,
    cr:      CohortRates,
) -> dict[str, float]:
    """Base EVE + delta EVE using pre-indexed cr.cohort_disc_q — no CurveTensors.

    Shock substitution = pick scenario column from cohort_disc_q, single einsum.
    Returns dict: 'base' -> EVE, scenario_id -> delta_EVE.
    """
    max_q    = cr.cf_fixed_int_frac.shape[1]
    base_idx = cr.scenario_index("base")
    valid    = (np.arange(max_q)[None, :] < cr.cf_n_q[:, None]).astype(float)

    cf_masked = cr.cf_fixed_int_frac * valid                           # (n, max_q)
    pv_frac   = np.einsum("iq,iqs->is", cf_masked, cr.cohort_disc_q)   # (n, n_scen)
    use_eff   = np.any(np.abs(cr.eve_pv_frac) > 1e-18, axis=(1, 2))
    if use_eff.any():
        pv_frac[use_eff, :] = cr.eve_pv_frac[use_eff, :, :].sum(axis=1)
    pv_mat    = pv_frac * (balance * params.sign)[:, None]   # sign: +1 assets, -1 liabilities

    eve_base_raw = float(pv_mat[:, base_idx].sum())
    eve_base  = float(np.dot(balance, params.eve_pv_factor))
    result: dict[str, float] = {"base": eve_base}
    for rs, scen in enumerate(cr.rate_scenario_ids):
        scen = str(scen)
        if scen == "base":
            continue
        result[scen] = float(pv_mat[:, rs].sum()) - eve_base_raw

    # Fallback: cohorts with no CF schedule AND no eve_pv_frac (e.g. IRS product 0000).
    # Use delta_eve_unit calibrated from SQL — same as accuracy_check._approach_b_scen.
    blank = (~use_eff) & (cr.cf_n_q == 0)
    if blank.any():
        scen_p_idx = {str(s): i for i, s in enumerate(params.scenario_ids)}
        for scen, s_idx in scen_p_idx.items():
            if scen in result:
                result[scen] += float(
                    np.dot(balance[blank], params.delta_eve_unit[blank, s_idx])
                )
    return result


def compute_eve_cf_fast_by_product(
    balance: np.ndarray,
    params:  BalanceSheetParams,
    cr:      CohortRates,
) -> dict[tuple, np.ndarray]:
    """delta EVE per product per scenario using cohort_disc_q.

    Returns dict: (product_code, bs_side, currency) -> (n_scen,) array.
    Index 0 = base EVE level, indices 1+ = delta_EVE.
    """
    max_q    = cr.cf_fixed_int_frac.shape[1]
    base_idx = cr.scenario_index("base")
    valid    = (np.arange(max_q)[None, :] < cr.cf_n_q[:, None]).astype(float)

    cf_masked = cr.cf_fixed_int_frac * valid
    pv_frac   = np.einsum("iq,iqs->is", cf_masked, cr.cohort_disc_q)
    use_eff   = np.any(np.abs(cr.eve_pv_frac) > 1e-18, axis=(1, 2))
    if use_eff.any():
        pv_frac[use_eff, :] = cr.eve_pv_frac[use_eff, :, :].sum(axis=1)
    pv_mat    = pv_frac * (balance * params.sign)[:, None]

    agg: dict[tuple, np.ndarray] = {}
    for i, (pc, sid, ccy) in enumerate(
        zip(params.product_code, params.bs_side, params.currency)
    ):
        k = (str(pc), str(sid), str(ccy))
        if k not in agg:
            agg[k] = np.zeros(cr.n_rate_scen, dtype=float)
        agg[k][0] += balance[i] * params.eve_pv_factor[i]
        for rs in range(cr.n_rate_scen):
            if rs != base_idx:
                agg[k][rs] += pv_mat[i, rs] - pv_mat[i, base_idx]

    # Fallback: cohorts with no CF schedule AND no eve_pv_frac (e.g. IRS product 0000).
    # Use delta_eve_unit calibrated from SQL — same as accuracy_check._approach_b_scen.
    blank = (~use_eff) & (cr.cf_n_q == 0)
    if blank.any():
        scen_p_idx = {str(s): i for i, s in enumerate(params.scenario_ids)}
        for i in np.where(blank)[0]:
            pc  = str(params.product_code[i])
            sid = str(params.bs_side[i])
            ccy = str(params.currency[i])
            k = (pc, sid, ccy)
            if k not in agg:
                agg[k] = np.zeros(cr.n_rate_scen, dtype=float)
                agg[k][0] += balance[i] * params.eve_pv_factor[i]
            for rs, scen in enumerate(cr.rate_scenario_ids):
                scen = str(scen)
                if scen == "base" or scen not in scen_p_idx:
                    continue
                agg[k][rs] += balance[i] * params.delta_eve_unit[i, scen_p_idx[scen]]
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Bias correction helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_nii_bias(
    nii_cf_by_product: dict[tuple, np.ndarray],
    exact_nii_base:    dict[tuple, float],
) -> dict[tuple, float]:
    """bias_nii[product] = exact_nii_base - fast_nii_cf_base.

    exact_nii_base: (product_code, bs_side, currency) -> exact base NII from pipeline.
    """
    base_idx = 0   # base is always index 0 in rate_scenario_ids
    bias: dict[tuple, float] = {}
    all_keys = set(nii_cf_by_product) | set(exact_nii_base)
    for k in all_keys:
        fast_b  = float(nii_cf_by_product[k][base_idx]) if k in nii_cf_by_product else 0.0
        exact_b = exact_nii_base.get(k, 0.0)
        bias[k] = exact_b - fast_b
    return bias


def apply_nii_bias(
    nii_cf_by_product: dict[tuple, np.ndarray],
    bias_nii:          dict[tuple, float],
) -> dict[tuple, np.ndarray]:
    """Add bias_nii to all scenarios for each product.

    The bias is assumed constant across scenarios (first-order correction).
    """
    corrected: dict[tuple, np.ndarray] = {}
    for k, arr in nii_cf_by_product.items():
        b = bias_nii.get(k, 0.0)
        corrected[k] = arr + b
    return corrected
