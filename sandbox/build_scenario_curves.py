"""build_scenario_curves.py
=========================
Generate stylized yield curve scenarios for the sandbox visualisation.

Produces sandbox/scenario_curves.npz with:
  - 15 forward-rate curves (5 shapes × 3 levels) + discount-factor arrays.
  - Per-product IRRBB unit metrics for each curve:
      hyp_nii_unit_rate  (15, n)      NII per PLN at hypothetical base
      hyp_delta_nii_unit (15, n, S)   floor-adjusted delta NII per shock
      hyp_eve_pv_factor  (15, n)      EVE PV factor at hypothetical base
      hyp_delta_eve_unit (15, n, S)   CF-based delta EVE per shock
      hyp_shock_ids      (S,)         EBA shock scenario IDs

delta_NII is computed with 0%-floor clipping at the hypothetical rate level so
asymmetric floor effects (e.g. term deposits in low-rate environments) are
captured correctly.  delta_EVE uses full CF re-discounting under the
hypothetical CurveTensors.

Run from the bank_project root:
    python sandbox/build_scenario_curves.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

# ── tenor grid ────────────────────────────────────────────────────────────────
N_MONTHS = 360
T_YR     = np.arange(1, N_MONTHS + 1) / 12.0   # 1/12 … 30 years

# ── shape catalogue ───────────────────────────────────────────────────────────
CURVE_TYPES  = ["normal", "steep", "humped", "flat", "inverted"]
LEVELS       = ["low", "medium", "high"]
LEVEL_ANCHOR = {"low": 0.005, "medium": 0.025, "high": 0.050}

CURVE_LABELS = {
    "normal":   "Normal (rising)",
    "steep":    "Steep",
    "humped":   "Humped",
    "flat":     "Flat",
    "inverted": "Inverted",
}
LEVEL_LABELS = {
    "low":    "Low  (~0.5%)",
    "medium": "Medium (~2.5%)",
    "high":   "High  (~5.0%)",
}


# ── shape functions ───────────────────────────────────────────────────────────

def _normal(anchor: float, t: np.ndarray) -> np.ndarray:
    return anchor + 0.015 * (1.0 - np.exp(-t / 8.0))

def _steep(anchor: float, t: np.ndarray) -> np.ndarray:
    return anchor + 0.040 * (1.0 - np.exp(-t / 5.0))

def _humped(anchor: float, t: np.ndarray) -> np.ndarray:
    tau = 4.0
    return anchor + 0.030 * (t / tau) * np.exp(1.0 - t / tau)

def _flat(anchor: float, t: np.ndarray) -> np.ndarray:
    return np.full_like(t, anchor)

def _inverted(anchor: float, t: np.ndarray) -> np.ndarray:
    short_end = anchor + 0.020
    long_end  = max(anchor - 0.010, 0.0)
    return long_end + (short_end - long_end) * np.exp(-t / 5.0)

_SHAPE_FN = {
    "normal": _normal, "steep": _steep, "humped": _humped,
    "flat": _flat, "inverted": _inverted,
}

def _fwd_to_disc(fwd: np.ndarray) -> np.ndarray:
    return np.exp(-np.cumsum(fwd / 12.0))


# ── curve generation ──────────────────────────────────────────────────────────

def build_scenario_curves() -> dict:
    fwd_list, disc_list, scen_ids = [], [], []
    for ctype in CURVE_TYPES:
        fn = _SHAPE_FN[ctype]
        for lvl in LEVELS:
            anchor = LEVEL_ANCHOR[lvl]
            fwd    = np.clip(fn(anchor, T_YR), 0.0, None)
            disc   = _fwd_to_disc(fwd)
            fwd_list.append(fwd)
            disc_list.append(disc)
            scen_ids.append(f"{ctype}_{lvl}")
    return {
        "fwd_rates":    np.array(fwd_list,  dtype=float),   # (15, 360)
        "disc_factors": np.array(disc_list, dtype=float),   # (15, 360)
        "scenario_ids": np.array(scen_ids,  dtype=object),  # (15,)
        "curve_types":  np.array(CURVE_TYPES,  dtype=object),
        "levels":       np.array(LEVELS,       dtype=object),
        "n_months":     np.array([N_MONTHS]),
    }


# ── IRRBB pre-computation ─────────────────────────────────────────────────────

def build_hyp_irrbb_metrics(params, cr, curves, sc_data: dict) -> dict:
    """Pre-compute per-product IRRBB unit metrics for all 15 hypothetical curves.

    NII base rate
    -------------
    Level-shifted from the market calibration using the floor-clipped client
    rate formula:  clip(coeff_a * fwd_hyp + coeff_b, floor, cap).

    delta_NII
    ---------
    Floor-adjusted at the hypothetical base level.  For a term deposit with a
    0% floor in a low-rate environment a downward shock correctly contributes
    little or zero delta, unlike the calibrated linear approximation.

    EVE PV factor + delta_EVE
    -------------------------
    Full CF re-discounting using the hypothetical CurveTensors (same cashflow
    fractions, new discount factors).  Preserves the existing bias correction
    anchor for the base EVE level.

    Returns arrays shaped (n_hyp, n_products, …) for direct npz storage.
    """
    from nii_eve_cf_fast import (
        _eve_cf_cohort_matrix,
        _apply_fixed_rate_calibration,
        _apply_calibrated_delta_fallback,
        _apply_base_nii_bias,
    )
    from hyp_engine import build_hyp_curve_tensors, build_hyp_rate_matrix

    n   = len(params.product_code)
    n_m = curves.n_months

    p_scen_lst  = [str(s) for s in params.scenario_ids]
    cr_scen_lst = [str(s) for s in cr.rate_scenario_ids]
    cr_base_idx = cr_scen_lst.index("base")

    # EBA shock IDs: curves.scenario_ids also carries own_100_dn/up and
    # own_1_dn/up (curve-shape-only sensitivity/PV01 scenarios), but those
    # were never run through the NII/EVE calibration pipeline -- they don't
    # exist in cr.rate_scenario_ids at all. Restrict to scenarios that are
    # actually calibrated, or every lookup below silently falls through to
    # its `except: pass` and reports a misleading 0.00% "no impact" instead
    # of "not computed" (2026-08-15 fix).
    shock_ids   = [s for s in cr_scen_lst if s != "base"]
    n_shock     = len(shock_ids)

    # Baseline balance — needed for NII and EVE per-unit computation
    base_bal = params.balance_arr.copy()
    safe_bal = np.where(base_bal > 0, base_bal, 1.0)

    # Blank EVE rows: no CF schedule, no pre-stored eve_pv_frac → calibrated fallback
    blank_eve = (cr.cf_n_q == 0) & ~np.any(np.abs(cr.eve_pv_frac) > 1e-18, axis=(1, 2))

    n_hyp = len(sc_data["scenario_ids"])
    all_nii_unit       = np.zeros((n_hyp, n),          dtype=float)
    all_delta_nii_unit = np.zeros((n_hyp, n, n_shock),  dtype=float)
    all_eve_pv_factor  = np.zeros((n_hyp, n),          dtype=float)
    all_delta_eve_unit = np.zeros((n_hyp, n, n_shock),  dtype=float)

    for i, sid in enumerate(sc_data["scenario_ids"]):
        print(f"  [{i+1:2d}/{n_hyp}] {sid} ...", end="", flush=True)
        fwd_hyp    = sc_data["fwd_rates"][i]            # (360,) decimal
        hyp_curves = build_hyp_curve_tensors(curves, fwd_hyp)

        # ── CF-based NII at hyp curves ───────────────────────────────────────
        hyp_rm, hyp_rnm = build_hyp_rate_matrix(params, cr, hyp_curves)

        amount_x_out = (base_bal * params.sign)[:, None] * params.cohort_interest_yf_m
        nii_c        = np.einsum("im,ims->is", amount_x_out, hyp_rm)

        cap_x_remain = (base_bal * params.sign)[:, None] * params.cohort_capital_remain_m
        nii_r        = np.einsum("im,ims->is", cap_x_remain, hyp_rnm)

        nii_total = nii_c + nii_r
        _apply_fixed_rate_calibration(nii_total, base_bal, params, cr)
        _apply_calibrated_delta_fallback(nii_total, base_bal, params, cr)
        # Capture base NII before bias anchoring so it reflects the hypothetical
        # curve level, not the calibrated params.nii_unit_rate.  The bias is only
        # needed to keep shocked deltas consistent with the exact calibration.
        all_nii_unit[i] = nii_total[:, cr_base_idx] / safe_bal
        _apply_base_nii_bias(nii_total, base_bal, params, cr)
        for k, scen in enumerate(shock_ids):
            try:
                s_cr = cr_scen_lst.index(scen)
                all_delta_nii_unit[i, :, k] = (
                    nii_total[:, s_cr] - nii_total[:, cr_base_idx]
                ) / safe_bal
            except ValueError:
                pass

        # ── EVE CF re-discounting at hyp curves ──────────────────────────────
        eve_mat = _eve_cf_cohort_matrix(base_bal, params, cr, hyp_curves,
                                         use_precomputed_frac=False)  # (n, n_cr_scen)

        # EVE PV factor per unit balance (anchored to CF computation at hyp base)
        all_eve_pv_factor[i] = eve_mat[:, cr_base_idx] / safe_bal

        # delta EVE per shock
        for k, scen in enumerate(shock_ids):
            try:
                s_cr = cr_scen_lst.index(scen)
                all_delta_eve_unit[i, :, k] = (
                    (eve_mat[:, s_cr] - eve_mat[:, cr_base_idx]) / safe_bal
                )
            except ValueError:
                try:
                    calib = params.delta_eve_unit[:, p_scen_lst.index(scen)]
                    all_delta_eve_unit[i, :, k] = calib
                except ValueError:
                    pass

        # Fallback for blank EVE rows (no CF, no eve_pv_frac)
        if blank_eve.any():
            for k, scen in enumerate(shock_ids):
                try:
                    all_delta_eve_unit[i, blank_eve, k] = (
                        params.delta_eve_unit[blank_eve, p_scen_lst.index(scen)]
                    )
                except ValueError:
                    pass

        print(" done")

    return {
        "hyp_nii_unit_rate":  all_nii_unit,          # (15, n)
        "hyp_delta_nii_unit": all_delta_nii_unit,    # (15, n, S)
        "hyp_eve_pv_factor":  all_eve_pv_factor,     # (15, n)
        "hyp_delta_eve_unit": all_delta_eve_unit,    # (15, n, S)
        "hyp_shock_ids":      np.array(shock_ids, dtype=object),  # (S,)
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _HERE    = os.path.dirname(os.path.abspath(__file__))
    _ROOT    = os.path.abspath(os.path.join(_HERE, ".."))
    _OPTCODE = os.path.join(_ROOT, "optimize_prep", "python_code")
    for _p in [_HERE, _OPTCODE]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from bs_vector import BalanceSheetParams, CohortRates, CurveTensors  # noqa: E402

    PARAMS_NPZ = os.path.join(_ROOT, "optimize_prep", "output", "product_params.npz")
    CURVES_NPZ = os.path.join(_ROOT, "optimize_prep", "output", "curve_tensors.npz")
    OUT_PATH   = os.path.join(_HERE, "scenario_curves.npz")

    print("Building scenario curves...")
    sc_data = build_scenario_curves()
    print(f"  {len(sc_data['scenario_ids'])} scenarios: {list(sc_data['scenario_ids'])}")

    print("\nLoading pipeline params & curves...")
    params = BalanceSheetParams.load(PARAMS_NPZ)
    cr     = CohortRates.load(PARAMS_NPZ)
    curves = CurveTensors.load(CURVES_NPZ)
    print(f"  Products: {len(params.product_code)}"
          f"  |  EBA scenarios: {[s for s in curves.scenario_ids if s != 'base']}")

    print("\nPre-computing IRRBB unit metrics for all hypothetical curves...")
    irrbb = build_hyp_irrbb_metrics(params, cr, curves, sc_data)
    print(f"\n  hyp_nii_unit_rate:  {irrbb['hyp_nii_unit_rate'].shape}")
    print(f"  hyp_delta_nii_unit: {irrbb['hyp_delta_nii_unit'].shape}")
    print(f"  hyp_eve_pv_factor:  {irrbb['hyp_eve_pv_factor'].shape}")
    print(f"  hyp_delta_eve_unit: {irrbb['hyp_delta_eve_unit'].shape}")
    print(f"  hyp_shock_ids:      {list(irrbb['hyp_shock_ids'])}")

    # cohort_id lets sandbox/app.py detect a stale cache (product_params.npz
    # regenerated with a different cohort set) and fall back to the linear
    # approximation instead of a shape-mismatch crash — see ftp_store.py's
    # load_ftp_rates() for the same convention.
    out = {**sc_data, **irrbb, "cohort_id": params.cohort_id}
    np.savez(OUT_PATH, **out)
    mb = sum(v.nbytes for v in out.values() if hasattr(v, "nbytes")) / 1e6
    print(f"\nSaved -> {OUT_PATH}  (~{mb:.1f} MB)")
