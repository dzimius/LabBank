"""mc_scenario_basis.py
======================
Maps arbitrary Nelson-Siegel Monte Carlo curve shocks onto per-product NII
sensitivities, WITHOUT re-running the CF-based cohort rate-matrix pipeline
for every Monte Carlo draw.

Idea
----
Any smooth zero-rate shock Δr(t) used in this codebase decomposes into 3
independent tenor-shape functions (the same 3 that underlie the EBA/RTS
shock formulas in eba_shock_curves.py):
    f_const(t) = 1
    f_s(t)     = exp(-t)          (short-rate decay)
    f_l(t)     = 1 - exp(-t)      (long-rate decay)

The 6 official EBA scenarios are known, fixed-size combinations of these 3
shapes (par_up/dn = pure f_const; sr_up/dn = pure f_s; steep/flat = mixes of
f_s and f_l).  bs_optimizer._compute_lp_coefficients() already evaluated the
*exact* CF-based ΔNII per product for each of those 6 scenarios (A_nii_delta).
Recovering per-product sensitivity to the 3 underlying shapes is therefore a
small (over-determined) linear solve — no new cohort-level computation needed.

Once we have per-product sensitivity to {f_const, f_s, f_l}, any new NS draw
(itself a linear combination of f_const/f_s/f_l once re-expressed in that
basis) gets its ΔNII approximated by linear superposition — the same
first-order-linear assumption the rest of this optimizer already relies on
for its regulatory SOT constraints.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_IRRBB  = os.path.normpath(os.path.join(_HERE, "..", "..", "irrbb_calc", "python_code"))
if _IRRBB not in sys.path:
    sys.path.insert(0, _IRRBB)

from eba_shock_curves import get_shock_params, shock_bps_at_tenors  # noqa: E402

# Official scenarios used to recover shape sensitivities.
# par_up/par_dn -> pure f_const ; sr_up/sr_dn -> pure f_s ; steep/flat -> mix of f_s, f_l
_CONST_SCENS = ["par_up", "par_dn"]
_MIXED_SCENS = ["sr_up", "sr_dn", "steep", "flat"]


def _shape_coeffs(scenario: str, params: dict) -> tuple[float, float]:
    """(coeff on f_s, coeff on f_l) for one EBA scenario, from the Article-3 formula."""
    R_s, R_l = params["short"], params["long"]
    return {
        "sr_up":  (+R_s, 0.0),
        "sr_dn":  (-R_s, 0.0),
        "steep":  (-0.65 * R_s, +0.90 * R_l),
        "flat":   (+0.80 * R_s, -0.75 * R_l),
    }[scenario]


def recover_shape_sensitivities(
    A_nii_delta: np.ndarray,
    shocked_scens: list[str],
    currency: str = "PLN",
) -> np.ndarray:
    """Recover per-product ΔNII sensitivity to {f_const, f_s, f_l} (per bp of shape).

    Parameters
    ----------
    A_nii_delta   : (n_scen, n_prod) — from bs_optimizer._compute_lp_coefficients,
                    in "% T1 x 100 per unit x_j" units (same as used for SOT constraints).
    shocked_scens : scenario id order matching A_nii_delta's rows.

    Returns
    -------
    sens : (3, n_prod) rows = [sens_const, sens_fs, sens_fl], same units as A_nii_delta
           (per bp of the respective shape function).
    """
    params = get_shock_params(currency)
    idx = {s: i for i, s in enumerate(shocked_scens)}
    n_prod = A_nii_delta.shape[1]

    # ── constant (level) sensitivity: par_up (+R_par) / par_dn (-R_par) ──────────
    R_par = params["par"]
    coeff_const = np.array([[+R_par], [-R_par]])              # (2, 1)
    resp_const  = np.vstack([A_nii_delta[idx[s]] for s in _CONST_SCENS])  # (2, n_prod)
    sens_const  = np.linalg.lstsq(coeff_const, resp_const, rcond=None)[0]  # (1, n_prod)

    # ── short/long-decay sensitivity: sr_up/dn (pure f_s) + steep/flat (mixed) ──
    coeff_mixed = np.array([_shape_coeffs(s, params) for s in _MIXED_SCENS])  # (4, 2)
    resp_mixed  = np.vstack([A_nii_delta[idx[s]] for s in _MIXED_SCENS])     # (4, n_prod)
    sens_fs_fl  = np.linalg.lstsq(coeff_mixed, resp_mixed, rcond=None)[0]    # (2, n_prod)

    return np.vstack([sens_const, sens_fs_fl])  # (3, n_prod): [const, fs, fl]


def ns_basis_projection_matrix(tau: float, t_grid: np.ndarray) -> np.ndarray:
    """Q (3, 3): projects NS loadings {1, L1(t,tau), L2(t,tau)} onto {f_const, f_s, f_l}.

    (c_const, c_fs, c_fl) = (beta0, beta1, beta2) @ Q  for any NS shock,
    via ordinary least squares over the tenor grid (both bases are smooth,
    well-conditioned functions of tenor).
    """
    from ns_curve_model import ns_design_matrix  # local import: avoid circular path setup

    X_ns = ns_design_matrix(t_grid, tau)                 # (T, 3) = [1, L1, L2]
    f_s  = np.exp(-t_grid)
    f_l  = 1.0 - np.exp(-t_grid)
    X_eba = np.column_stack([np.ones_like(t_grid), f_s, f_l])  # (T, 3) = [1, f_s, f_l]

    # For each NS loading column, find its best-fit combination of {1, f_s, f_l}
    Q = np.linalg.lstsq(X_eba, X_ns, rcond=None)[0]      # (3, 3): rows f_const/f_s/f_l, cols beta0/beta1/beta2
    return Q.T  # (3, 3): rows beta0/beta1/beta2, cols const/fs/fl -> so beta @ Q gives (c_const,c_fs,c_fl)


def build_delta_nii_operator(
    A_nii_delta: np.ndarray,
    shocked_scens: list[str],
    tau: float,
    t_grid: np.ndarray,
    t1: float,
    currency: str = "PLN",
) -> np.ndarray:
    """R (3, n_prod): the linear map from an NS factor draw (d_beta0,d_beta1,d_beta2)
    in bps to a per-product PLN NII-delta coefficient vector.

    Usage: for draws (N, 3) -> delta_nii_matrix (N, n_prod) = draws @ R
    """
    sens = recover_shape_sensitivities(A_nii_delta, shocked_scens, currency)  # (3, n_prod)
    Q    = ns_basis_projection_matrix(tau, t_grid)                            # (3, 3)
    R    = (Q @ sens) * (t1 / 100.0)   # undo the "% T1 x 100" scaling -> PLN per unit x_j
    return R
