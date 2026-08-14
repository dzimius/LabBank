"""ep_fast.py
=============
Economic Profit (EP) decomposition at an arbitrary balance-sheet mix,
reusable outside the optimizer's own LP loop.

Ported from bs_optimization/notebooks/optimization_report.ipynb's
compute_ep_components() helper (Sections 4/5/9) -- same formulas
(EP = Margin(over FTP) + Fee - EL - CoC - OpEx - AcqCost), just parameterised
instead of closing over notebook-kernel globals, so it can be imported by
LabBank's Metrics tab to show a baseline-vs-modified EP waterfall the same
instant way NII/EVE/LCR/NSFR already work there.

Moved here from bs_optimization/python_code/ (2026-08-14) so sandbox/app.py
never has to import from bs_optimization/ -- everything this module needs
(product_map, nii_eve_cf_fast, rwa_fast, ftp_store) already lives in this
package.
"""
from __future__ import annotations

import numpy as np

from product_map import _ProductMap, OPEX_RATE
from nii_eve_cf_fast import compute_nii_cf_all
from rwa_fast import compute_rwa_fast
from ftp_store import load_ftp_rates, margin_unit_rate


def build_ep_context(params) -> tuple[_ProductMap, np.ndarray]:
    """One-time-per-book setup: product<->cohort map + margin-over-FTP rate
    per cohort. Cache the result (e.g. st.cache_data) -- cheap but not free,
    and both are pure functions of `params` (product_params.npz)."""
    pm = _ProductMap(params)
    ftp_rate = load_ftp_rates(params.cohort_id)
    if ftp_rate is None:
        ftp_rate = np.zeros_like(params.nii_unit_rate)
    margin_rate = margin_unit_rate(params.nii_unit_rate, ftp_rate, params.bs_side)
    return pm, margin_rate


def compute_ep_components(amounts, pm, params, cr, margin_rate, mask_irs=False):
    """EP decomposition (nii/ftp/margin/fee/el/coc/opex/acq_cost/ep) at
    per-cohort `amounts` (PLN, one entry per params.cohort_id row) -- the
    representation sandbox/app.py's balance-sheet editor already produces
    (see baseline.compute_weights), so no product-level weight vector needs
    building by the caller.

    Internally aggregates `amounts` up to product level (via pm.cohort_to_prod)
    only for the two genuinely product-level EP terms: the price-volume
    elasticity correction and AcqCost (marketing cost, charged only on
    balance growth above `pm.base_prod_w`, i.e. zero when `amounts` matches
    the baseline book exactly).

    mask_irs=True zeroes IRS's (product '0000') contribution to the
    informational NII/FTP figures only, matching how EVE/NII SOT constraints
    treat the always-pinned IRS book elsewhere in bs_optimizer.py. Margin/
    Fee/EL/CoC/OpEx/AcqCost are hedge-view-invariant by the same convention.
    """
    amounts = np.asarray(amounts, dtype=float)
    if mask_irs:
        irs_mask = np.array([str(pc) == "0000" for pc in params.product_code])
        amounts_nii = np.where(irs_mask, 0.0, amounts)
    else:
        amounts_nii = amounts
    nii = compute_nii_cf_all(amounts_nii, params, cr)["base"]

    x_w_prod = np.zeros(pm.n_prod, dtype=float)
    np.add.at(x_w_prod, pm.cohort_to_prod, amounts / params.total_assets)

    elast_corr = (float(np.dot(pm.vol_elast_prod * (x_w_prod - pm.base_prod_w), x_w_prod))
                  * params.total_assets)
    margin = float(np.dot(amounts, margin_rate)) + elast_corr
    fee = float(np.dot(amounts, params.fee_unit_rate))
    ftp = margin - nii
    rwa = compute_rwa_fast(amounts, params)
    el = float(np.dot(amounts, params.el_unit))
    coc = rwa * params.cet1_target * params.coc_rate
    opex = OPEX_RATE * float(np.sum(amounts[~params.is_equity]))
    acq_cost = (float(np.dot(pm.acq_cost_prod, np.maximum(0.0, x_w_prod - pm.base_prod_w)))
                * params.total_assets)
    ep = margin + fee - el - coc - opex - acq_cost
    return dict(nii=nii, ftp=ftp, margin=margin, fee=fee, el=el, coc=coc,
                opex=opex, acq_cost=acq_cost, ep=ep)
