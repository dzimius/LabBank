"""lcr_fast.py
=============
Fast LCR computation for the optimization hot path.

LCR = HQLA / Net Cash Outflows  (per currency)

    HQLA         = Σ amounts[i] × hqla_factor[i]
                   (hqla_factor = 1 - haircut for HQLA assets, 0 otherwise)

    Outflows_30d = Σ amounts[i] × lcr_runoff[i]
                   (lcr_runoff = run-off rate for liabilities, 0 for assets)

    Inflows_30d  = min(0.75 × raw_inflows, 0.75 × outflows)
                   raw_inflows = Σ amounts[i] × inflow_30d_frac[i]  (assets only)

    Net Outflows = max(outflows - inflows, 0)

All regulatory parameters (hqla_factor, lcr_runoff, inflow_30d_frac) are
extracted once by extract_params.py and stored in product_params.npz.
This makes LCR a simple dot product per currency — no iteration over
individual schedules.

No pandas in the hot path.
"""
from __future__ import annotations

import numpy as np
from bs_vector import BalanceSheetParams


def compute_lcr_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> dict[str, float]:
    """Compute LCR per currency.

    Parameters
    ----------
    amounts : float64 (n,) — PLN balance per product
    params  : BalanceSheetParams

    Returns
    -------
    dict currency → LCR ratio (float, NaN if net_outflows == 0)
    """
    result = {}
    for ccy in params.unique_currencies:
        mask = params.currency == ccy

        hqla        = float(np.dot(amounts * mask, params.hqla_factor))
        outflows    = float(np.dot(amounts * mask, params.lcr_runoff))
        raw_inflows = float(np.dot(amounts * mask, params.inflow_30d_frac))

        # Basel III §33(d): inflows capped at 75% of outflows,
        # 75% recognition rate on raw inflows
        inflows     = min(0.75 * raw_inflows, 0.75 * outflows)
        net_outflows = max(outflows - inflows, 0.0)

        lcr = hqla / net_outflows if net_outflows > 0 else float("nan")
        result[ccy] = lcr

    return result


def compute_lcr_components_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    currency: str,
) -> dict[str, float]:
    """Detailed LCR components for a single currency.

    Returns dict with keys:
        hqla, outflows_30d, raw_inflows_30d, inflows_30d,
        net_outflows_30d, lcr
    """
    mask = params.currency == currency

    hqla        = float(np.dot(amounts * mask, params.hqla_factor))
    outflows    = float(np.dot(amounts * mask, params.lcr_runoff))
    raw_inflows = float(np.dot(amounts * mask, params.inflow_30d_frac))

    inflows      = min(0.75 * raw_inflows, 0.75 * outflows)
    net_outflows = max(outflows - inflows, 0.0)
    lcr          = hqla / net_outflows if net_outflows > 0 else float("nan")

    return {
        "hqla":              hqla,
        "outflows_30d":      outflows,
        "raw_inflows_30d":   raw_inflows,
        "inflows_30d":       inflows,
        "net_outflows_30d":  net_outflows,
        "lcr":               lcr,
    }


def lcr_constraint_value(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    currency: str = "PLN",
    min_lcr: float = 1.0,
) -> float:
    """Constraint value for optimizer: LCR(currency) - min_lcr ≥ 0.

    Returns positive value when constraint is satisfied.
    Used directly as scipy.optimize inequality constraint.
    """
    lcr_dict = compute_lcr_fast(amounts, params)
    lcr      = lcr_dict.get(currency, float("nan"))
    if np.isnan(lcr):
        return float("inf")   # no outflows → constraint trivially satisfied
    return lcr - min_lcr


def worst_lcr(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> tuple[float, str]:
    """Return (worst LCR, currency) across all currencies."""
    lcr_dict = compute_lcr_fast(amounts, params)
    worst_ccy = min(lcr_dict, key=lambda c: lcr_dict[c] if not np.isnan(lcr_dict[c]) else np.inf)
    return lcr_dict[worst_ccy], worst_ccy
