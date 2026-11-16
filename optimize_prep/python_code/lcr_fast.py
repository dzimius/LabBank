"""lcr_fast.py
=============
Fast LCR computation for the optimization hot path.

LCR = HQLA / Net Cash Outflows  (per currency)

    HQLA         = Σ amounts[i] × hqla_factor[i]
                   (hqla_factor = 1 - haircut for HQLA assets, 0 otherwise)

    Outflows_30d = Σ amounts[i] × lcr_runoff[i]
                   (lcr_runoff = run-off rate for liabilities, 0 for assets)

    Inflows_30d  = min(raw_inflows, 0.75 × outflows)   (raw already counterparty-weighted)
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
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from labbank_common.bank_config import home_ccy



def _ccy_mask(params: BalanceSheetParams, ccy: str) -> np.ndarray:
    """Rows counted for ``ccy``. The home currency's ratio is the CONSOLIDATED one (every
    currency, amounts already in home currency): that is the ratio the regulatory minimum
    applies to. A foreign currency (e.g. USD in the EUR bank) gets only its own rows; its
    LCR is informational (the pipeline report shows it), not an optimizer constraint."""
    if ccy == home_ccy():
        return np.ones(len(params.currency), dtype=bool)
    return params.currency == ccy


def _ratio_currencies(params: BalanceSheetParams) -> list[str]:
    """Currencies with a constrained ratio: the home currency (consolidated) only."""
    return [home_ccy()] if home_ccy() in params.unique_currencies else list(params.unique_currencies)

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
    for ccy in _ratio_currencies(params):
        mask = _ccy_mask(params, ccy)

        hqla        = float(np.dot(amounts * mask, params.hqla_factor))
        outflows    = float(np.dot(amounts * mask, params.lcr_runoff))
        if ccy == home_ccy():          # standalone committed facilities (consolidated ratio)
            outflows += params.commit_lcr_outflow * params.commit_scale(amounts)
        raw_inflows = float(np.dot(amounts * mask, params.inflow_30d_frac))

        # inflow_30d_frac is already counterparty-weighted (lcr_rules.inflow_rate);
        # LCR DA Art. 33: recognised inflows capped at 75% of outflows
        inflows     = min(raw_inflows, 0.75 * outflows)
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
    mask = _ccy_mask(params, currency)

    hqla        = float(np.dot(amounts * mask, params.hqla_factor))
    outflows    = float(np.dot(amounts * mask, params.lcr_runoff))
    if currency == home_ccy():
        outflows += params.commit_lcr_outflow * params.commit_scale(amounts)
    raw_inflows = float(np.dot(amounts * mask, params.inflow_30d_frac))

    inflows      = min(raw_inflows, 0.75 * outflows)
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
    currency: str | None = None,
    min_lcr: float = 1.0,
) -> float:
    """Constraint value for optimizer: LCR(currency) - min_lcr ≥ 0.

    Returns positive value when constraint is satisfied.
    Used directly as scipy.optimize inequality constraint.
    """
    currency = currency or home_ccy()
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
