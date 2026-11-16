"""nsfr_fast.py
==============
Fast NSFR computation for the optimization hot path.

NSFR = ASF / RSF  (per currency)

    ASF = Σ amounts[i] × asf_factor[i]   (liabilities + equity only)
    RSF = Σ amounts[i] × rsf_factor[i]   (assets only)

NSFR is already exactly linear in balance amounts in the existing pipeline.
This module is a pure-numpy wrapper — no approximation involved.

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
    NSFR is informational (the pipeline report shows it), not an optimizer constraint."""
    if ccy == home_ccy():
        return np.ones(len(params.currency), dtype=bool)
    return params.currency == ccy


def _ratio_currencies(params: BalanceSheetParams) -> list[str]:
    """Currencies with a constrained ratio: the home currency (consolidated) only."""
    return [home_ccy()] if home_ccy() in params.unique_currencies else list(params.unique_currencies)

def compute_nsfr_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> dict[str, float]:
    """Compute NSFR per currency.

    Parameters
    ----------
    amounts : float64 (n,) — PLN balance per product
    params  : BalanceSheetParams

    Returns
    -------
    dict currency → NSFR ratio (float, NaN if RSF == 0)
    """
    result = {}
    for ccy in _ratio_currencies(params):
        mask = _ccy_mask(params, ccy)

        asf  = float(np.dot(amounts * mask, params.asf_factor))
        rsf  = float(np.dot(amounts * mask, params.rsf_factor))
        if ccy == home_ccy():          # standalone committed facilities (consolidated ratio)
            rsf += params.commit_rsf * params.commit_scale(amounts)
        nsfr = asf / rsf if rsf > 0 else float("nan")
        result[ccy] = nsfr

    return result


def compute_nsfr_components_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    currency: str,
) -> dict[str, float]:
    """Detailed NSFR components for a single currency.

    Returns dict with keys: asf, rsf, nsfr
    """
    mask = _ccy_mask(params, currency)
    asf  = float(np.dot(amounts * mask, params.asf_factor))
    rsf  = float(np.dot(amounts * mask, params.rsf_factor))
    if currency == home_ccy():
        rsf += params.commit_rsf * params.commit_scale(amounts)
    nsfr = asf / rsf if rsf > 0 else float("nan")
    return {"asf": asf, "rsf": rsf, "nsfr": nsfr}


def nsfr_constraint_value(
    amounts: np.ndarray,
    params: BalanceSheetParams,
    currency: str | None = None,
    min_nsfr: float = 1.0,
) -> float:
    """Constraint value for optimizer: NSFR(currency) - min_nsfr ≥ 0.

    Returns positive value when constraint is satisfied.
    Used directly as scipy.optimize inequality constraint.
    """
    currency = currency or home_ccy()
    nsfr_dict = compute_nsfr_fast(amounts, params)
    nsfr      = nsfr_dict.get(currency, float("nan"))
    if np.isnan(nsfr):
        return float("inf")
    return nsfr - min_nsfr


def worst_nsfr(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> tuple[float, str]:
    """Return (worst NSFR, currency) across all currencies."""
    nsfr_dict = compute_nsfr_fast(amounts, params)
    worst_ccy = min(nsfr_dict, key=lambda c: nsfr_dict[c] if not np.isnan(nsfr_dict[c]) else np.inf)
    return nsfr_dict[worst_ccy], worst_ccy
