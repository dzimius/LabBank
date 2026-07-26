"""rwa_fast.py
=============
Fast RWA computation for the optimization hot path.

RWA = Σ amounts[i] × rwa_factor[i]   (assets only; liabilities/equity have rwa_factor = 0)

Exactly linear in balance amounts by construction.
"""
from __future__ import annotations

import numpy as np
from bs_vector import BalanceSheetParams


def compute_rwa_fast(
    amounts: np.ndarray,
    params: BalanceSheetParams,
) -> float:
    """Total Risk-Weighted Assets in PLN.

    Parameters
    ----------
    amounts : float64 (n,) — PLN balance per cohort
    params  : BalanceSheetParams

    Returns
    -------
    RWA in PLN
    """
    return float(np.dot(amounts, params.rwa_factor))
