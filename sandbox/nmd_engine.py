"""nmd_engine.py
================
Non-Maturity Deposit (NMD) behavioral model stress engine.

Computes analytical delta NII and delta EVE from a change in the
outstanding-percentage vector loaded from dep_beh_models_ir.xlsx.

The math follows the same CF model used by the full pipeline:
    outstanding(k) = B × pct_prev(k)          pct_prev[0]=1, pct_prev[k]=pct[k-1]
    capital_cf(k)  = B × (pct_prev(k) - pct(k))       always ≥ 0
    interest_cf(k) = outstanding(k) × rate × period_yf(k)

NII sign convention (L deposit, sign=-1):
    delta_NII = sign × Σ_{k∈horizon} [d_interest(k) + d_capital(k) × renewal × remain_yf]

EVE sign convention:
    delta_EVE[s] = sign × Σ_k [(d_interest(k) + d_capital(k)) × disc_factor(k, s)]
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

NMD_PRODUCTS = {
    "6000": "Current Account",
    "8000": "Saving Account",
}

_HERE = os.path.dirname(os.path.abspath(__file__))
NMD_BEH_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "balance_gen_add_data", "input", "dep_beh_models_ir.xlsx")
)


def _tenor_to_yf(label: str) -> float:
    s = str(label).strip().upper()
    if s.endswith("D"):
        try:
            return int(s[:-1]) / 365.0
        except ValueError:
            return 0.0
    if s.endswith("M"):
        try:
            return int(s[:-1]) / 12.0
        except ValueError:
            return 0.0
    if s.endswith("Y"):
        try:
            return float(s[:-1])
        except ValueError:
            return 0.0
    return 0.0


def load_nmd_model(path: str | None = None) -> dict[str, pd.DataFrame]:
    """Load NMD behavioral models from dep_beh_models_ir.xlsx.

    Returns
    -------
    dict mapping product_code → DataFrame with columns:
        tenor   : str   e.g. '1D', '1M', ..., '75M'
        cum_yf  : float cumulative year fraction
        pct     : float outstanding fraction 0.0–1.0
    Only non-NaN rows are included; result is sorted by cum_yf ascending.
    """
    if path is None:
        path = NMD_BEH_PATH
    df = pd.read_excel(path)
    result: dict[str, pd.DataFrame] = {}
    for pc_str in NMD_PRODUCTS:
        col_int = int(pc_str)
        col = col_int if col_int in df.columns else pc_str
        if col not in df.columns:
            continue
        sub = df[["tenor", col]].dropna(subset=[col]).copy()
        sub.columns = ["tenor", "pct"]
        sub["cum_yf"] = sub["tenor"].apply(_tenor_to_yf)
        sub = sub.sort_values("cum_yf").reset_index(drop=True)
        result[pc_str] = sub
    return result


def compute_nmd_cashflows(
    balance: float,
    rate: float,        # annualised deposit rate (decimal, always positive)
    pct: np.ndarray,    # (K,) outstanding fractions at each tenor bucket end
    cum_yf: np.ndarray, # (K,) cumulative year fractions (sorted ascending)
) -> dict:
    """Compute NMD cash flow arrays for a single product.

    pct[k] = fraction of balance still outstanding at end of bucket k.
    pct_prev[0] = 1.0 (100% at inception), pct_prev[k] = pct[k-1].

    Returns
    -------
    dict with keys: outstanding, capital_cf, interest_cf, period_yf  — all (K,).
    """
    K = len(pct)
    period_yf = np.empty(K)
    period_yf[0] = float(cum_yf[0])
    if K > 1:
        period_yf[1:] = np.diff(cum_yf)

    pct_prev = np.empty(K)
    pct_prev[0] = 1.0
    if K > 1:
        pct_prev[1:] = pct[:-1]

    outstanding = balance * pct_prev
    capital_cf  = balance * (pct_prev - pct)
    interest_cf = outstanding * rate * period_yf

    return {
        "outstanding": outstanding,
        "capital_cf":  capital_cf,
        "interest_cf": interest_cf,
        "period_yf":   period_yf,
    }


def compute_nmd_delta(
    balance: float,
    rate: float,                      # deposit rate (positive decimal)
    sign: float,                      # -1 for L deposit, +1 for A
    pct_old: np.ndarray,              # (K,) baseline outstanding fractions
    pct_new: np.ndarray,              # (K,) stressed outstanding fractions
    cum_yf: np.ndarray,               # (K,) cumulative year fractions (sorted)
    curves,                           # CurveTensors
    currency: str,
    shocked_scenario_ids: list[str],  # e.g. ['par_up', 'par_dn', ...]
    renewal_rate: float = 0.0,        # new-business rate on returned capital
    horizon_yf: float = 1.0,          # NII horizon in years
) -> dict:
    """Compute delta NII and delta EVE from a change in the NMD outstanding-pct vector.

    Returns
    -------
    dict with keys:
        delta_nii      : float   total ΔNII within horizon (PLN)
        delta_eve_base : float   ΔEVE under base curve (PLN)
        delta_eve      : dict[str, float]  scenario_id → ΔEVE (PLN)
    """
    cf_old = compute_nmd_cashflows(balance, rate, pct_old, cum_yf)
    cf_new = compute_nmd_cashflows(balance, rate, pct_new, cum_yf)

    d_interest = cf_new["interest_cf"] - cf_old["interest_cf"]
    d_capital  = cf_new["capital_cf"]  - cf_old["capital_cf"]
    d_total    = d_interest + d_capital

    # ── NII delta (within 1-year horizon) ─────────────────────────────────────
    horizon_mask = cum_yf <= horizon_yf
    remain_yf    = np.maximum(0.0, horizon_yf - cum_yf)

    delta_nii = float(sign) * float(
        (d_interest * horizon_mask).sum()
        + (d_capital * horizon_mask * remain_yf * renewal_rate).sum()
    )

    # ── EVE delta (per scenario) ───────────────────────────────────────────────
    ccy_idx  = int(curves.currency_index(currency))
    n_months = int(curves.disc_factors.shape[2])

    # Month index for each CF bucket: round(cum_yf × 12) – 1, clamped to [0, n-1].
    # 1D row (cum_yf ≈ 0.0027) maps to month index 0.
    m_idx = np.round(cum_yf * 12).astype(int) - 1
    m_idx = np.clip(m_idx, 0, n_months - 1)

    def _disc(scen: str) -> np.ndarray:
        s_idx = int(curves.scenario_index(scen))
        return curves.disc_factors[s_idx, ccy_idx, m_idx]

    base_disc      = _disc("base")
    delta_eve_base = float(sign) * float((d_total * base_disc).sum())

    delta_eve: dict[str, float] = {}
    for scen in shocked_scenario_ids:
        try:
            disc_s = _disc(scen)
        except KeyError:
            disc_s = base_disc
        delta_eve[scen] = float(sign) * float((d_total * disc_s).sum())

    return {
        "delta_nii":       delta_nii,
        "delta_eve_base":  delta_eve_base,
        "delta_eve":       delta_eve,
    }
