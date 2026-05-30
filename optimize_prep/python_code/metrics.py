"""metrics.py
============
Unified entry point for the optimization hot path.

Computes all four regulatory metrics from a single weight vector in one call.
This is the function called by bs_optimization's objective and constraint
functions — it must be fast (< 5ms for a single evaluation).

Usage
-----
    from bs_vector import BalanceSheetParams, CurveTensors
    from metrics import AllMetrics, compute_all_metrics, load_params_and_curves

    params, curves = load_params_and_curves(
        params_path="optimize_prep/output/product_params.npz",
        curves_path="optimize_prep/output/curve_tensors.npz",
    )

    # weights: fraction of total_assets per product (same length as bs_structure rows)
    w = params.current_weights()
    metrics = compute_all_metrics(w, params, curves, total_assets=10_000_000_000)

    print(f"NII base: {metrics.nii_base:,.0f}")
    print(f"LCR PLN: {metrics.lcr['PLN']:.2%}")
    print(f"NSFR PLN: {metrics.nsfr['PLN']:.2%}")
    print(f"Worst delta_EVE: {metrics.worst_delta_eve:,.0f} ({metrics.worst_eve_scenario})")
    print(f"Worst delta_NII: {metrics.worst_delta_nii:,.0f} ({metrics.worst_nii_scenario})")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from bs_vector import BalanceSheetParams, CurveTensors, CohortRates
from nii_fast  import compute_nii_scenarios, compute_nii_all_scenarios
from eve_fast  import compute_eve_base_fast, compute_eve_all_scenarios, compute_worst_delta_eve
from lcr_fast  import compute_lcr_fast, worst_lcr
from nsfr_fast import compute_nsfr_fast, worst_nsfr


# ─────────────────────────────────────────────────────────────────────────────
# AllMetrics container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AllMetrics:
    """All four regulatory metrics computed from a single weight vector.

    Attributes
    ----------
    nii_base             : base NII in PLN (12M horizon)
    delta_nii            : dict scenario_id → delta_NII in PLN
    eve_base             : base EVE in PLN
    delta_eve            : dict scenario_id → delta_EVE in PLN
    lcr                  : dict currency → LCR ratio
    nsfr                 : dict currency → NSFR ratio
    worst_delta_eve      : minimum delta_EVE across all scenarios
    worst_eve_scenario   : scenario_id of worst_delta_eve
    worst_delta_nii      : minimum delta_NII across all scenarios
    worst_nii_scenario   : scenario_id of worst_delta_nii
    total_assets         : total assets used (PLN)
    """
    nii_base:           float
    delta_nii:          dict[str, float]
    eve_base:           float
    delta_eve:          dict[str, float]
    lcr:                dict[str, float]
    nsfr:               dict[str, float]
    worst_delta_eve:    float
    worst_eve_scenario: str
    worst_delta_nii:    float
    worst_nii_scenario: str
    total_assets:       float

    # ── EBA SOT ratios (computed on demand) ──────────────────────────────────
    def sot_nii_pct(self, scenario_id: str, tier1_capital: float) -> float:
        """delta_NII / T1 × 100. EBA threshold: > -5%."""
        return self.delta_nii.get(scenario_id, 0.0) / tier1_capital * 100.0

    def sot_eve_pct(self, scenario_id: str, tier1_capital: float) -> float:
        """delta_EVE / T1 × 100. EBA threshold: > -15%."""
        return self.delta_eve.get(scenario_id, 0.0) / tier1_capital * 100.0

    def worst_sot_eve_pct(self, tier1_capital: float) -> float:
        return self.worst_delta_eve / tier1_capital * 100.0

    def worst_sot_nii_pct(self, tier1_capital: float) -> float:
        return self.worst_delta_nii / tier1_capital * 100.0

    def is_feasible(
        self,
        tier1_capital: float,
        min_lcr:  float = 1.00,
        min_nsfr: float = 1.00,
        max_sot_eve_pct: float = -15.0,
        max_sot_nii_pct: float = -5.0,
    ) -> bool:
        """Return True if all regulatory constraints are satisfied."""
        for ccy_lcr in self.lcr.values():
            if not np.isnan(ccy_lcr) and ccy_lcr < min_lcr:
                return False
        for ccy_nsfr in self.nsfr.values():
            if not np.isnan(ccy_nsfr) and ccy_nsfr < min_nsfr:
                return False
        if self.worst_sot_eve_pct(tier1_capital) < max_sot_eve_pct:
            return False
        if self.worst_sot_nii_pct(tier1_capital) < max_sot_nii_pct:
            return False
        return True

    def constraint_slack(
        self,
        tier1_capital: float,
        min_lcr:  float = 1.00,
        min_nsfr: float = 1.00,
        max_sot_eve_pct: float = -15.0,
        max_sot_nii_pct: float = -5.0,
    ) -> dict[str, float]:
        """Slack for each constraint (positive = satisfied).

        Useful for diagnostics and penalty-function approaches.
        """
        slacks = {}
        for ccy, lcr in self.lcr.items():
            slacks[f"lcr_{ccy}"] = (lcr - min_lcr) if not np.isnan(lcr) else float("inf")
        for ccy, nsfr in self.nsfr.items():
            slacks[f"nsfr_{ccy}"] = (nsfr - min_nsfr) if not np.isnan(nsfr) else float("inf")
        slacks["sot_eve"] = self.worst_sot_eve_pct(tier1_capital) - max_sot_eve_pct
        slacks["sot_nii"] = self.worst_sot_nii_pct(tier1_capital) - max_sot_nii_pct
        return slacks


# ─────────────────────────────────────────────────────────────────────────────
# Main computation function (called in optimizer hot path)
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    weights: np.ndarray,
    params: BalanceSheetParams,
    curves: CurveTensors,
    total_assets: float,
) -> AllMetrics:
    """Compute all four metrics from a weight vector.

    This is the single function called by the optimizer on every evaluation.
    Target: < 5 ms per call.

    Parameters
    ----------
    weights      : float64 (n,) — fraction of total_assets per product.
                   Caller must ensure Σ_assets(weights) ≈ 1.0 and
                   Σ_liabilities+equity(weights) ≈ 1.0.
    params       : BalanceSheetParams — loaded once at optimizer startup
    curves       : CurveTensors — loaded once at optimizer startup
    total_assets : total balance sheet size in PLN

    Returns
    -------
    AllMetrics dataclass with all computed values.
    """
    amounts = weights * total_assets

    # ── NII ───────────────────────────────────────────────────────────────────
    # Use calibrated delta_nii_unit (curves=None) — exact at the calibration
    # point and linear in amounts, matching the IRRBB pipeline for all shock shapes.
    # Schedule-based (curves provided) diverges for non-parallel shocks.
    nii_scenarios = compute_nii_all_scenarios(amounts, params, curves=None)
    nii_base       = nii_scenarios.pop("base")
    delta_nii      = nii_scenarios

    worst_dnii     = min(delta_nii.values()) if delta_nii else 0.0
    worst_nii_scen = min(delta_nii, key=delta_nii.get) if delta_nii else ""

    # ── EVE ───────────────────────────────────────────────────────────────────
    eve_scenarios  = compute_eve_all_scenarios(amounts, params)
    eve_base       = eve_scenarios.pop("base")
    delta_eve      = eve_scenarios

    worst_deve, worst_eve_scen = compute_worst_delta_eve(amounts, params)

    # ── LCR ───────────────────────────────────────────────────────────────────
    lcr  = compute_lcr_fast(amounts, params)

    # ── NSFR ──────────────────────────────────────────────────────────────────
    nsfr = compute_nsfr_fast(amounts, params)

    return AllMetrics(
        nii_base           = nii_base,
        delta_nii          = delta_nii,
        eve_base           = eve_base,
        delta_eve          = delta_eve,
        lcr                = lcr,
        nsfr               = nsfr,
        worst_delta_eve    = worst_deve,
        worst_eve_scenario = worst_eve_scen,
        worst_delta_nii    = worst_dnii,
        worst_nii_scenario = worst_nii_scen,
        total_assets       = total_assets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────────────

def load_params_and_curves(
    params_path: str | None = None,
    curves_path: str | None = None,
) -> tuple[BalanceSheetParams, CurveTensors]:
    """Load BalanceSheetParams and CurveTensors from disk.

    Defaults to output/ relative to this file's directory.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "output")

    if params_path is None:
        params_path = os.path.join(out_dir, "product_params.npz")
    if curves_path is None:
        curves_path = os.path.join(out_dir, "curve_tensors.npz")

    params = BalanceSheetParams.load(params_path)
    curves = CurveTensors.load(curves_path)
    return params, curves


def load_cohort_rates(params_path: str | None = None) -> CohortRates:
    """Load CohortRates (cr_* arrays) from product_params.npz."""
    here = os.path.dirname(os.path.abspath(__file__))
    if params_path is None:
        params_path = os.path.join(here, "..", "output", "product_params.npz")
    return CohortRates.load(params_path)
