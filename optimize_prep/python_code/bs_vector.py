"""bs_vector.py
==============
BalanceSheetParams — frozen per-product parameter arrays loaded from
product_params.npz, plus CurveTensors loaded from curve_tensors.npz.

These two objects are created ONCE at optimizer startup and passed into
every fast metric call.  All data is pure NumPy — no pandas, no SQL.

Usage
-----
    from bs_vector import BalanceSheetParams, CurveTensors

    params = BalanceSheetParams.load("optimize_prep/output/product_params.npz")
    curves = CurveTensors.load("optimize_prep/output/curve_tensors.npz")

    # current weights (sum to 1.0 over assets, 1.0 over liabilities)
    w = params.bs_pct_current / 100.0

    # amounts per product at given total_assets
    amounts = params.weights_to_amounts(w, total_assets=10_000_000_000)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# BalanceSheetParams
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BalanceSheetParams:
    """Frozen parameter snapshot for all n products (bs_structure rows).

    All float arrays have shape (n,).
    delta_nii_unit and delta_eve_unit have shape (n, S) where S = #scenarios.
    """
    # ── identification (object arrays, not used in hot path) ──────────────────
    product_code: np.ndarray      # str, (n,)
    product_name: np.ndarray      # str, (n,)
    bs_side:      np.ndarray      # str, (n,)  'A' | 'L' | 'E'
    currency:     np.ndarray      # str, (n,)
    sign:         np.ndarray      # float64, (n,)  +1 for A, -1 for L/E

    # ── weights ───────────────────────────────────────────────────────────────
    bs_pct_current: np.ndarray    # float64, (n,)  current bs_percentage values

    # ── NII ───────────────────────────────────────────────────────────────────
    nii_unit_rate:  np.ndarray    # float64, (n,)   NII per PLN balance (base)
    delta_nii_unit: np.ndarray    # float64, (n, S) delta_NII per PLN per scenario

    # ── EVE ───────────────────────────────────────────────────────────────────
    eve_pv_factor:  np.ndarray    # float64, (n,)   PV / balance (base)
    d_mod:          np.ndarray    # float64, (n,)   modified duration (signed, years)
    delta_eve_unit: np.ndarray    # float64, (n, S) delta_EVE per PLN per scenario

    # ── LCR / NSFR ────────────────────────────────────────────────────────────
    hqla_factor:    np.ndarray    # float64, (n,)   (1-haircut) for HQLA assets, 0 elsewhere
    lcr_runoff:     np.ndarray    # float64, (n,)   LCR run-off rate (liabilities), 0 for assets
    asf_factor:     np.ndarray    # float64, (n,)   ASF factor (liabilities/equity), 0 for assets
    rsf_factor:     np.ndarray    # float64, (n,)   RSF factor (assets), 0 for liabilities
    inflow_30d_frac: np.ndarray   # float64, (n,)   30-day inflow fraction (assets only)
    amort_frac_1y:  np.ndarray    # float64, (n,)   1Y amortisation fraction

    # ── rate coefficients ─────────────────────────────────────────────────────
    coeff_a:        np.ndarray    # float64, (n,)
    coeff_b:        np.ndarray    # float64, (n,)  decimal spread
    client_floor:   np.ndarray    # float64, (n,)  rate floor (decimal)
    client_cap:     np.ndarray    # float64, (n,)  rate cap (decimal)

    # ── repricing ─────────────────────────────────────────────────────────────
    repricing_tenor_m: np.ndarray  # float64, (n,)  months to first repricing

    # ── scenario labels ───────────────────────────────────────────────────────
    scenario_ids: np.ndarray      # str, (S,)  e.g. ['par_up', 'par_dn', ...]

    # ── metadata ──────────────────────────────────────────────────────────────
    total_assets: float
    report_date:  str
    balance_arr:  np.ndarray      # float64, (n,) actual balance amounts at extraction

    # ── derived masks (not stored in npz, computed on load) ──────────────────
    is_asset:    np.ndarray = field(default=None, compare=False)   # bool (n,)
    is_liability: np.ndarray = field(default=None, compare=False)  # bool (n,)
    is_equity:   np.ndarray = field(default=None, compare=False)   # bool (n,)

    def __post_init__(self):
        # computed masks — must bypass frozen=True via object.__setattr__
        object.__setattr__(self, "is_asset",    self.bs_side == "A")
        object.__setattr__(self, "is_liability", self.bs_side == "L")
        object.__setattr__(self, "is_equity",   self.bs_side == "E")

    # ── unique currencies in the book ─────────────────────────────────────────
    @property
    def unique_currencies(self) -> list[str]:
        return sorted(set(self.currency.tolist()))

    def scenario_index(self, scenario_id: str) -> int:
        """Return column index for scenario_id in delta_* arrays."""
        idx = np.where(self.scenario_ids == scenario_id)[0]
        if len(idx) == 0:
            raise KeyError(f"Scenario '{scenario_id}' not found. "
                           f"Available: {list(self.scenario_ids)}")
        return int(idx[0])

    # ── weight ↔ amount conversion ─────────────────────────────────────────────
    def weights_to_amounts(self, weights: np.ndarray, total_assets: float) -> np.ndarray:
        """Convert weight fractions to PLN amounts.

        weights[i] = fraction of total_assets allocated to product i.
        For assets: Σ w_assets = 1.0 means 100% of total_assets.
        For liabilities/equity: Σ w_liab = 1.0 means 100% of total_assets.

        The exact pipeline uses bs_percentage / 100 × total_balance where
        total_balance = 2 × total_assets, but that is baked into bs_percentage
        already (assets sum to 50%, liabilities sum to 50% of total_balance).
        Here weights are fractions of total_assets directly.

        Returns amounts array (n,) in PLN.
        """
        return weights * total_assets

    def pct_to_weights(self, pct: np.ndarray | None = None) -> np.ndarray:
        """Convert bs_percentage array (0-100) to weight fractions.

        If pct is None, uses bs_pct_current.
        The existing pipeline stores:
            bs_percentage = product_balance / total_balance × 100
            total_balance = 2 × total_assets
        So:  product_balance = total_assets × 2 × bs_percentage / 100
             weight = product_balance / total_assets = 2 × bs_percentage / 100
        """
        if pct is None:
            pct = self.bs_pct_current
        return 2.0 * pct / 100.0

    def current_weights(self) -> np.ndarray:
        """Return weight fractions for current bs_percentage."""
        return self.pct_to_weights()

    @classmethod
    def load(cls, npz_path: str) -> "BalanceSheetParams":
        """Load from product_params.npz."""
        data = np.load(npz_path, allow_pickle=True)
        return cls(
            product_code      = data["product_code"],
            product_name      = data["product_name"],
            bs_side           = data["bs_side"],
            currency          = data["currency"],
            sign              = data["sign"].astype(float),
            bs_pct_current    = data["bs_pct_current"].astype(float),
            nii_unit_rate     = data["nii_unit_rate"].astype(float),
            delta_nii_unit    = data["delta_nii_unit"].astype(float),
            eve_pv_factor     = data["eve_pv_factor"].astype(float),
            d_mod             = data["d_mod"].astype(float),
            delta_eve_unit    = data["delta_eve_unit"].astype(float),
            hqla_factor       = data["hqla_factor"].astype(float),
            lcr_runoff        = data["lcr_runoff"].astype(float),
            asf_factor        = data["asf_factor"].astype(float),
            rsf_factor        = data["rsf_factor"].astype(float),
            inflow_30d_frac   = data["inflow_30d_frac"].astype(float),
            amort_frac_1y     = data["amort_frac_1y"].astype(float),
            coeff_a           = data["coeff_a"].astype(float),
            coeff_b           = data["coeff_b"].astype(float),
            client_floor      = data["client_floor"].astype(float),
            client_cap        = data["client_cap"].astype(float),
            repricing_tenor_m = data["repricing_tenor_m"].astype(float),
            scenario_ids      = data["scenario_ids"],
            total_assets      = float(data["total_assets"][0]),
            report_date       = str(data["report_date"][0]),
            balance_arr       = data["balance_arr"].astype(float),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CurveTensors
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CurveTensors:
    """Pre-interpolated curve grids for all scenarios and currencies.

    Shapes
    ------
    disc_factors : (n_scenarios, n_currencies, 360)  d_f at month m
    fwd_rates    : (n_scenarios, n_currencies, 360)  annualised fwd rate at month m
    """
    disc_factors: np.ndarray   # (S, C, 360)
    fwd_rates:    np.ndarray   # (S, C, 360)
    scenario_ids: np.ndarray   # str (S,)
    currencies:   np.ndarray   # str (C,)
    curve_names:  np.ndarray   # str (C,)
    report_date:  str
    n_months:     int

    def scenario_index(self, scenario_id: str) -> int:
        idx = np.where(self.scenario_ids == scenario_id)[0]
        if len(idx) == 0:
            raise KeyError(f"Scenario '{scenario_id}' not found.")
        return int(idx[0])

    def currency_index(self, ccy: str) -> int:
        idx = np.where(self.currencies == ccy)[0]
        if len(idx) == 0:
            raise KeyError(f"Currency '{ccy}' not found.")
        return int(idx[0])

    def get_fwd_curve(self, scenario_id: str, currency: str) -> np.ndarray:
        """Return (360,) annualised forward rate array for given scenario/currency."""
        return self.fwd_rates[self.scenario_index(scenario_id),
                              self.currency_index(currency)]

    def get_disc_curve(self, scenario_id: str, currency: str) -> np.ndarray:
        """Return (360,) discount factor array for given scenario/currency."""
        return self.disc_factors[self.scenario_index(scenario_id),
                                 self.currency_index(currency)]

    @classmethod
    def load(cls, npz_path: str) -> "CurveTensors":
        """Load from curve_tensors.npz."""
        data = np.load(npz_path, allow_pickle=True)
        return cls(
            disc_factors = data["disc_factors"].astype(float),
            fwd_rates    = data["fwd_rates"].astype(float),
            scenario_ids = data["scenario_ids"],
            currencies   = data["currencies"],
            curve_names  = data["curve_names"],
            report_date  = str(data["report_date"][0]),
            n_months     = int(data["n_months"][0]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Constraint helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_weight_bounds(params: BalanceSheetParams) -> list[tuple[float, float]]:
    """Default weight bounds per product (0 … current_weight × 2).

    The upper bound is set to 2× current weight to keep the search in
    a realistic neighbourhood.  The caller can override these for specific
    products.
    """
    current_w = params.current_weights()
    bounds = []
    for w in current_w:
        lo = 0.0
        hi = max(w * 2.0, 0.02)   # always allow at least 2% if currently non-zero
        bounds.append((lo, hi))
    return bounds


def asset_sum_constraint(weights: np.ndarray, params: BalanceSheetParams) -> float:
    """Σ weights for assets == 1.0  (balance sheet sum constraint, asset side)."""
    return float(weights[params.is_asset].sum() - 1.0)


def liability_sum_constraint(weights: np.ndarray, params: BalanceSheetParams) -> float:
    """Σ weights for liabilities+equity == 1.0  (funding side sum constraint)."""
    return float(weights[params.is_liability | params.is_equity].sum() - 1.0)
