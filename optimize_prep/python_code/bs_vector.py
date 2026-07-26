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
    cohort_id:    np.ndarray      # str, (n,)  unique entry ID e.g. "1000_A_PLN_2024_01"
    product_code: np.ndarray      # str, (n,)
    product_name: np.ndarray      # str, (n,)
    bs_side:      np.ndarray      # str, (n,)  'A' | 'L' | 'E'
    currency:     np.ndarray      # str, (n,)
    sign:         np.ndarray      # float64, (n,)  +1 for A, -1 for L/E
    start_year:   np.ndarray      # float64, (n,)  -1 for single-row products
    start_month:  np.ndarray      # float64, (n,)  -1 for single-row products
    is_cohort:    np.ndarray      # bool, (n,)

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
    rwa_factor:     np.ndarray    # float64, (n,)   Basel III risk weight (assets only), 0 for L/E
    inflow_30d_frac: np.ndarray   # float64, (n,)   30-day inflow fraction (assets only)
    amort_frac_1y:  np.ndarray    # float64, (n,)   1Y amortisation fraction

    # ── Credit risk (Expected Loss) ───────────────────────────────────────────
    pd_rate:        np.ndarray    # float64, (n,)   probability of default (assets only)
    lgd_rate:       np.ndarray    # float64, (n,)   loss given default (assets only)
    el_unit:        np.ndarray    # float64, (n,)   PD × LGD — EL per PLN of balance

    # ── Non-interest fee income ───────────────────────────────────────────────
    fee_unit_rate:  np.ndarray    # float64, (n,)   flat yearly fee income per PLN balance
                                   # (e.g. mortgage/cash-loan servicing fees); 0 where unset

    # ── Marketing / customer-acquisition cost ─────────────────────────────────
    acq_cost_rate:  np.ndarray    # float64, (n,)   yearly cost rate applied only to a
                                   # product's GROWTH above baseline weight in the
                                   # optimizer (not the whole balance); 0 where unset

    # ── Cost of capital ───────────────────────────────────────────────────────
    coc_rate:       float         # scalar — CoC hurdle rate (e.g. 0.10 = 10%)
    cet1_target:    float         # scalar — target CET1 ratio (e.g. 0.12 = 12%)

    # ── Price-volume elasticity ───────────────────────────────────────────────
    vol_elasticity:        np.ndarray    # float64, (n,)  NII rate change per 1pp weight increase
    nii_unit_rate_new_biz: np.ndarray    # float64, (n,)  NII rate for new business origination

    # ── Substitution (cannibalism) pairs ──────────────────────────────────────
    subst_src_pc:   np.ndarray    # object,  (k,)  product_code of source product
    subst_src_side: np.ndarray    # object,  (k,)  bs_side of source
    subst_dst_pc:   np.ndarray    # object,  (k,)  product_code of dest product
    subst_dst_side: np.ndarray    # object,  (k,)  bs_side of dest
    subst_rates:    np.ndarray    # float64, (k,)  substitution rate per pair

    # ── rate coefficients ─────────────────────────────────────────────────────
    coeff_a:        np.ndarray    # float64, (n,)
    coeff_b:        np.ndarray    # float64, (n,)  decimal spread
    client_floor:   np.ndarray    # float64, (n,)  rate floor (decimal)
    client_cap:     np.ndarray    # float64, (n,)  rate cap (decimal)

    # ── repricing ─────────────────────────────────────────────────────────────
    repricing_tenor_m: np.ndarray  # float64, (n,)  months to first repricing
    rate_type:         np.ndarray  # str,     (n,)  'F' fixed / 'V' floating / '' single-row
    coupon_rate:       np.ndarray  # float64, (n,)  contracted rate for fixed cohorts; NaN otherwise

    # ── schedule monthly profile (from cf.products) ───────────────────────────
    cohort_outstanding_m: np.ndarray  # float64, (n, 12)  normalised outstanding per month
    cohort_capital_m:     np.ndarray  # float64, (n, 12)  normalised capital per month
    cohort_interest_yf_m: np.ndarray  # float64, (n, 12)  normalised outstanding * actual cf_yf
    cohort_capital_remain_m: np.ndarray  # float64, (n, 12) normalised capital * actual remain_yf
    cohort_locked_rate:   np.ndarray  # float64, (n,)     locked-period weighted rate
    cohort_t_first_m:     np.ndarray  # float64, (n,)     months to first future fixing

    # ── scenario labels ───────────────────────────────────────────────────────
    scenario_ids: np.ndarray      # str, (S,)  e.g. ['par_up', 'par_dn', ...]

    # ── metadata ──────────────────────────────────────────────────────────────
    total_assets: float
    report_date:  str
    balance_arr:  np.ndarray      # float64, (n,) actual balance amounts at extraction

    # ── derived masks (not stored in npz, computed on load) ──────────────────
    is_asset:     np.ndarray = field(default=None, compare=False)  # bool (n,)
    is_liability: np.ndarray = field(default=None, compare=False)  # bool (n,)
    is_equity:    np.ndarray = field(default=None, compare=False)  # bool (n,)
    is_fixed:     np.ndarray = field(default=None, compare=False)  # bool (n,)

    def __post_init__(self):
        # computed masks — must bypass frozen=True via object.__setattr__
        object.__setattr__(self, "is_asset",    self.bs_side == "A")
        object.__setattr__(self, "is_liability", self.bs_side == "L")
        object.__setattr__(self, "is_equity",   self.bs_side == "E")
        object.__setattr__(self, "is_fixed",    self.rate_type == "F")

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
        n = len(data["cohort_id"])
        return cls(
            cohort_id         = data["cohort_id"],
            product_code      = data["product_code"],
            product_name      = data["product_name"],
            bs_side           = data["bs_side"],
            currency          = data["currency"],
            sign              = data["sign"].astype(float),
            start_year        = data["start_year"].astype(float),
            start_month       = data["start_month"].astype(float),
            is_cohort         = data["is_cohort"].astype(bool),
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
            rwa_factor        = data["rwa_factor"].astype(float) if "rwa_factor" in data else np.zeros(n, dtype=float),
            pd_rate           = data["pd_rate"].astype(float)   if "pd_rate"   in data else np.zeros(n, dtype=float),
            lgd_rate          = data["lgd_rate"].astype(float)  if "lgd_rate"  in data else np.zeros(n, dtype=float),
            el_unit           = data["el_unit"].astype(float)   if "el_unit"   in data else np.zeros(n, dtype=float),
            fee_unit_rate     = (data["fee_unit_rate"].astype(float)
                                 if "fee_unit_rate" in data
                                 else np.zeros(n, dtype=float)),
            acq_cost_rate     = (data["acq_cost_rate"].astype(float)
                                 if "acq_cost_rate" in data
                                 else np.zeros(n, dtype=float)),
            coc_rate          = float(data["coc_rate"][0])      if "coc_rate"  in data else 0.10,
            cet1_target       = float(data["cet1_target"][0])   if "cet1_target" in data else 0.12,
            vol_elasticity        = (data["vol_elasticity"].astype(float)
                                     if "vol_elasticity" in data
                                     else np.zeros(n, dtype=float)),
            nii_unit_rate_new_biz = (data["nii_unit_rate_new_biz"].astype(float)
                                     if "nii_unit_rate_new_biz" in data
                                     else data["nii_unit_rate"].astype(float)),
            subst_src_pc    = (data["subst_src_pc"]             if "subst_src_pc"   in data else np.array([], dtype=object)),
            subst_src_side  = (data["subst_src_side"]           if "subst_src_side" in data else np.array([], dtype=object)),
            subst_dst_pc    = (data["subst_dst_pc"]             if "subst_dst_pc"   in data else np.array([], dtype=object)),
            subst_dst_side  = (data["subst_dst_side"]           if "subst_dst_side" in data else np.array([], dtype=object)),
            subst_rates     = (data["subst_rates"].astype(float) if "subst_rates"    in data else np.array([], dtype=float)),
            inflow_30d_frac   = data["inflow_30d_frac"].astype(float),
            amort_frac_1y     = data["amort_frac_1y"].astype(float),
            coeff_a           = data["coeff_a"].astype(float),
            coeff_b           = data["coeff_b"].astype(float),
            client_floor      = data["client_floor"].astype(float),
            client_cap        = data["client_cap"].astype(float),
            repricing_tenor_m = data["repricing_tenor_m"].astype(float),
            rate_type         = data["rate_type"].astype(str),
            coupon_rate       = data["coupon_rate"].astype(float),
            scenario_ids      = data["scenario_ids"],
            total_assets      = float(data["total_assets"][0]),
            report_date       = str(data["report_date"][0]),
            balance_arr       = data["balance_arr"].astype(float),
            # schedule monthly profile — zeros/999 if not present (old npz)
            cohort_outstanding_m = (
                data["cohort_outstanding_m"].astype(float)
                if "cohort_outstanding_m" in data
                else np.zeros((n, 12))
            ),
            cohort_capital_m = (
                data["cohort_capital_m"].astype(float)
                if "cohort_capital_m" in data
                else np.zeros((n, 12))
            ),
            cohort_interest_yf_m = (
                data["cohort_interest_yf_m"].astype(float)
                if "cohort_interest_yf_m" in data
                else (
                    data["cohort_outstanding_m"].astype(float) / 12.0
                    if "cohort_outstanding_m" in data
                    else np.zeros((n, 12))
                )
            ),
            cohort_capital_remain_m = (
                data["cohort_capital_remain_m"].astype(float)
                if "cohort_capital_remain_m" in data
                else (
                    data["cohort_capital_m"].astype(float)
                    * np.array([(12 - m - 0.5) / 12.0 for m in range(12)], dtype=float)[None, :]
                    if "cohort_capital_m" in data
                    else np.zeros((n, 12))
                )
            ),
            cohort_locked_rate = (
                data["cohort_locked_rate"].astype(float)
                if "cohort_locked_rate" in data
                else np.zeros(n)
            ),
            cohort_t_first_m = (
                data["cohort_t_first_m"].astype(float)
                if "cohort_t_first_m" in data
                else np.full(n, 999.0)
            ),
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
# CohortRates  — CF-based rate/discount tables for fast NII + EVE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CohortRates:
    """Pre-computed per-cohort rate and CF tables for CF-based NII/EVE calculation.

    All arrays have first dimension n (number of cohort/single-row entries),
    aligned with BalanceSheetParams.

    rate_matrix[n, 12, n_rate_scen]
        Client interest rate for cohort i in calendar month m under scenario s.
        Base scenario is index 0, shocked scenarios follow in SHOCKED_SCENARIO_IDS order.
        Fixed cohorts: coupon_rate (constant across months and scenarios).
        Float cohorts locked period: locked_rate (constant across scenarios).
        Float cohorts floating period: clip(base_fwd + coeff_a*(shock_fwd-base_fwd)+coeff_b, floor, cap).

    delta_eve_dur_unit[n, n_rate_scen]
        Pre-computed duration-based delta EVE per PLN of balance per scenario.
        = -sign * D_mod * delta_fwd(D_mod_tenor, scenario).
        Zero for base scenario.

    cf_capital_frac[n, max_q]   capital CF / balance per quarterly bucket (all cohorts)
    cf_fixed_int_frac[n, max_q] (capital + interest) CF / balance for FIXED cohorts;
                                 equals cf_capital_frac for floating cohorts
    eve_pv_frac[n, max_q, n_rate_scen]
        Optional scenario-specific discounted EVE PV / balance.  Non-zero rows
        override cf_fixed_int_frac * cohort_disc_q in the fast EVE path.
    cf_yf[n, max_q]             year-fraction of CF bucket midpoint for discounting
    cf_n_q[n]                   number of valid CF buckets per cohort

    ccy_idx[n]                  index into CurveTensors.currencies per cohort
    rate_scenario_ids[n_scen]   scenario labels (base first, then shocked)
    """
    rate_matrix:         np.ndarray   # (n, 12, n_rate_scen)   float
    renewal_rate_matrix: np.ndarray   # (n, 12, n_rate_scen)   float  new-business rate per month/scenario
    delta_eve_dur_unit:  np.ndarray   # (n, n_rate_scen)        float
    cf_capital_frac:     np.ndarray   # (n, max_q)              float
    cf_fixed_int_frac:   np.ndarray   # (n, max_q)              float
    eve_pv_frac:         np.ndarray   # (n, max_q, n_rate_scen) float
    cf_yf:               np.ndarray   # (n, max_q)              float
    cf_n_q:              np.ndarray   # (n,)                    int
    ccy_idx:             np.ndarray   # (n,)                    int
    rate_scenario_ids:   np.ndarray   # (n_rate_scen,)          str
    cohort_disc_q:       np.ndarray   # (n, max_q, n_rate_scen) float — pre-indexed DFs

    @classmethod
    def load(cls, npz_path: str) -> "CohortRates":
        data = np.load(npz_path, allow_pickle=True)
        n_entries = len(data["cr_cf_n_q"])
        n_q       = data["cr_cf_capital_frac"].shape[1]
        n_scen    = len(data["cr_rate_scenario_ids"])
        return cls(
            rate_matrix        = data["cr_rate_matrix"].astype(float),
            renewal_rate_matrix = (
                data["cr_renewal_rate_matrix"].astype(float)
                if "cr_renewal_rate_matrix" in data
                else np.zeros_like(data["cr_rate_matrix"], dtype=float)
            ),
            delta_eve_dur_unit = data["cr_delta_eve_dur_unit"].astype(float),
            cf_capital_frac    = data["cr_cf_capital_frac"].astype(float),
            cf_fixed_int_frac  = data["cr_cf_fixed_int_frac"].astype(float),
            eve_pv_frac        = (
                data["cr_eve_pv_frac"].astype(float)
                if "cr_eve_pv_frac" in data
                else np.zeros((n_entries, n_q, n_scen), dtype=float)
            ),
            cf_yf              = data["cr_cf_yf"].astype(float),
            cf_n_q             = data["cr_cf_n_q"].astype(int),
            ccy_idx            = data["cr_ccy_idx"].astype(int),
            rate_scenario_ids  = data["cr_rate_scenario_ids"],
            cohort_disc_q      = (
                data["cr_cohort_disc_q"].astype(float)
                if "cr_cohort_disc_q" in data
                else np.zeros((n_entries, n_q, n_scen), dtype=float)
            ),
        )

    @property
    def n_rate_scen(self) -> int:
        return len(self.rate_scenario_ids)

    def scenario_index(self, scenario_id: str) -> int:
        idx = np.where(self.rate_scenario_ids == scenario_id)[0]
        if len(idx) == 0:
            raise KeyError(f"Scenario '{scenario_id}' not found in rate_scenario_ids.")
        return int(idx[0])


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
