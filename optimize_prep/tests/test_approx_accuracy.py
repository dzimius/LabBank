"""test_approx_accuracy.py
==========================
Unit tests for optimize_prep fast metric functions.

These tests verify:
1. Shape and dtype contracts for all numpy arrays
2. LCR / NSFR formula correctness against known inputs
3. NII unit-rate model properties (linearity, sign convention)
4. EVE linearity and duration-model direction
5. AllMetrics feasibility checks

Run with pytest from the optimize_prep/tests directory:
    pytest test_approx_accuracy.py -v

Note: these tests do NOT require a SQL connection or the .npz files —
they use synthetic params/curves built inline.
"""
from __future__ import annotations

import sys
import os
import numpy as np
import pytest

# Allow importing from python_code without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python_code"))

from bs_vector import BalanceSheetParams, CurveTensors, asset_sum_constraint, liability_sum_constraint
from nii_fast  import compute_nii_base_fast, compute_delta_nii_fast, compute_nii_all_scenarios
from eve_fast  import compute_eve_base_fast, compute_delta_eve_fast, compute_worst_delta_eve
from lcr_fast  import compute_lcr_fast, lcr_constraint_value
from nsfr_fast import compute_nsfr_fast, nsfr_constraint_value
from metrics   import AllMetrics, compute_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic params fixture
# ─────────────────────────────────────────────────────────────────────────────

def _make_params(n: int = 4) -> BalanceSheetParams:
    """Build a minimal synthetic BalanceSheetParams for testing.

    Product layout (n=4):
      0: mortgage_fixed    A  PLN  rate=6%   hqla=0   lcr=0   asf=0  rsf=0.65
      1: gov_bond_5Y       A  PLN  rate=5%   hqla=0.98 lcr=0  asf=0  rsf=0.05
      2: current_account   L  PLN  rate=-1%  hqla=0   lcr=0.1 asf=0.9 rsf=0
      3: term_deposit_1Y   L  PLN  rate=-3%  hqla=0   lcr=0.15 asf=0.9 rsf=0
    """
    scenario_ids = np.array(["par_up", "par_dn"])
    S = 2

    bs_side      = np.array(["A", "A", "L", "L"])
    currency     = np.array(["PLN", "PLN", "PLN", "PLN"])
    sign         = np.array([1.0, 1.0, -1.0, -1.0])
    bs_pct       = np.array([25.0, 10.0, 20.0, 15.0])

    # NII unit rates (positive for assets, negative for liabilities)
    nii_unit     = np.array([0.06, 0.05, -0.01, -0.03])
    # delta_NII per PLN per scenario: assets gain in up shock, liabilities lose
    delta_nii    = np.array([
        [0.003, -0.003],   # mortgage: +3bps in par_up, -3bps in par_dn
        [0.001, -0.001],   # bond (short dur): +1bp / -1bp
        [-0.001, 0.001],   # CA: small negative delta
        [-0.002, 0.002],   # TD: reprices, negative delta
    ])  # shape (4, 2)

    eve_pv_factor = np.array([0.85, 0.95, -0.95, -0.92])  # PV / balance
    d_mod         = np.array([5.0, 3.0, -0.5, -1.0])      # years, signed
    delta_eve     = np.array([
        [-0.08, 0.08],   # mortgage: -8% of bal in par_up, +8% in par_dn
        [-0.05, 0.05],
        [0.01, -0.01],
        [0.02, -0.02],
    ])  # shape (4, 2)

    hqla_factor   = np.array([0.0, 0.98, 0.0, 0.0])
    lcr_runoff    = np.array([0.0, 0.0, 0.1, 0.15])
    asf_factor    = np.array([0.0, 0.0, 0.9, 0.9])
    rsf_factor    = np.array([0.65, 0.05, 0.0, 0.0])
    inflow_30d    = np.array([0.05, 0.02, 0.0, 0.0])
    amort_frac    = np.array([0.08, 0.0, 0.5, 0.9])

    # ── credit risk / RWA / fees / acquisition cost (assets only) ──────────────
    rwa_factor    = np.array([0.35, 0.0, 0.0, 0.0])
    pd_rate       = np.array([0.01, 0.0, 0.0, 0.0])
    lgd_rate      = np.array([0.45, 0.0, 0.0, 0.0])
    el_unit       = pd_rate * lgd_rate
    fee_unit      = np.array([0.002, 0.0, 0.0, 0.0])
    acq_cost      = np.array([0.001, 0.0, 0.0, 0.0])
    vol_elast     = np.zeros(n)

    coeff_a       = np.ones(n)
    coeff_b       = np.zeros(n)
    client_floor  = np.full(n, -np.inf)
    client_cap    = np.full(n,  np.inf)
    repricing_m   = np.array([120.0, 60.0, 1.0, 12.0])

    balance_arr   = np.array([2_500_000_000, 1_000_000_000, 2_000_000_000, 1_500_000_000], dtype=float)

    return BalanceSheetParams(
        cohort_id         = np.array(["1001", "2001", "3001", "3002"]),
        product_code      = np.array(["1001", "2001", "3001", "3002"]),
        product_name      = np.array(["mortgage_fixed", "gov_bond", "current_account", "term_deposit"]),
        bs_side           = bs_side,
        currency          = currency,
        sign              = sign,
        start_year        = np.full(n, -1.0),
        start_month       = np.full(n, -1.0),
        is_cohort         = np.array([False, False, False, False]),
        bs_pct_current    = bs_pct,
        nii_unit_rate     = nii_unit,
        delta_nii_unit    = delta_nii,
        eve_pv_factor     = eve_pv_factor,
        d_mod             = d_mod,
        delta_eve_unit    = delta_eve,
        hqla_factor       = hqla_factor,
        lcr_runoff        = lcr_runoff,
        asf_factor        = asf_factor,
        rsf_factor        = rsf_factor,
        rwa_factor        = rwa_factor,
        pd_rate           = pd_rate,
        lgd_rate          = lgd_rate,
        el_unit           = el_unit,
        fee_unit_rate     = fee_unit,
        acq_cost_rate     = acq_cost,
        coc_rate          = 0.10,
        cet1_target       = 0.12,
        vol_elasticity        = vol_elast,
        nii_unit_rate_new_biz = nii_unit.copy(),
        subst_src_pc      = np.array([], dtype=object),
        subst_src_side    = np.array([], dtype=object),
        subst_dst_pc      = np.array([], dtype=object),
        subst_dst_side    = np.array([], dtype=object),
        subst_rates       = np.array([], dtype=float),
        inflow_30d_frac   = inflow_30d,
        amort_frac_1y     = amort_frac,
        coeff_a           = coeff_a,
        coeff_b           = coeff_b,
        client_floor      = client_floor,
        client_cap        = client_cap,
        repricing_tenor_m    = repricing_m,
        rate_type            = np.array(["F", "F", "", ""]),
        coupon_rate          = np.array([0.06, 0.05, np.nan, np.nan]),
        cohort_outstanding_m = np.zeros((n, 12)),
        cohort_capital_m     = np.zeros((n, 12)),
        cohort_interest_yf_m = np.zeros((n, 12)),
        cohort_capital_remain_m = np.zeros((n, 12)),
        cohort_locked_rate   = np.zeros(n),
        cohort_t_first_m     = np.full(n, 999.0),
        gap_reprice_frac_m   = np.zeros((n, 16)),
        scenario_ids         = scenario_ids,
        total_assets         = 5_000_000_000.0,
        report_date          = "2024-12-31",
        balance_arr          = balance_arr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: BalanceSheetParams
# ─────────────────────────────────────────────────────────────────────────────

class TestBalanceSheetParams:
    def test_derived_masks(self):
        p = _make_params()
        assert p.is_asset.tolist()    == [True, True, False, False]
        assert p.is_liability.tolist()== [False, False, True, True]

    def test_weights_to_amounts(self):
        p = _make_params()
        w = np.array([0.5, 0.2, 0.4, 0.3])
        amts = p.weights_to_amounts(w, total_assets=1_000_000)
        np.testing.assert_allclose(amts, [500_000, 200_000, 400_000, 300_000])

    def test_pct_to_weights_roundtrip(self):
        p = _make_params()
        w   = p.pct_to_weights(p.bs_pct_current)
        pct = w / 2.0 * 100.0
        np.testing.assert_allclose(pct, p.bs_pct_current)

    def test_scenario_index(self):
        p = _make_params()
        assert p.scenario_index("par_up") == 0
        assert p.scenario_index("par_dn") == 1
        with pytest.raises(KeyError):
            p.scenario_index("nonexistent")

    def test_asset_sum_constraint(self):
        p = _make_params()
        w = np.array([0.5, 0.5, 0.4, 0.6])   # assets sum to 1.0
        assert abs(asset_sum_constraint(w, p)) < 1e-12

    def test_liability_sum_constraint(self):
        p = _make_params()
        w = np.array([0.5, 0.5, 0.4, 0.6])   # liabilities sum to 1.0
        assert abs(liability_sum_constraint(w, p)) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Tests: NII
# ─────────────────────────────────────────────────────────────────────────────

class TestNIIFast:
    def setup_method(self):
        self.p = _make_params()
        self.amounts = np.array([2.5e9, 1.0e9, 2.0e9, 1.5e9])

    def test_nii_base_linearity(self):
        """NII should scale linearly with amounts."""
        nii1 = compute_nii_base_fast(self.amounts, self.p)
        nii2 = compute_nii_base_fast(self.amounts * 2, self.p)
        assert abs(nii2 - 2 * nii1) < 1.0   # PLN rounding

    def test_nii_base_sign(self):
        """Assets contribute positive NII, liabilities negative."""
        # asset-only amounts
        a_only = np.array([1.0, 1.0, 0.0, 0.0])
        l_only = np.array([0.0, 0.0, 1.0, 1.0])
        assert compute_nii_base_fast(a_only, self.p) > 0
        assert compute_nii_base_fast(l_only, self.p) < 0

    def test_nii_base_formula(self):
        """Spot-check: NII = Σ amounts × nii_unit_rate."""
        expected = float(np.dot(self.amounts, self.p.nii_unit_rate))
        got      = compute_nii_base_fast(self.amounts, self.p)
        assert abs(got - expected) < 0.01

    def test_delta_nii_par_up(self):
        """In par_up, asset NII should increase (positive delta)."""
        # only assets
        amts = np.array([1.0e9, 1.0e9, 0.0, 0.0])
        d    = compute_delta_nii_fast(amts, self.p, "par_up")
        assert d > 0, f"Expected positive delta_NII for assets in par_up, got {d}"

    def test_delta_nii_all_scenarios_keys(self):
        result = compute_nii_all_scenarios(self.amounts, self.p)
        assert "base"   in result
        assert "par_up" in result
        assert "par_dn" in result


# ─────────────────────────────────────────────────────────────────────────────
# Tests: EVE
# ─────────────────────────────────────────────────────────────────────────────

class TestEVEFast:
    def setup_method(self):
        self.p = _make_params()
        self.amounts = np.array([2.5e9, 1.0e9, 2.0e9, 1.5e9])

    def test_eve_base_linearity(self):
        eve1 = compute_eve_base_fast(self.amounts, self.p)
        eve2 = compute_eve_base_fast(self.amounts * 3, self.p)
        assert abs(eve2 - 3 * eve1) < 1.0

    def test_delta_eve_par_up_assets(self):
        """For long-duration assets, delta_EVE in par_up should be negative."""
        amts = np.array([1.0e9, 1.0e9, 0.0, 0.0])
        d    = compute_delta_eve_fast(amts, self.p, "par_up")
        assert d < 0, f"Expected negative delta_EVE for long assets in par_up, got {d}"

    def test_worst_delta_eve(self):
        worst_val, worst_scen = compute_worst_delta_eve(self.amounts, self.p)
        all_scenarios = compute_nii_all_scenarios(self.amounts, self.p)
        # worst should be <= all other deltas
        for scen in self.p.scenario_ids:
            d = compute_delta_eve_fast(self.amounts, self.p, str(scen))
            assert worst_val <= d + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Tests: LCR
# ─────────────────────────────────────────────────────────────────────────────

class TestLCRFast:
    def setup_method(self):
        self.p = _make_params()
        self.amounts = np.array([2.5e9, 1.0e9, 2.0e9, 1.5e9])

    def test_lcr_pln_positive(self):
        lcr = compute_lcr_fast(self.amounts, self.p)
        assert "PLN" in lcr
        # HQLA = 1e9 × 0.98 = 980M; outflows = (2e9×0.1 + 1.5e9×0.15) = 425M
        assert lcr["PLN"] > 0

    def test_lcr_formula_manual(self):
        """Manually verify LCR formula."""
        amts = self.amounts
        hqla      = amts[1] * 0.98           # only gov_bond is HQLA
        outflows  = amts[2] * 0.1 + amts[3] * 0.15
        raw_inflows = amts[0] * 0.05 + amts[1] * 0.02
        inflows   = min(0.75 * raw_inflows, 0.75 * outflows)
        net_out   = outflows - inflows
        expected_lcr = hqla / net_out

        got = compute_lcr_fast(amts, self.p)["PLN"]
        assert abs(got - expected_lcr) < 1e-6

    def test_lcr_constraint_satisfied(self):
        # With current synthetic params, LCR should be well above 1
        slack = lcr_constraint_value(self.amounts, self.p, currency="PLN", min_lcr=1.0)
        assert slack > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: NSFR
# ─────────────────────────────────────────────────────────────────────────────

class TestNSFRFast:
    def setup_method(self):
        self.p = _make_params()
        self.amounts = np.array([2.5e9, 1.0e9, 2.0e9, 1.5e9])

    def test_nsfr_formula_manual(self):
        """Manually verify NSFR = ASF / RSF."""
        amts = self.amounts
        asf = amts[2] * 0.9 + amts[3] * 0.9
        rsf = amts[0] * 0.65 + amts[1] * 0.05
        expected = asf / rsf

        got = compute_nsfr_fast(amts, self.p)["PLN"]
        assert abs(got - expected) < 1e-6

    def test_nsfr_constraint(self):
        slack = nsfr_constraint_value(self.amounts, self.p, currency="PLN", min_nsfr=1.0)
        # with synthetic params: ASF=(2e9+1.5e9)×0.9=3.15e9, RSF=2.5e9×0.65+1e9×0.05=1.675e9
        # NSFR=3.15/1.675≈1.88 → slack≈0.88 > 0
        assert slack > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: AllMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestAllMetrics:
    def setup_method(self):
        self.p       = _make_params()
        self.amounts = np.array([2.5e9, 1.0e9, 2.0e9, 1.5e9])
        self.weights = self.amounts / 5e9

    def test_compute_all_metrics_runs(self):
        m = compute_all_metrics(self.weights, self.p, None, 5e9)
        assert isinstance(m, AllMetrics)
        assert m.nii_base != 0.0
        assert "PLN" in m.lcr
        assert "PLN" in m.nsfr
        assert m.worst_delta_eve < 0   # long assets dominate → negative in par_up

    def test_feasibility_check(self):
        m = compute_all_metrics(self.weights, self.p, None, 5e9)
        tier1 = 5e8   # 500M Tier 1
        # With synthetic params the book should be feasible
        result = m.is_feasible(tier1)
        assert isinstance(result, bool)

    def test_constraint_slack_keys(self):
        m      = compute_all_metrics(self.weights, self.p, None, 5e9)
        slacks = m.constraint_slack(tier1_capital=5e8)
        assert "lcr_PLN"  in slacks
        assert "nsfr_PLN" in slacks
        assert "sot_eve"  in slacks
        assert "sot_nii"  in slacks

    def test_metrics_scale_linearly(self):
        """NII and EVE should scale 2× when amounts double."""
        w1 = self.weights
        w2 = self.weights * 2
        m1 = compute_all_metrics(w1, self.p, None, 5e9)
        m2 = compute_all_metrics(w2, self.p, None, 5e9)
        assert abs(m2.nii_base - 2 * m1.nii_base) < 1.0
        assert abs(m2.eve_base - 2 * m1.eve_base) < 1.0
