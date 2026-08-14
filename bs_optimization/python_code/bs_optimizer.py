"""bs_optimizer.py
==================
Balance sheet optimization — maximize Economic Profit subject to regulatory constraints.

Economic Profit = Margin + Fee - EL - CoC - OpEx   (2026-07-09 / 2026-07-14)
    Margin = margin-over-FTP income (client rate net of transfer-pricing cost;
             see ftp_store.margin_unit_rate) — NOT raw client-rate NII, which
             is still computed/shown everywhere as an informational reference
    Fee    = non-interest fee income = Σ fee_unit_rate × balance (per-product
             static assumption from bank_data.xlsx's bs_structure sheet)
    EL     = expected loss = PD × LGD × EAD   (EAD = balance sheet volume)
    CoC    = cost of capital = RWA × CET1_target × coc_rate
    OpEx   = flat operating cost rate × balance (both assets and liabilities)

Three modes
-----------
full       : all non-equity product weights are free in [0, 1]
partial    : products NOT in fixed_products are free; the rest are pinned
max_shift  : each free product weight is bounded to [max(0, w-shift), w+shift];
             equity and fixed_products are pinned at current weights

Constraints
-----------
  sum(asset weights)   = baseline asset sum   (balance sheet integrity)
  sum(funding weights) = baseline funding sum
  delta_EVE[s] / tier1 >= sot_eve_floor  for all shocked scenarios s
  delta_NII[s] / tier1 >= sot_nii_floor  for all shocked scenarios s
  LCR[ccy]  >= min_lcr  for all currencies
  NSFR[ccy] >= min_nsfr for all currencies
  weights[j] >= 0

Optimizer
---------
Primary: scipy.optimize.linprog (HiGHS) — exact LP solver.
  EP is linear in product weights (NII, EL, and CoC are all linear in balance).
  EVE SOT, NII-SOT-delta, and NSFR constraints are also linear in weights.
  LCR is checked post-hoc; if violated a SLSQP polish step is applied.
Fallback: SLSQP from the LP solution (handles edge cases).
Cohort weights are derived from product weights by proportional scaling.
Equity products (bs_side='E') are always pinned — they represent regulatory T1 capital.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import minimize, linprog

# bs_vector, nii_eve_cf_fast, lcr_fast, nsfr_fast live in optimize_prep/python_code
_OPTPREP = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "optimize_prep", "python_code")
)
if _OPTPREP not in sys.path:
    sys.path.insert(0, _OPTPREP)

from bs_vector import BalanceSheetParams, CohortRates
from nii_eve_cf_fast import compute_nii_cf_all, compute_eve_cf_fast_all
from lcr_fast        import compute_lcr_fast
from nsfr_fast       import compute_nsfr_fast
from rwa_fast        import compute_rwa_fast
from bias_store      import load_bias_corrections, apply_eve_bias, apply_nii_bias
from ftp_store       import load_ftp_rates, margin_unit_rate
# _ProductMap/OPEX_RATE moved to optimize_prep/python_code/product_map.py
# (2026-08-14) so sandbox/app.py's Finance Metrics tab never has to import
# from bs_optimization/ -- re-imported here so every existing
# `from bs_optimizer import _ProductMap, OPEX_RATE` elsewhere keeps working.
from product_map     import _ProductMap, OPEX_RATE
# Flat operating cost, % of balance/yr, same rate for every non-equity product
# (both assets and liabilities -- both loan servicing and deposit/funding
# servicing cost money). Not per-product/per-cohort by design (2026-07-09,
# per user: "average value equals for all product types"). Halved from 0.013
# to 0.0065 (2026-07-14): the 1.3%/yr benchmark figure was meant as ~1.3% of
# TOTAL ASSETS, but this rate is applied to the non-equity balance on BOTH
# sides of the balance sheet (assets + liabilities, ~2x total assets) -- so
# 0.0065 x (~2x TA) recovers the intended ~1.3%-of-TA effective cost.

OptMode = Literal["full", "partial", "max_shift", "custom"]


# ─────────────────────────────────────────────────────────────────────────────
# Public API types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationConfig:
    """Configuration for one optimization run.

    Parameters
    ----------
    mode
        'full'       : all non-equity products are free in [0, 1]
        'partial'    : only products NOT in fixed_products are free in [0, 1]
        'max_shift'  : each free product bounded to [w-max_shift, w+max_shift]
    tier1_capital
        T1 capital in PLN (denominator for SOT % constraints).
    fixed_products
        Set of (product_code, bs_side) pairs pinned at current weight.
        'partial'    : products NOT to optimise (complement are the free ones)
        'max_shift'  : products that cannot shift at all (zero shift)
        'full'       : additionally pinned products (equity is always pinned)
    max_shift
        Max weight movement per product as fraction of total_assets (e.g. 0.03 = 3pp).
        Used only in 'max_shift' mode.
    sot_eve_floor / sot_nii_floor
        Regulatory floor in % of T1 (EBA default −15 / −5).
    sot_eve_buffer / sot_nii_buffer
        Positive value relaxes the effective floor by that many % T1 to absorb
        method B approximation error.  EVE error ~2.7% T1, NII error ~0.7% T1.
        Alternatively set the floor directly (e.g. −17 with buffer 0).
    """
    mode:            OptMode                  = "full"
    tier1_capital:   float                    = 0.0
    # 0.0 means auto-derive from equity balances in params (recommended)
    fixed_products:  set[tuple[str, str]]     = field(default_factory=set)
    max_shift:       float                    = 0.03
    sot_eve_floor:   float                    = -15.0
    sot_nii_floor:   float                    = -5.0
    sot_eve_buffer:  float                    = 3.0
    sot_nii_buffer:  float                    = 0.0
    include_irs:     bool                     = True
    min_lcr:         float                    = 1.0
    min_nsfr:        float                    = 1.0
    min_t1_rwa:      float                    = 0.0
    # CET1 floor: T1 / RWA >= min_t1_rwa.  0.0 = no constraint.
    max_iter:        int                      = 500
    tol:             float                    = 1e-8
    custom_bounds:   dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    # 'custom' mode only: maps (product_code, bs_side) -> (lb, ub) as fractions of total_assets.
    use_slack:       bool                                        = False
    # When True: add one non-negative slack per regulatory constraint (EVE/NII SOT, NSFR, RWA).
    # The LP is always feasible; violated constraints show up as slack > 0.
    # Use for infeasibility diagnosis. QP polish and LCR warm-start are skipped.
    slack_penalty:   float                                       = 1e6
    # Big-M penalty per unit of slack in the objective. Must be large enough that
    # the optimizer prefers feasibility over profit but not so large that it causes
    # numerical issues (1e6 works for PLN-denominated banks with T1 ~1e9).
    eve_breach_weight: float                                     = 0.0
    nii_breach_weight: float                                     = 0.0
    # PLN cost charged in the objective per average pp-T1 of EVE/NII-SOT breach
    # severity across the shocked scenarios: penalty = weight * (T1/100) / n_scen
    # per scenario -- same unit convention as hedge_optimizer.solve_optimal_ladder's
    # penalty_weight, so a value tuned there (e.g. ~50) is a reasonable starting
    # point here too; sweep for this book's own EP/breach trade-off curve.
    # 0.0 (default) = that floor stays a HARD constraint, exactly as before.
    # >0.0 turns that floor into a soft cost: the optimizer may breach it if the
    # EP gain outweighs the penalty (e.g. accept a few SOT breaches out of many
    # scenarios rather than sacrifice a lot of NII to stay clean everywhere).
    # Both floors are still regulatory hard limits per EBA -- treating them as
    # soft here is a deliberate modeling choice to explore the trade-off curve,
    # not a claim that a breach is compliant. Ignored (both hard) when
    # use_slack=True, which takes precedence as the pure infeasibility diagnostic.


@dataclass
class ProductChange:
    product_code: str
    bs_side:      str
    weight_old:   float   # fraction of total_assets
    weight_new:   float
    delta_weight: float
    pct_old:      float   # weight × 100
    pct_new:      float
    delta_pct:    float   # percentage-point change


@dataclass
class OptimizationResult:
    success:               bool
    message:               str
    mode:                  str
    n_iter:                int
    elapsed_s:             float
    nii_old:               float   # PLN — client-rate NII (Method B), informational/SOT only
    nii_new:               float   # PLN
    nii_improvement_m:     float   # PLN
    nii_improvement_pct:   float   # %
    margin_old:            float   # PLN — margin-over-FTP income at baseline weights; drives EP
    margin_new:            float   # PLN — margin-over-FTP income at optimized weights
    fee_old:                float   # PLN — non-interest fee income at baseline weights; drives EP
    fee_new:                float   # PLN — non-interest fee income at optimized weights
    el_old:                float   # PLN — expected loss at baseline weights
    el_new:                float   # PLN — expected loss at optimized weights
    coc_old:                float   # PLN — cost of capital at baseline weights
    coc_new:                float   # PLN — cost of capital at optimized weights
    opex_old:               float   # PLN — operating cost at baseline weights (OPEX_RATE x non-equity balance)
    opex_new:               float   # PLN — operating cost at optimized weights
    acq_cost_old:           float   # PLN — marketing/acquisition cost at baseline (always 0.0, no growth vs itself)
    acq_cost_new:           float   # PLN — acq cost at optimized weights (only on growth above baseline)
    ep_old:                float   # PLN — economic profit at baseline (Margin + Fee - EL - CoC - OpEx - AcqCost)
    ep_new:                float   # PLN — economic profit at optimized weights
    ep_improvement_m:      float   # PLN — EP improvement
    ep_improvement_pct:    float   # %
    product_changes:       list[ProductChange]
    sot_eve:               dict[str, float]   # scenario -> delta_EVE / T1 * 100
    sot_nii:               dict[str, float]   # scenario -> delta_NII / T1 * 100
    lcr:                   dict[str, float]   # ccy -> LCR ratio
    nsfr:                  dict[str, float]   # ccy -> NSFR ratio
    rwa:                   float              # total RWA in PLN
    t1_rwa_ratio:          float              # T1 / RWA
    feasible:              bool
    constraint_violations: dict[str, float]   # name -> slack (negative = violated)
    weights_old:           np.ndarray         # cohort-level fractions (n,)
    weights_new:           np.ndarray         # cohort-level fractions (n,)
    sot_eve_floor:         float   # effective EVE floor (= regulatory floor - eve_buffer)
    sot_nii_floor:         float   # effective NII floor (= regulatory floor - nii_buffer)
    sot_eve_buffer:        float
    sot_nii_buffer:        float
    include_irs:           bool
    tier1_capital:         float   # actual T1 used (PLN)
    min_lcr:               float
    min_nsfr:              float
    min_t1_rwa:            float
    use_slack:             bool
    slack_values:          dict[str, float]  # constraint -> violation magnitude (0 = feasible)
    eve_breach_weight:     float   # 0 = EVE floor was hard; >0 = soft, this PLN/pp-T1 weight used
    nii_breach_weight:     float   # 0 = NII floor was hard; >0 = soft, this PLN/pp-T1 weight used
    # Units: eve_*/nii_* in % of T1; nsfr in PLN (RSF-ASF); rwa in PLN (excess over cap)

    def print_summary(self) -> None:
        status = "SUCCESS" if self.success else "FAILED"
        feas   = "feasible" if self.feasible else "INFEASIBLE"
        print(f"\nOptimization [{self.mode}] - {status}  ({feas})")
        print(f"  T1 capital: {self.tier1_capital/1e6:,.0f}M PLN")
        irs_view = "hedged (IRS included)" if self.include_irs else "unhedged (IRS excluded)"
        print(f"  IR derivatives: {irs_view}")
        if self.sot_eve_buffer > 0 or self.sot_nii_buffer > 0:
            print(f"  [EVE buffer +{self.sot_eve_buffer:.1f}% T1 / NII buffer +{self.sot_nii_buffer:.1f}% T1]")
        if self.eve_breach_weight > 0 or self.nii_breach_weight > 0:
            print(f"  [Soft floors: EVE weight={self.eve_breach_weight:.1f} PLN/pp-T1"
                  f"  NII weight={self.nii_breach_weight:.1f} PLN/pp-T1 -- breaches allowed if EP gain outweighs]")
        print(f"  {self.message}")
        print(f"  Iterations: {self.n_iter}   Elapsed: {self.elapsed_s:.2f}s")
        print(
            f"  NII base : {self.nii_old / 1e6:>10,.2f}M  ->  {self.nii_new / 1e6:>10,.2f}M"
            f"   improvement: +{self.nii_improvement_m / 1e6:,.2f}M (+{self.nii_improvement_pct:.2f}%)"
            f"   [client-rate, informational -- see Margin below for what drives EP]"
        )
        print(
            f"  Margin   : {self.margin_old / 1e6:>10,.2f}M  ->  {self.margin_new / 1e6:>10,.2f}M"
            f"   (margin-over-FTP income)"
        )
        print(
            f"  Fee      : {self.fee_old / 1e6:>10,.2f}M  ->  {self.fee_new / 1e6:>10,.2f}M"
            f"   (non-interest fee income)"
        )
        print(
            f"  EL       : {self.el_old  / 1e6:>10,.2f}M  ->  {self.el_new  / 1e6:>10,.2f}M"
        )
        print(
            f"  CoC      : {self.coc_old / 1e6:>10,.2f}M  ->  {self.coc_new / 1e6:>10,.2f}M"
        )
        print(
            f"  OpEx     : {self.opex_old / 1e6:>10,.2f}M  ->  {self.opex_new / 1e6:>10,.2f}M"
        )
        print(
            f"  AcqCost  : {self.acq_cost_old / 1e6:>10,.2f}M  ->  {self.acq_cost_new / 1e6:>10,.2f}M"
            f"   (marketing cost, only on growth above baseline)"
        )
        sign = "+" if self.ep_improvement_m >= 0 else ""
        print(
            f"  Econ. P. : {self.ep_old  / 1e6:>10,.2f}M  ->  {self.ep_new  / 1e6:>10,.2f}M"
            f"   improvement: {sign}{self.ep_improvement_m / 1e6:,.2f}M ({sign}{self.ep_improvement_pct:.2f}%)"
        )

        print(f"\n  EVE SOT  (effective floor {self.sot_eve_floor:.1f}% T1):")
        for s, v in sorted(self.sot_eve.items()):
            ok = "OK" if v >= self.sot_eve_floor else "!!"
            print(f"    {s:<16s}: {v:+7.2f}% T1  [{ok}]")

        print(f"  NII SOT  (effective floor {self.sot_nii_floor:.1f}% T1):")
        for s, v in sorted(self.sot_nii.items()):
            ok = "OK" if v >= self.sot_nii_floor else "!!"
            print(f"    {s:<16s}: {v:+7.2f}% T1  [{ok}]")

        for ccy, lcr_v in sorted(self.lcr.items()):
            ok = "OK" if not np.isnan(lcr_v) and lcr_v >= self.min_lcr else "!!"
            print(f"  LCR  {ccy}: {lcr_v:.3f}  [{ok}]")
        for ccy, nsfr_v in sorted(self.nsfr.items()):
            ok = "OK" if not np.isnan(nsfr_v) and nsfr_v >= self.min_nsfr else "!!"
            print(f"  NSFR {ccy}: {nsfr_v:.3f}  [{ok}]")
        rwa_ok = (self.min_t1_rwa <= 0.0 or self.t1_rwa_ratio >= self.min_t1_rwa)
        rwa_tag = "OK" if rwa_ok else "!!"
        print(f"  RWA: {self.rwa/1e6:,.0f}M PLN   T1/RWA: {self.t1_rwa_ratio*100:.1f}%"
              f"  (min {self.min_t1_rwa*100:.1f}%)  [{rwa_tag}]")

        if self.use_slack:
            n_viol = sum(1 for v in self.slack_values.values() if v > 1e-6)
            if n_viol > 0:
                print(f"\n  Slack analysis — {n_viol} constraint(s) infeasible without relaxation:")
                for k, v in sorted(self.slack_values.items()):
                    unit = "% T1" if k.startswith(("eve_", "nii_")) else "PLN"
                    tag  = "!!" if v > 1e-6 else "OK"
                    print(f"    {k:<20s}: violation = {v:+.6f} {unit}  [{tag}]")
            else:
                print("\n  Slack analysis: all regulatory constraints feasible (all slacks = 0)")

        if self.constraint_violations:
            print("\n  Constraint violations:")
            for k, v in self.constraint_violations.items():
                print(f"    {k}: {v:+.4f}")

        print("\n  Product weight changes:")
        top = sorted(self.product_changes, key=lambda c: abs(c.delta_pct), reverse=True)
        for c in top:
            if abs(c.delta_pct) < 0.0001:
                continue
            print(
                f"    {c.product_code:6s} {c.bs_side}  "
                f"{c.pct_old:6.2f}% -> {c.pct_new:6.2f}%  "
                f"({c.delta_pct:+.2f} pp)"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_IRS_CODE = "0000"


def _is_feasible(x: np.ndarray, constraints: list[dict], tol: float = 1e-4) -> bool:
    """Check every SLSQP-style constraint dict (as built in optimize_nii) holds
    at x. Used to reject SLSQP polish/restore iterates that "improve" the
    objective number but landed on an infeasible point (SLSQP failing to
    converge does not mean its last iterate satisfies the constraints it was
    given -- accepting it unconditionally was a real bug: it let a nominally-
    hard EVE/NII/LCR/NSFR/RWA floor get silently breached).
    """
    for c in constraints:
        val = c["fun"](x)
        if c["type"] == "ineq" and val < -tol:
            return False
        if c["type"] == "eq" and abs(val) > tol:
            return False
    return True


def _resolve_irs(params: BalanceSheetParams) -> tuple[np.ndarray, set[tuple[str, str]], float]:
    """IRS (product_code='0000') cohort mask, product set, and total notional.

    IRS is always pinned at its current notional wherever it's used — see the
    IRS handling comment in optimize_nii(). Factored out so joint_optimizer.py
    (which layers NEW swap-ladder notional on top of this same pinned book)
    stays consistent with optimize_nii without duplicating the logic.
    """
    irs_cohort_mask = np.array(
        [str(pc) == _IRS_CODE for pc in params.product_code], dtype=bool
    )
    irs_products = {
        (str(pc), str(side))
        for pc, side in zip(params.product_code, params.bs_side)
        if str(pc) == _IRS_CODE
    }
    irs_notional = float(params.balance_arr[irs_cohort_mask].sum())
    return irs_cohort_mask, irs_products, irs_notional


def _build_bounds(
    config: OptimizationConfig, pm: "_ProductMap", active_fixed: set[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-product (lb, ub) weight bounds by config.mode. Shared by optimize_nii
    and joint_optimizer.py so both interpret mode/fixed_products/max_shift/
    custom_bounds identically.
    """
    lb = np.zeros(pm.n_prod, dtype=float)
    ub = np.ones(pm.n_prod,  dtype=float)

    if config.mode == "full":
        for j, (pc, side) in enumerate(pm.products):
            if side == "E" or (pc, side) in active_fixed:
                lb[j] = ub[j] = pm.base_prod_w[j]

    elif config.mode == "partial":
        lb[:] = pm.base_prod_w
        ub[:] = pm.base_prod_w
        for j, (pc, side) in enumerate(pm.products):
            if side != "E" and (pc, side) not in active_fixed:
                lb[j] = 0.0
                ub[j] = 1.0

    elif config.mode == "max_shift":
        shift = config.max_shift
        lb = np.maximum(0.0, pm.base_prod_w - shift)
        ub = pm.base_prod_w + shift
        for j, (pc, side) in enumerate(pm.products):
            if side == "E" or (pc, side) in active_fixed:
                lb[j] = ub[j] = pm.base_prod_w[j]

    elif config.mode == "custom":
        for j, (pc, side) in enumerate(pm.products):
            if side == "E" or (pc, side) in active_fixed:
                lb[j] = ub[j] = pm.base_prod_w[j]
            elif (pc, side) in config.custom_bounds:
                lb[j], ub[j] = config.custom_bounds[(pc, side)]
            # else: free in [0, 1] (no custom bound provided for this product)

    else:
        raise ValueError(f"Unknown optimization mode: {config.mode!r}")

    return lb, ub


# ─────────────────────────────────────────────────────────────────────────────
# LP coefficient builder
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lp_coefficients(
    pm:              _ProductMap,
    params:          BalanceSheetParams,
    cr:              CohortRates,
    t1:              float,
    irs_cohort_mask: np.ndarray,
    include_irs:     bool,
    bias_eve,
    bias_nii,
    bias_scens,
    shocked_scens:   list[str],
    margin_rate:     np.ndarray,
    margin_rate_nb:  np.ndarray,
    fee_rate:        np.ndarray,
) -> tuple:
    """Compute per-product linear coefficients for the LP solve.

    The EP objective and EVE/NII-delta/NSFR constraints are all linear in the
    product weight vector x.  This function extracts those coefficients via one
    baseline evaluation plus n_prod small perturbations (one per free product).

    margin_rate / margin_rate_nb are the FTP-adjusted margin-over-FTP income
    rates (see ftp_store.margin_unit_rate) — used here INSTEAD of raw
    params.nii_unit_rate for the EP objective. This does NOT affect A_eve/
    A_nii_delta (still client-rate CF-based, via compute_eve_cf_fast_all/
    compute_nii_cf_all below) — those feed the regulatory EVE/NII SOT
    constraints, which stay client-rate based regardless of FTP.

    fee_rate is the static non-interest fee-income assumption per cohort
    (params.fee_unit_rate, from bank_data.xlsx) — same value for existing and
    new business (no new-biz variant exists for it), added on top of margin.

    Returns
    -------
    c_nii       : (n_prod,)    margin-over-FTP income per unit of x_j [PLN/fraction]
    A_eve       : (n_scen, n_prod)  d(delta_EVE/T1*100) / d(x_j)
    A_nii_delta : (n_scen, n_prod)  d(delta_NII/T1*100) / d(x_j)
    A_nsfr      : (n_prod,)    RSF - ASF per unit x_j  [constraint: <= 0]
    c_rwa       : (n_prod,)    RWA contribution per unit of x_j [PLN/fraction]
    c_el        : (n_prod,)    EL contribution per unit of x_j [PLN/fraction]
    c_ep        : (n_prod,)    EP (book margin) = Margin + Fee - EL - CoC - OpEx per unit of x_j
    c_ep_lp     : (n_prod,)    LP objective: EP(new-biz margin) + linearised elasticity
    c_fee       : (n_prod,)    fee income per unit of x_j [PLN/fraction]
    """
    TA = params.total_assets
    n_scen = len(shocked_scens)

    # OpEx: flat OPEX_RATE on every non-equity product (see module constant) ──
    c_opex = np.where(pm.equity_mask, 0.0, OPEX_RATE) * TA

    # EP income objective: exactly linear in x via margin_rate (client rate
    # net of FTP) instead of raw nii_unit_rate ─────────────────────────────────
    c_nii = np.bincount(
        pm.cohort_to_prod,
        weights=pm.fraction * margin_rate,
        minlength=pm.n_prod,
    ) * TA

    # NSFR: ASF - RSF >= 0  ↔  (RSF - ASF) @ x <= 0  ─────────────────────────
    asf_coeff = np.bincount(
        pm.cohort_to_prod, weights=pm.fraction * params.asf_factor, minlength=pm.n_prod
    ) * TA
    rsf_coeff = np.bincount(
        pm.cohort_to_prod, weights=pm.fraction * params.rsf_factor, minlength=pm.n_prod
    ) * TA
    A_nsfr = rsf_coeff - asf_coeff  # constraint: A_nsfr @ x <= 0

    # RWA: Σ rwa_factor_i * amount_i — linear in x  ───────────────────────────
    c_rwa = np.bincount(
        pm.cohort_to_prod, weights=pm.fraction * params.rwa_factor, minlength=pm.n_prod
    ) * TA

    # EL: Σ el_unit_i * amount_i — linear in x  ──────────────────────────────
    c_el = np.bincount(
        pm.cohort_to_prod, weights=pm.fraction * params.el_unit, minlength=pm.n_prod
    ) * TA

    # Fee income: Σ fee_unit_rate_i * amount_i — linear in x, same for existing
    # and new business (no new-biz fee variant in the xlsx) ──────────────────
    c_fee = np.bincount(
        pm.cohort_to_prod, weights=pm.fraction * fee_rate, minlength=pm.n_prod
    ) * TA

    # EP = Margin + Fee - EL - CoC - OpEx  (CoC = RWA × CET1_target × coc_rate)
    c_ep = c_nii + c_fee - c_el - c_rwa * params.cet1_target * params.coc_rate - c_opex

    # New-business margin income (FTP-adjusted; new-biz client rate is still a
    # proxy = book rate, per the existing placeholder -- only the FTP side of
    # the spread reflects today's curve)
    c_nii_nb = np.bincount(
        pm.cohort_to_prod,
        weights=pm.fraction * margin_rate_nb,
        minlength=pm.n_prod,
    ) * TA

    # LP objective: EP with new-biz rates + linearised elasticity at x_base
    # First-order Taylor of Σ elast_j × (x_j - x_base_j) × x_j around x_base:
    #   gradient_j = elast_j × x_base_j   (correction is zero at baseline)
    c_ep_nb = c_nii_nb + c_fee - c_el - c_rwa * params.cet1_target * params.coc_rate - c_opex
    c_ep_lp = c_ep_nb + pm.vol_elast_prod * pm.base_prod_w * TA

    # EVE and NII-delta: linearise numerically (one pass per product) ──────────
    A_eve       = np.zeros((n_scen, pm.n_prod))
    A_nii_delta = np.zeros((n_scen, pm.n_prod))

    base_amounts = pm.to_amounts(pm.base_prod_w, TA)
    if not include_irs:
        base_amounts = np.where(irs_cohort_mask, 0.0, base_amounts)

    base_eve = compute_eve_cf_fast_all(base_amounts, params, cr)
    base_nii = compute_nii_cf_all(base_amounts, params, cr)
    if bias_eve is not None:
        apply_eve_bias(base_eve, base_amounts, bias_eve, bias_scens)
        apply_nii_bias(base_nii, base_amounts, bias_nii, bias_scens)

    nii_base_0 = base_nii["base"]
    base_delta_nii = {s: base_nii.get(s, nii_base_0) - nii_base_0 for s in shocked_scens}

    eps = 1e-4  # weight-space step; large enough for numerical stability
    for j in range(pm.n_prod):
        xp = pm.base_prod_w.copy()
        xp[j] += eps
        ap = pm.to_amounts(xp, TA)
        if not include_irs:
            ap = np.where(irs_cohort_mask, 0.0, ap)

        eve_p = compute_eve_cf_fast_all(ap, params, cr)
        nii_p = compute_nii_cf_all(ap, params, cr)
        if bias_eve is not None:
            apply_eve_bias(eve_p, ap, bias_eve, bias_scens)
            apply_nii_bias(nii_p, ap, bias_nii, bias_scens)

        nii_base_p = nii_p["base"]
        for k, s in enumerate(shocked_scens):
            A_eve[k, j] = (eve_p.get(s, 0.0) - base_eve.get(s, 0.0)) / eps / t1 * 100.0
            delta_p = nii_p.get(s, nii_base_p) - nii_base_p
            A_nii_delta[k, j] = (delta_p - base_delta_nii[s]) / eps / t1 * 100.0

    return c_nii, A_eve, A_nii_delta, A_nsfr, c_rwa, c_el, c_ep, c_ep_lp, c_fee


# ─────────────────────────────────────────────────────────────────────────────
# Substitution constraint builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_subst_constraints(
    pm:     "_ProductMap",
    params: BalanceSheetParams,
) -> tuple:
    """Build substitution (cannibalism) LP rows and SLSQP constraints.

    For each pair (src → dst) with rate r:
        When dest grows by Δw_dst, src must shrink by at least r × Δw_dst.
        Constraint: w_src + r × w_dst >= w_base_src + r × w_base_dst
        LP row:    -w_src - r × w_dst  <=  -(w_base_src + r × w_base_dst)

    Returns (A_subst, b_subst, slsqp_constraints).
    A_subst is None when no valid pairs exist.
    """
    if len(params.subst_rates) == 0:
        return None, None, []

    prod_idx = {(str(pc), str(side)): j for j, (pc, side) in enumerate(pm.products)}
    rows_A: list = []
    rows_b: list = []
    slsqp: list  = []

    for k in range(len(params.subst_rates)):
        src_key = (str(params.subst_src_pc[k]),   str(params.subst_src_side[k]))
        dst_key = (str(params.subst_dst_pc[k]),   str(params.subst_dst_side[k]))
        rate    = float(params.subst_rates[k])
        j_src   = prod_idx.get(src_key)
        j_dst   = prod_idx.get(dst_key)
        if j_src is None or j_dst is None:
            continue

        row         = np.zeros(pm.n_prod, dtype=float)
        row[j_src]  = -1.0
        row[j_dst]  = -rate
        base_val    = float(pm.base_prod_w[j_src]) + rate * float(pm.base_prod_w[j_dst])
        rows_A.append(row)
        rows_b.append(-base_val)

        # SLSQP: ineq = x_src + rate*x_dst - base_val >= 0
        slsqp.append({
            "type": "ineq",
            "fun": lambda x, _js=j_src, _jd=j_dst, _r=rate, _b=base_val: (
                x[_js] + _r * x[_jd] - _b
            ),
        })

    if not rows_A:
        return None, None, []

    return np.vstack(rows_A), np.array(rows_b, dtype=float), slsqp


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def optimize_nii(
    config: OptimizationConfig,
    params: BalanceSheetParams,
    cr:     CohortRates,
) -> OptimizationResult:
    """Maximize NII base (method B) subject to regulatory constraints.

    Parameters
    ----------
    config  : OptimizationConfig — mode, thresholds, shift bounds
    params  : BalanceSheetParams — frozen parameter arrays loaded from npz
    cr      : CohortRates — CF-based rate/discount tables loaded from npz

    Returns
    -------
    OptimizationResult — solution weights, regulatory metrics, product changes
    """
    ta  = params.total_assets
    t1  = (config.tier1_capital if config.tier1_capital > 0
           else float(params.balance_arr[params.is_equity].sum()))
    pm  = _ProductMap(params)

    # ── IRS handling ──────────────────────────────────────────────────────────
    # IRS (product_code='0000') represent existing derivative contracts.
    # They are always pinned at current notional — we never let the optimizer
    # vary them freely.  The include_irs flag controls whether their EVE/NII
    # contribution is counted in constraint evaluation:
    #   True  = hedged view  (IRS offsets banking-book duration risk)
    #   False = unhedged view (optimizer works on pure banking book)
    irs_cohort_mask, irs_products, irs_notional = _resolve_irs(params)
    # Build active fixed set: user-pinned products + IRS (always pinned)
    active_fixed = config.fixed_products | irs_products

    if irs_products:
        irs_view = "hedged (IRS in metrics)" if config.include_irs else "unhedged (IRS excluded from metrics)"
        print(f"  IR derivatives: {irs_view}"
              f"  |  notional {irs_notional/1e6:,.0f}M  |  {len(irs_products)} cohort(s) — always pinned")
    else:
        print("  IR derivatives: none found in params (product_code='0000' not present)")

    # Load per-product per-scenario bias corrections (exact − method B)
    _bias_eve, _bias_nii, _bias_scens = load_bias_corrections()
    if _bias_eve is not None:
        print(f"  Bias corrections loaded: {len(_bias_scens)} scenarios, {_bias_eve.shape[0]} cohorts")
        print(f"    EVE correction at baseline: "
              + "  ".join(f"{s}={float(np.dot(params.balance_arr, _bias_eve[:,j]))/1e6:+.1f}M"
                          for j, s in enumerate(_bias_scens)))
    else:
        print("  No bias corrections — run accuracy_check.py to improve SOT accuracy")

    # Load per-cohort FTP rates (market + liquidity component, see ftp_store.py).
    # Missing cache degrades gracefully to zero FTP (margin == client-rate NII,
    # today's behavior) rather than failing — run build_ftp_rates.py to populate it.
    _ftp_rate = load_ftp_rates(params.cohort_id)
    if _ftp_rate is None:
        print("  No FTP cache found — run build_ftp_rates.py; EP margin will equal client-rate NII until then")
        _ftp_rate = np.zeros_like(params.nii_unit_rate)
    else:
        print(f"  FTP rates loaded: mean {float(np.mean(_ftp_rate[_ftp_rate > 0]))*100:.2f}% "
              f"(market + liquidity component)")
    _margin_rate    = margin_unit_rate(params.nii_unit_rate,          _ftp_rate, params.bs_side)
    _margin_rate_nb = margin_unit_rate(params.nii_unit_rate_new_biz,  _ftp_rate, params.bs_side)
    _opex_mask = ~params.is_equity   # opex applies to every non-equity product, A and L

    # Non-interest fee income (static per-product assumption from bank_data.xlsx's
    # bs_structure sheet, e.g. mortgage/cash-loan servicing fees). Zero for
    # products with no fee assumption (deposits, bonds, equity) — same
    # graceful-default convention as PD/LGD/vol_elasticity.
    _fee_rate = params.fee_unit_rate

    # Baseline NII / Margin / EL / CoC / OpEx / EP for reference (method B at current balance_arr)
    _base_amounts  = (params.balance_arr if config.include_irs
                      else np.where(irs_cohort_mask, 0.0, params.balance_arr))
    _base_nii_dict = compute_nii_cf_all(_base_amounts, params, cr)
    nii_base_old   = _base_nii_dict["base"]           # client-rate NII (Method B) — informational/SOT only
    margin_base_old = float(np.dot(_base_amounts, _margin_rate))   # margin-over-FTP income — drives EP
    fee_base_old   = float(np.dot(_base_amounts, _fee_rate))       # non-interest fee income — drives EP
    _base_rwa      = float(np.dot(_base_amounts, params.rwa_factor))
    el_base_old    = float(np.dot(_base_amounts, params.el_unit))
    coc_base_old   = _base_rwa * params.cet1_target * params.coc_rate
    opex_base_old  = OPEX_RATE * float(np.sum(_base_amounts[_opex_mask]))
    acq_cost_base_old = 0.0   # growth above baseline is 0 AT baseline, by definition
    ep_base_old    = margin_base_old + fee_base_old - el_base_old - coc_base_old - opex_base_old - acq_cost_base_old

    # Effective floors = regulatory floor minus buffer (positive buffer loosens)
    eve_floor_eff = config.sot_eve_floor - config.sot_eve_buffer
    nii_floor_eff = config.sot_nii_floor - config.sot_nii_buffer
    shocked_scens = [str(s) for s in cr.rate_scenario_ids if str(s) != "base"]
    n_scen        = len(shocked_scens)

    # Soft-breach mode: a floor becomes a weighted objective penalty instead of a
    # hard constraint. Mutually exclusive with use_slack (diagnostic mode wins —
    # it needs every regulatory row slacked uniformly to find ANY feasible point).
    soft_eve = (not config.use_slack) and config.eve_breach_weight > 0.0
    soft_nii = (not config.use_slack) and config.nii_breach_weight > 0.0
    w_eve = config.eve_breach_weight * (t1 / 100.0) / n_scen if soft_eve else 0.0
    w_nii = config.nii_breach_weight * (t1 / 100.0) / n_scen if soft_nii else 0.0
    if soft_eve or soft_nii:
        print(f"  Soft floors: EVE={'weight ' + format(config.eve_breach_weight, '.1f') if soft_eve else 'HARD'}"
              f"   NII={'weight ' + format(config.nii_breach_weight, '.1f') if soft_nii else 'HARD'}"
              f"   (EP still dominates unless breach severity is large)")

    # ── Evaluation cache — SLSQP calls objective + all constraints at same x ──
    _cache: dict = {"x": None, "nii": None, "eve": None, "lcr": None, "nsfr": None, "rwa": None}

    def _refresh(x: np.ndarray) -> None:
        if _cache["x"] is not None and np.array_equal(x, _cache["x"]):
            return
        amounts = pm.to_amounts(x, ta)
        _cache["x"] = x.copy()
        # Strip IRS from metric amounts when in unhedged view
        amounts_m = amounts if config.include_irs else np.where(irs_cohort_mask, 0.0, amounts)
        _cache["nii"]  = compute_nii_cf_all(amounts_m, params, cr)
        _cache["eve"]  = compute_eve_cf_fast_all(amounts_m, params, cr)
        _cache["lcr"]  = compute_lcr_fast(amounts_m, params)
        _cache["nsfr"] = compute_nsfr_fast(amounts_m, params)
        _cache["rwa"]  = compute_rwa_fast(amounts_m, params)
        if _bias_eve is not None:
            apply_eve_bias(_cache["eve"], amounts_m, _bias_eve, _bias_scens)
            apply_nii_bias(_cache["nii"], amounts_m, _bias_nii, _bias_scens)

    def _objective(x: np.ndarray) -> float:
        """EP objective: Margin + Fee + elasticity_correction - EL - CoC - OpEx - AcqCost - soft_breach_penalty.

        Mirrors the LP objective (c_ep_lp / _compute_lp_coefficients + the growth-
        cost epigraph augmentation below) exactly, so this QP polish step refines
        the SAME quantity the LP just optimized instead of silently drifting
        toward plain client-rate NII (2026-07-14 fix -- this function had never
        been updated for Margin/FTP (2026-07-09), OpEx (2026-07-09), or Fee/
        AcqCost (2026-07-14), even though it's the actual objective used whenever
        any product has non-zero vol_elasticity).
        """
        _refresh(x)
        amounts = pm.to_amounts(x, ta)
        margin = float(np.dot(amounts, _margin_rate))
        fee    = float(np.dot(amounts, _fee_rate))
        # Quadratic elasticity correction: Σ_j elast_j × (x_j - x_base_j) × x_j × TA
        # (client-rate effect -- applies to margin the same way it applies to NII,
        # since FTP doesn't move with volume; see the new-solution block below)
        elast_correction = float(np.dot(pm.vol_elast_prod * (x - pm.base_prod_w), x)) * ta
        el   = float(np.dot(amounts, params.el_unit))
        coc  = _cache["rwa"] * params.cet1_target * params.coc_rate
        opex = OPEX_RATE * float(np.sum(amounts[_opex_mask]))
        # Marketing/acquisition cost: only on growth above baseline weight (see
        # the LP epigraph augmentation for why this is g_j = max(0, x_j-base_j))
        acq_cost = float(np.dot(pm.acq_cost_prod, np.maximum(0.0, x - pm.base_prod_w))) * ta
        penalty = 0.0
        if soft_eve:
            for s in shocked_scens:
                breach = eve_floor_eff - _cache["eve"].get(s, 0.0) / t1 * 100.0
                if breach > 0.0:
                    penalty += w_eve * breach
        if soft_nii:
            # NII SOT stays client-rate based (regulatory constraint), unlike the
            # income term above -- see module docstring / ftp_store.py.
            nii_base_x = _cache["nii"]["base"]
            for s in shocked_scens:
                delta  = _cache["nii"].get(s, nii_base_x) - nii_base_x
                breach = nii_floor_eff - delta / t1 * 100.0
                if breach > 0.0:
                    penalty += w_nii * breach
        return -(margin + fee + elast_correction - el - coc - opex - acq_cost - penalty)

    # ── Constraints ──────────────────────────────────────────────────────────
    constraints: list[dict] = []

    # Balance sheet sum constraints (equality)
    constraints.append({
        "type": "eq",
        "fun": lambda x: float(x[pm.asset_mask].sum()) - pm.asset_sum,
    })
    constraints.append({
        "type": "eq",
        "fun": lambda x: float(x[pm.fund_mask].sum()) - pm.fund_sum,
    })

    # EVE SOT: delta_EVE[s] / T1 >= eve_floor_eff  (%) — hard only when not soft_eve
    if not soft_eve:
        for s in shocked_scens:
            def _c_eve(x, _s=s):
                _refresh(x)
                return _cache["eve"].get(_s, 0.0) / t1 * 100.0 - eve_floor_eff
            constraints.append({"type": "ineq", "fun": _c_eve})

    # NII SOT: delta_NII[s] / T1 >= nii_floor_eff  (%) — hard only when not soft_nii
    if not soft_nii:
        for s in shocked_scens:
            def _c_nii(x, _s=s):
                _refresh(x)
                delta = _cache["nii"].get(_s, _cache["nii"]["base"]) - _cache["nii"]["base"]
                return delta / t1 * 100.0 - nii_floor_eff
            constraints.append({"type": "ineq", "fun": _c_nii})

    # LCR per currency
    for ccy in params.unique_currencies:
        def _c_lcr(x, _c=ccy):
            _refresh(x)
            v = _cache["lcr"].get(_c, float("nan"))
            return (v - config.min_lcr) if not np.isnan(v) else 999.0
        constraints.append({"type": "ineq", "fun": _c_lcr})

    # NSFR per currency
    for ccy in params.unique_currencies:
        def _c_nsfr(x, _c=ccy):
            _refresh(x)
            v = _cache["nsfr"].get(_c, float("nan"))
            return (v - config.min_nsfr) if not np.isnan(v) else 999.0
        constraints.append({"type": "ineq", "fun": _c_nsfr})

    # T1/RWA floor -- was already enforced in the LP (A_reg row) but missing
    # here, so SLSQP polish/restore steps could silently drift past it.
    if config.min_t1_rwa > 0.0:
        def _c_rwa(x):
            _refresh(x)
            rwa_v = _cache["rwa"]
            t1_rwa_v = t1 / rwa_v if rwa_v > 0 else float("inf")
            return t1_rwa_v - config.min_t1_rwa
        constraints.append({"type": "ineq", "fun": _c_rwa})

    # Substitution/cannibalism: w_src + rate × w_dst >= baseline (linear)
    A_subst, b_subst, _subst_slsqp = _build_subst_constraints(pm, params)
    if _subst_slsqp:
        print(f"  Substitution constraints: {len(_subst_slsqp)} pairs active")
        constraints.extend(_subst_slsqp)

    # Flag whether QP polish is needed (any non-zero price-volume elasticity)
    _has_elasticity = bool(np.any(pm.vol_elast_prod != 0.0))
    if _has_elasticity:
        n_elast = int(np.count_nonzero(pm.vol_elast_prod))
        print(f"  Price-volume elasticity: {n_elast} products active — LP warm-start + QP polish")

    # ── Bounds by mode ───────────────────────────────────────────────────────
    lb, ub = _build_bounds(config, pm, active_fixed)
    bounds = list(zip(lb.tolist(), ub.tolist()))
    x0 = pm.base_prod_w.copy()

    # ── Primary solve: LP (linprog/HiGHS) ────────────────────────────────────
    # The problem is a true LP: linear NII objective, linear EVE/NII-delta
    # constraints (analytically linear in balance), linear NSFR.
    # LCR is also linear but treated post-hoc (it's always satisfied at LP opt).
    t_start = time.time()
    print("  Building LP coefficients...")
    c_nii, A_eve, A_nii_delta, A_nsfr, c_rwa, c_el, c_ep, c_ep_lp, c_fee = _compute_lp_coefficients(
        pm, params, cr, t1, irs_cohort_mask, config.include_irs,
        _bias_eve, _bias_nii, _bias_scens, shocked_scens,
        _margin_rate, _margin_rate_nb, _fee_rate,
    )

    # ── Regulatory inequality constraints ────────────────────────────────────────
    # In linprog "≤" form: -A_eve @ x ≤ -eve_floor, A_nsfr @ x ≤ 0, c_rwa @ x ≤ T1/cap
    # EVE/NII rows are only HARD here when not running in that floor's soft-breach
    # mode — when soft, they move to the weighted slack block built below instead.
    hard_rows, hard_rhs = [], []
    if not soft_eve:
        hard_rows.append(-A_eve);       hard_rhs.append(np.full(n_scen, -eve_floor_eff))
    if not soft_nii:
        hard_rows.append(-A_nii_delta); hard_rhs.append(np.full(n_scen, -nii_floor_eff))
    hard_rows.append(A_nsfr[None, :]);  hard_rhs.append(np.array([0.0]))
    n_rwa_slack = 0
    if config.min_t1_rwa > 0.0:
        hard_rows.append(c_rwa[None, :])
        hard_rhs.append(np.array([t1 / config.min_t1_rwa]))
        n_rwa_slack = 1
    A_reg = np.vstack(hard_rows)
    b_reg = np.concatenate(hard_rhs)
    n_reg = A_reg.shape[0]

    # Balance-sheet equality: assets sum to asset_sum, funding to fund_sum
    A_eq_lp = np.zeros((2, pm.n_prod))
    A_eq_lp[0, pm.asset_mask] = 1.0
    A_eq_lp[1, pm.fund_mask]  = 1.0
    b_eq_lp = np.array([pm.asset_sum, pm.fund_sum])

    # Soft-breach rows: EVE and/or NII floor(s) selected for soft treatment, each
    # weighted at its own PLN/pp-T1 cost (see eve_breach_weight/nii_breach_weight).
    soft_rows, soft_rhs, soft_w_blocks, soft_names = [], [], [], []
    if soft_eve:
        soft_rows.append(-A_eve);       soft_rhs.append(np.full(n_scen, -eve_floor_eff))
        soft_w_blocks.append(np.full(n_scen, w_eve))
        soft_names += [f"eve_{s}" for s in shocked_scens]
    if soft_nii:
        soft_rows.append(-A_nii_delta); soft_rhs.append(np.full(n_scen, -nii_floor_eff))
        soft_w_blocks.append(np.full(n_scen, w_nii))
        soft_names += [f"nii_{s}" for s in shocked_scens]

    # ── Slack augmentation ─────────────────────────────────────────────────────
    # Each slacked constraint g(x) ≤ b gets a slack s ≥ 0: g(x) - s ≤ b → s = max(0, g(x)-b).
    # Diagnostic mode (use_slack=True): ALL regulatory rows slacked at uniform Big-M,
    #   to find ANY feasible point regardless of cost — see slack_penalty docstring.
    # Soft-breach mode (eve/nii_breach_weight > 0): only the floor(s) the user chose
    #   are slacked, each at its own PLN/pp-T1 weight so EP can outweigh a modest
    #   breach. NSFR/RWA/substitution constraints stay hard in both modes.
    if config.use_slack:
        n_slack = n_reg
        slack_names = (
            [f"eve_{s}" for s in shocked_scens]
            + [f"nii_{s}" for s in shocked_scens]
            + ["nsfr"]
            + (["rwa"] if n_rwa_slack > 0 else [])
        )
        c_lp      = np.concatenate([-c_ep_lp, np.full(n_slack, config.slack_penalty)])
        A_reg_aug = np.hstack([A_reg, -np.eye(n_reg)])
        A_eq_aug  = np.hstack([A_eq_lp, np.zeros((2, n_slack))])
        if A_subst is not None:
            A_subst_aug = np.hstack([A_subst, np.zeros((len(A_subst), n_slack))])
            A_ub = np.vstack([A_reg_aug, A_subst_aug])
            b_ub = np.concatenate([b_reg, b_subst])
        else:
            A_ub = A_reg_aug
            b_ub = b_reg
        bounds_lp = list(zip(lb.tolist(), ub.tolist())) + [(0.0, None)] * n_slack
        print(f"  Slack mode: {n_slack} slack variable(s) added  (M = {config.slack_penalty:.0e})")
    elif soft_names:
        n_slack     = len(soft_names)
        slack_names = soft_names
        A_soft      = np.vstack(soft_rows)
        b_soft      = np.concatenate(soft_rhs)
        w_soft      = np.concatenate(soft_w_blocks)
        c_lp        = np.concatenate([-c_ep_lp, w_soft])
        A_hard_aug  = np.hstack([A_reg,  np.zeros((n_reg,  n_slack))])
        A_soft_aug  = np.hstack([A_soft, -np.eye(n_slack)])
        A_eq_aug    = np.hstack([A_eq_lp, np.zeros((2, n_slack))])
        if A_subst is not None:
            A_subst_aug = np.hstack([A_subst, np.zeros((len(A_subst), n_slack))])
            A_ub = np.vstack([A_hard_aug, A_soft_aug, A_subst_aug])
            b_ub = np.concatenate([b_reg, b_soft, b_subst])
        else:
            A_ub = np.vstack([A_hard_aug, A_soft_aug])
            b_ub = np.concatenate([b_reg, b_soft])
        bounds_lp = list(zip(lb.tolist(), ub.tolist())) + [(0.0, None)] * n_slack
        print(f"  Soft-breach mode: {n_slack} weighted slack variable(s) added")
    else:
        n_slack     = 0
        slack_names = []
        c_lp        = -c_ep_lp
        A_eq_aug    = A_eq_lp
        if A_subst is not None:
            A_ub = np.vstack([A_reg, A_subst])
            b_ub = np.concatenate([b_reg, b_subst])
        else:
            A_ub = A_reg
            b_ub = b_reg
        bounds_lp = bounds

    # ── Growth-cost epigraph augmentation ────────────────────────────────────
    # Marketing/acquisition cost (2026-07-14): products with a non-zero
    # acq_cost_rate get charged that rate only on the GROWTH of their weight
    # above baseline (pm.base_prod_w), not on the whole balance -- unlike
    # OpEx, which is flat on the whole book. max(0, x_j - base_j) is not
    # itself linear, but its MINIMUM over an auxiliary variable is a standard
    # LP epigraph reformulation:
    #     g_j >= x_j - base_j   (row below)
    #     g_j >= 0              (bound below)
    #     minimize ... + acq_cost_prod_j * TA * g_j
    # Since g_j only has a cost (never a benefit) in a minimization, the
    # solver always drives it down to exactly max(0, x_j - base_j) at
    # optimality -- never more. Inserted as its own variable block BETWEEN
    # x (n_prod) and any slack the three branches above appended, so none of
    # that branch-specific code needed to change. Always added (even when
    # acq_cost_prod is all zero) so the [x, g, s] variable layout is constant.
    n_slack_cols = A_ub.shape[1] - pm.n_prod
    A_ub = np.hstack([
        A_ub[:, :pm.n_prod], np.zeros((A_ub.shape[0], pm.n_prod)), A_ub[:, pm.n_prod:]
    ])
    A_eq_aug = np.hstack([
        A_eq_aug[:, :pm.n_prod], np.zeros((A_eq_aug.shape[0], pm.n_prod)), A_eq_aug[:, pm.n_prod:]
    ])
    A_growth = np.hstack([
        np.eye(pm.n_prod), -np.eye(pm.n_prod), np.zeros((pm.n_prod, n_slack_cols))
    ])
    A_ub = np.vstack([A_ub, A_growth])
    b_ub = np.concatenate([b_ub, pm.base_prod_w])
    c_lp = np.concatenate([c_lp[:pm.n_prod], pm.acq_cost_prod * ta, c_lp[pm.n_prod:]])
    bounds_lp = bounds_lp[:pm.n_prod] + [(0.0, None)] * pm.n_prod + bounds_lp[pm.n_prod:]

    lp_res = linprog(
        c_lp,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq_aug, b_eq=b_eq_lp,
        bounds=bounds_lp,
        method="highs",
    )
    elapsed = time.time() - t_start
    lp_nit = getattr(lp_res, "nit", 0)

    s_sol: np.ndarray = np.array([])
    if lp_res.success:
        z_sol = lp_res.x
        x_sol = z_sol[:pm.n_prod]
        # z_sol layout is [x (n_prod), g (n_prod, growth-cost epigraph), s (n_slack)]
        s_sol = z_sol[2 * pm.n_prod:] if n_slack > 0 else np.array([])
        print(f"  LP (HiGHS): optimal  nit={lp_nit}  elapsed={elapsed:.1f}s")
    else:
        if config.use_slack:
            # Slack makes the LP always feasible; this path indicates a numerical issue
            print(f"  LP (HiGHS) failed even with slack ({lp_res.message}) — returning baseline")
            x_sol = pm.base_prod_w.copy()
        else:
            # Fallback: SLSQP from x0 (should rarely be needed)
            print(f"  LP (HiGHS) failed ({lp_res.message}), falling back to SLSQP...")
            _slsqp_opts = {"maxiter": config.max_iter, "ftol": config.tol,
                           "eps": 1e-5, "disp": False}
            fb_res = minimize(
                _objective, x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options=_slsqp_opts,
            )
            x_sol = fb_res.x
            print(f"  SLSQP fallback: success={fb_res.success}  msg={fb_res.message}")

    # ── QP polish: only in normal mode (not slack/diagnosis mode) ────────────
    if _has_elasticity and lp_res.success and not config.use_slack:
        print("  QP polish (price-volume elasticity)...")
        _qp_opts = {"maxiter": config.max_iter, "ftol": config.tol, "eps": 1e-5, "disp": False}
        qp_res = minimize(
            _objective, x_sol,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=_qp_opts,
        )
        qp_improves = qp_res.success or (np.isfinite(qp_res.fun) and qp_res.fun < _objective(x_sol))
        if qp_improves and _is_feasible(qp_res.x, constraints):
            x_sol = qp_res.x
            print(f"  SLSQP QP: success={qp_res.success}  msg={qp_res.message}")
        elif qp_improves:
            print(f"  SLSQP QP: rejected -- iterate improves the objective but violates a hard "
                  f"constraint (success={qp_res.success}, msg={qp_res.message}); keeping LP solution")

    # Post-hoc LCR check — skipped in slack mode (diagnosis only)
    _refresh(x_sol)
    if not config.use_slack:
        lcr_ok = all(
            _cache["lcr"].get(ccy, float("nan")) >= config.min_lcr
            for ccy in params.unique_currencies
            if not np.isnan(_cache["lcr"].get(ccy, float("nan")))
        )
        if not lcr_ok:
            print("  LCR violated at LP solution — running SLSQP warm-start to restore...")
            _slsqp_opts = {"maxiter": config.max_iter, "ftol": config.tol,
                           "eps": 1e-5, "disp": False}
            lcr_res = minimize(
                _objective, x_sol,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options=_slsqp_opts,
            )
            if _is_feasible(lcr_res.x, constraints):
                x_sol = lcr_res.x
                print(f"  SLSQP warm-start: success={lcr_res.success}  msg={lcr_res.message}")
            else:
                print(f"  SLSQP warm-start: rejected -- iterate still violates a hard constraint "
                      f"(success={lcr_res.success}, msg={lcr_res.message}); keeping prior solution "
                      f"(LCR may remain violated -- see feasibility/violations in the result)")
            _refresh(x_sol)

    # Build slack dict (populated when use_slack=True, or in soft-breach mode)
    slack_values: dict[str, float] = {}
    if len(s_sol) > 0 and len(slack_names) > 0:
        for name, val in zip(slack_names, s_sol):
            slack_values[name] = float(val)
        n_violated = sum(1 for v in slack_values.values() if v > 1e-6)
        _label = "Slack diagnosis" if config.use_slack else "Soft-breach severity"
        if n_violated > 0:
            print(f"  {_label}: {n_violated} constraint(s) violated")
            for k, v in slack_values.items():
                if v > 1e-6:
                    print(f"    {k}: violation = {v:.6f}")
        else:
            print(f"  {_label}: all regulatory constraints feasible")

    nii_cf_new      = _cache["nii"]["base"]
    # Elasticity correction: rate penalty from volume changes at the solution.
    # Applies to margin income the same way it did to NII (it's a client-rate
    # effect; FTP doesn't move with volume).
    elast_corr_new  = float(np.dot(pm.vol_elast_prod * (x_sol - pm.base_prod_w), x_sol)) * ta
    nii_base_new    = nii_cf_new + elast_corr_new   # NII (client-rate, Method B) — informational/SOT only
    nii_improvement = nii_base_new - nii_base_old
    _sol_amounts    = pm.to_amounts(x_sol, ta)
    margin_new      = float(np.dot(_sol_amounts, _margin_rate)) + elast_corr_new   # margin-over-FTP — drives EP
    fee_new         = float(np.dot(_sol_amounts, _fee_rate))                       # fee income — drives EP
    el_new          = float(np.dot(_sol_amounts, params.el_unit))
    coc_new         = _cache["rwa"] * params.cet1_target * params.coc_rate
    opex_new        = OPEX_RATE * float(np.sum(_sol_amounts[_opex_mask]))
    # Marketing/acquisition cost: only on growth above baseline weight (see the
    # LP epigraph augmentation above for why this equals max(0, x_j-base_j))
    acq_cost_new    = float(np.dot(pm.acq_cost_prod, np.maximum(0.0, x_sol - pm.base_prod_w))) * ta
    ep_new          = margin_new + fee_new - el_new - coc_new - opex_new - acq_cost_new
    ep_improvement  = ep_new - ep_base_old

    sot_eve_out: dict[str, float] = {}
    sot_nii_out: dict[str, float] = {}
    for s in shocked_scens:
        sot_eve_out[s] = _cache["eve"].get(s, 0.0) / t1 * 100.0
        sot_nii_out[s] = (_cache["nii"].get(s, nii_base_new) - nii_base_new) / t1 * 100.0

    # Feasibility — checked against EFFECTIVE floors (with buffer applied)
    feasible   = True
    violations: dict[str, float] = {}
    for s, v in sot_eve_out.items():
        slack = v - eve_floor_eff
        if slack < -1e-5:
            feasible = False
            violations[f"eve_{s}"] = slack
    for s, v in sot_nii_out.items():
        slack = v - nii_floor_eff
        if slack < -1e-5:
            feasible = False
            violations[f"nii_{s}"] = slack
    for ccy, lcr_v in _cache["lcr"].items():
        if not np.isnan(lcr_v):
            slack = lcr_v - config.min_lcr
            if slack < -1e-5:
                feasible = False
                violations[f"lcr_{ccy}"] = slack
    for ccy, nsfr_v in _cache["nsfr"].items():
        if not np.isnan(nsfr_v):
            slack = nsfr_v - config.min_nsfr
            if slack < -1e-5:
                feasible = False
                violations[f"nsfr_{ccy}"] = slack
    rwa_val = _cache["rwa"]
    t1_rwa  = t1 / rwa_val if rwa_val > 0 else float("nan")
    if config.min_t1_rwa > 0.0 and not np.isnan(t1_rwa):
        slack = t1_rwa - config.min_t1_rwa
        if slack < -1e-5:
            feasible = False
            violations["t1_rwa"] = slack

    changes = [
        ProductChange(
            product_code = pc,
            bs_side      = side,
            weight_old   = float(pm.base_prod_w[j]),
            weight_new   = float(x_sol[j]),
            delta_weight = float(x_sol[j] - pm.base_prod_w[j]),
            pct_old      = float(pm.base_prod_w[j]) * 100.0,
            pct_new      = float(x_sol[j])           * 100.0,
            delta_pct    = float(x_sol[j] - pm.base_prod_w[j]) * 100.0,
        )
        for j, (pc, side) in enumerate(pm.products)
    ]

    return OptimizationResult(
        success               = lp_res.success or _has_elasticity,
        message               = lp_res.message,
        mode                  = config.mode,
        n_iter                = lp_nit,
        elapsed_s             = elapsed,
        nii_old               = nii_base_old,
        nii_new               = nii_base_new,
        nii_improvement_m     = nii_improvement,
        nii_improvement_pct   = (
            nii_improvement / abs(nii_base_old) * 100.0
            if nii_base_old != 0.0 else float("nan")
        ),
        margin_old            = margin_base_old,
        margin_new            = margin_new,
        fee_old               = fee_base_old,
        fee_new               = fee_new,
        el_old                = el_base_old,
        el_new                = el_new,
        coc_old               = coc_base_old,
        coc_new               = coc_new,
        opex_old              = opex_base_old,
        opex_new              = opex_new,
        acq_cost_old          = acq_cost_base_old,
        acq_cost_new          = acq_cost_new,
        ep_old                = ep_base_old,
        ep_new                = ep_new,
        ep_improvement_m      = ep_improvement,
        ep_improvement_pct    = (
            ep_improvement / abs(ep_base_old) * 100.0
            if ep_base_old != 0.0 else float("nan")
        ),
        product_changes       = changes,
        sot_eve               = sot_eve_out,
        sot_nii               = sot_nii_out,
        lcr                   = dict(_cache["lcr"]),
        nsfr                  = dict(_cache["nsfr"]),
        rwa                   = rwa_val,
        t1_rwa_ratio          = t1_rwa,
        feasible              = feasible,
        constraint_violations = violations,
        weights_old           = pm.base_cohort_w,
        weights_new           = pm.to_amounts(x_sol, ta) / ta,
        sot_eve_floor         = eve_floor_eff,
        sot_nii_floor         = nii_floor_eff,
        sot_eve_buffer        = config.sot_eve_buffer,
        sot_nii_buffer        = config.sot_nii_buffer,
        include_irs           = config.include_irs,
        tier1_capital         = t1,
        min_lcr               = config.min_lcr,
        min_nsfr              = config.min_nsfr,
        min_t1_rwa            = config.min_t1_rwa,
        use_slack             = config.use_slack,
        slack_values          = slack_values,
        eve_breach_weight     = config.eve_breach_weight,
        nii_breach_weight     = config.nii_breach_weight,
    )
