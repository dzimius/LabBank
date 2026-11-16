"""joint_optimizer.py
=====================
Joint LP: balance-sheet product weights (bs_optimizer.py) + NEW incremental
swap-ladder notional (swap_ladder.py's 300 buckets, the same ones
hedge_optimizer.py sizes), optimized TOGETHER against the same 7 official
EBA SOT scenarios bs_optimizer.py uses (base + par_up/dn, steep, flat,
sr_up/dn, own):

    max   EP_bs(x_bs)
          - eve_breach_weight * sum(eve_slack)  -  nii_breach_weight * sum(nii_slack)
    s.t.  balance-sheet identity, NSFR, RWA cap, substitution   (x_bs only, hard)
          EVE_SOT[s] = A_eve_bs[s] @ x_bs + A_eve_swap[s] @ x_swap + eve_slack[s] >= eve_floor
          NII_SOT[s] = A_nii_bs[s] @ x_bs + A_nii_swap[s] @ x_swap + nii_slack[s] >= nii_floor
          x_bs bounds per bs_config.mode (full/partial/max_shift/custom)
          0 <= x_swap[i] <= notional_cap ;  sum(x_swap) <= total_notional_cap

x_swap's own carry (EP_carry_swap) is deliberately NOT in the objective
(2026-07-09 change, per user feedback): the swap overlay is included ONLY
via its effect on the EVE/NII constraint rows above (the A_eve_swap/
A_nii_swap terms), never rewarded directly for its own P&L. Before this
change the objective was `EP_bs + EP_carry_swap - breach_penalty`, which let
the optimizer treat the new overlay as a second, largely unconstrained
profit source and buy up to the notional cap chasing carry (compounded by
two now-fixed swap_ladder.py mispricing issues -- IRS_MARGIN_BPS and
seasoned_notional_cap, see that module). With carry removed, a swap bucket
only enters the solution while it's still reducing binding breach severity;
once a row is satisfied, growing that bucket further is objective-neutral,
so there's no incentive to keep piling in. `ep_carry_swap` is still computed
and reported on JointResult for transparency -- it's informational now, not
part of what's maximized (mirrors the same change in hedge_optimizer.py).

This is bs_optimizer.py's "Approach 1" (pure BS, EP + weighted breach) with
the swap ladder unfrozen too -- i.e. "Approach 2": best balance-sheet
structure AND best new-hedge overlay, chosen at the same time instead of
hedge_optimizer.py's two-step "fix the book, then size a ladder on top of
it" (see hedge_optimizer.py's docstring for why IT keeps the book fixed --
this module deliberately does NOT, and that's the whole point of it).

Scope / what this does NOT do (both matching hedge_optimizer.py's existing
scope, not a new gap introduced here):
  - Does not re-derive the EXISTING ~840M IRS book. bs_optimizer always pins
    it inside x_bs (see _resolve_irs); x_swap is NEW notional layered on top.
  - NSFR / LCR / RWA are modeled on the banking-book side only. The swap
    overlay's notional isn't assumed to move those ratios.
  - No price-volume elasticity QP polish (bs_optimizer's nonlinear step for
    a handful of rate-sensitive products) -- this is a pure LP. Fine for the
    EVE/NII/EP trade-off question this module answers; re-run bs_optimizer
    directly if elasticity-accurate volumes are what you need.

Scenario alignment note: swap buckets are priced here via swap_ladder's
CurveTensors path (same tensor nii_eve_cf_fast.py uses for every other
product), against the SAME 7 scenarios as bs_optimizer -- NOT the 165-curve
robustness bank hedge_optimizer.py/price_ladder_against_curve_bank uses
(that's a supplementary basis-risk stress test over ALTERNATIVE starting
curves, a different question from "what does today's official EBA SOT test
show for this candidate hedge").
"""
from __future__ import annotations

import os as _o, sys as _s   # python_code/<group>/ layout: every group folder on sys.path
_PY = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))   # bs_optimization/python_code
for _g in ("optimizers", "scenarios", "hedging", "reporting"):
    if _o.path.join(_PY, _g) not in _s.path:
        _s.path.insert(0, _o.path.join(_PY, _g))

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, os.path.normpath(os.path.join(_PY, "..", "..")))
from labbank_common.bank_config import get_bank, home_ccy
OPTPREP_OUT = str(get_bank().output_dir("optimize_prep"))   # runs/<bank_id>/optimize_prep

_HERE    = _PY
_OPTPREP = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "python_code"))
for _p in (_HERE, _OPTPREP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bs_vector import BalanceSheetParams, CohortRates
from nii_eve_cf_fast import compute_nii_cf_all, compute_eve_cf_fast_all
from lcr_fast  import compute_lcr_fast
from nsfr_fast import compute_nsfr_fast
from rwa_fast  import compute_rwa_fast
from bias_store import load_bias_corrections, apply_eve_bias, apply_nii_bias
from ftp_store  import load_ftp_rates, margin_unit_rate, equity_benefit_unit_rate

import swap_ladder as sl                                                   # noqa: E402
from bs_optimizer import (                                                 # noqa: E402
    OptimizationConfig, ProductChange, OPEX_RATE,
    _ProductMap, _resolve_irs, _build_bounds,
    _compute_lp_coefficients, _build_subst_constraints,
)
from product_map import opex_mask as _opex_rows                            # noqa: E402

NOTIONAL_UNIT = 1e6
# Same conditioning fix as hedge_optimizer.py: swap sensitivities are
# ~1e-10 PLN/T1 per PLN notional while notional bounds are ~1e9 PLN -- solve
# in units of 1e6 PLN so every LP coefficient sits in a comparable range.


@dataclass
class JointOptions:
    """Optional extensions of optimize_bs_and_ladder (all off by default = original behaviour).

    A product *family* is a product and its tenor/fixing variants: same first three
    digits of product_code, same bs_side (e.g. 1100 and 1103 are one family).
    """
    new_swaps_today_only:    bool = False   # only pay-fixed swaps traded today at par, tenors may be
                                            # fractional (0.5 = 6M); no seasoned/back-dated buckets
    swap_carry_in_objective: bool = False   # new swaps' first-year carry counts in the objective
    family_cap:              bool = False   # a family moves by at most bs_config.max_shift in total
    min_weights:     dict | None = None     # (pc, side) -> lower weight bound (contractual run-off floor)
    concession_threshold: dict | None = None  # (family, side) -> weight up to which growth needs no
                                              # pricing concession (organic allowance)
    concession_rate:      dict | None = None  # (family, side) -> annual cost per unit of balance above
                                              # the threshold (lower loan spread / higher deposit rate
                                              # on the additional business only)
    eve_scale:       dict | None = None     # (pc, side) -> multiplier on the product's EVE sensitivity
                                            # (stress tests, e.g. weaker deposit stability)
    acq_cost_scale:  float = 1.0            # multiplier on every product's acquisition-cost rate
    swap_reset_months: tuple = (3,)         # floating-leg reset frequencies offered for today's swaps
                                            # (3 = 3M EURIBOR, 6 = 6M EURIBOR; single-curve, no basis)
    skippable_swaps: dict | None = None     # bucket_id -> (tenor_years, notional[, fixed_rate]): planned pay-fixed
                                            # swaps the optimiser may leave out, priced as a payer of that tenor
                                            # (at par, or at fixed_rate) with the sign flipped and the notional as
                                            # upper bound
    forced_swaps:    dict | None = None     # bucket_id -> notional fixed in the solution (e.g. AGED_ buckets: the
                                            # existing swaps that mature inside the horizon leave the book)
    line_caps:       dict | None = None     # business line -> (families, side): the line's total weight
                                            # moves by at most bs_config.max_shift (e.g. consumer loans +
                                            # floating consumer + cards together)
    family_max:      dict | None = None     # (family, side) -> upper weight of the family (market capacity:
                                            # how fast clients can be won, independent of the line cap)
    family_range:    dict | None = None     # (family, side) -> (lower, upper) weight of the family (glide path:
                                            # a first step stays between the plan and the long-term target)
    lcr_min:         float | None = None    # consolidated LCR floor, linear: HQLA >= L x net outflows
                                            # with the 75% inflow cap as a second row
    robust:          dict | None = None     # robust_curves.load_robust(): EVE/NII floors on other base
                                            # curves and EP = EP_today + ep_lambda x sum_k w_k (EP_k - EP_today)
    subst_by_family: bool = False           # substitution pairs on family totals (a shift between a product and
                                            # its variants / new-business twin is not a client substitution)
    family_bound_only: set | None = None    # (pc, side) whose own max_shift band is dropped (0..1): an existing
                                            # book and its new-business twin move inside the family cap only
    max_weights:     dict | None = None     # (pc, side) -> upper weight bound (existing book: old vintages
                                            # cannot grow, new volume goes to the product's new-business twin)
    acq_by_family:   bool = False           # acquisition cost on the growth of the FAMILY above its baseline
                                            # (a shift between a product and its variants / new-business twin
                                            # is not new clients), at the family main product's rate
    sale_cost:       dict | None = None     # (pc, side) -> (floor weight, annual cost per unit sold): volume
                                            # below the floor is sold before maturity (market-value result +
                                            # transaction cost, spread over the remaining life)
    group_min:       dict | None = None     # name -> ([(pc, side), ...], lower weight): the products together
                                            # stay above the weight (a product and its new-business twin
                                            # shrink by at most a set amount)


def _family(pc: str) -> str:
    return str(pc)[:3]


SWAP_BID_OFFER_BPS = float(os.environ.get("LABBANK_SWAP_COST_BPS", 12.0))  # 2026-10-30: a payer traded today pays par + 12bp a year,
                           # all-in: bid-offer, operating cost (systems, valuation, monitoring) and the risk of early
                           # termination; without it swaps were free and the swap-vs-balance-sheet choice was a tie


def _today_swap_specs(tenors: tuple, reset_months: tuple = (3,), skippable: dict | None = None) -> list:
    """(bucket_id, tenor, elapsed_m=0, reset_months, fixed rate, freq_m, sign) for swaps traded
    today at par + SWAP_BID_OFFER_BPS; sign -1 = a planned payer left out (SKIP_ buckets)."""
    ct = sl.CurveTensors.load(sl._CURVE_TENSORS_PATH)
    disc = ct.disc_factors[ct.scenario_index("base"), ct.currency_index(sl.CURRENCY)]

    def _par(T, f):
        rm = np.arange(f, T * 12 + 1e-9, f, dtype=float)
        dfs = np.array([sl._df_at_month(disc, m) for m in rm])
        return rm, float((1.0 - dfs[-1]) / (f / 12.0 * dfs.sum())) + SWAP_BID_OFFER_BPS / 1e4

    specs = []
    for f in reset_months:
        for T in tenors:
            if T * 12 <= f:        # one period, fixed at trade: no rate risk, only rounding carry
                continue
            rm, par = _par(T, f)
            specs.append((f"NEW_{T:g}Y" + ("" if f == sl.FIXING_FREQ_M else f"_{f}M"), float(T), 0, rm, par, f, 1.0))
    for bid, (T, _n, *k) in (skippable or {}).items():
        rm, par = _par(T, sl.FIXING_FREQ_M)
        # an existing swap that leaves the book (AGED_...) keeps its own fixed rate; a planned one is at par
        specs.append((bid, float(T), 0, rm, float(k[0]) if k and k[0] is not None else par, sl.FIXING_FREQ_M, -1.0))
    return specs


@dataclass
class JointResult:
    success:                bool
    message:                str
    elapsed_s:              float
    tier1_capital:          float
    ep_bs_old:              float   # PLN -- baseline BS EP (today's weights, existing IRS, no new swap)
    ep_bs_new:              float   # PLN -- BS EP at the optimized weights (excludes new swap carry)
    margin_bs_old:          float   # PLN -- margin-over-FTP income at baseline weights; drives ep_bs_old
    margin_bs_new:          float   # PLN -- margin-over-FTP income at optimized weights
    equity_benefit_bs_old:  float   # PLN -- FTP credit for capital-funded balance at baseline; drives ep_bs_old
    equity_benefit_bs_new:  float   # PLN -- equity benefit at optimized weights
    fee_bs_old:             float   # PLN -- non-interest fee income at baseline weights; drives ep_bs_old
    fee_bs_new:             float   # PLN -- non-interest fee income at optimized weights
    acq_cost_bs_old:        float   # PLN -- marketing/acquisition cost at baseline (always 0.0)
    acq_cost_bs_new:        float   # PLN -- acq cost at optimized weights (only on growth above baseline)
    opex_bs_old:            float   # PLN -- operating cost at baseline weights
    opex_bs_new:            float   # PLN -- operating cost at optimized weights
    ep_carry_swap:          float   # PLN/yr -- new swap ladder's realized carry, REPORTED ONLY:
                                     # not part of the objective (x_swap is chosen purely for its
                                     # EVE/NII constraint-row effect; see module docstring)
    ep_total_new:           float   # PLN -- ep_bs_new + ep_carry_swap (informational total)
    ep_total_improvement_m: float   # PLN -- ep_total_new - ep_bs_old (informational total)
    sot_eve:                dict    # scenario -> delta_EVE/T1*100 (%), BS + new swap combined
    sot_nii:                dict    # scenario -> delta_NII/T1*100 (%), BS + new swap combined
    sot_eve_floor:          float
    sot_nii_floor:          float
    eve_breach_weight:      float
    nii_breach_weight:      float
    breach_slack:           dict    # name -> pp-T1 severity (0 or absent = clean)
    product_changes:        list    # list[ProductChange], BS side
    weights_bs_new:          np.ndarray  # (n_prod,) BS-side optimized product weights,
                                          # for external EP-decomposition/reporting
                                          
    swap_notional:          dict    # bucket_id -> notional PLN (new overlay only)
    swap_direction:         float   # +1 pay-fixed/receive-float, -1 flipped
    ladder:                 dict    # full ladder dict (bucket_ids/elapsed_m/tenor_years/fixed_rate/...)
                                     # -- lets hedge_export.py build deal rows without recomputation
    lcr:                    dict
    nsfr:                   dict
    rwa:                    float
    # whole-bank view (client-rate NII instead of margin over FTP), filled for every run
    nii_base_old:           float = 0.0   # client-rate 12M NII at baseline weights (incl. existing IRS)
    nii_base_new:           float = 0.0   # client-rate 12M NII at optimized weights, before concessions
    el_bs_old:              float = 0.0
    el_bs_new:              float = 0.0
    coc_bs_old:             float = 0.0
    coc_bs_new:             float = 0.0
    concession_cost_old:    float = 0.0   # pricing concessions (JointOptions.concession_*) at baseline
    concession_cost_new:    float = 0.0   # pricing concessions at optimized weights
    sale_cost_new:          float = 0.0   # annualised cost of selling marketable assets below their floor

    def whole_bank_ep(self, new: bool = True) -> float:
        """Client NII + fees - EL - CoC - OpEx - AcqCost - concessions - sale cost (+ new swaps'
        first-year NII): the bank's own result, including treasury's mismatch, unlike the FTP-based EP."""
        if new:
            return (self.nii_base_new + self.ep_carry_swap + self.fee_bs_new - self.el_bs_new
                    - self.coc_bs_new - self.opex_bs_new - self.acq_cost_bs_new - self.concession_cost_new
                    - self.sale_cost_new)
        return (self.nii_base_old + self.fee_bs_old - self.el_bs_old - self.coc_bs_old
                - self.opex_bs_old - self.acq_cost_bs_old - self.concession_cost_old)

    def print_summary(self, top_n_buckets: int = 10) -> None:
        status = "SUCCESS" if self.success else "FAILED"
        print(f"\nJoint BS + IRS optimization - {status}")
        print(f"  T1 capital: {self.tier1_capital/1e6:,.0f}M {home_ccy()}")
        if self.eve_breach_weight > 0 or self.nii_breach_weight > 0:
            print(f"  [Soft floors: EVE weight={self.eve_breach_weight:.1f} {home_ccy()}/pp-T1"
                  f"  NII weight={self.nii_breach_weight:.1f} {home_ccy()}/pp-T1]")
        print(f"  {self.message}   elapsed={self.elapsed_s:.2f}s")
        print(
            f"  BS Margin : {self.margin_bs_old/1e6:>10,.2f}M  ->  {self.margin_bs_new/1e6:>10,.2f}M"
            f"   (margin-over-FTP income)"
        )
        print(
            f"  BS EqBen. : {self.equity_benefit_bs_old/1e6:>10,.2f}M  ->  {self.equity_benefit_bs_new/1e6:>10,.2f}M"
            f"   (FTP credit for capital-funded balance)"
        )
        print(
            f"  BS Fee    : {self.fee_bs_old/1e6:>10,.2f}M  ->  {self.fee_bs_new/1e6:>10,.2f}M"
            f"   (non-interest fee income)"
        )
        print(
            f"  BS OpEx   : {self.opex_bs_old/1e6:>10,.2f}M  ->  {self.opex_bs_new/1e6:>10,.2f}M"
        )
        print(
            f"  BS AcqCost: {self.acq_cost_bs_old/1e6:>10,.2f}M  ->  {self.acq_cost_bs_new/1e6:>10,.2f}M"
            f"   (marketing cost, only on growth above baseline)"
        )
        print(
            f"  BS EP     : {self.ep_bs_old/1e6:>10,.2f}M  ->  {self.ep_bs_new/1e6:>10,.2f}M"
        )
        print(f"  Swap carry (new overlay, reported only -- not optimized for): "
              f"{self.ep_carry_swap/1e6:+,.2f}M/yr")
        sign = "+" if self.ep_total_improvement_m >= 0 else ""
        print(
            f"  TOTAL EP (incl. swap carry): {self.ep_bs_old/1e6:>10,.2f}M  ->  {self.ep_total_new/1e6:>10,.2f}M"
            f"   improvement: {sign}{self.ep_total_improvement_m/1e6:,.2f}M"
        )

        print(f"\n  EVE SOT  (floor {self.sot_eve_floor:.1f}% T1):")
        for s, v in sorted(self.sot_eve.items()):
            ok = "OK" if v >= self.sot_eve_floor else "!!"
            print(f"    {s:<16s}: {v:+7.2f}% T1  [{ok}]")
        print(f"  NII SOT  (floor {self.sot_nii_floor:.1f}% T1):")
        for s, v in sorted(self.sot_nii.items()):
            ok = "OK" if v >= self.sot_nii_floor else "!!"
            print(f"    {s:<16s}: {v:+7.2f}% T1  [{ok}]")

        if self.breach_slack:
            print("\n  Soft-breach severity:")
            for k, v in sorted(self.breach_slack.items()):
                if v > 1e-6:
                    print(f"    {k}: {v:.4f} pp T1")

        for ccy, v in sorted(self.lcr.items()):
            print(f"  LCR  {ccy}: {v:.3f}")
        for ccy, v in sorted(self.nsfr.items()):
            print(f"  NSFR {ccy}: {v:.3f}")
        print(f"  RWA: {self.rwa/1e6:,.0f}M {home_ccy()}")

        print(f"\n  Product weight changes (BS side):")
        top = sorted(self.product_changes, key=lambda c: abs(c.delta_pct), reverse=True)
        for c in top:
            if abs(c.delta_pct) < 0.0001:
                continue
            print(f"    {c.product_code:6s} {c.bs_side}  "
                  f"{c.pct_old:6.2f}% -> {c.pct_new:6.2f}%  ({c.delta_pct:+.2f} pp)")

        direction_lbl = "pay-fixed/receive-float" if self.swap_direction > 0 else "receive-fixed/pay-float"
        nz = [(b, n) for b, n in self.swap_notional.items() if n > 1.0]
        total = sum(n for _, n in nz)
        print(f"\n  New swap overlay ({direction_lbl}): {len(nz)} active bucket(s), "
              f"total {total/1e6:,.1f}M {home_ccy()}")
        for bid, notional in sorted(nz, key=lambda t: -t[1])[:top_n_buckets]:
            print(f"    {bid:16s}  {notional/1e6:,.1f}M")


def _nii_first_fixed(disc, disc_base, reset_months, freq_m) -> float:
    """Floating-leg NII per unit notional over 12 months, the first period at the base-curve fixing."""
    first = sl._price_bucket_nii(disc_base, reset_months[:1], freq_m=freq_m)
    return first + sl._price_bucket_nii(disc, reset_months[1:], freq_m=freq_m) if len(reset_months) > 1 else first


ROBUST_SHOCK_FLOOR = -0.015   # EBA post-shock floor at the short end (-150 bp); a 0% floor would turn a
                              # down shock into a rise on a negative-rate regime curve


def _price_specs_on_ns(specs: list, beta, scens: list[str], direction: float) -> dict:
    """Per unit notional, on an NS base curve: EVE / NII deltas per shock (first floating period fixed
    at the curve's base fixing, as for today's swaps) and the first-year carry."""
    from eba_shock_curves import get_shock_params
    sp = get_shock_params(sl.CURRENCY)
    beta = np.asarray(beta, float)
    base = sl._ns_disc_curve_monthly(beta, "base", sp)
    disc = {s: sl._ns_disc_curve_monthly(beta, s, sp, rate_floor=ROBUST_SHOCK_FLOOR) for s in scens}
    n = len(specs)
    eve = np.zeros((len(scens), n)); nii = np.zeros((len(scens), n)); carry = np.zeros(n)
    for b, spec in enumerate(specs):
        _bid, _T, _e, rm, k, f, sign = spec
        d = direction * sign
        e0 = sl._price_bucket_eve(base, rm, k, freq_m=f)
        n0 = _nii_first_fixed(base, base, rm, f)
        for i, s in enumerate(scens):
            eve[i, b] = d * (sl._price_bucket_eve(disc[s], rm, k, freq_m=f) - e0)
            nii[i, b] = d * (_nii_first_fixed(disc[s], base, rm, f) - n0)
        carry[b] = d * n0 - d * k * (min(12.0, float(rm[-1])) / 12.0)
    return dict(eve=eve, nii=nii, carry=carry)


def _price_ladder_vs_scenarios(shocked_scens: list[str], tenors: tuple, direction: float | None,
                               today_only: bool = False, reset_months: tuple = (3,),
                               skippable: dict | None = None):
    """Price every swap_ladder.py bucket's EVE/NII sensitivity against the SAME
    scenario set bs_optimizer.py uses (base + shocked_scens), via CurveTensors
    -- the exact machinery build_ladder_bank() uses, generalised to whatever
    scenario list is passed in (build_ladder_bank hardcodes its own 6; this
    needs bs_optimizer's 7, including 'own', which IS present in CurveTensors).
    """
    if direction is None:
        direction = sl._detect_hedge_direction()

    ct = sl.CurveTensors.load(sl._CURVE_TENSORS_PATH)
    ccy_idx = ct.currency_index(sl.CURRENCY)

    if today_only:
        specs = _today_swap_specs(tenors, reset_months, skippable)
    else:
        panel = sl.load_historical_panel()
        ns_ts = sl.fit_ns_timeseries(panel, sl.DIEBOLD_LI_TAU)
        specs = sl._build_bucket_specs(tenors, ns_ts, direction=direction)
    n_buckets = len(specs)

    all_scens = ["base"] + list(shocked_scens)
    disc_by_scen = {s: ct.disc_factors[ct.scenario_index(s), ccy_idx] for s in all_scens}

    eve_level = {s: np.empty(n_buckets) for s in all_scens}
    nii_level = {s: np.empty(n_buckets) for s in all_scens}
    ep_carry  = np.empty(n_buckets)

    for bi, spec in enumerate(specs):
        _bid, _T, _e, rm, fixed_rate = spec[:5]
        freq, sign = (spec[5], spec[6]) if len(spec) > 5 else (sl.FIXING_FREQ_M, 1.0)
        d = direction * sign
        for s in all_scens:
            disc = disc_by_scen[s]
            eve_level[s][bi] = d * sl._price_bucket_eve(disc, rm, fixed_rate, freq_m=freq)
            if today_only:      # traded today: the first floating period is fixed at trade, a shock cannot move it
                nii_level[s][bi] = d * _nii_first_fixed(disc, disc_by_scen["base"], rm, freq)
            else:
                nii_level[s][bi] = d * sl._price_bucket_nii(disc, rm, freq_m=freq)
        year_h = min(12.0, float(rm[-1]))
        ep_carry[bi] = nii_level["base"][bi] - d * fixed_rate * (year_h / 12.0)

    delta_eve = {s: eve_level[s] - eve_level["base"] for s in shocked_scens}
    delta_nii = {s: nii_level[s] - nii_level["base"] for s in shocked_scens}

    return dict(
        bucket_ids=[s[0] for s in specs],
        elapsed_m=np.array([s[2] for s in specs], dtype=int),
        tenor_years=np.array([s[1] for s in specs], dtype=float),
        fixed_rate=np.array([s[4] for s in specs], dtype=float),
        delta_eve_pln=delta_eve, delta_nii_pln=delta_nii,
        ep_carry_pln=ep_carry, direction=direction,
    )


def optimize_bs_and_ladder(
    bs_config:               OptimizationConfig,
    params:                  BalanceSheetParams,
    cr:                      CohortRates,
    notional_cap:            float | None = 1e9,
    total_notional_cap:      float | None = 5e9,
    tenors:                  tuple = sl.TENORS_YEARS,
    seasoned_elapsed_months: int = 12,
    seasoned_notional_cap:   float | None = 1e9,
    options:                 JointOptions | None = None,
) -> JointResult:
    """Jointly optimize BS product weights and a new swap-ladder overlay.

    bs_config drives everything on the balance-sheet side exactly like
    bs_optimizer.optimize_nii: mode/fixed_products/max_shift/custom_bounds
    for x_bs, sot_eve_floor/buffer and sot_nii_floor/buffer for the floors,
    and eve_breach_weight/nii_breach_weight for the SAME soft-vs-hard choice
    (0 = hard constraint; >0 = PLN cost per average pp-T1 of breach severity,
    applied to the COMBINED BS+swap breach, not per side).

    seasoned_elapsed_months / seasoned_notional_cap: see hedge_optimizer.
    solve_optimal_ladder's docstring -- same mechanism, same reason (buckets
    originated seasoned_elapsed_months+ months ago carry a historical-rate-
    drift windfall separate from IRS_MARGIN_BPS, so they get their own
    smaller sub-cap on top of notional_cap/total_notional_cap).
    """
    t_start = time.time()
    ta = params.total_assets
    t1 = (bs_config.tier1_capital if bs_config.tier1_capital > 0
          else float(params.balance_arr[params.is_equity].sum()))
    pm = _ProductMap(params)
    opt = options or JointOptions()
    fam_keys = sorted({(_family(pc), side) for pc, side in pm.products})
    fam_of = np.array([fam_keys.index((_family(pc), side)) for pc, side in pm.products])

    irs_cohort_mask, irs_products, irs_notional = _resolve_irs(params)
    active_fixed = bs_config.fixed_products | irs_products
    if irs_products:
        print(f"  Existing IR derivatives (pinned): {irs_notional/1e6:,.0f}M notional "
              f"-- new swap overlay is layered ON TOP of this, not a replacement")

    bias_eve, bias_nii, bias_scens = load_bias_corrections()

    # FTP + opex (see bs_optimizer.py / ftp_store.py) -- same substitution as
    # the BS-only optimizer, so Section 9's "BS-only vs Joint" comparison
    # stays an apples-to-apples EP definition.
    ftp_rate = load_ftp_rates(params.cohort_id)
    if ftp_rate is None:
        print("  No FTP cache found -- run build_ftp_rates.py; EP margin will equal client-rate NII until then")
        ftp_rate = np.zeros_like(params.nii_unit_rate)
    margin_rate    = margin_unit_rate(params.nii_unit_rate,         ftp_rate, params.bs_side)
    margin_rate_nb = margin_unit_rate(params.nii_unit_rate_new_biz, ftp_rate, params.bs_side)
    # Equity Benefit: FTP credit for the capital-funded share of each cohort's
    # balance (see ftp_store.equity_benefit_unit_rate) -- depends only on
    # ftp_rate/rwa_factor/cet1_target, so one array serves both the existing-
    # and new-business EP objectives (unlike margin_rate/margin_rate_nb above).
    equity_benefit_rate = equity_benefit_unit_rate(ftp_rate, params.rwa_factor, params.cet1_target)
    fee_rate  = params.fee_unit_rate
    opex_mask = _opex_rows(params)

    _base_amounts = (params.balance_arr if bs_config.include_irs
                      else np.where(irs_cohort_mask, 0.0, params.balance_arr))
    _base_nii = compute_nii_cf_all(_base_amounts, params, cr)
    nii_base_old = _base_nii["base"]              # client-rate NII, informational only
    margin_base_old = float(np.dot(_base_amounts, margin_rate))
    equity_benefit_base_old = float(np.dot(_base_amounts, equity_benefit_rate))
    fee_base_old = float(np.dot(_base_amounts, fee_rate))
    _base_rwa   = float(np.dot(_base_amounts, params.rwa_factor))
    el_base_old = float(np.dot(_base_amounts, params.el_unit))
    coc_base_old = _base_rwa * params.cet1_target * params.coc_rate
    opex_base_old = OPEX_RATE * float(np.sum(_base_amounts[opex_mask]))
    acq_cost_base_old = 0.0   # growth above baseline is 0 AT baseline, by definition
    ep_bs_old = (margin_base_old + equity_benefit_base_old + fee_base_old
                 - el_base_old - coc_base_old - opex_base_old - acq_cost_base_old)

    shocked_scens = [str(s) for s in cr.rate_scenario_ids if str(s) != "base"]
    n_scen = len(shocked_scens)
    eve_floor_eff = bs_config.sot_eve_floor - bs_config.sot_eve_buffer
    nii_floor_eff = bs_config.sot_nii_floor - bs_config.sot_nii_buffer

    soft_eve = bs_config.eve_breach_weight > 0.0
    soft_nii = bs_config.nii_breach_weight > 0.0
    w_eve = bs_config.eve_breach_weight * (t1 / 100.0) / n_scen if soft_eve else 0.0
    w_nii = bs_config.nii_breach_weight * (t1 / 100.0) / n_scen if soft_nii else 0.0

    # ── BS-side LP coefficients (reuses bs_optimizer's exact machinery) ───────
    print("  Building BS-side LP coefficients...")
    c_nii, A_eve_bs, A_nii_bs, A_nsfr, c_rwa, c_el, c_ep, c_ep_lp, c_fee = _compute_lp_coefficients(
        pm, params, cr, t1, irs_cohort_mask, bs_config.include_irs,
        bias_eve, bias_nii, bias_scens, shocked_scens,
        margin_rate, margin_rate_nb, fee_rate, equity_benefit_rate, min_nsfr=bs_config.min_nsfr,
    )
    lb, ub = _build_bounds(bs_config, pm, active_fixed)
    if opt.family_bound_only:
        for j, key in enumerate(pm.products):
            if key in opt.family_bound_only and key not in active_fixed:
                lb[j], ub[j] = 0.0, 1.0
    if opt.min_weights:
        for j, key in enumerate(pm.products):
            if key in opt.min_weights and lb[j] < ub[j]:
                lb[j] = min(max(lb[j], float(opt.min_weights[key])), ub[j])
    if opt.max_weights:
        for j, key in enumerate(pm.products):
            if key in opt.max_weights:
                ub[j] = min(ub[j], float(opt.max_weights[key]))
                lb[j] = min(lb[j], ub[j])
    # the seasoned book of a product that moves through its plan-year cohorts cannot be undone
    lb = np.minimum(np.maximum(lb, pm.min_w), np.maximum(ub, pm.min_w))
    ub = np.maximum(ub, lb)
    bounds_bs = list(zip(lb.tolist(), ub.tolist()))
    A_subst, b_subst, _ = _build_subst_constraints(pm, params, by_family=opt.subst_by_family,
                                                    src_cap=ub, fam_cap=opt.family_max)

    A_eq_bs = np.zeros((2, pm.n_prod))
    A_eq_bs[0, pm.asset_mask] = 1.0
    A_eq_bs[1, pm.fund_mask]  = 1.0
    b_eq = np.array([pm.asset_sum, pm.fund_sum])

    # ── Swap-side LP coefficients ──────────────────────────────────────────
    print(f"  Pricing swap ladder ({len(tenors)} tenors) against bs_optimizer's "
          f"{n_scen} EBA scenarios...")
    ladder = _price_ladder_vs_scenarios(shocked_scens, tenors, direction=None,
                                        today_only=opt.new_swaps_today_only,
                                        reset_months=opt.swap_reset_months,
                                        skippable=opt.skippable_swaps)
    bucket_ids = ladder["bucket_ids"]
    n_buckets  = len(bucket_ids)
    direction  = ladder["direction"]

    eve_scale_prod = np.ones(pm.n_prod)
    if opt.eve_scale:
        for j, key in enumerate(pm.products):
            eve_scale_prod[j] = float(opt.eve_scale.get(key, 1.0))
        A_eve_raw = A_eve_bs.copy()
        A_eve_bs = A_eve_bs * eve_scale_prod[None, :]
    A_eve_swap = np.vstack([ladder["delta_eve_pln"][s] / t1 * 100.0 * NOTIONAL_UNIT
                             for s in shocked_scens])   # (n_scen, n_buckets)
    A_nii_swap = np.vstack([ladder["delta_nii_pln"][s] / t1 * 100.0 * NOTIONAL_UNIT
                             for s in shocked_scens])
    # ep_carry_pln is NOT in the objective (see module docstring) -- x_swap's
    # objective coefficient is 0 below; it only affects the solution via the
    # A_eve_swap/A_nii_swap constraint rows. Kept here only for the post-hoc
    # ep_carry_swap reporting field.

    cap_u = None if notional_cap is None else notional_cap / NOTIONAL_UNIT
    bounds_swap = [(0.0, cap_u)] * n_buckets
    for bi, bid in enumerate(bucket_ids):
        if opt.skippable_swaps and bid in opt.skippable_swaps:
            bounds_swap[bi] = (0.0, opt.skippable_swaps[bid][1] / NOTIONAL_UNIT)
        if opt.forced_swaps and bid in opt.forced_swaps:
            bounds_swap[bi] = (opt.forced_swaps[bid] / NOTIONAL_UNIT,) * 2
    seasoned_mask = ladder["elapsed_m"] > seasoned_elapsed_months

    n_x = pm.n_prod + n_buckets   # total non-slack decision variables

    # ── Joint EVE/NII rows: [A_eve_bs | A_eve_swap] @ [x_bs; x_swap] >= floor ─
    hard_rows, hard_rhs = [], []
    soft_rows, soft_rhs, soft_w, soft_names = [], [], [], []

    def _joint_row(A_bs_row, A_swap_row):
        return np.concatenate([A_bs_row, A_swap_row])

    if soft_eve:
        for k in range(n_scen):
            soft_rows.append(-_joint_row(A_eve_bs[k], A_eve_swap[k]))
            soft_rhs.append(-eve_floor_eff)
            soft_w.append(w_eve)
        soft_names += [f"eve_{s}" for s in shocked_scens]
    else:
        for k in range(n_scen):
            hard_rows.append(-_joint_row(A_eve_bs[k], A_eve_swap[k]))
            hard_rhs.append(-eve_floor_eff)

    if soft_nii:
        for k in range(n_scen):
            soft_rows.append(-_joint_row(A_nii_bs[k], A_nii_swap[k]))
            soft_rhs.append(-nii_floor_eff)
            soft_w.append(w_nii)
        soft_names += [f"nii_{s}" for s in shocked_scens]
    else:
        for k in range(n_scen):
            hard_rows.append(-_joint_row(A_nii_bs[k], A_nii_swap[k]))
            hard_rhs.append(-nii_floor_eff)

    # NSFR / RWA / substitution: BS-only, swap columns zero-padded, always hard
    hard_rows.append(_joint_row(A_nsfr, np.zeros(n_buckets)))
    hard_rhs.append(0.0)
    if bs_config.min_t1_rwa > 0.0:
        hard_rows.append(_joint_row(c_rwa, np.zeros(n_buckets)))
        hard_rhs.append(t1 / bs_config.min_t1_rwa)
    if A_subst is not None:
        for row, rhs in zip(A_subst, b_subst):
            hard_rows.append(_joint_row(row, np.zeros(n_buckets)))
            hard_rhs.append(rhs)
    if total_notional_cap is not None:
        free_b = np.array([0.0 if (opt.forced_swaps and b in opt.forced_swaps) else 1.0 for b in bucket_ids])
        hard_rows.append(_joint_row(np.zeros(pm.n_prod), free_b))
        hard_rhs.append(total_notional_cap / NOTIONAL_UNIT)
    if seasoned_notional_cap is not None and seasoned_mask.any():
        hard_rows.append(_joint_row(np.zeros(pm.n_prod), seasoned_mask.astype(float)))
        hard_rhs.append(seasoned_notional_cap / NOTIONAL_UNIT)

    if opt.family_cap:
        for f in range(len(fam_keys)):
            idx = np.where(fam_of == f)[0]
            if len(idx) < 2:
                continue
            row = np.zeros(pm.n_prod); row[idx] = 1.0
            w0 = float(pm.base_prod_w[idx].sum())
            hard_rows.append(_joint_row(row, np.zeros(n_buckets))); hard_rhs.append(w0 + bs_config.max_shift)
            hard_rows.append(_joint_row(-row, np.zeros(n_buckets))); hard_rhs.append(-max(0.0, w0 - bs_config.max_shift))

    if opt.line_caps:
        for line, (fams, side) in opt.line_caps.items():
            idx = np.array([j for j, (pc, sd) in enumerate(pm.products) if sd == side and _family(pc) in fams])
            if len(idx) == 0:
                continue
            row = np.zeros(pm.n_prod); row[idx] = 1.0
            w0 = float(pm.base_prod_w[idx].sum())
            hard_rows.append(_joint_row(row, np.zeros(n_buckets))); hard_rhs.append(w0 + bs_config.max_shift)
            hard_rows.append(_joint_row(-row, np.zeros(n_buckets))); hard_rhs.append(-max(0.0, w0 - bs_config.max_shift))

    if opt.family_max:
        for f, key in enumerate(fam_keys):
            if key in opt.family_max:
                row = (fam_of == f).astype(float)
                hard_rows.append(_joint_row(row, np.zeros(n_buckets))); hard_rhs.append(float(opt.family_max[key]))

    if opt.family_range:
        for f, key in enumerate(fam_keys):
            if key in opt.family_range:
                lo, hi = opt.family_range[key]
                row = (fam_of == f).astype(float)
                hard_rows.append(_joint_row(row, np.zeros(n_buckets))); hard_rhs.append(float(hi))
                hard_rows.append(_joint_row(-row, np.zeros(n_buckets))); hard_rhs.append(-float(lo))

    if opt.group_min:
        prod_j = {k: j for j, k in enumerate(pm.products)}
        for keys, lo in opt.group_min.values():
            idx = [prod_j[k] for k in keys if k in prod_j]
            if not idx or lo <= 0.0:
                continue
            row = np.zeros(pm.n_prod); row[idx] = 1.0
            hard_rows.append(_joint_row(-row, np.zeros(n_buckets))); hard_rhs.append(-float(lo))

    def _per_prod(v):          # cohort vector -> coefficient per product weight (_ProductMap.per_prod)
        return pm.per_prod(v) * ta

    if opt.lcr_min:
        # consolidated over every currency (amounts already in home currency), as lcr_fast reports it
        out_c = params.lcr_runoff
        if params.commit_lcr_outflow:
            out_c = out_c + params.commit_lcr_outflow / params.total_assets * (
                params.is_asset & ~np.isin(params.product_code.astype(str), ["0000", "0100"]))
        hq, out, inf = _per_prod(params.hqla_factor), _per_prod(out_c), _per_prod(params.inflow_30d_frac)
        L = float(opt.lcr_min)
        for row in (L * (out - inf) - hq, L * 0.25 * out - hq):      # HQLA >= L x max(out - in, 25% out)
            hard_rows.append(_joint_row(row, np.zeros(n_buckets))); hard_rhs.append(0.0)

    ep_bs_extra = np.zeros(pm.n_prod)
    ep_sw_extra = np.zeros(n_buckets)
    robust_rows = []          # (curve, metric, scenario, bs_row, swap_row, floor) for reporting
    robust_ep = []            # (curve, EUR per product weight, EUR per swap unit): EP_k - EP_today
    if opt.robust:
        rb = opt.robust
        specs = _today_swap_specs(tenors, opt.swap_reset_months, opt.skippable_swaps)
        assert [s[0] for s in specs] == list(bucket_ids), "robust swap specs out of step with the ladder"
        fit = _price_specs_on_ns(specs, rb["beta_fit"], shocked_scens, direction)
        hyp_idx = [rb["shock_ids"].index(s) for s in shocked_scens]
        for k, name in enumerate(rb["names"]):
            cur = _price_specs_on_ns(specs, rb["betas"][k], shocked_scens, direction)
            for i, s in enumerate(shocked_scens):
                d_e = rb["d_eve"][k][:, hyp_idx[i]] * ta / t1 * 100.0 * eve_scale_prod
                d_n = rb["d_nii"][k][:, hyp_idx[i]] * ta / t1 * 100.0
                sw_e = A_eve_swap[i] + (cur["eve"][i] - fit["eve"][i]) / t1 * 100.0 * NOTIONAL_UNIT
                sw_n = A_nii_swap[i] + (cur["nii"][i] - fit["nii"][i]) / t1 * 100.0 * NOTIONAL_UNIT
                for metric, bs_row, sw_row, floor in (("eve", A_eve_bs[i] + d_e, sw_e, rb["eve_floor"]),
                                                      ("nii", A_nii_bs[i] + d_n, sw_n, rb["nii_floor"])):
                    hard_rows.append(-_joint_row(bs_row, sw_row)); hard_rhs.append(-floor)
                    robust_rows.append((name, metric, s, bs_row, sw_row, floor))
            wk = rb["ep_lambda"] * float(rb["weights"][k])
            ep_bs_extra += wk * rb["d_level"][k] * ta
            ep_sw_extra += wk * (cur["carry"] - fit["carry"]) * NOTIONAL_UNIT
            robust_ep.append((name, rb["d_level"][k] * ta, (cur["carry"] - fit["carry"]) * NOTIONAL_UNIT))
    A_hard = np.vstack(hard_rows)
    b_hard = np.array(hard_rhs, dtype=float)
    n_hard = A_hard.shape[0]

    n_slack = len(soft_names)
    if n_slack > 0:
        A_soft = np.vstack(soft_rows)
        b_soft = np.array(soft_rhs, dtype=float)
        w_soft = np.array(soft_w, dtype=float)
        c_lp = np.concatenate([-c_ep_lp - ep_bs_extra, (-ladder["ep_carry_pln"] * NOTIONAL_UNIT - ep_sw_extra if opt.swap_carry_in_objective else np.zeros(n_buckets)), w_soft])
        A_hard_aug = np.hstack([A_hard, np.zeros((n_hard, n_slack))])
        A_soft_aug = np.hstack([A_soft, -np.eye(n_slack)])
        A_ub = np.vstack([A_hard_aug, A_soft_aug])
        b_ub = np.concatenate([b_hard, b_soft])
        bounds_all = bounds_bs + bounds_swap + [(0.0, None)] * n_slack
    else:
        c_lp = np.concatenate([-c_ep_lp - ep_bs_extra, (-ladder["ep_carry_pln"] * NOTIONAL_UNIT - ep_sw_extra if opt.swap_carry_in_objective else np.zeros(n_buckets))])
        A_ub = A_hard
        b_ub = b_hard
        bounds_all = bounds_bs + bounds_swap

    A_eq = np.hstack([A_eq_bs, np.zeros((2, n_buckets + n_slack))])

    # ── Growth-cost epigraph augmentation (2026-07-14) ───────────────────────
    # Same mechanism as bs_optimizer.py's optimize_nii() -- see that module for
    # the full explanation. g_j = max(0, x_bs_j - base_prod_w_j) via the
    # standard LP epigraph trick, inserted as its own variable block BETWEEN
    # x_bs (n_prod) and x_swap (n_buckets), so the swap/slack code above didn't
    # need to change. Always added (even if acq_cost_prod is all zero) so the
    # variable layout is constant.
    n_swap_slack_cols = A_ub.shape[1] - pm.n_prod   # n_buckets + n_slack
    A_ub = np.hstack([
        A_ub[:, :pm.n_prod], np.zeros((A_ub.shape[0], pm.n_prod)), A_ub[:, pm.n_prod:]
    ])
    A_eq = np.hstack([
        A_eq[:, :pm.n_prod], np.zeros((A_eq.shape[0], pm.n_prod)), A_eq[:, pm.n_prod:]
    ])
    acq_rate = pm.acq_cost_prod * opt.acq_cost_scale
    if opt.acq_by_family:
        # one growth variable per family, on the family's main product (largest baseline): g_f >= sum_f x - base_f
        fam_main = {}
        for j in range(pm.n_prod):
            f = fam_of[j]
            if f not in fam_main or pm.base_prod_w[j] > pm.base_prod_w[fam_main[f]]:
                fam_main[f] = j
        g_rows, g_rhs = [], []
        for f, jm in fam_main.items():
            row = np.zeros(2 * pm.n_prod + n_swap_slack_cols)
            row[:pm.n_prod] = (fam_of == f).astype(float)
            row[pm.n_prod + jm] = -1.0
            g_rows.append(row); g_rhs.append(float(pm.base_prod_w[fam_of == f].sum()))
        A_growth = np.vstack(g_rows)
        g_bounds = [(0.0, None) if j in fam_main.values() else (0.0, 0.0) for j in range(pm.n_prod)]
        acq_rate = np.array([acq_rate[fam_of == fam_of[j]].max() if j in fam_main.values() else 0.0
                             for j in range(pm.n_prod)])
        b_growth = np.array(g_rhs)
    else:
        A_growth = np.hstack([
            np.eye(pm.n_prod), -np.eye(pm.n_prod), np.zeros((pm.n_prod, n_swap_slack_cols))
        ])
        g_bounds = [(0.0, None)] * pm.n_prod
        b_growth = pm.base_prod_w
    A_ub = np.vstack([A_ub, A_growth])
    b_ub = np.concatenate([b_ub, b_growth])
    c_lp = np.concatenate([c_lp[:pm.n_prod], acq_rate * ta, c_lp[pm.n_prod:]])
    bounds_all = bounds_all[:pm.n_prod] + g_bounds + bounds_all[pm.n_prod:]

    def _acq_cost(x):
        if opt.acq_by_family:
            fam_w = np.bincount(fam_of, weights=x, minlength=len(fam_keys))
            fam_b = np.bincount(fam_of, weights=pm.base_prod_w, minlength=len(fam_keys))
            fam_r = np.zeros(len(fam_keys))
            np.maximum.at(fam_r, fam_of, acq_rate)
            return float(np.dot(fam_r, np.maximum(0.0, fam_w - fam_b))) * ta
        return float(np.dot(acq_rate, np.maximum(0.0, x - pm.base_prod_w))) * ta

    # ── Sale cost epigraph (JointOptions.sale_cost): s_j >= floor_j - x_j, s_j >= 0, cost rate_j * TA * s_j.
    sale_j, sale_floor, sale_rate = [], [], []
    for j, key in enumerate(pm.products):
        if opt.sale_cost and key in opt.sale_cost:
            fl, rt = opt.sale_cost[key]
            if fl > 0.0:
                sale_j.append(j); sale_floor.append(float(fl)); sale_rate.append(float(rt))
    if sale_j:
        n_s = len(sale_j)
        n_cols = A_ub.shape[1]
        rows = np.zeros((n_s, n_cols + n_s))
        for r, j in enumerate(sale_j):
            rows[r, j] = -1.0
            rows[r, n_cols + r] = -1.0
        A_ub = np.vstack([np.hstack([A_ub, np.zeros((A_ub.shape[0], n_s))]), rows])
        b_ub = np.concatenate([b_ub, -np.array(sale_floor)])
        A_eq = np.hstack([A_eq, np.zeros((A_eq.shape[0], n_s))])
        c_lp = np.concatenate([c_lp, np.array(sale_rate) * ta])
        bounds_all = bounds_all + [(0.0, None)] * n_s

    def _sale_cost(x):
        return float(sum(r * max(0.0, f - x[j]) for j, f, r in zip(sale_j, sale_floor, sale_rate))) * ta

    # ── Pricing-concession epigraph (JointOptions.concession_*): h_f >= sum_{j in f} x_j - thr_f,
    #    h_f >= 0, cost rate_f * TA * h_f. Appended as the LAST variable block. ──
    conc_thr = np.full(len(fam_keys), np.inf)
    conc_rate = np.zeros(len(fam_keys))
    if opt.concession_threshold and opt.concession_rate:
        for f, key in enumerate(fam_keys):
            if key in opt.concession_threshold and opt.concession_rate.get(key, 0.0) > 0.0:
                conc_thr[f] = float(opt.concession_threshold[key])
                conc_rate[f] = float(opt.concession_rate[key])
    conc_idx = np.where(np.isfinite(conc_thr))[0]
    n_conc = len(conc_idx)
    if n_conc:
        n_cols = A_ub.shape[1]
        rows = np.zeros((n_conc, n_cols + n_conc))
        for r, f in enumerate(conc_idx):
            rows[r, :pm.n_prod] = (fam_of == f).astype(float)
            rows[r, n_cols + r] = -1.0
        A_ub = np.vstack([np.hstack([A_ub, np.zeros((A_ub.shape[0], n_conc))]), rows])
        b_ub = np.concatenate([b_ub, conc_thr[conc_idx]])
        A_eq = np.hstack([A_eq, np.zeros((A_eq.shape[0], n_conc))])
        c_lp = np.concatenate([c_lp, conc_rate[conc_idx] * ta])
        bounds_all = bounds_all + [(0.0, None)] * n_conc

    def _concession_cost(x):
        fam_w = np.bincount(fam_of, weights=x, minlength=len(fam_keys))
        return float(np.sum(conc_rate[conc_idx] * np.maximum(0.0, fam_w[conc_idx] - conc_thr[conc_idx]))) * ta

    print(f"  Solving joint LP: {pm.n_prod} BS weights + {n_buckets} swap buckets"
          f" + {n_slack} soft-breach slack(s)...")
    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds_all, method="highs")
    elapsed = time.time() - t_start

    if res.success:
        x_bs   = res.x[:pm.n_prod]
        # res.x layout: [x_bs (n_prod), g (n_prod, growth-cost epigraph), x_swap (n_buckets), s (n_slack)]
        x_swap = res.x[2 * pm.n_prod:2 * pm.n_prod + n_buckets] * NOTIONAL_UNIT
        s_sol  = (res.x[2 * pm.n_prod + n_buckets:2 * pm.n_prod + n_buckets + n_slack]
                  if n_slack > 0 else np.array([]))
    else:
        print(f"  LP FAILED: {res.message} -- returning baseline (zero new swap)")
        x_bs   = pm.base_prod_w.copy()
        x_swap = np.zeros(n_buckets)
        s_sol  = np.zeros(n_slack)

    # ── Final reporting: exact BS-side reprice (with bias correction) + the
    #    swap ladder's linear delta on top (same combination hedge_optimizer's
    #    3-way comparison uses: exact book + analytic-DCF ladder overlay). ──
    amounts_bs = pm.to_amounts(x_bs, ta)
    amounts_m  = amounts_bs if bs_config.include_irs else np.where(irs_cohort_mask, 0.0, amounts_bs)
    nii_bs  = compute_nii_cf_all(amounts_m, params, cr)
    eve_bs  = compute_eve_cf_fast_all(amounts_m, params, cr)
    lcr_bs  = compute_lcr_fast(amounts_m, params)
    nsfr_bs = compute_nsfr_fast(amounts_m, params)
    rwa_bs  = compute_rwa_fast(amounts_m, params)
    if bias_eve is not None:
        apply_eve_bias(eve_bs, amounts_m, bias_eve, bias_scens)
        apply_nii_bias(nii_bs, amounts_m, bias_nii, bias_scens)

    nii_base_new = nii_bs["base"]              # client-rate NII, informational only
    margin_new   = float(np.dot(amounts_bs, margin_rate))
    equity_benefit_new = float(np.dot(amounts_bs, equity_benefit_rate))
    fee_new = float(np.dot(amounts_bs, fee_rate))
    el_new  = float(np.dot(amounts_bs, params.el_unit))
    coc_new = rwa_bs * params.cet1_target * params.coc_rate
    opex_new = OPEX_RATE * float(np.sum(amounts_bs[opex_mask]))
    acq_cost_new = _acq_cost(x_bs)
    conc_new = _concession_cost(x_bs)
    conc_old = _concession_cost(pm.base_prod_w)
    sale_new = _sale_cost(x_bs)
    ep_bs_old -= conc_old
    ep_bs_new = (margin_new + equity_benefit_new + fee_new
                 - el_new - coc_new - opex_new - acq_cost_new - conc_new - sale_new)
    ep_carry_swap = float(np.dot(ladder["ep_carry_pln"], x_swap))
    ep_total_new = ep_bs_new + ep_carry_swap

    sot_eve_out: dict[str, float] = {}
    sot_nii_out: dict[str, float] = {}
    for k, s in enumerate(shocked_scens):
        eve_bs_pct = eve_bs.get(s, 0.0) / t1 * 100.0
        nii_delta_bs_pct = (nii_bs.get(s, nii_base_new) - nii_base_new) / t1 * 100.0
        swap_eve_pct = float(np.dot(ladder["delta_eve_pln"][s], x_swap)) / t1 * 100.0
        swap_nii_pct = float(np.dot(ladder["delta_nii_pln"][s], x_swap)) / t1 * 100.0
        if opt.eve_scale:   # stress: same product-level scaling as the LP rows
            eve_bs_pct += float(np.dot((eve_scale_prod - 1.0) * A_eve_raw[k], x_bs))
        sot_eve_out[s] = eve_bs_pct + swap_eve_pct
        sot_nii_out[s] = nii_delta_bs_pct + swap_nii_pct

    breach_slack: dict[str, float] = {}
    if n_slack > 0:
        for name, val in zip(soft_names, s_sol):
            breach_slack[name] = float(val)

    changes = [
        ProductChange(
            product_code=pc, bs_side=side,
            weight_old=float(pm.base_prod_w[j]), weight_new=float(x_bs[j]),
            delta_weight=float(x_bs[j] - pm.base_prod_w[j]),
            pct_old=float(pm.base_prod_w[j]) * 100.0, pct_new=float(x_bs[j]) * 100.0,
            delta_pct=float(x_bs[j] - pm.base_prod_w[j]) * 100.0,
        )
        for j, (pc, side) in enumerate(pm.products)
    ]
    swap_notional = {bid: float(n) for bid, n in zip(bucket_ids, x_swap)}

    robust_out = {}
    xs_u = x_swap / NOTIONAL_UNIT
    for name, metric, s, bs_row, sw_row, floor in robust_rows:
        robust_out.setdefault(name, {"eve": {}, "nii": {}})[metric][s] = float(bs_row @ x_bs + sw_row @ xs_u)
    for name, lv, cv in robust_ep:
        robust_out[name]["ep_delta"] = float(lv @ x_bs + cv @ xs_u)
    result = JointResult(
        success=res.success, message=res.message, elapsed_s=elapsed,
        tier1_capital=t1,
        ep_bs_old=ep_bs_old, ep_bs_new=ep_bs_new,
        margin_bs_old=margin_base_old, margin_bs_new=margin_new,
        equity_benefit_bs_old=equity_benefit_base_old, equity_benefit_bs_new=equity_benefit_new,
        fee_bs_old=fee_base_old, fee_bs_new=fee_new,
        acq_cost_bs_old=acq_cost_base_old, acq_cost_bs_new=acq_cost_new,
        opex_bs_old=opex_base_old, opex_bs_new=opex_new,
        ep_carry_swap=ep_carry_swap, ep_total_new=ep_total_new,
        ep_total_improvement_m=ep_total_new - ep_bs_old,
        sot_eve=sot_eve_out, sot_nii=sot_nii_out,
        sot_eve_floor=eve_floor_eff, sot_nii_floor=nii_floor_eff,
        eve_breach_weight=bs_config.eve_breach_weight,
        nii_breach_weight=bs_config.nii_breach_weight,
        breach_slack=breach_slack,
        product_changes=changes,
        weights_bs_new=x_bs,
        swap_notional=swap_notional, swap_direction=direction, ladder=ladder,
        lcr=dict(lcr_bs), nsfr=dict(nsfr_bs), rwa=rwa_bs,
        nii_base_old=float(nii_base_old), nii_base_new=float(nii_base_new),
        el_bs_old=el_base_old, el_bs_new=el_new, coc_bs_old=coc_base_old, coc_bs_new=coc_new,
        concession_cost_old=conc_old, concession_cost_new=conc_new, sale_cost_new=sale_new,
    )
    result.robust = robust_out        # LP view per curve: worst EVE / NII per shock, % T1; EP_k - EP_today
    return result


if __name__ == "__main__":
    _PARAMS = os.path.join(OPTPREP_OUT, "product_params.npz")
    params = BalanceSheetParams.load(_PARAMS)
    cr     = CohortRates.load(_PARAMS)

    cfg = OptimizationConfig(
        mode="max_shift", max_shift=0.03,
        eve_breach_weight=20.0, nii_breach_weight=20.0,
    )
    result = optimize_bs_and_ladder(cfg, params, cr)
    result.print_summary()
