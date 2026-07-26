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

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

_HERE    = os.path.dirname(os.path.abspath(__file__))
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
from ftp_store  import load_ftp_rates, margin_unit_rate

import swap_ladder as sl                                                   # noqa: E402
from bs_optimizer import (                                                 # noqa: E402
    OptimizationConfig, ProductChange, OPEX_RATE,
    _ProductMap, _resolve_irs, _build_bounds,
    _compute_lp_coefficients, _build_subst_constraints,
)

NOTIONAL_UNIT = 1e6
# Same conditioning fix as hedge_optimizer.py: swap sensitivities are
# ~1e-10 PLN/T1 per PLN notional while notional bounds are ~1e9 PLN -- solve
# in units of 1e6 PLN so every LP coefficient sits in a comparable range.


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
                                          # for external EP-decomposition/reporting (mirrors
                                          # StochasticResult.weights_new)
    swap_notional:          dict    # bucket_id -> notional PLN (new overlay only)
    swap_direction:         float   # +1 pay-fixed/receive-float, -1 flipped
    ladder:                 dict    # full ladder dict (bucket_ids/elapsed_m/tenor_years/fixed_rate/...)
                                     # -- lets hedge_export.py build deal rows without recomputation
    lcr:                    dict
    nsfr:                   dict
    rwa:                    float

    def print_summary(self, top_n_buckets: int = 10) -> None:
        status = "SUCCESS" if self.success else "FAILED"
        print(f"\nJoint BS + IRS optimization - {status}")
        print(f"  T1 capital: {self.tier1_capital/1e6:,.0f}M PLN")
        if self.eve_breach_weight > 0 or self.nii_breach_weight > 0:
            print(f"  [Soft floors: EVE weight={self.eve_breach_weight:.1f} PLN/pp-T1"
                  f"  NII weight={self.nii_breach_weight:.1f} PLN/pp-T1]")
        print(f"  {self.message}   elapsed={self.elapsed_s:.2f}s")
        print(
            f"  BS Margin : {self.margin_bs_old/1e6:>10,.2f}M  ->  {self.margin_bs_new/1e6:>10,.2f}M"
            f"   (margin-over-FTP income)"
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
        print(f"  RWA: {self.rwa/1e6:,.0f}M PLN")

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
              f"total {total/1e6:,.1f}M PLN")
        for bid, notional in sorted(nz, key=lambda t: -t[1])[:top_n_buckets]:
            print(f"    {bid:16s}  {notional/1e6:,.1f}M")


def _price_ladder_vs_scenarios(shocked_scens: list[str], tenors: tuple, direction: float | None):
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

    panel = sl.load_historical_panel()
    ns_ts = sl.fit_ns_timeseries(panel, sl.DIEBOLD_LI_TAU)
    specs = sl._build_bucket_specs(tenors, ns_ts, direction=direction)
    n_buckets = len(specs)

    all_scens = ["base"] + list(shocked_scens)
    disc_by_scen = {s: ct.disc_factors[ct.scenario_index(s), ccy_idx] for s in all_scens}

    eve_level = {s: np.empty(n_buckets) for s in all_scens}
    nii_level = {s: np.empty(n_buckets) for s in all_scens}
    ep_carry  = np.empty(n_buckets)

    for bi, (_bid, _T, _e, reset_months, fixed_rate) in enumerate(specs):
        for s in all_scens:
            disc = disc_by_scen[s]
            eve_level[s][bi] = direction * sl._price_bucket_eve(disc, reset_months, fixed_rate)
            nii_level[s][bi] = direction * sl._price_bucket_nii(disc, reset_months)
        year_h = min(12.0, float(reset_months[-1]))
        ep_carry[bi] = nii_level["base"][bi] - direction * fixed_rate * (year_h / 12.0)

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
    fee_rate  = params.fee_unit_rate
    opex_mask = ~params.is_equity

    _base_amounts = (params.balance_arr if bs_config.include_irs
                      else np.where(irs_cohort_mask, 0.0, params.balance_arr))
    _base_nii = compute_nii_cf_all(_base_amounts, params, cr)
    nii_base_old = _base_nii["base"]              # client-rate NII, informational only
    margin_base_old = float(np.dot(_base_amounts, margin_rate))
    fee_base_old = float(np.dot(_base_amounts, fee_rate))
    _base_rwa   = float(np.dot(_base_amounts, params.rwa_factor))
    el_base_old = float(np.dot(_base_amounts, params.el_unit))
    coc_base_old = _base_rwa * params.cet1_target * params.coc_rate
    opex_base_old = OPEX_RATE * float(np.sum(_base_amounts[opex_mask]))
    acq_cost_base_old = 0.0   # growth above baseline is 0 AT baseline, by definition
    ep_bs_old = margin_base_old + fee_base_old - el_base_old - coc_base_old - opex_base_old - acq_cost_base_old

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
        margin_rate, margin_rate_nb, fee_rate,
    )
    lb, ub = _build_bounds(bs_config, pm, active_fixed)
    bounds_bs = list(zip(lb.tolist(), ub.tolist()))
    A_subst, b_subst, _ = _build_subst_constraints(pm, params)

    A_eq_bs = np.zeros((2, pm.n_prod))
    A_eq_bs[0, pm.asset_mask] = 1.0
    A_eq_bs[1, pm.fund_mask]  = 1.0
    b_eq = np.array([pm.asset_sum, pm.fund_sum])

    # ── Swap-side LP coefficients ──────────────────────────────────────────
    print(f"  Pricing swap ladder ({len(tenors)} tenors) against bs_optimizer's "
          f"{n_scen} EBA scenarios...")
    ladder = _price_ladder_vs_scenarios(shocked_scens, tenors, direction=None)
    bucket_ids = ladder["bucket_ids"]
    n_buckets  = len(bucket_ids)
    direction  = ladder["direction"]

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
        hard_rows.append(_joint_row(np.zeros(pm.n_prod), np.ones(n_buckets)))
        hard_rhs.append(total_notional_cap / NOTIONAL_UNIT)
    if seasoned_notional_cap is not None and seasoned_mask.any():
        hard_rows.append(_joint_row(np.zeros(pm.n_prod), seasoned_mask.astype(float)))
        hard_rhs.append(seasoned_notional_cap / NOTIONAL_UNIT)

    A_hard = np.vstack(hard_rows)
    b_hard = np.array(hard_rhs, dtype=float)
    n_hard = A_hard.shape[0]

    n_slack = len(soft_names)
    if n_slack > 0:
        A_soft = np.vstack(soft_rows)
        b_soft = np.array(soft_rhs, dtype=float)
        w_soft = np.array(soft_w, dtype=float)
        c_lp = np.concatenate([-c_ep_lp, np.zeros(n_buckets), w_soft])
        A_hard_aug = np.hstack([A_hard, np.zeros((n_hard, n_slack))])
        A_soft_aug = np.hstack([A_soft, -np.eye(n_slack)])
        A_ub = np.vstack([A_hard_aug, A_soft_aug])
        b_ub = np.concatenate([b_hard, b_soft])
        bounds_all = bounds_bs + bounds_swap + [(0.0, None)] * n_slack
    else:
        c_lp = np.concatenate([-c_ep_lp, np.zeros(n_buckets)])
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
    A_growth = np.hstack([
        np.eye(pm.n_prod), -np.eye(pm.n_prod), np.zeros((pm.n_prod, n_swap_slack_cols))
    ])
    A_ub = np.vstack([A_ub, A_growth])
    b_ub = np.concatenate([b_ub, pm.base_prod_w])
    c_lp = np.concatenate([c_lp[:pm.n_prod], pm.acq_cost_prod * ta, c_lp[pm.n_prod:]])
    bounds_all = bounds_all[:pm.n_prod] + [(0.0, None)] * pm.n_prod + bounds_all[pm.n_prod:]

    print(f"  Solving joint LP: {pm.n_prod} BS weights + {n_buckets} swap buckets"
          f" + {n_slack} soft-breach slack(s)...")
    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds_all, method="highs")
    elapsed = time.time() - t_start

    if res.success:
        x_bs   = res.x[:pm.n_prod]
        # res.x layout: [x_bs (n_prod), g (n_prod, growth-cost epigraph), x_swap (n_buckets), s (n_slack)]
        x_swap = res.x[2 * pm.n_prod:2 * pm.n_prod + n_buckets] * NOTIONAL_UNIT
        s_sol  = res.x[2 * pm.n_prod + n_buckets:] if n_slack > 0 else np.array([])
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
    fee_new = float(np.dot(amounts_bs, fee_rate))
    el_new  = float(np.dot(amounts_bs, params.el_unit))
    coc_new = rwa_bs * params.cet1_target * params.coc_rate
    opex_new = OPEX_RATE * float(np.sum(amounts_bs[opex_mask]))
    acq_cost_new = float(np.dot(pm.acq_cost_prod, np.maximum(0.0, x_bs - pm.base_prod_w))) * ta
    ep_bs_new = margin_new + fee_new - el_new - coc_new - opex_new - acq_cost_new
    ep_carry_swap = float(np.dot(ladder["ep_carry_pln"], x_swap))
    ep_total_new = ep_bs_new + ep_carry_swap

    sot_eve_out: dict[str, float] = {}
    sot_nii_out: dict[str, float] = {}
    for k, s in enumerate(shocked_scens):
        eve_bs_pct = eve_bs.get(s, 0.0) / t1 * 100.0
        nii_delta_bs_pct = (nii_bs.get(s, nii_base_new) - nii_base_new) / t1 * 100.0
        swap_eve_pct = float(np.dot(ladder["delta_eve_pln"][s], x_swap)) / t1 * 100.0
        swap_nii_pct = float(np.dot(ladder["delta_nii_pln"][s], x_swap)) / t1 * 100.0
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

    return JointResult(
        success=res.success, message=res.message, elapsed_s=elapsed,
        tier1_capital=t1,
        ep_bs_old=ep_bs_old, ep_bs_new=ep_bs_new,
        margin_bs_old=margin_base_old, margin_bs_new=margin_new,
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
    )


if __name__ == "__main__":
    _PARAMS = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "output", "product_params.npz"))
    params = BalanceSheetParams.load(_PARAMS)
    cr     = CohortRates.load(_PARAMS)

    cfg = OptimizationConfig(
        mode="max_shift", max_shift=0.03,
        eve_breach_weight=20.0, nii_breach_weight=20.0,
    )
    result = optimize_bs_and_ladder(cfg, params, cr)
    result.print_summary()
