"""stochastic_joint_optimizer.py
================================
Joint LP: balance-sheet product weights (bs_optimizer.py) + NEW incremental
swap-ladder notional (swap_ladder.py's 300 buckets), optimized TOGETHER to
maximize risk-adjusted profit over the SAME 500 mean-reverting Monte Carlo
curve scenarios stochastic_optimizer.py already uses -- not the 7 official
EBA scenarios joint_optimizer.py (Section 9) uses for its objective.

    max   E_s[EP_bs(x_bs,s) + EP_swap(x_swap,s)] - lambda * CVaR_alpha(-EP_total)
    s.t.  balance-sheet identity, NSFR, RWA cap, substitution      (x_bs only, hard)
          EVE_SOT[s] = A_eve_bs[s]@x_bs + A_eve_swap[s]@x_swap >= eve_floor   (7 official scenarios, hard)
          NII_SOT[s] = A_nii_bs[s]@x_bs + A_nii_swap[s]@x_swap >= nii_floor   (7 official scenarios, hard)
          x_bs bounds per bs_config.mode ; 0 <= x_swap[i] <= notional_cap ; sum(x_swap) <= total_notional_cap

Why the swap's carry IS in the objective here (unlike joint_optimizer.py)
---------------------------------------------------------------------------
joint_optimizer.py deliberately zeroes the swap's own carry out of ITS
objective (see that module's docstring) because its objective is pure BS EP
maximization subject to hard/soft regulatory rows -- the swap only needs to
affect those constraint rows, and giving it a nonzero carry coefficient there
made the optimizer chase carry as a second, unconstrained profit source.

Here the objective is CVaR over Monte Carlo scenarios, which needs to see
EVERY instrument's real scenario-varying payoff to value risk reduction at
all -- a swap with a zero objective coefficient could never reduce tail risk
in a CVaR objective, which would defeat the entire point of adding it. So the
swap's per-scenario carry (its NII, since a derivative overlay has no
FTP/fee/EL/CoC of its own -- unlike a balance-sheet product, its "EP" IS its
net interest carry) is included directly in the joint per-scenario matrix.

Consequence: because a swap with an apparent positive-mean/low-variance
payoff (given the linear/first-order approximations everywhere in this
codebase) would otherwise get bought up to the cap, the notional caps
(notional_cap, total_notional_cap, seasoned_notional_cap) are a REQUIRED
hard backstop here, not optional decoration -- exactly as in
joint_optimizer.py, just for a different underlying reason.

Scope (same limitations as joint_optimizer.py, not new here):
  - Does not re-derive the EXISTING ~840M IRS book (always pinned in x_bs).
  - NSFR/LCR/RWA modeled on the banking-book side only.
  - The regulatory EVE/NII SOT floor stays pinned to the 7 official EBA
    scenarios (via joint_optimizer._price_ladder_vs_scenarios, reused
    as-is) -- the Monte Carlo layer feeds the OBJECTIVE only, matching every
    other optimizer's convention in this codebase.
  - sot_eve/sot_nii (7-scenario, for reporting) combine an EXACT BS-side CF
    reprice with the swap ladder's ANALYTIC linear delta on top -- identical
    combination to how joint_optimizer.JointResult already reports today,
    NOT a from-scratch exact reprice of the combined book+swap instrument
    set (that would require injecting the new swap's cash flows into
    bank_reprice_at_weights.py's SQL-backed CF-schedule engine, which has no
    mechanism for synthetic new instruments -- a separate, harder follow-on).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

_HERE = os.path.dirname(os.path.abspath(__file__))
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

import swap_ladder as sl
import joint_optimizer as jo
from bs_optimizer import (
    OptimizationConfig, ProductChange,
    _ProductMap, _resolve_irs, _build_bounds, _build_subst_constraints,
)
from stochastic_optimizer import StochasticConfig, build_scenario_ep_matrix, DEFAULT_T_GRID

NOTIONAL_UNIT = jo.NOTIONAL_UNIT   # 1e6 PLN -- same LP-conditioning convention, don't re-derive


@dataclass
class StochasticJointResult:
    success:            bool
    message:            str
    n_scenarios:        int
    risk_lambda:        float
    cvar_alpha:         float
    elapsed_s:          float
    tier1_capital:      float

    weights_bs_new:     np.ndarray          # (n_prod,) BS-side optimized product weights
    product_changes:    list                # list[ProductChange], BS side
    swap_notional:      dict                # bucket_id -> PLN notional (new overlay only)
    swap_direction:     float
    ladder:             dict                # full ladder_mc dict (bucket_ids/elapsed_m/tenor_years/...)

    ep_mean_old:        float               # PLN -- E[EP] at baseline (BS only, zero new swap)
    ep_mean_new:        float               # PLN -- E[EP] at the optimized joint solution
    cvar_old:           float
    cvar_new:           float
    var_new:            float
    ep_scenarios_old:   np.ndarray          # (N,)
    ep_scenarios_new:   np.ndarray          # (N,)
    acq_cost_new:       float

    sot_eve:            dict                # scenario -> % T1, 7 official scenarios (BS exact + swap analytic)
    sot_nii:            dict
    sot_eve_floor:      float
    sot_nii_floor:      float

    lcr:                dict
    nsfr:               dict
    rwa:                float

    def print_summary(self, top_n_buckets: int = 10) -> None:
        status = "SUCCESS" if self.success else "FAILED"
        print(f"\nStochastic Joint BS + IRS optimization - {status}")
        print(f"  N scenarios: {self.n_scenarios}   lambda: {self.risk_lambda:.2f}"
              f"   CVaR alpha: {self.cvar_alpha:.2f}   elapsed: {self.elapsed_s:.2f}s")
        print(f"  {self.message}")
        print(f"  E[EP]           : {self.ep_mean_old/1e6:>10,.2f}M  ->  {self.ep_mean_new/1e6:>10,.2f}M"
              f"   ({(self.ep_mean_new-self.ep_mean_old)/1e6:+,.2f}M)")
        print(f"  Worst-tail EP{int(self.cvar_alpha*100)}: {self.cvar_old/1e6:>10,.2f}M  ->  {self.cvar_new/1e6:>10,.2f}M"
              f"   ({(self.cvar_new-self.cvar_old)/1e6:+,.2f}M)")

        print(f"\n  EVE SOT  (floor {self.sot_eve_floor:.1f}% T1):")
        for s, v in sorted(self.sot_eve.items()):
            ok = "OK" if v >= self.sot_eve_floor else "!!"
            print(f"    {s:<16s}: {v:+7.2f}% T1  [{ok}]")
        print(f"  NII SOT  (floor {self.sot_nii_floor:.1f}% T1):")
        for s, v in sorted(self.sot_nii.items()):
            ok = "OK" if v >= self.sot_nii_floor else "!!"
            print(f"    {s:<16s}: {v:+7.2f}% T1  [{ok}]")

        direction_lbl = "pay-fixed/receive-float" if self.swap_direction > 0 else "receive-fixed/pay-float"
        nz = [(b, n) for b, n in self.swap_notional.items() if n > 1.0]
        total = sum(n for _, n in nz)
        print(f"\n  New swap overlay ({direction_lbl}): {len(nz)} active bucket(s), "
              f"total {total/1e6:,.1f}M PLN")
        for bid, notional in sorted(nz, key=lambda t: -t[1])[:top_n_buckets]:
            print(f"    {bid:16s}  {notional/1e6:,.1f}M")

        print("\n  Product weight changes (BS side):")
        top = sorted(self.product_changes, key=lambda c: abs(c.delta_pct), reverse=True)
        for c in top:
            if abs(c.delta_pct) < 0.001:
                continue
            print(f"    {c.product_code:6s} {c.bs_side}  {c.pct_old:6.2f}% -> {c.pct_new:6.2f}%"
                  f"  ({c.delta_pct:+.2f} pp)")


def build_joint_scenario_matrices(
    config: OptimizationConfig,
    stoch_config: StochasticConfig,
    params: BalanceSheetParams,
    cr: CohortRates,
    factor_draws: np.ndarray,     # (N, 3) -- SAME array fed to build_scenario_ep_matrix
    today_beta: np.ndarray,       # (3,) absolute NS betas at the anchor date factor_draws is relative to
    tau: float,
    tenors: tuple = sl.TENORS_YEARS,
    t_grid: np.ndarray = DEFAULT_T_GRID,
):
    """Joint (N, n_prod + n_buckets) per-scenario EP matrix, plus all LP
    building blocks needed for both the CVaR objective and the 7-scenario
    regulatory rows. Mirrors build_scenario_ep_matrix()'s (matrix, blocks)
    contract so a future risk-lambda sweep can reuse the expensive part.
    """
    C_bs, lp_blocks = build_scenario_ep_matrix(config, stoch_config, params, cr,
                                                 factor_draws, tau, t_grid)

    ladder_mc = sl.price_ladder_vs_mc_curves(factor_draws, today_beta, tenors=tenors, direction=None, tau=tau)
    # Reuse the SAME detected hedge direction for the 7-scenario regulatory
    # ladder rather than re-detecting it (a DB round-trip) a second time and
    # risking the two ladders disagreeing on direction.
    ladder7 = jo._price_ladder_vs_scenarios(lp_blocks["shocked_scens"], tenors, direction=ladder_mc["direction"])

    n_buckets = len(ladder_mc["bucket_ids"])
    shocked_scens = lp_blocks["shocked_scens"]

    # Swap-side per-scenario carry P&L (PLN per unit notional), same
    # construction as the BS side's c_ep_lp[None,:] + delta_nii_matrix:
    # baseline-scenario carry (ep_carry_pln) plus the scenario-varying
    # floating-leg NII delta vs today (delta_nii_pln).
    C_swap = ladder_mc["ep_carry_pln"][None, :] + ladder_mc["delta_nii_pln"]   # (N, n_buckets)
    C_scenarios_joint = np.hstack([C_bs, C_swap])                              # (N, n_prod+n_buckets)

    A_eve_swap_7 = np.vstack([ladder7["delta_eve_pln"][s] for s in shocked_scens])   # (n_scen_reg, n_buckets)
    A_nii_swap_7 = np.vstack([ladder7["delta_nii_pln"][s] for s in shocked_scens])

    lp_blocks.update(dict(
        n_buckets=n_buckets,
        bucket_ids=ladder_mc["bucket_ids"],
        elapsed_m=ladder_mc["elapsed_m"],
        tenor_years=ladder_mc["tenor_years"],
        fixed_rate_swap=ladder_mc["fixed_rate"],
        ep_carry_pln=ladder_mc["ep_carry_pln"],
        swap_direction=ladder_mc["direction"],
        A_eve_swap_7=A_eve_swap_7,
        A_nii_swap_7=A_nii_swap_7,
        ladder7=ladder7,
        ladder_mc=ladder_mc,
    ))
    return C_scenarios_joint, lp_blocks


def optimize_stochastic_joint(
    config: OptimizationConfig,
    stoch_config: StochasticConfig,
    params: BalanceSheetParams,
    cr: CohortRates,
    C_scenarios_joint: np.ndarray,
    lp_blocks: dict,
    notional_cap: float | None = 1e9,
    total_notional_cap: float | None = 5e9,
    seasoned_elapsed_months: int = 12,
    seasoned_notional_cap: float | None = 1e9,
) -> StochasticJointResult:
    """Solve the joint CVaR-augmented LP given a precomputed joint EP scenario
    matrix and LP building blocks from build_joint_scenario_matrices().

    Variable layout (fixed order): [x_bs (n_prod), g (n_prod, growth-cost
    epigraph), x_swap (n_buckets), VaR (1), u_1..u_N (N, CVaR slack)].
    """
    t_start = time.time()
    pm = lp_blocks["pm"]
    t1 = lp_blocks["t1"]
    ta = lp_blocks["ta"]
    n_prod = pm.n_prod
    n_buckets = lp_blocks["n_buckets"]
    N = C_scenarios_joint.shape[0]
    alpha = stoch_config.cvar_alpha
    lam = stoch_config.risk_lambda

    C_bs = C_scenarios_joint[:, :n_prod]        # (N, n_prod), PLN per unit weight
    C_swap = C_scenarios_joint[:, n_prod:]      # (N, n_buckets), PLN per unit notional

    _IRS_CODE = "0000"
    irs_products = {
        (str(pc), str(side)) for pc, side in zip(params.product_code, params.bs_side)
        if str(pc) == _IRS_CODE
    }
    active_fixed = config.fixed_products | irs_products
    lb, ub = _build_bounds(config, pm, active_fixed)

    eve_floor_eff = config.sot_eve_floor - config.sot_eve_buffer
    nii_floor_eff = config.sot_nii_floor - config.sot_nii_buffer
    shocked_scens = lp_blocks["shocked_scens"]
    n_scen_reg = len(shocked_scens)

    # ── 7-scenario regulatory rows: BS side from lp_blocks (already in
    #    "%T1 x100 per unit x_j" units), swap side scaled to match, then to
    #    NOTIONAL_UNIT since x_swap is solved in units of 1e6 PLN ──────────
    A_eve_swap = lp_blocks["A_eve_swap_7"] / t1 * 100.0 * NOTIONAL_UNIT   # (n_scen_reg, n_buckets)
    A_nii_swap = lp_blocks["A_nii_swap_7"] / t1 * 100.0 * NOTIONAL_UNIT

    def _joint_bs_swap(row_bs, row_swap):
        return np.concatenate([row_bs, row_swap])

    reg_rows = [-_joint_bs_swap(lp_blocks["A_eve"][k], A_eve_swap[k]) for k in range(n_scen_reg)]
    reg_rhs = [-eve_floor_eff] * n_scen_reg
    reg_rows += [-_joint_bs_swap(lp_blocks["A_nii_delta"][k], A_nii_swap[k]) for k in range(n_scen_reg)]
    reg_rhs += [-nii_floor_eff] * n_scen_reg

    reg_rows.append(_joint_bs_swap(lp_blocks["A_nsfr"], np.zeros(n_buckets)))
    reg_rhs.append(0.0)
    if config.min_t1_rwa > 0.0:
        reg_rows.append(_joint_bs_swap(lp_blocks["c_rwa"], np.zeros(n_buckets)))
        reg_rhs.append(t1 / config.min_t1_rwa)

    A_subst, b_subst, _ = _build_subst_constraints(pm, params)
    if A_subst is not None:
        for row, rhs in zip(A_subst, b_subst):
            reg_rows.append(_joint_bs_swap(row, np.zeros(n_buckets)))
            reg_rhs.append(rhs)

    cap_u = None if notional_cap is None else notional_cap / NOTIONAL_UNIT
    bounds_swap = [(0.0, cap_u)] * n_buckets
    seasoned_mask = lp_blocks["elapsed_m"] > seasoned_elapsed_months
    if total_notional_cap is not None:
        reg_rows.append(_joint_bs_swap(np.zeros(n_prod), np.ones(n_buckets)))
        reg_rhs.append(total_notional_cap / NOTIONAL_UNIT)
    if seasoned_notional_cap is not None and seasoned_mask.any():
        reg_rows.append(_joint_bs_swap(np.zeros(n_prod), seasoned_mask.astype(float)))
        reg_rhs.append(seasoned_notional_cap / NOTIONAL_UNIT)

    A_reg = np.vstack(reg_rows)          # (n_reg, n_prod + n_buckets)
    b_reg = np.array(reg_rhs, dtype=float)
    n_reg = A_reg.shape[0]

    A_eq_bs = np.zeros((2, n_prod))
    A_eq_bs[0, pm.asset_mask] = 1.0
    A_eq_bs[1, pm.fund_mask] = 1.0
    b_eq = np.array([pm.asset_sum, pm.fund_sum])

    n_extra = 1 + N   # VaR + u_1..N

    c_lp = np.concatenate([
        -C_bs.mean(axis=0),                       # x_bs
        pm.acq_cost_prod * ta,                     # g (growth-cost epigraph)
        -C_swap.mean(axis=0) * NOTIONAL_UNIT,      # x_swap (per-1e6-PLN units)
        [lam],                                      # VaR
        np.full(N, lam / (N * (1.0 - alpha))),      # u_1..N
    ])

    # ── splice g's columns in at the n_prod offset, AFTER building simple
    #    2-block [x_bs | x_swap,VaR,u] rows -- same trick joint_optimizer.py
    #    and stochastic_optimizer.py both already use for their own growth
    #    epigraph augmentation ────────────────────────────────────────────
    A_reg_aug = np.hstack([
        A_reg[:, :n_prod], np.zeros((n_reg, n_prod)),
        A_reg[:, n_prod:], np.zeros((n_reg, n_extra)),
    ])
    A_eq_aug = np.hstack([A_eq_bs, np.zeros((2, n_prod + n_buckets + n_extra))])

    # CVaR rows: -C_bs@x_bs - C_swap*NOTIONAL_UNIT@x_swap - VaR - u_k <= 0
    A_cvar = np.hstack([
        -C_bs, np.zeros((N, n_prod)),
        -C_swap * NOTIONAL_UNIT,
        -np.ones((N, 1)), -np.eye(N),
    ])

    # Growth-cost epigraph: x_bs_j - g_j <= base_prod_w_j
    A_growth = np.hstack([
        np.eye(n_prod), -np.eye(n_prod),
        np.zeros((n_prod, n_buckets)), np.zeros((n_prod, n_extra)),
    ])

    A_ub = np.vstack([A_reg_aug, A_cvar, A_growth])
    b_ub = np.concatenate([b_reg, np.zeros(N), pm.base_prod_w])

    bounds = (list(zip(lb.tolist(), ub.tolist()))
              + [(0.0, None)] * n_prod
              + bounds_swap
              + [(None, None)]
              + [(0.0, None)] * N)

    lp_res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_aug, b_eq=b_eq,
                      bounds=bounds, method="highs")
    elapsed = time.time() - t_start

    if lp_res.success:
        x_bs = lp_res.x[:n_prod]
        x_swap = lp_res.x[2 * n_prod:2 * n_prod + n_buckets] * NOTIONAL_UNIT
    else:
        x_bs = pm.base_prod_w.copy()
        x_swap = np.zeros(n_buckets)

    acq_cost_new = float(np.dot(pm.acq_cost_prod, np.maximum(0.0, x_bs - pm.base_prod_w))) * ta
    # x_swap is already actual PLN notional (converted at the line above) and
    # C_swap is PLN P&L per PLN of notional -- C_swap @ x_swap directly gives
    # PLN P&L, no further unit conversion. (An earlier version of this line
    # divided x_swap by NOTIONAL_UNIT again here, silently understating the
    # swap's P&L contribution by 1e6x -- caught because it made the "more
    # flexibility can only weakly improve the optimum" LP sanity check fail:
    # the joint result looked WORSE than the notional_cap=0 degenerate case,
    # which is mathematically impossible for a correctly scaled relaxation.)
    ep_new = C_bs @ x_bs + C_swap @ x_swap - acq_cost_new
    ep_old = C_bs @ pm.base_prod_w       # baseline: today's BS weights, zero new swap overlay

    def _tail_ep(ep: np.ndarray) -> tuple:
        var_ep = float(np.quantile(ep, 1.0 - alpha))
        tail = ep[ep <= var_ep]
        tail_ep = float(tail.mean()) if len(tail) > 0 else var_ep
        return var_ep, tail_ep

    var_new, cvar_new = _tail_ep(ep_new)
    _, cvar_old = _tail_ep(ep_old)

    # ── SOT reporting (7 official scenarios): exact BS reprice + analytic
    #    swap delta on top -- identical combination to JointResult's ───────
    amounts_bs = pm.to_amounts(x_bs, ta)
    amounts_m = amounts_bs if config.include_irs else np.where(lp_blocks["irs_cohort_mask"], 0.0, amounts_bs)
    nii_bs = compute_nii_cf_all(amounts_m, params, cr)
    eve_bs = compute_eve_cf_fast_all(amounts_m, params, cr)
    lcr_bs = compute_lcr_fast(amounts_m, params)
    nsfr_bs = compute_nsfr_fast(amounts_m, params)
    rwa_bs = compute_rwa_fast(amounts_m, params)
    bias_eve, bias_nii, bias_scens = load_bias_corrections()
    if bias_eve is not None:
        apply_eve_bias(eve_bs, amounts_m, bias_eve, bias_scens)
        apply_nii_bias(nii_bs, amounts_m, bias_nii, bias_scens)
    nii_base_new = nii_bs["base"]

    ladder7 = lp_blocks["ladder7"]
    sot_eve_out: dict[str, float] = {}
    sot_nii_out: dict[str, float] = {}
    for s in shocked_scens:
        eve_bs_pct = eve_bs.get(s, 0.0) / t1 * 100.0
        nii_delta_bs_pct = (nii_bs.get(s, nii_base_new) - nii_base_new) / t1 * 100.0
        swap_eve_pct = float(np.dot(ladder7["delta_eve_pln"][s], x_swap)) / t1 * 100.0
        swap_nii_pct = float(np.dot(ladder7["delta_nii_pln"][s], x_swap)) / t1 * 100.0
        sot_eve_out[s] = eve_bs_pct + swap_eve_pct
        sot_nii_out[s] = nii_delta_bs_pct + swap_nii_pct

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
    swap_notional = {bid: float(n) for bid, n in zip(lp_blocks["bucket_ids"], x_swap)}

    return StochasticJointResult(
        success=lp_res.success, message=lp_res.message,
        n_scenarios=N, risk_lambda=lam, cvar_alpha=alpha, elapsed_s=elapsed,
        tier1_capital=t1,
        weights_bs_new=x_bs, product_changes=changes,
        swap_notional=swap_notional, swap_direction=lp_blocks["swap_direction"],
        ladder=lp_blocks["ladder_mc"],
        ep_mean_old=float(ep_old.mean()), ep_mean_new=float(ep_new.mean()),
        cvar_old=cvar_old, cvar_new=cvar_new, var_new=var_new,
        ep_scenarios_old=ep_old, ep_scenarios_new=ep_new, acq_cost_new=acq_cost_new,
        sot_eve=sot_eve_out, sot_nii=sot_nii_out,
        sot_eve_floor=eve_floor_eff, sot_nii_floor=nii_floor_eff,
        lcr=dict(lcr_bs), nsfr=dict(nsfr_bs), rwa=rwa_bs,
    )
