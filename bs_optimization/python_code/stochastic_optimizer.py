"""stochastic_optimizer.py
=========================
Phase A — stochastic single-period balance sheet optimization.

Maximizes expected Economic Profit across N Monte Carlo interest-rate curve
scenarios while controlling downside risk via CVaR (Rockafellar-Uryasev LP
formulation), subject to the SAME regulatory constraints (EVE/NII SOT, LCR,
NSFR, T1/RWA) evaluated on the official EBA shock scenarios only — Monte
Carlo scenarios feed the objective, not the regulatory constraint set.

    max_x  E_s[EP(x, s)] - lambda * CVaR_alpha( -EP(x, s) )
    s.t.   regulatory constraints on the 7 official EBA scenarios (unchanged)
           balance-sheet integrity, substitution, mode bounds (unchanged)

CVaR (Rockafellar-Uryasev, 1997) — linear in x via auxiliary variables
(VaR, u_1..u_N):
    CVaR_alpha(L) = min_VaR { VaR + 1/(N(1-alpha)) * sum_k max(L_k - VaR, 0) }
    u_k >= L_k - VaR ,  u_k >= 0 ,  L_k = -EP_k(x)

Reuses bs_optimizer._compute_lp_coefficients (for the base EP coefficients and
the 7-scenario A_nii_delta matrix) and mc_scenario_basis (to turn Monte Carlo
NS draws into per-product NII-delta coefficients without re-running the
CF-based cohort pipeline for every scenario).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPTPREP = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "python_code"))
for _p in (_HERE, _OPTPREP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bs_vector import BalanceSheetParams, CohortRates
from bias_store import load_bias_corrections
from ftp_store  import load_ftp_rates, margin_unit_rate
from bs_optimizer import (
    OptimizationConfig, ProductChange, _ProductMap,
    _compute_lp_coefficients, _build_subst_constraints,
)
from mc_scenario_basis import build_delta_nii_operator

DEFAULT_T_GRID = np.arange(1, 361) / 12.0   # monthly grid, 1..360 months in years


@dataclass
class StochasticConfig:
    n_scenarios:    int   = 500
    horizon_months: float = 3.0     # Monte Carlo horizon for the factor shocks
    cvar_alpha:     float = 0.95    # CVaR confidence level (worst 1-alpha tail)
    risk_lambda:    float = 1.0     # risk-aversion weight: 0 = pure E[EP], larger = more conservative
    seed:           int   = 42
    tau_years:      float | None = None   # None = grid-search per fit_ns_tau()


@dataclass
class StochasticResult:
    success:             bool
    message:             str
    n_scenarios:         int
    risk_lambda:         float
    cvar_alpha:          float
    weights_new:         np.ndarray          # (n_prod,) fraction of total_assets
    product_changes:     list[ProductChange]
    ep_mean_new:         float               # E[EP] at optimized weights (PLN)
    ep_mean_old:         float               # E[EP] at baseline weights (PLN)
    cvar_new:            float               # worst-tail EP (avg EP over worst 1-alpha scenarios), optimized (PLN)
    cvar_old:            float               # worst-tail EP, baseline (PLN)
    var_new:             float               # VaR-EP: EP threshold at the (1-alpha) quantile, optimized (PLN)
    acq_cost_new:        float               # PLN -- marketing/acquisition cost at optimized weights
                                              # (only on growth above baseline; always 0.0 at baseline)
    ep_scenarios_new:    np.ndarray          # (N,) realized EP draws at optimized weights
    ep_scenarios_old:    np.ndarray          # (N,) realized EP draws at baseline weights
    elapsed_s:            float

    def print_summary(self) -> None:
        status = "SUCCESS" if self.success else "FAILED"
        print(f"\nStochastic optimization (Phase A) - {status}")
        print(f"  N scenarios: {self.n_scenarios}   lambda: {self.risk_lambda:.2f}"
              f"   CVaR alpha: {self.cvar_alpha:.2f}   elapsed: {self.elapsed_s:.2f}s")
        print(f"  {self.message}")
        print(f"  E[EP]           : {self.ep_mean_old/1e6:>10,.2f}M  ->  {self.ep_mean_new/1e6:>10,.2f}M"
              f"   ({(self.ep_mean_new-self.ep_mean_old)/1e6:+,.2f}M)")
        print(f"  Worst-tail EP{int(self.cvar_alpha*100)}: {self.cvar_old/1e6:>10,.2f}M  ->  {self.cvar_new/1e6:>10,.2f}M"
              f"   ({(self.cvar_new-self.cvar_old)/1e6:+,.2f}M)   (avg EP over worst {int((1-self.cvar_alpha)*100)}% of scenarios)")
        print(f"  AcqCost         : {0.0:>10,.2f}M  ->  {self.acq_cost_new/1e6:>10,.2f}M"
              f"   (marketing cost, only on growth above baseline; already netted into E[EP]/CVaR above)")
        print("\n  Product weight changes:")
        for c in sorted(self.product_changes, key=lambda c: abs(c.delta_pct), reverse=True):
            if abs(c.delta_pct) < 0.001:
                continue
            print(f"    {c.product_code:6s} {c.bs_side}  {c.pct_old:6.2f}% -> {c.pct_new:6.2f}%"
                  f"  ({c.delta_pct:+.2f} pp)")


def _build_mode_bounds(config: OptimizationConfig, pm: _ProductMap, active_fixed: set) -> tuple:
    lb = np.zeros(pm.n_prod, dtype=float)
    ub = np.ones(pm.n_prod, dtype=float)

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
    else:
        raise ValueError(f"Unknown optimization mode: {config.mode!r}")

    return lb, ub


def build_scenario_ep_matrix(
    config: OptimizationConfig,
    stoch_config: StochasticConfig,
    params: BalanceSheetParams,
    cr: CohortRates,
    factor_draws: np.ndarray,     # (N, 3) from ns_curve_model.simulate_factor_shocks
    tau: float,
    t_grid: np.ndarray = DEFAULT_T_GRID,
):
    """Build the (N, n_prod) matrix of per-scenario EP-linear coefficients,
    plus the underlying LP building blocks (pm, c_ep_lp, t1, shocked_scens)
    needed to also build the regulatory-constraint side of the LP.
    """
    ta = params.total_assets
    t1 = (config.tier1_capital if config.tier1_capital > 0
          else float(params.balance_arr[params.is_equity].sum()))
    pm = _ProductMap(params)

    _IRS_CODE = "0000"
    irs_cohort_mask = np.array([str(pc) == _IRS_CODE for pc in params.product_code], dtype=bool)
    bias_eve, bias_nii, bias_scens = load_bias_corrections()
    shocked_scens = [str(s) for s in cr.rate_scenario_ids if str(s) != "base"]

    # Margin-over-FTP income (see bs_optimizer.py / ftp_store.py) -- same EP
    # definition as the other optimizers, so Section 5's frontier is
    # comparable to Section 4's baseline. Missing cache degrades to zero FTP.
    ftp_rate = load_ftp_rates(params.cohort_id)
    if ftp_rate is None:
        ftp_rate = np.zeros_like(params.nii_unit_rate)
    margin_rate    = margin_unit_rate(params.nii_unit_rate,         ftp_rate, params.bs_side)
    margin_rate_nb = margin_unit_rate(params.nii_unit_rate_new_biz, ftp_rate, params.bs_side)

    c_nii, A_eve, A_nii_delta, A_nsfr, c_rwa, c_el, c_ep, c_ep_lp, c_fee = _compute_lp_coefficients(
        pm, params, cr, t1, irs_cohort_mask, config.include_irs,
        bias_eve, bias_nii, bias_scens, shocked_scens,
        margin_rate, margin_rate_nb, params.fee_unit_rate,
    )

    R = build_delta_nii_operator(A_nii_delta, shocked_scens, tau, t_grid, t1)  # (3, n_prod)
    delta_nii_matrix = factor_draws @ R                                        # (N, n_prod)
    C_scenarios = c_ep_lp[None, :] + delta_nii_matrix                          # (N, n_prod)

    return C_scenarios, dict(
        pm=pm, t1=t1, ta=ta, c_ep_lp=c_ep_lp, A_eve=A_eve, A_nii_delta=A_nii_delta,
        A_nsfr=A_nsfr, c_rwa=c_rwa, shocked_scens=shocked_scens,
        irs_cohort_mask=irs_cohort_mask,
    )


def optimize_stochastic(
    config: OptimizationConfig,
    stoch_config: StochasticConfig,
    params: BalanceSheetParams,
    cr: CohortRates,
    C_scenarios: np.ndarray,
    lp_blocks: dict,
) -> StochasticResult:
    """Solve the CVaR-augmented LP given a precomputed (N, n_prod) EP scenario
    matrix and the regulatory LP building blocks from build_scenario_ep_matrix().

    Kept separate from build_scenario_ep_matrix() so a risk-frontier sweep over
    risk_lambda can reuse the same (expensive-ish) scenario matrix and only
    re-solve the (cheap) LP for each lambda.
    """
    t_start = time.time()
    pm = lp_blocks["pm"]
    t1 = lp_blocks["t1"]
    ta = lp_blocks["ta"]
    N, n_prod = C_scenarios.shape
    alpha  = stoch_config.cvar_alpha
    lam    = stoch_config.risk_lambda

    _IRS_CODE = "0000"
    irs_products = {
        (str(pc), str(side)) for pc, side in zip(params.product_code, params.bs_side)
        if str(pc) == _IRS_CODE
    }
    active_fixed = config.fixed_products | irs_products
    lb, ub = _build_mode_bounds(config, pm, active_fixed)

    eve_floor_eff = config.sot_eve_floor - config.sot_eve_buffer
    nii_floor_eff = config.sot_nii_floor - config.sot_nii_buffer
    shocked_scens = lp_blocks["shocked_scens"]
    n_scen_reg = len(shocked_scens)

    ineq_rows = [-lp_blocks["A_eve"], -lp_blocks["A_nii_delta"], lp_blocks["A_nsfr"][None, :]]
    ineq_rhs  = [np.full(n_scen_reg, -eve_floor_eff), np.full(n_scen_reg, -nii_floor_eff), np.array([0.0])]
    if config.min_t1_rwa > 0.0:
        ineq_rows.append(lp_blocks["c_rwa"][None, :])
        ineq_rhs.append(np.array([t1 / config.min_t1_rwa]))
    A_reg = np.vstack(ineq_rows)
    b_reg = np.concatenate(ineq_rhs)
    n_reg = A_reg.shape[0]

    A_eq_lp = np.zeros((2, pm.n_prod))
    A_eq_lp[0, pm.asset_mask] = 1.0
    A_eq_lp[1, pm.fund_mask]  = 1.0
    b_eq_lp = np.array([pm.asset_sum, pm.fund_sum])

    A_subst, b_subst, _ = _build_subst_constraints(pm, params)

    # ── variables z = [x (n_prod), g (n_prod, growth-cost epigraph), VaR (1), u_1..u_N (N)] ──
    # g_j = max(0, x_j - base_prod_w_j), same epigraph mechanism as
    # bs_optimizer.py's optimize_nii() -- see that module for the full
    # explanation. Growth cost is scenario-INDEPENDENT (a function of x alone),
    # so it's added once to the x/g objective block, not per Monte Carlo draw.
    n_extra = 1 + N
    c_lp = np.concatenate([
        -C_scenarios.mean(axis=0),
        pm.acq_cost_prod * ta,
        [lam],
        np.full(N, lam / (N * (1.0 - alpha))),
    ])

    A_reg_aug = np.hstack([A_reg, np.zeros((n_reg, n_prod)), np.zeros((n_reg, n_extra))])
    A_eq_aug  = np.hstack([A_eq_lp, np.zeros((2, n_prod)), np.zeros((2, n_extra))])
    rows = [A_reg_aug]
    rhs  = [b_reg]
    if A_subst is not None:
        rows.append(np.hstack([A_subst, np.zeros((A_subst.shape[0], n_prod)), np.zeros((A_subst.shape[0], n_extra))]))
        rhs.append(b_subst)

    # CVaR rows: -C_k @ x - VaR - u_k <= 0  (g columns zero -- CVaR doesn't involve g)
    A_cvar = np.hstack([-C_scenarios, np.zeros((N, n_prod)), -np.ones((N, 1)), -np.eye(N)])
    rows.append(A_cvar)
    rhs.append(np.zeros(N))

    # Growth-cost epigraph rows: x_j - g_j <= base_prod_w_j
    A_growth = np.hstack([np.eye(n_prod), -np.eye(n_prod), np.zeros((n_prod, n_extra))])
    rows.append(A_growth)
    rhs.append(pm.base_prod_w)

    A_ub = np.vstack(rows)
    b_ub = np.concatenate(rhs)

    bounds = (list(zip(lb.tolist(), ub.tolist())) + [(0.0, None)] * n_prod
              + [(None, None)] + [(0.0, None)] * N)

    lp_res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_aug, b_eq=b_eq_lp,
                      bounds=bounds, method="highs")
    elapsed = time.time() - t_start

    if lp_res.success:
        x_sol = lp_res.x[:pm.n_prod]
    else:
        x_sol = pm.base_prod_w.copy()

    acq_cost_new = float(np.dot(pm.acq_cost_prod, np.maximum(0.0, x_sol - pm.base_prod_w))) * ta
    ep_new = C_scenarios @ x_sol - acq_cost_new
    ep_old = C_scenarios @ pm.base_prod_w

    def _tail_ep(ep: np.ndarray) -> tuple:
        """VaR-EP (EP threshold at the (1-alpha) quantile) and the average EP
        over that worst tail — both reported in EP terms (positive = profit)."""
        var_ep = float(np.quantile(ep, 1.0 - alpha))
        tail = ep[ep <= var_ep]
        tail_ep = float(tail.mean()) if len(tail) > 0 else var_ep
        return var_ep, tail_ep

    var_new, cvar_new = _tail_ep(ep_new)
    _, cvar_old = _tail_ep(ep_old)

    changes = [
        ProductChange(
            product_code=pc, bs_side=side,
            weight_old=float(pm.base_prod_w[j]), weight_new=float(x_sol[j]),
            delta_weight=float(x_sol[j] - pm.base_prod_w[j]),
            pct_old=float(pm.base_prod_w[j]) * 100.0, pct_new=float(x_sol[j]) * 100.0,
            delta_pct=float(x_sol[j] - pm.base_prod_w[j]) * 100.0,
        )
        for j, (pc, side) in enumerate(pm.products)
    ]

    return StochasticResult(
        success=lp_res.success, message=lp_res.message,
        n_scenarios=N, risk_lambda=lam, cvar_alpha=alpha,
        weights_new=x_sol, product_changes=changes,
        ep_mean_new=float(ep_new.mean()), ep_mean_old=float(ep_old.mean()),
        cvar_new=cvar_new, cvar_old=cvar_old, var_new=var_new,
        ep_scenarios_new=ep_new, ep_scenarios_old=ep_old,
        acq_cost_new=acq_cost_new,
        elapsed_s=elapsed,
    )


def sweep_risk_frontier(
    config: OptimizationConfig,
    stoch_config: StochasticConfig,
    params: BalanceSheetParams,
    cr: CohortRates,
    C_scenarios: np.ndarray,
    lp_blocks: dict,
    lambdas: list,
) -> list:
    """Re-solve the (cheap) CVaR-LP for each risk_lambda, reusing the same
    (expensive-ish) scenario matrix. Returns a list of StochasticResult.
    """
    results = []
    for lam in lambdas:
        cfg_k = StochasticConfig(
            n_scenarios=stoch_config.n_scenarios, horizon_months=stoch_config.horizon_months,
            cvar_alpha=stoch_config.cvar_alpha, risk_lambda=lam, seed=stoch_config.seed,
        )
        results.append(optimize_stochastic(config, cfg_k, params, cr, C_scenarios, lp_blocks))
    return results
