"""irrbb_optimizer.py
=====================
"Natural hedge" balance-sheet optimizer: minimize IRRBB regulatory breach
severity (EVE + NII SOT shortfall below floor, summed across all 7 official
EBA scenarios) via product-mix reallocation ALONE -- no swap overlay, and
the sensitivities are built with include_irs=False (fixed, not user-
configurable) so the resulting structure reflects what the BANKING BOOK
alone can achieve, blind to the existing IRS book's own EVE/NII
contribution. IRS remains physically pinned in the balance sheet (same
_resolve_irs pinning every other optimizer in this codebase uses) -- this
optimizer just isn't given credit for it when scoring IRRBB robustness.

    min_x  eve_weight * sum_s max(0, eve_floor - deltaEVE_s(x)/T1*100)
         + nii_weight * sum_s max(0, nii_floor - deltaNII_s(x)/T1*100)
    s.t.   balance-sheet identity, NSFR, RWA cap, substitution   (hard)
           mode bounds (full/partial/max_shift/custom)

No EP term anywhere in the objective -- this is a pure robustness view,
answering "how IRRBB-robust could this balance sheet structure be if
profit weren't a consideration at all", as a reference point against the
EP-first structures in bs_optimizer.py / joint_optimizer.py. EP is still
computed and reported on the result (informational only, via the standard
Margin+Fee-EL-CoC-OpEx formula), never drives the choice.

Unlike every other optimizer in this codebase, the EVE/NII floors are NOT
hard constraints here -- they're the objective itself (fully soft, no floor
at all), since the whole point is to see how far the structure can be
pushed toward zero-breach, not just to clear the bar.
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
from lcr_fast   import compute_lcr_fast
from nsfr_fast  import compute_nsfr_fast
from rwa_fast   import compute_rwa_fast
from bias_store import load_bias_corrections, apply_eve_bias, apply_nii_bias
from ftp_store  import load_ftp_rates, margin_unit_rate

from bs_optimizer import (                                                 # noqa: E402
    OptimizationConfig, ProductChange, OPEX_RATE,
    _ProductMap, _resolve_irs, _build_bounds,
    _compute_lp_coefficients, _build_subst_constraints,
)


@dataclass
class IRRBBHedgeResult:
    success:        bool
    message:        str
    elapsed_s:      float
    tier1_capital:  float
    sot_eve_old:    dict    # scenario -> % T1, baseline weights, unhedged view
    sot_eve_new:    dict    # scenario -> % T1, optimized weights, unhedged view
    sot_nii_old:    dict
    sot_nii_new:    dict
    sot_eve_floor:  float   # effective floor (regulatory floor - buffer)
    sot_nii_floor:  float
    breach_eve_old: int     # count out of len(sot_eve_old)
    breach_eve_new: int
    breach_nii_old: int
    breach_nii_new: int
    severity_old:   float   # sum of shortfalls below floor, pp T1 (EVE+NII combined)
    severity_new:   float
    ep_old:         float   # PLN -- informational only, NOT optimized for
    ep_new:         float
    product_changes: list   # list[ProductChange]
    lcr:            dict
    nsfr:           dict
    rwa:            float
    t1_rwa_ratio:   float
    weights_new:    np.ndarray   # (n_prod,) product weights

    def print_summary(self, top_n: int = 15) -> None:
        status = "SUCCESS" if self.success else "FAILED"
        print(f"\nIRRBB natural-hedge optimization - {status}")
        print(f"  T1 capital: {self.tier1_capital/1e6:,.0f}M PLN  |  "
              f"unhedged view (IRS excluded from EVE/NII credit, still pinned in the book)")
        print(f"  {self.message}   elapsed={self.elapsed_s:.2f}s")
        n_scen = len(self.sot_eve_old)
        print(f"  EVE breaches (floor {self.sot_eve_floor:+.1f}% T1): "
              f"{self.breach_eve_old}/{n_scen}  ->  {self.breach_eve_new}/{n_scen}")
        print(f"  NII breaches (floor {self.sot_nii_floor:+.1f}% T1): "
              f"{self.breach_nii_old}/{n_scen}  ->  {self.breach_nii_new}/{n_scen}")
        print(f"  Total breach severity (pp T1, EVE+NII summed): "
              f"{self.severity_old:.2f}  ->  {self.severity_new:.2f}")
        print(f"  EP (informational, NOT optimized for): "
              f"{self.ep_old/1e6:,.1f}M  ->  {self.ep_new/1e6:,.1f}M")

        print(f"\n  EVE SOT (unhedged view):")
        for s, v in sorted(self.sot_eve_new.items()):
            ok = "OK" if v >= self.sot_eve_floor else "!!"
            print(f"    {s:<16s}: {v:+7.2f}% T1  [{ok}]")
        print(f"  NII SOT (unhedged view):")
        for s, v in sorted(self.sot_nii_new.items()):
            ok = "OK" if v >= self.sot_nii_floor else "!!"
            print(f"    {s:<16s}: {v:+7.2f}% T1  [{ok}]")

        for ccy, lcr_v in sorted(self.lcr.items()):
            print(f"  LCR  {ccy}: {lcr_v:.3f}")
        for ccy, nsfr_v in sorted(self.nsfr.items()):
            print(f"  NSFR {ccy}: {nsfr_v:.3f}")
        print(f"  RWA: {self.rwa/1e6:,.0f}M PLN   T1/RWA: {self.t1_rwa_ratio*100:.1f}%")

        print(f"\n  Product weight changes:")
        top = sorted(self.product_changes, key=lambda c: abs(c.delta_pct), reverse=True)[:top_n]
        for c in top:
            if abs(c.delta_pct) < 0.0001:
                continue
            print(f"    {c.product_code:6s} {c.bs_side}  "
                  f"{c.pct_old:6.2f}% -> {c.pct_new:6.2f}%  ({c.delta_pct:+.2f} pp)")


def optimize_min_irrbb_breach(
    config: OptimizationConfig,
    params: BalanceSheetParams,
    cr: CohortRates,
    eve_weight: float = 1.0,
    nii_weight: float = 1.0,
) -> IRRBBHedgeResult:
    t_start = time.time()
    pm = _ProductMap(params)
    ta = params.total_assets
    t1 = (config.tier1_capital if config.tier1_capital > 0
          else float(params.balance_arr[params.is_equity].sum()))

    irs_cohort_mask, irs_products, irs_notional = _resolve_irs(params)
    active_fixed = config.fixed_products | irs_products
    if irs_products:
        print(f"  IR derivatives: {irs_notional/1e6:,.0f}M notional -- always pinned, "
              f"EXCLUDED from EVE/NII credit by design (natural-hedge view)")

    bias_eve, bias_nii, bias_scens = load_bias_corrections()
    shocked_scens = [str(s) for s in cr.rate_scenario_ids if str(s) != "base"]
    n_scen = len(shocked_scens)

    ftp_rate = load_ftp_rates(params.cohort_id)
    if ftp_rate is None:
        ftp_rate = np.zeros_like(params.nii_unit_rate)
    margin_rate    = margin_unit_rate(params.nii_unit_rate,         ftp_rate, params.bs_side)
    margin_rate_nb = margin_unit_rate(params.nii_unit_rate_new_biz, ftp_rate, params.bs_side)

    print("  Building LP coefficients (unhedged view)...")
    _, A_eve, A_nii_delta, A_nsfr, c_rwa, c_el, _, _, _ = _compute_lp_coefficients(
        pm, params, cr, t1, irs_cohort_mask, False,
        bias_eve, bias_nii, bias_scens, shocked_scens,
        margin_rate, margin_rate_nb, params.fee_unit_rate,
    )

    lb, ub = _build_bounds(config, pm, active_fixed)
    bounds = list(zip(lb.tolist(), ub.tolist()))

    A_subst, b_subst, _ = _build_subst_constraints(pm, params)

    eve_floor_eff = config.sot_eve_floor - config.sot_eve_buffer
    nii_floor_eff = config.sot_nii_floor - config.sot_nii_buffer

    # ── Hard structural constraints -- NSFR / RWA cap only. EVE/NII floors are
    #    NOT here: they're fully absorbed into the slack objective below. ────
    hard_rows, hard_rhs = [A_nsfr[None, :]], [0.0]
    if config.min_t1_rwa > 0.0:
        hard_rows.append(c_rwa[None, :])
        hard_rhs.append(t1 / config.min_t1_rwa)
    A_reg = np.vstack(hard_rows)
    b_reg = np.array(hard_rhs, dtype=float)
    n_reg = A_reg.shape[0]

    A_eq_lp = np.zeros((2, pm.n_prod))
    A_eq_lp[0, pm.asset_mask] = 1.0
    A_eq_lp[1, pm.fund_mask]  = 1.0
    b_eq_lp = np.array([pm.asset_sum, pm.fund_sum])

    # ── Breach-severity slack: u_eve[s] >= max(0, eve_floor - deltaEVE_s(x)),
    #    u_nii[s] >= max(0, nii_floor - deltaNII_s(x)) -- variables z =
    #    [x (n_prod), u_eve (n_scen), u_nii (n_scen)] ─────────────────────────
    n_slack = 2 * n_scen
    A_eve_slack = np.hstack([-A_eve,       -np.eye(n_scen), np.zeros((n_scen, n_scen))])
    A_nii_slack = np.hstack([-A_nii_delta, np.zeros((n_scen, n_scen)), -np.eye(n_scen)])
    b_eve_slack = np.full(n_scen, -eve_floor_eff)
    b_nii_slack = np.full(n_scen, -nii_floor_eff)

    A_reg_aug = np.hstack([A_reg, np.zeros((n_reg, n_slack))])
    A_eq_aug  = np.hstack([A_eq_lp, np.zeros((2, n_slack))])
    rows = [A_reg_aug, A_eve_slack, A_nii_slack]
    rhs  = [b_reg, b_eve_slack, b_nii_slack]
    if A_subst is not None:
        rows.append(np.hstack([A_subst, np.zeros((A_subst.shape[0], n_slack))]))
        rhs.append(b_subst)
    A_ub = np.vstack(rows)
    b_ub = np.concatenate(rhs)

    c_lp = np.concatenate([
        np.zeros(pm.n_prod),
        np.full(n_scen, eve_weight),
        np.full(n_scen, nii_weight),
    ])
    bounds_lp = bounds + [(0.0, None)] * n_slack

    print(f"  Solving IRRBB-severity LP: {pm.n_prod} weights + {n_slack} breach-slack "
          f"variable(s), no EP term...")
    lp_res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_aug, b_eq=b_eq_lp,
                      bounds=bounds_lp, method="highs")
    elapsed = time.time() - t_start

    if lp_res.success:
        x_sol = lp_res.x[:pm.n_prod]
        print(f"  LP (HiGHS): optimal  elapsed={elapsed:.2f}s")
    else:
        print(f"  LP (HiGHS) failed ({lp_res.message}) -- returning baseline")
        x_sol = pm.base_prod_w.copy()

    # ── Exact CF-based reporting at the solution, unhedged view (IRS zeroed) ──
    amounts_new = pm.to_amounts(x_sol, ta)
    amounts_new_m = np.where(irs_cohort_mask, 0.0, amounts_new)
    nii_dict_new = compute_nii_cf_all(amounts_new_m, params, cr)
    eve_dict_new = compute_eve_cf_fast_all(amounts_new_m, params, cr)
    if bias_eve is not None:
        apply_eve_bias(eve_dict_new, amounts_new_m, bias_eve, bias_scens)
        apply_nii_bias(nii_dict_new, amounts_new_m, bias_nii, bias_scens)
    nii_base_new = nii_dict_new["base"]
    sot_eve_new = {s: eve_dict_new.get(s, 0.0) / t1 * 100.0 for s in shocked_scens}
    sot_nii_new = {s: (nii_dict_new.get(s, nii_base_new) - nii_base_new) / t1 * 100.0 for s in shocked_scens}

    amounts_old_m = np.where(irs_cohort_mask, 0.0, params.balance_arr)
    nii_dict_old = compute_nii_cf_all(amounts_old_m, params, cr)
    eve_dict_old = compute_eve_cf_fast_all(amounts_old_m, params, cr)
    if bias_eve is not None:
        apply_eve_bias(eve_dict_old, amounts_old_m, bias_eve, bias_scens)
        apply_nii_bias(nii_dict_old, amounts_old_m, bias_nii, bias_scens)
    nii_base_old = nii_dict_old["base"]
    sot_eve_old = {s: eve_dict_old.get(s, 0.0) / t1 * 100.0 for s in shocked_scens}
    sot_nii_old = {s: (nii_dict_old.get(s, nii_base_old) - nii_base_old) / t1 * 100.0 for s in shocked_scens}

    def _breach_count(sot, floor):
        return sum(1 for v in sot.values() if v < floor - 1e-6)

    def _severity(sot, floor):
        return sum(max(0.0, floor - v) for v in sot.values())

    breach_eve_old = _breach_count(sot_eve_old, eve_floor_eff)
    breach_eve_new = _breach_count(sot_eve_new, eve_floor_eff)
    breach_nii_old = _breach_count(sot_nii_old, nii_floor_eff)
    breach_nii_new = _breach_count(sot_nii_new, nii_floor_eff)
    severity_old = _severity(sot_eve_old, eve_floor_eff) + _severity(sot_nii_old, nii_floor_eff)
    severity_new = _severity(sot_eve_new, eve_floor_eff) + _severity(sot_nii_new, nii_floor_eff)

    # ── Informational EP (Margin+Fee-EL-CoC-OpEx), full amounts, NOT masked --
    #    matches every other optimizer's convention that margin/fee/EL/CoC/
    #    OpEx are hedge-view-invariant; never drives this optimizer's choice ──
    def _ep_at(amounts):
        margin = float(np.dot(amounts, margin_rate))
        fee    = float(np.dot(amounts, params.fee_unit_rate))
        rwa    = compute_rwa_fast(amounts, params)
        el     = float(np.dot(amounts, params.el_unit))
        coc    = rwa * params.cet1_target * params.coc_rate
        opex   = OPEX_RATE * float(np.sum(amounts[~params.is_equity]))
        return margin + fee - el - coc - opex, rwa

    ep_old, _rwa_old = _ep_at(params.balance_arr)
    ep_new, rwa_new  = _ep_at(amounts_new)

    lcr_dict  = compute_lcr_fast(amounts_new, params)
    nsfr_dict = compute_nsfr_fast(amounts_new, params)
    t1_rwa_ratio = t1 / rwa_new if rwa_new > 0 else float("nan")

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

    return IRRBBHedgeResult(
        success=lp_res.success, message=lp_res.message, elapsed_s=elapsed,
        tier1_capital=t1,
        sot_eve_old=sot_eve_old, sot_eve_new=sot_eve_new,
        sot_nii_old=sot_nii_old, sot_nii_new=sot_nii_new,
        sot_eve_floor=eve_floor_eff, sot_nii_floor=nii_floor_eff,
        breach_eve_old=breach_eve_old, breach_eve_new=breach_eve_new,
        breach_nii_old=breach_nii_old, breach_nii_new=breach_nii_new,
        severity_old=severity_old, severity_new=severity_new,
        ep_old=ep_old, ep_new=ep_new,
        product_changes=changes,
        lcr=dict(lcr_dict), nsfr=dict(nsfr_dict), rwa=rwa_new, t1_rwa_ratio=t1_rwa_ratio,
        weights_new=x_sol,
    )
