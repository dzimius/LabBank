"""stochastic_irrbb_optimizer.py
================================
"Natural hedge" optimizer, extended: minimize IRRBB regulatory breach
SEVERITY (EVE + NII SOT shortfall below floor) -- jointly over BS product
weights AND a new swap-ladder overlay -- across the SAME 500 mean-reverting
Monte Carlo curves x 6 official EBA shocks (3000 combos) used throughout
Section 5/9B, instead of irrbb_optimizer.py's 7-official-scenario set.

Built because Section 9B's stochastic_joint_optimizer.py (CVaR of drift-only
P&L, maximized) turned out NOT to minimize the real breach count: its BS
weights alone were LESS conservative than Section 5's BS-only optimum (the
CVaR objective treats the swap as a safety net and lets BS take more risk in
response), and the swap only partially clawed that back once tested exactly.
This module fixes that by making breach severity ITSELF the objective, no
EP/CVaR proxy anywhere -- mirrors irrbb_optimizer.py's design exactly (no
growth-cost epigraph either: there's no EP term for a growth cost to net
against).

    min_{x_bs,x_swap}  eve_weight * sum_{c,s} max(0, eve_floor - deltaEVE(c,s))
                      + nii_weight * sum_{c,s} max(0, nii_floor - deltaNII(c,s))
    s.t.  balance-sheet identity, NSFR, RWA cap, substitution   (x_bs only, hard)
          x_bs bounds per config.mode ; 0 <= x_swap[i] <= notional_cap ; sum(x_swap) <= total_notional_cap

BS-side sensitivity is a LOCAL-GRADIENT APPROXIMATION, not exact -- building
a genuine (3000, n_prod) gradient via finite-difference-of-exact-CF-engine at
all 500 curves would cost ~2 hours (500 curves x ~21 perturbations x
~0.8s/curve, per this session's own measured bank_reprice_at_weights.py
timing). Instead: reuse the EXISTING 7-official-scenario gradient
A_eve/A_nii_delta (bs_optimizer._compute_lp_coefficients, evaluated at
TODAY's curve) as a curve-INVARIANT local slope, added to the EXACT per-curve
baseline value already cached in irrbb_mc500_baseline.xlsx (from this
session's earlier bank_reprice_at_weights.py precompute):

    delta_eve_approx(c, s, x) = E0[c,s] + A_eve[s,:] @ (x - base_prod_w)

A first-order Taylor expansion around baseline weights -- valid because
config.mode="max_shift" keeps x within a small (+-3pp) neighborhood of
base_prod_w regardless of which curve c is being evaluated, so the
CURVE-DEPENDENT nonlinearity (rate floors/caps -- the actual source of the
breach-concentration pattern found earlier this session) is captured EXACTLY
through E0, and only the secondary local-slope term is approximated. This
means the LP's own reported severity is APPROXIMATE -- always verify the
solution with the exact CF engine afterward (see
precompute_stochastic_irrbb_mc500.py), exactly like every other
approximate-optimize/exact-verify optimizer in this codebase.

Swap-side sensitivity is EXACT (not approximated): swap_ladder.py's
analytic DCF pricing (price_ladder_against_curve_bank) doesn't need any
Taylor-expansion shortcut -- it prices every curve directly.

Hedged basis, by deliberate choice: irrbb_optimizer.py itself is unhedged
(include_irs=False, blind to the existing IRS book, "natural hedge" framing).
This module instead uses the HEDGED basis (include_irs=True, matching
optimizer_config.xlsx's default and every existing irrbb_mc500_*.xlsx cache)
-- the BS-side gradient is numerically identical either way (the CF engine is
linear per-cohort with no cross terms, and IRS is always pinned/fixed
regardless of include_irs), so hedged costs nothing extra and keeps this
module's results comparable to Section 9B's baseline/BS-only/joint rows,
which are all hedged.

Scenario-count trap: cr.rate_scenario_ids has 7 entries (includes "own"), but
swap_ladder.SHOCKED_SCENS and every irrbb_mc500_*.xlsx cache have only 6
("own" has no closed-form NS+shock formula) -- SCEN_ORDER below fixes the
6-scenario set and order used throughout this module; the 7-row BS gradient
is filtered down to these 6 BY NAME, never by position.

Row-order trap: bank_reprice_at_weights.py uses ProcessPoolExecutor +
as_completed(), so irrbb_mc500_*.xlsx rows are in COMPLETION order, not
curve (shift_idx) order -- load_mc500_baseline() pivots by shift_idx rather
than assuming row i = curve i (the same class of silent bug already caught
once this session, there via NOTIONAL_UNIT double-scaling).
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as spa
from scipy.optimize import linprog

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPTPREP = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "python_code"))
for _p in (_HERE, _OPTPREP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bs_vector import BalanceSheetParams, CohortRates
from lcr_fast   import compute_lcr_fast
from nsfr_fast  import compute_nsfr_fast
from rwa_fast   import compute_rwa_fast
from bias_store import load_bias_corrections
from ftp_store  import load_ftp_rates, margin_unit_rate

import swap_ladder as sl
import joint_optimizer as jo
from bs_optimizer import (
    OptimizationConfig, ProductChange,
    _ProductMap, _resolve_irs, _build_bounds, _compute_lp_coefficients, _build_subst_constraints,
)
from curve_scenario_bank import build_mc_scenario_bank
from bank_reprice_at_weights import load_cached
import anchor_sot_exact as ase

NOTIONAL_UNIT = jo.NOTIONAL_UNIT   # 1e6 PLN -- reuse, never re-literal "1e6"

SCEN_ORDER = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]   # fixed 6-scenario set/order,
# matches swap_ladder.SHOCKED_SCENS and every irrbb_mc500_*.xlsx cache's 'scenario' values -- "own"
# excluded (no closed-form NS+shock formula for it), unlike cr.rate_scenario_ids' 7.


def load_mc500_baseline(label: str = "baseline", out_dir: str | None = None):
    """Load irrbb_mc500_{label}.xlsx and pivot into (n_curve, 6) EVE/NII
    matrices, row i = shift_idx i, columns in SCEN_ORDER -- NEVER assume raw
    row order matches curve order (see module docstring's row-order trap).
    Returns (E0_eve, E0_nii), both (n_curve, 6), % T1.
    """
    df = load_cached(label, out_dir=out_dir, filename_prefix="irrbb_mc500")
    if df is None:
        raise FileNotFoundError(f"irrbb_mc500_{label}.xlsx not found -- run precompute_mc500_irrbb.py first")
    n_curve = int(df["shift_idx"].nunique())
    piv_eve = df.pivot_table(index="shift_idx", columns="scenario", values="delta_eve_pct_t1")
    piv_nii = df.pivot_table(index="shift_idx", columns="scenario", values="delta_nii_pct_t1")
    piv_eve = piv_eve.reindex(index=range(n_curve), columns=SCEN_ORDER)
    piv_nii = piv_nii.reindex(index=range(n_curve), columns=SCEN_ORDER)
    if piv_eve.isna().any().any() or piv_nii.isna().any().any():
        raise ValueError(f"irrbb_mc500_{label}.xlsx has a gap in shift_idx/scenario coverage -- cache is incomplete")
    return piv_eve.to_numpy(), piv_nii.to_numpy()


def build_joint_mc_severity_matrices(
    config: OptimizationConfig,
    params: BalanceSheetParams,
    cr: CohortRates,
    factor_draws: np.ndarray,     # (N, 3) -- SAME array used elsewhere in Section 5/9B
    today_beta: np.ndarray,       # (3,) absolute NS betas at the anchor date factor_draws is relative to
    tenors: tuple = sl.TENORS_YEARS,
    e0_label: str = "baseline",
) -> dict:
    """All LP building blocks for optimize_stochastic_irrbb(). Cheap (seconds):
    one _compute_lp_coefficients call (7-scenario gradient at today's curve,
    filtered to the 6-scenario SCEN_ORDER), one cached-xlsx load+pivot (exact
    per-curve baseline), one price_ladder_against_curve_bank call (exact,
    analytic, curve-then-shock -- matching E0's own convention, NOT the
    drift-only price_ladder_vs_mc_curves Section 9B's CVaR objective used).
    """
    pm = _ProductMap(params)
    t1 = (config.tier1_capital if config.tier1_capital > 0
          else float(params.balance_arr[params.is_equity].sum()))
    irs_cohort_mask, irs_products, irs_notional = _resolve_irs(params)
    active_fixed = config.fixed_products | irs_products

    bias_eve, bias_nii, bias_scens = load_bias_corrections()
    shocked_scens_7 = [str(s) for s in cr.rate_scenario_ids if str(s) != "base"]   # 7, includes 'own'

    ftp_rate = load_ftp_rates(params.cohort_id)
    if ftp_rate is None:
        ftp_rate = np.zeros_like(params.nii_unit_rate)
    margin_rate    = margin_unit_rate(params.nii_unit_rate,         ftp_rate, params.bs_side)
    margin_rate_nb = margin_unit_rate(params.nii_unit_rate_new_biz, ftp_rate, params.bs_side)

    # include_irs=config.include_irs (hedged, True) -- numerically a no-op for
    # A_eve/A_nii_delta either way (see module docstring), passed explicitly
    # for clarity/consistency with the hedged E0 basis, not irrbb_optimizer.py's
    # own include_irs=False.
    _, A_eve_7, A_nii_7, A_nsfr, c_rwa, c_el, _, _, _ = _compute_lp_coefficients(
        pm, params, cr, t1, irs_cohort_mask, config.include_irs,
        bias_eve, bias_nii, bias_scens, shocked_scens_7,
        margin_rate, margin_rate_nb, params.fee_unit_rate,
    )

    # ── Trap #1: filter 7 -> 6 scenarios BY NAME, fixed SCEN_ORDER ──────────
    idx7 = {s: k for k, s in enumerate(shocked_scens_7)}
    sel = [idx7[s] for s in SCEN_ORDER]
    A_eve_bs = A_eve_7[sel, :]   # (6, n_prod)
    A_nii_bs = A_nii_7[sel, :]   # (6, n_prod)

    # ── exact per-curve baseline level (trap #2 handled inside the loader) ──
    E0_eve, E0_nii = load_mc500_baseline(e0_label)   # (n_curve, 6) each, % T1

    # ── swap side: EXACT, curve-then-shock convention matching E0 exactly ──
    mc_bank = build_mc_scenario_bank(factor_draws, today_beta, ase.REPORT_DATE)
    ladder_mc = sl.price_ladder_against_curve_bank(tenors=tenors, direction=None, bank=mc_bank)
    assert list(ladder_mc["shift_idx"]) == list(range(len(mc_bank))), \
        "swap ladder curve ordering doesn't match mc_bank row order -- cannot combine by position"

    n_buckets = len(ladder_mc["bucket_ids"])
    A_eve_swap = np.stack([ladder_mc["delta_eve_pln"][s] for s in SCEN_ORDER], axis=1)   # (n_curve, 6, n_buckets)
    A_nii_swap = np.stack([ladder_mc["delta_nii_pln"][s] for s in SCEN_ORDER], axis=1)

    A_subst, b_subst, _ = _build_subst_constraints(pm, params)

    return dict(
        pm=pm, t1=t1, ta=params.total_assets, irs_cohort_mask=irs_cohort_mask,
        active_fixed=active_fixed, A_nsfr=A_nsfr, c_rwa=c_rwa, A_subst=A_subst, b_subst=b_subst,
        A_eve_bs=A_eve_bs, A_nii_bs=A_nii_bs,
        E0_eve=E0_eve, E0_nii=E0_nii,
        A_eve_swap=A_eve_swap, A_nii_swap=A_nii_swap,
        bucket_ids=ladder_mc["bucket_ids"], elapsed_m=ladder_mc["elapsed_m"],
        swap_direction=ladder_mc["direction"], ladder_mc=ladder_mc,
        n_curve=E0_eve.shape[0], n_scen=len(SCEN_ORDER), n_buckets=n_buckets,
    )


@dataclass
class StochasticIRRBBResult:
    success:                    bool
    message:                    str
    elapsed_s:                  float
    tier1_capital:              float
    n_curve:                    int
    n_scen:                     int
    eve_floor_eff:              float
    nii_floor_eff:              float
    eve_weight:                 float
    nii_weight:                 float
    weights_bs_new:             np.ndarray
    product_changes:            list
    swap_notional:              dict
    swap_direction:             float
    ladder:                     dict
    severity_eve_approx_old:    np.ndarray   # (n_curve, n_scen), at baseline (x_swap=0)
    severity_nii_approx_old:    np.ndarray
    severity_eve_approx_new:    np.ndarray   # (n_curve, n_scen), at solution
    severity_nii_approx_new:    np.ndarray
    severity_total_approx_old:  float
    severity_total_approx_new:  float
    breach_curve_eve_approx_old: int
    breach_curve_eve_approx_new: int
    breach_curve_nii_approx_old: int
    breach_curve_nii_approx_new: int
    lcr:                        dict
    nsfr:                       dict
    rwa:                        float

    def print_summary(self, top_n_buckets: int = 10) -> None:
        status = "SUCCESS" if self.success else "FAILED"
        print(f"\nStochastic IRRBB hedge optimization (breach-severity minimization) - {status}")
        print(f"  {self.n_curve} MC curves x {self.n_scen} official shocks = "
              f"{self.n_curve*self.n_scen} combos   elapsed={self.elapsed_s:.2f}s")
        print(f"  {self.message}")
        print(f"  Approx. total severity (pp T1): {self.severity_total_approx_old:.1f} -> "
              f"{self.severity_total_approx_new:.1f}")
        print(f"  Approx. EVE breaches (curves, out of {self.n_curve}): "
              f"{self.breach_curve_eve_approx_old} -> {self.breach_curve_eve_approx_new}")
        print(f"  Approx. NII breaches (curves, out of {self.n_curve}): "
              f"{self.breach_curve_nii_approx_old} -> {self.breach_curve_nii_approx_new}")

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


_BUCKET_TENOR_RE = re.compile(r"^LADDER_(\d+)Y_E(\d+)$")


def _tenor_grouping_matrix(bucket_ids: list, tenors: tuple = sl.TENORS_YEARS):
    """(n_buckets, n_tenor) matrix G where G[b, t] = 1/n_buckets_in_tenor_t if
    bucket b belongs to tenor t, else 0. "1 PLN of tenor-t notional" then means
    "1 PLN spread evenly across every one of tenor t's own monthly buckets" --
    see optimize_stochastic_irrbb_laddered.
    """
    tenor_of_bucket = np.array([int(_BUCKET_TENOR_RE.match(b).group(1)) for b in bucket_ids])
    n_buckets = len(bucket_ids)
    n_tenor = len(tenors)
    G = np.zeros((n_buckets, n_tenor))
    for ti, T in enumerate(tenors):
        mask = tenor_of_bucket == T
        n_T = int(mask.sum())
        if n_T > 0:
            G[mask, ti] = 1.0 / n_T
    return G, tenor_of_bucket


def optimize_stochastic_irrbb_laddered(
    config: OptimizationConfig,
    params: BalanceSheetParams,
    cr: CohortRates,
    lp_blocks: dict,
    fixed_bs_weights: np.ndarray,
    eve_weight: float = 1.0,
    nii_weight: float = 1.0,
    total_notional_cap_pct_ta: float | None = 0.05,
    per_tenor_cap_pct_ta: float | None = None,
) -> StochasticIRRBBResult:
    """Like optimize_stochastic_irrbb, but for the fixed_bs_weights case ONLY,
    and with a realistic-ladder-shaped decision instead of free bucket choice.

    optimize_stochastic_irrbb's free per-bucket LP (300 variables) naturally
    concentrates into whichever handful of buckets the cap allows -- a real LP
    vertex-solution tendency, not a realistic bank hedging program (a desk
    ladders a hedge with small, frequent transactions, not 1-5 lump sums).
    Adding a genuine "spread evenly" CONSTRAINT to that 300-variable joint
    BS+swap problem would be expensive (needs binary/indicator variables per
    bucket -- a mixed-integer reformulation of an already-large LP). But when
    the BS side is FIXED (this function requires fixed_bs_weights), the
    remaining swap-only problem is tiny, so a cleaner and CHEAPER trick works:
    optimize ONE decision variable PER TENOR (5, not 300) representing
    "notional spread evenly across every one of that tenor's own monthly
    vintage buckets," via a (n_buckets, n_tenor) averaging matrix G. This is
    mathematically exact for the even-spread assumption (not an approximation
    of it) -- G's own 1/n_T scaling means "x_tenor[T] PLN in tenor T" maps
    exactly onto "x_tenor[T]/n_T PLN in each of tenor T's n_T buckets," and the
    resulting per-bucket swap_notional dict is expanded via G before being
    returned, so it plugs into the exact-reprice pipeline
    (precompute_swap_on_stochastic_mc500.py) exactly like the free-LP result.

    total_notional_cap_pct_ta / per_tenor_cap_pct_ta: caps expressed as a
    fraction of params.total_assets rather than an absolute PLN figure, so
    they stay meaningful if the balance sheet's scale changes. per_tenor_cap
    defaults to None (uncapped) -- concentration risk is already limited by
    spreading each tenor across dozens of buckets, so an extra per-tenor cap
    is optional, not load-bearing the way the per-bucket cap was in the free
    LP.

    NSFR / RWA cap / substitution constraints are NOT re-added here (unlike
    optimize_stochastic_irrbb): they're x_bs-only, and x_bs is pinned to
    fixed_bs_weights, which came from an already-feasible solve -- re-checking
    them against a fixed, already-valid vector would be a no-op.
    """
    t_start = time.time()
    pm = lp_blocks["pm"]; t1 = lp_blocks["t1"]; ta = lp_blocks["ta"]
    n_prod = pm.n_prod
    n_curve = lp_blocks["n_curve"]; n_scen = lp_blocks["n_scen"]
    n_combo = n_curve * n_scen
    tenors = sl.TENORS_YEARS
    n_tenor = len(tenors)

    G, _tenor_of_bucket = _tenor_grouping_matrix(lp_blocks["bucket_ids"], tenors)

    bounds_bs = [(float(w), float(w)) for w in fixed_bs_weights]
    eve_floor_eff = config.sot_eve_floor - config.sot_eve_buffer
    nii_floor_eff = config.sot_nii_floor - config.sot_nii_buffer

    total_notional_cap = None if total_notional_cap_pct_ta is None else total_notional_cap_pct_ta * ta
    per_tenor_cap = None if per_tenor_cap_pct_ta is None else per_tenor_cap_pct_ta * ta
    cap_u = None if per_tenor_cap is None else per_tenor_cap / NOTIONAL_UNIT
    bounds_tenor = [(0.0, cap_u)] * n_tenor

    hard_rows, hard_rhs = [], []
    if total_notional_cap is not None:
        hard_rows.append(np.concatenate([np.zeros(n_prod), np.ones(n_tenor)]))
        hard_rhs.append(total_notional_cap / NOTIONAL_UNIT)

    A_hard = np.vstack(hard_rows) if hard_rows else np.zeros((0, n_prod + n_tenor))
    b_hard = np.array(hard_rhs, dtype=float)
    n_hard = A_hard.shape[0]

    A_eq_bs = np.zeros((2, n_prod))
    A_eq_bs[0, pm.asset_mask] = 1.0
    A_eq_bs[1, pm.fund_mask] = 1.0
    b_eq = np.array([pm.asset_sum, pm.fund_sum])

    base_dot_eve = lp_blocks["A_eve_bs"] @ pm.base_prod_w
    base_dot_nii = lp_blocks["A_nii_bs"] @ pm.base_prod_w
    rhs_eve = (lp_blocks["E0_eve"] - eve_floor_eff - base_dot_eve[None, :]).ravel()
    rhs_nii = (lp_blocks["E0_nii"] - nii_floor_eff - base_dot_nii[None, :]).ravel()

    A_eve_bs_tiled = np.tile(-lp_blocks["A_eve_bs"], (n_curve, 1))
    A_nii_bs_tiled = np.tile(-lp_blocks["A_nii_bs"], (n_curve, 1))

    # ── tenor-averaged swap sensitivity: (n_curve,n_scen,n_buckets) @ G ─────
    A_eve_swap_tenor = np.einsum("csn,nt->cst", lp_blocks["A_eve_swap"], G)   # (n_curve, n_scen, n_tenor)
    A_nii_swap_tenor = np.einsum("csn,nt->cst", lp_blocks["A_nii_swap"], G)
    A_eve_swap_flat = (-A_eve_swap_tenor / t1 * 100.0 * NOTIONAL_UNIT).reshape(n_combo, n_tenor)
    A_nii_swap_flat = (-A_nii_swap_tenor / t1 * 100.0 * NOTIONAL_UNIT).reshape(n_combo, n_tenor)

    eye_combo = spa.eye(n_combo, format="csr")
    zero_combo = spa.csr_matrix((n_combo, n_combo))

    A_eve_severity = spa.hstack([
        spa.csr_matrix(A_eve_bs_tiled), spa.csr_matrix(A_eve_swap_flat), -eye_combo, zero_combo,
    ], format="csr")
    A_nii_severity = spa.hstack([
        spa.csr_matrix(A_nii_bs_tiled), spa.csr_matrix(A_nii_swap_flat), zero_combo, -eye_combo,
    ], format="csr")

    A_hard_aug = spa.hstack([spa.csr_matrix(A_hard), spa.csr_matrix((n_hard, 2 * n_combo))])
    A_eq_aug = spa.hstack([spa.csr_matrix(A_eq_bs), spa.csr_matrix((2, n_tenor + 2 * n_combo))])

    A_ub = spa.vstack([A_hard_aug, A_eve_severity, A_nii_severity], format="csr")
    b_ub = np.concatenate([b_hard, rhs_eve, rhs_nii])

    c_lp = np.concatenate([
        np.zeros(n_prod), np.zeros(n_tenor),
        np.full(n_combo, eve_weight), np.full(n_combo, nii_weight),
    ])
    bounds = bounds_bs + bounds_tenor + [(0.0, None)] * (2 * n_combo)

    print(f"  Solving IRRBB-severity LP (laddered, tenor-aggregated): {n_prod} BS weights (fixed) + "
          f"{n_tenor} tenor totals (each spread evenly across its own monthly buckets, "
          f"{len(lp_blocks['bucket_ids'])} total) + {2*n_combo} breach-slack variable(s) "
          f"({n_curve} curves x {n_scen} scenarios), no EP term...")
    lp_res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_aug, b_eq=b_eq, bounds=bounds, method="highs")
    elapsed = time.time() - t_start

    if lp_res.success:
        x_bs = lp_res.x[:n_prod]
        x_tenor = lp_res.x[n_prod:n_prod + n_tenor] * NOTIONAL_UNIT   # (n_tenor,) PLN
    else:
        print(f"  LP FAILED: {lp_res.message} -- returning baseline (zero new swap)")
        x_bs = np.asarray(fixed_bs_weights, dtype=float)
        x_tenor = np.zeros(n_tenor)

    # ── expand tenor totals back to per-bucket notional (G's 1/n_T scaling
    #    is exactly the even split) ──────────────────────────────────────────
    x_swap = G @ x_tenor   # (n_buckets,) PLN

    def _severity_grid(E0, A_bs, A_swap_buckets, floor, x_bs_, x_swap_pln):
        delta = (E0 + (x_bs_ - pm.base_prod_w) @ A_bs.T
                 + np.einsum("csn,n->cs", A_swap_buckets, x_swap_pln) / t1 * 100.0)
        return np.maximum(0.0, floor - delta)

    sev_eve_new = _severity_grid(lp_blocks["E0_eve"], lp_blocks["A_eve_bs"], lp_blocks["A_eve_swap"],
                                  eve_floor_eff, x_bs, x_swap)
    sev_nii_new = _severity_grid(lp_blocks["E0_nii"], lp_blocks["A_nii_bs"], lp_blocks["A_nii_swap"],
                                  nii_floor_eff, x_bs, x_swap)
    sev_eve_old = np.maximum(0.0, eve_floor_eff - lp_blocks["E0_eve"])
    sev_nii_old = np.maximum(0.0, nii_floor_eff - lp_blocks["E0_nii"])

    def _breach_curves(sev_grid):
        return int((sev_grid > 1e-9).any(axis=1).sum())

    amounts_bs = pm.to_amounts(x_bs, ta)
    lcr_dict = compute_lcr_fast(amounts_bs, params)
    nsfr_dict = compute_nsfr_fast(amounts_bs, params)
    rwa_val = compute_rwa_fast(amounts_bs, params)

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

    return StochasticIRRBBResult(
        success=lp_res.success, message=lp_res.message, elapsed_s=elapsed,
        tier1_capital=t1, n_curve=n_curve, n_scen=n_scen,
        eve_floor_eff=eve_floor_eff, nii_floor_eff=nii_floor_eff,
        eve_weight=eve_weight, nii_weight=nii_weight,
        weights_bs_new=x_bs, product_changes=changes,
        swap_notional=swap_notional, swap_direction=lp_blocks["swap_direction"], ladder=lp_blocks["ladder_mc"],
        severity_eve_approx_old=sev_eve_old, severity_nii_approx_old=sev_nii_old,
        severity_eve_approx_new=sev_eve_new, severity_nii_approx_new=sev_nii_new,
        severity_total_approx_old=float(sev_eve_old.sum() + sev_nii_old.sum()),
        severity_total_approx_new=float(sev_eve_new.sum() + sev_nii_new.sum()),
        breach_curve_eve_approx_old=_breach_curves(sev_eve_old),
        breach_curve_eve_approx_new=_breach_curves(sev_eve_new),
        breach_curve_nii_approx_old=_breach_curves(sev_nii_old),
        breach_curve_nii_approx_new=_breach_curves(sev_nii_new),
        lcr=dict(lcr_dict), nsfr=dict(nsfr_dict), rwa=rwa_val,
    )


def optimize_stochastic_irrbb(
    config: OptimizationConfig,
    params: BalanceSheetParams,
    cr: CohortRates,
    lp_blocks: dict,
    eve_weight: float = 1.0,
    nii_weight: float = 1.0,
    notional_cap: float | None = 1e9,
    total_notional_cap: float | None = 5e9,
    seasoned_elapsed_months: int = 12,
    seasoned_notional_cap: float | None = 1e9,
    fixed_bs_weights: np.ndarray | None = None,
) -> StochasticIRRBBResult:
    """Variable layout (fixed order) -- NO growth-cost epigraph (no EP term
    anywhere in this objective, matching irrbb_optimizer.py exactly):

        z = [x_bs (n_prod), x_swap (n_buckets), u_eve (n_curve*n_scen), u_nii (n_curve*n_scen)]

    Combo (c,s) flat-indexes as c*n_scen+s (row-major), matching E0's own
    .ravel() layout exactly -- u_eve.reshape(n_curve, n_scen) recovers the
    per-(curve,scenario) grid with no index-order ambiguity.

    fixed_bs_weights: if supplied, PINS x_bs exactly to this vector
    (ignoring config.mode entirely) and solves ONLY for x_swap -- "given this
    already-chosen balance sheet (e.g. Section 5's BS-only stochastic
    optimum), what swap overlay best reduces breach severity on top of it,"
    rather than re-optimizing the BS side jointly. notional_cap should
    usually be set much smaller than the joint-optimization default in this
    mode (see module docstring) -- a large per-bucket cap lets the LP
    concentrate the whole hedge into 1-2 buckets (a real LP vertex-solution
    tendency, not unique to this cap), which is not how a real desk ladders
    a hedge across tenors/vintages; a smaller cap forces the solution to
    spread across many buckets to reach the same total notional.
    """
    t_start = time.time()
    pm = lp_blocks["pm"]; t1 = lp_blocks["t1"]; ta = lp_blocks["ta"]
    n_prod = pm.n_prod
    n_buckets = lp_blocks["n_buckets"]
    n_curve = lp_blocks["n_curve"]; n_scen = lp_blocks["n_scen"]
    n_combo = n_curve * n_scen

    if fixed_bs_weights is not None:
        bounds_bs = [(float(w), float(w)) for w in fixed_bs_weights]
    else:
        lb, ub = _build_bounds(config, pm, lp_blocks["active_fixed"])
        bounds_bs = list(zip(lb.tolist(), ub.tolist()))
    eve_floor_eff = config.sot_eve_floor - config.sot_eve_buffer
    nii_floor_eff = config.sot_nii_floor - config.sot_nii_buffer

    # ── Hard rows: NSFR + RWA cap + substitution -- x_bs only, swap columns
    #    zero-padded (swap doesn't touch RWA/NSFR/subst, same scope
    #    limitation joint_optimizer.py already documents) ────────────────────
    hard_rows, hard_rhs = [], []
    hard_rows.append(np.concatenate([lp_blocks["A_nsfr"], np.zeros(n_buckets)]))
    hard_rhs.append(0.0)
    if config.min_t1_rwa > 0.0:
        hard_rows.append(np.concatenate([lp_blocks["c_rwa"], np.zeros(n_buckets)]))
        hard_rhs.append(t1 / config.min_t1_rwa)
    if lp_blocks["A_subst"] is not None:
        for row, rhs in zip(lp_blocks["A_subst"], lp_blocks["b_subst"]):
            hard_rows.append(np.concatenate([row, np.zeros(n_buckets)]))
            hard_rhs.append(rhs)

    cap_u = None if notional_cap is None else notional_cap / NOTIONAL_UNIT
    bounds_swap = [(0.0, cap_u)] * n_buckets
    seasoned_mask = lp_blocks["elapsed_m"] > seasoned_elapsed_months
    if total_notional_cap is not None:
        hard_rows.append(np.concatenate([np.zeros(n_prod), np.ones(n_buckets)]))
        hard_rhs.append(total_notional_cap / NOTIONAL_UNIT)
    if seasoned_notional_cap is not None and seasoned_mask.any():
        hard_rows.append(np.concatenate([np.zeros(n_prod), seasoned_mask.astype(float)]))
        hard_rhs.append(seasoned_notional_cap / NOTIONAL_UNIT)

    A_hard = np.vstack(hard_rows)
    b_hard = np.array(hard_rhs, dtype=float)
    n_hard = A_hard.shape[0]

    A_eq_bs = np.zeros((2, n_prod))
    A_eq_bs[0, pm.asset_mask] = 1.0
    A_eq_bs[1, pm.fund_mask] = 1.0
    b_eq = np.array([pm.asset_sum, pm.fund_sum])

    # ── Severity epigraph rows, SPARSE (3000+3000 slack vars -- dense would
    #    materialize two (3000,3000) identity blocks, wasteful at this scale,
    #    60-400x every other LP in this codebase). ──────────────────────────
    base_dot_eve = lp_blocks["A_eve_bs"] @ pm.base_prod_w   # (n_scen,)
    base_dot_nii = lp_blocks["A_nii_bs"] @ pm.base_prod_w   # (n_scen,)

    # rhs[c,s] = E0[c,s] - floor - A_eve_bs[s,:]@base_prod_w  (see docstring algebra)
    rhs_eve = (lp_blocks["E0_eve"] - eve_floor_eff - base_dot_eve[None, :]).ravel()   # (n_combo,)
    rhs_nii = (lp_blocks["E0_nii"] - nii_floor_eff - base_dot_nii[None, :]).ravel()

    A_eve_bs_tiled = np.tile(-lp_blocks["A_eve_bs"], (n_curve, 1))   # (n_combo, n_prod)
    A_nii_bs_tiled = np.tile(-lp_blocks["A_nii_bs"], (n_curve, 1))

    A_eve_swap_flat = (-lp_blocks["A_eve_swap"] / t1 * 100.0 * NOTIONAL_UNIT).reshape(n_combo, n_buckets)
    A_nii_swap_flat = (-lp_blocks["A_nii_swap"] / t1 * 100.0 * NOTIONAL_UNIT).reshape(n_combo, n_buckets)

    eye_combo = spa.eye(n_combo, format="csr")
    zero_combo = spa.csr_matrix((n_combo, n_combo))

    A_eve_severity = spa.hstack([
        spa.csr_matrix(A_eve_bs_tiled), spa.csr_matrix(A_eve_swap_flat),
        -eye_combo, zero_combo,
    ], format="csr")
    A_nii_severity = spa.hstack([
        spa.csr_matrix(A_nii_bs_tiled), spa.csr_matrix(A_nii_swap_flat),
        zero_combo, -eye_combo,
    ], format="csr")

    A_hard_aug = spa.hstack([spa.csr_matrix(A_hard), spa.csr_matrix((n_hard, 2 * n_combo))])
    A_eq_aug = spa.hstack([spa.csr_matrix(A_eq_bs), spa.csr_matrix((2, n_buckets + 2 * n_combo))])

    A_ub = spa.vstack([A_hard_aug, A_eve_severity, A_nii_severity], format="csr")
    b_ub = np.concatenate([b_hard, rhs_eve, rhs_nii])

    c_lp = np.concatenate([
        np.zeros(n_prod), np.zeros(n_buckets),
        np.full(n_combo, eve_weight), np.full(n_combo, nii_weight),
    ])
    bounds = bounds_bs + bounds_swap + [(0.0, None)] * (2 * n_combo)

    print(f"  Solving IRRBB-severity LP: {n_prod} BS weights + {n_buckets} swap buckets + "
          f"{2*n_combo} breach-slack variable(s) ({n_curve} curves x {n_scen} scenarios), no EP term...")
    lp_res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_aug, b_eq=b_eq,
                      bounds=bounds, method="highs")
    elapsed = time.time() - t_start

    if lp_res.success:
        x_bs = lp_res.x[:n_prod]
        x_swap = lp_res.x[n_prod:n_prod + n_buckets] * NOTIONAL_UNIT
    else:
        print(f"  LP FAILED: {lp_res.message} -- returning baseline (zero new swap)")
        x_bs = pm.base_prod_w.copy()
        x_swap = np.zeros(n_buckets)

    # ── Recompute the (n_curve, n_scen) severity grids from the affine
    #    formula AT THE SOLUTION (not from lp_res.x's slack values directly,
    #    which HiGHS may report slightly loose-but-feasible) ────────────────
    def _severity_grid(E0, A_bs, A_swap, floor, x_bs_, x_swap_pln):
        delta = (E0 + (x_bs_ - pm.base_prod_w) @ A_bs.T
                 + np.einsum("csn,n->cs", A_swap, x_swap_pln) / t1 * 100.0)
        return np.maximum(0.0, floor - delta)

    sev_eve_new = _severity_grid(lp_blocks["E0_eve"], lp_blocks["A_eve_bs"], lp_blocks["A_eve_swap"],
                                  eve_floor_eff, x_bs, x_swap)
    sev_nii_new = _severity_grid(lp_blocks["E0_nii"], lp_blocks["A_nii_bs"], lp_blocks["A_nii_swap"],
                                  nii_floor_eff, x_bs, x_swap)
    sev_eve_old = np.maximum(0.0, eve_floor_eff - lp_blocks["E0_eve"])
    sev_nii_old = np.maximum(0.0, nii_floor_eff - lp_blocks["E0_nii"])

    def _breach_curves(sev_grid):
        return int((sev_grid > 1e-9).any(axis=1).sum())

    amounts_bs = pm.to_amounts(x_bs, ta)
    lcr_dict = compute_lcr_fast(amounts_bs, params)
    nsfr_dict = compute_nsfr_fast(amounts_bs, params)
    rwa_val = compute_rwa_fast(amounts_bs, params)

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

    return StochasticIRRBBResult(
        success=lp_res.success, message=lp_res.message, elapsed_s=elapsed,
        tier1_capital=t1, n_curve=n_curve, n_scen=n_scen,
        eve_floor_eff=eve_floor_eff, nii_floor_eff=nii_floor_eff,
        eve_weight=eve_weight, nii_weight=nii_weight,
        weights_bs_new=x_bs, product_changes=changes,
        swap_notional=swap_notional, swap_direction=lp_blocks["swap_direction"], ladder=lp_blocks["ladder_mc"],
        severity_eve_approx_old=sev_eve_old, severity_nii_approx_old=sev_nii_old,
        severity_eve_approx_new=sev_eve_new, severity_nii_approx_new=sev_nii_new,
        severity_total_approx_old=float(sev_eve_old.sum() + sev_nii_old.sum()),
        severity_total_approx_new=float(sev_eve_new.sum() + sev_nii_new.sum()),
        breach_curve_eve_approx_old=_breach_curves(sev_eve_old),
        breach_curve_eve_approx_new=_breach_curves(sev_eve_new),
        breach_curve_nii_approx_old=_breach_curves(sev_nii_old),
        breach_curve_nii_approx_new=_breach_curves(sev_nii_new),
        lcr=dict(lcr_dict), nsfr=dict(nsfr_dict), rwa=rwa_val,
    )
