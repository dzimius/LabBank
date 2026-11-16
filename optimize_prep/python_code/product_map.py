"""product_map.py
=================
Product<->cohort index mapping (_ProductMap) and the flat OpEx rate
(OPEX_RATE), shared by every bs_optimization solver and by the sandbox
Streamlit app's Finance Metrics tab (via ep_fast.py).

Moved here from bs_optimization/python_code/bs_optimizer.py (2026-08-14) so
sandbox/ never has to import from bs_optimization/ -- _ProductMap only
depends on BalanceSheetParams (this package) and numpy, no solver-specific
state, so it belongs at this shared layer rather than being optimizer-only.
bs_optimizer.py re-imports it from here so every existing
`from bs_optimizer import _ProductMap, OPEX_RATE` call site elsewhere in
bs_optimization/ keeps working unchanged.
"""
from __future__ import annotations

import numpy as np

from bs_vector import BalanceSheetParams

OPEX_RATE = 0.0065

# Derivatives (IRS book '0000', FX swaps '0100') carry no balance-volume operating
# cost: their notionals are not balance-sheet volume, and both legs are booked as
# separate A/L rows, so charging OPEX_RATE on them double-counted ~25M EUR/yr.
DERIVATIVE_CODES = frozenset({"0000", "0100"})


# Marketable securities are sold pro rata (every holding keeps its share), as the roll-forward sells
# them (b_s_rollforward.MARKETABLE): no plan-year-vintage-first rule for them.
MARKETABLE_NAMES = frozenset({"bond_fixed", "bond_float", "t_bill"})


def opex_mask(params: BalanceSheetParams) -> np.ndarray:
    """(n,) cohort mask of rows that bear OPEX_RATE: everything except equity and derivatives."""
    is_deriv = np.array([str(pc) in DERIVATIVE_CODES for pc in params.product_code], dtype=bool)
    return ~params.is_equity & ~is_deriv


def plan_year_start(report_date) -> "pd.Timestamp | None":
    """Report date of the base bank when `report_date` is that of a rolled-forward bank (eur_y1:
    2026-09-30): contracts started after it are the plan year's new business. None otherwise."""
    import pandas as pd
    try:
        from labbank_common.bank_config import get_bank
        bank = get_bank()
    except Exception:
        return None
    if bank.rollforward is None or pd.Timestamp(str(bank.report_date)) != pd.Timestamp(str(report_date)):
        return None
    return pd.Timestamp(str(get_bank(bank.rollforward.base_bank).report_date))


def plan_year_cohorts(params: BalanceSheetParams) -> np.ndarray:
    """(n,) mask of cohorts originated in the plan year (after plan_year_start) of contractual products
    other than marketable securities and derivatives; all False on a bank that is not rolled forward."""
    start = plan_year_start(params.report_date)
    if start is None:
        return np.zeros(len(params.cohort_id), dtype=bool)
    ym = np.nan_to_num(params.start_year) * 12 + np.nan_to_num(params.start_month)
    return (params.is_cohort & (ym > start.year * 12 + start.month)
            & ~np.isin(params.product_name.astype(str), list(MARKETABLE_NAMES))
            & ~np.isin(params.product_code.astype(str), list(DERIVATIVE_CODES)))


class _ProductMap:
    """Precomputed product <-> cohort index mappings for the optimizer hot path.

    Optimizer variables x[j] are product-level weights (fraction of total_assets).
    Cohort amounts are affine in x:
        amounts[i] = (offset[i] + fraction[i] * x[product_of_i]) * total_assets
    Default: proportional scaling, fraction[i] = baseline_cohort_weight[i] /
    baseline_product_weight[product_of_i], offset 0.
    On a rolled-forward bank a product with both seasoned contracts and plan-year
    new business (plan_year_cohorts) moves only through its plan-year cohorts: the
    contracts that existed before the plan year cannot be undone, so cutting the
    product below the plan means writing less of the plan's new business (at its
    rates and duration), and growing it means more of it. There fraction[i] is the
    cohort's share of the plan-year cohorts, offset holds the seasoned book, and
    min_w (the seasoned weight) is a lower bound on x.

    per_prod(v) = W @ v turns a cohort vector into per-product coefficients of a
    level sum(v * amounts) / total_assets. The offset's contribution is constant; it
    is spread over the asset columns (every solver fixes sum of asset weights =
    asset_sum), so W @ v . x is the exact level on the feasible set.
    """

    def __init__(self, params: BalanceSheetParams) -> None:
        base_w = params.balance_arr / params.total_assets   # (n,) fractions

        # Ordered unique products (preserving first-appearance order in npz)
        seen: dict[tuple[str, str], int] = {}
        products: list[tuple[str, str]] = []
        for pc, side in zip(params.product_code, params.bs_side):
            key = (str(pc), str(side))
            if key not in seen:
                seen[key] = len(products)
                products.append(key)

        n_prod  = len(products)
        prod_idx = {p: j for j, p in enumerate(products)}

        cohort_to_prod = np.array(
            [prod_idx[(str(pc), str(side))]
             for pc, side in zip(params.product_code, params.bs_side)],
            dtype=np.intp,
        )

        base_prod_w = np.zeros(n_prod, dtype=float)
        for i, j in enumerate(cohort_to_prod):
            base_prod_w[j] += float(base_w[i])

        prod_w_per_cohort = base_prod_w[cohort_to_prod]
        # a product with no base balance (catalogue product, bank_data bs_percentage 0) splits
        # its weight evenly over its rows -- it has one, the new-business cohort of extract_params
        n_rows = np.bincount(cohort_to_prod, minlength=n_prod)[cohort_to_prod]
        fraction = np.where(prod_w_per_cohort > 0.0, base_w / np.where(prod_w_per_cohort > 0.0, prod_w_per_cohort, 1.0),
                            1.0 / n_rows)
        base_fraction = fraction.copy()

        # plan-year vintage first (see class docstring)
        vint = plan_year_cohorts(params) & (base_w > 0.0)
        vint_w = np.bincount(cohort_to_prod, weights=np.where(vint, base_w, 0.0), minlength=n_prod)
        seas_w = base_prod_w - vint_w
        split = (vint_w > 0.0) & (seas_w > 1e-12)
        sp = split[cohort_to_prod]
        fraction = np.where(sp, np.where(vint, base_w / np.where(vint_w > 0, vint_w, 1.0)[cohort_to_prod], 0.0),
                            fraction)
        offset = base_w - fraction * base_prod_w[cohort_to_prod]          # 0 outside split products
        min_w = np.where(split, seas_w, 0.0)

        # Price-volume elasticity per product (same value for all cohorts within a product)
        vol_elast_prod = np.zeros(n_prod, dtype=float)
        for i, j in enumerate(cohort_to_prod):
            if vol_elast_prod[j] == 0.0 and abs(float(params.vol_elasticity[i])) > 0.0:
                vol_elast_prod[j] = float(params.vol_elasticity[i])

        # Marketing/acquisition cost rate per product (same value for all cohorts
        # within a product) -- charged only on growth above base_prod_w in the
        # optimizer, see the g_j epigraph variables in optimize_nii()
        acq_cost_prod = np.zeros(n_prod, dtype=float)
        for i, j in enumerate(cohort_to_prod):
            if acq_cost_prod[j] == 0.0 and abs(float(params.acq_cost_rate[i])) > 0.0:
                acq_cost_prod[j] = float(params.acq_cost_rate[i])

        self.products       = products
        self.n_prod         = n_prod
        self.cohort_to_prod = cohort_to_prod
        self.fraction       = fraction           # marginal cohort share of the product weight
        self.base_fraction  = base_fraction      # cohort share of the product at baseline
        self.offset         = offset             # seasoned book of split products (weight units), else 0
        self.split          = split              # (n_prod,) product moves through its plan-year cohorts
        self.min_w          = min_w              # (n_prod,) seasoned weight of split products, else 0
        self.plan_year      = vint               # (n,) plan-year cohorts with balance
        self.base_prod_w    = base_prod_w
        self.base_cohort_w  = base_w
        self.vol_elast_prod = vol_elast_prod
        self.acq_cost_prod  = acq_cost_prod

        self.asset_mask  = np.array([s == "A"         for _, s in products])
        self.fund_mask   = np.array([s in ("L", "E")  for _, s in products])
        self.equity_mask = np.array([s == "E"         for _, s in products])
        self.no_opex_mask = self.equity_mask | np.array([pc in DERIVATIVE_CODES for pc, _ in products])

        self.asset_sum = float(base_prod_w[self.asset_mask].sum())
        self.fund_sum  = float(base_prod_w[self.fund_mask].sum())

        W = np.zeros((n_prod, len(base_w)))
        W[cohort_to_prod, np.arange(len(base_w))] = fraction
        if offset.any():
            W[self.asset_mask] += offset / self.asset_sum
        self.W = W

    def to_amounts(self, x: np.ndarray, total_assets: float) -> np.ndarray:
        return (self.offset + self.fraction * x[self.cohort_to_prod]) * total_assets

    def per_prod(self, v: np.ndarray) -> np.ndarray:
        """Cohort vector -> per-product coefficients: sum(v * to_amounts(x, 1)) = per_prod(v) @ x on the
        feasible set (asset weights summing to asset_sum)."""
        return self.W @ v

    def fold_level(self, A: np.ndarray, level_at_zero) -> np.ndarray:
        """Add the offset's constant contribution (a level evaluated at x = 0, i.e. on the seasoned book
        alone) to coefficients A[..., n_prod] obtained by perturbing x."""
        A = np.array(A, dtype=float, copy=True)
        if self.offset.any():
            A[..., self.asset_mask] += np.asarray(level_at_zero, dtype=float)[..., None] / self.asset_sum
        return A

    def split_scales(self, x: np.ndarray) -> dict:
        """(pc, side) -> (seasoned scale, plan-year scale) of the product's balance at weights x, for
        repricing results kept per product and vintage (by-product caches, pipeline results by contract)."""
        out = {}
        for j, key in enumerate(self.products):
            b = self.base_prod_w[j]
            if self.split[j]:
                v = b - self.min_w[j]
                out[key] = (1.0, (float(x[j]) - self.min_w[j]) / v)
            else:
                s = float(x[j]) / b if b > 1e-12 else (1.0 if x[j] <= 1e-12 else float("nan"))
                out[key] = (s, s)
        return out
