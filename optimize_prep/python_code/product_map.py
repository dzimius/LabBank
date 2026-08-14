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


class _ProductMap:
    """Precomputed product <-> cohort index mappings for the optimizer hot path.

    Optimizer variables x[j] are product-level weights (fraction of total_assets).
    Cohort amounts are derived by proportional scaling:
        amounts[i] = fraction[i] * x[product_of_i] * total_assets
    where fraction[i] = baseline_cohort_weight[i] / baseline_product_weight[product_of_i].
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
        fraction = np.where(prod_w_per_cohort > 0.0, base_w / prod_w_per_cohort, 0.0)

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
        self.fraction       = fraction
        self.base_prod_w    = base_prod_w
        self.base_cohort_w  = base_w
        self.vol_elast_prod = vol_elast_prod
        self.acq_cost_prod  = acq_cost_prod

        self.asset_mask  = np.array([s == "A"         for _, s in products])
        self.fund_mask   = np.array([s in ("L", "E")  for _, s in products])
        self.equity_mask = np.array([s == "E"         for _, s in products])

        self.asset_sum = float(base_prod_w[self.asset_mask].sum())
        self.fund_sum  = float(base_prod_w[self.fund_mask].sum())

    def to_amounts(self, x: np.ndarray, total_assets: float) -> np.ndarray:
        return self.fraction * x[self.cohort_to_prod] * total_assets
