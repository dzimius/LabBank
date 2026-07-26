"""precompute_165_irrbb.py
=========================
Exact (Method B, full CF-based) EVE/NII repricing of the 165-curve anchor+
shift scenario bank, for both the baseline book and Section 4's single-period
EP-optimized weights -- regenerates irrbb_165_baseline.xlsx /
irrbb_165_ep_optimized.xlsx (loaded by the notebook's Section 4 breach
counter via bank_reprice_at_weights.load_cached("baseline"/"ep_optimized")).

Re-run whenever anything upstream of anchor_sot_exact.reprice_anchor_both
changes (e.g. eba_shock_curves.py's floor logic) -- these two caches use the
SAME reprice_anchor_both() as every other exact-reprice cache in this
notebook (Section 5/9B/9C/9D's MC500 caches), so they must be kept in sync
with those or the notebook shows numbers computed under different engine
versions side by side.

Usage
-----
    python precompute_165_irrbb.py
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PREP_PY = os.path.join(_ROOT, "optimize_prep", "python_code")
for _p in (_HERE, _PREP_PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    from bs_vector import BalanceSheetParams, CohortRates
    from bs_optimizer import _ProductMap, optimize_nii
    from optimizer_io import load_optimizer_config
    from bank_reprice_at_weights import run_and_cache

    _PARAMS = os.path.join(_ROOT, "optimize_prep", "output", "product_params.npz")
    _XL     = os.path.join(_ROOT, "bs_optimization", "input", "optimizer_config.xlsx")

    print("Loading params/config...")
    params = BalanceSheetParams.load(_PARAMS)
    cr     = CohortRates.load(_PARAMS)
    cfg    = load_optimizer_config(_XL, params)
    pm     = _ProductMap(params)

    print("Re-solving Section 4's single-period EP optimizer to get the SAME "
          "optimized weights the notebook displays (cheap LP solve, not the expensive part)...")
    result = optimize_nii(cfg, params, cr)
    print(f"  EP: {result.ep_old/1e6:,.1f}M -> {result.ep_new/1e6:,.1f}M")

    # result.weights_new is COHORT-level (bs_optimizer.OptimizationResult docstring);
    # run_and_cache needs PRODUCT-level weights matching pm.products' order --
    # product_changes is built via the SAME enumerate(pm.products) loop pm.base_prod_w
    # itself uses, so this ordering is safe.
    ep_optimized_weights = np.array([c.weight_new for c in result.product_changes])

    runs = [
        ("baseline",     pm.base_prod_w),
        ("ep_optimized", ep_optimized_weights),
    ]
    for label, target_weights in runs:
        print(f"\n{'='*70}\nRepricing 165-curve bank for '{label}' weights...")
        t0 = time.time()
        run_and_cache(
            label=label,
            products=pm.products,
            baseline_weights=pm.base_prod_w,
            target_weights=target_weights,
            include_irs=cfg.include_irs,
        )
        print(f"'{label}' done in {time.time()-t0:.0f}s")

    print("\nAll done.")


if __name__ == "__main__":
    main()
