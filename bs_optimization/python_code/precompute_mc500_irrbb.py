"""precompute_mc500_irrbb.py
===========================
Exact (Method B, full CF-based) EVE/NII repricing of Section 5's 500 Monte
Carlo curve scenarios, for both the baseline book and the stochastic-
optimized weights -- answers "in how many of the 500 MC scenarios does
EVE/NII SOT actually breach" using the SAME production-grade pricing engine
anchor_sot_exact.py / bank_reprice_at_weights.py already use for the 165-curve
stress test, just repricing a different curve set.

Rebuilds Section 5's inputs bit-for-bit (same seed=42, same N_SCENARIOS=500,
same mean-reverting 24-month horizon) so the curves repriced here are EXACTLY
the ones the notebook's stochastic optimizer already used for its E[EP]/CVaR
objective -- not a fresh/different random draw.

The 500 relative NS-factor draws are converted to absolute curves anchored
at anchor_eve_exact.REPORT_DATE (2024-12-31) -- NOT the freshest available
historical fit (which runs months later) -- via
curve_scenario_bank.build_mc_scenario_bank(), then repriced with
bank_reprice_at_weights.run_and_cache()'s new `bank=`/`filename_prefix=`
parameters.

Meant to be run once as a background step (like the existing 165-curve
cache) -- NOT inline in the notebook. Expected runtime: ~8 min/book with the
validated `process` backend (500 curves vs. 165 -> ~3x the existing 165-curve
test's ~2.5 min/book parallel time), ~16 min total for both books.

Usage
-----
    python precompute_mc500_irrbb.py
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
    from bs_optimizer import _ProductMap
    from optimizer_io import load_optimizer_config
    import ns_curve_model as nsm
    from stochastic_optimizer import StochasticConfig, build_scenario_ep_matrix, optimize_stochastic
    from curve_scenario_bank import build_mc_scenario_bank
    from bank_reprice_at_weights import run_and_cache
    import anchor_sot_exact as ase

    _PARAMS = os.path.join(_ROOT, "optimize_prep", "output", "product_params.npz")
    _XL     = os.path.join(_ROOT, "bs_optimization", "input", "optimizer_config.xlsx")

    print("Loading params/config...")
    params = BalanceSheetParams.load(_PARAMS)
    cr     = CohortRates.load(_PARAMS)
    cfg    = load_optimizer_config(_XL, params)
    pm     = _ProductMap(params)

    print("Rebuilding Section 5's NS calibration + PCA + mean-reverting Monte Carlo "
          "draws (seed=42, N=500, 24m horizon) -- must match the notebook exactly...")
    panel  = nsm.load_historical_panel()
    tau    = nsm.DIEBOLD_LI_TAU
    ns_ts  = nsm.fit_ns_timeseries(panel, tau)
    pca    = nsm.factor_changes_pca(ns_ts)

    # today_beta MUST be anchored at the exact CF engine's own report date
    # (anchor_eve_exact.REPORT_DATE = 2024-12-31), not ns_ts's freshest row
    # (which runs through 2025-09) -- see build_mc_scenario_bank()'s docstring.
    # Also the anchor point the mean-reverting simulation starts every path from.
    today_row  = ns_ts.asof(ase.REPORT_DATE)
    today_beta = today_row[["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
    print(f"  Anchoring MC draws at {ase.REPORT_DATE.date()} "
          f"(beta0={today_beta[0]:.1f}, beta1={today_beta[1]:.1f}, beta2={today_beta[2]:.1f})")

    N_SCENARIOS    = 500
    HORIZON_MONTHS = 24   # 2-year horizon -- long enough for mean-reversion to matter
    MC_SEED        = 42
    mr = nsm.fit_mean_reversion(ns_ts)
    factor_draws = nsm.simulate_mean_reverting_shocks(pca, mr, today_beta, n_scenarios=N_SCENARIOS,
                                                        horizon_months=HORIZON_MONTHS, seed=MC_SEED)

    print("Re-solving the stochastic optimizer to get the SAME optimized "
          "weights the notebook displays (cheap LP solve, not the expensive part)...")
    C_scenarios, lp_blocks = build_scenario_ep_matrix(cfg, StochasticConfig(), params, cr, factor_draws, tau)
    stoch_cfg = StochasticConfig(n_scenarios=N_SCENARIOS, horizon_months=HORIZON_MONTHS,
                                  cvar_alpha=0.95, risk_lambda=1.0, seed=MC_SEED)
    stoch_result = optimize_stochastic(cfg, stoch_cfg, params, cr, C_scenarios, lp_blocks)
    print(f"  Stochastic-optimized E[EP]: {stoch_result.ep_mean_old/1e6:,.1f}M -> "
          f"{stoch_result.ep_mean_new/1e6:,.1f}M")

    mc_bank = build_mc_scenario_bank(factor_draws, today_beta, ase.REPORT_DATE)
    print(f"  MC curve bank: {len(mc_bank)} curves")

    runs = [
        ("baseline",   pm.base_prod_w),
        ("stochastic", stoch_result.weights_new),
    ]
    for label, target_weights in runs:
        print(f"\n{'='*70}\nRepricing MC500 bank for '{label}' weights...")
        t0 = time.time()
        run_and_cache(
            label=label,
            products=pm.products,
            baseline_weights=pm.base_prod_w,
            target_weights=target_weights,
            include_irs=cfg.include_irs,
            bank=mc_bank,
            filename_prefix="irrbb_mc500",
        )
        print(f"'{label}' done in {time.time()-t0:.0f}s")

    print("\nAll done.")


if __name__ == "__main__":
    main()
