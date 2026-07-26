"""precompute_joint_stochastic_mc500.py
=======================================
Exact (Method B) EVE/NII breach counter for the Section 9B stochastic joint
BS+IRS structure, against the SAME 500 mean-reverting Monte Carlo curves used
throughout Section 5/9B -- answers "how many of the 500 realistic scenarios
still breach EVE/NII SOT once the new swap overlay is in place," directly
comparable to the existing baseline (320/500 NII) and BS-only-stochastic
(129/500 NII) exact breach counts already cached.

Two pieces, combined:
  1. BS side: exact CF-schedule reprice (bank_reprice_at_weights.py, the
     SAME SQL-backed engine used for baseline/stochastic) at the joint
     optimizer's BS weights (joint_stoch_result.weights_bs_new).
  2. Swap side: the new swap ladder overlay's EVE/NII delta, analytic DCF
     (swap_ladder.price_ladder_against_curve_bank, curve-then-6-official-
     shock convention -- same convention the BS-side exact engine uses, NOT
     the drift-only convention Section 9B's own CVaR objective uses), at the
     joint optimizer's chosen swap notional per bucket.

Combined delta_eve_pct_t1 = BS exact delta + swap analytic delta, per
(curve, scenario) -- saved in the SAME row format as the existing
irrbb_mc500_baseline.xlsx / irrbb_mc500_stochastic.xlsx caches, so the
notebook's existing _curve_breach_count() helper works unchanged.

Meant to be run once in the background (like the other MC500 precomputes
this session) -- NOT inline in the notebook. Expected runtime: ~15-25 min
(dominated by the BS-side exact CF reprice; the swap-side analytic pricing
takes seconds).

Usage
-----
    python precompute_joint_stochastic_mc500.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

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
    from stochastic_optimizer import StochasticConfig
    from stochastic_joint_optimizer import build_joint_scenario_matrices, optimize_stochastic_joint
    from curve_scenario_bank import build_mc_scenario_bank
    from bank_reprice_at_weights import run_and_cache
    import swap_ladder as sl
    import anchor_sot_exact as ase

    _PARAMS = os.path.join(_ROOT, "optimize_prep", "output", "product_params.npz")
    _XL     = os.path.join(_ROOT, "bs_optimization", "input", "optimizer_config.xlsx")

    print("Loading params/config...")
    params = BalanceSheetParams.load(_PARAMS)
    cr     = CohortRates.load(_PARAMS)
    cfg    = load_optimizer_config(_XL, params)
    pm     = _ProductMap(params)

    print("Rebuilding Section 5/9B's inputs bit-for-bit (seed=42, N=500, "
          "24m mean-reverting horizon) -- must match the notebook exactly...")
    panel  = nsm.load_historical_panel()
    tau    = nsm.DIEBOLD_LI_TAU
    ns_ts  = nsm.fit_ns_timeseries(panel, tau)
    pca    = nsm.factor_changes_pca(ns_ts)
    today_beta = ns_ts.asof(ase.REPORT_DATE)[["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
    mr = nsm.fit_mean_reversion(ns_ts)

    N_SCENARIOS, HORIZON_MONTHS, MC_SEED = 500, 24, 42
    factor_draws = nsm.simulate_mean_reverting_shocks(pca, mr, today_beta, n_scenarios=N_SCENARIOS,
                                                        horizon_months=HORIZON_MONTHS, seed=MC_SEED)
    stoch_cfg = StochasticConfig(n_scenarios=N_SCENARIOS, horizon_months=HORIZON_MONTHS,
                                  cvar_alpha=0.95, risk_lambda=1.0, seed=MC_SEED)

    print("Re-solving the stochastic joint BS+IRS optimizer to get the SAME "
          "weights/swap notional the notebook displays (cheap LP solve)...")
    C_joint, blocks = build_joint_scenario_matrices(cfg, stoch_cfg, params, cr, factor_draws, today_beta, tau)
    joint_result = optimize_stochastic_joint(cfg, stoch_cfg, params, cr, C_joint, blocks)
    print(f"  E[EP]: {joint_result.ep_mean_old/1e6:,.1f}M -> {joint_result.ep_mean_new/1e6:,.1f}M  |  "
          f"swap notional: {sum(joint_result.swap_notional.values())/1e6:,.1f}M")

    mc_bank = build_mc_scenario_bank(factor_draws, today_beta, ase.REPORT_DATE)
    print(f"  MC curve bank: {len(mc_bank)} curves")

    # ── 1. BS side: exact CF-schedule reprice at the joint optimizer's BS
    #    weights (the expensive part, ~15-25 min) ────────────────────────────
    print(f"\n{'='*70}\nExact BS-side reprice for the joint structure's BS weights...")
    t0 = time.time()
    bs_exact = run_and_cache(
        label="joint_stochastic",
        products=pm.products,
        baseline_weights=pm.base_prod_w,
        target_weights=joint_result.weights_bs_new,
        include_irs=cfg.include_irs,
        bank=mc_bank,
        filename_prefix="irrbb_mc500",
    )
    print(f"BS-side exact reprice done in {time.time()-t0:.0f}s")

    # ── 2. Swap side: exact (analytic DCF) curve-then-6-official-shock delta,
    #    same convention the BS-side exact engine uses -- NOT Section 9B's own
    #    drift-only CVaR-objective convention ────────────────────────────────
    print("\nPricing the new swap overlay against the same 500-curve bank "
          "(curve-then-shock convention, analytic DCF -- fast)...")
    t0 = time.time()
    ladder_mc = sl.price_ladder_against_curve_bank(
        tenors=sl.TENORS_YEARS, direction=joint_result.swap_direction, bank=mc_bank,
    )
    print(f"Swap-side pricing done in {time.time()-t0:.1f}s")

    assert list(ladder_mc["shift_idx"]) == list(range(len(mc_bank))), \
        "swap ladder curve ordering doesn't match mc_bank row order -- cannot combine by position"

    swap_notional_vec = np.array([joint_result.swap_notional.get(bid, 0.0) for bid in ladder_mc["bucket_ids"]])
    t1 = joint_result.tier1_capital

    # ── 3. Combine: total delta = BS exact delta + swap analytic delta,
    #    per (curve, scenario) -- same row format as the existing
    #    irrbb_mc500_baseline.xlsx / irrbb_mc500_stochastic.xlsx caches ──────
    print("\nCombining BS-side exact + swap-side analytic deltas...")
    combined_rows = []
    for _, row in bs_exact.iterrows():
        curve_idx = int(row["shift_idx"])   # mc_bank has shift_idx 0..N-1, unique per curve
        scen = row["scenario"]
        swap_eve_pln = float(ladder_mc["delta_eve_pln"][scen][curve_idx] @ swap_notional_vec)
        swap_nii_pln = float(ladder_mc["delta_nii_pln"][scen][curve_idx] @ swap_notional_vec)
        total_eve_pct_t1 = row["delta_eve_pct_t1"] + swap_eve_pln / t1 * 100.0
        total_nii_pct_t1 = row["delta_nii_pct_t1"] + swap_nii_pln / t1 * 100.0
        combined_rows.append({
            "shape": row["shape"], "level": row["level"], "is_anchor": row["is_anchor"],
            "shift_idx": row["shift_idx"], "anchor_date": row["anchor_date"], "scenario": scen,
            "delta_eve_pct_t1": total_eve_pct_t1, "delta_nii_pct_t1": total_nii_pct_t1,
        })
    combined_df = pd.DataFrame(combined_rows)

    out_dir = os.path.join(_ROOT, "bs_optimization", "output")
    out_path = os.path.join(out_dir, "irrbb_mc500_joint_stochastic_combined.xlsx")
    combined_df.to_excel(out_path, sheet_name="all_scenarios", index=False)
    print(f"Saved combined (BS+swap) result -> {out_path}  ({len(combined_df)} rows)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
