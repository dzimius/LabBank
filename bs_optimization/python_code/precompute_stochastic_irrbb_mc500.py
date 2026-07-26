"""precompute_stochastic_irrbb_mc500.py
========================================
Exact (Method B) EVE/NII breach counter for the Section 9C stochastic IRRBB
severity-minimizing structure, against the SAME 500 mean-reverting Monte
Carlo curves used throughout Section 5/9B/9C -- verifies whether directly
minimizing breach severity (instead of Section 9B's CVaR-of-drift-only-P&L
objective) actually achieves fewer real breaches than Section 5's BS-only
structure (129/500 NII) and Section 9B's joint structure (211/500 NII).

Two pieces, combined (identical pattern to precompute_joint_stochastic_mc500.py):
  1. BS side: exact CF-schedule reprice (bank_reprice_at_weights.py) at the
     severity optimizer's BS weights (result.weights_bs_new).
  2. Swap side: the new swap ladder overlay's EVE/NII delta, analytic DCF
     (swap_ladder.price_ladder_against_curve_bank, curve-then-6-official-
     shock convention -- the SAME convention this optimizer's own E0 cache
     and LP already use, unlike Section 9B's drift-only convention).

Combined delta_eve_pct_t1 = BS exact delta + swap analytic delta, per
(curve, scenario) -- saved in the SAME row format as the existing
irrbb_mc500_*.xlsx caches, so the notebook's existing _curve_breach_count()
helper works unchanged.

Meant to be run once in the background -- NOT inline in the notebook.
Expected runtime: ~15-25 min (dominated by the BS-side exact CF reprice).

Usage
-----
    python precompute_stochastic_irrbb_mc500.py
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
    from bs_optimizer import _ProductMap, OptimizationConfig
    from optimizer_io import load_optimizer_config
    import ns_curve_model as nsm
    from stochastic_irrbb_optimizer import build_joint_mc_severity_matrices, optimize_stochastic_irrbb
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

    # The E0 cache (irrbb_mc500_baseline.xlsx) is hedged -- this optimizer's
    # LP and this verification script must both stay on that same basis (see
    # stochastic_irrbb_optimizer.py's module docstring). Fail loud rather
    # than silently mismatch if the config ever changes.
    assert cfg.include_irs, "irrbb_mc500_baseline.xlsx (E0) is hedged; include_irs must stay True"

    print("Rebuilding Section 5/9B/9C's inputs bit-for-bit (seed=42, N=500, "
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

    print("Re-solving the stochastic IRRBB severity optimizer to get the SAME "
          "weights/swap notional the notebook displays (cheap sparse LP solve)...")
    lp_blocks = build_joint_mc_severity_matrices(cfg, params, cr, factor_draws, today_beta)
    cfg_free = OptimizationConfig(
        mode="max_shift", max_shift=cfg.max_shift,
        sot_eve_floor=cfg.sot_eve_floor, sot_eve_buffer=cfg.sot_eve_buffer,
        sot_nii_floor=cfg.sot_nii_floor, sot_nii_buffer=cfg.sot_nii_buffer,
        include_irs=cfg.include_irs, min_lcr=cfg.min_lcr, min_nsfr=cfg.min_nsfr,
        min_t1_rwa=cfg.min_t1_rwa, fixed_products=cfg.fixed_products,
    )
    result = optimize_stochastic_irrbb(cfg_free, params, cr, lp_blocks)
    print(f"  Approx. severity: {result.severity_total_approx_old:.1f} -> {result.severity_total_approx_new:.1f}  |  "
          f"swap notional: {sum(result.swap_notional.values())/1e6:,.1f}M")

    mc_bank = build_mc_scenario_bank(factor_draws, today_beta, ase.REPORT_DATE)
    print(f"  MC curve bank: {len(mc_bank)} curves")

    # ── 1. BS side: exact CF-schedule reprice at the severity optimizer's BS
    #    weights (the expensive part, ~15-25 min) ────────────────────────────
    print(f"\n{'='*70}\nExact BS-side reprice for the severity-minimizing structure's BS weights...")
    t0 = time.time()
    bs_exact = run_and_cache(
        label="stochastic_irrbb",
        products=pm.products,
        baseline_weights=pm.base_prod_w,
        target_weights=result.weights_bs_new,
        include_irs=cfg.include_irs,
        bank=mc_bank,
        filename_prefix="irrbb_mc500",
    )
    print(f"BS-side exact reprice done in {time.time()-t0:.0f}s")

    # ── 2. Swap side: exact (analytic DCF) curve-then-6-official-shock delta,
    #    same convention this optimizer's own E0/LP already used ────────────
    print("\nPricing the new swap overlay against the same 500-curve bank "
          "(curve-then-shock convention, analytic DCF -- fast)...")
    t0 = time.time()
    ladder_mc = sl.price_ladder_against_curve_bank(
        tenors=sl.TENORS_YEARS, direction=result.swap_direction, bank=mc_bank,
    )
    print(f"Swap-side pricing done in {time.time()-t0:.1f}s")

    assert list(ladder_mc["shift_idx"]) == list(range(len(mc_bank))), \
        "swap ladder curve ordering doesn't match mc_bank row order -- cannot combine by position"

    swap_notional_vec = np.array([result.swap_notional.get(bid, 0.0) for bid in ladder_mc["bucket_ids"]])
    t1 = result.tier1_capital

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
    out_path = os.path.join(out_dir, "irrbb_mc500_stochastic_irrbb_combined.xlsx")
    combined_df.to_excel(out_path, sheet_name="all_scenarios", index=False)
    print(f"Saved combined (BS+swap) result -> {out_path}  ({len(combined_df)} rows)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
