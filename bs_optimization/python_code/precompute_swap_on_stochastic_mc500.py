"""precompute_swap_on_stochastic_mc500.py
==========================================
Exact (Method B) EVE/NII breach counter for a NEW structure: Section 5's
BS-only stochastic-optimized weights, FIXED (not re-optimized), with a new
swap-ladder overlay chosen ONLY to minimize breach severity on top of that
already-chosen balance sheet -- i.e. "given we've already committed to
Section 5's product mix (the best exact performer so far, 129/500 NII
breaches), does layering a diversified swap hedge on top help further,
without touching the BS side at all."

Uses optimize_stochastic_irrbb_laddered (not the free per-bucket
optimize_stochastic_irrbb) -- since the BS side is fixed here, the remaining
swap-only problem is small enough to afford a genuinely realistic ladder
shape: ONE decision variable per tenor (5, not 300), each spread EVENLY
across every one of that tenor's own monthly vintage buckets, rather than
letting the LP freely concentrate into whichever 1-5 buckets a per-bucket
cap allows (a real LP vertex-solution tendency, not how a real desk ladders
a hedge -- it transacts small, frequent tranches). See
stochastic_irrbb_optimizer.optimize_stochastic_irrbb_laddered's docstring.

Two pieces, combined (identical pattern to the other MC500 precomputes this
session):
  1. BS side: exact CF-schedule reprice (bank_reprice_at_weights.py) at
     Section 5's OWN weights (stoch_result.weights_new) -- this is IDENTICAL
     to the already-cached irrbb_mc500_stochastic.xlsx, so this step is
     actually a no-op re-run for verification-consistency's sake; reusing
     the cache directly would also be valid, but re-running keeps this
     script self-contained and independent of prior cache freshness.
  2. Swap side: the new (diversified, capped) swap ladder's EVE/NII delta,
     analytic DCF (swap_ladder.price_ladder_against_curve_bank).

Combined delta_eve_pct_t1 = BS exact delta + swap analytic delta, per
(curve, scenario) -- saved in the SAME row format as every other
irrbb_mc500_*.xlsx cache.

Usage
-----
    python precompute_swap_on_stochastic_mc500.py
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

TOTAL_NOTIONAL_CAP_PCT_TA = 0.05   # 5% of total assets, aggregate across all tenors (no per-bucket cap
                                    # needed -- the laddered even-spread already limits concentration)


def main() -> None:
    from bs_vector import BalanceSheetParams, CohortRates
    from bs_optimizer import _ProductMap
    from optimizer_io import load_optimizer_config
    import ns_curve_model as nsm
    from stochastic_optimizer import StochasticConfig, build_scenario_ep_matrix, optimize_stochastic
    from stochastic_irrbb_optimizer import build_joint_mc_severity_matrices, optimize_stochastic_irrbb_laddered
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
    assert cfg.include_irs, "E0/exact caches are hedged; include_irs must stay True"

    print("Rebuilding Section 5's inputs bit-for-bit (seed=42, N=500, "
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

    print("Re-solving Section 5's BS-only stochastic optimizer (cheap LP solve)...")
    stoch_cfg = StochasticConfig(n_scenarios=N_SCENARIOS, horizon_months=HORIZON_MONTHS,
                                  cvar_alpha=0.95, risk_lambda=1.0, seed=MC_SEED)
    C_bs, blocks_bs = build_scenario_ep_matrix(cfg, stoch_cfg, params, cr, factor_draws, tau)
    stoch_result = optimize_stochastic(cfg, stoch_cfg, params, cr, C_bs, blocks_bs)
    print(f"  E[EP]: {stoch_result.ep_mean_old/1e6:,.1f}M -> {stoch_result.ep_mean_new/1e6:,.1f}M")

    print("Solving for a LADDERED swap overlay ON TOP of those FIXED BS weights "
          f"(total_notional_cap={TOTAL_NOTIONAL_CAP_PCT_TA*100:.0f}% of total assets, "
          "each used tenor spread evenly across its own monthly buckets)...")
    lp_blocks = build_joint_mc_severity_matrices(cfg, params, cr, factor_draws, today_beta)
    result = optimize_stochastic_irrbb_laddered(
        cfg, params, cr, lp_blocks,
        fixed_bs_weights=stoch_result.weights_new,
        total_notional_cap_pct_ta=TOTAL_NOTIONAL_CAP_PCT_TA,
    )
    nz = {b: n for b, n in result.swap_notional.items() if n > 1.0}
    print(f"  Swap overlay: {len(nz)} buckets, total {sum(nz.values())/1e6:,.1f}M")

    mc_bank = build_mc_scenario_bank(factor_draws, today_beta, ase.REPORT_DATE)
    print(f"  MC curve bank: {len(mc_bank)} curves")

    # ── 1. BS side: exact CF-schedule reprice at Section 5's OWN weights
    #    (the expensive part, ~15-25 min) ────────────────────────────────────
    print(f"\n{'='*70}\nExact BS-side reprice for Section 5's BS-only weights (fixed)...")
    t0 = time.time()
    bs_exact = run_and_cache(
        label="swap_on_stochastic",
        products=pm.products,
        baseline_weights=pm.base_prod_w,
        target_weights=stoch_result.weights_new,
        include_irs=cfg.include_irs,
        bank=mc_bank,
        filename_prefix="irrbb_mc500",
    )
    print(f"BS-side exact reprice done in {time.time()-t0:.0f}s")

    # ── 2. Swap side: exact (analytic DCF) curve-then-6-official-shock delta ──
    print("\nPricing the diversified swap overlay against the same 500-curve bank...")
    t0 = time.time()
    ladder_mc = sl.price_ladder_against_curve_bank(
        tenors=sl.TENORS_YEARS, direction=result.swap_direction, bank=mc_bank,
    )
    print(f"Swap-side pricing done in {time.time()-t0:.1f}s")

    assert list(ladder_mc["shift_idx"]) == list(range(len(mc_bank))), \
        "swap ladder curve ordering doesn't match mc_bank row order -- cannot combine by position"

    swap_notional_vec = np.array([result.swap_notional.get(bid, 0.0) for bid in ladder_mc["bucket_ids"]])
    t1 = result.tier1_capital

    # ── 3. Combine: total delta = BS exact delta + swap analytic delta ──────
    print("\nCombining BS-side exact + swap-side analytic deltas...")
    combined_rows = []
    for _, row in bs_exact.iterrows():
        curve_idx = int(row["shift_idx"])
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
    out_path = os.path.join(out_dir, "irrbb_mc500_swap_on_stochastic_combined.xlsx")
    combined_df.to_excel(out_path, sheet_name="all_scenarios", index=False)
    print(f"Saved combined (BS+swap) result -> {out_path}  ({len(combined_df)} rows)")

    # Also save the swap notional table itself for the notebook's ladder chart,
    # with each bucket's actual (hypothetical) historical origination date --
    # today's report date minus its elapsed_m -- so the chart can show a real
    # calendar-date x-axis instead of "elapsed months since origination."
    swap_df = pd.DataFrame(
        [{"bucket_id": b, "tenor_years": t, "elapsed_m": e, "notional_pln": n,
          "origination_date": ase.REPORT_DATE - pd.DateOffset(months=int(e))}
         for b, t, e, n in zip(ladder_mc["bucket_ids"], ladder_mc["tenor_years"], ladder_mc["elapsed_m"],
                                 swap_notional_vec)
         if n > 1.0]
    )
    swap_out_path = os.path.join(out_dir, "swap_on_stochastic_notional.xlsx")
    swap_df.to_excel(swap_out_path, sheet_name="notional", index=False)
    print(f"Saved swap notional table -> {swap_out_path}  ({len(swap_df)} active buckets)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
