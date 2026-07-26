"""hedge_comparison.py
=====================
Three-way comparison of EVE/NII SOT breaches across the full 165-curve bank
(curve_scenario_bank.py), all with the SAME real banking book (today's
actual structure -- see the plan doc for why it's kept fixed):

  1. Current hedge  -- today's real balance sheet as-is
                        (anchor_sot_exact.py default, include_irs=True)
  2. No hedge       -- today's real balance sheet with the IRS book excluded
                        (anchor_sot_exact.py, include_irs=False)
  3. Optimal hedge   -- today's real balance sheet + the notional-optimal
                        swap_ladder.py overlay (hedge_optimizer.py)

Every number here is exact, not a linear approximation: views 1/2 come from
the same contract-level CF engine anchor_sot_exact.py already validated
against production (-7.16% vs official -7.11% for par_up); view 3 adds the
ladder's exact analytic-DCF sensitivity (swap_ladder.py) on top of view 1's
already-exact numbers.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import anchor_sot_exact as ase
import hedge_optimizer as ho

OUTPUT_DIR    = ase.OUTPUT_DIR
CURRENT_XLSX  = os.path.join(OUTPUT_DIR, "anchor_sot_exact_full_bank.xlsx")
UNHEDGED_XLSX = os.path.join(OUTPUT_DIR, "anchor_sot_exact_unhedged_full_bank.xlsx")
OUT_XLSX      = os.path.join(OUTPUT_DIR, "hedge_comparison.xlsx")

EVE_FLOOR = ase.SOT_EVE_FLOOR_PCT_T1
NII_FLOOR = ase.SOT_NII_FLOOR_PCT_T1


def _load_or_run(cached_path: str, include_irs: bool) -> pd.DataFrame:
    if os.path.exists(cached_path):
        print(f"  Using cached {cached_path}")
        return pd.read_excel(cached_path, sheet_name="all_scenarios")
    print(f"  {cached_path} not found -- running full 165-curve reprice "
          f"(include_irs={include_irs}); this repeats the SQL/CF-heavy pass, "
          f"can take a while...")
    results = ase.run_full_bank(include_irs=include_irs)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(cached_path) as writer:
        results.to_excel(writer, sheet_name="all_scenarios", index=False)
    return results


def _breach_summary(df: pd.DataFrame, label: str) -> dict:
    n_total   = len(df)
    n_eve     = int((df["delta_eve_pct_t1"] < EVE_FLOOR).sum())
    n_nii     = int((df["delta_nii_pct_t1"] < NII_FLOOR).sum())
    worst_eve = float(df["delta_eve_pct_t1"].min())
    worst_nii = float(df["delta_nii_pct_t1"].min())
    print(f"  [{label}] EVE breaches: {n_eve}/{n_total}   NII breaches: {n_nii}/{n_total}"
          f"   worst EVE: {worst_eve:+.2f}% T1   worst NII: {worst_nii:+.2f}% T1")
    return dict(view=label, n_total=n_total, n_eve_breach=n_eve, n_nii_breach=n_nii,
                worst_eve_pct_t1=worst_eve, worst_nii_pct_t1=worst_nii)


def run_comparison(penalty_weight: float = 50.0, notional_cap: float | None = 1e9,
                    total_notional_cap: float | None = 5e9) -> dict:
    print("1/3 Current hedge (today's real book, as-is)...")
    current_df = _load_or_run(CURRENT_XLSX, include_irs=True)

    print("2/3 No hedge (IRS excluded)...")
    unhedged_df = _load_or_run(UNHEDGED_XLSX, include_irs=False)

    print("3/3 Optimal hedge (today's book + solved swap_ladder overlay)...")
    opt = ho.solve_optimal_ladder(notional_cap=notional_cap, total_notional_cap=total_notional_cap,
                                  penalty_weight=penalty_weight)
    ladder = opt["ladder"]
    t1 = opt["tier1_capital"]
    x = opt["notional"]

    optimal_df = current_df.copy()
    key_cols = ["shape", "level", "is_anchor", "shift_idx"]
    curve_df = pd.DataFrame({
        **{k: ladder[k] for k in key_cols},
        "_curve_idx": np.arange(len(ladder["shape"])),
    })
    for s in ho.sl.SHOCKED_SCENS:
        add_eve_pct = (ladder["delta_eve_pln"][s] @ x) / t1 * 100.0    # (n_curves,)
        add_nii_pct = (ladder["delta_nii_pln"][s] @ x) / t1 * 100.0
        add_df = curve_df.copy()
        add_df["scenario"] = s
        add_df["_add_eve"] = add_eve_pct
        add_df["_add_nii"] = add_nii_pct
        optimal_df = optimal_df.merge(
            add_df[key_cols + ["scenario", "_add_eve", "_add_nii"]],
            on=key_cols + ["scenario"], how="left",
        )
        mask = optimal_df["scenario"] == s
        optimal_df.loc[mask, "delta_eve_pct_t1"] += optimal_df.loc[mask, "_add_eve"].fillna(0.0)
        optimal_df.loc[mask, "delta_nii_pct_t1"] += optimal_df.loc[mask, "_add_nii"].fillna(0.0)
        optimal_df = optimal_df.drop(columns=["_add_eve", "_add_nii"])
    optimal_df["eve_sot_pass"] = optimal_df["delta_eve_pct_t1"] >= EVE_FLOOR
    optimal_df["nii_sot_pass"] = optimal_df["delta_nii_pct_t1"] >= NII_FLOOR

    print("\nBreach summary (out of 165 curves x 6 scenarios = 990):")
    summary = [
        _breach_summary(unhedged_df, "no_hedge"),
        _breach_summary(current_df,  "current_hedge"),
        _breach_summary(optimal_df,  "optimal_hedge"),
    ]
    summary_df = pd.DataFrame(summary)

    ladder_df = pd.DataFrame({
        "bucket_id": opt["bucket_ids"],
        "notional_pln": x,
    })
    ladder_df = ladder_df[ladder_df["notional_pln"] > 1.0].sort_values(
        "notional_pln", ascending=False).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        ladder_df.to_excel(writer, sheet_name="optimal_ladder_notionals", index=False)
        unhedged_df.to_excel(writer, sheet_name="no_hedge_detail", index=False)
        current_df.to_excel(writer, sheet_name="current_hedge_detail", index=False)
        optimal_df.to_excel(writer, sheet_name="optimal_hedge_detail", index=False)
    print(f"\nSaved {OUT_XLSX}")
    print(f"Optimal ladder: {len(ladder_df)} active buckets, "
          f"{ladder_df['notional_pln'].sum()/1e6:,.1f}M PLN added notional, "
          f"EP carry {opt['ep_carry_total']/1e6:,.2f}M PLN/yr")

    return dict(summary=summary_df, ladder=ladder_df, optimizer_result=opt)


if __name__ == "__main__":
    run_comparison()
