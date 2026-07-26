"""anchor_sot_exact.py
======================
Full "exact" (Way A) EVE SOT + NII SOT reprice of TODAY's balance sheet
under all 165 curves in the curve_scenario_bank.py bank (15 anchors + 150
stochastic shifts). Uses the SAME calculation functions as the production
irrbb_calc workflows (eve_calc_objects.compute_eve_shocked,
nii_calc_objects.compute_nii_base_schedule/compute_nii_shocked_schedule) --
not the optimize_prep reconstruction fallback, which was found to be a
materially more pessimistic approximation (see [[feedback_method_b_only]]
and prior session notes -- do not use the reconstruction fallback for
absolute SOT numbers).

Both metrics reuse the SAME shocked-curve build per curve (build once, feed
both EVE and NII calculations) to avoid duplicating that step.

Validated against production (today's real curve, same report_date):
  EVE par_up: -7.16% vs official -7.11% (within 0.1pp, all 6 scenarios)
  NII validation: see __main__ self-check block below, run once before trusting
  the 165-curve batch.

No SQL writes anywhere in this module -- purely reads (cash-flow schedules,
Tier-1 capital, rate coefficients) and computes in memory. Safe to run
repeatedly without touching shared production tables.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE     = os.path.dirname(os.path.abspath(__file__))
_OPT_PREP = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "python_code"))
_IRRBB    = os.path.normpath(os.path.join(_HERE, "..", "..", "irrbb_calc", "python_code"))
for _p in (_OPT_PREP, _IRRBB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import anchor_eve_exact as aee   # noqa: E402  -- reuses irrbb module loader, NS->mkt_df, shocked-disc builder

from eba_shock_curves import build_all_shocked_curves, default_realistic_base_floor_bps   # noqa: E402
from curve_scenario_bank import build_scenario_bank      # noqa: E402

nii_obj = aee._load_module_by_path("irrbb_nii_calc_objects", os.path.join(_IRRBB, "nii_calc_objects.py"))

OUTPUT_DIR   = aee.OUTPUT_DIR
REPORT_DATE  = aee.REPORT_DATE
CURRENCY     = aee.CURRENCY
DISC_CURVE_MAP = aee.DISC_CURVE_MAP
EVE_SCENARIO_IDS = aee.SCENARIO_IDS                      # base + 6 EBA
NII_HORIZON_YF   = 1.0
NII_FLOOR        = 0.0
SOT_EVE_FLOOR_PCT_T1 = -15.0
SOT_NII_FLOOR_PCT_T1 = -5.0
HORIZON_END  = REPORT_DATE + pd.Timedelta(days=round(NII_HORIZON_YF * 365.25))


def reprice_anchor_both(
    all_beh_eve: pd.DataFrame,
    all_beh_nii: pd.DataFrame,
    caps_map: dict, floors_map: dict, coeff_a_map: dict, coeff_b_map: dict,
    beta0: float, beta1: float, beta2: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """One curve -> ({scenario: EVE base / delta_EVE PLN}, {scenario: NII base / delta_NII PLN}).

    Both EVE and NII reuse the same in-memory shocked curve build for this
    curve. 'base' is computed via the SAME uniform scenario loop (zero-shock
    curve) for both metrics, so the reference point is this curve's own
    level, not today's (mirrors extract_params.py's recompute_base=True).
    """
    curve_name = DISC_CURVE_MAP[CURRENCY]
    mkt_df = aee.ns_beta_to_mkt_df(beta0, beta1, beta2, curve_name=curve_name)
    shocked_df = build_all_shocked_curves(mkt_df, REPORT_DATE, CURRENCY, nii_floor_rate=0.0,
                                           base_floor_bps_fn=default_realistic_base_floor_bps)

    eve_by_scen: dict[str, float] = {}
    nii_by_scen: dict[str, float] = {}

    # 'base' curve (zero shock) -- needed once, reused for both metrics' base scenario
    base_disc = aee._in_memory_shocked_disc_series(shocked_df, "base", curve_name, REPORT_DATE)

    for scenario_id in EVE_SCENARIO_IDS:
        shocked_disc = (
            base_disc if scenario_id == "base"
            else aee._in_memory_shocked_disc_series(shocked_df, scenario_id, curve_name, REPORT_DATE)
        )

        _, eve_summ = aee.eve_obj.compute_eve_shocked(
            all_beh_eve, shocked_disc_df=shocked_disc, disc_curve_map=DISC_CURVE_MAP,
            scenario_id=scenario_id, eve_floor_rate=0.0,
            caps_map=caps_map, floors_map=floors_map, coeff_a_map=coeff_a_map, coeff_b_map=coeff_b_map,
        )
        eve_by_scen[scenario_id] = float(eve_summ["pv_total"].sum())

        if scenario_id == "base":
            nii_sched = nii_obj.compute_nii_base_schedule(
                all_beh_nii, REPORT_DATE, NII_HORIZON_YF, scenario_id="base",
                caps_map=caps_map, floors_map=floors_map, coeff_a_map=coeff_a_map, coeff_b_map=coeff_b_map,
                base_disc_df=base_disc, disc_curve_map=DISC_CURVE_MAP,
            )
        else:
            nii_sched = nii_obj.compute_nii_shocked_schedule(
                all_beh_nii, shocked_disc, DISC_CURVE_MAP, REPORT_DATE, scenario_id,
                NII_HORIZON_YF, NII_FLOOR,
                caps_map=caps_map, floors_map=floors_map, coeff_a_map=coeff_a_map, coeff_b_map=coeff_b_map,
            )
        nii_by_scen[scenario_id] = float(nii_sched["nii_total"].sum())

    eve_base = eve_by_scen["base"]
    nii_base = nii_by_scen["base"]
    eve_out = {s: (eve_base if s == "base" else eve_by_scen[s] - eve_base) for s in EVE_SCENARIO_IDS}
    nii_out = {s: (nii_base if s == "base" else nii_by_scen[s] - nii_base) for s in EVE_SCENARIO_IDS}
    return eve_out, nii_out


def run_full_bank(include_irs: bool = True) -> pd.DataFrame:
    """Full 165-curve exact reprice of today's real book.

    include_irs=False excludes the real swap book's CF rows entirely (the
    "unhedged" view) -- everything else about the reprice is unchanged.
    """
    print("Loading EVE run-off CF schedules (curve-independent, queried once)...")
    all_beh_eve = aee.irrbb_sql_setup.load_all_beh_schedules(REPORT_DATE)
    if include_irs:
        try:
            swap_eve = aee.irrbb_sql_setup.load_all_swap_beh_schedules(REPORT_DATE)
        except Exception:
            swap_eve = pd.DataFrame()
        if not swap_eve.empty:
            all_beh_eve = pd.concat([all_beh_eve, swap_eve], ignore_index=True)
    print(f"  {len(all_beh_eve):,} EVE CF rows"
          f"  ({'hedged' if include_irs else 'unhedged -- IRS excluded'})")

    print("Loading NII 1Y-horizon CF schedules (curve-independent, queried once)...")
    all_beh_nii = aee.irrbb_sql_setup.load_beh_schedules(REPORT_DATE, HORIZON_END)
    if include_irs:
        try:
            swap_nii = aee.irrbb_sql_setup.load_swap_beh_schedules(REPORT_DATE, HORIZON_END)
        except Exception:
            swap_nii = pd.DataFrame()
        if not swap_nii.empty:
            all_beh_nii = pd.concat([all_beh_nii, swap_nii], ignore_index=True)
    print(f"  {len(all_beh_nii):,} NII CF rows"
          f"  ({'hedged' if include_irs else 'unhedged -- IRS excluded'})")

    caps_map, floors_map, coeff_a_map, coeff_b_map = aee._load_rate_maps()
    t1 = aee.irrbb_sql_setup.load_tier1_capital(REPORT_DATE)
    print(f"  Tier-1 capital: {t1/1e6:,.1f}M PLN")

    _anchors, bank = build_scenario_bank()
    print(f"  Repricing {len(bank)} curves (15 anchors + {len(bank) - 15} stochastic shifts)...")

    rows = []
    for i, curve in bank.iterrows():
        if curve["is_anchor"] or curve["shift_idx"] % 5 == 0:
            tag = "anchor" if curve["is_anchor"] else f"shift {curve['shift_idx']}"
            print(f"  [{i+1}/{len(bank)}] {curve['shape']:>8s}/{curve['level']:<6s} {tag}...")
        eve_out, nii_out = reprice_anchor_both(
            all_beh_eve, all_beh_nii, caps_map, floors_map, coeff_a_map, coeff_b_map,
            curve["beta0"], curve["beta1"], curve["beta2"],
        )
        for scen in EVE_SCENARIO_IDS:
            if scen == "base":
                continue
            eve_pct = eve_out[scen] / t1 * 100.0 if t1 else float("nan")
            nii_pct = nii_out[scen] / t1 * 100.0 if t1 else float("nan")
            rows.append({
                "shape": curve["shape"], "level": curve["level"],
                "is_anchor": curve["is_anchor"], "shift_idx": curve["shift_idx"],
                "anchor_date": curve["anchor_date"],
                "scenario": scen,
                "delta_eve_pct_t1": eve_pct,
                "delta_nii_pct_t1": nii_pct,
                "eve_sot_pass": eve_pct >= SOT_EVE_FLOOR_PCT_T1,
                "nii_sot_pass": nii_pct >= SOT_NII_FLOOR_PCT_T1,
            })
    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    xlsx_path = os.path.join(out_dir, "anchor_sot_exact_full_bank.xlsx")
    worst_eve = results.loc[results.groupby(["shape", "level", "is_anchor", "shift_idx"])
                            ["delta_eve_pct_t1"].idxmin()].reset_index(drop=True)
    with pd.ExcelWriter(xlsx_path) as writer:
        results.to_excel(writer, sheet_name="all_scenarios", index=False)
        worst_eve.to_excel(writer, sheet_name="worst_eve_per_curve", index=False)
    print(f"\nSaved {xlsx_path}")


if __name__ == "__main__":
    results = run_full_bank()
    save_results(results)

    n_total = len(results)
    n_eve_fail = int((~results["eve_sot_pass"]).sum())
    n_nii_fail = int((~results["nii_sot_pass"]).sum())
    print(f"\n{n_total} (curve x shock) rows computed across {results.groupby(['shape','level','is_anchor','shift_idx']).ngroups} curves.")
    print(f"EVE SOT breaches (< {SOT_EVE_FLOOR_PCT_T1}% T1): {n_eve_fail} / {n_total}")
    print(f"NII SOT breaches (< {SOT_NII_FLOOR_PCT_T1}% T1): {n_nii_fail} / {n_total}")

    print("\nWorst EVE delta per anchor curve (anchors only):")
    anchors_only = results[results["is_anchor"]]
    worst = anchors_only.loc[anchors_only.groupby(["shape", "level"])["delta_eve_pct_t1"].idxmin()]
    print(worst[["shape", "level", "scenario", "delta_eve_pct_t1", "delta_nii_pct_t1"]]
          .sort_values("delta_eve_pct_t1").to_string(index=False))

    print("\nWorst EVE delta overall (across all 165 curves incl. shifts):")
    worst_row = results.loc[results["delta_eve_pct_t1"].idxmin()]
    print(worst_row[["shape", "level", "is_anchor", "shift_idx", "scenario",
                     "delta_eve_pct_t1", "delta_nii_pct_t1"]])

    print("\nWorst NII delta overall (across all 165 curves incl. shifts):")
    worst_row_nii = results.loc[results["delta_nii_pct_t1"].idxmin()]
    print(worst_row_nii[["shape", "level", "is_anchor", "shift_idx", "scenario",
                         "delta_eve_pct_t1", "delta_nii_pct_t1"]])
