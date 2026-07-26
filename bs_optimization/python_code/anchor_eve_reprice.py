"""anchor_eve_reprice.py
========================
Full daily-CF EVE SOT reprice of TODAY's balance sheet under the 15 synthetic
anchor curves built by curve_scenario_bank.py.

Reuses the SAME cash-flow schedule + shock/discount machinery as the
production pipeline (extract_params._compute_cohort_eve_pv), with
recompute_base=True so the 'base' scenario also reflects the anchor curve's
own level rather than today's -- this is a genuine Method B (exact daily-CF)
reprice, not the duration-based approximation (see [[feedback_method_b_only]]
in memory: Method A must never substitute for Method B in reporting).

Scope, deliberately
--------------------
- Reuses TODAY's cash-flow schedule (timing, principal amounts, contracted
  rates) unchanged across anchors -- this is the correct SOT methodology
  (shock the CURVE level, not the realized cash-flow history). See
  extract_params._compute_cohort_eve_pv docstring for the recompute_base
  design.
- Skips the "exact override" tables (cf.eve_*_scenarios) that the production
  pipeline prefers when available. Those are written by a separate upstream
  workflow (eve_calc_workflow) for TODAY's report_date only -- reusing them
  here would be WRONG (not just less precise) for a hypothetical anchor
  curve, since they don't respond to the anchor at all.
- Evaluates the EXISTING (unchanged) balance sheet under each anchor -- a
  robustness SCREEN of the current structure, not yet a per-product LP
  sensitivity matrix. Wiring this into bs_optimizer's LP would require cash
  flow generation itself (cf.products, an even further upstream pipeline) to
  respond to hypothetical product-weight changes -- a separate, larger
  undertaking than this reprice.
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

import extract_params as ep                                   # noqa: E402
from bs_vector import BalanceSheetParams                      # noqa: E402
from eba_shock_curves import build_all_shocked_curves, default_realistic_base_floor_bps   # noqa: E402
from ns_curve_model import ns_design_matrix, DIEBOLD_LI_TAU     # noqa: E402
from curve_scenario_bank import build_scenario_bank             # noqa: E402

OUTPUT_DIR   = os.path.normpath(os.path.join(_HERE, "..", "output"))
PARAMS_NPZ   = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "output", "product_params.npz"))
N_MONTHS     = 360
SOT_EVE_FLOOR_PCT_T1 = -15.0   # EBA/RTS standard floor, matches bs_optimizer.py default


def ns_beta_to_mkt_df(
    beta0: float, beta1: float, beta2: float,
    tau: float = DIEBOLD_LI_TAU,
    n_months: int = N_MONTHS,
    curve_name: str = "PLN_disc_curve",
) -> pd.DataFrame:
    """Reconstruct a (curve_name, n_days, d_f) discount curve from NS betas.

    Monthly node grid (months 1..n_months), matching the schema
    eba_shock_curves.build_all_shocked_curves() expects as `mkt_df`.
    """
    months  = np.arange(1, n_months + 1)
    t_years = months / 12.0
    X       = ns_design_matrix(t_years, tau)          # (n_months, 3) = [1, L1, L2]
    r_bps   = X @ np.array([beta0, beta1, beta2])
    d_f     = np.exp(-(r_bps / 10_000.0) * t_years)
    n_days  = months * (365.25 / 12.0)
    return pd.DataFrame({"curve_name": curve_name, "n_days": n_days, "d_f": d_f})


def anchor_curves_dict(
    beta0: float, beta1: float, beta2: float,
    tau: float = DIEBOLD_LI_TAU,
    currency: str = "PLN",
) -> dict:
    """{scenario_id: {curve_name: (n_days, log_d_f)}} for one anchor's base + 6
    EBA shocks (+ 'own'), via the SAME shock formulas as production
    (eba_shock_curves.build_all_shocked_curves), applied to a synthetic
    NS-reconstructed curve instead of today's market curve.
    """
    curve_name = ep._CCY_CURVE.get(currency, "PLN_disc_curve")
    mkt_df  = ns_beta_to_mkt_df(beta0, beta1, beta2, tau, curve_name=curve_name)
    shocked = build_all_shocked_curves(mkt_df, pd.Timestamp("1900-01-01"), currency,
                                        base_floor_bps_fn=default_realistic_base_floor_bps)

    curves: dict = {}
    for (scen, cname), grp in shocked.groupby(["scenario_id", "curve_name"]):
        grp = grp.sort_values("n_days")
        nd  = grp["n_days"].to_numpy(dtype=float)
        ldf = np.log(np.maximum(grp["d_f"].to_numpy(dtype=float), 1e-12))
        curves.setdefault(str(scen), {})[str(cname)] = (nd, ldf)
    return curves


def reprice_anchor(
    df_schedule: pd.DataFrame,
    ir_coeff: pd.DataFrame,
    beta0: float, beta1: float, beta2: float,
    tau: float = DIEBOLD_LI_TAU,
    currency: str = "PLN",
) -> tuple[np.ndarray, dict, list[str]]:
    """Full daily-CF EVE reprice of TODAY's balance sheet under one anchor curve.

    Returns (pv_all, key_to_gid, scen_all) -- same shapes as
    extract_params._compute_cohort_eve_pv, but with base+shocks BOTH computed
    against the anchor curve (recompute_base=True), so deltas are measured
    against the anchor's own level, not today's.
    """
    curves = anchor_curves_dict(beta0, beta1, beta2, tau, currency)
    return ep._compute_cohort_eve_pv(
        df_schedule, curves, ir_coeff, ep.SHOCKED_SCENARIO_IDS, recompute_base=True
    )


def _sign_by_key(params: BalanceSheetParams) -> dict:
    out = {}
    for pc, side, ccy, sy, sm, sg in zip(
        params.product_code, params.bs_side, params.currency,
        params.start_year, params.start_month, params.sign,
    ):
        if not (np.isfinite(sy) and np.isfinite(sm)):
            continue   # single-row products (no cohort start date) never appear in key_to_gid
        out[(str(pc), str(side), str(ccy), int(sy), int(sm))] = float(sg)
    return out


def eve_totals_by_scenario(
    pv_all: np.ndarray, key_to_gid: dict, scen_all: list[str], sign_by_key: dict,
) -> dict[str, float]:
    """Aggregate pv_all -> {scenario: EVE level (base) / delta_EVE (shocks)}, whole book, PLN."""
    if pv_all.size == 0:
        return {s: 0.0 for s in scen_all}
    sign_arr = np.zeros(pv_all.shape[0])
    for ck, g in key_to_gid.items():
        sign_arr[g] = sign_by_key.get(ck, 0.0)
    pv_signed = pv_all.sum(axis=1) * sign_arr[:, None]         # (n_groups, n_scen)
    totals    = pv_signed.sum(axis=0)                           # (n_scen,)
    base_idx  = scen_all.index("base")
    eve_base  = float(totals[base_idx])
    return {
        s: (float(totals[i]) if i == base_idx else float(totals[i]) - eve_base)
        for i, s in enumerate(scen_all)
    }


def run_all_anchors() -> pd.DataFrame:
    """Full reprice of today's balance sheet under all 15 anchor curves.

    Returns one row per (shape, level, scenario): delta_EVE (PLN), delta_EVE
    as % of Tier-1, and a pass/fail flag against the -15% SOT floor.
    """
    print("Loading cash-flow schedule (curve-independent, queried once)...")
    df_schedule = ep._query_cohort_cf_schedule()
    print(f"  {len(df_schedule)} schedule rows, "
          f"{df_schedule[ep._COHORT_KEY].drop_duplicates().shape[0] if not df_schedule.empty else 0} cohort groups")

    ir_coeff = ep._load_rate_coefficients()
    params   = BalanceSheetParams.load(PARAMS_NPZ)
    t1       = float(params.balance_arr[params.is_equity].sum())
    sign_by_key = _sign_by_key(params)
    print(f"  Tier-1 capital (equity balance): {t1/1e6:,.1f}M PLN")

    anchors, _bank = build_scenario_bank()

    rows = []
    for _, anchor in anchors.iterrows():
        print(f"  Repricing anchor: {anchor['shape']:>8s} / {anchor['level']:<6s}  "
              f"(anchor_date={anchor['anchor_date'].date()})...")
        pv_all, key_to_gid, scen_all = reprice_anchor(
            df_schedule, ir_coeff, anchor["beta0"], anchor["beta1"], anchor["beta2"],
        )
        totals = eve_totals_by_scenario(pv_all, key_to_gid, scen_all, sign_by_key)
        eve_base = totals["base"]
        for scen in scen_all:
            if scen == "base":
                continue
            delta = totals[scen]
            delta_pct_t1 = delta / t1 * 100.0 if t1 else float("nan")
            rows.append({
                "shape": anchor["shape"],
                "level": anchor["level"],
                "anchor_date": anchor["anchor_date"],
                "scenario": scen,
                "eve_base_pln": eve_base,
                "delta_eve_pln": delta,
                "delta_eve_pct_t1": delta_pct_t1,
                "sot_pass": delta_pct_t1 >= SOT_EVE_FLOOR_PCT_T1,
            })

    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    xlsx_path = os.path.join(out_dir, "anchor_eve_reprice.xlsx")
    worst_per_anchor = (
        results.loc[results.groupby(["shape", "level"])["delta_eve_pct_t1"].idxmin()]
        .reset_index(drop=True)
    )
    with pd.ExcelWriter(xlsx_path) as writer:
        results.to_excel(writer, sheet_name="all_scenarios", index=False)
        worst_per_anchor.to_excel(writer, sheet_name="worst_shock_per_anchor", index=False)
    print(f"\nSaved {xlsx_path}")


if __name__ == "__main__":
    results = run_all_anchors()
    save_results(results)

    print(f"\n{len(results)} (anchor x shock) rows computed.")
    n_fail = int((~results["sot_pass"]).sum())
    print(f"SOT breaches (delta_EVE% T1 < {SOT_EVE_FLOOR_PCT_T1}): {n_fail} / {len(results)}")
    if n_fail:
        print("\nBreaching rows:")
        print(results.loc[~results["sot_pass"],
                          ["shape", "level", "scenario", "delta_eve_pct_t1"]]
              .to_string(index=False))

    worst = results.loc[results.groupby(["shape", "level"])["delta_eve_pct_t1"].idxmin()]
    print("\nWorst shock per anchor:")
    print(worst[["shape", "level", "scenario", "delta_eve_pct_t1", "sot_pass"]]
          .sort_values("delta_eve_pct_t1").to_string(index=False))
