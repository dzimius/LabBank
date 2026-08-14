"""anchor_eve_exact.py
======================
Full "exact" (Way A) daily-CF EVE SOT reprice of TODAY's balance sheet under
the 15 synthetic anchor curves from curve_scenario_bank.py -- using the SAME
computation functions as the production irrbb_calc/eve_calc_workflow.py
pipeline (eve_calc_objects.compute_eve_shocked_schedule), not the
optimize_prep reconstruction fallback (see anchor_eve_reprice.py, which is
now understood to be a materially more pessimistic approximation -- do not
use it for absolute SOT numbers, see memory).

Design
------
- Cash-flow schedules (all_beh) and per-product rate coefficients are
  curve-independent -- loaded ONCE, reused across all 15 anchors.
- For each anchor: build shocked discount curves in-memory (NS betas ->
  synthetic curve -> eba_shock_curves.build_all_shocked_curves), then call
  eve_calc_objects.compute_eve_shocked_schedule() directly for ALL 7
  scenarios (base + 6 EBA shocks) -- including 'base' itself, so the
  reference point is the anchor's own level, not today's (mirrors the
  recompute_base=True design in extract_params.py / anchor_eve_reprice.py).
- NEVER calls any sql_setup.write_*/reset_* function -- purely reads
  (cash-flow schedules, Tier-1 capital) and computes in memory. No shared
  production SQL table is touched, so this is safe to run repeatedly.

Module-name collision note
---------------------------
Both optimize_prep/python_code and irrbb_calc/python_code have a file named
sql_setup.py (different DB helpers, different schemas). This session already
has optimize_prep's sql_setup cached under the module name "sql_setup" (via
extract_params.py's `import sql_setup as opt_sql`), so a plain `import
sql_setup` here would silently return the WRONG (already-cached) module.
irrbb_calc's config.py/sql_setup.py/eve_calc_objects.py are therefore loaded
by explicit file path via importlib, under distinct names, to guarantee the
correct module is used regardless of import order.
"""
from __future__ import annotations

import importlib.util
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


def _load_module_by_path(name: str, path: str):
    """Load a module from an explicit file path under a private name, bypassing
    sys.modules name-collision with any same-named module already imported
    from a different directory (see module docstring)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


irrbb_config    = _load_module_by_path("irrbb_config",     os.path.join(_IRRBB, "config.py"))
irrbb_sql_setup = _load_module_by_path("irrbb_sql_setup",  os.path.join(_IRRBB, "sql_setup.py"))
eve_obj         = _load_module_by_path("irrbb_eve_calc_objects", os.path.join(_IRRBB, "eve_calc_objects.py"))

from eba_shock_curves import (                                                 # noqa: E402
    build_all_shocked_curves, _log_interp_daily_df, default_realistic_base_floor_bps,
)
from ns_curve_model import ns_design_matrix, DIEBOLD_LI_TAU                    # noqa: E402
from curve_scenario_bank import build_scenario_bank                            # noqa: E402

OUTPUT_DIR   = os.path.normpath(os.path.join(_HERE, "..", "output"))
REPORT_DATE  = pd.to_datetime("2026-06-30")   # matches extract_params.REPORT_DATE / eve_calc_workflow
CURRENCY     = "PLN"
EVE_FLOOR    = 0.0
DISC_CURVE_MAP = {"PLN": "PLN_disc_curve", "EUR": "EUR_disc_curve", "USD": "USD_disc_curve"}
SCENARIO_IDS = ["base", "par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]  # skip 'own' (not EBA)
SOT_EVE_FLOOR_PCT_T1 = -15.0

_IR_XLSX = os.path.normpath(os.path.join(_IRRBB, "..", "..", "balance_generate", "input_data", "interest_rt.xlsx"))


def _load_rate_maps() -> tuple[dict, dict, dict, dict]:
    """Same source/logic as eve_calc_workflow.py's CAPS_MAP/FLOORS_MAP/COEFF_A_MAP/COEFF_B_MAP."""
    ir_df = pd.read_excel(_IR_XLSX)
    caps = {
        str(int(r["product_code"])): float(r["client_cap"])
        for _, r in ir_df.iterrows()
        if "client_cap" in ir_df.columns and not pd.isna(r.get("client_cap"))
    }
    floors = {
        str(int(r["product_code"])): float(r["client_floor"])
        for _, r in ir_df.iterrows()
        if "client_floor" in ir_df.columns and not pd.isna(r.get("client_floor"))
    }
    coeff_a = {
        str(int(r["product_code"])): float(r["a"])
        for _, r in ir_df.iterrows()
        if "a" in ir_df.columns and not pd.isna(r.get("a")) and float(r["a"]) != 1.0
    }
    coeff_b = {
        str(int(r["product_code"])): float(r["b"]) / 100.0
        for _, r in ir_df.iterrows()
        if "b" in ir_df.columns and not pd.isna(r.get("b")) and float(r["b"]) != 0.0
    }
    return caps, floors, coeff_a, coeff_b


def ns_beta_to_mkt_df(
    beta0: float, beta1: float, beta2: float,
    tau: float = DIEBOLD_LI_TAU,
    n_months: int = 360,
    curve_name: str = "PLN_disc_curve",
) -> pd.DataFrame:
    """Reconstruct a (curve_name, n_days, d_f) discount curve from NS betas."""
    months  = np.arange(1, n_months + 1)
    t_years = months / 12.0
    X       = ns_design_matrix(t_years, tau)
    r_bps   = X @ np.array([beta0, beta1, beta2])
    d_f     = np.exp(-(r_bps / 10_000.0) * t_years)
    n_days  = months * (365.25 / 12.0)
    return pd.DataFrame({"curve_name": curve_name, "n_days": n_days, "d_f": d_f})


def _in_memory_shocked_disc_series(
    shocked_df: pd.DataFrame, scenario_id: str, curve_name: str, report_date: pd.Timestamp,
) -> pd.Series:
    """Build the (curve_name, node_date) -> d_f daily Series that
    eve_calc_objects.compute_eve_shocked_schedule expects, directly from an
    in-memory build_all_shocked_curves() result -- same output format as
    eba_shock_curves.load_irrbb_disc_curve(), but with zero DB round-trip.
    """
    grp = shocked_df[
        (shocked_df["scenario_id"] == scenario_id) & (shocked_df["curve_name"] == curve_name)
    ].sort_values("n_days")
    max_days = int(grp["n_days"].max())
    daily = _log_interp_daily_df(grp["n_days"].to_numpy(), grp["d_f"].to_numpy(), max_days)
    daily["node_date"] = report_date + pd.to_timedelta(daily["n_days"], unit="D")
    daily.insert(0, "curve_name", curve_name)
    return daily.set_index(["curve_name", "node_date"])["d_f"].sort_index()


def reprice_anchor_exact(
    all_beh: pd.DataFrame,
    caps_map: dict, floors_map: dict, coeff_a_map: dict, coeff_b_map: dict,
    beta0: float, beta1: float, beta2: float,
    tau: float = DIEBOLD_LI_TAU,
    currency: str = CURRENCY,
    report_date: pd.Timestamp = REPORT_DATE,
) -> dict[str, float]:
    """Exact (Way A) EVE for one anchor curve, ALL scenarios incl. base computed
    against the anchor's own level (not today's). Returns {scenario: EVE PLN
    (base) / delta_EVE PLN (shocks)}, whole book.
    """
    curve_name = DISC_CURVE_MAP[currency]
    mkt_df  = ns_beta_to_mkt_df(beta0, beta1, beta2, tau, curve_name=curve_name)
    shocked_df = build_all_shocked_curves(mkt_df, report_date, currency, nii_floor_rate=EVE_FLOOR,
                                           base_floor_bps_fn=default_realistic_base_floor_bps)

    eve_by_scen: dict[str, float] = {}
    for scenario_id in SCENARIO_IDS:
        shocked_disc = _in_memory_shocked_disc_series(shocked_df, scenario_id, curve_name, report_date)
        _, summ = eve_obj.compute_eve_shocked(
            all_beh,
            shocked_disc_df = shocked_disc,
            disc_curve_map  = DISC_CURVE_MAP,
            scenario_id     = scenario_id,
            eve_floor_rate  = EVE_FLOOR,
            caps_map        = caps_map,
            floors_map      = floors_map,
            coeff_a_map     = coeff_a_map,
            coeff_b_map     = coeff_b_map,
        )
        eve_by_scen[scenario_id] = float(summ["pv_total"].sum())

    eve_base = eve_by_scen["base"]
    return {
        s: (eve_base if s == "base" else eve_by_scen[s] - eve_base)
        for s in SCENARIO_IDS
    }


def run_all_anchors_exact() -> pd.DataFrame:
    print("Loading run-off CF schedules (curve-independent, queried once)...")
    all_beh = irrbb_sql_setup.load_all_beh_schedules(REPORT_DATE)
    try:
        swap_df = irrbb_sql_setup.load_all_swap_beh_schedules(REPORT_DATE)
    except Exception:
        swap_df = pd.DataFrame()
    if not swap_df.empty:
        all_beh = pd.concat([all_beh, swap_df], ignore_index=True)
    print(f"  {len(all_beh):,} CF rows across {all_beh['schedule_id'].nunique()} schedules")

    caps_map, floors_map, coeff_a_map, coeff_b_map = _load_rate_maps()
    t1 = irrbb_sql_setup.load_tier1_capital(REPORT_DATE)
    print(f"  Tier-1 capital: {t1/1e6:,.1f}M PLN")

    anchors, _bank = build_scenario_bank()

    rows = []
    for _, anchor in anchors.iterrows():
        print(f"  Repricing anchor (EXACT): {anchor['shape']:>8s} / {anchor['level']:<6s}  "
              f"(anchor_date={anchor['anchor_date'].date()})...")
        totals = reprice_anchor_exact(
            all_beh, caps_map, floors_map, coeff_a_map, coeff_b_map,
            anchor["beta0"], anchor["beta1"], anchor["beta2"],
        )
        eve_base = totals["base"]
        for scen in SCENARIO_IDS:
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
    xlsx_path = os.path.join(out_dir, "anchor_eve_exact.xlsx")
    worst_per_anchor = (
        results.loc[results.groupby(["shape", "level"])["delta_eve_pct_t1"].idxmin()]
        .reset_index(drop=True)
    )
    with pd.ExcelWriter(xlsx_path) as writer:
        results.to_excel(writer, sheet_name="all_scenarios", index=False)
        worst_per_anchor.to_excel(writer, sheet_name="worst_shock_per_anchor", index=False)
    print(f"\nSaved {xlsx_path}")


if __name__ == "__main__":
    results = run_all_anchors_exact()
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
