"""accuracy_check.py
====================
Compare fast metric approximations against exact pipeline results.

Run this after:
  1. extract_params.py     (builds product_params.npz)
  2. extract_curves.py     (builds curve_tensors.npz)
  3. All exact pipeline workflows (NII, EVE, LCR, NSFR)

Produces a comparison report showing relative errors per metric and scenario.
If errors are outside acceptable tolerances, the approximation needs tuning
before the optimizer results can be trusted.

Acceptable tolerances (soft targets)
-------------------------------------
  NII base          : < 5%
  delta_NII per scenario : < 10%
  EVE base          : < 5%
  delta_EVE per scenario : < 10%
  LCR               : < 2%
  NSFR              : < 2%

Outputs
-------
  optimize_prep/output/approx_accuracy_report.xlsx
  Console summary with pass/fail per metric
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ── path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "irrbb_calc",  "python_code"))
sys.path.insert(0, os.path.join(ROOT_DIR, "liq_calc",    "python_code"))
sys.path.insert(0, BASE_DIR)

from metrics import load_params_and_curves, compute_all_metrics

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

REPORT_DATE  = pd.to_datetime("2024-12-31")
TOTAL_ASSETS = 10_000_000_000
OUT_PATH     = os.path.join(BASE_DIR, "..", "output", "approx_accuracy_report.xlsx")

# Tolerances for pass/fail
TOL_NII_BASE      = 0.05
TOL_DELTA_NII     = 0.15
TOL_EVE_BASE      = 0.05
TOL_DELTA_EVE     = 0.15
TOL_LCR           = 0.02
TOL_NSFR          = 0.02


def _pct_error(approx: float, exact: float) -> float:
    """Relative error in %. Returns NaN if exact is 0."""
    if abs(exact) < 1.0:
        return float("nan")
    return (approx - exact) / abs(exact) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Load exact results from SQL
# ─────────────────────────────────────────────────────────────────────────────

def _load_exact_nii() -> dict[str, float]:
    """Load exact NII per scenario from irrbb.nii_results."""
    query = text("""
        SELECT scenario_id, SUM(nii_total) AS nii_total
        FROM irrbb.nii_results
        WHERE report_date = :rd
        GROUP BY scenario_id
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    return dict(zip(df["scenario_id"], df["nii_total"]))


def _load_exact_eve() -> dict[str, float]:
    """Load exact EVE per scenario from irrbb.eve_results."""
    query = text("""
        SELECT scenario_id, SUM(pv_total) AS pv_total
        FROM irrbb.eve_results
        WHERE report_date = :rd
        GROUP BY scenario_id
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    return dict(zip(df["scenario_id"], df["pv_total"]))


def _load_exact_lcr_nsfr() -> pd.DataFrame:
    """Load exact LCR/NSFR from results.lcr_nsfr."""
    query = text("""
        SELECT currency, lcr, nsfr
        FROM results.lcr_nsfr
        WHERE report_date = :rd
    """)
    return pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_accuracy_check() -> None:
    print("Loading fast metric parameters...")
    params, curves = load_params_and_curves()

    # Use current weights (current balance sheet = exact pipeline calibration point)
    w       = params.current_weights()
    metrics = compute_all_metrics(w, params, curves, TOTAL_ASSETS)

    print("Loading exact results from SQL...")
    exact_nii      = _load_exact_nii()
    exact_eve      = _load_exact_eve()
    exact_liq      = _load_exact_lcr_nsfr()

    rows = []

    # ── NII base ─────────────────────────────────────────────────────────────
    exact_nii_base = float(exact_nii.get("base", float("nan")))
    approx_nii_base = metrics.nii_base
    pct_err = _pct_error(approx_nii_base, exact_nii_base)
    passed  = abs(pct_err) < TOL_NII_BASE * 100 if not np.isnan(pct_err) else False
    rows.append({
        "metric": "NII_base", "scenario": "base", "currency": "ALL",
        "exact": exact_nii_base, "approx": approx_nii_base,
        "pct_error": pct_err, "tolerance_pct": TOL_NII_BASE * 100, "pass": passed,
    })
    print(f"  NII base  exact={exact_nii_base:+,.0f}  approx={approx_nii_base:+,.0f}  "
          f"err={pct_err:+.2f}%  {'PASS' if passed else 'FAIL'}")

    # ── delta NII per scenario ────────────────────────────────────────────────
    exact_nii_base_val = exact_nii.get("base", 0.0)
    for scen in params.scenario_ids:
        scen = str(scen)
        exact_shocked = float(exact_nii.get(scen, float("nan")))
        exact_delta   = exact_shocked - exact_nii_base_val if not np.isnan(exact_shocked) else float("nan")
        approx_delta  = metrics.delta_nii.get(scen, float("nan"))
        pct_err = _pct_error(approx_delta, exact_delta)
        passed  = abs(pct_err) < TOL_DELTA_NII * 100 if not np.isnan(pct_err) else False
        rows.append({
            "metric": "delta_NII", "scenario": scen, "currency": "ALL",
            "exact": exact_delta, "approx": approx_delta,
            "pct_error": pct_err, "tolerance_pct": TOL_DELTA_NII * 100, "pass": passed,
        })
        print(f"  delta_NII {scen:<10}  exact={exact_delta:+,.0f}  approx={approx_delta:+,.0f}  "
              f"err={pct_err:+.2f}%  {'PASS' if passed else 'FAIL'}")

    # ── EVE base ─────────────────────────────────────────────────────────────
    exact_eve_base  = float(exact_eve.get("base", float("nan")))
    approx_eve_base = metrics.eve_base
    pct_err = _pct_error(approx_eve_base, exact_eve_base)
    passed  = abs(pct_err) < TOL_EVE_BASE * 100 if not np.isnan(pct_err) else False
    rows.append({
        "metric": "EVE_base", "scenario": "base", "currency": "ALL",
        "exact": exact_eve_base, "approx": approx_eve_base,
        "pct_error": pct_err, "tolerance_pct": TOL_EVE_BASE * 100, "pass": passed,
    })
    print(f"  EVE base  exact={exact_eve_base:+,.0f}  approx={approx_eve_base:+,.0f}  "
          f"err={pct_err:+.2f}%  {'PASS' if passed else 'FAIL'}")

    # ── delta EVE per scenario ────────────────────────────────────────────────
    exact_eve_base_val = exact_eve.get("base", 0.0)
    for scen in params.scenario_ids:
        scen = str(scen)
        exact_shocked = float(exact_eve.get(scen, float("nan")))
        exact_delta   = exact_shocked - exact_eve_base_val if not np.isnan(exact_shocked) else float("nan")
        approx_delta  = metrics.delta_eve.get(scen, float("nan"))
        pct_err = _pct_error(approx_delta, exact_delta)
        passed  = abs(pct_err) < TOL_DELTA_EVE * 100 if not np.isnan(pct_err) else False
        rows.append({
            "metric": "delta_EVE", "scenario": scen, "currency": "ALL",
            "exact": exact_delta, "approx": approx_delta,
            "pct_error": pct_err, "tolerance_pct": TOL_DELTA_EVE * 100, "pass": passed,
        })
        print(f"  delta_EVE {scen:<10}  exact={exact_delta:+,.0f}  approx={approx_delta:+,.0f}  "
              f"err={pct_err:+.2f}%  {'PASS' if passed else 'FAIL'}")

    # ── LCR ──────────────────────────────────────────────────────────────────
    if not exact_liq.empty:
        for _, r in exact_liq.iterrows():
            ccy         = r["currency"]
            exact_lcr   = float(r["lcr"]) if pd.notna(r["lcr"]) else float("nan")
            approx_lcr  = metrics.lcr.get(ccy, float("nan"))
            pct_err = _pct_error(approx_lcr, exact_lcr)
            passed  = abs(pct_err) < TOL_LCR * 100 if not np.isnan(pct_err) else False
            rows.append({
                "metric": "LCR", "scenario": "n/a", "currency": ccy,
                "exact": exact_lcr, "approx": approx_lcr,
                "pct_error": pct_err, "tolerance_pct": TOL_LCR * 100, "pass": passed,
            })
            print(f"  LCR  {ccy}  exact={exact_lcr:.4f}  approx={approx_lcr:.4f}  "
                  f"err={pct_err:+.2f}%  {'PASS' if passed else 'FAIL'}")

            exact_nsfr  = float(r["nsfr"]) if pd.notna(r["nsfr"]) else float("nan")
            approx_nsfr = metrics.nsfr.get(ccy, float("nan"))
            pct_err = _pct_error(approx_nsfr, exact_nsfr)
            passed  = abs(pct_err) < TOL_NSFR * 100 if not np.isnan(pct_err) else False
            rows.append({
                "metric": "NSFR", "scenario": "n/a", "currency": ccy,
                "exact": exact_nsfr, "approx": approx_nsfr,
                "pct_error": pct_err, "tolerance_pct": TOL_NSFR * 100, "pass": passed,
            })
            print(f"  NSFR {ccy}  exact={exact_nsfr:.4f}  approx={approx_nsfr:.4f}  "
                  f"err={pct_err:+.2f}%  {'PASS' if passed else 'FAIL'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    report_df   = pd.DataFrame(rows)
    n_pass      = report_df["pass"].sum()
    n_total     = len(report_df)
    n_fail      = n_total - n_pass
    print(f"\nAccuracy summary: {n_pass}/{n_total} checks passed, {n_fail} failed")

    if n_fail > 0:
        print("\nFailed checks:")
        for _, r in report_df[~report_df["pass"]].iterrows():
            print(f"  {r['metric']:15s} {r['scenario']:12s} {r['currency']:5s}  "
                  f"err={r['pct_error']:+.2f}%  tol={r['tolerance_pct']:.1f}%")

    # ── Write Excel ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        report_df.to_excel(writer, sheet_name="accuracy_check", index=False)
    print(f"\nAccuracy report written to {OUT_PATH}")


if __name__ == "__main__":
    run_accuracy_check()
