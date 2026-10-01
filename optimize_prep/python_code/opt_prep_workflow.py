"""opt_prep_workflow.py
======================
Single entry point for the optimize_prep pipeline.

Run order
---------
  Step 1 — extract_params   : build product_params.npz + SQL opt_prep.product_params
  Step 2 — extract_curves   : build curve_tensors.npz  + SQL opt_prep.monthly_curves
  Step 3 — build_ftp_rates  : FTP rate per cohort (ftp_rates.npz + SQL opt_prep.ftp_rates)
  Step 4 — accuracy_check   : compare fast metrics vs exact pipeline, write Excel report

Prerequisites (must have been run first)
-----------------------------------------
  balance_generate workflow
  ir_derivatives workflow
  irrbb_calc/python_code/nii_calc_workflow.py
  irrbb_calc/python_code/eve_calc_workflow.py
  liq_calc workflow  (for LCR/NSFR accuracy check)

Outputs
-------
  optimize_prep/output/product_params.npz
  optimize_prep/output/params_inspection.xlsx
  optimize_prep/output/curve_tensors.npz
  optimize_prep/output/curves_inspection.xlsx
  optimize_prep/output/ftp_rates.npz
  optimize_prep/output/approx_accuracy_report.xlsx   ← main visible result
"""
from __future__ import annotations

import sys
import os
import runpy
import time

# ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from optimize_prep.python_code.extract_params import build_product_params
from optimize_prep.python_code.extract_curves import build_curve_tensors
from optimize_prep.python_code.accuracy_check import run_accuracy_check


def _step(label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")


def run_opt_prep() -> None:
    t0 = time.time()

    _step("Step 1/4 — Extract yield curve tensors")
    build_curve_tensors()

    _step("Step 2/4 — Extract product parameters (uses curve tensors)")
    build_product_params()

    # ftp_rates.npz is keyed on the cohort set: a regenerated product_params.npz
    # (new balance sheet draw) makes the old cache stale, and the optimizers then
    # silently fall back to FTP = 0 -- so rebuild it on every run.
    _step("Step 3/4 — FTP rates per cohort (uses product params + curve tensors)")
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    runpy.run_path(os.path.join(_here, "build_ftp_rates.py"), run_name="__main__")

    _step("Step 4/4 — Accuracy check (fast vs exact metrics)")
    run_accuracy_check()

    elapsed = time.time() - t0
    print(f"\nopt_prep pipeline complete in {elapsed:.1f}s")
    print("Main output: optimize_prep/output/approx_accuracy_report.xlsx")


if __name__ == "__main__":
    run_opt_prep()
