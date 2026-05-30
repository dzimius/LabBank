"""opt_prep_workflow.py
======================
Single entry point for the optimize_prep pipeline.

Run order
---------
  Step 1 — extract_params   : build product_params.npz + SQL opt_prep.product_params
  Step 2 — extract_curves   : build curve_tensors.npz  + SQL opt_prep.monthly_curves
  Step 3 — accuracy_check   : compare fast metrics vs exact pipeline, write Excel report

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
  optimize_prep/output/approx_accuracy_report.xlsx   ← main visible result
"""
from __future__ import annotations

import sys
import os
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

    _step("Step 1/3 — Extract yield curve tensors")
    build_curve_tensors()

    _step("Step 2/3 — Extract product parameters (uses curve tensors)")
    build_product_params()

    _step("Step 3/3 — Accuracy check (fast vs exact metrics)")
    run_accuracy_check()

    elapsed = time.time() - t0
    print(f"\nopt_prep pipeline complete in {elapsed:.1f}s")
    print("Main output: optimize_prep/output/approx_accuracy_report.xlsx")


if __name__ == "__main__":
    run_opt_prep()
