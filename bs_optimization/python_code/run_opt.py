"""run_opt.py
============cla
Run balance sheet optimization from Excel config.

Usage:
    python run_opt.py
"""
import os, sys

_HERE    = os.path.dirname(os.path.abspath(__file__))
_OPTPREP = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "python_code"))
for _p in (_HERE, _OPTPREP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bs_vector import BalanceSheetParams, CohortRates
from bs_optimizer import optimize_nii
from optimizer_io import load_optimizer_config

_PARAMS = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "output", "product_params.npz"))
_XL     = os.path.normpath(os.path.join(_HERE, "..", "input", "optimizer_config.xlsx"))

print("Loading balance sheet params...")
params = BalanceSheetParams.load(_PARAMS)
cr     = CohortRates.load(_PARAMS)

print("Loading optimizer config from Excel...")
cfg = load_optimizer_config(_XL, params)
print(f"  mode          = {cfg.mode}")
print(f"  max_shift     = {cfg.max_shift * 100:.1f}%")
print(f"  sot_eve_floor = {cfg.sot_eve_floor}%  (EVE buffer +{cfg.sot_eve_buffer}% T1)")
print(f"  sot_nii_floor = {cfg.sot_nii_floor}%  (NII buffer +{cfg.sot_nii_buffer}% T1)")
print(f"  include_irs   = {cfg.include_irs}  ({'hedged view' if cfg.include_irs else 'unhedged — banking book only'})")
print(f"  min_t1_rwa    = {cfg.min_t1_rwa*100:.1f}%  (0% = no constraint)")
print(f"  fixed products: {sorted(cfg.fixed_products)}")

print(f"\nRunning [{cfg.mode}] optimization (objective: Economic Profit = NII - EL - CoC)...")
result = optimize_nii(cfg, params, cr)
result.print_summary()
