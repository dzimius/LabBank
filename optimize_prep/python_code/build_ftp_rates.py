"""build_ftp_rates.py
=====================
One-time (re-run whenever product_params.npz or curve_tensors.npz change)
script to compute and cache FTP rates -- see ftp_store.py for the formula.

Usage:
    python build_ftp_rates.py
"""
import os

from bs_vector import BalanceSheetParams, CurveTensors
from ftp_store import compute_ftp_rates, save_ftp_rates

_HERE    = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.normpath(os.path.join(_HERE, "..", "output"))
_PARAMS  = os.path.join(_OUT_DIR, "product_params.npz")
_CURVES  = os.path.join(_OUT_DIR, "curve_tensors.npz")

print("Loading balance sheet params + curve tensors...")
params = BalanceSheetParams.load(_PARAMS)
curves = CurveTensors.load(_CURVES)

print("Computing FTP rates (market + liquidity component)...")
ftp_rate = compute_ftp_rates(params, curves)

n_nonzero = int((ftp_rate > 0).sum())
print(f"  {n_nonzero}/{len(ftp_rate)} cohorts priced (rest are equity, FTP=0)")
print(f"  FTP range: {ftp_rate[ftp_rate > 0].min()*100:.2f}% .. {ftp_rate[ftp_rate > 0].max()*100:.2f}%")

save_ftp_rates(ftp_rate, params.cohort_id)
