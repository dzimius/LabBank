"""build_ftp_rates.py
=====================
One-time (re-run whenever product_params.npz or curve_tensors.npz change)
script to compute and cache FTP rates -- see ftp_store.py for the formula.

Writes both:
  - optimize_prep/output/ftp_rates.npz  (fast-load cache, incl. market_rt/
    liq_spread breakdown, used by load_ftp_rates())
  - SQL opt_prep.ftp_rates              (report_date, cohort_id, market_rt,
    liq_spread, ftp_rate -- queryable breakdown, one full reset+reload per run)

Usage:
    python build_ftp_rates.py
"""
import os

import pandas as pd

from bs_vector import BalanceSheetParams, CurveTensors
from ftp_store import compute_ftp_components, REPORT_DATE, save_ftp_rates
from sql_setup import reset_ftp_rates, write_ftp_rates

_HERE    = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.normpath(os.path.join(_HERE, "..", "output"))
_PARAMS  = os.path.join(_OUT_DIR, "product_params.npz")
_CURVES  = os.path.join(_OUT_DIR, "curve_tensors.npz")

print("Loading balance sheet params + curve tensors...")
params = BalanceSheetParams.load(_PARAMS)
curves = CurveTensors.load(_CURVES)

print("Computing FTP rates (market + liquidity component)...")
market_rt, liq_spread = compute_ftp_components(params, curves)
ftp_rate = market_rt + liq_spread

n_nonzero = int((ftp_rate > 0).sum())
print(f"  {n_nonzero}/{len(ftp_rate)} cohorts priced (rest are equity, FTP=0)")
print(f"  FTP range: {ftp_rate[ftp_rate > 0].min()*100:.2f}% .. {ftp_rate[ftp_rate > 0].max()*100:.2f}%")

save_ftp_rates(ftp_rate, params.cohort_id, market_rt=market_rt, liq_spread=liq_spread)

print("Writing opt_prep.ftp_rates...")
df = pd.DataFrame({
    "report_date": REPORT_DATE.date(),
    "cohort_id":   params.cohort_id,
    "market_rt":   market_rt,
    "liq_spread":  liq_spread,
    "ftp_rate":    ftp_rate,
})
reset_ftp_rates()
write_ftp_rates(df)
print(f"  {len(df)} rows written to opt_prep.ftp_rates")
