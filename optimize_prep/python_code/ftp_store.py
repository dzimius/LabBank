"""ftp_store.py
==============
FTP (Funds Transfer Pricing) rate per cohort.

    ftp_rate = market_rate(tenor) + liquidity_spread(tenor)

Floating cohorts REFIX to the CURRENT market rate: today's (report-date)
base-scenario forward rate from CurveTensors, at tenor = repricing_tenor_m
(the fixing frequency, e.g. 3M) -- matches how the loan's own client rate
actually refixes.

Fixed cohorts are LOCKED AT ORIGINATION and never move afterward (2026-07-10
correction -- match-funded convention, per user feedback): treasury's own
funding for a fixed-rate loan was raised back when the loan was originated,
for the loan's FULL ORIGINAL term, so the transfer price reflects what that
funding actually cost then, not what today's curve implies for the (now
shorter) remaining life. Concretely: original_tenor_months =
repricing_tenor_m (remaining, as of report_date) + elapsed months since
origination (parsed from cohort_id's "..._{start_year}_{start_month}"
suffix); the rate is fit from the historical NS curve AT that origination
date, for that ORIGINAL tenor -- same historical-curve machinery
swap_ladder.py already uses for its seasoned buckets (ns_curve_model.py),
reused here rather than duplicated.

liquidity_spread(t) : linear 0% at t=0 months -> 0.5% at t=120 months (10Y),
                      capped at 0.5% beyond. Uses the SAME tenor basis as the
                      market-rate component for that cohort (current
                      remaining tenor for floating, locked original tenor
                      for fixed) -- so it's locked/refixes in step with it.

Deliberately NOT threaded through the SQL extraction pipeline
(extract_params.py) -- computed once from data already in product_params.npz
plus the already-cached curve_tensors.npz (floating) / the historical NS
panel (fixed), and cached the same way bias_store.py caches
accuracy_check.py's corrections (see that module for the precedent).
Re-running the full balance-generation ETL for one new field isn't
warranted here.

Used by bs_optimizer.py / joint_optimizer.py's margin_unit_rate() to derive
margin-over-FTP income (replacing client-rate NII in the EP objective).
NOT used by compute_nii_cf_all / compute_eve_cf_fast_all -- the EVE/NII SOT
regulatory calculation stays client-rate based, unaffected by this.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
FTP_PATH = os.path.normpath(os.path.join(_HERE, "..", "output", "ftp_rates.npz"))

# ns_curve_model.py (historical Nelson-Siegel fit) lives alongside this file
# (moved here from bs_optimization/python_code 2026-08-14, same move as
# _ProductMap/ep_fast.py -- see that commit) -- same NS-fit machinery
# swap_ladder.py uses for its seasoned buckets, reused rather than duplicated.

LIQUIDITY_CAP_BPS    = 50.0    # 0.50% at 10Y+
LIQUIDITY_CAP_MONTHS = 120.0   # 10Y
REPORT_DATE          = pd.Timestamp("2026-06-30")   # matches extract_params.REPORT_DATE


def _liquidity_spread(tenor_m: np.ndarray) -> np.ndarray:
    return (np.clip(tenor_m, 0.0, LIQUIDITY_CAP_MONTHS) / LIQUIDITY_CAP_MONTHS
            * LIQUIDITY_CAP_BPS / 10000.0)


def _elapsed_months_since_origination(cohort_id: str, report_date: pd.Timestamp = REPORT_DATE) -> int:
    """cohort_id format: '{product_code}_{bs_side}_{currency}_{start_year}_{start_month}'."""
    parts = cohort_id.split("_")
    start_year, start_month = int(parts[-2]), int(parts[-1])
    return (report_date.year - start_year) * 12 + (report_date.month - start_month)


def _historical_zero_rate(ns_ts, tenor_years: float, months_ago: int, tau: float) -> float:
    """Historical NS zero rate at `tenor_years`, fit at (report_date - months_ago)
    -- identical formula to swap_ladder._historical_zero_rate, reused for FTP."""
    from ns_curve_model import ns_design_matrix

    origin_date = REPORT_DATE - pd.DateOffset(months=int(months_ago))
    pos = ns_ts.index.searchsorted(origin_date)
    pos = int(min(max(pos, 0), len(ns_ts) - 1))
    beta = ns_ts.iloc[pos][["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
    X = ns_design_matrix(np.array([tenor_years]), tau)
    rate_bps = float((X @ beta)[0])
    return rate_bps / 10000.0


def compute_ftp_rates(params, curve_tensors) -> np.ndarray:
    """(n,) FTP rate per cohort, decimal (e.g. 0.035 = 3.5%). Zero for equity
    (equity has no client rate / FTP concept in this model -- its cost is
    already captured separately via cost of capital)."""
    n = len(params.cohort_id)
    ftp = np.zeros(n, dtype=float)
    base_idx = curve_tensors.scenario_index("base")
    is_float = (params.rate_type == "V") & ~params.is_equity
    is_fixed = (params.rate_type == "F") & ~params.is_equity

    n_other = int((~params.is_equity & ~is_float & ~is_fixed).sum())
    if n_other > 0:
        print(f"  [ftp_store] {n_other} non-equity cohort(s) have no F/V rate_type "
              f"(e.g. IRS product '0000', or 'single-row' aggregate products) -- "
              f"FTP=0 for these, margin == client rate unchanged")

    # ── Floating: refixes to today's (report-date) curve at the fixing tenor ──
    for ccy in np.unique(params.currency):
        ccy_mask = (params.currency == ccy) & is_float
        if not ccy_mask.any():
            continue
        fwd = curve_tensors.fwd_rates[base_idx, curve_tensors.currency_index(str(ccy))]
        tenor_m = np.clip(params.repricing_tenor_m[ccy_mask], 1.0, 360.0)
        idx = np.clip(np.round(tenor_m).astype(int) - 1, 0, 359)
        ftp[ccy_mask] = fwd[idx] + _liquidity_spread(tenor_m)

    # ── Fixed: locked at origination, historical curve, original tenor ────────
    fixed_idx = np.where(is_fixed)[0]
    if len(fixed_idx) > 0:
        from ns_curve_model import DIEBOLD_LI_TAU, fit_ns_timeseries, load_historical_panel

        panel = load_historical_panel()
        ns_ts = fit_ns_timeseries(panel, DIEBOLD_LI_TAU)
        for i in fixed_idx:
            elapsed_m = _elapsed_months_since_origination(str(params.cohort_id[i]))
            original_tenor_m = max(float(params.repricing_tenor_m[i]) + elapsed_m, 1.0)
            rate = _historical_zero_rate(ns_ts, original_tenor_m / 12.0, elapsed_m, DIEBOLD_LI_TAU)
            ftp[i] = rate + float(_liquidity_spread(np.array([original_tenor_m]))[0])

    return ftp


def margin_unit_rate(nii_unit_rate: np.ndarray, ftp_rate: np.ndarray, bs_side: np.ndarray) -> np.ndarray:
    """Margin-over-FTP income per unit balance, replacing client-rate NII in
    the EP objective/reporting. nii_unit_rate is already signed by side
    (positive for assets, negative for liabilities); this keeps that
    convention: assets lose ftp_rate (pay it to treasury), liabilities gain
    it (treasury absorbs part of the funding cost). Zero FTP (unloaded cache,
    or equity) degrades gracefully to margin_unit_rate == nii_unit_rate.
    """
    sign = np.where(bs_side == "A", 1.0, np.where(bs_side == "L", -1.0, 0.0))
    return nii_unit_rate - sign * ftp_rate


def save_ftp_rates(ftp_rate: np.ndarray, cohort_id: np.ndarray, path: str = FTP_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, ftp_rate=ftp_rate, cohort_id=cohort_id)
    print(f"FTP rates saved: {path}  ({len(cohort_id)} cohorts, "
          f"mean {float(np.mean(ftp_rate[ftp_rate > 0]))*100:.2f}%)")


def load_ftp_rates(cohort_id: np.ndarray, path: str = FTP_PATH) -> np.ndarray | None:
    """Load cached FTP rates, re-aligned to the given cohort_id order.
    Returns None if the cache is missing or its cohort set doesn't match
    (e.g. product_params.npz was regenerated) -- callers should degrade
    gracefully (zero FTP) rather than fail when this happens."""
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    cached_ids = data["cohort_id"]
    if len(cached_ids) != len(cohort_id) or not np.array_equal(cached_ids, cohort_id):
        return None
    return data["ftp_rate"].astype(float)
