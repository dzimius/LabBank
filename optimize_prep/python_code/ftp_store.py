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

NMD (current accounts, savings accounts -- product_name in NMD_PRODUCT_NAMES)
have no contractual repricing tenor to refix to or lock at origination, so
neither the floating nor the fixed convention above applies. Simple-example
convention (2026-08-15): market_rt = trailing NMD_LOOKBACK_MONTHS (12) simple
moving average of the 1Y point on the historical PLN fixing panel -- smooths
out day-to-day market noise the way a real replicating-portfolio NMD model
would, without building a full replicating-portfolio ladder here. RECOMPUTED
every report_date (the whole trailing window shifts), unlike fixed cohorts'
locked-at-origination rate. liq_spread uses the same liquidity_spread()
formula as everything else, at the single-row default repricing_tenor_m
(currently 12M for these products, from extract_params.py) -- not a
behavioural NMD duration, consistent with how every other single-row product
is treated.

liquidity_spread(t) : linear 0% at t=0 months -> 0.5% at t=120 months (10Y),
                      capped at 0.5% beyond. Uses the SAME tenor basis as the
                      market-rate component for that cohort (current
                      remaining tenor for floating, locked original tenor
                      for fixed, single-row default for NMD) -- so it's
                      locked/refixes in step with it.

Deliberately NOT threaded through the SQL extraction pipeline
(extract_params.py) -- computed once from data already in product_params.npz
plus the already-cached curve_tensors.npz (floating) / the historical NS
panel (fixed), and cached the same way bias_store.py caches
accuracy_check.py's corrections (see that module for the precedent).
Re-running the full balance-generation ETL for one new field isn't
warranted here.

Used by bs_optimizer.py / joint_optimizer.py's margin_unit_rate() to derive
margin-over-FTP income (replacing client-rate NII in the EP objective), and
by equity_benefit_unit_rate() to derive the offsetting FTP credit for the
capital-funded share of each cohort's balance (see that function's docstring
for why the credit is needed).
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

# NMD (current/savings accounts) -- identified by product_name, not product_code,
# to match the same naming convention b_s_gen_objects.compute_deposit_client_rt
# already uses (its 'deposit_names' set), rather than a separate hardcoded
# product-code allowlist.
NMD_PRODUCT_NAMES    = ("current_account", "saving_account")
NMD_LOOKBACK_MONTHS  = 12      # trailing window for the 1Y-rate moving average
NMD_MARKET_TENOR     = "1Y"    # historical panel column (see ns_curve_model.load_historical_panel)


def _liquidity_spread(tenor_m: np.ndarray) -> np.ndarray:
    return (np.clip(tenor_m, 0.0, LIQUIDITY_CAP_MONTHS) / LIQUIDITY_CAP_MONTHS
            * LIQUIDITY_CAP_BPS / 10000.0)


def _trailing_market_rate_avg(
    panel: pd.DataFrame,
    tenor_col: str = NMD_MARKET_TENOR,
    report_date: pd.Timestamp = REPORT_DATE,
    months: int = NMD_LOOKBACK_MONTHS,
) -> float:
    """Simple moving average of `tenor_col`'s historical rate (panel values are
    bps, see load_historical_panel) over the trailing `months` calendar months
    up to and including `report_date`. Returns decimal (e.g. 0.035 = 3.5%)."""
    window_start = report_date - pd.DateOffset(months=months)
    window = panel.loc[(panel.index > window_start) & (panel.index <= report_date), tenor_col]
    if window.empty:
        raise ValueError(f"No historical '{tenor_col}' observations in the trailing "
                          f"{months}M window ending {report_date.date()} -- check fixing_input.xlsx coverage")
    return float(window.mean()) / 10000.0


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


def floating_ftp_components(params, is_float: np.ndarray, curve_tensors) -> tuple[np.ndarray, np.ndarray]:
    """(market_rt, liq_spread), each (n,), for FLOATING cohorts only, refixed
    to `curve_tensors`'s base scenario at each cohort's fixing tenor. Zero
    elsewhere (fixed/equity/other cohorts -- caller combines with those
    separately). Split out of floating_ftp_rate() so callers that need the
    market-rate / liquidity-spread breakdown (e.g. SQL persistence) don't
    have to re-derive it from the summed rate."""
    n = len(params.cohort_id)
    market_rt = np.zeros(n, dtype=float)
    liq_spread = np.zeros(n, dtype=float)
    base_idx = curve_tensors.scenario_index("base")
    for ccy in np.unique(params.currency):
        ccy_mask = (params.currency == ccy) & is_float
        if not ccy_mask.any():
            continue
        fwd = curve_tensors.fwd_rates[base_idx, curve_tensors.currency_index(str(ccy))]
        tenor_m = np.clip(params.repricing_tenor_m[ccy_mask], 1.0, 360.0)
        idx = np.clip(np.round(tenor_m).astype(int) - 1, 0, 359)
        market_rt[ccy_mask] = fwd[idx]
        liq_spread[ccy_mask] = _liquidity_spread(tenor_m)
    return market_rt, liq_spread


def floating_ftp_rate(params, is_float: np.ndarray, curve_tensors) -> np.ndarray:
    """(n,) FTP market-rate + liquidity-spread component for FLOATING cohorts
    only. Factored out of compute_ftp_rates() so a hypothetical curve can
    reuse this exact floating-refix logic (see sandbox/app.py's Finance
    Metrics tab) without re-running the curve-independent fixed-cohort
    historical-NS-fit branch below on every Streamlit rerun -- fixed cohorts
    are locked at origination and genuinely don't move with a hypothetical
    CURRENT-environment curve."""
    market_rt, liq_spread = floating_ftp_components(params, is_float, curve_tensors)
    return market_rt + liq_spread


def compute_ftp_components(params, curve_tensors) -> tuple[np.ndarray, np.ndarray]:
    """(market_rt, liq_spread), each (n,), decimal, per cohort. Zero for
    equity (equity has no client rate / FTP concept in this model -- its
    cost is already captured separately via cost of capital). Split out of
    compute_ftp_rates() so callers (e.g. build_ftp_rates.py's SQL write)
    can persist market_rt/liq_spread as separate columns alongside the
    summed ftp_rate, rather than only the combined figure."""
    n = len(params.cohort_id)
    market_rt = np.zeros(n, dtype=float)
    liq_spread = np.zeros(n, dtype=float)
    is_float = (params.rate_type == "V") & ~params.is_equity
    is_fixed = (params.rate_type == "F") & ~params.is_equity
    is_nmd = np.isin(params.product_name.astype(str), NMD_PRODUCT_NAMES)

    n_other = int((~params.is_equity & ~is_float & ~is_fixed & ~is_nmd).sum())
    if n_other > 0:
        print(f"  [ftp_store] {n_other} non-equity cohort(s) have no F/V rate_type "
              f"(e.g. IRS product '0000', or 'single-row' aggregate products) -- "
              f"FTP=0 for these, margin == client rate unchanged")

    # ── Floating: refixes to today's (report-date) curve at the fixing tenor ──
    f_market, f_liq = floating_ftp_components(params, is_float, curve_tensors)
    market_rt += f_market
    liq_spread += f_liq

    # ── Fixed / NMD both need the historical fixing panel -- load once ────────
    fixed_idx = np.where(is_fixed)[0]
    nmd_idx = np.where(is_nmd)[0]
    if len(fixed_idx) > 0 or len(nmd_idx) > 0:
        from ns_curve_model import DIEBOLD_LI_TAU, fit_ns_timeseries, load_historical_panel

        panel = load_historical_panel()

        # ── Fixed: locked at origination, historical curve, original tenor ──
        if len(fixed_idx) > 0:
            ns_ts = fit_ns_timeseries(panel, DIEBOLD_LI_TAU)
            for i in fixed_idx:
                elapsed_m = _elapsed_months_since_origination(str(params.cohort_id[i]))
                original_tenor_m = max(float(params.repricing_tenor_m[i]) + elapsed_m, 1.0)
                rate = _historical_zero_rate(ns_ts, original_tenor_m / 12.0, elapsed_m, DIEBOLD_LI_TAU)
                market_rt[i] = rate
                liq_spread[i] = float(_liquidity_spread(np.array([original_tenor_m]))[0])

        # ── NMD: trailing 12M moving average of the 1Y market rate ──────────
        if len(nmd_idx) > 0:
            nmd_market = _trailing_market_rate_avg(panel)
            market_rt[nmd_idx] = nmd_market
            liq_spread[nmd_idx] = _liquidity_spread(params.repricing_tenor_m[nmd_idx])

    return market_rt, liq_spread


def compute_ftp_rates(params, curve_tensors) -> np.ndarray:
    """(n,) FTP rate per cohort, decimal (e.g. 0.035 = 3.5%). Zero for equity
    (equity has no client rate / FTP concept in this model -- its cost is
    already captured separately via cost of capital)."""
    market_rt, liq_spread = compute_ftp_components(params, curve_tensors)
    return market_rt + liq_spread


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


def equity_benefit_unit_rate(ftp_rate: np.ndarray, rwa_factor: np.ndarray, cet1_target: float) -> np.ndarray:
    """Equity Benefit per unit balance -- the FTP credit margin_unit_rate() is
    missing on its own. margin_unit_rate() charges an asset the FULL ftp_rate
    on 100% of its balance, as if every zloty were funded by transfer-priced
    debt; CoC (elsewhere: RWA x CET1_target x coc_rate) separately charges
    that SAME asset for the capital that actually funds capital_pct of it.
    Between the two, the capital-funded share pays twice: once via CoC for
    consuming capital, once via Margin for "owing" FTP-priced debt funding it
    never needed. Equity Benefit removes the second charge by crediting
    ftp_rate back on capital_pct of the balance -- economically, capital is
    free (non-interest-bearing) funding, so the fraction of the balance sheet
    it funds shouldn't also carry a market-rate funding cost.

    capital_pct = rwa_factor x cet1_target, clipped to [0, 1] (a balance can't
    be more than 100% capital-funded even if rwa_factor x cet1_target would
    imply otherwise for a very high-risk-weight cohort). rwa_factor is already
    0 for liability/equity rows in this codebase, so capital_pct -- and
    therefore this rate -- is naturally 0 for non-assets too, with no bs_side
    branching needed here (unlike margin_unit_rate(), which does need one).
    """
    capital_pct = np.clip(rwa_factor * cet1_target, 0.0, 1.0)
    return ftp_rate * capital_pct


def save_ftp_rates(
    ftp_rate: np.ndarray,
    cohort_id: np.ndarray,
    market_rt: np.ndarray | None = None,
    liq_spread: np.ndarray | None = None,
    path: str = FTP_PATH,
) -> None:
    """market_rt/liq_spread are optional (backward-compatible) -- when given,
    cached alongside ftp_rate so the breakdown survives a reload without
    recomputation. load_ftp_rates() still returns only the combined
    ftp_rate array; callers that need the breakdown read the npz directly
    (see build_ftp_rates.py) or opt_prep.ftp_rates in SQL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = dict(ftp_rate=ftp_rate, cohort_id=cohort_id)
    if market_rt is not None:
        kwargs["market_rt"] = market_rt
    if liq_spread is not None:
        kwargs["liq_spread"] = liq_spread
    np.savez(path, **kwargs)
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
