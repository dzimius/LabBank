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
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from labbank_common.bank_config import get_bank
BANK_OUT_DIR = str(get_bank().output_dir("optimize_prep"))   # runs/<bank_id>/optimize_prep

_HERE = os.path.dirname(os.path.abspath(__file__))
FTP_PATH = os.path.join(BANK_OUT_DIR, "ftp_rates.npz")

# ns_curve_model.py (historical Nelson-Siegel fit) lives alongside this file
# (moved here from bs_optimization/python_code 2026-08-14, same move as
# _ProductMap/ep_fast.py -- see that commit) -- same NS-fit machinery
# swap_ladder.py uses for its seasoned buckets, reused rather than duplicated.

LIQUIDITY_CAP_BPS    = 50.0    # 0.50% at 10Y+
LIQUIDITY_CAP_MONTHS = 120.0   # 10Y
REPORT_DATE          = pd.Timestamp(get_bank().report_date)   # matches extract_params.REPORT_DATE

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


# ── product terms (2026-10-28 review fixes) ──────────────────────────────────
# The FTP of a cohort has two parts with different tenors:
#   rate component      -- the period the client rate is fixed for: the fixing period (fixed products with a
#                          fixing_freq, e.g. a 25Y mortgage fixed for 10Y), else the contract's original term
#                          (bullet fixed bonds, loans, term deposits); for floating products the reset period.
#                          Fixed: the historical rate at the LAST reset (origination, or the latest re-fixing
#                          of a fixing period), for that tenor. Floating: today's term rate for the reset period.
#   liquidity component -- how long the funding is used: the repayment profile of the original contract
#                          (bullet: the term; amortising: half the term; revolving cards: 1 / monthly principal).
# Before: repricing_tenor_m (the FULL fixing period for fixed products with a fixing_freq) + the cohort's age
# was used for both, so a 5Y bond aged 4Y was priced at a 9Y rate, and a 5Y floating loan resetting on 3M
# paid the 3M liquidity spread; floating rates took the one-month forward in the reset month, not the
# term rate from today.
_MONTHS = {"D": 1.0 / 30.44, "W": 7.0 / 30.44, "M": 1.0, "Y": 12.0}


def _tenor_months(s) -> float | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().upper()
    if not s or s == "NM" or s[-1] not in _MONTHS:
        return None
    try:
        return float(s[:-1]) * _MONTHS[s[-1]]
    except ValueError:
        return None


_TERMS: dict | None = None


def product_terms() -> dict:
    """(product_code, bs_side) -> dict(rate_type, fix_m, mat_m, liq_m) from bs_structure incl. the catalogue
    variants (labbank_common.catalogue.read_bs_structure)."""
    global _TERMS
    if _TERMS is None:
        from labbank_common.catalogue import read_bs_structure
        bs = read_bs_structure(get_bank().balance_dir() / "bank_data.xlsx")
        out = {}
        for r in bs.itertuples():
            mat = _tenor_months(r.maturity)
            amort = float(r.amort_type) if pd.notna(r.amort_type) else 0.0
            if mat is None:
                liq = None
            elif amort == 3.0 and pd.notna(getattr(r, "card_monthly_principal_rate", np.nan)) \
                    and float(r.card_monthly_principal_rate) > 0:
                liq = min(mat, 1.0 / float(r.card_monthly_principal_rate))      # revolving: principal turns over
            elif amort == 0.0:
                liq = mat                                                       # bullet
            else:
                liq = mat / 2.0                                                 # amortising: average life
            out[(str(int(r.product_code)) if not isinstance(r.product_code, str) else r.product_code, r.bs_side)] = dict(
                rate_type=str(r.rate_type) if pd.notna(r.rate_type) else "", fix_m=_tenor_months(r.fixing_freq),
                mat_m=mat, liq_m=liq)
        _TERMS = out
    return _TERMS


def _terms_of(params, i) -> dict:
    t = product_terms().get((str(params.product_code[i]), str(params.bs_side[i])))
    if t is None:
        raise KeyError(f"[ftp_store] no bs_structure row for {params.product_code[i]}/{params.bs_side[i]}")
    return t


def _liq_tenor(params, i) -> float:
    t = _terms_of(params, i)
    if t["liq_m"] is not None:
        return t["liq_m"]
    return float(params.repricing_tenor_m[i])


FTP_HORIZON_M = 12     # the NII horizon: nii_unit_rate projects 12 months of income along the forwards


def _term_rate(fwd: np.ndarray, months: float, start_m: int = 0) -> float:
    """Annualised term rate for `months` starting at month `start_m`, from the monthly forward curve
    (compounded monthly)."""
    k = int(np.clip(np.round(months), 1, len(fwd) - start_m))
    return float((np.prod(1.0 + fwd[start_m:start_m + k] / 12.0) - 1.0) * 12.0 / k)


def _reset_rate_over_horizon(fwd: np.ndarray, reset_m: float, horizon_m: int = FTP_HORIZON_M) -> float:
    """Average FTP base rate of a position that re-fixes every `reset_m` months over the NII horizon: the
    forward term rate of each fixing period, weighted by the months it covers. A floating product's FTP
    resets with its client rate, so it follows the same forward path as nii_unit_rate (taking only
    today's fixing would give every floater the forward curve's slope as margin)."""
    k = max(int(round(reset_m)), 1)
    starts = list(range(0, horizon_m, k))
    w = [min(k, horizon_m - s) for s in starts]
    return float(np.average([_term_rate(fwd, k, s) for s in starts], weights=w))


def floating_ftp_components(params, is_float: np.ndarray, curve_tensors) -> tuple[np.ndarray, np.ndarray]:
    """(market_rt, liq_spread), each (n,), for FLOATING cohorts only: the forward term rate of the reset
    period over the NII horizon, from `curve_tensors`'s base scenario, and the liquidity spread for the
    repayment profile. Zero elsewhere. Split out of floating_ftp_rate() so callers that need the breakdown
    don't re-derive it."""
    n = len(params.cohort_id)
    market_rt = np.zeros(n, dtype=float)
    liq_spread = np.zeros(n, dtype=float)
    base_idx = curve_tensors.scenario_index("base")
    for ccy in np.unique(params.currency):
        idx = np.where((params.currency == ccy) & is_float)[0]
        if not len(idx):
            continue
        fwd = curve_tensors.fwd_rates[base_idx, curve_tensors.currency_index(str(ccy))]
        for i in idx:
            reset_m = _terms_of(params, i)["fix_m"] or float(params.repricing_tenor_m[i])
            market_rt[i] = _reset_rate_over_horizon(fwd, reset_m)
            liq_spread[i] = float(_liquidity_spread(np.array([_liq_tenor(params, i)]))[0])
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


def _panel_for(ccy: str):
    """Historical fixing panel (bps) of a currency: the bank's own file for the home currency, the foreign
    currency's file otherwise (labbank_common.fx.foreign_input_dir)."""
    from ns_curve_model import load_historical_panel
    if ccy == get_bank().market.ccy:
        return load_historical_panel()
    from labbank_common.fx import foreign_input_dir
    return load_historical_panel(str(foreign_input_dir(ccy) / "fixing_input.xlsx"), ccy=ccy)


def _nmd_profiles() -> tuple[dict, dict]:
    """(rate profile, liquidity profile): product_code -> survival S(m) at m = 0 (1D), 1, 2, ... months, from
    dep_beh_models_ir.xlsx (core) and dep_beh_models_liq.xlsx (stable). A product without a column is
    overnight money (S = 0)."""
    d = get_bank().models_dir()
    out = []
    for f in ("dep_beh_models_ir.xlsx", "dep_beh_models_liq.xlsx"):
        x = pd.read_excel(d / f)
        prof = {}
        for c in x.columns:
            if str(c) in ("report_date", "tenor"):
                continue
            s = x[c].astype(float).dropna().to_numpy()
            prof[str(c)] = s
        out.append(prof)
    return out[0], out[1]


def _nmd_tranche_weights(s: np.ndarray) -> tuple[float, np.ndarray]:
    """Survival S(0..M) -> (overnight weight 1 - S(0), weights of tranches maturing at months 1..M, the last
    holding what survives the profile)."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    w = np.append(s[:-1] - s[1:], s[-1])            # month k = 1..M
    return 1.0 - s[0], w


def compute_ftp_components(params, curve_tensors) -> tuple[np.ndarray, np.ndarray]:
    """(market_rt, liq_spread), each (n,), decimal, per cohort. Zero for equity and for derivatives
    (products 0000/0100 -- their carry is in the swap result). Split out of compute_ftp_rates() so callers
    (e.g. build_ftp_rates.py's SQL write) can persist market_rt/liq_spread as separate columns."""
    from ns_curve_model import DIEBOLD_LI_TAU, fit_ns_timeseries, ns_design_matrix

    n = len(params.cohort_id)
    market_rt = np.zeros(n, dtype=float)
    liq_spread = np.zeros(n, dtype=float)
    pcs = params.product_code.astype(str)
    is_deriv = np.isin(pcs, ("0000", "0100"))
    is_nmd = np.isin(params.product_name.astype(str), NMD_PRODUCT_NAMES) & ~params.is_equity
    is_float = (params.rate_type == "V") & ~params.is_equity & ~is_nmd & ~is_deriv
    is_fixed = (params.rate_type == "F") & ~params.is_equity & ~is_nmd & ~is_deriv
    # cash, ECB deposits and other single-row positions without F/V: priced at the overnight rate
    is_on = ~params.is_equity & ~is_nmd & ~is_deriv & ~is_float & ~is_fixed

    f_market, f_liq = floating_ftp_components(params, is_float, curve_tensors)
    market_rt += f_market
    liq_spread += f_liq

    panels, ns = {}, {}

    def ns_of(ccy):
        if ccy not in ns:
            panels[ccy] = _panel_for(ccy)
            ns[ccy] = fit_ns_timeseries(panels[ccy], DIEBOLD_LI_TAU)
        return ns[ccy]

    def ns_rate(ccy, tenor_m, months_ago):
        ts = ns_of(ccy)
        d = REPORT_DATE - pd.DateOffset(months=int(months_ago))
        pos = int(min(max(ts.index.searchsorted(d), 0), len(ts) - 1))
        beta = ts.iloc[pos][["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
        return float((ns_design_matrix(np.array([tenor_m / 12.0]), DIEBOLD_LI_TAU) @ beta)[0]) / 10000.0

    def on_rate(ccy):
        # overnight money re-prices continuously: the monthly forwards over the NII horizon, the same basis
        # as the income these positions are projected to earn
        fwd = curve_tensors.fwd_rates[curve_tensors.scenario_index("base"), curve_tensors.currency_index(ccy)]
        return _reset_rate_over_horizon(fwd, 1)

    # ── Fixed: the historical rate at the last reset, for the fixing period (or the original term) ──
    for i in np.where(is_fixed)[0]:
        t = _terms_of(params, i)
        age = _elapsed_months_since_origination(str(params.cohort_id[i])) if params.is_cohort[i] else 0
        if t["fix_m"]:                                   # re-fixes every fix_m months: last reset age % fix_m ago
            tenor_m, ago = t["fix_m"], age % max(int(round(t["fix_m"])), 1)
        elif t["mat_m"]:                                 # fixed to maturity: the original term at origination
            tenor_m, ago = t["mat_m"], age
        else:
            tenor_m, ago = max(float(params.repricing_tenor_m[i]) + age, 1.0), age
        locked = ns_rate(str(params.currency[i]), tenor_m, ago)
        # a fixing that ends inside the NII horizon is renewed (re-fixed) at the forward term rate, as the
        # NII engine renews it (labbank_common.curves.renewal_market_rate): blend over the horizon
        rem = max(tenor_m - ago, 0.0)
        if rem < FTP_HORIZON_M:
            fwd = curve_tensors.fwd_rates[curve_tensors.scenario_index("base"),
                                          curve_tensors.currency_index(str(params.currency[i]))]
            r0 = int(round(rem))
            k = max(int(round(tenor_m)), 1)
            starts = list(range(r0, FTP_HORIZON_M, k))
            w = [min(k, FTP_HORIZON_M - s) for s in starts]
            ren = sum(wi * _term_rate(fwd, k, s) for wi, s in zip(w, starts))
            locked = (r0 * locked + ren) / (r0 + sum(w))
        market_rt[i] = locked
        liq_spread[i] = float(_liquidity_spread(np.array([_liq_tenor(params, i)]))[0])

    # ── Overnight positions ──
    for i in np.where(is_on)[0]:
        market_rt[i] = on_rate(str(params.currency[i]))

    # ── NMD: a replicating portfolio from the deposit models, per product and currency ──
    # overnight part at today's overnight rate; core tranches maturing in month k at the trailing
    # NMD_LOOKBACK_MONTHS average of the k-month rate (smoothing, as a rolling tranche ladder would);
    # liquidity from the stable profile. The rate and liquidity profiles stay separate.
    if is_nmd.any():
        prof_ir, prof_liq = _nmd_profiles()
        cache = {}
        for i in np.where(is_nmd)[0]:
            pc, ccy = pcs[i], str(params.currency[i])
            if (pc, ccy) not in cache:
                ts = ns_of(ccy)
                win = ts.loc[(ts.index > REPORT_DATE - pd.DateOffset(months=NMD_LOOKBACK_MONTHS)) & (ts.index <= REPORT_DATE)]
                if win.empty:
                    raise ValueError(f"[ftp_store] no {ccy} fixings in the {NMD_LOOKBACK_MONTHS}M window to {REPORT_DATE.date()}")
                w_on, w = _nmd_tranche_weights(prof_ir.get(pc, np.array([0.0])))
                k = np.arange(1, len(w) + 1, dtype=float)
                X = ns_design_matrix(k / 12.0, DIEBOLD_LI_TAU)
                avg = (X @ win[["beta0", "beta1", "beta2"]].to_numpy(dtype=float).T).mean(axis=1) / 10000.0
                rate = w_on * on_rate(ccy) + float(w @ avg)
                l_on, lw = _nmd_tranche_weights(prof_liq.get(pc, np.array([0.0])))
                liq = float(lw @ _liquidity_spread(np.arange(1, len(lw) + 1, dtype=float)))
                cache[(pc, ccy)] = (rate, liq)
            market_rt[i], liq_spread[i] = cache[(pc, ccy)]

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


def load_ftp_rates(cohort_id: np.ndarray, path: str = FTP_PATH, strict: bool = True) -> np.ndarray | None:
    """Load cached FTP rates for the given cohort_id order.
    A missing cache or one built for another cohort set (product_params.npz regenerated without
    build_ftp_rates.py) raises: an optimiser that silently ran with FTP = 0 would rank products on
    client rates (2026-10-28). strict=False returns None instead, for display-only callers."""
    def _fail(msg):
        if strict:
            raise RuntimeError(f"[ftp_store] {msg} -- run optimize_prep/python_code/build_ftp_rates.py")
        return None
    if not os.path.exists(path):
        return _fail(f"no FTP cache at {path}")
    data = np.load(path, allow_pickle=True)
    cached_ids = data["cohort_id"]
    if len(cached_ids) != len(cohort_id) or not np.array_equal(cached_ids, cohort_id):
        return _fail("FTP cache does not match product_params.npz")
    return data["ftp_rate"].astype(float)
