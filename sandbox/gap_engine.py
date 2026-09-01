"""gap_engine.py
===============
Repricing (IR) gap and Liquidity gap for the sandbox.

NMD behavioral model
--------------------
Current and saving accounts use the monthly outstanding-fraction decay curves
from dep_beh_models_ir.xlsx (same source as the production pipeline).
The repricing amount at month m = balance × (outstanding[m-1] − outstanding[m]).
The first tenor point (1D) captures the non-stable/volatile portion that reprices
immediately; subsequent months capture the slow decay of the stable core.

Other products
--------------
Cohort products (loans, bonds, term deposits): use cohort_t_first_m from the npz
— the exact months-to-first-repricing computed during the full pipeline run.
Products with t_first_m = 999 that are NOT NMDs fall back to repricing_tenor_m.

IRS float legs: per-swap next-fixing date derived from start_date and fixing freq.
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

from irs_engine import _maturity_months

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEP_BEH_PATH = os.path.join(PROJECT_ROOT, "balance_gen_add_data", "input", "dep_beh_models_ir.xlsx")

REPORT_DATE_DEFAULT = date(2026, 6, 30)

# 12 monthly buckets + 4 longer-tenor buckets
BUCKETS       = ["1M","2M","3M","4M","5M","6M","7M","8M","9M","10M","11M","12M",
                 "1-2Y","2-3Y","3-5Y","5Y+"]
_BUCKET_UPPER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36, 60, 9999]

# NMD product codes that have behavioral repricing models
_NMD_CODES    = {6000, 8000, 6300}        # 6300 uses 6000 curve as proxy
_NMD_PROXY    = {6300: 6000}              # product codes with proxy mapping


def _bucket_idx(months: float) -> int:
    for i, upper in enumerate(_BUCKET_UPPER):
        if months <= upper:
            return i
    return len(BUCKETS) - 1


# lower/upper month bound of every bucket, e.g. "3M" -> (2, 3), "1-2Y" -> (12, 24)
_BUCKET_EDGES = [(0.0 if i == 0 else float(_BUCKET_UPPER[i - 1]), float(up))
                 for i, up in enumerate(_BUCKET_UPPER)]


def _profile_row(params, i: int, n_b: int) -> np.ndarray | None:
    """Row i's baked-in ir_gap_beh_a repricing profile (fraction of balance per
    bucket), or None if the npz predates the field or the product has no row."""
    gp = getattr(params, "gap_reprice_frac_m", None)
    if gp is None or i >= len(gp):
        return None
    pi = np.asarray(gp[i], dtype=float)
    return pi if pi.shape == (n_b,) and pi.any() else None


def _reprice_spread(balance: float, t_first_m: float, freq_m: float) -> np.ndarray:
    """Distribute a cohort's balance across repricing buckets.

    A cohort aggregates many contracts that each reprice every ``freq_m`` months.
    Their individual next-repricing dates are staggered, not simultaneous, so
    dumping the whole balance into one bucket (the cohort-average months-to-
    first-repricing) badly over-concentrates the near end of the ladder relative
    to the production ``irrbb.ir_gap_beh`` view, which places every contract at
    its own fixing date.

    Model the next-repricing month as uniform over
    ``[t_first_m - freq_m/2, t_first_m + freq_m/2]`` (mean = the cohort average),
    with the lower edge clamped at 0 — contracts already past their fixing
    reprice in the first bucket. The window is renormalised after clamping so
    the whole balance is always placed. Example: a quarterly-reset cohort in
    steady state (t_first ≈ 1.5, freq = 3) puts one third of the balance in each
    of the 1M / 2M / 3M buckets.
    """
    rep = np.zeros(len(BUCKETS))
    F   = max(float(freq_m), 1e-6)
    t1  = max(0.0, float(t_first_m))
    lo  = max(0.0, t1 - F / 2.0)
    hi  = max(t1 + F / 2.0, lo + 1e-6)
    width = hi - lo
    for b, (b_lo, b_hi) in enumerate(_BUCKET_EDGES):
        overlap = min(hi, b_hi) - max(lo, b_lo)
        if overlap > 0.0:
            rep[b] += balance * overlap / width
    return rep


def _freq_to_months(freq) -> float:
    if freq is None or (isinstance(freq, float) and np.isnan(freq)):
        return 3.0
    s = str(freq).strip().upper()
    if s.endswith("M"):
        try: return float(s[:-1])
        except ValueError: return 3.0
    if s.endswith("Y"):
        try: return float(s[:-1]) * 12.0
        except ValueError: return 12.0
    return 3.0


def _irs_next_float_repricing_m(start_date, float_fixing_freq_m: float,
                                 report_date: date) -> float:
    if start_date is None:
        return float_fixing_freq_m
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    try:
        start_date = pd.Timestamp(start_date).date()
    except Exception:
        return float_fixing_freq_m
    days_since   = (report_date - start_date).days
    months_since = days_since / 30.4375
    if months_since < 0:
        return float_fixing_freq_m + (-months_since)
    k      = int(months_since / float_fixing_freq_m)
    t_next = (k + 1) * float_fixing_freq_m - months_since
    return max(0.01, t_next)


# ── NMD behavioral model ─────────────────────────────────────────────────────

def _load_nmd_model(path: str) -> dict[int, tuple[list, list]]:
    """Load dep_beh_models_ir.xlsx.

    Returns dict: product_code → (months_list, outstanding_list)
    months_list[0] = 0.033 (1D), then 1, 2, 3, ...
    outstanding_list[i] = fraction of original balance still outstanding at month i.
    """
    df = pd.read_excel(path)
    nmd_product_cols = [c for c in df.columns if isinstance(c, (int, float))]

    models: dict[int, tuple[list, list]] = {}
    for pc in nmd_product_cols:
        pc_int = int(pc)
        months: list[float] = []
        outstanding: list[float] = []
        for _, row in df.iterrows():
            tenor = str(row["tenor"]).strip()
            val   = row[pc]
            if pd.isna(val):
                break          # model ends here for this product
            if tenor == "1D":
                m = 1.0 / 30.4375  # ~0.033 months
            elif tenor.endswith("M"):
                m = float(tenor[:-1])
            elif tenor.endswith("Y"):
                m = float(tenor[:-1]) * 12.0
            else:
                continue
            months.append(m)
            outstanding.append(float(val))
        if months:
            models[pc_int] = (months, outstanding)
    return models


def _nmd_repricing_by_bucket(balance: float,
                              product_code: int,
                              nmd_models: dict) -> np.ndarray:
    """Spread NMD balance across buckets using behavioral decay curve.

    repricing at month m = balance × (outstanding[m-1] − outstanding[m])
    where outstanding[-1] = 1.0 (nothing repriced yet).
    """
    n_b       = len(BUCKETS)
    repricing = np.zeros(n_b)

    # proxy mapping: 6300 (current_account_sme) uses 6000 curve
    model_code = _NMD_PROXY.get(product_code, product_code)
    if model_code not in nmd_models:
        return repricing

    months, outstanding = nmd_models[model_code]
    prev_out = 1.0

    for m, out in zip(months, outstanding):
        rep_frac = prev_out - out
        if rep_frac > 1e-9:
            repricing[_bucket_idx(m)] += balance * rep_frac
        prev_out = out

    # remaining fraction (model truncated before 100%) → last bucket
    if prev_out > 1e-9:
        repricing[-1] += balance * prev_out

    return repricing


# ── main gap computation ──────────────────────────────────────────────────────

def compute_repricing_gap(
    params,
    irs_df: pd.DataFrame,
    report_date: date | None = None,
    nmd_models: dict | None = None,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return repricing gap DataFrame.

    Columns: bucket, bs_assets, bs_liab, bs_gap,
             irs_assets, irs_liab, irs_gap, net_gap,
             cum_bs_gap, cum_net_gap

    weights : optional (n,) array, fraction of total_assets per npz row —
              same array `baseline.compute_weights()` returns for the Metrics
              tab. When given, each row's balance is taken as
              weights[i] * params.total_assets instead of the static
              params.balance_arr[i], so the gap reflects Balance Sheet tab
              edits. Omit for the shipped-baseline gap.
    """
    if report_date is None:
        report_date = REPORT_DATE_DEFAULT
    try:
        nmd_models_base = _load_nmd_model(DEP_BEH_PATH)
    except Exception:
        nmd_models_base = {}
    if nmd_models is None:
        nmd_models = nmd_models_base

    n_b  = len(BUCKETS)
    bs_a = np.zeros(n_b)
    bs_l = np.zeros(n_b)

    n = len(params.product_code)
    for i in range(n):
        pc_str = str(params.product_code[i])
        side   = str(params.bs_side[i])

        if pc_str == "0000" or side == "E":
            continue

        bal = (float(weights[i]) * float(params.total_assets) if weights is not None
               else float(params.balance_arr[i]))
        pc_int  = int(pc_str) if pc_str.isdigit() else -1

        if pc_int in _NMD_CODES:
            # ── NMD: report profile at baseline + live delta for decay edits ──
            # The frozen ir_gap_beh_a profile keeps the baseline gap identical to
            # the ALM report; the behavioural model supplies only the *change*
            # when the NMD Stress tab edits the decay curve.
            prof = _profile_row(params, i, n_b)
            live_now  = _nmd_repricing_by_bucket(bal, pc_int, nmd_models)
            live_base = _nmd_repricing_by_bucket(bal, pc_int, nmd_models_base)
            if prof is not None:
                repricing = bal * prof + (live_now - live_base)
            else:
                repricing = live_now
            if side == "A":
                bs_a += repricing
            elif side == "L":
                bs_l += repricing
        else:
            # ── Other products ──────────────────────────────────────────────
            # Preferred: the exact per-product repricing profile the ALM /
            # finance reports use (irrbb.ir_gap_beh_a), baked into the npz as a
            # fraction of product balance per bucket. Scaling it by this row's
            # (edited) balance reproduces the report gap at baseline and moves
            # correctly with Balance Sheet edits.
            prof = _profile_row(params, i, n_b)

            if prof is not None:
                spread = bal * prof
            elif not (bool(params.is_cohort[i]) if hasattr(params, "is_cohort") else True):
                # Single-row product with no ir_gap_beh_a row → not in the
                # production repricing gap either. Currently only product 3500
                # (non-interest-bearing cash), which genuinely does not reprice.
                spread = np.zeros(n_b)
            else:
                # Cohort missing from ir_gap_beh_a (or old npz without the
                # profile field): spread the balance over a reset-frequency-wide
                # window centred on the cohort-average first-repricing month.
                t1 = float(params.cohort_t_first_m[i])
                F  = float(params.repricing_tenor_m[i])
                if F >= 999:
                    F = t1 if t1 < 999 else 300.0
                if t1 >= 999:
                    t1 = F / 2.0
                spread = _reprice_spread(bal, t1, F)

            if side == "A":
                bs_a += spread
            elif side == "L":
                bs_l += spread

    # ── IRS: per-swap float leg at actual next fixing ─────────────────────────
    irs_a = np.zeros(n_b)
    irs_l = np.zeros(n_b)

    for _, row in irs_df.iterrows():
        notl = float(row.get("notional") or 0)
        if notl <= 0:
            continue
        pay_fixed = bool(row.get("pay_fixed", False))
        mat_m     = _maturity_months(row.get("maturity_date"), report_date)
        if mat_m <= 0:
            continue

        freq_m   = _freq_to_months(row.get("float_fixing_freq", "3M"))
        t_float  = _irs_next_float_repricing_m(row.get("start_date"), freq_m, report_date)
        float_bi = _bucket_idx(t_float)
        fixed_bi = _bucket_idx(mat_m)

        if not pay_fixed:
            irs_a[fixed_bi] += notl
            irs_l[float_bi] += notl
        else:
            irs_l[fixed_bi] += notl
            irs_a[float_bi] += notl

    bs_gap  = bs_a - bs_l
    irs_gap = irs_a - irs_l
    net_gap = bs_gap + irs_gap

    return pd.DataFrame({
        "bucket":      BUCKETS,
        "bs_assets":   bs_a,
        "bs_liab":     bs_l,
        "bs_gap":      bs_gap,
        "irs_assets":  irs_a,
        "irs_liab":    irs_l,
        "irs_gap":     irs_gap,
        "net_gap":     net_gap,
        "cum_bs_gap":  np.cumsum(bs_gap),
        "cum_net_gap": np.cumsum(net_gap),
    })


def compute_liq_gap(params, report_date: date | None = None,
                    weights: np.ndarray | None = None) -> pd.DataFrame:
    """12-month liquidity gap from cohort_capital_m.

    weights : optional (n,) array, fraction of total_assets per npz row — same
              array as ``compute_repricing_gap``. When given, each row's balance
              is ``weights[i] * params.total_assets`` so the liquidity gap
              reflects Balance Sheet tab edits; omit for the shipped baseline.
    """
    n_m = 12
    asset_cf = np.zeros(n_m)
    liab_cf  = np.zeros(n_m)

    n = len(params.product_code)
    for i in range(n):
        if str(params.product_code[i]) == "0000" or str(params.bs_side[i]) == "E":
            continue
        bal   = (float(weights[i]) * float(params.total_assets) if weights is not None
                 else float(params.balance_arr[i]))
        cap_m = params.cohort_capital_m[i]
        for m in range(n_m):
            cf = bal * float(cap_m[m])
            if str(params.bs_side[i]) == "A":
                asset_cf[m] += cf
            elif str(params.bs_side[i]) == "L":
                liab_cf[m] += cf

    net = asset_cf - liab_cf
    return pd.DataFrame({
        "month":         [f"{m+1}M" for m in range(n_m)],
        "asset_inflows":  asset_cf,
        "liab_outflows":  liab_cf,
        "net_liq_gap":    net,
        "cum_net":        np.cumsum(net),
    })
