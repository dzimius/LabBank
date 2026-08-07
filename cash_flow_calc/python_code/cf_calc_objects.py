from __future__ import annotations

import datetime
from bisect import bisect_left

import numpy as np
import pandas as pd
import QuantLib as ql

import config
import sql_setup

dict_cols_loan_fin_inst = {
    # balance_amt / init_balance_amt / amort_type pulled for vectorized payment computation
    # product_code links loan schedules to prepayment models (bs.models_loan table)
    # margin is the per-product spread over the market forward rate (loans only)
    'loans': ['schedule_id', 'currency', 'start_date', 'maturity_date', 'payment_freq',
              'fixing_freq', 'dc_conv', 'b_day_conv', "rate_index", 'disc_curve', 'fwd_curve',
              'balance_amt', 'init_balance_amt', 'amort_type', 'product_code', 'bs_side', 'margin'],
    'fin_inst': ['schedule_id', 'currency', 'start_date', 'maturity_date', 'payment_freq',
                 'fixing_freq', 'dc_conv', 'b_day_conv', "rate_index", 'disc_curve', 'fwd_curve',
                 'balance_amt', 'amort_type', 'bs_side', 'product_code'],
}

sched_tables = ['loans', 'fin_inst']

dict_cols_deposits = {
    # maturity_date = NULL means non-maturity deposit (overnight in origin schedule)
    # product_code links to bs.models_deposit for behavioural schedule
    # client_rt is the all-in rate paid to clients (a * market_rate + b), computed at balance generation
    'deposits': ['schedule_id', 'product_code', 'currency', 'rate_type', 'maturity_date', 'dc_conv',
                 'b_day_conv', 'disc_curve', 'balance_amt', 'bs_side', 'client_rt'],
}

# ── CF wide-table merge helpers ───────────────────────────────────────────────
# Rate/identity columns carried from the orig side (filled from beh for outer-only rows).
_CF_RATE_COLS = ['bs_side', 'rate_index', 'fixing_dt', 'cf_start_dt_delay', 'cf_end_dt',
                 'cf_yf', 'd_f', 'fwd_rt', 'margin', 'client_rt']

_CON_PAY_RENAME = {
    'outstanding_bal': 'con_outstanding',
    'capital_pmt':     'con_capital_pmt',
    'int_pmt':         'con_interest_pmt',
    'total_pmt':       'con_total_pmt',
}
_BEH_PAY_RENAME = {
    'outstanding_bal': 'beh_outstanding',
    'capital_pmt':     'beh_capital_pmt',
    'int_pmt':         'beh_interest_pmt',
    'total_pmt':       'beh_total_pmt',
}


def merge_cf_orig_beh(
    orig_df: pd.DataFrame,
    beh_df: pd.DataFrame,
    how: str = 'inner',
) -> pd.DataFrame:
    """Merge contractual (orig) and behavioural (beh) CF schedules into one wide row.

    how='inner'  — fin_inst: orig and beh are identical, same date grid.
    how='left'   — loans: orig grid is authoritative; beh may have fewer rows
                   (fully-prepaid periods absent from beh → filled with 0).
    how='outer'  — deposits: orig has one bullet row per schedule; beh has N
                   tenor-bucket rows.  Missing side filled with 0.

    Added columns in output:
        con_outstanding, con_capital_pmt, con_interest_pmt, con_total_pmt
        beh_outstanding, beh_capital_pmt, beh_interest_pmt, beh_total_pmt
        comp_capital_pmt, comp_interest_pmt, comp_total_pmt
        prepayment_pmt   (loans only — absent/NULL for all other products)
    """
    JOIN_KEYS = ['schedule_id', 'cf_start_dt']

    # Prepare orig side
    orig_rate = [c for c in _CF_RATE_COLS if c in orig_df.columns]
    orig_pay  = [c for c in _CON_PAY_RENAME if c in orig_df.columns]
    orig_prep = (
        orig_df[JOIN_KEYS + orig_rate + orig_pay]
        .rename(columns=_CON_PAY_RENAME)
    )

    # Prepare beh side — include rate cols only for outer join (to fill beh-only rows)
    beh_pay   = [c for c in _BEH_PAY_RENAME if c in beh_df.columns]
    beh_extra = ['prepayment_pmt'] if 'prepayment_pmt' in beh_df.columns else []
    if how == 'outer':
        beh_rate = [c for c in _CF_RATE_COLS if c in beh_df.columns]
        beh_sel  = JOIN_KEYS + beh_rate + beh_pay + beh_extra
    else:
        beh_sel  = JOIN_KEYS + beh_pay + beh_extra

    beh_prep = beh_df[beh_sel].rename(columns=_BEH_PAY_RENAME)

    merged = orig_prep.merge(beh_prep, on=JOIN_KEYS, how=how, suffixes=('', '_beh'))

    if how == 'outer':
        # Coalesce rate cols: orig value first; fall back to beh for beh-only rows
        for col in _CF_RATE_COLS:
            beh_col = f'{col}_beh'
            if beh_col in merged.columns:
                merged[col] = merged[col].fillna(merged[beh_col])
                merged.drop(columns=[beh_col], inplace=True)

    # Fill missing payment values with 0 on either side (left and outer both need this)
    if how in ('left', 'outer'):
        for col in list(_CON_PAY_RENAME.values()) + list(_BEH_PAY_RENAME.values()):
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

    # Behavioural components = beh - con
    merged['comp_capital_pmt']  = merged['beh_capital_pmt']  - merged['con_capital_pmt']
    merged['comp_interest_pmt'] = merged['beh_interest_pmt'] - merged['con_interest_pmt']
    merged['comp_total_pmt']    = merged['beh_total_pmt']    - merged['con_total_pmt']

    return merged


def get_calendar_from_currency(curr: str) -> ql.Calendar:
    currency_to_calendar = {
        'PLN': ql.Poland(),
        'EUR': ql.TARGET(),
        'USD': ql.UnitedStates(ql.UnitedStates.NYSE)
    }
    return currency_to_calendar.get(curr, ql.TARGET())

def get_dc_conv_from_str(b: str) -> ql.DayCounter:
    dc_conv_from_str = {
        'ACT/365': ql.Actual365Fixed(),
        'ACT/ACT': ql.ActualActual(ql.ActualActual.ISDA),
        '30/360': ql.Thirty360(ql.Thirty360.ISDA)
    }
    return dc_conv_from_str.get(b, ql.ActualActual(ql.ActualActual.ISDA))

def get_dc_from_currency(curr: str) -> ql.DayCounter:
    currency_to_dc_conv = {
        'PLN': ql.Actual365Fixed(),
        'USD': ql.ActualActual(ql.ActualActual.ISDA),
        'EUR': ql.Thirty360(ql.Thirty360.ISDA),
    }
    return currency_to_dc_conv.get(curr, ql.Actual365Fixed())

def tenor_to_months(tenor: str) -> int:
    """Convert tenor string to integer months using QuantLib Period units.

    Uses ql.Period so the conversion is exact and consistent with all other
    QuantLib date arithmetic in the pipeline.
    Days and weeks are converted to days (not months) — callers that need
    to compare day-based and month-based tenors must handle this separately.
    """
    p = ql.Period(tenor)
    unit = p.units()
    n    = p.length()
    if unit == ql.Days:   return n          # return days as-is; unit = 'D'
    if unit == ql.Weeks:  return n * 7      # weeks → days
    if unit == ql.Months: return n
    if unit == ql.Years:  return n * 12
    raise ValueError(f"Unsupported tenor: {tenor}")


def _period_to_canonical(tenor: str) -> tuple[int, int]:
    """Return (value, unit_code) where unit_code 0=days, 1=months.

    Weeks are normalised to days; years to months.
    This keeps day-based and month-based frequencies in separate integer spaces
    so block-size division is always exact.
    """
    p = ql.Period(tenor)
    unit = p.units()
    n    = p.length()
    if unit == ql.Days:   return n,      0
    if unit == ql.Weeks:  return n * 7,  0
    if unit == ql.Months: return n,      1
    if unit == ql.Years:  return n * 12, 1
    raise ValueError(f"Unsupported tenor: {tenor}")


def freeze_fixing_dates(fixing_dates: list[ql.Date], payment_freq: str, fixing_freq: str) -> list[ql.Date]:
    pay_val, pay_unit = _period_to_canonical(payment_freq)
    fix_val, fix_unit = _period_to_canonical(fixing_freq)

    if pay_unit != fix_unit:
        raise ValueError(
            f"Cannot mix day-based and month-based frequencies: "
            f"payment_freq={payment_freq}, fixing_freq={fixing_freq}"
        )

    if fix_val < pay_val:
        raise ValueError("Assumption violated: payment_freq must be <= fixing_freq")

    if fix_val % pay_val != 0:
        raise ValueError(
            f"Fixing freq {fixing_freq} not divisible by payment freq {payment_freq}"
        )

    block = fix_val // pay_val
    n = len(fixing_dates)

    out = []
    for i in range(n):
        anchor_i = (i // block) * block
        out.append(fixing_dates[anchor_i])
    return out

# def gen_orgin_sched_loan_fin_inst(row, report_date, disc_curves, fwd_curves, fixing_history):
#     r_dt_ql = ql.Date.from_date(report_date)
#
#     cal = get_calendar_from_currency(row.currency)
#     bdc = getattr(ql, row.b_day_conv)
#     dc_conv = get_dc_conv_from_str(row.dc_conv)
#
#     start_dt_ql = ql.Date.from_date(row.start_date)
#     end_dt_ql = ql.Date.from_date(row.maturity_date)
#
#     # --- payment schedule ---
#     p_freq = ql.Period(row.payment_freq)
#     pay_dates = payment_dates_advance_n(start_dt_ql, end_dt_ql, p_freq, cal, bdc)
#
#     # znajdź pierwszy okres po report_date
#     k = bisect_left(pay_dates, r_dt_ql)
#     start_idx = max(k - 1, 0)
#     pay_dates_filt = pay_dates[start_idx:]
#
#     if not pay_dates_filt:
#         return None
#
#     # --- start/end okresu odsetkowego ---
#     if start_idx > 0:
#         first_prev = pay_dates[start_idx - 1]
#     else:
#         first_prev = start_dt_ql
#
#     cf_start_dates = [first_prev] + pay_dates_filt[:-1]
#     cf_end_dates = pay_dates_filt
#
#     fix_shift = ql.Period(-2, ql.Days)
#     fixing_dates = [cal.advance(dt, fix_shift, bdc) for dt in cf_start_dates]
#
#     accrual_yf = [
#         dc_conv.yearFraction(d0, d1)
#         for d0, d1 in zip(cf_start_dates, cf_end_dates)
#     ]
#     if row.fixing_freq is None:
#         fixing_dates = [fixing_dates[0]] * len(fixing_dates)
#     else:
#         fixing_dates = freeze_fixing_dates(
#             fixing_dates,
#             payment_freq=row.payment_freq,
#             fixing_freq=row.fixing_freq,
#         )
#
#     result_df = pd.DataFrame(
#         {
#             "schedule_id": [row.schedule_id] * len(cf_end_dates),
#             "rate_index": [row.rate_index] * len(cf_end_dates),
#             "fixing_dt": fixing_dates,
#             "cf_start_dt": cf_start_dates,
#             "cf_end_dt": cf_end_dates,
#             "cf_yf": accrual_yf,
#         }
#     )
#
#     result_df["cf_start_dt"] = ql_column_to_datetime(result_df["cf_start_dt"])
#     result_df["cf_end_dt"] = ql_column_to_datetime(result_df["cf_end_dt"])
#     result_df["fixing_dt"] = ql_column_to_datetime(result_df["fixing_dt"])
#     result_df['join_dt'] = (
#         result_df.groupby('fixing_dt')['cf_start_dt']
#         .transform('min')
#     )
#
#     result_df = result_df.merge(
#         disc_curves.loc[
#             disc_curves['curve_name'] == row.disc_curve,
#             ['curve_name', 'node_date', 'd_f']
#         ],
#         how='left',
#         left_on='cf_end_dt',
#         right_on='node_date'
#     )
#     result_df = result_df.drop(columns=['node_date', 'curve_name'])
#     result_df = result_df.merge(
#         fwd_curves.loc[
#             (fwd_curves['curve_name'] == row.fwd_curve) & (fwd_curves['fixing_freq'] == row.fixing_freq),
#             ['curve_name', 'node_date', 'fwd_rt']
#         ],
#         how='left',
#         left_on='join_dt',
#         right_on='node_date'
#     )
#     mask = result_df["join_dt"] <= config.report_date
#     tmp = result_df.loc[mask, ["join_dt", "rate_index"]].copy()
#     tmp = tmp.merge(
#         fixing_history[["fixing_date", "rate_index", "rate"]],
#         left_on=["join_dt", "rate_index"],
#         right_on=["fixing_date", "rate_index"],
#         how="left",
#     )
#     result_df.loc[mask, "fwd_rt"] = tmp["rate"].to_numpy()/100
#     result_df = result_df.drop(columns=['node_date', 'join_dt', 'curve_name'])
#
#
#     return result_df

def gen_orgin_sched_loan_fin_inst(
    row: Any,
    report_date: pd.Timestamp,
    disc_df: pd.Series,   # MultiIndex: (curve_name, node_date) -> d_f
    fwd_df: pd.Series,    # MultiIndex: (curve_name, fixing_freq, node_date) -> fwd_rt
    fix_df: pd.Series,    # MultiIndex: (fixing_date, rate_index) -> rate
    ir_params: dict = None,
) -> Optional[pd.DataFrame]:
    r_dt_ql = ql.Date.from_date(report_date)

    cal = get_calendar_from_currency(row.currency)
    bdc = getattr(ql, row.b_day_conv)
    dc_conv = get_dc_conv_from_str(row.dc_conv)

    start_dt_ql = ql.Date.from_date(row.start_date)
    end_dt_ql = ql.Date.from_date(row.maturity_date)

    # --- payment schedule ---
    p_freq = ql.Period(row.payment_freq)
    pay_dates = payment_dates_advance_n(start_dt_ql, end_dt_ql, p_freq, cal, bdc)

    # znajdź pierwszy okres po report_date
    k = bisect_left(pay_dates, r_dt_ql)
    start_idx = max(k - 1, 0)
    pay_dates_filt = pay_dates[start_idx:]

    if not pay_dates_filt:
        return None

    # --- start/end okresu odsetkowego ---
    if start_idx > 0:
        first_prev = pay_dates[start_idx - 1]
    else:
        first_prev = start_dt_ql

    cf_start_dates = [first_prev] + pay_dates_filt[:-1]
    cf_end_dates = pay_dates_filt

    fix_shift = ql.Period(-2, ql.Days)
    fixing_dates = [cal.advance(dt, fix_shift, bdc) for dt in cf_start_dates]

    accrual_yf = [dc_conv.yearFraction(d0, d1) for d0, d1 in zip(cf_start_dates, cf_end_dates)]

    if row.fixing_freq is None:
        fixing_dates = [fixing_dates[0]] * len(fixing_dates)
    else:
        fixing_dates = freeze_fixing_dates(
            fixing_dates,
            payment_freq=row.payment_freq,
            fixing_freq=row.fixing_freq,
        )

    # delay shift: if product has a delay, shift cf_start_dt back by that many months
    pc = int(row.product_code) if hasattr(row, 'product_code') else None
    delay_months = (ir_params or {}).get(pc, {}).get('delay', 0) if pc is not None else 0

    if delay_months:
        delay_period = ql.Period(-delay_months, ql.Months)
        cf_start_dates_delay = [cal.advance(dt, delay_period, bdc) for dt in cf_start_dates]
    else:
        cf_start_dates_delay = cf_start_dates

    result_df = pd.DataFrame(
        {
            "schedule_id": [row.schedule_id] * len(cf_end_dates),
            "rate_index": [row.rate_index] * len(cf_end_dates),
            "fixing_dt": fixing_dates,
            "cf_start_dt": cf_start_dates,
            "cf_start_dt_delay": cf_start_dates_delay,
            "cf_end_dt": cf_end_dates,
            "cf_yf": accrual_yf,
        }
    )

    # QuantLib.Date -> datetime
    result_df["cf_start_dt"] = ql_column_to_datetime(result_df["cf_start_dt"])
    result_df["cf_start_dt_delay"] = ql_column_to_datetime(result_df["cf_start_dt_delay"])
    result_df["cf_end_dt"] = ql_column_to_datetime(result_df["cf_end_dt"])
    result_df["fixing_dt"] = ql_column_to_datetime(result_df["fixing_dt"])

    # join_dt = min(cf_start_dt_delay) per fixing_dt — uses delay-shifted date for fwd_rt lookup
    result_df["join_dt"] = (
        result_df.groupby("fixing_dt")["cf_start_dt_delay"]
        .transform("min")
    )

    n = len(result_df)

    # --- discount factors lookup (curve_name, node_date) -> d_f
    disc_idx = pd.MultiIndex.from_arrays(
        [[row.disc_curve] * n, result_df["cf_end_dt"].to_numpy()],
        names=["curve_name", "node_date"],
    )
    result_df["d_f"] = disc_df.reindex(disc_idx).to_numpy()

    # --- forward rates lookup (curve_name, fixing_freq, node_date) -> fwd_rt
    fwd_idx = pd.MultiIndex.from_arrays(
        [[row.fwd_curve] * n, [row.fixing_freq] * n, result_df["join_dt"].to_numpy()],
        names=["curve_name", "fixing_freq", "node_date"],
    )
    result_df["fwd_rt"] = fwd_df.reindex(fwd_idx).to_numpy()

    # --- overwrite z fixing_history dla join_dt <= report_date
    mask = result_df["fixing_dt"] <= report_date  # albo config.report_date
    if mask.any():
        fix_idx = pd.MultiIndex.from_arrays(
            [
                result_df.loc[mask, "fixing_dt"].to_numpy(),  # <-- tu zmiana
                result_df.loc[mask, "rate_index"].to_numpy(),
            ],
            names=["fixing_date", "rate_index"],
        )
        result_df.loc[mask, "fwd_rt"] = fix_df.reindex(fix_idx).to_numpy() / 100.0

    result_df = result_df.drop(columns=["join_dt"])
    return result_df

def gen_deposit_sched(
    row,
    report_date: pd.Timestamp,
    disc_df: pd.Series,
) -> Optional[pd.DataFrame]:
    """Generate a single-row origin schedule for a deposit.

    Term deposits   — one bullet payment (capital + interest) at maturity_date.
    Non-maturity    — maturity_date is NULL; overnight maturity (next business day).

    Rate is implied from the discount curve: fwd_rt = (1/d_f - 1) / cf_yf.
    """
    r_dt_ql = ql.Date.from_date(report_date)
    cal     = get_calendar_from_currency(row.currency)
    bdc     = getattr(ql, row.b_day_conv)
    dc_conv = get_dc_conv_from_str(row.dc_conv)

    is_non_maturity = pd.isnull(row.maturity_date)

    if is_non_maturity:
        start_dt_ql = r_dt_ql
        end_dt_ql   = cal.advance(r_dt_ql, ql.Period(1, ql.Days), bdc)
    else:
        end_dt_ql = ql.Date.from_date(row.maturity_date)
        if end_dt_ql <= r_dt_ql:
            return None  # already matured
        start_dt_ql = r_dt_ql

    cf_yf   = dc_conv.yearFraction(start_dt_ql, end_dt_ql)
    fix_shift  = ql.Period(-2, ql.Days)
    fix_dt_ql  = cal.advance(start_dt_ql, fix_shift, bdc)

    cf_start = pd.Timestamp(datetime.date(start_dt_ql.year(), start_dt_ql.month(), start_dt_ql.dayOfMonth()))
    cf_end   = pd.Timestamp(datetime.date(end_dt_ql.year(),   end_dt_ql.month(),   end_dt_ql.dayOfMonth()))
    fix_dt   = pd.Timestamp(datetime.date(fix_dt_ql.year(),   fix_dt_ql.month(),   fix_dt_ql.dayOfMonth()))

    d_f = disc_df.get((row.disc_curve, cf_end), np.nan)

    # Market reference rate implied by disc factor: fwd_rt = (1/d_f - 1) / yf
    if cf_yf > 0 and not np.isnan(d_f) and d_f > 0:
        fwd_rt = (1.0 / d_f - 1.0) / cf_yf
    else:
        fwd_rt = np.nan

    # Client rate is set at balance generation (a * market_rate + b); use for int_pmt
    client_rt = float(row.client_rt) if not pd.isnull(row.client_rt) else fwd_rt

    bal = float(row.balance_amt)

    return pd.DataFrame({
        "schedule_id":    [row.schedule_id],
        "bs_side":        [row.bs_side],
        "fixing_dt":      [fix_dt],
        "cf_start_dt":    [cf_start],
        "cf_end_dt":      [cf_end],
        "cf_yf":          [cf_yf],
        "d_f":            [d_f],
        "fwd_rt":         [fwd_rt],
        "client_rt":      [client_rt],
        "outstanding_bal":[bal],
        "int_pmt":        [bal * client_rt * cf_yf if not np.isnan(client_rt) else np.nan],
        "capital_pmt":    [bal],
    })


def tenor_str_to_ql_period(tenor: str) -> ql.Period:
    """Convert tenor string like '1D', '3M', '2Y' to QuantLib Period."""
    tenor = tenor.strip().upper()
    n = int(tenor[:-1])
    unit = tenor[-1]
    unit_map = {'D': ql.Days, 'W': ql.Weeks, 'M': ql.Months, 'Y': ql.Years}
    if unit not in unit_map:
        raise ValueError(f"Unsupported tenor unit: {tenor}")
    return ql.Period(n, unit_map[unit])


def compute_deposit_beh_schedule(
    row,
    report_date: pd.Timestamp,
    disc_df: pd.Series,
    models_depo: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Generate behavioural cash flow schedule for a deposit group.

    Uses models_deposit outstanding percentages (fraction of original balance
    remaining at each tenor) to derive capital payments.

    For each tenor k (sorted chronologically):
        outstanding_bal(k) = balance_amt * pct(k-1)   where pct(0) = 1.0
        capital_pmt(k)     = balance_amt * (pct(k-1) - pct(k))
        fwd_rt(k)          = marginal forward rate for period [start_k, end_k]
        int_pmt(k)         = outstanding_bal(k) * fwd_rt(k) * cf_yf(k)

    Example: balance=1M, pct[1D]=1.0, pct[1M]=0.7
        period 1D: outstanding_bal=1M, capital_pmt=0, int_pmt=1M*r*yf
        period 1M: outstanding_bal=1M, capital_pmt=0.3M, int_pmt=1M*r*yf
    """
    if models_depo.empty:
        return None

    r_dt_ql = ql.Date.from_date(report_date)
    cal = get_calendar_from_currency(row.currency)
    bdc = getattr(ql, row.b_day_conv)
    dc_conv = get_dc_conv_from_str(row.dc_conv)

    # resolve end dates and sort tenors chronologically
    models_depo = models_depo.copy().reset_index(drop=True)
    end_dates_ql = [
        cal.advance(r_dt_ql, tenor_str_to_ql_period(t), bdc)
        for t in models_depo['tenor']
    ]
    n_days = [int(d - r_dt_ql) for d in end_dates_ql]
    order = sorted(range(len(n_days)), key=lambda i: n_days[i])
    models_depo = models_depo.iloc[order].reset_index(drop=True)
    end_dates_ql = [end_dates_ql[i] for i in order]

    n = len(models_depo)
    bal = float(row.balance_amt)
    pct = models_depo['outstanding'].to_numpy(dtype=float)
    start_dates_ql = [r_dt_ql] + end_dates_ql[:-1]

    # client_rt is constant for all behavioural periods (rate set at balance generation)
    client_rt_val = float(row.client_rt) if not pd.isnull(row.client_rt) else np.nan

    rows = []
    for i in range(n):
        cf_start_ql = start_dates_ql[i]
        cf_end_ql = end_dates_ql[i]
        cf_yf = dc_conv.yearFraction(cf_start_ql, cf_end_ql)

        cf_start = pd.Timestamp(
            datetime.date(cf_start_ql.year(), cf_start_ql.month(), cf_start_ql.dayOfMonth())
        )
        cf_end = pd.Timestamp(
            datetime.date(cf_end_ql.year(), cf_end_ql.month(), cf_end_ql.dayOfMonth())
        )

        d_f_start = disc_df.get((row.disc_curve, cf_start), 1.0) if i > 0 else 1.0
        d_f_end = disc_df.get((row.disc_curve, cf_end), np.nan)

        # fwd_rt: marginal market rate for this period (kept for reference / basis risk)
        if cf_yf > 0 and not np.isnan(d_f_end) and d_f_end > 0 and d_f_start > 0:
            fwd_rt = (d_f_start / d_f_end - 1.0) / cf_yf
        else:
            fwd_rt = np.nan

        # Use client_rt if available, fall back to fwd_rt
        rate_for_int = client_rt_val if not np.isnan(client_rt_val) else fwd_rt

        pct_prev = 1.0 if i == 0 else pct[i - 1]
        pct_curr = pct[i]

        outstanding = bal * pct_prev
        capital = bal * (pct_prev - pct_curr)
        interest = outstanding * rate_for_int * cf_yf if not np.isnan(rate_for_int) else np.nan

        rows.append({
            'schedule_id':    row.schedule_id,
            'bs_side':        row.bs_side,
            'cf_start_dt':    cf_start,
            'cf_end_dt':      cf_end,
            'cf_yf':          cf_yf,
            'd_f':            d_f_end,
            'fwd_rt':         fwd_rt,
            'client_rt':      client_rt_val,
            'outstanding_bal': outstanding,
            'capital_pmt':    capital,
            'int_pmt':        interest,
        })

    return pd.DataFrame(rows) if rows else None


def payment_dates_advance_n(
    start_date: ql.Date,
    end_date: ql.Date,
    freq: ql.Period,
    calendar: ql.Calendar,
    bdc: int,
    eom: bool = False,
) -> list[ql.Date]:
    # Advance by `freq` directly rather than pre-counting via a month-diff
    # heuristic — the old month-diff count only made sense for Month/Year
    # periods and silently produced zero dates for Day/Week periods (e.g.
    # a 7D payment_freq), since a month-level date diff is ~0 within a
    # single-month schedule regardless of the actual day-level frequency.
    dates = []
    i = 1
    while True:
        d = calendar.advance(start_date, i * freq, bdc, eom)
        if d > end_date:
            break
        dates.append(d)
        i += 1
    return dates

def ql_column_to_datetime(series: pd.Series) -> pd.Series:
    """Convert pandas Series of QuantLib.Date -> pandas datetime."""
    return pd.to_datetime(
        [
            datetime.date(d.year(), d.month(), d.dayOfMonth())
            if isinstance(d, ql.Date) and d != ql.Date()
            else None
            for d in series.to_numpy()
        ],
        errors="coerce",
    )

def get_unique_curves() -> pd.DataFrame:
    uniq_discount_curves = sql_setup.sql_get_uniq_curves(['loans', 'fin_inst'], 'disc_curve')
    uniq_forward_curves  = sql_setup.sql_get_uniq_curves(['loans', 'fin_inst'], 'fwd_curve')
    return pd.concat([uniq_discount_curves, uniq_forward_curves])

def get_interpolated_curves(row: pd.Series) -> pd.DataFrame:
    if row['curve_type'] == 'fwd_curve':
        curve_df = get_daily_intrpl_curve(row['curve_name'], row['fixing_freq'])
    elif row['curve_type'] == 'disc_curve':
        curve_df = get_daily_intrpl_curve(row['curve_name'])
    return curve_df

def get_daily_intrpl_curve(curve_name: str, fixing_freq: Optional[str] = None) -> pd.DataFrame:
    df_sql = sql_setup.sql_select_specific_curve(curve_name)
    x_s = df_sql['n_days']
    y_s = df_sql['d_f']
    x = x_s.to_numpy(dtype=float)
    y = y_s.to_numpy(dtype=float)
    df_intpl = interpolate_d_f(x, y, config.report_date)
    df_intpl.insert(0, "curve_name", [curve_name] * len(df_intpl))

    ## if forward rate additionaly take fixing_freq into account
    if fixing_freq:
        df_intpl["fixing_freq"] =  [fixing_freq] * len(df_intpl)
        ccy = curve_name[:3]
        cal = get_calendar_from_currency(ccy)
        dc_conv = get_dc_from_currency(ccy)
        df_intpl['end_dt'] = [cal.advance(ql.Date.from_date(dt), ql.Period(fixing_freq), ql.ModifiedFollowing)
                  for dt in df_intpl['node_date'].dt.to_pydatetime()]
        df_intpl['year_frac'] = [dc_conv.yearFraction(ql.Date.from_date(s_dt), e_dt) for s_dt, e_dt
                                 in zip(df_intpl['node_date'], df_intpl['end_dt'])]
        df_helper = df_intpl.copy()[['node_date', 'd_f']].rename(columns={'node_date': 'date', 'd_f': 'd_f_2'})
        df_intpl["end_dt"] = pd.to_datetime([d.to_date() for d in df_intpl["end_dt"]])
        df_intpl = df_intpl.merge(df_helper, how= 'left',
                                          left_on= 'end_dt', right_on= 'date')
        df_intpl['fwd_rt'] = (df_intpl['d_f']/df_intpl['d_f_2'] - 1)/df_intpl['year_frac']
        df_intpl = df_intpl.drop(columns= ['date', 'end_dt', 'year_frac', 'd_f', 'd_f_2'])
        df_intpl = df_intpl.dropna(subset=['fwd_rt'])
    return df_intpl

def interpolate_d_f(x: np.ndarray, y: np.ndarray, report_date: pd.Timestamp) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if np.any(x == 0):
        y[x == 0] = 1.0
    else:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 1.0)

    grid_days = np.arange(0, int(x.max()) + 1, dtype=float)

    ln_y = np.log(y)
    ln_y_interp = np.interp(grid_days, x, ln_y)
    d_f_interp = np.exp(ln_y_interp)

    return pd.DataFrame({
        "n_days": grid_days.astype(int),
        "node_date": report_date + pd.to_timedelta(grid_days, unit="D"),
        "d_f": d_f_interp
    })


def compute_amort_schedule_vectorized(
    sched_df: pd.DataFrame,
    sched_params_df: pd.DataFrame,
    ir_params: dict | None = None,
    exact: bool = True,
) -> pd.DataFrame:
    """Vectorized computation of outstanding balance, interest and capital payments.

    Parameters
    ----------
    sched_df:
        Output of gen_orgin_sched_loan_fin_inst.
        Must contain: schedule_id, cf_end_dt, cf_yf, fwd_rt.
    sched_params_df:
        One row per schedule_id. Must contain: schedule_id, balance_amt, amort_type.
    exact : bool, default True
        True  — exact cumprod recursion per schedule_id (groupby.apply, ~1 Python
                call per schedule).  capital + int_pmt == annuity_pmt every period.
        False — fast 1-iter vectorised approximation (pure numpy, zero Python loops).
                r_const = mean(fwd_rt * cf_yf) per schedule; closed-form O(k).
                capital = A - O*r_const (consistent with O formula but not with
                actual int_pmt = O*fwd_rt*cf_yf).  Use for Monte-Carlo / scenarios.

    Returns
    -------
    sched_df with new columns: outstanding_bal, capital_pmt, int_pmt, annuity_pmt.
    annuity_pmt is NaN for bullet and constant-amort schedules.
    """
    PARAM_COLS = ['schedule_id', 'balance_amt', 'amort_type', 'bs_side']
    # margin only present for loans; product_code for both loans and fin_inst
    avail_params = PARAM_COLS + [c for c in ['margin', 'product_code'] if c in sched_params_df.columns]
    df = sched_df.merge(sched_params_df[avail_params], on='schedule_id', how='left')
    if 'margin' not in df.columns:
        df['margin'] = np.nan
    df = df.sort_values(['schedule_id', 'cf_end_dt']).reset_index(drop=True)

    # ── Floor parameters from ir_params (index_floor applied to input; client_floor to output) ──
    df['_idx_fl'] = np.nan
    df['_cl_fl']  = np.nan
    if ir_params and 'product_code' in df.columns:
        def _fl(pc, key):
            if pd.isna(pc):
                return np.nan
            v = ir_params.get(int(pc), {}).get(key)
            return float(v) if v is not None else np.nan
        _pc = pd.to_numeric(df['product_code'], errors='coerce')
        df['_idx_fl'] = [_fl(pc, 'index_floor') for pc in _pc]
        df['_cl_fl']  = [_fl(pc, 'client_floor') for pc in _pc]

    # ── Effective forward rate after applying index floor ────────────────────
    _eff = df['fwd_rt'].to_numpy(dtype=float)
    _ifl = df['_idx_fl'].to_numpy(dtype=float)
    _eff = np.where(np.isnan(_ifl), _eff, np.maximum(_eff, _ifl))
    df['_eff_fwd'] = _eff

    df['amort_type'] = pd.to_numeric(df['amort_type'], errors='coerce').fillna(0).astype(int)

    df['_rank']   = df.groupby('schedule_id').cumcount()
    df['_n']      = df.groupby('schedule_id')['_rank'].transform('max') + 1
    df['_is_last']= df['_rank'] == df['_n'] - 1

    df['outstanding_bal'] = np.nan
    df['capital_pmt']     = np.nan
    df['annuity_pmt']     = np.nan

    # ── BULLET (amort_type == 0) ─────────────────────────────────────────────
    m0 = df['amort_type'] == 0
    df.loc[m0, 'outstanding_bal'] = df.loc[m0, 'balance_amt']
    df.loc[m0, 'capital_pmt'] = np.where(
        df.loc[m0, '_is_last'], df.loc[m0, 'balance_amt'], 0.0
    )

    # ── CONSTANT AMORTIZATION (amort_type == 2) ──────────────────────────────
    m2 = df['amort_type'] == 2
    step = df.loc[m2, 'balance_amt'] / df.loc[m2, '_n']
    df.loc[m2, 'capital_pmt']     = step
    df.loc[m2, 'outstanding_bal'] = df.loc[m2, 'balance_amt'] - step * df.loc[m2, '_rank']

    # ── ANNUITY (amort_type == 1) ─────────────────────────────────────────────
    m1 = df['amort_type'] == 1
    if m1.any():
        if exact:
            # ── EXACT: cumprod recursion, one groupby.apply call per schedule ──
            # g_k = 1 + fwd_rt_k * cf_yf_k
            # G_k = g_0 * ... * g_{k-1},  G_0 = 1
            # A   = B / sum(1/G_k  for k=1..n)
            # O_k = G_k * (B - A * sum(1/G_j  for j=1..k))
            # C_k = A - O_k * fwd_rt_k * cf_yf_k     → capital + int_pmt == A
            def _exact_annuity_group(group: pd.DataFrame) -> pd.DataFrame:
                group = group.sort_values('cf_end_dt').reset_index(drop=True)
                fwd        = group['_eff_fwd'].to_numpy(dtype=float)
                margin_val = float(group['margin'].fillna(0.0).iloc[0])
                client_rt  = fwd + margin_val
                yf         = group['cf_yf'].to_numpy(dtype=float)
                fixing_dt  = group['fixing_dt'].to_numpy()
                B          = float(group['balance_amt'].iloc[0])
                n          = len(group)

                outstanding = np.zeros(n)
                capital     = np.zeros(n)
                annuity     = np.zeros(n)

                O_curr = B
                k = 0
                while k < n:
                    r = client_rt[k]
                    # Recompute annuity at this fixing: assume rate r stays constant
                    # for all remaining periods, but use exact per-period cf_yf.
                    yf_rem = yf[k:]
                    G_rem  = np.cumprod(1.0 + r * yf_rem)
                    A      = O_curr / (1.0 / G_rem).sum()

                    # Advance to end of this fixing segment
                    current_fix = fixing_dt[k]
                    seg_end = k + 1
                    while seg_end < n and fixing_dt[seg_end] == current_fix:
                        seg_end += 1

                    for i in range(k, seg_end):
                        outstanding[i] = O_curr
                        cap_i          = max(A - O_curr * r * yf[i], 0.0)
                        capital[i]     = cap_i
                        annuity[i]     = A
                        O_curr         = O_curr - cap_i

                    k = seg_end

                out = group.copy()
                out['outstanding_bal'] = outstanding
                out['capital_pmt']     = capital
                out['annuity_pmt']     = annuity
                return out

            filled = (
                df[m1]
                .groupby('schedule_id', group_keys=False)
                .apply(_exact_annuity_group)
            )
            df.loc[m1, 'outstanding_bal'] = filled['outstanding_bal'].to_numpy()
            df.loc[m1, 'capital_pmt']     = filled['capital_pmt'].to_numpy()
            df.loc[m1, 'annuity_pmt']     = filled['annuity_pmt'].to_numpy()

        else:
            # ── APPROX (exact=False): fully vectorised 1-iter, zero Python loops ─
            # r_const = mean(client_rt * cf_yf) per schedule  →  one groupby.transform
            # closed-form O(k) = B*(1+r)^k - A*((1+r)^k-1)/r  (pure numpy power)
            # capital = A - O*r_const  (consistent with O formula)
            # int_pmt computed from actual client_rt at the end  → capital+int ≠ A
            fwd1     = (df.loc[m1, '_eff_fwd'] + df.loc[m1, 'margin'].fillna(0.0)).to_numpy(dtype=float)
            yf1      = df.loc[m1, 'cf_yf'].to_numpy(dtype=float)
            n1       = df.loc[m1, '_n'].to_numpy(dtype=float)
            B1       = df.loc[m1, 'balance_amt'].to_numpy(dtype=float)
            k1       = df.loc[m1, '_rank'].to_numpy(dtype=float)
            sid1     = df.loc[m1, 'schedule_id'].to_numpy()

            r_const = (
                pd.Series(fwd1 * yf1, index=sid1)
                .groupby(level=0).transform('mean')
                .to_numpy(dtype=float)
            )
            safe_r = np.where(r_const > 1e-10, r_const, 1e-10)
            A1 = np.where(r_const > 1e-10,
                          B1 * safe_r / (1.0 - (1.0 + safe_r) ** (-n1)),
                          B1 / n1)
            factor = (1.0 + safe_r) ** k1
            O1 = np.where(r_const > 1e-10,
                          B1 * factor - A1 * (factor - 1.0) / safe_r,
                          B1 * (1.0 - k1 / n1))
            O1 = np.maximum(O1, 0.0)
            C1 = np.maximum(A1 - O1 * r_const, 0.0)

            df.loc[m1, 'outstanding_bal'] = O1
            df.loc[m1, 'capital_pmt']     = C1
            df.loc[m1, 'annuity_pmt']     = A1

    # ── client_rt = eff_fwd + margin, then apply client_floor if set ─────────
    df['client_rt'] = df['_eff_fwd'] + df['margin'].fillna(0.0)
    _cl = df['_cl_fl'].to_numpy(dtype=float)
    df['client_rt'] = np.where(
        np.isnan(_cl),
        df['client_rt'].to_numpy(float),
        np.maximum(df['client_rt'].to_numpy(float), _cl),
    )

    # ── Interest for all amort types ──────────────────────────────────────────
    df['int_pmt'] = df['outstanding_bal'] * df['client_rt'] * df['cf_yf']

    # ── total_pmt: rename annuity_pmt, fill NaN for non-annuity types ─────────
    df['_is_annuity'] = df['annuity_pmt'].notna()
    df['annuity_pmt'] = df['annuity_pmt'].fillna(df['capital_pmt'] + df['int_pmt'])
    df = df.rename(columns={'annuity_pmt': 'total_pmt'})

    drop_cols = ['_rank', '_n', '_is_last', 'balance_amt', 'amort_type', '_eff_fwd', '_idx_fl', '_cl_fl']
    if 'product_code' in df.columns:
        drop_cols.append('product_code')
    df = df.drop(columns=drop_cols)
    return df

def compute_adjusted_schedule(
    sched_df: pd.DataFrame,
    exact: bool = True,
) -> pd.DataFrame:
    """Compute prepayment-adjusted schedule on top of a contractual annuity schedule.

    Must be called AFTER compute_amort_schedule_vectorized because it needs the
    annuity_pmt column produced there.

    Parameters
    ----------
    sched_df:
        Output of compute_amort_schedule_vectorized.
        Must contain: schedule_id, fwd_rt, cf_yf, outstanding_bal, annuity_pmt,
        cpr_rate.  Only rows where annuity_pmt is not NaN are processed.
    exact : bool, default True
        True  — exact cumprod recursion per schedule (groupby.apply).
                h_k = 1 + fwd_rt_k*cf_yf_k - cpr_rate_k  (per-period)
                capital_adj + prepayment + int_adj == annuity_pmt every period.
        False — fully vectorised approximation (pure numpy, groupby.transform only).
                r_adj = mean(fwd_rt*cf_yf) - cpr_const  per schedule (both scalars).
                Closed-form O_adj(k) = B*(1+r_adj)^k - A*((1+r_adj)^k-1)/r_adj.
                capital_adj = A - O_adj*r_adj  (consistent with O formula).
                Assumes cpr_rate is constant within each schedule_id.
                Use for Monte-Carlo / scenarios where speed matters.

    Returns
    -------
    sched_df with new columns:
        outstanding_adj  — adjusted outstanding balance
        capital_adj      — contractual capital portion
        prepayment_pmt   — prepayment cash flow  (= cpr_rate * outstanding_adj)
        int_adj          — interest on adjusted outstanding
    """
    df = sched_df.copy()
    df['outstanding_adj'] = np.nan
    df['capital_adj']     = np.nan
    df['prepayment_pmt']  = np.nan
    df['int_adj']         = np.nan

    m1 = df['_is_annuity']
    if not m1.any():
        return df

    if exact:
        # ── EXACT: cumprod with per-period h_k = g_k - cpr_k (groupby.apply) ─
        def _adjusted_group(group: pd.DataFrame) -> pd.DataFrame:
            client_rt = group['client_rt'].to_numpy(dtype=float)
            yf  = group['cf_yf'].to_numpy(dtype=float)
            cpr = group['cpr_rate'].to_numpy(dtype=float)
            A   = float(group['total_pmt'].iloc[0])
            B   = float(group['outstanding_bal'].iloc[0])
            n   = len(group)

            h = 1.0 + client_rt * yf - cpr
            H = np.empty(n + 1); H[0] = 1.0
            np.cumprod(h, out=H[1:])

            Q = 1.0 / H
            cum_Q = np.empty(n); cum_Q[0] = 0.0
            if n > 1:
                np.cumsum(Q[1:n], out=cum_Q[1:])

            O  = np.maximum(H[:n] * (B - A * cum_Q), 0.0)
            I  = O * client_rt * yf
            C  = np.where(O > 0, np.maximum(A - I, 0.0), 0.0)
            PP = cpr * O

            out = group.copy()
            out['outstanding_adj'] = O
            out['capital_adj']     = C
            out['prepayment_pmt']  = PP
            out['int_adj']         = I
            return out

        filled = (
            df[m1]
            .groupby('schedule_id', group_keys=False)
            .apply(_adjusted_group)
        )
        for col in ['outstanding_adj', 'capital_adj', 'prepayment_pmt', 'int_adj']:
            df.loc[m1, col] = filled[col].to_numpy()

    else:
        # ── APPROX: fully vectorised, zero Python loops (groupby.transform only) ─
        # r_adj = r_const - cpr_const  where both are one scalar per schedule.
        # Closed-form: O_adj(k) = B*(1+r_adj)^k - A*((1+r_adj)^k - 1)/r_adj
        # capital_adj  = A - O_adj * r_adj   (consistent with O formula)
        # int_adj      = O_adj * fwd_rt * cf_yf  (actual rate)
        # Assumes cpr_rate is constant within each schedule_id.
        sub = df[m1]

        rank = sub.groupby('schedule_id').cumcount().to_numpy(dtype=float)
        n    = sub.groupby('schedule_id')['schedule_id'].transform('count').to_numpy(dtype=float)

        client_rt = sub['client_rt'].to_numpy(dtype=float)
        yf   = sub['cf_yf'].to_numpy(dtype=float)
        A    = sub['total_pmt'].to_numpy(dtype=float)
        B    = sub.groupby('schedule_id')['outstanding_bal'].transform('first').to_numpy(dtype=float)
        sid  = sub['schedule_id'].to_numpy()

        r_const = (
            pd.Series(client_rt * yf, index=sid)
            .groupby(level=0).transform('mean')
            .to_numpy(dtype=float)
        )
        cpr_const = sub.groupby('schedule_id')['cpr_rate'].transform('first').to_numpy(dtype=float)
        r_adj = r_const - cpr_const

        near_zero = np.abs(r_adj) <= 1e-10
        safe_r    = np.where(near_zero, 1e-10, r_adj)
        factor    = (1.0 + safe_r) ** rank

        O  = np.where(near_zero,
                      np.maximum(B - A * rank, 0.0),
                      np.maximum(B * factor - A * (factor - 1.0) / safe_r, 0.0))
        C  = np.where(O > 0, np.maximum(A - O * r_adj, 0.0), 0.0)
        PP = cpr_const * O
        I  = O * client_rt * yf

        df.loc[m1, 'outstanding_adj'] = O
        df.loc[m1, 'capital_adj']     = C
        df.loc[m1, 'prepayment_pmt']  = PP
        df.loc[m1, 'int_adj']         = I

    # Non-annuity rows (bullet, linear, etc.) — CPR adjustment is not applicable;
    # carry contractual values through so the beh_df filter (outstanding_bal > 0) keeps them.
    m_noann = ~df['_is_annuity']
    if m_noann.any():
        df.loc[m_noann, 'outstanding_adj'] = df.loc[m_noann, 'outstanding_bal']
        df.loc[m_noann, 'capital_adj']     = df.loc[m_noann, 'capital_pmt']
        df.loc[m_noann, 'prepayment_pmt']  = 0.0
        df.loc[m_noann, 'int_adj']         = df.loc[m_noann, 'int_pmt']

    return df


def exact_annuity_loop(
    sched_df: pd.DataFrame,
    schedule_id: Any,
    balance_amt: float,
) -> pd.DataFrame:
    """Exact annuity schedule using actual per-period forward rates.

    The constant annuity payment A is derived from the PV condition using the
    real fwd_rt for each period — no constant-rate approximation.

    Math
    ----
    Let g_k = 1 + fwd_rt_k * cf_yf_k   (simple growth factor for period k)
    Let P_0 = 1,  P_k = P_{k-1} / g_{k-1}   (cumulative discount to period k)

    PV condition:  B = A * sum(P_k  for k = 1..n)
    => A = B / sum(P_k for k = 1..n)

    Recursion:
        O_0 = B
        I_k = O_k * fwd_rt_k * cf_yf_k
        C_k = A - I_k
        O_{k+1} = O_k - C_k   (= O_k * g_k - A)

    Parameters
    ----------
    sched_df:
        Output of gen_orgin_sched_loan_fin_inst; must contain fwd_rt, cf_yf.
    schedule_id:
        The schedule to compute.
    balance_amt:
        Initial outstanding balance.

    Returns
    -------
    sched_df filtered to schedule_id with added columns:
        outstanding_bal, int_pmt, capital_pmt, annuity_pmt
    """
    df = (
        sched_df[sched_df['schedule_id'] == schedule_id]
        .sort_values('cf_end_dt')
        .reset_index(drop=True)
        .copy()
    )
    if df.empty:
        raise ValueError(f"schedule_id={schedule_id!r} not found in sched_df")

    fwd = df['fwd_rt'].to_numpy(dtype=float)
    yf  = df['cf_yf'].to_numpy(dtype=float)
    n   = len(df)
    B   = float(balance_amt)

    # cumulative discount factors P_k = prod(1/g_j for j < k), P_0 = 1
    P = np.empty(n + 1)
    P[0] = 1.0
    for j in range(n):
        P[j + 1] = P[j] / (1.0 + fwd[j] * yf[j])

    # exact constant annuity payment
    A = B / P[1:].sum()

    # recursive outstanding balance
    outstanding = np.empty(n)
    int_pmt     = np.empty(n)
    capital_pmt = np.empty(n)

    O = B
    for k in range(n):
        I = O * fwd[k] * yf[k]
        C = A - I
        outstanding[k] = O
        int_pmt[k]     = I
        capital_pmt[k] = C
        O = O - C

    df['outstanding_bal'] = outstanding
    df['int_pmt']         = int_pmt
    df['capital_pmt']     = capital_pmt
    df['annuity_pmt']     = A
    return df
