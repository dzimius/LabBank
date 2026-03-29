from __future__ import annotations

import datetime
import math
from bisect import bisect_left
from typing import Optional

import numpy as np
import pandas as pd
import QuantLib as ql


# ── QuantLib helpers (mirrors cash_flow_calc/cf_calc_objects.py) ──────────────

def get_calendar_from_currency(curr: str) -> ql.Calendar:
    return {"PLN": ql.Poland(),
            "EUR": ql.TARGET(),
            "USD": ql.UnitedStates(ql.UnitedStates.NYSE)}.get(curr, ql.TARGET())


def get_dc_conv_from_str(s: str) -> ql.DayCounter:
    return {"ACT/365": ql.Actual365Fixed(),
            "ACT/ACT": ql.ActualActual(ql.ActualActual.ISDA),
            "30/360":  ql.Thirty360(ql.Thirty360.ISDA),
            "ACT/360": ql.Actual360()}.get(s, ql.ActualActual(ql.ActualActual.ISDA))


def _ql_to_ts(d: ql.Date) -> pd.Timestamp:
    return pd.Timestamp(datetime.date(d.year(), d.month(), d.dayOfMonth()))


def _payment_dates(start: ql.Date, end: ql.Date,
                   freq: ql.Period, cal: ql.Calendar, bdc: int) -> list[ql.Date]:
    freq_m = freq.length() * (12 if freq.units() == ql.Years else 1)
    diff_m = (end.year() - start.year()) * 12 + (end.month() - start.month())
    n = diff_m // freq_m
    return [cal.advance(start, i * freq, bdc, False) for i in range(1, n + 1)]


def _freeze_fixing_dates(fixing_dates: list[ql.Date],
                         pay_freq: str, fix_freq: str) -> list[ql.Date]:
    def _months(t: str) -> int:
        t = t.strip().upper()
        return int(t[:-1]) * (12 if t.endswith("Y") else 1)

    pay_m, fix_m = _months(pay_freq), _months(fix_freq)
    block = fix_m // pay_m
    return [fixing_dates[(i // block) * block] for i in range(len(fixing_dates))]


# ── IRS leg schedule generators ───────────────────────────────────────────────

def gen_fixed_leg_schedule(
    row,
    report_date: pd.Timestamp,
    disc_df: pd.Series,          # MultiIndex (curve_name, node_date) -> d_f
) -> Optional[pd.DataFrame]:
    """Cash-flow schedule for the fixed leg of an IRS.

    No notional exchange in a vanilla IRS — capital_pmt = 0 for all periods
    (consistent with the floating leg).  The notional repricing at maturity is
    handled inside compute_ir_swap_gap via outstanding_bal, not capital_pmt.
    All fixing_dt values are set to swap start - 2D (always <= report_date) so
    the repricing-gap logic treats this leg as fixed-rate (is_floating = 0).
    """
    r_dt_ql  = ql.Date.from_date(report_date.date())
    cal      = get_calendar_from_currency(row.currency)
    bdc      = getattr(ql, row.b_day_conv)
    dc_conv  = get_dc_conv_from_str(row.dc_conv)

    start_ql = ql.Date.from_date(pd.Timestamp(row.start_date).date())
    end_ql   = ql.Date.from_date(pd.Timestamp(row.maturity_date).date())

    pay_dates = _payment_dates(start_ql, end_ql, ql.Period(row.pay_freq), cal, bdc)
    pay_dates_filt = [d for d in pay_dates if d > r_dt_ql]
    if not pay_dates_filt:
        return None

    k = next(i for i, d in enumerate(pay_dates) if d > r_dt_ql)
    first_start = pay_dates[k - 1] if k > 0 else start_ql
    cf_starts = [first_start] + pay_dates_filt[:-1]
    cf_ends   = pay_dates_filt

    # Original fixing date (always before report_date → is_floating = 0 in gap SQL)
    fix_shift  = ql.Period(-2, ql.Days)
    fixing_ql  = cal.advance(start_ql, fix_shift, bdc)
    fixing_pd  = _ql_to_ts(fixing_ql)

    n        = len(cf_ends)
    notional = float(row.notional)
    rate     = float(row.fixed_rate)

    rows = []
    for i, (s, e) in enumerate(zip(cf_starts, cf_ends)):
        cf_yf     = dc_conv.yearFraction(s, e)
        cf_s_pd   = _ql_to_ts(s)
        cf_e_pd   = _ql_to_ts(e)
        d_f       = disc_df.get((row.disc_curve, cf_e_pd), np.nan)
        int_pmt = notional * rate * cf_yf

        rows.append({
            "schedule_id":    str(row.schedule_id),
            "bs_side":        row.bs_side,
            "rate_index":     "FIXED",
            "fixing_dt":      fixing_pd,
            "cf_start_dt":    cf_s_pd,
            "cf_end_dt":      cf_e_pd,
            "cf_yf":          cf_yf,
            "d_f":            d_f,
            "fwd_rt":         rate,
            "outstanding_bal": notional,
            "capital_pmt":    0.0,        # no notional exchange in IRS
            "int_pmt":        int_pmt,
            "total_pmt":      int_pmt,
        })

    return pd.DataFrame(rows)


def gen_float_leg_schedule(
    row,
    report_date: pd.Timestamp,
    disc_df: pd.Series,          # MultiIndex (curve_name, node_date) -> d_f
    fwd_df:  pd.Series,          # MultiIndex (curve_name, fixing_freq, node_date) -> fwd_rt
    fix_df:  pd.Series,          # MultiIndex (fixing_date, rate_index) -> rate
) -> Optional[pd.DataFrame]:
    """Cash-flow schedule for the floating leg of an IRS.

    No capital exchange — capital_pmt = 0 for all periods.
    outstanding_bal = notional (used by gap SQL for outstanding_at_fix).
    """
    r_dt_ql  = ql.Date.from_date(report_date.date())
    cal      = get_calendar_from_currency(row.currency)
    bdc      = getattr(ql, row.b_day_conv)
    dc_conv  = get_dc_conv_from_str(row.dc_conv)

    start_ql = ql.Date.from_date(pd.Timestamp(row.start_date).date())
    end_ql   = ql.Date.from_date(pd.Timestamp(row.maturity_date).date())

    pay_dates = _payment_dates(start_ql, end_ql, ql.Period(row.pay_freq), cal, bdc)

    k         = bisect_left(pay_dates, r_dt_ql)
    start_idx = max(k - 1, 0)
    pay_dates_filt = pay_dates[start_idx:]
    if not pay_dates_filt:
        return None

    first_prev  = pay_dates[start_idx - 1] if start_idx > 0 else start_ql
    cf_starts   = [first_prev] + list(pay_dates_filt[:-1])
    cf_ends     = list(pay_dates_filt)

    fix_shift    = ql.Period(-2, ql.Days)
    fixing_dates = [cal.advance(d, fix_shift, bdc) for d in cf_starts]

    if row.fixing_freq is not None:
        fixing_dates = _freeze_fixing_dates(fixing_dates, row.pay_freq, row.fixing_freq)

    cf_yf_vec    = [dc_conv.yearFraction(s, e) for s, e in zip(cf_starts, cf_ends)]
    cf_start_pds = [_ql_to_ts(d) for d in cf_starts]
    cf_end_pds   = [_ql_to_ts(d) for d in cf_ends]
    fixing_pds   = [_ql_to_ts(d) for d in fixing_dates]

    n        = len(cf_ends)
    notional = float(row.notional)
    spread   = float(row.float_spread) if pd.notna(row.float_spread) else 0.0

    result = pd.DataFrame({
        "schedule_id":  [str(row.schedule_id)] * n,
        "bs_side":      [row.bs_side] * n,
        "rate_index":   [row.rate_index] * n,
        "fixing_dt":    fixing_pds,
        "cf_start_dt":  cf_start_pds,
        "cf_end_dt":    cf_end_pds,
        "cf_yf":        cf_yf_vec,
    })

    # join_dt: min cf_start_dt per fixing_dt (for fwd-curve lookup)
    result["join_dt"] = result.groupby("fixing_dt")["cf_start_dt"].transform("min")

    # Discount factors
    disc_idx = pd.MultiIndex.from_arrays(
        [[row.disc_curve] * n, result["cf_end_dt"].to_numpy()],
        names=["curve_name", "node_date"],
    )
    result["d_f"] = disc_df.reindex(disc_idx).to_numpy()

    # Forward rates
    fwd_idx = pd.MultiIndex.from_arrays(
        [[row.fwd_curve] * n, [row.fixing_freq] * n, result["join_dt"].to_numpy()],
        names=["curve_name", "fixing_freq", "node_date"],
    )
    result["fwd_rt"] = fwd_df.reindex(fwd_idx).to_numpy()

    # Override with historical fixings for past periods
    mask = result["fixing_dt"] <= report_date
    if mask.any():
        fix_idx = pd.MultiIndex.from_arrays(
            [result.loc[mask, "fixing_dt"].to_numpy(),
             result.loc[mask, "rate_index"].to_numpy()],
            names=["fixing_date", "rate_index"],
        )
        result.loc[mask, "fwd_rt"] = fix_df.reindex(fix_idx).to_numpy() / 100.0

    result = result.drop(columns=["join_dt"])

    result["outstanding_bal"] = notional
    result["capital_pmt"]     = 0.0
    result["int_pmt"]         = notional * (result["fwd_rt"] + spread) * result["cf_yf"]
    result["total_pmt"]       = result["int_pmt"]

    return result


# ── IR repricing gap computation ──────────────────────────────────────────────

_TENOR_SORT = {"1D": 0, ">30Y": 362}


def _month_buckets(
    cf_end_series: pd.Series, report_date: pd.Timestamp
) -> pd.Series:
    """Calendar-month bucket labels — same logic as cash_flow_calc/_month_buckets.

    Date d falls in bucket n where n is the smallest integer >= 1 such that
    d <= report_date + DateOffset(months=n).
    """
    raw = (
        (cf_end_series.dt.year  - report_date.year)  * 12
        + (cf_end_series.dt.month - report_date.month)
    )
    bucket_ends = pd.DatetimeIndex([
        report_date + pd.DateOffset(months=int(m)) for m in raw.clip(lower=0)
    ])
    over = cf_end_series.values > bucket_ends
    months = (raw + over.astype(int)).clip(lower=1)
    return months.map(lambda m: ">30Y" if m > 360 else f"{m}M")


def _tenor_sort_key(label: str) -> int:
    if label in _TENOR_SORT:
        return _TENOR_SORT[label]
    return int(label[:-1])


def _bucket_bounds(
    tenor_bucket: str, report_date: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp | float]:
    """Return (bucket_start_dt, bucket_end_dt) — exact calendar-month boundaries."""
    if tenor_bucket == "1D":
        return report_date, report_date + pd.Timedelta(days=1)
    if tenor_bucket == ">30Y":
        start = report_date + pd.DateOffset(months=360) + pd.Timedelta(days=1)
        return start, pd.NaT
    n = int(tenor_bucket[:-1])
    end   = report_date + pd.DateOffset(months=n)
    start = (report_date + pd.Timedelta(days=2)) if n == 1 else \
            (report_date + pd.DateOffset(months=n - 1) + pd.Timedelta(days=1))
    return start, end


def _aggregate_ir_gap(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.DataFrame:
    """Apply sign convention and aggregate gap_cf by (currency, bs_side, tenor_bucket).

    IRS has no overnight positions, so tenor buckets are assigned purely from the
    actual number of days between report_date and cf_end_dt — no min-date = 1D override.
    """
    df = df.copy()
    df["gap_cf"] = df["gap_cf"] * np.where(df["bs_side"] == "A", 1.0, -1.0)

    df["tenor_bucket"] = _month_buckets(df["cf_end_dt"], report_date)

    result = (
        df.groupby(["currency", "bs_side", "tenor_bucket"], sort=False)[["gap_cf"]]
        .sum()
        .reset_index()
    )

    def _bucket_date(row: pd.Series) -> pd.Timestamp:
        if row["tenor_bucket"] == "1D":
            return report_date + pd.Timedelta(days=1)
        if row["tenor_bucket"] == ">30Y":
            return report_date + pd.offsets.MonthEnd(361)
        return report_date + pd.offsets.MonthEnd(int(row["tenor_bucket"][:-1]))

    result["cf_end_dt"] = result.apply(_bucket_date, axis=1)

    bounds = result["tenor_bucket"].map(lambda t: _bucket_bounds(t, report_date))
    result["bucket_start_dt"] = [b[0] for b in bounds]
    result["bucket_end_dt"]   = [b[1] for b in bounds]

    result["_sort"] = result["tenor_bucket"].map(_tenor_sort_key)
    result = (
        result.sort_values(["currency", "bs_side", "_sort"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    return result[["currency", "bs_side", "tenor_bucket",
                   "bucket_start_dt", "bucket_end_dt", "cf_end_dt", "gap_cf"]]


def compute_ir_swap_gap(
    cf_df: pd.DataFrame,
    sched_df: pd.DataFrame,
    report_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute the IR repricing gap contribution from IRS CF schedules.

    Fixed leg (rate_index='FIXED', all fixing_dt <= report_date → is_floating=0):
        gap_cf = capital_pmt + int_pmt at each cf_end_dt.

    Floating leg (has future fixing_dt → is_floating=1):
        - Locked current interest: int_pmt of the first (current) period.
        - Outstanding at next fixing date: outstanding_bal placed at next_fixing_dt.
        - Past capital payments (=0 for IRS): contribute nothing.

    Returns DataFrame with columns: currency, bs_side, tenor_bucket, cf_end_dt, gap_cf.
    """
    # cf_df already carries bs_side — only join for currency and leg_type
    sched_lookup = (
        sched_df[["schedule_id", "currency", "leg_type"]]
        .copy()
        .assign(schedule_id=sched_df["schedule_id"].astype(str))
    )
    df = cf_df.merge(sched_lookup, on="schedule_id", how="left")
    df = df[df["cf_end_dt"] > report_date].copy()

    rows = []
    for _, grp in df.groupby("schedule_id"):
        currency  = grp["currency"].iloc[0]
        bs_side   = grp["bs_side"].iloc[0]   # comes from cf_df
        leg_type  = grp["leg_type"].iloc[0]
        grp       = grp.sort_values("cf_end_dt").reset_index(drop=True)

        if leg_type == "FIXED":
            # Interest locked at fixed rate → each coupon reprices at its payment date.
            # Notional reprices at maturity (last row) via outstanding_bal — no capital
            # exchange in the CF table but the full notional is at risk until swap end.
            for i, (_, r) in enumerate(grp.iterrows()):
                gap = float(r["int_pmt"])
                if i == len(grp) - 1:          # maturity: add repricing notional
                    gap += float(r["outstanding_bal"])
                if gap != 0:
                    rows.append({"currency": currency, "bs_side": bs_side,
                                 "cf_end_dt": r["cf_end_dt"], "gap_cf": gap})

        else:  # FLOAT
            future = grp[grp["fixing_dt"] > report_date]

            # First period: locked interest
            first = grp.iloc[0]
            if float(first["int_pmt"]) != 0:
                rows.append({"currency": currency, "bs_side": bs_side,
                             "cf_end_dt": first["cf_end_dt"],
                             "gap_cf": float(first["int_pmt"])})

            if not future.empty:
                next_fix_dt = future["fixing_dt"].min()
                notional_at_fix = (
                    grp.loc[grp["fixing_dt"] == next_fix_dt, "outstanding_bal"].max()
                )
                rows.append({"currency": currency, "bs_side": bs_side,
                             "cf_end_dt": next_fix_dt,
                             "gap_cf": float(notional_at_fix)})

    if not rows:
        return pd.DataFrame(columns=["currency", "bs_side", "tenor_bucket",
                                     "cf_end_dt", "gap_cf"])

    gap_raw = pd.DataFrame(rows)
    gap_raw["cf_end_dt"] = pd.to_datetime(gap_raw["cf_end_dt"])
    return _aggregate_ir_gap(gap_raw, report_date)
