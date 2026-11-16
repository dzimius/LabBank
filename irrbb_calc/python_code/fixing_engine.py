"""fixing_engine.py
=================
Shared rules for WHEN a cash flow's rate is set and OVER WHICH PERIOD it is projected --
used alike by base NII, shocked NII (irrbb.nii_results and the cf.nii_* tables) and EVE, so
the three paths cannot drift apart.

effective_rate_type  A fixed-rate product whose rate FIXES after the report date (e.g. a
                     mortgage fixed for 10 years, then reset) is floating from that reset on:
                     those cash flows are 'V' -- shocks move them. Only the rate already set
                     (fixing_dt <= report date) is locked. Product-level 'F' used to mean
                     "fixed forever", so post-reset coupons ignored every shock.

block_periods        One index fixing covers a whole fixing block (all coupons sharing a
                     fixing_dt): its forward runs from the block's accrual start (the index
                     value date -- not the fixing observation date two days earlier) to the
                     block end, on the INDEX day basis (EURIBOR ACT/360). Per-coupon forwards
                     gave a 3M-fixing monthly loan three different shocked rates per fixing.
                     The index end is rolled Modified Following on the currency calendar, as the
                     cash-flow engine sets the base forward (cf_calc_objects.get_daily_intrpl_curve).

horizon_accrual      NII over a one-year horizon counts interest EARNED in [report date,
                     horizon end] (Reg. (EU) 2024/856 Art. 4): a coupon period is clipped to
                     the window -- a coupon paid just after the horizon still counts for the
                     part accrued inside it, interest accrued before the report date does not.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import QuantLib as ql

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from labbank_common.curves import index_day_basis  # noqa: E402


def effective_rate_type(df: pd.DataFrame, report_date, nmd_codes=()) -> pd.Series:
    """rate_type with 'F' -> 'V' on cash flows whose rate fixes after the report date, and
    'F' on non-maturity deposits (nmd_codes): their IR profile is a repricing profile, so the
    outstanding balance keeps today's rate and only the repriced part (capital the profile
    releases) renews at beta x market + margin."""
    rt = df.get("rate_type", pd.Series("V", index=df.index)).copy()
    if "fixing_dt" in df.columns:
        future_fix = pd.to_datetime(df["fixing_dt"]).gt(pd.Timestamp(report_date)).fillna(False)
        rt = rt.where(~((rt == "F") & future_fix), "V")
    if len(nmd_codes) and "product_code" in df.columns:
        nmd = df["product_code"].astype(str).isin({str(c) for c in nmd_codes})
        if "product_type" in df.columns:
            nmd = nmd & df["product_type"].astype(str).eq("D")
        rt = rt.where(~nmd, "F")
    return rt


def block_periods(df: pd.DataFrame, mask: pd.Series | np.ndarray | None = None
                  ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(period_start, period_end, year_fraction) of the index forward of every row.

    Rows in ``mask`` with a fixing_dt: their fixing block (schedule_id, fixing_dt) from its
    first accrual start to its last accrual end, year fraction on the index day basis of
    the row's currency. Other rows: their own coupon period and cf_yf."""
    start = pd.to_datetime(df["cf_start_dt"]).copy()
    end = pd.to_datetime(df["cf_end_dt"]).copy()
    yf = df["cf_yf"].astype(float).copy()
    if "fixing_dt" not in df.columns or "schedule_id" not in df.columns:
        return start, end, yf
    m = df["fixing_dt"].notna()
    if mask is not None:
        m = m & pd.Series(np.asarray(mask, dtype=bool), index=df.index)
    if not m.any():
        return start, end, yf
    sub = df.loc[m]
    key = [sub["schedule_id"], sub["fixing_dt"]]
    start.loc[m] = pd.to_datetime(sub["cf_start_dt"]).groupby(key).transform("min")
    end.loc[m] = pd.to_datetime(sub["cf_end_dt"]).groupby(key).transform("max")
    # The index runs over its own tenor (fixing_freq) from the block start -- also where
    # only part of the block was loaded (the NII loader stops at the horizon) or the block
    # is a short final one; the cash-flow engine projects the base rate the same way.
    if "fixing_freq" in sub.columns:
        freq = sub["fixing_freq"].astype(object)
        months = freq.map({f: _tenor_months(f) for f in pd.unique(freq)})
        ok = months.notna()
        if ok.any():
            st = start.loc[m][ok]
            end.loc[ok[ok].index] = modified_following(
                add_months(st, months[ok].to_numpy().astype(int)), sub["currency"][ok].to_numpy())
    basis = sub["currency"].map(index_day_basis).astype(float)
    yf.loc[m] = (end.loc[m] - start.loc[m]).dt.days / basis
    return start, end, yf


def add_months(dates, months: np.ndarray) -> np.ndarray:
    """dates + months, clipped to the month end -- pd.DateOffset(months=n) semantics, vectorised
    (a per-row DateOffset loop was ~80% of the shocked EVE/NII repricing time)."""
    v = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    day = v.astype("datetime64[D]")
    time_of_day = v - day.astype("datetime64[ns]")
    month = day.astype("datetime64[M]") + np.asarray(months).astype("timedelta64[M]")
    first = month.astype("datetime64[D]")
    days_in_month = ((month + 1).astype("datetime64[D]") - first).astype(int)
    day_of_month = (day - day.astype("datetime64[M]").astype("datetime64[D]")).astype(int) + 1
    return (first + (np.minimum(day_of_month, days_in_month) - 1).astype("timedelta64[D]")
            ).astype("datetime64[ns]") + time_of_day


# the cash-flow engine's calendars (cf_calc_objects.get_calendar_from_currency)
_CALENDARS = {"PLN": ql.Poland(), "EUR": ql.TARGET(), "USD": ql.UnitedStates(ql.UnitedStates.NYSE),
              "GBP": ql.UnitedKingdom()}


@lru_cache(maxsize=None)
def _holidays(ccy: str) -> np.ndarray:
    """Weekday holidays of the currency's calendar, 1990-2100 (weekends come from the weekmask)."""
    cal = _CALENDARS.get(ccy, ql.TARGET())
    days = ql.Calendar.holidayList(cal, ql.Date(1, 1, 1990), ql.Date(31, 12, 2100))
    return np.array([np.datetime64(d.ISO()) for d in days], dtype="datetime64[D]")


def modified_following(dates, currencies) -> np.ndarray:
    """Roll dates to a business day of their currency's calendar, Modified Following
    (= QuantLib cal.advance(start, Period(n, Months), ModifiedFollowing) after add_months)."""
    v = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    out = v.copy()
    ccy = np.asarray(currencies).astype(str)
    for c in np.unique(ccy):
        i = ccy == c
        out[i] = np.busday_offset(v[i].astype("datetime64[D]"), 0, roll="modifiedfollowing",
                                  holidays=_holidays(c)).astype("datetime64[ns]")
    return out


def _tenor_months(tenor) -> float:
    """'3M' -> 3, '10Y' -> 120; day / week tenors and blanks -> NaN (block from the rows)."""
    t = str(tenor).strip().upper() if tenor is not None and not pd.isna(tenor) else ""
    if len(t) < 2 or not t[:-1].isdigit():
        return np.nan
    return float(int(t[:-1]) * (12 if t[-1] == "Y" else 1)) if t[-1] in "MY" else np.nan


def horizon_accrual(df: pd.DataFrame, report_date, horizon_end) -> np.ndarray:
    """Share of each coupon period [cf_start_dt, cf_end_dt] that falls inside
    [report_date, horizon_end] (0..1)."""
    s = pd.to_datetime(df["cf_start_dt"]).to_numpy("datetime64[D]")
    e = pd.to_datetime(df["cf_end_dt"]).to_numpy("datetime64[D]")
    lo = np.maximum(s, np.datetime64(pd.Timestamp(report_date).date()))
    hi = np.minimum(e, np.datetime64(pd.Timestamp(horizon_end).date()))
    inside = np.clip((hi - lo).astype(float), 0.0, None)
    total = (e - s).astype(float)
    return np.where(total > 0, inside / np.where(total > 0, total, 1.0), 0.0)


def add_horizon_accrual(df: pd.DataFrame, report_date, horizon_end) -> pd.DataFrame:
    """Loader post-processing: accr_frac column (see horizon_accrual)."""
    df["accr_frac"] = horizon_accrual(df, report_date, horizon_end) if len(df) else 0.0
    return df
