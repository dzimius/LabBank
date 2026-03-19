from __future__ import annotations

import numpy as np
import pandas as pd

_TENOR_SORT = {"1D": 0, ">30Y": 999}


def _tenor_to_yf(tenor_bucket: str) -> float:
    """Convert tenor bucket label to year fraction."""
    if tenor_bucket == "1D":
        return 1 / 365
    if tenor_bucket.endswith("M"):
        return int(tenor_bucket[:-1]) / 12
    return 999.0  # >30Y — beyond any horizon


def _tenor_sort_key(tenor_bucket: str) -> int:
    if tenor_bucket in _TENOR_SORT:
        return _TENOR_SORT[tenor_bucket]
    return int(tenor_bucket[:-1])  # strip trailing "M"


def compute_nii(
    ir_gap_beh: pd.DataFrame,
    horizon_yf: float = 1.0,
    shock_up: float = 0.01,
    shock_dn: float = -0.01,
    ca_gap: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute NII sensitivity from ir_gap_beh repricing gap.

    For each tenor bucket within the horizon:
        remain_yf   = horizon_yf - tenor_yf       (time left to earn/pay interest)
        NII_up      = net_gap_cf * shock_up * remain_yf
        NII_dn      = (net_gap_cf - net_ca_gap_cf) * shock_dn * remain_yf
        delta_NII   = NII_scenario - NII_base  (NII_base = 0, shocks are incremental)

    Current accounts (ca_gap) are floored at 0% rate — the -100bps shock cannot push
    their rate below 0, so they are excluded from the down shock only.

    Parameters
    ----------
    ir_gap_beh  : DataFrame with columns currency, bs_side, tenor_bucket, cf_end_dt, gap_cf
    horizon_yf  : NII horizon in year fractions (default 1.0 = 1 year)
    shock_up    : rate shock up in decimal (default +100bps = 0.01)
    shock_dn    : rate shock down in decimal (default -100bps = -0.01)
    ca_gap      : optional DataFrame (currency, cf_end_dt, ca_gap_cf) for current account gap;
                  when provided, ca_gap_cf is excluded from the down shock (floor = 0%)

    Returns
    -------
    detail_df   : per (currency, tenor_bucket) breakdown
    summary_df  : total NII per currency
    """
    df = ir_gap_beh.copy()

    # Attach current account gap (floored) per row, matched by (currency, cf_end_dt)
    if ca_gap is not None:
        df = df.merge(
            ca_gap[["currency", "cf_end_dt", "ca_gap_cf"]],
            on=["currency", "cf_end_dt"],
            how="left",
        )
        df["ca_gap_cf"] = df["ca_gap_cf"].fillna(0.0)
    else:
        df["ca_gap_cf"] = 0.0

    # Net gap per (currency, tenor_bucket) — sign already applied in ir_gap_beh
    net_gap = (
        df.groupby(["currency", "tenor_bucket"], sort=False)[["gap_cf", "ca_gap_cf"]]
        .sum()
        .reset_index()
        .rename(columns={"gap_cf": "net_gap_cf", "ca_gap_cf": "net_ca_gap_cf"})
    )

    # Tenor year fraction and filter to horizon
    net_gap["tenor_yf"] = net_gap["tenor_bucket"].map(_tenor_to_yf)
    net_gap = net_gap[net_gap["tenor_yf"] <= horizon_yf].copy()

    # Remaining year fraction within horizon
    net_gap["remain_yf"] = (horizon_yf - net_gap["tenor_yf"]).clip(lower=0.0)

    # NII sensitivity — down shock excludes CA (floored at 0%)
    net_gap["nii_base"]        = 0.0
    net_gap["nii_up_100bps"]   = net_gap["net_gap_cf"] * shock_up * net_gap["remain_yf"]
    net_gap["nii_dn_100bps"]   = (
        (net_gap["net_gap_cf"] - net_gap["net_ca_gap_cf"]) * shock_dn * net_gap["remain_yf"]
    )
    net_gap["delta_nii_up"]    = net_gap["nii_up_100bps"] - net_gap["nii_base"]
    net_gap["delta_nii_dn"]    = net_gap["nii_dn_100bps"] - net_gap["nii_base"]

    # Sort by currency then tenor
    net_gap["_sort"] = net_gap["tenor_bucket"].map(_tenor_sort_key)
    detail_df = (
        net_gap
        .sort_values(["currency", "_sort"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    detail_df = detail_df[[
        "currency", "tenor_bucket", "tenor_yf", "remain_yf",
        "net_gap_cf", "net_ca_gap_cf", "nii_base", "nii_up_100bps", "nii_dn_100bps",
        "delta_nii_up", "delta_nii_dn",
    ]]

    # Summary totals per currency
    summary_df = (
        detail_df
        .groupby("currency")[["net_gap_cf", "net_ca_gap_cf", "nii_base",
                               "nii_up_100bps", "nii_dn_100bps",
                               "delta_nii_up", "delta_nii_dn"]]
        .sum()
        .reset_index()
    )

    return detail_df, summary_df


def compute_nii_base(
    beh_df: pd.DataFrame,
    report_date: pd.Timestamp,
    horizon_yf: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute base NII from behavioral CF schedules (constant balance assumption, no rate shock).

    Two NII components per cash flow within the 1Y horizon:

    1. nii_interest  — interest from the existing schedule:
           int_pmt × sign
       This is the actual contractual/behavioural interest payment.

    2. nii_renewal   — interest on returned capital (constant balance):
           (capital_pmt + prepayment_pmt) × fwd_rt × remain_yf × sign
       When capital is repaid/returned within the horizon it is assumed immediately
       reinvested (assets) or re-borrowed (liabilities) at the current forward rate
       for the remaining time to horizon end.
           remain_yf = (horizon_end − cf_end_dt).days / 365

    Sign convention: A (asset) = +1 (income), L (liability) = −1 (expense).

    Parameters
    ----------
    beh_df      : unified behavioral CF DataFrame from sql_setup.load_beh_schedules()
    report_date : valuation date
    horizon_yf  : NII horizon in year fractions (default 1.0 = 1 year)

    Returns
    -------
    detail_df   : NII breakdown per (currency, source) — loans / deposits / fin_inst
    summary_df  : total NII per currency
    """
    horizon_end = report_date + pd.Timedelta(days=round(horizon_yf * 365.25))

    df = beh_df.copy()

    # Sign: asset = income (+1), liability = expense (-1)
    df["sign"] = np.where(df["bs_side"] == "A", 1.0, -1.0)

    # Component 1: interest from existing cash flow schedule
    df["nii_interest"] = df["int_pmt"].fillna(0.0) * df["sign"]

    # Component 2: renewal interest on returned capital (constant balance)
    df["remain_yf"] = (
        (horizon_end - df["cf_end_dt"]).dt.days / 365.0
    ).clip(lower=0.0)
    df["total_capital"] = df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)
    df["nii_renewal"] = (
        df["total_capital"] * df["fwd_rt"].fillna(0.0) * df["remain_yf"] * df["sign"]
    )

    df["nii_total"] = df["nii_interest"] + df["nii_renewal"]

    # Detail: per (currency, source)
    detail_df = (
        df.groupby(["currency", "source"])[["nii_interest", "nii_renewal", "nii_total"]]
        .sum()
        .reset_index()
        .sort_values(["currency", "source"])
        .reset_index(drop=True)
    )

    # Summary: per currency
    summary_df = (
        detail_df
        .groupby("currency")[["nii_interest", "nii_renewal", "nii_total"]]
        .sum()
        .reset_index()
    )

    return detail_df, summary_df
