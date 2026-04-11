"""EBA IRRBB Standardized SOT Objects
======================================
Per EBA/GL/2018/02 Annex III and EBA/RTS/2022/10.

Key feature: 50% supervisory haircut applied at TIME-BUCKET level (not total),
to positive bucket contributions only.

EVE SOT per bucket k:
    net_CF_k    = Σ (capital + prepayment + interest) × sign  [for CFs in bucket k]
    ΔPV_k       = net_CF_k × (d_f_shocked_k − d_f_base_k)
    ΔPV_k_reg   = ΔPV_k           if ΔPV_k ≤ 0   (loss — full weight)
                = 0.5 × ΔPV_k     if ΔPV_k > 0   (gain — 50% haircut)
    ΔEVE_reg    = Σ_k ΔPV_k_reg

NII SOT per bucket k (within 1Y horizon):
    Δr_k        = shocked spot rate − base spot rate at bucket representative point
    gap_adj_k   = gap_cf_k          if Δr_k ≥ 0
                = gap_cf_k − gap_cf_ca_k  if Δr_k < 0  (CA floored at 0% in down)
    ΔNII_k      = gap_adj_k × Δr_k × remain_yf_k
    ΔNII_k_reg  = ΔNII_k           if ΔNII_k ≤ 0
                = 0.5 × ΔNII_k     if ΔNII_k > 0
    ΔNII_reg    = Σ_k ΔNII_k_reg
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── EBA IRRBB Time Buckets (19 buckets, EBA/GL/2018/02 Annex III) ─────────────
# (label, lower_yf_exclusive, upper_yf_inclusive, midpoint_yf)
EBA_BUCKETS = [
    ("Overnight",  0.0,       1 / 365,        0.5 / 365          ),
    ("1M",         1 / 365,   1 / 12,          (1 / 365 + 1 / 12) / 2),
    ("3M",         1 / 12,    3 / 12,          2 / 12             ),
    ("6M",         3 / 12,    6 / 12,          4.5 / 12           ),
    ("9M",         6 / 12,    9 / 12,          7.5 / 12           ),
    ("1Y",         9 / 12,    1.0,             10.5 / 12          ),
    ("1.5Y",       1.0,       1.5,             1.25               ),
    ("2Y",         1.5,       2.0,             1.75               ),
    ("3Y",         2.0,       3.0,             2.5                ),
    ("4Y",         3.0,       4.0,             3.5                ),
    ("5Y",         4.0,       5.0,             4.5                ),
    ("6Y",         5.0,       6.0,             5.5                ),
    ("7Y",         6.0,       7.0,             6.5                ),
    ("8Y",         7.0,       8.0,             7.5                ),
    ("9Y",         8.0,       9.0,             8.5                ),
    ("10Y",        9.0,       10.0,            9.5                ),
    ("15Y",        10.0,      15.0,            12.5               ),
    ("20Y",        15.0,      20.0,            17.5               ),
    (">20Y",       20.0,      float("inf"),    25.0               ),
]

_LABELS       = [b[0] for b in EBA_BUCKETS]
_MIDS_YF      = {b[0]: b[3] for b in EBA_BUCKETS}
_BUCKET_ORDER = {lbl: i for i, lbl in enumerate(_LABELS)}

# NII SOT uses only the first 6 buckets (within 1-year horizon)
NII_BUCKET_LABELS = [b[0] for b in EBA_BUCKETS if b[2] <= 1.0]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _assign_eba_bucket(yf_arr: np.ndarray) -> np.ndarray:
    """Vectorised assignment of year-fraction values to EBA bucket labels."""
    result = np.full(len(yf_arr), ">20Y", dtype=object)
    for lbl, lo, hi, _mid in EBA_BUCKETS:
        if hi == float("inf"):
            mask = yf_arr > lo
        else:
            mask = (yf_arr > lo) & (yf_arr <= hi)
        result[mask] = lbl
    result[yf_arr <= 0] = "Overnight"
    return result


def _interp_base_df(
    mkt_df: pd.DataFrame,
    curve_name: str,
    target_days: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate base discount factors from mkt_df at given n_days."""
    c = mkt_df[mkt_df["curve_name"] == curve_name].sort_values("n_days")
    return np.interp(
        target_days,
        c["n_days"].to_numpy(dtype=float),
        c["d_f"].to_numpy(dtype=float),
    )


def _interp_shocked_df(
    shocked_disc_df: pd.Series,       # MultiIndex (curve_name, node_date) → d_f
    curve_name: str,
    report_date: pd.Timestamp,
    target_days: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate shocked discount factors at given n_days from report_date."""
    try:
        s = shocked_disc_df.xs(curve_name, level="curve_name")
    except KeyError:
        return np.full(len(target_days), np.nan)

    dates  = pd.DatetimeIndex(s.index)
    n_days = (dates - report_date).days.to_numpy(dtype=float)
    dfs    = s.to_numpy(dtype=float)
    idx    = np.argsort(n_days)
    return np.interp(target_days, n_days[idx], dfs[idx])


# ── EVE SOT ────────────────────────────────────────────────────────────────────

def compute_eve_sot(
    beh_df: pd.DataFrame,
    mkt_df: pd.DataFrame,
    shocked_disc_df: pd.Series,
    disc_curve_map: dict[str, str],
    report_date: pd.Timestamp,
    scenario_id: str,
    sot_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EBA standardised EVE SOT with per-bucket 50% supervisory haircut.

    Parameters
    ----------
    beh_df          : run-off CF schedules (all maturities), same layout as
                      sql_setup.load_all_beh_schedules()
    mkt_df          : base market curves (curve_name, n_days, d_f)
    shocked_disc_df : shocked daily discount curve Series indexed by
                      MultiIndex(curve_name, node_date)
    disc_curve_map  : currency → curve_name
    report_date     : valuation date
    scenario_id     : EBA scenario label (e.g. 'par_up')
    sot_weight      : supervisory haircut for positive contributions (default 0.5)

    Returns
    -------
    detail_df  : per (scenario, currency, EBA bucket)
    summary_df : per (scenario, currency) — ΔEVE, ΔEVE_reg
    """
    df = beh_df.copy()
    df["sign"]     = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["total_cf"] = (
        df["capital_pmt"].fillna(0.0)
        + df["prepayment_pmt"].fillna(0.0)
        + df["int_pmt"].fillna(0.0)
    ) * df["sign"]

    df["cf_yf_abs"] = (df["cf_end_dt"] - report_date).dt.days / 365.25
    df["eba_bucket"] = _assign_eba_bucket(df["cf_yf_abs"].to_numpy())

    net = (
        df.groupby(["currency", "eba_bucket"])["total_cf"]
        .sum()
        .reset_index()
    )
    net["bucket_order"] = net["eba_bucket"].map(_BUCKET_ORDER)
    net = net.sort_values(["currency", "bucket_order"]).reset_index(drop=True)

    # Midpoint days for all 19 buckets (used for d_f interpolation)
    mid_days = np.array([_MIDS_YF[lbl] * 365.25 for lbl in _LABELS])

    rows = []
    for ccy in net["currency"].unique():
        cn = disc_curve_map.get(ccy)
        if cn is None:
            continue

        base_dfs_mid    = _interp_base_df(mkt_df, cn, mid_days)
        shocked_dfs_mid = _interp_shocked_df(shocked_disc_df, cn, report_date, mid_days)
        base_lkp    = dict(zip(_LABELS, base_dfs_mid))
        shocked_lkp = dict(zip(_LABELS, shocked_dfs_mid))

        for _, row in net[net["currency"] == ccy].iterrows():
            lbl    = row["eba_bucket"]
            cf     = float(row["total_cf"])
            d_f_b  = base_lkp.get(lbl, np.nan)
            d_f_s  = shocked_lkp.get(lbl, np.nan)

            if np.isnan(d_f_b) or np.isnan(d_f_s):
                delta_df = delta_pv = delta_pv_reg = np.nan
            else:
                delta_df     = d_f_s - d_f_b
                delta_pv     = cf * delta_df
                delta_pv_reg = delta_pv if delta_pv <= 0 else sot_weight * delta_pv

            rows.append({
                "scenario_id":  scenario_id,
                "currency":     ccy,
                "eba_bucket":   lbl,
                "bucket_order": _BUCKET_ORDER[lbl],
                "net_cf":       cf,
                "d_f_base":     d_f_b,
                "d_f_shocked":  d_f_s,
                "delta_df":     delta_df,
                "delta_pv":     delta_pv,
                "delta_pv_reg": delta_pv_reg,
            })

    detail_df = (
        pd.DataFrame(rows)
        .sort_values(["scenario_id", "currency", "bucket_order"])
        .reset_index(drop=True)
    )
    summary_df = (
        detail_df
        .groupby(["scenario_id", "currency"])[["delta_pv", "delta_pv_reg"]]
        .sum()
        .reset_index()
    )
    return detail_df, summary_df


# ── NII SOT ────────────────────────────────────────────────────────────────────

def compute_nii_sot(
    ir_gap_beh: pd.DataFrame,
    mkt_df: pd.DataFrame,
    shocked_disc_df: pd.Series,
    disc_curve_map: dict[str, str],
    report_date: pd.Timestamp,
    scenario_id: str,
    horizon_yf: float = 1.0,
    sot_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EBA standardised NII SOT with per-bucket 50% supervisory haircut.

    Rate change at each tenor is derived from shocked vs base spot rates
    computed from the shocked discount curve (consistent with actual curve shapes
    for all 6 EBA scenarios — parallel, steep, flat, short-rate).

    Down-shock CA treatment: gap_cf_ca excluded when Δr < 0 at that tenor
    (same logic as compute_nii in nii_calc_objects.py).

    Parameters
    ----------
    ir_gap_beh      : repricing gap table (from sql_setup.load_ir_gap_beh())
    mkt_df          : base market curves (curve_name, n_days, d_f)
    shocked_disc_df : shocked daily discount curve Series
    disc_curve_map  : currency → curve_name
    report_date     : valuation date
    scenario_id     : EBA scenario label
    horizon_yf      : NII horizon in year fractions (default 1.0)
    sot_weight      : 0.5

    Returns
    -------
    detail_df  : per (scenario, currency, tenor / EBA bucket) with Δr, ΔNII, ΔNII_reg
    summary_df : per (scenario, currency) — ΔNII, ΔNII_reg
    """
    from nii_calc_objects import _tenor_to_yf

    df = ir_gap_beh.copy()
    for col in ("gap_cf_ca", "gap_cf_irs"):
        if col not in df.columns:
            df[col] = 0.0

    net = (
        df.groupby(["currency", "tenor_bucket"], sort=False)
        [["gap_cf", "gap_cf_ca", "gap_cf_irs"]]
        .sum()
        .reset_index()
    )
    net["tenor_yf"] = net["tenor_bucket"].map(_tenor_to_yf)
    net = net[net["tenor_yf"] <= horizon_yf].copy()
    net["remain_yf"]   = (horizon_yf - net["tenor_yf"]).clip(lower=0.0)
    net["eba_bucket"]   = _assign_eba_bucket(net["tenor_yf"].to_numpy())
    net["bucket_order"] = net["eba_bucket"].map(_BUCKET_ORDER)

    rows = []
    for ccy in net["currency"].unique():
        cn = disc_curve_map.get(ccy)
        if cn is None:
            continue

        base_curve = mkt_df[mkt_df["curve_name"] == cn].sort_values("n_days")
        base_days  = base_curve["n_days"].to_numpy(dtype=float)
        base_dfs   = base_curve["d_f"].to_numpy(dtype=float)

        try:
            s        = shocked_disc_df.xs(cn, level="curve_name")
            s_dates  = pd.DatetimeIndex(s.index)
            s_days   = (s_dates - report_date).days.to_numpy(dtype=float)
            s_dfs    = s.to_numpy(dtype=float)
            idx      = np.argsort(s_days)
            s_days   = s_days[idx]
            s_dfs    = s_dfs[idx]
            has_shocked = True
        except KeyError:
            has_shocked = False

        for _, row in net[net["currency"] == ccy].iterrows():
            tenor_yf  = float(row["tenor_yf"])
            remain_yf = float(row["remain_yf"])
            gap_bs    = float(row["gap_cf"])
            gap_irs   = float(row.get("gap_cf_irs", 0.0))
            gap       = gap_bs + gap_irs          # total gap incl. IRS hedges
            gap_no_ca = gap - float(row["gap_cf_ca"])
            t_days    = tenor_yf * 365.25

            if t_days > 0 and has_shocked:
                d_f_b   = float(np.interp(t_days, base_days, base_dfs))
                d_f_s   = float(np.interp(t_days, s_days, s_dfs))
                # Spot rate: r(t) = (1/d_f − 1) / t  [simple annualised]
                r_b     = (1.0 / d_f_b - 1.0) / tenor_yf if d_f_b > 0 else 0.0
                r_s     = (1.0 / d_f_s - 1.0) / tenor_yf if d_f_s > 0 else 0.0
                delta_r = r_s - r_b
            else:
                d_f_b = d_f_s = r_b = r_s = delta_r = 0.0

            # CA exclusion in down-rate buckets
            gap_adj       = gap_no_ca if delta_r < 0 else gap
            delta_nii     = gap_adj * delta_r * remain_yf
            delta_nii_reg = delta_nii if delta_nii <= 0 else sot_weight * delta_nii

            rows.append({
                "scenario_id":   scenario_id,
                "currency":      ccy,
                "tenor_bucket":  row["tenor_bucket"],
                "eba_bucket":    row["eba_bucket"],
                "bucket_order":  int(row["bucket_order"]),
                "tenor_yf":      tenor_yf,
                "remain_yf":     remain_yf,
                "gap_cf":        gap_bs,
                "gap_cf_irs":    gap_irs,
                "gap_cf_total":  gap,
                "gap_cf_ca":     float(row["gap_cf_ca"]),
                "gap_adj":       gap_adj,
                "r_base":        r_b,
                "r_shocked":     r_s,
                "delta_r":       delta_r,
                "delta_nii":     delta_nii,
                "delta_nii_reg": delta_nii_reg,
            })

    detail_df = (
        pd.DataFrame(rows)
        .sort_values(["scenario_id", "currency", "bucket_order", "tenor_yf"])
        .reset_index(drop=True)
    )
    summary_df = (
        detail_df
        .groupby(["scenario_id", "currency"])[["delta_nii", "delta_nii_reg"]]
        .sum()
        .reset_index()
    )
    return detail_df, summary_df
