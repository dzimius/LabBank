from __future__ import annotations

import numpy as np
import pandas as pd

# Reuse the per-product rate-limit helper from NII
from nii_calc_objects import _apply_rt_limits

_SCHED_GROUP_COLS = ["schedule_id", "product_type", "product_code",
                     "currency", "bs_side", "rate_type"]


def compute_eve_base(
    beh_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute base EVE from run-off behavioral CF schedules.

    No renewal assumption — the balance sheet is in run-off (existing book only).
    All future CFs are discounted at current market discount factors (base_df).

    EVE components per cash flow
    ----------------------------
    pv_capital  = (capital_pmt + prepayment_pmt) × base_df × sign
    pv_interest = int_pmt                        × base_df × sign
    pv_total    = pv_capital + pv_interest

    Sign convention: A (asset) = +1,  L (liability) = −1.

    Parameters
    ----------
    beh_df : run-off CF DataFrame from sql_setup.load_all_beh_schedules()
             Must include base_df column (discount factor at cf_end_dt).

    Returns
    -------
    detail_df  : EVE breakdown per (currency, source)
    summary_df : total EVE per currency
    """
    df = beh_df.copy()
    df["sign"]         = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["base_df_val"]  = df["base_df"].fillna(0.0)
    df["total_capital"]= df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)
    df["pv_capital"]   = df["total_capital"] * df["base_df_val"] * df["sign"]
    df["pv_interest"]  = df["int_pmt"].fillna(0.0) * df["base_df_val"] * df["sign"]
    df["pv_total"]     = df["pv_capital"] + df["pv_interest"]

    detail_df = (
        df.groupby(["currency", "source"])[["pv_capital", "pv_interest", "pv_total"]]
        .sum()
        .reset_index()
        .sort_values(["currency", "source"])
        .reset_index(drop=True)
    )
    summary_df = (
        detail_df.groupby("currency")[["pv_capital", "pv_interest", "pv_total"]]
        .sum()
        .reset_index()
    )
    return detail_df, summary_df


def compute_eve_shocked(
    beh_df: pd.DataFrame,
    shocked_disc_df: pd.Series,          # MultiIndex (curve_name, node_date) → d_f
    disc_curve_map: dict[str, str],      # currency → disc_curve_name
    scenario_id: str,
    eve_floor_rate: float = 0.0,
    caps_map:   dict | None = None,
    floors_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute EVE under a shocked rate scenario (run-off, no renewal).

    For each CF in beh_df (all maturities, no horizon cutoff):

    Discount factor
    ---------------
    Shocked d_f is looked up at cf_end_dt from the pre-loaded daily shocked
    curve grid, the same grid used for NII forward-rate derivation.

    Interest cash flows under shock
    --------------------------------
    - F (fixed)        : int_pmt unchanged — contracted rate locked.
    - V (variable)     : int_pmt_shocked = outstanding_bal × eff_rate_shocked × cf_yf
                         where eff_rate_shocked = max(floor, min(cap, fwd_rt_shocked))
                         fwd_rt_shocked derived from shocked disc curve:
                         (d_f_shocked(cf_start) / d_f_shocked(cf_end) − 1) / cf_yf
    - A (administrative): int_pmt treated as 0% (bank-managed, run-off).

    PV components
    -------------
    pv_capital  = (capital_pmt + prepayment_pmt) × d_f_shocked(cf_end) × sign
    pv_interest = int_pmt_shocked               × d_f_shocked(cf_end) × sign
    pv_total    = pv_capital + pv_interest

    Parameters
    ----------
    beh_df          : run-off CF DataFrame (all maturities)
    shocked_disc_df : daily shocked discount curve Series indexed by
                      (curve_name, node_date)
    disc_curve_map  : currency → curve_name mapping
    scenario_id     : e.g. 'par_up', 'steep', ...
    eve_floor_rate  : 0 % floor on forward rates (EBA standard)
    caps_map        : per-product client rate caps (decimal)
    floors_map      : per-product client rate floors (decimal)

    Returns
    -------
    detail_df  : per (currency, source, rate_type) breakdown + scenario_id
    summary_df : total EVE per currency + scenario_id
    """
    df = beh_df.copy()

    # ── Shocked discount factors at CF start and end dates ────────────────────
    def _lookup_df(dates: pd.Series, curve_name: str) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays(
            [[curve_name] * len(dates), dates.to_numpy()],
            names=["curve_name", "node_date"],
        )
        return shocked_disc_df.reindex(idx).to_numpy(dtype=float)

    d_f_shocked_end   = np.full(len(df), np.nan)
    d_f_shocked_start = np.full(len(df), np.nan)

    for ccy, grp in df.groupby("currency"):
        cn = disc_curve_map.get(ccy)
        if cn is None:
            continue
        idx = grp.index
        d_f_shocked_end[idx]   = _lookup_df(df.loc[idx, "cf_end_dt"],   cn)
        d_f_shocked_start[idx] = _lookup_df(df.loc[idx, "cf_start_dt"], cn)

    df["d_f_shocked"] = np.nan_to_num(d_f_shocked_end, nan=0.0)

    # ── Shocked forward rates (for variable products) ─────────────────────────
    yf = df["cf_yf"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_rt_raw = np.where(
            (d_f_shocked_end > 0) & (yf > 0) & ~np.isnan(d_f_shocked_end),
            (d_f_shocked_start / d_f_shocked_end - 1.0) / yf,
            np.nan,
        )
    df["fwd_rt_shocked"] = np.maximum(eve_floor_rate, np.nan_to_num(fwd_rt_raw, nan=0.0))

    # ── Effective rate per CF (same logic as NII) ─────────────────────────────
    rate_type_s = df.get("rate_type", pd.Series("V", index=df.index))
    df["eff_rate_shocked"] = df["eff_rate"].fillna(0.0)          # F: contracted
    mask_var   = rate_type_s == "V"
    mask_admin = rate_type_s == "A"

    if mask_var.any() and "product_code" in df.columns:
        df.loc[mask_var, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_var, "fwd_rt_shocked"],
            df.loc[mask_var, "product_code"],
            caps_map, floors_map,
        )
    elif mask_var.any():
        df.loc[mask_var, "eff_rate_shocked"] = df.loc[mask_var, "fwd_rt_shocked"]
    df.loc[mask_admin, "eff_rate_shocked"] = 0.0

    # ── Interest cash flows under shocked scenario ────────────────────────────
    # F: int_pmt unchanged (contractual); V: recomputed; A: 0
    int_pmt_shocked = df["int_pmt"].fillna(0.0).to_numpy(dtype=float).copy()
    var_idx = mask_var.to_numpy()
    int_pmt_shocked[var_idx] = (
        df.loc[mask_var, "outstanding_bal"].fillna(0.0).to_numpy()
        * df.loc[mask_var, "eff_rate_shocked"].to_numpy()
        * df.loc[mask_var, "cf_yf"].fillna(0.0).to_numpy()
    )
    int_pmt_shocked[mask_admin.to_numpy()] = 0.0
    df["int_pmt_shocked"] = int_pmt_shocked

    # ── PV components ─────────────────────────────────────────────────────────
    df["sign"]          = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["total_capital"] = df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)
    df["pv_capital"]    = df["total_capital"]   * df["d_f_shocked"] * df["sign"]
    df["pv_interest"]   = df["int_pmt_shocked"] * df["d_f_shocked"] * df["sign"]
    df["pv_total"]      = df["pv_capital"] + df["pv_interest"]
    df["scenario_id"]   = scenario_id

    group_cols = (["currency", "source", "rate_type"] if "rate_type" in df.columns
                  else ["currency", "source"])
    detail_df = (
        df.groupby(group_cols)[["pv_capital", "pv_interest", "pv_total"]]
        .sum()
        .reset_index()
        .assign(scenario_id=scenario_id)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    summary_df = (
        detail_df.groupby("currency")[["pv_capital", "pv_interest", "pv_total"]]
        .sum()
        .reset_index()
        .assign(scenario_id=scenario_id)
    )
    return detail_df, summary_df


def compute_eve_base_schedule(
    beh_df: pd.DataFrame,
    scenario_id: str = "base",
) -> pd.DataFrame:
    """Per-schedule base EVE for SQL storage.

    Groups at (_SCHED_GROUP_COLS) and sums PV components.

    Returns
    -------
    DataFrame with: scenario_id + _SCHED_GROUP_COLS + pv_capital / pv_interest / pv_total
    """
    df = beh_df.copy()
    df["sign"]          = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["base_df_val"]   = df["base_df"].fillna(0.0)
    df["total_capital"] = df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)
    df["pv_capital"]    = df["total_capital"] * df["base_df_val"] * df["sign"]
    df["pv_interest"]   = df["int_pmt"].fillna(0.0) * df["base_df_val"] * df["sign"]
    df["pv_total"]      = df["pv_capital"] + df["pv_interest"]

    group_cols = [c for c in _SCHED_GROUP_COLS if c in df.columns]
    return (
        df.groupby(group_cols)[["pv_capital", "pv_interest", "pv_total"]]
        .sum()
        .reset_index()
        .assign(scenario_id=scenario_id)
    )


def compute_eve_shocked_schedule(
    beh_df: pd.DataFrame,
    shocked_disc_df: pd.Series,
    disc_curve_map: dict[str, str],
    scenario_id: str,
    eve_floor_rate: float = 0.0,
    caps_map:    dict | None = None,
    floors_map:  dict | None = None,
    coeff_a_map: dict | None = None,
    coeff_b_map: dict | None = None,
) -> pd.DataFrame:
    """Per-schedule shocked EVE for SQL storage.

    Same shock logic as compute_eve_shocked() but grouped at schedule level.

    Returns
    -------
    DataFrame with: scenario_id + _SCHED_GROUP_COLS + pv_capital / pv_interest / pv_total
    """
    df = beh_df.copy()

    def _lookup_df(dates: pd.Series, curve_name: str) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays(
            [[curve_name] * len(dates), dates.to_numpy()],
            names=["curve_name", "node_date"],
        )
        return shocked_disc_df.reindex(idx).to_numpy(dtype=float)

    d_f_shocked_end   = np.full(len(df), np.nan)
    d_f_shocked_start = np.full(len(df), np.nan)

    for ccy, grp in df.groupby("currency"):
        cn = disc_curve_map.get(ccy)
        if cn is None:
            continue
        idx = grp.index
        d_f_shocked_end[idx]   = _lookup_df(df.loc[idx, "cf_end_dt"],   cn)
        d_f_shocked_start[idx] = _lookup_df(df.loc[idx, "cf_start_dt"], cn)

    df["d_f_shocked"] = np.nan_to_num(d_f_shocked_end, nan=0.0)

    yf = df["cf_yf"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_rt_raw = np.where(
            (d_f_shocked_end > 0) & (yf > 0) & ~np.isnan(d_f_shocked_end),
            (d_f_shocked_start / d_f_shocked_end - 1.0) / yf,
            np.nan,
        )
    df["fwd_rt_shocked"] = np.maximum(eve_floor_rate, np.nan_to_num(fwd_rt_raw, nan=0.0))

    rate_type_s = df.get("rate_type", pd.Series("V", index=df.index))
    df["eff_rate_shocked"] = df["eff_rate"].fillna(0.0)
    mask_var   = rate_type_s == "V"
    mask_admin = rate_type_s == "A"

    if mask_var.any() and "product_code" in df.columns:
        df.loc[mask_var, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_var, "fwd_rt_shocked"],
            df.loc[mask_var, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
        )
    elif mask_var.any():
        df.loc[mask_var, "eff_rate_shocked"] = df.loc[mask_var, "fwd_rt_shocked"]
    df.loc[mask_admin, "eff_rate_shocked"] = 0.0

    int_pmt_shocked = df["int_pmt"].fillna(0.0).to_numpy(dtype=float).copy()
    var_idx = mask_var.to_numpy()
    int_pmt_shocked[var_idx] = (
        df.loc[mask_var, "outstanding_bal"].fillna(0.0).to_numpy()
        * df.loc[mask_var, "eff_rate_shocked"].to_numpy()
        * df.loc[mask_var, "cf_yf"].fillna(0.0).to_numpy()
    )
    int_pmt_shocked[mask_admin.to_numpy()] = 0.0
    df["int_pmt_shocked"] = int_pmt_shocked

    df["sign"]          = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["total_capital"] = df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)
    df["pv_capital"]    = df["total_capital"]   * df["d_f_shocked"] * df["sign"]
    df["pv_interest"]   = df["int_pmt_shocked"] * df["d_f_shocked"] * df["sign"]
    df["pv_total"]      = df["pv_capital"] + df["pv_interest"]

    group_cols = [c for c in _SCHED_GROUP_COLS if c in df.columns]
    return (
        df.groupby(group_cols)[["pv_capital", "pv_interest", "pv_total"]]
        .sum()
        .reset_index()
        .assign(scenario_id=scenario_id)
    )


def compute_shocked_cf_detail(
    beh_df: pd.DataFrame,
    shocked_disc_df: pd.Series,
    disc_curve_map: dict[str, str],
    scenario_id: str,
    eve_floor_rate: float = 0.0,
    caps_map:    dict | None = None,
    floors_map:  dict | None = None,
    coeff_a_map: dict | None = None,
    coeff_b_map: dict | None = None,
) -> pd.DataFrame:
    """Return row-level CF schedule enriched with shocked fwd_rt, d_f, and int_pmt.

    Same shock logic as compute_eve_shocked_schedule() but returns individual
    CF rows (no groupby).  Used to populate cf.products_par_dn and
    cf.products_worst_eve SQL tables so shocked cash flows can be compared
    directly against the base cf.products table.

    Columns added
    -------------
    fwd_rt_shocked      shocked forward rate for the period (floor-applied)
    d_f_shocked         shocked discount factor at cf_end_dt
    d_f_shocked_start   shocked discount factor at cf_start_dt
    int_pmt_shocked     F → contractual (unchanged), V → recomputed, A → 0
    scenario_id         scenario label
    """
    df = beh_df.copy()

    def _lookup_df(dates: pd.Series, curve_name: str) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays(
            [[curve_name] * len(dates), dates.to_numpy()],
            names=["curve_name", "node_date"],
        )
        return shocked_disc_df.reindex(idx).to_numpy(dtype=float)

    d_f_shocked_end   = np.full(len(df), np.nan)
    d_f_shocked_start = np.full(len(df), np.nan)

    for ccy, grp in df.groupby("currency"):
        cn = disc_curve_map.get(ccy)
        if cn is None:
            continue
        idx = grp.index
        d_f_shocked_end[idx]   = _lookup_df(df.loc[idx, "cf_end_dt"],   cn)
        d_f_shocked_start[idx] = _lookup_df(df.loc[idx, "cf_start_dt"], cn)

    df["d_f_shocked"]       = np.nan_to_num(d_f_shocked_end,   nan=0.0)
    df["d_f_shocked_start"] = np.nan_to_num(d_f_shocked_start, nan=0.0)

    yf = df["cf_yf"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_rt_raw = np.where(
            (d_f_shocked_end > 0) & (yf > 0) & ~np.isnan(d_f_shocked_end),
            (d_f_shocked_start / d_f_shocked_end - 1.0) / yf,
            np.nan,
        )
    df["fwd_rt_shocked"] = np.maximum(eve_floor_rate, np.nan_to_num(fwd_rt_raw, nan=0.0))

    rate_type_s = df.get("rate_type", pd.Series("V", index=df.index))
    df["eff_rate_shocked"] = df["eff_rate"].fillna(0.0)
    mask_var   = rate_type_s == "V"
    mask_admin = rate_type_s == "A"

    if mask_var.any() and "product_code" in df.columns:
        df.loc[mask_var, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_var, "fwd_rt_shocked"],
            df.loc[mask_var, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
        )
    elif mask_var.any():
        df.loc[mask_var, "eff_rate_shocked"] = df.loc[mask_var, "fwd_rt_shocked"]
    df.loc[mask_admin, "eff_rate_shocked"] = 0.0

    int_pmt_shocked = df["int_pmt"].fillna(0.0).to_numpy(dtype=float).copy()
    var_idx = mask_var.to_numpy()
    int_pmt_shocked[var_idx] = (
        df.loc[mask_var, "outstanding_bal"].fillna(0.0).to_numpy()
        * df.loc[mask_var, "eff_rate_shocked"].to_numpy()
        * df.loc[mask_var, "cf_yf"].fillna(0.0).to_numpy()
    )
    int_pmt_shocked[mask_admin.to_numpy()] = 0.0
    df["int_pmt_shocked"] = int_pmt_shocked
    df["scenario_id"]     = scenario_id
    return df
