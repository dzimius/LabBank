from __future__ import annotations

import numpy as np
import pandas as pd

# Reuse the per-product rate-limit helper from NII
from nii_calc_objects import _apply_rt_limits
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # sibling fixing_engine
import fixing_engine as fe  # noqa: E402
from labbank_common.curves import forward_rate  # noqa: E402

_SCHED_GROUP_COLS = ["schedule_id", "product_type", "product_code",
                     "currency", "bs_side", "rate_type"]


def _require_discount(d_f_end: np.ndarray, df: pd.DataFrame, scenario: str = "") -> None:
    """Every future cash flow needs a discount factor: a missing curve node used to become
    d_f = 0 and silently erase the cash flow's PV (and its sensitivity)."""
    miss = np.isnan(np.asarray(d_f_end, dtype=float))
    if miss.any():
        bad = df.loc[miss, ["currency", "cf_end_dt"]]
        raise ValueError(
            f"EVE {scenario}: {int(miss.sum())} cash flow(s) without a discount factor "
            f"(currencies {sorted(bad['currency'].unique())}, latest {bad['cf_end_dt'].max()}) "
            "-- curve missing or too short; refusing to value them at 0")


def _lookup_shocked_df(
    dates: pd.Series,
    currency: pd.Series,
    shocked_disc_df: pd.Series,
    disc_curve_map: dict[str, str],
) -> pd.Series:
    """Look up shocked discount factors for each (date, currency) pair.

    Returns a Series indexed like `dates` with d_f values (NaN where unavailable).
    """
    out = pd.Series(np.nan, index=dates.index)
    for ccy in currency.unique():
        cn = disc_curve_map.get(ccy)
        if cn is None:
            continue
        mask = currency == ccy
        mi = pd.MultiIndex.from_arrays(
            [[cn] * int(mask.sum()), dates.loc[mask].to_numpy()],
            names=["curve_name", "node_date"],
        )
        out.loc[mask] = shocked_disc_df.reindex(mi).to_numpy(dtype=float)
    return out


# ── Multi-curve: projection curves for rate moves ─────────────────────────────
# A bank with OIS discounting + IBOR projection curves (labbank_common/curves.py) discounts
# on disc_curve_map but measures how a floating coupon moves on its currency's projection
# curve: shocked fwd = base CF fwd_rt + (shocked - base) projection forward. A zero shock
# then reproduces the base exactly although the coupon came from a tenor curve. Set by the
# workflow (set_projection_curves); None = single curve, forwards from disc_curve_map.
_PROJ_CURVE_MAP: dict | None = None
_BASE_PROJ_DF: pd.Series | None = None


def set_projection_curves(proj_curve_map: dict | None, base_proj_df: pd.Series | None) -> None:
    global _PROJ_CURVE_MAP, _BASE_PROJ_DF
    _PROJ_CURVE_MAP, _BASE_PROJ_DF = proj_curve_map, base_proj_df


def _build_shocked_disc_and_fwd(
    df: pd.DataFrame,
    shocked_disc_df: pd.Series,
    disc_curve_map: dict[str, str],
    eve_floor_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(d_f_end, d_f_start, fwd_rt_shocked, locked_mask): discounting on disc_curve_map;
    forwards there too (single curve) or as a projection-curve move (multi-curve)."""
    d_f_end, d_f_start, fwd, locked = _shocked_disc_and_fwd_on(df, shocked_disc_df, disc_curve_map, eve_floor_rate)
    if _PROJ_CURVE_MAP is None or _BASE_PROJ_DF is None:
        return d_f_end, d_f_start, fwd, locked
    _, _, f_s, lock_s = _shocked_disc_and_fwd_on(df, shocked_disc_df, _PROJ_CURVE_MAP, -1e9)
    _, _, f_b, _      = _shocked_disc_and_fwd_on(df, _BASE_PROJ_DF, _PROJ_CURVE_MAP, -1e9)
    base = df["fwd_rt"].fillna(0.0).to_numpy() if "fwd_rt" in df.columns else f_b
    fwd = np.where(lock_s, f_s, np.maximum(eve_floor_rate, base + (f_s - f_b)))
    return d_f_end, d_f_start, fwd, lock_s


def _shocked_disc_and_fwd_on(
    df: pd.DataFrame,
    shocked_disc_df: pd.Series,
    disc_curve_map: dict[str, str],
    eve_floor_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (d_f_end, d_f_start, fwd_rt_shocked, locked_mask) arrays indexed like df.

    Forward rate logic
    ------------------
    For variable-rate products where fixing_dt is present, the shocked forward
    rate is derived from the **fixing period** boundaries:
      period_start = fixing_dt
      period_end   = max(cf_end_dt) within (schedule_id, fixing_dt) group
      fwd_rt_shocked = (d_f(period_start) / d_f(period_end) − 1) / yf_period

    This ensures a 3M-fixing loan gets a 3M shocked rate on every monthly CF,
    matching the base-scenario behaviour where fwd_rt is constant within a
    fixing period.  When fixing_dt is before the shocked-curve start
    (d_f_shocked(fixing_dt) == 0), the rate is already locked → the caller
    substitutes the base fwd_rt (write_shocked_cf_products start_unavail check).

    All other rows use individual CF boundaries (d_f_start / d_f_end).

    locked_mask
    -----------
    Boolean array (True = rate already fixed at fixing_dt before report_date).
    Callers use this to substitute int_pmt directly for locked CFs, preserving
    the contracted client rate (including origination margin).
    """
    d_f_end_ser   = _lookup_shocked_df(df["cf_end_dt"],   df["currency"],
                                       shocked_disc_df, disc_curve_map)
    d_f_start_ser = _lookup_shocked_df(df["cf_start_dt"], df["currency"],
                                       shocked_disc_df, disc_curve_map)

    d_f_end   = d_f_end_ser.to_numpy()
    d_f_start = d_f_start_ser.to_numpy()

    # Default: individual CF fwd rate
    yf = df["cf_yf"].fillna(0.0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_rt_raw = np.where(
            (d_f_end > 0) & (yf > 0) & ~np.isnan(d_f_end),
            forward_rate(d_f_start, d_f_end, yf),
            np.nan,
        )
    fwd_rt_shocked = np.maximum(eve_floor_rate, np.nan_to_num(fwd_rt_raw, nan=0.0))

    locked_mask_ser = pd.Series(False, index=df.index)

    # ── Lock CFs whose period started before the shock curve ─────────────────
    # _lookup_shocked_df returns NaN (not 0) for dates before the curve start —
    # reindex on a MultiIndex produces NaN for missing keys.
    # When d_f_start is NaN the CF period began before report_date; the rate is
    # already contractually fixed.  Substitute base fwd_rt so that eff_rate_shocked
    # collapses to the contracted client rate (zero delta) and int_pmt_shocked
    # equals int_pmt.  Without this, nan_to_num converts NaN → 0 which produces
    # a wrong fwd_rt_shocked (= 0) for variable products.
    start_before_curve = np.isnan(d_f_start) & ~np.isnan(d_f_end) & (d_f_end > 0)
    if start_before_curve.any() and "fwd_rt" in df.columns:
        fwd_rt_shocked[start_before_curve] = (
            df.loc[start_before_curve, "fwd_rt"].fillna(0.0).to_numpy()
        )
        locked_mask_ser.loc[start_before_curve] = True

    # ── Fixing-period override for variable products with fixing_dt ───────────
    rate_type_s = df.get("rate_type", pd.Series("V", index=df.index))
    if "fixing_dt" in df.columns and "schedule_id" in df.columns:
        mask_var_fix = (rate_type_s == "V") & df["fixing_dt"].notna()
        if mask_var_fix.any():
            sub = df.loc[mask_var_fix]
            # Fixing block (fixing_engine.block_periods): from the block's accrual start
            # (index value date, not the fixing observation date) to its end, index day basis
            period_start_ser, period_end_ser, yf_period_ser = fe.block_periods(sub)

            d_f_pstart = _lookup_shocked_df(period_start_ser, sub["currency"],
                                            shocked_disc_df, disc_curve_map)
            d_f_pend   = _lookup_shocked_df(period_end_ser,   sub["currency"],
                                            shocked_disc_df, disc_curve_map)

            # a rate whose fixing date is before the curve start (<= report date) is set
            d_f_fix = _lookup_shocked_df(pd.to_datetime(sub["fixing_dt"]), sub["currency"],
                                         shocked_disc_df, disc_curve_map)
            valid = (d_f_pstart > 0) & (d_f_pend > 0) & (yf_period_ser > 0) & (d_f_fix > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                fwd_period = np.where(
                    valid,
                    forward_rate(d_f_pstart, d_f_pend,
                                 yf_period_ser.where(yf_period_ser > 0, np.nan)),
                    np.nan,
                )
            fwd_period = np.maximum(eve_floor_rate, np.nan_to_num(fwd_period, nan=0.0))

            # Where fixing_dt is on the shocked curve: use period shocked rate.
            # Where fixing_dt is before the curve start (rate already locked):
            # use base fwd_rt so that int_pmt_shocked == int_pmt (no change).
            base_fwd_locked = df.loc[mask_var_fix, "fwd_rt"].fillna(0.0).to_numpy()
            fwd_rt_shocked_ser = pd.Series(fwd_rt_shocked, index=df.index)
            fwd_rt_shocked_ser.loc[mask_var_fix] = np.where(
                valid.to_numpy(),
                fwd_period,
                base_fwd_locked,
            )
            fwd_rt_shocked = fwd_rt_shocked_ser.to_numpy()

            # Mark locked rows: fixing_dt before the shocked curve (valid == False)
            locked_mask_ser.loc[mask_var_fix] = ~valid.to_numpy()

    return d_f_end, d_f_start, fwd_rt_shocked, locked_mask_ser.to_numpy()


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
    caps_map:    dict | None = None,
    floors_map:  dict | None = None,
    coeff_a_map: dict | None = None,
    coeff_b_map: dict | None = None,
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
                         fwd_rt_shocked derived from shocked disc curve over the
                         fixing period: (d_f(fixing_dt) / d_f(period_end) − 1) / yf_period
                         so that e.g. a 3M-fixing loan gets a 3M shocked rate.
    - A (administrative): today's administered rate + beta × (shocked − base forward),
                         same stable-margin rule as V (current accounts: beta=0 → unchanged).

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

    # ── Shocked discount factors and forward rates ────────────────────────────
    d_f_shocked_end, _, fwd_rt_arr, locked_mask = _build_shocked_disc_and_fwd(
        df, shocked_disc_df, disc_curve_map, eve_floor_rate
    )
    _require_discount(d_f_shocked_end, df, scenario_id)
    df["d_f_shocked"]    = np.nan_to_num(d_f_shocked_end, nan=0.0)
    df["fwd_rt_shocked"] = fwd_rt_arr

    # ── Contracted client rate (includes origination margin) ─────────────────
    # eff_rate in beh_df = fwd_rt for V products (pure index, no margin).
    # We compute the full contracted client rate from int_pmt so that
    # eff_rate_shocked carries the margin and int_pmt_shocked is correct.
    _denom_all = (
        df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    ).replace(0.0, float("nan"))
    _contracted_rt = (df["int_pmt"].fillna(0.0) / _denom_all).fillna(0.0)

    # ── Effective rate per CF (same logic as NII) ─────────────────────────────
    rate_type_s = df.get("rate_type", pd.Series("V", index=df.index))
    df["eff_rate_shocked"] = _contracted_rt                       # F: contracted (unchanged below)
    mask_var   = rate_type_s == "V"
    mask_admin = rate_type_s == "A"

    if mask_var.any() and "product_code" in df.columns:
        df.loc[mask_var, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_var, "fwd_rt_shocked"],
            df.loc[mask_var, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            base_eff_rate = _contracted_rt.loc[mask_var],
            base_fwd_rt   = df.loc[mask_var, "fwd_rt"],
        )
    elif mask_var.any():
        # No product_code: stable margin without a/b transform
        df.loc[mask_var, "eff_rate_shocked"] = (
            _contracted_rt.loc[mask_var]
            + (df.loc[mask_var, "fwd_rt_shocked"] - df.loc[mask_var, "fwd_rt"].fillna(0.0))
        )
    if mask_admin.any() and "product_code" in df.columns:   # A: today's rate + beta x index move
        # Anchored like V: the base CFs carry today's administered rate for every future
        # period, so the shock must move it by beta x (shocked - base forward). Pricing it
        # from the rate model on the shocked forward instead re-prices the base forwards
        # too (a steep curve then shifted every shocked EVE by a constant offset).
        df.loc[mask_admin, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_admin, "fwd_rt_shocked"], df.loc[mask_admin, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            base_eff_rate = _contracted_rt.loc[mask_admin],
            base_fwd_rt   = df.loc[mask_admin, "fwd_rt"].fillna(df.loc[mask_admin, "fwd_rt_shocked"]),
        )
    else:
        df.loc[mask_admin, "eff_rate_shocked"] = 0.0

    # ── Interest cash flows under shocked scenario ────────────────────────────
    # F: int_pmt unchanged (contractual); V: recomputed with full client rate; A: 0
    # Locked V (fixing_dt before report_date): int_pmt unchanged — rate already fixed.
    int_pmt_shocked = df["int_pmt"].fillna(0.0).to_numpy(dtype=float).copy()
    var_idx = mask_var.to_numpy()
    int_pmt_shocked[var_idx] = (
        df.loc[mask_var, "outstanding_bal"].fillna(0.0).to_numpy()
        * df.loc[mask_var, "eff_rate_shocked"].to_numpy()
        * df.loc[mask_var, "cf_yf"].fillna(0.0).to_numpy()
    )
    # Locked variable CFs: rate was fixed before report_date → payment cannot change
    lock_var = locked_mask & var_idx
    if lock_var.any():
        int_pmt_shocked[lock_var] = df["int_pmt"].fillna(0.0).to_numpy()[lock_var]
    _adm = mask_admin.to_numpy()
    int_pmt_shocked[_adm] = (
        df.loc[mask_admin, "outstanding_bal"].fillna(0.0).to_numpy()
        * df.loc[mask_admin, "eff_rate_shocked"].to_numpy(dtype=float)
        * df.loc[mask_admin, "cf_yf"].fillna(0.0).to_numpy()
    )
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

    # ── Shocked discount factors and forward rates ────────────────────────────
    d_f_shocked_end, _, fwd_rt_arr, locked_mask = _build_shocked_disc_and_fwd(
        df, shocked_disc_df, disc_curve_map, eve_floor_rate
    )
    _require_discount(d_f_shocked_end, df, scenario_id)
    df["d_f_shocked"]    = np.nan_to_num(d_f_shocked_end, nan=0.0)
    df["fwd_rt_shocked"] = fwd_rt_arr

    _denom_all = (
        df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    ).replace(0.0, float("nan"))
    _contracted_rt = (df["int_pmt"].fillna(0.0) / _denom_all).fillna(0.0)

    rate_type_s = df.get("rate_type", pd.Series("V", index=df.index))
    df["eff_rate_shocked"] = _contracted_rt
    mask_var   = rate_type_s == "V"
    mask_admin = rate_type_s == "A"

    if mask_var.any() and "product_code" in df.columns:
        df.loc[mask_var, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_var, "fwd_rt_shocked"],
            df.loc[mask_var, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            base_eff_rate = _contracted_rt.loc[mask_var],
            base_fwd_rt   = df.loc[mask_var, "fwd_rt"],
        )
    elif mask_var.any():
        df.loc[mask_var, "eff_rate_shocked"] = (
            _contracted_rt.loc[mask_var]
            + (df.loc[mask_var, "fwd_rt_shocked"] - df.loc[mask_var, "fwd_rt"].fillna(0.0))
        )
    if mask_admin.any() and "product_code" in df.columns:   # A: today's rate + beta x index move
        # Anchored like V: the base CFs carry today's administered rate for every future
        # period, so the shock must move it by beta x (shocked - base forward). Pricing it
        # from the rate model on the shocked forward instead re-prices the base forwards
        # too (a steep curve then shifted every shocked EVE by a constant offset).
        df.loc[mask_admin, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_admin, "fwd_rt_shocked"], df.loc[mask_admin, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            base_eff_rate = _contracted_rt.loc[mask_admin],
            base_fwd_rt   = df.loc[mask_admin, "fwd_rt"].fillna(df.loc[mask_admin, "fwd_rt_shocked"]),
        )
    else:
        df.loc[mask_admin, "eff_rate_shocked"] = 0.0

    int_pmt_shocked = df["int_pmt"].fillna(0.0).to_numpy(dtype=float).copy()
    var_idx = mask_var.to_numpy()
    int_pmt_shocked[var_idx] = (
        df.loc[mask_var, "outstanding_bal"].fillna(0.0).to_numpy()
        * df.loc[mask_var, "eff_rate_shocked"].to_numpy()
        * df.loc[mask_var, "cf_yf"].fillna(0.0).to_numpy()
    )
    # Locked variable CFs: rate was fixed before report_date → payment cannot change
    lock_var = locked_mask & var_idx
    if lock_var.any():
        int_pmt_shocked[lock_var] = df["int_pmt"].fillna(0.0).to_numpy()[lock_var]
    _adm = mask_admin.to_numpy()
    int_pmt_shocked[_adm] = (
        df.loc[mask_admin, "outstanding_bal"].fillna(0.0).to_numpy()
        * df.loc[mask_admin, "eff_rate_shocked"].to_numpy(dtype=float)
        * df.loc[mask_admin, "cf_yf"].fillna(0.0).to_numpy()
    )
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


def compute_base_cf_detail(beh_df: pd.DataFrame, scenario_id: str = "base") -> pd.DataFrame:
    """Row-level CF schedule for the base scenario.

    Prepares beh_df in the same interface as compute_shocked_cf_detail() so it
    can be passed directly to sql_setup.write_shocked_cf_products().  Uses base
    discount factors and rates — no shock applied.

    d_f_shocked_start is reconstructed as base_df * (1 + fwd_rt * cf_yf) from the
    base forward rate identity: fwd_rt = (d_f_start / d_f_end - 1) / cf_yf.
    """
    df = beh_df.copy()
    df["scenario_id"]    = scenario_id
    df["fwd_rt_shocked"] = df["fwd_rt"].fillna(0.0)
    # base_df is present in load_all_beh_schedules (EVE run-off mode) but absent
    # in load_beh_schedules (NII horizon-cut mode).  Fall back to 0 when missing;
    # d_f columns are informational in NII tables and not used in NII calculations.
    _base_df = df["base_df"].fillna(0.0) if "base_df" in df.columns else 0.0
    df["d_f_shocked"]       = _base_df
    df["d_f_shocked_start"] = (
        _base_df * (1.0 + df["fwd_rt"].fillna(0.0) * df["cf_yf"].fillna(0.0))
    )
    # eff_rate_shocked = contracted client rate (int_pmt / (outstanding × cf_yf)).
    # Using contracted rate (not raw fwd_rt) keeps the semantics consistent with
    # shocked scenarios where eff_rate_shocked also includes the origination margin.
    _denom = (
        df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    ).replace(0.0, float("nan"))
    df["eff_rate_shocked"]  = (df["int_pmt"].fillna(0.0) / _denom).fillna(0.0)
    df["int_pmt_shocked"]   = df["int_pmt"].fillna(0.0)
    return df


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
    CF rows (no groupby).  Used to populate the analytical cf.eve_*_scenarios
    and cf.nii_*_scenarios SQL tables so shocked cash flows can be compared
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

    # ── Shocked discount factors and forward rates ────────────────────────────
    d_f_shocked_end, d_f_shocked_start, fwd_rt_arr, locked_mask = _build_shocked_disc_and_fwd(
        df, shocked_disc_df, disc_curve_map, eve_floor_rate
    )
    _require_discount(d_f_shocked_end, df, scenario_id)
    df["d_f_shocked"]       = np.nan_to_num(d_f_shocked_end,   nan=0.0)
    df["d_f_shocked_start"] = np.nan_to_num(d_f_shocked_start, nan=0.0)
    df["fwd_rt_shocked"]    = fwd_rt_arr

    _denom_all = (
        df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    ).replace(0.0, float("nan"))
    _contracted_rt = (df["int_pmt"].fillna(0.0) / _denom_all).fillna(0.0)

    rate_type_s = df.get("rate_type", pd.Series("V", index=df.index))
    df["eff_rate_shocked"] = _contracted_rt
    mask_var   = rate_type_s == "V"
    mask_admin = rate_type_s == "A"

    if mask_var.any() and "product_code" in df.columns:
        df.loc[mask_var, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_var, "fwd_rt_shocked"],
            df.loc[mask_var, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            base_eff_rate = _contracted_rt.loc[mask_var],
            base_fwd_rt   = df.loc[mask_var, "fwd_rt"],
        )
    elif mask_var.any():
        df.loc[mask_var, "eff_rate_shocked"] = (
            _contracted_rt.loc[mask_var]
            + (df.loc[mask_var, "fwd_rt_shocked"] - df.loc[mask_var, "fwd_rt"].fillna(0.0))
        )
    if mask_admin.any() and "product_code" in df.columns:   # A: today's rate + beta x index move
        # Anchored like V: the base CFs carry today's administered rate for every future
        # period, so the shock must move it by beta x (shocked - base forward). Pricing it
        # from the rate model on the shocked forward instead re-prices the base forwards
        # too (a steep curve then shifted every shocked EVE by a constant offset).
        df.loc[mask_admin, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_admin, "fwd_rt_shocked"], df.loc[mask_admin, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            base_eff_rate = _contracted_rt.loc[mask_admin],
            base_fwd_rt   = df.loc[mask_admin, "fwd_rt"].fillna(df.loc[mask_admin, "fwd_rt_shocked"]),
        )
    else:
        df.loc[mask_admin, "eff_rate_shocked"] = 0.0

    int_pmt_shocked = df["int_pmt"].fillna(0.0).to_numpy(dtype=float).copy()
    var_idx = mask_var.to_numpy()
    int_pmt_shocked[var_idx] = (
        df.loc[mask_var, "outstanding_bal"].fillna(0.0).to_numpy()
        * df.loc[mask_var, "eff_rate_shocked"].to_numpy()
        * df.loc[mask_var, "cf_yf"].fillna(0.0).to_numpy()
    )
    # Locked variable CFs: rate was fixed before report_date → payment cannot change
    lock_var = locked_mask & var_idx
    if lock_var.any():
        int_pmt_shocked[lock_var] = df["int_pmt"].fillna(0.0).to_numpy()[lock_var]
    _adm = mask_admin.to_numpy()
    int_pmt_shocked[_adm] = (
        df.loc[mask_admin, "outstanding_bal"].fillna(0.0).to_numpy()
        * df.loc[mask_admin, "eff_rate_shocked"].to_numpy(dtype=float)
        * df.loc[mask_admin, "cf_yf"].fillna(0.0).to_numpy()
    )
    df["int_pmt_shocked"] = int_pmt_shocked
    df["scenario_id"]     = scenario_id
    return df
