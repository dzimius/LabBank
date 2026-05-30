from __future__ import annotations

import numpy as np
import pandas as pd

_TENOR_SORT = {"1D": 0, ">30Y": 999}


def _apply_rt_limits(
    rate: "np.ndarray | pd.Series",
    product_codes: pd.Series,
    caps_map:      dict | None = None,
    floors_map:    dict | None = None,
    coeff_a_map:   dict | None = None,
    coeff_b_map:   dict | None = None,
    base_eff_rate: "np.ndarray | pd.Series | None" = None,
    base_fwd_rt:   "np.ndarray | pd.Series | None" = None,
) -> np.ndarray:
    """Apply per-product rate transform and floor/cap.

    Existing book — pass base_eff_rate and base_fwd_rt (stable-margin formula):
        client_rt = base_eff_rate + a * (rate - base_fwd_rt)
        base_eff_rate already contains the origination margin (b), so b is NOT
        added again.  With a=1: client_rt = base_eff_rate + delta_fwd (margin
        preserved exactly).

    Renewal / new business — omit base rates:
        client_rt = a * rate + b
        Here b is the contractual spread applied to the shocked index rate.

    Products absent from coeff maps default to a=1, b=0.
    All values must be in decimal.
    """
    shocked = np.asarray(rate, dtype=float)
    pcs     = product_codes.astype(str)
    a_v     = pcs.map(coeff_a_map or {}).fillna(1.0).to_numpy(dtype=float)
    b_v     = pcs.map(coeff_b_map or {}).fillna(0.0).to_numpy(dtype=float)

    if base_eff_rate is not None and base_fwd_rt is not None:
        base_e = np.asarray(base_eff_rate, dtype=float)
        base_f = np.asarray(base_fwd_rt,   dtype=float)
        # b is already embedded in base_eff_rate (origination margin) — do not add it again
        result = base_e + a_v * (shocked - base_f)
    else:
        result = a_v * shocked + b_v

    if floors_map:
        floor_v = pcs.map(floors_map).fillna(float("-inf")).to_numpy(dtype=float)
        result  = np.maximum(floor_v, result)
    if caps_map:
        cap_v   = pcs.map(caps_map).fillna(float("inf")).to_numpy(dtype=float)
        result  = np.minimum(cap_v, result)
    return result


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute NII sensitivity from ir_gap_beh repricing gap.

    ir_gap_beh must contain split columns written by the BS + IRS workflows:
      gap_cf     — total balance-sheet gap (no IRS)
      gap_cf_ca  — current-account portion (floored at 0% in down shock)
      gap_cf_irs — IRS gap contribution

    For each tenor bucket within the horizon:
        remain_yf              = horizon_yf - tenor_yf
        d_nii_up_100_no_swap   = gap_cf * shock_up * remain_yf
        d_nii_dn_100_no_swap   = (gap_cf - gap_cf_ca) * shock_dn * remain_yf
        d_nii_up_100_with_swap = (gap_cf + gap_cf_irs) * shock_up * remain_yf
        d_nii_dn_100_with_swap = (gap_cf + gap_cf_irs - gap_cf_ca) * shock_dn * remain_yf

    Parameters
    ----------
    ir_gap_beh : unified gap table with gap_cf, gap_cf_ca, gap_cf_irs columns
    horizon_yf : NII horizon in year fractions (default 1.0)
    shock_up   : rate shock up in decimal (default +100bps = 0.01)
    shock_dn   : rate shock down in decimal (default -100bps = -0.01)

    Returns
    -------
    detail_df  : per (currency, tenor_bucket) breakdown
    summary_df : total NII per currency
    """
    df = ir_gap_beh.copy()
    for col in ("gap_cf_ca", "gap_cf_irs"):
        if col not in df.columns:
            df[col] = 0.0

    # Aggregate gap components by (currency, tenor_bucket)
    net_gap = (
        df.groupby(["currency", "tenor_bucket"], sort=False)
        [["gap_cf", "gap_cf_ca", "gap_cf_irs"]]
        .sum()
        .reset_index()
    )

    # Tenor year fraction and filter to horizon
    net_gap["tenor_yf"]  = net_gap["tenor_bucket"].map(_tenor_to_yf)
    net_gap = net_gap[net_gap["tenor_yf"] <= horizon_yf].copy()
    net_gap["remain_yf"] = (horizon_yf - net_gap["tenor_yf"]).clip(lower=0.0)

    # Derived gap columns
    net_gap["net_gap_without_ca"]      = net_gap["gap_cf"] - net_gap["gap_cf_ca"]
    net_gap["net_gap_with_irs"]        = net_gap["gap_cf"] + net_gap["gap_cf_irs"]
    net_gap["net_gap_with_irs_no_ca"]  = net_gap["net_gap_with_irs"] - net_gap["gap_cf_ca"]

    # NII
    net_gap["d_nii_up_100_no_swap"]   = net_gap["gap_cf"]               * shock_up * net_gap["remain_yf"]
    net_gap["d_nii_dn_100_no_swap"]   = net_gap["net_gap_without_ca"]   * shock_dn * net_gap["remain_yf"]
    net_gap["d_nii_up_100_with_swap"] = net_gap["net_gap_with_irs"]     * shock_up * net_gap["remain_yf"]
    net_gap["d_nii_dn_100_with_swap"] = net_gap["net_gap_with_irs_no_ca"] * shock_dn * net_gap["remain_yf"]

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
        "gap_cf", "gap_cf_ca", "net_gap_without_ca",
        "gap_cf_irs", "net_gap_with_irs", "net_gap_with_irs_no_ca",
        "d_nii_up_100_no_swap", "d_nii_dn_100_no_swap",
        "d_nii_up_100_with_swap", "d_nii_dn_100_with_swap",
    ]]

    nii_cols = [
        "d_nii_up_100_no_swap", "d_nii_dn_100_no_swap",
        "d_nii_up_100_with_swap", "d_nii_dn_100_with_swap",
    ]
    summary_df = (
        detail_df
        .groupby("currency")[nii_cols]
        .sum()
        .reset_index()
    )

    return detail_df, summary_df


def compute_nii_shocked(
    beh_df: pd.DataFrame,
    shocked_disc_df: pd.Series,         # MultiIndex (curve_name, node_date) → d_f
    disc_curve_map: dict[str, str],     # currency → disc_curve_name
    report_date: pd.Timestamp,
    scenario_id: str,
    horizon_yf: float = 1.0,
    nii_floor_rate: float = 0.0,
    caps_map:    dict | None = None,
    floors_map:  dict | None = None,
    coeff_a_map: dict | None = None,
    coeff_b_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute NII under a shocked rate scenario (EBA constant balance sheet).

    For each CF in beh_df within the horizon:

    Existing-portfolio interest (eff_rate logic)
    --------------------------------------------
    - fixed products  : int_pmt unchanged (contractual rate locked).
                        eff_rate = int_pmt / (outstanding_bal × cf_yf).
    - floating products: int_pmt recomputed using shocked forward rate derived
                        from the shocked discount curve, with 0% floor.
                        Shocked fwd_rt ≈ (d_f_start / d_f_end − 1) / cf_yf.
    - current accounts: treated as float but fwd_rt = 0 (product rate already
                        0% — insensitive to down shocks; no special override needed
                        here because the product rate IS already 0 in eff_rate).

    Renewal interest (constant balance assumption)
    ----------------------------------------------
    Capital repaid within the horizon is reinvested/re-borrowed at the shocked
    forward rate for the remaining tenor to horizon-end:
      nii_renewal = capital_total × max(0, shocked_renewal_fwd_rt) × remain_yf × sign

    Returns
    -------
    detail_df  : per (currency, source, rate_type) breakdown
    summary_df : total NII per currency
    Both also include a 'scenario_id' column.
    """
    import numpy as np

    horizon_end = report_date + pd.Timedelta(days=round(horizon_yf * 365.25))
    df = beh_df[beh_df["cf_end_dt"] <= horizon_end].copy()

    # ── Shocked forward rates for each CF ─────────────────────────────────────
    # For each CF row: fwd_rt_shocked = (d_f(cf_start_dt) / d_f(cf_end_dt) - 1) / cf_yf
    # Uses the pre-loaded daily shocked disc curve (same curve for the currency).
    def _lookup_df(dates: pd.Series, curve_name: str) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays(
            [[curve_name] * len(dates), dates.to_numpy()],
            names=["curve_name", "node_date"],
        )
        return shocked_disc_df.reindex(idx).to_numpy(dtype=float)

    fwd_rt_shocked = np.full(len(df), np.nan)
    for ccy, grp in df.groupby("currency"):
        cn = disc_curve_map.get(ccy)
        if cn is None:
            continue
        idx = grp.index
        d_f_start = _lookup_df(df.loc[idx, "cf_start_dt"], cn)
        d_f_end   = _lookup_df(df.loc[idx, "cf_end_dt"],   cn)
        yf        = df.loc[idx, "cf_yf"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = np.where(
                (d_f_end > 0) & (yf > 0) & ~np.isnan(d_f_end),
                (d_f_start / d_f_end - 1.0) / yf,
                np.nan,
            )
        fwd_rt_shocked[idx] = fwd

    df["fwd_rt_shocked"] = np.maximum(nii_floor_rate, np.nan_to_num(fwd_rt_shocked, nan=0.0))

    # Lock CFs whose rate is already fixed — same two-condition logic as
    # compute_nii_shocked_schedule (see its comment for full rationale).
    if "fwd_rt" in df.columns:
        locked = pd.Series(False, index=df.index)
        if "cf_start_dt" in df.columns:
            locked = locked | (df["cf_start_dt"] < report_date)
        if "fixing_dt" in df.columns:
            locked = locked | (
                df["fixing_dt"].notna()
                & (pd.to_datetime(df["fixing_dt"]) < report_date)
            )
        if locked.any():
            df.loc[locked, "fwd_rt_shocked"] = df.loc[locked, "fwd_rt"].fillna(0.0)

    # ── Contracted client rate (full rate including margin b) ─────────────────
    # Back-computed from int_pmt so that base_eff_rate in the stable-margin
    # formula carries the origination spread (b), matching the approach used
    # in eve_calc_objects.compute_shocked_cf_detail.
    # For F products eff_rate already equals contracted_rt; for V products
    # eff_rate = fwd_rt (pure index, no b) — so we must use contracted_rt.
    _denom_all = (
        df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    ).replace(0.0, float("nan"))
    _contracted_rt = (df.get("int_pmt", pd.Series(0.0, index=df.index)).fillna(0.0)
                      / _denom_all).fillna(0.0)

    # ── Effective rate per CF (eff_rate_shocked) ───────────────────────────────
    # F (fixed)          → contracted_rt (unchanged)
    # V (variable)       → stable-margin: contracted_rt + a*(fwd_shocked - base_fwd)
    # A (administrative) → 0% regardless of scenario (bank-managed rate)
    rate_type_s = df.get("rate_type", pd.Series("V", index=df.index))
    df["eff_rate_shocked"] = _contracted_rt                                  # F: contracted
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
        df.loc[mask_var, "eff_rate_shocked"] = df.loc[mask_var, "fwd_rt_shocked"]
    df.loc[mask_admin, "eff_rate_shocked"] = 0.0                            # A: always 0%

    # ── NII components ─────────────────────────────────────────────────────────
    df["sign"] = np.where(df["bs_side"] == "A", 1.0, -1.0)

    # Interest on existing positions at shocked client rate
    df["nii_interest"] = (
        df["outstanding_bal"].fillna(0.0)
        * df["eff_rate_shocked"]
        * df["cf_yf"].fillna(0.0)
        * df["sign"]
    )

    # Renewal: ALL products renew at shocked market rate (not contracted),
    # consistent with compute_nii_shocked_schedule.  Fixed products that
    # mature within the horizon re-invest/re-borrow at the current shocked
    # market client rate, not at their old contracted rate.
    df["remain_yf"]    = ((horizon_end - df["cf_end_dt"]).dt.days / 365.0).clip(lower=0.0)
    df["total_capital"]= df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)
    if "product_code" in df.columns:
        renewal_rt = _apply_rt_limits(
            df["fwd_rt_shocked"], df["product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
        )
    else:
        renewal_rt = df["fwd_rt_shocked"].to_numpy()
    if "rate_type" in df.columns:
        renewal_rt = renewal_rt.copy()
        renewal_rt[(df["rate_type"] == "A").to_numpy()] = 0.0
    df["nii_renewal"]  = (
        df["total_capital"]
        * renewal_rt
        * df["remain_yf"]
        * df["sign"]
    )
    df["nii_total"]    = df["nii_interest"] + df["nii_renewal"]
    df["scenario_id"]  = scenario_id

    rate_type_col = df["rate_type"] if "rate_type" in df.columns else "float"
    group_cols    = ["currency", "source", "rate_type"] if "rate_type" in df.columns \
                    else ["currency", "source"]

    detail_df = (
        df.groupby(group_cols)[["nii_interest", "nii_renewal", "nii_total"]]
        .sum()
        .reset_index()
        .assign(scenario_id=scenario_id)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    summary_df = (
        detail_df.groupby("currency")[["nii_interest", "nii_renewal", "nii_total"]]
        .sum()
        .reset_index()
        .assign(scenario_id=scenario_id)
    )
    return detail_df, summary_df


_SCHED_GROUP_COLS = ["schedule_id", "product_type", "product_code",
                     "currency", "bs_side", "rate_type"]


def compute_nii_base_schedule(
    beh_df: pd.DataFrame,
    report_date: pd.Timestamp,
    horizon_yf: float = 1.0,
    scenario_id: str = "base",
    caps_map:    dict | None = None,
    floors_map:  dict | None = None,
    coeff_a_map: dict | None = None,
    coeff_b_map: dict | None = None,
    base_disc_df: "pd.Series | None" = None,
    disc_curve_map: "dict | None" = None,
) -> pd.DataFrame:
    """Per-schedule NII for the base scenario (no rate shock).

    Groups at (schedule_id, product_type, product_code, currency, bs_side, rate_type).

    Renewal rate
    ------------
    For CFs whose period started before report_date (cf_start_dt < report_date), fwd_rt
    in the CF table stores the contracted/coupon rate for fixed products — NOT a current
    market rate.  When base_disc_df and disc_curve_map are provided, the renewal rate for
    these rows is derived from the base discount curve instead:
        fwd_ren = (1 / d_f_base(cf_end_dt) − 1) / (cf_end_dt − report_date).days * 365
    This is the same formula used in compute_nii_shocked_schedule for shocked scenarios,
    so delta_nii correctly reflects only the market rate change.

    For CFs starting on or after report_date, and as a fallback when base_disc_df is
    absent, fwd_rt is used unchanged (current market forward rate for those periods).

    Returns
    -------
    DataFrame with: scenario_id + _SCHED_GROUP_COLS + nii_interest/renewal/total
    """
    horizon_end = report_date + pd.Timedelta(days=round(horizon_yf * 365.25))
    df = beh_df[beh_df["cf_end_dt"] <= horizon_end].copy()
    df["sign"] = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["nii_interest"]  = df["int_pmt"].fillna(0.0) * df["sign"]
    df["remain_yf"]     = ((horizon_end - df["cf_end_dt"]).dt.days / 365.0).clip(lower=0.0)
    df["total_capital"] = df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)

    # Base renewal rate: use fwd_rt by default.
    # For CFs starting before report_date, fwd_rt may be the contracted rate (fixed
    # products) rather than the current market rate — override with the disc-curve-derived
    # spot rate so that base and shocked renewal use the same formula.
    renewal_base = (df["fwd_rt"].fillna(0.0) if "fwd_rt" in df.columns
                    else df["eff_rate"].fillna(0.0))
    if base_disc_df is not None and disc_curve_map is not None and "cf_start_dt" in df.columns:
        _before_report = df["cf_start_dt"] < report_date
        _yf_from_report = ((df["cf_end_dt"] - report_date).dt.days / 365.0).clip(lower=0.0)
        _valid_ren = _before_report & (_yf_from_report > 0)
        if _valid_ren.any():
            _d_f_base_end = pd.Series(np.nan, index=df.index)
            for ccy in df.loc[_valid_ren, "currency"].unique():
                cn = disc_curve_map.get(ccy)
                if cn is None:
                    continue
                cmask = _valid_ren & (df["currency"] == ccy)
                mi = pd.MultiIndex.from_arrays(
                    [[cn] * int(cmask.sum()), df.loc[cmask, "cf_end_dt"].to_numpy()],
                    names=["curve_name", "node_date"],
                )
                _d_f_base_end.loc[cmask] = base_disc_df.reindex(mi).to_numpy(dtype=float)
            _has_df = _valid_ren & _d_f_base_end.notna() & (_d_f_base_end > 0)
            if _has_df.any():
                renewal_base = renewal_base.copy()
                renewal_base.loc[_has_df] = np.maximum(
                    0.0,
                    (1.0 / _d_f_base_end.loc[_has_df].to_numpy() - 1.0)
                    / _yf_from_report.loc[_has_df].to_numpy(),
                )

    if "product_code" in df.columns:
        renewal_rt = _apply_rt_limits(renewal_base, df["product_code"],
                                      caps_map, floors_map, coeff_a_map, coeff_b_map)
    else:
        renewal_rt = renewal_base.to_numpy()
    if "rate_type" in df.columns:
        renewal_rt = renewal_rt.copy()
        renewal_rt[(df["rate_type"] == "A").to_numpy()] = 0.0
    df["nii_renewal"] = df["total_capital"] * renewal_rt * df["remain_yf"] * df["sign"]
    df["nii_total"] = df["nii_interest"] + df["nii_renewal"]

    group_cols = [c for c in _SCHED_GROUP_COLS if c in df.columns]
    return (
        df.groupby(group_cols)[["nii_interest", "nii_renewal", "nii_total"]]
        .sum()
        .reset_index()
        .assign(scenario_id=scenario_id)
    )


def compute_nii_shocked_schedule(
    beh_df: pd.DataFrame,
    shocked_disc_df: pd.Series,
    disc_curve_map: dict[str, str],
    report_date: pd.Timestamp,
    scenario_id: str,
    horizon_yf: float = 1.0,
    nii_floor_rate: float = 0.0,
    caps_map:    dict | None = None,
    floors_map:  dict | None = None,
    coeff_a_map: dict | None = None,
    coeff_b_map: dict | None = None,
) -> pd.DataFrame:
    """Per-schedule NII under a shocked rate scenario.

    For variable products: eff_rate_shocked = max(floor, min(cap, fwd_rt_shocked)).
    CAs with client_cap=0.001 (0.1%) stay capped regardless of shock size.

    Returns
    -------
    DataFrame with: scenario_id + _SCHED_GROUP_COLS + nii_interest/renewal/total
    """
    import numpy as np

    horizon_end = report_date + pd.Timedelta(days=round(horizon_yf * 365.25))
    df = beh_df[beh_df["cf_end_dt"] <= horizon_end].copy()

    def _lookup_df(dates: pd.Series, curve_name: str) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays(
            [[curve_name] * len(dates), dates.to_numpy()],
            names=["curve_name", "node_date"],
        )
        return shocked_disc_df.reindex(idx).to_numpy(dtype=float)

    fwd_rt_shocked = np.full(len(df), np.nan)
    d_f_end_arr   = np.full(len(df), np.nan)   # shocked d_f at cf_end_dt per CF
    for ccy, grp in df.groupby("currency"):
        cn = disc_curve_map.get(ccy)
        if cn is None:
            continue
        idx = grp.index
        d_f_start = _lookup_df(df.loc[idx, "cf_start_dt"], cn)
        d_f_end   = _lookup_df(df.loc[idx, "cf_end_dt"],   cn)
        yf        = df.loc[idx, "cf_yf"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = np.where(
                (d_f_end > 0) & (yf > 0) & ~np.isnan(d_f_end),
                (d_f_start / d_f_end - 1.0) / yf,
                np.nan,
            )
        fwd_rt_shocked[idx] = fwd
        d_f_end_arr[idx]    = d_f_end

    df["fwd_rt_shocked"] = np.maximum(nii_floor_rate, np.nan_to_num(fwd_rt_shocked, nan=0.0))

    # ── Fix fwd_rt_shocked for CFs starting before the shock curve ───────────
    # When cf_start_dt < report_date, d_f_start is NaN (shocked curve starts at
    # report_date).  nan_to_num converts NaN → 0, making fwd_rt_shocked = 0 for
    # these rows.  Substitute base fwd_rt for interest locking — renewal is
    # handled separately below with the shocked d_f at cf_end_dt.
    _before_curve = pd.Series(False, index=df.index)
    if "fwd_rt" in df.columns and "cf_start_dt" in df.columns:
        _before_curve = df["cf_start_dt"] < report_date
        if _before_curve.any():
            df.loc[_before_curve, "fwd_rt_shocked"] = (
                df.loc[_before_curve, "fwd_rt"].fillna(0.0)
            )

    # ── Lock detection for nii_interest ────────────────────────────────────────
    # Build fwd_rt_for_interest: like fwd_rt_shocked but substitutes base fwd_rt
    # for CFs whose interest rate is already contractually fixed.
    # Two cases — both must be caught (see comment in write_shocked_cf_products):
    #   1. cf_start_dt < report_date  — period already started
    #   2. fixing_dt   < report_date  — rate set before report_date even if
    #                                   cf_start_dt is on/after report_date
    #                                   (annual-reset loan fixed 2024-01-01,
    #                                    2025 interest CF starts 2024-12-31)
    # NOTE: fwd_rt_shocked (the full shocked rate) is intentionally kept unchanged
    # so that renewal uses the shocked market rate — capital that matures renews
    # at current market conditions regardless of the old fixing date.
    fwd_rt_for_interest = df["fwd_rt_shocked"].copy()
    if "fwd_rt" in df.columns:
        locked = pd.Series(False, index=df.index)
        if "cf_start_dt" in df.columns:
            locked = locked | (df["cf_start_dt"] < report_date)
        if "fixing_dt" in df.columns:
            locked = locked | (
                df["fixing_dt"].notna()
                & (pd.to_datetime(df["fixing_dt"]) < report_date)
            )
        if locked.any():
            fwd_rt_for_interest[locked] = df.loc[locked, "fwd_rt"].fillna(0.0)

    # ── Contracted client rate (full rate including margin b) ─────────────────
    # Same approach as eve_calc_objects.compute_shocked_cf_detail: back-compute
    # from int_pmt so that base_eff_rate in the stable-margin formula carries b.
    _denom_s = (
        df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    ).replace(0.0, float("nan"))
    _contracted_rt_s = (df.get("int_pmt", pd.Series(0.0, index=df.index)).fillna(0.0)
                        / _denom_s).fillna(0.0)

    # eff_rate_shocked: F=contracted (unchanged), V=stable-margin on locked rate, A=0%
    rate_type_s = df.get("rate_type", pd.Series("V", index=df.index))
    df["eff_rate_shocked"] = _contracted_rt_s                      # F: contracted rate
    mask_var   = rate_type_s == "V"
    mask_admin = rate_type_s == "A"
    if mask_var.any() and "product_code" in df.columns:
        df.loc[mask_var, "eff_rate_shocked"] = _apply_rt_limits(
            fwd_rt_for_interest.loc[mask_var],   # ← locking-aware rate for interest
            df.loc[mask_var, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            base_eff_rate = _contracted_rt_s.loc[mask_var],
            base_fwd_rt   = df.loc[mask_var, "fwd_rt"],
        )
    elif mask_var.any():
        df.loc[mask_var, "eff_rate_shocked"] = fwd_rt_for_interest.loc[mask_var]
    df.loc[mask_admin, "eff_rate_shocked"] = 0.0

    df["sign"] = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["nii_interest"] = (
        df["outstanding_bal"].fillna(0.0)
        * df["eff_rate_shocked"]
        * df["cf_yf"].fillna(0.0)
        * df["sign"]
    )
    df["remain_yf"]     = ((horizon_end - df["cf_end_dt"]).dt.days / 365.0).clip(lower=0.0)
    df["total_capital"] = df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)

    # ── Renewal forward rate ──────────────────────────────────────────────────
    # Default: fwd_rt_shocked (the CF-period shocked rate).
    # For start-before-curve CFs fwd_rt_shocked was set to base_fwd_rt (interest
    # locking).  For renewal, capital maturing from these CFs should re-price at
    # the shocked market rate, derived from the shocked d_f at cf_end_dt:
    #   rate = (1/d_f_end - 1) / year_frac_from_report_date
    # This produces a non-zero delta vs base for fixed-rate bonds and other
    # start-before-curve products.
    _yf_from_report = ((df["cf_end_dt"] - report_date).dt.days / 365.0).clip(lower=0.0)
    _d_f_end_ser    = pd.Series(np.nan_to_num(d_f_end_arr, nan=0.0), index=df.index)
    _ren_shock_mask = _before_curve & (_d_f_end_ser > 0) & (_yf_from_report > 0)
    fwd_rt_for_renewal = df["fwd_rt_shocked"].copy()
    if _ren_shock_mask.any():
        fwd_rt_for_renewal[_ren_shock_mask] = np.maximum(
            nii_floor_rate,
            (1.0 / _d_f_end_ser[_ren_shock_mask] - 1.0)
            / _yf_from_report[_ren_shock_mask],
        )

    # Renewal uses fwd_rt_for_renewal (shocked market rate, shock-curve derived)
    # so that maturing capital renews at current shocked conditions.
    if "product_code" in df.columns:
        renewal_rt_shocked = _apply_rt_limits(
            fwd_rt_for_renewal, df["product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
        )
    else:
        renewal_rt_shocked = fwd_rt_for_renewal.to_numpy()
    renewal_rt_shocked[mask_admin.to_numpy()] = 0.0
    df["nii_renewal"]   = (
        df["total_capital"] * renewal_rt_shocked * df["remain_yf"] * df["sign"]
    )
    df["nii_total"] = df["nii_interest"] + df["nii_renewal"]

    group_cols = [c for c in _SCHED_GROUP_COLS if c in df.columns]
    return (
        df.groupby(group_cols)[["nii_interest", "nii_renewal", "nii_total"]]
        .sum()
        .reset_index()
        .assign(scenario_id=scenario_id)
    )


def compute_nii_base(
    beh_df: pd.DataFrame,
    report_date: pd.Timestamp,
    horizon_yf: float = 1.0,
    caps_map:    dict | None = None,
    floors_map:  dict | None = None,
    coeff_a_map: dict | None = None,
    coeff_b_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute base NII from behavioral CF schedules (constant balance assumption, no rate shock).

    Two NII components per cash flow within the 1Y horizon:

    1. nii_interest  — interest from the existing schedule:
           int_pmt × sign
       This is the actual contractual/behavioural interest payment.

    2. nii_renewal   — interest on returned capital (constant balance):
           (capital_pmt + prepayment_pmt) × client_fwd_rt × remain_yf × sign
       Renewal uses fwd_rt with the same per-product linear transform and
       floor/cap applied as the shocked scenario, so delta_nii reflects only
       the market rate change (margins cancel between base and shocked).
           remain_yf = (horizon_end − cf_end_dt).days / 365

    Sign convention: A (asset) = +1 (income), L (liability) = −1 (expense).

    Parameters
    ----------
    beh_df      : unified behavioral CF DataFrame from sql_setup.load_beh_schedules()
    report_date : valuation date
    horizon_yf  : NII horizon in year fractions (default 1.0 = 1 year)
    caps_map, floors_map, coeff_a_map, coeff_b_map : per-product rate limits / transforms

    Returns
    -------
    detail_df   : NII breakdown per (currency, source) — loans / deposits / fin_inst
    summary_df  : total NII per currency
    """
    horizon_end = report_date + pd.Timedelta(days=round(horizon_yf * 365.25))

    df = beh_df[beh_df["cf_end_dt"] <= horizon_end].copy()

    # Sign: asset = income (+1), liability = expense (-1)
    df["sign"] = np.where(df["bs_side"] == "A", 1.0, -1.0)

    # Component 1: interest from existing cash flow schedule
    df["nii_interest"] = df["int_pmt"].fillna(0.0) * df["sign"]

    # Component 2: renewal interest on returned capital (constant balance)
    df["remain_yf"] = (
        (horizon_end - df["cf_end_dt"]).dt.days / 365.0
    ).clip(lower=0.0)
    df["total_capital"] = df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)

    # Apply same client-rate transform as shocked scenario so margins cancel in delta
    renewal_base = df["fwd_rt"].fillna(0.0)
    if "product_code" in df.columns:
        renewal_rt = _apply_rt_limits(
            renewal_base, df["product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
        )
    else:
        renewal_rt = renewal_base.to_numpy()
    if "rate_type" in df.columns:
        renewal_rt = renewal_rt.copy()
        renewal_rt[(df["rate_type"] == "A").to_numpy()] = 0.0

    df["nii_renewal"] = (
        df["total_capital"] * renewal_rt * df["remain_yf"] * df["sign"]
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
