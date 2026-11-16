from __future__ import annotations

import numpy as np
import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # sibling fixing_engine
import fixing_engine as fe  # noqa: E402
from labbank_common.curves import forward_rate, renewal_market_rate  # noqa: E402

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
    contract_margin: "np.ndarray | pd.Series | None" = None,
) -> np.ndarray:
    """Apply per-product rate transform and floor/cap.

    Existing book — pass base_eff_rate and base_fwd_rt (anchored formula):
        client_rt = base_eff_rate + F(rate) - F(base_fwd_rt)
        F = the full contractual formula clip(a * clip(index, index floor/cap) + b,
        client floor/cap) -- index limits from floors_map.index / caps_map.index
        (rate_models.LimitMap). Exact at zero shock; in the linear region it is
        base_eff_rate + a * (rate - base_fwd_rt) (margin preserved exactly).

    Renewal / new business / administered rates — omit base rates:
        client_rt = a * rate + b
        b = product-level spread (coeff_b_map) when the product has one, else
        contract_margin (the contract's own spread from cf.products.margin, e.g.
        a mortgage's origination margin -- EBA constant balance sheet renews
        like-for-like), else 0.  Same rule as labbank_common.rate_models.

    Products absent from coeff maps default to a=1.
    All values must be in decimal.
    """
    shocked = np.asarray(rate, dtype=float)
    pcs     = pd.Series(np.asarray(product_codes)).astype(str)
    a_v     = pcs.map(coeff_a_map or {}).fillna(1.0).to_numpy(dtype=float)
    b_v     = pcs.map(coeff_b_map or {}).to_numpy(dtype=float)
    if contract_margin is not None:
        cm = pd.to_numeric(pd.Series(np.asarray(contract_margin)), errors="coerce").to_numpy(dtype=float)
        b_v = np.where(np.isnan(b_v), cm, b_v)
    b_v     = np.nan_to_num(b_v, nan=0.0)

    def _lim(m, default):
        return (pcs.map(m).fillna(default).to_numpy(dtype=float) if m
                else np.full(len(pcs), default))
    # client limits = the maps themselves; index limits = their .index (rate_models.LimitMap)
    cfl, ccp = _lim(floors_map, -np.inf), _lim(caps_map, np.inf)
    ifl = _lim(getattr(floors_map, "index", None), -np.inf)
    icp = _lim(getattr(caps_map, "index", None), np.inf)

    def F(idx):
        """Full contractual formula (labbank_common.rate_models.client_rate)."""
        x = np.minimum(np.maximum(np.asarray(idx, dtype=float), ifl), icp)
        return np.minimum(np.maximum(a_v * x + b_v, cfl), ccp)

    if base_eff_rate is not None and base_fwd_rt is not None:
        base_e = np.asarray(base_eff_rate, dtype=float)
        base_f = np.asarray(base_fwd_rt,   dtype=float)
        # Anchored on today's contractual / administered rate (exact at zero shock), moved
        # by the change of the FULL formula: an index floor / client cap binding in base or
        # shock is respected (base + beta x index change ignored both).
        result = base_e + (F(shocked) - F(base_f))
        result = np.minimum(np.maximum(result, cfl), ccp)
    else:
        result = F(shocked)
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
    gap_beta: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple (first-order) NII sensitivity from the repricing gap -- a DIAGNOSTIC next to
    the cash-flow NII (irrbb.nii_results), not the regulatory number.

    *_unadj columns: the plain repricing gap (every unit reprices 1:1 with the market).
    *_beta  columns (when gap_beta is given): the beta-adjusted gap
    (labbank_common.rate_models.beta_adjusted_gap: each product x its effective client-rate
    beta, NMDs too along their repricing profile) -- the gap view consistent with the
    cash-flow engine's rate model. The old rule "current accounts insensitive to down shocks"
    (gap minus gap_cf_ca) is gone: account sensitivity is the beta of the rate model.

    gap_beta : DataFrame currency, tenor_bucket, gap_beta_bs, gap_beta_irs

    ir_gap_beh: gap_cf (balance sheet, no IRS) and gap_cf_irs per currency x tenor bucket.
    For each tenor bucket within the horizon (remain_yf = horizon_yf - tenor_yf):
        d_nii_<up|dn>_100_<no|with>_swap_unadj = gap (+ IRS) x shock x remain_yf
        d_nii_<up|dn>_100_with_swap_beta       = (gap_beta_bs + gap_beta_irs) x shock x remain_yf

    Parameters
    ----------
    ir_gap_beh : unified gap table with gap_cf, gap_cf_irs columns
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
    net_gap["net_gap_with_irs"]        = net_gap["gap_cf"] + net_gap["gap_cf_irs"]
    if gap_beta is not None and not gap_beta.empty:
        net_gap = net_gap.merge(gap_beta, on=["currency", "tenor_bucket"], how="left")
    for c in ("gap_beta_bs", "gap_beta_irs"):
        if c not in net_gap.columns:
            net_gap[c] = np.nan
    net_gap["net_gap_beta_with_irs"]   = net_gap["gap_beta_bs"] + net_gap["gap_beta_irs"].fillna(0.0)

    # NII: unadjusted (diagnostic) and beta-adjusted
    w = net_gap["remain_yf"]
    net_gap["d_nii_up_100_no_swap_unadj"]   = net_gap["gap_cf"]           * shock_up * w
    net_gap["d_nii_dn_100_no_swap_unadj"]   = net_gap["gap_cf"]           * shock_dn * w
    net_gap["d_nii_up_100_with_swap_unadj"] = net_gap["net_gap_with_irs"] * shock_up * w
    net_gap["d_nii_dn_100_with_swap_unadj"] = net_gap["net_gap_with_irs"] * shock_dn * w
    net_gap["d_nii_up_100_with_swap_beta"]  = net_gap["net_gap_beta_with_irs"] * shock_up * w
    net_gap["d_nii_dn_100_with_swap_beta"]  = net_gap["net_gap_beta_with_irs"] * shock_dn * w

    # Sort by currency then tenor
    net_gap["_sort"] = net_gap["tenor_bucket"].map(_tenor_sort_key)
    detail_df = (
        net_gap
        .sort_values(["currency", "_sort"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    nii_cols = [
        "d_nii_up_100_with_swap_beta", "d_nii_dn_100_with_swap_beta",
        "d_nii_up_100_no_swap_unadj", "d_nii_dn_100_no_swap_unadj",
        "d_nii_up_100_with_swap_unadj", "d_nii_dn_100_with_swap_unadj",
    ]
    detail_df = detail_df[[
        "currency", "tenor_bucket", "tenor_yf", "remain_yf",
        "gap_cf", "gap_cf_irs", "net_gap_with_irs",
        "gap_beta_bs", "gap_beta_irs", "net_gap_beta_with_irs",
    ] + nii_cols]
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
    df = beh_df[beh_df["cf_start_dt"] < horizon_end].copy()      # coupons overlapping the year
    if "accr_frac" not in df.columns:
        fe.add_horizon_accrual(df, report_date, horizon_end)

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
                forward_rate(d_f_start, d_f_end, yf),
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
    # A (administrative) → stable-margin like V: today's rate + a*(fwd_shocked - base_fwd)
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
    if mask_admin.any() and "product_code" in df.columns:                 # A: today's rate + beta x index move
        # anchored like V (see eve_calc_objects): base CFs carry today's administered rate
        df.loc[mask_admin, "eff_rate_shocked"] = _apply_rt_limits(
            df.loc[mask_admin, "fwd_rt_shocked"], df.loc[mask_admin, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            base_eff_rate = _contracted_rt.loc[mask_admin],
            base_fwd_rt   = df.loc[mask_admin, "fwd_rt"].fillna(df.loc[mask_admin, "fwd_rt_shocked"]),
        )

    # ── NII components ─────────────────────────────────────────────────────────
    df["sign"] = np.where(df["bs_side"] == "A", 1.0, -1.0)

    # Interest on existing positions at shocked client rate
    df["nii_interest"] = (
        df["outstanding_bal"].fillna(0.0)
        * df["eff_rate_shocked"]
        * df["cf_yf"].fillna(0.0)
        * df["accr_frac"]
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
            contract_margin=df.get("margin"),
        )
    else:
        renewal_rt = df["fwd_rt_shocked"].to_numpy()
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


def renewal_forward(
    df: pd.DataFrame,
    disc_df: pd.Series,
    disc_curve_map: dict[str, str],
    horizon_end: pd.Timestamp,
    floor_rate: float = 0.0,
) -> np.ndarray:
    """Market rate at which capital returned at cf_end_dt is renewed until the horizon end
    (constant balance sheet): labbank_common.curves.renewal_market_rate on the row's
    repricing period (column renew_months; absent = forward over [cf_end_dt, horizon_end]),
    floored at floor_rate. The ONE renewal definition used by the base and every shocked
    scenario (and by the cf.nii_* tables): only the curve passed in differs, so a 0 bp shock
    reproduces the base exactly. Rows renewing at or after the horizon get 0 (no renewal NII)."""
    out = np.zeros(len(df))
    months = df["renew_months"].to_numpy(dtype=float) if "renew_months" in df.columns else None
    for ccy in df["currency"].unique():
        cn = disc_curve_map.get(ccy)
        if cn is None:
            continue
        m = (df["currency"] == ccy).to_numpy()

        def _df_at(dates, cn=cn):
            idx = pd.MultiIndex.from_arrays([[cn] * len(dates), pd.to_datetime(pd.Series(dates)).to_numpy()],
                                            names=["curve_name", "node_date"])
            return disc_df.reindex(idx).to_numpy(dtype=float)

        out[m] = renewal_market_rate(df.loc[m, "cf_end_dt"], horizon_end,
                                     None if months is None else months[m], _df_at, floor_rate)
    return out


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
    nii_floor_rate: float = 0.0,
) -> pd.DataFrame:
    """Per-schedule NII for the base scenario: compute_nii_shocked_schedule run on the BASE
    curve. Base and shocked NII share one formula (interest anchoring and renewal); on the
    base curve every floating forward equals its own base forward, so interest is the
    contractual one, and renewal is renewal_forward() on the base curve.

    Returns DataFrame with: scenario_id + _SCHED_GROUP_COLS + nii_interest/renewal/total
    """
    if base_disc_df is None or disc_curve_map is None:
        raise ValueError("compute_nii_base_schedule needs the base curve (base_disc_df, disc_curve_map)")
    return compute_nii_shocked_schedule(
        beh_df, base_disc_df, disc_curve_map, report_date, scenario_id, horizon_yf, nii_floor_rate,
        caps_map=caps_map, floors_map=floors_map, coeff_a_map=coeff_a_map, coeff_b_map=coeff_b_map,
        base_disc_df=base_disc_df,
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
    base_disc_df: "pd.Series | None" = None,
    return_rows: bool = False,
) -> pd.DataFrame:
    """Per-schedule NII under a shocked rate scenario.

    return_rows: return the per-cash-flow rows (eff_rate_shocked, nii_interest, nii_renewal ...)
    instead of the schedule aggregation -- the cf.nii_* tables take their rates from here so they
    match irrbb.nii_results.

    base_disc_df: the base curve. Floating / administered interest is anchored as
    contracted rate + beta x (shocked forward - BASE forward), both forwards derived from
    the curves the same way (without it the stored fwd_rt stands in for the base forward).
    Renewal: renewal_forward() on the scenario curve -- one formula for base and every shock.

    For variable products: eff_rate_shocked = max(floor, min(cap, fwd_rt_shocked)).
    CAs with client_cap=0.001 (0.1%) stay capped regardless of shock size.

    Returns
    -------
    DataFrame with: scenario_id + _SCHED_GROUP_COLS + nii_interest/renewal/total
    """
    import numpy as np

    horizon_end = report_date + pd.Timedelta(days=round(horizon_yf * 365.25))
    df = beh_df[beh_df["cf_start_dt"] < horizon_end].copy()      # coupons overlapping the year
    if "accr_frac" not in df.columns:
        fe.add_horizon_accrual(df, report_date, horizon_end)

    # one index forward per FIXING BLOCK for floating rows (fixing_engine.block_periods)
    _rt = df.get("rate_type", pd.Series("V", index=df.index))
    _ps, _pe, _pyf = fe.block_periods(df, mask=(_rt == "V"))

    def _period_fwd(curve: pd.Series) -> np.ndarray:
        """Index forward over each row's fixing block (else its coupon period) on `curve`."""
        out = np.full(len(df), np.nan)
        for ccy, grp in df.groupby("currency"):
            cn = disc_curve_map.get(ccy)
            if cn is None:
                continue
            pos = df.index.get_indexer(grp.index)
            i_s = pd.MultiIndex.from_arrays([[cn] * len(grp), _ps.loc[grp.index].to_numpy()],
                                            names=["curve_name", "node_date"])
            i_e = pd.MultiIndex.from_arrays([[cn] * len(grp), _pe.loc[grp.index].to_numpy()],
                                            names=["curve_name", "node_date"])
            d_s = curve.reindex(i_s).to_numpy(dtype=float)
            d_e = curve.reindex(i_e).to_numpy(dtype=float)
            yf = _pyf.loc[grp.index].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[pos] = np.where((d_e > 0) & (yf > 0) & ~np.isnan(d_e), forward_rate(d_s, d_e, yf), np.nan)
        # a rate already set (period / fixing on or before the report date) needs no curve --
        # it is replaced by the stored fwd_rt below; any OTHER gap is missing market data,
        # which used to become a silent 0% projected rate
        rd = pd.Timestamp(report_date)
        fixed = (_ps < rd) | (df["cf_start_dt"] < rd)
        if "fixing_dt" in df.columns:
            fixed = fixed | pd.to_datetime(df["fixing_dt"]).le(rd).fillna(False)
        gap = np.isnan(out) & ~fixed.to_numpy()
        if gap.any():
            bad = df.loc[gap, ["currency", "cf_start_dt", "cf_end_dt"]]
            raise ValueError(f"NII {scenario_id}: {int(gap.sum())} floating cash flow(s) without a projected "
                             f"rate (currencies {sorted(bad['currency'].unique())}, up to "
                             f"{bad['cf_end_dt'].max()}) -- curve missing or too short")
        return np.maximum(nii_floor_rate, np.nan_to_num(out, nan=0.0))

    df["fwd_rt_shocked"] = _period_fwd(shocked_disc_df)
    if base_disc_df is not None:
        base_fwd_ref = pd.Series(_period_fwd(base_disc_df), index=df.index)
    elif "fwd_rt" in df.columns:
        base_fwd_ref = df["fwd_rt"].fillna(0.0).copy()
    else:
        base_fwd_ref = pd.Series(0.0, index=df.index)

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
            base_fwd_ref.loc[_before_curve] = df.loc[_before_curve, "fwd_rt"].fillna(0.0)

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
            base_fwd_ref.loc[locked] = df.loc[locked, "fwd_rt"].fillna(0.0)   # locked: no move

    # ── Contracted client rate (full rate including margin b) ─────────────────
    # Same approach as eve_calc_objects.compute_shocked_cf_detail: back-compute
    # from int_pmt so that base_eff_rate in the stable-margin formula carries b.
    _denom_s = (
        df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    ).replace(0.0, float("nan"))
    _contracted_rt_s = (df.get("int_pmt", pd.Series(0.0, index=df.index)).fillna(0.0)
                        / _denom_s).fillna(0.0)

    # eff_rate_shocked: F=contracted (unchanged), V and A = stable-margin on locked rate
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
            base_fwd_rt   = base_fwd_ref.loc[mask_var],
        )
    elif mask_var.any():
        df.loc[mask_var, "eff_rate_shocked"] = fwd_rt_for_interest.loc[mask_var]
    if mask_admin.any() and "product_code" in df.columns:        # A: today's rate + beta x index move
        # anchored like V (see eve_calc_objects): base CFs carry today's administered rate
        df.loc[mask_admin, "eff_rate_shocked"] = _apply_rt_limits(
            fwd_rt_for_interest.loc[mask_admin], df.loc[mask_admin, "product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            base_eff_rate = _contracted_rt_s.loc[mask_admin],
            base_fwd_rt   = base_fwd_ref.loc[mask_admin],
        )

    df["sign"] = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["nii_interest"] = (
        df["outstanding_bal"].fillna(0.0)
        * df["eff_rate_shocked"]
        * df["cf_yf"].fillna(0.0)
        * df["accr_frac"]                       # only the part earned inside the horizon
        * df["sign"]
    )
    df["remain_yf"]     = ((horizon_end - df["cf_end_dt"]).dt.days / 365.0).clip(lower=0.0)
    df["total_capital"] = df["capital_pmt"].fillna(0.0) + df["prepayment_pmt"].fillna(0.0)

    # ── Renewal: the shared definition (forward from cf_end_dt to the horizon) ──
    fwd_rt_for_renewal = pd.Series(
        renewal_forward(df, shocked_disc_df, disc_curve_map, horizon_end, nii_floor_rate), index=df.index)

    # Renewal uses fwd_rt_for_renewal (shocked market rate, shock-curve derived)
    # so that maturing capital renews at current shocked conditions.
    if "product_code" in df.columns:
        renewal_rt_shocked = _apply_rt_limits(
            fwd_rt_for_renewal, df["product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            contract_margin=df.get("margin"),
        )
    else:
        renewal_rt_shocked = fwd_rt_for_renewal.to_numpy()
    # NMD money that leaves only because of the scenario (irrbb_behaviour.nmd_migration) stays in
    # the bank as the product clients move to: renewed at that product's rate and tenor
    renew_cap = df["total_capital"]
    df["nii_migration"] = 0.0
    if "migr_capital" in df.columns and df["migr_capital"].fillna(0.0).ne(0.0).any():
        mm = df["migr_capital"].fillna(0.0).ne(0.0)
        mig = df.loc[mm, ["currency", "cf_end_dt"]].assign(renew_months=df.loc[mm, "migr_months"].to_numpy())
        mig_fwd = renewal_forward(mig, shocked_disc_df, disc_curve_map, horizon_end, nii_floor_rate)
        mig_rt = _apply_rt_limits(mig_fwd, df.loc[mm, "migr_product"], caps_map, floors_map,
                                  coeff_a_map, coeff_b_map)
        renew_cap = renew_cap - df["migr_capital"].fillna(0.0)
        df.loc[mm, "nii_migration"] = (df.loc[mm, "migr_capital"] * mig_rt
                                       * df.loc[mm, "remain_yf"] * df.loc[mm, "sign"])
    df["nii_renewal"]   = (
        renew_cap * renewal_rt_shocked * df["remain_yf"] * df["sign"] + df["nii_migration"]
    )
    df["nii_total"] = df["nii_interest"] + df["nii_renewal"]

    if return_rows:
        return df.assign(scenario_id=scenario_id)
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
    base_disc_df: "pd.Series | None" = None,
    disc_curve_map: "dict | None" = None,
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
    # renewal: the shared definition (renewal_forward) when the base curve is given
    if base_disc_df is not None and disc_curve_map is not None:
        horizon_end_r = report_date + pd.Timedelta(days=round(horizon_yf * 365.25))
        renewal_base = pd.Series(renewal_forward(df, base_disc_df, disc_curve_map, horizon_end_r, 0.0),
                                 index=df.index)
    else:
        renewal_base = df["fwd_rt"].fillna(0.0)
    if "product_code" in df.columns:
        renewal_rt = _apply_rt_limits(
            renewal_base, df["product_code"],
            caps_map, floors_map, coeff_a_map, coeff_b_map,
            contract_margin=df.get("margin"),
        )
    else:
        renewal_rt = renewal_base.to_numpy()

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
