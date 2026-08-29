"""nii_monthly_drill.py
======================
Monthly cashflow drill-down for NII fast vs exact comparison per cohort.

Fast model monthly NII for cohort i, month m, scenario s:
    nii[i, m, s] = balance[i] * sign[i] * outstanding_m[i, m] * rate_matrix[i, m, s] / 12

Exact monthly interest income comes from cf.products (beh_interest_pmt),
grouped to the same cohort key (product_code, start_year, start_month).

Outputs
-------
  optimize_prep/output/nii_monthly_drill.xlsx
    cohort_params      — one row per cohort: balance, t_first_m, repricing, coeff_a, profile
    fast_nii_interest  — fast interest NII per cohort (base), columns m01..m12
    fast_nii_renewal   — renewal NII per cohort (capital × new-biz rate × remain_yf)
    fast_nii_total     — interest + renewal combined
    fast_rates_base    — rate_matrix base values per cohort, columns m01..m12
    exact_nii_cohort   — exact monthly interest income from SQL, columns m01..m12
    compare_interest   — fast interest vs exact interest per cohort: total, abs_err, pct_err
    delta_nii_scen     — delta NII per cohort × scenario (interest + renewal)

Usage
-----
  python nii_monthly_drill.py                       # all cohort products
  python nii_monthly_drill.py 1000 2000             # filter by product_code
  python nii_monthly_drill.py 1000 --scen par_up    # one scenario for delta sheet
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, BASE_DIR)

from bs_vector import BalanceSheetParams, CohortRates
from nii_eve_cf_fast import _nii_renewal_matrix

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

REPORT_DATE = pd.to_datetime("2026-06-30")
NPZ_PARAMS  = os.path.join(BASE_DIR, "..", "output", "product_params.npz")
OUT_PATH    = os.path.join(BASE_DIR, "..", "output", "nii_monthly_drill.xlsx")

# SQL tables that carry cohort schedules
_SCHED_TABLES = ["sched.loans", "sched.deposits", "sched.fin_inst"]

MONTH_COLS = [f"m{m:02d}" for m in range(1, 13)]


# ─────────────────────────────────────────────────────────────────────────────
# SQL: exact monthly interest income per cohort
# ─────────────────────────────────────────────────────────────────────────────

def _load_exact_monthly_nii(product_codes: list[str]) -> pd.DataFrame:
    """Exact monthly interest income per (product_code, bs_side, currency,
    start_year, start_month, month_num).

    Two separate queries — no SQL JOIN — merged in pandas:
      Q1: sched tables  → schedule_id → (product_code, bs_side, currency, start_year, start_month)
      Q2: cf.products   → schedule_id, month_num, interest_pmt
    Each query is a simple filtered scan, executes in <1s.
    """
    codes_in = ", ".join(f"'{c}'" for c in product_codes)
    empty = pd.DataFrame(columns=[
        "product_code", "bs_side", "currency",
        "start_year", "start_month", "month_num", "interest_income",
    ])

    # ── Q1: schedule key map ──────────────────────────────────────────────────
    q_sched = text(f"""
        SELECT schedule_id,
               CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency,
               YEAR(start_date)  AS start_year,
               MONTH(start_date) AS start_month
        FROM sched.loans WITH (NOLOCK)
        WHERE CAST(product_code AS VARCHAR(4)) IN ({codes_in})
        UNION ALL
        SELECT schedule_id,
               CAST(product_code AS VARCHAR(4)),
               bs_side, currency,
               YEAR(start_date), MONTH(start_date)
        FROM sched.deposits WITH (NOLOCK)
        WHERE CAST(product_code AS VARCHAR(4)) IN ({codes_in})
        UNION ALL
        SELECT schedule_id,
               CAST(product_code AS VARCHAR(4)),
               bs_side, currency,
               YEAR(start_date), MONTH(start_date)
        FROM sched.fin_inst WITH (NOLOCK)
        WHERE CAST(product_code AS VARCHAR(4)) IN ({codes_in})
    """)

    # ── Q2: cf.products filtered rows ─────────────────────────────────────────
    q_cf = text(f"""
        SELECT schedule_id,
               DATEDIFF(month, :rd, cf_end_dt)  AS month_num,
               CAST(beh_interest_pmt AS FLOAT)  AS interest_pmt
        FROM cf.products WITH (NOLOCK)
        WHERE CAST(product_code AS VARCHAR(4)) IN ({codes_in})
          AND cf_end_dt > :rd
          AND cf_end_dt <= DATEADD(month, 12, :rd)
          AND COALESCE(beh_interest_pmt, 0.0) <> 0.0
    """)

    try:
        df_sched = pd.read_sql_query(q_sched, engine)
        df_sched["product_code"] = df_sched["product_code"].astype(str)
        df_sched = df_sched.drop_duplicates(subset=["schedule_id"])

        df_cf = pd.read_sql_query(q_cf, engine, params={"rd": REPORT_DATE})
        if df_cf.empty or df_sched.empty:
            return empty

        # merge in pandas — instant for these sizes
        merged = df_cf.merge(df_sched, on="schedule_id", how="inner")
        result = (
            merged
            .groupby(
                ["product_code", "bs_side", "currency",
                 "start_year", "start_month", "month_num"],
                as_index=False,
            )["interest_pmt"]
            .sum()
            .rename(columns={"interest_pmt": "interest_income"})
        )
        return result

    except Exception as e:
        print(f"  [warn] exact monthly NII query failed: {e}")
        return empty


def _pivot_monthly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot monthly data: rows = cohort, columns = m01..m12."""
    if df.empty:
        return df
    key = ["product_code", "bs_side", "currency", "start_year", "start_month"]
    df = df.copy()
    df["month_col"] = df["month_num"].apply(lambda m: f"m{int(m):02d}")
    pivoted = df.pivot_table(
        index=key, columns="month_col", values=value_col, aggfunc="sum", fill_value=0.0
    ).reset_index()
    pivoted.columns.name = None
    # ensure all 12 columns present
    for col in MONTH_COLS:
        if col not in pivoted.columns:
            pivoted[col] = 0.0
    return pivoted[key + MONTH_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Fast monthly NII from npz
# ─────────────────────────────────────────────────────────────────────────────

def _cohort_monthly_nii(
    params: BalanceSheetParams,
    cr: CohortRates,
    mask: np.ndarray,
) -> np.ndarray:
    """Fast monthly NII per cohort per scenario.

    Returns (n_masked, 12, n_scen) array.
    Month m, scenario s:
        nii[i, m, s] = balance[i] * sign[i] * outstanding_m[i, m] * rate_matrix[i, m, s] / 12
    """
    bal  = params.balance_arr[mask]
    sign = params.sign[mask]
    out  = params.cohort_outstanding_m[mask]       # (n, 12)
    rm   = cr.rate_matrix[mask]                    # (n, 12, n_scen)

    # (n, 12) weight * (n, 12, n_scen) rates -> (n, 12, n_scen)
    w = (bal * sign)[:, None] * out                # (n, 12)
    return w[:, :, None] * rm / 12.0               # (n, 12, n_scen)


def _build_cohort_key(params: BalanceSheetParams, mask: np.ndarray) -> pd.DataFrame:
    """DataFrame with cohort identification columns for masked subset."""
    idx = np.where(mask)[0]
    rows = []
    for i in idx:
        rows.append({
            "cohort_id":    str(params.cohort_id[i]),
            "product_code": str(params.product_code[i]),
            "bs_side":      str(params.bs_side[i]),
            "currency":     str(params.currency[i]),
            "start_year":   int(params.start_year[i]) if params.start_year[i] != -1 else -1,
            "start_month":  int(params.start_month[i]) if params.start_month[i] != -1 else -1,
        })
    return pd.DataFrame(rows)


def _fast_monthly_to_df(
    params: BalanceSheetParams,
    cr: CohortRates,
    mask: np.ndarray,
    scen_label: str,
    include_renewal: bool = False,
) -> pd.DataFrame:
    """Fast monthly NII for one scenario as a wide DataFrame.

    include_renewal=False  → interest component only (for compare vs exact)
    include_renewal=True   → interest + renewal (for total fast NII)
    """
    s_idx = cr.scenario_index(scen_label)
    nii3  = _cohort_monthly_nii(params, cr, mask)   # (n, 12, n_scen)
    nii12 = nii3[:, :, s_idx]                        # (n, 12)

    if include_renewal:
        bal = params.balance_arr[mask]
        # _nii_renewal_matrix returns (n, n_scen); distribute equally across months
        # proportional to capital_m weight — build per-month renewal
        from nii_eve_cf_fast import _REMAIN_YF
        sign = params.sign[mask]
        cap  = params.cohort_capital_m[mask]           # (n, 12)
        rrm  = cr.renewal_rate_matrix[mask, :, s_idx]  # (n, 12)
        remain = _REMAIN_YF[None, :]                   # (1, 12)
        ren12 = (bal * sign)[:, None] * cap * rrm * remain   # (n, 12)
        nii12 = nii12 + ren12

    key_df = _build_cohort_key(params, mask)
    for m in range(12):
        key_df[MONTH_COLS[m]] = nii12[:, m]
    return key_df


def _fast_rates_to_df(
    params: BalanceSheetParams,
    cr: CohortRates,
    mask: np.ndarray,
    scen_label: str,
) -> pd.DataFrame:
    """rate_matrix values for one scenario as a wide DataFrame."""
    s_idx  = cr.scenario_index(scen_label)
    rates  = cr.rate_matrix[mask, :, s_idx]           # (n, 12)

    key_df = _build_cohort_key(params, mask)
    for m in range(12):
        key_df[MONTH_COLS[m]] = (rates[:, m] * 100).round(4)  # in percent
    return key_df


def _cohort_params_df(params: BalanceSheetParams, mask: np.ndarray) -> pd.DataFrame:
    """One row per cohort with all key parameters + outstanding profile."""
    idx = np.where(mask)[0]
    rows = []
    for i in idx:
        r: dict = {
            "cohort_id":         str(params.cohort_id[i]),
            "product_code":      str(params.product_code[i]),
            "product_name":      str(params.product_name[i]),
            "bs_side":           str(params.bs_side[i]),
            "currency":          str(params.currency[i]),
            "start_year":        int(params.start_year[i]) if params.start_year[i] != -1 else -1,
            "start_month":       int(params.start_month[i]) if params.start_month[i] != -1 else -1,
            "balance":           round(params.balance_arr[i], 0),
            "rate_type":         str(params.rate_type[i]),
            "nii_unit_rate_pct": round(params.nii_unit_rate[i] * 100, 4),
            "coeff_a":           round(params.coeff_a[i], 4),
            "coeff_b_pct":       round(params.coeff_b[i] * 100, 4),
            "client_floor_pct":  (round(params.client_floor[i] * 100, 4)
                                  if np.isfinite(params.client_floor[i]) else None),
            "client_cap_pct":    (round(params.client_cap[i] * 100, 4)
                                  if np.isfinite(params.client_cap[i]) else None),
            "repricing_tenor_m": round(params.repricing_tenor_m[i], 1),
            "t_first_m":         round(params.cohort_t_first_m[i], 1),
            "coupon_rate_pct":   (round(params.coupon_rate[i] * 100, 4)
                                  if np.isfinite(params.coupon_rate[i]) else None),
            "locked_rate_pct":   round(params.cohort_locked_rate[i] * 100, 4),
        }
        for m in range(12):
            r[f"out_{MONTH_COLS[m]}"] = round(params.cohort_outstanding_m[i, m], 4)
        rows.append(r)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Compare fast vs exact at cohort level
# ─────────────────────────────────────────────────────────────────────────────

_COHORT_KEY  = ["product_code", "bs_side", "currency", "start_year", "start_month"]


def _compare_base(
    fast_wide: pd.DataFrame,
    exact_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Month-by-month comparison: fast − exact per cohort row.

    Also computes: total_fast, total_exact, abs_err, pct_err.
    """
    key = _COHORT_KEY
    merged = fast_wide.merge(
        exact_wide, on=key, how="outer", suffixes=("_fast", "_exact")
    ).fillna(0.0)

    for m_col in MONTH_COLS:
        f_col = f"{m_col}_fast"
        e_col = f"{m_col}_exact"
        if f_col in merged.columns and e_col in merged.columns:
            merged[f"{m_col}_diff"] = merged[f_col] - merged[e_col]

    # totals
    fast_month_cols  = [c for c in merged.columns if c.endswith("_fast")]
    exact_month_cols = [c for c in merged.columns if c.endswith("_exact")]
    merged["total_fast"]  = merged[fast_month_cols].sum(axis=1)
    merged["total_exact"] = merged[exact_month_cols].sum(axis=1)
    merged["abs_err"]     = merged["total_fast"] - merged["total_exact"]
    denom = merged["total_exact"].abs().replace(0.0, np.nan)
    merged["pct_err"] = (merged["abs_err"] / denom * 100).round(2)

    # sort by absolute error descending
    return merged.sort_values("abs_err", key=abs, ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Delta NII per cohort × scenario summary
# ─────────────────────────────────────────────────────────────────────────────

def _delta_nii_scen(
    params: BalanceSheetParams,
    cr: CohortRates,
    mask: np.ndarray,
) -> pd.DataFrame:
    """Delta NII per cohort per scenario (total over 12 months, interest + renewal)."""
    from nii_eve_cf_fast import _REMAIN_YF

    base_idx = cr.scenario_index("base")
    bal      = params.balance_arr[mask]
    sign     = params.sign[mask]
    cap      = params.cohort_capital_m[mask]           # (n, 12)

    int3  = _cohort_monthly_nii(params, cr, mask)      # (n, 12, n_scen) — interest
    remain = _REMAIN_YF[None, :, None]                 # (1, 12, 1)
    ren3  = ((bal * sign)[:, None, None]               # renewal: (n, 12, n_scen)
             * cap[:, :, None]
             * cr.renewal_rate_matrix[mask]
             * remain)
    nii_tot = (int3 + ren3).sum(axis=1)                # (n, n_scen)

    key_df = _build_cohort_key(params, mask)
    rows = []
    for i in range(len(key_df)):
        for s_idx, scen in enumerate(cr.rate_scenario_ids):
            scen = str(scen)
            if scen == "base":
                continue
            rows.append({
                **key_df.iloc[i].to_dict(),
                "scenario_id":   scen,
                "fast_base_nii": round(float(nii_tot[i, base_idx]), 0),
                "fast_scen_nii": round(float(nii_tot[i, s_idx]),    0),
                "fast_delta_nii": round(float(nii_tot[i, s_idx] - nii_tot[i, base_idx]), 0),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_drill(
    product_filter: list[str] | None = None,
    scen_filter:    list[str] | None = None,
) -> None:
    print("Loading product_params.npz...")
    params = BalanceSheetParams.load(NPZ_PARAMS)
    cr     = CohortRates.load(NPZ_PARAMS)

    # resolve product codes
    all_codes = sorted(set(str(c) for c in params.product_code[params.is_cohort]))
    if product_filter:
        codes = [c for c in product_filter if c in all_codes]
        if not codes:
            print(f"  [warn] None of {product_filter} are cohort product codes.")
            print(f"  Available: {all_codes}")
            return
    else:
        codes = all_codes
    print(f"  Products: {codes}")

    # mask: cohort rows for selected products
    mask = params.is_cohort & np.isin(params.product_code, codes)
    n_cohorts = int(mask.sum())
    print(f"  Cohort rows matched: {n_cohorts}")

    # scenarios
    all_scen = [str(s) for s in cr.rate_scenario_ids]
    if scen_filter:
        scens = [s for s in scen_filter if s in all_scen]
    else:
        scens = all_scen

    # ── Fast sheets ──────────────────────────────────────────────────────────
    print("Computing fast NII base (monthly per cohort)...")
    fast_base     = _fast_monthly_to_df(params, cr, mask, "base", include_renewal=False)
    fast_renewal  = _fast_monthly_to_df(params, cr, mask, "base", include_renewal=True)
    # build renewal-only by subtracting interest from total
    fast_renewal_only = fast_renewal.copy()
    for col in MONTH_COLS:
        fast_renewal_only[col] = fast_renewal[col] - fast_base[col]
    fast_rates_b  = _fast_rates_to_df(params, cr, mask, "base")
    cohort_params = _cohort_params_df(params, mask)

    # ── Exact from SQL ───────────────────────────────────────────────────────
    print("Querying exact monthly NII from cf.products...")
    exact_long = _load_exact_monthly_nii(codes)
    if not exact_long.empty:
        exact_wide = _pivot_monthly(exact_long, "interest_income")
        exact_wide.rename(columns={c: c for c in MONTH_COLS}, inplace=True)
    else:
        exact_wide = pd.DataFrame(columns=_COHORT_KEY + MONTH_COLS)

    # ── Compare ──────────────────────────────────────────────────────────────
    # Align keys: fast has cohort_id + start_year/month; exact matches by cohort key
    fast_base_cmp = fast_base[_COHORT_KEY + MONTH_COLS]
    compare = _compare_base(fast_base_cmp, exact_wide)

    # ── Delta NII per scenario ───────────────────────────────────────────────
    print("Computing delta NII per scenario per cohort...")
    delta_df = _delta_nii_scen(params, cr, mask)
    if scen_filter:
        delta_df = delta_df[delta_df["scenario_id"].isin(scen_filter)]

    # ── Write Excel ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Writing -> {OUT_PATH}")
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        cohort_params.to_excel(    writer, sheet_name="cohort_params",     index=False)
        fast_base.to_excel(        writer, sheet_name="fast_nii_interest",  index=False)
        fast_renewal_only.to_excel(writer, sheet_name="fast_nii_renewal",   index=False)
        fast_renewal.to_excel(     writer, sheet_name="fast_nii_total",     index=False)
        fast_rates_b.to_excel(     writer, sheet_name="fast_rates_base",    index=False)
        if not exact_wide.empty:
            exact_wide.to_excel(   writer, sheet_name="exact_nii_cohort",   index=False)
        compare.to_excel(          writer, sheet_name="compare_interest",   index=False)
        delta_df.to_excel(         writer, sheet_name="delta_nii_scen",     index=False)

    print("Done. Sheets:")
    print("  cohort_params      — balance, t_first_m, repricing, coeff_a, outstanding profile")
    print("  fast_nii_interest  — monthly fast interest NII per cohort, m01..m12 (PLN)")
    print("  fast_nii_renewal   — monthly renewal NII (capital × new-biz rate × remain_yf)")
    print("  fast_nii_total     — interest + renewal combined")
    print("  fast_rates_base    — monthly client rates from rate_matrix (%, base)")
    print("  exact_nii_cohort   — monthly exact interest income from cf.products (PLN)")
    print("  compare_interest   — fast interest vs exact: total, abs_err, pct_err")
    print("  delta_nii_scen     — delta NII per cohort per scenario (interest + renewal)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monthly NII drill per cohort")
    parser.add_argument("product_codes", nargs="*",
                        help="product_code(s) to include (default: all cohort products)")
    parser.add_argument("--scen", nargs="*", dest="scen",
                        help="scenario(s) to include in delta sheet (default: all)")
    args = parser.parse_args()
    run_drill(
        product_filter=args.product_codes or None,
        scen_filter=args.scen or None,
    )
