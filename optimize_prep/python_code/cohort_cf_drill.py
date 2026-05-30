"""cohort_cf_drill.py
====================
Write opt_prep.cohort_cf_compare: fast-model CFs vs exact cf.products CFs,
row-by-row, so you can pick any cohort in SSMS and inspect date-level detail.

Three row types (cf_type column):
  'fast_m'  12 monthly rows from rate_matrix × cohort_outstanding_m (NII)
  'fast_q'  quarterly rows from cf_fixed_int_frac × cohort_disc_q   (EVE)
  'exact'   one row per behavioral CF from cf.products               (base)

Key comparisons the table enables
-----------------------------------
  outstanding & interest_pmt  → fast_m vs exact: is the rate_matrix rate right?
  nii_renewal                 → exact only (fast_m has 0): shows the missing
                                renewal component that causes large NII errors
  pv_total (fast_q vs exact)  → quarterly bucket approximation vs per-CF PV
  margin = effective_rate − fwd_rt → why floating cohorts drift from par

Usage (CLI filters — all optional, all cohort products by default)
------------------------------------------------------------------
  python cohort_cf_drill.py                           # all cohorts
  python cohort_cf_drill.py 1000                      # product 1000 only
  python cohort_cf_drill.py 1000 A PLN                # product / side / ccy
  python cohort_cf_drill.py 1000 A PLN 2023 6         # single cohort
  python cohort_cf_drill.py 2100 L PLN . . base,par_up  # specific scenarios

After running, query in SSMS:
  SELECT * FROM opt_prep.cohort_cf_compare
  WHERE cohort_id = '1000_A_PLN_2023_06'
  ORDER BY cf_type, scenario_id, bucket_idx, cf_end_dt
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import sql_setup as opt_sql
from bs_vector import BalanceSheetParams, CurveTensors, CohortRates

engine    = opt_sql.engine
REPORT_DATE  = pd.to_datetime("2024-12-31")
HORIZON_END  = REPORT_DATE + pd.Timedelta(days=365)
NPZ_PARAMS   = os.path.join(BASE_DIR, "..", "output", "product_params.npz")
NPZ_CURVES   = os.path.join(BASE_DIR, "..", "output", "curve_tensors.npz")

_COHORT_CODES = frozenset({"1000","1100","2000","2100","4100",
                            "3000","3100","7060","7900","5000"})

_ALL_SCENARIOS = ["base", "par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]


# ─────────────────────────────────────────────────────────────────────────────
# Exact CFs from cf.products
# ─────────────────────────────────────────────────────────────────────────────

def _load_exact_cfs(
    product_code: str | None,
    bs_side:      str | None,
    currency:     str | None,
    start_year:   int | None,
    start_month:  int | None,
) -> pd.DataFrame:
    """Return CF rows from cf.products for cohorts matching the filter.

    Joins sched.{loans,deposits,fin_inst} to get start_date/rate_type.
    Returns base-scenario CFs only (beh_* columns + fwd_rt + d_f).
    """
    _c_in = ", ".join(f"'{c}'" for c in sorted(_COHORT_CODES))
    pc_filter    = f"AND CAST(s.product_code AS VARCHAR(4)) = '{product_code}'" if product_code else ""
    side_filter  = f"AND s.bs_side = '{bs_side}'"    if bs_side    else ""
    ccy_filter   = f"AND s.currency = '{currency}'"  if currency   else ""
    sy_filter    = f"AND YEAR(s.start_date) = {start_year}"   if start_year   else ""
    sm_filter    = f"AND MONTH(s.start_date) = {start_month}" if start_month  else ""

    q = text(f"""
        WITH sched_base AS (
            SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id,
                   CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, start_date,
                   ISNULL(rate_type, 'V') AS rate_type
            FROM sched.loans
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_c_in})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)),
                   CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date,
                   ISNULL(rate_type, 'V')
            FROM sched.deposits
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_c_in})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)),
                   CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date,
                   ISNULL(rate_type, 'F')
            FROM sched.fin_inst
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_c_in})
        )
        SELECT
            s.product_code,
            s.bs_side,
            s.currency,
            YEAR(s.start_date)                       AS start_year,
            MONTH(s.start_date)                      AS start_month,
            s.rate_type,
            p.fixing_dt,
            p.cf_start_dt,
            p.cf_end_dt,
            CAST(p.cf_yf AS FLOAT)                   AS cf_yf,
            CAST(p.fwd_rt AS FLOAT)                  AS fwd_rt,
            CAST(p.d_f AS FLOAT)                     AS d_f,
            CAST(ISNULL(p.beh_outstanding,   0) AS FLOAT) AS outstanding,
            CAST(ISNULL(p.beh_capital_pmt,   0) AS FLOAT) AS capital_pmt,
            CAST(ISNULL(p.prepayment_pmt,    0) AS FLOAT) AS prepayment_pmt,
            CAST(ISNULL(p.beh_interest_pmt,  0) AS FLOAT) AS interest_pmt
        FROM cf.products p
        JOIN sched_base s
          ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
        WHERE p.cf_end_dt > :rd
          AND COALESCE(p.beh_total_pmt, 0) <> 0
          {pc_filter} {side_filter} {ccy_filter} {sy_filter} {sm_filter}
    """)
    df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
    df["cf_start_dt"] = pd.to_datetime(df["cf_start_dt"])
    df["cf_end_dt"]   = pd.to_datetime(df["cf_end_dt"])
    df["product_code"] = df["product_code"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build exact rows (one per CF, base scenario)
# ─────────────────────────────────────────────────────────────────────────────

def _exact_rows(
    cf_df:    pd.DataFrame,
    cohort_id: str,
    coeff_b:  float,
    coeff_a:  float,
    floor_:   float,
    cap_:     float,
    sign:     float,
) -> list[dict]:
    """Compute NII + EVE columns for each exact CF row."""
    rows = []
    for _, r in cf_df.iterrows():
        outstanding = float(r["outstanding"])
        capital     = float(r["capital_pmt"]) + float(r["prepayment_pmt"])
        interest    = float(r["interest_pmt"])
        cf_yf       = float(r["cf_yf"]) if r["cf_yf"] else 0.0
        fwd_rt      = float(r["fwd_rt"]) if pd.notna(r["fwd_rt"]) else 0.0
        d_f         = float(r["d_f"])    if pd.notna(r["d_f"])    else 1.0

        # effective contracted client rate back-computed from int_pmt
        denom = outstanding * cf_yf
        eff_rate = interest / denom if denom > 1.0 else fwd_rt
        margin   = eff_rate - fwd_rt

        yf_from_report = (r["cf_end_dt"] - REPORT_DATE).days / 365.0

        # NII components (base scenario)
        nii_interest = interest * sign

        remain_yf = max(0.0, (HORIZON_END - r["cf_end_dt"]).days / 365.0)
        # renewal rate = a × fwd_rt + b (new-money formula, same as exact pipeline)
        renewal_rt = float(np.clip(coeff_a * fwd_rt + coeff_b, floor_, cap_))
        nii_renewal = capital * renewal_rt * remain_yf * sign
        nii_total   = nii_interest + nii_renewal

        # EVE components (base d_f)
        pv_capital  = capital  * d_f * sign
        pv_interest = interest * d_f * sign
        pv_total    = pv_capital + pv_interest

        rows.append({
            "cohort_id":      cohort_id,
            "cf_type":        "exact",
            "scenario_id":    "base",
            "bucket_idx":     None,
            "cf_start_dt":    r["cf_start_dt"].date() if pd.notna(r["cf_start_dt"]) else None,
            "cf_end_dt":      r["cf_end_dt"].date(),
            "yf_from_report": round(yf_from_report, 6),
            "outstanding":    round(outstanding,  2),
            "capital_pmt":    round(capital,      2),
            "interest_pmt":   round(interest,     2),
            "effective_rate": round(eff_rate,     8),
            "fwd_rt":         round(fwd_rt,       8),
            "margin":         round(margin,       8),
            "nii_interest":   round(nii_interest, 2),
            "remain_yf":      round(remain_yf,    6),
            "nii_renewal":    round(nii_renewal,  2),
            "nii_total":      round(nii_total,    2),
            "disc_factor":    round(d_f,          8),
            "pv_capital":     round(pv_capital,   2),
            "pv_interest":    round(pv_interest,  2),
            "pv_total":       round(pv_total,     2),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Build fast monthly rows (NII, one per calendar month × scenario)
# ─────────────────────────────────────────────────────────────────────────────

def _fast_monthly_rows(
    cohort_id:     str,
    balance:       float,
    sign:          float,
    coeff_b:       float,
    outstanding_m: np.ndarray,   # (12,) fractions
    capital_m:     np.ndarray,   # (12,) fractions
    rate_matrix_i: np.ndarray,   # (12, n_scen)
    disc_m:        np.ndarray,   # (n_scen, 360) disc factors
    cr_scenario_ids: list[str],
    scenarios: list[str],
) -> list[dict]:
    rows = []
    for m in range(12):
        month_start = REPORT_DATE + pd.DateOffset(months=m)
        month_end   = REPORT_DATE + pd.DateOffset(months=m + 1)
        remain_yf   = max(0.0, (HORIZON_END - month_end).days / 365.0)
        yf_mid      = (m + 0.5) / 12.0

        outstanding_pln = balance * outstanding_m[m]
        capital_pln     = balance * capital_m[m]
        disc_idx        = min(m, disc_m.shape[1] - 1)

        for s_idx, scen in enumerate(cr_scenario_ids):
            if scen not in scenarios:
                continue
            rate    = float(rate_matrix_i[m, s_idx])
            disc    = float(disc_m[s_idx, disc_idx]) if disc_m.shape[0] > s_idx else 1.0

            # NII: interest on outstanding — NO renewal (fast model limitation)
            nii_int = outstanding_pln * sign * rate / 12.0
            nii_ren = 0.0   # fast model has no renewal component
            # approximate fwd_rt = rate - coeff_b (before floor/cap)
            fwd_approx = rate - coeff_b

            rows.append({
                "cohort_id":      cohort_id,
                "cf_type":        "fast_m",
                "scenario_id":    scen,
                "bucket_idx":     m,
                "cf_start_dt":    month_start.date(),
                "cf_end_dt":      month_end.date(),
                "yf_from_report": round(yf_mid, 6),
                "outstanding":    round(outstanding_pln, 2),
                "capital_pmt":    round(capital_pln,     2),
                "interest_pmt":   round(nii_int,         2),  # signed already
                "effective_rate": round(rate,            8),
                "fwd_rt":         round(fwd_approx,      8),
                "margin":         round(coeff_b,         8),
                "nii_interest":   round(nii_int,         2),
                "remain_yf":      round(remain_yf,       6),
                "nii_renewal":    0.0,
                "nii_total":      round(nii_int,         2),
                "disc_factor":    round(disc,            8),
                "pv_capital":     round(capital_pln * disc * sign, 2),
                "pv_interest":    None,
                "pv_total":       None,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Build fast quarterly rows (EVE, one per quarter bucket × scenario)
# ─────────────────────────────────────────────────────────────────────────────

def _fast_quarterly_rows(
    cohort_id:       str,
    balance:         float,
    sign:            float,
    cf_cap_frac:     np.ndarray,   # (max_q,) capital fraction
    cf_tot_frac:     np.ndarray,   # (max_q,) total CF fraction (cap+int for fixed, cap for float)
    cf_yf:           np.ndarray,   # (max_q,) year-fractions
    cf_n_q:          int,
    cohort_disc_q:   np.ndarray,   # (max_q, n_scen)
    cr_scenario_ids: list[str],
    scenarios: list[str],
) -> list[dict]:
    rows = []
    for q in range(cf_n_q):
        yf     = float(cf_yf[q])
        cf_dt  = REPORT_DATE + pd.Timedelta(days=round(yf * 365.25))
        cap_pln = balance * float(cf_cap_frac[q])
        tot_pln = balance * float(cf_tot_frac[q])
        int_pln = tot_pln - cap_pln     # 0 for floating cohorts

        for s_idx, scen in enumerate(cr_scenario_ids):
            if scen not in scenarios:
                continue
            disc     = float(cohort_disc_q[q, s_idx])
            pv_cap   = cap_pln * disc * sign
            pv_int   = int_pln * disc * sign
            pv_total = tot_pln * disc * sign

            rows.append({
                "cohort_id":      cohort_id,
                "cf_type":        "fast_q",
                "scenario_id":    scen,
                "bucket_idx":     q,
                "cf_start_dt":    None,
                "cf_end_dt":      cf_dt.date(),
                "yf_from_report": round(yf, 6),
                "outstanding":    None,
                "capital_pmt":    round(cap_pln,  2),
                "interest_pmt":   round(int_pln,  2),
                "effective_rate": None,
                "fwd_rt":         None,
                "margin":         None,
                "nii_interest":   None,
                "remain_yf":      None,
                "nii_renewal":    None,
                "nii_total":      None,
                "disc_factor":    round(disc,     8),
                "pv_capital":     round(pv_cap,   2),
                "pv_interest":    round(pv_int,   2),
                "pv_total":       round(pv_total, 2),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_cohort_cf_drill(
    product_code: str | None  = None,
    bs_side:      str | None  = None,
    currency:     str | None  = None,
    start_year:   int | None  = None,
    start_month:  int | None  = None,
    scenarios:    list[str]   | None = None,
    reset_table:  bool = True,
) -> None:
    if scenarios is None:
        scenarios = _ALL_SCENARIOS

    print("Loading NPZ parameters...")
    params = BalanceSheetParams.load(NPZ_PARAMS)
    cr     = CohortRates.load(NPZ_PARAMS)
    ct     = CurveTensors.load(NPZ_CURVES)

    cr_scen_ids = list(cr.rate_scenario_ids)
    ct_scen_ids = list(ct.scenario_ids)
    ccy_list    = list(ct.currencies)

    print("Loading exact CFs from cf.products...")
    exact_df = _load_exact_cfs(product_code, bs_side, currency, start_year, start_month)
    if exact_df.empty:
        print("  No exact CFs found matching the filter.")
        return
    print(f"  {len(exact_df):,} CF rows for "
          f"{exact_df.groupby(['product_code','bs_side','currency','start_year','start_month']).ngroups} cohorts")

    # Build a fast-lookup dict: cohort_key → entry index in params arrays
    param_cohort_keys: dict[tuple, int] = {}
    for idx in range(len(params.product_code)):
        if not params.is_cohort[idx]:
            continue
        sy = int(params.start_year[idx])
        sm = int(params.start_month[idx])
        k  = (str(params.product_code[idx]), str(params.bs_side[idx]),
              str(params.currency[idx]), sy, sm)
        param_cohort_keys[k] = idx

    all_rows: list[dict] = []
    cohort_groups = exact_df.groupby(
        ["product_code", "bs_side", "currency", "start_year", "start_month"]
    )
    n_cohorts = len(cohort_groups)
    print(f"Processing {n_cohorts} cohorts...")

    for (pc, sid, ccy, sy, sm), cf_grp in cohort_groups:
        pc = str(pc); sid = str(sid); ccy = str(ccy)
        sy = int(sy);  sm = int(sm)

        cohort_key = (pc, sid, ccy, sy, sm)
        cohort_id  = f"{pc}_{sid}_{ccy}_{sy:04d}_{sm:02d}"
        rate_typ   = str(cf_grp["rate_type"].iloc[0]) if "rate_type" in cf_grp.columns else "V"

        # ── Identify cohort index in params ──────────────────────────────────
        idx = param_cohort_keys.get(cohort_key)
        if idx is None:
            print(f"  [skip] {cohort_id}: not found in product_params.npz")
            continue

        balance  = float(params.balance_arr[idx])
        sign_i   = float(params.sign[idx])
        coeff_a  = float(params.coeff_a[idx])
        coeff_b  = float(params.coeff_b[idx])
        floor_   = float(params.client_floor[idx]) if np.isfinite(params.client_floor[idx]) else -np.inf
        cap_     = float(params.client_cap[idx])   if np.isfinite(params.client_cap[idx])   else  np.inf

        if balance <= 0:
            print(f"  [skip] {cohort_id}: zero balance")
            continue

        # ── Disc factor grid for this currency (n_scen, 360) ─────────────────
        c_idx = ccy_list.index(ccy) if ccy in ccy_list else 0
        # disc_m[s, m] = disc factor at end of month m+1 for scenario s in ct order
        disc_m_ct = ct.disc_factors[:, c_idx, :]   # (n_ct_scen, 360)

        # Remap ct scenarios → cr scenarios for fast_monthly
        disc_m_cr = np.ones((len(cr_scen_ids), 360), dtype=float)
        for cr_s, scen_label in enumerate(cr_scen_ids):
            if scen_label in ct_scen_ids:
                disc_m_cr[cr_s] = disc_m_ct[ct_scen_ids.index(scen_label)]

        # ── Exact rows ────────────────────────────────────────────────────────
        exact_rows = _exact_rows(cf_grp, cohort_id, coeff_b, coeff_a, floor_, cap_, sign_i)
        for r in exact_rows:
            r.update({
                "product_code": pc, "bs_side": sid,
                "currency": ccy, "rate_type": rate_typ,
            })
        all_rows.extend(exact_rows)

        # ── Fast monthly rows ─────────────────────────────────────────────────
        fast_m_rows = _fast_monthly_rows(
            cohort_id,
            balance, sign_i, coeff_b,
            params.cohort_outstanding_m[idx],
            params.cohort_capital_m[idx],
            cr.rate_matrix[idx],          # (12, n_cr_scen)
            disc_m_cr,
            cr_scen_ids,
            scenarios,
        )
        for r in fast_m_rows:
            r.update({
                "product_code": pc, "bs_side": sid,
                "currency": ccy, "rate_type": rate_typ,
            })
        all_rows.extend(fast_m_rows)

        # ── Fast quarterly EVE rows ───────────────────────────────────────────
        n_q = int(cr.cf_n_q[idx])
        if n_q > 0:
            fast_q_rows = _fast_quarterly_rows(
                cohort_id,
                balance, sign_i,
                cr.cf_capital_frac[idx],
                cr.cf_fixed_int_frac[idx],
                cr.cf_yf[idx],
                n_q,
                cr.cohort_disc_q[idx],    # (max_q, n_cr_scen)
                cr_scen_ids,
                scenarios,
            )
            for r in fast_q_rows:
                r.update({
                    "product_code": pc, "bs_side": sid,
                    "currency": ccy, "rate_type": rate_typ,
                })
            all_rows.extend(fast_q_rows)

    if not all_rows:
        print("No rows generated — nothing to write.")
        return

    df_out = pd.DataFrame(all_rows)
    df_out["report_date"] = REPORT_DATE

    print(f"Total rows: {len(df_out):,}  "
          f"(exact={( df_out['cf_type']=='exact').sum():,}  "
          f"fast_m={( df_out['cf_type']=='fast_m').sum():,}  "
          f"fast_q={( df_out['cf_type']=='fast_q').sum():,})")

    if reset_table:
        print("Creating opt_prep.cohort_cf_compare...")
        opt_sql.reset_cohort_cf_compare()

    print("Writing to SQL...")
    opt_sql.write_cohort_cf_compare(df_out)
    print(f"Done — {len(df_out):,} rows written to opt_prep.cohort_cf_compare")
    print()
    print("Query examples in SSMS:")
    print("  -- NII: exact vs fast per month for one cohort")
    print("  SELECT cf_type, scenario_id, cf_end_dt, outstanding, effective_rate,")
    print("         nii_interest, nii_renewal, nii_total")
    print("  FROM opt_prep.cohort_cf_compare")
    print("  WHERE cohort_id = '<cohort_id>' AND scenario_id IN ('base','par_up')")
    print("  ORDER BY cf_type, scenario_id, cf_end_dt;")
    print()
    print("  -- EVE: quarterly PV buckets")
    print("  SELECT cf_type, scenario_id, cf_end_dt, capital_pmt, disc_factor, pv_total")
    print("  FROM opt_prep.cohort_cf_compare")
    print("  WHERE cohort_id = '<cohort_id>' AND cf_type IN ('fast_q','exact')")
    print("  ORDER BY cf_type, scenario_id, cf_end_dt;")


if __name__ == "__main__":
    args = sys.argv[1:]

    def _get(i, cast=str, default=None):
        val = args[i].strip() if i < len(args) else None
        if val is None or val == ".":
            return default
        return cast(val)

    pc_arg  = _get(0)
    sid_arg = _get(1)
    ccy_arg = _get(2)
    sy_arg  = _get(3, int)
    sm_arg  = _get(4, int)
    scen_raw = _get(5)
    scen_arg = [s.strip() for s in scen_raw.split(",")] if scen_raw else None

    run_cohort_cf_drill(
        product_code = pc_arg,
        bs_side      = sid_arg,
        currency     = ccy_arg,
        start_year   = sy_arg,
        start_month  = sm_arg,
        scenarios    = scen_arg,
    )
