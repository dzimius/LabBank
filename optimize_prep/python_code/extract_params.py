"""extract_params.py  (v2 — monthly cohort aggregation)
=======================================================
Extracts per-optimizer-entry parameters into opt_prep SQL tables + product_params.npz.

Aggregation strategy
--------------------
* COHORT products (loans, mortgages, bonds, term deposits):
    One row per (product_code, bs_side, currency, start_year, start_month).
    NII and duration computed from cf.products joined to sched.* via schedule_id.
    EVE and scenario deltas derived from modified duration approximation.

* SINGLE-ROW products (current accounts, saving accounts, cash, equity):
    One row per (product_code, bs_side, currency).
    NII and EVE read from irrbb.nii_results / irrbb.eve_results (behavioural outputs).

Run after: balance_generate → balance_gen_add_data → cash_flow_calc
           → ir_derivatives → nii_calc_workflow → eve_calc_workflow
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, BASE_DIR)

import sql_setup as opt_sql

engine = opt_sql.engine

# ── configuration ──────────────────────────────────────────────────────────────
REPORT_DATE     = pd.to_datetime("2024-12-31")
TOTAL_ASSETS    = 10_000_000_000
HORIZON_DAYS    = 365
HORIZON_30D     = 30
PAR_SHOCK_RATE  = 0.02
HORIZON_END     = REPORT_DATE + pd.Timedelta(days=HORIZON_DAYS)
HORIZON_30D_END = REPORT_DATE + pd.Timedelta(days=HORIZON_30D)

BS_PATH       = os.path.join(ROOT_DIR, "balance_generate", "input_data", "bank_data.xlsx")
INTEREST_PATH = os.path.join(ROOT_DIR, "balance_generate", "input_data", "interest_rt.xlsx")
NPZ_OUT       = os.path.join(BASE_DIR, "..", "output", "product_params.npz")
EXCEL_OUT     = os.path.join(BASE_DIR, "..", "output", "params_inspection.xlsx")

SHOCKED_SCENARIO_IDS = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn", "own"]

# Monthly cohort products — have meaningful start_date and scheduled cash flows
COHORT_PRODUCT_CODES = frozenset({"1000", "1100", "2000", "2100", "4100",
                                   "3000", "3100", "7060", "7900", "5000"})
# Single-row products — behavioural models, no maturity, or equity
SINGLE_ROW_PRODUCT_CODES = frozenset({"3500", "6000", "8000", "6300",
                                       "5300", "5100", "5400"})
# IRS/derivatives — balance from schemat.ir_swaps.notional (separate table)
IRS_PRODUCT_CODES = frozenset({"0000"})

# Approximate Δrate per EBA scenario (used for duration-based NII/EVE sensitivity)
SCENARIO_RATE_SHOCKS: dict[str, float] = {
    "par_up":  0.020,
    "par_dn": -0.020,
    "steep":   0.010,
    "flat":   -0.010,
    "sr_up":   0.025,
    "sr_dn":  -0.025,
}

_C_IN = ", ".join(f"'{c}'" for c in sorted(COHORT_PRODUCT_CODES))
_S_IN = ", ".join(f"'{c}'" for c in sorted(SINGLE_ROW_PRODUCT_CODES))
_COHORT_KEY = ["product_code", "bs_side", "currency", "start_year", "start_month"]
_PROD_KEY   = ["product_code", "bs_side", "currency"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _freq_to_months(freq) -> float:
    if freq is None or (isinstance(freq, float) and np.isnan(freq)):
        return 12.0
    s = str(freq).strip().upper()
    if s.endswith("M"):
        try:
            return float(s[:-1])
        except ValueError:
            return 12.0
    if s.endswith("Y"):
        try:
            return float(s[:-1]) * 12.0
        except ValueError:
            return 12.0
    return 12.0


def _make_cohort_id(row: pd.Series) -> str:
    pc  = str(row["product_code"])
    sid = str(row["bs_side"])
    ccy = str(row["currency"])
    sy  = row.get("start_year")
    sm  = row.get("start_month")
    if pd.notna(sy) and pd.notna(sm):
        return f"{pc}_{sid}_{ccy}_{int(sy):04d}_{int(sm):02d}"
    return f"{pc}_{sid}_{ccy}"


def _try_query(q, params=None) -> pd.DataFrame:
    try:
        return pd.read_sql_query(q, engine, params=params or {})
    except Exception as e:
        print(f"  [warn] query skipped: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Static metadata
# ─────────────────────────────────────────────────────────────────────────────

def _load_bs_structure() -> pd.DataFrame:
    df = pd.read_excel(BS_PATH, sheet_name="bs_structure")
    df["sign"]         = df["bs_side"].map({"A": 1.0, "L": -1.0, "E": -1.0}).fillna(-1.0)
    df["full_pct"]     = df["bs_percentage"].fillna(0.0)
    df["product_code"] = df["product_code"].astype(str)
    # Fill NaN regulatory weights with 0.0 so _meta lookups never return NaN
    for _col in ["LCR", "ASF", "RSF", "haircut"]:
        if _col in df.columns:
            df[_col] = df[_col].fillna(0.0)
    # Compute hqla_factor = (1 - haircut) for HQLA assets, 0 elsewhere
    if "haircut" in df.columns and "hqla_class" in df.columns:
        df["hqla_factor"] = (1.0 - df["haircut"]).where(df["hqla_class"].notna(), 0.0)
    else:
        df["hqla_factor"] = 0.0
    return df


def _load_rate_coefficients() -> pd.DataFrame:
    df = pd.read_excel(INTEREST_PATH)
    df["product_code"] = df["product_code"].astype(str)
    return df.set_index("product_code")


# ─────────────────────────────────────────────────────────────────────────────
# Cohort product loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_cohort_balance() -> pd.DataFrame:
    """Balance from schemat tables grouped by (product_code, bs_side, currency, start_year, start_month)."""
    q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency,
               YEAR(start_date)  AS start_year,
               MONTH(start_date) AS start_month,
               SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
        FROM schemat.loans
        WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency, YEAR(start_date), MONTH(start_date)
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)), bs_side, currency,
               YEAR(start_date), MONTH(start_date),
               SUM(CAST(balance_amt AS FLOAT))
        FROM schemat.deposits
        WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency, YEAR(start_date), MONTH(start_date)
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)), bs_side, currency,
               YEAR(start_date), MONTH(start_date),
               SUM(CAST(balance_amt AS FLOAT))
        FROM schemat.financial_instruments
        WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency, YEAR(start_date), MONTH(start_date)
    """)
    df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df.groupby(_COHORT_KEY, as_index=False)["balance_amt"].sum()


def _load_cohort_cf_stats() -> pd.DataFrame:
    """CF-level stats per cohort from cf.products JOIN sched.* via schedule_id.

    Computes per cohort:
        nii_interest    — SUM(interest within 1Y horizon)
        total_outstanding, capital_30d, capital_1y  — for LCR/NSFR fractions
        dur_numer/dur_denom — for Macaulay duration computation
    """
    q = text(f"""
        SELECT
            s.product_code,
            s.bs_side,
            s.currency,
            YEAR(s.start_date)  AS start_year,
            MONTH(s.start_date) AS start_month,
            SUM(CAST(p.beh_outstanding  AS FLOAT) * p.cf_yf)           AS bal_yf,
            SUM(CASE WHEN p.cf_end_dt <= :he
                     THEN CAST(p.beh_interest_pmt AS FLOAT)
                     ELSE 0.0 END)                                      AS nii_interest,
            SUM(CAST(p.beh_outstanding  AS FLOAT))                      AS total_outstanding,
            SUM(CASE WHEN p.cf_end_dt <= :h30
                     THEN CAST(p.beh_capital_pmt AS FLOAT)
                          + ISNULL(CAST(p.prepayment_pmt AS FLOAT), 0.0)
                     ELSE 0.0 END)                                      AS capital_30d,
            SUM(CASE WHEN p.cf_end_dt <= :he
                     THEN CAST(p.beh_capital_pmt AS FLOAT)
                          + ISNULL(CAST(p.prepayment_pmt AS FLOAT), 0.0)
                     ELSE 0.0 END)                                      AS capital_1y,
            SUM((CAST(p.beh_interest_pmt AS FLOAT)
                 + CAST(p.beh_capital_pmt AS FLOAT)) * p.cf_yf)        AS dur_numer,
            SUM(CAST(p.beh_interest_pmt  AS FLOAT)
                + CAST(p.beh_capital_pmt AS FLOAT))                     AS dur_denom
        FROM cf.products p
        JOIN (
            SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id,
                   CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, start_date
            FROM sched.loans
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)),
                   CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date
            FROM sched.deposits
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)),
                   CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date
            FROM sched.fin_inst
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        ) s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
        WHERE p.cf_end_dt > :rd
          AND COALESCE(p.beh_total_pmt, 0) <> 0
        GROUP BY s.product_code, s.bs_side, s.currency,
                 YEAR(s.start_date), MONTH(s.start_date)
    """)
    df = pd.read_sql_query(
        q, engine,
        params={"rd": REPORT_DATE, "he": HORIZON_END, "h30": HORIZON_30D_END},
    )
    df["product_code"] = df["product_code"].astype(str)
    return df


def _load_cohort_cf_monthly() -> pd.DataFrame:
    """CF bucketed into calendar months (FLOOR(cf_yf × 12)) per cohort.

    Returns columns: _COHORT_KEY + [month_bucket_idx, capital_cf, interest_cf]
    Used to compute proper PV by discounting each monthly bucket with the
    interpolated base / shocked disc curve.
    """
    q = text(f"""
        WITH sched_key AS (
            SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id,
                   CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, start_date
            FROM sched.loans
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION
            SELECT CAST(schedule_id AS VARCHAR(8)),
                   CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date
            FROM sched.deposits
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION
            SELECT CAST(schedule_id AS VARCHAR(8)),
                   CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date
            FROM sched.fin_inst
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        )
        SELECT
            s.product_code,
            s.bs_side,
            s.currency,
            YEAR(s.start_date)   AS start_year,
            MONTH(s.start_date)  AS start_month,
            FLOOR(p.cf_yf * 12)  AS month_bucket_idx,
            SUM(CAST(p.beh_capital_pmt  AS FLOAT)
                + ISNULL(CAST(p.prepayment_pmt AS FLOAT), 0.0)) AS capital_cf,
            SUM(CAST(p.beh_interest_pmt AS FLOAT))               AS interest_cf
        FROM cf.products p
        JOIN sched_key s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
          AND CAST(p.product_code AS VARCHAR(4)) = s.product_code
        WHERE CAST(p.product_code AS VARCHAR(4)) IN ({_C_IN})
          AND p.cf_end_dt > :rd
          AND COALESCE(p.beh_total_pmt, 0) <> 0
        GROUP BY s.product_code, s.bs_side, s.currency,
                 YEAR(s.start_date), MONTH(s.start_date),
                 FLOOR(p.cf_yf * 12)
    """)
    df = _try_query(q, {"rd": REPORT_DATE})
    if not df.empty:
        df["product_code"]     = df["product_code"].astype(str)
        df["month_bucket_idx"] = df["month_bucket_idx"].astype(int)
    return df


def _analytical_cf_schedule(
    balance: float,
    annual_coupon: float,
    term_m: int,
    rate_type: str,
    t_first_m: int = 0,
    bucket_size_m: int = 3,
) -> pd.DataFrame:
    """Generate analytical amortisation CF schedule for one cohort.

    For fixed-rate (rate_type='F') products:
        Run-to-first-repricing approach:
        - monthly P+I from month 0 to t_first_m-1 (contracted coupon)
        - at month t_first_m: bullet payment for remaining principal
          (after repricing the loan is treated as floating → near par → 0 EVE sensitivity)
        cf_tot = capital + interest  (interest matters for PV of fixed-rate CFs)

    For floating-rate (rate_type='V') products:
        - full capital amortisation schedule over `term_m` months
          (interest excluded: floating rate reprices near par → minimal sensitivity)
        - cf_tot = capital only

    CFs are bucketed into `bucket_size_m`-month intervals.
    Returns DataFrame with columns [month_bucket_idx, capital_cf, interest_cf].
    """
    if term_m <= 0 or balance <= 0:
        return pd.DataFrame(columns=["month_bucket_idx", "capital_cf", "interest_cf"])

    r_m   = max(annual_coupon, 0.0) / 12.0   # monthly rate
    # For fixed-rate: horizon = time to first repricing (run-to-first-repricing).
    # For floating-rate: horizon = full remaining term (capital-only CFs).
    if rate_type == "F":
        horizon = int(max(t_first_m, 1))   # at least 1 month
        # Amortise over the FULL remaining term but only keep CFs up to repricing
        gen_m   = term_m
    else:
        horizon = term_m
        gen_m   = term_m

    # Monthly annuity payment (fully amortising over full term)
    if r_m > 1e-9:
        pmt = balance * r_m / (1.0 - (1.0 + r_m) ** (-gen_m))
    else:
        pmt = balance / gen_m   # zero-rate → equal capital repayment

    cap_arr  = np.zeros(gen_m, dtype=float)
    int_arr  = np.zeros(gen_m, dtype=float)
    rem_bal  = balance

    for m in range(gen_m):
        int_m       = rem_bal * r_m
        cap_m       = min(pmt - int_m, rem_bal)
        int_arr[m]  = int_m
        cap_arr[m]  = cap_m
        rem_bal    -= cap_m
        if rem_bal < 1.0:
            break

    if rate_type == "F":
        # Truncate at horizon; remaining balance becomes a bullet at month horizon
        h = min(horizon, gen_m)
        rem_at_h = float(balance - cap_arr[:h].sum())
        cap_arr_use = cap_arr[:h].copy()
        int_arr_use = int_arr[:h].copy()
        if rem_at_h > 1.0 and h <= gen_m:
            # Add bullet payment at the repricing month
            cap_arr_use = np.append(cap_arr_use, rem_at_h)
            int_arr_use = np.append(int_arr_use, 0.0)
            m_indices   = np.append(np.arange(h), h)
        else:
            m_indices = np.arange(len(cap_arr_use))
        effective_len = len(cap_arr_use)
    else:
        cap_arr_use = cap_arr
        int_arr_use = int_arr
        m_indices   = np.arange(gen_m)
        effective_len = gen_m

    # Bucket into bucket_size_m intervals (0-indexed from today)
    bucket_idx = m_indices // bucket_size_m
    rows = []
    for b in np.unique(bucket_idx):
        mask = bucket_idx == b
        rows.append({
            "month_bucket_idx": int(b * bucket_size_m),   # bucket start month
            "capital_cf":       float(cap_arr_use[mask].sum()),
            "interest_cf":      float(int_arr_use[mask].sum()) if rate_type == "F" else 0.0,
        })
    return pd.DataFrame(rows)


def _load_cohort_float_margins() -> dict:
    """Outstanding-weighted average margin per cohort for FLOATING products.

    margin is stored directly in cf.products as the per-CF origination spread.
    Returns dict: (product_code, bs_side, currency, start_year, start_month) -> wavg_margin
    Only cohort products are included; fixed-rate cohorts will simply not be looked up.
    """
    q = text(f"""
        SELECT
            CAST(s.product_code AS VARCHAR(4)) AS product_code,
            s.bs_side,
            s.currency,
            YEAR(s.start_date)  AS start_year,
            MONTH(s.start_date) AS start_month,
            SUM(CAST(p.beh_outstanding AS FLOAT) * CAST(p.margin AS FLOAT))
                / NULLIF(SUM(CAST(p.beh_outstanding AS FLOAT)), 0) AS wavg_margin
        FROM cf.products p
        JOIN (
            SELECT schedule_id, CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, start_date
            FROM sched.loans
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT schedule_id, CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date
            FROM sched.deposits
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT schedule_id, CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date
            FROM sched.fin_inst
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        ) s ON p.schedule_id = s.schedule_id
        WHERE CAST(p.product_code AS VARCHAR(4)) IN ({_C_IN})
          AND p.cf_end_dt > :rd
          AND COALESCE(p.beh_outstanding, 0) > 0
          AND p.margin IS NOT NULL
        GROUP BY
            s.product_code, s.bs_side, s.currency,
            YEAR(s.start_date), MONTH(s.start_date)
    """)
    df = _try_query(q, {"rd": REPORT_DATE})
    result: dict = {}
    if not df.empty:
        df["product_code"] = df["product_code"].astype(str)
        for _, r in df.iterrows():
            ck = (str(r["product_code"]), str(r["bs_side"]), str(r["currency"]),
                  int(r["start_year"]), int(r["start_month"]))
            v = r["wavg_margin"]
            if pd.notna(v):
                result[ck] = float(v)
    return result


def _load_disc_curves() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load all scenario disc curves from irrbb.curves.

    Returns nested dict:
        scenario_id → curve_name → (n_days_arr, log_df_arr)
    Used for log-linear interpolation at arbitrary tenor.
    """
    q = text("""
        SELECT scenario_id, curve_name, n_days, CAST(d_f AS FLOAT) AS d_f
        FROM irrbb.curves
        WHERE report_date = :rd
        ORDER BY scenario_id, curve_name, n_days
    """)
    df = _try_query(q, {"rd": REPORT_DATE})
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    if df.empty:
        return curves
    for (scen, cname), grp in df.groupby(["scenario_id", "curve_name"]):
        grp = grp.sort_values("n_days")
        nd  = grp["n_days"].to_numpy(dtype=float)
        ldf = np.log(np.maximum(grp["d_f"].to_numpy(dtype=float), 1e-12))
        curves.setdefault(str(scen), {})[str(cname)] = (nd, ldf)
    return curves


def _interp_disc(nd: np.ndarray, ldf: np.ndarray, yf: float) -> float:
    """Log-linear interpolation of disc factor at year-fraction yf."""
    n = yf * 365.25
    return float(np.exp(np.interp(n, nd, ldf, left=ldf[0], right=ldf[-1])))


# Standard currency → irrbb curve name
_CCY_CURVE = {"PLN": "PLN_disc_curve", "EUR": "EUR_disc_curve", "USD": "USD_disc_curve"}


def _cohort_pv(cf_m: pd.DataFrame, curves: dict, scenario_id: str, ccy: str) -> float:
    """Compute PV for one cohort using monthly CF buckets and disc curve.

    cf_m: rows for this cohort with columns month_bucket_idx, capital_cf, interest_cf.
    Each bucket's midpoint tenor = (month_bucket_idx + 0.5) / 12 years.
    """
    if cf_m.empty:
        return 0.0
    cname = _CCY_CURVE.get(ccy, "PLN_disc_curve")
    nd_ldf = curves.get(scenario_id, {}).get(cname)
    if nd_ldf is None:
        return 0.0
    nd, ldf = nd_ldf
    yf_arr = (cf_m["month_bucket_idx"].to_numpy(dtype=float) + 0.5) / 12.0
    df_arr = np.exp(np.interp(yf_arr * 365.25, nd, ldf, left=ldf[0], right=ldf[-1]))
    total_cf = cf_m["capital_cf"].to_numpy(dtype=float) + cf_m["interest_cf"].to_numpy(dtype=float)
    return float(np.dot(total_cf, df_arr))


def _load_cohort_nii_irrbb() -> pd.DataFrame:
    """Product-level NII from irrbb.nii_results for COHORT products (base + shocked).

    Returns: product_code, bs_side, currency, scenario_id, nii_total
    """
    q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, scenario_id,
               SUM(nii_total) AS nii_total
        FROM irrbb.nii_results
        WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = _try_query(q, {"rd": REPORT_DATE})
    if not df.empty:
        df["product_code"] = df["product_code"].astype(str)
    return df


def _load_cohort_eve_irrbb() -> pd.DataFrame:
    """Product-level EVE from irrbb.eve_results for COHORT products (base + shocked).

    Returns: product_code, bs_side, currency, scenario_id, pv_total
    """
    q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, scenario_id,
               SUM(pv_total) AS pv_total
        FROM irrbb.eve_results
        WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = _try_query(q, {"rd": REPORT_DATE})
    if not df.empty:
        df["product_code"] = df["product_code"].astype(str)
    return df


def _load_cohort_repricing() -> pd.DataFrame:
    """Balance-weighted repricing tenor (months) and dominant rate_type per cohort."""
    q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency,
               YEAR(start_date)  AS start_year,
               MONTH(start_date) AS start_month,
               rate_type, fixing_freq, maturity_date,
               SUM(balance_amt) AS balance_amt
        FROM sched.loans
        WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency,
                 YEAR(start_date), MONTH(start_date),
                 rate_type, fixing_freq, maturity_date
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)),
               bs_side, currency,
               YEAR(start_date), MONTH(start_date),
               rate_type, NULL AS fixing_freq, maturity_date,
               SUM(balance_amt)
        FROM sched.deposits
        WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency,
                 YEAR(start_date), MONTH(start_date),
                 rate_type, maturity_date
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)),
               bs_side, currency,
               YEAR(start_date), MONTH(start_date),
               rate_type, fixing_freq, maturity_date,
               SUM(balance_amt)
        FROM sched.fin_inst
        WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency,
                 YEAR(start_date), MONTH(start_date),
                 rate_type, fixing_freq, maturity_date
    """)
    df = _try_query(q)
    if df.empty:
        return pd.DataFrame(columns=_COHORT_KEY + ["repricing_tenor_m", "rate_type"])

    df["product_code"] = df["product_code"].astype(str)
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    df["balance_amt"]   = df["balance_amt"].fillna(0.0).astype(float)

    def _row_tenor(row) -> float:
        if str(row.get("rate_type", "V")) == "F":
            if pd.notna(row.get("maturity_date")):
                return max(1.0, (row["maturity_date"] - REPORT_DATE).days / 30.44)
            return 12.0
        return _freq_to_months(row.get("fixing_freq"))

    df["repricing_m"] = df.apply(_row_tenor, axis=1)
    df["bal_x_rep"]   = df["balance_amt"] * df["repricing_m"]

    agg = (df.groupby(_COHORT_KEY)
             .agg(bal_sum=("balance_amt", "sum"), bxr=("bal_x_rep", "sum"))
             .reset_index())
    agg["repricing_tenor_m"] = np.where(
        agg["bal_sum"] > 0, agg["bxr"] / agg["bal_sum"], 12.0)

    # dominant rate_type by balance
    rt = (df.groupby(_COHORT_KEY + ["rate_type"])["balance_amt"]
            .sum().reset_index()
            .sort_values("balance_amt", ascending=False)
            .drop_duplicates(subset=_COHORT_KEY))
    agg = agg.merge(rt[_COHORT_KEY + ["rate_type"]], on=_COHORT_KEY, how="left")
    return agg[_COHORT_KEY + ["repricing_tenor_m", "rate_type"]]


def _load_cohort_coupon_rates() -> pd.DataFrame:
    """Balance-weighted average client_rt per cohort from sched tables.

    Returns _COHORT_KEY + [wavg_client_rt].
    Only meaningful for fixed-rate cohorts — floating products will use
    the fwd curve via coeff_a/b at runtime so coupon_rate is ignored for them.
    """
    q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency,
               YEAR(start_date)  AS start_year,
               MONTH(start_date) AS start_month,
               SUM(CAST(balance_amt AS FLOAT) * ISNULL(CAST(client_rt AS FLOAT), 0.0))
                   / NULLIF(SUM(CAST(balance_amt AS FLOAT)), 0.0) AS wavg_client_rt
        FROM sched.loans
        WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency,
                 YEAR(start_date), MONTH(start_date)
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)),
               bs_side, currency,
               YEAR(start_date), MONTH(start_date),
               SUM(CAST(balance_amt AS FLOAT) * ISNULL(CAST(client_rt AS FLOAT), 0.0))
                   / NULLIF(SUM(CAST(balance_amt AS FLOAT)), 0.0)
        FROM sched.deposits
        WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency,
                 YEAR(start_date), MONTH(start_date)
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)),
               bs_side, currency,
               YEAR(start_date), MONTH(start_date),
               SUM(CAST(balance_amt AS FLOAT) * ISNULL(CAST(client_rt AS FLOAT), 0.0))
                   / NULLIF(SUM(CAST(balance_amt AS FLOAT)), 0.0)
        FROM sched.fin_inst
        WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency,
                 YEAR(start_date), MONTH(start_date)
    """)
    df = _try_query(q)
    if df.empty:
        return pd.DataFrame(columns=_COHORT_KEY + ["wavg_client_rt"])
    df["product_code"] = df["product_code"].astype(str)
    # If multiple sched tables contribute to the same cohort, take balance-weighted avg
    # (the UNION ALL may produce duplicates for products in multiple tables)
    return df.groupby(_COHORT_KEY, as_index=False)["wavg_client_rt"].mean()


def _load_cohort_monthly_schedule() -> tuple[dict, dict, dict, dict]:
    """Load behavioral CFs and aggregate to monthly buckets per cohort.

    Two GROUP BY queries — avoids transferring millions of individual CF rows:
      Q1: outstanding + capital grouped by (cohort, m_start, m_end)  — ~5K rows max
      Q2: locked_rate and t_first grouped by cohort                  — ~500 rows

    Returns 4 dicts keyed by (product_code, bs_side, currency, start_year, start_month):
        outstanding_by : np.ndarray(12,)  outstanding per calendar month (PLN, not fraction)
        capital_by     : np.ndarray(12,)  capital + prepayment per calendar month
        locked_rate_by : float            outstanding-weighted eff_rate for locked CFs
        t_first_by     : float            months to earliest future fixing (999 if none)
    """
    # Deduplicated sched CTE fragment — shared by Q1 and Q2.
    # ROW_NUMBER ensures each schedule_id contributes exactly one row even if it
    # appears in multiple sched tables (UNION ALL would otherwise inflate counts).
    _sched_key_cte = f"""
        sched_key AS (
            SELECT schedule_id, product_code, bs_side, currency, start_date
            FROM (
                SELECT
                    CAST(schedule_id AS VARCHAR(8)) AS schedule_id,
                    CAST(product_code AS VARCHAR(4)) AS product_code,
                    bs_side, currency, start_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY CAST(schedule_id AS VARCHAR(8))
                        ORDER BY product_code
                    ) AS rn
                FROM (
                    SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id,
                           CAST(product_code AS VARCHAR(4)) AS product_code,
                           bs_side, currency, start_date
                    FROM sched.loans
                    WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
                    UNION ALL
                    SELECT CAST(schedule_id AS VARCHAR(8)),
                           CAST(product_code AS VARCHAR(4)),
                           bs_side, currency, start_date
                    FROM sched.deposits
                    WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
                    UNION ALL
                    SELECT CAST(schedule_id AS VARCHAR(8)),
                           CAST(product_code AS VARCHAR(4)),
                           bs_side, currency, start_date
                    FROM sched.fin_inst
                    WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
                ) u
            ) d
            WHERE rn = 1
        )"""

    # Q1 — two-level aggregation:
    #   level 1 (per_loan_period): MAX(outstanding) per loan×period handles
    #             multiple CF rows in the same period sharing the same outstanding.
    #   level 2 (outer SELECT): SUM across loans within each cohort.
    q1 = text(f"""
        WITH {_sched_key_cte},
        period_cfs AS (
            SELECT
                s.product_code, s.bs_side, s.currency,
                YEAR(s.start_date)                      AS start_year,
                MONTH(s.start_date)                     AS start_month,
                CAST(p.schedule_id AS VARCHAR(8))        AS loan_id,
                CASE WHEN p.cf_start_dt <= :rd THEN 0
                     ELSE DATEDIFF(month, :rd, p.cf_start_dt)
                END                                      AS m_start,
                DATEDIFF(month, :rd, p.cf_end_dt)        AS m_end,
                ISNULL(CAST(p.beh_outstanding AS FLOAT), 0.0)           AS outstanding_val,
                ISNULL(CAST(p.beh_capital_pmt AS FLOAT), 0.0)
              + ISNULL(CAST(p.prepayment_pmt  AS FLOAT), 0.0)           AS capital_val
            FROM cf.products p
            JOIN sched_key s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
            WHERE CAST(p.product_code AS VARCHAR(4)) IN ({_C_IN})
              AND p.cf_end_dt > :rd
              AND COALESCE(p.beh_total_pmt, 0) <> 0
        ),
        per_loan_period AS (
            SELECT
                product_code, bs_side, currency, start_year, start_month,
                loan_id, m_start, m_end,
                MAX(outstanding_val) AS outstanding_val,
                SUM(capital_val)     AS capital_val
            FROM period_cfs
            GROUP BY
                product_code, bs_side, currency, start_year, start_month,
                loan_id, m_start, m_end
        )
        SELECT
            product_code, bs_side, currency, start_year, start_month,
            m_start, m_end,
            SUM(outstanding_val) AS outstanding_sum,
            SUM(capital_val)     AS capital_sum
        FROM per_loan_period
        GROUP BY
            product_code, bs_side, currency, start_year, start_month,
            m_start, m_end
    """)

    # Q2 — locked rate + t_first, one row per cohort.
    # Uses the same sched_key CTE so duplicates from UNION ALL are eliminated.
    # The locked_rate ratio (int/out) would cancel K× inflation, but MIN(fixing_dt)
    # and consistency with Q1 make deduplication the cleaner approach.
    q2 = text(f"""
        WITH {_sched_key_cte}
        SELECT
            s.product_code,
            s.bs_side, s.currency,
            YEAR(s.start_date)                                                              AS start_year,
            MONTH(s.start_date)                                                             AS start_month,
            SUM(CASE WHEN (p.fixing_dt IS NULL OR p.fixing_dt <= :rd)
                          AND ISNULL(CAST(p.cf_yf AS FLOAT), 0.0) > 0.0
                     THEN ISNULL(CAST(p.beh_interest_pmt AS FLOAT), 0.0)
                          / CAST(p.cf_yf AS FLOAT)
                     ELSE 0.0 END)                                                          AS locked_int_div_yf,
            SUM(CASE WHEN p.fixing_dt IS NULL OR p.fixing_dt <= :rd
                     THEN ISNULL(CAST(p.beh_outstanding AS FLOAT), 0.0)
                     ELSE 0.0 END)                                                          AS locked_out_sum,
            MIN(CASE WHEN p.fixing_dt > :rd
                     THEN CAST(DATEDIFF(day, :rd, p.fixing_dt) AS FLOAT) / 30.44
                     ELSE NULL END)                                                         AS min_fixing_months
        FROM cf.products p
        JOIN sched_key s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
        WHERE CAST(p.product_code AS VARCHAR(4)) IN ({_C_IN})
          AND p.cf_end_dt > :rd
          AND COALESCE(p.beh_total_pmt, 0) <> 0
        GROUP BY
            s.product_code, s.bs_side, s.currency,
            YEAR(s.start_date), MONTH(s.start_date)
    """)

    params = {"rd": REPORT_DATE}
    df1 = _try_query(q1, params)
    df2 = _try_query(q2, params)

    if df1.empty:
        return {}, {}, {}, {}

    df1["product_code"] = df1["product_code"].astype(str)
    for col in ["outstanding_sum", "capital_sum"]:
        df1[col] = pd.to_numeric(df1[col], errors="coerce").fillna(0.0)
    df1["m_start"] = pd.to_numeric(df1["m_start"], errors="coerce").fillna(0).astype(int)
    df1["m_end"]   = pd.to_numeric(df1["m_end"],   errors="coerce").fillna(1).astype(int).clip(1, 12)

    # ── Assign group IDs (one per cohort) ────────────────────────────────────
    gid = df1.groupby(
        ["product_code", "bs_side", "currency", "start_year", "start_month"],
        sort=False,
    ).ngroup().to_numpy(dtype=int)
    n_groups = int(gid.max()) + 1

    # ── Vectorised month expansion ────────────────────────────────────────────
    m_start_v = df1["m_start"].to_numpy(dtype=int)
    m_end_v   = df1["m_end"].to_numpy(dtype=int)
    out_v     = df1["outstanding_sum"].to_numpy(dtype=float)
    cap_v     = df1["capital_sum"].to_numpy(dtype=float)

    months = np.arange(12)
    mask_out    = (m_start_v[:, None] <= months[None, :]) & (months[None, :] < m_end_v[:, None])
    out_contrib = out_v[:, None] * mask_out

    cap_idx     = np.clip(m_end_v - 1, 0, 11)
    cap_contrib = np.zeros((len(df1), 12))
    np.add.at(cap_contrib, (np.arange(len(df1)), cap_idx), cap_v)

    outstanding_all = np.zeros((n_groups, 12))
    capital_all     = np.zeros((n_groups, 12))
    np.add.at(outstanding_all, gid, out_contrib)
    np.add.at(capital_all,     gid, cap_contrib)

    # ── Build cohort key → group_id map ──────────────────────────────────────
    key_to_gid: dict = {}
    for i, g in enumerate(gid):
        r = df1.iloc[i]
        ck = (str(r["product_code"]), str(r["bs_side"]), str(r["currency"]),
              int(r["start_year"]), int(r["start_month"]))
        if ck not in key_to_gid:
            key_to_gid[ck] = g

    # ── Locked rate + t_first from Q2 ────────────────────────────────────────
    locked_rate_g = np.zeros(n_groups)
    t_first_g     = np.full(n_groups, 999.0)

    if not df2.empty:
        df2["product_code"] = df2["product_code"].astype(str)
        for _, r2 in df2.iterrows():
            ck = (str(r2["product_code"]), str(r2["bs_side"]), str(r2["currency"]),
                  int(r2["start_year"]), int(r2["start_month"]))
            g = key_to_gid.get(ck)
            if g is None:
                continue
            locked_num = float(r2.get("locked_int_div_yf", 0.0) or 0.0)
            locked_den = float(r2.get("locked_out_sum",    0.0) or 0.0)
            if locked_den > 0:
                locked_rate_g[g] = locked_num / locked_den
            fix_m = r2.get("min_fixing_months")
            if pd.notna(fix_m):
                t_first_g[g] = float(fix_m)

    # ── Result dicts ──────────────────────────────────────────────────────────
    outstanding_by: dict = {}
    capital_by:     dict = {}
    locked_rate_by: dict = {}
    t_first_by:     dict = {}

    for ck, g in key_to_gid.items():
        outstanding_by[ck] = outstanding_all[g].copy()
        capital_by[ck]     = capital_all[g].copy()
        locked_rate_by[ck] = float(locked_rate_g[g])
        t_first_by[ck]     = float(t_first_g[g])

    return outstanding_by, capital_by, locked_rate_by, t_first_by


def _load_product_lcr_nsfr() -> pd.DataFrame:
    """Regulatory LCR/NSFR factors per (product_code, bs_side, currency) from schemat tables."""
    loans_q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code, bs_side, currency,
               AVG(CAST(1.0 - ISNULL(haircut, 0) AS FLOAT)) AS hqla_factor,
               0.0                                           AS lcr_runoff,
               0.0                                           AS asf_factor,
               AVG(COALESCE(CAST(rsf_weight AS FLOAT), 0.0)) AS rsf_factor
        FROM schemat.loans WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency
    """)
    fin_q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code, bs_side, currency,
               AVG(CAST(1.0 - ISNULL(haircut, 0) AS FLOAT))  AS hqla_factor,
               AVG(COALESCE(CAST(lcr_weight AS FLOAT), 0.0))  AS lcr_runoff,
               AVG(COALESCE(CAST(asf_weight AS FLOAT), 0.0))  AS asf_factor,
               AVG(COALESCE(CAST(rsf_weight AS FLOAT), 0.0))  AS rsf_factor
        FROM schemat.financial_instruments WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency
    """)
    dep_q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code, bs_side, currency,
               0.0                                           AS hqla_factor,
               AVG(COALESCE(CAST(lcr_weight AS FLOAT), 0.0)) AS lcr_runoff,
               AVG(COALESCE(CAST(asf_weight AS FLOAT), 0.0)) AS asf_factor,
               0.0                                           AS rsf_factor
        FROM schemat.deposits WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency
    """)
    parts = [_try_query(q, {"rd": REPORT_DATE}) for q in [loans_q, fin_q, dep_q]]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=_PROD_KEY + ["hqla_factor", "lcr_runoff", "asf_factor", "rsf_factor"])
    df = pd.concat(parts, ignore_index=True)
    df["product_code"] = df["product_code"].astype(str)
    return (df.groupby(_PROD_KEY)
              .agg(hqla_factor=("hqla_factor", "max"),
                   lcr_runoff  =("lcr_runoff",  "max"),
                   asf_factor  =("asf_factor",  "max"),
                   rsf_factor  =("rsf_factor",  "max"))
              .reset_index())


# ─────────────────────────────────────────────────────────────────────────────
# Single-row product loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_single_balance() -> pd.DataFrame:
    q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code, bs_side, currency,
               SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
        FROM schemat.loans WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_S_IN})
        GROUP BY product_code, bs_side, currency
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)), bs_side, currency,
               SUM(CAST(balance_amt AS FLOAT))
        FROM schemat.deposits WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_S_IN})
        GROUP BY product_code, bs_side, currency
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)), bs_side, currency,
               SUM(CAST(balance_amt AS FLOAT))
        FROM schemat.financial_instruments WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_S_IN})
        GROUP BY product_code, bs_side, currency
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)), 'E' AS bs_side,
               ISNULL(currency, 'PLN') AS currency,
               SUM(CAST(balance_amt AS FLOAT))
        FROM schemat.equity WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) IN ({_S_IN})
        GROUP BY product_code, currency
    """)
    df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df.groupby(_PROD_KEY, as_index=False)["balance_amt"].sum()


def _load_irs_balance() -> pd.DataFrame:
    """Notional balances for IRS/derivatives from schemat.ir_swaps.

    IRS data lives in a separate table from regular banking products.
    Returns (product_code, bs_side, currency, balance_amt) per leg side.
    """
    q = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code, bs_side, currency,
               SUM(CAST(notional AS FLOAT)) AS balance_amt
        FROM schemat.ir_swaps
        WHERE report_date = :rd
          AND CAST(product_code AS VARCHAR(4)) = '0000'
        GROUP BY product_code, bs_side, currency
    """)
    df = _try_query(q, {"rd": REPORT_DATE})
    if not df.empty:
        df["product_code"] = df["product_code"].astype(str)
    return df


def _load_single_nii() -> pd.DataFrame:
    q = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, scenario_id, SUM(nii_total) AS nii_total
        FROM irrbb.nii_results WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df[df["product_code"].isin(SINGLE_ROW_PRODUCT_CODES | IRS_PRODUCT_CODES)]


def _load_single_eve() -> pd.DataFrame:
    q = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, scenario_id, SUM(pv_total) AS pv_total
        FROM irrbb.eve_results WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df[df["product_code"].isin(SINGLE_ROW_PRODUCT_CODES | IRS_PRODUCT_CODES)]


# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_product_params() -> None:
    print("Loading bs_structure...")
    bs = _load_bs_structure()
    print(f"  {len(bs)} rows in bs_structure")

    print("Loading interest_rt.xlsx...")
    ir_coeff = _load_rate_coefficients()

    # ── cohort data ───────────────────────────────────────────────────────────
    print("Loading cohort balance from schemat tables...")
    coh_bal = _load_cohort_balance()
    print(f"  {len(coh_bal)} cohort rows (product × start_month)")
    _dep_bal = coh_bal[coh_bal["product_code"].isin(["3000", "3100"])]
    print(f"  [diag] deposit rows in coh_bal (3000/3100): {len(_dep_bal)}")
    if not _dep_bal.empty:
        print(_dep_bal[["product_code", "bs_side", "currency", "start_year", "start_month", "balance_amt"]].to_string(index=False))

    print("Loading cohort CF stats from cf.products...")
    coh_cf = _load_cohort_cf_stats()
    print(f"  {len(coh_cf)} cohort CF rows")
    _dep_cf = coh_cf[coh_cf["product_code"].isin(["3000", "3100"])]
    print(f"  [diag] deposit rows in coh_cf (3000/3100): {len(_dep_cf)}")
    if not _dep_cf.empty:
        print(_dep_cf[["product_code", "bs_side", "currency", "start_year", "start_month", "nii_interest"]].head(5).to_string(index=False))

    print("Loading cohort monthly CF buckets for PV computation...")
    coh_cf_q = _load_cohort_cf_monthly()
    print(f"  {len(coh_cf_q)} monthly CF bucket rows")
    _dep_cfq = coh_cf_q[coh_cf_q["product_code"].isin(["3000", "3100"])]
    print(f"  [diag] deposit rows in coh_cf_q (3000/3100): {len(_dep_cfq)}")
    if not _dep_cfq.empty:
        _dep_cfq_keys = _dep_cfq.groupby(["product_code","bs_side","currency","start_year","start_month"]).size()
        print(f"  Deposit cohort keys in coh_cf_q:\n{_dep_cfq_keys.to_string()}")
    else:
        _all_pcs = sorted(coh_cf_q["product_code"].unique().tolist()) if not coh_cf_q.empty else []
        print(f"  [diag] all product_codes in coh_cf_q: {_all_pcs}")

    print("Loading disc curves from irrbb.curves (base + shocked)...")
    disc_curves = _load_disc_curves()
    print(f"  Scenarios loaded: {list(disc_curves.keys())}")

    print("Loading cohort NII from irrbb.nii_results...")
    coh_nii_irrbb = _load_cohort_nii_irrbb()

    print("Loading cohort EVE from irrbb.eve_results...")
    coh_eve_irrbb = _load_cohort_eve_irrbb()

    print("Loading cohort repricing from sched tables...")
    coh_rep = _load_cohort_repricing()

    print("Loading cohort coupon rates from sched tables...")
    coh_coupon = _load_cohort_coupon_rates()
    print(f"  {len(coh_coupon)} cohort coupon rate rows")

    print("Loading cohort monthly schedule from cf.products...")
    monthly_out, monthly_cap, monthly_locked_rt, monthly_t_first = _load_cohort_monthly_schedule()
    print(f"  {len(monthly_out)} cohort schedule groups loaded")

    print("Loading cohort float margins from cf.products...")
    cohort_float_margins = _load_cohort_float_margins()
    print(f"  {len(cohort_float_margins)} cohort float margin entries loaded")

    print("Loading product LCR/NSFR factors from schemat tables...")
    lcr_nsfr = _load_product_lcr_nsfr()

    # ── single-row data ───────────────────────────────────────────────────────
    print("Loading single-row balance from schemat tables...")
    sng_bal = _load_single_balance()

    print("Loading IRS notional balance from schemat.ir_swaps...")
    irs_bal = _load_irs_balance()
    if not irs_bal.empty:
        sng_bal = pd.concat([sng_bal, irs_bal], ignore_index=True)
        print(f"  Added {len(irs_bal)} IRS rows -> sng_bal total: {len(sng_bal)}")

    print("Loading single-row NII from irrbb.nii_results...")
    sng_nii = _load_single_nii()

    print("Loading single-row EVE from irrbb.eve_results...")
    sng_eve = _load_single_eve()

    # ── bs_structure metadata lookup — key = (product_code, bs_side, currency) ─
    bs_meta = {}
    for _, brow in bs.iterrows():
        k = (str(brow["product_code"]), str(brow["bs_side"]), str(brow["currency"]))
        bs_meta[k] = brow.to_dict()

    def _meta(pc, sid, ccy, field, default=None):
        return bs_meta.get((str(pc), str(sid), str(ccy)), {}).get(field, default)

    # ── LCR: per-product effective runoff rates for term deposits ─────────────
    # The exact LCR uses contractual maturity dates from schemat.deposits (not
    # behavioural CFs from cf.products) to split deposits into within-30d (100%
    # runoff) and after-30d (weighted-average LCR rate across all TD products).
    # We query per-product contractual maturity splits and compute the effective
    # LCR runoff rate directly, matching the liq_calc td_maturity_split logic.
    _td_product_codes: set = set()
    _td_eff_lcr: dict = {}  # product_code → effective lcr_runoff rate

    if "product_name" in bs.columns:
        _td_rows = bs[(bs["bs_side"] == "L") & (bs["product_name"] == "term_deposit")].copy()
        _td_product_codes = set(_td_rows["product_code"].astype(str))

        if not _td_rows.empty:
            # td_default_weight: weighted avg LCR rate of all TD products (from bs_structure)
            # This mirrors the exact LCR's default_weight for the after-30d bucket.
            _td_rows["_bal"] = _td_rows["bs_percentage"].fillna(0.0)
            _total_td_bal = _td_rows["_bal"].sum()
            td_default_weight = float(
                (_td_rows["_bal"] * _td_rows["LCR"].fillna(0.0)).sum() / _total_td_bal
            ) if _total_td_bal > 0 else 0.0

            # Per-product contractual maturity split from schemat.deposits.
            # LCR uses contractual maturities, NOT behavioural CFs.
            _td_pc_in = ", ".join(f"'{pc}'" for pc in _td_product_codes)
            _q_td = text(f"""
                SELECT CAST(product_code AS VARCHAR(4)) AS product_code, currency,
                       SUM(CAST(balance_amt AS FLOAT)) AS total_bal,
                       SUM(CASE WHEN maturity_date <= DATEADD(day, 30, :rd)
                                THEN CAST(balance_amt AS FLOAT) ELSE 0.0 END) AS within_30d
                FROM schemat.deposits
                WHERE report_date = :rd
                  AND CAST(product_code AS VARCHAR(4)) IN ({_td_pc_in})
                GROUP BY product_code, currency
            """)
            try:
                _td_split_pp = _try_query(_q_td, {"rd": REPORT_DATE})
                _td_split_pp["product_code"] = _td_split_pp["product_code"].astype(str)
                for _, _r in _td_split_pp.iterrows():
                    _pc  = str(_r["product_code"])
                    _tot = float(_r["total_bal"] or 0.0)
                    _w30 = float(_r["within_30d"] or 0.0)
                    if _tot > 0:
                        _frac_30 = _w30 / _tot
                        _td_eff_lcr[_pc] = _frac_30 * 1.0 + (1.0 - _frac_30) * td_default_weight
            except Exception as _e:
                print(f"  [warn] TD per-product maturity split: {_e}")

    # ── LCR asset inflows: per-product 30d CF from cf.products ───────────────
    # Uses the same formula as liq_calc.load_30d_asset_inflows so the total
    # Σ(balance × inflow_30d_frac) ≈ exact LCR raw inflows (87.4M).
    # This avoids the prepayment_pmt inflation issue in the general capital_30d.
    # cf.products already has product_code/bs_side/currency — no JOIN needed.
    # Joining sched.loans UNION ALL sched.fin_inst inflates counts when a
    # schedule_id appears in both tables (different product attribution).
    _q_lcr_inflow = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency,
               SUM(COALESCE(CAST(beh_capital_pmt AS FLOAT), 0.0)
                   + COALESCE(CAST(beh_interest_pmt AS FLOAT), 0.0)) AS inflow_amt
        FROM cf.products
        WHERE bs_side = 'A'
          AND cf_end_dt > :rd AND cf_end_dt <= :h30
          AND COALESCE(beh_total_pmt, 0) <> 0
          AND CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency
    """)
    _lcr_inflow_map: dict = {}  # (pc, 'A', ccy) → inflow_amt
    try:
        _lcr_infl_df = _try_query(_q_lcr_inflow, {"rd": REPORT_DATE, "h30": HORIZON_30D_END})
        if not _lcr_infl_df.empty:
            _lcr_infl_df["product_code"] = _lcr_infl_df["product_code"].astype(str)
            for _, _r in _lcr_infl_df.iterrows():
                _k = (str(_r["product_code"]), str(_r["bs_side"]), str(_r["currency"]))
                _lcr_inflow_map[_k] = float(_r["inflow_amt"] or 0.0)
    except Exception as _e:
        print(f"  [warn] LCR per-product inflow query: {_e}")

    # ── product total balance per (product_code, bs_side, currency) ───────────
    prod_total_bal = coh_bal.groupby(_PROD_KEY)["balance_amt"].sum().to_dict()

    # ── product total interest CF (denominator for cohort NII share) ──────────
    prod_interest_total: dict = {}
    if not coh_cf.empty:
        for key, grp in coh_cf.groupby(_PROD_KEY):
            prod_interest_total[tuple(key)] = float(grp["nii_interest"].sum())

    # ── monthly CF groups indexed by cohort key tuple for O(1) lookup ────────
    coh_cf_q_groups: dict = {}
    if not coh_cf_q.empty:
        for key, grp in coh_cf_q.groupby(_COHORT_KEY):
            coh_cf_q_groups[tuple(key)] = grp.reset_index(drop=True)
    _dep_cfqg_keys = [k for k in coh_cf_q_groups if str(k[0]) in ("3000", "3100")]
    print(f"  [diag] deposit keys in coh_cf_q_groups: {len(_dep_cfqg_keys)}")
    if _dep_cfqg_keys:
        print(f"  Sample deposit keys: {_dep_cfqg_keys[:5]}")
    # Also show what deposit keys exist in coh_bal for comparison
    _dep_bal_keys = [(str(r.product_code), str(r.bs_side), str(r.currency), int(r.start_year), int(r.start_month)) for _, r in _dep_bal.iterrows()]
    print(f"  [diag] deposit keys expected from coh_bal: {_dep_bal_keys[:5]}")

    # ── irrbb NII lookup maps for cohort products ─────────────────────────────
    coh_nii_base_map: dict = {}   # (pc, sid, ccy) → base nii_total
    coh_nii_scen_map: dict = {}   # (pc, sid, ccy, scen) → shocked nii_total
    if not coh_nii_irrbb.empty:
        for _, r in coh_nii_irrbb.iterrows():
            k3 = (str(r["product_code"]), str(r["bs_side"]), str(r["currency"]))
            if str(r["scenario_id"]) == "base":
                coh_nii_base_map[k3] = float(r["nii_total"])
            else:
                coh_nii_scen_map[k3 + (str(r["scenario_id"]),)] = float(r["nii_total"])

    # ── irrbb EVE lookup maps for cohort products (signed: + for A, - for L/E) ─
    coh_eve_base_map: dict = {}   # (pc, sid, ccy) → base pv_total (signed)
    coh_eve_scen_map: dict = {}   # (pc, sid, ccy, scen) → shocked pv_total (signed)
    if not coh_eve_irrbb.empty:
        for _, r in coh_eve_irrbb.iterrows():
            k3 = (str(r["product_code"]), str(r["bs_side"]), str(r["currency"]))
            if str(r["scenario_id"]) == "base":
                coh_eve_base_map[k3] = float(r["pv_total"])
            else:
                coh_eve_scen_map[k3 + (str(r["scenario_id"]),)] = float(r["pv_total"])

    # ═════════════════════════════════════════════════════════════════════════
    # Build cohort rows
    # ═════════════════════════════════════════════════════════════════════════
    coh_rows = []

    # merge CF stats and repricing into balance frame
    coh = (coh_bal
           .merge(coh_cf,     on=_COHORT_KEY, how="left")
           .merge(coh_rep,    on=_COHORT_KEY, how="left")
           .merge(coh_coupon, on=_COHORT_KEY, how="left")
           .merge(lcr_nsfr,   on=_PROD_KEY,   how="left"))

    # Left-merge leaves NaN for cohorts with no matching CF rows;
    # treat those as zero so they don't propagate NaN through interest_share.
    for _col in ["nii_interest", "total_outstanding", "capital_30d", "capital_1y",
                 "dur_numer", "dur_denom", "bal_yf"]:
        if _col in coh.columns:
            coh[_col] = coh[_col].fillna(0.0)

    for _, row in coh.iterrows():
        pc  = str(row["product_code"])
        sid = str(row["bs_side"])
        ccy = str(row["currency"])
        bal = float(row["balance_amt"])

        # bs_pct proportional split of bs_structure weight
        prod_key = (pc, sid, ccy)
        prod_bal = prod_total_bal.get(prod_key, bal)
        bs_pct   = _meta(pc, sid, ccy, "full_pct", 0.0) * (bal / prod_bal if prod_bal > 0 else 0.0)

        # ── NII base: distribute product-level irrbb NII by balance share ───────
        # Balance share guarantees Σ(nii_unit × bal) = prod_nii_base exactly.
        prod_nii_base = coh_nii_base_map.get(prod_key, 0.0)
        nii_unit = prod_nii_base / prod_bal if prod_bal > 0 else 0.0

        # ── EVE base: distribute product-level irrbb EVE by balance share ───────
        # pv_total in irrbb.eve_results is already signed (+A, -L/E).
        # Balance share guarantees Σ(eve_pv_factor × bal) = prod_eve_base exactly.
        prod_eve_base = coh_eve_base_map.get(prod_key, 0.0)
        eve_pv_factor = prod_eve_base / prod_bal if prod_bal > 0 else 0.0

        # ── Modified duration: from shocked PV of CF buckets (for d_mod only) ───
        sign_i = float(_meta(pc, sid, ccy, "sign", -1.0))
        ck_tuple = (pc, sid, ccy, int(row["start_year"]), int(row["start_month"]))
        cf_q_rows = coh_cf_q_groups.get(ck_tuple, pd.DataFrame())
        pv_pu = _cohort_pv(cf_q_rows, disc_curves, "par_up", ccy)
        pv_pd = _cohort_pv(cf_q_rows, disc_curves, "par_dn", ccy)
        cohort_pv_base = _cohort_pv(cf_q_rows, disc_curves, "base", ccy)
        if bal > 0 and abs(pv_pu) + abs(pv_pd) > 0.0:
            d_mod_from_pv = -sign_i * (pv_pd - pv_pu) / (2.0 * PAR_SHOCK_RATE * bal)
        else:
            dur_n = float(row.get("dur_numer") or 0.0)
            dur_d = float(row.get("dur_denom") or 0.0)
            d_mod_from_pv = dur_n / dur_d if dur_d > 0 else 0.0

        # inflow / amort fractions — denominator is current balance (not sum-of-outstanding)
        infl_30 = float(row.get("capital_30d") or 0.0) / bal if bal > 0 else 0.0
        amrt_1y = float(row.get("capital_1y")  or 0.0) / bal if bal > 0 else 0.0

        # repricing
        rep_m    = float(row.get("repricing_tenor_m") or 12.0)
        rate_typ = str(row.get("rate_type") or "V")

        # coupon_rate: balance-weighted avg contracted client_rt for fixed cohorts.
        # For floating cohorts the rate is computed at runtime from fwd curve + coeff_a/b.
        wavg_client_rt = row.get("wavg_client_rt")
        coupon_r = float(wavg_client_rt) if (rate_typ == "F" and pd.notna(wavg_client_rt)) else np.nan

        # Finalise d_mod using rate_typ
        if rate_typ == "V":
            # Floating: effective duration = half the repricing period (stub only)
            d_mod = rep_m / 12.0 / 2.0
        else:
            d_mod = d_mod_from_pv

        nii_reprice_frac = amrt_1y   # fraction of balance renewing within 1Y (informational)

        # LCR/NSFR regulatory parameters
        hqla_f = float(_meta(pc, sid, ccy, "hqla_factor", 0.0) or 0.0) if sid == "A" else 0.0
        asf_f  = float(_meta(pc, sid, ccy, "ASF",         0.0) or 0.0) if sid in ("L", "E") else 0.0
        rsf_f  = float(_meta(pc, sid, ccy, "RSF",         0.0) or 0.0) if sid == "A" else 0.0

        if sid == "L":
            base_lcr = float(_meta(pc, sid, ccy, "LCR", 0.0) or 0.0)
            if pc in _td_product_codes:
                # Term deposits: use effective LCR rate derived from contractual maturity split
                # (within_30d @ 100% + after_30d @ td_default_weight), calibrated per product.
                lcr_r = _td_eff_lcr.get(pc, td_default_weight if _td_eff_lcr else base_lcr)
            else:
                # Non-term-deposit liabilities: exact LCR applies base_lcr to full balance.
                lcr_r = base_lcr
        else:
            lcr_r = 0.0

        # Inflows: all A-side CFs within 30 days, matching liq_calc formula exactly.
        # The exact liq_calc includes ALL A-side assets (including HQLA bonds)
        # in the raw inflow sum — HQLA assets are NOT excluded here.
        # L/E products have no inflows.
        if sid == "A":
            _prod_inflow = _lcr_inflow_map.get((pc, sid, ccy), 0.0)
            _prod_bal_val = float(prod_total_bal.get((pc, sid, ccy), 0.0) or 0.0)
            inflow_frac = _prod_inflow / _prod_bal_val if _prod_bal_val > 0 else 0.0
        else:
            inflow_frac = 0.0

        # rate coefficients from Excel
        coeff_a = 1.0; coeff_b = 0.0; cli_floor = np.nan; cli_cap = np.nan
        if pc in ir_coeff.index:
            r = ir_coeff.loc[pc]
            if not np.isnan(r.get("coeff_a",     np.nan)): coeff_a   = float(r["coeff_a"])
            if not np.isnan(r.get("coeff_b",     np.nan)): coeff_b   = float(r["coeff_b"])
            if not np.isnan(r.get("client_floor",np.nan)): cli_floor = float(r["client_floor"])
            if not np.isnan(r.get("client_cap",  np.nan)): cli_cap   = float(r["client_cap"])

        # ── schedule monthly profile ─────────────────────────────────────────────
        ck = (pc, sid, ccy, int(row["start_year"]), int(row["start_month"]))
        _out_m = monthly_out.get(ck, np.zeros(12))
        _cap_m = monthly_cap.get(ck, np.zeros(12))
        out_frac_m = _out_m / bal if bal > 0 else np.zeros(12)
        cap_frac_m = _cap_m / bal if bal > 0 else np.zeros(12)
        locked_rt   = monthly_locked_rt.get(ck, 0.0)
        t_first_m   = monthly_t_first.get(ck, 999.0)

        entry = {
            "cohort_id":           _make_cohort_id(row),
            "product_code":        pc,
            "product_name":        _meta(pc, sid, ccy, "product_name", pc),
            "bs_side":             sid,
            "currency":            ccy,
            "start_year":          int(row["start_year"]),
            "start_month":         int(row["start_month"]),
            "is_cohort":           True,
            "bs_pct_current":      bs_pct,
            "balance_amt":         bal,
            "sign":                _meta(pc, sid, ccy, "sign", -1.0),
            "nii_unit_rate":       nii_unit,
            "eve_pv_factor":       eve_pv_factor,
            "d_mod":               d_mod,
            "hqla_factor":         hqla_f,
            "lcr_runoff":          lcr_r,
            "asf_factor":          asf_f,
            "rsf_factor":          rsf_f,
            "inflow_30d_frac":     inflow_frac,
            "amort_frac_1y":       amrt_1y,
            "repricing_tenor_m":   rep_m,
            "rate_type":           rate_typ,
            "coupon_rate":         coupon_r,
            "nii_reprice_frac":    nii_reprice_frac,
            "coeff_a":             coeff_a,
            "coeff_b":             coeff_b,
            "client_floor":        cli_floor,
            "client_cap":          cli_cap,
            # schedule-based monthly profile
            "cohort_outstanding_m": out_frac_m,
            "cohort_capital_m":     cap_frac_m,
            "cohort_locked_rate":   locked_rt,
            "cohort_t_first_m":     t_first_m,
        }
        coh_rows.append(entry)

    # ═════════════════════════════════════════════════════════════════════════
    # Build single-row rows
    # ═════════════════════════════════════════════════════════════════════════
    sng_nii_base = (sng_nii[sng_nii["scenario_id"] == "base"]
                    .groupby(_PROD_KEY)["nii_total"].sum().to_dict())
    sng_eve_base = (sng_eve[sng_eve["scenario_id"] == "base"]
                    .groupby(_PROD_KEY)["pv_total"].sum().to_dict())
    sng_eve_pup  = (sng_eve[sng_eve["scenario_id"] == "par_up"]
                    .groupby(_PROD_KEY)["pv_total"].sum().to_dict())
    sng_eve_pdn  = (sng_eve[sng_eve["scenario_id"] == "par_dn"]
                    .groupby(_PROD_KEY)["pv_total"].sum().to_dict())

    sng_rows = []
    for _, srow in sng_bal.iterrows():
        pc  = str(srow["product_code"])
        sid = str(srow["bs_side"])
        ccy = str(srow["currency"])
        bal = float(srow["balance_amt"])
        k   = (pc, sid, ccy)

        nii_b  = sng_nii_base.get(k, 0.0)
        eve_b  = sng_eve_base.get(k, 0.0)
        eve_pu = sng_eve_pup.get(k, np.nan)
        eve_pd = sng_eve_pdn.get(k, np.nan)

        nii_unit   = nii_b / bal if bal > 0 else 0.0
        eve_factor = eve_b / bal if bal > 0 else 0.0
        if not (np.isnan(eve_pu) or np.isnan(eve_pd)):
            d_mod = -(eve_pd - eve_pu) / (2.0 * PAR_SHOCK_RATE * max(bal, 1.0))
        else:
            d_mod = 0.0

        bs_pct = _meta(pc, sid, ccy, "full_pct", 0.0)
        # IRS notionals not in bs_structure: derive bs_pct from notional / (2 × total_assets)
        if bs_pct == 0.0 and pc in IRS_PRODUCT_CODES and bal > 0:
            bs_pct = bal / (2.0 * TOTAL_ASSETS) * 100.0
        hqla_f = _meta(pc, sid, ccy, "hqla_factor", 0.0) or 0.0 if sid == "A" else 0.0
        lcr_r  = _meta(pc, sid, ccy, "LCR",         0.0) or 0.0 if sid == "L" else 0.0
        asf_f  = _meta(pc, sid, ccy, "ASF",         0.0) or 0.0 if sid in ("L","E") else 0.0
        rsf_f  = _meta(pc, sid, ccy, "RSF",         0.0) or 0.0 if sid == "A" else 0.0

        # IRS sign: derive from bs_side since IRS is not in bs_structure
        sign_sng = _meta(pc, sid, ccy, "sign", None)
        if sign_sng is None:
            sign_sng = 1.0 if sid == "A" else -1.0

        coeff_a = 1.0; coeff_b = 0.0; cli_floor = np.nan; cli_cap = np.nan
        if pc in ir_coeff.index:
            r = ir_coeff.loc[pc]
            if not np.isnan(r.get("coeff_a",     np.nan)): coeff_a   = float(r["coeff_a"])
            if not np.isnan(r.get("coeff_b",     np.nan)): coeff_b   = float(r["coeff_b"])
            if not np.isnan(r.get("client_floor",np.nan)): cli_floor = float(r["client_floor"])
            if not np.isnan(r.get("client_cap",  np.nan)): cli_cap   = float(r["client_cap"])

        entry = {
            "cohort_id":           f"{pc}_{sid}_{ccy}",
            "product_code":        pc,
            "product_name":        _meta(pc, sid, ccy, "product_name", pc),
            "bs_side":             sid,
            "currency":            ccy,
            "start_year":          None,
            "start_month":         None,
            "is_cohort":           False,
            "bs_pct_current":      bs_pct,
            "balance_amt":         bal,
            "sign":                sign_sng,
            "nii_unit_rate":       nii_unit,
            "eve_pv_factor":       eve_factor,
            "d_mod":               d_mod,
            "hqla_factor":         hqla_f,
            "lcr_runoff":          lcr_r,
            "asf_factor":          asf_f,
            "rsf_factor":          rsf_f,
            "inflow_30d_frac":     0.0,
            "amort_frac_1y":       0.0,
            "repricing_tenor_m":   12.0,
            "rate_type":           None,
            "coupon_rate":         None,
            "nii_reprice_frac":    0.0,
            "coeff_a":             coeff_a,
            "coeff_b":             coeff_b,
            "client_floor":        cli_floor,
            "client_cap":          cli_cap,
            # schedule fields — zero/999 for single-row products
            "cohort_outstanding_m": np.zeros(12),
            "cohort_capital_m":     np.zeros(12),
            "cohort_locked_rate":   0.0,
            "cohort_t_first_m":     999.0,
        }
        sng_rows.append(entry)

    # ═════════════════════════════════════════════════════════════════════════
    # Stack and build numpy arrays
    # ═════════════════════════════════════════════════════════════════════════
    all_rows = coh_rows + sng_rows
    n = len(all_rows)
    print(f"\nTotal optimizer entries: {n}  "
          f"({len(coh_rows)} cohort + {len(sng_rows)} single-row)")

    params_df = pd.DataFrame(all_rows)

    balance_arr   = params_df["balance_amt"].to_numpy(dtype=float)
    nii_unit_rate = params_df["nii_unit_rate"].to_numpy(dtype=float)
    eve_pv_factor = params_df["eve_pv_factor"].to_numpy(dtype=float)
    d_mod_arr     = params_df["d_mod"].to_numpy(dtype=float)

    # ── scenario delta arrays ─────────────────────────────────────────────────
    S = len(SHOCKED_SCENARIO_IDS)
    delta_nii_unit = np.zeros((n, S), dtype=float)
    delta_eve_unit = np.zeros((n, S), dtype=float)

    sng_nii_scen = {
        (r["product_code"], r["bs_side"], r["currency"], r["scenario_id"]): float(r["nii_total"])
        for _, r in sng_nii.iterrows()
    }
    sng_eve_scen = {
        (r["product_code"], r["bs_side"], r["currency"], r["scenario_id"]): float(r["pv_total"])
        for _, r in sng_eve.iterrows()
    }

    for i, row in enumerate(all_rows):
        pc, sid, ccy = row["product_code"], row["bs_side"], row["currency"]
        prod_key_i   = (pc, sid, ccy)
        bal_i        = max(row["balance_amt"], 1.0)

        if row["is_cohort"]:
            # Use irrbb product-level values distributed by balance share.
            # This mirrors single-row product logic and guarantees exact aggregates.
            prod_bal_i  = max(prod_total_bal.get(prod_key_i, bal_i), 1.0)
            nii_b_i     = coh_nii_base_map.get(prod_key_i, 0.0)
            eve_b_i     = coh_eve_base_map.get(prod_key_i, 0.0)

            for s_idx, scen in enumerate(SHOCKED_SCENARIO_IDS):
                nii_s = coh_nii_scen_map.get(prod_key_i + (scen,), np.nan)
                if not np.isnan(nii_s):
                    delta_nii_unit[i, s_idx] = (nii_s - nii_b_i) / prod_bal_i

                eve_s = coh_eve_scen_map.get(prod_key_i + (scen,), np.nan)
                if not np.isnan(eve_s):
                    delta_eve_unit[i, s_idx] = (eve_s - eve_b_i) / prod_bal_i
        else:
            nii_b_i = sng_nii_base.get(prod_key_i, 0.0)
            eve_b_i = sng_eve_base.get(prod_key_i, 0.0)
            for s_idx, scen in enumerate(SHOCKED_SCENARIO_IDS):
                ks = prod_key_i + (scen,)
                nii_s = sng_nii_scen.get(ks, np.nan)
                eve_s = sng_eve_scen.get(ks, np.nan)
                if not np.isnan(nii_s):
                    delta_nii_unit[i, s_idx] = (nii_s - nii_b_i) / bal_i
                if not np.isnan(eve_s):
                    delta_eve_unit[i, s_idx] = (eve_s - eve_b_i) / bal_i

    # Catch any residual NaN (missing scenario / zero-balance edge cases)
    np.nan_to_num(delta_nii_unit, copy=False, nan=0.0)
    np.nan_to_num(delta_eve_unit, copy=False, nan=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Write to SQL
    # ─────────────────────────────────────────────────────────────────────────
    sql_params = params_df.assign(report_date=REPORT_DATE)
    sql_params["client_floor"] = np.where(np.isfinite(params_df["client_floor"]),
                                           params_df["client_floor"], None)
    sql_params["client_cap"]   = np.where(np.isfinite(params_df["client_cap"]),
                                           params_df["client_cap"],   None)
    sql_params["coupon_rate"]  = np.where(params_df["coupon_rate"].notna(),
                                           params_df["coupon_rate"],  None)
    sql_params["rate_type"]    = params_df["rate_type"].where(params_df["rate_type"].notna(), None)

    print("Writing to SQL: opt_prep.product_params...")
    opt_sql.reset_product_params()
    opt_sql.write_product_params(sql_params)
    print(f"  Written {len(sql_params)} rows")

    scen_rows = []
    for i, row in enumerate(all_rows):
        for s_idx, scen in enumerate(SHOCKED_SCENARIO_IDS):
            scen_rows.append({
                "report_date":    REPORT_DATE,
                "cohort_id":      row["cohort_id"],
                "product_code":   row["product_code"],
                "bs_side":        row["bs_side"],
                "currency":       row["currency"],
                "scenario_id":    scen,
                "delta_nii_unit": delta_nii_unit[i, s_idx],
                "delta_eve_unit": delta_eve_unit[i, s_idx],
            })
    scenario_df = pd.DataFrame(scen_rows)

    print("Writing to SQL: opt_prep.product_scenario_params...")
    opt_sql.reset_product_scenario_params()
    opt_sql.write_product_scenario_params(scenario_df)
    print(f"  Written {len(scenario_df)} rows  ({n} entries × {S} scenarios)")

    # ─────────────────────────────────────────────────────────────────────────
    # Write Excel inspection
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(EXCEL_OUT), exist_ok=True)
    print(f"Writing inspection Excel -> {EXCEL_OUT}")

    ins = params_df.copy()
    ins["balance_amt_m"]      = (ins["balance_amt"] / 1e6).round(1)
    ins["nii_unit_rate_pct"]  = (ins["nii_unit_rate"]  * 100).round(4)
    ins["eve_pv_factor_pct"]  = (ins["eve_pv_factor"]  * 100).round(2)
    ins["hqla_factor_pct"]    = (ins["hqla_factor"]    * 100).round(1)
    ins["lcr_runoff_pct"]     = (ins["lcr_runoff"]     * 100).round(1)
    ins["asf_factor_pct"]     = (ins["asf_factor"]     * 100).round(1)
    ins["rsf_factor_pct"]     = (ins["rsf_factor"]     * 100).round(1)
    ins["inflow_30d_frac_pct"]= (ins["inflow_30d_frac"]* 100).round(2)
    ins["amort_frac_1y_pct"]  = (ins["amort_frac_1y"]  * 100).round(2)

    nii_piv = (scenario_df
               .pivot_table(index="cohort_id", columns="scenario_id",
                             values="delta_nii_unit", aggfunc="first")
               .rename(columns=lambda c: f"dNII_{c}_pct")
               .reset_index())
    for col in nii_piv.columns:
        if col.startswith("dNII_"):
            nii_piv[col] = (nii_piv[col] * 100).round(4)

    eve_piv = (scenario_df
               .pivot_table(index="cohort_id", columns="scenario_id",
                             values="delta_eve_unit", aggfunc="first")
               .rename(columns=lambda c: f"dEVE_{c}_pct")
               .reset_index())
    for col in eve_piv.columns:
        if col.startswith("dEVE_"):
            eve_piv[col] = (eve_piv[col] * 100).round(4)

    ins = ins.merge(nii_piv, on="cohort_id", how="left")
    ins = ins.merge(eve_piv, on="cohort_id", how="left")

    def _sc(df, cols):
        return df[[c for c in cols if c in df.columns]]

    base_cols  = ["cohort_id", "product_code", "product_name", "bs_side", "currency",
                  "start_year", "start_month", "is_cohort"]
    nii_cols   = base_cols + ["balance_amt_m", "nii_unit_rate_pct"] + [c for c in ins.columns if c.startswith("dNII_")]
    eve_cols   = base_cols + ["balance_amt_m", "eve_pv_factor_pct", "d_mod"] + [c for c in ins.columns if c.startswith("dEVE_")]
    liq_cols   = base_cols + ["balance_amt_m", "hqla_factor_pct", "lcr_runoff_pct",
                               "inflow_30d_frac_pct", "asf_factor_pct", "rsf_factor_pct"]
    rep_cols   = base_cols + ["balance_amt_m", "repricing_tenor_m", "amort_frac_1y_pct",
                               "coeff_a", "coeff_b", "client_floor", "client_cap"]
    ovr_cols   = base_cols + ["bs_pct_current", "balance_amt_m", "nii_unit_rate_pct",
                               "eve_pv_factor_pct", "d_mod", "repricing_tenor_m"]

    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
        _sc(ins, ovr_cols).to_excel(writer, sheet_name="Overview",         index=False)
        _sc(ins, nii_cols).to_excel(writer, sheet_name="NII_params",       index=False)
        _sc(ins, eve_cols).to_excel(writer, sheet_name="EVE_params",       index=False)
        _sc(ins, liq_cols).to_excel(writer, sheet_name="LCR_NSFR_params",  index=False)
        _sc(ins, rep_cols).to_excel(writer, sheet_name="Repricing_params",  index=False)
        scenario_df.drop(columns="report_date").to_excel(
            writer, sheet_name="Scenario_deltas_raw", index=False)
    print(f"  Sheets: Overview | NII_params | EVE_params | LCR_NSFR_params | "
          f"Repricing_params | Scenario_deltas_raw")

    # ─────────────────────────────────────────────────────────────────────────
    # Build schedule monthly arrays  (n, 12)
    # ─────────────────────────────────────────────────────────────────────────
    cohort_outstanding_m = np.vstack([
        r if r is not None else np.zeros(12)
        for r in params_df["cohort_outstanding_m"]
    ]).astype(float)
    cohort_capital_m = np.vstack([
        r if r is not None else np.zeros(12)
        for r in params_df["cohort_capital_m"]
    ]).astype(float)
    cohort_locked_rate = params_df["cohort_locked_rate"].fillna(0.0).to_numpy(dtype=float)
    cohort_t_first_m   = params_df["cohort_t_first_m"].fillna(999.0).to_numpy(dtype=float)

    # ─────────────────────────────────────────────────────────────────────────
    # CohortRates tables — rate_matrix, EVE duration units, quarterly CFs
    # ─────────────────────────────────────────────────────────────────────────
    from bs_vector import CurveTensors

    NPZ_CURVES = os.path.join(BASE_DIR, "..", "output", "curve_tensors.npz")
    _cr_rate_scen = ["base"] + SHOCKED_SCENARIO_IDS          # base always index 0
    _cr_n_scen    = len(_cr_rate_scen)

    try:
        ct = CurveTensors.load(NPZ_CURVES)
        _ct_scenarios  = list(ct.scenario_ids)
        _ct_currencies = list(ct.currencies)
        _base_s        = _ct_scenarios.index("base")
        _N_M           = ct.n_months                         # 360

        print(f"\nBuilding CohortRates arrays (n={n}, n_scen={_cr_n_scen})...")

        # ── currency index per entry ──────────────────────────────────────────
        _ccy_idx = np.array([
            _ct_currencies.index(c) if c in _ct_currencies else 0
            for c in params_df["currency"].tolist()
        ], dtype=int)

        # ── rate_matrix[n, 12, n_scen] ────────────────────────────────────────
        _rate_matrix = np.zeros((n, 12, _cr_n_scen), dtype=float)

        _rt_arr      = params_df["rate_type"].fillna("").to_numpy()
        _coupon_arr  = np.where(params_df["coupon_rate"].notna(),
                                params_df["coupon_rate"].fillna(0.0).astype(float), 0.0)
        _locked_arr  = cohort_locked_rate
        _t1_arr      = np.clip(np.round(cohort_t_first_m).astype(int), 1, 999)
        _F_arr       = np.maximum(params_df["repricing_tenor_m"].fillna(12.0).to_numpy(dtype=float), 1.0)
        _ca_arr      = params_df["coeff_a"].to_numpy(dtype=float)
        _cb_arr      = params_df["coeff_b"].to_numpy(dtype=float)
        _fl_arr      = np.where(np.isfinite(params_df["client_floor"]),
                                params_df["client_floor"].astype(float), -np.inf)
        _cp_arr      = np.where(np.isfinite(params_df["client_cap"]),
                                params_df["client_cap"].astype(float),   np.inf)

        def _fwd_F(disc, event_t, F):
            """F-month forward rate starting at event_t (integer months, 1-indexed)."""
            i_s = max(0, min(event_t - 1, _N_M - 1)) if event_t > 0 else -1
            i_e = max(0, min(event_t - 1 + int(F),   _N_M - 1))
            df_s = disc[i_s] if i_s >= 0 else 1.0
            df_e = max(disc[i_e], 1e-12)
            return (df_s / df_e - 1.0) * 12.0 / F

        for i in range(n):
            rt   = _rt_arr[i]
            ci   = _ccy_idx[i]
            base_disc = ct.disc_factors[_base_s, ci]

            if rt == "F":
                _rate_matrix[i, :, :] = _coupon_arr[i]
            else:
                t1 = int(_t1_arr[i])
                F  = float(_F_arr[i])
                ca = _ca_arr[i]
                fl = _fl_arr[i];  cp = _cp_arr[i]
                base_eff = _locked_arr[i]   # contracted rate for locked CFs

                # Cohort-specific margin for the floating period.
                # Use the outstanding-weighted average margin from cf.products
                # rather than the product-level coeff_b so the stable-margin base
                # matches what was actually earned on the floating CFs.
                cb_float = float(_cb_arr[i])
                if all_rows[i].get("is_cohort") and rt != "F":
                    _ck_i = (str(all_rows[i]["product_code"]),
                             str(all_rows[i]["bs_side"]),
                             str(all_rows[i]["currency"]),
                             int(all_rows[i].get("start_year") or 0),
                             int(all_rows[i].get("start_month") or 0))
                    cb_float = cohort_float_margins.get(_ck_i, cb_float)

                for m in range(12):
                    if m < t1:
                        _rate_matrix[i, m, :] = base_eff
                    else:
                        k      = int((m - t1) / F)
                        ev_t   = int(t1 + k * F)
                        fwd_b  = _fwd_F(base_disc, ev_t, F)
                        # Use stable-margin formula directly so floor/cap clips at the
                        # correct client-rate level regardless of historical base_eff.
                        _rate_matrix[i, m, 0] = np.clip(ca * fwd_b + cb_float, fl, cp)
                        for rs, scen in enumerate(_cr_rate_scen[1:], start=1):
                            if scen not in _ct_scenarios:
                                _rate_matrix[i, m, rs] = _rate_matrix[i, m, 0]
                                continue
                            sh_disc = ct.disc_factors[_ct_scenarios.index(scen), ci]
                            fwd_sh  = _fwd_F(sh_disc, ev_t, F)
                            _rate_matrix[i, m, rs] = np.clip(
                                ca * fwd_sh + cb_float, fl, cp)

        print(f"  rate_matrix built: shape {_rate_matrix.shape}")

        # ── renewal_rate_matrix[n, 12, n_scen] ───────────────────────────────
        # Rate applied to capital repaid in month m — new-business formula:
        # renewal_rate = clip(ca * fwd_at_m + cb, floor, cap)
        # fwd_at_m uses the midpoint of month m (m + 0.5) as the look-up tenor.
        _renewal_rate_matrix = np.zeros((n, 12, _cr_n_scen), dtype=float)
        for i in range(n):
            ci  = _ccy_idx[i]
            ca  = _ca_arr[i];  cb = _cb_arr[i]
            fl  = _fl_arr[i];  cp = _cp_arr[i]
            # use cohort-specific weighted margin if available; fall back to product coeff_b
            if all_rows[i].get("is_cohort") and str(_rt_arr[i]) != "F":
                ck_i = (str(all_rows[i]["product_code"]), str(all_rows[i]["bs_side"]),
                        str(all_rows[i]["currency"]),
                        int(all_rows[i].get("start_year") or 0),
                        int(all_rows[i].get("start_month") or 0))
                cb = cohort_float_margins.get(ck_i, cb)
            base_disc_i = ct.disc_factors[_base_s, ci]
            for m in range(12):
                fwd_b_m = _fwd_F(base_disc_i, m + 1, 1)
                _renewal_rate_matrix[i, m, 0] = np.clip(ca * fwd_b_m + cb, fl, cp)
                for rs, scen in enumerate(_cr_rate_scen[1:], start=1):
                    if scen not in _ct_scenarios:
                        _renewal_rate_matrix[i, m, rs] = _renewal_rate_matrix[i, m, 0]
                        continue
                    sh_disc = ct.disc_factors[_ct_scenarios.index(scen), ci]
                    fwd_sh_m = _fwd_F(sh_disc, m + 1, 1)
                    _renewal_rate_matrix[i, m, rs] = np.clip(ca * fwd_sh_m + cb, fl, cp)

        print(f"  renewal_rate_matrix built: shape {_renewal_rate_matrix.shape}")

        # ── delta_eve_dur_unit[n, n_scen] ─────────────────────────────────────
        # ΔEVE[i,s] = -sign[i] × D_mod[i] × balance[i] × Δfwd(D_mod_tenor, s)
        # Pre-computed per-entry so runtime = balance @ delta_eve_dur_unit[:,s]
        _delta_eve_dur = np.zeros((n, _cr_n_scen), dtype=float)
        _sign_arr = params_df["sign"].to_numpy(dtype=float)

        for i in range(n):
            ci     = _ccy_idx[i]
            dm     = float(d_mod_arr[i])
            dm_m   = max(0, min(int(round(dm * 12)) - 1, _N_M - 1))   # 0-idx month
            fwd_b  = ct.fwd_rates[_base_s, ci, dm_m]
            for rs, scen in enumerate(_cr_rate_scen[1:], start=1):
                if scen not in _ct_scenarios:
                    continue
                fwd_s = ct.fwd_rates[_ct_scenarios.index(scen), ci, dm_m]
                _delta_eve_dur[i, rs] = -_sign_arr[i] * dm * (fwd_s - fwd_b)

        print(f"  delta_eve_dur_unit built: shape {_delta_eve_dur.shape}")

        # ── monthly CF arrays for CF-based EVE ───────────────────────────────
        # Build per-cohort CF schedules.  For products where cf.products only
        # stores the next 1-2 months of CFs (typical for long-term loans),
        # fall back to an analytical annuity schedule so that EVE approach B
        # covers the full remaining life of the product.
        # Threshold: use analytical when database CFs cover < 25% of the
        # product's remaining repricing tenor.
        _cf_grps: list = []   # one DataFrame (or None) per row index
        _BUCKET_M = 3         # quarterly buckets for analytical schedule

        _rep_m_arr = params_df["repricing_tenor_m"].fillna(12.0).to_numpy(dtype=float)
        _cpn_arr2  = np.where(params_df["coupon_rate"].notna(),
                              params_df["coupon_rate"].to_numpy(dtype=float), 0.0)

        for i, row in enumerate(all_rows):
            if not row["is_cohort"]:
                _cf_grps.append(None)
                continue
            ck  = (row["product_code"], row["bs_side"], row["currency"],
                   int(row["start_year"]), int(row["start_month"]))
            grp = coh_cf_q_groups.get(ck)
            bal = float(row["balance_amt"])
            rt_i = str(row.get("rate_type", ""))
            rep_m = int(max(round(float(_rep_m_arr[i])), 1))

            # Use analytical schedule if database CFs cover less than 25% of tenor.
            # Only generate analytical CFs for products modelled in the exact EVE
            # pipeline (eve_pv_factor != 0); products with zero exact EVE are not
            # modelled → keep approach B at 0 to avoid artificial discrepancies.
            eve_pv_i = float(eve_pv_factor[i])
            db_max_m = (int(grp["month_bucket_idx"].max()) if grp is not None and not grp.empty else -1)
            use_analytical = (bal > 0
                              and eve_pv_i != 0.0
                              and db_max_m < max(rep_m * 0.25, 3))

            if use_analytical:
                cpn   = float(_cpn_arr2[i]) if rt_i == "F" else max(float(_cb_arr[i]), 0.0)
                t1_i  = int(max(round(float(_t1_arr[i])), 1))
                grp   = _analytical_cf_schedule(bal, cpn, rep_m, rt_i, t1_i, _BUCKET_M)
            _cf_grps.append(grp)

        _max_q = max(
            (len(g) for g in _cf_grps if g is not None and not g.empty),
            default=0
        )
        _max_q = max(_max_q, 1)
        _cf_cap  = np.zeros((n, _max_q), dtype=float)
        _cf_tot  = np.zeros((n, _max_q), dtype=float)
        _cf_yf   = np.zeros((n, _max_q), dtype=float)
        _cf_nq   = np.zeros(n, dtype=int)

        for i, row in enumerate(all_rows):
            if not row["is_cohort"]:
                continue
            grp = _cf_grps[i]
            if grp is None or grp.empty:
                continue
            bal = float(row["balance_amt"])
            if bal <= 0:
                continue
            m_idx  = grp["month_bucket_idx"].to_numpy(dtype=int)
            cap_cf = grp["capital_cf"].to_numpy(dtype=float)
            int_cf = grp["interest_cf"].to_numpy(dtype=float)
            nq     = len(m_idx)
            if nq > _max_q:
                m_idx  = m_idx[:_max_q]
                cap_cf = cap_cf[:_max_q]
                int_cf = int_cf[:_max_q]
                nq     = _max_q
            _cf_nq[i] = nq
            yf     = (m_idx + 0.5) / 12.0               # midpoint year-fraction
            _cf_cap[i, :nq]  = cap_cf / bal
            _cf_yf[i, :nq]   = yf
            # Fixed cohorts: contracted interest doesn't reprice → include in PV.
            # Floating cohorts: reprice near par → capital-only CF avoids double-
            # counting the rate-sensitive interest component.
            rt_i = str(row.get("rate_type", ""))
            if rt_i == "F":
                _cf_tot[i, :nq] = (cap_cf + int_cf) / bal
            else:
                _cf_tot[i, :nq] = cap_cf / bal

        print(f"  CF arrays built: max_q={_max_q}, entries with CFs={(_cf_nq > 0).sum()}")

        # ── cohort_disc_q[n, max_q, n_scen] — pre-indexed disc factors ────────
        # For each cohort × CF bucket × scenario:  disc factor at bucket midpoint.
        # At optimizer runtime, EVE CF = einsum("iq,iqs->is", cf_frac, cohort_disc_q)
        # — no CurveTensors lookup needed; shock substitution is just a slice.
        _cohort_disc_q = np.zeros((n, _max_q, _cr_n_scen), dtype=float)
        for i in range(n):
            nq = int(_cf_nq[i])
            if nq == 0:
                continue
            ci = int(_ccy_idx[i])
            cf_midx_i = np.clip(
                np.round(_cf_yf[i, :nq] * 12).astype(int) - 1, 0, _N_M - 1
            )
            for rs, scen_label in enumerate(_cr_rate_scen):
                try:
                    s_idx = _ct_scenarios.index(scen_label)
                except ValueError:
                    s_idx = _base_s
                _cohort_disc_q[i, :nq, rs] = ct.disc_factors[s_idx, ci, cf_midx_i]
        print(f"  cohort_disc_q built: shape {_cohort_disc_q.shape}")

        _cr_ok = True
    except Exception as _e:
        print(f"  [warn] CohortRates build failed (curve_tensors.npz missing?): {_e}")
        print(f"         Run extract_curves.py first, then re-run extract_params.py")
        _cr_ok = False
        _cr_n_scen_save = 1
        _rate_matrix          = np.zeros((n, 12, 1), dtype=float)
        _renewal_rate_matrix  = np.zeros((n, 12, 1), dtype=float)
        _delta_eve_dur        = np.zeros((n, 1), dtype=float)
        _cf_cap               = np.zeros((n, 1), dtype=float)
        _cf_tot               = np.zeros((n, 1), dtype=float)
        _cf_yf                = np.zeros((n, 1), dtype=float)
        _cf_nq                = np.zeros(n, dtype=int)
        _ccy_idx              = np.zeros(n, dtype=int)
        _cohort_disc_q        = np.zeros((n, 1, 1), dtype=float)

    # ─────────────────────────────────────────────────────────────────────────
    # Save .npz
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(NPZ_OUT), exist_ok=True)
    np.savez(
        NPZ_OUT,
        cohort_id         = params_df["cohort_id"].to_numpy(),
        product_code      = params_df["product_code"].to_numpy(),
        product_name      = params_df["product_name"].to_numpy(),
        bs_side           = params_df["bs_side"].to_numpy(),
        currency          = params_df["currency"].to_numpy(),
        sign              = params_df["sign"].to_numpy(dtype=float),
        start_year        = np.array([r if r is not None else -1 for r in params_df["start_year"]], dtype=float),
        start_month       = np.array([r if r is not None else -1 for r in params_df["start_month"]], dtype=float),
        is_cohort         = params_df["is_cohort"].to_numpy(),
        bs_pct_current    = params_df["bs_pct_current"].to_numpy(dtype=float),
        nii_unit_rate     = nii_unit_rate,
        delta_nii_unit    = delta_nii_unit,
        eve_pv_factor     = eve_pv_factor,
        d_mod             = d_mod_arr,
        delta_eve_unit    = delta_eve_unit,
        hqla_factor       = params_df["hqla_factor"].to_numpy(dtype=float),
        lcr_runoff        = params_df["lcr_runoff"].to_numpy(dtype=float),
        asf_factor        = params_df["asf_factor"].to_numpy(dtype=float),
        rsf_factor        = params_df["rsf_factor"].to_numpy(dtype=float),
        inflow_30d_frac   = params_df["inflow_30d_frac"].to_numpy(dtype=float),
        amort_frac_1y     = params_df["amort_frac_1y"].to_numpy(dtype=float),
        coeff_a           = params_df["coeff_a"].to_numpy(dtype=float),
        coeff_b           = params_df["coeff_b"].to_numpy(dtype=float),
        client_floor      = np.where(np.isfinite(params_df["client_floor"]),
                                     params_df["client_floor"], np.nan),
        client_cap        = np.where(np.isfinite(params_df["client_cap"]),
                                     params_df["client_cap"],   np.nan),
        repricing_tenor_m = params_df["repricing_tenor_m"].to_numpy(dtype=float),
        rate_type         = params_df["rate_type"].fillna("").to_numpy(),
        coupon_rate       = np.where(
                                params_df["coupon_rate"].notna(),
                                params_df["coupon_rate"].fillna(0.0).astype(float),
                                np.nan,
                            ),
        scenario_ids         = np.array(SHOCKED_SCENARIO_IDS),
        total_assets         = np.array([TOTAL_ASSETS]),
        report_date          = np.array([str(REPORT_DATE.date())]),
        balance_arr          = balance_arr,
        cohort_outstanding_m = cohort_outstanding_m,
        cohort_capital_m     = cohort_capital_m,
        cohort_locked_rate   = cohort_locked_rate,
        cohort_t_first_m     = cohort_t_first_m,
        # CohortRates (cr_*) — CF-based NII/EVE tables
        cr_rate_matrix          = _rate_matrix,
        cr_renewal_rate_matrix  = _renewal_rate_matrix,
        cr_delta_eve_dur_unit   = _delta_eve_dur,
        cr_cf_capital_frac    = _cf_cap,
        cr_cf_fixed_int_frac  = _cf_tot,
        cr_cf_yf              = _cf_yf,
        cr_cf_n_q             = _cf_nq,
        cr_ccy_idx            = _ccy_idx,
        cr_rate_scenario_ids  = np.array(_cr_rate_scen if _cr_ok else ["base"]),
        cr_cohort_disc_q      = _cohort_disc_q,
    )
    print(f"Saved product_params.npz -> {NPZ_OUT}")

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\n-- Parameter summary --")
    print(f"  Total entries             : {n}")
    print(f"  Cohort entries            : {len(coh_rows)}")
    print(f"  Single-row entries        : {len(sng_rows)}")
    print(f"  NII unit rate range       : [{nii_unit_rate.min()*100:.3f}%,  {nii_unit_rate.max()*100:.3f}%]")
    print(f"  Modified duration range   : [{d_mod_arr.min():.2f}y, {d_mod_arr.max():.2f}y]")
    print(f"  Repricing tenor range     : [{params_df['repricing_tenor_m'].min():.1f}m, "
          f"{params_df['repricing_tenor_m'].max():.1f}m]")


if __name__ == "__main__":
    build_product_params()
