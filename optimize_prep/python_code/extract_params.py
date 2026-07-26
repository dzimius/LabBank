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
_AMORTISING_FLOAT_RENEWAL_PRODUCTS = {"1100", "2100"}


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
    for _col in ["LCR", "ASF", "RSF", "haircut", "rwa_weight", "PD", "LGD", "vol_elasticity", "fee_unit_rate", "acq_cost_rate"]:
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
    df = df.rename(columns={"a": "coeff_a", "b": "coeff_b"})
    df["coeff_b"] = df["coeff_b"] / 100.0  # Excel stores percent (e.g. 0.50 = 50bps); convert to decimal
    return df.set_index("product_code")


def _load_fin_data() -> dict:
    """Load global financial parameters from the fin_data sheet in bank_data.xlsx.

    Expected layout: two columns 'parameter' and 'value'.
    Returns dict: {'CoC': 0.10, 'CET1': 0.12, ...}
    Defaults to CoC=0.10 and CET1=0.12 if the sheet is missing.
    """
    try:
        df = pd.read_excel(BS_PATH, sheet_name="fin_data")
        df.columns = [str(c).strip() for c in df.columns]
        if "parameter" in df.columns and "value" in df.columns:
            return {str(r["parameter"]): float(r["value"]) for _, r in df.iterrows()}
    except Exception as e:
        print(f"  [warn] fin_data sheet not found or unreadable ({e}); using defaults")
    return {}


def _load_subst_matrix() -> pd.DataFrame:
    """Load substitution (cannibalism) pairs from the subst_matrix sheet in bank_data.xlsx.

    Expected columns: source_code, source_side, dest_code, dest_side, subst_rate.
    subst_rate = fraction of dest product growth that comes FROM the source product.
    Returns empty DataFrame if sheet is missing.
    """
    try:
        df = pd.read_excel(BS_PATH, sheet_name="subst_matrix")
        df.columns = [str(c).strip() for c in df.columns]
        required = {"source_code", "source_side", "dest_code", "dest_side", "subst_rate"}
        if not required.issubset(df.columns):
            return pd.DataFrame(columns=list(required))
        df["subst_rate"] = pd.to_numeric(df["subst_rate"], errors="coerce").fillna(0.0)
        # Drop note rows (subst_rate == 0 or source_code is a long description string)
        df = df[df["subst_rate"] > 0.0].copy()

        def _to_pc(v) -> str:
            """Convert potentially float product code (e.g. 7060.0) to string '7060'."""
            try:
                return str(int(float(v)))
            except (ValueError, TypeError):
                return str(v).strip()

        df["source_code"] = df["source_code"].apply(_to_pc)
        df["dest_code"]   = df["dest_code"].apply(_to_pc)
        df["source_side"] = df["source_side"].astype(str).str.strip()
        df["dest_side"]   = df["dest_side"].astype(str).str.strip()
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"  [warn] subst_matrix sheet not found or unreadable ({e}); no substitution constraints")
        return pd.DataFrame(columns=["source_code", "source_side", "dest_code", "dest_side", "subst_rate"])


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
            mb.month_bucket_idx,
            SUM(CAST(p.beh_capital_pmt  AS FLOAT)
                + ISNULL(CAST(p.prepayment_pmt AS FLOAT), 0.0)) AS capital_cf,
            SUM(CAST(p.beh_interest_pmt AS FLOAT))               AS interest_cf
        FROM cf.products p
        JOIN sched_key s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
          AND CAST(p.product_code AS VARCHAR(4)) = s.product_code
        CROSS APPLY (
            SELECT CASE WHEN DATEDIFF(month, :rd, p.cf_end_dt) <= 0 THEN 0
                        ELSE DATEDIFF(month, :rd, p.cf_end_dt) - 1
                   END AS month_bucket_idx
        ) mb
        WHERE CAST(p.product_code AS VARCHAR(4)) IN ({_C_IN})
          AND p.cf_end_dt > :rd
          AND COALESCE(p.beh_total_pmt, 0) <> 0
        GROUP BY s.product_code, s.bs_side, s.currency,
                 YEAR(s.start_date), MONTH(s.start_date),
                 mb.month_bucket_idx
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


def _analytical_monthly_profile(
    balance: float,
    annual_coupon: float,
    term_m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Monthly outstanding and capital arrays for NII computation (12 months).

    Uses a full annuity amortization schedule — no run-to-first-repricing
    bullet because NII accrues on the full outstanding balance through the
    entire 12-month horizon regardless of when the rate reprices.

    Returns:
        out_m : (12,) outstanding balance at the start of each month (PLN)
        cap_m : (12,) capital repaid during each month (PLN)
    """
    if term_m <= 0 or balance <= 0:
        return np.zeros(12), np.zeros(12)

    r_m   = max(annual_coupon, 0.0) / 12.0
    gen_m = max(int(term_m), 1)

    if r_m > 1e-9:
        pmt = balance * r_m / (1.0 - (1.0 + r_m) ** (-gen_m))
    else:
        pmt = balance / gen_m

    out_m   = np.zeros(12)
    cap_m   = np.zeros(12)
    rem_bal = balance

    for m in range(min(12, gen_m)):
        out_m[m]  = rem_bal
        int_m     = rem_bal * r_m
        cap_val   = min(pmt - int_m, rem_bal)
        cap_m[m]  = cap_val
        rem_bal   = max(rem_bal - cap_val, 0.0)
        if rem_bal < 1.0:
            break

    return out_m, cap_m


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
            SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id, 'L' AS src,
                   CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, start_date
            FROM sched.loans
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)), 'D', CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date
            FROM sched.deposits
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)), 'F', CAST(product_code AS VARCHAR(4)),
                   bs_side, currency, start_date
            FROM sched.fin_inst
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        ) s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
           AND p.product_type = s.src
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


def _load_cohort_monthly_schedule() -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    """Load behavioral CFs and aggregate to monthly buckets per cohort.

    Two GROUP BY queries — avoids transferring millions of individual CF rows:
      Q1: outstanding + capital grouped by (cohort, m_start, m_end)  — ~5K rows max
      Q2: locked_rate and t_first grouped by cohort                  — ~500 rows

    Returns 4 dicts keyed by (product_code, bs_side, currency, start_year, start_month):
        outstanding_by : np.ndarray(12,)  outstanding per calendar month (PLN, not fraction)
        capital_by     : np.ndarray(12,)  capital + prepayment per calendar month
        locked_rate_by : float            outstanding-weighted eff_rate for locked CFs
        t_first_by     : float            months to earliest future fixing (999 if none)
        locked_frac_by : np.ndarray(12,)  locked outstanding share per month
        locked_rate_m_by: np.ndarray(12,) outstanding-weighted locked rate per month
        float_base_eff_m_by: np.ndarray(12,) outstanding-weighted base client rate
        float_base_fwd_m_by: np.ndarray(12,) outstanding-weighted base market forward
    """
    # Deduplicated sched CTE fragment — shared by Q1 and Q2.
    # Partitions by (src, schedule_id) so that loans, deposits, and fin_inst each
    # have independent schedule_id spaces.  A plain PARTITION BY schedule_id caused
    # deposits/fin_inst entries to be silently dropped whenever a loan shared the
    # same integer schedule_id, because '1000' < '7060' in the ORDER BY.
    _sched_key_cte = f"""
        sched_key AS (
            SELECT schedule_id, src, product_code, bs_side, currency, start_date
            FROM (
                SELECT
                    CAST(schedule_id AS VARCHAR(8)) AS schedule_id,
                    src,
                    CAST(product_code AS VARCHAR(4)) AS product_code,
                    bs_side, currency, start_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY src, CAST(schedule_id AS VARCHAR(8))
                        ORDER BY product_code
                    ) AS rn
                FROM (
                    SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id,
                           CAST(product_code AS VARCHAR(4)) AS product_code,
                           bs_side, currency, start_date, 'L' AS src
                    FROM sched.loans
                    WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
                    UNION ALL
                    SELECT CAST(schedule_id AS VARCHAR(8)),
                           CAST(product_code AS VARCHAR(4)),
                           bs_side, currency, start_date, 'D' AS src
                    FROM sched.deposits
                    WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
                    UNION ALL
                    SELECT CAST(schedule_id AS VARCHAR(8)),
                           CAST(product_code AS VARCHAR(4)),
                           bs_side, currency, start_date, 'F' AS src
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
              + ISNULL(CAST(p.prepayment_pmt  AS FLOAT), 0.0)           AS capital_val,
                CASE WHEN (p.cf_start_dt < :rd OR (p.fixing_dt IS NOT NULL AND p.fixing_dt < :rd))
                     THEN ISNULL(CAST(p.beh_outstanding AS FLOAT), 0.0)
                     ELSE 0.0 END                                       AS locked_outstanding_val,
                CASE WHEN (p.cf_start_dt < :rd OR (p.fixing_dt IS NOT NULL AND p.fixing_dt < :rd))
                          AND ISNULL(CAST(p.cf_yf AS FLOAT), 0.0) > 0.0
                     THEN ISNULL(CAST(p.beh_interest_pmt AS FLOAT), 0.0)
                          / CAST(p.cf_yf AS FLOAT)
                     ELSE 0.0 END                                       AS locked_int_div_yf_val,
                CASE WHEN NOT (p.cf_start_dt < :rd OR (p.fixing_dt IS NOT NULL AND p.fixing_dt < :rd))
                     THEN ISNULL(CAST(p.beh_outstanding AS FLOAT), 0.0)
                     ELSE 0.0 END                                       AS float_outstanding_val,
                CASE WHEN NOT (p.cf_start_dt < :rd OR (p.fixing_dt IS NOT NULL AND p.fixing_dt < :rd))
                          AND ISNULL(CAST(p.cf_yf AS FLOAT), 0.0) > 0.0
                     THEN ISNULL(CAST(p.beh_interest_pmt AS FLOAT), 0.0)
                          / CAST(p.cf_yf AS FLOAT)
                     ELSE 0.0 END                                       AS float_int_div_yf_val,
                CASE WHEN NOT (p.cf_start_dt < :rd OR (p.fixing_dt IS NOT NULL AND p.fixing_dt < :rd))
                     THEN ISNULL(CAST(p.beh_outstanding AS FLOAT), 0.0)
                          * ISNULL(CAST(p.fwd_rt AS FLOAT), 0.0)
                     ELSE 0.0 END                                       AS float_fwd_num_val
            FROM cf.products p
            JOIN sched_key s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
                             AND p.product_type = s.src
            WHERE CAST(p.product_code AS VARCHAR(4)) IN ({_C_IN})
              AND p.cf_end_dt > :rd
              AND COALESCE(p.beh_total_pmt, 0) <> 0
        ),
        per_loan_period AS (
            SELECT
                product_code, bs_side, currency, start_year, start_month,
                loan_id, m_start, m_end,
                MAX(outstanding_val) AS outstanding_val,
                SUM(capital_val)     AS capital_val,
                MAX(locked_outstanding_val) AS locked_outstanding_val,
                SUM(locked_int_div_yf_val)  AS locked_int_div_yf_val,
                MAX(float_outstanding_val)  AS float_outstanding_val,
                SUM(float_int_div_yf_val)   AS float_int_div_yf_val,
                MAX(float_fwd_num_val)      AS float_fwd_num_val
            FROM period_cfs
            GROUP BY
                product_code, bs_side, currency, start_year, start_month,
                loan_id, m_start, m_end
        )
        SELECT
            product_code, bs_side, currency, start_year, start_month,
            m_start, m_end,
            SUM(outstanding_val) AS outstanding_sum,
            SUM(capital_val)     AS capital_sum,
            SUM(locked_outstanding_val) AS locked_outstanding_sum,
            SUM(locked_int_div_yf_val)  AS locked_int_div_yf_sum,
            SUM(float_outstanding_val)  AS float_outstanding_sum,
            SUM(float_int_div_yf_val)   AS float_int_div_yf_sum,
            SUM(float_fwd_num_val)      AS float_fwd_num_sum
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
            SUM(CASE WHEN (p.cf_start_dt < :rd OR (p.fixing_dt IS NOT NULL AND p.fixing_dt < :rd))
                          AND ISNULL(CAST(p.cf_yf AS FLOAT), 0.0) > 0.0
                     THEN ISNULL(CAST(p.beh_interest_pmt AS FLOAT), 0.0)
                          / CAST(p.cf_yf AS FLOAT)
                     ELSE 0.0 END)                                                          AS locked_int_div_yf,
            SUM(CASE WHEN (p.cf_start_dt < :rd OR (p.fixing_dt IS NOT NULL AND p.fixing_dt < :rd))
                     THEN ISNULL(CAST(p.beh_outstanding AS FLOAT), 0.0)
                     ELSE 0.0 END)                                                          AS locked_out_sum,
            MIN(CASE WHEN p.fixing_dt > :rd
                     THEN CAST(DATEDIFF(day, :rd, p.fixing_dt) AS FLOAT) / 30.44
                     ELSE NULL END)                                                         AS min_fixing_months
        FROM cf.products p
        JOIN sched_key s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
                         AND p.product_type = s.src
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
        return {}, {}, {}, {}, {}, {}, {}, {}

    df1["product_code"] = df1["product_code"].astype(str)
    for col in [
        "outstanding_sum", "capital_sum",
        "locked_outstanding_sum", "locked_int_div_yf_sum",
        "float_outstanding_sum", "float_int_div_yf_sum", "float_fwd_num_sum",
    ]:
        df1[col] = pd.to_numeric(df1[col], errors="coerce").fillna(0.0)
    df1["m_start"]   = pd.to_numeric(df1["m_start"], errors="coerce").fillna(0).astype(int)
    df1["m_end_raw"] = pd.to_numeric(df1["m_end"],   errors="coerce").fillna(1).astype(int)
    df1["m_end"]     = df1["m_end_raw"].clip(1, 12)

    # ── Assign group IDs (one per cohort) ────────────────────────────────────
    gid = df1.groupby(
        ["product_code", "bs_side", "currency", "start_year", "start_month"],
        sort=False,
    ).ngroup().to_numpy(dtype=int)
    n_groups = int(gid.max()) + 1

    # ── Vectorised month expansion ────────────────────────────────────────────
    m_start_v   = df1["m_start"].to_numpy(dtype=int)
    m_end_v     = df1["m_end"].to_numpy(dtype=int)
    m_end_raw_v = df1["m_end_raw"].to_numpy(dtype=int)
    out_v       = df1["outstanding_sum"].to_numpy(dtype=float)
    cap_v       = df1["capital_sum"].to_numpy(dtype=float)
    lock_out_v  = df1["locked_outstanding_sum"].to_numpy(dtype=float)
    lock_num_v  = df1["locked_int_div_yf_sum"].to_numpy(dtype=float)
    float_out_v = df1["float_outstanding_sum"].to_numpy(dtype=float)
    float_eff_num_v = df1["float_int_div_yf_sum"].to_numpy(dtype=float)
    float_fwd_num_v = df1["float_fwd_num_sum"].to_numpy(dtype=float)

    months = np.arange(12)
    mask_out    = (m_start_v[:, None] <= months[None, :]) & (months[None, :] < m_end_v[:, None])
    out_contrib = out_v[:, None] * mask_out
    lock_out_contrib = lock_out_v[:, None] * mask_out
    lock_num_contrib = lock_num_v[:, None] * mask_out
    float_out_contrib = float_out_v[:, None] * mask_out
    float_eff_num_contrib = float_eff_num_v[:, None] * mask_out
    float_fwd_num_contrib = float_fwd_num_v[:, None] * mask_out

    cap_idx     = np.clip(m_end_raw_v - 1, 0, 11)
    cap_contrib = np.zeros((len(df1), 12))
    cap_in_horizon = (m_end_raw_v >= 1) & (m_end_raw_v <= 12)
    if cap_in_horizon.any():
        cap_rows = np.where(cap_in_horizon)[0]
        np.add.at(cap_contrib, (cap_rows, cap_idx[cap_in_horizon]), cap_v[cap_in_horizon])

    outstanding_all = np.zeros((n_groups, 12))
    capital_all     = np.zeros((n_groups, 12))
    locked_out_all  = np.zeros((n_groups, 12))
    locked_num_all  = np.zeros((n_groups, 12))
    float_out_all   = np.zeros((n_groups, 12))
    float_eff_num_all = np.zeros((n_groups, 12))
    float_fwd_num_all = np.zeros((n_groups, 12))
    np.add.at(outstanding_all, gid, out_contrib)
    np.add.at(capital_all,     gid, cap_contrib)
    np.add.at(locked_out_all,  gid, lock_out_contrib)
    np.add.at(locked_num_all,  gid, lock_num_contrib)
    np.add.at(float_out_all,   gid, float_out_contrib)
    np.add.at(float_eff_num_all, gid, float_eff_num_contrib)
    np.add.at(float_fwd_num_all, gid, float_fwd_num_contrib)

    locked_frac_all = np.divide(
        locked_out_all,
        outstanding_all,
        out=np.zeros_like(locked_out_all),
        where=outstanding_all > 0,
    )
    locked_rate_m_all = np.divide(
        locked_num_all,
        locked_out_all,
        out=np.zeros_like(locked_num_all),
        where=locked_out_all > 0,
    )
    float_base_eff_m_all = np.divide(
        float_eff_num_all,
        float_out_all,
        out=np.zeros_like(float_eff_num_all),
        where=float_out_all > 0,
    )
    float_base_fwd_m_all = np.divide(
        float_fwd_num_all,
        float_out_all,
        out=np.zeros_like(float_fwd_num_all),
        where=float_out_all > 0,
    )

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
    locked_frac_by: dict = {}
    locked_rate_m_by: dict = {}
    float_base_eff_m_by: dict = {}
    float_base_fwd_m_by: dict = {}

    for ck, g in key_to_gid.items():
        outstanding_by[ck] = outstanding_all[g].copy()
        capital_by[ck]     = capital_all[g].copy()
        locked_rate_by[ck] = float(locked_rate_g[g])
        t_first_by[ck]     = float(t_first_g[g])
        locked_frac_by[ck] = locked_frac_all[g].copy()
        locked_rate_m_by[ck] = locked_rate_m_all[g].copy()
        float_base_eff_m_by[ck] = float_base_eff_m_all[g].copy()
        float_base_fwd_m_by[ck] = float_base_fwd_m_all[g].copy()

    return (
        outstanding_by, capital_by, locked_rate_by, t_first_by,
        locked_frac_by, locked_rate_m_by,
        float_base_eff_m_by, float_base_fwd_m_by,
    )


def _load_cohort_effective_nii_tables(
    curves: dict,
    ir_coeff: pd.DataFrame,
    scenario_ids: list[str],
) -> tuple[dict, dict, dict, dict]:
    """Build monthly effective NII tables directly from daily CF rows.

    This is intentionally limited to amortising floating loan products.  Those
    rows have real 12M behavioural cash flows, and they are where the monthly
    rate_matrix approximation loses the most accuracy.

    Returns dicts keyed by cohort:
      interest_yf_by       raw sum(outstanding * cf_yf) by cf_end month
      capital_remain_by    raw sum(capital * actual remain_yf) by cf_end month
      rate_matrix_by       effective client rate [12, n_scen]
      renewal_rate_by      effective renewal rate [12, n_scen]
    """
    codes = sorted(_AMORTISING_FLOAT_RENEWAL_PRODUCTS)
    if not codes:
        return {}, {}, {}, {}
    codes_in = ", ".join(f"'{c}'" for c in codes)
    scen_all = ["base"] + list(scenario_ids)

    q = text(f"""
        WITH sched_key AS (
            SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id, 'L' AS src,
                   CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, start_date
            FROM sched.loans
            WHERE CAST(product_code AS VARCHAR(4)) IN ({codes_in})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)), 'D',
                   CAST(product_code AS VARCHAR(4)), bs_side, currency, start_date
            FROM sched.deposits
            WHERE CAST(product_code AS VARCHAR(4)) IN ({codes_in})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)), 'F',
                   CAST(product_code AS VARCHAR(4)), bs_side, currency, start_date
            FROM sched.fin_inst
            WHERE CAST(product_code AS VARCHAR(4)) IN ({codes_in})
        )
        SELECT
            s.product_code, s.bs_side, s.currency,
            YEAR(s.start_date) AS start_year,
            MONTH(s.start_date) AS start_month,
            DATEDIFF(month, :rd, p.cf_end_dt) AS m_end,
            p.cf_start_dt, p.cf_end_dt, p.fixing_dt,
            CAST(p.cf_yf AS FLOAT) AS cf_yf,
            ISNULL(CAST(p.beh_outstanding AS FLOAT), 0.0) AS outstanding,
            ISNULL(CAST(p.beh_capital_pmt AS FLOAT), 0.0)
              + ISNULL(CAST(p.prepayment_pmt AS FLOAT), 0.0) AS capital,
            ISNULL(CAST(p.beh_interest_pmt AS FLOAT), 0.0) AS interest,
            ISNULL(CAST(p.fwd_rt AS FLOAT), 0.0) AS base_fwd
        FROM cf.products p
        JOIN sched_key s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
                         AND p.product_type = s.src
        WHERE CAST(p.product_code AS VARCHAR(4)) IN ({codes_in})
          AND p.cf_end_dt > :rd
          AND p.cf_end_dt <= :he
          AND COALESCE(p.beh_total_pmt, 0) <> 0
    """)
    horizon_end = REPORT_DATE + pd.Timedelta(days=round(365.25))
    df = _try_query(q, {"rd": REPORT_DATE, "he": horizon_end})
    if df.empty:
        return {}, {}, {}, {}

    df["product_code"] = df["product_code"].astype(str)
    for c in ["cf_start_dt", "cf_end_dt", "fixing_dt"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["cf_yf", "outstanding", "capital", "interest", "base_fwd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["m_idx"] = pd.to_numeric(df["m_end"], errors="coerce").fillna(0).astype(int) - 1
    df = df[(df["m_idx"] >= 0) & (df["m_idx"] < 12)].copy()
    if df.empty:
        return {}, {}, {}, {}

    denom = (df["outstanding"] * df["cf_yf"]).replace(0.0, np.nan)
    df["contracted_rt"] = (df["interest"] / denom).fillna(0.0)
    df["int_w"] = (df["outstanding"] * df["cf_yf"]).clip(lower=0.0)
    df["remain_yf"] = ((horizon_end - df["cf_end_dt"]).dt.days / 365.0).clip(lower=0.0)
    df["cap_remain_w"] = (df["capital"] * df["remain_yf"]).clip(lower=0.0)
    locked = (df["cf_start_dt"] < REPORT_DATE) | (
        df["fixing_dt"].notna() & (df["fixing_dt"] < REPORT_DATE)
    )
    before_curve = df["cf_start_dt"] < REPORT_DATE

    pcs = df["product_code"].astype(str)
    a_v = pcs.map(ir_coeff["coeff_a"].to_dict()).fillna(1.0).to_numpy(dtype=float)
    b_v = pcs.map(ir_coeff["coeff_b"].to_dict()).fillna(0.0).to_numpy(dtype=float)
    floor_v = pcs.map(
        ir_coeff["client_floor"].dropna().to_dict()
        if "client_floor" in ir_coeff.columns else {}
    ).fillna(float("-inf")).to_numpy(dtype=float)
    cap_v = pcs.map(
        ir_coeff["client_cap"].dropna().to_dict()
        if "client_cap" in ir_coeff.columns else {}
    ).fillna(float("inf")).to_numpy(dtype=float)

    def _lookup_df(scenario: str, dates: pd.Series) -> np.ndarray:
        out = np.ones(len(dates), dtype=float)
        days = (pd.to_datetime(dates) - REPORT_DATE).dt.days.to_numpy(dtype=float)
        ccy_arr = df["currency"].astype(str).to_numpy()
        for ccy in np.unique(ccy_arr):
            mask = ccy_arr == ccy
            nd_ldf = curves.get(scenario, {}).get(_CCY_CURVE.get(ccy, "PLN_disc_curve"))
            if nd_ldf is None:
                continue
            nd, ldf = nd_ldf
            vals = np.exp(np.interp(days[mask], nd, ldf, left=ldf[0], right=ldf[-1]))
            vals = np.where(days[mask] <= 0.0, 1.0, vals)
            out[mask] = vals
        return out

    def _fwd_between(scenario: str, start: pd.Series, end: pd.Series, yf: np.ndarray) -> np.ndarray:
        df_s = _lookup_df(scenario, start)
        df_e = _lookup_df(scenario, end)
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = np.where((df_e > 0.0) & (yf > 0.0), (df_s / df_e - 1.0) / yf, 0.0)
        return np.maximum(0.0, np.nan_to_num(fwd, nan=0.0, posinf=0.0, neginf=0.0))

    cf_yf = df["cf_yf"].to_numpy(dtype=float)
    yf_from_report = ((df["cf_end_dt"] - REPORT_DATE).dt.days / 365.0).clip(lower=0.0).to_numpy(dtype=float)
    base_fwd = df["base_fwd"].to_numpy(dtype=float)
    contracted = df["contracted_rt"].to_numpy(dtype=float)
    int_w = df["int_w"].to_numpy(dtype=float)
    cap_remain_w = df["cap_remain_w"].to_numpy(dtype=float)
    locked_arr = locked.to_numpy(dtype=bool)
    before_arr = before_curve.to_numpy(dtype=bool)

    keys_df = df[_COHORT_KEY].drop_duplicates().reset_index(drop=True)
    key_to_gid = {
        (str(r.product_code), str(r.bs_side), str(r.currency), int(r.start_year), int(r.start_month)): i
        for i, r in keys_df.iterrows()
    }
    gid = np.array([
        key_to_gid[(str(r.product_code), str(r.bs_side), str(r.currency), int(r.start_year), int(r.start_month))]
        for _, r in df.iterrows()
    ], dtype=int)
    midx = df["m_idx"].to_numpy(dtype=int)
    n_groups = len(key_to_gid)
    n_scen = len(scen_all)

    int_w_all = np.zeros((n_groups, 12), dtype=float)
    cap_remain_all = np.zeros((n_groups, 12), dtype=float)
    rate_num_all = np.zeros((n_groups, 12, n_scen), dtype=float)
    ren_num_all = np.zeros((n_groups, 12, n_scen), dtype=float)
    np.add.at(int_w_all, (gid, midx), int_w)
    np.add.at(cap_remain_all, (gid, midx), cap_remain_w)

    for s_idx, scen in enumerate(scen_all):
        if scen == "base":
            eff_interest = contracted.copy()
            # Base renewal follows compute_nii_base_schedule: use fwd_rt for
            # future-start CFs, override only already-started periods with the
            # report-date-to-end base curve rate.
            fwd_ren = base_fwd.copy()
            if before_arr.any():
                dfe = _lookup_df("base", df["cf_end_dt"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    fwd_report_end = np.where(
                        (dfe > 0.0) & (yf_from_report > 0.0),
                        (1.0 / dfe - 1.0) / yf_from_report,
                        0.0,
                    )
                fwd_ren[before_arr] = np.maximum(0.0, np.nan_to_num(fwd_report_end[before_arr], nan=0.0))
        else:
            fwd_sh = _fwd_between(scen, df["cf_start_dt"], df["cf_end_dt"], cf_yf)
            fwd_interest = np.where(locked_arr, base_fwd, fwd_sh)
            eff_interest = contracted + a_v * (fwd_interest - base_fwd)

            fwd_ren = fwd_sh.copy()
            if before_arr.any():
                dfe = _lookup_df(scen, df["cf_end_dt"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    fwd_report_end = np.where(
                        (dfe > 0.0) & (yf_from_report > 0.0),
                        (1.0 / dfe - 1.0) / yf_from_report,
                        0.0,
                    )
                fwd_ren[before_arr] = np.maximum(0.0, np.nan_to_num(fwd_report_end[before_arr], nan=0.0))

        eff_interest = np.minimum(cap_v, np.maximum(floor_v, eff_interest))
        ren_rate = np.minimum(cap_v, np.maximum(floor_v, a_v * fwd_ren + b_v))
        np.add.at(rate_num_all[:, :, s_idx], (gid, midx), int_w * eff_interest)
        np.add.at(ren_num_all[:, :, s_idx], (gid, midx), cap_remain_w * ren_rate)

    rate_all = np.divide(
        rate_num_all,
        int_w_all[:, :, None],
        out=np.zeros_like(rate_num_all),
        where=int_w_all[:, :, None] > 0.0,
    )
    ren_rate_all = np.divide(
        ren_num_all,
        cap_remain_all[:, :, None],
        out=np.zeros_like(ren_num_all),
        where=cap_remain_all[:, :, None] > 0.0,
    )

    interest_yf_by: dict = {}
    capital_remain_by: dict = {}
    rate_matrix_by: dict = {}
    renewal_rate_by: dict = {}
    for ck, g in key_to_gid.items():
        interest_yf_by[ck] = int_w_all[g].copy()
        capital_remain_by[ck] = cap_remain_all[g].copy()
        rate_matrix_by[ck] = rate_all[g].copy()
        renewal_rate_by[ck] = ren_rate_all[g].copy()
    return interest_yf_by, capital_remain_by, rate_matrix_by, renewal_rate_by


def _query_cohort_cf_schedule() -> pd.DataFrame:
    """Curve-independent daily-CF schedule rows behind the EVE PV reconstruction.

    Split out of _load_cohort_effective_eve_pv_tables() so the schedule (cash-flow
    timing/amounts, contracted rates) can be queried ONCE and then repriced against
    multiple different market curves (see anchor_eve_reprice.py) without re-querying
    the DB per curve -- the schedule itself doesn't depend on which curve is used to
    discount/reprice it.
    """
    q = text(f"""
        WITH sched_key AS (
            SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id, 'L' AS src,
                   CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, start_date,
                   ISNULL(rate_type, 'V') AS rate_type
            FROM sched.loans
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)), 'D',
                   CAST(product_code AS VARCHAR(4)), bs_side, currency, start_date,
                   ISNULL(rate_type, 'V') AS rate_type
            FROM sched.deposits
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)), 'F',
                   CAST(product_code AS VARCHAR(4)), bs_side, currency, start_date,
                   ISNULL(rate_type, 'F') AS rate_type
            FROM sched.fin_inst
            WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        )
        SELECT
            CAST(p.schedule_id AS VARCHAR(8)) AS schedule_id,
            CAST(p.product_type AS VARCHAR(1)) AS product_type,
            s.product_code, s.bs_side, s.currency,
            YEAR(s.start_date) AS start_year,
            MONTH(s.start_date) AS start_month,
            CASE WHEN DATEDIFF(month, :rd, p.cf_end_dt) <= 0 THEN 0
                 ELSE DATEDIFF(month, :rd, p.cf_end_dt) - 1
            END AS m_idx,
            s.rate_type,
            p.cf_start_dt, p.cf_end_dt, p.fixing_dt,
            CAST(p.cf_yf AS FLOAT) AS cf_yf,
            ISNULL(CAST(p.fwd_rt AS FLOAT), 0.0) AS base_fwd,
            ISNULL(CAST(p.d_f AS FLOAT), 0.0) AS base_df,
            ISNULL(CAST(p.beh_outstanding AS FLOAT), 0.0) AS outstanding,
            ISNULL(CAST(p.beh_capital_pmt AS FLOAT), 0.0)
              + ISNULL(CAST(p.prepayment_pmt AS FLOAT), 0.0) AS capital,
            ISNULL(CAST(p.beh_interest_pmt AS FLOAT), 0.0) AS interest
        FROM cf.products p
        JOIN sched_key s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
                         AND CAST(p.product_type AS VARCHAR(1)) = s.src
                         AND CAST(p.product_code AS VARCHAR(4)) = s.product_code
        WHERE CAST(p.product_code AS VARCHAR(4)) IN ({_C_IN})
          AND p.cf_end_dt > :rd
          AND COALESCE(p.beh_total_pmt, 0) <> 0
    """)
    df = _try_query(q, {"rd": REPORT_DATE})
    if df.empty:
        return df

    df["product_code"] = df["product_code"].astype(str)
    for c in ["cf_start_dt", "cf_end_dt", "fixing_dt"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["m_idx", "cf_yf", "base_fwd", "base_df", "outstanding", "capital", "interest"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["m_idx"] = df["m_idx"].clip(lower=0).astype(int)
    if df.empty:
        return df

    denom = (df["outstanding"] * df["cf_yf"]).replace(0.0, np.nan)
    df["contracted_rt"] = (df["interest"] / denom).fillna(0.0)
    return df


def _compute_cohort_eve_pv(
    df: pd.DataFrame,
    curves: dict,
    ir_coeff: pd.DataFrame,
    scenario_ids: list[str],
    recompute_base: bool = False,
) -> tuple[np.ndarray, dict, list[str]]:
    """Per-cohort-group x month-bucket x scenario unsigned PV, from schedule df + curves.

    recompute_base=False (default -- today's production path): scenario 'base' reuses
    cf.products' stored interest/base_df (today's curve, baked in at cash-flow-generation
    time by cf_calc_workflow). Byte-identical to the original inline logic.

    recompute_base=True (anchor curve stress test -- see anchor_eve_reprice.py):
    scenario 'base' is ALSO recomputed via the same curve-lookup machinery as the
    shocked scenarios, against curves['base']. Needed whenever `curves` is a
    hypothetical (non-today) market curve, so the reference point for delta-EVE
    reflects that curve's own level, not today's baked-in cash flows -- otherwise
    shock deltas would be measured against the wrong base.
    """
    scen_all = ["base"] + list(scenario_ids)
    if df.empty:
        return np.zeros((0, 0, len(scen_all))), {}, scen_all

    pcs = df["product_code"].astype(str)
    a_v = pcs.map(ir_coeff["coeff_a"].to_dict()).fillna(1.0).to_numpy(dtype=float)
    floor_v = pcs.map(
        ir_coeff["client_floor"].dropna().to_dict()
        if "client_floor" in ir_coeff.columns else {}
    ).fillna(float("-inf")).to_numpy(dtype=float)
    cap_v = pcs.map(
        ir_coeff["client_cap"].dropna().to_dict()
        if "client_cap" in ir_coeff.columns else {}
    ).fillna(float("inf")).to_numpy(dtype=float)

    cf_yf = df["cf_yf"].to_numpy(dtype=float)
    base_fwd = df["base_fwd"].to_numpy(dtype=float)
    base_df = df["base_df"].to_numpy(dtype=float)
    outstanding = df["outstanding"].to_numpy(dtype=float)
    capital = df["capital"].to_numpy(dtype=float)
    interest = df["interest"].to_numpy(dtype=float)
    contracted = df["contracted_rt"].to_numpy(dtype=float)
    rate_type = df["rate_type"].fillna("V").astype(str).to_numpy()
    is_var = rate_type == "V"
    is_admin = rate_type == "A"
    start_before = (df["cf_start_dt"] < REPORT_DATE).to_numpy(dtype=bool)
    has_fixing = df["fixing_dt"].notna().to_numpy(dtype=bool)

    def _lookup_df(scenario: str, dates: pd.Series, *, nan_before: bool = False) -> np.ndarray:
        dates = pd.to_datetime(dates, errors="coerce")
        days = (dates - REPORT_DATE).dt.days.to_numpy(dtype=float)
        out = np.full(len(dates), np.nan if nan_before else 1.0, dtype=float)
        ccy_arr = df["currency"].astype(str).to_numpy()
        for ccy in np.unique(ccy_arr):
            mask = ccy_arr == ccy
            nd_ldf = curves.get(scenario, {}).get(_CCY_CURVE.get(ccy, "PLN_disc_curve"))
            if nd_ldf is None:
                continue
            nd, ldf = nd_ldf
            d = days[mask]
            vals = np.full(mask.sum(), np.nan if nan_before else 1.0, dtype=float)
            finite = np.isfinite(d)
            if finite.any():
                vals[finite] = np.exp(np.interp(d[finite], nd, ldf, left=ldf[0], right=ldf[-1]))
            if nan_before:
                vals[d < 0.0] = np.nan
            else:
                vals[d <= 0.0] = 1.0
            out[mask] = vals
        return out

    def _fwd_between(scenario: str, start: pd.Series, end: pd.Series, yf: np.ndarray) -> np.ndarray:
        df_s = _lookup_df(scenario, start, nan_before=False)
        df_e = _lookup_df(scenario, end, nan_before=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = np.where((df_e > 0.0) & (yf > 0.0), (df_s / df_e - 1.0) / yf, 0.0)
        return np.maximum(0.0, np.nan_to_num(fwd, nan=0.0, posinf=0.0, neginf=0.0))

    period_end = df["cf_end_dt"].copy()
    var_fix_mask = is_var & has_fixing
    if var_fix_mask.any():
        period_end.loc[var_fix_mask] = (
            df.loc[var_fix_mask]
              .groupby(["schedule_id", "product_type", "fixing_dt"])["cf_end_dt"]
              .transform("max")
        )
    period_start = df["cf_start_dt"].copy()
    period_start.loc[var_fix_mask] = df.loc[var_fix_mask, "fixing_dt"]
    period_yf = ((pd.to_datetime(period_end) - pd.to_datetime(period_start)).dt.days / 365.0).to_numpy(dtype=float)

    keys_df = df[_COHORT_KEY].drop_duplicates().reset_index(drop=True)
    key_to_gid = {
        (str(r.product_code), str(r.bs_side), str(r.currency), int(r.start_year), int(r.start_month)): i
        for i, r in keys_df.iterrows()
    }
    gid = np.array([
        key_to_gid[(str(r.product_code), str(r.bs_side), str(r.currency), int(r.start_year), int(r.start_month))]
        for _, r in df.iterrows()
    ], dtype=int)
    midx = df["m_idx"].to_numpy(dtype=int)
    n_groups = len(key_to_gid)
    n_buckets = int(midx.max()) + 1
    n_scen = len(scen_all)
    pv_all = np.zeros((n_groups, n_buckets, n_scen), dtype=float)

    base_df_eff = base_df.copy()
    missing_base_df = base_df_eff <= 0.0
    if missing_base_df.any():
        base_df_eff[missing_base_df] = _lookup_df("base", df["cf_end_dt"])[missing_base_df]

    # Reference forward rate for the shock-delta pivot (contracted + a*(fwd - ref)).
    # Default: today's stored base_fwd (cf.products, baked in by cf_calc_workflow).
    # recompute_base=True: the CURVE PASSED IN's own base level (curves['base']), so
    # that a hypothetical anchor curve's "base" is self-consistent with its shocks --
    # otherwise shock deltas would be measured against the wrong (today's) reference.
    if recompute_base:
        base_fwd_ref = _fwd_between("base", df["cf_start_dt"], df["cf_end_dt"], cf_yf)
    else:
        base_fwd_ref = base_fwd

    for s_idx, scen in enumerate(scen_all):
        if scen == "base" and not recompute_base:
            int_s = interest.copy()
            df_end = base_df_eff
        else:
            fwd_sh = _fwd_between(scen, df["cf_start_dt"], df["cf_end_dt"], cf_yf)
            locked = start_before.copy()
            if var_fix_mask.any():
                pstart_df = _lookup_df(scen, period_start, nan_before=True)
                pend_df = _lookup_df(scen, period_end, nan_before=True)
                valid = (pstart_df > 0.0) & (pend_df > 0.0) & (period_yf > 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    fwd_period = np.where(
                        valid,
                        (pstart_df / pend_df - 1.0) / period_yf,
                        0.0,
                    )
                fwd_period = np.maximum(0.0, np.nan_to_num(fwd_period, nan=0.0))
                fwd_sh[var_fix_mask] = np.where(
                    valid[var_fix_mask],
                    fwd_period[var_fix_mask],
                    base_fwd_ref[var_fix_mask],
                )
                locked[var_fix_mask] = ~valid[var_fix_mask]

            fwd_for_interest = np.where(locked, base_fwd_ref, fwd_sh)
            eff_rate = contracted + a_v * (fwd_for_interest - base_fwd_ref)
            eff_rate = np.minimum(cap_v, np.maximum(floor_v, eff_rate))

            int_s = interest.copy()
            int_s[is_var] = outstanding[is_var] * eff_rate[is_var] * cf_yf[is_var]
            lock_var = locked & is_var
            if lock_var.any():
                int_s[lock_var] = interest[lock_var]
            int_s[is_admin] = 0.0
            df_end = _lookup_df(scen, df["cf_end_dt"])

        pv = (capital + int_s) * df_end
        np.add.at(pv_all[:, :, s_idx], (gid, midx), pv)

    return pv_all, key_to_gid, scen_all


def _load_cohort_effective_eve_pv_tables(
    curves: dict,
    ir_coeff: pd.DataFrame,
    scenario_ids: list[str],
) -> dict:
    """Build scenario-specific monthly EVE PV tables from daily CF rows.

    Exact EVE is not just capital discounted on shocked curves.  Variable-rate
    rows also get shocked interest payments, while fixed rows keep contracted
    interest.  This table preserves that daily logic, then stores unsigned PV
    by cohort/month/scenario.  Runtime multiplies by sign and current balance.
    """
    df = _query_cohort_cf_schedule()
    if df.empty:
        return {}
    pv_all, key_to_gid, scen_all = _compute_cohort_eve_pv(
        df, curves, ir_coeff, scenario_ids, recompute_base=False
    )
    if pv_all.size == 0:
        return {}

    # Prefer the row-level EVE analytical tables when available.  They are
    # written by eve_calc_workflow with the exact shocked-interest logic, so
    # using them removes small reconstruction drift for products with special
    # fixing / rate-limit behaviour. NOTE: these tables are today-only (written
    # for REPORT_DATE by a separate upstream workflow) -- they are correctly
    # skipped when this function is reused for a hypothetical anchor curve
    # (see anchor_eve_reprice.py, which calls _compute_cohort_eve_pv directly
    # instead of this wrapper, so this override never applies there).
    exact_parts: list[pd.DataFrame] = []
    exact_sources = [
        ("eve_base_scenario",       ["base"]),
        ("eve_par_scenarios",       ["par_up", "par_dn"]),
        ("eve_short_scenarios",     ["sr_up", "sr_dn"]),
        ("eve_step_flat_scenarios", ["steep", "flat"]),
    ]
    for table_name, table_scenarios in exact_sources:
        scen_in = ", ".join(f"'{s}'" for s in table_scenarios if s in scen_all)
        if not scen_in:
            continue
        exact_q = text(f"""
            WITH sched_key AS (
                SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id, 'L' AS src,
                       CAST(product_code AS VARCHAR(4)) AS product_code,
                       bs_side, currency, start_date
                FROM sched.loans
                WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
                UNION ALL
                SELECT CAST(schedule_id AS VARCHAR(8)), 'D',
                       CAST(product_code AS VARCHAR(4)), bs_side, currency, start_date
                FROM sched.deposits
                WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
                UNION ALL
                SELECT CAST(schedule_id AS VARCHAR(8)), 'F',
                       CAST(product_code AS VARCHAR(4)), bs_side, currency, start_date
                FROM sched.fin_inst
                WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            )
            SELECT
                e.scenario_id,
                s.product_code, s.bs_side, s.currency,
                YEAR(s.start_date) AS start_year,
                MONTH(s.start_date) AS start_month,
                e.cf_end_dt,
                CASE WHEN e.bs_side = 'A' THEN 1.0 ELSE -1.0 END
                  * (ISNULL(CAST(e.pv_capital AS FLOAT), 0.0)
                     + ISNULL(CAST(e.pv_interest AS FLOAT), 0.0)) AS pv_unsigned
            FROM cf.{table_name} e
            JOIN sched_key s ON CAST(e.schedule_id AS VARCHAR(8)) = s.schedule_id
                             AND CAST(e.product_type AS VARCHAR(1)) = s.src
                             AND CAST(e.product_code AS VARCHAR(4)) = s.product_code
            WHERE e.report_date = :rd
              AND CAST(e.product_code AS VARCHAR(4)) IN ({_C_IN})
              AND e.scenario_id IN ({scen_in})
              AND e.cf_end_dt > :rd
        """)
        part = _try_query(exact_q, {"rd": REPORT_DATE})
        if not part.empty:
            exact_parts.append(part)

    if exact_parts:
        exact_df = pd.concat(exact_parts, ignore_index=True)
        exact_df["product_code"] = exact_df["product_code"].astype(str)
        exact_df["cf_end_dt"] = pd.to_datetime(exact_df["cf_end_dt"], errors="coerce")
        exact_df["m_idx"] = (
            (exact_df["cf_end_dt"].dt.year - REPORT_DATE.year) * 12
            + (exact_df["cf_end_dt"].dt.month - REPORT_DATE.month)
            - 1
        ).clip(lower=0).astype(int)
        exact_df["pv_unsigned"] = pd.to_numeric(
            exact_df["pv_unsigned"], errors="coerce"
        ).fillna(0.0)
        max_exact_m = int(exact_df["m_idx"].max()) if not exact_df.empty else -1
        if max_exact_m >= pv_all.shape[1]:
            pad = max_exact_m + 1 - pv_all.shape[1]
            pv_all = np.pad(pv_all, ((0, 0), (0, pad), (0, 0)), mode="constant")

        exact_grp = (
            exact_df.groupby(_COHORT_KEY + ["scenario_id", "m_idx"], as_index=False)["pv_unsigned"]
            .sum()
        )
        for (pc, side, ccy, sy, sm, scen), sub in exact_grp.groupby(_COHORT_KEY + ["scenario_id"]):
            ck = (str(pc), str(side), str(ccy), int(sy), int(sm))
            g = key_to_gid.get(ck)
            if g is None or scen not in scen_all:
                continue
            s_idx = scen_all.index(str(scen))
            pv_all[g, :, s_idx] = 0.0
            np.add.at(
                pv_all[g, :, s_idx],
                sub["m_idx"].to_numpy(dtype=int),
                sub["pv_unsigned"].to_numpy(dtype=float),
            )

    eve_pv_by: dict = {}
    for ck, g in key_to_gid.items():
        months = np.flatnonzero(np.any(np.abs(pv_all[g]) > 1e-10, axis=1))
        if len(months) == 0:
            continue
        eve_pv_by[ck] = (months.astype(int), pv_all[g, months, :].copy())
    return eve_pv_by


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


def _load_irs_outstanding_m() -> dict:
    """Compute monthly outstanding fractions for IRS floating legs.

    For each (product_code, bs_side, currency) group of floating IRS legs:
        outstanding_m[m] = Σ notional_i / total_notional
                           for swaps i that are floating (past next fixing) in month m.

    Locked months (before the next repricing date) and matured swaps are excluded
    from outstanding, so they contribute zero delta NII in the CF-based B model.

    Returns dict: {(pc, sid, ccy): (outstanding_m ndarray, dominant_F_months)}
    """
    q = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code, bs_side, currency,
               CAST(notional AS FLOAT) AS notional,
               start_date, maturity_date, fixing_freq
        FROM schemat.ir_swaps
        WHERE report_date = :rd
          AND leg_type = 'FLOAT'
          AND CAST(product_code AS VARCHAR(4)) IN ('0000')
    """)
    df = _try_query(q, {"rd": REPORT_DATE})
    if df.empty:
        return {}

    df["product_code"] = df["product_code"].astype(str)
    df["notional"]     = df["notional"].astype(float)
    df["start_date"]   = pd.to_datetime(df["start_date"],   errors="coerce")
    df["maturity_date"]= pd.to_datetime(df["maturity_date"],errors="coerce")

    _freq_to_m = {"3M": 3, "6M": 6, "1Y": 12}
    df["fix_m"] = df["fixing_freq"].map(_freq_to_m).fillna(3).astype(int)

    result: dict = {}
    for (pc, sid, ccy), grp in df.groupby(["product_code", "bs_side", "currency"]):
        total_n = grp["notional"].sum()
        if total_n <= 0:
            continue
        outstanding_m = np.zeros(12)
        freq_notional: dict = {}
        for _, row in grp.iterrows():
            n   = float(row["notional"])
            F   = int(row["fix_m"])
            mat = row["maturity_date"]
            start = row["start_date"]
            freq_notional[F] = freq_notional.get(F, 0.0) + n

            if pd.isna(start) or pd.isna(mat):
                outstanding_m += n / total_n  # unknown schedule: assume full
                continue

            months_to_mat = (mat.year  - REPORT_DATE.year)  * 12 + \
                             (mat.month - REPORT_DATE.month)
            months_since   = (REPORT_DATE.year  - start.year)  * 12 + \
                             (REPORT_DATE.month - start.month)
            t_first = F - (months_since % F)
            if t_first == F:
                t_first = F  # just repriced; next fixing in F months

            for m in range(12):
                if m + 1 > months_to_mat:
                    continue  # swap matured before this month
                if m + 1 <= t_first:
                    continue  # still locked until next fixing
                outstanding_m[m] += n / total_n

        dominant_F = max(freq_notional, key=freq_notional.get) if freq_notional else 3
        result[(str(pc), str(sid), str(ccy))] = (outstanding_m, int(dominant_F))

    return result


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

    print("Loading fin_data (CoC, CET1)...")
    fin_params  = _load_fin_data()
    coc_rate    = float(fin_params.get("CoC",  0.10))
    cet1_target = float(fin_params.get("CET1", 0.12))
    print(f"  CoC={coc_rate*100:.1f}%  CET1_target={cet1_target*100:.1f}%")

    print("Loading subst_matrix (substitution/cannibalism pairs)...")
    subst_df = _load_subst_matrix()
    print(f"  {len(subst_df)} substitution pairs loaded")

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
    (
        monthly_out,
        monthly_cap,
        monthly_locked_rt,
        monthly_t_first,
        monthly_locked_frac,
        monthly_locked_rate_m,
        monthly_float_base_eff_m,
        monthly_float_base_fwd_m,
    ) = _load_cohort_monthly_schedule()
    print(f"  {len(monthly_out)} cohort schedule groups loaded")

    print("Building effective monthly NII tables from daily CFs...")
    (
        monthly_interest_yf,
        monthly_capital_remain,
        monthly_effective_rate,
        monthly_effective_renewal_rate,
    ) = _load_cohort_effective_nii_tables(disc_curves, ir_coeff, SHOCKED_SCENARIO_IDS)
    print(f"  {len(monthly_effective_rate)} effective NII schedule groups loaded")

    print("Building effective monthly EVE PV tables from daily CFs...")
    monthly_effective_eve_pv = _load_cohort_effective_eve_pv_tables(
        disc_curves, ir_coeff, SHOCKED_SCENARIO_IDS
    )
    print(f"  {len(monthly_effective_eve_pv)} effective EVE schedule groups loaded")

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

    print("Computing IRS monthly outstanding profiles from schemat.ir_swaps...")
    irs_out_profiles = _load_irs_outstanding_m()
    print(f"  IRS outstanding profiles: {list(irs_out_profiles.keys())}")

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
        rwa_f  = float(_meta(pc, sid, ccy, "rwa_weight",  0.0) or 0.0) if sid == "A" else 0.0

        # Credit risk parameters (EL = PD × LGD × EAD; assets only)
        pd_r  = float(_meta(pc, sid, ccy, "PD",  0.0) or 0.0) if sid == "A" else 0.0
        lgd_r = float(_meta(pc, sid, ccy, "LGD", 0.0) or 0.0) if sid == "A" else 0.0

        # Price-volume elasticity: NII rate change per 1pp weight change (all sides)
        vol_elast = float(_meta(pc, sid, ccy, "vol_elasticity", 0.0) or 0.0)

        # Non-interest fee income: flat yearly rate per PLN balance (all sides --
        # 0.0 for every product without an explicit fee_unit_rate in bs_structure)
        fee_r = float(_meta(pc, sid, ccy, "fee_unit_rate", 0.0) or 0.0)

        # Marketing/acquisition cost: yearly rate applied only to growth ABOVE
        # baseline weight in the optimizer (all sides, 0.0 by default)
        acq_r = float(_meta(pc, sid, ccy, "acq_cost_rate", 0.0) or 0.0)

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

        # ── Analytical fallback for monthly NII schedule ──────────────────────
        # cf.products only stores 1-2 months of CFs for long-term loans, so
        # outstanding_m and capital_m would be mostly zero for products like
        # fixed-rate mortgages.  If SQL coverage is < 50% of the 12-month
        # horizon, regenerate both arrays from an analytical annuity schedule.
        # Rate used: contracted coupon for fixed-rate; coeff_b (margin) for
        # floating (their repricing is captured via rate_matrix, not coupon).
        # Condition also excludes products not in the NII pipeline (nii_unit=0).
        # ── Constant-balance extension for NII outstanding ────────────────────
        # The exact nii_results uses a gap-based methodology:
        #     delta_NII = gap × Δr × remain_yf  (per repricing tenor bucket)
        # This assumes each product's FULL BALANCE stays outstanding for the
        # entire 12-month NII horizon (rolled over at repriced rates after each
        # repricing event).  cf.products only covers the CURRENT repricing
        # period (1-3 months for floating, 2 months from DB for long-term fixed),
        # so outstanding_m has zeros for the remaining months.
        # Extend to constant balance whenever SQL gives < 6 months of coverage.
        # rate_matrix already switches to the market forward rate at t_first_m,
        # so the run-off NII = constant_balance × rate_matrix replicates the gap
        # formula exactly without needing a separate renewal term.
        _sql_out_0   = float(_out_m[0]) if np.any(_out_m > 0) else 0.0
        _sql_nonzero = int(np.count_nonzero(_out_m))
        # SQL aggregates ALL loans in a vintage pool; if the first-month
        # outstanding is >10× the individual-cohort balance the data is
        # mis-scaled and must be replaced with an analytical schedule.
        _sql_bad_scale = bal > 0 and _sql_out_0 > 10.0 * bal
        _has_real_monthly_schedule = _sql_nonzero >= 6 and not _sql_bad_scale

        if bal > 0 and nii_unit != 0.0 and (_sql_nonzero < 6 or _sql_bad_scale):
            if rep_m >= 12:
                if rate_typ == "F":
                    # Fixed-rate with long remaining term: bullet assumption.
                    # The bond matures after the 12M horizon — full balance
                    # outstanding for all 12 months, no capital within horizon.
                    # Annuity profile is wrong here because fixed-rate bonds
                    # in this bank are bullet instruments, not amortising.
                    _out_m = np.full(12, bal)
                    _cap_m = np.zeros(12)
                else:
                    # Floating/amortising: derive from analytical annuity.
                    _ann_coupon = coupon_r if not np.isnan(coupon_r) else max(coeff_b, 0.0)
                    _out_m, _cap_m = _analytical_monthly_profile(bal, _ann_coupon, int(rep_m))
            else:
                # Short-term / bullet: constant outstanding.
                _out_m = np.full(12, bal)
                if _sql_bad_scale:
                    _cap_m = np.zeros(12)   # reset mis-scaled SQL capital

        # Floating products already roll the whole balance through rate_matrix,
        # so adding monthly capital renewal double-counts repricing unless the
        # product is explicitly modelled as an amortising runoff bucket.
        _keep_float_renewal = (
            rate_typ == "V"
            and sid == "A"
            and pc in _AMORTISING_FLOAT_RENEWAL_PRODUCTS
            and _has_real_monthly_schedule
        )
        if rate_typ == "V" and not _keep_float_renewal:
            _cap_m = np.zeros(12)
        else:
            # Bullet capital fill: for SHORT-TERM FIXED-RATE products where SQL
            # records no amortisation, place the full repayment in the maturity
            # month so the renewal stream captures reinvestment at new market rates.
            if (rep_m < 12 and bal > 0 and nii_unit != 0.0
                    and np.sum(np.abs(_cap_m)) < 0.01 * bal):
                _cap_m = np.zeros(12)
                _cap_m[min(int(rep_m) - 1, 11)] = bal

        _interest_yf_m = _out_m / 12.0
        _cap_remain_m = _cap_m * np.array([(12 - m - 0.5) / 12.0 for m in range(12)], dtype=float)
        _eff_rate_m = monthly_effective_rate.get(ck)
        _eff_renewal_rate_m = monthly_effective_renewal_rate.get(ck)
        if _eff_rate_m is not None and _has_real_monthly_schedule:
            _interest_yf_m = monthly_interest_yf.get(ck, _interest_yf_m)
        if _eff_renewal_rate_m is not None and _keep_float_renewal:
            _cap_remain_m = monthly_capital_remain.get(ck, _cap_remain_m)
        if rate_typ == "V" and not _keep_float_renewal:
            _cap_remain_m = np.zeros(12)

        out_frac_m = _out_m / bal if bal > 0 else np.zeros(12)
        cap_frac_m = _cap_m / bal if bal > 0 else np.zeros(12)
        interest_yf_frac_m = _interest_yf_m / bal if bal > 0 else np.zeros(12)
        cap_remain_frac_m = _cap_remain_m / bal if bal > 0 else np.zeros(12)
        locked_rt   = monthly_locked_rt.get(ck, 0.0)
        t_first_m   = monthly_t_first.get(ck, 999.0)
        locked_frac_m = monthly_locked_frac.get(ck, np.zeros(12))
        locked_rate_m = monthly_locked_rate_m.get(ck, np.zeros(12))
        float_base_eff_m = monthly_float_base_eff_m.get(ck, np.zeros(12))
        float_base_fwd_m = monthly_float_base_fwd_m.get(ck, np.zeros(12))
        effective_rate_m = (
            np.asarray(_eff_rate_m, dtype=float)
            if _eff_rate_m is not None and _has_real_monthly_schedule
            else np.zeros((12, len(["base"] + SHOCKED_SCENARIO_IDS)), dtype=float)
        )
        effective_renewal_rate_m = (
            np.asarray(_eff_renewal_rate_m, dtype=float)
            if _eff_renewal_rate_m is not None and _keep_float_renewal
            else np.zeros((12, len(["base"] + SHOCKED_SCENARIO_IDS)), dtype=float)
        )

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
            "rwa_factor":          rwa_f,
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
            "pd_rate":             pd_r,
            "lgd_rate":            lgd_r,
            "vol_elasticity":      vol_elast,
            "fee_unit_rate":       fee_r,
            "acq_cost_rate":       acq_r,
            # schedule-based monthly profile
            "cohort_outstanding_m": out_frac_m,
            "cohort_capital_m":     cap_frac_m,
            "cohort_interest_yf_m":  interest_yf_frac_m,
            "cohort_capital_remain_m": cap_remain_frac_m,
            "cohort_locked_rate":   locked_rt,
            "cohort_t_first_m":     t_first_m,
            "cohort_locked_frac_m":  locked_frac_m,
            "cohort_locked_rate_m":  locked_rate_m,
            "cohort_float_base_eff_m": float_base_eff_m,
            "cohort_float_base_fwd_m": float_base_fwd_m,
            "cohort_effective_rate_m": effective_rate_m,
            "cohort_effective_renewal_rate_m": effective_renewal_rate_m,
            "cohort_has_real_monthly_schedule": bool(_has_real_monthly_schedule),
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

    # Products whose NII is rate-sensitive in the exact IRRBB model.
    # Only these get cohort_outstanding_m = ones(12) for approach B.
    # Equity, non-rate products, and IRS legs with zero exact delta stay at zeros.
    _sng_nii_base_flat = {
        (str(r["product_code"]), str(r["bs_side"]), str(r["currency"])): float(r["nii_total"])
        for _, r in sng_nii[sng_nii["scenario_id"] == "base"].iterrows()
    }
    sng_has_delta: set = set()
    for _, r in sng_nii[sng_nii["scenario_id"] != "base"].iterrows():
        k3 = (str(r["product_code"]), str(r["bs_side"]), str(r["currency"]))
        nii_b_k = _sng_nii_base_flat.get(k3, 0.0)
        if abs(float(r["nii_total"]) - nii_b_k) > 1.0:
            sng_has_delta.add(k3)

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
        hqla_f    = _meta(pc, sid, ccy, "hqla_factor",   0.0) or 0.0 if sid == "A" else 0.0
        lcr_r     = _meta(pc, sid, ccy, "LCR",           0.0) or 0.0 if sid == "L" else 0.0
        asf_f     = _meta(pc, sid, ccy, "ASF",           0.0) or 0.0 if sid in ("L","E") else 0.0
        rsf_f     = _meta(pc, sid, ccy, "RSF",           0.0) or 0.0 if sid == "A" else 0.0
        rwa_f     = _meta(pc, sid, ccy, "rwa_weight",    0.0) or 0.0 if sid == "A" else 0.0
        pd_r      = float(_meta(pc, sid, ccy, "PD",      0.0) or 0.0) if sid == "A" else 0.0
        lgd_r     = float(_meta(pc, sid, ccy, "LGD",     0.0) or 0.0) if sid == "A" else 0.0
        vol_elast = float(_meta(pc, sid, ccy, "vol_elasticity", 0.0) or 0.0)
        fee_r     = float(_meta(pc, sid, ccy, "fee_unit_rate", 0.0) or 0.0)
        acq_r     = float(_meta(pc, sid, ccy, "acq_cost_rate", 0.0) or 0.0)

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

        # IRS products: use swap-schedule-derived outstanding fractions and
        # dominant fixing frequency instead of generic ones(12).
        # Non-IRS single-row products with rate sensitivity get ones(12).
        _irs_key = (pc, sid, ccy)
        if _irs_key in irs_out_profiles:
            _sng_out_m, _sng_rep_m = irs_out_profiles[_irs_key]
            _sng_t1 = 0.0          # locking encoded in outstanding_m; no virtual lock
        elif (pc, sid, ccy) in sng_has_delta:
            _sng_out_m = np.ones(12)
            _sng_rep_m = 12.0
            _sng_t1    = 999.0
        else:
            _sng_out_m = np.zeros(12)
            _sng_rep_m = 12.0
            _sng_t1    = 999.0

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
            "rwa_factor":          rwa_f,
            "inflow_30d_frac":     0.0,
            "amort_frac_1y":       0.0,
            "pd_rate":             pd_r,
            "lgd_rate":            lgd_r,
            "vol_elasticity":      vol_elast,
            "fee_unit_rate":       fee_r,
            "acq_cost_rate":       acq_r,
            "repricing_tenor_m":   float(_sng_rep_m),
            "rate_type":           None,
            "coupon_rate":         None,
            "nii_reprice_frac":    0.0,
            "coeff_a":             coeff_a,
            "coeff_b":             coeff_b,
            "client_floor":        cli_floor,
            "client_cap":          cli_cap,
            "cohort_outstanding_m": _sng_out_m,
            "cohort_capital_m":    np.zeros(12),
            "cohort_interest_yf_m": _sng_out_m / 12.0,
            "cohort_capital_remain_m": np.zeros(12),
            "cohort_locked_rate":  0.0,
            "cohort_t_first_m":    _sng_t1,
            "cohort_locked_frac_m": np.zeros(12),
            "cohort_locked_rate_m": np.zeros(12),
            "cohort_float_base_eff_m": np.zeros(12),
            "cohort_float_base_fwd_m": np.zeros(12),
            "cohort_effective_rate_m": np.zeros((12, len(["base"] + SHOCKED_SCENARIO_IDS))),
            "cohort_effective_renewal_rate_m": np.zeros((12, len(["base"] + SHOCKED_SCENARIO_IDS))),
            "cohort_has_real_monthly_schedule": False,
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
    cohort_interest_yf_m = np.vstack([
        r if r is not None else np.zeros(12)
        for r in params_df["cohort_interest_yf_m"]
    ]).astype(float)
    cohort_capital_remain_m = np.vstack([
        r if r is not None else np.zeros(12)
        for r in params_df["cohort_capital_remain_m"]
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
            _eff_rm_i = np.asarray(
                all_rows[i].get("cohort_effective_rate_m", np.zeros((12, _cr_n_scen))),
                dtype=float,
            )
            if (
                _eff_rm_i.shape == (12, _cr_n_scen)
                and np.any(cohort_interest_yf_m[i] > 1e-12)
                and np.any(_eff_rm_i > 0.0)
            ):
                _rate_matrix[i, :, :] = _eff_rm_i
                continue

            if rt == "F":
                # Fixed-rate products: rate is locked at the contracted coupon
                # for the entire outstanding period.  No scenario sensitivity
                # arises from the locked coupon → zero delta NII from run-off.
                # (Repricing/renewal delta would require knowing the actual
                # maturity distribution per cohort, which is not available here.)
                _rate_matrix[i, :, :] = _coupon_arr[i]
            else:
                t1 = int(_t1_arr[i])
                F  = float(_F_arr[i])
                ca = _ca_arr[i]
                fl = _fl_arr[i];  cp = _cp_arr[i]
                base_eff = _locked_arr[i]   # contracted rate for locked CFs

                # Cohort-specific margin for the floating period.
                cb_float = float(_cb_arr[i])
                if all_rows[i].get("is_cohort") and rt != "F":
                    _ck_i = (str(all_rows[i]["product_code"]),
                             str(all_rows[i]["bs_side"]),
                             str(all_rows[i]["currency"]),
                             int(all_rows[i].get("start_year") or 0),
                             int(all_rows[i].get("start_month") or 0))
                    cb_float = cohort_float_margins.get(_ck_i, cb_float)

                # When t_first is missing (SQL default 999) AND the locked rate
                # is zero, derive t_first from cohort start date + repricing tenor.
                # For a floating bond repricing every F months, the next fixing is:
                #   t1 = F - (months_since_start % F)   [or 0 if months_since_start is divisible by F]
                # This avoids blanket t1=0 (Fix 2) which overestimates sensitivity
                # by treating all 12 months as floating when only the post-t1 months are.
                if t1 >= 12 and ca > 0.0 and abs(base_eff) < 1e-10:
                    s_year  = int(all_rows[i].get("start_year") or 0)
                    s_month = int(all_rows[i].get("start_month") or 0)
                    if s_year > 0 and s_month > 0 and F >= 2:
                        rep_m   = REPORT_DATE.year * 12 + REPORT_DATE.month
                        sta_m   = s_year * 12 + s_month
                        into_F  = int(rep_m - sta_m) % int(F)
                        t1      = 0 if into_F == 0 else int(F) - into_F
                    else:
                        t1 = 0   # F=1 (monthly) or unknown start: reprice immediately
                    base_eff = np.clip(
                        ca * _fwd_F(base_disc, max(1, t1), int(F)) + cb_float, fl, cp
                    )

                lock_frac_m = np.asarray(
                    all_rows[i].get("cohort_locked_frac_m", np.zeros(12)),
                    dtype=float,
                )
                lock_rate_m = np.asarray(
                    all_rows[i].get("cohort_locked_rate_m", np.zeros(12)),
                    dtype=float,
                )
                float_base_eff_m = np.asarray(
                    all_rows[i].get("cohort_float_base_eff_m", np.zeros(12)),
                    dtype=float,
                )
                float_base_fwd_m = np.asarray(
                    all_rows[i].get("cohort_float_base_fwd_m", np.zeros(12)),
                    dtype=float,
                )
                if lock_frac_m.shape != (12,) or not np.isfinite(lock_frac_m).all():
                    lock_frac_m = np.zeros(12)
                if lock_rate_m.shape != (12,) or not np.isfinite(lock_rate_m).all():
                    lock_rate_m = np.zeros(12)
                if float_base_eff_m.shape != (12,) or not np.isfinite(float_base_eff_m).all():
                    float_base_eff_m = np.zeros(12)
                if float_base_fwd_m.shape != (12,) or not np.isfinite(float_base_fwd_m).all():
                    float_base_fwd_m = np.zeros(12)

                use_monthly_lock_mix = (
                    str(rt) == "V"
                    and bool(all_rows[i].get("cohort_has_real_monthly_schedule", False))
                    and np.any(lock_frac_m > 1e-8)
                )

                if use_monthly_lock_mix:
                    for m in range(12):
                        lf = float(np.clip(lock_frac_m[m], 0.0, 1.0))
                        lr = float(lock_rate_m[m]) if lock_rate_m[m] > 0 else float(base_eff)

                        if m < t1:
                            ev_t = max(1, t1)
                        else:
                            k = int((m - t1) / F)
                            ev_t = int(t1 + k * F)
                        fwd_b = _fwd_F(base_disc, ev_t, F)
                        fb_m = float(float_base_fwd_m[m]) if float_base_fwd_m[m] > 0 else fwd_b
                        be_m = (
                            float(float_base_eff_m[m])
                            if float_base_eff_m[m] > 0
                            else np.clip(ca * fb_m + cb_float, fl, cp)
                        )
                        float_base = np.clip(be_m + ca * (fwd_b - fb_m), fl, cp)
                        _rate_matrix[i, m, 0] = lf * lr + (1.0 - lf) * float_base
                        for rs, scen in enumerate(_cr_rate_scen[1:], start=1):
                            if scen not in _ct_scenarios:
                                _rate_matrix[i, m, rs] = _rate_matrix[i, m, 0]
                                continue
                            sh_disc = ct.disc_factors[_ct_scenarios.index(scen), ci]
                            fwd_sh = _fwd_F(sh_disc, ev_t, F)
                            float_sh = np.clip(be_m + ca * (fwd_sh - fb_m), fl, cp)
                            _rate_matrix[i, m, rs] = lf * lr + (1.0 - lf) * float_sh
                else:
                    for m in range(12):
                        if m < t1:
                            _rate_matrix[i, m, :] = base_eff
                        else:
                            k      = int((m - t1) / F)
                            ev_t   = int(t1 + k * F)
                            fwd_b  = _fwd_F(base_disc, ev_t, F)
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
        # Use the same accrual-period convention as the daily NII engine: capital
        # paid in bucket m inherits the one-month forward over month m.
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
            _eff_rr_i = np.asarray(
                all_rows[i].get("cohort_effective_renewal_rate_m", np.zeros((12, _cr_n_scen))),
                dtype=float,
            )
            if (
                _eff_rr_i.shape == (12, _cr_n_scen)
                and np.any(cohort_capital_remain_m[i] > 1e-12)
                and np.any(_eff_rr_i > 0.0)
            ):
                _renewal_rate_matrix[i, :, :] = _eff_rr_i
                continue
            for m in range(12):
                fwd_b_m = _fwd_F(base_disc_i, m, 1)
                _renewal_rate_matrix[i, m, 0] = np.clip(ca * fwd_b_m + cb, fl, cp)
                for rs, scen in enumerate(_cr_rate_scen[1:], start=1):
                    if scen not in _ct_scenarios:
                        _renewal_rate_matrix[i, m, rs] = _renewal_rate_matrix[i, m, 0]
                        continue
                    sh_disc = ct.disc_factors[_ct_scenarios.index(scen), ci]
                    fwd_sh_m = _fwd_F(sh_disc, m, 1)
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
        _cf_grps: list = []        # one DataFrame (or None) per row index
        _eve_eff_grps: list = []   # optional (month_idx, pv_by_scenario) per row
        _BUCKET_M = 1              # monthly buckets; keep timing close to exact EVE

        _rep_m_arr = params_df["repricing_tenor_m"].fillna(12.0).to_numpy(dtype=float)
        _cpn_arr2  = np.where(params_df["coupon_rate"].notna(),
                              params_df["coupon_rate"].to_numpy(dtype=float), 0.0)

        for i, row in enumerate(all_rows):
            if not row["is_cohort"]:
                _cf_grps.append(None)
                _eve_eff_grps.append(None)
                continue
            ck  = (row["product_code"], row["bs_side"], row["currency"],
                   int(row["start_year"]), int(row["start_month"]))
            eff_eve = monthly_effective_eve_pv.get(ck)
            _eve_eff_grps.append(eff_eve)
            if eff_eve is not None:
                _cf_grps.append(None)
                continue
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
            [len(g) for g in _cf_grps if g is not None and not g.empty]
            + [len(e[0]) for e in _eve_eff_grps if e is not None]
            + [0]
        )
        _max_q = max(_max_q, 1)
        _cf_cap  = np.zeros((n, _max_q), dtype=float)
        _cf_tot  = np.zeros((n, _max_q), dtype=float)
        _cf_yf   = np.zeros((n, _max_q), dtype=float)
        _cf_nq   = np.zeros(n, dtype=int)
        _eve_pv_frac = np.zeros((n, _max_q, _cr_n_scen), dtype=float)

        for i, row in enumerate(all_rows):
            if not row["is_cohort"]:
                continue
            eff_eve = _eve_eff_grps[i]
            bal = float(row["balance_amt"])
            if eff_eve is not None and bal > 0:
                m_idx_eff, pv_eff = eff_eve
                m_idx_eff = np.asarray(m_idx_eff, dtype=int)
                pv_eff = np.asarray(pv_eff, dtype=float)
                nq = min(len(m_idx_eff), _max_q)
                if nq <= 0:
                    continue
                _cf_nq[i] = nq
                _cf_yf[i, :nq] = (m_idx_eff[:nq] + 0.5) / 12.0
                _cf_tot[i, :nq] = pv_eff[:nq, 0] / bal
                _eve_pv_frac[i, :nq, :] = pv_eff[:nq, :] / bal
                continue
            grp = _cf_grps[i]
            if grp is None or grp.empty:
                continue
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
        _eve_pv_frac          = np.zeros((n, 1, 1), dtype=float)

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
        rwa_factor        = params_df["rwa_factor"].to_numpy(dtype=float),
        pd_rate           = params_df["pd_rate"].fillna(0.0).to_numpy(dtype=float),
        lgd_rate          = params_df["lgd_rate"].fillna(0.0).to_numpy(dtype=float),
        el_unit           = (params_df["pd_rate"].fillna(0.0) * params_df["lgd_rate"].fillna(0.0)).to_numpy(dtype=float),
        fee_unit_rate     = params_df["fee_unit_rate"].fillna(0.0).to_numpy(dtype=float),
        acq_cost_rate     = params_df["acq_cost_rate"].fillna(0.0).to_numpy(dtype=float),
        coc_rate          = np.array([coc_rate]),
        cet1_target       = np.array([cet1_target]),
        # Price-volume elasticity: rate change per 1pp of product weight (all sides)
        vol_elasticity         = params_df["vol_elasticity"].fillna(0.0).to_numpy(dtype=float),
        # New-business NII rate: equals book rate for now (FTP data needed for market-rate adjustment)
        nii_unit_rate_new_biz  = nii_unit_rate,
        # Substitution/cannibalism pairs (k pairs)
        subst_src_pc           = subst_df["source_code"].to_numpy(dtype=object) if len(subst_df) else np.array([], dtype=object),
        subst_src_side         = subst_df["source_side"].to_numpy(dtype=object) if len(subst_df) else np.array([], dtype=object),
        subst_dst_pc           = subst_df["dest_code"].to_numpy(dtype=object)   if len(subst_df) else np.array([], dtype=object),
        subst_dst_side         = subst_df["dest_side"].to_numpy(dtype=object)   if len(subst_df) else np.array([], dtype=object),
        subst_rates            = subst_df["subst_rate"].to_numpy(dtype=float)   if len(subst_df) else np.array([], dtype=float),
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
        cohort_interest_yf_m = cohort_interest_yf_m,
        cohort_capital_remain_m = cohort_capital_remain_m,
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
        cr_eve_pv_frac        = _eve_pv_frac,
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
