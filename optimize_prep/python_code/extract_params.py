"""extract_params.py
===================
One-time extraction from SQL + Excel → product_params.npz

Run this script once (or whenever the balance sheet structure or market data
change materially) to produce the frozen parameter snapshot used by the fast
metric functions in nii_fast, eve_fast, lcr_fast, nsfr_fast.

The output file (product_params.npz) is indexed by bs_structure row order.
Every array has length n = number of rows in bs_structure.

Dependencies
------------
* Requires that the following pipelines have been run first:
    balance_generate → balance_gen_add_data → cash_flow_calc
    → ir_derivatives → nii_calc_workflow → eve_calc_workflow
* Reads from:
    - balance_generate/input_data/bank_data.xlsx  (bs_structure sheet)
    - balance_generate/input_data/interest_rt.xlsx
    - SQL: schemat.loans, schemat.deposits, schemat.fin_inst
    - SQL: cf.products
    - SQL: irrbb.nii_results  (base + all shocked scenarios)
    - SQL: irrbb.eve_results  (base + par_up + par_dn)
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ── path setup so we can reuse the engine connection ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "irrbb_calc", "python_code"))

from sqlalchemy import create_engine, text

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

# ── configuration ──────────────────────────────────────────────────────────────
REPORT_DATE    = pd.to_datetime("2024-12-31")
TOTAL_ASSETS   = 10_000_000_000
HORIZON_DAYS   = 365
HORIZON_30D    = 30
PAR_SHOCK_RATE = 0.02          # PLN EBA par shock = 200 bps, used for duration calc
HORIZON_END    = REPORT_DATE + pd.Timedelta(days=HORIZON_DAYS)
HORIZON_30D_END = REPORT_DATE + pd.Timedelta(days=HORIZON_30D)

BS_PATH       = os.path.join(ROOT_DIR, "balance_generate", "input_data", "bank_data.xlsx")
INTEREST_PATH = os.path.join(ROOT_DIR, "balance_generate", "input_data", "interest_rt.xlsx")
OUT_PATH      = os.path.join(BASE_DIR, "..", "output", "product_params.npz")

# EBA SOT scenario IDs stored in irrbb.nii_results / eve_results
SHOCKED_SCENARIO_IDS = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Balance sheet structure
# ─────────────────────────────────────────────────────────────────────────────

def _load_bs_structure() -> pd.DataFrame:
    """Load bs_structure from Excel.

    Adds derived columns:
      sign     : +1 for assets, -1 for liabilities/equity
      full_pct : same as bs_percentage (alias for clarity)
    """
    df = pd.read_excel(BS_PATH, sheet_name="bs_structure")
    df["sign"]     = df["bs_side"].map({"A": 1.0, "L": -1.0, "E": -1.0}).fillna(-1.0)
    df["full_pct"] = df["bs_percentage"].fillna(0.0)
    df["product_code"] = df["product_code"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rate coefficients from interest_rt.xlsx
# ─────────────────────────────────────────────────────────────────────────────

def _load_rate_coefficients() -> pd.DataFrame:
    """Load per-product rate coefficients.

    Returns DataFrame indexed by product_code (str) with columns:
        coeff_a, coeff_b (decimal), client_floor (decimal), client_cap (decimal)
    """
    df = pd.read_excel(INTEREST_PATH)
    df["product_code"] = df["product_code"].astype(int).astype(str)
    # b is stored in percent points in the file → convert to decimal
    if "b" in df.columns:
        df["b"] = df["b"] / 100.0
    rename = {"a": "coeff_a", "b": "coeff_b", "client_floor": "client_floor",
              "client_cap": "client_cap"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ("coeff_a", "coeff_b", "client_floor", "client_cap"):
        if col not in df.columns:
            df[col] = np.nan
    df = df.set_index("product_code")[["coeff_a", "coeff_b", "client_floor", "client_cap"]]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Balance per product_code from schemat tables
# ─────────────────────────────────────────────────────────────────────────────

def _load_balance_by_product() -> pd.DataFrame:
    """Total balance per (product_code, bs_side, currency) from schemat tables.

    Returns DataFrame with columns: product_code (str), bs_side, currency, balance_amt.
    """
    query = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency,
               SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
        FROM schemat.loans
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency,
               SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
        FROM schemat.deposits
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency,
               SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
        FROM schemat.fin_inst
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. NII rates from irrbb.nii_results
# ─────────────────────────────────────────────────────────────────────────────

def _load_nii_by_product() -> pd.DataFrame:
    """NII per (product_code, bs_side, currency) per scenario from irrbb.nii_results.

    Returns DataFrame with columns:
        product_code, bs_side, currency, scenario_id, nii_total
    """
    query = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side,
               currency,
               scenario_id,
               SUM(nii_total) AS nii_total
        FROM irrbb.nii_results
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVE PV and duration from irrbb.eve_results
# ─────────────────────────────────────────────────────────────────────────────

def _load_eve_by_product() -> pd.DataFrame:
    """EVE PV per (product_code, bs_side, currency) per scenario.

    Returns DataFrame with columns:
        product_code, bs_side, currency, scenario_id, pv_total
    """
    query = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side,
               currency,
               scenario_id,
               SUM(pv_total) AS pv_total
        FROM irrbb.eve_results
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. CF-based parameters (inflow fraction, amortisation fraction)
# ─────────────────────────────────────────────────────────────────────────────

def _load_cf_params() -> pd.DataFrame:
    """Per-product CF aggregates from cf.products.

    Returns DataFrame with columns:
        product_code (str), bs_side, currency,
        total_balance          : Σ outstanding_bal at cf_end_dt > report_date
        capital_within_horizon : Σ (capital + prepay) for CFs ≤ horizon_end  (1Y amortisation)
        capital_within_30d     : Σ (capital + prepay) for CFs ≤ horizon_30d_end (LCR inflows)
    """
    query = text("""
        SELECT
            CAST(product_code AS VARCHAR(4)) AS product_code,
            bs_side,
            currency,
            SUM(CAST(beh_outstanding AS FLOAT) * cf_yf) AS bal_yf,
            SUM(CASE WHEN cf_end_dt <= :he
                     THEN CAST(beh_capital_pmt AS FLOAT)
                          + COALESCE(CAST(prepayment_pmt AS FLOAT), 0.0)
                     ELSE 0.0 END) AS capital_within_horizon,
            SUM(CASE WHEN cf_end_dt <= :h30
                     THEN CAST(beh_capital_pmt AS FLOAT)
                          + COALESCE(CAST(prepayment_pmt AS FLOAT), 0.0)
                     ELSE 0.0 END) AS capital_within_30d,
            SUM(CAST(beh_outstanding AS FLOAT)) AS total_outstanding
        FROM cf.products
        WHERE cf_end_dt > :rd
          AND COALESCE(beh_total_pmt, 0) <> 0
        GROUP BY product_code, bs_side, currency
    """)
    df = pd.read_sql_query(
        query, engine,
        params={"rd": REPORT_DATE, "he": HORIZON_END, "h30": HORIZON_30D_END},
    )
    df["product_code"] = df["product_code"].astype(str)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Repricing tenor from sched tables
# ─────────────────────────────────────────────────────────────────────────────

def _load_repricing_tenor() -> pd.DataFrame:
    """Balance-weighted average repricing tenor per (product_code, bs_side, currency).

    For variable-rate loans: use fixing_freq (e.g. '3M' → 3 months).
    For fixed-rate loans: use remaining maturity (months from report_date).
    For deposits: similar logic.

    Returns columns: product_code, bs_side, currency, repricing_tenor_m (float)
    """
    def _freq_to_months(freq: str | None) -> float:
        if freq is None or (isinstance(freq, float) and np.isnan(freq)):
            return 12.0  # default: annual repricing
        freq = str(freq).strip().upper()
        if freq.endswith("M"):
            try:
                return float(freq[:-1])
            except ValueError:
                return 12.0
        if freq.endswith("Y"):
            try:
                return float(freq[:-1]) * 12.0
            except ValueError:
                return 12.0
        return 12.0

    query_loans = text("""
        SELECT
            CAST(product_code AS VARCHAR(4)) AS product_code,
            bs_side,
            currency,
            rate_type,
            fixing_freq,
            maturity_date,
            SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
        FROM schemat.loans
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, rate_type, fixing_freq, maturity_date
    """)
    query_dep = text("""
        SELECT
            CAST(product_code AS VARCHAR(4)) AS product_code,
            bs_side,
            currency,
            rate_type,
            NULL AS fixing_freq,
            maturity_date,
            SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
        FROM schemat.deposits
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, rate_type, maturity_date
    """)
    query_fi = text("""
        SELECT
            CAST(product_code AS VARCHAR(4)) AS product_code,
            bs_side,
            currency,
            rate_type,
            NULL AS fixing_freq,
            maturity_date,
            SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
        FROM schemat.fin_inst
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, rate_type, maturity_date
    """)
    parts = []
    for q in (query_loans, query_dep, query_fi):
        try:
            parts.append(pd.read_sql_query(q, engine, params={"rd": REPORT_DATE}))
        except Exception:
            pass
    if not parts:
        return pd.DataFrame(columns=["product_code", "bs_side", "currency", "repricing_tenor_m"])

    df = pd.concat(parts, ignore_index=True)
    df["product_code"] = df["product_code"].astype(str)
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    df["balance_amt"]   = df["balance_amt"].fillna(0.0).astype(float)

    # Compute per-row repricing tenor in months
    def _row_tenor(row: pd.Series) -> float:
        if str(row.get("rate_type", "V")) == "V":
            # variable: repricing is at fixing_freq intervals
            return _freq_to_months(row.get("fixing_freq"))
        else:
            # fixed: repricing at maturity
            if pd.notna(row.get("maturity_date")):
                months = (row["maturity_date"] - REPORT_DATE).days / 30.44
                return max(1.0, months)
            return 12.0  # fallback

    df["repricing_tenor_m"] = df.apply(_row_tenor, axis=1)

    # Balance-weighted average per product group
    df["bal_x_tenor"] = df["balance_amt"] * df["repricing_tenor_m"]
    agg = df.groupby(["product_code", "bs_side", "currency"]).agg(
        bal_sum       = ("balance_amt", "sum"),
        bal_x_tenor_s = ("bal_x_tenor", "sum"),
    ).reset_index()
    agg["repricing_tenor_m"] = np.where(
        agg["bal_sum"] > 0,
        agg["bal_x_tenor_s"] / agg["bal_sum"],
        12.0,
    )
    return agg[["product_code", "bs_side", "currency", "repricing_tenor_m"]]


# ─────────────────────────────────────────────────────────────────────────────
# 8. Assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_product_params() -> None:
    """Build and save product_params.npz.

    For each row in bs_structure computes:
      - nii_unit_rate     : NII / balance for base scenario
      - delta_nii_unit    : delta_NII / balance per shocked scenario (n × S)
      - eve_pv_factor     : EVE_base / balance
      - d_mod             : modified duration from par_up/par_dn EVE (signed, years)
      - delta_eve_unit    : delta_EVE / balance per shocked scenario (n × S)
      - hqla_factor       : (1-haircut) × [hqla_class set], else 0
      - lcr_runoff        : LCR run-off rate (liabilities), 0 for assets
      - asf_factor        : ASF factor (liabilities/equity), 0 for assets
      - rsf_factor        : RSF factor (assets), 0 for liabilities
      - inflow_30d_frac   : capital_within_30d / balance for assets
      - amort_frac_1y     : capital_within_1y / balance
      - repricing_tenor_m : balance-weighted average repricing tenor
      - coeff_a, coeff_b  : rate transform coefficients
      - client_floor, client_cap : rate limits
    """
    print("Loading bs_structure...")
    bs = _load_bs_structure()
    n  = len(bs)
    print(f"  {n} rows in bs_structure")

    print("Loading interest_rt.xlsx...")
    ir_coeff = _load_rate_coefficients()

    print("Loading balance by product from schemat tables...")
    bal_df = _load_balance_by_product()

    print("Loading NII results from irrbb.nii_results...")
    nii_df  = _load_nii_by_product()

    print("Loading EVE results from irrbb.eve_results...")
    eve_df  = _load_eve_by_product()

    print("Loading CF parameters from cf.products...")
    cf_df   = _load_cf_params()

    print("Loading repricing tenor from schemat tables...")
    rep_df  = _load_repricing_tenor()

    # ── Merge-key helper ──────────────────────────────────────────────────────
    # bs_structure rows are the authoritative index.
    # Match on (product_code str, bs_side, currency).
    def _merge_key(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["product_code"] = df["product_code"].astype(str)
        return df

    bs["_key"] = list(zip(bs["product_code"], bs["bs_side"], bs["currency"]))

    # ── Helper: build per-row lookup dict ─────────────────────────────────────
    def _build_key_dict(df: pd.DataFrame, val_col: str) -> dict:
        """dict[(product_code, bs_side, currency)] → value"""
        return dict(zip(
            zip(df["product_code"].astype(str), df["bs_side"], df["currency"]),
            df[val_col],
        ))

    # ── Balance per product group ─────────────────────────────────────────────
    bal_map = _build_key_dict(
        bal_df.groupby(["product_code", "bs_side", "currency"]).agg(
            balance_amt=("balance_amt", "sum")
        ).reset_index(),
        "balance_amt",
    )
    # Use bs_pct-implied balance as fallback (same scaling as the exact pipeline)
    full_balance = TOTAL_ASSETS * 2
    def _balance_for_row(row: pd.Series) -> float:
        k = (row["product_code"], row["bs_side"], row["currency"])
        return bal_map.get(k, full_balance * row["full_pct"] / 100.0)

    balance_arr = np.array([_balance_for_row(r) for _, r in bs.iterrows()], dtype=float)

    # ── NII unit rates ────────────────────────────────────────────────────────
    # base NII
    base_nii_map = _build_key_dict(
        nii_df[nii_df["scenario_id"] == "base"]
        .groupby(["product_code", "bs_side", "currency"])
        .agg(nii_total=("nii_total", "sum"))
        .reset_index(),
        "nii_total",
    )
    nii_unit_rate = np.array(
        [base_nii_map.get((r["product_code"], r["bs_side"], r["currency"]), 0.0) / max(balance_arr[i], 1.0)
         for i, (_, r) in enumerate(bs.iterrows())],
        dtype=float,
    )

    # shocked NII per scenario
    delta_nii_unit = np.zeros((n, len(SHOCKED_SCENARIO_IDS)), dtype=float)
    for s_idx, scen in enumerate(SHOCKED_SCENARIO_IDS):
        shocked_map = _build_key_dict(
            nii_df[nii_df["scenario_id"] == scen]
            .groupby(["product_code", "bs_side", "currency"])
            .agg(nii_total=("nii_total", "sum"))
            .reset_index(),
            "nii_total",
        )
        for i, (_, r) in enumerate(bs.iterrows()):
            k = (r["product_code"], r["bs_side"], r["currency"])
            shocked_nii = shocked_map.get(k, np.nan)
            base_nii    = base_nii_map.get(k, 0.0)
            if not np.isnan(shocked_nii):
                delta_nii_unit[i, s_idx] = (shocked_nii - base_nii) / max(balance_arr[i], 1.0)

    # ── EVE per unit balance ──────────────────────────────────────────────────
    base_eve_map = _build_key_dict(
        eve_df[eve_df["scenario_id"] == "base"]
        .groupby(["product_code", "bs_side", "currency"])
        .agg(pv_total=("pv_total", "sum"))
        .reset_index(),
        "pv_total",
    )
    eve_pv_factor = np.array(
        [base_eve_map.get((r["product_code"], r["bs_side"], r["currency"]), 0.0) / max(balance_arr[i], 1.0)
         for i, (_, r) in enumerate(bs.iterrows())],
        dtype=float,
    )

    # modified duration from par_up / par_dn EVE
    par_up_eve_map = _build_key_dict(
        eve_df[eve_df["scenario_id"] == "par_up"]
        .groupby(["product_code", "bs_side", "currency"])
        .agg(pv_total=("pv_total", "sum"))
        .reset_index(),
        "pv_total",
    )
    par_dn_eve_map = _build_key_dict(
        eve_df[eve_df["scenario_id"] == "par_dn"]
        .groupby(["product_code", "bs_side", "currency"])
        .agg(pv_total=("pv_total", "sum"))
        .reset_index(),
        "pv_total",
    )
    d_mod = np.zeros(n, dtype=float)
    for i, (_, r) in enumerate(bs.iterrows()):
        k    = (r["product_code"], r["bs_side"], r["currency"])
        bal  = max(balance_arr[i], 1.0)
        pv_up = par_up_eve_map.get(k, np.nan)
        pv_dn = par_dn_eve_map.get(k, np.nan)
        if not (np.isnan(pv_up) or np.isnan(pv_dn)):
            # d_mod = -(pv_dn - pv_up) / (2 * PAR_SHOCK_RATE * bal)
            # Note: sign included — assets have +d_mod (PV falls when r rises)
            d_mod[i] = -(pv_dn - pv_up) / (2.0 * PAR_SHOCK_RATE * bal)

    # shocked EVE per scenario
    delta_eve_unit = np.zeros((n, len(SHOCKED_SCENARIO_IDS)), dtype=float)
    for s_idx, scen in enumerate(SHOCKED_SCENARIO_IDS):
        shocked_eve_map = _build_key_dict(
            eve_df[eve_df["scenario_id"] == scen]
            .groupby(["product_code", "bs_side", "currency"])
            .agg(pv_total=("pv_total", "sum"))
            .reset_index(),
            "pv_total",
        )
        for i, (_, r) in enumerate(bs.iterrows()):
            k        = (r["product_code"], r["bs_side"], r["currency"])
            pv_shock = shocked_eve_map.get(k, np.nan)
            pv_base  = base_eve_map.get(k, 0.0)
            if not np.isnan(pv_shock):
                delta_eve_unit[i, s_idx] = (pv_shock - pv_base) / max(balance_arr[i], 1.0)

    # ── LCR / NSFR factors from bs_structure ─────────────────────────────────
    hqla_factor = np.where(
        bs["hqla_class"].notna() & (bs["bs_side"] == "A"),
        (1.0 - bs["haircut"].fillna(0.0)).to_numpy(dtype=float),
        0.0,
    )
    lcr_runoff = np.where(
        bs["LCR"].notna() & bs["bs_side"].isin(["L"]),
        bs["LCR"].fillna(0.0).to_numpy(dtype=float),
        0.0,
    )
    asf_factor = np.where(
        bs["ASF"].notna() & bs["bs_side"].isin(["L", "E"]),
        bs["ASF"].fillna(0.0).to_numpy(dtype=float),
        0.0,
    )
    rsf_factor = np.where(
        bs["RSF"].notna() & (bs["bs_side"] == "A"),
        bs["RSF"].fillna(0.0).to_numpy(dtype=float),
        0.0,
    )

    # ── CF-based inflow and amort fractions ───────────────────────────────────
    cf_map_30d  = _build_key_dict(cf_df, "capital_within_30d")
    cf_map_1y   = _build_key_dict(cf_df, "capital_within_horizon")
    cf_map_bal  = _build_key_dict(cf_df, "total_outstanding")

    inflow_30d_frac = np.zeros(n, dtype=float)
    amort_frac_1y   = np.zeros(n, dtype=float)

    for i, (_, r) in enumerate(bs.iterrows()):
        k   = (r["product_code"], r["bs_side"], r["currency"])
        bal = cf_map_bal.get(k, balance_arr[i])
        if bal > 0 and r["bs_side"] == "A":
            inflow_30d_frac[i] = cf_map_30d.get(k, 0.0) / bal
            amort_frac_1y[i]   = cf_map_1y.get(k, 0.0)  / bal

    # ── Repricing tenor ───────────────────────────────────────────────────────
    rep_map = _build_key_dict(rep_df, "repricing_tenor_m")
    repricing_tenor_m = np.array(
        [rep_map.get((r["product_code"], r["bs_side"], r["currency"]), 12.0)
         for _, r in bs.iterrows()],
        dtype=float,
    )
    # For V (variable) products: tenor is the fixing interval → clip at 12M
    # For F (fixed) products: tenor is remaining maturity → no clip (can exceed horizon)
    # We use the value as-is; callers interpret it as the first repricing point.

    # ── Rate coefficients ─────────────────────────────────────────────────────
    coeff_a     = np.ones(n,  dtype=float)
    coeff_b     = np.zeros(n, dtype=float)
    client_floor = np.full(n, -np.inf, dtype=float)
    client_cap   = np.full(n,  np.inf, dtype=float)

    for i, (_, r) in enumerate(bs.iterrows()):
        pc = r["product_code"]
        if pc in ir_coeff.index:
            row = ir_coeff.loc[pc]
            if not np.isnan(row.get("coeff_a", np.nan)):
                coeff_a[i]     = float(row["coeff_a"])
            if not np.isnan(row.get("coeff_b", np.nan)):
                coeff_b[i]     = float(row["coeff_b"])
            if not np.isnan(row.get("client_floor", np.nan)):
                client_floor[i] = float(row["client_floor"])
            if not np.isnan(row.get("client_cap", np.nan)):
                client_cap[i]  = float(row["client_cap"])

    # ── Derived: sign array ───────────────────────────────────────────────────
    sign_arr = bs["sign"].to_numpy(dtype=float)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        # identification (stored as object arrays for string data)
        product_code   = bs["product_code"].to_numpy(),
        product_name   = bs["product_name"].to_numpy() if "product_name" in bs.columns
                         else np.array([""] * n),
        bs_side        = bs["bs_side"].to_numpy(),
        currency       = bs["currency"].to_numpy(),
        sign           = sign_arr,
        # weights
        bs_pct_current = bs["full_pct"].to_numpy(dtype=float),
        # NII
        nii_unit_rate  = nii_unit_rate,
        delta_nii_unit = delta_nii_unit,   # shape (n, S)
        # EVE
        eve_pv_factor  = eve_pv_factor,
        d_mod          = d_mod,
        delta_eve_unit = delta_eve_unit,   # shape (n, S)
        # LCR / NSFR
        hqla_factor    = hqla_factor,
        lcr_runoff     = lcr_runoff,
        asf_factor     = asf_factor,
        rsf_factor     = rsf_factor,
        inflow_30d_frac= inflow_30d_frac,
        amort_frac_1y  = amort_frac_1y,
        # repricing
        repricing_tenor_m = repricing_tenor_m,
        # rate coefficients
        coeff_a        = coeff_a,
        coeff_b        = coeff_b,
        client_floor   = client_floor,
        client_cap     = client_cap,
        # scenario / metadata
        scenario_ids   = np.array(SHOCKED_SCENARIO_IDS),
        total_assets   = np.array([TOTAL_ASSETS]),
        report_date    = np.array([str(REPORT_DATE.date())]),
        balance_arr    = balance_arr,
    )
    print(f"Saved product_params.npz → {OUT_PATH}")
    print(f"  n_products={n}, n_scenarios={len(SHOCKED_SCENARIO_IDS)}")
    print(f"  NII unit rates range: [{nii_unit_rate.min():.4f}, {nii_unit_rate.max():.4f}]")
    print(f"  d_mod range: [{d_mod.min():.2f}, {d_mod.max():.2f}]")


if __name__ == "__main__":
    build_product_params()
