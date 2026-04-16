"""extract_params.py
===================
One-time extraction from SQL + Excel → opt_prep SQL tables + product_params.npz

Run this script once (or whenever the balance sheet structure or market data
change materially).

Outputs
-------
  SQL: opt_prep.product_params          — base parameters per product row
  SQL: opt_prep.product_scenario_params — delta NII / delta EVE per product × scenario
  File: optimize_prep/output/product_params.npz  — frozen numpy arrays for optimizer hot path
  File: optimize_prep/output/params_inspection.xlsx — human-readable summary for validation

Dependencies
------------
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

import sql_setup as opt_sql

engine = opt_sql.engine

# ── configuration ──────────────────────────────────────────────────────────────
REPORT_DATE    = pd.to_datetime("2024-12-31")
TOTAL_ASSETS   = 10_000_000_000
HORIZON_DAYS   = 365
HORIZON_30D    = 30
PAR_SHOCK_RATE = 0.02        # PLN EBA par shock = 200 bps for duration calc
HORIZON_END    = REPORT_DATE + pd.Timedelta(days=HORIZON_DAYS)
HORIZON_30D_END= REPORT_DATE + pd.Timedelta(days=HORIZON_30D)

BS_PATH       = os.path.join(ROOT_DIR, "balance_generate", "input_data", "bank_data.xlsx")
INTEREST_PATH = os.path.join(ROOT_DIR, "balance_generate", "input_data", "interest_rt.xlsx")
NPZ_OUT       = os.path.join(BASE_DIR, "..", "output", "product_params.npz")
EXCEL_OUT     = os.path.join(BASE_DIR, "..", "output", "params_inspection.xlsx")

SHOCKED_SCENARIO_IDS = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]


# ─────────────────────────────────────────────────────────────────────────────
# Source data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_bs_structure() -> pd.DataFrame:
    df = pd.read_excel(BS_PATH, sheet_name="bs_structure")
    df["sign"]         = df["bs_side"].map({"A": 1.0, "L": -1.0, "E": -1.0}).fillna(-1.0)
    df["full_pct"]     = df["bs_percentage"].fillna(0.0)
    df["product_code"] = df["product_code"].astype(str)
    return df


def _load_rate_coefficients() -> pd.DataFrame:
    df = pd.read_excel(INTEREST_PATH)
    df["product_code"] = df["product_code"].astype(int).astype(str)
    if "b" in df.columns:
        df["b"] = df["b"] / 100.0
    rename = {"a": "coeff_a", "b": "coeff_b",
              "client_floor": "client_floor", "client_cap": "client_cap"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ("coeff_a", "coeff_b", "client_floor", "client_cap"):
        if col not in df.columns:
            df[col] = np.nan
    return df.set_index("product_code")[["coeff_a", "coeff_b", "client_floor", "client_cap"]]


def _load_balance_by_product() -> pd.DataFrame:
    query = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency,
               SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
        FROM schemat.loans   WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)), bs_side, currency,
               SUM(CAST(balance_amt AS FLOAT))
        FROM schemat.deposits WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)), bs_side, currency,
               SUM(CAST(balance_amt AS FLOAT))
        FROM schemat.fin_inst WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


def _load_nii_by_product() -> pd.DataFrame:
    query = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, scenario_id,
               SUM(nii_total) AS nii_total
        FROM irrbb.nii_results
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


def _load_eve_by_product() -> pd.DataFrame:
    query = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, scenario_id,
               SUM(pv_total) AS pv_total
        FROM irrbb.eve_results
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


def _load_cf_params() -> pd.DataFrame:
    query = text("""
        SELECT
            CAST(product_code AS VARCHAR(4)) AS product_code,
            bs_side, currency,
            SUM(CAST(beh_outstanding AS FLOAT) * cf_yf)     AS bal_yf,
            SUM(CASE WHEN cf_end_dt <= :he
                     THEN CAST(beh_capital_pmt AS FLOAT)
                          + COALESCE(CAST(prepayment_pmt AS FLOAT), 0.0)
                     ELSE 0.0 END)                           AS capital_within_horizon,
            SUM(CASE WHEN cf_end_dt <= :h30
                     THEN CAST(beh_capital_pmt AS FLOAT)
                          + COALESCE(CAST(prepayment_pmt AS FLOAT), 0.0)
                     ELSE 0.0 END)                           AS capital_within_30d,
            SUM(CAST(beh_outstanding AS FLOAT))              AS total_outstanding
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


def _load_repricing_tenor() -> pd.DataFrame:
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

    parts = []
    for q in [
        text("""
            SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, rate_type, fixing_freq, maturity_date,
                   SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
            FROM schemat.loans WHERE report_date = :rd
            GROUP BY product_code, bs_side, currency, rate_type, fixing_freq, maturity_date
        """),
        text("""
            SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, rate_type,
                   NULL AS fixing_freq, maturity_date,
                   SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
            FROM schemat.deposits WHERE report_date = :rd
            GROUP BY product_code, bs_side, currency, rate_type, maturity_date
        """),
        text("""
            SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, rate_type,
                   NULL AS fixing_freq, maturity_date,
                   SUM(CAST(balance_amt AS FLOAT)) AS balance_amt
            FROM schemat.fin_inst WHERE report_date = :rd
            GROUP BY product_code, bs_side, currency, rate_type, maturity_date
        """),
    ]:
        try:
            parts.append(pd.read_sql_query(q, engine, params={"rd": REPORT_DATE}))
        except Exception:
            pass

    if not parts:
        return pd.DataFrame(columns=["product_code", "bs_side", "currency", "repricing_tenor_m"])

    df = pd.concat(parts, ignore_index=True)
    df["product_code"]  = df["product_code"].astype(str)
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    df["balance_amt"]   = df["balance_amt"].fillna(0.0).astype(float)

    def _row_tenor(row) -> float:
        if str(row.get("rate_type", "V")) == "V":
            return _freq_to_months(row.get("fixing_freq"))
        if pd.notna(row.get("maturity_date")):
            months = (row["maturity_date"] - REPORT_DATE).days / 30.44
            return max(1.0, months)
        return 12.0

    df["repricing_tenor_m"] = df.apply(_row_tenor, axis=1)
    df["bal_x_tenor"]       = df["balance_amt"] * df["repricing_tenor_m"]
    agg = df.groupby(["product_code", "bs_side", "currency"]).agg(
        bal_sum       =("balance_amt", "sum"),
        bal_x_tenor_s =("bal_x_tenor", "sum"),
    ).reset_index()
    agg["repricing_tenor_m"] = np.where(
        agg["bal_sum"] > 0,
        agg["bal_x_tenor_s"] / agg["bal_sum"],
        12.0,
    )
    return agg[["product_code", "bs_side", "currency", "repricing_tenor_m"]]


# ─────────────────────────────────────────────────────────────────────────────
# Assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_product_params() -> None:
    print("Loading bs_structure...")
    bs = _load_bs_structure()
    n  = len(bs)
    print(f"  {n} rows in bs_structure")

    print("Loading interest_rt.xlsx...")
    ir_coeff = _load_rate_coefficients()

    print("Loading balance from schemat tables...")
    bal_df  = _load_balance_by_product()

    print("Loading NII results from irrbb.nii_results...")
    nii_df  = _load_nii_by_product()

    print("Loading EVE results from irrbb.eve_results...")
    eve_df  = _load_eve_by_product()

    print("Loading CF parameters from cf.products...")
    cf_df   = _load_cf_params()

    print("Loading repricing tenor from schemat tables...")
    rep_df  = _load_repricing_tenor()

    # ── key → value lookup helpers ────────────────────────────────────────────
    def _key_dict(df: pd.DataFrame, val_col: str) -> dict:
        return dict(zip(
            zip(df["product_code"].astype(str), df["bs_side"], df["currency"]),
            df[val_col],
        ))

    full_balance = TOTAL_ASSETS * 2
    bal_map = _key_dict(
        bal_df.groupby(["product_code", "bs_side", "currency"])
        .agg(balance_amt=("balance_amt", "sum")).reset_index(),
        "balance_amt",
    )
    def _bal(i: int, row: pd.Series) -> float:
        k = (row["product_code"], row["bs_side"], row["currency"])
        return bal_map.get(k, full_balance * row["full_pct"] / 100.0)

    balance_arr = np.array([_bal(i, r) for i, (_, r) in enumerate(bs.iterrows())], dtype=float)

    # ── NII unit rates ────────────────────────────────────────────────────────
    base_nii_map = _key_dict(
        nii_df[nii_df["scenario_id"] == "base"]
        .groupby(["product_code", "bs_side", "currency"])
        .agg(nii_total=("nii_total", "sum")).reset_index(),
        "nii_total",
    )
    nii_unit_rate = np.array(
        [base_nii_map.get((r["product_code"], r["bs_side"], r["currency"]), 0.0)
         / max(balance_arr[i], 1.0)
         for i, (_, r) in enumerate(bs.iterrows())],
        dtype=float,
    )

    delta_nii_unit = np.zeros((n, len(SHOCKED_SCENARIO_IDS)), dtype=float)
    for s_idx, scen in enumerate(SHOCKED_SCENARIO_IDS):
        shocked_map = _key_dict(
            nii_df[nii_df["scenario_id"] == scen]
            .groupby(["product_code", "bs_side", "currency"])
            .agg(nii_total=("nii_total", "sum")).reset_index(),
            "nii_total",
        )
        for i, (_, r) in enumerate(bs.iterrows()):
            k = (r["product_code"], r["bs_side"], r["currency"])
            sh = shocked_map.get(k, np.nan)
            ba = base_nii_map.get(k, 0.0)
            if not np.isnan(sh):
                delta_nii_unit[i, s_idx] = (sh - ba) / max(balance_arr[i], 1.0)

    # ── EVE per unit balance ──────────────────────────────────────────────────
    base_eve_map = _key_dict(
        eve_df[eve_df["scenario_id"] == "base"]
        .groupby(["product_code", "bs_side", "currency"])
        .agg(pv_total=("pv_total", "sum")).reset_index(),
        "pv_total",
    )
    eve_pv_factor = np.array(
        [base_eve_map.get((r["product_code"], r["bs_side"], r["currency"]), 0.0)
         / max(balance_arr[i], 1.0)
         for i, (_, r) in enumerate(bs.iterrows())],
        dtype=float,
    )

    par_up_map = _key_dict(
        eve_df[eve_df["scenario_id"] == "par_up"]
        .groupby(["product_code", "bs_side", "currency"])
        .agg(pv_total=("pv_total", "sum")).reset_index(),
        "pv_total",
    )
    par_dn_map = _key_dict(
        eve_df[eve_df["scenario_id"] == "par_dn"]
        .groupby(["product_code", "bs_side", "currency"])
        .agg(pv_total=("pv_total", "sum")).reset_index(),
        "pv_total",
    )
    d_mod = np.zeros(n, dtype=float)
    for i, (_, r) in enumerate(bs.iterrows()):
        k   = (r["product_code"], r["bs_side"], r["currency"])
        bal = max(balance_arr[i], 1.0)
        pu  = par_up_map.get(k, np.nan)
        pd_ = par_dn_map.get(k, np.nan)
        if not (np.isnan(pu) or np.isnan(pd_)):
            d_mod[i] = -(pd_ - pu) / (2.0 * PAR_SHOCK_RATE * bal)

    delta_eve_unit = np.zeros((n, len(SHOCKED_SCENARIO_IDS)), dtype=float)
    for s_idx, scen in enumerate(SHOCKED_SCENARIO_IDS):
        sh_eve_map = _key_dict(
            eve_df[eve_df["scenario_id"] == scen]
            .groupby(["product_code", "bs_side", "currency"])
            .agg(pv_total=("pv_total", "sum")).reset_index(),
            "pv_total",
        )
        for i, (_, r) in enumerate(bs.iterrows()):
            k  = (r["product_code"], r["bs_side"], r["currency"])
            sh = sh_eve_map.get(k, np.nan)
            ba = base_eve_map.get(k, 0.0)
            if not np.isnan(sh):
                delta_eve_unit[i, s_idx] = (sh - ba) / max(balance_arr[i], 1.0)

    # ── LCR / NSFR factors ────────────────────────────────────────────────────
    hqla_factor = np.where(
        bs["hqla_class"].notna() & (bs["bs_side"] == "A"),
        (1.0 - bs["haircut"].fillna(0.0)).to_numpy(dtype=float), 0.0)
    lcr_runoff  = np.where(
        bs["LCR"].notna() & (bs["bs_side"] == "L"),
        bs["LCR"].fillna(0.0).to_numpy(dtype=float), 0.0)
    asf_factor  = np.where(
        bs["ASF"].notna() & bs["bs_side"].isin(["L", "E"]),
        bs["ASF"].fillna(0.0).to_numpy(dtype=float), 0.0)
    rsf_factor  = np.where(
        bs["RSF"].notna() & (bs["bs_side"] == "A"),
        bs["RSF"].fillna(0.0).to_numpy(dtype=float), 0.0)

    # ── CF-based fractions ────────────────────────────────────────────────────
    cf_30d  = _key_dict(cf_df, "capital_within_30d")
    cf_1y   = _key_dict(cf_df, "capital_within_horizon")
    cf_bal  = _key_dict(cf_df, "total_outstanding")

    inflow_30d_frac = np.zeros(n, dtype=float)
    amort_frac_1y   = np.zeros(n, dtype=float)
    for i, (_, r) in enumerate(bs.iterrows()):
        k   = (r["product_code"], r["bs_side"], r["currency"])
        bal = cf_bal.get(k, balance_arr[i])
        if bal > 0 and r["bs_side"] == "A":
            inflow_30d_frac[i] = cf_30d.get(k, 0.0) / bal
            amort_frac_1y[i]   = cf_1y.get(k, 0.0)  / bal

    # ── Repricing tenor ───────────────────────────────────────────────────────
    rep_map = _key_dict(rep_df, "repricing_tenor_m")
    repricing_tenor_m = np.array(
        [rep_map.get((r["product_code"], r["bs_side"], r["currency"]), 12.0)
         for _, r in bs.iterrows()],
        dtype=float,
    )

    # ── Rate coefficients ─────────────────────────────────────────────────────
    coeff_a      = np.ones(n,  dtype=float)
    coeff_b      = np.zeros(n, dtype=float)
    client_floor = np.full(n, -np.inf, dtype=float)
    client_cap   = np.full(n,  np.inf, dtype=float)
    for i, (_, r) in enumerate(bs.iterrows()):
        pc = r["product_code"]
        if pc in ir_coeff.index:
            row = ir_coeff.loc[pc]
            if not np.isnan(row.get("coeff_a",     np.nan)): coeff_a[i]      = float(row["coeff_a"])
            if not np.isnan(row.get("coeff_b",     np.nan)): coeff_b[i]      = float(row["coeff_b"])
            if not np.isnan(row.get("client_floor",np.nan)): client_floor[i] = float(row["client_floor"])
            if not np.isnan(row.get("client_cap",  np.nan)): client_cap[i]   = float(row["client_cap"])

    sign_arr = bs["sign"].to_numpy(dtype=float)

    # ─────────────────────────────────────────────────────────────────────────
    # Build output DataFrames
    # ─────────────────────────────────────────────────────────────────────────

    # ── product_params DataFrame ──────────────────────────────────────────────
    params_df = pd.DataFrame({
        "report_date":       REPORT_DATE,
        "product_code":      bs["product_code"].to_numpy(),
        "product_name":      bs["product_name"].to_numpy() if "product_name" in bs.columns
                             else [""] * n,
        "bs_side":           bs["bs_side"].to_numpy(),
        "currency":          bs["currency"].to_numpy(),
        "bs_pct_current":    bs["full_pct"].to_numpy(dtype=float),
        "balance_amt":       balance_arr,
        "nii_unit_rate":     nii_unit_rate,
        "eve_pv_factor":     eve_pv_factor,
        "d_mod":             d_mod,
        "hqla_factor":       hqla_factor,
        "lcr_runoff":        lcr_runoff,
        "asf_factor":        asf_factor,
        "rsf_factor":        rsf_factor,
        "inflow_30d_frac":   inflow_30d_frac,
        "amort_frac_1y":     amort_frac_1y,
        "repricing_tenor_m": repricing_tenor_m,
        "coeff_a":           coeff_a,
        "coeff_b":           coeff_b,
        "client_floor":      np.where(np.isfinite(client_floor), client_floor, np.nan),
        "client_cap":        np.where(np.isfinite(client_cap),   client_cap,   np.nan),
    })

    # ── product_scenario_params DataFrame ─────────────────────────────────────
    scen_rows = []
    for i, (_, r) in enumerate(bs.iterrows()):
        for s_idx, scen in enumerate(SHOCKED_SCENARIO_IDS):
            scen_rows.append({
                "report_date":    REPORT_DATE,
                "product_code":   r["product_code"],
                "bs_side":        r["bs_side"],
                "currency":       r["currency"],
                "scenario_id":    scen,
                "delta_nii_unit": delta_nii_unit[i, s_idx],
                "delta_eve_unit": delta_eve_unit[i, s_idx],
            })
    scenario_df = pd.DataFrame(scen_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Write to SQL
    # ─────────────────────────────────────────────────────────────────────────
    print("Writing to SQL: opt_prep.product_params...")
    opt_sql.reset_product_params()
    opt_sql.write_product_params(params_df)
    print(f"  Written {len(params_df)} rows")

    print("Writing to SQL: opt_prep.product_scenario_params...")
    opt_sql.reset_product_scenario_params()
    opt_sql.write_product_scenario_params(scenario_df)
    print(f"  Written {len(scenario_df)} rows  "
          f"({n} products × {len(SHOCKED_SCENARIO_IDS)} scenarios)")

    # ─────────────────────────────────────────────────────────────────────────
    # Write Excel inspection report
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(EXCEL_OUT), exist_ok=True)
    print(f"Writing inspection Excel → {EXCEL_OUT}")

    # Format for readability: rates as %, amounts in millions
    inspect = params_df.copy()
    inspect["balance_amt_m"]      = (inspect["balance_amt"] / 1e6).round(1)
    inspect["nii_unit_rate_pct"]  = (inspect["nii_unit_rate"]  * 100).round(4)
    inspect["eve_pv_factor_pct"]  = (inspect["eve_pv_factor"]  * 100).round(2)
    inspect["hqla_factor_pct"]    = (inspect["hqla_factor"]    * 100).round(1)
    inspect["lcr_runoff_pct"]     = (inspect["lcr_runoff"]     * 100).round(1)
    inspect["asf_factor_pct"]     = (inspect["asf_factor"]     * 100).round(1)
    inspect["rsf_factor_pct"]     = (inspect["rsf_factor"]     * 100).round(1)
    inspect["inflow_30d_frac_pct"]= (inspect["inflow_30d_frac"]* 100).round(2)
    inspect["amort_frac_1y_pct"]  = (inspect["amort_frac_1y"]  * 100).round(2)

    # Pivot delta_nii and delta_eve per scenario as separate columns
    nii_pivot = (
        scenario_df
        .pivot_table(index=["product_code", "bs_side", "currency"],
                     columns="scenario_id",
                     values="delta_nii_unit",
                     aggfunc="first")
        .rename(columns=lambda c: f"dNII_{c}_pct")
        .reset_index()
    )
    for col in nii_pivot.columns:
        if col.startswith("dNII_"):
            nii_pivot[col] = (nii_pivot[col] * 100).round(4)

    eve_pivot = (
        scenario_df
        .pivot_table(index=["product_code", "bs_side", "currency"],
                     columns="scenario_id",
                     values="delta_eve_unit",
                     aggfunc="first")
        .rename(columns=lambda c: f"dEVE_{c}_pct")
        .reset_index()
    )
    for col in eve_pivot.columns:
        if col.startswith("dEVE_"):
            eve_pivot[col] = (eve_pivot[col] * 100).round(4)

    inspect = (
        inspect
        .merge(nii_pivot, on=["product_code", "bs_side", "currency"], how="left")
        .merge(eve_pivot, on=["product_code", "bs_side", "currency"], how="left")
    )

    cols_overview = [
        "product_code", "product_name", "bs_side", "currency",
        "bs_pct_current", "balance_amt_m",
        "nii_unit_rate_pct", "eve_pv_factor_pct", "d_mod",
        "repricing_tenor_m",
        "hqla_factor_pct", "lcr_runoff_pct", "asf_factor_pct", "rsf_factor_pct",
        "inflow_30d_frac_pct", "amort_frac_1y_pct",
        "coeff_a", "coeff_b", "client_floor", "client_cap",
    ]
    cols_nii = (
        ["product_code", "product_name", "bs_side", "currency", "balance_amt_m",
         "nii_unit_rate_pct"]
        + [c for c in inspect.columns if c.startswith("dNII_")]
    )
    cols_eve = (
        ["product_code", "product_name", "bs_side", "currency", "balance_amt_m",
         "eve_pv_factor_pct", "d_mod"]
        + [c for c in inspect.columns if c.startswith("dEVE_")]
    )
    cols_liq = [
        "product_code", "product_name", "bs_side", "currency", "balance_amt_m",
        "hqla_factor_pct", "lcr_runoff_pct", "inflow_30d_frac_pct",
        "asf_factor_pct", "rsf_factor_pct",
    ]
    cols_repricing = [
        "product_code", "product_name", "bs_side", "currency", "balance_amt_m",
        "repricing_tenor_m", "amort_frac_1y_pct", "coeff_a", "coeff_b",
        "client_floor", "client_cap",
    ]

    def _safe_cols(df, cols):
        return df[[c for c in cols if c in df.columns]]

    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
        _safe_cols(inspect, cols_overview).to_excel(
            writer, sheet_name="Overview", index=False)
        _safe_cols(inspect, cols_nii).to_excel(
            writer, sheet_name="NII_params", index=False)
        _safe_cols(inspect, cols_eve).to_excel(
            writer, sheet_name="EVE_params", index=False)
        _safe_cols(inspect, cols_liq).to_excel(
            writer, sheet_name="LCR_NSFR_params", index=False)
        _safe_cols(inspect, cols_repricing).to_excel(
            writer, sheet_name="Repricing_params", index=False)
        scenario_df.drop(columns="report_date").to_excel(
            writer, sheet_name="Scenario_deltas_raw", index=False)
    print(f"  Sheets: Overview | NII_params | EVE_params | LCR_NSFR_params | "
          f"Repricing_params | Scenario_deltas_raw")

    # ─────────────────────────────────────────────────────────────────────────
    # Save .npz for optimizer hot path
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(NPZ_OUT), exist_ok=True)
    np.savez(
        NPZ_OUT,
        product_code      = bs["product_code"].to_numpy(),
        product_name      = bs["product_name"].to_numpy() if "product_name" in bs.columns
                            else np.array([""] * n),
        bs_side           = bs["bs_side"].to_numpy(),
        currency          = bs["currency"].to_numpy(),
        sign              = sign_arr,
        bs_pct_current    = bs["full_pct"].to_numpy(dtype=float),
        nii_unit_rate     = nii_unit_rate,
        delta_nii_unit    = delta_nii_unit,
        eve_pv_factor     = eve_pv_factor,
        d_mod             = d_mod,
        delta_eve_unit    = delta_eve_unit,
        hqla_factor       = hqla_factor,
        lcr_runoff        = lcr_runoff,
        asf_factor        = asf_factor,
        rsf_factor        = rsf_factor,
        inflow_30d_frac   = inflow_30d_frac,
        amort_frac_1y     = amort_frac_1y,
        coeff_a           = coeff_a,
        coeff_b           = coeff_b,
        client_floor      = client_floor,
        client_cap        = client_cap,
        repricing_tenor_m = repricing_tenor_m,
        scenario_ids      = np.array(SHOCKED_SCENARIO_IDS),
        total_assets      = np.array([TOTAL_ASSETS]),
        report_date       = np.array([str(REPORT_DATE.date())]),
        balance_arr       = balance_arr,
    )
    print(f"Saved product_params.npz → {NPZ_OUT}")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n── Product parameter summary ──────────────────────────────────────")
    print(f"  Products (bs_structure rows): {n}")
    print(f"  NII unit rate range    : [{nii_unit_rate.min()*100:.3f}%,  "
          f"{nii_unit_rate.max()*100:.3f}%]")
    print(f"  Modified duration range: [{d_mod.min():.2f}y, {d_mod.max():.2f}y]")
    print(f"  Repricing tenor range  : [{repricing_tenor_m.min():.1f}m, "
          f"{repricing_tenor_m.max():.1f}m]")
    print(f"  Products with HQLA     : {(hqla_factor > 0).sum()}")
    print(f"  Products with LCR runoff: {(lcr_runoff > 0).sum()}")
    print(f"  Products with ASF      : {(asf_factor > 0).sum()}")
    print(f"  Products with RSF      : {(rsf_factor > 0).sum()}")


if __name__ == "__main__":
    build_product_params()
