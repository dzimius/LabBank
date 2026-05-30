from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, String, Date, MetaData
from sqlalchemy.dialects.mssql import DECIMAL

import config
from nii_calc_objects import _apply_rt_limits

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

metadata = MetaData(schema=None)

NiiResults = Table(
    "nii_results", metadata,
    Column("report_date",    Date,           nullable=False),
    Column("scenario_id",    String(20),     nullable=False),
    Column("schedule_id",    String(8),      nullable=False),
    Column("product_type",   String(1),      nullable=False),
    Column("product_code",   String(4),      nullable=False),
    Column("currency",       String(3),      nullable=False),
    Column("bs_side",        String(1),      nullable=False),
    Column("rate_type",      String(1),      nullable=True),
    Column("nii_interest",   DECIMAL(18, 2), nullable=False),
    Column("nii_renewal",    DECIMAL(18, 2), nullable=False),
    Column("nii_total",      DECIMAL(18, 2), nullable=False),
    schema="irrbb",
)

IrrbbReport = Table(
    "irrbb_report", metadata,
    Column("report_date",       Date,           nullable=False),
    Column("scenario_id",       String(20),     nullable=False),
    Column("currency",          String(3),      nullable=False),
    Column("delta_nii",         DECIMAL(18, 2), nullable=True),
    Column("sot_nii_pct",       DECIMAL(10, 4), nullable=True),
    Column("delta_nii_reg",     DECIMAL(18, 2), nullable=True),
    Column("sot_nii_pct_reg",   DECIMAL(10, 4), nullable=True),
    Column("delta_eve",         DECIMAL(18, 2), nullable=True),
    Column("sot_eve_pct",       DECIMAL(10, 4), nullable=True),
    Column("delta_eve_reg",     DECIMAL(18, 2), nullable=True),
    Column("sot_eve_pct_reg",   DECIMAL(10, 4), nullable=True),
    schema="irrbb",
)

# results schema — combined NII + EVE report with base/shocked absolute values
_results_metadata = MetaData(schema=None)

ResultsIrrbbReport = Table(
    "irrbb_report", _results_metadata,
    Column("report_date",   Date,           nullable=False),
    Column("currency",      String(3),      nullable=False),
    Column("scenario_id",   String(20),     nullable=False),
    Column("nii_base",      DECIMAL(18, 2), nullable=True),
    Column("nii_shocked",   DECIMAL(18, 2), nullable=True),
    Column("delta_nii",     DECIMAL(18, 2), nullable=True),
    Column("sot_nii_pct",   DECIMAL(10, 4), nullable=True),
    Column("eve_base",      DECIMAL(18, 2), nullable=True),
    Column("eve_shocked",   DECIMAL(18, 2), nullable=True),
    Column("delta_eve",     DECIMAL(18, 2), nullable=True),
    Column("sot_eve_pct",   DECIMAL(10, 4), nullable=True),
    schema="results",
)

EveResults = Table(
    "eve_results", metadata,
    Column("report_date",    Date,           nullable=False),
    Column("scenario_id",    String(20),     nullable=False),
    Column("schedule_id",    String(8),      nullable=False),
    Column("product_type",   String(1),      nullable=False),
    Column("product_code",   String(4),      nullable=False),
    Column("currency",       String(3),      nullable=False),
    Column("bs_side",        String(1),      nullable=False),
    Column("rate_type",      String(1),      nullable=True),
    Column("pv_capital",     DECIMAL(18, 2), nullable=False),
    Column("pv_interest",    DECIMAL(18, 2), nullable=False),
    Column("pv_total",       DECIMAL(18, 2), nullable=False),
    schema="irrbb",
)


def load_ir_gap_beh() -> pd.DataFrame:
    """Load behavioural BS repricing gap.

    Columns: currency, bs_side, tenor_bucket, cf_end_dt, gap_cf, gap_cf_ca, gap_cf_irs.
      gap_cf     — total balance-sheet gap (loans + deposits + bonds incl. CA), NO IRS
      gap_cf_ca  — current-account portion of gap_cf (for 0%-floor in down shock)
      gap_cf_irs — IRS gap (added by irs_workflow; 0 if irs_workflow not yet run)
    """
    query = text("""
        SELECT currency, bs_side, tenor_bucket,
               bucket_start_dt, bucket_end_dt, cf_end_dt,
               gap_cf, gap_cf_ca, gap_cf_irs
        FROM irrbb.ir_gap_beh
    """)
    df = pd.read_sql_query(query, engine)
    df["cf_end_dt"] = pd.to_datetime(df["cf_end_dt"])
    return df


def load_mkt_curves(report_date: pd.Timestamp) -> pd.DataFrame:
    """Load all base market discount curves for report_date from mkt.curves.

    Returns DataFrame with columns: curve_name, n_days, d_f.
    Used by eba_shock_curves to build the 8 shocked scenario curves.
    """
    query = text("""
        SELECT curve_name, n_days, d_f
        FROM   mkt.curves
        WHERE  curve_date = :rd
        ORDER  BY curve_name, n_days
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date})


def load_beh_schedules(report_date: pd.Timestamp, horizon_end: pd.Timestamp) -> pd.DataFrame:
    """Load behavioral CF schedules for all products within (report_date, horizon_end].

    Queries cf.products (beh_* columns), joins sched.* for rate_type.
    Returns unified DataFrame with source column ('loans', 'deposits', 'fin_inst').

    Columns
    -------
    source, schedule_id, bs_side, currency,
    cf_start_dt, cf_end_dt, cf_yf, fwd_rt,
    outstanding_bal, capital_pmt, prepayment_pmt, int_pmt,
    rate_type    — 'F'=fixed | 'V'=variable  (from sched.* rate_type column)
    eff_rate     — effective rate for base scenario:
                   fixed  → int_pmt / (outstanding_bal * cf_yf)  i.e. contracted rate
                   float  → fwd_rt  (market forward rate, with floor applied upstream)
    """
    query = text("""
        WITH sched_rt AS (
            SELECT schedule_id, ISNULL(rate_type, 'V') AS rate_type, 'L' AS product_type
            FROM sched.loans
            UNION ALL
            SELECT schedule_id, ISNULL(rate_type, 'V') AS rate_type, 'D' AS product_type
            FROM sched.deposits
            UNION ALL
            SELECT schedule_id, ISNULL(rate_type, 'F') AS rate_type, 'F' AS product_type
            FROM sched.fin_inst
        )
        SELECT
            CASE p.product_type
                WHEN 'L' THEN 'loans'
                WHEN 'D' THEN 'deposits'
                ELSE 'fin_inst'
            END AS source,
            p.schedule_id,
            p.product_type,
            p.product_code,
            p.bs_side,
            p.currency,
            p.fixing_dt,
            p.cf_start_dt,
            p.cf_end_dt,
            p.cf_yf,
            p.fwd_rt,
            p.beh_outstanding                    AS outstanding_bal,
            p.beh_capital_pmt                    AS capital_pmt,
            ISNULL(p.prepayment_pmt, 0.0)        AS prepayment_pmt,
            p.beh_interest_pmt                   AS int_pmt,
            ISNULL(r.rate_type,
                   CASE p.product_type WHEN 'F' THEN 'F' ELSE 'V' END
            )                                    AS rate_type
        FROM cf.products p
        LEFT JOIN sched_rt r
            ON r.schedule_id = p.schedule_id
           AND r.product_type = p.product_type
        WHERE p.cf_end_dt > :rd
          AND p.cf_end_dt <= :he
          AND COALESCE(p.beh_total_pmt, 0) <> 0
    """)
    df = pd.read_sql_query(query, engine, params={"rd": report_date, "he": horizon_end})
    df["cf_start_dt"] = pd.to_datetime(df["cf_start_dt"])
    df["cf_end_dt"]   = pd.to_datetime(df["cf_end_dt"])

    # eff_rate: actual client rate = int_pmt / (outstanding * cf_yf) for all products.
    # For fixed products this is the contracted rate (locked at origination).
    # For variable products this is a*fwd_rt + b (market index rate + origination spread).
    # Using contracted_rate here ensures the stable-margin formula in compute_nii_shocked_schedule
    # gives eff_rate_shocked = a*fwd_shocked + b, so delta = a*(fwd_shocked - fwd_rt) with no
    # spurious subtraction of b (which previously caused all shocked NII to be below base).
    denom = df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    contracted_rate = (
        df["int_pmt"].fillna(0.0) / denom.replace(0.0, float("nan"))
    ).fillna(0.0)
    df["eff_rate"] = contracted_rate
    return df


def load_swap_beh_schedules(report_date: pd.Timestamp, horizon_end: pd.Timestamp) -> pd.DataFrame:
    """Load IR swap CF schedules within (report_date, horizon_end].

    Joins cf.ir_swap_beh with sched.ir_swaps to get currency and leg_type.
    Returns DataFrame with the same column layout as load_beh_schedules().
    """
    query = text("""
        SELECT
            'ir_swap'                                           AS source,
            CAST(b.schedule_id AS VARCHAR(8))                  AS schedule_id,
            'S'                                                AS product_type,
            s.product_code,
            b.bs_side,
            s.currency,
            CAST(NULL AS DATE)                                 AS fixing_dt,
            b.cf_start_dt,
            b.cf_end_dt,
            b.cf_yf,
            b.fwd_rt,
            b.outstanding_bal,
            b.capital_pmt,
            0.0                                                AS prepayment_pmt,
            b.int_pmt,
            CASE s.leg_type WHEN 'FIXED' THEN 'F' ELSE 'V' END AS rate_type
        FROM cf.ir_swap_beh b
        JOIN sched.ir_swaps s ON CAST(s.schedule_id AS VARCHAR(8)) = CAST(b.schedule_id AS VARCHAR(8))
        WHERE b.cf_end_dt > :rd
          AND b.cf_end_dt <= :he
          AND ISNULL(b.total_pmt, 0) <> 0
    """)
    df = pd.read_sql_query(query, engine, params={"rd": report_date, "he": horizon_end})
    if df.empty:
        return df
    df["cf_start_dt"] = pd.to_datetime(df["cf_start_dt"])
    df["cf_end_dt"]   = pd.to_datetime(df["cf_end_dt"])

    denom = df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    contracted_rate = (
        df["int_pmt"].fillna(0.0) / denom.replace(0.0, float("nan"))
    ).fillna(0.0)
    df["eff_rate"] = contracted_rate
    return df


def load_nii_summary(report_date: pd.Timestamp) -> pd.DataFrame:
    """Load NII totals per (scenario_id, currency) from irrbb.nii_results.

    Returns columns: scenario_id, currency, nii_total.
    Used to compute CF-based delta_nii = shocked_nii - base_nii.
    """
    query = text("""
        SELECT scenario_id, currency, SUM(nii_total) AS nii_total
        FROM irrbb.nii_results
        WHERE report_date = :rd
        GROUP BY scenario_id, currency
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date})


def load_eve_summary(report_date: pd.Timestamp) -> pd.DataFrame:
    """Load EVE totals per (scenario_id, currency) from irrbb.eve_results.

    Returns columns: scenario_id, currency, pv_total.
    Used to compute CF-based delta_eve = shocked_pv - base_pv.
    """
    query = text("""
        SELECT scenario_id, currency, SUM(pv_total) AS pv_total
        FROM irrbb.eve_results
        WHERE report_date = :rd
        GROUP BY scenario_id, currency
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date})


def reset_nii_results() -> None:
    """Drop and recreate irrbb.nii_results."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS irrbb.nii_results"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[NiiResults], checkfirst=False)


def write_nii_results(df: pd.DataFrame, report_date: pd.Timestamp) -> None:
    """Append schedule-level NII rows to irrbb.nii_results."""
    if df.empty:
        return
    out = df.copy()
    out["report_date"] = report_date
    cols = [c.name for c in NiiResults.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols].where(pd.notnull(out[cols]), None)
    out.to_sql("nii_results", engine, schema="irrbb",
               if_exists="append", index=False, chunksize=10_000)


def load_all_beh_schedules(report_date: pd.Timestamp) -> pd.DataFrame:
    """Load ALL behavioral CF schedules beyond report_date (no horizon cutoff).

    Used for EVE calculation (run-off mode — existing book only, no renewal).
    Same column layout as load_beh_schedules() plus base_df (discount factor
    at cf_end_dt from cf.products).

    Columns
    -------
    source, schedule_id, product_type, product_code, bs_side, currency,
    cf_start_dt, cf_end_dt, cf_yf, fwd_rt, base_df,
    outstanding_bal, capital_pmt, prepayment_pmt, int_pmt,
    rate_type, eff_rate
    """
    query = text("""
        WITH sched_rt AS (
            SELECT schedule_id, ISNULL(rate_type, 'V') AS rate_type, 'L' AS product_type
            FROM sched.loans
            UNION ALL
            SELECT schedule_id, ISNULL(rate_type, 'V') AS rate_type, 'D' AS product_type
            FROM sched.deposits
            UNION ALL
            SELECT schedule_id, ISNULL(rate_type, 'F') AS rate_type, 'F' AS product_type
            FROM sched.fin_inst
        )
        SELECT
            CASE p.product_type
                WHEN 'L' THEN 'loans'
                WHEN 'D' THEN 'deposits'
                ELSE 'fin_inst'
            END AS source,
            p.schedule_id,
            p.product_type,
            p.product_code,
            p.bs_side,
            p.currency,
            p.fixing_dt,
            p.cf_start_dt,
            p.cf_end_dt,
            p.cf_yf,
            p.fwd_rt,
            p.d_f                                        AS base_df,
            p.beh_outstanding                            AS outstanding_bal,
            p.beh_capital_pmt                            AS capital_pmt,
            ISNULL(p.prepayment_pmt, 0.0)                AS prepayment_pmt,
            p.beh_interest_pmt                           AS int_pmt,
            ISNULL(r.rate_type,
                   CASE p.product_type WHEN 'F' THEN 'F' ELSE 'V' END
            )                                            AS rate_type
        FROM cf.products p
        LEFT JOIN sched_rt r
            ON r.schedule_id = p.schedule_id
           AND r.product_type = p.product_type
        WHERE p.cf_end_dt > :rd
          AND COALESCE(p.beh_total_pmt, 0) <> 0
    """)
    df = pd.read_sql_query(query, engine, params={"rd": report_date})
    df["cf_start_dt"] = pd.to_datetime(df["cf_start_dt"])
    df["cf_end_dt"]   = pd.to_datetime(df["cf_end_dt"])

    # eff_rate: F=fixed → contracted rate (int_pmt / (outstanding * cf_yf))
    #           V=variable → fwd_rt (market index rate; margin preserved via reporting, not shock calc)
    denom = df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    contracted_rate = (
        df["int_pmt"].fillna(0.0) / denom.replace(0.0, float("nan"))
    ).fillna(0.0)
    df["eff_rate"] = df["fwd_rt"].fillna(0.0)
    mask_fixed = df["rate_type"] == "F"
    df.loc[mask_fixed, "eff_rate"] = contracted_rate[mask_fixed]
    return df


def load_all_swap_beh_schedules(report_date: pd.Timestamp) -> pd.DataFrame:
    """Load ALL IR swap CF schedules beyond report_date (no horizon cutoff).

    Used for EVE — same layout as load_all_beh_schedules() including base_df.
    """
    query = text("""
        SELECT
            'ir_swap'                                           AS source,
            CAST(b.schedule_id AS VARCHAR(8))                  AS schedule_id,
            'S'                                                AS product_type,
            s.product_code,
            b.bs_side,
            s.currency,
            CAST(NULL AS DATE)                                 AS fixing_dt,
            b.cf_start_dt,
            b.cf_end_dt,
            b.cf_yf,
            b.fwd_rt,
            b.d_f                                              AS base_df,
            b.outstanding_bal,
            b.capital_pmt,
            0.0                                                AS prepayment_pmt,
            b.int_pmt,
            CASE s.leg_type WHEN 'FIXED' THEN 'F' ELSE 'V' END AS rate_type
        FROM cf.ir_swap_beh b
        JOIN sched.ir_swaps s ON CAST(s.schedule_id AS VARCHAR(8)) = CAST(b.schedule_id AS VARCHAR(8))
        WHERE b.cf_end_dt > :rd
          AND ISNULL(b.total_pmt, 0) <> 0
    """)
    df = pd.read_sql_query(query, engine, params={"rd": report_date})
    if df.empty:
        return df
    df["cf_start_dt"] = pd.to_datetime(df["cf_start_dt"])
    df["cf_end_dt"]   = pd.to_datetime(df["cf_end_dt"])

    denom = df["outstanding_bal"].fillna(0.0) * df["cf_yf"].fillna(0.0)
    contracted_rate = (
        df["int_pmt"].fillna(0.0) / denom.replace(0.0, float("nan"))
    ).fillna(0.0)
    df["eff_rate"] = df["fwd_rt"].fillna(0.0)
    mask_fixed = df["rate_type"] == "F"
    df.loc[mask_fixed, "eff_rate"] = contracted_rate[mask_fixed]
    return df


def reset_eve_results() -> None:
    """Drop and recreate irrbb.eve_results."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS irrbb.eve_results"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[EveResults], checkfirst=False)


def write_eve_results(df: pd.DataFrame, report_date: pd.Timestamp) -> None:
    """Append schedule-level EVE rows to irrbb.eve_results."""
    if df.empty:
        return
    out = df.copy()
    out["report_date"] = report_date
    cols = [c.name for c in EveResults.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols].where(pd.notnull(out[cols]), None)
    out.to_sql("eve_results", engine, schema="irrbb",
               if_exists="append", index=False, chunksize=10_000)


def reset_irrbb_report() -> None:
    """Drop and recreate irrbb.irrbb_report."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS irrbb.irrbb_report"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[IrrbbReport], checkfirst=False)


def upsert_irrbb_report(
    df: pd.DataFrame,
    report_date: pd.Timestamp,
    mode: str,  # 'nii' or 'eve'
) -> None:
    """MERGE scenario-level SOT results into irrbb.irrbb_report.

    mode='nii': upserts delta_nii / sot_nii_pct (leaves delta_eve / sot_eve_pct untouched).
    mode='eve': upserts delta_eve / sot_eve_pct (leaves delta_nii / sot_nii_pct untouched).
    df must have columns: scenario_id, currency, and the two mode-specific value columns.
    """
    if df.empty:
        return

    if mode == "nii":
        merge_sql = text("""
            MERGE irrbb.irrbb_report AS t
            USING (VALUES (:rd, :sid, :ccy, :delta_nii, :sot_nii_pct, :delta_nii_reg, :sot_nii_pct_reg))
                AS s(report_date, scenario_id, currency, delta_nii, sot_nii_pct, delta_nii_reg, sot_nii_pct_reg)
            ON  t.report_date = s.report_date
            AND t.scenario_id = s.scenario_id
            AND t.currency    = s.currency
            WHEN MATCHED THEN
                UPDATE SET t.delta_nii       = s.delta_nii,
                           t.sot_nii_pct     = s.sot_nii_pct,
                           t.delta_nii_reg   = s.delta_nii_reg,
                           t.sot_nii_pct_reg = s.sot_nii_pct_reg
            WHEN NOT MATCHED THEN
                INSERT (report_date, scenario_id, currency, delta_nii, sot_nii_pct, delta_nii_reg, sot_nii_pct_reg)
                VALUES (s.report_date, s.scenario_id, s.currency, s.delta_nii, s.sot_nii_pct, s.delta_nii_reg, s.sot_nii_pct_reg);
        """)
        with engine.begin() as conn:
            for _, row in df.iterrows():
                conn.execute(merge_sql, {
                    "rd":              report_date,
                    "sid":             row["scenario_id"],
                    "ccy":             row["currency"],
                    "delta_nii":       float(row["delta_nii"])       if pd.notna(row.get("delta_nii"))       else None,
                    "sot_nii_pct":     float(row["sot_nii_pct"])     if pd.notna(row.get("sot_nii_pct"))     else None,
                    "delta_nii_reg":   float(row["delta_nii_reg"])   if pd.notna(row.get("delta_nii_reg"))   else None,
                    "sot_nii_pct_reg": float(row["sot_nii_pct_reg"]) if pd.notna(row.get("sot_nii_pct_reg")) else None,
                })

    elif mode == "eve":
        merge_sql = text("""
            MERGE irrbb.irrbb_report AS t
            USING (VALUES (:rd, :sid, :ccy, :delta_eve, :sot_eve_pct, :delta_eve_reg, :sot_eve_pct_reg))
                AS s(report_date, scenario_id, currency, delta_eve, sot_eve_pct, delta_eve_reg, sot_eve_pct_reg)
            ON  t.report_date = s.report_date
            AND t.scenario_id = s.scenario_id
            AND t.currency    = s.currency
            WHEN MATCHED THEN
                UPDATE SET t.delta_eve       = s.delta_eve,
                           t.sot_eve_pct     = s.sot_eve_pct,
                           t.delta_eve_reg   = s.delta_eve_reg,
                           t.sot_eve_pct_reg = s.sot_eve_pct_reg
            WHEN NOT MATCHED THEN
                INSERT (report_date, scenario_id, currency, delta_eve, sot_eve_pct, delta_eve_reg, sot_eve_pct_reg)
                VALUES (s.report_date, s.scenario_id, s.currency, s.delta_eve, s.sot_eve_pct, s.delta_eve_reg, s.sot_eve_pct_reg);
        """)
        with engine.begin() as conn:
            for _, row in df.iterrows():
                conn.execute(merge_sql, {
                    "rd":              report_date,
                    "sid":             row["scenario_id"],
                    "ccy":             row["currency"],
                    "delta_eve":       float(row["delta_eve"])       if pd.notna(row.get("delta_eve"))       else None,
                    "sot_eve_pct":     float(row["sot_eve_pct"])     if pd.notna(row.get("sot_eve_pct"))     else None,
                    "delta_eve_reg":   float(row["delta_eve_reg"])   if pd.notna(row.get("delta_eve_reg"))   else None,
                    "sot_eve_pct_reg": float(row["sot_eve_pct_reg"]) if pd.notna(row.get("sot_eve_pct_reg")) else None,
                })


def _ensure_results_schema() -> None:
    with engine.begin() as conn:
        conn.execute(text(
            "IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'results') "
            "EXEC('CREATE SCHEMA results');"
        ))


def reset_results_irrbb_report() -> None:
    """Drop and recreate results.irrbb_report (base + shocked absolute values)."""
    _ensure_results_schema()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS results.irrbb_report"))
    with engine.begin() as conn:
        _results_metadata.create_all(bind=conn, tables=[ResultsIrrbbReport], checkfirst=False)


def write_results_irrbb_report(df: pd.DataFrame) -> None:
    """Append rows to results.irrbb_report."""
    if df.empty:
        return
    out = df.copy()
    cols = [c.name for c in ResultsIrrbbReport.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols].where(pd.notnull(out[cols]), None)
    out.to_sql("irrbb_report", engine, schema="results",
               if_exists="append", index=False, chunksize=10_000)


# Midpoint year-fraction for each EBA tenor bucket (used for d_f_bucket).
# Midpoint = average of the bucket's lower and upper year-fraction boundary.
_BUCKET_MIDPOINT_YF: dict[str, float] = {
    "1D":  0.5 / 365,
    "1M":  (1 / 365 + 1 / 12) / 2,
    "3M":  (1 / 12  + 3 / 12) / 2,
    "6M":  (3 / 12  + 6 / 12) / 2,
    "9M":  (6 / 12  + 9 / 12) / 2,
    "12M": (9 / 12  + 1.0)    / 2,
    "2Y":  (1.0     + 2.0)    / 2,
    "3Y":  (2.0     + 3.0)    / 2,
    "5Y":  (3.0     + 5.0)    / 2,
    "7Y":  (5.0     + 7.0)    / 2,
    "10Y": (7.0     + 10.0)   / 2,
    "15Y": (10.0    + 15.0)   / 2,
    "20Y": (15.0    + 20.0)   / 2,
    "30Y": (20.0    + 30.0)   / 2,
    ">30Y": 35.0,
}

_TENOR_BUCKET_CUTOFFS: list[tuple[float, str]] = [
    (1 / 365,  "1D"),
    (1 / 12,   "1M"),
    (3 / 12,   "3M"),
    (6 / 12,   "6M"),
    (9 / 12,   "9M"),
    (1.0,      "12M"),
    (2.0,      "2Y"),
    (3.0,      "3Y"),
    (5.0,      "5Y"),
    (7.0,      "7Y"),
    (10.0,     "10Y"),
    (15.0,     "15Y"),
    (20.0,     "20Y"),
    (30.0,     "30Y"),
    (float("inf"), ">30Y"),
]


def _assign_tenor_bucket(cf_end_dt: pd.Series, report_date: pd.Timestamp) -> pd.Series:
    """Return EBA-standard tenor bucket label for each cf_end_dt row."""
    yf = (cf_end_dt - report_date).dt.days / 365.0
    buckets = pd.Series(">30Y", index=cf_end_dt.index, dtype="object")
    # Iterate largest-to-smallest so the smallest bucket that covers yf wins.
    for cutoff, label in reversed(_TENOR_BUCKET_CUTOFFS):
        buckets = buckets.where(yf > cutoff, label)
    return buckets


def reset_shocked_cf_products(table_name: str, table_type: str = "eve") -> None:
    """Drop and recreate cf.[table_name] with shocked CF schedule schema.

    table_type='eve'  — used for cf.products_worst_eve:
      pv_capital   — (capital + prepayment) × d_f_shocked × sign
      pv_interest  — beh_interest_pmt × d_f_shocked × sign

    table_type='nii'  — used for cf.products_par_dn:
      remain_yf    — (horizon_end − cf_end_dt) / 365, clipped ≥ 0 (NII horizon = 1 yr)
      ren_client_rt — renewal client rate = a × fwd_rt + b (per-product transform)
      nii_interest — beh_outstanding × client_rt × cf_yf × sign
      nii_renewal  — (beh_capital_pmt + prepayment_pmt) × ren_client_rt × remain_yf × sign
      nii_total    — nii_interest + nii_renewal

    Both types share the common identity columns plus:
      client_rt    — effective shocked client rate (after margins / floor / cap)
      margin       — client_rt minus shocked index fwd_rt
      tenor_bucket — EBA-standard time bucket of cf_end_dt from report_date
      d_f / d_f_start / d_f_bucket — shocked discount factors (curve analytics)
    """
    if table_type == "nii":
        metric_cols = """
                remain_yf        DECIMAL(10, 6)  NULL,
                ren_client_rt    DECIMAL(18, 8)  NULL,
                nii_interest     DECIMAL(18, 2)  NULL,
                nii_renewal      DECIMAL(18, 2)  NULL,
                nii_total        DECIMAL(18, 2)  NULL"""
    else:
        metric_cols = """
                pv_capital       DECIMAL(18, 2)  NULL,
                pv_interest      DECIMAL(18, 2)  NULL"""

    with engine.begin() as conn:
        conn.execute(text(
            f"IF OBJECT_ID('cf.{table_name}', 'U') IS NOT NULL DROP TABLE cf.[{table_name}]"
        ))
        conn.execute(text(f"""
            CREATE TABLE cf.[{table_name}] (
                report_date      DATE            NOT NULL,
                scenario_id      VARCHAR(10)     NOT NULL,
                source           VARCHAR(10)     NOT NULL,
                schedule_id      VARCHAR(8)      NOT NULL,
                product_type     VARCHAR(1)      NOT NULL,
                product_code     VARCHAR(4)      NOT NULL,
                bs_side          VARCHAR(1)      NOT NULL,
                currency         VARCHAR(3)      NOT NULL,
                rate_type        VARCHAR(1)      NULL,
                fixing_dt        DATE            NULL,
                tenor_bucket     VARCHAR(8)      NULL,
                cf_start_dt      DATE            NOT NULL,
                cf_end_dt        DATE            NOT NULL,
                cf_yf            DECIMAL(10, 6)  NULL,
                year_frac        DECIMAL(10, 6)  NULL,
                beh_outstanding  DECIMAL(18, 2)  NULL,
                beh_capital_pmt  DECIMAL(18, 2)  NULL,
                prepayment_pmt   DECIMAL(18, 2)  NULL,
                beh_interest_pmt DECIMAL(18, 2)  NULL,
                client_rt        DECIMAL(18, 8)  NULL,
                fwd_rt           DECIMAL(18, 8)  NULL,
                margin           DECIMAL(18, 8)  NULL,
                d_f              DECIMAL(18, 8)  NULL,
                d_f_start        DECIMAL(18, 8)  NULL,
                d_f_bucket       DECIMAL(18, 8)  NULL,{metric_cols}
            );
        """))


def write_shocked_cf_products(
    df: pd.DataFrame,
    report_date: pd.Timestamp,
    table_name: str,
    shocked_disc_df: pd.Series | None = None,
    disc_curve_map: dict | None = None,
    table_type: str = "eve",
    nii_horizon_yf: float = 1.0,
    coeff_a_map: dict | None = None,
    coeff_b_map: dict | None = None,
    caps_map:    dict | None = None,
    floors_map:  dict | None = None,
) -> None:
    """Append shocked CF rows to cf.[table_name].

    Expects df to contain the columns produced by
    eve_calc_objects.compute_shocked_cf_detail().

    table_type='eve' (default, cf.products_worst_eve):
      pv_capital    — (capital + prepayment) × d_f_shocked × sign
      pv_interest   — int_pmt_shocked × d_f_shocked × sign

    table_type='nii' (cf.products_par_dn):
      remain_yf     — max(0, horizon_end − cf_end_dt) / 365  [horizon = nii_horizon_yf]
      ren_client_rt — renewal client rate = a × fwd_rt + b (per-product transform)
      nii_interest  — beh_outstanding × client_rt × cf_yf × sign
      nii_renewal   — (beh_capital_pmt + prepayment_pmt) × ren_client_rt × remain_yf × sign
      nii_total     — nii_interest + nii_renewal

    Common derived columns (both types):
      client_rt    — eff_rate_shocked (rate charged to / paid by client)
      margin       — client_rt minus base fwd_rt (origination spread)
      tenor_bucket — EBA-standard bucket of cf_end_dt relative to report_date
      d_f_bucket   — shocked d_f at the midpoint date of the tenor bucket
    """
    if df.empty:
        return
    out = df.copy()

    # ── Fix fwd_rt for periods starting before report_date ───────────────────
    # When cf_start_dt < report_date the shocked curve has no value for that
    # start date → d_f_shocked_start = 0 → fwd_rt_shocked collapses to 0.
    # The rate for this period was already fixed at fixing_dt: substitute the
    # base fwd_rt from cf.products (the actual index rate in effect).
    #
    # Use cf_start_dt < report_date as the primary condition (not d_f_shocked_start == 0)
    # because the base scenario sets d_f_shocked_start = 0 for ALL rows (base_df not
    # loaded in NII mode), which would otherwise incorrectly mark future-start CFs as
    # unavailable and force them to use the disc-curve spot rate instead of fwd_rt.
    if "cf_start_dt" in out.columns:
        start_unavail = pd.to_datetime(out["cf_start_dt"]) < report_date
    else:
        start_unavail = (
            out.get("d_f_shocked_start", pd.Series(1.0, index=out.index)).fillna(1.0) == 0
        )
    if start_unavail.any() and "fwd_rt" in out.columns:
        out.loc[start_unavail, "fwd_rt_shocked"] = (
            out.loc[start_unavail, "fwd_rt"].fillna(0.0)
        )

    # ── Derived analytical columns ────────────────────────────────────────────
    # client_rt = rate actually charged/paid by client (includes origination margin).
    # eff_rate_shocked now contains the full client rate for all rate types, so
    # client_rt = eff_rate_shocked directly (no margin addition needed).
    _denom = (
        out.get("outstanding_bal", pd.Series(0.0, index=out.index)).fillna(0.0)
        * out.get("cf_yf", pd.Series(0.0, index=out.index)).fillna(0.0)
    )
    _contracted_rt = (
        out.get("int_pmt", pd.Series(0.0, index=out.index)).fillna(0.0)
        / _denom.replace(0.0, float("nan"))
    ).fillna(0.0)
    _base_fwd    = out.get("fwd_rt", pd.Series(0.0, index=out.index)).fillna(0.0)
    _base_margin = _contracted_rt - _base_fwd
    _eff_rt_sh   = out.get("eff_rate_shocked", pd.Series(0.0, index=out.index)).fillna(0.0)
    _rate_type   = out.get("rate_type", pd.Series("V", index=out.index))
    _mask_fixed  = _rate_type == "F"
    # eff_rate_shocked already includes the origination margin for all rate types.
    # Fixed: contractual rate is unchanged — override to ensure consistency.
    out["client_rt"] = _eff_rt_sh
    out.loc[_mask_fixed, "client_rt"] = _contracted_rt[_mask_fixed]
    out["margin"] = _base_margin
    sign      = out["bs_side"].map({"A": 1.0}).fillna(-1.0)
    cap_total = (out.get("capital_pmt",   pd.Series(0.0, index=out.index)).fillna(0.0)
                 + out.get("prepayment_pmt", pd.Series(0.0, index=out.index)).fillna(0.0))
    out["tenor_bucket"] = _assign_tenor_bucket(out["cf_end_dt"], report_date)
    out["year_frac"]    = (out["cf_end_dt"] - report_date).dt.days / 365.0

    if table_type == "nii":
        import numpy as np
        # NII metric columns — 1-year horizon
        horizon_end = report_date + pd.Timedelta(days=round(nii_horizon_yf * 365.25))
        out["remain_yf"] = (
            (horizon_end - out["cf_end_dt"]).dt.days / 365.0
        ).clip(lower=0.0)
        _fwd_sh = out.get("fwd_rt_shocked", pd.Series(0.0, index=out.index)).fillna(0.0)

        # ── Renewal forward rate ──────────────────────────────────────────────
        # Two classes of CF:
        #
        # 1. Past-start (cf_start_dt < report_date): fwd_rt is the contracted/coupon
        #    rate for fixed products — NOT a current market rate.  The renewal rate
        #    must be derived from the disc curve at cf_end_dt:
        #      fwd_ren = (1 / d_f(cf_end_dt) − 1) / year_frac_from_report_date
        #    Both base and shocked use this formula so delta_nii reflects only Δr.
        #    For base: d_f from shocked_disc_df (which IS the base curve).
        #    For shocked: d_f from d_f_shocked (already in the CF detail df).
        #
        # 2. Future-start (cf_start_dt >= report_date): fwd_rt is the current market
        #    forward rate.  Use fwd_rt_shocked directly (= fwd_rt for base, = shocked
        #    forward rate for shocked scenarios).  This gives consistent base vs shocked
        #    renewal that differs by exactly the shock size.
        #
        # The start_unavail flag is set from cf_start_dt (not d_f_shocked_start)
        # to avoid incorrectly marking base-scenario future-start CFs as unavailable
        # (base_df = 0 in NII mode → d_f_shocked_start = 0 for every row).
        _d_f_sh = out.get("d_f_shocked", pd.Series(0.0, index=out.index)).fillna(0.0)

        # Build d_f_for_ren: prefer d_f_shocked from df (already fetched for
        # shocked scenarios); for the base (d_f_shocked = 0) fetch from the curve.
        _d_f_for_ren = _d_f_sh.to_numpy().copy()
        if shocked_disc_df is not None and disc_curve_map is not None:
            _needs_lookup = start_unavail & (_d_f_sh == 0)
            if _needs_lookup.any():
                _fetched = pd.Series(0.0, index=out.index)
                for _ccy in out.loc[_needs_lookup, "currency"].unique():
                    _cn = disc_curve_map.get(_ccy)
                    if _cn is None:
                        continue
                    _cmask = _needs_lookup & (out["currency"] == _ccy)
                    _mi = pd.MultiIndex.from_arrays(
                        [[_cn] * int(_cmask.sum()),
                         out.loc[_cmask, "cf_end_dt"].to_numpy()],
                        names=["curve_name", "node_date"],
                    )
                    _fetched.loc[_cmask] = (
                        shocked_disc_df.reindex(_mi).fillna(0.0).to_numpy(dtype=float)
                    )
                _d_f_for_ren[_needs_lookup.to_numpy()] = (
                    _fetched.loc[_needs_lookup].to_numpy()
                )

        _shock_ren_valid = start_unavail.to_numpy() & (_d_f_for_ren > 0) & (out["year_frac"].to_numpy() > 0)
        if _shock_ren_valid.any():
            _fwd_ren = _fwd_sh.to_numpy().copy()
            _fwd_ren[_shock_ren_valid] = np.maximum(
                0.0,
                (1.0 / _d_f_for_ren[_shock_ren_valid] - 1.0)
                / out["year_frac"].to_numpy()[_shock_ren_valid],
            )
        else:
            _fwd_ren = _fwd_sh.to_numpy().copy()

        # ── Future-start CF in base scenario: replace contracted/index rate with
        #    disc-curve forward rate ────────────────────────────────────────────
        # For future-start CFs (cf_start_dt >= report_date) the base scenario has
        # d_f_shocked = 0 (compute_base_cf_detail never loads base_df in NII mode).
        # This means _fwd_ren still holds fwd_rt_shocked = fwd_rt, which is:
        #   • FIXED products (1000, 2000): contracted/coupon rate  (e.g. 5.72 %)
        #   • VARIABLE products (1100, 2100): cap-adjusted index rate (e.g. 4.73 %)
        # Both differ from the shocked renewal rate (which comes from irrbb.curves),
        # so delta_nii reflects the coupon/index-vs-market gap instead of just Δr.
        #
        # Fix: derive the forward rate from the BASE disc curve.  Period boundaries
        # mirror _build_shocked_disc_and_fwd exactly:
        #   FIXED / variable without fixing_dt:
        #     fwd_ren = (d_f_base(cf_start_dt) / d_f_base(cf_end_dt) − 1) / cf_yf
        #   Variable with fixing_dt (rate_type == "V"):
        #     period = fixing_dt → max(cf_end_dt) in (schedule_id, fixing_dt) group
        #     fwd_ren = (d_f_base(period_start) / d_f_base(period_end) − 1) / period_yf
        # This ensures base and shocked renewal use identical period boundaries, so
        # delta_nii at 0-shock is zero and delta_nii at ±Xbps reflects only the shock.
        _future_base_mask = ~start_unavail & (_d_f_sh == 0)
        if (
            shocked_disc_df is not None
            and disc_curve_map is not None
            and _future_base_mask.any()
            and "cf_start_dt" in out.columns
        ):
            # Determine period boundaries — mirrors _build_shocked_disc_and_fwd:
            # Variable products with fixing_dt: use the fixing period
            #   (fixing_dt → max(cf_end_dt) within (schedule_id, fixing_dt) group)
            # so that base and shocked renewal use the same period boundaries.
            # All other rows: individual CF period (cf_start_dt → cf_end_dt).
            _period_start = out["cf_start_dt"].copy()
            _period_end   = out["cf_end_dt"].copy()
            _rate_type_fb = out.get("rate_type", pd.Series("V", index=out.index))
            if "fixing_dt" in out.columns and "schedule_id" in out.columns:
                _mask_var_fix = (
                    _future_base_mask
                    & (_rate_type_fb == "V")
                    & out["fixing_dt"].notna()
                )
                if _mask_var_fix.any():
                    _period_start.loc[_mask_var_fix] = pd.to_datetime(
                        out.loc[_mask_var_fix, "fixing_dt"]
                    )
                    _period_end.loc[_mask_var_fix] = (
                        out.loc[_mask_var_fix]
                        .groupby(["schedule_id", "fixing_dt"])["cf_end_dt"]
                        .transform("max")
                    )

            _cf_yf_period = (_period_end - _period_start).dt.days / 365.0
            _valid_fb = _future_base_mask & (_cf_yf_period > 0)
            if _valid_fb.any():
                _d_f_start_fb = pd.Series(0.0, index=out.index)
                _d_f_end_fb   = pd.Series(0.0, index=out.index)
                for _ccy in out.loc[_valid_fb, "currency"].unique():
                    _cn = disc_curve_map.get(_ccy)
                    if _cn is None:
                        continue
                    _cmask = _valid_fb & (out["currency"] == _ccy)
                    _mi_s = pd.MultiIndex.from_arrays(
                        [[_cn] * int(_cmask.sum()),
                         _period_start.loc[_cmask].to_numpy()],
                        names=["curve_name", "node_date"],
                    )
                    _d_f_start_fb.loc[_cmask] = (
                        shocked_disc_df.reindex(_mi_s).fillna(0.0).to_numpy(dtype=float)
                    )
                    _mi_e = pd.MultiIndex.from_arrays(
                        [[_cn] * int(_cmask.sum()),
                         _period_end.loc[_cmask].to_numpy()],
                        names=["curve_name", "node_date"],
                    )
                    _d_f_end_fb.loc[_cmask] = (
                        shocked_disc_df.reindex(_mi_e).fillna(0.0).to_numpy(dtype=float)
                    )
                _fb_ren_valid = (
                    _valid_fb.to_numpy()
                    & (_d_f_start_fb.to_numpy() > 0)
                    & (_d_f_end_fb.to_numpy()   > 0)
                )
                if _fb_ren_valid.any():
                    _fwd_ren[_fb_ren_valid] = np.maximum(
                        0.0,
                        (
                            _d_f_start_fb.to_numpy()[_fb_ren_valid]
                            / _d_f_end_fb.to_numpy()[_fb_ren_valid]
                            - 1.0
                        )
                        / _cf_yf_period.to_numpy()[_fb_ren_valid],
                    )

        # ren_client_rt: per-product linear transform of fwd_rt at renewal
        #   = a x fwd_rt + b  (new-business formula, no base_eff_rate)
        # Same transform used by compute_nii_base_schedule / compute_nii_shocked_schedule.
        # Admin (A) products renew at 0 regardless of market rate.
        if "product_code" in out.columns:
            _ren_rt = _apply_rt_limits(
                _fwd_ren, out["product_code"],
                caps_map=caps_map, floors_map=floors_map,
                coeff_a_map=coeff_a_map, coeff_b_map=coeff_b_map,
            )
        else:
            _ren_rt = _fwd_ren.copy()
        if "rate_type" in out.columns:
            _ren_rt = _ren_rt.copy()
            _ren_rt[(out["rate_type"] == "A").to_numpy()] = 0.0
        out["ren_client_rt"] = _ren_rt

        # ── client_rt override for future-start variable base CFs ────────────────
        # Base scenario sets client_rt = contracted_rt (backed out of int_pmt).
        # Shocked scenarios use the stable-margin formula:
        #   client_rt = contracted_rt + a × (mkt_fwd_shocked − contracted_fwd_rt)
        # For future-start variable CFs in the base this produces a large mismatch:
        # delta_nii_interest = balance × (origination_rate − market_fwd) × yf × sign
        # instead of balance × shock × yf × sign.
        #
        # Fix: apply the same stable-margin formula using the base market forward
        # (held in _fwd_ren for these rows after the future-base block above).
        # After this override, delta_nii_interest ≈ balance × shock × yf × sign.
        if shocked_disc_df is not None:
            _fb_var_int_mask = _future_base_mask & (_rate_type == "V")
            if _fb_var_int_mask.any() and "product_code" in out.columns:
                out.loc[_fb_var_int_mask, "client_rt"] = _apply_rt_limits(
                    pd.Series(_fwd_ren, index=out.index).loc[_fb_var_int_mask],
                    out.loc[_fb_var_int_mask, "product_code"],
                    caps_map=caps_map, floors_map=floors_map,
                    coeff_a_map=coeff_a_map, coeff_b_map=coeff_b_map,
                    base_eff_rate=_contracted_rt.loc[_fb_var_int_mask],
                    base_fwd_rt=_base_fwd.loc[_fb_var_int_mask],
                )
            elif _fb_var_int_mask.any():
                out.loc[_fb_var_int_mask, "client_rt"] = (
                    _contracted_rt.loc[_fb_var_int_mask]
                    + pd.Series(_fwd_ren, index=out.index).loc[_fb_var_int_mask]
                    - _base_fwd.loc[_fb_var_int_mask]
                )

        out["nii_interest"] = (
            out.get("outstanding_bal", pd.Series(0.0, index=out.index)).fillna(0.0)
            * out["client_rt"]
            * out.get("cf_yf", pd.Series(0.0, index=out.index)).fillna(0.0)
            * sign
        )
        out["nii_renewal"] = cap_total * out["ren_client_rt"] * out["remain_yf"] * sign
        out["nii_total"]   = out["nii_interest"] + out["nii_renewal"]
    else:
        # EVE metric columns
        d_f = out.get("d_f_shocked", pd.Series(0.0, index=out.index))
        out["pv_capital"]  = cap_total * d_f * sign
        out["pv_interest"] = out.get("int_pmt_shocked", pd.Series(0.0, index=out.index)).fillna(0.0) * d_f * sign

    # d_f_bucket: shocked d_f at the midpoint date of each tenor bucket.
    # Constant for all CFs sharing the same bucket — lets users verify that
    # discount factors move coherently across the term structure.
    if shocked_disc_df is not None and disc_curve_map is not None:
        bucket_mid_dates = {
            bkt: report_date + pd.Timedelta(days=round(yf * 365))
            for bkt, yf in _BUCKET_MIDPOINT_YF.items()
        }
        df_bucket = pd.Series(float("nan"), index=out.index)
        for ccy in out["currency"].unique():
            cn = disc_curve_map.get(ccy)
            if cn is None:
                continue
            ccy_mask = out["currency"] == ccy
            for bkt, mid_dt in bucket_mid_dates.items():
                bkt_mask = ccy_mask & (out["tenor_bucket"] == bkt)
                if not bkt_mask.any():
                    continue
                mi = pd.MultiIndex.from_arrays(
                    [[cn], [mid_dt]], names=["curve_name", "node_date"]
                )
                val = shocked_disc_df.reindex(mi).iloc[0]
                df_bucket.loc[bkt_mask] = float(val) if pd.notna(val) else float("nan")
        out["d_f_bucket"] = df_bucket
    else:
        out["d_f_bucket"] = float("nan")

    # ── Rename to match table schema ──────────────────────────────────────────
    # Drop base fwd_rt before renaming fwd_rt_shocked → fwd_rt (avoid duplicate)
    out = out.drop(columns=["fwd_rt"], errors="ignore")
    out = out.rename(columns={
        "outstanding_bal":   "beh_outstanding",
        "capital_pmt":       "beh_capital_pmt",
        "int_pmt_shocked":   "beh_interest_pmt",
        "fwd_rt_shocked":    "fwd_rt",
        "d_f_shocked":       "d_f",
        "d_f_shocked_start": "d_f_start",
    })
    out["report_date"] = report_date

    _common_cols = [
        "report_date", "scenario_id", "source", "schedule_id",
        "product_type", "product_code", "bs_side", "currency", "rate_type",
        "fixing_dt", "tenor_bucket", "cf_start_dt", "cf_end_dt", "cf_yf",
        "year_frac",
        "beh_outstanding", "beh_capital_pmt", "prepayment_pmt", "beh_interest_pmt",
        "client_rt", "fwd_rt", "margin", "d_f", "d_f_start", "d_f_bucket",
    ]
    if table_type == "nii":
        table_cols = _common_cols + ["remain_yf", "ren_client_rt", "nii_interest", "nii_renewal", "nii_total"]
    else:
        table_cols = _common_cols + ["pv_capital", "pv_interest"]
    out = out[[c for c in table_cols if c in out.columns]]
    out.to_sql(table_name, engine, schema="cf",
               if_exists="append", index=False, chunksize=10_000)


def load_tier1_capital(report_date: pd.Timestamp) -> float:
    """Sum all equity balances for report_date from schemat.equity as Tier 1 capital."""
    result = pd.read_sql_query(
        text("SELECT ISNULL(SUM(balance_amt), 0) AS tier1 FROM schemat.equity WHERE report_date = :rd"),
        engine, params={"rd": report_date},
    )
    return float(result["tier1"].iloc[0])
