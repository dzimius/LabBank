"""SQL setup for liq_calc — results schema with two separate tables:
  results.lcr_nsfr    — LCR and NSFR metrics only
  results.irrbb_report — NII and EVE metrics per scenario
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, String, Date, MetaData
from sqlalchemy.dialects.mssql import DECIMAL

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

metadata = MetaData(schema=None)

# ── LCR / NSFR — one row per (report_date, currency) ────────────────────────
LcrNsfr = Table(
    "lcr_nsfr", metadata,
    Column("report_date",       Date,           nullable=False),
    Column("currency",          String(3),      nullable=False),
    Column("hqla",              DECIMAL(18, 2), nullable=True),
    Column("outflows_30d",      DECIMAL(18, 2), nullable=True),
    Column("inflows_30d",       DECIMAL(18, 2), nullable=True),
    Column("net_outflows_30d",  DECIMAL(18, 2), nullable=True),
    Column("lcr",               DECIMAL(8,  4), nullable=True),
    Column("asf",               DECIMAL(18, 2), nullable=True),
    Column("rsf",               DECIMAL(18, 2), nullable=True),
    Column("nsfr",              DECIMAL(8,  4), nullable=True),
    schema="results",
)

# ── IRRBB — one row per (report_date, currency, scenario_id) ─────────────────
IrrbbReport = Table(
    "irrbb_report", metadata,
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


def _ensure_results_schema() -> None:
    with engine.begin() as conn:
        conn.execute(text(
            "IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'results') "
            "EXEC('CREATE SCHEMA results');"
        ))


def reset_lcr_nsfr() -> None:
    _ensure_results_schema()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS results.lcr_nsfr"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[LcrNsfr], checkfirst=False)


def reset_irrbb_report() -> None:
    _ensure_results_schema()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS results.irrbb_report"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[IrrbbReport], checkfirst=False)


def _write(df: pd.DataFrame, table: Table, table_name: str, schema: str) -> None:
    if df.empty:
        return
    out = df.copy()
    cols = [c.name for c in table.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols].where(pd.notnull(out[cols]), None)
    out.to_sql(table_name, engine, schema=schema,
               if_exists="append", index=False, chunksize=10_000)


def write_lcr_nsfr(df: pd.DataFrame) -> None:
    _write(df, LcrNsfr, "lcr_nsfr", "results")


def write_irrbb_report(df: pd.DataFrame) -> None:
    _write(df, IrrbbReport, "irrbb_report", "results")


def load_tier1_capital(report_date: pd.Timestamp) -> float:
    """Sum all equity balances for report_date from schemat.equity as Tier 1 capital."""
    result = pd.read_sql_query(
        text("SELECT ISNULL(SUM(balance_amt), 0) AS tier1 FROM schemat.equity WHERE report_date = :rd"),
        engine, params={"rd": report_date},
    )
    return float(result["tier1"].iloc[0])


def load_nii_by_scenario(report_date: pd.Timestamp) -> pd.DataFrame:
    query = text("""
        SELECT currency, scenario_id, SUM(nii_total) AS nii_total
        FROM irrbb.nii_results
        WHERE report_date = :rd
        GROUP BY currency, scenario_id
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date})


def load_eve_by_scenario(report_date: pd.Timestamp) -> pd.DataFrame:
    query = text("""
        SELECT currency, scenario_id, SUM(pv_total) AS pv_total
        FROM irrbb.eve_results
        WHERE report_date = :rd
        GROUP BY currency, scenario_id
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date})


def load_td_maturity_split(report_date: pd.Timestamp) -> pd.DataFrame:
    """Load L-side deposit balances split by whether they mature within 30 days.

    Returns DataFrame: currency, within_30d_amt, after_30d_amt
    Used to apply 100% LCR run-off to deposits maturing within the stress window.
    """
    horizon_30d = report_date + pd.Timedelta(days=30)
    query = text("""
        SELECT currency,
               SUM(CASE WHEN maturity_date <= :h30 THEN balance_amt ELSE 0 END) AS within_30d_amt,
               SUM(CASE WHEN maturity_date >  :h30 THEN balance_amt ELSE 0 END) AS after_30d_amt
        FROM schemat.deposits
        WHERE bs_side = 'L'
          AND report_date = :rd
        GROUP BY currency
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date, "h30": horizon_30d})


def load_30d_asset_inflows(report_date: pd.Timestamp) -> dict[str, float]:
    import pandas as _pd
    horizon_30d = report_date + _pd.Timedelta(days=30)
    query = text("""
        SELECT currency,
               SUM(COALESCE(beh_capital_pmt,0) + COALESCE(beh_interest_pmt,0)) AS inflow_amt
        FROM cf.products
        WHERE bs_side = 'A'
          AND cf_end_dt > :rd AND cf_end_dt <= :h30
          AND COALESCE(beh_total_pmt, 0) <> 0
        GROUP BY currency
    """)
    df = pd.read_sql_query(query, engine, params={"rd": report_date, "h30": horizon_30d})
    return dict(zip(df["currency"], df["inflow_amt"]))
