"""SQL setup for liq_calc — results schema with two separate tables:
  results.lcr_nsfr    — LCR and NSFR metrics only
  results.irrbb_report — NII and EVE metrics per scenario
"""
from __future__ import annotations

import os
import sys
import pandas as pd
from sqlalchemy import text, Table, Column, String, Date, MetaData
from sqlalchemy.dialects.mssql import DECIMAL

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common.bank_config import get_engine
from labbank_common import run_store  # noqa: E402  history per (report_date, run_id), docs/04

engine = get_engine()   # active bank's database (LABBANK_BANK, default eur)

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
    run_store.reset_table(engine, "lcr_nsfr", "results", run_store.sa_create(metadata, LcrNsfr))


def reset_irrbb_report() -> None:
    _ensure_results_schema()
    run_store.reset_table(engine, "irrbb_report", "results", run_store.sa_create(metadata, IrrbbReport))


def _write(df: pd.DataFrame, table: Table, table_name: str, schema: str) -> None:
    if df.empty:
        return
    out = df.copy()
    cols = [c.name for c in table.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols].where(pd.notnull(out[cols]), None)
    out = run_store.stamp(out, run_store.current_run())
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
    """L-side term-deposit balances per (currency, product_code) split by contractual
    maturity within / after 30 days (non-maturity deposits have no maturity_date and fall
    in neither). Per product so counterparties are not pooled (labbank_common.lcr_rules).

    Returns DataFrame: currency, product_code, within_30d_amt, after_30d_amt
    """
    horizon_30d = report_date + pd.Timedelta(days=30)
    query = text("""
        SELECT currency, CAST(product_code AS VARCHAR(4)) AS product_code,
               SUM(CASE WHEN maturity_date <= :h30 THEN balance_amt ELSE 0 END) AS within_30d_amt,
               SUM(CASE WHEN maturity_date >  :h30 THEN balance_amt ELSE 0 END) AS after_30d_amt
        FROM schemat.deposits
        WHERE bs_side = 'L'
          AND report_date = :rd
        GROUP BY currency, product_code
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date, "h30": horizon_30d})


def load_30d_asset_inflows(report_date: pd.Timestamp, exclude_codes) -> pd.DataFrame:
    """CONTRACTUAL 30-day asset inflows per (currency, product_code), without the products
    in exclude_codes (labbank_common.lcr_rules.non_inflow_codes: HQLA + cards). Regulatory
    inflows are contractual receipts -- the behavioural (prepayment-adjusted) schedule is a
    liquidity forecast, not an eligible LCR inflow (it used beh_* before)."""
    import pandas as _pd
    horizon_30d = report_date + _pd.Timedelta(days=30)
    excl = ", ".join(f"'{str(c)}'" for c in exclude_codes) or "''"
    query = text(f"""
        SELECT currency, CAST(product_code AS VARCHAR(4)) AS product_code,
               SUM(COALESCE(con_capital_pmt,0) + COALESCE(con_interest_pmt,0)) AS inflow_amt
        FROM cf.products
        WHERE bs_side = 'A'
          AND CAST(product_code AS VARCHAR(4)) NOT IN ({excl})
          AND cf_end_dt > :rd AND cf_end_dt <= :h30
          AND COALESCE(con_total_pmt, 0) <> 0
        GROUP BY currency, product_code
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date, "h30": horizon_30d})


def load_hqla_market_value_ratio(report_date: pd.Timestamp, hqla_codes, spreads_bp=None) -> dict[str, float]:
    """labbank_common.lcr_rules.hqla_market_value_ratio on this bank's database."""
    from labbank_common.lcr_rules import hqla_market_value_ratio
    return hqla_market_value_ratio(engine, report_date, hqla_codes, spreads_bp)


# ── Pillar 2 liquidity (liquidity_p2.py) ─────────────────────────────────────────

def load_deposits_by_client(report_date: pd.Timestamp) -> pd.DataFrame:
    """L-side deposits with their client_id (dbo.transactions), own currency units.
    Columns: client_id, client_type_id, product_code, product_name, currency, balance_amt."""
    query = text("""
        SELECT t.client_id, d.client_type_id, CAST(d.product_code AS VARCHAR(4)) AS product_code,
               d.product_name, d.currency, d.balance_amt
        FROM schemat.deposits d
        JOIN dbo.transactions t ON t.transaction_id = d.transaction_id
        WHERE d.bs_side = 'L' AND d.report_date = :rd
        OPTION (MAXDOP 1)
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date})


def load_issued_debt(report_date: pd.Timestamp) -> pd.DataFrame:
    """L-side financial instruments (issued bonds) per product, own currency units.
    Columns: product_code, product_name, currency, balance_amt."""
    query = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code, product_name, currency,
               SUM(balance_amt) AS balance_amt
        FROM schemat.financial_instruments
        WHERE bs_side = 'L' AND report_date = :rd
        GROUP BY product_code, product_name, currency
    """)
    return pd.read_sql_query(query, engine, params={"rd": report_date})


def load_behavioural_flows(report_date: pd.Timestamp, exclude_codes) -> pd.DataFrame:
    """Behavioural cash flows of the whole book after the report date, own currency units.

    Non-maturity deposits that have a liquidity model come from cf.products_liq; every
    other product (loans with prepayments, securities, term deposits, issued bonds, deposits
    without a liquidity model) from cf.products. exclude_codes (HQLA) are left out.
    Columns: product_code, currency, bs_side, cf_end_dt, total_pmt, principal_pmt."""
    excl = ", ".join(f"'{str(c)}'" for c in exclude_codes) or "''"
    query = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code, currency, bs_side, cf_end_dt,
               COALESCE(beh_capital_pmt, 0) + COALESCE(beh_interest_pmt, 0) AS total_pmt,
               COALESCE(beh_capital_pmt, 0) AS principal_pmt
        FROM cf.products_liq
        WHERE cf_end_dt > :rd

        UNION ALL

        SELECT CAST(product_code AS VARCHAR(4)), currency, bs_side, cf_end_dt,
               COALESCE(beh_capital_pmt, 0) + COALESCE(prepayment_pmt, 0) + COALESCE(beh_interest_pmt, 0),
               COALESCE(beh_capital_pmt, 0) + COALESCE(prepayment_pmt, 0)
        FROM cf.products
        WHERE cf_end_dt > :rd
          AND CAST(product_code AS VARCHAR(4)) NOT IN ({excl})
          AND product_code NOT IN (SELECT DISTINCT product_code FROM cf.products_liq)
    """)
    df = pd.read_sql_query(query, engine, params={"rd": report_date})
    df["cf_end_dt"] = pd.to_datetime(df["cf_end_dt"])
    return df
