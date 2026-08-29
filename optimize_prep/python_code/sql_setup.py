"""sql_setup.py
==============
SQL schema and table definitions for the opt_prep schema.

Schema: opt_prep
Tables
------
  opt_prep.monthly_curves         — pre-interpolated monthly disc/fwd curve grid
  opt_prep.product_params         — per-product base parameters (one row per product)
  opt_prep.product_scenario_params— per-product × scenario sensitivity (delta NII, delta EVE)

Follows the same pattern as irrbb_calc/sql_setup.py:
  reset_*()   — drop + recreate table
  write_*()   — bulk-insert DataFrame
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import (
    create_engine, text, Table, Column, MetaData,
    Integer, String, Date, Boolean,
)
from sqlalchemy.dialects.mssql import DECIMAL

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

metadata = MetaData(schema=None)

# ─────────────────────────────────────────────────────────────────────────────
# Table definitions
# ─────────────────────────────────────────────────────────────────────────────

MonthlyCurves = Table(
    "monthly_curves", metadata,
    # Identifies which report run this row belongs to
    Column("report_date",   Date,           nullable=False),
    # One of: base, par_up, par_dn, steep, flat, sr_up, sr_dn, own, own_*
    Column("scenario_id",   String(20),     nullable=False),
    # Currency the curve belongs to (PLN, EUR, USD)
    Column("currency",      String(3),      nullable=False),
    # Full curve name as stored in irrbb.curves (e.g. PLN_disc_curve)
    Column("curve_name",    String(30),     nullable=False),
    # Month index 1..360 (1 = 1 month from report_date, 360 = 30 years)
    Column("month_idx",     Integer,        nullable=False),
    # Discount factor at this month (log-linear interpolated from daily nodes)
    Column("disc_factor",   DECIMAL(18, 8), nullable=False),
    # Annualised forward rate for the 1-month period ending at month_idx
    # = (disc_factor[m-1] / disc_factor[m] - 1) × 12
    Column("fwd_rate_ann",  DECIMAL(18, 8), nullable=False),
    schema="opt_prep",
)

ProductParams = Table(
    "product_params", metadata,
    Column("report_date",       Date,           nullable=False),
    # Unique optimizer entry identifier  e.g. "1000_A_PLN_2024_01" or "6000_L_PLN"
    Column("cohort_id",         String(30),     nullable=False),
    # From bs_structure
    Column("product_code",      String(4),      nullable=False),
    Column("product_name",      String(64),     nullable=True),
    Column("bs_side",           String(1),      nullable=False),   # A / L / E
    Column("currency",          String(3),      nullable=False),
    # True = monthly cohort row, False = single-row / behavioural product
    Column("is_cohort",         Boolean,        nullable=True),
    # Monthly cohort start year/month (NULL for single-row / behavioural products)
    Column("start_year",        Integer,        nullable=True),
    Column("start_month",       Integer,        nullable=True),
    # Current allocation weight (from bs_structure.bs_percentage)
    Column("bs_pct_current",    DECIMAL(8, 4),  nullable=True),
    # Actual balance at report_date (from schemat.* tables, PLN)
    Column("balance_amt",       DECIMAL(18, 2), nullable=True),
    # NII per PLN of balance over 12M horizon (base scenario)
    # = Σ(nii_total_base) / Σ(balance) for this product group
    Column("nii_unit_rate",     DECIMAL(18, 8), nullable=True),
    # EVE present value per PLN of balance (base scenario)
    # = Σ(pv_total_base) / Σ(balance)
    Column("eve_pv_factor",     DECIMAL(18, 8), nullable=True),
    # Modified duration (years, signed: positive for assets, negative for liabilities)
    # = -(pv_par_dn - pv_par_up) / (2 × Δr_par × balance)
    Column("d_mod",             DECIMAL(10, 4), nullable=True),
    # LCR / NSFR factors from bs_structure
    Column("hqla_factor",       DECIMAL(8, 4),  nullable=True),   # (1-haircut) for HQLA assets
    Column("lcr_runoff",        DECIMAL(8, 4),  nullable=True),   # run-off rate (liabilities)
    Column("asf_factor",        DECIMAL(8, 4),  nullable=True),   # ASF factor (liabilities/equity)
    Column("rsf_factor",        DECIMAL(8, 4),  nullable=True),   # RSF factor (assets)
    # From cf.products aggregation
    Column("inflow_30d_frac",   DECIMAL(8, 4),  nullable=True),   # 30-day inflow / balance (assets)
    Column("amort_frac_1y",     DECIMAL(8, 4),  nullable=True),   # 1Y capital repayment / balance
    Column("repricing_tenor_m", DECIMAL(8, 2),  nullable=True),   # months to first repricing
    # From interest_rt.xlsx
    Column("coeff_a",           DECIMAL(8, 4),  nullable=True),   # rate pass-through coefficient
    Column("coeff_b",           DECIMAL(10, 6), nullable=True),   # spread (decimal, not bps)
    Column("client_floor",      DECIMAL(10, 6), nullable=True),   # client rate floor
    Column("client_cap",        DECIMAL(10, 6), nullable=True),   # client rate cap (NULL = no cap)
    # For formula-based NII (cohort products only)
    Column("rate_type",         String(1),      nullable=True),   # 'F' fixed / 'V' floating; NULL for single-row
    Column("coupon_rate",       DECIMAL(10, 6), nullable=True),   # contracted rate (fixed cohorts); NULL for floating/single-row
    schema="opt_prep",
)

ProductScenarioParams = Table(
    "product_scenario_params", metadata,
    Column("report_date",    Date,           nullable=False),
    Column("cohort_id",      String(30),     nullable=False),
    Column("product_code",   String(4),      nullable=False),
    Column("bs_side",        String(1),      nullable=False),
    Column("currency",       String(3),      nullable=False),
    # EBA SOT scenario: par_up, par_dn, steep, flat, sr_up, sr_dn
    Column("scenario_id",    String(20),     nullable=False),
    # delta_NII per PLN of balance = (nii_shocked - nii_base) / balance
    Column("delta_nii_unit", DECIMAL(18, 8), nullable=True),
    # delta_EVE per PLN of balance = (pv_shocked - pv_base) / balance
    Column("delta_eve_unit", DECIMAL(18, 8), nullable=True),
    schema="opt_prep",
)


# ─────────────────────────────────────────────────────────────────────────────
# Schema creation
# ─────────────────────────────────────────────────────────────────────────────

def ensure_schema() -> None:
    """Create opt_prep schema if it does not exist."""
    with engine.begin() as conn:
        conn.execute(text(
            "IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'opt_prep') "
            "EXEC('CREATE SCHEMA [opt_prep]');"
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Reset (drop + recreate) helpers
# ─────────────────────────────────────────────────────────────────────────────

def reset_monthly_curves() -> None:
    ensure_schema()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS opt_prep.monthly_curves"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[MonthlyCurves], checkfirst=False)


def reset_product_params() -> None:
    ensure_schema()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS opt_prep.product_params"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[ProductParams], checkfirst=False)


def reset_product_scenario_params() -> None:
    ensure_schema()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS opt_prep.product_scenario_params"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[ProductScenarioParams], checkfirst=False)


def reset_all() -> None:
    """Drop and recreate all opt_prep tables."""
    reset_monthly_curves()
    reset_product_params()
    reset_product_scenario_params()


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write(df: pd.DataFrame, table: Table, chunksize: int = 10_000) -> None:
    """Generic bulk-insert with NULL coercion."""
    if df.empty:
        return
    cols = [c.name for c in table.columns]
    out  = df[[c for c in cols if c in df.columns]].copy()
    # Fill missing expected columns with None
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols].where(pd.notnull(out[cols]), None)
    out.to_sql(
        table.name, engine,
        schema=table.schema,
        if_exists="append",
        index=False,
        chunksize=chunksize,
    )


def write_monthly_curves(df: pd.DataFrame) -> None:
    """Append rows to opt_prep.monthly_curves.

    Expected columns: report_date, scenario_id, currency, curve_name,
                      month_idx, disc_factor, fwd_rate_ann
    """
    _write(df, MonthlyCurves)


def write_product_params(df: pd.DataFrame) -> None:
    """Append rows to opt_prep.product_params."""
    _write(df, ProductParams)


def write_product_scenario_params(df: pd.DataFrame) -> None:
    """Append rows to opt_prep.product_scenario_params."""
    _write(df, ProductScenarioParams)


# ─────────────────────────────────────────────────────────────────────────────
# cohort_cf_compare — fast vs exact CF drill-down table
# ─────────────────────────────────────────────────────────────────────────────

CohortCfCompare = Table(
    "cohort_cf_compare", metadata,
    Column("report_date",     Date,           nullable=False),
    Column("cohort_id",       String(30),     nullable=False),
    Column("product_code",    String(4),      nullable=False),
    Column("bs_side",         String(1),      nullable=False),
    Column("currency",        String(3),      nullable=False),
    Column("rate_type",       String(1),      nullable=True),
    # 'fast_m'=monthly NII buckets, 'fast_q'=quarterly EVE buckets, 'exact'=cf.products
    Column("cf_type",         String(8),      nullable=False),
    Column("scenario_id",     String(20),     nullable=False),
    # month 0-11 for fast_m; quarter index for fast_q; NULL for exact
    Column("bucket_idx",      Integer,        nullable=True),
    # calendar dates
    Column("cf_start_dt",     Date,           nullable=True),
    Column("cf_end_dt",       Date,           nullable=True),
    # year-fraction from report_date to CF midpoint / end
    Column("yf_from_report",  DECIMAL(10, 6), nullable=True),
    # cashflow amounts (PLN, absolute — sign applied via sign column for income/expense)
    Column("outstanding",     DECIMAL(18, 2), nullable=True),
    Column("capital_pmt",     DECIMAL(18, 2), nullable=True),
    Column("interest_pmt",    DECIMAL(18, 2), nullable=True),
    # rates (decimal, annualised)
    Column("effective_rate",  DECIMAL(18, 8), nullable=True),
    Column("fwd_rt",          DECIMAL(18, 8), nullable=True),
    Column("margin",          DECIMAL(18, 8), nullable=True),
    # NII components (signed: income +, expense -)
    Column("nii_interest",    DECIMAL(18, 2), nullable=True),
    Column("remain_yf",       DECIMAL(10, 6), nullable=True),
    Column("nii_renewal",     DECIMAL(18, 2), nullable=True),
    Column("nii_total",       DECIMAL(18, 2), nullable=True),
    # EVE components (signed)
    Column("disc_factor",     DECIMAL(18, 8), nullable=True),
    Column("pv_capital",      DECIMAL(18, 2), nullable=True),
    Column("pv_interest",     DECIMAL(18, 2), nullable=True),
    Column("pv_total",        DECIMAL(18, 2), nullable=True),
    schema="opt_prep",
)


def reset_cohort_cf_compare() -> None:
    ensure_schema()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS opt_prep.cohort_cf_compare"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[CohortCfCompare], checkfirst=False)


def write_cohort_cf_compare(df: pd.DataFrame) -> None:
    """Append rows to opt_prep.cohort_cf_compare."""
    _write(df, CohortCfCompare)


# ─────────────────────────────────────────────────────────────────────────────
# ftp_rates — Funds Transfer Pricing rate per cohort (see ftp_store.py)
# ─────────────────────────────────────────────────────────────────────────────

FtpRates = Table(
    "ftp_rates", metadata,
    Column("report_date",  Date,           nullable=False),
    Column("cohort_id",    String(30),     nullable=False),
    # market-rate component: floating = today's curve fwd rate at fixing tenor;
    # fixed = historical NS zero rate at origination, for the original tenor
    Column("market_rt",    DECIMAL(18, 8), nullable=True),
    # liquidity_spread(tenor): 0% at 0M -> 0.5% at 120M, capped beyond
    Column("liq_spread",   DECIMAL(18, 8), nullable=True),
    # ftp_rate = market_rt + liq_spread (decimal, e.g. 0.035 = 3.5%)
    Column("ftp_rate",     DECIMAL(18, 8), nullable=True),
    schema="opt_prep",
)


def reset_ftp_rates() -> None:
    ensure_schema()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS opt_prep.ftp_rates"))
    with engine.begin() as conn:
        metadata.create_all(bind=conn, tables=[FtpRates], checkfirst=False)


def write_ftp_rates(df: pd.DataFrame) -> None:
    """Append rows to opt_prep.ftp_rates."""
    _write(df, FtpRates)
