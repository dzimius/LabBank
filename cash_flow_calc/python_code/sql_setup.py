from __future__ import annotations

import math
from typing import Generator, Iterator, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, event, MetaData, Table, Column
from sqlalchemy import Integer, String, ForeignKey, Date, text, DECIMAL
from sqlalchemy.engine import Engine

import config


engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)
metadata = MetaData()

# cf schema — single wide table combining contractual and behavioural cash flows.
# product_type: 'L'=loan, 'F'=financial instrument, 'D'=deposit.
# con_* columns: contractual (orig) schedule.
# beh_* columns: behavioural schedule.
# comp_* columns: behavioural component = beh - con.
# prepayment_pmt: loan debugging column, NULL for all other product types.
Products_d = Table(
    "products", metadata,
    Column("schedule_id",      String(8),      primary_key=True, nullable=False),
    Column("product_type",     String(1),      primary_key=True, nullable=False),
    Column("product_code",     String(4),      nullable=False),
    Column("currency",         String(3),      nullable=False),
    Column("bs_side",          String(1),      nullable=False),
    Column("rate_index",       String(10),     nullable=True),
    Column("fixing_dt",        Date,           nullable=True),
    Column("cf_start_dt",      Date,           primary_key=True, nullable=False),
    Column("cf_start_dt_delay", Date,          nullable=True),
    Column("cf_end_dt",        Date,           nullable=False),
    Column("cf_yf",            DECIMAL(18, 6), nullable=False),
    Column("d_f",              DECIMAL(18, 6), nullable=True),
    Column("fwd_rt",           DECIMAL(18, 6), nullable=True),
    Column("margin",           DECIMAL(6, 4),  nullable=True),
    Column("client_rt",        DECIMAL(18, 6), nullable=True),
    Column("con_outstanding",  DECIMAL(18, 2), nullable=True),
    Column("con_capital_pmt",  DECIMAL(18, 2), nullable=True),
    Column("con_interest_pmt", DECIMAL(18, 2), nullable=True),
    Column("con_total_pmt",    DECIMAL(18, 2), nullable=True),
    Column("beh_outstanding",  DECIMAL(18, 2), nullable=True),
    Column("beh_capital_pmt",  DECIMAL(18, 2), nullable=True),
    Column("beh_interest_pmt", DECIMAL(18, 2), nullable=True),
    Column("beh_total_pmt",    DECIMAL(18, 2), nullable=True),
    Column("comp_capital_pmt",  DECIMAL(18, 2), nullable=True),
    Column("comp_interest_pmt", DECIMAL(18, 2), nullable=True),
    Column("comp_total_pmt",    DECIMAL(18, 2), nullable=True),
    Column("prepayment_pmt",   DECIMAL(18, 2), nullable=True),
    schema="cf",
)

# cf.products_liq — deposit-only LIQ behavioral cashflows.
# Same row structure as cf.products but without contractual (con_*) and
# component (comp_*) columns.  Populated from bs.models_deposit_liq.
Products_d_liq = Table(
    "products_liq", metadata,
    Column("schedule_id",      String(8),      primary_key=True, nullable=False),
    Column("product_type",     String(1),      primary_key=True, nullable=False),
    Column("product_code",     String(4),      nullable=False),
    Column("currency",         String(3),      nullable=False),
    Column("bs_side",          String(1),      nullable=False),
    Column("cf_start_dt",      Date,           primary_key=True, nullable=False),
    Column("cf_end_dt",        Date,           nullable=False),
    Column("cf_yf",            DECIMAL(18, 6), nullable=False),
    Column("d_f",              DECIMAL(18, 6), nullable=True),
    Column("fwd_rt",           DECIMAL(18, 6), nullable=True),
    Column("client_rt",        DECIMAL(18, 6), nullable=True),
    Column("beh_outstanding",  DECIMAL(18, 2), nullable=True),
    Column("beh_capital_pmt",  DECIMAL(18, 2), nullable=True),
    Column("beh_interest_pmt", DECIMAL(18, 2), nullable=True),
    Column("beh_total_pmt",    DECIMAL(18, 2), nullable=True),
    schema="cf",
)

_CF_TABLES    = ["products", "products_liq"]
_IRRBB_TABLES = ["liq_gap_orig", "liq_gap_beh", "ir_gap_orig", "ir_gap_beh", "ir_gap_beh_a"]


def _ensure_schemas(conn) -> None:
    for schema in ("cf", "irrbb"):
        conn.execute(text(
            f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{schema}') "
            f"EXEC('CREATE SCHEMA {schema}');"
        ))


def reset_data(mode: int, report_date=None) -> None:
    """
    mode=0 -> drop + recreate cf/irrbb tables
    mode=1 -> DELETE specific report_date data
    """
    with engine.begin() as conn:
        _ensure_schemas(conn)

        if mode == 0:
            for t in _CF_TABLES:
                conn.execute(text(f"IF OBJECT_ID('cf.{t}', 'U') IS NOT NULL DROP TABLE cf.{t};"))
            for t in _IRRBB_TABLES:
                conn.execute(text(f"IF OBJECT_ID('irrbb.{t}', 'U') IS NOT NULL DROP TABLE irrbb.{t};"))
            metadata.create_all(bind=conn, checkfirst=False)

        elif mode == 1:
            if report_date is None:
                raise ValueError("For mode=1 report_date is needed")
            for t in _CF_TABLES:
                conn.execute(text(f"DELETE FROM cf.{t} WHERE report_date = :rd"), {"rd": report_date})
            for t in _IRRBB_TABLES:
                conn.execute(text(f"DELETE FROM irrbb.{t} WHERE report_date = :rd"), {"rd": report_date})
        else:
            raise ValueError("Mode should be 0 or 1")



def sql_get_params(
    engine: Engine,
    table_name: str,
    columns: list[str],
    chunksize: int = 50_000,
    schema: str = "dbo",
) -> Iterator[pd.DataFrame]:
    where_clauses = []
    params = []

    query = f"""
    SELECT {", ".join(columns)}
    FROM {schema}.{table_name}
    """

    return pd.read_sql_query(query, engine, params=tuple(params), chunksize=chunksize)

def load_sched_params(table: str, cols: list[str], schema: str = "sched") -> Iterator[pd.DataFrame]:
    return sql_get_params(engine=engine, table_name=table, columns=cols, schema=schema)

@event.listens_for(engine, "before_cursor_execute")
def _set_fast_executemany(conn, cursor, statement, parameters, context, executemany):
    if executemany:
        cursor.fast_executemany = True


def write_df(df: pd.DataFrame, table: str, schema: str = "dbo", chunksize: int = 50_000):
    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method=None,
    )


def iter_schedule_batches(
    engine: Engine,
    table_name: str,
    batch_size: int = 1000,
    schema: str = "dbo",
) -> Generator[pd.DataFrame, None, None]:
    query = f"""
    WITH ids AS (
        SELECT DISTINCT schedule_id,
               ROW_NUMBER() OVER (ORDER BY schedule_id) AS rn
        FROM {schema}.{table_name}
    )
    SELECT s.*
    FROM {schema}.{table_name} s
    JOIN ids i ON s.schedule_id = i.schedule_id
    WHERE i.rn BETWEEN ? AND ?
    ORDER BY s.schedule_id, s.cf_start_dt
    """

    start = 1

    while True:
        df = pd.read_sql(query, engine, params=[start, start + batch_size - 1])
        if df.empty:
            break
        yield df
        start += batch_size

def sql_get_uniq_curves(tables: list[str], curve_type: str) -> pd.DataFrame:
    union_parts = []

    for table in tables:
        union_parts.append(f"""
            SELECT {curve_type} as curve_name, case when '{curve_type}'='fwd_curve' then fixing_freq
            else NULL end as fixing_freq, '{curve_type}' as curve_type
            FROM sched.{table}
            GROUP BY {curve_type}, fixing_freq
        """)

    union_sql = "\nUNION ALL\n".join(union_parts)

    query = f"""
    SELECT DISTINCT t.curve_name, t.fixing_freq, t.curve_type
    FROM (
        {union_sql}
    ) t
    """
    return pd.read_sql_query(query, engine)

def sql_select_specific_curve(curve_name: str) -> pd.DataFrame:
    query = text("""
        SELECT curve_name, n_days, year_frac, zero_rate, d_f
        FROM mkt.curves
        WHERE curve_date = :report_date
          AND curve_name = :curve_name
    """)
    return pd.read_sql_query(
        query,
        engine,
        params={"report_date": config.report_date, "curve_name": curve_name}
    )

def sql_select_fixings() -> pd.DataFrame:
    query = "SELECT * FROM mkt.fixings"
    return pd.read_sql_query(query, engine)

def sql_select_models_loan() -> pd.DataFrame:
    query = text("""
        SELECT product_code, tenor, prep_rate
        FROM bs.models_loan
        WHERE report_date = :report_date
    """)
    return pd.read_sql_query(query, engine, params={"report_date": config.report_date})


def sql_select_models_deposit_ir() -> pd.DataFrame:
    query = text("""
        SELECT product_code, tenor, outstanding
        FROM bs.models_deposit_ir
        WHERE report_date = :report_date
        ORDER BY product_code, tenor
    """)
    return pd.read_sql_query(query, engine, params={"report_date": config.report_date})


def sql_select_models_deposit_liq() -> pd.DataFrame:
    query = text("""
        SELECT product_code, tenor, outstanding
        FROM bs.models_deposit_liq
        WHERE report_date = :report_date
        ORDER BY product_code, tenor
    """)
    return pd.read_sql_query(query, engine, params={"report_date": config.report_date})


_TENOR_SORT = {"1D": 0, ">30Y": 362}


def _month_buckets(
    cf_end_series: pd.Series, report_date: pd.Timestamp
) -> pd.Series:
    """Assign nM bucket labels using exact calendar-month boundaries.

    Date d falls in bucket n where n is the smallest integer >= 1 such that
    d <= report_date + DateOffset(months=n).  Uses DateOffset for comparison
    so month-end report dates (e.g. Dec 31) are handled correctly: Feb 28 → 2M,
    Mar 13 → 3M, rather than the wrong (day < report_day) approximation.
    """
    raw = (
        (cf_end_series.dt.year  - report_date.year)  * 12
        + (cf_end_series.dt.month - report_date.month)
    )
    bucket_ends = pd.DatetimeIndex([
        report_date + pd.DateOffset(months=int(m)) for m in raw.clip(lower=0)
    ])
    over = cf_end_series.values > bucket_ends
    months = (raw + over.astype(int)).clip(lower=1)
    return months.map(lambda m: ">30Y" if m > 360 else f"{m}M")


def _tenor_sort_key(label: str) -> int:
    if label in _TENOR_SORT:
        return _TENOR_SORT[label]
    return int(label[:-1])  # strip trailing 'M'


def _bucket_bounds(
    tenor_bucket: str, report_date: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp | float]:
    """Return (bucket_start_dt, bucket_end_dt) for the given tenor bucket.

    Boundaries are exact calendar-month dates derived from report_date:
      1D  : [report_date,   report_date + 1 day]
      1M  : [report_date+2d, report_date + 1 month]
      nM  : [prev_end + 1d,  report_date + n months]   (n >= 2)
      >30Y: [360M_end + 1d,  NaT]
    """
    if tenor_bucket == "1D":
        return report_date, report_date + pd.Timedelta(days=1)
    if tenor_bucket == ">30Y":
        start = report_date + pd.DateOffset(months=360) + pd.Timedelta(days=1)
        return start, pd.NaT
    n = int(tenor_bucket[:-1])
    end   = report_date + pd.DateOffset(months=n)
    start = (report_date + pd.Timedelta(days=2)) if n == 1 else \
            (report_date + pd.DateOffset(months=n - 1) + pd.Timedelta(days=1))
    return start, end


def _aggregate_gap(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.DataFrame:
    """Apply sign, bucket by tenor and aggregate total payments per currency as gap_cf."""
    df = df.copy()
    sign = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["gap_cf"] = (df["capital_pmt"] + df["int_pmt"]) * sign

    # Overnight bucket = minimum cf_end_dt per currency (next business day in that currency)
    overnight_dates = df.groupby("currency")["cf_end_dt"].transform("min")
    df["tenor_bucket"] = np.where(
        df["cf_end_dt"] == overnight_dates,
        "1D",
        _month_buckets(df["cf_end_dt"], report_date),
    )

    # Aggregate payments — one row per (currency, bs_side, tenor_bucket)
    result = (
        df.groupby(["currency", "bs_side", "tenor_bucket"], sort=False)[["gap_cf"]]
        .sum()
        .reset_index()
    )

    # Compute representative cf_end_dt: overnight date from data, EOM for monthly buckets
    overnight_by_ccy = (
        df[df["tenor_bucket"] == "1D"]
        .groupby("currency")["cf_end_dt"]
        .first()
    )

    def _bucket_date(row: pd.Series) -> pd.Timestamp:
        if row["tenor_bucket"] == "1D":
            return overnight_by_ccy.get(row["currency"], pd.NaT)
        if row["tenor_bucket"] == ">30Y":
            return report_date + pd.offsets.MonthEnd(361)
        return report_date + pd.offsets.MonthEnd(int(row["tenor_bucket"][:-1]))

    result["cf_end_dt"] = result.apply(_bucket_date, axis=1)

    bounds = result["tenor_bucket"].map(lambda t: _bucket_bounds(t, report_date))
    result["bucket_start_dt"] = [b[0] for b in bounds]
    result["bucket_end_dt"]   = [b[1] for b in bounds]

    result["_sort"] = result["tenor_bucket"].map(_tenor_sort_key)
    result = (
        result.sort_values(["currency", "bs_side", "_sort"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    return result[["currency", "bs_side", "tenor_bucket",
                   "bucket_start_dt", "bucket_end_dt", "cf_end_dt", "gap_cf"]]


def compute_liq_gap(report_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute contractual and behavioural liquidity gap tables.

    Cash flows are placed at their payment date (cf_end_dt).
    Tenor buckets: '1D' (overnight), '1M'..'360M', '>30Y'.
    Columns: currency, bs_side, tenor_bucket, cf_end_dt, gap_cf.

    Behavioural LIQ gap uses separate models per product type:
      - Deposits : LIQ behavioral model from cf.products_liq
      - Loans / fin_inst: IR behavioral model from cf.products (no separate LIQ model)
    """
    params = {"rd": report_date}

    orig_query = text("""
        SELECT currency, bs_side, cf_end_dt,
               con_capital_pmt  AS capital_pmt,
               con_interest_pmt AS int_pmt
        FROM cf.products
        WHERE cf_end_dt > :rd
          AND COALESCE(con_total_pmt, 0) <> 0
    """)

    # Deposits use LIQ model (cf.products_liq); loans + fin_inst use IR beh from cf.products
    beh_query = text("""
        SELECT currency, bs_side, cf_end_dt,
               beh_capital_pmt AS capital_pmt,
               beh_interest_pmt AS int_pmt
        FROM cf.products_liq
        WHERE cf_end_dt > :rd
          AND COALESCE(beh_total_pmt, 0) <> 0

        UNION ALL

        SELECT currency, bs_side, cf_end_dt,
               beh_capital_pmt + COALESCE(prepayment_pmt, 0) AS capital_pmt,
               beh_interest_pmt AS int_pmt
        FROM cf.products
        WHERE cf_end_dt > :rd
          AND product_type IN ('L', 'F')
          AND COALESCE(beh_total_pmt, 0) <> 0
    """)

    orig_df = pd.read_sql_query(orig_query, engine, params=params)
    beh_df  = pd.read_sql_query(beh_query,  engine, params=params)

    orig_df["cf_end_dt"] = pd.to_datetime(orig_df["cf_end_dt"])
    beh_df["cf_end_dt"]  = pd.to_datetime(beh_df["cf_end_dt"])

    return _aggregate_gap(orig_df, report_date), _aggregate_gap(beh_df, report_date)


def _aggregate_ir_gap(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.DataFrame:
    """Sign, bucket by tenor and aggregate gap_cf per currency."""
    df = df.copy()
    sign = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["gap_cf"] = df["gap_cf"] * sign

    overnight_dates = df.groupby("currency")["cf_end_dt"].transform("min")
    df["tenor_bucket"] = np.where(
        df["cf_end_dt"] == overnight_dates,
        "1D",
        _month_buckets(df["cf_end_dt"], report_date),
    )

    result = (
        df.groupby(["currency", "bs_side", "tenor_bucket"], sort=False)[["gap_cf"]]
        .sum()
        .reset_index()
    )

    overnight_by_ccy = (
        df[df["tenor_bucket"] == "1D"]
        .groupby("currency")["cf_end_dt"]
        .first()
    )

    def _bucket_date(row: pd.Series) -> pd.Timestamp:
        if row["tenor_bucket"] == "1D":
            return overnight_by_ccy.get(row["currency"], pd.NaT)
        if row["tenor_bucket"] == ">30Y":
            return report_date + pd.offsets.MonthEnd(361)
        return report_date + pd.offsets.MonthEnd(int(row["tenor_bucket"][:-1]))

    result["cf_end_dt"] = result.apply(_bucket_date, axis=1)

    bounds = result["tenor_bucket"].map(lambda t: _bucket_bounds(t, report_date))
    result["bucket_start_dt"] = [b[0] for b in bounds]
    result["bucket_end_dt"]   = [b[1] for b in bounds]

    result["_sort"] = result["tenor_bucket"].map(_tenor_sort_key)
    result = (
        result.sort_values(["currency", "bs_side", "_sort"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    return result[["currency", "bs_side", "tenor_bucket",
                   "bucket_start_dt", "bucket_end_dt", "cf_end_dt", "gap_cf"]]


def _aggregate_ir_gap_a(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.DataFrame:
    """Same as _aggregate_ir_gap but keeps product_code as an additional dimension."""
    df = df.copy()
    sign = np.where(df["bs_side"] == "A", 1.0, -1.0)
    df["gap_cf"] = df["gap_cf"] * sign

    overnight_dates = df.groupby("currency")["cf_end_dt"].transform("min")
    df["tenor_bucket"] = np.where(
        df["cf_end_dt"] == overnight_dates,
        "1D",
        _month_buckets(df["cf_end_dt"], report_date),
    )

    result = (
        df.groupby(["currency", "bs_side", "product_code", "tenor_bucket"], sort=False)[["gap_cf"]]
        .sum()
        .reset_index()
    )

    overnight_by_ccy = (
        df[df["tenor_bucket"] == "1D"]
        .groupby("currency")["cf_end_dt"]
        .first()
    )

    def _bucket_date(row: pd.Series) -> pd.Timestamp:
        if row["tenor_bucket"] == "1D":
            return overnight_by_ccy.get(row["currency"], pd.NaT)
        if row["tenor_bucket"] == ">30Y":
            return report_date + pd.offsets.MonthEnd(361)
        return report_date + pd.offsets.MonthEnd(int(row["tenor_bucket"][:-1]))

    result["cf_end_dt"] = result.apply(_bucket_date, axis=1)

    bounds = result["tenor_bucket"].map(lambda t: _bucket_bounds(t, report_date))
    result["bucket_start_dt"] = [b[0] for b in bounds]
    result["bucket_end_dt"]   = [b[1] for b in bounds]

    result["_sort"] = result["tenor_bucket"].map(_tenor_sort_key)
    result = (
        result.sort_values(["currency", "bs_side", "product_code", "_sort"])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    return result[["currency", "bs_side", "product_code", "tenor_bucket",
                   "bucket_start_dt", "bucket_end_dt", "cf_end_dt", "gap_cf"]]


def compute_ir_gap(report_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute contractual and behavioural interest rate gap (repricing gap) tables.

    Loans / fin_inst (product_type IN ('L','F')) — repricing template:
        Fixed rate  : one repricing event at maturity (capital + int).
        Floating rate: capital at each cf_end_dt + first int only (locked-in coupon)
                       + outstanding at next fixing date.

    Deposits (product_type = 'D'): capital + int per row (each period is its own repricing).

    Columns: currency, bs_side, tenor_bucket, cf_end_dt, gap_cf.
    """
    params = {"rd": report_date}

    def _repricing_q(capital_expr: str, int_expr: str, outstanding_expr: str,
                     include_pc: bool = False) -> text:
        pc_col    = ", s.product_code" if include_pc else ""
        pc_select = "product_code," if include_pc else ""
        pc_fo     = ", product_code" if include_pc else ""
        return text(f"""
            WITH floating_scheds AS (
                SELECT schedule_id, MIN(fixing_dt) AS next_fixing_dt
                FROM cf.products
                WHERE cf_end_dt > :rd AND fixing_dt > :rd
                  AND product_type IN ('L', 'F')
                GROUP BY schedule_id
            ),
            base AS (
                SELECT s.schedule_id, s.currency, s.bs_side {pc_col},
                       s.cf_end_dt,
                       {capital_expr}     AS capital_pmt,
                       {int_expr}         AS int_pmt,
                       {outstanding_expr} AS outstanding_bal,
                       s.fixing_dt,
                       ROW_NUMBER() OVER (PARTITION BY s.schedule_id
                                         ORDER BY s.cf_end_dt) AS rn,
                       CASE WHEN f.schedule_id IS NOT NULL THEN 1 ELSE 0 END AS is_floating,
                       f.next_fixing_dt
                FROM cf.products s
                LEFT JOIN floating_scheds f ON s.schedule_id = f.schedule_id
                WHERE s.cf_end_dt > :rd
                  AND s.product_type IN ('L', 'F')
            ),
            float_outstanding AS (
                SELECT schedule_id, next_fixing_dt, currency, bs_side {pc_fo},
                       MAX(outstanding_bal) AS outstanding_at_fix
                FROM base
                WHERE is_floating = 1 AND fixing_dt = next_fixing_dt
                GROUP BY schedule_id, next_fixing_dt, currency, bs_side {pc_fo}
            )
            SELECT currency, bs_side, {pc_select} cf_end_dt,
                   capital_pmt + int_pmt AS gap_cf
            FROM base WHERE is_floating = 0

            UNION ALL

            SELECT currency, bs_side, {pc_select} cf_end_dt,
                   capital_pmt AS gap_cf
            FROM base WHERE is_floating = 1 AND fixing_dt <= :rd

            UNION ALL

            SELECT currency, bs_side, {pc_select} cf_end_dt,
                   int_pmt AS gap_cf
            FROM base WHERE is_floating = 1 AND rn = 1

            UNION ALL

            SELECT currency, bs_side, {pc_select} next_fixing_dt AS cf_end_dt,
                   outstanding_at_fix AS gap_cf
            FROM float_outstanding
        """)

    def _deposit_q(capital_expr: str, int_expr: str, side_filter: str,
                   include_pc: bool = False) -> text:
        pc_select = "product_code," if include_pc else ""
        return text(f"""
            SELECT currency, bs_side, {pc_select} cf_end_dt,
                   {capital_expr} + {int_expr} AS gap_cf
            FROM cf.products
            WHERE cf_end_dt > :rd
              AND product_type = 'D'
              AND {side_filter}
        """)

    def _fetch(q) -> pd.DataFrame:
        df = pd.read_sql_query(q, engine, params=params)
        df["cf_end_dt"] = pd.to_datetime(df["cf_end_dt"])
        return df

    _orig_filter = "COALESCE(con_total_pmt, 0) <> 0"
    _beh_filter  = "COALESCE(beh_total_pmt, 0) <> 0"

    orig_df = pd.concat([
        _fetch(_repricing_q("s.con_capital_pmt", "s.con_interest_pmt", "s.con_outstanding")),
        _fetch(_deposit_q("con_capital_pmt", "con_interest_pmt", _orig_filter)),
    ], ignore_index=True)

    beh_df = pd.concat([
        _fetch(_repricing_q(
            "s.beh_capital_pmt + COALESCE(s.prepayment_pmt, 0)",
            "s.beh_interest_pmt", "s.beh_outstanding")),
        _fetch(_deposit_q("beh_capital_pmt", "beh_interest_pmt", _beh_filter)),
    ], ignore_index=True)

    beh_a_df = pd.concat([
        _fetch(_repricing_q(
            "s.beh_capital_pmt + COALESCE(s.prepayment_pmt, 0)",
            "s.beh_interest_pmt", "s.beh_outstanding", include_pc=True)),
        _fetch(_deposit_q("beh_capital_pmt", "beh_interest_pmt", _beh_filter, include_pc=True)),
    ], ignore_index=True)

    ir_gap_orig  = _aggregate_ir_gap(orig_df, report_date)
    ir_gap_beh   = _aggregate_ir_gap(beh_df, report_date)
    ir_gap_beh_a = _aggregate_ir_gap_a(beh_a_df, report_date)

    # Add gap_cf_ca (CA portion, product_code 6000) to ir_gap_beh for direct NII use
    ca_agg = (
        ir_gap_beh_a[ir_gap_beh_a["product_code"].astype(str) == "6000"]
        .groupby(["currency", "bs_side", "tenor_bucket", "cf_end_dt"], sort=False)["gap_cf"]
        .sum()
        .reset_index()
        .rename(columns={"gap_cf": "gap_cf_ca"})
    )
    ir_gap_beh = ir_gap_beh.merge(
        ca_agg, on=["currency", "bs_side", "tenor_bucket", "cf_end_dt"], how="left"
    )
    ir_gap_beh["gap_cf_ca"]  = ir_gap_beh["gap_cf_ca"].fillna(0.0)
    ir_gap_beh["gap_cf_irs"] = 0.0  # filled in by irs_workflow

    return ir_gap_orig, ir_gap_beh, ir_gap_beh_a
