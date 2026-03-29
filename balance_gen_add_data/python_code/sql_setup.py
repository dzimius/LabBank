from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy import Integer, String, ForeignKey, Date, text
from sqlalchemy.dialects.mssql import DECIMAL
from sqlalchemy.engine import Engine

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)


metadata = MetaData(schema=None)

# Schema mapping: which schema each table lives in
TABLE_SCHEMAS: dict[str, str] = {
    "curves":        "mkt",
    "fixings":       "mkt",
    "models_loan":   "bs",
    "models_deposit":"bs",
}

def _ensure_schemas() -> None:
    """Create mkt, bs, and sched schemas if they don't exist."""
    for s in ("mkt", "bs", "sched"):
        with engine.begin() as conn:
            conn.execute(text(
                f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{s}') "
                f"EXEC('CREATE SCHEMA [{s}]');"
            ))

Curves = Table(
    "curves", metadata,
    Column("curve_date", Date, nullable=False),
    Column("tenor", String(4), nullable=False),
    Column("year_frac", DECIMAL(18, 2), nullable=False),
    Column("n_days", Integer, nullable=False),
    Column("mat_date", Date, nullable=False),
    Column("zero_rate", DECIMAL(18, 6), nullable=False),
    Column("d_f", DECIMAL(18, 6), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("curve_name", String(20), nullable=False),
    schema="mkt",
)

Fixing = Table(
    "fixings", metadata,
    Column("fixing_date", Date, nullable=False),
    Column("rate_index", String(8), nullable=False),
    Column("tenor", String(4), nullable=False),
    Column("rate", DECIMAL(18, 6), nullable=False),
    Column("currency", String(3), nullable=False),
    schema="mkt",
)

Loan_mod = Table(
    "models_loan", metadata,
    Column("report_date", Date, nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("tenor", String(4), nullable=False),
    Column("prep_rate", DECIMAL(18, 2), nullable=False),
    schema="bs",
)

Depo_mod = Table(
    "models_deposit", metadata,
    Column("report_date", Date, nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("tenor", String(4), nullable=False),
    Column("outstanding", DECIMAL(18, 2), nullable=False),
    schema="bs",
)



TABLES = {
    "curves": Curves,
    "fixings": Fixing,
    "models_loan": Loan_mod,
    "models_deposit": Depo_mod
}

def append_df_to_table(df: pd.DataFrame, table_name: str) -> None:
    """Dopasuj df do schematu tabeli i wykonaj append. Brakujące kolumny -> NULL."""
    schema = TABLE_SCHEMAS[table_name]
    tbl = TABLES[table_name]
    cols = [c.name for c in tbl.columns]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df2 = df[cols].copy()
    df2 = df2.where(pd.notnull(df2), None)

    df2.to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists="append",
        index=False,
        chunksize=10_000,
        method=None,
    )

def reset_data_models(mode: int, report_date: str, tables: list) -> None:
    """
    mode=0 -> drop + recreate tabele
    mode=1 -> DELETE specific report_date data
    """
    with engine.begin() as conn:
        if mode == 0:
            for t in tables:
                schema = TABLE_SCHEMAS.get(t, "bs")
                conn.execute(text(
                    f"IF OBJECT_ID('{schema}.{t}', 'U') IS NOT NULL "
                    f"DROP TABLE {schema}.{t};"
                ))
            metadata.create_all(bind=conn, checkfirst=True)

        elif mode == 1:
            if report_date is None:
                raise ValueError("Report_date is needed")

            for t in tables:
                schema = TABLE_SCHEMAS.get(t, "bs")
                conn.execute(
                    text(f"DELETE FROM {schema}.{t} WHERE report_date = :rd"),
                    {"rd": report_date},
                )
        else:
            raise ValueError("Mode should be 0 or 1")


def reset_data_remove_always(schema_tables: list[str]) -> None:
    """Accept 'schema.table' strings (e.g. 'mkt.curves')."""
    with engine.begin() as conn:
        for st in schema_tables:
            conn.execute(text(f"DROP TABLE IF EXISTS {st};"))

def create_sched_id_tbl_sql(
    engine: Engine,
    source_table: str,
    target_table: str,
    columns: list[str],
    sum_cols: Optional[list[str]] = None,
    avg_cols: Optional[list[str]] = None,
    null_cols: Optional[list[str]] = None,
    source_schema: str = "dbo",
    target_schema: str = "bs",
) -> None:
    """Create a schedule-id table by grouping source_table rows.

    null_cols: columns not present in the source table; added as NULL in output
               so all sched tables share a uniform column set.
    """
    sum_cols  = sum_cols  or []
    avg_cols  = avg_cols  or []
    null_cols = null_cols or []

    cols_sql     = ", ".join(columns)
    order_by_sql = ", ".join(columns)

    agg_parts = (
        [f"SUM({c}) AS {c}" for c in sum_cols]
        + [f"AVG({c}) AS {c}" for c in avg_cols]
    )
    agg_sql      = ",\n        ".join(agg_parts)
    extra_cols   = sum_cols + avg_cols
    extra_select = ", " + ", ".join(extra_cols) if extra_cols else ""

    null_grp_sql    = (",\n        " + ",\n        ".join(f"NULL AS {c}" for c in null_cols)) if null_cols else ""
    null_out_select = (", " + ", ".join(null_cols)) if null_cols else ""

    sql = f"""
        DROP TABLE IF EXISTS {target_schema}.{target_table};

        WITH grp AS (
            SELECT
                {cols_sql}
                {"," if agg_sql else ""}
                {agg_sql}
                {null_grp_sql}
            FROM {source_schema}.{source_table}
            GROUP BY {cols_sql}
        ),
        grp_id AS (
            SELECT
                DENSE_RANK() OVER (ORDER BY {order_by_sql}) AS schedule_id,
                {cols_sql}
                {extra_select}
                {null_out_select}
            FROM grp
        )
        SELECT
            schedule_id,
            {cols_sql}
            {extra_select}
            {null_out_select}
        INTO {target_schema}.{target_table}
        FROM grp_id;
        """

    with engine.begin() as conn:
        conn.execute(text(sql))

def update_schedule_id_sql(
    engine: Engine,
    table_name: str,
    sched_table_name: str,
    columns: list[str],
    source_schema: str = "dbo",
    sched_schema: str = "bs",
) -> None:
    on_sql = " AND ".join([
        f"(a.{c} = b.{c} OR (a.{c} IS NULL AND b.{c} IS NULL))"
        for c in columns
    ])
    sql = f"""
    UPDATE a
       SET a.schedule_id = b.schedule_id
    FROM {source_schema}.{table_name} a
    LEFT JOIN {sched_schema}.{sched_table_name} b
      ON {on_sql};
    """

    with engine.begin() as conn:
        conn.execute(text(sql))

