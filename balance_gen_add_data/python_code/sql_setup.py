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


metadata = MetaData(schema=None)   # np. schema="dbo" jeśli używasz

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
)

Fixing = Table(
    "fixings", metadata,
    Column("fixing_date", Date, nullable=False),
    Column("rate_index", String(8), nullable=False),
    Column("tenor", String(4), nullable=False),
    Column("rate", DECIMAL(18, 6), nullable=False),
    Column("currency", String(3), nullable=False)
)

Loan_mod = Table(
    "models_loan", metadata,
    Column("report_date", Date, nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("tenor", String(4), nullable=False),
    Column("prep_rate", DECIMAL(18, 2), nullable=False)
)

Depo_mod = Table(
    "models_deposit", metadata,
    Column("report_date", Date, nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("tenor", String(4), nullable=False),
    Column("outstanding", DECIMAL(18, 2), nullable=False)
)



TABLES = {
    "curves": Curves,
    "fixings": Fixing,
    "models_loan": Loan_mod,
    "models_deposit": Depo_mod
}

def append_df_to_table(df: pd.DataFrame, table_name: str) -> None:
    """Dopasuj df do schematu tabeli i wykonaj append. Brakujące kolumny -> NULL."""
    tbl = TABLES[table_name]
    # kolejność i lista kolumn wg tabeli:
    cols = [c.name for c in tbl.columns]
    # dołóż brakujące kolumny (NaN -> później NULL)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    # utnij nadmiarowe kolumny i ustaw kolejność
    df2 = df[cols].copy()
    # NaN/NaT -> None, żeby poszło jako NULL
    df2 = df2.where(pd.notnull(df2), None)

    # append
    df2.to_sql(
        table_name,
        engine,
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
                conn.execute(
                    text(
                        f"IF OBJECT_ID('dbo.{t}', 'U') IS NOT NULL "
                        f"DROP TABLE dbo.{t};"
                    )
                )
            metadata.create_all(bind=conn, checkfirst=True)

        elif mode == 1:
            if report_date is None:
                raise ValueError("Report_date is needed")

            for t in tables:
                conn.execute(
                    text(f"DELETE FROM dbo.{t} WHERE report_date = :rd"),
                    {"rd": report_date},
                )
        else:
            raise ValueError("Mode should be 0 or 1")


def reset_data_remove_always(tables: list[str]) -> None:
    with engine.begin() as conn:
        for t in tables:
            conn.execute(
                text(
                    f"DROP TABLE IF EXISTS dbo.{t};"
                )
            )

def create_sched_id_tbl_sql(
    engine: Engine,
    source_table: str,
    target_table: str,
    columns: list[str],
    sum_cols: Optional[list[str]] = None,
    schema: str = "dbo",
) -> None:
    sum_cols = sum_cols or []
    cols_sql = ", ".join(columns)
    order_by_sql = ", ".join(columns)

    # SUM(col) AS col
    sum_sql = ",\n        ".join([f"SUM({c}) AS {c}" for c in sum_cols])
    sum_select = ", " + ", ".join(sum_cols) if sum_cols else ""

    sql = f"""
        DROP TABLE IF EXISTS {schema}.{target_table};
    
        WITH grp AS (
            SELECT 
                {cols_sql}
                {"," if sum_sql else ""}
                {sum_sql}
            FROM {schema}.{source_table}
            GROUP BY {cols_sql}
        ),
        grp_id AS (
            SELECT
                DENSE_RANK() OVER (ORDER BY {order_by_sql}) AS schedule_id,
                {cols_sql}
                {sum_select}
            FROM grp
        )
        SELECT
            schedule_id,
            {cols_sql}
            {sum_select}
        INTO {schema}.{target_table}
        FROM grp_id;
        """

    with engine.begin() as conn:
        conn.execute(text(sql))

def update_schedule_id_sql(
    engine: Engine,
    table_name: str,
    sched_table_name: str,
    columns: list[str],
    schema: str = "dbo",
) -> None:
    on_sql = " AND ".join([
        f"(a.{c} = b.{c} OR (a.{c} IS NULL AND b.{c} IS NULL))"
        for c in columns
    ])
    sql = f"""
    UPDATE a
       SET a.schedule_id = b.schedule_id
    FROM {schema}.{table_name} a
    LEFT JOIN {schema}.{sched_table_name} b
      ON {on_sql};
    """

    with engine.begin() as conn:
        conn.execute(text(sql))

