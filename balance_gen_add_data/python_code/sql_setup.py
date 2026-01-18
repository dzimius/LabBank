import numpy as np
from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy import Integer, String, ForeignKey, Date, text
from sqlalchemy.dialects.mssql import DECIMAL
import pandas as pd
import re

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
    Column("mat_date", Date, nullable=False),
    Column("int_rate", DECIMAL(18, 6), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("curve_type", String(20), nullable=False),


)

Loan = Table(
    "models_loan", metadata,
    Column("report_date", Date, nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("tenor", String(4), nullable=False),
    Column("prep_rate", DECIMAL(18, 2), nullable=False)
)

Depo = Table(
    "models_deposit", metadata,
    Column("report_date", Date, nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("tenor", String(4), nullable=False),
    Column("outstanding", DECIMAL(18, 2), nullable=False)
)



TABLES = {
    "curves": Curves,
    "models_loan":Loan,
    "models_deposit": Depo
}

def append_df_to_table(df: pd.DataFrame, table_name: str):
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

def reset_data(mode: int, report_date: str) -> None:
    """
    mode=0 -> drop + recreate tabele
    mode=1 -> DELETE specific report_date data
    """
    tables = [
        "curves",
        "models_loan",
        "models_deposit"
    ]

    with engine.begin() as conn:
        if mode == 0:
            for t in tables:
                conn.execute(
                    text(
                        f"IF OBJECT_ID('dbo.{t}', 'U') IS NOT NULL "
                        f"DROP TABLE dbo.{t};"
                    )
                )
            metadata.create_all(bind=conn, checkfirst=False)

        elif mode == 1:
            if report_date is None:
                raise ValueError("For mode=1 report_date is needed")

            for t in tables:
                conn.execute(
                    text(f"DELETE FROM dbo.{t} WHERE report_date = :rd"),
                    {"rd": report_date},
                )
        else:
            raise ValueError("Mode should be 0 or 1")