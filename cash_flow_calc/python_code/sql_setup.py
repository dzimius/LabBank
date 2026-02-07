import pandas as pd
from sqlalchemy import create_engine
import re


engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)


def sql_get_params(engine, table_name, columns, filters, chunksize=50_000, schema="dbo"):
    where_clauses = []
    params = []

    for col, val in filters.items():
        where_clauses.append(f"{col} = ?")
        params.append(val)

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    query = f"""
    SELECT {", ".join(columns)}
    FROM {schema}.{table_name}
    WHERE {where_sql}
    """

    return pd.read_sql_query(query, engine, params=tuple(params), chunksize=chunksize)

def load_sched_params(table, cols, date):
    chunks_loans = sql_get_params(
        engine=engine,
        table_name=table,
        columns=cols,
        filters={
            #"rate_type": r_type,
            "report_date": date
        }
    )
    return chunks_loans

