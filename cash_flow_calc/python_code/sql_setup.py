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

def sql_get_params(engine, table_name, columns, report_date, chunksize=50_000, schema="dbo"):
    query = f"""
    SELECT {", ".join(columns)}
    FROM {schema}.{table_name}
    WHERE report_date = ?
    """
    return pd.read_sql_query(query, engine, params=(report_date,), chunksize=chunksize)