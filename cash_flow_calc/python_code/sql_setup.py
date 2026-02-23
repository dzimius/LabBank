import pandas as pd
import config
from sqlalchemy import create_engine,event, MetaData, Table, Column
from sqlalchemy import Integer, String, ForeignKey, Date, text, DECIMAL


engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)
metadata = MetaData(schema="dbo")

Loan_sched_d = Table(
    "loan_sched_dates", metadata,
    Column("schedule_id", String(8), primary_key=True, nullable=False),
    Column("fixing_dt", Date, nullable=False),
    Column("cf_start_dt", Date, primary_key=True, nullable=False),
    Column("cf_end_dt", Date, nullable=False),
    Column("cf_yf", DECIMAL(18, 2), nullable=False),
    schema="dbo",
)

Fin_inst_sched_d = Table(
    "fin_inst_sched_dates", metadata,
    Column("schedule_id", String(8), primary_key=True, nullable=False),
    Column("fixing_dt", Date, nullable=False),
    Column("cf_start_dt", Date, primary_key=True, nullable=False),
    Column("cf_end_dt", Date, nullable=False),
    Column("cf_yf", DECIMAL(18, 2), nullable=False),
    schema="dbo",
)
metadata.create_all(engine)


def reset_data(mode: int, report_date=None) -> None:
    """
    mode=0 -> drop + recreate tabele
    mode=1 -> DELETE specific report_date data
    """
    tables = [
        "loan_sched_dates",
        "fin_inst_sched_dates"
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



def sql_get_params(engine, table_name, columns, chunksize=50_000, schema="dbo"):
    where_clauses = []
    params = []

    query = f"""
    SELECT {", ".join(columns)}
    FROM {schema}.{table_name}
    """

    return pd.read_sql_query(query, engine, params=tuple(params), chunksize=chunksize)

def load_sched_params(table, cols):
    chunks_loans = sql_get_params(
        engine=engine,
        table_name=table,
        columns=cols,
    )
    return chunks_loans

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


def iter_schedule_batches(engine, table_name, batch_size=1000, schema="dbo"):
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

def sql_get_uniq_curves(tables, curve_type):
    union_parts = []

    for table in tables:
        union_parts.append(f"""
            SELECT {curve_type} as curve_name, case when '{curve_type}'='fwd_curve' then fixing_freq
            else NULL end as fixing_freq, '{curve_type}' as curve_type
            FROM {table}
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

def sql_select_specific_curve(curve_name):
    query = text("""
        SELECT curve_name, n_days, year_frac, zero_rate, d_f
        FROM curves
        WHERE curve_date = :report_date
          AND curve_name = :curve_name
    """)
    return pd.read_sql_query(
        query,
        engine,
        params={"report_date": config.report_date, "curve_name": curve_name}
    )

def sql_select_fixings():
    query = """
    SELECT *  FROM fixings
    """
    return pd.read_sql_query(query, engine)


