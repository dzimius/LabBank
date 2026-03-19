from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text

import config

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)


def load_ir_gap_beh() -> pd.DataFrame:
    query = text("""
        SELECT currency, bs_side, tenor_bucket, cf_end_dt, gap_cf
        FROM irrbb.ir_gap_beh
    """)
    df = pd.read_sql_query(query, engine)
    df["cf_end_dt"] = pd.to_datetime(df["cf_end_dt"])
    return df


def load_ir_gap_ca() -> pd.DataFrame:
    """Load current account (product_code=6000) repricing gap from ir_gap_beh_a.

    Reads directly from the analytical gap table (already aggregated by tenor bucket,
    gap_cf = capital_pmt + int_pmt with sign applied) so values match ir_gap_beh exactly.
    Returns (currency, bs_side, cf_end_dt, ca_gap_cf).
    Used to apply floor = 0% on the down shock in NII sensitivity.
    """
    query = text("""
        SELECT currency, bs_side, cf_end_dt,
               SUM(gap_cf) AS ca_gap_cf
        FROM irrbb.ir_gap_beh_a
        WHERE product_name = '6000'
        GROUP BY currency, bs_side, cf_end_dt
    """)
    df = pd.read_sql_query(query, engine)
    df["cf_end_dt"] = pd.to_datetime(df["cf_end_dt"])
    return df


def load_beh_schedules(report_date: pd.Timestamp, horizon_end: pd.Timestamp) -> pd.DataFrame:
    """Load behavioral CF schedules for all products within (report_date, horizon_end].

    Joins with sched.* tables to get currency.
    Returns unified DataFrame with source column ('loans', 'deposits', 'fin_inst').
    Columns: source, schedule_id, bs_side, currency, cf_start_dt, cf_end_dt,
             cf_yf, fwd_rt, outstanding_bal, capital_pmt, prepayment_pmt, int_pmt.
    """
    query = text("""
        SELECT 'loans' AS source,
               l.schedule_id, l.bs_side, s.currency,
               l.cf_start_dt, l.cf_end_dt, l.cf_yf, l.fwd_rt,
               l.outstanding_bal, l.capital_pmt,
               ISNULL(l.prepayment_pmt, 0.0) AS prepayment_pmt,
               l.int_pmt
        FROM cf.loan_beh l
        JOIN sched.loans s ON l.schedule_id = s.schedule_id
        WHERE l.cf_end_dt > :rd AND l.cf_end_dt <= :he

        UNION ALL

        SELECT 'deposits' AS source,
               d.schedule_id, d.bs_side, s.currency,
               d.cf_start_dt, d.cf_end_dt, d.cf_yf, d.fwd_rt,
               d.outstanding_bal, d.capital_pmt,
               0.0 AS prepayment_pmt,
               d.int_pmt
        FROM cf.deposit_beh d
        JOIN sched.deposits s ON d.schedule_id = s.schedule_id
        WHERE d.cf_end_dt > :rd AND d.cf_end_dt <= :he

        UNION ALL

        SELECT 'fin_inst' AS source,
               f.schedule_id, f.bs_side, s.currency,
               f.cf_start_dt, f.cf_end_dt, f.cf_yf, f.fwd_rt,
               f.outstanding_bal, f.capital_pmt,
               0.0 AS prepayment_pmt,
               f.int_pmt
        FROM cf.fin_inst_beh f
        JOIN sched.fin_inst s ON f.schedule_id = s.schedule_id
        WHERE f.cf_end_dt > :rd AND f.cf_end_dt <= :he
    """)
    df = pd.read_sql_query(query, engine, params={"rd": report_date, "he": horizon_end})
    df["cf_start_dt"] = pd.to_datetime(df["cf_start_dt"])
    df["cf_end_dt"]   = pd.to_datetime(df["cf_end_dt"])
    return df
