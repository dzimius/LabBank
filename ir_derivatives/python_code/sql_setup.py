from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, text
from sqlalchemy.dialects.mssql import DECIMAL

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

metadata = MetaData(schema=None)

# ── schemat.ir_swaps — leg-level: two rows per swap (FIXED + FLOAT) ───────────
# swap_id groups the two legs of one transaction.
# leg_id  is the primary key (e.g. "IRS-2024-001-F" / "IRS-2024-001-V").
IRSwaps = Table(
    "ir_swaps", metadata,
    Column("report_date",   Date,           nullable=False),
    Column("swap_id",       String(13),     nullable=False),   # parent swap reference
    Column("leg_id",        String(16),     primary_key=True, nullable=False),
    Column("product_code",  String(4),      nullable=False),   # always '0000' for IRS
    Column("leg_type",      String(5),      nullable=False),   # FIXED / FLOAT
    Column("bs_side",       String(1),      nullable=False),   # A / L
    Column("swap_type",     String(4),      nullable=False),   # IRS / OIS / CIRS
    Column("notional",      DECIMAL(18, 2), nullable=False),
    Column("currency",      String(3),      nullable=False),
    Column("start_date",    Date,           nullable=False),
    Column("maturity_date", Date,           nullable=False),
    Column("pay_freq",      String(4),      nullable=False),
    Column("fixing_freq",   String(4),      nullable=True),    # NULL for FIXED leg
    Column("rate_index",    String(11),     nullable=False),   # 'FIXED' or e.g. 'WIBOR3M'
    Column("dc_conv",       String(7),      nullable=False),
    Column("b_day_conv",    String(25),     nullable=False),
    Column("fixed_rate",    DECIMAL(9, 6),  nullable=True),    # NULL for FLOAT leg
    Column("float_spread",  DECIMAL(9, 6),  nullable=True),    # NULL for FIXED leg
    Column("disc_curve",    String(20),     nullable=False),
    Column("fwd_curve",     String(20),     nullable=False),
    Column("schedule_id",   Integer,        nullable=True),    # set after sched creation
    schema="schemat",
)

# ── cf.ir_swap_orig / cf.ir_swap_beh — cash-flow schedule ────────────────────
# orig == beh for swaps (no behavioural adjustments).
IrSwapOrig = Table(
    "ir_swap_orig", metadata,
    Column("schedule_id",    String(8),      primary_key=True, nullable=False),
    Column("bs_side",        String(1),      nullable=False),
    Column("rate_index",     String(10),     nullable=False),
    Column("fixing_dt",      Date,           nullable=False),
    Column("cf_start_dt",    Date,           primary_key=True, nullable=False),
    Column("cf_end_dt",      Date,           nullable=False),
    Column("cf_yf",          DECIMAL(18, 6), nullable=False),
    Column("d_f",            DECIMAL(18, 6), nullable=True),
    Column("fwd_rt",         DECIMAL(18, 6), nullable=True),
    Column("outstanding_bal",DECIMAL(18, 2), nullable=True),
    Column("capital_pmt",    DECIMAL(18, 2), nullable=True),
    Column("int_pmt",        DECIMAL(18, 2), nullable=True),
    Column("total_pmt",      DECIMAL(18, 2), nullable=True),
    schema="cf",
)

IrSwapBeh = Table(
    "ir_swap_beh", metadata,
    Column("schedule_id",    String(8),      primary_key=True, nullable=False),
    Column("bs_side",        String(1),      nullable=False),
    Column("rate_index",     String(10),     nullable=False),
    Column("fixing_dt",      Date,           nullable=False),
    Column("cf_start_dt",    Date,           primary_key=True, nullable=False),
    Column("cf_end_dt",      Date,           nullable=False),
    Column("cf_yf",          DECIMAL(18, 6), nullable=False),
    Column("d_f",            DECIMAL(18, 6), nullable=True),
    Column("fwd_rt",         DECIMAL(18, 6), nullable=True),
    Column("outstanding_bal",DECIMAL(18, 2), nullable=True),
    Column("capital_pmt",    DECIMAL(18, 2), nullable=True),
    Column("int_pmt",        DECIMAL(18, 2), nullable=True),
    Column("total_pmt",      DECIMAL(18, 2), nullable=True),
    schema="cf",
)


def _ensure_schemas() -> None:
    for s in ("schemat", "sched", "cf", "irrbb"):
        with engine.begin() as conn:
            conn.execute(text(
                f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{s}') "
                f"EXEC('CREATE SCHEMA [{s}]');"
            ))


_IRS_SQL_TABLES = [
    "schemat.ir_swaps",
    "sched.ir_swaps",
    "cf.ir_swap_orig",
    "cf.ir_swap_beh",
    "irrbb.ir_swap_gap_beh",
]


def reset_data(mode: int) -> None:
    """mode=0: drop + recreate.  mode=1: delete all rows."""
    _ensure_schemas()
    with engine.begin() as conn:
        if mode == 0:
            for t in _IRS_SQL_TABLES:
                conn.execute(text(
                    f"IF OBJECT_ID('{t}', 'U') IS NOT NULL DROP TABLE {t};"
                ))
            metadata.create_all(bind=conn, checkfirst=False)
        elif mode == 1:
            for t in _IRS_SQL_TABLES:
                conn.execute(text(
                    f"IF OBJECT_ID('{t}', 'U') IS NOT NULL DELETE FROM {t};"
                ))
        else:
            raise ValueError("mode must be 0 or 1")


def _build_legs_df(df_swaps: pd.DataFrame) -> pd.DataFrame:
    """Expand swap-level input (one row per swap) into leg-level rows (two per swap).

    Returns a DataFrame ready to write to schemat.ir_swaps / sched.ir_swaps.
    schedule_id is assigned sequentially: fixed leg = 2k-1, float leg = 2k.
    """
    rows = []
    sched_id = 1
    for _, row in df_swaps.iterrows():
        fixed_side = "L" if int(row.pay_fixed) == 1 else "A"
        float_side = "A" if int(row.pay_fixed) == 1 else "L"

        # Fixed leg
        rows.append({
            "report_date":   row.report_date,
            "swap_id":       row.swap_id,
            "leg_id":        f"{row.swap_id}-F",
            "product_code":  "0000",
            "leg_type":      "FIXED",
            "bs_side":       fixed_side,
            "swap_type":     row.swap_type,
            "notional":      float(row.notional),
            "currency":      row.currency,
            "start_date":    row.start_date,
            "maturity_date": row.maturity_date,
            "pay_freq":      row.fixed_pay_freq,
            "fixing_freq":   None,
            "rate_index":    "FIXED",
            "dc_conv":       row.fixed_dc_conv,
            "b_day_conv":    row.fixed_b_day_conv,
            "fixed_rate":    float(row.fixed_rate),
            "float_spread":  None,
            "disc_curve":    row.disc_curve,
            "fwd_curve":     row.fwd_curve,
            "schedule_id":   sched_id,
        })
        sched_id += 1

        # Float leg
        rows.append({
            "report_date":   row.report_date,
            "swap_id":       row.swap_id,
            "leg_id":        f"{row.swap_id}-V",
            "product_code":  "0000",
            "leg_type":      "FLOAT",
            "bs_side":       float_side,
            "swap_type":     row.swap_type,
            "notional":      float(row.notional),
            "currency":      row.currency,
            "start_date":    row.start_date,
            "maturity_date": row.maturity_date,
            "pay_freq":      row.float_pay_freq,
            "fixing_freq":   row.float_fixing_freq,
            "rate_index":    row.float_rate_index,
            "dc_conv":       row.float_dc_conv,
            "b_day_conv":    row.float_b_day_conv,
            "fixed_rate":    None,
            "float_spread":  float(row.float_spread) if pd.notna(row.float_spread) else 0.0,
            "disc_curve":    row.disc_curve,
            "fwd_curve":     row.fwd_curve,
            "schedule_id":   sched_id,
        })
        sched_id += 1

    legs = pd.DataFrame(rows)
    legs["start_date"]    = pd.to_datetime(legs["start_date"])
    legs["maturity_date"] = pd.to_datetime(legs["maturity_date"])
    return legs


def load_and_insert_ir_swaps(df_swaps: pd.DataFrame) -> pd.DataFrame:
    """Expand swap-level DataFrame to leg-level, write to schemat.ir_swaps
    and sched.ir_swaps, and return the legs DataFrame for in-memory use."""
    legs = _build_legs_df(df_swaps)

    # schemat.ir_swaps — match SQLAlchemy column list
    sa_cols = [c.name for c in IRSwaps.columns]
    df_w = legs.copy()
    for c in sa_cols:
        if c not in df_w.columns:
            df_w[c] = pd.NA
    df_w = df_w[sa_cols].where(pd.notnull(df_w[sa_cols]), None)
    df_w.to_sql("ir_swaps", engine, schema="schemat",
                if_exists="append", index=False, chunksize=10_000)

    # sched.ir_swaps — full legs DataFrame (for CF generation joins)
    legs.to_sql("ir_swaps", engine, schema="sched",
                if_exists="replace", index=False)

    return legs


def write_df(df: pd.DataFrame, table: str, schema: str = "cf",
             chunksize: int = 50_000) -> None:
    df.to_sql(table, engine, schema=schema,
              if_exists="append", index=False, chunksize=chunksize)


# ── Market data loading ───────────────────────────────────────────────────────

def _interpolate_d_f(x: np.ndarray, y: np.ndarray,
                     report_date: pd.Timestamp) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    if not np.any(x == 0):
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 1.0)
    else:
        y[x == 0] = 1.0
    grid = np.arange(0, int(x.max()) + 1, dtype=float)
    ln_y_interp = np.interp(grid, x, np.log(y))
    return pd.DataFrame({
        "n_days":    grid.astype(int),
        "node_date": report_date + pd.to_timedelta(grid, unit="D"),
        "d_f":       np.exp(ln_y_interp),
    })


def load_disc_curve(curve_name: str, report_date: pd.Timestamp) -> pd.Series:
    """Return Series indexed by (curve_name, node_date) -> d_f."""
    q = text("""
        SELECT n_days, d_f FROM mkt.curves
        WHERE curve_date = :rd AND curve_name = :cn
    """)
    df = pd.read_sql_query(q, engine, params={"rd": report_date, "cn": curve_name})
    interp = _interpolate_d_f(df["n_days"].to_numpy(), df["d_f"].to_numpy(), report_date)
    interp.insert(0, "curve_name", curve_name)
    return interp.set_index(["curve_name", "node_date"])["d_f"].sort_index()


def load_fwd_curve(curve_name: str, fixing_freq: str,
                   currency: str, report_date: pd.Timestamp) -> pd.Series:
    """Return Series indexed by (curve_name, fixing_freq, node_date) -> fwd_rt."""
    import QuantLib as ql

    _DC  = {"PLN": ql.Actual365Fixed(),
             "USD": ql.ActualActual(ql.ActualActual.ISDA),
             "EUR": ql.Thirty360(ql.Thirty360.ISDA)}
    _CAL = {"PLN": ql.Poland(), "EUR": ql.TARGET(),
             "USD": ql.UnitedStates(ql.UnitedStates.NYSE)}

    dc  = _DC.get(currency, ql.Actual365Fixed())
    cal = _CAL.get(currency, ql.TARGET())

    q = text("""
        SELECT n_days, d_f FROM mkt.curves
        WHERE curve_date = :rd AND curve_name = :cn
    """)
    df = pd.read_sql_query(q, engine, params={"rd": report_date, "cn": curve_name})
    interp = _interpolate_d_f(df["n_days"].to_numpy(), df["d_f"].to_numpy(), report_date)

    fp = ql.Period(fixing_freq)
    end_dts = [
        cal.advance(ql.Date.from_date(d.date()), fp, ql.ModifiedFollowing)
        for d in interp["node_date"]
    ]
    end_pds = pd.to_datetime(
        [f"{d.year()}-{d.month():02d}-{d.dayOfMonth():02d}" for d in end_dts]
    )
    yf_vec = np.array([
        dc.yearFraction(ql.Date.from_date(s.date()), e)
        for s, e in zip(interp["node_date"], end_dts)
    ])
    df_helper = interp.set_index("node_date")["d_f"]
    d_f_end   = end_pds.map(df_helper).to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_rt = np.where(
            (yf_vec > 0) & (d_f_end > 0) & ~np.isnan(d_f_end),
            (interp["d_f"].to_numpy() / d_f_end - 1.0) / yf_vec,
            np.nan,
        )

    result = pd.DataFrame({
        "curve_name":  curve_name,
        "fixing_freq": fixing_freq,
        "node_date":   interp["node_date"],
        "fwd_rt":      fwd_rt,
    }).dropna(subset=["fwd_rt"])

    return result.set_index(["curve_name", "fixing_freq", "node_date"])["fwd_rt"].sort_index()


def load_fixings() -> pd.Series:
    """Return Series indexed by (fixing_date, rate_index) -> rate."""
    df = pd.read_sql_query("SELECT fixing_date, rate_index, rate FROM mkt.fixings", engine)
    df["fixing_date"] = pd.to_datetime(df["fixing_date"])
    return df.set_index(["fixing_date", "rate_index"])["rate"].sort_index()


def load_sched_ir_swaps() -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM sched.ir_swaps", engine)
    df["start_date"]    = pd.to_datetime(df["start_date"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])
    return df


def append_irs_to_ir_gap_tables(
    swap_gap: pd.DataFrame,
    report_date: pd.Timestamp,
) -> None:
    """Merge IRS gap into irrbb.ir_gap_beh (gap_cf_irs column) and append
    IRS rows (product_code='0000') to irrbb.ir_gap_beh_a.

    For tenor buckets already present in ir_gap_beh the gap_cf_irs is updated;
    for IRS-only buckets new rows are inserted (gap_cf=0, gap_cf_ca=0).
    bucket_start_dt / bucket_end_dt are (re-)derived for all rows so the
    bounds columns stay consistent after the merge.
    """
    import irs_objects as irs_obj  # local import to avoid circular dep

    merge_keys = ["currency", "bs_side", "tenor_bucket", "cf_end_dt"]

    # ── 1. Update ir_gap_beh ──────────────────────────────────────────────────
    existing = pd.read_sql_query("SELECT * FROM irrbb.ir_gap_beh", engine)
    for dt_col in ("cf_end_dt", "bucket_start_dt", "bucket_end_dt"):
        if dt_col in existing.columns:
            existing[dt_col] = pd.to_datetime(existing[dt_col])

    # Ensure split columns exist (backward-compat if table predates this change)
    for col in ("gap_cf_ca", "gap_cf_irs"):
        if col not in existing.columns:
            existing[col] = 0.0

    irs_agg = (
        swap_gap
        .groupby(merge_keys, sort=False)["gap_cf"]
        .sum()
        .reset_index()
        .rename(columns={"gap_cf": "_irs_new"})
    )

    merged = existing.merge(irs_agg, on=merge_keys, how="outer")
    merged["gap_cf"]     = merged["gap_cf"].fillna(0.0)
    merged["gap_cf_ca"]  = merged["gap_cf_ca"].fillna(0.0)
    merged["gap_cf_irs"] = merged["_irs_new"].fillna(0.0)
    merged = merged.drop(columns=["_irs_new"])

    # Re-derive bucket bounds for all rows (fills NaT for IRS-only new rows)
    bounds = merged["tenor_bucket"].map(
        lambda t: irs_obj._bucket_bounds(t, report_date)
    )
    merged["bucket_start_dt"] = [b[0] for b in bounds]
    merged["bucket_end_dt"]   = [b[1] for b in bounds]

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM irrbb.ir_gap_beh"))
    merged.to_sql("ir_gap_beh", engine, schema="irrbb",
                  if_exists="append", index=False, chunksize=10_000)

    # ── 2. Append IRS rows to ir_gap_beh_a ───────────────────────────────────
    keep = ["currency", "bs_side", "tenor_bucket",
            "bucket_start_dt", "bucket_end_dt", "cf_end_dt", "gap_cf"]
    irs_a = swap_gap[[c for c in keep if c in swap_gap.columns]].copy()
    irs_a["product_code"] = "0000"
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM irrbb.ir_gap_beh_a WHERE product_code = '0000'"))
    irs_a.to_sql("ir_gap_beh_a", engine, schema="irrbb",
                 if_exists="append", index=False, chunksize=10_000)
