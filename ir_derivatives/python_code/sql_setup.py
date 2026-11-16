from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, Column, Integer, String, Date, text
from sqlalchemy.dialects.mssql import DECIMAL

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common.bank_config import get_engine

engine = get_engine()   # active bank's database (LABBANK_BANK, default eur)

metadata = MetaData(schema=None)

# ── schemat.ir_swaps — leg-level: two rows per swap (FIXED + FLOAT) ───────────
# swap_id groups the two legs of one transaction.
# leg_id  is the primary key (e.g. "IRS-2024-001-F" / "IRS-2024-001-V").
IRSwaps = Table(
    "ir_swaps", metadata,
    Column("report_date",   Date,           nullable=False),
    Column("swap_id",       String(13),     nullable=False),   # parent swap reference
    Column("leg_id",        String(16),     primary_key=True, nullable=False),
    Column("product_code",  String(4),      nullable=False),   # '0000' swap book, '0100' FX deals
    Column("leg_type",      String(5),      nullable=False),   # FIXED / FLOAT / FX
    Column("bs_side",       String(1),      nullable=False),   # A / L
    Column("swap_type",     String(4),      nullable=False),   # IRS / OIS / CIRS / FXS (FX deal)
    Column("notional",      DECIMAL(18, 2), nullable=False),
    Column("currency",      String(3),      nullable=False),
    Column("start_date",    Date,           nullable=False),
    Column("maturity_date", Date,           nullable=False),
    Column("pay_freq",      String(4),      nullable=False),
    Column("fixing_freq",   String(4),      nullable=True),    # NULL for FIXED leg
    Column("rate_index",    String(16),     nullable=False),   # 'FIXED' or e.g. 'WIBOR3M'
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
    Column("rate_index",     String(16),     nullable=False),
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
    Column("rate_index",     String(16),     nullable=False),
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
        # CIRS: the float leg is in its own currency and notional, on its currency's curves
        fl_ccy = row.get("float_currency") if pd.notna(row.get("float_currency")) else row.currency
        fl_not = row.get("float_notional") if pd.notna(row.get("float_notional")) else row.notional
        two_ccy = fl_ccy != row.currency
        exch = int(row.get("notional_exchange")) if pd.notna(row.get("notional_exchange")) else 0

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
            "notional_exchange": exch,
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
            "notional":      float(fl_not),
            "currency":      fl_ccy,
            "start_date":    row.start_date,
            "maturity_date": row.maturity_date,
            "pay_freq":      row.float_pay_freq,
            "fixing_freq":   row.float_fixing_freq,
            "rate_index":    row.float_rate_index,
            "dc_conv":       row.float_dc_conv,
            "b_day_conv":    row.float_b_day_conv,
            "fixed_rate":    None,
            "float_spread":  float(row.float_spread) if pd.notna(row.float_spread) else 0.0,
            "disc_curve":    f"{fl_ccy}_disc_curve" if two_ccy else row.disc_curve,
            "fwd_curve":     f"{fl_ccy}_fwd_curve" if two_ccy else row.fwd_curve,
            "schedule_id":   sched_id,
            "notional_exchange": exch,
        })
        sched_id += 1

    legs = pd.DataFrame(rows)
    legs["start_date"]    = pd.to_datetime(legs["start_date"])
    legs["maturity_date"] = pd.to_datetime(legs["maturity_date"])
    return legs


def _build_fx_legs_df(fx_legs: pd.DataFrame, report_date: pd.Timestamp,
                      first_schedule_id: int) -> pd.DataFrame:
    """FX deal legs (labbank_common.derivatives.fx_legs, one per deal and currency) in the
    schemat.ir_swaps leg layout: product_code '0100', leg_type 'FX', zero-coupon ('ZC'),
    fixed_rate = the implied rate of the swap points, each currency on its own curves."""
    if fx_legs.empty:
        return pd.DataFrame()
    import QuantLib as ql
    dc = ql.Actual360()
    rows = []
    for i, r in enumerate(fx_legs.itertuples(index=False)):
        yf = dc.yearFraction(ql.Date.from_date(r.start_date.date()), ql.Date.from_date(r.maturity_date.date()))
        rows.append({
            "report_date":   report_date,
            "swap_id":       r.deal_id,
            "leg_id":        f"{r.deal_id}-{r.currency}",
            "product_code":  "0100",
            "leg_type":      "FX",
            "bs_side":       r.bs_side,
            "swap_type":     "FXS",
            "notional":      float(r.notional),
            "currency":      r.currency,
            "start_date":    r.start_date,
            "maturity_date": r.maturity_date,
            "pay_freq":      "ZC",
            "fixing_freq":   None,
            "rate_index":    "FIXED",
            "dc_conv":       "ACT/360",
            "b_day_conv":    "ModifiedFollowing",
            "fixed_rate":    round(r.interest / (r.outstanding * yf), 6) if r.outstanding and yf > 0 else 0.0,
            "float_spread":  None,
            "disc_curve":    f"{r.currency}_disc_curve",
            "fwd_curve":     f"{r.currency}_fwd_curve",
            "schedule_id":   first_schedule_id + i,
            # carried in sched.ir_swaps only (CF generation)
            "outstanding_bal": float(r.outstanding),
            "interest":        float(r.interest),
            "near_amount":     float(r.near_amount),
        })
    return pd.DataFrame(rows)


def write_fx_deals(deals: pd.DataFrame) -> None:
    """schemat.fx_deals: the fx_spot sheet as booked (liquidity reads the exchanges from
    here). Only banks with FX deals get the table."""
    with engine.begin() as conn:
        conn.execute(text("IF OBJECT_ID('schemat.fx_deals', 'U') IS NOT NULL DROP TABLE schemat.fx_deals;"))
    if not deals.empty:
        deals.to_sql("fx_deals", engine, schema="schemat", if_exists="replace", index=False)


def load_and_insert_ir_swaps(df_swaps: pd.DataFrame, fx_legs: pd.DataFrame | None = None,
                             report_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Expand swap-level DataFrame to leg-level (+ the FX deal legs), write to
    schemat.ir_swaps and sched.ir_swaps, and return the legs DataFrame for in-memory use."""
    legs = _build_legs_df(df_swaps) if not df_swaps.empty else pd.DataFrame()
    if fx_legs is not None and not fx_legs.empty:
        fx_rows = _build_fx_legs_df(fx_legs, report_date, len(legs) + 1)
        legs = pd.concat([legs, fx_rows], ignore_index=True)

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

    # Forward rate in the IBOR's own accrual basis: the float coupon is notional x fwd x
    # yf(float_dc_conv), so fwd must be quoted on the same day count (EURIBOR ACT/360,
    # WIBOR ACT/365). EUR used 30/360 here: 6M forwards came out ~1.1% too high (182/180
    # days) and shocked revaluations, which re-derive the forward on ACT/360, drifted
    # from the base by a constant offset.
    from labbank_common.curves import index_day_basis      # shared with cash_flow_calc
    _DC  = {c: (ql.Actual360() if index_day_basis(c) == 360 else ql.Actual365Fixed())
            for c in ("PLN", "USD", "EUR", "GBP")}
    _CAL = {"PLN": ql.Poland(), "EUR": ql.TARGET(),
             "USD": ql.UnitedStates(ql.UnitedStates.NYSE), "GBP": ql.UnitedKingdom()}

    dc  = _DC.get(currency, ql.Actual365Fixed())
    cal = _CAL.get(currency, ql.TARGET())

    q = text("""
        SELECT n_days, d_f FROM mkt.curves
        WHERE curve_date = :rd AND curve_name = :cn
    """)
    # projection curve of this IBOR tenor when the bank has one (multi-curve, labbank_common/
    # curves.py); the result keeps the leg's fwd_curve name as its key
    from labbank_common.curves import projection_curve
    names = frozenset(pd.read_sql_query(text("SELECT DISTINCT curve_name FROM mkt.curves WHERE curve_date = :rd"),
                                        engine, params={"rd": report_date})["curve_name"])
    df = pd.read_sql_query(q, engine, params={"rd": report_date,
                                              "cn": projection_curve(curve_name, fixing_freq, names)})
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
    gap_col: str = "gap_cf_irs",
    product_code: str = "0000",
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
    for col in ("gap_cf_ca", "gap_cf_irs", gap_col):
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
    merged["gap_cf_irs"] = merged["gap_cf_irs"].fillna(0.0)
    if "gap_cf_fx" in merged.columns:
        merged["gap_cf_fx"] = merged["gap_cf_fx"].fillna(0.0)
    merged[gap_col]      = merged["_irs_new"].fillna(0.0)   # assigned, not added: rerun-safe
    merged = merged.drop(columns=["_irs_new"])

    # Re-derive bucket bounds for all rows (fills NaT for IRS-only new rows)
    bounds = merged["tenor_bucket"].map(
        lambda t: irs_obj._bucket_bounds(t, report_date)
    )
    merged["bucket_start_dt"] = [b[0] for b in bounds]
    merged["bucket_end_dt"]   = [b[1] for b in bounds]

    with engine.begin() as conn:
        for col in merged.columns.intersection(["gap_cf_ca", "gap_cf_irs", "gap_cf_fx"]):
            conn.execute(text(
                f"IF COL_LENGTH('irrbb.ir_gap_beh', '{col}') IS NULL "
                f"ALTER TABLE irrbb.ir_gap_beh ADD {col} FLOAT NULL"))
        conn.execute(text("DELETE FROM irrbb.ir_gap_beh"))
    merged.to_sql("ir_gap_beh", engine, schema="irrbb",
                  if_exists="append", index=False, chunksize=10_000)

    # ── 2. Append IRS rows to ir_gap_beh_a ───────────────────────────────────
    keep = ["currency", "bs_side", "tenor_bucket",
            "bucket_start_dt", "bucket_end_dt", "cf_end_dt", "gap_cf"]
    irs_a = swap_gap[[c for c in keep if c in swap_gap.columns]].copy()
    irs_a["product_code"] = product_code
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM irrbb.ir_gap_beh_a WHERE product_code = :pc"), {"pc": product_code})
    irs_a.to_sql("ir_gap_beh_a", engine, schema="irrbb",
                 if_exists="append", index=False, chunksize=10_000)
