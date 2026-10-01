"""rate_models.py
================
Single source of truth for the client-rate model (interest_rt.xlsx -> bs.models_rate).

One formula for every product, used by balance generation, the cash-flow
engine, NII / EVE, the fast re-pricing layer and the sandbox:

    client_rt = clip( beta * clip(index, index_floor, index_cap) + spread,
                      client_floor, client_cap )

    spread = margin_pct / 100      when the product has a product-level margin
                                   (margin_pct set and != 0 -- e.g. a deposit tariff)
           = contract margin       otherwise (e.g. mortgages: margin drawn per
                                   contract at generation, stored in schemat.loans)

So margin_pct blank or 0 both mean "no product-level spread": loans then keep
their own contract margin, deposits / instruments without one get 0.

Storage
-------
Input file : balance_gen_add_data/input/interest_rt.xlsx  (one row per
             report_date x product_code; bs_side A / L / A/L is descriptive --
             liability rows must set beta, asset rows may leave beta and
             margin_pct blank = beta 1 + contract margin) -- loaded by the add_data stage into
             SQL table bs.models_rate, next to the other behavioural models.
Everything downstream of add_data reads bs.models_rate via load_rate_models(),
so a bank feeding its own data can insert rows into bs.models_rate directly
and never touch the Excel file.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXCEL_PATH = os.path.join(PROJECT_ROOT, "balance_gen_add_data", "input", "interest_rt.xlsx")
SQL_TABLE = "bs.models_rate"

RATE_COLS = ["beta", "margin_pct", "index_floor", "client_floor", "index_cap", "client_cap"]


# ── loading ──────────────────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = {"report_date", "product_code"} - set(df.columns)
    if missing:
        raise ValueError(f"rate model is missing columns: {sorted(missing)}")
    for c in RATE_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.normalize()
    df["product_code"] = df["product_code"].astype(float).astype(int).astype(str)
    # bs_side is descriptive (A, L, or A/L for a product on both sides, e.g. 7900);
    # the model stays keyed by product_code. Liability rows (deposit tariffs) must
    # state beta explicitly -- a blank beta there would silently mean full pass-through.
    if "bs_side" not in df.columns:
        df["bs_side"] = None
    df["bs_side"] = df["bs_side"].where(df["bs_side"].notna(), None)
    liab_blank = df["bs_side"].astype(str).str.upper().eq("L") & df["beta"].isna()
    if liab_blank.any():
        bad = sorted(df.loc[liab_blank, "product_code"].unique())
        raise ValueError(f"liability products {bad}: beta must be set explicitly (blank beta = 1)")
    # blank beta = 1 (follows the index); blank margin_pct = use the contract margin
    df["beta"] = df["beta"].fillna(1.0)
    both_floors = df["index_floor"].notna() & df["client_floor"].notna()
    if both_floors.any():
        bad = sorted(df.loc[both_floors, "product_code"].unique())
        raise ValueError(f"products {bad}: set index_floor OR client_floor, not both")
    dup = df.duplicated(["report_date", "product_code"])
    if dup.any():
        bad = sorted(df.loc[dup, "product_code"].unique())
        raise ValueError(f"duplicate rate-model rows for products {bad}")
    return df[["report_date", "product_code", "bs_side"] + RATE_COLS].reset_index(drop=True)


def _select_report_date(df: pd.DataFrame, report_date, source: str) -> pd.DataFrame:
    rd = pd.Timestamp(report_date).normalize()
    out = df[df["report_date"] == rd]
    if out.empty:
        have = sorted(df["report_date"].dt.date.unique())
        raise ValueError(f"no rate-model rows for report_date {rd.date()} in {source} (have: {have})")
    return out.set_index("product_code")


def read_rate_excel(report_date=None, path: str = EXCEL_PATH) -> pd.DataFrame:
    """Rows of the Excel input. With report_date: that date only, indexed by product_code."""
    df = _normalise(pd.read_excel(path))
    return df if report_date is None else _select_report_date(df, report_date, path)


def load_rate_models(engine, report_date, required_codes=None) -> pd.DataFrame:
    """bs.models_rate for one report_date, indexed by product_code (str).

    required_codes: product codes that must have a row (e.g. every deposit
    product) -- a missing row raises instead of silently defaulting to beta=1.
    """
    from sqlalchemy import text  # lazy: the formula helpers need no DB stack
    q = text(f"SELECT * FROM {SQL_TABLE} WHERE report_date = :rd")
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"rd": pd.Timestamp(report_date).date()})
    if df.empty:
        raise ValueError(
            f"{SQL_TABLE} has no rows for report_date {pd.Timestamp(report_date).date()} -- "
            "run the add_data stage (b_s_add_data_workflow.py) or insert the bank's rate model."
        )
    out = _normalise(df).set_index("product_code")
    if required_codes is not None:
        missing = sorted({str(int(c)) for c in required_codes} - set(out.index))
        if missing:
            raise ValueError(f"{SQL_TABLE} has no row for products {missing} on {pd.Timestamp(report_date).date()}")
    return out


def deposit_product_codes(engine) -> list[str]:
    """Deposit products present in sched.deposits -- every one needs a rate-model row."""
    from sqlalchemy import text
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT DISTINCT product_code FROM sched.deposits"), conn)
    return sorted(df["product_code"].astype(float).astype(int).astype(str))


# ── views used by the different engines ───────────────────────────────────────

def product_spread(rm: pd.DataFrame) -> pd.Series:
    """Product-level spread (decimal); NaN where the contract margin applies."""
    m = rm["margin_pct"] / 100.0
    return m.where(m.notna() & (m != 0.0))


def rate_maps(rm: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """(caps, floors, coeff_a, coeff_b) dicts keyed by product_code str, as used by
    the NII / EVE / sandbox code. coeff_b only holds product-level spreads."""
    caps = rm["client_cap"].dropna().astype(float).to_dict()
    floors = rm["client_floor"].dropna().astype(float).to_dict()
    coeff_a = rm.loc[rm["beta"] != 1.0, "beta"].astype(float).to_dict()
    coeff_b = product_spread(rm).dropna().astype(float).to_dict()
    return caps, floors, coeff_a, coeff_b


def cf_params(rm: pd.DataFrame) -> dict[int, dict]:
    """{product_code(int): {beta, spread, index_floor, index_cap, client_floor, client_cap}}
    for the cash-flow engine (NaN / None = not set)."""
    spread = product_spread(rm)
    out = {}
    for pc, r in rm.iterrows():
        out[int(pc)] = {
            "beta": float(r["beta"]),
            "spread": None if pd.isna(spread[pc]) else float(spread[pc]),
            **{k: (None if pd.isna(r[k]) else float(r[k]))
               for k in ("index_floor", "index_cap", "client_floor", "client_cap")},
        }
    return out


# ── the formula ──────────────────────────────────────────────────────────────

def client_rate(index, spread, beta=1.0, index_floor=np.nan, index_cap=np.nan,
                client_floor=np.nan, client_cap=np.nan):
    """Vectorised client-rate formula (all inputs decimal; NaN limit = not set)."""
    idx = np.asarray(index, dtype=float)
    idx = np.where(np.isnan(index_floor), idx, np.maximum(idx, index_floor))
    idx = np.where(np.isnan(index_cap), idx, np.minimum(idx, index_cap))
    rt = np.asarray(beta, dtype=float) * idx + np.nan_to_num(np.asarray(spread, dtype=float), nan=0.0)
    rt = np.where(np.isnan(client_floor), rt, np.maximum(rt, client_floor))
    rt = np.where(np.isnan(client_cap), rt, np.minimum(rt, client_cap))
    return rt
