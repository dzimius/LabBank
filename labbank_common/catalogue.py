"""catalogue.py
============
Product VARIANTS the bank can sell but does not hold: sheet ``catalogue`` of
``inputs/<bank_id>/balance/bank_data.xlsx``, one row per variant.

  parent_code   product_code of a bs_structure row (the product it is a variant of)
  bs_side       side of that row (7900 is on both sides)
  product_code  the variant's own code (4 digits, not used in bs_structure)
  own_name      the variant's own name
  maturity, payment_freq, fixing_freq, rate_index, amort_type, rate_type
                the contract terms that differ -- blank = the parent's (rate_type V turns a
                fixed parent into its floating variant, e.g. a floating-rate bond)

Everything else comes from the parent: its bs_structure row (product_name = generator
class, currency, client type, day count, curves, LCR / NSFR / RWA / PD / LGD / fees /
acquisition cost) and its behavioural models -- the client-rate model (interest_rt.xlsx),
the prepayment model (loan_beh_models.xlsx), the term-deposit redemption model
(dep_redemption_models.xlsx) and the NMD models (dep_beh_models_*.xlsx). A variant is
therefore the parent's product with other contract terms: it prepays, redeems and reprices
like the parent.

Every reader of bs_structure goes through ``read_bs_structure``: the variants become
ordinary rows with bs_percentage 0 -- catalogue products (nothing held today), which the
roll-forward can sell (growth plan) and the optimizers can buy (catalogue_params.py). The
LabBank app (sandbox) hides them. Expansion is idempotent: a variant whose code is already
a bs_structure row (a rolled-forward bank that sold it) is not added twice, and its model
rows are only copied where the model files do not have them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SHEET = "catalogue"
OVERRIDES = ["maturity", "payment_freq", "fixing_freq", "rate_index", "amort_type", "rate_type"]


def bank_data_path(bank=None) -> Path:
    from labbank_common.bank_config import BankConfig, get_bank
    b = bank if isinstance(bank, BankConfig) else get_bank(bank)
    return b.balance_dir() / "bank_data.xlsx"


def _code(x) -> str:
    return str(int(float(x)))


def read_variants(path) -> pd.DataFrame:
    """The catalogue sheet (empty frame if the workbook has none)."""
    try:
        df = pd.read_excel(path, sheet_name=SHEET)
    except ValueError:                                   # no such sheet
        return pd.DataFrame(columns=["parent_code", "bs_side", "product_code", "own_name", *OVERRIDES])
    df = df.dropna(subset=["product_code"]).copy()
    if df.empty:
        return df
    for c in ("parent_code", "bs_side", "own_name"):
        if c not in df.columns or df[c].isna().any():
            raise ValueError(f"catalogue sheet: {c} is required on every row")
    df["parent_code"] = df["parent_code"].map(_code)
    df["product_code"] = df["product_code"].map(_code)
    for c in OVERRIDES:
        if c not in df.columns:
            df[c] = pd.NA
    dup = df["product_code"].duplicated()
    if dup.any():
        raise ValueError(f"catalogue sheet: duplicate product_code {sorted(df.loc[dup, 'product_code'])}")
    return df.reset_index(drop=True)


def expand_bs_structure(bs: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    """bs_structure + one bs_percentage-0 row per variant not already in it."""
    if variants.empty:
        return bs
    codes = bs["product_code"].map(_code)
    rows = []
    for v in variants.itertuples(index=False):
        if (codes == v.product_code).any():
            continue                                     # already a product of this bank
        par = bs[(codes == v.parent_code) & (bs["bs_side"] == v.bs_side)]
        if len(par) != 1:
            raise ValueError(f"catalogue variant {v.product_code}: parent {v.parent_code}/{v.bs_side} "
                             f"must be exactly one bs_structure row (found {len(par)})")
        r = par.iloc[0].copy()
        r["product_code"] = type(r["product_code"])(int(v.product_code)) if not isinstance(r["product_code"], str) \
            else v.product_code
        r["own_name"] = v.own_name
        r["bs_percentage"] = 0.0
        for c in OVERRIDES:
            val = getattr(v, c, None)                    # a catalogue may lack a newer override column
            if pd.notna(val):
                r[c] = val
        rows.append(r)
    return pd.concat([bs, pd.DataFrame(rows)], ignore_index=True) if rows else bs


def read_bs_structure(path) -> pd.DataFrame:
    """bs_structure of bank_data.xlsx with the catalogue variants as bs_percentage-0 rows."""
    return expand_bs_structure(pd.read_excel(path, sheet_name="bs_structure"), read_variants(path))


def parent_map(bank=None, path=None) -> dict[str, str]:
    """{variant product_code: parent product_code} (strings)."""
    v = read_variants(path or bank_data_path(bank))
    return dict(zip(v["product_code"], v["parent_code"]))


def expand_rows(df: pd.DataFrame, pmap: dict[str, str], col: str = "product_code") -> pd.DataFrame:
    """Model table keyed by a product_code column: the parent's rows copied for each variant
    that has none (interest_rt.xlsx, dep_redemption_models.xlsx)."""
    if not pmap or df.empty:
        return df
    codes = df[col].map(_code)
    add = []
    for var, par in pmap.items():
        if (codes == var).any():
            continue
        rows = df[codes == par].copy()
        if rows.empty:
            continue                                     # parent has no model row either
        orig = df[col].iloc[0]
        rows[col] = var if isinstance(orig, str) else type(orig)(int(var))
        add.append(rows)
    return pd.concat([df, *add], ignore_index=True) if add else df


def expand_columns(df: pd.DataFrame, pmap: dict[str, str]) -> pd.DataFrame:
    """Model table with one column per product_code: the parent's column copied for each
    variant that has none (loan_beh_models.xlsx, dep_beh_models_*.xlsx)."""
    if not pmap:
        return df
    by_code = {}
    for c in df.columns:
        try:
            by_code[_code(c)] = c
        except (TypeError, ValueError):
            pass                                         # report_date, tenor
    out = df.copy()
    for var, par in pmap.items():
        if var in by_code or par not in by_code:
            continue
        src = by_code[par]
        out[type(src)(int(var)) if not isinstance(src, str) else var] = df[src]
    return out
