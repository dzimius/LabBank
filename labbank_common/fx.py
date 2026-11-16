"""fx.py
=====
Foreign-currency positions inside one bank: market inputs per currency and FX conversion
to the bank's home (reporting) currency.

A bank books most products in its home currency (``bank.market.ccy``) and may hold a few
in other currencies (``bank.market.foreign_ccys``, e.g. the EUR bank's USD corporate
current account and USD corporate loan). Every position is stored in its OWN currency
(contract amounts, cash flows, NII / EVE per currency); amounts are converted to the home
currency only where currencies are added up (balance-sheet totals, LCR / NSFR, IRRBB
totals vs Tier 1, the optimizer / sandbox tensors).

Market inputs
-------------
The home currency's curves / fixings come from the bank's own input folder
(``inputs/<bank_id>/models/``); a foreign currency's from the shared
currency folder (``inputs/market/<ccy>/``, e.g. ``input/usd/``), so the
USD curve is built once and used by every bank that holds USD.

FX rates
--------
``fx_input.xlsx`` in the bank's input folder: ECB euro reference rates, long format
(date, currency, rate), rate = units of the currency per 1 unit of home currency.
Loaded into SQL table mkt.fx_rates by the add_data stage. Conversion uses the rate on the
report date (or the last one before it).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from labbank_common.bank_config import MARKET_ROOT, BankConfig, get_bank

FX_FILE = "fx_input.xlsx"
# FX hedging policy: FX swaps cover this share of the short open position in a currency
# (the roll-forward sizes its 3M FX-swap ladder with it; the ALM report checks the book against it)
FX_HEDGE_RATIO = 0.95
FX_LADDER_N = 3            # ladder of 3 x 3M FX swaps, one traded per month


def _bank(bank: BankConfig | str | None) -> BankConfig:
    return bank if isinstance(bank, BankConfig) else get_bank(bank)


def currencies(bank: BankConfig | str | None = None) -> list[str]:
    """Home currency first, then the foreign currencies the bank holds."""
    b = _bank(bank)
    return [b.market.ccy, *b.market.foreign_ccys]


def market_input_files(name: str, bank: BankConfig | str | None = None) -> list[Path]:
    """``name`` (curve_input.xlsx / fixing_input.xlsx) for every currency of the bank."""
    b = _bank(bank)
    files = [b.market_dir() / name]
    files += [foreign_input_dir(c, b) / name for c in b.market.foreign_ccys]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"market input missing: {missing}")
    return files


def foreign_input_dir(ccy: str, bank: BankConfig | str | None = None) -> Path:
    """Market inputs of a foreign currency: the shared inputs/market/<ccy>/, or for a
    rolled-forward bank its own rolled copy (<rolled_inputs>/<ccy>/, extended to its report date)."""
    b = _bank(bank)
    if b.rollforward is not None:
        return b.market_dir() / ccy.lower()
    return MARKET_ROOT / ccy.lower()


def read_market_input(name: str, bank: BankConfig | str | None = None) -> pd.DataFrame:
    """All currencies' rows of a market input file, concatenated."""
    return pd.concat([pd.read_excel(f) for f in market_input_files(name, bank)], ignore_index=True)


@lru_cache(maxsize=None)
def _fx_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def fx_input_path(bank: BankConfig | str | None = None) -> Path:
    return _bank(bank).market_dir() / FX_FILE


def fx_rates(bank: BankConfig | str | None = None, date=None) -> dict[str, float]:
    """Units of each foreign currency per 1 unit of home currency on ``date`` (default:
    the report date; the last available rate on or before it). Home currency -> 1.0."""
    b = _bank(bank)
    out = {b.market.ccy: 1.0}
    if not b.market.foreign_ccys:
        return out
    date = pd.Timestamp(date or b.report_date)
    tbl = _fx_table(str(fx_input_path(b)))
    for c in b.market.foreign_ccys:
        s = tbl[(tbl["currency"] == c) & (tbl["date"] <= date)]
        if s.empty:
            raise ValueError(f"no {c} FX rate on or before {date.date()} in {fx_input_path(b)}")
        out[c] = float(s["rate"].iloc[-1])
    return out


def to_home(bank: BankConfig | str | None = None, date=None) -> dict[str, float]:
    """Multiplier per currency: home-currency value of 1 unit of the currency."""
    return {c: 1.0 / r for c, r in fx_rates(bank, date).items()}


def to_home_series(ccy: pd.Series, bank: BankConfig | str | None = None, date=None) -> pd.Series:
    """Per-row conversion factor for a currency column (unknown currency -> error)."""
    m = to_home(bank, date)
    unknown = set(ccy.dropna().astype(str)) - set(m)
    if unknown:
        raise ValueError(f"no FX rate for {sorted(unknown)} (bank {_bank(bank).bank_id})")
    return ccy.astype(str).map(m).astype(float)
