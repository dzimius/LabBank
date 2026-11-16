"""Accounting classification and parameters shared by the accounting workflows.

Design: docs/07_pnl_capital_accounting_design.md.

Classification
--------------
bs_structure (bank_data.xlsx) columns, per product:
  accounting   AC / FVOCI / FVTPL   blank = AC (loans, deposits, issued bonds, cash, equity)
  portfolio    liquidity / investment / trading   blank = from the class (FVOCI liquidity,
               AC investment, FVTPL trading); only securities carry one
derivatives.xlsx sheet swap, column hedge_accounting (labbank_common/derivatives.py):
  none        the swap is FVTPL: its fair-value change goes to P&L
  fair_value  fair-value macro hedge, assumed 100% effective: the hedged items' fair-value
              adjustment offsets the swap's change in P&L (both lines shown, net 0)
  cash_flow   the change goes to the cash-flow hedge reserve (OCI), filtered out of CET1
              (CRR art. 33(1)(a))
FX deals (product 0100) are economic hedges of foreign-currency positions: FVTPL.

Parameters (bank_data.xlsx sheet fin_data, parameter / value)
-------------------------------------------------------------
  CIT              corporate income tax rate             default 0.30 (Germany: 15% + 5.5% soli
                                                                       + ~14% trade tax = 29.8%)
  dividend_payout  share of last financial year's profit default 0.30
  dividend_month   calendar month the dividend is paid   default 5 (after the AGM)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from labbank_common import catalogue  # noqa: E402
from labbank_common.bank_config import BankConfig, get_bank  # noqa: E402

CLASSES = ("AC", "FVOCI", "FVTPL")
PORTFOLIO_OF = {"AC": "investment", "FVOCI": "liquidity", "FVTPL": "trading"}
SECURITY_NAMES = frozenset({"bond_fixed", "bond_float", "t_bill"})
EQUITY_NAMES = frozenset({"retained_earnings", "common_shares", "risk_reserves"})
EBA6 = ("par_up", "par_dn", "sr_up", "sr_dn", "steep", "flat")
FIN_DEFAULTS = {"CIT": 0.30, "dividend_payout": 0.30, "dividend_month": 5.0, "CET1": 0.12}
IRS_CODE, FX_CODE = "0000", "0100"


def _bank(bank) -> BankConfig:
    return bank if isinstance(bank, BankConfig) else get_bank(bank)


def bank_data_path(bank) -> "os.PathLike":
    return _bank(bank).balance_dir() / "bank_data.xlsx"


def fin_params(bank) -> dict:
    """fin_data of bank_data.xlsx with the accounting defaults filled in."""
    out = dict(FIN_DEFAULTS)
    try:
        fd = pd.read_excel(bank_data_path(bank), sheet_name="fin_data")
        out.update({str(r.parameter).strip(): float(r.value) for r in fd.itertuples() if pd.notna(r.value)})
    except (ValueError, FileNotFoundError):
        pass
    return out


def product_classes(bank, path=None) -> pd.DataFrame:
    """One row per (product_code, bs_side) of bs_structure with accounting, portfolio,
    is_security, is_equity and the columns the P&L needs (PD, LGD, fees, costs, spread, rwa)."""
    bs = catalogue.read_bs_structure(path or bank_data_path(bank)).copy()
    bs["product_code"] = bs["product_code"].astype(int)
    for c in ("accounting", "portfolio", "mv_spread_bp", "PD", "LGD", "fee_unit_rate", "acq_cost_rate",
              "rwa_weight"):
        if c not in bs.columns:
            bs[c] = np.nan
    b = _bank(bank)
    if path is None and b.rollforward is not None:
        # a rolled bank's bank_data is generated from its base's: the classification is the base's
        # (rolled inputs written before the columns existed lack them)
        src = product_classes(b.rollforward.base_bank).set_index(["product_code", "bs_side"])
        idx = pd.MultiIndex.from_frame(bs[["product_code", "bs_side"]])
        for c in ("accounting", "portfolio", "mv_spread_bp"):
            own = bs[c].astype(object)
            bs[c] = own.where(own.notna(), pd.Series(src[c].reindex(idx).to_numpy(), index=bs.index))
    acc = bs["accounting"].astype(object).where(bs["accounting"].notna(), "AC").astype(str).str.strip().str.upper()
    bad = set(acc) - set(CLASSES)
    if bad:
        raise ValueError(f"bs_structure: accounting must be one of {CLASSES}, got {sorted(bad)}")
    bs["accounting"] = acc
    bs["is_security"] = (bs["bs_side"] == "A") & bs["product_name"].isin(SECURITY_NAMES)
    bs["is_equity"] = bs["product_name"].isin(EQUITY_NAMES)
    fv_non_sec = (bs["accounting"] != "AC") & ~bs["is_security"]
    if fv_non_sec.any():
        raise ValueError("bs_structure: FVOCI / FVTPL is only modelled for securities, not for "
                         f"{bs.loc[fv_non_sec, 'own_name'].tolist()}")
    port = bs["portfolio"].astype(object)
    bs["portfolio"] = np.where(bs["is_security"], port.where(port.notna(), bs["accounting"].map(PORTFOLIO_OF)), None)
    bs["mv_spread_bp"] = pd.to_numeric(bs["mv_spread_bp"], errors="coerce").fillna(0.0)
    keep = ["product_code", "bs_side", "own_name", "product_name", "currency", "rate_type", "accounting",
            "portfolio", "is_security", "is_equity", "mv_spread_bp", "PD", "LGD", "fee_unit_rate",
            "acq_cost_rate", "rwa_weight", "bs_percentage"]
    return bs.drop_duplicates(["product_code", "bs_side"])[keep].reset_index(drop=True)


def swap_book(bank) -> pd.DataFrame:
    """The bank's swap book (derivatives.read_swap_book). A rolled bank's held trades take their
    hedge_accounting flag from the base bank's book: the designation is made at trade date and the
    roll-forward only copies the trade (rolled inputs written before the column existed lack it)."""
    from labbank_common import derivatives as deriv
    b = _bank(bank)
    sw = deriv.read_swap_book(b).reset_index(drop=True)
    if b.rollforward is not None:
        base = deriv.read_swap_book(b.rollforward.base_bank).set_index("swap_id")["hedge_accounting"]
        sw["hedge_accounting"] = sw["swap_id"].map(base).fillna(sw["hedge_accounting"])
    return sw


def swap_flags(bank) -> pd.DataFrame:
    """swap_id, swap_type, pay_fixed, notional, currency, maturity_date, hedge_accounting."""
    return swap_book(bank)[["swap_id", "swap_type", "pay_fixed", "notional", "currency", "start_date",
                            "maturity_date", "fixed_rate", "hedge_accounting"]]


def interp_df(nodes_days: np.ndarray, nodes_df: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Discount factors at ``days`` by log-linear interpolation (flat zero rate beyond the ends)."""
    x = np.asarray(nodes_days, float)
    y = np.log(np.asarray(nodes_df, float))
    d = np.asarray(days, float)
    out = np.interp(d, x, y)
    hi = d > x[-1]
    if hi.any():                                  # flat zero rate beyond the last node
        out[hi] = y[-1] / x[-1] * d[hi]
    lo = d < x[0]
    if lo.any():
        out[lo] = y[0] / x[0] * d[lo]
    out[d <= 0] = 0.0
    return np.exp(out)
