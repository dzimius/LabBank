"""derivatives.py
==============
The bank's derivatives book: one input workbook per bank,
``inputs/<bank_id>/derivatives/derivatives.xlsx`` (next to bank_data.xlsx: the
bank's main input files sit together), with two sheets.

``swap`` -- the swap book, one row per swap
-------------------------------------------
Two legs per swap: a FIXED leg and a FLOAT leg (``pay_fixed`` = 1: the bank pays fixed).

  swap_type   currencies                          status
  IRS         one   (currency == float_currency)  priced
  OIS         one   (overnight index on the float leg)          not priced yet
  CIRS        two   (fixed leg in ``currency``, float leg in ``float_currency``,  priced
                     notionals exchanged at start and maturity)

Columns: swap_id, swap_type, pay_fixed, notional, currency, start_date, maturity_date,
fixed_rate, fixed_pay_freq, fixed_dc_conv, fixed_b_day_conv, float_rate_index,
float_pay_freq, float_fixing_freq, float_spread, float_dc_conv, float_b_day_conv,
disc_curve, fwd_curve (curves of ``currency``), and for CIRS: float_currency,
float_notional, notional_exchange (blank = IRS defaults: same currency / notional, 0).
hedge_accounting: none (FVTPL: the swap's fair-value change goes to P&L), fair_value (fair-value
macro hedge: the hedged items' adjustment offsets it in P&L) or cash_flow (the change goes to the
cash-flow hedge reserve in OCI); blank = fair_value. Read by accounting/ only.

A CIRS is two legs, each priced in its own currency on that currency's curves (the float leg's
curves are ``<float_currency>_disc_curve`` / ``_fwd_curve``). With notional_exchange = 1 each leg
repays its notional at maturity (and pays / receives it at the start when that is still ahead), so
the legs carry the FX position: the bank paying GBP fixed against EUR float is short the GBP
notional and long the EUR one. The cross-currency basis is whatever float_spread says (0 = none).

``fx_spot`` -- FX deals, one row per deal
-----------------------------------------
Every deal is up to two exchanges of principal, written out from the BANK's side:

  deal_type   near exchange   far exchange
  FX_SPOT     yes             --
  FX_FWD      --              yes
  FX_SWAP     yes             yes (the opposite way)

Columns: deal_id, deal_type, ccy_pair (quote, e.g. EURUSD = USD per 1 EUR), trade_date,
near_date, near_buy_ccy, near_buy_amt, near_sell_ccy, near_sell_amt, near_rate,
far_date, far_buy_ccy, far_buy_amt, far_sell_ccy, far_sell_amt, far_rate.

Example (EUR bank hedging its USD deposits): an FX_SWAP that sells 80M USD for EUR at the
near date and buys the 80M USD back at the far date. After the near date only the far
exchange is outstanding: a USD receivable (long USD, closes the open position of the USD
deposits) and a EUR payable.

Each deal becomes one leg per currency (``fx_legs``), product_code 0100, so the leg
flows go through the same cash-flow / IRRBB machinery as the IRS legs, each on its own
currency's curve.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from labbank_common.bank_config import BankConfig, get_bank

FILE = "derivatives.xlsx"
SWAP_SHEET = "swap"
FX_SHEET = "fx_spot"

IRS_PRODUCT_CODE = "0000"
FX_PRODUCT_CODE = "0100"

SWAP_TYPES_PRICED = {"IRS", "CIRS"}
SWAP_TYPES_ONE_CCY = {"IRS", "OIS"}
SWAP_TYPES_TWO_CCY = {"CIRS"}
FX_DEAL_TYPES = {"FX_SPOT", "FX_FWD", "FX_SWAP"}

SWAP_COLUMNS = [
    "report_date", "swap_id", "swap_type", "pay_fixed", "notional", "currency",
    "start_date", "maturity_date", "fixed_rate", "fixed_pay_freq", "fixed_dc_conv",
    "fixed_b_day_conv", "float_rate_index", "float_pay_freq", "float_fixing_freq",
    "float_spread", "float_dc_conv", "float_b_day_conv", "disc_curve", "fwd_curve",
    "float_currency", "float_notional", "notional_exchange", "hedge_accounting",
]
HEDGE_ACCOUNTING = ("none", "fair_value", "cash_flow")
HEDGE_ACCOUNTING_DEFAULT = "fair_value"
FX_COLUMNS = [
    "report_date", "deal_id", "deal_type", "ccy_pair", "trade_date",
    "near_date", "near_buy_ccy", "near_buy_amt", "near_sell_ccy", "near_sell_amt", "near_rate",
    "far_date", "far_buy_ccy", "far_buy_amt", "far_sell_ccy", "far_sell_amt", "far_rate",
]


def input_path(bank: BankConfig | str | None = None) -> Path:
    b = bank if isinstance(bank, BankConfig) else get_bank(bank)
    return b.derivatives_dir() / FILE


def read_swaps(bank: BankConfig | str | None = None, path: Path | None = None) -> pd.DataFrame:
    """The ``swap`` sheet with defaults filled and the currency rules checked."""
    src = path or input_path(bank)
    df = pd.read_excel(src, sheet_name=SWAP_SHEET)
    if "hedge_accounting" in df.columns:     # "None" typed in Excel is a value here, not a missing cell
        raw = pd.read_excel(src, sheet_name=SWAP_SHEET, usecols=["hedge_accounting"], keep_default_na=False,
                            dtype=str)["hedge_accounting"].str.strip()
        df["hedge_accounting"] = raw.where(raw != "", pd.NA).to_numpy()
    for c in SWAP_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    if df.empty:
        return df[SWAP_COLUMNS]
    df["swap_type"] = df["swap_type"].fillna("IRS").astype(str).str.upper()
    df["pay_fixed"] = df["pay_fixed"].fillna(1).astype(int)
    df["float_spread"] = pd.to_numeric(df["float_spread"], errors="coerce").fillna(0.0)
    df["float_currency"] = df["float_currency"].fillna(df["currency"])
    raw_fl = pd.to_numeric(df["float_notional"], errors="coerce")
    no_fl = df["swap_type"].isin(SWAP_TYPES_TWO_CCY) & raw_fl.isna()
    if no_fl.any():      # an IRS default (same notional) is wrong in another currency
        raise ValueError(f"swap sheet: CIRS needs float_notional: {df.loc[no_fl, 'swap_id'].tolist()}")
    df["float_notional"] = raw_fl.fillna(df["notional"])
    df["notional_exchange"] = pd.to_numeric(df["notional_exchange"], errors="coerce").fillna(0).astype(int)
    df["hedge_accounting"] = (df["hedge_accounting"].astype(object).where(df["hedge_accounting"].notna(),
                              HEDGE_ACCOUNTING_DEFAULT).astype(str).str.strip().str.lower())
    if df["float_b_day_conv"].isna().any():
        df["float_b_day_conv"] = df["float_b_day_conv"].fillna(df["fixed_b_day_conv"])
    for c in ("report_date", "start_date", "maturity_date"):
        df[c] = pd.to_datetime(df[c])
    validate_swaps(df)
    return df[SWAP_COLUMNS]


def validate_swaps(df: pd.DataFrame) -> None:
    if "hedge_accounting" in df.columns:
        bad = set(df["hedge_accounting"].dropna()) - set(HEDGE_ACCOUNTING)
        if bad:
            raise ValueError(f"swap sheet: hedge_accounting must be one of {HEDGE_ACCOUNTING}, got {sorted(bad)}")
    unknown = set(df["swap_type"]) - SWAP_TYPES_ONE_CCY - SWAP_TYPES_TWO_CCY
    if unknown:
        raise ValueError(f"swap sheet: unknown swap_type {sorted(unknown)}")
    one = df["swap_type"].isin(SWAP_TYPES_ONE_CCY) & (df["currency"] != df["float_currency"])
    if one.any():
        raise ValueError(f"swap sheet: IRS/OIS must have one currency: {df.loc[one, 'swap_id'].tolist()}")
    two = df["swap_type"].isin(SWAP_TYPES_TWO_CCY) & (df["currency"] == df["float_currency"])
    if two.any():
        raise ValueError(f"swap sheet: CIRS needs two currencies: {df.loc[two, 'swap_id'].tolist()}")
    todo = ~df["swap_type"].isin(SWAP_TYPES_PRICED)
    if todo.any():
        raise NotImplementedError(
            f"swap sheet: {sorted(set(df.loc[todo, 'swap_type']))} rows "
            f"{df.loc[todo, 'swap_id'].tolist()} -- the input layout is ready, pricing is not "
            f"built yet (only {sorted(SWAP_TYPES_PRICED)})")


def read_fx_deals(bank: BankConfig | str | None = None, path: Path | None = None) -> pd.DataFrame:
    """The ``fx_spot`` sheet (empty frame with the columns if the bank has no FX deals)."""
    try:
        df = pd.read_excel(path or input_path(bank), sheet_name=FX_SHEET)
    except ValueError:                       # sheet missing
        df = pd.DataFrame(columns=FX_COLUMNS)
    for c in FX_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[FX_COLUMNS].copy()
    for c in ("report_date", "trade_date", "near_date", "far_date"):
        df[c] = pd.to_datetime(df[c])
    bad = set(df["deal_type"].dropna()) - FX_DEAL_TYPES
    if bad:
        raise ValueError(f"fx_spot sheet: unknown deal_type {sorted(bad)}")
    return df


def fx_exchanges(deals: pd.DataFrame) -> pd.DataFrame:
    """One row per currency amount exchanged: deal_id, date, currency, amount (+ received
    by the bank, - paid), leg ('near' / 'far')."""
    rows = []
    for d in deals.itertuples(index=False):
        for leg in ("near", "far"):
            date = getattr(d, f"{leg}_date")
            if pd.isna(date):
                continue
            rows.append((d.deal_id, date, getattr(d, f"{leg}_buy_ccy"), float(getattr(d, f"{leg}_buy_amt")), leg))
            rows.append((d.deal_id, date, getattr(d, f"{leg}_sell_ccy"), -float(getattr(d, f"{leg}_sell_amt")), leg))
    return pd.DataFrame(rows, columns=["deal_id", "date", "currency", "amount", "leg"])


def fx_legs(deals: pd.DataFrame, report_date) -> pd.DataFrame:
    """Outstanding position per (deal, currency): the exchanges still to settle after
    ``report_date``, written as one instrument per currency.

    A leg whose last exchange is a receipt is an asset (bs_side A), a payment a liability (L).
    When the near exchange is settled, the far exchange is a zero-coupon position:
      outstanding = |near amount| (what the bank gave / got at the near date),
      interest    = |far amount| - |near amount| (the swap points; zero on the currency
                    whose amount is the same at both dates).
    Columns: deal_id, deal_type, currency, bs_side, start_date, maturity_date, notional
    (far amount), outstanding, interest, near_amount (only when the near date is still
    ahead -- a forward-starting deal: paid at start, received at maturity or vice versa).
    """
    rd = pd.Timestamp(report_date)
    ex = fx_exchanges(deals)
    out = []
    for (deal_id, ccy), g in ex.groupby(["deal_id", "currency"], sort=False):
        deal = deals.loc[deals["deal_id"] == deal_id].iloc[0]
        g = g.sort_values("date")
        future = g[g["date"] > rd]
        if future.empty:
            continue
        last = future.iloc[-1]
        side = "A" if last["amount"] > 0 else "L"
        far_abs = abs(float(last["amount"]))
        near = g[g["leg"] == "near"]
        near_abs = abs(float(near["amount"].iloc[0])) if (len(g) > 1 and not near.empty) else far_abs
        start = near["date"].iloc[0] if not near.empty else deal["trade_date"]
        out.append({
            "deal_id": deal_id, "deal_type": deal["deal_type"], "currency": ccy, "bs_side": side,
            "start_date": pd.Timestamp(start), "maturity_date": pd.Timestamp(last["date"]),
            "notional": far_abs, "outstanding": near_abs, "interest": far_abs - near_abs,
            "near_amount": (float(near["amount"].iloc[0])
                            if (not near.empty and near["date"].iloc[0] > rd and len(future) > 1) else 0.0),
        })
    return pd.DataFrame(out, columns=["deal_id", "deal_type", "currency", "bs_side", "start_date",
                                      "maturity_date", "notional", "outstanding", "interest", "near_amount"])


# ── asset swaps: generated from the bond book ────────────────────────────────
# Sheet ``asset_swap`` (optional) -- one row per bond product to hedge:
#   product_code      bond product (e.g. 3000 fixed government bonds)
#   hedge_pct         share of that product's nominal that is asset-swapped (0..1); whole
#                     bonds are swapped (a bond is hedged or not), picked at random with the
#                     bank seed until the share is reached as closely as possible
#   float_rate_index  index of the received float leg (e.g. EUR_ASK_6M), float_freq (6M)
#   asw_spread_bp     float-leg spread in bp (blank = 0: a par asset swap of a bond whose
#                     coupon is the swap rate at issue)
# Each swap mirrors its bond: notional, start / maturity date, fixed rate = the bond coupon,
# fixed-leg frequency, day count and business-day rule = the bond's. The bank PAYS the bond
# coupon and receives the float leg: economically a floater, still Level 1 HQLA.
# The bonds exist only after balance generation, so ir_derivatives builds the swaps at run
# time and writes them to <output>/ir_derivatives/<bank>/asset_swaps.xlsx; read_swap_book()
# = manual swap sheet + that file (what the pipeline priced), used by the sandbox.

ASSET_SWAP_SHEET = "asset_swap"
ASSET_SWAP_FILE = "asset_swaps.xlsx"


def read_asset_swap_rules(bank: BankConfig | str | None = None, path: Path | None = None) -> pd.DataFrame:
    try:
        df = pd.read_excel(path or input_path(bank), sheet_name=ASSET_SWAP_SHEET)
    except ValueError:                                   # no such sheet: no asset swaps
        return pd.DataFrame()
    df = df.dropna(subset=["product_code"]).copy()
    if df.empty:
        return df
    df["product_code"] = df["product_code"].astype(int).astype(str)
    df["hedge_pct"] = pd.to_numeric(df["hedge_pct"], errors="coerce")
    if df["hedge_pct"].isna().any() or ((df["hedge_pct"] < 0) | (df["hedge_pct"] > 1)).any():
        raise ValueError("asset_swap sheet: hedge_pct must be in [0, 1]")
    if df["product_code"].duplicated().any():
        raise ValueError("asset_swap sheet: one row per product_code")
    for c in ("float_rate_index", "float_freq"):
        if df[c].isna().any():
            raise ValueError(f"asset_swap sheet: {c} is required")
    df["asw_spread_bp"] = pd.to_numeric(df.get("asw_spread_bp"), errors="coerce").fillna(0.0)
    return df


def build_asset_swaps(rules: pd.DataFrame, bonds: pd.DataFrame, report_date, seed: int) -> pd.DataFrame:
    """Swap rows (SWAP_COLUMNS layout) mirroring the bonds selected by ``rules``.

    bonds: schemat.financial_instruments rows (transaction_id, product_code, balance_amt,
    currency, start_date, maturity_date, client_rt, payment_freq, dc_conv, b_day_conv,
    disc_curve, fwd_curve).
    """
    if rules is None or rules.empty or bonds is None or bonds.empty:
        return pd.DataFrame(columns=SWAP_COLUMNS)
    rd = pd.Timestamp(report_date)
    b = bonds.copy()
    b["product_code"] = b["product_code"].astype(str)
    b["maturity_date"] = pd.to_datetime(b["maturity_date"])
    b = b[b["maturity_date"] > rd].sort_values("transaction_id").reset_index(drop=True)
    rng = pd.Series(range(len(b))).sample(frac=1.0, random_state=seed).to_numpy()
    rows = []
    for _, r in rules.iterrows():
        pb = b[b["product_code"] == r["product_code"]]
        if pb.empty:
            raise ValueError(f"asset_swap sheet: no live bonds of product {r['product_code']}")
        target = float(r["hedge_pct"]) * float(pb["balance_amt"].sum())
        cum, n = 0.0, 0
        mine = set(pb.index)
        for i in [i for i in rng if i in mine]:
            x = b.loc[i]
            if cum + x["balance_amt"] / 2.0 > target:     # closest whole-bond fit
                continue
            cum += float(x["balance_amt"]); n += 1
            rows.append({
                "report_date": rd, "swap_id": f"ASW{r['product_code']}_{n:03d}", "swap_type": "IRS",
                "pay_fixed": 1, "notional": float(x["balance_amt"]), "currency": x["currency"],
                "start_date": pd.Timestamp(x["start_date"]), "maturity_date": x["maturity_date"],
                "fixed_rate": float(x["client_rt"]), "fixed_pay_freq": x["payment_freq"],
                "fixed_dc_conv": x["dc_conv"], "fixed_b_day_conv": x["b_day_conv"],
                "float_rate_index": r["float_rate_index"], "float_pay_freq": r["float_freq"],
                "float_fixing_freq": r["float_freq"], "float_spread": float(r["asw_spread_bp"]) / 1e4,
                "float_dc_conv": "ACT/360", "float_b_day_conv": x["b_day_conv"],
                "disc_curve": x["disc_curve"], "fwd_curve": x["fwd_curve"],
                "float_currency": x["currency"], "float_notional": float(x["balance_amt"]),
                "notional_exchange": 0, "hedged_transaction_id": str(x["transaction_id"]),
            })
    out = pd.DataFrame(rows)
    if "hedge_accounting" not in out.columns:
        out["hedge_accounting"] = HEDGE_ACCOUNTING_DEFAULT
    if not out.empty:
        validate_swaps(out)
    return out


def asset_swap_output_path(bank: BankConfig | str | None = None) -> Path:
    """Path only: the folder is created by the writer (no empty output folders for banks
    without asset swaps)."""
    b = bank if isinstance(bank, BankConfig) else get_bank(bank)
    return b.output_dir("ir_derivatives", create=False) / ASSET_SWAP_FILE


def read_swap_book(bank: BankConfig | str | None = None, path: Path | None = None) -> pd.DataFrame:
    """The whole swap book the pipeline prices: the ``swap`` sheet + the generated asset
    swaps of the last ir_derivatives run (if any)."""
    manual = read_swaps(bank, path)
    gen_path = asset_swap_output_path(bank)
    if not gen_path.exists():
        return manual
    gen = pd.read_excel(gen_path)
    if gen.empty:
        return manual
    for c in ("report_date", "start_date", "maturity_date"):
        gen[c] = pd.to_datetime(gen[c])
    if "hedge_accounting" not in gen.columns:            # files written before the column existed
        gen["hedge_accounting"] = HEDGE_ACCOUNTING_DEFAULT
    return pd.concat([manual, gen.reindex(columns=SWAP_COLUMNS)], ignore_index=True)
