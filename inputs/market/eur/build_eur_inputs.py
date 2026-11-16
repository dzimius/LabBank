"""Build the EUR market inputs (EUR counterpart of input/pln/curve_melt.ipynb).

Raw files (euribor_raw/)                         -> outputs (this folder)
  euribor_3m-1w-1m-6m-12m_*.csv  EURIBOR daily    -> fixing_input.xlsx  (EUR_ASK/BID_<tenor>, 2005..)
  ecb_estr_eonia.csv             €STR / EONIA     -> curve_input.xlsx   (daily quotes 1D..10Y)
  eur_irs_pdma.csv               EUR IRS (PDMA daily market reports, Bloomberg source)
  ecb_aaa_spot.csv               ECB AAA government spot curve (backfill only)
  ecb_fx.csv                     ECB euro reference FX rates -> fx_input.xlsx (units of currency per 1 EUR)

Rules
-----
1D        €STR from 2019-10-01, EONIA before (EONIA = €STR + 8.5 bp after the switch).
1M..1Y    EURIBOR fixings (12M = the 1Y tenor).
IRS       PDMA reports give 1Y,3Y,5Y,7Y,10Y,15Y,20Y,30Y on report dates. They are put on the
          EURIBOR business-day calendar and linearly interpolated in time (the PLN IRS panel is
          treated the same way); 2Y and 4Y are linear in tenor between their neighbours.
          Dates before the first PDMA report: ECB AAA government spot yield of the same tenor
          + the mean (IRS - AAA) spread over the first 12 months of overlap (labelled backfill).
ASK = BID (as for PLN: the book has no bid/ask spread on fixings).

Run from the project root:  python inputs/market/eur/build_eur_inputs.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "euribor_raw")
START = pd.Timestamp("2005-01-03")          # same history start as the PLN fixings
CURVE_IRS = ["2Y", "3Y", "4Y", "5Y", "7Y", "10Y"]
FIX_IRS = ["2Y", "5Y", "7Y", "10Y"]         # IRS tenors used as product rate indexes (10Y: fixed mortgages, 2Y: 24M deposits)
BACKFILL_WINDOW_DAYS = 365


def load_money_market() -> pd.DataFrame:
    """Daily 1D / 1M / 3M / 6M / 1Y (percent) on the EURIBOR business-day calendar."""
    f = sorted(glob.glob(os.path.join(RAW, "euribor_*.csv")))[-1]
    eb = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    eb = eb.rename(columns={"1m": "1M", "3m": "3M", "6m": "6M", "12m": "1Y"})[["1M", "3M", "6M", "1Y"]]
    on = pd.read_csv(os.path.join(RAW, "ecb_estr_eonia.csv"), parse_dates=["date"]).set_index("date")
    on_1d = on["estr"].combine_first(on["eonia"])          # €STR where it exists, EONIA before
    mm = eb.join(on_1d.rename("1D"), how="left")
    mm["1D"] = mm["1D"].ffill()                            # TARGET holidays where only EURIBOR fixed
    return mm[mm.index >= START].dropna()


def load_irs(bdays: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """Daily IRS panel on `bdays`; returns (panel, is_backfill flag)."""
    irs = pd.read_csv(os.path.join(RAW, "eur_irs_pdma.csv"), parse_dates=["date"]).set_index("date").sort_index()
    irs = irs[~irs.index.duplicated(keep="last")]
    # tenor interpolation for curve points the reports do not quote
    irs["2Y"] = (irs["1Y"] + irs["3Y"]) / 2.0
    irs["4Y"] = (irs["3Y"] + irs["5Y"]) / 2.0
    first, last = irs.index.min(), irs.index.max()
    daily = (irs.reindex(irs.index.union(bdays)).interpolate(method="time")
                .reindex(bdays))
    daily.loc[daily.index > last] = np.nan
    daily = daily.ffill()                                  # carry the last report forward (days only)

    # backfill before the first report: AAA government spot + mean IRS-AAA spread
    aaa = pd.read_csv(os.path.join(RAW, "ecb_aaa_spot.csv"), parse_dates=["date"]).set_index("date")
    aaa = aaa.rename(columns=lambda c: c.replace("aaa_", "")).reindex(bdays).ffill()
    window = (bdays >= first) & (bdays < first + pd.Timedelta(days=BACKFILL_WINDOW_DAYS))
    before = bdays < first
    for t in CURVE_IRS:
        spread = (daily.loc[window, t] - aaa.loc[window, t]).mean()
        daily.loc[before, t] = aaa.loc[before, t] + spread
        print(f"  backfill {t:>3}: AAA + {spread * 100:+.1f} bp (mean IRS-AAA, first {BACKFILL_WINDOW_DAYS}d of PDMA)")
    return daily[CURVE_IRS], pd.Series(before, index=bdays)


def build_fx_input() -> pd.DataFrame:
    """fx_input.xlsx: ECB euro reference rates, long format (date, currency, rate), rate = units
    of the currency per 1 EUR. Converts the bank's foreign-currency positions to EUR."""
    raw = pd.read_csv(os.path.join(RAW, "ecb_fx.csv"), parse_dates=["date"]).set_index("date")
    fx = (raw.loc[raw.index >= START].reset_index()
          .melt(id_vars="date", var_name="currency", value_name="rate").dropna())
    fx = fx.sort_values(["date", "currency"])[["date", "currency", "rate"]]
    fx.to_excel(os.path.join(HERE, "fx_input.xlsx"), index=False)
    return fx


def main() -> None:
    mm = load_money_market()
    irs, backfill = load_irs(mm.index)
    panel = mm.join(irs)
    print(f"panel {panel.index.min().date()} .. {panel.index.max().date()}  rows={len(panel)}  "
          f"IRS backfilled before {panel.index[~backfill.values].min().date()}")

    # fixing_input.xlsx  -- same layout as input/pln/fixing_input.xlsx
    fx_cols = ["1D", "1M", "3M", "6M", "1Y"] + FIX_IRS
    fx = panel[fx_cols].reset_index().melt(id_vars="date", var_name="tenor", value_name="rate")
    fx = pd.concat([fx.assign(rate_index="EUR_ASK_" + fx.tenor), fx.assign(rate_index="EUR_BID_" + fx.tenor)])
    fx["currency"] = "EUR"
    fx = fx[["date", "currency", "rate_index", "rate", "tenor"]].sort_values(["date", "rate_index"])
    fx.to_excel(os.path.join(HERE, "fixing_input.xlsx"), index=False)

    # curve_input.xlsx  -- same layout as input/pln/curve_input.xlsx (ir_type labels EUR-specific)
    names = {"1D": "ESTR", "1M": "EUR1M", "3M": "EUR3M", "6M": "EUR6M", "1Y": "EUR1Y",
             **{t: f"IRS EUR {t}" for t in CURVE_IRS}}
    cv = panel.reset_index().melt(id_vars="date", var_name="tenor", value_name="rate")
    cv["ir_type"] = cv.tenor.map(names)
    cv["currency"] = "EUR"
    cv = cv[["date", "currency", "ir_type", "rate", "tenor"]].sort_values(["date", "tenor"])
    cv.to_excel(os.path.join(HERE, "curve_input.xlsx"), index=False)
    fxr = build_fx_input()
    print(f"fixing_input.xlsx {len(fx):,} rows | curve_input.xlsx {len(cv):,} rows | fx_input.xlsx {len(fxr):,} rows")


if __name__ == "__main__":
    main()
