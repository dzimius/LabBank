"""Tenor / OIS basis of the EUR curves -> curve_basis.csv (used by the add_data stage).

The bank's EUR curve (curve_input.xlsx: EURIBOR fixings + EUR IRS) is a EURIBOR-6M-type curve
(within +-3.6 bp of the clearing house's EUR.EURIBOR.6M, 1Y-10Y). The other curves of a
multi-curve setup are that curve plus a basis taken from the clearing house's settlement
curves of 2026-10-07 (../clearing_curves/):

  disc  = EUR.ESTR.1D       - EUR.EURIBOR.6M   -> EUR_disc_curve   (discounting, risk-free)
  1M    = EUR.EURIBOR.1M    - EUR.EURIBOR.6M   -> EUR_fwd_curve_1M (projection of 1M fixings)
  3M    = EUR.EURIBOR.3M    - EUR.EURIBOR.6M   -> EUR_fwd_curve_3M
  1Y    = EUR.EURIBOR.1Y    - EUR.EURIBOR.6M   -> EUR_fwd_curve_1Y
  (6M = the bank's curve itself: EUR_fwd_curve)

spread = difference of continuously compounded zero rates (ACT/365) at the same maturity in
days. One basis term structure for every curve date (a basis moves slowly; one clearing day
is available). Run from the project root:
    python inputs/market/eur/build_curve_basis.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "clearing_curves", "settlement_eur_usd_gbp_pln_20261007.csv")
BASE = "EUR.EURIBOR.6M"
CURVES = {"disc": "EUR.ESTR.1D", "1M": "EUR.EURIBOR.1M", "3M": "EUR.EURIBOR.3M", "1Y": "EUR.EURIBOR.1Y"}
GRID = np.unique(np.r_[1, 7, 14, np.arange(30, 361, 30), np.arange(365, 30 * 365 + 1, 91), 30 * 365])


def zero(df: pd.DataFrame, cid: str) -> pd.Series:
    c = df[(df["Curve ID"] == cid) & (df["Value Type"] == "S")]
    d = c["Maturity Offset"].to_numpy(float)
    z = -np.log(c["Value"].to_numpy(float)) / (d / 365.0)
    return pd.Series(np.interp(GRID, d, z), index=GRID)


def main() -> None:
    df = pd.read_csv(SRC)
    base = zero(df, BASE)
    out = pd.concat([pd.DataFrame({"curve": k, "n_days": GRID, "spread": (zero(df, cid) - base).to_numpy()})
                     for k, cid in CURVES.items()])
    out["source"] = "clearing settlement 2026-10-07, vs EUR.EURIBOR.6M"
    out.to_csv(os.path.join(HERE, "curve_basis.csv"), index=False)
    piv = out.pivot(index="n_days", columns="curve", values="spread") * 1e4
    print(piv.loc[[30, 90, 180, 360, 365 + 91 * 4, 365 + 91 * 16, 365 + 91 * 36]].round(1).to_string())


if __name__ == "__main__":
    main()
