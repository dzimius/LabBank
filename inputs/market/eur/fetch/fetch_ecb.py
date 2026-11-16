"""Fetch ECB series used by build_eur_inputs.py into ../euribor_raw/.

  ecb_estr_eonia.csv  €STR (EST, 2019-10 ->) and EONIA (EON, 1999-2021), daily, percent
  ecb_aaa_spot.csv    ECB euro-area AAA government spot curve 1Y..10Y (YC, 2004 ->), daily
  ecb_fx.csv          ECB euro reference FX rates (EXR), daily, units of currency per 1 EUR

Run with the OpenBB/fetch venv (needs requests + truststore):
    venv_openbb/Scripts/python.exe inputs/market/eur/fetch/fetch_ecb.py
"""
import io
import os

import truststore; truststore.inject_into_ssl()   # Windows cert store (ECB API fails on certifi here)
import pandas as pd
import requests

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "euribor_raw")
B = "https://data-api.ecb.europa.eu/service/data/"


def get(key: str, name: str) -> pd.Series:
    r = requests.get(B + key, params={"format": "csvdata"}, timeout=180)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))[["TIME_PERIOD", "OBS_VALUE"]]
    return d.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": name}).set_index("date")[name]


if __name__ == "__main__":
    on = pd.concat([get("EST/B.EU000A2X2A25.WT", "estr"), get("EON/D.EONIA_TO.RATE", "eonia")], axis=1).sort_index()
    on.to_csv(os.path.join(RAW, "ecb_estr_eonia.csv"))
    yc = pd.concat([get(f"YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{t}", f"aaa_{t}")
                    for t in ["1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y"]], axis=1).sort_index()
    yc.to_csv(os.path.join(RAW, "ecb_aaa_spot.csv"))
    fx = pd.concat([get(f"EXR/D.{c}.EUR.SP00.A", c) for c in ["USD", "GBP", "PLN"]], axis=1).sort_index()
    fx.to_csv(os.path.join(RAW, "ecb_fx.csv"))
    print(f"overnight {on.index.min()}..{on.index.max()} | AAA {yc.index.min()}..{yc.index.max()}")
