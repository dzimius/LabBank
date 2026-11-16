"""Credit spread risk in the banking book (CSRBB) of the bank's securities.

EBA/GL/2022/14 asks for CSRBB under both the economic value and the earnings view. A security is in
scope when bs_structure gives it a cqs (credit quality step, 0 = sovereign). Spread shocks per cqs come
from sheet csrbb_shocks of bank_data.xlsx (moderate_bp, severe_bp): an assumption of the bank.

  market value   PV of the contractual flows on the discount curve plus the security's market
                 spread (mv_spread_bp): sum CF x d_f x exp(-s x t)
  economic view  change in that PV when s widens by the shock
  CET1 view      the part of it on FVOCI holdings (accounting = FVOCI) hits own funds at once
  earnings view  principal maturing inside 12M is reinvested at the wider spread for the rest of
                 the year: a gain, and the only earnings effect, since a coupon does not move
  CS01           PV change for +1 bp

Runs only for a bank with extended_reports; writes results.csrbb and runs/<bank>/irrbb_calc/
csrbb_results.xlsx.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from labbank_common import catalogue  # noqa: E402
from labbank_common.bank_config import get_bank, get_engine  # noqa: E402

SHEET = "csrbb_shocks"


def pv(cf: np.ndarray, df: np.ndarray, t: np.ndarray, spread_bp: float) -> float:
    return float(np.sum(cf * df * np.exp(-spread_bp / 1e4 * t)))


def reinvestment_gain(capital: np.ndarray, t: np.ndarray, shock_bp: float) -> float:
    """12M earnings effect: principal repaid at t < 1 earns the wider spread for 1 - t years."""
    m = t < 1.0
    return float(np.sum(capital[m] * shock_bp / 1e4 * (1.0 - t[m])))


def product_rows(cf: pd.DataFrame, spec: pd.Series, shocks: pd.DataFrame) -> list[dict]:
    """One row per severity for one security product. cf columns: t, total, capital, d_f."""
    c, d, t = cf["total"].to_numpy(float), cf["d_f"].to_numpy(float), cf["t"].to_numpy(float)
    s0 = 0.0 if pd.isna(spec.get("mv_spread_bp")) else float(spec["mv_spread_bp"])
    base = pv(c, d, t, s0)
    cs01 = pv(c, d, t, s0 + 1.0) - base
    fvoci = str(spec.get("accounting", "")).strip().upper() == "FVOCI"
    row = shocks[shocks["cqs"] == int(spec["cqs"])]
    if row.empty:
        raise ValueError(f"csrbb_shocks: no row for cqs {spec['cqs']} (product {spec['product_code']})")
    out = []
    for sev in ("moderate", "severe"):
        bp = float(row.iloc[0][f"{sev}_bp"])
        delta = pv(c, d, t, s0 + bp) - base
        out.append({
            "product_code": str(int(spec["product_code"])), "own_name": spec["own_name"],
            "cqs": int(spec["cqs"]), "accounting": spec.get("accounting"), "severity": sev,
            "shock_bp": bp, "market_value": base, "cs01": cs01,
            "spread_duration": -cs01 / base * 1e4 if base else np.nan,
            "delta_ev": delta, "delta_cet1": delta if fvoci else 0.0,
            "delta_nii_12m": reinvestment_gain(cf["capital"].to_numpy(float), t, bp),
        })
    return out


def main() -> None:
    bank = get_bank()
    if not bank.extended_reports:
        print(f"{bank.bank_id}: no extended reports, CSRBB skipped")
        return
    path = bank.balance_dir() / "bank_data.xlsx"
    bs = catalogue.read_bs_structure(path)
    spec = bs[(bs["bs_side"] == "A") & bs["cqs"].notna() & (bs["bs_percentage"] > 0)]
    shocks = pd.read_excel(path, sheet_name=SHEET)
    rd = pd.Timestamp(bank.report_date)
    eng = get_engine(bank)
    cf = pd.read_sql(text("""SELECT CAST(product_code AS VARCHAR(4)) AS product_code, cf_end_dt,
                                    con_total_pmt AS total, con_capital_pmt AS capital, d_f
                             FROM cf.products WHERE bs_side = 'A' AND cf_end_dt > :rd"""), eng, params={"rd": rd})
    cf["t"] = (pd.to_datetime(cf["cf_end_dt"]) - rd).dt.days / 365.0
    cf[["total", "capital", "d_f"]] = cf[["total", "capital", "d_f"]].fillna(0.0)
    rows = []
    for _, s in spec.iterrows():
        g = cf[cf["product_code"] == str(int(s["product_code"]))]
        if g.empty:
            raise ValueError(f"CSRBB: product {s['product_code']} has no cash flows in cf.products")
        rows += product_rows(g, s, shocks)
    res = pd.DataFrame(rows)
    tot = (res.groupby("severity", sort=False)[["market_value", "cs01", "delta_ev", "delta_cet1", "delta_nii_12m"]]
           .sum().reset_index())
    tier1 = float(pd.read_sql(text("SELECT SUM(balance_amt) AS t1 FROM schemat.equity WHERE report_date = :rd"),
                              eng, params={"rd": rd})["t1"].iloc[0])
    tot["delta_ev_pct_t1"] = tot["delta_ev"] / tier1 * 100.0
    tot["delta_cet1_pct_t1"] = tot["delta_cet1"] / tier1 * 100.0
    print(res[["own_name", "severity", "shock_bp", "market_value", "spread_duration", "delta_ev", "delta_cet1",
               "delta_nii_12m"]].to_string(index=False))
    print(tot.to_string(index=False))
    from labbank_common import run_store
    run_store.replace_run(eng, res, "csrbb", "results", run_store.current_run(bank))
    res.insert(0, "report_date", rd)
    with pd.ExcelWriter(bank.output_dir("irrbb_calc") / "csrbb_results.xlsx") as w:
        res.to_excel(w, sheet_name="by_product", index=False)
        tot.to_excel(w, sheet_name="total", index=False)


if __name__ == "__main__":
    main()
