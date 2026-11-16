"""Accounting view of the book at one date: carrying amounts, market values, OCI and the
EBA-scenario table ΔEVE / ΔOCI / ΔP&L / ΔCET1 (docs/07, step 1).

Market value
------------
Every value comes from the IRRBB EVE cash flows, so it is the EVE engine's pricing:

  securities   cash-flow level rows of cf.eve_*_scenario(s) (source fin_inst), each discounted
               at the scenario curve AND the security's market spread mv_spread_bp:
               sum pv_row x exp(-s x t). Floating coupons are already re-projected per scenario.
  swaps, FX    irrbb.eve_results per leg (risk-free, no spread), legs summed per swap / deal
  loans, deposits and issued bonds (AC)
               irrbb.eve_results 'base' per product: a memo fair value on the IRRBB basis

EVE prices are dirty (the current coupon in full). The clean value takes off the interest accrued
from the period start to the report date (cf.products, cf.ir_swap_beh). Foreign currencies at the
report-date rate.

Carrying amount and equity
--------------------------
The generated book carries every position at its nominal. Purchase price = 100 (bonds are bought at
issue, schemat.financial_instruments.purchase_price), so amortised cost = nominal. The accounting
balance sheet adds to the generated one:

  FVOCI securities   clean MV - nominal           asset, and the OCI reserve (equity)
  FVTPL securities   clean MV - nominal           asset, and retained earnings (past P&L)
  swaps, FX deals    clean MV                     derivative asset (MV > 0) / liability (MV < 0)
    hedge_accounting none        -> retained earnings (past P&L)
                     fair_value  -> FV hedge adjustment of the hedged items, -MV (no equity effect;
                                    designated at trade, when MV was 0, so the cumulative change = MV)
                     cash_flow   -> cash-flow hedge reserve (OCI)

A = L + E holds by construction; the script checks it. Credit allowances: the generated balances are
read as net of the 12M ECL stock (shown as a memo: PD x LGD x balance), so equity is unchanged by it.
No deferred tax on the OCI (docs/07 §3.1).

Scenario table
--------------
Per EBA scenario, instantaneous shock, the spread held:
  ΔEVE          results.irrbb_report (the IRRBB number)
  ΔOCI          FVOCI securities' change
  ΔP&L          FVTPL securities + swaps without hedge accounting + FX deals (pre-tax)
  ΔCFH reserve  cash-flow hedges (OCI, filtered out of CET1)
  hidden AC     change of AC securities: in EVE, not in capital (the "SVB" number)
  ΔCET1         ΔOCI + ΔP&L, pre-tax; ΔCET1 after tax uses CIT on both (as if deferred tax applied)

Writes acct.positions, acct.balance_sheet, acct.scenario_impact (history kept per report date
and run, labbank_common/run_store.py) and runs/<bank>/accounting/accounting_valuation.xlsx.

Usage: python accounting/python_code/acct_valuation_workflow.py [--bank eur]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acct_common as ac  # noqa: E402
from labbank_common import fx  # noqa: E402
from labbank_common.bank_config import get_bank, get_engine  # noqa: E402

SCEN_TABLE = {"base": "cf.eve_base_scenario", "par_up": "cf.eve_par_scenarios", "par_dn": "cf.eve_par_scenarios",
              "sr_up": "cf.eve_short_scenarios", "sr_dn": "cf.eve_short_scenarios",
              "steep": "cf.eve_step_flat_scenarios", "flat": "cf.eve_step_flat_scenarios"}
SCENARIOS = ("base",) + ac.EBA6


def _accrued(start, end, amount, rd) -> np.ndarray:
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    tot = (e - s).dt.days.to_numpy(float)
    el = (rd - s).dt.days.to_numpy(float)
    frac = np.divide(el, tot, out=np.zeros_like(el), where=tot > 0)
    inside = (s < rd) & (e > rd)
    return np.where(inside.to_numpy(), frac * np.asarray(amount, float), 0.0)


def security_values(eng, rd, cls: pd.DataFrame, to_home: dict) -> pd.DataFrame:
    """(scenario_id, product_code, mv_dirty) per security product, home currency."""
    sec = cls[cls["is_security"]]
    codes = [str(c) for c in sec["product_code"]]
    spread = dict(zip(sec["product_code"].astype(str), sec["mv_spread_bp"]))
    frames = []
    for tbl in sorted(set(SCEN_TABLE.values())):
        q = text(f"""SELECT scenario_id, CAST(product_code AS VARCHAR(4)) AS product_code, currency, year_frac,
                            pv_capital + pv_interest AS pv
                     FROM {tbl} WHERE source = 'fin_inst' AND bs_side = 'A' AND report_date = :rd""")
        frames.append(pd.read_sql(q, eng, params={"rd": rd.date()}))
    cf = pd.concat(frames, ignore_index=True)
    cf = cf[cf["product_code"].isin(codes) & cf["scenario_id"].isin(SCENARIOS)].copy()
    s = cf["product_code"].map(spread).astype(float).to_numpy()
    cf["mv"] = cf["pv"] * np.exp(-s / 1e4 * cf["year_frac"].clip(lower=0.0)) * cf["currency"].map(to_home)
    cf["pv_rf"] = cf["pv"] * cf["currency"].map(to_home)
    return cf.groupby(["scenario_id", "product_code"], as_index=False)[["mv", "pv_rf"]].sum()


def security_accrued(eng, rd, codes: list[str], to_home: dict) -> pd.Series:
    q = text("""SELECT CAST(product_code AS VARCHAR(4)) AS product_code, currency, cf_start_dt, cf_end_dt,
                       con_interest_pmt FROM cf.products
                WHERE bs_side = 'A' AND cf_start_dt < :rd AND cf_end_dt > :rd""")
    cf = pd.read_sql(q, eng, params={"rd": rd.date()})
    cf = cf[cf["product_code"].isin(codes)]
    if cf.empty:
        return pd.Series(dtype=float)
    cf["acc"] = _accrued(cf["cf_start_dt"], cf["cf_end_dt"], cf["con_interest_pmt"].fillna(0.0), rd) \
        * cf["currency"].map(to_home)
    return cf.groupby("product_code")["acc"].sum()


def derivative_values(eng, rd, to_home: dict) -> pd.DataFrame:
    """(scenario_id, swap_id, product_code, mv_dirty) per swap / FX deal; accrued per swap in attrs."""
    legs = pd.read_sql(text("""SELECT CAST(product_code AS VARCHAR(4)) AS product_code, swap_id, schedule_id, bs_side
                               FROM schemat.ir_swaps WHERE report_date = :rd"""), eng, params={"rd": rd.date()})
    if legs.empty:
        return pd.DataFrame(columns=["scenario_id", "swap_id", "product_code", "mv"])
    ev = pd.read_sql(text("""SELECT scenario_id, CAST(product_code AS VARCHAR(4)) AS product_code, schedule_id,
                                    currency, pv_total FROM irrbb.eve_results
                             WHERE report_date = :rd AND product_code IN ('0000', '0100')"""),
                     eng, params={"rd": rd.date()})
    ev = ev[ev["scenario_id"].isin(SCENARIOS)]
    ev["schedule_id"] = ev["schedule_id"].astype(int)
    legs["schedule_id"] = legs["schedule_id"].astype(int)
    m = ev.merge(legs, on=["product_code", "schedule_id"], how="left")
    if m["swap_id"].isna().any():
        raise ValueError(f"eve_results swap legs without a swap: {m.loc[m['swap_id'].isna(), 'schedule_id'].unique()}")
    m["mv"] = m["pv_total"] * m["currency"].map(to_home)
    out = m.groupby(["scenario_id", "swap_id", "product_code"], as_index=False)["mv"].sum()
    # accrued net coupon (receive +, pay -) from the swap cash flows
    cf = pd.read_sql(text("""SELECT schedule_id, bs_side, cf_start_dt, cf_end_dt, int_pmt FROM cf.ir_swap_beh
                             WHERE cf_start_dt < :rd AND cf_end_dt > :rd"""), eng, params={"rd": rd.date()})
    cf["schedule_id"] = cf["schedule_id"].astype(int)
    cf = cf.merge(legs.drop(columns="bs_side"), on="schedule_id", how="inner")
    cur = pd.read_sql(text("SELECT schedule_id, currency FROM sched.ir_swaps"), eng) \
        if "currency" in pd.read_sql(text("SELECT TOP 0 * FROM sched.ir_swaps"), eng).columns else None
    if cur is not None:
        cf = cf.merge(cur.astype({"schedule_id": int}).drop_duplicates("schedule_id"), on="schedule_id", how="left")
    else:
        cf["currency"] = None
    sign = np.where(cf["bs_side"] == "A", 1.0, -1.0)
    k = cf["currency"].map(to_home).fillna(1.0).to_numpy(float)
    cf["acc"] = sign * _accrued(cf["cf_start_dt"], cf["cf_end_dt"], cf["int_pmt"].fillna(0.0), rd) * k
    out.attrs["accrued"] = cf.groupby("swap_id")["acc"].sum()
    return out


def run(bank_id: str | None = None, write: bool = True) -> dict:
    bank = get_bank(bank_id)
    os.environ["LABBANK_BANK"] = bank.bank_id
    eng = get_engine(bank)
    rd = pd.Timestamp(bank.report_date)
    to_home = fx.to_home(bank, rd)
    fin = ac.fin_params(bank)
    cls = ac.product_classes(bank)

    # ── nominal book ─────────────────────────────────────────────────────────────────────
    tx = pd.read_sql(text("""SELECT product_code, bs_side, currency, SUM(balance_amt) AS nominal
                             FROM dbo.transactions WHERE report_date = :rd
                             GROUP BY product_code, bs_side, currency"""), eng, params={"rd": rd.date()})
    tx["nominal"] = tx["nominal"] * tx["currency"].map(to_home)
    tx["product_code"] = tx["product_code"].astype(int)
    tx["bs_side"] = np.where(tx["bs_side"] == "A", "A", np.where(tx["bs_side"] == "E", "E", "L"))
    pos = tx.groupby(["product_code", "bs_side"], as_index=False)["nominal"].sum()
    pos = pos.merge(cls, on=["product_code", "bs_side"], how="left")
    pos["accounting"] = pos["accounting"].fillna("AC")
    pos["is_security"] = pos["is_security"].fillna(False).astype(bool)
    tier1 = float(pos.loc[pos["bs_side"] == "E", "nominal"].sum())

    # ── securities ───────────────────────────────────────────────────────────────────────
    sv = security_values(eng, rd, cls, to_home)
    codes = sv["product_code"].unique().tolist()
    acc = security_accrued(eng, rd, codes, to_home)
    base = sv[sv["scenario_id"] == "base"].set_index("product_code")
    pc = pos["product_code"].astype(str)
    pos["accrued"] = pc.map(acc).fillna(0.0).where(pos["is_security"], 0.0)
    pos["mv_dirty"] = pc.map(base["mv"]).where(pos["is_security"])
    pos["mv_clean"] = pos["mv_dirty"] - pos["accrued"]
    pos["unrealised"] = (pos["mv_clean"] - pos["nominal"]).where(pos["is_security"])
    miss = pos["is_security"] & pos["mv_dirty"].isna() & (pos["nominal"].abs() > 0)
    if miss.any():
        raise ValueError(f"no EVE cash flows for securities {pos.loc[miss, 'product_code'].tolist()}")

    # memo fair value of the rest on the IRRBB basis (risk-free, behavioural flows)
    ev = pd.read_sql(text("""SELECT CAST(product_code AS VARCHAR(4)) AS product_code, bs_side, currency,
                                    SUM(pv_total) AS pv FROM irrbb.eve_results
                             WHERE report_date = :rd AND scenario_id = 'base'
                               AND product_code NOT IN ('0000', '0100')
                             GROUP BY product_code, bs_side, currency"""), eng, params={"rd": rd.date()})
    ev["pv"] = ev["pv"] * ev["currency"].map(to_home)
    ev = ev.assign(product_code=ev["product_code"].astype(int)).groupby(["product_code", "bs_side"])["pv"].sum().abs()
    key = pd.MultiIndex.from_frame(pos[["product_code", "bs_side"]])
    pos["fv_memo_irrbb"] = ev.reindex(key).to_numpy()
    pos["oci_reserve"] = np.where(pos["accounting"] == "FVOCI", pos["unrealised"].fillna(0.0), 0.0)
    pos["fvtpl_reval"] = np.where(pos["accounting"] == "FVTPL", pos["unrealised"].fillna(0.0), 0.0)
    pos["carrying"] = pos["nominal"] + pos["oci_reserve"] + pos["fvtpl_reval"]
    pd_ = pd.to_numeric(pos["PD"], errors="coerce").fillna(0.0)
    lgd = pd.to_numeric(pos["LGD"], errors="coerce").fillna(0.0)
    pos["ecl_12m_memo"] = np.where(pos["bs_side"] == "A", pd_ * lgd * pos["nominal"], 0.0)

    # ── derivatives ──────────────────────────────────────────────────────────────────────
    dv = derivative_values(eng, rd, to_home)
    flags = ac.swap_flags(bank).set_index("swap_id")["hedge_accounting"]
    der = dv[dv["scenario_id"] == "base"].copy()
    der["accrued"] = der["swap_id"].map(dv.attrs.get("accrued", pd.Series(dtype=float))).fillna(0.0)
    der["mv_clean"] = der["mv"] - der["accrued"]
    der["hedge_accounting"] = np.where(der["product_code"] == ac.FX_CODE, "none",
                                       der["swap_id"].map(flags).fillna("fair_value"))
    unknown = (der["product_code"] == ac.IRS_CODE) & ~der["swap_id"].isin(flags.index)
    if unknown.any():
        print(f"  [warn] swaps not in derivatives.xlsx (default fair_value): {der.loc[unknown, 'swap_id'].tolist()}")
    ha = der["hedge_accounting"]
    deriv_asset = float(der.loc[der["mv_clean"] > 0, "mv_clean"].sum())
    deriv_liab = float(-der.loc[der["mv_clean"] < 0, "mv_clean"].sum())
    fvh_adj = float(-der.loc[ha == "fair_value", "mv_clean"].sum())        # hedged items' adjustment
    cfh_res = float(der.loc[ha == "cash_flow", "mv_clean"].sum())
    deriv_pl = float(der.loc[ha == "none", "mv_clean"].sum())

    # ── accounting balance sheet ─────────────────────────────────────────────────────────
    oci = float(pos["oci_reserve"].sum())
    sec_pl = float(pos["fvtpl_reval"].sum())
    assets_nom = float(pos.loc[pos["bs_side"] == "A", "nominal"].sum())
    liab_nom = float(pos.loc[pos["bs_side"] == "L", "nominal"].sum())
    lines = [
        ("A", "loans, cash, AC securities (nominal)", float(pos.loc[(pos.bs_side == "A") & (pos.accounting == "AC"), "nominal"].sum())),
        ("A", "FVOCI securities (clean market value)", float(pos.loc[pos.accounting == "FVOCI", "carrying"].sum())),
        ("A", "FVTPL securities (clean market value)", float(pos.loc[pos.accounting == "FVTPL", "carrying"].sum())),
        ("A", "derivative assets", deriv_asset),
        ("A", "fair-value hedge adjustment of hedged items", fvh_adj),
        ("L", "liabilities (nominal)", liab_nom),
        ("L", "derivative liabilities", deriv_liab),
        ("E", "share capital, reserves, retained earnings (book)", tier1),
        ("E", "retained earnings: past FVTPL revaluation", sec_pl + deriv_pl),
        ("E", "OCI reserve (FVOCI)", oci),
        ("E", "cash-flow hedge reserve (OCI)", cfh_res),
    ]
    bsheet = pd.DataFrame(lines, columns=["side", "line", "amount"])
    a = bsheet.loc[bsheet.side == "A", "amount"].sum()
    le = bsheet.loc[bsheet.side != "A", "amount"].sum()
    book_gap = assets_nom - liab_nom - tier1          # the generated book itself (FX rounding in a multi-ccy book)
    gap = a - le - book_gap                           # the accounting adjustments must net to zero
    if abs(gap) > 1.0:
        raise AssertionError(f"accounting balance sheet does not close: A - (L + E) = {gap:,.2f}")
    cet1 = tier1 + oci + sec_pl + deriv_pl          # CFH reserve filtered (CRR art. 33(1)(a))

    # ── scenario impact ──────────────────────────────────────────────────────────────────
    rep = pd.read_sql(text("SELECT currency, scenario_id, delta_eve, sot_eve_pct FROM results.irrbb_report "
                           "WHERE report_date = :rd"), eng, params={"rd": rd.date()})
    tot_ccy = "ALL" if (rep["currency"] == "ALL").any() else bank.market.ccy
    rep = rep[rep["currency"] == tot_ccy].set_index("scenario_id")
    secv = sv.pivot(index="product_code", columns="scenario_id", values="mv")
    dvv = dv.pivot_table(index="swap_id", columns="scenario_id", values="mv", aggfunc="sum")
    cl = cls.assign(pc=cls["product_code"].astype(str)).drop_duplicates("pc").set_index("pc")["accounting"]
    sec_cls = secv.index.map(cl).fillna("AC")
    d_ha = der.set_index("swap_id")["hedge_accounting"].reindex(dvv.index).fillna("fair_value")
    cit = fin["CIT"]
    rows = []
    for s in ac.EBA6:
        dsec = secv[s] - secv["base"]
        dder = dvv[s] - dvv["base"] if s in dvv.columns else pd.Series(0.0, index=dvv.index)
        d_oci = float(dsec[sec_cls == "FVOCI"].sum())
        d_pl = float(dsec[sec_cls == "FVTPL"].sum() + dder[d_ha == "none"].sum())
        d_fvh_swap = float(dder[d_ha == "fair_value"].sum())
        rows.append(dict(
            scenario_id=s, delta_eve=float(rep.loc[s, "delta_eve"]), delta_eve_pct_t1=float(rep.loc[s, "sot_eve_pct"]),
            delta_oci=d_oci, delta_pl=d_pl, delta_cfh_reserve=float(dder[d_ha == "cash_flow"].sum()),
            fv_hedge_swaps=d_fvh_swap, fv_hedge_adjustment=-d_fvh_swap,
            hidden_ac_securities=float(dsec[sec_cls == "AC"].sum()),
            delta_cet1=d_oci + d_pl, delta_cet1_after_tax=(d_oci + d_pl) * (1.0 - cit),
        ))
    imp = pd.DataFrame(rows)
    for c in ("delta_oci", "delta_pl", "delta_cet1", "hidden_ac_securities"):
        imp[f"{c}_pct_cet1"] = imp[c] / cet1 * 100.0

    summary = dict(bank=bank.bank_id, report_date=rd.date(), total_assets_nominal=assets_nom,
                   total_assets_accounting=float(a), book_equity=tier1, oci_reserve=oci,
                   cfh_reserve=cfh_res, fvtpl_reval=sec_pl + deriv_pl, cet1_accounting=cet1,
                   ecl_12m_stock_memo=float(pos["ecl_12m_memo"].sum()), derivatives_net_mv=float(der["mv_clean"].sum()),
                   fv_hedge_adjustment=fvh_adj, balance_check=float(gap), book_gap_memo=float(book_gap), CIT=cit,
                   dividend_payout=fin["dividend_payout"])
    pos_out = pos[["product_code", "bs_side", "own_name", "currency", "accounting", "portfolio", "nominal",
                   "accrued", "mv_dirty", "mv_clean", "carrying", "unrealised", "oci_reserve", "fvtpl_reval",
                   "fv_memo_irrbb", "ecl_12m_memo"]].copy()
    der_out = der.rename(columns={"mv": "mv_dirty"})[["swap_id", "product_code", "hedge_accounting", "accrued",
                                                      "mv_dirty", "mv_clean"]]
    if write:
        from labbank_common import run_store
        out_dir = bank.output_dir("accounting")
        with pd.ExcelWriter(out_dir / "accounting_valuation.xlsx") as w:
            pd.Series(summary).rename("value").rename_axis("item").reset_index().to_excel(w, sheet_name="summary", index=False)
            bsheet.to_excel(w, sheet_name="balance_sheet", index=False)
            pos_out.to_excel(w, sheet_name="positions", index=False)
            der_out.to_excel(w, sheet_name="derivatives", index=False)
            imp.to_excel(w, sheet_name="scenario_impact", index=False)
        run_store.replace_run(eng, pos_out.assign(kind="product"), "positions", "acct")
        run_store.replace_run(eng, der_out, "derivatives", "acct")
        run_store.replace_run(eng, bsheet, "balance_sheet", "acct")
        run_store.replace_run(eng, imp, "scenario_impact", "acct")
        print(f"Saved -> {out_dir / 'accounting_valuation.xlsx'}")
    return dict(summary=summary, balance_sheet=bsheet, positions=pos_out, derivatives=der_out, impact=imp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=None)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    r = run(a.bank, write=not a.no_write)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print({k: (round(v / 1e6, 2) if isinstance(v, float) and abs(v) > 1e3 else v) for k, v in r["summary"].items()})
    print(r["balance_sheet"].assign(amount=lambda d: (d.amount / 1e6).round(2)).to_string(index=False))
    p = r["positions"]
    print((p[p["accounting"] != "AC"].set_index("own_name")[["nominal", "mv_clean", "oci_reserve", "fvtpl_reval"]]
           / 1e6).round(2).to_string())
    print(r["derivatives"].assign(mv_clean=lambda d: (d.mv_clean / 1e6).round(2))[
        ["swap_id", "hedge_accounting", "mv_clean"]].to_string(index=False))
    imp = r["impact"].set_index("scenario_id")
    print((imp[["delta_eve", "delta_oci", "delta_pl", "hidden_ac_securities", "delta_cet1"]] / 1e6).round(2)
          .join(imp[["delta_eve_pct_t1", "delta_cet1_pct_cet1"]].round(2)).to_string())


if __name__ == "__main__":
    main()
