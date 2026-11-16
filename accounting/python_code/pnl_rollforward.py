"""Monthly P&L, equity and capital over the one-year roll-forward (docs/07, step 2).

Deterministic, base bank -> rolled bank (eur -> eur_y1), month by month inside the window.


Inputs
------
  balances   visual_rep/rollforward_report.monthly_balances: the plan path per product and month
             (existing book aged + new business to the growth plan), the same path the roll-forward
             books into the rolled bank
  interest   visual_rep/dynamic_nii: monthly interest per product on that path, on the IRRBB NII engine
             (one engine: the year's interest here = the plan-year NII), swaps' net coupons included
  values     acct_pricing: securities and swaps at every month-end on the path's curve (mkt.curves of
             the rolled bank)
  flags      acct_common: accounting / portfolio per product, hedge_accounting per swap, CIT,
             dividend payout and month (bank_data fin_data of the BASE bank)

P&L lines per month (home currency)
-----------------------------------
  interest             dynamic_nii, per product (0000 = swap net coupons)
  interest_cash_plug   opening cash surplus x central-bank deposit rate (3510), or deficit x interbank
                       borrowing rate (7900 L): the profit the plan does not use
  fees                 fee_unit_rate x average balance x days / 365
  credit_losses        12M ECL of stage 1: -(write-offs PD x LGD x average balance x days / 365 + change of
                       the ECL stock PD x LGD x balance); the plan's balances are net of defaults, so a
                       write-off is a cash loss replaced by new business
  opex                 -OPEX_RATE (0.65%) x average balance of every non-equity, non-derivative product
  acq_cost             -acq_cost_rate x growth of the product's balance in the month (as in the EP)
  fvtpl_derivatives    change of the clean MV of swaps without hedge accounting and of FX deals
  fv_hedge_swaps       change of the clean MV of fair-value-hedging swaps ...
  fv_hedge_adjustment  ... and the hedged items' adjustment, -(MV - MV at designation): net 0 (100%
                       effective), except a new swap's day-one value
  fvtpl_securities     change of FVTPL securities' revaluation, realised_securities: sales (FVOCI
                       reserve recycled, FVTPL realised) when the plan sells bonds down to its target
  tax                  -CIT x profit before tax, cumulative in the calendar year, floored at 0 (no loss
                       carry-forward, no deferred tax)

Equity, cash and capital
------------------------
  retained earnings(t) = retained(t-1) + net profit(t) - dividends(t)            (monthly, user decision)
  OCI reserve(t)       = FVOCI securities' clean MV - amortised cost (bought at par: nominal)
  dividend             payout x last financial year's profit, paid in dividend_month. The base book's
                       financial year before the window is not modelled: its profit = the modelled
                       Oct-Dec months x 4
  cash(t)              = cash(t-1) + net profit - non-cash P&L - dividends - (plan equity change):
                       the result's cash that the plan's balance sheet does not use. Surplus at the
                       central bank, deficit borrowed interbank (the only two plugs, docs/07 §3.2)
  CET1(t)              = share capital + reserves + retained earnings + OCI reserve - foreseeable
                       dividends (payout x the unpaid / year-to-date profit); cash-flow hedge reserve
                       filtered. RWA = rwa_weight x balance (on-balance credit risk, as the optimiser)
Checks per month: A = L + E (cash from the ledger, not as a balance), Δequity = profit - dividends + ΔOCI,
interest of the year = dynamic_nii's moving-balance-sheet NII.

The plan's own equity (scaled to the growth plan) is kept for comparison: "plan" columns.

Writes runs/<bank>/accounting/pnl_rollforward.xlsx and pnl.lines, equity.rollforward, cash.ledger,
capital.monthly, acct.monthly_values (labbank_common/run_store: history per report_date and run).

Usage: python accounting/python_code/pnl_rollforward.py [--bank eur_y1] [--recompute-nii]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import acct_common as ac  # noqa: E402
import acct_pricing as ap  # noqa: E402

ROOT = ac.ROOT
for p in (os.path.join(ROOT, "visual_rep"), os.path.join(ROOT, "balance_generate", "python_code")):
    if p not in sys.path:
        sys.path.insert(0, p)
from labbank_common.bank_config import chain_root, get_bank, get_engine  # noqa: E402

OPEX_RATE = 0.0065            # optimize_prep/python_code/product_map.OPEX_RATE (kept equal; checked in tests)
CB_DEPOSIT = (3510, "A")      # cash surplus
INTERBANK_BORROWING = (7900, "L")
NONCASH = ("fvtpl_derivatives", "fv_hedge_swaps", "fv_hedge_adjustment", "fvtpl_securities", "ecl_stock_change")


# ── inputs ───────────────────────────────────────────────────────────────────────────────────

def plan_path(bank) -> pd.DataFrame:
    from rollforward_report import monthly_balances
    return monthly_balances(bank)


def interest_by_month(bank, path, recompute: bool) -> pd.DataFrame:
    f = bank.output_dir("visual_rep") / "dynamic_nii.xlsx"
    rf = bank.output_dir("balance_generate", create=False) / "rollforward_summary.xlsx"
    if not recompute and f.exists() and (not rf.exists() or f.stat().st_mtime >= rf.stat().st_mtime):
        df = pd.read_excel(f, sheet_name="by_product_month")
        yr = pd.read_excel(f, sheet_name="year").set_index("item").iloc[:, 0].to_dict()
        df.attrs["year"] = yr
        df.attrs["source"] = str(f)
        return df
    import dynamic_nii as dn
    env = os.environ.get("LABBANK_BANK")
    try:
        df = dn.dynamic_nii(bank, path)
    finally:
        if env is None:
            os.environ.pop("LABBANK_BANK", None)
        else:
            os.environ["LABBANK_BANK"] = env
    df.attrs["source"] = "dynamic_nii (recomputed)"
    return df


def previous_stage(base) -> dict | None:
    """Closing state of the base bank when it is itself a
    rolled bank with its own P&L. None for a generated base."""
    if base.rollforward is None:
        return None
    f = base.output_dir("accounting", create=False) / "pnl_rollforward.xlsx"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: the base bank {base.bank_id} needs its P&L first")
    summ = pd.read_excel(f, sheet_name="summary").set_index("item")["value"]
    sheets = pd.ExcelFile(f).sheet_names
    hist = pd.read_excel(f, sheet_name="history" if "history" in sheets else "pnl")
    hist["date"] = pd.to_datetime(hist["date"])
    return dict(summary=summ, history=hist[["date", "profit_before_tax", "tax", "net_profit"]])


def swap_universe(bank, base) -> pd.DataFrame:
    """Every swap alive at some point of the window: the base book plus the rolled bank's trades that
    start inside the window (new swaps). hedge_accounting from the base book for held trades."""
    b = ac.swap_book(base)
    r = ac.swap_book(bank)
    new = r[~r["swap_id"].isin(b["swap_id"])]
    sw = pd.concat([b, new], ignore_index=True)
    sw["start_date"] = pd.to_datetime(sw["start_date"])
    sw["maturity_date"] = pd.to_datetime(sw["maturity_date"])
    sw["is_new"] = ~sw["swap_id"].isin(b["swap_id"])
    return sw[sw["maturity_date"] > pd.Timestamp(base.report_date)].reset_index(drop=True)


# ── the loop ─────────────────────────────────────────────────────────────────────────────────

def run(bank_id: str | None = "eur_y1", recompute_nii: bool = False, write: bool = True) -> dict:
    bank = get_bank(bank_id)
    if bank.rollforward is None:
        raise ValueError(f"{bank.bank_id} is not a rolled-forward bank")
    base = get_bank(bank.rollforward.base_bank)
    base_rd, new_rd = pd.Timestamp(base.report_date), pd.Timestamp(bank.report_date)
    root = chain_root(bank)
    root_rd = pd.Timestamp(root.report_date)
    prev = previous_stage(base)                   # a rolled base: what it closed with
    dates = [base_rd] + [base_rd + pd.offsets.MonthEnd(m) for m in range(1, 13)]
    days = np.array([(dates[m] - dates[m - 1]).days for m in range(1, 13)], float)
    fin = ac.fin_params(base)
    cit, payout, div_month = fin["CIT"], fin["dividend_payout"], int(fin["dividend_month"])
    cls = ac.product_classes(base).set_index(["product_code", "bs_side"])
    try:                                          # catalogue products of the rolled bank (new codes)
        extra = ac.product_classes(bank).set_index(["product_code", "bs_side"])
        cls = pd.concat([cls, extra[~extra.index.isin(cls.index)]])
    except Exception:                             # noqa: BLE001 -- the rolled bank's file is optional here
        pass

    print("plan path ...")
    path = plan_path(bank)
    tot = path.pivot_table(index=["product_code", "bs_side"], columns="month", values="total", aggfunc="sum").fillna(0.0)
    ex = path.pivot_table(index=["product_code", "bs_side"], columns="month", values="existing", aggfunc="sum").fillna(0.0)
    keys = tot.index
    info = cls.reindex(keys)
    if info["own_name"].isna().any():
        raise ValueError(f"products of the path without a bs_structure row: {list(keys[info['own_name'].isna()])}")
    is_eq = info["is_equity"].fillna(False).astype(bool).to_numpy()
    side = np.array([k[1] for k in keys])
    T = tot.to_numpy()                                           # (n, 13)
    avg = (T[:, :-1] + T[:, 1:]) / 2.0                           # (n, 12)
    E_plan = T[is_eq].sum(axis=0)
    A_plan = T[side == "A"].sum(axis=0)
    L_plan = T[(side != "A") & ~is_eq].sum(axis=0)
    if np.max(np.abs(A_plan - L_plan - E_plan)) > 1.0:
        raise AssertionError("plan path: A != L + E")

    print("interest (IRRBB NII engine on the path) ...")
    nii = interest_by_month(bank, path, recompute_nii)
    nii_pm = nii.pivot_table(index=["product_code", "bs_side"], columns="month", values="interest", aggfunc="sum").fillna(0.0)
    rate_pm = nii.pivot_table(index=["product_code", "bs_side"], columns="month", values="rate", aggfunc="first")

    # ── per-product lines ────────────────────────────────────────────────────────────────
    f = lambda c: pd.to_numeric(info[c], errors="coerce").fillna(0.0).to_numpy()[:, None]  # noqa: E731
    yf = days[None, :] / 365.0
    fees = f("fee_unit_rate") * avg * yf
    asset = (side == "A")[:, None]
    el = np.where(asset, f("PD") * f("LGD"), 0.0)
    stock = el * T                                                # 12M ECL stock
    write_off = -(el * avg * yf)
    ecl_change = -(stock[:, 1:] - stock[:, :-1])
    deriv = np.array([str(k[0]) in ("0", "100") for k in keys])[:, None]
    opex = np.where(~is_eq[:, None] & ~deriv, -OPEX_RATE * avg * yf, 0.0)
    acq = -f("acq_cost_rate") * np.maximum(T[:, 1:] - T[:, :-1], 0.0)

    # ── securities at fair value ─────────────────────────────────────────────────────────
    eng_b, eng_r = get_engine(base), get_engine(bank)
    fv_codes = [k for k in keys if bool(info.loc[k, "is_security"]) and info.loc[k, "accounting"] in ("FVOCI", "FVTPL")]
    names = {"EUR_disc_curve"}
    curves = None
    sec_rows = []
    fi_b = pd.read_sql(text("SELECT * FROM schemat.financial_instruments WHERE report_date = :rd AND bs_side = 'A'"),
                       eng_b, params={"rd": base_rd.date()})
    fi_r = pd.read_sql(text("SELECT * FROM schemat.financial_instruments WHERE report_date = :rd AND bs_side = 'A' "
                            "AND start_date > :b"), eng_r, params={"rd": new_rd.date(), "b": base_rd.date()})
    for fr in (fi_b, fi_r):
        fr["product_code"] = fr["product_code"].astype(int)
        names |= set(fr["disc_curve"].dropna())
    sw = swap_universe(bank, base)
    names |= set(sw["disc_curve"].dropna())
    fwd_names = set(sw["fwd_curve"].dropna())
    allc = pd.read_sql(text("SELECT DISTINCT curve_name FROM mkt.curves"), eng_r)["curve_name"]
    names |= {c for c in allc if any(c == f_ or c.startswith(f_ + "_") for f_ in fwd_names)}
    print("curves along the path ...")
    curves = ap.PathCurves(eng_r, names, min(sw["start_date"].min(), base_rd - pd.DateOffset(months=13))
                           if len(sw) else base_rd - pd.DateOffset(months=13), new_rd)

    reval = np.zeros((len(keys), 13))
    realised = np.zeros((len(keys), 12))
    for i, k in enumerate(keys):
        if k not in fv_codes:
            continue
        spread = float(info.loc[k, "mv_spread_bp"] or 0.0)
        old = fi_b[fi_b["product_code"] == k[0]]
        new = fi_r[fi_r["product_code"] == k[0]]
        prev_frac = 1.0
        for m, d in enumerate(dates):
            alive = old[pd.to_datetime(old["maturity_date"]) > d]
            aged = float(alive["balance_amt"].sum())
            mv_old = sum(ap.security_value(r, d, curves, spread)[0] for r in alive.itertuples())
            p_old = mv_old / aged if aged > 0 else 1.0
            held_old = min(float(ex.loc[k, m]), aged)
            frac = held_old / aged if aged > 0 else 0.0
            nb = new[(pd.to_datetime(new["start_date"]) <= d) & (pd.to_datetime(new["maturity_date"]) > d)]
            nom_n = float(nb["balance_amt"].sum())
            mv_n = sum(ap.security_value(r, d, curves, spread)[0] for r in nb.itertuples())
            p_new = mv_n / nom_n if nom_n > 0 else 1.0
            held_new = float(T[i, m]) - held_old
            reval[i, m] = held_old * (p_old - 1.0) + held_new * (p_new - 1.0)
            if m and frac < prev_frac - 1e-12:                  # sold pro rata down to the plan
                realised[i, m - 1] = aged * (prev_frac - frac) * (p_old - 1.0)
            prev_frac = frac
            sec_rows.append(dict(month=m, date=d, product_code=k[0], bs_side=k[1], accounting=info.loc[k, "accounting"],
                                 held_existing=held_old, price_existing=p_old, held_new=held_new, price_new=p_new,
                                 revaluation=reval[i, m]))
    acc_cls = info["accounting"].to_numpy()
    oci = reval[acc_cls == "FVOCI"].sum(axis=0)
    tpl_r = reval[acc_cls == "FVTPL"].sum(axis=0)
    fvtpl_sec = np.where((acc_cls == "FVTPL")[:, None], reval[:, 1:] - reval[:, :-1] + realised, 0.0)
    real_oci = np.where((acc_cls == "FVOCI")[:, None], realised, 0.0)

    # ── derivatives ──────────────────────────────────────────────────────────────────────
    print(f"swaps at every month-end ({len(sw)}) ...")
    mv = np.zeros((len(sw), 13))
    mv0 = np.zeros(len(sw))
    skipped = []
    for j, s in enumerate(sw.itertuples()):
        if str(s.swap_type).upper() != "IRS":
            skipped.append(s.swap_id)
            continue
        for m, d in enumerate(dates):
            if s.start_date <= d:
                mv[j, m] = ap.swap_value(s, d, curves)[0]
        if s.start_date > root_rd:     # traded inside a modelled window (this stage or an earlier one)
            mv0[j] = ap.swap_value(s, s.start_date, curves)[0]
    if skipped:
        print(f"  [warn] not valued monthly (cross-currency): {skipped}")
    ha = sw["hedge_accounting"].to_numpy()
    started = np.array([[s.start_date <= d for d in dates] for s in sw.itertuples()]) if len(sw) else np.zeros((0, 13), bool)
    designated = np.where(started, mv - mv0[:, None], 0.0)
    fvh_adj = -designated[ha == "fair_value"].sum(axis=0)
    cfh = designated[ha == "cash_flow"].sum(axis=0)
    dmv = mv[:, 1:] - mv[:, :-1]
    fvtpl_der = dmv[ha == "none"].sum(axis=0)
    fvh_swaps = dmv[ha == "fair_value"].sum(axis=0)
    fvh_line = fvh_adj[1:] - fvh_adj[:-1]
    d_assets = np.maximum(mv, 0.0).sum(axis=0)
    d_liabs = np.maximum(-mv, 0.0).sum(axis=0)

    # ── P&L, tax, dividends, cash: month by month ───────────────────────────────────────
    def _r(key, m):
        try:
            v = float(rate_pm.loc[key, m])
            return v if np.isfinite(v) else 0.0
        except KeyError:
            return 0.0
    interest = nii_pm.reindex(keys).fillna(0.0).to_numpy()
    swap_int = nii_pm.loc[(0, "A")].to_numpy() if (0, "A") in nii_pm.index else np.zeros(12)
    S = stock.sum(axis=0)
    E0 = {k: float(T[i, 0]) for i, k in enumerate(keys) if is_eq[i]}
    re0 = float(sum(v for (pc, sd), v in E0.items() if info.loc[(pc, sd), "product_name"] == "retained_earnings"))
    other_eq = float(sum(E0.values())) - re0
    # Opening retained earnings = the book's (nominal, cash-based) + the non-cash results already in
    # equity: FVTPL swaps' and securities' value, the day-one value of hedging swaps traded inside an
    # earlier modelled window, and (rolled base) the ECL allowance it carries.
    # A generated base only has the first two (held swaps are designated at 0).
    ecl_carry = float(prev["summary"].get("ecl_allowance_carry_close", 0.0)) if prev else 0.0
    held0 = (ha != "none") & (sw["start_date"] <= base_rd).to_numpy() if len(sw) else np.zeros(0, bool)
    designated_0 = float(mv0[held0].sum()) if len(sw) else 0.0
    noncash_open = float(mv[ha == "none", 0].sum()) + float(tpl_r[0]) + designated_0 + ecl_carry
    re = re0 + noncash_open
    cash, tax_ytd, pbt_ytd, year = 0.0, 0.0, 0.0, dates[1].year
    # calendar-year history: tax is cumulative in the year, the dividend pays last year's profit;
    # a window (Oct-Sep) cuts the year, so the base's months come along
    np_hist = []
    hist_rows = []
    if prev is not None:
        h = prev["history"]
        np_hist = [(d_, float(v)) for d_, v in zip(h["date"], h["net_profit"])]
        same = h["date"].dt.year == year
        pbt_ytd, tax_ytd = float(h.loc[same, "profit_before_tax"].sum()), float(-h.loc[same, "tax"].sum())
        hist_rows = h.to_dict("records")
    root_factor = 12.0 / max(1, 12 - root_rd.month)          # the root's year: modelled months annualised

    def year_profit(y: int):
        v = [x for dd, x in np_hist if dd.year == y]
        if not v:
            return None
        return sum(v) * (root_factor if y == root_rd.year else 1.0)

    rows, eq_rows, cash_rows, cap_rows, bs_rows = [], [], [], [], []
    fy_prev_est, div_paid = None, False
    lines_m = []
    for m in range(1, 13):
        d = dates[m]
        if d.year != year:
            year, tax_ytd, pbt_ytd = d.year, 0.0, 0.0
        r_plug = _r(CB_DEPOSIT, m) if cash >= 0 else _r(INTERBANK_BORROWING, m)
        plug_int = cash * r_plug * days[m - 1] / 365.0
        L = {
            "interest": float(interest[:, m - 1].sum() + swap_int[m - 1]),
            "interest_cash_plug": plug_int,
            "fees": float(fees[:, m - 1].sum()),
            "credit_write_offs": float(write_off[:, m - 1].sum()),
            "ecl_stock_change": float(ecl_change[:, m - 1].sum()),
            "opex": float(opex[:, m - 1].sum()),
            "acq_cost": float(acq[:, m - 1].sum()),
            "fvtpl_derivatives": float(fvtpl_der[m - 1]),
            "fv_hedge_swaps": float(fvh_swaps[m - 1]),
            "fv_hedge_adjustment": float(fvh_line[m - 1]),
            "fvtpl_securities": float(fvtpl_sec[:, m - 1].sum()),
            "realised_fvoci": float(real_oci[:, m - 1].sum()),
        }
        pbt = sum(L.values())
        pbt_ytd += pbt
        tax_due = cit * max(pbt_ytd, 0.0)
        L["tax"] = -(tax_due - tax_ytd)
        tax_ytd = tax_due
        npf = pbt + L["tax"]
        np_hist.append((d, npf))
        hist_rows.append(dict(date=d, profit_before_tax=pbt, tax=L["tax"], net_profit=npf))
        div = 0.0
        if d.month == div_month and not div_paid:
            fy_prev_est = year_profit(d.year - 1)                 # last financial year's profit
            if fy_prev_est is not None:
                div, div_paid = payout * max(fy_prev_est, 0.0), True
        noncash = sum(L[k] for k in NONCASH)
        cash_open = cash
        cash = cash + npf - noncash - div - (E_plan[m] - E_plan[m - 1])
        re_open = re
        re = re + npf - div
        lines_m.append(dict(month=m, date=d, **L, profit_before_tax=pbt, net_profit=npf))
        eq_rows += [
            dict(month=m, date=d, line="retained_earnings", opening=re_open, net_profit=npf, dividends=-div,
                 oci=0.0, closing=re),
            dict(month=m, date=d, line="share_capital_and_reserves", opening=other_eq, net_profit=0.0, dividends=0.0,
                 oci=0.0, closing=other_eq),
            dict(month=m, date=d, line="oci_reserve", opening=float(oci[m - 1]), net_profit=0.0, dividends=0.0,
                 oci=float(oci[m] - oci[m - 1]), closing=float(oci[m])),
            dict(month=m, date=d, line="cash_flow_hedge_reserve", opening=float(cfh[m - 1]), net_profit=0.0,
                 dividends=0.0, oci=float(cfh[m] - cfh[m - 1]), closing=float(cfh[m])),
        ]
        cash_rows.append(dict(month=m, date=d, opening=cash_open, net_profit=npf, non_cash=-noncash, dividends=-div,
                              plan_equity_change=-(E_plan[m] - E_plan[m - 1]), closing=cash,
                              account="central_bank_deposit" if cash >= 0 else "interbank_borrowing",
                              rate=r_plug))
        equity = other_eq + re + oci[m] + cfh[m]
        A = A_plan[m] + oci[m] + tpl_r[m] + d_assets[m] + fvh_adj[m] - (S[m] - S[0]) + ecl_carry + max(cash, 0.0)
        Lb = L_plan[m] + d_liabs[m] + max(-cash, 0.0)
        bs_rows.append(dict(month=m, date=d, assets_plan=A_plan[m], securities_reval=oci[m] + tpl_r[m],
                            derivative_assets=d_assets[m], fv_hedge_adjustment=fvh_adj[m],
                            ecl_allowance_change=-(S[m] - S[0]) + ecl_carry, cash_surplus=max(cash, 0.0), total_assets=A,
                            liabilities_plan=L_plan[m], derivative_liabilities=d_liabs[m],
                            cash_borrowing=max(-cash, 0.0), total_liabilities=Lb, equity=equity,
                            equity_plan=E_plan[m], check_a_minus_l_minus_e=A - Lb - equity))
        # foreseeable dividends: last year's profit until it is paid in dividend_month, + this year's so far
        last_fy = year_profit(d.year - 1) if d.month < div_month else None
        foreseeable = payout * max(last_fy or 0.0, 0.0) + payout * max(year_profit(d.year) or 0.0, 0.0)
        rwa = float(np.nansum(f("rwa_weight")[:, 0] * T[:, m] * (side == "A")))
        cet1 = other_eq + re + oci[m] - foreseeable
        cap_rows.append(dict(month=m, date=d, cet1=cet1, foreseeable_dividends=foreseeable, rwa=rwa,
                             cet1_ratio=cet1 / rwa if rwa else np.nan, equity_plan=E_plan[m],
                             cet1_ratio_plan=E_plan[m] / rwa if rwa else np.nan,
                             cet1_target=fin.get("CET1", np.nan)))
    pnl = pd.DataFrame(lines_m)
    eq = pd.DataFrame(eq_rows)
    cashl = pd.DataFrame(cash_rows)
    cap = pd.DataFrame(cap_rows)
    bsm = pd.DataFrame(bs_rows)

    # long P&L lines per product (the per-product parts)
    long = []
    for i, k in enumerate(keys):
        nm = info.loc[k, "own_name"]
        for m in range(1, 13):
            for line, arr in (("interest", interest), ("fees", fees), ("credit_write_offs", write_off),
                              ("ecl_stock_change", ecl_change), ("opex", opex), ("acq_cost", acq),
                              ("fvtpl_securities", fvtpl_sec), ("realised_fvoci", real_oci)):
                v = float(arr[i, m - 1])
                if v:
                    long.append(dict(month=m, date=dates[m], product_code=k[0], bs_side=k[1], own_name=nm,
                                     line=line, amount=v))
    for m in range(1, 13):
        for line in ("interest_cash_plug", "fvtpl_derivatives", "fv_hedge_swaps", "fv_hedge_adjustment", "tax"):
            long.append(dict(month=m, date=dates[m], product_code=-1, bs_side="", own_name="bank", line=line,
                             amount=float(pnl.loc[m - 1, line])))
        long.append(dict(month=m, date=dates[m], product_code=0, bs_side="A", own_name="irs_book",
                         line="interest", amount=float(swap_int[m - 1])))
    lines = pd.DataFrame(long)

    # ── checks ───────────────────────────────────────────────────────────────────────────
    eq_tot = eq.groupby("month")[["opening", "net_profit", "dividends", "oci", "closing"]].sum()
    d_eq = (eq_tot["closing"] - eq_tot["opening"] - eq_tot["net_profit"] - eq_tot["dividends"] - eq_tot["oci"]).abs().max()
    yr_nii = float(pnl["interest"].sum())
    dn_moving = float(nii.attrs.get("year", {}).get("moving", np.nan))
    checks = dict(
        max_abs_a_minus_l_minus_e=float(bsm["check_a_minus_l_minus_e"].abs().max()),
        max_abs_equity_identity=float(d_eq),
        interest_year=yr_nii, dynamic_nii_moving_year=dn_moving, interest_vs_dynamic_nii=yr_nii - dn_moving,
        lines_sum_vs_net_profit=float(lines["amount"].sum() - pnl["net_profit"].sum()),
        interest_source=nii.attrs.get("source"),
    )
    yr = pnl.drop(columns=["month", "date"]).sum()
    eq_open = float(other_eq + re0 + noncash_open + oci[0] + cfh[0])
    prev_close = float(prev["summary"]["equity_close"]) if prev else np.nan
    summary = dict(bank=bank.bank_id, base_bank=base.bank_id, root_bank=root.bank_id,
                   window=f"{base_rd.date()} -> {new_rd.date()}",
                   CIT=cit, dividend_payout=payout, dividend_month=div_month,
                   fy_base_profit_estimate=fy_prev_est, dividends_paid=float(-cashl["dividends"].sum()),
                   equity_open=eq_open, equity_close_previous_stage=prev_close,
                   equity_continuity_gap=eq_open - prev_close if prev else 0.0,
                   noncash_equity_open=noncash_open, ecl_allowance_carry_open=ecl_carry,
                   ecl_allowance_carry_close=ecl_carry - float(S[-1] - S[0]),
                   equity_close=float(bsm["equity"].iloc[-1]), equity_plan_close=float(E_plan[-1]),
                   cash_surplus_close=float(cashl["closing"].iloc[-1]),
                   cet1_ratio_close=float(cap["cet1_ratio"].iloc[-1]),
                   cet1_ratio_plan_close=float(cap["cet1_ratio_plan"].iloc[-1]),
                   roe=float(yr["net_profit"] / (other_eq + re0)), skipped_swaps=", ".join(skipped), **checks)
    values = pd.DataFrame([dict(month=m, date=d, swap_id=s.swap_id, hedge_accounting=s.hedge_accounting, mv_clean=mv[j, m])
                           for j, s in enumerate(sw.itertuples()) for m, d in enumerate(dates)])
    out = dict(summary=summary, pnl=pnl, year=yr, lines=lines, equity=eq, cash=cashl, capital=cap,
               balance_sheet=bsm, securities=pd.DataFrame(sec_rows), swaps=values,
               history=pd.DataFrame(hist_rows)[["date", "profit_before_tax", "tax", "net_profit"]])
    if write:
        _write(bank, out)
    return out


def _write(bank, out: dict) -> None:
    from labbank_common import run_store
    d = bank.output_dir("accounting")
    with pd.ExcelWriter(d / "pnl_rollforward.xlsx") as w:
        pd.Series(out["summary"]).rename("value").rename_axis("item").reset_index().to_excel(w, sheet_name="summary", index=False)
        out["year"].rename("amount").rename_axis("line").reset_index().to_excel(w, sheet_name="year", index=False)
        for k in ("pnl", "equity", "cash", "capital", "balance_sheet", "lines", "securities", "swaps", "history"):
            out[k].to_excel(w, sheet_name=k, index=False)
    eng = get_engine(bank)
    run = run_store.current_run(bank)
    for df, tbl, sch in ((out["lines"], "lines", "pnl"), (out["pnl"], "monthly", "pnl"),
                         (out["equity"], "rollforward", "equity"), (out["cash"], "ledger", "cash"),
                         (out["capital"], "monthly", "capital"), (out["balance_sheet"], "monthly_balance_sheet", "acct"),
                         (out["swaps"], "monthly_swap_values", "acct")):
        run_store.replace_run(eng, df, tbl, sch, run)
    print(f"Saved -> {d / 'pnl_rollforward.xlsx'} and pnl/equity/cash/capital tables ({run.report_date.date()}, {run.run_id})")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--bank", default=None, help="rolled-forward bank (default: LABBANK_BANK)")
    ap_.add_argument("--recompute-nii", action="store_true")
    ap_.add_argument("--no-write", action="store_true")
    a = ap_.parse_args()
    bank = get_bank(a.bank)
    if bank.rollforward is None:
        print(f"{bank.bank_id}: not a rolled-forward bank, monthly P&L skipped")
        return
    r = run(bank.bank_id, a.recompute_nii, write=not a.no_write)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p = r["pnl"].set_index("month").drop(columns="date")
    print((p / 1e6).round(2).to_string())
    print((r["year"] / 1e6).round(2).to_string())
    c = r["capital"].set_index("month")[["cet1", "rwa", "cet1_ratio", "cet1_ratio_plan"]]
    print(c.assign(cet1=c.cet1 / 1e6, rwa=c.rwa / 1e6).round(4).to_string())
    print({k: (round(v / 1e6, 3) if isinstance(v, float) and abs(v) > 1e3 else v) for k, v in r["summary"].items()})


if __name__ == "__main__":
    main()
