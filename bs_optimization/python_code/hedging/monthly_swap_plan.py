"""monthly_swap_plan.py
=====================
When in the plan year, in which tenor and on which floating leg should the bank buy its payer swaps?

The balance sheet is fixed (the plan, or one of the optimiser's portfolios); the swaps the plan trades inside
the year are taken out and bought again by a linear program:

    variables    x[m, T, f] >= 0    notional of payers of tenor T (2/3/4/5/7/10Y) on a f = 3M / 6M EURIBOR leg
                                    traded in month m of the plan year, at that date's par quote + cost
    limits       EVE and NII (6 EBA shocks, % Tier 1) inside the appetite at EVERY month end of the year
    size         at most MONTH_CAP of new notional a month
    objective    "carry": interest of the new swaps in the plan year + in the 12 months after it (what the ALM
                          desk reports), or
                 "cost":  trading cost only, cost_bp x annuity (swaps at par have no other value cost)
    slack        a month end whose limits no swap mix can meet is reported, not hidden

Book at the month ends: the exact cash-flow EVE / NII deltas of today's bank and of the plan bank without the
plan's new swaps, a straight line between them (the roll-forward moves balances gradually; the months in between
are not repriced). Swaps: dated_swaps.py (engine-equal pricing at the plan date), aged to each month end, on the
plan date's curves.

    LABBANK_BANK=eur_y1 python monthly_swap_plan.py --book plan
    LABBANK_BANK=eur_y1 python monthly_swap_plan.py --book alm_proposal --runs btrm_article_runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sqlalchemy import text

_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _g in ("optimizers", "scenarios", "hedging", "reporting"):
    sys.path.insert(0, os.path.join(_PY, _g))
sys.path.insert(0, os.path.normpath(os.path.join(_PY, "..", "..", "optimize_prep", "python_code")))
import dated_swaps as ds  # noqa: E402
from bs_vector import CurveTensors  # noqa: E402
from labbank_common.bank_config import get_bank, get_engine  # noqa: E402

TENORS = (2, 3, 4, 5, 7, 10)
LEGS = (3, 6)
COST_BP = float(os.environ.get("LABBANK_SWAP_COST_BPS", 12.0))
EVE_LIM, NII_LIM = -10.0, -4.0
MONTH_CAP = 100e6
SLACK_PEN = 1e3                         # EUR m per 1pp of Tier 1 outside the limits
SCEN = ("par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn", "own")


def sql_deltas(engine, rd) -> tuple[dict, dict]:
    out = []
    for tbl, col in (("irrbb.eve_results", "pv_total"), ("irrbb.nii_results", "nii_total")):
        d = pd.read_sql_query(text(f"SELECT scenario_id, SUM(CAST({col} AS FLOAT)) v FROM {tbl} "
                                   "WHERE report_date=:rd GROUP BY scenario_id"), engine, params={"rd": rd})
        v = d.set_index("scenario_id")["v"]
        out.append({s: float(v[s] - v["base"]) for s in SCEN})
    return tuple(out)


def discs() -> dict:
    ct = CurveTensors.load(os.path.join(str(ds.BANK.output_dir("optimize_prep")), "curve_tensors.npz"))
    c = ct.currency_index(ds.CCY)
    return {s: ct.disc_factors[ct.scenario_index(s), c] for s in ("base",) + SCEN}


def swap_deltas(comps: list, D: dict) -> tuple[np.ndarray, np.ndarray]:
    """(EVE, NII) deltas per unit notional, one per scenario."""
    if not comps:
        z = np.zeros(len(SCEN))
        return z, z
    e0, n0 = ds.eve(D["base"], comps), ds.float_nii(D["base"], comps)
    return (np.array([ds.eve(D[s], comps) - e0 for s in SCEN]),
            np.array([ds.float_nii(D[s], comps) - n0 for s in SCEN]))


def book_path(book: str, runs_name: str, D: dict, planned: list):
    """EVE / NII deltas (EUR) of the book without the plan year's new swaps at month 0 and at the plan date, and
    Tier 1 at both dates."""
    base = get_bank(ds.BANK.rollforward.base_bank)
    e0, n0 = sql_deltas(get_engine(base), pd.Timestamp(base.report_date))
    j0 = json.load(open(os.path.join(str(base.output_dir("bs_optimization")), "btrm_article_runs.json")))
    j1 = json.load(open(os.path.join(str(ds.BANK.output_dir("bs_optimization")), f"{runs_name}.json")))[book]
    t0, t1 = float(j0["current"]["tier1"]), float(j1["tier1"])
    e1 = np.array([j1["exact"]["eve"][s] for s in SCEN]) / 100.0 * t1
    n1 = np.array([j1["exact"]["nii"][s] for s in SCEN]) / 100.0 * t1
    for t in planned:                                        # the plan's own new swaps leave the book ...
        c = [ds.make(t["start"], t["maturity"], t["f"], t["k"], 1.0, t["nf"] / t["n"])]
        de, dn = swap_deltas(c, D)
        e1 -= t["n"] * de; n1 -= t["n"] * dn
    for bid, n in (j1.get("swaps") or {}).items():           # ... and so does the portfolio's swap overlay
        comps = (j1.get("swap_components") or {}).get(bid)
        if comps is None:
            raise ValueError(f"{book}: swaps without swap_components -- rerun btrm_article_runs (plan dates)")
        de, dn = swap_deltas(comps, D)
        sg = -1.0 if bid.startswith("SKIP") else 1.0
        e1 -= sg * n * de; n1 -= sg * n * dn
    return (np.array([e0[s] for s in SCEN]), np.array([n0[s] for s in SCEN])), (e1, n1), (t0, t1)


def solve(book: str, runs_name: str, objective: str, tail_only: bool = False, month_cap: float = MONTH_CAP) -> dict:
    D = discs()
    base_rd = pd.Timestamp(get_bank(ds.BANK.rollforward.base_bank).report_date)
    planned = ds.planned_trades(get_engine(ds.BANK), base_rd)
    n_m = (ds.RD.year - base_rd.year) * 12 + ds.RD.month - base_rd.month
    ends = [base_rd + pd.offsets.MonthEnd(k) for k in range(1, n_m + 1)]
    dates = [ds._roll((e - pd.offsets.MonthBegin(1)) + pd.Timedelta(days=14), 0) for e in ends]     # mid month
    (e0, n0), (e1, n1), (t0, t1) = book_path(book, runs_name, D, planned)

    inst = [(m, T, f) for m in range(n_m) for T in TENORS for f in LEGS]
    nx, nj, ns = len(inst), n_m, len(SCEN)
    A, rhs = [], []
    obj_c, obj_y, obj_cost = np.zeros(nx), np.zeros(nx), np.zeros(nx)
    sens = np.zeros((2, nj, ns, nx))
    for i, (m, T, f) in enumerate(inst):
        tr = [(dates[m], 1.0)]
        for j in range(m, nj):
            sens[0, j, :, i], sens[1, j, :, i] = swap_deltas(ds.strip(tr, T, f, COST_BP, asof=ends[j]), D)
        c = ds.strip(tr, T, f, COST_BP)
        obj_c[i] = ds.float_nii(D["base"], c) - ds.fixed_nii(c)
        obj_y[i] = ds.window_interest(dates[m], f, c[0]["k"], ends, base_rd).sum()
        obj_cost[i] = -COST_BP / 1e4 * float(ds._df(D["base"], c[0]["fixed_m"]).sum())
    # limits at every month end, slack per (month, metric)
    book_e, book_n, tier = [], [], []
    U = 1e6                                                  # EUR m
    for j in range(nj):
        w = (ends[j] - base_rd).days / (ds.RD - base_rd).days
        tj = t0 + w * (t1 - t0)
        be, bn = e0 + w * (e1 - e0), n0 + w * (n1 - n0)
        book_e.append(be / tj * 100); book_n.append(bn / tj * 100); tier.append(tj)
        if tail_only and j < nj - 1:
            continue
        for k, (b, lim) in enumerate(((be, EVE_LIM), (bn, NII_LIM))):
            for s in range(ns):
                row = np.zeros(nx + 2 * nj)
                row[:nx] = -sens[k, j, s] * U / tj * 100
                row[nx + 2 * j + k] = -1.0
                A.append(row); rhs.append(b[s] / tj * 100 - lim)
    for m in range(n_m):
        row = np.zeros(nx + 2 * nj)
        row[[i for i, q in enumerate(inst) if q[0] == m]] = 1.0
        A.append(row); rhs.append(month_cap / U)
    ob = obj_cost if objective == "cost" else obj_c + obj_y
    c = np.concatenate([-ob, np.full(2 * nj, SLACK_PEN)])
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(rhs), bounds=[(0, None)] * len(c), method="highs")
    if r.status != 0:
        raise RuntimeError(r.message)
    x, sl_ = r.x[:nx] * U, r.x[nx:].reshape(nj, 2)
    path = []
    for j in range(nj):
        ev = book_e[j] + sens[0, j] @ x / tier[j] * 100
        ni = book_n[j] + sens[1, j] @ x / tier[j] * 100
        path.append(dict(month=j + 1, date=str(ends[j].date()), eve_book=float(book_e[j].min()), eve=float(ev.min()),
                         nii_book=float(book_n[j].min()), nii=float(ni.min()),
                         slack_eve=float(sl_[j, 0]), slack_nii=float(sl_[j, 1])))
    trades = [dict(date=str(dates[m].date()), tenor=T, leg=f, notional=float(v),
                   fixed=ds.par_quote(T, dates[m]) + COST_BP / 1e4)
              for (m, T, f), v in zip(inst, x) if v > 1e5]
    return dict(book=book, objective=objective, tail_only=tail_only, trades=trades, total=float(x.sum()),
                year_interest=float(obj_y @ x), carry_after=float(obj_c @ x), cost_npv=float(obj_cost @ x), path=path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="plan")
    ap.add_argument("--runs", default="btrm_article_runs")
    ap.add_argument("--month-cap", type=float, default=MONTH_CAP)
    a = ap.parse_args()
    res = []
    for obj in ("cost", "carry"):
        for tail in (False, True):
            r = solve(a.book, a.runs, obj, tail, a.month_cap)
            res.append(r)
            print(f"\n[{a.book}] objective {obj}, limits {'at the plan date only' if tail else 'every month end'}: "
                  f"{r['total'] / 1e6:.0f}m, interest in the plan year {r['year_interest'] / 1e6:+.2f}m, "
                  f"12M after {r['carry_after'] / 1e6:+.2f}m, trading cost {r['cost_npv'] / 1e6:+.2f}m")
            print(pd.DataFrame(r["trades"]).round(4).to_string(index=False) if r["trades"] else "  no swaps")
            print(pd.DataFrame(r["path"]).round(2).to_string(index=False))
    out = os.path.join(str(ds.BANK.output_dir("bs_optimization")), f"monthly_swap_plan_{a.book}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
