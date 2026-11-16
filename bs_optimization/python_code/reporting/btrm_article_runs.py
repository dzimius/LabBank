"""btrm_article_runs.py
=====================
Every optimisation run behind the BTRM paper "Steering to a better balance sheet",
for one bank per process (module-level bank state in the optimisers):

    LABBANK_BANK=eur    python btrm_article_runs.py
    LABBANK_BANK=eur_y1 python btrm_article_runs.py

Settings (JointOptions, see joint_optimizer.py):
  - new pay-fixed swaps traded today only, at par, tenors 6M/2Y/5Y/7Y/10Y on a 3M or a 6M EURIBOR floating
    leg; first-year carry in the objective. On the plan, the plan's own new payers (traded inside the plan
    year) may be left out (SKIP_PLAN bucket) and replaced by other tenors / legs.
  - product-family cap (a product and its variants move by <= max_shift in total)
  - contractual run-off floors: loans and the covered bond cannot fall below the balance of existing
    contracts still outstanding at the end of the horizon (3Y for the long-term target from today,
    the plan year for runs on the plan); bonds are marketable, deposits are not floored
  - pricing concessions: half of the plan's growth of a family is organic (unchanged spreads), counted
    from today's balance; volume above that costs CONCESSION_BPS a year on the additional balance
    (lower offered loan spread / higher rate on newly attracted deposits). Runs on the plan get the
    allowance once (the plan already used it), so growth above the plan pays from the first euro.
  - derivatives carry no operating cost (product_map.DERIVATIVE_CODES)

Each final portfolio is re-checked against the pipeline's exact cash-flow results: the per-product
EVE/NII deltas in irrbb.eve_results / irrbb.nii_results are rescaled by each product's balance ratio
(the bank_reprice_at_weights.py method, on the report-date curves); variants with no balance today use
their family's main product profile; new swaps add their priced deltas.

Output: runs/<bank>/bs_optimization/btrm_article_runs.json (+ .pkl with the JointResult objects).
"""
from __future__ import annotations

import os as _o, sys as _s
_PY = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
for _g in ("optimizers", "scenarios", "hedging", "reporting"):
    if _o.path.join(_PY, _g) not in _s.path:
        _s.path.insert(0, _o.path.join(_PY, _g))
_ROOT = _o.path.normpath(_o.path.join(_PY, "..", ".."))
for _p in (_ROOT, _o.path.join(_ROOT, "optimize_prep", "python_code")):
    if _p not in _s.path:
        _s.path.insert(0, _p)

import dataclasses
import json
import os
import pickle

import numpy as np
import pandas as pd
from sqlalchemy import text

from labbank_common.bank_config import get_bank, get_engine
from labbank_common import fx as _fx
from bs_vector import BalanceSheetParams, CohortRates
from optimizer_io import load_optimizer_config
from product_map import _ProductMap, plan_year_start
import joint_optimizer as jo
import robust_curves as rc

BASE_BANK = "eur"                       # the bank the plan starts from
TENORS = (0.5, 2, 5, 7, 10)
APPETITE = (-10.0, -4.0)                # risk appetite on today's curve (EVE, NII), % T1
INTERNAL = (-12.0, -4.2)                # internal limit: binds on every other curve (robust runs)
EVE_SWEEP = (-12.0, -11.0, -9.0)        # price of the EVE appetite (other EVE floors, NII floor unchanged)
EP_LAMBDA = 0.5                         # EP = EP today + 0.5 x weighted change on the other curves
GROWTH_CAP = {0.05: 0.5, 0.025: 0.2}    # market capacity: a client product family grows by at most 50% of its
                                        # balance over the long-term horizon, 20% in one year beyond the plan,
GROWTH_FLOOR = 0.01                     # but always by up to 1% of assets (room to launch a small product)
SHRINK_CAP = {0.05: (0.05, 0.5),        # a product (with its new-business twin) shrinks by at most
              0.025: (0.025, 0.3)}      # max(share of assets, share of its own balance): 5% / 50% over the
                                        # long-term horizon, 2.5% / 30% in the one-year proposal
LCR_MIN = 2.15                          # liquidity floor; the book holds far more, so it rarely binds
LINES = {                               # business lines: each moves by at most max_shift of assets in total
    "mortgages": (("100", "110"), "A"), "consumer": (("200", "210", "220"), "A"),
    "business": (("400", "410", "420", "430"), "A"), "securities": (("300", "320"), "A"),
    "cash": (("350", "351"), "A"), "interbank_a": (("790",), "A"),
    "current_acc": (("600", "630"), "L"), "savings": (("800",), "L"), "term_dep": (("706", "707"), "L"),
    "covered": (("500",), "L"), "interbank_l": (("790",), "L"),
}
EVE_MODEL_BUFFER = 0.2                  # optimiser EVE floor 0.2pp inside the appetite: the fast linear layer
                                        # differs from the full-engine check by up to ~0.15pp
NII_MODEL_BUFFER = 0.1                  # optimiser NII floor 0.1pp inside the appetite on the plan: variants with
                                        # no balance are checked on their family's profile, ~0.06pp apart; today's
                                        # book matches the check exactly (2026-10-27), so no buffer there
RESET_MONTHS = (3, 6)                   # floating legs offered for new swaps: 3M and 6M EURIBOR
PINNED = {("0100", "A"), ("0100", "L"), ("6350", "L"), ("3500", "A")}   # 6350 corporate accounts: no business line, not a lever
LOAN_FAMS = ("100", "110", "200", "210", "220", "400", "410", "420", "430")
DEPOSIT_FAMS = ("600", "630", "706", "707", "800")
CONCESSION_BPS = 50.0
ORGANIC_SHARE = 0.5                     # share of the plan's growth achievable at unchanged spreads
LT_YEARS = 3                            # horizon of the long-term target
GLIDE_TOL = 0.002                       # glide path: a first-year client family stays between the plan and
                                        # the long-term target, +/- 0.2% of assets (EUR 20m)
NMD_STRESS = {("6000", "L"): 0.75, ("6300", "L"): 0.75, ("8000", "L"): 0.75}

BANK = get_bank()
BANK_ID = os.environ.get("LABBANK_BANK", "eur")
RD = pd.Timestamp(BANK.report_date)
ENGINE = get_engine(BANK)
OUT_DIR = os.path.join(str(get_bank(BANK_ID).output_dir("bs_optimization")))


def _q(sql: str, engine=ENGINE, **p) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), engine, params=p)


def _fam(pc) -> str:
    return str(pc)[:3]


# ── inputs ───────────────────────────────────────────────────────────────────
def balances(engine, rd) -> pd.Series:
    """(product_code, bs_side) -> balance in EUR at rd (USD converted)."""
    b = _q("SELECT product_code, bs_side, currency, SUM(balance_amt) bal FROM dbo.transactions "
           "WHERE report_date=:rd GROUP BY product_code, bs_side, currency", engine, rd=rd)
    fx = _fx.to_home(BANK, rd)
    b["bal"] = b["bal"] * b["currency"].map(fx).astype(float)
    b["product_code"] = b["product_code"].astype(str).str.zfill(4)
    return b.groupby(["product_code", "bs_side"])["bal"].sum()


def runoff_floor_today(years: int, codes=None) -> dict:
    """Existing contracts still outstanding after `years` (behavioural run-off incl. prepayment).
    codes: (pc, side) keys to return; default loans and the covered bond."""
    r = _q("""SELECT product_code, bs_side,
                     SUM(CASE WHEN cf_end_dt <= DATEADD(year, :y, :rd)
                              THEN ISNULL(beh_capital_pmt,0)+ISNULL(prepayment_pmt,0) ELSE 0 END) ro
              FROM cf.products WHERE cf_end_dt > :rd GROUP BY product_code, bs_side""", y=years, rd=RD)
    r["product_code"] = r["product_code"].astype(str).str.zfill(4)
    ro = r.set_index(["product_code", "bs_side"])["ro"]
    bal = balances(ENGINE, RD)
    out = {}
    for (pc, side), v in bal.items():
        keep = ((pc, side) in codes) if codes is not None else ((side == "A" and _fam(pc) in LOAN_FAMS) or pc == "5000")
        if keep:
            out[(pc, side)] = max(0.0, float(v) - float(ro.get((pc, side), 0.0)))
    return out


# ── existing book vs new business ────────────────────────────────────────────
# Every contractual product has a new-business twin in the catalogue (code: first three digits + "9", e.g.
# 1009 = a fixed mortgage originated on the report date at today's rates and new-business margins). The
# existing contracts (the parent product's cohorts) are never enlarged: additional volume is the twin.
#   long-term target (3Y): the existing book is fixed at its run-off level after 3 years; everything above
#                          is new business at today's pricing.
#   runs on the plan:      the existing book (aged contracts + the plan's own new business) can be cut down to
#                          the contracts that existed before the plan year, not enlarged.
#   marketable securities: may be sold below the existing balance; a sale pays the market-value result and
#                          SALE_TC_BP, spread over the remaining life (SALE_COST).
SALE_TC_BP = 5.0
MARKETABLE = {("3000", "A"), ("3200", "A")}


def twin_parents() -> dict:
    """(parent pc, side) -> twin pc for the new-business twins of the catalogue sheet."""
    from labbank_common import catalogue
    v = catalogue.read_variants(catalogue.bank_data_path(BANK))
    return {(p, s): c for p, s, c in zip(v["parent_code"], v["bs_side"], v["product_code"]) if c.endswith("9")}


def sale_cost_rate(params, pc: str, side: str) -> float:
    """Annual cost per unit sold: (book - market value + transaction cost) / remaining duration, >= 0 (a sale
    gain is not booked)."""
    m = (params.product_code.astype(str) == pc) & (params.bs_side.astype(str) == side) & (params.balance_arr > 0)
    if not m.any():
        return 0.0
    w = params.balance_arr[m]
    mv = float(np.average(params.eve_pv_factor[m], weights=w))
    dur = max(float(np.average(params.d_mod[m], weights=w)), 0.25)
    return max(0.0, 1.0 - mv) / dur + SALE_TC_BP / 1e4 / dur


def runoff_floor_plan() -> dict:
    """Contracts of the plan book that existed before the plan year (they cannot be undone)."""
    base_rd = pd.Timestamp(get_bank(BASE_BANK).report_date)
    out = {}
    for tbl in ("schemat.loans", "schemat.financial_instruments"):
        r = _q(f"SELECT product_code, bs_side, SUM(balance_amt) bal FROM {tbl} "
               f"WHERE start_date <= :b GROUP BY product_code, bs_side", b=base_rd)
        for pc, side, v in zip(r["product_code"].astype(str).str.zfill(4), r["bs_side"], r["bal"]):
            if (side == "A" and _fam(pc) in LOAN_FAMS) or pc == "5000":
                out[(pc, side)] = float(v)
    return out


def planned_swaps() -> dict:
    """Pay-fixed swaps the plan trades inside the plan year: one SKIP_PLAN bucket (notional-weighted
    remaining tenor at the plan date, total notional)."""
    base_rd = pd.Timestamp(get_bank(BASE_BANK).report_date)
    d = _q("SELECT swap_id, MIN(start_date) s, MAX(maturity_date) m, MAX(CAST(notional AS FLOAT)) n "
           "FROM schemat.ir_swaps WHERE report_date=:rd AND swap_type='IRS' GROUP BY swap_id", rd=RD)
    d = d[pd.to_datetime(d["s"]) > base_rd]
    if d.empty:
        return {}
    yrs = (pd.to_datetime(d["m"]) - RD).dt.days / 365.25
    return {"SKIP_PLAN": (round(float(np.average(yrs, weights=d["n"])) * 2) / 2, float(d["n"].sum()))}


def aged_swaps(years: int) -> dict:
    """Existing pay-fixed swaps that mature within `years`: they are no longer in the book at the end of the
    horizon. bucket_id 'AGED_<remaining years>Y_<swap_id>' -> (remaining years on the 3M grid, notional, fixed
    rate), removed by the optimiser as a forced left-out payer at its own fixed rate."""
    d = _q("SELECT swap_id, MAX(maturity_date) m, MAX(CAST(notional AS FLOAT)) n, "
           "MAX(CASE WHEN leg_type <> 'FLOAT' THEN fixed_rate END) k FROM schemat.ir_swaps "
           "WHERE report_date=:rd AND swap_type='IRS' GROUP BY swap_id", rd=RD)
    d["m"] = pd.to_datetime(d["m"])
    d = d[d["m"] <= RD + pd.DateOffset(years=years)]
    out = {}
    for r in d.itertuples():
        T = max(0.25, round((r.m - RD).days / 365.25 * 4) / 4)
        out[f"AGED_{T:g}Y_{r.swap_id}"] = (T, float(r.n), float(r.k))
    return out


def plan_growth() -> pd.Series:
    """Plan growth per family (EUR): plan book minus today's book."""
    today = balances(get_engine(get_bank(BASE_BANK)), pd.Timestamp(get_bank(BASE_BANK).report_date))
    plan = balances(get_engine(get_bank("eur_y1")), pd.Timestamp(get_bank("eur_y1").report_date))
    t = today.groupby([today.index.get_level_values(0).str[:3], today.index.get_level_values(1)]).sum()
    p = plan.groupby([plan.index.get_level_values(0).str[:3], plan.index.get_level_values(1)]).sum()
    return pd.concat([t.rename("today"), p.rename("plan")], axis=1).fillna(0.0)


def concession_inputs(ta: float, years_allowance: float) -> tuple[dict, dict]:
    """Thresholds (weights of this bank's TA) and rates per (family, side)."""
    g = plan_growth()
    thr, rate = {}, {}
    for (fam, side), row in g.iterrows():
        if (side == "A" and fam in LOAN_FAMS) or (side == "L" and fam in DEPOSIT_FAMS):
            allowance = ORGANIC_SHARE * max(0.0, row["plan"] - row["today"]) * years_allowance
            thr[(fam, side)] = (row["today"] + allowance) / ta
            rate[(fam, side)] = CONCESSION_BPS / 1e4
    return thr, rate


def shrink_floors(base_w: dict, tw: dict, max_shift: float) -> dict | None:
    """Product (+ its new-business twin) -> lower weight: base - max(abs share of assets, share of balance)."""
    c = SHRINK_CAP.get(max_shift)
    if c is None:
        return None
    twin_of = {(c, s): (p, s) for (p, s), c in tw.items()}  # twin -> parent key
    groups = {}
    for (pc, side), w in base_w.items():
        if pc in ("0000", "0100") or side == "E" or (pc, side) in PINNED:
            continue
        par = twin_of.get((pc, side), (pc, side))
        groups[par] = {par, (tw[par], side)} if par in tw else {par}
    out = {}
    for par, keys in groups.items():
        w0 = sum(base_w.get(k, 0.0) for k in keys)
        if w0 > 0.0:
            out[f"{par[0]}|{par[1]}"] = (sorted(keys), w0 - max(c[0], c[1] * w0))
    return out


def family_caps(params, max_shift: float) -> dict | None:
    """(family, side) -> max weight for client product families (market capacity)."""
    g = GROWTH_CAP.get(max_shift)
    if g is None:
        return None
    pm = _ProductMap(params)
    base = {}
    for (pc, side), w in zip(pm.products, pm.base_prod_w):
        base[(_fam(pc), side)] = base.get((_fam(pc), side), 0.0) + float(w)
    return {k: w + max(g * w, GROWTH_FLOOR) for k, w in base.items()
            if (k[1] == "A" and k[0] in LOAN_FAMS) or (k[1] == "L" and k[0] in DEPOSIT_FAMS)}


# ── exact check on the pipeline's cash flows ─────────────────────────────────
def exact_check(r, params) -> dict:
    pm = _ProductMap(params)
    fx = _fx.to_home(BANK, RD)
    # plan-year contracts (started after the base bank's date) apart: a product that moves through them
    # (_ProductMap.split) keeps its seasoned book and rescales only these
    py = plan_year_start(RD)
    py = (py if py is not None else RD).date()
    res = {}
    for tbl, col in (("irrbb.eve_results", "pv_total"), ("irrbb.nii_results", "nii_total")):
        d = _q(f"""SELECT r.scenario_id, r.product_code, r.bs_side, r.currency, ISNULL(s.nb, 0) nb, SUM(r.{col}) v
                   FROM {tbl} r LEFT JOIN (
                        SELECT pt, schedule_id, CASE WHEN start_date > :py THEN 1 ELSE 0 END nb FROM (
                            SELECT 'L' pt, schedule_id, start_date FROM sched.loans UNION ALL
                            SELECT 'D', schedule_id, start_date FROM sched.deposits UNION ALL
                            SELECT 'F', schedule_id, start_date FROM sched.fin_inst) u) s
                     ON s.pt = r.product_type AND s.schedule_id = r.schedule_id
                   WHERE r.report_date=:rd
                   GROUP BY r.scenario_id, r.product_code, r.bs_side, r.currency, ISNULL(s.nb, 0)""", rd=RD, py=py)
        d["v"] = d["v"] * d["currency"].map(fx).astype(float)
        d["product_code"] = d["product_code"].astype(str).str.zfill(4)
        p = d.pivot_table(index=["product_code", "bs_side", "nb"], columns="scenario_id", values="v",
                          aggfunc="sum").fillna(0.0)
        res[tbl] = p
    base_w = dict(zip(pm.products, pm.base_prod_w))
    new_w = dict(zip(pm.products, r.weights_bs_new))
    # new-business / catalogue units (no cash flows in the book): their own contractual pricing on today's
    # curve (the npz unit values of catalogue_params); everything else rescales the pipeline's results
    is_unit = np.array([bool(params.is_cohort[i]) and params.balance_arr[i] == 0.0
                        and str(params.cohort_id[i]).endswith(f"_{RD.year}_{RD.month:02d}")
                        for i in range(len(params.cohort_id))])
    unit_prod = set(int(j) for j in pm.cohort_to_prod[is_unit])
    # family scale on the family's main product (largest baseline); other variants ride on its profile
    scale = {}
    fams = {}
    for j, (pc, side) in enumerate(pm.products):
        if j not in unit_prod:
            fams.setdefault((_fam(pc), side), []).append((pc, side))
    prod_j = {k: j for j, k in enumerate(pm.products)}
    for key, members in fams.items():
        main = max(members, key=lambda k: base_w[k])
        if base_w[main] <= 0:
            continue
        tgt, m = sum(new_w[k] for k in members), float(pm.min_w[prod_j[main]])
        if m > 0.0:                                   # seasoned book kept, plan-year contracts rescaled
            scale[main + (0,)] = 1.0
            scale[main + (1,)] = (tgt - m) / (base_w[main] - m)
        else:
            scale[main + (0,)] = scale[main + (1,)] = tgt / base_w[main]
    scen = [str(s) for s in params.scenario_ids]
    amt_u = np.zeros(len(params.cohort_id))
    for i in np.where(is_unit)[0]:
        amt_u[i] = new_w[pm.products[pm.cohort_to_prod[i]]] * params.total_assets * pm.fraction[i]
    t1 = r.tier1_capital
    out = {"eve": {}, "nii": {}}
    for kind, tbl in (("eve", "irrbb.eve_results"), ("nii", "irrbb.nii_results")):
        p = res[tbl]
        sc = np.array([scale.get(k, 1.0) for k in p.index])
        tot = p.mul(sc, axis=0).sum()
        unit_d = params.delta_eve_unit if kind == "eve" else params.delta_nii_unit
        for s in [c for c in p.columns if c != "base"]:
            swap = float(np.dot(r.ladder[f"delta_{kind}_pln"][s], [r.swap_notional[b] for b in r.ladder["bucket_ids"]])) \
                if s in r.ladder[f"delta_{kind}_pln"] else 0.0
            units = float(np.dot(amt_u, unit_d[:, scen.index(s)])) if s in scen else 0.0
            out[kind][s] = (float(tot[s] - tot["base"]) + swap + units) / t1 * 100.0
        if kind == "nii":
            out["nii_base"] = float(tot["base"]) + float(np.dot(amt_u, params.nii_unit_rate))
    return out


# ── runs ─────────────────────────────────────────────────────────────────────
def run(tag: str, max_shift: float, floors: tuple, floor_kind: str, allowance_years: float,
        params, cr, cfg0, no_swaps: bool = False, robust: tuple | None = INTERNAL, **opt_over) -> dict:
    """robust: EVE/NII floors on the other base curves (robust_curves.py), None = today's curve only."""
    ta = params.total_assets
    pm0 = _ProductMap(params)
    base_w = dict(zip(pm0.products, pm0.base_prod_w))
    tw = twin_parents()
    maxw, sale = {}, {}
    if floor_kind in ("today3y", "renewed"):
        mins = runoff_floor_today(LT_YEARS)
        fl = runoff_floor_today(LT_YEARS, codes=set(tw))
        for k in tw:                                   # existing book fixed at its run-off level
            if k not in base_w:
                continue
            v = fl.get(k, 0.0)
            maxw[k] = v / ta
            if k in MARKETABLE:
                mins.pop(k, None)
                sale[k] = (v / ta, sale_cost_rate(params, *k))
            else:
                mins[k] = v
    elif floor_kind == "plan":
        mins = runoff_floor_plan()
        for k in tw:                                   # existing book: can be cut to the pre-plan contracts
            if k not in base_w:
                continue
            maxw[k] = base_w[k]
            if k in MARKETABLE:
                sale[k] = (base_w[k], sale_cost_rate(params, *k))
    else:
        mins = {}
    if floor_kind == "renewed":
        # today's structure with the run-off replaced like for like: every family at today's weight, the
        # existing book at its run-off level, the rest in the family's new-business twin, nothing else moves
        twins = {(c, s) for (p, s), c in tw.items()}
        fam_rng = {}
        for (pc, side), w in base_w.items():
            if (pc, side) in twins:
                continue
            if (pc, side) not in tw:
                mins[(pc, side)] = w * ta; maxw[(pc, side)] = w
            fam_rng[(_fam(pc), side)] = fam_rng.get((_fam(pc), side), 0.0) + w
        opt_over["family_range"] = {k: (v - 1e-9, v + 1e-9) for k, v in fam_rng.items()
                                    if any((c[:3], s) == k for c, s in twins)}
        for k in MARKETABLE & set(maxw):
            mins[k] = maxw[k] * ta                    # no sales: the bonds run off like the loans
        sale = {}
    # the long-term structure is the book after LT_YEARS: existing swaps maturing by then leave it too
    skip = None if (no_swaps or BANK_ID == BASE_BANK) else planned_swaps()
    forced = {}
    if floor_kind in ("today3y", "renewed"):
        aged = aged_swaps(LT_YEARS)
        skip = {**(skip or {}), **aged}
        forced = {b: v[1] for b, v in aged.items()}
        if floor_kind == "renewed":            # the programme continues like for like: 7Y payers at today's par
            forced["NEW_7Y"] = sum(v[1] for v in aged.values())
    fixed = opt_over.pop("fixed_swaps", None)      # bucket -> notional, the only swaps in the solution
    if fixed:
        forced.update(fixed)
    thr, rate = concession_inputs(ta, allowance_years)
    bps = opt_over.pop("concession_bps", None)
    if bps is not None:
        rate = {k: bps / 1e4 for k in rate}
    options = jo.JointOptions(
        new_swaps_today_only=True, swap_carry_in_objective=True, family_cap=True,
        min_weights={k: v / ta for k, v in mins.items()},
        concession_threshold=thr, concession_rate=rate, swap_reset_months=opt_over.pop("swap_reset_months", RESET_MONTHS),
        skippable_swaps=skip, forced_swaps=forced or None,
        line_caps=LINES, lcr_min=LCR_MIN, family_max=family_caps(params, max_shift),
        # the EVE floor on the other curves gets the same model buffer as on today's curve (2026-10-28: the
        # long-term target sat at -12.00 in the fast layer and -12.07 in the full-engine check)
        robust=None if robust is None else rc.load_robust(_ProductMap(params), params, EP_LAMBDA,
                                                          robust[0] + EVE_MODEL_BUFFER, robust[1]),
        group_min=None if floor_kind == "renewed" else shrink_floors(base_w, tw, max_shift),
        max_weights=maxw or None, acq_by_family=True, sale_cost=sale or None, subst_by_family=True,
        family_bound_only=(set(tw) | {(c, s) for (p, s), c in tw.items()}) if maxw else None,
        **opt_over)
    cfg = dataclasses.replace(cfg0, mode="max_shift", max_shift=max_shift,
                              sot_eve_floor=floors[0], sot_nii_floor=floors[1],
                              sot_eve_buffer=-EVE_MODEL_BUFFER,
                              sot_nii_buffer=0.0 if BANK_ID == BASE_BANK else -NII_MODEL_BUFFER,
                              eve_breach_weight=0.0, nii_breach_weight=0.0,
                              fixed_products=set(cfg0.fixed_products) | PINNED)
    r = jo.optimize_bs_and_ladder(cfg, params, cr,
                                  notional_cap=0.0 if no_swaps else 1.5e9,
                                  total_notional_cap=None if no_swaps else (0.0 if fixed else 3e9),
                                  tenors=TENORS, seasoned_notional_cap=None, options=options)
    ex = exact_check(r, params)
    pm = _ProductMap(params)
    out = dict(
        tag=tag, bank=BANK_ID, success=bool(r.success), max_shift=max_shift, floors=list(floors),
        total_assets=ta, tier1=r.tier1_capital,
        weights_old={f"{pc}|{s}": float(w) for (pc, s), w in zip(pm.products, pm.base_prod_w)},
        weights_new={f"{pc}|{s}": float(w) for (pc, s), w in zip(pm.products, r.weights_bs_new)},
        swaps={k: v for k, v in r.swap_notional.items() if v > 1e3},
        swap_fixed=dict(zip(r.ladder["bucket_ids"], map(float, r.ladder["fixed_rate"]))),
        ep_ftp_old=r.ep_bs_old, ep_ftp_new=r.ep_total_new,
        ep_bank_old=r.whole_bank_ep(new=False), ep_bank_new=r.whole_bank_ep(new=True),
        nii_old=r.nii_base_old, nii_new=r.nii_base_new + r.ep_carry_swap,
        concession_old=r.concession_cost_old, concession_new=r.concession_cost_new,
        acq_new=r.acq_cost_bs_new, opex_old=r.opex_bs_old, opex_new=r.opex_bs_new,
        el_old=r.el_bs_old, el_new=r.el_bs_new, coc_old=r.coc_bs_old, coc_new=r.coc_bs_new,
        fee_old=r.fee_bs_old, fee_new=r.fee_bs_new, swap_carry=r.ep_carry_swap,
        sot_eve=r.sot_eve, sot_nii=r.sot_nii, exact=ex,
        lcr=r.lcr.get("EUR"), nsfr=r.nsfr.get("EUR"),
        min_weights={f"{k[0]}|{k[1]}": v / ta for k, v in mins.items()},
        skip_tenor=(options.skippable_swaps or {}).get("SKIP_PLAN", (None,))[0],
        robust=getattr(r, "robust", {}), robust_floors=list(robust) if robust else None,
    )
    to_exact_basis(out)
    print(f"[{tag}] EP ftp {out['ep_ftp_old']/1e6:.1f}->{out['ep_ftp_new']/1e6:.1f}  "
          f"EP bank {out['ep_bank_old']/1e6:.1f}->{out['ep_bank_new']/1e6:.1f}  "
          f"EVE {min(r.sot_eve.values()):+.2f} (exact {min(ex['eve'].values()):+.2f})  "
          f"NII {min(r.sot_nii.values()):+.2f} (exact {min(ex['nii'].values()):+.2f})  LCR {out['lcr']:.0%}  swaps {out['swaps']}")
    for name, v in out["robust"].items():
        print(f"     {name:9s} EVE {min(v['eve'].values()):+6.2f}  NII {min(v['nii'].values()):+6.2f}  EP {v['ep_delta']/1e6:+6.1f}")
    return out, r


def to_exact_basis(out: dict) -> dict:
    """Put NII and EP of the new portfolio on the exact (full cash-flow) basis, in place.

    The optimiser's fast layer prices the new portfolio's 12M NII; exact_check reprices it with the full
    engine. EVE/NII deltas were already exact, EP was not (2026-11-09: eur_y1 proposal +9.7m). The gap
    (layer - exact NII, swap carry included on both sides) is taken off NII, whole-bank EP and FTP EP;
    the layer values stay as *_layer. Idempotent."""
    if "nii_new_layer" in out or "nii_base" not in (out.get("exact") or {}):
        return out
    gap = out["nii_new"] - (out["exact"]["nii_base"] + out["swap_carry"])
    for k in ("nii_new", "ep_bank_new", "ep_ftp_new"):
        out[f"{k}_layer"] = out[k]
        out[k] = out[k] - gap
    out["nii_gap_layer_exact"] = gap
    return out


def glide_range(lt: dict) -> dict:
    """(family, side) -> (lo, hi): client families between the plan and the long-term target (from the plan)."""
    plan, tgt = {}, {}
    for src, dst in ((lt["weights_old"], plan), (lt["weights_new"], tgt)):
        for k, w in src.items():
            pc, side = k.split("|")
            if (side == "A" and _fam(pc) in LOAN_FAMS) or (side == "L" and _fam(pc) in DEPOSIT_FAMS):
                dst[(_fam(pc), side)] = dst.get((_fam(pc), side), 0.0) + w
    # the tolerance only extends beyond the target, never back past the plan: a family moves towards the
    # target or stays at the plan (a family the target leaves at the plan may move by the tolerance)
    def rng(p, t):
        if abs(t - p) < 1e-4:                           # below 0.01% of assets: no direction
            return max(0.0, p - GLIDE_TOL), p + GLIDE_TOL
        return (p, t + GLIDE_TOL) if t > p else (max(0.0, t - GLIDE_TOL), p)
    return {k: rng(plan[k], tgt[k]) for k in plan}


def main() -> None:
    out_dir = BANK.output_dir("optimize_prep")
    params = BalanceSheetParams.load(os.path.join(out_dir, "product_params.npz"))
    cr = CohortRates.load(os.path.join(out_dir, "product_params.npz"))
    cfg0 = load_optimizer_config(os.path.join(_ROOT, "bs_optimization", "input", "optimizer_config.xlsx"), params)
    os.makedirs(OUT_DIR, exist_ok=True)
    runs = []
    EVAL = (-99.0, -99.0)                # baseline books: report every curve, constrain nothing
    if BANK_ID == BASE_BANK:
        runs.append(("current", dict(max_shift=0.0, floors=(-30, -30), floor_kind="none", allowance_years=LT_YEARS, no_swaps=True, robust=EVAL)))
        # today's structure with three years of run-off replaced like for like at today's pricing: the fair
        # baseline for the long-term target (which also replaces the run-off at today's pricing)
        runs.append(("current_renewed", dict(max_shift=0.05, floors=(-99, -99), floor_kind="renewed", allowance_years=LT_YEARS, no_swaps=True, robust=EVAL)))
        # the same renewed structure brought back to the appetite with swaps alone (on top of the continued
        # 7Y programme): how many swaps today's structure needs, against the long-term target's
        runs.append(("renewed_swaps", dict(max_shift=0.05, floors=APPETITE, floor_kind="renewed", allowance_years=LT_YEARS, robust=None)))
        runs.append(("lt_profit_only", dict(max_shift=0.05, floors=APPETITE, floor_kind="today3y", allowance_years=LT_YEARS, robust=None)))
        runs.append(("lt_target", dict(max_shift=0.05, floors=APPETITE, floor_kind="today3y", allowance_years=LT_YEARS)))
        runs.append(("lt_same_risk", dict(max_shift=0.05, floors=None, floor_kind="today3y", allowance_years=LT_YEARS)))
        for f in EVE_SWEEP:
            runs.append((f"lt_target_eve{-f:g}", dict(max_shift=0.05, floors=(f, APPETITE[1]), floor_kind="today3y", allowance_years=LT_YEARS)))
        # covered bond held at today's balance (run-off reissued): how much of the target's EP is the smaller issuance
        runs.append(("force_cb", dict(max_shift=0.05, floors=APPETITE, floor_kind="today3y", allowance_years=LT_YEARS,
                                      force={("500", "L"): (0.0, 1e12)})))
    else:
        runs.append(("plan", dict(max_shift=0.0, floors=(-30, -30), floor_kind="none", allowance_years=1, no_swaps=True, robust=EVAL)))
        # the first step and the target built from the plan may replace the plan's new 7Y/3M payers by new
        # payers on a 6M EURIBOR leg (2026-10-28: the optimiser then drops all of them for 10Y/6M at the same EP;
        # keeping the plan's 7Y/3M book was a choice, not the optimum)
        SO = dict(swap_reset_months=(6,))
        # 2026-10-28: the proposal family trades the plan's own notional (167m) on 10Y/6M instead of 7Y/3M, the
        # balance sheet does the rest; the swap routes below keep the notional free (optimum ~247m)
        _pn = planned_swaps().get("SKIP_PLAN", (0, 0.0))[1]
        SF = dict(SO, fixed_swaps={"NEW_10Y_6M": _pn, "SKIP_PLAN": _pn})   # SKIP_PLAN = plan payers left out
        runs.append(("lt_target", dict(max_shift=0.05, floors=APPETITE, floor_kind="plan", allowance_years=1, **SO)))
        runs.append(("lt_target_swaps", dict(max_shift=0.05, floors=APPETITE, floor_kind="plan", allowance_years=1)))
        runs.append(("proposal_profit_only", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1, robust=None, **SF)))
        runs.append(("alm_proposal", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1, **SF)))
        runs.append(("proposal_no_glide", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1, glide=False, **SF)))
        runs.append(("proposal_same_risk", dict(max_shift=0.025, floors=None, floor_kind="plan", allowance_years=1, **SF)))
        # the swap route: new swaps allowed (both legs give the same EP; the 6M leg carries less NII risk)
        runs.append(("proposal_with_swaps", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1, swap_reset_months=(6,))))
        runs.append(("proposal_3m_only", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1, swap_reset_months=(3,))))
        runs.append(("proposal_both_legs", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1)))
        for f in EVE_SWEEP:
            runs.append((f"alm_proposal_eve{-f:g}", dict(max_shift=0.025, floors=(f, APPETITE[1]), floor_kind="plan", allowance_years=1, **SF)))
        runs.append(("sens_acq_low", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1, acq_cost_scale=0.5, **SF)))
        runs.append(("sens_acq_high", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1, acq_cost_scale=2.0, **SF)))
        runs.append(("sens_concession_100", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1, concession_bps=100.0, **SF)))
        # the levers of the layer-by-layer analysis forced into the proposal: what each costs once the
        # optimiser has to take it (2026-10-28; Table 3 sizes cut to what the line and growth caps allow:
        # funding line moves by at most 2.5% of assets, floating mortgages grow by at most 1% of assets)
        runs.append(("force_td", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1,
                                      force={("706", "L"): (-1e12, -250e6)}, **SF)))
        runs.append(("force_mfloat", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1,
                                          force={("110", "A"): (107e6, 1e12)}, **SF)))
        runs.append(("force_sme", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1,
                                       force={("410", "A"): (70e6, 1e12)}, **SF)))
        runs.append(("force_cb", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1,
                                      force={("500", "L"): (0.0, 1e12)}, **SF)))
        runs.append(("force_cb_lt", dict(max_shift=0.05, floors=APPETITE, floor_kind="plan", allowance_years=1,
                                         force={("500", "L"): (0.0, 1e12)}, **SO)))
        # weaker deposits: the balance sheet alone cannot reach the appetite -- swaps allowed
        runs.append(("sens_nmd_weaker", dict(max_shift=0.025, floors=APPETITE, floor_kind="plan", allowance_years=1, eve_scale=NMD_STRESS, swap_reset_months=(6,))))
    results, objs = {}, {}
    only = [t for t in os.environ.get("BTRM_ONLY", "").split(",") if t]   # rerun these tags, keep the rest
    if only:
        assert all(kw["floors"] is not None for t, kw in runs if t in only), "same-risk runs need the full set"
        runs = [(t, kw) for t, kw in runs if t in only]
        results = json.load(open(os.path.join(OUT_DIR, "btrm_article_runs.json")))
        objs = pickle.load(open(os.path.join(OUT_DIR, "btrm_article_runs.pkl"), "rb"))
    base_sot = None
    for tag, kw in runs:
        kw = dict(kw)
        if kw.pop("glide", True) and BANK_ID != BASE_BANK and kw["max_shift"] == 0.025:
            kw["family_range"] = glide_range(results["lt_target"])
        force = kw.pop("force", None)       # (family, side) -> (lo, hi) change against the plan, EUR
        if force:
            base = results["plan"] if "plan" in results else results["current"]
            pw, ta_p = {}, base["total_assets"]
            for k, w in base["weights_new"].items():
                pc, side = k.split("|")
                pw[(_fam(pc), side)] = pw.get((_fam(pc), side), 0.0) + w
            kw["family_range"] = {**(kw.get("family_range") or {}),
                                  **{k: (max(0.0, pw[k] + lo / ta_p), pw[k] + hi / ta_p) for k, (lo, hi) in force.items()}}
        if kw["floors"] is None:            # same-risk: floors at the starting portfolio's own worst values
            kw["floors"] = (min(base_sot["eve"].values()), min(base_sot["nii"].values()))
        out, r = run(tag, params=params, cr=cr, cfg0=cfg0, **kw)
        results[tag] = out; objs[tag] = r
        if base_sot is None:
            base_sot = dict(eve=r.sot_eve, nii=r.sot_nii)
    with open(os.path.join(OUT_DIR, "btrm_article_runs.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    with open(os.path.join(OUT_DIR, "btrm_article_runs.pkl"), "wb") as f:
        pickle.dump(objs, f)
    print("saved", os.path.join(OUT_DIR, "btrm_article_runs.json"))


if __name__ == "__main__":
    main()
