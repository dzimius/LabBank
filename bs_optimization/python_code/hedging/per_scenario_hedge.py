"""per_scenario_hedge.py
=====================
"If a scenario breaches the internal IRRBB limits, how much new IRS do we
need?" -- answered separately for EACH of the 500 Monte Carlo curves.

The other MC500 optimizers find ONE swap portfolio for all 500 curves at
once. This module instead treats every curve as its own possible future:
the bank sees that curve, checks its 6 EBA SOT shocks against the internal
limits, and if any shock breaches it buys the MINIMUM total notional of new
par swaps that brings all 6 back inside the limits.

    for each curve c with a breach:
        min_x  sum_i x_i                                 (x_i >= 0, notional)
        s.t.   E0_eve[c,s] + sum_i a_eve[c,s,i] x_i / T1 * 100 >= eve_limit   for every shock s
               E0_nii[c,s] + sum_i a_nii[c,s,i] x_i / T1 * 100 >= nii_limit   for every shock s
               x_i <= instrument_cap

    instruments i = new par swaps, tenors HEDGE_TENORS x start HEDGE_STARTS_M
    (spot and forward-starting), pay-fixed AND receive-fixed (separate
    non-negative variables, so the LP picks the direction itself).

E0 (the book before the new hedge, existing IRS book included) is the EXACT
CF-engine reprice already cached by precompute_mc500_irrbb.py
(irrbb_mc500_<label>.xlsx). The new swaps are priced analytically against
the same curve + shock (same NS discount curves swap_ladder.py uses), so the
combined number is exact on the book side and analytic on the swap side --
a split of exact book and analytic swaps.

A curve whose LP is infeasible (no swap mix within the cap fixes it, e.g.
EVE needs pay-fixed while NII needs receive-fixed in the same curve) is
counted as "not hedgeable with IRS alone". With spot-starting swaps only
this happened on PLN (NII breaching in par_dn AND in the flattener): every
spot swap moves 1Y NII by the same 1Y-forward average, one NII lever only.
Forward-starting swaps add the missing levers.

The "unhedged" book (existing IRS book removed) is a stress variant: it
reprices the book without its swaps (~15 min, cached as
irrbb_mc500_baseline_noirs.xlsx) to show how much hedge the LP would choose
from scratch, compared with the IRS book the bank actually holds.

Usage
-----
    python per_scenario_hedge.py                      # current book (with existing IRS)
    python per_scenario_hedge.py --book unhedged      # without the existing IRS book
    python per_scenario_hedge.py --eve-limit -12 --nii-limit -4
"""
from __future__ import annotations

import os as _o, sys as _s   # python_code/<group>/ layout: every group folder on sys.path
_PY = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))   # bs_optimization/python_code
for _g in ("optimizers", "scenarios", "hedging", "reporting"):
    if _o.path.join(_PY, _g) not in _s.path:
        _s.path.insert(0, _o.path.join(_PY, _g))

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, os.path.normpath(os.path.join(_PY, "..", "..")))
from labbank_common.bank_config import get_bank, home_ccy  # noqa: E402

_HERE = _PY
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PREP_PY = os.path.join(_ROOT, "optimize_prep", "python_code")
for _p in (_HERE, _PREP_PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OPTPREP_OUT = str(get_bank().output_dir("optimize_prep"))
OUT_DIR     = str(get_bank().output_dir("bs_optimization"))

EVE_LIMIT_PCT_T1 = -12.0     # internal limit (regulatory SOT: -15)
NII_LIMIT_PCT_T1 = -4.0      # internal limit (regulatory SOT: -5)
HEDGE_TENORS     = (2, 3, 5, 7, 10)
HEDGE_STARTS_M   = (0, 3, 6, 9)  # spot and forward-starting (months from today)
HEDGE_SPREAD_BPS = 5.0       # dealer spread against the bank, per year of notional -- hedge cost only
INSTRUMENT_CAP_PCT_TA = 0.25 # max notional per instrument, fraction of total assets
SCEN_ORDER = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]
NII_SCENS_ALL = tuple(SCEN_ORDER)          # LabBank reports apply the NII floor to all 6 shocks
NII_SCENS_PAR = ("par_up", "par_dn")       # EBA NII SOT itself uses only the two parallel shocks

BOOKS = {
    # book name -> (cache label, include existing IRS book)
    "current":  ("baseline",       True),
    "unhedged": ("baseline_noirs", False),
}


def forward_shift_betas(horizon_months: int = 24) -> np.ndarray:
    """(3,) NS-factor change from today's curve to today's FORWARD curve `horizon_months` ahead: zero rates
    from the base forward tensor (curve_tensors.npz, the curve the CF engine prices on) at today and at the
    horizon, each fitted with the same Nelson-Siegel design, differenced -- the market's expected move."""
    import ns_curve_model as nsm
    from bs_vector import CurveTensors
    from labbank_common.bank_config import get_bank
    ct = CurveTensors.load(os.path.join(OPTPREP_OUT, "curve_tensors.npz"))
    f = ct.fwd_rates[ct.scenario_index("base"), ct.currency_index(get_bank().market.ccy)]
    tenors_y = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    X = nsm.ns_design_matrix(tenors_y, nsm.DIEBOLD_LI_TAU)

    def zero_bps(start):
        k = np.round(tenors_y * 12).astype(int)
        return np.array([((np.prod(1.0 + f[start:start + n] / 12.0) - 1.0) * 12.0 / n) * 1e4 for n in k])

    fit = lambda y: np.linalg.lstsq(X, y, rcond=None)[0]
    return fit(zero_bps(int(horizon_months))) - fit(zero_bps(0))


def build_mc500_bank(seed: int = 9157, n: int = 500, center: str | None = None):
    """The SAME 500 curves as precompute_mc500_irrbb.py (seed 42, 24m mean-reverting); another seed
    gives an independent sample from the same model (out-of-sample tests).
    center="forward": the same draws re-centred so that their mean after 24 months is today's forward
    curve 24 months ahead (forward_shift_betas) instead of the historical-mean drift of the mean-reverting
    model -- dispersion and co-movement unchanged (2026-10-28)."""
    import ns_curve_model as nsm
    from curve_scenario_bank import build_mc_scenario_bank
    import anchor_sot_exact as ase

    panel  = nsm.load_historical_panel()
    tau    = nsm.DIEBOLD_LI_TAU
    ns_ts  = nsm.fit_ns_timeseries(panel, tau)
    pca    = nsm.factor_changes_pca(ns_ts)
    today_beta = ns_ts.asof(ase.REPORT_DATE)[["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
    mr = nsm.fit_mean_reversion(ns_ts)
    factor_draws = nsm.simulate_mean_reverting_shocks(pca, mr, today_beta, n_scenarios=n,
                                                        horizon_months=24, seed=seed)
    if center == "forward":
        factor_draws = factor_draws - factor_draws.mean(axis=0) + forward_shift_betas(24)
    elif center is not None:
        raise ValueError(f"unknown MC centre {center!r}")
    return build_mc_scenario_bank(factor_draws, today_beta, ase.REPORT_DATE)


def _cache_path(label: str) -> str:
    return os.path.join(OUT_DIR, f"irrbb_mc500_{label}.xlsx")


def ensure_book_cache(book: str, mc_bank: pd.DataFrame) -> str:
    """Exact reprice of the book on the 500 curves unless a cache NEWER than
    the book (product_params.npz) exists. Returns the cache label to use.

    A stale shared cache (e.g. irrbb_mc500_baseline.xlsx left from an older
    balance-sheet draw) is not overwritten -- the other optimization caches
    and the report were built with it -- the fresh reprice goes to
    irrbb_mc500_<label>_fresh.xlsx instead."""
    from bank_reprice_at_weights import run_and_cache
    label, include_irs = BOOKS[book]
    book_mtime = os.path.getmtime(os.path.join(OPTPREP_OUT, "product_params.npz"))
    for lab in (label, f"{label}_fresh"):
        if os.path.exists(_cache_path(lab)) and os.path.getmtime(_cache_path(lab)) >= book_mtime:
            return lab
    if os.path.exists(_cache_path(label)):
        print(f"irrbb_mc500_{label}.xlsx is older than product_params.npz (book changed since) -- "
              f"repricing into irrbb_mc500_{label}_fresh.xlsx")
        label = f"{label}_fresh"
    from bs_vector import BalanceSheetParams
    from bs_optimizer import _ProductMap
    pm = _ProductMap(BalanceSheetParams.load(os.path.join(OPTPREP_OUT, "product_params.npz")))
    print(f"Exact reprice of '{book}' book, 500 curves x 6 shocks (~15 min)...")
    t0 = time.time()
    run_and_cache(label=label, products=pm.products, baseline_weights=pm.base_prod_w,
                  target_weights=pm.base_prod_w, include_irs=include_irs, bank=mc_bank,
                  out_dir=OUT_DIR, filename_prefix="irrbb_mc500")
    print(f"  done in {time.time()-t0:.0f}s")
    return label


def price_new_swaps(mc_bank: pd.DataFrame, tenors=HEDGE_TENORS, starts=HEDGE_STARTS_M,
                    spread_bps=HEDGE_SPREAD_BPS) -> dict:
    """Per unit notional, for every curve x shock x instrument: dEVE and dNII
    of a NEW par swap (fixed rate = that curve's base par rate, spread against
    the bank). Instruments = every (start, tenor) pair, pay-fixed then
    receive-fixed.

    Forward-starting swaps matter for NII: a spot-starting swap's 1Y NII only
    sees the average 1Y forward, identical for every tenor and direction, so
    spot swaps give the LP ONE NII lever. A swap starting in s months only
    earns the forwards from s to 12 months, which lets the hedge separate
    e.g. the flattener (short end up, 6-12M forwards down) from par_dn."""
    import swap_ladder as sl
    from eba_shock_curves import get_shock_params

    shock_params = get_shock_params(home_ccy())
    specs = [(st, T) for st in starts for T in tenors]
    n_c, n_t = len(mc_bank), len(specs)
    n_i = 2 * n_t
    d_eve = np.zeros((n_c, len(SCEN_ORDER), n_i))
    d_nii = np.zeros((n_c, len(SCEN_ORDER), n_i))
    par = np.zeros((n_c, n_t))
    freq = sl.FIXING_FREQ_M
    resets = [np.arange(st + freq, st + T * 12 + 1, freq, dtype=float) for st, T in specs]
    acc = freq / 12.0

    def legs(disc, st, T, rm):
        pv_float = sl._df_at_month(disc, st) - sl._df_at_month(disc, st + T * 12)
        annuity  = acc * sum(sl._df_at_month(disc, m) for m in rm)
        nii_flt  = sl._price_bucket_nii(disc, rm)                 # floating-leg income inside the 1Y horizon
        return pv_float, annuity, nii_flt

    for ci, (_, curve) in enumerate(mc_bank.iterrows()):
        beta = curve[["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
        base = sl._ns_disc_curve_monthly(beta, "base", shock_params)
        base_legs = [legs(base, st, T, rm) for (st, T), rm in zip(specs, resets)]
        par[ci] = [f / a for f, a, _ in base_legs]
        k_pay = par[ci] + spread_bps / 1e4
        k_rec = par[ci] - spread_bps / 1e4
        for si, s in enumerate(SCEN_ORDER):
            shocked = sl._ns_disc_curve_monthly(beta, s, shock_params)
            for ti, ((st, T), rm) in enumerate(zip(specs, resets)):
                f1, a1, n1 = legs(shocked, st, T, rm)
                f0, a0, n0 = base_legs[ti]
                d_float, d_ann, d_nii_flt = f1 - f0, a1 - a0, n1 - n0
                d_eve[ci, si, ti]       = d_float - k_pay[ti] * d_ann       # pay fixed, receive float
                d_eve[ci, si, n_t + ti] = -d_float + k_rec[ti] * d_ann      # receive fixed, pay float
                d_nii[ci, si, ti]       = d_nii_flt                         # fixed leg NII never moves
                d_nii[ci, si, n_t + ti] = -d_nii_flt
    label = [f"{T}Y" if st == 0 else f"{st}Mx{T}Y" for st, T in specs]
    names = [f"pay {l}" for l in label] + [f"rec {l}" for l in label]
    return dict(d_eve=d_eve, d_nii=d_nii, par=par, names=names)


LP_UNIT = 1e6          # LP variables in millions: in currency units the coefficients are ~1e-8
                       # per unit and HiGHS accepted "optimal" points that still breached NII
                       # by 0.03-0.08pp (found on PLN) -- never solve in raw currency units
LIMIT_TOL_PCT = 1e-6


def solve_curve(e_eve, e_nii, a_eve, a_nii, t1, eve_lim, nii_lim, cap):
    """Minimum-notional hedge for one curve. Returns (x in currency, status).

    Infeasible (no swap mix inside the cap meets both limits) -> best effort:
    minimise the total remaining shortfall below the limits, notional as a
    tie-breaker, status "not hedgeable"."""
    n_i = a_eve.shape[1]
    scale = 100.0 / t1 * LP_UNIT                       # % T1 per million notional
    # a @ x * scale >= lim - e   ->   -a*scale @ x <= e - lim
    A = np.vstack([-a_eve * scale, -a_nii * scale])
    b = np.concatenate([e_eve - eve_lim, e_nii - nii_lim])
    bounds = [(0, cap / LP_UNIT)] * n_i
    res = linprog(np.ones(n_i), A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if res.status == 0:
        x = res.x * LP_UNIT
        if (e_eve + a_eve @ x * 100 / t1).min() >= eve_lim - LIMIT_TOL_PCT and \
           (e_nii + a_nii @ x * 100 / t1).min() >= nii_lim - LIMIT_TOL_PCT:
            return x, "hedged"
    # best effort: slack s_k >= 0 per constraint, min sum(s) + tiny * notional
    n_k = A.shape[0]
    c = np.concatenate([np.full(n_i, 1e-6), np.ones(n_k)])
    A_s = np.hstack([A, -np.eye(n_k)])
    res = linprog(c, A_ub=A_s, b_ub=b, bounds=bounds + [(0, None)] * n_k, method="highs")
    x = res.x[:n_i] * LP_UNIT if res.status == 0 else np.zeros(n_i)
    return x, "not hedgeable"


def static_hedge(E_eve, E_nii, a_eve, a_nii, t1, eve_lim, nii_lim, cap):
    """ONE swap book decided today, before knowing which of the 500 curves
    comes: minimise the total shortfall below the limits summed over all
    curves x shocks (notional as a tie-breaker). Returns x in currency."""
    n_c, n_e = E_eve.shape
    n_n, n_i = E_nii.shape[1], a_eve.shape[2]
    scale = 100.0 / t1 * LP_UNIT
    A = np.vstack([-a_eve.reshape(n_c * n_e, n_i) * scale, -a_nii.reshape(n_c * n_n, n_i) * scale])
    b = np.concatenate([(E_eve - eve_lim).ravel(), (E_nii - nii_lim).ravel()])
    n_k = A.shape[0]
    import scipy.sparse as spa
    A_s = spa.hstack([spa.csr_matrix(A), -spa.eye(n_k)]).tocsr()
    c = np.concatenate([np.full(n_i, 1e-6), np.ones(n_k)])
    res = linprog(c, A_ub=A_s, b_ub=b, bounds=[(0, cap / LP_UNIT)] * n_i + [(0, None)] * n_k, method="highs")
    return res.x[:n_i] * LP_UNIT if res.status == 0 else np.zeros(n_i)


def run(book: str = "current", eve_limit: float = EVE_LIMIT_PCT_T1, nii_limit: float = NII_LIMIT_PCT_T1,
        save: bool = True, nii_scens: tuple = NII_SCENS_ALL) -> dict:
    from bank_reprice_at_weights import load_mc500_baseline
    import anchor_sot_exact as ase
    from bs_vector import BalanceSheetParams

    ccy = home_ccy()
    mc_bank = build_mc500_bank()
    label = ensure_book_cache(book, mc_bank)
    print(f"Book before hedge: irrbb_mc500_{label}.xlsx")
    E_eve, E_nii = load_mc500_baseline(label, out_dir=OUT_DIR)                # (500, 6) % T1
    nii_idx = [SCEN_ORDER.index(s) for s in nii_scens]
    E_nii = E_nii[:, nii_idx]                                                  # NII shocks under the limit
    t1 = ase.aee.irrbb_sql_setup.load_tier1_capital(ase.REPORT_DATE)
    ta = BalanceSheetParams.load(os.path.join(OPTPREP_OUT, "product_params.npz")).total_assets
    cap = INSTRUMENT_CAP_PCT_TA * ta

    print(f"Pricing new par swaps {HEDGE_TENORS} pay/receive on {len(mc_bank)} curves x 6 shocks...")
    sw = price_new_swaps(mc_bank)
    sw["d_nii"] = sw["d_nii"][:, nii_idx, :]
    n_c, n_i = E_eve.shape[0], len(sw["names"])

    breach_eve = E_eve.min(axis=1) < eve_limit
    breach_nii = E_nii.min(axis=1) < nii_limit
    X = np.zeros((n_c, n_i))
    status = np.where(breach_eve | breach_nii, "", "no breach").astype(object)
    for c in np.where(breach_eve | breach_nii)[0]:
        X[c], status[c] = solve_curve(E_eve[c], E_nii[c], sw["d_eve"][c], sw["d_nii"][c],
                                       t1, eve_limit, nii_limit, cap)

    scale = 100.0 / t1
    H_eve = E_eve + np.einsum("csi,ci->cs", sw["d_eve"], X) * scale
    H_nii = E_nii + np.einsum("csi,ci->cs", sw["d_nii"], X) * scale
    notional = X.sum(axis=1)
    hedged = status == "hedged"
    unhedgeable = status == "not hedgeable"

    # one book for all curves, decided before the curve is known
    x_static = static_hedge(E_eve, E_nii, sw["d_eve"], sw["d_nii"], t1, eve_limit, nii_limit, cap)
    S_eve = E_eve + np.einsum("csi,i->cs", sw["d_eve"], x_static) * scale
    S_nii = E_nii + np.einsum("csi,i->cs", sw["d_nii"], x_static) * scale
    static_breach = (S_eve.min(1) < eve_limit - LIMIT_TOL_PCT) | (S_nii.min(1) < nii_limit - LIMIT_TOL_PCT)

    per_curve = pd.DataFrame({
        "curve": np.arange(n_c), "status": status,
        "worst_eve_before": E_eve.min(1), "worst_eve_scen": np.array(SCEN_ORDER)[E_eve.argmin(1)],
        "worst_nii_before": E_nii.min(1), "worst_nii_scen": np.array(nii_scens)[E_nii.argmin(1)],
        "worst_eve_after": H_eve.min(1), "worst_nii_after": H_nii.min(1),
        "hedge_notional": notional,
        "static_worst_eve": S_eve.min(1), "static_worst_nii": S_nii.min(1), "static_breach": static_breach,
        **{f"notional_{n.replace(' ', '_')}": X[:, i] for i, n in enumerate(sw["names"])},
    })

    def _q(v, p):
        return float(np.percentile(v, p)) if len(v) else 0.0

    nb = int((breach_eve | breach_nii).sum())
    hn = notional[hedged]
    summary = pd.DataFrame([
        ("Curves (500 MC x 6 EBA shocks each)", n_c),
        ("NII limit applied to shocks", ", ".join(nii_scens)),
        (f"Breach before hedge: EVE < {eve_limit:g}% T1", int(breach_eve.sum())),
        (f"Breach before hedge: NII < {nii_limit:g}% T1", int(breach_nii.sum())),
        ("Breach before hedge: either", nb),
        ("Hedged back inside limits", int(hedged.sum())),
        ("Not hedgeable with IRS alone", int((status == "not hedgeable").sum())),
        ("Breach after hedge (best effort where not hedgeable)",
         int(((H_eve.min(1) < eve_limit - LIMIT_TOL_PCT) | (H_nii.min(1) < nii_limit - LIMIT_TOL_PCT)).sum())),
        (f"Avg hedge notional, all 500 curves (M {ccy})", notional[hedged].sum() / n_c / 1e6),
        (f"Avg hedge notional, breaching curves (M {ccy})", hn.mean() / 1e6 if len(hn) else 0.0),
        (f"P95 hedge notional, breaching curves (M {ccy})", _q(hn, 95) / 1e6),
        (f"Max hedge notional (M {ccy})", hn.max() / 1e6 if len(hn) else 0.0),
        (f"Hedge notional, % of total assets (avg, breaching)", hn.mean() / ta * 100 if len(hn) else 0.0),
        (f"Spread cost of avg hedge, {HEDGE_SPREAD_BPS:g} bp/yr (M {ccy}/yr)",
         (hn.mean() if len(hn) else 0.0) * HEDGE_SPREAD_BPS / 1e4 / 1e6),
        (f"Avg best-effort notional, not-hedgeable curves (M {ccy})",
         notional[unhedgeable].mean() / 1e6 if unhedgeable.any() else 0.0),
        ("ONE book for all curves: breaches", int(static_breach.sum())),
        (f"ONE book for all curves: notional (M {ccy})", x_static.sum() / 1e6),
        ("Worst EVE before / after (% T1)", f"{E_eve.min():.2f} / {H_eve.min():.2f}"),
        ("Worst NII before / after (% T1)", f"{E_nii.min():.2f} / {H_nii.min():.2f}"),
    ], columns=["measure", "value"])

    by_instr = pd.DataFrame({
        "instrument": sw["names"],
        f"one_book_notional_M_{ccy}": x_static / 1e6,
        "curves_using_it": (X[hedged] > 1.0).sum(axis=0),
        f"avg_notional_M_{ccy}_when_hedging": X[hedged].mean(axis=0) / 1e6 if hedged.any() else 0.0,
    })

    utilisation = pd.DataFrame({
        "measure": ["EVE", "NII"],
        "limit_pct_t1": [eve_limit, nii_limit],
        "worst_pct_t1": [E_eve.min(), E_nii.min()],
        "p5_of_worst_pct_t1": [np.percentile(E_eve.min(1), 5), np.percentile(E_nii.min(1), 5)],
        "median_worst_pct_t1": [np.median(E_eve.min(1)), np.median(E_nii.min(1))],
        "limit_used_at_worst_pct": [E_eve.min() / eve_limit * 100, E_nii.min() / nii_limit * 100],
    })

    out = dict(cache_label=label, x_static=x_static, E_eve=E_eve, E_nii=E_nii, H_eve=H_eve, H_nii=H_nii,
               S_eve=S_eve, S_nii=S_nii, X=X, names=sw["names"], par=sw["par"], nii_scens=nii_scens, summary=summary, by_instrument=by_instr, utilisation=utilisation, per_curve=per_curve,
               t1=t1, total_assets=ta)
    if save:
        suffix = "" if tuple(nii_scens) == NII_SCENS_ALL else "_niipar"
        path = os.path.join(OUT_DIR, f"hedge_mc500_{book}{suffix}.xlsx")
        with pd.ExcelWriter(path) as xw:
            summary.to_excel(xw, sheet_name="summary", index=False)
            utilisation.to_excel(xw, sheet_name="limit_utilisation", index=False)
            by_instr.to_excel(xw, sheet_name="by_instrument", index=False)
            per_curve.to_excel(xw, sheet_name="per_curve", index=False)
        print(f"Saved -> {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--book", choices=sorted(BOOKS), default="current")
    ap.add_argument("--eve-limit", type=float, default=EVE_LIMIT_PCT_T1)
    ap.add_argument("--nii-limit", type=float, default=NII_LIMIT_PCT_T1)
    ap.add_argument("--nii-par-only", action="store_true",
                    help="apply the NII limit to par_up/par_dn only (EBA NII SOT), not all 6 shocks")
    a = ap.parse_args()
    out = run(a.book, a.eve_limit, a.nii_limit,
              nii_scens=NII_SCENS_PAR if a.nii_par_only else NII_SCENS_ALL)
    pd.set_option("display.width", 200)
    print(f"\n{get_bank().name} -- book '{a.book}', T1 {out['t1']/1e6:,.1f}M {home_ccy()}")
    print(out["summary"].to_string(index=False))
    print()
    print(out["utilisation"].round(2).to_string(index=False))
    print()
    print(out["by_instrument"].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
