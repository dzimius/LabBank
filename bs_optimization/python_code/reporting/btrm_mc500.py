"""btrm_mc500.py
==============
Monte Carlo view of the BTRM paper's portfolios: each balance sheet is repriced on the first N_CURVES of 500
simulated EUR curves (per_scenario_hedge.build_mc500_bank, 24-month mean-reverting Nelson-Siegel factor
paths), with the 6 EBA shocks applied on top of every curve. The test sample uses its own seed (MC_SEED,
default TEST_SEED): the optimiser's robust curves (robust_curves.py) come from the seed-42 sample, so the
test is out of sample. BTRM_MC_SEED=42 reproduces the old in-sample test. For each curve the
minimum notional of new par swaps that brings EVE/NII back inside the risk appetite is solved
(per_scenario_hedge.solve_curve).

    LABBANK_BANK=eur    python btrm_mc500.py      # current + long-term target
    LABBANK_BANK=eur_y1 python btrm_mc500.py      # plan + ALM proposal

Portfolio weights come from btrm_article_runs.json. The exact reprice rescales the product's cash-flow
schedules (bank_reprice_at_weights), so a variant with no balance today (e.g. a 6M-reset mortgage or a
1-month term deposit) is repriced with its family's main product profile.
Profitability per curve: EP of the fixed balance sheet = its EP at today's curve + the change in 12M base NII
(full engine, book + the run's own new swaps) from today's curve to the simulated one; the other EP terms
(fees, EL, capital, OpEx, acquisition, concessions) do not depend on the curve.
Output: runs/<bank>/bs_optimization/btrm_mc500.json (test seed; btrm_mc500_seed<s>.json for any other seed)
(+ irrbb_mc500_btrm_<tag>.xlsx caches).
"""
from __future__ import annotations

import os as _o, sys as _s
_o.environ["LABBANK_NII_LEVEL_VS_TODAY"] = "1"     # per-curve NII level vs today's curve (EP per curve)
_PY = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
for _g in ("optimizers", "scenarios", "hedging", "reporting"):
    if _o.path.join(_PY, _g) not in _s.path:
        _s.path.insert(0, _o.path.join(_PY, _g))
_ROOT = _o.path.normpath(_o.path.join(_PY, "..", ".."))
for _p in (_ROOT, _o.path.join(_ROOT, "optimize_prep", "python_code")):
    if _p not in _s.path:
        _s.path.insert(0, _p)

import hashlib
import json
import os
import time

import numpy as np

from labbank_common.bank_config import get_bank
from bs_vector import BalanceSheetParams
from product_map import _ProductMap
import per_scenario_hedge as psh
from bank_reprice_at_weights import run_and_cache, load_cached, load_mc500_baseline
import anchor_sot_exact as ase

BANK_ID = os.environ.get("LABBANK_BANK", "eur")
OUT_DIR = psh.OUT_DIR
PORTFOLIOS = {"eur": ("lt_target", "current"), "eur_y1": ("alm_proposal", "plan")}
EVE_LIMIT, NII_LIMIT = -12.0, -4.2    # internal limit: binds on every curve other than today's (appetite -10 / -4)
N_WORKERS = 8          # Windows: 14 spawned workers each pickling ~650k CF rows failed with OSError 22
N_CURVES = int(os.environ.get("BTRM_MC_CURVES", 500))   # pandas ~2.4 s/curve on 8 workers
DESIGN_SEED, TEST_SEED = 9157, 31415    # robust_curves.py draws its curves from the design sample
MC_SEED = int(os.environ.get("BTRM_MC_SEED", TEST_SEED))
MC_CENTER = os.environ.get("BTRM_MC_CENTER") or None   # "forward": curves re-centred on today's forward curve
_N_SFX = "" if N_CURVES == 500 else f"_n{N_CURVES}"   # another sample size never overwrites the article's file
OUT_JSON = ((f"btrm_mc500{_N_SFX}.json" if MC_SEED == TEST_SEED else f"btrm_mc500_seed{MC_SEED}{_N_SFX}.json")
            if MC_CENTER is None
            else f"btrm_mc_{MC_CENTER}{N_CURVES}.json")
RUNS_NAME = os.environ.get("BTRM_RUNS", "btrm_article_runs")   # another runs file gets its own output names
_R_SFX = RUNS_NAME.removeprefix("btrm_article_runs")
OUT_JSON = OUT_JSON[:-5] + _R_SFX + ".json"
_SEED_SFX = "" if MC_SEED == DESIGN_SEED else f"_s{MC_SEED}"   # seed-42 caches keep their old names
_SEED_SFX += f"_{MC_CENTER}" if MC_CENTER else ""
# by_product (default): every product priced once per curve (mc_by_product.py), each portfolio = weighted sum;
# rescale: the whole book rescaled to the portfolio and repriced per portfolio (bank_reprice_at_weights.py)
METHOD = os.environ.get("BTRM_MC_METHOD", "by_product")
import mc_by_product as mbp


def by_product_cache(bank, kind: str) -> dict:
    """Per-product results on `bank` for this bank's current inputs, computed once and cached."""
    tag = mbp.cache_tag(bank, os.path.join(psh.OPTPREP_OUT, "product_params.npz"),
                        os.path.join(str(get_bank(BANK_ID).output_dir("irrbb_calc")), "eve_results.xlsx"))
    sfx = "" if kind == "today" else _SEED_SFX
    import mc_packed
    sfx += "_beh" if mc_packed.behaviour_on() else ""     # rate-dependent behaviour per curve: its own cache
    path = os.path.join(str(get_bank(BANK_ID).output_dir("bs_optimization")), f"mc_by_product_{kind}{sfx}_{tag}.npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    t0 = time.time()
    r = mbp.run_by_product(bank, n_workers=N_WORKERS if len(bank) > 1 else 1)
    np.savez(path, **r)
    print(f"  per-product cache {kind}: {len(bank)} curves in {time.time() - t0:.0f}s -> {path}")
    return r


def family_weights(pm: _ProductMap, weights: dict, own: set | None = None) -> np.ndarray:
    """Target weights on pm.products for the cash-flow reprice; variants with no baseline are folded into the
    family's main product, except the units in `own` (priced on their own schedule, see unit_overlay): they
    get weight 0 here."""
    own = own or set()
    w = np.array([weights.get(f"{pc}|{side}", b) for (pc, side), b in zip(pm.products, pm.base_prod_w)])
    fam = {}
    for j, (pc, side) in enumerate(pm.products):
        fam.setdefault((pc[:3], side), []).append(j)
    for idx in fam.values():
        main = max(idx, key=lambda j: pm.base_prod_w[j])
        for j in idx:
            if j in own:
                w[j] = 0.0
            elif j != main and pm.base_prod_w[j] <= 1e-12 and w[j] > 0:
                w[main] += w[j]; w[j] = 0.0
    return w


def unit_specs_by_product(pm: _ProductMap, params) -> dict:
    """product index -> catalogue_params.unit_specs entry: the new-business / catalogue units (no cash flows
    in the book) that are priced on their own contractual schedule on every curve."""
    import catalogue_params as cp
    from labbank_common.bank_config import get_engine
    return {int(pm.cohort_to_prod[s[0]]): s for s in cp.unit_specs(params, get_bank(), get_engine(get_bank()))}


def unit_overlay(bank, weights: dict, pm: _ProductMap, specs: dict, ta: float, t1: float, ref_beta):
    """(E, N, d_level): EVE / NII deltas (% T1, per curve x EBA shock) and the 12M base NII change vs today's
    curve (EUR, per curve) of the portfolio's new-business units, each priced on its own schedule at today's
    coupon on every curve (the same NS + EBA-shock curves as the swap overlay)."""
    import catalogue_params as cp
    import swap_ladder as sl
    from eba_shock_curves import get_shock_params
    from labbank_common.bank_config import home_ccy
    sp = get_shock_params(home_ccy())
    scen = ["base"] + list(psh.SCEN_ORDER)
    amt = {j: float(weights.get(f"{pm.products[j][0]}|{pm.products[j][1]}", 0.0)) * ta for j in specs}
    amt = {j: a for j, a in amt.items() if abs(a) > 1.0}
    E = np.zeros((len(bank), len(psh.SCEN_ORDER))); N = np.zeros_like(E); lvl = np.zeros(len(bank))
    if not amt:
        return E, N, lvl
    def curves(beta):
        return np.array([sl._ns_disc_curve_monthly(np.asarray(beta, float), s, sp) for s in scen])
    ref = curves(ref_beta)
    ref_lvl = sum(a * cp.unit_on_curve(specs[j], ref, scen)[0] for j, a in amt.items())
    for ci, (_, c) in enumerate(bank.iterrows()):
        dm = curves(c[["beta0", "beta1", "beta2"]].to_numpy(float))
        for j, a in amt.items():
            nb, dn, de = cp.unit_on_curve(specs[j], dm, scen)
            E[ci] += a * de / t1 * 100.0; N[ci] += a * dn / t1 * 100.0; lvl[ci] += a * nb
    return E, N, lvl - ref_lvl


def overlay_deltas(mc_bank, run: dict, t1: float):
    """(n_curves, n_scen) EVE / NII deltas, % T1, of the optimiser's swap overlay (new swaps at the
    fixed rate they were traded at, planned payers left out with the sign flipped), repriced on
    every simulated curve with the EBA shocks."""
    import swap_ladder as sl
    from eba_shock_curves import get_shock_params
    from labbank_common.bank_config import home_ccy
    sp = get_shock_params(home_ccy())
    legs, strips = [], []
    comps = run.get("swap_components") or {}
    for bid, n in run.get("swaps", {}).items():
        if bid in comps:                       # a strip on its own trade dates (dated_swaps)
            strips.append((-n if bid.startswith("SKIP") else n, comps[bid]))
            continue
        k = run["swap_fixed"][bid]
        sign = -1.0 if bid.startswith(("SKIP", "AGED")) else 1.0          # AGED: an existing swap leaving the book
        T = float(bid.split("_")[1].rstrip("Y")) if bid.startswith(("NEW", "AGED")) else float(run.get("skip_tenor", 7.0))
        f = 6 if bid.endswith("_6M") else sl.FIXING_FREQ_M
        legs.append((sign * n, k, np.arange(f, T * 12 + 1e-9, f, dtype=float), f))
    E = np.zeros((len(mc_bank), len(psh.SCEN_ORDER))); N = np.zeros_like(E)
    if not legs and not strips:
        return E, N
    import dated_swaps as ds
    for ci, (_, c) in enumerate(mc_bank.iterrows()):
        beta = c[["beta0", "beta1", "beta2"]].to_numpy(float)
        base = sl._ns_disc_curve_monthly(beta, "base", sp)
        b = [(sl._price_bucket_eve(base, rm, k, freq_m=f), sl._price_bucket_nii(base, rm, freq_m=f)) for _, k, rm, f in legs]
        for si, s in enumerate(psh.SCEN_ORDER):
            sh = sl._ns_disc_curve_monthly(beta, s, sp)
            for n, c_ in strips:
                E[ci, si] += n * (ds.eve(sh, c_) - ds.eve(base, c_)) / t1 * 100.0
                N[ci, si] += n * (ds.float_nii(sh, c_) - ds.float_nii(base, c_)) / t1 * 100.0
            for (n, k, rm, f), (e0, n0) in zip(legs, b):
                E[ci, si] += n * (sl._price_bucket_eve(sh, rm, k, freq_m=f) - e0) / t1 * 100.0
                # first floating period fixed at trade (as in the optimiser): only later periods move
                n1 = sl._price_bucket_nii(base, rm[:1], freq_m=f) + (sl._price_bucket_nii(sh, rm[1:], freq_m=f) if len(rm) > 1 else 0.0)
                N[ci, si] += n * (n1 - n0) / t1 * 100.0
    return E, N


def today_curve(mc_bank):
    """One-row scenario bank with today's NS betas (the reference of the per-curve base NII)."""
    import ns_curve_model as nsm
    ns_ts = nsm.fit_ns_timeseries(nsm.load_historical_panel(), nsm.DIEBOLD_LI_TAU)
    row = mc_bank.iloc[[0]].copy()
    row[["beta0", "beta1", "beta2"]] = ns_ts.asof(ase.REPORT_DATE)[["beta0", "beta1", "beta2"]].to_numpy(float)
    row["is_anchor"] = False; row["shift_idx"] = 0; row["shape"] = "today"; row["level"] = "today"
    return row.reset_index(drop=True)


def base_nii(label: str) -> np.ndarray:
    """Per-curve 12M base NII (EUR) of the book, row i = shift_idx i."""
    df = load_cached(label, out_dir=OUT_DIR, filename_prefix="irrbb_mc500")
    return df.groupby("shift_idx")["base_nii_amt"].first().sort_index().to_numpy(float)


def overlay_base_nii(bank, run: dict, ref_beta) -> np.ndarray:
    """Per-curve change (EUR) of the overlay swaps' 12M floating-leg NII vs today's curve; the first
    floating period is fixed at trade, the fixed leg does not move."""
    import swap_ladder as sl
    from eba_shock_curves import get_shock_params
    from labbank_common.bank_config import home_ccy
    sp = get_shock_params(home_ccy())
    out = np.zeros(len(bank))
    ref = sl._ns_disc_curve_monthly(np.asarray(ref_beta, float), "base", sp)
    import dated_swaps as ds
    comps = run.get("swap_components") or {}
    for bid, n in run.get("swaps", {}).items():
        sign = -1.0 if bid.startswith(("SKIP", "AGED")) else 1.0          # AGED: an existing swap leaving the book
        if bid in comps:
            n0 = ds.float_nii(ref, comps[bid])
            for ci, (_, c) in enumerate(bank.iterrows()):
                d = sl._ns_disc_curve_monthly(c[["beta0", "beta1", "beta2"]].to_numpy(float), "base", sp)
                out[ci] += sign * n * (ds.float_nii(d, comps[bid]) - n0)
            continue
        T = float(bid.split("_")[1].rstrip("Y")) if bid.startswith(("NEW", "AGED")) else float(run.get("skip_tenor", 7.0))
        f = 6 if bid.endswith("_6M") else sl.FIXING_FREQ_M
        rm = np.arange(f, T * 12 + 1e-9, f, dtype=float)[1:]
        if not len(rm):
            continue
        n0 = sl._price_bucket_nii(ref, rm, freq_m=f)
        for ci, (_, c) in enumerate(bank.iterrows()):
            d = sl._ns_disc_curve_monthly(c[["beta0", "beta1", "beta2"]].to_numpy(float), "base", sp)
            out[ci] += sign * n * (sl._price_bucket_nii(d, rm, freq_m=f) - n0)
    return out


def portfolio_label(tag: str, run: dict) -> str:
    """Cache label tied to the portfolio's content and to the bank's inputs (product profiles, last pipeline
    EVE run), so neither a changed optimisation nor a re-run pipeline with new models can reuse a stale cache."""
    key = [sorted(run["weights_new"].items()), sorted(run.get("swaps", {}).items())]
    if run.get("swap_components"):
        key.append(run["swap_components"])
    h = hashlib.sha1(json.dumps(key, default=float).encode())
    for p in (os.path.join(psh.OPTPREP_OUT, "product_params.npz"),
              os.path.join(str(get_bank(BANK_ID).output_dir("irrbb_calc")), "eve_results.xlsx")):
        with open(p, "rb") as f:
            h.update(f.read())
    h.update(b"plan-year split v2")                     # products move through their plan-year contracts
                                                        # (v1 composed from a cache that lost them)
    h = h.hexdigest()[:8]
    return f"btrm_v4_{tag}_{h}"


def main() -> None:
    runs = json.load(open(os.path.join(str(get_bank(BANK_ID).output_dir("bs_optimization")), f"{RUNS_NAME}.json")))
    params = BalanceSheetParams.load(os.path.join(psh.OPTPREP_OUT, "product_params.npz"))
    pm = _ProductMap(params)
    ta = params.total_assets
    # N <= 500: the first N of the 500-curve sample (the article's runs); N > 500: a sample of N from the same
    # model and seed -- the draws depend on n, so its first 500 are NOT the 500-curve sample
    mc_bank = (psh.build_mc500_bank(seed=MC_SEED, n=max(500, N_CURVES)).iloc[:N_CURVES].reset_index(drop=True)
               if MC_CENTER is None
               else psh.build_mc500_bank(seed=MC_SEED, n=N_CURVES, center=MC_CENTER))
    t1 = ase.aee.irrbb_sql_setup.load_tier1_capital(ase.REPORT_DATE)
    print("Pricing new par swaps on the 500 curves...")
    sw = psh.price_new_swaps(mc_bank)
    cap = psh.INSTRUMENT_CAP_PCT_TA * ta
    # curve descriptors: NS zero rates at 3M and 10Y of each simulated base curve
    from ns_curve_model import ns_design_matrix, DIEBOLD_LI_TAU
    X3 = ns_design_matrix(np.array([0.25, 10.0]), DIEBOLD_LI_TAU)
    betas = mc_bank[["beta0", "beta1", "beta2"]].to_numpy(float)
    z = betas @ X3.T / 100.0                          # bps -> %
    today = today_curve(mc_bank)
    specs = unit_specs_by_product(pm, params)
    out = {"seed": MC_SEED, "curve_3m": z[:, 0].tolist(), "curve_10y": z[:, 1].tolist(), "portfolios": {}}
    for tag in PORTFOLIOS[BANK_ID]:
        stem = portfolio_label(tag, runs[tag])
        label = f"{stem}_n{N_CURVES}{_SEED_SFX}"
        w = family_weights(pm, runs[tag]["weights_new"], own=set(specs))
        for lab, bank, nw in ((label, mc_bank, N_WORKERS), (f"{stem}_today", today, 1)):
            if not os.path.exists(os.path.join(OUT_DIR, f"irrbb_mc500_{lab}.xlsx")):
                t0 = time.time()
                if METHOD == "by_product":       # every product priced once per curve, portfolio = weighted sum
                    cache = by_product_cache(bank, "today" if lab.endswith("_today") else f"n{len(bank)}")
                    df = mbp.compose(cache, pm.products, pm.base_prod_w, w, t1, bank, min_w=pm.min_w)
                    df.to_excel(os.path.join(OUT_DIR, f"irrbb_mc500_{lab}.xlsx"), sheet_name="all_scenarios", index=False)
                else:                            # the whole book rescaled and repriced per portfolio
                    run_and_cache(label=lab, products=pm.products, baseline_weights=pm.base_prod_w,
                                  target_weights=w, include_irs=True, bank=bank, out_dir=OUT_DIR,
                                  filename_prefix="irrbb_mc500", n_workers=nw, min_w=pm.min_w)
                print(f"  {tag}: repriced {lab} in {time.time() - t0:.0f}s ({METHOD})")
        ref_beta = today[["beta0", "beta1", "beta2"]].iloc[0]
        u_eve, u_nii, u_lvl = unit_overlay(mc_bank, runs[tag]["weights_new"], pm, specs, ta, t1, ref_beta)
        d_nii = base_nii(label) - base_nii(f"{stem}_today")[0]
        d_nii = d_nii + overlay_base_nii(mc_bank, runs[tag], ref_beta) + u_lvl
        ep = (runs[tag]["ep_bank_new"] + d_nii) / 1e6
        E_eve, E_nii = load_mc500_baseline(label, out_dir=OUT_DIR)
        o_eve, o_nii = overlay_deltas(mc_bank, runs[tag], t1)
        E_eve, E_nii = E_eve + o_eve + u_eve, E_nii + o_nii + u_nii
        n_c = E_eve.shape[0]
        X = np.zeros((n_c, len(sw["names"])))
        status = np.array(["no breach"] * n_c, dtype=object)
        breach = (E_eve.min(1) < EVE_LIMIT) | (E_nii.min(1) < NII_LIMIT)
        for c in np.where(breach)[0]:
            X[c], status[c] = psh.solve_curve(E_eve[c], E_nii[c], sw["d_eve"][c], sw["d_nii"][c],
                                              t1, EVE_LIMIT, NII_LIMIT, cap)
        pay = np.array([n.startswith("pay") for n in sw["names"]])
        out["portfolios"][tag] = dict(
            worst_eve=E_eve.min(1).tolist(), worst_nii=E_nii.min(1).tolist(),
            breach_eve=int((E_eve.min(1) < EVE_LIMIT).sum()), breach_nii=int((E_nii.min(1) < NII_LIMIT).sum()),
            breach_any=int(breach.sum()), not_hedgeable=int((status == "not hedgeable").sum()),
            hedge_notional=X.sum(1).tolist(), hedge_pay=X[:, pay].sum(1).tolist(), hedge_receive=X[:, ~pay].sum(1).tolist(),
            p5_eve=float(np.percentile(E_eve.min(1), 5)), p5_nii=float(np.percentile(E_nii.min(1), 5)),
            median_eve=float(np.median(E_eve.min(1))), median_nii=float(np.median(E_nii.min(1))),
            ep=ep.tolist(), ep_today=runs[tag]["ep_bank_new"] / 1e6,
            ep_median=float(np.median(ep)), ep_p5=float(np.percentile(ep, 5)))
        p = out["portfolios"][tag]
        print(f"[{tag}] breaches EVE {p['breach_eve']} NII {p['breach_nii']} any {p['breach_any']} "
              f"(not hedgeable {p['not_hedgeable']})  avg hedge {np.mean(p['hedge_notional'])/1e6:.0f}M  "
              f"P5 worst EVE {p['p5_eve']:.2f} NII {p['p5_nii']:.2f}  EP today {p['ep_today']:.0f} "
              f"median {p['ep_median']:.0f} P5 {p['ep_p5']:.0f}")
        with open(os.path.join(str(get_bank(BANK_ID).output_dir("bs_optimization")), OUT_JSON), "w") as f:
            json.dump(out, f)                     # after every portfolio: partial results are usable
        print("saved", tag)


if __name__ == "__main__":
    main()
