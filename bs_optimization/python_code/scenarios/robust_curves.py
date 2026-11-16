"""robust_curves.py
=================
Alternative base curves for robust balance-sheet optimisation, and the full-engine per-product
EVE/NII sensitivities on each of them.

    LABBANK_BANK=eur    python robust_curves.py     # long-term target: 4 historical rate regimes
    LABBANK_BANK=eur_y1 python robust_curves.py     # one-year proposal: the plan's curve -/+25bp

Curves (Nelson-Siegel betas, the same model as the Monte Carlo):
  - eur (long-term target): the EUR history 2005-today split into five equal-time buckets by the 3M
    rate; the two lowest buckets are both the negative-rate era and are merged. Each regime curve is the
    average NS curve of its bucket; its EP weight is its share of history (0.4 / 0.2 / 0.2 / 0.2).
  - eur_y1 (one-year proposal): the base curve shifted in parallel by -25bp and +25bp (weight 0.5
    each; 2026-10-30, was the 1Y forward + simulated 10th/90th-percentile curves). The optimiser adds
    curve k minus today's fit to the plan's exact coefficients, so the shift applies to the plan's curve.

Sensitivities: sandbox/build_scenario_curves.build_hyp_irrbb_metrics_exact (production re-pricing of
the existing run-off / 12M cash flows, EBA shocks and floors) on each curve AND on today's NS fit; the
optimiser adds the difference (curve k minus today's fit) to its exact today coefficients, so today's
numbers stay exact and the fit error of the NS model cancels.

Output: runs/<bank>/bs_optimization/robust_curves.npz
"""
from __future__ import annotations

import os as _o, sys as _s
_PY = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
for _g in ("optimizers", "scenarios", "hedging", "reporting"):
    if _o.path.join(_PY, _g) not in _s.path:
        _s.path.insert(0, _o.path.join(_PY, _g))
_ROOT = _o.path.normpath(_o.path.join(_PY, "..", ".."))
for _p in (_ROOT, _o.path.join(_ROOT, "optimize_prep", "python_code"), _o.path.join(_ROOT, "sandbox")):
    if _p not in _s.path:
        _s.path.insert(0, _p)

import os
import time

import numpy as np

import ns_curve_model as nsm
import anchor_sot_exact as ase
import swap_ladder as sl
from eba_shock_curves import get_shock_params
from labbank_common.bank_config import get_bank, home_ccy

BANK_ID = os.environ.get("LABBANK_BANK", "eur")
OUT_PATH = os.path.join(str(get_bank(BANK_ID).output_dir("bs_optimization")), "robust_curves.npz")
N_MONTHS = 360


def _ns_ts():
    return nsm.fit_ns_timeseries(nsm.load_historical_panel(), nsm.DIEBOLD_LI_TAU)


def today_beta(ts=None) -> np.ndarray:
    ts = _ns_ts() if ts is None else ts
    return ts.asof(ase.REPORT_DATE)[["beta0", "beta1", "beta2"]].to_numpy(float)


def history_regimes(ts=None) -> list[dict]:
    """Four regimes from the 3M-rate quintiles of the whole history (lowest two merged)."""
    ts = _ns_ts() if ts is None else ts
    B = ts[["beta0", "beta1", "beta2"]].to_numpy(float)
    r3 = B @ nsm.ns_design_matrix(np.array([0.25]), nsm.DIEBOLD_LI_TAU).T[:, 0] / 100.0
    q = np.quantile(r3, [0, .2, .4, .6, .8, 1])
    groups = [("negative", (0, 2), 0.4), ("low", (2, 3), 0.2), ("normal", (3, 4), 0.2), ("high", (4, 5), 0.2)]
    out = []
    for name, (a, b), w in groups:
        m = (r3 >= q[a]) & (r3 <= q[b])
        out.append(dict(name=name, beta=B[m].mean(0), weight=w))
    return out


def _fit_ns(t_years: np.ndarray, zero_pct: np.ndarray) -> np.ndarray:
    X = nsm.ns_design_matrix(t_years, nsm.DIEBOLD_LI_TAU)
    return np.linalg.lstsq(X, zero_pct * 100.0, rcond=None)[0]     # betas in bps, as the NS panel


PLAN_SHIFT_BP = 25.0


def plan_curves() -> list[dict]:
    """Today's NS curve shifted in parallel by -/+PLAN_SHIFT_BP (beta0 is the level, in bps)."""
    b0 = today_beta()
    out = []
    for name, sgn in (("down", -1.0), ("up", 1.0)):
        b = b0.copy()
        b[0] += sgn * PLAN_SHIFT_BP
        out.append(dict(name=name, beta=b, weight=0.5))
    return out


def curve_set() -> list[dict]:
    return history_regimes() if BANK_ID == "eur" else plan_curves()


def _monthly_fwd(beta: np.ndarray) -> np.ndarray:
    disc = sl._ns_disc_curve_monthly(np.asarray(beta, float), "base", get_shock_params(home_ccy()))[:N_MONTHS]
    logd = np.log(np.concatenate([[1.0], disc]))
    return -np.diff(logd) * 12.0


def zero_rates(beta) -> dict:
    X = nsm.ns_design_matrix(np.array([0.25, 2.0, 10.0]), nsm.DIEBOLD_LI_TAU)
    z = X @ np.asarray(beta, float) / 100.0
    return {"3M": float(z[0]), "2Y": float(z[1]), "10Y": float(z[2])}


def load_robust(pm, params, ep_lambda: float, eve_floor: float, nii_floor: float, path: str = OUT_PATH) -> dict:
    """Per-product differences (curve k minus today's fit) per EUR of balance, for the optimiser.

    d_eve / d_nii: (K, n_prod, n_shock) change of the shock deltas; d_level: (K, n_prod) change of
    the 12M base NII. A product with no balance today (a new variant) takes its family's main
    product. Cohort values are product-level (broadcast), so the balance-weighted mean is exact."""
    z = np.load(path, allow_pickle=True)
    if not np.array_equal(z["cohort_id"], params.cohort_id):
        raise ValueError(f"{path} was built for another product_params.npz -- rerun robust_curves.py")
    names = [str(s) for s in z["names"]]
    i0 = names.index("today_fit")
    ks = [i for i in range(len(names)) if i != i0]
    def per_prod(a):                                    # (n_cohort, ...) -> (n_prod, ...), _ProductMap.per_prod
        return np.tensordot(pm.W, a, axes=(1, 0))
    fam_main = {}
    for j, (pc, side) in enumerate(pm.products):
        k = (str(pc)[:3], side)
        if k not in fam_main or pm.base_prod_w[j] > pm.base_prod_w[fam_main[k]]:
            fam_main[k] = j
    own = z["own_unit"] if "own_unit" in z.files else np.zeros(len(params.cohort_id), dtype=bool)
    own_prod = np.zeros(pm.n_prod, dtype=bool)
    own_prod[pm.cohort_to_prod[own]] = True       # priced on its own schedule on every curve
    src = np.array([j if (pm.base_prod_w[j] > 0 or own_prod[j]) else fam_main[(str(pc)[:3], side)]
                    for j, (pc, side) in enumerate(pm.products)])
    out = dict(names=[names[i] for i in ks], betas=z["betas"][ks], beta_fit=z["betas"][i0],
               weights=z["weights"][ks], shock_ids=[str(s) for s in z["hyp_shock_ids"]],
               ep_lambda=ep_lambda, eve_floor=eve_floor, nii_floor=nii_floor)
    for key, arr in (("d_eve", "hyp_delta_eve_unit"), ("d_nii", "hyp_delta_nii_unit"), ("d_level", "hyp_nii_unit_rate")):
        a = z[arr]
        out[key] = np.stack([per_prod(a[k] - a[i0])[src] for k in ks])
    return out


def main() -> None:
    from bs_vector import BalanceSheetParams
    from labbank_common.bank_config import get_bank
    import build_scenario_curves as bsc
    params = BalanceSheetParams.load(str(get_bank().output_dir("optimize_prep") / "product_params.npz"))
    curves = [dict(name="today_fit", beta=today_beta(), weight=0.0)] + curve_set()
    for c in curves:
        print(f"  {c['name']:10s} weight {c['weight']:.2f}  " + "  ".join(f"{k} {v:5.2f}" for k, v in zero_rates(c["beta"]).items()))
    sc = {"fwd_rates": np.array([_monthly_fwd(c["beta"]) for c in curves]),
          "scenario_ids": np.array([c["name"] for c in curves], dtype=object),
          "n_months": np.array([N_MONTHS])}
    t0 = time.time()
    disc_m = []
    m = bsc.build_hyp_irrbb_metrics_exact(params, sc, str(params.report_date), disc_m_out=disc_m)
    print(f"  repriced {len(curves)} curves in {time.time() - t0:.0f}s")
    # new-business / catalogue units have no cash flows in the book: price their own contractual schedule
    # (today's coupon) on each curve, so they do not borrow the existing book's profile
    import catalogue_params as cp
    from eba_shock_curves import ALL_SCENARIO_IDS
    from labbank_common.bank_config import get_engine
    specs = cp.unit_specs(params, get_bank(), get_engine(get_bank()))
    shock_ids = [str(s) for s in m["hyp_shock_ids"]]
    scen_all = ["base"] + shock_ids
    order = [list(ALL_SCENARIO_IDS).index(s) for s in scen_all]
    own = np.zeros(len(params.cohort_id), dtype=bool)
    for k in range(len(curves)):
        dm = disc_m[k][order]
        for spec in specs:
            i = spec[0]
            nb, dn, de = cp.unit_on_curve(spec, dm, scen_all)
            m["hyp_nii_unit_rate"][k, i] = nb
            m["hyp_delta_nii_unit"][k, i] = dn
            m["hyp_delta_eve_unit"][k, i] = de
            own[i] = True
    print(f"  {len(specs)} new-business units priced on their own schedule")
    np.savez(OUT_PATH, names=sc["scenario_ids"], betas=np.array([c["beta"] for c in curves]),
             weights=np.array([c["weight"] for c in curves]), cohort_id=params.cohort_id, own_unit=own, **m)
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
