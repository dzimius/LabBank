"""mc_by_product.py
=================
Monte Carlo repricing PER PRODUCT, once per curve; any portfolio is then a weighted sum.

bank_reprice_at_weights.run_full_bank_at_weights reprices the whole book on every curve x EBA shock for ONE
portfolio: it rescales each product's cash-flow schedules by target / baseline weight and sums the bank. Every
CF-amount column it rescales enters EVE and NII linearly (rates, floors, caps and discount factors are per
unit), so

    result(portfolio, curve, shock) = sum_p  scale_p x result(p, curve, shock)

with scale_p = target_p / baseline_p (1 for products the portfolio does not move, e.g. the swap book). This
module prices every (product_code, bs_side) once per curve with the SAME curve construction and the SAME
production pricing functions as anchor_sot_exact.reprice_anchor_both (row-level variants that keep the
product), caches the per-product results, and composes portfolios from the cache in milliseconds.

    LABBANK_BANK=eur_y1 python mc_by_product.py --curves 4 --workers 1     # timing + validation run
    LABBANK_BANK=eur python mc_by_product.py --curves 100 --backend numpy --compare pandas  # same curves, 2 backends

Backends (argument backend, else env LABBANK_MC_BACKEND, default "pandas"):
  pandas  the production pricing functions per curve and scenario, n_workers processes
  numpy   packed arrays on the CPU (mc_packed.py, irrbb_calc/python_code/packed_pricing.py)
Both return the same dict (keys, eve, nii, nii_level per curve x product x scenario); the packed one
reproduce pandas to ~1e-14 relative. t_price is the pricing time of the backend; the cache records it.

On a rolled-forward bank the plan-year contracts of each product are a key of their own ("pc|side|new",
bank_reprice_at_weights.mark_plan_year), so a product that moves through its plan-year business
(_ProductMap.split) composes as seasoned x 1 + plan year x (target - min_w) / (baseline - min_w).

Cache: runs/<bank>/bs_optimization/mc_by_product_<tag>.npz
  keys (K,) "pc|side" or "pc|side|new", scen (S,) EVE_SCENARIO_IDS incl. base, eve / nii (C, K, S) home currency: column base =
  the curve's own base level, other columns = shocked minus base; nii_level (C, K) 12M base NII measured
  against today's curve (LABBANK_NII_LEVEL_VS_TODAY); curve meta.
"""
from __future__ import annotations

import os as _o, sys as _s
_PY = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
for _g in ("optimizers", "scenarios", "hedging", "reporting"):
    if _o.path.join(_PY, _g) not in _s.path:
        _s.path.insert(0, _o.path.join(_PY, _g))

import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import anchor_sot_exact as ase
from labbank_common import fx
from labbank_common.bank_config import get_bank

_W = {}


def row_keys(df: pd.DataFrame) -> pd.Series:
    """"pc|side", or "pc|side|new" for a plan-year contract (column plan_year, mark_plan_year)."""
    k = df["product_code"].astype(str).str.zfill(4) + "|" + df["bs_side"].astype(str)
    if "plan_year" in df.columns:
        k = k.where(~df["plan_year"].to_numpy(bool), k + "|new")
    return k


def _keys_and_fx(df: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Row -> key index and home-currency factor (rows whose key is not in `keys` get -1)."""
    k = row_keys(df)
    pos = {key: i for i, key in enumerate(keys)}
    idx = k.map(pos).fillna(-1).astype(int).to_numpy()
    f = fx.to_home_series(df["currency"], get_bank(), ase.REPORT_DATE).to_numpy(dtype=float)
    return idx, f


def _by_key(df: pd.DataFrame, col: str, keys: list[str]) -> np.ndarray:
    # the pricing functions do not carry the input's plan_year column: mark the output again, or plan-year
    # rows fall under the seasoned key and products made only of plan-year contracts drop out (2026-10-27)
    if any(k.endswith("|new") for k in keys) and "plan_year" not in df.columns:
        from bank_reprice_at_weights import mark_plan_year
        df = mark_plan_year(df)
    idx, f = _keys_and_fx(df, keys)
    m = idx >= 0
    if not m[df[col].to_numpy(dtype=float) != 0].all():
        raise ValueError(f"{col}: rows with no key, e.g. {sorted(set(row_keys(df)[~m]))[:5]}")
    out = np.zeros(len(keys))
    np.add.at(out, idx[m], df[col].to_numpy(dtype=float)[m] * f[m])
    return out


def reprice_curve_by_product(all_beh_eve, all_beh_nii, caps_map, floors_map, coeff_a_map, coeff_b_map,
                             beta0, beta1, beta2, keys: list[str]):
    """One curve -> (eve (K, S), nii (K, S), nii_level (K,)): per product, column 0 = base level, others =
    shocked minus base; the same curves and pricing as anchor_sot_exact.reprice_anchor_both."""
    aee, nii_obj = ase.aee, ase.nii_obj
    curve_name = ase.DISC_CURVE_MAP[ase.CURRENCY]
    mkt_df = aee.ns_beta_to_mkt_df(beta0, beta1, beta2, curve_name=curve_name)
    shocked_df = ase.build_all_shocked_curves(mkt_df, ase.REPORT_DATE, ase.CURRENCY, nii_floor_rate=None,
                                              base_floor_bps_fn=ase.default_realistic_base_floor_bps)
    base_disc = aee.with_foreign(aee._in_memory_shocked_disc_series(shocked_df, "base", curve_name, ase.REPORT_DATE), "base")
    S = len(ase.EVE_SCENARIO_IDS)
    eve = np.zeros((len(keys), S)); nii = np.zeros((len(keys), S))
    maps = dict(caps_map=caps_map, floors_map=floors_map, coeff_a_map=coeff_a_map, coeff_b_map=coeff_b_map)
    for j, sid in enumerate(ase.EVE_SCENARIO_IDS):
        disc = base_disc if sid == "base" else aee.with_foreign(
            aee._in_memory_shocked_disc_series(shocked_df, sid, curve_name, ase.REPORT_DATE), sid)
        es = aee.eve_obj.compute_eve_shocked_schedule(all_beh_eve, disc, ase.DISC_CURVE_MAP, sid,
                                                      eve_floor_rate=float("-inf"), **maps)
        eve[:, j] = _by_key(es, "pv_total", keys)
        if sid == "base":
            ns = nii_obj.compute_nii_base_schedule(all_beh_nii, ase.REPORT_DATE, ase.NII_HORIZON_YF, scenario_id="base",
                                                   base_disc_df=base_disc, disc_curve_map=ase.DISC_CURVE_MAP, **maps)
        else:
            ns = nii_obj.compute_nii_shocked_schedule(all_beh_nii, disc, ase.DISC_CURVE_MAP, ase.REPORT_DATE, sid,
                                                      ase.NII_HORIZON_YF, ase.NII_FLOOR, base_disc_df=base_disc, **maps)
        nii[:, j] = _by_key(ns, "nii_total", keys)
    eve[:, 1:] -= eve[:, :1]
    nii[:, 1:] -= nii[:, :1]
    lvl = nii[:, 0].copy()
    if ase.NII_LEVEL_VS_TODAY:
        ls = nii_obj.compute_nii_shocked_schedule(all_beh_nii, base_disc, ase.DISC_CURVE_MAP, ase.REPORT_DATE, "base",
                                                  ase.NII_HORIZON_YF, ase.NII_FLOOR, base_disc_df=ase.today_base_disc(),
                                                  **maps)
        lvl = _by_key(ls, "nii_total", keys)
    return eve, nii, lvl


def _init(eve, nii, maps, keys):
    _W.update(eve=eve, nii=nii, maps=maps, keys=keys)


def _work(i, beta):
    return i, reprice_curve_by_product(_W["eve"], _W["nii"], *_W["maps"], *beta, keys=_W["keys"])


def load_schedules(include_irs: bool = True):
    aee = ase.aee
    eve = aee.irrbb_sql_setup.load_all_beh_schedules(ase.REPORT_DATE)
    nii = aee.irrbb_sql_setup.load_beh_schedules(ase.REPORT_DATE, ase.HORIZON_END)
    if include_irs:
        for loader, args, name in ((aee.irrbb_sql_setup.load_all_swap_beh_schedules, (ase.REPORT_DATE,), "eve"),
                                   (aee.irrbb_sql_setup.load_swap_beh_schedules, (ase.REPORT_DATE, ase.HORIZON_END), "nii")):
            try:
                sw = loader(*args)
            except Exception:
                sw = pd.DataFrame()
            if not sw.empty:
                if name == "eve":
                    eve = pd.concat([eve, sw], ignore_index=True)
                else:
                    nii = pd.concat([nii, sw], ignore_index=True)
    from bank_reprice_at_weights import mark_plan_year
    return mark_plan_year(eve), mark_plan_year(nii)


def backend_name(backend: str | None = None) -> str:
    b = (backend or os.environ.get("LABBANK_MC_BACKEND") or "pandas").lower()
    import mc_packed
    if b not in mc_packed.BACKENDS:
        raise ValueError(f"MC backend {b!r}: one of {mc_packed.BACKENDS}")
    return b


def run_by_product(bank: pd.DataFrame, n_workers: int = 8, include_irs: bool = True,
                   backend: str | None = None, behaviour: bool | None = None, gamma_one: bool = False) -> dict:
    """Per-product results on every curve of `bank` (columns beta0/beta1/beta2 + meta). behaviour (else
    LABBANK_MC_BEHAVIOUR=1): rate-dependent CPR / TDRR / NMD per curve, packed backends only (mc_packed)."""
    backend = backend_name(backend)
    import mc_packed
    if mc_packed.behaviour_on(behaviour) and backend == "pandas":
        raise ValueError("rate-dependent behaviour per curve needs the packed backend (numpy)")
    t0 = time.time()
    eve_df, nii_df = load_schedules(include_irs)
    maps = ase.aee._load_rate_maps()
    keys = sorted(set(row_keys(eve_df).unique()) | set(row_keys(nii_df).unique()))
    t_load = time.time() - t0
    if backend != "pandas":
        import mc_packed
        r = mc_packed.run_packed(bank, eve_df, nii_df, keys, backend, behaviour, gamma_one)
        out = dict(keys=np.array(keys), scen=np.array(ase.EVE_SCENARIO_IDS), eve=r["eve"], nii=r["nii"],
                   nii_level=r["nii_level"], betas=bank[["beta0", "beta1", "beta2"]].to_numpy(float),
                   shift_idx=bank["shift_idx"].to_numpy(), n_rows=np.array([len(eve_df), len(nii_df)]),
                   t_load=np.array([t_load]), t_price=r["t_price"], t_prep=r["t_prep"], backend=np.array([backend]),
                   behaviour=r["behaviour"])
        for k in ("gamma", "gamma_blocks"):
            if k in r:
                out[k] = r[k]
        return out
    C, K, S = len(bank), len(keys), len(ase.EVE_SCENARIO_IDS)
    eve = np.zeros((C, K, S)); nii = np.zeros((C, K, S)); lvl = np.zeros((C, K))
    betas = bank[["beta0", "beta1", "beta2"]].to_numpy(float)
    t1 = time.time()
    if n_workers <= 1:
        for i, b in enumerate(betas):
            eve[i], nii[i], lvl[i] = reprice_curve_by_product(eve_df, nii_df, *maps, *b, keys=keys)
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init, initargs=(eve_df, nii_df, maps, keys)) as pool:
            futs = [pool.submit(_work, i, tuple(b)) for i, b in enumerate(betas)]
            for n, f in enumerate(as_completed(futs), 1):
                i, (e, ni, lv) = f.result()
                eve[i], nii[i], lvl[i] = e, ni, lv
                if n % 50 == 0 or n == C:
                    print(f"  [{n}/{C}] curves  ({time.time() - t1:.0f}s)")
    t_price = time.time() - t1
    return dict(keys=np.array(keys), scen=np.array(ase.EVE_SCENARIO_IDS), eve=eve, nii=nii, nii_level=lvl,
                betas=betas, shift_idx=bank["shift_idx"].to_numpy(), n_rows=np.array([len(eve_df), len(nii_df)]),
                t_load=np.array([t_load]), t_price=np.array([t_price]), backend=np.array(["pandas"]),
                behaviour=np.array([False]))


def cache_tag(bank: pd.DataFrame, params_path: str, eve_results_path: str) -> str:
    """Hash of the curve set and of the bank's inputs (product profiles, last pipeline EVE run)."""
    h = hashlib.sha1(np.ascontiguousarray(bank[["beta0", "beta1", "beta2"]].to_numpy(float)).tobytes())
    h.update(b"plan-year keys v2")                       # v1 lost the plan-year rows in the priced output
    for p in (params_path, eve_results_path):
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def compose(cache: dict, products, baseline_weights, target_weights, t1: float, bank: pd.DataFrame,
            min_w=None) -> pd.DataFrame:
    """Portfolio results from the per-product cache, in run_full_bank_at_weights' output schema.
    min_w: pm.min_w (products that move through their plan-year contracts), None = proportional."""
    from bank_reprice_at_weights import compute_product_scale_map
    sm = compute_product_scale_map(products, baseline_weights, target_weights, min_w)
    keys = [str(k) for k in cache["keys"]]
    scale = np.array([sm.get(tuple(k.split("|")), sm.get(tuple(k.split("|")[:2]), 1.0)) for k in keys])
    if np.isnan(scale).any():
        raise ValueError(f"scale NaN for {[k for k, s in zip(keys, scale) if np.isnan(s)]}")
    E = np.einsum("cks,k->cs", cache["eve"], scale)
    N = np.einsum("cks,k->cs", cache["nii"], scale)
    L = cache["nii_level"] @ scale
    scen = [str(s) for s in cache["scen"]]
    rows = []
    for c, (_, cv) in enumerate(bank.iterrows()):
        for j, s in enumerate(scen):
            if s == "base":
                continue
            e, n = E[c, j] / t1 * 100.0, N[c, j] / t1 * 100.0
            rows.append(dict(shape=cv["shape"], level=cv["level"], is_anchor=cv["is_anchor"], shift_idx=cv["shift_idx"],
                             anchor_date=cv.get("anchor_date"), scenario=s, delta_eve_pct_t1=e, delta_nii_pct_t1=n,
                             eve_sot_pass=e >= ase.SOT_EVE_FLOOR_PCT_T1, nii_sot_pass=n >= ase.SOT_NII_FLOOR_PCT_T1,
                             base_eve_amt=E[c, 0], base_nii_amt=L[c] if ase.NII_LEVEL_VS_TODAY else N[c, 0]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--backend", default=None, help="pandas / numpy (default LABBANK_MC_BACKEND or pandas)")
    ap.add_argument("--compare", default=None, help="a second backend priced on the same curves")
    ap.add_argument("--cache", default=None, help="an existing mc_by_product_*.npz: its curves, compared with its results")
    a = ap.parse_args()
    if a.cache:
        ref = dict(np.load(a.cache, allow_pickle=True))
        bank = pd.DataFrame(ref["betas"], columns=["beta0", "beta1", "beta2"]).assign(shift_idx=ref["shift_idx"])
        ref_name = f"cache {os.path.basename(a.cache)} ({str(ref.get('backend', ['pandas'])[0])})"
    else:
        import per_scenario_hedge as psh
        bank = psh.build_mc500_bank().iloc[:a.curves].reset_index(drop=True)
        ref, ref_name = None, None
    r = run_by_product(bank, n_workers=a.workers, backend=a.backend)
    b = str(r["backend"][0])
    print(f"{b}: load {r['t_load'][0]:.1f}s, price {r['t_price'][0]:.1f}s for {len(bank)} curves "
          f"({r['t_price'][0] / len(bank):.3f}s/curve); rows EVE {r['n_rows'][0]:,} NII {r['n_rows'][1]:,}")
    if a.compare:
        ref = run_by_product(bank, n_workers=a.workers, backend=a.compare)
        ref_name = f"{a.compare} ({ref['t_price'][0]:.1f}s)"
    if ref is not None:
        if list(ref["keys"]) != list(r["keys"]):
            raise SystemExit(f"product keys differ from {ref_name}")
        for k in ("eve", "nii", "nii_level"):
            d = np.abs(r[k] - ref[k])
            print(f"  {k:9s} vs {ref_name}: max |diff| {d.max():.3g} EUR, relative {d.max() / max(np.abs(ref[k]).max(), 1):.2e}")
        print(f"  speed-up vs {ref_name}: x{float(ref['t_price'][0]) / max(float(r['t_price'][0]), 1e-9):,.0f} (pricing)")
