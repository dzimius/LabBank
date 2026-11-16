"""bank_reprice_at_weights.py
=============================
Parameterizes anchor_sot_exact.py's exact 165-curve x 7-scenario repricing
by an ARBITRARY balance-sheet product weight vector, instead of always
using today's real SQL book. Reuses the exact SAME CF-schedule-based
pricing engine (eve_calc_objects.compute_eve_shocked, nii_calc_objects.
compute_nii_*_schedule) that anchor_sot_exact.py / hedge_optimizer.py
already use and have validated against production -- only the BALANCE
going into those schedules changes, never the pricing math itself.

Rescaling mechanism
--------------------
The optimizer (bs_optimizer.py/joint_optimizer.py)
works by moving PRODUCT-level
weights while preserving each product's baseline COHORT-level proportional
split (see _ProductMap.fraction) -- so a target weight vector is fully
characterized by one scale factor per (product_code, bs_side):

    scale[pc, side] = target_weight[pc, side] / baseline_weight[pc, side]

Applying that SAME scale factor to every SQL behavioral-schedule row for
that product (outstanding_bal, capital_pmt, prepayment_pmt, int_pmt --
NOT cf_yf/fwd_rt/base_df/eff_rate, which are per-unit rate/date quantities)
reproduces the exact-pricing-equivalent of "what if this product had a
different aggregate balance", without needing per-cohort/vintage
granularity in the SQL schedule at all.

Exception (rolled-forward bank, _ProductMap.split): a product with seasoned
contracts and plan-year new business moves only through the plan-year
contracts (schedules started after product_map.plan_year_start). Pass the
optimiser's min_w: those products' seasoned rows keep scale 1 and their
plan-year rows (mark_plan_year) get (target - min_w) / (baseline - min_w).

Parallelization
----------------
The 165 curves are fully independent (read-only shared inputs, no
cross-curve state) -- an embarrassingly parallel workload. Sequential:
~4s/curve x 165 curves ~= 11 minutes/book.

Default backend is stdlib concurrent.futures.ProcessPoolExecutor:
4.6x speedup (152s vs 693s on a 16-core machine), validated bit-identical
to the sequential path (max diff 8e-14, floating-point noise) against the
independently-cached load_today_book_165() ground truth.

A dask.distributed backend (backend="dask") also exists in this file. It
was built because the long-term goal is a codebase that scales from "a few
local cores today" to "a real multi-machine cluster later" without a
rewrite (PySpark was explicitly rejected for that role -- JVM startup lag
makes it overkill for local/small runs -- Dask's local Client() gives that
growth path for free by just pointing Client() at scheduler_address=...
later; run_full_bank_at_weights() itself doesn't change).

FOUND AND FIXED (2026-07-15) a real correctness bug during validation:
scattering caps_map/floors_map/coeff_a_map/coeff_b_map as FOUR SEPARATE
client.scatter(..., broadcast=True) calls silently mixed their VALUES --
e.g. floors_map['6000'] would come back as caps_map['6000']'s value.
Root-caused via a client.scatter()+immediate client.gather() round-trip on
just those four small dicts (proved the corruption happens at/before the
FIRST gather, not from later cross-contamination) -- a Dask key-collision
issue specific to scattering several small, similarly-shaped objects in
quick succession (not a documented Dask limitation as far as this session
found, but empirically reproducible and empirically fixed). Fix: bundle
ALL shared read-only inputs (both CF-schedule dataframes AND all four rate
maps) into ONE dict and make a SINGLE client.scatter() call for it, rather
than one scatter() per object -- see _reprice_one_curve_dask. Re-validated
after the fix with the same rigor as the process backend: full 165-curve
run, cross-checked against the independently-cached load_today_book_165()
-- max |delta_eve diff| 8.08e-14, max |delta_nii diff| 8.88e-16 (floating-
point-noise level, same order as the process backend's 7.99e-14/8.88e-16).
Wall time 225-247s (165/165 curves + ~90s Dask cluster startup), vs the
process backend's 152-155s -- the process backend remains faster on this
job size; Dask's advantage is the free path to a real cluster later via
scheduler_address=, not raw speed on a single machine today.

Meant to be run as a cached background step (see run_and_cache()), not
inline in the notebook.
"""
from __future__ import annotations

import os as _o, sys as _s   # python_code/<group>/ layout: every group folder on sys.path
_PY = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))   # bs_optimization/python_code
for _g in ("optimizers", "scenarios", "hedging", "reporting"):
    if _o.path.join(_PY, _g) not in _s.path:
        _s.path.insert(0, _o.path.join(_PY, _g))

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_HERE = _PY
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import anchor_sot_exact as ase
import sys as _sys
_sys.path.insert(0, os.path.normpath(os.path.join(_PY, "..", "..")))
from labbank_common.bank_config import home_ccy  # noqa: E402
from product_map import plan_year_start  # noqa: E402


_SCALE_COLS = ["outstanding_bal", "capital_pmt", "prepayment_pmt", "int_pmt"]

# ── ProcessPoolExecutor worker-process globals, populated once per process
#    by _init_worker() -- the validated, default parallel backend. ─────────
_W_EVE = None
_W_NII = None
_W_CAPS = None
_W_FLOORS = None
_W_COEFF_A = None
_W_COEFF_B = None


def _init_worker(all_beh_eve, all_beh_nii, caps_map, floors_map, coeff_a_map, coeff_b_map):
    """ProcessPoolExecutor initializer: runs ONCE per worker process (not per
    task), so the ~400-500K-row CF-schedule dataframes are pickled/sent across
    the process boundary once per worker, not once per curve."""
    global _W_EVE, _W_NII, _W_CAPS, _W_FLOORS, _W_COEFF_A, _W_COEFF_B
    _W_EVE, _W_NII = all_beh_eve, all_beh_nii
    _W_CAPS, _W_FLOORS = caps_map, floors_map
    _W_COEFF_A, _W_COEFF_B = coeff_a_map, coeff_b_map


def _reprice_one_curve_worker(curve: dict) -> tuple[dict, dict, dict]:
    """Runs in a ProcessPoolExecutor worker process against the globals
    _init_worker() set up. Returns (curve_meta, eve_out, nii_out)."""
    eve_out, nii_out = ase.reprice_anchor_both(
        _W_EVE, _W_NII, _W_CAPS, _W_FLOORS, _W_COEFF_A, _W_COEFF_B,
        curve["beta0"], curve["beta1"], curve["beta2"],
    )
    return curve, eve_out, nii_out


def _reprice_one_curve_dask(curve, shared_bundle):
    """Dask backend worker fn. `shared_bundle` is ONE scattered dict holding
    all_beh_eve/all_beh_nii/caps_map/floors_map/coeff_a_map/coeff_b_map
    together -- see module docstring's "Dask key-collision bug" section for
    why this MUST be a single bundled scatter, not one scatter() call per
    object (root-caused 2026-07-15: scattering several small, similarly-
    shaped dicts via separate broadcast=True calls in quick succession
    silently mixes their values -- confirmed via client.scatter/gather
    round-trip on caps_map/floors_map/coeff_a_map/coeff_b_map alone, fixed
    by bundling all 4 into one dict and scattering that single object)."""
    eve_out, nii_out = ase.reprice_anchor_both(
        shared_bundle["all_beh_eve"], shared_bundle["all_beh_nii"],
        shared_bundle["caps_map"], shared_bundle["floors_map"],
        shared_bundle["coeff_a_map"], shared_bundle["coeff_b_map"],
        curve["beta0"], curve["beta1"], curve["beta2"],
    )
    return curve, eve_out, nii_out


def compute_product_scale_map(
    products: list[tuple[str, str]],
    baseline_weights: np.ndarray,
    target_weights: np.ndarray,
    min_w: np.ndarray | None = None,
) -> dict[tuple[str, str], float]:
    """(product_code, bs_side) -> target/baseline weight ratio.

    baseline_weights/target_weights are aligned to `products` (e.g.
    pm.products with pm.base_prod_w and a target x vector) -- both
    expressed as fractions of total_assets, so the ratio is scale-
    invariant (the total_assets denominator cancels).

    min_w (pm.min_w): a product with min_w > 0 moves through its plan-year
    contracts -- key -> 1.0 for the seasoned rows and (pc, side, "new") ->
    (target - min_w) / (baseline - min_w) for the plan-year rows.
    """
    scale_map: dict[tuple[str, str], float] = {}
    for j, ((pc, side), b, t) in enumerate(zip(products, baseline_weights, target_weights)):
        key = (str(pc), str(side))
        m = float(min_w[j]) if min_w is not None else 0.0
        if m > 0.0 and b - m > 1e-12:
            scale_map[key] = 1.0
            scale_map[key + ("new",)] = (float(t) - m) / (float(b) - m)
        elif b > 1e-12:
            scale_map[key] = float(t) / float(b)
        else:
            scale_map[key] = 1.0 if t <= 1e-12 else float("nan")
    return scale_map


_PLAN_YEAR_IDS: dict = {}


def mark_plan_year(df: pd.DataFrame) -> pd.DataFrame:
    """Column plan_year: the row's contract (product_type, schedule_id) started after the base bank's
    report date (product_map.plan_year_start); all False on a bank that is not rolled forward."""
    df = df.copy()
    df["plan_year"] = False
    start = plan_year_start(ase.REPORT_DATE)
    if start is None or df.empty or "schedule_id" not in df.columns:
        return df
    if start not in _PLAN_YEAR_IDS:
        from sqlalchemy import text
        from labbank_common.bank_config import get_bank, get_engine
        q = text("SELECT 'L' pt, schedule_id FROM sched.loans WHERE start_date > :d UNION ALL "
                 "SELECT 'D', schedule_id FROM sched.deposits WHERE start_date > :d UNION ALL "
                 "SELECT 'F', schedule_id FROM sched.fin_inst WHERE start_date > :d")
        ids = pd.read_sql(q, get_engine(get_bank()), params={"d": start.date()})
        _PLAN_YEAR_IDS[start] = set(ids["pt"].astype(str) + "|" + ids["schedule_id"].astype(str))
    key = df["product_type"].astype(str) + "|" + df["schedule_id"].astype(str)
    df["plan_year"] = key.isin(_PLAN_YEAR_IDS[start]).to_numpy()
    return df


def rescale_beh_schedules(df: pd.DataFrame, scale_map: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Rescale CF-amount columns by each row's (product_code, bs_side)
    factor. Rows for products not present in scale_map are left UNCHANGED
    (implicit ratio 1.0) -- e.g. IRS ('0000'), always pinned regardless of
    BS reallocation. Plan-year rows (column plan_year) of a split product
    take the map's (product_code, bs_side, "new") factor.
    """
    if df.empty:
        return df
    df = df.copy()
    keys = list(zip(df["product_code"].astype(str), df["bs_side"].astype(str)))
    new = df["plan_year"].to_numpy(bool) if "plan_year" in df.columns else np.zeros(len(df), dtype=bool)
    scale = np.array([scale_map.get(k + ("new",), scale_map.get(k, 1.0)) if n else scale_map.get(k, 1.0)
                      for k, n in zip(keys, new)], dtype=float)
    if np.isnan(scale).any():
        bad = sorted({k for k, s in zip(keys, scale) if np.isnan(s)})
        raise ValueError(f"Scale factor is NaN (target>0 but baseline==0) for products: {bad}")
    for col in _SCALE_COLS:
        if col in df.columns:
            df[col] = df[col].to_numpy(dtype=float) * scale
    return df


def run_full_bank_at_weights(
    products: list[tuple[str, str]],
    baseline_weights: np.ndarray,
    target_weights: np.ndarray,
    include_irs: bool = True,
    progress_every: int = 10,
    n_workers: int | None = None,
    backend: str = "process",
    scheduler_address: str | None = None,
    bank: pd.DataFrame | None = None,
    min_w: np.ndarray | None = None,
) -> pd.DataFrame:
    """Exact 165-curve x 7-scenario reprice at an ARBITRARY set of product
    weights -- same output schema as anchor_sot_exact.run_full_bank().

    min_w: the optimiser's pm.min_w -- products that move through their plan-year contracts only
    (compute_product_scale_map). None = every product rescaled proportionally.

    n_workers: None (default) -> min(os.cpu_count()-2, 14) worker processes.
    1 -> sequential loop (debugging / clean traceback, ignores `backend`).
    backend: "process" (default, VALIDATED) or "dask" (EXPERIMENTAL, NOT
    validated -- see module docstring; do not use for anything you need to
    trust the numbers from yet).
    scheduler_address: dask backend only -- None spins up a local cluster;
    a real address connects to a remote/multi-machine cluster instead.
    bank: optional caller-supplied curve bank (DataFrame with columns
    shape/level/is_anchor/shift_idx/anchor_date/beta0/beta1/beta2), e.g. from
    curve_scenario_bank.build_mc_scenario_bank() -- lets this function reprice
    a DIFFERENT curve set (such as Section 5's 500 Monte Carlo draws) than
    the standard 165-curve anchor+shift bank. Defaults to
    ase.build_scenario_bank()'s own bank (today's behavior, unchanged) when
    not supplied. The repricing loop below is agnostic to where the bank
    came from -- it only ever reads those columns.
    Sequential: ~11 min/book. Parallel (process backend, 16 cores): ~2.5 min/book.
    """
    scale_map = compute_product_scale_map(products, baseline_weights, target_weights, min_w)

    print("Loading EVE run-off CF schedules...")
    all_beh_eve = ase.aee.irrbb_sql_setup.load_all_beh_schedules(ase.REPORT_DATE)
    if include_irs:
        try:
            swap_eve = ase.aee.irrbb_sql_setup.load_all_swap_beh_schedules(ase.REPORT_DATE)
        except Exception:
            swap_eve = pd.DataFrame()
        if not swap_eve.empty:
            all_beh_eve = pd.concat([all_beh_eve, swap_eve], ignore_index=True)
    all_beh_eve = rescale_beh_schedules(mark_plan_year(all_beh_eve), scale_map)
    print(f"  {len(all_beh_eve):,} EVE CF rows ({'hedged' if include_irs else 'unhedged'})")

    print("Loading NII 1Y-horizon CF schedules...")
    all_beh_nii = ase.aee.irrbb_sql_setup.load_beh_schedules(ase.REPORT_DATE, ase.HORIZON_END)
    if include_irs:
        try:
            swap_nii = ase.aee.irrbb_sql_setup.load_swap_beh_schedules(ase.REPORT_DATE, ase.HORIZON_END)
        except Exception:
            swap_nii = pd.DataFrame()
        if not swap_nii.empty:
            all_beh_nii = pd.concat([all_beh_nii, swap_nii], ignore_index=True)
    all_beh_nii = rescale_beh_schedules(mark_plan_year(all_beh_nii), scale_map)
    print(f"  {len(all_beh_nii):,} NII CF rows ({'hedged' if include_irs else 'unhedged'})")

    caps_map, floors_map, coeff_a_map, coeff_b_map = ase.aee._load_rate_maps()
    t1 = ase.aee.irrbb_sql_setup.load_tier1_capital(ase.REPORT_DATE)
    print(f"  Tier-1 capital: {t1/1e6:,.1f}M {home_ccy()}")

    if bank is None:
        _anchors, bank = ase.build_scenario_bank()
    n_curves = len(bank)
    print(f"  Repricing {n_curves} curves (15 anchors + {n_curves - 15} stochastic shifts)...")

    def _rows_from_reprice(curve_meta, eve_out, nii_out) -> list[dict]:
        out = []
        for scen in ase.EVE_SCENARIO_IDS:
            if scen == "base":
                continue
            eve_pct = eve_out[scen] / t1 * 100.0 if t1 else float("nan")
            nii_pct = nii_out[scen] / t1 * 100.0 if t1 else float("nan")
            out.append({
                "shape": curve_meta["shape"], "level": curve_meta["level"],
                "is_anchor": curve_meta["is_anchor"], "shift_idx": curve_meta["shift_idx"],
                "anchor_date": curve_meta["anchor_date"],
                "scenario": scen,
                "delta_eve_pct_t1": eve_pct,
                "delta_nii_pct_t1": nii_pct,
                "eve_sot_pass": eve_pct >= ase.SOT_EVE_FLOOR_PCT_T1,
                "nii_sot_pass": nii_pct >= ase.SOT_NII_FLOOR_PCT_T1,
                # this curve's own base levels (same on every scenario row of the curve)
                "base_eve_amt": eve_out["base"],
                "base_nii_amt": nii_out.get("base_vs_today", nii_out["base"]),
            })
        return out

    t_start = time.time()
    rows: list[dict] = []

    if n_workers == 1:
        # ── Sequential fallback (debugging / clean traceback) ────────────────
        for i, curve in bank.iterrows():
            if curve["is_anchor"] or curve["shift_idx"] % progress_every == 0:
                tag = "anchor" if curve["is_anchor"] else f"shift {curve['shift_idx']}"
                print(f"  [{i+1}/{n_curves}] {curve['shape']:>8s}/{curve['level']:<6s} {tag}...  "
                      f"({time.time()-t_start:.0f}s elapsed)")
            eve_out, nii_out = ase.reprice_anchor_both(
                all_beh_eve, all_beh_nii, caps_map, floors_map, coeff_a_map, coeff_b_map,
                curve["beta0"], curve["beta1"], curve["beta2"],
            )
            rows.extend(_rows_from_reprice(curve, eve_out, nii_out))

    elif backend == "dask":
        from dask.distributed import Client, as_completed as dask_as_completed

        n_proc = n_workers if n_workers is not None else max(1, min((os.cpu_count() or 4) - 2, 14))
        if scheduler_address:
            print(f"  Connecting to Dask cluster at {scheduler_address}...")
            client = Client(scheduler_address)
        else:
            print(f"  Starting local Dask cluster ({n_proc} worker processes)...")
            client = Client(n_workers=n_proc, threads_per_worker=1, processes=True,
                             silence_logs="ERROR")
        try:
            print(f"  Dask dashboard: {client.dashboard_link}")
            # ONE bundled scatter call, not one per object -- see
            # _reprice_one_curve_dask's docstring for why (Dask key-collision
            # bug when scattering several small similarly-shaped dicts
            # separately).
            shared_bundle = {
                "all_beh_eve": all_beh_eve, "all_beh_nii": all_beh_nii,
                "caps_map": caps_map, "floors_map": floors_map,
                "coeff_a_map": coeff_a_map, "coeff_b_map": coeff_b_map,
            }
            bundle_f = client.scatter(shared_bundle, broadcast=True)

            futures = [
                client.submit(_reprice_one_curve_dask, dict(curve), bundle_f)
                for _, curve in bank.iterrows()
            ]
            n_done = 0
            for fut in dask_as_completed(futures):
                curve_meta, eve_out, nii_out = fut.result()
                rows.extend(_rows_from_reprice(curve_meta, eve_out, nii_out))
                n_done += 1
                if n_done % progress_every == 0 or n_done == n_curves:
                    print(f"  [{n_done}/{n_curves}] curves done  ({time.time()-t_start:.0f}s elapsed)")
        finally:
            client.close()

    else:
        # ── Parallel via ProcessPoolExecutor (default, VALIDATED backend) ────
        n_proc = n_workers if n_workers is not None else max(1, min((os.cpu_count() or 4) - 2, 14))
        print(f"  Parallel repricing across {n_proc} worker processes (backend=process)...")
        curve_dicts = [dict(curve) for _, curve in bank.iterrows()]
        n_done = 0
        with ProcessPoolExecutor(
            max_workers=n_proc, initializer=_init_worker,
            initargs=(all_beh_eve, all_beh_nii, caps_map, floors_map, coeff_a_map, coeff_b_map),
        ) as pool:
            futures = [pool.submit(_reprice_one_curve_worker, cd) for cd in curve_dicts]
            for fut in as_completed(futures):
                curve_meta, eve_out, nii_out = fut.result()
                rows.extend(_rows_from_reprice(curve_meta, eve_out, nii_out))
                n_done += 1
                if n_done % progress_every == 0 or n_done == n_curves:
                    print(f"  [{n_done}/{n_curves}] curves done  ({time.time()-t_start:.0f}s elapsed)")

    print(f"  Done in {time.time() - t_start:.0f}s")
    return pd.DataFrame(rows)


def run_and_cache(
    label: str,
    products: list[tuple[str, str]],
    baseline_weights: np.ndarray,
    target_weights: np.ndarray,
    out_dir: str | None = None,
    include_irs: bool = True,
    n_workers: int | None = None,
    backend: str = "process",
    scheduler_address: str | None = None,
    bank: pd.DataFrame | None = None,
    filename_prefix: str = "irrbb_165",
    min_w: np.ndarray | None = None,
) -> pd.DataFrame:
    """run_full_bank_at_weights() + save to a cached xlsx, mirroring
    anchor_sot_exact.save_results()'s pattern. `label` becomes the filename
    stem, e.g. 'baseline' -> irrbb_165_baseline.xlsx.

    bank: passed straight through to run_full_bank_at_weights() -- supply a
    caller-built curve bank (e.g. build_mc_scenario_bank()) to reprice a
    different curve set than the standard 165-curve one.
    filename_prefix: change from the default "irrbb_165" when caching a
    non-165-curve run (e.g. "irrbb_mc500"), so the filename doesn't claim
    "165" for a bank of a different size.
    """
    if out_dir is None:
        out_dir = ase.OUTPUT_DIR
    results = run_full_bank_at_weights(products, baseline_weights, target_weights,
                                        include_irs=include_irs, n_workers=n_workers,
                                        backend=backend, scheduler_address=scheduler_address,
                                        bank=bank, min_w=min_w)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filename_prefix}_{label}.xlsx")
    results.to_excel(out_path, sheet_name="all_scenarios", index=False)
    print(f"Saved -> {out_path}")
    return results


def load_cached(label: str, out_dir: str | None = None, filename_prefix: str = "irrbb_165"):
    if out_dir is None:
        out_dir = ase.OUTPUT_DIR
    path = os.path.join(out_dir, f"{filename_prefix}_{label}.xlsx")
    if os.path.exists(path):
        return pd.read_excel(path, sheet_name="all_scenarios")
    return None


MC500_SCEN_ORDER = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]   


def load_mc500_baseline(label: str = "baseline", out_dir: str | None = None):
    """Load irrbb_mc500_{label}.xlsx and pivot into (n_curve, 6) EVE/NII
    matrices, row i = shift_idx i, columns in MC500_SCEN_ORDER -- NEVER assume raw
    row order matches curve order (the cache rows are not written in curve order).
    Returns (E0_eve, E0_nii), both (n_curve, 6), % T1.
    """
    df = load_cached(label, out_dir=out_dir, filename_prefix="irrbb_mc500")
    if df is None:
        raise FileNotFoundError(f"irrbb_mc500_{label}.xlsx not found -- build it first (per_scenario_hedge.ensure_book_cache)")
    n_curve = int(df["shift_idx"].nunique())
    piv_eve = df.pivot_table(index="shift_idx", columns="scenario", values="delta_eve_pct_t1")
    piv_nii = df.pivot_table(index="shift_idx", columns="scenario", values="delta_nii_pct_t1")
    piv_eve = piv_eve.reindex(index=range(n_curve), columns=MC500_SCEN_ORDER)
    piv_nii = piv_nii.reindex(index=range(n_curve), columns=MC500_SCEN_ORDER)
    if piv_eve.isna().any().any() or piv_nii.isna().any().any():
        raise ValueError(f"irrbb_mc500_{label}.xlsx has a gap in shift_idx/scenario coverage -- cache is incomplete")
    return piv_eve.to_numpy(), piv_nii.to_numpy()
