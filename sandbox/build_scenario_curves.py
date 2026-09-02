"""build_scenario_curves.py
=========================
Pre-compute the sandbox's 15 stylised hypothetical yield curves and the exact
per-product IRRBB metrics for each.

Produces sandbox/scenario_curves.npz with:
  - 15 forward-rate curves (5 shapes × 3 levels) + discount-factor arrays
  - Per-product IRRBB unit metrics for each curve:
      hyp_nii_unit_rate  (15, n)      NII / balance at the hypothetical base
      hyp_delta_nii_unit (15, n, S)   ΔNII / balance per EBA shock
      hyp_eve_pv_factor  (15, n)      EVE / balance at the hypothetical base
      hyp_delta_eve_unit (15, n, S)   ΔEVE / balance per EBA shock
      hyp_shock_ids      (S,)         EBA shock scenario IDs
      cohort_id          (n,)         staleness guard for the sandbox

Metrics — two paths, exact preferred:
  1. build_hyp_irrbb_metrics_exact  (default; needs SQL).  For each curve, feeds
     the existing run-off / 12M CF streams to the *production* re-pricing
     functions (eve_calc_objects / nii_calc_objects) with build_all_shocked_curves
     producing that curve's base + 7 EBA shocked discount curves.  No CF
     regeneration.  Captures NMD behavioural EVE, the real EBA 0%-floor logic,
     and convexity.  Product-level results are broadcast to cohorts by
     prod_metric / prod_balance (same mapping extract_params uses at base).
  2. build_hyp_irrbb_metrics       (fallback; pure NumPy).  Reduced CF model,
     matches sandbox/hyp_engine.compute_hyp_metrics.  Used only if SQL is
     unavailable at precompute time.

Run from the bank_project root (or via the Dagster asset hyp_scenario_curves):
    python sandbox/build_scenario_curves.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

# ── tenor grid ────────────────────────────────────────────────────────────────
N_MONTHS = 360
T_YR     = np.arange(1, N_MONTHS + 1) / 12.0   # 1/12 … 30 years

# ── shape catalogue ───────────────────────────────────────────────────────────
CURVE_TYPES = ["normal", "steep", "humped", "flat", "inverted"]
LEVELS      = ["current", "low", "high"]
# Levels are a SHIFT (bps) off today's real short rate (r0), not an absolute
# target -- so "Low" always means "today, but lower", regardless of where
# today's curve happens to sit. "Current" = no shift, so Flat+Current is
# exactly today's short rate held flat for the whole tenor (the sanity-check
# case: shape=flat, level=current must be a perfectly horizontal line).
LEVEL_SHIFT_BPS = {"current": 0.0, "low": -300.0, "high": 150.0}

CURVE_LABELS = {
    "normal":   "Normal (rising)",
    "steep":    "Steep",
    "humped":   "Humped",
    "flat":     "Flat",
    "inverted": "Inverted",
}
LEVEL_LABELS = {
    "current": "Current level",
    "low":     "Low  (today − 300bp)",
    "high":    "High (today + 150bp)",
}


# ── shape functions ───────────────────────────────────────────────────────────

def _normal(anchor: float, t: np.ndarray) -> np.ndarray:
    return anchor + 0.015 * (1.0 - np.exp(-t / 8.0))

def _steep(anchor: float, t: np.ndarray) -> np.ndarray:
    return anchor + 0.040 * (1.0 - np.exp(-t / 5.0))

def _humped(anchor: float, t: np.ndarray) -> np.ndarray:
    tau = 4.0
    return anchor + 0.030 * (t / tau) * np.exp(1.0 - t / tau)

def _flat(anchor: float, t: np.ndarray) -> np.ndarray:
    return np.full_like(t, anchor)

def _inverted(anchor: float, t: np.ndarray) -> np.ndarray:
    short_end = anchor + 0.020
    long_end  = max(anchor - 0.010, 0.0)
    return long_end + (short_end - long_end) * np.exp(-t / 5.0)

_SHAPE_FN = {
    "normal": _normal, "steep": _steep, "humped": _humped,
    "flat": _flat, "inverted": _inverted,
}

def _fwd_to_disc(fwd: np.ndarray) -> np.ndarray:
    return np.exp(-np.cumsum(fwd / 12.0))


# ── real→hypothetical blend ───────────────────────────────────────────────────
# _normal/_steep/_humped/_flat/_inverted above define the LONG-RUN TARGET shape
# for a given level anchor -- not the curve itself. Every one of the 15
# stylised curves is anchored to TODAY's real short rate (continuous with the
# current curve, no overnight jump) and ramps LINEARLY into its target
# shape/level over RAMP_YEARS, then follows the target exactly.
# (Previous version used an exponential ramp with a 5y time constant, which
# only reaches ~95% of the target by year 15 -- read as "never actually gets
# there" for Low/High. A short linear ramp gets there in RAMP_YEARS and holds,
# per user feedback 2026-08-17.)
RAMP_YEARS = 0.5   # 6 months: fully in the hypothetical level/shape by here


def _blend_to_target(r0: float, target: np.ndarray, t: np.ndarray) -> np.ndarray:
    """r0 at t=0, ramping linearly to `target` by t=RAMP_YEARS, then = target."""
    frac = np.clip(t / RAMP_YEARS, 0.0, 1.0)
    return r0 + (target - r0) * frac


# ── curve generation ──────────────────────────────────────────────────────────

def build_scenario_curves(real_fwd_pln: np.ndarray) -> dict:
    """real_fwd_pln: (360,) today's actual PLN base forward curve (from
    curve_tensors.npz) -- every hypothetical curve is anchored to its short
    end via _blend_to_target() instead of starting at the stylised level."""
    r0 = float(real_fwd_pln[0])   # today's ~1-month rate, the common anchor
    fwd_list, disc_list, scen_ids = [], [], []
    for ctype in CURVE_TYPES:
        fn = _SHAPE_FN[ctype]
        for lvl in LEVELS:
            anchor = r0 + LEVEL_SHIFT_BPS[lvl] / 10_000.0
            target = fn(anchor, T_YR)
            fwd    = np.clip(_blend_to_target(r0, target, T_YR), 0.0, None)
            disc   = _fwd_to_disc(fwd)
            fwd_list.append(fwd)
            disc_list.append(disc)
            scen_ids.append(f"{ctype}_{lvl}")
    return {
        "fwd_rates":    np.array(fwd_list,  dtype=float),   # (15, 360)
        "disc_factors": np.array(disc_list, dtype=float),   # (15, 360)
        "scenario_ids": np.array(scen_ids,  dtype=object),  # (15,)
        "curve_types":  np.array(CURVE_TYPES,  dtype=object),
        "levels":       np.array(LEVELS,       dtype=object),
        "n_months":     np.array([N_MONTHS]),
    }


# ── IRRBB pre-computation ─────────────────────────────────────────────────────

def build_hyp_irrbb_metrics(params, cr, curves, sc_data: dict) -> dict:
    """Pre-compute per-product IRRBB unit metrics for all 15 hypothetical curves.

    NII base rate
    -------------
    Level-shifted from the market calibration using the floor-clipped client
    rate formula:  clip(coeff_a * fwd_hyp + coeff_b, floor, cap).

    delta_NII
    ---------
    Floor-adjusted at the hypothetical base level.  For a term deposit with a
    0% floor in a low-rate environment a downward shock correctly contributes
    little or zero delta, unlike the calibrated linear approximation.

    EVE PV factor + delta_EVE
    -------------------------
    Full CF re-discounting using the hypothetical CurveTensors (same cashflow
    fractions, new discount factors).  Preserves the existing bias correction
    anchor for the base EVE level.

    Returns arrays shaped (n_hyp, n_products, …) for direct npz storage.
    """
    from nii_eve_cf_fast import (
        _eve_cf_cohort_matrix,
        _apply_fixed_rate_calibration,
        _apply_calibrated_delta_fallback,
        _apply_base_nii_bias,
        _fallback_mask,
    )
    from hyp_engine import build_hyp_curve_tensors, build_hyp_rate_matrix

    n   = len(params.product_code)
    n_m = curves.n_months

    p_scen_lst  = [str(s) for s in params.scenario_ids]
    cr_scen_lst = [str(s) for s in cr.rate_scenario_ids]
    cr_base_idx = cr_scen_lst.index("base")

    # EBA shock IDs: curves.scenario_ids also carries own_100_dn/up and
    # own_1_dn/up (curve-shape-only sensitivity/PV01 scenarios), but those
    # were never run through the NII/EVE calibration pipeline -- they don't
    # exist in cr.rate_scenario_ids at all. Restrict to scenarios that are
    # actually calibrated, or every lookup below silently falls through to
    # its `except: pass` and reports a misleading 0.00% "no impact" instead
    # of "not computed" (2026-08-15 fix).
    shock_ids   = [s for s in cr_scen_lst if s != "base"]
    n_shock     = len(shock_ids)

    # Baseline balance — needed for NII and EVE per-unit computation
    base_bal = params.balance_arr.copy()
    safe_bal = np.where(base_bal > 0, base_bal, 1.0)

    # Blank EVE rows: no CF schedule, no pre-stored eve_pv_frac → calibrated fallback
    blank_eve = (cr.cf_n_q == 0) & ~np.any(np.abs(cr.eve_pv_frac) > 1e-18, axis=(1, 2))

    # NII base for non-cohort / synthetic-constant rows (e.g. saving_account,
    # t_bill): the CF einsum below is structurally zero for them regardless of
    # curve (cohort_interest_yf_m is all-zero, see _fallback_mask), and the
    # capture of all_nii_unit[i] happens BEFORE _apply_base_nii_bias fixes the
    # base column -- so without this, these rows would report a hyp NII of
    # exactly 0 for every one of the 15 hypothetical curves. Reprice them
    # directly off their own client-rate formula (same one hyp_engine's
    # build_hyp_rate_matrix uses) as a delta on top of the real calibrated
    # rate, so "hyp == today" reproduces params.nii_unit_rate exactly.
    # (2026-08-17)
    nii_fb_mask  = _fallback_mask(params)
    nii_fb_tenor = np.clip(params.repricing_tenor_m.astype(int) - 1, 0, n_m - 1)
    nii_fb_a     = np.nan_to_num(params.coeff_a,      nan=0.0)
    nii_fb_b     = np.nan_to_num(params.coeff_b,      nan=0.0)
    nii_fb_floor = np.nan_to_num(params.client_floor, nan=-1.0)
    nii_fb_cap   = np.nan_to_num(params.client_cap,   nan=1.0)
    real_fwd_pln = curves.get_fwd_curve("base", "PLN")

    def _client_rate(fwd: np.ndarray) -> np.ndarray:
        return np.clip(
            nii_fb_a[nii_fb_mask] * fwd[nii_fb_tenor[nii_fb_mask]] + nii_fb_b[nii_fb_mask],
            nii_fb_floor[nii_fb_mask], nii_fb_cap[nii_fb_mask],
        )

    nii_fb_rate_real = _client_rate(real_fwd_pln)

    n_hyp = len(sc_data["scenario_ids"])
    all_nii_unit       = np.zeros((n_hyp, n),          dtype=float)
    all_delta_nii_unit = np.zeros((n_hyp, n, n_shock),  dtype=float)
    all_eve_pv_factor  = np.zeros((n_hyp, n),          dtype=float)
    all_delta_eve_unit = np.zeros((n_hyp, n, n_shock),  dtype=float)

    for i, sid in enumerate(sc_data["scenario_ids"]):
        print(f"  [{i+1:2d}/{n_hyp}] {sid} ...", end="", flush=True)
        fwd_hyp    = sc_data["fwd_rates"][i]            # (360,) decimal
        hyp_curves = build_hyp_curve_tensors(curves, fwd_hyp)

        # ── CF-based NII at hyp curves ───────────────────────────────────────
        hyp_rm, hyp_rnm = build_hyp_rate_matrix(params, cr, hyp_curves)

        amount_x_out = (base_bal * params.sign)[:, None] * params.cohort_interest_yf_m
        nii_c        = np.einsum("im,ims->is", amount_x_out, hyp_rm)

        cap_x_remain = (base_bal * params.sign)[:, None] * params.cohort_capital_remain_m
        nii_r        = np.einsum("im,ims->is", cap_x_remain, hyp_rnm)

        nii_total = nii_c + nii_r
        _apply_fixed_rate_calibration(nii_total, base_bal, params, cr)
        _apply_calibrated_delta_fallback(nii_total, base_bal, params, cr)
        # Capture base NII before bias anchoring so it reflects the hypothetical
        # curve level, not the calibrated params.nii_unit_rate.  The bias is only
        # needed to keep shocked deltas consistent with the exact calibration.
        all_nii_unit[i] = nii_total[:, cr_base_idx] / safe_bal
        if nii_fb_mask.any():
            nii_fb_rate_hyp = _client_rate(fwd_hyp)
            all_nii_unit[i, nii_fb_mask] = (
                params.nii_unit_rate[nii_fb_mask] + (nii_fb_rate_hyp - nii_fb_rate_real)
            )
        _apply_base_nii_bias(nii_total, base_bal, params, cr)
        for k, scen in enumerate(shock_ids):
            try:
                s_cr = cr_scen_lst.index(scen)
                all_delta_nii_unit[i, :, k] = (
                    nii_total[:, s_cr] - nii_total[:, cr_base_idx]
                ) / safe_bal
            except ValueError:
                pass

        # ── EVE CF re-discounting at hyp curves ──────────────────────────────
        eve_mat = _eve_cf_cohort_matrix(base_bal, params, cr, hyp_curves,
                                         use_precomputed_frac=False)  # (n, n_cr_scen)

        # EVE PV factor per unit balance (anchored to CF computation at hyp base)
        all_eve_pv_factor[i] = eve_mat[:, cr_base_idx] / safe_bal
        # Blank rows (NMD, IRS: no CF schedule) get 0 from the CF matrix above —
        # fall back to the calibrated market EVE PV factor so the base level is
        # not silently missing the whole non-maturity book.  (The level effect of
        # the hypothetical curve on NMD EVE is second order and left uncaptured.)
        if blank_eve.any():
            all_eve_pv_factor[i, blank_eve] = params.eve_pv_factor[blank_eve]

        # delta EVE per shock
        for k, scen in enumerate(shock_ids):
            try:
                s_cr = cr_scen_lst.index(scen)
                all_delta_eve_unit[i, :, k] = (
                    (eve_mat[:, s_cr] - eve_mat[:, cr_base_idx]) / safe_bal
                )
            except ValueError:
                try:
                    calib = params.delta_eve_unit[:, p_scen_lst.index(scen)]
                    all_delta_eve_unit[i, :, k] = calib
                except ValueError:
                    pass

        # Fallback for blank EVE rows (no CF, no eve_pv_frac)
        if blank_eve.any():
            for k, scen in enumerate(shock_ids):
                try:
                    all_delta_eve_unit[i, blank_eve, k] = (
                        params.delta_eve_unit[blank_eve, p_scen_lst.index(scen)]
                    )
                except ValueError:
                    pass

        print(" done")

    return {
        "hyp_nii_unit_rate":  all_nii_unit,          # (15, n)
        "hyp_delta_nii_unit": all_delta_nii_unit,    # (15, n, S)
        "hyp_eve_pv_factor":  all_eve_pv_factor,     # (15, n)
        "hyp_delta_eve_unit": all_delta_eve_unit,    # (15, n, S)
        "hyp_shock_ids":      np.array(shock_ids, dtype=object),  # (S,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXACT per-product IRRBB for every hypothetical curve
# ─────────────────────────────────────────────────────────────────────────────

def build_hyp_irrbb_metrics_exact(params, sc_data, report_date):
    """Per-product NII + EVE (base + 7 EBA shocks) for each of the 15 hypothetical
    curves, computed with the **production** re-pricing functions.

    The exact pipeline separates "generate cash flows" (once) from "re-price under
    a curve" (N times).  Here we feed the existing run-off / 12M CF streams to
    ``eve_calc_objects.compute_eve_shocked_schedule`` and
    ``nii_calc_objects.compute_nii_shocked_schedule`` with, for each hypothetical
    curve, ``build_all_shocked_curves`` producing that curve's base + 7 shocked
    discount curves.  No cash-flow regeneration.

    Product-level results are broadcast to every cohort of the product by
    ``prod_metric / prod_balance`` — identical to how ``extract_params`` maps
    ``irrbb.eve_results`` onto cohorts for the base curve.

    Requires SQL (offline precompute step only).  Returns the same dict shape as
    ``build_hyp_irrbb_metrics``.
    """
    _root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _IRRBB = os.path.join(_root, "irrbb_calc", "python_code")
    if _IRRBB not in sys.path:
        sys.path.insert(0, _IRRBB)
    import pandas as pd
    import config as _cfg
    _cfg.report_date = pd.to_datetime(report_date)
    import sql_setup as _sql
    import eba_shock_curves as _esc
    import eve_calc_objects as _eve
    import nii_calc_objects as _nii

    HORIZON_YF  = 1.0
    horizon_end = _cfg.report_date + pd.Timedelta(days=round(HORIZON_YF * 365.25))
    DISC_MAP    = {"PLN": "PLN_disc_curve", "EUR": "EUR_disc_curve", "USD": "USD_disc_curve"}
    OWN_BPS     = -100.0

    _ir = pd.read_excel(os.path.join(_root, "balance_generate", "input_data", "interest_rt.xlsx"))
    def _map(col, transform=lambda x: x, skip_if=lambda x: False):
        out = {}
        for _, r in _ir.iterrows():
            v = r.get(col)
            if col in _ir.columns and not pd.isna(v) and not skip_if(float(v)):
                out[str(int(r["product_code"]))] = transform(float(v))
        return out
    CAPS   = _map("client_cap")
    FLOORS = _map("client_floor")
    A_MAP  = _map("a", skip_if=lambda v: v == 1.0)
    B_MAP  = _map("b", transform=lambda v: v / 100.0, skip_if=lambda v: v == 0.0)

    # CF streams (loaded once)
    eve_beh = _sql.load_all_beh_schedules(_cfg.report_date)
    try:
        _sw = _sql.load_all_swap_beh_schedules(_cfg.report_date)
        if not _sw.empty:
            eve_beh = pd.concat([eve_beh, _sw], ignore_index=True)
    except Exception:
        pass
    nii_beh = _sql.load_beh_schedules(_cfg.report_date, horizon_end)
    try:
        _sw = _sql.load_swap_beh_schedules(_cfg.report_date, horizon_end)
        if not _sw.empty:
            nii_beh = pd.concat([nii_beh, _sw], ignore_index=True)
    except Exception:
        pass
    print(f"  CF rows: EVE {len(eve_beh):,}  NII {len(nii_beh):,}")

    mkt_df   = _sql.load_mkt_curves(_cfg.report_date)          # curve_name, n_days, d_f
    pln_mask = mkt_df["curve_name"] == DISC_MAP["PLN"]
    pln_days = mkt_df.loc[pln_mask, "n_days"].to_numpy(dtype=float)

    n   = len(params.product_code)
    key = np.array([f"{params.product_code[i]}|{params.bs_side[i]}" for i in range(n)])
    prod_bal = {k: float(params.balance_arr[key == k].sum()) for k in np.unique(key)}

    shock_ids = [s for s in _esc.ALL_SCENARIO_IDS if s != "base"]       # 7
    n_hyp     = len(sc_data["scenario_ids"])
    out_nii   = np.zeros((n_hyp, n));                 out_dnii = np.zeros((n_hyp, n, len(shock_ids)))
    out_eve   = np.zeros((n_hyp, n));                 out_deve = np.zeros((n_hyp, n, len(shock_ids)))

    N_M = int(np.asarray(sc_data["n_months"]).flat[0])
    yr  = np.arange(1, N_M + 1) / 12.0

    def _daily_disc(sub):
        parts = []
        for ccy, cn in DISC_MAP.items():
            g = sub[sub["curve_name"] == cn]
            if g.empty:
                continue
            daily = _esc._log_interp_daily_df(g["n_days"].to_numpy(), g["d_f"].to_numpy(),
                                              int(g["n_days"].max()))
            daily["node_date"] = _cfg.report_date + pd.to_timedelta(daily["n_days"], unit="D")
            daily.insert(0, "curve_name", cn)
            parts.append(daily.set_index(["curve_name", "node_date"])["d_f"].sort_index())
        return pd.concat(parts).sort_index()

    for i, sid in enumerate(sc_data["scenario_ids"]):
        fwd_hyp  = np.asarray(sc_data["fwd_rates"][i], dtype=float)[:N_M]     # annualised, monthly
        df_hyp_m = np.exp(-np.cumsum(fwd_hyp / 12.0))
        z_hyp_m  = -np.log(np.clip(df_hyp_m, 1e-12, None)) / yr
        z_nodes  = np.interp(pln_days / 365.25, yr, z_hyp_m, left=z_hyp_m[0], right=z_hyp_m[-1])
        df_nodes = np.where(pln_days > 0, np.exp(-z_nodes * (pln_days / 365.25)), 1.0)

        hyp_mkt = mkt_df.copy()
        hyp_mkt.loc[pln_mask, "d_f"] = df_nodes

        shocked = _esc.build_all_shocked_curves(hyp_mkt, _cfg.report_date, "PLN", OWN_BPS, 0.0)

        p_nii, p_eve = {}, {}
        for scen in _esc.ALL_SCENARIO_IDS:
            disc = _daily_disc(shocked[shocked["scenario_id"] == scen])
            es = _eve.compute_eve_shocked_schedule(
                eve_beh, disc, DISC_MAP, scen, 0.0, CAPS, FLOORS, A_MAP, B_MAP)
            ns = _nii.compute_nii_shocked_schedule(
                nii_beh, disc, DISC_MAP, _cfg.report_date, scen, HORIZON_YF, 0.0,
                CAPS, FLOORS, A_MAP, B_MAP)
            es["k"] = es["product_code"].astype(str) + "|" + es["bs_side"].astype(str)
            ns["k"] = ns["product_code"].astype(str) + "|" + ns["bs_side"].astype(str)
            p_eve[scen] = es.groupby("k")["pv_total"].sum().to_dict()
            p_nii[scen] = ns.groupby("k")["nii_total"].sum().to_dict()

        for j in range(n):
            k  = key[j]
            pb = prod_bal.get(k, 0.0)
            if pb <= 0:
                continue
            eb = p_eve["base"].get(k, 0.0)
            nb = p_nii["base"].get(k, 0.0)
            out_eve[i, j] = eb / pb
            out_nii[i, j] = nb / pb
            for kk, s in enumerate(shock_ids):
                out_deve[i, j, kk] = (p_eve[s].get(k, 0.0) - eb) / pb
                out_dnii[i, j, kk] = (p_nii[s].get(k, 0.0) - nb) / pb

        print(f"  [{i+1:2d}/{n_hyp}] {sid:16s}  EVE base {sum(p_eve['base'].values())/1e6:+8.0f} M   "
              f"NII base {sum(p_nii['base'].values())/1e6:+7.0f} M")

    return {
        "hyp_nii_unit_rate":  out_nii,
        "hyp_delta_nii_unit": out_dnii,
        "hyp_eve_pv_factor":  out_eve,
        "hyp_delta_eve_unit": out_deve,
        "hyp_shock_ids":      np.array(shock_ids, dtype=object),
    }


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _HERE    = os.path.dirname(os.path.abspath(__file__))
    _ROOT    = os.path.abspath(os.path.join(_HERE, ".."))
    _OPTCODE = os.path.join(_ROOT, "optimize_prep", "python_code")
    for _p in [_HERE, _OPTCODE]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from bs_vector import BalanceSheetParams, CohortRates, CurveTensors  # noqa: E402

    PARAMS_NPZ = os.path.join(_ROOT, "optimize_prep", "output", "product_params.npz")
    CURVES_NPZ = os.path.join(_ROOT, "optimize_prep", "output", "curve_tensors.npz")
    OUT_PATH   = os.path.join(_HERE, "scenario_curves.npz")

    print("Loading pipeline params & curves...")
    params = BalanceSheetParams.load(PARAMS_NPZ)
    cr     = CohortRates.load(PARAMS_NPZ)
    curves = CurveTensors.load(CURVES_NPZ)
    print(f"  Products: {len(params.product_code)}"
          f"  |  EBA scenarios: {[s for s in curves.scenario_ids if s != 'base']}")

    print("\nBuilding scenario curves (anchored to today's real PLN short rate)...")
    real_fwd_pln = curves.get_fwd_curve("base", "PLN")
    sc_data = build_scenario_curves(real_fwd_pln)
    print(f"  {len(sc_data['scenario_ids'])} scenarios: {list(sc_data['scenario_ids'])}")

    _report_date = str(params.report_date)
    try:
        print("\nPre-computing EXACT per-product IRRBB for all hypothetical curves "
              "(production re-pricing functions, needs SQL)...")
        irrbb = build_hyp_irrbb_metrics_exact(params, sc_data, _report_date)
        print("  -> exact path OK")
    except Exception as _exc:
        print(f"  !! exact path failed ({type(_exc).__name__}: {_exc})")
        print("  -> falling back to the reduced numpy model (compute_hyp_metrics parity)")
        irrbb = build_hyp_irrbb_metrics(params, cr, curves, sc_data)
    print(f"\n  hyp_nii_unit_rate:  {irrbb['hyp_nii_unit_rate'].shape}")
    print(f"  hyp_delta_nii_unit: {irrbb['hyp_delta_nii_unit'].shape}")
    print(f"  hyp_eve_pv_factor:  {irrbb['hyp_eve_pv_factor'].shape}")
    print(f"  hyp_delta_eve_unit: {irrbb['hyp_delta_eve_unit'].shape}")
    print(f"  hyp_shock_ids:      {list(irrbb['hyp_shock_ids'])}")

    # cohort_id lets sandbox/app.py detect a stale cache (product_params.npz
    # regenerated with a different cohort set) and fall back to the linear
    # approximation instead of a shape-mismatch crash — see ftp_store.py's
    # load_ftp_rates() for the same convention.
    out = {**sc_data, **irrbb, "cohort_id": params.cohort_id}
    np.savez(OUT_PATH, **out)
    mb = sum(v.nbytes for v in out.values() if hasattr(v, "nbytes")) / 1e6
    print(f"\nSaved -> {OUT_PATH}  (~{mb:.1f} MB)")
