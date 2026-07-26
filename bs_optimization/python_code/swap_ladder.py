"""swap_ladder.py
================
Synthetic seasoned-swap "ladder" used by the optimal-hedge comparison in
hedge_comparison.py.

For each tenor T in {3,4,5,6,7}Y and each elapsed month e in [0, T*12), this
builds one hypothetical pay-fixed/receive-float swap bucket that could
already be sitting on the balance sheet today (transacted e months ago,
T*12-e months of remaining life, quarterly WIBOR3M-style repricing) --
300 buckets in total.

Each bucket's EVE/NII sensitivity to the 6 official EBA shock scenarios is
priced directly against CurveTensors (optimize_prep/output/curve_tensors.npz)
-- the SAME discount-factor tensor nii_eve_cf_fast.py uses for every other
product in this optimizer -- via the standard single-curve swap-pricing
identities:

    PV(float leg, no spread) = notional * (DF(t_first) - DF(maturity))
    PV(fixed leg)            = notional * fixed_rate * sum_i accrual_i * DF(coupon_i)

This deliberately does NOT go through irs_objects.py's QuantLib schedule
generator: that machinery expects discount/forward curves in a
(curve_name, node_date) MultiIndex format sourced from ir_derivatives' own
SQL curve tables, a different curve representation than the NS-beta/
CurveTensors world this optimizer is built on. Pricing directly against
CurveTensors avoids that format mismatch and stays consistent with how
every other product's sensitivity is computed here.

Fixed rate per bucket is approximated as the historical NS-curve zero rate
at the bucket's tenor, fit at the bucket's actual origination date
(today - e months) -- these buckets represent swaps that really could have
been transacted back then, not hypothetical future trades. This is a
simplification (a true par rate needs float-leg PV-weighting), consistent
with the first-order approximations already used elsewhere in this
optimizer (see mc_scenario_basis.py).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE    = os.path.dirname(os.path.abspath(__file__))
_OPTPREP = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "python_code"))
for _p in (_HERE, _OPTPREP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bs_vector import CurveTensors                                             # noqa: E402
from ns_curve_model import (                                                    # noqa: E402
    DIEBOLD_LI_TAU, ns_design_matrix, ns_loadings,
    fit_ns_timeseries, load_historical_panel,
)
from curve_scenario_bank import build_scenario_bank                            # noqa: E402

_IRRBB = os.path.normpath(os.path.join(_HERE, "..", "..", "irrbb_calc", "python_code"))
if _IRRBB not in sys.path:
    sys.path.insert(0, _IRRBB)
from eba_shock_curves import (                                                  # noqa: E402
    get_shock_params, shock_bps_at_tenors, default_realistic_base_floor_bps,
)

CURRENCY        = "PLN"
REPORT_DATE     = pd.Timestamp("2024-12-31")     # matches extract_params.REPORT_DATE
SCENARIO_IDS    = ["base", "par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]
SHOCKED_SCENS   = SCENARIO_IDS[1:]
TENORS_YEARS    = (3, 4, 5, 6, 7)
FIXING_FREQ_M   = 3     # WIBOR3M-style quarterly repricing, both legs

IRS_MARGIN_BPS  = 25.0
# Every ladder bucket's fixed rate was previously priced at the raw curve zero
# rate -- i.e. fair mid-market, with zero transaction cost. That made the
# optimizer's swap overlay look almost frictionlessly profitable and it
# consistently bought up to the notional cap. In reality, a bank transacting
# a swap faces a dealer/market spread that works AGAINST whichever side it's
# on: if receive-fixed/pay-float, the fixed rate actually received clears
# BELOW the theoretical curve rate (commonly ~20bps, in a market where many
# banks want the same hedge direction, per user feedback); if pay-fixed, the
# bank would pay ABOVE the curve rate. Applied by sign of `direction` in
# _build_bucket_specs/build_ladder_bank -- see their docstrings. 25bps is a
# starting assumption, not a fitted value -- retune once real swap-desk
# quotes are available to calibrate against.

_CURVE_TENSORS_PATH = os.path.normpath(
    os.path.join(_OPTPREP, "..", "output", "curve_tensors.npz")
)


@dataclass
class LadderBank:
    bucket_ids:     list            # (n_buckets,) e.g. "LADDER_5Y_E014"
    tenor_years:    np.ndarray       # (n_buckets,) float
    elapsed_m:      np.ndarray       # (n_buckets,) int
    eve_level:      dict             # scenario_id -> (n_buckets,) PV per unit notional
    nii_level:      dict             # scenario_id -> (n_buckets,) 1Y NII per unit notional
    ep_carry_base:  np.ndarray       # (n_buckets,) approx net 1Y carry per unit notional, base scenario
    direction:      float            # +1.0 = pay-fixed/receive-float, -1.0 = flipped
    notional_cap:   float            # per-bucket upper bound, PLN (fraction-of-TA handled by caller)


def _df_at_month(disc_curve: np.ndarray, month: float) -> float:
    """Discount factor at `month` months from today (month<=0 -> 1.0)."""
    if month <= 0:
        return 1.0
    idx = int(round(month)) - 1
    idx = min(max(idx, 0), disc_curve.shape[0] - 1)
    return float(disc_curve[idx])


def _reset_months(tenor_years: int, elapsed_m: int, freq_m: int = FIXING_FREQ_M) -> np.ndarray:
    """Future coupon/reset months (from today, >0) for a bucket originated
    `elapsed_m` months ago with tenor `tenor_years`, repricing every `freq_m`
    months. Last entry always equals the bucket's remaining life (maturity)."""
    total_m = tenor_years * 12
    k_max = total_m // freq_m
    all_months = np.array([-elapsed_m + k * freq_m for k in range(1, k_max + 1)], dtype=float)
    return all_months[all_months > 0]


def _historical_zero_rate(ns_ts: pd.DataFrame, tenor_years: float, months_ago: int,
                           tau: float = DIEBOLD_LI_TAU) -> float:
    """Historical NS zero rate at `tenor_years`, fit at (today - months_ago)."""
    origin_date = REPORT_DATE - pd.DateOffset(months=int(months_ago))
    pos = ns_ts.index.searchsorted(origin_date)
    pos = int(min(max(pos, 0), len(ns_ts) - 1))
    beta = ns_ts.iloc[pos][["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
    X = ns_design_matrix(np.array([tenor_years]), tau)     # (1, 3)
    rate_bps = float((X @ beta)[0])
    return rate_bps / 10000.0                              # bps -> decimal


def _build_bucket_specs(
    tenors: tuple, ns_ts: pd.DataFrame,
    direction: float = 1.0, margin_bps: float = IRS_MARGIN_BPS,
) -> list:
    """List of (bucket_id, tenor_years, elapsed_m, reset_months, fixed_rate).

    fixed_rate = curve zero rate + direction * margin -- the market spread
    always works against the bank's side of the trade (see IRS_MARGIN_BPS):
    pay-fixed (direction=+1) pays a HIGHER rate than curve; receive-fixed
    (direction=-1) receives a LOWER rate than curve. margin_bps=0 recovers
    the old curve-mid pricing.
    """
    margin = direction * margin_bps / 10000.0
    specs = []
    for T in tenors:
        total_m = T * 12
        for e in range(total_m):
            reset_months = _reset_months(T, e)
            if len(reset_months) == 0:
                continue
            fixed_rate = _historical_zero_rate(ns_ts, float(T), e) + margin
            specs.append((f"LADDER_{T}Y_E{e:03d}", float(T), e, reset_months, fixed_rate))
    return specs


NII_RATE_FLOOR = 0.0
# Two-stage floor, mirroring eba_shock_curves.apply_shock_to_disc_curve exactly
# (same function reused via default_realistic_base_floor_bps, so both engines
# stay consistent):
#   1. The BASE (unshocked) curve is floored realistically (graduated, rises
#      with tenor -- see default_realistic_base_floor_bps) -- this is a "real
#      curve" floor, not an EBA one, and applies to scenario="base" too. Before
#      this, an MC drift could reconstruct a base curve with a long end pinned
#      identically flat for 10-20+ years, which has no PLN historical precedent.
#   2. The 6 official EBA shocks are added ON TOP of that already-floored base,
#      then EBA's own regulatory floor (NII_RATE_FLOOR, flat 0%) is applied to
#      the shocked result -- exactly the original (unchanged) EBA treatment.


def _ns_disc_curve_monthly(beta: np.ndarray, scenario: str, shock_params: dict,
                            tau: float = DIEBOLD_LI_TAU, n_months: int = 360,
                            rate_floor: float = NII_RATE_FLOOR) -> np.ndarray:
    """(n_months,) discount factors from an NS curve (beta0,beta1,beta2) plus the
    EBA shock for `scenario`, analytic (no SQL/QuantLib) -- same formula family
    as ns_curve_model.py + irrbb_calc/eba_shock_curves.py, continuously compounded.

    rate_floor: absolute EBA-shock-floor (decimal, default 0%), applied only to
    the 6 shocked scenarios -- see module-level comment above for the base-curve
    floor (default_realistic_base_floor_bps), which applies unconditionally.
    """
    t_years = np.arange(1, n_months + 1) / 12.0
    l1, l2 = ns_loadings(t_years, tau)
    base_r_bps = beta[0] + beta[1] * l1 + beta[2] * l2
    base_r_bps = np.maximum(default_realistic_base_floor_bps(t_years), base_r_bps)
    if scenario == "base":
        r = base_r_bps / 10000.0
    else:
        shock_bps = shock_bps_at_tenors(t_years, scenario, shock_params)
        r = np.maximum(rate_floor, (base_r_bps + shock_bps) / 10000.0)
    return np.exp(-r * t_years)


def price_ladder_against_curve_bank(tenors: tuple = TENORS_YEARS, direction: float | None = None,
                                     bank: pd.DataFrame | None = None) -> dict:
    """Exact (analytic DCF, no SQL/QuantLib) EVE/NII sensitivity for every ladder
    bucket against all 165 curve_scenario_bank.py curves x 6 EBA scenarios.

    bank: optional caller-supplied curve bank (DataFrame with columns
    shape/level/is_anchor/shift_idx/anchor_date/beta0/beta1/beta2), e.g. from
    curve_scenario_bank.build_mc_scenario_bank() -- lets this function price
    the ladder against a DIFFERENT curve set (such as Section 5's 500 Monte
    Carlo draws, tested curve-then-6-official-shocks just like the 165-curve
    case) than the standard 165-curve bank. Defaults to build_scenario_bank()'s
    own bank (today's behavior, unchanged) when not supplied.

    Returns dict with:
      bucket_ids   : list[str]                  (n_buckets,)
      curve_shape/level/is_anchor/shift_idx      (n_curves,)  -- from the bank used
      delta_eve_pln, delta_nii_pln : dict[scenario] -> (n_curves, n_buckets) PLN per unit notional
      ep_carry_pln  : (n_buckets,) approx 1Y net carry per unit notional at TODAY's curve
      direction     : float, sign applied (matches build_ladder_bank's convention)
    """
    if direction is None:
        direction = _detect_hedge_direction()

    panel = load_historical_panel()
    ns_ts = fit_ns_timeseries(panel, DIEBOLD_LI_TAU)
    specs = _build_bucket_specs(tenors, ns_ts, direction=direction)
    n_buckets = len(specs)

    if bank is None:
        _, bank = build_scenario_bank()
    n_curves = len(bank)
    shock_params = get_shock_params(CURRENCY)

    delta_eve = {s: np.zeros((n_curves, n_buckets)) for s in SHOCKED_SCENS}
    delta_nii = {s: np.zeros((n_curves, n_buckets)) for s in SHOCKED_SCENS}

    today_beta = ns_ts.iloc[-1][["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
    today_base_disc = _ns_disc_curve_monthly(today_beta, "base", shock_params)
    ep_carry = np.zeros(n_buckets)
    for bi, (_bid, _T, _e, reset_months, fixed_rate) in enumerate(specs):
        nii_b = direction * _price_bucket_nii(today_base_disc, reset_months)
        year_h = min(12.0, float(reset_months[-1]))
        ep_carry[bi] = nii_b - direction * fixed_rate * (year_h / 12.0)

    for ci, (_, curve) in enumerate(bank.iterrows()):
        beta = curve[["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
        base_disc = _ns_disc_curve_monthly(beta, "base", shock_params)
        base_eve = np.empty(n_buckets)
        base_nii = np.empty(n_buckets)
        for bi, (_bid, _T, _e, reset_months, fixed_rate) in enumerate(specs):
            base_eve[bi] = direction * _price_bucket_eve(base_disc, reset_months, fixed_rate)
            base_nii[bi] = direction * _price_bucket_nii(base_disc, reset_months)

        for s in SHOCKED_SCENS:
            shocked_disc = _ns_disc_curve_monthly(beta, s, shock_params)
            for bi, (_bid, _T, _e, reset_months, fixed_rate) in enumerate(specs):
                eve_s = direction * _price_bucket_eve(shocked_disc, reset_months, fixed_rate)
                nii_s = direction * _price_bucket_nii(shocked_disc, reset_months)
                delta_eve[s][ci, bi] = eve_s - base_eve[bi]
                delta_nii[s][ci, bi] = nii_s - base_nii[bi]

    return dict(
        bucket_ids=[s[0] for s in specs],
        elapsed_m=np.array([s[2] for s in specs], dtype=int),
        tenor_years=np.array([s[1] for s in specs], dtype=float),
        shape=bank["shape"].to_numpy(), level=bank["level"].to_numpy(),
        is_anchor=bank["is_anchor"].to_numpy(), shift_idx=bank["shift_idx"].to_numpy(),
        delta_eve_pln=delta_eve, delta_nii_pln=delta_nii,
        ep_carry_pln=ep_carry, direction=direction,
    )


def price_ladder_vs_mc_curves(
    factor_draws: np.ndarray,          # (N, 3) relative (d_beta0,d_beta1,d_beta2), bps
    today_beta: np.ndarray,            # (3,) absolute (beta0,beta1,beta2) at the SAME
                                        # anchor date factor_draws is relative to
    tenors: tuple = TENORS_YEARS,
    direction: float | None = None,
    tau: float = DIEBOLD_LI_TAU,
    n_months: int = 360,
) -> dict:
    """Exact (analytic DCF) EVE/NII sensitivity for every ladder bucket
    against N arbitrary Monte Carlo curves, DRIFT-ONLY -- each MC curve is
    priced directly against TODAY's curve (scenario="base" passed to
    _ns_disc_curve_monthly on both sides, no further EBA shock layered on
    top), unlike price_ladder_against_curve_bank() above (which prices the
    165-curve historical bank AGAINST each of the 6 official EBA shocks --
    "if today's curve had been curve X, would a standard shock still be
    fine"). Here, each MC curve already IS a complete alternate future
    state (e.g. from ns_curve_model.simulate_mean_reverting_shocks) -- the
    question is simply "what does this bucket's EVE/NII look like if THIS
    curve had held today," matching how stochastic_optimizer.py's
    build_scenario_ep_matrix() already treats the BS side of the same
    factor_draws (factor_draws @ R, no extra shock). Do not conflate the
    two conventions.

    Returns dict with:
      bucket_ids, elapsed_m, tenor_years, fixed_rate    (n_buckets,)
      delta_eve_pln, delta_nii_pln                       (N, n_buckets), PLN per unit notional, vs today
      ep_carry_pln                                       (n_buckets,), at TODAY's own curve
      direction
    """
    if direction is None:
        direction = _detect_hedge_direction()

    panel = load_historical_panel()
    ns_ts = fit_ns_timeseries(panel, tau)
    specs = _build_bucket_specs(tenors, ns_ts, direction=direction)   # curve-independent, built once
    n_buckets = len(specs)

    shock_params = get_shock_params(CURRENCY)
    today_beta = np.asarray(today_beta, dtype=float)
    mc_betas = today_beta[None, :] + np.asarray(factor_draws, dtype=float)   # (N, 3)
    n_scen = mc_betas.shape[0]

    # ── today's own curve, priced once ───────────────────────────────────────
    today_disc = _ns_disc_curve_monthly(today_beta, "base", shock_params, tau, n_months)
    base_eve = np.empty(n_buckets)
    base_nii = np.empty(n_buckets)
    ep_carry = np.empty(n_buckets)
    fixed_rate_arr = np.empty(n_buckets)
    for bi, (_bid, _T, _e, reset_months, fixed_rate) in enumerate(specs):
        base_eve[bi] = direction * _price_bucket_eve(today_disc, reset_months, fixed_rate)
        base_nii[bi] = direction * _price_bucket_nii(today_disc, reset_months)
        year_h = min(12.0, float(reset_months[-1]))
        ep_carry[bi] = base_nii[bi] - direction * fixed_rate * (year_h / 12.0)
        fixed_rate_arr[bi] = fixed_rate

    # ── each MC curve, priced against the SAME curve-independent bucket specs ──
    delta_eve = np.empty((n_scen, n_buckets))
    delta_nii = np.empty((n_scen, n_buckets))
    for n in range(n_scen):
        disc_n = _ns_disc_curve_monthly(mc_betas[n], "base", shock_params, tau, n_months)
        for bi, (_bid, _T, _e, reset_months, fixed_rate) in enumerate(specs):
            eve_n = direction * _price_bucket_eve(disc_n, reset_months, fixed_rate)
            nii_n = direction * _price_bucket_nii(disc_n, reset_months)
            delta_eve[n, bi] = eve_n - base_eve[bi]
            delta_nii[n, bi] = nii_n - base_nii[bi]

    return dict(
        bucket_ids=[s[0] for s in specs],
        elapsed_m=np.array([s[2] for s in specs], dtype=int),
        tenor_years=np.array([s[1] for s in specs], dtype=float),
        fixed_rate=fixed_rate_arr,
        delta_eve_pln=delta_eve, delta_nii_pln=delta_nii,
        ep_carry_pln=ep_carry, direction=direction,
    )


def _price_bucket_eve(disc_curve: np.ndarray, reset_months: np.ndarray, fixed_rate: float,
                       freq_m: int = FIXING_FREQ_M) -> float:
    """Net PV per unit notional (receive-float leg - pay-fixed leg) at one curve."""
    t_first  = float(reset_months[0])
    maturity = float(reset_months[-1])
    accrual  = freq_m / 12.0

    pv_float = _df_at_month(disc_curve, t_first) - _df_at_month(disc_curve, maturity)
    pv_fixed = fixed_rate * accrual * sum(_df_at_month(disc_curve, m) for m in reset_months)
    return pv_float - pv_fixed


def _price_bucket_nii(disc_curve: np.ndarray, reset_months: np.ndarray,
                       freq_m: int = FIXING_FREQ_M) -> float:
    """Floating-leg NII per unit notional over the 1Y (12-month) horizon.

    Fixed leg contributes zero NII delta across scenarios (its rate never
    changes), so only the floating leg's forward-rate exposure matters here.
    """
    total = 0.0
    for m_end in reset_months:
        m_start = m_end - freq_m
        if m_start >= 12.0:
            break
        window = min(freq_m, 12.0 - max(m_start, 0.0))
        if window <= 0.0:
            continue
        df_s = _df_at_month(disc_curve, max(m_start, 0.0))
        df_e = _df_at_month(disc_curve, m_end)
        fwd = (df_s / df_e - 1.0) * (12.0 / freq_m) if df_e > 1e-12 else 0.0
        total += fwd * (window / 12.0)
    return total


def _detect_hedge_direction() -> float:
    """+1.0 if today's real swap book is (net) pay-fixed/receive-float, else -1.0.

    Best-effort DB check; falls back to +1.0 (the conventional direction for
    hedging a bank whose assets are longer-duration/more fixed-rate than its
    liabilities) if the query fails or no real book is found.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "irrbb_sql_setup_ladder", os.path.join(_IRRBB, "sql_setup.py"))
    sql_setup = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(sql_setup)
        from sqlalchemy import text
        q = text("""
            SELECT leg_type, bs_side, SUM(CAST(notional AS FLOAT)) AS notional
            FROM schemat.ir_swaps
            WHERE report_date = :rd AND CAST(product_code AS VARCHAR(4)) = '0000'
            GROUP BY leg_type, bs_side
        """)
        df = pd.read_sql_query(q, sql_setup.engine, params={"rd": REPORT_DATE})
    except Exception as e:
        print(f"  [swap_ladder] direction detection failed ({e}); "
              f"defaulting to pay-fixed/receive-float")
        return 1.0

    fixed_rows = df[df["leg_type"] == "FIXED"]
    if fixed_rows.empty:
        print("  [swap_ladder] no real swap book found; defaulting to pay-fixed/receive-float")
        return 1.0
    # bs_side == 'L' on the fixed leg -> paying fixed (fixed leg is a liability CF)
    dominant = fixed_rows.loc[fixed_rows["notional"].idxmax()]
    return 1.0 if str(dominant["bs_side"]) == "L" else -1.0


def build_ladder_bank(
    tenors: tuple = TENORS_YEARS,
    notional_cap: float = 200e6,
    direction: float | None = None,
) -> LadderBank:
    """Build all (tenor, elapsed_month) buckets and price them via CurveTensors.

    notional_cap is expressed in PLN (per bucket upper bound); the caller
    (stochastic_optimizer.py) converts to a fraction of total_assets for the
    LP bound.
    """
    ct = CurveTensors.load(_CURVE_TENSORS_PATH)
    ccy_idx = ct.currency_index(CURRENCY)

    panel = load_historical_panel()
    ns_ts = fit_ns_timeseries(panel, DIEBOLD_LI_TAU)

    if direction is None:
        direction = _detect_hedge_direction()

    bucket_ids: list = []
    tenor_arr: list = []
    elapsed_arr: list = []
    eve_level = {s: [] for s in SCENARIO_IDS}
    nii_level = {s: [] for s in SCENARIO_IDS}
    ep_carry: list = []

    disc_by_scen = {
        s: ct.disc_factors[ct.scenario_index(s), ccy_idx] for s in SCENARIO_IDS
    }

    margin = direction * IRS_MARGIN_BPS / 10000.0
    for T in tenors:
        total_m = T * 12
        for e in range(total_m):
            reset_months = _reset_months(T, e)
            if len(reset_months) == 0:
                continue
            fixed_rate = _historical_zero_rate(ns_ts, float(T), e) + margin

            for s in SCENARIO_IDS:
                disc_curve = disc_by_scen[s]
                eve_level[s].append(direction * _price_bucket_eve(disc_curve, reset_months, fixed_rate))
                nii_level[s].append(direction * _price_bucket_nii(disc_curve, reset_months))

            year_h = min(12.0, float(reset_months[-1]))
            ep_carry.append(nii_level["base"][-1] - direction * fixed_rate * (year_h / 12.0))

            bucket_ids.append(f"LADDER_{T}Y_E{e:03d}")
            tenor_arr.append(float(T))
            elapsed_arr.append(e)

    eve_level = {s: np.array(v, dtype=float) for s, v in eve_level.items()}
    nii_level = {s: np.array(v, dtype=float) for s, v in nii_level.items()}

    return LadderBank(
        bucket_ids=bucket_ids,
        tenor_years=np.array(tenor_arr, dtype=float),
        elapsed_m=np.array(elapsed_arr, dtype=int),
        eve_level=eve_level,
        nii_level=nii_level,
        ep_carry_base=np.array(ep_carry, dtype=float),
        direction=direction,
        notional_cap=notional_cap,
    )


if __name__ == "__main__":
    bank = build_ladder_bank()
    n = len(bank.bucket_ids)
    print(f"Ladder bank built: {n} buckets (tenors {TENORS_YEARS}, direction "
          f"{'pay-fixed/receive-float' if bank.direction > 0 else 'receive-fixed/pay-float'})")
    for s in SHOCKED_SCENS:
        d_eve = bank.eve_level[s] - bank.eve_level["base"]
        d_nii = bank.nii_level[s] - bank.nii_level["base"]
        print(f"  {s:8s}  delta_EVE/unit range [{d_eve.min():+.5f}, {d_eve.max():+.5f}]"
              f"   delta_NII/unit range [{d_nii.min():+.5f}, {d_nii.max():+.5f}]")
    print(f"  ep_carry_base/unit range [{bank.ep_carry_base.min():+.5f}, {bank.ep_carry_base.max():+.5f}]")
