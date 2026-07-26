"""hedge_optimizer.py
====================
Small standalone LP that chooses the optimal notional for each of the 300
swap_ladder.py buckets, with the real banking book FIXED at today's actual
structure (no balance-sheet re-optimization -- see plan doc for why).

    min_x   penalty_weight * mean(v)
    s.t.    EVE floor (hard, all 165x6 curve/scenario combos):
                today_eve_pct[k] + A_eve_pct[k,:] @ x >= EVE_FLOOR
            NII floor (soft, via slack v[k] >= 0):
                today_nii_pct[k] + A_nii_pct[k,:] @ x + v[k] >= NII_FLOOR
            0 <= x[i] <= notional_cap

The ladder's own carry (ep_carry_pln) is NOT in the objective (2026-07-09
change, per user feedback) -- swap notional is used ONLY for its effect on
the EVE/NII constraint rows above, never rewarded directly for its own P&L.
Before this change the LP was `max ep_carry @ x - penalty*mean(v)`, which let
the optimizer treat the ladder as a profit source in its own right and buy
up to the notional cap chasing carry (worsened further by two now-fixed
mispricing issues in swap_ladder.py: zero transaction margin, and seasoned
buckets' historical-rate-drift windfall -- see IRS_MARGIN_BPS and
seasoned_notional_cap there). Removing carry from the objective means a
bucket only enters the solution when it's still reducing binding breach
severity; once a row is satisfied, growing that bucket further is objective-
neutral, so the LP has no incentive to keep piling in. ep_carry_total is
still computed and returned for reporting -- it's informational now, not
what's being maximized.

Every coefficient here is exact (not a linear approximation): the ladder side
comes from swap_ladder.py's analytic DCF pricing against all 165 curves, and
the banking-book side comes from anchor_sot_exact.py's already-computed exact
165-curve reprice of today's real book (unchanged across all 3 comparison
views in hedge_comparison.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linprog

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import swap_ladder as sl                # noqa: E402
import anchor_sot_exact as ase          # noqa: E402

SOT_EVE_FLOOR_PCT_T1 = ase.SOT_EVE_FLOOR_PCT_T1
SOT_NII_FLOOR_PCT_T1 = ase.SOT_NII_FLOOR_PCT_T1


def load_today_book_165(bank_xlsx: str | None = None) -> pd.DataFrame:
    """Today's real book's exact 165x6 sensitivity -- loads the cached run if
    present, else recomputes via anchor_sot_exact.run_full_bank()."""
    if bank_xlsx is None:
        bank_xlsx = os.path.join(ase.OUTPUT_DIR, "anchor_sot_exact_full_bank.xlsx")
    if os.path.exists(bank_xlsx):
        return pd.read_excel(bank_xlsx, sheet_name="all_scenarios")
    print("  [hedge_optimizer] anchor_sot_exact_full_bank.xlsx not found; recomputing...")
    return ase.run_full_bank()


_KEY_COLS = ["shape", "level", "is_anchor", "shift_idx"]


def _aligned_today_values(today: pd.DataFrame, ladder: dict, scenario: str, col: str) -> np.ndarray:
    """today[col] for `scenario`, reordered to match ladder's curve row order."""
    curve_df = pd.DataFrame({
        **{k: ladder[k] for k in _KEY_COLS},
        "_curve_idx": np.arange(len(ladder["shape"])),
    })
    scen_today = today[today["scenario"] == scenario][_KEY_COLS + [col]]
    merged = curve_df.merge(scen_today, on=_KEY_COLS, how="left").sort_values("_curve_idx")
    if merged[col].isna().any():
        raise ValueError(f"Could not align {merged[col].isna().sum()} curve rows for "
                          f"scenario={scenario!r}, col={col!r} -- curve_scenario_bank.py "
                          f"output may have changed since anchor_sot_exact_full_bank.xlsx "
                          f"was generated. Re-run anchor_sot_exact.py.")
    return merged[col].to_numpy(dtype=float)


def solve_optimal_ladder(
    notional_cap: float | None = 1e9,
    total_notional_cap: float | None = 5e9,
    penalty_weight: float = 50.0,
    tenors: tuple = sl.TENORS_YEARS,
    seasoned_elapsed_months: int = 12,
    seasoned_notional_cap: float | None = 1e9,
) -> dict:
    """penalty_weight is a unitless multiplier on the PLN-equivalent cost of
    average NII-SOT breach severity (1.0 = one PLN of breach severity treated
    as costing exactly one PLN, same scale as ep_carry) -- >1 prioritises
    breach reduction over the ladder's own carry economics.

    Default of 50 was picked by sweeping 5/50/500/5000: below ~50 the ladder
    barely reduces the breach count (it just chases carry); above ~500 the
    marginal breach reduction flattens out (~135-140 of 990 combos) while EP
    carry turns negative -- 50 is the knee of that curve, still net-positive
    carry with a real (~30%) breach reduction vs baseline.

    notional_cap / total_notional_cap are NOT arbitrary: some near-maturity
    buckets have almost-zero EVE sensitivity (nearly matured, discount-factor
    differences are tiny) but still carry positive carry, so with no bound at
    all the LP is genuinely UNBOUNDED (it scales that bucket to infinity --
    confirmed empirically, HiGHS reports "model_status is Unbounded"). A
    per-bucket cap alone forces the optimizer to spread across many buckets
    at the ceiling; adding an aggregate total_notional_cap on top gives it
    room to redistribute instead of piling every bucket up to its individual
    limit.

    seasoned_elapsed_months / seasoned_notional_cap tighten this further:
    a bucket's fixed rate is fit to the historical curve AT ITS ORIGINATION
    DATE (see swap_ladder._historical_zero_rate), so a bucket originated
    `seasoned_elapsed_months`+ months ago can carry a large "windfall" versus
    today's curve purely from real rate drift since then (PLN rates moved a
    lot 2022-2024) -- a different, larger effect than the IRS_MARGIN_BPS
    transaction-cost adjustment, and the reason the optimizer kept piling
    into the same handful of ~2-3y-seasoned buckets even after that fix.
    This is economically real (a bank COULD have transacted back then and
    would genuinely hold that windfall), but concentrating the whole ladder
    into a few such buckets overstates how much of it is realistically
    available today, so seasoned buckets get their OWN smaller sub-cap
    (default 1e9, same order as one bucket's own cap) on top of the general
    notional_cap/total_notional_cap -- unseasoned (near-today-dated) buckets
    are unaffected and still share the full total_notional_cap headroom.
    Set seasoned_notional_cap=None to disable this sub-cap.
    """
    today  = load_today_book_165()
    ladder = sl.price_ladder_against_curve_bank(tenors=tenors)
    t1 = ase.aee.irrbb_sql_setup.load_tier1_capital(ase.REPORT_DATE)

    bucket_ids = ladder["bucket_ids"]
    n_buckets  = len(bucket_ids)
    n_curves   = len(ladder["shape"])
    seasoned_mask = ladder["elapsed_m"] > seasoned_elapsed_months

    # Solve for x in units of NOTIONAL_UNIT PLN, not raw PLN: the ladder's
    # per-unit-notional sensitivity (PLN / T1) is ~1e-10 while notional bounds
    # are ~1e9 PLN -- a ~19-order-of-magnitude spread in one LP that made
    # HiGHS return a solution that reports "Optimal" but does NOT actually
    # satisfy A_ub @ x <= b_ub when checked directly (confirmed empirically:
    # res.slack and a manual b_ub - A_ub@res.x recomputation disagreed by
    # several points on identical rows -- a numerical-conditioning failure,
    # not a logic bug). Rescaling x to millions of PLN brings every
    # coefficient into a comparable order of magnitude and fixes it.
    NOTIONAL_UNIT = 1e6

    rows_eve, rhs_eve = [], []
    rows_nii, rhs_nii = [], []
    for s in sl.SHOCKED_SCENS:
        today_eve_pct = _aligned_today_values(today, ladder, s, "delta_eve_pct_t1")   # (n_curves,)
        today_nii_pct = _aligned_today_values(today, ladder, s, "delta_nii_pct_t1")

        a_eve_pct = ladder["delta_eve_pln"][s] / t1 * 100.0 * NOTIONAL_UNIT    # (n_curves, n_buckets)
        a_nii_pct = ladder["delta_nii_pln"][s] / t1 * 100.0 * NOTIONAL_UNIT

        rows_eve.append(-a_eve_pct)
        rhs_eve.append(today_eve_pct - SOT_EVE_FLOOR_PCT_T1)

        rows_nii.append(-a_nii_pct)
        rhs_nii.append(today_nii_pct - SOT_NII_FLOOR_PCT_T1)

    A_eve = np.vstack(rows_eve)     # (n_curves * 6, n_buckets)
    b_eve = np.concatenate(rhs_eve)
    A_nii = np.vstack(rows_nii)
    b_nii = np.concatenate(rhs_nii)
    n_combo = A_eve.shape[0]
    assert n_combo == n_curves * len(sl.SHOCKED_SCENS)

    # variables z = [x_u (n_buckets, units of NOTIONAL_UNIT PLN), v (n_combo, pp of T1)]
    # ep_carry is NOT in the objective (see module docstring) -- x_u's objective
    # coefficient is 0, so a bucket only enters the solution via its effect on
    # the EVE/NII constraint rows below (i.e. reducing binding breach slack).
    slack_pln_per_unit = penalty_weight * (t1 / 100.0) / n_combo
    c = np.concatenate([np.zeros(n_buckets), np.full(n_combo, slack_pln_per_unit)])

    A_ub_eve = np.hstack([A_eve, np.zeros((n_combo, n_combo))])
    A_ub_nii = np.hstack([A_nii, -np.eye(n_combo)])
    rows = [A_ub_eve, A_ub_nii]
    rhs  = [b_eve, b_nii]
    if total_notional_cap is not None:
        rows.append(np.concatenate([np.ones(n_buckets), np.zeros(n_combo)])[None, :])
        rhs.append(np.array([total_notional_cap / NOTIONAL_UNIT]))
    if seasoned_notional_cap is not None and seasoned_mask.any():
        rows.append(np.concatenate([seasoned_mask.astype(float), np.zeros(n_combo)])[None, :])
        rhs.append(np.array([seasoned_notional_cap / NOTIONAL_UNIT]))
    A_ub = np.vstack(rows)
    b_ub = np.concatenate(rhs)

    cap_u = None if notional_cap is None else notional_cap / NOTIONAL_UNIT
    bounds = [(0.0, cap_u)] * n_buckets + [(0.0, None)] * n_combo   # notional_cap=None -> uncapped

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    x_u = res.x[:n_buckets] if res.success else np.zeros(n_buckets)
    v   = res.x[n_buckets:] if res.success else np.zeros(n_combo)
    x   = x_u * NOTIONAL_UNIT    # back to PLN for the external API

    if res.success:
        resid = A_ub @ res.x - b_ub
        if resid.max() > 1e-3:
            print(f"  [hedge_optimizer] WARNING: solver reported success but max "
                  f"constraint residual is {resid.max():.4f} (should be <=0) -- "
                  f"treat this solution with caution.")

    return dict(
        success=res.success, message=res.message,
        bucket_ids=bucket_ids, notional=x,
        breach_slack_pct_t1=v,
        ep_carry_total=float(np.dot(ladder["ep_carry_pln"], x)),
        tier1_capital=t1,
        direction=ladder["direction"],
        ladder=ladder,
    )


if __name__ == "__main__":
    result = solve_optimal_ladder()
    print(f"Solve: {'SUCCESS' if result['success'] else 'FAILED'} - {result['message']}")
    x = result["notional"]
    nz = x > 1.0
    print(f"Active buckets: {int(nz.sum())} / {len(result['bucket_ids'])}"
          f"   total added notional: {x.sum()/1e6:,.1f}M PLN"
          f"   ({'receive-fixed/pay-float' if result['direction'] < 0 else 'pay-fixed/receive-float'})")
    print(f"EP carry from ladder: {result['ep_carry_total']/1e6:,.2f}M PLN/yr")
    print(f"Residual NII breach slack (pp T1): max={result['breach_slack_pct_t1'].max():.3f}"
          f"  mean={result['breach_slack_pct_t1'].mean():.4f}"
          f"  n_active={(result['breach_slack_pct_t1'] > 1e-6).sum()}")
    top = sorted(zip(result["bucket_ids"], x.tolist()), key=lambda t: -t[1])[:10]
    print("Top buckets by notional:")
    for bid, notional in top:
        if notional > 1.0:
            print(f"  {bid:16s}  {notional/1e6:,.1f}M")
