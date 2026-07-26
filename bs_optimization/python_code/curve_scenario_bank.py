"""curve_scenario_bank.py
========================
Builds a bank of Nelson-Siegel curve scenarios for Phase A stress testing,
combining:
  1. The 5 shape x 3 level archetypes used in the `sandbox` app (normal,
     steep, humped, flat, inverted x low, medium, high) -- but instead of
     the sandbox's synthetic forward-rate formulas, each archetype is
     anchored to a REAL historical PLN date whose fitted NS betas
     (beta0=level, beta1=slope, beta2=curvature -- see ns_curve_model.py)
     best represent that shape/level combination.
  2. Small stochastic shifts around each anchor, drawn from the same
     historical factor-change covariance used elsewhere in Phase A
     (ns_curve_model.factor_changes_pca), just re-centered on the anchor
     instead of on today's curve.

Why anchor to real dates and not fit betas to the sandbox curves
------------------------------------------------------------------
The sandbox curves (sandbox/build_scenario_curves.py) are arbitrary
exponential-decay constructs for UI illustration, not calibrated to PLN
market behaviour. Anchoring each shape/level bucket to an actual observed
historical date keeps every scenario auditable ("this stress point is what
PLN rates looked like on <date>") and reuses the existing NS fit instead of
inventing a new curve model.

Classification is a simple, deterministic heuristic on (slope, curvature)
z-scores -- see `classify_dates()`. It is a first cut: buckets with no
historical observations fall back to the nearest shape at any level, and
are flagged `is_fallback=True` in the output so this can be revisited.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ns_curve_model import (
    DIEBOLD_LI_TAU,
    FactorPCA,
    factor_changes_pca,
    fit_ns_timeseries,
    load_historical_panel,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(_HERE, "..", "output"))

SHAPE_TYPES = ["flat", "normal", "steep", "humped", "inverted"]
LEVEL_TYPES = ["low", "medium", "high"]

# Thresholds on standardized slope/curvature z-scores. Heuristic, not fit --
# see module docstring. slope := -beta1 (positive = upward/normal sloping,
# per Diebold-Li sign convention: r(0)=beta0+beta1, r(inf)=beta0).
_FLAT_Z = 0.4
_HUMP_Z = 0.75
_STEEP_Z = 1.0

# Default horizon for the stochastic shifts around each anchor. Deliberately
# much shorter than the 3-month horizon used for the main 500-curve MC (which
# feeds the EP/CVaR objective): PLN NS-factor history includes multi-hundred-bp
# regime moves (2008 crisis, 2020 cuts, 2022 hiking cycle), so a 3-month-ahead
# draw is wide enough to jump a "flat" anchor into "inverted" territory (58%
# of shifts land in a different shape bucket than their anchor at 3mo, checked
# empirically -- see reclassify_shifts()). 0.25 months keeps shifts closer to
# "local jitter around a regime" while still producing real variation; it is a
# tunable assumption, not a fitted value -- reclassify_shifts() reports the
# escape rate every run so this can be revisited.
SHIFT_HORIZON_MONTHS = 0.25


def _classify_shape(z_slope: float, z_curv: float) -> str:
    if abs(z_slope) < _FLAT_Z and abs(z_curv) < _FLAT_Z:
        return "flat"
    if abs(z_curv) >= _HUMP_Z and abs(z_curv) >= abs(z_slope):
        return "humped"
    if z_slope >= _STEEP_Z:
        return "steep"
    if z_slope > 0:
        return "normal"
    return "inverted"


def classify_dates(ns_ts: pd.DataFrame) -> pd.DataFrame:
    """Add shape/level/slope/curvature/z-score columns to a beta0/beta1/beta2 timeseries."""
    out = ns_ts.copy()
    out["slope"] = -out["beta1"]
    out["curvature"] = out["beta2"]

    z_slope = (out["slope"] - out["slope"].mean()) / out["slope"].std()
    z_curv = (out["curvature"] - out["curvature"].mean()) / out["curvature"].std()
    out["z_slope"] = z_slope
    out["z_curv"] = z_curv

    out["shape"] = [_classify_shape(zs, zc) for zs, zc in zip(z_slope, z_curv)]

    low_cut, high_cut = out["beta0"].quantile([1 / 3, 2 / 3])
    out["level"] = np.where(
        out["beta0"] <= low_cut, "low", np.where(out["beta0"] <= high_cut, "medium", "high")
    )
    return out


def select_anchors(ns_ts_classified: pd.DataFrame) -> pd.DataFrame:
    """Pick one representative historical date per (shape, level) bucket.

    Representative = the date within the bucket closest (standardized
    Euclidean distance on beta0/beta1/beta2) to the bucket's own mean.
    Empty buckets fall back to the nearest date matching shape only (any
    level), flagged `is_fallback=True`.
    """
    beta_std = ns_ts_classified[["beta0", "beta1", "beta2"]].std()
    rows = []
    for shape in SHAPE_TYPES:
        shape_pool = ns_ts_classified[ns_ts_classified["shape"] == shape]
        for level in LEVEL_TYPES:
            bucket = shape_pool[shape_pool["level"] == level]
            is_fallback = bucket.empty
            pool = shape_pool if is_fallback else bucket
            if pool.empty:
                # shape itself never observed historically -- skip, report separately
                continue
            target = pool[["beta0", "beta1", "beta2"]].mean()
            dist = (
                ((pool[["beta0", "beta1", "beta2"]] - target) / beta_std) ** 2
            ).sum(axis=1)
            anchor_date = dist.idxmin()
            anchor = pool.loc[anchor_date]
            rows.append(
                {
                    "shape": shape,
                    "level": level,
                    "anchor_date": anchor_date,
                    "beta0": anchor["beta0"],
                    "beta1": anchor["beta1"],
                    "beta2": anchor["beta2"],
                    "is_fallback": is_fallback,
                    "bucket_n_obs": len(bucket),
                }
            )
    return pd.DataFrame(rows)


def _shift_around_anchor(
    anchor_beta: np.ndarray, pca: FactorPCA, n_shifts: int, horizon_months: float, seed: int
) -> np.ndarray:
    """n_shifts draws of (beta0,beta1,beta2), centered on anchor_beta, using the same
    PCA-covariance random-walk scaling as ns_curve_model.simulate_factor_shocks."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_shifts, 3))
    scale = np.sqrt(np.clip(pca.eigvals, 0.0, None) * horizon_months)
    draws = (z * scale) @ pca.eigvecs.T
    return anchor_beta[None, :] + draws


def reclassify_shifts(bank: pd.DataFrame, ns_ts_classified: pd.DataFrame) -> pd.DataFrame:
    """Reclassify each shifted curve's shape using the SAME historical z-score
    reference frame as the original anchor classification, so escapes are an
    apples-to-apples check rather than an artifact of a different reference.
    Adds a `reclass_shape` column; anchors reclassify to themselves by construction.
    """
    mean_slope, std_slope = ns_ts_classified["slope"].mean(), ns_ts_classified["slope"].std()
    mean_curv, std_curv = ns_ts_classified["curvature"].mean(), ns_ts_classified["curvature"].std()

    slope = -bank["beta1"]
    z_slope = (slope - mean_slope) / std_slope
    z_curv = (bank["beta2"] - mean_curv) / std_curv

    out = bank.copy()
    out["reclass_shape"] = [_classify_shape(zs, zc) for zs, zc in zip(z_slope, z_curv)]
    return out


def build_scenario_bank(
    fixing_path: str | None = None,
    n_shifts: int = 10,
    horizon_months: float = SHIFT_HORIZON_MONTHS,
    seed_base: int = 1000,
    tau: float = DIEBOLD_LI_TAU,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full curve scenario bank.

    Returns
    -------
    anchors : (n_buckets, ...) one row per populated (shape, level) bucket
    bank    : anchors + stochastic shifts, one row per curve, columns
              [shape, level, anchor_date, is_anchor, shift_idx, beta0, beta1, beta2]
    """
    panel = load_historical_panel(fixing_path) if fixing_path else load_historical_panel()
    ns_ts = fit_ns_timeseries(panel, tau)
    ns_ts_classified = classify_dates(ns_ts)
    anchors = select_anchors(ns_ts_classified)
    pca = factor_changes_pca(ns_ts)

    rows = []
    for i, anchor in anchors.iterrows():
        anchor_beta = anchor[["beta0", "beta1", "beta2"]].to_numpy(dtype=float)
        rows.append(
            {
                "shape": anchor["shape"],
                "level": anchor["level"],
                "anchor_date": anchor["anchor_date"],
                "is_anchor": True,
                "shift_idx": -1,
                "beta0": anchor_beta[0],
                "beta1": anchor_beta[1],
                "beta2": anchor_beta[2],
            }
        )
        shifted = _shift_around_anchor(
            anchor_beta, pca, n_shifts, horizon_months, seed=seed_base + i
        )
        for k, beta in enumerate(shifted):
            rows.append(
                {
                    "shape": anchor["shape"],
                    "level": anchor["level"],
                    "anchor_date": anchor["anchor_date"],
                    "is_anchor": False,
                    "shift_idx": k,
                    "beta0": beta[0],
                    "beta1": beta[1],
                    "beta2": beta[2],
                }
            )
    bank = pd.DataFrame(rows)
    bank = reclassify_shifts(bank, ns_ts_classified)
    return anchors, bank


def build_mc_scenario_bank(
    factor_draws: np.ndarray,
    today_beta: np.ndarray,
    report_date,
) -> pd.DataFrame:
    """Convert a set of relative Monte Carlo NS-factor draws (as returned by
    ns_curve_model.simulate_factor_shocks -- see optimization_report.ipynb
    Section 5) into the SAME bank schema build_scenario_bank() produces, so
    bank_reprice_at_weights.run_full_bank_at_weights()'s `bank=` parameter can
    reprice them with the exact CF-based engine instead of the 165-curve
    anchor+shift set.

    factor_draws : (N, 3) relative (d_beta0, d_beta1, d_beta2) shocks --
                   RELATIVE to today's curve, not absolute levels.
    today_beta   : (3,) absolute (beta0, beta1, beta2) for the book's actual
                   report date -- e.g. ns_ts.asof(report_date)[["beta0",
                   "beta1","beta2"]].to_numpy(). Must be anchored at the same
                   report date the exact CF engine prices against
                   (anchor_eve_exact.REPORT_DATE), NOT the freshest available
                   historical fit -- those two dates can differ by months,
                   which would silently mismatch the "as of" date between the
                   simulated curve and the book being repriced.
    report_date  : stored in the output's `anchor_date` column only, for
                   bookkeeping/display -- not used in any calculation here.

    Returns a DataFrame with columns [shape, level, anchor_date, is_anchor,
    shift_idx, beta0, beta1, beta2] -- shape/level are both "mc" (these
    curves have no historical shape/level classification, unlike the 165
    anchor-derived curves) and is_anchor is always False (every draw is an
    independent Monte Carlo scenario, not an anchor+shift pair).
    """
    betas = np.asarray(today_beta, dtype=float)[None, :] + np.asarray(factor_draws, dtype=float)
    n = betas.shape[0]
    return pd.DataFrame({
        "shape": ["mc"] * n,
        "level": ["mc"] * n,
        "anchor_date": [report_date] * n,
        "is_anchor": [False] * n,
        "shift_idx": np.arange(n),
        "beta0": betas[:, 0],
        "beta1": betas[:, 1],
        "beta2": betas[:, 2],
    })


def save_scenario_bank(anchors: pd.DataFrame, bank: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "curve_scenario_bank.npz")
    np.savez(
        npz_path,
        shape=bank["shape"].to_numpy(),
        level=bank["level"].to_numpy(),
        is_anchor=bank["is_anchor"].to_numpy(),
        shift_idx=bank["shift_idx"].to_numpy(),
        beta0=bank["beta0"].to_numpy(),
        beta1=bank["beta1"].to_numpy(),
        beta2=bank["beta2"].to_numpy(),
    )
    xlsx_path = os.path.join(out_dir, "curve_scenario_bank_inspection.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        anchors.to_excel(writer, sheet_name="anchors", index=False)
        bank.to_excel(writer, sheet_name="full_bank", index=False)


if __name__ == "__main__":
    anchors, bank = build_scenario_bank()

    print(f"Historical panel classified; {len(anchors)}/{len(SHAPE_TYPES) * len(LEVEL_TYPES)} "
          f"(shape,level) buckets populated (incl. fallbacks).\n")
    print(anchors[["shape", "level", "anchor_date", "is_fallback", "bucket_n_obs"]]
          .to_string(index=False))

    missing = set((s, l) for s in SHAPE_TYPES for l in LEVEL_TYPES) - set(
        zip(anchors["shape"], anchors["level"])
    )
    if missing:
        print(f"\nBuckets with NO historical observation for that shape at all "
              f"(shape has zero dates, any level): {sorted(missing)}")

    print(f"\nTotal scenario bank size: {len(bank)} curves "
          f"({len(anchors)} anchors x (1 + {bank['shift_idx'].max() + 1} shifts))")

    shifts = bank[~bank["is_anchor"]]
    escape_rate = (shifts["reclass_shape"] != shifts["shape"]).mean()
    print(f"\nShift escape rate (shifted curve reclassifies to a DIFFERENT shape "
          f"than its anchor): {escape_rate:.1%} at horizon_months={SHIFT_HORIZON_MONTHS} "
          f"-- see SHIFT_HORIZON_MONTHS comment in this file if this needs retuning.")

    save_scenario_bank(anchors, bank)
    print(f"\nSaved to {OUTPUT_DIR}\\curve_scenario_bank.npz and "
          f"{OUTPUT_DIR}\\curve_scenario_bank_inspection.xlsx")
