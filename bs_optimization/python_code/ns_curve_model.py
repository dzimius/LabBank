"""ns_curve_model.py
===================
Dynamic Nelson-Siegel (Diebold-Li) calibration on historical PLN fixings,
used to Monte-Carlo simulate interest-rate curve scenarios for Phase A
(stochastic single-period balance sheet optimization).

Pipeline
--------
1. load_historical_panel()      — 20y daily PLN fixing panel (1D..7Y), mid rates in bps
2. fit_ns_tau() / fit_ns_timeseries() — Diebold-Li NS fit (fixed tau, OLS betas per date)
3. factor_changes_pca()         — PCA of monthly factor changes -> 3 orthogonal risk factors
4. simulate_factor_shocks()     — single-shot driftless-random-walk MC draws of
                                   (d_beta0, d_beta1, d_beta2) over a short horizon
5. fit_mean_reversion() / simulate_mean_reverting_shocks() — AR(1)/OU-style
   mean-reverting alternative for LONGER horizons (12-24mo): a pure random
   walk has no pull back toward historically-typical levels, so over a long
   enough horizon it just gets arbitrarily wide around TODAY's level, however
   unusual that level is historically. Fit on the SAME full 20y monthly
   series factor_changes_pca() already uses (2005-01-03..2025-09-10, PLN
   today ~5.06% level sits at roughly the 75th-90th historical percentile —
   the mean-reverting model is what lets a Monte Carlo scenario set reach
   back down toward historically-typical (or even the ~1-2% seen 2020-2022)
   levels with realistic probability, rather than only by chance if the
   horizon/variance happens to be wide enough.

NS parameterisation (Diebold & Li, 2006), rates in basis points:
    L1(t, tau) = (1 - exp(-t/tau)) / (t/tau)
    L2(t, tau) = L1(t, tau) - exp(-t/tau)
    r(t)       = beta0 + beta1 * L1(t, tau) + beta2 * L2(t, tau)
beta0 = level, beta1 = slope, beta2 = curvature.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXING_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "balance_gen_add_data", "input", "fixing_input.xlsx")
)

TENOR_YEARS = {"1D": 1.0 / 365.0, "1M": 1.0 / 12.0, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "5Y": 5.0, "7Y": 7.0}

# Classical Diebold & Li (2006) choice: tau = 30 months maximises the curvature
# loading L2(t,tau) at the 2-3y tenor, which is where medium-term banking-book
# repricing risk concentrates. NOT re-derived by RMSE grid search: minimising
# raw fit RMSE over this 7-tenor panel is degenerate here — it monotonically
# prefers tau -> 0 (the noisy 1D/1M points dominate the objective and pull the
# curvature hump to the very front end). See fit_ns_tau() docstring.
DIEBOLD_LI_TAU = 2.5


# ─────────────────────────────────────────────────────────────────────────────
# NS loading functions
# ─────────────────────────────────────────────────────────────────────────────

def ns_loadings(t: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Diebold-Li loadings L1 (slope), L2 (curvature) at tenors t (years)."""
    t = np.asarray(t, dtype=float)
    x = np.where(t > 1e-9, t / tau, 1e-9)
    l1 = (1.0 - np.exp(-x)) / x
    l2 = l1 - np.exp(-x)
    return l1, l2


def ns_design_matrix(t: np.ndarray, tau: float) -> np.ndarray:
    """Design matrix X (len(t), 3) = [1, L1, L2] for OLS beta fit."""
    l1, l2 = ns_loadings(t, tau)
    return np.column_stack([np.ones_like(l1), l1, l2])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Historical panel
# ─────────────────────────────────────────────────────────────────────────────

def load_historical_panel(path: str = DEFAULT_FIXING_PATH) -> pd.DataFrame:
    """Load PLN fixing history -> wide panel indexed by date, columns = tenor labels.

    Values are mid rates (average of ASK/BID) in **basis points** (rate_pct * 100),
    matching the units of eba_shock_curves.shock_bps_at_tenors().
    """
    df = pd.read_excel(path)
    df = df[df["currency"] == "PLN"]
    mid = df.groupby(["date", "tenor"])["rate"].mean().mul(100.0)  # % -> bps
    panel = mid.unstack("tenor").sort_index()
    panel = panel[[c for c in TENOR_YEARS if c in panel.columns]]
    panel = panel.dropna(how="any")
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 2. Diebold-Li fit
# ─────────────────────────────────────────────────────────────────────────────

def fit_ns_tau(panel: pd.DataFrame, tau_grid: np.ndarray | None = None) -> tuple[float, pd.DataFrame]:
    """Diagnostic sweep of fit RMSE across a tau grid — NOT used to auto-select
    tau for simulation (see DIEBOLD_LI_TAU).  With only 7 tenor nodes spanning
    1D-7Y, raw-RMSE argmin is degenerate: it monotonically decreases as
    tau -> 0, since a very small tau lets the curvature term fit the noisy
    overnight/1M points almost exactly at the cost of the medium/long end that
    actually matters for balance-sheet repricing risk. Use this sweep only to
    sanity-check that DIEBOLD_LI_TAU sits in a reasonable region of the curve,
    not to pick tau by argmin.

    Returns (argmin_tau, rmse_by_tau DataFrame) — argmin_tau is informational only.
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.25, 10.0, 60)
    t_nodes = np.array([TENOR_YEARS[c] for c in panel.columns])
    y = panel.to_numpy()  # (n_dates, n_tenors)

    rmses = []
    for tau in tau_grid:
        X = ns_design_matrix(t_nodes, tau)              # (n_tenors, 3)
        betas = np.linalg.lstsq(X, y.T, rcond=None)[0]   # (3, n_dates)
        fitted = (X @ betas).T                            # (n_dates, n_tenors)
        resid = y - fitted
        rmses.append(float(np.sqrt(np.mean(resid ** 2))))

    rmse_df = pd.DataFrame({"tau": tau_grid, "rmse_bps": rmses})
    best_tau = float(tau_grid[int(np.argmin(rmses))])
    return best_tau, rmse_df


def fit_ns_timeseries(panel: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Per-date OLS fit of (beta0, beta1, beta2) given a fixed tau.

    Vectorised: same design matrix X for every date, so
        betas_all = pinv(X) @ panel.values.T
    solves all dates in one shot.
    """
    t_nodes = np.array([TENOR_YEARS[c] for c in panel.columns])
    X = ns_design_matrix(t_nodes, tau)                # (n_tenors, 3)
    betas = np.linalg.lstsq(X, panel.to_numpy().T, rcond=None)[0]  # (3, n_dates)
    return pd.DataFrame(
        {"beta0": betas[0], "beta1": betas[1], "beta2": betas[2]},
        index=panel.index,
    )


def ns_fit_quality(panel: pd.DataFrame, ns_ts: pd.DataFrame, tau: float) -> pd.Series:
    """Per-tenor RMSE (bps) of the fitted curve vs. the historical panel — sanity check."""
    t_nodes = np.array([TENOR_YEARS[c] for c in panel.columns])
    X = ns_design_matrix(t_nodes, tau)
    fitted = ns_ts[["beta0", "beta1", "beta2"]].to_numpy() @ X.T
    resid = panel.to_numpy() - fitted
    return pd.Series(np.sqrt(np.mean(resid ** 2, axis=0)), index=panel.columns, name="rmse_bps")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA of monthly factor changes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FactorPCA:
    changes:       pd.DataFrame   # monthly d(beta0,beta1,beta2), bps
    cov:           np.ndarray     # (3, 3)
    eigvals:       np.ndarray     # (3,) descending
    eigvecs:       np.ndarray     # (3, 3) columns = principal directions, descending
    explained_pct: np.ndarray     # (3,) % variance explained per PC


def factor_changes_pca(ns_ts: pd.DataFrame, freq: str = "ME") -> FactorPCA:
    """PCA of month-over-month changes in the NS factors.

    beta0/beta1/beta2 are resampled to month-end (last obs) before differencing,
    matching the 1-month horizon used for the single-period optimizer.
    """
    monthly = ns_ts.resample(freq).last().dropna()
    changes = monthly.diff().dropna()
    cov = changes.cov().to_numpy()

    eigvals, eigvecs = np.linalg.eigh(cov)     # ascending
    order = np.argsort(eigvals)[::-1]          # descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    explained_pct = eigvals / eigvals.sum() * 100.0

    return FactorPCA(changes=changes, cov=cov, eigvals=eigvals, eigvecs=eigvecs,
                      explained_pct=explained_pct)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Monte Carlo simulation of factor shocks
# ─────────────────────────────────────────────────────────────────────────────

def simulate_factor_shocks(
    pca: FactorPCA,
    n_scenarios: int,
    horizon_months: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Monte Carlo draws of (d_beta0, d_beta1, d_beta2) in bps over `horizon_months`.

    Sampled explicitly in PCA space: Z ~ N(0, I) in principal coordinates, scaled by
    sqrt(eigval * horizon_months) (random-walk variance scaling), then rotated back
    via the eigenvector basis.  Mathematically equivalent to drawing directly from
    N(0, horizon_months * cov), but keeps the PCA decomposition explicit for reporting.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_scenarios, 3))                       # (N, 3) in PC space
    scale = np.sqrt(np.clip(pca.eigvals, 0.0, None) * horizon_months)  # (3,)
    draws = (z * scale) @ pca.eigvecs.T                              # (N, 3) back to (beta0,beta1,beta2)
    return draws


@dataclass
class MeanReversionParams:
    kappa:    np.ndarray   # (3,) monthly reversion speed per NS factor, in (0, 1)
    beta_bar: np.ndarray   # (3,) long-run mean per NS factor, bps


def fit_mean_reversion(ns_ts: pd.DataFrame, freq: str = "ME") -> MeanReversionParams:
    """Fit an AR(1)/discretized-Ornstein-Uhlenbeck mean-reversion model per NS
    factor, on the monthly LEVEL series (not the monthly CHANGES
    factor_changes_pca() uses) -- same full historical window, same monthly
    resampling convention, just regressing the change on the lagged LEVEL
    instead of only looking at the changes' covariance:

        d_beta_t = kappa * (beta_bar - beta_{t-1}) + noise

    fit per factor via OLS of d_beta_t on beta_{t-1} (change = a + b*level,
    kappa = -b, beta_bar = a/kappa). Using the SAME full 2005-2025 history
    factor_changes_pca() already uses for the innovation covariance, rather
    than a shorter/different lookback window, so the reversion TARGET and
    the shock COVARIANCE both come from one consistent historical sample.

    Raises ValueError if any factor's fit is non-mean-reverting (kappa <= 0
    or >= 1, i.e. explosive or non-stationary at a monthly step) -- this
    would indicate the historical series itself doesn't support a
    mean-reverting model at this frequency, not something to silently paper
    over with a fallback.
    """
    monthly = ns_ts.resample(freq).last().dropna()
    kappa = np.zeros(3)
    beta_bar = np.zeros(3)
    for i, col in enumerate(("beta0", "beta1", "beta2")):
        level = monthly[col].iloc[:-1].to_numpy(dtype=float)
        change = monthly[col].diff().iloc[1:].to_numpy(dtype=float)
        X = np.column_stack([np.ones_like(level), level])
        (a, b), *_ = np.linalg.lstsq(X, change, rcond=None)
        k = -b
        if not (0.0 < k < 1.0):
            raise ValueError(f"{col}: fitted monthly reversion speed kappa={k:.4f} is not in "
                              f"(0,1) -- historical series does not support a stable mean-reverting fit")
        kappa[i] = k
        beta_bar[i] = a / k
    return MeanReversionParams(kappa=kappa, beta_bar=beta_bar)


def simulate_mean_reverting_shocks(
    pca: FactorPCA,
    mr: MeanReversionParams,
    today_beta: np.ndarray,
    n_scenarios: int,
    horizon_months: float = 24.0,
    seed: int = 42,
    step_months: float = 1.0,
) -> np.ndarray:
    """Monte Carlo draws of (d_beta0, d_beta1, d_beta2) over `horizon_months`,
    simulated via `horizon_months / step_months` monthly steps of a
    mean-reverting (discretized OU) process anchored at today_beta, instead
    of simulate_factor_shocks()'s single-shot driftless random walk.

    Each step applies (1) reversion toward mr.beta_bar at speed mr.kappa and
    (2) a random shock drawn with EXACTLY the same PCA eigenstructure/scaling
    convention simulate_factor_shocks() uses (per-step, scaled by
    step_months) -- so joint factor co-movement is unchanged from the
    already-validated model; only the drift term is new.

    Returns (N, 3) shocks RELATIVE TO today_beta -- a drop-in replacement for
    simulate_factor_shocks()'s output wherever it's consumed (e.g.
    stochastic_optimizer.build_scenario_ep_matrix(),
    curve_scenario_bank.build_mc_scenario_bank()) -- those callers don't need
    to change at all, only how `factor_draws` itself gets generated.
    """
    rng = np.random.default_rng(seed)
    n_steps = max(1, round(horizon_months / step_months))
    scale = np.sqrt(np.clip(pca.eigvals, 0.0, None) * step_months)   # (3,) per-step vol, PC space
    beta = np.tile(np.asarray(today_beta, dtype=float), (n_scenarios, 1))   # (N, 3)
    for _ in range(n_steps):
        z = rng.standard_normal((n_scenarios, 3))
        shock = (z * scale) @ pca.eigvecs.T                          # (N, 3) in beta space
        beta = beta + mr.kappa[None, :] * (mr.beta_bar[None, :] - beta) + shock
    return beta - np.asarray(today_beta, dtype=float)[None, :]


def ns_curve_bps(delta_betas: np.ndarray, tau: float, t_grid: np.ndarray) -> np.ndarray:
    """Reconstruct Δr(t) in bps for one or many draws.

    delta_betas : (3,) or (N, 3)
    Returns     : (len(t_grid),) if input was 1-D, else (N, len(t_grid))
    """
    X = ns_design_matrix(t_grid, tau)                 # (T, 3)
    was_1d = np.ndim(delta_betas) == 1
    out = np.atleast_2d(delta_betas) @ X.T            # (N, T)
    return out[0] if was_1d else out
