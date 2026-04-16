"""extract_curves.py
====================
One-time extraction of yield curves from irrbb.curves → curve_tensors.npz

Reads the shocked and base discount curve grids stored by nii_calc_workflow,
interpolates them onto a regular monthly grid (month 1 … 360), and saves
the discount factors and derived forward rates.

Output shape: (n_scenarios, n_currencies, 360)  for both disc_factors and fwd_rates.
  n_scenarios  — all irrbb scenarios stored in irrbb.curves for report_date
  n_currencies — PLN, EUR, USD (or however many curves exist)
  360          — months 1 … 360 (30Y horizon)
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

REPORT_DATE  = pd.to_datetime("2024-12-31")
N_MONTHS     = 360                        # monthly grid: 1 … 360 months
OUT_PATH     = os.path.join(BASE_DIR, "..", "output", "curve_tensors.npz")

# Mapping currency → disc_curve_name (must match what nii_calc_workflow uses)
DISC_CURVE_MAP = {
    "PLN": "PLN_disc_curve",
    "EUR": "EUR_disc_curve",
    "USD": "USD_disc_curve",
}


def _load_irrbb_curves() -> pd.DataFrame:
    """Load all shocked curves from irrbb.curves for report_date.

    Returns DataFrame with columns:
        scenario_id, curve_name, node_date, d_f
    """
    query = text("""
        SELECT scenario_id, curve_name, node_date, d_f
        FROM   irrbb.curves
        WHERE  report_date = :rd
        ORDER  BY scenario_id, curve_name, node_date
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    df["node_date"] = pd.to_datetime(df["node_date"])
    return df


def _interpolate_to_monthly_grid(
    node_dates: pd.Series,
    d_f_values: pd.Series,
    report_date: pd.Timestamp,
    n_months: int,
) -> np.ndarray:
    """Interpolate discount factors to a regular monthly grid.

    Monthly grid: report_date + m months for m in 1 … n_months.
    Uses linear interpolation in log(d_f) vs n_days space (log-linear = constant
    continuously compounded rate between nodes — standard for discount curves).

    Returns array of shape (n_months,) with d_f[m-1] = d_f at month m.
    """
    # Convert to n_days from report_date
    x_nodes = (node_dates - report_date).dt.days.to_numpy(dtype=float)
    y_nodes = np.log(np.maximum(d_f_values.to_numpy(dtype=float), 1e-12))

    # Target: 30.44 days per month approximation
    x_target = np.arange(1, n_months + 1) * (365.25 / 12)

    # Extrapolate flat at both ends (constant forward rate at tail)
    y_interp = np.interp(x_target, x_nodes, y_nodes,
                         left=y_nodes[0], right=y_nodes[-1])
    return np.exp(y_interp)


def _disc_to_fwd_rates(disc_arr: np.ndarray) -> np.ndarray:
    """Convert monthly discount factor array to annualised forward rates.

    fwd_rate[m] = (d_f[m-1] / d_f[m] - 1) × 12   (monthly compounding → annual)
    For month 0 (period [0, 1M]): use d_f[0] directly.

    Returns array of shape (n_months,).
    """
    n = len(disc_arr)
    fwd = np.empty(n, dtype=float)
    # First period: [report_date, month_1], d_f_start = 1.0
    fwd[0] = (1.0 / disc_arr[0] - 1.0) * 12.0
    # Subsequent periods
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd[1:] = np.where(
            disc_arr[1:] > 0,
            (disc_arr[:-1] / disc_arr[1:] - 1.0) * 12.0,
            0.0,
        )
    return fwd


def build_curve_tensors() -> None:
    """Build and save curve_tensors.npz."""
    print("Loading irrbb.curves from SQL...")
    curves_df = _load_irrbb_curves()

    scenario_ids = sorted(curves_df["scenario_id"].unique())
    currencies   = list(DISC_CURVE_MAP.keys())
    n_scen = len(scenario_ids)
    n_ccy  = len(currencies)

    disc_tensor = np.ones((n_scen, n_ccy, N_MONTHS), dtype=float)
    fwd_tensor  = np.zeros((n_scen, n_ccy, N_MONTHS), dtype=float)

    for s_idx, scen in enumerate(scenario_ids):
        scen_df = curves_df[curves_df["scenario_id"] == scen]
        for c_idx, ccy in enumerate(currencies):
            curve_name = DISC_CURVE_MAP[ccy]
            ccy_df = scen_df[scen_df["curve_name"] == curve_name].sort_values("node_date")
            if ccy_df.empty:
                print(f"  Warning: no curve data for scenario={scen}, curve={curve_name}")
                continue
            disc_arr = _interpolate_to_monthly_grid(
                ccy_df["node_date"], ccy_df["d_f"], REPORT_DATE, N_MONTHS
            )
            fwd_arr  = _disc_to_fwd_rates(disc_arr)
            disc_tensor[s_idx, c_idx] = disc_arr
            fwd_tensor[s_idx, c_idx]  = np.maximum(0.0, fwd_arr)  # EBA 0% floor
        print(f"  Processed scenario: {scen}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(
        OUT_PATH,
        disc_factors  = disc_tensor,   # (n_scen, n_ccy, 360)
        fwd_rates     = fwd_tensor,    # (n_scen, n_ccy, 360)
        scenario_ids  = np.array(scenario_ids),
        currencies    = np.array(currencies),
        curve_names   = np.array([DISC_CURVE_MAP[c] for c in currencies]),
        report_date   = np.array([str(REPORT_DATE.date())]),
        n_months      = np.array([N_MONTHS]),
    )
    print(f"Saved curve_tensors.npz → {OUT_PATH}")
    print(f"  Scenarios: {scenario_ids}")
    print(f"  Currencies: {currencies}")
    print(f"  Grid: {N_MONTHS} monthly points")


if __name__ == "__main__":
    build_curve_tensors()
