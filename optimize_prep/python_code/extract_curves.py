"""extract_curves.py
====================
One-time extraction of yield curves from irrbb.curves
-> opt_prep.monthly_curves SQL table + curve_tensors.npz + Excel

Reads the shocked and base discount curve grids stored by nii_calc_workflow,
interpolates them onto a regular monthly grid (month 1 … 360), and saves
the discount factors and derived annualised forward rates.

Outputs
-------
  SQL: opt_prep.monthly_curves      — 360 rows × n_scenarios × n_currencies
  File: optimize_prep/output/curve_tensors.npz  — numpy tensor for hot path
  File: optimize_prep/output/curves_inspection.xlsx — human-readable grid
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sqlalchemy import text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")

import sql_setup as opt_sql

engine = opt_sql.engine

REPORT_DATE  = pd.to_datetime("2024-12-31")
N_MONTHS     = 360
NPZ_OUT      = os.path.join(BASE_DIR, "..", "output", "curve_tensors.npz")
EXCEL_OUT    = os.path.join(BASE_DIR, "..", "output", "curves_inspection.xlsx")

DISC_CURVE_MAP = {
    "PLN": "PLN_disc_curve",
    "EUR": "EUR_disc_curve",
    "USD": "USD_disc_curve",
}

# Month labels used in Excel (1M, 2M, ... 12M, 18M, 24M, ... 30Y)
_LABEL_MONTHS = list(range(1, 13)) + list(range(18, 61, 6)) + list(range(72, 121, 12)) \
                + [180, 240, 300, 360]


def _load_irrbb_curves() -> pd.DataFrame:
    query = text("""
        SELECT scenario_id, curve_name, n_days, d_f
        FROM   irrbb.curves
        WHERE  report_date = :rd
        ORDER  BY scenario_id, curve_name, n_days
    """)
    df = pd.read_sql_query(query, engine, params={"rd": REPORT_DATE})
    df["n_days"] = df["n_days"].astype(float)
    return df


def _interpolate_to_monthly(
    n_days: pd.Series,
    d_f_values: pd.Series,
    n_months: int,
) -> np.ndarray:
    """Log-linear interpolation to monthly grid.

    Monthly grid: m × (365.25/12) days for m = 1 … n_months.
    Returns float64 array of shape (n_months,).
    """
    x_nodes  = n_days.to_numpy(dtype=float)
    y_nodes  = np.log(np.maximum(d_f_values.to_numpy(dtype=float), 1e-12))
    x_target = np.arange(1, n_months + 1) * (365.25 / 12)
    y_interp = np.interp(x_target, x_nodes, y_nodes,
                         left=y_nodes[0], right=y_nodes[-1])
    return np.exp(y_interp)


def _disc_to_fwd(disc_arr: np.ndarray) -> np.ndarray:
    """Annualised monthly forward rates from monthly discount factors.

    fwd[0] = (1/d_f[0] - 1) × 12        # period [0, 1M]
    fwd[m] = (d_f[m-1] / d_f[m] - 1) × 12   # period [m-1, m] for m ≥ 1
    """
    fwd    = np.empty(len(disc_arr), dtype=float)
    fwd[0] = (1.0 / disc_arr[0] - 1.0) * 12.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd[1:] = np.where(
            disc_arr[1:] > 0,
            (disc_arr[:-1] / disc_arr[1:] - 1.0) * 12.0,
            0.0,
        )
    return np.maximum(0.0, fwd)   # EBA 0% floor


def build_curve_tensors() -> None:
    print("Loading irrbb.curves from SQL...")
    curves_df    = _load_irrbb_curves()
    scenario_ids = sorted(curves_df["scenario_id"].unique())
    currencies   = list(DISC_CURVE_MAP.keys())
    n_scen = len(scenario_ids)
    n_ccy  = len(currencies)

    disc_tensor = np.ones((n_scen, n_ccy, N_MONTHS), dtype=float)
    fwd_tensor  = np.zeros((n_scen, n_ccy, N_MONTHS), dtype=float)

    # ── Interpolate to monthly grid ───────────────────────────────────────────
    for s_idx, scen in enumerate(scenario_ids):
        scen_df = curves_df[curves_df["scenario_id"] == scen]
        for c_idx, ccy in enumerate(currencies):
            cname  = DISC_CURVE_MAP[ccy]
            ccy_df = scen_df[scen_df["curve_name"] == cname].sort_values("n_days")
            if ccy_df.empty:
                print(f"  Warning: no curve for scenario={scen}, curve={cname}")
                continue
            disc = _interpolate_to_monthly(
                ccy_df["n_days"], ccy_df["d_f"], N_MONTHS)
            fwd  = _disc_to_fwd(disc)
            disc_tensor[s_idx, c_idx] = disc
            fwd_tensor[s_idx, c_idx]  = fwd
        print(f"  Interpolated scenario: {scen}")

    # ─────────────────────────────────────────────────────────────────────────
    # Build flat DataFrame for SQL + Excel
    # ─────────────────────────────────────────────────────────────────────────
    rows = []
    month_arr = np.arange(1, N_MONTHS + 1)
    for s_idx, scen in enumerate(scenario_ids):
        for c_idx, ccy in enumerate(currencies):
            for m in range(N_MONTHS):
                rows.append({
                    "report_date":  REPORT_DATE,
                    "scenario_id":  scen,
                    "currency":     ccy,
                    "curve_name":   DISC_CURVE_MAP[ccy],
                    "month_idx":    m + 1,
                    "disc_factor":  float(disc_tensor[s_idx, c_idx, m]),
                    "fwd_rate_ann": float(fwd_tensor[s_idx, c_idx, m]),
                })
    curves_long = pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Write to SQL
    # ─────────────────────────────────────────────────────────────────────────
    print(f"Writing to SQL: opt_prep.monthly_curves  "
          f"({len(curves_long):,} rows)...")
    opt_sql.reset_monthly_curves()
    opt_sql.write_monthly_curves(curves_long)
    print(f"  Written {len(curves_long):,} rows  "
          f"({n_scen} scenarios × {n_ccy} currencies × {N_MONTHS} months)")

    # ─────────────────────────────────────────────────────────────────────────
    # Write Excel inspection file
    # ─────────────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(EXCEL_OUT), exist_ok=True)
    print(f"Writing curves_inspection.xlsx -> {EXCEL_OUT}")

    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
        for ccy in currencies:
            # One sheet per currency: rows = selected month tenors,
            # columns = scenario × (disc_factor | fwd_rate)
            label_rows = []
            for m_label in _LABEL_MONTHS:
                if m_label > N_MONTHS:
                    break
                row_data = {"month": m_label,
                            "tenor": f"{m_label}M" if m_label < 12
                                     else f"{m_label // 12}Y"}
                for scen in scenario_ids:
                    s_idx = scenario_ids.index(scen)
                    c_idx = currencies.index(ccy)
                    df_val  = disc_tensor[s_idx, c_idx, m_label - 1]
                    fwd_val = fwd_tensor[s_idx,  c_idx, m_label - 1]
                    row_data[f"{scen}_df"]      = round(float(df_val),  6)
                    row_data[f"{scen}_fwd_pct"] = round(float(fwd_val) * 100, 4)
                label_rows.append(row_data)
            sheet_df = pd.DataFrame(label_rows)
            sheet_name = f"Curves_{ccy}"
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  Sheet {sheet_name}: {len(sheet_df)} tenor rows × {len(scenario_ids)} scenarios")

        # Summary sheet: forward rates at 1M, 3M, 12M, 60M, 120M across scenarios
        summary_rows = []
        highlight_months = [1, 3, 6, 12, 24, 60, 120, 240]
        for m in highlight_months:
            if m > N_MONTHS:
                break
            for ccy in currencies:
                c_idx = currencies.index(ccy)
                row_data = {
                    "month": m,
                    "tenor": f"{m}M" if m < 12 else f"{m // 12}Y",
                    "currency": ccy,
                }
                for scen in scenario_ids:
                    s_idx = scenario_ids.index(scen)
                    row_data[f"fwd_{scen}_pct"] = round(
                        float(fwd_tensor[s_idx, c_idx, m - 1]) * 100, 4)
                summary_rows.append(row_data)
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="Fwd_rates_summary", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Save .npz for optimizer hot path
    # ─────────────────────────────────────────────────────────────────────────
    np.savez(
        NPZ_OUT,
        disc_factors  = disc_tensor,
        fwd_rates     = fwd_tensor,
        scenario_ids  = np.array(scenario_ids),
        currencies    = np.array(currencies),
        curve_names   = np.array([DISC_CURVE_MAP[c] for c in currencies]),
        report_date   = np.array([str(REPORT_DATE.date())]),
        n_months      = np.array([N_MONTHS]),
    )
    print(f"Saved curve_tensors.npz -> {NPZ_OUT}")

    # ── console summary ───────────────────────────────────────────────────────
    print("\n-- Curve summary --------------------------------------------------")
    for ccy in currencies:
        c_idx = currencies.index(ccy)
        base_idx = scenario_ids.index("base") if "base" in scenario_ids else 0
        fwd_1m  = fwd_tensor[base_idx, c_idx, 0]   * 100
        fwd_12m = fwd_tensor[base_idx, c_idx, 11]  * 100
        fwd_60m = fwd_tensor[base_idx, c_idx, 59]  * 100
        fwd_120m= fwd_tensor[base_idx, c_idx, 119] * 100
        print(f"  {ccy} base fwd rates:  "
              f"1M={fwd_1m:.2f}%  12M={fwd_12m:.2f}%  "
              f"5Y={fwd_60m:.2f}%  10Y={fwd_120m:.2f}%")
    print(f"  Scenarios: {scenario_ids}")


if __name__ == "__main__":
    build_curve_tensors()
