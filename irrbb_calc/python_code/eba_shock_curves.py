"""EBA SOT Shock Curves
======================
Implements the 6 EBA SOT supervisory shock scenarios from EBA/RTS/2022/10
(Commission Delegated Regulation (EU) 2024/857), plus a configurable own scenario.

Sources
-------
  EBA/RTS/2022/10 Final Draft RTS on SOTs:
    - Annex I    : currency-specific shock sizes (basis points)
    - Article 3  : shock parameterisation formulas

Scenario IDs
------------
  base   — no shock (current market curve at report_date)
  par_up — parallel up:    Δr(t) = +R̄_par  (all tenors)
  par_dn — parallel down:  Δr(t) = −R̄_par  (all tenors, floor applied)
  steep  — steepener:      Δr(t) = −0.65·R̄_s·exp(−t) + 0.90·R̄_l·(1−exp(−t))
  flat   — flattener:      Δr(t) = +0.80·R̄_s·exp(−t) − 0.75·R̄_l·(1−exp(−t))
  sr_up  — short rate up:  Δr(t) = +R̄_s·exp(−t)
  sr_dn  — short rate down:Δr(t) = −R̄_s·exp(−t)  (floor applied)
  own    — configurable:   parallel shift of own_shock_bps (default −100 bps)

Rate floor for NII: 0 % (all products; current accounts insensitive to down shocks
handled separately in NII calc via gap_cf_ca).

Usage
-----
  import eba_shock_curves as esc
  shocked_df = esc.build_all_shocked_curves(mkt_df, report_date, currency="PLN")
  esc.write_irrbb_curves(engine, shocked_df, report_date)
  esc.generate_shock_excel(mkt_df, report_date, currency="PLN",
                           output_path="output/irrbb_shock_curves.xlsx")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text


# ── Annex I: Currency-specific shock sizes (basis points) ────────────────────
# Source: EBA/RTS/2022/10, Annex I (extracted from the official PDF)
# To recalibrate per Article 2 (every 5 years or on material rate move),
# update the values here — the rest of the code adapts automatically.
SHOCK_PARAMS_BPS: dict[str, dict[str, float]] = {
    # ── First group ──────────────────────────────────────────────────────────
    "ARS": {"par": 400, "short": 500, "long": 300},
    "AUD": {"par": 300, "short": 450, "long": 200},
    "BGN": {"par": 250, "short": 350, "long": 150},
    "BRL": {"par": 400, "short": 500, "long": 300},
    "CAD": {"par": 200, "short": 300, "long": 150},
    "CHF": {"par": 100, "short": 150, "long": 100},
    "CNY": {"par": 250, "short": 300, "long": 150},
    "CZK": {"par": 200, "short": 250, "long": 100},
    "DKK": {"par": 200, "short": 250, "long": 150},
    "EUR": {"par": 200, "short": 250, "long": 100},
    "GBP": {"par": 250, "short": 300, "long": 150},
    # ── Second group ─────────────────────────────────────────────────────────
    "HKD": {"par": 200, "short": 250, "long": 100},
    "HRK": {"par": 250, "short": 400, "long": 200},
    "HUF": {"par": 300, "short": 450, "long": 200},
    "IDR": {"par": 400, "short": 500, "long": 350},
    "INR": {"par": 400, "short": 500, "long": 300},
    "JPY": {"par": 100, "short": 100, "long": 100},
    "KRW": {"par": 300, "short": 400, "long": 200},
    "MXN": {"par": 400, "short": 500, "long": 300},
    "PLN": {"par": 250, "short": 350, "long": 150},  # ← primary currency
    "RON": {"par": 350, "short": 500, "long": 250},
    "RUB": {"par": 400, "short": 500, "long": 300},
    # ── Third group ──────────────────────────────────────────────────────────
    "SAR": {"par": 200, "short": 300, "long": 150},
    "SEK": {"par": 200, "short": 300, "long": 150},
    "SGD": {"par": 150, "short": 200, "long": 100},
    "TRY": {"par": 400, "short": 500, "long": 300},
    "USD": {"par": 200, "short": 300, "long": 150},
    "ZAR": {"par": 400, "short": 500, "long": 300},
}

ALL_SCENARIO_IDS = ["base", "par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn", "own"]

SCENARIO_LABELS = {
    "base":   "Base (no shock)",
    "par_up": "Parallel up",
    "par_dn": "Parallel down",
    "steep":  "Steepener",
    "flat":   "Flattener",
    "sr_up":  "Short rate up",
    "sr_dn":  "Short rate down",
    "own":    "Own scenario",
}

# ── Own parallel scenarios (±100 bps and ±1 bps) ─────────────────────────────
# Used for cf.eve_own_scenarios and cf.nii_own_scenarios analytical tables.
# ±100 bps = sensitivity / stress view; ±1 bps = PV01/NII01 approximation.
OWN_SCENARIO_IDS: list[str] = ["own_100_up", "own_100_dn", "own_1_up", "own_1_dn"]

OWN_SCENARIO_BPS: dict[str, float] = {
    "own_100_up": +100.0,
    "own_100_dn": -100.0,
    "own_1_up":   +1.0,
    "own_1_dn":   -1.0,
}

OWN_SCENARIO_LABELS: dict[str, str] = {
    "own_100_up": "Own +100 bps",
    "own_100_dn": "Own -100 bps",
    "own_1_up":   "Own +1 bps",
    "own_1_dn":   "Own -1 bps",
}


# ── Shock formula (Article 3) ─────────────────────────────────────────────────

def get_shock_params(currency: str) -> dict[str, float]:
    """Return {"par": bps, "short": bps, "long": bps} for a currency.

    Raises ValueError if the currency is not in the Annex I table; in that case
    calibrate per Article 2 and add the entry to SHOCK_PARAMS_BPS.
    """
    if currency not in SHOCK_PARAMS_BPS:
        raise ValueError(
            f"Currency '{currency}' not in EBA/RTS/2022/10 Annex I. "
            "Calibrate per Article 2 and add to SHOCK_PARAMS_BPS."
        )
    return SHOCK_PARAMS_BPS[currency]


def shock_bps_at_tenors(
    t: np.ndarray,
    scenario: str,
    params: dict[str, float],
    own_shock_bps: float = -100.0,
) -> np.ndarray:
    """Return rate shock in **basis points** at each tenor t (in years).

    EBA/RTS/2022/10 Article 3 formulas
    -----------------------------------
    Let R_par, R_s, R_l be the shock magnitudes from Annex I (in bps).
    Define:  f_s(t) = exp(−t),  f_l(t) = 1 − exp(−t)   [t in years]

    par_up / par_dn:  Δr = ±R_par
    sr_up  / sr_dn:   Δr = ±R_s · f_s(t)
    steep:            Δr = −0.65·R_s·f_s(t) + 0.90·R_l·f_l(t)
    flat:             Δr = +0.80·R_s·f_s(t) − 0.75·R_l·f_l(t)
    own:              Δr = own_shock_bps  (parallel)

    Parameters
    ----------
    t              : 1-D array of tenors in years
    scenario       : one of ALL_SCENARIO_IDS
    params         : dict from get_shock_params(currency)
    own_shock_bps  : signed bps for the 'own' scenario
    """
    t = np.asarray(t, dtype=float)

    # Scenarios that don't need currency params — check first to avoid KeyError
    if scenario == "base":
        return np.zeros_like(t)
    elif scenario == "own":
        return np.full_like(t, float(own_shock_bps))
    elif scenario in OWN_SCENARIO_BPS:
        return np.full_like(t, float(OWN_SCENARIO_BPS[scenario]))

    # EBA scenarios — require currency-specific shock params
    R_par = float(params["par"])
    R_s   = float(params["short"])
    R_l   = float(params["long"])

    f_s = np.exp(-t)          # 1 at t=0, → 0 for long tenors
    f_l = 1.0 - np.exp(-t)   # 0 at t=0, → 1 for long tenors

    if scenario == "par_up":
        return np.full_like(t, +R_par)
    elif scenario == "par_dn":
        return np.full_like(t, -R_par)
    elif scenario == "sr_up":
        return +R_s * f_s
    elif scenario == "sr_dn":
        return -R_s * f_s
    elif scenario == "steep":
        return -0.65 * R_s * f_s + 0.90 * R_l * f_l
    elif scenario == "flat":
        return +0.80 * R_s * f_s - 0.75 * R_l * f_l
    else:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Valid EBA: {ALL_SCENARIO_IDS}, Own: {OWN_SCENARIO_IDS}"
        )


# ── Curve shock application ───────────────────────────────────────────────────

def _log_interp_daily_df(
    n_days_nodes: np.ndarray,
    d_f_nodes: np.ndarray,
    max_days: int,
) -> pd.DataFrame:
    """Log-linear interpolation of discount factors to a daily grid 0..max_days.

    Same methodology as cash_flow_calc/sql_setup._interpolate_d_f.
    """
    n_days_nodes = np.asarray(n_days_nodes, dtype=float)
    d_f_nodes    = np.asarray(d_f_nodes,    dtype=float)
    idx  = np.argsort(n_days_nodes)
    x, y = n_days_nodes[idx], d_f_nodes[idx]
    if not np.any(x == 0):
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 1.0)
    else:
        y[x == 0] = 1.0
    grid          = np.arange(0, max_days + 1, dtype=float)
    ln_y_interp   = np.interp(grid, x, np.log(np.clip(y, 1e-12, None)))
    return pd.DataFrame({"n_days": grid.astype(int), "d_f": np.exp(ln_y_interp)})


def apply_shock_to_disc_curve(
    base_df: pd.DataFrame,
    scenario: str,
    params: dict[str, float],
    own_shock_bps: float = -100.0,
    nii_floor_rate: float = 0.0,
) -> pd.DataFrame:
    """Shift a discount curve (n_days, d_f) by the EBA shock for one scenario.

    Algorithm
    ---------
    1. Derive continuous zero rates: r(t) = −ln(d_f) / t
    2. Compute Δr(t) using shock formula (bps → decimal)
    3. Shocked zero rate = max(nii_floor_rate, r(t) + Δr(t))
    4. Back to d_f: exp(−r_shocked · t)
       Note: d_f at t = 0 is always 1.0.

    Returns a copy of base_df with 'd_f' replaced by shocked values.
    """
    df = base_df[["n_days", "d_f"]].copy()
    t  = df["n_days"].to_numpy() / 365.25

    with np.errstate(divide="ignore", invalid="ignore"):
        zero_rate = np.where(
            t > 0,
            -np.log(np.clip(df["d_f"].to_numpy(), 1e-12, None)) / t,
            0.0,
        )

    shock_dec   = shock_bps_at_tenors(t, scenario, params, own_shock_bps) / 10_000.0
    shocked_r   = np.maximum(nii_floor_rate, zero_rate + shock_dec)
    d_f_shocked = np.where(t > 0, np.exp(-shocked_r * t), 1.0)

    df["d_f"]      = d_f_shocked
    df["zero_rt"]  = shocked_r          # continuous zero rate after shock (0 at t=0)
    return df


# ── Build all scenarios ───────────────────────────────────────────────────────

def build_all_shocked_curves(
    mkt_df: pd.DataFrame,
    report_date: pd.Timestamp,
    currency: str,
    own_shock_bps: float = -100.0,
    nii_floor_rate: float = 0.0,
) -> pd.DataFrame:
    """Compute shocked discount curves for all 8 scenarios and all curve_names.

    Parameters
    ----------
    mkt_df        : market curves for report_date.
                    Must have columns: curve_name, n_days, d_f
    report_date   : valuation date
    currency      : currency code (e.g. 'PLN') — determines shock magnitudes
    own_shock_bps : parallel shift for the 'own' scenario (default −100 bps)
    nii_floor_rate: absolute rate floor applied after shock (default 0.0 = 0%)

    Returns
    -------
    DataFrame with columns: report_date, scenario_id, curve_name, n_days, d_f
    Stored in SQL table irrbb.curves (written by write_irrbb_curves).
    """
    params = get_shock_params(currency)
    parts  = []

    for scenario in ALL_SCENARIO_IDS:
        for curve_name, grp in mkt_df.groupby("curve_name"):
            shocked = apply_shock_to_disc_curve(
                grp[["n_days", "d_f"]].reset_index(drop=True),
                scenario, params, own_shock_bps, nii_floor_rate,
            )
            shocked.insert(0, "curve_name",  curve_name)
            shocked.insert(0, "scenario_id", scenario)
            shocked.insert(0, "report_date", report_date)
            parts.append(shocked)

    return pd.concat(parts, ignore_index=True)


def build_own_shocked_curves(
    mkt_df: pd.DataFrame,
    report_date: pd.Timestamp,
    nii_floor_rate: float = 0.0,
) -> pd.DataFrame:
    """Compute shocked discount curves for the 4 own parallel scenarios.

    Scenarios (OWN_SCENARIO_IDS):
      own_100_up (+100 bps) / own_100_dn (−100 bps) — stress / sensitivity
      own_1_up   (+1 bps)   / own_1_dn   (−1 bps)   — PV01 / NII01 approximation

    Returns same column layout as build_all_shocked_curves() so the result
    can be passed directly to write_irrbb_curves() with replace=False.
    """
    parts = []
    for scenario in OWN_SCENARIO_IDS:
        for curve_name, grp in mkt_df.groupby("curve_name"):
            shocked = apply_shock_to_disc_curve(
                grp[["n_days", "d_f"]].reset_index(drop=True),
                scenario=scenario,
                params={},              # unused — OWN_SCENARIO_BPS handles the shift
                nii_floor_rate=nii_floor_rate,
            )
            shocked.insert(0, "curve_name",  curve_name)
            shocked.insert(0, "scenario_id", scenario)
            shocked.insert(0, "report_date", report_date)
            parts.append(shocked)
    return pd.concat(parts, ignore_index=True)


# ── SQL I/O ───────────────────────────────────────────────────────────────────

def write_irrbb_curves(
    engine,
    shocked_df: pd.DataFrame,
    report_date: pd.Timestamp,
    replace: bool = True,
) -> None:
    """Write shocked curves to irrbb.curves.

    replace=True  (default): delete all existing rows for report_date first.
    replace=False           : append-only — used when adding own scenarios after
                              EBA curves have already been written.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            IF OBJECT_ID('irrbb.curves', 'U') IS NULL
            CREATE TABLE irrbb.curves (
                report_date  DATE           NOT NULL,
                scenario_id  VARCHAR(10)    NOT NULL,
                curve_name   VARCHAR(20)    NOT NULL,
                n_days       INT            NOT NULL,
                d_f          DECIMAL(18, 8) NOT NULL,
                zero_rt      DECIMAL(18, 8) NULL,
                CONSTRAINT pk_irrbb_curves
                    PRIMARY KEY (report_date, scenario_id, curve_name, n_days)
            );
        """))
        # Add zero_rt to pre-existing tables that were created without it
        conn.execute(text("""
            IF NOT EXISTS (
                SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'irrbb' AND TABLE_NAME = 'curves'
                  AND COLUMN_NAME = 'zero_rt'
            )
                ALTER TABLE irrbb.curves ADD zero_rt DECIMAL(18, 8) NULL;
        """))
        if replace:
            conn.execute(
                text("DELETE FROM irrbb.curves WHERE report_date = :rd"),
                {"rd": report_date},
            )
    shocked_df.to_sql(
        "curves", engine, schema="irrbb",
        if_exists="append", index=False, chunksize=10_000,
    )


def load_irrbb_disc_curve(
    engine,
    curve_name: str,
    scenario_id: str,
    report_date: pd.Timestamp,
) -> pd.Series:
    """Load a shocked discount curve from irrbb.curves as a daily Series.

    Returns Series indexed by node_date (Timestamp) → d_f.
    Same interface as load_disc_curve in cash_flow_calc/sql_setup.py so the
    shocked NII calculation can swap base ↔ shocked curves transparently.
    """
    q = text("""
        SELECT n_days, d_f
        FROM   irrbb.curves
        WHERE  report_date = :rd
          AND  scenario_id = :sid
          AND  curve_name  = :cn
        ORDER BY n_days
    """)
    df = pd.read_sql_query(
        q, engine, params={"rd": report_date, "sid": scenario_id, "cn": curve_name}
    )
    max_days = int(df["n_days"].max())
    daily    = _log_interp_daily_df(
        df["n_days"].to_numpy(), df["d_f"].to_numpy(), max_days
    )
    daily["node_date"] = report_date + pd.to_timedelta(daily["n_days"], unit="D")
    daily.insert(0, "curve_name", curve_name)
    return daily.set_index(["curve_name", "node_date"])["d_f"].sort_index()


# ── Excel output ──────────────────────────────────────────────────────────────

def generate_shock_excel(
    mkt_df: pd.DataFrame,
    report_date: pd.Timestamp,
    currency: str,
    output_path: str,
    own_shock_bps: float = -100.0,
    nii_floor_rate: float = 0.0,
    horizon_days: int = 365,
) -> None:
    """Write the shocked curves to Excel for review and audit.

    Sheets produced
    ---------------
    Shock_Params
        EBA Annex I parameters for the selected currency plus scenario labels.

    Scenario_Shapes
        Shock in bps and shocked zero rate (%) at the 11 standard EBA tenor
        nodes (0, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y) for every
        curve_name in mkt_df and every scenario.  Use this to verify the
        scenario shapes before running the NII calc.

    Daily_Zero_Rate_{curve_name}
        One sheet per curve: a day-by-day table over the NII horizon
        (report_date + 1 day … report_date + horizon_days).
        Columns: date, tenor_y, base, par_up, par_dn, steep, flat,
                 sr_up, sr_dn, own  (zero rates in %).
        This is the shocked curve the NII engine will use for each repricing date.
    """
    params = get_shock_params(currency)

    # ── Sheet 1: Shock parameters ─────────────────────────────────────────────
    par_rows = [
        {"Parameter": "Currency",                "Value": currency},
        {"Parameter": "Parallel shock (bps)",    "Value": params["par"]},
        {"Parameter": "Short-rate shock (bps)",  "Value": params["short"]},
        {"Parameter": "Long-rate shock (bps)",   "Value": params["long"]},
        {"Parameter": "Own scenario shock (bps)","Value": own_shock_bps},
        {"Parameter": "NII rate floor",           "Value": f"{nii_floor_rate*100:.1f}%"},
        {"Parameter": "Source",
         "Value": "EBA/RTS/2022/10 Annex I (Commission Delegated Reg. (EU) 2024/857)"},
        {"Parameter": "Recalibration", "Value": "At least every 5 years (Article 2)"},
    ]
    for sid in ALL_SCENARIO_IDS:
        par_rows.append({
            "Parameter": f"Scenario '{sid}'",
            "Value": SCENARIO_LABELS[sid],
        })
    df_params = pd.DataFrame(par_rows)

    # ── Sheet 2: Scenario shapes at standard EBA tenors ───────────────────────
    eba_tenors_y = np.array([0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    eba_tenor_labels = [
        "0D", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"
    ]

    shape_rows = []
    for curve_name, grp in mkt_df.groupby("curve_name"):
        # base zero rates at standard EBA tenors (interpolated)
        t_nodes    = grp["n_days"].to_numpy() / 365.25
        df_vals    = grp["d_f"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            r_nodes = np.where(t_nodes > 0, -np.log(np.clip(df_vals, 1e-12, None)) / t_nodes, 0.0)
        r_base_at_eba = np.interp(eba_tenors_y, t_nodes, r_nodes)

        for i, (t, lbl) in enumerate(zip(eba_tenors_y, eba_tenor_labels)):
            r_b = r_base_at_eba[i]
            row = {
                "curve_name":  curve_name,
                "tenor":       lbl,
                "tenor_y":     t,
                "base_rate_%": round(r_b * 100, 4),
                "base_d_f":    round(float(np.exp(-r_b * t)) if t > 0 else 1.0, 8),
            }
            for sid in ALL_SCENARIO_IDS[1:]:   # skip 'base'
                delta_bps = float(shock_bps_at_tenors(
                    np.array([t]), sid, params, own_shock_bps
                )[0])
                r_shocked = max(nii_floor_rate, r_b + delta_bps / 10_000)
                row[f"{sid}_shock_bps"]   = round(delta_bps, 2)
                row[f"{sid}_shocked_r_%"] = round(r_shocked * 100, 4)
                row[f"{sid}_d_f"]         = round(float(np.exp(-r_shocked * t)) if t > 0 else 1.0, 8)
            shape_rows.append(row)
    df_shapes = pd.DataFrame(shape_rows)

    # ── Sheets 3+: Day-by-day zero rates for each curve ───────────────────────
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_params.to_excel(writer,  sheet_name="Shock_Params",      index=False)
        df_shapes.to_excel(writer,  sheet_name="Scenario_Shapes",   index=False)

        for curve_name, grp in mkt_df.groupby("curve_name"):
            # Build base zero-rate function via daily interpolation
            max_node = int(grp["n_days"].max())
            max_days = max(horizon_days, max_node)
            base_daily = _log_interp_daily_df(
                grp["n_days"].to_numpy(), grp["d_f"].to_numpy(), max_days
            )
            t_daily = base_daily["n_days"].to_numpy() / 365.25
            with np.errstate(divide="ignore", invalid="ignore"):
                r_base_daily = np.where(
                    t_daily > 0,
                    -np.log(np.clip(base_daily["d_f"].to_numpy(), 1e-12, None)) / t_daily,
                    0.0,
                )

            daily_rows = {"date": [], "tenor_y": []}
            for sid in ALL_SCENARIO_IDS:
                daily_rows[sid]           = []   # zero rate %
                daily_rows[f"{sid}_df"]   = []   # discount factor

            for d in range(1, horizon_days + 1):
                t   = d / 365.25
                r_b = float(np.interp(t, t_daily, r_base_daily))
                daily_rows["date"].append(report_date + pd.Timedelta(days=d))
                daily_rows["tenor_y"].append(round(t, 6))
                daily_rows["base"].append(round(r_b * 100, 4))
                daily_rows["base_df"].append(round(float(np.exp(-r_b * t)), 8))
                for sid in ALL_SCENARIO_IDS[1:]:
                    delta_bps = float(shock_bps_at_tenors(
                        np.array([t]), sid, params, own_shock_bps
                    )[0])
                    r_sh = max(nii_floor_rate, r_b + delta_bps / 10_000)
                    daily_rows[sid].append(round(r_sh * 100, 4))
                    daily_rows[f"{sid}_df"].append(round(float(np.exp(-r_sh * t)), 8))

            # Interleave (zero_rt%, d_f) per scenario for easy side-by-side comparison
            col_order = ["date", "tenor_y"]
            for s in ALL_SCENARIO_IDS:
                col_order += [s, f"{s}_df"]
            df_daily = pd.DataFrame(daily_rows)[col_order]

            renamed_cols = ["date", "tenor_y"]
            for s in ALL_SCENARIO_IDS:
                renamed_cols += [f"{SCENARIO_LABELS[s]} zero_rt_%", f"{SCENARIO_LABELS[s]} d_f"]
            df_daily.columns = pd.Index(renamed_cols)
            sheet = f"Daily_{curve_name}"[:31]   # Excel 31-char sheet name limit
            df_daily.to_excel(writer, sheet_name=sheet, index=False)

    print(f"  Shock curves Excel written to {output_path}")
