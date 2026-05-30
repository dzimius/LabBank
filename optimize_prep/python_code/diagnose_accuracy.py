"""diagnose_accuracy.py
======================
Granular comparison of approx vs exact NII and EVE by product_code × scenario.

Outputs:
  optimize_prep/output/accuracy_by_product.xlsx
    Sheet "NII_by_product"      — NII base and delta_NII errors per product_code
    Sheet "EVE_by_product"      — EVE base and delta_EVE errors per product_code
    Sheet "EVE_fixed_vs_float"  — EVE delta split by rate_type (F vs V) per product_code
    Sheet "NII_cohort_share"    — NII interest share per cohort (for share quality check)
    Sheet "root_cause_summary"  — per-product ranked root-cause table

Run after: opt_prep_workflow.py (extract_params + extract_curves steps).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from bs_vector import BalanceSheetParams, CurveTensors
from nii_fast import _cohort_delta_nii_sched

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

REPORT_DATE  = pd.to_datetime("2024-12-31")
OUT_PATH     = os.path.join(BASE_DIR, "..", "output", "accuracy_by_product.xlsx")
NPZ_PARAMS   = os.path.join(BASE_DIR, "..", "output", "product_params.npz")
NPZ_CURVES   = os.path.join(BASE_DIR, "..", "output", "curve_tensors.npz")

SHOCKED_SCENARIOS = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]

_COHORT_CODES = frozenset({"1000","1100","2000","2100","4100","3000","3100","7060","7900","5000"})
_SINGLE_CODES = frozenset({"3500","6000","8000","6300","5300","5100","5400"})
_C_IN = ", ".join(f"'{c}'" for c in sorted(_COHORT_CODES))


# -----------------------------------------------------------
# Exact results from irrbb
# -----------------------------------------------------------

def _exact_nii() -> pd.DataFrame:
    """irrbb.nii_results aggregated by (product_code, bs_side, currency, scenario_id)."""
    q = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, scenario_id,
               SUM(nii_total) AS nii_total
        FROM irrbb.nii_results
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


def _exact_eve() -> pd.DataFrame:
    """irrbb.eve_results aggregated by (product_code, bs_side, currency, scenario_id)."""
    q = text("""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, scenario_id,
               SUM(pv_total) AS pv_total
        FROM irrbb.eve_results
        WHERE report_date = :rd
        GROUP BY product_code, bs_side, currency, scenario_id
    """)
    df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


def _cohort_rate_type() -> pd.DataFrame:
    """Dominant rate_type (F/V) and balance per cohort (product_code, bs_side, currency).

    Used to understand how much of each product is fixed vs floating.
    """
    q = text(f"""
        SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, rate_type,
               SUM(balance_amt) AS balance_amt
        FROM sched.loans
        WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency, rate_type
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)),
               bs_side, currency, rate_type,
               SUM(balance_amt)
        FROM sched.deposits
        WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency, rate_type
        UNION ALL
        SELECT CAST(product_code AS VARCHAR(4)),
               bs_side, currency, rate_type,
               SUM(balance_amt)
        FROM sched.fin_inst
        WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        GROUP BY product_code, bs_side, currency, rate_type
    """)
    try:
        df = pd.read_sql_query(q, engine)
        df["product_code"] = df["product_code"].astype(str)
        return df
    except Exception as e:
        print(f"  [warn] rate_type query: {e}")
        return pd.DataFrame(columns=["product_code", "bs_side", "currency", "rate_type", "balance_amt"])


# -----------------------------------------------------------
# Approx results from opt_prep
# -----------------------------------------------------------

def _approx_params() -> pd.DataFrame:
    """opt_prep.product_params — one row per cohort/single-row entry."""
    q = text("""
        SELECT cohort_id, CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, is_cohort,
               CAST(balance_amt AS FLOAT)    AS balance_amt,
               CAST(nii_unit_rate AS FLOAT)  AS nii_unit_rate,
               CAST(eve_pv_factor AS FLOAT)  AS eve_pv_factor,
               CAST(d_mod        AS FLOAT)   AS d_mod
        FROM opt_prep.product_params
        WHERE report_date = :rd
    """)
    df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


def _approx_scenarios() -> pd.DataFrame:
    """opt_prep.product_scenario_params — one row per cohort × scenario."""
    q = text("""
        SELECT cohort_id, CAST(product_code AS VARCHAR(4)) AS product_code,
               bs_side, currency, scenario_id,
               CAST(delta_nii_unit AS FLOAT) AS delta_nii_unit,
               CAST(delta_eve_unit AS FLOAT) AS delta_eve_unit
        FROM opt_prep.product_scenario_params
        WHERE report_date = :rd
    """)
    df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
    df["product_code"] = df["product_code"].astype(str)
    return df


# -----------------------------------------------------------
# Build product-level approx NII / EVE
# -----------------------------------------------------------

def _build_approx_nii_eve(params: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Aggregate approx NII / EVE per (product_code, bs_side, currency).

    approx_nii  = Σ_cohorts  balance × nii_unit_rate
    approx_eve  = Σ_cohorts  balance × eve_pv_factor
    approx_dNII = Σ_cohorts  balance × delta_nii_unit  (per scenario)
    approx_dEVE = Σ_cohorts  balance × delta_eve_unit  (per scenario)
    """
    params = params.copy()
    params["approx_nii"] = params["balance_amt"] * params["nii_unit_rate"]
    params["approx_eve"] = params["balance_amt"] * params["eve_pv_factor"]

    base_agg = (params
                .groupby(["product_code", "bs_side", "currency"])
                .agg(approx_nii=("approx_nii", "sum"),
                     approx_eve=("approx_eve", "sum"),
                     balance_amt=("balance_amt", "sum"))
                .reset_index())

    # Scenario deltas
    scen_joined = scenarios.merge(
        params[["cohort_id", "balance_amt"]], on="cohort_id", how="left")
    scen_joined["approx_dNII"] = scen_joined["balance_amt"] * scen_joined["delta_nii_unit"]
    scen_joined["approx_dEVE"] = scen_joined["balance_amt"] * scen_joined["delta_eve_unit"]

    scen_agg = (scen_joined
                .groupby(["product_code", "bs_side", "currency", "scenario_id"])
                .agg(approx_dNII=("approx_dNII", "sum"),
                     approx_dEVE=("approx_dEVE", "sum"))
                .reset_index())

    return base_agg, scen_agg


# -----------------------------------------------------------
# Compare and annotate errors
# -----------------------------------------------------------

def _pct(approx, exact):
    """Percentage error; NaN when exact is near zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(np.abs(exact) > 1.0,
                       (approx - exact) / np.abs(exact) * 100.0,
                       np.nan)
    return err


def _build_nii_comparison(exact_nii, base_agg, scen_agg) -> pd.DataFrame:
    key = ["product_code", "bs_side", "currency"]

    # base
    exact_base = (exact_nii[exact_nii["scenario_id"] == "base"]
                  .rename(columns={"nii_total": "exact_nii"})
                  .drop(columns="scenario_id"))
    merged = base_agg.merge(exact_base, on=key, how="outer").fillna(0)
    merged["nii_err_pct"] = _pct(merged["approx_nii"], merged["exact_nii"])
    merged["nii_abs_err"] = merged["approx_nii"] - merged["exact_nii"]

    # delta NII per scenario
    exact_base_val = exact_nii[exact_nii["scenario_id"] == "base"].copy()
    exact_base_val = exact_base_val.rename(columns={"nii_total": "nii_base"})[key + ["nii_base"]]
    exact_shocked  = exact_nii[exact_nii["scenario_id"] != "base"].copy()
    exact_shocked  = exact_shocked.merge(exact_base_val, on=key, how="left")
    exact_shocked["exact_dNII"] = exact_shocked["nii_total"] - exact_shocked["nii_base"]

    delta_merged = scen_agg.merge(
        exact_shocked[key + ["scenario_id", "exact_dNII"]], on=key + ["scenario_id"], how="outer"
    ).fillna(0)
    delta_merged["dNII_err_pct"] = _pct(delta_merged["approx_dNII"], delta_merged["exact_dNII"])
    delta_merged["dNII_abs_err"] = delta_merged["approx_dNII"] - delta_merged["exact_dNII"]

    return merged, delta_merged


def _build_eve_comparison(exact_eve, base_agg, scen_agg) -> pd.DataFrame:
    key = ["product_code", "bs_side", "currency"]

    exact_base = (exact_eve[exact_eve["scenario_id"] == "base"]
                  .rename(columns={"pv_total": "exact_eve"})
                  .drop(columns="scenario_id"))
    merged = base_agg.merge(exact_base, on=key, how="outer").fillna(0)
    merged["eve_err_pct"] = _pct(merged["approx_eve"], merged["exact_eve"])
    merged["eve_abs_err"] = merged["approx_eve"] - merged["exact_eve"]

    exact_base_val = exact_eve[exact_eve["scenario_id"] == "base"].copy()
    exact_base_val = exact_base_val.rename(columns={"pv_total": "eve_base"})[key + ["eve_base"]]
    exact_shocked  = exact_eve[exact_eve["scenario_id"] != "base"].copy()
    exact_shocked  = exact_shocked.merge(exact_base_val, on=key, how="left")
    exact_shocked["exact_dEVE"] = exact_shocked["pv_total"] - exact_shocked["eve_base"]

    delta_merged = scen_agg.merge(
        exact_shocked[key + ["scenario_id", "exact_dEVE"]], on=key + ["scenario_id"], how="outer"
    ).fillna(0)
    delta_merged["dEVE_err_pct"] = _pct(delta_merged["approx_dEVE"], delta_merged["exact_dEVE"])
    delta_merged["dEVE_abs_err"] = delta_merged["approx_dEVE"] - delta_merged["exact_dEVE"]

    return merged, delta_merged


def _root_cause_table(nii_base, eve_base, nii_delta, eve_delta, rate_type_df) -> pd.DataFrame:
    """Summary table: product × root-cause dimension ranked by absolute error."""
    key = ["product_code", "bs_side", "currency"]

    # biggest NII base errors
    nii_b = nii_base.copy()
    nii_b["abs_nii_err_M"] = (nii_b["nii_abs_err"] / 1e6).round(1)

    # biggest EVE base errors
    eve_b = eve_base.copy()
    eve_b["abs_eve_err_M"] = (eve_b["eve_abs_err"] / 1e6).round(1)

    # worst delta_NII scenario per product — sort by abs error, take first row per group
    _dnii = (nii_delta
             .assign(abs_err=lambda x: x["dNII_abs_err"].abs())
             .sort_values("abs_err", ascending=False))
    worst_dnii = (_dnii.groupby(key)
                  .first()[["scenario_id", "dNII_abs_err"]]
                  .rename(columns={"scenario_id": "worst_dNII_scenario",
                                   "dNII_abs_err": "_raw"})
                  .reset_index())
    worst_dnii["worst_dNII_err_M"] = (worst_dnii.pop("_raw") / 1e6).round(1)

    # worst delta_EVE scenario per product
    _deve = (eve_delta
             .assign(abs_err=lambda x: x["dEVE_abs_err"].abs())
             .sort_values("abs_err", ascending=False))
    worst_deve = (_deve.groupby(key)
                  .first()[["scenario_id", "dEVE_abs_err"]]
                  .rename(columns={"scenario_id": "worst_dEVE_scenario",
                                   "dEVE_abs_err": "_raw"})
                  .reset_index())
    worst_deve["worst_dEVE_err_M"] = (worst_deve.pop("_raw") / 1e6).round(1)

    # float_pct: % of product balance that is floating-rate
    if not rate_type_df.empty:
        total_bal = (rate_type_df.groupby(key)["balance_amt"].sum()
                     .reset_index().rename(columns={"balance_amt": "total_bal"}))
        float_bal = (rate_type_df[rate_type_df["rate_type"] != "F"]
                     .groupby(key)["balance_amt"].sum()
                     .reset_index().rename(columns={"balance_amt": "float_bal"}))
        rt = total_bal.merge(float_bal, on=key, how="left").fillna(0)
        rt["float_pct"] = (rt["float_bal"] / rt["total_bal"] * 100).round(1)
    else:
        rt = pd.DataFrame(columns=key + ["float_pct"])

    summary = (nii_b[key + ["balance_amt", "abs_nii_err_M", "nii_err_pct"]]
               .merge(eve_b[key + ["abs_eve_err_M", "eve_err_pct"]], on=key, how="outer")
               .merge(worst_dnii, on=key, how="left")
               .merge(worst_deve, on=key, how="left")
               .merge(rt[key + ["float_pct"]], on=key, how="left"))

    # rank by combined EVE + NII absolute error
    summary["total_abs_err_M"] = (summary["abs_nii_err_M"].abs().fillna(0)
                                  + summary["abs_eve_err_M"].abs().fillna(0))
    return summary.sort_values("total_abs_err_M", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------
# EVE fixed vs floating breakdown
# -----------------------------------------------------------

def _eve_fixed_float_breakdown(exact_eve, rate_type_df) -> pd.DataFrame:
    """Compare rate_type balance composition with exact delta_EVE to assess
    how much of the error comes from treating floating as fixed.

    For floating-rate products: exact delta_EVE is driven by repricing gap only
    (small D_mod ~= repricing_tenor). Our approx uses full remaining maturity
    -> systematically overestimates.
    """
    key = ["product_code", "bs_side", "currency"]
    if rate_type_df.empty:
        return pd.DataFrame()

    total_bal = (rate_type_df.groupby(key)["balance_amt"].sum()
                 .reset_index().rename(columns={"balance_amt": "total_bal"}))
    fixed_bal  = (rate_type_df[rate_type_df["rate_type"] == "F"]
                  .groupby(key)["balance_amt"].sum()
                  .reset_index().rename(columns={"balance_amt": "fixed_bal"}))
    float_bal  = (rate_type_df[rate_type_df["rate_type"] != "F"]
                  .groupby(key)["balance_amt"].sum()
                  .reset_index().rename(columns={"balance_amt": "float_bal"}))

    rt = (total_bal
          .merge(fixed_bal, on=key, how="left")
          .merge(float_bal, on=key, how="left")
          .fillna(0))
    rt["fixed_pct"] = (rt["fixed_bal"] / rt["total_bal"] * 100).round(1)
    rt["float_pct"] = (rt["float_bal"] / rt["total_bal"] * 100).round(1)

    # Exact delta_EVE for par_up (standard stress scenario)
    exact_base_val = (exact_eve[exact_eve["scenario_id"] == "base"]
                      .rename(columns={"pv_total": "eve_base"})[key + ["eve_base"]])
    par_up_exact = (exact_eve[exact_eve["scenario_id"] == "par_up"]
                    .merge(exact_base_val, on=key, how="left"))
    par_up_exact["exact_dEVE_par_up"] = par_up_exact["pv_total"] - par_up_exact["eve_base"]
    par_up_exact = par_up_exact[key + ["exact_dEVE_par_up"]]

    result = rt.merge(par_up_exact, on=key, how="left")
    result["exact_dEVE_M"] = (result["exact_dEVE_par_up"] / 1e6).round(1)

    # Expected D_mod: for a mix product, effective D_mod ≈ fixed_pct×D_fixed + float_pct×D_float
    # Where D_float ≈ repricing_tenor. Floating products look like D_mod ≈ 0–1yr vs
    # fixed D_mod ≈ 5–10yr. Our approx uses full remaining maturity for both.
    result["note"] = result["float_pct"].apply(
        lambda x: "HIGH FLOAT — approx overestimates delta_EVE" if x > 40
        else ("MIX" if x > 10 else "MAINLY FIXED — approx should be ok"))
    return result.sort_values("exact_dEVE_M").reset_index(drop=True)


# -----------------------------------------------------------
# NII cohort share quality
# -----------------------------------------------------------

def _nii_share_quality() -> pd.DataFrame:
    """Check how well the NII interest share distributes product NII to cohorts.

    Loads cohort CF stats (nii_interest per cohort) and checks if shares sum to 1
    and whether the distribution is reasonable (not dominated by one cohort).
    """
    q = text(f"""
        SELECT
            s.product_code, s.bs_side, s.currency,
            YEAR(s.start_date)  AS start_year,
            MONTH(s.start_date) AS start_month,
            SUM(CAST(p.beh_outstanding AS FLOAT))           AS total_outstanding,
            SUM(CASE WHEN p.cf_end_dt <= DATEADD(day,365,:rd)
                     THEN CAST(p.beh_interest_pmt AS FLOAT)
                     ELSE 0.0 END)                          AS nii_interest
        FROM cf.products p
        JOIN (
            SELECT CAST(schedule_id AS VARCHAR(8)) AS schedule_id,
                   CAST(product_code AS VARCHAR(4)) AS product_code,
                   bs_side, currency, start_date
            FROM sched.loans   WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)),
                   CAST(product_code AS VARCHAR(4)), bs_side, currency, start_date
            FROM sched.deposits WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
            UNION ALL
            SELECT CAST(schedule_id AS VARCHAR(8)),
                   CAST(product_code AS VARCHAR(4)), bs_side, currency, start_date
            FROM sched.fin_inst WHERE CAST(product_code AS VARCHAR(4)) IN ({_C_IN})
        ) s ON CAST(p.schedule_id AS VARCHAR(8)) = s.schedule_id
        WHERE p.cf_end_dt > :rd
          AND COALESCE(p.beh_total_pmt, 0) <> 0
        GROUP BY s.product_code, s.bs_side, s.currency,
                 YEAR(s.start_date), MONTH(s.start_date)
    """)
    try:
        df = pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})
        df["product_code"] = df["product_code"].astype(str)
        key = ["product_code", "bs_side", "currency"]
        prod_totals = df.groupby(key)["nii_interest"].sum().reset_index()
        prod_totals = prod_totals.rename(columns={"nii_interest": "prod_nii_interest"})
        df = df.merge(prod_totals, on=key)
        df["nii_share"] = df["nii_interest"] / df["prod_nii_interest"].replace(0, np.nan)

        # Key insight columns
        df["cohort_label"] = (df["product_code"] + "_"
                              + df["start_year"].astype(str) + "_"
                              + df["start_month"].astype(str).str.zfill(2))
        df["share_pct"] = (df["nii_share"] * 100).round(2)
        return df[key + ["start_year", "start_month", "cohort_label",
                          "nii_interest", "share_pct", "total_outstanding"]].sort_values(
            key + ["start_year", "start_month"])
    except Exception as e:
        print(f"  [warn] nii_share_quality: {e}")
        return pd.DataFrame()


# -----------------------------------------------------------
# Real fast delta NII per product (schedule formula, not calibrated unit)
# -----------------------------------------------------------

def _fast_delta_nii_by_product(
    params: BalanceSheetParams,
    curves: CurveTensors,
) -> pd.DataFrame:
    """Per-product delta NII using the actual schedule-based fast formula.

    Cohort products: _cohort_delta_nii_sched (forward-rate model) — has real error.
    Single-row:      calibrated delta_nii_unit                    — exact by construction.

    Returns columns: product_code, bs_side, currency, scenario_id, approx_dNII
    """
    amounts   = params.balance_arr
    prod_key  = list(zip(params.product_code.tolist(),
                         params.bs_side.tolist(),
                         params.currency.tolist()))
    floor_arr = np.where(np.isfinite(params.client_floor), params.client_floor, -np.inf)
    cap_arr   = np.where(np.isfinite(params.client_cap),   params.client_cap,    np.inf)
    t1_int_arr = np.clip(np.round(params.cohort_t_first_m).astype(int), 1, 999)
    horizon_m  = 12

    scen_rows = []
    for s_idx, scen in enumerate(params.scenario_ids):
        scen = str(scen)
        delta_per_entry = np.zeros(len(amounts))

        # single-row / behavioural: calibrated
        mask_sng = ~params.is_cohort
        delta_per_entry[mask_sng] = (
            amounts[mask_sng] * params.delta_nii_unit[mask_sng, s_idx]
        )

        # cohort: schedule-based, grouped by (currency, repricing_freq, t_first)
        for ccy in params.unique_currencies:
            base_disc    = curves.get_disc_curve("base", ccy)
            shocked_disc = curves.get_disc_curve(scen,   ccy)
            mask_ccy = params.is_cohort & (params.currency == ccy)
            if not mask_ccy.any():
                continue
            for F in np.unique(params.repricing_tenor_m[mask_ccy]):
                mask_F  = mask_ccy & (params.repricing_tenor_m == F)
                gidx_F  = np.where(mask_F)[0]
                t1_F    = t1_int_arr[mask_F]
                for t1 in np.unique(t1_F):
                    if t1 >= horizon_m:
                        continue
                    gidx = gidx_F[t1_F == t1]
                    d = _cohort_delta_nii_sched(
                        amounts[gidx],
                        params.cohort_outstanding_m[gidx, :],
                        int(t1), float(F),
                        params.coeff_a[gidx],
                        params.coeff_b[gidx],
                        floor_arr[gidx],
                        cap_arr[gidx],
                        params.sign[gidx],
                        shocked_disc, base_disc,
                    )
                    delta_per_entry[gidx] = d

        # aggregate entries → product level
        agg: dict = {}
        for i, k in enumerate(prod_key):
            agg[k] = agg.get(k, 0.0) + delta_per_entry[i]

        for (pc, sid, ccy), val in agg.items():
            scen_rows.append({
                "product_code": pc, "bs_side": sid, "currency": ccy,
                "scenario_id": scen, "approx_dNII": val,
            })

    return pd.DataFrame(scen_rows)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def run_diagnosis() -> None:
    print("Loading exact NII from irrbb.nii_results...")
    exact_nii = _exact_nii()
    print(f"  {len(exact_nii)} rows  "
          f"({exact_nii['product_code'].nunique()} products, "
          f"{exact_nii['scenario_id'].nunique()} scenarios)")

    print("Loading exact EVE from irrbb.eve_results...")
    exact_eve = _exact_eve()
    print(f"  {len(exact_eve)} rows")

    print("Loading approx params from opt_prep SQL (for EVE delta)...")
    params    = _approx_params()
    scenarios = _approx_scenarios()
    print(f"  {len(params)} product entries, {len(scenarios)} scenario rows")

    print("Loading npz params + curves for real schedule-based delta NII...")
    params_npz = BalanceSheetParams.load(NPZ_PARAMS)
    curves_npz = CurveTensors.load(NPZ_CURVES)

    print("Loading rate_type breakdown from sched tables...")
    rate_type_df = _cohort_rate_type()

    base_agg, scen_agg = _build_approx_nii_eve(params, scenarios)

    # Replace calibrated approx_dNII with the real schedule-based fast formula
    fast_delta_nii = _fast_delta_nii_by_product(params_npz, curves_npz)
    scen_key = ["product_code", "bs_side", "currency", "scenario_id"]
    scen_agg = (scen_agg
                .drop(columns=["approx_dNII"])
                .merge(fast_delta_nii, on=scen_key, how="left"))

    nii_base_cmp, nii_delta_cmp = _build_nii_comparison(exact_nii, base_agg, scen_agg)
    eve_base_cmp, eve_delta_cmp = _build_eve_comparison(exact_eve, base_agg, scen_agg)

    root_cause = _root_cause_table(nii_base_cmp, eve_base_cmp,
                                   nii_delta_cmp, eve_delta_cmp, rate_type_df)

    eve_ff = _eve_fixed_float_breakdown(exact_eve, rate_type_df)

    print("Loading NII cohort share quality...")
    nii_shares = _nii_share_quality()

    # -- console summary -------------------------------------
    print("\n-- Top-5 products by absolute NII error (base) ---")
    top_nii = nii_base_cmp.dropna(subset=["nii_err_pct"]).nlargest(5, "nii_abs_err")
    for _, r in top_nii.iterrows():
        print(f"  {r['product_code']:4s}/{r['bs_side']}/{r['currency']}  "
              f"exact={r['exact_nii']/1e6:+.1f}M  approx={r['approx_nii']/1e6:+.1f}M  "
              f"err={r['nii_err_pct']:+.1f}%")

    print("\n-- Top-5 products by absolute EVE error (base) ---")
    top_eve = eve_base_cmp.dropna(subset=["eve_err_pct"]).reindex(
        eve_base_cmp["eve_abs_err"].abs().nlargest(5).index)
    for _, r in top_eve.iterrows():
        print(f"  {r['product_code']:4s}/{r['bs_side']}/{r['currency']}  "
              f"exact={r['exact_eve']/1e6:+.1f}M  approx={r['approx_eve']/1e6:+.1f}M  "
              f"err={r['eve_err_pct']:+.1f}%")

    print("\n-- Top-5 products by worst delta_EVE error --------")
    top_deve = (eve_delta_cmp.dropna(subset=["dEVE_err_pct"])
                .reindex(eve_delta_cmp["dEVE_abs_err"].abs().nlargest(5).index))
    for _, r in top_deve.iterrows():
        print(f"  {r['product_code']:4s}/{r['bs_side']}/{r['currency']} "
              f"scen={r['scenario_id']:8s}  "
              f"exact={r['exact_dEVE']/1e6:+.1f}M  approx={r['approx_dEVE']/1e6:+.1f}M  "
              f"err={r['dEVE_err_pct']:+.1f}%")

    print("\n-- Fixed vs floating composition of cohort products -----------------")
    if not eve_ff.empty:
        for _, r in eve_ff.iterrows():
            print(f"  {r['product_code']:4s}/{r['bs_side']}/{r['currency']}  "
                  f"fixed={r['fixed_pct']:.0f}%  float={r['float_pct']:.0f}%  "
                  f"exact_dEVE_parUp={r.get('exact_dEVE_M', 'n/a'):+.1f}M  {r['note']}")

    # -- Combine NII + EVE into single sheets per level --
    key      = ["product_code", "bs_side", "currency"]
    scen_key = key + ["scenario_id"]

    base_combined = nii_base_cmp.merge(
        eve_base_cmp[key + ["exact_eve", "eve_err_pct", "eve_abs_err"]],
        on=key, how="outer",
    )
    base_combined = base_combined[[
        "product_code", "bs_side", "currency", "balance_amt",
        "approx_nii", "exact_nii", "nii_err_pct", "nii_abs_err",
        "approx_eve", "exact_eve", "eve_err_pct", "eve_abs_err",
    ]]

    delta_combined = nii_delta_cmp.merge(
        eve_delta_cmp[scen_key + ["exact_dEVE", "dEVE_err_pct", "dEVE_abs_err"]],
        on=scen_key, how="outer",
    )
    delta_combined = delta_combined[[
        "product_code", "bs_side", "currency", "scenario_id",
        "approx_dNII", "exact_dNII", "dNII_err_pct", "dNII_abs_err",
        "approx_dEVE", "exact_dEVE", "dEVE_err_pct", "dEVE_abs_err",
    ]]

    # -- Excel output ----------------------------------------
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"\nWriting -> {OUT_PATH}")

    def _fmt_M(df, cols):
        d = df.copy()
        for c in cols:
            if c in d.columns:
                d[c] = (d[c] / 1e6).round(2)
        return d

    money_cols_base  = ["approx_nii", "exact_nii", "nii_abs_err",
                        "approx_eve", "exact_eve", "eve_abs_err"]
    money_cols_delta = ["approx_dNII", "exact_dNII", "dNII_abs_err",
                        "approx_dEVE", "exact_dEVE", "dEVE_abs_err"]

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        root_cause.to_excel(writer, sheet_name="root_cause_summary", index=False)

        _fmt_M(base_combined, money_cols_base).to_excel(
            writer, sheet_name="Base_by_product", index=False)
        _fmt_M(delta_combined, money_cols_delta).sort_values(
            ["scenario_id", "product_code"]).to_excel(
            writer, sheet_name="Delta_by_product", index=False)
        if not eve_ff.empty:
            eve_ff.to_excel(writer, sheet_name="EVE_fixed_vs_float", index=False)
        if not nii_shares.empty:
            nii_shares.to_excel(writer, sheet_name="NII_cohort_share",  index=False)
        if not rate_type_df.empty:
            rate_type_df.to_excel(writer, sheet_name="rate_type_balance", index=False)

    print(f"Done. Sheets: root_cause_summary | Base_by_product | Delta_by_product "
          f"| EVE_fixed_vs_float | NII_cohort_share | rate_type_balance")


if __name__ == "__main__":
    run_diagnosis()
