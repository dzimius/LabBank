"""accuracy_check.py
====================
Compare CF-based fast metric approximations (method B) against exact pipeline results.

Method B — CF-based: NII from rate_matrix[n,12,n_scen],
                     EVE from cohort_disc_q[n,max_q,n_scen]
           Works for any weight vector; no calibrated fallback except for single-row
           products without CF schedules (equity, non-rate-sensitive).

Output sheets:
  Base      — NII and EVE per product code, base scenario
  Scenarios — delta_NII and delta_EVE per product × scenario (raw + bias-corrected)
  LCR_NSFR  — LCR and NSFR ratios
  Summary   — portfolio-level errors vs SOT thresholds (err/T1 perspective)
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")
sys.path.insert(0, BASE_DIR)

from bs_vector import BalanceSheetParams, CurveTensors, CohortRates
from metrics import load_params_and_curves
from nii_fast import compute_nii_schedule_base_by_product
from lcr_fast import compute_lcr_fast
from nsfr_fast import compute_nsfr_fast
from nii_eve_cf_fast import (
    compute_nii_cf_all,
    compute_nii_cf_by_product,
    compute_eve_cf_fast_all,
    compute_eve_cf_fast_by_product,
)

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)

REPORT_DATE  = pd.to_datetime("2026-06-30")
TOTAL_ASSETS = 10_000_000_000
OUT_PATH     = os.path.join(BASE_DIR, "..", "output", "approx_accuracy_report.xlsx")
NPZ_PARAMS   = os.path.join(BASE_DIR, "..", "output", "product_params.npz")

# ── tolerances ─────────────────────────────────────────────────────────────────
# 1% error per product weight: if a product's weight changes by ε, the metric
# error should stay below TOL × metric.
TOL_NII  = 0.01   # 1 %
TOL_EVE  = 0.01   # 1 %
TOL_LCR  = 0.02
TOL_NSFR = 0.02


def _pct_err(approx: float, exact: float) -> float:
    if abs(exact) < 1.0:
        return float("nan")
    return (approx - exact) / abs(exact) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Exact results from SQL
# ─────────────────────────────────────────────────────────────────────────────

def _load_exact_nii_by_product() -> pd.DataFrame:
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


def _load_exact_eve_by_product() -> pd.DataFrame:
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


def _load_exact_lcr_nsfr() -> pd.DataFrame:
    q = text("""
        SELECT currency, lcr, nsfr
        FROM results.lcr_nsfr
        WHERE report_date = :rd
    """)
    return pd.read_sql_query(q, engine, params={"rd": REPORT_DATE})


def _load_tier1_capital() -> float:
    result = pd.read_sql_query(
        text("SELECT ISNULL(SUM(balance_amt), 0) AS tier1 FROM schemat.equity WHERE report_date = :rd"),
        engine, params={"rd": REPORT_DATE},
    )
    return float(result["tier1"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Approach B — CF-based  (rate_matrix NII + cohort_disc_q EVE)
# No calibrated fallback — single-row products use their cohort_outstanding_m
# and cohort_capital_m (zeros for non-rate-sensitive products, computed
# profiles for rate-sensitive ones like 8000 L and 0000 L).
# ─────────────────────────────────────────────────────────────────────────────

def _approach_b_base(
    params: BalanceSheetParams,
    cr: CohortRates,
    amounts: np.ndarray,
) -> pd.DataFrame:
    """Base NII (rate_matrix) and EVE (cohort_disc_q) per product.

    NII_B base: CF-based (rate_matrix × outstanding_m); calibrated fallback only
    for products where CF gives near-zero but calibrated is non-zero (e.g. admin-
    priced products like 6000 L with outstanding_m=zeros). This fallback does NOT
    affect delta NII — it only fixes the base NII level for display.

    delta NII_B: purely CF-based — no calibrated fallback — see _approach_b_scen.

    EVE_B: CF-based from cohort_disc_q; calibrated fallback for single-row
    products that have no CF schedule to avoid spurious EVE basis errors.
    """
    base_idx = cr.scenario_index("base")

    nii_cf = compute_nii_cf_by_product(amounts, params, cr)
    eve_cf = compute_eve_cf_fast_by_product(amounts, params, cr)

    # Calibrated base values for single-row products (fallback only)
    sngl_nii: dict = {}
    sngl_eve: dict = {}
    for i, (pc, sid, ccy) in enumerate(
        zip(params.product_code, params.bs_side, params.currency)
    ):
        if not params.is_cohort[i]:
            k = (str(pc), str(sid), str(ccy))
            sngl_nii[k] = sngl_nii.get(k, 0.0) + amounts[i] * params.nii_unit_rate[i]
            sngl_eve[k] = sngl_eve.get(k, 0.0) + amounts[i] * params.eve_pv_factor[i]

    rows = []
    for k in set(nii_cf) | set(eve_cf) | set(sngl_nii) | set(sngl_eve):
        pc, sid, ccy = k
        nii_b = float(nii_cf[k][base_idx]) if k in nii_cf else 0.0
        eve_b = float(eve_cf[k][0])         if k in eve_cf else 0.0
        # Fallback for base NII and EVE when CF gives near-zero and calibrated is non-zero
        if abs(nii_b) < 1.0 and abs(sngl_nii.get(k, 0.0)) > 1.0:
            nii_b = sngl_nii[k]
        if abs(eve_b) < 1.0 and abs(sngl_eve.get(k, 0.0)) > 1.0:
            eve_b = sngl_eve[k]
        rows.append({
            "product_code": pc, "bs_side": sid, "currency": ccy,
            "NII_B": nii_b, "EVE_B": eve_b,
        })
    return pd.DataFrame(rows)


def _approach_b_scen(
    params: BalanceSheetParams,
    cr: CohortRates,
    amounts: np.ndarray,
) -> pd.DataFrame:
    """delta NII (rate_matrix) and delta EVE (cohort_disc_q) per product × scenario.

    delta NII_B: pure CF-based — no calibrated fallback.
    delta EVE_B: CF-based; calibrated fallback for single-row products without
    CF schedules so EVE errors reflect real duration gaps, not missing data.
    """
    base_idx = cr.scenario_index("base")

    nii_cf = compute_nii_cf_by_product(amounts, params, cr)
    eve_cf = compute_eve_cf_fast_by_product(amounts, params, cr)

    # EVE calibrated fallback (single-row products without CF schedules)
    sngl_scen_eve: dict = {}
    for s_idx, scen in enumerate(params.scenario_ids):
        scen = str(scen)
        for i, (pc, sid, ccy) in enumerate(
            zip(params.product_code, params.bs_side, params.currency)
        ):
            if not params.is_cohort[i]:
                k4 = (str(pc), str(sid), str(ccy), scen)
                sngl_scen_eve[k4] = sngl_scen_eve.get(k4, 0.0) + (
                    amounts[i] * params.delta_eve_unit[i, s_idx]
                )

    rows = []
    for rs, scen_label in enumerate(cr.rate_scenario_ids):
        scen_label = str(scen_label)
        if scen_label == "base":
            continue
        all_prod_keys = set(nii_cf) | set(eve_cf) | {
            k[:3] for k in sngl_scen_eve if k[3] == scen_label
        }
        for k in all_prod_keys:
            pc, sid, ccy = k
            dnii_b = float(nii_cf[k][rs]) - float(nii_cf[k][base_idx]) if k in nii_cf else 0.0
            deve_b = float(eve_cf[k][rs])                                if k in eve_cf else 0.0
            sngl_deve = sngl_scen_eve.get((pc, sid, ccy, scen_label), 0.0)
            if abs(deve_b) < 1.0 and abs(sngl_deve) > 1.0:
                deve_b = sngl_deve

            rows.append({
                "product_code": pc, "bs_side": sid, "currency": ccy,
                "scenario_id": scen_label,
                "dNII_B": dnii_b, "dEVE_B": deve_b,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_accuracy_check() -> None:
    print("Loading fast metric parameters...")
    params, curves = load_params_and_curves()
    cr = CohortRates.load(NPZ_PARAMS)

    amounts = params.balance_arr

    print("Loading exact results from SQL...")
    exact_nii_df = _load_exact_nii_by_product()
    exact_eve_df = _load_exact_eve_by_product()
    exact_liq    = _load_exact_lcr_nsfr()
    tier1        = _load_tier1_capital()
    print(f"Tier 1 capital: {tier1:,.0f}  (SOT NII threshold: 5% = {tier1*0.05:,.0f}  EVE: 15% = {tier1*0.15:,.0f})")

    print("Computing method B (CF-based, rate_matrix + cohort_disc_q)...")
    base_B = _approach_b_base(params, cr, amounts)
    scen_B = _approach_b_scen(params, cr, amounts)

    print("Computing schedule-based base NII (ca*fwd+cb formula)...")
    sched_base_map = compute_nii_schedule_base_by_product(amounts, params, curves)
    sched_base_df  = pd.DataFrame([
        {"product_code": k[0], "bs_side": k[1], "currency": k[2], "NII_sched": v}
        for k, v in sched_base_map.items()
    ])

    _KEY      = ["product_code", "bs_side", "currency"]
    _SCEN_KEY = _KEY + ["scenario_id"]

    # ── Sheet 1: Base ─────────────────────────────────────────────────────────
    exact_nii_base = (exact_nii_df[exact_nii_df["scenario_id"] == "base"]
                      .rename(columns={"nii_total": "NII_exact"})[_KEY + ["NII_exact"]])
    exact_eve_base = (exact_eve_df[exact_eve_df["scenario_id"] == "base"]
                      .rename(columns={"pv_total": "EVE_exact"})[_KEY + ["EVE_exact"]])

    base_sheet = (base_B
                  .merge(sched_base_df,   on=_KEY, how="outer")
                  .merge(exact_nii_base,  on=_KEY, how="outer")
                  .merge(exact_eve_base,  on=_KEY, how="outer"))
    base_sheet = base_sheet.fillna(0.0)

    base_sheet["NII_err_B"]     = [_pct_err(f, e)
        for f, e in zip(base_sheet["NII_B"],     base_sheet["NII_exact"])]
    base_sheet["NII_err_sched"] = [_pct_err(f, e)
        for f, e in zip(base_sheet["NII_sched"], base_sheet["NII_exact"])]
    base_sheet["EVE_err_B"]     = [_pct_err(f, e)
        for f, e in zip(base_sheet["EVE_B"],     base_sheet["EVE_exact"])]

    # Bias = exact_base − fast_base (PLN); used to first-order-correct shocked deltas.
    base_sheet["NII_bias"]  = base_sheet["NII_exact"] - base_sheet["NII_sched"]
    base_sheet["EVE_bias"]  = base_sheet["EVE_exact"] - base_sheet["EVE_B"]

    for col in ["balance", "NII_exact", "NII_B", "NII_sched",
                "NII_bias", "EVE_exact", "EVE_B", "EVE_bias"]:
        if col in base_sheet.columns:
            base_sheet[col] = base_sheet[col].round(0)
    for col in ["NII_err_B", "NII_err_sched", "EVE_err_B"]:
        base_sheet[col] = base_sheet[col].round(2)

    base_sheet = base_sheet[
        _KEY + ["NII_exact", "NII_B",     "NII_err_B",
                "NII_sched", "NII_err_sched", "NII_bias",
                "EVE_exact", "EVE_B",     "EVE_err_B", "EVE_bias"]
    ].sort_values(_KEY).reset_index(drop=True)

    # ── Sheet 2: Scenarios ────────────────────────────────────────────────────
    exact_nii_scen = (exact_nii_df[exact_nii_df["scenario_id"] != "base"]
                      .rename(columns={"nii_total": "NII_shocked_exact"}))
    exact_eve_scen = (exact_eve_df[exact_eve_df["scenario_id"] != "base"]
                      .rename(columns={"pv_total": "EVE_shocked_exact"}))

    exact_nii_base_map = (exact_nii_df[exact_nii_df["scenario_id"] == "base"]
                          .set_index(_KEY)["nii_total"].to_dict())
    exact_eve_base_map = (exact_eve_df[exact_eve_df["scenario_id"] == "base"]
                          .set_index(_KEY)["pv_total"].to_dict())

    exact_nii_scen = exact_nii_scen.copy()
    exact_eve_scen = exact_eve_scen.copy()

    exact_nii_scen["dNII_exact"] = exact_nii_scen.apply(
        lambda r: r["NII_shocked_exact"] - exact_nii_base_map.get(
            (r["product_code"], r["bs_side"], r["currency"]), 0.0), axis=1)
    exact_eve_scen["dEVE_exact"] = exact_eve_scen.apply(
        lambda r: r["EVE_shocked_exact"] - exact_eve_base_map.get(
            (r["product_code"], r["bs_side"], r["currency"]), 0.0), axis=1)

    exact_nii_delta = exact_nii_scen[_SCEN_KEY + ["dNII_exact"]]
    exact_eve_delta = exact_eve_scen[_SCEN_KEY + ["dEVE_exact"]]

    scen_sheet = (scen_B
                  .merge(exact_nii_delta, on=_SCEN_KEY, how="outer")
                  .merge(exact_eve_delta, on=_SCEN_KEY, how="outer"))
    scen_sheet = scen_sheet.fillna(0.0)

    nii_bias_map = base_sheet.set_index(_KEY)["NII_bias"].to_dict()
    eve_bias_map = base_sheet.set_index(_KEY)["EVE_bias"].to_dict()

    def _lookup_bias(row, bias_map):
        return bias_map.get(
            (row["product_code"], row["bs_side"], row["currency"]), 0.0
        )

    scen_sheet["dNII_B_biascorr"] = (
        scen_sheet["dNII_B"]
        + scen_sheet.apply(lambda r: _lookup_bias(r, nii_bias_map), axis=1)
    )
    scen_sheet["dEVE_B_biascorr"] = (
        scen_sheet["dEVE_B"]
        + scen_sheet.apply(lambda r: _lookup_bias(r, eve_bias_map), axis=1)
    )

    scen_sheet["dNII_err_B"]          = [_pct_err(f, e)
        for f, e in zip(scen_sheet["dNII_B"],          scen_sheet["dNII_exact"])]
    scen_sheet["dNII_err_B_biascorr"] = [_pct_err(f, e)
        for f, e in zip(scen_sheet["dNII_B_biascorr"], scen_sheet["dNII_exact"])]
    scen_sheet["dEVE_err_B"]          = [_pct_err(f, e)
        for f, e in zip(scen_sheet["dEVE_B"],          scen_sheet["dEVE_exact"])]
    scen_sheet["dEVE_err_B_biascorr"] = [_pct_err(f, e)
        for f, e in zip(scen_sheet["dEVE_B_biascorr"], scen_sheet["dEVE_exact"])]

    for col in ["dNII_exact", "dNII_B", "dNII_B_biascorr",
                "dEVE_exact", "dEVE_B", "dEVE_B_biascorr"]:
        if col in scen_sheet.columns:
            scen_sheet[col] = scen_sheet[col].round(0)
    for col in ["dNII_err_B", "dNII_err_B_biascorr", "dEVE_err_B", "dEVE_err_B_biascorr"]:
        scen_sheet[col] = scen_sheet[col].round(2)

    scen_sheet = scen_sheet[
        _SCEN_KEY + [
            "dNII_exact",
            "dNII_B",          "dNII_err_B",
            "dNII_B_biascorr", "dNII_err_B_biascorr",
            "dEVE_exact",
            "dEVE_B",          "dEVE_err_B",
            "dEVE_B_biascorr", "dEVE_err_B_biascorr",
        ]
    ].sort_values(_SCEN_KEY).reset_index(drop=True)

    # ── Save per-unit bias corrections for optimizer / sandbox ────────────────
    try:
        from bias_store import save_bias_corrections

        prod_bal: dict[tuple[str, str, str], float] = {}
        for i, (pc, sid, ccy) in enumerate(zip(
            params.product_code, params.bs_side, params.currency
        )):
            k = (str(pc), str(sid), str(ccy))
            prod_bal[k] = prod_bal.get(k, 0.0) + float(amounts[i])

        # per-product per-scenario gap: (exact − fast) in PLN
        gap_eve: dict[tuple[str, str, str, str], float] = {}
        gap_nii: dict[tuple[str, str, str, str], float] = {}
        for _, row in scen_sheet.iterrows():
            k = (str(row["product_code"]), str(row["bs_side"]),
                 str(row["currency"]),     str(row["scenario_id"]))
            gap_eve[k] = float(row["dEVE_exact"]) - float(row["dEVE_B"])
            gap_nii[k] = float(row["dNII_exact"]) - float(row["dNII_B"])

        shocked_scens = [str(s) for s in params.scenario_ids]
        n   = len(params.balance_arr)
        n_s = len(shocked_scens)
        bias_eve_u = np.zeros((n, n_s), dtype=float)
        bias_nii_u = np.zeros((n, n_s), dtype=float)
        for i, (pc, sid, ccy) in enumerate(zip(
            params.product_code, params.bs_side, params.currency
        )):
            k3 = (str(pc), str(sid), str(ccy))
            pb = prod_bal.get(k3, 0.0)
            if pb <= 0.0:
                continue
            for s_idx, scen in enumerate(shocked_scens):
                bias_eve_u[i, s_idx] = gap_eve.get((str(pc), str(sid), str(ccy), scen), 0.0) / pb
                bias_nii_u[i, s_idx] = gap_nii.get((str(pc), str(sid), str(ccy), scen), 0.0) / pb

        save_bias_corrections(bias_eve_u, bias_nii_u, shocked_scens)
    except Exception as exc:
        print(f"  WARNING: bias corrections not saved — {exc}")

    # ── CF schedule diagnostic: per-cohort yf range and disc change ──────────
    _diag_pc = {"1000", "2000", "2100", "3000", "3100", "7060"}
    print("\n[diag] CF schedule per cohort entry (products 1000/2000/2100/3000/3100/7060):")
    print(f"  {'pc':6s} {'side':4s} {'sy':5s} {'sm':3s} {'bal':>14s} {'nq':>4s} "
          f"{'yf_first':>9s} {'yf_last':>9s} {'cf_frac_sum':>12s} "
          f"{'disc_base0':>10s} {'disc_pup0':>10s} "
          f"{'disc_basN':>10s} {'disc_pupN':>10s}")
    base_idx_cr = cr.scenario_index("base")
    par_up_idx_cr = cr.scenario_index("par_up")
    for i, (pc, sid, ccy, sy, sm) in enumerate(zip(
        params.product_code, params.bs_side, params.currency,
        params.start_year, params.start_month
    )):
        if str(pc) not in _diag_pc:
            continue
        nq = int(cr.cf_n_q[i])
        if nq == 0:
            continue
        bal = float(amounts[i])
        cf_tot_sum  = float(cr.cf_fixed_int_frac[i, :nq].sum())
        cf_yf_first = float(cr.cf_yf[i, 0])
        cf_yf_last  = float(cr.cf_yf[i, nq - 1])
        db0  = float(cr.cohort_disc_q[i, 0,      base_idx_cr])
        dp0  = float(cr.cohort_disc_q[i, 0,      par_up_idx_cr])
        dbN  = float(cr.cohort_disc_q[i, nq - 1, base_idx_cr])
        dpN  = float(cr.cohort_disc_q[i, nq - 1, par_up_idx_cr])
        print(f"  {pc:6s} {sid:4s} {int(sy):5d} {int(sm):3d} {bal:>14,.0f} {nq:>4d} "
              f"{cf_yf_first:>9.4f} {cf_yf_last:>9.4f} {cf_tot_sum:>12.4f} "
              f"{db0:>10.6f} {dp0:>10.6f} "
              f"{dbN:>10.6f} {dpN:>10.6f}")

    # ── Per-product delta diagnostic for par_up and sr_up ────────────────────
    for _diag_scen in ["par_up", "sr_up"]:
        _ss = scen_sheet[scen_sheet["scenario_id"] == _diag_scen].copy()
        if _ss.empty:
            continue
        _ss["EVE_gap"] = _ss["dEVE_B"] - _ss["dEVE_exact"]
        _ss["NII_gap"] = _ss["dNII_B"] - _ss["dNII_exact"]
        print(f"\n[diag] Per-product delta decomposition — scenario: {_diag_scen}")
        print(f"  {'product':8s} {'side':4s}  {'dEVE_exact':>14s}  {'dEVE_B':>14s}  {'EVE_gap':>14s}  "
              f"{'dNII_exact':>14s}  {'dNII_B':>14s}  {'NII_gap':>14s}")
        for _, r in _ss.sort_values("EVE_gap").iterrows():
            print(f"  {r.product_code:8s} {r.bs_side:4s}  "
                  f"{r.dEVE_exact:>14,.0f}  {r.dEVE_B:>14,.0f}  {r['EVE_gap']:>14,.0f}  "
                  f"{r.dNII_exact:>14,.0f}  {r.dNII_B:>14,.0f}  {r['NII_gap']:>14,.0f}")
        print(f"  {'TOTAL':12s}  "
              f"{_ss.dEVE_exact.sum():>14,.0f}  {_ss.dEVE_B.sum():>14,.0f}  {_ss.EVE_gap.sum():>14,.0f}  "
              f"{_ss.dNII_exact.sum():>14,.0f}  {_ss.dNII_B.sum():>14,.0f}  {_ss.NII_gap.sum():>14,.0f}")

    # ── Sheet 3: LCR / NSFR ──────────────────────────────────────────────────
    _lcr_fast  = compute_lcr_fast(amounts, params)
    _nsfr_fast = compute_nsfr_fast(amounts, params)
    liq_rows = []
    if not exact_liq.empty:
        for _, r in exact_liq.iterrows():
            ccy        = r["currency"]
            exact_lcr  = float(r["lcr"])  if pd.notna(r["lcr"])  else float("nan")
            exact_nsfr = float(r["nsfr"]) if pd.notna(r["nsfr"]) else float("nan")
            approx_lcr  = _lcr_fast.get(ccy,  float("nan"))
            approx_nsfr = _nsfr_fast.get(ccy, float("nan"))
            liq_rows.append({
                "currency":     ccy,
                "LCR_exact":    round(exact_lcr,  4),
                "LCR_fast":     round(approx_lcr,  4),
                "LCR_err_pct":  round(_pct_err(approx_lcr,  exact_lcr),  2),
                "NSFR_exact":   round(exact_nsfr, 4),
                "NSFR_fast":    round(approx_nsfr, 4),
                "NSFR_err_pct": round(_pct_err(approx_nsfr, exact_nsfr), 2),
            })
    liq_sheet = pd.DataFrame(liq_rows)

    # ── Sheet 4: Summary ──────────────────────────────────────────────────────
    summary_rows = []

    def _chk(label, scen, ccy, approx_b, exact, tol, tier1=None, sot_thresh=None):
        err_b  = _pct_err(approx_b, exact)
        pass_b = abs(err_b) < tol * 100 if not np.isnan(err_b) else False

        if tier1 and tier1 > 0:
            err_t1_b      = (approx_b - exact) / tier1 * 100.0
            sot_exact_t1  = exact    / tier1 * 100.0
            sot_b_t1      = approx_b / tier1 * 100.0
        else:
            err_t1_b = sot_exact_t1 = sot_b_t1 = float("nan")

        summary_rows.append({
            "metric":          label,
            "scenario":        scen,
            "currency":        ccy,
            "exact_PLN":       round(exact, 0)    if abs(exact) >= 1    else exact,
            "fast_B_PLN":      round(approx_b, 0) if abs(approx_b) >= 1 else approx_b,
            "SOT_exact_%T1":   round(sot_exact_t1, 2) if not np.isnan(sot_exact_t1) else float("nan"),
            "SOT_B_%T1":       round(sot_b_t1,     2) if not np.isnan(sot_b_t1)     else float("nan"),
            "err_B_%":         round(err_b, 2),
            "err_B_pct_T1":    round(err_t1_b, 3) if not np.isnan(err_t1_b) else float("nan"),
            "pass_B":          "PASS" if pass_b else "FAIL",
            "tolerance_%":     tol * 100,
            "sot_thresh_%T1":  sot_thresh if sot_thresh is not None else float("nan"),
        })

        sot_str    = (f"  SOT: exact={sot_exact_t1:+.2f}%T1  B={sot_b_t1:+.2f}%T1"
                      f"  err={err_t1_b:+.2f}%T1") if not np.isnan(sot_exact_t1) else ""
        thresh_str = f"  [thresh={sot_thresh:+.0f}%T1]" if sot_thresh is not None else ""
        print(f"  {label:<20s} {scen:<12s} {ccy:<5s}"
              f"{sot_str}  {'OK' if pass_b else '--'}{thresh_str}")
        return pass_b

    # Portfolio totals
    exact_nii_tot = {
        s: float(exact_nii_df[exact_nii_df["scenario_id"] == s]["nii_total"].sum())
        for s in exact_nii_df["scenario_id"].unique()
    }
    exact_eve_tot = {
        s: float(exact_eve_df[exact_eve_df["scenario_id"] == s]["pv_total"].sum())
        for s in exact_eve_df["scenario_id"].unique()
    }

    nii_base_exact = exact_nii_tot.get("base", float("nan"))
    eve_base_exact = exact_eve_tot.get("base", float("nan"))

    # B approach totals: sum per-product DataFrames (includes single-row calibrated fallback)
    nii_base_B = float(base_B["NII_B"].sum())
    eve_base_B = float(base_B["EVE_B"].sum())

    scen_B_tot = scen_B.groupby("scenario_id")[["dNII_B", "dEVE_B"]].sum()

    # ── EVE B per-product diagnostic ──────────────────────────────────────────
    print("\n[diag] EVE breakdown by product (|EVE_exact|>1M or |EVE_B|>1M):")
    _sign_map: dict = {}
    _cfnq_map: dict = {}
    _bal_map:  dict = {}
    for i, (pc, sid, ccy) in enumerate(
        zip(params.product_code, params.bs_side, params.currency)
    ):
        k = (str(pc), str(sid), str(ccy))
        _sign_map[k] = float(params.sign[i])
        _cfnq_map[k] = _cfnq_map.get(k, 0) + int(cr.cf_n_q[i])
        _bal_map[k]  = _bal_map.get(k, 0.0) + float(amounts[i])
    _evd = base_sheet[
        (base_sheet["EVE_exact"].abs() > 1e6) | (base_sheet["EVE_B"].abs() > 1e6)
    ].copy()
    _evd["sign"]    = _evd.apply(lambda r: _sign_map.get((r.product_code, r.bs_side, r.currency), 0), axis=1)
    _evd["cf_n_q"]  = _evd.apply(lambda r: _cfnq_map.get((r.product_code, r.bs_side, r.currency), 0), axis=1)
    _evd["balance"] = _evd.apply(lambda r: _bal_map.get((r.product_code, r.bs_side, r.currency), 0.0), axis=1)
    _evd["EVE_gap"] = _evd["EVE_B"] - _evd["EVE_exact"]
    print(_evd[["product_code", "bs_side", "balance", "EVE_exact", "EVE_B",
                "sign", "cf_n_q", "EVE_gap"]].sort_values("EVE_gap").to_string(index=False))
    print(f"  [diag] EVE_B total from table: {_evd['EVE_B'].sum():+,.0f}  EVE_exact total: {_evd['EVE_exact'].sum():+,.0f}")
    print(f"  [diag] Reported EVE_B: {eve_base_B:+,.0f}  Residual not in table: {eve_base_B - _evd['EVE_B'].sum():+,.0f}")

    # ── Gradient validation: optimizer signal quality ─────────────────────────
    # For NII maximization the optimizer follows ∂NII/∂weight_i per product.
    # With proportional scaling this equals NII_i / balance_i.
    # Two failure modes: wrong sign (optimizer goes the wrong direction) and
    # wrong relative rank (optimizer prefers the wrong product at the margin).
    grad_rows = []
    for _, row in base_sheet.iterrows():
        k   = (row["product_code"], row["bs_side"], row["currency"])
        bal = _bal_map.get(k, 0.0)
        if bal == 0.0:
            continue
        nii_exact_rate = float(row["NII_exact"]) / bal
        nii_b_rate     = float(row["NII_B"])     / bal
        eve_exact_rate = float(row["EVE_exact"]) / bal
        eve_b_rate     = float(row["EVE_B"])     / bal
        nii_sign_ok    = int(np.sign(nii_b_rate)) == int(np.sign(nii_exact_rate))
        eve_sign_ok    = int(np.sign(eve_b_rate)) == int(np.sign(eve_exact_rate))
        grad_rows.append({
            "product_code":   row["product_code"],
            "bs_side":        row["bs_side"],
            "currency":       row["currency"],
            "balance":        round(bal, 0),
            "NII_exact_rate": round(nii_exact_rate * 100, 4),   # % per PLN balance
            "NII_B_rate":     round(nii_b_rate     * 100, 4),
            "NII_rate_err_%": round(_pct_err(nii_b_rate, nii_exact_rate), 2),
            "NII_sign_ok":    nii_sign_ok,
            "EVE_exact_rate": round(eve_exact_rate, 6),
            "EVE_B_rate":     round(eve_b_rate,     6),
            "EVE_rate_err_%": round(_pct_err(eve_b_rate, eve_exact_rate), 2),
            "EVE_sign_ok":    eve_sign_ok,
        })
    grad_sheet = pd.DataFrame(grad_rows).sort_values(["bs_side", "NII_exact_rate"],
                                                      ascending=[True, False]).reset_index(drop=True)

    # Spearman rank correlation and sign mismatch summary
    valid_nii = grad_sheet.dropna(subset=["NII_exact_rate", "NII_B_rate"])
    valid_nii = valid_nii[valid_nii["NII_exact_rate"].abs() > 0]
    valid_eve = grad_sheet.dropna(subset=["EVE_exact_rate", "EVE_B_rate"])
    valid_eve = valid_eve[valid_eve["EVE_exact_rate"].abs() > 0]

    def _spearman(a: "pd.Series", b: "pd.Series") -> float:
        rk_a = a.rank()
        rk_b = b.rank()
        n    = len(a)
        if n < 2:
            return float("nan")
        d2   = ((rk_a - rk_b) ** 2).sum()
        return 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))

    nii_rho         = _spearman(valid_nii["NII_exact_rate"], valid_nii["NII_B_rate"])
    eve_rho         = _spearman(valid_eve["EVE_exact_rate"], valid_eve["EVE_B_rate"])
    nii_sign_errors = int((~valid_nii["NII_sign_ok"]).sum())
    eve_sign_errors = int((~valid_eve["EVE_sign_ok"]).sum())

    print(f"\n--- Gradient validation (optimizer signal quality) ---")
    print(f"  NII sensitivity  Spearman rho={nii_rho:+.4f}  sign errors={nii_sign_errors}/{len(valid_nii)}")
    print(f"  EVE sensitivity  Spearman rho={eve_rho:+.4f}  sign errors={eve_sign_errors}/{len(valid_eve)}")
    if nii_sign_errors:
        bad = valid_nii[~valid_nii["NII_sign_ok"]]
        print("  WARNING: NII gradient sign wrong for:")
        for _, r in bad.iterrows():
            print(f"    {r.product_code:8s} {r.bs_side}  "
                  f"exact_rate={r.NII_exact_rate:+.4f}%  B_rate={r.NII_B_rate:+.4f}%  "
                  f"balance={r.balance:,.0f}")
    else:
        print("  NII gradient sign: all correct — optimizer direction is trustworthy")
    if eve_sign_errors:
        bad = valid_eve[~valid_eve["EVE_sign_ok"]]
        print("  WARNING: EVE gradient sign wrong for:")
        for _, r in bad.iterrows():
            print(f"    {r.product_code:8s} {r.bs_side}  "
                  f"exact_rate={r.EVE_exact_rate:+.6f}  B_rate={r.EVE_B_rate:+.6f}  "
                  f"balance={r.balance:,.0f}")

    _chk("NII_base", "base", "ALL", nii_base_B, nii_base_exact, TOL_NII)
    _chk("EVE_base", "base", "ALL", eve_base_B, eve_base_exact, TOL_EVE)

    for scen in [s for s in params.scenario_ids if str(s) != "base"]:
        scen = str(scen)
        exact_dnii = exact_nii_tot.get(scen, float("nan")) - nii_base_exact
        exact_deve = exact_eve_tot.get(scen, float("nan")) - eve_base_exact
        dnii_b = float(scen_B_tot.loc[scen, "dNII_B"]) if scen in scen_B_tot.index else float("nan")
        deve_b = float(scen_B_tot.loc[scen, "dEVE_B"]) if scen in scen_B_tot.index else float("nan")
        _chk("delta_NII", scen, "ALL", dnii_b, exact_dnii, TOL_NII, tier1=tier1, sot_thresh=-5.0)
        _chk("delta_EVE", scen, "ALL", deve_b, exact_deve, TOL_EVE, tier1=tier1, sot_thresh=-15.0)

    if not exact_liq.empty:
        for _, r in exact_liq.iterrows():
            ccy        = r["currency"]
            exact_lcr  = float(r["lcr"])
            exact_nsfr = float(r["nsfr"])
            _chk("LCR",  "n/a", ccy, _lcr_fast.get(ccy, float("nan")),  exact_lcr,  TOL_LCR)
            _chk("NSFR", "n/a", ccy, _nsfr_fast.get(ccy, float("nan")), exact_nsfr, TOL_NSFR)

    summary_df = pd.DataFrame(summary_rows)
    n_pass_b = (summary_df["pass_B"] == "PASS").sum()
    n_total  = len(summary_df)
    print(f"\nAccuracy summary: B={n_pass_b}/{n_total} PASS")

    # ── Write Excel ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        base_sheet.to_excel(writer, sheet_name="Base",      index=False)
        scen_sheet.to_excel(writer, sheet_name="Scenarios", index=False)
        if not liq_sheet.empty:
            liq_sheet.to_excel(writer, sheet_name="LCR_NSFR", index=False)
        summary_df.to_excel(writer, sheet_name="Summary",   index=False)
        grad_sheet.to_excel(writer, sheet_name="Gradient",  index=False)

    print(f"Report written to {OUT_PATH}")
    print("  Sheets: Base | Scenarios | LCR_NSFR | Summary | Gradient")
    print("  Gradient sheet — optimizer signal quality per product:")
    print("    NII_exact_rate / NII_B_rate = NII per 1 PLN balance (% pa) — exact vs fast")
    print("    NII_sign_ok    = True if optimizer direction is correct for this product")
    print("    Spearman rho   = rank correlation — how well B ranks NII-efficiency of products")
    print("  Summary columns:")
    print("    err_B_pct_T1  = (fast_B - exact) / Tier1 * 100  [SOT perspective]")
    print("    sot_thresh_%  = EBA threshold as % of T1 (-5 NII, -15 EVE)")


if __name__ == "__main__":
    run_accuracy_check()
