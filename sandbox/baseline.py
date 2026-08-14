"""baseline.py
=============
Data loaders and weight-computation helpers for the sandbox app.

All loaders are cached so they run only once per Streamlit session.
The optimize_prep fast-metric engine is imported here so the rest of the
app never needs to touch sys.path directly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPTPREP_CODE  = os.path.join(PROJECT_ROOT, "optimize_prep", "python_code")
BSOPT_CODE    = os.path.join(PROJECT_ROOT, "bs_optimization", "python_code")

for _p in (OPTPREP_CODE, BSOPT_CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bs_vector import BalanceSheetParams, CurveTensors, CohortRates  # noqa: E402
from metrics   import compute_all_metrics, AllMetrics, reset_bias_cache  # noqa: E402
from ep_fast   import build_ep_context, compute_ep_components  # noqa: E402

# ── file paths ─────────────────────────────────────────────────────────────────
PARAMS_PATH         = os.path.join(PROJECT_ROOT, "optimize_prep", "output", "product_params.npz")
CURVES_PATH         = os.path.join(PROJECT_ROOT, "optimize_prep", "output", "curve_tensors.npz")
SCENARIO_CURVES_PATH = os.path.join(PROJECT_ROOT, "sandbox", "scenario_curves.npz")
BS_PATH     = os.path.join(PROJECT_ROOT, "balance_generate", "input_data", "bank_data.xlsx")
IRS_PATH    = os.path.join(PROJECT_ROOT, "ir_derivatives",   "input",  "irs_input.xlsx")

TOLERANCE   = 0.05   # % tolerance for balance-sheet sum validation


# ── cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_params() -> BalanceSheetParams:
    # TECH DEBT: sandbox shares BalanceSheetParams (and the optimizer's npz) with
    # bs_optimizer.py, so it pulls in optimizer-only fields (vol_elasticity,
    # subst_matrix, PD/LGD, CoC) that are irrelevant here.
    # Future fix: give the sandbox its own lightweight ETL reading directly from
    # bank_data.xlsx (bs_structure sheet only) without any optimizer columns.
    return BalanceSheetParams.load(PARAMS_PATH)


@st.cache_resource
def load_curves() -> CurveTensors:
    return CurveTensors.load(CURVES_PATH)


@st.cache_resource
def load_cohort_rates() -> CohortRates:
    return CohortRates.load(PARAMS_PATH)


@st.cache_resource
def load_ep_context() -> tuple:
    """Product<->cohort map + margin-over-FTP rate per cohort, needed for the
    Economic Profit waterfall (Metrics tab). Ironically, these are exactly the
    "optimizer-only fields... irrelevant here" load_params() warns about above
    (vol_elasticity, acq_cost_rate, coc_rate, cet1_target, fee_unit_rate) --
    the EP feature is what finally puts them to use."""
    return build_ep_context(load_params())


def compute_ep(amounts: np.ndarray, cr: CohortRates, mask_irs: bool = False) -> dict:
    """EP decomposition at per-cohort `amounts` (PLN) -- thin wrapper around
    ep_fast.compute_ep_components binding params/pm/margin_rate from the
    cached loaders so callers only pass what actually varies (the weights)."""
    params = load_params()
    pm, margin_rate = load_ep_context()
    return compute_ep_components(amounts, pm, params, cr, margin_rate, mask_irs=mask_irs)


@st.cache_data
def load_scenario_curves() -> dict:
    """Load stylised scenario curves and pre-computed IRRBB unit metrics.

    Base keys (always present):
      fwd_rates    (15, 360)  annualised forward rates
      disc_factors (15, 360)  discount factors
      scenario_ids (15,)      e.g. 'normal_low', 'steep_medium', ...
      curve_types  (5,)
      levels       (3,)
      n_months     int

    IRRBB unit metrics (present after running build_scenario_curves.py):
      hyp_nii_unit_rate  (15, n)     NII per PLN balance at hyp base
      hyp_delta_nii_unit (15, n, S)  floor-adjusted delta NII per PLN per shock
      hyp_eve_pv_factor  (15, n)     EVE PV factor at hyp base
      hyp_delta_eve_unit (15, n, S)  CF-based delta EVE per PLN per shock
      hyp_shock_ids      (S,)        EBA shock scenario IDs (without 'base')
    """
    data = np.load(SCENARIO_CURVES_PATH, allow_pickle=True)
    result = {
        "fwd_rates":    data["fwd_rates"].astype(float),
        "disc_factors": data["disc_factors"].astype(float),
        "scenario_ids": data["scenario_ids"].tolist(),
        "curve_types":  data["curve_types"].tolist(),
        "levels":       data["levels"].tolist(),
        "n_months":     int(data["n_months"][0]),
    }
    for key in ("hyp_nii_unit_rate", "hyp_delta_nii_unit",
                "hyp_eve_pv_factor", "hyp_delta_eve_unit", "hyp_shock_ids"):
        if key in data:
            result[key] = data[key].astype(float) if key != "hyp_shock_ids" else data[key].tolist()
    if "cohort_id" in data:
        result["cohort_id"] = data["cohort_id"]
    return result


@st.cache_data
def load_bs_structure() -> pd.DataFrame:
    df = pd.read_excel(BS_PATH, sheet_name="bs_structure")
    return df[["product_code", "own_name", "bs_side", "bs_percentage", "currency"]].copy()


@st.cache_data
def load_irs_baseline() -> pd.DataFrame:
    df = pd.read_excel(IRS_PATH)
    df["start_date"]    = pd.to_datetime(df["start_date"]).dt.date
    df["maturity_date"] = pd.to_datetime(df["maturity_date"]).dt.date
    keep = ["swap_id", "notional", "pay_fixed", "currency",
            "start_date", "maturity_date", "fixed_rate",
            "float_rate_index", "float_fixing_freq", "float_pay_freq",
            "float_spread", "disc_curve", "fwd_curve"]
    return df[[c for c in keep if c in df.columns]].copy()


# ── balance-sheet editor helpers ──────────────────────────────────────────────

def build_bs_editor_df(params: BalanceSheetParams, bs_struct: pd.DataFrame) -> pd.DataFrame:
    """Return the DataFrame used by st.data_editor in the Balance Sheet tab.

    Columns
    -------
    product_code  str   (hidden in editor, used as join key)
    own_name      str   display label
    bs_side       str   A / L / E
    currency      str
    current_pct   float % of total_assets at baseline
    new_pct       float % of total_assets (editable — starts equal to current_pct)
    """
    # sum balance_arr per (product_code, bs_side) from npz
    balance_map: dict[tuple[str, str], float] = {}
    for pc, side, bal in zip(params.product_code, params.bs_side, params.balance_arr):
        key = (str(pc), str(side))
        balance_map[key] = balance_map.get(key, 0.0) + float(bal)

    # weighted average effective rate: nii_unit_rate × sign gives the absolute
    # annualised rate (positive for both assets and liabilities)
    rate_numer: dict[tuple[str, str], float] = {}
    for pc, side, bal, rate, sgn in zip(
        params.product_code, params.bs_side,
        params.balance_arr, params.nii_unit_rate, params.sign,
    ):
        key = (str(pc), str(side))
        rate_numer[key] = rate_numer.get(key, 0.0) + float(bal) * float(rate) * float(sgn)

    rows = []
    for _, r in bs_struct.iterrows():
        key  = (str(r["product_code"]), str(r["bs_side"]))
        bal  = balance_map.get(key, 0.0)
        cpct = round(bal / float(params.total_assets) * 100.0, 4)
        avg_rate = round(rate_numer.get(key, 0.0) / bal * 100.0, 4) if bal > 0 else 0.0
        rows.append({
            "product_code": str(r["product_code"]),
            "own_name":     str(r["own_name"]),
            "bs_side":      str(r["bs_side"]),
            "currency":     str(r["currency"]),
            "current_pct":  cpct,
            "new_pct":      cpct,
            "avg_rate":     avg_rate,
        })

    return pd.DataFrame(rows)


def compute_weights(params: BalanceSheetParams,
                    new_pcts: dict[tuple[str, str], float]) -> np.ndarray:
    """Scale cohort weights proportionally when product-level % changes.

    Parameters
    ----------
    params   : BalanceSheetParams loaded from npz
    new_pcts : {(product_code_str, bs_side_str): new_display_pct}
               where display_pct is % of total_assets

    Returns
    -------
    weights array (n,) — fraction of total_assets per npz row
    """
    base_w = params.balance_arr / float(params.total_assets)
    new_w  = base_w.copy()

    # pre-compute current total weight per product
    prod_w: dict[tuple[str, str], float] = {}
    for pc, side, w in zip(params.product_code, params.bs_side, base_w):
        key = (str(pc), str(side))
        prod_w[key] = prod_w.get(key, 0.0) + float(w)

    for i, (pc, side) in enumerate(zip(params.product_code, params.bs_side)):
        key = (str(pc), str(side))
        if key not in new_pcts or str(pc) == "0000":
            continue
        old_w = prod_w.get(key, 0.0)
        if old_w > 0.0:
            scale    = (new_pcts[key] / 100.0) / old_w
            new_w[i] = base_w[i] * scale

    return new_w


def run_metrics(weights: np.ndarray,
                params: BalanceSheetParams,
                curves: CurveTensors,
                total_assets: float) -> AllMetrics:
    return compute_all_metrics(weights, params, curves, total_assets)


# ── NMD behavioral model loaders ─────────────────────────────────────────────

NMD_BEH_PATH = os.path.join(PROJECT_ROOT, "balance_gen_add_data", "input", "dep_beh_models_ir.xlsx")

_SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))
if _SANDBOX_DIR not in sys.path:
    sys.path.insert(0, _SANDBOX_DIR)

from nmd_engine import load_nmd_model  # noqa: E402


@st.cache_data
def load_nmd_model_df() -> dict:
    """Load NMD outstanding-pct models from dep_beh_models_ir.xlsx."""
    return load_nmd_model(NMD_BEH_PATH)


def get_nmd_product_info(params: BalanceSheetParams) -> dict:
    """Extract balance, effective rate, sign, and currency for NMD products.

    Returns
    -------
    dict mapping product_code ('6000', '8000') →
        {balance: float, rate: float, sign: float, currency: str}
    rate = abs weighted-average deposit rate (annualised decimal, always positive).
    """
    NMD_PCS = {"6000", "8000"}
    info: dict[str, dict] = {}
    for pc, side, bal, nii, sgn, ccy in zip(
        params.product_code, params.bs_side,
        params.balance_arr, params.nii_unit_rate,
        params.sign, params.currency,
    ):
        if str(pc) not in NMD_PCS:
            continue
        key = str(pc)
        if key not in info:
            info[key] = {"balance": 0.0, "nii_num": 0.0, "sign": float(sgn), "currency": str(ccy)}
        info[key]["balance"] += float(bal)
        info[key]["nii_num"] += float(bal) * float(nii) * float(sgn)
    for v in info.values():
        b = v.pop("nii_num")
        v["rate"] = (b / v["balance"]) if v["balance"] > 0 else 0.0
    return info


# ── IRS metric adjustment ─────────────────────────────────────────────────────

def apply_irs_delta(
    base_m: AllMetrics,
    ana_base: dict,
    ana_new: dict,
) -> dict:
    """Overlay the IRS change on top of already-computed BS metrics.

    The approach keeps the npz-calibrated BS metrics exact and adds only the
    delta from the user's IRS edits:

        final = base_m + (ana_new − ana_base)

    Parameters
    ----------
    base_m   : AllMetrics from compute_all_metrics (includes baseline npz IRS)
    ana_base : analytical IRS metrics for the ORIGINAL IRS book
    ana_new  : analytical IRS metrics for the USER-EDITED IRS book

    Returns
    -------
    Plain dict with keys: nii_base, eve_base, delta_nii, delta_eve, lcr, nsfr.
    LCR and NSFR are unchanged (IRS has no LCR/NSFR weights in this model).
    """
    irs_delta_nii = {s: ana_new["delta_nii"].get(s, 0.0) - ana_base["delta_nii"].get(s, 0.0)
                     for s in base_m.delta_nii}
    irs_delta_eve = {s: ana_new["delta_eve"].get(s, 0.0) - ana_base["delta_eve"].get(s, 0.0)
                     for s in base_m.delta_eve}
    irs_nii_shift = ana_new["nii_base"] - ana_base["nii_base"]
    irs_eve_shift = ana_new["eve_base"] - ana_base["eve_base"]

    return {
        "nii_base":  base_m.nii_base + irs_nii_shift,
        "eve_base":  base_m.eve_base + irs_eve_shift,
        "delta_nii": {s: base_m.delta_nii[s] + irs_delta_nii[s] for s in base_m.delta_nii},
        "delta_eve": {s: base_m.delta_eve[s] + irs_delta_eve[s] for s in base_m.delta_eve},
        "lcr":       base_m.lcr,
        "nsfr":      base_m.nsfr,
        "rwa":       base_m.rwa,
    }
