"""LabBank Sandbox — interactive ALM explorer.

Run:
    streamlit run sandbox/app.py   (from bank_project root)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from baseline import (
    TOLERANCE,
    apply_irs_delta,
    build_bs_editor_df,
    compute_ep,
    compute_weights,
    get_nmd_product_info,
    load_bs_structure,
    load_cohort_rates,
    load_curves,
    load_irs_baseline,
    load_nmd_model_df,
    load_params,
    load_scenario_curves,
    reset_bias_cache,
    run_metrics,
)
from gap_engine  import compute_repricing_gap, compute_liq_gap
from irs_engine  import compute_irs_metrics
from nmd_engine  import NMD_PRODUCTS, compute_nmd_delta, load_nmd_model
from hyp_engine  import compute_hyp_metrics, build_hyp_curve_tensors
from ep_fast     import hyp_margin_rate

# ── hypothetical scenario labels ──────────────────────────────────────────────
_CURVE_LABELS = {
    "normal":   "Normal (rising)",
    "steep":    "Steep",
    "humped":   "Humped",
    "flat":     "Flat",
    "inverted": "Inverted",
}
_LEVEL_LABELS = {
    "low":    "Low  (~0.5%)",
    "medium": "Medium (~2.5%)",
    "high":   "High  (~5.0%)",
}


def _build_stressed_nmd_for_gap(
    nmd_models_df: dict,
    stressed_pct: dict,
) -> dict:
    """Convert stressed pct arrays to the gap_engine dict format.

    gap_engine expects: dict[int, tuple[months_list, outstanding_list]]
    """
    result = {}
    for pc_str, df in nmd_models_df.items():
        pct_arr = stressed_pct.get(pc_str, df["pct"].to_numpy())
        months = []
        for tenor in df["tenor"]:
            t = str(tenor).strip().upper()
            if t == "1D":
                months.append(1.0 / 30.4375)
            elif t.endswith("M"):
                months.append(float(t[:-1]))
            elif t.endswith("Y"):
                months.append(float(t[:-1]) * 12.0)
        result[int(pc_str)] = (months, list(pct_arr))
    # proxy: 6300 (current_account_sme) uses same curve as 6000
    if 6000 in result:
        result[6300] = result[6000]
    return result

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LabBank Sandbox", page_icon="🏦", layout="wide")
st.markdown("""
<style>
[data-testid="stMetricDelta"] { font-size: 1.35rem; }
[data-testid="stMetricDelta"] svg { width: 1.35rem; height: 1.35rem; }
</style>
""", unsafe_allow_html=True)
st.title("🏦 LabBank ALM Sandbox")

# ── data loaders ──────────────────────────────────────────────────────────────
params      = load_params()
curves      = load_curves()
cohort_rates = load_cohort_rates()
bs_struct   = load_bs_structure()
irs_base    = load_irs_baseline()

@st.cache_data
def _ana_irs_base() -> dict:
    return compute_irs_metrics(irs_base, curves)

ana_irs_baseline = _ana_irs_base()


def _compute_adj(bs_df: pd.DataFrame, irs_df: pd.DataFrame, total_assets: float) -> dict:
    new_pcts = {(r["product_code"], r["bs_side"]): r["new_pct"]
                for _, r in bs_df.iterrows()}
    w   = compute_weights(params, new_pcts)
    m   = run_metrics(w, params, curves, total_assets)
    ana = compute_irs_metrics(irs_df, curves)
    return apply_irs_delta(m, ana_irs_baseline, ana)


def _bs_with_notional(base_df: pd.DataFrame, cur_df: pd.DataFrame, total_assets: float) -> pd.DataFrame:
    """Return base_df with new_pct and notional_m reflecting current edits."""
    df = base_df.copy()
    pct_map = cur_df.set_index("own_name")["new_pct"].to_dict()
    df["new_pct"] = df["own_name"].map(pct_map).fillna(df["new_pct"])
    df["notional_m"] = (df["new_pct"] / 100 * total_assets / 1e6).round(0)
    return df


def _get(m, key: str, scen: str = "") -> float:
    if isinstance(m, dict):
        if key == "nii_base":  return m["nii_base"]
        if key == "eve_base":  return m["eve_base"]
        if key == "delta_nii": return m["delta_nii"][scen]
        if key == "delta_eve": return m["delta_eve"][scen]
        if key == "lcr":       return m["lcr"].get("PLN", float("nan"))
        if key == "nsfr":      return m["nsfr"].get("PLN", float("nan"))
        if key == "rwa":       return m.get("rwa", float("nan"))
    else:
        if key == "nii_base":  return m.nii_base
        if key == "eve_base":  return m.eve_base
        if key == "delta_nii": return m.delta_nii[scen]
        if key == "delta_eve": return m.delta_eve[scen]
        if key == "lcr":       return m.lcr.get("PLN", float("nan"))
        if key == "nsfr":      return m.nsfr.get("PLN", float("nan"))
        if key == "rwa":       return m.rwa
    return float("nan")


# ── session state ─────────────────────────────────────────────────────────────
def _fresh_bs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (asset_df, fund_df) from the baseline npz — used as stable editor bases."""
    full = build_bs_editor_df(params, bs_struct)
    a    = full[full["bs_side"] == "A"].reset_index(drop=True)
    f    = full[full["bs_side"].isin(["L", "E"])].reset_index(drop=True)
    return a, f


def _init():
    a, f = _fresh_bs()
    _nmd_models_df = load_nmd_model_df()
    _shocked = list(params.scenario_ids)
    defaults = {
        "asset_base":          a,
        "fund_base":           f,
        "irs_base_ed":         irs_base.copy(),
        "total_assets":        float(params.total_assets),
        "reset_counter":       0,
        "t1_capital":          1_000_000_000.0,
        "_cur_asset":          a.copy(),
        "_cur_fund":           f.copy(),
        "_cur_irs":            irs_base.copy(),
        # NMD stress state — keyed by product_code
        "nmd_stressed_pct":    {pc: df["pct"].to_numpy().copy()
                                for pc, df in _nmd_models_df.items()},
        "nmd_delta_nii":       0.0,
        "nmd_delta_eve_base":  0.0,
        "nmd_delta_eve_sh":    {s: 0.0 for s in _shocked},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_bs, tab_irs, tab_nmd, tab_metrics, tab_profit, tab_gap, tab_curves = st.tabs(
    ["⚖️  Balance Sheet", "🔄  IRS Book", "🏦  NMD Stress", "📈  ALM Metrics", "💰  Finance Metrics", "📊  Gap Analysis", "📉  Market Curves"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — MARKET CURVES
# (executes FIRST, right after the tabs are created, even though it's visually
# the last tab -- its hypothetical-curve selectors need to be resolved before
# ALM Metrics/Finance Metrics run, and rendering them here means their widget
# state can never be perturbed by anything happening in another tab's code
# later in this same script pass, e.g. Balance Sheet validation. See the
# git history for 2026-08-15 if this needs more context.)
# ═════════════════════════════════════════════════════════════════════════════
with tab_curves:
    st.subheader("Market Curves")
    st.caption(
        "Zero-coupon (spot) rates derived from the pre-computed curve tensors. "
        "Base scenario = current market; shock scenarios = IRRBB standard shocks. "
        "PLN only — EUR/USD carry no real market data in this demo book, and the "
        "forward-rate view is dropped in favour of the smoother zero curve (a forward "
        "curve is a derivative-like quantity and reads as a step function even off a "
        "smooth zero curve)."
    )

    _n_m    = curves.n_months
    _months = np.arange(1, _n_m + 1)
    _yrs    = _months / 12.0            # tenor in years (x-axis)

    _all_scens   = curves.scenario_ids.tolist()
    _base_scen   = _all_scens[0]
    _shock_scens = _all_scens[1:]
    _clrs        = px.colors.qualitative.Plotly

    _ccy_sel = "PLN"
    _ci_sel  = curves.currency_index(_ccy_sel)

    def _zero_pct(scen_idx: int) -> np.ndarray:
        _df = np.maximum(curves.disc_factors[scen_idx, _ci_sel, :_n_m], 1e-10)
        return -np.log(_df) / _yrs * 100

    # ── Plot 1: Base zero curve ───────────────────────────────────────────────
    st.markdown("**Base (current) market curve — PLN**")

    fig_base_curve = go.Figure()
    fig_base_curve.add_trace(go.Scatter(
        x=_yrs, y=_zero_pct(0),
        name="PLN zero",
        line=dict(color=_clrs[0], width=2),
    ))
    fig_base_curve.update_layout(
        title=f"Zero (Spot) Rate — {_base_scen}",
        height=400,
        xaxis=dict(title="Tenor (years)"),
        yaxis=dict(title="Rate (%)", ticksuffix="%"),
        margin=dict(t=60, b=5, l=5, r=5),
    )
    st.plotly_chart(fig_base_curve, use_container_width=True)

    st.divider()

    # ── Plot 2: Scenario zero curves ──────────────────────────────────────────
    st.markdown("**IRRBB scenario zero curves — PLN**")

    fig_scen_curves = go.Figure()
    for _si, _scen in enumerate(_all_scens):
        _is_base = (_si == 0)
        fig_scen_curves.add_trace(go.Scatter(
            x=_yrs, y=_zero_pct(_si),
            name=_scen,
            line=dict(
                color=_clrs[_si % len(_clrs)],
                width=3 if _is_base else 1.5,
                dash="solid" if _is_base else "dot",
            ),
        ))

    fig_scen_curves.update_layout(
        title=f"Zero Rate by Scenario — {_ccy_sel}",
        height=420,
        xaxis=dict(title="Tenor (years)"),
        yaxis=dict(title="Rate (%)", ticksuffix="%"),
        legend=dict(orientation="h", y=1.12, font_size=11),
        margin=dict(t=60, b=5, l=5, r=5),
    )
    st.plotly_chart(fig_scen_curves, use_container_width=True)

    # ── Plot 3: Shock shapes (delta from base in bps) ─────────────────────────
    if _shock_scens:
        st.markdown("**Shock shape — shift from base (bps)**")

        _base_fwd_sel = curves.fwd_rates[0, _ci_sel, :_n_m]
        fig_deltas = go.Figure()
        for _si, _scen in enumerate(_all_scens[1:], start=1):
            _delta_bps = (curves.fwd_rates[_si, _ci_sel, :_n_m] - _base_fwd_sel) * 10_000
            fig_deltas.add_trace(go.Scatter(
                x=_yrs, y=_delta_bps,
                name=_scen,
                line=dict(color=_clrs[_si % len(_clrs)], width=1.5),
            ))
        fig_deltas.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
        fig_deltas.update_layout(
            title=f"Rate Shock Profile — {_ccy_sel}",
            height=300,
            xaxis=dict(title="Tenor (years)"),
            yaxis=dict(title="Shift (bps)"),
            legend=dict(orientation="h", y=1.12, font_size=11),
            margin=dict(t=50, b=5, l=5, r=5),
        )
        st.plotly_chart(fig_deltas, use_container_width=True)

    st.caption(
        f"Report date: **{curves.report_date}** | "
        f"Scenarios: {len(_all_scens)}  ({_base_scen} + {len(_shock_scens)} shocks) | "
        f"Currency: PLN | "
        f"Horizon: {_n_m} months ({_n_m // 12} years)"
    )

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # HYPOTHETICAL SCENARIO CURVES — shape × level explorer + IRRBB metrics
    # ═══════════════════════════════════════════════════════════════════════════
    _sc_data = load_scenario_curves()
    st.subheader("Hypothetical Rate Scenarios & IRRBB Metrics")
    st.caption(
        "Select a curve shape and rate level.  The PLN base curve is replaced "
        "by the chosen stylised curve; EBA shock shapes are applied on top.  "
        "NII, EVE, and SOT metrics are recomputed for the new environment."
    )
    st.info(
        "Selecting a curve here updates the **ALM Metrics** and **Finance Metrics** "
        "tabs with hypothetical measures for that rate environment.",
        icon="ℹ️",
    )
    _hc1, _hc2 = st.columns(2)
    with _hc1:
        _sel_shape = st.radio(
            "Curve shape",
            options=["current"] + _sc_data["curve_types"],
            format_func=lambda x: "Current (base)" if x == "current"
                                  else _CURVE_LABELS.get(x, x),
            horizontal=False,
            key="hyp_shape_sel",
        )
    with _hc2:
        _level_disabled = (_sel_shape == "current")
        _sel_level = st.radio(
            "Rate level" + ("  (n/a for current)" if _level_disabled else ""),
            options=_sc_data["levels"],
            format_func=lambda x: _LEVEL_LABELS.get(x, x),
            horizontal=False,
            key="hyp_level_sel",
            disabled=_level_disabled,
        )

    if _sel_shape != "current":
        try:
            _hyp_idx = _sc_data["scenario_ids"].index(f"{_sel_shape}_{_sel_level}")
            st.session_state["hyp_fwd_pln"] = _sc_data["fwd_rates"][_hyp_idx].copy()
            st.session_state["hyp_label"]   = (
                f"{_CURVE_LABELS.get(_sel_shape, _sel_shape)} / "
                f"{_LEVEL_LABELS.get(_sel_level, _sel_level)}"
            )
        except ValueError:
            st.session_state["hyp_fwd_pln"] = None
            st.session_state["hyp_label"]   = None
    else:
        st.session_state["hyp_fwd_pln"] = None
        st.session_state["hyp_label"]   = None

    _sc_yrs  = np.arange(1, _sc_data["n_months"] + 1) / 12.0

    _SHAPE_COLORS = {
        "normal": "#4C72B0", "steep": "#55A868",
        "humped": "#C44E52", "flat":  "#8172B2", "inverted": "#CCB974",
    }
    _LVL_COLORS = {"low": "#AED6F1", "medium": "#2980B9", "high": "#1A5276"}

    # ── retrieve forward-rate arrays (decimal) ────────────────────────────────
    _pln_idx  = curves.currency_index("PLN")
    _n_sc     = _sc_data["n_months"]
    _cur_fwd  = curves.fwd_rates[0, _pln_idx, :_n_sc]           # decimal
    _cur_fwd_pct  = _cur_fwd * 100.0
    _cur_df   = np.maximum(curves.disc_factors[0, _pln_idx, :_n_sc], 1e-10)
    _cur_zero = -np.log(_cur_df) / _sc_yrs * 100.0

    if _sel_shape == "current":
        _hyp_fwd_dec = _cur_fwd                                  # decimal, (360,)
        _hyp_fwd_pct = _cur_fwd_pct
        _hyp_zero    = _cur_zero
        _hyp_label   = "Current (base)"
    else:
        _sid         = f"{_sel_shape}_{_sel_level}"
        _sc_idx      = _sc_data["scenario_ids"].index(_sid)
        _hyp_fwd_dec = _sc_data["fwd_rates"][_sc_idx]            # decimal
        _hyp_fwd_pct = _hyp_fwd_dec * 100.0
        _hyp_df      = np.maximum(_sc_data["disc_factors"][_sc_idx], 1e-10)
        _hyp_zero    = -np.log(_hyp_df) / _sc_yrs * 100.0
        _hyp_label   = f"{_CURVE_LABELS[_sel_shape]} / {_LEVEL_LABELS[_sel_level]}"

    # ── curve comparison chart ─────────────────────────────────────────────────
    _fig_hyp = go.Figure()
    _fig_hyp.add_trace(go.Scatter(x=_sc_yrs, y=_cur_fwd_pct, name="Current fwd",
                                  line=dict(color="#4C72B0", width=2.5)))
    _fig_hyp.add_trace(go.Scatter(x=_sc_yrs, y=_cur_zero, name="Current zero",
                                  line=dict(color="#4C72B0", width=1.5, dash="dash")))
    if _sel_shape != "current":
        _fig_hyp.add_trace(go.Scatter(x=_sc_yrs, y=_hyp_fwd_pct, name=f"{_hyp_label} fwd",
                                      line=dict(color="#DD8452", width=2.5)))
        _fig_hyp.add_trace(go.Scatter(x=_sc_yrs, y=_hyp_zero, name=f"{_hyp_label} zero",
                                      line=dict(color="#DD8452", width=1.5, dash="dash")))
    _fig_hyp.update_layout(
        title=f"Forward & Zero Rates — {_hyp_label} vs Current",
        height=380, xaxis=dict(title="Tenor (years)"),
        yaxis=dict(title="Rate (%)", ticksuffix="%"),
        legend=dict(orientation="h", y=1.12, font_size=11),
        margin=dict(t=60, b=5, l=5, r=5),
    )
    st.plotly_chart(_fig_hyp, use_container_width=True)

    # ── all-shapes overview + all-levels overview (side by side) ──────────────
    _ov_c1, _ov_c2 = st.columns(2)
    _lvl_ov = _sel_level if not _level_disabled else "medium"

    _ov_fig = go.Figure()
    _ov_fig.add_trace(go.Scatter(x=_sc_yrs, y=_cur_fwd_pct, name="Current",
                                 line=dict(color="black", width=2)))
    for _ct in _sc_data["curve_types"]:
        _ix = _sc_data["scenario_ids"].index(f"{_ct}_{_lvl_ov}")
        _ov_fig.add_trace(go.Scatter(
            x=_sc_yrs, y=_sc_data["fwd_rates"][_ix] * 100.0,
            name=_CURVE_LABELS.get(_ct, _ct),
            line=dict(color=_SHAPE_COLORS.get(_ct, "grey"), width=2,
                      dash="solid" if _ct == _sel_shape else "dot"),
            opacity=1.0 if _ct == _sel_shape else 0.55,
        ))
    _ov_fig.update_layout(
        title=f"All shapes — {_LEVEL_LABELS.get(_lvl_ov, _lvl_ov)}",
        height=320, xaxis=dict(title="Tenor (years)"),
        yaxis=dict(title="Rate (%)", ticksuffix="%"),
        legend=dict(orientation="h", y=1.18, font_size=10),
        margin=dict(t=55, b=5, l=5, r=5),
    )
    _ov_c1.plotly_chart(_ov_fig, use_container_width=True)

    if _sel_shape != "current":
        _lv_fig = go.Figure()
        _lv_fig.add_trace(go.Scatter(x=_sc_yrs, y=_cur_fwd_pct, name="Current",
                                     line=dict(color="black", width=2)))
        for _lv in _sc_data["levels"]:
            _ix = _sc_data["scenario_ids"].index(f"{_sel_shape}_{_lv}")
            _lv_fig.add_trace(go.Scatter(
                x=_sc_yrs, y=_sc_data["fwd_rates"][_ix] * 100.0,
                name=_LEVEL_LABELS.get(_lv, _lv),
                line=dict(color=_LVL_COLORS.get(_lv, "grey"), width=2,
                          dash="solid" if _lv == _sel_level else "dot"),
                opacity=1.0 if _lv == _sel_level else 0.55,
            ))
        _lv_fig.update_layout(
            title=f"{_CURVE_LABELS.get(_sel_shape, '')} — all levels",
            height=320, xaxis=dict(title="Tenor (years)"),
            yaxis=dict(title="Rate (%)", ticksuffix="%"),
            legend=dict(orientation="h", y=1.18, font_size=10),
            margin=dict(t=55, b=5, l=5, r=5),
        )
        _ov_c2.plotly_chart(_lv_fig, use_container_width=True)

    # ── IRRBB metrics — compute only when not "current" ───────────────────────
    st.divider()
    if _sel_shape == "current":
        st.info("Select a hypothetical curve shape above to compute IRRBB metrics "
                "for that rate environment.", icon="ℹ️")
    else:
        st.markdown(f"**IRRBB Metrics — {_hyp_label}**")
        st.caption(
            "NII base: floating-product rates repriced to hypothetical curve level; "
            "fixed products unchanged.  "
            "delta_NII: calibrated shock sensitivities (level-independent).  "
            "EVE base + delta_EVE: CF re-discounting with hypothetical curves."
        )

        # balance from current BS state
        _ta       = st.session_state.total_assets
        _base_w   = params.balance_arr / float(params.total_assets)
        _balance  = _base_w * _ta

        with st.spinner("Computing hypothetical IRRBB metrics..."):
            _hyp_m = compute_hyp_metrics(
                balance     = _balance,
                params      = params,
                cr          = cohort_rates,
                curves      = curves,
                fwd_hyp_pln = _hyp_fwd_dec,
            )
        # base metrics (current environment, for comparison)
        _cur_m  = run_metrics(_base_w, params, curves, _ta)
        _t1_cap = st.session_state.get("t1_capital", 1_000_000_000.0)

        # ── KPI row ───────────────────────────────────────────────────────────
        _km = st.columns(4)
        _km[0].metric("NII base (current)",      f"{_cur_m.nii_base/1e6:,.0f} M")
        _km[1].metric("NII base (hyp)",          f"{_hyp_m['nii_base']/1e6:,.0f} M",
                      delta=f"{(_hyp_m['nii_base']-_cur_m.nii_base)/1e6:+,.1f} M")
        _km[2].metric("EVE base (current)",      f"{_cur_m.eve_base/1e6:,.0f} M")
        _km[3].metric("EVE base (hyp)",          f"{_hyp_m['eve_base']/1e6:,.0f} M",
                      delta=f"{(_hyp_m['eve_base']-_cur_m.eve_base)/1e6:+,.1f} M")

        # ── scenario charts ───────────────────────────────────────────────────
        _shared_scens = sorted(
            set(_hyp_m["delta_nii"]) & set(_cur_m.delta_nii)
        )

        def _hyp_bar(title, cur_vals, hyp_vals, scens, unit="M PLN"):
            fig = go.Figure([
                go.Bar(name="Current", x=scens, y=cur_vals,
                       marker_color="#4C72B0", opacity=0.8,
                       text=[f"{v:.1f}" for v in cur_vals], textposition="outside"),
                go.Bar(name=_hyp_label, x=scens, y=hyp_vals,
                       marker_color="#DD8452",
                       text=[f"{v:.1f}" for v in hyp_vals], textposition="outside"),
            ])
            fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
            fig.update_layout(title=title, barmode="group", height=340,
                              yaxis_title=unit,
                              legend=dict(orientation="h", y=1.08),
                              margin=dict(t=50, b=5, l=5, r=5))
            return fig

        _gc1, _gc2 = st.columns(2)
        _gc1.plotly_chart(_hyp_bar(
            "delta NII by Scenario (M PLN)",
            [_cur_m.delta_nii.get(s, 0)/1e6 for s in _shared_scens],
            [_hyp_m["delta_nii"].get(s, 0)/1e6 for s in _shared_scens],
            _shared_scens,
        ), use_container_width=True)
        _gc2.plotly_chart(_hyp_bar(
            "delta EVE by Scenario (M PLN)",
            [_cur_m.delta_eve.get(s, 0)/1e6 for s in _shared_scens],
            [_hyp_m["delta_eve"].get(s, 0)/1e6 for s in _shared_scens],
            _shared_scens,
        ), use_container_width=True)

        # ── SOT comparison table ───────────────────────────────────────────────
        st.markdown("**EBA Supervisory Outlier Test (SOT)**")
        _sot_rows = []
        for _s in _shared_scens:
            _c_eve = _cur_m.delta_eve.get(_s, 0) / _t1_cap * 100
            _h_eve = _hyp_m["delta_eve"].get(_s, 0) / _t1_cap * 100
            _c_nii = _cur_m.delta_nii.get(_s, 0) / _t1_cap * 100
            _h_nii = _hyp_m["delta_nii"].get(_s, 0) / _t1_cap * 100
            _sot_rows.append({
                "Scenario":            _s,
                "dEVE/T1 current (%)": f"{_c_eve:.1f}",
                "dEVE/T1 hyp (%)":     f"{_h_eve:.1f} {'OK' if _h_eve>=-15 else 'FAIL'}",
                "dNII/T1 current (%)": f"{_c_nii:.1f}",
                "dNII/T1 hyp (%)":     f"{_h_nii:.1f} {'OK' if _h_nii>=-5 else 'FAIL'}",
            })
        st.dataframe(pd.DataFrame(_sot_rows), hide_index=True, use_container_width=True)

        _worst_eve_h = min(_hyp_m["delta_eve"].values()) / _t1_cap * 100
        _worst_nii_h = min(_hyp_m["delta_nii"].values()) / _t1_cap * 100
        _c1, _c2 = st.columns(2)
        (_c1.success if _worst_eve_h >= -15 else _c1.error)(
            f"EVE SOT: {'PASS' if _worst_eve_h>=-15 else 'FAIL'} "
            f"(worst {_worst_eve_h:.1f}%  threshold −15%)"
        )
        (_c2.success if _worst_nii_h >= -5 else _c2.error)(
            f"NII SOT: {'PASS' if _worst_nii_h>=-5 else 'FAIL'} "
            f"(worst {_worst_nii_h:.1f}%  threshold −5%)"
        )

    st.caption(
        f"Hypothetical scenarios: **{len(_sc_data['scenario_ids'])}** "
        f"({len(_sc_data['curve_types'])} shapes × {len(_sc_data['levels'])} levels) | "
        f"Stored in sandbox/scenario_curves.npz"
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — BALANCE SHEET
# ═════════════════════════════════════════════════════════════════════════════
with tab_bs:

    c_ta, c_rst = st.columns([4, 1])
    with c_ta:
        total_assets_input = st.number_input(
            "Total Assets (PLN)", min_value=1_000_000_000, max_value=100_000_000_000,
            value=int(st.session_state.total_assets), step=500_000_000, format="%d",
        )
        st.session_state.total_assets = float(total_assets_input)
    with c_rst:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ Reset BS", use_container_width=True):
            a, f = _fresh_bs()
            st.session_state.asset_base    = a
            st.session_state.fund_base     = f
            st.session_state["_cur_asset"] = a.copy()
            st.session_state["_cur_fund"]  = f.copy()
            st.session_state.reset_counter += 1
            st.rerun()

    st.info("Edit the **✏️ New %** column — type the new percentage directly. "
            "Both sides must sum to 100%.", icon="✏️")

    rc = st.session_state.reset_counter

    # ── editors — always receive the STABLE baseline; editor holds its own deltas
    _ECFG = {
        "own_name":    st.column_config.TextColumn("Product",     disabled=True, width="medium"),
        "role":        st.column_config.TextColumn("Type",        disabled=True, width="small"),
        "currency":    st.column_config.TextColumn("CCY",         disabled=True, width="small"),
        "current_pct": st.column_config.NumberColumn(
            "Base %",   disabled=True, format="%.1f", width="small"),
        "new_pct": st.column_config.NumberColumn(
            "✏️ New %", min_value=0.0, max_value=100.0, step=0.5, format="%.1f",
            width="small",
            help="Enter your new allocation. Both sides must sum to 100%."),
        "avg_rate": st.column_config.NumberColumn(
            "Rate %", disabled=True, format="%.2f%%", width="small",
            help="Weighted-average effective interest rate from the NII model "
                 "(assets: earned rate; liabilities: cost rate). Read-only."),
        "notional_m": st.column_config.NumberColumn(
            "Notional M", disabled=True, format="%.0f", width="small"),
    }

    col_a, col_f = st.columns(2, gap="large")
    with col_a:
        st.subheader("Assets")
        edited_assets = st.data_editor(
            _bs_with_notional(st.session_state.asset_base, st.session_state.asset_base, total_assets_input),
            column_config=_ECFG,
            column_order=["own_name", "currency", "current_pct", "new_pct", "avg_rate", "notional_m"],
            hide_index=True, use_container_width=True,
            key=f"ed_a_{rc}",
        )
        asset_sum = float(edited_assets["new_pct"].sum())
        asset_ok  = abs(asset_sum - 100.0) < TOLERANCE
        if asset_ok:
            st.success(f"Sum: **{asset_sum:.2f}%** ✓")
        else:
            diff = asset_sum - 100.0
            st.error(f"Sum: **{asset_sum:.2f}%** — "
                     f"{'add' if diff < 0 else 'remove'} **{abs(diff):.2f}%**")

    with col_f:
        st.subheader("Liabilities + Equity")
        _fund_disp = _bs_with_notional(st.session_state.fund_base, st.session_state.fund_base, total_assets_input)
        _fund_disp["role"] = _fund_disp["bs_side"].map({"E": "E (T1)", "L": "L", "A": ""}).fillna("")
        edited_fund = st.data_editor(
            _fund_disp,
            column_config=_ECFG,
            column_order=["own_name", "role", "currency", "current_pct", "new_pct", "avg_rate", "notional_m"],
            hide_index=True, use_container_width=True,
            key=f"ed_f_{rc}",
        )
        fund_sum = float(edited_fund["new_pct"].sum())
        fund_ok  = abs(fund_sum - 100.0) < TOLERANCE
        if fund_ok:
            st.success(f"Sum: **{fund_sum:.2f}%** ✓")
        else:
            diff = fund_sum - 100.0
            st.error(f"Sum: **{fund_sum:.2f}%** — "
                     f"{'add' if diff < 0 else 'remove'} **{abs(diff):.2f}%**")
        # derive T1 from equity rows — store for Metrics tab
        _equity_pct = float(edited_fund.loc[edited_fund["bs_side"] == "E", "new_pct"].sum())
        st.session_state.t1_capital = _equity_pct / 100.0 * total_assets_input

    st.session_state.bs_valid = asset_ok and fund_ok
    if not st.session_state.bs_valid:
        st.warning("⚠️ Both sides must sum to 100% before metrics are computed.")

    # snapshot current edits
    st.session_state["_cur_asset"] = edited_assets.copy()
    st.session_state["_cur_fund"]  = edited_fund.copy()

    # composition chart
    st.divider()
    st.subheader("Composition")

    def _comp_chart(edit_df: pd.DataFrame, base_df: pd.DataFrame, title: str) -> go.Figure:
        fig  = go.Figure()
        clrs = px.colors.qualitative.Plotly
        for i, (_, row) in enumerate(base_df.iterrows()):
            c = clrs[i % len(clrs)]
            new_pct_val = float(edit_df.loc[edit_df["own_name"] == row["own_name"], "new_pct"].values[0])
            fig.add_trace(go.Bar(
                name=row["own_name"],
                y=["Baseline", "Modified"],
                x=[row["current_pct"], new_pct_val],
                orientation="h", marker_color=c,
                text=[f"{row['current_pct']:.1f}%", f"{new_pct_val:.1f}%"],
                textposition="inside", insidetextanchor="middle",
            ))
        fig.update_layout(
            barmode="stack", title=title, height=160,
            margin=dict(t=35, b=5, l=5, r=5),
            legend=dict(orientation="h", y=-0.3, font_size=10),
            xaxis=dict(range=[0, 102], ticksuffix="%"),
        )
        return fig

    ca, cf = st.columns(2)
    ca.plotly_chart(_comp_chart(edited_assets, st.session_state.asset_base, "Assets"),
                    use_container_width=True)
    cf.plotly_chart(_comp_chart(edited_fund, st.session_state.fund_base, "Liabilities + Equity"),
                    use_container_width=True)
    st.caption(f"Report Date: **{params.report_date}** | "
               f"Total Assets: **{total_assets_input/1e9:.1f} B PLN**")

    st.divider()
    if st.button(
        "🔄 Reload my data", use_container_width=False,
        help="Clear cached balance sheet, curves, and npz tensors, then reload "
             "from disk. Use this after regenerating your own balance sheet "
             "(labbank_data_job) so LabBank picks up the new files without "
             "restarting Streamlit. Resets the Balance Sheet tab to the new "
             "baseline — if the product mix changed, edits against the old "
             "mix can't be carried forward meaningfully.",
    ):
        st.cache_data.clear()
        st.cache_resource.clear()
        reset_bias_cache()
        a, f = _fresh_bs()
        st.session_state.asset_base    = a
        st.session_state.fund_base     = f
        st.session_state["_cur_asset"] = a.copy()
        st.session_state["_cur_fund"]  = f.copy()
        st.session_state.reset_counter += 1
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — IRS BOOK
# ═════════════════════════════════════════════════════════════════════════════
with tab_irs:

    c_hd, c_rst2 = st.columns([5, 1])
    c_hd.subheader("Interest Rate Swap Book")
    with c_rst2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ Reset IRS", use_container_width=True):
            st.session_state.irs_base_ed = irs_base.copy()
            st.session_state.reset_counter += 1
            st.rerun()

    st.info(
        "**Convention** — *Pay Fixed?* **unchecked (☐)** = bank **receives fixed, pays floating** "
        "(WIBOR). This is the correct hedge for floating-rate loan books: "
        "the bank pays WIBOR to the IRS counterparty, offsetting WIBOR received from borrowers, "
        "leaving a fixed spread.\n\n"
        "Edit notional or fixed rate. Add rows with **+**; delete via row checkbox.",
        icon="ℹ️",
    )

    edited_irs = st.data_editor(
        st.session_state.irs_base_ed,        # stable base — editor holds own deltas
        num_rows="dynamic", use_container_width=True,
        column_config={
            "swap_id":          st.column_config.TextColumn("Swap ID", width="small"),
            "notional":         st.column_config.NumberColumn(
                                    "Notional (PLN)", min_value=0, step=1_000_000,
                                    format="%d", width="large"),
            "pay_fixed":        st.column_config.CheckboxColumn(
                                    "Pay Fixed?", width="small",
                                    help="☐ = receive fixed / pay WIBOR  |  ☑ = pay fixed / receive WIBOR"),
            "currency":         st.column_config.TextColumn("CCY", width="small"),
            "start_date":       st.column_config.DateColumn("Start", width="small"),
            "maturity_date":    st.column_config.DateColumn("Maturity", width="small"),
            "fixed_rate":       st.column_config.NumberColumn(
                                    "Fixed Rate", min_value=0.0, max_value=1.0,
                                    step=0.0025, format="%.4f", width="small"),
            "float_rate_index": st.column_config.TextColumn("Float Index", width="small"),
            "float_spread":     st.column_config.NumberColumn("Float Spread",
                                    format="%.4f", width="small"),
            "disc_curve":       st.column_config.TextColumn("Disc Curve", width="small"),
            "fwd_curve":        st.column_config.TextColumn("Fwd Curve", width="small"),
        },
        hide_index=True, key=f"ed_irs_{rc}",
    )
    st.session_state["_cur_irs"] = edited_irs.copy()

    if len(edited_irs) > 0:
        st.divider()
        st.subheader("Leg Summary")

        def _leg_stats(df: pd.DataFrame) -> dict:
            pay_m    = df["pay_fixed"].astype(bool)
            notl     = df["notional"].fillna(0)
            recv_n   = float(notl[~pay_m].sum())
            pay_n    = float(notl[pay_m].sum())
            recv_sub = df[~pay_m]
            pay_sub  = df[pay_m]
            wt_recv  = float((recv_sub["fixed_rate"].fillna(0) * recv_sub["notional"].fillna(0)).sum()
                             / max(recv_n, 1))
            wt_pay   = float((pay_sub["fixed_rate"].fillna(0) * pay_sub["notional"].fillna(0)).sum()
                             / max(pay_n, 1))
            n_recv   = int((~pay_m).sum())
            n_pay    = int(pay_m.sum())
            return {"recv_n": recv_n, "pay_n": pay_n,
                    "wt_recv": wt_recv, "wt_pay": wt_pay,
                    "n_recv": n_recv, "n_pay": n_pay}

        base_s = _leg_stats(irs_base)
        curr_s = _leg_stats(edited_irs)

        rows_leg = [
            {"Leg": "Receive-Fixed (pay WIBOR)",
             "Metric": "Notional (M PLN)",
             "Baseline": f"{base_s['recv_n']/1e6:,.0f}",
             "Current":  f"{curr_s['recv_n']/1e6:,.0f}",
             "Delta":    f"{(curr_s['recv_n']-base_s['recv_n'])/1e6:+,.0f}"},
            {"Leg": "Receive-Fixed (pay WIBOR)",
             "Metric": "# Swaps",
             "Baseline": str(base_s['n_recv']),
             "Current":  str(curr_s['n_recv']),
             "Delta":    f"{curr_s['n_recv']-base_s['n_recv']:+d}"},
            {"Leg": "Receive-Fixed (pay WIBOR)",
             "Metric": "Wtd Avg Fixed Rate (received)",
             "Baseline": f"{base_s['wt_recv']:.3%}",
             "Current":  f"{curr_s['wt_recv']:.3%}",
             "Delta":    f"{(curr_s['wt_recv']-base_s['wt_recv'])*10000:+.1f} bp"},
            {"Leg": "Pay-Fixed (receive WIBOR)",
             "Metric": "Notional (M PLN)",
             "Baseline": f"{base_s['pay_n']/1e6:,.0f}",
             "Current":  f"{curr_s['pay_n']/1e6:,.0f}",
             "Delta":    f"{(curr_s['pay_n']-base_s['pay_n'])/1e6:+,.0f}"},
            {"Leg": "Pay-Fixed (receive WIBOR)",
             "Metric": "# Swaps",
             "Baseline": str(base_s['n_pay']),
             "Current":  str(curr_s['n_pay']),
             "Delta":    f"{curr_s['n_pay']-base_s['n_pay']:+d}"},
            {"Leg": "Pay-Fixed (receive WIBOR)",
             "Metric": "Wtd Avg Fixed Rate (paid)",
             "Baseline": f"{base_s['wt_pay']:.3%}" if base_s['pay_n'] > 0 else "—",
             "Current":  f"{curr_s['wt_pay']:.3%}" if curr_s['pay_n'] > 0 else "—",
             "Delta":    f"{(curr_s['wt_pay']-base_s['wt_pay'])*10000:+.1f} bp" if
                         (base_s['pay_n'] > 0 or curr_s['pay_n'] > 0) else "—"},
            {"Leg": "Net",
             "Metric": "Net Receive-Fixed Notional (M PLN)",
             "Baseline": f"{(base_s['recv_n']-base_s['pay_n'])/1e6:,.0f}",
             "Current":  f"{(curr_s['recv_n']-curr_s['pay_n'])/1e6:,.0f}",
             "Delta":    f"{((curr_s['recv_n']-curr_s['pay_n'])-(base_s['recv_n']-base_s['pay_n']))/1e6:+,.0f}"},
        ]
        st.dataframe(pd.DataFrame(rows_leg), hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — METRICS
# ═════════════════════════════════════════════════════════════════════════════
with tab_metrics:

    if not st.session_state.get("bs_valid", False):
        st.warning("⚠️ Fix the balance sheet (both sides = 100%) to compute metrics.")
        st.stop()

    t1 = st.session_state.t1_capital
    ta_val = st.session_state.total_assets

    # build combined bs_df from both editors
    combined_bs = pd.concat(
        [st.session_state["_cur_asset"], st.session_state["_cur_fund"]],
        ignore_index=True,
    )
    base_w  = params.balance_arr / float(params.total_assets)
    base_m  = run_metrics(base_w, params, curves, ta_val)
    mod_adj = _compute_adj(combined_bs, st.session_state["_cur_irs"], ta_val)
    scens   = list(base_m.delta_nii.keys())

    # ── Hypothetical curve overlay ────────────────────────────────────────────
    _hyp_fwd = st.session_state.get("hyp_fwd_pln")
    _hyp_lbl = st.session_state.get("hyp_label", "")
    if _hyp_fwd is not None:
        _new_pcts_c = {(r["product_code"], r["bs_side"]): r["new_pct"]
                       for _, r in combined_bs.iterrows()}
        _w_c    = compute_weights(params, _new_pcts_c)
        _bal_c  = _w_c * ta_val
        _mkt_mc = run_metrics(_w_c, params, curves, ta_val)

        _sc_ld  = load_scenario_curves()
        # scenario_curves.npz is a per-cohort cache (see build_scenario_curves.py)
        # that goes stale whenever product_params.npz is regenerated with a
        # different cohort set (e.g. a new product added) -- fall back to the
        # linear approximation below instead of a shape-mismatch crash.
        _has_pc = (
            "hyp_nii_unit_rate" in _sc_ld
            and "cohort_id" in _sc_ld
            and len(_sc_ld["cohort_id"]) == len(params.cohort_id)
            and np.array_equal(_sc_ld["cohort_id"], params.cohort_id)
        )
        _hyp_shape = st.session_state.get("hyp_shape_sel", "current")
        _hyp_level = st.session_state.get("hyp_level_sel", "")
        _hyp_key   = f"{_hyp_shape}_{_hyp_level}"
        try:
            _pc_idx = _sc_ld["scenario_ids"].index(_hyp_key) if _has_pc else None
        except ValueError:
            _pc_idx = None

        if _has_pc and _pc_idx is not None:
            # ── pre-computed path: floor-correct NII delta, CF-correct EVE ──
            _hyp_nii = float(np.dot(_bal_c, np.nan_to_num(_sc_ld["hyp_nii_unit_rate"][_pc_idx])))
            _hyp_eve = float(np.dot(_bal_c, np.nan_to_num(_sc_ld["hyp_eve_pv_factor"][_pc_idx])))
            mod_adj["nii_base"] += _hyp_nii - _mkt_mc.nii_base
            mod_adj["eve_base"] += _hyp_eve - _mkt_mc.eve_base

            _shock_ids = list(_sc_ld["hyp_shock_ids"])
            for _s in list(mod_adj["delta_nii"]):
                if _s in _shock_ids:
                    _k    = _shock_ids.index(_s)
                    _irs  = mod_adj["delta_nii"][_s] - _mkt_mc.delta_nii.get(_s, 0.0)
                    mod_adj["delta_nii"][_s] = (
                        float(np.dot(_bal_c, np.nan_to_num(_sc_ld["hyp_delta_nii_unit"][_pc_idx, :, _k])))
                        + _irs
                    )
            for _s in list(mod_adj["delta_eve"]):
                if _s in _shock_ids:
                    _k    = _shock_ids.index(_s)
                    _irs  = mod_adj["delta_eve"][_s] - _mkt_mc.delta_eve.get(_s, 0.0)
                    mod_adj["delta_eve"][_s] = (
                        float(np.dot(_bal_c, np.nan_to_num(_sc_ld["hyp_delta_eve_unit"][_pc_idx, :, _k])))
                        + _irs
                    )
        else:
            # ── fallback: linear approximation (run build_scenario_curves.py) ─
            _hyp_mc = compute_hyp_metrics(_bal_c, params, cohort_rates, curves, _hyp_fwd)
            mod_adj["nii_base"] += _hyp_mc["nii_base"] - _mkt_mc.nii_base
            mod_adj["eve_base"] += _hyp_mc["eve_base"] - _mkt_mc.eve_base
            if not _has_pc:
                st.caption("ℹ️ Run `python sandbox/build_scenario_curves.py` for "
                           "floor-correct delta_NII and CF-correct delta_EVE.")

    # ── NMD overlay: add analytical delta from NMD stress tab ────────────────
    _nmd_nii      = st.session_state.get("nmd_delta_nii",      0.0)
    _nmd_eve_base = st.session_state.get("nmd_delta_eve_base", 0.0)
    _nmd_eve_sh   = st.session_state.get("nmd_delta_eve_sh",   {})
    mod_adj["nii_base"] += _nmd_nii
    mod_adj["eve_base"] += _nmd_eve_base
    # delta_eve[s] = EVE_s − EVE_base; NMD shifts both → relative delta shifts by (Δs − Δbase)
    for _s in mod_adj["delta_eve"]:
        mod_adj["delta_eve"][_s] += _nmd_eve_sh.get(_s, 0.0) - _nmd_eve_base

    # ── KPIs — two rows: Baseline / Modified ─────────────────────────────────
    st.subheader("Summary")

    def _pp(new, old):
        return f"{(new-old)*100:+.1f} pp" if not (np.isnan(new) or np.isnan(old)) else None

    # ── T1/RWA configurable limit ─────────────────────────────────────────────
    min_t1_rwa_pct = st.number_input(
        "Min T1/RWA ratio (%)", min_value=0.0, max_value=50.0, value=18.0, step=0.5,
        format="%.1f", help="CET1 floor: Tier 1 Capital / RWA. Basel III minimum is 8%; internal limit typically 10–18%.",
    )
    min_t1_rwa = min_t1_rwa_pct / 100.0

    kpi_defs = [
        ("NII",     lambda m: _get(m,"nii_base"),   "{:,.0f} M", lambda b,m: f"{(m-b)/1e6:+,.1f} M"),
        ("EVE",     lambda m: _get(m,"eve_base"),    "{:,.0f} M", lambda b,m: f"{(m-b)/1e6:+,.1f} M"),
        ("LCR PLN", lambda m: _get(m,"lcr"),         "{:.2%}",    lambda b,m: _pp(m,b)),
        ("NSFR PLN",lambda m: _get(m,"nsfr"),        "{:.2%}",    lambda b,m: _pp(m,b)),
    ]
    b_scale = [1/1e6, 1/1e6, 1, 1]

    brow = st.columns(4)
    mrow = st.columns(4)
    for i, (lbl, fn, fmt, dfn) in enumerate(kpi_defs):
        bv = fn(base_m) * b_scale[i]
        mv = fn(mod_adj) * b_scale[i]
        brow[i].metric(f"Baseline — {lbl}", fmt.format(bv))
        mrow[i].metric(f"Modified — {lbl}", fmt.format(mv), delta=dfn(fn(base_m), fn(mod_adj)))

    # ── RWA & T1/RWA row ─────────────────────────────────────────────────────
    rwa_b   = _get(base_m, "rwa")
    rwa_m   = _get(mod_adj, "rwa")
    t1r_b   = t1 / rwa_b  if rwa_b > 0 else float("nan")
    t1r_m   = t1 / rwa_m  if rwa_m > 0 else float("nan")

    rwa_row = st.columns(4)
    rwa_row[0].metric("Baseline — RWA",      f"{rwa_b/1e6:,.0f} M")
    rwa_row[1].metric("Modified — RWA",      f"{rwa_m/1e6:,.0f} M",
                      delta=f"{(rwa_m-rwa_b)/1e6:+,.1f} M")
    rwa_row[2].metric("Baseline — T1/RWA",   f"{t1r_b:.1%}")
    rwa_row[3].metric("Modified — T1/RWA",   f"{t1r_m:.1%}",
                      delta=_pp(t1r_m, t1r_b))

    # pass/fail for T1/RWA
    t1r_ok = (not np.isnan(t1r_m)) and t1r_m >= min_t1_rwa
    (st.success if t1r_ok else st.error)(
        f"T1/RWA: {'PASS' if t1r_ok else 'FAIL'}  "
        f"(modified {t1r_m:.1%}  vs  limit {min_t1_rwa:.1%}"
        + (f"  |  RWA {rwa_m/1e6:,.0f} M  T1 {t1/1e6:,.0f} M)" if not np.isnan(rwa_m) else ")")
    )

    _nmd_active = abs(_nmd_nii) > 1e3 or abs(_nmd_eve_base) > 1e3
    st.caption(
        f"Baseline = exact pipeline (npz-calibrated). "
        f"Modified = BS scale change + IRS analytical delta"
        + (" + NMD behavioral stress" if _nmd_active else "")
        + (f" + curve **{_hyp_lbl}**" if _hyp_fwd is not None else "")
        + f". **Tier 1 Capital: {t1/1e6:,.0f} M PLN** (equity rows from Balance Sheet)."
    )

    st.divider()

    # ── EBA SOT ───────────────────────────────────────────────────────────────
    st.subheader("EBA Supervisory Outlier Test")
    st.caption("dEVE/T1 > −15%,  dNII/T1 > −5%")

    sot_eb = {s: base_m.delta_eve[s]/t1*100 for s in scens}
    sot_nb = {s: base_m.delta_nii[s]/t1*100 for s in scens}
    sot_em = {s: _get(mod_adj,"delta_eve",s)/t1*100 for s in scens}
    sot_nm = {s: _get(mod_adj,"delta_nii",s)/t1*100 for s in scens}

    def _sot_chart(title, bv, mv, thr):
        lbls = list(bv.keys())
        b    = list(bv.values())
        m    = list(mv.values())
        fig  = go.Figure()
        fig.add_trace(go.Bar(y=lbls, x=b, orientation="h", name="Baseline",
                             marker_color="#4C72B0", opacity=0.7,
                             text=[f"{v:.1f}%" for v in b], textposition="outside"))
        fig.add_trace(go.Bar(y=lbls, x=m, orientation="h", name="Modified",
                             marker_color=["crimson" if v<thr else "#DD8452" for v in m],
                             text=[f"{v:.1f}%" for v in m], textposition="outside"))
        fig.add_vline(x=thr, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"{thr}%", annotation_position="top right")
        fig.update_layout(title=title, barmode="group", height=300,
                          xaxis_ticksuffix="%", legend=dict(orientation="h", y=1.1),
                          margin=dict(t=50, b=5, l=5, r=5))
        return fig

    sc1, sc2 = st.columns(2)
    sc1.plotly_chart(_sot_chart("Δ EVE / T1 (%)", sot_eb, sot_em, -15.0), use_container_width=True)
    sc2.plotly_chart(_sot_chart("Δ NII / T1 (%)", sot_nb, sot_nm, -5.0),  use_container_width=True)

    we = min(sot_em.values()); wn = min(sot_nm.values())
    pa, pb = st.columns(2)
    (pa.success if we>=-15 else pa.error)(f"EVE SOT: {'PASS' if we>=-15 else 'FAIL'}  (worst {we:.1f}%)")
    (pb.success if wn>=-5  else pb.error)(f"NII SOT: {'PASS' if wn>=-5  else 'FAIL'}  (worst {wn:.1f}%)")

    st.divider()

    # ── scenario charts ───────────────────────────────────────────────────────
    def _gbar(title, bv, mv, xlbls):
        fig = go.Figure([
            go.Bar(name="Baseline", x=xlbls, y=bv, marker_color="#4C72B0",
                   text=[f"{v:.1f}" for v in bv], textposition="outside"),
            go.Bar(name="Modified", x=xlbls, y=mv, marker_color="#DD8452",
                   text=[f"{v:.1f}" for v in mv], textposition="outside"),
        ])
        fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
        fig.update_layout(title=title, barmode="group", height=380,
                          legend=dict(orientation="h", y=1.08),
                          margin=dict(t=50, b=5, l=5, r=5), yaxis_title="M PLN")
        return fig

    gc1, gc2 = st.columns(2)
    gc1.plotly_chart(_gbar("Δ NII by Scenario (M PLN)",
                           [base_m.delta_nii[s]/1e6 for s in scens],
                           [_get(mod_adj,"delta_nii",s)/1e6 for s in scens], scens),
                     use_container_width=True)
    gc2.plotly_chart(_gbar("Δ EVE by Scenario (M PLN)",
                           [base_m.delta_eve[s]/1e6 for s in scens],
                           [_get(mod_adj,"delta_eve",s)/1e6 for s in scens], scens),
                     use_container_width=True)

    st.divider()

    # ── IRS expander + full table ─────────────────────────────────────────────
    cur_irs = st.session_state["_cur_irs"]
    ana_new  = compute_irs_metrics(cur_irs, curves)
    with st.expander("IRS contribution detail"):
        irs_rows = [
            {"Metric": "NII base",
             "Baseline (npz)": f"{ana_irs_baseline['nii_base']/1e6:.2f}M",
             "Modified":       f"{ana_new['nii_base']/1e6:.2f}M",
             "Delta":          f"{(ana_new['nii_base']-ana_irs_baseline['nii_base'])/1e6:+.2f}M"},
            {"Metric": "EVE base",
             "Baseline (npz)": f"{ana_irs_baseline['eve_base']/1e6:.2f}M",
             "Modified":       f"{ana_new['eve_base']/1e6:.2f}M",
             "Delta":          f"{(ana_new['eve_base']-ana_irs_baseline['eve_base'])/1e6:+.2f}M"},
        ]
        for s in scens:
            irs_rows.append({"Metric": f"dNII {s}",
                "Baseline (npz)": f"{ana_irs_baseline['delta_nii'].get(s,0)/1e6:.2f}M",
                "Modified":       f"{ana_new['delta_nii'].get(s,0)/1e6:.2f}M",
                "Delta":          f"{(ana_new['delta_nii'].get(s,0)-ana_irs_baseline['delta_nii'].get(s,0))/1e6:+.2f}M"})
        for s in scens:
            irs_rows.append({"Metric": f"dEVE {s}",
                "Baseline (npz)": f"{ana_irs_baseline['delta_eve'].get(s,0)/1e6:.2f}M",
                "Modified":       f"{ana_new['delta_eve'].get(s,0)/1e6:.2f}M",
                "Delta":          f"{(ana_new['delta_eve'].get(s,0)-ana_irs_baseline['delta_eve'].get(s,0))/1e6:+.2f}M"})
        st.dataframe(pd.DataFrame(irs_rows).rename(columns={"Metric":"Metric"}),
                     hide_index=True, use_container_width=True)

    st.subheader("Full Scenario Table")
    tbl = []
    for lbl, bv, mv in [("NII base (M)", base_m.nii_base, _get(mod_adj,"nii_base")),
                         ("EVE base (M)", base_m.eve_base, _get(mod_adj,"eve_base"))]:
        tbl.append({"Metric": lbl, "Baseline": f"{bv/1e6:,.1f}", "Modified": f"{mv/1e6:,.1f}",
                    "Delta": f"{(mv-bv)/1e6:+,.1f}"})
    for s in scens:
        bv=base_m.delta_nii[s]; mv=_get(mod_adj,"delta_nii",s)
        tbl.append({"Metric": f"dNII {s} (M)", "Baseline": f"{bv/1e6:,.1f}",
                    "Modified": f"{mv/1e6:,.1f}", "Delta": f"{(mv-bv)/1e6:+,.1f}"})
    for s in scens:
        bv=base_m.delta_eve[s]; mv=_get(mod_adj,"delta_eve",s)
        tbl.append({"Metric": f"dEVE {s} (M)", "Baseline": f"{bv/1e6:,.1f}",
                    "Modified": f"{mv/1e6:,.1f}", "Delta": f"{(mv-bv)/1e6:+,.1f}"})
    for s in scens:
        bv=base_m.delta_eve[s]/t1*100; mv=_get(mod_adj,"delta_eve",s)/t1*100
        tbl.append({"Metric": f"dEVE/T1 {s} (%)",
                    "Baseline": f"{bv:.1f}%", "Modified": f"{mv:.1f}% {'✓' if mv>=-15 else '✗'}",
                    "Delta": f"{(mv-bv):+.1f} pp"})
    for s in scens:
        bv=base_m.delta_nii[s]/t1*100; mv=_get(mod_adj,"delta_nii",s)/t1*100
        tbl.append({"Metric": f"dNII/T1 {s} (%)",
                    "Baseline": f"{bv:.1f}%", "Modified": f"{mv:.1f}% {'✓' if mv>=-5 else '✗'}",
                    "Delta": f"{(mv-bv):+.1f} pp"})
    for ccy, bv in base_m.lcr.items():
        mv=_get(mod_adj,"lcr")
        tbl.append({"Metric": f"LCR {ccy}", "Baseline": f"{bv:.2%}", "Modified": f"{mv:.2%}",
                    "Delta": f"{(mv-bv)*100:+.1f} pp"})
    for ccy, bv in base_m.nsfr.items():
        mv=_get(mod_adj,"nsfr")
        tbl.append({"Metric": f"NSFR {ccy}", "Baseline": f"{bv:.2%}", "Modified": f"{mv:.2%}",
                    "Delta": f"{(mv-bv)*100:+.1f} pp"})
    st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

    # ── LCR / NSFR diagnostic ─────────────────────────────────────────────────
    st.divider()
    st.subheader("LCR / NSFR Diagnostic")
    st.caption(
        "LCR = HQLA / Net Outflows₃₀d.  "
        "HQLA = liquid assets (bonds + cash) × haircut.  "
        "If HQLA = 0 you've zeroed all liquid assets — check bond rows in the Balance Sheet."
    )

    from lcr_fast import compute_lcr_components_fast

    # build modified amounts from the same weights used for mod_adj
    _mod_new_pcts = {(r["product_code"], r["bs_side"]): r["new_pct"]
                     for _, r in combined_bs.iterrows()}
    _mod_w        = compute_weights(params, _mod_new_pcts)
    _mod_amounts  = _mod_w * ta_val
    _base_amounts = (params.balance_arr / float(params.total_assets)) * ta_val

    lcr_rows = []
    for ccy in sorted(set(params.currency.tolist())):
        bc = compute_lcr_components_fast(_base_amounts, params, ccy)
        mc = compute_lcr_components_fast(_mod_amounts,  params, ccy)
        lcr_rows.append({
            "CCY": ccy,
            "": "Baseline",
            "HQLA (M)":          f"{bc['hqla']/1e6:,.1f}",
            "Outflows 30d (M)":  f"{bc['outflows_30d']/1e6:,.1f}",
            "Inflows 30d (M)":   f"{bc['inflows_30d']/1e6:,.1f}",
            "Net Outflows (M)":  f"{bc['net_outflows_30d']/1e6:,.1f}",
            "LCR":               f"{bc['lcr']:.2%}" if not np.isnan(bc['lcr']) else "n/a",
        })
        lcr_rows.append({
            "CCY": ccy,
            "": "Modified",
            "HQLA (M)":          f"{mc['hqla']/1e6:,.1f}",
            "Outflows 30d (M)":  f"{mc['outflows_30d']/1e6:,.1f}",
            "Inflows 30d (M)":   f"{mc['inflows_30d']/1e6:,.1f}",
            "Net Outflows (M)":  f"{mc['net_outflows_30d']/1e6:,.1f}",
            "LCR":               f"{mc['lcr']:.2%}" if not np.isnan(mc['lcr']) else "n/a",
        })
        if mc["hqla"] == 0 and bc["hqla"] > 0:
            st.error(
                f"⚠️ **{ccy} HQLA = 0** — all liquid assets (bond_fixed_gov, bond_float_gov, "
                f"cash_gov) have been set to 0% in the Balance Sheet. "
                f"Check those rows and press **↺ Reset BS** if needed."
            )

    st.dataframe(pd.DataFrame(lcr_rows), hide_index=True, use_container_width=True)

    # show per-product new_pct for HQLA assets so the user can see the issue
    hqla_mask = params.hqla_factor > 0
    hqla_pcs  = set(zip(params.product_code[hqla_mask], params.bs_side[hqla_mask]))
    hqla_rows = [r for _, r in combined_bs.iterrows() if (r["product_code"], r["bs_side"]) in hqla_pcs]
    if hqla_rows:
        with st.expander("HQLA asset allocations (from Balance Sheet editor)"):
            st.dataframe(
                pd.DataFrame(hqla_rows)[["own_name", "current_pct", "new_pct"]],
                hide_index=True, use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4b — FINANCE METRICS (Economic Profit)
# ═════════════════════════════════════════════════════════════════════════════
with tab_profit:

    # ── Economic Profit ────────────────────────────────────────────────────────
    st.subheader("Economic Profit")
    st.caption(
        "EP = Margin (over FTP) + Fee − Expected Loss − Cost of Capital − OpEx "
        "− AcqCost — the same objective the balance-sheet optimizer "
        "(`bs_optimization/`) maximises, computed here instantly for whatever "
        "mix is on the Balance Sheet tab right now, baseline vs. modified, "
        "including the IRS Book tab's swap edits."
    )

    _mod_new_pcts_ep = {(r["product_code"], r["bs_side"]): r["new_pct"]
                        for _, r in combined_bs.iterrows()}
    _mod_amounts_ep  = compute_weights(params, _mod_new_pcts_ep) * ta_val

    # ── Hypothetical curve overlay: Margin/FTP move with the curve too, not
    # just NII/EVE (ALM Metrics tab) -- floating-rate FTP refixes to the
    # hypothetical curve exactly like floating client rates do; fixed-rate
    # FTP stays locked at origination (unchanged, by design -- see
    # ftp_store.py). Same precomputed-cache convention as the ALM Metrics
    # tab: only activates when scenario_curves.npz has this exact hyp
    # scenario AND its cohort set still matches product_params.npz; falls
    # back to the real curve (today's behavior) otherwise. (2026-08-15)
    _hyp_fwd_ep = st.session_state.get("hyp_fwd_pln")
    _ep_margin_override, _ep_nii_override = None, None
    if _hyp_fwd_ep is not None:
        _sc_ld_ep  = load_scenario_curves()
        _hyp_key_ep = f"{st.session_state.get('hyp_shape_sel', 'current')}_{st.session_state.get('hyp_level_sel', '')}"
        _has_hyp_ep = (
            "hyp_nii_unit_rate" in _sc_ld_ep
            and "cohort_id" in _sc_ld_ep
            and len(_sc_ld_ep["cohort_id"]) == len(params.cohort_id)
            and np.array_equal(_sc_ld_ep["cohort_id"], params.cohort_id)
            and _hyp_key_ep in _sc_ld_ep["scenario_ids"]
        )
        if _has_hyp_ep:
            _pc_idx_ep = _sc_ld_ep["scenario_ids"].index(_hyp_key_ep)
            _ep_nii_override    = _sc_ld_ep["hyp_nii_unit_rate"][_pc_idx_ep]
            _hyp_curves_ep       = build_hyp_curve_tensors(curves, _hyp_fwd_ep)
            _ep_margin_override  = hyp_margin_rate(params, _hyp_curves_ep)
            st.info(f"Margin/FTP/NII below reflect the **{st.session_state.get('hyp_label', '')}** "
                    "curve selected on the Market Curves tab.", icon="ℹ️")
        else:
            st.caption("ℹ️ A hypothetical curve is selected on the Market Curves tab, but this "
                       "book/cache doesn't support it here yet — showing the real curve. "
                       "Run `python sandbox/build_scenario_curves.py` if this persists.")

    ep_base = compute_ep(params.balance_arr.copy(), cohort_rates,
                         margin_rate_override=_ep_margin_override, nii_unit_override=_ep_nii_override)
    ep_mod  = compute_ep(_mod_amounts_ep, cohort_rates,
                         margin_rate_override=_ep_margin_override, nii_unit_override=_ep_nii_override)

    # ── Explicit real-vs-hypothetical comparison, same pattern as the ALM
    # Metrics tab's "NII base (current)" / "NII base (hyp)" KPIs. The
    # waterfalls below apply the SAME curve to both Baseline and Modified
    # (they only differ by balance-sheet mix), so on an unedited sheet both
    # panels move TOGETHER and look identical to each other -- correct, but
    # easy to misread as "nothing happened" with no on-screen reference to
    # what the real curve would have given. This row is that reference,
    # computed on the baseline mix so it isolates the curve's effect from
    # any balance-sheet edit. (2026-08-15)
    if _ep_margin_override is not None:
        _ep_real_for_cmp = compute_ep(params.balance_arr.copy(), cohort_rates)
        _kc1, _kc2 = st.columns(2)
        _kc1.metric("EP on baseline mix — real curve", f"{_ep_real_for_cmp['ep']/1e6:+,.0f} M")
        _kc2.metric(f"EP on baseline mix — {st.session_state.get('hyp_label', '')}",
                    f"{ep_base['ep']/1e6:+,.0f} M",
                    delta=f"{(ep_base['ep'] - _ep_real_for_cmp['ep'])/1e6:+,.1f} M vs. real curve")

    # ── IRS overlay: compute_weights() always leaves product '0000' (IRS) at
    # its static npz baseline (see baseline.compute_weights), same as every
    # other metric in this tab -- the user's IRS Book edits are layered on
    # top analytically, same convention as _compute_adj()/apply_irs_delta()
    # above. FTP=0 for the swap book (verified: margin_rate == nii_unit_rate
    # for product '0000'), so its margin moves exactly with its NII -- add
    # the same delta to both. Fee/EL/CoC/OpEx/AcqCost have no live formula
    # for the swap book here (same limitation the NII/EVE overlay accepts).
    _ana_irs_new_ep = compute_irs_metrics(st.session_state["_cur_irs"], curves)
    _irs_nii_delta  = _ana_irs_new_ep["nii_base"] - ana_irs_baseline["nii_base"]
    ep_mod["nii"]    += _irs_nii_delta
    ep_mod["margin"] += _irs_nii_delta
    ep_mod["ep"]     += _irs_nii_delta

    def _ep_waterfall(title, comp, total_color):
        cats = ["NII", "± FTP", "Margin", "+ Fee", "− EL", "− CoC", "− OpEx", "− AcqCost", "EP"]
        vals = [comp["nii"], comp["ftp"], comp["margin"], comp["fee"],
                -comp["el"], -comp["coc"], -comp["opex"], -comp["acq_cost"], comp["ep"]]
        meas = ["relative", "relative", "total", "relative", "relative",
                "relative", "relative", "relative", "total"]
        fig = go.Figure(go.Waterfall(
            x=cats, y=[v / 1e6 for v in vals], measure=meas,
            text=[f"{v/1e6:+,.0f}M" for v in vals], textposition="outside",
            increasing=dict(marker_color="#2E7D32"), decreasing=dict(marker_color="crimson"),
            totals=dict(marker_color=total_color),
            connector=dict(line=dict(color="lightgrey")),
        ))
        fig.update_layout(title=title, height=380, showlegend=False,
                          margin=dict(t=50, b=5, l=5, r=5), yaxis_title="M PLN")
        return fig

    ep1, ep2 = st.columns(2)
    ep1.plotly_chart(_ep_waterfall(f"Baseline — EP {ep_base['ep']/1e6:+,.0f}M", ep_base, "#4C72B0"),
                     use_container_width=True)
    ep2.plotly_chart(_ep_waterfall(f"Modified — EP {ep_mod['ep']/1e6:+,.0f}M", ep_mod, "#DD8452"),
                     use_container_width=True)

    st.metric("Δ Economic Profit (Modified − Baseline)",
             f"{(ep_mod['ep'] - ep_base['ep'])/1e6:+,.1f} M PLN")
    st.caption(
        "AcqCost is charged only on balance growth above the baseline weight "
        "per product (zero if Modified matches Baseline) — moving weight "
        "*into* a product shows up here even though the app's other metrics "
        "don't have an acquisition-cost concept."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — GAP ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_gap:

    if not st.session_state.get("bs_valid", False):
        st.warning("⚠️ Fix the balance sheet (both sides = 100%) to compute the gap.")
        st.stop()

    cur_irs_gap = st.session_state["_cur_irs"]

    # Build stressed NMD models for the gap engine from session-state pct vectors
    _nmd_models_df  = load_nmd_model_df()
    _nmd_str_pct    = st.session_state.get("nmd_stressed_pct", {})
    _nmd_gap_models = _build_stressed_nmd_for_gap(_nmd_models_df, _nmd_str_pct)

    # Balance Sheet tab edits → scaled weights, same machinery as the Metrics tab
    _combined_bs_gap = pd.concat(
        [st.session_state["_cur_asset"], st.session_state["_cur_fund"]],
        ignore_index=True,
    )
    _new_pcts_gap = {(str(r["product_code"]), str(r["bs_side"])): r["new_pct"]
                      for _, r in _combined_bs_gap.iterrows()}
    _bs_weights_gap = compute_weights(params, _new_pcts_gap)

    rep = compute_repricing_gap(params, cur_irs_gap, nmd_models=_nmd_gap_models,
                                 weights=_bs_weights_gap)
    # fully shipped-baseline gap (no BS/IRS/NMD edits) — for the delta arrows below
    rep_base = compute_repricing_gap(params, irs_base)

    buckets  = rep["bucket"].tolist()
    bs_a_m   = (rep["bs_assets"] / 1e6).tolist()   # positive (green)
    bs_l_m   = (rep["bs_liab"]   / 1e6).tolist()   # shown as negative (red)
    irs_g_m  = (rep["irs_gap"]   / 1e6).tolist()   # black
    net_m    = (rep["net_gap"]   / 1e6).tolist()
    cum_bs_m = (rep["cum_bs_gap"]  / 1e6).tolist()
    cum_net_m= (rep["cum_net_gap"] / 1e6).tolist()

    base_bs_a_m  = (rep_base["bs_assets"] / 1e6).tolist()
    base_bs_l_m  = (rep_base["bs_liab"]   / 1e6).tolist()
    base_irs_g_m = (rep_base["irs_gap"]   / 1e6).tolist()

    def _sign_colors(vals, pos_clr="#388E3C", neg_clr="#D32F2F"):
        return [pos_clr if v >= 0 else neg_clr for v in vals]

    # vertical separator between monthly and longer-term buckets
    SEP_X = 11.5   # between M12 and 1-2Y in category index

    def _add_sep(fig):
        fig.add_vline(x=SEP_X, line_dash="dot", line_color="#BDBDBD", line_width=1.5,
                      annotation_text="12M ↔ longer", annotation_position="top")

    st.subheader("Repricing Gap (IR Gap)")

    # ── Panel 1: green (assets) + red (liabilities) at SAME position, black IRS adjacent ──
    # offsetgroup="bs" → green and red share one bar slot (one goes up, one down)
    # offsetgroup="irs" → IRS bar placed next to the BS pair
    st.markdown("**Panel 1 — Asset repricing (green ↑) and Liability repricing (red ↓) "
                "at the same position, with IRS contribution (black) adjacent**")
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name="BS Assets repricing", x=buckets, y=bs_a_m,
        marker_color="#388E3C", offsetgroup="bs",
        text=[f"{v:.0f}" if v > 80 else "" for v in bs_a_m],
        textposition="inside", insidetextanchor="start",
    ))
    fig1.add_trace(go.Bar(
        name="BS Liabilities repricing", x=buckets, y=[-v for v in bs_l_m],
        marker_color="#D32F2F", offsetgroup="bs",  # same group → overlaid at same x
        text=[f"{v:.0f}" if v > 80 else "" for v in bs_l_m],
        textposition="inside", insidetextanchor="start",
    ))
    fig1.add_trace(go.Bar(
        name="IRS net", x=buckets, y=irs_g_m,
        marker_color="#212121", opacity=0.85, offsetgroup="irs",  # different group → adjacent
        text=[f"{v:+.0f}" if abs(v) > 20 else "" for v in irs_g_m],
        textposition="outside",
    ))
    fig1.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    _add_sep(fig1)
    fig1.update_layout(
        barmode="group", height=440, yaxis_title="M PLN",
        legend=dict(orientation="h", y=1.06),
        margin=dict(t=40, b=5, l=5, r=5),
        bargroupgap=0.15,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Panel 2: two side-by-side net gap charts — without and with IRS ─────────
    st.markdown("**Panel 2 — Net repricing gap: without IRS (left) vs with IRS (right)**")

    bs_net_m  = [a - l for a, l in zip(bs_a_m, bs_l_m)]   # BS-only net gap
    net_m_v   = [(a - l + irs) for a, l, irs in zip(bs_a_m, bs_l_m, irs_g_m)]  # with IRS

    # shared y-axis range across both charts so bars are visually comparable
    base_bs_net_m = [a - l for a, l in zip(base_bs_a_m, base_bs_l_m)]
    base_net_m_v  = [(a - l + irs) for a, l, irs in zip(base_bs_a_m, base_bs_l_m, base_irs_g_m)]
    delta_bs_net  = [c - b for c, b in zip(bs_net_m, base_bs_net_m)]
    delta_net_v   = [c - b for c, b in zip(net_m_v,  base_net_m_v)]

    _all_p2    = bs_net_m + net_m_v
    _p2_span   = (max(_all_p2) - min(_all_p2)) or 100
    _p2_pad    = _p2_span * 0.15
    _p2_ymax   = max(_all_p2) + _p2_pad
    _p2_delta_y = min(_all_p2) - _p2_pad          # row where the delta labels sit
    _p2_ymin   = _p2_delta_y - _p2_span * 0.12    # extra headroom below the delta row

    def _delta_annotations(fig, deltas, thresh=1.0):
        """Small ▲/▼ + value per bucket, below the bars, vs. shipped baseline.
        Uses data ('y') coordinates, not 'paper' — 'paper' y<0 sits below the
        whole canvas and gets clipped, so it never rendered."""
        for b, d in zip(buckets, deltas):
            if abs(d) < thresh:
                continue
            arrow = "▲" if d > 0 else "▼"
            color = "#2E7D32" if d > 0 else "#C62828"
            fig.add_annotation(
                x=b, y=_p2_delta_y, xref="x", yref="y",
                text=f"{arrow}{d:+.0f}", showarrow=False,
                font=dict(size=15, color=color),
            )

    def _net_gap_bar(title: str, values: list, deltas: list) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=buckets, y=values,
            marker_color=_sign_colors(values),
            text=[f"{v:+.0f}" if abs(v) > 30 else "" for v in values],
            textposition="outside",
            showlegend=False,
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
        _add_sep(fig)
        _delta_annotations(fig, deltas)
        fig.update_layout(
            title=title, height=420, yaxis_title="M PLN",
            yaxis=dict(range=[_p2_ymin, _p2_ymax]),
            margin=dict(t=45, b=5, l=5, r=5),
        )
        return fig

    st.caption("▲/▼ below each bucket: change vs. the shipped baseline "
               "(unedited Balance Sheet, IRS book, and NMD decay).")
    p2a, p2b = st.columns(2)
    p2a.plotly_chart(_net_gap_bar("Net Gap — Balance Sheet only", bs_net_m, delta_bs_net),
                     use_container_width=True)
    p2b.plotly_chart(_net_gap_bar("Net Gap — Balance Sheet + IRS", net_m_v, delta_net_v),
                     use_container_width=True)

    # ── Panel 3: cumulative net gap ───────────────────────────────────────────
    st.markdown("**Panel 3 — Cumulative net gap (BS + IRS)**")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Cum BS gap", x=buckets, y=cum_bs_m,
        marker_color=_sign_colors(cum_bs_m), opacity=0.45,
    ))
    fig3.add_trace(go.Scatter(
        name="Cum Net gap", x=buckets, y=cum_net_m,
        mode="lines+markers+text",
        line=dict(color="black", width=2),
        marker=dict(size=7, color="black"),
        text=[f"{v:+.0f}" for v in cum_net_m], textposition="top center",
    ))
    fig3.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    _add_sep(fig3)
    fig3.update_layout(
        height=380, yaxis_title="M PLN (cumulative)",
        legend=dict(orientation="h", y=1.06),
        margin=dict(t=40, b=5, l=5, r=5),
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Repricing gap data table"):
        tbl_rep = rep.copy()
        for c in ["bs_assets","bs_liab","bs_gap","irs_assets","irs_liab",
                  "irs_gap","net_gap","cum_bs_gap","cum_net_gap"]:
            tbl_rep[c] = (tbl_rep[c] / 1e6).round(0)
        st.dataframe(tbl_rep, hide_index=True, use_container_width=True)

    st.divider()

    # ── Liquidity Gap ─────────────────────────────────────────────────────────
    st.subheader("Liquidity Gap — 12-Month Capital Flows")
    st.caption("Principal inflows from assets (blue) and outflows from liabilities (red) "
               "per month.  Net LIQ gap = inflows − outflows (dashed line).")

    liq      = compute_liq_gap(params)
    months_l = liq["month"].tolist()
    a_in     = (liq["asset_inflows"] / 1e6).tolist()
    l_out    = (liq["liab_outflows"] / 1e6).tolist()
    net_l    = (liq["net_liq_gap"]   / 1e6).tolist()
    cum_l    = (liq["cum_net"]       / 1e6).tolist()

    fig_liq = go.Figure()
    fig_liq.add_trace(go.Bar(name="Asset inflows",  x=months_l, y=a_in,
                             marker_color="#1565C0",
                             text=[f"{v:.0f}" for v in a_in], textposition="outside"))
    fig_liq.add_trace(go.Bar(name="Liab outflows",  x=months_l, y=[-v for v in l_out],
                             marker_color="#B71C1C",
                             text=[f"{v:.0f}" for v in l_out], textposition="outside"))
    fig_liq.add_trace(go.Scatter(name="Net LIQ gap", x=months_l, y=net_l,
                                 mode="lines+markers",
                                 line=dict(color="black", width=2, dash="dash"),
                                 marker=dict(size=6)))
    fig_liq.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    fig_liq.update_layout(barmode="overlay", height=420, yaxis_title="M PLN",
                           legend=dict(orientation="h", y=1.06),
                           margin=dict(t=40, b=5, l=5, r=5))
    st.plotly_chart(fig_liq, use_container_width=True)

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Bar(name="Cumulative Net LIQ", x=months_l, y=cum_l,
                             marker_color=_sign_colors(cum_l),
                             text=[f"{v:+.0f}" for v in cum_l], textposition="outside"))
    fig_cum.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
    fig_cum.update_layout(title="Cumulative Net Liquidity Gap (M PLN)", height=300,
                           yaxis_title="M PLN", margin=dict(t=40, b=5, l=5, r=5))
    st.plotly_chart(fig_cum, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — NMD BEHAVIORAL MODEL STRESS
# ═════════════════════════════════════════════════════════════════════════════
with tab_nmd:
    st.subheader("Non-Maturity Deposit — Behavioral Model Stress")
    st.caption(
        "Edit the **✏️ Stressed %** column to change the outstanding-percentage profile. "
        "Values represent the fraction of deposits still on-book at each tenor. "
        "The **1D** row captures overnight repricing (20% of the stock by default). "
        "ΔNII and ΔEVE are propagated to the **Metrics** and **Gap Analysis** tabs."
    )

    # ── data loaders ──────────────────────────────────────────────────────────
    nmd_models    = load_nmd_model_df()
    nmd_prod_info = get_nmd_product_info(params)

    # ── product selector + reset ──────────────────────────────────────────────
    pc_labels = [f"{pc} — {name}" for pc, name in NMD_PRODUCTS.items()
                 if pc in nmd_models and pc in nmd_prod_info]
    if not pc_labels:
        st.warning("NMD behavioral model data not found. Re-run the pipeline.")
    else:
        c_sel, c_rst_nmd = st.columns([5, 1])
        with c_sel:
            pc_sel_label = st.radio("Select product", pc_labels, horizontal=True)
        with c_rst_nmd:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("↺ Reset NMD", use_container_width=True):
                for _pc, _df in nmd_models.items():
                    st.session_state["nmd_stressed_pct"][_pc] = _df["pct"].to_numpy().copy()
                st.session_state["nmd_delta_nii"]      = 0.0
                st.session_state["nmd_delta_eve_base"] = 0.0
                st.session_state["nmd_delta_eve_sh"]   = {s: 0.0 for s in params.scenario_ids}
                st.session_state.reset_counter += 1
                st.rerun()

        pc_sel   = pc_sel_label.split(" ")[0]
        model_df = nmd_models[pc_sel]
        prod     = nmd_prod_info[pc_sel]
        balance  = prod["balance"]
        rate     = prod["rate"]
        sign     = prod["sign"]
        currency = prod["currency"]

        shocked_scens = list(params.scenario_ids)

        st.divider()

        # ── build editor DataFrame — initialise Stressed % from session state ─
        K         = len(model_df)
        pct_base  = model_df["pct"].to_numpy(dtype=float)
        cum_yf    = model_df["cum_yf"].to_numpy(dtype=float)
        tenor_lbl = model_df["tenor"].tolist()

        pct_prev_base = np.empty(K)
        pct_prev_base[0] = 1.0
        if K > 1:
            pct_prev_base[1:] = pct_base[:-1]
        outflow_base_pct = (pct_prev_base - pct_base) * 100.0

        # Always seed the editor from the STABLE baseline, never from session
        # state -- each product already gets its own widget key (nmd_ed_{pc_sel}_{rc})
        # below, so Streamlit's own per-key state already preserves edits when
        # switching products/tabs. Feeding session_state back into `data=` here
        # created a feedback loop (this rerun's seed = last rerun's own output),
        # which is the classic Streamlit data_editor desync that needs a second
        # identical edit before it "sticks" -- same bug class the Balance Sheet
        # editor already avoids by always passing its static baseline (2026-08-15 fix).
        _init_stressed = pct_base
        editor_df = pd.DataFrame({
            "Tenor":          tenor_lbl,
            "Baseline %":     (pct_base * 100.0).round(2),
            "Stressed %":     (_init_stressed * 100.0).round(2),
            "Outflow base %": outflow_base_pct.round(2),
        })

        col_ed, col_ch = st.columns([2, 3], gap="large")

        with col_ed:
            st.markdown(
                f"**{pc_sel} — {NMD_PRODUCTS[pc_sel]}** "
                f"| Balance: **{balance/1e6:,.0f} M PLN** "
                f"| Deposit rate: **{rate*100:.4f}%**"
            )
            edited_nmd = st.data_editor(
                editor_df,
                column_config={
                    "Tenor": st.column_config.TextColumn(
                        "Tenor", disabled=True, width="small"),
                    "Baseline %": st.column_config.NumberColumn(
                        "Baseline %", disabled=True, format="%.2f", width="small"),
                    "Stressed %": st.column_config.NumberColumn(
                        "✏️ Stressed %", min_value=0.0, max_value=100.0,
                        step=1.0, format="%.2f", width="small",
                        help="Outstanding fraction (% of balance) — edit to stress."),
                    "Outflow base %": st.column_config.NumberColumn(
                        "Outflow base %", disabled=True, format="%.2f", width="small"),
                },
                hide_index=True,
                use_container_width=True,
                key=f"nmd_ed_{pc_sel}_{rc}",
            )

            pct_stressed = np.clip(
                edited_nmd["Stressed %"].to_numpy(dtype=float) / 100.0, 0.0, 1.0
            )
            # Persist this product's stressed pct so the other tabs can read it
            st.session_state["nmd_stressed_pct"][pc_sel] = pct_stressed

            if K > 1 and np.any(np.diff(pct_stressed) > 0):
                st.warning(
                    "⚠️ Outstanding % is not monotonically decreasing at every step. "
                    "NMD models typically have pct(k) ≤ pct(k-1). Check your edits."
                )

            pct_prev_str = np.empty(K)
            pct_prev_str[0] = 1.0
            if K > 1:
                pct_prev_str[1:] = pct_stressed[:-1]
            outflow_str_pct = (pct_prev_str - pct_stressed) * 100.0

        with col_ch:
            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(
                x=tenor_lbl, y=(pct_base * 100.0).tolist(),
                name="Baseline", fill="tozeroy",
                line=dict(color="#4C72B0", width=2),
                fillcolor="rgba(76,114,176,0.20)",
            ))
            fig_prof.add_trace(go.Scatter(
                x=tenor_lbl, y=(pct_stressed * 100.0).tolist(),
                name="Stressed", fill="tozeroy",
                line=dict(color="#DD8452", width=2, dash="dash"),
                fillcolor="rgba(221,132,82,0.20)",
            ))
            fig_prof.update_layout(
                title="Outstanding Profile (% of balance)",
                height=280, yaxis_ticksuffix="%",
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=45, b=5, l=5, r=5),
                xaxis=dict(tickangle=45, nticks=20),
            )
            st.plotly_chart(fig_prof, use_container_width=True)

            fig_out = go.Figure()
            fig_out.add_trace(go.Bar(
                x=tenor_lbl, y=outflow_base_pct.tolist(),
                name="Baseline outflow", marker_color="#4C72B0", opacity=0.65,
            ))
            fig_out.add_trace(go.Bar(
                x=tenor_lbl, y=outflow_str_pct.tolist(),
                name="Stressed outflow", marker_color="#DD8452", opacity=0.8,
            ))
            fig_out.update_layout(
                title="Period Outflow per Tenor Bucket (% of balance)",
                barmode="group", height=240, yaxis_ticksuffix="%",
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=45, b=5, l=5, r=5),
                xaxis=dict(tickangle=45, nticks=20),
            )
            st.plotly_chart(fig_out, use_container_width=True)

        st.divider()

        # ── aggregate delta across ALL NMD products ────────────────────────────
        # Compute total ΔNII + ΔEVE for every product (using current session state pct).
        # This total flows into the Metrics and Gap Analysis tabs.
        _total_dnii      = 0.0
        _total_deve_base = 0.0
        _total_deve_sh   = {s: 0.0 for s in shocked_scens}

        for _pc, _mdf in nmd_models.items():
            if _pc not in nmd_prod_info:
                continue
            _prod     = nmd_prod_info[_pc]
            _pct_old  = _mdf["pct"].to_numpy(dtype=float)
            _pct_new  = st.session_state["nmd_stressed_pct"].get(_pc, _pct_old)
            _cum_yf   = _mdf["cum_yf"].to_numpy(dtype=float)
            _r = compute_nmd_delta(
                balance=_prod["balance"], rate=_prod["rate"], sign=_prod["sign"],
                pct_old=_pct_old, pct_new=_pct_new, cum_yf=_cum_yf,
                curves=curves, currency=_prod["currency"],
                shocked_scenario_ids=shocked_scens,
                horizon_yf=1.0,
            )
            _total_dnii      += _r["delta_nii"]
            _total_deve_base += _r["delta_eve_base"]
            for _s in shocked_scens:
                _total_deve_sh[_s] = _total_deve_sh.get(_s, 0.0) + _r["delta_eve"].get(_s, 0.0)

        # Persist totals so Metrics tab reads them
        st.session_state["nmd_delta_nii"]      = _total_dnii
        st.session_state["nmd_delta_eve_base"] = _total_deve_base
        st.session_state["nmd_delta_eve_sh"]   = _total_deve_sh

        st.caption(
            f"Model shown: **{pc_sel} — {NMD_PRODUCTS[pc_sel]}** | "
            f"Balance: {balance/1e6:,.0f} M PLN | Rate: {rate*100:.4f}% | "
            f"Tenors: {K} | "
            "ΔNII / ΔEVE impact is on the **Metrics** and **Gap Analysis** tabs."
        )
