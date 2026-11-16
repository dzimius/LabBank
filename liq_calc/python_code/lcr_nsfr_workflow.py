"""LCR / NSFR Workflow
=====================
Run order: balance_generate → balance_gen_add_data → cash_flow_calc → THIS FILE

Produces
--------
1. results.lcr_nsfr  (SQL) — one row per (report_date, currency):
     hqla, outflows_30d, inflows_30d, net_outflows_30d, lcr, asf, rsf, nsfr

2. output/<bank>/lcr_nsfr_report.xlsx  — LCR and NSFR

Note: results.irrbb_report and output/<bank>/irrbb_report.xlsx are now written by
eba_sot_workflow.py (irrbb_calc) — run that after nii_calc_workflow + eve_calc_workflow.
"""
import os
import numpy as np
import pandas as pd

import sql_setup
from labbank_common.bank_config import get_bank, total_assets   # sql_setup puts the project root on sys.path
import lcr_nsfr_objects as liq_obj
from labbank_common import fx
from labbank_common import derivatives as deriv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

REPORT_DATE    = pd.Timestamp(get_bank().report_date)
TOTAL_ASSETS   = total_assets()   # the bank's own book size (bank_config.total_assets)

BS_STRUCT_PATH = get_bank().balance_dir() / "bank_data.xlsx"

# ── 1. Load balance-sheet structure ──────────────────────────────────────────
print(f"{REPORT_DATE.date()}  Loading bs_structure...")
from labbank_common import catalogue
bs_struct = catalogue.read_bs_structure(BS_STRUCT_PATH)

# ── 2. Load 30-day asset inflows from cf.products ────────────────────────────
print("Loading 30-day asset inflows from cf.products...")
from labbank_common import lcr_rules
_no_inflow = lcr_rules.non_inflow_codes(bs_struct)       # HQLA (already in the buffer) + cards
print(f"  not counted as inflows: {_no_inflow}")
# contractual receipts per product x counterparty inflow rate (financial 100%, else 50%)
_infl = sql_setup.load_30d_asset_inflows(REPORT_DATE, _no_inflow)
_ctype = lcr_rules.product_client_types(bs_struct)
_infl["rate"] = [lcr_rules.inflow_rate(_ctype.get((pc, "A"))) for pc in _infl["product_code"]]
_infl["recognised"] = _infl["inflow_amt"] * _infl["rate"]
print(_infl.to_string(index=False))
inflows_30d = _infl.groupby("currency")["recognised"].sum().to_dict()

# HQLA at market value (LCR DA Art. 9): market value / book per HQLA product
HQLA_MV = sql_setup.load_hqla_market_value_ratio(REPORT_DATE, lcr_rules.hqla_product_codes(bs_struct),
                                                lcr_rules.market_spreads_bp(bs_struct))
print(f"  HQLA market value / book: { {k: round(v, 4) for k, v in HQLA_MV.items()} }")

# ── 2b. Load deposit maturity split (within 30d → 100% run-off) ───────────────
print("Loading deposit maturity split from schemat.deposits...")
try:
    td_split = sql_setup.load_td_maturity_split(REPORT_DATE)
    print(f"  Deposit maturity split:\n{td_split.to_string(index=False)}")
except Exception as e:
    print(f"  Warning: could not load deposit maturity split ({e}). Using bs_struct weights only.")
    td_split = None

# ── 2c. Home-currency amounts ─────────────────────────────────────────────────
# bs_structure balances are already home-currency shares; the cash-flow inflows and the
# deposit maturity split come from SQL in each currency's own units -> convert to home.
TO_HOME = fx.to_home(get_bank(), REPORT_DATE)
inflows_30d = {c: v * TO_HOME[c] for c, v in inflows_30d.items()}
if td_split is not None and not td_split.empty:
    _f = fx.to_home_series(td_split["currency"], get_bank(), REPORT_DATE)
    td_split = td_split.assign(within_30d_amt=td_split["within_30d_amt"] * _f,
                               after_30d_amt=td_split["after_30d_amt"] * _f)

# FX deals (schemat.fx_deals): exchanges settling within 30 days, per currency gross --
# receipts are inflows, payments outflows. In the all-currency LCR they are netted (CRR
# LCR DA Art. 21: derivative flows are taken net where exchanged simultaneously).
fx_in, fx_out, fx_net_home = {}, {}, 0.0
try:
    _deals = pd.read_sql("SELECT * FROM schemat.fx_deals", sql_setup.engine)
except Exception:
    _deals = pd.DataFrame()
if not _deals.empty:
    for _c in ("near_date", "far_date", "trade_date"):
        _deals[_c] = pd.to_datetime(_deals[_c])
    _ex = deriv.fx_exchanges(_deals)
    _ex = _ex[(_ex["date"] > REPORT_DATE) & (_ex["date"] <= REPORT_DATE + pd.Timedelta(days=30))]
    for _, _r in _ex.iterrows():
        _v = _r["amount"] * TO_HOME[_r["currency"]]
        (fx_in if _v > 0 else fx_out)[_r["currency"]] = (fx_in if _v > 0 else fx_out).get(_r["currency"], 0.0) + abs(_v)
        fx_net_home += _v
    print(f"  FX deals settling within 30d (home ccy): in {fx_in}  out {fx_out}  net {fx_net_home:,.0f}")
inflows_30d = {c: inflows_30d.get(c, 0.0) + fx_in.get(c, 0.0) for c in set(inflows_30d) | set(fx_in)}

# Committed facilities (schemat.commitments): undrawn amounts, home currency
com_out, com_rsf = {}, {}
from labbank_common import commitments as cmt
_com = cmt.read_register(sql_setup.engine)      # empty only when the bank has no facilities
if not _com.empty:
    _f = _com["currency"].map(TO_HOME)
    _com["lcr_out_home"] = _com["available_amt_30d"] * _com["undrawn_lcr"] * _f
    _com["rsf_home"]     = _com["undrawn_amt"] * _com["undrawn_rsf"] * _f
    com_out = _com.groupby("currency")["lcr_out_home"].sum().to_dict()
    com_rsf = _com.groupby("currency")["rsf_home"].sum().to_dict()
    print("  Committed facilities (home ccy):")
    print(_com[["commitment_code", "own_name", "currency", "limit_amt", "drawn_amt", "undrawn_amt",
                "lcr_out_home", "rsf_home"]].to_string(index=False))

# A multi-currency bank also gets currency = 'ALL': the consolidated ratio over every
# currency in home-currency terms (the regulatory minimum applies to this one; per-currency
# LCRs show the currency mismatch, e.g. USD deposits without USD HQLA).
MULTI_CCY = bs_struct["currency"].nunique() > 1


def _all_ccy(bs: pd.DataFrame) -> pd.DataFrame:
    return bs.assign(currency="ALL")


# ── 3. Compute LCR ───────────────────────────────────────────────────────────
print("Computing LCR...")
lcr_df = liq_obj.compute_lcr(bs_struct, TOTAL_ASSETS, inflows_30d, td_split, fx_out, com_out,
                             hqla_mv_ratio=HQLA_MV)
if MULTI_CCY:
    _split_all = (None if td_split is None or td_split.empty else
                  td_split.assign(currency="ALL").groupby(["currency", "product_code"], as_index=False)
                  .sum(numeric_only=True))
    _in_all = sum(inflows_30d.values()) - sum(fx_in.values()) + max(fx_net_home, 0.0)
    lcr_df = pd.concat([lcr_df, liq_obj.compute_lcr(_all_ccy(bs_struct), TOTAL_ASSETS,
                                                    {"ALL": _in_all}, _split_all,
                                                    {"ALL": max(-fx_net_home, 0.0)},
                                                    {"ALL": sum(com_out.values())},
                                                    hqla_mv_ratio=HQLA_MV)],
                       ignore_index=True)
for _, r in lcr_df.iterrows():
    print(
        f"  {r['currency']}  HQLA={r['hqla']:,.0f}  "
        f"Outflows={r['outflows_30d']:,.0f}  "
        f"Inflows={r['inflows_30d']:,.0f}  "
        f"Net={r['net_outflows_30d']:,.0f}  "
        + (f"LCR={r['lcr']:.2%}" if pd.notna(r['lcr']) else "LCR=N/A")
    )

# ── 4. Compute NSFR ──────────────────────────────────────────────────────────
print("Computing NSFR...")
nsfr_df = liq_obj.compute_nsfr(bs_struct, TOTAL_ASSETS, com_rsf)
if MULTI_CCY:
    nsfr_df = pd.concat([nsfr_df, liq_obj.compute_nsfr(_all_ccy(bs_struct), TOTAL_ASSETS,
                                                       {"ALL": sum(com_rsf.values())})], ignore_index=True)
for _, r in nsfr_df.iterrows():
    print(
        f"  {r['currency']}  ASF={r['asf']:,.0f}  RSF={r['rsf']:,.0f}  "
        + (f"NSFR={r['nsfr']:.2%}" if pd.notna(r['nsfr']) else "NSFR=N/A")
    )

# ── 5. Write LCR / NSFR to SQL ───────────────────────────────────────────────
print("Writing results.lcr_nsfr to SQL...")
liq_summary = lcr_df.merge(nsfr_df, on="currency", how="outer")
liq_summary.insert(0, "report_date", REPORT_DATE)

sql_setup.reset_lcr_nsfr()
sql_setup.write_lcr_nsfr(liq_summary)
print(f"  Written {len(liq_summary)} rows to results.lcr_nsfr.")

# ── 5b. Projected facility drawdowns (baseline + stress), by currency ──────────
# Drawings are cash out from the undrawn amount, repaid later (drawdown_models sheet):
# the behavioural / stressed liquidity view of the commitments; LCR above uses the
# regulatory factor instead (a different input on purpose).
draw_df = pd.DataFrame()
sql_setup.run_store.reset_table(sql_setup.engine, "commitment_drawdowns", "results")   # this run's rows only
if not _com.empty:
    _models = cmt.read_drawdown_models(BS_STRUCT_PATH)
    _com["availability_end_date"] = pd.to_datetime(_com["availability_end_date"])
    draw_df = cmt.drawdown_flows(_com, _models, REPORT_DATE)
    if not draw_df.empty:
        draw_df["amount_home"] = draw_df["amount"] * draw_df["currency"].map(TO_HOME)
        draw_df.insert(0, "report_date", REPORT_DATE)
        sql_setup.run_store.append(sql_setup.engine, draw_df, "commitment_drawdowns", "results")
        print("  Projected drawdowns (home ccy, net per scenario, 12M):")
        print(draw_df[draw_df["month"] <= 12].groupby(["scenario", "flow_type"])["amount_home"].sum().to_string())

# ── 6. Write LCR / NSFR to Excel ─────────────────────────────────────────────
lcr_nsfr_path = str(get_bank().output_dir("liq_calc") / "lcr_nsfr_report.xlsx")
print(f"Writing LCR/NSFR Excel to {lcr_nsfr_path}...")
with pd.ExcelWriter(lcr_nsfr_path, engine="openpyxl") as writer:
    liq_summary.to_excel(writer, sheet_name="LCR_NSFR", index=False)
    if not _com.empty:
        _com.to_excel(writer, sheet_name="Commitments", index=False)
    if not draw_df.empty:
        draw_df.to_excel(writer, sheet_name="Drawdowns", index=False)
print(f"  Written to {lcr_nsfr_path}")

# ── 7. Pillar 2 liquidity: buffer, funding concentration, top depositors, survival ──
print("Pillar 2 liquidity metrics...")
import liquidity_p2 as p2

# 7a. Liquidity buffer = LCR HQLA per product (all currencies, home ccy)
hqla_df = p2.hqla_breakdown(bs_struct, TOTAL_ASSETS, HQLA_MV)
BUFFER = float(hqla_df["buffer_value"].sum())

# 7b. Funding sources (liabilities without equity), home ccy
_dep = sql_setup.load_deposits_by_client(REPORT_DATE)
_dep["balance_home"] = _dep["balance_amt"].astype(float) * _dep["currency"].map(TO_HOME)
_debt = sql_setup.load_issued_debt(REPORT_DATE)
_debt["balance_home"] = _debt["balance_amt"].astype(float) * _debt["currency"].map(TO_HOME)
_ctype_l = {pc: ct for (pc, side), ct in lcr_rules.product_client_types(bs_struct).items() if side == "L"}
_debt["client_type_id"] = _debt["product_code"].map(_ctype_l).fillna(8).astype(int)
funding = pd.concat([
    _dep.groupby(["product_code", "product_name", "client_type_id", "currency"], as_index=False)["balance_home"].sum(),
    _debt[["product_code", "product_name", "client_type_id", "currency", "balance_home"]]],
    ignore_index=True)
TOTAL_LIAB = float(funding["balance_home"].sum())
conc_df, hhi_df = p2.funding_concentration(funding)
top_df = p2.top_depositors(_dep, TOTAL_LIAB, n=10)

# 7c. Survival horizon -- base scenario (behavioural run-off, no rollover, no new business)
flows = sql_setup.load_behavioural_flows(REPORT_DATE, lcr_rules.hqla_product_codes(bs_struct))
_sign = flows["bs_side"].map({"A": 1.0, "L": -1.0})
_fx = flows["currency"].map(TO_HOME)
flows["amount_home"] = flows["total_pmt"].astype(float) * _sign * _fx
flows["principal_home"] = flows["principal_pmt"].astype(float) * _fx
flows["date"] = flows["cf_end_dt"]
_extra = []
if not draw_df.empty:            # committed facilities, baseline drawing model
    _d = draw_df[draw_df["scenario"] == "baseline"]
    _extra.append(pd.DataFrame({
        "date": [REPORT_DATE + pd.offsets.MonthEnd(int(m)) for m in _d["month"]],
        "amount_home": _d["amount_home"].astype(float).values, "source": "commitments"}))
if not _deals.empty:             # FX deal exchanges still to settle
    _ex = deriv.fx_exchanges(_deals)
    _extra.append(pd.DataFrame({"date": pd.to_datetime(_ex["date"]),
                                "amount_home": _ex["amount"] * _ex["currency"].map(TO_HOME),
                                "source": "fx_deals"}))
surv_flows = pd.concat([flows.assign(source="book")[["date", "amount_home", "source"]], *_extra],
                       ignore_index=True)
prof = p2.survival_profile(surv_flows, BUFFER, REPORT_DATE)
SURV = p2.survival_days(prof)
points = p2.profile_points(prof)
_wh_codes = {pc for pc, ct in _ctype_l.items() if ct in p2.WHOLESALE_TYPES}
wh_mat = p2.wholesale_maturities(flows[flows["bs_side"] == "L"], _wh_codes, REPORT_DATE)

_lcr_all = lcr_df.loc[lcr_df["currency"] == ("ALL" if MULTI_CCY else bs_struct["currency"].iloc[0])].iloc[0]
summary = pd.DataFrame([
    ("liquidity_buffer",          BUFFER),
    ("buffer_pct_total_assets",   BUFFER / TOTAL_ASSETS),
    ("buffer_pct_deposits",       BUFFER / float(_dep["balance_home"].sum())),
    ("net_outflows_30d_lcr",      float(_lcr_all["net_outflows_30d"])),
    ("total_liabilities",         TOTAL_LIAB),
    ("total_deposits",            float(_dep["balance_home"].sum())),
    ("wholesale_funding_share",   float(funding.loc[funding["client_type_id"].isin(p2.WHOLESALE_TYPES), "balance_home"].sum()) / TOTAL_LIAB),
    ("hhi_counterparty",          float(hhi_df.set_index("dimension").loc["counterparty", "hhi"])),
    ("hhi_product",               float(hhi_df.set_index("dimension").loc["product", "hhi"])),
    ("top10_share_deposits",      float(top_df["share_deposits"].sum())),
    ("top10_share_liabilities",   float(top_df["share_liabilities"].sum())),
    ("n_counterparties_above_1pct", float(top_df["above_almm_1pct"].sum())),
    ("survival_days_base",        float(SURV) if SURV is not None else np.nan),
    ("survival_horizon_checked",  float(prof["day"].iloc[-1])),
    ("min_position_base",         float(prof["position"].min())),
], columns=["metric", "value"])
summary.insert(0, "report_date", REPORT_DATE)
summary.insert(1, "scenario", "base")
print(summary.to_string(index=False))
print(f"  Survival horizon (base): "
      + (f"{SURV} days" if SURV is not None else f"> {int(prof['day'].iloc[-1])} days (never negative)"))

_out = {"liq_p2_summary": summary,
        "liq_p2_buffer": hqla_df.assign(report_date=REPORT_DATE),
        "liq_p2_concentration": conc_df.assign(report_date=REPORT_DATE),
        "liq_p2_top_depositors": top_df.assign(report_date=REPORT_DATE),
        "liq_p2_survival": prof.assign(report_date=REPORT_DATE, scenario="base"),
        "liq_p2_wholesale_maturity": wh_mat.assign(report_date=REPORT_DATE)}
for _name, _df in _out.items():
    sql_setup.run_store.replace_run(sql_setup.engine, _df, _name, "results")
print(f"  Written {', '.join('results.' + n for n in _out)}")

p2_path = str(get_bank().output_dir("liq_calc") / "liquidity_p2_report.xlsx")
with pd.ExcelWriter(p2_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    hqla_df.to_excel(writer, sheet_name="Buffer", index=False)
    conc_df.to_excel(writer, sheet_name="Concentration", index=False)
    hhi_df.to_excel(writer, sheet_name="HHI", index=False)
    top_df.to_excel(writer, sheet_name="Top10_depositors", index=False)
    points.to_excel(writer, sheet_name="Survival_points", index=False)
    prof.to_excel(writer, sheet_name="Survival_daily", index=False)
    wh_mat.to_excel(writer, sheet_name="Wholesale_maturity", index=False)
print(f"  Written to {p2_path}")

print("Done.")
