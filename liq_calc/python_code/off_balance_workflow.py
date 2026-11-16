"""Off-balance report: what each facility and guarantee costs and earns, and how drawings react to rates.

Per line of schemat.commitments:
  exposure      limit, drawn, undrawn, utilisation
  capital       EAD = undrawn x CCF, RWA, capital at the bank's CET1 target, expected loss
  liquidity     LCR outflow, NSFR required stable funding
  income        commitment or guarantee fee on the undrawn amount, per year
  drawings      expected drawn amount at 1M / 3M / 6M / 12M, baseline and stress
  rate view     12M drawings under each regulatory rate scenario: the baseline draw x gamma, gamma from
                labbank_common/rate_behaviour.py (behaviour DRAW), capped at the undrawn amount

Runs only for a bank with extended_reports; writes results.off_balance, results.off_balance_rate and
runs/<bank>/liq_calc/off_balance_report.xlsx.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from labbank_common import catalogue  # noqa: E402
from labbank_common import commitments as cmt  # noqa: E402
from labbank_common import rate_behaviour as rb  # noqa: E402
from labbank_common.bank_config import get_bank, get_engine  # noqa: E402

TENORS = (1, 3, 6, 12)


def cum_draw(models: pd.DataFrame, model_id, scenario: str, month: int) -> float:
    """Cumulative drawn share of the undrawn amount at a month; 0 before the first tenor."""
    m = models[(models["drawdown_model_id"] == model_id) & (models["scenario"] == scenario)
               & (models["months"] <= month)]
    return float(m.sort_values("months")["cum_draw_pct"].iloc[-1]) if not m.empty else 0.0


def credit_terms(f: pd.Series, products: pd.DataFrame) -> tuple[float, float, float]:
    """(risk weight, PD, LGD) of a line: a linked line takes its drawn product's, as the register
    leaves them empty for it; a line without a drawn product has its own."""
    code = f["drawn_product_code"]
    if pd.notna(code) and str(code).strip():
        p = products.loc[int(float(code))]
        return float(p["rwa_weight"]), float(p["PD"]), float(p["LGD"])
    return float(f["rwa_weight"] or 0.0), float(f["PD"] or 0.0), float(f["LGD"] or 0.0)


def line_metrics(reg: pd.DataFrame, models: pd.DataFrame, cet1_target: float,
                 products: pd.DataFrame) -> pd.DataFrame:
    """products: bs_structure asset rows indexed by product_code."""
    rows = []
    for _, f in reg.iterrows():
        und = float(f["undrawn_amt"])
        ead = und * float(f["undrawn_ccf"])
        weight, pd_, lgd = credit_terms(f, products)
        rwa = ead * weight
        r = {"commitment_code": f["commitment_code"], "own_name": f["own_name"],
             "facility_type": f["facility_type"], "revocability": f["revocability"],
             "limit_amt": float(f["limit_amt"]), "drawn_amt": float(f["drawn_amt"]), "undrawn_amt": und,
             "utilisation": float(f["drawn_amt"]) / float(f["limit_amt"]) if f["limit_amt"] else 0.0,
             "ccf": float(f["undrawn_ccf"]), "ead": ead, "rwa": rwa, "capital": rwa * cet1_target,
             "expected_loss": ead * pd_ * lgd,
             "lcr_outflow": float(f["available_amt_30d"]) * float(f["undrawn_lcr"]),
             "nsfr_rsf": und * float(f["undrawn_rsf"]),
             "fee_income": und * float(f["commitment_fee_rate"] or 0.0)}
        for scen in ("baseline", "stress"):
            for m in TENORS:
                r[f"draw_{scen}_{m}m"] = und * cum_draw(models, f["drawdown_model_id"], scen, m)
        rows.append(r)
    return pd.DataFrame(rows)


def rate_view(reg: pd.DataFrame, models: pd.DataFrame, bank) -> pd.DataFrame:
    table = rb.gamma_table(bank)
    rows = []
    for scen, gs in table.items():
        for _, f in reg.iterrows():
            base = float(f["undrawn_amt"]) * cum_draw(models, f["drawdown_model_id"], "baseline", 12)
            g = gs.get(("DRAW", str(int(f["commitment_code"]))), 1.0)
            draw = min(base * g, float(f["undrawn_amt"]))
            rows.append({"scenario": scen, "commitment_code": f["commitment_code"], "own_name": f["own_name"],
                         "gamma": g, "draw_baseline_12m": base, "draw_scenario_12m": draw,
                         "delta_draw_12m": draw - base})
    return pd.DataFrame(rows)


def main() -> None:
    bank = get_bank()
    if not bank.extended_reports:
        print(f"{bank.bank_id}: no extended reports, off-balance report skipped")
        return
    eng = get_engine(bank)
    reg = cmt.read_register(eng)
    if reg.empty:
        print(f"{bank.bank_id}: no commitments")
        return
    path = bank.balance_dir() / "bank_data.xlsx"
    models = cmt.read_drawdown_models(path)
    fin = pd.read_excel(path, sheet_name="fin_data").set_index("parameter")["value"]
    bs = catalogue.read_bs_structure(path)
    products = bs[bs["bs_side"] == "A"].assign(
        product_code=lambda d: d["product_code"].astype(int)).drop_duplicates("product_code").set_index("product_code")
    lines = line_metrics(reg, models, float(fin["CET1"]), products)
    rate = rate_view(reg, models, bank)
    num = lines.select_dtypes(include=[np.number]).columns.drop(["commitment_code", "utilisation", "ccf"], errors="ignore")
    total = lines.groupby("facility_type")[list(num)].sum().reset_index()
    by_scen = rate.groupby("scenario", sort=False)[["draw_baseline_12m", "draw_scenario_12m", "delta_draw_12m"]].sum().reset_index()
    print(lines[["own_name", "limit_amt", "undrawn_amt", "ccf", "ead", "rwa", "lcr_outflow", "fee_income"]].to_string(index=False))
    print(by_scen.to_string(index=False))
    rd = pd.Timestamp(bank.report_date)
    from labbank_common import run_store
    run_store.replace_run(eng, lines, "off_balance", "results", run_store.current_run(bank))
    run_store.replace_run(eng, rate, "off_balance_rate", "results", run_store.current_run(bank))
    with pd.ExcelWriter(bank.output_dir("liq_calc") / "off_balance_report.xlsx") as w:
        lines.to_excel(w, sheet_name="by_line", index=False)
        total.to_excel(w, sheet_name="by_type", index=False)
        rate.to_excel(w, sheet_name="rate_view", index=False)
        by_scen.to_excel(w, sheet_name="rate_view_total", index=False)


if __name__ == "__main__":
    main()
