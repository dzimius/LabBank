"""Tests for the off-balance report and the CSRBB arithmetic."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ob = _load("liq_calc/python_code/off_balance_workflow.py", "off_balance_workflow")
cs = _load("irrbb_calc/python_code/csrbb_workflow.py", "csrbb_workflow")

MODELS = pd.DataFrame({"drawdown_model_id": ["m"] * 2, "scenario": ["baseline"] * 2,
                       "months": [1, 12], "cum_draw_pct": [0.1, 0.5]})
PRODUCTS = pd.DataFrame({"rwa_weight": [0.75], "PD": [0.02], "LGD": [0.5]}, index=pd.Index([4100], name="product_code"))


def _line(**kw):
    return pd.Series({"commitment_code": 9000, "own_name": "x", "facility_type": "credit",
                      "revocability": "irrevocable", "limit_amt": 100.0, "drawn_amt": 60.0, "undrawn_amt": 40.0,
                      "undrawn_ccf": 0.5, "rwa_weight": 0.0, "PD": 0.0, "LGD": 0.0, "available_amt_30d": 40.0,
                      "undrawn_lcr": 0.1, "undrawn_rsf": 0.05, "commitment_fee_rate": 0.01,
                      "drawdown_model_id": "m", "drawn_product_code": None, **kw})


def test_linked_line_takes_the_risk_terms_of_its_drawn_product():
    m = ob.line_metrics(pd.DataFrame([_line(drawn_product_code="4100")]), MODELS, 0.12, PRODUCTS).iloc[0]
    assert m["ead"] == pytest.approx(20.0)
    assert m["rwa"] == pytest.approx(15.0)
    assert m["expected_loss"] == pytest.approx(20.0 * 0.02 * 0.5)
    assert m["capital"] == pytest.approx(15.0 * 0.12)


def test_standalone_line_keeps_its_own_risk_terms():
    m = ob.line_metrics(pd.DataFrame([_line(rwa_weight=1.0, PD=0.01, LGD=0.4)]), MODELS, 0.12, PRODUCTS).iloc[0]
    assert m["rwa"] == pytest.approx(20.0)
    assert m["fee_income"] == pytest.approx(0.4)
    assert m["lcr_outflow"] == pytest.approx(4.0)


def test_cumulative_draw_steps_on_the_tenors():
    assert ob.cum_draw(MODELS, "m", "baseline", 0) == 0.0
    assert ob.cum_draw(MODELS, "m", "baseline", 6) == 0.1
    assert ob.cum_draw(MODELS, "m", "baseline", 12) == 0.5


def test_spread_widening_lowers_value_and_zero_changes_nothing():
    cf, df, t = np.array([5.0, 105.0]), np.array([0.98, 0.95]), np.array([1.0, 2.0])
    base = cs.pv(cf, df, t, 50.0)
    assert cs.pv(cf, df, t, 50.0) - base == 0.0
    assert cs.pv(cf, df, t, 150.0) < base


def test_only_principal_inside_12m_earns_the_wider_spread():
    assert cs.reinvestment_gain(np.array([100.0, 100.0]), np.array([0.5, 2.0]), 100.0) == pytest.approx(0.5)


def test_fvoci_hits_capital_and_amortised_cost_does_not():
    cf = pd.DataFrame({"total": [105.0], "capital": [100.0], "d_f": [0.95], "t": [2.0]})
    shocks = pd.DataFrame({"cqs": [3], "moderate_bp": [100.0], "severe_bp": [200.0]})
    spec = pd.Series({"product_code": 3310, "own_name": "b", "cqs": 3, "mv_spread_bp": 0.0, "accounting": "FVOCI"})
    a = cs.product_rows(cf, spec, shocks)
    b = cs.product_rows(cf, spec.copy().rename(None).pipe(lambda s: s.where(s.index != "accounting", "AC")), shocks)
    assert a[0]["delta_cet1"] == a[0]["delta_ev"] < 0
    assert b[0]["delta_cet1"] == 0.0
    assert a[1]["delta_ev"] < a[0]["delta_ev"]
    with pytest.raises(ValueError):
        cs.product_rows(cf, spec.copy().where(spec.index != "cqs", 9), shocks)
