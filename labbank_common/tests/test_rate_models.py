"""Tests for the shared client-rate model (labbank_common/rate_models.py)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from labbank_common import rate_models as rm  # noqa: E402


def _model(rows):
    df = pd.DataFrame(rows)
    df.insert(0, "report_date", "2026-06-30")
    return rm._normalise(df).set_index("product_code")


MODEL = _model([
    # deposit tariffs: product-level spread
    {"product_code": 8000, "beta": 0.6, "margin_pct": 0.12, "client_floor": 0.0},
    {"product_code": 6300, "beta": 0.1, "margin_pct": 0.0, "client_floor": 0.0, "client_cap": 0.001},
    # loan: margin_pct 0 -> contract margin applies
    {"product_code": 1100, "beta": 1.0, "margin_pct": 0.0, "index_floor": 0.0},
    # blank beta defaults to 1
    {"product_code": 3200, "beta": np.nan, "margin_pct": -0.75, "index_floor": 0.0},
])


def test_client_rate_formula_and_limits():
    assert rm.client_rate(0.04, 0.0012, 0.6) == pytest.approx(0.6 * 0.04 + 0.0012)
    # index floor applies before beta, client floor/cap after
    assert rm.client_rate(-0.01, 0.02, 1.0, index_floor=0.0) == pytest.approx(0.02)
    assert rm.client_rate(0.04, 0.0, 0.1, client_cap=0.001) == pytest.approx(0.001)
    assert rm.client_rate(0.001, -0.01, 1.0, client_floor=0.0) == pytest.approx(0.0)
    assert rm.client_rate(0.09, 0.0, 1.0, index_cap=0.05) == pytest.approx(0.05)


def test_client_rate_vectorised_with_nan_limits():
    out = rm.client_rate(np.array([0.03, 0.05]), np.array([0.02, np.nan]), np.array([1.0, 0.5]),
                         index_floor=np.array([np.nan, 0.0]), client_cap=np.array([0.045, np.nan]))
    np.testing.assert_allclose(out, [0.045, 0.025])


def test_product_spread_blank_or_zero_means_contract_margin():
    sp = rm.product_spread(MODEL)
    assert sp["8000"] == pytest.approx(0.0012)
    assert sp["3200"] == pytest.approx(-0.0075)
    assert np.isnan(sp["1100"]) and np.isnan(sp["6300"])


def test_rate_maps_match_nii_eve_conventions():
    caps, floors, coeff_a, coeff_b = rm.rate_maps(MODEL)
    assert caps == {"6300": 0.001}
    assert floors == {"8000": 0.0, "6300": 0.0}
    assert coeff_a == {"8000": 0.6, "6300": 0.1}        # beta == 1 omitted
    assert set(coeff_b) == {"8000", "3200"}             # only product-level spreads


def test_cf_params():
    p = rm.cf_params(MODEL)
    assert p[1100] == {"beta": 1.0, "spread": None, "index_floor": 0.0, "index_cap": None,
                       "client_floor": None, "client_cap": None}
    assert p[3200]["beta"] == 1.0 and p[3200]["spread"] == pytest.approx(-0.0075)


def test_normalise_rejects_bad_input():
    with pytest.raises(ValueError, match="not both"):
        _model([{"product_code": 1, "beta": 1, "index_floor": 0.0, "client_floor": 0.0}])
    with pytest.raises(ValueError, match="duplicate"):
        _model([{"product_code": 1, "beta": 1}, {"product_code": 1, "beta": 0.5}])


def test_shipped_excel_is_valid():
    df = rm.read_rate_excel()
    assert not df.empty and df["beta"].notna().all()


# ── beta-adjusted repricing gap ──────────────────────────────────────────────

def test_effective_beta_plain_floor_and_cap():
    from labbank_common.rate_models import effective_beta
    assert effective_beta(0.03, 0.0, 0.6, client_floor=0.0) == pytest.approx(0.6)
    # PLN 6300: beta 0.1 capped at 0.1% -> already at the cap at a 3.8% index
    assert effective_beta(0.038, 0.0, 0.1, client_floor=0.0, client_cap=0.001) == pytest.approx(0.0)
    # client floor binding (index far below zero) -> no sensitivity
    assert effective_beta(-0.02, 0.0, 0.5, client_floor=0.0) == pytest.approx(0.0)


def test_beta_adjusted_gap_nmd_vs_term():
    from labbank_common.rate_models import beta_adjusted_gap, is_nmd
    gap = np.array([100.0, 50.0, 0.0, 850.0])
    # NMDs follow their repricing profile like any other product (no first-bucket lump)
    np.testing.assert_allclose(beta_adjusted_gap(gap, 0.6), 0.6 * gap)
    np.testing.assert_allclose(beta_adjusted_gap(gap, 0.8), 0.8 * gap)
    assert is_nmd("current_account") and is_nmd("saving_account") and not is_nmd("term_deposit")


# ── shocked client rate = anchored change of the FULL formula ────────────────

def _shock_helper():
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                     "irrbb_calc", "python_code")))
    from nii_calc_objects import _apply_rt_limits
    return _apply_rt_limits


def test_shocked_rate_respects_index_floor_and_client_cap():
    f = _shock_helper()
    pcs = pd.Series(["1100", "6300"])
    floors = rm.LimitMap({"6300": 0.0}, {"1100": 0.0})          # loan: index floor 0
    caps = rm.LimitMap({"6300": 0.001})                          # account: capped at 0.1%
    a, b = {"6300": 0.1}, {"1100": 0.02}
    # loan: index -1% -> +1%, spread 2%: base 2% (floored index), shocked 3% (was 4%)
    # account: beta 0.1 capped at 0.1%: index 3.8% -> 1.3% stays at the cap (was 0%)
    out = f(np.array([0.01, 0.013]), pcs, caps, floors, a, b,
            base_eff_rate=np.array([0.02, 0.001]), base_fwd_rt=np.array([-0.01, 0.038]))
    np.testing.assert_allclose(out, [0.03, 0.001])
    # zero shock reproduces the base exactly, even off the formula (administered anchor)
    out0 = f(np.array([-0.01, 0.038]), pcs, caps, floors, a, b,
             base_eff_rate=np.array([0.0215, 0.0007]), base_fwd_rt=np.array([-0.01, 0.038]))
    np.testing.assert_allclose(out0, [0.0215, 0.0007])


def test_rate_maps_carry_index_limits():
    caps, floors, _, _ = rm.rate_maps(MODEL)
    assert isinstance(floors, rm.LimitMap) and floors.index.get("3200") == 0.0
    import pickle
    assert pickle.loads(pickle.dumps(floors)).index == floors.index
