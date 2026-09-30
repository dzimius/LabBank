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
