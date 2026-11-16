import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
import labbank_common.lcr_rules as lr  # noqa: E402
from labbank_common.lcr_rules import hqla_product_codes, non_inflow_codes  # noqa: E402


def test_hqla_assets_and_cards_are_not_inflows():
    bs = pd.DataFrame({"bs_side": ["A", "A", "A", "L"], "product_code": [3200, 1100, 3000, 6000],
                       "hqla_class": [1.0, None, 1.0, None]})
    assert hqla_product_codes(bs) == {"3000", "3200"}
    assert non_inflow_codes(bs) == ["2200", "2210", "3000", "3200"]     # mortgages still inflow


def test_hqla_market_value_ratio_keeps_cash_at_book(monkeypatch):
    import labbank_common.lcr_rules as lr
    monkeypatch.setattr(lr.pd, "read_sql_query", lambda *a, **k: pd.DataFrame(
        {"product_code": ["3000", "3500"], "ratio": [0.99, 0.0]}))
    assert lr.hqla_market_value_ratio(None, "2026-06-30", ["3000", "3500"]) == {"3000": 0.99}


def test_hqla_level_reads_numbers_and_labels():
    assert [lr.hqla_level(c) for c in (1.0, "1", 2, "2A", 3.0, "2b")] == ["1", "1", "2A", "2A", "2B", "2B"]
    with pytest.raises(ValueError):
        lr.hqla_level("4")


def test_hqla_caps_bind_only_above_the_limits():
    assert lr.capped_hqla(100.0, 10.0, 5.0) == pytest.approx(115.0)
    assert lr.capped_hqla(100.0, 0.0, 0.0) == pytest.approx(100.0)
    assert lr.capped_hqla(60.0, 40.0, 0.0) == pytest.approx(100.0)        # Level 2 exactly 40%
    assert lr.capped_hqla(60.0, 100.0, 0.0) == pytest.approx(100.0)       # 40% cap
    assert lr.capped_hqla(85.0, 0.0, 100.0) == pytest.approx(100.0)       # 2B at 15%
