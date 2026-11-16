"""Tests for the rate-dependent multipliers (labbank_common/rate_behaviour.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from labbank_common import rate_behaviour as rb  # noqa: E402

CPR = rb.Sensitivity("1000", "CPR", "10Y", -1, 0.6, 1.4, 200.0)
TD = rb.Sensitivity("7060", "TDRR", "6M", 1, 0.6, 1.4, 200.0)


def test_zero_shock_reproduces_baseline():
    assert rb.gamma(0.0, CPR) == 1.0
    assert rb.gamma(0.0, rb.Sensitivity("1", "CPR", "1Y", 1, 0.5, 2.0, 100.0, 25.0)) == 1.0
    assert rb.on_grid(rb.gamma(0.0, TD)) == 1.0


def test_eba_anchor_at_the_scale_point():
    assert rb.gamma(-200.0, CPR) == pytest.approx(1.2)
    assert rb.gamma(200.0, CPR) == pytest.approx(0.8)
    assert rb.gamma(200.0, TD) == pytest.approx(1.2)
    assert rb.gamma(-200.0, TD) == pytest.approx(0.8)


def test_smaller_move_gives_a_smaller_multiplier():
    assert 0.8 < rb.gamma(90.0, CPR) < 1.0


def test_monotonic_and_bounded():
    d = np.linspace(-3000, 3000, 241)
    g = np.array([rb.gamma(x, CPR) for x in d])
    assert (np.diff(g) <= 0).all()
    assert g.min() > 0.6 and g.max() < 1.4


def test_asymmetric_limits_and_deadband():
    s = rb.Sensitivity("1000", "CPR", "10Y", -1, 0.8, 2.0, 100.0, 50.0)
    assert rb.gamma(-40.0, s) == 1.0 and rb.gamma(40.0, s) == 1.0
    assert rb.gamma(-150.0, s) - 1.0 > 1.0 - rb.gamma(150.0, s)


def test_survival_scaling_keeps_the_end_points():
    s = rb.scale_survival([1.0, 0.64, 0.0], 0.5)
    assert s.tolist() == pytest.approx([1.0, 0.8, 0.0])
    assert rb.scale_survival([1.0, 0.7], 1.0).tolist() == pytest.approx([1.0, 0.7])


@pytest.mark.parametrize("kw", [dict(direction=0), dict(gamma_min=1.1), dict(gamma_max=0.9),
                                dict(scale_bp=0.0), dict(behaviour="X"), dict(ref_tenor="ten")])
def test_bad_inputs_are_rejected(kw):
    base = dict(product_code="1", behaviour="CPR", ref_tenor="1Y", direction=1, gamma_min=0.5,
                gamma_max=1.5, scale_bp=100.0)
    with pytest.raises(ValueError):
        rb.Sensitivity(**{**base, **kw})


def test_apply_variant_uses_the_multiplier_of_each_product(monkeypatch):
    import pandas as pd
    from labbank_common import irrbb_behaviour as ib
    base = pd.DataFrame({"schedule_id": [1, 2, 3], "product_type": ["L", "L", "L"],
                         "product_code": ["1000", "2000", "4000"], "amt": [100.0, 100.0, 100.0]})
    scen = pd.DataFrame({"schedule_id": [1, 1, 2, 2], "product_type": ["L"] * 4,
                         "product_code": ["1000", "1000", "2000", "2000"],
                         "gamma": [0.8, 0.9, 0.8, 0.9], "amt": [80.0, 90.0, 81.0, 91.0]})
    monkeypatch.setenv("LABBANK_BEHAVIOUR", "rate")
    monkeypatch.setattr(rb, "scenario_gammas", lambda s, bank=None: {("L", "1000"): 0.8, ("L", "2000"): 0.9})
    out = ib.apply_variant(base, scen, "par_up").sort_values("schedule_id")
    assert out["amt"].tolist() == [80.0, 91.0, 100.0]
    monkeypatch.setattr(rb, "scenario_gammas", lambda s, bank=None: {})
    assert ib.apply_variant(base, scen, "own_1_up") is base


def test_nmd_multipliers_keep_a_finer_grid():
    assert rb.on_grid(1.1187, "NMD_CORE") == 1.12
    assert rb.on_grid(1.1187, "CPR") == 1.1


def test_eba_mode_leaves_nmd_variants_alone(monkeypatch):
    import pandas as pd
    from labbank_common import irrbb_behaviour as ib
    base = pd.DataFrame({"schedule_id": [1, 2], "product_type": ["D", "D"],
                         "product_code": ["7060", "6000"], "amt": [100.0, 100.0]})
    scen = pd.DataFrame({"schedule_id": [1, 2], "product_type": ["D", "D"], "product_code": ["7060", "6000"],
                         "gamma": [1.2, 1.2], "amt": [120.0, 120.0]})
    monkeypatch.setenv("LABBANK_BEHAVIOUR", "eba")
    monkeypatch.setattr(rb, "nmd_codes", lambda bank=None: {"6000"})
    assert ib.apply_variant(base, scen, "par_up").sort_values("schedule_id")["amt"].tolist() == [120.0, 100.0]


def test_a_run_can_be_limited_to_some_behaviours(monkeypatch):
    table = {"par_up": {("CPR", "1000"): 0.8, ("NMD_CORE", "6000"): 1.2}}
    monkeypatch.setattr(rb, "gamma_table", lambda bank=None: table)
    assert rb.scenario_gammas("par_up") == {("L", "1000"): 0.8, ("D", "6000"): 1.2}
    monkeypatch.setenv("LABBANK_RATE_BEHAVIOURS", "CPR")
    assert rb.scenario_gammas("par_up") == {("L", "1000"): 0.8}


def test_modes_switch_multipliers_and_baseline_redemption(monkeypatch):
    from labbank_common import irrbb_behaviour as ib
    for mode, multipliers, redemption in (("baseline", False, False), ("flat", False, True),
                                          ("eba", True, True), ("rate", True, True)):
        monkeypatch.setenv("LABBANK_BEHAVIOUR", mode)
        assert ib.scenario_multipliers_on() is multipliers
        assert ib.baseline_redemption_on() is redemption


def test_scenario_shock_shapes():
    assert rb.scenario_shock_bp("par_up", "10Y", "EUR") == pytest.approx(200.0)
    assert rb.scenario_shock_bp("par_dn", "10Y", "EUR") == pytest.approx(-200.0)
    assert rb.scenario_shock_bp("steep", "10Y", "EUR") > 0 > rb.scenario_shock_bp("steep", "3M", "EUR")
