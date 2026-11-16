"""Tests for the EBA scenario behavioural multipliers (labbank_common/irrbb_behaviour.py)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from labbank_common import irrbb_behaviour as ib  # noqa: E402

EBA_SCENARIOS = ["par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn"]


def test_cpr_multipliers_follow_rts_art_9_3():
    # rates up -> fewer prepayments; rates down -> more
    assert {s: ib.CPR_GAMMA[s] for s in EBA_SCENARIOS} == {
        "par_up": 0.8, "par_dn": 1.2, "steep": 0.8, "flat": 1.2, "sr_up": 0.8, "sr_dn": 1.2,
    }


def test_tdrr_multipliers_follow_rts_art_10_5():
    # short rates up -> more early redemptions; steepener cuts short rates
    assert {s: ib.TDRR_GAMMA[s] for s in EBA_SCENARIOS} == {
        "par_up": 1.2, "par_dn": 0.8, "steep": 0.8, "flat": 1.2, "sr_up": 1.2, "sr_dn": 0.8,
    }


def test_sensitivity_and_base_scenarios_keep_baseline():
    for s in ("base", "own_1_up", "own_1_dn", "anything_else"):
        assert ib.gammas_for(s) == (1.0, 1.0)


def test_every_scenario_gamma_has_a_precomputed_variant():
    used = set(ib.CPR_GAMMA.values()) | set(ib.TDRR_GAMMA.values())
    assert used <= set(ib.VARIANT_GAMMAS)


@pytest.mark.parametrize("gamma", [0.8, 1.0, 1.2])
def test_scale_period_cpr_scales_the_annual_rate(gamma):
    p, yf = 0.005, 31 / 365
    annual = 1 - (1 - p) ** (1 / yf)
    q = float(ib.scale_period_cpr(p, yf, gamma))
    assert 1 - (1 - q) ** (1 / yf) == pytest.approx(gamma * annual)


def test_scale_period_cpr_caps_at_full_prepayment_and_handles_zero_yf():
    assert float(ib.scale_period_cpr(0.9, 1.0, 1.2)) == pytest.approx(1.0)
    assert float(ib.scale_period_cpr(0.01, 0.0, 1.2)) == pytest.approx(0.01)


def test_scale_tdrr_caps_at_one():
    assert ib.scale_tdrr(0.1, 1.2) == pytest.approx(0.12)
    assert ib.scale_tdrr(0.9, 1.2) == 1.0


def _rows(sid, ptype, amounts, **extra):
    return pd.DataFrame({"schedule_id": sid, "product_type": ptype, "amt": amounts, **extra})


def test_apply_variant_swaps_in_scope_schedules_only():
    base = pd.concat([
        _rows("1", "L", [100.0, 100.0]),        # fixed loan, in scope
        _rows("1", "D", [50.0]),                # same id, other table: untouched by loan gamma
        _rows("2", "L", [70.0]),                # floating loan, not in scope
        _rows("9", "D", [30.0, 30.0]),          # term deposit, in scope
        _rows("S1", "S", [5.0]),                # swap leg
    ], ignore_index=True)
    scen = pd.concat([
        _rows("1", "L", [80.0], gamma=0.8), _rows("1", "L", [120.0], gamma=1.2),
        _rows("9", "D", [24.0, 36.0], gamma=0.8), _rows("9", "D", [36.0, 24.0], gamma=1.2),
    ], ignore_index=True)

    up = ib.apply_variant(base, scen, "par_up", enabled=True)       # CPR 0.8, TDRR 1.2
    assert "gamma" not in up.columns
    assert up.loc[(up.schedule_id == "1") & (up.product_type == "L"), "amt"].tolist() == [80.0]
    assert up.loc[up.schedule_id == "9", "amt"].tolist() == [36.0, 24.0]
    assert up.loc[(up.schedule_id == "1") & (up.product_type == "D"), "amt"].tolist() == [50.0]
    assert up.loc[up.schedule_id == "2", "amt"].tolist() == [70.0]
    assert up.loc[up.schedule_id == "S1", "amt"].tolist() == [5.0]

    steep = ib.apply_variant(base, scen, "steep", enabled=True)     # CPR 0.8, TDRR 0.8
    assert steep.loc[steep.schedule_id == "9", "amt"].tolist() == [24.0, 36.0]

    # baseline scenarios and a missing variant table leave the base rows as they are
    assert ib.apply_variant(base, scen, "own_1_up", enabled=True) is base
    assert ib.apply_variant(base, pd.DataFrame(), "par_up", enabled=True) is base

    # switched off: baseline behaviour in every scenario
    assert ib.apply_variant(base, scen, "par_up", enabled=False) is base


def test_switch_is_off_by_default_and_apply_variant_follows_it(monkeypatch):
    assert ib.APPLY_EBA_BEHAVIOUR is False
    base = _rows("1", "L", [100.0])
    scen = _rows("1", "L", [80.0], gamma=0.8)
    assert ib.apply_variant(base, scen, "par_up") is base
    monkeypatch.setattr(ib, "APPLY_EBA_BEHAVIOUR", True)
    assert ib.apply_variant(base, scen, "par_up")["amt"].tolist() == [80.0]
