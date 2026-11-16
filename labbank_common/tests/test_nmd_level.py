"""NMD core profile by rate level (rate_behaviour NMD_LEVEL) and its roll-forward hook."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labbank_common import rate_behaviour as rb

ROOT = Path(__file__).resolve().parents[2]
LEVEL = rb.Sensitivity("6000", "NMD_LEVEL", "3M", 1, 0.79, 1.21, 500.0, 0.0, 2.6)
PROFILE = pd.DataFrame({"report_date": "2026-09-30", "tenor": ["1D", "12M", "24M", "36M"],
                        6000: [0.9, 0.66, 0.54, np.nan], 8000: [0.585, 0.273, 0.195, 0.117]})


def test_same_level_keeps_the_profile():
    out, used = rb.roll_nmd_profile(PROFILE, [LEVEL], 2.6, 2.6)
    assert used == {"6000": pytest.approx(1.0)}
    pd.testing.assert_frame_equal(out, PROFILE)


def test_higher_level_shortens_and_lower_lengthens():
    up, _ = rb.roll_nmd_profile(PROFILE, [LEVEL], 2.6, 4.0)
    dn, _ = rb.roll_nmd_profile(PROFILE, [LEVEL], 2.6, 1.0)
    assert (up[6000].iloc[:3] < PROFILE[6000].iloc[:3]).all()
    assert (dn[6000].iloc[:3] > PROFILE[6000].iloc[:3]).all()
    assert np.isnan(up[6000].iloc[3])
    pd.testing.assert_series_equal(up[8000], PROFILE[8000])        # no NMD_LEVEL row: unchanged


def test_rolling_there_and_back_returns_the_profile():
    there, _ = rb.roll_nmd_profile(PROFILE, [LEVEL], 2.6, 4.0)
    back, _ = rb.roll_nmd_profile(there, [LEVEL], 4.0, 2.6)
    np.testing.assert_allclose(back[6000].iloc[:3], PROFILE[6000].iloc[:3])


def test_level_rows_stay_out_of_scenarios_and_need_a_reference():
    with pytest.raises(ValueError):
        rb.Sensitivity("6000", "NMD_LEVEL", "3M", 1, 0.8, 1.2, 500.0)
    df = pd.DataFrame([dict(product_code=6000, behaviour="NMD_LEVEL", ref_tenor="3M", direction=1, gamma_min=0.8,
                            gamma_max=1.2, scale_bp=500.0, deadband_bp=0.0, ref_rate_pct=2.6),
                       dict(product_code=6000, behaviour="NMD_CORE", ref_tenor="3M", direction=1, gamma_min=1.0,
                            gamma_max=1.5, scale_bp=500.0, deadband_bp=np.nan, ref_rate_pct=np.nan)])
    models = rb.models_from_frame(df)
    assert [m.ref_rate_pct for m in models] == [2.6, None]


def test_market_rate_is_the_mid_of_the_last_fixing():
    fix = pd.DataFrame({"date": ["2026-09-29", "2026-09-30", "2026-09-30", "2026-10-01"], "tenor": "3M",
                        "rate": [2.0, 2.6, 2.8, 9.9]})
    assert rb.market_rate_pct(fix, "2026-09-30") == pytest.approx(2.7)


def _bank(tmp, sub, rd):
    d = tmp / sub
    d.mkdir()
    return SimpleNamespace(models_dir=lambda: d, report_date=rd)


def test_rollforward_moves_both_profiles_as_one_model(tmp_path):
    sys.path.insert(0, str(ROOT / "balance_generate" / "python_code"))
    rf = pytest.importorskip("b_s_rollforward")
    base, new = _bank(tmp_path, "base", "2026-09-30"), _bank(tmp_path, "new", "2027-09-30")
    pd.DataFrame([dict(product_code=6000, behaviour="NMD_LEVEL", ref_tenor="3M", direction=1, gamma_min=0.79,
                       gamma_max=1.21, scale_bp=500.0, deadband_bp=0.0, ref_rate_pct=2.6)]
                 ).to_excel(base.models_dir() / rb.MODEL_FILE, index=False)
    PROFILE.to_excel(new.models_dir() / "dep_beh_models_ir.xlsx", index=False)
    PROFILE.to_excel(new.models_dir() / "dep_beh_models_liq.xlsx", index=False)
    pd.DataFrame({"product_code": [6000], "bs_side": ["L"], "beta": [0.0]}).to_excel(
        new.models_dir() / "interest_rt.xlsx", index=False)
    fix_base = pd.DataFrame({"date": ["2026-09-30"], "tenor": ["3M"], "rate": [2.6]})
    fix_new = pd.DataFrame({"date": ["2027-09-30"], "tenor": ["3M"], "rate": [3.4]})
    rf.roll_rate_sensitivity(new, base, fix_base, fix_new)
    ir = pd.read_excel(new.models_dir() / "dep_beh_models_ir.xlsx")
    liq = pd.read_excel(new.models_dir() / "dep_beh_models_liq.xlsx")
    assert (new.models_dir() / rb.MODEL_FILE).exists()
    assert (ir[6000].iloc[:3] < PROFILE[6000].iloc[:3]).all()
    assert (liq[6000].iloc[:3] < PROFILE[6000].iloc[:3]).all()          # the stable profile moves too
    np.testing.assert_allclose(ir[6000].iloc[:3], liq[6000].iloc[:3])   # core = (1 - beta) x stable, beta 0
    np.testing.assert_allclose(liq[8000], PROFILE[8000])                 # no NMD_LEVEL row


def test_rollforward_without_the_file_does_nothing(tmp_path):
    sys.path.insert(0, str(ROOT / "balance_generate" / "python_code"))
    rf = pytest.importorskip("b_s_rollforward")
    base, new = _bank(tmp_path, "base", "2026-09-30"), _bank(tmp_path, "new", "2027-09-30")
    rf.roll_rate_sensitivity(new, base, None, None)
    assert not (new.models_dir() / rb.MODEL_FILE).exists()


def test_stretch_profile_is_longer_and_ends_at_zero():
    sys.path.insert(0, str(ROOT / "inputs" / "eur_ext"))
    build = pytest.importorskip("build_eur_ext_inputs")
    months = np.array([0, 12, 24, 36, 48, 60])
    s = np.array([0.9, 0.6, 0.3, 0.0, np.nan, np.nan])
    out = build.stretch_profile(s, months, 1.25)
    assert out[0] == pytest.approx(0.9)                    # the 1D share does not move
    assert (out[1:3] > s[1:3]).all()
    assert out[4] == 0.0 and np.isnan(out[5])               # 36M x 1.25 = 45M: 0 at 48M, blank after
    np.testing.assert_allclose(build.stretch_profile(s, months, 1.0)[:4], s[:4])


def test_rollforward_rebuilds_the_core_from_the_rolled_stable_profile(tmp_path):
    sys.path.insert(0, str(ROOT / "balance_generate" / "python_code"))
    rf = pytest.importorskip("b_s_rollforward")
    base, new = _bank(tmp_path, "base", "2026-09-30"), _bank(tmp_path, "new", "2027-09-30")
    pd.DataFrame([dict(product_code=8000, behaviour="NMD_LEVEL", ref_tenor="3M", direction=1, gamma_min=0.86,
                       gamma_max=1.14, scale_bp=500.0, deadband_bp=0.0, ref_rate_pct=2.6)]
                 ).to_excel(base.models_dir() / rb.MODEL_FILE, index=False)
    liq = PROFILE[["report_date", "tenor", 8000]].copy()
    ir = liq.assign(**{"8000": 0.65 * liq[8000]}).drop(columns=8000)
    ir.columns = ["report_date", "tenor", 8000]
    liq.to_excel(new.models_dir() / "dep_beh_models_liq.xlsx", index=False)
    ir.to_excel(new.models_dir() / "dep_beh_models_ir.xlsx", index=False)
    pd.DataFrame({"product_code": [8000], "bs_side": ["L"], "beta": [0.35]}).to_excel(
        new.models_dir() / "interest_rt.xlsx", index=False)
    rf.roll_rate_sensitivity(new, base, pd.DataFrame({"date": ["2026-09-30"], "tenor": ["3M"], "rate": [2.6]}),
                             pd.DataFrame({"date": ["2027-09-30"], "tenor": ["3M"], "rate": [4.0]}))
    ir_new = pd.read_excel(new.models_dir() / "dep_beh_models_ir.xlsx")
    liq_new = pd.read_excel(new.models_dir() / "dep_beh_models_liq.xlsx")
    assert (liq_new[8000] < liq[8000]).all()
    np.testing.assert_allclose(ir_new[8000], 0.65 * liq_new[8000])
