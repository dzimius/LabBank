"""NMD money that leaves because of a rate scenario renews in NII as the product clients move to."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labbank_common import irrbb_behaviour as ib
from labbank_common import rate_behaviour as rb

ROOT = Path(__file__).resolve().parents[2]
RD = pd.Timestamp("2026-09-30")


def _rows(code, caps, sched=1, ptype="D", renew=1.0, bal=1000.0):
    ends = [RD + pd.DateOffset(months=i + 1) for i in range(len(caps))]
    return pd.DataFrame({"schedule_id": sched, "product_type": ptype, "product_code": code, "currency": "EUR",
                         "cf_start_dt": [RD + pd.DateOffset(months=i) for i in range(len(caps))],
                         "cf_end_dt": ends, "capital_pmt": caps, "prepayment_pmt": 0.0, "outstanding_bal": bal,
                         "cf_yf": 1 / 12, "int_pmt": 0.0, "bs_side": "L", "rate_type": "F", "fwd_rt": 0.0,
                         "renew_months": renew, "margin": 0.0})


def test_migrated_capital_is_the_scenario_outflow_beyond_the_base():
    base = pd.concat([_rows("6000", [10.0, 10.0, 10.0]), _rows("7060", [0.0, 0.0, 500.0], sched=2, renew=12.0)])
    shocked = pd.concat([_rows("6000", [15.0, 12.0, 8.0]), base[base["product_code"] == "7060"]])
    out = ib.nmd_migration(base, shocked, {"6000": ("7060", None)})
    m = out[out["product_code"] == "6000"]
    assert m["migr_capital"].tolist() == pytest.approx([5.0, 2.0, -2.0])
    assert (m["migr_product"] == "7060").all() and (m["migr_months"] == 12.0).all()
    assert (out.loc[out["product_code"] == "7060", "migr_capital"] == 0.0).all()


def test_no_scenario_or_no_target_changes_nothing():
    base = _rows("6000", [10.0])
    assert ib.nmd_migration(base, base, {"6000": ("7060", None)}) is base
    assert ib.nmd_migration(base, base.copy(), {}).equals(base)
    with pytest.raises(ValueError):
        ib.nmd_migration(base, base.copy(), {"6000": ("7060", None)})  # no 7060 book to take a tenor from
    out = ib.nmd_migration(base, base.copy(), {"6000": ("7070", 3.0)}) # a stated tenor needs no book
    assert (out["migr_months"] == 3.0).all() and (out["migr_capital"] == 0.0).all()


def test_migrate_to_is_read_from_the_model_file():
    df = pd.DataFrame([dict(product_code=6000, behaviour="NMD_CORE", ref_tenor="3M", direction=1, gamma_min=1.0,
                            gamma_max=1.5, scale_bp=500.0, deadband_bp=0.0, migrate_to=7060.0, migrate_months=6.0),
                       dict(product_code=1000, behaviour="CPR", ref_tenor="10Y", direction=-1, gamma_min=0.6,
                            gamma_max=1.4, scale_bp=200.0, deadband_bp=0.0, migrate_to=np.nan, migrate_months=np.nan)])
    assert [(m.migrate_to, m.migrate_months) for m in rb.models_from_frame(df)] == [("7060", 6.0), (None, None)]
    with pytest.raises(ValueError):
        rb.Sensitivity("1000", "CPR", "10Y", -1, 0.6, 1.4, 200.0, migrate_to="7060")


def _flat_curve(rate=0.03, days=800):
    dates = pd.date_range(RD, periods=days, freq="D")
    dfs = np.exp(-rate * (dates - RD).days.to_numpy() / 365.0)
    return pd.Series(dfs, index=pd.MultiIndex.from_arrays([["EUR_DISC"] * days, dates],
                                                          names=["curve_name", "node_date"]))


def test_nii_renews_the_migrated_part_at_the_target_rate():
    sys.path.insert(0, str(ROOT / "irrbb_calc" / "python_code"))
    nii = pytest.importorskip("nii_calc_objects")
    curve = _flat_curve()
    rows = _rows("6000", [100.0, 0.0, 0.0]).assign(migr_capital=[40.0, 0.0, 0.0], migr_product="7060",
                                                   migr_months=12.0)
    kw = dict(coeff_a_map={"6000": 0.0, "7060": 0.7}, coeff_b_map={"6000": 0.0, "7060": 0.0})
    with_m = nii.compute_nii_shocked_schedule(rows, curve, {"EUR": "EUR_DISC"}, RD, "par_up", return_rows=True, **kw)
    without = nii.compute_nii_shocked_schedule(rows.drop(columns=["migr_capital", "migr_product", "migr_months"]),
                                               curve, {"EUR": "EUR_DISC"}, RD, "par_up", return_rows=True, **kw)
    assert without["nii_renewal"].sum() == pytest.approx(0.0)       # current account renews at 0%
    r0 = with_m.iloc[0]
    assert r0["nii_migration"] == pytest.approx(-40.0 * 0.7 * 0.0305 * r0["remain_yf"], rel=0.02)
    assert with_m["nii_renewal"].sum() == pytest.approx(with_m["nii_migration"].sum())
