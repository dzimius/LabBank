import numpy as np
import pandas as pd
import pytest

from labbank_common import commitments as cmt

RD = "2026-09-30"


def _line(**kw):
    row = dict(commitment_code="9450", own_name="rcf", drawn_product_code="", currency="EUR",
               client_type_id=7, facility_type="credit", revocability="irrevocable", revolving=1,
               utilization=np.nan, limit_pct=1.0, availability="5Y", commitment_fee_rate=0.0035,
               drawdown_model_id="corp", undrawn_lcr=0.10, undrawn_rsf=0.05, undrawn_ccf=0.40,
               rwa_weight=1.0, PD=0.01, LGD=0.4)
    row.update(kw)
    return pd.DataFrame([row])


def test_unused_line_is_all_undrawn_and_off_balance():
    reg = cmt.build_register(_line(), {}, 10e9, {"EUR": 1.0}, RD).iloc[0]
    assert reg.limit_amt == pytest.approx(100e6) and reg.drawn_amt == 0 and reg.undrawn_amt == pytest.approx(100e6)
    assert reg.undrawn_amt * reg.undrawn_lcr == pytest.approx(10e6)      # 10% LCR outflow
    assert reg.undrawn_amt * reg.undrawn_rsf == pytest.approx(5e6)       # 5% RSF


def test_linked_line_limit_reconciles_to_drawn():
    line = _line(drawn_product_code="4100", utilization=0.75, limit_pct=np.nan)
    reg = cmt.build_register(line, {"4100": 300e6}, 10e9, {"EUR": 1.0}, RD).iloc[0]
    assert reg.limit_amt == pytest.approx(400e6)
    assert reg.limit_amt == pytest.approx(reg.drawn_amt + reg.undrawn_amt)


def test_validation():
    cmt.validate(_line())
    with pytest.raises(ValueError, match="limit_pct"):          # standalone line needs a limit
        cmt.validate(_line(limit_pct=np.nan, utilization=0.5))
    with pytest.raises(ValueError, match="utilization"):        # linked line needs utilisation
        cmt.validate(_line(drawn_product_code="4100"))
    with pytest.raises(ValueError):                              # zero utilisation = no drawn product
        cmt.FacilityTerms(utilization=0.0)
    two = pd.concat([_line(drawn_product_code="2200", utilization=0.4, limit_pct=np.nan),
                     _line(commitment_code="9201", drawn_product_code="2200", utilization=0.5,
                           limit_pct=np.nan)])
    with pytest.raises(ValueError, match="counted twice"):
        cmt.validate(two)


def _models(cum=(0.1, 0.2, 0.25, 0.3)):
    return pd.DataFrame([dict(drawdown_model_id="corp", scenario="stress", tenor=t, cum_draw_pct=c,
                              drawn_repay="12M", months=m)
                         for t, c, m in zip(["1M", "3M", "6M", "12M"], cum, [1, 3, 6, 12])])


def test_drawdowns_never_exceed_undrawn_and_are_repaid():
    reg = cmt.build_register(_line(), {}, 10e9, {"EUR": 1.0}, RD)
    fl = cmt.drawdown_flows(reg, _models(), RD)
    draws = fl[fl.flow_type == "drawdown"].amount
    assert -draws.sum() == pytest.approx(0.30 * 100e6)
    assert fl.amount.sum() == pytest.approx(0.0)                 # every drawing comes back


def test_expiry_stops_new_drawings():
    reg = cmt.build_register(_line(availability="3M"), {}, 10e9, {"EUR": 1.0}, RD)
    fl = cmt.drawdown_flows(reg, _models(), RD)
    assert fl[fl.flow_type == "drawdown"].month.max() == 3


def test_line_expiring_within_30_days_has_no_lcr_capacity():
    reg = cmt.build_register(_line(availability="0M"), {}, 10e9, {"EUR": 1.0}, RD).iloc[0]
    assert reg.available_amt_30d == 0.0


def test_line_ending_soon_is_still_fully_drawable_for_lcr():
    # 1M from 2026-09-30 ends on 2026-10-30 (30 days): drawable today -> full undrawn counts
    reg = cmt.build_register(_line(availability="1M"), {}, 10e9, {"EUR": 1.0}, RD).iloc[0]
    assert reg.available_amt_30d == pytest.approx(reg.undrawn_amt)
    assert cmt.FacilityTerms.from_row(_line(availability="1M").iloc[0]).drawable
    assert not cmt.FacilityTerms.from_row(_line(availability="0M").iloc[0]).drawable
    assert cmt.FacilityTerms.from_row(_line(availability=np.nan).iloc[0]).drawable


BS = pd.DataFrame({"bs_side": ["A", "A"], "product_code": [4100, 2200], "currency": ["EUR", "EUR"]})


def test_validation_rejects_silent_understatement():
    with pytest.raises(ValueError, match="duplicate commitment_code"):
        cmt.validate(pd.concat([_line(), _line()]))
    with pytest.raises(ValueError, match="limit_pct"):
        cmt.validate(_line(limit_pct=-1.0))
    with pytest.raises(ValueError, match="currency"):
        cmt.validate(_line(currency=np.nan))
    with pytest.raises(ValueError, match="tenor"):
        cmt.validate(_line(availability="10D"))
    with pytest.raises(ValueError, match="PD"):
        cmt.validate(_line(PD=1.5))
    linked = dict(drawn_product_code="4100", utilization=0.75, limit_pct=np.nan)
    cmt.validate(_line(**linked), bs_struct=BS)
    with pytest.raises(ValueError, match="not an asset"):
        cmt.validate(_line(**{**linked, "drawn_product_code": "4999"}), bs_struct=BS)
    with pytest.raises(ValueError, match="differs"):
        cmt.validate(_line(**linked, currency="USD"), bs_struct=BS)


def test_validation_unknown_drawdown_model():
    cmt.validate(_line(), models=_models())                       # 'corp' exists
    with pytest.raises(ValueError, match="drawdown_model_id"):
        cmt.validate(_line(drawdown_model_id="nope"), models=_models())


def _raw_models(rows):
    return pd.DataFrame([dict(drawdown_model_id="corp", scenario="stress", tenor=t,
                              cum_draw_pct=c, drawn_repay="12M") for t, c in rows])


def test_drawdown_models_checked_in_tenor_order():
    # typed out of order but monotone by tenor -> accepted and returned sorted
    df = cmt.validate_drawdown_models(_raw_models([("12M", 0.3), ("1M", 0.1), ("6M", 0.25), ("3M", 0.2)]))
    assert df["months"].tolist() == [1, 3, 6, 12]
    with pytest.raises(ValueError, match="decrease"):              # '1Y' sorts after '6M'
        cmt.validate_drawdown_models(_raw_models([("1M", 0.1), ("1Y", 0.05), ("6M", 0.2)]))
    with pytest.raises(ValueError, match="duplicate tenor"):       # 12M == 1Y
        cmt.validate_drawdown_models(_raw_models([("12M", 0.3), ("1Y", 0.3)]))
    with pytest.raises(ValueError, match="missing"):
        cmt.validate_drawdown_models(_raw_models([("1M", np.nan)]))
    with pytest.raises(ValueError, match="tenor"):
        cmt.validate_drawdown_models(_raw_models([("1D", 0.1)]))


def test_standalone_totals_include_expected_loss():
    reg = cmt.build_register(_line(), {}, 10e9, {"EUR": 1.0}, RD)
    tot = cmt.standalone_totals(reg, {"EUR": 1.0})
    # 100M undrawn x CCF 40% x PD 1% x LGD 40%
    assert tot["el"] == pytest.approx(100e6 * 0.40 * 0.01 * 0.40)
    assert tot["rwa"] == pytest.approx(100e6 * 0.40 * 1.0)
