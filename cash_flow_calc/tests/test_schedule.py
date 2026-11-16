"""End-to-end invariants of the loan / instrument schedule (gen_orgin_sched_loan_fin_inst +
compute_amort_schedule_vectorized) -- not just helpers."""
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cash_flow_calc", "python_code"))
import cf_calc_objects as cfo  # noqa: E402

RD = pd.Timestamp("2026-09-30")
EMPTY = pd.Series(dtype=float)
# same shape as the workflow's fix_df; fixing_asof needs the two index levels even with no quotes
NO_FIXINGS = pd.Series(dtype=float, name="rate", index=pd.MultiIndex.from_arrays(
    [pd.DatetimeIndex([]), pd.Index([], dtype=object)], names=["fixing_date", "rate_index"]))


def _row(**kw):
    r = dict(schedule_id=1, currency="EUR", b_day_conv="ModifiedFollowing", dc_conv="ACT/360",
             amort_type=2, start_date=pd.Timestamp("2025-01-15").date(),
             maturity_date=pd.Timestamp("2030-01-15").date(), payment_freq="1M", fixing_freq=None,
             rate_index="EUR_ASK_5Y", disc_curve="EUR_disc_curve", fwd_curve="EUR_fwd_curve")
    r.update(kw)
    return SimpleNamespace(**r)


def _sched(row, balance=1_000_000.0):
    s = cfo.gen_orgin_sched_loan_fin_inst(row, RD, EMPTY, EMPTY, NO_FIXINGS)
    s["fwd_rt"] = 0.03
    p = pd.DataFrame([{"schedule_id": row.schedule_id, "balance_amt": balance,
                       "amort_type": row.amort_type, "bs_side": "A"}])
    return cfo.compute_amort_schedule_vectorized(s, p)


def test_no_past_payment_and_full_balance_amortised_from_today():
    s = _sched(_row())
    assert (s["cf_end_dt"] > RD).all()                              # no past instalment
    assert s["cf_start_dt"].iloc[0] < RD < s["cf_end_dt"].iloc[0]   # current accrual period kept
    assert s["outstanding_bal"].iloc[0] == pytest.approx(1_000_000)
    assert s["capital_pmt"].sum() == pytest.approx(1_000_000)       # all of today's balance repaid


@pytest.mark.parametrize("amort", [1, 2])
def test_amortising_types_repay_today_balance(amort):
    s = _sched(_row(amort_type=amort))
    assert s["outstanding_bal"].iloc[0] == pytest.approx(1_000_000)
    assert s["capital_pmt"].sum() == pytest.approx(1_000_000, rel=1e-6)


def test_payment_on_report_date_counts_as_paid():
    s = _sched(_row(start_date=pd.Timestamp("2025-09-30").date(),
                    maturity_date=pd.Timestamp("2027-09-30").date()))
    assert s["cf_end_dt"].min() > RD and s["cf_start_dt"].iloc[0] == RD


def test_final_stub_pays_at_maturity():
    s = _sched(_row(amort_type=0, start_date=pd.Timestamp("2026-01-15").date(),
                    maturity_date=pd.Timestamp("2027-04-28").date()))
    assert s["cf_end_dt"].max() == pd.Timestamp("2027-04-28")       # bullet at maturity, not 15 Apr (13-day stub)
    last = s.iloc[-1]
    assert last["cf_start_dt"] == pd.Timestamp("2027-04-15") and last["capital_pmt"] == pytest.approx(1_000_000)
