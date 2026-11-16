"""Tests for the accounting view (accounting/), the history store (run_store) and the run registry."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "accounting", "python_code"))

import acct_common as ac  # noqa: E402
import acct_pricing as ap  # noqa: E402
from labbank_common import derivatives as deriv  # noqa: E402
from labbank_common import run_store  # noqa: E402
from labbank_common.bank_config import get_bank  # noqa: E402


class FlatCurves:
    """PathCurves stand-in: one flat continuously-compounded rate on every curve and date."""

    names = {"EUR_disc_curve", "EUR_fwd_curve"}

    def __init__(self, r):
        self.r = r

    def df(self, name, d, targets):
        t = (pd.to_datetime(pd.Series(targets)) - pd.Timestamp(d)).dt.days.to_numpy(float) / 365.0
        return np.exp(-self.r * t)

    def fwd(self, name, d, a, b, dc="ACT/360"):
        da, db = self.df(name, d, [a, b])
        return (da / db - 1.0) / ap.year_frac(pd.Timestamp(a), pd.Timestamp(b), dc)


def _bond(**kw):
    row = dict(start_date=pd.Timestamp("2025-03-31"), maturity_date=pd.Timestamp("2030-03-31"), balance_amt=100.0,
               payment_freq="1Y", dc_conv="ACT/365", client_rt=0.03, index_rt=0.03, rate_type="F",
               disc_curve="EUR_disc_curve")
    row.update(kw)
    return SimpleNamespace(**row)


def test_fixed_bond_prices_above_par_when_rates_fall_and_accrues():
    d = pd.Timestamp("2026-09-30")
    hi, acc = ap.security_value(_bond(), d, FlatCurves(0.05), 0.0)
    lo, _ = ap.security_value(_bond(), d, FlatCurves(0.01), 0.0)
    assert lo > 100.0 > hi
    assert acc == pytest.approx(3.0 * 183 / 365, rel=1e-6)          # half a year of a 3% annual coupon
    wider, _ = ap.security_value(_bond(), d, FlatCurves(0.01), 50.0)
    assert wider < lo                                                 # spread lowers the value


def test_security_value_is_zero_once_matured_or_not_yet_bought():
    assert ap.security_value(_bond(), pd.Timestamp("2030-04-01"), FlatCurves(0.03), 0.0) == (0.0, 0.0)
    assert ap.security_value(_bond(start_date=pd.Timestamp("2027-01-01")), pd.Timestamp("2026-09-30"),
                             FlatCurves(0.03), 0.0) == (0.0, 0.0)


def test_floating_note_at_market_spread_is_at_par():
    v, _ = ap.security_value(_bond(rate_type="V", client_rt=0.04, index_rt=0.03, payment_freq="3M",
                                   dc_conv="ACT/360"), pd.Timestamp("2026-09-30"), FlatCurves(0.02), 100.0)
    assert v == pytest.approx(100.0, abs=1e-9)


def _swap(**kw):
    s = dict(swap_id="S", notional=1e6, start_date=pd.Timestamp("2026-09-30"),
             maturity_date=pd.Timestamp("2031-09-30"), fixed_rate=0.0, fixed_pay_freq="1Y", fixed_dc_conv="30/360",
             float_pay_freq="6M", float_fixing_freq="6M", float_dc_conv="ACT/360", float_spread=0.0, pay_fixed=1,
             disc_curve="EUR_disc_curve", fwd_curve="EUR_fwd_curve")
    s.update(kw)
    return SimpleNamespace(**s)


def test_pay_fixed_swap_gains_when_rates_rise_and_receiver_is_the_mirror():
    d = pd.Timestamp("2026-09-30")
    c = FlatCurves(0.03)
    lo, _ = ap.swap_value(_swap(fixed_rate=0.02), d, c)
    hi, _ = ap.swap_value(_swap(fixed_rate=0.04), d, c)
    assert lo > 0 > hi
    rec, _ = ap.swap_value(_swap(fixed_rate=0.02, pay_fixed=0), d, c)
    assert rec == pytest.approx(-lo)
    assert ap.swap_value(_swap(fixed_rate=0.02), pd.Timestamp("2031-10-01"), c) == (0.0, 0.0)


def test_interp_df_log_linear_and_flat_zero_beyond_the_curve():
    x, y = np.array([1.0, 365.0]), np.exp(-0.03 * np.array([1.0, 365.0]) / 365.0)
    assert ac.interp_df(x, y, np.array([730.0]))[0] == pytest.approx(np.exp(-0.06), rel=1e-9)
    assert ac.interp_df(x, y, np.array([0.0]))[0] == 1.0


def test_hedge_accounting_flag_defaults_and_validation(tmp_path):
    row = dict(swap_id="IRS_X", swap_type="IRS", pay_fixed=1, notional=1e6, currency="EUR",
               start_date="2026-01-01", maturity_date="2030-01-01", fixed_rate=0.02, fixed_pay_freq="1Y",
               fixed_dc_conv="30/360", fixed_b_day_conv="ModifiedFollowing", float_rate_index="EUR_ASK_6M",
               float_pay_freq="6M", float_fixing_freq="6M", float_dc_conv="ACT/360", disc_curve="EUR_disc_curve",
               fwd_curve="EUR_fwd_curve")
    p = tmp_path / "d.xlsx"
    pd.DataFrame([row]).to_excel(p, sheet_name="swap", index=False)
    assert deriv.read_swaps(path=p)["hedge_accounting"].iloc[0] == "fair_value"
    pd.DataFrame([{**row, "hedge_accounting": "None"}]).to_excel(p, sheet_name="swap", index=False)
    assert deriv.read_swaps(path=p)["hedge_accounting"].iloc[0] == "none"
    pd.DataFrame([{**row, "hedge_accounting": "macro"}]).to_excel(p, sheet_name="swap", index=False)
    with pytest.raises(ValueError, match="hedge_accounting"):
        deriv.read_swaps(path=p)


def test_eur_inputs_flag_one_fvtpl_swap_and_fvoci_tbills():
    from labbank_common.bank_config import BANKS
    for b in ("eur", "eur_ext", "eur_multi"):
        if b != "eur" and (b not in BANKS or not BANKS[b].derivatives_dir().exists()):   # not in this copy
            continue
        f = ac.swap_flags(b)
        assert (f["hedge_accounting"] == "none").sum() == 1, b
        assert f.loc[f["hedge_accounting"] == "none", "swap_id"].iloc[0] == "IRS_020"
        c = ac.product_classes(b).set_index(["product_code", "bs_side"])
        assert c.loc[(3200, "A"), "accounting"] == "FVOCI" and c.loc[(3200, "A"), "portfolio"] == "liquidity"
        assert c.loc[(1000, "A"), "accounting"] == "AC" and pd.isna(c.loc[(1000, "A"), "portfolio"])
        fin = ac.fin_params(b)
        assert fin["CIT"] == 0.30 and fin["dividend_payout"] == 0.30


def test_opex_rate_matches_the_optimiser():
    sys.path.insert(0, os.path.join(ROOT, "optimize_prep", "python_code"))
    from product_map import OPEX_RATE
    import pnl_rollforward
    assert pnl_rollforward.OPEX_RATE == OPEX_RATE


def test_current_run_defaults():
    assert run_store.current_run("eur").run_id == "actual"
    r = run_store.current_run("eur_y1")
    assert r.run_id == "fc1y" and r.base_report_date == pd.Timestamp("2026-09-30") and r.kind == "forecast"


def test_report_date_override_reaches_only_the_environment_bank(monkeypatch):
    monkeypatch.setenv("LABBANK_BANK", "eur")
    monkeypatch.setenv("LABBANK_REPORT_DATE", "2026-10-31")
    assert get_bank().report_date == "2026-10-31"
    assert get_bank("eur").report_date == "2026-10-31"
    assert get_bank("pln").report_date == "2026-06-30"
    monkeypatch.delenv("LABBANK_REPORT_DATE")
    assert get_bank().report_date == "2026-09-30"


def test_pnl_equity_targets_keep_the_balance_sheet_closed(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "balance_generate", "python_code"))
    import b_s_rollforward as rf
    t = pd.DataFrame({"product_code": [1000, 3510, 7060, 5100, 5300, 5400], "bs_side": list("AALEEE"),
                      "product_name": ["mortgage_fixed", "cash", "term_deposit", "common_shares",
                                       "retained_earnings", "risk_reserves"],
                      "balance_base": [900.0, 100.0, 800.0, 80.0, 70.0, 50.0],
                      "target": [990.0, 110.0, 880.0, 88.0, 77.0, 55.0]})
    d = tmp_path / "accounting"
    d.mkdir()
    pd.DataFrame({"item": ["cash_surplus_close"], "value": [12.5]}).to_excel(d / "pnl_rollforward.xlsx",
                                                                               sheet_name="summary", index=False)
    bank = SimpleNamespace(output_dir=lambda m, create=False: tmp_path / m)
    out = rf.pnl_equity_targets(bank, t).set_index(["product_code", "bs_side"])["target"]
    assert out[(5100, "E")] == 80.0 and out[(5400, "E")] == 50.0           # held at base
    assert out[(5300, "E")] == pytest.approx(88 + 77 + 55 + 12.5 - 130)     # plan equity + cash - the rest
    assert out[(3510, "A")] == 122.5
    a = out[[k for k in out.index if k[1] == "A"]].sum()
    assert a == pytest.approx(out[[k for k in out.index if k[1] != "A"]].sum())


# ── SQL: history isolation (needs the SQL Server; skipped without it) ───────────────────────

def _engine():
    try:
        from labbank_common.bank_config import get_engine
        eng = get_engine("eur")
        with eng.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return eng
    except Exception:  # noqa: BLE001
        pytest.skip("SQL Server not reachable")


def test_replace_run_touches_only_its_own_run():
    eng = _engine()
    a = run_store.RunContext("eur", pd.Timestamp("2026-09-30"), "actual")
    b = run_store.RunContext("eur", pd.Timestamp("2026-09-30"), "fc_test")
    c = run_store.RunContext("eur", pd.Timestamp("2026-10-31"), "actual")
    tbl = "run_store_isolation"
    try:
        for r, v in ((a, 1.0), (b, 2.0), (c, 3.0)):
            run_store.replace_run(eng, pd.DataFrame({"x": [v, v]}), tbl, "tmp_test", r)
        run_store.replace_run(eng, pd.DataFrame({"x": [9.0], "extra": ["new col"]}), tbl, "tmp_test", b)
        got = {r.run_id + str(r.report_date.date()): run_store.read_run(eng, tbl, r, schema="tmp_test")["x"].tolist()
               for r in (a, b, c)}
        assert got == {"actual2026-09-30": [1.0, 1.0], "fc_test2026-09-30": [9.0], "actual2026-10-31": [3.0, 3.0]}
        run_store.register_run(eng, "running", b)
        run_store.register_run(eng, "ok", b)
        from sqlalchemy import text
        st = pd.read_sql(text("SELECT status FROM meta.runs WHERE run_id = 'fc_test'"), eng)["status"].tolist()
        assert st == ["ok"]
    finally:
        with eng.begin() as conn:
            conn.exec_driver_sql("IF OBJECT_ID('tmp_test.run_store_isolation') IS NOT NULL "
                                 "DROP TABLE tmp_test.run_store_isolation")
            conn.exec_driver_sql("IF OBJECT_ID('meta.runs') IS NOT NULL DELETE FROM meta.runs WHERE run_id = 'fc_test'")
            conn.exec_driver_sql("IF SCHEMA_ID('tmp_test') IS NOT NULL DROP SCHEMA tmp_test")


def test_eur_y1_monthly_pnl_closes_if_present():
    f = get_bank("eur_y1").output_dir("accounting", create=False) / "pnl_rollforward.xlsx"
    if not f.exists():
        pytest.skip("pnl_rollforward.xlsx not generated")
    s = pd.read_excel(f, sheet_name="summary").set_index("item")["value"]
    assert abs(float(s["max_abs_a_minus_l_minus_e"])) < 1.0
    assert abs(float(s["max_abs_equity_identity"])) < 1.0
    assert abs(float(s["interest_vs_dynamic_nii"])) < 1.0
