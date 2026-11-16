import pandas as pd
import pytest

from labbank_common import derivatives as deriv


def _swap(**kw):
    row = dict(swap_id="S1", swap_type="IRS", currency="EUR", float_currency="EUR")
    row.update(kw)
    return pd.DataFrame([row])


def test_irs_must_have_one_currency():
    with pytest.raises(ValueError, match="one currency"):
        deriv.validate_swaps(_swap(float_currency="USD"))


def test_cirs_needs_two_currencies():
    with pytest.raises(ValueError, match="two currencies"):
        deriv.validate_swaps(_swap(swap_type="CIRS"))


def test_cirs_is_priced_ois_is_not_yet():
    deriv.validate_swaps(_swap(swap_type="CIRS", float_currency="USD"))
    with pytest.raises(NotImplementedError, match="OIS"):
        deriv.validate_swaps(_swap(swap_type="OIS"))


def test_irs_passes():
    deriv.validate_swaps(_swap())


def _fx_swap(near="2026-09-15", far="2026-12-15"):
    return pd.DataFrame([dict(
        deal_id="FXS_1", deal_type="FX_SWAP", ccy_pair="EURUSD", trade_date=pd.Timestamp("2026-09-11"),
        near_date=pd.Timestamp(near), near_buy_ccy="EUR", near_buy_amt=69_000_000.0,
        near_sell_ccy="USD", near_sell_amt=80_000_000.0, near_rate=1.1594,
        far_date=pd.Timestamp(far), far_buy_ccy="USD", far_buy_amt=80_000_000.0,
        far_sell_ccy="EUR", far_sell_amt=68_800_000.0, far_rate=1.1628)])


def test_fx_swap_after_near_date_is_long_usd_short_eur():
    legs = deriv.fx_legs(_fx_swap(), "2026-09-30").set_index("currency")
    usd, eur = legs.loc["USD"], legs.loc["EUR"]
    assert usd.bs_side == "A" and usd.notional == 80e6 and usd.interest == 0.0
    assert eur.bs_side == "L" and eur.outstanding == 69e6
    assert eur.interest == pytest.approx(-0.2e6)          # swap points: EUR paid back < received
    assert usd.near_amount == 0.0 and eur.near_amount == 0.0


def test_fx_swap_matured_has_no_legs():
    assert deriv.fx_legs(_fx_swap(far="2026-09-20"), "2026-09-30").empty


def test_forward_starting_fx_swap_keeps_near_exchange():
    legs = deriv.fx_legs(_fx_swap(near="2026-10-15", far="2027-01-15"), "2026-09-30").set_index("currency")
    assert legs.loc["USD", "near_amount"] == -80e6         # USD paid at the near date
    assert legs.loc["EUR", "near_amount"] == 69e6


def test_open_position_from_exchanges():
    ex = deriv.fx_exchanges(_fx_swap())
    fut = ex[ex["date"] > "2026-09-30"].groupby("currency")["amount"].sum()
    assert fut["USD"] == 80e6 and fut["EUR"] == -68.8e6


# ── asset swaps ──────────────────────────────────────────────────────────────

def _bonds(n=10, nominal=50e6):
    return pd.DataFrame({
        "transaction_id": [f"13000{i:08d}" for i in range(n)], "product_code": "3000",
        "balance_amt": nominal, "currency": "EUR",
        "start_date": pd.Timestamp("2024-06-28"), "maturity_date": pd.Timestamp("2029-06-28"),
        "client_rt": 0.0255, "payment_freq": "1Y", "dc_conv": "ACT/ACT",
        "b_day_conv": "ModifiedFollowing", "disc_curve": "EUR_disc_curve", "fwd_curve": "EUR_fwd_curve"})


def _rules(pct=0.8):
    return pd.DataFrame({"product_code": ["3000"], "hedge_pct": [pct], "float_rate_index": ["EUR_ASK_6M"],
                         "float_freq": ["6M"], "asw_spread_bp": [0.0]})


def test_asset_swaps_mirror_their_bonds_and_hit_the_hedge_share():
    from labbank_common import derivatives as d
    sw = d.build_asset_swaps(_rules(0.8), _bonds(), "2026-09-30", seed=1)
    assert len(sw) == 8 and sw["notional"].sum() == pytest.approx(400e6)
    r = sw.iloc[0]
    assert r.pay_fixed == 1 and r.fixed_rate == pytest.approx(0.0255)        # pays the bond coupon
    assert (r.fixed_pay_freq, r.fixed_dc_conv) == ("1Y", "ACT/ACT")             # bond's conventions
    assert r.maturity_date == pd.Timestamp("2029-06-28") and r.float_rate_index == "EUR_ASK_6M"
    assert sw["hedged_transaction_id"].is_unique and sw["swap_id"].str.len().max() <= 13
    # deterministic for a seed, and a matured bond is never hedged
    pd.testing.assert_frame_equal(sw, d.build_asset_swaps(_rules(0.8), _bonds(), "2026-09-30", seed=1))
    old = _bonds(1).assign(transaction_id="x", maturity_date=pd.Timestamp("2026-01-30"))
    assert "x" not in set(d.build_asset_swaps(_rules(1.0), pd.concat([_bonds(), old]),
                                              "2026-09-30", seed=1)["hedged_transaction_id"])
    assert d.build_asset_swaps(_rules(0.0), _bonds(), "2026-09-30", seed=1).empty
