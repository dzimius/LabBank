"""Holding fields of securities (nominal / acquisition date / price) and the t_bill guard."""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "balance_generate", "python_code"))
from b_s_gen_objects import add_security_holding_fields  # noqa: E402


def _tx(**kw):
    r = dict(product_name="t_bill", balance_amt=985_221.67, client_rt=0.03, dc_conv="ACT/360",
             start_date=pd.Timestamp("2026-04-03"), maturity_date=pd.Timestamp("2026-09-30"),
             payment_freq="12M", maturity="12M")
    r.update(kw)
    return pd.DataFrame([r])


def test_bill_nominal_is_the_redemption_amount():
    out = add_security_holding_fields(_tx()).iloc[0]       # 180 days at 3% ACT/360
    assert out.nominal_amt == pytest.approx(1_000_000.0, abs=0.01)
    assert out.purchase_price == pytest.approx(98.522167, abs=1e-5)
    assert out.acquisition_date == pd.Timestamp("2026-04-03")


def test_bond_at_par_and_existing_holdings_kept():
    out = add_security_holding_fields(_tx(product_name="bond_fixed", payment_freq="1Y", maturity="5Y")).iloc[0]
    assert out.nominal_amt == pytest.approx(985_221.67) and out.purchase_price == pytest.approx(100.0)
    aged = add_security_holding_fields(_tx(nominal_amt=1_234.0)).iloc[0]    # roll-forward keeps it
    assert aged.nominal_amt == 1_234.0


def test_bill_with_interim_payments_is_rejected():
    with pytest.raises(ValueError, match="pay once at maturity"):
        add_security_holding_fields(_tx(payment_freq="1M"))
