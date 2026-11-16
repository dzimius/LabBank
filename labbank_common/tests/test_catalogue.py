"""Catalogue variants (bank_data sheet catalogue): expansion of bs_structure and the models."""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from labbank_common import catalogue as cat  # noqa: E402

BS = pd.DataFrame({
    "bs_percentage": [20.0, 6.0, 1.0, 1.0],
    "bs_side": ["A", "L", "A", "L"],
    "product_code": [1100, 7060, 7900, 7900],
    "product_name": ["mortgage_float", "term_deposit", "term_deposit", "term_deposit"],
    "own_name": ["mortgage_float_ind", "term_deposit_ind", "td_bank_a", "td_bank_l"],
    "maturity": ["30Y", "6M", "1M", "1M"],
    "payment_freq": ["1M", "6M", "1M", "1M"],
    "fixing_freq": ["3M", None, "1M", "1M"],
    "rate_index": ["EUR_ASK_3M", "EUR_BID_6M", "EUR_ASK_1M", "EUR_BID_1M"],
    "amort_type": [1.0, 0.0, 0.0, 0.0],
    "LCR": [None, 0.05, None, 1.0],
})
VARIANTS = pd.DataFrame({
    "parent_code": ["1100", "7060", "7900"],
    "bs_side": ["A", "L", "L"],
    "product_code": ["1101", "7064", "7901"],
    "own_name": ["mortgage_float_ind_20y_1m", "term_deposit_ind_24m", "td_bank_l_3m"],
    "maturity": ["20Y", "24M", "3M"],
    "payment_freq": [pd.NA, "12M", pd.NA],
    "fixing_freq": ["1M", pd.NA, pd.NA],
    "rate_index": ["EUR_ASK_1M", "EUR_BID_2Y", pd.NA],
    "amort_type": [pd.NA, pd.NA, pd.NA],
})


def test_variant_rows_inherit_parent_with_overrides():
    out = cat.expand_bs_structure(BS, VARIANTS)
    assert len(out) == len(BS) + 3
    v = out.set_index(out["product_code"].astype(str)).loc["1101"]
    assert v["bs_percentage"] == 0.0 and v["product_name"] == "mortgage_float"
    assert (v["maturity"], v["fixing_freq"], v["rate_index"]) == ("20Y", "1M", "EUR_ASK_1M")
    assert v["payment_freq"] == "1M" and v["amort_type"] == 1.0          # blank = parent's
    td = out.set_index(out["product_code"].astype(str)).loc["7064"]
    assert (td["maturity"], td["payment_freq"], td["LCR"]) == ("24M", "12M", 0.05)
    # 7900 sits on both sides: the variant copies the side it names
    tb = out[out["product_code"].astype(str) == "7901"].iloc[0]
    assert (tb["bs_side"], tb["rate_index"], tb["LCR"]) == ("L", "EUR_BID_1M", 1.0)


def test_expansion_is_idempotent():
    once = cat.expand_bs_structure(BS, VARIANTS)
    assert len(cat.expand_bs_structure(once, VARIANTS)) == len(once)


def test_unknown_parent_is_an_error():
    bad = VARIANTS.assign(parent_code=["9999", "7060", "7900"])
    with pytest.raises(ValueError, match="parent 9999"):
        cat.expand_bs_structure(BS, bad)


def test_model_rows_and_columns_copied_from_parent():
    pmap = dict(zip(VARIANTS["product_code"], VARIANTS["parent_code"]))
    rate = pd.DataFrame({"product_code": [1100, 7060], "beta": [1.0, 0.8]})
    r = cat.expand_rows(rate, pmap).set_index("product_code")["beta"]
    assert r[1101] == 1.0 and r[7064] == 0.8 and 7901 not in r.index     # 7900 has no row here
    loans = pd.DataFrame({"report_date": ["2026-09-30"], "tenor": ["1M"], 1100: [0.002]})
    lc = cat.expand_columns(loans, pmap)
    assert lc[1101].iloc[0] == 0.002 and 7064 not in lc.columns
    # a variant that already has its own model keeps it
    own = pd.DataFrame({"product_code": [1100, 1101], "beta": [1.0, 0.5]})
    assert cat.expand_rows(own, pmap).set_index("product_code")["beta"][1101] == 0.5


def test_workbook_without_catalogue_sheet(tmp_path):
    p = tmp_path / "bank_data.xlsx"
    with pd.ExcelWriter(p) as xw:
        BS.to_excel(xw, sheet_name="bs_structure", index=False)
    assert cat.read_variants(p).empty
    assert len(cat.read_bs_structure(p)) == len(BS)
    assert cat.parent_map(path=p) == {}
