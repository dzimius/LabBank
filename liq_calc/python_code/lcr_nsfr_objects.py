"""LCR and NSFR computation objects.

LCR (Liquidity Coverage Ratio) — Basel III, 30-day stress horizon
------------------------------------------------------------------
LCR = HQLA / Net Cash Outflows

  HQLA         = Σ_assets  balance_amt × (1 - haircut)  [where hqla_class is set]
  Outflows_30d = Σ_liab    balance_amt × LCR_run_off     [LCR column in bs_structure]
  Inflows_30d  = min(Σ contractual 30d inflows × inflow rate, 0.75 × Outflows_30d)
  Net Outflows = Outflows_30d − Inflows_30d

Inflow rates by counterparty of the product (labbank_common.lcr_rules, LCR DA Art. 32):
financial customers 100%, retail / non-financial 50% -- applied by the caller to the
contractual (not behavioural) 30-day receipts; HQLA and cards give no inflow. Term deposits:
financial counterparties run off what matures within 30 days, retail / non-financial keep
their product rate. HQLA at market value (Art. 9) via hqla_mv_ratio.

NSFR (Net Stable Funding Ratio) — Basel III / EBA
--------------------------------------------------
NSFR = ASF / RSF

  ASF = Σ_liab+equity  balance_amt × ASF_factor   [ASF column in bs_structure]
  RSF = Σ_assets       balance_amt × RSF_factor   [RSF column in bs_structure]

Committed facilities (off balance sheet, credit-card limits included) come in from the
facility register (schemat.commitments, labbank_common/commitments.py) as two overlays per
currency: LCR outflow = available-within-30-days x undrawn_lcr, RSF = undrawn x undrawn_rsf.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# ---------------------------------------------------------------------------
# LCR
# ---------------------------------------------------------------------------

def compute_lcr(
    bs_struct: pd.DataFrame,
    total_assets: float,
    inflows_30d_by_ccy: dict[str, float] | None = None,
    td_maturity_split: pd.DataFrame | None = None,
    derivative_outflows_by_ccy: dict[str, float] | None = None,
    commitment_outflows_by_ccy: dict[str, float] | None = None,
    hqla_mv_ratio: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute LCR per currency from balance-sheet structure.

    Parameters
    ----------
    bs_struct          : bs_structure sheet from bank_data.xlsx, with columns:
                         bs_side, product_name, currency, bs_percentage, hqla_class, haircut, LCR
    total_assets       : total asset amount (used with bs_percentage to get balance_amt)
    inflows_30d_by_ccy : dict currency → raw 30-day inflow amount from cf.products.
                         If None, inflows are assumed 0 (conservative).
    td_maturity_split  : DataFrame from sql_setup.load_td_maturity_split() with columns:
                         currency, product_code, within_30d_amt, after_30d_amt -- the
                         within-30d share drives financial counterparties' term deposits
                         (lcr_rules.term_deposit_runoff).
    hqla_mv_ratio      : product_code -> market value / book (sql_setup.load_hqla_market_value_ratio);
                         missing = 1.
    derivative_outflows_by_ccy : dict currency -> 30-day payments on derivatives (FX deal
                         exchanges), added to outflows at 100%; their receipts come in
                         through inflows_30d_by_ccy.

    Returns
    -------
    DataFrame with one row per currency:
      currency, hqla, outflows_30d, inflows_30d, net_outflows_30d, lcr
    """
    full_balance = total_assets * 2  # assets + liabilities/equity
    df = bs_struct.copy()
    df["balance_amt"] = full_balance * df["bs_percentage"] / 100.0

    inflows_map  = inflows_30d_by_ccy or {}
    from labbank_common import lcr_rules
    mv_ratio = hqla_mv_ratio or {}
    frac30 = {}
    if td_maturity_split is not None and not td_maturity_split.empty:
        for r in td_maturity_split.itertuples():
            tot = float(r.within_30d_amt) + float(r.after_30d_amt)
            if tot > 0:
                frac30[(r.currency, str(r.product_code))] = float(r.within_30d_amt) / tot
    currencies = df["currency"].unique()
    rows = []

    for ccy in currencies:
        ccy_df = df[df["currency"] == ccy]

        # HQLA: asset rows with hqla_class set
        hqla_mask = (ccy_df["bs_side"] == "A") & ccy_df["hqla_class"].notna()
        _mv = ccy_df.loc[hqla_mask, "product_code"].map(
            lambda pc: mv_ratio.get(str(int(pc)), 1.0)).astype(float)
        _value = (
            ccy_df.loc[hqla_mask, "balance_amt"]
            * (1.0 - ccy_df.loc[hqla_mask, "haircut"].fillna(0.0))
            * _mv                                              # market value, not book
        )
        hqla = _value.sum()
        _level = ccy_df.loc[hqla_mask, "hqla_class"].map(lcr_rules.hqla_level)
        _l2a, _l2b = _value[_level == "2A"].sum(), _value[_level == "2B"].sum()
        if _l2a or _l2b:                                       # composition caps (Art. 17)
            hqla = lcr_rules.capped_hqla(hqla - _l2a - _l2b, _l2a, _l2b)

        # Outflows: liability rows with LCR run-off rate set
        out_mask    = (ccy_df["bs_side"] == "L") & ccy_df["LCR"].notna()
        dep_mask    = out_mask & (ccy_df.get("product_name", pd.Series(dtype=str)) == "term_deposit")
        non_dep_mask = out_mask & ~dep_mask

        # Non-deposit outflows: standard calculation
        outflows = (
            ccy_df.loc[non_dep_mask, "balance_amt"]
            * ccy_df.loc[non_dep_mask, "LCR"]
        ).sum()

        # Undrawn committed facilities (cards included): contingent outflows, not assets.
        outflows += (commitment_outflows_by_ccy or {}).get(ccy, 0.0)

        # Term-deposit outflows per product and counterparty (lcr_rules.term_deposit_runoff)
        for _, r in ccy_df.loc[dep_mask].iterrows():
            rate = lcr_rules.term_deposit_runoff(
                r["LCR"], r.get("client_type_id"),
                frac30.get((ccy, str(int(r["product_code"]))), 1.0))
            outflows += r["balance_amt"] * rate

        # Inflows: already counterparty-weighted by the caller (lcr_rules.inflow_rate);
        # aggregate cap at 75% of outflows (LCR DA Art. 33)
        outflows += (derivative_outflows_by_ccy or {}).get(ccy, 0.0)
        raw_inflows = inflows_map.get(ccy, 0.0)
        cap = 0.75 * outflows
        inflows = min(raw_inflows, cap)

        net_outflows = max(outflows - inflows, 0.0)
        lcr = hqla / net_outflows if net_outflows > 0 else np.nan

        rows.append({
            "currency":         ccy,
            "hqla":             hqla,
            "outflows_30d":     outflows,
            "inflows_30d":      inflows,
            "net_outflows_30d": net_outflows,
            "lcr":              lcr,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NSFR
# ---------------------------------------------------------------------------

def compute_nsfr(
    bs_struct: pd.DataFrame,
    total_assets: float,
    commitment_rsf_by_ccy: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute NSFR per currency from balance-sheet structure.

    Parameters
    ----------
    bs_struct    : bs_structure sheet; needs columns: bs_side, currency,
                   bs_percentage, ASF, RSF
    total_assets : total asset amount

    Returns
    -------
    DataFrame with one row per currency:
      currency, asf, rsf, nsfr
    """
    full_balance = total_assets * 2
    df = bs_struct.copy()
    df["balance_amt"] = full_balance * df["bs_percentage"] / 100.0

    currencies = df["currency"].unique()
    rows = []

    for ccy in currencies:
        ccy_df = df[df["currency"] == ccy]

        # ASF: liabilities + equity rows with ASF factor set
        asf_mask = ccy_df["bs_side"].isin(["L", "E"]) & ccy_df["ASF"].notna()
        asf = (
            ccy_df.loc[asf_mask, "balance_amt"]
            * ccy_df.loc[asf_mask, "ASF"]
        ).sum()

        # RSF: asset rows with RSF factor set
        rsf_mask = (ccy_df["bs_side"] == "A") & ccy_df["RSF"].notna()
        rsf = (
            ccy_df.loc[rsf_mask, "balance_amt"]
            * ccy_df.loc[rsf_mask, "RSF"]
        ).sum()

        rsf += (commitment_rsf_by_ccy or {}).get(ccy, 0.0)     # undrawn committed facilities
        nsfr = asf / rsf if rsf > 0 else np.nan

        rows.append({
            "currency": ccy,
            "asf":      asf,
            "rsf":      rsf,
            "nsfr":     nsfr,
        })

    return pd.DataFrame(rows)
