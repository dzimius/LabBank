"""LCR and NSFR computation objects.

LCR (Liquidity Coverage Ratio) — Basel III, 30-day stress horizon
------------------------------------------------------------------
LCR = HQLA / Net Cash Outflows

  HQLA         = Σ_assets  balance_amt × (1 - haircut)  [where hqla_class is set]
  Outflows_30d = Σ_liab    balance_amt × LCR_run_off     [LCR column in bs_structure]
  Inflows_30d  = min(CF asset inflows within 30d × 0.75, 0.75 × Outflows_30d)
  Net Outflows = Outflows_30d − Inflows_30d

NSFR (Net Stable Funding Ratio) — Basel III / EBA
--------------------------------------------------
NSFR = ASF / RSF

  ASF = Σ_liab+equity  balance_amt × ASF_factor   [ASF column in bs_structure]
  RSF = Σ_assets       balance_amt × RSF_factor   [RSF column in bs_structure]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# LCR
# ---------------------------------------------------------------------------

def compute_lcr(
    bs_struct: pd.DataFrame,
    total_assets: float,
    inflows_30d_by_ccy: dict[str, float] | None = None,
    td_maturity_split: pd.DataFrame | None = None,
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
                         currency, within_30d_amt, after_30d_amt.
                         When provided, deposits maturing within 30 days receive a 100%
                         run-off weight instead of the default LCR weight from bs_struct.

    Returns
    -------
    DataFrame with one row per currency:
      currency, hqla, outflows_30d, inflows_30d, net_outflows_30d, lcr
    """
    full_balance = total_assets * 2  # assets + liabilities/equity
    df = bs_struct.copy()
    df["balance_amt"] = full_balance * df["bs_percentage"] / 100.0

    inflows_map  = inflows_30d_by_ccy or {}
    split_by_ccy = (
        td_maturity_split.set_index("currency")
        if td_maturity_split is not None and not td_maturity_split.empty
        else None
    )
    currencies = df["currency"].unique()
    rows = []

    for ccy in currencies:
        ccy_df = df[df["currency"] == ccy]

        # HQLA: asset rows with hqla_class set
        hqla_mask = (ccy_df["bs_side"] == "A") & ccy_df["hqla_class"].notna()
        hqla = (
            ccy_df.loc[hqla_mask, "balance_amt"]
            * (1.0 - ccy_df.loc[hqla_mask, "haircut"].fillna(0.0))
        ).sum()

        # Outflows: liability rows with LCR run-off rate set
        out_mask    = (ccy_df["bs_side"] == "L") & ccy_df["LCR"].notna()
        dep_mask    = out_mask & (ccy_df.get("product_name", pd.Series(dtype=str)) == "term_deposit")
        non_dep_mask = out_mask & ~dep_mask

        # Non-deposit outflows: standard calculation
        outflows = (
            ccy_df.loc[non_dep_mask, "balance_amt"]
            * ccy_df.loc[non_dep_mask, "LCR"]
        ).sum()

        # Deposit outflows: maturity-aware if split data available
        if split_by_ccy is not None and ccy in split_by_ccy.index:
            within_30d = float(split_by_ccy.loc[ccy, "within_30d_amt"])
            after_30d  = float(split_by_ccy.loc[ccy, "after_30d_amt"])

            # Derive the default LCR weight for deposits from bs_struct
            # (weighted average across all deposit rows for this currency)
            dep_rows = ccy_df[dep_mask]
            if not dep_rows.empty:
                total_dep_bal = dep_rows["balance_amt"].sum()
                default_weight = (
                    (dep_rows["balance_amt"] * dep_rows["LCR"]).sum() / total_dep_bal
                    if total_dep_bal > 0 else 0.0
                )
            else:
                default_weight = 0.0

            outflows += within_30d * 1.0 + after_30d * default_weight
        else:
            # No split data: fall back to standard bs_struct calculation
            outflows += (
                ccy_df.loc[dep_mask, "balance_amt"]
                * ccy_df.loc[dep_mask, "LCR"]
            ).sum()

        # Inflows capped at 75% of outflows (Basel III §33(d))
        raw_inflows = inflows_map.get(ccy, 0.0) * 0.75  # 75% recognition rate
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

        nsfr = asf / rsf if rsf > 0 else np.nan

        rows.append({
            "currency": ccy,
            "asf":      asf,
            "rsf":      rsf,
            "nsfr":     nsfr,
        })

    return pd.DataFrame(rows)
