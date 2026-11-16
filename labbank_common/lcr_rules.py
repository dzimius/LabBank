"""lcr_rules.py
=============
Which asset cash flows count as LCR inflows -- one definition shared by the exact LCR
(liq_calc) and the optimizer's per-product inflow factors (optimize_prep/extract_params).

- HQLA (bs_structure.hqla_class set): already in the liquidity buffer at market value, so its
  maturing principal and coupons are NOT inflows as well (EU LCR Delegated Act 2015/61,
  Art. 32(6)); counting both double-counts e.g. a T-bill maturing within 30 days.
- Credit cards (2200 / 2210): revolving balances, no contractual 30-day inflow.
"""
from __future__ import annotations

import pandas as pd

CARD_CODES = ("2200", "2210")


def hqla_level(hqla_class) -> str:
    """bs_structure.hqla_class as '1', '2A' or '2B'. Numeric 1 / 2 / 3 mean the same three levels."""
    c = str(hqla_class).strip().upper()
    c = {"1.0": "1", "2": "2A", "2.0": "2A", "3": "2B", "3.0": "2B"}.get(c, c)
    if c not in ("1", "2A", "2B"):
        raise ValueError(f"unknown hqla_class {hqla_class!r}")
    return c


def capped_hqla(level1: float, level2a: float, level2b: float) -> float:
    """Liquidity buffer after the composition caps (LCR DA Art. 17): Level 2 at most 40% and Level 2B
    at most 15% of the buffer. Amounts are after haircuts."""
    adj_2b = max(level2b - 15.0 / 85.0 * (level1 + level2a), level2b - 15.0 / 60.0 * level1, 0.0)
    adj_l2 = max(level2a + level2b - adj_2b - 2.0 / 3.0 * level1, 0.0)
    return level1 + level2a + level2b - adj_2b - adj_l2


def hqla_product_codes(bs_struct: pd.DataFrame) -> set[str]:
    """Asset product codes that sit in the HQLA buffer."""
    a = bs_struct[(bs_struct["bs_side"] == "A") & bs_struct["hqla_class"].notna()]
    return {str(int(c)) for c in a["product_code"]}


def non_inflow_codes(bs_struct: pd.DataFrame) -> list[str]:
    """Asset product codes whose 30-day cash flows are not LCR inflows (sorted)."""
    return sorted(set(CARD_CODES) | hqla_product_codes(bs_struct))


# ── counterparty rules (client_type_id of bank_config: 1 government, 4 individuals,
#    5 SME, 7 non-financial corporates, 8 bond investors, 9 banks, 14 equity) ───────
FINANCIAL_CLIENT_TYPES = {9}


def is_financial(client_type_id) -> bool:
    try:
        return int(client_type_id) in FINANCIAL_CLIENT_TYPES
    except (TypeError, ValueError):
        return False


def inflow_rate(client_type_id) -> float:
    """Recognised share of a performing contractual 30-day inflow (Art. 32(3)(a)):
    financial customers 100%, retail / non-financial customers 50%. (The flat 75% used
    before overstated the retail and corporate loan instalments that dominate inflows.)"""
    return 1.0 if is_financial(client_type_id) else 0.5


def term_deposit_runoff(lcr_weight: float, client_type_id, frac_within_30d: float) -> float:
    """Outflow rate of a term-deposit product (per unit of its balance).

    Financial counterparties: what matures within 30 days runs off in full, the rest is
    not a 30-day outflow (Art. 27-28). Retail / non-financial: the product's own retail
    rate (Art. 24-25) whatever the maturity -- maturing within 30 days does not make a
    retail deposit a 100% outflow (it used to)."""
    if is_financial(client_type_id):
        return float(frac_within_30d)
    return 0.0 if pd.isna(lcr_weight) else float(lcr_weight)


def product_client_types(bs_struct: pd.DataFrame) -> dict[tuple[str, str], int]:
    """(product_code, bs_side) -> client_type_id from bs_structure."""
    return {(str(int(r.product_code)), str(r.bs_side)): int(r.client_type_id)
            for r in bs_struct.itertuples() if pd.notna(r.client_type_id)}


def market_spreads_bp(bs_struct: pd.DataFrame) -> dict[str, float]:
    """Optional bs_structure column mv_spread_bp: the security's market yield spread over the
    discount curve (e.g. +100 for a sovereign trading 1pp above the swap / OIS curve), used
    for its market value. Blank / missing column = 0."""
    if "mv_spread_bp" not in bs_struct.columns:
        return {}
    a = bs_struct[(bs_struct["bs_side"] == "A") & bs_struct["mv_spread_bp"].notna()]
    return {str(int(r.product_code)): float(r.mv_spread_bp) for r in a.itertuples()}


def hqla_market_value_ratio(engine, report_date, hqla_codes, spreads_bp: dict | None = None) -> dict[str, float]:
    """Market value / book balance per HQLA product (LCR Art. 9: the buffer at market
    value). Market value = PV of the security's remaining contractual cash flows on
    today's discount curve (cf.products.d_f) plus the product's market spread
    (spreads_bp, see market_spreads_bp; dirty price). Products without cash flows or with only zero payments (cash at the central
    bank) are absent -> 1."""
    codes = ", ".join(f"'{str(c)}'" for c in hqla_codes) or "''"
    sp = {str(k): float(v) for k, v in (spreads_bp or {}).items() if v}
    spread_case = ("CASE CAST(product_code AS VARCHAR(4)) "
                   + " ".join(f"WHEN '{k}' THEN {v!r}" for k, v in sp.items()) + " ELSE 0 END"
                   if sp else "0")
    from sqlalchemy import text
    query = text(f"""
        WITH pv AS (
            SELECT CAST(product_code AS VARCHAR(4)) AS product_code,
                   SUM(COALESCE(con_total_pmt, 0) * COALESCE(d_f, 0)
                       * EXP(-({spread_case}) / 10000.0
                             * DATEDIFF(day, :rd, cf_end_dt) / 365.0)) AS mv
            FROM cf.products
            WHERE bs_side = 'A' AND cf_end_dt > :rd
              AND CAST(product_code AS VARCHAR(4)) IN ({codes})
            GROUP BY product_code),
        bal AS (
            SELECT CAST(product_code AS VARCHAR(4)) AS product_code, SUM(balance_amt) AS book
            FROM schemat.financial_instruments
            WHERE report_date = :rd AND bs_side = 'A'
              AND CAST(product_code AS VARCHAR(4)) IN ({codes})
            GROUP BY product_code)
        SELECT pv.product_code, pv.mv / NULLIF(bal.book, 0) AS ratio
        FROM pv JOIN bal ON bal.product_code = pv.product_code
    """)
    df = pd.read_sql_query(query, engine, params={"rd": pd.Timestamp(report_date)})
    # products whose cash-flow rows carry no payments (e.g. central-bank cash with zero-amount
    # rows) have no market value from cash flows: they stay at book (ratio 1)
    return {str(r.product_code): float(r.ratio) for r in df.itertuples()
            if pd.notna(r.ratio) and r.ratio > 0}
