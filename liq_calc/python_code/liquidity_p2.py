"""Pillar 2 liquidity metrics (ILAAP view) -- next to the Pillar 1 LCR / NSFR.

Four measures, all in the home currency:

1. Liquidity buffer      HQLA per product at market value after haircut -- the same
                         amounts as the LCR numerator (lcr_nsfr_objects.compute_lcr).
2. Funding concentration Share of each funding source in total liabilities by counterparty
                         type, product and currency, with the Herfindahl index
                         HHI = sum(share^2) x 10 000 (< 1 500 diversified, > 2 500 concentrated).
3. Top 10 depositors     Largest depositors by client_id (all their deposit accounts summed),
                         share of deposits and of total liabilities. EBA ALMM C 67.00 reports
                         every counterparty above 1 % of total liabilities.
4. Survival horizon      Days until buffer + cumulative net cash flows turns negative.
                         Base scenario: behavioural run-off of the book as it stands (deposits
                         by their liquidity model, loans with prepayments, nothing rolled over,
                         no new business), committed-facility drawings (baseline model) and FX
                         deal exchanges. HQLA cash flows are left out: the buffer already counts
                         those assets at market value. Stress scenarios (deposit withdrawals,
                         etc.) are applied later by editing the cash flows before the call.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from labbank_common import lcr_rules  # noqa: E402

# client_type_id of bank_config / lcr_rules
CLIENT_TYPE_NAMES = {1: "government", 4: "retail", 5: "SME", 7: "corporate",
                     8: "bond investors", 9: "banks", 14: "equity"}
WHOLESALE_TYPES = {8, 9}            # market funding: banks and bond investors
ALMM_THRESHOLD = 0.01               # ALMM C 67.00: counterparties > 1 % of total liabilities
PROFILE_DAYS = {"1D": 1, "1W": 7, "2W": 14, "1M": 30, "2M": 60, "3M": 91,
                "6M": 182, "9M": 273, "12M": 365}


# ── 1. Liquidity buffer ────────────────────────────────────────────────────────

def hqla_breakdown(bs_struct: pd.DataFrame, total_assets: float,
                   mv_ratio: dict[str, float] | None = None) -> pd.DataFrame:
    """HQLA per product: book balance, market value / book, haircut and buffer value.

    Same balances as compute_lcr: bs_percentage of the full balance (assets + liabilities).
    Balances are home currency (bs_structure shares are home-currency shares)."""
    mv_ratio = mv_ratio or {}
    df = bs_struct[(bs_struct["bs_side"] == "A") & bs_struct["hqla_class"].notna()].copy()
    df["product_code"] = df["product_code"].astype(int).astype(str)
    df["balance"] = total_assets * 2 * df["bs_percentage"] / 100.0
    df["mv_ratio"] = df["product_code"].map(lambda pc: mv_ratio.get(pc, 1.0)).astype(float)
    df["haircut"] = df["haircut"].fillna(0.0)
    df["buffer_value"] = df["balance"] * df["mv_ratio"] * (1.0 - df["haircut"])
    df["level"] = df["hqla_class"].map(lambda c: f"Level {lcr_rules.hqla_level(c)}")
    return df[["product_code", "product_name", "currency", "level", "balance", "mv_ratio",
               "haircut", "buffer_value"]].reset_index(drop=True)


# ── 2. Funding concentration ───────────────────────────────────────────────────

def funding_concentration(funding: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shares and HHI of total liabilities (equity excluded) by three dimensions.

    funding : one row per liability product line -- product_code, product_name,
              client_type_id, currency, balance_home.
    Returns (shares, hhi): shares = dimension, category, balance, share;
    hhi = dimension, hhi (0-10 000)."""
    f = funding.copy()
    f["counterparty"] = f["client_type_id"].map(lambda c: CLIENT_TYPE_NAMES.get(int(c), str(c)))
    f["product"] = f["product_code"].astype(str) + " " + f["product_name"]
    total = f["balance_home"].sum()
    rows, hhi = [], []
    for dim in ("counterparty", "product", "currency"):
        g = f.groupby(dim)["balance_home"].sum().sort_values(ascending=False)
        s = g / total
        rows += [(dim, cat, bal, sh) for cat, bal, sh in zip(g.index, g.values, s.values)]
        hhi.append((dim, float((s ** 2).sum() * 10_000)))
    return (pd.DataFrame(rows, columns=["dimension", "category", "balance", "share"]),
            pd.DataFrame(hhi, columns=["dimension", "hhi"]))


def wholesale_maturities(cf: pd.DataFrame, wholesale_codes: set[str], report_date) -> pd.DataFrame:
    """Contractual principal of wholesale funding falling due within 1M / 3M / 6M / 12M
    (cumulative). cf: product_code, cf_end_dt, principal_home (positive)."""
    w = cf[cf["product_code"].isin(wholesale_codes)]
    rd = pd.Timestamp(report_date)
    out = []
    for lbl, days in (("1M", 30), ("3M", 91), ("6M", 182), ("12M", 365)):
        due = w.loc[(w["cf_end_dt"] > rd) & (w["cf_end_dt"] <= rd + pd.Timedelta(days=days)),
                    "principal_home"].sum()
        out.append((lbl, float(due)))
    return pd.DataFrame(out, columns=["horizon", "maturing"])


# ── 3. Top depositors ──────────────────────────────────────────────────────────

def top_depositors(deposits: pd.DataFrame, total_liabilities: float, n: int = 10) -> pd.DataFrame:
    """Largest depositors by client_id (all deposit accounts of a client summed).

    deposits : client_id, client_type_id, product_name, balance_home."""
    d = deposits.copy()
    g = (d.groupby("client_id")
          .agg(balance=("balance_home", "sum"),
               client_type_id=("client_type_id", "first"),
               products=("product_name", lambda s: ", ".join(sorted(set(s))))))
    g = g.sort_values("balance", ascending=False)
    total_dep = g["balance"].sum()
    top = g.head(n).reset_index()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["counterparty"] = top["client_type_id"].map(lambda c: CLIENT_TYPE_NAMES.get(int(c), str(c)))
    top["share_deposits"] = top["balance"] / total_dep
    top["share_liabilities"] = top["balance"] / total_liabilities
    top["cum_share_deposits"] = top["share_deposits"].cumsum()
    top["above_almm_1pct"] = top["share_liabilities"] > ALMM_THRESHOLD
    return top[["rank", "client_id", "counterparty", "products", "balance", "share_deposits",
                "share_liabilities", "cum_share_deposits", "above_almm_1pct"]]


# ── 4. Survival horizon ────────────────────────────────────────────────────────

def survival_profile(flows: pd.DataFrame, buffer: float, report_date,
                     horizon_days: int = 730) -> pd.DataFrame:
    """Daily liquidity position = buffer + cumulative net cash flow.

    flows : date, amount_home (+ cash in, - cash out). Flows on or before the report date
            are ignored. Returns day, date, net_cf, cum_net_cf, position."""
    rd = pd.Timestamp(report_date)
    days = pd.date_range(rd, rd + pd.Timedelta(days=horizon_days), freq="D")
    f = flows[(flows["date"] > rd) & (flows["date"] <= days[-1])]
    net = f.groupby(f["date"].dt.normalize())["amount_home"].sum().reindex(days, fill_value=0.0)
    prof = pd.DataFrame({"date": days, "net_cf": net.values})
    prof.insert(0, "day", (prof["date"] - rd).dt.days)
    prof["cum_net_cf"] = prof["net_cf"].cumsum()
    prof["position"] = buffer + prof["cum_net_cf"]
    return prof


def survival_days(profile: pd.DataFrame) -> int | None:
    """First day the position is negative; None = the bank survives the whole horizon."""
    neg = profile.loc[profile["position"] < 0, "day"]
    return int(neg.iloc[0]) if len(neg) else None


def profile_points(profile: pd.DataFrame) -> pd.DataFrame:
    """Position at the standard horizons (1D ... 12M)."""
    rows = []
    for lbl, d in PROFILE_DAYS.items():
        r = profile[profile["day"] <= d].iloc[-1]
        rows.append((lbl, d, float(r["cum_net_cf"]), float(r["position"])))
    return pd.DataFrame(rows, columns=["horizon", "day", "cum_net_cf", "position"])
