"""commitments.py
==============
Committed facilities: the UNDRAWN part of credit / liquidity lines, off balance sheet.

A facility = drawn loan (an asset in bs_structure, ordinary asset risk) + unused commitment
(off balance sheet: contingent liquidity outflows, stable funding, credit exposure via a
CCF, future drawings). The unused part never enters the asset total or the 50/50
balance-sheet weights.

Inputs (bank_data.xlsx)
-----------------------
sheet ``commitments`` -- one row per facility line (credit cards included):
  commitment_code, own_name    identification (codes 9xxx)
  drawn_product_code           the bs_structure asset the drawings sit in (blank = none:
                               a line nobody has drawn yet)
  currency, client_type_id
  facility_type                credit | liquidity | guarantee | trade (a guarantee or letter of credit
                               has no drawn product: its limit is the amount issued, a call is a drawing)
  revocability                 irrevocable | conditionally_revocable | unconditionally_cancellable
  revolving                    1 = repayments restore available capacity
  utilization                  drawn / limit for a line linked to a drawn product (0 < u <= 1);
                               limit = drawn / u, undrawn = limit - drawn (never typed in)
  limit_pct                    limit as % of total assets for a line WITHOUT a drawn product
                               (zero utilisation); undrawn = limit
  availability                 tenor until the customer can no longer draw (blank = open-ended);
                               while it has not ended, the WHOLE undrawn amount counts as drawable
                               within 30 days for the LCR (a line ending on day 20 can still be
                               drawn in full on day 1) -- FacilityTerms.drawable, shared by the
                               register and the optimizer extract
  commitment_fee_rate          annual fee on the undrawn amount (not in NII yet)
  drawdown_model_id            row group in sheet drawdown_models
  undrawn_lcr                  LCR outflow factor on the amount drawable within 30 days
                               (EU LCR DA Art. 31: retail 5%, non-financial corporate credit 10%,
                               liquidity facilities 30%)
  undrawn_rsf                  NSFR RSF factor (5% irrevocable / conditionally revocable)
  undrawn_ccf                  credit conversion factor for RWA / expected loss
  rwa_weight, PD, LGD          only for lines without a drawn product (linked lines use the
                               drawn product's)

sheet ``drawdown_models`` -- drawdown_model_id, scenario (baseline | stress), tenor (1M..),
  cum_draw_pct: cumulative share of today's undrawn amount drawn by that tenor;
  drawn_repay: tenor after which a drawing is repaid (liquidity inflow).

An LCR factor, a CCF and a drawdown probability are three different inputs on purpose.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SHEET = "commitments"
MODEL_SHEET = "drawdown_models"
FACILITY_TYPES = {"credit", "liquidity", "guarantee", "trade"}
REVOCABILITY = {"irrevocable", "conditionally_revocable", "unconditionally_cancellable"}
_TENOR = re.compile(r"^\s*(\d+)\s*([MY])\s*$", re.IGNORECASE)


def _num(row, key, default=np.nan) -> float:
    v = row.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


@dataclass(frozen=True)
class FacilityTerms:
    """Off-balance terms of one facility line; the generic successor of the card-only terms."""
    utilization: float = float("nan")      # drawn / limit (linked lines)
    undrawn_lcr: float = 0.0
    undrawn_rsf: float = 0.0
    undrawn_ccf: float = 0.0
    availability_m: float = float("nan")   # months the line stays drawable (NaN = open-ended)

    def __post_init__(self):
        for name in ("undrawn_lcr", "undrawn_rsf", "undrawn_ccf"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be a fraction in [0, 1], got {v}")
        if not math.isnan(self.utilization) and not 0.0 < self.utilization <= 1.0:
            raise ValueError(f"utilization must be in (0, 1], got {self.utilization} "
                             "(a line with no drawings has no drawn product: use limit_pct)")

    @classmethod
    def from_row(cls, row) -> "FacilityTerms":
        return cls(utilization=_num(row, "utilization"),
                   undrawn_lcr=_num(row, "undrawn_lcr", 0.0),
                   undrawn_rsf=_num(row, "undrawn_rsf", 0.0),
                   undrawn_ccf=_num(row, "undrawn_ccf", 0.0),
                   availability_m=_availability_months(row.get("availability")))

    @property
    def drawable(self) -> bool:
        """Can the customer still draw today? Then the whole undrawn amount is drawable
        within the LCR's 30 days, however soon availability ends after that."""
        return math.isnan(self.availability_m) or self.availability_m > 0

    @property
    def undrawn_per_drawn(self) -> float:
        """Undrawn amount per unit of drawn balance (linked lines only)."""
        return 0.0 if math.isnan(self.utilization) else 1.0 / self.utilization - 1.0

    @property
    def ead_multiplier(self) -> float:
        """Exposure at default per unit of drawn balance: drawn + CCF x undrawn."""
        return 1.0 + self.undrawn_per_drawn * self.undrawn_ccf


def read_commitments(path) -> pd.DataFrame:
    """The commitments sheet, validated (empty frame if the bank has none)."""
    try:
        df = pd.read_excel(path, sheet_name=SHEET)
    except ValueError:
        return pd.DataFrame()
    df = df.dropna(subset=["commitment_code"]).copy()
    if df.empty:
        return df
    df["commitment_code"] = df["commitment_code"].astype(int).astype(str)
    df["drawn_product_code"] = df["drawn_product_code"].apply(
        lambda v: "" if pd.isna(v) else str(int(v)))
    from labbank_common import catalogue
    bs = catalogue.read_bs_structure(path)
    validate(df, bs_struct=bs, models=read_drawdown_models(path))
    return df


def validate(df: pd.DataFrame, bs_struct: pd.DataFrame | None = None,
             models: pd.DataFrame | None = None) -> None:
    """Reject inputs that would otherwise give a silently wrong (usually understated)
    exposure. bs_struct / models: cross-checks of linked products and drawdown models."""
    dup = df["commitment_code"].duplicated()
    if dup.any():
        raise ValueError(f"commitments: duplicate commitment_code {sorted(df.loc[dup, 'commitment_code'])}")
    ccy = df["currency"]
    if ccy.isna().any() or (ccy.astype(str).str.strip() == "").any():
        raise ValueError("commitments: every line needs a currency")
    bad = set(df["facility_type"]) - FACILITY_TYPES
    if bad:
        raise ValueError(f"commitments: unknown facility_type {sorted(bad)}")
    bad = set(df["revocability"]) - REVOCABILITY
    if bad:
        raise ValueError(f"commitments: unknown revocability {sorted(bad)}")
    for _, r in df.iterrows():
        FacilityTerms.from_row(r)                     # factor ranges
        linked = r["drawn_product_code"] != ""
        has_u = not math.isnan(_num(r, "utilization"))
        has_l = not math.isnan(_num(r, "limit_pct"))
        if linked and not (has_u and not has_l):
            raise ValueError(f"commitment {r['commitment_code']}: a line with a drawn product "
                             "is sized by utilization (not limit_pct)")
        if not linked and not (has_l and not has_u):
            raise ValueError(f"commitment {r['commitment_code']}: a line without a drawn product "
                             "is sized by limit_pct (not utilization)")
        if has_l and not (math.isfinite(_num(r, "limit_pct")) and _num(r, "limit_pct") > 0):
            raise ValueError(f"commitment {r['commitment_code']}: limit_pct must be a positive % of total assets")
        for k in ("PD", "LGD", "rwa_weight", "commitment_fee_rate"):
            v = _num(r, k, 0.0)
            if not (math.isfinite(v) and v >= 0) or (k in ("PD", "LGD") and v > 1):
                raise ValueError(f"commitment {r['commitment_code']}: invalid {k} = {r.get(k)}")
        _availability_months(r.get("availability"))           # tenor syntax
        if bs_struct is not None and linked:
            a = bs_struct[(bs_struct["bs_side"] == "A") &
                          (pd.to_numeric(bs_struct["product_code"], errors="coerce")
                           == int(r["drawn_product_code"]))]
            if a.empty:
                raise ValueError(f"commitment {r['commitment_code']}: drawn_product_code "
                                 f"{r['drawn_product_code']} is not an asset in bs_structure")
            if "currency" in a.columns and r["currency"] not in set(a["currency"]):
                raise ValueError(f"commitment {r['commitment_code']}: currency {r['currency']} differs "
                                 f"from its drawn product's {sorted(set(a['currency']))}")
        mid = r.get("drawdown_model_id")
        if (models is not None and isinstance(mid, str) and mid.strip()
                and mid not in set(models["drawdown_model_id"])):
            raise ValueError(f"commitment {r['commitment_code']}: drawdown_model_id {mid!r} "
                             "not in sheet drawdown_models")
    dup = df["drawn_product_code"][df["drawn_product_code"] != ""].duplicated()
    if dup.any():
        raise ValueError("commitments: one drawn product can belong to one facility line only "
                         "(else its undrawn amount is counted twice)")


def terms_by_drawn_product(commit: pd.DataFrame) -> dict[str, FacilityTerms]:
    """drawn product_code -> FacilityTerms (linked lines)."""
    if commit is None or commit.empty:
        return {}
    return {r["drawn_product_code"]: FacilityTerms.from_row(r)
            for _, r in commit.iterrows() if r["drawn_product_code"] != ""}


def build_register(commit: pd.DataFrame, drawn_by_product: dict[str, float],
                   total_assets_home: float, to_home: dict[str, float],
                   report_date) -> pd.DataFrame:
    """Facility register at the report date (amounts in the line's own currency).

    drawn_by_product: product_code -> drawn balance (own currency) of linked products.
    limit = drawn / utilization (linked) or limit_pct % x total assets (standalone);
    undrawn = limit - drawn; available_30d = undrawn unless availability ends within 30 days.
    """
    if commit is None or commit.empty:
        return pd.DataFrame()
    rd = pd.Timestamp(report_date)
    rows = []
    for _, r in commit.iterrows():
        t = FacilityTerms.from_row(r)
        ccy = r["currency"]
        if r["drawn_product_code"]:
            if r["drawn_product_code"] not in drawn_by_product:
                print(f"  [warn] commitment {r['commitment_code']}: drawn product "
                      f"{r['drawn_product_code']} has no balance in the book -> limit 0")
            drawn = float(drawn_by_product.get(r["drawn_product_code"], 0.0))
            limit = drawn / t.utilization if drawn > 0 else 0.0
        else:
            drawn = 0.0
            limit = _num(r, "limit_pct") / 100.0 * total_assets_home / to_home[ccy]
        undrawn = max(limit - drawn, 0.0)
        avail_end = (pd.NaT if math.isnan(t.availability_m)
                     else rd + pd.DateOffset(months=int(t.availability_m)))
        avail_30d = undrawn if t.drawable else 0.0
        rows.append({
            "report_date": rd, "commitment_code": r["commitment_code"], "own_name": r["own_name"],
            "drawn_product_code": r["drawn_product_code"] or None, "currency": ccy,
            "client_type_id": int(r["client_type_id"]), "facility_type": r["facility_type"],
            "revocability": r["revocability"], "revolving": int(_num(r, "revolving", 0)),
            "limit_amt": limit, "drawn_amt": drawn, "undrawn_amt": undrawn,
            "available_amt_30d": avail_30d, "availability_end_date": avail_end,
            "commitment_fee_rate": _num(r, "commitment_fee_rate", 0.0),
            "drawdown_model_id": r.get("drawdown_model_id"),
            "undrawn_lcr": t.undrawn_lcr, "undrawn_rsf": t.undrawn_rsf, "undrawn_ccf": t.undrawn_ccf,
            "rwa_weight": _num(r, "rwa_weight", 0.0), "PD": _num(r, "PD", 0.0), "LGD": _num(r, "LGD", 0.0),
        })
    return pd.DataFrame(rows)


def _months(tenor: str) -> int:
    """'3M' -> 3, '2Y' -> 24; anything else (e.g. '1D', '6W', '') raises."""
    m = _TENOR.match(str(tenor))
    if not m:
        raise ValueError(f"tenor {tenor!r}: expected <n>M or <n>Y")
    return int(m.group(1)) * (12 if m.group(2).upper() == "Y" else 1)


def _availability_months(avail) -> float:
    """Availability tenor in months; NaN when blank (open-ended)."""
    if avail is None or (isinstance(avail, float) and math.isnan(avail)) or str(avail).strip() == "":
        return float("nan")
    return float(_months(avail))


def read_drawdown_models(path) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=MODEL_SHEET)
    except ValueError:
        return pd.DataFrame(columns=["drawdown_model_id", "scenario", "tenor", "cum_draw_pct", "drawn_repay"])
    return validate_drawdown_models(df)


def validate_drawdown_models(df: pd.DataFrame) -> pd.DataFrame:
    """Checks in TENOR order (rows may be typed in any order); returns the frame sorted."""
    df = df.dropna(how="all").copy()
    if df["cum_draw_pct"].isna().any():
        raise ValueError("drawdown_models: missing cum_draw_pct")
    df["months"] = df["tenor"].map(_months)
    df["drawn_repay"].dropna().map(_months)                    # repay tenor syntax
    key = ["drawdown_model_id", "scenario"]
    if df.duplicated(key + ["months"]).any():
        raise ValueError("drawdown_models: duplicate tenor within a model / scenario")
    df = df.sort_values(key + ["months"]).reset_index(drop=True)
    if (df.groupby(key)["cum_draw_pct"].apply(lambda s: (s.diff().dropna() < 0).any())).any():
        raise ValueError("drawdown_models: cum_draw_pct must not decrease with tenor")
    if ((df["cum_draw_pct"] < 0) | (df["cum_draw_pct"] > 1)).any():
        raise ValueError("drawdown_models: cum_draw_pct must be in [0, 1] (drawings never exceed availability)")
    return df


def standalone_totals(reg: pd.DataFrame, to_home) -> dict[str, float]:
    """Home-currency totals of the lines WITHOUT a drawn product (the optimizer's fixed
    commit_* scalars): LCR outflow, RSF, RWA and expected loss (undrawn x CCF x PD x LGD)."""
    com = reg[reg["drawn_product_code"].isna() | (reg["drawn_product_code"].astype(str) == "")] \
        if not reg.empty else reg
    if com.empty:
        return {"lcr": 0.0, "rsf": 0.0, "rwa": 0.0, "el": 0.0}
    f = com["currency"].map(to_home).astype(float)
    ead = com["undrawn_amt"] * com["undrawn_ccf"] * f
    return {"lcr": float((com["available_amt_30d"] * com["undrawn_lcr"] * f).sum()),
            "rsf": float((com["undrawn_amt"] * com["undrawn_rsf"] * f).sum()),
            "rwa": float((ead * com["rwa_weight"]).sum()),
            "el":  float((ead * com["PD"] * com["LGD"]).sum())}


def read_register(engine) -> pd.DataFrame:
    """schemat.commitments, or an empty frame when the bank has no facilities (the table
    is dropped then). Any other read error propagates: a failed read must not look like
    'no commitments' and understate LCR outflows / RSF / RWA."""
    from sqlalchemy import text
    with engine.connect() as c:
        exists = c.execute(text("SELECT OBJECT_ID('schemat.commitments', 'U')")).scalar() is not None
    return pd.read_sql("SELECT * FROM schemat.commitments", engine) if exists else pd.DataFrame()


def drawdown_flows(register: pd.DataFrame, models: pd.DataFrame, report_date) -> pd.DataFrame:
    """Monthly liquidity flows of projected drawings, per facility line and scenario:
    drawing = -(increase in cum_draw_pct) x undrawn (cash out, own currency); the drawn
    amount comes back as +repayment ``drawn_repay`` months later. Drawings stop at the
    availability end. Columns: commitment_code, currency, scenario, month, date, flow_type,
    amount."""
    if register is None or register.empty or models.empty:
        return pd.DataFrame(columns=["commitment_code", "currency", "scenario", "month", "date",
                                     "flow_type", "amount"])
    rd = pd.Timestamp(report_date)
    out = []
    for _, f in register.iterrows():
        mod = models[models["drawdown_model_id"] == f["drawdown_model_id"]]
        for scen, g in mod.groupby("scenario"):
            g = g.sort_values("months")
            prev = 0.0
            for _, m in g.iterrows():
                date = rd + pd.DateOffset(months=int(m["months"]))
                if pd.notna(f["availability_end_date"]) and date > f["availability_end_date"]:
                    break
                draw = (float(m["cum_draw_pct"]) - prev) * float(f["undrawn_amt"])
                prev = float(m["cum_draw_pct"])
                if draw <= 0:
                    continue
                out.append((f["commitment_code"], f["currency"], scen, int(m["months"]), date, "drawdown", -draw))
                rep = _months(m["drawn_repay"]) if isinstance(m.get("drawn_repay"), str) else 0
                if rep:
                    out.append((f["commitment_code"], f["currency"], scen, int(m["months"]) + rep,
                                date + pd.DateOffset(months=rep), "repayment", draw))
    return pd.DataFrame(out, columns=["commitment_code", "currency", "scenario", "month", "date",
                                      "flow_type", "amount"])


def inject_card_terms(bs_struct: pd.DataFrame, commit: pd.DataFrame) -> pd.DataFrame:
    """Copy the facility terms of lines linked to a drawn product into bs_structure's card_*
    columns (the card generator reads its utilisation there, labbank_common.credit_cards)."""
    bs = bs_struct.copy()
    for pc, t in terms_by_drawn_product(commit).items():
        m = bs["product_code"].astype(str) == pc
        for k in ("utilization", "undrawn_lcr", "undrawn_rsf", "undrawn_ccf"):
            bs.loc[m, "card_" + k] = getattr(t, k)
    return bs


def write_register(engine, commit: pd.DataFrame, all_tx: pd.DataFrame, total_assets_home: float,
                   to_home: dict[str, float], report_date) -> pd.DataFrame:
    """Build the register from the generated / rolled book and write schemat.commitments
    (dropped when the bank has no facilities). Drawn amounts reconcile to the linked assets."""
    with engine.begin() as c:
        c.exec_driver_sql("IF OBJECT_ID('schemat.commitments', 'U') IS NOT NULL DROP TABLE schemat.commitments;")
    if commit is None or commit.empty:
        return pd.DataFrame()
    a = all_tx[all_tx["bs_side"] == "A"]
    drawn = a.assign(pc=a["product_code"].astype(int).astype(str)).groupby("pc")["balance_amt"].sum().to_dict()
    reg = build_register(commit, drawn, total_assets_home, to_home, report_date)
    reg.to_sql("commitments", engine, schema="schemat", if_exists="replace", index=False)
    return reg
