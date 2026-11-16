"""Rate-dependent behaviour: the baseline behavioural models stay, a multiplier moves them.

The baseline models (prepayment CPR, term-deposit redemption TDRR, NMD core profile, facility
drawings) are not replaced. Under a rate scenario each one is scaled by a multiplier gamma that
depends on how much the rate moved at the product's own reference tenor:

    x      = direction x delta_bp            delta_bp = scenario shock at ref_tenor
    x'     = sign(x) x max(|x| - deadband_bp, 0)
    h      = (2 / pi) x arctan(|x'| / scale_bp)              0 <= h < 1, the S-curve
    gamma  = 1 + (gamma_max - 1) x h         when x > 0
    gamma  = 1 - (1 - gamma_min) x h         when x < 0

The arctangent is the shape of the Richard-Roll refinancing S-curve: close to linear for small
moves, saturating for large ones. gamma(0) = 1 exactly, so a zero shock reproduces the baseline.

Relation to the EBA multipliers (Reg. (EU) 2024/857): the RTS uses a fixed 0.8 / 1.2 for any
scenario. Here h(scale_bp) = 0.5, so gamma_max 1.4, gamma_min 0.6, scale_bp 200 gives 1.2 / 0.8
at a 200 bp move and less for a smaller one: a 90 bp rise at 10Y in the steepener is 0.89, not 0.8.

Input: inputs/<bank>/models/rate_sensitivity.xlsx, one row per product and behaviour:
  product_code   product or commitment code
  behaviour      CPR | TDRR | NMD_CORE | DRAW
  ref_tenor      tenor whose rate move drives it (10Y for a 10Y fixed mortgage, 6M for a 6M deposit)
  direction      +1 the behaviour grows when rates rise, -1 when they fall
  gamma_min, gamma_max, scale_bp, deadband_bp
  ref_rate_pct   NMD_LEVEL only: the rate level of the profile file

What gamma multiplies:
  CPR       the annual prepayment rate (as the RTS does)
  TDRR      the cumulative redemption rate
  NMD_CORE  the hazard of the core profile: S(t) -> S(t) ** gamma, so S(0) stays 1
  DRAW      the cumulative drawn share of a facility
  NMD_LEVEL the baseline core profile itself, by the rate level of the report date (below)

Rate level (NMD_LEVEL). The shock multipliers measure a move away from the report date. A profile
calibrated at one rate level does not hold at another: in a roll-forward that ends at a 4% 3M rate
the core of a current account is shorter than at 2.6%. A NMD_LEVEL row holds the long-run response
and ref_rate_pct, the 3M rate at which the profile file holds. The profile at a date with rate r is

    S(t | r) = S_ref(t) ** gamma(r - ref_rate_pct)

and roll_nmd_profile moves a profile from one date to the next. The shock multipliers then apply on
top of that date's profile, so a shock is always a move from the date's own level. NMD_LEVEL never
enters a scenario variant; it changes only the inputs a roll-forward writes.

Multipliers are rounded to GRID so that a few CF variants serve every scenario; a nonzero move can
round to 1.0, which then simply keeps the baseline. NMD_CORE has a finer grid: its calibrated multipliers
are close to 1, and few rows carry them.
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from labbank_common.bank_config import PROJECT_ROOT, BankConfig, get_bank

MODEL_FILE = "rate_sensitivity.xlsx"
BEHAVIOURS = {"CPR", "TDRR", "NMD_CORE", "DRAW", "NMD_LEVEL"}
LEVEL = "NMD_LEVEL"
PTYPE = {"CPR": "L", "TDRR": "D", "NMD_CORE": "D"}     # product_type of the CF rows it scales
GRID = 0.05
FINE_GRID = {"NMD_CORE": 0.01}
# stress scenarios; the +-1 bp own scenarios measure a sensitivity and keep the baseline
SCENARIOS = ("par_up", "par_dn", "steep", "flat", "sr_up", "sr_dn", "own", "own_100_up", "own_100_dn")
_TENOR = re.compile(r"^\s*(\d+)\s*([DMY])\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Sensitivity:
    product_code: str
    behaviour: str
    ref_tenor: str
    direction: int
    gamma_min: float
    gamma_max: float
    scale_bp: float
    deadband_bp: float = 0.0
    ref_rate_pct: float | None = None
    migrate_to: str | None = None
    migrate_months: float | None = None

    def __post_init__(self):
        if self.behaviour not in BEHAVIOURS:
            raise ValueError(f"{self.product_code}: unknown behaviour {self.behaviour!r}")
        if self.direction not in (1, -1):
            raise ValueError(f"{self.product_code}: direction must be +1 or -1")
        if not 0.0 <= self.gamma_min <= 1.0 <= self.gamma_max:
            raise ValueError(f"{self.product_code}: need 0 <= gamma_min <= 1 <= gamma_max")
        if self.scale_bp <= 0 or self.deadband_bp < 0:
            raise ValueError(f"{self.product_code}: scale_bp must be > 0, deadband_bp >= 0")
        tenor_years(self.ref_tenor)
        if self.behaviour == LEVEL and self.ref_rate_pct is None:
            raise ValueError(f"{self.product_code}: NMD_LEVEL needs ref_rate_pct")
        if self.migrate_to is not None and self.behaviour != "NMD_CORE":
            raise ValueError(f"{self.product_code}: migrate_to belongs to an NMD_CORE row")


def tenor_years(tenor: str) -> float:
    m = _TENOR.match(str(tenor))
    if not m:
        raise ValueError(f"bad tenor {tenor!r}")
    n, unit = int(m[1]), m[2].upper()
    return {"D": n / 365.0, "M": n / 12.0, "Y": float(n)}[unit]


def gamma(delta_bp: float, s: Sensitivity) -> float:
    """Multiplier on the baseline behaviour for a rate move of delta_bp at the reference tenor."""
    x = s.direction * float(delta_bp)
    size = max(abs(x) - s.deadband_bp, 0.0)
    h = (2.0 / math.pi) * math.atan(size / s.scale_bp)
    return 1.0 + (s.gamma_max - 1.0) * h if x > 0 else 1.0 - (1.0 - s.gamma_min) * h


def on_grid(g: float, behaviour: str | None = None) -> float:
    step = FINE_GRID.get(behaviour, GRID)
    return round(round(g / step) * step, 10)


def scale_survival(outstanding, g: float) -> np.ndarray:
    """NMD core profile under a multiplier on its hazard: S(t) ** g. S = 1 stays 1, S = 0 stays 0."""
    return np.power(np.clip(np.asarray(outstanding, dtype=float), 0.0, 1.0), g)


def read_models(bank: BankConfig | None = None) -> list[Sensitivity]:
    """Rows of rate_sensitivity.xlsx; [] for a bank without the file."""
    path = (bank or get_bank()).models_dir() / MODEL_FILE
    if not path.exists():
        return []
    return models_from_frame(pd.read_excel(path))


def models_from_frame(df: pd.DataFrame) -> list[Sensitivity]:
    ref = df["ref_rate_pct"] if "ref_rate_pct" in df else pd.Series(np.nan, index=df.index)
    mig = df["migrate_to"] if "migrate_to" in df else pd.Series(np.nan, index=df.index)
    mmo = df["migrate_months"] if "migrate_months" in df else pd.Series(np.nan, index=df.index)
    return [Sensitivity(str(int(r.product_code)), str(r.behaviour).strip().upper(), str(r.ref_tenor),
                        int(r.direction), float(r.gamma_min), float(r.gamma_max), float(r.scale_bp),
                        0.0 if pd.isna(r.deadband_bp) else float(r.deadband_bp),
                        None if pd.isna(rr) else float(rr), None if pd.isna(mm) else str(int(mm)),
                        None if pd.isna(mo) else float(mo))
            for r, rr, mm, mo in zip(df.itertuples(), ref, mig, mmo)]


def migration_targets(bank: BankConfig | None = None) -> dict[str, tuple[str, float | None]]:
    """{NMD product: (product its rate-driven outflow moves to, renewal months or None)}
    from the NMD_CORE rows with migrate_to; None takes the tenor from the target's book."""
    return {s.product_code: (s.migrate_to, s.migrate_months) for s in read_models(bank)
            if s.behaviour == "NMD_CORE" and s.migrate_to is not None}


def scenario_shock_bp(scenario_id: str, tenor: str, currency: str) -> float:
    """Regulatory scenario shock at a tenor, in bp."""
    py = str(PROJECT_ROOT / "irrbb_calc" / "python_code")
    if py not in sys.path:
        sys.path.append(py)
    import eba_shock_curves as esc
    t = np.array([tenor_years(tenor)])
    return float(esc.shock_bps_at_tenors(t, scenario_id, esc.get_shock_params(currency))[0])


@lru_cache(maxsize=None)
def _table(bank_id: str) -> dict:
    bank = get_bank(bank_id)
    out: dict[str, dict[tuple[str, str], float]] = {}
    for s in read_models(bank):
        if s.behaviour == LEVEL:
            continue
        for scen in SCENARIOS:
            g = on_grid(gamma(scenario_shock_bp(scen, s.ref_tenor, bank.market.ccy), s), s.behaviour)
            out.setdefault(scen, {})[(s.behaviour, s.product_code)] = g
    return out


def gamma_table(bank: BankConfig | None = None) -> dict[str, dict[tuple[str, str], float]]:
    """scenario_id -> {(behaviour, product_code): gamma on the grid}."""
    return _table((bank or get_bank()).bank_id)


def scenario_gammas(scenario_id: str, bank: BankConfig | None = None) -> dict[tuple[str, str], float]:
    """{(product_type, product_code): gamma} of the CF behaviours in a scenario, 1.0 left out.
    LABBANK_RATE_BEHAVIOURS=CPR,TDRR limits a run to some behaviours, to attribute the result."""
    t = gamma_table(bank).get(scenario_id, {})
    only = os.environ.get("LABBANK_RATE_BEHAVIOURS")
    keep = set(only.upper().split(",")) if only else set(PTYPE)
    return {(PTYPE[b], c): g for (b, c), g in t.items() if b in PTYPE and b in keep and g != 1.0}


def nmd_codes(bank: BankConfig | None = None) -> set[str]:
    """Products with an NMD_CORE multiplier: their deposit variants are not term deposits."""
    return {c for t in gamma_table(bank).values() for (b, c) in t if b == "NMD_CORE"}


def variant_gammas(bank: BankConfig | None = None, behaviour: str | None = None) -> tuple[float, ...]:
    """Multipliers any scenario needs for a behaviour (all CF behaviours if None), 1.0 left out."""
    vals = {g for t in gamma_table(bank).values() for (b, _), g in t.items()
            if g != 1.0 and (b in PTYPE if behaviour is None else b == behaviour)}
    return tuple(sorted(vals))


def product_variant_gammas(code: str, behaviour: str, bank: BankConfig | None = None) -> tuple[float, ...]:
    """Multipliers one product needs across the scenarios, 1.0 left out."""
    vals = {t[(behaviour, str(code))] for t in gamma_table(bank).values() if (behaviour, str(code)) in t}
    return tuple(sorted(v for v in vals if v != 1.0))


def market_rate_pct(fixings: pd.DataFrame, date, tenor: str = "3M") -> float:
    """Mid fixing at a tenor on the last date up to `date` (fixing_input.xlsx layout)."""
    f = fixings[(fixings["tenor"] == tenor) & (pd.to_datetime(fixings["date"]) <= pd.Timestamp(date))]
    if f.empty:
        raise ValueError(f"no {tenor} fixing up to {date}")
    last = pd.to_datetime(f["date"]).max()
    return float(f.loc[pd.to_datetime(f["date"]) == last, "rate"].mean())


def level_gamma(s: Sensitivity, rate_pct: float) -> float:
    """Multiplier of the profile at a rate level against the level the profile was set at."""
    return gamma((rate_pct - s.ref_rate_pct) * 100.0, s)


def roll_nmd_profile(profile: pd.DataFrame, models: list[Sensitivity], rate_from: float,
                     rate_to: float) -> tuple[pd.DataFrame, dict[str, float]]:
    """Profile file (tenor x product columns) moved from a date at rate_from to one at rate_to:
    S_to = S_from ** (gamma(rate_to) / gamma(rate_from)), both against ref_rate_pct. Products
    without a NMD_LEVEL row keep their profile. Returns the new profile and the exponents."""
    out, used = profile.copy(), {}
    cols = {str(c): c for c in profile.columns}
    for s in models:
        if s.behaviour != LEVEL or s.product_code not in cols:
            continue
        g = level_gamma(s, rate_to) / level_gamma(s, rate_from)
        c = cols[s.product_code]
        out[c] = np.where(profile[c].isna(), np.nan, scale_survival(profile[c].fillna(0.0), g))
        used[s.product_code] = g
    return out, used


def mode(bank: BankConfig | None = None) -> str:
    """Behaviour mode of the run: baseline | flat | eba | rate. LABBANK_BEHAVIOUR overrides the bank's
    setting, so one CF run can be measured under each mode. flat = baseline redemption in the CFs,
    no multiplier in any scenario."""
    return os.environ.get("LABBANK_BEHAVIOUR") or (bank or get_bank()).behaviour
