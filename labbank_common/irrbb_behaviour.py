"""Scenario-dependent behavioural multipliers for IRRBB shock scenarios.

Commission Delegated Regulation (EU) 2024/857 (RTS on the IRRBB standardised
approach) scales two baseline behavioural rates per shock scenario:

  Art. 9(3)  conditional prepayment rate (CPR) of retail fixed-rate loans
             x0.8 where rates rise   -- parallel up, steepener, short up
             x1.2 where rates fall   -- parallel down, flattener, short down
  Art. 10(5) cumulative term deposit redemption rate (TDRR) of retail
             fixed-rate term deposits
             x1.2 where short rates rise -- parallel up, flattener, short up
             x0.8 where short rates fall -- parallel down, steepener, short down

Steepener / flattener get opposite multipliers for the two products: a
steepener lifts long rates (fewer mortgage refinancings) but cuts short rates
(less reason to break a short deposit and re-deposit).

Scope (which contracts are scaled) is data-driven, not listed here:
  - loans:    rate_type 'F' with a prepayment model in bs.models_loan
  - deposits: rate_type 'F' term deposits with a row in bs.models_deposit_tdrr
Floating-rate loans keep their baseline CPR in every scenario (Art. 9 covers
fixed-rate loans only).

The ±1 bp own scenarios are PV01 / NII01 sensitivities, not stress scenarios:
a 20% behavioural jump on a 1 bp move would swamp the rate effect, so they
keep the baseline rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from labbank_common import rate_behaviour as rb

# ── Master switch ─────────────────────────────────────────────────────────────
# False: every scenario uses the baseline behaviour (as before this layer existed) and
#        term deposits keep their contractual bullet in cf.products.
# True:  term deposits get the baseline early redemption (TDRR) in cf.products, and
#        EVE / NII / SOT swap in the CPR / TDRR x gamma variant per shock scenario.
# The variants are computed and stored in cf.products_scen either way.  After flipping
# it, rerun from the cash-flow step (Dagster recalc_from_sql_job), not just IRRBB --
# the baseline term-deposit schedule in cf.products depends on it.
APPLY_EBA_BEHAVIOUR: bool = False

# The two non-baseline multipliers -- the CF engine pre-computes one variant per value.
VARIANT_GAMMAS: tuple[float, ...] = (0.8, 1.2)

# scenario_id -> multiplier on the baseline annual CPR (Art. 9(3))
CPR_GAMMA: dict[str, float] = {
    "par_up": 0.8, "par_dn": 1.2,
    "steep":  0.8, "flat":   1.2,
    "sr_up":  0.8, "sr_dn":  1.2,
    "own":        1.2,           # eve/nii own scenario = -100 bp parallel (OWN_SHOCK_BPS)
    "own_100_up": 0.8, "own_100_dn": 1.2,
}

# scenario_id -> multiplier on the baseline cumulative TDRR (Art. 10(5))
TDRR_GAMMA: dict[str, float] = {
    "par_up": 1.2, "par_dn": 0.8,
    "steep":  0.8, "flat":   1.2,
    "sr_up":  1.2, "sr_dn":  0.8,
    "own":        0.8,
    "own_100_up": 1.2, "own_100_dn": 0.8,
}


def scenario_multipliers_on() -> bool:
    """Scenarios use behavioural multipliers (EBA or rate-dependent) in this run."""
    return APPLY_EBA_BEHAVIOUR or rb.mode() in ("eba", "rate")


def baseline_redemption_on() -> bool:
    """Term deposits carry their baseline early redemption in cf.products."""
    return APPLY_EBA_BEHAVIOUR or rb.mode() in ("flat", "eba", "rate")


def cf_variant_gammas(behaviour: str) -> tuple[float, ...]:
    """Multipliers the CF engine pre-computes: the EBA pair, plus the ones the bank's
    rate-dependent models reach (labbank_common/rate_behaviour.py)."""
    return tuple(sorted(set(VARIANT_GAMMAS) | set(rb.variant_gammas(behaviour=behaviour))))


def gammas_for(scenario_id: str) -> tuple[float, float]:
    """(CPR multiplier, TDRR multiplier) for a scenario; 1.0 = baseline."""
    return CPR_GAMMA.get(scenario_id, 1.0), TDRR_GAMMA.get(scenario_id, 1.0)


def scale_period_cpr(period_rate, yf, gamma: float):
    """Scale a per-period prepayment rate by gamma on its ANNUAL equivalent.

    The RTS multiplies the annual CPR; bs.models_loan stores a per-period
    (monthly) rate.  annual = 1 - (1 - p)^(1/yf);  p' = 1 - (1 - min(1, gamma*annual))^yf.
    """
    p  = np.asarray(period_rate, dtype=float)
    yf = np.asarray(yf, dtype=float)
    safe_yf = np.where(yf > 0, yf, 1.0)
    annual  = 1.0 - np.power(1.0 - np.clip(p, 0.0, 1.0), 1.0 / safe_yf)
    scaled  = np.minimum(1.0, gamma * annual)
    out = 1.0 - np.power(1.0 - scaled, safe_yf)
    return np.where(yf > 0, out, p)


def scale_tdrr(tdrr: float, gamma: float) -> float:
    """Scenario TDRR = min(1, gamma * baseline)."""
    return min(1.0, gamma * float(tdrr))


def nmd_migration(base_df: pd.DataFrame, shocked_df: pd.DataFrame,
                  targets: dict[str, tuple[str, float | None]] | None = None) -> pd.DataFrame:
    """Mark the part of an NMD's capital that leaves only because of the scenario.

    In a scenario the NMD variant (S^gamma) repays more capital inside the horizon than the
    baseline profile. EVE treats that money as gone; under a constant balance sheet it stays in
    the bank, but in the product clients move to (rate_sensitivity.xlsx, migrate_to: current
    account -> term deposit), not in the account it left. NII renews it as that product.

    Adds to shocked_df: migr_capital (scenario capital - baseline capital of the same cash flow
    of the same schedule; signed: a slower outflow gives a negative one, money that stays instead
    of going to the target), migr_product and migr_months (the target's renewal tenor: migrate_months,
    else the balance-weighted repricing period of its book). Without targets, unchanged."""
    if targets is None:
        targets = rb.migration_targets()
    if not targets or shocked_df is base_df:
        return shocked_df
    out = shocked_df.copy()
    out["migr_capital"] = 0.0
    out["migr_product"] = None
    out["migr_months"] = np.nan
    key = ["schedule_id", "product_type", "cf_end_dt"]
    cap = lambda d: d["capital_pmt"].fillna(0.0) + d["prepayment_pmt"].fillna(0.0)   # noqa: E731
    codes = out["product_code"].astype(str)
    for src, (dst, months) in targets.items():
        m = codes == src
        if not m.any():
            continue
        if months is None:
            tgt = base_df[base_df["product_code"].astype(str) == dst]
            bal = tgt["outstanding_bal"].fillna(0.0).abs() if "renew_months" in tgt else pd.Series(dtype=float)
            if bal.sum() <= 0:
                raise ValueError(f"NMD migration {src} -> {dst}: no migrate_months and no {dst} book "
                                 f"to take a tenor from")
            months = float(np.average(tgt["renew_months"].fillna(0.0), weights=bal))
        b = base_df[base_df["product_code"].astype(str) == src]
        base_cap = b.assign(_c=cap(b)).groupby(key)["_c"].sum()
        s = out.loc[m]
        s_cap = s.assign(_c=cap(s)).groupby(key)["_c"].sum()
        diff = s_cap.sub(base_cap.reindex(s_cap.index).fillna(0.0))
        n = s.groupby(key)["capital_pmt"].transform("size")
        idx = pd.MultiIndex.from_frame(s[key])
        out.loc[m, "migr_capital"] = diff.reindex(idx).to_numpy() / n.to_numpy()
        out.loc[m, "migr_product"] = dst
        out.loc[m, "migr_months"] = months
    return out


def apply_variant(base_df: pd.DataFrame, scen_df: pd.DataFrame, scenario_id: str,
                  enabled: bool | None = None) -> pd.DataFrame:
    """Swap in the scenario's behavioural CF variant for the contracts in scope.

    base_df  -- CF rows from cf.products (baseline behaviour), any loader layout
    scen_df  -- rows from cf.products_scen in the SAME layout plus a 'gamma' column
    Every (schedule_id, product_type) present in scen_df is in scope: its base rows
    are dropped and replaced by the scen_df rows whose gamma matches the scenario.
    enabled=None follows scenario_multipliers_on(); when off, base_df is returned unchanged.
    In rate mode the multiplier is per product (rate_behaviour.scenario_gammas), else per type.
    """
    if enabled is None:
        enabled = scenario_multipliers_on()
    if not enabled or scen_df is None or scen_df.empty:
        return base_df
    if rb.mode() == "rate":
        codes = scen_df["product_code"].astype(str)
        groups = [(scen_df[(scen_df["product_type"] == t) & (codes == c)], g)
                  for (t, c), g in rb.scenario_gammas(scenario_id).items()]
    else:
        g_cpr, g_tdrr = gammas_for(scenario_id)
        groups = [(scen_df[scen_df["product_type"] == t], g) for t, g in (("L", g_cpr), ("D", g_tdrr)) if g != 1.0]
        nmd = rb.nmd_codes()            # the RTS multipliers cover term deposits, not NMD variants
        if nmd:
            groups = [(d[~d["product_code"].astype(str).isin(nmd)], g) for d, g in groups]
    pieces = []
    drop_keys = []
    for typ, g in groups:
        if typ.empty:
            continue
        ptype = typ["product_type"].iloc[0]
        drop_keys.append(typ["schedule_id"].astype(str).drop_duplicates() + "|" + ptype)
        pieces.append(typ[np.isclose(typ["gamma"].astype(float), g)].drop(columns="gamma"))
    if not pieces:
        return base_df
    keys = pd.concat(drop_keys).unique()
    base_keys = base_df["schedule_id"].astype(str) + "|" + base_df["product_type"].astype(str)
    kept = base_df[~base_keys.isin(keys)]
    return pd.concat([kept] + pieces, ignore_index=True)
