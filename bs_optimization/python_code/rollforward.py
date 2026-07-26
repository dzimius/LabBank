"""rollforward.py
=================
Phase B — monthly balance-sheet rollforward simulator.

All 5 pieces implemented:

    piece 1 — state model + monthly loop skeleton: ages the t=0 book along the
              existing per-cohort 12-month schedule (cohort_outstanding_m)
    piece 2 — growth engine: fixed-% default (replenish month-0 balance) or an
              explicit RollforwardConfig.growth_targets override, spun off as
              a new synthetic cohort per product per month
    piece 3 — new-cohort pricing: prices this month's unpriced synthetic/plug
              cohorts with the exact new-business renewal-rate formula
              (clip(coeff_a*fwd_1m + coeff_b, floor, cap); see
              extract_params._renewal_rate_matrix / _fwd_F)
    piece 4 — balance-sheet plug: bonds (50/50 fixed 3000 / float 3100) on
              assets when funding exceeds assets, "other deposits" (9000,
              new product) @ placeholder 1% on funding when assets exceed
              funding -- closes the identity exactly every month
    piece 5 — P&L rollup: retained earnings = monthly NII only (no EL/CoC/
              opex), fed back into the 5300 retained_earnings equity cohort

Horizon is capped at 12 months this phase because cohort_outstanding_m only
covers 12 months forward from report_date — see memory
project_bs_rollforward_phaseb.md for the full design record. Limits
(LCR/NSFR/RWA/SOT) are whole-balance-sheet only and are NOT enforced here;
a rolled-forward book is allowed to breach them, that's a diagnostic output
for a later phase, not something this module guards against.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPTPREP = os.path.normpath(os.path.join(_HERE, "..", "..", "optimize_prep", "python_code"))
for _p in (_HERE, _OPTPREP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bs_vector import BalanceSheetParams, CohortRates, CurveTensors

MAX_SCHEDULE_MONTHS = 12

# IRS notional is a hedging instrument, not organically grown/replenished by
# the business-forecast growth engine, and also excluded from the piece-5 NII
# rollup (swap NII isn't modeled here). Equity (bs_side 'E') is excluded too --
# it's driven by retained earnings / capital actions (piece 5), not growth.
_EXCLUDED_GROWTH_PRODUCT_CODES = {"0000"}

# Piece 4 plug products. Bonds (3000 fixed / 3100 float) already exist in the
# base catalog and get priced through the normal new-business formula.
# "Other deposits" (9000) does not exist in the base catalog -- placeholder
# flat rate per the user's explicit call, to be refined later.
_PLUG_ASSET_PRODUCTS = (("3000", "F"), ("3100", "V"))
_PLUG_LIABILITY_PRODUCT_CODE = "9000"
_PLUG_LIABILITY_PRODUCT_NAME = "other_deposits"
_PLUG_LIABILITY_RATE = 0.01
_PLUG_CCY = "PLN"

_RETAINED_EARNINGS_PRODUCT_CODE = "5300"


@dataclass
class RollforwardConfig:
    horizon_months: int = 12
    # (product_code, bs_side, currency) -> array of length horizon_months,
    # absolute PLN target balance for month 1..horizon_months. Products with
    # no entry here default to the "fixed % of total" behavior: replenish
    # back to their month-0 balance every month.
    growth_targets: dict[tuple[str, str, str], np.ndarray] = field(default_factory=dict)


@dataclass
class CohortState:
    cohort_id: str
    product_code: str
    product_name: str
    bs_side: str            # 'A' | 'L' | 'E'
    currency: str
    sign: float
    rate_type: str          # 'F' | 'V' | 'A' | ''
    coupon_rate: float
    origin: str              # 'base' | 'synthetic' | 'plug'
    vintage_month: int       # rollforward month index this cohort was created; 0 for base book
    base_idx: int | None     # row index into the source BalanceSheetParams (base cohorts only)
    balance: float


@dataclass
class RollforwardState:
    month: int
    cohorts: list[CohortState]
    retained_earnings: float = 0.0

    def totals(self) -> dict[str, float]:
        assets = sum(c.balance for c in self.cohorts if c.bs_side == "A")
        liabilities = sum(c.balance for c in self.cohorts if c.bs_side == "L")
        equity = sum(c.balance for c in self.cohorts if c.bs_side == "E")
        return {
            "assets": assets,
            "liabilities": liabilities,
            "equity": equity,
            "identity_gap": assets - (liabilities + equity),
        }


def build_initial_state(params: BalanceSheetParams) -> RollforwardState:
    n = len(params.cohort_id)
    cohorts = [
        CohortState(
            cohort_id=str(params.cohort_id[i]),
            product_code=str(params.product_code[i]),
            product_name=str(params.product_name[i]),
            bs_side=str(params.bs_side[i]),
            currency=str(params.currency[i]),
            sign=float(params.sign[i]),
            rate_type=str(params.rate_type[i]),
            coupon_rate=float(params.coupon_rate[i]),
            origin="base",
            vintage_month=0,
            base_idx=i,
            balance=float(params.balance_arr[i]),
        )
        for i in range(n)
    ]
    return RollforwardState(month=0, cohorts=cohorts, retained_earnings=0.0)


def _age_base_cohorts(state: RollforwardState, params: BalanceSheetParams) -> RollforwardState:
    next_month = state.month + 1
    if next_month > MAX_SCHEDULE_MONTHS:
        raise ValueError(
            f"base-cohort schedule only covers {MAX_SCHEDULE_MONTHS} months "
            f"(cohort_outstanding_m); cannot age to month {next_month} without "
            "a schedule-extension engine (out of scope for piece 1)"
        )
    aged = []
    for c in state.cohorts:
        # cohort_outstanding_m is a real balance-decay schedule only for
        # is_cohort=True instrument rows (loans/mortgages/bonds/term deposits).
        # For single-row products (equity, NMDs, cash, IRS) it's zeros(12) by
        # construction -- an NII rate-sensitivity weight, not a balance
        # schedule -- so those rows hold their balance flat in piece 1.
        if c.origin == "base" and c.base_idx is not None and bool(params.is_cohort[c.base_idx]):
            frac = float(params.cohort_outstanding_m[c.base_idx, next_month - 1])
            new_balance = float(params.balance_arr[c.base_idx]) * frac
        else:
            new_balance = c.balance
        aged.append(replace(c, balance=new_balance))
    return RollforwardState(month=next_month, cohorts=aged, retained_earnings=state.retained_earnings)


def _base_product_totals(params: BalanceSheetParams) -> dict[tuple[str, str, str], float]:
    totals: dict[tuple[str, str, str], float] = {}
    for i in range(len(params.cohort_id)):
        if str(params.bs_side[i]) == "E" or str(params.product_code[i]) in _EXCLUDED_GROWTH_PRODUCT_CODES:
            continue
        key = (str(params.product_code[i]), str(params.bs_side[i]), str(params.currency[i]))
        totals[key] = totals.get(key, 0.0) + float(params.balance_arr[i])
    return totals


def _representative_base_idx(params: BalanceSheetParams, key: tuple[str, str, str]) -> int | None:
    for i in range(len(params.cohort_id)):
        if (str(params.product_code[i]), str(params.bs_side[i]), str(params.currency[i])) == key:
            return i
    return None


def _apply_growth(state: RollforwardState, params: BalanceSheetParams,
                   config: RollforwardConfig) -> RollforwardState:
    """Piece 2: fixed-% default (replenish back to month-0 balance) or Excel
    growth index, whichever is higher, spun off as a new synthetic cohort for
    this month's shortfall. Only handles growth (target > current); a target
    below current balance is a no-op -- there's no shrink/paydown mechanism
    yet, matching the design as described (see memory
    project_bs_rollforward_phaseb.md)."""
    base_totals = _base_product_totals(params)

    current_totals: dict[tuple[str, str, str], float] = {}
    for c in state.cohorts:
        if c.bs_side == "E" or c.product_code in _EXCLUDED_GROWTH_PRODUCT_CODES:
            continue
        key = (c.product_code, c.bs_side, c.currency)
        current_totals[key] = current_totals.get(key, 0.0) + c.balance

    new_cohorts = list(state.cohorts)
    for key, base_balance in base_totals.items():
        pc, side, ccy = key
        index_path = config.growth_targets.get(key)
        target = float(index_path[state.month - 1]) if index_path is not None else base_balance
        gap = target - current_totals.get(key, 0.0)
        if gap <= 0:
            continue
        rep_idx = _representative_base_idx(params, key)
        rate_type = str(params.rate_type[rep_idx]) if rep_idx is not None else ""
        sign = float(params.sign[rep_idx]) if rep_idx is not None else (1.0 if side == "A" else -1.0)
        product_name = str(params.product_name[rep_idx]) if rep_idx is not None else pc
        new_cohorts.append(CohortState(
            cohort_id=f"{pc}_{side}_{ccy}_synth_{state.month:02d}",
            product_code=pc,
            product_name=product_name,
            bs_side=side,
            currency=ccy,
            sign=sign,
            rate_type=rate_type,
            coupon_rate=float("nan"),   # unpriced -- piece 3 assigns the contract rate
            origin="synthetic",
            vintage_month=state.month,
            base_idx=None,
            balance=gap,
        ))
    return replace(state, cohorts=new_cohorts)


def _fwd_1m(disc_curve: np.ndarray, event_t: int) -> float:
    """1-month forward rate starting at event_t (0-indexed months from
    report_date). Exact replica of extract_params._fwd_F(disc, event_t, F=1),
    the same helper that builds renewal_rate_matrix -- kept as a standalone
    formula here rather than importing extract_params, which pulls in the
    full SQL-backed cohort-extraction pipeline."""
    n_m = len(disc_curve)
    i_s = max(0, min(event_t - 1, n_m - 1)) if event_t > 0 else -1
    i_e = max(0, min(event_t, n_m - 1))
    df_s = disc_curve[i_s] if i_s >= 0 else 1.0
    df_e = max(disc_curve[i_e], 1e-12)
    return (df_s / df_e - 1.0) * 12.0


def _price_new_cohorts(state: RollforwardState, params: BalanceSheetParams,
                        curves: CurveTensors, config: RollforwardConfig) -> RollforwardState:
    """Piece 3: price this month's unpriced synthetic cohorts (coupon_rate=NaN)
    with the exact new-business renewal-rate formula already used for renewal
    NII: clip(coeff_a * fwd_1m + coeff_b, client_floor, client_cap), evaluated
    on the base-scenario curve at this cohort's origination month. See
    extract_params._renewal_rate_matrix -- same coeff_a/coeff_b/floor/cap per
    product, same _fwd_F(disc, event_t=m, F=1) 1-month-forward convention."""
    m_idx = state.month - 1
    priced = []
    for c in state.cohorts:
        if c.origin in ("synthetic", "plug") and np.isnan(c.coupon_rate):
            key = (c.product_code, c.bs_side, c.currency)
            rep_idx = _representative_base_idx(params, key)
            if rep_idx is None:
                priced.append(c)
                continue
            ca = float(params.coeff_a[rep_idx])
            cb = float(params.coeff_b[rep_idx])
            fl = float(params.client_floor[rep_idx]) if np.isfinite(params.client_floor[rep_idx]) else -np.inf
            cp = float(params.client_cap[rep_idx]) if np.isfinite(params.client_cap[rep_idx]) else np.inf
            base_disc = curves.get_disc_curve("base", c.currency)
            fwd = _fwd_1m(base_disc, m_idx)
            rate = float(np.clip(ca * fwd + cb, fl, cp))
            c = replace(c, coupon_rate=rate)
        priced.append(c)
    return replace(state, cohorts=priced)


def _apply_plug(state: RollforwardState, params: BalanceSheetParams,
                 config: RollforwardConfig) -> RollforwardState:
    """Piece 4: close the balance-sheet identity exactly, every month. Assets
    in excess of funding get deployed into bonds (50/50 fixed 3000 / float
    3100, real market-rate pricing via _price_new_cohorts). Funding in excess
    of assets gets raised via "other deposits" (9000, not in the base
    catalog) at a flat 1% placeholder rate. Limits (LCR/NSFR/RWA/SOT) are NOT
    checked here -- a plug that breaches them is a valid diagnostic finding
    about the business forecast, not something to guard against (see memory
    project_bs_rollforward_phaseb.md). Plug cohorts are replaced, not
    accumulated, every month -- they represent this month's residual
    funding/investment action, not a running chain of originations like the
    growth engine's synthetic cohorts."""
    non_plug = [c for c in state.cohorts if c.origin != "plug"]
    assets = sum(c.balance for c in non_plug if c.bs_side == "A")
    liabilities = sum(c.balance for c in non_plug if c.bs_side == "L")
    equity = sum(c.balance for c in non_plug if c.bs_side == "E")
    gap = assets - (liabilities + equity)

    plug_cohorts: list[CohortState] = []
    if gap > 0:
        plug_cohorts.append(CohortState(
            cohort_id=f"{_PLUG_LIABILITY_PRODUCT_CODE}_L_{_PLUG_CCY}_plug",
            product_code=_PLUG_LIABILITY_PRODUCT_CODE,
            product_name=_PLUG_LIABILITY_PRODUCT_NAME,
            bs_side="L", currency=_PLUG_CCY, sign=-1.0, rate_type="V",
            coupon_rate=_PLUG_LIABILITY_RATE,
            origin="plug", vintage_month=state.month, base_idx=None,
            balance=gap,
        ))
    elif gap < 0:
        half = abs(gap) / 2.0
        for pc, rt in _PLUG_ASSET_PRODUCTS:
            rep_idx = _representative_base_idx(params, (pc, "A", _PLUG_CCY))
            product_name = str(params.product_name[rep_idx]) if rep_idx is not None else pc
            plug_cohorts.append(CohortState(
                cohort_id=f"{pc}_A_{_PLUG_CCY}_plug",
                product_code=pc, product_name=product_name,
                bs_side="A", currency=_PLUG_CCY, sign=1.0, rate_type=rt,
                coupon_rate=float("nan"),   # priced via real bond coefficients in _price_new_cohorts
                origin="plug", vintage_month=state.month, base_idx=None,
                balance=half,
            ))
    return replace(state, cohorts=non_plug + plug_cohorts)


def _update_pnl(state: RollforwardState, params: BalanceSheetParams,
                 cr: CohortRates) -> RollforwardState:
    """Piece 5: monthly NII rollup -> retained earnings. NII only -- EL and
    CoC are product-comparison metrics elsewhere in the book, not real cash
    costs, and are deliberately excluded (see project_bs_rollforward_phaseb
    memory). No opex either, since no such assumption exists yet in
    BalanceSheetParams. Equity and IRS (product 0000) don't contribute to
    NII and are skipped. Base cohorts use the exact per-month client rate
    from cr.rate_matrix (correctly reflects fixed-coupon vs. floating
    repricing); synthetic/plug cohorts use the rate fixed at origination
    (piece 3/4) -- they don't reprice again after creation, a known
    simplification of this phase."""
    m_idx = state.month - 1
    base_scen = cr.scenario_index("base")
    monthly_nii = 0.0
    for c in state.cohorts:
        if c.bs_side == "E" or c.product_code in _EXCLUDED_GROWTH_PRODUCT_CODES:
            continue
        if c.origin == "base" and c.base_idx is not None:
            rate = float(cr.rate_matrix[c.base_idx, m_idx, base_scen])
        else:
            rate = c.coupon_rate
        monthly_nii += c.sign * c.balance * rate / 12.0

    cum_retained = state.retained_earnings + monthly_nii
    updated = []
    for c in state.cohorts:
        if (c.product_code == _RETAINED_EARNINGS_PRODUCT_CODE and c.bs_side == "E"
                and c.base_idx is not None):
            c = replace(c, balance=float(params.balance_arr[c.base_idx]) + cum_retained)
        updated.append(c)
    return replace(state, cohorts=updated, retained_earnings=cum_retained)


def step_month(state: RollforwardState, params: BalanceSheetParams, cr: CohortRates,
                curves: CurveTensors, config: RollforwardConfig) -> RollforwardState:
    # NII first, on cohorts as aged (pre-growth/plug) -- this month's new
    # originations start earning/paying interest next month, once they're
    # "existing" cohorts. Plug runs LAST so it closes the identity against
    # the final month-end totals, including the equity bump from retained
    # earnings; running it any earlier reopens a gap the size of that
    # month's NII (equity moved with no offsetting asset/liability entry).
    state = _age_base_cohorts(state, params)
    state = _update_pnl(state, params, cr)
    state = _apply_growth(state, params, config)
    state = _apply_plug(state, params, config)
    state = _price_new_cohorts(state, params, curves, config)
    return state


def run_rollforward(params: BalanceSheetParams, cr: CohortRates, curves: CurveTensors,
                     config: RollforwardConfig | None = None) -> list[RollforwardState]:
    config = config or RollforwardConfig()
    state = build_initial_state(params)
    history = [state]
    for _ in range(config.horizon_months):
        state = step_month(state, params, cr, curves, config)
        history.append(state)
    return history


def state_to_frame(state: RollforwardState) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "cohort_id": c.cohort_id,
            "product_code": c.product_code,
            "product_name": c.product_name,
            "bs_side": c.bs_side,
            "currency": c.currency,
            "rate_type": c.rate_type,
            "origin": c.origin,
            "vintage_month": c.vintage_month,
            "balance": c.balance,
            "coupon_rate": c.coupon_rate,
        }
        for c in state.cohorts
    ])


def history_to_totals_frame(history: list[RollforwardState]) -> pd.DataFrame:
    rows = []
    for state in history:
        t = state.totals()
        t["month"] = state.month
        t["retained_earnings"] = state.retained_earnings
        rows.append(t)
    return pd.DataFrame(rows).set_index("month")
