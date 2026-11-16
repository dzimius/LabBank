"""Roll a bank's book forward to a later report date (transaction level).

Used instead of generation for a bank with ``bank_config.RollforwardCfg`` (e.g. eur_y1 =
the EUR book on 2027-09-30). b_s_gen_workflow.py dispatches here.

What happens to each contract between the base and the new report date
------------------------------------------------------------------------
- Loans (mortgages, cash and investment loans): keep their ids, start/maturity dates,
  margins and fixed rates. The balance amortises on its own schedule (same annuity /
  linear formula the generator used) and prepays with the behavioural prepayment model
  (loan_beh_models.xlsx, monthly rate by loan age -- the model the CF engine uses).
  Loans maturing on or before the new date drop out.
- Term deposits, bonds, T-bills, the covered bond: bullet; matured ones drop out. Bonds and
  T-bills still above their growth-plan target after maturities are sold pro rata down to it.
- Current and savings accounts with a liquidity behavioural model (dep_beh_models_liq.xlsx column):
  a monthly flow ledger (nmd_ledger: opening, outflow = the model's monthly decay x opening balance,
  day-1 stress drop excluded; inflow = what keeps the balance on the growth path; closing) is written
  to output/<bank>/nmd_ledger.xlsx and schemat.nmd_ledger. Today's accounts keep
  s(12M)/s(1D) of their balance (2026-10-20: 6000 78.8%, 8000 70%, 6350 50%): the run-off share
  of the accounts closes (random, seeded), the survivors shrink so the product keeps exactly that
  balance; the rest of the growth target (= this year's inflows minus their own outflows) is opened
  as new accounts at the standard rate. (User decisions 2026-10-19/20; before, NMD accounts were
  only scaled to target.) An NMD
  without a model column (6300 SME current account) is scaled like the products below.
- Credit cards, cash, equity: no run-off. The accounts stay and every balance is scaled to the
  product's growth target.
- New business: for every contractual product, target minus what survived is generated
  with the normal product generators, start dates inside the roll-forward window, priced
  on the window's fixings. New ids continue each product's numbering.
- Catalogue products (bank_data bs_percentage 0, no contracts in the base book): a positive
  target is all new business, generated from the product's bank_data row like any other new
  business (a scaled product opens new accounts); target 0 keeps the product in the catalogue
  unsold. A held scaled product with target 0 closes its accounts.
- Floating contracts reprice to the fixing on the new report date.

Targets come from the growth plan in the base bank's forecast folder (inputs/<base>/forecast): total
assets growth % (cell B1) and the new composition (share_new_pct, % of total assets; each side
sums to 100%, so assets = liabilities + equity by construction). One liability/equity row with
absorbs_difference = Y takes up any difference when run-off leaves a product above its target.

Market data after the base date (RollforwardCfg.market_path)
-------------------------------------------------------------
No information after the base report date is used; real quotes already downloaded after
the base date are replaced.
- "forwards": every quote on a date d in the window is the base curve's forward for
  [d, d + tenor], shifted so it equals the observed quote on the base date (keeps bid/ask
  spreads and basis between quote series) -- "today's curve, rolled down its forwards".
- "unchanged": every quote stays at its base-date value (random-walk view). Forwards
  carry a term premium, so they tend to over-predict rises on an upward-sloping curve.
- "hybrid" (eur_y1): forwards at the short end, unchanged at the long end. A quote of tenor T
  moves by w(T) x the change of the RISK-FREE (OIS) forward for [d, d + T]: w = 1 up to 2Y,
  0 from 5Y, linear in between (a 2Y-3Y blend left 2Y 15 bp above 3Y on 2026-09-30 -- a hump
  made by the method, there is no quote between 2Y and 3Y). The OIS curve = the base quotes minus each quote's spread over
  ESTR (curve_basis.csv: EURIBOR 1M ~0, 3M ~13, 6M ~31, 12M ~33 bp, swaps vs 6M ~30-20 bp), so
  the EURIBOR credit / term spread stays at today's level and is not read as expected rate
  rises. Why (EUR quotes 2005-2026, 1Y ahead): forwards beat "unchanged" up to 2Y (RMSE 93 vs
  124 bp at 1Y), about equal at 3-4Y, "unchanged" wins at 5Y (81 vs 91 bp); the forward bias flips sign with
  the rate cycle, so no fixed term premium is subtracted. A currency without curve_basis.csv
  uses its own curve's forwards (no spread removed).

Foreign currencies, facilities, derivatives (2026-10-13)
--------------------------------------------------------
- Growth plan shares are % of total assets in the HOME currency; a foreign-currency product's
  target is converted to its own currency at the new date's FX rate.
- FX path follows market_path: "forwards" / "hybrid" = covered-interest-parity forward of the
  base-date spot (S x DF_home / DF_foreign on the base curves), "unchanged" = spot held.
- Foreign market data (curves, fixings) roll exactly like the home currency's, into
  rolled_inputs/<ccy>/; curve_basis.csv (multi-curve) is carried over.
- Committed facilities (bank_data sheet commitments) grow with the book: a line linked to a
  drawn product keeps its utilisation (limit grows with the drawn balance), a line without
  drawings keeps its limit_pct of total assets. The register is rebuilt on the rolled book.
- Derivatives: swaps / FX deals maturing before the new date drop out. New IRS in the window
  come from inputs/<base>/forecast/new_swaps.xlsx (forecast_swaps.py): option 0 = none,
  1 = the last 12 months' swaps traded again one year later (same notionals and months),
  2 = the same with notionals grown with the balance sheet. Every planned trade starts after
  the base report date (no back-dated trades); without the file no new IRS. FX hedging is a
  policy that continues: for each currency the base bank hedged with FX swaps, the rolled bank
  gets the same ladder of 3M FX swaps (one per month, last three months), sized to 95% of its
  open position in that currency, priced at the rolled spot and 3M fixings (CIP forward).
  CIRS hedging (2026-11-07) continues too: the CIRS still alive are kept and one new CIRS, traded in
  the last month on the terms of the base bank's latest CIRS in that currency, tops the hedge up to
  95% of the rolled position (roll_cirs_hedge; an over-hedge is not unwound).

The rolled bank's inputs are GENERATED here (bank_data.xlsx with the grown structure,
behavioural/rate models re-dated, curve/fixing history extended, IRS book without matured
trades + planned new swaps) into runs/<bank>/rolled_inputs -- bank_config.input_dir points a rolled
bank there, so every later pipeline step runs unchanged. The only hand-made inputs are the
forecast files in inputs/<base bank>/forecast (growth plan, new swaps).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import QuantLib as ql
from sqlalchemy import create_engine, text

import config
import sql_setup
from b_s_gen_objects import (ProductFactory, compute_asset_rates, compute_deposit_client_rt,
                             decorate_from_bs_struct, get_num_of_months, ql_date_to_pd_date,
                             get_calendar_from_currency, seed_product)

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common import rate_models  # noqa: E402
from labbank_common import fx as fxm  # noqa: E402
from labbank_common import commitments as cmt  # noqa: E402
from labbank_common import catalogue  # noqa: E402
from labbank_common.bank_config import BankConfig, get_bank, get_engine  # noqa: E402
from labbank_common import rate_behaviour as rate_beh  # noqa: E402

_ADD_DATA_PY = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', 'balance_gen_add_data', 'python_code'))

AMORTISING = {'mortgage_fixed', 'mortgage_float', 'cash_loan_fixed', 'cash_loan_float',
              'investment_loan_fixed', 'investment_loan_float'}
BULLET     = {'term_deposit', 'bond_fixed', 'bond_float', 't_bill', 'issued_bond'}
SCALED     = {'current_account', 'saving_account', 'credit_card_revolver', 'credit_card_transactor',
              'cash', 'retained_earnings', 'common_shares', 'risk_reserves'}
CHILD_TABLES = ['loans', 'deposits', 'financial_instruments', 'equity']
NMD = {'current_account', 'saving_account'}        # run off with their liquidity behavioural model
MARKETABLE = {'bond_fixed', 'bond_float', 't_bill'}  # sold pro rata when above the growth-plan target

# copied from the base bank with report_date moved to the new date
MODEL_FILES = ['dep_beh_models_ir.xlsx', 'dep_beh_models_liq.xlsx',
               'dep_redemption_models.xlsx', 'loan_beh_models.xlsx', 'interest_rt.xlsx']


# ── market data ──────────────────────────────────────────────────────────────

def _period(tenor: str) -> ql.Period:
    n, u = int(tenor[:-1]), tenor[-1].upper()
    return ql.Period(n, {'D': ql.Days, 'W': ql.Weeks, 'M': ql.Months, 'Y': ql.Years}[u])


def _base_curve(curve_df: pd.DataFrame, base_rd: pd.Timestamp, ccy: str) -> tuple[ql.YieldTermStructureHandle, ql.Calendar]:
    if _ADD_DATA_PY not in sys.path:
        sys.path.insert(0, _ADD_DATA_PY)
    import b_s_add_data_objects as ado
    day = curve_df.loc[curve_df['date'] <= base_rd, 'date'].max()
    q = curve_df[(curve_df['date'] == day) & (curve_df['currency'] == ccy)]
    quotes = {t: float(q.loc[q['tenor'] == t, 'rate'].iloc[0]) / 100.0
              for t in ['1D', '1M', '3M', '6M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']}
    cal = ado.get_calendar_from_currency(ccy)
    return ado.build_curve_ql(ql.Date.from_date(day.date()), quotes, cal), cal


def _model_quote(yts, cal: ql.Calendar, d: pd.Timestamp, tenor: str) -> float:
    """Forward quote (decimal) for [d, d+tenor] on the base curve, in the curve's own
    quote convention: simple ACT/365 up to 1Y, annually compounded beyond."""
    dc = ql.Actual365Fixed()
    # the curve starts at settlement (base date + 2 business days): windows starting
    # earlier are shifted to settlement (only the first days of the window are affected)
    start = max(ql.Date.from_date(d.date()), yts.referenceDate())
    end = cal.advance(start, _period(tenor), ql.ModifiedFollowing)
    yf = dc.yearFraction(start, end)
    ratio = yts.discount(start) / yts.discount(end)
    if yf <= 1.0 + 1e-9:
        return (ratio - 1.0) / yf
    return ratio ** (1.0 / yf) - 1.0


HYBRID_FULL_Y, HYBRID_ZERO_Y = 2.0, 5.0          # hybrid: forwards up to 2Y, unchanged from 5Y
_TENOR_DAYS = {'1D': 1, '1M': 30, '3M': 90, '6M': 180, '1Y': 365}


def _tenor_years(tenor: str) -> float:
    n, u = int(tenor[:-1]), tenor[-1].upper()
    return n * {'D': 1 / 365, 'W': 7 / 365, 'M': 1 / 12, 'Y': 1.0}[u]


def hybrid_weight(tenor: str) -> float:
    y = _tenor_years(tenor)
    return float(np.clip((HYBRID_ZERO_Y - y) / (HYBRID_ZERO_Y - HYBRID_FULL_Y), 0.0, 1.0))


def ois_spread_bp(basis: pd.DataFrame, tenor: str) -> float:
    """Spread of the base quote of ``tenor`` over ESTR (bp, zero-rate difference at its maturity,
    curve_basis.csv): money-market fixings vs their own EURIBOR curve, swaps vs the 6M curve."""
    if tenor == '1D':
        return 0.0                                          # the 1D quote is ESTR itself
    days = _TENOR_DAYS.get(tenor, round(_tenor_years(tenor) * 365))
    piv = basis.pivot_table(index='n_days', columns='curve', values='spread')
    at = lambda col: float(np.interp(days, piv.index.to_numpy(float), piv[col].to_numpy(float)))
    own = {'1M': '1M', '3M': '3M', '1Y': '1Y'}.get(tenor)   # 6M and swaps: the 6M curve itself (0)
    return -(at('disc') - (at(own) if own else 0.0)) * 1e4


def _ois_curve(curve_df: pd.DataFrame, base_rd: pd.Timestamp, ccy: str, basis_path):
    """Risk-free curve of the base date: every base quote minus its spread over ESTR. None
    without a basis file (the hybrid path then uses the bank's own curve)."""
    import os.path
    if not os.path.exists(basis_path):
        return None
    if _ADD_DATA_PY not in sys.path:
        sys.path.insert(0, _ADD_DATA_PY)
    import b_s_add_data_objects as ado
    basis = pd.read_csv(basis_path)
    day = curve_df.loc[curve_df['date'] <= base_rd, 'date'].max()
    q = curve_df[(curve_df['date'] == day) & (curve_df['currency'] == ccy)]
    quotes = {t: float(q.loc[q['tenor'] == t, 'rate'].iloc[0]) / 100.0 - ois_spread_bp(basis, t) / 1e4
              for t in ['1D', '1M', '3M', '6M', '1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']}
    cal = ado.get_calendar_from_currency(ccy)
    return ado.build_curve_ql(ql.Date.from_date(day.date()), quotes, cal)


def roll_quotes(hist: pd.DataFrame, key_cols: list[str], yts, cal, base_rd, new_rd,
                market_path: str = 'forwards', yts_ois=None) -> pd.DataFrame:
    """History up to base_rd + quotes on every business day in (base_rd, new_rd]:
    forward-implied (market_path='forwards'), held at the base-date value ('unchanged') or
    'hybrid': moved by w(tenor) x the change of the OIS forward (yts_ois; None = yts)."""
    if market_path not in ('forwards', 'unchanged', 'hybrid'):
        raise ValueError(f"market_path must be 'forwards', 'unchanged' or 'hybrid', got {market_path!r}")
    if market_path == 'hybrid':
        yts = yts_ois if yts_ois is not None else yts
    past = hist[hist['date'] <= base_rd]
    last_day = past['date'].max()
    tmpl = past[past['date'] == last_day].copy()
    days = [d for d in pd.date_range(base_rd + pd.Timedelta(days=1), new_rd, freq='D')
            if cal.isBusinessDay(ql.Date.from_date(d.date()))]
    if pd.Timestamp(new_rd) not in days:
        days.append(pd.Timestamp(new_rd))      # the report date always gets a quote
    base_model = {t: _model_quote(yts, cal, last_day, t) for t in tmpl['tenor'].unique()}
    rows = []
    for d in days:
        fwd = (base_model if market_path == 'unchanged'
               else {t: _model_quote(yts, cal, d, t) for t in base_model})
        r = tmpl.copy()
        r['date'] = d
        w = (lambda t: hybrid_weight(t)) if market_path == 'hybrid' else (lambda t: 1.0)
        r['rate'] = r['rate'] + r['tenor'].map(lambda t: w(t) * (fwd[t] - base_model[t]) * 100.0)
        rows.append(r)
    out = pd.concat([past] + rows, ignore_index=True)
    return out.sort_values(['date'] + key_cols).reset_index(drop=True)


# ── inputs of the rolled bank ────────────────────────────────────────────────

def _with_report_date(df: pd.DataFrame, new_rd) -> pd.DataFrame:
    if 'report_date' in df.columns:
        df = df.copy()
        df['report_date'] = pd.Timestamp(new_rd)
    return df


def fx_path(bank: BankConfig, base: BankConfig) -> pd.DataFrame:
    """FX rates (units of currency per 1 home unit) for every day in (base date, new date]:
    the base-date spot rolled along covered interest parity ('forwards', 'hybrid') or held ('unchanged')."""
    new_rd, base_rd = pd.Timestamp(bank.report_date), pd.Timestamp(base.report_date)
    if not bank.market.foreign_ccys:
        return pd.DataFrame(columns=['date', 'currency', 'rate'])
    spot = fxm.fx_rates(base, base_rd)
    home_curves = pd.read_excel(base.market_dir() / 'curve_input.xlsx')
    y_home, _ = _base_curve(home_curves, base_rd, base.market.ccy)
    days = pd.date_range(base_rd + pd.Timedelta(days=1), new_rd, freq='D')
    rows = []
    for c in bank.market.foreign_ccys:
        y_fc, _ = _base_curve(pd.read_excel(fxm.foreign_input_dir(c, base) / 'curve_input.xlsx'), base_rd, c)
        for d in days:
            if bank.rollforward.market_path == 'unchanged':
                r = spot[c]
            else:
                qd = max(ql.Date.from_date(d.date()), y_home.referenceDate())
                r = spot[c] * y_home.discount(qd) / y_fc.discount(qd)
            rows.append((d, c, r))
    return pd.DataFrame(rows, columns=['date', 'currency', 'rate'])


def swap_changes_path(bank: BankConfig) -> Path:
    """runs/<rolled bank>/forecast_1y/swap_changes.xlsx: what the roll-forward did to the swap book."""
    return bank.output_dir('forecast_1y') / 'swap_changes.xlsx'


def write_swap_changes(bank: BankConfig, irs: pd.DataFrame, new: pd.DataFrame | None, option: int,
                       growth: float | None, new_rd) -> None:
    """The swaps held today, the ones maturing inside the window and the new trades of new_swaps.xlsx
    as actually booked (notional of the chosen option, fixed rate priced on the start date) -- with
    option 1 / 2 they are not visible in the input file itself."""
    import forecast_swaps
    cols = ['swap_id', 'like_swap_id', 'swap_type', 'pay_fixed', 'notional', 'currency', 'start_date', 'maturity_date',
            'fixed_rate', 'float_rate_index']
    new = new if new is not None else irs.iloc[0:0]
    new_ids = set(new['swap_id'])
    held = irs[~irs['swap_id'].isin(new_ids)].copy()
    mat = pd.to_datetime(held['maturity_date'])
    held['status'] = np.where(mat > new_rd, 'kept', 'matures in the window')
    new = new.assign(status=f'new (option {option})')
    book = pd.concat([held, new], ignore_index=True).reindex(columns=['status'] + cols)
    summary = pd.DataFrame({
        'item': ['option', 'meaning', 'notional growth % (option 2)', 'held today', 'matures in the window',
                 'new trades', 'book at the new report date'],
        'value': [option, forecast_swaps.OPTIONS[option], growth if option == 2 else None,
                  f"{len(held)} swaps, {held['notional'].sum() / 1e6:,.0f}M",
                  f"{(mat <= new_rd).sum()} swaps, {held.loc[mat <= new_rd, 'notional'].sum() / 1e6:,.0f}M",
                  f"{len(new)} swaps, {new['notional'].sum() / 1e6:,.0f}M",
                  f"{(mat > new_rd).sum() + len(new)} swaps, "
                  f"{(held.loc[mat > new_rd, 'notional'].sum() + new['notional'].sum()) / 1e6:,.0f}M"]})
    p = swap_changes_path(bank)
    p.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(p) as w:
        summary.to_excel(w, sheet_name='summary', index=False)
        book[book['status'].str.startswith('new')].to_excel(w, sheet_name='new_swaps', index=False)
        book.to_excel(w, sheet_name='swap_book', index=False)
    print(f'  swap changes -> {p}')


def roll_rate_sensitivity(bank: BankConfig, base: BankConfig, fix_base: pd.DataFrame,
                          fix_new: pd.DataFrame) -> None:
    """Carry rate_sensitivity.xlsx over and move the NMD profiles to the new date's 3M level
    (labbank_common/rate_behaviour.py, NMD_LEVEL). Without the file, or without NMD_LEVEL rows,
    nothing changes.

    Both profiles move. The stable (liquidity) profile is rolled; where the base bank's core
    profile is (1 - beta) x stable, the rolled core is rebuilt by the same rule, so the two files
    stay one model. A product whose files do not follow that rule has each file rolled on its own.
    The roll-forward has no rate-driven migration between products: money that a higher level
    takes out of current accounts is not booked as term deposits, so the shorter stable profile is
    the conservative side for liquidity."""
    src = base.models_dir() / rate_beh.MODEL_FILE
    if not src.exists():
        return
    import shutil
    shutil.copy(src, bank.models_dir() / rate_beh.MODEL_FILE)
    models = rate_beh.models_from_frame(pd.read_excel(src))
    if not any(m.behaviour == rate_beh.LEVEL for m in models):
        return
    r_from = rate_beh.market_rate_pct(fix_base, base.report_date)
    r_to = rate_beh.market_rate_pct(fix_new, bank.report_date)
    d = bank.models_dir()
    ir, liq = pd.read_excel(d / 'dep_beh_models_ir.xlsx'), pd.read_excel(d / 'dep_beh_models_liq.xlsx')
    ir_new, used = rate_beh.roll_nmd_profile(ir, models, r_from, r_to)
    liq_new, _ = rate_beh.roll_nmd_profile(liq, models, r_from, r_to)
    beta = {}
    if (d / 'interest_rt.xlsx').exists():
        rt = pd.read_excel(d / 'interest_rt.xlsx')
        beta = rt[rt['bs_side'] == 'L'].drop_duplicates('product_code').set_index('product_code')['beta'].to_dict()
    cols_ir, cols_liq = {str(c): c for c in ir.columns}, {str(c): c for c in liq.columns}
    same_grid = list(ir['tenor']) == list(liq['tenor'])
    for code in used:
        b = beta.get(int(code))
        if b is None or pd.isna(b) or code not in cols_liq or not same_grid:
            continue
        ci, cl = cols_ir[code], cols_liq[code]
        if np.allclose(ir[ci].fillna(0.0), (1 - b) * liq[cl].fillna(0.0), atol=1e-6):
            ir_new[ci] = (1 - b) * liq_new[cl]
    ir_new.to_excel(d / 'dep_beh_models_ir.xlsx', index=False)
    liq_new.to_excel(d / 'dep_beh_models_liq.xlsx', index=False)
    print(f'  NMD profiles (stable and core): 3M {r_from:.2f}% -> {r_to:.2f}%, exponent '
          + ', '.join(f'{c} {g:.3f}' for c, g in used.items()))


def build_inputs(bank: BankConfig, base: BankConfig, targets: pd.DataFrame,
                 fx_roll: pd.DataFrame | None = None) -> None:
    new_rd, base_rd = pd.Timestamp(bank.report_date), pd.Timestamp(base.report_date)
    ccy = bank.market.ccy

    # balance_generate: bank_data with the grown structure
    src = base.balance_dir() / 'bank_data.xlsx'
    dst_dir = bank.balance_dir(); dst_dir.mkdir(parents=True, exist_ok=True)
    sheets = pd.read_excel(src, sheet_name=None)
    # catalogue variants become ordinary rows (sold ones get their target; the catalogue sheet
    # is kept, re-expanding it is a no-op for codes already in bs_structure)
    bs = _with_report_date(catalogue.read_bs_structure(src), new_rd)
    full = 2.0 * targets.loc[targets['bs_side'] == 'A', 'target'].sum()
    tgt = targets.set_index(['product_code', 'bs_side'])['target']
    bs['bs_percentage'] = [round(tgt[(int(p), s)] / full * 100.0, 10) for p, s in zip(bs['product_code'], bs['bs_side'])]
    sheets['bs_structure'] = bs
    with pd.ExcelWriter(dst_dir / 'bank_data.xlsx') as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)

    # behavioural + client-rate models
    d = bank.models_dir(); d.mkdir(parents=True, exist_ok=True)
    for f in MODEL_FILES:
        _with_report_date(pd.read_excel(base.models_dir() / f), new_rd).to_excel(d / f, index=False)

    # market data: history to the base date + forward-implied window
    add_in = base.market_dir()
    bank.market_dir().mkdir(parents=True, exist_ok=True)
    curves = pd.read_excel(add_in / 'curve_input.xlsx')
    fixings = pd.read_excel(add_in / 'fixing_input.xlsx')
    yts, cal = _base_curve(curves, base_rd, ccy)
    path = bank.rollforward.market_path
    ois = _ois_curve(curves, base_rd, ccy, add_in / 'curve_basis.csv') if path == 'hybrid' else None
    roll_quotes(curves, ['ir_type'], yts, cal, base_rd, new_rd, path, ois).to_excel(
        bank.market_dir() / 'curve_input.xlsx', index=False)
    roll_quotes(fixings, ['rate_index'], yts, cal, base_rd, new_rd, path, ois).to_excel(
        bank.market_dir() / 'fixing_input.xlsx', index=False)
    import shutil
    if (add_in / 'curve_basis.csv').exists():                 # multi-curve basis carries over
        shutil.copy(add_in / 'curve_basis.csv', bank.market_dir() / 'curve_basis.csv')
    roll_rate_sensitivity(bank, base, fixings, pd.read_excel(bank.market_dir() / 'fixing_input.xlsx'))

    # foreign currencies: same roll, into rolled_inputs/<ccy>/; FX history + rolled path
    for c in bank.market.foreign_ccys:
        src_d, dst_d = fxm.foreign_input_dir(c, base), fxm.foreign_input_dir(c, bank)
        dst_d.mkdir(parents=True, exist_ok=True)
        cv = pd.read_excel(src_d / 'curve_input.xlsx')
        y_c, cal_c = _base_curve(cv, base_rd, c)
        ois_c = _ois_curve(cv, base_rd, c, src_d / 'curve_basis.csv') if path == 'hybrid' else None
        roll_quotes(cv, ['ir_type'], y_c, cal_c, base_rd, new_rd, path, ois_c).to_excel(dst_d / 'curve_input.xlsx', index=False)
        roll_quotes(pd.read_excel(src_d / 'fixing_input.xlsx'), ['rate_index'], y_c, cal_c, base_rd, new_rd,
                    path, ois_c).to_excel(dst_d / 'fixing_input.xlsx', index=False)
        if (src_d / 'curve_basis.csv').exists():
            shutil.copy(src_d / 'curve_basis.csv', dst_d / 'curve_basis.csv')
    if bank.market.foreign_ccys:
        hist = pd.read_excel(fxm.fx_input_path(base))
        hist = hist[pd.to_datetime(hist['date']) <= base_rd]
        pd.concat([hist, fx_roll], ignore_index=True).to_excel(fxm.fx_input_path(bank), index=False)

    # derivatives.xlsx: same swaps and FX deals, matured ones dropped. The base bank's
    # generated asset swaps become ordinary trades of the rolled book (the hedges stay on
    # their bonds; the asset_swap rules sheet is not carried, so new bonds are unhedged).
    # Planned trades of the window come from inputs/<base>/forecast/new_swaps.xlsx
    # (forecast_swaps.py; option 0 or no file = none).
    from labbank_common import derivatives as deriv
    import forecast_swaps
    irs = deriv.read_swap_book(base)
    ns = forecast_swaps.read(bank)
    if ns is not None:
        fc, option, growth = ns
        if not fc.empty:
            cv = [pd.read_excel(bank.market_dir() / 'curve_input.xlsx')]
            cv += [pd.read_excel(fxm.foreign_input_dir(c, bank) / 'curve_input.xlsx') for c in bank.market.foreign_ccys]
            fc = forecast_swaps.price(fc, pd.concat(cv, ignore_index=True))
            irs = pd.concat([irs, fc.reindex(columns=irs.columns)], ignore_index=True)
        print(f'  {bank.rollforward.swaps_file}: option {option} ({forecast_swaps.OPTIONS[option]}'
              + (f', x {1 + growth / 100:.4f}' if option == 2 else '') + f'): {len(fc)} new swaps '
              + f'{fc["notional"].sum() / 1e6:,.0f}M' + (', fixed ' + ', '.join(f'{r:.2%}' for r in fc['fixed_rate'])
                                                     if len(fc) else ''))
    write_swap_changes(bank, irs, fc if ns is not None else None, ns[1] if ns is not None else 0,
                       ns[2] if ns is not None else None, new_rd)
    irs = _with_report_date(irs[pd.to_datetime(irs['maturity_date']) > new_rd], new_rd)
    fxd = deriv.read_fx_deals(base)
    fxd = fxd[fxd['far_date'].fillna(fxd['near_date']) > new_rd]
    d = deriv.input_path(bank).parent; d.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(d / deriv.FILE) as w:
        irs.to_excel(w, sheet_name=deriv.SWAP_SHEET, index=False)
        _with_report_date(fxd, new_rd).to_excel(w, sheet_name=deriv.FX_SHEET, index=False)

    note = (f"# GENERATED -- do not edit\n\n"
            f"Written by the roll-forward of `{base.bank_id}` ({base.report_date}) to `{bank.bank_id}` "
            f"({bank.report_date}) -- `balance_generate/python_code/b_s_rollforward.py`, on every run of\n"
            f"`balance_generate/python_code/b_s_gen_workflow.py` with LABBANK_BANK={bank.bank_id}.\n\n"
            f"To change the {bank.bank_id} book, edit the INPUTS instead:\n"
            f"- forecast inputs: `inputs/{base.bank_id}/forecast/{bank.rollforward.growth_file}` (growth per product) and "
            f"`{bank.rollforward.swaps_file}` (IRS traded inside the window)\n"
            f"- everything else (products, swaps held today, models): `inputs/{base.bank_id}` and `inputs/market`\n"
            f"- market path after {base.report_date} ('{bank.rollforward.market_path}'): `RollforwardCfg` in "
            f"`labbank_common/bank_config.py`\n\n"
            f"They are outputs (the input folders of BankConfig point a rolled bank here): every later "
            f"pipeline step reads its inputs through them, so it runs unchanged on the rolled book: the "
            f"grown balance-sheet structure (bank_data.xlsx), the models with the new report_date, the market "
            f"data extended to {bank.report_date}, and the swap book without matured trades.\n")
    (bank.balance_dir().parent / 'README.md').write_text(note, encoding='utf-8')


# ── growth targets ───────────────────────────────────────────────────────────

def read_growth_plan(path) -> tuple[float, pd.DataFrame]:
    """growth_plan.xlsx: B1 = total assets growth % over the year; the product table starts at
    the row whose first cell is 'product_code' (rows below it without a product code -- the
    totals / CHECK block -- are ignored). Formula columns are for the reader only: their values
    are not stored by the file writer, so they are recalculated here."""
    raw = pd.read_excel(path, sheet_name='growth_plan', header=None)
    total_growth = float(raw.iloc[0, 1])
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip() == 'product_code'][0]
    gp = pd.read_excel(path, sheet_name='growth_plan', header=hdr)
    gp = gp[pd.to_numeric(gp['product_code'], errors='coerce').notna()].copy()
    gp['product_code'] = gp['product_code'].astype(int)
    gp['share_new_pct'] = pd.to_numeric(gp['share_new_pct'], errors='coerce')
    gp['plug'] = gp['absorbs_difference'].fillna('').astype(str).str.upper().eq('Y')
    return total_growth, gp


def growth_targets(base: BankConfig, book: pd.DataFrame, growth_file: str,
                   to_home_base: dict | None = None, to_home_new: dict | None = None) -> pd.DataFrame:
    """Targets = total assets in 1 year x share_new_pct (same arithmetic as the sheet's formulas).
    Assets and liabilities + equity each sum to 100% of total assets, so A = L + E by construction.
    Amounts in the HOME currency (book balances at the base FX); target_native = the target in
    the product's own currency at the new date's FX rate."""
    to_home_base = to_home_base or {}
    to_home_new = to_home_new or {}
    path = base.forecast_dir() / growth_file
    total_growth, gp = read_growth_plan(path)
    if gp['share_new_pct'].isna().any():
        raise ValueError(f'{growth_file}: share_new_pct missing for {gp.loc[gp.share_new_pct.isna(), "own_name"].tolist()}')
    for side, label in (('A', 'assets'), ('LE', 'liabilities + equity')):
        sh = gp.loc[gp['bs_side'].isin(list(side)), 'share_new_pct'].sum()
        if abs(sh - 100.0) > 1e-3:
            raise ValueError(f'{growth_file}: share_new_pct of {label} sums to {sh:.4f}%, must be 100% '
                             '(the CHECK cell in the sheet shows it)')
    if gp['plug'].sum() != 1 or (gp.loc[gp['plug'], 'bs_side'] == 'A').any():
        raise ValueError(f'{growth_file}: exactly one liability / equity row needs absorbs_difference = Y')
    key = ['product_code', 'bs_side']
    bk = book.assign(product_code=book['product_code'].astype(int))
    bk['balance_home'] = bk['balance_amt'] * bk['currency'].map(lambda c: to_home_base.get(c, 1.0))
    bal = bk.groupby(key)['balance_home'].sum().rename('balance_base')
    # products come from bank_data (bs_percentage 0 = in the catalogue, no contracts yet):
    # the plan must list every one of them, the book holds a subset
    bs = catalogue.read_bs_structure(base.balance_dir() / 'bank_data.xlsx')
    prods = bs[key + ['currency']].drop_duplicates(key).assign(product_code=lambda d: d['product_code'].astype(int))
    # the plan's currency column is for the reader; bank_data's currency is the one used
    t = gp.drop(columns='currency', errors='ignore').merge(prods, on=key, how='outer', indicator=True)
    if (t['_merge'] != 'both').any():
        raise ValueError(f'{growth_file} and bank_data.xlsx (bs_structure) list different products -- every '
                         f'bank_data product needs a growth-plan row (share_new_pct 0 = not sold):\n'
                         f'{t.loc[t["_merge"] != "both", key + ["_merge"]]}')
    held = bal.index.difference(pd.MultiIndex.from_frame(prods[key]))
    if len(held):
        raise ValueError(f'the base book holds products missing from bank_data.xlsx: {list(held)}')
    t = t.drop(columns='_merge').merge(bal.reset_index(), on=key, how='left')
    t['balance_base'] = t['balance_base'].fillna(0.0)          # catalogue product, nothing held yet
    t['to_home_new'] = t['currency'].map(lambda c: to_home_new.get(c, 1.0))
    ta_new = t.loc[t['bs_side'] == 'A', 'balance_base'].sum() * (1.0 + total_growth / 100.0)
    # money: to the cent (float noise like 630,000,000.0000002 otherwise becomes a dust contract)
    t['target'] = (ta_new * t['share_new_pct'] / 100.0).round(2)
    # target 0 = kept in the catalogue but not sold; a product starting from 0 has no growth %
    t['growth_pct'] = np.where(t['balance_base'] > 0, (t['target'] / t['balance_base'].where(t['balance_base'] > 0) - 1.0) * 100.0, np.nan)
    if (t['target'] < 0).any():
        raise ValueError(f'growth plan gives a negative target:\n{t[t["target"] < 0]}')
    return t


EQUITY_ENV = 'LABBANK_ROLLFORWARD_EQUITY'


def pnl_equity_targets(bank: BankConfig, t: pd.DataFrame) -> pd.DataFrame:
    """RollforwardCfg.equity = 'pnl': equity from the year's result instead of the growth plan
    (accounting/python_code/pnl_rollforward.py, run on the plan first; docs/07 §3.2).

    The plan sets the asset and liability mix; share capital and reserves stay at the base
    balance; retained earnings = the plan's total equity + the cash the result left after
    dividends (the P&L ledger's cash surplus). The surplus sits at the central bank (3510 A), a
    deficit is borrowed interbank (7900 L): A = L + E still holds by construction. The book stays
    at nominal: OCI, derivative values and the hedge adjustment are added by the accounting
    valuation of the rolled bank, not booked as contracts."""
    f = bank.output_dir('accounting', create=False) / 'pnl_rollforward.xlsx'
    if not f.exists():
        raise FileNotFoundError(f'{f} missing: run accounting/python_code/pnl_rollforward.py --bank '
                                f'{bank.bank_id} on the plan (equity = "plan") first')
    summ = pd.read_excel(f, sheet_name='summary').set_index('item')['value']
    cash = float(summ['cash_surplus_close'])
    t = t.copy()
    eq = t['product_name'].isin(['retained_earnings', 'common_shares', 'risk_reserves'])
    re = t['product_name'] == 'retained_earnings'
    if re.sum() != 1:
        raise ValueError('equity = "pnl" needs exactly one retained_earnings row in the growth plan')
    e_plan = float(t.loc[eq, 'target'].sum())
    t.loc[eq & ~re, 'target'] = t.loc[eq & ~re, 'balance_base']
    t.loc[re, 'target'] = round(e_plan + cash - float(t.loc[eq & ~re, 'target'].sum()), 2)
    acct = (3510, 'A') if cash >= 0 else (7900, 'L')
    row = (t['product_code'] == acct[0]) & (t['bs_side'] == acct[1])
    if row.sum() != 1:
        raise ValueError(f'equity = "pnl": no growth-plan row {acct} for the cash surplus / deficit')
    t.loc[row, 'target'] = round(float(t.loc[row, 'target'].iloc[0]) + abs(cash), 2)
    t['growth_pct'] = np.where(t['balance_base'] > 0, (t['target'] / t['balance_base'].where(t['balance_base'] > 0)
                                                       - 1.0) * 100.0, np.nan)
    print(f'  equity from the P&L: retained earnings {t.loc[re, "target"].iloc[0]/1e6:,.1f}M, cash '
          f'{cash/1e6:+,.1f}M to {acct}')
    return t


# ── base book ────────────────────────────────────────────────────────────────

def load_book(base: BankConfig) -> pd.DataFrame:
    eng = get_engine(base)
    rd = pd.Timestamp(base.report_date)
    parts = []
    for t in CHILD_TABLES:
        df = pd.read_sql(text(f'SELECT * FROM schemat.{t} WHERE report_date = :rd'), eng, params={'rd': rd})
        parts.append(df)
    book = pd.concat(parts, ignore_index=True)
    # parent table: client_id, plus bs_side / client_type_id that schemat.equity does not store
    tx = pd.read_sql(text('SELECT transaction_id, client_id, bs_side, client_type_id '
                          'FROM dbo.transactions WHERE report_date = :rd'), eng, params={'rd': rd})
    book = book.merge(tx, on='transaction_id', how='left', suffixes=('', '_tx'))
    for c in ('bs_side', 'client_type_id'):
        book[c] = book[c].fillna(book.pop(f'{c}_tx')) if c in book.columns else book.pop(f'{c}_tx')
    for c in book.columns:
        if c in ('start_date', 'maturity_date', 'report_date'):
            book[c] = pd.to_datetime(book[c])
        elif book[c].dtype == object and c not in ('transaction_id',):
            conv = pd.to_numeric(book[c], errors='coerce')
            if conv.notna().sum() == book[c].notna().sum() and book[c].notna().any():
                book[c] = conv
    book['product_code'] = book['product_code'].astype(int)
    return book


# ── ageing ───────────────────────────────────────────────────────────────────

def _months(a: pd.Series, b: pd.Timestamp) -> np.ndarray:
    return ((b.year - a.dt.year) * 12 + (b.month - a.dt.month)).to_numpy()


def _k(amort, m, M, r):
    """Outstanding fraction after m of M months (bullet 0 / annuity 1 / linear 2)."""
    m = np.clip(m, 0, M)
    lin = np.where(M > 0, (M - m) / np.maximum(M, 1), 0.0)
    rm = r / 12.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ann = np.where(rm > 0, ((1 + rm) ** M - (1 + rm) ** m) / ((1 + rm) ** M - 1.0), lin)
    return np.where(amort == 0, 1.0, np.where(amort == 2, lin, ann))


def _prepay_survival(pc: np.ndarray, m_old: np.ndarray, m_new: np.ndarray, models: pd.DataFrame) -> np.ndarray:
    """Product of (1 - monthly prepayment rate at each loan age) over the window --
    the same tenor = loan-age-in-months lookup cf_calc_workflow.py uses."""
    rates = {}
    for col in models.columns:
        try:
            code = int(col)
        except (TypeError, ValueError):
            continue
        s = models.set_index('tenor')[col]
        rates[code] = {int(t[:-1]): float(v) for t, v in s.items() if str(t).endswith('M') and pd.notna(v)}
    out = np.ones(len(pc))
    for i, (p, a, b) in enumerate(zip(pc, m_old, m_new)):
        r = rates.get(int(p))
        if r:
            out[i] = np.prod([1.0 - r.get(age, 0.0) for age in range(a + 1, b + 1)])
    return out


def age_book(book: pd.DataFrame, new_rd: pd.Timestamp, avg_rate_map: dict, loan_models: pd.DataFrame) -> pd.DataFrame:
    b = book.copy()
    name = b['product_name']
    matured = b['maturity_date'].notna() & (b['maturity_date'] <= new_rd) & name.isin(AMORTISING | BULLET)
    b = b[~matured].copy()
    name = b['product_name']

    am = name.isin(AMORTISING)
    if am.any():
        x = b[am]
        M = np.array([get_num_of_months(m) for m in x['maturity']])
        m_old = _months(x['start_date'], pd.Timestamp(x['report_date'].iloc[0]))
        m_new = _months(x['start_date'], new_rd)
        r = x['product_name'].map(lambda n: avg_rate_map.get(n, 3) / 100.0).to_numpy()
        amort = pd.to_numeric(x['amort_type'], errors='coerce').fillna(1).astype(int).to_numpy()
        k_old, k_new = _k(amort, m_old, M, r), _k(amort, m_new, M, r)
        ratio = np.where(k_old > 0, k_new / np.where(k_old > 0, k_old, 1.0), 0.0)
        surv = _prepay_survival(x['product_code'].to_numpy(), m_old, m_new, loan_models)
        b.loc[am, 'balance_amt'] = np.round(x['balance_amt'].to_numpy() * ratio * surv, 2)
        b = b[~(am & (b['balance_amt'] <= 0.005))].copy()
    b['report_date'] = new_rd
    return b


# ── new business ─────────────────────────────────────────────────────────────

def _renumber(new: pd.DataFrame, used_ids: pd.Series) -> pd.DataFrame:
    """Generator ids restart at 1 per product -- continue after the highest id ever used."""
    new = new.copy()
    prefix_new = new['transaction_id'].str[:6]
    used = used_ids.str[:6].to_frame('p').assign(n=used_ids.str[6:].astype(int)).groupby('p')['n'].max()
    seq = new.groupby(prefix_new).cumcount() + 1
    start = prefix_new.map(used).fillna(0).astype(int)
    new['transaction_id'] = prefix_new + (start + seq).map(lambda n: f'{n:07d}')
    return new


def nmd_hazards(base: BankConfig, months: int) -> dict[int, np.ndarray]:
    """Monthly outflow rate of each NMD product (liquidity behavioural model, dep_beh_models_liq.xlsx,
    one survival column per product code): h_k = 1 - s(kM) / s((k-1)M), k = 1..months, with s(0M) =
    s(1D) -- the day-1 drop is a liquidity-STRESS block (LCR / ILAAP), not business behaviour, so the
    forecast leaves it out (2026-10-20 user decision)."""
    m = pd.read_excel(base.models_dir() / 'dep_beh_models_liq.xlsx').set_index('tenor')
    need = ['1D'] + [f'{k}M' for k in range(1, months + 1)]
    if not set(need) <= set(m.index):
        raise ValueError(f'dep_beh_models_liq.xlsx needs the tenors 1D, 1M..{months}M')
    out = {}
    for c in m.columns:
        if not str(c).isdigit():
            continue
        sv = m.loc[need, c].astype(float)
        if sv.isna().all():
            continue
        sv = sv.ffill().fillna(0.0).to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            h = np.where(sv[:-1] > 0, 1.0 - sv[1:] / sv[:-1], 1.0)
        out[int(c)] = np.clip(h, 0.0, 1.0)
    return out


def nmd_ledger(opening: float, target: float, hazards: np.ndarray, base_rd) -> pd.DataFrame:
    """Monthly flows of one NMD product (pool view, the rate applies to the whole opening balance, so
    new money runs off too): outflow = h_k x opening, closing = the growth path (opening compounded
    monthly to the target), inflow = closing - opening + outflow (>= 0; a product planned to shrink
    faster than it runs off gets the extra as outflow)."""
    n = len(hazards)
    path = [opening * (target / opening) ** (k / n) if opening > 0 and target > 0 else target for k in range(n + 1)]
    rows, o = [], opening
    for k in range(n):
        out = o * hazards[k]
        close = path[k + 1]
        inn = close - o + out
        if inn < 0:
            out, inn = o - close, 0.0
        rows.append(dict(month=k + 1, date=pd.Timestamp(base_rd) + pd.offsets.MonthEnd(k + 1),
                         opening=o, outflow=out, inflow=inn, closing=close, outflow_rate=hazards[k]))
        o = close
    return pd.DataFrame(rows)


def run_off_nmd(aged: pd.DataFrame, pc: int, side: str, attrition: float, target: float, seed: int) -> pd.DataFrame:
    """Existing accounts of one NMD product after the year: a random (seeded) ``attrition`` share of
    the accounts closes, the rest are scaled so the product keeps exactly (1 - attrition) of its
    balance -- or the growth target, if that is lower. The volume up to the target is new accounts."""
    m = (aged['product_code'] == pc) & (aged['bs_side'] == side)
    idx = aged.index[m]
    bal = float(aged.loc[idx, 'balance_amt'].sum())
    keep_bal = min(bal * (1.0 - attrition), target)
    n_close = int(round(attrition * len(idx)))
    n_close = min(n_close, len(idx) - 1) if keep_bal > 0 else len(idx)
    closed = np.random.default_rng(seed + pc).choice(idx, size=n_close, replace=False)
    aged = aged.drop(index=closed)
    rest = idx.difference(closed)
    f = keep_bal / float(aged.loc[rest, 'balance_amt'].sum()) if len(rest) else 0.0
    for c in ('balance_amt', 'init_balance_amt'):
        if c in aged.columns:
            aged.loc[rest, c] = aged.loc[rest, c] * f
    return aged


def generate_new_business(bs_struct: pd.DataFrame, gaps: pd.DataFrame, base_rd, new_rd, seed: int) -> pd.DataFrame:
    config.report_date = pd.Timestamp(new_rd)
    config.balance_start_date = pd.Timestamp(base_rd) + pd.Timedelta(days=1)   # starts inside the window only
    df_client_t = pd.read_excel('dictionaries/client_type.xlsx')
    rows = bs_struct.merge(df_client_t, on='client_type_id', how='left')
    parts = []
    book_margins = config.margin_map
    config.margin_map = config.bank.generation.new_business_margin_map   # new originations: new-business margins
    try:
        for _, g in gaps.iterrows():
            row = rows[(rows['product_code'] == g['product_code']) & (rows['bs_side'] == g['bs_side'])].iloc[0].copy()
            row['balance_amt'] = g['new_business']
            seed_product(seed, g['product_code'], g['bs_side'])
            parts.append(ProductFactory.create(row).build_result_df())
    finally:
        config.margin_map = book_margins
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── FX hedge ladder ──────────────────────────────────────────────────────────

HEDGE_RATIO = fxm.FX_HEDGE_RATIO


def roll_fx_hedge(bank: BankConfig, base: BankConfig, all_tx: pd.DataFrame, fixings: pd.DataFrame,
                  fx_roll: pd.DataFrame) -> pd.DataFrame:
    """The base bank's FX-swap hedging continues: per currency it hedged with FX swaps, three 3M
    FX swaps traded in the last three months (one per month), together HEDGE_RATIO of the rolled
    book's short position in that currency; the far exchange buys the currency back (long it),
    the same shape as the base ladder. Rates: rolled spot (fx_roll) and rolled 3M fixings,
    forward by covered interest parity."""
    from labbank_common import derivatives as deriv
    new_rd = pd.Timestamp(bank.report_date)
    path = deriv.input_path(bank)
    swaps = pd.read_excel(path, sheet_name=deriv.SWAP_SHEET)
    kept = deriv.read_fx_deals(bank)
    base_deals = deriv.read_fx_deals(base)
    home = bank.market.ccy
    hedged = set(base_deals.loc[base_deals['deal_type'] == 'FX_SWAP', 'far_buy_ccy']) - {home}
    fix = fixings.copy()
    fix['fixing_date'] = pd.to_datetime(fix['fixing_date'])
    sgn = np.where(all_tx['bs_side'] == 'A', 1.0, np.where(all_tx['bs_side'] == 'L', -1.0, 0.0))
    new_rows = []
    for c in sorted(hedged):
        nop = float((all_tx['balance_amt'] * sgn)[(all_tx['currency'] == c).to_numpy()].sum())
        if nop >= 0:
            continue
        each = round(-nop * HEDGE_RATIO / fxm.FX_LADDER_N / 1e6) * 1e6
        tdc = ql.JointCalendar(get_calendar_from_currency(home), get_calendar_from_currency(c))
        fr = fx_roll[fx_roll['currency'] == c].set_index('date')['rate']

        def _rate(idx, t):
            f = fix[(fix['rate_index'] == idx) & (fix['fixing_date'] <= t)].sort_values('fixing_date')
            return float(f['rate'].iloc[-1]) / 100.0

        for k, mback in enumerate((2, 1, 0)):
            t = (new_rd - pd.DateOffset(months=mback)).replace(day=13)
            near_q = tdc.advance(ql.Date.from_date(t.date()), 2, ql.Days)
            far_q = tdc.advance(near_q, ql.Period(3, ql.Months), ql.ModifiedFollowing)
            near, far = ql_date_to_pd_date(near_q), ql_date_to_pd_date(far_q)
            S = float(fr[fr.index <= t].iloc[-1])
            d = (far - near).days
            F = S * (1 + _rate(f'{c}_ASK_3M', t) * d / 360) / (1 + _rate(f'{home}_ASK_3M', t) * d / 360)
            new_rows.append(dict(report_date=new_rd, deal_id=f'FXS_Y{k + 1:02d}_{c}', deal_type='FX_SWAP',
                                 ccy_pair=f'{home}{c}', trade_date=t,
                                 near_date=near, near_buy_ccy=home, near_buy_amt=round(each / S, 2),
                                 near_sell_ccy=c, near_sell_amt=each, near_rate=round(S, 6),
                                 far_date=far, far_buy_ccy=c, far_buy_amt=each,
                                 far_sell_ccy=home, far_sell_amt=round(each / F, 2), far_rate=round(F, 6)))
        print(f'  FX hedge rolled: {c} open position {nop/1e6:,.0f}M -> 3 x {each/1e6:,.0f}M 3M FX swaps')
    fxd = pd.concat([kept, pd.DataFrame(new_rows, columns=deriv.FX_COLUMNS)], ignore_index=True)[deriv.FX_COLUMNS]
    with pd.ExcelWriter(path) as w:
        swaps.to_excel(w, sheet_name=deriv.SWAP_SHEET, index=False)
        fxd.to_excel(w, sheet_name=deriv.FX_SHEET, index=False)
    return fxd


# ── main ─────────────────────────────────────────────────────────────────────

def cirs_position(swaps: pd.DataFrame, ccy: str) -> float:
    """Open position the CIRS legs carry in ``ccy`` (+ long): the fixed leg is in ``currency``, the
    float leg in ``float_currency``; pay_fixed = 1 pays the fixed leg (short its currency)."""
    s = swaps[swaps['swap_type'].astype(str).str.upper() == 'CIRS']
    pay = s['pay_fixed'].fillna(1).astype(int) == 1
    fixed = np.where(pay, -1.0, 1.0) * s['notional'].astype(float) * (s['currency'] == ccy)
    flt = np.where(pay, 1.0, -1.0) * s['float_notional'].astype(float) * (s['float_currency'] == ccy)
    return float(fixed.sum() + flt.sum())


def cirs_top_up(nop_bs: float, held: float, ratio: float = HEDGE_RATIO) -> float:
    """Notional of the new CIRS leg in the currency (signed like a position: - = the bank pays the
    currency) that brings the hedge to ``ratio`` of the balance-sheet position. Only trades in the
    direction of the existing hedge; an over-hedge is not unwound (0)."""
    need = -ratio * nop_bs - held
    if abs(need) < 1e6 or (held != 0 and np.sign(need) != np.sign(held)):
        return 0.0
    return round(need / 1e6) * 1e6


def roll_cirs_hedge(bank: BankConfig, base: BankConfig, all_tx: pd.DataFrame, fixings: pd.DataFrame,
                    fx_roll: pd.DataFrame) -> pd.DataFrame:
    """The base bank's CIRS hedging continues. Per currency it hedged with CIRS (fixed leg in that
    currency), the CIRS still alive on the new date are kept; one new CIRS traded on the 13th of the
    last month of the window (start two business days later) tops the hedge up to HEDGE_RATIO of the
    rolled open position. Terms are those of the base bank's latest CIRS in that currency (tenor,
    frequencies, other leg); fixed rate = the rolled par quote of the tenor on the trade date, other
    notional = the currency notional at the rolled spot. A matured CIRS is replaced the same way. The
    hedge is never reduced: an over-hedge (the position shrank) stays and shows as open position."""
    from labbank_common import derivatives as deriv
    new_rd = pd.Timestamp(bank.report_date)
    path = deriv.input_path(bank)
    sheets = pd.read_excel(path, sheet_name=None)
    swaps = sheets[deriv.SWAP_SHEET]
    base_book = deriv.read_swap_book(base)
    base_cirs = base_book[base_book['swap_type'] == 'CIRS']
    if base_cirs.empty:
        return swaps
    home = bank.market.ccy
    fix = fixings.copy()
    fix['fixing_date'] = pd.to_datetime(fix['fixing_date'])
    sgn = np.where(all_tx['bs_side'] == 'A', 1.0, np.where(all_tx['bs_side'] == 'L', -1.0, 0.0))
    nums = pd.to_numeric(swaps['swap_id'].astype(str).str.extract(r'CIRS_(\d+)$')[0], errors='coerce')
    nxt = int(nums.max()) + 1 if nums.notna().any() else 1
    new_rows = []
    for c in sorted(set(base_cirs['currency']) - {home}):
        nop = float((all_tx['balance_amt'] * sgn)[(all_tx['currency'] == c).to_numpy()].sum())
        held = cirs_position(swaps, c)
        leg = cirs_top_up(nop, held)
        if leg == 0.0:
            print(f'  CIRS hedge {c}: open position {nop / 1e6:+,.0f}M, CIRS {held / 1e6:+,.0f}M -- no new trade')
            continue
        tmpl = base_cirs[base_cirs['currency'] == c].sort_values('start_date').iloc[-1]
        months = _months(pd.Series([tmpl['start_date']]), pd.Timestamp(tmpl['maturity_date']))[0]
        tenor = f'{int(round(months / 12))}Y'
        t = new_rd.replace(day=13)
        tdc = ql.JointCalendar(get_calendar_from_currency(home), get_calendar_from_currency(c))
        start = ql_date_to_pd_date(tdc.advance(ql.Date.from_date(t.date()), 2, ql.Days))
        mat = ql_date_to_pd_date(tdc.advance(ql.Date.from_date(start.date()), ql.Period(int(months), ql.Months),
                                             ql.ModifiedFollowing))
        q = fix[(fix['rate_index'] == f'{c}_ASK_{tenor}') & (fix['fixing_date'] <= t)].sort_values('fixing_date')
        if q.empty:
            raise ValueError(f'CIRS hedge {c}: no {c}_ASK_{tenor} fixing up to {t.date()}')
        fr = fx_roll[(fx_roll['currency'] == c) & (fx_roll['date'] <= t)].sort_values('date')
        S = float(fr['rate'].iloc[-1])
        row = tmpl.copy()
        row.update(dict(report_date=new_rd, swap_id=f'CIRS_{nxt:03d}', pay_fixed=1 if leg < 0 else 0,
                        notional=abs(leg), start_date=start, maturity_date=mat,
                        fixed_rate=round(float(q['rate'].iloc[-1]) / 100.0, 6),
                        float_notional=round(abs(leg) / S, 2)))
        new_rows.append(row)
        nxt += 1
        print(f'  CIRS hedge {c}: open position {nop / 1e6:+,.0f}M, CIRS held {held / 1e6:+,.0f}M -> new '
              f'{row["swap_id"]} {"pay" if leg < 0 else "receive"} {c} fixed {abs(leg) / 1e6:,.0f}M {tenor} at '
              f'{row["fixed_rate"]:.3%}, start {start.date()}')
    if new_rows:
        sheets[deriv.SWAP_SHEET] = swaps = pd.concat([swaps, pd.DataFrame(new_rows)[swaps.columns]],
                                                     ignore_index=True)
        with pd.ExcelWriter(path) as w:
            for name, df in sheets.items():
                df.to_excel(w, sheet_name=name, index=False)
    return swaps


def ensure_database(bank: BankConfig) -> None:
    master = create_engine(bank.sql.url.replace(f'/{bank.sql.db_name}?', '/master?'),
                           isolation_level='AUTOCOMMIT', future=True)
    with master.connect() as c:
        c.execute(text(f"IF DB_ID('{bank.sql.db_name}') IS NULL CREATE DATABASE [{bank.sql.db_name}]"))
    master.dispose()


def run_rollforward(bank: BankConfig) -> pd.DataFrame:
    rf = bank.rollforward
    base = get_bank(rf.base_bank)
    base_rd, new_rd = pd.Timestamp(base.report_date), pd.Timestamp(bank.report_date)
    print(f'Roll-forward {base.bank_id} {base_rd.date()} -> {bank.bank_id} {new_rd.date()}')

    book = load_book(base)
    fx_roll = fx_path(bank, base)
    to_home_base = fxm.to_home(base, base_rd)
    to_home_new = {bank.market.ccy: 1.0, **{c: 1.0 / float(fx_roll.loc[fx_roll['currency'] == c, 'rate'].iloc[-1])
                                             for c in bank.market.foreign_ccys}}
    targets = growth_targets(base, book, rf.growth_file, to_home_base, to_home_new)
    # LABBANK_ROLLFORWARD_EQUITY: 'plan' (default) takes the equity of the growth plan
    
    equity = os.environ.get(EQUITY_ENV) or getattr(rf, 'equity', 'plan')
    print(f'  equity: {equity!r}')
    if equity == 'pnl':
        targets = pnl_equity_targets(bank, targets)
    loan_models = catalogue.expand_columns(
        pd.read_excel(base.models_dir() / 'loan_beh_models.xlsx'), catalogue.parent_map(base))
    aged = age_book(book, new_rd, bank.generation.avg_rate_map, loan_models)

    # scaled products: same accounts, balances to target
    key = ['product_code', 'bs_side']
    tgt = (targets['target'] / targets['to_home_new']).set_axis(pd.MultiIndex.from_frame(targets[key]))
    sc = aged['product_name'].isin(SCALED)
    stopped = [(pc, side) for (pc, side), grp in aged[sc].groupby(key) if tgt[(pc, side)] <= 0]
    if stopped:                     # target 0: the product's accounts close (no 0-balance rows)
        aged = aged[~pd.MultiIndex.from_frame(aged[key]).isin(stopped)].copy()
        sc = aged['product_name'].isin(SCALED)
    # NMDs with a behavioural model run off instead: accounts close, survivors shrink, the volume
    # up to the target comes as new accounts (new business below)
    months = (new_rd.year - base_rd.year) * 12 + new_rd.month - base_rd.month
    haz = nmd_hazards(base, months)
    runoff, ledgers = set(), []
    for (pc, side), grp in aged[aged['product_name'].isin(NMD)].groupby(key):
        if int(pc) not in haz:
            print(f'  NMD {pc}: no behavioural model column in dep_beh_models_liq.xlsx -- scaled to target, no run-off')
            continue
        n0, b0 = len(grp), grp['balance_amt'].sum()
        led = nmd_ledger(b0, tgt[(pc, side)], haz[int(pc)], base_rd)
        ledgers.append(led.assign(product_code=pc, bs_side=side, product_name=grp['product_name'].iloc[0],
                                  currency=grp['currency'].iloc[0]))
        stock_surv = float(np.prod(1.0 - haz[int(pc)]))          # today's balance still there at month 12
        aged = run_off_nmd(aged, int(pc), side, 1.0 - stock_surv, tgt[(pc, side)], bank.generation.seed)
        g1 = aged[(aged['product_code'] == pc) & (aged['bs_side'] == side)]
        print(f'  NMD {pc}: run-off {1 - stock_surv:.1%} over {months}M (liq model, day-1 drop excluded), outflow '
              f'{led["outflow"].sum() / 1e6:,.1f}M, inflow {led["inflow"].sum() / 1e6:,.1f}M -> {n0 - len(g1)} of {n0} accounts closed, '
              f'{b0 / 1e6:,.1f}M -> {g1["balance_amt"].sum() / 1e6:,.1f}M existing, target {tgt[(pc, side)] / 1e6:,.1f}M')
        runoff.add((pc, side))
    sc = aged['product_name'].isin(SCALED) & ~pd.MultiIndex.from_frame(aged[key]).isin(list(runoff))
    for (pc, side), grp in aged[sc].groupby(key):
        f = tgt[(pc, side)] / grp['balance_amt'].sum()
        for c in ('balance_amt', 'init_balance_amt', 'credit_limit', 'undrawn_amt'):
            if c in aged.columns:
                aged.loc[grp.index, c] = aged.loc[grp.index, c] * f
    # dates that are defined relative to the report date
    cal = get_calendar_from_currency(bank.market.ccy)
    nrd_ql = ql.Date.from_date(new_rd.date())
    card = aged['product_name'].isin({'credit_card_revolver', 'credit_card_transactor'})
    for mat in aged.loc[card, 'maturity'].dropna().unique():
        m = card & (aged['maturity'] == mat)
        aged.loc[m, 'maturity_date'] = ql_date_to_pd_date(cal.advance(nrd_ql, ql.Period(mat), ql.ModifiedFollowing))
    cash = aged['product_name'] == 'cash'
    aged.loc[cash, 'start_date'] = new_rd
    aged.loc[cash, 'maturity_date'] = ql_date_to_pd_date(cal.advance(nrd_ql, ql.Period(1, ql.Days), ql.Following))

    # marketable securities above target are sold pro rata (every holding keeps the same share);
    # the sale result is not booked
    mk = aged['product_name'].isin(MARKETABLE)
    for (pc, side), grp in aged[mk].groupby(key):
        held = grp['balance_amt'].sum()
        if held > tgt[(pc, side)] + 1.0:
            f = tgt[(pc, side)] / held
            for c in ('balance_amt', 'init_balance_amt'):
                if c in aged.columns:
                    aged.loc[grp.index, c] = aged.loc[grp.index, c] * f
            print(f'  {pc} {side}: sold {(held - tgt[(pc, side)]) / 1e6:,.1f}M of {held / 1e6:,.1f}M (pro rata) to reach the target')

    # new business = target - survivors; a product whose survivors already exceed the
    # target keeps them (effective target = survivors) and the plug re-balances
    # in the home currency (foreign survivors at the new FX rate); new business back in its own
    surv = aged.groupby(key)['balance_amt'].sum()
    gaps = targets[key + ['product_name', 'target', 'plug', 'to_home_new']].copy()
    gaps['survived'] = [surv.get((p, s), 0.0) * f for p, s, f in
                        zip(gaps['product_code'], gaps['bs_side'], gaps['to_home_new'])]
    contractual = ~gaps['product_name'].isin(SCALED) | pd.MultiIndex.from_frame(gaps[key]).isin(list(runoff))
    over = gaps[contractual & (gaps['survived'] > gaps['target'] + 1.0)]
    gaps['target_eff'] = np.where(contractual, np.maximum(gaps['target'], gaps['survived']), gaps['target'])
    pl = gaps['plug']
    gaps.loc[pl, 'target_eff'] = (gaps.loc[gaps['bs_side'] == 'A', 'target_eff'].sum()
                                  - gaps.loc[(gaps['bs_side'] != 'A') & ~pl, 'target_eff'].sum())
    if (gaps.loc[pl, 'target_eff'] < gaps.loc[pl, 'survived']).any():
        raise ValueError('plug product would have to shrink below its surviving contracts -- pick another plug')
    # scaled products (accounts, cards, cash, equity) grow by scaling their accounts; one with
    # no accounts yet (a catalogue product at 0% in the base book) opens new ones instead
    opened = ~contractual & (gaps['survived'] <= 0.0)
    gaps['new_business'] = np.where(contractual | opened, gaps['target_eff'] - gaps['survived'], 0.0) / gaps['to_home_new']
    targets['target'] = gaps['target_eff'].to_numpy()

    build_inputs(bank, base, targets, fx_roll)
    print(f'  inputs written (bank_data, models, curves/fixings: market_path={rf.market_path!r}, IRS book)')
    bank_data = bank.balance_dir() / 'bank_data.xlsx'
    bs_struct = catalogue.read_bs_structure(bank_data)
    bs_struct['amort_type'] = bs_struct['amort_type'].astype('Int64')
    commit = cmt.read_commitments(bank_data)
    bs_struct = cmt.inject_card_terms(bs_struct, commit)       # card utilisation for new card business

    new = generate_new_business(bs_struct, gaps[gaps['new_business'] > 1.0], base_rd, new_rd, bank.generation.seed)
    new = _renumber(new, book['transaction_id'])
    new = decorate_from_bs_struct(new, bs_struct)
    new['vintage'] = 'new'
    aged['vintage'] = 'existing'

    all_tx = pd.concat([aged, new], ignore_index=True)
    interest_rt = rate_models.read_rate_excel(new_rd).reset_index()
    fixings = fxm.read_market_input('fixing_input.xlsx', bank).rename(columns={'date': 'fixing_date'})
    all_tx = all_tx.drop(columns=['index_rt', 'client_rt'], errors='ignore')
    all_tx = compute_deposit_client_rt(all_tx, interest_rt, bs_struct, fixings, new_rd)
    all_tx = compute_asset_rates(all_tx, interest_rt, bs_struct, fixings, new_rd)
    all_tx['product_code'] = all_tx['product_code'].astype(int)
    all_tx = sql_setup.add_client_id(all_tx.drop(columns=['client_id'], errors='ignore'), seed=42, pct=0.4)

    # write
    ensure_database(bank)
    sql_setup.reset_data(0, report_date=new_rd)
    sql_setup.append_df_to_table(all_tx, 'transactions')
    for cls, table in ProductFactory.table_registry.items():
        pnames = [k for k, v in ProductFactory.class_registry.items() if v is cls]
        subset = all_tx[all_tx['product_name'].isin(pnames)]
        if not subset.empty:
            sql_setup.append_df_to_table(subset, table)

    # committed facilities on the rolled book; FX hedge ladder rolled
    ta_new = float(targets.loc[targets['bs_side'] == 'A', 'target'].sum())
    reg = cmt.write_register(sql_setup.engine, commit, all_tx, ta_new, to_home_new, new_rd)
    if not reg.empty:
        print(reg[['commitment_code', 'own_name', 'limit_amt', 'drawn_amt', 'undrawn_amt']].to_string(index=False))
    roll_fx_hedge(bank, base, all_tx, fixings, fx_roll)
    roll_cirs_hedge(bank, base, all_tx, fixings, fx_roll)

    # summary (home currency)
    all_tx['_home'] = all_tx['balance_amt'] * all_tx['currency'].map(lambda c: to_home_new.get(c, 1.0))
    by = all_tx.groupby(key + ['vintage'])['_home'].sum().unstack('vintage').fillna(0.0)
    cnt = all_tx.groupby(key + ['vintage']).size().unstack('vintage').fillna(0).astype(int)
    summary = targets[key + ['own_name', 'balance_base', 'growth_pct', 'target', 'plug']].merge(
        by.reset_index(), on=key, how='left').merge(
        cnt.add_prefix('n_').reset_index(), on=key, how='left')
    summary['matured_or_repaid'] = np.where(contractual, summary['balance_base'] - gaps['survived'].to_numpy(), 0.0)
    out_dir = bank.output_dir('balance_generate')
    summary.to_excel(out_dir / 'rollforward_summary.xlsx', index=False)
    if ledgers:                     # NMD monthly flows: opening / outflow / inflow / closing per product
        led = pd.concat(ledgers, ignore_index=True)[['product_code', 'bs_side', 'product_name', 'currency', 'month', 'date',
                                                     'opening', 'outflow', 'inflow', 'closing', 'outflow_rate']]
        led.to_excel(out_dir / 'nmd_ledger.xlsx', index=False)
        led.assign(report_date=new_rd).to_sql('nmd_ledger', sql_setup.engine, schema='schemat',
                                              if_exists='replace', index=False)
    pd.set_option('display.width', 220)
    show = summary.copy()
    for c in ('balance_base', 'target', 'existing', 'new', 'matured_or_repaid'):
        show[c] = show[c] / 1e6
    print(show.round(1).to_string(index=False))
    print(f'  total assets {targets.loc[targets.bs_side == "A", "balance_base"].sum()/1e6:,.0f}M -> '
          f'{all_tx.loc[all_tx.bs_side == "A", "_home"].sum()/1e6:,.0f}M; '
          f'liabilities+equity {all_tx.loc[all_tx.bs_side != "A", "_home"].sum()/1e6:,.0f}M')
    if not over.empty:
        print('  NOTE: run-off left these products ABOVE target (kept, plug absorbs):')
        print(over.to_string(index=False))
    return summary
