"""New IRS of a 1-year roll-forward: inputs/<base bank>/forecast/new_swaps.xlsx.

The swaps the bank HOLDS are in its main input, inputs/<bank>/derivatives/derivatives.xlsx
(sheet ``swap``): the roll-forward keeps the ones still alive on the new report date. This file
only lists trades the bank plans to make INSIDE the forecast window -- start date after the base
report date and on / before the new report date (no back-dated trades).

Sheet ``new_swaps``
  B1  option   0 = no new IRS (the book only runs off)
               1 = the rows below, notional as listed (the swaps of the last 12 months, traded again
                   one year later: same months, same notionals, same terms)
               2 = the same rows, notional x (1 + balance growth)
  B2  balance growth % for option 2 (blank = growth_plan.xlsx B1, same folder)
  table (header row 'swap_id'): one row per planned swap. notional_opt2 is a formula for the
  reader (the roll-forward recalculates it). fixed_rate blank = the rolled market's IRS quote of
  the swap's tenor on its start date (how the base ladder was priced); a number = that rate.
  Rows can be deleted, added or edited by hand.

Build / refresh the table from the current swap book (keeps B1 / B2):
    python balance_generate/python_code/forecast_swaps.py --bank eur
"""
from __future__ import annotations

import argparse
import os
import sys

import openpyxl
import pandas as pd
import QuantLib as ql
from openpyxl.styles import Font, PatternFill

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common import derivatives as deriv  # noqa: E402
from labbank_common.bank_config import BankConfig, get_bank  # noqa: E402

SHEET = 'new_swaps'
FIRST = 6                                   # header row 5, first swap row 6
OPTIONS = {0: 'no new IRS', 1: 'same notionals', 2: 'notionals grown with the balance sheet'}
INPUT = PatternFill('solid', fgColor='DDEBF7')
HEADER = PatternFill('solid', fgColor='EDEDED')
COLS = ['swap_id', 'like_swap_id', 'start_date', 'maturity_date', 'notional', 'notional_opt2',
        'fixed_rate'] + [c for c in deriv.SWAP_COLUMNS
                         if c not in ('report_date', 'swap_id', 'notional', 'start_date', 'maturity_date',
                                      'fixed_rate')]


def path_for(base: BankConfig):
    return base.forecast_dir() / (base.rollforward.swaps_file if base.rollforward else 'new_swaps.xlsx')


def _months(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + b.month - a.month


def _plan_growth(base: BankConfig) -> float:
    from b_s_rollforward import read_growth_plan
    return read_growth_plan(base.forecast_dir() / 'growth_plan.xlsx')[0]


def template_rows(book: pd.DataFrame, base_rd: pd.Timestamp) -> pd.DataFrame:
    """The swaps that started in the 12 months up to the base report date, one year later."""
    from b_s_gen_objects import get_calendar_from_currency, ql_date_to_pd_date
    yr = book[(book['start_date'] > base_rd - pd.DateOffset(years=1)) & (book['start_date'] <= base_rd)]
    # HDG_ swaps are sized again every year, not repeated
    yr = yr[~yr['swap_id'].astype(str).str.startswith('HDG_')]
    if 'swap_type' in yr.columns:          # CIRS follow the FX hedge rule (b_s_rollforward.roll_cirs_hedge)
        yr = yr[yr['swap_type'].fillna('IRS').astype(str).str.upper() == 'IRS']
    yr = yr.sort_values(['start_date', 'swap_id'])
    nums = pd.to_numeric(book['swap_id'].astype(str).str.extract(r'(\d+)$')[0], errors='coerce')
    nxt = int(nums.max()) + 1 if nums.notna().any() else 1
    rows = []
    for k, r in enumerate(yr.itertuples(index=False)):
        cal = get_calendar_from_currency(r.currency)
        start = cal.advance(ql.Date.from_date(r.start_date.date()), ql.Period(1, ql.Years), ql.ModifiedFollowing)
        mat = cal.advance(start, ql.Period(_months(r.start_date, r.maturity_date), ql.Months), ql.ModifiedFollowing)
        row = {c: getattr(r, c) for c in COLS if hasattr(r, c)}
        row.update(swap_id=f'IRS_{nxt + k:03d}', like_swap_id=r.swap_id,
                   start_date=ql_date_to_pd_date(start), maturity_date=ql_date_to_pd_date(mat),
                   notional=float(r.notional), notional_opt2=None, fixed_rate=None)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLS)


def _settings(path) -> tuple[int, float | None]:
    raw = pd.read_excel(path, sheet_name=SHEET, header=None, nrows=2)
    option = int(raw.iloc[0, 1])
    if option not in OPTIONS:
        raise ValueError(f'{path} {SHEET}!B1: option must be one of {sorted(OPTIONS)}, got {raw.iloc[0, 1]!r}')
    g = pd.to_numeric(raw.iloc[1, 1], errors='coerce')
    return option, (None if pd.isna(g) else float(g))


def build(bank_id: str) -> None:
    base = get_bank(bank_id)
    path = path_for(base)
    option, growth = _settings(path) if path.exists() else (0, None)
    base_rd = pd.Timestamp(base.report_date)
    fc = template_rows(deriv.read_swap_book(base), base_rd)
    g_plan = _plan_growth(base)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = SHEET
    ws['A1'], ws['B1'] = 'option', option
    ws['C1'] = '; '.join(f'{k} = {v}' for k, v in OPTIONS.items())
    ws['A2'], ws['B2'] = 'balance_growth_pct', growth
    ws['C2'] = f'option 2 only; blank = growth_plan.xlsx B1 (now {g_plan:g}%)'
    ws['A3'] = (f'Planned trades only: start_date after {base_rd.date()} (base report date) and on / before the '
                f'new report date. The swaps held today are in inputs/{base.bank_id}/balance/'
                f'derivatives.xlsx. Prefilled: the swaps of {(base_rd - pd.DateOffset(years=1)).date()} - '
                f'{base_rd.date()} one year later; fixed_rate blank = rolled IRS quote on the start date.')
    for c in ('B1', 'B2'):
        ws[c].fill = INPUT
    for j, h in enumerate(COLS, start=1):
        cell = ws.cell(row=FIRST - 1, column=j, value=h)
        cell.font, cell.fill = Font(bold=True), HEADER
        ws.column_dimensions[cell.column_letter].width = 15
    ln = openpyxl.utils.get_column_letter(COLS.index('notional') + 1)
    for i, r in enumerate(fc.itertuples(index=False), start=FIRST):
        for j, h in enumerate(COLS, start=1):
            v = getattr(r, h)
            if h == 'notional_opt2':
                v = f'=ROUND({ln}{i}*(1+IF($B$2="",{g_plan},$B$2)/100),-6)'
            elif isinstance(v, pd.Timestamp):
                v = v.to_pydatetime()
            elif v is not None and not isinstance(v, str) and pd.isna(v):
                v = None
            ws.cell(row=i, column=j, value=v)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    print(f'{path}: {len(fc)} planned swaps, {fc["notional"].sum() / 1e6:,.0f}M (option {option}: {OPTIONS[option]})')


def read(rolled: BankConfig) -> tuple[pd.DataFrame, int, float] | None:
    """(new swaps with the notional of the chosen option, option, growth %) for a rolled bank,
    or None without a file. Option 0 returns an empty frame. Fixed rates are not set here."""
    base = get_bank(rolled.rollforward.base_bank)
    path = base.forecast_dir() / rolled.rollforward.swaps_file
    if not path.exists():
        return None
    option, growth = _settings(path)
    if growth is None:
        growth = _plan_growth(base)
    base_rd, new_rd = pd.Timestamp(base.report_date), pd.Timestamp(rolled.report_date)
    if base.rollforward is None:
        fc = pd.read_excel(path, sheet_name=SHEET, header=FIRST - 2)
        fc = fc[fc['swap_id'].notna()].copy()
    else:
        # the base is itself a rolled bank: the file's rows are dated for the first window.
        # The rule stays (B1 option / B2 growth of the chain root's file); the trades are the base's own
        # last 12 months (its new swaps) one year later, as `build` prefills the file for a generated bank
        fc = template_rows(deriv.read_swap_book(base), base_rd)
        print(f'  new swaps from the last 12 months of {base.bank_id}: {len(fc)} trades (rule of {path})')
    for c in ('start_date', 'maturity_date'):
        fc[c] = pd.to_datetime(fc[c])
    bad = fc[(fc['start_date'] <= base_rd) | (fc['start_date'] > new_rd)]
    if not bad.empty:
        raise ValueError(f'{path.name}: start_date must be in ({base_rd.date()}, {new_rd.date()}] -- no back-dated '
                         f'or forward-starting trades: {bad["swap_id"].tolist()}')
    held = set(deriv.read_swap_book(base)['swap_id'])
    if held & set(fc['swap_id']):
        raise ValueError(f'{path.name}: swap_id already in the held book: {sorted(held & set(fc["swap_id"]))}')
    fc['notional'] = pd.to_numeric(fc['notional'])
    if option == 2:
        fc['notional'] = (fc['notional'] * (1 + growth / 100.0) / 1e6).round() * 1e6
    if 'float_notional' in fc.columns:     # single-currency IRS: both legs on the (grown) notional
        irs = fc.get('swap_type', pd.Series('IRS', index=fc.index)).fillna('IRS').astype(str).str.upper() == 'IRS'
        fc.loc[irs, 'float_notional'] = fc.loc[irs, 'notional']
    return (fc.iloc[0:0] if option == 0 else fc), option, growth


def price(fc: pd.DataFrame, curve_input: pd.DataFrame) -> pd.DataFrame:
    """fixed_rate of every swap without one: the quote 'IRS <ccy> <n>Y' on the last curve date
    <= its start date (decimal, 4 dp, as the base ladder)."""
    fc = fc.copy()
    fc['fixed_rate'] = pd.to_numeric(fc['fixed_rate'], errors='coerce')
    for i, r in fc[fc['fixed_rate'].isna()].iterrows():
        n = round(_months(r['start_date'], r['maturity_date']) / 12)
        q = curve_input[(curve_input['ir_type'] == f'IRS {r["currency"]} {n}Y')
                        & (curve_input['date'] <= r['start_date'])].sort_values('date')
        if q.empty:
            raise ValueError(f'new_swaps {r["swap_id"]}: no IRS {r["currency"]} {n}Y quote before {r["start_date"].date()}')
        fc.loc[i, 'fixed_rate'] = round(float(q['rate'].iloc[-1]) / 100.0, 4)
    return fc


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bank', default='eur', help='base bank whose roll-forward reads the file')
    build(ap.parse_args().bank)
