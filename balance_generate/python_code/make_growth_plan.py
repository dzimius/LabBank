"""Rebuild a bank's growth plan (inputs/<bank>/forecast/growth_plan.xlsx) from its bank_data.xlsx.

The growth plan must list exactly the products of bank_data.xlsx (bs_structure) -- the
roll-forward (b_s_rollforward.py) stops otherwise. After adding, removing or re-weighting
products in bank_data.xlsx, rebuild it:

    python balance_generate/python_code/make_growth_plan.py --bank eur

- balance_today_M = bank_data bs_percentage x 2 x total assets (the generated book, home ccy)
- share_new_pct (the input, blue) is kept from the existing file for every product already in
  it; a product new to the plan starts at today's composition (catalogue products, 0% today,
  at 0 = kept in the catalogue, not sold). Edit the blue cells to plan a different mix.
- currency: the product's own currency (balance_today_M / target_M are in the home currency).
- B1 (total assets growth %) and the absorbs_difference row are kept from the existing file
  (defaults: 0% and the first liability term deposit).
"""
from __future__ import annotations

import argparse
import os
import sys

import openpyxl
import pandas as pd
from openpyxl.styles import Font, PatternFill

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common.bank_config import get_bank, total_assets  # noqa: E402

INPUT = PatternFill('solid', fgColor='DDEBF7')     # blue: cells the user edits
HEADER = PatternFill('solid', fgColor='EDEDED')
COLS = ['product_code', 'bs_side', 'own_name', 'product_name', 'currency', 'balance_today_M', 'share_today_pct',
        'share_new_pct', 'target_M', 'growth_pct', 'absorbs_difference']
WIDTHS = [30, 8, 30, 22, 9, 16, 15, 14, 12, 11, 18]
FIRST = 6                                           # first product row (header on row 5)

NOTES = [
    'HOW TO USE',
    '1. Cell B1 (blue): growth of total assets over the year, in % (e.g. 2.65).',
    '2. Column share_new_pct (blue): the balance-sheet composition in one year, in % of total assets. '
    'Assets must sum to 100 and liabilities + equity must sum to 100 -- then assets = liabilities + equity automatically.',
    '3. Formulas show each product: target_M = total assets in 1 year x share, growth_pct = its growth vs today. '
    'The totals rows and CHECK show whether both sides sum to 100% (A - (L+E) = 0).',
    'Starting from product growth rates instead: work out the shares yourself (target / total in 1 year); '
    'the totals and CHECK confirm the result.',
    "share_today_pct is today's composition -- the generated file starts share_new_pct there (every product grows at B1).",
    'Catalogue products (bs_percentage 0 in bank_data.xlsx, nothing held today): share_new_pct 0 keeps them in the '
    'catalogue unsold; a positive share sells the whole target as new business (new accounts for current / savings '
    'accounts, cards, cash). share_new_pct 0 for a product held today: its contracts run off, accounts close.',
    'absorbs_difference = Y (one liability or equity row): used ONLY if run-off leaves a product above its target '
    '(it would have to shrink faster than its contracts mature); that product then absorbs the difference so the books still balance.',
    'The roll-forward (balance_generate/python_code/b_s_rollforward.py) reads B1 and the share_new_pct column and repeats '
    'this arithmetic; it stops if a side does not sum to 100% or the products differ from bank_data.xlsx.',
    'Existing contracts age (amortise, prepay, mature); only the volume missing versus target_M is generated as new business. '
    'Current / savings accounts, cards, cash and equity keep their accounts and scale to the target.',
    'Rebuild the +1Y books after editing:  python release_check/release_check.py --bank eur_y1 --run',
    'Products added to / removed from bank_data.xlsx: rebuild this file with  python balance_generate/python_code/make_growth_plan.py '
    '--bank <bank>  (keeps B1, the absorbs_difference row and the share_new_pct of products already in the plan; '
    'a new product starts at today\'s composition).',
]


def _old_settings(path) -> tuple[float, tuple[int, str] | None, dict]:
    if not path.exists():
        return 0.0, None, {}
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from b_s_rollforward import read_growth_plan
    growth, gp = read_growth_plan(path)
    plug = gp.loc[gp['plug'], ['product_code', 'bs_side']]
    shares = {(int(r.product_code), str(r.bs_side)): float(r.share_new_pct)
              for r in gp.itertuples() if pd.notna(r.share_new_pct)}
    return growth, (int(plug.iloc[0, 0]), str(plug.iloc[0, 1])) if len(plug) else None, shares


def build(bank_id: str) -> None:
    bank = get_bank(bank_id)
    folder = bank.balance_dir()
    path = bank.forecast_dir() / (bank.rollforward.growth_file if bank.rollforward else 'growth_plan.xlsx')
    path.parent.mkdir(parents=True, exist_ok=True)
    growth, plug, old_shares = _old_settings(path)
    from labbank_common import catalogue
    bs = catalogue.read_bs_structure(folder / 'bank_data.xlsx')   # catalogue variants: rows at 0%
    bs = bs.drop_duplicates(['product_code', 'bs_side'])
    side_order = {'A': 0, 'L': 1, 'E': 2}
    bs = bs.assign(_o=bs['bs_side'].map(side_order)).sort_values('_o', kind='stable').drop(columns='_o')
    if plug is None or not ((bs['product_code'] == plug[0]) & (bs['bs_side'] == plug[1])).any():
        td = bs[(bs['bs_side'] == 'L') & (bs['product_name'] == 'term_deposit')]
        plug = (int(td['product_code'].iloc[0]), 'L')
    ta_m = total_assets(bank) / 1e6

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'growth_plan'
    n = len(bs)
    last = FIRST + n - 1
    a_last = FIRST + int((bs['bs_side'] == 'A').sum()) - 1
    bold = Font(bold=True)
    ws['A1'], ws['B1'] = 'Total assets growth % (1 year)', growth
    ws['B1'].fill, ws['B1'].number_format = INPUT, '0.00'
    ws['A2'], ws['B2'] = 'Total assets today (M)', f'=SUM(F{FIRST}:F{a_last})'
    ws['A3'], ws['B3'] = 'Total assets in 1 year (M)', '=B2*(1+B1/100)'
    for c in ('B2', 'B3'):
        ws[c].number_format = '#,##0.0'
    for c in ('A1', 'A2', 'A3'):
        ws[c].font = bold
    for j, h in enumerate(COLS, 1):
        cell = ws.cell(FIRST - 1, j, h)
        cell.font, cell.fill = bold, HEADER
    for i, r in enumerate(bs.itertuples(), FIRST):
        today = float(r.bs_percentage) * 2.0 * ta_m / 100.0
        share = old_shares.get((int(r.product_code), r.bs_side), round(float(r.bs_percentage) * 2.0, 10))
        vals = [int(r.product_code), r.bs_side, r.own_name, r.product_name, r.currency, round(today, 4),
                f'=F{i}/$B$2*100', share, f'=$B$3*H{i}/100', f'=IF(F{i}=0,"",(I{i}/F{i}-1)*100)',
                'Y' if (int(r.product_code), r.bs_side) == plug else None]
        for j, v in enumerate(vals, 1):
            ws.cell(i, j, v)
        ws.cell(i, 6).number_format = ws.cell(i, 9).number_format = '#,##0.0'
        ws.cell(i, 7).number_format = ws.cell(i, 8).number_format = '0.000'
        ws.cell(i, 10).number_format = '+0.00;-0.00;0.00'
        ws.cell(i, 8).fill = ws.cell(i, 11).fill = INPUT
    t = last + 2
    for k, (label, lo, hi) in enumerate([('TOTAL ASSETS', FIRST, a_last),
                                         ('TOTAL LIABILITIES + EQUITY', a_last + 1, last)]):
        ws.cell(t + k, 3, label).font = bold
        for j, col, fmt in ((6, 'F', '#,##0.0'), (7, 'G', '0.000'), (8, 'H', '0.000'), (9, 'I', '#,##0.0')):
            c = ws.cell(t + k, j, f'=SUM({col}{lo}:{col}{hi})')
            c.number_format = fmt
    ws.cell(t + 2, 3, 'ASSETS - (LIABILITIES + EQUITY)').font = bold
    for j, col in ((6, 'F'), (7, 'G'), (8, 'H'), (9, 'I')):
        ws.cell(t + 2, j, f'={col}{t}-{col}{t + 1}')
    ws.cell(t + 3, 3, 'CHECK').font = bold
    ws.cell(t + 3, 8, f'=IF(AND(ABS(H{t}-100)<0.001,ABS(H{t + 1}-100)<0.001),'
                      '"OK: both sides 100%","FIX: each side must sum to 100%")')
    for j, w in enumerate(WIDTHS):
        ws.column_dimensions[chr(65 + j)].width = w
    ws.freeze_panes = f'A{FIRST}'

    notes = wb.create_sheet('notes')
    notes.column_dimensions['A'].width = 160
    for i, line in enumerate(NOTES + [f'balance_today_M is the {bank_id.upper()} book on its report date '
                                      f'(bank_data bs_percentage x total assets {ta_m:,.0f}M) -- rebuild this '
                                      'file if bank_data.xlsx changes.'], 1):
        notes.cell(i, 1, line)
    notes['A1'].font = bold
    wb.save(path)
    a = bs.loc[bs['bs_side'] == 'A', 'bs_percentage'].sum() * 2
    print(f'{path}: {n} products ({(bs["bs_percentage"] == 0).sum()} catalogue at 0%), growth {growth}%, '
          f'plug {plug}, asset shares sum {a:.4f}%')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bank', default='eur', help='bank whose bank_data.xlsx defines the products')
    build(ap.parse_args().bank)
