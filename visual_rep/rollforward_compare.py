"""Today's book vs the rolled-forward book (e.g. eur -> eur_y1): balance sheet, yields and
cost of funding, IRRBB SOT, liquidity.

Yield / cost of funding over the year is a SIMPLE path, by design: the balance-weighted
client rate of each book (start and end of the year) with the months in between
interpolated linearly. It shows the direction and size of the change from the book
structure and market data, not an exact monthly P&L.

Usage:  python visual_rep/rollforward_compare.py [--bank eur_y1]
Writes runs/<bank>/visual_rep/rollforward_compare.xlsx
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
from labbank_common.bank_config import get_bank, get_engine  # noqa: E402

RATE_TABLES = ['loans', 'deposits', 'financial_instruments']


def book_rates(bank) -> pd.DataFrame:
    """Balance and balance-weighted client rate per product (equity: no rate)."""
    eng = get_engine(bank)
    rd = pd.Timestamp(bank.report_date)
    parts = [pd.read_sql(text(f'SELECT product_code, own_name, bs_side, currency, CAST(balance_amt AS FLOAT) bal, '
                              f'CAST(client_rt AS FLOAT) rt FROM schemat.{t} WHERE report_date = :rd'),
                         eng, params={'rd': rd}) for t in RATE_TABLES]
    eq = pd.read_sql(text("SELECT product_code, own_name, 'E' bs_side, currency, CAST(balance_amt AS FLOAT) bal, "
                          "NULL rt FROM schemat.equity WHERE report_date = :rd"), eng, params={'rd': rd})
    df = pd.concat(parts + [eq], ignore_index=True)
    from labbank_common import fx              # foreign-currency products in the home currency
    df['bal'] = df['bal'] * fx.to_home_series(df['currency'], bank, rd)
    df['rt'] = df['rt'].fillna(0.0)
    df['int'] = df['bal'] * df['rt']
    g = df.groupby(['product_code', 'bs_side', 'own_name'], as_index=False)[['bal', 'int']].sum()
    g['rate_pct'] = np.where(g['bal'] > 0, g['int'] / g['bal'] * 100, np.nan)
    return g


def headline(g: pd.DataFrame) -> dict:
    a, l = g[g['bs_side'] == 'A'], g[g['bs_side'] == 'L']
    ta = a['bal'].sum()
    y = a['int'].sum() / ta * 100
    cof = l['int'].sum() / (l['bal'].sum() + g.loc[g['bs_side'] == 'E', 'bal'].sum()) * 100   # equity funds at 0%
    return {'total_assets': ta, 'asset_yield_pct': y, 'cost_of_funding_pct': cof, 'nim_static_pct': y - cof}


def irrbb(bank) -> pd.DataFrame:
    r = pd.read_sql('SELECT currency, scenario_id, nii_base, delta_nii, sot_nii_pct, eve_base, delta_eve, sot_eve_pct '
                    'FROM results.irrbb_report', get_engine(bank))
    if (r['currency'] == 'ALL').any():         # multi-currency bank: the all-currency total
        r = r[r['currency'] == 'ALL']
    return r.drop(columns='currency')


def liquidity(bank) -> pd.DataFrame:
    l = pd.read_sql('SELECT currency, lcr, nsfr FROM results.lcr_nsfr', get_engine(bank))
    if (l['currency'] == 'ALL').any():         # consolidated ratio, labelled with the home currency
        l = l[l['currency'] == 'ALL'].assign(currency=bank.market.ccy)
    return l


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='eur_y1')
    a = ap.parse_args()
    new = get_bank(a.bank)
    base = get_bank(new.rollforward.base_bank)
    d0, d1 = base.report_date, new.report_date

    g0, g1 = book_rates(base), book_rates(new)
    bs = g0.merge(g1, on=['product_code', 'bs_side', 'own_name'], how='outer', suffixes=(f'_{d0}', f'_{d1}'))
    bs['change_pct'] = (bs[f'bal_{d1}'] / bs[f'bal_{d0}'] - 1) * 100
    bs = bs.drop(columns=[c for c in bs.columns if c.startswith('int_')]).sort_values(['bs_side', 'product_code'])

    h0, h1 = headline(g0), headline(g1)
    months = np.arange(0, 13)
    path = pd.DataFrame({'month': months,
                         'date': [pd.Timestamp(d0) + pd.DateOffset(months=int(m)) for m in months]})
    for k in ('asset_yield_pct', 'cost_of_funding_pct', 'nim_static_pct'):
        path[k] = h0[k] + (h1[k] - h0[k]) * months / 12
    avg = {k: float(path[k].iloc[:12].mean()) for k in ('asset_yield_pct', 'cost_of_funding_pct', 'nim_static_pct')}

    head = pd.DataFrame([
        ('Total assets (M)', h0['total_assets'] / 1e6, h1['total_assets'] / 1e6, np.nan),
        ('Asset yield, % (balance-weighted client rate)', h0['asset_yield_pct'], h1['asset_yield_pct'], avg['asset_yield_pct']),
        ('Cost of funding, % (liabilities + equity at 0%)', h0['cost_of_funding_pct'], h1['cost_of_funding_pct'], avg['cost_of_funding_pct']),
        ('Static margin, pp', h0['nim_static_pct'], h1['nim_static_pct'], avg['nim_static_pct']),
    ], columns=['measure', d0, d1, 'average over the year'])

    r0, r1 = irrbb(base), irrbb(new)
    ir = r0.merge(r1, on='scenario_id', suffixes=(f'_{d0}', f'_{d1}'))
    liq = liquidity(base).merge(liquidity(new), on='currency', suffixes=(f'_{d0}', f'_{d1}'))

    out_dir = new.output_dir('visual_rep')
    path_xlsx = out_dir / 'rollforward_compare.xlsx'
    with pd.ExcelWriter(path_xlsx) as xw:
        head.to_excel(xw, sheet_name='headline', index=False)
        path.to_excel(xw, sheet_name='monthly_path', index=False)
        bs.to_excel(xw, sheet_name='balance_sheet', index=False)
        ir.to_excel(xw, sheet_name='irrbb_sot', index=False)
        liq.to_excel(xw, sheet_name='liquidity', index=False)
    pd.set_option('display.width', 220)
    print(head.round(3).to_string(index=False))
    print()
    show = bs.copy()
    for c in show.columns:
        if c.startswith('bal_'):
            show[c] = show[c] / 1e6
    print(show.round(2).to_string(index=False))
    print()
    print(ir[['scenario_id', f'sot_nii_pct_{d0}', f'sot_nii_pct_{d1}', f'sot_eve_pct_{d0}', f'sot_eve_pct_{d1}']]
          .round(2).to_string(index=False))
    print(f"NII base {r0['nii_base'].iloc[0]/1e6:,.1f}M -> {r1['nii_base'].iloc[0]/1e6:,.1f}M, "
          f"EVE base {r0['eve_base'].iloc[0]/1e6:,.1f}M -> {r1['eve_base'].iloc[0]/1e6:,.1f}M")
    print(liq.round(3).to_string(index=False))
    print(f'Saved -> {path_xlsx}')


if __name__ == '__main__':
    main()
