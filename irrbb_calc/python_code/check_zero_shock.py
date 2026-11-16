"""Zero-shock consistency check for the EVE engine.

Revalues the book with the SHOCKED-scenario code (compute_eve_shocked) on the unshocked
base curve. Every product must give the same EVE as compute_eve_base; any difference is
a constant offset that leaks into every shocked scenario (SOT, ±1 bp, ±100 bp) as if it
were rate risk.

Usage:  LABBANK_BANK=eur python irrbb_calc/python_code/check_zero_shock.py [--tol 0.01]
Also checks NII: in the cf.nii_* tables the +1 bp and -1 bp NII changes must cancel per product
((up + down) / 2 = the zero-shock offset; --tol-nii million).
Exit code 1 if any product differs by more than --tol million.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config      # noqa: E402
import sql_setup   # noqa: E402
import eve_calc_objects as eve_obj  # noqa: E402
import eba_shock_curves as esc      # noqa: E402
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common import rate_models  # noqa: E402
from labbank_common.bank_config import get_bank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol-nii', type=float, default=0.05, help='NII +-1bp offset tolerance per product, million')
    ap.add_argument('--tol', type=float, default=0.1, help='tolerance per product, million (forward-rate rounding on floating products is ~0.01-0.04M)')
    a = ap.parse_args()
    rd = config.report_date = pd.Timestamp(get_bank().report_date)
    ccy = get_bank().market.ccy
    caps, floors, ca, cb = rate_models.rate_maps(rate_models.load_rate_models(
        sql_setup.engine, rd, required_codes=rate_models.deposit_product_codes(sql_setup.engine)))
    beh = sql_setup.load_all_beh_schedules(rd)
    try:
        swaps = sql_setup.load_all_swap_beh_schedules(rd)
    except Exception:
        swaps = pd.DataFrame()
    beh = pd.concat([beh, swaps], ignore_index=True) if not swaps.empty else beh
    # every currency the bank books (home + foreign), each on its own curve
    disc_map = {c: f'{c}_disc_curve' for c in (ccy, *get_bank().market.foreign_ccys)}
    base_disc = pd.concat([esc.load_irrbb_disc_curve(sql_setup.engine, cn, 'base', rd)
                           for cn in disc_map.values()]).sort_index()
    # multi-curve bank: rate moves measured on the projection curves (as eve_calc_workflow)
    from labbank_common.curves import projection_curve_map
    names = set(sql_setup.load_mkt_curves(rd)['curve_name'])
    if any('_fwd_curve_' in n for n in names):
        proj = {c: n for c, n in projection_curve_map(disc_map).items() if n in names}
        base_proj = pd.concat([esc.load_irrbb_disc_curve(sql_setup.engine, n, 'base', rd)
                               for n in proj.values()]).sort_index()
        eve_obj.set_projection_curves(proj, base_proj)
        base_disc = pd.concat([base_disc, base_proj]).sort_index()
    else:
        eve_obj.set_projection_curves(dict(disc_map), base_disc)    # as eve_calc_workflow

    key = ['product_code', 'bs_side', 'rate_type']
    def by_product(df):
        df = df.copy()
        df['sign'] = (df['bs_side'] == 'A').map({True: 1.0, False: -1.0})
        return df

    b = by_product(beh)
    b['pv'] = ((b['capital_pmt'].fillna(0) + b['prepayment_pmt'].fillna(0) + b['int_pmt'].fillna(0))
               * b['base_df'].fillna(0) * b['sign'])
    base = b.groupby(key, dropna=False)['pv'].sum()

    # shocked-code path on the base curve, per product (re-run on each product slice)
    rows = []
    for k, g in beh.groupby(key, dropna=False):
        _, s = eve_obj.compute_eve_shocked(g, base_disc, disc_map, 'base', 0.0, caps, floors, ca, cb)
        rows.append((*k, float(s['pv_total'].sum())))
    shocked = pd.DataFrame(rows, columns=key + ['pv']).set_index(key)['pv']
    diff = ((shocked - base) / 1e6).rename('diff_M')
    bad = diff[diff.abs() > a.tol]
    print(f'{get_bank().bank_id} {rd.date()}: zero-shock EVE difference per product (M), tol {a.tol}')
    print(diff.round(4).to_string())
    print(f'TOTAL {diff.sum():+.4f}M  ->  {"FAIL" if len(bad) else "OK"}')

    # NII: +1 bp and -1 bp must cancel per product in the cf.nii_* tables written by the
    # pipeline (a 1 bp move is linear); (up + down) / 2 is the zero-shock offset
    from sqlalchemy import text
    q = text('SELECT scenario_id, product_code, bs_side, SUM(nii_total) v FROM cf.nii_own_scenarios '
             "WHERE report_date = :rd AND scenario_id IN ('own_1_up', 'own_1_dn') GROUP BY scenario_id, product_code, bs_side")
    qb = text('SELECT product_code, bs_side, SUM(nii_total) v FROM cf.nii_base_scenario '
              'WHERE report_date = :rd GROUP BY product_code, bs_side')
    own = pd.read_sql(q, sql_setup.engine, params={'rd': rd}).pivot_table(
        index=['product_code', 'bs_side'], columns='scenario_id', values='v')
    nb = pd.read_sql(qb, sql_setup.engine, params={'rd': rd}).set_index(['product_code', 'bs_side'])['v']
    off = ((own['own_1_up'] - nb) + (own['own_1_dn'] - nb)) / 2 / 1e6
    nbad = off[off.abs() > a.tol_nii]
    print(f'NII +-1bp offset per product (M), tol {a.tol_nii}: '
          + ', '.join(f'{k[0]}/{k[1]} {v:+.3f}' for k, v in off.items() if abs(v) > 0.005))
    print(f'NII TOTAL offset {off.sum():+.4f}M (+1bp {(own.own_1_up - nb).sum()/1e6:+.3f}M, '
          f'-1bp {(own.own_1_dn - nb).sum()/1e6:+.3f}M)  ->  {"FAIL" if len(nbad) else "OK"}')
    return 1 if (len(bad) or len(nbad)) else 0


if __name__ == '__main__':
    sys.exit(main())
