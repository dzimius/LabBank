"""NII over the roll-forward year on the MOVING balance sheet, on the IRRBB NII engine.

One engine for every NII number: the plan year is the IRRBB constant-balance-sheet NII of
the base bank plus the plan's balance changes, priced with the same renewal rule.

  constant balance sheet  irrbb nii_calc_objects.compute_nii_shocked_schedule on the base
                          bank's book and curve (= irrbb.nii_results 'base'): contractual
                          interest, floating / administered rates on the curve's forwards,
                          capital repaid inside the year renewed like for like on the
                          product's repricing period (labbank_common.curves.renewal_market_rate);
  plan changes            balance path of the roll-forward (rollforward_report.monthly_balances)
                          minus today's balance; each month's change is new business (or, if
                          negative, renewal not made) from mid-month to the year end, at the
                          same renewal rate on the product's repricing period and the same
                          client-rate model (beta, spread / average contract margin, floor, cap);
  swaps                   today's book inside the IRRBB rows (it runs off, no renewal), plus the
                          plan's new swaps (forecast_1y swap_changes.xlsx): notional x (index
                          forward from the start to the year end - fixed rate), ACT/360.

The rate path is the base bank's curve (forwards), as in the IRRBB NII. Each row's interest is
spread over the months it accrues in. Foreign-currency products in the home currency at the
base-date rate. Output layout unchanged (by_product_month / monthly), product 0000 = swaps.

Usage:  python visual_rep/dynamic_nii.py [--bank eur_y1]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'balance_generate', 'python_code'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labbank_common.bank_config import get_bank  # noqa: E402

NO_RATE = {'retained_earnings', 'common_shares', 'risk_reserves'}
NII_FLOOR = float('-inf')            # as nii_calc_workflow: limits come from the rate model


def _irrbb(base_id: str):
    """IRRBB modules bound to the base bank's database (they read LABBANK_BANK on import)."""
    os.environ['LABBANK_BANK'] = base_id
    sys.path.insert(0, os.path.join(ROOT, 'irrbb_calc', 'python_code'))
    for n in ('config', 'sql_setup', 'nii_calc_objects', 'eba_shock_curves'):   # other modules' namesakes
        sys.modules.pop(n, None)
    import config, sql_setup, nii_calc_objects, eba_shock_curves  # noqa: E401
    return config, sql_setup, nii_calc_objects, eba_shock_curves


def _month_shares(a: pd.Series, b: pd.Series, dates: list) -> np.ndarray:
    """Share of each interval [a, b] falling in month m (rows x 12), by days."""
    a = pd.to_datetime(a).to_numpy('datetime64[D]')
    b = pd.to_datetime(b).to_numpy('datetime64[D]')
    tot = (b - a).astype(float)
    out = np.zeros((len(a), 12))
    for m in range(12):
        lo = np.maximum(a, np.datetime64(dates[m].date()))
        hi = np.minimum(b, np.datetime64(dates[m + 1].date()))
        out[:, m] = np.clip((hi - lo).astype(float), 0.0, None)
    return np.divide(out, tot[:, None], out=np.zeros_like(out), where=tot[:, None] > 0)


def dynamic_nii(bank, path: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly interest per product: columns month, date, product_code, bs_side, own_name,
    avg_balance, rate, interest (signed: assets +, liabilities -), plus swaps as product 0000."""
    from rollforward_report import monthly_balances
    from labbank_common import catalogue, fx, rate_models
    from labbank_common.curves import renew_months, renewal_market_rate
    if path is None:
        path = monthly_balances(bank)
    base = get_bank(bank.rollforward.base_bank)
    config, sql_setup, nii_obj, esc = _irrbb(base.bank_id)
    rd = pd.Timestamp(base.report_date)
    hor = rd + pd.Timedelta(days=round(365.25))
    dates = [rd] + [rd + pd.offsets.MonthEnd(m) for m in range(1, 13)]
    to_home = fx.to_home(base, rd)

    # ── constant balance sheet: the IRRBB base NII rows ──────────────────────────────────
    maps = rate_models.rate_maps(rate_models.load_rate_models(
        sql_setup.engine, rd, required_codes=rate_models.deposit_product_codes(sql_setup.engine)))
    caps, floors, coeff_a, coeff_b = maps
    mkt = sql_setup.load_mkt_curves(rd)
    from labbank_common.curves import projection_curve_map
    cmap = {'PLN': 'PLN_disc_curve', 'EUR': 'EUR_disc_curve', 'USD': 'USD_disc_curve', 'GBP': 'GBP_disc_curve'}
    if mkt['curve_name'].str.contains('_fwd_curve_').any():
        cmap = {c: n for c, n in projection_curve_map(cmap).items() if n in set(mkt['curve_name'])}
    disc = pd.concat([esc.load_irrbb_disc_curve(sql_setup.engine, n, 'base', rd) for n in cmap.values()]).sort_index()
    beh = sql_setup.load_beh_schedules(rd, hor)
    sw = sql_setup.load_swap_beh_schedules(rd, hor)
    rows = nii_obj.compute_nii_shocked_schedule(
        pd.concat([beh, sw], ignore_index=True), disc, cmap, rd, 'base', 1.0, NII_FLOOR,
        caps_map=caps, floors_map=floors, coeff_a_map=coeff_a, coeff_b_map=coeff_b,
        base_disc_df=disc, return_rows=True)
    rows['pc'] = np.where(rows['source'] == 'ir_swap', 0, pd.to_numeric(rows['product_code'], errors='coerce'))
    rows['side'] = np.where(rows['source'] == 'ir_swap', 'A', rows['bs_side'])
    k_home = rows['currency'].map(to_home).astype(float).to_numpy()
    s_int = _month_shares(rows['cf_start_dt'].clip(lower=rd), rows['cf_end_dt'].clip(upper=hor), dates)
    s_ren = _month_shares(rows['cf_end_dt'].clip(upper=hor), pd.Series([hor] * len(rows)), dates)
    monthly = (s_int * (rows['nii_interest'].to_numpy() * k_home)[:, None]
               + s_ren * (rows['nii_renewal'].to_numpy() * k_home)[:, None])
    const = pd.DataFrame(monthly, columns=range(1, 13)).assign(pc=rows['pc'].astype(int).to_numpy(),
                                                              side=rows['side'].to_numpy())
    const = const.groupby(['pc', 'side']).sum()

    # ── plan changes: path minus today's balance, priced as new business ─────────────────
    struct = catalogue.read_bs_structure(bank.balance_dir() / 'bank_data.xlsx')
    struct['product_code'] = struct['product_code'].astype(int)
    prod = struct.drop_duplicates(['product_code', 'bs_side']).set_index(['product_code', 'bs_side'])
    parent = {int(v): int(p) for v, p in catalogue.parent_map(bank).items()}
    margin = (beh.assign(pc=pd.to_numeric(beh['product_code']), w=beh['outstanding_bal'].fillna(0.0))
              .dropna(subset=['margin']).groupby('pc')
              .apply(lambda g: np.average(g['margin'], weights=g['w']) if g['w'].sum() > 0 else np.nan))

    def _model_code(pc: int) -> str:
        """Rate-model / margin key: the product, else its catalogue parent, else the family main code."""
        for c in (pc, parent.get(pc), pc // 10 * 10):
            if c is not None and (str(c) in coeff_a or c in margin.index):
                return str(c)
        return str(pc)

    path = path.set_index(['month', 'product_code', 'bs_side'])['total']
    mid = [(dates[m - 1] + (dates[m] - dates[m - 1]) / 2).normalize() for m in range(1, 13)]
    grow = {}
    for (pc, side), p in prod.iterrows():
        if p['product_name'] in NO_RATE or (0, pc, side) not in path.index:
            continue
        tot = np.array([path.loc[(m, pc, side)] for m in range(13)], dtype=float)
        g = np.diff(tot - tot[0])                                  # change originated in month m
        if not np.any(np.abs(g) > 1.0):
            continue
        mc = _model_code(pc)
        ccy = str(p['currency'])
        cn = cmap.get(ccy)

        def _df_at(d, cn=cn):
            idx = pd.MultiIndex.from_arrays([[cn] * len(d), pd.to_datetime(pd.Series(d)).to_numpy()],
                                            names=['curve_name', 'node_date'])
            return disc.reindex(idx).to_numpy(dtype=float)

        months = renew_months([p['fixing_freq']], [p['rate_type']],
                              [catalogue_months(p['maturity'])])[0]
        mkt_rt = renewal_market_rate(pd.Series(mid), hor, np.full(12, months), _df_at, NII_FLOOR)
        cm = float(margin.get(int(mc), np.nan)) if int(mc) in margin.index else np.nan
        cl_rt = nii_obj._apply_rt_limits(mkt_rt, pd.Series([mc] * 12), caps, floors, coeff_a, coeff_b,
                                         contract_margin=np.full(12, cm))
        sign = 1.0 if side == 'A' else -1.0
        acc = _month_shares(pd.Series(mid), pd.Series([hor] * 12), dates) * \
            ((pd.Series([hor] * 12) - pd.Series(mid)).dt.days.to_numpy() / 365.0)[:, None]
        grow[(pc, side)] = sign * (g * cl_rt) @ acc               # interest per month (12,)

    # ── the plan's new swaps ─────────────────────────────────────────────────────────────
    new_sw = np.zeros(12)
    sc = bank.output_dir('forecast_1y') / 'swap_changes.xlsx'
    if sc.exists():
        ns = pd.read_excel(sc, sheet_name='new_swaps')
        cn = cmap.get(base.market.ccy)

        def _df_eur(d):
            idx = pd.MultiIndex.from_arrays([[cn] * len(d), pd.to_datetime(pd.Series(d)).to_numpy()],
                                            names=['curve_name', 'node_date'])
            return disc.reindex(idx).to_numpy(dtype=float)
        for s in ns.itertuples():
            st, en = pd.Timestamp(s.start_date), min(pd.Timestamp(s.maturity_date), hor)
            if st >= hor:
                continue
            flt = renewal_market_rate(pd.Series([st]), en, None, _df_eur, NII_FLOOR)[0]
            sgn = 1.0 if int(s.pay_fixed) else -1.0
            sh = _month_shares(pd.Series([st]), pd.Series([en]), dates)[0]
            new_sw += sgn * float(s.notional) * (flt - float(s.fixed_rate)) * (en - st).days / 360.0 * sh

    out = []
    keys = sorted(set(const.index) | set(grow) | {(0, 'A')})
    for (pc, side) in keys:
        name = 'irs_book' if pc == 0 else (prod.loc[(pc, side), 'own_name'] if (pc, side) in prod.index else str(pc))
        if pc != 0 and (pc, side) in prod.index and prod.loc[(pc, side), 'product_name'] in NO_RATE:
            continue
        inte = (const.loc[(pc, side)].to_numpy() if (pc, side) in const.index else np.zeros(12)) \
            + grow.get((pc, side), np.zeros(12)) + (new_sw if pc == 0 else 0.0)
        for m in range(1, 13):
            if pc == 0:
                bal = np.nan
            elif (m, pc, side) in path.index:
                bal = (path.loc[(m - 1, pc, side)] + path.loc[(m, pc, side)]) / 2.0
            else:
                bal = 0.0
            days = (dates[m] - dates[m - 1]).days
            rate = inte[m - 1] / (abs(bal) * days / 365.0) if bal and np.isfinite(bal) else np.nan
            out.append(dict(month=m, date=dates[m], product_code=pc, bs_side=side, own_name=name,
                            avg_balance=bal, rate=rate, interest=inte[m - 1]))
    res = pd.DataFrame(out)
    # the year in one engine: constant balance sheet (= IRRBB base NII) + the plan's changes by group
    grp = {'loans': 0.0, 'securities_cash': 0.0, 'funding': 0.0}
    for (pc, side), v in grow.items():
        g = 'funding' if side != 'A' else ('securities_cash' if str(pc)[0] in '37' else 'loans')
        grp[g] += float(v.sum())
    res.attrs['year'] = dict(constant=float(const.to_numpy().sum()), new_swaps=float(new_sw.sum()), **grp,
                             moving=float(res['interest'].sum()))
    return res


def catalogue_months(maturity) -> float:
    """Contract term in months of a catalogue maturity ('25Y', '6M'); NaN for 'NM' / '1D'."""
    t = str(maturity).strip().upper()
    try:
        n = float(t[:-1])
    except ValueError:
        return np.nan
    return {'Y': n * 12, 'M': n}.get(t[-1], np.nan)


def summary(bank, df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby('month').apply(lambda g: pd.Series({
        'interest_income': g.loc[(g.bs_side == 'A') & (g.product_code != 0), 'interest'].sum(),
        'interest_expense': g.loc[g.bs_side != 'A', 'interest'].sum(),
        'swaps': g.loc[g.product_code == 0, 'interest'].sum()}))
    m['nii'] = m.sum(axis=1)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='eur_y1')
    ap.add_argument('--constant', action='store_true',
                    help='constant balance sheet (path = today): must equal the IRRBB base NII')
    a = ap.parse_args()
    bank = get_bank(a.bank)
    path = None
    if a.constant:
        from rollforward_report import monthly_balances
        path = monthly_balances(bank)
        t0 = path[path['month'] == 0].set_index(['product_code', 'bs_side'])['total']
        path['total'] = path.set_index(['product_code', 'bs_side']).index.map(t0).to_numpy()
    df = dynamic_nii(bank, path)
    s = summary(bank, df)
    pd.set_option('display.width', 200)
    print((s / 1e6).round(2).to_string())
    print(f"Year: NII {s['nii'].sum()/1e6:,.1f}M (income {s['interest_income'].sum()/1e6:,.1f}, "
          f"expense {s['interest_expense'].sum()/1e6:,.1f}, swaps {s['swaps'].sum()/1e6:,.1f})")
    if a.constant:
        return
    by = df.groupby(['product_code', 'bs_side', 'own_name'])['interest'].sum() / 1e6
    print(by.round(1).to_string())
    out = bank.output_dir('visual_rep') / 'dynamic_nii.xlsx'
    with pd.ExcelWriter(out) as xw:
        df.to_excel(xw, sheet_name='by_product_month', index=False)
        s.reset_index().to_excel(xw, sheet_name='monthly', index=False)
        pd.Series(df.attrs['year']).rename('eur').rename_axis('item').reset_index().to_excel(
            xw, sheet_name='year', index=False)
    print({k: round(v / 1e6, 1) for k, v in df.attrs['year'].items()})
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
