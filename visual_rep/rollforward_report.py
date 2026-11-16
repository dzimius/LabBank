"""One-year roll-forward report (static HTML): today's book vs the rolled-forward books.

Balance sheet in time, month by month:
  existing contracts -- the base book aged to each month-end with the roll-forward's own
                        rules (amortisation, prepayment model, maturity), exact per contract;
  product total      -- base balance grown monthly by the growth plan, (1 + g)^(m/12);
                        the plug product balances assets = liabilities + equity; a product
                        whose surviving contracts already exceed its path keeps them
                        (bonds and T-bills are sold pro rata down to the path instead);
  new business       -- total minus existing (non-maturity products, cards, cash and equity
                        keep their accounts: no split).
Month 12 equals the rolled book in SQL (checked below).

Yield / cost of funding over the year: start and end books' balance-weighted client rates,
months in between interpolated (rollforward_compare.py), one line per market path.

Usage:  python visual_rep/rollforward_report.py [--banks eur_y1]
Writes  runs/<first bank>/visual_rep/rollforward_report.html
"""
from __future__ import annotations

import argparse
import html
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import text

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'balance_generate', 'python_code'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labbank_common.bank_config import get_bank, get_engine  # noqa: E402

# reference categorical palette (dataviz skill), fixed order -- colour follows the entity
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e7e6e2', '#fcfcfb'

CLASSES = {   # aggregated classes for the stacked composition, in stack order
    'A': [('Mortgages', [(1000, 'A'), (1100, 'A')]),
          ('Consumer loans & cards', [(2000, 'A'), (2100, 'A'), (2200, 'A')]),
          ('Corporate & SME loans', [(4000, 'A'), (4100, 'A'), (4200, 'A'), (4150, 'A')]),
          ('Government bonds', [(3000, 'A'), (3100, 'A')]),
          ('Liquidity: cash, T-bills, interbank', [(3500, 'A'), (3510, 'A'), (3200, 'A'), (7900, 'A')])],
    'L': [('Current accounts', [(6000, 'L'), (6300, 'L'), (6350, 'L')]),
          ('Savings accounts', [(8000, 'L')]),
          ('Term deposits', [(7060, 'L'), (7070, 'L'), (7900, 'L')]),
          ('Covered bonds', [(5000, 'L')]),
          ('Equity', [(5100, 'E'), (5300, 'E'), (5400, 'E')])],
}
LINE_GROUPS = [
    ('Loans', [(1000, 'A'), (1100, 'A'), (2000, 'A'), (2100, 'A'), (4000, 'A'), (4100, 'A'), (4200, 'A'), (4150, 'A'), (2200, 'A')]),
    ('Securities and liquidity', [(3000, 'A'), (3100, 'A'), (3200, 'A'), (3500, 'A'), (3510, 'A'), (7900, 'A')]),
    ('Deposits', [(6000, 'L'), (8000, 'L'), (7060, 'L'), (7070, 'L'), (7900, 'L'), (6300, 'L'), (6350, 'L')]),
    ('Covered bonds and equity', [(5000, 'L'), (5300, 'E'), (5100, 'E'), (5400, 'E')]),
]
LABEL = {
    (1000, 'A'): 'Mortgages fixed', (1100, 'A'): 'Mortgages floating', (2000, 'A'): 'Cash loans fixed',
    (2100, 'A'): 'Cash loans floating', (2200, 'A'): 'Credit cards', (4000, 'A'): 'SME loans fixed', (4100, 'A'): 'SME loans floating', (4200, 'A'): 'Corporate investment loans', (4150, 'A'): 'Corporate loans USD',
    (3000, 'A'): 'Gov. bonds fixed', (3100, 'A'): 'Gov. bonds floating', (3200, 'A'): 'T-bills',
    (3500, 'A'): 'Cash (vault + min. reserves)', (3510, 'A'): 'Central bank reserves', (7900, 'A'): 'Interbank placements',
    (6000, 'L'): 'Current accounts retail', (6300, 'L'): 'Current accounts SME', (6350, 'L'): 'Current accounts corporate', (8000, 'L'): 'Savings accounts',
    (7060, 'L'): 'Term deposits retail', (7070, 'L'): 'Term deposits SME', (7900, 'L'): 'Interbank deposits', (5000, 'L'): 'Covered bond',
    (5100, 'E'): 'Common shares', (5300, 'E'): 'Retained earnings', (5400, 'E'): 'Risk reserves',
}



PATH_LABEL = {'forwards': 'rates follow forwards', 'unchanged': 'curve unchanged',
              'hybrid': 'hybrid curve (OIS forwards to 2Y, unchanged from 5Y)'}


def _label(k) -> str:
    """Short label of (product_code, bs_side); products not listed above (e.g. a new catalogue
    product) fall back to labbank_common.product_labels."""
    if k in LABEL:
        return LABEL[k]
    from labbank_common.product_labels import product_label
    return product_label(str(k[0]), k[1])


# ── data ─────────────────────────────────────────────────────────────────────

def monthly_balances(bank) -> pd.DataFrame:
    """Rows (month, product_code, bs_side, existing, new, total) for m = 0..12."""
    import b_s_rollforward as rf
    base = get_bank(bank.rollforward.base_bank)
    base_rd = pd.Timestamp(base.report_date)
    book = rf.load_book(base)
    # one FX basis for the whole path: foreign-currency products in the home currency at the
    # base-date rate (the rolled book itself converts at its own report-date rate)
    from labbank_common import fx
    to_home = fx.to_home(base, base_rd)
    book['balance_amt'] = book['balance_amt'] * book['currency'].map(to_home)
    gp = rf.growth_targets(base, book, bank.rollforward.growth_file)
    from labbank_common import catalogue
    loan_models = catalogue.expand_columns(
        pd.read_excel(base.models_dir() / 'loan_beh_models.xlsx'), catalogue.parent_map(bank))
    key = ['product_code', 'bs_side']
    gp = gp.set_index(key)
    scaled = gp['product_name'].isin(rf.SCALED)
    marketable = gp['product_name'].isin(rf.MARKETABLE)   # sold pro rata down to the path, as in the roll-forward
    plug = gp.index[gp['plug']][0]
    rows = []
    for m in range(13):
        d = (base_rd + pd.offsets.MonthEnd(m)) if m else base_rd
        aged = rf.age_book(book, d, bank.generation.avg_rate_map, loan_models) if m else book
        exist = aged.groupby(key)['balance_amt'].sum().reindex(gp.index).fillna(0.0)
        total = gp['balance_base'] * (1.0 + gp['growth_pct'] / 100.0) ** (m / 12.0)
        total = total.where(gp['balance_base'] > 0, gp['target'] * m / 12.0)   # catalogue product: from 0
        total = total.where(scaled | marketable, np.maximum(total, exist))    # survivors above path are kept
        exist = exist.where(~marketable, np.minimum(exist, total))
        total[plug] = 0.0
        a = total[[i for i in total.index if i[1] == 'A']].sum()
        total[plug] = a - total[[i for i in total.index if i[1] != 'A']].sum()
        exist = exist.where(~scaled, total)                                   # accounts kept, scaled
        for k in gp.index:
            rows.append(dict(month=m, date=d, product_code=k[0], bs_side=k[1],
                             existing=float(exist[k]), new=float(total[k] - exist[k]), total=float(total[k])))
    return pd.DataFrame(rows)


def book_check(df: pd.DataFrame, bank) -> float:
    """Largest gap (M) between the month-12 path and the rolled book in SQL."""
    from labbank_common import fx
    sql = pd.read_sql('SELECT CAST(product_code AS INT) product_code, bs_side, currency, SUM(CAST(balance_amt AS FLOAT)) bal '
                      'FROM dbo.transactions GROUP BY product_code, bs_side, currency', get_engine(bank))
    base = get_bank(bank.rollforward.base_bank)            # same FX basis as monthly_balances
    sql['bal'] = sql['bal'] * fx.to_home_series(sql['currency'], base, base.report_date)
    sql = sql.groupby(['product_code', 'bs_side'], as_index=False)['bal'].sum()
    m12 = df[df['month'] == 12].merge(sql, on=['product_code', 'bs_side'], how='outer')
    return float((m12['total'] - m12['bal']).abs().max() / 1e6)


def headline(bank_ids: list[str]) -> pd.DataFrame:
    import rollforward_compare as rc
    first = get_bank(bank_ids[0])
    base = get_bank(first.rollforward.base_bank)
    banks = [base] + [get_bank(b) for b in bank_ids]
    cols = {}
    for b in banks:
        g = rc.book_rates(b)
        h = rc.headline(g)
        ir = rc.irrbb(b).set_index('scenario_id')
        shocks = ir.drop(index=['base', 'own'], errors='ignore')
        liq = rc.liquidity(b).iloc[0]
        # MC500 per-scenario hedge (bs_optimization/per_scenario_hedge.py) is NOT recomputed here:
        # show it only when it is newer than the bank's book (product_params.npz), else 'stale'
        hp = b.output_dir('bs_optimization', create=False) / 'hedge_mc500_current.xlsx'
        book_t = (b.output_dir('optimize_prep', create=False) / 'product_params.npz').stat().st_mtime
        hedge = (pd.read_excel(hp, sheet_name='summary').set_index('measure')['value']
                 if hp.exists() and hp.stat().st_mtime >= book_t else None)
        stale = (f'stale ({pd.Timestamp(hp.stat().st_mtime, unit="s"):%Y-%m-%d}): rerun per_scenario_hedge.py'
                 if hp.exists() else 'not run: per_scenario_hedge.py')
        label = 'Today' if b is base else f'+1Y, {PATH_LABEL[b.rollforward.market_path]}'
        cols[f'{label}<br><span class="sub">{b.report_date}</span>'] = [
            f"{h['total_assets']/1e6:,.0f}",
            f"{h['asset_yield_pct']:.2f}%", f"{h['cost_of_funding_pct']:.2f}%", f"{h['nim_static_pct']:.2f} pp",
            f"{ir['nii_base'].iloc[0]/1e6:,.0f}",
            f"{shocks['sot_nii_pct'].min():+.2f}% ({shocks['sot_nii_pct'].idxmin()})",
            f"{shocks['sot_eve_pct'].min():+.2f}% ({shocks['sot_eve_pct'].idxmin()})",
            f"{liq['lcr']*100:.0f}% / {liq['nsfr']*100:.0f}%",
            f"{int(hedge['Breach before hedge: either'])} / 500" if hedge is not None else stale,
            (hedge['Worst EVE before / after (% T1)'].split(' / ')[0] + '% / '
             + hedge['Worst NII before / after (% T1)'].split(' / ')[0] + '%') if hedge is not None else stale,
        ]
    rows = ['Total assets (M EUR)', 'Asset yield', 'Cost of funding', 'Static margin', 'NII, 1 year (M EUR)',
            'SOT NII, worst shock (% T1)', 'SOT EVE, worst shock (% T1)', 'LCR / NSFR',
            'MC500 curves breaching -12% EVE / -4% NII', 'MC500 worst EVE / worst NII (% T1)']
    return pd.DataFrame(cols, index=rows)


def rate_paths(bank_ids: list[str]) -> pd.DataFrame:
    import rollforward_compare as rc
    base = get_bank(get_bank(bank_ids[0]).rollforward.base_bank)
    h0 = rc.headline(rc.book_rates(base))
    rows = []
    for b in bank_ids:
        bk = get_bank(b)
        h1 = rc.headline(rc.book_rates(bk))
        for m in range(13):
            rows.append(dict(path=bk.rollforward.market_path, month=m,
                             date=pd.Timestamp(base.report_date) + pd.offsets.MonthEnd(m) if m else pd.Timestamp(base.report_date),
                             **{k: h0[k] + (h1[k] - h0[k]) * m / 12 for k in ('asset_yield_pct', 'cost_of_funding_pct', 'nim_static_pct')}))
    return pd.DataFrame(rows)


# ── charts ───────────────────────────────────────────────────────────────────

def _layout(fig: go.Figure, title: str, ytitle: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor='left', font=dict(size=15, color=INK)),
        height=height, margin=dict(l=60, r=150, t=48, b=40), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='IBM Plex Sans, Segoe UI, sans-serif', size=12, color=INK2),
        legend=dict(orientation='v', x=1.01, y=1, font=dict(size=11, color=INK2)),
        hovermode='x unified', hoverlabel=dict(bgcolor='white', font=dict(color=INK)))
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=INK2))
    fig.update_yaxes(title=dict(text=ytitle, font=dict(color=INK2)), gridcolor=GRID, zeroline=False,
                     tickfont=dict(color=INK2), separatethousands=True)
    return fig


def product_lines(df: pd.DataFrame, title: str, keys: list) -> go.Figure:
    fig = go.Figure()
    for i, k in enumerate(keys):
        s = df[(df['product_code'] == k[0]) & (df['bs_side'] == k[1])].sort_values('month')
        if s.empty:                     # product not in this book (e.g. dropped from the mix)
            continue
        chg = (s['total'] / s['total'].iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=s['date'], y=s['total'] / 1e6, name=_label(k), mode='lines+markers',
            line=dict(color=SERIES[i % 8], width=2), marker=dict(size=5),
            customdata=np.c_[chg], hovertemplate='%{y:,.0f}M (%{customdata[0]:+.1f}% vs today)<extra>%{fullData.name}</extra>'))
    return _layout(fig, title, 'Balance, M EUR')


def composition(df: pd.DataFrame, side: str) -> go.Figure:
    q = df[df['month'].isin([0, 3, 6, 9, 12])]
    x = [f"{'Today' if m == 0 else f'Q{m // 3}'}<br>{d:%Y-%m-%d}" for m, d in
         q[['month', 'date']].drop_duplicates().itertuples(index=False)]
    fig = go.Figure()
    for i, (name, keys) in enumerate(CLASSES[side]):
        mask = pd.Series(list(zip(q['product_code'], q['bs_side']))).isin(keys).to_numpy()
        v = q[mask].groupby('month')['total'].sum()
        tot = q[q['bs_side'] == 'A'].groupby('month')['total'].sum() if side == 'A' else \
              q[q['bs_side'] != 'A'].groupby('month')['total'].sum()
        fig.add_trace(go.Bar(x=x, y=v.values / 1e6, name=name, marker=dict(color=SERIES[i], line=dict(color=SURFACE, width=2)),
                             customdata=np.c_[v.values / tot.values * 100],
                             hovertemplate='%{y:,.0f}M (%{customdata[0]:.1f}% of total)<extra>' + name + '</extra>'))
    fig.update_layout(barmode='stack', bargap=0.35)
    return _layout(fig, 'Assets' if side == 'A' else 'Liabilities and equity', 'M EUR', height=420)


def existing_vs_new(df: pd.DataFrame) -> go.Figure:
    loans = [k for k in LINE_GROUPS[0][1] if k != (2200, 'A')] + [(3000, 'A'), (3100, 'A')]
    mask = pd.Series(list(zip(df['product_code'], df['bs_side']))).isin(loans).to_numpy()
    g = df[mask].groupby('date')[['existing', 'new']].sum() / 1e6
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g['existing'], name="Contracts on today's book", stackgroup='one',
                             line=dict(color=SERIES[0], width=2), hovertemplate='%{y:,.0f}M<extra>existing</extra>'))
    fig.add_trace(go.Scatter(x=g.index, y=g['new'], name='New business', stackgroup='one',
                             line=dict(color=SERIES[1], width=2), hovertemplate='%{y:,.0f}M<extra>new</extra>'))
    return _layout(fig, 'Loans and bonds: run-off of today\'s contracts and new business', 'M EUR')


def rate_chart(rp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    names = {'asset_yield_pct': 'Asset yield', 'cost_of_funding_pct': 'Cost of funding'}
    one_path = rp['path'].nunique() == 1               # one rates view: the name is enough
    for i, (k, nm) in enumerate(names.items()):
        for p, dash in (('forwards', 'solid'), ('hybrid', 'solid'), ('unchanged', 'dot')):
            s = rp[rp['path'] == p]
            if s.empty:
                continue
            fig.add_trace(go.Scatter(x=s['date'], y=s[k], mode='lines+markers', marker=dict(size=5),
                                     name=nm if one_path else f'{nm}, {PATH_LABEL[p]}',
                                     line=dict(color=SERIES[i], width=2, dash=dash),
                                     hovertemplate='%{y:.2f}%<extra>%{fullData.name}</extra>'))
    fig = _layout(fig, 'Asset yield and cost of funding over the year', '%')
    fig.update_yaxes(ticksuffix='%', separatethousands=False, rangemode='tozero')
    fig.update_layout(legend=dict(orientation='h', x=0, y=-0.18), margin=dict(r=20))
    return fig


def nmd_flow_chart(led: pd.DataFrame) -> go.Figure:
    """One NMD product's monthly ledger: inflow up, outflow down, closing balance on the right axis."""
    ccy = led['currency'].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=led['date'], y=led['inflow'] / 1e6, name='Inflow (growth plan)', marker_color=SERIES[1],
                         hovertemplate='%{y:,.1f}M<extra>inflow</extra>'))
    fig.add_trace(go.Bar(x=led['date'], y=-led['outflow'] / 1e6, name='Outflow (liquidity model)', marker_color=SERIES[0],
                         customdata=np.c_[led['outflow_rate'] * 100],
                         hovertemplate='%{y:,.1f}M (%{customdata[0]:.2f}% of opening)<extra>outflow</extra>'))
    fig.add_trace(go.Scatter(x=led['date'], y=led['closing'] / 1e6, name='Closing balance', yaxis='y2', mode='lines+markers',
                             line=dict(color=INK, width=2), marker=dict(size=5),
                             hovertemplate='%{y:,.1f}M<extra>closing</extra>'))
    fig = _layout(fig, f"{_label((int(led['product_code'].iloc[0]), led['bs_side'].iloc[0]))}: monthly flows",
                  f'Flow, M {ccy}')
    fig.update_layout(barmode='relative', legend=dict(orientation='h', x=0, y=-0.18), margin=dict(r=60),
                      yaxis2=dict(overlaying='y', side='right', title=dict(text=f'Balance, M {ccy}', font=dict(color=INK2)),
                                  showgrid=False, tickfont=dict(color=INK2), separatethousands=True))
    return fig


def nmd_flow_table(led: pd.DataFrame) -> str:
    rows = ''.join(
        f'<tr><td>{r.date:%Y-%m}</td><td>{r.opening/1e6:,.1f}</td><td>{r.outflow/1e6:,.1f}</td>'
        f'<td>{r.outflow_rate*100:.2f}%</td><td>{r.inflow/1e6:,.1f}</td><td>{r.closing/1e6:,.1f}</td></tr>'
        for r in led.itertuples())
    tot = (f'<tr class="tot"><td>Year</td><td>{led["opening"].iloc[0]/1e6:,.1f}</td><td>{led["outflow"].sum()/1e6:,.1f}</td>'
           f'<td></td><td>{led["inflow"].sum()/1e6:,.1f}</td><td>{led["closing"].iloc[-1]/1e6:,.1f}</td></tr>')
    lbl = _label((int(led['product_code'].iloc[0]), led['bs_side'].iloc[0]))
    return (f'<div class="tbl"><table><tr><th>{html.escape(lbl)} (M {led["currency"].iloc[0]})</th><th>Opening</th>'
            f'<th>Outflow</th><th>Outflow rate</th><th>Inflow</th><th>Closing</th></tr>{rows}{tot}</table></div>')


def nii_chart(dyn: dict) -> go.Figure:
    fig = go.Figure()
    for i, (label, m) in enumerate(dyn.items()):
        fig.add_trace(go.Bar(x=m.index, y=m['nii'] / 1e6, name=label,
                             marker=dict(color=SERIES[i], line=dict(color=SURFACE, width=2)),
                             customdata=np.c_[m['interest_income'] / 1e6, m['interest_expense'] / 1e6, m['swaps'] / 1e6],
                             hovertemplate='NII %{y:,.1f}M<br>income %{customdata[0]:,.1f}M · expense %{customdata[1]:,.1f}M'
                                           ' · swaps %{customdata[2]:+,.1f}M<extra>' + label + '</extra>'))
    fig.update_layout(barmode='group', bargap=0.3)
    fig = _layout(fig, 'NII per month on the moving balance sheet', 'M EUR')
    fig.update_xaxes(tickformat='%b %Y')
    return fig


# ── page ─────────────────────────────────────────────────────────────────────

CSS = """
:root { --bg:#f4f5f3; --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --rule:#e2e1dc; --accent:#2a78d6; }
body { background:var(--bg); color:var(--ink); font:15px/1.55 "IBM Plex Sans","Segoe UI",system-ui,sans-serif; margin:0; }
main { max-width:1180px; margin:0 auto; padding:32px 20px 64px; display:grid; gap:28px; }
h1 { font:600 1.8rem/1.2 "IBM Plex Serif",Georgia,serif; margin:0; }
h2 { font:600 1.2rem/1.3 "IBM Plex Serif",Georgia,serif; margin:0 0 6px; }
p { margin:0; max-width:78ch; } .muted { color:var(--muted); font-size:.9rem; }
.eyebrow { font:500 .75rem "IBM Plex Mono",Consolas,monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
section { display:grid; gap:10px; }
.card { background:var(--surface); border:1px solid var(--rule); padding:8px 10px; min-width:0; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:16px; }
table { border-collapse:collapse; width:100%; font-size:.92rem; background:var(--surface); }
th,td { padding:8px 12px; border-bottom:1px solid var(--rule); text-align:right; white-space:nowrap; }
th:first-child,td:first-child { text-align:left; }
th { font-weight:600; vertical-align:bottom; } .sub { font-weight:400; color:var(--muted); font-size:.8rem; }
td { font-family:"IBM Plex Mono",Consolas,monospace; font-variant-numeric:tabular-nums; }
tr.tot td { font-weight:600; border-top:2px solid var(--rule); }
td:first-child { font-family:inherit; }
.tbl { overflow-x:auto; border:1px solid var(--rule); }
ul { margin:0; padding-left:1.2em; max-width:78ch; }
"""


def swap_book_html(path) -> tuple[str, str]:
    """Summary + new trades + full book table from the roll-forward's swap_changes.xlsx."""
    if not path.exists():
        return '<p class="muted">No swap_changes.xlsx (rerun the roll-forward).</p>', 'the swaps held today stay'
    x = pd.read_excel(path, sheet_name=None)
    summ = ''.join(f'<tr><td>{html.escape(str(r.item))}</td><td>{html.escape("" if pd.isna(r.value) else str(r.value))}</td></tr>'
                   for r in x['summary'].itertuples())
    def rows(df):
        return ''.join(
            f'<tr><td>{html.escape(str(r.status))}</td><td>{r.swap_id}</td>'
            f'<td>{"" if pd.isna(r.like_swap_id) else r.like_swap_id}</td>'
            f'<td>{"pay" if bool(r.pay_fixed) else "receive"} fixed</td><td>{r.notional / 1e6:,.0f}</td>'
            f'<td>{pd.Timestamp(r.start_date):%Y-%m-%d}</td><td>{pd.Timestamp(r.maturity_date):%Y-%m-%d}</td>'
            f'<td>{r.fixed_rate * 100:.2f}%</td><td>{r.float_rate_index}</td></tr>' for r in df.itertuples())
    head = ('<tr><th>Status</th><th>Swap</th><th>Like</th><th>Fixed leg</th><th>Notional (M)</th><th>Start</th>'
            '<th>Maturity</th><th>Fixed rate</th><th>Float index</th></tr>')
    book = x['swap_book']
    matured = book[book['status'].str.startswith('matures')]
    kept_txt = ('the swaps held today stay (none matures within the year)' if matured.empty else
                f"{len(matured)} of the swaps held today mature{'s' if len(matured) == 1 else ''} within the year "
                f"({matured['notional'].sum() / 1e6:,.0f}M), the rest stay")
    out = (f'<div class="tbl"><table><tr><th></th><th></th></tr>{summ}</table></div>'
           f'<h3>New trades</h3><div class="tbl"><table>{head}{rows(x["new_swaps"]) or "<tr><td>none</td></tr>"}</table></div>'
           f'<h3>Whole book</h3><div class="tbl"><table>{head}{rows(book)}</table></div>')
    return out, kept_txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--banks', default='eur_y1')
    a = ap.parse_args()
    bank_ids = a.banks.split(',')
    first = get_bank(bank_ids[0])
    base = get_bank(first.rollforward.base_bank)

    df = monthly_balances(first)
    gaps = {b: book_check(df, get_bank(b)) for b in bank_ids}
    print('month-12 path vs SQL book, max gap (M):', {k: round(v, 3) for k, v in gaps.items()})
    head = headline(bank_ids)
    rates_txt = {
        'forwards': "today's forwards are realised.",
        'unchanged': 'every quote stays at its level of the base date.',
        'hybrid': ('risk-free (ESTR) forwards up to 2Y with the EURIBOR-ESTR spreads of today kept, the curve unchanged '
                   'from 5Y, a linear blend in between (forwards forecast better at the short end, an unchanged '
                   'curve at the long end).'),
    }[first.rollforward.market_path]
    import forecast_swaps as fsw
    _ns = fsw.read(first)
    swaps_txt = ('no new IRS (forecast_1y new_swaps.xlsx missing).' if _ns is None else
                 f'new IRS from forecast_1y new_swaps.xlsx, option {_ns[1]} ({fsw.OPTIONS[_ns[1]]}): '
                 f'{len(_ns[0])} swaps, {_ns[0]["notional"].sum() / 1e6:,.0f}M.')
    rp = rate_paths(bank_ids)
    import dynamic_nii as dn
    import rollforward_compare as rc
    dyn, nii_rows = {}, []
    irrbb_today = rc.irrbb(base)['nii_base'].iloc[0]
    for b in bank_ids:
        bk = get_bank(b)
        label = PATH_LABEL[bk.rollforward.market_path].capitalize()
        m = dn.summary(bk, dn.dynamic_nii(bk, path=df))
        m.index = [pd.Timestamp(base.report_date) + pd.offsets.MonthEnd(i) for i in m.index]
        dyn[label] = m
        nii_rows.append((label, m['interest_income'].sum(), m['interest_expense'].sum(), m['swaps'].sum(),
                         m['nii'].sum(), rc.irrbb(bk)['nii_base'].iloc[0]))

    figs = []
    def add(fig):
        figs.append(fig.to_html(full_html=False, include_plotlyjs=('inline' if not figs else False),
                                config={'displaylogo': False, 'responsive': True}))
        return figs[-1]

    comp_a, comp_l = add(composition(df, 'A')), add(composition(df, 'L'))
    lines = [add(product_lines(df, t, k)) for t, k in LINE_GROUPS]
    evn = add(existing_vs_new(df))
    led_path = first.output_dir('balance_generate', create=False) / 'nmd_ledger.xlsx'
    nmd_html = ''
    if led_path.exists():
        led_all = pd.read_excel(led_path)
        for (pc, sd), led in led_all.groupby(['product_code', 'bs_side'], sort=False):
            nmd_html += (f'<div class="grid2"><div class="card">{add(nmd_flow_chart(led))}</div>'
                         f'<div>{nmd_flow_table(led)}</div></div>')
    rates = add(rate_chart(rp))
    nii_fig = add(nii_chart(dyn))
    nii_html = ''.join(
        f'<tr><td>{html.escape(l)}</td><td>{inc/1e6:,.1f}</td><td>{exp/1e6:,.1f}</td><td>{sw/1e6:+,.1f}</td>'
        f'<td><b>{tot/1e6:,.1f}</b></td><td>{irrbb_today/1e6:,.1f}</td><td>{ir1/1e6:,.1f}</td></tr>'
        for l, inc, exp, sw, tot, ir1 in nii_rows)

    import b_s_rollforward as rf
    swap_html, swaps_kept_txt = swap_book_html(rf.swap_changes_path(first))
    _, gp = rf.read_growth_plan(base.forecast_dir() / first.rollforward.growth_file)
    plug = gp.loc[gp['plug'], 'own_name'].iloc[0]
    m12 = df[df['month'] == 12].set_index(['product_code', 'bs_side'])
    m0 = df[df['month'] == 0].set_index(['product_code', 'bs_side'])
    tbl = pd.DataFrame({'Product': [_label(k) for k in m12.index],
                        'Today (M)': (m0['total'] / 1e6).round(0).to_numpy(),
                        '+1Y (M)': (m12['total'] / 1e6).round(0).to_numpy(),
                        'Change': ((m12['total'] / m0['total'] - 1) * 100).round(1).to_numpy(),
                        'Existing at +1Y (M)': (m12['existing'] / 1e6).round(0).to_numpy(),
                        'New business (M)': (m12['new'] / 1e6).round(0).to_numpy()})
    tbl_html = ''.join(
        f"<tr><td>{html.escape(r.Product)}</td><td>{r[2]:,.0f}</td><td>{r[3]:,.0f}</td><td>{r[4]:+.1f}%</td>"
        f"<td>{r[5]:,.0f}</td><td>{r[6]:,.0f}</td></tr>" for r in tbl.itertuples())
    head_html = ('<tr><th></th>' + ''.join(f'<th>{c}</th>' for c in head.columns) + '</tr>' +
                 ''.join(f'<tr><td>{i}</td>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>'
                         for i, row in head.iterrows()))

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>EUR One-Year Roll-Forward</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style></head><body><main>
<header style="display:grid;gap:8px">
  <span class="eyebrow">{html.escape(base.name)} · {base.report_date} → {first.report_date}</span>
  <h1>The EUR book one year forward</h1>
  <p>Today's contracts aged contract by contract (amortisation, prepayment model, maturities) plus one year of new business
  from the growth plan. Rates after {base.report_date}: {html.escape(rates_txt)}</p>
</header>

<section><h2>Results</h2><div class="tbl"><table>{head_html}</table></div>
<p class="muted">SOT = EBA supervisory outlier test, worst of the 6 shocks. MC500 = 500 Monte Carlo curves × 6 shocks against the internal limits
(EVE −12%, NII −4% of Tier 1) before any new hedge. Yield and cost of funding are balance-weighted client rates; equity funds at 0%.</p></section>

<section><h2>NII over the next 12 months</h2>
<p>Interest actually earned and paid during {pd.Timestamp(base.report_date) + pd.Timedelta(days=1):%d.%m.%Y} – {pd.Timestamp(first.report_date):%d.%m.%Y}
on the balance sheet as it moves: existing contracts amortise, mature and reprice with their index, new business comes in
at the month's market rate + its margin. The IRRBB NII next to it keeps the balance sheet constant and is the same
measure the supervisory test uses.</p>
<div class="tbl"><table><tr><th>Rates after {base.report_date}</th><th>Interest income (M)</th><th>Interest expense (M)</th>
<th>Swaps (M)</th><th>NII over the year (M)</th><th>IRRBB NII, constant balance sheet, today (M)</th>
<th>IRRBB NII, constant balance sheet, +1Y book (M)</th></tr>{nii_html}</table></div>
<div class="card">{nii_fig}</div>
<p class="muted">Rates: fixed contracts keep their rate; floating and administered rates move with their index × the rate model's beta,
within its floors and caps. New business margin = the average margin of the product's new contracts in the rolled book.
Interest = average monthly balance × rate × days/365; equity and cash earn 0.</p></section>

<section><h2>Composition of the balance sheet, quarter by quarter</h2>
<div class="grid2"><div class="card">{comp_a}</div><div class="card">{comp_l}</div></div></section>

<section><h2>Each product month by month</h2>
<p class="muted">Hover for the balance and its change versus today. Balances grow monthly with the growth plan;
the plan fixes the composition in one year (each side 100% of total assets); between month 0 and 12
{html.escape(plug)} balances assets against liabilities and equity.</p>
<div class="grid2">{''.join(f'<div class="card">{x}</div>' for x in lines)}</div></section>

<section><h2>Current and savings accounts: monthly flows</h2>
<p>Each month: outflow = the liquidity behavioural model's monthly decay × the opening balance (the model's day-1 stress drop
is left out: it is a liquidity-stress assumption, not client behaviour; new money runs off too); inflow = what keeps the balance
on the growth plan; closing = opening − outflow + inflow. At +1Y today's accounts keep their run-off survival (that share of
accounts closes, the rest shrink); the rest of the balance is new accounts at the standard rate.</p>
{nmd_html or '<p class="muted">No NMD ledger (balance_generate output nmd_ledger.xlsx missing).</p>'}</section>

<section><h2>Run-off and new business</h2>
<div class="grid2"><div class="card">{evn}</div><div class="card">{rates}</div></div>
<p class="muted">Yield and cost of funding: start and end of year are the actual books; months in between are interpolated,
an indication of the path, not a monthly P&amp;L.</p></section>

<section><h2>Products at +1 year</h2><div class="tbl"><table>
<tr><th>Product</th><th>Today (M)</th><th>+1Y (M)</th><th>Change</th><th>Existing at +1Y (M)</th><th>New business (M)</th></tr>
{tbl_html}</table></div></section>

<section><h2>Swap book</h2>
<p>The swaps held today, the ones maturing inside the year and the new trades as the roll-forward booked them
(notional of the chosen new_swaps.xlsx option, fixed rate = the IRS quote of the swap's tenor on its start date).
Also in runs/{html.escape(first.bank_id)}/forecast_1y/swap_changes.xlsx.</p>
{swap_html}</section>

<section><h2>Assumptions</h2><ul class="muted">
<li>Loans amortise on their own schedule and prepay with the bank's prepayment model (by loan age); matured term deposits,
bonds, T-bills and covered bonds drop out.</li>
<li>Current and savings accounts run off with their liquidity behavioural model's 12-month point (that share of accounts closes,
the rest shrink); the volume up to the growth plan is opened as new accounts at the standard rate. An account product without a
behavioural model, cards, cash and equity keep their accounts and scale to the growth plan.</li>
<li>New business starts inside the year and is priced on the fixings of its start date.</li>
<li>Swaps: {html.escape(swaps_kept_txt)}; {html.escape(swaps_txt)}</li>
</ul></section>
</main>
<script>/* charts drawn before the grid settled keep their first width: resize them to their cards */
window.addEventListener('load', () => document.querySelectorAll('.plotly-graph-div').forEach(d => Plotly.Plots.resize(d)));
</script></body></html>"""
    out = first.output_dir('visual_rep') / 'rollforward_report.html'
    out.write_text(page, encoding='utf-8')
    print(f'Saved -> {out}  ({len(page)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
