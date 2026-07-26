#!/usr/bin/env python3
"""
Generate all PNG assets for the LabBank Beamer presentation.
Run from the visual_rep/ directory:  python make_beamer_assets.py
Outputs go to visual_rep/beamer_assets/
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beamer_assets')
os.makedirs(OUT, exist_ok=True)

REPORT_DATE = pd.to_datetime('2024-12-31')
CCY = 'PLN'
RD  = REPORT_DATE

ENGINE = create_engine(
    'mssql+pyodbc://maciek_d/bank_gen'
    '?driver=ODBC+Driver+17+for+SQL+Server'
    '&Trusted_Connection=yes',
    future=True,
)

# ── Colours (match existing reports) ─────────────────────────────────────────
C_A   = '#1565C0'
C_L   = '#C62828'
C_N   = '#2E7D32'
C_EQ  = '#6A1B9A'
C_UP  = '#D32F2F'
C_DN  = '#1565C0'
C_REN = '#0288D1'
C_FIX = '#1B5E20'
C_VAR = '#F57F17'
C_ADM = '#78909C'
C_OK  = '#2E7D32'

PRODUCT_COLORS = {
    'Mortgage Fixed':      '#1565C0',
    'Mortgage Float':      '#42A5F5',
    'Consumer Loan Fixed': '#2E7D32',
    'Consumer Loan Float': '#66BB6A',
    'Bond / Securities':   '#F57F17',
    'Interbank / Other':   '#78909C',
    'Term Deposit':        '#B71C1C',
    'Savings Account':     '#EF5350',
    'Current Account':     '#FF8A65',
    'Issued Bond':         '#8D6E63',
}

PROD_LABELS = {
    '0000': 'Interbank / Other',
    '1000': 'Mortgage Fixed',       '1100': 'Mortgage Float',
    '2000': 'Consumer Loan Fixed',  '2100': 'Consumer Loan Float',
    '3000': 'Bond / Securities',    '3100': 'Bond / Securities',
    '5000': 'Issued Bond',          '6000': 'Current Account',
    '7060': 'Term Deposit',         '8000': 'Savings Account',
}

SCEN_LABELS = {
    'base':   'Base',
    'par_up': '+250 bps',
    'par_dn': '−250 bps',
    'steep':  'Steepener',
    'flat':   'Flattener',
    'sr_up':  'Short +350 bps',
    'sr_dn':  'Short −350 bps',
}

SCEN_STYLE = {
    'base':   ('#212121', '-',  2.2, 'Base'),
    'par_up': ('#C62828', '-',  1.6, '+250 bps (parallel)'),
    'par_dn': ('#1565C0', '-',  1.6, '−250 bps (parallel)'),
    'steep':  ('#E65100', '--', 1.4, 'Steepener'),
    'flat':   ('#0277BD', '--', 1.4, 'Flattener'),
    'sr_up':  ('#AD1457', ':',  1.5, 'Short rate +350 bps'),
    'sr_dn':  ('#0277BD', ':',  1.5, 'Short rate −350 bps'),
}

plt.rcParams.update({
    'figure.dpi': 150, 'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': 0.35, 'grid.linestyle': '--',
    'font.size': 9, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.titlepad': 8, 'legend.fontsize': 8, 'legend.framealpha': 0.8,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

def rsql(q, **p):
    return pd.read_sql_query(text(q), ENGINE, params=p)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved → {name}')

def tkey(lbl):
    if lbl == '1D':   return 0
    if lbl == '>30Y': return 999_999
    if lbl.endswith('Y'): return int(lbl[:-1]) * 12
    if lbl.endswith('M'): return int(lbl[:-1])
    return 99_999


# ── 1. PIPELINE DIAGRAM ───────────────────────────────────────────────────────
def make_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    boxes = [
        (0.4, 2.0, 2.2, 1.5, '#E3F2FD', '#1565C0',
         'Input Data\n(Excel)', 'bank_data.xlsx\nirs_input.xlsx'),
        (3.2, 2.0, 2.4, 1.5, '#E8F5E9', '#2E7D32',
         'Balance Sheet\nGeneration', 'dbo.transactions\nschemat.loans/deposits'),
        (6.2, 2.75, 2.3, 1.5, '#FFF3E0', '#E65100',
         'Cash Flow\nEngine', 'cf.products\ncf.nii_base_scenario'),
        (6.2, 0.75, 2.3, 1.5, '#F3E5F5', '#6A1B9A',
         'IR Derivatives\n(IRS Book)', 'schemat.ir_swaps\nrepricing gap adj.'),
        (9.4, 2.75, 2.3, 1.5, '#FFEBEE', '#C62828',
         'IRRBB\nCalculation', 'NII · EVE · EBA SOT\nshock curves'),
        (9.4, 0.75, 2.3, 1.5, '#E0F7FA', '#00695C',
         'Liquidity\nRisk', 'LCR · NSFR\n30d & 1yr horizon'),
        (12.0, 1.75, 1.8, 1.5, '#ECEFF1', '#37474F',
         'Reporting\n& Viz', 'HTML · PDF\nPower BI'),
    ]

    for (x, y, w, h, fc, ec, title, sub) in boxes:
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle='round,pad=0.08',
                              facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title,
                ha='center', va='center', fontsize=9, fontweight='bold', color=ec)
        ax.text(x + w/2, y + h*0.22, sub,
                ha='center', va='center', fontsize=7, color='#555555')

    arrows = [
        (2.6, 2.75, 3.2, 2.75),
        (5.6, 2.75, 6.2, 3.5),
        (5.6, 2.75, 6.2, 1.5),
        (8.5, 3.5,  9.4, 3.5),
        (8.5, 1.5,  9.4, 1.5),
        (11.7, 3.5, 12.0, 2.75),
        (11.7, 1.5, 12.0, 2.25),
    ]
    for (x1, y1, x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#455A64',
                                   lw=1.6, connectionstyle='arc3,rad=0.0'))

    ax.set_title('LabBank — System Architecture', fontsize=13, fontweight='bold',
                 pad=6, color='#1A237E')
    fig.tight_layout()
    save(fig, 'pipeline.png')


# ── 2. SQL SCHEMA DIAGRAM ─────────────────────────────────────────────────────
def make_sql_schema():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    schemas = [
        (0.3, 4.5, 2.8, 1.8, '#E3F2FD', '#1565C0', 'dbo',
         'transactions\nequity', 'Master balance sheet'),
        (0.3, 2.0, 2.8, 2.0, '#E8F5E9', '#2E7D32', 'schemat',
         'loans · deposits\nequity · ir_swaps\nsched.ir_swaps', 'Product schedules\n& swap book'),
        (0.3, 0.2, 2.8, 1.5, '#FFF8E1', '#F57F17', 'mkt',
         'curves', 'Discount / fwd curves\nPLN · EUR · USD'),
        (4.8, 4.0, 2.8, 2.2, '#FFF3E0', '#E65100', 'cf',
         'products\nnii_base_scenario\nnii_own_scenarios\neve_own_scenarios', 'Cash flow outputs'),
        (4.8, 1.0, 2.8, 2.5, '#FFEBEE', '#C62828', 'irrbb',
         'curves · ir_gap_beh\nnii_results · eve_results\nirrbb_report\nliq_gap_beh', 'IRRBB risk metrics'),
        (9.4, 3.5, 2.8, 1.8, '#E0F7FA', '#00695C', 'results',
         'lcr_nsfr', 'Liquidity ratios\nLCR · NSFR'),
    ]

    for (x, y, w, h, fc, ec, schema, tables, desc) in schemas:
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle='round,pad=0.1',
                              facecolor=fc, edgecolor=ec, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.25, f'[{schema}]',
                ha='center', va='center', fontsize=9, fontweight='bold', color=ec)
        ax.text(x + w/2, y + h/2 - 0.1, tables,
                ha='center', va='center', fontsize=7.5, color='#333333', linespacing=1.5)
        ax.text(x + w/2, y + 0.2, desc,
                ha='center', va='center', fontsize=7, color='#888888', style='italic')

    links = [
        (3.1, 5.4, 4.8, 5.4, 'reads →'),
        (3.1, 3.0, 4.8, 3.0, 'reads →'),
        (1.7, 4.5, 1.7, 4.0, ''),
        (3.1, 1.0, 4.8, 2.25, 'reads →'),
        (7.6, 5.1, 9.4, 4.9, 'feeds →'),
        (7.6, 2.25, 9.4, 4.4, 'feeds →'),
    ]
    for (x1, y1, x2, y2, lbl) in links:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#607D8B', lw=1.4))
        if lbl:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.12, lbl, ha='center', fontsize=7, color='#607D8B')

    ax.set_title('LabBank — SQL Schema Overview', fontsize=13, fontweight='bold',
                 pad=8, color='#1A237E')
    fig.tight_layout()
    save(fig, 'sql_schema.png')


# ── 3. BALANCE SHEET STRUCTURE ────────────────────────────────────────────────
def make_bs_structure():
    bs = rsql('''
        SELECT product_code, bs_side, SUM(balance_amt) AS bal
        FROM dbo.transactions
        WHERE report_date=:rd AND currency=:ccy
        GROUP BY product_code, bs_side
    ''', rd=RD, ccy=CCY)
    eq = float(rsql(
        'SELECT ISNULL(SUM(balance_amt),0) AS v FROM schemat.equity WHERE report_date=:rd',
        rd=RD)['v'].iloc[0])

    bs['label'] = bs['product_code'].map(PROD_LABELS).fillna(bs['product_code'])
    assets = bs[bs['bs_side'] == 'A'].sort_values('bal', ascending=False)
    liabs  = bs[bs['bs_side'] == 'L'].sort_values('bal', ascending=False)
    tot_a  = assets['bal'].sum()
    tot_l  = liabs['bal'].sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                                    gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle(f'Balance Sheet Structure  |  {CCY}  |  {RD.date()}',
                 fontsize=13, fontweight='bold')

    def stacked_h(ax, df, total, title, default_color, is_asset):
        cum = 0.0
        colors = [PRODUCT_COLORS.get(r['label'], default_color)
                  for _, r in df.iterrows()]
        for (_, row), col in zip(df.iterrows(), colors):
            v = abs(row['bal']) / 1e9
            ax.barh(0, v, left=cum/1e9, color=col, alpha=0.85,
                    edgecolor='white', height=0.55)
            if v > 0.3:
                ax.text(cum/1e9 + v/2, 0, f'{row["label"]}\n{v:.1f}B',
                        ha='center', va='center', fontsize=7.5,
                        fontweight='bold', color='white')
            cum += abs(row['bal'])
        if not is_asset:
            ax.barh(0, eq/1e9, left=cum/1e9, color=C_EQ, alpha=0.85,
                    edgecolor='white', height=0.55)
            ax.text(cum/1e9 + eq/1e9/2, 0, f'Equity\n{eq/1e9:.1f}B',
                    ha='center', va='center', fontsize=7.5,
                    fontweight='bold', color='white')
            cum += eq
        ax.set_xlim(0, max(tot_a, tot_l + eq)/1e9 * 1.05)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
        ax.set_xlabel('B PLN')
        ax.set_title(f'{title}\n(total {total/1e9:.1f} B PLN)')

    stacked_h(ax1, assets, tot_a, 'Assets', C_A, True)
    stacked_h(ax2, liabs,  tot_l + eq, 'Liabilities + Equity', C_L, False)
    fig.tight_layout()
    save(fig, 'bs_structure.png')


# ── 4. RATE TYPE EXPOSURE ─────────────────────────────────────────────────────
def make_rate_type():
    nii_d = rsql('''
        SELECT product_code, bs_side, rate_type,
               SUM(nii_interest) AS nii_int,
               SUM(beh_outstanding) AS bal
        FROM cf.nii_base_scenario
        WHERE report_date=:rd AND currency=:ccy AND scenario_id='base'
        GROUP BY product_code, bs_side, rate_type
    ''', rd=RD, ccy=CCY)

    RT_COLORS = {'F': C_FIX, 'V': C_VAR, 'A': C_ADM}
    RT_LABELS = {'F': 'Fixed', 'V': 'Variable', 'A': 'Admin rate'}

    fig, (ax_a, ax_l) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f'Rate Type Exposure  |  {CCY}  |  {RD.date()}',
                 fontsize=13, fontweight='bold')

    for ax, side, title in [(ax_a, 'A', 'Assets'),
                             (ax_l, 'L', 'Liabilities')]:
        grp = (nii_d[nii_d['bs_side'] == side]
               .groupby('rate_type')
               .agg(bal=('bal', 'sum'))
               .reset_index())
        tot = grp['bal'].abs().sum()
        grp = grp.sort_values('bal', ascending=False)
        x = np.arange(len(grp))
        cols = [RT_COLORS.get(rt, '#999') for rt in grp['rate_type']]
        bars = ax.bar(x, grp['bal'].abs()/1e9, color=cols, alpha=0.85, edgecolor='white', width=0.5)
        for bar, (_, row) in zip(bars, grp.iterrows()):
            pct = abs(row['bal']) / tot * 100 if tot else 0
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f'{pct:.0f}%', ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([RT_LABELS.get(rt, rt) for rt in grp['rate_type']], fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
        ax.set_ylabel('B PLN')
        ax.set_title(f'{title}  (total {tot/1e9:.1f} B PLN)')
        ax.legend(handles=[mpatches.Patch(color=RT_COLORS[k], label=v)
                            for k, v in RT_LABELS.items() if k in grp['rate_type'].values],
                  fontsize=8)

    fig.tight_layout()
    save(fig, 'rate_type.png')


# ── 5. NII WATERFALL ──────────────────────────────────────────────────────────
def make_nii_waterfall():
    nii = rsql('''
        SELECT product_code, bs_side,
               SUM(nii_interest) AS nii_int
        FROM cf.nii_base_scenario
        WHERE report_date=:rd AND currency=:ccy AND scenario_id='base'
        GROUP BY product_code, bs_side
    ''', rd=RD, ccy=CCY)
    nii['label'] = nii['product_code'].map(PROD_LABELS).fillna(nii['product_code'])
    assets_p = nii[nii['bs_side'] == 'A'].sort_values('nii_int', ascending=False)
    liabs_p  = nii[nii['bs_side'] == 'L'].sort_values('nii_int')
    nii_net  = nii['nii_int'].sum()

    wf_labels, wf_vals, wf_bots, wf_cols = [], [], [], []
    running = 0.0
    for _, row in assets_p.iterrows():
        wf_labels.append(row['label'])
        wf_vals.append(row['nii_int'] / 1e6)
        wf_bots.append(running / 1e6)
        wf_cols.append(PRODUCT_COLORS.get(row['label'], C_N))
        running += row['nii_int']
    for _, row in liabs_p.iterrows():
        wf_labels.append(row['label'])
        wf_vals.append(row['nii_int'] / 1e6)
        wf_bots.append(running / 1e6)
        wf_cols.append(PRODUCT_COLORS.get(row['label'], C_L))
        running += row['nii_int']
    wf_labels.append('Net Interest\nIncome')
    wf_vals.append(nii_net / 1e6)
    wf_bots.append(0.0)
    wf_cols.append(C_A)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f'NII Waterfall — Income → Expense → Net  |  {CCY}  |  {RD.date()}',
                 fontsize=12, fontweight='bold')
    x = np.arange(len(wf_labels))
    bars = ax.bar(x, wf_vals, bottom=wf_bots, color=wf_cols, alpha=0.85,
                  edgecolor='white', linewidth=0.8, width=0.65)
    for bar, val in zip(bars, wf_vals):
        ypos = bar.get_y() + bar.get_height() + (2 if val >= 0 else -5)
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.0f}M',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=7.5, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(wf_labels, rotation=30, ha='right', fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}M'))
    ax.set_ylabel('M PLN')
    fig.tight_layout()
    save(fig, 'nii_waterfall.png')


# ── 6. NII BRIDGE — LOCKED vs RENEWAL ────────────────────────────────────────
def make_nii_bridge():
    br = rsql('''
        SELECT bs_side,
               SUM(nii_interest) AS nii_int,
               SUM(nii_renewal)  AS nii_ren,
               SUM(nii_total)    AS nii_tot
        FROM cf.nii_base_scenario
        WHERE report_date=:rd AND currency=:ccy AND scenario_id='base'
        GROUP BY bs_side
    ''', rd=RD, ccy=CCY)

    labels = ['Asset\nInterest', 'Asset\nRenewal', 'Liability\nInterest', 'Liability\nRenewal']
    vals   = []
    colors = [C_N, C_REN, C_L, '#EF9A9A']
    for side, col in [('A', 'nii_int'), ('A', 'nii_ren'),
                      ('L', 'nii_int'), ('L', 'nii_ren')]:
        row = br[br['bs_side'] == side]
        vals.append(float(row[col].sum()) / 1e6 if not row.empty else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle(f'NII Bridge — Locked Interest vs Renewal  |  {CCY}  |  {RD.date()}',
                 fontsize=12, fontweight='bold')

    ax2.bar(range(4), vals, color=colors, alpha=0.85, edgecolor='white', width=0.55)
    for i, v in enumerate(vals):
        ax2.text(i, v/2 + (0 if v > 0 else v), f'{v:+.0f}M',
                 ha='center', va='center', fontsize=9,
                 fontweight='bold', color='white')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}M'))
    ax2.set_title('Aggregate Components')
    ax2.set_ylabel('M PLN')

    tot_int = vals[0] + vals[2]
    tot_ren = vals[1] + vals[3]
    net_nii = tot_int + tot_ren
    bridge_lbls = ['Locked\nInterest', 'Renewal\n(repricing)', 'Net\nNII']
    bridge_vals = [tot_int, tot_ren, net_nii]
    bridge_bots = [0, tot_int, 0]
    bridge_cols = [C_N, C_REN, C_A]
    ax1.bar(range(3), bridge_vals, bottom=bridge_bots, color=bridge_cols,
            alpha=0.85, edgecolor='white', width=0.5)
    for i, (v, b) in enumerate(zip(bridge_vals, bridge_bots)):
        ax1.text(i, b + v/2, f'{v:+.0f}M', ha='center', va='center',
                 fontsize=10, fontweight='bold', color='white')
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(bridge_lbls, fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}M'))
    ax1.set_title('NII Decomposition')
    ax1.set_ylabel('M PLN')
    ax1.text(2.55, net_nii * 0.6,
             f'Locked\nimmune to\nrate shocks\n\nRenewal\nreprices\nwith market',
             fontsize=8, color='#555', bbox=dict(boxstyle='round,pad=0.4',
             fc='lightyellow', ec='#ccc', alpha=0.9))
    fig.tight_layout()
    save(fig, 'nii_bridge.png')


# ── 7. REPRICING GAP ──────────────────────────────────────────────────────────
def make_repricing_gap():
    try:
        gap = rsql('''
            SELECT tenor_bucket, gap_cf, ISNULL(gap_cf_ca, 0) AS gap_irs
            FROM irrbb.ir_gap_beh WHERE currency=:ccy
        ''', ccy=CCY)
    except Exception:
        gap = rsql('''
            SELECT tenor_bucket, gap_cf, 0.0 AS gap_irs
            FROM irrbb.ir_gap_beh WHERE currency=:ccy
        ''', ccy=CCY)

    gap['_s'] = gap['tenor_bucket'].map(tkey)
    gap = gap.sort_values('_s').reset_index(drop=True)
    gap_24 = gap[gap['_s'] <= 24]
    x = np.arange(len(gap_24))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'Repricing Gap Ladder  |  {CCY}  |  {RD.date()}',
                 fontsize=12, fontweight='bold')

    bar_col = [C_A if v >= 0 else C_L for v in gap_24['gap_cf']]
    ax1.bar(x, gap_24['gap_cf'] / 1e9, color=bar_col, alpha=0.80, edgecolor='white')
    if gap_24['gap_irs'].abs().sum() > 0:
        ax1.bar(x, gap_24['gap_irs'] / 1e9, color='#F57F17', alpha=0.65,
                edgecolor='white', label='IRS adjustment')
        ax1.legend()
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(gap_24['tenor_bucket'], rotation=45, ha='right', fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
    ax1.set_ylabel('B PLN')
    ax1.set_title('Net Repricing Gap by Tenor Bucket  (positive = asset-sensitive)')
    ax1.add_patch(mpatches.Patch(color=C_A, alpha=0.7, label='Asset-sensitive (+)'))
    ax1.add_patch(mpatches.Patch(color=C_L, alpha=0.7, label='Liability-sensitive (−)'))
    ax1.legend(fontsize=8)

    cum = gap_24['gap_cf'].cumsum() / 1e9
    ax2.plot(x, cum, color=C_N, lw=2.0, marker='o', markersize=4)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.fill_between(x, cum, 0, where=(cum >= 0), alpha=0.15, color=C_A)
    ax2.fill_between(x, cum, 0, where=(cum < 0), alpha=0.15, color=C_L)
    ax2.set_xticks(x)
    ax2.set_xticklabels(gap_24['tenor_bucket'], rotation=45, ha='right', fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
    ax2.set_ylabel('B PLN')
    ax2.set_title('Cumulative Gap')
    fig.tight_layout()
    save(fig, 'repricing_gap.png')


# ── 8. EBA SHOCK CURVES ───────────────────────────────────────────────────────
def make_shock_curves():
    try:
        crv = rsql('''
            SELECT scenario_id, n_days, zero_rt
            FROM irrbb.curves
            WHERE report_date=:rd
              AND curve_name='PLN_disc_curve'
              AND scenario_id IN ('base','par_up','par_dn','steep','flat','sr_up','sr_dn')
            ORDER BY scenario_id, n_days
        ''', rd=RD)
    except Exception:
        print('  [shock_curves] irrbb.curves not available — skipping')
        return

    if crv.empty:
        print('  [shock_curves] no data returned — skipping')
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(f'EBA IRRBB Shock Scenarios — PLN Yield Curves  |  {RD.date()}',
                 fontsize=12, fontweight='bold')

    for sid, grp in crv.groupby('scenario_id'):
        grp = grp.sort_values('n_days')
        col, ls, lw, lbl = SCEN_STYLE.get(sid, ('#888', '-', 1.2, sid))
        ax.plot(grp['n_days'] / 365.25, grp['zero_rt'] * 100,
                color=col, ls=ls, lw=lw, label=lbl)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}Y'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
    ax.set_xlabel('Tenor (years)')
    ax.set_ylabel('Zero rate (%)')
    ax.set_title('Parallel ±250 bps  |  Steepener / Flattener  |  Short rate ±350 bps')
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    save(fig, 'shock_curves.png')


# ── 9. NII BY SCENARIO ────────────────────────────────────────────────────────
def make_nii_scenarios():
    sc = rsql('''
        SELECT scenario_id,
               SUM(nii_interest) AS nii_int,
               SUM(nii_renewal)  AS nii_ren,
               SUM(nii_total)    AS nii_tot
        FROM irrbb.nii_results
        WHERE report_date=:rd AND currency=:ccy
        GROUP BY scenario_id
    ''', rd=RD, ccy=CCY)

    if sc.empty:
        print('  [nii_scenarios] no data — skipping')
        return

    order = ['base', 'par_up', 'par_dn', 'steep', 'flat', 'sr_up', 'sr_dn']
    sc['_ord'] = sc['scenario_id'].map(lambda s: order.index(s) if s in order else 99)
    sc = sc.sort_values('_ord').reset_index(drop=True)
    sc['label'] = sc['scenario_id'].map(SCEN_LABELS).fillna(sc['scenario_id'])

    base_nii = float(sc[sc['scenario_id'] == 'base']['nii_tot'].sum())
    x = np.arange(len(sc))

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f'NII by Scenario (locked + renewal)  |  {CCY}  |  {RD.date()}',
                 fontsize=12, fontweight='bold')

    bar_cols = [C_A if v >= base_nii else C_L for v in sc['nii_tot']]
    bars = ax.bar(x, sc['nii_int'] / 1e6, color=bar_cols, alpha=0.85,
                  edgecolor='white', label='Locked interest')
    ax.bar(x, sc['nii_ren'] / 1e6, bottom=sc['nii_int'] / 1e6,
           color=bar_cols, alpha=0.38, hatch='///', edgecolor='grey',
           linewidth=0.5, label='Renewal')
    for i, (_, row) in enumerate(sc.iterrows()):
        v = row['nii_tot'] / 1e6
        d = v - base_nii / 1e6
        txt = f'{v:.0f}M' if row['scenario_id'] == 'base' else f'{d:+.0f}M'
        ax.text(i, v + 2, txt, ha='center', fontsize=8, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sc['label'], fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}M'))
    ax.set_ylabel('M PLN')
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'nii_scenarios.png')


# ── 10. EVE BY SCENARIO ───────────────────────────────────────────────────────
def make_eve_results():
    ev = rsql('''
        SELECT scenario_id,
               SUM(pv_total) AS eve
        FROM irrbb.eve_results
        WHERE report_date=:rd AND currency=:ccy
        GROUP BY scenario_id
    ''', rd=RD, ccy=CCY)

    if ev.empty:
        print('  [eve_results] no data — skipping')
        return

    order = ['base', 'par_up', 'par_dn', 'steep', 'flat', 'sr_up', 'sr_dn']
    ev['_ord'] = ev['scenario_id'].map(lambda s: order.index(s) if s in order else 99)
    ev = ev.sort_values('_ord').reset_index(drop=True)
    ev['label'] = ev['scenario_id'].map(SCEN_LABELS).fillna(ev['scenario_id'])
    base_eve = float(ev[ev['scenario_id'] == 'base']['eve'].sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={'width_ratios': [1.2, 1]})
    fig.suptitle(f'Economic Value of Equity (EVE)  |  {CCY}  |  {RD.date()}',
                 fontsize=12, fontweight='bold')

    x = np.arange(len(ev))
    cols = [C_A if v >= base_eve else C_L for v in ev['eve']]
    ax1.bar(x, ev['eve'] / 1e9, color=cols, alpha=0.85, edgecolor='white')
    for i, (_, row) in enumerate(ev.iterrows()):
        ax1.text(i, row['eve'] / 1e9 + 0.03, f'{row["eve"]/1e9:.2f}B',
                 ha='center', fontsize=8, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ev['label'], rotation=25, ha='right', fontsize=8.5)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
    ax1.set_ylabel('B PLN')
    ax1.set_title('EVE level by scenario (PV of all run-off CFs)')

    ev_non_base = ev[ev['scenario_id'] != 'base'].copy()
    ev_non_base['delta'] = ev_non_base['eve'] - base_eve
    delta_cols = [C_A if v >= 0 else C_L for v in ev_non_base['delta']]
    x2 = np.arange(len(ev_non_base))
    ax2.barh(x2, ev_non_base['delta'] / 1e6, color=delta_cols, alpha=0.85, edgecolor='white')
    for i, (_, row) in enumerate(ev_non_base.iterrows()):
        v = row['delta'] / 1e6
        ax2.text(v + (2 if v >= 0 else -2), i, f'{v:+.0f}M',
                 va='center', ha='left' if v >= 0 else 'right', fontsize=8)
    ax2.set_yticks(x2)
    ax2.set_yticklabels(ev_non_base['label'], fontsize=8.5)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}M'))
    ax2.set_xlabel('ΔEVE (M PLN vs base)')
    ax2.set_title('ΔEVE vs base scenario')
    fig.tight_layout()
    save(fig, 'eve_results.png')


# ── 11. EBA SOT ───────────────────────────────────────────────────────────────
def make_eba_sot():
    try:
        sot = rsql('''
            SELECT scenario_id, currency,
                   delta_nii, sot_nii_pct, delta_nii_reg, sot_nii_pct_reg,
                   delta_eve, sot_eve_pct, delta_eve_reg, sot_eve_pct_reg
            FROM irrbb.irrbb_report WHERE report_date=:rd
        ''', rd=RD)
    except Exception:
        print('  [eba_sot] irrbb.irrbb_report not available — skipping')
        return

    if sot.empty:
        print('  [eba_sot] no data — skipping')
        return

    sot = sot[sot['currency'] == CCY].copy()
    order = ['par_up', 'par_dn', 'steep', 'flat', 'sr_up', 'sr_dn']
    sot['_ord'] = sot['scenario_id'].map(lambda s: order.index(s) if s in order else 99)
    sot = sot.sort_values('_ord').reset_index(drop=True)
    sot['label'] = sot['scenario_id'].map(SCEN_LABELS).fillna(sot['scenario_id'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'EBA Supervisory Outlier Test (SOT)  |  {CCY}  |  {RD.date()}',
                 fontsize=12, fontweight='bold')

    y = np.arange(len(sot))

    for ax, col_pct, threshold, title, lbl in [
        (ax1, 'sot_eve_pct_reg', -15, 'ΔEVE / Tier 1 Capital (%)', 'EVE SOT'),
        (ax2, 'sot_nii_pct_reg', -5,  'ΔNII / Tier 1 Capital (%)', 'NII SOT'),
    ]:
        vals = sot[col_pct].tolist()
        bar_cols = [C_L if v < threshold else C_A for v in vals]
        ax.barh(y, vals, color=bar_cols, alpha=0.82, edgecolor='white')
        for i, v in enumerate(vals):
            ax.text(v + (0.3 if v >= 0 else -0.3), i, f'{v:+.1f}%',
                    va='center', ha='left' if v >= 0 else 'right', fontsize=8.5)
        ax.axvline(threshold, color=C_L, lw=2.0, ls='--',
                   label=f'Threshold {threshold}%')
        ax.axvline(0, color='black', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sot['label'], fontsize=9)
        ax.set_xlabel(lbl)
        ax.set_title(f'{lbl}\n(50% haircut applied to positive Δ)')
        ax.legend(fontsize=8)
        breach = sot[sot[col_pct] < threshold]
        if not breach.empty:
            ax.text(0.02, 0.02, '⚠ BREACH detected',
                    transform=ax.transAxes, fontsize=9, color=C_L,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec=C_L))

    fig.tight_layout()
    save(fig, 'eba_sot.png')


# ── 12. LCR / NSFR ───────────────────────────────────────────────────────────
def make_lcr_nsfr():
    try:
        lq = rsql('''
            SELECT currency, hqla, outflows_30d, inflows_30d,
                   net_outflows_30d, lcr, asf, rsf, nsfr
            FROM results.lcr_nsfr WHERE report_date=:rd
        ''', rd=RD)
    except Exception:
        print('  [lcr_nsfr] results.lcr_nsfr not available — skipping')
        return

    if lq.empty:
        print('  [lcr_nsfr] no data — skipping')
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(f'Liquidity Coverage Ratio & Net Stable Funding Ratio  |  {RD.date()}',
                 fontsize=12, fontweight='bold')

    tot = lq.sum(numeric_only=True)
    hqla_v    = tot.get('hqla', 0)
    outflow_v = tot.get('net_outflows_30d', 0)
    lcr_agg   = hqla_v / outflow_v * 100 if outflow_v > 0 else 0
    asf_v     = tot.get('asf', 0)
    rsf_v     = tot.get('rsf', 0)
    nsfr_agg  = asf_v / rsf_v * 100 if rsf_v > 0 else 0

    # Waterfall: HQLA vs net outflows → LCR
    comp_vals  = [hqla_v / 1e9, -outflow_v / 1e9]
    comp_lbls  = ['HQLA\n(Level 1)', 'Net Outflows\n30-day']
    comp_cols  = [C_OK, C_L]
    axes[0].bar(range(2), comp_vals, color=comp_cols, alpha=0.85, edgecolor='white', width=0.5)
    for i, v in enumerate(comp_vals):
        axes[0].text(i, v + (0.05 if v > 0 else -0.15), f'{v:.1f}B',
                     ha='center', fontsize=9, fontweight='bold')
    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].set_xticks(range(2))
    axes[0].set_xticklabels(comp_lbls, fontsize=9)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
    axes[0].set_ylabel('B PLN')
    axes[0].set_title(f'LCR Components\n(aggregate: {lcr_agg:.0f}%)')

    # LCR by currency
    lq2 = lq.sort_values('currency')
    if 'lcr' in lq2.columns:
        lcr_pct = lq2['lcr'] * 100
        bar_c   = [C_OK if v >= 100 else C_L for v in lcr_pct]
        axes[1].bar(range(len(lq2)), lcr_pct, color=bar_c, alpha=0.85, edgecolor='white', width=0.5)
        for i, v in enumerate(lcr_pct):
            axes[1].text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')
        axes[1].axhline(100, color='black', lw=1.8, ls='--', label='Min 100%')
        axes[1].set_xticks(range(len(lq2)))
        axes[1].set_xticklabels(lq2['currency'], fontsize=10)
        axes[1].set_ylabel('LCR (%)')
        axes[1].set_title('LCR by Currency')
        axes[1].legend(fontsize=8)

    # NSFR by currency
    if 'asf' in lq2.columns and 'rsf' in lq2.columns:
        nsfr_pct = (lq2['asf'] / lq2['rsf'].replace(0, np.nan) * 100).fillna(0)
        bar_c3   = [C_OK if v >= 100 else C_L for v in nsfr_pct]
        axes[2].bar(range(len(lq2)), nsfr_pct, color=bar_c3, alpha=0.85, edgecolor='white', width=0.5)
        for i, v in enumerate(nsfr_pct):
            axes[2].text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=9, fontweight='bold')
        axes[2].axhline(100, color='black', lw=1.8, ls='--', label='Min 100%')
        axes[2].set_xticks(range(len(lq2)))
        axes[2].set_xticklabels(lq2['currency'], fontsize=10)
        axes[2].set_ylabel('NSFR (%)')
        axes[2].set_title('NSFR by Currency')
        axes[2].legend(fontsize=8)

    fig.tight_layout()
    save(fig, 'lcr_nsfr.png')


# ── 13. LIQUIDITY GAP ─────────────────────────────────────────────────────────
def make_liq_gap():
    lg = rsql('''
        SELECT tenor_bucket, bs_side,
               bucket_start_dt, bucket_end_dt, gap_cf
        FROM irrbb.liq_gap_beh WHERE currency=:ccy
    ''', ccy=CCY)

    if lg.empty:
        print('  [liq_gap] no data — skipping')
        return

    lg['_s'] = lg['tenor_bucket'].map(tkey)
    lg = lg[lg['_s'] <= 60].copy()
    agg = lg.groupby('tenor_bucket').agg(gap=('gap_cf', 'sum'), _s=('_s', 'first')).reset_index()
    agg = agg.sort_values('_s').reset_index(drop=True)
    agg['cum'] = agg['gap'].cumsum()
    x = np.arange(len(agg))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    gridspec_kw={'height_ratios': [1.6, 1]})
    fig.suptitle(f'Behavioural Liquidity Gap  |  {CCY}  |  {RD.date()}',
                 fontsize=12, fontweight='bold')

    bar_cols = [C_A if v >= 0 else C_L for v in agg['gap']]
    ax1.bar(x, agg['gap'] / 1e9, color=bar_cols, alpha=0.80, edgecolor='white')
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg['tenor_bucket'], rotation=45, ha='right', fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
    ax1.set_ylabel('B PLN')
    ax1.set_title('Periodic Net Cash Flow by Tenor Bucket  (positive = net inflow)')

    ax2.plot(x, agg['cum'] / 1e9, color=C_N, lw=2.0, marker='o', markersize=4)
    ax2.fill_between(x, agg['cum'] / 1e9, 0,
                     where=(agg['cum'] >= 0), alpha=0.15, color=C_A)
    ax2.fill_between(x, agg['cum'] / 1e9, 0,
                     where=(agg['cum'] < 0), alpha=0.15, color=C_L)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(agg['tenor_bucket'], rotation=45, ha='right', fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}B'))
    ax2.set_ylabel('B PLN')
    ax2.set_title('Cumulative Liquidity Gap')
    fig.tight_layout()
    save(fig, 'liq_gap.png')


# ── RUN ALL ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    steps = [
        ('pipeline diagram',   make_pipeline),
        ('SQL schema',         make_sql_schema),
        ('balance sheet',      make_bs_structure),
        ('rate type exposure', make_rate_type),
        ('NII waterfall',      make_nii_waterfall),
        ('NII bridge',         make_nii_bridge),
        ('repricing gap',      make_repricing_gap),
        ('shock curves',       make_shock_curves),
        ('NII scenarios',      make_nii_scenarios),
        ('EVE results',        make_eve_results),
        ('EBA SOT',            make_eba_sot),
        ('LCR / NSFR',         make_lcr_nsfr),
        ('liquidity gap',      make_liq_gap),
    ]
    for name, fn in steps:
        print(f'\n[{name}]')
        try:
            fn()
        except Exception as exc:
            print(f'  ERROR: {exc}')

    print(f'\nDone. Assets in: {OUT}')
