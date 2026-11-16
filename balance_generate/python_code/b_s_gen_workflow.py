import pandas as pd
import numpy as np
import os
import sys
from b_s_gen_objects import (ProductFactory, compute_deposit_client_rt, compute_asset_rates, decorate_from_bs_struct,
                             seed_product)
from sqlalchemy import create_engine
import config
import sql_setup
from sqlalchemy import text

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
from labbank_common import rate_models
from labbank_common import fx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

# A rolled-forward bank (bank_config.RollforwardCfg) is not generated: its base bank's book
# is aged to its report date and only the missing volume is new business.
if config.bank.rollforward is not None:
    from b_s_rollforward import run_rollforward
    run_rollforward(config.bank)
    sys.exit(0)


config.report_date = pd.Timestamp(config.bank.report_date)  # the bank's report date
config.balance_start_date = pd.to_datetime(config.bank.generation.balance_start_date)
total_assets = config.bank.generation.total_assets
full_balance_amt = total_assets * 2 #assets + liabilities+equity = 2*assets

mode = 0 # 0 -- create new tables and remove old , 1 -- only delete rows without creating new schema of table

_bank_data = config.bank.balance_dir() / 'bank_data.xlsx'
from labbank_common import catalogue
df_bs_struct = catalogue.read_bs_structure(_bank_data)   # + catalogue variants at 0% (not generated)

# Committed facilities (off balance sheet, labbank_common/commitments.py). Card limits are a
# facility line like any other: the card generator takes its utilisation from there.
from labbank_common import commitments as cmt
df_commit = cmt.read_commitments(_bank_data)
df_bs_struct = cmt.inject_card_terms(df_bs_struct, df_commit)

df_client_t = pd.read_excel('dictionaries/client_type.xlsx')
# bs_percentage is a share of the balance sheet in the HOME currency; a foreign-currency product
# (e.g. USD in the EUR bank) is booked in its own currency: home amount x units per home unit.
df_bs_struct['balance_amt'] = (full_balance_amt * df_bs_struct['bs_percentage'] / 100
                               / fx.to_home_series(df_bs_struct['currency'], config.bank, config.report_date))
df_bs_struct['amort_type'] = df_bs_struct['amort_type'].astype('Int64')
df_result = df_bs_struct.merge(df_client_t, on='client_type_id', how='left')

# Client-rate model (beta, margin_pct, floors, caps per product) and historical fixings.
# The model lives with the other behavioural models in inputs/<bank>/models and is
# loaded into SQL (bs.models_rate) by the add_data stage; generation reads the Excel
# directly because it runs before add_data. rate_index per product comes from bank_data.xlsx.
interest_rt = rate_models.read_rate_excel(config.report_date).reset_index()
from labbank_common.credit_cards import validate_card_inputs
validate_card_inputs(df_bs_struct, interest_rt.set_index('product_code'))
fixings_raw = fx.read_market_input('fixing_input.xlsx', config.bank)   # every currency the bank holds
fixings_raw = fixings_raw.rename(columns={'date': 'fixing_date'})

# Validate inputs before the existing reset operation.
sql_setup.reset_data(mode, report_date=config.report_date)

# Reproducible book: every draw below uses numpy's global RNG (truncnorm.rvs, np.random.choice),
# re-seeded per product so a mix change re-draws only the products whose balance changed
np.random.seed(config.bank.generation.seed)

product_objects = {}


parts = []
for _, row in df_result.iterrows():
    # bs_percentage = 0: a catalogue product the bank offers but does not hold yet. No
    # contracts; its bank_data row stays the product's definition for the roll-forward and
    # the optimizers.
    if not row['balance_amt'] > 0:
        continue
    seed_product(config.bank.generation.seed, row['product_code'], row['bs_side'])
    p = ProductFactory.create(row)
    df = p.build_result_df()
    parts.append(df)                       # NIE zapisujemy jeszcze do transactions

# nadaj client_id i zrób parowania na CAŁOŚCI
all_tx = pd.concat(parts, ignore_index=True)
all_tx = sql_setup.add_client_id(all_tx, seed=42, pct=0.4)

all_tx = decorate_from_bs_struct(all_tx, df_bs_struct)

# Compute curr_rt and client_rt for deposit products (formula from interest_rt.xlsx)
all_tx = compute_deposit_client_rt(all_tx, interest_rt, df_bs_struct, fixings_raw, config.report_date)

# Compute curr_rt and client_rt for loan and fin_inst products (current fixing + margin)
all_tx = compute_asset_rates(all_tx, interest_rt, df_bs_struct, fixings_raw, config.report_date)

# teraz dopiero:
sql_setup.append_df_to_table(all_tx, "transactions")   # parent
# potem dzieci:
for cls, table in ProductFactory.table_registry.items():
    pnames = [k for k,v in ProductFactory.class_registry.items() if v is cls]
    subset = all_tx[all_tx["product_name"].isin(pnames)]
    if not subset.empty:
        sql_setup.append_df_to_table(subset, table)

# Facility register (schemat.commitments): limits, drawn and undrawn amounts per facility line.
# Drawn amounts reconcile to the linked asset products; undrawn = limit - drawn, off balance.
register = cmt.write_register(sql_setup.engine, df_commit, all_tx, total_assets,
                              fx.to_home(config.bank, config.report_date), config.report_date)
if not register.empty:
    print(register[['commitment_code', 'own_name', 'currency', 'limit_amt', 'drawn_amt', 'undrawn_amt']].to_string(index=False))
