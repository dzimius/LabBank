from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, text
from sqlalchemy import Integer, String, ForeignKey, Date
from sqlalchemy.dialects.mssql import DECIMAL

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)


metadata = MetaData(schema=None)


def _ensure_schemas() -> None:
    """Create schemat schema if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text(
            "IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'schemat') "
            "EXEC('CREATE SCHEMA [schemat]');"
        ))


Transactions = Table(
    "transactions", metadata,
    Column("report_date", Date, nullable=False),
    Column("transaction_id", String(13), primary_key=True, nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("own_name", String(64), nullable=True),
    Column("bs_side", String(1), nullable=False),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("client_id", Integer, nullable=False),
    Column("client_type_id", Integer, nullable=False),
    schema="dbo",
)

Loans = Table(
    "loans", metadata,
    Column("report_date", Date, nullable=False),
    Column("transaction_id", String(13), ForeignKey('dbo.transactions.transaction_id'), nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("own_name", String(64), nullable=True),
    Column("bs_side", String(1), nullable=False),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("start_date", Date, nullable=True),
    Column("maturity_date", Date, nullable=True),
    Column("currency", String(3), nullable=False),
    Column("client_type_id", Integer, nullable=False),
    Column("rate_type", String(1), nullable=True),
    Column("maturity", String(4), nullable=True),
    Column("rate_index", String(11), nullable=True),
    Column("payment_freq", String(8), nullable=True),
    Column("fixing_freq", String(8), nullable=True),
    Column("dc_conv", String(7), nullable=True),
    Column("b_day_conv", String(25), nullable=True),
    Column("amort_type", String(1), nullable=True),
    Column("init_balance_amt", DECIMAL(18, 2), nullable=True),
    Column("margin", DECIMAL(6, 4), nullable=True),
    Column("index_rt", DECIMAL(18, 6), nullable=True),
    Column("client_rt", DECIMAL(18, 6), nullable=True),
    Column("disc_curve", String(20), nullable=True),
    Column("fwd_curve", String(20), nullable=True),
    Column("schedule_id", Integer, nullable=True),
    Column("hqla_class", String(4),   nullable=True),
    Column("haircut",    DECIMAL(6, 4), nullable=True),
    Column("rsf_weight", DECIMAL(6, 4), nullable=True),
    schema="schemat",
)

Deposits = Table(
    "deposits", metadata,
    Column("report_date", Date, nullable=False),
    Column("transaction_id", String(13), ForeignKey('dbo.transactions.transaction_id'), nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("own_name", String(64), nullable=True),
    Column("bs_side", String(1), nullable=False),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("start_date", Date, nullable=True),
    Column("maturity_date", Date, nullable=True),
    Column("currency", String(3), nullable=False),
    Column("client_type_id", Integer, nullable=False),
    Column("maturity", String(4), nullable=True),
    Column("amort_type", String(1), nullable=True),
    Column("rate_type", String(1), nullable=True),
    Column("dc_conv", String(7), nullable=True),
    Column("b_day_conv", String(25), nullable=True),
    Column("disc_curve", String(20), nullable=True),
    Column("fwd_curve", String(20), nullable=True),
    Column("index_rt", DECIMAL(18, 6), nullable=True),
    Column("client_rt", DECIMAL(18, 6), nullable=True),
    Column("schedule_id", Integer, nullable=True),
    Column("lcr_weight", DECIMAL(6, 4), nullable=True),
    Column("asf_weight", DECIMAL(6, 4), nullable=True),
    schema="schemat",
)

FinancialInstruments = Table(
    "financial_instruments", metadata,
    Column("report_date", Date, nullable=False),
    Column("transaction_id", String(13), ForeignKey('dbo.transactions.transaction_id'), nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("own_name", String(64), nullable=True),
    Column("bs_side", String(1), nullable=False),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("start_date", Date, nullable=True),
    Column("maturity_date", Date, nullable=True),
    Column("currency", String(3), nullable=False),
    Column("maturity", String(4), nullable=True),
    Column("amort_type", String(1), nullable=True),
    Column("rate_type", String(1), nullable=True),
    Column("rate_index", String(11), nullable=True),
    Column("payment_freq", String(8), nullable=True),
    Column("fixing_freq", String(8), nullable=True),
    Column("dc_conv", String(7), nullable=True),
    Column("b_day_conv", String(25), nullable=True),
    Column("disc_curve", String(20), nullable=True),
    Column("fwd_curve", String(20), nullable=True),
    Column("index_rt", DECIMAL(18, 6), nullable=True),
    Column("client_rt", DECIMAL(18, 6), nullable=True),
    Column("schedule_id", Integer, nullable=True),
    Column("hqla_class", String(4),    nullable=True),
    Column("haircut",    DECIMAL(6, 4), nullable=True),
    Column("lcr_weight", DECIMAL(6, 4), nullable=True),
    Column("asf_weight", DECIMAL(6, 4), nullable=True),
    Column("rsf_weight", DECIMAL(6, 4), nullable=True),
    schema="schemat",
)

Equity = Table(
    "equity", metadata,
    Column("report_date", Date, nullable=False),
    Column("transaction_id", String(13), ForeignKey('dbo.transactions.transaction_id'), nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("own_name", String(64), nullable=True),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("currency", String(3), nullable=True),
    schema="schemat",
)

# Rejestr tabel do wygodnego użycia
TABLES = {
    "transactions": Transactions,
    "loans": Loans,
    "deposits": Deposits,
    "financial_instruments": FinancialInstruments,
    "equity": Equity,
}

def append_df_to_table(df: pd.DataFrame, table_name: str) -> None:
    """Dopasuj df do schematu tabeli i wykonaj append. Brakujące kolumny -> NULL."""
    tbl = TABLES[table_name]
    schema = tbl.schema
    # kolejność i lista kolumn wg tabeli:
    cols = [c.name for c in tbl.columns]
    # dołóż brakujące kolumny (NaN -> później NULL)
    df_work = df.copy()

    for c in cols:
        if c not in df_work.columns:
            df_work[c] = pd.NA

    df2 = df_work[cols]
    df2 = df2.where(pd.notnull(df2), None)

    df2.to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists="append",
        index=False,
        chunksize=10_000,
        method=None,
    )

# ------------------------------------------------------------
# 4) ŁADOWANIE DANYCH
# ------------------------------------------------------------
# (A) Zbiorcza 'transactions' — tylko id i balance_amt

def add_client_id(transactions_df: pd.DataFrame, seed: Optional[int] = None, pct: float = 0.4) -> pd.DataFrame:
    """
    1) Nadaje unikalne client_id każdemu wierszowi.
    2) loans -> current_account: kopiuje client_id z loanów do części CA.
    3) Z pozostałych CA wybiera ~pct i przepisuje ich client_id do istniejących TD/SA (50/50).
    """
    if seed is not None:
        np.random.seed(seed)

    df = transactions_df.copy()
    df['product_name'] = df['product_name'].astype(str)

    # 0) unikalne ID dla KAŻDEGO wiersza
    df['client_id'] = np.arange(1, len(df) + 1)

    # 1) loans → current_account : ten sam client_id
    loan_pat = re.compile(r'(?:cash[\s_]*loan|mortgage)', flags=re.IGNORECASE)
    loans_mask = df['product_name'].str.contains(loan_pat, regex=True, na=False)
    loans_ids_unique = df.loc[loans_mask, 'client_id'].drop_duplicates()

    ca_idx_all = df.index[df['product_name'].eq('current_account')]
    n_pair = int(min(len(loans_ids_unique), len(ca_idx_all)))
    chosen_ca_for_loans = pd.Index([])
    if n_pair > 0:
        # (losowo) które CA sparować z loans
        chosen_ca_for_loans = pd.Index(np.random.choice(ca_idx_all, size=n_pair, replace=False))
        df.loc[chosen_ca_for_loans, 'client_id'] = loans_ids_unique.iloc[:n_pair].values

    # 2) Z POZOSTAŁYCH current_account wybierz % klientów i przepisz ich client_id do TD/SA
    ca_idx_other = df.index[df['product_name'].eq('current_account')].difference(chosen_ca_for_loans)

    td_idx_all = df.index[df['product_name'].eq('term_deposit')]
    sa_idx_all = df.index[df['product_name'].eq('saving_account')]

    n_other_ca = len(ca_idx_other)
    # granice według Twojej reguły + dostępność TD/SA
    bound1 = int(np.floor(pct * n_other_ca))
    bound2 = int(np.floor(pct * ((df['product_name'] == 'term_deposit').sum()
                                 + (df['product_name'] == 'current_account').sum())))
    available_slots = len(td_idx_all) + len(sa_idx_all)
    target = min(bound1, bound2, available_slots)

    if target > 0 and n_other_ca > 0 and available_slots > 0:
        chosen_ca_for_tdsa = np.random.choice(ca_idx_other, size=target, replace=False)
        chosen_ids = df.loc[chosen_ca_for_tdsa, 'client_id'].tolist()

        # rozdział ~50/50 na TD i SA, z losowym wyborem wierszy TD/SA
        td_take = min(target // 2, len(td_idx_all))
        sa_take = min(target - td_take, len(sa_idx_all))

        if td_take > 0:
            pick_td = np.random.choice(td_idx_all, size=td_take, replace=False)
            df.loc[pick_td, 'client_id'] = chosen_ids[:td_take]

        if sa_take > 0:
            pick_sa = np.random.choice(sa_idx_all, size=sa_take, replace=False)
            df.loc[pick_sa, 'client_id'] = chosen_ids[td_take:td_take + sa_take]

    return df

def append_transactions(transactions_df: pd.DataFrame) -> None:
    needed = ["transaction_id", "product_code", "product_name", "bs_side", "balance_amt", "currency",
              "client_id", "client_type_id"]
    missing = [c for c in needed if c not in transactions_df.columns]
    if missing:
        for c in missing:
            transactions_df[c] = pd.NA
    transactions_df = add_client_id(transactions_df)
    append_df_to_table(transactions_df[needed], "transactions")

def assign_client_ids(transactions_df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
    """Nadaje unikalne client_id, łączy loans ↔ current_account,
    a dla 40% pozostałych CA losowo zmienia produkt na TD/SA (50/50)."""
    if seed is not None:
        np.random.seed(seed)

    df = transactions_df.copy()
    df['product_name'] = df['product_name'].astype(str)

    # 1) unikalne ID dla każdego wiersza
    df['client_id'] = np.arange(1, len(df) + 1)

    # 2) loans ↔ current_account
    loan_pat = re.compile(r'(?:cash[\s_]*loan|mortgage)', flags=re.IGNORECASE)
    loans_mask = df['product_name'].str.contains(loan_pat, regex=True, na=False)
    loans_ids_unique = df.loc[loans_mask, 'client_id'].drop_duplicates()

    current_idx_all = df.index[df['product_name'].eq('current_account')]
    n = min(len(loans_ids_unique), len(current_idx_all))
    if n > 0:
        df.loc[current_idx_all[:n], 'client_id'] = loans_ids_unique.iloc[:n].values

    # 3) 40% pozostałych CA → TD/SA 50/50
    current_idx_other = df.index[df['product_name'].eq('current_account')].difference(current_idx_all[:n])
    n_other_ca = len(current_idx_other)
    n_td = (df['product_name'] == 'term_deposit').sum()
    n_ca_total = (df['product_name'] == 'current_account').sum()
    target = min(int(0.4 * n_other_ca), int(0.4 * (n_td + n_ca_total)))

    if target > 0 and n_other_ca > 0:
        chosen_idx = np.random.choice(current_idx_other, size=target, replace=False)
        new_products = np.random.choice(['term_deposit', 'saving_account'], size=target)
        df.loc[chosen_idx, 'product_name'] = pd.Series(new_products, index=chosen_idx)

    return df

def reset_data(mode: int, report_date=None) -> None:
    """
    mode=0 -> drop + recreate tabele
    mode=1 -> DELETE specific report_date data
    """
    child_tables = ["loans", "deposits", "financial_instruments", "equity"]

    _ensure_schemas()
    with engine.begin() as conn:
        if mode == 0:
            for t in child_tables:
                conn.execute(text(
                    f"IF OBJECT_ID('schemat.{t}', 'U') IS NOT NULL DROP TABLE schemat.{t};"
                ))
            conn.execute(text(
                "IF OBJECT_ID('dbo.transactions', 'U') IS NOT NULL DROP TABLE dbo.transactions;"
            ))
            metadata.create_all(bind=conn, checkfirst=False)

        elif mode == 1:
            if report_date is None:
                raise ValueError("For mode=1 report_date is needed")

            for t in child_tables:
                conn.execute(
                    text(f"DELETE FROM schemat.{t} WHERE report_date = :rd"),
                    {"rd": report_date},
                )
            conn.execute(
                text("DELETE FROM dbo.transactions WHERE report_date = :rd"),
                {"rd": report_date},
            )
        else:
            raise ValueError("Mode should be 0 or 1")