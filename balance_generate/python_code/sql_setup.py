from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.dialects.mssql import DECIMAL
import pandas as pd

engine = create_engine(
    "mssql+pyodbc://maciek_d/bank_gen"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Trusted_Connection=yes",
    fast_executemany=True,
    future=True,
)


metadata = MetaData(schema=None)   # np. schema="dbo" jeśli używasz

Transactions = Table(
    "transactions", metadata,
    Column("transaction_id", String(13), primary_key=True, nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("bs_side", String(1), nullable=False),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("client_type_id", Integer, nullable=False)
)

Loans = Table(
    "loans", metadata,
    Column("transaction_id", String(13), ForeignKey('transactions.transaction_id'), nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("bs_side", String(1), nullable=False),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("client_type_id", Integer, nullable=False),
    Column("rate_type", String(1), nullable=True),
    Column("maturity", String(4), nullable=True),
    Column("rate_index", String(8), nullable=True),
    Column("amort_type", String(1), nullable=True),
    Column("init_balance_amt", DECIMAL(18, 2), nullable=True),
    Column("margin", DECIMAL(6, 4), nullable=True),
)

Deposits = Table(
    "deposits", metadata,
    Column("transaction_id", String(13), ForeignKey('transactions.transaction_id'), nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("bs_side", String(1), nullable=False),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("client_type_id", Integer, nullable=False),
    Column("maturity", String(4), nullable=True),
    Column("rate_type", String(1), nullable=True)
)

FinancialInstruments = Table(
    "financial_instruments", metadata,
    Column("transaction_id", String(13), ForeignKey('transactions.transaction_id'), nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("maturity", String(4), nullable=True),
    Column("rate_type", String(1), nullable=True),
    Column("rate_index", String(8), nullable=True)
)

Equity = Table(
    "equity", metadata,
    Column("transaction_id", String(13), ForeignKey('transactions.transaction_id'), nullable=False),
    Column("product_code", String(4), nullable=False),
    Column("product_name", String(64), nullable=False),
    Column("balance_amt", DECIMAL(18, 2), nullable=False),
    Column("currency", String(3), nullable=True)
)

# Rejestr tabel do wygodnego użycia
TABLES = {
    "transactions": Transactions,
    "loans": Loans,
    "deposits": Deposits,
    "financial_instruments": FinancialInstruments,
    "equity": Equity,
}

def append_df_to_table(df: pd.DataFrame, table_name: str):
    """Dopasuj df do schematu tabeli i wykonaj append. Brakujące kolumny -> NULL."""
    tbl = TABLES[table_name]
    # kolejność i lista kolumn wg tabeli:
    cols = [c.name for c in tbl.columns]
    # dołóż brakujące kolumny (NaN -> później NULL)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    # utnij nadmiarowe kolumny i ustaw kolejność
    df2 = df[cols].copy()
    # NaN/NaT -> None, żeby poszło jako NULL
    df2 = df2.where(pd.notnull(df2), None)

    # append
    df2.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        chunksize=10_000,
        method=None,
    )

# ------------------------------------------------------------
# 4) ŁADOWANIE DANYCH
# ------------------------------------------------------------
# (A) Zbiorcza 'transactions' — tylko id i balance_amt
def append_transactions(df):
    needed = ["transaction_id", "product_code", "product_name", "bs_side", "balance_amt", "currency", "client_type_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = pd.NA
    append_df_to_table(df[needed], "transactions")