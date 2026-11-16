"""History-keeping SQL writes: one row set per (report_date, run_id) instead of drop-and-recreate.

Design: docs/04_monthly_reporting_design.md §3-4. A table written through this module keeps every
run: a rerun replaces only its own rows.

  run_id   'actual' for a reported book, 'fc1y' for the one-year roll-forward (a rolled bank such as
           eur_y1 is run 'fc1y' until the forecast moves into its base bank's database, §5.1);
           LABBANK_RUN overrides it.

  current_run(bank)                     RunContext(bank_id, report_date, run_id)
  ensure_schema(engine, schema)         CREATE SCHEMA if missing
  ensure_run_columns(engine, table)     adds report_date / run_id to an existing table (idempotent;
                                        rows written before get the run's key: they are that run)
  clear_run(engine, table, run)         DELETE FROM table WHERE report_date = :rd AND run_id = :run
  append(engine, df, table, schema)     stamps report_date + run_id, to_sql(append)
  replace_run(engine, df, table, schema)  ensure + clear + append: the history-safe "replace"
  reset_table(engine, table, schema, create)  a writer's reset: create if missing + clear this run
  register_run(engine, status, ...)     meta.runs upsert (running / ok / failed)
LABBANK_REBUILD_SCHEMA=1 makes reset_table drop and recreate (today's old behaviour), for schema changes.
  run_filter(alias)                     "report_date = :rd AND run_id = :run" for readers
  read_run(engine, table, run, where)   SELECT * of one run

Readers must filter by run once a table holds several (RUN_FILTER), or they sum runs silently.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text

from labbank_common.bank_config import BankConfig, PROJECT_ROOT, get_bank

RUN_ENV = "LABBANK_RUN"
ACTUAL = "actual"
FORECAST_1Y = "fc1y"
RUN_FILTER = "report_date = :rd AND run_id = :run"
REBUILD_ENV = "LABBANK_REBUILD_SCHEMA"   # "1": drop and recreate instead (schema changes; loses history)


@dataclass(frozen=True)
class RunContext:
    bank_id: str
    report_date: pd.Timestamp
    run_id: str
    base_report_date: pd.Timestamp | None = None

    @property
    def kind(self) -> str:
        return "actual" if self.run_id == ACTUAL else "forecast"

    @property
    def params(self) -> dict:
        return {"rd": self.report_date.date(), "run": self.run_id}


def current_run(bank: BankConfig | str | None = None, run_id: str | None = None) -> RunContext:
    b = bank if isinstance(bank, BankConfig) else get_bank(bank)
    default = FORECAST_1Y if b.rollforward is not None else ACTUAL
    run = run_id or os.environ.get(RUN_ENV) or default
    base = pd.Timestamp(get_bank(b.rollforward.base_bank).report_date) if b.rollforward is not None else None
    return RunContext(b.bank_id, pd.Timestamp(b.report_date), run, base)


def run_filter(alias: str = "") -> str:
    p = f"{alias}." if alias else ""
    return f"{p}report_date = :rd AND {p}run_id = :run"


def _split(table: str, schema: str | None) -> tuple[str, str]:
    if schema is None:
        schema, table = table.split(".", 1) if "." in table else ("dbo", table)
    return schema, table


def ensure_schema(engine, schema: str) -> None:
    with engine.begin() as c:
        c.execute(text(f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{schema}') "
                       f"EXEC('CREATE SCHEMA [{schema}]');"))


def table_exists(engine, table: str, schema: str | None = None) -> bool:
    s, t = _split(table, schema)
    with engine.connect() as c:
        return c.execute(text("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = :s AND TABLE_NAME = :t"),
                         {"s": s, "t": t}).first() is not None


def columns(engine, table: str, schema: str | None = None) -> list[str]:
    s, t = _split(table, schema)
    with engine.connect() as c:
        r = c.execute(text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = :s "
                           "AND TABLE_NAME = :t ORDER BY ORDINAL_POSITION"), {"s": s, "t": t})
        return [x[0] for x in r]


def ensure_run_columns(engine, table: str, schema: str | None = None, run: RunContext | None = None) -> None:
    """Add report_date / run_id to an existing table. Rows already there get the run's key
    (report_date only where it is missing): a table written by one run before the change holds
    exactly that run."""
    s, t = _split(table, schema)
    if not table_exists(engine, t, s):
        return
    run = run or current_run()
    cols = columns(engine, t, s)
    with engine.begin() as c:
        if "report_date" not in cols:
            c.execute(text(f"ALTER TABLE [{s}].[{t}] ADD report_date DATE NULL"))
            c.execute(text(f"UPDATE [{s}].[{t}] SET report_date = :rd"), {"rd": run.report_date.date()})
        if "run_id" not in cols:
            c.execute(text(f"ALTER TABLE [{s}].[{t}] ADD run_id VARCHAR(20) NULL"))
            c.execute(text(f"UPDATE [{s}].[{t}] SET run_id = :run"), {"run": run.run_id})


def clear_run(engine, table: str, schema: str | None = None, run: RunContext | None = None) -> int:
    s, t = _split(table, schema)
    if not table_exists(engine, t, s):
        return 0
    run = run or current_run()
    with engine.begin() as c:
        return c.execute(text(f"DELETE FROM [{s}].[{t}] WHERE {RUN_FILTER}"), run.params).rowcount


def stamp(df: pd.DataFrame, run: RunContext) -> pd.DataFrame:
    out = df.copy()
    for c in ("report_date", "run_id"):
        if c in out.columns:
            out = out.drop(columns=c)
    out.insert(0, "run_id", run.run_id)
    out.insert(0, "report_date", run.report_date.normalize())
    return out


def append(engine, df: pd.DataFrame, table: str, schema: str | None = None, run: RunContext | None = None,
           chunksize: int = 10_000) -> None:
    s, t = _split(table, schema)
    run = run or current_run()
    ensure_schema(engine, s)
    out = stamp(df, run)
    if table_exists(engine, t, s):                # a new column of the frame: add it (never drop)
        have = set(columns(engine, t, s))
        new = [c for c in out.columns if c not in have]
        if new:
            with engine.begin() as c:
                for col in new:
                    typ = "FLOAT" if pd.api.types.is_numeric_dtype(out[col]) else "NVARCHAR(255)"
                    c.execute(text(f"ALTER TABLE [{s}].[{t}] ADD [{col}] {typ} NULL"))
    out.to_sql(t, engine, schema=s, if_exists="append", index=False, chunksize=chunksize)


def replace_run(engine, df: pd.DataFrame, table: str, schema: str | None = None,
                run: RunContext | None = None) -> None:
    """The history-safe replacement of ``to_sql(if_exists='replace')``: only this run's rows go."""
    run = run or current_run()
    ensure_run_columns(engine, table, schema, run)
    clear_run(engine, table, schema, run)
    append(engine, df, table, schema, run)


def read_run(engine, table: str, run: RunContext | None = None, where: str = "", schema: str | None = None,
             columns_sql: str = "*") -> pd.DataFrame:
    s, t = _split(table, schema)
    run = run or current_run()
    extra = f" AND ({where})" if where else ""
    return pd.read_sql(text(f"SELECT {columns_sql} FROM [{s}].[{t}] WHERE {RUN_FILTER}{extra}"), engine,
                       params=run.params)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def register_run(engine, status: str, run: RunContext | None = None, market_path: str | None = None,
                 step: str | None = None) -> None:
    """meta.runs upsert: (report_date, run_id) -> base_report_date, kind, market_path, git_sha,
    started_at (first registration), finished_at (status ok / failed), status, last_step."""
    run = run or current_run()
    ensure_schema(engine, "meta")
    with engine.begin() as c:
        c.execute(text("""
            IF OBJECT_ID('meta.runs') IS NULL
            CREATE TABLE meta.runs (
                report_date DATE NOT NULL, run_id VARCHAR(20) NOT NULL, bank_id VARCHAR(40) NULL,
                base_report_date DATE NULL, kind VARCHAR(10) NOT NULL, market_path VARCHAR(20) NULL,
                git_sha VARCHAR(40) NULL, started_at DATETIME2 NULL, finished_at DATETIME2 NULL,
                status VARCHAR(10) NOT NULL, last_step VARCHAR(80) NULL,
                CONSTRAINT pk_meta_runs PRIMARY KEY (report_date, run_id))"""))
        p = dict(run.params, bank=run.bank_id, base=run.base_report_date.date() if run.base_report_date is not None
                 else None, kind=run.kind, mp=market_path, sha=_git_sha(), st=status, step=step)
        c.execute(text("""
            MERGE meta.runs AS t
            USING (SELECT :rd AS report_date, :run AS run_id) AS s
               ON t.report_date = s.report_date AND t.run_id = s.run_id
            WHEN MATCHED THEN UPDATE SET status = :st, git_sha = COALESCE(:sha, t.git_sha),
                 last_step = COALESCE(:step, t.last_step), market_path = COALESCE(:mp, t.market_path),
                 started_at = CASE WHEN :st = 'running' AND t.status <> 'running' THEN SYSDATETIME() ELSE t.started_at END,
                 finished_at = CASE WHEN :st IN ('ok', 'failed') THEN SYSDATETIME() ELSE NULL END
            WHEN NOT MATCHED THEN INSERT (report_date, run_id, bank_id, base_report_date, kind, market_path, git_sha,
                 started_at, finished_at, status, last_step)
                 VALUES (:rd, :run, :bank, :base, :kind, :mp, :sha, SYSDATETIME(),
                         CASE WHEN :st IN ('ok', 'failed') THEN SYSDATETIME() END, :st, :step);"""), p)


def latest_ok(engine, run_id: str = ACTUAL) -> pd.Timestamp | None:
    if not table_exists(engine, "runs", "meta"):
        return None
    with engine.connect() as c:
        r = c.execute(text("SELECT MAX(report_date) FROM meta.runs WHERE run_id = :run AND status = 'ok'"),
                      {"run": run_id}).scalar()
    return pd.Timestamp(r) if r is not None else None


def reset_table(engine, table: str, schema: str | None = None, create=None, run: RunContext | None = None) -> None:
    """History-safe replacement of a writer's DROP + CREATE: create the table when it is missing
    (``create(conn)``, e.g. ``lambda c: metadata.create_all(bind=c, tables=[T], checkfirst=True)``),
    add the run columns, delete this run's rows. Other dates and runs stay."""
    s, t = _split(table, schema)
    run = run or current_run()
    ensure_schema(engine, s)
    if os.environ.get(REBUILD_ENV) == "1" and table_exists(engine, t, s):   # schema migration only
        with engine.begin() as c:
            c.execute(text(f"DROP TABLE [{s}].[{t}]"))
    if not table_exists(engine, t, s):
        if create is None:
            return
        with engine.begin() as c:
            create(c)
    ensure_run_columns(engine, t, s, run)
    clear_run(engine, t, s, run)


def sa_create(metadata, table):
    """``create`` argument of reset_table for a SQLAlchemy Table."""
    return lambda conn: metadata.create_all(bind=conn, tables=[table], checkfirst=True)
