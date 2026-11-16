"""Per-bank release check: compare each bank's results with its accepted reference.

Every bank's balance sheet is generated from a fixed seed (bank_config.GenerationCfg.seed),
so the same code + the same inputs reproduce the same results exactly. A shared-code change
made for one bank must therefore leave every other bank UNCHANGED; anything else is shown.

    python release_check/release_check.py                     # check all banks' current state
    python release_check/release_check.py --run               # rerun each pipeline first (~20 min/bank)
    python release_check/release_check.py --bank eur --run
    python release_check/release_check.py --bank eur --accept # current EUR state = new EUR reference

Result per bank: UNCHANGED / CHANGED (which tables, files and key numbers moved) /
CRASHED (which pipeline step failed). Exit code 0 = all unchanged, 1 = changes, 2 = crash.

A reference (references/<bank>.json, a few KB) stores per SQL table its row count and an
order-independent fingerprint, a fingerprint per output file (npz arrays, Excel sheets) and
readable key numbers (IRRBB report per scenario, LCR/NSFR, balance by product, fast metrics).
Only accept a bank whose change you intended -- the other banks keep their references.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from labbank_common.bank_config import BANKS, get_bank, get_engine, total_assets  # noqa: E402

REF_DIR = Path(__file__).resolve().parent / "references"
DECIMALS = 6                     # float rounding before fingerprinting (results are exact anyway)

from labbank_common import pipeline as _pipeline  # noqa: E402
PIPELINE = _pipeline.STEPS + _pipeline.ACCOUNTING_STEPS      # one list: labbank_common/pipeline.py
OUTPUT_MODULES = ["irrbb_calc", "liq_calc", "optimize_prep", "sandbox"]
SKIP_SCHEMAS = {"meta"}          # run registry: timestamps / git sha change on every run
FILE_LOCK = "[Errno 22] Invalid argument"      # transient Windows lock while overwriting outputs


# ── fingerprints ──────────────────────────────────────────────────────────────

def _frame_fingerprint(df: pd.DataFrame) -> str:
    """Order-independent fingerprint of a table: sum of per-row hashes over sorted columns."""
    if df.empty:
        return "empty"
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    df = df[sorted(df.columns)]
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_float_dtype(s) or s.dtype == object and _is_decimal(s):
            df[c] = pd.to_numeric(s, errors="coerce").round(DECIMALS)
        elif pd.api.types.is_datetime64_any_dtype(s):
            df[c] = s.astype(str)
        elif s.dtype == object:
            df[c] = s.astype(str)
    h = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)
    return f"{int(h.sum(dtype=np.uint64)):016x}"


def _is_decimal(s: pd.Series) -> bool:
    v = s.dropna()
    return len(v) > 0 and type(v.iloc[0]).__name__ == "Decimal"


def _array_fingerprint(a: np.ndarray) -> str:
    if a.dtype.kind == "f":
        a = np.round(a, DECIMALS)
    elif a.dtype.kind not in "iub":
        a = a.astype(str)
    return f"{a.shape}:{pd.util.hash_array(np.ravel(a)).sum(dtype=np.uint64):016x}"


def sql_fingerprints(bank) -> dict:
    """Row count + order-independent checksum per table, computed inside SQL Server
    (floats rounded to DECIMALS) -- seconds instead of pulling millions of rows to Python."""
    eng = get_engine(bank)
    cols = pd.read_sql(
        "SELECT TABLE_SCHEMA sch, TABLE_NAME tbl, COLUMN_NAME col, DATA_TYPE typ "
        "FROM INFORMATION_SCHEMA.COLUMNS c WHERE TABLE_NAME <> 'sysdiagrams' AND EXISTS "
        "(SELECT 1 FROM INFORMATION_SCHEMA.TABLES t WHERE t.TABLE_SCHEMA = c.TABLE_SCHEMA "
        " AND t.TABLE_NAME = c.TABLE_NAME AND t.TABLE_TYPE = 'BASE TABLE') "
        "ORDER BY TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME", eng)
    out = {}
    from labbank_common import run_store
    run = run_store.current_run(bank)
    for (sch, tbl), g in cols.groupby(["sch", "tbl"], sort=True):
        if sch in SKIP_SCHEMAS:
            continue
        exprs = []
        # history tables (docs/04 §8): fingerprint this bank's run only; the run_id column itself is
        # bookkeeping, so a table that gained it still matches a reference recorded before
        where = ""
        if {"report_date", "run_id"} <= set(g.col):
            where = f" WHERE report_date = '{run.report_date.date()}' AND run_id = '{run.run_id}'"
        for col, typ in zip(g.col, g.typ):
            if col == "run_id":
                continue
            c = f"[{col}]"
            if typ in ("float", "real", "decimal", "numeric", "money"):
                exprs.append(f"ROUND(CAST({c} AS FLOAT), {DECIMALS})")
            elif typ in ("text", "ntext", "image", "xml"):
                exprs.append(f"CAST({c} AS NVARCHAR(4000))")
            else:
                exprs.append(c)
        q = (f"SELECT COUNT_BIG(*) n, CHECKSUM_AGG(BINARY_CHECKSUM({', '.join(exprs)})) h "
             f"FROM [{sch}].[{tbl}]{where}")
        n, h = pd.read_sql(q, eng).iloc[0]
        out[f"{sch}.{tbl}"] = {"rows": int(n), "fingerprint": "empty" if pd.isna(h) else f"{int(h) & 0xFFFFFFFF:08x}"}
    return out


def file_fingerprints(bank) -> dict:
    out = {}
    for module in OUTPUT_MODULES:
        folder = bank.output_dir(module, create=False)
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            key = f"{module}/{f.name}"
            if f.suffix == ".npz":
                with np.load(f, allow_pickle=True) as z:
                    out[key] = {k: _array_fingerprint(z[k]) for k in sorted(z.files)}
            elif f.suffix == ".xlsx":
                sheets = pd.read_excel(f, sheet_name=None)
                out[key] = {s: _frame_fingerprint(d) for s, d in sorted(sheets.items())}
    return out


def _run_where(eng, table: str, bank) -> str:
    """SQL condition selecting the bank's own report date (and run, once the table has run_id)."""
    from labbank_common import run_store
    run = run_store.current_run(bank)
    cols = set(run_store.columns(eng, table))
    cond = [f"report_date = '{run.report_date.date()}'"] if "report_date" in cols else ["1 = 1"]
    if "run_id" in cols:
        cond.append(f"run_id = '{run.run_id}'")
    return " AND ".join(cond)


def key_numbers(bank) -> dict:
    eng = get_engine(bank)
    out = {}
    try:
        r = pd.read_sql("SELECT currency, scenario_id, nii_base, delta_nii, sot_nii_pct, eve_base, delta_eve, sot_eve_pct "
                        f"FROM results.irrbb_report WHERE {_run_where(eng, 'results.irrbb_report', bank)}", eng)
        if (r["currency"] == "ALL").any():          # multi-currency bank: the all-currency total
            r = r[r["currency"] == "ALL"]
        out["irrbb"] = {row.scenario_id: {k: round(float(getattr(row, k)), 4) for k in
                                          ("nii_base", "delta_nii", "sot_nii_pct", "eve_base", "delta_eve", "sot_eve_pct")}
                        for row in r.itertuples()}
    except Exception as e:
        out["irrbb"] = f"unavailable: {type(e).__name__}"
    try:
        l = pd.read_sql(f"SELECT currency, lcr, nsfr FROM results.lcr_nsfr WHERE {_run_where(eng, 'results.lcr_nsfr', bank)}", eng)
        out["liquidity"] = {row.currency: {"lcr": round(float(row.lcr), 6), "nsfr": round(float(row.nsfr), 6)}
                            for row in l.itertuples()}
    except Exception as e:
        out["liquidity"] = f"unavailable: {type(e).__name__}"
    try:
        b = pd.read_sql("SELECT product_code, bs_side, COUNT(*) n, SUM(CAST(balance_amt AS FLOAT)) bal "
                        f"FROM dbo.transactions WHERE {_run_where(eng, 'dbo.transactions', bank)} "
                        "GROUP BY product_code, bs_side", eng)
        out["balance_by_product"] = {f"{row.product_code}/{row.bs_side}": {"n": int(row.n), "balance": round(float(row.bal), 2)}
                                     for row in b.itertuples()}
    except Exception as e:
        out["balance_by_product"] = f"unavailable: {type(e).__name__}"
    out["fast_metrics"] = fast_metrics(bank)
    return out


def fast_metrics(bank) -> dict | str:
    """Sandbox / optimizer fast-approximation metrics from the bank's committed npz files."""
    folder = bank.output_dir("optimize_prep", create=False)
    params, curves = folder / "product_params.npz", folder / "curve_tensors.npz"
    if not (params.exists() and curves.exists()):
        return "unavailable: npz files missing"
    sys.path.insert(0, str(ROOT / "optimize_prep" / "python_code"))
    from metrics import compute_all_metrics, load_params_and_curves, reset_bias_cache   # noqa: E402
    reset_bias_cache(str(folder / "bias_corrections.npz"))   # this bank's file, whatever LABBANK_BANK says
    p, c = load_params_and_curves(params_path=str(params), curves_path=str(curves))
    m = compute_all_metrics(p.current_weights(), p, c, total_assets(bank))
    ccy = bank.market.ccy
    return {"nii_base": round(float(m.nii_base), 2), "eve_base": round(float(m.eve_base), 2),
            "worst_delta_eve": round(float(m.worst_delta_eve), 2), "worst_delta_nii": round(float(m.worst_delta_nii), 2),
            "lcr": round(float(m.lcr.get(ccy, np.nan)), 6), "nsfr": round(float(m.nsfr.get(ccy, np.nan)), 6)}


def state(bank) -> dict:
    return {"bank": bank.bank_id, "report_date": bank.report_date, "seed": bank.generation.seed,
            "recorded": datetime.now().isoformat(timespec="seconds"),
            "key_numbers": key_numbers(bank), "sql": sql_fingerprints(bank), "files": file_fingerprints(bank)}


# ── pipeline run ──────────────────────────────────────────────────────────────

def run_pipeline(bank, log_dir: Path) -> str | None:
    """Run the full pipeline for one bank; returns the failed step (None = success).
    The run is registered in the bank's meta.runs (labbank_common/pipeline.run_steps)."""
    return _pipeline.run_steps(bank, PIPELINE, log_dir)


# ── comparison ────────────────────────────────────────────────────────────────

def _diff_numbers(ref, cur, path="") -> list[str]:
    if isinstance(ref, dict) and isinstance(cur, dict):
        out = []
        for k in sorted(set(ref) | set(cur), key=str):
            if k not in cur:
                out.append(f"{path}{k}: removed")
            elif k not in ref:
                out.append(f"{path}{k}: new")
            else:
                out += _diff_numbers(ref[k], cur[k], f"{path}{k}.")
        return out
    if ref != cur:
        if isinstance(ref, (int, float)) and isinstance(cur, (int, float)):
            return [f"{path[:-1]}: {ref:,.4f} -> {cur:,.4f} ({cur - ref:+,.4f})"]
        return [f"{path[:-1]}: {ref} -> {cur}"]
    return []


def compare(ref: dict, cur: dict) -> list[str]:
    lines = []
    for section in ("sql", "files"):
        r, c = ref.get(section, {}), cur.get(section, {})
        for k in sorted(set(r) | set(c)):
            if k not in c:
                lines.append(f"  {section}: {k} missing")
            elif k not in r:
                lines.append(f"  {section}: {k} new")
            elif r[k] != c[k]:
                if section == "sql" and r[k]["rows"] != c[k]["rows"]:
                    lines.append(f"  sql: {k} rows {r[k]['rows']:,} -> {c[k]['rows']:,}")
                elif section == "sql":
                    lines.append(f"  sql: {k} values changed ({c[k]['rows']:,} rows)")
                else:
                    parts = sorted(p for p in set(r[k]) | set(c[k]) if r[k].get(p) != c[k].get(p))
                    lines.append(f"  file: {k} changed: {', '.join(parts[:8])}{' ...' if len(parts) > 8 else ''}")
    nums = _diff_numbers(ref.get("key_numbers", {}), cur.get("key_numbers", {}))
    if nums:
        lines.append("  key numbers:")
        lines += [f"    {n}" for n in nums[:60]] + ([f"    ... {len(nums) - 60} more"] if len(nums) > 60 else [])
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", default="all", help=f"bank id ({', '.join(BANKS)}) or 'all'")
    ap.add_argument("--run", action="store_true", help="rerun the full pipeline per bank before checking")
    ap.add_argument("--accept", action="store_true", help="store the current state as the bank's new reference")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    banks = list(BANKS) if args.bank == "all" else [args.bank]
    if args.accept and args.bank == "all":
        ap.error("--accept needs an explicit --bank (accept one bank at a time, on purpose)")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "release_check" / "logs"
    log_dir.mkdir(exist_ok=True)

    worst = 0
    for bank_id in banks:
        bank = get_bank(bank_id)
        print(f"\n=== {bank.name} ({bank.bank_id}, report date {bank.report_date})")
        if args.run:
            print("  running pipeline ...")
            failed = run_pipeline(bank, log_dir)
            if failed:
                print(f"  RESULT: CRASHED at {failed}")
                worst = max(worst, 2)
                continue
        cur = state(bank)
        ref_path = REF_DIR / f"{bank.bank_id}.json"
        if args.accept:
            ref_path.write_text(json.dumps(cur, indent=1, sort_keys=True), encoding="utf-8")
            print(f"  ACCEPTED -> {ref_path.relative_to(ROOT)} "
                  f"({len(cur['sql'])} tables, {len(cur['files'])} files)")
            continue
        if not ref_path.exists():
            print(f"  no reference yet -- create one with: --bank {bank.bank_id} --accept")
            worst = max(worst, 1)
            continue
        diff = compare(json.loads(ref_path.read_text(encoding="utf-8")), cur)
        if diff:
            print("  RESULT: CHANGED"); print("\n".join(diff))
            worst = max(worst, 1)
        else:
            print(f"  RESULT: UNCHANGED ({len(cur['sql'])} tables, {len(cur['files'])} files, key numbers identical)")
    return worst


if __name__ == "__main__":
    sys.exit(main())
