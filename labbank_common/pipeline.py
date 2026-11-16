"""The pipeline's steps in order, and one runner for them (docs/04 §5.3).

Used by the CLI (``python -m labbank_common.cli run``) and by release_check. Every step is a script
that reads its bank / date / run from the environment (LABBANK_BANK, LABBANK_REPORT_DATE, LABBANK_RUN),
so each one still runs on its own. A run is registered in meta.runs of the bank's database
(running -> ok / failed, with the last step), labbank_common/run_store.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from labbank_common.bank_config import PROJECT_ROOT, BankConfig, get_engine

STEPS = [                     # same order as the Dagster labbank_data_job
    "balance_generate/python_code/b_s_gen_workflow.py",
    "balance_gen_add_data/python_code/b_s_add_data_workflow.py",
    "cash_flow_calc/python_code/cf_calc_workflow.py",
    "ir_derivatives/python_code/swap_workflow.py",
    "irrbb_calc/python_code/nii_calc_workflow.py",
    "irrbb_calc/python_code/eve_calc_workflow.py",
    "irrbb_calc/python_code/eba_sot_workflow.py",
    "irrbb_calc/python_code/csrbb_workflow.py",
    "liq_calc/python_code/lcr_nsfr_workflow.py",
    "liq_calc/python_code/off_balance_workflow.py",
    "optimize_prep/python_code/opt_prep_workflow.py",
    "sandbox/build_scenario_curves.py",
]
# accounting view (docs/07): valuation at the report date for every bank; the monthly P&L of the
# window for a rolled-forward bank (pnl_rollforward skips the others)
ACCOUNTING_STEPS = [
    "accounting/python_code/acct_valuation_workflow.py",
    "accounting/python_code/pnl_rollforward.py",
]
FILE_LOCK = "[Errno 22] Invalid argument"      # transient Windows lock while overwriting outputs


def step_env(bank: BankConfig, report_date: str | None = None, run_id: str | None = None) -> dict:
    env = {**os.environ, "LABBANK_BANK": bank.bank_id, "PYTHONIOENCODING": "utf-8"}
    if report_date:
        env["LABBANK_REPORT_DATE"] = report_date
    if run_id:
        env["LABBANK_RUN"] = run_id
    return env


def run_steps(bank: BankConfig, steps: list[str], log_dir: Path, env: dict | None = None,
              register: bool = True) -> str | None:
    """Run ``steps`` for one bank; returns the failed step (None = success)."""
    from labbank_common import run_store
    env = env or step_env(bank)
    log_dir.mkdir(parents=True, exist_ok=True)
    run = run_store.current_run(bank, env.get("LABBANK_RUN"))
    eng = get_engine(bank)
    mp = bank.rollforward.market_path if bank.rollforward is not None else None
    if register:
        run_store.register_run(eng, "running", run, mp)
    for step in steps:
        for attempt in range(1, 5):
            t0 = time.time()
            log = log_dir / f"{bank.bank_id}_{Path(step).stem}.log"
            with open(log, "w", encoding="utf-8", errors="replace") as fh:
                rc = subprocess.run([sys.executable, str(PROJECT_ROOT / step)], cwd=PROJECT_ROOT, env=env,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                print(f"    {Path(step).name:32s} ok  ({time.time() - t0:5.0f}s)")
                if register:
                    run_store.register_run(eng, "running", run, step=Path(step).stem)
                break
            if FILE_LOCK in log.read_text(encoding="utf-8", errors="replace") and attempt < 4:
                print(f"    {Path(step).name:32s} file lock, retry {attempt}")
                time.sleep(5)
                continue
            print(f"    {Path(step).name:32s} FAILED (log: {log})")
            if register:
                run_store.register_run(eng, "failed", run, step=Path(step).stem)
            return step
    if register:
        run_store.register_run(eng, "ok", run)
    return None


def select_steps(from_step: str | None = None, only: list[str] | None = None, accounting: bool = True) -> list[str]:
    steps = STEPS + (ACCOUNTING_STEPS if accounting else [])
    names = [Path(s).stem for s in steps]
    if only:
        missing = set(only) - set(names)
        if missing:
            raise ValueError(f"unknown steps {sorted(missing)}; known: {names}")
        return [s for s in steps if Path(s).stem in only]
    if from_step:
        if from_step not in names:
            raise ValueError(f"unknown step {from_step!r}; known: {names}")
        return steps[names.index(from_step):]
    return steps
