"""LabBank command line (docs/04 §5.3).

    python -m labbank_common.cli run --bank eur                          # whole pipeline + accounting
    python -m labbank_common.cli run --bank eur --from cf_calc_workflow  # partial rerun (step = script stem)
    python -m labbank_common.cli run --bank eur_y1 --only pnl_rollforward
    python -m labbank_common.cli runs --bank eur                         # meta.runs of the bank's database

--date sets LABBANK_REPORT_DATE (the bank's inputs must be for that date); --run sets LABBANK_RUN
(default 'actual', 'fc1y' for a rolled-forward bank). A forecast whose base actual is registered and
not 'ok' in the base bank's meta.runs is refused unless --force.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from labbank_common import pipeline, run_store  # noqa: E402
from labbank_common.bank_config import RUNS_ROOT, get_bank, get_engine  # noqa: E402


def cmd_run(a) -> int:
    bank = get_bank(a.bank)
    env = pipeline.step_env(bank, a.date, a.run)
    if bank.rollforward is not None and not a.force:
        base = get_bank(bank.rollforward.base_bank)
        beng = get_engine(base)
        if run_store.table_exists(beng, "runs", "meta"):
            st = pd.read_sql(text("SELECT status FROM meta.runs WHERE report_date = :rd AND run_id = 'actual'"),
                             beng, params={"rd": pd.Timestamp(base.report_date).date()})
            if not st.empty and st["status"].iloc[0] != "ok":
                print(f"base {base.bank_id} {base.report_date} is '{st['status'].iloc[0]}', not ok: "
                      "run it first or pass --force")
                return 2
    steps = pipeline.select_steps(a.from_step, a.only, accounting=not a.no_accounting)
    print(f"{bank.bank_id}: {len(steps)} steps")
    failed = pipeline.run_steps(bank, steps, RUNS_ROOT / bank.bank_id / "logs", env)
    return 1 if failed else 0


def cmd_runs(a) -> int:
    eng = get_engine(get_bank(a.bank))
    if not run_store.table_exists(eng, "runs", "meta"):
        print("no meta.runs yet")
        return 0
    pd.set_option("display.width", 200)
    print(pd.read_sql(text("SELECT * FROM meta.runs ORDER BY report_date, run_id"), eng).to_string(index=False))
    return 0




def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="labbank")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bank", required=True)
    r.add_argument("--date")
    r.add_argument("--run")
    r.add_argument("--from", dest="from_step")
    r.add_argument("--only", nargs="+")
    r.add_argument("--no-accounting", action="store_true")
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("runs")
    s.add_argument("--bank", required=True)
    s.set_defaults(fn=cmd_runs)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
