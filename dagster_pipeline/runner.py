"""Shared subprocess runner used by all Dagster assets."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# On Windows an output file being scanned (Defender / OneDrive) can briefly refuse to be
# overwritten: OSError [Errno 22] Invalid argument. It clears within seconds, so a step that
# failed with exactly that error is rerun; any other failure stops the run immediately.
FILE_LOCK_MARKER = "[Errno 22] Invalid argument"
FILE_LOCK_RETRIES = 3
FILE_LOCK_DELAY_S = 5


def _run_once(context, script_path: Path, env: dict) -> tuple[int, bool]:
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    file_lock = False
    for line in proc.stdout:
        context.log.info(line.rstrip())
        file_lock = file_lock or FILE_LOCK_MARKER in line
    proc.wait()
    return proc.returncode, file_lock


def run_workflow(context, script_path: Path) -> None:
    """Run a workflow script as a subprocess and stream output to Dagster logs."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # Bank partition -> labbank_common.bank_config.get_bank() in the workflow
    if context.has_partition_key:
        env["LABBANK_BANK"] = context.partition_key

    context.log.info(f"Starting: {script_path.name} (bank: {env.get('LABBANK_BANK', 'default')})")

    for attempt in range(1, FILE_LOCK_RETRIES + 2):
        returncode, file_lock = _run_once(context, script_path, env)
        if returncode == 0:
            return
        if not file_lock or attempt > FILE_LOCK_RETRIES:
            break
        context.log.warning(f"{script_path.name}: output file locked (Errno 22), "
                            f"retry {attempt}/{FILE_LOCK_RETRIES} in {FILE_LOCK_DELAY_S}s")
        time.sleep(FILE_LOCK_DELAY_S)

    raise RuntimeError(
        f"Script {script_path.name} failed with exit code {returncode}"
    )
