"""Shared subprocess runner used by all Dagster assets."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_workflow(context, script_path: Path) -> None:
    """Run a workflow script as a subprocess and stream output to Dagster logs."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    context.log.info(f"Starting: {script_path.name}")

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    for line in proc.stdout:
        context.log.info(line.rstrip())
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"Script {script_path.name} failed with exit code {proc.returncode}"
        )
