"""Export optimization_report.ipynb to a code-free PDF (or HTML fallback).

Usage
-----
    python export_optimization_report.py

The script:
  1. Executes optimization_report.ipynb in-place (cell outputs refreshed).
  2. Converts the executed notebook to PDF via nbconvert --to webpdf
     (uses a headless Chromium; no LaTeX required), with --no-input so only
     markdown, tables, and charts are included -- no source code.
  3. If webpdf fails (e.g. Playwright not installed), falls back to HTML.

Output is written next to the notebook:
  - optimization_report_clean.pdf   (or .html on fallback)

This is a code-free companion to optimization_report.html (the existing
full export, which still includes source code) -- see export_report.py in
visual_rep/ for the identical pattern used on bank_report.ipynb.

Requirements
------------
    pip install nbconvert nbformat jupyter-core
    pip install nbconvert[webpdf]          # for PDF via Chromium
    playwright install chromium            # one-time browser download
"""
import os
import subprocess
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).with_name("optimization_report.ipynb")
PDF_OUT  = NOTEBOOK.with_name("optimization_report_clean.pdf")
HTML_OUT = NOTEBOOK.with_name("optimization_report_clean.html")

# Suppress Python runtime warnings (zmq Proactor event loop on Windows, etc.)
_ENV = {**os.environ, "PYTHONWARNINGS": "ignore::RuntimeWarning"}


def run(cmd: list[str], *, capture_stderr: bool = False) -> tuple[int, str]:
    """Run a subprocess; return (returncode, captured_stderr)."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=False,
        env=_ENV,
        stderr=subprocess.PIPE if capture_stderr else None,
    )
    stderr_text = result.stderr.decode(errors="replace") if capture_stderr else ""
    return result.returncode, stderr_text


def main() -> None:
    nb = str(NOTEBOOK)

    # ── 1. Execute the notebook ────────────────────────────────────────────────
    print("Executing notebook…")
    rc, _ = run([
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1800",
        "--log-level=ERROR",          # suppress INFO / WARNING lines
        "--inplace",
        nb,
    ])
    if rc != 0:
        print("ERROR: notebook execution failed (see output above).")
        sys.exit(rc)

    # ── 2. Try PDF via webpdf (headless Chromium) ──────────────────────────────
    print("\nConverting to PDF (webpdf, code-free)…")
    rc, stderr = run([
        sys.executable, "-m", "nbconvert",
        "--to", "webpdf",
        "--no-input",
        "--log-level=ERROR",
        "--output", str(PDF_OUT),
        nb,
    ], capture_stderr=True)

    if rc == 0:
        print(f"\nDone — PDF saved to: {PDF_OUT}")
        return

    # ── 3. Fallback: HTML ──────────────────────────────────────────────────────
    if "playwright" in stderr.lower() or "ModuleNotFoundError" in stderr:
        print("\nwebpdf skipped — Playwright not installed. Falling back to HTML.")
        print("  Tip: pip install nbconvert[webpdf] && playwright install chromium")
    else:
        # Unexpected webpdf failure — surface the actual error
        print("\nwebpdf failed (unexpected error). Falling back to HTML.")
        if stderr.strip():
            print(stderr.strip())

    rc, _ = run([
        sys.executable, "-m", "nbconvert",
        "--to", "html",
        "--no-input",
        "--log-level=ERROR",
        "--output", str(HTML_OUT),
        nb,
    ])
    if rc == 0:
        print(f"\nDone — HTML saved to: {HTML_OUT}")
    else:
        print("\nERROR: HTML export also failed.")
        sys.exit(rc)


if __name__ == "__main__":
    main()
