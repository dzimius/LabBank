"""Export bank_report.ipynb to PDF (or HTML fallback).

Usage
-----
    python export_report.py

The script:
  1. Executes bank_report.ipynb in-place (cell outputs refreshed).
  2. Converts the executed notebook to PDF via nbconvert --to webpdf
     (uses a headless Chromium; no LaTeX required).
  3. If webpdf fails (e.g. Playwright not installed), falls back to HTML.

Output is written next to the notebook:
  - bank_report.pdf   (or bank_report.html on fallback)

Requirements
------------
    pip install nbconvert nbformat jupyter-core
    pip install nbconvert[webpdf]          # for PDF via Chromium
    playwright install chromium            # one-time browser download
"""
import subprocess
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).with_name("bank_report.ipynb")
PDF_OUT  = NOTEBOOK.with_suffix(".pdf")
HTML_OUT = NOTEBOOK.with_suffix(".html")


def run(cmd: list[str]) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    nb = str(NOTEBOOK)

    # ── 1. Execute the notebook ────────────────────────────────────────────────
    print("Executing notebook…")
    rc = run([
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=600",
        "--inplace",
        nb,
    ])
    if rc != 0:
        print("ERROR: notebook execution failed (see output above).")
        sys.exit(rc)

    # ── 2. Try PDF via webpdf (headless Chromium) ──────────────────────────────
    print("\nConverting to PDF (webpdf)…")
    rc = run([
        sys.executable, "-m", "nbconvert",
        "--to", "webpdf",
        "--no-input",
        "--output", str(PDF_OUT),
        nb,
    ])
    if rc == 0:
        print(f"\nDone — PDF saved to: {PDF_OUT}")
        return

    # ── 3. Fallback: HTML ──────────────────────────────────────────────────────
    print("\nwebpdf failed. Falling back to HTML…")
    print("(To enable PDF export, run: pip install nbconvert[webpdf] && playwright install chromium)")
    rc = run([
        sys.executable, "-m", "nbconvert",
        "--to", "html",
        "--no-input",
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
