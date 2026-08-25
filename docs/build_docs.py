"""build_docs.py
================
Converts the Markdown docs in this folder into standalone, styled HTML
"paper" versions. Markdown is the source of truth (edit the .md files);
run this script to regenerate the .html files after editing.

Requires the `markdown` package (not a runtime dependency of the ALM
pipeline itself, so it's not in requirements.txt):

    pip install markdown

Usage:
    python docs/build_docs.py
"""
from __future__ import annotations

import os
import markdown

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_FILES = [
    ("01_setup_guide.md", "LabBank — Setup Guide"),
    ("02_methodology.md", "LabBank — Methodology"),
    ("03_technical_notes.md", "LabBank — Technical Notes"),
]

CSS = """
:root {
  color-scheme: light dark;
}
body {
  max-width: 840px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem;
  font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #1a1a1a;
  background: #fff;
}
@media (prefers-color-scheme: dark) {
  body { color: #e8e8e8; background: #14161a; }
  a { color: #7cb0ff; }
  code, pre { background: #1e2128 !important; color: #e8e8e8; }
  table th { background: #22262e !important; }
  table, th, td { border-color: #383c44 !important; }
  blockquote { border-left-color: #444 !important; color: #b8b8b8 !important; }
  hr { border-color: #383c44 !important; }
}
h1 { font-size: 2rem; border-bottom: 3px solid #2c5f8a; padding-bottom: 0.4rem; }
h2 { font-size: 1.4rem; margin-top: 2.2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.3rem; }
h3 { font-size: 1.15rem; margin-top: 1.6rem; }
code { background: #f2f2f2; padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; }
pre { background: #f2f2f2; padding: 1rem; border-radius: 8px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.95em; }
th, td { border: 1px solid #ccc; padding: 0.5rem 0.7rem; text-align: left; }
th { background: #f2f2f2; }
blockquote { border-left: 4px solid #2c5f8a; margin: 1rem 0; padding: 0.3rem 1rem; color: #444; font-style: italic; }
.doc-header { color: #666; font-size: 0.9rem; margin-bottom: 2rem; }
hr { border: none; border-top: 1px solid #ddd; margin: 2.5rem 0; }
@media print {
  body { max-width: 100%; padding: 0 1cm; }
  a { color: inherit; text-decoration: none; }
}
"""


def build(md_filename: str, title: str) -> None:
    md_path = os.path.join(DOCS_DIR, md_filename)
    html_path = os.path.join(DOCS_DIR, md_filename.replace(".md", ".html"))

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="doc-header">LabBank &mdash; bank_project</div>
{body}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {html_path}")


if __name__ == "__main__":
    for fname, title in DOC_FILES:
        if os.path.exists(os.path.join(DOCS_DIR, fname)):
            build(fname, title)
