# LabBank Overleaf report

Self-contained folder — upload `main.tex` + `images/` as-is to a new Overleaf project.
This is a first-iteration plan, not a finished camera-ready document yet, but it
should compile cleanly on Overleaf as of this version.

## Status

- **`images/pipeline.png`** — done. Exported from the draw.io GUI (post-Dagster-bracket)
  and wired into both `\imgfull{pipeline.png}` calls (title page + Section 2).
- **`images/labbank_balance_sheet_annotated.png`** — done, and no longer an SVG. It was
  originally `.svg`, embedded via `\usepackage{svg}` + `\includesvg`, but that pipeline
  (Inkscape's LaTeX-overlay text export) rendered garbled/duplicated text on Overleaf —
  confirmed the source SVG itself was fine (renders cleanly in a browser), so the fault was
  purely in the Inkscape/shell-escape conversion step. Fixed by rasterising the SVG to PNG
  once (via a headless canvas render) and embedding it like every other figure with
  `\imgfull`; `\usepackage{svg}`, `\svgpath`, and the `\svgfull` macro were removed from
  `main.tex` entirely since this was their only use.
- **`images/labbank_balance_sheet_annotated.svg`/`.png` — updated for the sidebar removal
  in `sandbox/app.py`.** The "Reload my data" button moved from a left sidebar (now removed
  entirely) to the bottom of the Balance Sheet tab. The mockup (source of truth:
  `docs/images/labbank_balance_sheet_annotated.svg`) was redrawn to match: no sidebar column,
  content shifted to fill the reclaimed width, a divider + button added below the Composition
  charts, and callout ⑦ moved to point at it with updated legend text. Re-rasterised to PNG
  the same way as above and re-copied into `images/`.
- **New Section 3, "The LabBank App — Your Digital Twin, Interactively"** — walks through
  the Streamlit app: quick-start command, the annotated Balance Sheet tab, a one-line summary
  of the other five tabs, and a worked "change something, see what happens" example with two
  side-by-side before/after placeholder boxes.
- The framing across the doc (title page, Section 1) was reworded around the
  "digital twin you can test on" pitch rather than a pure architecture pitch.
- **`images/transaction_erd.png`** — done. New `make_transaction_erd()` in
  `visual_rep/make_beamer_assets.py` (matplotlib, no DB write, static schema layout) — an
  SSMS-style database diagram with `dbo.transactions` as the hub and 1:1 FK lines out to
  `schemat.loans/deposits/financial_instruments/equity`. Wired in under Section 5 ("Data
  Model"), as a new subsection distinct from the schema-level `sql_schema.png` flow diagram.
- **New Section 6 subsection, "How the balance sheet is generated"** — describes the
  truncated-normal balance draws per (product, client-type) bucket and the historical-fixing-
  by-vintage rate assignment (fixed-rate contracts lock the WIBOR fixing at their own start
  date; floating/administrative ones reprice off a recent fixing).
- **`images/repricing_gap.png`** — done. `make_repricing_gap()` already existed in
  `make_beamer_assets.py` but wasn't wired into `main.tex` even though the "Repricing gap &
  interest rate swaps" subsection describes it with zero figures; also fixed a pre-existing
  bug there (`ax.add_patch(mpatches.Patch(...))` is invalid — `Patch` is abstract, needed
  `ax.legend(handles=[...])`) that silently dropped part of the legend. Now embedded where
  the repricing gap is discussed (Section 6).
- **Balance Sheet \& Product Universe table** — the `bs_structure.png` stacked-bar chart was
  unreadable once the book had 9+ products per side (labels overlapping, tiny segments like
  "0000"/"1900"/"0300" with no room for a legible label). Replaced with two plain
  `booktabs` tables (Assets / Liabilities+Equity — product, B PLN, % of side, total row),
  matching the product-table style already used in `visual_rep/bank_report.html`. Figures
  pulled live from `dbo.transactions`/`schemat.equity` for the 2024-12-31 demo book.
  `bs_structure.png` removed from `images/` since it's no longer referenced.
- **NII section** — `nii_bridge.png` (locked-vs-renewal split for the base scenario only)
  replaced with `nii_scenarios.png` — `make_nii_scenarios()` already existed but wasn't
  wired in; it shows the same locked/renewal split (via hatching) for all 7 EBA scenarios at
  once, so it strictly subsumes what the bridge chart showed. `nii_bridge.png` removed from
  `images/`.
- **Supervisory Outlier Test subsection** — previously text/formula only, no figure. Added
  `eba_sot.png` (`make_eba_sot()`, already existed, wasn't wired in) — a tornado-style
  horizontal-bar chart per scenario with the −15%/−5% breach threshold lines drawn in, plus a
  `booktabs` summary table (ΔEVE_reg, ΔEVE_reg/T1, ΔNII_reg, ΔNII_reg/T1, breach flag per
  scenario) matching the SOT table already in `visual_rep/bank_report.ipynb`. Added
  `\label{sec:tryityourself}` to the "Try It Yourself" section so the table's footnote
  (pointing at the balance-sheet optimiser roadmap) can `\ref` it.

## Before publishing

1. **Replace the two `sqlplaceholder` boxes in Section 7** ("See It Yourself") with real
   SQL Server Management Studio / Azure Data Studio screenshots — a contract row + its
   `cf.products` cash-flow schedule, and the final `results.irrbb_report` / `results.lcr_nsfr` output.
2. **Replace the two `figplaceholder` before/after boxes in Section 3** ("Change something,
   see what happens") with real Metrics-tab screenshots — baseline vs. after the 10pp
   fixed→floating mortgage shift described there (or swap in whatever example you actually run).
3. **Fill in the contact placeholders** — email + LinkedIn — in the title page and the
   "Contact" section at the end of `main.tex` (search for `REPLACE_WITH_`).
4. Compile twice in Overleaf (pdflatex) so the table of contents and cross-references resolve.

## What's in it

`main.tex` is a ~14-section article (not the beamer deck) built for a general/LinkedIn
audience, pitched as a digital twin you can learn ALM by testing rather than a pure
architecture writeup: the pitch, architecture + pipeline diagram, the Streamlit app
walkthrough with a before/after worked example, Dagster orchestration, SQL schema diagram,
a full schedule-of-tables reference, live-query placeholders, balance sheet, cash flows,
IRRBB, liquidity, reporting, differentiators, and how to try it yourself. Content is pulled
from `docs/02_methodology.md` and `README.md`, condensed and reworded for a pitch document
rather than technical documentation. Section cross-references use `\label`/`\ref`, not
hardcoded numbers, so inserting/reordering sections won't silently break them.

Not done yet, for later: a 1-pager version (mentioned, deferred).
