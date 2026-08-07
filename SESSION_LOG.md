# Session Log

Running log of what Claude did in each work session on the GitHub-publish plan. Newest entry on top.

---

## 2026-07-26 — Session 1: hygiene, labbank_data_job, reload button, tracking gap found

**Done:**
- `.gitignore`: added `.dagster_home/`, `.tmp_dagster_home_*/`, and the copyrighted De Gruyter PDF (never to be committed).
- Added `LICENSE` (MIT).
- Grepped all `sql_setup.py` files for secrets — none found; all use Windows Trusted Connection, just a hardcoded local hostname (`maciek_d`) that the setup doc will need to flag as "change this to your instance."
- Added `labbank_data_job` to `dagster_pipeline/jobs.py` — chains the full ALM pipeline (`full_run_job`'s assets) plus `optimize_prep_tensors` in one job, so regenerating a custom balance sheet and refreshing LabBank's npz tensors is one action instead of two. Verified it registers correctly via `Definitions`.
- Added a "🔄 Reload my data" sidebar button to `sandbox/app.py` (the LabBank Streamlit app) that clears `st.cache_resource`/`st.cache_data` and reruns — needed because the data loaders in `sandbox/baseline.py` cache with no file-mtime check, so a plain browser refresh doesn't pick up regenerated npz files. Verified live in the browser: app ran with zero SQL Server reachable, all tabs/metrics computed correctly, reload button cleared caches and re-rendered without error.
- Added `.claude/launch.json` so the LabBank app can be launched for preview/testing going forward.

**Found (not yet resolved):** the repo's last commit (`e5298f2 "before dagster"`, 2026-05-30) predates the entire Dagster orchestration layer, the Streamlit LabBank app, the current README, `pyproject.toml`, and the whole `bs_optimization/` module — all of that is currently uncommitted working-tree state. Staged (`git add`, not committed) the sandbox app, `.gitignore`, `LICENSE`, `.claude/launch.json`, and `dagster_pipeline/jobs.py`. Did **not** commit — need to check with the user how they want this backlog split into commits.

**Next up:** README rewrite (two clear paths: LabBank-only vs full ETL), then the three docs.

**Resolved:** user chose "one big commit now" for the untracked backlog. Committed everything in `cf357a3` — Dagster pipeline, LabBank Streamlit app, bs_optimization, README, pyproject.toml, plus this session's labbank_data_job/reload-button/hygiene work. 134 files, repo now actually reflects working state for the first time since the "before dagster" commit.

**Also done:** rewrote `README.md` — added a "Quick start: two ways to use this project" section up top (path A: LabBank-only, no SQL Server; path B: full ETL + `labbank_data_job` + reload button for your own balance sheet), added `bs_optimization/` to the functional modules table badged 🚧 roadmap, added `labbank_data_job` to the Dagster jobs table, linked the new LICENSE. Not yet committed (small, will bundle with next batch of changes unless told otherwise).

**Docs (user decision: Markdown as source of truth, HTML/PDF as the final polished output):**
- Wrote `docs/01_setup_guide.md` — Path A (LabBank only) and Path B (full ETL + the edit-Excel/labbank_data_job/reload-button loop), with troubleshooting tables for both.
- Wrote `docs/02_methodology.md` — condensed from `visual_rep/labbank_presentation.tex` (architecture, pipeline steps, cash flow engine, behavioural modelling, repricing gap, NII/EVE, EBA shock construction, SOT formula, LCR/NSFR, behavioural liquidity gap, bs_optimization roadmap note).
- Added `docs/build_docs.py` — converts the markdown docs to styled standalone HTML ("paper" look, light/dark aware, print-friendly). Requires `pip install markdown` (one-off doc-build tool, deliberately not added to requirements.txt since it's not a pipeline runtime dependency). Ran it — `01_setup_guide.html` and `02_methodology.html` generated and spot-checked in browser.
- `docs/03_technical_notes.md` — written from the Explore agent's findings (spot-checked several of its strongest claims by grep before trusting them; all confirmed accurate, one even found an extra instance the agent missed). Covers: architecture/data-flow quick reference, portability blockers (hardcoded absolute Windows paths in 8 workflow scripts, `TOTAL_ASSETS`/`REPORT_DATE` duplicated 4-10x with no sync check), weak quantitative assumptions (static non-rate-dependent prepayment is the standout — no convexity in EVE/NII shocks; no FX layer despite scaffolding; three inconsistent silent-exception-swallowing idioms), dead code (`build_irrbb_report` confirmed unused via grep, 6 orphaned optimize_prep scripts, a stale already-applied migration script), fragile coupling (no assertion that `bs_structure`'s product codes are covered by the hardcoded cohort/single-row/IRS sets — **flagged as fix-before-adding-new-products**, directly relevant to the credit-card/loan-split/T-bill backlog), plus a section cross-referencing already-fixed bugs and already-known tech debt so nothing gets "rediscovered." Ends with a suggested priority order.

All three docs' HTML rebuilt via `docs/build_docs.py` and spot-checked in-browser (tables render correctly).

**Session complete — all 8 planned points done.**

## 2026-07-26 — Session 2: line-count audit, committed docs, stopped tracking report HTML

User asked why the repo showed ~73k lines vs ~23k earlier and suspected copy-paste. Audited: hashed every `.py` file (no real duplicates, just two empty `__init__.py`), then broke down line counts by file type. Root cause: `bs_optimization/notebooks/` (37,284 lines) and `visual_rep/`'s report HTML/ipynb (21,905 lines) were untracked before last session's consolidation commit and got swept in — almost all of it rendered Plotly/Jupyter HTML output, not hand-written code. Real Python: 33,401 lines across 99 files. Gave the user a full folder/subfolder map with per-directory line counts.

User decided: keep `.ipynb` tracked, stop tracking the notebook-exported `.html` reports (they duplicate the notebook's own content and can be regenerated on demand). Untracked (`git rm --cached`, kept on disk) `visual_rep/{bank_report,finance_report}.html` and `bs_optimization/notebooks/optimization_report{,_clean}.html`; added them to `.gitignore` with a comment explaining hand-authored HTML (`docs/*.html`, `optimize_prep/method_b_explainer.html`) is intentionally exempt. Updated README's references to these files to point at the regeneration commands (`export_report.py`, `export_optimization_report.py`, or plain `nbconvert`) instead of implying the HTML ships in the repo.

Committed everything from session 1 that was still pending (README quick-start rewrite, all 3 docs + build script, the gitignore/tracking cleanup) in `7390851` — 15 files, +1,165/−45,931 lines (the deletion is almost entirely the untracked report HTML).

## 2026-07-26 — Session 3: methodology review feedback

User ran the docs and gave 6 concrete corrections/additions. Verified each against the code before writing (didn't just take the request at face value):

1. Setup guide's job table listed `optimize_prep_job` even though the guide's whole point is reaching LabBank via `labbank_data_job` (which already includes it) — removed that row, pointed to README for the full job list.
2. Methodology said prepayment (CPR) was "a constant rate" — checked `cf_calc_workflow.py:132-136` and `b_s_add_data_objects.py:78-80`: it's actually set per (product, tenor) in `loan_beh_models.xlsx`, reloaded every pipeline run, so fully editable — but it does NOT respond to the rate shock itself within a scenario (confirmed via the `_apply_calibrated...`/cpr_rate-constant-per-schedule logic already flagged in the technical notes). Rewrote to state both things precisely instead of just "constant."
3. Checked whether non-current-account deposits also floor at 0% — traced `_apply_rt_limits` in `nii_calc_objects.py` back to `FLOORS_MAP`, sourced from `interest_rt.xlsx`'s `client_floor` column. Confirmed by reading the actual demo file: savings account (8000) and term deposit (7060) both have `client_floor=0.0` set, same as current accounts — so yes, implemented, but it's a per-product Excel setting, not a hardcoded blanket rule. Documented the mechanism and which products currently have it set.
4. Added a full "Products on the balance sheet" section — pulled the real 14-product + equity table directly from `bank_data.xlsx`'s `bs_structure` sheet (rate type, maturity, amortising, payment frequency, reset tenor), plus a note on the other columns (rwa/PD/LGD, HQLA/ASF/RSF, optimizer-only fields) and on 7900 being the one code used on both balance-sheet sides.
5. Added an IRS setup/mechanics section — `irs_input.xlsx` column reference, the demo book's actual composition (9 receive-fixed swaps, 4.5%, 40-50M PLN each), and how NII/EVE/gap-placement are computed for a swap (pulled from `sandbox/irs_engine.py`'s documented conventions).
6. Added an explicit NII formula (locked + renewal components) alongside the EVE/LCR/NSFR ones already in the doc.

Also caught and gitignored an Excel lock file (`~$loan_beh_models.xlsx`) that had been sitting untracked since before this whole effort started. Rebuilt HTML, spot-checked the new product table renders. Committed as `3e74f7c`.

## 2026-07-26 — Session 4: methodology deep-dive (product types, SQL data model, LabBank walkthrough)

User asked for a bigger expansion: product section should describe available product TYPES (with typical/illustrative risk parameters) before showing the shipped demo's concrete instances, plus SQL table descriptions and a LabBank UI walkthrough with an annotated screen, plus pipeline diagrams in both docs.

- Reused the project's existing `visual_rep/beamer_assets/pipeline.png` and `sql_schema.png` (already well-made, no need to draw new ones) — embedded pipeline.png in both the setup guide and methodology, and sql_schema.png in a new "Data model — SQL schema reference" section covering all 9 schemas (dbo, schemat, sched, mkt, bs, cf, irrbb, results, opt_prep) with what's in each and where to start exploring via SELECT.
- Restructured the product section: added "Available product types" (asset types: mortgage/consumer loan/SME loan/gov bond/cash/interbank, with RWA and PD/LGD; liability types: interbank deposit/current account/savings/term deposit/issued bond/equity, with LCR outflow rate and ASF) with an explicit caveat that these are illustrative, regulatory-inspired but not precisely calibrated, fully-editable Excel values -- not the real thing. The existing concrete demo table now follows as "The shipped demo instance."
- Added an IRS setup/mechanics section in the previous session; this session added the LabBank walkthrough covering all 6 tabs (Balance Sheet, IRS Book, NMD Stress, Metrics, Gap Analysis, Market Curves) with descriptions of what's editable/shown in each.
- For "screen with marked elements" specifically: no tool in this environment saves a live browser screenshot to a file, so instead hand-built an annotated SVG wireframe (`docs/images/labbank_balance_sheet_annotated.svg`) of the Balance Sheet tab -- numbered callouts for the tab bar, Total Assets input, Reset button, editable New% column, sum validation, composition chart, and the Reload-my-data sidebar button, with a legend explaining each. Verified it renders correctly in-browser before embedding. This is a diagram, not a real screenshot -- said so explicitly rather than implying otherwise.

Verified all embedded images (2 PNGs, 1 SVG) resolve correctly via JS naturalWidth/complete checks in both built HTML docs. Not yet committed -- ready whenever asked.

## 2026-07-26/27 — Session 5: trash cleanup, doc corrections, refreshed reports

User manually removed `balance_gen_add_data/dziala.csv`, `balance_generate/plan_projektu_bank.pptx`, `balance_generate/schema.xlsx`, `visual_rep/ir_gap.pbix`. Asked me to check for more orphans. Cross-referenced every PNG and suspicious xlsx/db file against actual code usage (not just "looks odd") and found 3 more genuine orphans: `cash_flow_calc/python_code/test.xlsx` (unreferenced anywhere), `irrbb_calc/output/irrbb_results.db` (unreferenced, old SQLite-era leftover), `liq_calc/output/irrbb_report.xlsx` (confirmed stale -- `lcr_nsfr_workflow.py:12` has its own comment saying this file is written by irrbb_calc now, not here). User confirmed via later message -> deleted all three. Also flagged (not touched, low priority) that `labbank_presentation.tex` references 3 images that don't exist (`labbank_architecture.png`, `labbank_sql_schema.png` -- only `.drawio` sources exist -- and `repricing_gap.png`, no source at all).

Fixed two doc gaps user caught: setup guide's "generate your own" loop didn't mention `interest_rt.xlsx` (where rate coefficients/floors/caps live) -- added. Removed all Power BI/`.pbix` references from README (5 spots) and methodology doc (1 spot) to match the file's removal.

Discussed LaTeX/Beamer vs HTML for docs (user's question, no action) -- clarified that compiled PDFs need no reader-side LaTeX install, only compile-side; recommended keeping HTML for the living/frequently-edited docs and saving LaTeX/Beamer for a future one-time "seller material" one-pager, since that pipeline (`make_beamer_assets.py` + `labbank_presentation.tex`) already exists in the repo.

User asked whether `bank_report.ipynb`/`finance_report.ipynb` were up to date -- checked mtimes (April, pre-Dagster) and cross-referenced every SQL table they reference against the current schema (all still match) -- concluded stale output but likely-fine code, recommended re-running. User said to run it now: confirmed SQL Server was up and fully populated (138k transactions, `cf.products`, `irrbb.*`, `results.lcr_nsfr` all populated), ran `export_report.py` and `make_finance_report.py` -- both notebooks re-executed cleanly with no errors, HTML re-rendered (untracked per the earlier gitignore rule), and the fresh numbers (NII ~614-650M, EVE ~2174M) match what LabBank showed earlier in the session -- good cross-check that everything is internally consistent.

Nothing committed yet this session -- pending: 6 file deletions, README/doc edits, the 2 refreshed notebooks, the new SVG diagram.

## 2026-07-27 — Session 6: added Economic Profit to the finance report

User asked whether Economic Profit (EP) -- the metric `bs_optimization/` optimizes -- could be added to `finance_report.ipynb`, both total and per-product. Investigated feasibility first rather than assuming: traced the exact EP formula from `bs_optimizer.py` (`EP = Margin-over-FTP + Fee - EL - CoC - OpEx`, deliberately excluding AcqCost since that's a growth-only cost, not a steady-state one) and confirmed which components live where -- PD/LGD/RWA/fee/CoC-rate/CET1-target all already sit in `optimize_prep/output/product_params.npz` (built from `bank_data.xlsx`, not in SQL), and the FTP rate needed for Margin lives in `ftp_rates.npz` (confirmed populated: 491/500 cohorts, ~4.7% avg). Verified sign conventions live before writing any code (balance_arr unsigned, nii_unit_rate pre-signed by side, rwa_factor/el_unit correctly zero for liabilities) rather than guessing.

Important structural finding: `finance_report.ipynb` is not hand-edited -- `make_finance_report.py` regenerates it from scratch every run from `CELL_*` source strings, so hand-editing the `.ipynb` directly would've been silently overwritten next run. Added `CELL_EP_MD`/`CELL_EP` to the generator instead, which load `product_params.npz` + `ftp_rates.npz` directly (same hybrid SQL+npz pattern LabBank's `sandbox/baseline.py` already uses, not a new architecture), compute Margin/Fee/EL/CoC/OpEx per cohort, aggregate to per-product-code, and print + chart both the whole-book total and the per-product breakdown. Added a TOC row and an explanatory markdown cell up front stating the data-source distinction and the AcqCost exclusion explicitly.

Ran it: whole-book EP = +75.7M PLN, corrected to +76 bps of total assets (initially computed bps against combined assets+liabilities+equity balance ~20.8B, which doesn't match how NIM is expressed elsewhere in the same report -- caught this during review and fixed the denominator to total assets for consistency, re-ran to confirm). Per-product spread from Consumer Loan Float at +733bps (strongest) down to Interbank/Other liability side at -630bps and Savings Account at -416bps (weakest) -- a genuinely different ranking than the raw-margin-only "Product Margin Analysis" section already in the report, which is the point of a fully-loaded EP view. 22 cells now (was 20), executed clean, HTML re-exported.

User asked for a waterfall/bar chart of the book-level EP bridge too (Margin -> Fee -> -EL -> -CoC -> -OpEx -> EP), matching the style of the existing NII waterfall in section 1. Added it as a third chart in the EP cell. Hit one bug on the first run: used single-backslash `\n` in the new bar labels, which the OUTER file's own triple-quoted string parsing collapsed into a real newline before the notebook cell source was even written, producing an unterminated-string SyntaxError when the notebook's kernel tried to parse it -- fixed by escaping to `\\n` (matching the convention already used elsewhere in this generator file for cells that need literal `\n` in their embedded source). Re-ran, executed clean, verified the new image's dimensions match the requested figsize exactly.
