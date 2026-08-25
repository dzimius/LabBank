# LabBank

LabBank is a laboratory banking ALM system that connects risk methodology with a working data engineering stack. It generates a synthetic bank balance sheet, enriches it with market and behavioural assumptions, produces contractual and behavioural cash flows, and calculates IRRBB and liquidity metrics from the same auditable data foundation.

The project is designed for people who sit between ALM, risk analytics, finance, data engineering, and reporting. It is not a toy notebook and it is not a closed black-box model. The goal is to make the full ALM chain visible: from transaction-level inputs, through SQL-backed cash flows, to regulatory-style outputs and reporting.

## Quick start: two ways to use this project

**A) LabBank only — no SQL Server, no setup beyond Python.** Explore a pre-built synthetic bank's NII, EVE, EBA SOT, LCR, and NSFR interactively, stress the balance sheet, IRS book, and NMD behavioural assumptions, and see the impact in real time.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run sandbox\app.py
```

This is the fastest path to see what the project does. LabBank reads only local files (`optimize_prep/output/*.npz`, `balance_generate/input_data/bank_data.xlsx`, and a few other Excel inputs) — nothing here touches a database.

**B) Full ETL pipeline with SQL Server — generate your own balance sheet.** If you want to build a balance sheet from your own numbers (not just stress the shipped demo one), you'll need SQL Server as the integration/audit layer — see [Setup](#setup) below. Once it's running:

1. Edit `balance_generate/input_data/bank_data.xlsx` (and `ir_derivatives/input/irs_input.xlsx` for swaps) with your own data.
2. Run the `labbank_data_job` in Dagster (`dagster dev`, then launch this job) — it regenerates the full balance sheet, recomputes IRRBB/liquidity, and rebuilds the npz tensors LabBank reads, all in one step.
3. In the already-running Streamlit app, click **"🔄 Reload my data"** in the sidebar — LabBank now shows your balance sheet instead of the shipped demo one.

Path A and B share the same LabBank app; B just changes what data it's pointed at.

## What the project demonstrates

- End-to-end balance sheet generation for loans, deposits, financial instruments (including a T-bill product covering the HQLA short end), equity, cash accounts, and interest rate swaps.
- Cash-flow engine with contractual schedules and behavioural assumptions such as loan prepayment and non-maturity deposit decay.
- IRRBB calculations for NII, EVE, and EBA Supervisory Outlier Test scenarios.
- Liquidity risk calculations for LCR and NSFR.
- SQL Server as the integration and audit layer between modules.
- Dagster orchestration for reproducible pipeline runs and business-case jobs.
- Reporting through Excel, Jupyter/HTML, and Beamer assets.
- Optimisation preparation layer with vectorised tensors for future balance sheet optimisation work.

## Project idea

A bank balance sheet produces many future cash flows. ALM metrics such as NII, EVE, LCR, and NSFR should be calculated from a consistent source, otherwise reconciliation becomes the hardest part of the process.

LabBank follows one core principle:

> Every number in the final reports should trace back to transaction-level data and cash-flow rows stored in SQL.

This makes the project useful both as an ALM methodology sandbox and as a technical reference architecture for Python + SQL Server + Dagster based analytical pipelines.

## Visual overview

The repository already contains visual assets generated for the Beamer presentation. The most useful ones for a GitHub README are:

![Pipeline overview](visual_rep/beamer_assets/pipeline.png)

![SQL schema overview](visual_rep/beamer_assets/sql_schema.png)

Recommended additional README visuals:

- Export `visual_rep/beamer_assets/labbank_architecture.drawio` to `visual_rep/beamer_assets/labbank_architecture.png` and place it near the top of this README.
- Keep `visual_rep/beamer_assets/shock_curves.png` for the IRRBB methodology section.
- Keep `visual_rep/beamer_assets/nii_bridge.png`, `eve_results.png`, and `lcr_nsfr.png` as compact result examples.
- Use `visual_rep/beamer_assets/liq_gap.png` if you want to show behavioural liquidity risk visually.

## Documentation

- [`docs/01_setup_guide.md`](docs/01_setup_guide.md) — installing and running both quick-start paths.
- [`docs/02_methodology.md`](docs/02_methodology.md) — the ALM/IRRBB/liquidity methodology behind the calculations.
- [`docs/03_technical_notes.md`](docs/03_technical_notes.md) — candid, file:line-specific maintainer notes on architecture, known weak spots, and technical debt.

## Functional modules

| Area | Folder | Main responsibility |
| --- | --- | --- |
| Balance sheet generation | `balance_generate/` | Creates synthetic transactions and product-level balance sheet tables. |
| Market and behavioural enrichment | `balance_gen_add_data/` | Loads curves, fixings, schedules, and behavioural model parameters. |
| Cash-flow engine | `cash_flow_calc/` | Produces contractual and behavioural cash-flow schedules and repricing/liquidity gaps. |
| Interest rate derivatives | `ir_derivatives/` | Loads IRS positions, creates swap cash flows, and overlays the repricing gap. |
| IRRBB | `irrbb_calc/` | Calculates NII, EVE, EBA shock scenarios, and Supervisory Outlier Test outputs. |
| Liquidity risk | `liq_calc/` | Calculates LCR and NSFR from balance sheet and cash-flow data. |
| Optimisation preparation | `optimize_prep/` | Builds the fast approximation tensors LabBank reads, plus accuracy checks against the exact pipeline. |
| Interactive exploration (LabBank) | `sandbox/` | Streamlit app — stress the balance sheet, IRS book, and NMD assumptions and see NII/EVE/SOT/LCR/NSFR update live. No SQL Server required. |
| Orchestration | `dagster_pipeline/` | Defines Dagster assets and jobs over the workflow scripts. |
| Reporting | `visual_rep/` | Contains notebooks, HTML reports, Beamer presentation, and chart assets. |
| Documentation | `docs/` | Setup guide and ALM/architecture methodology write-ups — see [Documentation](#documentation) below. |
| 🔒 Balance sheet optimisation (private) | `bs_optimization/` | Four solvers (deterministic, joint BS+swap, stochastic, natural-hedge) that optimise the balance sheet under EVE/NII/LCR/NSFR constraints. Functional and fairly mature, but lives in a **private git submodule** — see "Optimisation preparation and balance sheet optimisation" further down this README. |

## Pipeline flow

1. Generate synthetic balance sheet transactions from Excel inputs and product assumptions.
2. Load market data, historical fixings, behavioural assumptions, and schedule tables.
3. Generate contractual and behavioural cash flows for all products.
4. Add interest rate swap cash flows and merge the IRS gap into the IRRBB repricing gap.
5. Build EBA shock curves and calculate NII over the 1-year horizon.
6. Calculate EVE as the present value of run-off cash flows under base and shocked curves.
7. Apply EBA SOT logic, including conservative treatment of gains.
8. Calculate LCR and NSFR.
9. Export Excel, notebook, HTML, and Beamer reporting outputs.

## ALM and risk methodology

### IRRBB

The IRRBB layer calculates:

- NII under base and shocked scenarios.
- EVE under run-off assumptions.
- EBA shock scenarios, including parallel up/down, steepener, flattener, short-rate up/down, and an own scenario.
- Supervisory Outlier Test outputs versus Tier 1 capital.
- Repricing gap effects before and after the IRS overlay.

The implementation separates the economic logic from the orchestration layer. Workflow scripts perform the calculations; Dagster wraps those scripts as assets so the pipeline can be rerun by business use case.

### Liquidity

The liquidity layer calculates:

- LCR as high-quality liquid assets divided by stressed 30-day net outflows.
- NSFR as available stable funding divided by required stable funding.
- Behavioural liquidity gaps using cash-flow schedules, prepayment, and deposit decay assumptions.

### Behavioural modelling

The project includes behavioural assumptions used in ALM practice:

- Loan prepayment assumptions.
- Non-maturity deposit decay for interest-rate and liquidity views.
- Product-level repricing logic.
- Fixed and floating rate product treatment.
- Interest rate swap overlay without changing the underlying balance sheet.

## Data model

SQL Server is used as the central integration layer. The modules write to dedicated schemas, including balance sheet, market data, schedules, cash flows, IRRBB, optimisation preparation, and result tables.

The important design choice is that downstream modules read from upstream SQL tables instead of passing hidden in-memory state. This makes runs easier to audit, rerun, and inspect.

The pipeline writes across nine schemas:

| Schema | Purpose | Example tables |
| --- | --- | --- |
| `dbo` | Landing zone for generated transactions | `transactions` |
| `schemat` | Product-level balance sheet | `loans`, `deposits`, `financial_instruments`, `equity`, `ir_swaps` |
| `mkt` | Market data | `curves`, `fixings` |
| `bs` | Behavioural model parameters | `models_loan`, `models_deposit_ir`, `models_deposit_liq` |
| `sched` | Schedule/leg-level detail | `ir_swaps` (swap leg schedules) |
| `cf` | Cash-flow engine output | `products`, `products_liq`, `ir_swap_orig`, `ir_swap_beh`, plus one dynamically-named `products_<scenario>` table per EBA shock scenario |
| `irrbb` | IRRBB calculation results | `ir_gap_orig`, `ir_gap_beh`, `ir_gap_beh_a`, `liq_gap_orig`, `liq_gap_beh`, `ir_swap_gap_beh`, `nii_results`, `eve_results`, `irrbb_report` |
| `results` | Final reporting tables | `irrbb_report`, `lcr_nsfr` |
| `opt_prep` | Fast-approximation tensors backing LabBank's sandbox | `monthly_curves`, `product_params`, `product_scenario_params` |

## Dagster orchestration

Dagster assets are defined in `dagster_pipeline/assets/`. The project includes several business-case jobs:

| Job | Purpose |
| --- | --- |
| `balance_sheet_job` | Generate the balance sheet, load enrichment data, and process IRS positions. |
| `full_run_job` | Run the complete ALM pipeline from balance sheet to IRRBB and liquidity outputs. |
| `irs_update_job` | Reload swaps and recompute IRRBB without rebuilding the balance sheet. |
| `irrbb_recalc_job` | Recompute NII, EVE, and SOT from existing cash flows after market curve changes. |
| `liq_only_job` | Refresh LCR and NSFR only. |
| `optimize_prep_job` | Rebuild fast approximation tensors and run accuracy checks. |
| `labbank_data_job` | Full pipeline + optimize_prep in one job. Run this after editing the input Excel files to regenerate your own balance sheet and refresh LabBank's data — see [Quick start, path B](#quick-start-two-ways-to-use-this-project). |

Run the Dagster UI from the repository root:

```powershell
dagster dev
```

The Dagster module is configured in `pyproject.toml`:

```toml
[tool.dagster]
module_name = "dagster_pipeline.definitions"
```

## Technology stack

- Python
- Streamlit and Plotly (interactive LabBank sandbox)
- pandas, NumPy, SciPy
- SQLAlchemy and pyodbc
- SQL Server with Windows trusted authentication
- QuantLib
- Dagster
- Dask (optional distributed backend for large stress-test batches)
- Jupyter
- matplotlib
- Excel inputs and outputs
- LaTeX Beamer

## Setup

This section covers the full ETL pipeline (path B). If you only want to run LabBank against the shipped demo balance sheet (path A), you don't need any of this — just `pip install -r requirements.txt` and `streamlit run sandbox/app.py`.

Create and activate a Python environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The current SQL connection strings use SQL Server trusted authentication and expect an available SQL Server database similar to:

```text
mssql+pyodbc://maciek_d/bank_gen?driver=ODBC+Driver+17+for+SQL+Server&Trusted_Connection=yes
```

Before running the pipeline on another machine, review the `sql_setup.py` files in the module folders and adjust:

- SQL Server name / instance.
- Database name.
- ODBC driver version.
- Authentication method.

## Running workflows directly

Most modules can be run as standalone workflow scripts, for example:

```powershell
python balance_generate\python_code\b_s_gen_workflow.py
python balance_gen_add_data\python_code\b_s_add_data_workflow.py
python cash_flow_calc\python_code\cf_calc_workflow.py
python ir_derivatives\python_code\irs_workflow.py
python irrbb_calc\python_code\nii_calc_workflow.py
python irrbb_calc\python_code\eve_calc_workflow.py
python irrbb_calc\python_code\eba_sot_workflow.py
python liq_calc\python_code\lcr_nsfr_workflow.py
```

For normal use, prefer Dagster because it preserves dependency order and gives a run history.

## Reporting outputs

The repository includes several reporting layers:

- `visual_rep/ALM_report.ipynb` for ALM and compliance reporting (run `python visual_rep/export_report.py` to generate `ALM_report.html`/`.pdf` — the export isn't tracked in git).
- `visual_rep/finance_report.ipynb` for finance-oriented reporting (same export pattern).
- `visual_rep/labbank_presentation.tex` for the Beamer presentation.
- `irrbb_calc/output/*.xlsx` for IRRBB outputs.
- `liq_calc/output/lcr_nsfr_report.xlsx` for liquidity outputs.
- `optimize_prep/output/*.xlsx` and `*.npz` for optimisation preparation and accuracy checking.

## Optimisation preparation and balance sheet optimisation (🔒 private)

The `optimize_prep/` module prepares vectorised data structures for fast metric approximation. It creates product parameter tensors and curve tensors, then checks approximate calculations against exact workflow outputs. These tensors are also exactly what LabBank (`sandbox/`) reads to run interactively without a database. `optimize_prep/` is fully public, part of this repo.

`bs_optimization/` builds on that layer with four solvers:

- A deterministic balance-sheet optimiser (economic-profit objective, optional soft EVE/NII breach penalty).
- A joint balance-sheet + free-swap-overlay optimiser.
- A stochastic optimiser (Monte Carlo scenario sampling).
- A natural-hedge optimiser that minimises EVE/NII breach severity without economic-profit optimisation.

This layer is functional and fairly mature, but it now lives in **`LabBank-Optimization`, a private git submodule** — cloning this repository gives you an empty `bs_optimization/` folder, not the code itself, since the submodule is not public. Reach out to the author for access. It is also not yet part of the guided LabBank path (Path A/B above); treat it as a preview of where Phase 3 is heading rather than a finished, documented feature.

## Repository status

This is a laboratory project, not production banking software. It is intended to show methodology, architecture, and implementation patterns for ALM analytics.

Some assumptions are intentionally simplified or synthetic:

- Input balance sheet data is generated for demonstration.
- Behavioural models are illustrative and should be calibrated before any real use.
- SQL connection settings are local and need adjustment per environment.
- Regulatory implementation should be independently validated before production use.

## Suggested next improvements

- Add an exported architecture PNG from the Draw.io source file.
- Move SQL connection settings to environment variables or a central config file.
- Reconcile the Quick Start section above with `docs/01_setup_guide.md` so the two don't drift out of sync.
- Expand tests beyond the fast-approximation layer (`optimize_prep/tests/`) to cover core IRRBB calculations.
- Add GitHub Actions checks for formatting and unit tests where SQL dependencies can be mocked.

## License and disclaimer

Licensed under the [MIT License](LICENSE).

This repository is an educational and analytical engineering project. It is not financial advice, not a regulatory submission engine, and not a production risk system without further validation, controls, documentation, and model governance.
