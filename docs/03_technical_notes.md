# LabBank — Technical Notes

A code-level companion to the [Methodology](02_methodology.md) doc. That doc covers *what* LabBank calculates and the formulas behind it; this one covers *how the codebase is put together* — module responsibilities, the key classes and functions in each, the design patterns behind them, and the tradeoffs made along the way. Written for engineers reading the source, not just the results.

## 1. Architecture and data flow

```
Excel inputs → Python workflow scripts (orchestrated by Dagster) → SQL Server (9 schemas) → Reports
```

| Module | Writes to | Depends on |
| --- | --- | --- |
| `balance_generate/` | `dbo.transactions`, `schemat.*` | `bank_data.xlsx` |
| `balance_gen_add_data/` | `mkt.curves`, `mkt.fixings`, `bs.*` | market/behavioural Excel inputs |
| `cash_flow_calc/` | `cf.products`, `irrbb.ir_gap_*` | `schemat.*`, `bs.*` |
| `ir_derivatives/` | `cf.ir_swap_*`, merges into `irrbb.ir_gap_beh` | `irs_input.xlsx`, `mkt.curves` |
| `irrbb_calc/` | `irrbb.*`, `results.irrbb_report` | `cf.products`, `irrbb.ir_gap_beh` |
| `liq_calc/` | `results.lcr_nsfr` | `cf.products`, `schemat.*` |
| `optimize_prep/` | `optimize_prep/output/*.npz` | everything above (reads via SQL) |
| `sandbox/` (LabBank) | nothing (read-only) | `optimize_prep/output/*.npz`, `bank_data.xlsx` |
| `bs_optimization/` (private submodule — `LabBank-Optimization`) | `bs_optimization/output/*` | `optimize_prep/output/*.npz` |
| `dagster_pipeline/` | orchestration only | wraps all workflow scripts as subprocess calls (see §3) |

Every module can be run two ways: as a standalone `python <module>/python_code/<x>_workflow.py` script, or as a Dagster asset that wraps the same script. Both paths execute identical code — Dagster adds scheduling, dependency order, and run history on top, nothing more (see §3).

## 2. Code walkthrough, module by module

### `balance_generate/` — synthetic balance sheet generation

Built around an **Abstract Factory + Template Method** pair in `b_s_gen_objects.py`:

- `ProductGen` (abstract base, `:305`) — the template method `build_result_df()` (`:333`) calls `gen_set_of_transactions()` then `add_parameters()`, both left abstract for subclasses to fill in.
- One concrete subclass per product family — `LoansFixedGen`, `LoansFloatGen`, `BondsFixedGen`, `BondsFloatGen`, `SavingAccountsGen`, `CurrentAccountsGen`, `TermDepositsGen` (`:505, :645, :787, :862, :344, :393, :445`).
- `ProductFactory` (`:1416`) — a `class_registry` dict maps each product name (`"mortgage_fixed"`, `"t_bill"`, …) to its generator class, and a `table_registry` maps the class to its destination SQL table. Adding a new product type means adding one dict entry plus a new `ProductGen` subclass, not touching the workflow script.

Transaction sizing uses `generate_truncated_normal()` (`:80`), built on `scipy.stats.truncnorm`, for both balances *and* margins/rates — `generate_balances()` (`:98`) samples individual transaction sizes, then `cumsum` + `searchsorted` snaps the running total to a target book size with a leftover "plug" transaction absorbing the remainder.

One detail worth calling out: origination dates aren't sampled uniformly. `generate_random_dates()`/`generate_random_bond_dates()` (`:148, :175`) weight business days by `(1 + annual_growth)^years_since_start`, so a book grown synthetically over N years skews toward more recent originations — closer to how a real growing bank's vintage distribution looks than a flat random draw would.

### `cash_flow_calc/` — the cash-flow engine

Deliberately functional rather than class-based — `cf_calc_objects.py` is a set of pure(ish) functions operating on DataFrames:

- `gen_orgin_sched_loan_fin_inst()` (`:310`) — builds the QuantLib `ql.Schedule` for loans/bonds from the product's `disc_curve`/`fwd_curve`, day-count convention (via `get_dc_conv_from_str`, `:126`), and business-day convention.
- `gen_deposit_sched()` (`:423`) — single-row schedule for deposits/NMDs; infers the forward rate directly from the discount factor (`fwd_rt = (1/d_f − 1) / year_fraction`) rather than a separate curve lookup, since these products have no amortization schedule to project.
- `compute_amort_schedule_vectorized()` (`:698`) — the amortization engine, with an `exact` flag: `exact=True` runs a per-schedule recursive cumulative product (`groupby.apply`) for true annuity behaviour; `exact=False` swaps in a closed-form numpy approximation when scenario-count speed matters more than exactness.
- `exact_annuity_loop()` (`:1034`) computes the annuity payment from the *actual* per-period forward curve rather than a flat-rate assumption: `g_k = 1 + fwd_rate_k · year_fraction_k`, `payment = balance / Σ (1/g_k accumulated)`.

Contractual vs. behavioural cash flows aren't a class split — `merge_cf_orig_beh()` (`:54`) stitches the QuantLib-driven "origin" (contractual) schedule together with behavioural overlays (prepayment, NMD decay) into one unified `cf.products` row set, tagged so downstream consumers can select either view.

### `irrbb_calc/` — NII, EVE, and the EBA shock scenarios

Also functional. `compute_nii()` (`nii_calc_objects.py:71`) derives NII sensitivity from the repricing gap (`ir_gap_beh`), with and without the IRS overlay, with and without the administrative-rate floor on current accounts. `compute_eve_base()`/`compute_eve_shocked()` (`eve_calc_objects.py:149, :198`) present-value the behavioural cash flows against base vs. shocked discount factors, sign-adjusted by `bs_side`.

The shock mechanics live in `eba_shock_curves.py`:

- `shock_bps_at_tenors()` (`:128`) implements the EBA/RTS 2022/10 Article 3 shock shapes — parallel, steepener/flattener use `exp(−t)` / `(1 − exp(−t))` tenor-decay functions of the currency-specific basis-point parameters in `SHOCK_PARAMS_BPS` (`:45`).
- `apply_shock_to_disc_curve()` (`:241`) converts a discount curve to continuous zero rates, applies the tenor-shaped Δr(t), floors at 0%, and re-discounts — this shocked curve then flows straight into `compute_nii_shocked`/`compute_eve_shocked`.

One implementation detail worth knowing about if you're extending the Monte Carlo layer: an optional `base_floor_bps_fn` pre-shock floor (`:216`, `default_realistic_base_floor_bps`) exists purely to stop randomly-simulated curves from pinning at an implausible flat long end — it's explicitly *not* an EBA-defined figure, just a numerical guard for synthetic curve paths.

### `optimize_prep/` — the fast-approximation tensor layer

This is the part of the codebase most worth understanding well, because it's what lets the Streamlit sandbox run interactively with no database: every balance-sheet edit re-prices NII/EVE/LCR/NSFR in milliseconds against precomputed tensors instead of re-running the full SQL + QuantLib pipeline.

- `BalanceSheetParams` (`bs_vector.py:35`, a frozen dataclass) — one row per cohort, holding parallel numpy arrays: unit rates, `d_mod` (modified duration), `delta_nii_unit`/`delta_eve_unit` per EBA scenario, and the LCR/NSFR/RWA/expected-loss factors needed for optimisation.
- `CurveTensors` / `CohortRates` (`bs_vector.py:299, :357`) — precomputed discount/forward curve grids and a per-cohort `rate_matrix[n_cohorts, 12, n_scenarios]`.
- `build_product_params()` (`extract_params.py:1882`) — the SQL → tensor ETL. Loads `bs_structure`, per-cohort cash-flow statistics, the substitution matrix, and rate coefficients, then asserts every `product_code` present in the balance sheet is covered by one of three hand-maintained registries (`COHORT_PRODUCT_CODES`, `SINGLE_ROW_PRODUCT_CODES`, `IRS_PRODUCT_CODES`) — a new product that's missing from all three fails the build loudly instead of silently vanishing from the tensors. `diagnose_accuracy.py` and `cohort_cf_drill.py` import these same three sets rather than keeping their own copies, so there's one place to update when a product is added.

The approximation itself, in `nii_eve_cf_fast.py`:

- **NII** is a single `einsum` (`_nii_cohort_matrix`, `:55`): `NII[cohort, scenario] = balance · sign · Σ_month (outstanding[cohort, month] · rate_matrix[cohort, month, scenario]) / 12` — O(cohorts × 12 × scenarios), microseconds for a book with under ~2,000 cohorts.
- **EVE** offers two modes: a duration-based shortcut (`−sign · modified_duration · Δforward_rate`) for speed, or a full cash-flow-discounting mode for accuracy.
- Both are corrected with a precalibrated additive bias term (`bias_nii`/`bias_eve`) — the gap between the fast approximation and the exact engine at the *base* scenario, assumed constant under shocked scenarios. That's a deliberate speed/accuracy tradeoff, not an oversight: `optimize_prep/output/approx_accuracy_report.xlsx` quantifies exactly how close the approximation stays across the full 7-scenario set, and `bias_store.py` degrades gracefully (recomputes if the cached bias's cohort count doesn't match the current tensors — see the cache-staleness note in §4).

In short: this layer trades a one-time calibration cost for O(1) scenario evaluation — a precomputed linear-operator cache standing in for SQL + QuantLib repricing.

### `sandbox/` — the LabBank Streamlit app

`app.py` (~1,400 lines) renders six tabs via `st.tabs()`: Balance Sheet, IRS Book, NMD Stress, ALM Metrics, Gap Analysis, Market Curves. Every edit re-runs the fast-approximation math from `optimize_prep/` (previous section) in-process against `BalanceSheetParams`/`CurveTensors` loaded once from `.npz` files — no SQL round-trip, which is what makes the interaction feel instant.

One quirk worth knowing if you're reading the source top-to-bottom: the Market Curves tab's *body* executes early in script order — before ALM Metrics — even though it renders last on the page, because its "hypothetical scenario" selectors need to resolve before the Metrics tab's NII/EVE recompute can use them. Streamlit re-runs the whole script top-to-bottom on every interaction, so this is a documented ordering workaround, not a bug.

### `dagster_pipeline/` — orchestration

Six asset files under `dagster_pipeline/assets/` (`balance_sheet.py`, `cash_flows.py`, `ir_derivatives.py`, `irrbb.py`, `liquidity.py`, `optimize_prep.py`). Each `@asset`-decorated function is a thin wrapper — e.g. `balance_transactions` (`assets/balance_sheet.py:14`) — calling `run_workflow(context, script_path)` (`runner.py:12`), which `subprocess.Popen`'s the *same standalone workflow script* used outside Dagster and streams its stdout into Dagster's logger. `deps=[...]` declarations sequence *when* assets run relative to each other; the actual data handoff between stages happens through SQL Server, not through Dagster's own type system. `jobs.py` composes these assets into the business-case jobs listed in the README (`balance_sheet_job`, `full_run_job`, `labbank_data_job`, …) via `AssetSelection`.

This is a deliberate tradeoff, not a gap: keeping every workflow script independently runnable (`python <script>.py`, no Dagster required) was a higher priority than a tighter Dagster-native `IOManager` contract between stages. The cost is that a stage silently writing a differently-shaped table only surfaces as a downstream error, not a Dagster-visible schema violation — worth knowing if you're debugging an unexpected `KeyError` two stages downstream of where the actual change happened.

### `bs_optimization/` — balance sheet optimiser (private submodule)

Briefly, since the code itself isn't public: `bs_optimizer.py` formulates balance-sheet reweighting as a genuine Linear Program — the economic-profit objective and every EVE/NII-delta/LCR/NSFR constraint are linear in the product weights, because they're built from the precomputed unit-deltas in `BalanceSheetParams` (§2, `optimize_prep/`). It's solved via `scipy.optimize.linprog(method="highs")`, with a `scipy.optimize.minimize(method="SLSQP")` fallback for the rarer nonlinear case (price-volume elasticity). Regulatory floors (EVE/NII/NSFR/Tier-1-RWA) are hard inequality constraints by default, or move into a weighted-slack objective in "soft breach" mode — see the [project report](https://github.com/dzimius/LabBank/releases/download/v1/LabBank.pdf) for the full methodology and results.

## 3. Design decisions and rationale

**SQL Server as the integration bus, not an implementation detail.** Every module writes its output to SQL rather than passing objects in memory between stages. This was a deliberate choice, not a performance-neutral default: it means any stage can be re-run independently, and the entire pipeline state at any point is inspectable with a plain `SELECT` — see [Setup guide, Path B](01_setup_guide.md#path-b--full-etl-pipeline-with-sql-server-generate-your-own-balance-sheet) for querying it directly. The cost is the subprocess/SQL round-trip latency that made the fast-approximation tensor layer (§2, `optimize_prep/`) necessary for anything interactive.

**Dagster wraps scripts, it doesn't replace them.** See §2's `dagster_pipeline/` note — every workflow stays runnable standalone. Orchestration is additive, never a hard dependency.

**Two representations of the same math.** The exact engine (SQL + QuantLib, `cash_flow_calc/`/`irrbb_calc/`) and the fast approximation (`optimize_prep/`) aren't redundant — they serve different jobs. The exact engine is the source of truth and what every report is built from; the fast layer exists purely so the sandbox and the optimiser can explore thousands of what-if balance sheets per second, calibrated against the exact engine rather than replacing it.

## 4. Known limitations and scope

Honest caveats about where this project simplifies, so they're not mistaken for oversights:

- **Prepayment is a static product×tenor lookup, not rate-dependent.** The CPR assigned to a cohort at cash-flow build time is reused unchanged across every EBA shock scenario — a par-down shock gets the same prepayment cash flows as a par-up shock. This means negative convexity (prepayment speeding up as rates fall) isn't modelled, which understates EVE sensitivity for fixed-rate mortgages in down-shock scenarios. A real fix would be a `CPR(Δr)` function; worth knowing if you're evaluating this project's rigor for a mortgage-heavy book specifically.
- **No FX conversion layer**, despite per-currency curve/calendar scaffolding throughout the codebase. Currently a no-op because the synthetic balance sheet is 100% PLN — a multi-currency book would need an `exchange_rate` table and a conversion step before aggregation.
- **Risk parameters (RWA weights, PD/LGD, LCR/ASF/RSF factors) are illustrative**, loosely shaped like real Basel/CRR and LCR/NSFR categories but not a precise regulatory calibration — see the product tables in the [Methodology doc](02_methodology.md#products-on-the-balance-sheet). They're plain Excel columns, meant to be replaced with your own institution's numbers.
- **The fast-approximation layer's bias correction is calibrated once at the base scenario and held constant across shocks** (§2) — a documented speed/accuracy tradeoff, quantified in `optimize_prep/output/approx_accuracy_report.xlsx` rather than left unmeasured.
- **Global constants (`TOTAL_ASSETS`, `REPORT_DATE`, the SQL connection string) are duplicated per module** rather than imported from one shared config — a straightforward centralization that just hasn't been prioritized yet, since every module needs to stay independently runnable (§3).

Behavioural models and the balance sheet itself are synthetic/illustrative by design — this project demonstrates *how* an ALM pipeline should be built and how its metrics relate to each other, not a validated, governed production risk system.

## 5. Extending the project

**Adding a new product** (e.g. credit cards, a corporate working-capital/investment-loan split, central bank cash): add a row to `bank_data.xlsx`'s `bs_structure` sheet, add a matching `ProductGen` subclass + `ProductFactory` registry entry in `balance_generate/python_code/b_s_gen_objects.py` (§2), and add the new product code to whichever of `COHORT_PRODUCT_CODES` / `SINGLE_ROW_PRODUCT_CODES` / `IRS_PRODUCT_CODES` it belongs to in `optimize_prep/python_code/extract_params.py`. The coverage assertion in `build_product_params()` (§2) will fail loudly if that last step is missed, instead of the product silently disappearing from the fast-approximation tensors.

**Adding an EBA scenario:** extend `SHOCK_PARAMS_BPS` in `irrbb_calc/python_code/eba_shock_curves.py` (§2) with the new scenario's tenor-shape parameters — the shock application, NII/EVE recompute, and SOT logic are all scenario-agnostic and pick it up automatically.

**Running just one stage after a change:** use the narrower Dagster jobs (`irrbb_recalc_job`, `liq_only_job`, …) rather than `full_run_job` — see the [README's job table](https://github.com/dzimius/LabBank#dagster-orchestration) for which one matches your change.
