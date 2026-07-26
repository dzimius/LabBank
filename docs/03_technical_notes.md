# LabBank — Technical Notes (Internal)

For the maintainer, not written for a public audience. Candid, file:line-specific, ranked by how much it matters rather than how easy it is to fix. Findings below were verified against the code (spot-checked file:line references), not guessed.

## 1. Architecture and data flow, quick reference

See [Methodology](02_methodology.md) for the full explanation. For maintenance purposes, the short version:

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
| `bs_optimization/` | `bs_optimization/output/*` | `optimize_prep/output/*.npz` |
| `dagster_pipeline/` | orchestration only | wraps all workflow scripts as subprocess calls (see §4, finding 16) |

Nine SQL schemas in total: `dbo`, `schemat`, `mkt`, `bs`, `cf`, `irrbb`, `results`, `opt_prep`, plus whatever `liq_calc` uses for its own tables.

---

## 2. Portability blockers (fix before anyone else clones this)

These aren't "weak assumptions" — they're the difference between the pipeline running for a second person at all.

1. **Hardcoded absolute Windows path + `os.chdir()` in every workflow script.** All of these literally hardcode `"C:/Users/dzimi/Documents/data_engineering/data_projects/git_hub_projects/bank_project/<module>"` and `os.chdir()` into it before running:
   - `balance_generate/python_code/b_s_gen_workflow.py:9`
   - `balance_gen_add_data/python_code/b_s_add_data_workflow.py:8`
   - `cash_flow_calc/python_code/cf_calc_workflow.py:10`
   - `irrbb_calc/python_code/{nii_calc_workflow.py:30, eve_calc_workflow.py:31, eba_sot_workflow.py:34}`
   - `ir_derivatives/python_code/irs_workflow.py:23`
   - `liq_calc/python_code/lcr_nsfr_workflow.py:21`

   Even with SQL Server pointed correctly, none of these scripts run after `git clone` to a different path, for a second contributor, or in CI. Fix: derive `BASE_DIR` from `os.path.dirname(os.path.abspath(__file__))` like `optimize_prep/python_code/extract_params.py:27-28` already does correctly — that file is the template to copy from.

2. **`sql_setup.py` duplicated 7 times** with an identical hardcoded connection string (`mssql+pyodbc://maciek_d/bank_gen...`) — `balance_generate`, `balance_gen_add_data`, `cash_flow_calc`, `ir_derivatives`, `irrbb_calc`, `liq_calc`, `optimize_prep`. Beyond the connection string itself, the CRUD helpers (`write_df`/`reset_data`/engine creation) are hand-copied with subtly different defaults (e.g. default schema varies between `'dbo'` and `'cf'` across copies — `cash_flow_calc/python_code/sql_setup.py:97-120` vs `ir_derivatives/python_code/sql_setup.py:104,215`). Fix: one shared `db_config.py` at repo root with the connection string as an environment variable, imported by all 7 modules.

3. **The same hardcoded constants are duplicated independently across many files**, with nothing enforcing they stay in sync:
   - `TOTAL_ASSETS = 10_000_000_000` in `balance_generate/python_code/b_s_gen_workflow.py:15` (as `total_assets`, lowercase), `liq_calc/python_code/lcr_nsfr_workflow.py:25`, `optimize_prep/python_code/extract_params.py:37`, `optimize_prep/python_code/accuracy_check.py:49`.
   - `REPORT_DATE = 2024-12-31` in at least 10 files across `balance_generate`, `cash_flow_calc`, `irrbb_calc` (×3), `ir_derivatives` (×2), `balance_gen_add_data`, `optimize_prep/extract_params.py`, `bs_optimization/swap_ladder.py`.

   Moving the report date or scaling total assets means editing every copy in lockstep by hand. `swap_ladder.py`'s comment "matches extract_params.REPORT_DATE" is a comment, not an assertion — nothing actually checks it. Worth centralizing into one `config.py` imported everywhere, or at minimum adding a startup assertion that cross-checks a couple of the copies.

---

## 3. Weak or simplified quantitative assumptions

Ranked by how much a practitioner reviewing this project's rigor would care.

4. **Prepayment is a static product×tenor lookup, not rate-dependent — no convexity in EVE/NII shocks.** `cash_flow_calc/python_code/cf_calc_workflow.py:132-136` joins a static `prep_rate` per product/tenor once at base cash-flow build time; the resulting cash flows are then reused **unchanged** by every EBA shock scenario in `irrbb_calc/python_code/{nii_calc_objects.py, eve_calc_objects.py}`. A par_dn scenario gets identical prepayment cash flows to par_up. This is the single biggest thing a real IRRBB reviewer would flag: negative convexity from rate-dependent prepayment is absent, understating EVE sensitivity for fixed-rate mortgages in down-shock scenarios and overstating it in up-shock. At minimum this deserves an explicit caveat in the methodology doc; a real fix would be a `CPR(Δr)` function.

5. **No FX conversion layer despite multi-currency scaffolding.** Per-currency curves and calendars exist throughout (`cash_flow_calc/python_code/cf_calc_objects.py:118-124`, `irrbb_calc/python_code/eba_shock_curves.py`, `optimize_prep/python_code/extract_params.py:580`), but there is no `exchange_rate`/`fx_spot` table or conversion step anywhere in the repo. Currently a no-op because the synthetic balance sheet is 100% PLN — but it's one Excel edit away from silently summing raw USD/EUR balances into PLN totals in LCR/NSFR/EVE/NII aggregation, with no guard rail.

6. **`vol_elasticity` exists in the schema but is only consumed by the optimizer, never by the core IRRBB engine.** `optimize_prep/python_code/extract_params.py:127` carries this column, but `irrbb_calc/python_code/{nii_calc_objects.py, eve_calc_objects.py}` only re-price existing balances under a shock — there's no deposit-volume or current-account-migration response modeled in NII/EVE itself. Standard constant-balance-sheet assumption for SOT purposes, but worth stating explicitly since the elasticity machinery living elsewhere could be mistaken for being wired into the core engine.

7. **The fast optimizer's bias correction is calibrated once at the base point and held constant across all shocked scenarios.** `optimize_prep/python_code/nii_eve_cf_fast.py:15-27,42-43,539-569` — self-documented in the docstring ("in shocked scenarios the bias is assumed constant"), so lower priority since the author already flagged it. Means the fast engine's shocked deltas are exact-adjacent only near the base point.

8. **`bs_optimization/python_code/swap_ladder.py:73-85`'s 25bps `IRS_MARGIN_BPS` is an admitted placeholder**, already self-documented as "a starting assumption, not a fitted value — retune once real swap-desk quotes are available." Flagging for completeness; the author has already done the diligence here.

9. **Silent exception-swallowing is a repeated (three separate, inconsistent) idiom across the optimizer stack:**
   - `optimize_prep/python_code/extract_params.py:109-114` (`_try_query`) catches `Exception` around every SQL load (used in 8+ places) and returns an empty DataFrame with just a `print` — a typo'd column name or schema change silently zeroes cohort data in `product_params.npz`, no error raised.
   - `nii_eve_cf_fast.py`'s calibration fallbacks (`_apply_fixed_rate_calibration`/`_apply_calibrated_delta_fallback`, lines 98-188) patch known mismatches row-by-row rather than structurally.
   - `bs_optimization/python_code/swap_ladder.py:380-412` (`_detect_hedge_direction`) defaults to `+1.0` on any DB failure, print-only, no exception surfaced.

   None of these share a logging convention — a maintainer debugging "why is this number wrong" has three different silent-failure code paths to rule out before finding the real cause.

---

## 4. Redundant / duplicated code

10. **`liq_calc/python_code/lcr_nsfr_objects.py:194-295` (`build_irrbb_report`) is dead code** — confirmed via repo-wide grep, it's never called anywhere. It independently re-implements the same delta/Tier-1-capital% SOT calculation that already exists in `irrbb_calc/python_code/{eba_sot_workflow.py:184-190, eve_calc_workflow.py:197-202, nii_calc_workflow.py:304-305}`. It also has a suspicious `tier1_capital: float = 1.0` default (should be hundreds of millions of PLN) which would silently produce nonsense if this function were ever revived without noticing. Delete it, or wire it in — not both existing.

11. **`COHORT_PRODUCT_CODES`/`SINGLE_ROW_PRODUCT_CODES` frozensets are hand-copied in 3+ files** with no shared source of truth: `optimize_prep/python_code/extract_params.py:52-56`, `diagnose_accuracy.py:44-45`, `cohort_cf_drill.py:55`, plus a partial copy in `accuracy_check.py:409`. See fragile-coupling finding below for the drift risk this creates.

---

## 5. Dead code / orphaned files

12. **Six files in `optimize_prep/python_code/` are not called by `opt_prep_workflow.py`, any Dagster asset, or tests**: `nii_formula_example.py`, `diagnose_accuracy.py`, `patch_npz_rwa.py`, `inspect_npz.py`, `cohort_cf_drill.py`, `nii_monthly_drill.py`. They sit in the same directory as production modules with no separation (no `scripts/`/`debug/` subfolder), so they silently rot as the real pipeline evolves — `diagnose_accuracy.py`'s hand-copied product-code sets (finding 11) are already stale relative to `extract_params.py` with nothing to catch the drift. Recommend moving these to a `optimize_prep/scratch/` or `debug/` folder, or deleting the ones that are truly superseded.

13. **`patch_npz_rwa.py` is an explicitly one-off migration** ("Run once after adding rwa_weight to bank_data.xlsx"). Since `rwa_weight` is already a standard column consumed in `extract_params.py:127`'s fillna list, this migration has almost certainly already been applied — the script is dead weight that will confuse a future reader into thinking `rwa_factor` still needs patching in. Safe to delete.

14. **Stale, never-resolved TODO in `cash_flow_calc/python_code/cf_calc_workflow.py:16-19`**: a Polish comment says mode=1 should only append schedules not already in existing tables, but `mode` is hardcoded to `0` and `sql_setup.reset_data`'s actual mode=1 branch does a full `DELETE FROM ... WHERE report_date = :rd` — not the incremental behavior the comment describes. `mode=1` is never exercised in production, so this is effectively untested, half-described functionality. Either implement it properly or delete the comment and the unused branch.

---

## 6. Fragile coupling

15. **No validation that `bs_structure`'s actual product codes are covered by `COHORT_PRODUCT_CODES ∪ SINGLE_ROW_PRODUCT_CODES ∪ IRS_PRODUCT_CODES`.** `bank_data.xlsx`'s `bs_structure` sheet is the real source of truth for which products exist, but the three sets in `extract_params.py:52-58` are a manually maintained partition with no assertion that `set(bs['product_code']) - (COHORT | SINGLE_ROW | IRS)` is empty. Add a new product to `bank_data.xlsx` — including the credit card / working-capital-vs-investment-loan / T-bill / central-bank-cash products on the roadmap — and it silently vanishes from `product_params.npz` with no error, only quietly-wrong (missing) totals downstream in the optimizer and LabBank. **Add this assertion before adding new products, not after.**

16. **Dagster provides zero schema/lineage validation between pipeline stages.** `dagster_pipeline/runner.py:12-35` (`run_workflow`) just `subprocess.Popen`s the target script and streams stdout. The `deps=[...]` declarations in `dagster_pipeline/assets/*.py` only sequence *when* scripts run, not *what* they exchange — the actual interface between stages is "whatever rows happen to be in SQL Server tables at the time," with no `IOManager`, no asset check, no dtype/shape contract. An upstream script silently writing a differently-shaped table only surfaces as a downstream `KeyError` or one of the silent `_try_query` catches (finding 9), never as a Dagster-visible contract violation.

17. **`extract_params.py:2436`: IRS notional weighting is derived as `notional / (2 × TOTAL_ASSETS)`,** so the swap book's implicit weight in the optimizer silently depends on this file's specific copy of `TOTAL_ASSETS` (finding 3) being correct. If `balance_generate`'s copy and this one ever diverge, IRS notional weighting goes wrong with no error — just a quietly-mis-scaled swap exposure in the optimizer.

---

## 7. Already fixed (context only, no action needed)

These were bugs caught and resolved earlier in development — listed here so they aren't rediscovered and "fixed" twice, and so the fix rationale isn't lost:

- **NII renewal rate bug** — `fwd_rt` for fixed-rate products is the coupon rate, not the market rate; renewal now correctly uses a disc-curve lookup for past-start cash flows.
- **SLSQP infeasible-accept bug** (`bs_optimization/python_code/bs_optimizer.py`) — polish/restore steps were accepting failed SLSQP iterates that violated hard floors; now gated on `_is_feasible()`.
- **IRS fixed-rate margin + seasoning** (`bs_optimization/python_code/swap_ladder.py`) — was pricing at raw curve mid with no spread, and seasoned buckets carried a historical rate-drift windfall; fixed with `IRS_MARGIN_BPS` and a seasoned-notional cap.

## 8. Already documented tech debt (not new findings, just cross-referenced)

- `sandbox/baseline.py:42-46` — LabBank shares `BalanceSheetParams`/the optimizer's npz, pulling in irrelevant optimizer-only fields (`vol_elasticity`, `subst_matrix`, PD/LGD, CoC). Author-flagged fix: give LabBank its own lightweight ETL reading directly from `bank_data.xlsx`, no optimizer columns.
- `sandbox/app.py` (~line 71) — product 6300 (`current_account_sme`) reuses product 6000's NMD decay curve as a proxy approximation rather than having its own calibrated behavioural model.
- Behavioural models and the balance sheet itself are synthetic/illustrative — already stated in `README.md`'s Repository status section, restated in the methodology doc for the public audience.

---

## Suggested order of operations

If/when there's time to work through this list, roughly in priority order:

1. Fix the hardcoded absolute paths (§2.1) — blocks anyone else from running this at all, and it's a mechanical fix (copy the `__file__`-relative pattern already used correctly in `extract_params.py`).
2. Add the product-code coverage assertion (§6.15) **before** adding the new products (credit cards, corporate loan split, T-bills, central bank cash) from the roadmap — otherwise a new product can silently vanish from the optimizer/LabBank tensors with no error.
3. Centralize `TOTAL_ASSETS`/`REPORT_DATE`/the SQL connection string into one config module (§2.2, §2.3) — mechanical, but removes an entire category of "forgot to update the other 7 copies" bugs.
4. Delete or relocate the dead files (§5) — low risk, immediate clarity improvement.
5. Everything in §3 (quantitative assumptions) is a "know about it, decide if it matters for your audience" list rather than a to-do list — the prepayment convexity gap (§3.4) is the one most worth a caveat in the methodology doc if not fixed outright.
