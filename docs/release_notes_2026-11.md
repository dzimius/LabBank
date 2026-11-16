# LabBank release 2026-11

The first public version was the PLN bank (October 2026). This release adds the EUR bank, the
automatic one-year roll-forward of its balance sheet, and the balance-sheet + swap optimisation
used for the EUR results. It is a selection of the development code, not all of it; see
"Not in this release".

## What is in it

**Core pipeline (all modules updated)**

- One codebase for several banks: `labbank_common/bank_config.py` defines each bank (SQL database,
  report date, input and output folders). Shipped banks: `pln` (30 Jun 2026) and `eur` (30 Sep 2026),
  plus `eur_y1`, the EUR bank one year later (30 Sep 2027), built by the roll-forward.
- New folder layout: inputs in `inputs/<bank>/{balance,derivatives,models,forecast}`, market data in
  `inputs/market/<ccy>/`, all results in `runs/<bank>/<module>/` (not tracked), LabBank demo data in
  `demo/<bank>/`.
- EUR bank: euro-area market data (EURIBOR, ESTR, EUR IRS, ECB deposit statistics), multi-curve
  (ESTR discounting, EURIBOR 1M/3M/6M/12M projection), a pay-fixed IRS ladder, committed facilities
  off balance sheet, credit cards, catalogue products (products the bank can sell but does not hold).
- Cash flows: last available fixing, ModifiedFollowing calendar, capped last annuity instalment.
- IRRBB: EBA RTS shocks and floor, one NII engine for base and shocks (renewal on the product's
  repricing period), NMD core profile locked at today's rate, EBA CPR/TDRR multipliers (switch, off
  by default), CSRBB, beta-adjusted gap, a packed NumPy pricing engine for many curves.
- Liquidity: LCR/NSFR counterparty and market-value rules, Pillar 2 (ILAAP) metrics, off-balance
  outflows.
- FTP: separate rate and liquidity tenors, forward-consistent term rates, NMD replicating portfolio.
- Accounting view: valuation at the report date and, for a rolled-forward bank, the monthly P&L and
  capital of the plan year (`accounting/`).
- Results history: every run is recorded in `meta.runs` of the bank's database.
- One command line for the whole pipeline (`python -m labbank_common.cli`), the same steps as the
  Dagster job.
- Release check (`release_check/`): golden-master comparison of a bank's results with its stored
  reference, so a change for one bank cannot silently change another.

**Roll-forward (+1 year)**

`balance_generate/python_code/b_s_rollforward.py` ages every contract of the `eur` book to
30 Sep 2027 (amortisation, prepayment, maturities, NMD outflows), adds one year of new business to
reach the growth plan, keeps the live swaps and adds the planned ones. Market data for the year:
OIS forwards up to 2 years, unchanged curve from 5 years, interpolated in between. The result is
the bank `eur_y1`, which then runs through the normal pipeline.

Inputs (`inputs/eur/forecast/`):
- `growth_plan.xlsx`: total-assets growth and the balance-sheet mix in one year.
- `new_swaps.xlsx`: swaps planned inside the year (option 0 none, 1 same as the last 12 months,
  2 the same scaled by growth).

**Optimisation (joint balance sheet + swaps)**

`bs_optimization/` holds only what these runs need (see its README). For each bank
(`eur` = today, `eur_y1` = the plan) the optimiser chooses the balance-sheet mix and new par
pay-fixed swaps that maximise economic profit inside the EVE and NII risk appetite, with
business-line, growth and run-off limits and pricing concessions for growth above the plan.
- Robust curves: the EVE/NII limits also hold on other base curves (historical rate regimes for
  `eur`, the plan curve +/-25bp for `eur_y1`).
- Runs: today's book, long-term target, one-year ALM proposal on the plan, the swap routes and
  sensitivities (`reporting/btrm_article_runs.py`).
- Checks: every result is re-priced on the full engine; a full-engine check on the robust curves;
  an out-of-sample Monte Carlo test (500 curves) with the minimum new-swap notional per curve.

## How to run

Set up Python and SQL Server as in `docs/01_setup_guide.md` (one empty database per bank:
`bank_gen_eur`, `bank_gen_eur_y1`; the roll-forward creates its own if missing). Commands are run
from the repository root.

```bash
# 1. EUR bank: whole pipeline + accounting
python -m labbank_common.cli run --bank eur
# 2. roll-forward to 30 Sep 2027 and the pipeline on the rolled book
python -m labbank_common.cli run --bank eur_y1
# 3. optional: compare with the shipped references
python release_check/release_check.py --bank eur
python release_check/release_check.py --bank eur_y1
```

To change the plan, edit `inputs/eur/forecast/growth_plan.xlsx` / `new_swaps.xlsx` and rerun step 2.
After changing products in `bank_data.xlsx` or swaps in `derivatives.xlsx`, rebuild the templates
first: `python balance_generate/python_code/make_growth_plan.py --bank eur` and
`python balance_generate/python_code/forecast_swaps.py --bank eur`.

Optimisation, one bank per process, after steps 1-2 (Git Bash syntax; in PowerShell set
`$env:LABBANK_BANK="eur"` first):

```bash
LABBANK_BANK=eur    python bs_optimization/python_code/scenarios/robust_curves.py
LABBANK_BANK=eur_y1 python bs_optimization/python_code/scenarios/robust_curves.py
LABBANK_BANK=eur    python bs_optimization/python_code/reporting/btrm_article_runs.py
LABBANK_BANK=eur_y1 python bs_optimization/python_code/reporting/btrm_article_runs.py
# checks (optional)
LABBANK_BANK=eur_y1 python bs_optimization/python_code/reporting/btrm_robust_check.py
LABBANK_BANK=eur    python bs_optimization/python_code/reporting/btrm_mc500.py
LABBANK_BANK=eur_y1 python bs_optimization/python_code/reporting/btrm_mc500.py
LABBANK_BANK=eur    python bs_optimization/python_code/hedging/per_scenario_hedge.py --book current
```

Results: `runs/<bank>/bs_optimization/btrm_article_runs.json` (one entry per run: mix, swaps, EP,
EVE/NII on every curve), `btrm_robust_check.json`, `btrm_mc500.json`. The Monte Carlo test runs
on NumPy (`LABBANK_MC_BACKEND=numpy`; `pandas` is the slower production loop).

Run times on a desktop PC: EUR pipeline ~25 min, roll-forward + eur_y1 pipeline ~27 min,
robust curves ~2 min per bank, article runs under 30 s, Monte Carlo (500 curves, NumPy) ~2.5 min
per bank, robust check ~2-3 min, per-scenario hedge ~20 min.

## Market data and random numbers

- EUR curve: public market data. EUR IRS quotes 1Y-30Y come from the daily market reports of the
  Greek Public Debt Management Agency (PDMA), short tenors from EURIBOR fixings, the overnight
  point from ECB (€STR / EONIA); ECB AAA government yields fill dates before the first PDMA report.
  The built inputs (`inputs/market/eur/curve_input.xlsx`, `fixing_input.xlsx`) are included. The
  raw PDMA and EURIBOR files are not (their terms of reuse are unclear); to rebuild the inputs,
  fetch them as described in `inputs/market/eur/fetch/README.md`.
- The EUR book is drawn at random from `bank_data.xlsx`, and the Monte Carlo test draws its own
  curves. This release uses its own seeds, so its numbers are close to, but not the same as, results
  published elsewhere. The shipped references (`release_check/references/`) and the LabBank demo
  data match this release.

## Not in this release

- The optimisation report (notebook, HTML/PDF).
- The raw PDMA / EURIBOR files (see above) and the clearing-house
  settlement file used to build `inputs/market/eur/curve_basis.csv` (the built file is included).

## Known limitations

- SQL Server is required for the pipeline (by design: every step lands in tables you can query).
  The server name in `bank_config.SqlCfg` must be set to your instance.
- Fixed-rate FTP takes its market rate from a Nelson-Siegel fit of the curve, which misses the swap
  quotes (30 Sep 2026: +17bp at 5Y, -57bp at 10Y). Product margins over FTP are off by that much
  for long fixed-rate products; EVE and NII do not use FTP.
- The optimiser uses a linear fast layer; its EVE differs from the full engine by up to ~0.15pp of
  Tier 1, so it keeps a 0.2pp buffer inside the appetite. Final numbers come from the full-engine
  re-pricing.
- The optimisation scripts hold their settings as constants at the top of each file and are written
  for the `eur` and `eur_y1` banks.
- EBA CPR/TDRR multipliers are off by default and not used in the optimisation.
