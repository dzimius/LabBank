# LabBank — Methodology

A practical description of what LabBank calculates and why, written for practitioners, managers, and students — not a regulatory text, and not an academic paper. Where a formula is genuinely necessary to understand what's happening, it's included; otherwise this stays in plain language.

## The idea

A bank's balance sheet produces thousands of cash flows every day. Regulatory metrics — NII, EVE, LCR, NSFR — must all be derived from those cash flows in a consistent, auditable, reproducible way. In practice this is hard: rates reset, balances prepay, deposits decay, and shock scenarios need to be applied simultaneously across every position in every currency.

Most systems hide this complexity behind aggregated dashboards. LabBank makes it explicit, on one core principle:

> Every number in every report traces back to a single row in a single cash-flow table. No aggregation happens before it is recorded.

That means NII, EVE, LCR, and NSFR are not four separate models — they're four different aggregations of the *same* underlying cash-flow schedule.

## Architecture, in four layers

```
Excel inputs  →  Python modules (orchestrated by Dagster)  →  SQL Server (9 schemas)  →  Reports
```

SQL Server is the single integration bus. Every step reads from tables the previous step wrote, instead of passing hidden in-memory state — which means every stage can be re-run independently and audited by querying the database directly.

## Pipeline, step by step

1. **Balance sheet generation** — synthetic transactions are drawn from `bank_data.xlsx` using truncated-normal balance distributions and historical WIBOR rates. Written to `dbo.transactions` and five `schemat.*` tables (loans, deposits, financial instruments, equity).
2. **Market data enrichment** — yield curves bootstrapped into `mkt.curves`; historical fixings loaded into `mkt.fixings`; behavioural model parameters (prepayment, NMD decay) loaded into `bs.*`.
3. **Cash flow generation** — for every transaction, a full schedule of contractual amortisation plus behavioural adjustments is produced and written to `cf.products`. Repricing gaps are written to `irrbb.ir_gap_*`.
4. **IR derivatives (optional)** — swap legs read from `irs_input.xlsx`, cash-flow schedules written to `cf.ir_swap_*`, and the swap gap merged into `irrbb.ir_gap_beh`.
5. **IRRBB calculation** — six EBA shock scenarios applied to all curves; NII computed over a 1-year horizon, EVE as the present value of run-off cash flows. Both written to `irrbb.*`.
6. **Liquidity calculation** — LCR (HQLA vs. stressed 30-day net outflows) and NSFR (available vs. required stable funding over 1 year), written to `results.lcr_nsfr`.
7. **Reporting** — Jupyter notebooks read live from SQL and export to HTML; Power BI connects directly to `schemat.*` for interactive repricing-gap exploration; Excel outputs for IRRBB and LCR/NSFR.

Steps 5, 6, and 7 can each be re-run independently without touching the balance sheet or cash flows — enforced by the Dagster job definitions (`irrbb_recalc_job`, `liq_only_job`, etc).

## Cash flow engine and behavioural modelling

For each product, the engine generates two things per period: **capital cash flows** (contractual amortisation plus behavioural adjustments) and **interest cash flows** (coupon payments at the contracted client rate).

Behavioural assumptions applied:

- **Loans** — a constant prepayment rate (CPR) applied monthly to the outstanding balance.
- **Non-maturity deposits (NMD)** — an exponential decay model: a fraction leaves each month, the remainder is treated as sticky long-term funding.
- **Admin-rate floor** — interest on current accounts is capped at 0% and never passed through to the client as a negative rate.

The key output table, `cf.products`, has one row per (transaction, payment date), including both the contractual and behavioural outstanding balance and cash flow, the market forward rate, and the all-in client rate. NII is computed from the interest cash flows within the 1-year horizon; EVE uses the full behavioural run-off schedule to maturity.

## Repricing gap and interest rate swaps

The **repricing gap** is the net volume of assets minus liabilities repricing in each tenor bucket. A positive gap means more assets reprice in that bucket — the bank benefits from rising rates there; a negative gap signals liability-sensitive exposure.

Interest rate swaps overlay this gap **without changing the underlying balance sheet**: the swap gap is computed separately and merged into the balance-sheet repricing gap before IRRBB is calculated. This is what lets LabBank show "balance sheet only" vs. "balance sheet + IRS" gap views side by side.

## IRRBB — NII

NII decomposes into two parts per scenario:

- **Locked interest** — income from cash flows already contracted (rate and notional fixed). Immune to rate shocks.
- **Renewal** — income from maturing capital reinvested at market rates within the 1-year horizon. This part reprices with the curve.

A bank with more assets than liabilities repricing in the short end is **asset-sensitive**: parallel-up shocks help NII, parallel-down shocks hurt it.

## IRRBB — EVE

EVE is the present value of all future cash flows under a **run-off assumption** — no new business, the existing book winds down to maturity:

```
EVE = Σᵢ CFᵢ_behavioural × discount_factor(tᵢ)
ΔEVE_scenario = EVE_scenario − EVE_base
```

Because EVE uses all maturities (not a 1-year cut-off), it's the slower calculation of the two. A negative ΔEVE means economic net worth falls under that scenario — typically because long fixed-rate assets lose value faster than liabilities as rates rise. This is what creates the classic NII-vs-EVE trade-off: a bank positioned to benefit from rising rates on NII is usually the same bank that loses EVE in that scenario.

## EBA shock scenario construction

Six supervisory scenarios (per EBA/RTS/2022/10) are applied to each currency curve — parallel up/down, steepener, flattener, and short-rate up/down — plus an "own" scenario. Shocks are floored at 0% per tenor (rates can't be shocked below zero in this construction). Example PLN parameters: parallel shock ±250 bps, short-rate ±350 bps, long-rate ±150 bps; the "own" scenario adds a −100 bps parallel shock.

## Supervisory Outlier Test (EBA SOT)

The SOT applies a **50% haircut per tenor bucket** to any scenario leg that produces a *gain*, before summing across buckets:

```
ΔEVE_regulatory = Σ_b [ min(ΔEVE_b, 0) + 0.5 × max(ΔEVE_b, 0) ]
```

Applying the haircut at bucket level (not on the total) is deliberate and conservative — it prevents a gain in one bucket from offsetting a loss in another before the haircut is applied. The same bucket-level logic is applied to the NII SOT.

Thresholds:

- **EVE SOT**: ΔEVE_regulatory / Tier 1 capital < −15% → breach.
- **NII SOT**: ΔNII_regulatory / Tier 1 capital < −5% → breach.

## Liquidity — LCR and NSFR

```
LCR  = HQLA / stressed net outflows (30 days)   ≥ 100%
NSFR = Available Stable Funding / Required Stable Funding   ≥ 100%
```

- **HQLA** — Level 1 liquid assets (cash, central bank reserves, sovereign bonds) at their respective haircuts.
- **Net outflows** — stressed deposit run-off and interbank maturities within 30 days.
- **ASF** — stable funding weighted by tenor and type (equity and long-term debt score highest).
- **RSF** — assets weighted by illiquidity (loans and illiquid instruments score highest).

## Behavioural liquidity gap

Beyond the regulatory ratios, the liquidity gap shows the net cash position across the full tenor ladder (up to 60 months), using behavioural cash flows — loan prepayments included, NMD decay applied. A sustained negative cumulative gap identifies the point at which the bank would need additional funding — the primary input for contingency funding planning, distinct from the point-in-time LCR/NSFR ratios.

## Exploring this interactively — LabBank

The Streamlit app (`sandbox/app.py`) lets you stress every piece described above without touching SQL: change balance sheet composition, add or edit swaps, stress NMD decay assumptions, and immediately see NII, EVE, EBA SOT, LCR, and NSFR respond — plus the repricing gap and liquidity gap charts. See the [setup guide](01_setup_guide.md) to run it.

## What's next — balance sheet optimisation

Balance sheet optimisation (constrained economic-profit vs. ΔEVE/ΔNII breach trade-off) is the next phase, implemented in `bs_optimization/` as four solvers (deterministic, joint balance-sheet + swap overlay, stochastic Monte Carlo, and a natural-hedge minimiser). It's functional but not yet part of the guided LabBank path — treat it as a preview of where this project is heading rather than a finished, documented feature.

## A note on scope

Behavioural models (prepayment, NMD decay) are illustrative rather than calibrated to a real portfolio, and the balance sheet itself is synthetic. This project demonstrates *how* an ALM pipeline should be built and how its metrics relate to each other — it is not a substitute for a validated, governed production risk system.
