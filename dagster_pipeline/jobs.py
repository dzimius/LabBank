from dagster import AssetSelection, define_asset_job

from dagster_pipeline.assets import (
    balance_transactions,
    balance_add_data,
    cash_flows,
    ir_swaps,
    nii_results,
    eve_results,
    eba_sot_results,
    lcr_nsfr_results,
    optimize_prep_tensors,
    hyp_scenario_curves,
)

# ── Balance sheet generation (with optional IRS positions) ────────────────────
# Generates the full balance sheet and loads all market data.
# ir_swaps runs here so IRS positions (schemat.ir_swaps, cf.ir_swap_*) are
# visible immediately after BS generation. The IRS gap merge into
# irrbb.ir_gap_beh is deferred gracefully until cash_flows has been run.
# ir_swaps is skipped entirely when irs_input.xlsx has no active swaps.
balance_sheet_job = define_asset_job(
    name="balance_sheet_job",
    selection=AssetSelection.assets(
        balance_transactions,
        balance_add_data,
        ir_swaps,
    ),
    description=(
        "Full balance sheet generation: synthetic transactions → market data "
        "(curves, fixings, behavioral models, schedule tables) → IRS positions. "
        "IRS gap merge is deferred until cash_flows is run. "
        "ir_swaps skipped when irs_input.xlsx has no active swaps."
    ),
)

# ── Full run (balance sheet → cash flows → IRS → IRRBB + Liquidity) ───────────
# ir_swaps skips gracefully when irs_input.xlsx has no active swaps.
full_run_job = define_asset_job(
    name="full_run_job",
    selection=AssetSelection.assets(
        balance_transactions,
        balance_add_data,
        cash_flows,
        ir_swaps,
        nii_results,
        eve_results,
        eba_sot_results,
        lcr_nsfr_results,
    ),
    description=(
        "Complete ALM pipeline: generate balance sheet → load market data → "
        "build cash flows → process IRS → compute IRRBB (NII, EVE, EBA SOT) "
        "and Liquidity (LCR, NSFR). "
        "ir_swaps is a no-op when irs_input.xlsx has no active swaps."
    ),
)

# ── IRS update → recompute IRRBB ──────────────────────────────────────────────
# Use when IRS trades change but the balance sheet and cash flows are unchanged.
# Reads irs_input.xlsx, re-merges swap gap into irrbb.ir_gap_beh, then reruns
# NII, EVE, and EBA SOT. lcr_nsfr_results is unaffected and not rerun.
irs_update_job = define_asset_job(
    name="irs_update_job",
    selection=AssetSelection.assets(
        ir_swaps,
        nii_results,
        eve_results,
        eba_sot_results,
    ),
    description=(
        "IRS trade update workflow: reload swaps from irs_input.xlsx, "
        "merge new swap gap into irrbb.ir_gap_beh, then recompute "
        "NII, EVE, and EBA SOT. Balance sheet and cash flows are unchanged."
    ),
)

# ── IRRBB-only recalculation ───────────────────────────────────────────────────
# Use when only the market curves need refreshing (e.g. manual curve edit in mkt.curves).
# Cash flows, IRS gap, and balance sheet data remain unchanged.
irrbb_recalc_job = define_asset_job(
    name="irrbb_recalc_job",
    selection=AssetSelection.assets(
        nii_results,
        eve_results,
        eba_sot_results,
    ),
    description=(
        "Recompute IRRBB metrics only (NII, EVE, EBA SOT). "
        "Use after a manual market curve update. "
        "Cash flows, IRS gap, and balance sheet data remain unchanged in the DB."
    ),
)

# ── Liquidity standalone ───────────────────────────────────────────────────────
liq_only_job = define_asset_job(
    name="liq_only_job",
    selection=AssetSelection.assets(lcr_nsfr_results),
    description=(
        "Standalone LCR / NSFR calculation. "
        "Requires cash_flows to already be materialized."
    ),
)

# ── Optimizer prep ────────────────────────────────────────────────────────────
# Run after any full pipeline run to refresh the fast-approximation tensors.
optimize_prep_job = define_asset_job(
    name="optimize_prep_job",
    selection=AssetSelection.assets(optimize_prep_tensors, hyp_scenario_curves),
    description=(
        "Build optimizer curve tensors and product parameters, run the "
        "fast-vs-exact accuracy check, then pre-compute exact IRRBB for the "
        "sandbox's 15 hypothetical curves. Run after full_run_job or irs_update_job."
    ),
)

# ── LabBank data refresh (full_run_job + optimize_prep in one job) ────────────
# Use this after editing balance_generate/input_data/bank_data.xlsx (and/or
# irs_input.xlsx) to regenerate your own balance sheet AND refresh the npz
# tensors that sandbox/app.py (LabBank) reads. One job instead of running
# full_run_job then optimize_prep_job separately.
labbank_data_job = define_asset_job(
    name="labbank_data_job",
    selection=AssetSelection.assets(
        balance_transactions,
        balance_add_data,
        cash_flows,
        ir_swaps,
        nii_results,
        eve_results,
        eba_sot_results,
        lcr_nsfr_results,
        optimize_prep_tensors,
        hyp_scenario_curves,
    ),
    description=(
        "Full pipeline + optimizer prep in one job: regenerate the balance "
        "sheet from balance_generate/input_data/bank_data.xlsx, run the "
        "complete ALM pipeline, then rebuild the npz tensors LabBank "
        "(sandbox/app.py) reads. Run this after editing the input Excel "
        "files, then click 'Reload my data' in the running Streamlit app."
    ),
)

all_jobs = [
    balance_sheet_job,
    full_run_job,
    irs_update_job,
    irrbb_recalc_job,
    liq_only_job,
    optimize_prep_job,
    labbank_data_job,
]
