from dagster import asset, MaterializeResult, MetadataValue
from dagster_pipeline.runner import PROJECT_ROOT, run_workflow


@asset(
    deps=["eba_sot_results", "lcr_nsfr_results"],
    group_name="optimize",
    compute_kind="python",
    description=(
        "Build optimizer tensors and validate fast-metric approximations. "
        "Step 1: extract yield curve tensors (curve_tensors.npz). "
        "Step 2: extract product parameters (product_params.npz). "
        "Step 3: accuracy check — fast vs exact NII/EVE/LCR/NSFR. "
        "Writes optimize_prep/output/*.npz, *.xlsx."
    ),
)
def optimize_prep_tensors(context) -> MaterializeResult:
    script = PROJECT_ROOT / "optimize_prep" / "python_code" / "opt_prep_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )


@asset(
    deps=["optimize_prep_tensors"],
    group_name="optimize",
    compute_kind="python",
    description=(
        "Pre-compute exact per-product IRRBB (NII + EVE, base + 7 EBA shocks) for "
        "the 15 stylised hypothetical yield curves the LabBank sandbox offers. "
        "Re-prices the existing run-off / 12M CF streams under each curve with the "
        "production pipeline functions — no CF regeneration. Writes "
        "sandbox/scenario_curves.npz (read by the sandbox with a cohort-set "
        "staleness guard; falls back to the analytical model if stale)."
    ),
)
def hyp_scenario_curves(context) -> MaterializeResult:
    script = PROJECT_ROOT / "sandbox" / "build_scenario_curves.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )
