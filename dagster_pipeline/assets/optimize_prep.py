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
