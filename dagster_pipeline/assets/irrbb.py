from dagster import asset, MaterializeResult, MetadataValue
from dagster_pipeline.runner import PROJECT_ROOT, run_workflow

_IRRBB_DIR = PROJECT_ROOT / "irrbb_calc" / "python_code"


@asset(
    deps=["cash_flows", "ir_swaps"],
    group_name="irrbb",
    compute_kind="python",
    description=(
        "Build EBA shocked yield curves and compute NII for base + all stressed scenarios. "
        "Writes irrbb.curves (shocked), irrbb.nii_results, irrbb.nii_*_scenario tables, "
        "and output/nii_results.xlsx + output/irrbb_shock_curves.xlsx."
    ),
)
def nii_results(context) -> MaterializeResult:
    script = _IRRBB_DIR / "nii_calc_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )


@asset(
    deps=["nii_results"],
    group_name="irrbb",
    compute_kind="python",
    description=(
        "Compute EVE (run-off PV) for base + all stressed scenarios. "
        "Runs after nii_results to avoid concurrent writes to irrbb.curves. "
        "Writes irrbb.eve_results, irrbb.eve_*_scenario tables, "
        "and output/eve_results.xlsx."
    ),
)
def eve_results(context) -> MaterializeResult:
    script = _IRRBB_DIR / "eve_calc_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )


@asset(
    deps=["nii_results", "eve_results"],
    group_name="irrbb",
    compute_kind="python",
    description=(
        "Compute EBA SOT with bucket-level 50% supervisory haircut. "
        "Writes irrbb.irrbb_report, results.irrbb_report, "
        "output/eba_sot_results.xlsx, and output/irrbb_report.xlsx."
    ),
)
def eba_sot_results(context) -> MaterializeResult:
    script = _IRRBB_DIR / "eba_sot_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )
