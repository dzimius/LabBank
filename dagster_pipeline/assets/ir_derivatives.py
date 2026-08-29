import pandas as pd
from dagster import asset, MaterializeResult, MetadataValue
from dagster_pipeline.runner import PROJECT_ROOT, run_workflow

_REPORT_DATE = pd.to_datetime("2026-06-30")


@asset(
    deps=["cash_flows"],
    group_name="derivatives",
    compute_kind="python",
    description=(
        "Load IRS transactions from irs_input.xlsx, generate fixed/float CF schedules, "
        "and merge the IRS repricing gap into irrbb.ir_gap_beh. "
        "Skipped gracefully when no active swaps exist for the report date."
    ),
)
def ir_swaps(context) -> MaterializeResult:
    irs_input = PROJECT_ROOT / "ir_derivatives" / "input" / "irs_input.xlsx"

    df = pd.read_excel(irs_input)
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])
    active = df[df["maturity_date"] > _REPORT_DATE]

    if active.empty:
        context.log.warning(
            "No active IRS found in irs_input.xlsx for report_date 2024-12-31. "
            "Skipping IRS workflow — swap gap tables will remain empty."
        )
        return MaterializeResult(
            metadata={
                "swaps_count": MetadataValue.int(0),
                "skipped": MetadataValue.bool(True),
            }
        )

    context.log.info(f"Found {len(active)} active IRS — running IRS workflow.")
    script = PROJECT_ROOT / "ir_derivatives" / "python_code" / "irs_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={
            "swaps_count": MetadataValue.int(len(active)),
            "skipped": MetadataValue.bool(False),
            "script": MetadataValue.path(str(script)),
        }
    )
