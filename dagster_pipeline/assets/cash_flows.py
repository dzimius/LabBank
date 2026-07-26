from dagster import asset, MaterializeResult, MetadataValue
from dagster_pipeline.runner import PROJECT_ROOT, run_workflow


@asset(
    deps=["balance_add_data"],
    group_name="cash_flows",
    compute_kind="python",
    description=(
        "Generate contractual and behavioural cash flow schedules for all products. "
        "Writes cf.products, cf.products_liq, irrbb.ir_gap_*, irrbb.liq_gap_*."
    ),
)
def cash_flows(context) -> MaterializeResult:
    script = PROJECT_ROOT / "cash_flow_calc" / "python_code" / "cf_calc_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )
