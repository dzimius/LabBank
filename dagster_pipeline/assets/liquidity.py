from dagster import asset, MaterializeResult, MetadataValue
from dagster_pipeline.runner import PROJECT_ROOT, run_workflow


@asset(
    deps=["cash_flows"],
    group_name="liquidity",
    compute_kind="python",
    description=(
        "Compute LCR (30-day) and NSFR (1-year) liquidity ratios. "
        "Reads cf.products, schemat.deposits, and bank_data.xlsx regulatory weights. "
        "Writes results.lcr_nsfr and output/lcr_nsfr_report.xlsx."
    ),
)
def lcr_nsfr_results(context) -> MaterializeResult:
    script = PROJECT_ROOT / "liq_calc" / "python_code" / "lcr_nsfr_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )
