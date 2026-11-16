from dagster import asset, MaterializeResult, MetadataValue
from dagster_pipeline.partitions import BANK_PARTITIONS
from dagster_pipeline.runner import PROJECT_ROOT, run_workflow


@asset(
    partitions_def=BANK_PARTITIONS,
    deps=["cash_flows"],
    group_name="liquidity",
    compute_kind="python",
    description=(
        "Compute LCR (30-day) and NSFR (1-year) liquidity ratios. "
        "Reads cf.products, schemat.deposits, and bank_data.xlsx regulatory weights. "
        "Writes results.lcr_nsfr and output/<bank>/lcr_nsfr_report.xlsx."
    ),
)
def lcr_nsfr_results(context) -> MaterializeResult:
    script = PROJECT_ROOT / "liq_calc" / "python_code" / "lcr_nsfr_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )


@asset(
    partitions_def=BANK_PARTITIONS,
    deps=["lcr_nsfr_results"],
    group_name="liquidity",
    compute_kind="python",
    description="Off-balance report, for banks with extended reports. Writes results.off_balance*.",
)
def off_balance_results(context) -> MaterializeResult:
    script = PROJECT_ROOT / "liq_calc" / "python_code" / "off_balance_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(metadata={"script": MetadataValue.path(str(script))})
