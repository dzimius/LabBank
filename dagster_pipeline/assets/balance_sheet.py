from dagster import asset, MaterializeResult, MetadataValue
from dagster_pipeline.runner import PROJECT_ROOT, run_workflow


@asset(
    group_name="balance_sheet",
    compute_kind="python",
    description=(
        "Generate synthetic balance sheet transactions. "
        "Writes to schemat.transactions, schemat.loans, schemat.deposits, "
        "schemat.financial_instruments, schemat.equity, schemat.cash_accounts."
    ),
)
def balance_transactions(context) -> MaterializeResult:
    script = PROJECT_ROOT / "balance_generate" / "python_code" / "b_s_gen_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )


@asset(
    deps=["balance_transactions"],
    group_name="balance_sheet",
    compute_kind="python",
    description=(
        "Load market data and build schedule ID tables. "
        "Writes mkt.curves, mkt.fixings, sched.loans, sched.deposits, "
        "sched.fin_inst, and schemat.models_* behavioral models."
    ),
)
def balance_add_data(context) -> MaterializeResult:
    script = (
        PROJECT_ROOT / "balance_gen_add_data" / "python_code" / "b_s_add_data_workflow.py"
    )
    run_workflow(context, script)
    return MaterializeResult(
        metadata={"script": MetadataValue.path(str(script))}
    )
