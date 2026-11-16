import pandas as pd
from dagster import asset, MaterializeResult, MetadataValue
from dagster_pipeline.partitions import BANK_PARTITIONS
from labbank_common.bank_config import get_bank
from dagster_pipeline.runner import PROJECT_ROOT, run_workflow



@asset(
    partitions_def=BANK_PARTITIONS,
    deps=["cash_flows"],
    group_name="derivatives",
    compute_kind="python",
    description=(
        "Load the swap book and FX deals from derivatives.xlsx (sheets swap / fx_spot), "
        "generate leg CF schedules, and merge their repricing gap into irrbb.ir_gap_beh. "
        "Skipped gracefully when no active swaps or FX deals exist for the report date."
    ),
)
def ir_swaps(context) -> MaterializeResult:
    bank = get_bank(context.partition_key)
    report_date = pd.Timestamp(bank.report_date)
    from labbank_common import derivatives as deriv
    irs_input = deriv.input_path(bank)

    df = pd.read_excel(irs_input, sheet_name=deriv.SWAP_SHEET)
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])
    fxd = deriv.read_fx_deals(bank)
    active = df[df["maturity_date"] > report_date]
    active_fx = fxd[fxd["far_date"].fillna(fxd["near_date"]) > report_date]

    if active.empty and active_fx.empty:
        context.log.warning(
            f"No active swaps / FX deals in {irs_input} for report_date {report_date.date()}. "
            "Skipping swap workflow — swap gap tables will remain empty."
        )
        return MaterializeResult(
            metadata={
                "swaps_count": MetadataValue.int(0),
                "skipped": MetadataValue.bool(True),
            }
        )

    context.log.info(f"Found {len(active)} active swaps, {len(active_fx)} FX deals — running swap workflow.")
    script = PROJECT_ROOT / "ir_derivatives" / "python_code" / "swap_workflow.py"
    run_workflow(context, script)
    return MaterializeResult(
        metadata={
            "swaps_count": MetadataValue.int(len(active)),
            "fx_deals_count": MetadataValue.int(len(active_fx)),
            "skipped": MetadataValue.bool(False),
            "script": MetadataValue.path(str(script)),
        }
    )
