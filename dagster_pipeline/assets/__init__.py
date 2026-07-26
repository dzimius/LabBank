from dagster_pipeline.assets.balance_sheet import balance_transactions, balance_add_data
from dagster_pipeline.assets.cash_flows import cash_flows
from dagster_pipeline.assets.ir_derivatives import ir_swaps
from dagster_pipeline.assets.irrbb import nii_results, eve_results, eba_sot_results
from dagster_pipeline.assets.liquidity import lcr_nsfr_results
from dagster_pipeline.assets.optimize_prep import optimize_prep_tensors

all_assets = [
    balance_transactions,
    balance_add_data,
    cash_flows,
    ir_swaps,
    nii_results,
    eve_results,
    eba_sot_results,
    lcr_nsfr_results,
    optimize_prep_tensors,
]
