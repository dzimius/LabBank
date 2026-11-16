# Forecast inputs of the 1-year roll-forward

One folder per BASE bank (`inputs/<bank>/forecast/`). Every bank rolled forward from it
(`labbank_common/bank_config.py`, `RollforwardCfg`: eur_y1 from eur, market data on the "hybrid" path) reads these files;
`balance_generate/python_code/b_s_rollforward.py` turns them into the rolled bank's inputs.

| File | What | Rebuild |
|---|---|---|
| `growth_plan.xlsx` | total-assets growth % (B1) and the balance-sheet mix in one year | `python balance_generate/python_code/make_growth_plan.py --bank eur` |
| `new_swaps.xlsx` | IRS the bank plans to trade inside the year: B1 option 0 = none, 1 = the last 12 months' swaps again one year later (same notionals and months), 2 = the same with notionals x (1 + growth). Start dates after the base report date only. | `python balance_generate/python_code/forecast_swaps.py --bank eur` |

Products the bank can sell but does not hold (bank_data `bs_percentage` 0, and the tenor / fixing variants of the `catalogue` sheet) are growth-plan rows at `share_new_pct` 0: a positive share sells the whole target as new business.

What the bank holds today is NOT here: products in `inputs/<bank>/balance/bank_data.xlsx`,
swaps and FX deals in `inputs/<bank>/derivatives/derivatives.xlsx`.
After editing: `python release_check/release_check.py --bank eur_y1 --run`.
