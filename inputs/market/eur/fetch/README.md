# EUR raw market data — how it is fetched

All scripts write into `../euribor_raw/`; `../build_eur_inputs.py` then turns the raw files into
`fixing_input.xlsx` and `curve_input.xlsx`. Fetch scripts need the separate `venv_openbb`
(requests, pypdf, truststore); the build script runs in the project `venv`.

| Raw file | Source | Script |
|---|---|---|
| `euribor_3m-1w-1m-6m-12m_*.csv` | EURIBOR daily fixings, pasted manually | — |
| `ecb_estr_eonia.csv`, `ecb_aaa_spot.csv` | ECB Data Portal API | `fetch_ecb.py` |
| `eur_irs_pdma.csv` | EUR IRS 1Y–30Y from the Greek Public Debt Management Agency's daily market reports (Bloomberg source): every report from 2026, two per month before (2018-04 →) | `pdma_index.py` → `pdma_extract.py` |
| `settlement-prices_20261007.csv` | Clearing-house settlement curves (not in the repository); only needed to rebuild `../curve_basis.csv` with `../build_curve_basis.py` | — |

```powershell
venv_openbb\Scripts\python.exe inputs\market\eur\fetch\fetch_ecb.py
venv_openbb\Scripts\python.exe inputs\market\eur\fetch\pdma_index.py pdma_index.csv
venv_openbb\Scripts\python.exe -I inputs\market\eur\fetch\pdma_extract.py pdma_index.csv <download_dir> inputs\market\eur\euribor_raw\eur_irs_pdma.csv
venv\Scripts\python.exe inputs\market\eur\build_eur_inputs.py
```

Validation (2026-10-07): the curve bootstrapped from EURIBOR + PDMA IRS is within ±3.6 bp of the
clearing house's EUR.EURIBOR.6M zero curve for 1Y–10Y (17–29 bp above €STR OIS = EURIBOR/€STR basis;
the project discounts single-curve, as for PLN).

Note on redistribution: the IRS quotes originate from Bloomberg via PDMA's public reports — check
the terms before publishing the raw files in a public repository.
