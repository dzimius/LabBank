# Diagnostics (ad-hoc, not part of the pipeline)
One-off drill-downs into the optimize_prep fast layer; run by hand. Workbooks go to
`runs/<bank>/optimize_prep/` and are not tracked.

| Script | What it writes |
|---|---|
| `diagnose_accuracy.py` | `accuracy_by_product.xlsx` — approx vs exact NII / EVE per product × scenario |
| `cohort_cf_drill.py` | SQL table `opt_prep.cohort_cf_compare` — fast vs exact cash flows per cohort |
| `nii_monthly_drill.py` | `nii_monthly_drill.xlsx` — month-by-month NII renewal |
| `nii_formula_example.py` | `nii_formula_example.xlsx` — the NII formula worked in Excel |
| `inspect_npz.py` | `<name>_dump.xlsx` — the npz caches as sheets |

