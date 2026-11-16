"""Demo pack: the files LabBank (sandbox/) needs after a clean clone.

Results live in runs/ and are not in git, so a fresh clone has none. demo/<bank_id>/ holds a frozen
copy of the five files the app reads, refreshed only for a release, not on every pipeline run.

    python -m labbank_common.demo --refresh pln eur      # runs/ -> demo/, before a release
"""
from __future__ import annotations

import argparse
import shutil

from labbank_common.bank_config import PROJECT_ROOT, BankConfig, get_bank

DEMO_ROOT = PROJECT_ROOT / "demo"
DEMO_FILES = {
    "optimize_prep": ("product_params.npz", "curve_tensors.npz", "ftp_rates.npz", "bias_corrections.npz"),
    "sandbox": ("scenario_curves.npz",),
}


def install(bank: BankConfig) -> bool:
    """Copy the demo pack into runs/ if the bank has no results yet. Existing results are never
    touched: a pipeline run always wins over the demo. Returns True if it copied the pack."""
    src = DEMO_ROOT / bank.bank_id
    if (bank.output_dir("optimize_prep", create=False) / "product_params.npz").exists() or not src.exists():
        return False
    for module, names in DEMO_FILES.items():
        dst = bank.output_dir(module)
        for name in names:
            shutil.copy2(src / name, dst / name)
    return True


def refresh(bank: BankConfig) -> None:
    """Copy the bank's current results into demo/<bank_id>/."""
    dst = DEMO_ROOT / bank.bank_id
    dst.mkdir(parents=True, exist_ok=True)
    for module, names in DEMO_FILES.items():
        for name in names:
            shutil.copy2(bank.output_dir(module, create=False) / name, dst / name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", nargs="+", metavar="BANK", required=True)
    for bank_id in ap.parse_args().refresh:
        refresh(get_bank(bank_id))
        print(f"demo/{bank_id} refreshed")
