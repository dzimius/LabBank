"""Fast per-bank check (no SQL Server): every bank's committed sandbox / optimizer tensors
still load and give the accepted fast metrics.

Catches a shared-code change that breaks or silently changes ANOTHER bank's app.
References come from `python release_check/release_check.py --bank <id> --accept`.
Run with:  pytest release_check/
"""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from labbank_common import demo  # noqa: E402
from labbank_common.bank_config import BANKS  # noqa: E402

REF_DIR = Path(__file__).resolve().parent / "references"


@pytest.mark.parametrize("bank_id", sorted(BANKS))
def test_fast_metrics_match_reference(bank_id):
    ref_path = REF_DIR / f"{bank_id}.json"
    if not ref_path.exists():
        pytest.skip(f"no reference for {bank_id} yet")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))["key_numbers"]["fast_metrics"]
    if isinstance(ref, str):
        pytest.skip(f"{bank_id}: reference has no fast metrics ({ref})")

    bank = BANKS[bank_id]
    demo.install(bank)      # a clean clone has no runs/ yet
    # A bank without a demo pack (eur_y1, eur_ext) has no tensors in a clean clone. PLN and EUR
    # have one in demo/: a missing file there must still fail.
    if not (demo.DEMO_ROOT / bank_id).exists() and not (
            bank.output_dir("optimize_prep", create=False) / "product_params.npz").exists():
        pytest.skip(f"{bank_id}: generated tensors not present (rebuild with the pipeline)")

    from release_check import fast_metrics
    cur = fast_metrics(bank)
    assert not isinstance(cur, str), f"{bank_id}: {cur}"
    for key, expected in ref.items():
        got = cur[key]
        assert math.isfinite(got), f"{bank_id}.{key} is not finite: {got}"
        assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-6), \
            f"{bank_id}.{key}: reference {expected:,.6f}, now {got:,.6f}"
