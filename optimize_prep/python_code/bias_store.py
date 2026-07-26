"""bias_store.py
===============
Load / save per-product EVE and NII bias corrections.

Bias = (exact_pipeline - method_B) per unit of product balance, per shocked scenario.

Saved by accuracy_check.py after every comparison run.
Used by:
  - bs_optimizer.py  (corrects fast EVE/NII constraints toward exact values)
  - metrics.py       (corrects sandbox EVE/NII display toward exact values)

Scaling assumption
------------------
When product weight doubles, its bias doubles proportionally.
This is exact for the linear term and a good first-order approximation for
the CF-based method B errors.

    corrected_delta_EVE(s) = fast_delta_EVE(s)
                             + sum_i(amounts[i] * bias_eve_unit[i, s])
"""
from __future__ import annotations

import os
import numpy as np

_HERE      = os.path.dirname(os.path.abspath(__file__))
BIAS_PATH  = os.path.normpath(os.path.join(_HERE, "..", "output", "bias_corrections.npz"))


def save_bias_corrections(
    bias_eve_unit: np.ndarray,   # (n_cohorts, n_shocked_scenarios)
    bias_nii_unit: np.ndarray,   # (n_cohorts, n_shocked_scenarios)
    scenario_ids:  list[str],    # length n_shocked_scenarios (excludes base)
    path: str = BIAS_PATH,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        bias_eve_unit = bias_eve_unit,
        bias_nii_unit = bias_nii_unit,
        scenario_ids  = np.array(scenario_ids, dtype=object),
    )
    print(f"Bias corrections saved: {path}  ({len(scenario_ids)} scenarios, {bias_eve_unit.shape[0]} cohorts)")


def load_bias_corrections(
    path: str = BIAS_PATH,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str] | None]:
    """Load bias corrections.

    Returns
    -------
    (bias_eve_unit, bias_nii_unit, scenario_ids) or (None, None, None) if file not found.
    """
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True)
    return (
        data["bias_eve_unit"],
        data["bias_nii_unit"],
        list(data["scenario_ids"].astype(str)),
    )


def apply_eve_bias(
    result:        dict[str, float],   # from compute_eve_cf_fast_all — modified in place
    amounts:       np.ndarray,
    bias_eve_unit: np.ndarray,
    scenario_ids:  list[str],
) -> dict[str, float]:
    """Add EVE bias correction to fast result dict (in place, also returned)."""
    for s_idx, scen in enumerate(scenario_ids):
        if scen in result:
            result[scen] += float(np.dot(amounts, bias_eve_unit[:, s_idx]))
    return result


def apply_nii_bias(
    result:        dict[str, float],   # from compute_nii_cf_all — modified in place
    amounts:       np.ndarray,
    bias_nii_unit: np.ndarray,
    scenario_ids:  list[str],
) -> dict[str, float]:
    """Add NII bias correction to fast result dict (in place, also returned).

    Shifts each shocked scenario's absolute NII so delta_NII matches exact.
    Base NII is unchanged (bias only applies to shocked deltas).
    """
    for s_idx, scen in enumerate(scenario_ids):
        if scen in result:
            result[scen] += float(np.dot(amounts, bias_nii_unit[:, s_idx]))
    return result
