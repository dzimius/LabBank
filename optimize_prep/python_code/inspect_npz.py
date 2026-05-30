"""inspect_npz.py
================
Dumps any .npz file to Excel — one sheet per array.
Scalars and 1-D arrays → single column. 2-D arrays → full matrix.

Usage
-----
    python inspect_npz.py                          # inspects both default files
    python inspect_npz.py path/to/file.npz         # inspects one file
"""
from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

DEFAULT_FILES = [
    os.path.join(OUTPUT_DIR, "product_params.npz"),
    os.path.join(OUTPUT_DIR, "curve_tensors.npz"),
]


def npz_to_excel(npz_path: str) -> str:
    data = np.load(npz_path, allow_pickle=True)
    out_path = npz_path.replace(".npz", "_dump.xlsx")

    print(f"\nFile: {os.path.basename(npz_path)}")
    print(f"  Arrays: {list(data.keys())}")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        # ── Index sheet: name / shape / dtype for every array ────────────────
        index_rows = []
        for key in data.keys():
            arr = data[key]
            index_rows.append({
                "array":  key,
                "shape":  str(arr.shape),
                "dtype":  str(arr.dtype),
                "sample": str(arr.flat[0]) if arr.size > 0 else "",
            })
        pd.DataFrame(index_rows).to_excel(writer, sheet_name="INDEX", index=False)

        # ── One sheet per array ───────────────────────────────────────────────
        for key in data.keys():
            arr = np.array(data[key])
            sheet = key[:31]  # Excel sheet name limit

            if arr.ndim == 0:
                pd.DataFrame([[arr.item()]], columns=[key]).to_excel(
                    writer, sheet_name=sheet, index=False)

            elif arr.ndim == 1:
                pd.DataFrame({key: arr}).to_excel(
                    writer, sheet_name=sheet, index=False)

            elif arr.ndim == 2:
                pd.DataFrame(arr).to_excel(
                    writer, sheet_name=sheet, index=True, header=True)

            else:
                # 3-D: flatten first axis into rows labelled dim0=x
                rows = []
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        row = {"dim0": i, "dim1": j}
                        for k in range(arr.shape[2]):
                            row[f"m{k+1}"] = arr[i, j, k]
                        rows.append(row)
                pd.DataFrame(rows).to_excel(
                    writer, sheet_name=sheet, index=False)

            print(f"  sheet '{sheet}': shape={arr.shape}")

    print(f"  → {out_path}")
    return out_path


if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FILES
    for f in files:
        if not os.path.exists(f):
            print(f"Not found: {f}")
            continue
        npz_to_excel(f)
