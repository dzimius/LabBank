"""patch_npz_rwa.py
==================
Add rwa_factor array to existing product_params.npz from bank_data.xlsx.

Run once after adding rwa_weight to bank_data.xlsx.
Does not require a SQL connection — reads only the Excel config.
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(_HERE, "..", ".."))

BS_PATH  = os.path.join(ROOT_DIR, "balance_generate", "input_data", "bank_data.xlsx")
NPZ_PATH = os.path.join(_HERE, "..", "output", "product_params.npz")


def patch():
    bs = pd.read_excel(BS_PATH, sheet_name="bs_structure")
    bs["product_code"] = bs["product_code"].astype(str)
    bs["rwa_weight"] = pd.to_numeric(bs.get("rwa_weight", 0.0), errors="coerce").fillna(0.0)

    # Build (product_code, bs_side) → rwa_weight lookup
    rwa_map: dict[tuple[str, str], float] = {}
    for _, row in bs.iterrows():
        rwa_map[(str(row["product_code"]), str(row["bs_side"]))] = float(row["rwa_weight"])

    data = np.load(NPZ_PATH, allow_pickle=True)
    pc  = data["product_code"]
    sid = data["bs_side"]
    n   = len(pc)

    rwa_factor = np.array([
        rwa_map.get((str(pc[i]), str(sid[i])), 0.0)
        for i in range(n)
    ], dtype=float)

    # Rebuild npz with rwa_factor added
    out = {k: data[k] for k in data.files}
    out["rwa_factor"] = rwa_factor
    np.savez(NPZ_PATH, **out)

    ta = float(np.asarray(data["total_assets"]).flat[0])
    bal = data["balance_arr"].astype(float)
    rwa_total = float(np.dot(bal, rwa_factor))
    print(f"rwa_factor added: {n} cohorts")
    print(f"Baseline RWA:  {rwa_total/1e6:,.0f}M PLN")
    print(f"Total assets:  {ta/1e6:,.0f}M PLN   (balance sum: {bal.sum()/1e6:,.0f}M)")
    # T1 from equity products
    equity_mask = np.array([str(s) == "E" for s in sid])
    t1_implicit = float(bal[equity_mask].sum())
    print(f"Equity (T1):   {t1_implicit/1e6:,.0f}M PLN")
    if rwa_total > 0:
        print(f"T1 / RWA:      {t1_implicit/rwa_total*100:.1f}%")
    print(f"Saved: {NPZ_PATH}")


if __name__ == "__main__":
    patch()
