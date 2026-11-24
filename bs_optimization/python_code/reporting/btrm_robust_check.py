"""btrm_robust_check.py
=====================
Full-engine check of the robust optimisation: every article portfolio repriced on the alternative base
curves of robust_curves.py (and on today's NS fit), with the portfolio's own new swaps on top, compared
with the optimiser's linear view (JointResult.robust).

    LABBANK_BANK=eur    python btrm_robust_check.py
    LABBANK_BANK=eur_y1 python btrm_robust_check.py

Output: runs/<bank>/bs_optimization/btrm_robust_check.json
  {tag: {curve: {"eve": worst % T1, "nii": worst % T1, "nii_level": EUR m vs today's fit}}}
"""
from __future__ import annotations

import os as _o, sys as _s
_o.environ["LABBANK_NII_LEVEL_VS_TODAY"] = "1"
_PY = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
for _g in ("optimizers", "scenarios", "hedging", "reporting"):
    if _o.path.join(_PY, _g) not in _s.path:
        _s.path.insert(0, _o.path.join(_PY, _g))

import json
import os

import numpy as np

import btrm_mc500 as mc
import robust_curves as rc
import per_scenario_hedge as psh

TAGS = {"eur": ("current", "current_renewed", "lt_profit_only", "lt_target"), "eur_y1": ("plan", "proposal_profit_only", "alm_proposal")}


def main() -> None:
    runs = json.load(open(os.path.join(str(mc.get_bank(mc.BANK_ID).output_dir("bs_optimization")), f"{mc.RUNS_NAME}.json")))
    params = mc.BalanceSheetParams.load(os.path.join(psh.OPTPREP_OUT, "product_params.npz"))
    pm = mc._ProductMap(params)
    t1 = mc.ase.aee.irrbb_sql_setup.load_tier1_capital(mc.ase.REPORT_DATE)
    z = np.load(rc.OUT_PATH, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    import hashlib
    _curve_salt = "_" + hashlib.md5(np.round(z["betas"], 6).tobytes() + "|".join(names).encode()).hexdigest()[:6]
    tmpl = psh.build_mc500_bank().iloc[[0]]
    bank = tmpl.loc[tmpl.index.repeat(len(names))].reset_index(drop=True)
    bank[["beta0", "beta1", "beta2"]] = z["betas"]
    bank["shift_idx"] = range(len(names)); bank["shape"] = names; bank["level"] = names; bank["is_anchor"] = False
    out = {}
    specs = mc.unit_specs_by_product(pm, params)
    ta = params.total_assets
    for tag in TAGS[mc.BANK_ID]:
        # the cache must follow the curve set too (2026-10-30: the y1 curves changed, same portfolios hit stale files)
        label = mc.portfolio_label(tag, runs[tag]) + "_robust" + _curve_salt
        if not os.path.exists(os.path.join(mc.OUT_DIR, f"irrbb_mc500_{label}.xlsx")):
            mc.run_and_cache(label=label, products=pm.products, baseline_weights=pm.base_prod_w,
                             target_weights=mc.family_weights(pm, runs[tag]["weights_new"], own=set(specs)),
                             include_irs=True, bank=bank, out_dir=mc.OUT_DIR, filename_prefix="irrbb_mc500",
                             n_workers=len(names), min_w=pm.min_w)
        E, N = mc.load_mc500_baseline(label, out_dir=mc.OUT_DIR)
        oe, on = mc.overlay_deltas(bank, runs[tag], t1)
        ref = z["betas"][names.index("today_fit")]
        ue, un, ul = mc.unit_overlay(bank, runs[tag]["weights_new"], pm, specs, ta, t1, ref)
        oe, on = oe + ue, on + un
        lvl = mc.base_nii(label) + mc.overlay_base_nii(bank, runs[tag], ref) + ul
        i0 = names.index("today_fit")
        out[tag] = {n: dict(eve=float((E + oe)[i].min()), nii=float((N + on)[i].min()),
                            nii_level=float(lvl[i] - lvl[i0]) / 1e6) for i, n in enumerate(names)}
        lp = runs[tag].get("robust", {})
        for n, v in out[tag].items():
            l = lp.get(n)
            print(f"[{tag}] {n:9s} EVE {v['eve']:+6.2f}" + (f" (LP {min(l['eve'].values()):+6.2f})" if l else "")
                  + f"  NII {v['nii']:+6.2f}" + (f" (LP {min(l['nii'].values()):+6.2f})" if l else "")
                  + f"  NII level {v['nii_level']:+6.1f}m" + (f" (LP EP {l['ep_delta'] / 1e6:+6.1f})" if l else ""))
    with open(os.path.join(str(mc.get_bank(mc.BANK_ID).output_dir("bs_optimization")), f"btrm_robust_check{mc._R_SFX}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
