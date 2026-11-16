"""catalogue_params.py
=====================
Unit parameters for CATALOGUE products: rows of bank_data.xlsx (bs_structure) with
bs_percentage = 0 -- products the bank offers but does not hold yet. The starting book has
no contracts for them, so extract_params.py finds no cohorts, CFs or IRRBB results to
calibrate on; without this module they would get no npz row and the optimizers could never
sell them.

Each catalogue product gets one optimizer row with balance 0 (the base metrics are
unchanged): a new-business cohort originated on the report date, per unit of balance.

Cohort products (loans, bonds, term deposits, ...)
---------------------------------------------------
- Price: the generator's rule (b_s_gen_objects.compute_asset_rates) for a contract starting
  on the report date -- index = the product's rate_index fixing on the report date, spread =
  the product-level margin_pct of the rate model (interest_rt.xlsx) or, without one, the mean
  contract margin the generator draws (bank_config generation.margin_map), client rate =
  rate_models.client_rate(...) with the product's beta / floors / caps.
- Cash flows: the contractual schedule of that contract from its bank_data row (maturity,
  payment_freq, amort_type 0 bullet / 1 annuity / 2 linear); no prepayment.
- EVE: every CF discounted on the curve tensors (base + each EBA scenario) up to the first
  repricing -- maturity, or the fixation (fixing_freq) of a fixed rate, or the first reset of
  a floating rate -- where the outstanding comes back at par (the run-to-repricing convention
  extract_params uses for EVE). The per-scenario PVs
  go into the cr_eve_pv_frac table, so the fast EVE layer prices the row exactly like this.
- NII (12M): fixed = the coupon on the scheduled outstanding plus repaid capital renewed at
  the new-business rate (1M forward + margin, as the renewal stream of every cohort);
  floating = constant balance, locked coupon until the first reset, then beta x forward +
  margin (the rate_matrix rule of extract_params). Shocked NII from the same formulas on the
  shocked curves -> delta_nii_unit.
- Liquidity: 30-day contractual receipts (LCR inflow, counterparty rate of lcr_rules) and
  1Y amortisation from the same schedule; all other factors (HQLA, ASF/RSF, RWA, PD/LGD,
  fees, acquisition cost, facility terms) from the bank_data row as for held products.

Single-row products (current / saving accounts, cash, equity)
--------------------------------------------------------------
Balance-sheet metadata as usual; NII per unit = the client rate on today's fixing, EVE
factor = face value. Their behavioural (NMD) repricing and run-off are NOT modelled for a
product the bank does not hold yet: delta NII / EVE = 0 -- a printed warning says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from labbank_common import rate_models
from labbank_common.lcr_rules import inflow_rate as _inflow_rate

PAR_SHOCK_RATE = 0.02


def _months(tenor, default: int = 12) -> int:
    """'5Y' -> 60, '3M' -> 3, '12M' -> 12, '7D' -> 1 (whole months, at least 1)."""
    if tenor is None or (isinstance(tenor, float) and np.isnan(tenor)):
        return default
    s = str(tenor).strip().upper()
    try:
        n = float(s[:-1])
    except ValueError:
        return default
    return max(int(round({"Y": n * 12, "M": n, "D": n / 30.44, "W": n / 4.35}.get(s[-1], default))), 1)


def catalogue_rows(bs: pd.DataFrame, held: set) -> pd.DataFrame:
    """bank_data rows at 0% whose (product_code, bs_side, currency) the book does not hold."""
    zero = bs[bs["full_pct"].fillna(0.0) <= 0.0]
    keep = [(str(r.product_code), str(r.bs_side), str(r.currency)) not in held for r in zero.itertuples()]
    return zero[keep].copy()


def fixing_on(fixings: pd.DataFrame, rate_index, date) -> float:
    """Fixing (decimal) of rate_index on or before date; NaN without one."""
    if not isinstance(rate_index, str) or not rate_index:
        return np.nan
    f = fixings[(fixings["rate_index"] == rate_index) & (fixings["date"] <= pd.Timestamp(date))]
    return float(f.sort_values("date")["rate"].iloc[-1]) / 100.0 if len(f) else np.nan


def price(row, rm_row, fixings, report_date, margin_map: dict) -> dict:
    """Client rate of a contract starting on the report date (generator rule)."""
    beta = 1.0
    lim = dict(index_floor=np.nan, index_cap=np.nan, client_floor=np.nan, client_cap=np.nan)
    b_pct = np.nan
    if rm_row is not None:
        beta = float(rm_row["beta"])
        b_pct = float(rm_row["margin_pct"])
        lim = {k: float(rm_row[k]) for k in lim}
    if not np.isnan(b_pct) and b_pct != 0.0:
        spread = b_pct / 100.0
    else:
        spread = float(margin_map.get(str(row["product_name"]), {}).get("mean", 0.0))
    index = fixing_on(fixings, row.get("rate_index"), report_date)
    rate = float(rate_models.client_rate(0.0 if np.isnan(index) else index, spread, beta, **lim))
    return dict(index=index, spread=spread, beta=beta, client_rt=rate, **lim)


def schedule(term_m: int, pay_m: int, amort_type: int, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Monthly contractual (capital, interest) per unit of balance, months 1..term_m."""
    cap = np.zeros(term_m)
    intr = np.zeros(term_m)
    pay_m = max(min(pay_m, term_m), 1)
    pay_months = list(range(pay_m, term_m + 1, pay_m))
    if pay_months[-1] != term_m:
        pay_months.append(term_m)
    n_pay = len(pay_months)
    out, prev = 1.0, 0
    r_p = rate * pay_m / 12.0
    for k, m in enumerate(pay_months):
        i_amt = out * rate * (m - prev) / 12.0
        if amort_type == 1 and k < n_pay - 1:            # annuity
            pmt = r_p / (1.0 - (1.0 + r_p) ** (-n_pay)) if abs(r_p) > 1e-12 else 1.0 / n_pay
            c_amt = min(max(pmt - out * r_p, 0.0), out)
        elif amort_type == 2 and k < n_pay - 1:          # linear
            c_amt = 1.0 / n_pay
        elif k == n_pay - 1:                              # last payment: the rest
            c_amt = out
        else:                                             # bullet
            c_amt = 0.0
        cap[m - 1] += c_amt
        intr[m - 1] += i_amt
        out -= c_amt
        prev = m
    return cap, intr


def _fwd_F(disc: np.ndarray, event_t: int, F: float) -> float:
    """F-month forward from month event_t (extract_params' rate_matrix rule)."""
    n = len(disc)
    i_s = max(0, min(event_t - 1, n - 1)) if event_t > 0 else -1
    i_e = max(0, min(event_t - 1 + int(F), n - 1))
    df_s = disc[i_s] if i_s >= 0 else 1.0
    return (df_s / max(disc[i_e], 1e-12) - 1.0) * 12.0 / F


def unit_cohort(row, pr: dict, discs: dict, scen_ids: list, sign: float) -> dict:
    """Per-unit NII / EVE / schedule parameters of a new-business cohort.

    discs: scenario_id -> (360,) discount factors of the row's currency, month k at k-1.
    scen_ids: ['base', shocked...] -- column order of the PV and NII arrays."""
    rt = str(row["rate_type"]) if pd.notna(row.get("rate_type")) else "F"
    term_m = _months(row.get("maturity"))
    pay_m = _months(row.get("payment_freq"), default=term_m)
    amort = int(row["amort_type"]) if pd.notna(row.get("amort_type")) else 0
    # first repricing: the fixing period (floating reset, or a fixed rate's fixation such as
    # a 25Y mortgage fixed for 10Y), else maturity
    rep_m = min(_months(row.get("fixing_freq"), default=term_m), term_m)
    r0 = pr["client_rt"]
    cap, intr = schedule(term_m, pay_m, amort, r0)

    # EVE CFs: the locked coupon to the first repricing, the outstanding back at par there
    cf = (cap + intr)[:rep_m].copy()
    cf[-1] += 1.0 - cap[:rep_m].sum()
    n_cf = len(cf)
    pv = np.column_stack([cf * discs[s][:n_cf] for s in scen_ids])          # (n_cf, n_scen)
    keep = np.abs(cf) > 0
    m_idx = np.arange(n_cf)[keep]                                           # bucket = month - 1
    pv = pv[keep]
    pv_tot = pv.sum(axis=0)
    b = scen_ids.index("base")
    eve_pv_factor = sign * pv_tot[b]
    delta_eve = sign * (pv_tot - pv_tot[b])
    if rt == "F":
        d_mod = -sign * (pv_tot[scen_ids.index("par_dn")] - pv_tot[scen_ids.index("par_up")]) / (2.0 * PAR_SHOCK_RATE)
    else:
        d_mod = rep_m / 12.0 / 2.0

    # NII (12M) per scenario
    ca, cb = pr["beta"], pr["spread"]
    fl = pr["client_floor"] if np.isfinite(pr["client_floor"]) else -np.inf
    cp = pr["client_cap"] if np.isfinite(pr["client_cap"]) else np.inf
    remain = np.array([(12 - m - 0.5) / 12.0 for m in range(12)])
    if rt == "F":
        outstanding = 1.0 - np.concatenate([[0.0], np.cumsum(cap)])[:term_m]
        out_m = np.zeros(12)
        out_m[:min(12, term_m)] = outstanding[:12]
        cap_m = np.zeros(12)
        cap_m[:min(12, term_m)] = cap[:12]
        cap_remain_m = cap_m * remain
        nii = []
        for s in scen_ids:
            renew = np.clip([ca * _fwd_F(discs[s], m, 1) + cb for m in range(12)], fl, cp)
            nii.append(float(out_m.sum() / 12.0 * r0 + np.dot(cap_remain_m, renew)))
        t_first, locked = 999.0, r0
    else:
        out_m = np.ones(12)
        cap_m = np.zeros(12)
        cap_remain_m = np.zeros(12)
        F, t1 = float(rep_m), int(rep_m)
        nii = []
        for s in scen_ids:
            path = [r0 if m < t1 else
                    float(np.clip(ca * _fwd_F(discs[s], int(t1 + int((m - t1) / F) * F), F) + cb, fl, cp))
                    for m in range(12)]
            nii.append(float(np.dot(out_m / 12.0, path)))
        t_first, locked = float(t1), r0
    nii = np.asarray(nii) * sign
    first_30d = float(cap[0] + intr[0]) if pay_m == 1 else 0.0
    return dict(
        rate_type=rt, repricing_tenor_m=float(rep_m), coupon_rate=r0 if rt == "F" else np.nan,
        margin=cb, nii_unit_rate=float(nii[b]), delta_nii=nii - nii[b],
        eve_pv_factor=float(eve_pv_factor), delta_eve=delta_eve, d_mod=float(d_mod),
        eve_eff=(m_idx, pv), out_m=out_m, cap_m=cap_m, interest_yf_m=out_m / 12.0,
        cap_remain_m=cap_remain_m, t_first_m=t_first, locked_rate=locked,
        first_30d=first_30d, amort_1y=float(cap[:12].sum()),
    )


def unit_specs(params, bank, engine) -> list:
    """(cohort index, bank_data row, today's price, sign) of every catalogue cohort of params: the
    new-business units extract_params built (balance 0, originated on the report date), rebuilt with the
    same inputs so they can be repriced on other curves (robust_curves.py, Monte Carlo)."""
    from labbank_common import catalogue, fx
    rd = pd.Timestamp(str(params.report_date))
    bs = catalogue.read_bs_structure(bank.balance_dir() / "bank_data.xlsx")
    bs["sign"] = bs["bs_side"].map({"A": 1.0, "L": -1.0, "E": -1.0}).fillna(-1.0)
    bs["product_code"] = bs["product_code"].astype(str)
    rm = rate_models.load_rate_models(engine, rd)
    fix = fx.read_market_input("fixing_input.xlsx", bank)
    fix["date"] = pd.to_datetime(fix["date"])
    out = []
    tag = f"_{rd.year}_{rd.month:02d}"
    for i, cid in enumerate(params.cohort_id):
        if params.balance_arr[i] != 0.0 or not str(cid).endswith(tag) or not bool(params.is_cohort[i]):
            continue
        pc, side, ccy = str(params.product_code[i]), str(params.bs_side[i]), str(params.currency[i])
        r = bs[(bs["product_code"] == pc) & (bs["bs_side"] == side) & (bs["currency"] == ccy)
               & (bs["bs_percentage"].fillna(0.0) <= 0.0)]
        if len(r) != 1:
            continue
        crow = r.iloc[0]
        pr = price(crow, rm.loc[pc] if pc in rm.index else None, fix, rd, bank.generation.new_business_margin_map)
        out.append((i, crow, pr, float(crow["sign"])))
    return out


def unit_on_curve(spec, disc_m: np.ndarray, scen_ids: list) -> tuple[float, np.ndarray, np.ndarray]:
    """(12M NII per unit, delta NII per shock, delta EVE per shock) of a unit_specs entry on one curve.
    disc_m: (len(scen_ids), >=360) DF at month k = 1.. per scenario; scen_ids[0] must be 'base'."""
    _, crow, pr, sign = spec
    u = unit_cohort(crow, pr, {s: disc_m[k] for k, s in enumerate(scen_ids)}, scen_ids, sign)
    return u["nii_unit_rate"], np.asarray(u["delta_nii"])[1:], np.asarray(u["delta_eve"])[1:]


def unit_single(row, pr: dict, sign: float) -> dict:
    """Single-row catalogue product: rate on today's fixing, face value, no sensitivity."""
    rate = pr["client_rt"] if isinstance(row.get("rate_index"), str) else 0.0
    return dict(nii_unit_rate=sign * rate, eve_pv_factor=sign * 1.0)
