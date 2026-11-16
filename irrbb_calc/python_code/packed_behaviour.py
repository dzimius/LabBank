"""Rate-dependent behaviour on the packed pricing backend (NumPy).

Per curve, the multiplier gamma of each product (labbank_common/rate_behaviour.py: CPR, TDRR, NMD_CORE)
is set from the curve's move at the product's reference tenor, and the product's cash flows are
regenerated with it: prepayments of fixed annuity loans, term-deposit
redemptions, NMD survival S(t)^gamma. NII renews the NMD outflow beyond the baseline as the migration
product (irrbb_behaviour.nmd_migration). At gamma = 1 every flow is the stored baseline schedule.

Checked against the production generators (cf_calc_objects.compute_adjusted_schedule,
gen_term_deposit_redemption_sched, compute_deposit_beh_schedule) and pricing functions at gamma 0.83 /
1 / 1.17. Used by
bs_optimization/python_code/scenarios/mc_packed.py.
"""
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import QuantLib as ql
from sqlalchemy import bindparam, text

_ROOT = Path(__file__).resolve().parents[2]
for _d in (_ROOT / "cash_flow_calc" / "python_code", _ROOT / "irrbb_calc" / "python_code"):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))
import cf_calc_objects as cf  # noqa: E402
import packed_pricing as p  # noqa: E402
from labbank_common import curves, rate_behaviour as rb  # noqa: E402


MODEL_TYPES = {"CPR", "TDRR", "NMD_CORE"}


def models_for_inputs(model_sets):
    selected = [{(rb.PTYPE[m.behaviour], m.product_code): m for m in models if m.behaviour in MODEL_TYPES}
                for models in model_sets]
    if any(set(models) != set(selected[0]) for models in selected):
        raise ValueError("model sets must contain the same products and behaviour families")
    return selected


def loan_frame(inputs, date):
    frame = inputs["loans"].sort_values(["schedule_id", "cf_end_dt"]).reset_index(drop=True)
    for col in ("start_date", "cf_start_dt", "cf_end_dt", "fixing_dt"):
        frame[col] = pd.to_datetime(frame[col])
    if frame[["client_rt", "con_total_pmt", "balance_amt"]].isna().any().any():
        raise ValueError("incomplete contract terms for CPR")
    ages = (frame["cf_end_dt"].dt.year - frame["start_date"].dt.year) * 12
    ages += frame["cf_end_dt"].dt.month - frame["start_date"].dt.month
    model = inputs["loan_model"].assign(product_code=lambda x: x["product_code"].astype(str))
    model = model.set_index(["product_code", "tenor"])["prep_rate"]
    keys = pd.MultiIndex.from_arrays([frame["product_code"], ages.astype(str) + "M"])
    frame["cpr"] = model.reindex(keys).fillna(0.).to_numpy()
    frame["renew_months"] = curves.renew_months(frame["fixing_freq"], frame["rate_type"], frame["term_months"])
    frame["rate_type"] = p.fe.effective_rate_type(frame, date)
    frame["exact_yf"] = exact_yf(frame)
    return frame


def exact_yf(frame):
    days = (frame["cf_end_dt"] - frame["cf_start_dt"]).dt.days.to_numpy()
    out = np.empty(len(frame))
    for name, indices in frame.groupby("dc_conv").indices.items():
        if name in ("ACT/360", "ACT/365"):
            out[indices] = days[indices] / int(name[4:])
        else:
            dc = cf.get_dc_conv_from_str(name)
            rows = frame.iloc[indices]
            out[indices] = [dc.yearFraction(ql.Date.from_date(a.date()), ql.Date.from_date(b.date()))
                            for a, b in zip(rows["cf_start_dt"], rows["cf_end_dt"])]
    return out


def deposit_frame(inputs, grid, models):
    nmd = inputs["nmd_model"]
    tdrr = inputs["term_model"].set_index("product_code")["tdrr"]
    base = grid.series(0)
    frames = []
    for row in inputs["deposits"].itertuples(index=False):
        code = str(row.product_code)
        model = models[("D", code)]
        term = (None if pd.isna(row.maturity_date) else
                (row.maturity_date.year - row.start_date.year) * 12 + row.maturity_date.month - row.start_date.month)
        if model.behaviour == "TDRR" and row.rate_type == "F" and term is not None:
            tau = float(tdrr[row.product_code])
            schedules = cf.gen_term_deposit_redemption_sched(row, grid.date, base, tau)
            if schedules is None:
                continue
            frame = schedules[1].assign(tau=tau, overnight=[True, False])
        elif model.behaviour == "NMD_CORE" and term is None:
            unit = row._replace(balance_amt=1.)
            profile = nmd.loc[nmd["product_code"].eq(row.product_code)]
            frame = cf.compute_deposit_beh_schedule(unit, grid.date, base, profile)
            if frame is None:
                raise ValueError(f"NMD product {code} is missing a survival profile")
            frame = frame.assign(s0=frame["outstanding_bal"], s1=frame["outstanding_bal"] - frame["capital_pmt"],
                                 client_rt=frame["int_pmt"] / (frame["outstanding_bal"] * frame["cf_yf"]))
        else:
            continue
        if "fixing_dt" not in frame:
            frame["fixing_dt"] = pd.NaT
        frame = frame.assign(product_type="D", product_code=code, currency=row.currency, rate_type="F",
                             fixing_freq=row.fixing_freq, margin=0., balance_amt=float(row.balance_amt),
                             base_df=frame["d_f"], exact_yf=frame["cf_yf"], cf_yf=frame["cf_yf"].round(6),
                             renew_months=curves.renew_months([row.fixing_freq], [row.rate_type], [term])[0])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def loan_terms(rows):
    first = np.flatnonzero(~rows["schedule_id"].duplicated()).astype(np.int32)
    end = np.r_[first[1:], len(rows)].astype(np.int32)
    yf = rows["exact_yf"].to_numpy(dtype=float)
    q = rows["cpr"].to_numpy(dtype=float)
    annual = 1. - (1. - q) ** (1. / np.where(yf > 0, yf, 1.))
    rate = rows["client_rt"].to_numpy(dtype=float)
    balance = rows["balance_amt"].to_numpy(dtype=float)[first]
    discount = (1. + np.repeat(rate[first], end - first) * yf)
    # The first contractual annuity is based on the reset rate and the remaining date grid.
    pv = np.asarray([np.sum(1. / np.cumprod(discount[a:b])) for a, b in zip(first, end)])
    return dict(first=first, end=end, yf=yf, annual=annual, period_cpr=q, rate=rate * yf,
                balance=balance, annuity=balance / pv)


def cpr_cpu(terms, gamma):
    """Vectorised NumPy fallback, advancing all contracts at a payment index together."""
    first, end = terms["first"], terms["end"]
    n = len(terms["rate"])
    out = np.empty((len(gamma), 2, n))
    raw = np.broadcast_to(terms["balance"], (len(gamma), len(first))).copy()
    for i in range(int((end - first).max())):
        active = first + i < end
        j = first[active] + i
        yf, rate, annuity = terms["yf"][j], terms["rate"][j], terms["annuity"][active]
        annual = np.minimum(1., gamma[:, None] * terms["annual"][j])
        q = np.where(yf > 0., 1. - (1. - annual) ** np.where(yf > 0., yf, 1.), terms["period_cpr"][j])
        outstanding = np.maximum(raw[:, active], 0.)
        pp = q * outstanding
        principal = np.clip(annuity - outstanding * rate, 0., np.maximum(outstanding - pp, 0.))
        out[:, 0, j] = outstanding
        out[:, 1, j] = pp + principal
        raw[:, active] = (1. + rate - q) * raw[:, active] - annuity
    return out[:, 0], out[:, 1]




def dynamic_flows(block, gamma, xp, cpr):
    terms = block["terms"]
    if block["family"] == "CPR":
        return cpr(terms, gamma)
    g = gamma[:, None]
    if block["family"] == "TDRR":
        tau = xp.minimum(1., g * terms["tau"])
        surviving = terms["balance"] * (1. - tau)
        return xp.where(terms["overnight"], terms["balance"], surviving), xp.where(
            terms["overnight"], terms["balance"] * tau, surviving)
    start, end = terms["s0"] ** g, terms["s1"] ** g
    return terms["balance"] * start, terms["balance"] * (start - end)


def geometry(rows, grid, maps, code, ccy, target=None):
    discount, project = grid.discount[ccy], grid.projection[ccy]
    pv_idx = grid.index(discount, rows["cf_end_dt"])
    if np.any(pv_idx == grid.invalid):
        raise ValueError(f"EVE dates outside the grid for {code}")
    yf = p.numbers(rows, "cf_yf")
    int_factor = p.numbers(rows, "client_rt") * p.numbers(rows, "exact_yf")
    contractual = np.divide(int_factor, yf, out=np.zeros_like(yf), where=yf > 0.)
    terms = p.rate_terms(maps, code)
    eve_proj = p.projection(rows, grid, project, eve=True)
    moving = (rows["rate_type"].eq("V").to_numpy() & ~eve_proj["locked"]) | rows["rate_type"].eq("A").to_numpy()
    ev = p.subset_projection(eve_proj, moving)
    fwd = p.numbers(rows, "fwd_rt", np.nan)[moving]
    horizon = grid.date + pd.Timedelta(days=365)
    near = rows.loc[rows["cf_start_dt"].lt(horizon)]
    near_idx = near.index.to_numpy(dtype=np.int32)
    ni = p.projection(near, grid, project, eve=False)
    variable = near["rate_type"].isin(["V", "A"]).to_numpy()
    floating = variable & ~ni["locked"]
    npj = p.subset_projection(ni, floating)
    rem = np.maximum((horizon - near["cf_end_dt"]).dt.days.to_numpy() / 365., 0.)
    renewal = near.loc[rem > 0]
    lookup = renewal.index.to_numpy(dtype=np.int32)
    anchor = grid.eve_projection          # as packed_pricing.pack_eve: single-curve = raw forwards
    n_move = int(moving.sum())
    out = dict(pv_idx=pv_idx, yf=yf, int_factor=int_factor, contract=contractual, terms=terms,
               eve_move=np.flatnonzero(moving).astype(np.int32), eve_proj=ev,
               eve_base=p.forwards(grid.values[:1], ev, np)[0] if anchor else np.zeros(n_move),
               eve_fwd=np.nan_to_num(fwd, nan=0.) if anchor else np.zeros(n_move),
               eve_base_client=p.client(fwd, terms, np), eve_admin_missing=rows["rate_type"].eq("A").to_numpy()[moving] & np.isnan(fwd),
               nii_idx=near_idx, nii_yf=yf[near_idx] * p.fe.horizon_accrual(near, grid.date, horizon),
               nii_static=np.where(variable, np.clip(contractual[near_idx], terms[4], terms[5]), contractual[near_idx]),
               nii_move=np.flatnonzero(floating).astype(np.int32), nii_proj=npj,
               nii_base_client=p.client(p.forwards(grid.values[:1], npj, np)[0], terms, np),
               ren_idx=lookup, remain=rem[rem > 0],
               ren=p.renewal_geometry(renewal, p.numbers(renewal, "renew_months", np.nan), grid, project, horizon),
               spread=maps[3].get(code, p.numbers(renewal, "margin")))
    if target is not None:
        dst, months = target
        out.update(mig=p.renewal_geometry(renewal, np.full(len(renewal), months), grid, project, horizon),
                   mig_terms=p.rate_terms(maps, dst), base_capital=(rows["balance_amt"] * (rows["s0"] - rows["s1"])).to_numpy()[lookup])
    return out


def value_flows(discount, opening, capital, g, xp, reference=None, renewal_floor=None):
    """NII (interest, renewal) and EVE (PV capital, PV interest) of regenerated flows. reference: the
    curves floating NII is anchored to (as packed_pricing.price_nii; None = the packing curve);
    renewal_floor: floor of the renewal market rate per curve (NII base at 0, shocks unfloored)."""
    terms = g["terms"]
    factor = discount[:, g["pv_idx"]]
    pv_capital = (capital * factor).sum(axis=1)
    interest = opening * g["int_factor"]
    ii = g["eve_move"]
    if ii.size:
        proj = p.forwards(discount, g["eve_proj"], xp)
        proj = xp.where(g["eve_proj"]["locked"], proj, g["eve_fwd"] + (proj - g["eve_base"]))
        delta = xp.where(g["eve_admin_missing"], 0., p.client(proj, terms, xp) - g["eve_base_client"])
        rate = xp.clip(g["contract"][ii] + delta, terms[4], terms[5])
        interest[:, ii] = opening[:, ii] * g["yf"][ii] * rate
    pv_interest = xp.nansum(interest * factor, axis=1)
    nii_rate = xp.broadcast_to(g["nii_static"], (len(discount), len(g["nii_idx"]))).copy()
    j = g["nii_move"]
    if j.size:
        base_client = (g["nii_base_client"] if reference is None
                       else p.client(p.forwards(reference, g["nii_proj"], xp), terms, xp))
        delta = p.client(p.forwards(discount, g["nii_proj"], xp), terms, xp) - base_client
        nii_rate[:, j] = xp.clip(g["nii_static"][j] + delta, terms[4], terms[5])
    nii_interest = (opening[:, g["nii_idx"]] * nii_rate * g["nii_yf"]).sum(axis=1)
    ren_mkt = p.renewal_forwards(discount, g["ren"], xp)
    if renewal_floor is not None:
        ren_mkt = xp.maximum(ren_mkt, renewal_floor[:, None])
    rate = p.client(ren_mkt, terms, xp, g["spread"])
    released = capital[:, g["ren_idx"]]
    nii_renewal = (released * g["remain"] * rate).sum(axis=1)
    if "mig" in g:
        migration = released - g["base_capital"]
        mig_mkt = p.renewal_forwards(discount, g["mig"], xp)
        if renewal_floor is not None:
            mig_mkt = xp.maximum(mig_mkt, renewal_floor[:, None])
        migrated_rate = p.client(mig_mkt, g["mig_terms"], xp)
        nii_renewal += (migration * g["remain"] * (migrated_rate - rate)).sum(axis=1)
    return xp.stack([nii_interest, nii_renewal, pv_capital, pv_interest], axis=1)


def pack_models(models, grid, currency):
    t = np.array([rb.tenor_years(m.ref_tenor) for m in models])
    date = grid.date + pd.to_timedelta(np.maximum(1, np.round(t * 365.25)), unit="D")
    idx = grid.index(grid.discount[currency], date)
    if np.any(idx == grid.invalid):
        raise ValueError("behaviour reference tenor outside the discount curve")
    return dict(tenor=t, idx=idx, base=grid.values[0, idx],
                params=np.array([[m.direction, m.deadband_bp, m.scale_bp, m.gamma_min, m.gamma_max]
                                 for m in models]))


def scenario_gamma(discount, model_id, terms, standard_bp, step, xp):
    idx = terms["idx"][model_id]
    t = terms["tenor"][model_id]
    simulated = xp.log(terms["base"][model_id] / discount[xp.arange(len(discount)), idx]) * 10_000. / t
    delta = xp.where(xp.isnan(standard_bp), simulated, standard_bp)
    direction, deadband, scale, lo, hi = terms["params"][model_id].T
    x = direction * delta
    response = (2. / np.pi) * xp.arctan(xp.maximum(xp.abs(x) - deadband, 0.) / scale)
    gamma = 1. + xp.where(x > 0., hi - 1., lo - 1.) * response
    return xp.where(xp.isnan(standard_bp), gamma, xp.rint(gamma / step) * step)


# ── inputs, packing and pricing of the behavioural book ─────────────────────────────────────────

def load_inputs(bank, engine) -> dict:
    """Contract rows and model tables of the products with a CPR / TDRR / NMD_CORE model (no stored
    gamma variants are read): fixed annuity loans with their contractual schedule, deposits, the
    baseline prepayment, NMD (IR file) and TDRR models of the report date."""
    date = pd.Timestamp(bank.report_date)
    models = rb.read_models(bank)
    loan_codes = sorted({int(m.product_code) for m in models if m.behaviour == "CPR"})
    deposit_codes = sorted({int(m.product_code) for m in models if m.behaviour in ("TDRR", "NMD_CORE")})
    t0 = perf_counter()

    def read(query, codes=None):
        statement, params = text(query), {"rd": date}
        if codes is not None:
            statement = statement.bindparams(bindparam("codes", expanding=True))
            params["codes"] = codes or [-1]
        return pd.read_sql_query(statement, engine, params=params)

    loans = read("""
        SELECT p.schedule_id, p.product_type, p.product_code, p.bs_side, p.currency, p.fixing_dt,
               p.cf_start_dt, p.cf_end_dt, p.cf_yf, p.fwd_rt, p.margin, p.d_f AS base_df,
               p.client_rt, p.con_total_pmt, p.con_outstanding, l.balance_amt,
               l.start_date, l.fixing_freq, l.dc_conv,
               DATEDIFF(month, l.start_date, l.maturity_date) AS term_months, l.rate_type
        FROM cf.products p JOIN sched.loans l ON p.schedule_id = l.schedule_id
        WHERE p.product_type = 'L' AND p.cf_end_dt > :rd AND l.product_code IN :codes
          AND l.rate_type = 'F' AND l.amort_type = 1 AND p.con_outstanding > 0
        OPTION (MAXDOP 1)
    """, loan_codes)
    deposits = read("""
        SELECT schedule_id, product_code, bs_side, currency, fixing_freq, start_date,
               maturity_date, dc_conv, b_day_conv, disc_curve, client_rt, balance_amt, rate_type
        FROM sched.deposits WHERE product_code IN :codes
    """, deposit_codes)
    return dict(bank=bank.bank_id, report_date=date, models=models, loans=loans, deposits=deposits,
                loan_model=read("SELECT product_code, tenor, prep_rate FROM bs.models_loan WHERE report_date = :rd"),
                nmd_model=read("SELECT product_code, tenor, outstanding FROM bs.models_deposit_ir WHERE report_date = :rd"),
                term_model=read("SELECT product_code, tdrr FROM bs.models_deposit_tdrr WHERE report_date = :rd"),
                load_seconds=perf_counter() - t0)


def _target(model, full_nii):
    if model.migrate_to is None:
        return None
    if model.migrate_months is not None:
        return model.migrate_to, model.migrate_months
    tgt = full_nii.loc[full_nii["product_code"].astype(str).eq(model.migrate_to)]
    bal = tgt["outstanding_bal"].fillna(0.).abs()
    if bal.sum() <= 0.:
        raise ValueError(f"no target book for migration to {model.migrate_to}")
    return model.migrate_to, float(np.average(tgt["renew_months"].fillna(0.), weights=bal))


def prepare(snapshot: dict, inputs: dict, grid, rate_maps, plan_year_ids=frozenset(), group_columns=None):
    """Split the book: (static snapshot without the behavioural schedules, dynamic blocks, samples).
    grid: a CurveGrid whose row 0 is TODAY's curve (base flows, gamma reference). Group keys are
    group_columns (default packed_pricing.GROUP); a plan_year column is set from plan_year_ids."""
    group_columns = group_columns or p.GROUP
    models = {(rb.PTYPE[m.behaviour], m.product_code): m for m in inputs["models"] if m.behaviour in MODEL_TYPES}
    deposits = deposit_frame(inputs, grid, models)
    loans = loan_frame(inputs, grid.date)
    rows = pd.concat([loans, deposits], ignore_index=True)
    if rows.empty:
        return snapshot, [], []
    rows["schedule_id"] = rows["schedule_id"].astype(str)
    rows["plan_year"] = (rows["product_type"].astype(str) + "|" + rows["schedule_id"]).isin(set(plan_year_ids))
    cols = ["product_type", "schedule_id"]
    affected = pd.MultiIndex.from_frame(rows[cols].drop_duplicates())
    static = dict(snapshot)
    for name in ("eve", "nii"):
        df = snapshot[name]
        key = pd.MultiIndex.from_arrays([df["product_type"].astype(str), df["schedule_id"].astype(str)])
        static[name] = df.loc[~key.isin(affected)]
    blocks, samples = [], []
    for key, frame in rows.groupby(group_columns, sort=True):
        frame = frame.reset_index(drop=True)
        ptype, code, side, currency = key[:4]
        model = models[(ptype, code)]
        family = model.behaviour
        if family == "CPR":
            terms = loan_terms(frame)
        elif family == "TDRR":
            terms = dict(balance=p.numbers(frame, "balance_amt"), tau=p.numbers(frame, "tau"),
                         overnight=frame["overnight"].astype(bool).to_numpy())
        else:
            terms = dict(balance=p.numbers(frame, "balance_amt"), s0=p.numbers(frame, "s0").clip(0., 1.),
                         s1=p.numbers(frame, "s1").clip(0., 1.))
        target = _target(model, snapshot["nii"])
        blocks.append(dict(key=tuple(key), family=family, model=model, terms=terms,
                           driver=pack_models([model], grid, currency),
                           geo=geometry(frame, grid, rate_maps, code, currency, target), count=len(frame),
                           schedules=int(frame["schedule_id"].nunique()), sign=1. if side == "A" else -1.))
        first = frame["schedule_id"].iloc[0]
        samples.append((blocks[-1], frame.loc[frame["schedule_id"].eq(first)].reset_index(drop=True), target))
    return static, blocks, samples


def realised_gamma(discount, block, xp):
    """Continuous gamma per curve from the move of the curve's zero rate at the reference tenor
    against today's curve (the packing grid's row 0), in bp."""
    ids = xp.zeros(len(discount), dtype=xp.int32)
    nan = xp.full(len(discount), xp.nan)
    return scenario_gamma(discount, ids, block["driver"], nan, 1.0, xp)   # step unused: no rounding


def price_dynamic(discount, blocks, group_index, n_groups, xp, cpr, reference=None, renewal_floor=None,
                  gamma_one=False, cells=2_000_000):
    """(curves, groups, 4 components) of the behavioural blocks and the gamma per (curve, block)."""
    out = xp.zeros((len(discount), n_groups, len(p.COMPONENTS)), dtype=xp.float64)
    gammas = xp.ones((len(discount), len(blocks)), dtype=xp.float64)
    for i, b in enumerate(blocks):
        g = xp.ones(len(discount), dtype=xp.float64) if gamma_one else realised_gamma(discount, b, xp)
        gammas[:, i] = g
        batch = max(1, cells // b["count"])
        for o in range(0, len(discount), batch):
            part = slice(o, o + batch)
            opening, capital = dynamic_flows(b, g[part], xp, cpr)
            value = value_flows(discount[part], opening, capital, b["geo"], xp,
                                None if reference is None else reference[part],
                                None if renewal_floor is None else renewal_floor[part])
            out[part, group_index[b["key"]]] += b["sign"] * value
    return out, gammas
