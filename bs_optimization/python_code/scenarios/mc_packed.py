"""mc_packed.py
============
NumPy backend of mc_by_product.run_by_product: the same curves, the same cash-flow rows, the same
product keys and the same cache layout, priced by irrbb_calc/python_code/packed_pricing.py.


Per curve (NS betas of `bank`): the home curve is rebuilt from the betas with the realistic base floor, the
six EBA shocks are applied with their floors (anchor_sot_exact.build_all_shocked_curves, same functions),
foreign curves take the market curve with the scenario's shock; NII base is floored at 0 and shocks are not
(as compute_nii_base_schedule / ase.NII_FLOOR); nii_level is the base NII anchored to today's curve when
LABBANK_NII_LEVEL_VS_TODAY=1. Curves go to the device in chunks (CHUNK curves x 7 scenarios), so memory use
stays bounded.

Behaviour (behaviour=True or env LABBANK_MC_BEHAVIOUR=1; packed backends only): the products with a CPR /
TDRR / NMD_CORE model in rate_sensitivity.xlsx get, on every (curve x scenario), the multiplier of that
curve's move at the product's reference tenor against TODAY's market curve (continuous, no grid), and their
cash flows are regenerated with it (irrbb_calc/python_code/packed_behaviour.py; the NMD outflow beyond the
baseline renews as the migration product). Off, every curve prices the stored baseline schedules, as the
pandas loop does. The production EBA SOT instead takes gamma from the nominal shock on today's curve, on
the 0.01 / 0.05 grid.

"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _d in (ROOT, ROOT / "irrbb_calc" / "python_code", ROOT / "optimize_prep" / "python_code"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import anchor_sot_exact as ase  # noqa: E402
import packed_pricing as p  # noqa: E402
import packed_behaviour as pb  # noqa: E402
from labbank_common import fx, rate_models  # noqa: E402
from labbank_common.bank_config import get_bank  # noqa: E402

BACKENDS = ("pandas", "numpy")
CHUNK = int(os.environ.get("LABBANK_MC_CHUNK", "250"))     # curves per device batch
KEYS = p.GROUP + ["plan_year"]


class _NumpyModule:
    """NumPy, with the names this module calls."""
    

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(x):
        return np.asarray(x)


def array_module(backend: str):
    if backend == "numpy":
        return _NumpyModule()
    raise ValueError(f"backend {backend!r}: packed backends are {BACKENDS[1:]}")




def behaviour_on(behaviour: bool | None = None) -> bool:
    return bool(behaviour) if behaviour is not None else os.environ.get("LABBANK_MC_BEHAVIOUR") == "1"


def snapshot(eve_df: pd.DataFrame, nii_df: pd.DataFrame) -> dict:
    """The production book (mc_by_product.load_schedules) + market, rate models and FX of the report date."""
    sql = ase.aee.irrbb_sql_setup
    bank = get_bank()
    for df in (eve_df, nii_df):
        if "plan_year" not in df.columns:
            df["plan_year"] = False
    return dict(bank=bank.bank_id, report_date=ase.REPORT_DATE, eve=eve_df, nii=nii_df,
                eve_variants=pd.DataFrame(), nii_variants=pd.DataFrame(),
                models=rate_models.load_rate_models(sql.engine, ase.REPORT_DATE),
                market=sql.load_mkt_curves(ase.REPORT_DATE), fx_to_home=fx.to_home(bank, ase.REPORT_DATE))


def curve_grid(snap: dict, betas: np.ndarray, with_today: bool) -> p.CurveGrid:
    """Discount factors per (curve x scenario) row, home curve from the betas (anchor_sot_exact's
    construction), plus today's market curve as the last row when `with_today`."""
    from ns_curve_model import DIEBOLD_LI_TAU, ns_design_matrix
    esc = p.esc
    shocks = ase.EVE_SCENARIO_IDS
    bank = get_bank(snap["bank"])
    market, home = snap["market"], bank.market.ccy
    name = bank.market.disc_curve
    currencies = set(snap["eve"]["currency"]) | set(snap["nii"]["currency"])
    month = np.arange(1, 361)
    years, days = month / 12., month * (365.25 / 12.)
    zero = np.maximum((betas @ ns_design_matrix(years, DIEBOLD_LI_TAU).T) / 10_000.,
                      esc.default_realistic_base_floor_bps(years) / 10_000.)
    nodes = pd.concat([market[market["curve_name"].ne(name)],
                       pd.DataFrame(dict(curve_name=name, n_days=days, d_f=np.exp(-zero[0] * years)))],
                      ignore_index=True)
    grid = p.CurveGrid(nodes, snap["report_date"], [p.Scenario(s) for s in shocks], currencies, single_curve=True)
    n_rows = len(betas) * len(shocks) + (1 if with_today else 0)
    values = np.empty((n_rows, grid.values.shape[1]), dtype=np.float64)
    start = grid.names.index(name) * grid.stride
    stop = start + grid.lengths[name]
    n_days = stop - start - 1
    params = esc.get_shock_params(home)
    if with_today:
        today = esc.apply_shock_to_disc_curve(market.loc[market["curve_name"].eq(name)], "base", params,
                                              base_floor_bps_fn=esc.default_realistic_base_floor_bps)
        values[-1] = grid.values[0]
        values[-1, start:stop] = esc._log_interp_daily_df(today["n_days"], today["d_f"], n_days)["d_f"]
    stop_rows = len(betas) * len(shocks)
    for j, shock in enumerate(shocks):
        shifted = zero
        if j:
            shifted = esc.floor_shocked(zero, zero + esc.shock_bps_at_tenors(years, shock, params) / 10_000., years)
        values[j:stop_rows:len(shocks)] = grid.values[j]          # foreign curves: market curve, same shock
        for target, df in zip(values[j:stop_rows:len(shocks), start:stop], np.exp(-shifted * years)):
            target[:] = esc._log_interp_daily_df(days, df, n_days)["d_f"]
    grid.values = values
    return grid


def _compose(result, level, keys, groups, fx_home, n_shocks):
    """(rows, groups, components) -> eve / nii (C, K, S) and nii_level (C, K) in the home currency."""
    values = np.zeros((result.shape[0], len(keys), 4))
    pos = {k: i for i, k in enumerate(keys)}
    col = [pos[f"{str(code).zfill(4)}|{side}" + ("|new" if new else "")] for _, code, side, _, new in groups]
    f = np.array([fx_home[ccy] for _, _, _, ccy, _ in groups])
    np.add.at(values, (slice(None), col), result * f[None, :, None])
    curves = values.reshape(-1, n_shocks, len(keys), 4).transpose(0, 2, 1, 3)
    nii, eve = curves[..., :2].sum(axis=-1), curves[..., 2:].sum(axis=-1)
    nii[:, :, 1:] -= nii[:, :, :1]
    eve[:, :, 1:] -= eve[:, :, :1]
    lvl = np.zeros((len(level), len(keys)))
    np.add.at(lvl, (slice(None), col), level[:, :, :2].sum(axis=-1) * f[None, :])
    return eve, nii, lvl


def _pack(snap, grid, eve_df, xp, backend, beh):
    """Static plan (+ behavioural blocks) on the device, with the group list they index."""
    book, blocks, cpr = snap, [], None
    if beh:
        today = copy.copy(grid)
        today.values = grid.values[-1:]                  # row 0 = today's market curve (gamma reference)
        py = eve_df["plan_year"].to_numpy(bool)
        ids = set(eve_df.loc[py, "product_type"].astype(str) + "|" + eve_df.loc[py, "schedule_id"].astype(str))
        inputs = pb.load_inputs(get_bank(), ase.aee.irrbb_sql_setup.engine)
        book, blocks, _ = pb.prepare(snap, inputs, today, rate_models.rate_maps(snap["models"]), ids, KEYS)
        cpr = pb.cpr_cpu
        print("  behaviour: " + ", ".join(f"{b['key'][1]} {b['family']} ({b['schedules']} schedules)" for b in blocks))
    static_groups, plan = p.prepare_books(book, grid, [p.Scenario("base")], group_columns=KEYS, stored_eve_base=False)
    groups = sorted(set(static_groups) | {b["key"] for b in blocks})
    gi = {g: i for i, g in enumerate(groups)}
    plan = [(m, gi[static_groups[g]], sg, ix, v) for m, g, sg, ix, v in plan]
    dev_plan = [(m, g, sg, ix, p.to_device(v, xp)) for m, g, sg, ix, v in plan]
    dev_blocks = [p.to_device(b, xp) for b in blocks]
    return groups, gi, dev_plan, dev_blocks, cpr


def run_packed(bank: pd.DataFrame, eve_df: pd.DataFrame, nii_df: pd.DataFrame, keys: list[str],
               backend: str, behaviour: bool | None = None, gamma_one: bool = False) -> dict:
    """eve / nii / nii_level per (curve, product key, scenario) as run_by_product, on `backend`.
    behaviour: rate-dependent CPR / TDRR / NMD per curve (module docstring); gamma_one keeps the
    behavioural blocks at gamma 1 -- a check that must equal behaviour off."""
    xp = array_module(backend)
    sync = lambda: None  # noqa: E731
    t0 = time.time()
    snap = snapshot(eve_df, nii_df)
    S = len(ase.EVE_SCENARIO_IDS)
    betas = bank[["beta0", "beta1", "beta2"]].to_numpy(float)
    level_today = ase.NII_LEVEL_VS_TODAY
    beh = behaviour_on(behaviour)
    eve = np.zeros((len(bank), len(keys), S)); nii = np.zeros_like(eve); lvl = np.zeros((len(bank), len(keys)))
    gam = None
    t_prep, t_price, packed = 0.0, 0.0, None
    for c0 in range(0, len(bank), CHUNK):
        b = betas[c0:c0 + CHUNK]
        n_price = len(b) * S
        t1 = time.time()
        grid = curve_grid(snap, b, with_today=level_today or beh)
        if packed is None:                        # the book is packed once; only the curves change
            packed = _pack(snap, grid, eve_df, xp, backend, beh)
        groups, gi, dev_plan, dev_blocks, cpr = packed
        ids = np.arange(n_price, dtype=np.int32)
        base_ids = ids[::S]
        floors = np.full(n_price, ase.NII_FLOOR)
        floors[base_ids] = 0.0
        regular = [(m, g, sg, xp.asarray(ids), v) for m, g, sg, _, v in dev_plan]
        level_plan = [(m, g, sg, xp.asarray(base_ids), v) for m, g, sg, _, v in dev_plan if m == 0]
        t_prep += time.time() - t1
        t1 = time.time()
        disc = xp.asarray(grid.values)
        own_base = xp.asarray(np.repeat(base_ids, S))
        res = p.price_book(disc, regular, len(groups), n_price, xp, own_base, xp.asarray(floors))
        today_ix = xp.asarray(np.full(n_price, n_price, dtype=np.int32))
        lv = p.price_book(disc, level_plan, len(groups), n_price, xp, today_ix)[::S] if level_today else None
        if dev_blocks:
            d, g = pb.price_dynamic(disc[:n_price], dev_blocks, gi, len(groups), xp, cpr, reference=disc[own_base],
                                    renewal_floor=xp.asarray(floors), gamma_one=gamma_one)
            res = res + d
            if level_today:
                dl, _ = pb.price_dynamic(disc[base_ids], dev_blocks, gi, len(groups), xp, cpr,
                                         reference=disc[today_ix[:len(base_ids)]], gamma_one=gamma_one)
                lv = lv + dl
            g = xp.asnumpy(g)
            gam = g if gam is None else np.vstack([gam, g])
        res = xp.asnumpy(res)
        lv = xp.asnumpy(lv) if lv is not None else res[::S]
        sync()
        t_price += time.time() - t1
        e, n, l = _compose(res, lv, keys, groups, snap["fx_to_home"], S)
        if not level_today:
            l = n[:, :, 0]
        eve[c0:c0 + CHUNK], nii[c0:c0 + CHUNK], lvl[c0:c0 + CHUNK] = e, n, l
        print(f"  [{min(c0 + CHUNK, len(bank))}/{len(bank)}] curves on {backend} ({t_prep + t_price:.1f}s)")
    out = dict(eve=eve, nii=nii, nii_level=lvl, t_prep=np.array([t_prep]), t_price=np.array([t_price]),
               t_total=np.array([time.time() - t0]), behaviour=np.array([beh]))
    if gam is not None:
        blocks = packed[3]
        out["gamma"] = gam.reshape(len(bank), S, len(blocks))          # curve x scenario x block
        out["gamma_blocks"] = np.array([f"{b['key'][1]}|{b['key'][2]}|{b['family']}" + ("|new" if b["key"][4] else "")
                                        for b in blocks])
    return out
