"""Packed NII / EVE pricing of the production bank's cash-flow schedules, on NumPy.

The cash-flow rows of the IRRBB book (the same frames irrbb_calc's sql_setup loaders return) are packed
once per product group into index arrays: curve positions, forward-rate geometry, renewal geometry,
rate-model terms. Pricing many curves is then array arithmetic on a (curves x curve-days) matrix of
discount factors; every function takes the array
module as ``xp``.

It reproduces irrbb_calc's nii_calc_objects.compute_nii_shocked_schedule and
eve_calc_objects.compute_eve_shocked_schedule (anchored floating / administered rates, rate-model
limits, renewal on the product tenor, NMD migration) to ~1e-14 relative. Used by
bs_optimization/python_code/scenarios/mc_packed.py (Monte Carlo backend of mc_by_product).

"""

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "irrbb_calc" / "python_code")]
import eba_shock_curves as esc  # noqa: E402
import fixing_engine as fe  # noqa: E402
from labbank_common import curves, irrbb_behaviour as behaviour, rate_behaviour, rate_models  # noqa: E402
from labbank_common.bank_config import get_bank  # noqa: E402


PRODUCT = ["product_type", "product_code"]
GROUP = PRODUCT + ["bs_side", "currency"]
COMPONENTS = ("nii_interest", "nii_renewal", "pv_capital", "pv_interest")


@dataclass(frozen=True)
class Scenario:
    name: str
    factors: tuple[float, float, float] | None = None

    def shock_bps(self, t):
        level, slope, curvature = self.factors
        short = np.exp(-np.asarray(t) / esc.SHOCK_DECAY_YEARS)
        return level + slope * (1 - 2 * short) + curvature * 4 * short * (1 - short)


def scenarios(count, seed):
    result = [Scenario(name) for name in esc.ALL_SCENARIO_IDS]
    rng = np.random.default_rng(seed)
    factors = np.clip(rng.standard_normal((max(0, count - len(result)), 3)), -3, 3) * (105, 70, 50)
    result.extend(Scenario(f"mc_{i:04}", tuple(row)) for i, row in enumerate(factors, 1))
    return result[:count]


class CurveGrid:
    def __init__(self, market, date, scenario_list, currencies, *, single_curve=False):
        self.date = date
        available = set(market["curve_name"])
        self.discount = {ccy: f"{ccy}_disc_curve" for ccy in sorted(currencies)}
        if not set(self.discount.values()) <= available:
            raise ValueError("missing discount curve for a book currency")
        self.projection = self.discount
        self.eve_projection = not single_curve
        if not single_curve and any("_fwd_curve_" in name for name in available):
            self.projection = {ccy: name for ccy, name in curves.projection_curve_map(self.discount).items()
                               if name in available}
        self.names = sorted(set(self.discount.values()) | set(self.projection.values()))
        nodes = {name: market.loc[market["curve_name"].eq(name), ["n_days", "d_f"]] for name in self.names}
        self.lengths = {name: int(frame["n_days"].max()) + 1 for name, frame in nodes.items()}
        self.stride = max(self.lengths.values())
        self.invalid = len(self.names) * self.stride
        self.values = np.full((len(scenario_list), self.invalid + 1), np.nan)
        for name, frame in nodes.items():
            days = frame["n_days"].to_numpy(dtype=float)
            df = frame["d_f"].to_numpy(dtype=float)
            t = days / 365.25
            zero = -np.log(np.clip(df, 1e-12, None)) / np.where(t > 0, t, 1.)
            params = esc.get_shock_params(esc.curve_currency(name, "EUR"))
            start = self.names.index(name) * self.stride
            stop = start + self.lengths[name]
            for i, scenario in enumerate(scenario_list):
                if scenario.factors is None:
                    shocked = esc.apply_shock_to_disc_curve(frame, scenario.name, params)["d_f"].to_numpy()
                else:
                    rates = esc.floor_shocked(zero, zero + scenario.shock_bps(t) / 10_000, t)
                    shocked = np.exp(-rates * t)
                self.values[i, start:stop] = esc._log_interp_daily_df(days, shocked, stop - start - 1)["d_f"]

    def index(self, name, dates):
        day = ((np.asarray(dates, dtype="datetime64[D]") - np.datetime64(self.date.date()))
               .astype(np.int64))
        valid = (day >= 0) & (day < self.lengths[name])
        return np.where(valid, self.names.index(name) * self.stride + day, self.invalid).astype(np.int32)

    def series(self, i):
        series = {}
        for n, name in enumerate(self.names):
            start = n * self.stride
            end = start + self.lengths[name]
            series[name] = pd.Series(self.values[i, start:end], pd.date_range(self.date, periods=end - start))
        return pd.concat(series, names=["curve_name", "node_date"]).sort_index()


def numbers(frame, column, default=0.):
    if column in frame:
        return frame[column].fillna(default).to_numpy(dtype=np.float64)
    return np.full(len(frame), default, dtype=np.float64)


def rate_terms(maps, code):
    cap, floor, beta, spread = maps
    return np.asarray([beta.get(code, 1.), spread.get(code, 0.), floor.index.get(code, -np.inf),
                       cap.index.get(code, np.inf), floor.get(code, -np.inf), cap.get(code, np.inf)])


def client(rate, terms, xp, spread=None):
    beta, b, lo, hi, floor, cap = terms
    return xp.clip(beta * xp.clip(rate, lo, hi) + (b if spread is None else spread), floor, cap)


def projection(df, grid, curve, *, eve):
    variable_fixing = df["rate_type"].eq("V") & df["fixing_dt"].notna()
    start, end, yf = fe.block_periods(df, mask=variable_fixing)
    s, e = grid.index(curve, start), grid.index(curve, end)
    yf = yf.fillna(0).to_numpy()
    valid = (s != grid.invalid) & (e != grid.invalid) & (yf > 0)
    before = df["cf_start_dt"].lt(grid.date).to_numpy()
    fwd = numbers(df, "fwd_rt", np.nan)
    if eve:
        fixed_missing = variable_fixing.to_numpy() & (grid.index(curve, df["fixing_dt"]) == grid.invalid)
        valid &= ~fixed_missing
        locked = np.where(variable_fixing, ~valid, before)
        substitute = np.where(variable_fixing, ~valid, before)
    else:
        fixed = pd.to_datetime(df["fixing_dt"])
        missing = ~valid & ~(start.lt(grid.date) | df["cf_start_dt"].lt(grid.date) | fixed.le(grid.date))
        if missing.any():
            raise ValueError(f"{int(missing.sum())} NII rows without forward-rate inputs on {curve}")
        locked = before | fixed.lt(grid.date).to_numpy()
        substitute = np.zeros(len(df), dtype=bool)
    return dict(start=s, end=e, yf=np.where(yf > 0, yf, 1.), valid=valid,
                locked=locked, fallback=np.where(substitute, np.nan_to_num(fwd, nan=0.), 0.))


def forwards(discount, proj, xp):
    s, e, yf = proj["start"], proj["end"], proj["yf"]
    ratio = xp.where(proj["valid"], discount[:, s] / discount[:, e], 1.)
    rate = (ratio - 1.) / yf
    # Compounding applies to index periods longer than a year, e.g. after a 10Y fixed period.
    long = proj["long"]
    if long.size:
        rate[:, long] = ratio[:, long] ** (1. / yf[long]) - 1.
    return xp.where(proj["valid"], rate, proj["fallback"])


def subset_projection(proj, selected):
    out = {key: value[selected] for key, value in proj.items()}
    out["long"] = np.flatnonzero(out["yf"] > 1. + 1e-9).astype(np.int32)
    return out


def renewal_geometry(rows, months, grid, curve, horizon):
    """One tenor schedule per (capital repayment date, renewed tenor)."""
    keys = pd.MultiIndex.from_arrays([rows["cf_end_dt"], np.nan_to_num(months, nan=0.)])
    ids, unique = keys.factorize(sort=False)
    dates = pd.DatetimeIndex(unique.get_level_values(0))
    mo = unique.get_level_values(1).to_numpy(dtype=float)
    remain = (horizon - dates).days.to_numpy(dtype=float) / 365.
    fixed = (mo / 12. > remain + 1e-9) if curves.RENEWAL_MODE == "tenor" else np.zeros(len(mo), bool)
    end = np.full(len(dates), horizon.to_datetime64())
    for m in np.unique(mo[fixed]):
        sel = fixed & (mo == m)
        offset = pd.DateOffset(months=int(m)) if m <= 12 else pd.DateOffset(years=round(m / 12))
        end[sel] = (dates[sel] + offset).to_numpy()
    yf = (pd.DatetimeIndex(end) - dates).days.to_numpy(dtype=float) / 365.
    start_idx = grid.index(curve, dates)
    end_idx = grid.index(curve, end)
    if np.any((start_idx == grid.invalid) | (end_idx == grid.invalid) | (yf <= 0)):
        raise ValueError(f"renewal dates outside the {curve} grid or nonpositive accrual")
    swaps = []
    for m in np.unique(mo[fixed & (mo > 12)]):
        sel = np.flatnonzero(fixed & (mo == m))
        idx = np.stack([grid.index(curve, dates[sel] + pd.DateOffset(years=k))
                        for k in range(1, round(m / 12) + 1)], axis=-1)
        if np.any(idx == grid.invalid):
            raise ValueError(f"renewal swap dates outside the {curve} grid")
        swaps.append((sel, idx))
    return dict(start=start_idx, end=end_idx, yf=yf, swaps=tuple(swaps), lookup=ids.astype(np.int32))


def renewal_forwards(discount, geometry, xp):
    start, end = discount[:, geometry["start"]], discount[:, geometry["end"]]
    rates = (start / end - 1.) / geometry["yf"]
    for pos, annual in geometry["swaps"]:
        rates[:, pos] = (start[:, pos] - end[:, pos]) / discount[:, annual].sum(axis=-1)
    return rates[:, geometry["lookup"]]


def pack_nii(df, maps, grid, code, currency, horizon):
    curve = grid.projection[currency]
    projection_all = projection(df, grid, curve, eve=False)
    yf = numbers(df, "cf_yf")
    denom = numbers(df, "outstanding_bal") * yf
    contracted = np.divide(numbers(df, "int_pmt"), denom, out=np.zeros(len(df)), where=denom != 0.)
    exposure = denom * numbers(df, "accr_frac")
    terms = rate_terms(maps, code)
    variable = df["rate_type"].isin(["V", "A"]).to_numpy()
    floating = variable & ~projection_all["locked"] & (exposure != 0.)
    proj = subset_projection(projection_all, floating)
    base_rate = forwards(grid.values[:1], proj, np)[0]
    static_rate = np.where(variable, np.clip(contracted, terms[4], terms[5]), contracted)
    remain = np.maximum((horizon - df["cf_end_dt"]).dt.days.to_numpy() / 365., 0.)
    capital = numbers(df, "capital_pmt") + numbers(df, "prepayment_pmt")
    migrated = numbers(df, "migr_capital")
    weight = (capital - migrated) * remain
    renew = weight != 0.
    margin = numbers(df, "margin")[renew]
    out = dict(proj=proj, contracted=contracted[floating], exposure=exposure[floating],
               base_client=client(base_rate, terms, np),
               constant=np.asarray(np.dot(static_rate[~floating], exposure[~floating])), terms=terms,
               ren_weight=weight[renew], ren_spread=np.full_like(margin, maps[3][code]) if code in maps[3] else margin,
               ren=renewal_geometry(df.loc[renew], numbers(df, "renew_months", np.nan)[renew], grid, curve, horizon))
    mig = (migrated * remain) != 0.
    if mig.any():
        targets = df.loc[mig, "migr_product"].unique()
        if len(targets) != 1:
            raise ValueError(f"expected one migration destination for product {code}")
        out.update(mig=renewal_geometry(df.loc[mig], numbers(df, "migr_months", np.nan)[mig], grid, curve, horizon),
                   mig_weight=(migrated * remain)[mig], mig_terms=rate_terms(maps, targets[0]))
    return out


def pack_eve(df, maps, grid, code, currency):
    discount = grid.discount[currency]
    project = grid.projection.get(currency, discount)
    d = grid.index(discount, df["cf_end_dt"])
    if np.any(d == grid.invalid):
        raise ValueError(f"EVE CF dates outside the {discount} grid for {code}")
    proj = projection(df, grid, project, eve=True)
    variable = df["rate_type"].eq("V").to_numpy()
    admin = df["rate_type"].eq("A").to_numpy()
    change = (variable & ~proj["locked"]) | admin
    capital = numbers(df, "capital_pmt") + numbers(df, "prepayment_pmt")
    interest = numbers(df, "int_pmt")
    denom = numbers(df, "outstanding_bal") * numbers(df, "cf_yf")
    contracted = np.divide(interest, denom, out=np.zeros_like(interest), where=denom != 0.)
    base = numbers(df, "base_df")
    unique_d, inverse = np.unique(d, return_inverse=True)
    static_interest = np.where(change, 0., interest)
    projected = subset_projection(proj, change)
    terms = rate_terms(maps, code)
    fwd = numbers(df, "fwd_rt", np.nan)[change]
    return dict(proj=projected, terms=terms,
                d_static=unique_d, capital=np.bincount(inverse, weights=capital),
                interest=np.bincount(inverse, weights=static_interest),
                stored_pv=np.asarray([np.dot(capital, base), np.dot(interest, base)]),
                d_float=d[change], contracted=contracted[change], exposure=denom[change],
                fwd=np.nan_to_num(fwd, nan=0.), no_admin_ref=admin[change] & np.isnan(fwd),
                base_client=client(fwd, terms, np), base_rate=forwards(grid.values[:1], projected, np)[0],
                uses_projection=grid.eve_projection and project == grid.projection.get(currency))


def price_nii(discount, p, xp, reference=None, renewal_floor=None):
    terms = p["terms"]
    interest = xp.broadcast_to(p["constant"], (len(discount),))
    if p["exposure"].size:
        market = forwards(discount, p["proj"], xp)
        base_client = p["base_client"] if reference is None else client(forwards(reference, p["proj"], xp), terms, xp)
        delta = client(market, terms, xp) - base_client
        rate = xp.clip(p["contracted"] + delta, terms[4], terms[5])
        interest = interest + (p["exposure"] * rate).sum(axis=-1)
    renewal = xp.zeros(len(discount), dtype=xp.float64)
    if p["ren_weight"].size:
        market = renewal_forwards(discount, p["ren"], xp)
        if renewal_floor is not None:
            market = xp.maximum(market, renewal_floor[:, None])
        renewal = (p["ren_weight"] * client(market, terms, xp, p["ren_spread"])).sum(axis=-1)
    if "mig" in p:
        migrate = client(renewal_forwards(discount, p["mig"], xp), p["mig_terms"], xp)
        renewal += (p["mig_weight"] * migrate).sum(axis=-1)
    return xp.stack([interest, renewal], axis=-1)


def price_eve(discount, p, xp):
    static = discount[:, p["d_static"]]
    capital = (static * p["capital"]).sum(axis=-1)
    interest = (static * p["interest"]).sum(axis=-1)
    if p["d_float"].size:
        fwd = forwards(discount, p["proj"], xp)
        if p["uses_projection"]:
            movement = fwd - p["base_rate"]
            fwd = xp.where(p["proj"]["locked"], fwd, p["fwd"] + movement)
        terms = p["terms"]
        delta = xp.where(p["no_admin_ref"], 0., client(fwd, terms, xp) - p["base_client"])
        rate = xp.clip(p["contracted"] + delta, terms[4], terms[5])
        interest += xp.nansum(discount[:, p["d_float"]] * p["exposure"] * rate, axis=-1)
    return xp.stack([capital, interest], axis=-1)


class VariantBook:
    def __init__(self, base, variants, group_columns, horizon=None):
        self.base = base
        self.variants = variants
        self.horizon = horizon
        self.groups = base.groupby(group_columns, sort=False).indices
        self.variant_groups = {} if variants.empty else variants.groupby(group_columns, sort=False).indices
        self.targets = rate_behaviour.migration_targets() if horizon is not None else {}

    def available(self, key):
        rows = self.variant_groups.get(key)
        return np.r_[1., self.variants.iloc[rows]["gamma"].astype(float).unique()] if rows is not None else np.array([1.])

    def select(self, key, gamma):
        frame = self.base.iloc[self.groups[key]]
        if gamma == 1.:
            return frame
        variants = self.variants.iloc[self.variant_groups[key]]
        selected = variants.loc[np.isclose(variants["gamma"].astype(float), gamma)]
        scoped = variants["schedule_id"].unique()
        if selected.empty:
            raise ValueError(f"missing gamma={gamma} for {key}")
        frame = pd.concat([frame.loc[~frame["schedule_id"].isin(scoped)], selected], ignore_index=True)
        if key[1] in self.targets:
            frame = behaviour.nmd_migration(self.base, frame, {key[1]: self.targets[key[1]]})
        return frame


def gamma_plan(book, key, scenario_list, sensitivities, std_gamma):
    available = book.available(key)
    if len(available) == 1:
        return {1.: np.arange(len(scenario_list), dtype=np.int32)}
    product = key[:2]
    sensitivity = sensitivities.get(product)
    selection = {}
    for i, scenario in enumerate(scenario_list):
        if scenario.factors is None:
            gamma = std_gamma[scenario.name].get(product, 1.)
            if not np.isclose(available, gamma).any():
                raise ValueError(f"production variant gamma={gamma} absent for {key} in {scenario.name}")
        elif sensitivity is not None:
            shift = scenario.shock_bps(rate_behaviour.tenor_years(sensitivity.ref_tenor))
            desired = rate_behaviour.gamma(shift, sensitivity)
            gamma = available[np.abs(available - desired).argmin()]
        else:
            gamma = 1.
        selection.setdefault(float(gamma), []).append(i)
    return {gamma: np.asarray(indices, dtype=np.int32) for gamma, indices in selection.items()}


def prepare_books(snapshot, grid, scenario_list, *, group_columns=GROUP, stored_eve_base=True):
    date = snapshot["report_date"]
    bank = get_bank(snapshot["bank"])
    horizon = date + pd.Timedelta(days=365)
    nii = VariantBook(snapshot["nii"], snapshot["nii_variants"], group_columns, horizon)
    eve = VariantBook(snapshot["eve"], snapshot["eve_variants"], group_columns)
    groups = sorted(set(nii.groups) | set(eve.groups))
    maps = rate_models.rate_maps(snapshot["models"])
    only = os.environ.get("LABBANK_RATE_BEHAVIOURS")
    enabled = set(only.upper().split(",")) if only else set(rate_behaviour.PTYPE)
    mode = rate_behaviour.mode(bank)
    sensitivities = {(rate_behaviour.PTYPE[s.behaviour], s.product_code): s
                     for s in rate_behaviour.read_models(bank) if s.behaviour in enabled & set(rate_behaviour.PTYPE)
                     and mode == "rate"}
    if mode == "rate":
        std_gamma = {s.name: rate_behaviour.scenario_gammas(s.name, bank) for s in scenario_list}
    else:
        std_gamma = {}
        for s in scenario_list:
            cpr, tdrr = behaviour.gammas_for(s.name)
            std_gamma[s.name] = {key[:2]: cpr if key[0] == "L" else tdrr
                                 for key in groups if key[0] in ("L", "D") and key[1] not in rate_behaviour.nmd_codes(bank)}
    packed = []
    for gi, key in enumerate(groups):
        _, code, side, currency = key[:4]
        for metric, book in enumerate((nii, eve)):
            if key not in book.groups:
                continue
            for gamma, indices in gamma_plan(book, key, scenario_list, sensitivities, std_gamma).items():
                frame = book.select(key, gamma)
                if metric == 0:
                    values = pack_nii(frame, maps, grid, code, currency, horizon)
                else:
                    values = pack_eve(frame, maps, grid, code, currency)
                    values["includes_base"] = stored_eve_base and indices[0] == 0
                packed.append((metric, gi, 1. if side == "A" else -1., indices, values))
    return groups, packed


def price_book(discount, plan, groups, count, xp, reference_indices=None, nii_floors=None):
    result = xp.zeros((count, groups, len(COMPONENTS)), dtype=xp.float64)
    for metric, group, sign, indices, packed in plan:
        if packed.get("includes_base", False):
            result[0, group, 2:] = sign * packed["stored_pv"]
            indices = indices[1:]
        if not len(indices):
            continue
        row_count = max(len(packed["proj"]["start"]), len(packed.get("ren_weight", ())), 1)
        batch = max(1, 1_000_000 // row_count)
        for offset in range(0, len(indices), batch):
            ii = indices[offset:offset + batch]
            curve = discount[ii]
            if metric == 0:
                reference = None if reference_indices is None else discount[reference_indices[ii]]
                floor = None if nii_floors is None else nii_floors[ii]
                value = price_nii(curve, packed, xp, reference, floor)
            else:
                value = price_eve(curve, packed, xp)
            result[ii, group, 2 * metric:2 * metric + 2] = sign * value
    return result


def limit_plan(plan, count):
    return [(metric, group, sign, indices[indices < count], p)
            for metric, group, sign, indices, p in plan if indices[0] < count]


def to_device(value, cp):
    if isinstance(value, np.ndarray):
        return cp.asarray(value)
    if isinstance(value, dict):
        return {key: to_device(item, cp) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_device(item, cp) for item in value)
    return value
