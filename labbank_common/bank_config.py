"""Per-bank configuration: one codebase, one SQL database + one input/output folder per bank.

Every bank is a frozen ``BankConfig`` built from sections (sql, market, generation).
PLN is the full definition; other banks are ``dataclasses.replace(PLN, ...)`` and only
list what differs.

Selecting the bank
------------------
``get_bank()`` returns the active bank:
  1. the explicit ``bank_id`` argument, else
  2. the ``LABBANK_BANK`` environment variable (``pln`` / ``eur``), else
  3. ``DEFAULT_BANK``.
``LABBANK_REPORT_DATE`` overrides the report date of the environment's bank (docs/04);
``LABBANK_RUN`` names the run (labbank_common/run_store.py).
Dagster passes the partition key through ``LABBANK_BANK`` to every workflow subprocess.

Folder layout
-------------
  inputs/<bank_id>/balance|derivatives|models|forecast/   hand-made inputs of a bank
  inputs/market/<ccy>/                                    market data, shared per currency
  runs/<bank_id>/<module>/    everything generated, e.g. runs/eur/irrbb_calc/eve_results.xlsx
  demo/<bank_id>/             the files LabBank needs after a clean clone (labbank_common/demo.py)
Shared dictionaries (client types, amortisation types) stay outside the bank folders.

What does NOT live here
-----------------------
- Currency conventions (calendar, day count, EBA shock sizes per currency): those are facts
  about a currency, not a bank, and stay keyed by currency where they are used.
- Code mappings that are identical for every bank (bs_side_map, rate_type_map).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_ROOT = PROJECT_ROOT / "inputs"
MARKET_ROOT = INPUTS_ROOT / "market"
RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_BANK = "eur"   # active development bank (the PLN bank stays available via LABBANK_BANK=pln)
ENV_VAR = "LABBANK_BANK"


# ── Sections ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SqlCfg:
    db_name: str
    server: str = "localhost"
    driver: str = "ODBC+Driver+17+for+SQL+Server"

    @property
    def url(self) -> str:
        return (f"mssql+pyodbc://{self.server}/{self.db_name}"
                f"?driver={self.driver}&Trusted_Connection=yes")


@dataclass(frozen=True)
class MarketCfg:
    ccy: str                 # home / reporting currency of the bank
    float_index: str         # family name of the IBOR the book reprices on (labels, IRS legs)
    # Other currencies the bank books products in (bank_data.xlsx 'currency' column). Their
    # curves / fixings come from inputs/market/<ccy>/, conversion to the home
    # currency from fx_input.xlsx -- see labbank_common/fx.py.
    foreign_ccys: tuple[str, ...] = ()

    @property
    def disc_curve(self) -> str:
        return f"{self.ccy}_disc_curve"

    @property
    def fwd_curve(self) -> str:
        return f"{self.ccy}_fwd_curve"


@dataclass(frozen=True)
class GenerationCfg:
    """Synthetic balance-sheet generation (balance_generate)."""
    total_assets: float
    balance_start_date: str
    # Seed of numpy's global RNG for the whole generation run (contract sizes, start dates,
    # margins, client links): the same seed + the same inputs = the same balance sheet.
    seed: int
    # Contract size distribution per product and client_type_id (truncated normal).
    stats_dict: dict = field(repr=False)
    # Average client rate (%) per loan product, used where no rate model applies.
    avg_rate_map: dict = field(repr=False)
    # Contract margin distribution over the index per loan product.
    margin_map: dict = field(repr=False)
    # Margin distribution for NEW originations (roll-forward new business, catalogue /
    # new-business units of the optimiser); None = margin_map. The generated book keeps margin_map.
    nb_margin_map: dict | None = field(default=None, repr=False)

    @property
    def new_business_margin_map(self) -> dict:
        return self.nb_margin_map if self.nb_margin_map is not None else self.margin_map


@dataclass(frozen=True)
class RollforwardCfg:
    """The bank is NOT generated from scratch: it is another bank's book aged to this
    bank's report_date (balance_generate/python_code/b_s_rollforward.py).

    Existing contracts keep their ids and amortise / mature / prepay; only the volume
    missing versus the growth plan is generated as new business, with start dates
    inside the roll-forward window. Market data after the base date uses no new
    information: see market_path."""
    base_bank: str           # bank whose book (SQL) and inputs are rolled forward
    growth_file: str = "growth_plan.xlsx"   # in the BASE bank's forecast folder (inputs/<base>/forecast)
    swaps_file: str = "new_swaps.xlsx"      # same folder: IRS traded inside the window (optional)
    # market data after the base date:
    #   "forwards"  -- the base curve's forwards are realised (expectations view; forwards carry
    #                  a term premium, so an upward-sloping curve "promises" rises)
    #   "unchanged" -- every quote stays at its base-date value (random-walk view)
    #   "hybrid"    -- risk-free (OIS) forwards up to 2Y, unchanged from 5Y, linear in between;
    #                  EURIBOR-ESTR spreads held (b_s_rollforward.py docstring has the backtest)
    market_path: str = "forwards"
    # equity at the new date:
    #   "plan" -- scaled to the growth plan's equity share (the plug product closes A = L + E)
    #   "pnl"  -- from the year's result (accounting/python_code/pnl_rollforward.py): share capital
    #             and reserves held, retained earnings + profit - dividends, the cash the plan does
    #             not use at the central bank (b_s_rollforward.pnl_equity_targets)
    equity: str = "plan"


@dataclass(frozen=True)
class BankConfig:
    bank_id: str
    name: str
    report_date: str         # valuation / balance-sheet date, ISO 'YYYY-MM-DD'
    sql: SqlCfg
    market: MarketCfg
    generation: GenerationCfg
    rollforward: RollforwardCfg | None = None
    # behaviour under rate scenarios (labbank_common/rate_behaviour.py): baseline = no multiplier,
    # eba = the fixed RTS 2024/857 multipliers, rate = multipliers from the size of the rate move
    behaviour: str = "baseline"
    extended_reports: bool = False       # CSRBB and off-balance reports

    def _input_root(self) -> Path:
        """``inputs/<bank_id>``. A rolled-forward bank has no hand-made inputs: the roll-forward
        GENERATES them from the base bank's inputs + growth plan (grown bank_data, models re-dated,
        market data extended, swaps without matured trades), so they live with its results:
        ``runs/<bank_id>/rolled_inputs``."""
        if self.rollforward is not None:
            return RUNS_ROOT / self.bank_id / "rolled_inputs"
        return INPUTS_ROOT / self.bank_id

    def balance_dir(self) -> Path:
        """Balance-sheet structure: bank_data.xlsx."""
        return self._input_root() / "balance"

    def derivatives_dir(self) -> Path:
        """Swap book and FX deals: derivatives.xlsx."""
        return self._input_root() / "derivatives"

    def models_dir(self) -> Path:
        """Behavioural and client-rate models of the bank."""
        return self._input_root() / "models"

    def market_dir(self) -> Path:
        """Market data of the home currency, shared by every bank in that currency:
        ``inputs/market/<ccy>``. A rolled-forward bank has its own copy extended to its date."""
        if self.rollforward is not None:
            return self._input_root() / "market"
        return MARKET_ROOT / self.market.ccy.lower()

    def forecast_dir(self) -> Path:
        """``inputs/<bank_id>/forecast``: the 1-year forecast inputs of a BASE bank (growth plan,
        new swaps), read by the roll-forward of every bank rolled from it. A rolled bank that is itself
        the base of another roll-forward has no forecast folder of
        its own: it reads the chain root's (the same plan each year)."""
        return INPUTS_ROOT / chain_root(self).bank_id / "forecast"

    def output_dir(self, module: str, create: bool = True) -> Path:
        """``runs/<bank_id>/<module>``: everything generated, not in git."""
        p = RUNS_ROOT / self.bank_id / module
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p


# ── PLN bank (full definition) ────────────────────────────────────────────────

_PLN_STATS = {
    "credit_card": {
        4: {'mean': 4_000, 'std_dev': 3_000, 'lower_bound': 100, 'upper_bound': 50_000},
        6: {'mean': 4_000, 'std_dev': 3_000, 'lower_bound': 100, 'upper_bound': 50_000},
    },
    "current_account": {
        2: {'mean': 300_000, 'std_dev': 250_000, 'lower_bound': 0, 'upper_bound': 30_000_000},
        4: {'mean': 45_000,  'std_dev': 100_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        5: {'mean': 200_000, 'std_dev': 200_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
        6: {'mean': 60_000,  'std_dev': 80_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        7: {'mean': 5_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
        8: {'mean': 7_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
    },
    "saving_account": {
        2: {'mean': 300_000, 'std_dev': 250_000, 'lower_bound': 0, 'upper_bound': 30_000_000},
        4: {'mean': 45_000,  'std_dev': 100_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        5: {'mean': 200_000, 'std_dev': 200_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
        6: {'mean': 60_000,  'std_dev': 80_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        7: {'mean': 5_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
        8: {'mean': 7_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
    },
    "term_deposit": {
        2: {'mean': 300_000, 'std_dev': 250_000, 'lower_bound': 0, 'upper_bound': 30_000_000},
        4: {'mean': 20_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        5: {'mean': 200_000, 'std_dev': 150_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
        6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        7: {'mean': 5_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
        8: {'mean': 7_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 70_000_000},
        9: {'mean': 50_000_000, 'std_dev': 30_000_000, 'lower_bound': 5_000_000, 'upper_bound': 200_000_000},
    },
    "mortgage_fixed": {
        4: {'mean': 600_000,  'std_dev': 350_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        5: {'mean': 200_000, 'std_dev': 150_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
        6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
    },
    "mortgage_float": {
        4: {'mean': 600_000,  'std_dev': 350_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        5: {'mean': 200_000, 'std_dev': 150_000, 'lower_bound': 0, 'upper_bound': 50_000_000},
        6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
    },
    "cash_loan_fixed": {
        4: {'mean': 60_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
    },
    "cash_loan_float": {
        4: {'mean': 60_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
        6: {'mean': 30_000,  'std_dev': 40_000,  'lower_bound': 0, 'upper_bound': 15_000_000},
    },
    "investment_loan_fixed": {
        5: {'mean': 1_000_000, 'std_dev': 2_000_000, 'lower_bound': 300_000, 'upper_bound': 10_000_000},
        7: {'mean': 15_000_000, 'std_dev': 15_000_000, 'lower_bound': 2_000_000, 'upper_bound': 70_000_000},
    },
    "investment_loan_float": {
        5: {'mean': 1_000_000, 'std_dev': 2_000_000, 'lower_bound': 300_000, 'upper_bound': 10_000_000},
        7: {'mean': 15_000_000, 'std_dev': 15_000_000, 'lower_bound': 2_000_000, 'upper_bound': 70_000_000},
    },
    "bond_fixed": {
        1: {'mean': 50_000_000,  'std_dev': 70_000_000,  'lower_bound': 5_000_000, 'upper_bound': 150_000_000},
        2: {'mean': 2_000_000, 'std_dev': 2_000_000, 'lower_bound': 100_000, 'upper_bound': 20_000_000},
        7: {'mean': 2_000_000,  'std_dev': 3_000_000,  'lower_bound': 0, 'upper_bound': 20_000_000},
        8: {'mean': 2_000_000,  'std_dev': 3_000_000,  'lower_bound': 0, 'upper_bound': 20_000_000},
    },
    "bond_float": {
        1: {'mean': 50_000_000, 'std_dev': 70_000_000, 'lower_bound': 5_000_000, 'upper_bound': 150_000_000},
        2: {'mean': 2_000_000, 'std_dev': 2_000_000, 'lower_bound': 100_000, 'upper_bound': 20_000_000},
        7: {'mean': 2_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 20_000_000},
        8: {'mean': 2_000_000, 'std_dev': 3_000_000, 'lower_bound': 0, 'upper_bound': 20_000_000},
    },
    "issued_bond": {
        8: {'mean': 50_000_000, 'std_dev': 40_000_000, 'lower_bound': 10_000_000, 'upper_bound': 200_000_000},
    },
    "t_bill": {
        1: {'mean': 20_000_000, 'std_dev': 15_000_000, 'lower_bound': 2_000_000, 'upper_bound': 60_000_000},
    },
}

_PLN_AVG_RATE = {
    "mortgage_fixed": 6,
    "mortgage_float": 5.8,
    "cash_loan_fixed": 8.5,
    "cash_loan_float": 8.9,
    "investment_loan_fixed": 5.5,
    "investment_loan_float": 5.9,
}

_PLN_MARGIN = {
    "mortgage_fixed":        {'mean': 2/100,   'std_dev': 0.5/100, 'lower_bound': 0.25/100, 'upper_bound': 4/100},
    "mortgage_float":        {'mean': 2/100,   'std_dev': 0.5/100, 'lower_bound': 0.25/100, 'upper_bound': 4/100},
    "cash_loan_fixed":       {'mean': 6/100,   'std_dev': 0.5/100, 'lower_bound': 4/100,    'upper_bound': 11/100},
    "cash_loan_float":       {'mean': 6.5/100, 'std_dev': 0.5/100, 'lower_bound': 4/100,    'upper_bound': 11/100},
    "investment_loan_fixed": {'mean': 1/100,   'std_dev': 1/100,   'lower_bound': 0.5/100,  'upper_bound': 5/100},
    "investment_loan_float": {'mean': 1/100,   'std_dev': 1/100,   'lower_bound': 0.5/100,  'upper_bound': 5/100},
}

PLN = BankConfig(
    bank_id="pln",
    name="LabBank PLN",
    report_date="2026-06-30",
    sql=SqlCfg(db_name="bank_gen_pln"),
    market=MarketCfg(ccy="PLN", float_index="WIBOR"),
    generation=GenerationCfg(
        total_assets=10_000_000_000,
        balance_start_date="2015-01-01",
        seed=20260630,
        stats_dict=_PLN_STATS,
        avg_rate_map=_PLN_AVG_RATE,
        margin_map=_PLN_MARGIN,
    ),
)


# ── EUR bank (differences from PLN only) ──────────────────────────────────────
# Contract-size distributions are still the PLN copy. Margins follow euro-area pricing:
# ECB MIR new-business rates (Oct-2025..Sep-2026) minus the matching EURIBOR / EUR IRS
# rate -- floating mortgages ~+1.4pp, 1-10Y fixed mortgages ~+0.65pp, consumer credit
# +4.1..5.1pp, NFC loans >1M ~+1.1pp (Poland: ~2pp mortgages, 6-6.5pp cash loans).
# The product mix (85% fixed mortgages, 10Y fixation, deposit split, covered bond)
# lives in inputs/eur/balance/bank_data.xlsx.

_EUR_AVG_RATE = {   # amortisation approximation only (initial <-> current balance)
    "mortgage_fixed": 3.4,
    "mortgage_float": 3.6,
    "cash_loan_fixed": 7.0,
    "cash_loan_float": 7.0,
    "investment_loan_fixed": 3.3,
    "investment_loan_float": 3.3,
}

# Contract margins over the index, whole book and new business (2026-10-25 German-market calibration, before
# funding, credit, operating and capital cost; fixed/floating mortgage gap 30bp, no fixed/floating difference
# on consumer and corporate loans). Until 2026-10-25: mortgage fixed 0.70 (0.30-1.50), consumer 4.50 / 4.70
# (3.00-7.00), investment 1.10 (0.50-2.50).
_EUR_MARGIN = {
    "mortgage_fixed":        {'mean': 1.1/100, 'std_dev': 0.25/100, 'lower_bound': 0.6/100,  'upper_bound': 1.8/100},
    "mortgage_float":        {'mean': 1.4/100, 'std_dev': 0.3/100,  'lower_bound': 0.9/100,  'upper_bound': 2.2/100},
    "cash_loan_fixed":       {'mean': 5.5/100, 'std_dev': 1.0/100,  'lower_bound': 3.0/100,  'upper_bound': 8.5/100},
    "cash_loan_float":       {'mean': 5.5/100, 'std_dev': 1.0/100,  'lower_bound': 3.0/100,  'upper_bound': 8.5/100},
    "investment_loan_fixed": {'mean': 1.5/100, 'std_dev': 0.45/100, 'lower_bound': 0.75/100, 'upper_bound': 3.0/100},
    "investment_loan_float": {'mean': 1.5/100, 'std_dev': 0.45/100, 'lower_bound': 0.75/100, 'upper_bound': 3.0/100},
}
_EUR_NB_MARGIN = _EUR_MARGIN      # new originations: the same calibration (a separate map is possible)

EUR = replace(
    PLN,
    bank_id="eur",
    name="LabBank EUR",
    report_date="2026-09-30",
    sql=SqlCfg(db_name="bank_gen_eur"),
    # single currency since 2026-11-05 (the USD corporate products moved to EUR corporate products)
    market=MarketCfg(ccy="EUR", float_index="EURIBOR"),
    generation=replace(PLN.generation, seed=735191, avg_rate_map=_EUR_AVG_RATE, margin_map=_EUR_MARGIN,
                       nb_margin_map=_EUR_NB_MARGIN),
)


# EUR book one year later: today's contracts aged to 2027-09-30 + one year of new business
# (forecast inputs: inputs/eur/forecast/growth_plan.xlsx + new_swaps.xlsx). Total assets are set by
# the roll-forward from the growth plan, not by generation.total_assets.
EUR_Y1 = replace(
    EUR,
    bank_id="eur_y1",
    name="LabBank EUR +1Y",
    report_date="2027-09-30",
    sql=SqlCfg(db_name="bank_gen_eur_y1"),
    generation=replace(EUR.generation, seed=735192),
    rollforward=RollforwardCfg(base_bank="eur", market_path="hybrid"),
)




BANKS: dict[str, BankConfig] = {b.bank_id: b for b in (PLN, EUR, EUR_Y1)}
                                                       


def chain_root(bank: BankConfig) -> BankConfig:
    """The generated bank at the start of a roll-forward chain (itself for a generated bank)."""
    seen = set()
    while bank.rollforward is not None:
        if bank.bank_id in seen:
            raise ValueError(f"roll-forward cycle at {bank.bank_id}")
        seen.add(bank.bank_id)
        bank = BANKS[bank.rollforward.base_bank]
    return bank


def chain(bank: BankConfig | str) -> list[BankConfig]:
    """Root first, ``bank`` last: eur_y1 -> [eur, eur_y1]."""
    b = bank if isinstance(bank, BankConfig) else BANKS[bank]
    out = [b]
    while out[-1].rollforward is not None:
        out.append(BANKS[out[-1].rollforward.base_bank])
    return out[::-1]


DATE_ENV_VAR = "LABBANK_REPORT_DATE"


def get_bank(bank_id: str | None = None) -> BankConfig:
    """The bank, with LABBANK_REPORT_DATE applied when it is the environment's bank (docs/04 §5.1:
    the report date is a run parameter). The override never reaches another bank a workflow opens
    (a roll-forward's base, dynamic_nii's IRRBB modules)."""
    env_bank = (os.environ.get(ENV_VAR) or DEFAULT_BANK).lower()
    bank_id = (bank_id or env_bank).lower()
    try:
        bank = BANKS[bank_id]
    except KeyError:
        raise ValueError(f"Unknown bank '{bank_id}'. Known: {sorted(BANKS)}") from None
    rd = os.environ.get(DATE_ENV_VAR)
    if rd and bank_id == env_bank and rd != bank.report_date:
        import pandas as pd
        bank = replace(bank, report_date=pd.Timestamp(rd).strftime("%Y-%m-%d"))
    return bank


@lru_cache(maxsize=None)
def _engine_for(url: str):
    from sqlalchemy import create_engine
    return create_engine(url, fast_executemany=True, future=True)


def get_engine(bank: BankConfig | str | None = None):
    """SQLAlchemy engine for the bank's database (one cached engine per database)."""
    if not isinstance(bank, BankConfig):
        bank = get_bank(bank)
    return _engine_for(bank.sql.url)


def total_assets(bank: BankConfig | str | None = None) -> float:
    """Total assets of the bank's book. Generated banks: generation.total_assets. A rolled-forward
    bank's size comes from its growth plan, recorded by the roll-forward in
    runs/<bank>/balance_generate/rollforward_summary.xlsx."""
    if not isinstance(bank, BankConfig):
        bank = get_bank(bank)
    if bank.rollforward is None:
        return bank.generation.total_assets
    import pandas as pd
    path = bank.output_dir("balance_generate", create=False) / "rollforward_summary.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run balance generation (roll-forward) for {bank.bank_id} first")
    s = pd.read_excel(path)
    return float(s.loc[s["bs_side"] == "A", "target"].sum())


def home_ccy() -> str:
    """Home / reporting currency of the active bank (default for single-currency calcs)."""
    return get_bank().market.ccy


def report_date():
    """Report date of the active bank as a pandas Timestamp."""
    import pandas as pd
    return pd.Timestamp(get_bank().report_date)
