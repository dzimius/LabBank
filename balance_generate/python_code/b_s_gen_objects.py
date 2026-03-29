from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.stats import truncnorm

import config

##### pozniej mozna przy zwyklym kodzie dodac context i tam krzywe, report date itp, to chat pokazywal jako ustawienie
##### w klasach

def get_num_of_months(period: str) -> int:
    if period[-1] == 'Y':
        months = float(period[:-1]) * 12
    elif period[-1] == 'M':
        months = period[:-1]
    return int(months)

def _as_py_datetimes(x: Any) -> list[dt.datetime]:
    idx = pd.DatetimeIndex(x)
    return [d.to_pydatetime().replace(tzinfo=None) for d in idx]

def as_datetime(x: Any) -> dt.datetime:
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime().replace(tzinfo=None)
    if isinstance(x, np.datetime64):
        return pd.Timestamp(x).to_pydatetime().replace(tzinfo=None)
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return dt.datetime.combine(x, dt.time())
    if isinstance(x, dt.datetime):
        return x.replace(tzinfo=None)
    return pd.to_datetime(x).to_pydatetime().replace(tzinfo=None)

def get_calendar_from_currency(curr: str) -> ql.Calendar:
    currency_to_calendar = {
        'PLN': ql.Poland(),
        'EUR': ql.TARGET(),
        'USD': ql.UnitedStates(ql.UnitedStates.NYSE)
    }
    return currency_to_calendar.get(curr, ql.TARGET())


def outstanding_annuity(initial_notional: float, maturity_months: int, months_passed: int, annual_rate: float) -> float:
    r = annual_rate / 12.0
    M = maturity_months
    m = months_passed
    if r == 0:
        # Degenerates to equal principal
        return round(max(initial_notional * (M - m) / M, 0), 2)
    return round(initial_notional * ((1+r)**M - (1+r)**m) / ((1+r)**M - 1), 2)

def outstanding_constant_amort(initial_notional: float, maturity_months: int, months_passed: int) -> float:
    monthly_principal = initial_notional / maturity_months
    outstanding = initial_notional - monthly_principal * months_passed
    return max(outstanding, 0)

def generate_loan_current_balance_amt(
    init_b_amt: float,
    amort_type: int,
    start_date: Any,
    report_date: Any,
    maturity: str,
    int_rate: float,
) -> float:
    start_date = pd.Timestamp(start_date)
    report_date = pd.Timestamp(report_date)
    n_of_months = get_num_of_months(maturity)
    n_of_months_passed = (report_date.year - start_date.year) * 12 + (report_date.month - start_date.month)
    if amort_type == 1:
        c_balance_amt = outstanding_annuity(init_b_amt, n_of_months, n_of_months_passed, int_rate)
    elif amort_type == 2:
        c_balance_amt = outstanding_constant_amort(init_b_amt, n_of_months, n_of_months_passed)
    return c_balance_amt

def generate_truncated_normal(
    mean: float,
    std_dev: float,
    lower_bound: float,
    upper_bound: float,
    n: int = 1,
    rounding: bool = False,
    round_to: float = 1000,
) -> np.ndarray:
    a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
    trunc_dist = truncnorm(a, b, loc=mean, scale=std_dev)
    samples = trunc_dist.rvs(n)
    if rounding:
        result = np.ceil(samples / round_to) * round_to
    else:
        result = np.round(samples, 2)
    return result

def generate_balances(
    full_balance_amt: float,
    mean: float,
    std_dev: float,
    lower_bound: float,
    upper_bound: float,
    overfill_factor: float = 1.15,
    round: Optional[int] = None,
) -> list[float]:
    n_estimated = int(full_balance_amt / mean * overfill_factor)
    if not round:
        b_amounts = generate_truncated_normal(mean, std_dev, lower_bound, upper_bound,
                                          n=n_estimated)
    else:
        b_amounts = generate_truncated_normal(mean, std_dev, lower_bound, upper_bound,
                                          n=n_estimated, rounding=True, round_to=round)
    cumsum = np.cumsum(b_amounts)
    idx = np.searchsorted(cumsum, full_balance_amt)
    balances = list(b_amounts[:idx])
    last_value = full_balance_amt - sum(balances)
    if last_value > 0:
        balances.append(last_value)
    return balances

def gen_init_bal_loan(
    mean: float,
    std_dev: float,
    lower_bound: float,
    upper_bound: float,
    round: Optional[int] = None,
) -> np.ndarray:
    if not round:
        b_amount = generate_truncated_normal(mean, std_dev, lower_bound, upper_bound,
                                              n=1)
    else:
        b_amount = generate_truncated_normal(mean, std_dev, lower_bound, upper_bound,
                                              n=1, rounding=True, round_to=round)
    return b_amount


# def generate_random_dates(start_date, end_date, n, seed=None):
#     start = pd.to_datetime(start_date)
#     end = pd.to_datetime(end_date)
#     if seed is not None:
#         np.random.seed(seed)
#     business_days = pd.date_range(start, end, freq='B') #without saturday and sunday
#     random_dates = np.random.choice(business_days, size=n,
#                                     replace=True)  # bez zmiany funkcjonalności (duplikaty dozwolone)
#     return pd.Index(random_dates).to_numpy(dtype="datetime64[D]")

def generate_random_dates(
    start_date: Any,
    end_date: Any,
    n: int,
    annual_growth: float = 0.15,
    seed: Optional[int] = None,
) -> np.ndarray:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if seed is not None:
        np.random.seed(seed)

    business_days = pd.date_range(start, end, freq='B')  # bez weekendów
    total_years = (end - start).days / 365.25

    # Wagi rosnące co roku o 35%
    days_since_start = (business_days - start).days
    years_since_start = days_since_start / 365.25
    weights = (1 + annual_growth) ** years_since_start

    # Konwersja na numpy i normalizacja
    weights = np.array(weights)
    weights = weights / weights.sum()

    random_dates = np.random.choice(business_days, size=n, replace=True, p=weights)
    return pd.Index(random_dates).to_numpy(dtype="datetime64[D]")

def generate_random_bond_dates(
    start_date: Any,
    end_date: Any,
    n: int,
    freq: str = "3M",
    annual_growth: float = 0.15,
    seed: Optional[int] = None,
) -> list[dt.datetime]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Normalize deprecated pandas freq aliases (e.g. "3M" -> "3ME")
    import re as _re
    freq = _re.sub(r'^(\d*)(M)$', lambda m: (m.group(1) or '') + 'ME', freq)
    freq = _re.sub(r'^(\d*)(Q)$', lambda m: (m.group(1) or '') + 'QE', freq)
    freq = _re.sub(r'^(\d*)(Y|A)$', lambda m: (m.group(1) or '') + 'YE', freq)
    raw_dates = pd.date_range(start, end, freq=freq)

    if len(raw_dates) == 0:
        raise ValueError("There is no dates to generate - first_issue_date not properly chosen.")

    def to_next_business_day(d):
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        return d

    adjusted = [to_next_business_day(d) for d in raw_dates]
    adjusted_dates = pd.DatetimeIndex(adjusted)
    adjusted_dates = adjusted_dates[adjusted_dates <= end]

    days_since_start = (adjusted_dates - start).days
    years_since_start = days_since_start / 365.25
    weights = (1 + annual_growth) ** years_since_start
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    random_dates = np.random.choice(adjusted_dates, size=n, replace=True, p=weights)
    return _as_py_datetimes(random_dates)

def ql_date_to_pd_date(ql_date: ql.Date) -> pd.Timestamp:
    return pd.Timestamp(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())

def _rescale_curr_and_init(
    balances: list[float],
    init_balances: list[float],
    target_sum: float,
) -> tuple[list[float], list[float]]:
    b = np.asarray(balances, dtype=float)
    i = np.asarray(init_balances, dtype=float)
    bsum = float(b.sum())
    if bsum <= 0:
        raise ValueError("Suma currentów <= 0 — nie ma czego skalować")
    s = float(target_sum) / bsum           # <-- kluczowe: target / sum(current)
    b *= s                                 # przeskaluj currenty
    i *= s                                 # TYM SAMYM S przeskaluj initiale
    # mikro-korekta sumy (numeryka)
    diff = float(target_sum) - float(b.sum())
    if abs(diff) > 1e-6 and len(b) > 0:
        b[-1] += diff
        i[-1] += diff * (i[-1] / b[-1])    # zachowaj proporcję initial/current na ostatniej pozycji
    return i.tolist(), b.tolist()

def initial_from_current(
    curr_amt: float,
    amort_type: int,
    start_date: Any,
    report_date: Any,
    maturity: str,
    annual_rate: float,
) -> float:
    """Wyznacz initial z current (bullet=0, annuity=1, constant=2)."""
    start_date = pd.Timestamp(start_date)
    report_date = pd.Timestamp(report_date)
    M = get_num_of_months(maturity)
    m = (report_date.year - start_date.year) * 12 + (report_date.month - start_date.month)
    m = max(0, min(m, M))  # clamp do [0,M]

    if amort_type == 0:  # bullet
        return float(curr_amt)

    if amort_type == 2:  # stała amortyzacja
        k = (M - m) / M if M > 0 else 0.0
    else:  # annuity == 1
        r = annual_rate / 12.0
        if r == 0:
            k = (M - m) / M if M > 0 else 0.0
        else:
            num = (1 + r) ** M - (1 + r) ** m
            den = (1 + r) ** M - 1.0
            k = num / den if den != 0 else 0.0

    if k <= 0:
        return float(curr_amt)  # fallback, żeby nie dzielić przez 0
    return float(curr_amt) / k

def current_from_initial(
    init_amt: float,
    amort_type: int,
    start_date: Any,
    report_date: Any,
    maturity: str,
    annual_rate: float,
) -> float:
    """Wyznacz bieżący pozostały kapitał (current) z nominału (initial)
    dla amortyzacji: bullet=0, annuity=1, constant=2.
    """
    start_date = pd.Timestamp(start_date)
    report_date = pd.Timestamp(report_date)
    M = get_num_of_months(maturity)
    m = (report_date.year - start_date.year) * 12 + (report_date.month - start_date.month)
    m = max(0, min(m, M))  # clamp do [0, M]

    if amort_type == 0:  # bullet — cały kapitał spłacany na końcu
        return round(float(init_amt),2)

    if amort_type == 2:  # stała amortyzacja (równe raty kapitałowe)
        k = (M - m) / M if M > 0 else 0.0

    else:  # annuity == 1 — raty równe (kapitał + odsetki)
        r = annual_rate / 12.0
        if r == 0:
            k = (M - m) / M if M > 0 else 0.0
        else:
            num = (1 + r) ** M - (1 + r) ** m
            den = (1 + r) ** M - 1.0
            k = num / den if den != 0 else 0.0

    return round(float(init_amt) * k, 2)

# === Abstrakcyjna klasa bazowa ===
class ProductGen(ABC):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve):
        self.product_name = product_name
        self.product_code = product_code
        self.bs_side = bs_side
        self.balance_amt = balance_amt
        self.currency = currency
        self.client_type_id = client_type_id
        self.rate_type = rate_type
        self.dc_conv = dc_conv
        self.b_day_conv = b_day_conv
        self.disc_curve = disc_curve
        self.fwd_curve = fwd_curve

    @abstractmethod
    def gen_set_of_transactions(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def add_parameters(self, set_of_transactions: pd.DataFrame) -> pd.DataFrame:
        pass

    @classmethod
    @abstractmethod
    def create_from_row(cls, row: pd.Series) -> ProductGen:
        pass

    def build_result_df(self) -> pd.DataFrame:
        transactions = self.gen_set_of_transactions()
        return self.add_parameters(transactions)

    def add_ids(self, n: int) -> list[str]:
        side_code = config.bs_side_map.get(self.bs_side)
        rate_code = config.rate_type_map.get(self.rate_type, "0")
        return [f"{side_code}{self.product_code:02d}{rate_code}{i:07d}" for i in range(1, n+1)]


# === NMD classes ===
class SavingAccountsGen(ProductGen):

    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve)

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        balances = generate_balances(self.balance_amt, stats['mean'], stats['std_dev'],
                                     stats['lower_bound'], stats['upper_bound'])
        return pd.DataFrame({'balance_amt': balances})
    
    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        return pd.DataFrame({
            'report_date': config.report_date,
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': self.currency,
            'client_type_id': self.client_type_id,
            'rate_type': self.rate_type,
            'dc_conv': self.dc_conv,
            'b_day_conv': self.b_day_conv,
            'disc_curve': self.disc_curve,
            'fwd_curve': self.fwd_curve
            })
    

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_name=row["product_name"],
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            dc_conv= row["dc_conv"],
            b_day_conv= row["b_day_conv"],
            disc_curve=row["disc_curve"],
            fwd_curve=row["fwd_curve"],
        )

 
class CurrentAccountsGen(ProductGen):

    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve)

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        balances = generate_balances(self.balance_amt, stats['mean'], stats['std_dev'],
                                     stats['lower_bound'], stats['upper_bound'])
        return pd.DataFrame({'balance_amt': balances})
    
    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        array_of_currencies = np.full(len(set_of_transactions), self.currency)
        array_of_client_types = np.full(len(set_of_transactions), self.client_type_id)
        return pd.DataFrame({
            'report_date': config.report_date,
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': array_of_currencies,
            'client_type_id': array_of_client_types,
            'rate_type': self.rate_type,
            'dc_conv': self.dc_conv,
            'b_day_conv': self.b_day_conv,
            'disc_curve': self.disc_curve,
            'fwd_curve': self.fwd_curve
            })

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_name=row["product_name"],
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            dc_conv=row["dc_conv"],
            b_day_conv=row["b_day_conv"],
            disc_curve=row["disc_curve"],
            fwd_curve=row["fwd_curve"],

        )
    

# === Class TermDeposit ===
class TermDepositsGen(ProductGen):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity,
                 dc_conv, b_day_conv, disc_curve, fwd_curve):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve)
        self.maturity = maturity
        self.cal = get_calendar_from_currency(self.currency)

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        balances = generate_balances(self.balance_amt, stats['mean'], stats['std_dev'],
                                     stats['lower_bound'], stats['upper_bound'], round=500)
        starting_dates = generate_random_dates(max(config.balance_start_date,
                                ql_date_to_pd_date(ql.Date.from_date(config.report_date) - ql.Period(self.maturity))),
                                config.report_date,
                                len(balances))
        maturity_dates = [ql_date_to_pd_date(self.cal.advance(ql.Date.from_date(pd.Timestamp(s_dt).date()),
                                                              ql.Period(self.maturity), ql.ModifiedFollowing)) for s_dt in starting_dates]
        return pd.DataFrame({'balance_amt': balances, 'start_date': starting_dates, 'maturity_date': maturity_dates})
    
    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        return pd.DataFrame({
            'report_date': config.report_date,
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': self.currency,
            'client_type_id': self.client_type_id,
            'rate_type': self.rate_type,
            'maturity': self.maturity,
            'start_date': set_of_transactions['start_date'],
            'maturity_date': set_of_transactions['maturity_date'],
            'dc_conv': self.dc_conv,
            'b_day_conv': self.b_day_conv,
            'disc_curve': self.disc_curve,
            'fwd_curve': self.fwd_curve
            })

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_name=row["product_name"],
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            maturity=row["maturity"],
            dc_conv=row["dc_conv"],
            b_day_conv=row["b_day_conv"],
            disc_curve=row["disc_curve"],
            fwd_curve=row["fwd_curve"],
        )


# === Loan Classes ===
class LoansFixedGen(ProductGen):

    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity,
                 rate_index, amort_type, fixing_freq, payment_freq, dc_conv, b_day_conv,  disc_curve, fwd_curve):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve)
        self.maturity = maturity
        self.amort_type = amort_type
        self.rate_index = rate_index
        self.fixing_freq = maturity if pd.isna(fixing_freq) else fixing_freq
        self.payment_freq = payment_freq
        self.cal = get_calendar_from_currency(self.currency)

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        avg_rate = config.avg_rate_map.get(self.product_name, 3) / 100
        overfill = 1.95
        init_balances_raw = generate_balances(
            full_balance_amt=self.balance_amt,
            mean=stats['mean'],
            std_dev=stats['std_dev'],
            lower_bound=stats['lower_bound'],
            upper_bound=stats['upper_bound'],
            overfill_factor=overfill,
            round=5000
        )

        starting_dates_raw = generate_random_dates(
            max(config.balance_start_date,
                ql_date_to_pd_date(ql.Date.from_date(config.report_date) - ql.Period(self.maturity))),
            config.report_date,
            len(init_balances_raw)
        )

        current_balances_raw = np.array([
            current_from_initial(init, self.amort_type, sdt, config.report_date,
                                 self.maturity, avg_rate)
            for init, sdt in zip(init_balances_raw, starting_dates_raw)
        ])

        cumsum = np.cumsum(current_balances_raw)
        idx = np.searchsorted(cumsum, self.balance_amt)
        init_balances = list(init_balances_raw[:idx])
        starting_dates = list(starting_dates_raw[:idx])
        last_current = self.balance_amt - sum(current_balances_raw[:idx])
        if last_current > 0:
            last_init = initial_from_current(last_current, self.amort_type, config.report_date,
                                             config.report_date, self.maturity, avg_rate)
            init_balances.append(last_init)
            starting_dates.append(config.report_date)

        maturity_dates = [
            ql_date_to_pd_date(self.cal.advance(ql.Date.from_date(pd.Timestamp(s_dt).date()),
                                                ql.Period(self.maturity),
                                                ql.ModifiedFollowing))
            for s_dt in starting_dates
        ]

        margin_dict = config.margin_map.get(self.product_name)
        margins = generate_truncated_normal(
            margin_dict['mean'], margin_dict['std_dev'],
            margin_dict['lower_bound'], margin_dict['upper_bound'],
            n=len(init_balances),
            rounding=True,
            round_to=0.10 / 100
        )

        balances = [
            current_from_initial(init, self.amort_type, sdt, config.report_date,
                                 self.maturity, avg_rate)
            for init, sdt in zip(init_balances, starting_dates)
        ]
        init_balances = (np.ceil(np.asarray(init_balances, dtype=float) / 5000.0) * 5000.0).astype(float).tolist()
        total = sum(balances)
        diff = float(self.balance_amt - total)
        if abs(diff) > 1e-6 and len(balances) > 0:
            balances[-1] += diff

        return pd.DataFrame({
            'balance_amt': balances,
            'init_balance_amt': init_balances,
            'start_date': starting_dates,
            'maturity_date': maturity_dates,
            'margin': margins
        })
    
    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        array_of_indexes = np.full(len(set_of_transactions), self.rate_index)
        array_of_fixing_freq = np.full(len(set_of_transactions), self.fixing_freq)
        array_of_payment_freq = np.full(len(set_of_transactions), self.payment_freq)
        array_of_amort_types = np.full(len(set_of_transactions), self.amort_type)
        return pd.DataFrame({
            'report_date': config.report_date,
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': self.currency,
            'client_type_id': self.client_type_id,
            'rate_type': self.rate_type,
            'maturity': self.maturity,
            'start_date': set_of_transactions['start_date'],
            'maturity_date': set_of_transactions['maturity_date'],
            'rate_index': array_of_indexes,
            'fixing_freq': array_of_fixing_freq,
            'payment_freq': array_of_payment_freq,
            'amort_type': array_of_amort_types,
            'init_balance_amt': set_of_transactions['init_balance_amt'],
            'margin': set_of_transactions['margin'],
            'dc_conv': self.dc_conv,
            'b_day_conv': self.b_day_conv,
            'disc_curve': self.disc_curve,
            'fwd_curve': self.fwd_curve
            })


    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_name=row["product_name"],
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            maturity=row["maturity"],
            fixing_freq=row["fixing_freq"],
            payment_freq=row["payment_freq"],
            rate_index=row["rate_index"],
            amort_type=row["amort_type"],
            dc_conv=row["dc_conv"],
            b_day_conv=row["b_day_conv"],
            disc_curve=row["disc_curve"],
            fwd_curve=row["fwd_curve"],
        )


class LoansFloatGen(ProductGen):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity,
                 rate_index, amort_type, fixing_freq, payment_freq,
                 dc_conv, b_day_conv, disc_curve, fwd_curve):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve)
        self.maturity = maturity
        self.rate_index = rate_index
        self.fixing_freq = maturity if pd.isna(fixing_freq) else fixing_freq
        self.payment_freq = payment_freq
        self.amort_type = amort_type
        self.cal = get_calendar_from_currency(self.currency)

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        avg_rate = config.avg_rate_map.get(self.product_name, 3) / 100
        overfill = 1.95
        init_balances_raw = generate_balances(
            full_balance_amt=self.balance_amt,
            mean=stats['mean'],
            std_dev=stats['std_dev'],
            lower_bound=stats['lower_bound'],
            upper_bound=stats['upper_bound'],
            overfill_factor=overfill,
            round=5000
        )

        starting_dates_raw = generate_random_dates(
            max(config.balance_start_date,
                ql_date_to_pd_date(ql.Date.from_date(config.report_date) - ql.Period(self.maturity))),
            config.report_date,
            len(init_balances_raw)
        )

        current_balances_raw = np.array([
            current_from_initial(init, self.amort_type, sdt, config.report_date,
                                 self.maturity, avg_rate)
            for init, sdt in zip(init_balances_raw, starting_dates_raw)
        ])

        cumsum = np.cumsum(current_balances_raw)
        idx = np.searchsorted(cumsum, self.balance_amt)
        init_balances = list(init_balances_raw[:idx])
        starting_dates = list(starting_dates_raw[:idx])
        last_current = self.balance_amt - sum(current_balances_raw[:idx])
        if last_current > 0:
            last_init = initial_from_current(last_current, self.amort_type, config.report_date,
                                             config.report_date, self.maturity, avg_rate)
            init_balances.append(last_init)
            starting_dates.append(config.report_date)

        maturity_dates = [
            ql_date_to_pd_date(self.cal.advance(ql.Date.from_date(pd.Timestamp(s_dt).date()),
                                                ql.Period(self.maturity),
                                                ql.ModifiedFollowing))
            for s_dt in starting_dates
        ]

        margin_dict = config.margin_map.get(self.product_name)
        margins = generate_truncated_normal(
            margin_dict['mean'], margin_dict['std_dev'],
            margin_dict['lower_bound'], margin_dict['upper_bound'],
            n=len(init_balances),
            rounding=True,
            round_to=0.1 / 100
        )

        balances = [
            current_from_initial(init, self.amort_type, sdt, config.report_date,
                                 self.maturity, avg_rate)
            for init, sdt in zip(init_balances, starting_dates)
        ]
        init_balances = (np.ceil(np.asarray(init_balances, dtype=float) / 5000.0) * 5000.0).astype(float).tolist()
        total = sum(balances)
        diff = float(self.balance_amt - total)
        if abs(diff) > 1e-6 and len(balances) > 0:
            balances[-1] += diff

        return pd.DataFrame({
            'balance_amt': balances,
            'init_balance_amt': init_balances,
            'start_date': starting_dates,
            'maturity_date': maturity_dates,
            'margin': margins
        })

    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        array_of_currencies = np.full(len(set_of_transactions), self.currency)
        array_of_fixing_freq = np.full(len(set_of_transactions), self.fixing_freq)
        array_of_payment_freq = np.full(len(set_of_transactions), self.payment_freq)
        array_of_client_types = np.full(len(set_of_transactions), self.client_type_id)
        array_of_maturities = np.full(len(set_of_transactions), self.maturity)
        array_of_indexes = np.full(len(set_of_transactions), self.rate_index)
        array_of_amort_types = np.full(len(set_of_transactions), self.amort_type)
        return pd.DataFrame({
            'report_date': config.report_date,
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': array_of_currencies,
            'client_type_id': array_of_client_types,
            'rate_type': self.rate_type,
            'maturity': array_of_maturities,
            'start_date': set_of_transactions['start_date'],
            'maturity_date': set_of_transactions['maturity_date'],
            'rate_index': array_of_indexes,
            'fixing_freq': array_of_fixing_freq,
            'payment_freq': array_of_payment_freq,
            'amort_type':array_of_amort_types,
            'init_balance_amt': set_of_transactions['init_balance_amt'],
            'margin': set_of_transactions['margin'],
            'dc_conv': self.dc_conv,
            'b_day_conv': self.b_day_conv,
            'disc_curve': self.disc_curve,
            'fwd_curve': self.fwd_curve
        })

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_name=row["product_name"],
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            maturity=row["maturity"],
            fixing_freq=row["fixing_freq"],
            payment_freq=row["payment_freq"],
            amort_type=row["amort_type"],
            rate_index=row["rate_index"],
            dc_conv=row["dc_conv"],
            b_day_conv=row["b_day_conv"],
            disc_curve=row["disc_curve"],
            fwd_curve=row["fwd_curve"],
        )


class BondsFixedGen(ProductGen):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 amort_type, rate_index, payment_freq, fixing_freq, maturity,
                 dc_conv, b_day_conv, disc_curve, fwd_curve):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve)
        self.maturity = maturity
        self.payment_freq = payment_freq
        self.fixing_freq = maturity if pd.isna(fixing_freq) else fixing_freq
        self.rate_index = rate_index
        self.amort_type = amort_type
        self.cal = get_calendar_from_currency(self.currency)

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        balances = generate_balances(self.balance_amt, stats['mean'], stats['std_dev'],
                                     stats['lower_bound'], stats['upper_bound'], round=5000)
        starting_dates = generate_random_bond_dates(max(config.balance_start_date,
                                                   ql_date_to_pd_date(ql.Date.from_date(config.report_date) - ql.Period(
                                                       self.maturity))),
                                               config.report_date,
                                               len(balances))
        starting_dates = [as_datetime(d) for d in starting_dates]
        maturity_dates = [ql_date_to_pd_date(self.cal.advance(ql.Date.from_date(pd.Timestamp(s_dt).date()),
                                                              ql.Period(self.maturity), ql.ModifiedFollowing)) for s_dt
                          in starting_dates]
        return pd.DataFrame({'balance_amt': balances, 'start_date': starting_dates, 'maturity_date': maturity_dates})

    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        return pd.DataFrame({
            'report_date': config.report_date,
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': self.currency,
            'client_type_id': self.client_type_id,
            'rate_type': self.rate_type,
            'start_date': set_of_transactions['start_date'],
            'maturity_date': set_of_transactions['maturity_date'],
            'maturity': self.maturity,
            'amort_type': self.amort_type,
            'rate_index': self.rate_index,
            'dc_conv': self.dc_conv,
            'fixing_freq': self.fixing_freq,
            'payment_freq': self.payment_freq,
            'b_day_conv': self.b_day_conv,
            'disc_curve': self.disc_curve,
            'fwd_curve': self.fwd_curve
        })

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_name=row["product_name"],
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            maturity=row["maturity"],
            amort_type=row["amort_type"],
            rate_index=row["rate_index"],
            dc_conv=row["dc_conv"],
            fixing_freq=row["fixing_freq"],
            payment_freq=row["payment_freq"],
            b_day_conv=row["b_day_conv"],
            disc_curve=row["disc_curve"],
            fwd_curve=row["fwd_curve"],
        )


class BondsFloatGen(ProductGen):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity,
                 amort_type, rate_index,fixing_freq, payment_freq,
                 dc_conv, b_day_conv, disc_curve, fwd_curve):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv, disc_curve, fwd_curve)
        self.maturity = maturity
        self.rate_index = rate_index
        self.amort_type = amort_type
        self.fixing_freq = maturity if pd.isna(fixing_freq) else fixing_freq
        self.payment_freq = payment_freq
        self.cal = get_calendar_from_currency(self.currency)

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        balances = generate_balances(self.balance_amt, stats['mean'], stats['std_dev'],
                                     stats['lower_bound'], stats['upper_bound'], round=5000)
        starting_dates = generate_random_bond_dates(max(config.balance_start_date,
                                                        ql_date_to_pd_date(
                                                            ql.Date.from_date(config.report_date) - ql.Period(
                                                                self.maturity))),
                                                    config.report_date,
                                                    len(balances))
        starting_dates = [as_datetime(d) for d in starting_dates]
        maturity_dates = [ql_date_to_pd_date(self.cal.advance(ql.Date.from_date(pd.Timestamp(s_dt).date()),
                                                              ql.Period(self.maturity), ql.ModifiedFollowing)) for s_dt
                          in starting_dates]
        return pd.DataFrame({'balance_amt': balances,'start_date': starting_dates, 'maturity_date': maturity_dates})

    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        return pd.DataFrame({
            'report_date': config.report_date,
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': self.currency,
            'client_type_id': self.client_type_id,
            'start_date': set_of_transactions['start_date'],
            'maturity_date': set_of_transactions['maturity_date'],
            'rate_type': self.rate_type,
            'maturity': self.maturity,
            'amort_type': self.amort_type,
            'fixing_freq':self.fixing_freq,
            'payment_freq':self.payment_freq,
            'rate_index': self.rate_index,
            'dc_conv': self.dc_conv,
            'b_day_conv': self.b_day_conv,
            'disc_curve': self.disc_curve,
            'fwd_curve': self.fwd_curve
        })

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_name=row["product_name"],
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            maturity=row["maturity"],
            amort_type=row["amort_type"],
            rate_index=row["rate_index"],
            dc_conv=row["dc_conv"],
            fixing_freq=row["fixing_freq"],
            payment_freq=row["payment_freq"],
            b_day_conv=row["b_day_conv"],
            disc_curve=row["disc_curve"],
            fwd_curve=row["fwd_curve"],
        )
#
# class LettersOfCreditGen(ProductGen):
#     def __init__(self, balance_amt, margin, maturity):
#         super().__init__(balance_amt)
#         self.margin = margin
#         self.maturity = maturity
#
#     def gen_set_of_transactions(self):
#         return self.balance_amt * self.margin * self.maturity
#
#     @classmethod
#     def create_from_row(cls, row):
#         return cls(
#             balance_amt=row["balance_amt"],
#             margin=row["margin"],
#             maturity=row["maturity"]
#         )
#
# class LoanCommitmentsGen(ProductGen):
#     def __init__(self, balance_amt, margin, maturity):
#         super().__init__(balance_amt)
#         self.margin = margin
#         self.maturity = maturity
#
#     def gen_set_of_transactions(self):
#         return self.balance_amt * self.margin * self.maturity
#
#     @classmethod
#     def create_from_row(cls, row):
#         return cls(
#             balance_amt=row["balance_amt"],
#             margin=row["margin"],
#             maturity=row["maturity"]
#         )
#
# class GuaranteesGen(ProductGen):
#     def __init__(self, balance_amt, margin, maturity):
#         super().__init__(balance_amt)
#         self.margin = margin
#         self.maturity = maturity
#
#     def gen_set_of_transactions(self):
#         return self.balance_amt * self.margin * self.maturity
#
#     @classmethod
#     def create_from_row(cls, row):
#         return cls(
#             balance_amt=row["balance_amt"],
#             margin=row["margin"],
#             maturity=row["maturity"]
#         )


class IssuedBondsGen(ProductGen):
    def __init__(self, balance_amt, interest_rate, maturity):
        super().__init__(balance_amt)
        self.interest_rate = interest_rate
        self.maturity = maturity

    def gen_set_of_transactions(self):
        return self.balance_amt * self.interest_rate * self.maturity

    @classmethod
    def create_from_row(cls, row):
        return cls(
            balance_amt=row["balance_amt"],
            interest_rate=row["interest_rate"],
            maturity=row["maturity"]
        )


class IssuedStocksGen(ProductGen):
    def __init__(self, balance_amt, dividend_rate, maturity):
        super().__init__(balance_amt)
        self.dividend_rate = dividend_rate
        self.maturity = maturity

    def gen_set_of_transactions(self):
        return self.balance_amt * self.dividend_rate * self.maturity

    @classmethod
    def create_from_row(cls, row):
        return cls(
            balance_amt=row["balance_amt"],
            dividend_rate=row["dividend_rate"],
            maturity=row["maturity"]
        )


class OneRowDummyProductGen(ProductGen):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv,disc_curve, fwd_curve):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type,
                 dc_conv, b_day_conv,disc_curve, fwd_curve)

    def gen_set_of_transactions(self):
        balances = [self.balance_amt]
        return pd.DataFrame({'balance_amt': balances})

    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        array_of_currencies = np.full(len(set_of_transactions), self.currency)
        array_of_client_types = np.full(len(set_of_transactions), self.client_type_id)
        return pd.DataFrame({
            'report_date': config.report_date,
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': array_of_currencies,
            'client_type_id': array_of_client_types,
            'rate_type': self.rate_type,
            'dc_conv': self.dc_conv,
            'b_day_conv': self.b_day_conv,
            'disc_curve': self.disc_curve,
            'fwd_curve': self.fwd_curve
        })

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_name=row["product_name"],
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            dc_conv=row["dc_conv"],
            b_day_conv=row["b_day_conv"],
            disc_curve=row["disc_curve"],
            fwd_curve=row["fwd_curve"],
        )


class CashGen(ProductGen):
    """Single-row overnight cash position (central bank reserves / vault cash).

    IRRBB treatment: cash has no interest rate sensitivity — it reprices overnight.
    Placed in the 1D bucket for both liquidity and IR gap.
      - maturity_date = next business day (overnight)
      - amort_type    = 0  (bullet: full balance returned overnight)
      - rate_type     = 'F', client_rt ≈ 0  (non-interest bearing)
    EVE impact ≈ face value (d_f ≈ 1.0 at overnight).
    Stored in financial_instruments so it flows through CF calc and IRRBB.
    """
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency,
                 client_type_id, rate_type, dc_conv, b_day_conv, disc_curve, fwd_curve):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency,
                         client_type_id, rate_type, dc_conv, b_day_conv, disc_curve, fwd_curve)
        self.cal = get_calendar_from_currency(currency)

    def gen_set_of_transactions(self):
        return pd.DataFrame({'balance_amt': [self.balance_amt]})

    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        report_dt = pd.Timestamp(config.report_date)
        maturity_dt = ql_date_to_pd_date(
            self.cal.advance(
                ql.Date.from_date(report_dt.date()),
                ql.Period(1, ql.Days),
                ql.Following,
            )
        )
        return pd.DataFrame({
            'report_date':    config.report_date,
            'transaction_id': array_of_ids,
            'product_name':   self.product_name,
            'product_code':   self.product_code,
            'bs_side':        self.bs_side,
            'balance_amt':    set_of_transactions['balance_amt'],
            'currency':       self.currency,
            'client_type_id': self.client_type_id,
            'rate_type':      self.rate_type,
            'start_date':     report_dt,
            'maturity_date':  maturity_dt,
            'maturity':       '1D',
            'amort_type':     0,
            'rate_index':     None,
            'dc_conv':        self.dc_conv,
            'fixing_freq':    '1M',
            'payment_freq':   '1M',
            'b_day_conv':     self.b_day_conv,
            'disc_curve':     self.disc_curve,
            'fwd_curve':      self.fwd_curve,
            'client_rt':      0.0,
            'index_rt':       0.0,
        })

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_name=row["product_name"],
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            dc_conv=row["dc_conv"],
            b_day_conv=row["b_day_conv"],
            disc_curve=row["disc_curve"],
            fwd_curve=row["fwd_curve"],
        )


def compute_deposit_client_rt(
    all_tx: pd.DataFrame,
    interest_rt: pd.DataFrame,
    bs_struct: pd.DataFrame,
    fixings: pd.DataFrame,
    report_date: pd.Timestamp,
) -> pd.DataFrame:
    """Add client_rt column (decimal) to deposit rows in all_tx.

    Formula per product: client_rt = a * market_rate + b/100
      - Fixed deposits (rate_type='F'):   market_rate = fixing at start_date
      - Other deposits (rate_type='A/V'): market_rate = fixing at (report_date - delay months)
      - If a=0: client_rt = b/100 (e.g. 0% for current accounts)

    interest_rt columns : product_code, a, b (in % points), delay (months)
    bs_struct columns   : product_code, rate_index (rate_index already encodes the tenor,
                          e.g. 'PLN_BID_6M'); add rate_index for any deposit product with a != 0
    fixings columns     : fixing_date, rate_index, rate (in % points), ...
                          lookup key is (fixing_date, rate_index) — tenor is embedded in rate_index
    """
    deposit_names = {'current_account', 'saving_account', 'term_deposit'}
    # Asset-side term deposits (interbank placements) are handled by compute_asset_rates
    deposit_mask = all_tx['product_name'].isin(deposit_names) & (all_tx['bs_side'] == 'L')

    all_tx = all_tx.copy()
    all_tx['index_rt']  = np.nan
    all_tx['client_rt'] = np.nan

    if not deposit_mask.any():
        return all_tx

    ir = interest_rt.copy()
    ir['product_code'] = ir['product_code'].astype(int)
    ir = ir.set_index('product_code')

    # rate_index per product_code from bank_data_only_dep (same across all client_type rows)
    # rate_index encodes both the benchmark and tenor, e.g. 'PLN_BID_6M'
    pc_to_rate_index = (
        bs_struct[['product_code', 'rate_index']]
        .dropna(subset=['rate_index'])
        .drop_duplicates('product_code')
        .set_index('product_code')['rate_index']
        .astype(str)
    )

    # Validate all deposit product codes are covered by interest_rt
    deposit_pcs = set(all_tx.loc[deposit_mask, 'product_code'].astype(int).unique())
    missing = deposit_pcs - set(ir.index)
    if missing:
        raise ValueError(
            f"product_codes missing from interest_rt.xlsx: {sorted(missing)}. "
            "Add them before running balance generation."
        )

    fix = fixings.copy()
    fix['fixing_date'] = pd.to_datetime(fix['fixing_date'])
    fix = fix.sort_values('fixing_date').reset_index(drop=True)

    def _get_fixing_rate_pct(rate_index: str, target_date: pd.Timestamp) -> float:
        """Return most recent rate (% points) for rate_index on or before target_date."""
        sub = fix[
            (fix['rate_index'] == rate_index) &
            (fix['fixing_date'] <= target_date)
        ]
        if sub.empty:
            raise ValueError(
                f"No fixing found for rate_index='{rate_index}' on or before "
                f"{target_date.date()}. Check fixing_input.xlsx."
            )
        return float(sub.iloc[-1]['rate'])

    for pc in sorted(deposit_pcs):
        row_ir = ir.loc[pc]
        a     = float(row_ir['a'])
        b_pct = float(row_ir['b'])   # percentage points, e.g. 0.5 → 0.005 in decimal
        delay = int(row_ir['delay'])

        # Floor parameters — at most one of index_floor / client_floor may be set
        index_floor = (
            float(row_ir['index_floor'])
            if 'index_floor' in row_ir.index and not pd.isna(row_ir['index_floor'])
            else None
        )
        client_floor = (
            float(row_ir['client_floor'])
            if 'client_floor' in row_ir.index and not pd.isna(row_ir['client_floor'])
            else None
        )
        if index_floor is not None and client_floor is not None:
            raise ValueError(
                f"product_code {pc}: both index_floor and client_floor are set in "
                "interest_rt.xlsx. Only one floor type is allowed per product."
            )

        def _apply_floor(mkt_rate_dec: float) -> float:
            """Apply index_floor (to input) or client_floor (to output)."""
            if index_floor is not None:
                mkt_rate_dec = max(mkt_rate_dec, index_floor)
            result = a * mkt_rate_dec + b_pct / 100.0
            if client_floor is not None:
                result = max(result, client_floor)
            return result

        pc_mask = deposit_mask & (all_tx['product_code'].astype(int) == pc)

        if a == 0.0:
            raw = b_pct / 100.0
            all_tx.loc[pc_mask, 'client_rt'] = max(raw, client_floor) if client_floor is not None else raw
            # index_rt stays NaN (no market rate dependency)
            continue

        if pc not in pc_to_rate_index.index:
            raise ValueError(
                f"product_code {pc} has a != 0 but no rate_index in bank_data_only_dep.xlsx. "
                "Add rate_index for this product."
            )
        rate_index = pc_to_rate_index[pc]

        rate_type = all_tx.loc[pc_mask, 'rate_type'].iloc[0] if pc_mask.any() else None

        if rate_type == 'F':
            # Fixed deposit: index_rt and client_rt locked at start_date fixing
            unique_dates = all_tx.loc[pc_mask, 'start_date'].dropna().unique()
            date_to_index  = {}
            date_to_client = {}
            for sd in unique_dates:
                sd_ts = pd.Timestamp(sd)
                mkt_pct = _get_fixing_rate_pct(rate_index, sd_ts)
                date_to_index[sd_ts]  = mkt_pct / 100.0
                date_to_client[sd_ts] = _apply_floor(mkt_pct / 100.0)
            all_tx.loc[pc_mask, 'index_rt'] = (
                all_tx.loc[pc_mask, 'start_date']
                .apply(lambda d: date_to_index.get(pd.Timestamp(d), np.nan) if pd.notna(d) else np.nan)
                .values
            )
            all_tx.loc[pc_mask, 'client_rt'] = (
                all_tx.loc[pc_mask, 'start_date']
                .apply(lambda d: date_to_client.get(pd.Timestamp(d), np.nan) if pd.notna(d) else np.nan)
                .values
            )
        else:
            # Variable / Admin: index_rt and client_rt from fixing at (report_date - delay)
            target_date = pd.Timestamp(report_date) - pd.DateOffset(months=delay)
            mkt_pct = _get_fixing_rate_pct(rate_index, target_date)
            all_tx.loc[pc_mask, 'index_rt']  = mkt_pct / 100.0
            all_tx.loc[pc_mask, 'client_rt'] = _apply_floor(mkt_pct / 100.0)

    return all_tx


def compute_asset_rates(
    all_tx: pd.DataFrame,
    interest_rt: pd.DataFrame,   # product_code, index_floor, ...
    bs_struct: pd.DataFrame,     # product_code, rate_type, rate_index
    fixings: pd.DataFrame,
    report_date: pd.Timestamp,
) -> pd.DataFrame:
    """Add index_rt and client_rt to loan and fin_inst rows in all_tx.

    index_rt = effective index rate for this contract (decimal):
      - Floating (rate_type='V'): fixing at report_date  (reprices with market)
      - Fixed    (rate_type='F'): fixing at start_date   (locked at origination)
    client_rt = a * max(index_rt, index_floor) + b/100
      a, b from interest_rt.xlsx (b in % points); default a=1, b=0.
    """
    asset_names = {
        'mortgage_fixed', 'mortgage_float', 'cash_loan_fixed', 'cash_loan_float',
        'investment_loan_fixed', 'investment_loan_float', 'bond_fixed', 'bond_float',
        'issued_bond', 'term_deposit',
    }
    asset_mask = all_tx['product_name'].isin(asset_names)

    all_tx = all_tx.copy()
    if 'index_rt' not in all_tx.columns:
        all_tx['index_rt'] = np.nan
    if 'client_rt' not in all_tx.columns:
        all_tx['client_rt'] = np.nan

    if not asset_mask.any():
        return all_tx

    ir = interest_rt.copy()
    ir['product_code'] = ir['product_code'].astype(int)
    ir = ir.set_index('product_code')

    # Index by (product_code, bs_side) so products that appear on both sides
    # (e.g. interbank term deposit 7900: A=placement, L=funding) resolve correctly.
    pc_bs_to_rate_index = (
        bs_struct[['product_code', 'bs_side', 'rate_index']]
        .dropna(subset=['rate_index'])
        .drop_duplicates(['product_code', 'bs_side'])
        .set_index(['product_code', 'bs_side'])['rate_index']
        .astype(str)
    )

    fix = fixings.copy()
    fix['fixing_date'] = pd.to_datetime(fix['fixing_date'])
    fix = fix.sort_values('fixing_date').reset_index(drop=True)

    def _get_fixing_dec(rate_index: str, target_date: pd.Timestamp) -> float:
        """Return fixing rate in decimal (not %) on or before target_date."""
        sub = fix[(fix['rate_index'] == rate_index) & (fix['fixing_date'] <= target_date)]
        if sub.empty:
            return np.nan
        return float(sub.iloc[-1]['rate']) / 100.0

    report_ts = pd.Timestamp(report_date)
    asset_pcs = all_tx.loc[asset_mask, 'product_code'].astype(int).unique()

    for pc in sorted(asset_pcs):
        pc_mask = asset_mask & (all_tx['product_code'].astype(int) == pc)

        # Resolve rate_index per (product_code, bs_side) — handles products on both sides
        bs_val = all_tx.loc[pc_mask, 'bs_side'].iloc[0] if pc_mask.any() else None
        key = (pc, bs_val)
        if key not in pc_bs_to_rate_index.index:
            continue  # no rate_index (e.g., cash, retained_earnings) — skip

        rate_index = pc_bs_to_rate_index[key]

        # Rate formula parameters from interest_rt (same convention as deposit products)
        index_floor = None
        a     = 1.0
        b_dec = 0.0
        if pc in ir.index:
            row_ir = ir.loc[pc]
            if 'index_floor' in row_ir.index and not pd.isna(row_ir.get('index_floor', np.nan)):
                index_floor = float(row_ir['index_floor'])
            if 'a' in row_ir.index and not pd.isna(row_ir.get('a', np.nan)):
                a = float(row_ir['a'])
            if 'b' in row_ir.index and not pd.isna(row_ir.get('b', np.nan)):
                b_dec = float(row_ir['b']) / 100.0   # % points → decimal

        def _client_rt(index_rt_dec: float) -> float:
            floored = max(index_rt_dec, index_floor) if index_floor is not None else index_rt_dec
            return a * floored + b_dec

        float_mask = pc_mask & (all_tx['rate_type'] == 'V')
        fixed_mask = pc_mask & (all_tx['rate_type'] == 'F')

        # Override per-contract margin with product-level b from interest_rt
        if 'margin' in all_tx.columns and b_dec != 0.0:
            all_tx.loc[pc_mask, 'margin'] = b_dec

        # Floating: index_rt = fixing at report_date; client_rt = a * max(index_rt, floor) + b
        if float_mask.any():
            float_val = _get_fixing_dec(rate_index, report_ts)
            all_tx.loc[float_mask, 'index_rt']  = float_val
            if not np.isnan(float_val):
                all_tx.loc[float_mask, 'client_rt'] = _client_rt(float_val)

        # Fixed: index_rt = fixing at start_date; client_rt = a * max(index_rt, floor) + b
        if fixed_mask.any():
            unique_starts = all_tx.loc[fixed_mask, 'start_date'].dropna().unique()
            start_to_rt = {}
            for sd in unique_starts:
                sd_ts = pd.Timestamp(sd)
                start_to_rt[sd_ts] = _get_fixing_dec(rate_index, sd_ts)
            index_rt_s = all_tx.loc[fixed_mask, 'start_date'].apply(
                lambda d: start_to_rt.get(pd.Timestamp(d), np.nan) if pd.notna(d) else np.nan
            )
            all_tx.loc[fixed_mask, 'index_rt']  = index_rt_s.values
            all_tx.loc[fixed_mask, 'client_rt'] = index_rt_s.apply(
                lambda rt: _client_rt(rt) if not np.isnan(rt) else np.nan
            ).values

    return all_tx


# === Factory Class ===
class ProductFactory:
    class_registry = {
        "mortgage_fixed": LoansFixedGen,
        "mortgage_float": LoansFloatGen,
        "cash_loan_fixed": LoansFixedGen,
        "cash_loan_float": LoansFloatGen,
        "investment_loan_fixed": LoansFixedGen,
        "investment_loan_float": LoansFloatGen,
        "bond_fixed": BondsFixedGen,
        "bond_float": BondsFloatGen,
        "issued_bond": BondsFloatGen,
        "cash": CashGen,
        "saving_account": SavingAccountsGen,
        "current_account": CurrentAccountsGen,
        "term_deposit": TermDepositsGen,
        "retained_earnings": OneRowDummyProductGen,
        "common_shares":     OneRowDummyProductGen,
        "risk_reserves":     OneRowDummyProductGen,
    }
    table_registry = {
        LoansFixedGen: "loans",
        LoansFloatGen: "loans",
        SavingAccountsGen: "deposits",
        CurrentAccountsGen: "deposits",
        TermDepositsGen: "deposits",
        BondsFixedGen: "financial_instruments",
        BondsFloatGen: "financial_instruments",
        CashGen: "financial_instruments",
        OneRowDummyProductGen: "equity",
    }

    @classmethod
    def create(cls, row: pd.Series) -> ProductGen:
        ptype = row["product_name"]
        product_class = cls.class_registry[ptype]
        return product_class.create_from_row(row)

    # @classmethod
    # def create(cls, row):
    #     ptype = row["prod_type"]
    #     try:
    #         product_class = cls.registry[ptype]
    #         return product_class.create_from_row(row)
    #     except KeyError:
    #         raise ValueError(f"Unknown product type: {ptype}")