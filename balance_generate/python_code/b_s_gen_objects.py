import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import truncnorm
import QuantLib as ql
import numpy as np
import config

##### pozniej mozna przy zwyklym kodzie dodac context i tam krzywe, report date itp, to chat pokazywal jako ustawienie
##### w klasach

def get_num_of_months(period):
    if period[-1] == 'Y':
        months = float(period[:-1]) * 12
    elif period[-1] == 'M':
        months = period[:-1]
    return int(months)

def get_calendar_from_currency(curr:str) -> object:
    currency_to_calendar = {
        'PLN': ql.Poland(),
        'EUR': ql.TARGET(),
        'USD': ql.UnitedStates(ql.UnitedStates.NYSE)
    }
    return currency_to_calendar.get(curr, ql.TARGET())


def outstanding_annuity(initial_notional, maturity_months, months_passed, annual_rate):
    r = annual_rate / 12.0
    M = maturity_months
    m = months_passed
    if r == 0:
        # Degenerates to equal principal
        return round(max(initial_notional * (M - m) / M, 0), 2)
    return round(initial_notional * ((1+r)**M - (1+r)**m) / ((1+r)**M - 1), 2)

def outstanding_constant_amort(initial_notional, maturity_months, months_passed):
    monthly_principal = initial_notional / maturity_months
    outstanding = initial_notional - monthly_principal * months_passed
    return max(outstanding, 0)
def generate_loan_current_balance_amt(init_b_amt, amort_type, start_date, report_date, maturity, int_rate):
    start_date = pd.Timestamp(start_date)
    report_date = pd.Timestamp(report_date)
    n_of_months = get_num_of_months(maturity)
    n_of_months_passed = (report_date.year - start_date.year) * 12 + (report_date.month - start_date.month)
    if amort_type == 1:
        c_balance_amt = outstanding_annuity(init_b_amt, n_of_months, n_of_months_passed, int_rate)
    elif amort_type == 2:
        c_balance_amt = outstanding_constant_amort(init_b_amt, n_of_months, n_of_months_passed)
    return c_balance_amt

def generate_truncated_normal(mean, std_dev, lower_bound, upper_bound, n=1, rounding=False, round_to=1000):
    a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
    trunc_dist = truncnorm(a, b, loc=mean, scale=std_dev)
    samples = trunc_dist.rvs(n)
    if rounding:
        result = np.ceil(samples / round_to) * round_to
    else:
        result = np.round(samples, 2)
    return result

def generate_balances(full_balance_amt, mean, std_dev, lower_bound, upper_bound,
                              overfill_factor=1.15, round = None):
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

def gen_init_bal_loan(mean, std_dev, lower_bound, upper_bound, round = None):
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

def generate_random_dates(start_date, end_date, n, annual_growth=0.15, seed=None):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if seed is not None:
        np.random.seed(seed)

    business_days = pd.date_range(start, end, freq='B')  # bez weekendów
    total_years = (end - start).days / 365.25

    # Wagi rosnące co roku o 5%
    days_since_start = (business_days - start).days
    years_since_start = days_since_start / 365.25
    weights = (1 + annual_growth) ** years_since_start

    # Konwersja na numpy i normalizacja
    weights = np.array(weights)
    weights = weights / weights.sum()

    random_dates = np.random.choice(business_days, size=n, replace=True, p=weights)
    return pd.Index(random_dates).to_numpy(dtype="datetime64[D]")

def ql_date_to_pd_date(ql_date):
    return pd.Timestamp(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())

def _rescale_curr_and_init(balances, init_balances, target_sum):
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

def initial_from_current(curr_amt, amort_type, start_date, report_date, maturity, annual_rate):
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

def current_from_initial(init_amt, amort_type, start_date, report_date, maturity, annual_rate):
    """Wyznacz bieżący pozostały kapitał (current) z nominału (initial)
    dla amortyzacji: bullet=0, annuity=1, constant=2.
    """
    start_date = pd.Timestamp(start_date)
    report_date = pd.Timestamp(report_date)
    M = get_num_of_months(maturity)
    m = (report_date.year - start_date.year) * 12 + (report_date.month - start_date.month)
    m = max(0, min(m, M))  # clamp do [0, M]

    if amort_type == 0:  # bullet — cały kapitał spłacany na końcu
        return float(init_amt)

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

    return float(init_amt) * k

# === Abstrakcyjna klasa bazowa ===
class ProductGen(ABC):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type):
        self.product_name = product_name
        self.product_code = product_code
        self.bs_side = bs_side
        self.balance_amt = balance_amt
        self.currency = currency
        self.client_type_id = client_type_id
        self.rate_type = rate_type

    @abstractmethod
    def gen_set_of_transactions(self):
        pass

    @abstractmethod
    def add_parameters(self, set_of_transactions):
        pass

    @classmethod
    @abstractmethod
    def create_from_row(cls, row):
        pass

    def build_result_df(self):
        transactions = self.gen_set_of_transactions()
        return self.add_parameters(transactions)
    
    def add_ids(self, n):
        side_code = config.bs_side_map.get(self.bs_side)
        rate_code = config.rate_type_map.get(self.rate_type, "0")
        return [f"{side_code}{self.product_code:02d}{rate_code}{i:07d}" for i in range(1, n+1)]


# === NMD classes ===
class SavingAccountsGen(ProductGen):

    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type)

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        balances = generate_balances(self.balance_amt, stats['mean'], stats['std_dev'],
                                     stats['lower_bound'], stats['upper_bound'])
        return pd.DataFrame({'balance_amt': balances})
    
    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        return pd.DataFrame({
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': self.currency,
            'client_type_id': self.client_type_id,
            'rate_type': self.rate_type
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
            rate_type=row["rate_type"]
        )

 
class CurrentAccountsGen(ProductGen):

    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type)

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
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': array_of_currencies,
            'client_type_id': array_of_client_types,
            'rate_type': self.rate_type
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
            rate_type=row["rate_type"]
        )
    

# === Class TermDeposit ===
class TermDepositsGen(ProductGen):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type)
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
            'maturity_date': set_of_transactions['maturity_date']
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
            maturity=row["maturity"]
        )


# === Loan Classes ===
class LoansFixedGen(ProductGen):

    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity, amort_type):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type)
        self.maturity = maturity
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
            round_to=0.05 / 100
        )

        balances = [
            current_from_initial(init, self.amort_type, sdt, config.report_date,
                                 self.maturity, avg_rate)
            for init, sdt in zip(init_balances, starting_dates)
        ]

        init_balances = (np.ceil(np.asarray(init_balances, dtype=float) / 5000.0) * 5000.0).astype(float).tolist()

        return pd.DataFrame({
            'balance_amt': balances,
            'init_balance_amt': init_balances,
            'start_date': starting_dates,
            'maturity_date': maturity_dates,
            'margin': margins
        })
    
    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        return pd.DataFrame({
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
            'init_balance_amt': set_of_transactions['init_balance_amt'],
            'margin': set_of_transactions['margin']
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
            amort_type=row["amort_type"]
        )


class LoansFloatGen(ProductGen):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity,
                 rate_index, amort_type):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type)
        self.maturity = maturity
        self.rate_index = rate_index
        self.amort_type = amort_type
        self.cal = get_calendar_from_currency(self.currency)

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        balances = generate_balances(self.balance_amt, stats['mean'], stats['std_dev'],
                                     stats['lower_bound'], stats['upper_bound'])
        starting_dates = generate_random_dates(max(config.balance_start_date,
            ql_date_to_pd_date(ql.Date.from_date(config.report_date) - ql.Period(self.maturity))),
            config.report_date,
            len(balances))
        maturity_dates = [ql_date_to_pd_date(self.cal.advance(ql.Date.from_date(pd.Timestamp(s_dt).date()),
                                                              ql.Period(self.maturity), ql.ModifiedFollowing)) for s_dt in
                          starting_dates]
        margin_dict = config.margin_map.get(self.product_name)
        margins = generate_truncated_normal(margin_dict['mean'], margin_dict['std_dev'],
                                            margin_dict['lower_bound'], margin_dict['upper_bound'],
                                            n=len(balances), rounding=True, round_to=0.05 / 100)
        avg_rate = config.avg_rate_map.get(self.product_name, 3)/100
        initials_raw = [
            initial_from_current(cur, self.amort_type, sdt, config.report_date, self.maturity, avg_rate)
            for cur, sdt in zip(balances, starting_dates)
        ]

        # 5) ZAOKRĄGLENIE initial *W GÓRĘ* do 5000
        init_balances = (np.ceil(np.asarray(initials_raw, dtype=float) / 5000.0) * 5000.0).astype(float).tolist()

        return pd.DataFrame({'balance_amt': balances, 'init_balance_amt': init_balances,
                             'start_date': starting_dates,'maturity_date': maturity_dates, 'margin': margins})

    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        array_of_currencies = np.full(len(set_of_transactions), self.currency)
        array_of_client_types = np.full(len(set_of_transactions), self.client_type_id)
        array_of_maturities = np.full(len(set_of_transactions), self.maturity)
        array_of_indexes = np.full(len(set_of_transactions), self.rate_index)
        return pd.DataFrame({
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
            'init_balance_amt': set_of_transactions['init_balance_amt'],
            'margin': set_of_transactions['margin']
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
            rate_index=row["rate_index"]
        )


class BondsFixedGen(ProductGen):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type)
        self.maturity = maturity

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        balances = generate_balances(self.balance_amt, stats['mean'], stats['std_dev'],
                                     stats['lower_bound'], stats['upper_bound'], round=5000)
        return pd.DataFrame({'balance_amt': balances})

    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        return pd.DataFrame({
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': self.currency,
            'client_type_id': self.client_type_id,
            'rate_type': self.rate_type,
            'maturity': self.maturity
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
            maturity=row["maturity"]
        )


class BondsFloatGen(ProductGen):
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity,
                 rate_index):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type)
        self.maturity = maturity
        self.rate_index = rate_index

    def gen_set_of_transactions(self):
        stats = config.stats_dict[self.product_name][self.client_type_id]
        balances = generate_balances(self.balance_amt, stats['mean'], stats['std_dev'],
                                     stats['lower_bound'], stats['upper_bound'], round=5000)
        return pd.DataFrame({'balance_amt': balances})

    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        return pd.DataFrame({
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': self.currency,
            'client_type_id': self.client_type_id,
            'rate_type': self.rate_type,
            'maturity': self.maturity,
            'rate_index': self.rate_index
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
            rate_index=row["rate_index"]
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
    def __init__(self, product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type):
        super().__init__(product_name, product_code, bs_side, balance_amt, currency, client_type_id, rate_type)

    def gen_set_of_transactions(self):
        balances = [self.balance_amt]
        return pd.DataFrame({'balance_amt': balances})

    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        array_of_currencies = np.full(len(set_of_transactions), self.currency)
        array_of_client_types = np.full(len(set_of_transactions), self.client_type_id)
        return pd.DataFrame({
            'transaction_id': array_of_ids,
            'product_name': self.product_name,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': array_of_currencies,
            'client_type_id': array_of_client_types,
            'rate_type': self.rate_type
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
            rate_type=row["rate_type"]
        )
    

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
        "issued_bonds": IssuedBondsGen,
        "issued_stocks": IssuedStocksGen,
        "saving_account": SavingAccountsGen,
        "current_account": CurrentAccountsGen,
        "term_deposit": TermDepositsGen,
        "retained_earnings": OneRowDummyProductGen,
    }
    table_registry = {
        LoansFixedGen: "loans",
        LoansFloatGen: "loans",
        SavingAccountsGen: "deposits",
        CurrentAccountsGen: "deposits",
        TermDepositsGen: "deposits",
        BondsFixedGen: "financial_instruments",
        BondsFloatGen: "financial_instruments",
        IssuedBondsGen: "equity",
        IssuedStocksGen: "equity"
    }

    @classmethod
    def create(cls, row):
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