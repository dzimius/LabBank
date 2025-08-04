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


def outstanding_annuity(initial_notional, maturity_months, months_passed, annual_rate):
    r = annual_rate / 12.0
    M = maturity_months
    m = months_passed
    if r == 0:
        # Degenerates to equal principal
        return max(initial_notional * (M - m) / M, 0)
    return initial_notional * ((1+r)**M - (1+r)**m) / ((1+r)**M - 1)

def outstanding_constant_amort(initial_notional, maturity_months, months_passed):
    monthly_principal = initial_notional / maturity_months
    outstanding = initial_notional - monthly_principal * months_passed
    return max(outstanding, 0)
def generate_loan_current_balance_amt(init_b_amt, amort_type, start_date, report_date, maturity, int_rate):
    start_date = pd.Timestamp(start_date)
    report_date = pd.Timestamp(report_date)
    n_of_months_passed = get_num_of_months(maturity)
    n_of_months = (report_date.year - start_date.year) * 12 + (report_date.month - start_date.month)
    if amort_type == 1:
        c_balance_amt = outstanding_annuity(init_b_amt, n_of_months, n_of_months_passed, int_rate)
    elif amort_type == 2:
        c_balance_amt = outstanding_constant_amort(init_b_amt, n_of_months, n_of_months_passed)
    return c_balance_amt

def generate_truncated_normal(mean, std_dev, lower_bound, upper_bound, n=1, rounding=False, round_n=1000):
    a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
    trunc_dist = truncnorm(a, b, loc=mean, scale=std_dev)
    samples = trunc_dist.rvs(n)
    if rounding:
        result = np.ceil(samples / round_n) * round_n
    else:
        result = np.round(samples, 2)
    return result

def generate_random_dates(start_date, end_date, n, seed=None):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if seed is not None:
        np.random.seed(seed)
    # Generate random float days and convert to datetime
    random_days = np.random.uniform(0, (end - start).days, size=n)
    random_dates = start + pd.to_timedelta(random_days, unit="D")
    return random_dates.to_numpy(dtype="datetime64[D]")

def ql_date_to_pd_date(ql_date):
    return pd.Timestamp(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())


# === Abstrakcyjna klasa bazowa ===
class ProductGen(ABC):
    def __init__(self, product_code, bs_side, balance_amt, currency, client_type_id, rate_type):
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
    stats_dict = config.s_acc_stats_dict

    def __init__(self, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, current_rate):
        super().__init__(product_code, bs_side, balance_amt, currency, client_type_id, rate_type)
        self.current_rate = current_rate

    def gen_set_of_transactions(self):
        stats = self.stats_dict[self.client_type_id]
        mean = stats['mean']
        std_dev = stats['std_dev']
        lower_bound = stats['lower_bound']
        upper_bound = stats['upper_bound']
        n_estimated = int(self.balance_amt / mean * 1.15)
        b_amounts = generate_truncated_normal(mean, std_dev, lower_bound, upper_bound, n=n_estimated)
        cumsum = np.cumsum(b_amounts)
        idx = np.searchsorted(cumsum, self.balance_amt)
        balances = list(b_amounts[:idx])
        last_value = self.balance_amt - sum(balances)
        if last_value > 0:
            balances.append(last_value)

        return pd.DataFrame({'balance_amt': balances})
    
    def add_parameters(self, set_of_transactions):
        array_of_ids = self.add_ids(len(set_of_transactions))
        array_of_rate_types = np.full(len(set_of_transactions), self.rate_type)
        array_of_c_rates = np.full(len(set_of_transactions), self.current_rate)
        array_of_currencies = np.full(len(set_of_transactions), self.currency)
        array_of_client_types = np.full(len(set_of_transactions), self.client_type_id)
        return pd.DataFrame({
            'id': array_of_ids,
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': array_of_currencies,
            'client_type_id': array_of_client_types,
            'rate_type': array_of_rate_types,
            'current_rate': array_of_c_rates
            })
    

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            current_rate=row["current_rate"]
        )

 
class CurrentAccountsGen(ProductGen):
    stats_dict = config.c_acc_stats_dict

    def __init__(self, product_code, bs_side, balance_amt, currency, client_type_id, rate_type, current_rate):
        super().__init__(product_code, bs_side, balance_amt, currency, client_type_id, rate_type)
        self.current_rate = current_rate

    def gen_set_of_transactions(self):
        stats = self.stats_dict[self.client_type_id]
        mean = stats['mean']
        std_dev = stats['std_dev']
        lower_bound = stats['lower_bound']
        upper_bound = stats['upper_bound']
        n_estimated = int(self.balance_amt / mean * 1.15)
        b_amounts = generate_truncated_normal(mean, std_dev, lower_bound, upper_bound, n=n_estimated)
        cumsum = np.cumsum(b_amounts)
        idx = np.searchsorted(cumsum, self.balance_amt)
        balances = list(b_amounts[:idx])
        last_value = self.balance_amt - sum(balances)
        if last_value > 0:
            balances.append(last_value)
        return pd.DataFrame({'balance_amt': balances})
    
    def add_parameters(self, set_of_transactions):
        array_of_c_rates = np.full(len(set_of_transactions), self.current_rate)
        array_of_currencies = np.full(len(set_of_transactions), self.currency)
        array_of_client_types = np.full(len(set_of_transactions), self.client_type_id)
        return pd.DataFrame({
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': array_of_currencies,
            'client_type_id': array_of_client_types,
            'rate_type': self.rate_type,
            'current_rate': array_of_c_rates
            })

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            current_rate=row["current_rate"]
        )
    

# === Class TermDeposit ===
class TermDepositsGen(ProductGen):
    stats_dict = config.t_dep_stats_dict

    def __init__(self,product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity, current_rate):
        super().__init__(product_code, bs_side, balance_amt, currency, client_type_id, rate_type)
        self.maturity = maturity
        self.current_rate = current_rate

    def gen_set_of_transactions(self):
        stats = self.stats_dict[self.client_type_id]
        mean = stats['mean']
        std_dev = stats['std_dev']
        lower_bound = stats['lower_bound']
        upper_bound = stats['upper_bound']
        n_estimated = int(self.balance_amt / mean * 1.15)
        b_amounts = generate_truncated_normal(mean, std_dev, lower_bound, upper_bound, n=n_estimated,
                                              rounding = True, round_n = 500)
        cumsum = np.cumsum(b_amounts)
        idx = np.searchsorted(cumsum, self.balance_amt)
        balances = list(b_amounts[:idx])
        last_value = self.balance_amt - sum(balances)
        if last_value > 0:
            balances.append(last_value)
        starting_dates = generate_random_dates(config.report_date,
                                                        ql_date_to_pd_date(ql.Date.from_date(config.report_date) - ql.Period(self.maturity)),
                                                        len(balances))
        return pd.DataFrame({'balance_amt': balances, 'starting_date': starting_dates})
    
    def add_parameters(self, set_of_transactions):
        array_of_c_rates = np.full(len(set_of_transactions), self.current_rate)
        array_of_currencies = np.full(len(set_of_transactions), self.currency)
        array_of_client_types = np.full(len(set_of_transactions), self.client_type_id)
        array_of_maturities = np.full(len(set_of_transactions), self.maturity)
        array_of_starting_dates = generate_random_dates(config.report_date,
                                                        ql_date_to_pd_date(ql.Date.from_date(config.report_date) - ql.Period(self.maturity)),
                                                        len(set_of_transactions))
        return pd.DataFrame({
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': array_of_currencies,
            'client_type_id': array_of_client_types,
            'rate_type': self.rate_type,
            'maturity': array_of_maturities,
            'starting_date': array_of_starting_dates,
            'current_rate': array_of_c_rates,
            })

    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            maturity=row["maturity"],
            current_rate=row["current_rate"]
        )


# === Loan Classes ===
class LoansFixedGen(ProductGen):
    stats_dict = config.t_dep_stats_dict

    def __init__(self,product_code, bs_side, balance_amt, currency, client_type_id, rate_type, maturity, current_rate, amort_type):
        super().__init__(product_code, bs_side, balance_amt, currency, client_type_id, rate_type)
        self.maturity = maturity
        self.current_rate = current_rate
        self.amort_type = amort_type

    def gen_set_of_transactions(self):
        stats = self.stats_dict[self.client_type_id]
        mean = stats['mean']
        std_dev = stats['std_dev']
        lower_bound = stats['lower_bound']
        upper_bound = stats['upper_bound']
        n_estimated = int(self.balance_amt / mean * 1.15)
        b_amounts = generate_truncated_normal(mean, std_dev, lower_bound, upper_bound,
                                            n=n_estimated, rounding = True, round_n = 5000)
        cumsum = np.cumsum(b_amounts)
        idx = np.searchsorted(cumsum, self.balance_amt)
        balances = list(b_amounts[:idx])
        last_value = self.balance_amt - sum(balances)
        if last_value > 0:
            balances.append(last_value)
        starting_dates = generate_random_dates(config.report_date,
                                ql_date_to_pd_date(ql.Date.from_date(config.report_date) - ql.Period(self.maturity)),
                                len(balances))
        if self.amort_type == 0:
            init_balances = balances
        else:
            init_balances = [generate_loan_current_balance_amt(
                i_balance, self.amort_type, s_dates, config.report_date, self.maturity, self.current_rate)
                    for (i_balance, s_dates) in zip(balances, starting_dates)]
        return pd.DataFrame({'balance_amt': balances,
                             'init_balance_amt': init_balances, 'starting_date': starting_dates})
    
    def add_parameters(self, set_of_transactions):
        array_of_c_rates = np.full(len(set_of_transactions), self.current_rate)
        array_of_currencies = np.full(len(set_of_transactions), self.currency)
        array_of_client_types = np.full(len(set_of_transactions), self.client_type_id)
        array_of_maturities = np.full(len(set_of_transactions), self.maturity)
        return pd.DataFrame({
            'product_code': self.product_code,
            'bs_side': self.bs_side,
            'balance_amt': set_of_transactions['balance_amt'],
            'currency': array_of_currencies,
            'client_type_id': array_of_client_types,
            'rate_type': self.rate_type,
            'maturity': array_of_maturities,
            'starting_date': set_of_transactions['starting_date'],
            'current_rate': array_of_c_rates,
            'init_balance_amt': set_of_transactions['init_balance_amt']
            })


    @classmethod
    def create_from_row(cls, row):
        return cls(
            product_code=row["product_code"],
            bs_side=row["bs_side"],
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            rate_type=row["rate_type"],
            maturity=row["maturity"],
            current_rate=row["current_rate"],
            amort_type=row["amort_type"]
        )


class LoansFloatGen(ProductGen):
    def __init__(self, balance_amt, margin, maturity):
        super().__init__(balance_amt)
        self.margin = margin
        self.maturity = maturity

    def gen_set_of_transactions(self):
        return self.balance_amt * self.margin * self.maturity

    @classmethod
    def create_from_row(cls, row):
        return cls(
            balance_amt=row["balance_amt"],
            margin=row["margin"],
            maturity=row["maturity"]
        )


class BondsFixedGen(ProductGen):
    def __init__(self, balance_amt, margin, maturity):
        super().__init__(balance_amt)
        self.margin = margin
        self.maturity = maturity

    def gen_set_of_transactions(self):
        return self.balance_amt * self.margin * self.maturity

    @classmethod
    def create_from_row(cls, row):
        return cls(
            balance_amt=row["balance_amt"],
            margin=row["margin"],
            maturity=row["maturity"]
        )


class BondsFloatGen(ProductGen):
    def __init__(self, balance_amt, margin, maturity):
        super().__init__(balance_amt)
        self.margin = margin
        self.maturity = maturity

    def gen_set_of_transactions(self):
        return self.balance_amt * self.margin * self.maturity

    @classmethod
    def create_from_row(cls, row):
        return cls(
            balance_amt=row["balance_amt"],
            margin=row["margin"],
            maturity=row["maturity"]
        )


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
    def __init__(self, balance_amt, currency, client_type_id, interest_rate=None, term=None, margin=None, maturity=None):
        self.balance_amt = balance_amt
        self.currency = currency
        self.client_type_id = client_type_id
        self.interest_rate = interest_rate
        self.term = term
        self.margin = margin
        self.maturity = maturity

    @classmethod
    def create_from_row(cls, row):
        return cls(
            balance_amt=row["balance_amt"],
            currency=row["currency"],
            client_type_id=row["client_type_id"],
            interest_rate=row("interest_rate"),
            term=row.get("term"),
            margin=row.get("margin"),
            maturity=row.get("maturity")
        )
    

# === Factory Class ===
class ProductFactory:
    registry = {
        "mortgage_fixed": LoansFixedGen,
        "mortgage_float": LoansFloatGen,
        "cash_loan_fixed": LoansFixedGen,
        "cash_loan_float": LoansFloatGen,
        "bond_fixed": BondsFixedGen,
        "bond_float": BondsFloatGen,
        "issued_bonds": IssuedBondsGen,
        "issued_stocks": IssuedStocksGen,
        "saving_account": SavingAccountsGen,
        "current_account": CurrentAccountsGen,
        "term_deposit": TermDepositsGen,
        "retained_earnings": OneRowDummyProductGen,
    }

    @classmethod
    def create(cls, row):
        ptype = row["product_name"]
        product_class = cls.registry[ptype]
        return product_class.create_from_row(row)

    # @classmethod
    # def create(cls, row):
    #     ptype = row["prod_type"]
    #     try:
    #         product_class = cls.registry[ptype]
    #         return product_class.create_from_row(row)
    #     except KeyError:
    #         raise ValueError(f"Unknown product type: {ptype}")