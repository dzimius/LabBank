"""Credit-card assumptions shared by generation, cash flows and risk extraction.

Balances are drawn principal, not facility limits. Rates below are fractions.
Run-off excludes new spending; constant-BS NII renewals live in the IRRBB engine.
"""
from dataclasses import dataclass
import math
import numpy as np

CARD_CODES = frozenset({'2200', '2210'})


def validate_card_inputs(structure, rate_models):
    """Require explicit card pricing before generation."""
    seen = {}
    for _, row in structure.iterrows():
        if row.get('product_name') not in ('credit_card_revolver', 'credit_card_transactor'):
            continue
        code = str(int(row['product_code']))
        expected = '2210' if row['product_name'] == 'credit_card_transactor' else '2200'
        if code != expected:
            raise ValueError(f'credit card requires product_code {expected}')
        terms = CardTerms.from_row(row)
        if code in seen and seen[code] != terms:
            raise ValueError(f'card {code} must have consistent facility terms across client rows')
        seen[code] = terms
        if code not in rate_models.index:
            raise ValueError(f'credit card {code} requires an explicit interest_rt.xlsx model')
        model = rate_models.loc[code]
        if expected == '2210':
            if float(model['beta']) != 0 or float(model['client_cap']) != 0:
                raise ValueError('transactors require beta=0 and client_cap=0')
            if row['maturity'] != '1M':
                raise ValueError('transactor run-off horizon must be 1M')


def effective_risk_factors(row):
    terms = CardTerms.from_row(row)
    return dict(lcr=terms.undrawn_per_drawn * terms.undrawn_lcr,
                rsf=float(row.get('RSF', 0)) + terms.undrawn_per_drawn * terms.undrawn_rsf,
                rwa=float(row.get('rwa_weight', 0)) * terms.ead_multiplier,
                el=float(row.get('PD', 0)) * float(row.get('LGD', 0)) * terms.ead_multiplier)


@dataclass(frozen=True)
class CardTerms:
    utilization: float = 0.40
    monthly_principal_rate: float = 0.05
    undrawn_lcr: float = 0.05
    undrawn_rsf: float = 0.05
    undrawn_ccf: float = 0.40

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be a finite fraction in [0, 1]')
        if self.utilization == 0 or self.monthly_principal_rate == 0:
            raise ValueError('utilization and monthly_principal_rate must be positive')

    @classmethod
    def from_row(cls, row):
        values = {}
        for name, field in cls.__dataclass_fields__.items():
            value = row.get('card_' + name, field.default)
            values[name] = field.default if value is None or np.isnan(float(value)) else float(value)
        return cls(**values)

    @property
    def undrawn_per_drawn(self):
        return 1 / self.utilization - 1

    @property
    def ead_multiplier(self):
        return 1 + self.undrawn_per_drawn * self.undrawn_ccf


def runoff(balance, periods, monthly_principal_rate):
    """Monthly fraction-of-opening-principal repayments; terminal residual repaid.

Interest is paid separately, so the principal rate is not a minimum total-payment
percentage. The last repayment closes the explicitly chosen behavioural horizon.
"""
    CardTerms(monthly_principal_rate=monthly_principal_rate)
    if not math.isfinite(balance) or balance < 0 or periods < 1:
        raise ValueError('nonnegative finite balance and positive periods required')
    opening = balance * (1 - monthly_principal_rate) ** np.arange(periods)
    principal = opening * monthly_principal_rate
    principal[-1] = opening[-1]
    return opening, principal
