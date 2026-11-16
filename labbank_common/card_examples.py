"""Complete optional bank_data/interest_rt rows; no baseline allocation changes."""
from labbank_common.bank_config import get_bank


def example_rows(bank_id='eur'):
    bank = get_bank(bank_id)
    ccy = bank.market.ccy
    base = dict(report_date=bank.report_date, bs_percentage=0.5, bs_side='A',
                product_code=2200, product_name='credit_card_revolver',
                own_name='Retail credit cards — revolvers', currency=ccy,
                client_type_id=4, rate_type='V', maturity='10Y', amort_type=3,
                payment_freq='1M', fixing_freq='1M', rate_index=f'{ccy}_ASK_1M',
                dc_conv='ACT/360' if ccy == 'EUR' else 'ACT/365',
                b_day_conv='ModifiedFollowing', disc_curve=bank.market.disc_curve,
                fwd_curve=bank.market.fwd_curve, tier_type=None, LCR=None,
                hqla_class=None, haircut=None, ASF=None, RSF=0.85,
                rwa_weight=0.75, PD=0.04, LGD=0.65, vol_elasticity=-0.003,
                fee_unit_rate=0.01, acq_cost_rate=0.01,
                card_utilization=0.40, card_monthly_principal_rate=0.05,
                card_undrawn_lcr=0.05, card_undrawn_rsf=0.05, card_undrawn_ccf=0.40)
    transactor = dict(base, product_code=2210, product_name='credit_card_transactor',
                      own_name='Retail credit cards — transactors', bs_percentage=0.25,
                      maturity='1M', card_utilization=0.25, card_monthly_principal_rate=1.0,
                      PD=0.01, fee_unit_rate=0.02)
    rate = dict(report_date=bank.report_date, product_code=2200, bs_side='A', beta=0.5,
                margin_pct=12.0 if ccy == 'EUR' else 14.0, index_floor=None,
                client_floor=0.0, index_cap=None, client_cap=0.20)
    free = dict(rate, product_code=2210, beta=0.0, margin_pct=0.0, client_cap=0.0)
    return {'bs_structure': [base, transactor], 'interest_rt': [rate, free]}


if __name__ == '__main__':
    import json
    print(json.dumps({bank: example_rows(bank) for bank in ('eur', 'pln')}, indent=2))
