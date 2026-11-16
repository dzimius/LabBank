import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from labbank_common.credit_cards import CardTerms, runoff, validate_card_inputs, effective_risk_factors
from labbank_common.card_examples import example_rows
from labbank_common.bank_config import get_bank
from labbank_common import rate_models


def load_module(path, monkeypatch, config):
    monkeypatch.setitem(sys.modules, 'config', config)
    monkeypatch.setitem(sys.modules, 'sql_setup', SimpleNamespace())
    spec = importlib.util.spec_from_file_location(path.stem + '_card_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('bank_id', ['pln', 'eur'])
@pytest.mark.parametrize('kind', [0, 1])
def test_generation_cashflows_and_behaviour(bank_id, kind, monkeypatch):
    bank = get_bank(bank_id)
    cfg = SimpleNamespace(report_date=pd.Timestamp(bank.report_date),
                          balance_start_date=pd.Timestamp('2015-01-01'),
                          stats_dict=bank.generation.stats_dict,
                          bs_side_map={'A':'1'}, rate_type_map={'V':'2'},
                          avg_rate_map=bank.generation.avg_rate_map)
    gen = load_module(ROOT/'balance_generate/python_code/b_s_gen_objects.py', monkeypatch, cfg)
    cf = load_module(ROOT/'cash_flow_calc/python_code/cf_calc_objects.py', monkeypatch, cfg)
    examples = example_rows(bank_id)
    struct = pd.DataFrame(examples['bs_structure'])
    models = rate_models._normalise(pd.DataFrame(examples['interest_rt'])).set_index('product_code')
    validate_card_inputs(struct, models)
    row = pd.Series(examples['bs_structure'][kind])
    row['balance_amt'] = 25_000.0
    generated = gen.ProductFactory.create(row).build_result_df()
    assert generated.balance_amt.sum() == pytest.approx(25_000)
    assert np.allclose(generated.credit_limit, generated.balance_amt + generated.undrawn_amt)
    params = generated.iloc[[0]].copy()
    params['schedule_id'] = 1
    # A daily valuation curve and monthly forward series covering the card horizon.
    dates = pd.date_range(cfg.report_date - pd.Timedelta(days=10), periods=3700)
    disc = pd.Series(np.exp(-0.03*np.arange(len(dates))/365),
                     index=pd.MultiIndex.from_product([[bank.market.disc_curve], dates]))
    forward = pd.Series(0.03, index=pd.MultiIndex.from_product([[bank.market.fwd_curve], ['1M'], dates]))
    fix = pd.Series(3.0, index=pd.MultiIndex.from_product([dates, [row.rate_index]]))
    sched = cf.gen_orgin_sched_loan_fin_inst(next(params.itertuples(index=False)), cfg.report_date,
                                            disc, forward, fix)
    rm = models.loc[str(row.product_code)]
    pricing = {row.product_code: dict(beta=rm.beta, spread=rm.margin_pct/100,
                                      client_floor=rm.client_floor, client_cap=rm.client_cap)}
    payments = cf.compute_amort_schedule_vectorized(sched, params, ir_params=pricing)
    assert payments.capital_pmt.sum() == pytest.approx(params.balance_amt.iloc[0])
    assert np.isfinite(payments[['outstanding_bal','capital_pmt','int_pmt']]).all().all()
    if kind == 1:
        assert len(payments) == 1
        assert payments.int_pmt.sum() == 0
    else:
        assert payments.capital_pmt.iloc[0] == pytest.approx(params.balance_amt.iloc[0]*0.05)
        assert payments.int_pmt.sum() > 0
    payments['cpr_rate'] = 0.0
    behavioural = cf.compute_adjusted_schedule(payments)
    assert np.allclose(behavioural.outstanding_adj, payments.outstanding_bal)
    assert np.allclose(behavioural.capital_adj, payments.capital_pmt)
    nii = load_module(ROOT/'irrbb_calc/python_code/nii_calc_objects.py', monkeypatch, cfg)
    monkeypatch.setitem(sys.modules, 'nii_calc_objects', nii)
    eve = load_module(ROOT/'irrbb_calc/python_code/eve_calc_objects.py', monkeypatch, cfg)
    beh = payments.copy()
    beh['currency'] = bank.market.ccy
    beh['product_code'] = str(row.product_code)
    beh['rate_type'] = 'V'
    beh['source'] = 'loans'
    beh['product_type'] = 'L'
    beh['prepayment_pmt'] = 0.0
    beh['eff_rate'] = beh['fwd_rt']
    limits = dict(caps_map={str(row.product_code): rm.client_cap},
                  floors_map={str(row.product_code): 0.0},
                  coeff_a_map={str(row.product_code): rm.beta},
                  coeff_b_map={str(row.product_code): rm.margin_pct/100})
    _, ns = nii.compute_nii_shocked(beh, disc, {bank.market.ccy:bank.market.disc_curve},
                                   cfg.report_date, 'base', **limits)
    _, es = eve.compute_eve_shocked(beh, disc, {bank.market.ccy:bank.market.disc_curve},
                                   'base', **limits)
    assert np.isfinite(ns.nii_total).all()
    assert np.isfinite(es.pv_total).all()
    shocked = disc * np.exp(-0.02*np.arange(len(dates))/365)
    _, nu = nii.compute_nii_shocked(beh, shocked, {bank.market.ccy:bank.market.disc_curve},
                                   cfg.report_date, 'par_up', **limits)
    if kind == 1:
        assert ns.nii_total.sum() == 0
        assert nu.nii_total.sum() == 0
    else:
        assert nu.nii_total.sum() > ns.nii_total.sum()
    # These schedule-level functions are the production workflow entry points.
    nsched = nii.compute_nii_shocked_schedule(beh, shocked,
                 {bank.market.ccy:bank.market.disc_curve}, cfg.report_date, 'par_up', **limits)
    esched = eve.compute_eve_shocked_schedule(beh, shocked,
                 {bank.market.ccy:bank.market.disc_curve}, 'par_up', **limits)
    assert np.isfinite(nsched.nii_total).all()
    assert np.isfinite(esched.pv_total).all()
    if kind == 1:
        assert nsched.nii_total.sum() == 0


def test_limit_risk_and_liquidity(monkeypatch):
    liq = load_module(ROOT/'liq_calc/python_code/lcr_nsfr_objects.py', monkeypatch, SimpleNamespace())
    card = example_rows()['bs_structure'][0]
    terms = CardTerms.from_row(card)
    assert terms.undrawn_per_drawn == 1.5
    assert terms.ead_multiplier == 1.6
    factors = effective_risk_factors(card)
    assert factors['rwa'] == pytest.approx(1.2)
    assert factors['el'] == pytest.approx(0.04*0.65*1.6)
    drawn = 100_000
    row = dict(card, bs_percentage=50.0)
    df = pd.DataFrame([row])
    # card limits are a facility line of the commitments register (labbank_common.commitments)
    from labbank_common import commitments as cmt
    line = pd.DataFrame([dict(commitment_code='9200', own_name='card_limits', drawn_product_code='2200',
                              currency='EUR', client_type_id=4, facility_type='credit',
                              revocability='unconditionally_cancellable', revolving=1,
                              utilization=terms.utilization, limit_pct=np.nan, availability=None,
                              undrawn_lcr=terms.undrawn_lcr, undrawn_rsf=terms.undrawn_rsf,
                              undrawn_ccf=terms.undrawn_ccf)])
    reg = cmt.build_register(line, {'2200': drawn}, 1e9, {'EUR': 1.0}, '2026-09-30')
    assert reg.limit_amt.iloc[0] == pytest.approx(reg.drawn_amt.iloc[0] + reg.undrawn_amt.iloc[0])
    out = {'EUR': float(reg.available_amt_30d.iloc[0] * reg.undrawn_lcr.iloc[0])}
    rsf = {'EUR': float(reg.undrawn_amt.iloc[0] * reg.undrawn_rsf.iloc[0])}
    assert liq.compute_lcr(df, drawn, commitment_outflows_by_ccy=out).outflows_30d.iloc[0] == pytest.approx(drawn*factors['lcr'])
    assert liq.compute_nsfr(df, drawn, rsf).rsf.iloc[0] == pytest.approx(drawn*factors['rsf'])
    sys.path.insert(0, str(ROOT/'optimize_prep/python_code'))
    from lcr_fast import compute_lcr_components_fast
    from nsfr_fast import compute_nsfr_components_fast
    from rwa_fast import compute_rwa_fast
    params = SimpleNamespace(currency=np.array(['EUR']), hqla_factor=np.array([0.]),
                             lcr_runoff=np.array([factors['lcr']]), inflow_30d_frac=np.array([0.]),
                             asf_factor=np.array([0.]), rsf_factor=np.array([factors['rsf']]),
                             rwa_factor=np.array([factors['rwa']]),
                             commit_lcr_outflow=0.0, commit_rsf=0.0, commit_rwa=0.0,
                             commit_scale=lambda amounts: 0.0)
    amounts = np.array([drawn])
    assert compute_lcr_components_fast(amounts, params, 'EUR')['outflows_30d'] == pytest.approx(drawn*factors['lcr'])
    assert compute_nsfr_components_fast(amounts, params, 'EUR')['rsf'] == pytest.approx(drawn*factors['rsf'])
    assert compute_rwa_fast(amounts, params) == pytest.approx(drawn*factors['rwa'])


@pytest.mark.parametrize('value', [0, -0.1, 1.1, float('nan')])
def test_invalid_utilization(value):
    with pytest.raises(ValueError):
        CardTerms(utilization=value)


def test_runoff_conservation_and_tail():
    opening, principal = runoff(1000, 120, .05)
    assert principal.sum() == pytest.approx(1000)
    assert np.allclose(opening[:-1]-principal[:-1], opening[1:])
    assert principal[-1] / 1000 < 0.003


def test_missing_or_interest_bearing_transactor_model():
    ex = example_rows()
    struct = pd.DataFrame(ex['bs_structure'])
    models = rate_models._normalise(pd.DataFrame(ex['interest_rt'])).set_index('product_code')
    with pytest.raises(ValueError, match='explicit'):
        validate_card_inputs(struct, models.iloc[:1])
    models.loc['2210', 'beta'] = 1.0
    with pytest.raises(ValueError, match='transactors'):
        validate_card_inputs(struct, models)
