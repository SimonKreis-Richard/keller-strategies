"""The savings projection: every mechanism checked against arithmetic done in the test.

Offline. The strategy records are synthetic, the backtest is injected, and no expected number
is read off the module under test: annuity values, tax bills and the REER/CELI identity are
computed here, by hand, from their textbook definitions.
"""
import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import margin_sizing as ms          # noqa: E402
from common import metrics as metrics_mod       # noqa: E402
from common import projection as proj           # noqa: E402
from tools import projection as tool            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAX = proj.TaxAssumptions(marginal_rate_now=0.40, marginal_rate_retirement=0.40,
                          capital_gains_inclusion=0.5)
FX = SimpleNamespace(usdcad=1.40, source='test rate', as_of='2026-09-23')


def _flat(r, months):
    return np.full((1, months), float(r))


def _account(kind='CELI', start=0.0, monthly=0.0, **kw):
    return proj.AccountPlan(name=kind.lower(), kind=kind, start_balance=start,
                            monthly_contribution=monthly, **kw)


def _run(account, paths, tax=TAX, **kw):
    sched = proj.schedule(account, tax, np.atleast_2d(paths).shape[1])
    return proj.simulate(paths, account, sched, tax, **kw)


def _entry(name='SYNTH', role='strategy', seed=3, n=240, mu=0.008, sd=0.03):
    idx = pd.date_range('2000-01-31', periods=n, freq='ME')
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(mu, sd, n), index=idx)
    rf = pd.Series(0.0015, index=idx)
    m = metrics_mod.calculate_metrics(returns, rf=rf)
    return {'name': name, 'role': role, 'returns': returns, 'rf_series': rf,
            'sharpe': m['sharpe'], 'vol': m['vol'], 'rf_annual': m['rf_annual']}


class TestTheMechanicsMatchTheTextbook(unittest.TestCase):

    def test_a_constant_return_with_contributions_is_the_annuity_formula(self):
        p, c, r, n = 10_000.0, 250.0, 0.006, 180
        expected = p * (1 + r) ** n + c * ((1 + r) ** n - 1) / r   # ordinary annuity
        v = _run(_account('CELI', p, c), _flat(r, n))
        self.assertAlmostEqual(float(v[0, -1]), expected, places=6)
        self.assertAlmostEqual(float(v[0, 0]), p)

    def test_reer_equals_celi_when_the_rate_is_the_same_in_and_out(self):
        rng = np.random.default_rng(7)
        paths = rng.normal(0.006, 0.04, (50, 120))
        celi = _run(_account('CELI', 0.0, 300.0), paths)
        reer = _run(_account('REER', 0.0, 300.0, reinvest_refund=True), paths)
        np.testing.assert_allclose(reer, celi, rtol=1e-12)

    def test_reer_beats_celi_only_when_the_rate_falls_at_withdrawal(self):
        paths = _flat(0.005, 120)
        low = proj.TaxAssumptions(0.40, 0.25)
        celi = _run(_account('CELI', 0.0, 300.0), paths, low)[0, -1]
        reer = _run(_account('REER', 0.0, 300.0), paths, low)[0, -1]
        # Deposits are 300/0.6 = 500; the balance keeps 1 - 0.25 of it.
        self.assertAlmostEqual(reer / celi, (1 - 0.25) / (1 - 0.40), places=9)

    def test_a_reer_start_balance_is_worth_its_after_tax_value(self):
        v = _run(_account('REER', 10_000.0), _flat(0.0, 12))
        self.assertAlmostEqual(float(v[0, 0]), 6_000.0)

    def test_celiapp_is_deducted_in_and_untaxed_out(self):
        paths = _flat(0.0, 12)
        v = _run(_account('CELIAPP', 0.0, 600.0), paths)[0, -1]
        self.assertAlmostEqual(float(v), 12 * 600.0 / 0.6)


class TestTheTaxableAccount(unittest.TestCase):

    def test_a_loss_is_carried_into_the_next_years_gain(self):
        path = np.zeros((1, 24))
        path[0, 3] = -0.10      # year 1: 1000 -> 900, loss 100 carried
        path[0, 15] = 0.20      # year 2: 900 -> 1080, gain 180, 80 taxable
        v = _run(_account('NON_ENREGISTRE', 1000.0), path)
        self.assertAlmostEqual(float(v[0, 12]), 900.0)
        self.assertAlmostEqual(float(v[0, -1]), 1080.0 - 80.0 * 0.40 * 0.5)

    def test_contributions_are_not_taxed_as_gains(self):
        v = _run(_account('NON_ENREGISTRE', 0.0, 100.0), _flat(0.0, 24))
        self.assertAlmostEqual(float(v[0, -1]), 2400.0)

    def test_the_gain_of_a_year_is_taxed_once_at_the_inclusion_rate(self):
        r = 0.01
        v = _run(_account('NON_ENREGISTRE', 1000.0), _flat(r, 12))
        gain = 1000.0 * ((1 + r) ** 12 - 1)
        self.assertAlmostEqual(float(v[0, -1]), 1000.0 + gain * (1 - 0.40 * 0.5), places=9)

    def test_a_partial_last_year_is_taxed_at_the_horizon(self):
        v = _run(_account('NON_ENREGISTRE', 1000.0), _flat(0.01, 18))
        after_one = 1000.0 + 1000.0 * (1.01 ** 12 - 1) * 0.8
        expected = after_one + after_one * (1.01 ** 6 - 1) * 0.8
        self.assertAlmostEqual(float(v[0, -1]), expected, places=9)

    def test_interest_can_be_taxed_in_full(self):
        v = _run(_account('NON_ENREGISTRE', 1000.0), _flat(0.01, 12), income_inclusion=1.0)
        self.assertAlmostEqual(float(v[0, -1]), 1000.0 + 1000.0 * (1.01 ** 12 - 1) * 0.6,
                               places=9)

    def test_the_celi_is_never_taxed(self):
        v = _run(_account('CELI', 1000.0), _flat(0.01, 12))
        self.assertAlmostEqual(float(v[0, -1]), 1000.0 * 1.01 ** 12, places=9)


class TestContributionsAreScheduledAsWritten(unittest.TestCase):

    def test_the_cap_binds_on_deposits_and_the_excess_is_reported(self):
        a = _account('CELI', monthly=1000.0, annual_cap=7000.0)
        s = proj.schedule(a, TAX, 24)
        self.assertAlmostEqual(s.deposits[:12].sum(), 7000.0)
        self.assertAlmostEqual(s.deposits[12:].sum(), 7000.0)
        self.assertEqual(s.capped, {1: 5000.0, 2: 5000.0})

    def test_a_grossed_up_deposit_is_capped_and_its_effort_follows(self):
        a = _account('REER', monthly=600.0, annual_cap=6000.0)   # wants 1000 a month
        s = proj.schedule(a, TAX, 12)
        self.assertAlmostEqual(s.deposits.sum(), 6000.0)
        self.assertAlmostEqual(s.effort.sum(), 6000.0 * 0.6)
        self.assertAlmostEqual(s.capped[1], 12 * 600.0 - 3600.0)

    def test_a_kept_refund_is_counted_and_not_invested(self):
        a = _account('REER', monthly=600.0, reinvest_refund=False)
        s = proj.schedule(a, TAX, 12)
        self.assertAlmostEqual(s.deposits.sum(), 7200.0)
        self.assertAlmostEqual(s.refunds_kept.sum(), 7200.0 * 0.40)

    def test_contributions_grow_once_a_year(self):
        a = _account('CELI', monthly=100.0, contribution_growth=0.10)
        s = proj.schedule(a, TAX, 36)
        self.assertAlmostEqual(s.deposits[11], 100.0)
        self.assertAlmostEqual(s.deposits[12], 110.0)
        self.assertAlmostEqual(s.deposits[35], 121.0)


class TestTheReturnIsTheHaircutOne(unittest.TestCase):

    def test_haircut_sharpe_is_the_one_the_kelly_gate_uses(self):
        e = _entry()
        m = metrics_mod.calculate_metrics(e['returns'], rf=0.02)
        kpis = ms.ModelKPIs(name='SYNTH', monthly_returns=e['returns'], max_dd=m['max_dd'],
                            max_dd_months=m['max_dd_months'], sharpe=m['sharpe'], vol=m['vol'],
                            cagr=m['cagr'], rf_annual=m['rf_annual'], offensive_weight_mean=0.75,
                            holds_leveraged_product=False, held_tickers=('SPY',),
                            daily_max_dd=m['max_dd'] * 1.1)
        policy = ms.MarginPolicy(safety_factor_k=3.0, maintenance_margin=0.30,
                                 borrow_rate_annual=0.05, risk_free_annual=0.035,
                                 borrowing_capacity_leverage=4.0, seed=11, n_bootstrap=200,
                                 n_trials=19, trial_sharpe_sd=0.15, min_history_months=120)
        rec = ms.recommend_leverage(kpis, policy)
        cut = ms.haircut_sharpe(m['sharpe'], e['returns'], 19, 0.15, policy.sharpe_z)
        self.assertEqual(cut.used, rec.sharpe_used)
        self.assertEqual(cut.deflated, rec.sharpe_deflated)

    def test_the_selection_haircut_is_computed_from_its_definition(self):
        e = _entry()
        a = proj.return_assumption(e, 19, 0.112)
        self.assertAlmostEqual(a.haircut.deflated,
                               e['sharpe'] - ms.expected_max_sharpe(19, 0.112))
        se = ms.sharpe_standard_error(e['sharpe'], e['returns'])
        self.assertAlmostEqual(a.haircut.lower_bound, e['sharpe'] - se)
        self.assertEqual(a.haircut.used, min(a.haircut.deflated, a.haircut.lower_bound))
        self.assertAlmostEqual(a.mean_excess_used, a.haircut.used * e['vol'])
        self.assertLess(a.mean_excess_used, a.mean_excess_observed)

    def test_a_benchmark_takes_no_selection_haircut(self):
        a = proj.return_assumption(_entry(role='benchmark'), 19, 0.112)
        self.assertFalse(a.selected)
        self.assertTrue(math.isnan(a.haircut.deflated))
        self.assertEqual(a.haircut.used, a.haircut.lower_bound)

    def test_a_negative_haircut_sharpe_is_projected_not_floored(self):
        a = proj.return_assumption(_entry(mu=0.0005, sd=0.04), 19, 0.112)
        self.assertLess(a.haircut.used, 0)
        self.assertLess(a.mean_excess_used, 0)

    def test_a_selected_entry_without_its_population_is_refused(self):
        with self.assertRaises(proj.ProjectionError):
            proj.return_assumption(_entry(), None, None)

    def test_recentring_moves_the_level_and_keeps_the_shape(self):
        e = _entry()
        out = proj.recentre(e['returns'], e['rf_series'], 0.03, 0.02)
        rf_m = 1.02 ** (1 / 12) - 1
        self.assertAlmostEqual(float((out - rf_m).mean()), 0.03 / 12, places=12)
        excess = e['returns'] - e['rf_series']
        self.assertAlmostEqual(float(out.std()), float(excess.std()), places=12)
        self.assertAlmostEqual(float(out.corr(e['returns'])), 1.0, places=12)

    def test_recentring_at_the_observed_sharpe_changes_nothing_but_the_cash_rate(self):
        e = _entry()
        observed = e['sharpe'] * e['vol']
        out = proj.recentre(e['returns'], e['rf_series'], observed, 0.0)
        excess = e['returns'] - e['rf_series']
        np.testing.assert_allclose(out.to_numpy(), excess.to_numpy(), atol=1e-12)


class TestTheProjection(unittest.TestCase):

    def _project(self, seed=5, **kw):
        accounts = [_account('CELI', 10_000.0, 500.0), _account('REER', 5_000.0, 200.0)]
        return proj.project([_entry(), _entry('BENCH', role='benchmark', seed=9)], accounts,
                            TAX, kw.get('years', 10), kw.get('inflation', 0.02), 19, 0.112,
                            'test population', n_paths=300, seed=seed)

    def test_an_unseeded_projection_is_refused(self):
        with self.assertRaises(proj.ProjectionError):
            self._project(seed=None)

    def test_the_same_seed_gives_the_same_projection(self):
        a, b = self._project(), self._project()
        for name in a.values:
            np.testing.assert_array_equal(a.values[name], b.values[name])

    def test_real_values_are_nominal_over_the_inflation_index(self):
        p = self._project(inflation=0.03)
        s = proj.summary(p, 'SYNTH', 120)
        for nominal, real in zip(s['nominal'], s['real']):
            self.assertAlmostEqual(real, nominal / 1.03 ** 10, places=6)

    def test_invested_is_the_after_tax_start_plus_the_effort(self):
        p = self._project()
        self.assertAlmostEqual(float(p.invested[0]), 10_000.0 + 5_000.0 * 0.6)
        self.assertAlmostEqual(float(p.invested[-1]), 13_000.0 + 120 * 700.0)

    def test_the_savings_account_is_the_cash_rate_with_no_spread(self):
        p = self._project()
        self.assertEqual(p.values['savings account'].shape[0], 1)
        self.assertIn('savings account', p.values)

    def test_the_strategy_and_its_benchmark_share_one_history(self):
        # b is a month-by-month function of a, so it survives resampling only if every path
        # draws the same months for both.
        a = _entry()['returns']
        s = proj.bootstrap_paths({'a': a, 'b': 2.0 * a + 0.001}, 60, 50, seed=4)
        self.assertEqual(s['a'].shape, (50, 60))
        np.testing.assert_allclose(s['b'], 2.0 * s['a'] + 0.001, rtol=0, atol=1e-15)

    def test_the_other_currency_is_a_conversion_at_todays_rate(self):
        p = self._project()
        p.fx = FX
        s = proj.summary(p, 'SYNTH', 120)
        for amount, other in zip(s['nominal'], s['other']):
            self.assertAlmostEqual(other, amount / 1.40)
        p.currency = 'USD'
        self.assertAlmostEqual(p.to_other(1000.0), 1400.0)
        self.assertEqual(p.other_currency, 'CAD')

    def test_the_report_shows_both_currencies_or_says_why_not(self):
        p = self._project()
        p.fx = FX
        text = '\n'.join(proj.report_lines(p))
        self.assertIn('1 USD = 1.4000 CAD (test rate, 2026-09-23)', text)
        self.assertIn('median, USD', text)
        p.fx, p.fx_note = None, 'every source failed'
        text = '\n'.join(proj.report_lines(p))
        self.assertIn('USD not shown: every source failed', text)
        self.assertNotIn('median, USD', text)

    def test_the_report_states_its_assumptions_before_any_number(self):
        text = '\n'.join(proj.report_lines(self._project()))
        self.assertIn('not a forecast', text)
        self.assertLess(text.index('not a forecast'), text.index('after 5 years'))
        self.assertLess(text.index('used'), text.index('after 5 years'))
        self.assertIn('NOT modelled', text)


class TestTheConfigIsStrict(unittest.TestCase):

    def _cfg(self, **over):
        block = {'strategy': 'HAA_G12', 'comparator': 'SPY_Benchmark', 'horizon_years': 20,
                 'inflation': 0.02, 'currency': 'CAD', 'usdcad': None,
                 'tax': {'marginal_rate_now': 0.37, 'marginal_rate_retirement': 0.3},
                 'accounts': {'A': {'kind': 'TFSA', 'monthly_contribution': 100}}}
        block.update(over)
        return {'BROKER_ACCOUNTS': [{'account_name': 'A', 'account_balance': 1234.0,
                                     'account_priority': 1}],
                'PROJECTION': block}

    def test_a_good_block_parses_and_takes_the_broker_balance(self):
        settings, accounts, tax = proj.parse_config(self._cfg())
        self.assertEqual(accounts[0].kind, 'CELI')
        self.assertEqual(accounts[0].start_balance, 1234.0)
        self.assertEqual(tax.capital_gains_inclusion, 0.5)
        self.assertEqual(settings['strategy'], 'HAA_G12')

    def test_the_example_config_parses(self):
        with open(os.path.join(ROOT, 'user_config.example.json'), encoding='utf-8') as fh:
            settings, accounts, _tax = proj.parse_config(json.load(fh))
        self.assertEqual({a.kind for a in accounts}, {'CELI', 'REER', 'NON_ENREGISTRE'})

    def test_refusals(self):
        bad = [
            {'PROJECTION': None},
            self._cfg(accounts={'A': {'kind': 'RRIF'}}),
            self._cfg(tax={'marginal_rate_now': 1.0, 'marginal_rate_retirement': 0.3}),
            self._cfg(tax={'marginal_rate_now': 0.3}),
            self._cfg(inflation=True),
            self._cfg(horizon_years=0),
            self._cfg(colour='blue'),
            self._cfg(accounts={'A': {'kind': 'CELI', 'start_balance': 5.0}}),
            self._cfg(accounts={'A': {'kind': 'CELI'}, 'B': {'kind': 'CELI'}}),
            self._cfg(accounts={'A': {'kind': 'CELI', 'reinvest_refund': True}}),
            self._cfg(accounts={'A': {'kind': 'REER', 'reinvest_refund': 'yes'}}),
            self._cfg(accounts={'A': {'kind': 'CELI', 'monthly_contribution': -5}}),
            self._cfg(accounts={}),
            self._cfg(comparator='HAA_G12'),
            self._cfg(currency=None),
            self._cfg(currency='EUR'),
            self._cfg(usdcad=0),
            self._cfg(usdcad='1.38'),
        ]
        for cfg in bad:
            with self.subTest(cfg=cfg.get('PROJECTION')):
                with self.assertRaises(proj.ConfigError):
                    proj.parse_config(cfg)

    def test_a_broker_account_without_a_kind_is_refused(self):
        cfg = self._cfg()
        cfg['BROKER_ACCOUNTS'].append({'account_name': 'B', 'account_balance': 1.0,
                                       'account_priority': 2})
        with self.assertRaisesRegex(proj.ConfigError, 'B'):
            proj.parse_config(cfg)


def _boom(*_a, **_k):
    raise AssertionError('must not be called')


class TestTheTool(unittest.TestCase):

    def _run_fn(self, calls):
        def run(names, **over):
            calls.append((tuple(names), over))
            rows = {'HAA_G12': _entry('HAA_G12'),
                    'SPY_Benchmark': _entry('SPY_Benchmark', role='benchmark', seed=9)}
            return ([rows[n] for n in names],)
        return run

    def _main(self, argv, cfg, calls, fx_resolver=None):
        out = io.StringIO()
        with redirect_stdout(out):
            code = tool.main(argv, run_fn=self._run_fn(calls), config_loader=lambda: cfg,
                             population_fn=lambda: (19, 0.112, 'test population'),
                             fx_resolver=fx_resolver or (lambda manual: FX))
        return code, out.getvalue()

    def test_the_rate_typed_on_the_command_line_reaches_the_resolver(self):
        seen = []
        code, text = self._main(['--no-save', '--years', '2', '--rate', '1.3850'],
                                TestTheConfigIsStrict()._cfg(), [],
                                fx_resolver=lambda manual: seen.append(manual) or FX)
        self.assertEqual(code, 0, text)
        self.assertEqual(seen, [1.3850])
        self.assertIn('median, USD', text)

    def test_no_rate_hides_the_second_currency_and_says_why(self):
        def fail(manual):
            raise ValueError('no USD/CAD rate could be obtained\n  Bank of Canada: timeout')
        code, text = self._main(['--no-save', '--years', '2'], TestTheConfigIsStrict()._cfg(),
                                [], fx_resolver=fail)
        self.assertEqual(code, 0, text)
        self.assertIn('USD not shown: no USD/CAD rate could be obtained', text)
        self.assertNotIn('median, USD', text)

    def test_it_runs_offline_at_leverage_one_and_prints_assumptions_first(self):
        calls = []
        cfg = TestTheConfigIsStrict()._cfg()
        code, text = self._main(['--no-save', '--years', '6'], cfg, calls)
        self.assertEqual(code, 0, text)
        self.assertEqual(calls, [(('HAA_G12', 'SPY_Benchmark'), {'LEVERAGE_FACTOR': 1.0})])
        self.assertLess(text.index('not a forecast'), text.index('after 5 years'))
        self.assertIn('after 6 years', text)

    def test_a_missing_block_is_refused_before_any_backtest(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = tool.main(['--no-save'], run_fn=_boom, config_loader=lambda: {},
                             population_fn=_boom)
        self.assertEqual(code, 1)
        self.assertIn('PROJECTION', out.getvalue())

    def test_a_missing_population_is_a_refusal(self):
        def none():
            raise proj.ProjectionError('no selection population')
        out = io.StringIO()
        with redirect_stdout(out):
            code = tool.main(['--no-save'], run_fn=_boom,
                             config_loader=lambda: TestTheConfigIsStrict()._cfg(),
                             population_fn=none)
        self.assertEqual(code, 1)

    def test_the_record_is_saved_where_git_does_not_look(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = tool.main(['--years', '2'], run_fn=self._run_fn(calls),
                                 config_loader=lambda: TestTheConfigIsStrict()._cfg(),
                                 population_fn=lambda: (19, 0.112, 'p'), results_dir=tmp,
                                 fx_resolver=lambda manual: FX)
            self.assertEqual(code, 0)
            files = os.listdir(tmp)
            self.assertEqual(len(files), 1)
            with open(os.path.join(tmp, files[0]), encoding='utf-8') as fh:
                record = json.load(fh)
            self.assertIn('HAA_G12', record['returns'])
            self.assertEqual(record['fx']['usdcad'], 1.40)
        self.assertTrue(tool.RESULTS_DIR.endswith('backtest_results'))


if __name__ == '__main__':
    unittest.main()
