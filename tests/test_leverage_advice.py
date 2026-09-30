"""The driver that turns a backtest into a leverage recommendation.

`common/margin_sizing.py` has its own tests for the arithmetic and for every failure mode it
names. What is tested here is the ASSEMBLY, which is where this repository's own choices live:
which broker assumptions stand in for a broker nobody has chosen, where the trial count comes
from, what happens to an entry whose KPIs are incomplete, and whether the column a user reads
still distinguishes "the broker would have closed you out" from "borrowed money would not have
helped". Those two print the same 1.00x.
"""

import re
import os
import unittest

import numpy as np
import pandas as pd

from common import leverage_advice as la
from common import margin_sizing as ms
from common.letf_mapper import LETFMapper
from common.selection import selection_trials, trial_sharpe_spread


def _returns(n=240, mu=0.0065, sigma=0.020, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2005-01-31', periods=n, freq='ME')
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


def _entry(name='X', held=('SPY', 'IEF'), sharpe=0.9, seed=3, n=240, **over):
    r = _returns(n=n, seed=seed)
    d = {
        'name': name, 'is_active': True, 'role': 'strategy',
        'returns': r, 'max_dd': -0.18, 'max_dd_months': 14,
        'sharpe': sharpe, 'vol': 0.11, 'cagr': 0.085, 'rf_annual': 0.02,
        'offensive_weight_mean': 0.7, 'held_tickers': held,
        'holds_leveraged_product': any(t in la.PRODUCT_MULTIPLE for t in held),
        'daily_max_dd': -0.23, 'in_ranked_window': True,
    }
    d.update(over)
    return d


def _suite():
    """Enough entries for a trial spread, spanning an unlevered book and a 3x one."""
    return [_entry('A', sharpe=0.9, seed=1), _entry('B', sharpe=0.6, seed=2),
            _entry('C', sharpe=1.2, seed=3),
            _entry('LEV', held=('UPRO', 'IEF'), sharpe=0.8, seed=4)]


class TestTheMaintenanceTable(unittest.TestCase):
    """The one input with no defensible default, supplied from a stated convention."""

    def test_the_multiple_is_derived_from_the_two_admissible_maps(self):
        self.assertEqual(set(la.PRODUCT_MULTIPLE),
                         set(LETFMapper.MAP_2X.values()) | set(LETFMapper.MAP_3X.values()))
        for img in LETFMapper.MAP_2X.values():
            self.assertEqual(la.product_multiple(img), 2.0, img)
        for img in LETFMapper.MAP_3X.values():
            self.assertEqual(la.product_multiple(img), 3.0, img)

    def test_an_ordinary_fund_carries_the_base_rate(self):
        self.assertEqual(la.product_multiple('SPY'), 1.0)
        self.assertEqual(la.maintenance_map(('SPY', 'IEF'), 0.30), {'SPY': 0.30, 'IEF': 0.30})

    def test_a_leveraged_product_is_charged_its_own_multiple(self):
        m = la.maintenance_map(('UPRO', 'SSO', 'BIL'), 0.30)
        self.assertAlmostEqual(m['UPRO'], 0.90)
        self.assertAlmostEqual(m['SSO'], 0.60)
        self.assertAlmostEqual(m['BIL'], 0.30)

    def test_the_mapping_covers_every_held_ticker_so_cap_one_is_calculable(self):
        """`resolve_maintenance_margin` refuses a mapping with a hole. It must never get one."""
        held = ('UPRO', 'TMF', 'GLD', 'BIL')
        kpis = ms.ModelKPIs(
            name='k', monthly_returns=_returns(), max_dd=-0.4, max_dd_months=20, sharpe=0.8,
            vol=0.3, cagr=0.1, rf_annual=0.02, offensive_weight_mean=0.8,
            holds_leveraged_product=True, held_tickers=held, daily_max_dd=-0.5)
        policy = ms.MarginPolicy(maintenance_margin=la.maintenance_map(held), seed=1,
                                 borrow_rate_annual=0.06, risk_free_annual=0.02)
        # The MAXIMUM over the holdings binds, which for a 3x book is 0.90.
        self.assertAlmostEqual(ms.resolve_maintenance_margin(policy, kpis), 0.90)


class TestTheHaircutDescribesTheSearchNotTheFilter(unittest.TestCase):
    """AUD-06: the Max margin column must not depend on which boxes were ticked.

    The multiple-testing haircut exists to price the search that produced the pick, and that
    search was always the whole registry. Deriving it from the RUN made a three-entry run
    receive a milder haircut than a thirty-six-entry one -- measured 2026-08-01, `HAA_G12` at
    1.35x against 1.27x -- with the smaller, less justified population flattering the answer.

    The population now comes from `tests/fixtures/run_facts.json`, which is written by
    `tools/emit_facts.py` from a full-registry run.
    """

    def test_the_population_is_the_registrys_not_the_runs(self):
        registry_n, registry_sd = la.registry_trial_population()
        self.assertIsNotNone(registry_n, 'run_facts.json carries no selection block; '
                                         'regenerate it with tools/emit_facts.py')
        run_n, _run_sd = trial_sharpe_spread(_suite())
        self.assertNotEqual(run_n, registry_n,
                            'the fixture suite happens to match the registry size, so this '
                            'test cannot tell the two sources apart — change the fixture')
        policy = la.build_policy(_suite())
        self.assertEqual(policy.n_trials, registry_n)
        self.assertAlmostEqual(policy.trial_sharpe_sd, registry_sd)

    def test_a_subset_and_a_larger_run_get_the_same_haircut(self):
        """The property the defect broke, asserted directly."""
        big = _suite()
        small = big[:3]
        self.assertEqual(la.build_policy(small).n_trials, la.build_policy(big).n_trials)
        self.assertEqual(la.build_policy(small).trial_sharpe_sd,
                         la.build_policy(big).trial_sharpe_sd)

    def test_the_assumption_line_names_the_registry_as_the_source(self):
        advice = la.advise(_suite())
        text = '\n'.join(la.assumption_lines(advice))
        self.assertIn('derived from the REGISTRY', text)
        self.assertIn('AUD-06', text)

    def test_a_missing_facts_file_falls_back_LOUDLY(self):
        """A silent fallback would restore the defect and hide it, which is worse than the
        defect: the number would be flattering AND unremarked."""
        from unittest.mock import patch
        with patch.object(la, 'REGISTRY_FACTS', os.path.join(os.sep, 'nope', 'missing.json')):
            self.assertEqual(la.registry_trial_population(), (None, None))
            advice = la.advise(_suite())
            text = '\n'.join(la.assumption_lines(advice))
        self.assertIn('derived from THIS RUN', text)
        self.assertIn('flatters', text.lower())
        self.assertIn('tools/emit_facts.py', text)

    def test_a_malformed_facts_file_is_treated_as_absent(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            broken = os.path.join(tmp, 'run_facts.json')
            with open(broken, 'w', encoding='utf-8') as fh:
                fh.write('{ this is not json')
            with patch.object(la, 'REGISTRY_FACTS', broken):
                self.assertEqual(la.registry_trial_population(), (None, None))


class TestWhatItTakesFromTheRun(unittest.TestCase):

    def test_the_trial_count_excludes_controls_and_exploratory_entries(self):
        rows = [_entry('S1'), _entry('S2'), _entry('S3'),
                _entry('CTRL', role='control'), _entry('X3', role='exploratory'),
                _entry('BENCH', is_active=False, role='benchmark')]
        self.assertEqual([d['name'] for d in selection_trials(rows)], ['S1', 'S2', 'S3'])
        n, sd = trial_sharpe_spread(rows)
        self.assertEqual(n, 3)
        self.assertGreaterEqual(sd, 0.0)

    def test_a_non_finite_sharpe_leaves_both_the_count_and_the_spread(self):
        """AUD-04: the pair feeds one E[max of N] formula, so N must count exactly the
        Sharpes the spread was measured over — a NaN trial widened nothing."""
        rows = [_entry('S1', sharpe=0.9), _entry('S2', sharpe=0.6),
                _entry('S3', sharpe=1.2), _entry('NAN', sharpe=float('nan'))]
        n, sd = trial_sharpe_spread(rows)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(sd, float(np.std([0.9, 0.6, 1.2], ddof=1)))

    def test_participation_ratio_refuses_a_window_shorter_than_its_columns(self):
        """AUD-03: zero-filling the gaps would bias correlations toward zero and inflate
        the PR — a harsher penalty dressed as a measurement. Refusal is the honest branch."""
        from common.selection import participation_ratio
        idx = pd.date_range('2020-01-31', periods=3, freq='ME')
        short = {f'S{i}': pd.Series([0.01, 0.02, -0.01], index=idx) for i in range(5)}
        self.assertIsNone(participation_ratio(short))

    def test_too_few_trials_anywhere_means_no_haircut_and_the_assumptions_say_so(self):
        """With no registry artefact AND a single-entry run there is no population to
        derive a haircut from, so none is applied and the omission is stated.

        Before AUD-06 this was the behaviour for ANY small run, which was the defect: the
        registry's search does not stop counting because somebody ticked one box. A
        single-entry run now inherits the registry's twenty trials; it is only when the
        artefact is missing too that the haircut genuinely has no basis.
        """
        from unittest.mock import patch
        with patch.object(la, 'REGISTRY_FACTS', os.path.join(os.sep, 'nope', 'absent.json')):
            advice = la.advise([_entry('only')])
            lines = la.assumption_lines(advice)
        self.assertIsNone(advice.policy.n_trials)
        self.assertTrue(any('No multiple-testing haircut' in a for a in lines))

    def test_a_single_entry_run_still_inherits_the_registrys_search(self):
        """The AUD-06 property from the smallest possible run."""
        registry_n, _sd = la.registry_trial_population()
        advice = la.advise([_entry('only')])
        self.assertEqual(advice.policy.n_trials, registry_n)

    def test_the_spread_is_measured_not_chosen(self):
        """The haircut coefficient is derived from a cross-sectional spread, never picked.

        Since AUD-06 the spread `build_policy` uses is the REGISTRY's rather than this
        run's, so the property is asserted in two halves: `trial_sharpe_spread` still
        measures whatever population it is handed, and `build_policy` falls back to exactly
        that measurement when the registry artefact is unavailable. What must never happen
        is a constant appearing from nowhere.
        """
        from unittest.mock import patch
        rows = _suite()
        n, sd = trial_sharpe_spread(rows)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(sd, float(np.std([0.9, 0.6, 1.2, 0.8], ddof=1)))
        with patch.object(la, 'REGISTRY_FACTS', os.path.join(os.sep, 'nope', 'absent.json')):
            self.assertEqual(la.build_policy(rows).trial_sharpe_sd, sd)

    def test_the_risk_free_rate_is_the_one_the_sharpes_were_netted_against(self):
        rows = [_entry('late', in_ranked_window=False, rf_annual=0.09),
                _entry('ranked', in_ranked_window=True, rf_annual=0.02)]
        self.assertAlmostEqual(la.build_policy(rows).risk_free_annual, 0.02)


class TestNoBrokerIsRequired(unittest.TestCase):
    """The question this module was written to settle."""

    def test_an_unsupplied_credit_line_leaves_the_axis_unbounded_and_named(self):
        advice = la.advise(_suite(), capacity_leverage=None)
        rec = advice.by_name['A']
        self.assertIsNone(rec.cap_value('borrowing_capacity'))
        self.assertTrue(any('capacity' in a.lower() for a in rec.invalidating_assumptions))
        # And the answer is still a number: CAP 1 is what the whole result rests on.
        self.assertIsNotNone(rec.recommended_leverage)

    def test_a_supplied_credit_line_can_bind_and_is_named(self):
        advice = la.advise(_suite(), capacity_leverage=1.05)
        self.assertEqual(advice.by_name['A'].binding_constraint, 'borrowing_capacity')
        self.assertEqual(la.short_constraint(advice.by_name['A']), 'credit line')

    def test_the_safety_factor_is_the_only_preference_and_it_moves_the_answer(self):
        prudent = la.advise(_suite(), k=5.0).by_name['A'].recommended_leverage
        aggressive = la.advise(_suite(), k=2.0).by_name['A'].recommended_leverage
        self.assertLess(prudent, aggressive)

    def test_a_higher_maintenance_requirement_lowers_the_answer(self):
        low = la.advise(_suite(), maintenance_base=0.25).by_name['A'].recommended_leverage
        high = la.advise(_suite(), maintenance_base=0.50).by_name['A'].recommended_leverage
        self.assertLess(high, low)


class TestRefusalsAreLoud(unittest.TestCase):

    def test_a_missing_kpi_names_the_field_rather_than_dropping_the_row(self):
        rows = _suite() + [_entry('BROKEN', offensive_weight_mean=None)]
        advice = la.advise(rows)
        self.assertNotIn('BROKEN', advice.by_name)
        self.assertEqual(len(advice.skipped), 1)
        name, why = advice.skipped[0]
        self.assertEqual(name, 'BROKEN')
        self.assertIn('offensive_weight_mean', why)

    def test_an_entry_with_no_holdings_cannot_be_sized(self):
        advice = la.advise(_suite() + [_entry('EMPTY', held=())])
        self.assertIn('EMPTY', dict(advice.skipped))

    def test_a_missing_daily_drawdown_is_a_non_calculable_ROW_not_a_missing_row(self):
        """The distinction matters: the entry still appears, saying what it could not answer."""
        advice = la.advise(_suite() + [_entry('NODAILY', daily_max_dd=None)])
        rec = advice.by_name['NODAILY']
        self.assertIsNone(rec.recommended_leverage)
        self.assertIn('CAP 1', rec.binding_constraint)
        self.assertEqual(la.cell(rec)[0], 'n/c')

    def test_a_short_history_is_not_extrapolated(self):
        advice = la.advise(_suite() + [_entry('SHORT', n=60)])
        self.assertIsNone(advice.by_name['SHORT'].recommended_leverage)


class TestTheColumnKeepsTheDistinction(unittest.TestCase):

    def test_the_kelly_gate_and_a_margin_call_both_print_one_but_read_differently(self):
        gated = la.advise([_entry(f'G{i}', sharpe=0.02, seed=i, cagr=0.02, vol=0.30)
                           for i in range(4)])
        rec = gated.by_name['G0']
        self.assertEqual(rec.recommended_leverage, 1.0)
        self.assertEqual(la.cell(rec), ('1.00x', 'no leverage'))

        called = la.advise(_suite(), k=40.0)
        rec = called.by_name['A']
        self.assertEqual(rec.recommended_leverage, 1.0)
        self.assertEqual(la.cell(rec), ('1.00x', 'margin call'))

    def test_the_headline_states_the_finding_when_nothing_is_sizeable(self):
        advice = la.advise(_suite(), k=40.0)
        self.assertIn('No entry', la.headline(advice))

    def test_a_mixed_registry_still_aligns_in_one_table(self):
        """A 3x book faces a different maintenance rate than an unlevered one, and that is a
        property of the HOLDINGS. `compare_table` allows exactly that field to differ."""
        text = la.advise(_suite()).table()
        for name in ('A', 'B', 'C', 'LEV'):
            self.assertIn(name, text)

    def test_a_levered_run_says_the_column_is_margin_on_top(self):
        advice = la.advise(_suite(), run_leverage=1.3)
        self.assertTrue(any('ON TOP' in a for a in la.assumption_lines(advice)))


class TestTheEngineSuppliesWhatTheDriverNeeds(unittest.TestCase):
    """`run_backtest` is the only producer of these KPIs, and the two files must not drift.

    Read off the source rather than by running a backtest: the run costs a minute and needs the
    price cache, and what is being asserted is a wiring fact, not a numerical one.
    """

    def test_run_backtest_attaches_every_required_kpi(self):
        import main
        with open(main.__file__, encoding='utf-8') as fh:
            src = fh.read()
        body = src[src.index('def run_backtest('):src.index('def _fmt_window(')]
        for key in la.REQUIRED_KPIS:
            self.assertRegex(body, rf"'{re.escape(key)}':",
                             f'run_backtest no longer attaches {key}, which leverage_advice '
                             f'requires')

    def test_the_report_calls_the_driver(self):
        import main
        self.assertTrue(hasattr(main, '_sustainable_leverage_section'))


class TestTheEvidenceBehindTheColumn(unittest.TestCase):
    """Every figure here was computed on every run and rendered nowhere until 2026-08-01.

    `report()` formats most of them and has exactly one caller: a test. `compare_table` is the
    only renderer with a consumer, and it prints what is needed to ACT while dropping what is
    needed to DOUBT. These assertions exist so a future tidy-up of the wide table cannot
    quietly return the numbers to invisibility.
    """

    def setUp(self):
        self.advice = la.advise(_suite())
        self.text = self.advice.evidence()

    def test_the_deflated_sharpe_probability_finally_reaches_a_screen(self):
        """The module's own answer to "is this Sharpe real given how many were searched" —
        the multiple-testing correction the whole selection apparatus exists to apply."""
        self.assertIn('P(DSR)', self.text)
        rec = next(iter(self.advice.by_name.values()))
        self.assertIn(f'{rec.deflated_sharpe_probability:.1%}', self.text)

    def test_the_block_sensitivity_is_visible_which_is_its_entire_purpose(self):
        """`margin_sizing` says, in its own words, that this exists "so the one genuinely
        arbitrary choice is visible instead" of hidden. It was hidden."""
        for months in (6, 12, 24):
            self.assertIn(f'DD@{months}mo', self.text)

    def test_the_drawdown_is_shown_as_the_two_factors_it_is_composed_of(self):
        """The adjusted drawdown is deliberately built as a product of two named factors
        rather than one opaque daily bootstrap. Printing only the product turns the
        decomposition back into the opaque number it was designed to replace."""
        self.assertIn('xsample', self.text)
        self.assertIn('xintra', self.text)
        self.assertIn('DD_boot', self.text)

    def test_all_four_sharpe_variants_appear_with_the_used_one_identified(self):
        self.assertIn('SR_obs', self.text)
        self.assertIn('SR_def', self.text)
        self.assertIn('SR_low', self.text)
        self.assertIn('SR_use', self.text)
        self.assertIn('whichever doubt bites harder', self.text)

    def test_selection_error_and_sampling_error_are_named_as_different_objections(self):
        """The deflation answers "N variants were searched"; the Mertens standard error
        answers "this sample is short". Collapsing them into one caption would lose the
        distinction that makes reporting both worthwhile."""
        self.assertIn('Bailey & Lopez de Prado', self.text)
        self.assertIn('Mertens', self.text)
        self.assertIn('SAMPLING error, a different', self.text)

    def test_every_sized_model_gets_a_row(self):
        for name in self.advice.by_name:
            self.assertIn(name, self.text)

    def test_an_empty_advice_says_so_rather_than_printing_a_header(self):
        empty = la.Advice(by_name={}, policy=None, skipped=(), run_leverage=1.0,
                          maintenance_base=la.MAINTENANCE_BASE)
        self.assertEqual(empty.evidence(), '(nothing to size)')

    def test_the_cli_section_prints_it(self):
        import inspect
        import main
        src = inspect.getsource(main._sustainable_leverage_section)
        self.assertIn('advice.evidence()', src)


if __name__ == '__main__':
    unittest.main()
