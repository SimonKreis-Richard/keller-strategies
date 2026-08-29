"""
The evaluation window is not a setting any more — these tests are what keeps it that way.

Deleting the START_DATE box only helps if nothing quietly grows back into its place. Three
distinct failures would do that, and each has a test here:

* a segmentation that does not actually tile the era (a gap silently drops months, an overlap
  silently double-counts them),
* a panel that reads the ranked window instead of the strategy's own history, which is the
  original bias wearing a regime panel as a disguise,
* a ranked table whose rows are measured over different spans and compared anyway.

One test is a genuine EXTERNAL anchor: the 2022 bear market's turning points are re-derived
from the frozen daily fixture with a mechanical rule and checked against the dates in
`common/eras.py`, which were written from the published record. If somebody edits those dates
to flatter a strategy, arithmetic on real prices contradicts them.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from common import eras
from common.data_engine import PriceStore
from tests.test_audit import make_daily_store

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
FROZEN_ADJ_CLOSE = os.path.join(FIXTURES, 'frozen_daily_adj_close.csv')
FROZEN_ADJ_OPEN = os.path.join(FIXTURES, 'frozen_daily_adj_open.csv')

ERA_END = pd.Timestamp('2026-06-30')


def month_ends(start, end):
    return pd.date_range(start, end, freq='ME')


class TestPartitionsAreRealPartitions(unittest.TestCase):
    """Exhaustive and disjoint, or the panel is arithmetic with holes in it."""

    def test_validate_passes_for_every_shipped_segmentation(self):
        for seg in eras.SEGMENTATIONS:
            with self.subTest(seg.key):
                self.assertTrue(eras.validate_partition(seg))

    def test_every_month_of_the_era_lands_in_exactly_one_segment(self):
        months = month_ends(eras.COMMON_ERA_START, ERA_END)
        for seg in eras.SEGMENTATIONS:
            segments = eras.resolved_segments(seg, ERA_END)
            for m in months:
                hits = [s.key for s in segments if s.start <= m <= s.end]
                with self.subTest(seg=seg.key, month=str(m.date())):
                    self.assertEqual(len(hits), 1,
                                     f'{m.date()} belongs to {hits} in {seg.key} — a '
                                     f'partition must place every month exactly once')

    def test_no_segment_starts_before_the_era(self):
        floor = pd.Timestamp(eras.COMMON_ERA_START)
        for seg in eras.SEGMENTATIONS:
            for s in seg.segments:
                with self.subTest(seg=seg.key, segment=s.key):
                    self.assertGreaterEqual(pd.Timestamp(s.start), floor)

    def test_a_gap_is_rejected(self):
        broken = eras.BUSINESS_CYCLE._replace(segments=(
            eras.Segment('a', eras.COMMON_ERA_START, '2010-12-31', 'a', False),
            eras.Segment('b', '2011-02-01', None, 'b', False),   # one month missing
        ))
        with self.assertRaises(ValueError):
            eras.validate_partition(broken)

    def test_an_overlap_is_rejected(self):
        broken = eras.BUSINESS_CYCLE._replace(segments=(
            eras.Segment('a', eras.COMMON_ERA_START, '2010-12-31', 'a', False),
            eras.Segment('b', '2010-06-01', None, 'b', False),
        ))
        with self.assertRaises(ValueError):
            eras.validate_partition(broken)


class TestSegmentsCompoundBackToTheEra(unittest.TestCase):
    """The property that makes a partition an honest decomposition rather than a selection.

    If the segments of a partition multiply back to the era's own total return, then no
    month was dropped and none was counted twice. A cherry-picked set of windows cannot
    satisfy this, which is precisely why it is worth asserting.
    """

    def setUp(self):
        idx = month_ends(eras.COMMON_ERA_START, ERA_END)
        rng = np.random.default_rng(20260728)
        self.returns = pd.Series(rng.normal(0.006, 0.035, len(idx)), index=idx)

    def test_product_over_segments_equals_product_over_the_era(self):
        whole = float((1.0 + self.returns).prod())
        for seg in eras.SEGMENTATIONS:
            pieces = 1.0
            for s in eras.resolved_segments(seg, ERA_END):
                window = self.returns.loc[(self.returns.index >= s.start)
                                          & (self.returns.index <= s.end)]
                pieces *= float((1.0 + window).prod())
            with self.subTest(seg.key):
                self.assertAlmostEqual(pieces, whole, places=10)

    def test_month_counts_add_up(self):
        for seg in eras.SEGMENTATIONS:
            total = sum(len(self.returns.loc[(self.returns.index >= s.start)
                                             & (self.returns.index <= s.end)])
                        for s in eras.resolved_segments(seg, ERA_END))
            with self.subTest(seg.key):
                self.assertEqual(total, len(self.returns))


class TestCellCoverage(unittest.TestCase):
    """`n/a`, `~partial` and a plain number must mean three different things."""

    def setUp(self):
        self.idx = month_ends('2012-01-31', '2024-12-31')
        self.returns = pd.Series(0.01, index=self.idx)

    def test_a_month_end_return_covers_a_segment_that_opens_that_month(self):
        """A return dated 2012-01-31 covers a segment starting 2012-01-01, in full.

        Monthly returns are stamped at month END. Comparing those timestamps to a segment's
        month START would mark a fully-covered segment `partial`, or drop it as `n/a` — which
        is exactly what the first version of this code did to HAA through the 2008 recession.
        """
        cell = eras.segment_cell(self.returns, self.idx[0],
                                 pd.Timestamp('2012-01-01'), pd.Timestamp('2012-12-31'))
        self.assertNotIn('na', cell)
        self.assertFalse(cell['partial'])
        self.assertEqual(cell['n_months'], 12)

    def test_starting_inside_a_segment_is_measured_and_flagged(self):
        cell = eras.segment_cell(self.returns, pd.Timestamp('2012-07-31'),
                                 pd.Timestamp('2012-01-01'), pd.Timestamp('2012-12-31'))
        self.assertTrue(cell['partial'], 'a partial window must never be presented as full')
        self.assertEqual(cell['n_months'], 6)

    def test_inception_after_the_segment_is_na_never_zero(self):
        cell = eras.segment_cell(self.returns, self.idx[0],
                                 pd.Timestamp('2008-07-01'), pd.Timestamp('2009-06-30'))
        self.assertIn('na', cell)
        self.assertIn('2012-01', cell['na'])
        self.assertNotIn('return', cell)


class TestCommonWindow(unittest.TestCase):
    def test_the_latest_inception_binds_and_is_named(self):
        start, binding = eras.common_window(
            {'early': pd.Timestamp('2008-06-30'), 'late': pd.Timestamp('2011-03-31')},
            floor=pd.Timestamp('2008-06-30'))
        self.assertEqual(str(start.date()), '2011-03-31')
        self.assertEqual(binding, ('late',))

    def test_the_floor_binds_when_everything_predates_it(self):
        start, binding = eras.common_window(
            {'a': pd.Timestamp('2005-01-31'), 'b': pd.Timestamp('2006-01-31')},
            floor=pd.Timestamp('2008-06-30'))
        self.assertEqual(str(start.date()), '2008-06-30')
        self.assertEqual(binding, (), 'nothing pushed the window past the floor, so nothing '
                                      'should be blamed for it')

    def test_every_tied_strategy_is_named(self):
        """Ties are the normal case here, not the exotic one: four leveraged G4 variants
        share UGL's 2008-12 inception. Naming one of them makes the report's own advice
        false — dropping it lengthens nothing, because the other three still bind."""
        start, binding = eras.common_window(
            {'early': pd.Timestamp('2008-06-30'),
             'late_a': pd.Timestamp('2011-03-31'),
             'late_b': pd.Timestamp('2011-03-31'),
             'late_c': pd.Timestamp('2011-03-31')},
            floor=pd.Timestamp('2008-06-30'))
        self.assertEqual(str(start.date()), '2011-03-31')
        self.assertEqual(binding, ('late_a', 'late_b', 'late_c'))

    def test_the_binding_set_does_not_depend_on_insertion_order(self):
        """`max()` returned whichever tied key it met first, so re-ordering the registry
        silently reassigned the blame. The answer is a property of the dates alone."""
        dates = {'z': pd.Timestamp('2011-03-31'), 'a': pd.Timestamp('2011-03-31'),
                 'm': pd.Timestamp('2009-01-31')}
        forward = eras.common_window(dates, floor=pd.Timestamp('2008-06-30'))
        reverse = eras.common_window(dict(reversed(list(dates.items()))),
                                     floor=pd.Timestamp('2008-06-30'))
        self.assertEqual(forward, reverse)
        self.assertEqual(forward[1], ('a', 'z'))


class TestEquityCycleAgainstRealPrices(unittest.TestCase):
    """External anchor: re-derive the 2022 turning points from the frozen fixture.

    `common/eras.py` states peak 2022-01-03 and trough 2022-10-12 from the published record.
    A mechanical drawdown rule on the fixture's real SPY closes has to agree, or one of the
    two is wrong. This is the one segmentation boundary this repository's own data can check.
    """

    def setUp(self):
        self.store = PriceStore.from_daily_fixture(FROZEN_ADJ_CLOSE, FROZEN_ADJ_OPEN)

    def test_the_2022_peak_and_trough_are_where_eras_says_they_are(self):
        spy = self.store.adj_close()['SPY'].dropna().loc['2021-06':'2023-06']
        drawdown = spy / spy.cummax() - 1.0
        trough = drawdown.idxmin()
        peak = spy.loc[:trough].idxmax()

        self.assertEqual(str(peak.date()), '2022-01-03')
        self.assertEqual(str(trough.date()), '2022-10-12')
        self.assertLess(drawdown.min(), -0.20, 'a bear market is a 20% fall by definition')

        bear = next(s for s in eras.EQUITY_CYCLE.segments if s.key == 'bear_2022')
        bull = next(s for s in eras.EQUITY_CYCLE.segments if s.key == 'bull_2022')
        # The peak month opens the bear; the trough month opens the recovery, because a
        # monthly series cannot split October 2022 in half.
        self.assertEqual(pd.Period(bear.start, 'M'), pd.Period(peak, 'M'))
        self.assertEqual(pd.Period(bull.start, 'M'), pd.Period(trough, 'M'))


class TestTheWindowIsNotASetting(unittest.TestCase):
    """main.py must not read a window from anywhere a person can type into."""

    def test_module_constants_are_the_frozen_era(self):
        import main
        self.assertEqual(main.START_DATE, eras.COMMON_ERA_START)
        self.assertIsNone(main.END_DATE,
                          'END_DATE must be None so the run ends at the last COMPLETE month '
                          'in the data, not on a date somebody picked')

    def test_the_retired_keys_are_announced_not_silently_ignored(self):
        from common import user_config
        self.assertIn('START_DATE', user_config._REMOVED)
        self.assertIn('END_DATE', user_config._REMOVED)


class TestRankedRowsShareOneWindow(unittest.TestCase):
    """End to end: two strategies with different inceptions, one comparable table."""

    @classmethod
    def setUpClass(cls):
        import main

        # A store where BIL — HAA's defensive asset — appears two years late. Every other
        # ticker runs from the start, so HAA_G12 is genuinely the binding strategy.
        base = make_daily_store(n_days=2600, start='2013-01-01')
        closes, opens = base.adj_close().copy(), base.adj_open().copy()
        late = closes.index < pd.Timestamp('2015-01-01')
        closes.loc[late, 'BIL'] = np.nan
        opens.loc[late, 'BIL'] = np.nan
        store = PriceStore.from_adjusted(closes, opens, source='synthetic')

        config = {
            'START_DATE': eras.COMMON_ERA_START, 'END_DATE': None,
            'DATA_START_DATE': '2013-01-01', 'EXECUTION_MODE': False,
            'CURRENT_EXECUTION_DATE': None,
            'LEVERAGE_FACTOR': 1.0, 'MARGIN_BORROW_RATE': 0.06,
            'MARGIN_FOLLOWS_SIGNAL': True, 'COST_PCT_PER_SIDE': 0.001,
            'LOOKBACK_MONTHS': 13, 'EXECUTION_CONVENTION': 'next_open',
            'CASH_TICKER': 'BIL', 'COVERAGE_POLICY': 'trim', 'RF_ANNUAL_FALLBACK': 0.03,
        }
        prices, s_w, s_u = main.build_signal_panel(store, config)
        strategies = [main.ALL_STRATEGIES[n]() for n in ('SPY_Benchmark', 'HAA_G12')]
        cls.metrics, _ = main.run_backtest(prices, s_w, s_u, strategies, config, store=store)

    def test_both_strategies_were_measured(self):
        self.assertEqual(len(self.metrics), 2)

    def test_every_row_spans_exactly_the_same_months(self):
        firsts = {str(pd.Timestamp(d['first_return']).date()) for d in self.metrics}
        lasts = {str(pd.Timestamp(d['last_return']).date()) for d in self.metrics}
        self.assertEqual(len(firsts), 1, f'ranked rows span different starts: {firsts}')
        self.assertEqual(len(lasts), 1, f'ranked rows span different ends: {lasts}')

    def test_the_late_strategy_is_named_as_the_binding_one(self):
        for d in self.metrics:
            self.assertEqual(d['window_binding'], ('HAA_G12',),
                             'the window was shortened by a specific strategy and the '
                             'report has to say which')

    def test_the_regime_panel_still_sees_the_benchmark_full_history(self):
        """SPY predates the common window; the panel must measure it anyway."""
        spy = next(d for d in self.metrics if d['name'] == 'SPY_Benchmark')
        self.assertLess(pd.Timestamp(spy['first_return_full']),
                        pd.Timestamp(spy['first_return']),
                        'the full-history pass must reach back past the ranked window, or '
                        'the panels inherit exactly the bias they exist to remove')


if __name__ == '__main__':
    unittest.main()


class TestCustomWrapsDoNotShortenTheHeadlineWindow(unittest.TestCase):
    """REPORT-002 — who is allowed to shorten every other row?

    The CUSTOM leveraged G4 wraps share UGL's 2008-12 inception and were dragging the
    ranked window from 2008-06 to 2010-02 for every entry. The flagship drawdown column
    therefore excluded the 2008 crisis, immediately after a commit whose entire purpose was
    to put 2008 into the sample -- and HAA_G12 reported -7.99% where its own history says
    -11.24%.

    That is the same coverage cost the project used to justify DELETING DAA1_G12 ("dragging
    the comparison window from 2008-07 to 2011-04"). Four entries were doing it and were
    kept, so the rule has to be stated rather than applied case by case: an entry nobody
    published does not get to shorten the measurement of entries somebody did. Nothing is
    dropped -- a late entry is still measured, over its own history, in a separate block.
    """

    @classmethod
    def _run(cls, policy):
        import main

        # SPY and BIL from the start; UWM (the 2x wrap's image) two years late.
        base = make_daily_store(n_days=2600, start='2013-01-01')
        closes, opens = base.adj_close().copy(), base.adj_open().copy()
        for frame in (closes, opens):
            if 'UWM' not in frame.columns:
                frame['UWM'] = frame['SPY'] * 0.5
        late = closes.index < pd.Timestamp('2015-01-01')
        closes.loc[late, 'UWM'] = np.nan
        opens.loc[late, 'UWM'] = np.nan
        store = PriceStore.from_adjusted(closes, opens, source='synthetic')

        config = {
            'START_DATE': eras.COMMON_ERA_START, 'END_DATE': None,
            'DATA_START_DATE': '2013-01-01', 'EXECUTION_MODE': False,
            'CURRENT_EXECUTION_DATE': None,
            'LEVERAGE_FACTOR': 1.0, 'MARGIN_BORROW_RATE': 0.06,
            'MARGIN_FOLLOWS_SIGNAL': True, 'COST_PCT_PER_SIDE': 0.001,
            'LOOKBACK_MONTHS': 13, 'EXECUTION_CONVENTION': 'next_open',
            'CASH_TICKER': 'BIL', 'COVERAGE_POLICY': 'trim', 'RF_ANNUAL_FALLBACK': 0.03,
            'RANKED_WINDOW_POLICY': policy,
        }
        prices, s_w, s_u = main.build_signal_panel(store, config)
        strategies = [main.ALL_STRATEGIES[n]()
                      for n in ('SPY_Benchmark', 'HAA_G12', 'HAA_G3_Leveraged_2X')]
        metrics, _ = main.run_backtest(prices, s_w, s_u, strategies, config, store=store)
        return {d['name']: d for d in metrics}

    def test_a_custom_wrap_is_measured_but_does_not_set_the_window(self):
        m = self._run('strategies')
        wrap = m['HAA_G3_Leveraged_2X']
        published = [m['SPY_Benchmark'], m['HAA_G12']]

        self.assertFalse(wrap['in_ranked_window'],
                         'the wrap cannot cover the shared window, so it must be reported '
                         'separately rather than sorted in beside rows it cannot be '
                         'compared with')
        self.assertTrue(all(d['in_ranked_window'] for d in published))
        for d in published:
            self.assertLess(pd.Timestamp(d['first_return']),
                            pd.Timestamp(wrap['first_return']),
                            'the published entries must reach back past the wrap')
        # ...and it is still MEASURED. Excluding it from the window is not excluding it.
        self.assertGreater(wrap['n_periods'], 0)
        # Both published entries start together here, so they legitimately tie and bind the
        # window between them. What must never appear in that set is the wrap.
        self.assertEqual(wrap['window_binding'], ('HAA_G12', 'SPY_Benchmark'))
        self.assertNotIn('HAA_G3_Leveraged_2X', wrap['window_binding'],
                         'a custom wrap may not be blamed for a window it was not allowed '
                         'to set')

    def test_policy_all_restores_one_window_over_everything(self):
        m = self._run('all')
        self.assertTrue(all(d['in_ranked_window'] for d in m.values()))
        starts = {str(pd.Timestamp(d['first_return']).date()) for d in m.values()}
        self.assertEqual(len(starts), 1, f'policy=all must share one window: {starts}')
        self.assertEqual(m['HAA_G12']['window_binding'], ('HAA_G3_Leveraged_2X',))

    def test_the_published_rows_gain_the_months_the_wrap_would_have_cost(self):
        """The whole point, measured: the same strategy, the same data, two policies."""
        strict, everything = self._run('strategies'), self._run('all')
        self.assertLess(pd.Timestamp(strict['HAA_G12']['first_return']),
                        pd.Timestamp(everything['HAA_G12']['first_return']))
        self.assertGreater(strict['HAA_G12']['n_periods'],
                           everything['HAA_G12']['n_periods'])

    def test_an_unknown_policy_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            self._run('whatever')


class TestSegmentLeaderboard(unittest.TestCase):
    """Ranking INSIDE a segment, under the same rule as the ranked table above it.

    The matrix panel answers "how did this strategy behave in each regime?". The
    leaderboard answers "who led this regime?" — and the moment you ask that, coverage
    stops being a footnote and becomes the whole question. A strategy that entered the GFC
    segment four months before its trough shows the recovery without the fall; ranking it
    against one that lived all nineteen months rebuilds date-selection bias inside the very
    panel that exists to remove it, in the direction that flatters the latecomer.
    """

    #: Three strategies over one segment: full coverage, late entry, never existed.
    #: Deliberately WITHOUT the annualised keys, so it also exercises the degrade path.
    PANEL = {
        'FULL_GOOD': {'gfc': {'return': 0.20, 'max_dd': -0.05, 'n_months': 19,
                              'partial': False}},
        'FULL_BAD': {'gfc': {'return': -0.35, 'max_dd': -0.50, 'n_months': 19,
                             'partial': False}},
        'LATECOMER': {'gfc': {'return': 0.60, 'max_dd': -0.01, 'n_months': 4,
                              'partial': True}},
        'ABSENT': {'gfc': {'na': 'inception 2010-02'}},
    }

    #: The same shape, but long enough to carry annualised metrics. Note the ordering by
    #: SORTINO is the REVERSE of the ordering by return, so a test that ranks on it cannot
    #: pass by accident.
    RICH = {
        'STEADY': {'gfc': {'return': 0.10, 'max_dd': -0.04, 'n_months': 19, 'partial': False,
                           'cagr': 0.062, 'sharpe': 1.40, 'sortino': 2.10, 'upi': 3.0,
                           'vol': 0.05}},
        'WILD': {'gfc': {'return': 0.40, 'max_dd': -0.30, 'n_months': 19, 'partial': False,
                         'cagr': 0.240, 'sharpe': 0.70, 'sortino': 0.90, 'upi': 1.0,
                         'vol': 0.32}},
    }

    def test_only_full_coverage_is_ranked(self):
        lb = eras.segment_leaderboard(self.PANEL, 'gfc')
        self.assertEqual([d['name'] for d in lb.ranked], ['FULL_GOOD', 'FULL_BAD'])
        self.assertEqual([n for n, _, _ in lb.partial], ['LATECOMER'])
        self.assertEqual([n for n, _ in lb.absent], ['ABSENT'])

    def test_the_latecomer_does_not_take_first_place(self):
        """It has the best number in the segment by a wide margin, and that is exactly why
        it must not be ranked: +60% over the last four months of a bear market is a fact
        about when it started, not about the strategy."""
        lb = eras.segment_leaderboard(self.PANEL, 'gfc')
        self.assertNotIn('LATECOMER', [d['name'] for d in lb.ranked])
        self.assertEqual(lb.partial[0][1], 4, 'the months it DID cover must be reported')

    def test_every_ranked_row_spans_the_same_months(self):
        """The property the whole design rests on, asserted rather than assumed."""
        lb = eras.segment_leaderboard(self.PANEL, 'gfc')
        self.assertEqual(len({d['n_months'] for d in lb.ranked}), 1)

    def test_ranking_by_drawdown_orders_by_least_bad(self):
        lb = eras.segment_leaderboard(self.PANEL, 'gfc', rank_by='max_dd')
        self.assertEqual([d['name'] for d in lb.ranked], ['FULL_GOOD', 'FULL_BAD'])
        self.assertGreater(lb.ranked[0]['max_dd'], lb.ranked[1]['max_dd'])
        self.assertEqual(lb.rank_by, 'max_dd')

    def test_every_headline_metric_can_order_a_long_segment(self):
        """The requirement: the same columns and the same chosen metric as the main table."""
        for key in ('cagr', 'sharpe', 'sortino', 'upi'):
            with self.subTest(rank_by=key):
                lb = eras.segment_leaderboard(self.RICH, 'gfc', rank_by=key)
                self.assertEqual(lb.rank_by, key, 'a long segment must NOT degrade')
                self.assertGreaterEqual(lb.ranked[0][key], lb.ranked[1][key])
        # Sortino puts STEADY first; total return puts WILD first. Same rows, opposite order —
        # which is the whole reason the panel has to say which key it used.
        self.assertEqual(eras.segment_leaderboard(self.RICH, 'gfc', 'sortino').ranked[0]['name'],
                         'STEADY')
        self.assertEqual(eras.segment_leaderboard(self.RICH, 'gfc', 'return').ranked[0]['name'],
                         'WILD')

    def test_volatility_is_the_one_metric_where_lower_wins(self):
        lb = eras.segment_leaderboard(self.RICH, 'gfc', rank_by='vol')
        self.assertEqual([d['name'] for d in lb.ranked], ['STEADY', 'WILD'])
        self.assertLess(lb.ranked[0]['vol'], lb.ranked[1]['vol'])

    def test_an_annualised_metric_degrades_to_return_rather_than_raising(self):
        """A segment with no annualised metrics must still produce a ranking — and must SAY
        that it ranked by return, because a Sortino column of `n/a` ordered by an invisible
        key looks like a ranking and is not one."""
        lb = eras.segment_leaderboard(self.PANEL, 'gfc', rank_by='sortino')
        self.assertEqual(lb.rank_by, 'return', 'the caller must be told it degraded')
        self.assertEqual([d['name'] for d in lb.ranked], ['FULL_GOOD', 'FULL_BAD'])

    def test_an_unknown_metric_still_raises(self):
        """Degrading is for a data limitation. An unknown key is a caller bug."""
        with self.assertRaises(ValueError):
            eras.segment_leaderboard(self.PANEL, 'gfc', rank_by='profit')

    def test_a_segment_nobody_covers_yields_an_empty_ranking_not_a_crash(self):
        panel = {'A': {'gfc': {'na': 'inception 2010-02'}}}
        lb = eras.segment_leaderboard(panel, 'gfc')
        self.assertEqual(lb.ranked, [])
        self.assertEqual(len(lb.absent), 1)

    def test_it_agrees_with_the_matrix_panel_on_real_data(self):
        """End to end: the leaderboard must be a re-ordering of the panel, never a
        re-computation that could drift from it."""
        import main
        store = PriceStore.from_daily_fixture(FROZEN_ADJ_CLOSE, FROZEN_ADJ_OPEN)
        config = {
            'START_DATE': eras.COMMON_ERA_START, 'END_DATE': None,
            'DATA_START_DATE': '2012-01-01', 'EXECUTION_MODE': False,
            'CURRENT_EXECUTION_DATE': None, 'LEVERAGE_FACTOR': 1.0,
            'MARGIN_BORROW_RATE': 0.06, 'MARGIN_FOLLOWS_SIGNAL': True,
            'COST_PCT_PER_SIDE': 0.001, 'LOOKBACK_MONTHS': 13,
            'EXECUTION_CONVENTION': 'next_open', 'CASH_TICKER': 'BIL',
            'COVERAGE_POLICY': 'trim', 'RF_ANNUAL_FALLBACK': 0.03,
        }
        prices, s_w, s_u = main.build_signal_panel(store, config)
        strategies = [main.ALL_STRATEGIES[n]() for n in ('SPY_Benchmark', 'HAA_G12')]
        metrics, _ = main.run_backtest(prices, s_w, s_u, strategies, config, store=store)
        _, last = eras.era_bounds(metrics)
        seg = eras.EQUITY_CYCLE
        panel = eras.partition_panel(metrics, seg, last)
        for s in eras.resolved_segments(seg, last):
            lb = eras.segment_leaderboard(panel, s.key)
            ranked, partial, absent = lb.ranked, lb.partial, lb.absent
            with self.subTest(segment=s.key):
                self.assertEqual(len(ranked) + len(partial) + len(absent), len(panel))
                for d in ranked:
                    self.assertEqual(d['return'], panel[d['name']][s.key]['return'])

    def test_short_segments_carry_no_annualised_metrics_on_real_data(self):
        """The guard that matters most: `contraction_2020` is 2 months long, so annualising a
        CAGR over it would raise a two-month return to the sixth power. Every segment shorter
        than SEGMENT_MIN_MONTHS must omit the annualised fields entirely, and every segment at
        or above it must carry them."""
        import main
        from common.metrics import SEGMENT_MIN_MONTHS, ANNUALISED_KEYS
        store = PriceStore.from_daily_fixture(FROZEN_ADJ_CLOSE, FROZEN_ADJ_OPEN)
        config = {
            'START_DATE': eras.COMMON_ERA_START, 'END_DATE': None,
            'DATA_START_DATE': '2012-01-01', 'EXECUTION_MODE': False,
            'CURRENT_EXECUTION_DATE': None, 'LEVERAGE_FACTOR': 1.0,
            'MARGIN_BORROW_RATE': 0.06, 'MARGIN_FOLLOWS_SIGNAL': True,
            'COST_PCT_PER_SIDE': 0.001, 'LOOKBACK_MONTHS': 13,
            'EXECUTION_CONVENTION': 'next_open', 'CASH_TICKER': 'BIL',
            'COVERAGE_POLICY': 'trim', 'RF_ANNUAL_FALLBACK': 0.03,
        }
        prices, s_w, s_u = main.build_signal_panel(store, config)
        strategies = [main.ALL_STRATEGIES[n]() for n in ('SPY_Benchmark', 'HAA_G12')]
        metrics, _ = main.run_backtest(prices, s_w, s_u, strategies, config, store=store)
        _, last = eras.era_bounds(metrics)

        seen_short = seen_long = False
        for seg in eras.SEGMENTATIONS:
            panel = eras.partition_panel(metrics, seg, last)
            for row in panel.values():
                for key, cell in row.items():
                    if 'na' in cell:
                        continue
                    has = [k for k in ANNUALISED_KEYS if k in cell]
                    with self.subTest(segment=key, n=cell['n_months']):
                        # `return`, `max_dd` and `n_months` need no annualisation and are
                        # unconditional. Everything else is gated on length.
                        self.assertIn('return', cell)
                        self.assertIn('max_dd', cell)
                        if cell['n_months'] < SEGMENT_MIN_MONTHS:
                            self.assertEqual(has, [], 'annualised a segment under the floor')
                            seen_short = True
                        else:
                            self.assertEqual(sorted(has), sorted(ANNUALISED_KEYS))
                            seen_long = True
        self.assertTrue(seen_short, 'the fixture produced no short segment to check')
        self.assertTrue(seen_long, 'the fixture produced no long segment to check')
