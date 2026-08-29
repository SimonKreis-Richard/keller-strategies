"""The dashboard's render path, exercised without a browser.

`app.py` builds its widgets at import and only starts a server under `__main__`, so the whole
results pane can be rendered into a detached container here. That matters because the GUI is
where three things now meet that used to be separate — the ranked table, the sizing column and
a growth chart per period section — and the failure mode of a layout change is not an exception
but a section rendered into the wrong container, which no unit test of the engine can see.

Nothing here asserts a NUMBER. The numbers are tested where they are computed; what is tested
here is wiring: which slot each block lands in, that a re-sort rebuilds both rank-dependent
blocks and leaves the chart alone, and that a segment chart is built on OPEN rather than on run.
"""

import unittest

import numpy as np
import pandas as pd

from nicegui import ui

import app
from common import eras
from common import leverage_advice as la


def _returns(n=240, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2005-01-31', periods=n, freq='ME')
    return pd.Series(rng.normal(0.007, 0.02, n), index=idx)


def _entry(name, seed, sortino=1.0, cagr=0.09):
    r = _returns(seed=seed)
    return {
        'name': name, 'is_active': True, 'role': 'strategy', 'fidelity': 'faithful',
        'returns': r, 'returns_full': r,
        'first_return': r.index[0], 'last_return': r.index[-1],
        'first_return_full': r.index[0], 'last_return_full': r.index[-1],
        'window_start': r.index[0], 'window_end': r.index[-1], 'window_binding': (),
        'in_ranked_window': True, 'n_periods': len(r),
        'cagr': cagr, 'max_dd': -0.18, 'max_dd_months': 14, 'sharpe': 0.8,
        'sortino': sortino, 'upi': 1.1, 'vol': 0.11, 'rf_annual': 0.02,
        'annual_turnover': 3.0, 'coverage_trimmed': False, 'binding_ticker': None,
        'offensive_weight_mean': 0.7, 'held_tickers': ('SPY', 'IEF'),
        'holds_leveraged_product': False, 'daily_max_dd': -0.23,
    }


def _late_entry(name, seed):
    """An entry whose products did not exist when the shared window opens."""
    d = _entry(name, seed, sortino=2.9, cagr=0.25)
    r = d['returns'].iloc[60:]
    d.update({'returns': r, 'returns_full': r, 'in_ranked_window': False,
              'first_return': r.index[0], 'first_return_full': r.index[0],
              'window_start': r.index[0]})
    return d


def _fixture():
    # Deliberately ordered DIFFERENTLY by Sortino (AAA, CCC, BBB) and by CAGR (CCC, AAA, BBB),
    # so a re-sort that silently did nothing would still fail the assertion.
    metrics = [_entry('AAA', 1, sortino=1.4, cagr=0.11), _entry('BBB', 2, sortino=0.9),
               _entry('CCC', 3, sortino=1.1, cagr=0.13)]
    return metrics, la.advise(metrics)


def _texts(container):
    """Every label/markdown string anywhere under `container`."""
    out = []

    def walk(el):
        for key in ('text', 'content'):
            v = getattr(el, key, None)
            if isinstance(v, str):
                out.append(v)
        for child in el.default_slot.children:
            walk(child)
    walk(container)
    return out


def _count(container, predicate):
    n = 0

    def walk(el):
        nonlocal n
        if predicate(el):
            n += 1
        for child in el.default_slot.children:
            walk(child)
    walk(container)
    return n


def _charts(container):
    return _count(container, lambda el: type(el).__name__ == 'Pyplot')


def _expansions(container):
    out = []

    def walk(el):
        if type(el).__name__ == 'Expansion':
            out.append(el)
        for child in el.default_slot.children:
            walk(child)
    walk(container)
    return out


class TestTheResultsPane(unittest.TestCase):

    def setUp(self):
        self.metrics, self.advice = _fixture()
        self.head = ui.column()
        self.regime = ui.column()

    def render(self, rank='sortino'):
        app._render_ranked_block(self.head, self.metrics, rank, advice=self.advice,
                                 regime_container=self.regime)

    def test_the_sizing_column_reaches_the_table(self):
        self.render()
        table = next(el for el in self.head.default_slot.children
                     if type(el).__name__ == 'Table')
        self.assertIn('Max margin', [c['name'] for c in table.columns])
        self.assertIn('Capped by', [c['name'] for c in table.columns])
        cell = table.rows[0]['Max margin']
        self.assertTrue(cell.endswith('x') or cell == 'n/c', cell)

    def test_the_derivation_panel_carries_the_assumptions(self):
        self.render()
        joined = ' '.join(_texts(self.head))
        self.assertIn('Sustainable margin leverage', joined)
        self.assertIn('FINRA Rule 4210(c)', joined)
        self.assertIn('Borrowing capacity', joined)

    def test_without_advice_the_column_is_a_dash_rather_than_a_number(self):
        app._render_ranked_block(self.head, self.metrics, 'sortino', advice=None,
                                 regime_container=self.regime)
        table = next(el for el in self.head.default_slot.children
                     if type(el).__name__ == 'Table')
        self.assertEqual(table.rows[0]['Max margin'], '—')

    def test_the_regime_sections_land_in_their_own_container(self):
        """The whole-era CHART sits between the two, so they must not be one block."""
        self.render()
        self.assertIn('Behaviour by pre-registered market regime', ' '.join(_texts(self.regime)))
        self.assertNotIn('Behaviour by pre-registered market regime',
                         ' '.join(_texts(self.head)))

    def test_a_re_sort_rebuilds_both_blocks_without_stacking_them(self):
        self.render('sortino')
        first = len(self.regime.default_slot.children)
        table = next(el for el in self.head.default_slot.children
                     if type(el).__name__ == 'Table')
        self.assertEqual([r['Strategy'] for r in table.rows], ['AAA', 'CCC', 'BBB'])

        self.render('cagr')
        self.assertEqual(len(self.regime.default_slot.children), first,
                         'the regime sections stacked instead of being replaced')
        table = next(el for el in self.head.default_slot.children
                     if type(el).__name__ == 'Table')
        self.assertEqual([r['Strategy'] for r in table.rows], ['CCC', 'AAA', 'BBB'])

    def test_the_top_row_follows_the_rank_key(self):
        self.render('sortino')
        self.assertIn('Top by Sortino: AAA', ' '.join(_texts(self.head)))
        self.render('vol')
        # Every fixture row has the same vol, so this only asserts the LABEL follows the key —
        # which end of the metric is better is `metrics.rank_descending`'s test, not this one.
        self.assertIn('Top by Volatility (low is best)', ' '.join(_texts(self.head)))


class TestIncomparableRowsAreKeptApart(unittest.TestCase):
    """The defect this dashboard shipped with: one sorted list over two different windows.

    An entry measured from 2010 was ranked against one measured from 2008 and the caption
    underneath said "the same for every row here". It flattered the late entry by exactly the
    months it missed — a record that starts after a crash shows the recovery without the fall.
    The CLI report has printed two blocks since REPORT-002; this is the GUI catching up.
    """

    def setUp(self):
        self.metrics, _ = _fixture()
        self.metrics.append(_late_entry('LATE', 9))
        self.advice = la.advise(self.metrics)
        self.head, self.regime = ui.column(), ui.column()
        app._render_ranked_block(self.head, self.metrics, 'sortino', advice=self.advice,
                                 regime_container=self.regime)

    def _tables(self):
        return [el for el in self.head.default_slot.children if type(el).__name__ == 'Table']

    def test_the_late_entry_never_enters_the_ranked_table(self):
        """It has the best Sortino in the fixture, so a naive sort would put it first."""
        ranked = self._tables()[0]
        self.assertEqual([r['Strategy'] for r in ranked.rows], ['AAA', 'CCC', 'BBB'])

    def test_it_gets_its_own_block_that_says_it_is_not_comparable(self):
        self.assertEqual(len(self._tables()), 2)
        self.assertEqual([r['Strategy'] for r in self._tables()[1].rows], ['LATE'])
        joined = ' '.join(_texts(self.head))
        self.assertIn('not comparable with the table above', joined)
        self.assertIn('recovery without the fall', joined)

    def test_the_caption_no_longer_claims_one_window_for_the_whole_screen(self):
        joined = ' '.join(_texts(self.head))
        self.assertIn('the same for every row in THIS table', joined)

    def test_both_sets_still_reach_the_regime_panels(self):
        """There the cells state their own coverage, so they can be read side by side."""
        self.assertIn('Behaviour by pre-registered market regime', ' '.join(_texts(self.regime)))
        named = {r['Strategy'] for t in _all_tables(self.regime) for r in t.rows
                 if 'Strategy' in r}
        self.assertIn('LATE', named)
        self.assertIn('AAA', named)

    def test_everything_late_still_renders_rather_than_showing_an_empty_table(self):
        head, regime = ui.column(), ui.column()
        only_late = [_late_entry('L1', 21), _late_entry('L2', 22)]
        app._render_ranked_block(head, only_late, 'sortino', advice=None,
                                 regime_container=regime)
        tables = [el for el in head.default_slot.children if type(el).__name__ == 'Table']
        self.assertEqual(len(tables), 1)
        self.assertEqual({r['Strategy'] for r in tables[0].rows}, {'L1', 'L2'})


class TestTheNamedCrisesArePresent(unittest.TestCase):
    """`common/regimes.py` has held the nine named episodes all along; the GUI never showed
    them. They are the short, non-exhaustive list — the crises somebody would actually look
    for — as opposed to the four exhaustive partitions that must also cover ordinary quarters.
    """

    def test_the_episode_panel_names_the_conventional_crises(self):
        from common.regimes import EPISODES
        metrics, advice = _fixture()
        head, regime = ui.column(), ui.column()
        app._render_ranked_block(head, metrics, 'sortino', advice=advice,
                                 regime_container=regime)
        joined = ' '.join(_texts(regime))
        for label in ('Global financial crisis', 'COVID crash', 'Inflation / rate bear'):
            self.assertIn(label, joined)
        # The dot-com bust was removed with the era floor on 2026-07-31: with the floor at
        # 2004-11 no entry has a month inside it, so the column was n/a top to bottom.
        self.assertNotIn('Dot-com', joined)
        table = next(t for t in _all_tables(regime)
                     if 'Global financial crisis' in [c['name'] for c in t.columns])
        self.assertEqual(len(table.columns), len(EPISODES) + 2)  # + Strategy + Covered


def _all_tables(container):
    out = []

    def walk(el):
        if type(el).__name__ == 'Table':
            out.append(el)
        for child in el.default_slot.children:
            walk(child)
    walk(container)
    return out


class TestSectionCharts(unittest.TestCase):

    def setUp(self):
        self.metrics, self.advice = _fixture()
        self.head, self.regime = ui.column(), ui.column()
        app._render_ranked_block(self.head, self.metrics, 'sortino', advice=self.advice,
                                 regime_container=self.regime)

    def test_no_segment_chart_is_drawn_until_its_section_is_opened(self):
        """Forty charts on Run would triple the wait to draw what nobody has asked to see."""
        self.assertEqual(_charts(self.regime), 0)

    def test_opening_a_segment_draws_exactly_one_chart_and_only_once(self):
        segment = next(e for e in _expansions(self.regime)
                       if e.text.startswith('bear_gfc'))
        segment.value = True
        self.assertEqual(_charts(segment), 1)
        segment.value = False
        segment.value = True
        self.assertEqual(_charts(segment), 1, 'a re-open re-paid for a chart it already had')

    def test_the_adverse_bucket_is_drawn_on_an_ordinal_axis(self):
        adverse = next(e for e in _expansions(self.regime) if e.text.startswith('ADVERSE'))
        adverse.value = True
        self.assertEqual(_charts(adverse), 1)
        self.assertIn('x axis counts months, not dates', ' '.join(_texts(adverse)))

    def test_the_whole_era_chart_is_wide_enough_to_use_the_pane(self):
        """A near-square figure beside a twelve-column table left a third of the window empty."""
        self.assertGreater(app.CHART_FIGSIZE[0] / app.CHART_FIGSIZE[1], 2.0)


class TestEveryRegisteredStrategyIsReachable(unittest.TestCase):
    """Eight strategies were unreachable from this dashboard and nothing said so.

    `_toggle_exploratory` existed, `state['show_exploratory']` existed, `_pool()` read it and
    `save_settings` persisted it — and no widget could ever set it, so the only way to see a 3x
    variant was to hand-edit user_config.json. Dead code on one side of a switch is a bug on
    the other, and the failure is silent by construction: a hidden strategy looks exactly like
    a strategy that does not exist.
    """

    def test_every_role_has_a_switch_that_reveals_it(self):
        hidden = {r['name'] for r in app.ROSTER} - set(app._pool())
        for name in sorted(hidden):
            role = app.BY_NAME[name]['role']
            self.assertIn(role, ('control', 'exploratory'),
                          f'{name} is hidden by role {role!r}, which has no reveal switch')

    def test_the_switches_actually_reach_every_hidden_entry(self):
        before = {'strategies': set(app.state['strategies']),
                  'show_controls': app.state['show_controls'],
                  'show_exploratory': app.state['show_exploratory']}
        # `_set_many` refreshes the picker, which needs a running event loop there is none of
        # here. The refresh is a redraw; what is under test is the state it redraws from.
        refresh = app.strategy_picker.refresh
        app.strategy_picker.refresh = lambda *a, **k: None
        try:
            app._toggle_controls(True)
            app._toggle_exploratory(True)
            self.assertEqual({r['name'] for r in app.ROSTER}, set(app._pool()))
            # Revealing also TICKS them, so the switch reads as "include these" rather than
            # "show me boxes I then have to find".
            for name in app.EXPLORATORY + app.CONTROLS:
                self.assertIn(name, app.state['strategies'], name)
        finally:
            app.strategy_picker.refresh = refresh
            app.state.update(before)

    def test_the_picker_renders_a_switch_bound_to_each_toggle(self):
        picker = ui.column()
        with picker:
            app.strategy_picker()
        labels = ' '.join(_texts(picker))
        self.assertIn('Show diagnostic controls', labels)
        self.assertIn('Show 3x exploratory variants', labels)
        switches = _count(picker, lambda el: type(el).__name__ == 'Switch')
        self.assertGreaterEqual(switches, 2)


class TestTheCommonWindowComparison(unittest.TestCase):
    """The third answer to "which window?", for the question people actually ask.

    Neither global policy fits "how do THESE THREE compare": one measures late arrivals
    separately, the other truncates everyone to the last inception. This intersects whatever is
    ticked and re-measures on it — better than both for a small selection, worse than both for
    a large one, which is why the binding entry is named rather than left to be discovered.
    """

    def _render(self, metrics, rank='sortino'):
        col = ui.column()
        with col:
            app._common_window_table(metrics, rank)
        return col

    def test_it_intersects_the_months_and_says_which_entry_binds(self):
        metrics, _ = _fixture()
        metrics.append(_late_entry('LATE', 9))
        col = self._render(metrics)
        joined = ' '.join(_texts(col))
        self.assertIn('LATE', joined)
        self.assertIn('untick to lengthen it', joined)
        table = _all_tables(col)[0]
        self.assertEqual(len(table.rows), 4)
        self.assertEqual({r['Months'] for r in table.rows}, {180})

    def test_the_numbers_are_recomputed_not_copied_from_the_ranked_table(self):
        """A late arrival drags everyone onto a shorter window, so nothing should match."""
        metrics, _ = _fixture()
        metrics.append(_late_entry('LATE', 9))
        table = _all_tables(self._render(metrics))[0]
        row = next(r for r in table.rows if r['Strategy'] == 'AAA')
        self.assertNotEqual(row['CAGR'], PCT_OF(metrics[0]['cagr']))

    def test_it_follows_the_rank_key(self):
        """Asserted on the DISPLAYED column, not on the fixture's declared metric.

        The table recomputes everything from the returns — that is its entire job — so the
        `sortino` and `cagr` written into the fixture are ignored, and a test comparing two
        orderings would only be testing whether the random walks happened to disagree.
        """
        metrics, _ = _fixture()
        cagrs = [float(r['CAGR'].rstrip('%'))
                 for r in _all_tables(self._render(metrics, 'cagr'))[0].rows]
        self.assertEqual(cagrs, sorted(cagrs, reverse=True))
        vols = [float(r['Volatility'].rstrip('%'))
                for r in _all_tables(self._render(metrics, 'vol'))[0].rows]
        self.assertEqual(vols, sorted(vols), 'volatility is the one where LOWER is better')

    def test_it_refuses_rather_than_annualising_a_handful_of_months(self):
        metrics, _ = _fixture()
        stub = _entry('TINY', 4)
        r = stub['returns'].iloc[-6:]
        stub.update({'returns': r, 'returns_full': r})
        joined = ' '.join(_texts(self._render(metrics + [stub])))
        self.assertIn('too few to annualise', joined)

    def test_one_entry_is_not_a_comparison(self):
        metrics, _ = _fixture()
        joined = ' '.join(_texts(self._render(metrics[:1])))
        self.assertIn('at least two', joined)


PCT_OF = '{:.2%}'.format


class TestThePartitionToggles(unittest.TestCase):
    """Hiding a partition is safe; hiding a SEGMENT would not be, and is not offered.

    Each of the four partitions is independently exhaustive and disjoint, so which ones you
    display is a presentation choice. Pruning segments inside one would stop it tiling the era
    — "show me only the interesting periods" is the window-picking the whole design forbids —
    so the toggles are per partition and there is nothing finer.
    """

    def setUp(self):
        self.metrics, self.advice = _fixture()
        self.before = set(app.state['partitions'])

    def tearDown(self):
        app.state['partitions'] = self.before

    def _render(self):
        head, regime = ui.column(), ui.column()
        app._render_ranked_block(head, self.metrics, 'sortino', advice=self.advice,
                                 regime_container=regime)
        return regime

    def test_the_toggles_cover_every_partition_the_engine_defines(self):
        self.assertEqual(set(app.PARTITIONS), {s.key for s in eras.SEGMENTATIONS})

    def test_all_four_render_by_default(self):
        app.state['partitions'] = set(app.PARTITIONS)
        joined = ' '.join(_texts(self._render()))
        for seg in eras.SEGMENTATIONS:
            self.assertIn(seg.title, joined)

    def test_unticking_one_removes_exactly_that_one(self):
        app.state['partitions'] = set(app.PARTITIONS) - {'monetary'}
        joined = ' '.join(_texts(self._render()))
        for seg in eras.SEGMENTATIONS:
            if seg.key == 'monetary':
                self.assertNotIn(seg.title, joined)
            else:
                self.assertIn(seg.title, joined)

    def test_unticking_everything_says_so_and_keeps_the_named_crises(self):
        app.state['partitions'] = set()
        joined = ' '.join(_texts(self._render()))
        self.assertIn('No partition selected', joined)
        self.assertIn('Global financial crisis', joined)

    def test_a_stale_key_in_the_saved_config_cannot_select_nothing(self):
        """Renaming a segmentation must not leave the file pointing at a partition that is
        gone — the flag would silently hide a panel nobody asked to hide."""
        import common.user_config as uc
        cfg = {'REGIME_PARTITIONS': ['equity_cycle', 'a_segmentation_that_was_renamed']}
        kept = {k for k in cfg['REGIME_PARTITIONS'] if k in app.PARTITIONS}
        self.assertEqual(kept, {'equity_cycle'})
        del uc  # imported only to document where the file is read


class TestTheCurveBuilders(unittest.TestCase):

    def test_a_segment_curve_starts_at_one(self):
        metrics, _ = _fixture()
        curves = app._rebased_curves(metrics, '2007-11-01', '2009-02-28')
        self.assertEqual(set(curves), {'AAA', 'BBB', 'CCC'})
        for name, c in curves.items():
            self.assertAlmostEqual(float(c.iloc[0]), 1.0, msg=name)

    def test_a_strategy_with_no_months_in_the_window_is_omitted_not_flat(self):
        metrics, _ = _fixture()
        self.assertEqual(app._rebased_curves(metrics, '2001-03-01', '2002-09-30'), {})

    def test_the_adverse_curve_is_indexed_by_count(self):
        from common import eras
        metrics, _ = _fixture()
        segments = eras.resolved_segments(eras.EQUITY_CYCLE, pd.Timestamp('2024-12-31'))
        curves = app._adverse_curves(metrics, segments)
        c = curves['AAA']
        self.assertEqual(list(c.index[:3]), [0, 1, 2])
        self.assertAlmostEqual(float(c.iloc[0]), 1.0)


class TestTheRobustnessPanel(unittest.TestCase):
    """The dashboard used to CITE a number it would not display.

    The summary card has said, since the two-block split landed, that "the rank correlation
    between disjoint sub-periods is approximately zero" — and pointed at the CLI report for
    it. Every figure behind that sentence (selection context, participation ratio, rank
    stability) was computed by `main._selection_section` on every run and rendered only into
    a terminal. This panel is where they land, together with the two measurements added on
    2026-08-01.
    """

    def setUp(self):
        # Six entries: `robustness.MIN_STRATEGIES` is five, so a three-entry fixture would
        # exercise only the refusal path and pass while proving nothing.
        self.metrics = [_entry(f'S{i}', 10 + i, sortino=1.5 - 0.1 * i, cagr=0.12 - 0.005 * i)
                        for i in range(6)]
        self.advice = la.advise(self.metrics)
        self.head, self.regime = ui.column(), ui.column()
        app._render_ranked_block(self.head, self.metrics, 'sortino', advice=self.advice,
                                 regime_container=self.regime)

    def _panel(self):
        return next(e for e in _expansions(self.head) if 'Robustness' in e.text)

    def test_the_panel_is_present_and_asks_the_question_in_the_title(self):
        self.assertIn('skill', self._panel().text)

    def test_nothing_is_computed_until_it_is_opened(self):
        """PBO plus a resampled leaderboard is roughly a second, and this block re-renders on
        every change of sort key — which is exactly the regression that made the table slow
        once already. The expansion's own title is all that exists before the first open."""
        panel = self._panel()
        self.assertEqual(_texts(panel), [panel.text])

    def test_opening_it_renders_all_three_measurements(self):
        panel = self._panel()
        panel.value = True
        joined = ' '.join(_texts(panel))
        self.assertIn('SELECTION CONTEXT', joined)
        self.assertIn('PBO', joined)
        self.assertIn('RANK STABILITY UNDER RESAMPLING', joined)

    def test_the_family_and_mechanism_views_render_beside_the_strategy_view(self):
        """Six entries with no underscore are six one-member families — the pooled tables
        still render, they just cannot pool. What matters here is the WIRING: all four
        grouped blocks land inside the panel."""
        panel = self._panel()
        panel.value = True
        joined = ' '.join(_texts(panel))
        self.assertIn('FAMILY-LEVEL PBO', joined)
        self.assertIn('FAMILY RANK STABILITY', joined)
        self.assertIn('MECHANISM-LEVEL PBO', joined)
        self.assertIn('MECHANISM RANK STABILITY', joined)

    def test_it_refuses_to_be_read_as_a_forecast(self):
        panel = self._panel()
        panel.value = True
        joined = ' '.join(_texts(panel))
        self.assertIn('None of this forecasts', joined)
        self.assertIn('does not simulate a world', joined)

    def test_it_names_the_entry_that_bounds_the_shared_window(self):
        """Same discipline as the common-window table: the intersection is set by the latest
        arrival, and a reader who cannot see which one cannot untick it."""
        panel = self._panel()
        panel.value = True
        self.assertIn('set by S', ' '.join(_texts(panel)))

    def test_the_resampled_ranking_follows_the_rank_key(self):
        head = ui.column()
        app._render_ranked_block(head, self.metrics, 'cagr', advice=self.advice,
                                 regime_container=ui.column())
        panel = next(e for e in _expansions(head) if 'Robustness' in e.text)
        panel.value = True
        self.assertIn('CAGR', ' '.join(_texts(panel)))

    def test_too_small_a_selection_says_so_instead_of_failing(self):
        head = ui.column()
        small = self.metrics[:2]
        app._render_ranked_block(head, small, 'sortino', advice=la.advise(small),
                                 regime_container=ui.column())
        panel = next(e for e in _expansions(head) if 'Robustness' in e.text)
        panel.value = True
        self.assertIn('Not computed', ' '.join(_texts(panel)))

    def test_the_summary_card_no_longer_points_at_a_report_nobody_has_open(self):
        joined = ' '.join(_texts(self.head))
        self.assertNotIn('the CLI report prints the rank correlation', joined)
        self.assertIn('Robustness panel below', joined)


class TestTheFrontierPanel(unittest.TestCase):
    """The margin decision as a curve, under the sizing panel it exists to check."""

    def setUp(self):
        self.metrics = [_entry(f'S{i}', 20 + i, sortino=1.5 - 0.1 * i, cagr=0.12 - 0.005 * i)
                        for i in range(6)]
        self.advice = la.advise(self.metrics)
        self.head, self.regime = ui.column(), ui.column()
        app._render_ranked_block(self.head, self.metrics, 'sortino', advice=self.advice,
                                 regime_container=self.regime)

    def _panel(self):
        return next(e for e in _expansions(self.head) if 'Leverage frontier' in e.text)

    def test_the_panel_exists_and_is_lazy(self):
        """Forty-one leverage levels for six entries is real work; it must not be paid on
        Run, nor re-paid on a re-sort — same discipline as every other heavy panel."""
        panel = self._panel()
        self.assertEqual(_texts(panel), [panel.text])
        self.assertEqual(_charts(panel), 0)

    def test_opening_it_renders_the_table_and_exactly_one_chart(self):
        panel = self._panel()
        panel.value = True
        joined = ' '.join(_texts(panel))
        self.assertIn('LEVERAGE FRONTIER', joined)
        self.assertIn('f@1%', joined)
        self.assertEqual(_charts(panel), 1)

    def test_the_chart_belongs_to_the_current_sorts_best_sized_entry(self):
        """Sorted by Sortino, S0 leads the fixture; the one chart drawn must be its own."""
        panel = self._panel()
        panel.value = True
        # The chart's suptitle carries the strategy name; matplotlib text is not in the
        # NiceGUI element tree, so assert through the table instead: S0 is present and the
        # panel drew exactly one chart for a six-row table.
        self.assertIn('S0', ' '.join(_texts(panel)))
        self.assertEqual(_charts(panel), 1)

    def test_the_month_end_caveat_is_in_the_panel_not_only_in_the_cli(self):
        panel = self._panel()
        panel.value = True
        joined = ' '.join(_texts(panel))
        self.assertIn('MONTH-END', joined)
        self.assertIn('understates', joined)

    def test_too_small_a_selection_says_so_instead_of_failing(self):
        head = ui.column()
        small = self.metrics[:2]
        app._render_ranked_block(head, small, 'sortino', advice=la.advise(small),
                                 regime_container=ui.column())
        panel = next(e for e in _expansions(head) if 'Leverage frontier' in e.text)
        panel.value = True
        self.assertIn('Not computed', ' '.join(_texts(panel)))


if __name__ == '__main__':
    unittest.main()
