"""
Tests for `common/robustness.py` — PBO by CSCV, and rank stability under joint resampling.

Two properties get most of the attention, because they are the two that make the numbers
worth printing:

* **Determinism.** PBO has no RNG at all, and the rank bootstrap has a required seed. A
  robustness measure you can re-roll until it flatters you is not a robustness measure.
* **Recovering a KNOWN answer.** A frame of pure noise must produce a PBO near the 50% a
  coin flip produces; a frame with one genuinely superior strategy must produce a low one.
  Both are constructed here rather than asserted, so the implementation is checked against
  what the statistic is supposed to mean rather than against its own last output.
"""
import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import robustness as rb                                    # noqa: E402
from common.margin_sizing import NotCalculable                         # noqa: E402


def _frame(n_strat=8, n_months=200, seed=7, edge=None):
    """A returns frame. `edge` adds a monthly drift to column 0 and to nobody else."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2008-01-31', periods=n_months, freq='ME')
    data = rng.normal(0.006, 0.04, size=(n_months, n_strat))
    if edge:
        data[:, 0] += edge
    return pd.DataFrame(data, index=idx,
                        columns=[f'S{i}' for i in range(n_strat)])


def _entry(name, series, **over):
    d = {'name': name, 'returns': series, 'returns_full': series, 'is_active': True,
         'role': 'strategy', 'in_ranked_window': True}
    d.update(over)
    return d


class TestTheSharedRectangle(unittest.TestCase):

    def test_it_names_the_entry_that_set_the_start(self):
        long_ = _frame(1, 200, seed=1).iloc[:, 0]
        short = long_.iloc[60:]
        frame, binding = rb.common_frame(
            [_entry(f'E{i}', long_) for i in range(5)] + [_entry('LATE', short)])
        self.assertEqual(binding, 'LATE')
        self.assertEqual(len(frame), len(short))

    def test_it_prefers_the_entries_the_ranked_table_ranks(self):
        """The default that buys back 2008.

        Admitting a late arrival lets it set the intersection for everyone; on the real
        registry that moved the shared window from 2008-07 to 2010-02 and cross-validated a
        momentum leaderboard over a window containing no crisis at all.
        """
        long_ = _frame(1, 200, seed=2).iloc[:, 0]
        late = long_.iloc[60:]
        rows = [_entry(f'E{i}', long_) for i in range(6)] + \
               [_entry('LATE', late, in_ranked_window=False)]
        frame, binding = rb.common_frame(rows)
        self.assertEqual(len(frame), 200)
        self.assertNotIn('LATE', frame.columns)
        # ... and switching it off restores the shorter, wider frame.
        wide, wide_binding = rb.common_frame(rows, ranked_only=False)
        self.assertEqual(len(wide), len(late))
        self.assertEqual(wide_binding, 'LATE')

    def test_the_restriction_yields_rather_than_leaving_too_few_to_compare(self):
        long_ = _frame(1, 200, seed=3).iloc[:, 0]
        rows = [_entry('A', long_), _entry('B', long_)] + \
               [_entry(f'L{i}', long_, in_ranked_window=False) for i in range(5)]
        frame, _ = rb.common_frame(rows)
        self.assertEqual(frame.shape[1], 7)

    def test_it_excludes_controls_and_exploratory_entries(self):
        s = _frame(1, 200, seed=4).iloc[:, 0]
        rows = [_entry(f'E{i}', s) for i in range(5)] + \
               [_entry('BENCH', s, role='benchmark'),
                _entry('WILD', s, role='exploratory'),
                _entry('OFF', s, is_active=False)]
        frame, _ = rb.common_frame(rows)
        self.assertEqual(sorted(frame.columns), [f'E{i}' for i in range(5)])

    def test_it_refuses_a_leaderboard_too_small_to_cross_validate(self):
        s = _frame(1, 200, seed=5).iloc[:, 0]
        with self.assertRaises(NotCalculable):
            rb.common_frame([_entry('A', s), _entry('B', s)])


class TestPBORecoversTheAnswerItShould(unittest.TestCase):

    def test_pure_noise_averages_out_near_a_coin_flip(self):
        """Under the null the in-sample winner is the luckiest, and lands below the OOS
        median about half the time — but that is an expectation over REPEATED SAMPLES, not a
        property of any one of them.

        Measured across twelve draws, single-sample PBO ranges 0.10 to 0.95 with a median of
        0.43. Whichever noise strategy happens to hold the highest full-sample mean tends to
        win both halves of that particular draw, so one seed says almost nothing. Averaging
        first is what makes this an assertion about the statistic rather than about seed 11.
        """
        vals = [rb.pbo(frame=_frame(10, 240, seed=100 + s)).pbo for s in range(12)]
        self.assertGreater(float(np.mean(vals)), 0.30)
        self.assertLess(float(np.mean(vals)), 0.70)

    def test_one_genuinely_better_strategy_drives_it_down(self):
        """A 1.2%/month edge on one column out of ten. It wins in-sample because it is
        better, so it keeps winning out-of-sample, so lambda is positive nearly always. And
        unlike the null, this holds seed by seed — which is the discrimination that matters.
        """
        for s in range(6):
            res = rb.pbo(frame=_frame(10, 240, seed=100 + s, edge=0.012))
            self.assertLess(res.pbo, 0.15, f'seed {s}')
            self.assertGreater(res.winner_share[0][1], 0.80, f'seed {s}')

    def test_the_null_and_the_edge_are_far_apart_on_the_same_draw(self):
        """Same underlying noise, one column lifted. PBO must fall a long way."""
        for s in range(6):
            noise = rb.pbo(frame=_frame(10, 240, seed=200 + s)).pbo
            real = rb.pbo(frame=_frame(10, 240, seed=200 + s, edge=0.012)).pbo
            self.assertLess(real, noise, f'seed {s}')

    def test_no_degradation_slope_is_reported(self):
        """It was implemented, measured, and removed the same day: over an edge sweep it ran
        -0.56 (null) -> -0.29 (mild edge) -> -1.00 (strong edge), i.e. non-monotone and
        pinned at its worst-looking value on the case with the STRONGEST real edge. For a
        fixed strategy the two equal halves satisfy IS + OOS = 2 x full-sample mean, so once
        one entry wins every split the regression recovers that identity and nothing else.

        Pinned as a test because the statistic is an obvious thing to add back.
        """
        res = rb.pbo(frame=_frame(8, 200, seed=14))
        self.assertFalse(hasattr(res, 'degradation_slope'))
        self.assertNotIn('degradation', '\n'.join(rb.pbo_lines(res)))


class TestPBOIsDeterministic(unittest.TestCase):

    def test_two_calls_give_bit_identical_answers(self):
        """The property that makes this number unshoppable: no seed exists to change."""
        frame = _frame(9, 200, seed=21)
        a, b = rb.pbo(frame=frame), rb.pbo(frame=frame)
        self.assertEqual(a.pbo, b.pbo)
        self.assertEqual(a.n_splits, b.n_splits)
        np.testing.assert_array_equal(a.logits, b.logits)

    def test_the_split_count_is_the_binomial_coefficient(self):
        res = rb.pbo(frame=_frame(9, 200, seed=22), n_blocks=10)
        self.assertEqual(res.n_splits, math.comb(10, 5))

    def test_every_split_has_its_mirror(self):
        """'Combinatorially symmetric' is the S/2-of-S enumeration, and it is what removes
        any question of which half came first."""
        res = rb.pbo(frame=_frame(9, 200, seed=23), n_blocks=8)
        self.assertEqual(res.n_splits, math.comb(8, 4))
        self.assertEqual(res.n_splits % 2, 0)


class TestPBORefusesRatherThanGuessing(unittest.TestCase):

    def test_an_odd_block_count_cannot_split_in_half(self):
        with self.assertRaises(NotCalculable):
            rb.pbo(frame=_frame(8, 200), n_blocks=15)

    def test_too_few_strategies_to_rank(self):
        with self.assertRaises(NotCalculable):
            rb.pbo(frame=_frame(3, 200))

    def test_blocks_too_short_to_carry_a_sharpe(self):
        with self.assertRaisesRegex(NotCalculable, 'months per block'):
            rb.pbo(frame=_frame(8, 40), n_blocks=16)

    def test_it_refuses_a_block_count_whose_enumeration_would_explode(self):
        with self.assertRaises(NotCalculable):
            rb.pbo(frame=_frame(8, 600), n_blocks=24)

    def test_the_remainder_is_dropped_from_the_front_and_counted(self):
        """The earliest months go, the recent ones stay, and the count is reported rather
        than absorbed."""
        res = rb.pbo(frame=_frame(8, 205), n_blocks=16)
        self.assertEqual(res.dropped_months, 205 - 12 * 16)
        self.assertEqual(res.block_months, 12)


class TestTheSufficientStatisticsShortcut(unittest.TestCase):

    def test_a_block_sharpe_matches_the_direct_computation(self):
        """The matmul over block totals must give exactly what slicing the returns gives,
        or CSCV is fast and wrong."""
        frame = _frame(4, 120, seed=31)
        values = frame.to_numpy(dtype=float)
        count, s1, s2, dropped = rb._sufficient_statistics(values, 8)
        self.assertEqual(dropped, 0)
        mask = np.zeros((8, 1))
        mask[[0, 2, 5, 7], 0] = 1.0
        got = rb._sharpe_from_totals(count * 4, s1 @ mask, s2 @ mask)[:, 0]

        rows = np.concatenate([values[b * count:(b + 1) * count] for b in (0, 2, 5, 7)])
        want = rows.mean(axis=0) * math.sqrt(12.0) / rows.std(axis=0, ddof=1)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_a_non_calculable_sharpe_ranks_worst_and_never_best(self):
        """`np.argsort` places NaN LAST, which in a descending ranking would silently hand
        the top of the table to a strategy whose Sharpe could not be computed."""
        a = np.array([[1.0], [np.nan], [0.5]])
        self.assertEqual(rb._ranks_descending(a)[1, 0], 1)
        self.assertEqual(rb._ranks_descending(a)[0, 0], 3)


class TestTheResampledRanking(unittest.TestCase):

    def test_the_resampling_is_joint_across_strategies(self):
        """The property the whole exercise rests on: ONE set of time indices, applied to
        every column. Two identical columns must therefore come back with bit-identical SCORE
        distributions — impossible under independent draws, which would give them different
        histories and so different quantiles.

        Their RANKS still differ by one, because ties break by position rather than averaging
        (which is what keeps `p_first` summing to 1). That is the tie rule showing through,
        not the resampling failing, so it is asserted rather than tolerated.
        """
        frame = _frame(6, 180, seed=41)
        frame['CLONE'] = frame['S0']
        res = rb.rank_bootstrap(frame=frame, binding='x', rank_key='sharpe', n_paths=200)
        rows = {r['name']: r for r in res.rows}
        self.assertEqual(rows['S0']['p5'], rows['CLONE']['p5'])
        self.assertEqual(rows['S0']['p95'], rows['CLONE']['p95'])
        self.assertEqual(rows['S0']['observed'], rows['CLONE']['observed'])
        self.assertLessEqual(abs(rows['S0']['median_rank'] - rows['CLONE']['median_rank']), 1.0)

    def test_independent_resampling_would_have_been_detected(self):
        """The negative control for the test above: draw each column with its own indices and
        the clone's quantiles come apart. Without this, the assertion of equality could be
        passing for a reason other than jointness."""
        frame = _frame(6, 180, seed=41)
        frame['CLONE'] = frame['S0']
        from common.margin_sizing import stationary_bootstrap_indices
        rng = np.random.default_rng(rb.SEED)
        vals = frame.to_numpy(dtype=float)
        q = []
        for i in range(vals.shape[1]):
            idx = stationary_bootstrap_indices(len(vals), len(vals), 12, 200, rng)
            s = rb._path_metrics(vals[idx, i], 'sharpe', 0.0)
            q.append(float(np.quantile(s, 0.05)))
        names = list(frame.columns)
        self.assertNotEqual(q[names.index('S0')], q[names.index('CLONE')])

    def test_the_seed_makes_it_reproducible(self):
        frame = _frame(6, 180, seed=42)
        a = rb.rank_bootstrap(frame=frame, binding='x', n_paths=300)
        b = rb.rank_bootstrap(frame=frame, binding='x', n_paths=300)
        self.assertEqual([r['p_top_k'] for r in a.rows], [r['p_top_k'] for r in b.rows])

    def test_a_different_seed_moves_it(self):
        frame = _frame(6, 180, seed=43)
        a = rb.rank_bootstrap(frame=frame, binding='x', n_paths=300, seed=1)
        b = rb.rank_bootstrap(frame=frame, binding='x', n_paths=300, seed=2)
        self.assertNotEqual([r['p_top_k'] for r in a.rows], [r['p_top_k'] for r in b.rows])

    def test_an_unseeded_bootstrap_is_refused(self):
        with self.assertRaisesRegex(NotCalculable, 'seed'):
            rb.rank_bootstrap(frame=_frame(6, 180), binding='x', seed=None)

    def test_a_dominant_strategy_is_top_k_nearly_always(self):
        res = rb.rank_bootstrap(frame=_frame(8, 240, seed=44, edge=0.015), binding='x',
                                n_paths=500)
        self.assertEqual(res.rows[0]['name'], 'S0')
        self.assertGreater(res.rows[0]['p_top_k'], 0.95)

    def test_observed_rank_one_does_not_imply_a_high_probability_of_top_k(self):
        """The entire reason the two columns sit side by side: on noise, whoever won is
        rank 1 and is nonetheless unremarkable under resampling."""
        res = rb.rank_bootstrap(frame=_frame(10, 240, seed=45), binding='x', n_paths=500)
        self.assertEqual(res.rows[0]['observed_rank'], 1)
        self.assertLess(res.rows[0]['p_first'], 0.60)

    def test_probabilities_sum_the_way_ranks_require(self):
        res = rb.rank_bootstrap(frame=_frame(7, 180, seed=46), binding='x', n_paths=400)
        self.assertAlmostEqual(sum(r['p_first'] for r in res.rows), 1.0, places=9)
        self.assertAlmostEqual(sum(r['p_top_k'] for r in res.rows), 3.0, places=9)

    def test_lower_is_better_metrics_rank_from_the_other_end(self):
        """`vol` is the one metric in RANK_KEYS you want less of, and it must not be silently
        ranked as though more were better."""
        frame = _frame(6, 180, seed=47)
        frame['CALM'] = frame['S0'] * 0.2
        res = rb.rank_bootstrap(frame=frame, binding='x', rank_key='vol', n_paths=300)
        self.assertEqual(res.rows[0]['name'], 'CALM')
        self.assertGreater(res.rows[0]['p_first'], 0.99)

    def test_every_rank_key_the_dashboard_offers_is_supported(self):
        from common.metrics import RANK_KEYS
        frame = _frame(6, 180, seed=48)
        for key in RANK_KEYS:
            res = rb.rank_bootstrap(frame=frame, binding='x', rank_key=key, n_paths=100)
            self.assertEqual(len(res.rows), 6, key)
            self.assertTrue(all(np.isfinite(r['observed']) for r in res.rows), key)

    def test_an_unknown_rank_key_is_refused_by_name(self):
        with self.assertRaisesRegex(NotCalculable, 'rank_key'):
            rb.rank_bootstrap(frame=_frame(6, 180), binding='x', rank_key='calmar')

    def test_the_observed_value_matches_calculate_metrics(self):
        """The resampler's own metric on the UNRESAMPLED path must equal what the rest of the
        repository reports for that series, or the observed column and the table above it are
        two different measurements wearing one name."""
        from common.metrics import calculate_metrics
        frame = _frame(5, 180, seed=49)
        res = rb.rank_bootstrap(frame=frame, binding='x', rank_key='sortino', n_paths=50)
        for row in res.rows:
            want = calculate_metrics(frame[row['name']])['sortino']
            self.assertAlmostEqual(row['observed'], want, places=9, msg=row['name'])


class TestTheBootstrapIsTheONEBootstrap(unittest.TestCase):

    def test_it_reuses_the_draw_margin_sizing_uses(self):
        """`stationary_bootstrap_indices` was extracted from `_stationary_bootstrap_paths` so
        the leaderboard resampler and the drawdown quantile cannot disagree about what a
        resampled month is. This pins that the extraction did not change a single draw."""
        from common import margin_sizing as ms
        r = np.asarray(_frame(1, 120, seed=51).iloc[:, 0], dtype=float)
        rng_a = np.random.default_rng(999)
        want = r[ms.stationary_bootstrap_indices(len(r), 60, 12, 25, rng_a)]
        rng_b = np.random.default_rng(999)
        got = ms._stationary_bootstrap_paths(r, 60, 12, 25, rng_b)
        np.testing.assert_array_equal(got, want)

    def test_the_indices_stay_inside_the_series(self):
        rng = np.random.default_rng(7)
        from common.margin_sizing import stationary_bootstrap_indices
        idx = stationary_bootstrap_indices(37, 200, 12, 50, rng)
        self.assertEqual(idx.shape, (50, 200))
        self.assertGreaterEqual(int(idx.min()), 0)
        self.assertLess(int(idx.max()), 37)

    def test_blocks_are_contiguous_far_more_often_than_chance(self):
        """A stationary bootstrap with a 12-month expected block must step +1 about 11 times
        out of 12. Uniform resampling would step +1 once in n."""
        rng = np.random.default_rng(8)
        from common.margin_sizing import stationary_bootstrap_indices
        idx = stationary_bootstrap_indices(120, 400, 12, 40, rng)
        step = (idx[:, 1:] - idx[:, :-1]) % 120
        self.assertGreater(float((step == 1).mean()), 0.80)


class TestTheReportLines(unittest.TestCase):

    def test_pbo_lines_state_the_binding_entry_and_the_split_count(self):
        res = rb.pbo(frame=_frame(8, 200, seed=61), binding='LATE_ONE')
        text = '\n'.join(rb.pbo_lines(res))
        self.assertIn('LATE_ONE', text)
        self.assertIn('12870', text)
        self.assertIn('PBO', text)

    def test_pbo_lines_say_what_the_measure_cannot_see(self):
        """The limit is not a footnote — a low PBO says our ranking is stable, NOT that the
        strategies are unoverfitted, and the difference is the whole reason the number is
        safe to publish."""
        text = '\n'.join(rb.pbo_lines(rb.pbo(frame=_frame(8, 200, seed=62), binding='b')))
        self.assertIn('It cannot see the choices made before', text)

    def test_rank_lines_refuse_to_read_as_a_forecast(self):
        res = rb.rank_bootstrap(frame=_frame(6, 180, seed=63), binding='b', n_paths=100)
        text = '\n'.join(rb.rank_lines(res))
        self.assertIn('does not simulate a world', text)
        self.assertIn(str(res.seed), text)

    def test_rank_lines_can_be_truncated_for_a_narrow_panel(self):
        res = rb.rank_bootstrap(frame=_frame(9, 180, seed=64), binding='b', n_paths=100)
        self.assertLess(len(rb.rank_lines(res, limit=3)), len(rb.rank_lines(res)))


class TestTheRfConvention(unittest.TestCase):
    """AUD-02: the ordering being cross-validated must be netted against the SAME cash rate
    as the leaderboard it cross-validates. rf=0 flatters low-vol entries (the shift is
    rf/vol), which is enough to flip near-ties in `winner_share`."""

    def test_realised_rf_reads_the_first_ranked_row(self):
        rows = [{'name': 'A', 'in_ranked_window': False, 'rf_annual': 0.99},
                {'name': 'B', 'in_ranked_window': True, 'rf_annual': 0.0139},
                {'name': 'C', 'in_ranked_window': True, 'rf_annual': 0.5}]
        self.assertEqual(rb.realised_rf(rows), 0.0139)

    def test_realised_rf_falls_back_to_the_first_row_then_to_zero(self):
        self.assertEqual(rb.realised_rf([{'name': 'A', 'rf_annual': 0.02}]), 0.02)
        self.assertEqual(rb.realised_rf([{'name': 'A'}]), 0.0)
        self.assertEqual(rb.realised_rf([]), 0.0)

    @staticmethod
    def _flip_frame():
        """LOWVOL out-Sharpes HIGHRET at rf=0 and loses to it at rf=6% — the audit's
        mechanism, constructed rather than asserted."""
        rng = np.random.default_rng(65)
        idx = pd.date_range('2008-01-31', periods=200, freq='ME')
        data = {'LOWVOL': rng.normal(0.003, 0.01, 200),
                'HIGHRET': rng.normal(0.009, 0.04, 200)}
        for i in range(4):
            data[f'N{i}'] = rng.normal(0.002, 0.04, 200)
        return pd.DataFrame(data, index=idx)

    def test_the_rate_can_flip_the_in_sample_winner(self):
        frame = self._flip_frame()
        at_zero = rb.pbo(frame=frame, binding='b', rf_annual=0.0)
        at_real = rb.pbo(frame=frame, binding='b', rf_annual=0.06)
        self.assertEqual(at_zero.winner_share[0][0], 'LOWVOL')
        self.assertNotEqual(at_real.winner_share[0][0], 'LOWVOL')

    def test_the_lines_say_which_rate_was_used(self):
        frame = self._flip_frame()
        self.assertIn('net of rf 6.00%/yr',
                      '\n'.join(rb.pbo_lines(rb.pbo(frame=frame, binding='b',
                                                    rf_annual=0.06))))
        self.assertIn('rf 6.00%/yr',
                      '\n'.join(rb.rank_lines(rb.rank_bootstrap(
                          frame=frame, binding='b', n_paths=50, rf_annual=0.06))))

    def test_the_cli_section_nets_at_the_run_rf_not_zero(self):
        """The call-site pin: `_robustness_section` must resolve rf off the metrics it was
        given, so a default of 0.0 can never silently return."""
        import main
        s = _frame(1, 200, seed=66).iloc[:, 0]
        metrics = [_entry(f'E{i}', s, rf_annual=0.05) for i in range(5)]
        text = '\n'.join(main._robustness_section(metrics, 'sharpe'))
        self.assertIn('net of rf 5.00%/yr', text)


class TestFamiliesAndMechanisms(unittest.TestCase):

    def test_family_is_the_prefix_before_the_first_underscore(self):
        self.assertEqual(rb.family_of('HAA_G12'), 'HAA')
        self.assertEqual(rb.family_of('GEM_G2_Classic'), 'GEM')
        self.assertEqual(rb.family_of('PAA2_G12'), 'PAA2')
        self.assertEqual(rb.family_of('DM_G3_Leveraged_2X'), 'DM')

    def test_every_registered_entry_has_a_named_mechanism(self):
        """The map must cover the REGISTRY, not most of it — an unknown family silently
        forming its own group is the designed fallback for new families, not an acceptable
        state for the current ones."""
        import main
        for name in main.ALL_STRATEGIES:
            self.assertIn(rb.family_of(name), rb.FAMILY_MECHANISM, name)

    def test_an_unknown_family_groups_under_its_own_name(self):
        self.assertEqual(rb.mechanism_of('XYZ_G4'), 'XYZ')

    def test_daa_and_baa_pool_into_the_same_mechanism(self):
        """Both protect with a dedicated canary basket — that is the pooling that makes the
        mechanism view say something the family view cannot."""
        self.assertEqual(rb.mechanism_of('DAA_G12'), rb.mechanism_of('BAA_G12'))
        self.assertNotEqual(rb.mechanism_of('DAA_G12'), rb.mechanism_of('HAA_G12'))


def _family_frame(n_months=240, seed=71, edge_family=None, edge=0.010):
    """Ten strategies in five families of two. `edge_family` lifts BOTH its members."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2008-01-31', periods=n_months, freq='ME')
    fams = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
    cols = [f'{f}_{i}' for f in fams for i in (1, 2)]
    data = rng.normal(0.006, 0.04, size=(n_months, len(cols)))
    if edge_family:
        for j, c in enumerate(cols):
            if c.startswith(edge_family):
                data[:, j] += edge
    return pd.DataFrame(data, index=idx, columns=cols)


class TestGroupPBO(unittest.TestCase):

    def test_it_is_deterministic(self):
        frame = _family_frame()
        a = rb.group_pbo(frame=frame, binding='x')
        b = rb.group_pbo(frame=frame, binding='x')
        self.assertEqual(a.pbo, b.pbo)
        np.testing.assert_array_equal(a.logits, b.logits)

    def test_a_family_edge_drives_the_family_pbo_down(self):
        """The pooling claim itself: lift BOTH members of one family and the family-level
        choice becomes stable, because the noise explanation now has to win twice per
        split instead of once."""
        res = rb.group_pbo(frame=_family_frame(edge_family='AAA'), binding='x')
        self.assertLess(res.pbo, 0.15)
        self.assertEqual(res.winner_share[0][0], 'AAA')
        self.assertGreater(res.winner_share[0][1], 0.85)

    def test_the_median_not_the_best_member_scores_a_group(self):
        """One spectacular member and one dead one must NOT make a family of two look like
        the spectacular member: the median of two is their mean, dragged by the dead one.
        Cherry-picking the best member per split would smuggle back the variant-level
        selection this measurement removes."""
        frame = _family_frame(seed=72)
        frame['FFF_1'] = frame['AAA_1'] + 0.012           # one strong member...
        frame['FFF_2'] = frame['AAA_2'] - 0.012           # ...and one weak one
        res = rb.group_pbo(frame=frame, binding='x')
        # The mixed family must not dominate the in-sample wins.
        share = dict((g, p) for g, p in res.winner_share)
        self.assertLess(share.get('FFF', 0.0), 0.5)

    def test_it_refuses_too_few_groups(self):
        frame = _family_frame()[['AAA_1', 'AAA_2', 'BBB_1', 'BBB_2', 'CCC_1']]
        with self.assertRaisesRegex(NotCalculable, 'groups'):
            rb.group_pbo(frame=frame, binding='x')

    def test_member_counts_are_recorded(self):
        res = rb.group_pbo(frame=_family_frame(), binding='x')
        self.assertEqual(res.n_groups, 5)
        self.assertEqual(len(res.members['AAA']), 2)


class TestGroupRankBootstrap(unittest.TestCase):

    def test_p_first_sums_to_one_because_groups_partition_the_table(self):
        res = rb.group_rank_bootstrap(frame=_family_frame(), binding='x', n_paths=300)
        self.assertAlmostEqual(sum(r['p_first'] for r in res.rows), 1.0, places=9)

    def test_the_group_probability_dominates_every_member(self):
        """P(any member top-k) >= each member's own P(top-k), on the SAME draw — guaranteed
        by construction if the two tables really share one set of resampled histories."""
        frame = _family_frame(seed=73)
        strat = rb.rank_bootstrap(frame=frame, binding='x', n_paths=300)
        fam = rb.group_rank_bootstrap(frame=frame, binding='x', n_paths=300)
        member_p = {r['name']: r['p_top_k'] for r in strat.rows}
        for row in fam.rows:
            members = [n for n in frame.columns if rb.family_of(n) == row['group']]
            self.assertGreaterEqual(row['p_top_k'] + 1e-12,
                                    max(member_p[n] for n in members), row['group'])

    def test_a_lifted_family_claims_the_top_nearly_always(self):
        res = rb.group_rank_bootstrap(frame=_family_frame(edge_family='AAA', edge=0.015),
                                      binding='x', n_paths=300)
        top = next(r for r in res.rows if r['group'] == 'AAA')
        self.assertEqual(top['observed_best_rank'], 1)
        self.assertGreater(top['p_first'], 0.95)

    def test_the_lines_name_the_grouping_and_the_partition_identity(self):
        res = rb.group_rank_bootstrap(frame=_family_frame(), binding='x', n_paths=100,
                                      grouping='mechanism', groups=rb.mechanism_of)
        text = '\n'.join(rb.group_rank_lines(res))
        self.assertIn('MECHANISM RANK STABILITY', text)
        self.assertIn('P(1st) sums to 100%', text)


class TestTheFrontierClosedForm(unittest.TestCase):
    """`_frontier_from_paths` against arithmetic done by hand, then against the walker."""

    def test_unlevered_never_calls_and_matches_buy_and_hold(self):
        paths = np.array([[0.10, -0.20, 0.05]])
        p, cagr = rb._frontier_from_paths(paths, 1.0, 0.30, 0.06)
        self.assertEqual(p, 0.0)
        want = (1.10 * 0.80 * 1.05) ** (12.0 / 3.0) - 1.0
        self.assertAlmostEqual(float(cagr[0]), want, places=12)

    def test_the_critical_return_by_hand(self):
        """f=2, m=0.30, c=0: r_crit = 1/(2*0.7) - 1 = -28.57%. A -35% month calls; a -28%
        month does not."""
        called, _ = rb._frontier_from_paths(np.array([[0.0, -0.35, 0.0]]), 2.0, 0.30, 0.0)
        self.assertEqual(called, 1.0)
        safe, _ = rb._frontier_from_paths(np.array([[0.0, -0.28, 0.0]]), 2.0, 0.30, 0.0)
        self.assertEqual(safe, 0.0)

    def test_interest_tightens_the_threshold(self):
        """At 12%/yr the same -28% month DOES call: r_crit moves to 1.01/1.4 - 1 = -27.86%.
        The borrow rate is not a detail — it is part of the failure boundary."""
        p, _ = rb._frontier_from_paths(np.array([[0.0, -0.28, 0.0]]), 2.0, 0.30, 0.12)
        self.assertEqual(p, 1.0)

    def test_the_wealth_kept_at_the_forced_close_by_hand(self):
        """Call in month two of four at f=2, m=0.30, c=0: the holder keeps
        2*(1-0.35) - 1 = 30% of the month-open equity, then sits in cash."""
        paths = np.array([[0.0, -0.35, 0.50, 0.50]])
        p, cagr = rb._frontier_from_paths(paths, 2.0, 0.30, 0.0)
        self.assertEqual(p, 1.0)
        self.assertAlmostEqual(float(cagr[0]), 0.30 ** (12.0 / 4.0) - 1.0, places=12)

    def test_ruin_is_counted_not_censored(self):
        """A month bad enough to wipe the equity entirely reports -100%, not an exclusion."""
        p, cagr = rb._frontier_from_paths(np.array([[-0.60]]), 2.0, 0.30, 0.0)
        self.assertEqual(p, 1.0)
        self.assertEqual(float(cagr[0]), -1.0)

    def test_p_call_is_monotone_in_leverage(self):
        rng = np.random.default_rng(9)
        paths = rng.normal(0.005, 0.05, size=(500, 120))
        ps = [rb._frontier_from_paths(paths, f, 0.30, 0.06)[0]
              for f in (1.0, 1.5, 2.0, 2.5, 3.0)]
        self.assertEqual(ps, sorted(ps))

    def test_the_closed_form_matches_the_reference_walker(self):
        """The decisive one: `simulate_margin_path` walks the same path step by step with
        the same policy, and the two must agree on WHETHER a call fires, WHEN, and on the
        surviving equity — or the frontier is fast and wrong."""
        from common.margin_sizing import simulate_margin_path
        rng = np.random.default_rng(10)
        for f in (1.3, 1.8, 2.4):
            for m, c in ((0.30, 0.0), (0.30, 0.12), (0.60, 0.06)):
                paths = rng.normal(0.004, 0.06, size=(40, 60))
                p_call, cagr = rb._frontier_from_paths(paths, f, m, c)
                for row in range(paths.shape[0]):
                    out = simulate_margin_path(paths[row], f, m, policy='reset_monthly',
                                               borrow_rate_annual=c)
                    # The per-MONTH borrow rate, exactly as both implementations charge it.
                    r_crit = (f - 1.0) * (1.0 + c / 12.0) / (f * (1.0 - m)) - 1.0
                    called = bool((paths[row] < r_crit).any())
                    self.assertEqual(out.liquidated, called, (f, m, c, row))
                    if called:
                        self.assertEqual(out.step, int(np.argmax(paths[row] < r_crit)))
                    else:
                        # the walker's terminal equity is the product of the same factors
                        want = float(np.prod(1.0 + f * paths[row] - (f - 1.0) * (c / 12.0)))
                        self.assertAlmostEqual(out.equity[-1], want, places=9)


class TestTheFrontierPublic(unittest.TestCase):

    def _run(self, frame, **over):
        maintenance = {n: 0.30 for n in frame.columns}
        args = dict(frame=frame, binding='x', maintenance=maintenance,
                    borrow_rate_annual=0.06, n_paths=200)
        args.update(over)
        return rb.leverage_frontier(**args)

    def test_it_walks_the_same_histories_as_the_ranking(self):
        """Same seed, same draw: the frontier's unlevered 5th percentile must equal the
        rank bootstrap's CAGR p5 to the last bit. If this ever breaks, the two panels are
        describing different worlds while claiming one."""
        frame = _frame(6, 180, seed=81)
        res = self._run(frame)
        ranked = rb.rank_bootstrap(frame=frame, binding='x', rank_key='cagr', n_paths=200)
        p5 = {r['name']: r['p5'] for r in ranked.rows}
        for row in res.rows:
            self.assertEqual(row.curve[1.0][2], p5[row.name], row.name)

    def test_it_is_deterministic(self):
        frame = _frame(6, 180, seed=82)
        a, b = self._run(frame), self._run(frame)
        for ra, rb_ in zip(a.rows, b.rows):
            self.assertEqual(ra.curve, rb_.curve)

    def test_entries_without_a_maintenance_requirement_are_skipped_by_name(self):
        frame = _frame(6, 180, seed=83)
        maintenance = {n: 0.30 for n in list(frame.columns)[:-1]}
        res = self._run(frame, maintenance=maintenance)
        self.assertEqual(len(res.rows), 5)
        self.assertEqual(res.skipped[0][0], list(frame.columns)[-1])

    def test_the_loose_threshold_is_never_tighter_than_the_strict_one(self):
        for row in self._run(_frame(8, 200, seed=84)).rows:
            self.assertGreaterEqual(row.f_loose, row.f_strict)
            self.assertGreaterEqual(row.f_strict, 1.0)

    def test_a_tame_series_pushes_the_kelly_peak_off_the_grid_and_says_so(self):
        """Strong drift, low volatility: the growth optimum sits beyond 3x, and the honest
        answer is '>3.00', not the grid's edge wearing a number."""
        rng = np.random.default_rng(85)
        idx = pd.date_range('2010-01-31', periods=180, freq='ME')
        tame = pd.DataFrame({f'S{i}': rng.normal(0.010, 0.015, 180) for i in range(5)},
                            index=idx)
        res = self._run(tame)
        self.assertTrue(all(r.kelly_censored for r in res.rows))
        self.assertIn('>3.00', '\n'.join(rb.frontier_lines(res)))

    def test_a_volatile_series_has_an_interior_kelly_peak(self):
        """μ≈0.5%/mo on σ≈6%/mo puts the geometric optimum well inside the grid, and the
        median curve must actually BEND — the whole reason to draw it."""
        rng = np.random.default_rng(86)
        idx = pd.date_range('2010-01-31', periods=240, freq='ME')
        wild = pd.DataFrame({f'S{i}': rng.normal(0.005, 0.06, 240) for i in range(5)},
                            index=idx)
        res = self._run(wild, maintenance={n: 0.10 for n in wild.columns})
        for row in res.rows:
            self.assertFalse(row.kelly_censored, row.name)
            self.assertGreater(row.cagr_at_kelly, row.curve[3.0][1], row.name)

    def test_the_recommended_leverage_is_evaluated_on_the_paths(self):
        frame = _frame(6, 180, seed=87)
        rec = {n: 1.4 for n in frame.columns}
        res = self._run(frame, recommended=rec)
        for row in res.rows:
            self.assertTrue(np.isfinite(row.p_call_at_recommended))
            # ...and it must agree with the curve at the nearest grid point.
            self.assertAlmostEqual(row.p_call_at_recommended, row.curve[1.4][0], places=12)

    def test_an_unlevered_recommendation_has_zero_call_probability(self):
        frame = _frame(6, 180, seed=88)
        res = self._run(frame, recommended={n: 1.0 for n in frame.columns})
        for row in res.rows:
            self.assertEqual(row.p_call_at_recommended, 0.0)

    def test_a_grid_that_does_not_anchor_at_1x_is_refused(self):
        with self.assertRaisesRegex(NotCalculable, '1.0x'):
            self._run(_frame(6, 180, seed=89), f_grid=(1.5, 2.0))

    def test_the_lines_state_the_month_end_limitation(self):
        text = '\n'.join(rb.frontier_lines(self._run(_frame(6, 180, seed=90))))
        self.assertIn('MONTH-END', text)
        self.assertIn('understates', text)
        self.assertIn('cross-check', text)


if __name__ == '__main__':
    unittest.main()
