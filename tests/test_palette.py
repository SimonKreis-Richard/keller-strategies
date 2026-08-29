"""Tests for the chart encoding and the picker's roster.

These guard two things a screenshot cannot: that every family the registry can produce has
a colour assigned (a missing key used to fall through to grey, silently merging a family
into the neutrals), and that the picker's notion of "selectable" is derived from the
strategy classes rather than from a list someone has to remember to update.
"""

import os
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import palette  # noqa: E402


class TestFamilyOf(unittest.TestCase):
    def test_role_beats_the_name(self):
        # SPY_Benchmark must not create a family called "SPY".
        self.assertEqual(palette.family_of('SPY_Benchmark', 'benchmark'), 'Benchmark')
        self.assertEqual(palette.family_of('Golden_Butterfly', 'benchmark'), 'Benchmark')
        self.assertEqual(palette.family_of('HAA_G1_Simple', 'control'), 'Control')

    def test_trailing_digits_are_variant_not_family(self):
        # PAA2 is Keller & Butler's protection factor a=2, not a separate family. Splitting
        # on '_' alone stranded PAA2_G12 in a one-member family with no colour.
        self.assertEqual(palette.family_of('PAA2_G12'), 'PAA')
        self.assertEqual(palette.family_of('PAA_G3_Leveraged_2X'), 'PAA')

    def test_ordinary_names(self):
        for name, want in [('HAA_G12', 'HAA'), ('BAA_G3_Leveraged_2X', 'BAA'),
                           ('DM_G8_Composite', 'DM'), ('GTAA_G5', 'GTAA')]:
            self.assertEqual(palette.family_of(name), want, name)


class TestRegistryIsFullyColoured(unittest.TestCase):
    """Every family the LIVE registry produces must have an explicit colour."""

    def setUp(self):
        import main as engine
        self.roster = engine.strategy_roster()

    def test_every_family_has_a_colour(self):
        fams = {d['family'] for d in self.roster}
        missing = sorted(f for f in fams if f not in palette.FAMILY_COLORS)
        self.assertEqual(missing, [], f'families with no colour: {missing}')

    def test_every_family_is_in_the_display_order(self):
        fams = {d['family'] for d in self.roster}
        self.assertEqual(sorted(f for f in fams if f not in palette.FAMILY_ORDER), [])

    def test_colours_are_distinct(self):
        used = [palette.FAMILY_COLORS[f] for f in {d['family'] for d in self.roster}]
        self.assertEqual(len(used), len(set(used)), 'two families share a hue')

    def test_roles_are_read_from_the_classes(self):
        by_name = {d['name']: d for d in self.roster}
        # The two degenerate single-asset diagnostics, and nothing else.
        self.assertEqual(sorted(d['name'] for d in self.roster if d['role'] == 'control'),
                         ['BAA_G1_SPY', 'HAA_G1_Simple'])
        # Passive references. The three levered ones were added 2026-07-29 to answer "how much
        # of a wrap's record is the timing rule and how much is just leverage?", which could not
        # be asked while every reference here was 1x.
        self.assertEqual(sorted(d['name'] for d in self.roster if d['role'] == 'benchmark'),
                         ['Golden_Butterfly', 'RiskParity_3X', 'SPY_2X_Benchmark',
                          'SPY_3X_Benchmark', 'SPY_Benchmark', 'Sixty_Forty_1X'])
        # A leveraged wrap must never inherit its parent's fidelity claim.
        self.assertEqual(by_name['BAA_G3_Leveraged_2X']['fidelity'], 'custom')
        self.assertEqual(by_name['BAA_G12']['fidelity'], 'faithful')


class TestLineStyles(unittest.TestCase):
    def test_same_family_shares_a_hue_and_differs_in_stroke(self):
        entries = [('BAA_G12', 'strategy'), ('BAA_G4', 'strategy'),
                   ('BAA_G3_Leveraged_2X', 'strategy'), ('HAA_G12', 'strategy')]
        st = palette.line_styles(entries)
        baa = ['BAA_G12', 'BAA_G4', 'BAA_G3_Leveraged_2X']
        self.assertEqual(len({st[n]['color'] for n in baa}), 1)
        self.assertNotEqual(st['BAA_G12']['color'], st['HAA_G12']['color'])
        # Distinguishable without colour: dash pattern and marker both vary.
        self.assertEqual(len({st[n]['dashes'] for n in baa}), 3)
        self.assertEqual(len({st[n]['marker'] for n in baa}), 3)

    def test_stroke_is_stable_under_deselection(self):
        """A strategy must keep its dash pattern when its siblings are unticked.

        Otherwise every tick of a checkbox reshuffles the encoding and no two runs of the
        dashboard can be read against each other. This is what `universe` buys.
        """
        universe = [('BAA_G12', 'strategy'), ('BAA_G4', 'strategy'),
                    ('BAA_G4_Leveraged_2X', 'strategy')]
        full = palette.line_styles(universe, universe=universe)
        # Untick BAA_G12, the one that was sorting first.
        subset = palette.line_styles(universe[1:], universe=universe)
        for name in ('BAA_G4', 'BAA_G4_Leveraged_2X'):
            self.assertEqual(full[name]['dashes'], subset[name]['dashes'], name)
            self.assertEqual(full[name]['marker'], subset[name]['marker'], name)

    def test_without_a_universe_the_index_is_local(self):
        """The documented fallback: no `universe` means the index comes from `entries`."""
        universe = [('BAA_G12', 'strategy'), ('BAA_G4', 'strategy')]
        self.assertNotEqual(palette.line_styles(universe)['BAA_G4']['dashes'],
                            palette.line_styles(universe[1:])['BAA_G4']['dashes'])

    def test_plot_order_never_matters(self):
        a = palette.line_styles([('HAA_G12', 'strategy'), ('HAA_G8_Balanced', 'strategy')])
        b = palette.line_styles([('HAA_G8_Balanced', 'strategy'), ('HAA_G12', 'strategy')])
        self.assertEqual(a, b)

    def test_plot_kwargs_strips_private_keys(self):
        st = palette.line_styles([('HAA_G12', 'strategy')])['HAA_G12']
        self.assertIn('_family', st)
        kw = palette.plot_kwargs(st)
        self.assertFalse([k for k in kw if k.startswith('_')])
        # And what remains must be acceptable to matplotlib.
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 2], **kw)
        plt.close(fig)

    def test_benchmarks_are_neutral_and_behind(self):
        """Role now drives colour, alpha and zorder — NOT width.

        Width was reassigned to the leverage RATIO on 2026-07-29, because HAA reached seven
        entries and `_DASHES` has five patterns, and because the ratio is the thing most worth
        seeing on a chart of levered variants and was not encoded at all. A benchmark is already
        identified by its neutral grey and its name at the right edge.
        """
        st = palette.line_styles([('SPY_Benchmark', 'benchmark'), ('HAA_G12', 'strategy')])
        self.assertEqual(st['SPY_Benchmark']['color'], palette.FAMILY_COLORS['Benchmark'])
        self.assertLess(st['SPY_Benchmark']['zorder'], st['HAA_G12']['zorder'])
        self.assertLess(st['SPY_Benchmark']['alpha'], st['HAA_G12']['alpha'])
        # Same ratio (both 1x) => same width. Role must no longer move it.
        self.assertEqual(st['SPY_Benchmark']['linewidth'], st['HAA_G12']['linewidth'])

    def test_line_width_encodes_the_leverage_ratio(self):
        """The requirement: two entries differing ONLY in ratio stay distinguishable.

        `HAA_G3_Leveraged_2X` and `HAA_G3_Leveraged_3X` share a family (same colour) and a
        universe, so they would share a dash pattern too if the index happened to collide.
        Width is what separates them, and it must be monotonic in the ratio.
        """
        entries = [('HAA_G12', 'strategy'),
                   ('HAA_G3_Leveraged_2X', 'strategy'),
                   ('HAA_G3_Leveraged_3X', 'exploratory')]
        ratios = {'HAA_G12': 1.0, 'HAA_G3_Leveraged_2X': 2.0, 'HAA_G3_Leveraged_3X': 3.0}
        st = palette.line_styles(entries, ratios=ratios)
        w = [st[n]['linewidth'] for n, _ in entries]
        self.assertEqual(w, sorted(w), 'width must increase with the ratio')
        self.assertEqual(len(set(w)), 3, 'each ratio needs its own width')

    def test_ratio_falls_back_to_the_key_suffix_but_prefers_the_attribute(self):
        """`ratio_of` reads the declared leverage first. The suffix is a fallback for callers
        holding only a key — and a poor one: name-parsing is how `family_of('PAA2_G12')` once
        returned "PAA2", reading Keller's protection factor as if it were a ratio."""
        self.assertEqual(palette.ratio_of('HAA_G3_Leveraged_3X'), 3.0)
        self.assertEqual(palette.ratio_of('HAA_G3_Leveraged_2X'), 2.0)
        self.assertEqual(palette.ratio_of('HAA_G12'), 1.0)
        self.assertEqual(palette.ratio_of('PAA2_G12'), 1.0, 'the 2 is a protection factor')
        # The attribute wins over the suffix when they disagree.
        self.assertEqual(palette.ratio_of('SPY_3X_Benchmark', leverage=1.0), 1.0)


class TestLabelLines(unittest.TestCase):
    """The right-edge labels are the channel that removes legend cross-referencing, so the
    two properties that make them readable are worth pinning: every series gets one, and
    their vertical ORDER matches the lines' vertical order."""

    def _axes(self, finals):
        fig, ax = plt.subplots()
        idx = pd.date_range('2020-01-31', periods=24, freq='ME')
        endpoints = {}
        for name, final in finals.items():
            curve = np.linspace(1.0, final, len(idx))
            ax.plot(idx, curve, label=name)
            endpoints[name] = final
        ax.set_yscale('log')
        return fig, ax, endpoints

    def test_every_series_is_labelled(self):
        finals = {f's{i}': 1.5 + 0.1 * i for i in range(12)}
        fig, ax, ep = self._axes(finals)
        st = palette.line_styles([(n, 'strategy') for n in finals])
        palette.label_lines(ax, ep, st)
        texts = [t.get_text() for t in ax.texts]
        self.assertEqual(sorted(texts), sorted(finals))
        plt.close(fig)

    def test_order_is_preserved_and_labels_never_collide(self):
        # Deliberately near-identical endpoints: this is the case the old chart failed on.
        finals = {'A': 3.000, 'B': 3.005, 'C': 3.010, 'D': 3.015, 'E': 8.0, 'F': 1.2}
        fig, ax, ep = self._axes(finals)
        st = palette.line_styles([(n, 'strategy') for n in finals])
        palette.label_lines(ax, ep, st, min_gap=0.05)
        placed = sorted(((t.get_position()[1], t.get_text()) for t in ax.texts),
                        reverse=True)
        self.assertEqual([n for _, n in placed],
                         [n for n, _ in sorted(finals.items(), key=lambda kv: -kv[1])])
        ys = [y for y, _ in placed]
        for hi, lo in zip(ys, ys[1:]):
            self.assertGreaterEqual(hi - lo, 0.05 - 1e-9)
        plt.close(fig)

    def test_stack_stays_inside_the_axes(self):
        finals = {f's{i}': 5.0 + 0.001 * i for i in range(15)}
        fig, ax, ep = self._axes(finals)
        st = palette.line_styles([(n, 'strategy') for n in finals])
        palette.label_lines(ax, ep, st, min_gap=0.04)
        self.assertLessEqual(max(t.get_position()[1] for t in ax.texts), 1.0 + 1e-9)
        plt.close(fig)

    def test_non_positive_endpoint_on_a_log_axis_is_skipped_not_crashed(self):
        finals = {'ok': 2.0, 'wiped_out': 1.0}
        fig, ax, ep = self._axes(finals)
        ep['wiped_out'] = 0.0
        st = palette.line_styles([(n, 'strategy') for n in finals])
        palette.label_lines(ax, ep, st)
        self.assertEqual([t.get_text() for t in ax.texts], ['ok'])
        plt.close(fig)

    def test_empty_input_is_a_no_op(self):
        fig, ax = plt.subplots()
        palette.label_lines(ax, {}, {})
        self.assertEqual(len(ax.texts), 0)
        plt.close(fig)


class TestFamilyLegend(unittest.TestCase):
    def test_one_handle_per_family_not_per_strategy(self):
        entries = [('BAA_G12', 'strategy'), ('BAA_G4', 'strategy'),
                   ('HAA_G12', 'strategy'), ('SPY_Benchmark', 'benchmark')]
        handles = palette.family_legend_handles(entries)
        self.assertEqual([h.get_label() for h in handles], ['HAA', 'BAA', 'Benchmark'])


if __name__ == '__main__':
    unittest.main()
