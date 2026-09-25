"""
The live refusals, now that a second code path produces orders.

`tests/test_live_pricing.py` already covers what these guards DO — an unverified store, a
spliced vintage inside and outside a strategy's universe, a levered defensive sleeve, a row
that does not sum to one. Those tests run through `compute_live_signals` and they still
pass, which is the evidence that extracting the block on 2026-09-17 changed no behaviour.

What is new, and what this file exists for, is a STRUCTURAL property that could not be
stated while the guards lived inside their only caller:

    the refusals are in ONE place, and `compute_live_signals` delegates to it rather than
    carrying its own copy.

That matters because `common/ca_mapping.py` and its driver size the same allocation row into
real Canadian orders. Two copies of a guard drift, and the direction they drift in is always
the same: the second copy is written from the first, and then the first is fixed. This
repository has already paid for that once — the backtest's guards took until 2026-07-29 to
reach the live path, and the comment above them still says so.

The delegation test below is the one that would catch a re-inlined copy. A test that only
asserted the refusals still fire would pass perfectly against two divergent copies, because
one of them would still be right.
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

import main
from common.live_guards import live_guards, refusal_message

SIGNAL_DATE = pd.Timestamp('2026-06-30')


def _strategy(offensive=('SPY',), defensive=(), canary=()):
    """A real BaseStrategy, not a SimpleNamespace: `sleeves()` is shape-checked since
    2026-07-28 and a stub that bypassed the checker would test a path production cannot
    take."""
    class _Stub(main.BaseStrategy):
        def __init__(self):
            super().__init__('TEST', score_type='unweighted')

        def sleeves(self):
            return {'offensive': set(offensive), 'defensive': set(defensive),
                    'canary': list(canary)}

        def generate_allocations(self, prices, scores, r12, r3):
            return pd.DataFrame({'SPY': [1.0]}, index=[SIGNAL_DATE])

    return _Stub()


def _panel():
    prices = pd.DataFrame({'SPY': [300.0], 'TIP': [100.0], 'GLD': [200.0]},
                          index=[SIGNAL_DATE])
    alloc = pd.DataFrame({'SPY': [1.0], 'TIP': [0.0], 'GLD': [0.0]}, index=[SIGNAL_DATE])
    return prices, alloc


def _store(status, tickers=(), constructed=None):
    """Only what the guards read. `constructed` maps ticker -> real_from Timestamp."""
    constructed = constructed or {}
    verification = None if status is None else {
        'status': status,
        'violations': [{'ticker': t, 'date': '2026-06-03', 'step': -0.0073,
                        'implied_distribution_pct': 0.73} for t in tickers]}
    return SimpleNamespace(verification=verification,
                           constructed_before=lambda t: constructed.get(t))


class TestTheRefusalsLiveInOnePlace(unittest.TestCase):
    """The property that could not be stated before the extraction."""

    def test_compute_live_signals_delegates_rather_than_carrying_a_copy(self):
        """Replace the guard function with one that always objects. If `compute_live_signals`
        still sizes orders, it is running its own copy and the extraction is cosmetic."""
        prices, _ = _panel()
        strat = _strategy()
        config = {'CURRENT_EXECUTION_DATE': '2026-07-01', 'STRATEGIES_TO_DISPLAY': [],
                  'LEVERAGE_FACTOR': 1.0, 'SAFETY_MARGIN_PCT': 0.0,
                  'MINIMUM_TRADE_PCT': 0.5, 'FLEXIBILITY_BAND_PCT': 0.0,
                  'FLUSH_ROUND_UP_BAND_PCT': 0.0, 'FRACTIONAL_SHARES': False,
                  'SHARE_LOT_SIZE': 1}
        accounts = [{'account_name': 'A', 'account_balance': 10000.0,
                     'account_priority': 1}]
        s_w = pd.DataFrame({'SPY': [1.0], 'TIP': [1.0], 'GLD': [1.0]}, index=[SIGNAL_DATE])

        with patch.object(main, 'live_guards', return_value=['  SENTINEL OBJECTION']):
            _, results = main.compute_live_signals(prices, s_w, s_w, [strat], config,
                                                   accounts, store=_store('ok'))
        self.assertIn('SENTINEL OBJECTION', results[0]['error'])
        self.assertNotIn('sizing', results[0])

    def test_the_refusal_sentence_comes_from_the_shared_helper(self):
        """Both order-producing paths must describe the same refusal identically; a user who
        reads two wordings believes there are two problems."""
        self.assertEqual(refusal_message(['  a', '  b']),
                         'refusing to size this allocation:\n  a\n  b')


class TestTheGuardsThemselves(unittest.TestCase):
    """Called directly, which is how the Canadian execution path will call them."""

    def test_a_verified_store_raises_no_objection(self):
        prices, alloc = _panel()
        self.assertEqual(live_guards(_strategy(), alloc, prices, SIGNAL_DATE, _store('ok')),
                         [])

    def test_a_store_that_verified_nothing_is_refused(self):
        prices, alloc = _panel()
        for status in ('not_applicable', 'skipped'):
            problems = live_guards(_strategy(), alloc, prices, SIGNAL_DATE, _store(status))
            self.assertTrue(any('never verified' in p for p in problems), status)

    def test_a_store_with_no_verification_attribute_is_refused(self):
        """Fail-closed on the shape of the object, not only on its value."""
        prices, alloc = _panel()
        problems = live_guards(_strategy(), alloc, prices, SIGNAL_DATE, _store(None))
        self.assertTrue(any('never verified' in p for p in problems))

    def test_a_seam_on_a_held_ticker_is_refused(self):
        prices, alloc = _panel()
        problems = live_guards(_strategy(), alloc, prices, SIGNAL_DATE,
                               _store('disagrees', ['SPY']))
        self.assertTrue(any('SPY' in p and 'vintage' in p for p in problems))

    def test_a_seam_on_the_canary_alone_is_refused(self):
        """The blast radius covers the canary, which never appears in the allocation row.
        A dead canary sends the whole book to cash; a corrupted one does it for no reason.
        This is the 2026-09-01 incident itself."""
        prices, alloc = _panel()
        strat = _strategy(offensive=('SPY',), canary=('TIP',))
        problems = live_guards(strat, alloc, prices, SIGNAL_DATE, _store('disagrees', ['TIP']))
        self.assertTrue(any('TIP' in p for p in problems))

    def test_a_seam_on_an_unheld_but_ranked_candidate_is_refused(self):
        """GLD is in the offensive universe at weight zero this month. Its score is what
        decided it was not selected, so a corrupted one changes the selection."""
        prices, alloc = _panel()
        strat = _strategy(offensive=('SPY', 'GLD'))
        problems = live_guards(strat, alloc, prices, SIGNAL_DATE, _store('disagrees', ['GLD']))
        self.assertTrue(any('GLD' in p for p in problems))

    def test_a_seam_outside_this_strategys_universe_leaves_it_alone(self):
        """Scoped per strategy on purpose: one fund's restatement must not halt the whole
        registry."""
        prices, alloc = _panel()
        problems = live_guards(_strategy(offensive=('SPY',)), alloc, prices, SIGNAL_DATE,
                               _store('disagrees', ['GLD']))
        self.assertEqual(problems, [])

    def test_a_constructed_span_under_a_held_position_is_refused(self):
        prices, alloc = _panel()
        store = _store('ok', constructed={'SPY': pd.Timestamp('2030-01-01')})
        problems = live_guards(_strategy(), alloc, prices, SIGNAL_DATE, store)
        self.assertTrue(any('CONSTRUCTED' in p for p in problems))

    def test_no_store_runs_the_weight_checks_and_skips_the_data_checks(self):
        """A caller with no store never claimed to have verified one, so the two data checks
        do not apply. The weight invariant still does — and must, because it is the check
        that has nothing to do with the store."""
        prices, alloc = _panel()
        self.assertEqual(live_guards(_strategy(), alloc, prices, SIGNAL_DATE, None), [])

        broken = alloc.copy()
        broken.loc[SIGNAL_DATE, 'SPY'] = 0.4
        self.assertTrue(live_guards(_strategy(), broken, prices, SIGNAL_DATE, None))


if __name__ == '__main__':
    unittest.main()
