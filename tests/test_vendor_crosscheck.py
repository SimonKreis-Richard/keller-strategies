"""
The second-vendor cross-check, driven entirely offline.

`tools/vendor_crosscheck.py` is the only code in this project that talks to a vendor other
than Yahoo, and it lives in `tools/` precisely so `unittest discover -s tests` can never
collect it and open a socket. That leaves a gap this file fills: the CLASSIFICATION is the
part that can be wrong in a way nobody notices, so it is exercised here through an injected
`fetch`, against CSV text written by hand in the test rather than produced by the code.

The first case asserted is the fail-open one, because it is the one that matters. Stooq
answers a rate limit with HTTP 200 and a plain-text body; a tool that let `pd.read_csv`
swallow that would report "0 disagreements" forever, and would be worse than no tool at all
— it would retire the suspicion that motivated writing it.
"""
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from tools import vendor_crosscheck as vc


_NO_NETWORK = None


def setUpModule():
    """Make a socket in THIS file an error, not a slow test.

    Written after a test in this very file reached stooq.com because it patched the module
    attribute while `crosscheck` still held the original in a default argument. A guarantee
    that depends on remembering how Python binds defaults is not a guarantee.
    """
    global _NO_NETWORK

    def forbidden(*args, **kwargs):
        raise AssertionError('a test in test_vendor_crosscheck.py opened a socket')

    _NO_NETWORK = unittest.mock.patch.object(vc.urllib.request, 'urlopen', forbidden)
    _NO_NETWORK.start()


def tearDownModule():
    if _NO_NETWORK is not None:
        _NO_NETWORK.stop()


def _csv(index, closes):
    rows = ['Date,Open,High,Low,Close,Volume']
    for d, c in zip(index, closes):
        rows.append('{:%Y-%m-%d},{c},{c},{c},{c},1000'.format(d, c=round(float(c), 4)))
    return '\n'.join(rows) + '\n'


class _Store:
    """Only what `crosscheck` reads."""

    def __init__(self, frame):
        self._frames = {'close': frame}


def _series(n=800, start=100.0, drift=0.0004, seed=5):
    idx = pd.bdate_range('2023-01-02', periods=n)
    rng = np.random.default_rng(seed)
    steps = 1.0 + drift + rng.normal(0.0, 0.004, n)
    return pd.Series(start * np.cumprod(steps), index=idx)


class TestTheParserRefusesAnythingThatIsNotTheCSV(unittest.TestCase):

    def test_a_rate_limit_body_is_unavailable_not_agreement(self):
        """HTTP 200 with prose in it. The whole point of the strict header assertion."""
        ours = _series()
        store = _Store(pd.DataFrame({'SPY': ours}))
        res = vc.crosscheck(store, fetch=lambda s: 'Exceeded the daily hits limit',
                            tickers=['SPY'])
        self.assertEqual(res['SPY']['status'], 'unavailable')
        self.assertNotEqual(res['SPY']['status'], 'agrees')

    def test_an_empty_body_is_unavailable(self):
        store = _Store(pd.DataFrame({'SPY': _series()}))
        res = vc.crosscheck(store, fetch=lambda s: '', tickers=['SPY'])
        self.assertEqual(res['SPY']['status'], 'unavailable')

    def test_a_network_failure_is_a_status_not_a_crash(self):
        def boom(symbol):
            raise OSError('connection reset')

        store = _Store(pd.DataFrame({'SPY': _series()}))
        res = vc.crosscheck(store, fetch=boom, tickers=['SPY'])
        self.assertEqual(res['SPY']['status'], 'unavailable')
        self.assertIn('connection reset', res['SPY']['reason'])

    def test_a_well_formed_csv_parses(self):
        s = _series(n=40)
        got = vc.parse_stooq_csv(_csv(s.index, s.values))
        self.assertEqual(len(got), 40)


class TestTheClassification(unittest.TestCase):

    def test_identical_returns_at_a_different_price_level_agree(self):
        """Vendors can disagree about the LEVEL and carry identical returns — a constant
        offset holds no information about anything this project measures, so the median
        normalisation must absorb it."""
        ours = _series()
        theirs = ours * 1.5
        store = _Store(pd.DataFrame({'SPY': ours}))
        res = vc.crosscheck(store, fetch=lambda s: _csv(theirs.index, theirs.values),
                            tickers=['SPY'])
        self.assertEqual(res['SPY']['status'], 'agrees', res['SPY'])

    def test_an_unapplied_split_is_named_as_a_split_not_a_price_bug(self):
        """A split eight years back corrupts every long-run metric and never touches the
        last two years, so the monthly test alone cannot see it."""
        ours = _series()
        theirs = ours.copy()
        theirs.iloc[:300] = theirs.iloc[:300] * 2.0     # a 2:1 the other vendor never applied
        store = _Store(pd.DataFrame({'SPY': ours}))
        res = vc.crosscheck(store, fetch=lambda s: _csv(theirs.index, theirs.values),
                            tickers=['SPY'])
        self.assertEqual(res['SPY']['status'], 'suspected_split', res['SPY'])
        self.assertGreater(res['SPY']['level_ratio_spread'], vc.SPLIT_RATIO_SPREAD)

    def test_one_diverging_month_is_caught_and_named(self):
        ours = _series()
        theirs = ours.copy()
        target = theirs.index[(theirs.index.year == 2024) & (theirs.index.month == 5)]
        theirs.loc[target[-1]] = float(theirs.loc[target[-1]]) * 1.01
        store = _Store(pd.DataFrame({'SPY': ours}))
        res = vc.crosscheck(store, fetch=lambda s: _csv(theirs.index, theirs.values),
                            tickers=['SPY'])
        self.assertEqual(res['SPY']['status'], 'disagrees', res['SPY'])
        # Moving ONE month-end price moves TWO monthly returns -- the one that ends there
        # and the one that starts there -- so the guard may legitimately name either. What
        # it may not do is name a month nowhere near the perturbation.
        self.assertIn(res['SPY']['worst_month'], ('2024-05', '2024-06'))
        self.assertAlmostEqual(res['SPY']['worst_diff'], 0.01, places=3)
        self.assertGreater(res['SPY']['worst_diff'], vc.MONTHLY_TOLERANCE)

    def test_too_little_overlap_is_never_agreement(self):
        ours = _series(n=800)
        theirs = ours.iloc[-40:]                        # about two months
        store = _Store(pd.DataFrame({'SPY': ours}))
        res = vc.crosscheck(store, fetch=lambda s: _csv(theirs.index, theirs.values),
                            tickers=['SPY'])
        self.assertEqual(res['SPY']['status'], 'insufficient_overlap')

    def test_a_ticker_the_store_does_not_hold_is_unavailable(self):
        store = _Store(pd.DataFrame({'SPY': _series()}))
        res = vc.crosscheck(store, fetch=lambda s: '', tickers=['NOPE'])
        self.assertEqual(res['NOPE']['status'], 'unavailable')


class TestTheReportCannotBeMisread(unittest.TestCase):

    def test_the_summary_states_how_many_were_actually_compared(self):
        """"0 disagreements" over 37 unreachable tickers is not reassurance, and the report
        must make that impossible to read the wrong way."""
        results = {'A': {'status': 'agrees', 'n_overlap_months': 24, 'worst_diff': 0.0001,
                         'level_ratio_spread': 1.0, 'reason': None},
                   'B': {'status': 'unavailable', 'n_overlap_months': 0, 'worst_diff': None,
                         'level_ratio_spread': None, 'reason': 'timeout'}}
        text = '\n'.join(vc.report_lines(results))
        self.assertIn('1 of 2 tickers were actually COMPARED', text)
        self.assertIn('unavailable', text)

    def test_an_alarm_exits_nonzero(self):
        """A scheduler must be able to act without parsing the table."""
        import types
        ours = _series()
        theirs = ours.copy()
        theirs.iloc[-25:] = theirs.iloc[-25:] * 1.05
        store = _Store(pd.DataFrame({'SPY': ours}))

        fake_engine = types.SimpleNamespace(load_data=lambda cfg: (None, None, None, store))
        store.provenance = lambda: {'sha256': 'deadbeef'}
        # `main` resolves the source through PROVIDERS, so that is what a test must
        # redirect. Patching `fetch_stooq_csv` alone leaves the table pointing at the real
        # one -- the module-wide offline lock caught exactly that while this was written.
        with unittest.mock.patch.dict(sys.modules, {'main': fake_engine}), \
                unittest.mock.patch.dict(
                    vc.PROVIDERS, {'stooq': lambda s: _csv(theirs.index, theirs.values)}), \
                unittest.mock.patch('tools.backtest_driver.build_config',
                                    lambda **kw: {}):
            code = vc.main(['--tickers', 'SPY', '--no-save'])
        self.assertEqual(code, 1)

    def test_a_vendor_that_refuses_automated_access_is_not_an_alarm(self):
        """Exit 0, deliberately. "I could not ask" is not "they disagreed", and a
        scheduler that treats the two alike teaches its reader to ignore it.

        Probed 2026-09-01: Stooq now answers with an HTML proof-of-work page. This asserts
        the shape of that outcome rather than the vendor's current mood."""
        import types
        store = _Store(pd.DataFrame({'SPY': _series()}))
        store.provenance = lambda: {'sha256': 'deadbeef'}
        fake_engine = types.SimpleNamespace(load_data=lambda cfg: (None, None, None, store))

        def gated(symbol):
            return '<!DOCTYPE html><html><head></head><body>challenge</body></html>'

        with unittest.mock.patch.dict(sys.modules, {'main': fake_engine}), \
                unittest.mock.patch.dict(vc.PROVIDERS, {'stooq': gated}), \
                unittest.mock.patch('tools.backtest_driver.build_config',
                                    lambda **kw: {}):
            code = vc.main(['--tickers', 'SPY', '--no-save'])
        self.assertEqual(code, 0)


class TestTheSuiteStaysOffline(unittest.TestCase):
    """The property that makes keeping this tool outside `tests/` worth the awkwardness."""

    def test_nothing_here_reaches_the_network(self):
        ours = _series()
        store = _Store(pd.DataFrame({'SPY': ours}))
        res = vc.crosscheck(store, fetch=lambda s: _csv(ours.index, ours.values),
                            tickers=['SPY'])
        self.assertEqual(res['SPY']['status'], 'agrees')

    def test_the_default_fetch_is_resolved_at_call_time(self):
        """So patching the module attribute actually redirects it. Without this, a test
        that patches `vc.fetch_stooq_csv` silently keeps using the real one."""
        ours = _series()
        store = _Store(pd.DataFrame({'SPY': ours}))
        with unittest.mock.patch.object(vc, 'fetch_stooq_csv',
                                        lambda s: _csv(ours.index, ours.values)):
            res = vc.crosscheck(store, tickers=['SPY'])
        self.assertEqual(res['SPY']['status'], 'agrees')

    def test_an_injected_fetch_is_the_one_that_runs(self):
        """If a refactor lets the real fetch back into the injected path, the module-wide
        lock turns this red rather than merely slow."""
        ours = _series()
        store = _Store(pd.DataFrame({'SPY': ours}))
        calls = []

        def spy(symbol):
            calls.append(symbol)
            return _csv(ours.index, ours.values)

        vc.crosscheck(store, fetch=spy, tickers=['SPY'])
        self.assertEqual(calls, ['SPY'])


if __name__ == '__main__':
    unittest.main()
