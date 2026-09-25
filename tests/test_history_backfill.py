"""
The 2026-09-23 history extension — mutual-fund and index donors that carry the era to 2000.

`tests/test_guards.py::TestConstructedHistory` fences a single splice: no overlap with the
fund's own history, no step at the junction, every span in provenance. This file fences what
the extension added on top, each of which fails differently:

* CHAINS. VEA is spliced from EFA, and EFA from SWISX. Spliced in the wrong order, VEA would
  stop at EFA's own 2001 inception and nothing would say so.
* DONORS ARE NEVER TRADABLE. A mutual fund or an index that survived into the panel could be
  ranked, held or read as a signal. They are dropped once used, as ^IRX always was.
* THE CALENDAR BELONGS TO WHAT TRADES. A London gold fixing on a US holiday must not become a
  session on which a fill is priced.
* A MISSING DONOR IS RECORDED, NEVER SILENT. Its recipient then starts later, and the coverage
  line shows that; `missing_donors` is what says the donor, not the fund, is why.
* THE ONE NON-YAHOO SOURCE PARSES STRICTLY and is never reached by this suite.

Plus two structural claims no data is needed for: every strategy allowed to set the shared
window can reach the era, and the data start is derived from the era rather than chosen.

Everything is built in memory or in a temporary directory. `_http_get` is replaced by a
function that raises for the whole module, so a refactor that reached the network from here
would turn the suite red rather than slow.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from common import data_engine as de
from common.data_engine import HISTORY_BACKFILL, PriceStore


def _no_network(url):
    raise AssertionError(f'the test suite tried to reach {url}')


def setUpModule():
    global _lock
    _lock = patch.object(de, '_http_get', _no_network)
    _lock.start()


def tearDownModule():
    _lock.stop()


def _growth(idx, rate, scale=100.0):
    return pd.Series(scale * (1.0 + rate) ** np.arange(len(idx)), index=idx)


def _memory_store(frame, requested):
    """A store over `frame`, asked for `requested` only — everything else is a donor.

    Replays `PriceStore.__init__`'s ORDER on raw frames: extend first, stale policy after.
    `from_adjusted` alone would forward-fill before extending, which hides exactly the gaps
    the calendar and junction tests below are about.
    """
    store = PriceStore.from_adjusted(frame, frame.copy(), source='synthetic')
    store._frames = {f: frame.copy() for f in ('open', 'close', 'adj_close')}
    store.requested = list(requested)
    store.constructed = {}
    store.missing_donors = []
    store._extend_history()
    store._apply_stale_policy()
    return store


class TestTheRegistryIsWellFormed(unittest.TestCase):

    def test_every_declared_substitute_carries_its_reason_and_its_evidence(self):
        """A splice with no recorded measurement is an assertion, not a finding."""
        for recipient, b in HISTORY_BACKFILL.items():
            with self.subTest(recipient):
                self.assertTrue(b.donor and b.why and b.evidence)
                self.assertNotEqual(b.donor, recipient)

    def test_donors_are_resolved_transitively(self):
        self.assertEqual(de.donors_for(['VEA']), ['EFA', 'SWISX'])
        self.assertEqual(de.donors_for(['BND']), ['AGG', 'VBMFX'])
        self.assertEqual(de.donors_for(['SPY']), [])

    def test_a_donor_that_is_itself_a_recipient_is_spliced_first(self):
        order = de.splice_order()
        for inner, outer in (('EFA', 'VEA'), ('EEM', 'VWO'), ('AGG', 'BND')):
            with self.subTest(outer):
                self.assertLess(order.index(inner), order.index(outer))

    def test_a_cycle_is_refused_rather_than_looped(self):
        cyclic = {'AAA': de.Backfill('BBB', 'w', 'e'), 'BBB': de.Backfill('AAA', 'w', 'e')}
        with patch.object(de, 'HISTORY_BACKFILL', cyclic):
            with self.assertRaises(ValueError):
                de.splice_order()

    def test_no_recipient_is_dead_weight(self):
        """Every recipient is a ticker the store is actually asked for."""
        import main
        for recipient in HISTORY_BACKFILL:
            with self.subTest(recipient):
                self.assertIn(recipient, main.TICKERS)

    def test_no_donor_is_a_tradable_ticker_unless_something_trades_it(self):
        """Donor-only symbols must not be in main.TICKERS, or they would survive the drop."""
        import main
        tradable_donors = {'EFA', 'EEM', 'AGG'}     # traded, or kept as before 2026-09-23
        for b in HISTORY_BACKFILL.values():
            if b.donor not in tradable_donors:
                with self.subTest(b.donor):
                    self.assertNotIn(b.donor, main.TICKERS)


class TestChainsAreSplicedInnermostFirst(unittest.TestCase):
    """VEA <- EFA <- SWISX, with each fund on its own price scale."""

    @classmethod
    def setUpClass(cls):
        idx = pd.bdate_range('2000-01-03', periods=900)
        swisx = _growth(idx, 0.0004)
        efa = pd.Series(np.nan, index=idx)
        efa.iloc[300:] = swisx.iloc[300:].values * 0.5          # EFA opens at day 300
        vea = pd.Series(np.nan, index=idx)
        vea.iloc[600:] = efa.iloc[600:].values * 0.3            # VEA opens at day 600
        frame = pd.DataFrame({'VEA': vea, 'EFA': efa, 'SWISX': swisx,
                              'SPY': _growth(idx, 0.0003)})
        cls.idx = idx
        cls.store = _memory_store(frame, ['VEA', 'EFA', 'SPY'])
        cls.ac = cls.store.adj_close()

    def test_the_outer_recipient_reaches_the_innermost_donor(self):
        self.assertEqual(self.ac['VEA'].first_valid_index(), self.idx[0],
                         'VEA stopped at EFA\'s own inception: the chain was spliced '
                         'outermost-first')

    def test_the_record_names_the_whole_chain(self):
        self.assertEqual(self.store.constructed['VEA']['chain'], ['EFA', 'SWISX'])
        self.assertEqual(self.store.constructed['EFA']['chain'], ['SWISX'])

    def test_both_junctions_carry_the_donors_return_not_a_step(self):
        r_swisx = pd.Series(1.0004, index=self.idx) - 1.0
        for recipient, day in (('EFA', 300), ('VEA', 600)):
            with self.subTest(recipient):
                got = self.ac[recipient].pct_change().iloc[day]
                self.assertAlmostEqual(float(got), float(r_swisx.iloc[day]), places=10)

    def test_the_real_history_after_each_inception_is_untouched(self):
        after = self.ac.iloc[600:]
        self.assertTrue(np.allclose(after['VEA'].values, after['EFA'].values * 0.3))


class TestDonorsNeverReachThePanel(unittest.TestCase):

    def test_a_donor_only_symbol_is_dropped_once_spliced(self):
        idx = pd.bdate_range('2000-01-03', periods=400)
        tip = pd.Series(np.nan, index=idx)
        tip.iloc[200:] = 100.0
        frame = pd.DataFrame({'TIP': tip, 'ACITX': _growth(idx, 0.0002, 10.0),
                              'SPY': _growth(idx, 0.0003)})
        store = _memory_store(frame, ['TIP', 'SPY'])
        self.assertIn('TIP', store.constructed)
        self.assertNotIn('ACITX', store.adj_close().columns)
        self.assertNotIn('ACITX', store.tickers)

    def test_a_requested_donor_stays(self):
        """EFA is both a donor (to VEA) and traded (by GTAA_G5). Asked for, it stays."""
        idx = pd.bdate_range('2000-01-03', periods=400)
        vea = pd.Series(np.nan, index=idx)
        vea.iloc[200:] = 50.0
        frame = pd.DataFrame({'VEA': vea, 'EFA': _growth(idx, 0.0002), 'SPY': 100.0})
        store = _memory_store(frame, ['VEA', 'EFA', 'SPY'])
        self.assertIn('EFA', store.adj_close().columns)


class TestTheCalendarBelongsToWhatTrades(unittest.TestCase):

    def test_a_date_only_a_donor_carries_is_removed(self):
        idx = pd.bdate_range('2000-01-03', periods=300)
        holiday = idx[150]
        spy = _growth(idx, 0.0003)
        spy.loc[holiday] = np.nan                   # the US market was shut that day ...
        iwm = pd.Series(np.nan, index=idx)
        iwm.iloc[250:] = 90.0
        frame = pd.DataFrame({'SPY': spy, 'IWM': iwm,
                              'NAESX': _growth(idx, 0.0002, 20.0)})    # ... the fund struck
        store = _memory_store(frame, ['SPY', 'IWM'])
        self.assertNotIn(holiday, store.adj_close().index,
                         'a donor added a US session on which a fill could be priced')
        self.assertIn('IWM', store.constructed)

    def test_the_calendar_is_untouched_when_nothing_is_extra(self):
        idx = pd.bdate_range('2000-01-03', periods=300)
        frame = pd.DataFrame({'SPY': _growth(idx, 0.0003), 'NAESX': _growth(idx, 0.0002)})
        store = _memory_store(frame, ['SPY'])
        self.assertEqual(len(store.adj_close()), len(idx))


class TestAMissingDonorIsNeverSilent(unittest.TestCase):

    def test_an_absent_donor_is_recorded_and_the_recipient_left_alone(self):
        idx = pd.bdate_range('2000-01-03', periods=300)
        hyg = pd.Series(np.nan, index=idx)
        hyg.iloc[150:] = 80.0
        store = _memory_store(pd.DataFrame({'HYG': hyg, 'SPY': 100.0}, index=idx),
                              ['HYG', 'SPY'])
        self.assertNotIn('HYG', store.constructed)
        self.assertEqual(store.adj_close()['HYG'].first_valid_index(), idx[150])
        self.assertTrue(any(m.startswith('HYG<-VWEHX') for m in store.missing_donors),
                        store.missing_donors)
        self.assertEqual(store.provenance()['missing_donors'], store.missing_donors)

    def test_a_junction_a_few_sessions_stale_is_bridged(self):
        """A fixing missing on the recipient's first day: the last one within the stale
        limit anchors the junction, the same tolerance every other gap gets."""
        idx = pd.bdate_range('2000-01-03', periods=300)
        gld = pd.Series(np.nan, index=idx)
        gld.iloc[200:] = 400.0
        lbma = _growth(idx, 0.0001, 300.0)
        lbma.iloc[199:201] = np.nan                     # no fixing on the first GLD day
        frame = pd.DataFrame({'GLD': gld, 'LBMA_GOLD_PM': lbma, 'SPY': 100.0})
        store = _memory_store(frame, ['GLD', 'SPY'])
        self.assertIn('GLD', store.constructed)
        self.assertEqual(store.missing_donors, [])

    def test_a_junction_too_stale_is_refused_not_bridged(self):
        idx = pd.bdate_range('2000-01-03', periods=300)
        gld = pd.Series(np.nan, index=idx)
        gld.iloc[200:] = 400.0
        lbma = _growth(idx, 0.0001, 300.0)
        lbma.iloc[190:201] = np.nan                     # eleven sessions without a fixing
        frame = pd.DataFrame({'GLD': gld, 'LBMA_GOLD_PM': lbma, 'SPY': 100.0})
        store = _memory_store(frame, ['GLD', 'SPY'])
        self.assertNotIn('GLD', store.constructed)
        self.assertTrue(any('GLD<-LBMA_GOLD_PM' in m for m in store.missing_donors))


class TestTheLbmaParserIsStrict(unittest.TestCase):
    """Every body refused here is one a tolerant reader would have turned into a short GLD."""

    @staticmethod
    def _body(n=1500, usd=1800.0):
        days = pd.bdate_range('1990-01-01', periods=n)
        return json.dumps([{'is_cms_locked': 0, 'd': str(d.date()), 'v': [usd, 1400.0, None]}
                           for d in days])

    def test_a_well_formed_body_parses_to_the_usd_column(self):
        s = de.parse_lbma_json(self._body())
        self.assertEqual(len(s), 1500)
        self.assertTrue((s == 1800.0).all(), 'read the GBP or EUR column instead of USD')

    def test_an_html_gateway_page_is_refused(self):
        with self.assertRaises(ValueError):
            de.parse_lbma_json('<!DOCTYPE html><html><body>Access denied</body></html>')

    def test_an_empty_list_is_refused(self):
        with self.assertRaises(ValueError):
            de.parse_lbma_json('[]')

    def test_a_truncated_history_is_refused(self):
        with self.assertRaises(ValueError):
            de.parse_lbma_json(self._body(n=50))

    def test_a_row_of_the_wrong_shape_is_refused(self):
        with self.assertRaises(ValueError):
            de.parse_lbma_json(json.dumps([{'date': '2000-01-03', 'usd': 280.0}] * 2000))

    def test_a_fixing_with_no_usd_value_is_skipped_not_zeroed(self):
        rows = json.loads(self._body())
        rows[10]['v'][0] = None
        s = de.parse_lbma_json(json.dumps(rows))
        self.assertEqual(len(s), 1499)
        self.assertFalse((s == 0).any())


class TestTheExternalDonorIsReadFromItsCache(unittest.TestCase):
    """The on-disk path, end to end, with the network locked."""

    def _write(self, tmp, idx):
        gld = pd.Series(np.nan, index=idx)
        gld.iloc[200:] = 400.0
        frame = pd.DataFrame({'GLD': gld, 'SPY': _growth(idx, 0.0003)})
        for field in ('open', 'close', 'adj_close'):
            frame.to_csv(os.path.join(tmp, f'daily_{field}.csv'))

    def test_a_cached_fixing_extends_gld_and_is_recorded(self):
        idx = pd.bdate_range('2000-01-03', periods=400)
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, idx)
            _growth(idx, 0.0001, 300.0).rename('LBMA_GOLD_PM').to_frame().to_csv(
                os.path.join(tmp, 'donor_LBMA_GOLD_PM.csv'))
            store = PriceStore(['GLD', 'SPY'], start='2000-01-03', cache_dir=tmp,
                               download=False, refresh_hours=0.0)
        self.assertEqual(store.adj_close()['GLD'].first_valid_index(), idx[0])
        self.assertNotIn('LBMA_GOLD_PM', store.adj_close().columns)
        ext = store.provenance()['external_donors']['LBMA_GOLD_PM']
        self.assertEqual(len(ext['sha256']), 64)
        self.assertIn('lbma.org.uk', ext['source'])

    def test_no_cache_and_no_download_is_recorded_not_fetched(self):
        idx = pd.bdate_range('2000-01-03', periods=400)
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, idx)
            store = PriceStore(['GLD', 'SPY'], start='2000-01-03', cache_dir=tmp,
                               download=False, refresh_hours=0.0)
        self.assertEqual(store.adj_close()['GLD'].first_valid_index(), idx[200])
        self.assertTrue(any('LBMA_GOLD_PM' in m for m in store.missing_donors))


class TestTheCacheIsExtendedAtTheHead(unittest.TestCase):
    """Moving DATA_START_DATE earlier must reach the cache, not be silently ignored."""

    def test_a_cache_opening_after_the_start_is_downloaded_again(self):
        idx = pd.bdate_range('2005-01-03', periods=300)
        calls = []

        def fake_download(self, tickers, start, quiet=False):
            calls.append(pd.Timestamp(start))
            full = pd.bdate_range('2000-01-03', periods=1600)
            f = pd.DataFrame({t: _growth(full, 0.0003) for t in tickers})
            return {'open': f, 'close': f, 'adj_close': f.copy()}

        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.DataFrame({'SPY': _growth(idx, 0.0003)})
            for field in ('open', 'close', 'adj_close'):
                frame.to_csv(os.path.join(tmp, f'daily_{field}.csv'))
            with patch.object(PriceStore, '_download', fake_download):
                store = PriceStore(['SPY'], start='2000-01-03', cache_dir=tmp,
                                   download=True, refresh_hours=0.0)
        self.assertEqual(calls[0], pd.Timestamp('2000-01-03'))
        self.assertEqual(store.adj_close().index[0], pd.Timestamp('2000-01-03'))

    def test_the_tail_refresh_never_asks_for_a_donor(self):
        """A mutual fund distributes monthly; refreshing donors would re-download their whole
        history every month to learn nothing, since only pre-inception bars are read."""
        idx = pd.bdate_range('2000-01-03', periods=600)
        asked = []

        def fake_download(self, tickers, start, quiet=False):
            asked.append(list(tickers))
            f = pd.DataFrame({t: _growth(idx, 0.0003) for t in tickers})
            return {'open': f, 'close': f, 'adj_close': f.copy()}

        with tempfile.TemporaryDirectory() as tmp:
            hyg = _growth(idx, 0.0002)
            hyg.iloc[:300] = np.nan
            frame = pd.DataFrame({'HYG': hyg, 'VWEHX': _growth(idx, 0.0002, 5.0),
                                  'SPY': _growth(idx, 0.0003)})
            for field in ('open', 'close', 'adj_close'):
                frame.to_csv(os.path.join(tmp, f'daily_{field}.csv'))
            with patch.object(PriceStore, '_download', fake_download):
                PriceStore(['HYG', 'SPY'], start='2000-01-03', cache_dir=tmp,
                           download=True, refresh_hours=0.0)
        self.assertTrue(asked, 'the refresh never ran, so this test checks nothing')
        for tickers in asked:
            self.assertNotIn('VWEHX', tickers)


class TestEveryWindowSetterCanReachTheEra(unittest.TestCase):
    """Structural: the era is 2000-01 only if nothing allowed to set the window depends on a
    ticker that neither predates the data start nor has a declared donor. Checkable without
    a single price, so a strategy added on a late ETF fails HERE rather than by quietly
    pulling the shared window forward the next time someone runs the full registry."""

    #: Tickers whose OWN history predates main.DATA_START_DATE, with their inception.
    NATIVE = {'SPY': '1993-01-29', 'EWJ': '1996-03-18'}

    def test_every_signal_ticker_of_every_window_setter_is_covered(self):
        import main
        synthetic = set(de.SYNTHETIC_CASH)
        for name, factory in main.ALL_STRATEGIES.items():
            strat = factory()
            if not main.may_set_ranked_window(strat):
                continue
            sl = strat.sleeves()
            tickers = set(sl['offensive']) | set(sl['defensive']) | set(sl['canary'])
            for t in sorted(tickers):
                with self.subTest(strategy=name, ticker=t):
                    ok = (t in HISTORY_BACKFILL or t in synthetic
                          or (t in self.NATIVE
                              and pd.Timestamp(self.NATIVE[t])
                              <= pd.Timestamp(main.DATA_START_DATE)))
                    self.assertTrue(ok, f'{t} has no route back to {main.DATA_START_DATE}')

    def test_the_data_start_is_derived_from_the_era(self):
        """The first return of the era needs a decision the month before, which needs
        LOOKBACK_MONTHS complete months of prices before THAT."""
        import main
        from common import eras
        first_return = pd.Period(eras.COMMON_ERA_START, 'M')
        self.assertEqual(pd.Period(main.DATA_START_DATE, 'M') + main.LOOKBACK_MONTHS + 1,
                         first_return)


if __name__ == '__main__':
    unittest.main()


class TestTheFidelityArithmetic(unittest.TestCase):
    """`tools/proxy_fidelity.measure` — the only part of that tool the suite can reach.

    Anchored on cases whose answer is known without the code: a series against itself, and a
    donor that beats the ETF by a known constant every month.
    """

    def setUp(self):
        from tools import proxy_fidelity
        self.measure = proxy_fidelity.measure
        idx = pd.bdate_range('2001-01-01', periods=21 * 120)
        rng = np.random.default_rng(20260923)
        self.etf = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx))),
                             index=idx)

    def test_a_series_against_itself_is_perfect(self):
        m = self.measure(self.etf, self.etf)
        self.assertAlmostEqual(m['corr'], 1.0, places=12)
        self.assertEqual(m['sign'], 1.0)
        self.assertAlmostEqual(m['drift'], 0.0, places=12)

    def test_a_known_monthly_edge_shows_as_drift_of_the_right_size(self):
        monthly = self.etf.resample('ME').last()
        edge = 1.001 ** np.arange(len(monthly))               # +0.1% a month, compounded
        donor = pd.Series(monthly.values * edge, index=monthly.index)
        m = self.measure(self.etf, donor)
        self.assertGreater(m['drift'], 0.0)
        self.assertAlmostEqual(m['drift'], 1.001 ** 12 * (1 + 0.0) - 1, delta=0.004)

    def test_too_short_an_overlap_is_none_never_agreement(self):
        self.assertIsNone(self.measure(self.etf.iloc[:21 * 20], self.etf.iloc[:21 * 20]))
