"""
Guards: the things the engine must REFUSE to do.

Every test here fails against `HEAD 3f5c775` — that is the point. The pre-audit engine had
no notion of an incomplete month, no notion of coverage, and no notion of an allocation it
could not price; it answered every one of these cases with a plausible-looking number.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from common.data_engine import (PriceStore, IncompleteMonthError, DataGapError, MAX_STALE_DAYS,
                                ADJUSTMENT_STEP_TOLERANCE, DETECT_WINDOW_DAYS)
from common.coverage import coverage_report, earliest_valid_start, signal_universe
from common.ledger import ExecutionConfig, run_ledger, WeightInvariantError
from common.manifest import build_manifest
from common.metrics import calculate_metrics
from tests.test_audit import make_daily_store

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
FROZEN_MONTHLY = os.path.join(FIXTURES, 'frozen_prices_2026-06-08.csv')


class TestIncompleteMonth(unittest.TestCase):
    """T3.1 / T7.1 — a month is not a month until it has finished."""

    def setUp(self):
        self.store = PriceStore.from_monthly_csv(FROZEN_MONTHLY)

    def test_the_fixture_still_contains_the_bad_row(self):
        """If this fails, someone 'fixed' the fixture and destroyed the regression case."""
        self.assertEqual(str(self.store.as_of().date()), '2026-06-30')

    def test_partial_month_is_refused(self):
        """The fixture's last row is labelled 2026-06-30 but holds 2026-06-05 prices.

        `resample('ME').last()` published it as a month-end, and the live signal built from
        it picked DBC and VNQ where the finished month picks VGK and VWO — two of six
        positions wrong, in a repo that places real orders.
        """
        with self.assertRaises(IncompleteMonthError) as ctx:
            self.store.assert_month_complete('2026-06-15')
        self.assertIn('2026-06', str(ctx.exception))
        self.assertIn('2026-05', str(ctx.exception))   # names the last month it CAN vouch for

    def test_incomplete_month_is_absent_from_the_calendar(self):
        month_ends = self.store.month_end_dates()
        self.assertEqual(str(month_ends[-1].date()), '2026-05-31')
        self.assertNotIn(pd.Period('2026-06', 'M'), set(pd.PeriodIndex(month_ends, freq='M')))

    def test_a_finished_month_is_accepted(self):
        self.store.assert_month_complete('2026-05-20')   # must not raise


class TestRunDateInvariance(unittest.TestCase):
    """The owner's stated design intent, encoded so it cannot regress.

    He executes in the evening, sometimes several days into the new month, and wants the
    signal to be **exactly** what it would have been on the 1st — the last completed
    month-end, never a sliding window. That was always the intent, and
    `signal_date = prices.index[prices.index < exec_dt][-1]` has expressed it correctly since
    the initial commit.

    It was never the run date that broke. It was what the month-end ROW CONTAINED: the old
    monthly cache wrote a row the first time the program ran during a month, labelled it with
    `resample('ME')`'s synthetic month-end, and — because the incremental refresh appended
    only strictly-newer dates — never corrected it. So running mid-month wrote a partial month
    under a finished month's name and froze it there. Perfectly stable, and stably wrong.

    Two properties are needed, and only together do they deliver the intent:
      1. the signal does not depend on the day you run   (this class, first test)
      2. the month-end row is the real month-end         (this class, second test)
    """

    @classmethod
    def setUpClass(cls):
        cls.store = make_daily_store()

    def _signal(self, run_date, name='HAA_G12'):
        import main
        cfg = {'DATA_START_DATE': '2015-06-01', 'EXECUTION_MODE': True,
               'CURRENT_EXECUTION_DATE': run_date, 'END_DATE': run_date}
        prices, sw, su = main.build_signal_panel(self.store, cfg)
        available = prices.index[prices.index < pd.to_datetime(run_date)]
        signal_date = available[-1]
        strat = main.ALL_STRATEGIES[name]()
        scores = su if strat.score_type == 'unweighted' else sw
        alloc = strat.generate_allocations(prices.loc[:signal_date],
                                           scores.loc[:signal_date], None, None).iloc[-1]
        return signal_date, sorted(alloc[alloc > 1e-9].index.tolist())

    def test_the_day_of_the_month_you_run_on_changes_nothing(self):
        """Run on the 1st, the 3rd, mid-month or the 28th — identical basket."""
        month = self.store.month_end_dates()[-1] + pd.Timedelta(days=1)
        runs = [month + pd.Timedelta(days=n) for n in (0, 2, 6, 14, 25)]
        results = [self._signal(str(r.date())) for r in runs]
        first_date, first_basket = results[0]
        for run, (date, basket) in zip(runs, results):
            with self.subTest(run_date=str(run.date())):
                self.assertEqual(date, first_date)
                self.assertEqual(basket, first_basket)

    def test_the_signal_date_is_a_finished_month(self):
        """The other half of the intent: the row must be the REAL month-end.

        Stability alone is not enough — the pre-audit engine was perfectly stable across run
        dates while serving a five-day-old price under a month-end label.
        """
        run_date = self.store.month_end_dates()[-1] + pd.Timedelta(days=3)
        signal_date, _ = self._signal(str(run_date.date()))
        # It is the last COMPLETE month's real last trading day...
        self.assertEqual(signal_date, self.store.month_end_dates()[-1])
        # ... and the store can vouch for that month having actually ended.
        self.store.assert_month_complete(signal_date)
        # ... and no LATER month has leaked in.
        self.assertLess(pd.Period(signal_date, 'M'), pd.Period(run_date, 'M'))


class TestConstructedHistory(unittest.TestCase):
    """2026-07-29 — extending history backwards is inventing data unless it is fenced.

    Three fences, and each is a separate test because each fails differently:

    1. A constructed span must END before the ticker's own first real observation. If it
       ever overlapped, a synthetic price could reach the live order path — the one place
       in this repository that spends real money.
    2. The splice must not create a step. Chain-linking on LEVEL means the return on the
       junction day is the DONOR's return, not a jump between two funds' price scales.
    3. Every constructed span must be in `provenance()`, so it travels with every manifest.
    """

    def _store(self):
        """A store whose donors start earlier than their recipients, built in memory."""
        idx = pd.bdate_range('2000-01-03', periods=1800)
        n = len(idx)
        donor = pd.Series(100.0 * (1.0004 ** np.arange(n)), index=idx)
        # Recipient exists only for the last 800 days, on a different price scale.
        recip = pd.Series(np.nan, index=idx)
        recip.iloc[-800:] = donor.iloc[-800:].values * 0.37
        frame = pd.DataFrame({'VEA': recip, 'EFA': donor, 'SPY': donor * 1.5})
        store = PriceStore.from_adjusted(frame, frame.copy(), source='synthetic')
        # from_adjusted deliberately does NOT extend; drive the mechanism directly.
        store.constructed = {}
        store._extend_history()
        return store, idx

    def test_a_constructed_span_never_reaches_the_real_history(self):
        store, _ = self._store()
        rec = store.constructed.get('VEA')
        self.assertIsNotNone(rec, 'the splice did not run')
        self.assertLess(pd.Timestamp(rec['to']), pd.Timestamp(rec['real_from']),
                        'a constructed price overlapped the fund\'s own history — that is '
                        'the path by which invented data could reach a live order')

    def test_the_splice_creates_no_step(self):
        """The junction return must be the donor's, not a jump between price scales."""
        store, _ = self._store()
        ac = store.adj_close()
        junction = pd.Timestamp(store.constructed['VEA']['real_from'])
        spliced = ac['VEA'].pct_change().loc[junction]
        donor = ac['EFA'].pct_change().loc[junction]
        self.assertAlmostEqual(float(spliced), float(donor), places=10,
                               msg='chain-linking must preserve the donor RETURN and discard '
                                   'its price level entirely')

    def test_the_recipient_keeps_its_own_prices_after_inception(self):
        store, _ = self._store()
        ac = store.adj_close()
        junction = pd.Timestamp(store.constructed['VEA']['real_from'])
        after = ac.loc[ac.index >= junction]
        # Recipient was built at 0.37x the donor; the splice must not have rescaled that.
        self.assertTrue(np.allclose(after['VEA'].values, after['EFA'].values * 0.37),
                        'the traded fund\'s own prices were overwritten — only the LEADING '
                        'gap may ever be filled')

    def test_provenance_carries_every_constructed_span(self):
        store, _ = self._store()
        prov = store.provenance()
        self.assertIn('constructed_history', prov)
        self.assertIn('VEA', prov['constructed_history'])
        for key in ('kind', 'source', 'why', 'from', 'to', 'real_from'):
            self.assertIn(key, prov['constructed_history']['VEA'])

    def test_a_frozen_fixture_is_never_extended(self):
        """The golden master rests on the fixture. A fixture that grew history silently
        would stop being frozen, and the ratchet would be measuring a moving target."""
        store = PriceStore.from_monthly_csv(FROZEN_MONTHLY)
        self.assertEqual(store.constructed, {})

    def test_the_rate_symbol_is_not_a_tradable_column(self):
        """^IRX is a YIELD. If it survived in the frames, momentum could rank it and a
        universe declaration could hold it, which would be nonsense priced as a portfolio."""
        from common.data_engine import SYNTHETIC_CASH
        idx = pd.bdate_range('2000-01-03', periods=1200)
        n = len(idx)
        rate = pd.Series(4.5, index=idx)               # 4.5% annualised, flat
        bil = pd.Series(np.nan, index=idx)
        bil.iloc[-400:] = 91.0 * (1.00005 ** np.arange(400))
        frame = pd.DataFrame({'BIL': bil, '^IRX': rate, 'SPY': 100.0 * (1.0003 ** np.arange(n))})
        store = PriceStore.from_adjusted(frame, frame.copy(), source='synthetic')
        store.constructed = {}
        store._extend_history()

        self.assertIn('BIL', store.constructed)
        self.assertNotIn('^IRX', store.adj_close().columns)
        self.assertNotIn('^IRX', store.tickers)

    def test_the_synthetic_bill_accrues_at_the_bond_equivalent_of_its_quote(self):
        """External anchor: the arithmetic of a T-bill, done here from first principles.

        `^IRX` publishes a DISCOUNT rate. A 91-day bill quoted at 4.50% costs
        `100 - 4.50 x 91/360 = 98.8625` and repays 100, so it RETURNS 4.70% a year, not
        4.50%. This test computes that conversion itself and requires the store to match —
        the two classic failures being to skip the conversion entirely (20bp/yr too low for
        a decade) and to divide by 12 or 252 somewhere (wrong by a factor)."""
        from common.data_engine import SYNTHETIC_CASH
        expense = SYNTHETIC_CASH['BIL'][1]
        price = 1.0 - 0.045 * 91.0 / 360.0
        bond_equivalent = (1.0 + (1.0 - price) / price) ** (365.0 / 91.0) - 1.0
        self.assertAlmostEqual(bond_equivalent, 0.0470, places=3)   # the anchor itself
        idx = pd.bdate_range('2000-01-03', periods=1200)
        n = len(idx)
        rate = pd.Series(4.5, index=idx)
        bil = pd.Series(np.nan, index=idx)
        bil.iloc[-400:] = 91.0
        frame = pd.DataFrame({'BIL': bil, '^IRX': rate, 'SPY': pd.Series(100.0, index=idx)})
        store = PriceStore.from_adjusted(frame, frame.copy(), source='synthetic')
        store.constructed = {}
        store._extend_history()

        s = store.adj_close()['BIL'].loc[:store.constructed['BIL']['to']]
        years = (s.index[-1] - s.index[0]).days / 365.25
        cagr = (float(s.iloc[-1]) / float(s.iloc[0])) ** (1 / years) - 1
        self.assertAlmostEqual(cagr, bond_equivalent - expense, places=3,
                               msg=f'{cagr:.4%} — a bill quoted at a 4.50% DISCOUNT returns '
                                   f'{bond_equivalent:.4%}, and the constructed series must '
                                   f'show that net of the {expense:.4%} expense ratio')


class TestRiskFreeProvenance(unittest.TestCase):
    """The risk-free rate must say which of its months are constructed.

    `_extend_history` deliberately makes a constructed span indistinguishable from real
    history *for* `first_tradable_date`, because the coverage guard needs exactly that.
    `build_rf_series` used the same call as its honesty test, so the fence that helps
    coverage silently defeated disclosure: with BIL accrued back to 2000 from ^IRX there
    were no NaNs left to count, and the header read "realised BIL total return from
    2000-01-03" while 88 of 318 monthly observations were synthetic. Every Sharpe, Sortino
    and UPI for a strategy opening before BIL's 2007-05 inception was net of it.
    """

    def _store_with_synthetic_bil(self):
        idx = pd.bdate_range('2004-01-01', periods=900)
        rate = pd.Series(4.5, index=idx)
        bil = pd.Series(np.nan, index=idx)
        bil.iloc[-400:] = 91.0                      # the fund itself starts 500 days in
        frame = pd.DataFrame({'BIL': bil, '^IRX': rate,
                              'SPY': pd.Series(100.0, index=idx)})
        store = PriceStore.from_adjusted(frame, frame.copy(), source='synthetic')
        store.constructed = {}
        store._extend_history()
        return store

    def test_the_description_names_the_constructed_months_and_their_source(self):
        from common.metrics import build_rf_series
        store = self._store_with_synthetic_bil()
        dates = store.month_end_dates()
        _, desc = build_rf_series(store, dates, 'BIL')

        self.assertIn('CONSTRUCTED', desc,
                      'a synthetic rate reported as "realised" is the exact failure this '
                      'repository treats as disqualifying everywhere else')
        self.assertIn('^IRX', desc, 'name the source, or the claim cannot be checked')
        real_from = store.constructed_before('BIL')
        self.assertIsNotNone(real_from)
        # The count must be the month-ends that predate the fund, and no others.
        want = int((dates < real_from).sum()) - 1     # the first month-end has no pct_change
        self.assertIn(f'{want} earlier month(s) CONSTRUCTED', desc)

    def test_nothing_constructed_still_reads_realised_with_no_qualifier(self):
        """The frozen-fixture path builds no synthetic history, and must not be made to
        apologise for data that is genuinely the fund's own."""
        from common.metrics import build_rf_series
        idx = pd.bdate_range('2012-01-02', periods=600)
        frame = pd.DataFrame({'BIL': 91.0 * (1.00002 ** np.arange(len(idx))),
                              'SPY': 100.0}, index=idx)
        store = PriceStore.from_adjusted(frame, frame.copy(), source='synthetic')
        _, desc = build_rf_series(store, store.month_end_dates(), 'BIL')
        self.assertNotIn('CONSTRUCTED', desc)
        self.assertIn('realised', desc)


class TestCoverage(unittest.TestCase):
    """T3.2 / T7.2 — never measure a strategy over years its assets did not exist."""

    @classmethod
    def setUpClass(cls):
        cls.store = make_daily_store()

    def test_missing_ticker_is_a_configuration_error_not_a_trim(self):
        from strategies.base import BaseStrategy

        class _NeedsUnknown(BaseStrategy):
            def __init__(self):
                super().__init__('NeedsUnknown')

            def sleeves(self):
                return {'offensive': {'NOT_A_TICKER'}, 'defensive': set(), 'canary': []}

        with self.assertRaises(KeyError):
            coverage_report(_NeedsUnknown(), self.store, '2016-01-01')

    def test_leveraged_wrap_declares_its_letf_images(self):
        """Coverage must look at what is HELD (UWM/QLD/SSO), not at the 1x signal assets.

        This is the declaration that stops a 2x variant being measured across a bear market
        its products did not live through.
        """
        import main
        uni = signal_universe(main.ALL_STRATEGIES['HAA_G3_Leveraged_2X']())
        self.assertLessEqual({'SSO', 'QLD', 'UWM'}, uni)

    def test_late_inception_trims_the_window_and_says_why(self):
        """A real store: TQQQ starts 2010-02, so no 3x QQQ result before ~2011-03 exists."""
        import main
        adj = self.store.adj_close().copy()
        opn = self.store.adj_open().copy()
        # Blank UWM before a cut-off to simulate a late product inception.
        cut = adj.index[400]
        adj.loc[adj.index < cut, 'UWM'] = np.nan
        opn.loc[opn.index < cut, 'UWM'] = np.nan
        store = PriceStore.from_adjusted(adj, opn)

        strat = main.ALL_STRATEGIES['HAA_G3_Leveraged_2X']()
        rep = coverage_report(strat, store, requested_start=adj.index[0], warmup_months=13)
        self.assertTrue(rep['trimmed'])
        self.assertEqual(rep['binding_ticker'], 'UWM')
        self.assertGreaterEqual(rep['earliest'], cut)
        self.assertIn('UWM', rep['message'])
        self.assertIn('binding', rep['message'])

    def test_earliest_valid_start_leaves_room_for_the_warmup(self):
        import main
        strat = main.ALL_STRATEGIES['HAA_G12']()
        first = self.store.month_end_dates()[0]
        earliest = earliest_valid_start(strat, self.store, warmup_months=13)
        n_before = len(self.store.month_end_dates(end=earliest))
        self.assertGreaterEqual(n_before, 14)          # 13 warm-up months + the decision
        self.assertGreater(earliest, first)


class TestRegimePanelIgnoresTheRequestedWindow(unittest.TestCase):
    """The regime panel must be immune to date-selection bias, including the caller's own.

    A panel truncated at whatever START_DATE happened to be requested is not an antidote to
    arbitrary date selection — it is another expression of it. Before this was fixed, a run
    starting in 2015 printed `n/a (inception 2015-01)` for the GFC *even for SPY*, which has
    data back to 2000: the run's parameters were silently deciding which history existed.

    So: the episode list is frozen in `common/regimes.py`, and each cell is measured over the
    strategy's FULL available history. An `n/a` therefore means the assets did not exist — a
    fact about the market — never "you asked for a shorter window".
    """

    #: Starts 2013-01 so that the 2015-16 episode sits AFTER the 13-month warm-up but
    #: BEFORE the late START_DATE below — which is exactly the case that used to vanish.
    @classmethod
    def setUpClass(cls):
        cls.store = make_daily_store(n_days=2090, start='2013-01-01')

    def _panel(self, start):
        import main
        from common.regimes import episode_panel
        config = {
            'START_DATE': start, 'END_DATE': '2021-01-01',
            'DATA_START_DATE': '2013-01-01', 'CURRENT_EXECUTION_DATE': '2021-01-01',
            'EXECUTION_MODE': False, 'LEVERAGE_FACTOR': 1.0, 'MARGIN_BORROW_RATE': 0.06,
            'MARGIN_FOLLOWS_SIGNAL': True, 'COST_PCT_PER_SIDE': 0.001,
            'LOOKBACK_MONTHS': 13, 'EXECUTION_CONVENTION': 'next_open',
            'CASH_TICKER': 'BIL', 'COVERAGE_POLICY': 'trim', 'RF_ANNUAL_FALLBACK': 0.0,
        }
        prices, sw, su = main.build_signal_panel(self.store, config)
        metrics, _ = main.run_backtest(prices, sw, su,
                                       [main.ALL_STRATEGIES['SPY_Benchmark']()],
                                       config, store=self.store)
        return episode_panel(metrics)['SPY_Benchmark']

    def test_an_early_and_a_late_start_give_the_same_episode_cells(self):
        early, late = self._panel('2014-06-01'), self._panel('2019-01-01')
        for key in ('q4_2018', 'covid'):
            with self.subTest(episode=key):
                self.assertNotIn('na', early[key], f'{key} unmeasurable — bad test window')
                self.assertEqual(sorted(early[key]), sorted(late[key]))
                self.assertAlmostEqual(early[key]['return'], late[key]['return'], places=10)
                self.assertAlmostEqual(early[key]['max_dd'], late[key]['max_dd'], places=10)

    def test_but_it_stops_at_the_ERA_floor_which_is_not_a_request(self):
        """The other side of the same coin, and the two are one wrong word apart.

        `coverage_report` answers a question about the DATA and correctly ignores the caller,
        so its `earliest` can predate the era: SPY is measurable from 2001-03 while the era
        opens 2004-11. `run_backtest` clamps to `eras.COMMON_ERA_START` for that reason — and
        clamping to `start_floor` instead would look identical in production (they are the same
        object there) while making the panel inherit the caller's dates, which the three tests
        around this one exist to forbid. It was written the wrong way first, and they caught it.

        Asserted on the SOURCE because the fixture store starts in 2013, long after any era
        floor this repository will ever have: a behavioural test here would pass whatever the
        code said.
        """
        import inspect

        import main
        src = inspect.getsource(main.run_backtest)
        clamp = [ln for ln in src.splitlines() if "cov['earliest'] = max(" in ln]
        self.assertEqual(len(clamp), 1, 'the era-floor clamp is gone or duplicated')
        window = src[src.index("cov['earliest'] = max("):][:200]
        self.assertIn('COMMON_ERA_START', window,
                      'the clamp no longer uses the derived era floor')
        self.assertNotIn('start_floor', window,
                         'the clamp is using the CALLER\'S start date, so the regime panel '
                         'now inherits whatever window was requested')

    def test_an_episode_before_the_requested_start_is_still_measured(self):
        """The case that used to fail: asking for 2019 onwards must not erase 2015-16."""
        late = self._panel('2019-01-01')
        self.assertNotIn('na', late['china_oil'],
                         'the 2015-16 episode vanished because START_DATE was 2019 — the '
                         'panel is inheriting the caller\'s date choice again')

    def test_headline_metrics_DO_still_honour_the_requested_window(self):
        """The other half: the ranked table must remain the window the user asked for.

        Full history is for the regime panel and the publication split, not for the headline.
        """
        import main
        config = {
            'START_DATE': '2019-01-01', 'END_DATE': '2021-01-01',
            'DATA_START_DATE': '2013-01-01', 'CURRENT_EXECUTION_DATE': '2021-01-01',
            'EXECUTION_MODE': False, 'LEVERAGE_FACTOR': 1.0, 'MARGIN_BORROW_RATE': 0.06,
            'MARGIN_FOLLOWS_SIGNAL': True, 'COST_PCT_PER_SIDE': 0.001,
            'LOOKBACK_MONTHS': 13, 'EXECUTION_CONVENTION': 'next_open',
            'CASH_TICKER': 'BIL', 'COVERAGE_POLICY': 'trim', 'RF_ANNUAL_FALLBACK': 0.0,
        }
        prices, sw, su = main.build_signal_panel(self.store, config)
        metrics, _ = main.run_backtest(prices, sw, su,
                                       [main.ALL_STRATEGIES['SPY_Benchmark']()],
                                       config, store=self.store)
        d = metrics[0]
        self.assertGreaterEqual(d['first_return'], pd.Timestamp('2019-01-01'))
        self.assertLess(d['first_return_full'], d['first_return'])


class TestWeightInvariants(unittest.TestCase):
    """T3.3 / T7.4 — an allocation the engine cannot price is refused, not scored as 0%."""

    @classmethod
    def setUpClass(cls):
        cls.store = make_daily_store()
        cls.dec = cls.store.month_end_dates()[20:40]

    def _cfg(self, **kw):
        kw.setdefault('convention', 'next_open')
        kw.setdefault('cost_bps_per_side', 0.0)
        return ExecutionConfig(**kw)

    def test_weights_that_do_not_sum_to_one_are_refused(self):
        """DAA_G12 and VAA_G12 really did produce months summing to 0.0, silently
        reported as a flat 0% return."""
        w = pd.DataFrame(0.0, index=self.dec, columns=['SPY', 'IEF'])
        w['SPY'] = 1.0
        w.iloc[5] = 0.0                                # the month that used to vanish
        with self.assertRaises(WeightInvariantError) as ctx:
            run_ledger(w, self.store, self._cfg(), label='Broken')
        self.assertIn('weights sum to 0.000000', str(ctx.exception))
        self.assertIn(str(self.dec[5].date()), str(ctx.exception))

    def test_holding_a_ticker_with_no_price_is_refused(self):
        adj, opn = self.store.adj_close().copy(), self.store.adj_open().copy()
        adj['UPRO'] = np.nan
        opn['UPRO'] = np.nan
        store = PriceStore.from_adjusted(adj, opn)
        w = pd.DataFrame(0.0, index=self.dec, columns=['UPRO', 'IEF'])
        w['UPRO'] = 1.0
        with self.assertRaises(WeightInvariantError) as ctx:
            run_ledger(w, store, self._cfg(), label='Ghost')
        self.assertIn('UPRO', str(ctx.exception))
        self.assertIn('no price', str(ctx.exception))

    def test_every_registered_strategy_prices_cleanly(self):
        """All 39 registry keys, end to end, with the invariants ON."""
        import main
        config = {
            'START_DATE': '2016-09-01', 'END_DATE': '2021-01-01',
            'DATA_START_DATE': '2015-06-01', 'CURRENT_EXECUTION_DATE': '2021-01-01',
            'EXECUTION_MODE': False, 'LEVERAGE_FACTOR': 1.0, 'MARGIN_BORROW_RATE': 0.06,
            'MARGIN_FOLLOWS_SIGNAL': True, 'COST_PCT_PER_SIDE': 0.001,
            'LOOKBACK_MONTHS': 13, 'EXECUTION_CONVENTION': 'next_open',
            'CASH_TICKER': 'BIL', 'COVERAGE_POLICY': 'trim', 'RF_ANNUAL_FALLBACK': 0.0,
        }
        prices, sw, su = main.build_signal_panel(self.store, config)
        strategies = [f() for f in main.ALL_STRATEGIES.values()]
        metrics, _ = main.run_backtest(prices, sw, su, strategies, config, store=self.store)
        scored = {d['name'] for d in metrics}
        missing = {s.name for s in strategies} - scored
        self.assertEqual(missing, set(), f'strategies that failed to price: {sorted(missing)}')
        for d in metrics:
            with self.subTest(strategy=d['name']):
                self.assertTrue(np.isfinite(d['cagr']))
                self.assertLessEqual(d['max_dd'], 1e-12)


class TestMissingDataPolicy(unittest.TestCase):
    """T3.4 — one policy, applied once, recorded."""

    def _store(self, gap_len, strict):
        idx = pd.bdate_range('2020-01-01', periods=120)
        s = pd.Series(np.linspace(100, 130, 120), index=idx)
        frame = pd.DataFrame({'SPY': s, 'BIL': s * 0 + 100.0})
        frame.iloc[50:50 + gap_len, 0] = np.nan
        return PriceStore.from_adjusted(frame, frame, strict_gaps=strict)

    def test_short_gap_is_filled_and_recorded(self):
        store = self._store(MAX_STALE_DAYS, strict=False)
        self.assertFalse(store.adj_close()['SPY'].isna().any())
        logged = {e['ticker']: e for e in store.provenance()['forward_filled']}
        self.assertEqual(logged['SPY']['longest_run'], MAX_STALE_DAYS)

    def test_long_gap_is_an_error_not_a_silent_carry(self):
        with self.assertRaises(DataGapError) as ctx:
            self._store(MAX_STALE_DAYS + 3, strict=True)
        self.assertIn('SPY', str(ctx.exception))

    def test_leading_nans_are_never_filled(self):
        """Pre-inception must stay NaN, or first_tradable_date means nothing and the
        coverage guard has nothing to bite on."""
        idx = pd.bdate_range('2020-01-01', periods=60)
        frame = pd.DataFrame({'SPY': np.linspace(100, 120, 60)}, index=idx)
        frame.iloc[:20, 0] = np.nan
        store = PriceStore.from_adjusted(frame, frame)
        self.assertTrue(store.adj_close()['SPY'].iloc[:20].isna().all())
        self.assertEqual(store.first_tradable_date('SPY'), idx[20])

    def test_every_canary_treats_nan_as_risk_off(self):
        """A canary with no data must never read as bullish.

        RAA's `any(score <= 0)` returned False for NaN, structurally disabling its crash
        protection before BND existed. RAA is gone, but the policy is asserted across what
        remains so it cannot come back in.
        """
        import main
        from common.momentum import calc_13612w, calc_13612u
        from tests.test_audit import make_prices
        prices = make_prices()
        blind = prices.copy()
        for strat in (main.ALL_STRATEGIES['HAA_G12'](), main.ALL_STRATEGIES['DAA_G12'](),
                      main.ALL_STRATEGIES['BAA_G12']()):
            canary = list(strat.sleeves()['canary'])
            b = blind.copy()
            b[canary] = np.nan
            sw, su = calc_13612w(b), calc_13612u(b)
            scores = su if strat.score_type == 'unweighted' else sw
            alloc = strat.generate_allocations(b, scores, None, None)
            defensive = strat.sleeves()['defensive']
            with self.subTest(strategy=strat.name):
                held = alloc.iloc[13:]
                off_weight = held.drop(columns=[c for c in defensive if c in held.columns],
                                       errors='ignore').sum(axis=1)
                self.assertLess(float(off_weight.max()), 1e-9,
                                f'{strat.name} stayed risk-ON with a blind canary')


class TestDrawdownFromInitialWealth(unittest.TestCase):
    """T4.3 / T7.5 — a first-month crash used to be invisible."""

    def test_first_month_drawdown_is_visible(self):
        r = pd.Series([-0.5, 0.0, 0.1],
                      index=pd.date_range('2020-01-31', periods=3, freq='ME'))
        self.assertAlmostEqual(calculate_metrics(r)['max_dd'], -0.5, places=12)

    def test_wealth_curve_starts_at_one(self):
        r = pd.Series([0.1, 0.1], index=pd.date_range('2020-01-31', periods=2, freq='ME'))
        self.assertAlmostEqual(calculate_metrics(r)['cum_ret'].iloc[0], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestUninvestedCashIsCapitalNotVapour(unittest.TestCase):
    """Weight that reaches no ticker must still be money.

    `run_ledger` carries the book as notional per ticker, so anything not assigned to a
    ticker was not carried at all. With a cash ticker the residue goes there and this never
    arises; with `cash_ticker=None`, or a cash ticker absent from the store, `target`
    simply omitted it and it left the portfolio.

    The size of that is worth stating: on a DEAD FLAT market a book that is 80% invested
    reported equity 1.0 -> 0.80 -> 0.64 -> 0.512. Not a one-off 20% loss -- 20% compounded
    at every single rebalance, while the recorded warning said the weight "earns 0%".

    Production never reached it. `strict_invariants` refuses a row that does not sum to 1,
    and a nonzero residue additionally needs a ticker missing from the price store. But
    unreachability is a property of today's call sites, not of this function, and the
    docstring makes its claim here.
    """

    @staticmethod
    def _flat_market():
        days = pd.bdate_range('2020-01-01', periods=200)
        px = pd.DataFrame({'SPY': 100.0, 'BIL': 100.0}, index=days)
        return PriceStore.from_frames(px, open_=px, close=px), days

    def _run(self, cash_ticker, weight=0.8):
        from common.ledger import ExecutionConfig, run_ledger
        store, days = self._flat_market()
        dec = pd.DatetimeIndex([days[0], days[40], days[80], days[120]])
        w = pd.DataFrame({'SPY': weight}, index=dec)
        cfg = ExecutionConfig(cash_ticker=cash_ticker, cost_bps_per_side=0.0,
                              charge_terminal_liquidation=False, strict_invariants=False)
        return run_ledger(w, store, cfg, label='probe')

    def test_a_flat_market_leaves_equity_flat_with_no_cash_ticker(self):
        led = self._run(None)
        for value in led.equity:
            self.assertAlmostEqual(float(value), 1.0, places=12)

    def test_the_same_holds_when_the_cash_ticker_is_absent_from_the_store(self):
        led = self._run('NOT_IN_STORE')
        for value in led.equity:
            self.assertAlmostEqual(float(value), 1.0, places=12)

    def test_the_idle_residue_is_reported_as_cash_weight(self):
        """It is cash, so it must appear in the cash weight — otherwise the execution
        panel shows a 20% hole and calls the book fully invested."""
        led = self._run(None, weight=0.75)
        for value in led.cash_weight:
            self.assertAlmostEqual(float(value), 0.25, places=12)

    def test_only_the_invested_part_participates_in_a_rally(self):
        """Idle cash earns 0%: the arithmetic that makes the fix a fix rather than a
        different error. 80% invested in an asset that doubles gives 1.8, not 2.0."""
        from common.ledger import ExecutionConfig, run_ledger
        days = pd.bdate_range('2020-01-01', periods=200)
        spy = pd.Series(100.0, index=days)
        spy.iloc[41:] = 200.0                       # doubles just after the second fill
        px = pd.DataFrame({'SPY': spy})
        store = PriceStore.from_frames(px, open_=px, close=px)
        dec = pd.DatetimeIndex([days[0], days[40], days[80]])
        w = pd.DataFrame({'SPY': 0.8}, index=dec)
        cfg = ExecutionConfig(cash_ticker=None, cost_bps_per_side=0.0,
                              charge_terminal_liquidation=False, strict_invariants=False)
        led = run_ledger(w, store, cfg, label='probe')
        self.assertAlmostEqual(float(led.equity.iloc[-1]), 1.8, places=12)


class TestTheCacheIsNotRecheckedOnEveryRun(unittest.TestCase):
    """A cache that already holds the newest bar it can must not phone Yahoo again.

    The only guard used to be `if last >= today: return`, comparing the newest cached bar
    against the CALENDAR date. That is false for the whole of every trading day — the newest
    bar is yesterday's close until this session ends — so every run paid a full network
    round-trip for 90 days of every ticker and rewrote ~10 MB of CSV to learn that nothing
    had changed. Measured on 35 tickers: 0.18s to read the cache, 9.09s with the refresh.

    Skipping is safe because of what the cache is FOR: signals come from COMPLETE months,
    and live order sizing calls `get_live_prices()`, a separate real-time quote fetch that
    never touches this cache. An intraday-stale daily cache cannot change a decision.

    Skipping is only acceptable because it is VISIBLE — `refresh_skipped` is set and the
    report prints it. A silent stale cache would be a correctness bug wearing a speed-up.
    """

    def _cached_store(self, tmp, refresh_hours):
        idx = pd.bdate_range('2024-01-01', periods=400)
        frame = pd.DataFrame({'SPY': 100.0 + np.arange(len(idx)), 'BIL': 91.0}, index=idx)
        for field in ('adj_close', 'close', 'open'):
            frame.to_csv(os.path.join(tmp, f'daily_{field}.csv'))
        return PriceStore(['SPY', 'BIL'], start='2024-01-01', cache_dir=tmp,
                          download=False, refresh_hours=refresh_hours)

    def test_a_recent_check_skips_the_download_and_says_so(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = self._cached_store(tmp, refresh_hours=6.0)
            store._stamp_refresh()                       # pretend we just checked
            calls = []
            store._download = lambda *a, **k: calls.append(1)
            store._refresh_tail()
            self.assertEqual(calls, [], 'the network was hit for a cache checked seconds ago')
            self.assertIsNotNone(store.refresh_skipped,
                                 'a skipped refresh must be reported, never silent')
            self.assertIn('--refresh', store.refresh_skipped,
                          'tell the user how to override it')

    def test_an_expired_interval_downloads_again(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = self._cached_store(tmp, refresh_hours=6.0)
            with open(store._refresh_stamp_path(), 'w', encoding='utf-8') as fh:
                fh.write((pd.Timestamp.now() - pd.Timedelta(hours=7)).isoformat())
            calls = []

            def fake(tickers, start, quiet=False):
                calls.append(start)
                raise RuntimeError('offline')            # the download path stops here

            store._download = fake
            store._refresh_tail()
            self.assertEqual(len(calls), 1, 'a stale stamp must trigger the re-download')

    def test_zero_hours_always_downloads(self):
        """`--refresh` and CACHE_REFRESH_HOURS=0 must reach the network unconditionally."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = self._cached_store(tmp, refresh_hours=0.0)
            store._stamp_refresh()
            calls = []

            def fake(tickers, start, quiet=False):
                calls.append(start)
                raise RuntimeError('offline')

            store._download = fake
            store._refresh_tail()
            self.assertEqual(len(calls), 1)
            self.assertIsNone(store.refresh_skipped)

    def test_a_missing_stamp_refreshes_rather_than_assuming_freshness(self):
        """The failure direction matters: no stamp means unknown, and unknown must mean
        check. Assuming a cache is fresh because we cannot prove it is stale is how a
        month-old signal reaches an order."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = self._cached_store(tmp, refresh_hours=6.0)
            self.assertIsNone(store._last_refresh_attempt())
            calls = []

            def fake(tickers, start, quiet=False):
                calls.append(start)
                raise RuntimeError('offline')

            store._download = fake
            store._refresh_tail()
            self.assertEqual(len(calls), 1)

    def test_a_failed_stamp_write_leaves_the_previous_stamp_intact(self):
        """Atomic replace, so a crash mid-write cannot destroy a good stamp.

        The read path was already safe — a truncated stamp parses as ValueError and means
        'unknown', which forces a refresh. This asserts the stronger property: an
        interrupted write loses NOTHING, and leaves no temp file behind to puzzle over.
        """
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            store = self._cached_store(tmp, refresh_hours=6.0)
            store._stamp_refresh()
            good = store._last_refresh_attempt()
            self.assertIsNotNone(good)

            with patch('common.data_engine.os.replace', side_effect=OSError('disk full')):
                store._stamp_refresh()          # must swallow, not raise

            self.assertEqual(store._last_refresh_attempt(), good,
                             'a failed write destroyed a stamp that was already good')
            leftovers = [n for n in os.listdir(tmp) if n.startswith('.last_refresh-')]
            self.assertEqual(leftovers, [], f'temp files left behind: {leftovers}')


class TestTheAdjustmentVintageStaysConsistent(unittest.TestCase):
    """A cached total-return series must carry ONE dividend-adjustment vintage.

    Found 2026-09-01. `adj_close` is a total-return series, so the day a fund goes
    ex-dividend Yahoo rescales every bar before that date. The incremental refresh rewrites
    only a trailing window, so after any distribution the cache held two vintages spliced at
    the window edge — recent bars adjusted for the new dividend, older bars not.

    Nothing about the splice looks wrong: every price is plausible and the series is smooth.
    But a return whose endpoints straddle the seam is measured across a discontinuity no
    market ever traded, and the error has a direction — older bars are too HIGH, so momentum
    reads too LOW. Measured on TIP: r6 and r12 understated by 0.73pp each, the 13612U canary
    score by 0.36pp, and the live HAA signal flipped from alive to dead on it.

    The invariant asserted here is the one that matters and the one no self-consistency test
    could see: a return computed FROM THE STORE must equal the same return computed on the
    vendor's current series.
    """

    TICKER = 'ZZZ'

    def _write_cache(self, tmp, series):
        frame = pd.DataFrame({self.TICKER: series})
        for field in ('adj_close', 'close', 'open'):
            frame.to_csv(os.path.join(tmp, f'daily_{field}.csv'))

    def _vintages(self):
        """(cached, current) — the same bars before and after one distribution.

        A dividend rescales every bar strictly before its ex-date and leaves later bars
        alone, which is exactly the shape that produces a seam.
        """
        idx = pd.bdate_range('2024-01-01', periods=500)
        cached = pd.Series(100.0 + 0.01 * np.arange(len(idx)), index=idx)
        current = cached.copy()
        ex_date = idx[-30]
        current.loc[idx < ex_date] *= 0.99          # ~1% distribution, restated backwards
        return idx, cached, current

    def _store_with_stub(self, tmp, cached, current):
        self._write_cache(tmp, cached)
        store = PriceStore([self.TICKER], start='2024-01-01', cache_dir=tmp,
                           download=False, refresh_hours=0.0)
        calls = []

        def fake_download(tickers, start, quiet=False):
            calls.append(pd.Timestamp(start))
            window = current.loc[current.index >= pd.Timestamp(start)]
            frame = pd.DataFrame({t: window for t in tickers})
            return {f: frame.copy() for f in ('open', 'close', 'adj_close')}

        store._download = fake_download
        return store, calls

    def test_a_restated_adjustment_rebuilds_the_whole_column(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            idx, cached, current = self._vintages()
            store, calls = self._store_with_stub(tmp, cached, current)
            store._refresh_tail()

            got = store._frames['adj_close'][self.TICKER].reindex(idx)
            pd.testing.assert_series_equal(got, current.reindex(idx), check_names=False,
                                           rtol=1e-12)
            self.assertEqual(store.readjusted, [self.TICKER],
                             'a restated history must be recorded, never silently repaired')
            self.assertEqual(len(calls), 2,
                             'one windowed refresh, then one full re-download of the ticker '
                             'whose adjustment moved')

    def test_the_seam_would_have_corrupted_a_twelve_month_return(self):
        """The consequence, asserted directly: momentum across the seam must be right.

        Without the repair the store keeps old bars at the cached vintage while the trailing
        window carries the new one, and a 12-month return straddling the join is wrong by the
        distribution -- about a full percentage point here, which is the order of magnitude
        that decides a canary.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            idx, cached, current = self._vintages()
            store, _ = self._store_with_stub(tmp, cached, current)
            store._refresh_tail()

            last, back = idx[-1], idx[-253]
            got = store._frames['adj_close'][self.TICKER]
            r_store = float(got.loc[last]) / float(got.loc[back]) - 1.0
            r_true = float(current.loc[last]) / float(current.loc[back]) - 1.0
            r_spliced = float(current.loc[last]) / float(cached.loc[back]) - 1.0

            self.assertAlmostEqual(r_store, r_true, places=12)
            self.assertGreater(abs(r_true - r_spliced), 0.008,
                               'the fixture must actually reproduce a seam worth catching')

    def test_an_unchanged_history_is_not_re_downloaded(self):
        """The check must not turn every refresh into a full rebuild: with a few dozen
        tickers something has almost always just gone ex, and rebuilding everything for one
        fund's dividend would trade a real speed-up for nothing."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            idx, cached, _ = self._vintages()
            store, calls = self._store_with_stub(tmp, cached, cached)
            store._refresh_tail()
            self.assertEqual(store.readjusted, [])
            self.assertEqual(len(calls), 1, 'only the windowed refresh should have run')

    def test_a_settling_bar_is_not_mistaken_for_a_restatement(self):
        """The newest bars change as a session closes. That is not a re-adjustment, and
        treating it as one would rebuild the world every afternoon."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            idx, cached, _ = self._vintages()
            current = cached.copy()
            current.iloc[-1] *= 1.004               # today's close moved; history did not
            store, calls = self._store_with_stub(tmp, cached, current)
            store._refresh_tail()
            self.assertEqual(store.readjusted, [])
            self.assertEqual(len(calls), 1)


class TestTheCacheIsOneAdjustmentVintage(unittest.TestCase):
    """The offline half of the 2026-09-01 defect: notice the seam without asking anyone.

    `TestTheAdjustmentVintageStaysConsistent` above pins the REPAIR — that a restated
    ticker gets re-downloaded whole. It only fires when a refresh runs. This class pins the
    DETECTION, which must work when no refresh runs at all: offline, `download=False`, or
    inside the refresh throttle. Those are the paths the defect actually survived on.

    The invariant is a property of the vendor's data model, not of this code:
    `f = adj_close / close` is the cumulative dividend-adjustment factor, dividends only
    ever scale EARLIER bars down, and raw closes are untouched — so f can only rise. Two
    vintages spliced together put a downward step in f exactly at the seam.
    """

    #: The stale-vintage inflation, as measured on the real cache on 2026-09-01.
    BUMP = 1.007338

    @classmethod
    def _panel(cls, n=400, seam_at=None):
        """(close, adj_close). `seam_at` inflates the older bars the way staleness does.

        The adjustment factor is a STEP function, not a ramp: a real one moves only on
        ex-dividend days and is flat between them. That also makes the expected seam step
        exact arithmetic rather than something read off the series under test.
        """
        idx = pd.bdate_range('2024-01-02', periods=n)
        close = pd.Series(100.0 + 0.01 * np.arange(n), index=idx)
        factor = pd.Series(0.94, index=idx)
        factor.iloc[n // 3:] = 0.97          # two notional distributions, both UPWARD
        factor.iloc[2 * n // 3:] = 1.00
        adj = close * factor
        if seam_at is not None:
            adj.loc[adj.index < idx[seam_at]] *= cls.BUMP
        return close, adj

    def _store(self, tmp, close, adj, ticker='ZZZ'):
        for field, series in (('close', close), ('open', close), ('adj_close', adj)):
            pd.DataFrame({ticker: series}).to_csv(
                os.path.join(tmp, 'daily_{}.csv'.format(field)))
        return PriceStore([ticker], start='2024-01-01', cache_dir=tmp,
                          download=False, refresh_hours=0.0)

    def test_a_spliced_vintage_is_caught_with_no_network_at_all(self):
        """The case the whole guard exists for: nothing downloads, and it still knows."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            close, adj = self._panel(seam_at=300)
            store = self._store(tmp, close, adj)
            v = store.verification
            self.assertEqual(v['status'], 'disagrees')
            self.assertEqual(len(v['violations']), 1)
            hit = v['violations'][0]
            self.assertEqual(hit['ticker'], 'ZZZ')
            self.assertEqual(hit['date'], str(close.index[300].date()),
                             'the guard must name the seam, not merely report one exists')
            self.assertAlmostEqual(hit['step'], 1.0 / self.BUMP - 1.0, places=9,
                                   msg='the step is exactly the stale inflation undone')
            text = '\n'.join(store.verification_lines())
            self.assertIn('DATA CHECK FAILED', text)
            self.assertIn('ZZZ', text)

    def test_a_single_vintage_passes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            close, adj = self._panel()
            store = self._store(tmp, close, adj)
            self.assertEqual(store.verification['status'], 'ok')
            self.assertEqual(store.verification['violations'], [])
            self.assertEqual(store.verification_lines(), [],
                             'a clean panel must print nothing at all')

    def test_a_degenerate_pair_reports_not_applicable_rather_than_ok(self):
        """`close == adj_close` makes f identically 1, so the check can see NOTHING.

        Reporting that as 'ok' would be an 'ok' meaning 'I checked nothing' — the fail-open
        shape that let a cp1252 decode error report a modified worktree as clean. Every
        frozen fixture in this suite is built this way, so this is the common case, not an
        edge case.
        """
        idx = pd.bdate_range('2024-01-02', periods=200)
        frame = pd.DataFrame({'ZZZ': 100.0 + 0.01 * np.arange(len(idx))}, index=idx)
        store = PriceStore.from_adjusted(frame, frame.copy())
        self.assertEqual(store.verification['status'], 'not_applicable')
        self.assertIn('close IS adj_close', store.verification['reason'])
        self.assertFalse(store.has_raw_close)

    def test_a_constructed_span_junction_is_not_mistaken_for_a_seam(self):
        """A splice joins two funds on two price scales, so f steps at the junction.

        Measured on the real store, those junctions step -0.126 (VWO), -0.232 (BIL) and
        -0.018 (BND) — an order of magnitude past the tolerance. The test asserts BOTH that
        the guard stays quiet AND that the raw step is real, so the exclusion cannot be
        deleted and pass by accident.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            idx = pd.bdate_range('2024-01-02', periods=400)
            n = len(idx)
            # `_splice_donors` takes k from adj_close and applies it to ALL THREE fields,
            # so the spliced head carries the DONOR's adjustment factor unchanged. The
            # junction therefore steps by (recipient factor / donor factor) - 1, and a
            # DOWNWARD step needs an old donor with little accumulated adjustment meeting a
            # younger recipient with more. That is the real shape: on the working store the
            # junctions measured -0.126 (VWO), -0.232 (BIL) and -0.018 (BND).
            donor_close = pd.Series(50.0 + 0.02 * np.arange(n), index=idx)
            donor_adj = donor_close * 0.99
            recip_close = pd.Series(np.nan, index=idx)
            recip_adj = pd.Series(np.nan, index=idx)
            recip_close.iloc[200:] = donor_close.iloc[200:].values * 2.5
            recip_adj.iloc[200:] = recip_close.iloc[200:].values * 0.90
            for field, dser, rser in (('close', donor_close, recip_close),
                                      ('open', donor_close, recip_close),
                                      ('adj_close', donor_adj, recip_adj)):
                pd.DataFrame({'EEM': dser, 'VWO': rser}).to_csv(
                    os.path.join(tmp, 'daily_{}.csv'.format(field)))
            store = PriceStore(['VWO', 'EEM'], start='2024-01-01', cache_dir=tmp,
                               download=False, refresh_hours=0.0)

            self.assertIn('VWO', store.constructed, 'the fixture must actually splice')
            self.assertEqual(store.verification['status'], 'ok',
                             'a splice junction is a level change, not a restatement: '
                             '{}'.format(store.verification['violations']))

            # ... and the junction really would have fired without the exclusion.
            adj, raw = store._frames['adj_close'], store._frames['close']
            f = (adj['VWO'] / raw['VWO']).dropna()
            worst = float((f / f.shift(1) - 1.0).dropna().min())
            self.assertLess(worst, -0.05,
                            'the fixture no longer reproduces a junction step, so this '
                            'test would pass even with the exclusion removed')

    def test_the_verdict_and_the_detector_settings_travel_in_provenance(self):
        """A manifest pinned WHICH data was used and never whether anything checked it,
        nor which detector version judged it. Reports from before and after the
        2026-09-01 fix were indistinguishable."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            close, adj = self._panel(seam_at=300)
            prov = self._store(tmp, close, adj).provenance()
            self.assertEqual(prov['verification']['status'], 'disagrees')
            self.assertEqual(prov['readjusted'], [])
            self.assertIn('refresh_skipped', prov)
            self.assertEqual(prov['adjustment_policy']['adjustment_step_tolerance'],
                             ADJUSTMENT_STEP_TOLERANCE)
            self.assertEqual(prov['adjustment_policy']['detect_window_days'],
                             DETECT_WINDOW_DAYS)

    def test_the_newest_bars_are_excluded_so_a_pending_dividend_is_not_an_alarm(self):
        """Yahoo can publish an adjusted close for an announced-but-unapplied dividend on
        the newest bar. That is settlement noise, not a spliced history."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            close, adj = self._panel()
            adj.iloc[-1] *= 1.02                      # today's factor jumps
            store = self._store(tmp, close, adj)
            self.assertEqual(store.verification['status'], 'ok')


class TestTheManifestRecordsWhatTheLedgerDoesNotDo(unittest.TestCase):
    """A saved artefact must carry the caveats that qualify its own numbers.

    Two of them are not in the metrics and cannot be recovered from them later: that the
    ledger never tested a maintenance requirement, and whether the risk-free rate was
    traded or accrued from a published yield. A levered CAGR with neither recorded reads
    as a result when it is an upper bound.
    """

    def _manifest(self, leverage):
        config = {'EXECUTION_CONVENTION': 'next_open', 'COST_PCT_PER_SIDE': 0.10,
                  'CASH_TICKER': 'BIL', 'COVERAGE_POLICY': 'strict',
                  'LEVERAGE_FACTOR': leverage, 'MARGIN_FOLLOWS_SIGNAL': True,
                  'START_DATE': '2001-03-31',
                  'BROKER_ACCOUNTS': [{'name': 'CELI', 'balance': 12345.67}]}
        metrics = [{'name': 'SYNTH', 'first_return': None, 'last_return': None,
                    'n_periods': 0, 'rf_annual': 0.019,
                    'rf_desc': 'CONSTRUCTED: 87 of 302 months accrued from ^IRX'}]
        return build_manifest(config, None, metrics)

    def test_the_absent_maintenance_test_is_recorded_even_when_leverage_is_off(self):
        """False at 1.0x too: it is a property of run_ledger, not of the setting. A reader
        must not have to infer it from the leverage field being greater than one."""
        for lev in (1.0, 2.0):
            with self.subTest(leverage=lev):
                self.assertIs(self._manifest(lev)['execution']['maintenance_margin_monitored'],
                              False)

    def test_the_risk_free_provenance_travels_with_the_ratios_it_qualifies(self):
        entry = self._manifest(1.0)['measured']['per_strategy'][0]
        self.assertIn('CONSTRUCTED', entry['rf_desc'] or '',
                      'a Sharpe against an accrued rate must say so where it is stored')

    def test_a_dirty_worktree_is_never_reported_as_clean(self):
        """Found 2026-08-29. `_git` ran with `text=True` and no explicit encoding, so on
        Windows it decoded git's output as cp1252. One accented character anywhere in the
        diff raised UnicodeDecodeError inside subprocess's reader thread, the broad `except`
        returned None, and `git_state` reported a MODIFIED tree as `dirty: False` with no
        diff hash — silently, in the permissive direction, in the provenance layer that
        stamps every saved report.

        Built on a throwaway repository so it asserts the behaviour rather than the state of
        whatever tree the suite happens to run in.
        """
        import subprocess
        import tempfile
        from common.manifest import git_state

        if subprocess.run(('git', '--version'), capture_output=True).returncode != 0:
            self.skipTest('git not available')

        with tempfile.TemporaryDirectory() as tmp:
            def git(*args):
                subprocess.run(('git',) + args, cwd=tmp, capture_output=True, check=True)

            git('init', '-q')
            git('config', 'user.email', 'test@example.invalid')
            git('config', 'user.name', 'Test')
            target = os.path.join(tmp, 'note.md')
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('plain ascii\n')
            git('add', '-A')
            git('commit', '-qm', 'initial')

            self.assertFalse(git_state(tmp)['dirty'], 'an untouched tree is clean')

            # The choice of character is the whole test, so it is written as an escape and
            # explained. cp1252 leaves 0x81/0x8D/0x8F/0x90/0x9D UNDEFINED; every other byte
            # decodes to something. U+FE0F (the emoji variation selector, the invisible half
            # of "⚠️") encodes to EF B8 8F, and that 0x8F is what raises. A plain accented
            # letter or an em dash would NOT reproduce the bug — cp1252 decodes those to
            # mojibake without complaint — so do not "simplify" this string: it would leave
            # the test passing against the defect, which is how it was first written.
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('modified ⚠️ — the diff now carries a byte cp1252 '
                         'cannot decode\n')

            state = git_state(tmp)
            self.assertTrue(state['dirty'],
                            'a modified worktree must report dirty even when the diff is '
                            'not ASCII — "same commit" must never silently mean "same code"')
            self.assertIsNotNone(state['diff_sha256'],
                                 'a dirty tree must carry a diff hash to distinguish it '
                                 'from another dirty tree at the same commit')

    def test_no_balance_reaches_the_manifest(self):
        """The manifest is a public artefact and user_config.json is gitignored. This is the
        seam where a real balance could cross into the repository."""
        import json
        blob = json.dumps(self._manifest(2.0), default=str)
        self.assertNotIn('12345.67', blob)
        self.assertNotIn('CELI', blob)
        self.assertEqual(self._manifest(2.0)['config']['BROKER_ACCOUNTS'], '<redacted>')
