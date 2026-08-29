"""Live-price sizing tests (network-free — the quote fetcher is stubbed).

The monthly SIGNAL decides what to hold; share quantities are sized at the latest
market quote so orders match what the broker charges. These tests lock in:
  - live quotes override the month-end close in the share math,
  - a ticker missing from the quotes falls back to month-end + warning,
  - a total fetch failure falls back everywhere, loudly, and still sizes orders.

Scenario: one $10,000 account, SPY targeted at 50% ($5,000).
Month-end close $300 → 16 shares ($4,800). Live quote $250 → 20 shares ($5,000).
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import main

SIGNAL_DATE = pd.Timestamp('2026-06-30')


def _fixtures():
    prices = pd.DataFrame({'SPY': [300.0]}, index=[SIGNAL_DATE])
    alloc = pd.DataFrame({'SPY': [0.5]}, index=[SIGNAL_DATE])
    s_w = pd.DataFrame({'SPY': [1.0]}, index=[SIGNAL_DATE])
    accounts = [{'account_name': 'A', 'account_balance': 10000.0,
                 'initial_balance': 10000.0, 'account_priority': 1}]
    config = {'SAFETY_MARGIN_PCT': 0.0, 'MINIMUM_TRADE_PCT': 0.5,
              'FLEXIBILITY_BAND_PCT': 0.0, 'FLUSH_ROUND_UP_BAND_PCT': 0.0,
              'FRACTIONAL_SHARES': False, 'SHARE_LOT_SIZE': 1}
    # A real BaseStrategy, not a SimpleNamespace: since 2026-07-30 `size_positions`
    # resolves sleeves through the shape-checked `_sleeves()` (REPORT-001), and a stub
    # that bypasses the checker would be testing a path production cannot take.
    class _Stub(main.BaseStrategy):
        def __init__(self):
            super().__init__('TEST')
            self.offensive, self.defensive, self.canary = ['SPY'], [], []

        def sleeves(self):
            return {'offensive': {'SPY'}, 'defensive': set(), 'canary': []}

        def generate_allocations(self, prices, scores, r12, r3):
            return pd.DataFrame({'SPY': [1.0]}, index=[SIGNAL_DATE])

    return prices, alloc, s_w, accounts, config, _Stub()


class TestLivePriceSizing(unittest.TestCase):
    def test_live_quote_overrides_month_end(self):
        prices, alloc, s_w, accounts, config, strat = _fixtures()
        sizing = main.size_positions(alloc, prices, SIGNAL_DATE, accounts, config, strat, s_w,
                                     live_prices=pd.Series({'SPY': 250.0}))
        order = sizing['orders'][0]
        self.assertEqual(order['shares'], 20.0)          # 5000 / 250
        self.assertEqual(order['price'], 250.0)
        self.assertEqual(sizing['warnings'], [])

    def test_missing_ticker_falls_back_with_warning(self):
        prices, alloc, s_w, accounts, config, strat = _fixtures()
        sizing = main.size_positions(alloc, prices, SIGNAL_DATE, accounts, config, strat, s_w,
                                     live_prices=pd.Series({'QQQ': 500.0}))  # SPY absent
        order = sizing['orders'][0]
        self.assertEqual(order['price'], 300.0)          # month-end close
        self.assertEqual(order['shares'], 16.0)
        self.assertTrue(any('No live quote for SPY' in w for w in sizing['warnings']))


class TestComputeLiveSignalsPricing(unittest.TestCase):
    def _run(self, fetcher):
        prices, _, s_w, accounts, config, strat = _fixtures()
        strat.is_active = True
        strat.score_type = 'unweighted'
        # Fully invested. The 50%-of-account scenario in `_fixtures` exercises the SIZING
        # arithmetic directly; a row reaching compute_live_signals must sum to 1, which the
        # live path now enforces with the same invariant the ledger applies per rebalance.
        strat.generate_allocations = lambda *a, **k: pd.DataFrame({'SPY': [1.0]},
                                                                  index=[SIGNAL_DATE])
        config.update({'CURRENT_EXECUTION_DATE': '2026-07-01', 'STRATEGIES_TO_DISPLAY': []})
        with patch.object(main, 'get_live_prices', side_effect=fetcher):
            return main.compute_live_signals(prices, s_w, s_w, [strat], config, accounts)

    def test_live_mode_used_when_fetch_succeeds(self):
        signal_date, results = self._run(lambda t: (pd.Series({'SPY': 250.0}), pd.Timestamp('2026-07-01')))
        self.assertEqual(signal_date, SIGNAL_DATE)
        res = results[0]
        self.assertEqual(res['pricing']['mode'], 'live')
        self.assertEqual(res['sizing']['orders'][0]['price'], 250.0)

    def test_fetch_failure_falls_back_loudly(self):
        def boom(t):
            raise ConnectionError('offline')
        _, results = self._run(boom)
        res = results[0]
        self.assertEqual(res['pricing']['mode'], 'month-end')
        self.assertEqual(res['sizing']['orders'][0]['price'], 300.0)   # still sizes orders
        self.assertTrue(any('Live quote fetch failed' in w for w in res['sizing']['warnings']))




class TestLivePathEnforcesTheBacktestGuards(unittest.TestCase):
    """DESIGN-001 — until 2026-07-29 the guards ran only where no money moves.

    `run_backtest` called `assert_unlevered_defensive`, `coverage_report` and
    `validate_targets`; `compute_live_signals` called none of them and went straight from
    `generate_allocations` to `size_positions`, which turns weights into whole-share orders
    against real balances. Latently satisfied is not enforced: a strategy added through
    user_config.json's STRATEGIES list arrived unchecked.
    """

    @staticmethod
    def _panel():
        idx = pd.date_range('2024-01-31', periods=16, freq='ME')
        prices = pd.DataFrame({'SPY': 100.0, 'BIL': 91.0, 'UST': 50.0, 'IEF': 95.0},
                              index=idx)
        scores = pd.DataFrame(0.05, index=idx, columns=prices.columns)
        return prices, scores

    @staticmethod
    def _config():
        return {'CURRENT_EXECUTION_DATE': '2025-05-15', 'STRATEGIES_TO_DISPLAY': [],
                'SAFETY_MARGIN_PCT': 0.0, 'MINIMUM_TRADE_PCT': 0.5,
                'FLEXIBILITY_BAND_PCT': 0.0, 'FLUSH_ROUND_UP_BAND_PCT': 0.0,
                'FRACTIONAL_SHARES': False, 'SHARE_LOT_SIZE': 1}

    def _run(self, strat):
        prices, scores = self._panel()
        accounts = [{'account_name': 'A', 'account_balance': 10000.0,
                     'initial_balance': 10000.0, 'account_priority': 1}]
        return main.compute_live_signals(prices, scores, scores, [strat], self._config(),
                                         accounts)

    def test_a_levered_defensive_sleeve_is_refused_before_any_order_is_sized(self):
        """UST is a 2x treasury ETF. A defensive sleeve holding it DOUBLES the risk it
        exists to cut — the reason DAA1_G12 was deleted. The backtest has refused this
        since 2026-07-28; the live path did not."""
        class LeveredDefence(main.BaseStrategy):
            def __init__(self):
                super().__init__('BAD_DEFENCE')
                self.offensive, self.defensive = ['SPY'], ['UST']

            def sleeves(self):
                return {'offensive': {'SPY'}, 'defensive': {'UST'}, 'canary': []}

            def generate_allocations(self, prices, scores, r12, r3):
                return pd.DataFrame({'SPY': 1.0}, index=prices.index)

        _, results = self._run(LeveredDefence())
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0]['error'])
        self.assertIn('UST', results[0]['error'])
        self.assertNotIn('sizing', results[0], 'nothing may be sized once a guard fires')

    def test_a_row_that_does_not_sum_to_one_is_refused(self):
        """The invariant `validate_targets` enforces per rebalance, applied to the one row
        the live path acts on. 80% invested is not an 80% portfolio — it is a portfolio
        whose remaining fifth nobody has decided about."""
        class ShortRow(main.BaseStrategy):
            def __init__(self):
                super().__init__('SHORT_ROW')
                self.offensive, self.defensive = ['SPY'], ['BIL']

            def sleeves(self):
                return {'offensive': {'SPY'}, 'defensive': {'BIL'}, 'canary': []}

            def generate_allocations(self, prices, scores, r12, r3):
                return pd.DataFrame({'SPY': 0.8}, index=prices.index)

        _, results = self._run(ShortRow())
        self.assertIn('weights sum to 0.800000', results[0]['error'])

    def test_a_well_formed_strategy_still_sizes_normally(self):
        """The guards must refuse the two cases above and nothing else."""
        class Fine(main.BaseStrategy):
            def __init__(self):
                super().__init__('FINE')
                self.offensive, self.defensive = ['SPY'], ['BIL']

            def sleeves(self):
                return {'offensive': {'SPY'}, 'defensive': {'BIL'}, 'canary': []}

            def generate_allocations(self, prices, scores, r12, r3):
                return pd.DataFrame({'SPY': 1.0}, index=prices.index)

        with patch('main.get_live_prices', side_effect=RuntimeError('offline')):
            _, results = self._run(Fine())
        self.assertIsNone(results[0]['error'])
        self.assertEqual(results[0]['sizing']['orders'][0]['shares'], 100.0)


class TestSleeveLabellingForWraps(unittest.TestCase):
    """The live offensive/defensive split must resolve for LETF images (REPORT-001).

    `size_positions` read `strat.offensive` / `strat.defensive` — the 1x SIGNAL tickers —
    while a wrap's orders hold the LETF images (`SSO`, `QLD`, ...). So every leveraged
    entry sized with mode "N/A" and a 0/0 split: the readout vanished for exactly the
    entries where knowing the levered fraction is the point. The attribute-sniffing class
    of defect, surviving on the live path two days after being "fixed at the root".
    """

    def _size(self, strat, held_ticker, canary_score):
        cols = sorted(strat._sleeves()[0] | strat._sleeves()[1] | set(strat._sleeves()[2])
                      | {held_ticker})
        prices = pd.DataFrame({t: [100.0] for t in cols}, index=[SIGNAL_DATE])
        alloc = pd.DataFrame({held_ticker: [1.0]}, index=[SIGNAL_DATE])
        s_w = pd.DataFrame({t: [canary_score] for t in cols}, index=[SIGNAL_DATE])
        accounts = [{'account_name': 'A', 'account_balance': 10000.0,
                     'initial_balance': 10000.0, 'account_priority': 1}]
        config = {'SAFETY_MARGIN_PCT': 0.0, 'MINIMUM_TRADE_PCT': 0.5,
                  'FLEXIBILITY_BAND_PCT': 0.0, 'FLUSH_ROUND_UP_BAND_PCT': 0.0,
                  'FRACTIONAL_SHARES': False, 'SHARE_LOT_SIZE': 1}
        return main.size_positions(alloc, prices, SIGNAL_DATE, accounts, config, strat, s_w)

    def test_a_letf_image_is_labelled_offensive(self):
        """SSO is in the wrap's sleeves()['offensive'] and NOT in `strat.offensive` (which
        holds the 1x signal tickers) — the exact case the attribute read got wrong."""
        strat = main.ALL_STRATEGIES['DAA_G3_Leveraged_2X']()
        self.assertNotIn('SSO', strat.offensive, 'fixture premise: SSO is an image only')
        sizing = self._size(strat, 'SSO', canary_score=0.05)
        self.assertEqual(sizing['orders'][0]['mode'], 'Offensive')
        self.assertGreater(sizing['total_off_wt'], 0.0)

    def test_a_defensive_holding_is_labelled_defensive(self):
        strat = main.ALL_STRATEGIES['DAA_G3_Leveraged_2X']()
        sizing = self._size(strat, 'IEF', canary_score=-0.05)
        self.assertEqual(sizing['orders'][0]['mode'], 'Defensive')
        self.assertGreater(sizing['total_def_wt'], 0.0)

    def test_every_leveraged_entry_resolves_a_nonzero_split(self):
        """Parameterised over all fourteen leveraged keys: whatever the wrap holds, the
        split must never be the silent 0/0 that hid the readout."""
        from common.letf_mapper import holds_leveraged_product
        keys = [k for k, cls in main.ALL_STRATEGIES.items()
                if holds_leveraged_product(cls()) and cls().is_active]
        self.assertGreaterEqual(len(keys), 8, f'fixture premise broken: {keys}')
        for key in keys:
            with self.subTest(key=key):
                strat = main.ALL_STRATEGIES[key]()
                defensive, offensive, _ = strat._sleeves()
                # `offensive` the ATTRIBUTE holds 1x signal tickers where it exists at all
                # (DMLeveraged declares none) — which is the point of this test: the
                # resolver must not depend on it. Prefer a pure image; fall back to any
                # offensive holding.
                signal = set(getattr(strat, 'offensive', []) or [])
                image = sorted(offensive - signal - defensive) or sorted(offensive - defensive)
                sizing = self._size(strat, image[0], canary_score=0.05)
                self.assertGreater(sizing['total_off_wt'] + sizing['total_def_wt'], 0.0,
                                   f'{key}: sleeve split is 0/0 for held {image[0]}')


class TestLiveLeverageParity(unittest.TestCase):
    """EXEC-001, decided as option (b): live sizing stays at 1x and SAYS SO.

    The backtest applies LEVERAGE_FACTOR; `size_positions` never reads it. Both true, both
    reasonable — the defect was that nothing on screen said so, so a user calibrating with
    margin_sizing.py and running --live received orders that do not implement the sizing
    they were just given. The warning is inserted in `compute_live_signals`, which both the
    CLI report and the GUI Live tab consume, so the two cannot diverge.
    """

    def _run(self, leverage):
        prices, _, s_w, accounts, config, strat = _fixtures()
        strat.is_active = True
        strat.score_type = 'unweighted'
        config.update({'CURRENT_EXECUTION_DATE': '2026-07-01', 'STRATEGIES_TO_DISPLAY': [],
                       'LEVERAGE_FACTOR': leverage})
        with patch.object(main, 'get_live_prices',
                          side_effect=lambda t: (pd.Series({'SPY': 250.0}),
                                                 pd.Timestamp('2026-07-01'))):
            _, results = main.compute_live_signals(prices, s_w, s_w, [strat], config,
                                                   accounts)
        return results[0]['sizing']

    def test_a_levered_config_warns_and_sizes_at_1x(self):
        sizing = self._run(leverage=1.3)
        self.assertTrue(any('BACKTEST-ONLY' in w for w in sizing['warnings']),
                        f'no leverage warning in {sizing["warnings"]}')
        total = sum(o['value'] for o in sizing['orders'])
        self.assertLessEqual(total, 10000.0 + 1e-6,
                             'live orders must never deploy borrowed money')

    def test_an_unlevered_config_does_not_warn(self):
        sizing = self._run(leverage=1.0)
        self.assertFalse(any('BACKTEST-ONLY' in w for w in sizing['warnings']),
                         sizing['warnings'])


class TestUnfillablePositionsAreReported(unittest.TestCase):
    """An under-filled position must produce a warning naming the asset and the shortfall.

    `size_positions` can skip an account (MINIMUM_TRADE_PCT), exhaust the usable balance
    (SAFETY_MARGIN_PCT), or floor away most of a slice (whole shares) — and until
    2026-07-30 said nothing, leaving the shortfall to be inferred from a smaller number
    three columns over. On the path that spends money, silence reads as "fully deployed".
    """

    def _size(self, config_over, balance=10000.0):
        prices, alloc, s_w, accounts, config, strat = _fixtures()
        accounts[0]['account_balance'] = accounts[0]['initial_balance'] = balance
        config.update(config_over)
        return main.size_positions(alloc, prices, SIGNAL_DATE, accounts, config, strat, s_w)

    def test_a_safety_reserve_shortfall_is_named(self):
        # 50% target of $10k = $5,000; a 60% reserve leaves $4,000 usable -> $1,000 short.
        sizing = self._size({'SAFETY_MARGIN_PCT': 60.0})
        self.assertTrue(any('SPY' in w and 'NOT fully deployed' in w
                            for w in sizing['warnings']),
                        f'shortfall not reported: {sizing["warnings"]}')

    def test_a_normal_fill_does_not_warn(self):
        sizing = self._size({})
        self.assertFalse(any('NOT fully deployed' in w for w in sizing['warnings']),
                         sizing['warnings'])


class TestTrailingMonthIsAvailableOnceTheCalendarClosesIt(unittest.TestCase):
    """A month is finished when it is finished, not when the next one starts trading.

    `_complete_month_ends` withholds a month until a later-month observation proves it
    ended. That is the right rule for a backtest — it makes the result independent of the
    day it was run — and the wrong proxy for live execution on the one day that matters.
    Run on 2026-08-03 before any August bar has been cached and July's month-end is
    withheld too, so orders get sized from the JUNE decision, a full month stale.
    """

    @staticmethod
    def _store(last_bar):
        from common.data_engine import PriceStore
        idx = pd.bdate_range('2025-01-01', last_bar)
        px = pd.DataFrame({'SPY': 100.0}, index=idx)
        return PriceStore.from_frames(px, open_=px, close=px)

    def test_the_strict_rule_still_withholds_the_running_month(self):
        store = self._store('2026-07-31')
        self.assertEqual(str(store.month_end_dates()[-1].date()), '2026-06-30')

    def test_the_calendar_closes_a_month_no_further_session_can_reach(self):
        """2026-07-31 is a Friday and the last business day of July, so once the day is
        past, July cannot receive another session."""
        store = self._store('2026-07-31')
        got = store.month_end_dates(closed_by='2026-08-03')
        self.assertEqual(str(got[-1].date()), '2026-07-31')

    def test_a_bar_that_may_still_be_running_is_never_admitted(self):
        """`closed_by` equal to the last observation means the session could be in
        progress, and Yahoo publishes a partial bar for it. A partial bar must never be
        used as a month-end close."""
        store = self._store('2026-07-31')
        self.assertEqual(str(store.month_end_dates(closed_by='2026-07-31')[-1].date()),
                         '2026-06-30')

    def test_a_month_that_could_still_receive_a_session_is_never_admitted(self):
        """Last bar 2026-07-29 leaves 07-30 and 07-31 as business days: the month is not
        over, whatever date you are standing on."""
        store = self._store('2026-07-29')
        self.assertEqual(str(store.month_end_dates(closed_by='2026-08-05')[-1].date()),
                         '2026-06-30')

    def test_backtests_are_bit_for_bit_unaffected(self):
        store = self._store('2026-07-31')
        self.assertTrue(store.month_end_dates().equals(
            store.month_end_dates(closed_by=None)))

if __name__ == '__main__':
    unittest.main()
