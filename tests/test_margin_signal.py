"""
Signal-following margin leverage: borrow only against the offensive sleeve.

The leveraged-ETF design de-levers by itself — rotating from UPRO into IEF drops exposure
to 1x. Flat margin does not: the loan stays drawn, so the defensive sleeve is held with
borrowed money and the portfolio rides the drawdown levered. These tests pin the fix.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


def _idx(n, start='2016-01-31'):
    return pd.date_range(start, periods=n, freq='ME')


class _Canaried(BaseStrategy):
    """DAA-shaped: separate canary, a dual-role asset (LQD), a pure defensive one (IEF)."""
    def __init__(self):
        super().__init__('Canaried')
        self.canary = ['VWO']

    def sleeves(self):
        return {'offensive': {'SPY', 'QQQ', 'LQD'}, 'defensive': {'IEF', 'LQD'},
                'canary': list(self.canary)}


class _CashOnly(BaseStrategy):
    """A strategy whose only defensive bucket is a single cash proxy."""
    def __init__(self):
        super().__init__('CashOnly')

    def sleeves(self):
        return {'offensive': {'SPY'}, 'defensive': {'SHY'}, 'canary': []}


class _Overlapping(BaseStrategy):
    """Two static portfolios that differ in ONE slot.

    Only the slot that actually switches is the defensive sleeve; the three assets held in
    both portfolios are structural and stay offensive. The pre-2026-07 engine inferred this
    by diffing two weight dicts, which is exactly the kind of guesswork `sleeves()` removes —
    but the arithmetic it produced was right, so it is pinned here.
    """
    def __init__(self):
        super().__init__('Overlapping')

    def sleeves(self):
        return {'offensive': {'QQQ', 'IWD', 'GLD', 'IEF'}, 'defensive': {'SHY'},
                'canary': []}


class TestSleeveResolution(unittest.TestCase):

    def test_pure_defensive_asset_is_always_defensive(self):
        idx = _idx(2)
        alloc = pd.DataFrame({'SPY': [1.0, 0.0], 'IEF': [0.0, 1.0]}, index=idx)
        scores = pd.DataFrame({'VWO': [1.0, 1.0]}, index=idx)   # canary alive both months
        off = _Canaried().offensive_weight(alloc, scores)
        self.assertAlmostEqual(off.iloc[0], 1.0)   # fully offensive
        self.assertAlmostEqual(off.iloc[1], 0.0)   # IEF is never offensive

    def test_dual_role_asset_follows_the_canary(self):
        idx = _idx(2)
        alloc = pd.DataFrame({'LQD': [1.0, 1.0]}, index=idx)
        scores = pd.DataFrame({'VWO': [1.0, -1.0]}, index=idx)  # alive, then dead
        off = _Canaried().offensive_weight(alloc, scores)
        self.assertAlmostEqual(off.iloc[0], 1.0)   # canary alive -> LQD is an offensive pick
        self.assertAlmostEqual(off.iloc[1], 0.0)   # canary dead  -> LQD is the defensive sleeve

    def test_dual_role_defaults_to_offensive_without_a_canary(self):
        """Conservative: never claim de-escalation the signal cannot prove."""
        strat = _Canaried()
        strat.canary = []
        idx = _idx(1)
        alloc = pd.DataFrame({'LQD': [1.0]}, index=idx)
        self.assertAlmostEqual(strat.offensive_weight(alloc, None).iloc[0], 1.0)

    def test_bare_cash_attribute_is_recognised(self):
        idx = _idx(2)
        alloc = pd.DataFrame({'SPY': [1.0, 0.0], 'SHY': [0.0, 1.0]}, index=idx)
        off = _CashOnly().offensive_weight(alloc, None)
        self.assertAlmostEqual(off.iloc[0], 1.0)
        self.assertAlmostEqual(off.iloc[1], 0.0)

    def test_overlap_only_counts_the_switched_asset(self):
        """IWD/GLD/IEF sit in BOTH portfolios — they are not the defensive switch.
        Only SHY is, so the bear portfolio stays 75% offensive, not 100% defensive."""
        idx = _idx(1)
        alloc = pd.DataFrame({'SHY': [0.25], 'IWD': [0.25], 'GLD': [0.25], 'IEF': [0.25]}, index=idx)
        self.assertAlmostEqual(_Overlapping().offensive_weight(alloc, None).iloc[0], 0.75)

    def test_partial_defensive_weight(self):
        """DAA's graded cash: half offensive, half defensive -> 0.5."""
        idx = _idx(1)
        alloc = pd.DataFrame({'SPY': [0.5], 'IEF': [0.5]}, index=idx)
        scores = pd.DataFrame({'VWO': [1.0]}, index=idx)
        self.assertAlmostEqual(_Canaried().offensive_weight(alloc, scores).iloc[0], 0.5)

    def test_uninvested_residue_is_not_borrowed_against(self):
        """A 40%-invested month borrows against 40%, not 100%."""
        idx = _idx(1)
        alloc = pd.DataFrame({'SPY': [0.4]}, index=idx)
        scores = pd.DataFrame({'VWO': [1.0]}, index=idx)
        self.assertAlmostEqual(_Canaried().offensive_weight(alloc, scores).iloc[0], 0.4)

    def test_every_registered_strategy_exposes_a_sleeve_split(self):
        """offensive_weight must be finite and in [0,1] for every strategy in the registry."""
        import main
        from tests.test_audit import make_prices
        from common.momentum import calc_13612w, calc_13612u
        prices = make_prices()
        s_w, s_u = calc_13612w(prices), calc_13612u(prices)

        for name, factory in main.ALL_STRATEGIES.items():
            strat = factory()
            scores = s_u if strat.score_type == 'unweighted' else s_w
            with self.subTest(strategy=name):
                alloc = strat.generate_allocations(prices, scores, None, None)
                off = strat.offensive_weight(alloc, scores)
                self.assertTrue(np.isfinite(off.to_numpy()).all(), f"{name}: non-finite")
                self.assertTrue(((off >= -1e-9) & (off <= 1.0 + 1e-9)).all(), f"{name}: out of [0,1]")


class TestBacktestLeverage(unittest.TestCase):
    """End-to-end: the engine must actually de-lever, and charge interest accordingly."""

    @staticmethod
    def _config(follows_signal, leverage, borrow_rate=0.12):
        return {
            'START_DATE': '2016-09-01', 'END_DATE': '2021-01-01',
            'DATA_START_DATE': '2015-06-01', 'CURRENT_EXECUTION_DATE': '2021-01-01',
            'EXECUTION_MODE': False,
            'LEVERAGE_FACTOR': leverage, 'MARGIN_BORROW_RATE': borrow_rate,
            'MARGIN_FOLLOWS_SIGNAL': follows_signal, 'COST_PCT_PER_SIDE': 0.0,
            'LOOKBACK_MONTHS': 13, 'EXECUTION_CONVENTION': 'next_open',
            'CASH_TICKER': 'BIL', 'COVERAGE_POLICY': 'trim', 'RF_ANNUAL_FALLBACK': 0.0,
        }

    def _run(self, follows_signal, leverage=2.0, borrow_rate=0.12, name='DAA_G12'):
        import main
        from tests.test_audit import make_daily_store
        store = make_daily_store()
        config = self._config(follows_signal, leverage, borrow_rate)
        prices, sw, su = main.build_signal_panel(store, config)
        metrics, _ = main.run_backtest(prices, sw, su, [main.ALL_STRATEGIES[name]()],
                                       config, store=store)
        self.assertTrue(metrics, f'{name} produced no metrics')
        return metrics[0]

    def test_signal_following_leverage_never_exceeds_the_cap(self):
        m = self._run(follows_signal=True, leverage=2.0)
        self.assertLessEqual(m['max_lev'], 2.0 + 1e-9)
        self.assertGreaterEqual(m['min_lev'], 1.0 - 1e-9)

    def test_flat_leverage_is_constant(self):
        m = self._run(follows_signal=False, leverage=2.0)
        self.assertAlmostEqual(m['min_lev'], 2.0)
        self.assertAlmostEqual(m['max_lev'], 2.0)

    def test_signal_following_borrows_less_on_average(self):
        """The whole point: average realized leverage must sit below the flat cap."""
        following = self._run(follows_signal=True, leverage=2.0)
        flat = self._run(follows_signal=False, leverage=2.0)
        self.assertLessEqual(following['avg_lev'], flat['avg_lev'] + 1e-9)

    def test_unlevered_run_reports_1x_and_is_unaffected_by_the_flag(self):
        a = self._run(follows_signal=True, leverage=1.0)
        b = self._run(follows_signal=False, leverage=1.0)
        self.assertAlmostEqual(a['avg_lev'], 1.0)
        self.assertAlmostEqual(b['avg_lev'], 1.0)
        self.assertAlmostEqual(a['cagr'], b['cagr'], places=12)

    def test_benchmarks_stay_at_1x(self):
        m = self._run(follows_signal=True, leverage=2.0, name='SPY_Benchmark')
        self.assertAlmostEqual(m['avg_lev'], 1.0)
        self.assertAlmostEqual(m['max_lev'], 1.0)


class TestMissingMaskIsLoud(unittest.TestCase):
    """`leverage_follows_signal=True` with no defensive mask must warn, not silently flatten.

    `run_ledger`'s signal-following branch requires `sleeves={'defensive_mask': ...}`; when
    the mask is absent it applies FLAT leverage — the configured behaviour's opposite — and
    until 2026-07-30 said nothing. `run_backtest` always supplies the mask, so production
    was never wrong; the exposure is every research caller and future call site, for whom
    the A/B this flag exists to make would silently return no difference (2026-07-30 audit,
    LOGIC-002). A warning and not a raise: a caller deliberately passing no sleeves is a
    legitimate use, the same treatment `cash_ticker=None` already gets.
    """

    def _run(self, sleeves):
        from common.data_engine import PriceStore
        from common.ledger import ExecutionConfig, run_ledger
        idx = pd.bdate_range('2020-01-01', periods=200)
        daily = pd.DataFrame({'SPY': 100.0, 'BIL': 91.0}, index=idx)   # dead-flat market
        store = PriceStore.from_adjusted(daily, daily)
        months = daily.resample('ME').last().index[:-1]
        targets = pd.DataFrame(1.0, index=months, columns=['SPY'])
        cfg = ExecutionConfig(convention='next_close', cost_bps_per_side=0.0,
                              cash_ticker='BIL', leverage=2.0, borrow_rate=0.10,
                              leverage_follows_signal=True)
        return run_ledger(targets, store, cfg, sleeves=sleeves)

    def test_a_missing_mask_warns_and_applies_flat_leverage(self):
        res = self._run(sleeves=None)
        self.assertTrue(any('FLAT leverage' in w for w in res.warnings),
                        f'no FLAT-leverage warning in {res.warnings}')
        # And the leverage really is flat — the warning must describe what happened.
        self.assertAlmostEqual(float(np.mean(res.effective_leverage)), 2.0, places=9)

    def test_a_present_mask_does_not_warn(self):
        idx = pd.bdate_range('2020-01-01', periods=200)
        months = pd.DataFrame({'SPY': 100.0}, index=idx).resample('ME').last().index[:-1]
        mask = pd.DataFrame(False, index=months, columns=['SPY'])
        res = self._run(sleeves={'defensive_mask': mask})
        self.assertFalse(any('FLAT leverage' in w for w in res.warnings), res.warnings)


class TestProductionConfigEnforcesStrictGaps(unittest.TestCase):
    """The DataGapError guard must be ON in every production run.

    ARCHITECTURE.md described it as enforced; until 2026-07-30 `STRICT_GAPS` appeared in no
    production config dict, so every CLI and GUI run built the store with the guard off
    (2026-07-30 audit, DATA-001). Inert on the committed cache — provenance() reports
    long_gaps: [] — but the stated first line of defence has to actually be deployed.
    """

    def test_the_dashboard_default_is_true(self):
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            import main
        self.assertTrue(main.STRICT_GAPS)

    def test_load_store_defaults_to_strict_when_the_key_is_absent(self):
        """A config dict that never thought about gaps gets the guard, not the exemption —
        'unknown must mean check', the same direction the refresh stamp follows."""
        import inspect, io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            import main
        src = inspect.getsource(main.load_store)
        self.assertIn("config.get('STRICT_GAPS', True)", src,
                      'load_store must default STRICT_GAPS to True for absent keys')

    def test_a_long_interior_gap_refuses_under_the_production_default(self):
        from common.data_engine import PriceStore, DataGapError
        idx = pd.bdate_range('2023-01-02', periods=120)
        spy = pd.Series(100.0 + np.arange(120.0), index=idx)
        spy.iloc[40:50] = np.nan                      # a 10-trading-day interior gap
        frame = pd.DataFrame({'SPY': spy, 'BIL': 91.0})
        with self.assertRaises(DataGapError):
            PriceStore.from_frames(frame, strict_gaps=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
