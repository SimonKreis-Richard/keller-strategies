"""
Deterministic, network-free regression tests for the Keller strategies engine.

Run with:   python -m unittest tests.test_audit
       or:  python -m unittest discover -s tests

Unlike the previous smoke version (which printed PASS as long as nothing raised),
these tests assert real invariants:
  - LETF translation: offensive sleeve is leveraged, defensive sleeve stays 1x
    (the Bug-1 regression guard), and the leverage_assets exclusion works.
  - Metrics math (CAGR / drawdown / Sharpe) on known inputs.
  - Benchmarks allocate exactly as specified.
  - Every strategy produces finite, non-negative weights that never sum to > 1,
    on a synthetic price history (no Yahoo Finance / FRED calls).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from common.letf_mapper import LETFMapper
from common.metrics import calculate_metrics
from common.momentum import calc_13612w, calc_13612u


# --------------------------------------------------------------------------- #
# Synthetic data helpers (deterministic, no network)
# --------------------------------------------------------------------------- #
def make_prices(n_months=48, start='2016-01-31'):
    """Deterministic monthly prices for every ticker the engine knows about.

    Gentle upward drift (varied per asset so momentum rankings are distinct) plus a
    small deterministic ripple so per-asset variance is non-zero (keeps covariance /
    correlation based strategies well-defined). No RNG is used.
    """
    from main import TICKERS
    idx = pd.date_range(start, periods=n_months, freq='ME')
    m = np.arange(n_months)
    data = {}
    for k, t in enumerate(TICKERS):
        drift = 0.002 + (k % 7) * 0.0015
        ripple = 1.0 + 0.01 * np.sin((m + k) * 0.5)
        data[t] = 100.0 * (1 + drift) ** m * ripple
    return pd.DataFrame(data, index=idx)


def make_daily_store(n_days=1400, start='2015-06-01', strict_gaps=False):
    """Deterministic DAILY store for every known ticker, for end-to-end engine tests.

    Same shape of series as `make_prices` but on business days, so the store can produce
    real month-end trading dates and an open for the session after each decision. The open
    is deliberately NOT equal to the close (it lags by a fraction of a day's drift), which
    is what makes an execution-convention test able to tell the two apart at all.
    """
    from main import TICKERS
    from common.data_engine import PriceStore
    idx = pd.bdate_range(start, periods=n_days)
    d = np.arange(n_days)
    closes, opens = {}, {}
    for k, t in enumerate(TICKERS):
        drift = (0.002 + (k % 7) * 0.0015) / 21.0          # per business day
        ripple = 1.0 + 0.01 * np.sin((d + k) * 0.5 / 21.0)
        close = 100.0 * (1 + drift) ** d * ripple
        closes[t] = close
        opens[t] = close / (1 + drift) ** 0.5              # opens sit strictly below closes
    return PriceStore.from_adjusted(pd.DataFrame(closes, index=idx),
                                    pd.DataFrame(opens, index=idx),
                                    source='synthetic', strict_gaps=strict_gaps)


# --------------------------------------------------------------------------- #
# LETF mapper — the Bug-1 / leverage_assets regression guard
# --------------------------------------------------------------------------- #
class TestLETFMapper(unittest.TestCase):
    def _prices(self):
        cols = ['SPY', 'IEF', 'TLT', 'GLD', 'UPRO', 'TMF', 'SSO', 'UGL']
        idx = pd.date_range('2020-01-31', periods=2, freq='ME')
        return pd.DataFrame(1.0, index=idx, columns=cols)

    def test_defensive_held_1x_not_dropped_to_cash(self):
        """Bug 1: unmapped (defensive) weight must pass through at 1x, not vanish."""
        prices = self._prices()
        alloc = pd.DataFrame(0.0, index=prices.index, columns=['SPY', 'IEF'])
        alloc['SPY'] = 0.6
        alloc['IEF'] = 0.4

        out = LETFMapper(3).translate(alloc, prices)
        # SPY redirected to UPRO; IEF (no LETF) kept on its own ticker at 1x.
        self.assertAlmostEqual(out['UPRO'].iloc[0], 0.6)
        self.assertAlmostEqual(out['IEF'].iloc[0], 0.4)
        self.assertAlmostEqual(out['SPY'].iloc[0], 0.0)
        # Total invested weight is preserved (no silent leakage to 0%-return cash).
        self.assertAlmostEqual(out.iloc[0].sum(), 1.0)

    def test_leverage_assets_excludes_dual_role_asset(self):
        """TLT in leverage_assets -> TMF; TLT excluded -> held 1x even though mappable."""
        prices = self._prices()
        alloc = pd.DataFrame(0.0, index=prices.index, columns=['SPY', 'TLT'])
        alloc['SPY'] = 0.5
        alloc['TLT'] = 0.5

        # Legacy behaviour (no leverage_assets): TLT is mappable -> TMF.
        legacy = LETFMapper(3).translate(alloc, prices)
        self.assertAlmostEqual(legacy['TMF'].iloc[0], 0.5)
        self.assertAlmostEqual(legacy['TLT'].iloc[0], 0.0)

        # With TLT excluded from leverage: SPY->UPRO, TLT stays 1x on its own ticker.
        excluded = LETFMapper(3).translate(alloc, prices, leverage_assets={'SPY'})
        self.assertAlmostEqual(excluded['UPRO'].iloc[0], 0.5)
        self.assertAlmostEqual(excluded['TLT'].iloc[0], 0.5)
        self.assertAlmostEqual(excluded['TMF'].iloc[0], 0.0)

    def test_2x_mapping(self):
        prices = self._prices()
        alloc = pd.DataFrame(0.0, index=prices.index, columns=['SPY', 'GLD'])
        alloc['SPY'] = 0.5
        alloc['GLD'] = 0.5
        out = LETFMapper(2).translate(alloc, prices)
        self.assertAlmostEqual(out['SSO'].iloc[0], 0.5)  # SPY 2x
        self.assertAlmostEqual(out['UGL'].iloc[0], 0.5)  # GLD 2x

    def test_no_viable_2x_treasury_falls_through_to_1x(self):
        """TLT has no admissible 2x product (UBT is ~$65M, under the liquidity floor).

        `translate` therefore keeps it at 1x. That is correct for a DEFENSIVE holding, and
        exactly why `validate_universe` must forbid TLT in a 2x OFFENSIVE universe: there
        the same fall-through would silently mix 2x and 1x exposure.
        """
        prices = self._prices()
        alloc = pd.DataFrame(0.0, index=prices.index, columns=['SPY', 'TLT'])
        alloc['SPY'] = 0.5
        alloc['TLT'] = 0.5
        out = LETFMapper(2).translate(alloc, prices)
        self.assertAlmostEqual(out['SSO'].iloc[0], 0.5)
        self.assertAlmostEqual(out['TLT'].iloc[0], 0.5)  # held 1x, not dropped


# --------------------------------------------------------------------------- #
# Metrics math
# --------------------------------------------------------------------------- #
class TestMetrics(unittest.TestCase):
    def _r(self, values, start='2016-01-31'):
        return pd.Series(values, index=pd.date_range(start, periods=len(values), freq='ME'))

    def test_zero_returns(self):
        m = calculate_metrics(self._r([0.0] * 24))
        self.assertAlmostEqual(m['cagr'], 0.0)
        self.assertAlmostEqual(m['max_dd'], 0.0)
        self.assertAlmostEqual(m['vol'], 0.0)
        self.assertAlmostEqual(m['sharpe'], 0.0)      # guarded when vol == 0
        self.assertAlmostEqual(m['cum_ret'].iloc[-1], 1.0)
        self.assertAlmostEqual(m['ulcer_index'], 0.0)
        # A never-underwater series has an undefined UPI, not an infinite one.
        self.assertTrue(np.isnan(m['upi']))

    def test_constant_positive_cagr(self):
        m = calculate_metrics(self._r([0.01] * 12))   # 1%/month for 12 months -> 1 year
        self.assertAlmostEqual(m['cagr'], 1.01 ** 12 - 1, places=6)
        self.assertAlmostEqual(m['max_dd'], 0.0)      # monotonic up -> no drawdown
        self.assertAlmostEqual(m['vol'], 0.0)         # constant -> zero volatility

    def test_drawdown_is_negative(self):
        self.assertLess(calculate_metrics(self._r([0.10, 0.10, -0.50, 0.05]))['max_dd'], 0.0)

    def test_risk_free_rate_is_actually_subtracted(self):
        """`rf` was accepted and never passed for the life of the project (audit M3)."""
        r = self._r([0.01] * 24 + [-0.005] * 12)
        self.assertLess(calculate_metrics(r, rf=0.04)['sharpe'],
                        calculate_metrics(r, rf=0.0)['sharpe'])


# --------------------------------------------------------------------------- #
# Strategy invariants on synthetic data
# --------------------------------------------------------------------------- #
class TestStrategyInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No network patching needed: Yahoo is now the only external data source, and these
        # tests run entirely on the synthetic panel from make_prices().
        cls.prices = make_prices()
        cls.scores_w = calc_13612w(cls.prices)
        cls.scores_u = calc_13612u(cls.prices)

    def _alloc(self, strat):
        scores = self.scores_u if strat.score_type == 'unweighted' else self.scores_w
        return strat.generate_allocations(self.prices, scores, None, None)

    def _all_strategies(self):
        """Every REGISTERED strategy, read from the registry itself.

        This used to be a hand-maintained list, which drifted from `main.ALL_STRATEGIES`
        twice: entries were added to the registry without ever being invariant-checked, and
        the 2026-07-28 cut left it importing classes that no longer exist. Reading the
        registry means a new entry is checked the moment it is registered, and a deleted one
        cannot leave a dangling import behind.
        """
        import main
        return [factory() for factory in main.ALL_STRATEGIES.values()]

    def test_weights_are_finite_nonnegative_and_bounded(self):
        for strat in self._all_strategies():
            with self.subTest(strategy=strat.name):
                alloc = self._alloc(strat)
                vals = alloc.values
                self.assertFalse(np.isnan(vals).any(), f"{strat.name}: NaN weights")
                self.assertTrue(np.isfinite(vals).all(), f"{strat.name}: non-finite weights")
                self.assertTrue((vals >= -1e-9).all(), f"{strat.name}: negative weights")
                sums = alloc.iloc[13:].sum(axis=1)
                self.assertTrue((sums <= 1.0 + 1e-6).all(),
                                f"{strat.name}: weights sum > 1 (max {sums.max():.4f})")

    def test_leveraged_wraps_redirect_offensive_originals(self):
        """A leveraged wrap must never hold an offensive original (e.g. SPY) directly —
        it should be redirected to its LETF."""
        from strategies.daa_leveraged import DAA_G4_Leveraged
        from strategies.haa_leveraged import HAA_G3_Leveraged
        from strategies.baa_leveraged import BAA_G3_Leveraged
        for strat in (DAA_G4_Leveraged(2), HAA_G3_Leveraged(3), BAA_G3_Leveraged(3)):
            with self.subTest(strategy=strat.name):
                alloc = self._alloc(strat)
                # SPY is in every offensive universe and always mappable -> never held raw.
                self.assertLess(alloc['SPY'].abs().max(), 1e-9,
                                f"{strat.name}: SPY held directly instead of via LETF")

    def test_every_leveraged_universe_is_ratio_uniform(self):
        """No leveraged WRAP may put offensive weight on a foreign ratio (RULE 1).

        Two things were wrong with the earlier version of this test and both were latent rather
        than failing:

        * it skipped any strategy with no `leverage` attribute, which was every benchmark —
          until `BaseStrategy.leverage = 1.0` landed on 2026-07-29 and the skip stopped
          applying. A `SPY_3X_Benchmark` holding UPRO has `leverage == 1.0` (it maps nothing;
          it names the product directly) and is not a wrap, so the rule does not govern it;
        * the ratio test was `'3' in str(lev)`, a substring match. It reads correctly for
          `3` and `3.0` and would also fire for **1.3** — the margin factor this repository
          uses elsewhere. Numeric now.
        """
        from common.letf_mapper import LETFMapper
        two_x = set(LETFMapper(2).map.values())
        three_x = set(LETFMapper(3).map.values())
        seen_wrap = False
        for strat in self._all_strategies():
            lev = float(getattr(strat, 'leverage', 1.0) or 1.0)
            if lev == 1.0:
                continue                       # not a wrap — it maps nothing
            seen_wrap = True
            foreign = (two_x - three_x) if lev >= 3.0 else (three_x - two_x)
            with self.subTest(strategy=strat.name, leverage=lev):
                alloc = self._alloc(strat)
                held = [t for t in foreign if t in alloc.columns
                        and alloc[t].abs().max() > 1e-9]
                self.assertEqual(held, [],
                                 f"{strat.name}: allocated to off-ratio products {held}")
        self.assertTrue(seen_wrap, 'the registry has leveraged wraps; this test must see them')

    def test_levered_benchmarks_hold_one_ratio_too(self):
        """The wraps' rule stated for the entries the wrap rule does not reach.

        `SPY_2X_Benchmark`, `SPY_3X_Benchmark` and `RiskParity_3X` name their LETFs directly
        rather than mapping onto them, so `leverage` stays 1.0 and the test above skips them.
        The uniformity requirement is the same and is checked here: a benchmark mixing UPRO with
        SSO would have an effective leverage nobody chose.
        """
        import main
        from common.letf_mapper import LETFMapper
        two_x = set(LETFMapper(2).map.values())
        three_x = set(LETFMapper(3).map.values())
        for name in ('SPY_2X_Benchmark', 'SPY_3X_Benchmark', 'RiskParity_3X'):
            strat = main.ALL_STRATEGIES[name]()
            held = set(strat.sleeves()['offensive'])
            with self.subTest(strategy=name):
                self.assertFalse(held & two_x and held & three_x,
                                 f'{name} mixes 2x and 3x products: {sorted(held)}')
                self.assertTrue(held & (two_x | three_x),
                                f'{name} is registered as a levered benchmark but holds no LETF')


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
class TestBenchmarks(unittest.TestCase):
    def test_spy_benchmark_full_spy(self):
        from main import SPY_Benchmark
        prices = make_prices()
        alloc = SPY_Benchmark().generate_allocations(prices, None, None, None)
        self.assertTrue((alloc['SPY'] == 1.0).all())
        # Only SPY carries weight.
        self.assertAlmostEqual(alloc.drop(columns=['SPY']).values.sum(), 0.0)

    def test_golden_butterfly_weights(self):
        from main import Golden_Butterfly
        prices = make_prices()
        alloc = Golden_Butterfly().generate_allocations(prices, None, None, None)
        for asset in ['VTI', 'IJS', 'TLT', 'SHY', 'GLD']:
            self.assertTrue((alloc[asset] == 0.2).all(), f"{asset} should be 20%")


if __name__ == '__main__':
    unittest.main(verbosity=2)
