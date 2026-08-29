"""
Tests with an anchor OUTSIDE the code.

The 2026-07-28 audit's sharpest structural finding was not any single bug: it was that all
55 tests in this repo asserted `f(x) == f(x)`. Every one of them passed against an engine
with four critical defects, because each line was locally consistent with every other line.

A test only has audit value if what it compares against was not produced by the thing under
test. The anchors here are: arithmetic done by hand in the test file, a published results
table from the source paper, and a mechanism that must produce a known-zero effect in a case
where it cannot possibly apply.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from common.data_engine import PriceStore
from common.ledger import ExecutionConfig, run_ledger
from common.metrics import calculate_metrics, build_rf_series
from common.momentum import calc_13612u, calc_13612w

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def frozen_store():
    return PriceStore.from_daily_fixture(
        os.path.join(FIXTURES, 'frozen_daily_adj_close.csv'),
        os.path.join(FIXTURES, 'frozen_daily_adj_open.csv'))


def _panel(growth, n_months=14, start='2015-01-31'):
    """Prices with an EXACTLY known momentum score: P_i = 100 * (1+g)^i.

    Then r_N = (1+g)^N - 1 for every N, so 13612U = ((1+g) + (1+g)^3 + (1+g)^6 +
    (1+g)^12 - 4) / 4 — a closed form the test computes itself, without calling the
    engine's momentum code.
    """
    idx = pd.date_range(start, periods=n_months, freq='ME')
    return pd.DataFrame({t: 100.0 * (1.0 + g) ** np.arange(n_months)
                         for t, g in growth.items()}, index=idx)


def _expected_13612u(g):
    return ((1 + g) + (1 + g) ** 3 + (1 + g) ** 6 + (1 + g) ** 12 - 4) / 4.0


class TestHandComputedSignal(unittest.TestCase):
    """T7.6 — the only test in the suite that checks a strategy against arithmetic done
    outside it. Everything else compares the code to other code."""

    #: HAA-8's universe plus its TIP canary and its two defensive candidates.
    BASE = {'SPY': 0.05, 'IWM': 0.04, 'VEA': 0.03, 'VWO': -0.02, 'VNQ': -0.03,
            'DBC': -0.04, 'IEF': 0.005, 'TLT': 0.002, 'TIP': 0.01, 'BIL': 0.0001}

    def _run(self, growth):
        from strategies.haa import HAA_Balanced
        prices = _panel(growth)
        scores = calc_13612u(prices)
        alloc = HAA_Balanced().generate_allocations(prices, scores, None, None)
        return prices, scores, alloc.iloc[-1]

    def test_momentum_score_matches_the_closed_form(self):
        prices, scores, _ = self._run(self.BASE)
        for ticker, g in self.BASE.items():
            with self.subTest(ticker=ticker):
                self.assertAlmostEqual(float(scores[ticker].iloc[-1]),
                                       _expected_13612u(g), places=12)

    def test_top_four_are_selected_equal_weight(self):
        """Canary alive; the four highest-scoring offensive assets, 25% each."""
        _, _, held = self._run(self.BASE)
        held = held[held > 0]
        self.assertEqual(sorted(held.index), ['IEF', 'IWM', 'SPY', 'VEA'])
        for t in held.index:
            self.assertAlmostEqual(float(held[t]), 0.25, places=12)

    def test_a_negative_pick_is_replaced_by_the_best_defensive_asset(self):
        """HAA's rule: a selected asset with non-positive momentum is swapped for the
        better of BIL / IEF — not dropped, and not left as cash."""
        g = dict(self.BASE, IEF=-0.01, TLT=-0.001)
        # Ranking is now SPY > IWM > VEA > TLT, and TLT scores below zero.
        self.assertLess(_expected_13612u(g['TLT']), 0.0)
        self.assertGreater(_expected_13612u(g['BIL']), _expected_13612u(g['IEF']))
        _, _, held = self._run(g)
        held = held[held > 0]
        self.assertEqual(sorted(held.index), ['BIL', 'IWM', 'SPY', 'VEA'])
        self.assertAlmostEqual(float(held['BIL']), 0.25, places=12)

    def test_dead_canary_goes_fully_defensive(self):
        """TIP below zero => 100% in the better defensive asset, here IEF."""
        _, _, held = self._run(dict(self.BASE, TIP=-0.01))
        held = held[held > 0]
        self.assertEqual(list(held.index), ['IEF'])
        self.assertAlmostEqual(float(held['IEF']), 1.0, places=12)

    def test_weights_always_sum_to_one_after_warmup(self):
        for label, g in (('base', self.BASE),
                         ('substitution', dict(self.BASE, IEF=-0.01, TLT=-0.001)),
                         ('risk-off', dict(self.BASE, TIP=-0.01))):
            with self.subTest(case=label):
                prices = _panel(g)
                from strategies.haa import HAA_Balanced
                alloc = HAA_Balanced().generate_allocations(prices, calc_13612u(prices),
                                                            None, None)
                sums = alloc.iloc[12:].sum(axis=1)
                self.assertTrue(np.allclose(sums, 1.0), f'{label}: {sums.tolist()}')


class TestExecutionConventionIsMaterial(unittest.TestCase):
    """T7.3 — guard against silently reverting to same-bar execution.

    Filling at the close that PRODUCED the signal is impossible: the signal is not known
    until that close has printed. The engine must therefore be able to tell the two apart,
    and the difference must come from TRADING — which is what the benchmark control below
    establishes.

    Honest note on magnitude. The 2026-07-28 audit originally put this at +2.09 pp of CAGR
    on HAA-G8. That figure counted the WHOLE close-to-next-open leg the rotation sits through
    (~1.97 pp/yr), but open-to-open pricing does not forfeit that leg — it earns it on the
    previous month's basket instead. Only the part attributable to positions that actually
    CHANGED is look-ahead. Measured apples-to-apples through this ledger, same costs, only
    the fill price differing, it is **+0.37 pp** on HAA-G8 over 2013-2024 (HAA_G12 +0.32,
    DAA_G12 +0.34, VAA_G12 -0.10 — the sign is not guaranteed; the leveraged wrap measured
    at the time, DAA_G3_Leveraged_2X, showed +0.87 before its deletion under RULE 5).
    Smaller than first reported, still a return nobody could have traded.
    """

    @classmethod
    def setUpClass(cls):
        cls.store = frozen_store()
        cls.month_ends = cls.store.month_end_dates()
        cls.prices = cls.store.monthly_adj_close(cls.month_ends)
        cls.decisions = cls.month_ends[(cls.month_ends >= '2013-01-01')
                                       & (cls.month_ends <= '2024-12-31')]
        cls.rf, _ = build_rf_series(cls.store, cls.decisions, 'BIL')

    def _cagr(self, name, convention):
        import main
        strat = main.ALL_STRATEGIES[name]()
        scores = (calc_13612u(self.prices) if strat.score_type == 'unweighted'
                  else calc_13612w(self.prices))
        targets = strat.generate_allocations(self.prices, scores, None, None) \
                       .reindex(self.decisions)
        led = run_ledger(targets, self.store,
                         ExecutionConfig(convention=convention, cost_bps_per_side=10.0,
                                         cash_ticker='BIL'), label=name)
        return calculate_metrics(led.returns, rf=self.rf)['cagr']

    def test_same_bar_execution_flatters_a_trading_strategy(self):
        gap = self._cagr('HAA_G8_Balanced', 'signal_close') - \
              self._cagr('HAA_G8_Balanced', 'next_open')
        self.assertGreater(gap, 0.0020, f'gap collapsed to {gap:.4%} — has the ledger '
                                        f'quietly reverted to same-bar fills?')

    def test_a_strategy_that_never_trades_is_indifferent_to_the_convention(self):
        """The control. SPY_Benchmark holds one asset forever, so there is no rotation to
        capture a free leg on. If this ever grew, the gap above would be an artefact of the
        price series rather than of execution."""
        gap = abs(self._cagr('SPY_Benchmark', 'signal_close') -
                  self._cagr('SPY_Benchmark', 'next_open'))
        self.assertLess(gap, 0.0005, f'a never-trading benchmark moved {gap:.4%}')

    def test_next_open_is_unavailable_without_intraday_data(self):
        """Refuse, rather than silently falling back to the close — the fallback is the
        very defect being removed."""
        monthly = PriceStore.from_monthly_csv(
            os.path.join(FIXTURES, 'frozen_prices_2026-06-08.csv'))
        w = pd.DataFrame(1.0, index=monthly.month_end_dates()[-5:], columns=['SPY'])
        with self.assertRaises(ValueError) as ctx:
            run_ledger(w, monthly, ExecutionConfig(convention='next_open'), label='x')
        self.assertIn('has_intraday', str(ctx.exception))


class TestCostConvention(unittest.TestCase):
    """T2.4 — Keller's HAA paper: "we assume a one-way transaction costs of 0.1% (=TC)".

    The pre-audit engine charged `sum|dw|/2 x TC`, which is HALF of one-way. Here the anchor
    is the paper's sentence, expressed as arithmetic the test does itself.
    """

    @classmethod
    def setUpClass(cls):
        cls.store = frozen_store()
        cls.dec = cls.store.month_end_dates()[-14:]

    def _ledger(self, bps, weights):
        return run_ledger(weights, self.store,
                          ExecutionConfig(convention='next_open', cost_bps_per_side=bps,
                                          cash_ticker=None,
                                          charge_terminal_liquidation=False), label='cost')

    def test_a_full_rotation_costs_two_legs(self):
        """A -> B is a sell AND a buy. 0.1% one-way therefore costs 0.2% of equity."""
        w = pd.DataFrame(0.0, index=self.dec, columns=['SPY', 'IEF'])
        w['SPY'] = 1.0
        w.iloc[7:] = 0.0
        w.iloc[7:, w.columns.get_loc('IEF')] = 1.0     # one clean rotation, mid-sample
        led = self._ledger(10.0, w)
        rotation = float(led.cost_paid.iloc[7])
        self.assertAlmostEqual(rotation / float(led.equity.iloc[7]), 0.0020, places=6)

    def test_initial_deployment_is_charged_once(self):
        w = pd.DataFrame(0.0, index=self.dec, columns=['SPY', 'IEF'])
        w['SPY'] = 1.0
        led = self._ledger(10.0, w)
        self.assertAlmostEqual(led.initial_deployment_cost, 0.0010, places=6)
        # ... and holding costs nothing thereafter.
        self.assertAlmostEqual(float(led.cost_paid.iloc[1:].sum()), 0.0, places=9)

    def test_cost_scales_with_the_levered_notional(self):
        """Trades are sized on the LEVERED book, so cost must be too (audit m2/m8)."""
        w = pd.DataFrame(0.0, index=self.dec, columns=['SPY', 'IEF'])
        w['SPY'] = 1.0
        plain = run_ledger(w, self.store, ExecutionConfig(
            convention='next_open', cost_bps_per_side=10.0, cash_ticker=None,
            leverage=1.0, charge_terminal_liquidation=False), label='1x')
        levered = run_ledger(w, self.store, ExecutionConfig(
            convention='next_open', cost_bps_per_side=10.0, cash_ticker=None,
            leverage=2.0, leverage_follows_signal=False,
            charge_terminal_liquidation=False), label='2x')
        self.assertAlmostEqual(levered.initial_deployment_cost,
                               2.0 * plain.initial_deployment_cost, places=9)


class TestCashIsAnAccount(unittest.TestCase):
    """Uninvested weight earns a cash return, not 0%. The pre-audit engine simply lost it."""

    @classmethod
    def setUpClass(cls):
        cls.store = frozen_store()
        cls.dec = cls.store.month_end_dates()[-25:]

    def test_half_invested_book_still_earns_on_the_idle_half(self):
        w = pd.DataFrame(0.0, index=self.dec, columns=['SPY'])
        w['SPY'] = 0.5                      # 50% uninvested every month
        cfg = dict(convention='next_open', cost_bps_per_side=0.0,
                   charge_terminal_liquidation=False, strict_invariants=False)
        with_cash = run_ledger(w, self.store, ExecutionConfig(cash_ticker='BIL', **cfg),
                               label='cash')
        without = run_ledger(w, self.store, ExecutionConfig(cash_ticker=None, **cfg),
                             label='nocash')
        self.assertGreater(float(with_cash.equity.iloc[-1]), float(without.equity.iloc[-1]))
        self.assertAlmostEqual(float(with_cash.cash_weight.mean()), 0.5, places=6)
        self.assertTrue(any('0%' in w for w in without.warnings))


class TestPublishedAnchor(unittest.TestCase):
    """T7.7 — the one comparison against a number this project did not produce.

    Keller's HAA paper (SSRN 4346906) reports, over **Dec-1970 → Dec-2022** with TC = 0.1%:

        HAA-8  (NO=8,  TO=4, ND=2, TD=1, NP=1)   CAGR 15.9%   MaxDD  -9.7%   Sharpe 1.21
        HAA-12 (NO=12, TO=6, ND=2, TD=1, NP=1)   CAGR 15.9%   MaxDD -10.7%   Sharpe 1.19

    Allocate Smartly report ~15.8% / -10.0% / 1.25 for HAA-Balanced on their own data.

    **This test is not a reproduction and must never be described as one.** The engine has
    no pre-2000 history at all, so it cannot see 52 years, two inflation regimes, or the
    1970s commodity bull that does most of the work in the published CAGR. What it CAN do is
    refuse a result whose SHAPE is impossible: a defensive rotation strategy that prints a
    50% drawdown, or a UPI two orders of magnitude away from the published one, is broken
    regardless of the sample.

    Note for anyone tempted to prefer HAA-12 on performance: in Keller's OWN full-sample
    results HAA-8 beats HAA-12 on both max drawdown and Sharpe at identical CAGR.
    """

    @classmethod
    def setUpClass(cls):
        import main
        store = frozen_store()
        month_ends = store.month_end_dates()
        prices = store.monthly_adj_close(month_ends)
        dec = month_ends[(month_ends >= '2013-01-01') & (month_ends <= '2024-12-31')]
        rf, _ = build_rf_series(store, dec, 'BIL')
        strat = main.ALL_STRATEGIES['HAA_G8_Balanced']()
        targets = strat.generate_allocations(prices, calc_13612u(prices), None, None) \
                       .reindex(dec)
        led = run_ledger(targets, store,
                         ExecutionConfig(convention='next_open', cost_bps_per_side=10.0,
                                         cash_ticker='BIL'), label='HAA_G8_Balanced')
        cls.m = calculate_metrics(led.returns, rf=rf)

    def test_drawdown_stays_in_the_published_neighbourhood(self):
        """Published -9.7% over 52 years. A rotation strategy with a TIP canary and a
        defensive sleeve cannot legitimately print a deep equity-like drawdown."""
        self.assertGreater(self.m['max_dd'], -0.25)
        self.assertLess(self.m['max_dd'], -0.005)   # ... nor a suspiciously perfect one

    def test_volatility_is_bond_like_not_equity_like(self):
        """Published vol 9.4%. Ours is a different, calmer sample; the band is wide."""
        self.assertTrue(0.03 < self.m['vol'] < 0.16, f"vol {self.m['vol']:.2%}")

    def test_upi_is_single_digit_and_positive(self):
        """Keller reports UPI 4.88 for HAA-8 (4.50 for HAA-12) over the full sample. On an
        ETF-era window the level will differ; what would signal a broken Ulcer Index is a
        negative value or one orders of magnitude out."""
        self.assertTrue(0.2 < self.m['upi'] < 10.0, f"UPI {self.m['upi']:.2f}")

    def test_ulcer_index_is_between_zero_and_max_drawdown(self):
        """RMS drawdown must sit inside [0, |MaxDD|] by construction — an arithmetic
        identity, so a violation means the wealth curve is malformed."""
        self.assertGreater(self.m['ulcer_index'], 0.0)
        self.assertLess(self.m['ulcer_index'], abs(self.m['max_dd']))


class TestRegistry(unittest.TestCase):
    """T10.5 — the registry cannot silently regrow."""

    EXPECTED = 36
    EXPECTED_BY_FAMILY = {'HAA': 7, 'DAA': 7, 'VAA': 2, 'BAA': 7, 'PAA': 1,
                          'DM': 5, 'GTAA': 1, 'benchmarks': 6}

    def test_registry_size(self):
        import main
        self.assertEqual(len(main.ALL_STRATEGIES), self.EXPECTED,
                         'The registry changed size. That is a decision, not an accident: '
                         'update this count deliberately and record the reasoning, as '
                         'main.ALL_STRATEGIES does for the 64 -> 39 -> 25 -> 22 -> 23 -> 27 '
                         '-> 35 -> 36 moves.')

    def test_registry_composition_by_family(self):
        """The per-family counts, which `EXPECTED_BY_FAMILY` recorded but nothing asserted
        until 2026-07-29. A total that stays the same while two families trade an entry is
        exactly the change `test_registry_size` cannot see."""
        import main
        from collections import Counter
        from common.palette import family_of
        got = Counter()
        for d in main.strategy_roster():
            # Controls count under their OWN family (HAA_G1_Simple is an HAA entry — it exists
            # to be subtracted from HAA), so the family is read from the NAME. Only benchmarks
            # form a bucket of their own, having no family to belong to.
            got['benchmarks' if d['role'] == 'benchmark'
                else family_of(d['name'], 'strategy')] += 1
        self.assertEqual(dict(got), self.EXPECTED_BY_FAMILY)
        self.assertEqual(sum(self.EXPECTED_BY_FAMILY.values()), self.EXPECTED,
                         'the per-family counts must add up to the total')

    def test_only_families_with_an_exogenous_signal_have_leveraged_wraps(self):
        """RULE 4 (common/letf_mapper.py): a wrap may change what is HELD, never what
        decides to DE-RISK.

        VAA and PAA declare no canary on purpose — their protection is a breadth count over
        their own offensive universe. A wrap must restrict that universe to what executes at
        a uniform LETF multiple, which rewrites the protection rule rather than merely
        narrowing the portfolio. Their wraps were deleted 2026-07-29 and must not return.
        This is a structural claim, checkable without looking at a single return.
        """
        import main
        wrapped = {k.split('_')[0] for k in main.ALL_STRATEGIES if 'Leveraged' in k}
        self.assertEqual(wrapped, {'HAA', 'BAA', 'DAA', 'DM'})
        for name, factory in main.ALL_STRATEGIES.items():
            if 'Leveraged' not in name:
                continue
            with self.subTest(strategy=name):
                canary = factory().sleeves().get('canary') or []
                self.assertTrue(canary or name.startswith('DM'),
                                f'{name} has a leveraged wrap but declares no canary, so '
                                f'its de-risking signal is its own offensive universe')

    def test_no_three_x_entry_can_reach_the_selection_statistics(self):
        """REPLACES `test_no_three_x_variant_is_registered` (2026-07-29).

        The old test asserted the registry held no `*_3X` key at all. What it was actually
        protecting was the RANKED TABLE and the best-of-N trial count — no 3x product predates
        2008-11, so no 3x drawdown here has ever seen a bear market, and a 20%-CAGR row with a
        structurally unobservable worst case does not belong among things one might hold.

        `role` protects that directly, and better: 3x entries are now registered and MEASURED,
        which is the point of having them, while `main.py`'s selection machinery filters on
        `role == 'strategy'` so they add no trials. The guard is replaced rather than deleted —
        the property being defended is unchanged, only the mechanism.
        """
        import main
        offenders = [n for n, f in main.ALL_STRATEGIES.items()
                     if n.upper().endswith('_3X')
                     and getattr(f(), 'role', 'strategy') == 'strategy']
        self.assertEqual(offenders, [],
                         "a 3x entry with role='strategy' would be counted as a trial by the "
                         "best-of-N machinery and ranked as a portfolio one might hold; use "
                         "role='exploratory' or role='benchmark'")

    def test_no_levered_entry_may_set_the_shared_ranked_window(self):
        """The predicate `main.run_backtest` actually uses, asserted over the whole registry.

        A levered entry that could set the window would shorten every other row by years, and
        the fidelity label is not enough to stop it — see `main.may_set_ranked_window`.
        """
        import main
        from common.letf_mapper import holds_leveraged_product
        levered = [(n, f()) for n, f in main.ALL_STRATEGIES.items()]
        levered = [(n, s) for n, s in levered if holds_leveraged_product(s)]
        self.assertTrue(levered, 'the registry has levered entries; this test must see them')
        for name, strat in levered:
            with self.subTest(strategy=name, fidelity=getattr(strat, 'fidelity', '?')):
                self.assertFalse(main.may_set_ranked_window(strat),
                                 f'{name} holds a leveraged product and would shorten every '
                                 f'other row; LETFs all launched 2006 or later')

    def test_a_faithful_levered_benchmark_is_still_barred(self):
        """The specific hole the structural check closes. `SPY_3X_Benchmark` is labelled
        `faithful` — correctly, there being no rule to be unfaithful to — so the fidelity filter
        alone would have admitted it as a window setter and dragged 2008 out of the table."""
        import main
        strat = main.ALL_STRATEGIES['SPY_3X_Benchmark']()
        self.assertEqual(strat.fidelity, 'faithful',
                         'if this label changes the test is no longer covering the hole')
        self.assertFalse(main.may_set_ranked_window(strat))
        # ...while an UNLEVERED faithful benchmark is still allowed to set it.
        self.assertTrue(main.may_set_ranked_window(main.ALL_STRATEGIES['SPY_Benchmark']()))

    def test_policy_all_still_lets_everything_set_the_window(self):
        """The escape hatch must keep working: `RANKED_WINDOW_POLICY='all'` is how LEVERAGE.md
        section 8's like-for-like table of custom wraps is produced."""
        import main
        for name in ('SPY_3X_Benchmark', 'DAA_G3_Leveraged_2X', 'HAA_G12'):
            with self.subTest(strategy=name):
                self.assertTrue(main.may_set_ranked_window(
                    main.ALL_STRATEGIES[name](), policy='all'))

    def test_the_levered_window_check_sees_both_kinds_of_levered_entry(self):
        """Two shapes reach `holds_leveraged_product`, and it must catch both: a benchmark that
        NAMES an LETF in its sleeve, and a wrap that names 1x tickers and swaps in the LETF at
        generate time (caught by `leverage != 1`). A check that saw only the first would let
        every wrap set the window."""
        import main
        from common.letf_mapper import holds_leveraged_product
        # names the LETF directly
        self.assertTrue(holds_leveraged_product(main.ALL_STRATEGIES['SPY_3X_Benchmark']()))
        self.assertTrue(holds_leveraged_product(main.ALL_STRATEGIES['RiskParity_3X']()))
        # declares SPY/QQQ/IWM and maps them at generate time
        wrap = main.ALL_STRATEGIES['DAA_G3_Leveraged_2X']()
        self.assertNotIn('SSO', set(wrap.offensive), 'the wrap declares 1x tickers')
        self.assertTrue(holds_leveraged_product(wrap), 'caught via .leverage')
        # and it must not fire on anything unlevered
        for clean in ('HAA_G12', 'SPY_Benchmark', 'Golden_Butterfly', 'Sixty_Forty_1X'):
            with self.subTest(strategy=clean):
                self.assertFalse(holds_leveraged_product(main.ALL_STRATEGIES[clean]()))

    def test_deleted_families_stay_deleted(self):
        import main
        for family in ('FAA', 'MAA', 'EAA', 'LAA', 'RAA', 'CAA'):
            with self.subTest(family=family):
                self.assertEqual([k for k in main.ALL_STRATEGIES
                                  if k.upper().startswith(family)], [])

    def test_no_registered_strategy_levers_its_defence(self):
        """Defence is held at 1x. Always.

        `DAA1_G12` declared `UST` — a 2x 7-10y treasury ETF — as a defensive candidate, so
        its risk-off state doubled duration risk and the engine still called it risk-off.
        It was deleted 2026-07-28. This is the check that keeps anything like it out.
        """
        import main
        from common.letf_mapper import assert_unlevered_defensive
        for name, factory in main.ALL_STRATEGIES.items():
            with self.subTest(strategy=name):
                assert_unlevered_defensive(factory())      # must not raise

    def test_the_levered_defence_check_actually_fires(self):
        """A guard nobody has seen fail is a guard nobody has tested."""
        import main
        from common.letf_mapper import assert_unlevered_defensive

        class Levered(main.ALL_STRATEGIES['DAA_G12']):
            def sleeves(self):
                s = super().sleeves()
                return {**s, 'defensive': set(s['defensive']) | {'UST'}}

        with self.assertRaises(ValueError) as ctx:
            assert_unlevered_defensive(Levered())
        self.assertIn('UST', str(ctx.exception))

    def test_leveraged_ticker_list_covers_both_maps(self):
        """The ban is only as good as the list of products it knows are levered."""
        from common.letf_mapper import LETFMapper, LEVERAGED_TICKERS
        images = set(LETFMapper.MAP_2X.values()) | set(LETFMapper.MAP_3X.values())
        self.assertTrue(images <= LEVERAGED_TICKERS,
                        f'mapped but not listed as leveraged: {sorted(images - LEVERAGED_TICKERS)}')

    def test_every_registered_strategy_declares_its_fidelity_and_role(self):
        """A number without a provenance label is half a number."""
        import main
        for name, factory in main.ALL_STRATEGIES.items():
            s = factory()
            with self.subTest(strategy=name):
                self.assertIn(s.fidelity, ('faithful', 'proxy', 'custom'))
                self.assertIn(s.role, ('strategy', 'control', 'benchmark', 'exploratory'))

    def test_every_registered_strategy_cites_a_source_for_its_fidelity_claim(self):
        """`fidelity` alone is an unfalsifiable string, and it was wrong three times.

        DM_G8_Composite, HAA_G1_Simple and VAA_G4 all carried the wrong label, always in
        the under-claiming direction, and the only test on any of them checked that the
        value was one of three legal words — which every wrong answer also is. A citation
        does not make the label correct; it makes it CHECKABLE against the PDF sitting in
        academic-papers/, which is the difference between a claim and an assertion. A
        `custom` entry cites what it departs from, so the departure is reviewable too.
        """
        import main
        for name, factory in main.ALL_STRATEGIES.items():
            with self.subTest(strategy=name):
                src = (getattr(factory(), 'source', '') or '').strip()
                self.assertTrue(src, f'{name} claims a fidelity with no citation behind it')
                self.assertGreater(len(src), 20,
                                   f'{name}: "{src}" does not identify a paper and a section')

    def test_a_leveraged_wrap_can_never_claim_fidelity(self):
        """Its signal is the paper's; its universe was invented here to fit the products."""
        import main
        for name, factory in main.ALL_STRATEGIES.items():
            if 'Leveraged' in name:
                with self.subTest(strategy=name):
                    self.assertEqual(factory().fidelity, 'custom')

    def test_every_registered_strategy_declares_sleeves(self):
        import main
        for name, factory in main.ALL_STRATEGIES.items():
            with self.subTest(strategy=name):
                s = factory().sleeves()
                self.assertEqual(set(s), {'offensive', 'defensive', 'canary'})
                self.assertTrue(s['offensive'] or s['defensive'],
                                f'{name} declares an empty book')


if __name__ == '__main__':
    unittest.main(verbosity=2)
