import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from common.letf_mapper import (LETFMapper, assert_protection_survives_restriction,
                                assert_unlevered_defensive)
from common.momentum import calc_13612w, calc_13612u
from strategies.daa_leveraged import DAALeveraged, DAA_G4_Leveraged
from strategies.haa_leveraged import HAA_G3_Leveraged, HAA_G4_Leveraged
from strategies.baa_leveraged import BAA_G3_Leveraged
from strategies.haa import HAA_12


def _prices(n=48, start='2016-01-31'):
    """Deterministic monthly prices for every ticker (incl. LETFs), gently rising."""
    from main import TICKERS
    idx = pd.date_range(start, periods=n, freq='ME')
    m = np.arange(n)
    data = {}
    for k, t in enumerate(TICKERS):
        drift = 0.002 + (k % 7) * 0.0015
        ripple = 1.0 + 0.01 * np.sin((m + k) * 0.5)
        data[t] = 100.0 * (1 + drift) ** m * ripple
    return pd.DataFrame(data, index=idx)


class TestLETFMapper(unittest.TestCase):

    def test_3x_mapping(self):
        mapper = LETFMapper(leverage=3)
        self.assertEqual(mapper.get_letf('SPY'), 'UPRO')
        self.assertEqual(mapper.get_letf('QQQ'), 'TQQQ')
        self.assertEqual(mapper.get_letf('IWM'), 'TNA')
        self.assertEqual(mapper.get_letf('TLT'), 'TMF')
        self.assertEqual(mapper.get_letf('EEM'), 'EDC')

    def test_2x_mapping(self):
        mapper = LETFMapper(leverage=2)
        self.assertEqual(mapper.get_letf('SPY'), 'SSO')
        self.assertEqual(mapper.get_letf('QQQ'), 'QLD')
        self.assertEqual(mapper.get_letf('IWM'), 'UWM')
        self.assertEqual(mapper.get_letf('GLD'), 'UGL')

    def test_rejected_products_are_not_mapped(self):
        """Products removed by the LETF audit must stay unmapped at every ratio.

        Removed for sub-$100M AUM (issuer-closure risk): EURL/DRN (3x), UBT/EET/EFO/URE
        (2x). Removed for wrong underlying: UCO (2x WTI crude, not a broad commodity
        index). GLD has no 3x product at all.
        """
        for ratio in (2, 3):
            mapper = LETFMapper(leverage=ratio)
            for asset in ('VGK', 'VNQ', 'GSG'):
                self.assertIsNone(mapper.get_letf(asset),
                                  f"{asset} must not map at {ratio}x")
            self.assertIsNone(mapper.get_letf('EWJ'))
        # Ratio-specific: no 3x gold; no viable 2x treasury or 2x EM.
        self.assertIsNone(LETFMapper(leverage=3).get_letf('GLD'))
        self.assertIsNone(LETFMapper(leverage=2).get_letf('TLT'))
        self.assertIsNone(LETFMapper(leverage=2).get_letf('EEM'))

    def test_retained_products_clear_the_liquidity_floor(self):
        mapper2, mapper3 = LETFMapper(leverage=2), LETFMapper(leverage=3)
        for letf in set(mapper2.map.values()) | set(mapper3.map.values()):
            aum = LETFMapper.RETAINED_AUM_USD.get(letf)
            self.assertIsNotNone(aum, f"{letf} is mapped but has no recorded AUM")
            self.assertGreaterEqual(aum, LETFMapper.LIQUIDITY_FLOOR_USD,
                                    f"{letf} is below the liquidity floor")


class TestUniverseCoherence(unittest.TestCase):
    """RULE 1: an offensive sleeve must execute at ONE uniform multiple."""

    def test_validate_accepts_uniform_universe(self):
        LETFMapper(leverage=3).validate_universe(['SPY', 'QQQ', 'IWM'])
        LETFMapper(leverage=2).validate_universe(['SPY', 'QQQ', 'IWM', 'GLD'])

    def test_validate_rejects_mixed_ratio_universe(self):
        # GLD would be UGL (2x) among 3x siblings -> mixed sleeve.
        with self.assertRaises(ValueError):
            LETFMapper(leverage=3).validate_universe(['SPY', 'QQQ', 'GLD'])
        # TLT has no viable 2x product -> would fall through to 1x.
        with self.assertRaises(ValueError):
            LETFMapper(leverage=2).validate_universe(['SPY', 'TLT'])

    def test_g4_cannot_be_constructed_at_3x(self):
        """The guard must fire at construction, not silently at execution."""
        with self.assertRaises(ValueError):
            DAA_G4_Leveraged(3)

    def test_keller_g12_universe_is_inadmissible(self):
        """The full G12 offensive universe fails at both ratios — this is why the
        DAA/VAA/BAA/PAA/GEM full-universe leveraged wraps were removed."""
        g12 = ['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'VWO', 'VNQ', 'GSG', 'GLD', 'TLT',
               'HYG', 'LQD']
        for ratio in (2, 3):
            with self.assertRaises(ValueError):
                LETFMapper(leverage=ratio).validate_universe(g12)


class TestDAALeveraged(unittest.TestCase):

    def test_canary_independent_of_offensive(self):
        """DAA's canary (VWO/BND) is separate from the offensive (leverageable) universe."""
        strategy = DAA_G4_Leveraged(2)
        self.assertTrue(set(strategy.canary).isdisjoint(strategy.offensive))

    def test_signal_trade_separation_2x(self):
        """Offensive originals are executed via LETFs, never held as the raw 1x ETF."""
        strategy = DAA_G4_Leveraged(2)
        prices = _prices()
        alloc = strategy.generate_allocations(prices, calc_13612w(prices), None, None)

        # Every offensive original is mapped -> none may carry weight directly.
        for original in ('SPY', 'QQQ', 'IWM', 'GLD'):
            self.assertLess(alloc[original].abs().max(), 1e-9,
                            f"{original} held raw instead of via its LETF")
        # On a rising market the strategy goes offensive, so leveraged ETFs are held.
        letf_cols = ['SSO', 'QLD', 'UWM', 'UGL']
        self.assertGreater(alloc[letf_cols].abs().to_numpy().sum(), 0.0)

    def test_offensive_weight_lands_only_on_one_ratio(self):
        """3x G3: all offensive weight must sit on 3x tickers, never on a 2x product."""
        strategy = BAA_G3_Leveraged(3)
        prices = _prices()
        alloc = strategy.generate_allocations(prices, calc_13612w(prices), None, None)
        for two_x in ('SSO', 'QLD', 'UWM', 'UGL'):
            if two_x in alloc.columns:
                self.assertLess(alloc[two_x].abs().max(), 1e-9,
                                f"3x strategy allocated to the 2x product {two_x}")


class TestCashLadderSurvives(unittest.TestCase):
    """RULE 5: `floor(b*T/B)/T` is Keller's Easy Trading ROUNDING of CF = b/B.

    It is harmless while T >= B and destructive below it. A wrap sets T from the size of the
    restricted universe, so it can walk into T < B without anyone choosing to.

    The rule was first written as a FILTER (delete any variant with T < B), which deleted
    `DAA_G3_Leveraged_2X` on 2026-07-29. It reads better as a rule that CHOOSES T —
    `T = max(n//2, B)` — and the variant was restored the same day. These tests pin the
    constructive form: the arithmetic that makes the rule necessary, and the T every wrap
    actually gets.
    """

    def test_the_ladder_that_was_lost(self):
        """The arithmetic itself, so the reason survives without reading the history."""
        def slots(b, T, B):
            n = int(np.floor(b * T / B))
            return 1.0 if n >= T else n / T

        # Paper (T=6,B=2) and every admissible wrap (T=2,B=2) keep the middle rung...
        self.assertEqual([slots(b, 6, 2) for b in (0, 1, 2)], [0.0, 0.5, 1.0])
        self.assertEqual([slots(b, 2, 2) for b in (0, 1, 2)], [0.0, 0.5, 1.0])
        # ...T=1 against B=2 does not: b=1 de-risks by nothing at all. This is the state
        # `T = max(n//2, B)` exists to make unreachable.
        self.assertEqual([slots(b, 1, 2) for b in (0, 1, 2)], [0.0, 0.0, 1.0])

    def test_daa_g3_gets_a_ladder_that_resolves_every_canary_count(self):
        """The restored G3. `n//2` would give T=1; the floor at B lifts it to 2.

        Without the floor this constructor produced a variant that held 2x equity through
        COVID for -35.1% and through 2011 for -25.0%, because b=1 de-risked by nothing.
        """
        s = DAALeveraged(['SPY', 'QQQ', 'IWM'], 2)
        self.assertEqual(len(s.offensive), 3)
        self.assertEqual((s.T, s.B), (2, 2), 'max(1, 3//2, B=2) must be 2, not 3//2 = 1')

        def cf(b):
            n = int(np.floor(b * s.T / s.B))
            return 1.0 if n >= s.T else n / s.T
        self.assertEqual([cf(b) for b in (0, 1, 2)], [0.0, 0.5, 1.0],
                         'the middle rung is the whole point of the floor at B')

    def test_daa_g4_is_admissible_and_unchanged_by_the_floor(self):
        """G4 already had T=2 from n//2, so `max(n//2, B)` must not move it.

        This is the check that the floor was added without disturbing anything: G4 is in the
        golden master, and its numbers may not shift.
        """
        s = DAA_G4_Leveraged(2)
        self.assertEqual((s.T, s.B), (2, 2))
        self.assertEqual(s.T, len(s.offensive) // 2,
                         'for G4 the floor must be inactive — n//2 already clears B')

    def test_guard_ignores_families_with_no_ladder(self):
        """B<=1 is binary in the paper too (BAA); no B at all means no ladder (HAA)."""
        assert_protection_survives_restriction(BAA_G3_Leveraged(2))   # B = 1
        assert_protection_survives_restriction(HAA_G3_Leveraged(2))   # no B


class TestHAALeveraged(unittest.TestCase):
    """RULE 4: the wrap may change what is HELD, never what decides to DE-RISK.

    HAA is the family that passes this most cleanly, which is why it gained a wrap on the
    same day VAA and PAA lost theirs.
    """

    def test_the_canary_is_exogenous_and_survives_the_restriction(self):
        for n, strat in (('G3', HAA_G3_Leveraged(2)), ('G4', HAA_G4_Leveraged(2))):
            self.assertEqual(strat.canary, ['TIP'], n)
            self.assertTrue(set(strat.canary).isdisjoint(strat.offensive), n)
            # The whole point: identical to the unlevered parent's canary.
            self.assertEqual(strat.canary, HAA_12().canary, n)

    def test_to_is_kellers_own_ratio_not_an_invented_one(self):
        """HAA publishes TO = NO/2 (HAA-12: 6 of 12; HAA-8: 4 of 8), so n//2 is the
        paper's rule here rather than a parameter the wrap had to make up."""
        self.assertEqual(HAA_G3_Leveraged(2).TO, 1)
        self.assertEqual(HAA_G4_Leveraged(2).TO, 2)
        self.assertEqual(HAA_12().TO, len(HAA_12().offensive) // 2)

    def test_defence_is_bil_ief_and_held_at_1x(self):
        for strat in (HAA_G3_Leveraged(2), HAA_G4_Leveraged(2)):
            self.assertEqual(sorted(strat.defensive), ['BIL', 'IEF'])
            assert_unlevered_defensive(strat)

    def test_offensive_originals_are_traded_as_letfs(self):
        strat = HAA_G4_Leveraged(2)
        prices = _prices()
        alloc = strat.generate_allocations(prices, calc_13612u(prices), None, None)
        for original in ('SPY', 'QQQ', 'IWM', 'GLD'):
            self.assertLess(alloc[original].abs().max(), 1e-9,
                            f'{original} held raw instead of via its LETF')
        self.assertGreater(alloc[['SSO', 'QLD', 'UWM', 'UGL']].abs().to_numpy().sum(), 0.0)

    def test_a_wrap_can_never_inherit_the_parents_fidelity(self):
        for strat in (HAA_G3_Leveraged(2), HAA_G4_Leveraged(2)):
            self.assertEqual(strat.fidelity, 'custom')
            self.assertIn('departs from', strat.source)
            self.assertIn('HAA', strat.source)

    def test_it_scores_on_13612u_like_every_haa(self):
        """HAA uses the UNWEIGHTED momentum; main.py routes scores by `score_type`, so a
        wrap that lost the attribute would be silently fed the wrong signal."""
        self.assertEqual(HAA_G3_Leveraged(2).score_type, 'unweighted')


if __name__ == '__main__':
    unittest.main()
