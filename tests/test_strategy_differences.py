"""The surviving leveraged wraps must be four different objects, not four names for one.

Rewritten 2026-07-29. This module used to prove that DAA/VAA/PAA wraps diverge; VAA's and
PAA's wraps were deleted that day under RULE 4 (see common/letf_mapper.py), so the same
question is now asked of the four that remain — HAA, BAA, DAA and DM. All four run on the
IDENTICAL restricted universe (SPY/QQQ/IWM/GLD or a subset of it), which is exactly what
makes the question sharp: with the universe held constant, any divergence is the signal,
and no divergence would mean the registry is carrying duplicates.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from common.momentum import calc_13612w, calc_13612u
from strategies.haa_leveraged import HAA_G4_Leveraged
from strategies.daa_leveraged import DAA_G4_Leveraged
from strategies.baa_leveraged import BAA_G4_Leveraged
from strategies.gem_leveraged import DM_G3_Leveraged


def _prices(n=48, start='2016-01-31'):
    from main import TICKERS
    idx = pd.date_range(start, periods=n, freq='ME')
    m = np.arange(n)
    data = {}
    for k, t in enumerate(TICKERS):
        drift = 0.002 + (k % 7) * 0.0015
        ripple = 1.0 + 0.01 * np.sin((m + k) * 0.5)
        data[t] = 100.0 * (1 + drift) ** m * ripple
    return pd.DataFrame(data, index=idx)


def _alloc(strat, prices):
    """Feed each strategy the momentum it actually scores on, as main.py does."""
    scores = calc_13612u(prices) if strat.score_type == 'unweighted' else calc_13612w(prices)
    return strat.generate_allocations(prices, scores, None, None)


class TestStrategyDifferences(unittest.TestCase):

    def test_the_four_surviving_wraps_have_distinct_names(self):
        names = [HAA_G4_Leveraged(2).name, BAA_G4_Leveraged(2).name,
                 DAA_G4_Leveraged(2).name, DM_G3_Leveraged(2).name]
        self.assertEqual(len(set(names)), 4, names)

    def test_each_family_reads_a_different_canary(self):
        """The declaration, before any allocation: three exogenous canaries, all different,
        and DM with none because its absolute-momentum test is per-asset."""
        self.assertEqual(HAA_G4_Leveraged(2).sleeves()['canary'], ['TIP'])
        self.assertEqual(DAA_G4_Leveraged(2).sleeves()['canary'], ['VWO', 'BND'])
        self.assertEqual(sorted(BAA_G4_Leveraged(2).sleeves()['canary']),
                         ['BND', 'SPY', 'VEA', 'VWO'])

    def test_canary_disagreement_produces_different_allocations(self):
        """Drive DAA's canary bearish while leaving HAA's and BAA's alone.

        VWO and BND are DAA's whole canary and only one quarter of BAA's, and HAA does not
        look at them at all. So on the SAME universe and the SAME month the three must
        allocate differently — that difference is the only thing distinguishing them once
        the offensive sleeve has been restricted to four tickers.
        """
        prices = _prices()
        n = len(prices)
        prices['VWO'] = np.linspace(130, 80, n)
        prices['BND'] = np.linspace(130, 80, n)

        haa = _alloc(HAA_G4_Leveraged(2), prices)
        baa = _alloc(BAA_G4_Leveraged(2), prices)
        daa = _alloc(DAA_G4_Leveraged(2), prices)

        self.assertFalse(daa.equals(haa), 'DAA and HAA allocate identically')
        self.assertFalse(daa.equals(baa), 'DAA and BAA allocate identically')
        self.assertFalse(haa.equals(baa), 'HAA and BAA allocate identically')

    def test_haa_ignores_the_daa_canary_entirely(self):
        """The sharpest version: collapse VWO and BND, and HAA must not notice.

        This is RULE 4 seen from the inside. HAA's protection reads TIP and nothing else,
        so a signal that empties DAA's canary cannot move it — which is precisely why HAA
        admits a leveraged wrap at all.
        """
        prices = _prices()
        baseline = _alloc(HAA_G4_Leveraged(2), prices)
        wrecked = prices.copy()
        wrecked['VWO'] = np.linspace(130, 40, len(prices))
        wrecked['BND'] = np.linspace(130, 40, len(prices))
        self.assertTrue(_alloc(HAA_G4_Leveraged(2), wrecked).equals(baseline))

    def test_dm_uses_a_different_momentum_from_the_keller_families(self):
        """DM ranks on 12-month total return; the Keller families use 13612. When an asset
        is a 12m winner but a recent loser the two must disagree."""
        prices = _prices(n=40)
        n = len(prices)
        prices['SPY'] = np.linspace(100, 120, n)
        qqq = np.empty(n)
        qqq[:28] = 60.0
        qqq[28:36] = np.linspace(65, 140, 8)
        qqq[36:] = np.linspace(135, 110, n - 36)
        prices['QQQ'] = qqq

        dm = _alloc(DM_G3_Leveraged(3), prices)
        haa = _alloc(HAA_G4_Leveraged(2), prices)
        self.assertFalse(dm.equals(haa), 'DM and HAA produce identical allocations')


if __name__ == '__main__':
    unittest.main()
