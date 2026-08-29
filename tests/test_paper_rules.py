"""
Each strategy against an independent implementation of its paper's rule.

This file exists because of a hole the 2026-07-29 audit found by walking into it. Every
strategy test in this repo asserted SELF-CONSISTENCY — weights sum to one, a dead canary
de-risks, the top four are equal-weighted. Not one of them compared an allocation against
the rule its paper actually states, so two defects sat in plain sight for months:

* DAA filtered its Top-T to strictly-positive momentum, which its paper disclaims three
  times over;
* SMA12 averaged twelve prices where Keller defines it over thirteen.

Both would have died the day a test asked "and what does the paper say?". The mutation
test for this file is the one the audit ran: change `rolling(13)` to `rolling(9)`, drop
DAA's cash slots, swap 13612W for 13612U, invert a BAA canary comparison — each must break
at least one assertion HERE, not merely move a number in the golden master. A golden master
can only say that a number changed; it cannot say which rule is right.

The anchors are hand-built panels whose correct allocation is derivable on paper, and
arithmetic written out in the test. Nothing here calls production code to decide what
production code should have produced.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from common.momentum import calc_13612u, calc_13612w, calc_sma10_ratio, calc_sma12_ratio


def _geometric(growth, n_months=20, start='2015-01-31'):
    """Prices with an exactly known shape: P_i = 100 * (1+g)^i, one column per ticker."""
    idx = pd.date_range(start, periods=n_months, freq='ME')
    return pd.DataFrame({t: 100.0 * (1.0 + g) ** np.arange(n_months)
                         for t, g in growth.items()}, index=idx)


def _scores(rows, n_months=14, start='2015-01-31'):
    """A scores panel whose LAST row is `rows` and whose earlier rows are benign.

    The strategies loop from i=12, so the panel needs 13 warm-up rows before the row under
    test. The warm-up rows carry a mild positive score for every ticker, which keeps them
    out of the way: the assertion is always about the final row.
    """
    idx = pd.date_range(start, periods=n_months, freq='ME')
    frame = pd.DataFrame(0.01, index=idx, columns=list(rows))
    frame.iloc[-1] = pd.Series(rows)
    return frame


# --------------------------------------------------------------------------------- #
# Momentum filters
# --------------------------------------------------------------------------------- #
class TestSMAWindows(unittest.TestCase):
    """Keller and Faber use the same NAME for two different windows, and the difference is
    not cosmetic: it shifts every BAA and PAA score.

    Keller, BAA (SSRN 4166845) n.5: "The SMA(12) momentum (with lag 12 months) equals the
    present price pt divided by the average of the last 13 asset prices including the
    present (also noted as SMA13), minus 1."

    Faber (2006) counts prices, not lags: SMA(10) is the average of ten.
    """

    PANEL = {'A': 0.02, 'B': -0.01, 'C': 0.0}

    def test_sma12_is_the_ratio_to_the_last_thirteen_prices(self):
        p = _geometric(self.PANEL)
        got = calc_sma12_ratio(p)
        for t in self.PANEL:
            for i in (12, 15, 19):
                with self.subTest(ticker=t, row=i):
                    # The average of p[i-12] .. p[i] inclusive — thirteen prices, counted
                    # here and not by the function under test.
                    want = float(p[t].iloc[i - 12:i + 1].mean())
                    self.assertAlmostEqual(float(got[t].iloc[i]),
                                           float(p[t].iloc[i]) / want - 1.0, places=12)

    def test_sma12_warmup_ends_on_the_thirteenth_row(self):
        """Pins the window length from the other side: twelve prices are not enough.

        Without this, a future 'consistency' edit back to rolling(12) would silently
        lengthen every BAA and PAA history by one month and no test would notice.
        """
        got = calc_sma12_ratio(_geometric(self.PANEL))
        self.assertTrue(got['A'].iloc[:12].isna().all())
        self.assertTrue(np.isfinite(float(got['A'].iloc[12])))

    def test_sma12_weights_the_last_twelve_returns_twelve_down_to_one(self):
        """The same definition seen from Keller & Butler's side, and the reason it matters.

        PAA (SSRN 2759734) §1: SMA(12) "is based on a linearly decreasing weight filter over
        the previous 12 monthly returns (so the most recent month return is weighted 12
        times ... up to the oldest (12th) month one time)".

        With S = mean(p_t .. p_t-12), the identity below is exact:

            p_t/S - 1 = (1/13S) * sum_{k=0..11} (12-k) * (p_t-k - p_t-k-1)

        i.e. weights 12, 11, ..., 1 on the last twelve monthly price changes. Average only
        TWELVE prices and the weights become 11, 10, ..., 1, 0 — the oldest month drops out
        entirely and the filter spans eleven returns, not the paper's twelve.
        """
        p = _geometric({'A': 0.02})['A']
        i = 15
        S = float(p.iloc[i - 12:i + 1].mean())
        weighted = sum((12 - k) * (float(p.iloc[i - k]) - float(p.iloc[i - k - 1]))
                       for k in range(12))
        self.assertAlmostEqual(float(calc_sma12_ratio(p.to_frame()).iloc[i, 0]),
                               weighted / (13.0 * S), places=12)

    def test_sma10_still_averages_ten_prices(self):
        """Faber's filter is NOT Keller's. Guards against a well-meaning edit that makes
        the two 'consistent' and silently breaks GTAA."""
        p = _geometric(self.PANEL)
        got = calc_sma10_ratio(p)
        want = float(p['A'].iloc[6:16].mean())          # ten prices, p[6]..p[15]
        self.assertAlmostEqual(float(got['A'].iloc[15]),
                               float(p['A'].iloc[15]) / want - 1.0, places=12)
        self.assertTrue(got['A'].iloc[:9].isna().all())
        self.assertTrue(np.isfinite(float(got['A'].iloc[9])))


class TestMomentumMissingData(unittest.TestCase):
    """A gap in the panel must stay a gap.

    `pct_change` defaulted to `fill_method='pad'` through pandas 2.1. Under that default a
    delisted product's last price is carried forward into a live momentum score — exactly
    what `_apply_stale_policy` writes NaN to prevent.
    """

    def test_an_interior_gap_does_not_carry_forward(self):
        p = _geometric({'A': 0.02}, n_months=20)
        p.iloc[14, 0] = np.nan
        for name, fn in (('13612W', calc_13612w), ('13612U', calc_13612u)):
            with self.subTest(filter=name):
                s = fn(p)['A']
                self.assertTrue(pd.isna(s.iloc[14]))
                # r1 at row 15 reads row 14, which is missing.
                self.assertTrue(pd.isna(s.iloc[15]))


# --------------------------------------------------------------------------------- #
# DAA — Keller & Keuning, SSRN 3212862
# --------------------------------------------------------------------------------- #
class TestDAASelectionRule(unittest.TestCase):
    """DAA's Top-T is sign-agnostic. The paper says so three times.

    §8: "we did not use absolute momentum (ie. eliminating bad assets in favor of cash),
    except for the number of bad canary assets ... So for DAA, we did not eliminate bad
    assets in the top T selection of risky assets, although we reduced the top T"

    Conclusions: "EW-Top T (T<=N, long only), without intrinsic or absolute momentum."

    n.17 records absolute momentum on the risky Top-T as an alternative they TRIED and did
    not adopt. Until 2026-07-29 this repository implemented that rejected alternative.

    The canary alone decides how much goes to cash; the offensive sleeve's own breadth
    never does. That separation is the whole difference between DAA and VAA.
    """

    #: Two positive, ten negative, canary fully alive. Under the paper the negative names
    #: are still held — they are the best available, and DAA's protection is the canary.
    ROW = {'SPY': 0.10, 'VWO': 0.08,
           'IWM': -0.01, 'QQQ': -0.02, 'VGK': -0.03, 'EWJ': -0.04,
           'VNQ': -0.05, 'GSG': -0.06, 'GLD': -0.07, 'TLT': -0.08, 'HYG': -0.09,
           'LQD': -0.10,
           'BND': 0.05, 'SHY': 0.01, 'IEF': 0.02}

    def _held(self, row):
        from strategies.daa import DAA_G12
        alloc = DAA_G12()._generate(_scores(row))
        held = alloc.iloc[-1]
        return held[held.abs() > 1e-12]

    def test_top_t_is_held_regardless_of_sign(self):
        held = self._held(self.ROW)
        # b = 0 (VWO and BND both positive) -> no cash slots -> the full T=6, equal weight.
        self.assertEqual(sorted(held.index), ['EWJ', 'IWM', 'QQQ', 'SPY', 'VGK', 'VWO'])
        for t in held.index:
            self.assertAlmostEqual(float(held[t]), 1.0 / 6.0, places=12)
        self.assertNotIn('IEF', held.index)     # nothing defensive while the canary is alive

    def test_a_single_positive_asset_does_not_become_the_whole_book(self):
        """The failure mode the old filter produced, stated as a portfolio.

        Eleven of twelve offensive assets negative and the canary alive: filtering to
        positives leaves ONE name, and `(1-cf)/1` puts 100% of the book into a single
        emerging-market equity fund in the worst breadth month the universe can produce.
        The paper holds the six best, whatever their sign.
        """
        held = self._held(dict(self.ROW, SPY=-0.005))       # only VWO left positive
        self.assertEqual(sorted(held.index), ['EWJ', 'IWM', 'QQQ', 'SPY', 'VGK', 'VWO'])
        for t in held.index:
            self.assertAlmostEqual(float(held[t]), 1.0 / 6.0, places=12)

    def test_the_cash_fraction_comes_from_the_canary_alone(self):
        """b=1 of B=2 -> floor(1*6/2) = 3 cash slots -> CF = 1/2, three risky names.

        Offensive breadth is unchanged from the first test; only the canary moved. If the
        offensive sign filter were still present the risky sleeve would hold two names, not
        three, and the split would not be 1/6 each.
        """
        held = self._held(dict(self.ROW, BND=-0.01))
        self.assertAlmostEqual(float(held['IEF']), 0.5, places=12)
        risky = held.drop('IEF')
        self.assertEqual(sorted(risky.index), ['IWM', 'SPY', 'VWO'])
        for t in risky.index:
            self.assertAlmostEqual(float(risky[t]), 0.5 / 3.0, places=12)

    def test_a_dead_canary_still_goes_fully_defensive(self):
        """b=2 of B=2 -> floor(2*6/2)=6 >= T. The canary is the ONLY route to full defence."""
        held = self._held(dict(self.ROW, BND=-0.01, VWO=-0.01))
        self.assertEqual(list(held.index), ['IEF'])
        self.assertAlmostEqual(float(held['IEF']), 1.0, places=12)


class TestDAAEasyTradingAtTopOne(unittest.TestCase):
    """DAA n.8, verbatim: "We also have added the rule that with T=1, CF is simply b/B, in
    line with the ET idea, with b the number of bad assets."

    The floor formula loses the middle rung at T=1 — floor(b*T/B) is 0 for every b < B — and
    the paper legislated the case directly rather than accept that. Unimplemented here until
    2026-07-30 (no registered DAA reaches T=1), found by the 2026-07-30 external audit.

    Mutation standard: reverting `strategies/daa.py` to the bare `floor(b*T/B)/T` at T=1
    must break `test_one_dead_canary_buys_half_cash` — under the bare floor the b=1 row
    holds 100% risky and 0% defensive.
    """

    ROW = {'SPY': 0.10, 'VWO': 0.08,
           'IWM': -0.01, 'QQQ': -0.02, 'VGK': -0.03, 'EWJ': -0.04,
           'VNQ': -0.05, 'GSG': -0.06, 'GLD': -0.07, 'TLT': -0.08, 'HYG': -0.09,
           'LQD': -0.10,
           'BND': 0.05, 'SHY': 0.01, 'IEF': 0.02}

    def _held(self, row):
        from strategies.daa import DAA_G12
        strat = DAA_G12()
        strat.T = 1                                  # the n.8 case; B stays 2
        alloc = strat._generate(_scores(row))
        held = alloc.iloc[-1]
        return held[held.abs() > 1e-12]

    def test_no_bad_canary_holds_the_single_best_asset_fully(self):
        held = self._held(self.ROW)                  # b=0 -> CF = 0/2 = 0
        self.assertEqual(list(held.index), ['SPY'])
        self.assertAlmostEqual(float(held['SPY']), 1.0, places=12)

    def test_one_dead_canary_buys_half_cash(self):
        """b=1 of B=2 -> CF = 1/2: the single risky slot at 50% beside the best defensive
        asset at 50%. This is the rung the bare floor formula deletes."""
        held = self._held(dict(self.ROW, BND=-0.01))
        self.assertEqual(sorted(held.index), ['IEF', 'SPY'])
        self.assertAlmostEqual(float(held['SPY']), 0.5, places=12)
        self.assertAlmostEqual(float(held['IEF']), 0.5, places=12)

    def test_two_dead_canaries_go_fully_defensive(self):
        held = self._held(dict(self.ROW, BND=-0.01, VWO=-0.01))
        self.assertEqual(list(held.index), ['IEF'])
        self.assertAlmostEqual(float(held['IEF']), 1.0, places=12)


# --------------------------------------------------------------------------------- #
# VAA — Keller & Keuning, SSRN 3002624
# --------------------------------------------------------------------------------- #
class TestVAAEasyTrading(unittest.TestCase):
    """VAA §4's Easy Trading rounding, tabulated.

    CF = (1/T) * rounddown(b*T/B), capped at 1. The rounding is the point: it quantises the
    cash fraction onto the T slots the portfolio actually has, so a rebalance never asks for
    a fraction of a position that cannot be traded.
    """

    #: Read from the strategy, never hard-coded. This test is about the CF ARITHMETIC, so it
    #: must not fail when the universe changes for an unrelated reason — which is exactly
    #: what happened on 2026-07-29, when VAA_G12 moved from EEM/IYR to VWO/VNQ.
    DEF = ['SHY', 'IEF']

    def _cash_fraction(self, n_bad, T, B):
        from strategies.vaa import VAA_G12
        strat = VAA_G12()
        strat.T, strat.B = T, B
        strat.defensive = list(self.DEF)
        off = list(strat.offensive)
        self.assertEqual(len(off), 12, 'the tabulated b values assume a 12-asset universe')
        row = {t: (0.05 if i >= n_bad else -0.05) for i, t in enumerate(off)}
        row.update({'SHY': 0.01, 'IEF': 0.02})       # IEF is the better defensive asset
        held = strat._generate(_scores(row)).iloc[-1]
        return float(held.reindex(self.DEF).fillna(0.0).sum())

    def test_easy_trading_rounds_the_cash_fraction_down_onto_the_slots(self):
        # T=3, B=4: b*T/B = 0, .75, 1.5, 2.25, 3 -> floor -> 0, 0, 1, 2, 3 slots of 1/3.
        for b, want in ((0, 0.0), (1, 0.0), (2, 1 / 3), (3, 2 / 3), (4, 1.0)):
            with self.subTest(bad_assets=b):
                self.assertAlmostEqual(self._cash_fraction(b, T=3, B=4), want, places=12)

    def test_the_registered_parameterisations(self):
        # VAA-G12 (Table 1): T=2, B=4 -> b*T/B = 0, .5, 1, 1.5, 2. One bad asset still buys
        # nothing; two and three both buy exactly half; four goes fully to cash.
        for b, want in ((0, 0.0), (1, 0.0), (2, 0.5), (3, 0.5), (4, 1.0)):
            with self.subTest(variant='G12', bad_assets=b):
                self.assertAlmostEqual(self._cash_fraction(b, T=2, B=4), want, places=12)
        # VAA-G4 (Table 5): T=1, B=1 -> binary. Any bad asset goes fully to cash.
        for b, want in ((0, 0.0), (1, 1.0), (2, 1.0)):
            with self.subTest(variant='G4', bad_assets=b):
                self.assertAlmostEqual(self._cash_fraction(b, T=1, B=1), want, places=12)

    def test_top_one_goes_all_in_cash_on_any_bad_asset_whatever_b_is(self):
        """VAA §4, verbatim: "When B=1 or T=1 the whole portfolio is fully invested in cash
        (when b>=1) or fully invested in the top T risky asset(s) (when b=0)."

        The floor formula only reproduces that at B=1; at T=1 with B>1 it yields
        floor(b/B)=0 and STAYS FULLY RISKY — the opposite of the paper's sentence.
        Implemented 2026-07-30 (2026-07-30 audit, METH-002 sibling). No registered VAA
        reaches it — VAA_G4 is T=1, B=1 — so the golden master cannot see this; only this
        test can. NOTE this is VAA's OWN rule and deliberately differs from DAA n.8, which
        grades the same case to CF = b/B: each paper governs its own family, and making
        these two agree would be exactly the kind of "consistency" fix that broke SMA12.

        Mutation standard: removing the `T == 1 and b >= 1` branch in `strategies/vaa.py`
        must break the b=1 case below (bare floor: floor(1*1/4) = 0 -> fully risky).
        """
        for b, want in ((0, 0.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)):
            with self.subTest(bad_assets=b):
                self.assertAlmostEqual(self._cash_fraction(b, T=1, B=4), want, places=12)


# --------------------------------------------------------------------------------- #
# PAA — Keller & Butler, SSRN 2759734
# --------------------------------------------------------------------------------- #
class TestPAAProtection(unittest.TestCase):
    """PAA's bond fraction: CF = (N-n)/(N - a*N/4), clipped to [0, 1].

    §3 names the three models by their protection factor a, and settles on a=2: "It is this
    PAA2 model which we consider our alternative for a 1-year term deposit." On N=12 that is
    CF = (12-n)/6, so protection is complete once half the universe has rolled over.
    """

    def _cash_fraction(self, n_positive):
        from strategies.paa import PAA2
        strat = PAA2()
        n = len(strat.offensive)
        # Prices, not scores: PAA computes SMA12 itself, so the panel has to make the sign
        # of each asset's own trend the thing under test.
        growth = {t: (0.02 if i < n_positive else -0.02)
                  for i, t in enumerate(strat.offensive)}
        growth['IEF'] = 0.001
        held = strat.generate_allocations(_geometric(growth), None, None, None).iloc[-1]
        # IEF carries the cash fraction; it is never in the offensive universe.
        return float(held.get('IEF', 0.0)), int((held.drop('IEF') > 1e-12).sum())

    def test_the_bond_fraction_is_the_paper_formula(self):
        for n in range(0, 13):
            with self.subTest(n_positive=n):
                cf, _ = self._cash_fraction(n)
                self.assertAlmostEqual(cf, max(0.0, min(1.0, (12 - n) / 6.0)), places=12)

    def test_protection_is_complete_at_half_the_universe(self):
        cf, n_risky = self._cash_fraction(6)
        self.assertAlmostEqual(cf, 1.0, places=12)
        self.assertEqual(n_risky, 0)

    def test_a_full_universe_holds_the_top_six_and_no_bonds(self):
        cf, n_risky = self._cash_fraction(12)
        self.assertAlmostEqual(cf, 0.0, places=12)
        self.assertEqual(n_risky, 6)


class TestPAASelectionHoldsOnlyGoodAssets(unittest.TestCase):
    """PAA recipe step 2, verbatim: "Determine the Top (Top<=N) good assets with the highest
    momentum ... If n<Top, only the n good assets (with positive momentum) will be included
    in this risky EW portfolio."

    This test pins a filter an external audit asked to REMOVE (2026-07-30, LOGIC-001),
    citing BAA §2's restatement: "like for PAA, we don't use absolute momentum for the Top6
    selection". Keller's 2022 recollection contradicts his own 2016 recipe, and the entry's
    `source` cites the 2016 paper, so the 2016 paper governs. Contrast DAA, where the SAME
    filter was a real defect (its paper disclaims it three times, removed 2026-07-29):
    identical line of code, opposite verdicts, each decided by its own primary source.

    PAA2 cannot show the difference — it holds risk only when n >= 7, where the top six are
    all positive either way — so the anchor is PAA0 (a=0, no clipping), where n < Top with
    CF < 1 is reachable. Mutation standard: replacing `off_mom[off_mom > 0]` with
    `off_mom.dropna()` in `strategies/paa.py` must break this test (six names held, three
    of them negative, at 0.25/6 each).
    """

    def test_with_three_good_assets_paa0_holds_exactly_those_three(self):
        from strategies.paa import PAA
        strat = PAA('PAA0_TEST', variant='PAA0')
        # 3 positive trends, 9 negative -> n=3: CF = (12-3)/12 = 0.75, risky sleeve 0.25.
        growth = {t: (0.02 if i < 3 else -0.02) for i, t in enumerate(strat.offensive)}
        growth['IEF'] = 0.001
        held = strat.generate_allocations(_geometric(growth), None, None, None).iloc[-1]
        good = list(strat.offensive[:3])
        self.assertAlmostEqual(float(held['IEF']), 0.75, places=12)
        for t in good:
            self.assertAlmostEqual(float(held[t]), 0.25 / 3.0, places=12,
                                    msg=f'{t} should carry (1-CF)/n, not (1-CF)/Top')
        for t in strat.offensive[3:]:
            self.assertAlmostEqual(float(held.get(t, 0.0)), 0.0, places=12,
                                    msg=f'{t} has negative momentum and must not be held '
                                        f'(recipe step 2: only the n good assets)')


# --------------------------------------------------------------------------------- #
# BAA — Keller, SSRN 4166845
# --------------------------------------------------------------------------------- #
class TestBAADefensiveFilter(unittest.TestCase):
    """BAA §2 step 3: the defensive basket takes the TD best by SMA12, and any slot whose
    own SMA12 does not beat BIL's is filled with BIL instead.

    The filter is not decoration. BIL is itself one of the seven candidates, so a slot can
    only fail the test while BIL outranks it — which means BIL doubles up rather than the
    weight going anywhere else.
    """

    def _defensive_book(self):
        from strategies.baa import BAA_G12
        strat = BAA_G12()
        growth = {t: -0.05 for t in strat.offensive}
        growth.update({t: -0.05 for t in strat.canary})     # canary down -> defensive month
        growth.update({'TIP': 0.03, 'BIL': 0.001, 'IEF': -0.002,
                       'DBC': -0.05, 'TLT': -0.05, 'LQD': -0.05, 'BND': -0.05})
        alloc = strat.generate_allocations(_geometric(growth), calc_13612w(_geometric(growth)),
                                           None, None)
        held = alloc.iloc[-1]
        return held[held.abs() > 1e-12]

    def test_a_slot_below_bil_is_filled_with_bil(self):
        """Ranking by SMA12 puts TIP first, BIL second, IEF third. IEF trends DOWN while
        BIL trends up, so the third slot fails the absolute test and becomes BIL: the book
        is one third TIP and two thirds BIL, and IEF is not held at all."""
        held = self._defensive_book()
        self.assertEqual(sorted(held.index), ['BIL', 'TIP'])
        self.assertAlmostEqual(float(held['TIP']), 1 / 3, places=12)
        self.assertAlmostEqual(float(held['BIL']), 2 / 3, places=12)

    def test_the_canary_is_tested_at_or_below_zero(self):
        """A canary sitting exactly at zero is DOWN. `< 0` instead of `<= 0` would hold a
        full offensive book in the month a canary's momentum has just died."""
        from strategies.baa import BAA_G12
        strat = BAA_G12()
        growth = {t: 0.02 for t in strat.offensive}
        growth.update({t: 0.02 for t in strat.canary})
        growth['VWO'] = 0.0                                  # exactly zero 13612W
        growth.update({'TIP': 0.01, 'BIL': 0.005, 'IEF': 0.004,
                       'DBC': 0.003, 'TLT': 0.002, 'LQD': 0.001, 'BND': 0.02})
        p = _geometric(growth)
        self.assertEqual(float(calc_13612w(p)['VWO'].iloc[-1]), 0.0)
        held = strat.generate_allocations(p, calc_13612w(p), None, None).iloc[-1]
        held = held[held.abs() > 1e-12]
        # Defensive month: the book is the TD=3 basket, not the TO=6 offensive one.
        self.assertTrue(set(held.index) <= set(strat.defensive))


# --------------------------------------------------------------------------------- #
# HAA — Keller & Keuning, SSRN 4346906
# --------------------------------------------------------------------------------- #
class TestHAACanaryBoundary(unittest.TestCase):
    """HAA §3: TIP's momentum "not positive" is the risk-off condition — so exactly zero
    is risk-off, not risk-on. The boundary was never tested, and `<` instead of `<=` is a
    one-character edit that no other test in the suite can see."""

    def test_a_canary_at_exactly_zero_is_risk_off(self):
        from strategies.haa import HAA_12
        strat = HAA_12()
        growth = {t: 0.02 for t in strat.offensive}
        growth.update({'TIP': 0.0, 'BIL': 0.001, 'IEF': 0.003})
        p = _geometric(growth)
        self.assertEqual(float(calc_13612u(p)['TIP'].iloc[-1]), 0.0)
        held = strat.generate_allocations(p, calc_13612u(p), None, None).iloc[-1]
        held = held[held.abs() > 1e-12]
        self.assertEqual(list(held.index), ['IEF'])          # the better of BIL / IEF
        self.assertAlmostEqual(float(held['IEF']), 1.0, places=12)


# --------------------------------------------------------------------------------- #
# GTAA — Faber (2006)
# --------------------------------------------------------------------------------- #
class TestGTAASleeves(unittest.TestCase):
    """Faber's five-asset timing model: each sleeve is 20% of the book, held only while its
    own price is above its own 10-month SMA, and every failing sleeve's 20% goes to cash.
    There is no ranking and no canary — the absolute filter IS the rule."""

    def test_each_failing_sleeve_sends_exactly_its_own_fifth_to_cash(self):
        from strategies.gtaa import GTAA_5
        strat = GTAA_5()
        for n_down in range(0, 6):
            with self.subTest(sleeves_down=n_down):
                growth = {t: (-0.02 if i < n_down else 0.02)
                          for i, t in enumerate(strat.universe)}
                growth['BIL'] = 0.0002
                held = strat.generate_allocations(_geometric(growth), None, None, None).iloc[-1]
                self.assertAlmostEqual(float(held['BIL']), n_down / 5.0, places=12)
                for t in strat.universe[n_down:]:
                    self.assertAlmostEqual(float(held[t]), 0.2, places=12)


class TestGEMFlowchart(unittest.TestCase):
    """Antonacci's GEM decision tree, in the BOOK's order — absolute momentum first,
    gauged on the S&P 500 alone; relative momentum second; defence into aggregate bonds.

    The ordering is the whole test. The 2012 paper's equities module (`DMComposite`) does
    relative FIRST and then checks the WINNER against T-bills; GEM gauges the regime on
    SPY regardless of who wins the relative leg. The two disagree in a specific month
    shape — ex-US wins relative, SPY beats bills, ex-US does not — and
    `test_the_gauge_is_spy_not_the_winner` is built on exactly that shape. Mutation
    standard: re-ordering the two tests, or gauging on the winner, must break it.
    """

    def _held(self, monthly_returns):
        """Prices from constant per-ticker monthly growth, so each 12m return is known."""
        from strategies.gem import GEMClassic
        idx = pd.date_range('2015-01-31', periods=20, freq='ME')
        prices = pd.DataFrame({t: 100.0 * (1.0 + g) ** np.arange(20)
                               for t, g in monthly_returns.items()}, index=idx)
        held = GEMClassic().generate_allocations(prices, None, None, None).iloc[-1]
        return held[held.abs() > 1e-12]

    def test_spy_beats_bills_and_veu_holds_spy(self):
        held = self._held({'SPY': 0.010, 'VEU': 0.005, 'BND': 0.002, 'BIL': 0.001})
        self.assertEqual(list(held.index), ['SPY'])
        self.assertAlmostEqual(float(held['SPY']), 1.0, places=12)

    def test_veu_wins_the_relative_leg(self):
        held = self._held({'SPY': 0.008, 'VEU': 0.012, 'BND': 0.002, 'BIL': 0.001})
        self.assertEqual(list(held.index), ['VEU'])

    def test_spy_below_bills_defends_into_bonds_whatever_veu_does(self):
        """VEU beats bills handsomely; SPY does not. GEM never consults the relative leg:
        bonds. (A winner-gauged implementation would hold VEU here.)"""
        held = self._held({'SPY': 0.000, 'VEU': 0.015, 'BND': 0.002, 'BIL': 0.001})
        self.assertEqual(list(held.index), ['BND'])
        self.assertAlmostEqual(float(held['BND']), 1.0, places=12)

    def test_the_gauge_is_spy_not_the_winner(self):
        """VEU wins relative but sits BELOW bills, while SPY beats bills. The book's tree
        holds VEU (SPY passed the absolute gate; VEU won the relative leg). The 2012
        module would check the winner (VEU) against bills, fail it, and hold T-bills —
        so this month distinguishes the two orderings observably."""
        # 12m compounded: SPY ~ +6.2%, BIL ~ +2.4%, VEU ~ +6.6% relative winner?  No —
        # VEU must beat SPY while trailing BIL, which is impossible if SPY > BIL. The
        # distinguishing shape is the OTHER branch: SPY > BIL, VEU > SPY, and VEU's own
        # absolute state is irrelevant BECAUSE it exceeds SPY's, which exceeds bills. The
        # observable ordering difference is therefore the bonds-vs-winner case above plus
        # THIS one: SPY fails the gate while remaining the relative winner — the 2012
        # module would hold T-bills (winner SPY fails absolute), GEM holds BND.
        held = self._held({'SPY': 0.0005, 'VEU': 0.0002, 'BND': 0.003, 'BIL': 0.001})
        self.assertEqual(list(held.index), ['BND'],
                         'GEM defends into AGGREGATE BONDS, never into the T-bill gauge')

    def test_missing_veu_defends(self):
        """Before VEU's inception the tree cannot run; the defensive asset carries the
        month (unreachable inside the measured window — coverage trims to VEU + warmup)."""
        from strategies.gem import GEMClassic
        idx = pd.date_range('2015-01-31', periods=20, freq='ME')
        prices = pd.DataFrame({'SPY': 100.0, 'VEU': np.nan, 'BND': 90.0, 'BIL': 91.0},
                              index=idx)
        held = GEMClassic().generate_allocations(prices, None, None, None).iloc[-1]
        self.assertAlmostEqual(float(held['BND']), 1.0, places=12)


class TestFidelityAgainstSource(unittest.TestCase):
    """Every registry key's fidelity claim, pinned to the paper that has to back it.

    FOUR labels have now been found wrong — DM_G8_Composite, HAA_G1_Simple, VAA_G4, and
    BAA_G1_SPY on 2026-07-30 — every one by a human re-reading a PDF, every one in the
    under-claiming direction, and the fourth AFTER the citation requirement was added. A
    required citation is checkable by a human and by nothing in CI. This table is the CI
    check: a pinned `(fidelity, SSRN id, section/figure token)` per key, so a wrong label
    fails a test instead of waiting for the next audit, and a new entry cannot register
    without a reviewer stating its label and locating it in a paper.

    Deliberately NOT a PDF parser. Extracting SSRN text in CI would be brittle, and the
    table's value is the review conversation it forces on the diff, not the string match.
    The strings pin three things: the label itself, the paper the `source` must cite, and
    the section/figure token that must appear in it.
    """

    # (fidelity, token that must appear in source, second token that must appear in source)
    # 'SSRN nnnn' pins the paper; the third element pins the section/figure. For entries
    # with no paper (benchmarks, wraps) the tokens pin the honest phrase instead.
    EXPECTED = {
        # --- unlevered Keller/Keller-&-co entries, checked against the local PDFs ------- #
        'HAA_G12':          ('faithful', 'SSRN 4346906', 'Fig 10'),
        'HAA_G8_Balanced':  ('faithful', 'SSRN 4346906', 'Fig 6'),
        'HAA_G1_Simple':    ('faithful', 'SSRN 4346906', '§6'),
        'DAA_G12':          ('faithful', 'SSRN 3212862', '§7'),
        'DAA_G4':           ('faithful', 'SSRN 3212862', '§3'),
        'DAA_G6':           ('faithful', 'SSRN 3212862', '§7'),
        'VAA_G12':          ('faithful', 'SSRN 3002624', 'n.11'),
        'VAA_G4':           ('faithful', 'SSRN 3002624', 'n.11'),
        'BAA_G12':          ('faithful', 'SSRN 4166845', 'Fig 3'),
        'BAA_G4':           ('faithful', 'SSRN 4166845', 'Fig 6'),
        # Keller's own BAA-SPY (§5, Fig 11). Labelled 'custom' until 2026-07-30 on a
        # docstring claim the paper contradicts — the mislabel this table exists to catch.
        'BAA_G1_SPY':       ('faithful', 'SSRN 4166845', 'Fig 11'),
        'PAA2_G12':         ('faithful', 'SSRN 2759734', '§3'),
        # --- non-Keller sources ---------------------------------------------------------- #
        'DM_G8_Composite':  ('proxy',    'SSRN 2042750', 'Table 10'),
        # GEM's source is Antonacci's BOOK, not an SSRN paper — the pin holds the title and
        # the ordering claim ("gauged on the S&P 500 first"), which is the rule most at
        # risk of being "made consistent" with the 2012 module.
        'GEM_G2_Classic':   ('proxy',    'Dual Momentum Investing', 'S&P 500 first'),
        'GTAA_G5':          ('proxy',    'SSRN 962461',  'Faber'),
        # --- leveraged wraps: custom by construction (LeveragedWrapMixin hard-codes it),
        #     and the source must say which paper's signal they depart from ---------------- #
        'HAA_G3_Leveraged_2X': ('custom', 'departs from', 'SSRN 4346906'),
        'HAA_G4_Leveraged_2X': ('custom', 'departs from', 'SSRN 4346906'),
        'HAA_G3_Leveraged_3X': ('custom', 'departs from', 'SSRN 4346906'),
        'HAA_G5_Leveraged_3X': ('custom', 'departs from', 'SSRN 4346906'),
        'DAA_G3_Leveraged_2X': ('custom', 'departs from', 'SSRN 3212862'),
        'DAA_G4_Leveraged_2X': ('custom', 'departs from', 'SSRN 3212862'),
        'DAA_G3_Leveraged_3X': ('custom', 'departs from', 'SSRN 3212862'),
        'DAA_G5_Leveraged_3X': ('custom', 'departs from', 'SSRN 3212862'),
        'BAA_G3_Leveraged_2X': ('custom', 'departs from', 'SSRN 4166845'),
        'BAA_G4_Leveraged_2X': ('custom', 'departs from', 'SSRN 4166845'),
        'BAA_G3_Leveraged_3X': ('custom', 'departs from', 'SSRN 4166845'),
        'BAA_G4_Leveraged_3X': ('custom', 'departs from', 'SSRN 4166845'),
        'DM_G3_Leveraged_2X':  ('custom', 'departs from', 'SSRN 2042750'),
        'DM_G3_Leveraged_3X':  ('custom', 'departs from', 'SSRN 2042750'),
        'DM_G5_Leveraged_3X':  ('custom', 'departs from', 'SSRN 2042750'),
        # --- passive benchmarks: no algorithm, so the source pins provenance ------------- #
        'SPY_Benchmark':    ('faithful', 'buy and hold', 'no rule'),
        'SPY_2X_Benchmark': ('faithful', 'buy and hold', 'no rule'),
        'SPY_3X_Benchmark': ('faithful', 'buy and hold', 'no rule'),
        'Sixty_Forty_1X':   ('faithful', '60/40', 'SPY'),
        'Golden_Butterfly': ('faithful', 'portfoliocharts.com', 'Golden Butterfly'),
        'RiskParity_3X':    ('proxy',    'bogleheads.org', 'Hedgefundie'),
    }

    def _registry(self):
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            import main
        return main.ALL_STRATEGIES

    def test_the_table_covers_the_registry_exactly(self):
        """Both directions: an unregistered pinned key is as stale as an unpinned registered
        one. A new entry must land in this table or fail here, which is the point."""
        registry = set(self._registry())
        pinned = set(self.EXPECTED)
        self.assertEqual(registry, pinned,
                         f'unpinned: {sorted(registry - pinned)}; '
                         f'stale pins: {sorted(pinned - registry)}')

    def test_every_label_and_citation_matches_its_pin(self):
        registry = self._registry()
        for key, (fidelity, tok1, tok2) in self.EXPECTED.items():
            with self.subTest(key=key):
                s = registry[key]()
                self.assertEqual(s.fidelity, fidelity,
                                 f'{key}: fidelity {s.fidelity!r} != pinned {fidelity!r} — '
                                 f'if the code is right, re-read the paper and update the '
                                 f'pin IN THE SAME COMMIT with the evidence in its message')
                for tok in (tok1, tok2):
                    self.assertIn(tok, s.source,
                                  f'{key}: source {s.source!r} lacks pinned token {tok!r}')

    def test_every_cited_ssrn_paper_is_in_the_repo(self):
        """A citation is only checkable if the PDF it points at is actually here."""
        import re
        papers_dir = os.path.join(os.path.dirname(__file__), '..', 'academic-papers')
        local = ''.join(os.listdir(papers_dir))
        cited = {m for _, t1, t2 in self.EXPECTED.values()
                 for tok in (t1, t2)
                 for m in re.findall(r'SSRN (\d+)', tok)}
        for ssrn in sorted(cited):
            self.assertIn(ssrn, local,
                          f'SSRN {ssrn} is cited by a fidelity pin but has no PDF in '
                          f'academic-papers/')


class TestDocsQuoteTheRegistrySize(unittest.TestCase):
    """The standing docs must not contradict the registry they describe.

    On 2026-07-30 an external audit found seven locations still saying '25 registry keys'
    and two saying 'the registry is all-2x' — both true on 2026-07-28 and false the next
    day, in the two files every reader is instructed to consult FIRST. Prose that quotes a
    count is a cache of the registry, and caches need invalidation.
    """

    DOCS = ('README.md', 'AGENTS.md', 'ARCHITECTURE.md', 'KNOWN_GAPS.md')

    def _read(self, name):
        root = os.path.join(os.path.dirname(__file__), '..')
        with open(os.path.join(root, name), encoding='utf-8') as fh:
            return fh.read()

    def test_no_doc_claims_the_registry_is_all_2x(self):
        for name in self.DOCS:
            with self.subTest(doc=name):
                text = self._read(name)
                self.assertNotIn('registry is all-2x', text,
                                 f'{name} still claims the registry is all-2x; twelve 3x '
                                 f'entries are registered under role=exploratory')

    def test_every_quoted_registry_count_matches_the_registry(self):
        """Catches every phrasing that HAS gone stale, not every number in prose.

        The original patterns pinned only 'all N registry keys' — and the count drifted
        again within a day of that test landing, in phrasings it did not cover ('35
        variants over ~18 years', 'grown to 35' — AUD-01, the fifth instance of this
        defect class). Each new escape gets its pattern added here.
        """
        import io, re, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            import main
        n = len(main.ALL_STRATEGIES)
        pats = [
            re.compile(r'(?:asserted for all|asserts? all|all)\s+(\d+)\s+'
                       r'(?:registry keys|registered|have one|by `tests)'),
            re.compile(r'(?:grown to|registry (?:of|holds|has))\s+(\d+)\b'),
            re.compile(r'\b(\d+)\s+variants over'),
            re.compile(r'\b(\d+)\s+(?:registered\s+)?(?:variants|entries) across'),
        ]
        for name in self.DOCS:
            with self.subTest(doc=name):
                text = self._read(name)
                for pat in pats:
                    for m in pat.finditer(text):
                        self.assertEqual(
                            int(m.group(1)), n,
                            f'{name} quotes a registry count of {m.group(1)}; '
                            f'the registry holds {n}. Context: '
                            f'...{m.string[max(0, m.start()-60):m.end()+20]}...')


if __name__ == '__main__':
    unittest.main()
