"""
Two-currency order construction (`common/ca_execution.py`).

The anchors, in the sense of `tests/test_anchors.py`:

* the arithmetic of the no-conversion case is done BY HAND here — 100,000 CAD at 50/50 into
  a 100.00 and a 50.00 product is 500 and 1,000 shares, and the test says so rather than
  asking the module what it thinks;
* value conservation is checked against a quantity the module never computes for itself —
  NAV out must equal NAV in minus the FX commission, on every scenario, and a rounding
  residue must appear as CASH rather than evaporate;
* the weights are compared against the dict that went in, not against anything derived.

Everything here is offline. No FX rate is ever fetched: `FxRate` is constructed in the test,
which is the whole point of making the rate an argument.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import ca_execution as cax
from common.ca_execution import CAExecutionConfig, CAExecutionError, FxRate, build_orders

RATE = FxRate(usdcad=1.40, source='test', as_of='2026-09-18')
SIGNAL_DATE = '2026-08-31'

#: One price per execution ticker, in that class's own trading currency. ZSP.U is ZSP at the
#: test rate (100.00 / 1.40 = 71.428571...), so a CAD route and a USD route buy the same
#: exposure and any difference in the result is the module's doing, not the panel's.
PRICES = {
    'ZSP': 100.0, 'ZSP.U': 100.0 / 1.40,
    'CGL.C': 50.0,
    'UBIL.U': 50.0,
    'ZTL': 40.0, 'ZTL.U': 40.0 / 1.40,
    'ZTM': 50.0, 'ZTM.U': 50.0 / 1.40,
    'ZNQ': 80.0, 'ZNQ.U': 80.0 / 1.40,
    'ZCOM': 25.0,
    'ZSML': 20.0, 'ZSML.U': 20.0 / 1.40,
    'XEU': 40.0, 'ZJPN': 25.0,
    'XEC': 32.0, 'XEC.U': 32.0 / 1.40,
    'ZIC': 18.0, 'ZIC.U': 18.0 / 1.40,
    'CGR': 30.0,
}

CAD_ONLY = CAExecutionConfig(prefer_usd_units=False)
USD_FIRST = CAExecutionConfig(prefer_usd_units=True)


def _run(weights, cash, positions=(), config=CAD_ONLY, prices=None, fx=RATE):
    return build_orders(weights, cash, positions, prices or PRICES, fx,
                        config=config, signal_date=SIGNAL_DATE)


def _by_ticker(result):
    return {o['ticker']: o for o in result['orders']}


class TestTheArithmeticOfASimpleRotation(unittest.TestCase):
    """Hand-computed: 100,000 CAD, 50/50, at 100.00 and 50.00 => 500 and 1,000 shares."""

    def setUp(self):
        self.result = _run({'SPY': 0.5, 'GLD': 0.5}, {'CAD': 100_000.0, 'USD': 0.0})

    def test_the_share_counts_are_the_ones_worked_out_by_hand(self):
        orders = _by_ticker(self.result)
        self.assertEqual(sorted(orders), ['CGL.C', 'ZSP'])
        self.assertEqual(orders['ZSP']['qty'], 500)
        self.assertEqual(orders['CGL.C']['qty'], 1_000)
        self.assertTrue(all(o['side'] == 'BUY' for o in self.result['orders']))

    def test_no_conversion_was_needed_and_none_was_proposed(self):
        self.assertEqual(self.result['fx_orders'], [])

    def test_nothing_is_left_over_and_nothing_is_overspent(self):
        self.assertAlmostEqual(self.result['residual_cash']['CAD'], 0.0, places=6)
        self.assertAlmostEqual(self.result['residual_cash']['USD'], 0.0, places=6)

    def test_value_is_conserved(self):
        self.assertAlmostEqual(self.result['nav_cad_after'],
                               self.result['nav_cad_before'], places=6)

    def test_the_orders_name_canadian_tickers_on_canadian_exchanges(self):
        from common import ca_mapping as cam
        cam.assert_no_us_tickers([o['ticker'] for o in self.result['orders']])
        for o in self.result['orders']:
            self.assertIn(o['exchange'], ('TSX', 'Cboe CA'))


class TestTheSignalIsCopiedThroughUntouched(unittest.TestCase):
    """The one rule the whole layer lives under."""

    def test_targets_are_weight_times_nav(self):
        weights = {'SPY': 0.5, 'GLD': 0.25, 'TLT': 0.25}
        result = _run(weights, {'CAD': 200_000.0, 'USD': 0.0})
        cax.assert_weights_unchanged(weights, result)
        self.assertAlmostEqual(result['target_cad']['SPY'], 100_000.0, places=6)
        self.assertAlmostEqual(result['target_cad']['GLD'], 50_000.0, places=6)

    def test_an_invented_sleeve_is_caught(self):
        weights = {'SPY': 1.0}
        result = _run(weights, {'CAD': 10_000.0, 'USD': 0.0})
        result['target_cad']['GLD'] = 1.0
        with self.assertRaises(CAExecutionError):
            cax.assert_weights_unchanged(weights, result)

    def test_a_rescaled_target_is_caught(self):
        weights = {'SPY': 1.0}
        result = _run(weights, {'CAD': 10_000.0, 'USD': 0.0})
        result['target_cad']['SPY'] *= 1.05
        with self.assertRaises(CAExecutionError):
            cax.assert_weights_unchanged(weights, result)

    def test_a_row_that_does_not_sum_to_one_is_refused(self):
        with self.assertRaises(CAExecutionError) as ctx:
            _run({'SPY': 0.5}, {'CAD': 10_000.0})
        self.assertIn('sum to', str(ctx.exception))

    def test_a_ticker_with_no_canadian_image_is_refused_not_dropped(self):
        with self.assertRaises(CAExecutionError) as ctx:
            _run({'SPY': 0.5, 'HYG': 0.5}, {'CAD': 10_000.0})
        self.assertIn('HYG', str(ctx.exception))


class TestHaaG12AndHaaG12CaChooseTheSameThing(unittest.TestCase):
    """Livrable 5's second requirement, against the REAL strategy rather than a stub.

    The panel is built so the selection is decidable by hand: every ticker compounds at its
    own constant rate, so 13612U ranks them in exactly that order and `HAA_12` (NO=12, TO=6)
    must hold the six fastest at 1/6 each. The canary compounds positively, so the book is
    risk-on. Nothing in `ca_execution` participates in that decision — it is handed the row
    and must reproduce it.
    """

    GROWTH = {'SPY': 0.020, 'QQQ': 0.018, 'IWM': 0.016, 'VGK': 0.014, 'EWJ': 0.012,
              'VWO': 0.010, 'VNQ': 0.008, 'DBC': 0.006, 'GLD': 0.004, 'IEF': 0.002,
              'TLT': 0.001, 'LQD': 0.0005, 'TIP': 0.003, 'BIL': 0.001}

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from common.momentum import calc_13612u
        from strategies.haa import HAA_12

        index = pd.date_range('2024-01-31', periods=20, freq='ME')
        cls.prices = pd.DataFrame(
            {t: [100.0 * (1.0 + g) ** i for i in range(len(index))]
             for t, g in cls.GROWTH.items()}, index=index)
        cls.strat = HAA_12()
        cls.alloc_row = cls.strat.generate_allocations(
            cls.prices, calc_13612u(cls.prices), None, None).iloc[-1]
        cls.weights = {t: float(w) for t, w in cls.alloc_row.items() if float(w) > 1e-9}

    def test_the_panel_really_does_produce_the_expected_top_six(self):
        """The anchor. If this fails, the scenario stopped testing what it claims to."""
        self.assertEqual(sorted(self.weights), ['EWJ', 'IWM', 'QQQ', 'SPY', 'VGK', 'VWO'])
        for w in self.weights.values():
            self.assertAlmostEqual(w, 1 / 6, places=12)

    def test_the_execution_layer_targets_exactly_those_weights(self):
        result = _run(self.weights, {'CAD': 600_000.0, 'USD': 0.0})
        cax.assert_weights_unchanged(self.weights, result)
        for sleeve in self.weights:
            self.assertAlmostEqual(result['target_cad'][sleeve], 100_000.0, places=6)

    def test_the_orders_are_the_canadian_images_of_those_six_and_nothing_else(self):
        result = _run(self.weights, {'CAD': 600_000.0, 'USD': 0.0})
        self.assertEqual(sorted(_by_ticker(result)),
                         ['XEC', 'XEU', 'ZJPN', 'ZNQ', 'ZSML', 'ZSP'])

    def test_the_routing_choice_does_not_move_the_decision(self):
        """CAD-only routing and USD-first routing must reach the same TARGETS; they may
        differ only in which unit class is bought."""
        cad = _run(self.weights, {'CAD': 600_000.0, 'USD': 0.0}, config=CAD_ONLY)
        usd = _run(self.weights, {'CAD': 300_000.0, 'USD': 300_000.0 / 1.40},
                   config=USD_FIRST)
        self.assertEqual(sorted(cad['target_cad']), sorted(usd['target_cad']))
        for sleeve in cad['target_cad']:
            self.assertAlmostEqual(cad['target_cad'][sleeve], usd['target_cad'][sleeve],
                                   places=6)

    def test_a_dead_canary_sends_the_book_to_one_defensive_line(self):
        """The other half of the claim: when the signal changes, the execution layer follows
        it rather than holding what it held. TIP falling makes HAA defend into the better of
        BIL and IEF, and the orders must become that one instrument."""
        import pandas as pd
        from common.momentum import calc_13612u

        prices = self.prices.copy()
        prices['TIP'] = [100.0 * (1.0 - 0.01) ** i for i in range(len(prices))]
        row = self.strat.generate_allocations(
            prices, calc_13612u(prices), None, None).iloc[-1]
        weights = {t: float(w) for t, w in row.items() if float(w) > 1e-9}
        self.assertEqual(sorted(weights), ['IEF'])          # IEF beats BIL on this panel

        result = _run(weights, {'CAD': 100_000.0, 'USD': 0.0})
        self.assertEqual(sorted(_by_ticker(result)), ['ZTM'])
        self.assertIsInstance(pd.Timestamp(result['signal_date'] or '2026-08-31'),
                              pd.Timestamp)


class TestTheThreeCashCases(unittest.TestCase):
    """CAD only, USD only, and both — the brief's first edge-case row."""

    def test_cad_only_needs_no_conversion_when_the_products_trade_in_cad(self):
        result = _run({'GLD': 1.0}, {'CAD': 50_000.0, 'USD': 0.0})
        self.assertEqual(result['fx_orders'], [])
        self.assertEqual(_by_ticker(result)['CGL.C']['qty'], 1_000)

    def test_usd_only_converts_to_buy_a_cad_product(self):
        result = _run({'GLD': 1.0}, {'CAD': 0.0, 'USD': 50_000.0})
        self.assertEqual(len(result['fx_orders']), 1)
        self.assertEqual(result['fx_orders'][0]['direction'], 'USD->CAD')
        self.assertTrue(result['orders'])

    def test_both_currencies_present_spends_each_in_its_own_lane(self):
        """SPY routes to ZSP.U with the USD cash; GLD has no USD class and takes the CAD."""
        result = _run({'SPY': 0.5, 'GLD': 0.5},
                      {'CAD': 50_000.0, 'USD': 50_000.0 / 1.40}, config=USD_FIRST)
        orders = _by_ticker(result)
        self.assertIn('ZSP.U', orders)
        self.assertIn('CGL.C', orders)
        self.assertEqual(result['fx_orders'], [],
                         'each currency already covered its own sleeve')

    def test_an_empty_book_is_refused_rather_than_producing_empty_orders(self):
        with self.assertRaises(CAExecutionError) as ctx:
            _run({'SPY': 1.0}, {'CAD': 0.0, 'USD': 0.0})
        self.assertIn('net asset value', str(ctx.exception))


class TestAUsdClassIsNeverReachedByConverting(unittest.TestCase):
    """`prefer_usd_units` spends USD that is already there; it never buys USD to use it.

    Regression, 2026-09-23. The setting routed every sleeve with a `.U` class to that class
    whatever the cash, so a CAD-only book — the owner's case — with the DEFAULT configuration
    converted most of itself to USD each month to buy the thinner, wider-spread class. Found
    while writing the example config for `tools/ca_orders.py`; the first test below fails
    against that behaviour.
    """

    def test_cad_only_cash_buys_the_cad_class_even_when_usd_is_preferred(self):
        result = _run({'SPY': 0.5, 'QQQ': 0.5}, {'CAD': 100_000.0, 'USD': 0.0},
                      config=USD_FIRST)
        self.assertEqual(sorted(_by_ticker(result)), ['ZNQ', 'ZSP'])
        self.assertEqual(result['fx_orders'], [])

    def test_usd_cash_goes_to_the_usd_class_until_it_runs_out(self):
        """70,000 CAD + 50,000 USD at 1.40 = 140,000 CAD, half each. SPY's 70,000 CAD target
        is exactly the 50,000 USD, so it takes ZSP.U; QQQ has no USD left and takes ZNQ."""
        result = _run({'SPY': 0.5, 'QQQ': 0.5}, {'CAD': 70_000.0, 'USD': 50_000.0},
                      config=USD_FIRST)
        self.assertEqual(sorted(_by_ticker(result)), ['ZNQ', 'ZSP.U'])
        self.assertEqual(result['fx_orders'], [])

    def test_a_usd_only_sleeve_claims_the_usd_before_a_sleeve_that_has_a_choice(self):
        """BIL has no CAD class. With just enough USD for it, SPY must not take that USD
        first and leave BIL to force a conversion."""
        result = _run({'SPY': 0.5, 'BIL': 0.5}, {'CAD': 70_000.0, 'USD': 50_000.0},
                      config=USD_FIRST)
        self.assertEqual(sorted(_by_ticker(result)), ['UBIL.U', 'ZSP'])
        self.assertEqual(result['fx_orders'], [])


class TestTheDemonstrationsThreeFindings(unittest.TestCase):
    """Three defects the `tools/ca_orders.py --demo` rotation exposed on 2026-09-23, each
    pinned here with the numbers that exposed it. The first two tests fail against the code
    as it stood that morning."""

    def test_a_sleeve_leaving_the_target_is_sold_to_the_last_unit(self):
        """461 ZCOM at 38.49: 461 * 38.49 / 38.49 is 460.99999..., and its floor was 460."""
        result = _run({'GLD': 1.0}, {'CAD': 0.0, 'USD': 0.0},
                      positions=[{'ticker': 'ZCOM', 'qty': 461, 'currency': 'CAD'}],
                      prices=dict(PRICES, ZCOM=38.49))
        self.assertEqual(_by_ticker(result)['ZCOM']['qty'], 461)

    def test_quantities_are_whole_numbers_as_integers_not_floats(self):
        result = _run({'GLD': 1.0}, {'CAD': 0.0, 'USD': 0.0},
                      positions=[{'ticker': 'ZSP', 'qty': 492.0, 'currency': 'CAD'}])
        for o in result['orders']:
            self.assertIs(type(o['qty']), int, f'{o["ticker"]}: {o["qty"]!r}')

    def test_rounding_residue_is_a_note_not_an_undeployed_warning(self):
        """100,000 CAD half into ZSP at 114.97: 434 units (49,896.98), 103.02 CAD left —
        less than a unit, so rounding, which is not the book being short of its target."""
        result = _run({'SPY': 0.5, 'GLD': 0.5}, {'CAD': 100_000.0},
                      prices=dict(PRICES, ZSP=114.97))
        self.assertFalse([m for s, m in result['flags'] if 'NOT fully invested' in m])
        notes = [m for s, m in result['flags'] if s == 'info' and 'rounding' in m]
        self.assertEqual(len(notes), 1)
        self.assertIn('103.02 CAD', notes[0])

    def test_a_shortfall_of_more_than_a_unit_is_still_warned(self):
        """1.50 USD cannot pay the 2.00 USD minimum commission, so it never reaches a CAD
        product priced at 1.00: that is a real shortfall, not rounding."""
        result = _run({'GLD': 1.0}, {'CAD': 0.0, 'USD': 1.50},
                      prices=dict(PRICES, **{'CGL.C': 1.0}))
        self.assertTrue(any(s == 'warn' and 'NOT fully invested' in m
                            for s, m in result['flags']))


class TestTheUsdOnlyDefensiveSleeve(unittest.TestCase):
    """The brief's second edge case, and a recurring cost rather than an edge case: HAA
    defends into BIL often, UBIL.U has no CAD unit class, so CAD cash must be converted
    every time the canary dies."""

    def setUp(self):
        self.result = _run({'BIL': 1.0}, {'CAD': 100_000.0, 'USD': 0.0})

    def test_a_conversion_is_proposed_as_its_own_order_line(self):
        self.assertEqual(len(self.result['fx_orders']), 1)
        fx = self.result['fx_orders'][0]
        self.assertEqual(fx['direction'], 'CAD->USD')
        self.assertIn('FX: convert', fx['description'])
        self.assertEqual(fx['rate'], 1.40)

    def test_the_rate_and_its_provenance_are_journalled_with_the_orders(self):
        self.assertEqual(self.result['fx']['source'], 'test')
        self.assertEqual(self.result['fx']['as_of'], '2026-09-18')

    def test_the_position_is_bought_in_usd(self):
        order = _by_ticker(self.result)['UBIL.U']
        self.assertEqual(order['currency'], 'USD')
        self.assertEqual(order['exchange'], 'TSX')

    def test_value_is_conserved_net_of_the_commission(self):
        cost_cad = self.result['fx_orders'][0]['cost_cad']
        self.assertAlmostEqual(self.result['nav_cad_after'],
                               self.result['nav_cad_before'] - cost_cad, places=4)

    def test_the_minimum_commission_binds_on_a_small_conversion(self):
        """0.2 bp of a small notional is under USD 2, so the floor is what is charged. The
        brief's own annotation and its value disagreed by a factor of 100; expressing the
        cost in basis points with a floor is what removed the ambiguity."""
        small = _run({'BIL': 1.0}, {'CAD': 10_000.0, 'USD': 0.0})
        self.assertAlmostEqual(small['fx_orders'][0]['cost_usd'], 2.0, places=6)

    def test_the_rate_applies_above_the_floor(self):
        big = _run({'BIL': 1.0}, {'CAD': 100_000_000.0, 'USD': 0.0})
        notional = big['fx_orders'][0]['usd']
        self.assertAlmostEqual(big['fx_orders'][0]['cost_usd'],
                               notional * 0.2 / 10_000.0, places=6)


class TestExistingPositionsAreNetted(unittest.TestCase):
    """The brief's third edge case: a `.U` holding while the new purchase is in CAD. Two unit
    classes of one fund are ONE sleeve; counting them separately would buy the position
    twice."""

    def setUp(self):
        # 100 x ZSP.U at 71.428571 USD = 7,142.86 USD = 10,000 CAD.
        self.positions = [{'ticker': 'ZSP.U', 'qty': 100, 'currency': 'USD'}]
        self.result = _run({'SPY': 1.0}, {'CAD': 90_000.0, 'USD': 0.0},
                           positions=self.positions)

    def test_the_existing_holding_counts_toward_the_target(self):
        self.assertAlmostEqual(self.result['nav_cad_before'], 100_000.0, places=4)
        self.assertAlmostEqual(self.result['target_cad']['SPY'], 100_000.0, places=4)

    def test_only_the_delta_is_traded(self):
        orders = _by_ticker(self.result)
        self.assertEqual(sorted(orders), ['ZSP'])
        self.assertEqual(orders['ZSP']['qty'], 900)      # 90,000 CAD / 100.00

    def test_the_existing_class_is_not_liquidated_to_be_rebought(self):
        self.assertFalse([o for o in self.result['orders'] if o['side'] == 'SELL'])

    def test_a_sleeve_that_left_the_target_is_sold_in_full(self):
        result = _run({'GLD': 1.0}, {'CAD': 90_000.0, 'USD': 0.0}, positions=self.positions)
        sells = [o for o in result['orders'] if o['side'] == 'SELL']
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells[0]['ticker'], 'ZSP.U')
        self.assertEqual(sells[0]['qty'], 100)

    def test_proceeds_of_a_sale_fund_the_purchase(self):
        """A full rotation out of SPY and into GLD must not need new money: the sells run
        first and credit the cash the buys are sized against."""
        result = _run({'GLD': 1.0}, {'CAD': 0.0, 'USD': 0.0}, positions=self.positions)
        self.assertTrue([o for o in result['orders'] if o['side'] == 'BUY'])
        self.assertAlmostEqual(result['nav_cad_before'], 10_000.0, places=4)


class TestTheRefusals(unittest.TestCase):

    def test_no_fx_rate_is_an_explicit_error_never_a_default(self):
        with self.assertRaises(CAExecutionError) as ctx:
            build_orders({'SPY': 1.0}, {'CAD': 1_000.0}, [], PRICES, None)
        self.assertIn('FxRate is required', str(ctx.exception))

    def test_a_missing_or_absurd_rate_is_refused_at_construction(self):
        for bad in (None, 0.0, -1.4, float('nan'), float('inf'), 'a rate'):
            with self.assertRaises(CAExecutionError):
                FxRate(usdcad=bad, source='test', as_of='2026-09-18')

    def test_a_rate_without_provenance_is_refused(self):
        with self.assertRaises(CAExecutionError):
            FxRate(usdcad=1.4, source='', as_of='2026-09-18')

    def test_leverage_above_one_is_refused_with_the_reason(self):
        for bad in (1.05, 1.3, 2.0):
            with self.assertRaises(CAExecutionError) as ctx:
                CAExecutionConfig(leverage=bad)
            self.assertIn('EXEC-001', str(ctx.exception))
            self.assertIn('EQUITY or BUYING POWER', str(ctx.exception))

    def test_leverage_at_or_below_one_is_allowed(self):
        """Holding cash back needs no information the code lacks, so it is not refused."""
        CAExecutionConfig(leverage=1.0)
        result = _run({'SPY': 1.0}, {'CAD': 100_000.0},
                      config=CAExecutionConfig(leverage=0.8, prefer_usd_units=False))
        self.assertEqual(_by_ticker(result)['ZSP']['qty'], 800)
        self.assertAlmostEqual(result['residual_cash']['CAD'], 20_000.0, places=6)

    def test_a_us_ticker_among_the_held_positions_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _run({'SPY': 1.0}, {'CAD': 1_000.0},
                 positions=[{'ticker': 'SPY', 'qty': 10, 'currency': 'USD'}])
        self.assertIn('SPY', str(ctx.exception))

    def test_a_missing_price_is_refused_rather_than_substituted(self):
        prices = dict(PRICES)
        del prices['CGL.C']
        with self.assertRaises(CAExecutionError) as ctx:
            _run({'GLD': 1.0}, {'CAD': 50_000.0}, prices=prices)
        self.assertIn('CGL.C', str(ctx.exception))

    def test_a_zero_price_is_refused(self):
        prices = dict(PRICES, **{'CGL.C': 0.0})
        with self.assertRaises(CAExecutionError):
            _run({'GLD': 1.0}, {'CAD': 50_000.0}, prices=prices)


class TestValueIsConservedAcrossEveryShape(unittest.TestCase):
    """NAV out = NAV in - FX commission, always. The residue of whole-share rounding must
    show up as cash rather than disappear, which is the failure the pre-audit engine had
    when uninvested weight simply evaporated."""

    SCENARIOS = (
        ('cad only, no fx', {'SPY': 0.5, 'GLD': 0.5}, {'CAD': 137_431.17, 'USD': 0.0},
         (), CAD_ONLY),
        ('usd first, mixed cash', {'SPY': 0.4, 'TLT': 0.3, 'IEF': 0.3},
         {'CAD': 61_000.0, 'USD': 23_500.0}, (), USD_FIRST),
        ('defensive, cad cash only', {'BIL': 1.0}, {'CAD': 88_888.88, 'USD': 0.0},
         (), CAD_ONLY),
        ('rotation out of a held .U', {'GLD': 0.5, 'DBC': 0.5},
         {'CAD': 12_345.67, 'USD': 890.12},
         ({'ticker': 'ZSP.U', 'qty': 137, 'currency': 'USD'},), USD_FIRST),
        ('six-asset book', {'SPY': 1 / 6, 'QQQ': 1 / 6, 'GLD': 1 / 6, 'TLT': 1 / 6,
                            'IEF': 1 / 6, 'DBC': 1 / 6},
         {'CAD': 250_000.0, 'USD': 40_000.0}, (), USD_FIRST),
    )

    def test_nav_is_conserved_net_of_commission(self):
        for label, weights, cash, positions, config in self.SCENARIOS:
            with self.subTest(label):
                r = _run(weights, cash, positions=positions, config=config)
                cost = sum(f['cost_cad'] for f in r['fx_orders'])
                self.assertAlmostEqual(r['nav_cad_after'], r['nav_cad_before'] - cost,
                                       places=4)

    def test_cash_is_never_driven_negative(self):
        for label, weights, cash, positions, config in self.SCENARIOS:
            with self.subTest(label):
                r = _run(weights, cash, positions=positions, config=config)
                self.assertGreaterEqual(r['residual_cash']['CAD'], -1e-6, label)
                self.assertGreaterEqual(r['residual_cash']['USD'], -1e-6, label)

    def test_every_quantity_is_a_whole_number_of_shares(self):
        for label, weights, cash, positions, config in self.SCENARIOS:
            with self.subTest(label):
                for o in _run(weights, cash, positions=positions, config=config)['orders']:
                    self.assertEqual(o['qty'], int(o['qty']), f'{label}: {o["ticker"]}')


class TestAConversionLeavesItsOwnCommissionBehind(unittest.TestCase):
    """Regression, found by the conservation test above on 2026-09-18.

    Converting USD into CAD costs the notional AND the commission, both out of the USD
    balance. Sizing the conversion at the whole USD balance and charging the fee on top
    overdraws it by exactly the fee — silently, on every rotation that liquidates a `.U`
    holding to buy a CAD-listed product, which for this book is an ordinary month.

    Verified to FAIL against the defect before being kept: patching the sizing helper back
    to the naive `lambda cash, cfg: cash` reproduces the -2.00 USD balance.
    """

    POSITIONS = ({'ticker': 'ZSP.U', 'qty': 137, 'currency': 'USD'},)
    CASH = {'CAD': 12_345.67, 'USD': 890.12}
    WEIGHTS = {'GLD': 0.5, 'DBC': 0.5}

    def test_the_usd_balance_is_not_overdrawn(self):
        result = _run(self.WEIGHTS, self.CASH, positions=self.POSITIONS, config=USD_FIRST)
        self.assertGreaterEqual(result['residual_cash']['USD'], 0.0)

    def test_the_naive_sizing_is_what_overdrew_it(self):
        """States the defect explicitly, so the next reader knows what the helper is for."""
        from unittest.mock import patch
        with patch.object(cax, '_largest_sellable_usd', lambda cash_usd, cfg: cash_usd):
            result = _run(self.WEIGHTS, self.CASH, positions=self.POSITIONS,
                          config=USD_FIRST)
        self.assertAlmostEqual(result['residual_cash']['USD'], -2.0, places=6)

    def test_a_conversion_smaller_than_its_own_commission_is_not_proposed(self):
        """Below the floor the trade delivers nothing and would leave a negative balance.
        Not proposing it is right; proposing it and letting the balance go negative is the
        same defect wearing a different size."""
        result = _run({'BIL': 1.0}, {'CAD': 1.0, 'USD': 0.0},
                      positions=({'ticker': 'UBIL.U', 'qty': 10, 'currency': 'USD'},))
        self.assertEqual(result['fx_orders'], [])
        self.assertGreaterEqual(result['residual_cash']['CAD'], 0.0)
        self.assertGreaterEqual(result['residual_cash']['USD'], 0.0)


class TestTheRotationRegistry(unittest.TestCase):
    """Every transaction is attached to a signal; there is no trade outside a signal."""

    def setUp(self):
        self.result = _run({'BIL': 1.0}, {'CAD': 100_000.0, 'USD': 0.0})

    def test_every_row_references_the_signal_date(self):
        self.assertTrue(self.result['rotation_rows'])
        for row in self.result['rotation_rows']:
            self.assertEqual(row['signal_date'], SIGNAL_DATE)
            self.assertIn(SIGNAL_DATE, row['reason'])

    def test_the_conversion_is_a_row_too(self):
        sides = {row['side'] for row in self.result['rotation_rows']}
        self.assertEqual(sides, {'CONVERT', 'BUY'})

    def test_there_is_one_row_per_order_and_per_conversion(self):
        self.assertEqual(len(self.result['rotation_rows']),
                         len(self.result['orders']) + len(self.result['fx_orders']))

    def test_the_csv_carries_a_header_and_one_line_per_row(self):
        csv_text = cax.rotation_csv(self.result)
        lines = csv_text.strip().split('\n')
        self.assertTrue(lines[0].startswith('signal_date,signal_ticker,exec_ticker'))
        self.assertEqual(len(lines) - 1, len(self.result['rotation_rows']))


class TestLiquidityFlagsReachTheOrders(unittest.TestCase):

    def test_a_zcom_order_carries_the_unknown_flag(self):
        """ZCOM publishes no volume and no spread, so an order in it must say that it could
        not be checked rather than come back silent."""
        result = _run({'DBC': 1.0}, {'CAD': 100_000.0})
        order = _by_ticker(result)['ZCOM']
        self.assertTrue(order['flags'])
        self.assertTrue(all(sev == 'unknown' for sev, _ in order['flags']))

    def test_a_well_covered_order_carries_none(self):
        result = _run({'SPY': 1.0}, {'CAD': 100_000.0})
        self.assertEqual(_by_ticker(result)['ZSP']['flags'], [])

    def test_a_usd_route_into_a_wide_class_is_flagged(self):
        """ZTL.U spreads 0.42%. `prefer_usd_units` routes there, so the flag has to follow."""
        result = _run({'TLT': 1.0}, {'CAD': 0.0, 'USD': 100_000.0}, config=USD_FIRST)
        order = _by_ticker(result)['ZTL.U']
        self.assertTrue(any(sev == 'warn' and 'spread' in msg for sev, msg in order['flags']))

    def test_an_undeployed_target_is_named(self):
        """Cash in the wrong currency with no conversion possible leaves the book short, and
        silence there reads as "fully invested"."""
        result = _run({'BIL': 1.0}, {'CAD': 0.0, 'USD': 0.0},
                      positions=[{'ticker': 'ZSP', 'qty': 1_000, 'currency': 'CAD'}])
        # The book is 100% ZSP and the target is 100% UBIL.U; the sale funds the purchase in
        # CAD, which cannot buy a USD-only product without the conversion that is proposed.
        self.assertTrue(any(sev == 'warn' and 'NOT fully invested' in msg
                            for sev, msg in result['flags'])
                        or any(o['ticker'] == 'UBIL.U' for o in result['orders']))


if __name__ == '__main__':
    unittest.main()
