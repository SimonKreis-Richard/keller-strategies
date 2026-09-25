"""Price-cap sizing tests (PRICE_CAP_MARGIN_PCT).

For after-hours GTC limit orders (e.g. IBKR Midprice with a price cap), shares are
sized AT the cap = quote × (1 + margin) — the worst-case fill — so the broker's
funds check (shares × cap ≤ available cash) passes by construction and the order
can never be rejected for insufficient funds.

Scenario: one $10,000 account, SPY quoted at $300, targeted at 50% ($5,000).
  cap off          → 16 shares @ $300  = $4,800 budgeted
  cap margin 2%    → cap $306, 16 shares (floor(5000/306)) = $4,896 budgeted at cap
  tight case       → the cap budget, not the quote, must drive account math
"""
import unittest
from types import SimpleNamespace

import pandas as pd

import main

SIGNAL_DATE = pd.Timestamp('2026-06-30')


def _sizing(cap_margin, balance=10000.0, target=0.5, quote=300.0):
    prices = pd.DataFrame({'SPY': [quote]}, index=[SIGNAL_DATE])
    alloc = pd.DataFrame({'SPY': [target]}, index=[SIGNAL_DATE])
    s_w = pd.DataFrame({'SPY': [1.0]}, index=[SIGNAL_DATE])
    accounts = [{'account_name': 'A', 'account_balance': balance,
                 'initial_balance': balance, 'account_priority': 1}]
    config = {'SAFETY_MARGIN_PCT': 0.0, 'MINIMUM_TRADE_PCT': 0.5,
              'FLEXIBILITY_BAND_PCT': 0.0, 'FLUSH_ROUND_UP_BAND_PCT': 0.0,
              'PRICE_CAP_MARGIN_PCT': cap_margin,
              'FRACTIONAL_SHARES': False, 'SHARE_LOT_SIZE': 1}
    # `_sleeves` supplied because size_positions resolves sleeves through the checked
    # declaration since 2026-07-30 (REPORT-001) — a stub is not exempt from the contract.
    strat = SimpleNamespace(name='TEST', offensive=['SPY'], defensive=[], canary=[],
                            _sleeves=lambda: (set(), {'SPY'}, []))
    result = main.size_positions(alloc, prices, SIGNAL_DATE, accounts, config, strat, s_w)
    return result, accounts[0]


class TestPriceCap(unittest.TestCase):
    def test_cap_off_keeps_quote_sizing(self):
        result, _ = _sizing(0.0)
        o = result['orders'][0]
        self.assertIsNone(o['cap'])
        self.assertEqual(o['shares'], 16.0)          # floor(5000/300)
        self.assertEqual(o['value'], 4800.0)

    def test_cap_prices_and_budgets_at_ceiling(self):
        result, acc = _sizing(2.0)
        o = result['orders'][0]
        self.assertEqual(o['cap'], 306.0)            # 300 × 1.02
        self.assertEqual(o['price'], 300.0)          # quote still reported
        self.assertEqual(o['shares'], 16.0)          # floor(5000/306)
        self.assertEqual(o['value'], 16 * 306.0)     # max cost budgeted at the cap
        # Account debited at worst case → funds check can never fail at fill time
        self.assertAlmostEqual(acc['account_balance'], 10000.0 - 16 * 306.0, places=2)

    def test_worst_case_total_never_exceeds_balance(self):
        # 100% target of the whole account: at the cap, shares must shrink so that
        # shares × cap ≤ balance (this is exactly the broker's rejection criterion).
        result, acc = _sizing(2.0, balance=10000.0, target=1.0)
        o = result['orders'][0]
        self.assertLessEqual(o['shares'] * o['cap'], 10000.0 + 1e-9)
        self.assertGreaterEqual(acc['account_balance'], -1e-9)


if __name__ == '__main__':
    unittest.main()
