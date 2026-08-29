"""Guardrail tests for FLUSH_ROUND_UP_BAND_PCT in main.size_positions (live order sizing).

Scenario: one account of $10,000, one position SPY @ $300 targeting 50% ($5,000).
A whole-lot floor buys 16 shares ($4,800), stranding $200 of cash. Rounding up to 17
shares deploys $5,100 — a +1.0% overshoot of the 50% target, still inside the account's
usable balance. The band decides whether that round-up is allowed.
"""
import io
import contextlib
import unittest
from types import SimpleNamespace

import pandas as pd

import main

SIGNAL_DATE = pd.Timestamp('2026-06-30')


def deploy(flush_band, include_key=True, safety_pct=1.0):
    """Run size_positions on the scenario; return cash deployed from the account."""
    prices = pd.DataFrame({'SPY': [300.0]}, index=[SIGNAL_DATE])
    alloc = pd.DataFrame({'SPY': [0.5]}, index=[SIGNAL_DATE])
    s_w = pd.DataFrame({'SPY': [1.0]}, index=[SIGNAL_DATE])
    accounts = [{'account_name': 'A', 'account_balance': 10000.0,
                 'initial_balance': 10000.0, 'account_priority': 1}]
    config = {'SAFETY_MARGIN_PCT': safety_pct, 'MINIMUM_TRADE_PCT': 0.5,
              'FLEXIBILITY_BAND_PCT': 0.0, 'FRACTIONAL_SHARES': False, 'SHARE_LOT_SIZE': 1}
    if include_key:
        config['FLUSH_ROUND_UP_BAND_PCT'] = flush_band
    # `_sleeves` supplied because size_positions resolves sleeves through the checked
    # declaration since 2026-07-30 (REPORT-001) — a stub is not exempt from the contract.
    strat = SimpleNamespace(name='TEST', offensive=['SPY'], defensive=[], canary=[],
                            _sleeves=lambda: (set(), {'SPY'}, []))
    with contextlib.redirect_stdout(io.StringIO()):
        main.size_positions(alloc, prices, SIGNAL_DATE, accounts, config, strat, s_w)
    deployed = 10000.0 - accounts[0]['account_balance']
    reserve = 10000.0 * (safety_pct / 100.0)
    # Hard invariant: the safety reserve is never touched.
    assert accounts[0]['account_balance'] >= reserve - 1e-6, "breached safety reserve!"
    return deployed


class TestFlushRoundUp(unittest.TestCase):
    def test_off_by_default_floors_down(self):
        # band 0 => plain floor: 16 shares = $4,800
        self.assertAlmostEqual(deploy(0.0), 4800.0, places=2)

    def test_missing_key_behaves_as_off(self):
        # config without the key must reproduce legacy behavior (config.get default)
        self.assertAlmostEqual(deploy(0.0, include_key=False), 4800.0, places=2)

    def test_band_allows_round_up(self):
        # +1.0% overshoot <= 2.0% band => round up to 17 shares = $5,100
        self.assertAlmostEqual(deploy(2.0), 5100.0, places=2)

    def test_band_too_tight_blocks_round_up(self):
        # +1.0% overshoot > 0.5% band => stays at 16 shares
        self.assertAlmostEqual(deploy(0.5), 4800.0, places=2)

    def test_round_up_bounded_to_one_lot(self):
        # even a huge band only ever adds a single lot (never over-deploys)
        self.assertAlmostEqual(deploy(100.0), 5100.0, places=2)

    def test_never_breaches_reserve_when_cash_is_tight(self):
        # reserve so large the extra lot won't fit => must NOT round up (invariant asserted inside)
        # usable = 10000 - 10000*49% = 5100; val_up = 5100 fits exactly, so it rounds up.
        # push reserve to 49.5% => usable = 5050 < 5100 => round-up blocked, stays at 16.
        self.assertAlmostEqual(deploy(100.0, safety_pct=49.5), 4800.0, places=2)


if __name__ == '__main__':
    unittest.main()
