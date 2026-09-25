"""
Tests for the Canadian execution mapping (`common/ca_mapping.py`).

The anchor outside the code, in the sense of `tests/test_anchors.py`: the signal universe is
read from `strategies/haa.py` — the module under test does not get to say what HAA_G12 can
hold. Every coverage assertion therefore fails if EITHER side drifts, which is the point. A
mapping table that defined its own scope would agree with itself forever.

The hedging and exchange assertions are pinned to LITERAL values written out here rather
than to the module's own constants, so that widening `CANADIAN_EXCHANGES` cannot quietly
widen the test that is supposed to police it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import ca_mapping as cam
from strategies.haa import HAA_12, HAA_Balanced


class TestTheTableObeysItsOwnRules(unittest.TestCase):

    def test_every_execution_ticker_is_listed_in_canada(self):
        """RULE 1. Literal list, not the module's constant — see the module docstring."""
        for key, entry in cam.CA_MAPPING.items():
            self.assertIn(entry.exchange, ('TSX', 'Cboe CA'),
                          f'{key} -> {entry.exchange} is not a Canadian listing')
        cam.assert_listed_in_canada()

    def test_three_entries_are_on_cboe_canada_not_the_tsx(self):
        """Pinned because it was first recorded wrong, and because it is the kind of detail a
        broker's order ticket cares about. ZTM in particular was believed to be TSX-listed;
        its ETF Facts says Cboe CA."""
        cboe = sorted(k for k, e in cam.CA_MAPPING.items() if e.exchange == 'Cboe CA')
        self.assertEqual(cboe, ['DBC', 'IEF', 'TLT'])

    def test_nothing_is_currency_hedged(self):
        """RULE 2."""
        for key, entry in cam.CA_MAPPING.items():
            self.assertFalse(entry.hedged, f'{key} maps to a hedged product')
        cam.assert_unhedged()

    def test_a_hedged_entry_is_refused(self):
        """The guard must FAIL against the defect, not merely pass on clean data. The cp1252
        regression test passed against the bug it was written for; that is why this exists."""
        from dataclasses import replace
        broken = dict(cam.CA_MAPPING)
        broken['IWM'] = replace(broken['IWM'], exec_ticker_cad='XSU', hedged=True)
        with self.assertRaises(ValueError) as ctx:
            cam.assert_unhedged(broken)
        self.assertIn('IWM', str(ctx.exception))

    def test_an_exchange_outside_canada_is_refused(self):
        from dataclasses import replace
        broken = dict(cam.CA_MAPPING)
        broken['SPY'] = replace(broken['SPY'], exchange='NYSE Arca')
        with self.assertRaises(ValueError):
            cam.assert_listed_in_canada(broken)


class TestNoUsTickerCanReachAnOrder(unittest.TestCase):

    def test_the_real_execution_tickers_pass(self):
        cam.assert_no_us_tickers(sorted(cam.all_execution_tickers()))

    def test_a_us_ticker_in_an_order_list_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            cam.assert_no_us_tickers(['ZSP', 'SPY', 'CGL.C'])
        self.assertIn('SPY', str(ctx.exception))

    def test_an_unknown_canadian_looking_ticker_is_also_refused(self):
        """Rejects by default. A guard that only knows a list of bad tickers approves every
        ticker nobody thought of, and the set of tickers nobody thought of is unbounded."""
        with self.assertRaises(ValueError):
            cam.assert_no_us_tickers(['ZSP', 'XIU'])

    def test_both_unit_classes_are_admissible(self):
        """`.U` classes are the same Canadian trust in a different trading currency, so they
        must not trip the US-ticker guard on the strength of looking unusual."""
        cam.assert_no_us_tickers(['ZSP.U', 'ZTL.U', 'UBIL.U', 'XEC.U'])


class TestTheMappingCoversTheSignalUniverse(unittest.TestCase):
    """RULE 4, anchored on `strategies/haa.py` rather than on the table itself."""

    def setUp(self):
        self.strat = HAA_12()

    def test_it_covers_exactly_what_haa_g12_can_hold(self):
        cam.validate_covers(self.strat)

    def test_the_universe_is_the_thirteen_tradeable_tickers(self):
        sleeves = self.strat.sleeves()
        tradeable = set(sleeves['offensive']) | set(sleeves['defensive'])
        self.assertEqual(sorted(tradeable), sorted(cam.CA_MAPPING))
        self.assertEqual(len(cam.CA_MAPPING), 13)

    def test_ief_is_mapped_once_although_it_is_dual_role(self):
        """IEF is an offensive candidate AND one of two defensive candidates. The canary
        picks the role each month; the instrument is the same either way. Two entries would
        be two names for one position and would break netting."""
        sleeves = self.strat.sleeves()
        self.assertIn('IEF', sleeves['offensive'])
        self.assertIn('IEF', sleeves['defensive'])
        self.assertEqual(cam.CA_MAPPING['IEF'].exec_ticker_cad, 'ZTM')

    def test_the_canary_has_no_execution_image(self):
        """TIP is read, never traded. A mapping for it is a route to buying the thermometer."""
        self.assertEqual(self.strat.sleeves()['canary'], ['TIP'])
        self.assertNotIn('TIP', cam.CA_MAPPING)
        self.assertIn('TIP', cam.CANARY_ONLY)

    def test_mapping_the_canary_is_refused(self):
        from dataclasses import replace
        broken = dict(cam.CA_MAPPING)
        broken['TIP'] = replace(broken['IEF'], signal_ticker='TIP')
        with self.assertRaises(ValueError) as ctx:
            cam.validate_covers(self.strat, broken)
        self.assertIn('TIP', str(ctx.exception))

    def test_a_missing_image_is_refused_rather_than_dropped(self):
        broken = {k: v for k, v in cam.CA_MAPPING.items() if k != 'GLD'}
        with self.assertRaises(ValueError) as ctx:
            cam.validate_covers(self.strat, broken)
        self.assertIn('GLD', str(ctx.exception))

    def test_a_stale_extra_row_is_refused(self):
        """A row for something the strategy cannot hold is a route to an order nothing
        authorised. Tested on a table with ONE surplus row, because the real table pointed at
        a different strategy trips the missing-image check first (see the test below) and a
        guard that raises for the other reason proves nothing about this one."""
        from dataclasses import replace
        broken = dict(cam.CA_MAPPING)
        broken['HYG'] = replace(broken['LQD'], signal_ticker='HYG')
        with self.assertRaises(ValueError) as ctx:
            cam.validate_covers(self.strat, broken)
        self.assertIn('HYG', str(ctx.exception))

    def test_the_table_is_specific_to_haa_g12_and_says_so_when_pointed_elsewhere(self):
        """This table is HAA_G12's. HAA_G8 holds VEA, which G12 does not, so the same table
        must refuse rather than map eight of nine tickers and drop the ninth."""
        with self.assertRaises(ValueError) as ctx:
            cam.validate_covers(HAA_Balanced())
        self.assertIn('VEA', str(ctx.exception))


class TestLiquidityNeverFailsOpen(unittest.TestCase):
    """RULE 5. An empty flag list means "checked, clean" and must never mean "not checked"."""

    def test_zcom_reports_unknown_not_clean(self):
        flags = cam.liquidity_flags(cam.CA_MAPPING['DBC'])
        self.assertTrue(flags, 'ZCOM publishes no volume or spread; silence would read as OK')
        self.assertTrue(all(sev == 'unknown' for sev, _ in flags))
        self.assertTrue(any('could NOT be checked' in msg for _, msg in flags))

    def test_a_fully_reported_entry_is_clean(self):
        self.assertEqual(cam.liquidity_flags(cam.CA_MAPPING['SPY']), [])

    def test_an_oversized_order_is_flagged(self):
        entry = cam.CA_MAPPING['IEF']              # 7,553 units/day
        self.assertEqual(cam.liquidity_flags(entry, shares=2_000), [])
        flags = cam.liquidity_flags(entry, shares=5_000)
        self.assertEqual([sev for sev, _ in flags], ['warn'])

    def test_an_oversized_order_on_an_unreported_product_is_unknown_not_clean(self):
        """The dangerous combination: a big order in the one product whose turnover nobody
        publishes. It must not come back silent."""
        flags = cam.liquidity_flags(cam.CA_MAPPING['DBC'], shares=1_000_000)
        self.assertTrue(any(sev == 'unknown' for sev, _ in flags))

    def test_the_wide_usd_classes_are_flagged(self):
        """ZTL.U 0.42%, ZIC.U 0.36%, XEC.U 0.34% — all above the 0.25% threshold, and all on
        the routing path `prefer_usd_units` would take."""
        for key in ('TLT', 'LQD', 'VWO'):
            flags = cam.liquidity_flags(cam.CA_MAPPING[key], use_usd_class=True)
            self.assertTrue(any(sev == 'warn' and 'spread' in msg for sev, msg in flags),
                            f'{key} USD class should be flagged on spread')


class TestTheDeviationRegisterIsComplete(unittest.TestCase):
    """The scope `tools/ca_execution_gap.py` must cover. Pinned so the report cannot quietly
    discover its own scope, and so a later mapping change has to restate the claim."""

    def test_seven_lines_execute_a_different_asset_from_the_one_signalled(self):
        self.assertEqual(sorted(cam.deviations()),
                         ['DBC', 'IEF', 'IWM', 'LQD', 'VGK', 'VNQ', 'VWO'])

    def test_the_matched_lines_declare_no_deviation(self):
        for key in ('SPY', 'QQQ', 'EWJ', 'GLD', 'TLT', 'BIL'):
            self.assertIsNone(cam.CA_MAPPING[key].deviation,
                              f'{key} was recorded as a like-for-like match')


class TestRowProvenance(unittest.TestCase):
    """Every figure came from a document, and the documents have different reporting periods.
    A table of liquidity numbers with one implied date is prose caching a computation — the
    disease `common/facts.py` exists to treat."""

    def test_every_row_names_its_document_and_its_reporting_date(self):
        for key, entry in cam.CA_MAPPING.items():
            self.assertTrue(entry.source.startswith('https://'), f'{key} has no source URL')
            self.assertRegex(entry.facts_as_of, r'^\d{4}-\d{2}-\d{2}$', f'{key} as-of')
            self.assertRegex(entry.inception_date, r'^\d{4}-\d{2}-\d{2}$', f'{key} inception')

    def test_the_reporting_dates_really_do_differ(self):
        """If they ever collapse to one value, someone has normalised them by hand."""
        self.assertGreater(len({e.facts_as_of for e in cam.CA_MAPPING.values()}), 1)

    def test_every_entry_has_at_least_one_tradeable_unit_class(self):
        for key, entry in cam.CA_MAPPING.items():
            self.assertTrue(cam.execution_tickers(entry), f'{key} maps to nothing')

    def test_bil_has_no_cad_class_and_says_so(self):
        """USD-only, which is why its CAD liquidity fields are None. Structural, not missing
        data: every defensive rotation from CAD cash needs a conversion."""
        entry = cam.CA_MAPPING['BIL']
        self.assertIsNone(entry.exec_ticker_cad)
        self.assertEqual(entry.exec_ticker_usd, 'UBIL.U')
        self.assertEqual(entry.trade_currency, 'USD')


if __name__ == '__main__':
    unittest.main()
