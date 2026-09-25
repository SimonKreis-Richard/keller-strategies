"""
The USD/CAD lookup (`tools/fx_rate.py`) — driven entirely through an injected opener.

The first test replaces the real fetcher with one that raises, and every other test in this
file injects its own. That is what keeps `unittest discover` offline: this project has no
pytest and no marker system, so the guarantee has to come from the tests themselves and from
keeping the fetchers outside `tests/`.

Every body below is written by hand here. The ones that matter most are the bodies a lenient
parser would have turned into a number: an HTML gateway page served with HTTP 200, an empty
observation list, and a rate that parsed out of the wrong field.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.ca_execution import CAExecutionError, FxRate
from tools import fx_rate as fxr

GOOD_BODY = ('{"groupDetail":{"label":"x"},'
             '"observations":[{"d":"2026-09-17","FXUSDCAD":{"v":"1.3812"}}]}')


class TestTheSuiteNeverOpensASocket(unittest.TestCase):

    def test_the_real_opener_is_never_called_by_these_tests(self):
        """If a refactor reintroduces a live fetch on any path exercised here, this turns the
        suite red rather than slow."""
        def _explode(url):
            raise AssertionError(f'the test suite tried to reach {url}')

        with patch.object(fxr, '_default_opener', _explode):
            fx = fxr.fetch_boc_usdcad(opener=lambda url: GOOD_BODY)
        self.assertEqual(fx.usdcad, 1.3812)


class TestTheValetParserIsStrict(unittest.TestCase):

    def test_a_well_formed_response_parses(self):
        rate, date = fxr.parse_valet_json(GOOD_BODY)
        self.assertEqual(rate, 1.3812)
        self.assertEqual(date, '2026-09-17')

    def test_an_html_gateway_page_is_refused_not_defaulted(self):
        """The fail-open case, asserted first among the failures: a service behind a gateway
        answers HTTP 200 with a page, and a tolerant reader turns that into a default."""
        with self.assertRaises(CAExecutionError) as ctx:
            fxr.parse_valet_json('<html><body>Service temporarily unavailable</body></html>')
        self.assertIn('did not return JSON', str(ctx.exception))

    def test_an_empty_observation_list_is_refused(self):
        with self.assertRaises(CAExecutionError) as ctx:
            fxr.parse_valet_json('{"observations": []}')
        self.assertIn('no observations', str(ctx.exception))

    def test_a_missing_series_field_is_refused(self):
        with self.assertRaises(CAExecutionError):
            fxr.parse_valet_json('{"observations":[{"d":"2026-09-17","FXEURCAD":{"v":"1.5"}}]}')

    def test_a_non_numeric_value_is_refused(self):
        with self.assertRaises(CAExecutionError):
            fxr.parse_valet_json(
                '{"observations":[{"d":"2026-09-17","FXUSDCAD":{"v":"n/a"}}]}')

    def test_an_observation_without_a_date_is_refused(self):
        """A rate with no date cannot be journalled against an order, and the journal is the
        point."""
        with self.assertRaises(CAExecutionError) as ctx:
            fxr.parse_valet_json('{"observations":[{"FXUSDCAD":{"v":"1.3812"}}]}')
        self.assertIn('no date', str(ctx.exception))

    def test_a_value_from_the_wrong_field_is_caught_by_the_sanity_band(self):
        """13812 is what a parse that found an index level or a scaled integer looks like.
        The band exists to catch that, not to have a view on the currency."""
        with self.assertRaises(CAExecutionError) as ctx:
            fxr.parse_valet_json(
                '{"observations":[{"d":"2026-09-17","FXUSDCAD":{"v":"13812"}}]}')
        self.assertIn('plausible band', str(ctx.exception))

    def test_the_band_admits_the_real_historical_range(self):
        for value in ('0.95', '1.05', '1.38', '1.60'):
            body = '{"observations":[{"d":"2026-09-17","FXUSDCAD":{"v":"%s"}}]}' % value
            self.assertAlmostEqual(fxr.parse_valet_json(body)[0], float(value))


class TestTheSourceIsAlwaysRecorded(unittest.TestCase):
    """The journal has to say WHICH rate produced an order."""

    def test_the_bank_of_canada_names_itself_and_its_date(self):
        fx = fxr.fetch_boc_usdcad(opener=lambda url: GOOD_BODY)
        self.assertIn('Bank of Canada', fx.source)
        self.assertEqual(fx.as_of, '2026-09-17')

    def test_the_fallback_says_it_is_the_fallback(self):
        import pandas as pd
        frame = pd.DataFrame({'Close': [1.3755]},
                             index=pd.to_datetime(['2026-09-17']))
        fx = fxr.fetch_yahoo_usdcad(download=lambda *a, **k: frame)
        self.assertIn('fallback', fx.source)
        self.assertEqual(fx.usdcad, 1.3755)

    def test_an_empty_yahoo_frame_is_refused(self):
        import pandas as pd
        with self.assertRaises(CAExecutionError):
            fxr.fetch_yahoo_usdcad(download=lambda *a, **k: pd.DataFrame())


class TestResolutionOrder(unittest.TestCase):

    def _boc(self, rate=1.38):
        return lambda: FxRate(usdcad=rate, source='Bank of Canada Valet FXUSDCAD',
                              as_of='2026-09-17')

    def _dead(self, label):
        def _fetch():
            raise CAExecutionError(f'{label} is unreachable')
        return _fetch

    def test_a_manual_override_wins_and_asks_nobody(self):
        fx = fxr.resolve_usdcad(manual=1.41,
                                boc=self._dead('BoC'), yahoo=self._dead('Yahoo'))
        self.assertEqual(fx.usdcad, 1.41)
        self.assertEqual(fx.source, 'manual override')

    def test_the_bank_of_canada_is_preferred_over_yahoo(self):
        fx = fxr.resolve_usdcad(boc=self._boc(),
                                yahoo=lambda: FxRate(1.99, 'yfinance', '2026-09-17'))
        self.assertIn('Bank of Canada', fx.source)

    def test_yahoo_takes_over_when_the_bank_is_unreachable(self):
        fx = fxr.resolve_usdcad(boc=self._dead('BoC'),
                                yahoo=lambda: FxRate(1.3755, 'yfinance CAD=X (fallback)',
                                                     '2026-09-17'))
        self.assertIn('fallback', fx.source)

    def test_every_source_failing_raises_and_names_what_it_tried(self):
        """No default, no last-known-good cache. An invented rate misprices the whole book by
        one factor in one direction with no symptom — the shape of the 2026-09-01 splice."""
        with self.assertRaises(CAExecutionError) as ctx:
            fxr.resolve_usdcad(boc=self._dead('BoC'), yahoo=self._dead('Yahoo'))
        message = str(ctx.exception)
        self.assertIn('no default', message)
        self.assertIn('BoC is unreachable', message)
        self.assertIn('Yahoo is unreachable', message)

    def test_an_absurd_manual_override_is_still_refused(self):
        """The override skips the lookups, not the sanity check."""
        for bad in (0.0, -1.4, 42.0):
            with self.assertRaises(CAExecutionError):
                fxr.resolve_usdcad(manual=bad, boc=self._boc(), yahoo=self._dead('Yahoo'))


if __name__ == '__main__':
    unittest.main()
