"""Tests for the personal-settings loader (common/user_config.py).

The loader is the trust boundary between the gitignored user_config.json and the
engine: a missing file must fall back to defaults silently, but a MALFORMED file
must fail loudly — live order sizing should never run on placeholder balances
because of a JSON typo.
"""
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.user_config import load_user_config, save_user_config  # noqa: E402


class TestUserConfig(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(load_user_config(os.path.join(tempfile.gettempdir(), 'nope_does_not_exist.json')), {})

    def test_save_load_roundtrip(self):
        cfg = {'LEVERAGE_FACTOR': 1.3, 'STRATEGIES': ['HAA_G12'],
               'BROKER_ACCOUNTS': [{'account_name': 'TFSA', 'account_balance': 5000.0, 'account_priority': 1}]}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'user_config.json')
            save_user_config(cfg, path)
            self.assertEqual(load_user_config(path), cfg)

    def test_malformed_json_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'user_config.json')
            with open(path, 'w', encoding='utf-8') as f:
                f.write('{ this is not json }')
            with self.assertRaises(SystemExit):
                load_user_config(path)


class TestStrategySelection(unittest.TestCase):
    """`get_strategies_to_run` must give the CLI the same line-up the dashboard shows.

    The dashboard saves EXCLUDED_STRATEGIES (a deny-list). If the CLI only understood the
    old STRATEGIES allow-list, saving in the GUI would silently switch `python main.py`
    back to the hard-coded catalog in main.py — two tools, two answers, no warning.
    """

    def setUp(self):
        import main
        self.main = main
        self.args = types.SimpleNamespace(list=False, strategy=None)

    def _names(self, uc):
        with mock.patch.object(self.main, '_UC', uc):
            return [s.name for s in self.main.get_strategies_to_run(self.args)]

    def test_empty_deny_list_runs_everything_except_controls_and_3x(self):
        """Two roles are hidden by default, each behind its own switch: the degenerate
        single-asset controls, and the 3x `exploratory` variants (no 3x product predates
        2008-11, so their drawdowns have never seen a bear market)."""
        names = self._names({'EXCLUDED_STRATEGIES': []})
        hidden = set(self.main._CONTROL_KEYS) | set(self.main._EXPLORATORY_KEYS)
        self.assertEqual(set(names), set(self.main.ALL_STRATEGIES) - hidden)
        self.assertNotIn('HAA_G1_Simple', names)
        self.assertNotIn('HAA_G3_Leveraged_3X', names)
        # 2x is NOT hidden — only 3x.
        self.assertIn('HAA_G3_Leveraged_2X', names)

    def test_deny_list_removes_exactly_what_it_names(self):
        names = self._names({'EXCLUDED_STRATEGIES': ['DAA_G6', 'GTAA_G5']})
        self.assertNotIn('DAA_G6', names)
        self.assertNotIn('GTAA_G5', names)
        self.assertIn('HAA_G12', names)

    def test_show_controls_adds_the_diagnostics(self):
        names = self._names({'EXCLUDED_STRATEGIES': [], 'SHOW_CONTROLS': True})
        self.assertEqual(set(names),
                         set(self.main.ALL_STRATEGIES) - set(self.main._EXPLORATORY_KEYS))
        self.assertIn('HAA_G1_Simple', names)

    def test_show_exploratory_adds_the_3x_variants_and_nothing_else(self):
        """The two switches are independent on purpose: comparing 2x against 3x is a
        legitimate thing to want without also wanting the single-asset diagnostics."""
        names = self._names({'EXCLUDED_STRATEGIES': [], 'SHOW_EXPLORATORY': True})
        self.assertEqual(set(names),
                         set(self.main.ALL_STRATEGIES) - set(self.main._CONTROL_KEYS))
        self.assertIn('HAA_G3_Leveraged_3X', names)
        self.assertNotIn('HAA_G1_Simple', names)

    def test_both_switches_on_runs_the_whole_registry(self):
        names = self._names({'EXCLUDED_STRATEGIES': [], 'SHOW_CONTROLS': True,
                             'SHOW_EXPLORATORY': True})
        self.assertEqual(set(names), set(self.main.ALL_STRATEGIES))

    def test_legacy_allow_list_still_wins(self):
        names = self._names({'STRATEGIES': ['HAA_G12'], 'EXCLUDED_STRATEGIES': []})
        self.assertEqual(names, ['HAA_G12'])

    def test_no_key_falls_back_to_the_catalog(self):
        self.assertEqual(self._names({}),
                         [s.name for s in self.main.STRATEGIES_TO_RUN])

    def test_unknown_exclusion_warns_but_does_not_drop_anything(self):
        with mock.patch('builtins.print') as printed:
            names = self._names({'EXCLUDED_STRATEGIES': ['HAA_G99_Typo']})
        hidden = set(self.main._CONTROL_KEYS) | set(self.main._EXPLORATORY_KEYS)
        self.assertEqual(set(names), set(self.main.ALL_STRATEGIES) - hidden)
        self.assertTrue(any('HAA_G99_Typo' in str(c) for c in printed.call_args_list))

    def test_control_keys_are_read_from_the_classes(self):
        self.assertEqual(set(self.main._CONTROL_KEYS), {'HAA_G1_Simple', 'BAA_G1_SPY'})

    def test_exploratory_keys_are_read_from_the_classes(self):
        """Derived from `BaseStrategy.role`, which derives from `leverage` — so a new 3x
        factory joins this set without anybody maintaining a list. Two of the original eight
        3x entries were missed when the rule lived on `LeveragedWrapMixin`, because
        `DMLeveraged` does not use that mixin."""
        self.assertEqual(set(self.main._EXPLORATORY_KEYS),
                         {'HAA_G3_Leveraged_3X', 'HAA_G5_Leveraged_3X',
                          'DAA_G3_Leveraged_3X', 'DAA_G5_Leveraged_3X',
                          'BAA_G3_Leveraged_3X', 'BAA_G4_Leveraged_3X',
                          'DM_G3_Leveraged_3X', 'DM_G5_Leveraged_3X'})
        self.assertTrue(all(k.endswith('_3X') for k in self.main._EXPLORATORY_KEYS))


if __name__ == '__main__':
    unittest.main()
