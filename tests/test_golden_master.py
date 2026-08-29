"""
Golden master — the ratchet that makes every behaviour change visible.

This test does NOT assert that the engine is correct. It asserts that the engine
produces the same numbers it produced last time it was blessed. Its whole purpose is
to catch a number moving silently: during the 2026-07 remediation the engine is being
rewritten around an execution ledger, and without a frozen reference there is no way
to tell a fix from a regression.

Workflow when a fix legitimately moves a number:

    venv/Scripts/python.exe -m tests.test_golden_master --record

then commit the regenerated JSON **in the same commit as the behaviour change**, with
the reason in the commit message and a new entry appended to the file's `history`
array. That diff is the audit trail of which fix moved which number.

Input is frozen (`tests/fixtures/frozen_daily_adj_close.csv` and its open sibling), so this
test is deterministic and network-free. It uses DAILY bars because the production execution
convention fills at the session AFTER the decision — a monthly panel cannot express that.
"""
import json
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
FROZEN_ADJ_CLOSE = os.path.join(FIXTURES, 'frozen_daily_adj_close.csv')
FROZEN_ADJ_OPEN = os.path.join(FIXTURES, 'frozen_daily_adj_open.csv')
GOLDEN_JSON = os.path.join(FIXTURES, 'golden_master.json')

#: Deliberately spans the families with different mechanics: momentum-only (HAA),
#: breadth canary (VAA), separate canary (DAA), SMA + dual filter (BAA), protection
#: factor (PAA), an LETF wrap, and a passive benchmark as the control.
#:
#: The wrap slot held `DAA_G3_Leveraged_2X` until 2026-07-29, when RULE 5 removed it — with
#: T=1 against B=2 its cash ladder rounded a lone dead canary down to no de-risking at all.
#: `HAA_G3_Leveraged_2X` replaces it: the same three LETF images (SSO/QLD/UWM), so the
#: frozen fixture still covers it, on the family whose exogenous TIP canary makes it the
#: cleanest wrap in the registry. The G4 sizes need UGL, which the fixture does not carry —
#: and the fixture is frozen on purpose, so the test bends to it rather than the reverse.
STRATEGIES = [
    'HAA_G12', 'HAA_G8_Balanced', 'BAA_G12', 'DAA_G12',
    'VAA_G12', 'PAA2_G12', 'HAA_G3_Leveraged_2X', 'SPY_Benchmark',
]

#: The fixture starts 2012-01, so a 2015 start leaves ample warm-up for every strategy here
#: and no coverage trim fires — the golden master measures the engine, not the calendar.
CONFIG = {
    'START_DATE': '2015-01-01',
    'END_DATE': '2025-01-01',
    'LEVERAGE_FACTOR': 1.0,
    'MARGIN_BORROW_RATE': 0.06,
    'MARGIN_FOLLOWS_SIGNAL': True,
    'COST_PCT_PER_SIDE': 0.001,
    'LOOKBACK_MONTHS': 13,
    'DATA_START_DATE': '2012-01-01',
    'EXECUTION_MODE': False,
    'CURRENT_EXECUTION_DATE': '2025-01-01',
    'EXECUTION_CONVENTION': 'next_open',
    'CASH_TICKER': 'BIL',
    'COVERAGE_POLICY': 'trim',
    'RF_ANNUAL_FALLBACK': 0.03,
}

FIELDS = ('cagr', 'max_dd', 'sharpe', 'sortino', 'vol', 'upi')


def build_store():
    from common.data_engine import PriceStore
    return PriceStore.from_daily_fixture(FROZEN_ADJ_CLOSE, FROZEN_ADJ_OPEN)


def compute():
    """Run the pinned strategies over the frozen fixture. Returns {name: {field: value}}."""
    import main

    store = build_store()
    prices, scores_w, scores_u = main.build_signal_panel(store, CONFIG)
    strategies = [main.ALL_STRATEGIES[n]() for n in STRATEGIES]
    metrics_data, _ = main.run_backtest(prices, scores_w, scores_u, strategies, CONFIG,
                                        store=store)
    out = {}
    for d in metrics_data:
        out[d['name']] = {f: round(float(d[f]), 6) for f in FIELDS}
    return out


def record():
    prior = {}
    if os.path.exists(GOLDEN_JSON):
        with open(GOLDEN_JSON, encoding='utf-8') as fh:
            prior = json.load(fh)
    payload = {
        'description': prior.get(
            'description',
            'Frozen engine output. Regenerate ONLY alongside a deliberate behaviour '
            'change, and append the reason to `history`.'),
        'window': f"{CONFIG['START_DATE']} .. {CONFIG['END_DATE']} (as REQUESTED; the "
                  f"engine reports its actual measured window separately)",
        'fixture': os.path.basename(FROZEN_ADJ_CLOSE),
        'history': prior.get('history', []),
        'metrics': compute(),
    }
    with open(GOLDEN_JSON, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write('\n')
    print(f'Recorded {len(payload["metrics"])} strategies to {GOLDEN_JSON}')


class TestGoldenMaster(unittest.TestCase):
    """Every number below was blessed by a human. A diff here is a behaviour change."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GOLDEN_JSON):
            raise unittest.SkipTest('golden master not recorded yet — run with --record')
        with open(GOLDEN_JSON, encoding='utf-8') as fh:
            cls.expected = json.load(fh)['metrics']
        cls.actual = compute()

    def test_same_strategies_are_measured(self):
        self.assertEqual(sorted(self.expected), sorted(self.actual),
                         'the set of strategies changed — update the golden master '
                         'deliberately, do not let it drift')

    def test_metrics_match(self):
        diffs = []
        for name in sorted(self.expected):
            if name not in self.actual:
                continue
            for field in FIELDS:
                want = self.expected[name][field]
                got = round(float(self.actual[name][field]), 6)
                if abs(want - got) > 1e-6:
                    diffs.append(f'  {name:<24} {field:<8} golden={want:+.6f} now={got:+.6f} '
                                 f'delta={got - want:+.6f}')
        self.assertFalse(diffs, 'engine output moved:\n' + '\n'.join(diffs) +
                         '\n\nIf this is a deliberate fix, regenerate with '
                         '`python -m tests.test_golden_master --record` and commit the '
                         'JSON in the SAME commit as the fix.')


if __name__ == '__main__':
    if '--record' in sys.argv:
        record()
    else:
        unittest.main()
