"""Backtest-only driver — the safe way to measure anything in this repository.

WHY THIS EXISTS
---------------
`user_config.json` (gitignored, present on the owner's machine) sets ``EXECUTION_MODE=True``,
so a bare ``python main.py`` runs in LIVE mode and prints real brokerage account names and
dollar balances. Every measurement, re-baselining or report regeneration must therefore go
through a config where ``EXECUTION_MODE`` is False and ``BROKER_ACCOUNTS`` is absent.

This module hard-wires both and asserts them, so the live path is unreachable from here by
construction rather than by remembering. It lives INSIDE the repo on purpose: the equivalent
helper used to be written into a scratchpad with an absolute path, and it broke the moment the
project directory moved.

USE
---
    from tools.backtest_driver import run, build_config

    metrics, results, prices, store, cfg = run()                    # whole registry
    metrics, results, prices, store, cfg = run(['HAA_G12'])         # named entries

or from the command line, for a quick look with no report printed:

    venv/Scripts/python.exe -m tools.backtest_driver
    venv/Scripts/python.exe -m tools.backtest_driver HAA_G12 DAA_G12

`run()` returns plain data; nothing here prints a report. Compose it with the reporting
helpers in `main.py` if you need one, or read the metrics directly.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main as engine  # noqa: E402


def build_config(**over):
    """The engine's documented defaults, with the live path nailed shut.

    Every value is read off `main` rather than copied, so this driver cannot drift into
    measuring something the engine no longer does. `EXECUTION_MODE` is the one key that is
    NOT taken from the module: it is forced False and asserted after `over` is applied, so
    even an explicit caller override cannot re-open the live path.
    """
    cfg = {
        'START_DATE': engine.START_DATE,
        'END_DATE': engine.END_DATE,
        'LEVERAGE_FACTOR': 1.0,
        'MARGIN_BORROW_RATE': engine.MARGIN_BORROW_RATE,
        'MARGIN_FOLLOWS_SIGNAL': engine.MARGIN_FOLLOWS_SIGNAL,
        'EXECUTION_MODE': False,          # hard-wired; see the assert below
        'CURRENT_EXECUTION_DATE': engine.CURRENT_EXECUTION_DATE,
        'SAVE_FILES_TO_DISK': False,
        'STRATEGIES_TO_DISPLAY': 99,
        'TOP_N_COUNT': engine.TOP_N_COUNT,
        'RANK_BY': engine.RANK_BY,
        'CACHE_REFRESH_HOURS': 24.0 * 365,   # never re-download from a measurement run
        'SEGMENT_TOP_N': engine.SEGMENT_TOP_N,
        'RANKED_WINDOW_POLICY': engine.RANKED_WINDOW_POLICY,
        'COST_PCT_PER_SIDE': engine.COST_PCT_PER_SIDE,
        'EXECUTION_CONVENTION': engine.EXECUTION_CONVENTION,
        'CASH_TICKER': engine.CASH_TICKER,
        'COVERAGE_POLICY': engine.COVERAGE_POLICY,
        'RF_ANNUAL_FALLBACK': engine.RF_ANNUAL_FALLBACK,
        'LOOKBACK_MONTHS': engine.LOOKBACK_MONTHS,
        'DATA_START_DATE': engine.DATA_START_DATE,
        'CACHE_DIR': engine.CACHE_DIR,
        'SAFETY_MARGIN_PCT': engine.SAFETY_MARGIN_PCT,
        'FLEXIBILITY_BAND_PCT': engine.FLEXIBILITY_BAND_PCT,
        'FLUSH_ROUND_UP_BAND_PCT': engine.FLUSH_ROUND_UP_BAND_PCT,
        'PRICE_CAP_MARGIN_PCT': engine.PRICE_CAP_MARGIN_PCT,
        'MINIMUM_TRADE_PCT': engine.MINIMUM_TRADE_PCT,
        'SHARE_LOT_SIZE': engine.SHARE_LOT_SIZE,
        'FRACTIONAL_SHARES': engine.FRACTIONAL_SHARES,
    }
    cfg.update(over)
    cfg['EXECUTION_MODE'] = False
    cfg.pop('BROKER_ACCOUNTS', None)
    assert cfg['EXECUTION_MODE'] is False, 'this driver is backtest-only'
    assert 'BROKER_ACCOUNTS' not in cfg, 'this driver must never carry broker accounts'
    return cfg


def run(names=None, **over):
    """(metrics_data, results, prices, store, config) for `names` (default: the whole registry)."""
    cfg = build_config(**over)
    prices, scores_w, scores_u, store = engine.load_data(cfg)
    if names is None:
        strategies = [factory() for factory in engine.ALL_STRATEGIES.values()]
    else:
        strategies = [engine.ALL_STRATEGIES[n]() for n in names]
    metrics_data, results = engine.run_backtest(prices, scores_w, scores_u, strategies,
                                                cfg, store=store)
    return metrics_data, results, prices, store, cfg


if __name__ == '__main__':
    wanted = sys.argv[1:] or None
    metrics, _results, _prices, _store, _cfg = run(wanted)
    rows = [d for d in metrics if d.get('in_ranked_window')]
    print(f'{len(metrics)} entries measured, {len(rows)} in the ranked window')
    for d in sorted(rows, key=lambda d: -(d.get('sortino') or float('-inf')))[:10]:
        print(f"  {d['name']:<26} CAGR {d['cagr']:>7.2%}  MaxDD {d['max_dd']:>7.2%}  "
              f"Sortino {d.get('sortino', float('nan')):>5.2f}")
