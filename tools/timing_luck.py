"""
Timing-luck sweep: month-end is one draw from a distribution of rebalance schedules.

A monthly strategy could be rebalanced on the 1st trading day of the month, the 2nd, ... or
the last. The signal rule is identical in every case; only the calendar moves. The spread of
outcomes across those schedules is *timing luck* — variation that carries no information and
that a single-schedule backtest silently converts into a claim.

The two 2026-07-28 audits ran this sweep and disagreed:

    |                  | Claude (2013-2024) | Codex (2015-2024) |
    | month-end CAGR   | 5.93% (5th pct)    | 8.56% (80th pct)  |
    | month-end Sortino| 1.41  (35th pct)   | 2.11  (best)      |
    | month-end MaxDD  | -6.54% (95th pct)  | -6.31% (best)     |

The suspected cause was that the month-end schedule kept same-bar execution while the other
19 were priced at the next open, which would flatter it. This script settles it by applying
ONE convention uniformly to all 20 schedules through `common/ledger.py`.

    venv/Scripts/python.exe tools/timing_luck.py [--strategy HAA_G8_Balanced] [--start 2013-01-01]

Everything it prints is derived from the daily store, so it is reproducible from the cache.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import main
from common.ledger import ExecutionConfig, run_ledger
from common.metrics import build_rf_series, calculate_metrics
from common.momentum import calc_13612u, calc_13612w


def sweep(strategy_name, start, end, convention='next_open', cost=0.001, cash='BIL'):
    store = main.load_store({'DATA_START_DATE': '2000-01-01', 'CACHE_DIR': main.CACHE_DIR,
                             'ALLOW_DOWNLOAD': False, 'STRICT_GAPS': False})
    schedules = [f'day_{n}' for n in range(1, 20)] + ['month_end']
    rows = []
    for sched in schedules:
        dates = store.rebalance_dates(sched)
        # The signal must be sampled on the SAME calendar it is executed on, or the sweep
        # measures a mismatch between signal and execution rather than timing luck.
        panel = store.monthly_adj_close(dates)
        strat = main.ALL_STRATEGIES[strategy_name]()
        scores = (calc_13612u(panel) if strat.score_type == 'unweighted'
                  else calc_13612w(panel))
        try:
            alloc = strat.generate_allocations(panel, scores, None, None)
        except Exception as exc:
            print(f'  {sched}: signal failed ({exc})')
            continue
        window = dates[(dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))]
        targets = alloc.reindex(window).fillna(0.0)
        rf, _ = build_rf_series(store, window, cash)
        try:
            led = run_ledger(targets, store,
                             ExecutionConfig(convention=convention,
                                             cost_bps_per_side=cost * 10_000.0,
                                             cash_ticker=cash), label=f'{strategy_name}/{sched}')
        except Exception as exc:
            print(f'  {sched}: ledger refused ({str(exc).splitlines()[0]})')
            continue
        m = calculate_metrics(led.returns, rf=rf)
        rows.append({'schedule': sched, 'cagr': m['cagr'], 'max_dd': m['max_dd'],
                     'sharpe': m['sharpe'], 'sortino': m['sortino'], 'upi': m['upi'],
                     'vol': m['vol'], 'n': m['n_periods']})
    return pd.DataFrame(rows).set_index('schedule')


def report(df, label):
    print(f'\n=== {label} — {len(df)} schedules, one convention applied to all ===')
    print(f"{'schedule':<11}{'CAGR':>9}{'MaxDD':>9}{'Sharpe':>8}{'Sortino':>9}{'UPI':>7}{'n':>5}")
    for sched, r in df.iterrows():
        mark = '  <- month-end' if sched == 'month_end' else ''
        print(f"{sched:<11}{r['cagr']:>9.2%}{r['max_dd']:>9.2%}{r['sharpe']:>8.2f}"
              f"{r['sortino']:>9.2f}{r['upi']:>7.2f}{int(r['n']):>5}{mark}")
    print('-' * 60)
    for col, better_high in (('cagr', True), ('max_dd', True), ('sortino', True),
                             ('upi', True)):
        vals = df[col].to_numpy(dtype=float)
        me = float(df.loc['month_end', col])
        pct = 100.0 * (vals < me).mean() if better_high else 100.0 * (vals > me).mean()
        print(f'  {col:<8} spread {vals.min():>8.4f} .. {vals.max():>8.4f}   '
              f'month-end {me:>8.4f}  = {pct:>3.0f}th percentile')
    print('  A wide spread is not a defect in the code. It is the amount of any single '
          'schedule\'s\n  result that is calendar, not strategy.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--strategy', default='HAA_G8_Balanced')
    ap.add_argument('--convention', default='next_open')
    ap.add_argument('--windows', nargs='+', default=['2013-01-01:2024-12-31',
                                                     '2015-01-01:2024-12-31'])
    args = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    for w in args.windows:
        start, end = w.split(':')
        report(sweep(args.strategy, start, end, convention=args.convention),
               f'{args.strategy}  {start} .. {end}  ({args.convention})')
