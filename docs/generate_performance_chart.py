"""
Regenerate docs/performance.png — the README hero image.

Runs a small representative backtest through the project engine and saves a clean
log-scale cumulative-growth comparison. Re-run after engine changes:

    python docs/generate_performance_chart.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import palette
from common.eras import COMMON_ERA_START
from main import (ALL_STRATEGIES, CACHE_DIR, CASH_TICKER, DATA_START_DATE, load_data,
                  run_backtest, strategy_roster)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'performance.png')

# Representative line-up: conservative, balanced, defensive, leveraged, and the benchmark.
# SPY_2X_Benchmark added 2026-07-29: it is what `BAA_G4_Leveraged_2X` has to be read
# against. Without a 2x reference the levered line's distance from SPY conflates the
# timing rule with the leverage, which is the whole point of the passive levered
# benchmarks. It cannot shorten the chart — `may_set_ranked_window` bars anything
# holding an LETF from setting the window, whatever its fidelity label.
SELECTION = ['HAA_G12', 'BAA_G12', 'DAA_G12', 'BAA_G4_Leveraged_2X',
             'SPY_2X_Benchmark', 'SPY_Benchmark']

# One config, used for both data and the run — the pre-audit version kept two, and the
# window in the data half silently disagreed with the label on the chart.
CONFIG = {
    # No window: the era floor is frozen in common/eras.py and the run ends at the last
    # COMPLETE month in the data, so the chart cannot be a picture of a chosen period.
    'START_DATE': COMMON_ERA_START, 'END_DATE': None,
    'DATA_START_DATE': DATA_START_DATE, 'CACHE_DIR': CACHE_DIR,
    'EXECUTION_MODE': False, 'CURRENT_EXECUTION_DATE': None,
    'LEVERAGE_FACTOR': 1.0, 'MARGIN_BORROW_RATE': 0.06, 'MARGIN_FOLLOWS_SIGNAL': True,
    'COST_PCT_PER_SIDE': 0.001, 'LOOKBACK_MONTHS': 13,
    'EXECUTION_CONVENTION': 'next_open', 'CASH_TICKER': CASH_TICKER,
    'COVERAGE_POLICY': 'trim', 'RF_ANNUAL_FALLBACK': 0.03,
}


def main():
    prices, scores_w, scores_u, store = load_data(CONFIG)
    strategies = [ALL_STRATEGIES[n]() for n in SELECTION]
    metrics_data, results = run_backtest(prices, scores_w, scores_u, strategies, CONFIG,
                                         store=store)
    metrics_by_name = {d['name']: d for d in metrics_data}

    # Same family colours and strokes as the dashboard and the CLI report, indexed against
    # the WHOLE registry — so a reader who has seen this image recognises the hues in the
    # app, even though only five of twenty-three lines are drawn here.
    entries = [(s.name, getattr(s, 'role', 'strategy')) for s in strategies]
    _roster = strategy_roster()
    styles = palette.line_styles(entries,
                                 universe=[(d['name'], d['role']) for d in _roster],
                                 ratios={d['name']: d['leverage'] for d in _roster})

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, cum_ret in results.items():
        d = metrics_by_name[name]
        # Five lines is few enough that a legend still beats edge labels, and the legend can
        # carry the numbers as well as the name — so this chart keeps one.
        label = f"{name}  (CAGR {d['cagr']:.1%}, Sortino {d['sortino']:.2f})"
        ax.plot(cum_ret.index, cum_ret.values, label=label,
                **palette.plot_kwargs(styles[name]))

    # Title the window that was MEASURED, never the one that was requested. The withdrawn
    # pre-audit chart said 2019-2023 while measuring from 2020-02.
    spans = [d for d in metrics_data if d.get('first_return') is not None]
    lo = min(d['first_return'] for d in spans).date()
    hi = max(d['last_return'] for d in spans).date()
    ax.set_yscale('log')
    ax.set_title(f'Keller Strategies — Cumulative Growth ({lo} → {hi}, log scale)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1 (log)')
    ax.text(0.995, 0.02,
            f"fills at next open · cost {CONFIG['COST_PCT_PER_SIDE']:.2%}/side · "
            f"cash {CONFIG['CASH_TICKER']} · Sharpe/Sortino net of realised "
            f"{CONFIG['CASH_TICKER']}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7, alpha=0.65)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
