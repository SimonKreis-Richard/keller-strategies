"""Keller Strategies — desktop dashboard (NiceGUI).

A thin GUI over the engine in main.py — no strategy logic lives here.
  - Backtest tab : main.load_data + main.run_backtest (metrics table, log growth chart,
                   latest target weights)
  - Live tab     : main.compute_live_signals (whole-share orders across your broker
                   accounts, canary status, remaining cash)

Personal settings load from / save to user_config.json (gitignored) — the same file
main.py reads, so the CLI and the GUI always agree.

Run:  python app.py        (native desktop window if pywebview is installed,
                            otherwise opens in your default browser)
"""
import importlib.util

import matplotlib
matplotlib.use('Agg')  # headless backend — NiceGUI renders figures to SVG
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from nicegui import run, ui

import main as engine
from common import eras, leverage_advice, palette, robustness
from common.margin_sizing import NotCalculable
from common.metrics import rank_descending, wealth_curve
from common.user_config import load_user_config, save_user_config

_uc = load_user_config()

RANK_OPTIONS = {'sortino': 'Sortino', 'sharpe': 'Sharpe', 'cagr': 'CAGR',
                'upi': 'UPI (Ulcer Performance Index)',
                'vol': 'Volatility (low is best)', 'max_dd': 'Max Drawdown'}
PCT = '{:.2%}'.format
NUM = '{:.2f}'.format

#: Aspect ratio of every growth chart, as a matplotlib figsize. The absolute numbers set the
#: SHAPE and the font-to-plot ratio, not the rendered size: NiceGUI serialises the figure to
#: SVG and `nicegui.css` gives it `width: 100%`, so the browser scales it to whatever the pane
#: is. It is wide (a 2.6:1 box) because that is what the pane is — the old 11x6.5 figure drew a
#: near-square chart beside a 20-inch table and left the right third of the window empty.
CHART_FIGSIZE = (17.0, 6.5)

#: The named risk presets of `margin_sizing.MarginPolicy`, in multiples of drawdown. The only
#: preference parameter the sizing module has, so it is the only one the picker offers.
K_OPTIONS = {5.0: 'Prudent — survive 5x the stressed drawdown',
             3.0: 'Balanced — survive 3x (default)',
             2.0: 'Aggressive — survive 2x'}

#: {key: title} for the four pre-registered partitions, read from `eras` rather than listed.
#:
#: These are DISPLAY toggles and nothing else, and the distinction is the whole reason they are
#: safe. Each of the four is INDEPENDENTLY a complete partition of the era — exhaustive,
#: disjoint, and asserted by `tests/test_eras.py` to compound back to the era's own return — so
#: hiding one costs nothing and breaks nothing. Hiding SEGMENTS inside one would be a different
#: act entirely: the partition would stop tiling the era, and "show me only the interesting
#: periods" is precisely the window-choosing this design exists to forbid. The short list of
#: named crises (`common/regimes.py`, rendered above the partitions) is the sanctioned way to
#: look at stress episodes only — it is non-exhaustive BY CONSTRUCTION and says so.
PARTITIONS = {seg.key: seg.title for seg in eras.SEGMENTATIONS}

# --------------------------------------------------------------------------- #
# What the picker is allowed to offer
# --------------------------------------------------------------------------- #
# Roles come from the strategy classes via `engine.strategy_roster()`, not from a list
# maintained here — a hand-kept exclusion list in the GUI would rot the first time a
# strategy was added.
#
# 'control' entries (HAA_G1_Simple, BAA_G1_SPY) are degenerate single-asset diagnostics.
# They exist so a family's record can be split into "the timing rule" and "the universe",
# and they are excluded from the selection statistics for the same reason they are excluded
# here: a portfolio nobody would hold is not a portfolio to choose. They are still one
# switch away, because HAA_G1_Simple is paper-faithful and regularly ranks top-three, and
# hiding that permanently would be hiding a result rather than tidying a menu.
#
# 'benchmark' entries stay in the list: SPY and the Golden Butterfly are what the whole
# comparison is measured against.
ROSTER = engine.strategy_roster()
BY_NAME = {r['name']: r for r in ROSTER}
# 'exploratory' entries (the 3x variants) are hidden by default for the same reason controls
# are, and a different reason for it: not degeneracy but that no 3x product predates 2008-11,
# so their drawdown columns are bull-market numbers with no bear market behind them. They get
# their OWN switch rather than sharing the controls one, because they are real strategies with
# real signals and someone comparing 2x against 3x wants them without also wanting the
# single-asset diagnostics.
SELECTABLE = [r['name'] for r in ROSTER if r['role'] not in ('control', 'exploratory')]
CONTROLS = [r['name'] for r in ROSTER if r['role'] == 'control']
EXPLORATORY = [r['name'] for r in ROSTER if r['role'] == 'exploratory']
UNIVERSE = [(r['name'], r['role']) for r in ROSTER]

#: {name: execution multiple}, read from `BaseStrategy.leverage` via the roster — never parsed
#: off a key suffix. Feeds the chart's width channel and the ratio filter below.
RATIOS = {r['name']: r['leverage'] for r in ROSTER}

#: The distinct ratios actually present, ascending. Drives the 1x/2x/3x filter row: derived, so
#: adding a 4x product some day would surface a fourth toggle with no UI change.
PRESENT_RATIOS = sorted(set(RATIOS.values()))


# --------------------------------------------------------------------------- #
# Engine calls (plain functions, executed off the event loop via run.io_bound)
# --------------------------------------------------------------------------- #
def _compute_backtest(names, leverage, borrow, txn, follows_signal=True,
                      k=3.0, maintenance_base=leverage_advice.MAINTENANCE_BASE):
    # No start/end: the evaluation era is frozen in common/eras.py and runs to the last
    # COMPLETE month in the store. A window the user picks is a result the user picked.
    config = {
        'START_DATE': engine.START_DATE, 'END_DATE': None,
        'DATA_START_DATE': engine.DATA_START_DATE,
        'CACHE_DIR': engine.CACHE_DIR, 'EXECUTION_MODE': False,
        'CURRENT_EXECUTION_DATE': None, 'LEVERAGE_FACTOR': leverage,
        'MARGIN_BORROW_RATE': borrow, 'COST_PCT_PER_SIDE': txn,
        'MARGIN_FOLLOWS_SIGNAL': follows_signal,
        'LOOKBACK_MONTHS': engine.LOOKBACK_MONTHS,
        'EXECUTION_CONVENTION': engine.EXECUTION_CONVENTION,
        'CASH_TICKER': engine.CASH_TICKER,
        'COVERAGE_POLICY': engine.COVERAGE_POLICY,
        'RF_ANNUAL_FALLBACK': engine.RF_ANNUAL_FALLBACK,
        # GUI and CLI must not diverge on data guards (they did until 2026-07-30, when
        # STRICT_GAPS existed nowhere in production and both silently ran without it).
        'STRICT_GAPS': engine.STRICT_GAPS,
    }
    prices, scores_w, scores_u, store = engine.load_data(config)
    strategies = [engine.ALL_STRATEGIES[n]() for n in names if n in engine.ALL_STRATEGIES]
    metrics, results = engine.run_backtest(prices, scores_w, scores_u, strategies, config,
                                           store=store)
    # Sized here, inside the worker thread, because the drawdown bootstrap costs a few seconds
    # over a full registry and the event loop must not wear it. Borrowing capacity is left
    # unsupplied on purpose: it is the one cap that needs a real account, and an unbounded axis
    # that says so beats a plausible number nobody has.
    advice = leverage_advice.advise(
        metrics, k=k, borrow_rate_annual=borrow, maintenance_base=maintenance_base,
        capacity_leverage=engine.BORROWING_CAPACITY_LEVERAGE, run_leverage=leverage)
    return prices, scores_w, scores_u, strategies, metrics, results, advice


def _compute_live(names, exec_date, accounts, knobs):
    config = {
        'START_DATE': engine.START_DATE, 'DATA_START_DATE': engine.DATA_START_DATE,
        'CACHE_DIR': engine.CACHE_DIR, 'EXECUTION_MODE': True,
        'CURRENT_EXECUTION_DATE': exec_date, 'END_DATE': None,
        'STRICT_GAPS': engine.STRICT_GAPS,
        'STRATEGIES_TO_DISPLAY': [], **knobs,
    }
    prices, scores_w, scores_u, store = engine.load_data(config)
    strategies = [engine.ALL_STRATEGIES[n]() for n in names if n in engine.ALL_STRATEGIES]
    # The store is passed so the GUI gets the SAME refusals as the CLI: a levered defensive
    # sleeve, a row that does not sum to one, or a constructed price under a live order.
    return engine.compute_live_signals(prices, scores_w, scores_u, strategies, config,
                                       accounts, store=store)


def _latest_weights(strategies, prices, scores_w, scores_u):
    """Most recent non-zero target allocation per strategy ('what to hold now')."""
    rows = []
    for strat in strategies:
        scores = scores_u if strat.score_type == 'unweighted' else scores_w
        try:
            alloc = strat.generate_allocations(prices, scores, None, None)
        except Exception as e:
            rows.append({'Strategy': strat.name, 'Asset': f'(error: {e})', 'Weight': ''})
            continue
        held = alloc.iloc[-1]
        held = held[held.abs() > 1e-6].sort_values(ascending=False)
        if held.empty:
            rows.append({'Strategy': strat.name, 'Asset': 'CASH', 'Weight': PCT(1.0)})
        for asset, w in held.items():
            rows.append({'Strategy': strat.name,
                         'Asset': f'{asset} — {engine.TICKER_NAMES.get(asset, asset)}',
                         'Weight': PCT(float(w))})
    return rows


# --------------------------------------------------------------------------- #
# UI state (seeded from user_config.json, falling back to engine defaults)
# --------------------------------------------------------------------------- #
# The selection is stored as what the user turned OFF, not as what they turned on. Everything
# runs by default and the user unticks; an empty file therefore means "all", which is the
# behaviour you want from a fresh install and the one the old `STRATEGIES` allow-list got
# backwards — a saved list silently froze the dashboard to whatever the registry held on the
# day it was written, so strategies added later never appeared. `STRATEGIES` is migrated on
# first save and then dropped.
_excluded = set(_uc.get('EXCLUDED_STRATEGIES') or ())
state = {
    'strategies': {k for k in SELECTABLE if k not in _excluded},
    'show_controls': bool(_uc.get('SHOW_CONTROLS', False)),
    'show_exploratory': bool(_uc.get('SHOW_EXPLORATORY', False)),
    # Which partitions to DISPLAY. Stored as what is shown rather than what is hidden — unlike
    # the strategy picker, because the set is fixed at four and cannot grow behind your back.
    # An unknown key in the saved file is dropped rather than honoured, so renaming a
    # segmentation in `eras.py` cannot leave a dead entry selecting nothing.
    'partitions': ({k for k in _uc.get('REGIME_PARTITIONS') if k in PARTITIONS}
                   if _uc.get('REGIME_PARTITIONS') is not None else set(PARTITIONS)),
    'accounts': [dict(a) for a in engine.BROKER_ACCOUNTS],
}


def _pool():
    """Every name the picker currently offers (controls and 3x only when revealed)."""
    return (SELECTABLE
            + (CONTROLS if state['show_controls'] else [])
            + (EXPLORATORY if state['show_exploratory'] else []))


def selected_names():
    """Ticked names, in registry order, restricted to what is currently offered."""
    pool = set(_pool())
    return [r['name'] for r in ROSTER if r['name'] in pool and r['name'] in state['strategies']]


# --------------------------------------------------------------------------- #
# Strategy picker — a checklist, not a dropdown
# --------------------------------------------------------------------------- #
# Everything is ticked on open and the user unticks. Three levels of granularity, each of
# which refreshes the whole widget so the counters never lie about what is selected:
#   * "All strategies"  — the master switch
#   * one box per FAMILY — the unit people actually think in ("drop the levered ones")
#   * one box per entry
# The family box shows k/n and is ticked only when the family is complete, so a partially
# selected family is visibly partial rather than silently rounded to on or off.
def _set_many(names, on):
    for n in names:
        state['strategies'].add(n) if on else state['strategies'].discard(n)
    strategy_picker.refresh()


def _toggle_controls(on):
    state['show_controls'] = bool(on)
    # Revealing the controls ticks them, so the switch reads as "include these" rather than
    # merely "show me two boxes I then have to find and tick".
    _set_many(CONTROLS, on)


def _toggle_exploratory(on):
    state['show_exploratory'] = bool(on)
    _set_many(EXPLORATORY, on)


def _toggle_partition(key, on):
    """Show or hide one partition, re-rendering from the cache — the engine is not consulted.

    Same reasoning as `rank_changed`: which partitions you read is a presentation choice over
    numbers that already exist, and re-running the store to answer it would be absurd.
    """
    state['partitions'].add(key) if on else state['partitions'].discard(key)
    rank_changed()


def _ratio_label(r):
    return f'{r:g}x'


@ui.refreshable
def strategy_picker():
    pool = _pool()
    sel = state['strategies']
    n_sel = sum(1 for n in pool if n in sel)
    groups = palette.group_by_family([(n, BY_NAME[n]['role']) for n in pool])

    with ui.card().classes('w-full'):
        with ui.row().classes('w-full items-center justify-between no-wrap'):
            ui.checkbox('All strategies', value=(n_sel == len(pool)),
                        on_change=lambda e: _set_many(pool, e.value)) \
              .props('dense').classes('font-bold')
            ui.label(f'{n_sel}/{len(pool)}').classes('text-xs text-gray-500 font-mono')
        # RATIO filter — a second dimension that INTERSECTS the family groups rather than
        # nesting inside them. With up to three ratios per family, "show me every 3x" and
        # "show me all of HAA" are both things people want, and neither is a sub-case of the
        # other. Ticking 2x adds every 2x entry currently on offer, across all families.
        by_ratio = {}
        for n in pool:
            by_ratio.setdefault(RATIOS.get(n, 1.0), []).append(n)
        if len(by_ratio) > 1:
            ui.separator()
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                ui.label('Leverage:').classes('text-xs text-gray-500')
                for ratio in sorted(by_ratio):
                    names_r = by_ratio[ratio]
                    kr = sum(1 for n in names_r if n in sel)
                    ui.checkbox(_ratio_label(ratio), value=(kr == len(names_r)),
                                on_change=lambda e, ns=names_r: _set_many(ns, e.value))                       .props('dense').classes('text-xs')
                    ui.label(f'{kr}/{len(names_r)}').classes('text-xs text-gray-400 font-mono')
            ui.label('Cuts across the families below — line WIDTH encodes this on the growth '
                     'chart (1x thin, 2x medium, 3x thick).').classes('text-xs text-gray-500')
        ui.separator()
        with ui.column().classes('w-full gap-0 max-h-96 overflow-y-auto'):
            for family, names in groups:
                k = sum(1 for n in names if n in sel)
                colour = palette.FAMILY_COLORS.get(family, palette.FALLBACK_COLOR)
                with ui.row().classes('w-full items-center gap-1 no-wrap mt-1'):
                    # The swatch is the same hue the family gets in the growth chart, so the
                    # picker and the chart can be read against each other.
                    ui.html(f'<span style="display:inline-block;width:10px;height:10px;'
                            f'border-radius:2px;background:{colour}"></span>')
                    ui.checkbox(f'{family}', value=(k == len(names)),
                                on_change=lambda e, ns=names: _set_many(ns, e.value)) \
                      .props('dense').classes('text-sm font-bold')
                    ui.label(f'{k}/{len(names)}').classes('text-xs text-gray-400 font-mono')
                for name in names:
                    with ui.row().classes('w-full items-center gap-1 no-wrap pl-5'):
                        ui.checkbox(name, value=(name in sel),
                                    on_change=lambda e, n=name: _set_many([n], e.value)) \
                          .props('dense').classes('text-xs')
                        ui.label(BY_NAME[name]['fidelity']).classes('text-xs text-gray-400')
                        if RATIOS.get(name, 1.0) != 1.0:
                            ui.label(_ratio_label(RATIOS[name]))                               .classes('text-xs text-orange-700 font-mono')
        # A legacy allow-list still WINS in main.py (see `get_strategies_to_run`), so until
        # it is dropped the CLI and this checklist would disagree about what runs. Saying so
        # is better than silently rewriting a file the user owns, or silently disagreeing.
        if _uc.get('STRATEGIES'):
            with ui.row().classes('w-full items-start gap-1 no-wrap bg-orange-50 p-2'):
                ui.label('⚠️').classes('text-xs')
                ui.label(f'user_config.json still has the old STRATEGIES list '
                         f'({len(_uc["STRATEGIES"])} entries), and `python main.py` obeys '
                         f'that, not this checklist. Hit 💾 Save my settings once to retire '
                         f'it — the two agree from then on.').classes('text-xs text-orange-900')
        ui.separator()
        ui.switch('🔬 Show diagnostic controls', value=state['show_controls'],
                  on_change=lambda e: _toggle_controls(e.value)).props('dense').classes('text-xs')
        ui.label('Single-asset degenerate cases (HAA-Simple, BAA-SPY). Not portfolios — they '
                 'exist to separate a family\'s timing rule from its universe. Off by default.') \
          .classes('text-xs text-gray-500')
        # MISSING UNTIL 2026-07-31, and the omission made eight registered strategies
        # unreachable: `_toggle_exploratory` and `state['show_exploratory']` existed, `_pool()`
        # read the flag, `save_settings` persisted it — and nothing could ever set it. The only
        # way to see a 3x variant in this dashboard was to hand-edit SHOW_EXPLORATORY into
        # user_config.json. Dead code on one side of the switch is a bug on the other.
        ui.switch('🧪 Show 3x exploratory variants', value=state['show_exploratory'],
                  on_change=lambda e: _toggle_exploratory(e.value)) \
          .props('dense').classes('text-xs')
        ui.label(f'The {len(EXPLORATORY)} 3x wraps. Real strategies with real signals, measured '
                 f'in full — but NO 3x product predates 2008-11, so not one of their drawdowns '
                 f'has been through a bear market like 2008, and their rows start years after '
                 f'everything else. Off by default for that reason, not because they lose.') \
          .classes('text-xs text-gray-500')


def save_settings():
    cfg = dict(_uc)  # preserve keys the GUI doesn't manage (e.g. EXECUTION_MODE)
    # Actively DROP the retired window keys rather than merely ignoring them: leaving a
    # START_DATE in the file invites the belief that it still controls something.
    cfg.pop('START_DATE', None)
    cfg.pop('END_DATE', None)
    # Retired in favour of EXCLUDED_STRATEGIES below — see the note on `state`.
    cfg.pop('STRATEGIES', None)
    cfg.update({
        'EXCLUDED_STRATEGIES': sorted(set(_pool()) - state['strategies']),
        'SHOW_CONTROLS': bool(state['show_controls']),
        'SHOW_EXPLORATORY': bool(state['show_exploratory']),
        'REGIME_PARTITIONS': sorted(state['partitions']),
        'LEVERAGE_FACTOR': float(lev_in.value), 'MARGIN_BORROW_RATE': float(borrow_in.value) / 100.0,
        'MARGIN_FOLLOWS_SIGNAL': bool(follows_in.value),
        'COST_PCT_PER_SIDE': float(txn_in.value) / 100.0, 'RANK_BY': rank_in.value,
        'SAFETY_FACTOR_K': float(k_in.value), 'MAINTENANCE_BASE': float(maint_in.value) / 100.0,
        'CURRENT_EXECUTION_DATE': exec_date_in.value,
        'BROKER_ACCOUNTS': [dict(a) for a in state['accounts']],
        'SAFETY_MARGIN_PCT': float(safety_in.value), 'FLEXIBILITY_BAND_PCT': float(flex_in.value),
        'FLUSH_ROUND_UP_BAND_PCT': float(flush_in.value), 'PRICE_CAP_MARGIN_PCT': float(cap_in.value),
        'MINIMUM_TRADE_PCT': float(min_trade_in.value),
        'SHARE_LOT_SIZE': int(lot_in.value), 'FRACTIONAL_SHARES': bool(frac_in.value),
    })
    save_user_config(cfg)
    ui.notify('Settings saved to user_config.json (gitignored)', type='positive')


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
ui.page_title('Keller Strategies')
with ui.header().classes('items-center justify-between'):
    ui.label('📈 Keller Strategies — Quant Dashboard').classes('text-xl font-bold')
    ui.button('💾 Save my settings', on_click=save_settings).props('flat color=white')

with ui.tabs() as tabs:
    backtest_tab = ui.tab('Backtest')
    live_tab = ui.tab('Live signals')

with ui.tab_panels(tabs, value=backtest_tab).classes('w-full'):

    # ------------------------------ BACKTEST ------------------------------ #
    with ui.tab_panel(backtest_tab):
        with ui.row().classes('w-full gap-6'):
            with ui.column().classes('w-80'):
                strategy_picker()
                with ui.card().classes('w-full bg-blue-50'):
                    ui.label('📅 Evaluation era — fixed, not a setting') \
                      .classes('text-sm font-bold')
                    ui.label(f'From {engine.START_DATE[:7]} (the first month these assets '
                             f'allow) to the last complete month in the data. There is no '
                             f'start/end box because a window you choose is a result you '
                             f'chose.').classes('text-xs text-gray-600')
                    ui.label('Results are reported segment by segment across market regimes '
                             'dated by the NBER, the FOMC, the BLS and the S&P 500.') \
                      .classes('text-xs text-gray-600')
                    ui.separator()
                    ui.label('Partitions to display').classes('text-xs font-bold')
                    for _key, _title in PARTITIONS.items():
                        ui.checkbox(_title, value=_key in state['partitions'],
                                    on_change=lambda e, k=_key: _toggle_partition(k, e.value)) \
                          .props('dense').classes('text-xs')
                    ui.label('A DISPLAY choice, not a measurement one: each of the four is '
                             'independently exhaustive and disjoint, so hiding one changes '
                             'nothing about the others. Segments INSIDE a partition are not '
                             'individually hideable — dropping some would stop it tiling the '
                             'era, which is the one property that makes it an antidote to '
                             'picking windows rather than another way of picking them. For '
                             'stress episodes alone, read the named-crises table instead.') \
                      .classes('text-xs text-gray-500')
                lev_in = ui.number('Margin leverage (×)', value=_uc.get('LEVERAGE_FACTOR', 1.0),
                                   min=1.0, max=3.0, step=0.1).classes('w-full')
                borrow_in = ui.number('Margin borrow rate (%/yr)', value=_uc.get('MARGIN_BORROW_RATE', 0.06) * 100,
                                      min=0.0, max=15.0, step=0.5).classes('w-full')
                follows_in = ui.switch('Margin follows the signal',
                                       value=bool(_uc.get('MARGIN_FOLLOWS_SIGNAL', True))).classes('w-full')
                ui.label('On: borrow only against the offensive sleeve, so the loan is repaid as the '
                         'signal goes risk-off (defence held at 1x). Off: flat leverage, loan stays '
                         'drawn during drawdowns.').classes('text-xs text-gray-500')
                txn_in = ui.number('Transaction cost (%, one-way per leg)',
                                   value=_uc.get('COST_PCT_PER_SIDE',
                                                 _uc.get('TRANSACTION_COST_PCT', 0.001)) * 100,
                                   min=0.0, max=1.0, step=0.05).classes('w-full')
                ui.label('Charged on the notional actually traded, per leg — a full A→B '
                         'rotation costs twice this. Keller assumes 0.1% one-way.') \
                  .classes('text-xs text-gray-500')
                # --- what the "Max margin" column is sized against ------------------------- #
                # Two knobs, and only two, because `margin_sizing` has exactly one preference
                # parameter (k) and exactly one input with no defensible default (m). The
                # borrow rate above is the third thing it reads. Nothing else about a broker is
                # needed for the answer, and the cap that WOULD need one — the credit line —
                # reports itself unbounded rather than guessing.
                ui.separator()
                ui.label('⚖️ Sustainable leverage').classes('text-sm font-bold')
                k_in = ui.select(K_OPTIONS, label='Safety factor',
                                 value=float(_uc.get('SAFETY_FACTOR_K', 3.0))).classes('w-full')
                maint_in = ui.number('Maintenance margin base (%)',
                                     value=_uc.get('MAINTENANCE_BASE',
                                                   leverage_advice.MAINTENANCE_BASE) * 100,
                                     min=10.0, max=100.0, step=5.0).classes('w-full')
                ui.label('The broker\'s house requirement on an ordinary ETF position; a '
                         'leveraged product is charged this times the fund\'s own multiple '
                         '(2x → 60%, 3x → 90%). FINRA Rule 4210(c) floors it at 25%. An '
                         'assumption, not your broker — nobody has one yet.') \
                  .classes('text-xs text-gray-500')
                # Changing the sort key re-renders the results block from the in-memory
                # cache (see `rank_changed`) — the engine is NOT re-run, because sorting
                # is a presentation choice over numbers that already exist. Before a first
                # run there is nothing to re-render and the handler is a no-op.
                rank_in = ui.select(RANK_OPTIONS, label='Rank by',
                                    value=_uc.get('RANK_BY', 'sortino'),
                                    on_change=lambda e: rank_changed()).classes('w-full')
                run_bt_btn = ui.button('🚀 Run backtest', on_click=lambda: run_backtest_clicked()) \
                               .classes('w-full').props('color=primary')
            bt_results = ui.column().classes('flex-1 min-w-0')

    # -------------------------------- LIVE -------------------------------- #
    with ui.tab_panel(live_tab):
        with ui.row().classes('w-full gap-6'):
            with ui.column().classes('w-96'):
                ui.label('Broker accounts').classes('text-lg font-bold')
                ui.label('Filled in priority order (1 = first). Managed in user_config.json.') \
                  .classes('text-xs text-gray-500')

                @ui.refreshable
                def accounts_editor():
                    for acc in state['accounts']:
                        with ui.row().classes('w-full items-end gap-1'):
                            ui.input('Account').bind_value(acc, 'account_name').classes('w-24')
                            ui.number('Balance $', min=0).bind_value(acc, 'account_balance').classes('w-28')
                            ui.number('Prio', min=1, step=1).bind_value(acc, 'account_priority').classes('w-14')
                            ui.button(icon='delete', color='negative',
                                      on_click=lambda a=acc: (state['accounts'].remove(a),
                                                              accounts_editor.refresh())).props('flat dense')
                accounts_editor()
                ui.button('＋ Add account', on_click=lambda: (
                    state['accounts'].append({'account_name': 'NEW', 'account_balance': 0.0,
                                              'account_priority': len(state['accounts']) + 1}),
                    accounts_editor.refresh())).props('flat dense')

                ui.separator()
                ui.label('Order sizing').classes('text-lg font-bold')
                exec_date_in = ui.input('Execution date (YYYY-MM-DD)',
                                        value=_uc.get('CURRENT_EXECUTION_DATE', '2026-07-01')).classes('w-full')
                safety_in = ui.number('Safety margin reserve (%)', value=engine.SAFETY_MARGIN_PCT,
                                      min=0.0, step=0.1).classes('w-full')
                flex_in = ui.number('Flexibility band — underfill (%)', value=engine.FLEXIBILITY_BAND_PCT,
                                    min=0.0, step=0.5).classes('w-full')
                flush_in = ui.number('Flush round-up band — overshoot (%)', value=engine.FLUSH_ROUND_UP_BAND_PCT,
                                     min=0.0, step=0.5).classes('w-full') \
                             .tooltip('0 = off. Rounds the last lot UP to deploy idle cash, '
                                      'never beyond this overshoot or the safety reserve.')
                cap_in = ui.number('Price cap margin (%)', value=engine.PRICE_CAP_MARGIN_PCT,
                                   min=0.0, step=0.25).classes('w-full') \
                           .tooltip('0 = off. Limit-price ceiling for after-hours GTC orders '
                                    '(e.g. IBKR Midprice + cap): cap = quote × (1 + X%). Shares are '
                                    'sized AT the cap, so the funds check can never reject the order.')
                min_trade_in = ui.number('Minimum trade (%)', value=engine.MINIMUM_TRADE_PCT,
                                         min=0.0, step=0.5).classes('w-full')
                lot_in = ui.number('Share lot size', value=engine.SHARE_LOT_SIZE, min=1, step=1).classes('w-full')
                frac_in = ui.checkbox('Fractional shares', value=engine.FRACTIONAL_SHARES)
                run_live_btn = ui.button('🎯 Compute live orders', on_click=lambda: run_live_clicked()) \
                                 .classes('w-full').props('color=primary')
            live_results = ui.column().classes('flex-1 min-w-0')


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def _table(rows):
    """Render a list of dicts as a table in the CURRENT slot (call inside a `with`)."""
    if rows:
        ui.table(columns=[{'name': k, 'label': k, 'field': k, 'align': 'left'} for k in rows[0]],
                 rows=rows).classes('w-full')


# --------------------------------------------------------------------------------------- #
# The growth chart, drawn per PERIOD SECTION rather than once for all history
# --------------------------------------------------------------------------------------- #
# Until 2026-07-31 there was exactly one chart, over the whole ranked window, and every regime
# section below it carried tables only. That put the two halves of the same question in
# different languages: "what did this return in the GFC" was a number, and "what did the path
# look like" was available only for the eighteen-year window in which the GFC is four inches
# wide. The chart is now a function of a WINDOW, and every section that has a window draws one.
#
# Segment charts are built from `returns_full` — each strategy's entire measurable history, the
# same series the panel cells use — and re-based to 1.0 at the segment's own start. A strategy
# that entered mid-segment therefore starts its line mid-segment, which is the visual form of
# the `~` the panel prints, and the leaderboard underneath names it as not ranked.
def _rebased_curves(metrics, start, end):
    """{name: wealth curve} inside [start, end], each re-based to 1.0 at its own first month."""
    curves = {}
    for d in metrics:
        r = d.get('returns_full')
        if r is None or getattr(r, 'empty', True):
            r = d.get('returns')
        if r is None or r.empty:
            continue
        window = r[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
        if len(window) < 2:
            continue
        curves[d['name']] = wealth_curve(window)
    return curves


def _adverse_curves(metrics, segments):
    """{name: wealth curve} over every adverse month, compounded, on an ORDINAL axis.

    The ADVERSE bucket stitches months that are not contiguous — 2008 next to 2020 next to
    2022 — so there is no date axis that can honestly carry it. Plotting it against a real
    calendar would draw straight lines across the years in between and invite exactly the
    reading the bucket is not making. The x axis is therefore a COUNT of adverse months, which
    is what the number in the panel is a function of.
    """
    spans = [(pd.Timestamp(s.start), pd.Timestamp(s.end)) for s in segments if s.adverse]
    curves = {}
    for d in metrics:
        r = d.get('returns_full')
        if r is None or getattr(r, 'empty', True):
            r = d.get('returns')
        if r is None or r.empty:
            continue
        mask = pd.Series(False, index=r.index)
        for lo, hi in spans:
            mask |= (r.index >= lo) & (r.index <= hi)
        window = r[mask]
        if len(window) < 2:
            continue
        wealth = np.concatenate([[1.0], (1.0 + window.to_numpy(dtype=float)).cumprod()])
        curves[d['name']] = pd.Series(wealth, index=range(len(wealth)))
    return curves


def _growth_chart(curves, height=None, xlabel=None):
    """Draw one log-scale growth chart. Call inside a `with` block.

    Strokes are indexed against the FULL registry (`universe=UNIVERSE`), not against what is
    being drawn, so a strategy keeps its colour, dash and width in every section — the whole
    point of repeating the chart is that the same line can be followed from one regime to the
    next, and a per-chart style index would break that at the first segment somebody misses.
    """
    if not curves:
        ui.label('Nothing to plot over this span.').classes('text-xs text-gray-500')
        return
    figsize = CHART_FIGSIZE if height is None else (CHART_FIGSIZE[0], height)
    with ui.pyplot(figsize=figsize).classes('w-full'):
        ax = plt.gca()
        entries = [(n, BY_NAME.get(n, {}).get('role', 'strategy')) for n in curves]
        styles = palette.line_styles(entries, universe=UNIVERSE, ratios=RATIOS)
        for name, cum in curves.items():
            ax.plot(cum.index, cum.values, label=name, **palette.plot_kwargs(styles[name]))
        ax.set_yscale('log')
        ax.set_ylabel('Cumulative return (log)')
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(True, which='both', ls='--', alpha=0.4)
        ax.legend(handles=palette.family_legend_handles(entries), title='Family',
                  loc='upper left', fontsize=7, ncol=3, framealpha=0.85)
        # Reserve the right margin for the edge labels. `tight_layout` would undo this.
        plt.gcf().subplots_adjust(right=0.84, left=0.05, top=0.97,
                                  bottom=0.12 if xlabel else 0.08)
        palette.label_lines(ax, {n: float(s.iloc[-1]) for n, s in curves.items() if len(s)},
                            styles)


def _lazy_chart(expansion, build):
    """Draw `build()` the first time `expansion` is OPENED, then never again.

    Call INSIDE `with expansion:` — it places its own slot at the current position, so the
    chart lands wherever this is called and the rest of the section is written around it.

    Four segmentations of a dozen segments each is up to forty charts, and matplotlib costs
    roughly a second apiece. Rendering them all on Run would triple the wait to draw pictures
    nobody has asked to see yet; rendering on every open would re-pay it each time. So: once,
    on demand.
    """
    state = {'slot': ui.column().classes('w-full')}

    def opened(e):
        if e.value and not state.get('done'):
            state['done'] = True
            with state['slot']:
                build()
    expansion.on_value_change(opened)
    return state['slot']


def _cell_text(cell):
    """Panel cell as text: return / worst drawdown, `~` when the coverage is partial."""
    if 'na' in cell:
        return f"n/a ({cell['na']})"
    text = f"{cell['return']:+.1%} / {cell['max_dd']:.1%}"
    return ('~ ' + text) if cell.get('partial') else text


def _episode_panel(metrics):
    """The nine NAMED stress episodes — dot-com, the GFC, COVID, 2022 — as one table.

    Call inside a `with` block.

    Present in the CLI report since it was written and absent from this dashboard until
    2026-07-31, which is a real gap rather than a cosmetic one: the partitions below are
    EXHAUSTIVE, so most of their rows are ordinary quarters and Fed plateaux, and the four
    crises anyone actually wants to look at are scattered across four different tables under
    four different names. This is the short list, and unlike the partitions it is deliberately
    NOT exhaustive — these are named stress windows, nothing else.
    """
    from common.regimes import EPISODES, coverage_fraction, episode_panel

    panel = episode_panel(metrics)
    if not panel:
        return
    with ui.expansion('🔥 The named crises (dot-com, GFC, COVID, 2022 …)', value=True) \
            .classes('w-full'):
        ui.label('Each cell is the episode\'s total return and the worst drawdown inside it, '
                 'over the strategy\'s own full history. "n/a" means its assets did not exist '
                 'yet — most leveraged variants show it for the GFC, and that visible gap is '
                 'the point: it makes it impossible to believe a 2x or 3x wrap survived 2008. '
                 'These windows are NOT exhaustive and do not tile the era; the partitions '
                 'below do.').classes('text-xs text-gray-500')
        _table([{'Strategy': d['name'],
                 **{label: ('n/a' if 'na' in panel[d['name']][key]
                            else f"{panel[d['name']][key]['return']:+.1%} / "
                                 f"{panel[d['name']][key]['max_dd']:.1%}")
                    for key, _s, _e, label in EPISODES},
                 'Covered': f"{coverage_fraction(panel[d['name']]):.0%}"}
                for d in metrics if d['name'] in panel])
        for key, start, end, label in EPISODES:
            ui.label(f'{label} — {pd.Timestamp(start):%Y-%m}..{pd.Timestamp(end):%Y-%m}') \
              .classes('text-xs text-gray-500')


def _regime_tables(metrics, rank_by='return'):
    """The named crises, then one table per pre-registered segmentation.

    Call inside a `with` block. Reads each strategy's FULL measurable history, never the
    ranked window, so the panels cannot inherit anybody's choice of dates.

    `rank_by` is passed straight through to the per-segment leaderboards so they order on the
    same metric as the main performance table.
    """

    _first, last = eras.era_bounds(metrics)
    if last is None:
        return
    _episode_panel(metrics)
    ui.label('🧭 Behaviour by pre-registered market regime').classes('text-lg font-bold mt-6')
    ui.label(f'Era {eras.COMMON_ERA_START[:7]}..{last:%Y-%m}, cut into segments dated by the '
             f'NBER, the FOMC, the BLS and the S&P 500. Each cell is the segment\'s total '
             f'return and the worst drawdown inside it, over that strategy\'s own history. '
             f'"~" = the strategy began inside the segment; "n/a" = its assets did not exist '
             f'yet.').classes('text-xs text-gray-500')

    shown = [seg for seg in eras.SEGMENTATIONS if seg.key in state['partitions']]
    if not shown:
        ui.label('No partition selected — tick at least one on the left. The named crises '
                 'above are unaffected.').classes('text-xs text-orange-700')
    for seg in shown:
        segments = eras.resolved_segments(seg, last)
        panel = eras.partition_panel(metrics, seg, last)
        if not panel or not segments:
            continue
        keys = [s.key for s in segments]
        if any(s.adverse for s in segments):
            keys.append('ADVERSE')
        with ui.expansion(seg.title, value=(seg.key == 'equity_cycle')).classes('w-full'):
            ui.label(f'Dated by: {seg.source}').classes('text-xs text-gray-500')
            _table([{'Strategy': d['name'],
                     **{k: _cell_text(panel[d['name']][k]) for k in keys}}
                    for d in metrics if d['name'] in panel])
            for s in segments:
                ui.label(f"{s.key} — {s.start:%Y-%m}..{s.end:%Y-%m}"
                         f"{' [adverse]' if s.adverse else ''} — {s.label}") \
                  .classes('text-xs text-gray-500')
            if seg.note:
                ui.label(f'Note: {seg.note}').classes('text-xs text-gray-500 italic')
            _segment_leaderboards(panel, segments, any(s.adverse for s in segments),
                                  metrics, rank_by=rank_by)


def _segment_leaderboards(panel, segments, has_adverse, metrics, rank_by='return'):
    """A growth chart and a ranked mini-table per segment, each in its own expansion.

    Call inside a `with` block.

    The table above answers "how did THIS strategy behave in each regime?". These answer
    the question it makes you compute by eye across twenty-five rows: "who LED this regime?".

    Only strategies covering a segment in full are ranked, and that restriction is the
    point rather than an omission — see `eras.segment_leaderboard`. The ones that entered
    late are named underneath with what they did cover.

    Carries every column the main performance table does, and ranks by the same chosen metric.
    Segments shorter than `metrics.SEGMENT_MIN_MONTHS` show `n/a` for the annualised columns and
    are ranked by total return instead — each panel's header states the key that ordered it,
    because a Sortino column full of `n/a` sorted by an invisible key looks like a ranking and
    is not one.
    """
    from common.metrics import SEGMENT_MIN_MONTHS

    def num(d, key, fmt):
        v = d.get(key)
        return format(v, fmt) if v is not None and np.isfinite(v) else 'n/a'

    keys = [(s.key, s) for s in segments] + ([('ADVERSE', None)] if has_adverse else [])
    with ui.expansion('🏆 Who led each segment').classes('w-full'):
        ui.label(f'Ranked by {rank_by.upper()}, the same metric as the table above. Only '
                 f'strategies that covered a segment IN FULL are ranked — one that arrived '
                 f'after the crash would show the recovery without the fall. Segments under '
                 f'{SEGMENT_MIN_MONTHS} months carry no annualised metrics (a CAGR over two '
                 f'months is that return raised to the sixth power) and are ranked by total '
                 f'return; their header says so. Descriptions of what each regime rewarded, '
                 f'never forecasts.').classes('text-xs text-gray-500')
        for key, s in keys:
            lb = eras.segment_leaderboard(panel, key, rank_by=rank_by)
            ranked, partial, absent = lb.ranked, lb.partial, lb.absent
            span = (f"{s.start:%Y-%m}..{s.end:%Y-%m}"
                    f"{' · adverse' if s.adverse else ''}") if s is not None \
                else 'every adverse month, compounded'
            by = (f"by {lb.rank_by.upper()}" if lb.rank_by == rank_by
                  else f"by RETURN — too short for {rank_by.upper()}")
            head = f"{key} — {span} — {len(ranked)} ranked, {by}"
            exp = ui.expansion(head).classes('w-full')
            with exp:
                # The section's own chart, drawn the first time this is opened. Re-based to
                # 1.0 at the segment start, so the lines answer "who compounded through THIS
                # regime" rather than restating the eighteen-year picture at a smaller scale.
                if s is not None:
                    _lazy_chart(exp, lambda s=s: (
                        _growth_chart(_rebased_curves(metrics, s.start, s.end), height=5.5),
                        ui.label('Re-based to 1.0 at the segment start. A line that begins '
                                 'later is a strategy that entered mid-segment — those are '
                                 'the ones named "not ranked" below.')
                          .classes('text-xs text-gray-500')))
                else:
                    _lazy_chart(exp, lambda: (
                        _growth_chart(_adverse_curves(metrics, segments), height=5.5,
                                      xlabel='adverse months, compounded (NOT a calendar)'),
                        ui.label('Every adverse month of the era, compounded end to end. The '
                                 'x axis counts months, not dates: these months are not '
                                 'contiguous, and a date axis would draw straight lines '
                                 'across the good years between them.')
                          .classes('text-xs text-gray-500')))
                if ranked:
                    _table([{'#': i + 1, 'Strategy': d['name'],
                             'Return': f"{d['return']:+.1%}",
                             'CAGR': num(d, 'cagr', '.2%'),
                             'Max drawdown': f"{d['max_dd']:.2%}",
                             'Sharpe': num(d, 'sharpe', '.2f'),
                             'Sortino': num(d, 'sortino', '.2f'),
                             'UPI': num(d, 'upi', '.2f'),
                             'Volatility': num(d, 'vol', '.1%'),
                             'Months': d['n_months']}
                            for i, d in enumerate(ranked)])
                else:
                    ui.label('Nothing covers this segment in full.') \
                      .classes('text-xs text-gray-500')
                if partial:
                    ui.label('Not ranked (entered late): '
                             + ', '.join(f'{n} ({m} mo)' for n, m, _ in partial)) \
                      .classes('text-xs text-gray-500')
                if absent:
                    ui.label(f'Absent ({len(absent)}): '
                             + ', '.join(f'{n} — {why}' for n, why in absent[:6])
                             + (', …' if len(absent) > 6 else '')) \
                      .classes('text-xs text-gray-500')


# Last computed backtest, kept so a sort-order change re-renders WITHOUT re-running the
# engine. Sorting is a presentation choice over data already in memory; until 2026-07-30 the
# only way to apply it was the Run button, which reloaded the store and re-priced every
# ledger to reorder rows it already had.
_bt_cache = {}


def _leverage_panel(advice):
    """How the "Max margin" column was derived, folded away under the table it explains.

    A number that says "1.19x" and nothing else is a number to be believed or ignored, and
    neither is useful. What makes it actionable is the CAP that produced it — a model stopped
    by its own drawdown is a different object from one stopped by a credit line, and only the
    second can be relaxed by phoning a broker.
    """
    if advice is None or not advice.by_name:
        return
    with ui.expansion('⚖️ Sustainable margin leverage — how the column was derived') \
            .classes('w-full'):
        ui.label(leverage_advice.headline(advice)).classes('text-sm')
        ui.label('This asks what leverage SURVIVES, not what optimises: there is no target '
                 'return and no search over the backtest anywhere in the chain. The engine '
                 'that produced the table above never compares equity to a maintenance '
                 'requirement, so every levered row it prints is an un-liquidated upper '
                 'bound. This is where that risk is priced.').classes('text-xs text-gray-500')
        ui.markdown(f'```\n{advice.table()}\n```').classes('w-full text-xs')
        ui.label('Assumed, and none of it your broker\'s:').classes('text-sm font-bold mt-2')
        for line in leverage_advice.assumption_lines(advice):
            ui.label('• ' + line).classes('text-xs text-gray-600')
        if advice.skipped:
            ui.label('Not sized: ' + '; '.join(f'{n} ({why})' for n, why in advice.skipped)) \
              .classes('text-xs text-orange-700')


def _robustness_panel(metrics, rank):
    """Everything this dashboard has been POINTING AT without showing. Call in a `with`.

    The summary card above says, and has said since the two-block split landed: "the CLI
    report prints the rank correlation between disjoint sub-periods, and it is approximately
    zero." A dashboard that cites a number it declines to display is asking to be trusted
    about the one figure that argues against its own leaderboard. Three measurements, in
    increasing order of how much they cost to compute:

      * SELECTION CONTEXT and RANK STABILITY — already written by `main._selection_section`,
        already printed by the CLI, never rendered here. Reused rather than reimplemented:
        two answers to "how many trials was this" is one too many, which is the same reason
        `selection.selection_trials` exists at all.
      * PBO by CSCV — every equal split of the history, not the two or three the calendar
        suggested. Deterministic.
      * The leaderboard rebuilt on resampled histories, so "rank 1" can be read next to "top
        three in 90% of alternative histories", which are different claims.
    """
    with ui.expansion('🎲 Robustness — how much of this ranking is skill, and how much is '
                      'search?').classes('w-full') as exp:
        def build():
            ui.label('None of this forecasts. A bootstrap draws from the distribution it is '
                     'given: if these two decades were generous to US assets, every resampled '
                     'path is generous too. What it measures is whether the ORDERING above '
                     'survives being computed on months you did not choose it from.') \
              .classes('text-xs text-gray-500')

            lines = engine._selection_section(metrics, rank, rank_descending(rank))
            if lines:
                ui.markdown('```\n' + '\n'.join(lines).strip('\n') + '\n```') \
                  .classes('w-full text-xs')

            try:
                frame, binding = robustness.common_frame(metrics)
            except NotCalculable as exc:
                ui.label(f'Not computed: {exc}').classes('text-xs text-orange-700')
                return
            # Strategy level first, then the same two measurements POOLED by family and by
            # de-risking mechanism — because "which mechanism is solid" is a different
            # question from "which variant won", and the variant answer is the unstable one.
            # `rf` is the rate the leaderboard's Sharpes were already netted against (AUD-02).
            rf = robustness.realised_rf(metrics)
            builders = [
                lambda: robustness.pbo_lines(
                    robustness.pbo(frame=frame, binding=binding, rf_annual=rf)),
                lambda: robustness.rank_lines(
                    robustness.rank_bootstrap(frame=frame, binding=binding,
                                              rank_key=rank, rf_annual=rf)),
            ]
            for grouping, fn in (('family', robustness.family_of),
                                 ('mechanism', robustness.mechanism_of)):
                builders.append(lambda fn=fn, grouping=grouping: robustness.group_pbo_lines(
                    robustness.group_pbo(frame=frame, binding=binding, groups=fn,
                                         grouping=grouping, rf_annual=rf)))
                builders.append(lambda fn=fn, grouping=grouping: robustness.group_rank_lines(
                    robustness.group_rank_bootstrap(frame=frame, binding=binding,
                                                    groups=fn, grouping=grouping,
                                                    rank_key=rank, rf_annual=rf)))
            for build_lines in builders:
                try:
                    ui.markdown('```\n' + '\n'.join(build_lines()).strip('\n') + '\n```') \
                      .classes('w-full text-xs')
                except NotCalculable as exc:
                    ui.label(f'Not computed: {exc}').classes('text-xs text-orange-700')
        _lazy_chart(exp, build)


def _frontier_chart(row):
    """The frontier for ONE strategy — the current sort's best sized entry — as two stacked
    panels sharing the leverage axis: growth above, margin-call probability below.

    Two panels rather than a twin y-axis, deliberately: a second y-scale on one plot makes
    two series look correlated at whatever ratio the axis limits happen to imply, and this
    chart exists to be read off, not to impress.
    """
    fs = sorted(row.curve)
    p_call = [row.curve[f][0] for f in fs]
    med = [row.curve[f][1] for f in fs]
    p5 = [row.curve[f][2] for f in fs]
    with ui.pyplot(figsize=(CHART_FIGSIZE[0], 6.2)).classes('w-full'):
        fig = plt.gcf()
        ax_g, ax_p = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax_g.plot(fs, med, lw=2, color='#1d4ed8', label='median CAGR')
        ax_g.plot(fs, p5, lw=1.6, ls='--', color='#60a5fa', label='5th percentile CAGR')
        ax_g.axhline(0.0, color='gray', lw=0.8, alpha=0.5)
        if not row.kelly_censored:
            ax_g.plot([row.kelly_f], [row.cagr_at_kelly], 'o', ms=6, color='#1d4ed8')
            ax_g.annotate(f'Kelly {row.kelly_f:.2f}x', xy=(row.kelly_f, row.cagr_at_kelly),
                          xytext=(4, 6), textcoords='offset points', fontsize=8,
                          color='#374151')
        ax_g.set_ylabel('CAGR across resampled histories')
        ax_g.legend(loc='lower left', fontsize=8, framealpha=0.85)
        ax_p.plot(fs, p_call, lw=2, color='#b91c1c')
        ax_p.set_ylabel('P(margin call)')
        ax_p.set_ylim(bottom=0)
        ax_p.set_xlabel('constant margin leverage f')
        for ax in (ax_g, ax_p):
            ax.grid(True, ls='--', alpha=0.4)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        if row.recommended is not None and np.isfinite(row.p_call_at_recommended):
            for ax in (ax_g, ax_p):
                ax.axvline(row.recommended, color='#374151', lw=1.2, ls=':')
            ax_p.annotate(f'advice {row.recommended:.2f}x — P(call) '
                          f'{row.p_call_at_recommended:.1%}',
                          xy=(row.recommended, 0.0), xytext=(6, 10),
                          textcoords='offset points', fontsize=8, color='#374151')
        fig.suptitle(f'{row.name} — maintenance {row.maintenance:.0%}', fontsize=10)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.09, hspace=0.08)


def _frontier_panel(metrics, advice, rank):
    """The margin decision as a curve, folded away under the sizing panel it checks.

    The Max margin column above says ONE number per model; this walks the same resampled
    histories the Robustness panel ranked, at every leverage level, and reports how often
    the broker would have called — including AT the recommended level, which is the
    cross-check between the closed-form cap and an independent path-walking method.
    """
    if advice is None or not advice.by_name:
        return
    with ui.expansion('📈 Leverage frontier — P(margin call) and growth, by leverage level') \
            .classes('w-full') as exp:
        def build():
            try:
                frame, binding = robustness.common_frame(metrics)
                maintenance, recommended = {}, {}
                for name, rec in advice.by_name.items():
                    m = rec.maintenance_margin_used
                    if m is not None and pd.notna(m):
                        maintenance[name] = float(m)
                        recommended[name] = rec.recommended_leverage
                res = robustness.leverage_frontier(
                    frame=frame, binding=binding, maintenance=maintenance,
                    recommended=recommended,
                    borrow_rate_annual=float(_uc.get('MARGIN_BORROW_RATE',
                                                     engine.MARGIN_BORROW_RATE)))
            except NotCalculable as exc:
                ui.label(f'Not computed: {exc}').classes('text-xs text-orange-700')
                return
            ui.markdown('```\n' + '\n'.join(robustness.frontier_lines(res)).strip('\n')
                        + '\n```').classes('w-full text-xs')
            # One chart, for the current sort's best entry among the sized rows — the whole
            # table is above it for everyone else.
            by_metric = {d['name']: d.get(rank) for d in metrics}
            candidates = [r for r in res.rows
                          if by_metric.get(r.name) is not None and pd.notna(by_metric[r.name])]
            if candidates:
                pick = (max if rank_descending(rank) else min)(
                    candidates, key=lambda r: by_metric[r.name])
                _frontier_chart(pick)
        _lazy_chart(exp, build)


def _common_window_table(metrics, rank):
    """Every SELECTED entry re-measured over the months they can ALL trade. Call in a `with`.

    The third answer to a question with no free option. The ranked table gives one window to
    the entries that reach it and measures the rest separately; forcing one window over
    everything (`RANKED_WINDOW_POLICY='all'`) makes every row comparable at the cost of the
    2007-2009 bear, which is the most discriminating stretch in the record. Neither is right
    for the question "how do THESE THREE compare", and that is the question people actually
    ask.

    So: intersect the months of whatever is on screen, re-run `calculate_metrics` on each
    entry's own returns restricted to that intersection, and rank the result. Strictly more
    informative than either global policy for a small selection, and strictly worse for a large
    one — the intersection is set by the LATEST arrival, so leaving one 2011 wrap ticked drags
    everything to 2011. The header says which entry is binding, so that is visible rather than
    discovered.

    Nothing is re-run through the engine: these are the same monthly returns the tables above
    are built from, sliced. Costs milliseconds, which is why it needs no button of its own
    beyond the expansion.
    """
    from common.metrics import calculate_metrics

    usable = [d for d in metrics
              if d.get('returns_full') is not None and len(d['returns_full']) > 1]
    if len(usable) < 2:
        ui.label('Tick at least two strategies to compare on a common window.') \
          .classes('text-xs text-gray-500')
        return
    idx = None
    for d in usable:
        i = d['returns_full'].index
        idx = i if idx is None else idx.intersection(i)
    if idx is None or len(idx) < 12:
        ui.label(f'These entries share only {0 if idx is None else len(idx)} month(s) — too '
                 f'few to annualise anything. Untick whatever arrived last.') \
          .classes('text-xs text-orange-700')
        return

    # Who is holding the window back. Named, because the remedy is to untick it.
    latest = max(pd.Timestamp(d['returns_full'].index[0]) for d in usable)
    binding = sorted(d['name'] for d in usable
                     if pd.Timestamp(d['returns_full'].index[0]) == latest)
    rows = []
    for d in usable:
        r = d['returns_full'].loc[idx]
        rf = d.get('rf_series')
        m = calculate_metrics(r, rf=rf.reindex(idx) if rf is not None else 0.0)
        rows.append({'name': d['name'], **m})
    rows.sort(key=lambda x: (x.get(rank) if pd.notna(x.get(rank)) else -1e9),
              reverse=rank_descending(rank))

    ui.label(f'{len(rows)} entries, all re-measured over {idx[0]:%Y-%m}..{idx[-1]:%Y-%m} '
             f'({len(idx)} months) — the months every one of them could trade.') \
      .classes('text-sm')
    ui.label('Window set by ' + ', '.join(binding) + ' — untick to lengthen it. Every number '
             'below is recomputed on the intersection, so none of them matches the tables '
             'above, and that is the point rather than a discrepancy.') \
      .classes('text-xs text-gray-500')
    _table([{'Rank': i + 1, 'Strategy': d['name'], 'CAGR': PCT(d['cagr']),
             'Max Drawdown': PCT(d['max_dd']), 'Sharpe': NUM(d['sharpe']),
             'Sortino': NUM(d['sortino']), 'UPI': NUM(d.get('upi', float('nan'))),
             'Volatility': PCT(d['vol']), 'Months': d['n_periods']}
            for i, d in enumerate(rows)])


def _render_ranked_block(container, metrics, rank, advice=None, regime_container=None):
    """Everything that depends on the rank key: summary card, table, regime sections.

    Two containers, because the whole-era CHART sits between them and does not depend on the
    sort key — see `run_backtest_clicked`. With `regime_container` omitted the regime sections
    are written into `container` like everything else.
    """
    container.clear()
    if regime_container is not None:
        regime_container.clear()

    def order(rows):
        # Direction from `common.metrics.rank_descending`, not a local `rank != 'vol'` — the CLI
        # report, this table and the per-segment leaderboards must not be able to disagree about
        # which end of a metric is better.
        return sorted(rows, key=lambda d: (d.get(rank) if pd.notna(d.get(rank)) else -1e9),
                      reverse=rank_descending(rank))

    # TWO BLOCKS, as the CLI report has had since REPORT-002 and this dashboard had not.
    # An entry whose own products did not exist at the shared window's start is measured over
    # its OWN history, and its row is therefore NOT comparable with the rest. Sorting the two
    # sets into one list ranked a 2010-02 record against a 2008-07 one and printed, underneath,
    # "the same for every row here" — which was false whenever a late entry was on screen. It
    # flattered the late ones by exactly the months they missed: HAA_G4_Leveraged_2X ranked
    # second on a record that begins after the GFC it never traded.
    ranked = order([d for d in metrics if d.get('in_ranked_window', True)])
    off_window = order([d for d in metrics if not d.get('in_ranked_window', True)])
    with container:
        if not ranked and not off_window:
            ui.label('No results for this period.').classes('text-negative')
            return
        if not ranked:
            # Every selected entry starts late — there is no shared window to rank against.
            ranked, off_window = off_window, []
        best = ranked[0]
        win = next((d for d in ranked if d.get('window_start') is not None), None)
        if win is not None:
            why = (f'set by {win["window_binding"]}, the last of these to exist'
                   if win.get('window_binding') else 'the era floor')
            ui.label(f'Ranked window {pd.Timestamp(win["window_start"]):%Y-%m}..'
                     f'{pd.Timestamp(win["window_end"]):%Y-%m} — identical for every row, '
                     f'{why}.').classes('text-xs text-gray-500')
        ui.label(f'🏆 Top by {RANK_OPTIONS[rank]}: {best["name"]}').classes('text-lg font-bold')
        with ui.row().classes('gap-6'):
            for label, val in [('CAGR', PCT(best['cagr'])), ('Max DD', PCT(best['max_dd'])),
                               ('Sharpe', NUM(best['sharpe'])), ('Sortino', NUM(best['sortino'])),
                               ('UPI', NUM(best.get('upi', float('nan')))),
                               ('Volatility', PCT(best['vol']))]:
                with ui.column().classes('items-center'):
                    ui.label(label).classes('text-xs text-gray-500')
                    ui.label(val).classes('text-lg font-mono')
        ui.label('Ranking describes what this window rewarded. It does not forecast the next '
                 'one — the rank correlation between disjoint sub-periods is approximately '
                 'zero, and the Robustness panel below now shows it here rather than citing '
                 'a CLI report nobody has open.').classes('text-xs text-gray-500')

        ui.label('Performance comparison').classes('text-lg font-bold mt-4')
        by_name = advice.by_name if advice is not None else {}

        def margin_cells(name):
            return leverage_advice.cell(by_name.get(name)) if by_name else ('—', '')

        def perf_table(rows):
            _table([{'Rank': i + 1, 'Strategy': d['name'], 'CAGR': PCT(d['cagr']),
                     'Max Drawdown': PCT(d['max_dd']), 'Sharpe': NUM(d['sharpe']),
                     'Sortino': NUM(d['sortino']), 'UPI': NUM(d.get('upi', float('nan'))),
                     'Volatility': PCT(d['vol']),
                     # What margin this record could carry, and what stopped it there. Two
                     # columns rather than one because 1.00x from the Kelly gate ("borrowed
                     # money would have reduced growth") and 1.00x from margin survival ("the
                     # broker would have closed you out") are opposite findings that print the
                     # same number.
                     'Max margin': margin_cells(d['name'])[0],
                     'Capped by': margin_cells(d['name'])[1],
                     'Measured': (f"{pd.Timestamp(d['first_return']):%Y-%m}"
                                  f"..{pd.Timestamp(d['last_return']):%Y-%m}"
                                  if d.get('first_return') is not None else 'n/a'),
                     'Turn/yr': NUM(d.get('annual_turnover', float('nan'))),
                     'Coverage': (f"trimmed ({d.get('binding_ticker')})"
                                  if d.get('coverage_trimmed') else 'full')}
                    for i, d in enumerate(rows)])

        perf_table(ranked)
        ui.label('"Measured" is the window actually evaluated — the same for every row in THIS '
                 'table, which is what makes the ranking mean anything. '
                 '"Coverage: trimmed" means that strategy could not be measured from the era '
                 'floor and is what shortened the window for everyone. '
                 '"Turn/yr" is one-way notional traded per year as a multiple of equity. '
                 '"Max margin" is the borrowed leverage this record survives indefinitely at '
                 'the safety factor set on the left — a 2x/3x row already carries its leverage '
                 'inside the product, so its column is margin ON TOP of that, and it is '
                 'arithmetically unavailable.').classes('text-xs text-gray-500')

        if off_window:
            ui.label(f'⏳ Measured over their OWN history — not comparable with the table above '
                     f'({len(off_window)})').classes('text-lg font-bold mt-6')
            ui.label('These entries\' own products did not exist when the shared window opens, '
                     'so there is no honest way to give them a row in it. They are ranked here '
                     'among themselves, over windows that differ from the table above AND from '
                     'each other — read the "Measured" column before comparing any two numbers '
                     'on this screen. A record that begins after a crash shows the recovery '
                     'without the fall, and that is worth several points of CAGR.') \
              .classes('text-xs text-orange-700')
            perf_table(off_window)

        # The one view that puts EVERY ticked entry on identical months, whatever their
        # inception. Collapsed by default because it answers a narrower question than the
        # table above and answers it on a shorter window.
        with ui.expansion('🔗 Compare the whole selection on their COMMON window') \
                .classes('w-full'):
            _common_window_table(ranked + off_window, rank)

        # Sizing covers BOTH blocks — the caps are per-record and need no shared window.
        _leverage_panel(advice)

        # How much of the ordering above is a property of the strategies and how much of
        # having looked. Lazy: PBO and the resampled ranking together cost roughly a second,
        # and re-paying that on every change of sort key is exactly the regression that made
        # the table slow before.
        _robustness_panel(metrics, rank)

        # The margin decision as a curve — the same resampled histories, walked at leverage,
        # with the Max margin advice evaluated on them as the cross-check.
        _frontier_panel(metrics, advice, rank)

    with (regime_container if regime_container is not None else container):
        # BOTH blocks here, deliberately, and it is not a contradiction of the split above: the
        # regime panels are measured over each strategy's FULL history and every cell states its
        # own coverage, so a late entrant appears as "~" or "n/a" rather than as a comparable
        # number. That is the one place the two sets can be read side by side honestly.
        _regime_tables(ranked + off_window, rank_by=rank)


def rank_changed():
    """Re-sort the results already on screen — instantly, from memory.

    Only the rank-dependent block is rebuilt; the whole-era growth chart and the
    latest-allocation table do not depend on the sort key and are left untouched (the chart
    alone costs ~a second of matplotlib to redraw, for zero change). The engine is not
    consulted at all: re-sorting is not a new measurement. A change to leverage, costs, the
    safety factor or the ticked strategies still requires the Run button, because those
    change the NUMBERS.
    """
    if _bt_cache.get('metrics') is not None and _bt_cache.get('ranked_area') is not None:
        _render_ranked_block(_bt_cache['ranked_area'], _bt_cache['metrics'], rank_in.value,
                             advice=_bt_cache.get('advice'),
                             regime_container=_bt_cache.get('regime_area'))


async def run_backtest_clicked():
    names = selected_names()
    if not names:
        ui.notify('Tick at least one strategy.', type='warning')
        return
    run_bt_btn.disable()
    bt_results.clear()
    with bt_results:
        ui.spinner(size='lg')
        ui.label('Downloading data, running backtests & sizing the leverage each one carries…')
    try:
        prices, s_w, s_u, strategies, metrics, results, advice = await run.io_bound(
            _compute_backtest, names,
            float(lev_in.value), float(borrow_in.value) / 100.0, float(txn_in.value) / 100.0,
            bool(follows_in.value), float(k_in.value), float(maint_in.value) / 100.0)
    except Exception as e:
        bt_results.clear()
        with bt_results:
            ui.label(f'Backtest failed: {e}').classes('text-negative')
        return
    finally:
        run_bt_btn.enable()

    # THREE areas, not two, and the order matters. The whole-era section reads summary ->
    # table -> chart -> holdings, and the regime sections come after all of it — so the
    # rank-dependent parts cannot be one contiguous block, or the chart would be pushed below
    # every regime panel. The chart sits in its own static area between them and survives a
    # re-sort untouched.
    bt_results.clear()
    with bt_results:
        head_area = ui.column().classes('w-full')      # rank-dependent: summary, table, sizing
        chart_area = ui.column().classes('w-full')     # static: whole-era chart, holdings
        regime_area = ui.column().classes('w-full')    # rank-dependent: the period sections
    _bt_cache.update({'metrics': metrics, 'advice': advice,
                      'ranked_area': head_area, 'regime_area': regime_area})

    _render_ranked_block(head_area, metrics, rank_in.value, advice=advice,
                         regime_container=regime_area)

    with chart_area:
        ui.label('Cumulative growth over the whole measured era (log scale)') \
          .classes('text-lg font-bold mt-4')
        ui.label('Colour = family (same hue as the swatch in the picker). Dash pattern and '
                 'marker = variant within that family, and line WIDTH = leverage ratio '
                 '(1x thin, 2x medium, 3x thick). Each line is named at its right end, '
                 'in its own colour — the legend is only the fallback for lines that stop '
                 'before the right edge. The same chart is drawn again inside every regime '
                 'section below, re-based to that segment.').classes('text-xs text-gray-500')
        _growth_chart(results)
        ui.label('🎯 Latest target allocation').classes('text-lg font-bold mt-4')
        _table(_latest_weights(strategies, prices, s_w, s_u))


async def run_live_clicked():
    names = selected_names()
    if not names:
        ui.notify('Tick strategies in the Backtest tab first.', type='warning')
        return
    accounts = state['accounts']
    knobs = {'SAFETY_MARGIN_PCT': float(safety_in.value), 'FLEXIBILITY_BAND_PCT': float(flex_in.value),
             'FLUSH_ROUND_UP_BAND_PCT': float(flush_in.value), 'PRICE_CAP_MARGIN_PCT': float(cap_in.value),
             'MINIMUM_TRADE_PCT': float(min_trade_in.value),
             'SHARE_LOT_SIZE': int(lot_in.value), 'FRACTIONAL_SHARES': bool(frac_in.value)}
    run_live_btn.disable()
    live_results.clear()
    with live_results:
        ui.spinner(size='lg')
        ui.label('Computing live orders…')
    try:
        signal_date, results = await run.io_bound(
            _compute_live, names, exec_date_in.value, [dict(a) for a in accounts], knobs)
    except Exception as e:
        live_results.clear()
        with live_results:
            ui.label(f'Live computation failed: {e}').classes('text-negative')
        return
    finally:
        run_live_btn.enable()

    live_results.clear()
    with live_results:
        if signal_date is None:
            ui.label('Not enough price history before the execution date.').classes('text-negative')
            return
        ui.label(f'Live signals as of {signal_date.date()}').classes('text-xl font-bold')
        for res in results:
            with ui.card().classes('w-full'):
                ui.label(res['name']).classes('text-lg font-bold')
                if res['error']:
                    ui.label(f'Error computing signal: {res["error"]}').classes('text-negative')
                    continue
                sizing = res['sizing']
                p = res['pricing']
                ui.label(('Sized at LIVE quotes (' + p['asof'] + ')' if p['mode'] == 'live'
                          else 'Sized at month-end close ' + p['asof']) +
                         f' — signal: month-end {signal_date.date()}').classes('text-xs text-gray-500')
                with ui.row().classes('items-center gap-2'):
                    for c in res['canary']:
                        ui.badge(f'{c["asset"]}: {c["state"]} ({c["score"]:.2f})',
                                 color='green' if c['state'] == 'ALIVE' else 'red')
                    if sizing['total_off_wt'] > 0 or sizing['total_def_wt'] > 0:
                        ui.label(f'Offensive {PCT(sizing["total_off_wt"])} / '
                                 f'Defensive {PCT(sizing["total_def_wt"])}').classes('text-sm')
                    elif sizing['orders']:
                        # Same rule as the CLI: an unresolved sleeve is said, never blank.
                        ui.label('Mode UNRESOLVED — check sleeves()').classes(
                            'text-sm text-orange-600')
                for w in sizing['warnings']:
                    ui.label(f'⚠️ {w}').classes('text-orange-600 text-sm')
                _table([{'Asset': o['asset'], 'Name': o['name'], 'Type': o['mode'], 'Rank': o['rank'],
                         'Account': o['account'], 'Target %': PCT(o['target_wt']),
                         'Actual %': PCT(o['actual_wt']), 'Shares': f"{o['shares']:.2f}",
                         'Quote': f"${o['price']:,.2f}",
                         'Cap Price': f"${o['cap']:,.2f}" if o['cap'] is not None else '—',
                         'Max Cost': f"${o['value']:,.2f}", 'Fill': o['fill']}
                        for o in sizing['orders']])
                ui.label('Remaining cash: ' + ' | '.join(
                    f"{a['account_name']} ${a['account_balance']:,.2f}" for a in res['accounts'])) \
                  .classes('text-sm text-gray-600')


# --------------------------------------------------------------------------- #
# Entry point — native desktop window when pywebview is available.
# `python app.py --browser` forces browser mode (fallback if the native window
# fails on your system, e.g. missing WebView2 runtime).
# --------------------------------------------------------------------------- #
import sys
NATIVE = importlib.util.find_spec('webview') is not None and '--browser' not in sys.argv

if __name__ in {'__main__', '__mp_main__'}:
    if NATIVE:
        ui.run(native=True, window_size=(1500, 950), title='Keller Strategies', reload=False)
    else:
        ui.run(title='Keller Strategies', reload=False, show=True)
