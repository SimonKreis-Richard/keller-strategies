"""
HAA_G12_CA — this month's HAA_G12 decision, turned into orders for Canadian-listed funds.

    venv/Scripts/python.exe -m tools.ca_orders                  # reads ca_execution.json
    venv/Scripts/python.exe -m tools.ca_orders --rate 1.3850    # USD/CAD typed by hand
    venv/Scripts/python.exe -m tools.ca_orders --no-journal     # print only, write nothing
    venv/Scripts/python.exe -m tools.ca_orders --demo           # offline, fictional capital

WHAT IT DOES, IN ORDER, AND WHERE EACH STEP CAN REFUSE
------------------------------------------------------
1. Reads `ca_execution.json` — the cash and holdings, gitignored like `user_config.json`.
   Strict: an unknown key, a US ticker, a negative balance or `leverage > 1` stops it here,
   before a single byte is downloaded. `ca_execution.example.json` is the template.
2. Computes the HAA_G12 signal from the engine's own price store, exactly as the live path
   does — same panel, same month-end rule, same guards (`common/live_guards.py`). The signal
   is computed on the US tickers and is never touched afterwards: HAA_G12_CA changes what is
   HELD, never what is CHOSEN, and `ca_execution.assert_weights_unchanged` says so on every
   run. HAA_G12_CA is not a strategy and is not in the registry.
3. Fetches the last close of each Canadian fund from Yahoo (`.TO` for the TSX, `.NE` for
   Cboe Canada — symbols verified 2026-09-23). A holding or a target with no quote, or a
   quote older than `MAX_QUOTE_AGE_DAYS`, is a refusal, not a stale number.
4. Obtains USD/CAD from `tools/fx_rate.py`: `--rate` if given, else the Bank of Canada, else
   Yahoo. There is no default rate.
5. Builds the orders with `common/ca_execution.build_orders` — sells first, one conversion if
   a currency is short, then buys in whole units — and checks that no US ticker reached them.
6. Prints them and journals the run under `logs/execution_ca/` (gitignored as a directory):
   a JSON with every input and its provenance, and the rotation CSV, one row per trade, each
   naming the signal date that authorised it.

IT NEVER PLACES AN ORDER. The output is a plan to type into a broker. Fills, and therefore the
real FX rate and the real prices, come back from the broker, not from here.

LEVERAGE STAYS AT 1x (EXEC-001). `execution.leverage` above 1.0 is refused, for the reason
`common/ca_execution.py` gives.

The only sockets are in steps 2-4, and each fetcher is an argument of a pure function, so
`tests/test_ca_orders.py` drives the whole pipeline offline.
"""
import argparse
import csv
import datetime
import io
import json
import math
import os
import sys
from collections import namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common import ca_execution as cax                     # noqa: E402
from common import ca_mapping as cam                       # noqa: E402
from common.ca_execution import CAExecutionConfig, CAExecutionError, FxRate  # noqa: E402

CONFIG_PATH = os.path.join(ROOT, 'ca_execution.json')
EXAMPLE_PATH = os.path.join(ROOT, 'ca_execution.example.json')
DEMO_INPUT = os.path.join(ROOT, 'examples', 'ca_demo.json')
DEMO_CSV = os.path.join(ROOT, 'examples', 'ca_rotation_demo.csv')
JOURNAL_DIR = os.path.join(ROOT, 'logs', 'execution_ca')

STRATEGY = 'HAA_G12'

#: Yahoo's suffix per listing venue. Verified 2026-09-23 against every ticker in the table:
#: the three Cboe Canada funds (ZTL, ZTM, ZCOM) answer ONLY as `.NE` and return nothing as
#: `.TO`, which is also an independent check on `ca_mapping`'s exchange column.
YAHOO_SUFFIX = {'TSX': '.TO', 'Cboe CA': '.NE'}

#: A close older than this, in calendar days before the run date, is refused. Seven covers a
#: long weekend plus a thin `.U` class that did not trade for a day or two; a fund that has
#: not printed in a week is not one to size an order against.
MAX_QUOTE_AGE_DAYS = 7

#: How far back the quote download looks. Wider than `MAX_QUOTE_AGE_DAYS` so that a stale
#: quote is SEEN and refused by name, rather than missing and reported as "no data".
QUOTE_LOOKBACK_DAYS = 21

CONFIG_KEYS = frozenset({'_comment', 'cash', 'positions', 'execution', 'usdcad', 'prices'})
EXECUTION_KEYS = frozenset({'leverage', 'prefer_usd_units', 'fx_cost_bps', 'fx_cost_min_usd',
                            'min_trade_pct'})

Signal = namedtuple('Signal', 'weights signal_date canary stale_note provenance')


class ConfigError(ValueError):
    """`ca_execution.json` is missing or says something this driver will not guess about."""


# --------------------------------------------------------------------------------------- #
#  1. the configuration
# --------------------------------------------------------------------------------------- #

def _number(value, where, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ConfigError(f'{where} must be a number, got {value!r}.')
    if value < 0 or (positive and value == 0):
        raise ConfigError(f'{where} must be {"positive" if positive else "zero or more"}, '
                          f'got {value!r}.')
    return float(value)


def currency_of(ticker, mapping=None):
    """'CAD' or 'USD' — the trading currency of one execution ticker, read off the table."""
    mapping = cam.CA_MAPPING if mapping is None else mapping
    for entry in mapping.values():
        if ticker == entry.exec_ticker_cad:
            return 'CAD'
        if ticker == entry.exec_ticker_usd:
            return 'USD'
    raise ConfigError(f'{ticker} is not an execution ticker in common/ca_mapping.py.')


def exchange_of(ticker, mapping=None):
    mapping = cam.CA_MAPPING if mapping is None else mapping
    for entry in mapping.values():
        if ticker in cam.execution_tickers(entry):
            return entry.exchange
    raise ConfigError(f'{ticker} is not an execution ticker in common/ca_mapping.py.')


def parse_config(raw, mapping=None):
    """Validate a decoded `ca_execution.json`. Returns a dict the pipeline can use as is.

    Unknown keys are refused rather than ignored: a misspelt `"positons"` would otherwise
    produce a book with no holdings and orders to buy everything a second time.
    """
    mapping = cam.CA_MAPPING if mapping is None else mapping
    if not isinstance(raw, dict):
        raise ConfigError('the configuration must be a JSON object.')
    unknown = sorted(set(raw) - CONFIG_KEYS)
    if unknown:
        raise ConfigError(f'unknown key(s) {unknown}; allowed: {sorted(CONFIG_KEYS)}.')

    cash_raw = raw.get('cash') or {}
    if not isinstance(cash_raw, dict) or set(cash_raw) - {'CAD', 'USD'}:
        raise ConfigError('"cash" must be an object with at most the keys "CAD" and "USD".')
    cash = {c: _number(cash_raw.get(c, 0.0), f'cash.{c}') for c in ('CAD', 'USD')}

    positions = []
    for i, p in enumerate(raw.get('positions') or []):
        if not isinstance(p, dict) or set(p) - {'ticker', 'qty', 'currency'} \
                or 'ticker' not in p or 'qty' not in p:
            raise ConfigError(f'positions[{i}] must be {{"ticker": ..., "qty": ...}}, '
                              f'got {p!r}.')
        cam.assert_no_us_tickers([p['ticker']], mapping)
        currency = currency_of(p['ticker'], mapping)
        if p.get('currency') not in (None, currency):
            raise ConfigError(f'positions[{i}]: {p["ticker"]} trades in {currency}, not '
                              f'{p["currency"]}.')
        positions.append({'ticker': p['ticker'],
                          'qty': _number(p['qty'], f'positions[{i}].qty', positive=True),
                          'currency': currency})

    execution = raw.get('execution') or {}
    if not isinstance(execution, dict) or set(execution) - EXECUTION_KEYS:
        raise ConfigError(f'"execution" may only contain {sorted(EXECUTION_KEYS)}.')
    exec_config = CAExecutionConfig(**execution)          # refuses leverage > 1 (EXEC-001)

    usdcad = raw.get('usdcad')
    if usdcad is not None:
        usdcad = _number(usdcad, 'usdcad', positive=True)

    prices = {}
    for ticker, px in (raw.get('prices') or {}).items():
        cam.assert_no_us_tickers([ticker], mapping)
        prices[ticker] = _number(px, f'prices.{ticker}', positive=True)

    return {'cash': cash, 'positions': positions, 'exec_config': exec_config,
            'usdcad': usdcad, 'prices': prices}


def load_config(path=None, mapping=None):
    path = path or CONFIG_PATH
    if not os.path.exists(path):
        raise ConfigError(
            f'{path} does not exist. Copy ca_execution.example.json to ca_execution.json and '
            f'enter your own cash and holdings; the file is gitignored, like '
            f'user_config.json, and must stay that way.')
    with open(path, encoding='utf-8') as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(f'{path} is not valid JSON: {exc}') from exc
    return parse_config(raw, mapping)


# --------------------------------------------------------------------------------------- #
#  2. the signal — the engine's, untouched
# --------------------------------------------------------------------------------------- #

def compute_signal(exec_date=None):
    """HAA_G12's current decision, through the same guards as `main.compute_live_signals`.

    The configuration comes from `tools.backtest_driver.build_config`, which nails the live
    path shut: no `BROKER_ACCOUNTS` ever enters this process, so nothing here can print a
    brokerage balance. `EXECUTION_MODE=True` is set on a COPY passed to `build_signal_panel`
    alone, because that is how the panel learns that a month the calendar has closed may be
    used before a later bar proves it — the live month-end rule, and the only key it reads.
    """
    import pandas as pd
    import main as engine
    from common.letf_mapper import assert_unlevered_defensive
    from common.live_guards import live_guards, refusal_message
    from tools.backtest_driver import build_config

    exec_dt = pd.Timestamp(exec_date or datetime.date.today()).normalize()
    cfg = build_config(CACHE_REFRESH_HOURS=engine.LIVE_MAX_REFRESH_HOURS)
    store = engine.load_store(cfg)
    panel_cfg = dict(cfg, EXECUTION_MODE=True, CURRENT_EXECUTION_DATE=str(exec_dt.date()))
    prices, scores_w, scores_u = engine.build_signal_panel(store, panel_cfg)

    available = prices.index[prices.index < exec_dt]
    if len(available) == 0:
        raise CAExecutionError('the price store holds no complete month before '
                               f'{exec_dt.date()}.')
    signal_date = available[-1]
    stale_months = (pd.Period(exec_dt, freq='M') - pd.Period(signal_date, freq='M')).n
    stale_note = None
    if stale_months >= 2:
        stale_note = (f'SIGNAL IS {stale_months} MONTHS OLD: the latest complete month-end in '
                      f'the store is {signal_date.date()}. These orders are not this month\'s '
                      f'rotation. Refresh the price cache before trading.')

    strat = engine.ALL_STRATEGIES[STRATEGY]()
    assert_unlevered_defensive(strat)
    cam.validate_covers(strat)
    s_w = (scores_u if strat.score_type == 'unweighted' else scores_w).loc[:signal_date]
    alloc = strat.generate_allocations(prices.loc[:signal_date], s_w, None, None)
    problems = live_guards(strat, alloc, prices, signal_date, store)
    if problems:
        raise CAExecutionError(refusal_message(problems))

    row = alloc.iloc[-1]
    weights = {t: float(w) for t, w in row.items() if w > 1e-9}
    canary = {c: float(s_w.iloc[-1][c]) for c in strat.canary}
    return Signal(weights, str(signal_date.date()), canary, stale_note, store.provenance())


# --------------------------------------------------------------------------------------- #
#  3. the quotes
# --------------------------------------------------------------------------------------- #

def yahoo_symbol(ticker, mapping=None):
    """`ZSP.U` on the TSX -> `ZSP-U.TO`; `ZTL` on Cboe Canada -> `ZTL.NE`."""
    exchange = exchange_of(ticker, mapping)
    suffix = YAHOO_SUFFIX.get(exchange)
    if suffix is None:
        raise ConfigError(f'{ticker}: no Yahoo suffix is known for the exchange {exchange!r}.')
    return ticker.replace('.', '-') + suffix


def fetch_quotes(tickers, as_of, mapping=None, download=None):
    """`{ticker: (close, 'YYYY-MM-DD')}` — the last close on or before `as_of`.

    A ticker Yahoo returns nothing for is ABSENT from the result, never zero; `check_quotes`
    turns an absence into a refusal or a flag. Raw `Close`, not `Adj Close`: an order is sized
    at the price that trades.
    """
    import pandas as pd
    as_of = pd.Timestamp(as_of).normalize()
    symbols = {t: yahoo_symbol(t, mapping) for t in sorted(tickers)}
    if not symbols:
        return {}
    if download is None:
        import contextlib
        import warnings
        import yfinance as yf

        def download(*a, **k):
            with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return yf.download(*a, **k)
    frame = download(sorted(symbols.values()),
                     start=str((as_of - pd.Timedelta(days=QUOTE_LOOKBACK_DAYS)).date()),
                     end=str((as_of + pd.Timedelta(days=1)).date()),
                     progress=False, auto_adjust=False, group_by='column')
    if frame is None or len(frame) == 0:
        return {}
    closes = frame['Close']
    if not hasattr(closes, 'columns'):                   # a single symbol comes back flat
        closes = closes.to_frame(next(iter(symbols.values())))
    out = {}
    for ticker, symbol in symbols.items():
        if symbol not in closes.columns:
            continue
        series = closes[symbol].dropna()
        series = series[series.index <= as_of]
        series = series[series > 0]
        if not series.empty:
            out[ticker] = (float(series.iloc[-1]), str(series.index[-1].date()))
    return out


def quote_tickers(weights, positions, mapping=None):
    """(required, optional). Required: everything held, and the class of each target sleeve
    that a CAD book would buy (the CAD class, or the only class). Optional: the other class,
    which `build_orders` only reaches when USD cash is already there to pay for it."""
    mapping = cam.CA_MAPPING if mapping is None else mapping
    required = {p['ticker'] for p in positions}
    optional = set()
    for sleeve in weights:
        entry = mapping[sleeve]
        primary = entry.exec_ticker_cad or entry.exec_ticker_usd
        required.add(primary)
        optional.update(t for t in cam.execution_tickers(entry) if t != primary)
    return required, optional - required


def check_quotes(required, optional, quotes, as_of):
    """(prices, problems, notes). A problem is a refusal; a note is printed with the orders."""
    import pandas as pd
    as_of = pd.Timestamp(as_of).normalize()
    prices, problems, notes = {}, [], []
    for ticker in sorted(required | optional):
        if ticker not in quotes:
            (problems if ticker in required else notes).append(
                f'{ticker}: no quote was returned'
                + ('' if ticker in required else ' (an alternative class; it will not be '
                                                 'used)'))
            continue
        price, day = quotes[ticker]
        age = (as_of - pd.Timestamp(day)).days
        if age > MAX_QUOTE_AGE_DAYS:
            (problems if ticker in required else notes).append(
                f'{ticker}: the last close is {day}, {age} days before {as_of.date()} — too '
                f'old to size an order against')
            continue
        prices[ticker] = price
        if age > 3:
            notes.append(f'{ticker}: last close {day} ({age} days old)')
    return prices, problems, notes


# --------------------------------------------------------------------------------------- #
#  5. the orders — a pure function of everything fetched above
# --------------------------------------------------------------------------------------- #

def build(config, signal, prices, fx, mapping=None):
    """Orders for `signal`, sized against `config`'s book at `prices` and `fx`. No I/O."""
    if not isinstance(fx, FxRate):
        raise CAExecutionError('an FxRate is required; there is no default.')
    all_prices = dict(prices)
    all_prices.update(config['prices'])                  # a typed price always wins
    result = cax.build_orders(signal.weights, config['cash'], config['positions'], all_prices,
                              fx, config=config['exec_config'],
                              signal_date=signal.signal_date, mapping=mapping)
    cax.assert_weights_unchanged(signal.weights, result)
    cam.assert_no_us_tickers([o['ticker'] for o in result['orders']], mapping)
    return result


# --------------------------------------------------------------------------------------- #
#  6. presentation and the journal
# --------------------------------------------------------------------------------------- #

def render(result, signal, notes=()):
    """The orders as the operator will type them: sells, the conversion, then buys."""
    lines = [f'HAA_G12_CA — orders for the HAA_G12 signal of {signal.signal_date}']
    if signal.stale_note:
        lines += ['', f'  !!! {signal.stale_note}']
    for c, score in signal.canary.items():
        lines.append(f'  canary {c} 13612U {score:+.4f}: '
                     f'{"ALIVE — risk on" if score > 0 else "DEAD — fully defensive"}')
    lines.append('  target weights, copied from HAA_G12 unchanged: '
                 + '  '.join(f'{t} {w:.1%}' for t, w in sorted(signal.weights.items())))
    fx = result['fx']
    lines.append(f'  USD/CAD {fx["usdcad"]:.4f}   {fx["source"]}   as of {fx["as_of"]}')
    lines.append(f'  book before: {result["nav_cad_before"]:,.2f} CAD')
    lines.append('')
    head = f'  {"":7} {"qty":>9}  {"ticker":<8} {"exchange":<8} {"ccy":<4} {"price":>10} ' \
           f'{"value":>13}  for'
    lines += [head, '  ' + '-' * (len(head) - 2)]

    def order_line(o):
        return (f'  {o["side"]:<7} {o["qty"]:>9,.0f}  {o["ticker"]:<8} {o["exchange"]:<8} '
                f'{o["currency"]:<4} {o["price"]:>10,.4f} {o["value"]:>13,.2f}  '
                f'{o["signal_ticker"]}')
    lines += [order_line(o) for o in result['orders'] if o['side'] == 'SELL']
    for f in result['fx_orders']:
        lines.append(f'  CONVERT {f["description"]}  (commission {f["cost_usd"]:,.2f} USD)')
    lines += [order_line(o) for o in result['orders'] if o['side'] == 'BUY']
    if not result['orders'] and not result['fx_orders']:
        lines.append('  nothing to trade: the book already matches the target')
    cash = result['residual_cash']
    lines += ['', f'  cash left: {cash["CAD"]:,.2f} CAD   {cash["USD"]:,.2f} USD   '
                  f'(book after: {result["nav_cad_after"]:,.2f} CAD)']
    flags = list(dict.fromkeys(result['flags']))
    if flags or notes:
        lines += ['', '  to read before trading:']
        lines += [f'    [{sev}] {msg}' for sev, msg in flags]
        lines += [f'    [quote] {n}' for n in notes]
    lines += ['', '  Nothing has been sent to a broker. Prices are last closes and the FX rate '
                  'is indicative;', '  the fills are what count.']
    return lines


def journal(result, signal, quotes, notes, directory=None, now=None):
    """Write the run to `logs/execution_ca/`: `<signal>_<stamp>.json` and `.csv`.

    `logs/` is gitignored as a directory because these files carry real quantities and real
    dollar values. The JSON holds every input and its provenance, so a trade can be traced back
    to the quote, the rate and the price-store hash that produced it.
    """
    directory = directory or JOURNAL_DIR
    os.makedirs(directory, exist_ok=True)
    now = now or datetime.datetime.now()
    stem = os.path.join(directory, f'{signal.signal_date}_{now:%Y%m%dT%H%M%S}')
    record = {'written_at': now.isoformat(timespec='seconds'), 'strategy': STRATEGY,
              'signal': signal._asdict(), 'quotes': quotes, 'quote_notes': list(notes),
              'result': result}
    with open(stem + '.json', 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1, default=str)
    with open(stem + '.csv', 'w', encoding='utf-8', newline='') as fh:
        fh.write(cax.rotation_csv(result))
    return stem + '.json', stem + '.csv'


# --------------------------------------------------------------------------------------- #
#  the demonstration — offline, fictional capital, real signals and real prices
# --------------------------------------------------------------------------------------- #

def run_demo(path=None):
    """Replay `examples/ca_demo.json`: two months of HAA_G12_CA on a fictional book.

    Returns (results, csv_text). The signals, closes and FX rates in the file are RECORDED —
    what the engine, Yahoo and the Bank of Canada said on those dates — so the demonstration
    runs on a fresh clone with no price cache and no network, and `tests/test_ca_orders.py`
    can require that it reproduces `examples/ca_rotation_demo.csv` byte for byte. The capital
    is invented, and the file says so.
    """
    with open(path or DEMO_INPUT, encoding='utf-8') as fh:
        demo = json.load(fh)
    book = parse_config({'cash': demo['cash'], 'positions': [],
                         'execution': demo.get('execution', {})})
    results, rows = [], []
    for month in demo['months']:
        signal = Signal(month['weights'], month['signal_date'], month.get('canary', {}),
                        None, {'source': month['signal_source']})
        fx = FxRate(**month['fx'])
        result = build(book, signal, month['prices'], fx)
        results.append((signal, result))
        rows.extend(result['rotation_rows'])
        # carry the book forward: the orders as if filled at the prices they were sized at
        held = {p['ticker']: p['qty'] for p in book['positions']}
        for o in result['orders']:
            held[o['ticker']] = held.get(o['ticker'], 0) + (o['qty'] if o['side'] == 'BUY'
                                                            else -o['qty'])
        book = dict(book, cash=dict(result['residual_cash']),
                    positions=[{'ticker': t, 'qty': q, 'currency': currency_of(t)}
                               for t, q in sorted(held.items()) if q > 0])
    buf = io.StringIO()
    fields = ['signal_date', 'signal_ticker', 'exec_ticker', 'exchange', 'side', 'qty',
              'currency', 'price', 'value', 'reason']
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        out = {k: row.get(k) for k in fields}
        for k in ('price', 'value', 'qty'):
            if isinstance(out[k], float):
                out[k] = f'{out[k]:.4f}' if k == 'price' else f'{out[k]:.2f}'
        writer.writerow(out)
    return results, buf.getvalue()


# --------------------------------------------------------------------------------------- #

def main(argv=None, fetch_signal=compute_signal, fetch_quotes_fn=fetch_quotes, fx_resolver=None):
    parser = argparse.ArgumentParser(description='HAA_G12 orders in Canadian-listed funds. '
                                                 'Prints a plan; never places an order.')
    parser.add_argument('--config', default=None, help='default: ca_execution.json')
    parser.add_argument('--rate', type=float, default=None,
                        help='USD/CAD typed by hand; skips every FX lookup')
    parser.add_argument('--no-journal', action='store_true',
                        help='print only; write nothing under logs/')
    parser.add_argument('--demo', action='store_true',
                        help='offline demonstration on fictional capital (examples/)')
    parser.add_argument('--out', default=None,
                        help='with --demo: write the rotation CSV here')
    args = parser.parse_args(argv)

    if args.demo:
        results, text = run_demo()
        for signal, result in results:
            print('\n'.join(render(result, signal)))
            print()
        if args.out:
            with open(args.out, 'w', encoding='utf-8', newline='') as fh:
                fh.write(text)
            print(f'rotation CSV written to {args.out}')
        else:
            print(text)
        return 0

    try:
        config = load_config(args.config)
        signal = fetch_signal()
        required, optional = quote_tickers(signal.weights, config['positions'])
        today = datetime.date.today()
        quotes = fetch_quotes_fn((required | optional) - set(config['prices']), today)
        prices, problems, notes = check_quotes(required - set(config['prices']),
                                               optional - set(config['prices']), quotes,
                                               today)
        if problems:
            raise CAExecutionError('refusing to size orders without a usable price:\n  '
                                   + '\n  '.join(problems))
        if fx_resolver is None:
            from tools.fx_rate import resolve_usdcad as fx_resolver
        manual = args.rate if args.rate is not None else config['usdcad']
        fx = fx_resolver(manual=manual)
        result = build(config, signal, prices, fx)
    except (ConfigError, CAExecutionError, ValueError) as exc:
        print(f'HAA_G12_CA: {exc}')
        return 1

    print('\n'.join(render(result, signal, notes)))
    if not args.no_journal:
        paths = journal(result, signal, quotes, notes)
        print(f'\n  journal: {paths[0]}\n           {paths[1]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
