"""
Where the USD/CAD rate comes from, and why it is not in `common/`.

`common/ca_execution.py` takes an `FxRate` as an argument and never looks one up. This module
is the other half: the only place in the Canadian execution path that opens a socket. The
boundary is the one `tools/vendor_crosscheck.py` already draws — network code stays out of
the code that decides, so the decision is testable without a network and cannot fail because
a website is down.

`unittest discover -s tests` collects every `test_*.py` under `tests/`, and this project has
no pytest and no marker system, so keeping the fetchers out of that directory is what
GUARANTEES the suite stays offline. `tests/test_fx_rate.py` drives everything here through an
injected `opener` and replaces the real one with a function that raises: a refactor that
reintroduced a live fetch would turn the suite red rather than slow.

THREE SOURCES, IN ORDER, AND NO FOURTH
--------------------------------------
1. A MANUAL override. Always wins. A rate you typed is a rate you can defend.
2. The BANK OF CANADA's Valet API, series `FXUSDCAD`. Free, no key, no account, so
   `SETUP.md`'s promise that this project needs no credentials survives.

   WHAT THAT SERIES ACTUALLY IS, checked against the endpoint on 2026-09-18 rather than
   assumed: the API describes it as *"Daily average exchange rate: daily value of the US
   dollar expressed in Canadian dollars, for 1 unit of US dollar"*. It is a daily indicative
   rate published once, NOT a closing print and NOT the rate any broker will fill at. The
   brief called it "the previous day's closing rate" and this file said the same until the
   first real call showed otherwise.

   That matters in one direction only, and it is the harmless one: the rate here values the
   book and sizes a conversion, and the actual fill comes back from the broker. An order
   sized on an indicative rate can be a fraction of a percent off; an order sized on an
   invented one is off by whatever the invention was. What must never happen is the second,
   which is why there is no default below.
3. `CAD=X` via yfinance, which is already a dependency.

If all three fail, `resolve_usdcad` RAISES. There is no fallback constant and there is no
last-known-good cache. A stale or invented exchange rate misprices every line in the book by
the same factor, in the same direction, and shows no symptom at all — the same shape as the
adjustment-vintage splice found on 2026-09-01, and the reason that one survived five rounds
of QA.

PARSING IS STRICT ON PURPOSE
----------------------------
`parse_valet_json` requires the exact structure and refuses anything else. An API under load
or behind a gateway answers with HTTP 200 carrying HTML or a plain-text notice, and a lenient
parser turns that into "no observations" and then into a default. Every rejection here is a
body that a tolerant reader would have turned into a number.
"""
import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.ca_execution import CAExecutionError, FxRate  # noqa: E402

VALET_URL = 'https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?recent=1'
SERIES = 'FXUSDCAD'
TIMEOUT = 15
UA = 'keller-strategies/1.0 (personal research; contact via the repository)'

#: A rate outside this band is refused before it can reach an order. USD/CAD has traded
#: roughly 0.91-1.62 since the dollar floated; this is wider than that on both sides. It is a
#: SANITY bound, not a forecast: its job is to catch a parse that picked up an index level, a
#: percentage or a date, not to have an opinion about the currency.
PLAUSIBLE_RANGE = (0.50, 3.00)


def _default_opener(url):
    """The real fetch. Isolated so tests can replace it with something that raises."""
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        # errors='replace' rather than a strict decode: a mojibake byte in a message must not
        # raise where the caller would read the exception as "the service is down". This is
        # the cp1252 trap that once made a dirty working tree report as clean.
        return resp.read().decode('utf-8', errors='replace')


def _check_plausible(rate, where):
    lo, hi = PLAUSIBLE_RANGE
    if not (lo <= rate <= hi):
        raise CAExecutionError(
            f'{where} returned a USD/CAD rate of {rate!r}, outside the plausible band '
            f'{lo}-{hi}. That is a parse that found the wrong field, not a currency move.')
    return rate


def parse_valet_json(body):
    """`(rate, date)` from a Bank of Canada Valet response, or raise.

    Refuses anything that is not the documented shape. Everything this rejects is a body a
    lenient parser would have turned into a number or into a silent zero-observation result.
    """
    try:
        payload = json.loads(body)
    except (ValueError, TypeError) as exc:
        raise CAExecutionError(
            f'the Bank of Canada endpoint did not return JSON ({exc}). An API behind a '
            f'gateway answers HTTP 200 with HTML, and a tolerant parser turns that into a '
            f'default rate.') from exc
    if not isinstance(payload, dict):
        raise CAExecutionError('the Valet response was not a JSON object.')
    observations = payload.get('observations')
    if not isinstance(observations, list) or not observations:
        raise CAExecutionError(
            'the Valet response carried no observations. "No data" is not a rate, and must '
            'not become one.')
    latest = observations[-1]
    if not isinstance(latest, dict) or SERIES not in latest:
        raise CAExecutionError(f'the Valet observation has no {SERIES} field: {latest!r}')
    cell = latest[SERIES]
    value = cell.get('v') if isinstance(cell, dict) else cell
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise CAExecutionError(
            f'the {SERIES} value {value!r} is not a number.') from exc
    date = latest.get('d')
    if not date:
        raise CAExecutionError('the Valet observation carries no date; a rate without a date '
                               'cannot be journalled against an order.')
    return _check_plausible(rate, 'the Bank of Canada'), str(date)


def fetch_boc_usdcad(opener=_default_opener):
    """The Bank of Canada's published daily USD/CAD indicative rate, as an `FxRate`.

    CAD per USD, matching `FxRate`'s convention: 1 USD buys `usdcad` CAD.
    """
    rate, date = parse_valet_json(opener(VALET_URL))
    return FxRate(usdcad=rate, source='Bank of Canada Valet FXUSDCAD', as_of=date)


def fetch_yahoo_usdcad(download=None):
    """Fallback: `CAD=X` via yfinance, which is already a dependency.

    Second rather than first because Yahoo publishes a live quote of unstated provenance,
    while the Bank of Canada publishes a dated official series. Neither is a fill. The
    journal records which one was used, so a reader can tell them apart after the fact.
    """
    if download is None:
        import yfinance as yf
        download = yf.download
    frame = download('CAD=X', period='5d', interval='1d', progress=False, auto_adjust=False)
    if frame is None or len(frame) == 0:
        raise CAExecutionError('yfinance returned no rows for CAD=X.')
    closes = frame['Close']
    if hasattr(closes, 'columns'):
        closes = closes.iloc[:, 0]
    closes = closes.dropna()
    if closes.empty:
        raise CAExecutionError('yfinance returned only empty closes for CAD=X.')
    rate = float(closes.iloc[-1])
    return FxRate(usdcad=_check_plausible(rate, 'yfinance CAD=X'),
                  source='yfinance CAD=X (fallback)', as_of=str(closes.index[-1].date()))


def resolve_usdcad(manual=None, boc=fetch_boc_usdcad, yahoo=fetch_yahoo_usdcad):
    """Manual override, then the Bank of Canada, then Yahoo. Raises if all three fail.

    The failure list is carried into the exception rather than swallowed: "I could not get a
    rate" has to say what it tried, or the next reader retries the same three things by hand.
    """
    if manual is not None:
        return FxRate(usdcad=_check_plausible(float(manual), 'the manual override'),
                      source='manual override', as_of='supplied by the operator')
    failures = []
    for name, fetch in (('Bank of Canada', boc), ('yfinance CAD=X', yahoo)):
        try:
            return fetch()
        except Exception as exc:                                        # noqa: BLE001
            failures.append(f'{name}: {type(exc).__name__}: {exc}')
    raise CAExecutionError(
        'no USD/CAD rate could be obtained, and there is deliberately no default — an '
        'assumed rate misprices every line in the book by the same factor with no symptom. '
        'Pass one by hand. Tried:\n  ' + '\n  '.join(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Print the USD/CAD rate this project would '
                                                 'use, and where it came from.')
    parser.add_argument('--rate', type=float, default=None,
                        help='Manual override; skips every lookup.')
    args = parser.parse_args(argv)
    try:
        fx = resolve_usdcad(manual=args.rate)
    except CAExecutionError as exc:
        print(exc)
        return 1
    print(f'USD/CAD {fx.usdcad:.4f}   {fx.source}   as of {fx.as_of}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
