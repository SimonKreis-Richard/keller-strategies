"""
Second-vendor cross-check: does the cached panel agree with somebody who is not Yahoo?

WHY THIS EXISTS
---------------
On 2026-09-01 the cache was found holding two dividend-adjustment vintages spliced at the
refresh-window edge. It had survived five rounds of QA, and the reason was structural: every
test in `tests/` reads frozen fixtures, and every audit reproduced the repository's numbers
from the repository's own cache. **Nothing had ever compared the price panel against anything
outside it**, so a data layer that was wrong in a self-consistent way was invisible by
construction. `PriceStore._verify_adjustment_vintage` closed one half of that (it checks the
panel against a property of the vendor's own data model, offline). This is the other half: an
actual second opinion.

WHAT IS COMPARED, AND WHY IT IS THE RAW CLOSE
---------------------------------------------
Stooq serves unadjusted OHLC. This repository stores `close` (raw) alongside `adj_close`
(total return). Raw against raw is the same object on both sides, so the comparison needs no
assumption about how either vendor handles distributions. Two statistics, deliberately
different in reach:

* MONTHLY RETURNS over the last `MONTHS` complete months. Catches a wrong price, a missing
  session, a mis-stamped bar -- anything that would move a momentum score now.
* A SPLIT SENTINEL over the whole overlap. The price ratio between two vendors is normalised
  by its own median, so a constant level offset (which carries no information about returns)
  is ignored, while an unapplied historical split shows up as a 2x/3x STEP. The monthly test
  cannot see this: a split eight years ago corrupts every long-run metric in `LEVERAGE.md`
  and never touches the last two years.

WHY IT IS A TOOL AND NOT A TEST
-------------------------------
`unittest discover -s tests` collects every `test_*.py` under `tests/`, and this project has
no pytest, no marker system and no skip convention. The only way to GUARANTEE the suite stays
network-free is to keep the network code out of the discovered directory. An environment-variable
skip-guard would also break the owner's constraint that this project needs no configuration --
and would itself be a guard that fails open, which is the shape that produced the incident.

`tests/test_vendor_crosscheck.py` drives everything here through an injected `fetch`, so the
classification logic is covered without a socket.

STATUS OF THE DEFAULT SOURCE -- READ THIS BEFORE TRUSTING A CLEAN RUN
--------------------------------------------------------------------
**Stooq is gated and this tool cannot currently reach it.** Probed 2026-09-01: a plain
request to the CSV endpoint returns HTTP 200 carrying an HTML page with a JavaScript
proof-of-work challenge, not a CSV. That is an access control, and this tool does not try
to defeat it: every ticker simply comes back `unavailable`, which is the honest answer and
the reason `unavailable` was never allowed to count as agreement.

So the comparison machinery below is finished and tested, and the SOURCE is not settled.
The fetcher is one injectable function, so pointing it at another provider is a small
change; what it costs is the constraint that made Stooq attractive in the first place, since
the remaining no-account options for adjusted US ETF history are thin. `--provider` exists
for that. Until a working source is wired, running this prints a table of `unavailable` and
exits 0 -- deliberately not 1, because "I could not ask" is not "they disagreed".

USAGE
-----
    venv/Scripts/python.exe tools/vendor_crosscheck.py
    venv/Scripts/python.exe tools/vendor_crosscheck.py --tickers SPY TIP IEF

Exit code is 1 when any ticker alarms, so a scheduler can act without parsing the output.
"""
import argparse
import datetime as _dt
import json
import os
import sys
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_engine import SETTLED_DAYS                      # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Anonymous public CSV. No key, no account, no header auth.
STOOQ_URL = 'https://stooq.com/q/d/l/?s={symbol}.us&i=d'

#: Stooq answers a rate limit with HTTP 200 and a plain-text body, which `pd.read_csv` would
#: happily turn into a garbage frame -- a silent slide into "0 disagreements found". The
#: header is therefore asserted literally.
STOOQ_HEADER = 'Date,Open,High,Low,Close,Volume'

#: A month whose price return differs by more than this is a real disagreement. Two vendors
#: pricing the same session should agree to the cent; 25bp is loose enough to absorb a
#: different closing-auction snapshot and tight enough to catch anything that moves a signal.
MONTHLY_TOLERANCE = 0.0025

#: Spread of the median-normalised level ratio above which an unapplied split is suspected.
SPLIT_RATIO_SPREAD = 1.10

#: Complete months compared. Twelve would cover the longest momentum lookback; Stooq returns
#: the whole history in one request either way, so twenty-four is free and gives a second
#: independent draw of every distribution cycle.
MONTHS = 24

#: Below this many overlapping months the answer is "I could not tell", never "agrees".
MIN_OVERLAP_MONTHS = 6

#: Not a fund: an index yield used to construct pre-2007 cash, with no `.us` symbol. It is
#: already removed from the store by `_build_synthetic_cash`.
SKIP_TICKERS = {'^IRX'}

ALARM_STATUSES = ('disagrees', 'suspected_split')


def fetch_stooq_csv(symbol, timeout=15):
    """Return Stooq's CSV text for `symbol`, or raise. The ONLY function here that networks.

    Every detail is a trap this repository has already paid for once: an explicit timeout
    (a hung fetch must not wedge a scheduled job), an explicit User-Agent (Stooq returns an
    empty body to some defaults), and an explicit utf-8 decode with `errors='replace'` --
    the platform codec is what made `manifest.git_state` report a modified worktree as
    clean on Windows.
    """
    url = STOOQ_URL.format(symbol=symbol.lower())
    request = urllib.request.Request(url)
    request.add_header('User-Agent', 'keller-strategies vendor cross-check')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode('utf-8', errors='replace')
    # Since 2026-09-01 Stooq answers with HTTP 200 and a JavaScript proof-of-work page
    # instead of the CSV. Named explicitly rather than left to the header assertion, so the
    # report says "the vendor is refusing automated access" and not "the body looked odd" --
    # they call for entirely different responses, and only one of them is a data problem.
    lowered = body.lstrip()[:200].lower()
    if lowered.startswith('<!doctype html') or lowered.startswith('<html'):
        raise ValueError('the endpoint served an HTML challenge page, not CSV: Stooq is '
                         'gating automated access. This tool does not attempt to defeat '
                         'that; supply a different provider instead.')
    return body


#: Injectable sources. One entry today; the point of the table is that adding a second is a
#: line here plus a fetch function, not a rewrite of anything below.
PROVIDERS = {'stooq': fetch_stooq_csv}


def parse_stooq_csv(text):
    """CSV text -> a close Series indexed by date. Raises ValueError on anything else."""
    if not text:
        raise ValueError('empty body')
    first = text.lstrip().splitlines()[0].strip()
    if first != STOOQ_HEADER:
        raise ValueError('unexpected body, first line was: {!r}'.format(first[:80]))
    frame = pd.read_csv(pd.io.common.StringIO(text))
    if 'Date' not in frame.columns or 'Close' not in frame.columns:
        raise ValueError('CSV parsed but carries no Date/Close')
    series = pd.Series(pd.to_numeric(frame['Close'], errors='coerce').values,
                       index=pd.to_datetime(frame['Date'], errors='coerce'))
    series = series[series.index.notna()].dropna().sort_index()
    if series.empty:
        raise ValueError('CSV parsed but held no usable rows')
    return series


def _month_ends(series):
    settled = series[series.index <= series.index[-1] - pd.Timedelta(days=SETTLED_DAYS)]
    if settled.empty:
        return settled
    grouped = settled.groupby([settled.index.year, settled.index.month]).tail(1)
    return grouped[~grouped.index.duplicated()]


def compare_one(ours, theirs, months=MONTHS):
    """Classify one ticker. `ours` and `theirs` are raw close Series.

    Returns a dict carrying the verdict AND the evidence, because a bare status trains a
    reader to stop looking.
    """
    result = {'status': 'unavailable', 'n_overlap_months': 0, 'worst_month': None,
              'worst_diff': None, 'level_ratio_spread': None, 'reason': None}
    if ours is None or theirs is None or ours.empty or theirs.empty:
        result['reason'] = 'no data on one side'
        return result

    shared_days = ours.index.intersection(theirs.index)
    if len(shared_days) >= 30:
        ratio = (ours.reindex(shared_days) / theirs.reindex(shared_days)).dropna()
        ratio = ratio[ratio > 0]
        if len(ratio) >= 30:
            normalised = ratio / ratio.median()
            spread = float(normalised.max() / normalised.min())
            result['level_ratio_spread'] = spread
            if spread > SPLIT_RATIO_SPREAD:
                # Named apart from `disagrees` so the reader is not sent hunting a price bug
                # when the answer is a corporate action.
                result['status'] = 'suspected_split'
                result['reason'] = ('the level ratio between the two vendors spans '
                                    '{:.2f}x, which is a step, not a price disagreement'
                                    .format(spread))
                return result

    our_me, their_me = _month_ends(ours), _month_ends(theirs)
    shared = our_me.index.intersection(their_me.index)
    if len(shared) < MIN_OVERLAP_MONTHS + 1:
        result['n_overlap_months'] = max(len(shared) - 1, 0)
        result['status'] = 'insufficient_overlap'
        result['reason'] = 'only {} comparable month(s)'.format(result['n_overlap_months'])
        return result

    shared = shared[-(months + 1):]
    ours_r = our_me.reindex(shared).pct_change(fill_method=None).dropna()
    theirs_r = their_me.reindex(shared).pct_change(fill_method=None).dropna()
    diff = (ours_r - theirs_r).abs().dropna()
    result['n_overlap_months'] = int(len(diff))
    if diff.empty:
        result['status'] = 'insufficient_overlap'
        result['reason'] = 'no comparable monthly returns'
        return result

    result['worst_month'] = '{:%Y-%m}'.format(diff.idxmax())
    result['worst_diff'] = float(diff.max())
    result['status'] = 'disagrees' if diff.max() > MONTHLY_TOLERANCE else 'agrees'
    if result['status'] == 'disagrees':
        result['reason'] = ('{} differs by {:.2%}, past the {:.2%} tolerance'
                            .format(result['worst_month'], diff.max(), MONTHLY_TOLERANCE))
    return result


def crosscheck(store, fetch=None, tickers=None, months=MONTHS):
    """Compare every ticker's raw close against Stooq. `fetch` is injected so the tests
    exercise this whole path -- classification included -- without a socket.

    The default is resolved HERE and not in the signature, and that is not a style choice.
    A default argument binds the function object at def time, so `patch.object(vc,
    'fetch_stooq_csv', ...)` would leave it pointing at the real one: a test written that
    way believes it is offline while opening a socket. That happened while writing this
    file, and Stooq answered with a page of HTML.
    """
    fetch = fetch or fetch_stooq_csv
    ours_all = store._frames['close']
    names = [t for t in (tickers or list(ours_all.columns)) if t not in SKIP_TICKERS]
    out = {}
    for name in sorted(names):
        if name not in ours_all.columns:
            out[name] = {'status': 'unavailable', 'reason': 'not in the store',
                         'n_overlap_months': 0, 'worst_month': None, 'worst_diff': None,
                         'level_ratio_spread': None}
            continue
        try:
            theirs = parse_stooq_csv(fetch(name))
        except Exception as exc:                     # noqa: BLE001 - every failure is a status
            out[name] = {'status': 'unavailable', 'reason': '{}: {}'.format(
                type(exc).__name__, exc), 'n_overlap_months': 0, 'worst_month': None,
                'worst_diff': None, 'level_ratio_spread': None}
            continue
        out[name] = compare_one(ours_all[name].dropna(), theirs, months=months)
    return out


def report_lines(results):
    lines = ['', 'VENDOR CROSS-CHECK vs stooq.com (raw closes, {} months)'.format(MONTHS), '']
    head = '{:<8} {:<20} {:>7} {:>10} {:>9}  {}'.format(
        'ticker', 'status', 'months', 'worst', 'lvl.spread', 'note')
    lines += [head, '-' * len(head)]
    for name in sorted(results):
        r = results[name]
        worst = '{:.2%}'.format(r['worst_diff']) if r.get('worst_diff') is not None else '-'
        spread = ('{:.3f}'.format(r['level_ratio_spread'])
                  if r.get('level_ratio_spread') is not None else '-')
        lines.append('{:<8} {:<20} {:>7} {:>10} {:>9}  {}'.format(
            name, r['status'], r.get('n_overlap_months', 0), worst, spread,
            (r.get('reason') or '')[:60]))

    counts = {}
    for r in results.values():
        counts[r['status']] = counts.get(r['status'], 0) + 1
    checked = counts.get('agrees', 0) + sum(counts.get(s, 0) for s in ALARM_STATUSES)
    lines += [
        '',
        '  {} of {} tickers were actually COMPARED; {}'.format(
            checked, len(results),
            ', '.join('{} {}'.format(v, k) for k, v in sorted(counts.items()))),
        '  "0 disagreements" means nothing without that first number: an unreachable ticker',
        '  is `unavailable`, never agreement.',
    ]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--tickers', nargs='*', default=None)
    parser.add_argument('--provider', default='stooq', choices=sorted(PROVIDERS),
                        help='where the second opinion comes from')
    parser.add_argument('--months', type=int, default=MONTHS)
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args(argv)

    from tools.backtest_driver import build_config
    import main as engine

    cfg = build_config(CACHE_REFRESH_HOURS=24 * 365)
    _prices, _sw, _su, store = engine.load_data(cfg)

    results = crosscheck(store, fetch=PROVIDERS[args.provider], tickers=args.tickers,
                         months=args.months)
    print('\n'.join(report_lines(results)))

    if not args.no_save:
        out_dir = os.path.join(ROOT_DIR, 'backtest_results')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'vendor_crosscheck_{}.json'.format(
            _dt.date.today().isoformat()))
        with open(path, 'w', encoding='utf-8') as fh:
            # `provenance()` verbatim, so the verdict is tied to the exact cache it judged.
            # Without it a later refresh silently invalidates the report and nobody can tell.
            json.dump({'generated_at': _dt.datetime.now().isoformat(timespec='seconds'),
                       'tolerance': MONTHLY_TOLERANCE, 'months': args.months,
                       'data': store.provenance(), 'results': results},
                      fh, indent=2, sort_keys=True, default=str)
        print('\nwritten: {}'.format(path))

    alarms = [n for n, r in results.items() if r['status'] in ALARM_STATUSES]
    if alarms:
        print('\nALARM on: {}'.format(', '.join(sorted(alarms))))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
