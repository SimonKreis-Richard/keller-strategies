"""
Re-measure every history donor against the ETF it stands in for, and against the yardstick.

`common/data_engine.HISTORY_BACKFILL` records, beside each donor, the measurement it was
admitted on. That record is a quotation, and quotations go stale. This tool re-derives it
from the vendors, so the admission can be checked by anyone rather than taken on trust:

    venv/Scripts/python.exe -m tools.proxy_fidelity            # every donor + the yardstick
    venv/Scripts/python.exe -m tools.proxy_fidelity TIP HYG    # named recipients only

THE YARDSTICK is the point. A donor is admissible when it sits no further from its ETF than a
SECOND REAL ETF of the same asset class sits from it. That bar is set by the market, not by
this repository, and the tool prints it in the same units as the donors so the two can be
read side by side. The units are the ones the strategies read:

  corr      correlation of monthly returns over the real overlap
  sign%     share of month-ends where 13612U has the same sign for both series — the
            quantity every absolute filter and canary in this repository thresholds at zero
  drift     annualised return of the donor minus the ETF's, over the overlap
  GFC       total return of each over 2007-11..2009-02 (`eras` segment `bear_gfc`), where the
            overlap reaches it — the crisis in which a smoothed or survivorship-biased donor
            would show itself

Lives in `tools/`, not `tests/`, for the reason `tools/vendor_crosscheck.py` does: it opens
sockets, and `unittest discover` collects every `test_*.py`. The arithmetic is `measure()`,
a pure function the suite tests offline.
"""
import argparse
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import data_engine as de                        # noqa: E402
from common.momentum import calc_13612u                     # noqa: E402

#: Pairs of two REAL funds of the same asset class. Their distance is the bar a donor must meet.
YARDSTICK = [('TIP', 'SCHP'), ('GLD', 'IAU'), ('HYG', 'JNK'), ('LQD', 'VCIT'), ('DBC', 'GSG'),
             ('TLT', 'SPTL'), ('EEM', 'VWO'), ('VNQ', 'IYR'), ('SPY', 'IVV')]

GFC = ('2007-11-01', '2009-02-28')


def measure(etf, donor):
    """Fidelity of `donor` to `etf`, both daily price series, over their common months.

    Returns a dict, or None when fewer than 36 common months exist — too short to say
    anything, and never to be read as agreement.
    """
    e = etf.dropna().resample('ME').last().iloc[1:]       # the first month is partial
    d = donor.dropna().resample('ME').last()
    idx = e.index.intersection(d.index)
    e, d = e[idx], d[idx]
    re_, rd = e.pct_change(fill_method=None).dropna(), d.pct_change(fill_method=None).dropna()
    if len(re_) < 36:
        return None
    years = len(re_) / 12.0
    drift = (1 + rd).prod() ** (1 / years) - (1 + re_).prod() ** (1 / years)
    se = calc_13612u(pd.DataFrame({'x': e}))['x'].dropna()
    sd = calc_13612u(pd.DataFrame({'x': d}))['x'].dropna()
    j = se.index.intersection(sd.index)

    def seg(r):
        w = r[(r.index >= GFC[0]) & (r.index <= GFC[1])]
        return float((1 + w).prod() - 1) if len(w) >= 12 else None

    return {'months': int(len(re_)), 'corr': float(re_.corr(rd)),
            'sign': float((np.sign(se[j]) == np.sign(sd[j])).mean()),
            'drift': float(drift), 'gfc_etf': seg(re_), 'gfc_donor': seg(rd),
            'from': str(re_.index[0].date())}


def _yahoo(symbols):
    import yfinance as yf
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = yf.download(list(symbols), period='max', progress=False, auto_adjust=False,
                          group_by='column')
    adj = raw['Adj Close']
    return {s: adj[s].dropna() for s in symbols if s in adj.columns}


def _series(symbols):
    out = _yahoo([s for s in symbols if s not in de.EXTERNAL_DONORS])
    for s in symbols:
        if s in de.EXTERNAL_DONORS:
            url, parser, _ = de.EXTERNAL_DONORS[s]
            out[s] = getattr(de, parser)(de._http_get(url))
    return out


def _row(label_a, label_b, m, evidence=''):
    if m is None:
        return f'{label_a:<6}{label_b:<14} insufficient overlap (< 36 months) — NOT agreement'
    g = (f"{m['gfc_etf']:+6.1%} / {m['gfc_donor']:+6.1%}"
         if m['gfc_etf'] is not None and m['gfc_donor'] is not None else '        n/a      ')
    line = (f"{label_a:<6}{label_b:<14}{m['from'][:7]:>8}{m['months']:>5}{m['corr']:>7.3f}"
            f"{m['sign']:>7.1%}{m['drift']:>+8.2%}   {g}")
    return line + (f'\n{"":<20}recorded: {evidence}' if evidence else '')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Re-measure the history donors.')
    parser.add_argument('recipients', nargs='*', help='limit to these recipients')
    args = parser.parse_args(argv)

    pairs = [(r, b.donor, b.evidence) for r, b in de.HISTORY_BACKFILL.items()
             if not args.recipients or r in args.recipients]
    needed = sorted({s for r, d, _ in pairs for s in (r, d)}
                    | ({s for p in YARDSTICK for s in p} if not args.recipients else set()))
    data = _series(needed)

    head = (f"{'ETF':<6}{'vs':<14}{'from':>8}{'mo':>5}{'corr':>7}{'sign%':>7}{'drift':>8}"
            '   GFC etf / other')
    if not args.recipients:
        print('YARDSTICK — two real funds of the same asset class')
        print(head)
        for a, b in YARDSTICK:
            if a in data and b in data:
                print(_row(a, b, measure(data[a], data[b])))
        print()
    print('DONORS — as admitted in common/data_engine.HISTORY_BACKFILL')
    print(head)
    missing = []
    for recipient, donor, evidence in pairs:
        if recipient not in data or donor not in data:
            missing.append(f'{recipient}<-{donor}')
            continue
        print(_row(recipient, donor, measure(data[recipient], data[donor]), evidence))
    if missing:
        print(f'\nNOT MEASURED (a vendor returned nothing): {", ".join(missing)}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
