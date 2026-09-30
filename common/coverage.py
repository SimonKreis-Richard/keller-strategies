"""
Universe coverage: what a strategy needs, and from when it can honestly be measured.

The pre-audit engine had no notion of this. `DAA_G3_Leveraged_3X` backtested from a 2000
start produced **32 months with 100% of the portfolio on a ticker that did not exist**,
each reported as a +0.00% return, because `DataFrame.sum(axis=1)` skips NaN by default. The
strategy looked flat and safe through the dot-com bust it could not have traded.

Two distinct universes matter, and conflating them is how the RAA canary bug survived:

* **tradable** — everything the strategy can ever *hold*. This is what the ledger prices.
* **signal** — tradable plus the canaries. A canary is never traded, but a signal computed
  from a canary that has no data is not the strategy: it is whatever the NaN policy happens
  to do. BAA's BND canary starts 2007-04, so BAA before then is not BAA.

`earliest_valid_start` uses the **signal** universe for exactly that reason.
"""

import pandas as pd


def tradable_universe(strategy):
    """Every ticker `strategy` can ever hold, LETF images included."""
    s = strategy.sleeves()
    return set(s['offensive']) | set(s['defensive'])


def signal_universe(strategy):
    """Tradable universe plus canaries — everything the strategy needs data for."""
    return tradable_universe(strategy) | set(strategy.sleeves().get('canary') or [])


def inception_dates(strategy, store):
    """{ticker: first date with a price}. A ticker absent from the store maps to None."""
    return {t: store.first_tradable_date(t) for t in sorted(signal_universe(strategy))}


def binding_constraint(strategy, store):
    """(ticker, first_tradable_date) of the ticker that gates this strategy's history.

    Returns (None, None) if the strategy needs nothing the store lacks and every ticker is
    present from the start. Raises if a required ticker is missing from the store entirely —
    that is a configuration error, not a coverage limit, and must not be silently trimmed
    around.
    """
    dates = inception_dates(strategy, store)
    absent = [t for t, d in dates.items() if d is None]
    if absent:
        raise KeyError(
            f'{strategy.name}: the price store has no data at all for {absent}. Add the '
            f'ticker(s) to the download list or fix the strategy declaration — this is a '
            f'configuration error, not a short history, and trimming around it would '
            f'silently change what the strategy is.')
    if not dates:
        return None, None
    ticker = max(dates, key=lambda t: dates[t])
    return ticker, dates[ticker]


def earliest_valid_start(strategy, store, warmup_months=13):
    """First month-end at which every ticker the strategy needs has `warmup_months` of history.

    The warm-up is counted in COMPLETE months of the store's own calendar, so it is a real
    number of observations rather than a calendar approximation.
    """
    ticker, first = binding_constraint(strategy, store)
    month_ends = store.month_end_dates()
    if first is None or len(month_ends) == 0:
        return month_ends[0] if len(month_ends) else None
    after = month_ends[month_ends >= pd.Timestamp(first)]
    if len(after) <= warmup_months:
        return None      # not even a warm-up fits: the strategy is unmeasurable here
    return after[warmup_months]


def coverage_report(strategy, store, requested_start, warmup_months=13):
    """Everything the caller needs to trim honestly and say why.

    Returns ``{'earliest', 'binding_ticker', 'binding_inception', 'trimmed', 'message'}``.
    `earliest` is None when the strategy cannot be measured at all in this store.
    """
    ticker, first = binding_constraint(strategy, store)
    earliest = earliest_valid_start(strategy, store, warmup_months)
    requested = pd.to_datetime(requested_start)

    if earliest is None:
        return {'earliest': None, 'binding_ticker': ticker, 'binding_inception': first,
                'trimmed': True,
                'message': (f'{strategy.name}: unmeasurable — {ticker} only starts '
                            f'{pd.Timestamp(first).date()}, which leaves less than '
                            f'{warmup_months} months of warm-up in this store.')}

    trimmed = requested < earliest
    message = None
    if trimmed:
        message = (f'{strategy.name}: window trimmed to {earliest.date()} '
                   f'(requested {requested.date()}; binding: {ticker}, first tradable '
                   f'{pd.Timestamp(first).date()}, plus {warmup_months} months of warm-up)')
    return {'earliest': earliest, 'binding_ticker': ticker, 'binding_inception': first,
            'trimmed': trimmed, 'message': message}
