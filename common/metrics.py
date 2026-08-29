"""
Portfolio metrics.

Three things changed here on 2026-07-28, all of them audit findings:

1. **Drawdown is measured from initial wealth.** The old code built the wealth curve as
   `(1 + returns).cumprod()`, whose first element is already `1 + r0`. A drawdown that
   begins in month one was therefore invisible: `[-50%, 0%, +10%]` reported MaxDD = 0.00%.
2. **The risk-free rate is real.** `calculate_metrics` always accepted `rf` and was never
   passed one, so every Sharpe and Sortino in the repo divided an unadjusted return by
   volatility. On HAA_G12 over 2015-2025 that inflated Sharpe from 1.14 to 1.39.
3. **The return type is a dict.** It used to be a 6-tuple; adding a seventh value to a
   positional tuple unpacked at every call site is a silent-corruption bug waiting to
   happen, and UPI made a seventh value necessary.
"""

import numpy as np
import pandas as pd

#: Annualised cash rate assumed when NOTHING else is available for the configured cash
#: ticker — neither its own traded history nor a constructed span. 3.0% is roughly the
#: average 3-month T-bill yield over 2000-2007. It is an assumption, not an observation.
#:
#: For BIL this is now a dead end rather than a fallback: `PriceStore._extend_history`
#: accrues a synthetic bill from ^IRX back to the download start, so there is no month left
#: with no data at all. It still applies to a `CASH_TICKER` with no SYNTHETIC_CASH entry,
#: which is why it stays. `build_rf_series` counts and names all three provenances.
DEFAULT_RF_ANNUAL_PRE_BIL = 0.03


def _as_monthly_rf(rf, index):
    """Normalise `rf` to a monthly-return Series aligned to `index`.

    Accepts a scalar ANNUAL rate or a Series of realised monthly cash returns.
    """
    if isinstance(rf, pd.Series):
        return rf.reindex(index).fillna(0.0)
    return pd.Series(float(rf) / 12.0, index=index)


def annualised_rf(rf_monthly):
    """Geometric annualisation of a realised monthly cash series."""
    if len(rf_monthly) == 0:
        return 0.0
    total = float((1.0 + rf_monthly).prod())
    years = len(rf_monthly) / 12.0
    return total ** (1.0 / years) - 1.0 if years > 0 else 0.0


def wealth_curve(returns, start_label=None):
    """Cumulative wealth INCLUDING the initial 1.0.

    The leading 1.0 is what makes a first-month drawdown visible. `start_label` is the date
    the money was deployed (the first execution date); without one, a synthetic label one
    month before the first return is used so the series still plots sensibly.
    """
    if returns.empty:
        return pd.Series(dtype=float)
    if start_label is None:
        start_label = returns.index[0] - pd.offsets.MonthEnd(1)
    curve = (1.0 + returns).cumprod()
    return pd.concat([pd.Series([1.0], index=[pd.Timestamp(start_label)]), curve])


def drawdown_series(cum_ret):
    """Drawdown from the running peak, <= 0 everywhere. `cum_ret` must already carry the
    initial 1.0 — see `wealth_curve`."""
    return cum_ret / cum_ret.cummax() - 1.0


def max_drawdown_months(returns, start_label=None):
    """Length in periods of the deepest drawdown, measured PEAK TO TROUGH.

    Peak-to-trough rather than peak-to-recovery, and the distinction is not stylistic. The one
    consumer is `margin_sizing`, where this figure sets how much interest capitalises onto the
    debit balance before the broker's maintenance test can fire. That test fires at the trough:
    interest keeps accruing through the recovery, but by then it no longer threatens anything.
    Using peak-to-recovery would overstate `c` — sometimes by years — and understate the
    survivable leverage, which is a conservative error in a place where the module already has
    an explicit safety factor to carry that job.

    Returns 0 for a series that never draws down.
    """
    if returns is None or len(returns) == 0:
        return 0
    curve = wealth_curve(returns, start_label)
    dd = drawdown_series(curve)
    if float(dd.min()) >= 0.0:
        return 0
    trough = int(np.argmin(dd.to_numpy()))
    # The peak the trough is measured from is the last new high at or before it.
    running = curve.iloc[:trough + 1]
    peak = int(np.flatnonzero((running == running.cummax()).to_numpy()).max())
    return int(trough - peak)


def ulcer_index(cum_ret):
    """Root-mean-square drawdown (Martin's Ulcer Index).

    Unlike max drawdown, this penalises *depth and duration together*: a portfolio that
    spends years underwater scores badly even if it never prints a spectacular trough.

    The RMS runs over ALL observations, including the zeros at new highs — that is the
    standard definition and the one Keller's papers use. `cum_ret` MUST already include
    initial wealth 1.0, or the index inherits the same understatement max drawdown had.
    """
    if cum_ret is None or len(cum_ret) == 0:
        return float('nan')
    dd = drawdown_series(cum_ret)
    return float(np.sqrt((dd ** 2).mean()))


def ulcer_performance_index(cagr, rf_annual, ui):
    """UPI = (CAGR - rf) / Ulcer Index. Keller's 'Martin ratio', and the metric his papers
    lead every results table with.

    Returns NaN rather than infinity for a never-underwater series: a UI of zero means the
    ratio is undefined, not perfect.
    """
    if ui is None or not np.isfinite(ui) or ui <= 0:
        return float('nan')
    return float((cagr - rf_annual) / ui)


#: The metrics a table may be ranked by, in display order. Defined here rather than in
#: `main.py` because three places rank on them — the CLI report, the dashboard's main table,
#: and the per-segment leaderboards — and a fourth copy of the list is a fourth chance for the
#: orderings to disagree. `main.RANK_KEYS` re-exports this name.
RANK_KEYS = ('cagr', 'sharpe', 'sortino', 'upi', 'vol', 'max_dd')

#: Metrics where SMALLER is better. Note `max_dd` is NOT one of them: it is reported as a
#: negative number, so the largest value is the shallowest drawdown and descending is correct.
#: Volatility is the only one reported as a positive quantity you want less of.
LOWER_IS_BETTER = frozenset({'vol'})

#: The subset of `RANK_KEYS` whose value depends on ANNUALISING a sample — i.e. the ones that
#: stop meaning anything over a short window. `max_dd` is deliberately excluded: a drawdown is
#: read straight off the wealth path and is exactly as valid over two months as over two
#: decades. Keeping it out of this tuple also keeps ONE source for it in a segment cell
#: (`eras._stats`), rather than having `_annualised` overwrite it on long segments only.
ANNUALISED_KEYS = tuple(k for k in RANK_KEYS if k != 'max_dd')

#: Annualised metrics need a meaningful number of observations before the annualisation says
#: more about the strategy than about the arithmetic. Twelve months is the threshold, and it is
#: a judgement rather than a derivation: below a year, a CAGR is a two-month return raised to
#: the sixth power, a Sharpe is estimated from four observations, and a UPI over a span with no
#: new high is undefined. Segments shorter than this report total return and drawdown — which
#: need no annualisation — and `n/a` for the rest. `common/eras.py` enforces it.
SEGMENT_MIN_MONTHS = 12


def rank_descending(key):
    """True when a larger value of `key` is better. The single source of that decision."""
    return key not in LOWER_IS_BETTER


def calculate_metrics(returns, rf=0.0, start_label=None):
    """Key portfolio metrics for a series of periodic (monthly) returns.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns with a datetime index.
    rf : float or pd.Series
        Annualised risk-free rate, or a Series of realised monthly cash returns. Passing a
        realised series is strongly preferred: a scalar assumes a constant cash rate across
        a window that spans 0% and 5% policy regimes, which flatters every ratio measured
        through the zero-rate years and penalises the rest.
    start_label : Timestamp, optional
        Date the capital was deployed, used as the wealth curve's first index entry.

    Returns
    -------
    dict with keys: cagr, max_dd, max_dd_months, sharpe, sortino, vol, ulcer_index, upi,
    cum_ret, rf_annual, n_periods, first_date, last_date.

    Formulas
    --------
    - CAGR       = wealth_final ** (12 / n_months) - 1
    - MaxDD      = min(wealth / cummax(wealth) - 1), wealth starting at 1.0
    - Volatility = std(monthly) * sqrt(12)
    - Sharpe     = mean(excess) * 12 / volatility
    - Sortino    = mean(excess) * 12 / downside deviation, where downside deviation is
                   sqrt(mean(min(0, excess)^2)) * sqrt(12) over the WHOLE series — not
                   `returns[returns < 0].std()`, which would use the wrong degrees of
                   freedom and deviate around a sub-period mean.
    - UI / UPI   = see `ulcer_index` / `ulcer_performance_index`.
    """
    # NaN, not 0.0: "no returns" is not "a strategy that returned 0% with no drawdown".
    # Zero-filled ratios here were the exact failure mode this module was rewritten to
    # remove elsewhere ("a missing month is not a 0% return") surviving in its own empty
    # case. Not reachable from run_backtest today — a strategy with no returns is routed
    # to `failures` — but a direct caller comparing metrics would have ranked an empty
    # series as a flat, riskless one. n_periods: 0 stays a number; it is a count, and zero
    # is its true value.
    empty = {
        'cagr': float('nan'), 'max_dd': float('nan'), 'sharpe': float('nan'),
        'sortino': float('nan'), 'vol': float('nan'),
        'ulcer_index': float('nan'), 'upi': float('nan'),
        'cum_ret': pd.Series(dtype=float), 'rf_annual': float('nan'), 'n_periods': 0,
        'first_date': None, 'last_date': None, 'max_dd_months': 0,
    }
    if returns is None or returns.empty:
        return empty

    cum_ret = wealth_curve(returns, start_label)
    years = len(returns) / 12.0
    cagr = (cum_ret.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0.0

    max_dd = float(drawdown_series(cum_ret).min())

    rf_monthly = _as_monthly_rf(rf, returns.index)
    rf_annual = annualised_rf(rf_monthly)
    excess = returns - rf_monthly

    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (excess.mean() * 12) / ann_vol if ann_vol > 0 else 0.0

    downside_vol = np.sqrt(np.mean(np.minimum(0.0, excess) ** 2)) * np.sqrt(12)
    sortino = (excess.mean() * 12) / downside_vol if downside_vol > 0 else 0.0

    ui = ulcer_index(cum_ret)

    return {
        'cagr': float(cagr),
        'max_dd': max_dd,
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'vol': float(ann_vol),
        'ulcer_index': ui,
        'upi': ulcer_performance_index(float(cagr), rf_annual, ui),
        'cum_ret': cum_ret,
        'rf_annual': float(rf_annual),
        'n_periods': int(len(returns)),
        # Peak-to-trough duration of that same drawdown. Consumed by `margin_sizing` to size
        # the interest that capitalises onto a debit balance before the broker's maintenance
        # test can fire.
        'max_dd_months': max_drawdown_months(returns, start_label),
        'first_date': returns.index[0],
        'last_date': returns.index[-1],
    }


def build_rf_series(store, dates, cash_ticker='BIL',
                    pre_inception_annual=DEFAULT_RF_ANNUAL_PRE_BIL):
    """Monthly cash returns at `dates`, from the cash ticker's own price history.

    Returns (series, description). The description separates THREE provenances, because
    they are three different claims and a report that blends them is not reporting a
    risk-free rate at all:

    * **realised** — the fund's own traded history;
    * **constructed** — a synthetic span built by `PriceStore._extend_history`, e.g. BIL
      before 2007-05-30 accrued from ^IRX. A price no market produced;
    * **assumed** — a flat `pre_inception_annual`, used only where there is neither.

    The distinction was silently lost when the history extension landed. `_extend_history`
    deliberately makes a constructed span indistinguishable from real history *for*
    `first_tradable_date`, because the coverage guard needs exactly that. This function used
    the same call as its honesty test and counted assumed months as `realised.isna().sum()`,
    so once BIL had no NaNs left it reported "realised BIL total return from 2000-01-03" —
    88 of 318 monthly observations constructed, and the word for them was "realised". Every
    Sharpe, Sortino and UPI for a strategy opening before 2007-05 was netted against it.

    `constructed_before` is the fix: it is the one call that still tells the two apart.
    """
    dates = pd.DatetimeIndex(dates)
    if cash_ticker is None or cash_ticker not in store.adj_close().columns:
        return (pd.Series(pre_inception_annual / 12.0, index=dates),
                f'assumed flat {pre_inception_annual:.2%}/yr (no {cash_ticker} history)')

    px = store.adj_close()[cash_ticker].reindex(dates)
    realised = px.pct_change(fill_method=None)
    fallback = pd.Series(pre_inception_annual / 12.0, index=dates)
    series = realised.where(realised.notna(), fallback)

    n_assumed = int(realised.isna().sum())
    real_from = store.constructed_before(cash_ticker)
    rec = getattr(store, 'constructed', {}).get(cash_ticker) or {}
    # A month is constructed when it has a value AND that value predates the fund itself.
    n_constructed = (int((realised.notna() & (dates < real_from)).sum())
                     if real_from is not None else 0)
    n_realised = int(realised.notna().sum()) - n_constructed

    if n_constructed == 0 and n_assumed == 0:
        return series, f'realised {cash_ticker} total return'

    parts = []
    if n_realised:
        first_real = pd.Timestamp(real_from) if real_from is not None \
            else store.first_tradable_date(cash_ticker)
        parts.append(f'realised {cash_ticker} from '
                     f'{pd.Timestamp(first_real).date() if first_real is not None else "n/a"}')
    if n_constructed:
        parts.append(f'{n_constructed} earlier month(s) CONSTRUCTED from '
                     f'{rec.get("source", "?")}')
    if n_assumed:
        parts.append(f'{n_assumed} month(s) assumed flat at {pre_inception_annual:.2%}/yr')
    return series, '; '.join(parts)
