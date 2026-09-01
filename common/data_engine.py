"""
Price data layer.

The engine used to cache a MONTHLY panel built with ``resample('ME').last()``. That one
call was responsible for three separate audit findings: it emitted a month before the
month had finished, it labelled every row with a synthetic calendar month-end
(``2023-12-31``) that is not a trading date, and it left no way to price an execution at
anything other than the close that produced the signal.

`PriceStore` replaces it. It caches **daily** bars, keeps real trading dates, refuses to
emit a month until that month is over, and carries opens so a fill can be priced at the
session *after* the decision. The monthly panel the strategies consume is derived from it
(`monthly_adj_close`), so strategy code is unaffected by the change.

Yahoo Finance is the single external source THE ENGINE READS. Since 2026-09-01
`tools/vendor_crosscheck.py` reads a second vendor for verification only; no engine
path imports it, and it lives outside `tests/` so the suite stays network-free. `adj_close` is Yahoo's back-adjusted close;
`close` is the raw close. Their ratio is the adjustment factor, which is what puts the
*open* on the same adjusted scale as the closes:

    adj_open = open * (adj_close / close)

Without that, an execution priced at the open would mix an unadjusted price into an
adjusted series and manufacture a return at every dividend and split.
"""

import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd
import yfinance as yf

# Compute repository root directory (parent of 'common')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Longest run of missing daily observations that may be carried forward. Beyond this the
#: price is not "stale", it is absent, and carrying it invents a flat return over a period
#: the portfolio may well have been moving.
MAX_STALE_DAYS = 5

#: Calendar days of history re-downloaded and OVERWRITTEN on every incremental refresh.
#: The old code appended only strictly-newer rows, so a row written while its month was
#: still open stayed wrong forever. Overwriting a trailing window is what makes the cache
#: self-healing.
REFRESH_WINDOW_DAYS = 90

#: Calendar days of history compared against Yahoo on every refresh, to notice that a
#: distribution has restated the adjusted series. It must reach well OUTSIDE
#: `REFRESH_WINDOW_DAYS`: the bars inside that window are rewritten on every refresh and are
#: therefore always current, so comparing only those compares the cache against itself and
#: sees nothing. The seam this detects sits at the window edge, and the stale side of it is
#: everything older. 400 days also spans the longest momentum lookback (SMA13 needs 13
#: months), so a restatement is caught before it can reach a signal.
DETECT_WINDOW_DAYS = 400

#: Largest DOWNWARD move allowed in a ticker's cumulative dividend-adjustment factor
#: before it is called a spliced vintage. Measured 2026-09-01 across 37 tickers on the
#: repaired cache: the worst genuine step was -2.11e-06 (CSV/float round-trip noise), and
#: re-injecting the defect that flipped the HAA canary produced -0.0073. 1e-4 sits ~50x
#: above the noise and ~70x below that seam, and still catches a 0.05% distribution.
ADJUSTMENT_STEP_TOLERANCE = 1e-4

#: Relative tolerance below which two adjusted closes for the same bar are "the same price".
#: CSV round-trip noise is ~1e-12; the smallest dividend worth catching moves a price by
#: several 1e-4. Anything in between is a real re-adjustment, so the threshold is loose
#: enough to ignore float noise and tight enough to miss no distribution.
ADJUSTMENT_TOLERANCE = 1e-6

#: Bars newer than this (calendar days before the newest overlapping bar) are excluded from
#: the re-adjustment check. The last few rows legitimately change as a session settles, and
#: that is not a re-adjustment; a dividend rescales EVERY row before its ex-date, so the
#: signature is unmistakable in the older rows of the overlap.
SETTLED_DAYS = 5

#: Hours to wait before re-checking Yahoo for a cache that already holds the newest bar it
#: can. Signals come from COMPLETE months and live order sizing uses a separate real-time
#: quote call, so an intraday-stale daily cache cannot change a decision. See
#: `PriceStore._refresh_tail` for the measurement that motivated it.
REFRESH_MIN_HOURS = 6.0

_FIELDS = ('open', 'close', 'adj_close')

# --------------------------------------------------------------------------------------- #
# History extension — 2026-07-29
#
# WHY THIS EXISTS. Every strategy in this repository began after the 2008 crisis had already
# started, so none of them could be measured through the one event they exist to survive. The
# binding constraint was never the strategy: it was the inception date of a fund. BIL began
# 2007-05-30, VEA 2007-07-26, BND 2007-04-10 — all *after* the S&P's 2007-10-09 peak.
#
# WHAT IS AND IS NOT DONE ABOUT IT. Two mechanisms, both narrow and both declared:
#
#   1. HISTORY_BACKFILL splices an OLDER FUND TRACKING THE SAME INDEX into the leading gap.
#      This is not a substitution: the traded ticker never changes, so the live path keeps
#      buying the cheap Vanguard fund. Only the pre-inception history comes from elsewhere,
#      chain-linked at the level so returns are continuous.
#   2. SYNTHETIC_CASH builds a T-bill total return from the published 13-week bill rate,
#      because no cash ETF predates BIL and the obvious substitute is a trap: SHY returned
#      +6.62% in 2008 against BIL's +1.59%. Splicing SHY in would hand every risk-off month
#      of the crisis a bond rally and manufacture the result being measured.
#
# WHAT IS DELIBERATELY NOT FIXED. High yield: HYG (2007-04-11) is the oldest US high-yield
# ETF that exists — JNK is 2007-12-04 — so every strategy holding it stays bounded at 2008-05
# and there is nothing to be done about it. Same for mortgage REITs (REM, 2007-05) and broad
# commodities (DBC, 2006-02; DJP is later). Those are limits of the market's product history,
# not of this file, and no amount of construction should be allowed to paper over them.
#
# EVERYTHING HERE IS FLAGGED. `provenance()` reports every spliced and synthetic span, the
# report prints them, and every manifest carries them. A constructed price is not a price.

#: {recipient: (donor, why)}. The donor must track the SAME INDEX over the spliced region,
#: which all three do: VEA and VWO tracked MSCI EAFE / MSCI EM until 2013 — well after the
#: window used here — and BND/AGG both track the Bloomberg US Aggregate.
HISTORY_BACKFILL = {
    'VEA': ('EFA', 'both tracked MSCI EAFE over the spliced years; VEA moved to FTSE in 2013'),
    'VWO': ('EEM', 'both tracked MSCI EM over the spliced years; VWO moved to FTSE in 2013'),
    'BND': ('AGG', 'both track the Bloomberg US Aggregate Bond index'),
}

#: {ticker: (rate_symbol, annual_expense, why)}. The rate symbol is dropped from the store
#: once consumed — it is a yield, not a price, and nothing may ever hold or rank it.
SYNTHETIC_CASH = {
    'BIL': ('^IRX', 0.001356,
            '13-week T-bill discount rate, accrued ACT/360 and charged BIL\'s own expense '
            'ratio so the constructed series meets the real fund without a step'),
}


class IncompleteMonthError(ValueError):
    """Raised when a caller asks for a month the store cannot yet vouch for."""


class DataGapError(ValueError):
    """Raised when a ticker has an interior gap longer than MAX_STALE_DAYS."""


class PriceStore:
    """Daily OHLC-adjusted price store with real trading dates and complete months only.

    Construct from Yahoo (the normal path), from in-memory frames (tests), or from the
    legacy monthly cache (the frozen fixture). `has_intraday` tells callers whether an
    open-based execution convention is even available; the ledger refuses `next_open`
    when it is False rather than silently falling back to the close.
    """

    def __init__(self, tickers, start='2000-01-01', cache_dir='data/cache',
                 download=True, strict_gaps=True, refresh_hours=REFRESH_MIN_HOURS):
        self.tickers = list(dict.fromkeys(tickers))
        self.start = pd.to_datetime(start)
        self.cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(ROOT_DIR, cache_dir)
        self.strict_gaps = strict_gaps
        self.has_intraday = True
        self.source = 'yahoo'
        self.downloaded_at = None
        self.fill_log = []
        self.constructed = {}
        #: Hours before the trailing-window re-download is attempted again. 0 = every run.
        self.refresh_hours = float(refresh_hours)
        #: Set when a refresh was skipped because the cache was checked recently — the
        #: report prints it, so a stale cache is never silent.
        self.refresh_skipped = None
        #: Tickers whose FULL history was re-downloaded because Yahoo had re-adjusted it
        #: since the cache was written. Recorded rather than silent: it means every earlier
        #: momentum score for those tickers was computed on a different vintage.
        self.readjusted = []
        #: Momentum cost of a restatement, per ticker, filled by `_restatement_impact`.
        self.restatement_impact = []
        #: Whether `close` is a genuinely RAW series, DERIVED from the frames by
        #: `_verify_adjustment_vintage`. False makes the vintage check inapplicable rather
        #: than passing, which is the difference between 'I checked' and 'I saw nothing'.
        self.has_raw_close = None
        self._frames = {}

        self._load(download=download)
        self._extend_history()
        self._apply_stale_policy()
        #: Last, so it judges the panel consumers actually read — splices and stale-fills
        #: included. Runs on EVERY construction, including `download=False` and a skipped
        #: refresh, because those are precisely the paths on which nothing else looks.
        self.verification = self._verify_adjustment_vintage()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_frames(cls, adj_close, open_=None, close=None, source='memory',
                    strict_gaps=False):
        """Build a store from in-memory daily frames. Used by tests and by callers that
        already hold data. When `open_` is absent the store reports `has_intraday=False`,
        which makes an open-priced execution convention an explicit error rather than a
        silent same-bar fill."""
        store = cls.__new__(cls)
        store.tickers = list(adj_close.columns)
        store.start = adj_close.index[0] if len(adj_close) else pd.NaT
        store.cache_dir = None
        store.strict_gaps = strict_gaps
        store.source = source
        store.downloaded_at = None
        store.fill_log = []
        # In-memory frames are taken exactly as given. A frozen fixture that silently grew
        # spliced history would stop being frozen, and the golden master rests on it.
        store.constructed = {}
        # No cache, so nothing to refresh — but the attributes must exist, because this
        # path bypasses __init__ and every consumer reads them.
        store.refresh_hours = 0.0
        store.refresh_skipped = None
        store.readjusted = []
        store.restatement_impact = []
        adj = adj_close.sort_index()
        raw_close = close.sort_index() if close is not None else adj.copy()
        if open_ is None:
            store.has_intraday = False
            raw_open = raw_close.copy()
        else:
            store.has_intraday = True
            raw_open = open_.sort_index()
        store._frames = {'adj_close': adj, 'close': raw_close, 'open': raw_open}
        # A fixture built from adjusted closes has close == adj_close, so the adjustment
        # factor is 1 everywhere and the vintage check can see NOTHING. Saying so is the
        # point: an 'ok' that means 'I checked nothing' is the fail-open shape that let a
        # cp1252 decode error report a modified worktree as clean.
        store.has_raw_close = None
        store._apply_stale_policy()
        store.verification = store._verify_adjustment_vintage()
        return store

    @classmethod
    def from_adjusted(cls, adj_close, adj_open, source='adjusted-frames', strict_gaps=False):
        """Build a store from ALREADY-ADJUSTED closes and opens.

        Both series are on the same scale, so the adjustment factor is 1 by construction —
        which is exactly what a frozen fixture wants: no raw/adjusted pair to keep in sync,
        and no chance of a fixture drifting when Yahoo restates a dividend.
        """
        return cls.from_frames(adj_close, open_=adj_open, close=adj_close,
                               source=source, strict_gaps=strict_gaps)

    @classmethod
    def from_daily_fixture(cls, adj_close_path, adj_open_path, strict_gaps=False):
        """Load a frozen daily fixture written by `tests/fixtures` (see its MANIFEST)."""
        ac = pd.read_csv(adj_close_path, index_col=0, parse_dates=True).sort_index()
        ao = pd.read_csv(adj_open_path, index_col=0, parse_dates=True).sort_index()
        return cls.from_adjusted(ac, ao, source=f'fixture:{os.path.basename(adj_close_path)}',
                                 strict_gaps=strict_gaps)

    @classmethod
    def from_monthly_csv(cls, path, strict_gaps=False):
        """Build a store from the legacy MONTHLY adjusted-close cache.

        Each row is treated as one observation dated exactly as labelled. The
        complete-month rule then applies unchanged — a month is only emitted once the file
        contains an observation in a later month — which is precisely what makes the frozen
        fixture's dangling `2026-06-30` row testable.

        There are no opens here, so `has_intraday` is False and any open-priced convention
        raises instead of quietly reverting to same-bar execution.
        """
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        store = cls.from_frames(df, source=f'monthly-csv:{os.path.basename(path)}',
                                strict_gaps=strict_gaps)
        store._monthly_source_path = path
        return store

    # ------------------------------------------------------------------ #
    # Cache I/O
    # ------------------------------------------------------------------ #
    def _cache_path(self, field):
        return os.path.join(self.cache_dir, f'daily_{field}.csv')

    def _read_cache(self):
        frames = {}
        for field in _FIELDS:
            path = self._cache_path(field)
            if not os.path.exists(path):
                return None
            try:
                frames[field] = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
            except Exception as exc:
                print(f"Warning: unreadable cache '{path}' ({exc}). Re-downloading.")
                return None
        if any(f.empty for f in frames.values()):
            return None
        return frames

    def _write_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        for field in _FIELDS:
            self._frames[field].to_csv(self._cache_path(field))

    def _load(self, download=True):
        frames = self._read_cache()
        missing = [] if frames is None else [t for t in self.tickers
                                             if t not in frames['adj_close'].columns]

        if frames is not None and not missing:
            self._frames = frames
            if download:
                self._refresh_tail()
            return

        if not download:
            if frames is None:
                raise FileNotFoundError(
                    f'No daily cache in {self.cache_dir} and download=False.')
            self._frames = frames
            return

        if frames is not None and missing:
            print(f'Cache is missing {len(missing)} ticker(s) {missing[:6]}... re-downloading in full.')
        self._frames = self._download(self.tickers, self.start)
        self._write_cache()

    def _refresh_stamp_path(self):
        return os.path.join(self.cache_dir, '.last_refresh')

    def _last_refresh_attempt(self):
        try:
            with open(self._refresh_stamp_path(), encoding='utf-8') as fh:
                return pd.Timestamp(fh.read().strip())
        except (OSError, ValueError):
            return None

    def _stamp_refresh(self):
        """Write the stamp atomically: temp file in the same directory, then os.replace.

        The direction of failure was already safe — `_last_refresh_attempt` reads a
        truncated stamp as ValueError and returns None, which means 'unknown' and forces
        a refresh. So this buys no correctness the reader did not already have. It buys
        the CLI and the GUI running at once never leaving a half-written stamp behind for
        someone to puzzle over. os.replace is atomic on the same filesystem on Windows too.
        """
        path = self._refresh_stamp_path()
        try:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.', prefix='.last_refresh-')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                    fh.write(pd.Timestamp.now().isoformat(timespec='seconds'))
                os.replace(tmp, path)
            except OSError:
                try:
                    os.unlink(tmp)  # do not leave the temp file behind on a failed write
                except OSError:
                    pass
                raise
        except OSError:
            pass                    # a missing stamp costs a redundant refresh, nothing more

    def _refresh_tail(self):
        """Re-download and OVERWRITE the trailing window, at most once per interval.

        Overwriting rather than appending is the fix for the frozen-row bug: a month-end
        row written while the month was still open used to survive forever because the
        appender only accepted strictly-newer dates.

        THE INTERVAL IS NOT A MICRO-OPTIMISATION. The only guard used to be
        `if last >= today: return`, comparing the newest cached bar against the CALENDAR
        date. That condition is false for the whole of every trading day — the newest bar is
        yesterday's close until this session ends — so every single run paid a full network
        round-trip for 90 days of every ticker and then rewrote ~10 MB of CSV, to learn that
        nothing had changed. Measured on 35 tickers: 0.18s to read the cache, 9.09s with the
        refresh (1.72s warm). Run the dashboard three times in a row and two of those runs
        were pure waste.

        A monthly strategy cannot be harmed by a cache that is a few hours old: signals come
        from COMPLETE months, and live order sizing uses `get_live_prices()`, a separate
        real-time call that does not touch this cache at all. Pass `refresh_hours=0` (or
        `--refresh` on the CLI) to force it; the stamp is a plain file you can delete.
        """
        last = self._frames['adj_close'].index[-1]
        # NOTE: there is deliberately no `if last >= today: return` fast path any more.
        # "Do we already hold today's bar?" and "has Yahoo restated the history we hold?"
        # are different questions, and the first one used to silently answer the second.
        # Once a distribution restated the adjusted series, the cache held today's bar, took
        # this exit, and never noticed the seam for the rest of the day — which is exactly
        # how a stale vintage reached a live signal on 2026-09-01. The interval below is the
        # throttle, and it is the only one needed: it already bounds the network to one
        # round-trip per `refresh_hours`.
        if self.refresh_hours > 0:
            prev = self._last_refresh_attempt()
            if prev is not None:
                age_h = (pd.Timestamp.now() - prev).total_seconds() / 3600.0
                if age_h < self.refresh_hours:
                    self.refresh_skipped = (
                        f'cache last checked {age_h:.1f}h ago (< {self.refresh_hours}h); '
                        f'newest bar {last.date()}. Use --refresh to force.')
                    return
        # Fetch the DETECTION window, not just the rewrite window: a restated adjustment is
        # only visible on bars the refresh does not already overwrite (see DETECT_WINDOW_DAYS).
        since = max(self.start, last - pd.Timedelta(days=DETECT_WINDOW_DAYS))
        try:
            fresh = self._download(self.tickers, since, quiet=True)
        except Exception as exc:
            print(f'Warning: incremental refresh failed ({exc}). Proceeding with the cache '
                  f'as of {last.date()} — treat any signal from it as stale.')
            return
        stale = self._readjusted_tickers(fresh)
        # Measure the cost BEFORE the merge: this is the only moment at which the cached
        # vintage and the vendor's current one both exist.
        self.restatement_impact = self._restatement_impact(fresh, stale)
        for field in _FIELDS:
            old, new = self._frames[field], fresh[field]
            new = new.reindex(columns=old.columns.union(new.columns))
            old = old.reindex(columns=new.columns)
            combined = pd.concat([old.loc[old.index < new.index[0]], new])
            self._frames[field] = combined[~combined.index.duplicated(keep='last')].sort_index()
        if stale:
            self._redownload_full(stale)
        self.downloaded_at = pd.Timestamp.now().isoformat(timespec='seconds')
        self._stamp_refresh()
        self._write_cache()

    def _readjusted_tickers(self, fresh):
        """Tickers whose cached history is a DIFFERENT ADJUSTMENT VINTAGE from `fresh`.

        Found 2026-09-01, by asking whether a live canary signal could be trusted. It could
        not. `adj_close` is a total-return series, so when a fund goes ex-dividend Yahoo
        rescales EVERY bar before that date. This refresh overwrites only the trailing
        window, so after any distribution the cache holds two vintages spliced at the window
        edge: recent bars carrying the new adjustment, older bars carrying the old one.

        Nothing about that splice looks wrong. Every price is plausible, the series is
        smooth, and the join is invisible — but a return whose two endpoints straddle the
        seam is computed across a ~0.7% discontinuity that no market ever traded. Measured on
        TIP: r6 and r12 understated by 0.73pp each, the 13612U canary score understated by
        0.36pp, and the previous month's HAA signal flipped from alive to dead on it. The
        bias has a direction — stale bars are always too HIGH, because they are missing
        adjustments — so momentum reads systematically LOW and a canary dies too eagerly.

        The check is per ticker, and deliberately not a blanket rebuild: with 37 tickers
        something has almost always just gone ex, and re-downloading everything on every
        distribution would turn a 2-second refresh into a minute of network for one fund's
        dividend.
        """
        old, new = self._frames['adj_close'], fresh['adj_close']
        cols = [c for c in new.columns if c in old.columns]
        idx = new.index.intersection(old.index)
        if not cols or len(idx) == 0:
            return []
        idx = idx[idx <= idx[-1] - pd.Timedelta(days=SETTLED_DAYS)]
        if len(idx) == 0:
            return []
        a = old.loc[idx, cols].to_numpy(dtype=float)
        b = new.loc[idx, cols].to_numpy(dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            rel = np.abs(b - a) / np.abs(a)
        rel = np.where(np.isfinite(rel), rel, 0.0)
        return [c for j, c in enumerate(cols) if rel[:, j].max() > ADJUSTMENT_TOLERANCE]

    def _restatement_impact(self, fresh, stale):
        """What a restatement COSTS, in the units that decide: momentum lookbacks.

        `_readjusted_tickers` answers "did any bar move". That is a detection, and on its
        own it says nothing about whether anything downstream would have noticed. This
        answers "does it move a momentum score", which is the question the strategies ask.

        It exists because the figure that made the 2026-09-01 incident legible -- TIP's r6
        and r12 understated by 0.73pp each, the 13612U canary score by 0.36pp -- was
        computed by hand and then written into prose, where nothing could keep it true.
        Note the shape: r1 alone would have missed the defect entirely, because its base is
        inside the refresh window and therefore always current. Only the long legs cross the
        seam, which is exactly why a stale cache reads as weak momentum rather than as
        obviously broken data.

        Costs no network: `fresh` is already in hand.
        """
        out = []
        old_frame, new_frame = self._frames['adj_close'], fresh['adj_close']

        def month_ends(series):
            s = series.dropna()
            if s.empty:
                return s
            m = s.groupby([s.index.year, s.index.month]).tail(1)
            return m[~m.index.duplicated()]

        for ticker in stale:
            if ticker not in old_frame.columns or ticker not in new_frame.columns:
                continue
            old_me, new_me = month_ends(old_frame[ticker]), month_ends(new_frame[ticker])
            shared = old_me.index.intersection(new_me.index)
            if len(shared) < 13:
                # `fresh` spans DETECT_WINDOW_DAYS, so a 12-month lookback is normally just
                # inside it. When it is not, say nothing rather than report a partial cost.
                continue
            old_me, new_me = old_me.reindex(shared), new_me.reindex(shared)
            record = {'ticker': ticker}
            legs = {'r1': 2, 'r3': 4, 'r6': 7, 'r12': 13}
            deltas = []
            for name, back in legs.items():
                p_old, p_new = float(old_me.iloc[-1]), float(new_me.iloc[-1])
                r_old = p_old / float(old_me.iloc[-back]) - 1.0
                r_new = p_new / float(new_me.iloc[-back]) - 1.0
                record[name + '_pp'] = (r_new - r_old) * 100.0
                deltas.append(r_new - r_old)
            # The unweighted 13612U the HAA canary reads, so the number is directly
            # comparable to the threshold it is tested against.
            record['s13612u_pp'] = (sum(deltas) / 4.0) * 100.0
            out.append(record)
        return out

    def _redownload_full(self, tickers):
        """Replace those tickers' whole history, so the series carries ONE vintage.

        Patching only the seam would not do: the adjustment factor applies to every bar
        before the ex-date, which is most of the history. The column has to come back whole.
        """
        print(f'Re-adjusted history detected for {len(tickers)} ticker(s) '
              f'{sorted(tickers)[:6]}{"..." if len(tickers) > 6 else ""} — Yahoo has restated '
              f'their dividend adjustment. Re-downloading those in full so the cached series '
              f'is not two vintages spliced together.')
        try:
            full = self._download(tickers, self.start, quiet=True)
        except Exception as exc:
            print(f'Warning: full re-download failed ({exc}). The cached history for '
                  f'{sorted(tickers)} is a MIXED adjustment vintage — momentum over windows '
                  f'longer than {REFRESH_WINDOW_DAYS} days is unreliable until it succeeds.')
            return
        for field in _FIELDS:
            cur, new = self._frames[field], full[field]
            index = cur.index.union(new.index)
            cur = cur.reindex(index)
            for t in tickers:
                if t in new.columns:
                    cur[t] = new[t].reindex(index)
            self._frames[field] = cur.sort_index()
        self.readjusted = sorted(set(self.readjusted) | set(tickers))

    def _download(self, tickers, start, quiet=False):
        if not quiet:
            print(f'Downloading {len(tickers)} tickers of DAILY bars from Yahoo since '
                  f'{pd.to_datetime(start).date()}...')
        raw = yf.download(list(tickers), start=pd.to_datetime(start), progress=False,
                          auto_adjust=False, group_by='column')
        if raw.empty:
            raise ValueError('No data returned from yfinance.')

        out = {}
        for field, col in (('open', 'Open'), ('close', 'Close'), ('adj_close', 'Adj Close')):
            if isinstance(raw.columns, pd.MultiIndex):
                frame = raw[col].copy()
            else:
                frame = raw[[col]].copy()
                frame.columns = [tickers[0]]
            for t in tickers:
                if t not in frame.columns:
                    frame[t] = np.nan
            out[field] = frame[list(tickers)].sort_index()
        self.downloaded_at = pd.Timestamp.now().isoformat(timespec='seconds')
        return out

    # ------------------------------------------------------------------ #
    # History extension (declared, chain-linked, and recorded)
    # ------------------------------------------------------------------ #
    def _extend_history(self):
        """Fill each declared ticker's LEADING gap, and record exactly what was done.

        Runs before the stale policy, so a spliced span becomes part of the ticker's real
        history for `first_tradable_date` and therefore for the coverage guard. Runs after
        the download, so the donor series is present.

        Order matters: synthetic cash first (it consumes and then removes its rate symbol),
        then the fund splices.
        """
        self.constructed = {}
        if not self._frames:
            return
        self._build_synthetic_cash()
        self._splice_donors()

    def _build_synthetic_cash(self):
        ac = self._frames['adj_close']
        for ticker, (rate_sym, expense, why) in SYNTHETIC_CASH.items():
            if ticker not in ac.columns or rate_sym not in ac.columns:
                continue
            first = ac[ticker].first_valid_index()
            rate = ac[rate_sym]
            head = ac.index[ac.index < first] if first is not None else ac.index
            head = head[rate.reindex(head).notna()]
            if len(head) < 2:
                continue

            # ACT/360 on the published discount rate, less the fund's own expense ratio, so
            # the constructed series is comparable to the product that replaces it rather
            # than to an untradeable ideal. Day counts are clipped: a gap longer than a long
            # weekend means missing data, not two weeks of accrual.
            days = pd.Series(head, index=head).diff().dt.days.fillna(1).clip(lower=1, upper=5)
            # ^IRX is a DISCOUNT rate on a 91-day bill, not a rate of return. A bill bought
            # at a 4.50% discount costs 98.8625 and repays 100, which is a 4.70% annual
            # return. Skipping this conversion understates the cash sleeve by ~20bp a year
            # for a decade — small per month, and exactly the kind of small that compounds
            # into a wrong Sharpe.
            d = rate.reindex(head).ffill() / 100.0
            add_on = d / (1.0 - d * 91.0 / 360.0)
            accrual = add_on * days / 360.0 - expense * days / 365.0
            level = (1.0 + accrual.fillna(0.0)).cumprod()

            # Chain to the real fund: the last constructed day carries the fund's first
            # price, so the transition itself contributes no return. One day of foregone
            # bill accrual (~0.0014% at 5%/yr) buys a splice with no step in it.
            anchor = float(ac.loc[first, ticker]) if first is not None else 100.0
            level = level / float(level.iloc[-1]) * anchor
            for field in _FIELDS:
                # open == close over the constructed span: a bill fund has no intraday move
                # to model, only accrual, and adj_open is derived as open*adj_close/close.
                self._frames[field].loc[head, ticker] = level.values

            self.constructed[ticker] = {
                'kind': 'synthetic', 'source': rate_sym, 'why': why,
                'from': str(head[0].date()), 'to': str(head[-1].date()),
                'real_from': str(pd.Timestamp(first).date()) if first is not None else None,
            }

        # The rate symbols are yields, not prices. Drop them so nothing can hold one, rank
        # one, or wander into one through a universe declaration.
        rate_symbols = {r for (r, _, _) in SYNTHETIC_CASH.values()}
        for field in _FIELDS:
            f = self._frames[field]
            drop = [c for c in rate_symbols if c in f.columns]
            if drop:
                self._frames[field] = f.drop(columns=drop)
        self.tickers = [t for t in self.tickers if t not in rate_symbols]

    def _splice_donors(self):
        ac = self._frames['adj_close']
        for recipient, (donor, why) in HISTORY_BACKFILL.items():
            if recipient not in ac.columns or donor not in ac.columns:
                continue
            r_first, d_first = ac[recipient].first_valid_index(), ac[donor].first_valid_index()
            if r_first is None or d_first is None or d_first >= r_first:
                continue
            if pd.isna(ac.loc[r_first, donor]):
                continue

            # Chain-link on LEVEL at the recipient's own first day, so the donor contributes
            # its returns and none of its price. Scaling all three fields by the same factor
            # leaves adj_open = open * adj_close / close unchanged, which is what keeps the
            # spliced opens on the adjusted scale.
            k = float(ac.loc[r_first, recipient]) / float(ac.loc[r_first, donor])
            head = ac.index[(ac.index < r_first) & ac[donor].notna()]
            if len(head) == 0:
                continue
            for field in _FIELDS:
                f = self._frames[field]
                self._frames[field].loc[head, recipient] = f.loc[head, donor].values * k

            self.constructed[recipient] = {
                'kind': 'spliced', 'source': donor, 'why': why,
                'from': str(head[0].date()), 'to': str(head[-1].date()),
                'real_from': str(pd.Timestamp(r_first).date()),
            }

    def constructed_before(self, ticker):
        """First date at which `ticker`'s data is the fund's OWN, or None if it always was.

        The live path must never size an order on a constructed price. Every constructed
        span ends before its ticker's real inception by construction, so this is a fact the
        tests can assert rather than a promise the code makes.
        """
        rec = getattr(self, 'constructed', {}).get(ticker)
        return pd.Timestamp(rec['real_from']) if rec and rec.get('real_from') else None

    # ------------------------------------------------------------------ #
    # Missing-data policy (single policy, applied once, recorded)
    # ------------------------------------------------------------------ #
    def _apply_stale_policy(self):
        """Forward-fill at most MAX_STALE_DAYS trading days *inside* each ticker's life.

        Leading NaNs (before inception) are never filled — that is what makes
        `first_tradable_date` meaningful and what lets the coverage guard refuse a window
        the product did not exist in. A longer interior gap is not staleness, it is
        absence, and it raises under `strict_gaps`.
        """
        adj = self._frames['adj_close']
        gaps = []
        for col in adj.columns:
            s = adj[col]
            valid = s.notna()
            if not valid.any():
                continue
            lo, hi = valid.idxmax(), valid[::-1].idxmax()
            interior = s.loc[lo:hi]
            run, longest, filled = 0, 0, 0
            for is_na in interior.isna().values:
                if is_na:
                    run += 1
                    longest = max(longest, run)
                    filled += 1
                else:
                    run = 0
            if longest:
                self.fill_log.append({'ticker': col, 'filled_days': int(filled),
                                      'longest_run': int(longest)})
            if longest > MAX_STALE_DAYS:
                gaps.append(f'{col}: interior gap of {longest} trading days '
                            f'(max allowed {MAX_STALE_DAYS})')

        for field in _FIELDS:
            f = self._frames[field]
            filled = f.ffill(limit=MAX_STALE_DAYS)
            # ffill leaves leading NaNs alone (which is what makes first_tradable_date
            # meaningful) but would happily carry a price PAST a ticker's last real
            # observation. Cut that off: a delisted product has no price, not a flat one.
            for col in f.columns:
                last_real = f[col].last_valid_index()
                if last_real is not None:
                    filled.loc[filled.index > last_real, col] = np.nan
            self._frames[field] = filled

        self._month_ends = None
        self.long_gaps = gaps
        if gaps and self.strict_gaps:
            raise DataGapError(
                'Price history has gaps longer than MAX_STALE_DAYS=%d:\n  %s\n'
                'Carrying these forward would invent a flat return over a period the asset '
                'may well have been moving. Fix the data, shorten the window, or construct '
                'the store with strict_gaps=False and accept the recorded fills.'
                % (MAX_STALE_DAYS, '\n  '.join(gaps)))

    # ------------------------------------------------------------------ #
    # Public accessors
    # ------------------------------------------------------------------ #
    @property
    def trading_days(self):
        return self._frames['adj_close'].index

    def adj_close(self):
        return self._frames['adj_close']

    def adj_open(self):
        """Opens rebased onto the adjusted-close scale via adj_factor = adj_close / close."""
        if not self.has_intraday:
            raise ValueError('This store carries no opens (has_intraday=False). An '
                             'open-priced execution convention is unavailable; use '
                             "convention='signal_close' explicitly if that is what you mean.")
        factor = self._frames['adj_close'] / self._frames['close']
        return self._frames['open'] * factor

    def as_of(self):
        """Last observation in the store. Staleness is measured against this — a real
        trading date — never against a resample label.

        Note it may be an in-progress session: Yahoo publishes a partial bar for the current
        day. Nothing downstream depends on that, because results are gated on
        `_complete_month_ends`, which cannot include a month that has not ended.
        """
        return self.trading_days[-1]

    def first_tradable_date(self, ticker):
        """First date this ticker has a price. None if it is absent or empty."""
        adj = self._frames['adj_close']
        if ticker not in adj.columns:
            return None
        s = adj[ticker]
        return s.first_valid_index()

    # ------------------------------------------------------------------ #
    # Calendars
    # ------------------------------------------------------------------ #
    def _complete_month_ends(self):
        """Real last trading day of every COMPLETE month in the store.

        A month is complete once the store holds an observation in a LATER month. The
        trailing month is therefore never emitted while it is still running — which is the
        structural fix for C4: `resample('ME').last()` had no such notion and happily
        published a five-day-old price under a month-end label.
        """
        if self._month_ends is not None:
            return self._month_ends
        idx = self.trading_days
        if len(idx) == 0:
            self._month_ends = pd.DatetimeIndex([])
            return self._month_ends
        last_of_month = pd.Series(idx, index=idx.to_period('M')).groupby(level=0).last()
        # Drop the final month: nothing later proves it finished.
        self._month_ends = pd.DatetimeIndex(last_of_month.iloc[:-1].values)
        return self._month_ends

    def month_end_dates(self, start=None, end=None, closed_by=None):
        """Complete month-ends, optionally including a trailing month the CALENDAR closed.

        `closed_by` is a date you are standing on — the live path passes its execution
        date. It exists because "a later observation proves the month finished" is a proxy
        for "the month finished", and the proxy fails on exactly the day that matters. Run
        live on 2026-08-01, before any August bar exists, and July's month-end is withheld:
        the report then sizes orders from the JUNE decision, a full month stale, and says
        nothing about it.

        The trailing month is admitted only when both are true:

        * `closed_by` is strictly after the store's last observation — never equal, because
          Yahoo publishes a partial bar for the session in progress and a partial bar must
          never be used as a month-end close;
        * no business day remains in that month after the last observation, so no further
          session can arrive to change it.

        Both conditions failing just returns the strict answer, so this can only ever add a
        month the calendar has already closed. Backtests do not pass `closed_by` and are
        bit-for-bit unaffected.
        """
        idx = self._complete_month_ends()
        if closed_by is not None:
            idx = self._admit_closed_trailing_month(idx, pd.Timestamp(closed_by))
        if start is not None:
            idx = idx[idx >= pd.to_datetime(start)]
        if end is not None:
            idx = idx[idx <= pd.to_datetime(end)]
        return idx

    def _admit_closed_trailing_month(self, idx, asof):
        days = self.trading_days
        if len(days) == 0:
            return idx
        last = days[-1]
        if len(idx) and pd.Period(idx[-1], freq='M') >= pd.Period(last, freq='M'):
            return idx                                   # already emitted by the strict rule
        if asof <= last:
            return idx                                   # the last bar may still be running
        month_end = pd.Timestamp(last).to_period('M').end_time.normalize()
        if len(pd.bdate_range(last + pd.Timedelta(days=1), month_end)):
            return idx                                   # another session could still land
        return pd.DatetimeIndex(list(idx) + [last])

    def assert_month_complete(self, month):
        """Raise unless `month` (anything Period('M') accepts) is closed in this store."""
        want = pd.Period(pd.to_datetime(month), freq='M')
        have = self._complete_month_ends()
        if len(have) and want in set(pd.PeriodIndex(have, freq='M')):
            return
        last = pd.Period(have[-1], freq='M') if len(have) else 'none'
        raise IncompleteMonthError(
            f'{want} is not complete in this store (last complete month: {last}; last '
            f'observation: {self.as_of().date()}). A month is only usable once the store '
            f'holds a trading day in a LATER month; publishing it earlier is audit '
            f'finding C4.')

    def rebalance_dates(self, schedule='month_end', start=None, end=None):
        """Trading dates on which decisions are taken.

        `month_end` — the real last trading day of each complete month.
        `day_N`     — the Nth trading day of each month (N = 1..19), used by the
                      timing-luck sweep. The month must still be complete, so the
                      trailing month is excluded exactly as above.
        """
        if schedule == 'month_end':
            return self.month_end_dates(start, end)
        if not schedule.startswith('day_'):
            raise ValueError(f"Unknown schedule {schedule!r}: use 'month_end' or 'day_N'.")
        n = int(schedule.split('_')[1])
        if not 1 <= n <= 19:
            raise ValueError('day_N schedules are defined for N = 1..19.')
        idx = self.trading_days
        periods = idx.to_period('M')
        complete_months = set(pd.PeriodIndex(self._complete_month_ends(), freq='M'))
        picks = []
        for period, group in pd.Series(idx, index=periods).groupby(level=0):
            if period not in complete_months or len(group) < n:
                continue
            picks.append(group.iloc[n - 1])
        out = pd.DatetimeIndex(picks)
        if start is not None:
            out = out[out >= pd.to_datetime(start)]
        if end is not None:
            out = out[out <= pd.to_datetime(end)]
        return out

    def monthly_adj_close(self, dates):
        """Signal-grade monthly panel sampled at REAL trading dates."""
        dates = pd.DatetimeIndex(dates)
        return self._frames['adj_close'].reindex(dates)

    def next_trading_day(self, date):
        """First session strictly after `date`. None if the store ends there."""
        idx = self.trading_days
        pos = idx.searchsorted(pd.to_datetime(date), side='right')
        return idx[pos] if pos < len(idx) else None

    # ------------------------------------------------------------------ #
    # Provenance
    # ------------------------------------------------------------------ #
    def _verify_adjustment_vintage(self):
        """Is this panel ONE dividend-adjustment vintage, or two spliced together?

        `adj_close` is a total-return series and `close` is not, so their ratio

            f(t) = adj_close(t) / close(t)

        is the cumulative dividend-adjustment factor. Every distribution scales all EARLIER
        bars down and leaves raw closes alone, so **f can only rise with time**. A cache
        holding two vintages therefore shows a downward STEP in f exactly at the seam.

        Why this rather than a comparison against a fresh download: the refresh already
        overwrites its whole fetch window, so comparing the merged store against that fetch
        compares fresh data with itself and sees nothing. This test is internal, costs no
        network at all, and works on the paths where no refresh runs -- offline,
        `download=False`, or the throttle -- which is exactly where the 2026-09-01 defect
        survived. Measured across 37 tickers: 108 ms, worst genuine step -2.11e-06, and the
        injected defect -0.0073, a signal-to-noise ratio of about 3450.

        It is not self-referential either: it checks the cache against a property of the
        VENDOR's data model that no code in this repository produced.

        Two exclusions, both found by measurement rather than reasoning:

        * a CONSTRUCTED span is donor data on the donor's own price scale, so its junction
          is a legitimate level change, not a restated dividend. On the real store those
          junctions step -0.126 (VWO), -0.232 (BIL) and -0.018 (BND), each landing exactly
          on that span's `real_from`. `VEA` passes today only because EFA's factor happens
          to sit close, so ALL FOUR declared spans are excluded rather than the three that
          currently fire.
        * the newest `SETTLED_DAYS` bars, where a dividend can be announced and not yet
          applied.
        """
        result = {
            'status': 'not_applicable',
            'checked_at': pd.Timestamp.now().isoformat(timespec='seconds'),
            'tolerance': ADJUSTMENT_STEP_TOLERANCE,
            'n_tickers_checked': 0,
            'worst_step': None,
            'worst_ticker': None,
            'violations': [],
            'reason': None,
        }
        adj, raw = self._frames.get('adj_close'), self._frames.get('close')
        if adj is None or raw is None or not len(adj):
            result['reason'] = 'no adjusted/raw pair to compare'
            return result
        # Derived from the frames, never assumed. Whether a store has genuinely raw closes
        # is a property of its data, and a cache could hold a degenerate pair as easily as
        # an in-memory fixture does. Trusting a flag here would reproduce the fail-open this
        # method exists to close: reporting 'ok' after checking nothing.
        self.has_raw_close = not raw.equals(adj)
        if not self.has_raw_close:
            result['reason'] = ('close IS adj_close for this store, so the adjustment '
                                'factor is 1 by construction and this check can see '
                                'nothing')
            return result

        cutoff = adj.index[-1] - pd.Timedelta(days=SETTLED_DAYS)
        worst, worst_ticker, violations, checked = 0.0, None, [], 0

        for col in [c for c in adj.columns if c in raw.columns]:
            f = (adj[col] / raw[col]).replace([np.inf, -np.inf], np.nan).dropna()
            real_from = self.constructed_before(col)
            if real_from is not None:
                f = f[f.index >= real_from]
            f = f[f.index <= cutoff]
            if len(f) < 30:
                continue
            checked += 1
            step = (f / f.shift(1) - 1.0).dropna()
            if step.empty:
                continue
            low = float(step.min())
            if low < worst:
                worst, worst_ticker = low, col
            for date, s in step[step < -ADJUSTMENT_STEP_TOLERANCE].items():
                violations.append({
                    'ticker': col,
                    'date': str(date.date()),
                    'step': float(s),
                    # Reported so a -40% "step" reads as a split artefact rather than a
                    # dividend seam. They are different investigations.
                    'implied_distribution_pct': float(-s) * 100.0,
                })

        result['n_tickers_checked'] = checked
        result['worst_step'] = worst
        result['worst_ticker'] = worst_ticker
        result['violations'] = violations[:50]
        if checked == 0:
            result['reason'] = 'no column had enough settled bars to check'
        else:
            result['status'] = 'disagrees' if violations else 'ok'
        return result

    def verification_lines(self):
        """The verdict as report lines. Empty when there is nothing the reader must know."""
        v = getattr(self, 'verification', None)
        if not v or v['status'] == 'ok':
            return []
        if v['status'] == 'not_applicable':
            return ['  data check: NOT APPLICABLE -- ' + str(v.get('reason'))]
        worst = v['violations'][0]
        others = len(v['violations']) - 1
        plural = 'y' if len(v['violations']) == 1 else 'ies'
        return [
            '  ! DATA CHECK FAILED: {} adjustment discontinuit{} across {} tickers.'.format(
                len(v['violations']), plural, v['n_tickers_checked']),
            '    worst: {} at {} steps {:+.4%} (implies a {:.2f}% distribution){}'.format(
                worst['ticker'], worst['date'], worst['step'],
                worst['implied_distribution_pct'],
                ', and {} more'.format(others) if others else ''),
            '    The cached history holds more than one dividend-adjustment vintage. A '
            'return crossing that date',
            '    is measured across a discontinuity that never traded, and the error runs '
            'one way: stale bars are',
            '    too high, so momentum reads too low. Re-run with --refresh, or delete '
            'data/cache to rebuild.',
        ]

    def provenance(self):
        adj = self._frames['adj_close']
        payload = adj.to_csv().encode('utf-8')
        return {
            'sha256': hashlib.sha256(payload).hexdigest(),
            'source': self.source,
            'downloaded_at': self.downloaded_at,
            'rows': int(len(adj)),
            'tickers': int(adj.shape[1]),
            'first_observation': str(adj.index[0].date()) if len(adj) else None,
            'as_of': str(self.as_of().date()) if len(adj) else None,
            'last_complete_month': (str(pd.Period(self._complete_month_ends()[-1], freq='M'))
                                    if len(self._complete_month_ends()) else None),
            'has_intraday': self.has_intraday,
            'max_stale_days': MAX_STALE_DAYS,
            'forward_filled': self.fill_log,
            'long_gaps': list(getattr(self, 'long_gaps', [])),
            # A constructed price is not a price. Every spliced or synthetic span travels
            # with the artefact that used it, so no figure can be quoted without it.
            'constructed_history': getattr(self, 'constructed', {}),
            # The integrity verdict, and the machinery that produced it. `sha256` above
            # pins WHICH data was used; nothing pinned whether anything had checked it,
            # nor which detector version judged it. A report written before the 2026-09-01
            # vintage fix and one written after were indistinguishable in the manifest.
            'verification': getattr(self, 'verification', None),
            'readjusted': list(getattr(self, 'readjusted', [])),
            'restatement_impact': list(getattr(self, 'restatement_impact', [])),
            'refresh_skipped': getattr(self, 'refresh_skipped', None),
            'adjustment_policy': {
                'refresh_window_days': REFRESH_WINDOW_DAYS,
                'detect_window_days': DETECT_WINDOW_DAYS,
                'settled_days': SETTLED_DAYS,
                'adjustment_tolerance': ADJUSTMENT_TOLERANCE,
                'adjustment_step_tolerance': ADJUSTMENT_STEP_TOLERANCE,
            },
        }

    def provenance_json(self):
        return json.dumps(self.provenance(), indent=2, sort_keys=True)


def get_live_prices(tickers):
    """Fetch the most recent market price (raw Close) for each ticker.

    Used ONLY for live order sizing: the monthly SIGNAL stays based on month-end
    adjusted closes, but share quantities are computed at current prices so orders
    match what the broker will actually charge (a month-end close can be days old
    by execution time, which oversizes orders and gets them rejected).

    Returns (prices: pd.Series indexed by ticker, asof: pd.Timestamp of the quote).
    Raises on failure — callers fall back to month-end prices with a warning.
    """
    df = yf.download(list(tickers), period='5d', interval='1d', progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError("No live data returned from yfinance.")

    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            closes = df['Close']
        else:
            closes = df.xs('Close', axis=1, level=1)
    else:
        closes = df[['Close']].rename(columns={'Close': list(tickers)[0]})

    closes = closes.ffill()
    last = closes.iloc[-1]
    return last, closes.index[-1]


# NOTE: get_unemployment_data() (FRED UNRATE) was removed on 2026-07-28 together with the only
# two strategies that used it, LAA and RAA. Yahoo is now the engine's single external data
# source. Removing it also retires a "cannot verify" item: FRED serves the CURRENT vintage of
# UNRATE, which is revised, so LAA/RAA were reading values that were never published at the
# decision date. Restoring either strategy means restoring that caveat — and doing it properly
# requires ALFRED point-in-time vintages, not FRED.
#
# NOTE: get_data() (the monthly resample + append-only cache) was removed on 2026-07-28. It was
# the mechanism behind audit findings C1, C4 and the frozen-row bug. Use PriceStore.
