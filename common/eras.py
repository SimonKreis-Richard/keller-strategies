"""
Pre-registered market-regime segmentations, and the era over which the strategies in this
repository can be compared at all.

WHY THIS FILE EXISTS
--------------------
The engine used to take a `START_DATE` and an `END_DATE` from the user. Every number it
printed was therefore conditional on a choice nobody could justify: 2015 flatters trend
following, 2009 flatters buy-and-hold, 2007 flatters anything holding bonds. A backtest whose
headline moves when you nudge a date is not measuring a strategy, it is measuring the date.

So the window is gone. What replaces it is not "the longest possible history" — that is just
a different arbitrary choice, and it silently rewards whichever strategy happens to own the
oldest tickers. What replaces it is a **partition**: the era is cut into consecutive segments
whose boundaries were set by bodies outside this repository (the NBER's dating committee, the
FOMC's own rate decisions, the BLS's published CPI, and the conventional +/-20% rule on the
S&P 500), and every strategy is reported in every segment.

Four properties make a segmentation usable as an antidote rather than as another knob:

1. **Exogenous.** Each boundary is a dated public fact — an NBER announcement, an FOMC
   decision, a CPI print, an index peak. None of them can be moved because a strategy looks
   bad on one side of it. `validate_partition` runs at import; a typo is an ImportError.
2. **Exhaustive.** The segments tile the era end to end. There is no month that belongs to no
   segment, so no month can be quietly dropped.
3. **Disjoint.** No month is counted twice, so the segments of a partition compound back to
   the era's own return. `tests/test_eras.py` asserts exactly that.
4. **Frozen.** Adding, removing or moving a segment is a decision to be argued in a commit
   message, never a response to a result.

THE ERA STARTS 2000-01, BECAUSE EVERY PUBLISHED STRATEGY CAN NOW BE MEASURED THERE
----------------------------------------------------------------------------------
`COMMON_ERA_START` has moved three times.

  * 2026-07-29, **2008-07 -> 2001-03**: the history extension in `common/data_engine.py`
    pushed BIL, VEA, VWO and BND back past their fund inceptions, so the floor stopped being
    set by a PRODUCT (BIL, 2007-05) and started being set by a DOWNLOAD PARAMETER.
  * 2026-07-31, **2001-03 -> 2004-11**: that bought a stretch only `SPY_Benchmark` could
    populate — eight segments whose "leaderboard" had a single row. The floor became the first
    month at which a STRATEGY, not a benchmark, could be measured: GLD's 2004-11 inception.
  * 2026-09-23, **2004-11 -> 2000-01**, at the owner's decision. The products that bound
    every family — GLD, TIP, DBC, GSG, HYG, the Treasury ETFs, VEU — were extended backwards
    on MUTUAL FUNDS and INDICES admitted by measurement against a yardstick the market sets
    (see `common/data_engine.py`, block "the era moved to 2000-01"). Every published strategy
    in the registry is now measurable from 2000-01, so the 2026-07-31 objection is answered
    rather than overruled: no segment here is a leaderboard of one.
    `DM_G8_Composite` was deleted the same day: its mortgage-REIT leg has no honest donor.

The floor is still DERIVED, in the other direction now: `main.DATA_START_DATE` is 1998-11,
the 13 months of warm-up the first 2000-01 return needs.

What this bought: **the 2000-2002 dot-com bear, whole** (`bear_dotcom`, `contraction_2001`),
a second full equity bear before the GFC and a different KIND of one — a slow valuation bust
with no credit crisis, falling rates and rising bonds, where the GFC was a credit crisis. A
strategy whose protection only ever worked in one of those is fitted to that one.

What it still does not buy: the 1970s inflation, 1987, the early 1990s. The donors reach back
further than the era does, but the TIPS canary does not — the oldest TIPS fund opened 1997 —
and an era that HAA could not enter would reopen the leaderboard-of-one problem. Leveraged
products still begin 2006-2010 and are measured over their own history, as before.

EX-POST LABELS
--------------
Every boundary below was knowable only afterwards. The NBER dated the 2020 trough in July
2021; the S&P's 2022 low was a low only in hindsight. These segments describe how a strategy
BEHAVED in a regime, which is a description and does not decay. They must never be read as a
regime the strategy could have identified at the time, and nothing in the engine feeds them
back into a signal.
"""

from collections import namedtuple

import numpy as np
import pandas as pd

# Imported as a module, not by name, so `RANK_KEYS` / `SEGMENT_MIN_MONTHS` are read from the one
# place that defines them rather than copied here at import time.
from common import metrics as metrics_mod

#: First month of the era. Since 2026-09-23 every published strategy is measurable from it —
#: see the module docstring for the three moves and why this one needed no leaderboard of
#: one. The GFC segments open 2007-11 (`bear_gfc`) and 2007-12 (`contraction_2008`); the
#: floor may never rise above 2007-11 for that reason alone, and now sits eight years below.
COMMON_ERA_START = '2000-01-01'

#: A slice of the era. `end=None` means "open" — it runs to the last complete month in the
#: data. `adverse` marks segments defined by an objectively bad state of the world
#: (contraction, bear market, inflation above 4%) and is used only to build the combined
#: "all adverse months" column. It is never applied by judgement: a Fed hiking cycle is not
#: marked adverse, because "hiking is bad" is an opinion and this file may not hold opinions.
Segment = namedtuple('Segment', 'key start end label adverse')

#: A complete, gapless, non-overlapping cover of the era, with the authority that dated it.
Segmentation = namedtuple('Segmentation', 'key title source note segments')


# --------------------------------------------------------------------------------------- #
# 1. Business cycle — NBER Business Cycle Dating Committee
# --------------------------------------------------------------------------------------- #
# Monthly peaks and troughs: peak 2007-12, trough 2009-06; peak 2020-02, trough 2020-04.
# Convention: a contraction runs from the month AFTER the peak through the trough month.
BUSINESS_CYCLE = Segmentation(
    key='business_cycle',
    title='US BUSINESS CYCLE (NBER)',
    source='NBER Business Cycle Dating Committee, US business cycle expansions and '
           'contractions (peaks 2001-03, 2007-12 and 2020-02; troughs 2001-11, 2009-06 and '
           '2020-04). Cross-checked 2026-09-23 against FRED USREC, which is 1 for exactly '
           '2001-04..2001-11',
    note='A contraction runs from the month AFTER the peak through the trough month. The era '
         'opens 106 months into the 1991-2001 expansion, so that segment is measured from '
         '2000-01 rather than from the NBER trough.',
    segments=(
        Segment('expansion_1991', '2000-01-01', '2001-03-31',
                'Expansion (1991-03 -> 2001-03, 120 months, then the longest on record); the '
                'era covers its last 15', False),
        Segment('contraction_2001', '2001-04-01', '2001-11-30',
                'Contraction (dot-com bust and 9/11, 8 months)', True),
        Segment('expansion_2001', '2001-12-01', '2007-11-30',
                'Expansion (the housing and credit cycle; 73 months from the 2001-11 trough)',
                False),
        Segment('contraction_2008', '2007-12-01', '2009-06-30',
                'Contraction (GFC, 19 months, the longest since 1945)', True),
        Segment('expansion_2009', '2009-07-01', '2020-02-29',
                'Expansion (128 months, longest on record)', False),
        Segment('contraction_2020', '2020-03-01', '2020-04-30',
                'Contraction (COVID, 2 months, shortest on record)', True),
        Segment('expansion_2020', '2020-05-01', None,
                'Expansion', False),
    ),
)

# --------------------------------------------------------------------------------------- #
# 2. Equity market cycle — conventional +/-20% rule on the S&P 500 price index
# --------------------------------------------------------------------------------------- #
# Turning points are intramonth (2009-03-09, 2020-02-19, 2020-03-23, 2022-01-03, 2022-10-12).
# A monthly return series cannot split a month, so each month is assigned whole to the phase
# that dominated it: March 2009 (+8.5%) and October 2022 (+8.0%) are recoveries and belong to
# the bull that started inside them; February 2020 (-8.2%) belongs to the bear.
EQUITY_CYCLE = Segmentation(
    key='equity_cycle',
    title='EQUITY MARKET CYCLE (S&P 500, conventional +/-20% rule)',
    source='S&P 500 price index peaks and troughs: 2000-03-24 -> 2002-10-09 (-49.1%), '
           '2007-10-09 -> 2009-03-09 (-56.8%), 2020-02-19 -> 2020-03-23 (-33.9%), '
           '2022-01-03 -> 2022-10-12 (-25.4%)',
    note='Intramonth turning points are assigned to the month that dominated them; a monthly '
         'series cannot express a mid-month reversal: March 2000 (+9.7%) belongs to the bull '
         'that peaked inside it, October 2002 (+8.6%) to the bull that began inside it. The '
         'dot-com bear is dated as the published record dates it, one bear of -49.1%, although '
         'the S&P rose +21.4% close-to-close from 2001-09-21 to 2002-01-04: a strict +/-20% '
         'rule would split it in two, and this file follows the record rather than inventing a '
         'finer rule. Turning points re-derived 2026-09-23 from ^GSPC closes.',
    segments=(
        Segment('bull_1990', '2000-01-01', '2000-03-31',
                'Bull: trough 1990-10-11 -> peak 2000-03-24; the era covers its last 3 months',
                False),
        Segment('bear_dotcom', '2000-04-01', '2002-09-30',
                'Bear: peak 2000-03-24 -> trough 2002-10-09 (-49.1%, the dot-com bust; bonds '
                'rallied through it)', True),
        Segment('bull_2002', '2002-10-01', '2007-10-31',
                'Bull: trough 2002-10-09 -> peak 2007-10-09 (+101%)', False),
        Segment('bear_gfc', '2007-11-01', '2009-02-28',
                'Bear: peak 2007-10-09 -> trough 2009-03-09 (-56.8%)', True),
        Segment('bull_2009', '2009-03-01', '2020-01-31',
                'Bull: trough 2009-03-09 -> peak 2020-02-19 (longest on record)', False),
        Segment('bear_covid', '2020-02-01', '2020-03-31',
                'Bear: 2020-02-19 -> 2020-03-23 (-33.9%, fastest 30% fall on record)', True),
        Segment('bull_2020', '2020-04-01', '2021-12-31',
                'Bull: 2020-04 -> 2021-12 (stimulus recovery)', False),
        Segment('bear_2022', '2022-01-01', '2022-09-30',
                'Bear: 2022-01-03 -> 2022-10-12 (-25.4%, stocks and bonds fell together)',
                True),
        Segment('bull_2022', '2022-10-01', None,
                'Bull: 2022-10 ->', False),
    ),
)

# --------------------------------------------------------------------------------------- #
# 3. Monetary policy cycle — FOMC target federal funds rate
# --------------------------------------------------------------------------------------- #
# Boundaries are FOMC decisions, which are announced the day they happen: no hindsight is
# needed to date them, only to know which was the last of a sequence.
MONETARY = Segmentation(
    key='monetary',
    title='MONETARY POLICY CYCLE (FOMC target rate)',
    source='FOMC target federal funds rate decisions: six hikes 1999-06-30 to 2000-05-16 '
           '(4.75% -> 6.50%); thirteen cuts 2001-01-03 to 2003-06-25 (6.50% -> 1.00%: eleven '
           'in 2001, one each in 2002-11 and 2003-06); seventeen consecutive hikes 2004-06-30 '
           'to 2006-06-29 '
           '(1.00% -> 5.25%); first cut 2007-09-18; 0-0.25% from 2008-12-16; liftoff '
           '2015-12-16; last hike of that cycle 2018-12-19; three cuts Jul-Oct 2019; back to '
           '0-0.25% 2020-03-15; liftoff 2022-03-16 to 5.25-5.50% by 2023-07-26; first cut '
           '2024-09-18',
    note='No segment here is marked adverse. Whether tightening is bad for a trend-following '
         'strategy is precisely the question the panel is meant to answer, so it must not be '
         'assumed by the labelling.',
    segments=(
        Segment('hiking_1999', '2000-01-01', '2000-05-31',
                'The last three of six hikes (5.50% -> 6.50%, final hike 2000-05-16)', False),
        Segment('plateau_2000', '2000-06-01', '2000-12-31',
                'Held at 6.50% as the equity bear began', False),
        Segment('easing_2001', '2001-01-01', '2003-06-30',
                'Thirteen cuts (6.50% -> 1.00%), eleven of them in 2001', False),
        Segment('floor_2003', '2003-07-01', '2004-05-31',
                'Held at 1.00%', False),
        Segment('hiking_2004', '2004-06-01', '2006-06-30',
                'Seventeen consecutive hikes (1.00% -> 5.25%)', False),
        Segment('plateau_2006', '2006-07-01', '2007-08-31',
                'Held at 5.25% through the first credit tremors', False),
        Segment('crisis_easing', '2007-09-01', '2008-12-31',
                'Emergency easing to the zero bound (5.25% -> 0-0.25% in 15 months)', False),
        Segment('zirp_qe', '2009-01-01', '2015-11-30',
                'Zero bound + QE1/2/3 (0-0.25% held from 2008-12 to 2015-11)', False),
        Segment('hiking_2015', '2015-12-01', '2018-12-31',
                'Gradual tightening (0.25% -> 2.25-2.50%, nine hikes in three years)', False),
        Segment('easing_2019', '2019-01-01', '2020-02-29',
                'Pause, then three insurance cuts (Jul-Oct 2019)', False),
        Segment('pandemic_zirp', '2020-03-01', '2022-02-28',
                'Emergency return to the zero bound + unlimited QE', False),
        Segment('hiking_2022', '2022-03-01', '2023-07-31',
                'Fastest tightening since 1980 (0% -> 5.25-5.50% in 17 months)', False),
        Segment('plateau_2023', '2023-08-01', '2024-08-31',
                'Held at 5.25-5.50% ("higher for longer")', False),
        Segment('easing_2024', '2024-09-01', None,
                'Easing cycle (first cut 2024-09-18)', False),
    ),
)

# --------------------------------------------------------------------------------------- #
# 4. Inflation regime — BLS CPI-U, year-over-year, thresholds at 1% and 4%
# --------------------------------------------------------------------------------------- #
# The one table here that is hand-entered from a published series rather than computed from
# data this repository holds. KNOWN_GAPS.md records that, because it is the only layer a test
# cannot re-derive.
INFLATION = Segmentation(
    key='inflation',
    title='INFLATION REGIME (BLS CPI-U, year over year)',
    source='BLS CPI-U all items, NSA, 12-month change. Blocks cut at 1% and 4% on SUSTAINED '
           'readings, not single prints: between 1.07% (2002-06) and 3.76% (2000-03) for the '
           'whole of 2000-2004, so the Fed\'s 2002-2003 deflation scare never crossed 1%; '
           'negative 2009-03..2009-10 (trough -2.1%), at or above 4% from 2021-04 (4.2%) '
           'through 2023-05, peaking at 9.1% in 2022-06. The 2000-2004 range was re-derived '
           '2026-09-23 from FRED CPIAUCNS',
    note='Hand-entered from the published BLS series — the only segmentation here that no '
         'test in this repository can re-derive from its own data. Cutting on sustained '
         'readings rather than single prints is deliberate: the energy spikes of 2005 and '
         'mid-2008 crossed 4% for a few months each, and carving two-month segments around '
         'them would be fitting the ruler to the noise.',
    segments=(
        Segment('moderate_2000', '2000-01-01', '2008-06-30',
                'Moderate (YoY 1.1-3.8% through 2000-2004, mostly 2-4% after; brief energy '
                'spikes above 4% in 2005 and 2008)',
                False),
        Segment('deflation_scare', '2008-07-01', '2009-12-31',
                'Commodity spike unwinding into outright deflation (+5.6% -> -2.1% YoY)',
                True),
        Segment('below_target', '2010-01-01', '2021-03-31',
                'Persistently below target (YoY in the 1-3% band for eleven years)', False),
        Segment('inflation_surge', '2021-04-01', '2023-05-31',
                'Inflation surge (YoY at or above 4%, peak 9.1% in 2022-06)', True),
        Segment('disinflation', '2023-06-01', None,
                'Disinflation (YoY back below 4%)', False),
    ),
)

#: Every partition, in the order the report prints them. Coarse structure first.
SEGMENTATIONS = (BUSINESS_CYCLE, EQUITY_CYCLE, MONETARY, INFLATION)


# --------------------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------------------- #
def validate_partition(seg, era_start=COMMON_ERA_START):
    """Raise ValueError unless `seg` tiles the era with no gap, no overlap and no hole.

    Called at import for every segmentation in `SEGMENTATIONS`. A partition with a gap would
    silently drop months from the panel, and a partition with an overlap would count them
    twice — both would break the property the whole design rests on, and both are one typo
    away in a hand-written table.
    """
    if not seg.segments:
        raise ValueError(f'{seg.key}: no segments')
    first = pd.Timestamp(seg.segments[0].start)
    if first != pd.Timestamp(era_start):
        raise ValueError(f'{seg.key}: starts {first.date()}, era starts {era_start}')
    keys = [s.key for s in seg.segments]
    if len(set(keys)) != len(keys):
        raise ValueError(f'{seg.key}: duplicate segment keys in {keys}')
    for prev, nxt in zip(seg.segments, seg.segments[1:]):
        if prev.end is None:
            raise ValueError(f'{seg.key}: only the LAST segment may be open-ended '
                             f'(got an open {prev.key} before {nxt.key})')
        gap = pd.Timestamp(nxt.start) - pd.Timestamp(prev.end)
        if gap != pd.Timedelta(days=1):
            raise ValueError(
                f'{seg.key}: {prev.key} ends {prev.end} and {nxt.key} starts {nxt.start} — '
                f'segments must be consecutive to the day (gap {gap}). A partition that does '
                f'not tile the era is not a partition.')
    return True


for _s in SEGMENTATIONS:
    validate_partition(_s)
del _s


def resolved_segments(seg, era_end):
    """Segments with `end=None` closed at `era_end`, dropping any that start after it."""
    era_end = pd.Timestamp(era_end)
    out = []
    for s in seg.segments:
        start = pd.Timestamp(s.start)
        if start > era_end:
            continue
        end = pd.Timestamp(s.end) if s.end is not None else era_end
        out.append(s._replace(start=start, end=min(end, era_end)))
    return out


def common_window(earliest_by_name, floor=COMMON_ERA_START):
    """(start, binding_names) — the first date at which EVERY named strategy can be measured.

    The ranked table needs a single window or its rows are not comparable; that window is
    derived from the data (the latest inception among the strategies actually being compared)
    and floored at the era start. `binding_names` is a sorted TUPLE of every strategy
    attaining that latest date, empty when the floor binds.

    It is a tuple and not a name because ties are the normal case, not the exotic one: four
    leveraged G4 variants share UGL's 2008-12 inception and all four attain the maximum
    together. `max()` returned whichever one it happened to meet first, which made the
    report's remedial advice — "dropping X lengthens the window for everything else" —
    FALSE, since dropping any single one of four tied strategies changes nothing. It also
    made the blame depend on dict insertion order, so reordering the registry silently
    reassigned it. Returning the whole tied set is what makes the advice actionable.
    """
    floor = pd.Timestamp(floor)
    dated = {k: pd.Timestamp(v) for k, v in earliest_by_name.items() if v is not None}
    if not dated:
        return floor, ()
    start = max(dated.values())
    if start <= floor:
        return floor, ()
    return start, tuple(sorted(k for k, v in dated.items() if v == start))


# --------------------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------------------- #
def _stats(window):
    """(total return, worst drawdown inside the window, n months) for a monthly series.

    The leading 1.0 is what makes a drawdown that starts in the segment's FIRST month
    visible, exactly as in `metrics.wealth_curve`. Built with numpy rather than by
    concatenating two pandas objects: the old form produced a duplicate index label `0` and
    was correct only because pandas took a fast path when both operands shared an index.
    """
    w = np.concatenate([[1.0], (1.0 + window.to_numpy(dtype=float)).cumprod()])
    return (float(w[-1] - 1.0),
            float((w / np.maximum.accumulate(w) - 1.0).min()),
            int(len(window)))


def _annualised(window, rf=None):
    """The annualised metrics for a segment window, or `{}` when the segment is too short.

    Below `SEGMENT_MIN_MONTHS` this returns nothing at all rather than a number, which is the
    point: a CAGR over the 2-month `contraction_2020` is that return raised to the sixth power,
    and a Sortino over four observations is noise wearing a decimal point. The caller renders
    the absence as `n/a (n mo)` and says so in the caption.

    `rf` is the realised cash series; it is sliced to the window here so the ratios are netted
    against the rate that actually prevailed during the segment, not the whole-era average. A
    Sharpe for 2009 measured against a 2023 cash rate would be meaningless.
    """
    if len(window) < metrics_mod.SEGMENT_MIN_MONTHS:
        return {}
    rf_w = 0.0
    if rf is not None:
        rf_w = rf.reindex(window.index)
        if rf_w.isna().all():
            rf_w = 0.0
        else:
            rf_w = rf_w.fillna(0.0)
    m = metrics_mod.calculate_metrics(window, rf=rf_w)
    # ANNUALISED_KEYS, not RANK_KEYS: `max_dd` is in RANK_KEYS but needs no annualisation, and
    # `_stats` already supplies it at every length. Returning it here too would overwrite that
    # value on long segments only — one number from two code paths, differing silently if the
    # two drawdown implementations ever drift.
    return {k: m[k] for k in metrics_mod.ANNUALISED_KEYS if k in m}


def segment_cell(returns, first, start, end, rf=None):
    """One panel cell: measured, measured-but-partial, or an explicit `n/a`.

    `first` is the strategy's own first measurable month. A strategy that begins INSIDE a
    segment is reported over the part it can cover and flagged `partial` — printing `n/a` for
    a strategy that covers eleven of a segment's twelve months throws away a real
    observation, and printing an unflagged number pretends it saw the whole thing.

    `return`, `max_dd` and `n_months` are always present. The annualised metrics named in
    `metrics.RANK_KEYS` are present only when the covered span reaches
    `metrics.SEGMENT_MIN_MONTHS` — see `_annualised`. `max_dd` is computed by `_stats` at every
    length, because a drawdown needs no annualisation.
    """
    if returns is None or returns.empty:
        return {'na': 'no data'}
    first = pd.Timestamp(first) if first is not None else returns.index[0]
    # Compare MONTHS, not timestamps. A monthly return is dated at its month-END, so a
    # strategy whose first return is 2008-07-31 covers a segment starting 2008-07-01 in full;
    # comparing the raw timestamps would flag it partial, or worse, drop it as n/a.
    first_m, start_m, end_m = (pd.Period(x, freq='M') for x in (first, start, end))
    if first_m > end_m:
        return {'na': f'inception {first:%Y-%m}'}
    # Clamp to the strategy's own start as well as the segment's. `returns` normally begins
    # at `first` anyway, but a caller passing a longer series must not be able to credit a
    # strategy with months it could not trade.
    window = returns.loc[(returns.index >= max(start, first)) & (returns.index <= end)]
    if window.empty:
        return {'na': f'no data ({first:%Y-%m})'}
    total, dd, n = _stats(window)
    return {'return': total, 'max_dd': dd, 'n_months': n, 'partial': bool(first_m > start_m),
            **_annualised(window, rf)}


def partition_panel(metrics_data, seg, era_end, rf=None):
    """{strategy: {segment_key: cell}} for one segmentation, over each strategy's own history.

    Reads `returns_full` — the strategy's entire measurable history — never the series bounded
    by a requested window, which would make the panel an expression of date-selection bias
    instead of an antidote to it.

    When the segmentation marks any segment adverse, an extra `ADVERSE` entry compounds those
    segments' months together: what the strategy returned across every objectively bad month
    of the era, and nothing else.

    `rf` is the realised monthly cash series. When omitted it is taken from each entry's own
    `rf_series` — `run_backtest` attaches the one it used — so the ratios in a segment cell are
    netted against the same rate as the headline table. Passing neither leaves the annualised
    metrics computed against a zero cash rate, which flatters every Sharpe; callers that can
    supply it should.
    """
    segments = resolved_segments(seg, era_end)
    adverse_keys = [s.key for s in segments if s.adverse]
    panel = {}
    for d in metrics_data:
        returns = d.get('returns_full')
        if returns is None or returns.empty:
            returns = d.get('returns')
        if returns is None or returns.empty:
            continue
        rf_d = rf if rf is not None else d.get('rf_series')
        first = d.get('first_return_full') or d.get('first_return') or returns.index[0]
        row = {s.key: segment_cell(returns, first, s.start, s.end, rf_d) for s in segments}
        if adverse_keys:
            mask = pd.Series(False, index=returns.index)
            for s in segments:
                if s.adverse:
                    mask |= (returns.index >= s.start) & (returns.index <= s.end)
            window = returns[mask]
            if window.empty:
                row['ADVERSE'] = {'na': f'inception {pd.Timestamp(first):%Y-%m}'}
            else:
                total, dd, n = _stats(window)
                # ADVERSE compounds NON-CONTIGUOUS months, so its annualised metrics describe a
                # portfolio that only ever existed during crises. That is exactly the intended
                # reading — "what did this do across every bad month" — but the CAGR of a
                # stitched series is not a rate anyone earned over a calendar. Reported anyway,
                # with the caveat carried in the caption, because the alternative is to withhold
                # the drawdown and Sortino that make the bucket worth having.
                row['ADVERSE'] = {
                    'return': total, 'max_dd': dd, 'n_months': n,
                    'partial': any(row[k].get('partial') or 'na' in row[k]
                                   for k in adverse_keys),
                    **_annualised(window, rf_d)}
        panel[d['name']] = row
    return panel


#: What `segment_leaderboard` may be ranked by: the segment's own total return, plus every
#: metric the headline table offers. 'return' and 'max_dd' need no annualisation and are always
#: available; the rest require `SEGMENT_MIN_MONTHS` of coverage.
SEGMENT_RANK_KEYS = ('return',) + metrics_mod.RANK_KEYS

#: `segment_leaderboard`'s result. `rank_by` is the metric that ACTUALLY ordered the rows, which
#: is not always the one asked for — a segment shorter than `SEGMENT_MIN_MONTHS` has no
#: annualised metrics to sort on and falls back to total return. Callers must print this rather
#: than the requested key, or the caption will claim an ordering the table does not have.
Leaderboard = namedtuple('Leaderboard', 'ranked partial absent rank_by')


def segment_leaderboard(panel, segment_key, rank_by='return'):
    """Rank the strategies that cover ONE segment **in full**.

    Returns a `Leaderboard(ranked, partial, absent, rank_by)`:

    * ``ranked``  — ``[{'name', 'return', 'max_dd', 'n_months', ...}, ...]``, best first, and
      every entry measured over the *identical* span. Each dict also carries whichever of
      `metrics.RANK_KEYS` the segment was long enough to support;
    * ``partial`` — names that entered the segment late, with the months they did cover;
    * ``absent``  — names with no data in the segment at all, with the reason;
    * ``rank_by`` — the metric that actually ordered the rows. **Print this, not the argument.**

    **A partial coverer is not ranked, and that is the whole design.** The ranked table at
    the top of the report earns its meaning from one property: every row spans exactly the
    same months. A per-segment leaderboard that mixed a strategy covering 19 months of the
    GFC with one covering the last 4 would reproduce, inside each panel, precisely the
    date-selection bias the panels exist to remove — and it would do it in the flattering
    direction, since a strategy that arrives after the crash shows the recovery without the
    fall. So they are listed, with their coverage stated, and left out of the ordering.

    `rank_by` is any of `SEGMENT_RANK_KEYS`. **Annualised keys degrade rather than raise** on a
    segment below `SEGMENT_MIN_MONTHS`: the ordering falls back to 'return' and says so through
    the returned `rank_by`. Silence there would be the actual defect — a Sortino column full of
    `n/a` sorted by an invisible key looks like a ranking and is not one.

    An unknown key still raises, because that is a caller bug rather than a data limitation.
    """
    if rank_by not in SEGMENT_RANK_KEYS:
        raise ValueError(f'rank_by must be one of {SEGMENT_RANK_KEYS}, got {rank_by!r}')
    ranked, partial, absent = [], [], []
    for name, row in panel.items():
        cell = row.get(segment_key)
        if cell is None or 'na' in (cell or {}):
            absent.append((name, (cell or {}).get('na', 'no data')))
        elif cell.get('partial'):
            partial.append((name, cell['n_months'], cell['return']))
        else:
            keep = {'name': name, 'return': cell['return'], 'max_dd': cell['max_dd'],
                    'n_months': cell['n_months']}
            keep.update({k: cell[k] for k in metrics_mod.RANK_KEYS if k in cell})
            ranked.append(keep)

    # Degrade to total return when the requested metric is not present on every ranked row —
    # which happens exactly when the segment is too short to annualise. Mixing rows that have
    # a Sortino with rows that do not would sort on a missing key.
    effective = rank_by
    if ranked and any(rank_by not in d for d in ranked):
        effective = 'return'

    # `max_dd` is <= 0, so its best value is the LARGEST; only `vol` is ascending. One source
    # for that decision: metrics.rank_descending.
    ranked.sort(key=lambda d: d[effective],
                reverse=metrics_mod.rank_descending(effective))
    return Leaderboard(ranked, sorted(partial), sorted(absent), effective)


def era_bounds(metrics_data):
    """(first, last) measurable month across everything that was measured. (None, None) if empty."""
    firsts = [d.get('first_return_full') or d.get('first_return') for d in metrics_data]
    lasts = [d.get('last_return_full') or d.get('last_return') for d in metrics_data]
    firsts = [pd.Timestamp(x) for x in firsts if x is not None]
    lasts = [pd.Timestamp(x) for x in lasts if x is not None]
    if not firsts or not lasts:
        return None, None
    return min(firsts), max(lasts)
