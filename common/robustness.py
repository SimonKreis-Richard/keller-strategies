"""
Robustness — how much of a leaderboard survives contact with a different sample.

WHAT THIS MODULE DOES NOT DO, first, because the temptation is strong and the mistake is
expensive: it does not forecast, and it does not repair a biased sample. A bootstrap draws
from the distribution it is given. If twenty years of history handed US equity a ~10%/yr
drift, every resampled path carries a ~10%/yr drift, and ten thousand of them carry it ten
thousand times. Resampling widens the UNCERTAINTY around a mean; it cannot move the mean off
the sample it came from. Nothing here answers "what if the next twenty years are unkind" —
that question needs different DATA or a stated assumption, and neither is a simulation.

What it does answer is the question the sample CAN answer: given that the ranking was chosen
by looking at this history, how much of rank 1 is a property of the strategy and how much is
a property of having looked? Four measurements, from different directions:

1. **PBO — the probability of backtest overfitting** (Bailey, Borwein, Lopez de Prado & Zhu,
   *The Probability of Backtest Overfitting*, Journal of Computational Finance, 2017), by
   Combinatorially Symmetric Cross-Validation. Cut the common history into S disjoint blocks,
   enumerate every way of splitting them into equal in-sample and out-of-sample halves, pick
   the best strategy in-sample, and look up where it lands out-of-sample. PBO is the fraction
   of splits where the in-sample winner finishes at or below the OUT-OF-SAMPLE MEDIAN.

   Read it as: "if I choose the top of this table, how often is that choice no better than a
   coin flip on data I did not choose it from?" A PBO of 0.10 says the selection procedure
   mostly works. A PBO above 0.50 says the leaderboard is a ranking of noise.

   It is DETERMINISTIC — no RNG, no seed, nothing to tune but S. That is the single best
   property in this file, because it means the number cannot be shopped for.

2. **Rank stability under resampling** — the same leaderboard rebuilt on stationary-bootstrap
   resamples of the shared history, reporting how often each entry lands in the top k. This
   is the "all-terrain" question made countable: a strategy that is rank 1 on the observed
   path and top-3 in 38% of resamples is a different object from one that is rank 3 and
   top-3 in 71%.

   The resampling is JOINT — one set of time indices applied to every strategy at once. That
   is not an optimisation, it is the whole validity of the exercise. Resampling each entry
   independently would rank strategies that lived in different worlds, destroying exactly the
   cross-sectional correlation that makes one month good for HAA and bad for VAA. The block
   structure (Politis-Romano, shared with `margin_sizing`) preserves the serial dependence
   that momentum signals live on.

3. **The family view** — both measurements re-run at the level of the FAMILY (and of the
   de-risking MECHANISM), pooling evidence across variants. A family whose every variant
   scores well is harder to explain as luck than one lucky variant: the noise explanation
   has to win N times instead of once. If the family-level PBO comes out materially lower
   than the strategy-level one, the mechanism is robust and the variant choice is noise —
   which cashes out as "hold the family, stop agonising over the variant".

4. **The leverage frontier** — the same resampled histories, walked at constant margin
   leverage under the monthly-reset policy the ledger actually implements, reporting
   P(margin call), median CAGR and 5th-percentile CAGR per leverage level. It turns the
   sizing module's single recommendation into a curve, and evaluates P(call) AT the
   recommended level — two independent methods pointing at the same neighbourhood, or not.

THE LIMIT WORTH STATING OUT LOUD. PBO measures OUR selection among the entries in this
registry. It cannot see the selection that happened before the data reached us — Keller
choosing a 13-month lookback, a top-N, a breadth threshold, on the same decades. No
resampling of published results can recover that; only re-running the engine across a
parameter grid can, and that is a different (and more expensive) measurement than this one.
So: a low PBO here means "our ranking of these strategies is stable", NOT "these strategies
are not overfitted". The second claim needs a parameter sweep and this file does not make it.

No scipy — consistent with `common/selection.py`, everything here is numpy.
"""

import math
import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from common.margin_sizing import NotCalculable, stationary_bootstrap_indices
from common.metrics import LOWER_IS_BETTER, RANK_KEYS

#: Blocks for CSCV. Sixteen gives C(16,8) = 12 870 splits — enough that the PBO estimate is
#: stable to well under a percentage point, and small enough that every split is enumerated
#: rather than sampled, which is what keeps the result deterministic. The paper uses S = 16.
DEFAULT_BLOCKS = 16

#: Ceiling on S. C(20,10) = 184 756 is already 14x the default cost for no measurable gain in
#: precision, and C(24,12) is 2.7 million. Refused rather than silently downgraded.
MAX_BLOCKS = 20

#: Bootstrap paths and expected block length for `rank_bootstrap`. The block length matches
#: `MarginPolicy.block_months` deliberately: both are resampling the same monthly returns of
#: the same signals, and two different answers to "how long is a momentum regime" in one
#: repository would be one too many.
DEFAULT_PATHS = 2000
DEFAULT_BLOCK_MONTHS = 12

#: Shared with `leverage_advice.SEED`. Stated, never defaulted silently.
SEED = 20260731

#: A leaderboard of three cannot be cross-validated: the rank of the in-sample winner among
#: three entries takes three values, and the median it is compared against is one of them.
MIN_STRATEGIES = 5

#: Months per block, below which a block's Sharpe is an arithmetic artefact.
MIN_BLOCK_MONTHS = 6


# --------------------------------------------------------------------------------------- #
#  the shared rectangle
# --------------------------------------------------------------------------------------- #

def common_frame(metrics_data, trials_only=True, ranked_only=True):
    """(frame, binding_name) — every entry's monthly returns over the months they ALL share.

    Both measurements in this module are cross-sectional: they compare strategies to each
    other split by split, so they need a RECTANGLE — the same months for every column. That
    is a real cost and it is named rather than absorbed: the intersection is set by the
    latest-arriving entry, so `binding_name` comes back with the frame and every caller
    prints it, exactly as `_common_window_table` does in the dashboard.

    `trials_only` keeps the population identical to `selection.selection_trials` — the things
    you would actually choose between. Including a passive benchmark would make the ranking
    easier to win in-sample AND out-of-sample (it anchors the bottom in both halves), which
    biases PBO downwards: the winner beats a control it was never competing with.

    `ranked_only` restricts to the entries the ranked table itself ranks, and it is on by
    default because of what happens when it is off. Measured 2026-08-01, admitting every trial
    let `HAA_G4_Leveraged_2X` — whose UGL leg begins 2008-12 — set the intersection at 2010-02,
    and a cross-validation of a momentum leaderboard that contains no 2008 is cross-validating
    the wrong seventeen years. Dropping the late arrivals instead buys back the GFC for
    everyone else. It also makes this module answer a question about the leaderboard the
    dashboard actually shows, rather than about a different population that shares its name.
    """
    from common.selection import selection_trials
    rows = selection_trials(metrics_data) if trials_only else list(metrics_data)
    if ranked_only:
        in_window = [d for d in rows if d.get('in_ranked_window', True)]
        if len(in_window) >= MIN_STRATEGIES:
            rows = in_window
        # Below the floor the restriction is not applied, and the caller is told: the frame
        # then starts wherever the latest arrival starts, and `binding` names it.

    series = {}
    for d in rows:
        r = d.get('returns_full')
        if r is None or (hasattr(r, 'empty') and r.empty):
            r = d.get('returns')
        if r is not None and not r.empty:
            series[d['name']] = r
    if len(series) < MIN_STRATEGIES:
        raise NotCalculable(
            f'{len(series)} entries with returns; cross-validating a leaderboard needs at '
            f'least {MIN_STRATEGIES}')

    frame = pd.DataFrame(series).dropna(how='any')
    if frame.empty:
        raise NotCalculable('the selected entries share no month at all')

    # Which column forced the start. `idxmax` over first-valid-index across the UNCLIPPED
    # series, so the answer is "who arrived last", not "who happens to be first alphabetically".
    firsts = {name: s.first_valid_index() for name, s in series.items()}
    binding = max(firsts, key=lambda k: firsts[k]) if firsts else None
    return frame, binding


def realised_rf(metrics_data):
    """The scalar annual rf the ranked table's Sharpes were already netted against.

    Resolved exactly the way `leverage_advice.build_policy` resolves it — the first
    in-ranked-window row's `rf_annual`, falling back to the first row — so the three report
    sections (leaderboard, sizing, robustness) net against the SAME cash rate. Measured
    2026-08-01: leaving this at 0.0 while the leaderboard nets at the realised ~1.4%/yr
    flatters low-vol entries by up to +0.16 Sharpe relative to a 3x wrap's +0.04, which is
    enough to flip near-ties in `winner_share` — the ordering being cross-validated must be
    the ordering being shown.
    """
    rows = [d for d in metrics_data if d.get('in_ranked_window')]
    row = rows[0] if rows else (metrics_data[0] if metrics_data else None)
    if row is None or row.get('rf_annual') is None:
        return 0.0
    return float(row['rf_annual'])


def _excess(frame, rf_annual):
    """Monthly excess returns. A scalar annual rate, divided by twelve — deliberately not the
    realised rf series `metrics.build_rf_series` produces.

    Both measurements here are RANKINGS, and every column is netted against the same cash in
    the same month, so the realised path shifts every Sharpe by very nearly the same amount
    and changes the ordering hardly at all. Carrying the series through the CSCV sufficient
    statistics would double the bookkeeping to move a rank correlation in the third decimal.
    The scalar is an approximation, it is used only for ranking, and this is where it is
    admitted.
    """
    return frame - float(rf_annual) / 12.0


# --------------------------------------------------------------------------------------- #
#  1. PBO by combinatorially symmetric cross-validation
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PBOResult:
    pbo: float                      # P(in-sample winner lands at or below the OOS median)
    n_splits: int
    n_blocks: int
    block_months: int
    n_strategies: int
    n_months: int
    first: pd.Timestamp
    last: pd.Timestamp
    binding: str
    logits: np.ndarray              # one lambda per split
    winners: list                   # in-sample winner's NAME, one per split
    is_sharpe: np.ndarray           # winner's in-sample Sharpe, one per split
    oos_sharpe: np.ndarray          # the same winner's OUT-OF-SAMPLE Sharpe
    prob_oos_loss: float            # P(the chosen strategy loses money OOS, Sharpe < 0)
    dropped_months: int
    rf_annual: float = 0.0          # the scalar rate the Sharpes were netted against

    # NO `degradation_slope`. The OLS slope of `oos_sharpe` on `is_sharpe` is the obvious
    # companion statistic, it was implemented here on 2026-08-01, and it was removed the same
    # day because it does not measure what its name says. Measured over a sweep of injected
    # edges on ten synthetic strategies:
    #
    #     edge 0.000/mo  PBO 0.25  slope -0.56   winner holds 51% of splits
    #     edge 0.004/mo  PBO 0.07  slope -0.29   winner holds 83%
    #     edge 0.012/mo  PBO 0.00  slope -1.00   winner holds 100%
    #
    # Non-monotone, and pinned at exactly -1.00 in the case with the STRONGEST real edge. The
    # reason is arithmetic, not statistical: for a fixed strategy over a fixed sample the two
    # equal halves satisfy IS + OOS = 2 x full-sample mean, so once one strategy wins every
    # split the regression recovers that identity and nothing else. Captioning it "<= 0 means
    # in-sample excellence predicts nothing" would have inverted its meaning on exactly the
    # cases the reader cares about. `winner_share` and `prob_oos_loss` say what it was meant
    # to say, without the confound.

    @property
    def winner_share(self):
        """How the in-sample wins are distributed across entries, most frequent first.

        A leaderboard whose top row changes with the split is telling you something the PBO
        headline alone does not: not just "the winner does not persist" but "there is no
        winner". Both readings matter, so both are reported.
        """
        counts = {}
        for w in self.winners:
            counts[w] = counts.get(w, 0) + 1
        return sorted(((n, c / len(self.winners)) for n, c in counts.items()),
                      key=lambda kv: -kv[1])


def _sufficient_statistics(values, n_blocks):
    """(count, sum, sum of squares) per strategy per block, plus the months dropped.

    The trick that makes exhaustive CSCV cheap enough to be exhaustive. A Sharpe over the
    union of any set of blocks needs only these three totals, so a split costs a matrix
    multiply instead of a pass over the returns: 12 870 splits x 2 halves x N strategies of
    real work collapses into two `(N, S) @ (S, C)` products.

    The remainder is dropped from the FRONT — the earliest months go, the recent ones stay.
    Blocks must be equal-length or a split's two halves are not the same size and their
    Sharpes are not comparable.
    """
    n_months, n_strat = values.shape
    block = n_months // n_blocks
    dropped = n_months - block * n_blocks
    v = values[dropped:]                                    # (block*S, N)
    v = v.reshape(n_blocks, block, n_strat)
    return (block,
            v.sum(axis=1).T,                                # (N, S) sums
            (v ** 2).sum(axis=1).T,                         # (N, S) sums of squares
            dropped)


def _sharpe_from_totals(count, s1, s2):
    """Annualised Sharpe from block totals. `s1`/`s2` are (N, C); `count` is scalar.

    Sample standard deviation (ddof = 1), matching `metrics.calculate_metrics` — a population
    sd here and a sample sd there would make the two disagree about the same strategy on the
    same months, which is the kind of divergence this repository spends its tests preventing.
    """
    mean = s1 / count
    var = (s2 - count * mean ** 2) / (count - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sd > 0, mean * math.sqrt(12.0) / sd, np.nan)


def _ranks_descending(a):
    """1 = worst ... N = best, down axis 0. NaN ranks worst, never best.

    `np.argsort` puts NaN LAST, which in a descending-is-better ranking would silently hand a
    non-calculable Sharpe the top of the table. Replacing it with -inf first is one line and
    removes the possibility.
    """
    clean = np.where(np.isfinite(a), a, -np.inf)
    return np.argsort(np.argsort(clean, axis=0), axis=0) + 1


def _cscv_sharpes(frame, n_blocks, rf_annual):
    """The CSCV grid: annualised IS and OOS Sharpes for every strategy under every split.

    Factored out of `pbo` when the family-level view landed, because the family question —
    "pick the best FAMILY in-sample, where does the family land out-of-sample" — reads the
    same (N, C) matrices and aggregates them differently. Two enumerations of 12 870 splits
    that could drift apart would be worse than one.
    """
    n_blocks = int(n_blocks)
    if n_blocks % 2:
        raise NotCalculable(f'n_blocks must be even so the two halves match; got {n_blocks}')
    if not 4 <= n_blocks <= MAX_BLOCKS:
        raise NotCalculable(f'n_blocks must be between 4 and {MAX_BLOCKS}; got {n_blocks}')
    block = frame.shape[0] // n_blocks
    if block < MIN_BLOCK_MONTHS:
        raise NotCalculable(
            f'{frame.shape[0]} shared months over {n_blocks} blocks is {block} months per '
            f'block; below {MIN_BLOCK_MONTHS} a block Sharpe measures arithmetic, not the '
            f'strategy. Fewer blocks or a longer shared window.')

    values = _excess(frame, rf_annual).to_numpy(dtype=float)
    count, s1, s2, dropped = _sufficient_statistics(values, n_blocks)

    # Every split as a column of a 0/1 selector. (S, C), so `s1 @ M` is (N, C).
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    M = np.zeros((n_blocks, len(combos)), dtype=float)
    for j, c in enumerate(combos):
        M[list(c), j] = 1.0
    M_oos = 1.0 - M

    half = count * (n_blocks // 2)
    sr_is = _sharpe_from_totals(half, s1 @ M, s2 @ M)            # (N, C)
    sr_oos = _sharpe_from_totals(half, s1 @ M_oos, s2 @ M_oos)   # (N, C)
    return list(frame.columns), sr_is, sr_oos, len(combos), int(count), int(dropped)


def pbo(metrics_data=None, frame=None, binding=None, n_blocks=DEFAULT_BLOCKS, rf_annual=0.0):
    """Probability of backtest overfitting by CSCV. Deterministic: no seed, no sampling.

    Give it either `metrics_data` (it builds the shared rectangle itself) or a ready `frame`.

    The procedure, in the paper's terms and this code's variables:
      * cut the shared history into S equal blocks;
      * for each of the C(S, S/2) ways to choose HALF of them as in-sample, the complement is
        out-of-sample. Every split's mirror image is also enumerated, which is what the
        "combinatorially symmetric" in the name refers to and why the estimate needs no
        correction for which half came first;
      * n* = the strategy with the highest in-sample Sharpe;
      * omega = rank of n* out-of-sample, relative, in (0, 1);
      * lambda = logit(omega). PBO = P(lambda <= 0).

    Raises `NotCalculable` rather than returning a number it cannot stand behind.
    """
    if frame is None:
        if metrics_data is None:
            raise NotCalculable('pbo needs either metrics_data or a frame')
        frame, binding = common_frame(metrics_data)

    if frame.shape[1] < MIN_STRATEGIES:
        raise NotCalculable(f'{frame.shape[1]} strategies share this window; CSCV needs at '
                            f'least {MIN_STRATEGIES}')
    names, sr_is, sr_oos, n_splits, count, dropped = _cscv_sharpes(frame, n_blocks, rf_annual)

    best = np.nanargmax(np.where(np.isfinite(sr_is), sr_is, -np.inf), axis=0)   # (C,)
    cols = np.arange(n_splits)
    ranks = _ranks_descending(sr_oos)[best, cols]                 # rank of the winner, OOS

    n_strat = len(names)
    omega = ranks / (n_strat + 1.0)
    logits = np.log(omega / (1.0 - omega))

    is_win = sr_is[best, cols]
    oos_win = sr_oos[best, cols]
    ok = np.isfinite(oos_win)

    return PBOResult(
        pbo=float((logits <= 0).mean()),
        n_splits=n_splits, n_blocks=int(n_blocks), block_months=count,
        n_strategies=n_strat, n_months=int(frame.shape[0]),
        first=frame.index[0], last=frame.index[-1], binding=binding,
        logits=logits, winners=[names[i] for i in best],
        is_sharpe=is_win, oos_sharpe=oos_win,
        prob_oos_loss=float((oos_win[ok] < 0).mean()) if ok.any() else float('nan'),
        dropped_months=dropped, rf_annual=float(rf_annual))


def pbo_lines(res):
    """The PBO result as report lines. Shared by the CLI section and the dashboard panel."""
    verdict = ('the selection procedure carries real information'
               if res.pbo <= 0.25 else
               'the top of the table is weakly informative' if res.pbo <= 0.50 else
               'the ranking is indistinguishable from noise')
    lines = [
        '', 'PBO — probability of backtest overfitting (CSCV, deterministic)',
        f'  shared window                    : {res.first:%Y-%m}..{res.last:%Y-%m}'
        f'  ({res.n_months} months, set by {res.binding})',
        f'  {res.n_blocks} blocks of {res.block_months} months -> {res.n_splits} splits'
        f'   | {res.n_strategies} strategies   | Sharpes net of rf {res.rf_annual:.2%}/yr'
        + (f'   | {res.dropped_months} earliest month(s) dropped to make blocks equal'
           if res.dropped_months else ''),
        f'  PBO                              : {res.pbo:.1%}  <- {verdict}',
        f'  P(chosen strategy loses OOS)     : {res.prob_oos_loss:.1%}'
        '   (Sharpe below zero out of sample)',
    ]
    top = res.winner_share[:3]
    lines.append('  in-sample winner, by share of splits: '
                 + ', '.join(f'{n} {p:.0%}' for n, p in top))
    lines.append('  PBO answers "is OUR ranking of these entries stable". It cannot see the'
                 ' choices made before')
    lines.append('  the data reached us — a lookback or a top-N tuned on these same decades'
                 ' is invisible here.')
    return lines


# --------------------------------------------------------------------------------------- #
#  2. rank stability under joint resampling
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RankResult:
    rank_key: str
    rows: list                      # dicts, in observed-rank order
    n_paths: int
    block_months: int
    seed: int
    top_k: int
    n_months: int
    first: pd.Timestamp
    last: pd.Timestamp
    binding: str
    rf_annual: float = 0.0          # the scalar rate rf-relative rank keys are netted against


def _path_metrics(paths, rank_key, rf_annual):
    """`rank_key` for every resampled path of ONE strategy. `paths` is (n_paths, horizon).

    Vectorised over paths rather than looped, and computed the same way
    `metrics.calculate_metrics` computes it — including the leading 1.0 in the wealth curve,
    without which a drawdown starting in month one is invisible (audit finding #1 in
    `common/metrics.py`, and a resampler that reproduced the bug would report a shallower
    stress than the table beside it).
    """
    horizon = paths.shape[1]
    rf_m = float(rf_annual) / 12.0
    excess = paths - rf_m

    if rank_key == 'vol':
        return paths.std(axis=1, ddof=1) * math.sqrt(12.0)

    wealth = np.concatenate([np.ones((paths.shape[0], 1)),
                             np.cumprod(1.0 + paths, axis=1)], axis=1)
    if rank_key == 'cagr':
        return wealth[:, -1] ** (12.0 / horizon) - 1.0

    dd = wealth / np.maximum.accumulate(wealth, axis=1) - 1.0
    if rank_key == 'max_dd':
        return dd.min(axis=1)

    if rank_key == 'upi':
        cagr = wealth[:, -1] ** (12.0 / horizon) - 1.0
        ui = np.sqrt((dd ** 2).mean(axis=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(ui > 0, (cagr - float(rf_annual)) / ui, np.nan)

    if rank_key == 'sortino':
        downside = np.sqrt((np.minimum(0.0, excess) ** 2).mean(axis=1)) * math.sqrt(12.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(downside > 0, excess.mean(axis=1) * 12.0 / downside, np.nan)

    vol = paths.std(axis=1, ddof=1) * math.sqrt(12.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(vol > 0, excess.mean(axis=1) * 12.0 / vol, np.nan)


def _resampled_order(frame, rank_key, n_paths, block_months, seed, rf_annual):
    """names, per-path scores, per-path ranks, observed values, observed order, sign.

    Factored out of `rank_bootstrap` when the family view landed: the family table needs
    the same joint draw and the same per-path ranks, aggregated by group instead of by
    row — and the leverage frontier walks the same histories again, at leverage. One draw,
    three readers, no way for them to disagree about which alternative histories were run.
    """
    if rank_key not in RANK_KEYS:
        raise NotCalculable(f'rank_key must be one of {RANK_KEYS}; got {rank_key!r}')
    if seed is None:
        raise NotCalculable('seed is required — an unseeded bootstrap is not reproducible')
    if frame.shape[1] < 2:
        raise NotCalculable('a ranking needs at least two entries')
    if frame.shape[0] < block_months:
        raise NotCalculable(f'{frame.shape[0]} shared months is shorter than one '
                            f'{block_months}-month block')

    names = list(frame.columns)
    values = frame.to_numpy(dtype=float)                 # (T, N)
    horizon = frame.shape[0]
    rng = np.random.default_rng(int(seed))
    idx = stationary_bootstrap_indices(horizon, horizon, block_months, int(n_paths), rng)

    higher_is_better = rank_key not in LOWER_IS_BETTER
    scores = np.empty((len(names), int(n_paths)), dtype=float)
    for i in range(len(names)):
        scores[i] = _path_metrics(values[idx, i], rank_key, rf_annual)

    clean = np.where(np.isfinite(scores), scores, -np.inf if higher_is_better else np.inf)
    sign = -1.0 if higher_is_better else 1.0
    # rank 1 = best, whichever end of the metric that is. Ties break by POSITION rather than
    # averaging, which keeps `p_first` summing to exactly 1 and `p_top_k` to exactly k — the
    # two identities the caller's tests rest on. The cost is that two entries with literally
    # identical returns split the tie deterministically instead of sharing it; their SCORE
    # distributions still match to the last bit, which is the thing that would break if the
    # resampling were not joint.
    order = np.argsort(np.argsort(sign * clean, axis=0), axis=0) + 1

    observed = {}
    for i, name in enumerate(names):
        observed[name] = _path_metrics(values[:, i].reshape(1, -1), rank_key, rf_annual)[0]
    obs_order = sorted(names, key=lambda n: (sign * observed[n]
                                             if np.isfinite(observed[n]) else np.inf))
    return names, scores, order, observed, obs_order, sign


def rank_bootstrap(metrics_data=None, frame=None, binding=None, rank_key='sharpe',
                   n_paths=DEFAULT_PATHS, block_months=DEFAULT_BLOCK_MONTHS, seed=SEED,
                   top_k=3, rf_annual=0.0):
    """How often each entry reaches the top `top_k` across stationary-bootstrap resamples.

    ONE set of resampled time indices is drawn and applied to EVERY strategy, so each path is
    a coherent alternative history in which all of them lived through the same months in the
    same order. That is what makes the resulting ranking meaningful; drawing per strategy
    would compare entries from different worlds and quietly inflate the dispersion.

    The observed rank is reported next to the resampled distribution on purpose. The gap
    between "rank 1" and "top-3 in 38% of resamples" is the entire point of the measurement.
    """
    if frame is None:
        if metrics_data is None:
            raise NotCalculable('rank_bootstrap needs either metrics_data or a frame')
        frame, binding = common_frame(metrics_data)

    names, scores, order, observed, obs_order, sign = _resampled_order(
        frame, rank_key, n_paths, block_months, seed, rf_annual)

    rows = []
    for pos, name in enumerate(obs_order, start=1):
        i = names.index(name)
        rows.append({
            'name': name,
            'observed': float(observed[name]),
            'observed_rank': pos,
            'p_top_k': float((order[i] <= top_k).mean()),
            'p_first': float((order[i] == 1).mean()),
            'median_rank': float(np.median(order[i])),
            'p5': float(np.quantile(scores[i][np.isfinite(scores[i])], 0.05))
                  if np.isfinite(scores[i]).any() else float('nan'),
            'p95': float(np.quantile(scores[i][np.isfinite(scores[i])], 0.95))
                   if np.isfinite(scores[i]).any() else float('nan'),
        })

    return RankResult(rank_key=rank_key, rows=rows, n_paths=int(n_paths),
                      block_months=int(block_months), seed=int(seed), top_k=int(top_k),
                      n_months=int(frame.shape[0]), first=frame.index[0],
                      last=frame.index[-1], binding=binding, rf_annual=float(rf_annual))


def rank_lines(res, limit=None):
    """The resampled ranking as report lines, observed order preserved."""
    fmt = ('{:.2%}'.format if res.rank_key in ('cagr', 'vol', 'max_dd')
           else '{:.2f}'.format)
    head = (f'{"strategy":<26} {"rank":>4} {res.rank_key:>8} {"p5":>8} {"p95":>8} '
            f'{"med.rank":>8} {"P(top" + str(res.top_k) + ")":>9} {"P(1st)":>7}')
    lines = [
        '', f'RANK STABILITY UNDER RESAMPLING — {res.n_paths} joint stationary-bootstrap '
            f'paths, {res.rank_key.upper()}',
        f'  {res.first:%Y-%m}..{res.last:%Y-%m} ({res.n_months} shared months, set by '
        f'{res.binding}), block {res.block_months}mo, seed {res.seed}, '
        f'rf {res.rf_annual:.2%}/yr',
        '', head, '-' * len(head),
    ]
    for r in (res.rows[:limit] if limit else res.rows):
        lines.append(
            f'{r["name"]:<26} {r["observed_rank"]:>4} {fmt(r["observed"]):>8} '
            f'{fmt(r["p5"]):>8} {fmt(r["p95"]):>8} {r["median_rank"]:>8.1f} '
            f'{r["p_top_k"]:>9.1%} {r["p_first"]:>7.1%}')
    lines += [
        '',
        '  One set of resampled months, applied to every row at once — so each path is a',
        '  coherent alternative history rather than N unrelated ones. P(top) is the'
        ' all-terrain number:',
        '  a rank-3 entry that is top-3 in most resamples is a steadier choice than a rank-1'
        ' entry that is not.',
        '  It resamples THIS history. It does not simulate a world where these twenty years'
        ' went differently.',
    ]
    return lines


# --------------------------------------------------------------------------------------- #
#  3. the family view — pooling evidence across variants
# --------------------------------------------------------------------------------------- #

#: What actually decides DE-RISKING, per family — read off the strategy files, not off the
#: papers' abstracts. `mechanism_of` keys into this through `family_of`; an unknown family
#: falls back to its own name, so a new family forms its own group instead of crashing.
#:
#: The grouping exists because "which MECHANISM is solid" is a different question from
#: "which variant won", and the second has been measured (strategy-level PBO) to have an
#: unstable answer. A family whose every variant scores well is harder to explain as luck
#: than one lucky variant: the noise explanation has to win N times instead of once.
FAMILY_MECHANISM = {
    'HAA':  'canary: exogenous (TIP)',       # strategies/haa.py — a single TIP canary
    'DAA':  'canary: dedicated basket',      # strategies/daa.py — VWO+BND, CF = b/B
    'BAA':  'canary: dedicated basket',      # strategies/baa.py — SPY/VWO/VEA/BND, 13612W
    'VAA':  'breadth of own universe',       # strategies/vaa.py — bad-asset count
    'PAA2': 'breadth of own universe',       # strategies/paa.py — protection factor, a=2
    'GEM':  'absolute momentum vs cash',     # strategies/gem.py — no canary at all
    'DM':   'absolute momentum vs cash',     # strategies/gem.py — four modules of the same
    'GTAA': 'own-price trend (SMA10)',       # strategies/gtaa.py — per-asset filter
    # Passive families never reach a grouped table today (`common_frame` is trials-only);
    # these entries exist so `mechanism_of` covers the whole registry rather than most of it.
    'SPY': 'passive (no protection)',
    'Sixty': 'passive (no protection)',
    'Golden': 'passive (no protection)',
    'RiskParity': 'passive (no protection)',
}

#: Fewer groups than this and "where did the winner rank among the groups" has no room to
#: be answered: with three groups the OOS rank takes three values and the median is one of
#: them.
MIN_GROUPS = 4


def family_of(name):
    """`HAA_G12` -> `HAA`, `GEM_G2_Classic` -> `GEM`, `PAA2_G12` -> `PAA2`."""
    return str(name).split('_', 1)[0]


def mechanism_of(name):
    """The de-risking mechanism, so DAA and BAA (both dedicated-canary) pool their
    evidence. An unknown family groups under its own name rather than raising."""
    return FAMILY_MECHANISM.get(family_of(name), family_of(name))


@dataclass(frozen=True)
class GroupPBOResult:
    pbo: float
    grouping: str                   # 'family' or 'mechanism' — whatever the caller grouped by
    n_splits: int
    n_blocks: int
    block_months: int
    n_groups: int
    members: dict                   # {group label: tuple of member names}
    n_months: int
    first: pd.Timestamp
    last: pd.Timestamp
    binding: str
    logits: np.ndarray
    winners: list                   # in-sample winning GROUP, one per split
    dropped_months: int

    @property
    def winner_share(self):
        counts = {}
        for w in self.winners:
            counts[w] = counts.get(w, 0) + 1
        return sorted(((n, c / len(self.winners)) for n, c in counts.items()),
                      key=lambda kv: -kv[1])


def group_pbo(metrics_data=None, frame=None, binding=None, groups=family_of,
              grouping='family', n_blocks=DEFAULT_BLOCKS, rf_annual=0.0):
    """PBO at the GROUP level: pick the best group in-sample, look the GROUP up out-of-sample.

    A group's score in a half is the MEDIAN of its members' Sharpes over that half — the
    median rather than the best member, because "this family is good" is a claim about the
    mechanism, and cherry-picking the best member per split would smuggle back the exact
    variant-level selection this measurement exists to remove.

    Read beside the strategy-level PBO; the comparison is the point. Materially lower here
    means the MECHANISM is robust and the variant choice is noise — which cashes out as
    "hold the family, stop agonising over the variant". Similar or higher means the family
    label carries no information beyond its best variant's luck.
    """
    if frame is None:
        if metrics_data is None:
            raise NotCalculable('group_pbo needs either metrics_data or a frame')
        frame, binding = common_frame(metrics_data)

    members = {}
    for name in frame.columns:
        members.setdefault(groups(name), []).append(name)
    if len(members) < MIN_GROUPS:
        raise NotCalculable(f'{len(members)} {grouping} groups share this window; a grouped '
                            f'cross-validation needs at least {MIN_GROUPS}')

    names, sr_is, sr_oos, n_splits, count, dropped = _cscv_sharpes(frame, n_blocks, rf_annual)
    pos = {n: i for i, n in enumerate(names)}
    labels = sorted(members)
    with warnings.catch_warnings():
        # A group whose every member is non-calculable in some split medians to NaN with a
        # RuntimeWarning; `_ranks_descending` already ranks NaN worst, which is the intent.
        warnings.simplefilter('ignore', RuntimeWarning)
        g_is = np.vstack([np.nanmedian(sr_is[[pos[n] for n in members[g]]], axis=0)
                          for g in labels])
        g_oos = np.vstack([np.nanmedian(sr_oos[[pos[n] for n in members[g]]], axis=0)
                           for g in labels])

    best = np.nanargmax(np.where(np.isfinite(g_is), g_is, -np.inf), axis=0)
    cols = np.arange(n_splits)
    ranks = _ranks_descending(g_oos)[best, cols]
    omega = ranks / (len(labels) + 1.0)
    logits = np.log(omega / (1.0 - omega))

    return GroupPBOResult(
        pbo=float((logits <= 0).mean()), grouping=grouping,
        n_splits=n_splits, n_blocks=int(n_blocks), block_months=count,
        n_groups=len(labels), members={g: tuple(members[g]) for g in labels},
        n_months=int(frame.shape[0]), first=frame.index[0], last=frame.index[-1],
        binding=binding, logits=logits, winners=[labels[i] for i in best],
        dropped_months=dropped)


def group_pbo_lines(res):
    """The grouped PBO as report lines."""
    sizes = ', '.join(f'{g} ({len(m)})' for g, m in res.members.items())
    return [
        '', f'{res.grouping.upper()}-LEVEL PBO — pick the best {res.grouping} in-sample, '
            f'look the {res.grouping.upper()} up out-of-sample',
        f'  {res.n_groups} groups: {sizes}',
        f'  PBO ({res.grouping} level)         : {res.pbo:.1%}'
        f'   over the same {res.n_splits} splits as above',
        '  in-sample winner, by share of splits: '
        + ', '.join(f'{n} {p:.0%}' for n, p in res.winner_share[:3]),
        f'  Group score = MEDIAN of members per half; the best member per split would '
        f'smuggle back the variant-level',
        f'  selection this removes. Read against the strategy-level PBO: materially lower '
        f'means the {res.grouping} is',
        '  robust and the variant choice is noise.',
    ]


@dataclass(frozen=True)
class GroupRankResult:
    rank_key: str
    grouping: str
    rows: list
    n_paths: int
    block_months: int
    seed: int
    top_k: int
    n_months: int
    first: pd.Timestamp
    last: pd.Timestamp
    binding: str


def group_rank_bootstrap(metrics_data=None, frame=None, binding=None, groups=family_of,
                         grouping='family', rank_key='sharpe', n_paths=DEFAULT_PATHS,
                         block_months=DEFAULT_BLOCK_MONTHS, seed=SEED, top_k=3,
                         rf_annual=0.0):
    """How often ANY member of each group reaches the top `top_k`, across the same joint
    resamples the strategy-level table uses — same seed, same draw, by construction.

    "Any member" is the right reduction for the question this feeds: the practical decision
    is "hold the family's best variant", and the family delivers whenever any variant does.
    It also makes the group probabilities exact: groups partition the table, so P(1st) sums
    to one across groups, which the tests assert rather than assume.
    """
    if frame is None:
        if metrics_data is None:
            raise NotCalculable('group_rank_bootstrap needs either metrics_data or a frame')
        frame, binding = common_frame(metrics_data)

    names, scores, order, observed, obs_order, sign = _resampled_order(
        frame, rank_key, n_paths, block_months, seed, rf_annual)

    members = {}
    for i, name in enumerate(names):
        members.setdefault(groups(name), []).append(i)
    obs_rank = {n: p for p, n in enumerate(obs_order, start=1)}

    rows = []
    for label in sorted(members):
        idx = members[label]
        best = order[idx].min(axis=0)                 # best member's rank, one per path
        best_obs, best_member = min((obs_rank[names[i]], names[i]) for i in idx)
        rows.append({
            'group': label, 'n_members': len(idx), 'best_member': best_member,
            'observed_best_rank': best_obs,
            'median_best_rank': float(np.median(best)),
            'p_top_k': float((best <= top_k).mean()),
            'p_first': float((best == 1).mean()),
        })
    rows.sort(key=lambda r: r['observed_best_rank'])
    return GroupRankResult(rank_key=rank_key, grouping=grouping, rows=rows,
                           n_paths=int(n_paths), block_months=int(block_months),
                           seed=int(seed), top_k=int(top_k), n_months=int(frame.shape[0]),
                           first=frame.index[0], last=frame.index[-1], binding=binding)


def group_rank_lines(res):
    """The grouped resampled ranking as report lines, observed order preserved."""
    head = (f'{res.grouping:<26} {"members":>7} {"best member":<26} {"obs":>4} '
            f'{"med.best":>8} {"P(top" + str(res.top_k) + ")":>9} {"P(1st)":>7}')
    lines = [
        '', f'{res.grouping.upper()} RANK STABILITY — any member in the top {res.top_k}, '
            f'same {res.n_paths} joint resamples, {res.rank_key.upper()}',
        '', head, '-' * len(head),
    ]
    for r in res.rows:
        lines.append(
            f'{r["group"]:<26} {r["n_members"]:>7} {r["best_member"]:<26} '
            f'{r["observed_best_rank"]:>4} {r["median_best_rank"]:>8.1f} '
            f'{r["p_top_k"]:>9.1%} {r["p_first"]:>7.1%}')
    lines += [
        '',
        f'  "Any member": the {res.grouping} claims a slot whenever one of its variants '
        'lands there, because the',
        '  decision this feeds is "hold the family\'s best variant". P(1st) sums to 100% — '
        'groups partition the table.',
    ]
    return lines


# --------------------------------------------------------------------------------------- #
#  4. the leverage frontier — the same histories, walked at leverage
# --------------------------------------------------------------------------------------- #

#: The leverage grid: 1.00x to 3.00x by 0.05. The unlevered anchor MUST be the first point —
#: it is the one level whose margin-call probability is zero by construction, which is what
#: makes the `f@1%` / `f@5%` maxima well-defined. Beyond 3x nothing in this repository has
#: any business being.
FRONTIER_F_GRID = tuple(round(1.0 + 0.05 * i, 2) for i in range(41))

#: The two headline thresholds: the highest f whose margin-call probability across the
#: resampled histories stays at or below each.
FRONTIER_P_STRICT = 0.01
FRONTIER_P_LOOSE = 0.05


def _frontier_from_paths(paths, leverage, maintenance_margin, borrow_rate_annual):
    """(P(margin call), CAGR per path) at one constant leverage, monthly-reset margin.

    Closed form rather than a loop, and the closure is exact for the policy the ledger
    actually implements (`reset_monthly`): with the book reset to `f x equity` every month,
    equity compounds by ``g = 1 + f*r - (f-1)*c`` each month, and the maintenance test
    ``equity < m x assets`` reduces to a per-month condition on the return alone:

        r  <  r_crit  =  (f-1)(1+c) / (f(1-m)) - 1

    so a margin call happens on a path iff any month's return is below `r_crit` — an
    elementwise comparison across every path at once. `tests/test_robustness.py` pins this
    against `margin_sizing.simulate_margin_path` walking the same paths step by step, so
    the closed form and the reference walker cannot drift apart.

    A month with g <= 0 (equity wiped) always satisfies the call condition, so surviving
    paths have strictly positive growth factors and the cumulative product is safe. On a
    called path the holder keeps the equity remaining at the forced close —
    ``f(1+r) - (f-1)(1+c)`` of the month-open equity, floored at zero — and sits in cash
    for the remainder: ruin is counted, not censored. The cash earns NOTHING — flat, not
    rf (AUD-05). Accruing rf/12 on the residual would slightly raise called-path CAGRs;
    holding flat is the conservative convention, and it only touches paths that were
    called at all (0% of paths at every recommended level, measured on the real run).

    MONTH-END ONLY. A real maintenance test is continuous and the intra-month trough is
    invisible to monthly paths, so every probability from this function UNDERSTATES. CAP 1
    in `margin_sizing` carries the measured intraperiod factor for exactly that reason;
    this exists to check it from an independent direction, not to replace it.
    """
    r = np.asarray(paths, dtype=float)
    n_paths, horizon = r.shape
    f = float(leverage)
    m = float(maintenance_margin)
    c = float(borrow_rate_annual) / 12.0
    if f < 1.0:
        raise NotCalculable(f'leverage below 1.0x is de-leveraging, not margin; got {f}')
    if not 0.0 <= m < 1.0:
        raise NotCalculable(f'maintenance margin must be in [0, 1); got {m}')

    def _cagr(wealth):
        out = np.full(len(wealth), -1.0)
        pos = wealth > 0
        out[pos] = wealth[pos] ** (12.0 / horizon) - 1.0
        return out

    if f == 1.0:
        return 0.0, _cagr(np.prod(1.0 + r, axis=1))

    r_crit = (f - 1.0) * (1.0 + c) / (f * (1.0 - m)) - 1.0
    called = r < r_crit
    any_call = called.any(axis=1)

    g = 1.0 + f * r - (f - 1.0) * c
    cum = np.cumprod(g, axis=1)
    wealth = cum[:, -1].copy()
    if any_call.any():
        rows = np.flatnonzero(any_call)
        t = np.argmax(called[rows], axis=1)
        prefix = np.where(t > 0, cum[rows, np.maximum(t - 1, 0)], 1.0)
        kept = f * (1.0 + r[rows, t]) - (f - 1.0) * (1.0 + c)
        wealth[rows] = prefix * np.maximum(kept, 0.0)
    return float(any_call.mean()), _cagr(wealth)


@dataclass(frozen=True)
class FrontierRow:
    name: str
    maintenance: float
    recommended: float              # the sizing module's advice; None when non-calculable
    p_call_at_recommended: float    # nan when there is no advice to check
    f_strict: float                 # highest grid f with P(call) <= FRONTIER_P_STRICT
    f_loose: float                  # highest grid f with P(call) <= FRONTIER_P_LOOSE
    kelly_f: float                  # argmax of median CAGR over the grid
    kelly_censored: bool            # the peak sits on the grid's last point — truth is beyond
    cagr_1x: float                  # median CAGR at 1.0x
    cagr_at_kelly: float
    p5_at_kelly: float
    curve: dict                     # {f: (p_call, median_cagr, p5_cagr)} — the chart's data


@dataclass(frozen=True)
class FrontierResult:
    rows: list
    skipped: tuple
    f_grid: tuple
    n_paths: int
    block_months: int
    seed: int
    borrow_rate_annual: float
    n_months: int
    first: pd.Timestamp
    last: pd.Timestamp
    binding: str


def leverage_frontier(metrics_data=None, frame=None, binding=None, maintenance=None,
                      recommended=None, borrow_rate_annual=0.06, f_grid=FRONTIER_F_GRID,
                      n_paths=DEFAULT_PATHS, block_months=DEFAULT_BLOCK_MONTHS, seed=SEED):
    """The margin decision as a CURVE instead of a verdict.

    `maintenance` maps name -> the maintenance requirement CAP 1 used
    (`MarginRecommendation.maintenance_margin_used`), so the frontier and the closed-form
    cap can disagree about nothing but method. `recommended` maps name -> the advice the
    Max margin column prints; the frontier evaluates P(call) AT that exact leverage, which
    is the cross-check — one method from a bootstrapped drawdown quantile and a closed
    form, one from walking resampled histories month by month, pointing at the same
    neighbourhood. Or not, in which case one of them is wrong and that is worth knowing.

    Same seed and the same block draw as `rank_bootstrap`: these are the SAME alternative
    histories the ranking was tested on, so the ranking and the frontier cannot be
    answering questions about two different worlds.
    """
    if maintenance is None:
        raise NotCalculable('leverage_frontier needs a maintenance mapping — take it from '
                            'the sizing recommendations, so the two methods share one m')
    if seed is None:
        raise NotCalculable('seed is required — an unseeded bootstrap is not reproducible')
    if not f_grid or float(f_grid[0]) != 1.0:
        raise NotCalculable('the leverage grid must start at 1.0x — the unlevered anchor '
                            'is the one point whose call probability is zero by construction')
    if frame is None:
        if metrics_data is None:
            raise NotCalculable('leverage_frontier needs either metrics_data or a frame')
        frame, binding = common_frame(metrics_data)
    if frame.shape[0] < block_months:
        raise NotCalculable(f'{frame.shape[0]} shared months is shorter than one '
                            f'{block_months}-month block')

    usable, skipped = [], []
    for name in frame.columns:
        m = maintenance.get(name)
        if m is None or not np.isfinite(m):
            skipped.append((name, 'no maintenance requirement on its sizing record'))
        elif not 0.0 <= float(m) < 1.0:
            skipped.append((name, f'maintenance {float(m):.2f} outside [0, 1)'))
        else:
            usable.append(name)
    if not usable:
        raise NotCalculable('no entry carries a usable maintenance requirement')

    names = list(frame.columns)
    values = frame.to_numpy(dtype=float)
    horizon = frame.shape[0]
    rng = np.random.default_rng(int(seed))
    idx = stationary_bootstrap_indices(horizon, horizon, block_months, int(n_paths), rng)

    recommended = recommended or {}
    rows = []
    for name in usable:
        paths = values[idx, names.index(name)]
        m = float(maintenance[name])
        curve = {}
        for f in f_grid:
            p_call, cagr = _frontier_from_paths(paths, f, m, borrow_rate_annual)
            curve[f] = (p_call, float(np.median(cagr)), float(np.quantile(cagr, 0.05)))
        f_strict = max(f for f in f_grid if curve[f][0] <= FRONTIER_P_STRICT)
        f_loose = max(f for f in f_grid if curve[f][0] <= FRONTIER_P_LOOSE)
        kelly = max(f_grid, key=lambda f: curve[f][1])
        rec = recommended.get(name)
        if rec is None:
            p_at = float('nan')
        elif float(rec) <= 1.0:
            p_at = 0.0
        else:
            p_at, _ = _frontier_from_paths(paths, float(rec), m, borrow_rate_annual)
        rows.append(FrontierRow(
            name=name, maintenance=m, recommended=rec, p_call_at_recommended=p_at,
            f_strict=float(f_strict), f_loose=float(f_loose), kelly_f=float(kelly),
            kelly_censored=(kelly == f_grid[-1]),
            cagr_1x=curve[f_grid[0]][1], cagr_at_kelly=curve[kelly][1],
            p5_at_kelly=curve[kelly][2], curve=curve))
    rows.sort(key=lambda r: (-r.f_strict, r.name))
    return FrontierResult(rows=rows, skipped=tuple(skipped), f_grid=tuple(f_grid),
                          n_paths=int(n_paths), block_months=int(block_months),
                          seed=int(seed), borrow_rate_annual=float(borrow_rate_annual),
                          n_months=horizon, first=frame.index[0], last=frame.index[-1],
                          binding=binding)


def frontier_lines(res, limit=None):
    """The frontier as report lines, most margin-tolerant entry first."""
    head = (f'{"strategy":<26} {"m":>5} {"advice":>7} {"P@adv":>7} {"f@1%":>6} {"f@5%":>6} '
            f'{"Kelly":>6} {"CAGR@1x":>8} {"CAGR@K":>8} {"p5@K":>8}')
    lines = [
        '', f'LEVERAGE FRONTIER — the same {res.n_paths} resampled histories, walked at '
            'constant margin leverage',
        f'  {res.first:%Y-%m}..{res.last:%Y-%m} ({res.n_months} shared months, set by '
        f'{res.binding}), borrow {res.borrow_rate_annual:.1%}/yr, block '
        f'{res.block_months}mo, seed {res.seed}',
        '', head, '-' * len(head),
    ]
    for r in (res.rows[:limit] if limit else res.rows):
        rec = 'n/c' if r.recommended is None else f'{r.recommended:.2f}x'
        p_at = ('     —' if not np.isfinite(r.p_call_at_recommended)
                else f'{r.p_call_at_recommended:>6.1%}')
        kelly = f'>{res.f_grid[-1]:.2f}' if r.kelly_censored else f'{r.kelly_f:.2f}x'
        lines.append(
            f'{r.name:<26} {r.maintenance:>5.2f} {rec:>7} {p_at:>7} '
            f'{r.f_strict:>5.2f}x {r.f_loose:>5.2f}x {kelly:>6} '
            f'{r.cagr_1x:>8.2%} {r.cagr_at_kelly:>8.2%} {r.p5_at_kelly:>8.2%}')
    if res.skipped:
        lines.append('  not walked: ' + '; '.join(f'{n} ({why})' for n, why in res.skipped))
    lines += [
        '',
        '  f@1% / f@5%  the highest constant leverage whose margin-call probability across '
        'the resampled histories',
        '               stays at or below 1% / 5%. MONTH-END paths: a real call is '
        'CONTINUOUS, so every probability',
        '               here understates — and the advice column carries the measured '
        'intraperiod factor these paths',
        '               cannot see. The two methods share m and the borrow rate and nothing '
        'else; their agreement is',
        '               the cross-check.',
        '  Kelly        the f that maximises MEDIAN CAGR over the same histories, ruin '
        'included — the empirical peak',
        '               of the growth curve, read off resamples of THIS history. Not a '
        'forecast.',
        '  Constant f, fully drawn: the conservative envelope of MARGIN_FOLLOWS_SIGNAL, '
        'which de-levers with the',
        '  signal and therefore calls less often than this table. A called path keeps the '
        'residual equity in cash',
        '  earning NOTHING (flat, not rf) — conservative, and it only touches paths that '
        'were called.',
    ]
    return lines
