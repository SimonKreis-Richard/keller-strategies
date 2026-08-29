"""
Selection statistics: what a leaderboard's top row is worth.

Running 39 variants over one history and reporting the best one is a search, and a search
finds a maximum whether or not any skill is present. This module puts the number that
search alone would have produced next to the number that was produced, so the two can be
compared on the same screen.

Nothing here removes the ranked table. The owner uses it to see how each variant behaved in
each drawdown, which is a legitimate and valuable use. What it must never be read as is a
selection procedure — and `rank_stability` is the measurement that settles that question
empirically rather than by argument.

No scipy: it was dropped from the dependency list when CAA was deleted, and everything here
is a few lines of numpy.
"""

import math

import numpy as np
import pandas as pd


def norm_ppf(p, tol=1e-12):
    """Inverse standard normal CDF, by bisection on `math.erf`.

    Twenty lines instead of a scipy dependency, and exact to machine tolerance over the
    range this module uses. scipy was dropped when CAA was deleted and is not coming back
    for one quantile.
    """
    if not 0.0 < p < 1.0:
        return float('-inf') if p <= 0.0 else float('inf')
    lo, hi = -12.0, 12.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def expected_max_of_n_normals(n):
    """E[max] of n independent standard normals, via Blom's order-statistic approximation
    ``Phi^-1((n - 3/8) / (n + 1/4))``.

    About 2.13 for n = 39 and 2.34 for n = 64. The textbook asymptotic
    ``sqrt(2 ln n) - (ln ln n + ln 4pi)/(2 sqrt(2 ln n))`` is materially biased LOW at these
    sample sizes (1.98 for n = 37), which would understate the selection bar — the direction
    that flatters the leaderboard, so it is the wrong error to make here.

    This is what "the best of N" is worth when N strategies with zero skill are ranked on a
    finite sample.
    """
    if n <= 1:
        return 0.0
    return norm_ppf((n - 0.375) / (n + 0.25))


def selection_trials(metrics_data):
    """The entries that count as SELECTION TRIALS, out of everything that was measured.

    Controls AND exploratory entries are excluded on purpose. The multiple-testing count must be
    the number of things you would actually CHOOSE between; a single-asset degenerate case you
    would never hold is not a trial you ran, and neither is a 3x variant whose worst case no
    data can show (no 3x product predates 2008-11). Counting either would inflate the null
    threshold in the direction that flatters rank 1.

    That filter is why the 3x expansion was safe: the registry grew by twelve entries on
    2026-07-29 and the trial count moved 18 -> 19.

    Lives here, and not inline in the report that first needed it, because there are now TWO
    consumers — the selection section of the CLI report and the Sharpe haircut in
    `common/leverage_advice.py` — and a second copy of this filter is a second answer to "how
    many trials was this?". `tests/test_selection.py` pins them to the same number.
    """
    return [d for d in metrics_data
            if d.get('is_active') and d.get('role', 'strategy') == 'strategy']


def trial_sharpe_spread(metrics_data):
    """(n_trials, cross-sectional sd of the trials' Sharpes), or (None, None).

    `sd` is the MEASUREMENT that turns the multiple-testing haircut in `margin_sizing` from a
    chosen coefficient into a derived one: the expected maximum Sharpe across N trials under the
    null is a multiple of exactly this spread. Two trials cannot give a spread worth the name,
    so below three the pair is (None, None) and the haircut is skipped and said to be skipped.
    """
    trials = selection_trials(metrics_data)
    sharpes = [float(d['sharpe']) for d in trials
               if d.get('sharpe') is not None and np.isfinite(d['sharpe'])]
    if len(sharpes) < 3:
        return None, None
    # len(sharpes), not len(trials): the count and the spread must describe the same
    # population, because `margin_sizing` feeds the pair into one E[max of N] formula. A
    # trial with a non-finite Sharpe contributed nothing to the spread, so counting it
    # would overstate N against a spread it never widened (AUD-04).
    return len(sharpes), float(np.std(sharpes, ddof=1))


def selection_context(values, n_effective=None):
    """Where the top of a leaderboard sits relative to what pure selection produces.

    `n_effective` — the number of INDEPENDENT bets, when it is known to be smaller than the
    nominal count. It usually is: the pre-cut 64-variant suite had a participation ratio of
    3.0. Using the nominal count assumes independence and overstates the penalty.
    """
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) < 2:
        return None
    n = len(vals)
    n_eff = n_effective if n_effective and n_effective >= 1 else n
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    z_nominal = expected_max_of_n_normals(n)
    z_effective = expected_max_of_n_normals(n_eff)
    return {
        'n': n,
        'n_effective': float(n_eff),
        'mean': mean,
        'sd': sd,
        'observed_best': float(vals.max()),
        'threshold_nominal': mean + z_nominal * sd,
        'threshold_effective': mean + z_effective * sd,
        'z_nominal': z_nominal,
        'z_effective': z_effective,
    }


def participation_ratio(returns_by_name):
    """Effective number of independent bets in a set of return series.

    PR = (sum of eigenvalues)^2 / sum(eigenvalues^2) of the correlation matrix. A suite of
    39 variants that all trade the same momentum signal on overlapping universes does NOT
    provide 39 independent trials, and pretending it does makes the multiple-testing penalty
    far harsher than reality.
    """
    frame = pd.DataFrame(returns_by_name).dropna(how='any')
    if frame.shape[1] < 2 or frame.shape[0] < frame.shape[1]:
        # Not enough common history to estimate a full correlation matrix honestly — so
        # refuse, rather than zero-fill the gaps. Zero-filling biases every correlation
        # toward 0, inflates the PR, and hands back a harsher effective-trials penalty
        # dressed as a measurement (AUD-03). The caller already handles None.
        return None
    corr = frame.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    eig = np.linalg.eigvalsh(corr)
    eig = eig[eig > 0]
    if len(eig) == 0:
        return None
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())
    var = eig[::-1] / eig.sum()
    cum = np.cumsum(var)
    return {
        'participation_ratio': pr,
        'pc1_share': float(var[0]),
        'n_pcs_for_90pct': int(np.searchsorted(cum, 0.90) + 1),
        'n_series': int(frame.shape[1]),
        'n_months': int(frame.shape[0]),
    }


def _rank(values):
    """Average-tie ranks, ascending. Plain numpy — no scipy."""
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average ties
    sorted_vals = np.asarray(values)[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a, b):
    """(rho, approximate two-sided p) between two equal-length sequences.

    The p-value uses the usual t = rho*sqrt((n-2)/(1-rho^2)) statistic with a normal
    approximation to the tail. It is indicative, not exact; with n around 39 it is close
    enough to tell "no relationship" from "a relationship", which is all it is used for.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        return float('nan'), float('nan')
    ra, rb = _rank(a), _rank(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return rho, p


def rank_stability(returns_by_name, windows, metric):
    """Spearman rank correlation of `metric` between disjoint sub-periods.

    `windows` is a list of (label, start, end). `metric` is a callable taking a return
    Series and returning a scalar. Returns (pairwise rhos, per-window top-5 lists).

    A rho near zero means the leaderboard has no predictive content — while remaining an
    excellent DESCRIPTION of what each regime rewarded. Both facts belong on the same screen.
    """
    scores, tops = {}, {}
    for label, start, end in windows:
        row = {}
        for name, series in returns_by_name.items():
            if series is None or series.empty:
                continue
            w = series.loc[(series.index >= pd.to_datetime(start))
                           & (series.index <= pd.to_datetime(end))]
            if len(w) < 12:
                continue
            row[name] = metric(w)
        if len(row) >= 3:
            scores[label] = row
            tops[label] = [k for k, _ in sorted(row.items(), key=lambda kv: -kv[1])[:5]]

    pairs = []
    labels = list(scores)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = scores[labels[i]], scores[labels[j]]
            common = sorted(set(a) & set(b))
            if len(common) < 5:
                continue
            rho, p = spearman([a[k] for k in common], [b[k] for k in common])
            pairs.append({'a': labels[i], 'b': labels[j], 'rho': rho, 'p': p,
                          'n': len(common)})
    return pairs, tops
