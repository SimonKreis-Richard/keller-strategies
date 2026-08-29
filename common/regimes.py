"""
Named historical regimes, and the publication dates that separate in-sample from out-of-sample.

A leaderboard answers "which variant won?", which §T5.3 shows has no predictive content
(rank correlation between disjoint four-year windows: rho ~ 0). A regime panel answers
"how did each variant BEHAVE when this specific thing happened?", which is a description
rather than a forecast — and descriptions do not decay.

The episode list below is **frozen**. It was written from the historical record, not tuned
to results, and it must not be adjusted because a strategy looks bad in one of the windows.
Adding an episode is a decision to be argued in a commit message.

The critical rendering rule: a cell whose episode predates the strategy's coverage must
print `n/a (inception YYYY-MM)`. Never a number, never a blank, and above all never 0%. Most
leveraged variants will legitimately show `n/a` for dot-com and the GFC, and **that visible
gap is the point** — it makes it impossible to believe a 2x/3x variant survived 2008. No 3x
product predates 2008-11 (UPRO 2009-06, TQQQ 2010-02); the 2x products that do exist mostly
start 2006-2007.
"""

import numpy as np
import pandas as pd

#: (key, first month, last month, description). Decision-to-decision month boundaries.
#:
#: `dotcom` (2000-09..2002-09) was REMOVED on 2026-07-31, and the reason is coverage rather
#: than opinion: `eras.COMMON_ERA_START` moved to 2004-11 that day — the first month a strategy
#: rather than a passive benchmark can be measured — so no entry in this repository has a single
#: month inside that window and the column was `n/a` from top to bottom. A column that cannot
#: be populated is not a stress test, it is a reminder that the funds did not exist, and the
#: era docstring already says so at length. Restoring it would require lowering the era floor
#: back below 2003, which would restore the eight one-row "leaderboards" that move removed.
EPISODES = [
    ('gfc',            '2007-11-01', '2009-03-31', 'Global financial crisis'),
    ('euro_downgrade', '2011-05-01', '2011-09-30', 'Euro crisis / US downgrade'),
    ('taper',          '2013-05-01', '2013-09-30', 'Taper tantrum'),
    ('china_oil',      '2015-06-01', '2016-02-29', 'China devaluation / oil crash'),
    ('q4_2018',        '2018-10-01', '2018-12-31', 'Q4 2018 selloff'),
    ('covid',          '2020-02-01', '2020-03-31', 'COVID crash'),
    ('bear_2022',      '2022-01-01', '2022-10-31', 'Inflation / rate bear'),
    ('rates_2023',     '2023-08-01', '2023-10-31', 'Rate shock'),
]

#: When each family was PUBLISHED. Everything before this date is in-sample for its author,
#: however carefully the rules were pre-registered. Reported as its own column, never merged
#: into a headline figure. `DM` is Antonacci, *Risk Premia Harvesting Through Dual Momentum*
#: (SSRN 2042750), first version 2012-04-18 — the paper the four-module composite implements.
PUBLICATION_DATES = {
    'HAA': '2023-02-01',
    'BAA': '2022-07-01',
    'DAA': '2018-07-01',
    'VAA': '2017-07-01',
    'PAA': '2016-04-01',
    'DM': '2012-04-01',
    'GTAA': '2006-04-01',
}


def publication_date(strategy_name):
    """Publication date for a registry key, or None for benchmarks and custom variants.

    The family is the first token, with any trailing digits stripped: `PAA2_G12` is a PAA,
    because `PAA2` is Keller's name for the a=2 model and not a separate family. Without that
    strip, a renamed key would silently fall out of the before/after-publication split and
    the report would show one fewer row rather than an error — the quiet failure mode this
    repository keeps finding.
    """
    token = strategy_name.split('_')[0].upper()
    date = PUBLICATION_DATES.get(token) or PUBLICATION_DATES.get(token.rstrip('0123456789'))
    return pd.to_datetime(date) if date else None


def _episode_stats(returns, start, end):
    """(total return, worst drawdown inside the episode, n months) over [start, end]."""
    window = returns.loc[(returns.index >= pd.to_datetime(start))
                         & (returns.index <= pd.to_datetime(end))]
    if window.empty:
        return None
    # numpy, not two concatenated pandas objects: the old form built a duplicate index
    # label `0` and was correct only via a pandas fast path. Same change as eras._stats.
    w = np.concatenate([[1.0], (1.0 + window.to_numpy(dtype=float)).cumprod()])
    total = float(w[-1] - 1.0)
    dd = float((w / np.maximum.accumulate(w) - 1.0).min())
    return total, dd, len(window)


def episode_panel(metrics_data, episodes=EPISODES):
    """Per-strategy behaviour in each named regime.

    Uses `returns_full` — the strategy's **entire measurable history** — in preference to
    `returns`, which is bounded by the caller's requested START_DATE. That distinction is the
    whole point: a panel truncated at an arbitrary start date is not an antidote to
    date-selection bias, it is another expression of it. A run starting in 2015 would
    otherwise print `n/a` for the GFC even for SPY, which has data back to 2000.

    An `n/a` cell therefore means the strategy's own assets did not exist — a fact about the
    market, not about the parameters of this run.

    Returns ``{strategy: {episode_key: cell}}`` where a cell is either
    ``{'return', 'max_dd', 'n_months'}`` or ``{'na': 'inception YYYY-MM'}``.
    """
    panel = {}
    for d in metrics_data:
        returns = d.get('returns_full')
        if returns is None or returns.empty:
            returns = d.get('returns')
        if returns is None or returns.empty:
            continue
        first = pd.Timestamp(d.get('first_return_full') or d.get('first_return')
                             or returns.index[0])
        row = {}
        for key, start, end, _label in episodes:
            # Month granularity: a return dated 2007-11-30 covers an episode opening
            # 2007-11-01. Comparing raw timestamps would call that a missing episode.
            if pd.Period(first, freq='M') > pd.Period(pd.to_datetime(start), freq='M'):
                row[key] = {'na': f'inception {first.strftime("%Y-%m")}'}
                continue
            stats = _episode_stats(returns, start, end)
            if stats is None:
                row[key] = {'na': f'no data ({first.strftime("%Y-%m")})'}
                continue
            total, dd, n = stats
            row[key] = {'return': total, 'max_dd': dd, 'n_months': n}
        panel[d['name']] = row
    return panel


def coverage_fraction(row, episodes=EPISODES):
    """Share of the episode list this strategy can actually be measured over."""
    if not row:
        return 0.0
    measurable = sum(1 for key, *_ in episodes if 'na' not in row.get(key, {'na': 1}))
    return measurable / len(episodes)


def post_publication_split(returns, strategy_name):
    """(pre, post) return series either side of the family's publication date.

    Returns (None, returns) for benchmarks and custom variants, which have no publication
    date and therefore no in-sample period to separate out.
    """
    pub = publication_date(strategy_name)
    if pub is None or returns is None or returns.empty:
        return None, returns
    return returns[returns.index < pub], returns[returns.index >= pub]
