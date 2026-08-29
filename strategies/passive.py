"""Passive reference portfolios, including the levered ones.

WHY THESE EXIST, and it is the only justification needed. Every leveraged wrap in this
repository was compared only against **1x** references, so the question that matters could not
be answered: *how much of a wrap's record is the timing rule, and how much is just leverage?*
These four close that gap and add **zero degrees of freedom** — no universe choice, no signal,
no fitted parameter, nothing that could have been tuned:

    Sixty_Forty_1X      SPY 60 / IEF 40, monthly     the classic reference the repo lacked
    SPY_2X_Benchmark    SSO 100%, buy & hold         wrap - this = what the timing rule adds
    SPY_3X_Benchmark    UPRO 100%, buy & hold        this - SPY  = what leverage alone does
    RiskParity_3X       UPRO 55 / TMF 45, quarterly  this - SPY_3X = what diversification adds

The subtraction in the third column is the whole point. `HAA_G1_Simple` and `BAA_G1_SPY` do this
for the unlevered families — hold the timing rule, strip the universe — and nothing did it for
the levered ones.

WHY THE LEVERAGE RULES DO NOT REFUSE THESE. Worth stating so it is not re-litigated:

    RULE 1 (uniform ratio)      UPRO and TMF are both 3x; SSO alone is trivially uniform.
    RULE 2 ($100M AUM floor)    UPRO 3.97B, TMF 3.1B, SSO 5.1B. All clear it comfortably.
    RULE 3 (defence never 3x)   VACUOUS: these have no defensive sleeve. A static portfolio
                                never de-risks, so there is nothing to lever.
    RULE 4 / RULE 5             Do not apply. These are not wraps of anybody's signal; there
                                is no de-risking rule and no cash ladder to survive.

THE LADDER OF REFERENCES SKIPS 2x ON THE BOND SIDE, and this is where to look for it: there is
no admissible 2x treasury product (UBT holds ~$65M, below the RULE 2 floor). So a
`RiskParity_2X` is **not buildable** — the risk-parity reference jumps 1x to 3x with nothing in
between. That is a fact about the product market, not an omission here.

WHAT THESE COST IN COVERAGE. SSO 2006-06, TMF 2009-04, UPRO 2009-06. Plus 13 months of warm-up,
the two 3x entries begin ~2010. They are barred from setting the shared ranked window by
`letf_mapper.holds_leveraged_product` — a structural check, not a fidelity label — so they
cannot shorten the 23 entries they exist to be compared against. See the comment in
`main.run_backtest` for why the fidelity filter alone was not enough.
"""
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


def _static(prices, weights):
    """A constant target: the ledger rebalances back to `weights` every month."""
    alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for ticker, w in weights.items():
        if ticker in alloc.columns:
            alloc[ticker] = w
    return alloc


def _drifting(prices, weights, months):
    """Target weights that DRIFT between rebalances, resetting every `months` months.

    This is what makes a quarterly portfolio quarterly, and it cannot be expressed as a static
    target. Emitting a constant 55/45 makes the ledger trade back to 55/45 **every month** —
    a monthly-rebalanced portfolio wearing a quarterly name, with different turnover, different
    costs and a different return path from the design it claims to implement.

    So the target at month *t* is the portfolio the last rebalance actually left behind, grown
    at each holding's own realised return: `w_i * P_i(t) / P_i(r)`, renormalised. On a rebalance
    month it collapses back to `weights`, and the ledger sees no trade in between.

    Rebalance dates are pinned to the CALENDAR — January/April/July/October for `months=3` —
    not to every third row of the price panel. That distinction is not pedantry: an
    index-position rule (`i % months == 0`) makes the rebalance phase depend on where the panel
    happens to start, so moving `DATA_START_DATE` by one month would silently produce a
    different portfolio and a different record for the same named strategy.
    """
    held = [t for t in weights if t in prices.columns]
    alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if not held:
        return alloc

    px = prices[held]
    base_w = np.array([weights[t] for t in held], dtype=float)
    base_w = base_w / base_w.sum()

    anchor = None                      # prices at the most recent rebalance
    for i, date in enumerate(prices.index):
        row = px.iloc[i]
        if row.isna().any():
            continue                   # not all legs tradeable yet; coverage handles the trim
        on_schedule = (pd.Timestamp(date).month - 1) % months == 0
        if anchor is None or on_schedule:
            anchor = row.to_numpy(dtype=float)
            w = base_w
        else:
            grown = base_w * (row.to_numpy(dtype=float) / anchor)
            total = grown.sum()
            w = grown / total if total > 0 else base_w
        alloc.loc[date, held] = w
    return alloc


class Sixty_Forty_1X(BaseStrategy):
    """SPY 60 / IEF 40, rebalanced monthly. The reference this repository did not have."""
    fidelity = 'faithful'
    source = ('the classic 60/40 — SPY 60% / IEF 40%, rebalanced monthly. No published author '
              'to defer to and no rule to be faithful or unfaithful to; the label marks that '
              'there is nothing here to get wrong, as for SPY_Benchmark.')
    role = 'benchmark'

    def __init__(self):
        super().__init__('Sixty_Forty_1X', is_active=False)
        self.weights = {'SPY': 0.60, 'IEF': 0.40}

    def sleeves(self):
        # IEF is a STRUCTURAL allocation here, not a defensive switch — this portfolio never
        # de-risks. Declaring it defensive would make the margin logic believe the benchmark
        # de-levers, which it does not. Same reasoning as Golden_Butterfly.
        return {'offensive': set(self.weights), 'defensive': set(), 'canary': []}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        return _static(prices, self.weights)


class SPY_2X_Benchmark(BaseStrategy):
    """100% SSO, buy and hold. Subtract a 2x wrap from this to isolate its timing rule."""
    fidelity = 'faithful'
    source = 'buy and hold SSO (2x S&P 500) — no rule to be faithful or unfaithful to'
    role = 'benchmark'

    def __init__(self):
        super().__init__('SPY_2X_Benchmark', is_active=False)
        # `leverage` records the EXPOSURE this entry carries, not whether it maps anything. An
        # earlier version left it at 1.0 here on the grounds that a benchmark naming SSO
        # directly "maps nothing" — and the regenerated growth chart showed the consequence
        # immediately: a 100%-SSO line drawn at 1x width, and filed under "1x" by the picker's
        # ratio filter. Both consumers read this attribute as exposure, so exposure is what it
        # must hold. `role = 'benchmark'` above shadows the derived-role property, so setting
        # this cannot turn a reference into an `exploratory` entry.
        self.leverage = 2
        self.weights = {'SSO': 1.0}

    def sleeves(self):
        return {'offensive': {'SSO'}, 'defensive': set(), 'canary': []}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        return _static(prices, self.weights)


class SPY_3X_Benchmark(BaseStrategy):
    """100% UPRO, buy and hold. Unprotected 3x — the thing every 3x variant must beat."""
    fidelity = 'faithful'
    source = 'buy and hold UPRO (3x S&P 500) — no rule to be faithful or unfaithful to'
    role = 'benchmark'

    def __init__(self):
        super().__init__('SPY_3X_Benchmark', is_active=False)
        self.leverage = 3          # exposure carried, not a mapping multiple — see SPY_2X
        self.weights = {'UPRO': 1.0}

    def sleeves(self):
        return {'offensive': {'UPRO'}, 'defensive': set(), 'canary': []}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        return _static(prices, self.weights)


class RiskParity_3X(BaseStrategy):
    """Hedgefundie's Excellent Adventure: UPRO 55 / TMF 45, rebalanced QUARTERLY.

    Subtract `SPY_3X_Benchmark` from this and what remains is what diversification buys at 3x,
    with no timing rule involved at all — which is the one number that tells you whether the
    levered wraps' canaries are earning anything a static bond sleeve would not.

    **Quarterly, not monthly, and implemented as such.** The bogleheads thread specifies
    quarterly rebalancing; this engine is otherwise monthly. Emitting a static 55/45 would have
    made the ledger rebalance every month — a different portfolio with different turnover
    wearing this one's name. `_drifting` holds the weights between quarter-ends instead. The
    cost of that honesty is one non-obvious helper; the alternative was a silent deviation.

    **This is the one entry in this module that carries real risk of being read as advice.** It
    holds TMF, whose measured maximum drawdown since 2009-04 is -91.6%, in a portfolio with no
    de-risking rule whatsoever. rho(SPY, TLT) in monthly returns has been POSITIVE since 2020
    (+0.313, against -0.342 over 2012-2019), so the diversification the 55/45 split depends on
    is not a constant of nature. It is a reference point, and `role='benchmark'` keeps it out of
    the selection statistics.
    """
    fidelity = 'proxy'
    source = ("\"Hedgefundie's Excellent Adventure\", bogleheads.org forum thread, 2019: "
              "UPRO 55% / TMF 45%, rebalanced quarterly. 'proxy' rather than 'faithful' because "
              "the weights and schedule are his and the data, execution convention "
              "(next-open fills) and cost model are this engine's.")
    role = 'benchmark'

    #: Months between rebalances. 3 = quarterly, as published.
    REBALANCE_MONTHS = 3

    def __init__(self):
        super().__init__('RiskParity_3X', is_active=False)
        self.leverage = 3          # both legs are 3x, so the sleeve is uniformly 3x
        self.weights = {'UPRO': 0.55, 'TMF': 0.45}

    def sleeves(self):
        # TMF is a structural allocation, NOT a defensive sleeve — and this distinction is
        # load-bearing rather than cosmetic. Declaring TMF defensive would trip
        # `assert_unlevered_defensive` (RULE 3 forbids a levered defensive sleeve) and the
        # engine would refuse to price this entry at all. It is not defence: it is a 3x
        # duration bet held permanently, which is exactly what RULE 3 exists to keep out of a
        # sleeve that claims to reduce risk. Here it claims nothing of the kind.
        return {'offensive': set(self.weights), 'defensive': set(), 'canary': []}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        return _drifting(prices, self.weights, self.REBALANCE_MONTHS)
