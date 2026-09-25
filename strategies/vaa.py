"""
VAA (Vigilant Asset Allocation) – Keller 2017.
Canary = offensive universe itself (breadth), CF = b/B.
"""

from common.momentum import calc_13612w
from .base import BaseStrategy
import numpy as np
import pandas as pd

class VAA_G12(BaseStrategy):
    fidelity = 'faithful'
    source = ('Keller & Keuning, VAA, SSRN 3002624, Table 1 and n.11 (VAA-G12: T=2, B=4). '
              'Universe on the VANGUARD funds n.11 says were actually backtested — VWO and '
              'VNQ, not the iShares EEM and IYR the body text names.')

    def __init__(self):
        super().__init__("VAA_G12")
        # VWO and VNQ, not EEM and IYR. Changed 2026-07-29 — see the note below the class.
        self.offensive = ['SPY','IWM','QQQ','VGK','EWJ','VWO','VNQ','GSG','GLD','TLT','LQD','HYG']
        self.defensive = ['SHY','IEF','LQD']
        self.T = 2
        self.B = 4

    def sleeves(self):
        # VAA declares NO canary on purpose. Its crash signal is a BREADTH COUNT over the
        # offensive universe (b assets with non-positive momentum -> b*T/B cash slots), not
        # the "any canary down => risk-off" rule that defensive_mask implements. Feeding the
        # offensive universe in as a canary would mark LQD defensive in almost every month,
        # since some asset in a 12-asset universe is nearly always negative. With no canary,
        # dual-role LQD counts as offensive — the conservative direction, per BaseStrategy.
        return {'offensive': set(self.offensive),
                'defensive': set(self.defensive),
                'canary': []}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        return self._generate(scores_13612w)

    def _generate(self, scores):
        alloc = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        
        # FIX: Remove fragile .dropna(how='all') to preserve index integrity
        def_scores = scores[self.defensive]
        # Use dropna to avoid ValueError on all-NA, then reindex to keep the full index
        best_def_series = def_scores.dropna(how='all').idxmax(axis=1).reindex(def_scores.index)
        
        # FIX: Remove hardcoded range(12, len(scores)), rely on organic missing data logic
        for i in range(len(scores)):
            date = scores.index[i]
            row = scores.iloc[i]
            off_scores = row[self.offensive]
            
            # Extract the best defensive asset for the date (can be NaN)
            best_def = best_def_series.loc[date]
            
            # Pure math skip: only ignore rows where mathematically impossible to allocate
            if off_scores.isna().all() and pd.isna(best_def):
                continue
                
            b = (off_scores <= 0).sum() + off_scores.isna().sum()
            if self.T == 1 and b >= 1:
                # VAA §4, verbatim: "When B=1 or T=1 the whole portfolio is fully invested
                # in cash (when b>=1) or fully invested in the top T risky asset(s) (when
                # b=0)." The floor formula only reproduces that for B=1; at T=1 with B>1 it
                # gives floor(b/B)=0 and stays fully risky, the opposite of the paper's own
                # sentence. NOTE this is VAA's rule, not DAA's: the later DAA paper (n.8)
                # grades the same case to CF=b/B, and each paper governs its own strategy —
                # see strategies/daa.py for the graded version. No registered VAA reaches
                # this branch (VAA_G4 is T=1, B=1, where the floor already agrees).
                n_cash_slots = self.T
            else:
                n_cash_slots = int(np.floor(b * self.T / self.B))

            if n_cash_slots >= self.T:
                if pd.notna(best_def):
                    alloc.loc[date, best_def] = 1.0
                continue

            n_risky = self.T - n_cash_slots
            cf = n_cash_slots / self.T
            # The Top-T is SIGN-AGNOSTIC, same rule as DAA (corrected there 2026-07-29,
            # here 2026-07-30). VAA §5 declines absolute momentum on the selection and says
            # why: "in the case where b<N, the (rounded) cash fraction CF is always higher
            # with VAA than in the traditional Dual case. So no bad assets will show up in
            # the remaining Top assets." The cash ladder above has already spent the breadth
            # count; filtering the survivors to positive momentum would double-count it.
            # Measured over the real panel before the change: zero months where the filter
            # altered the Top-T (the §5 argument holds structurally at T=2, B=4) — so this
            # is provably a no-op on the registry, guarded by the golden master, and the
            # point is that the code now says what the paper says. `.dropna()` stays: a
            # missing score is missing data, not bad momentum, and must never be rankable.
            avail = off_scores.dropna()

            if len(avail) == 0:
                if pd.notna(best_def):
                    alloc.loc[date, best_def] = 1.0
                continue

            top_assets = avail.nlargest(min(n_risky, len(avail))).index.tolist()
            w_risky = (1.0 - cf) / len(top_assets)
            
            for a in top_assets:
                alloc.loc[date, a] = w_risky
                
            if cf > 0 and pd.notna(best_def):
                alloc.loc[date, best_def] += cf
                
        return alloc

# DELETED 2026-07-28: VAA_U6 and VAA_U15 — custom US-only universes (the algorithm was the
# paper's, the asset lists were not), which also put BIL inside the offensive momentum
# ranking. They were the two most DISTINCT entries in the registry by return correlation
# (rho_max 0.655 and 0.676), and that was not enough: distinctness produced by an unpublished
# universe is not evidence, it is a degree of freedom.

class VAA_G4(VAA_G12):
    """Keller & Keuning's VAA-G4, on Keller's OWN tickers.

    This carried `fidelity = 'proxy'` until 2026-07-29 on the stated grounds that "the paper
    specifies SPY, EFA, EEM, AGG" while the code uses SPY, VEA, VWO, BND. The paper's own
    footnote 11 contradicts the premise:

        "We actually used (proxies for) Vanguard ETFs VEA, VWO, VNQ, and BND instead of the
        mentioned (and more common) iShares ETFs EFA, EEM, IYR, and AGG, respectively, in
        nearly all our backtest since these ETFs has lower fees and similar AUM's"

    So the Vanguard names are the ones behind the published numbers, and the DAA paper
    restates VAA-G4 as "R4 = SPY, VEA, VWO, BND" directly. The tickers here are Keller's.

    The same footnote governs `VAA_G12`, which held EEM and IYR until 2026-07-29. Applying
    n.11 to G4 and not to G12 was an inconsistency rather than a decision, and the defence
    offered for it — "Keller treats the two fund families as interchangeable, so either set
    is his" — does not survive re-reading. The footnote does not say the funds are
    equivalent. It says WHICH ONES HE RAN, and gives his reason, and his reason still checks
    out: VWO costs 0.06% against EEM's 0.72%, VNQ 0.13% against IYR's 0.42% (2026-07).

    Switching cost performance — over 2015-2024, VAA_G12's CAGR fell 3.53% -> 3.03% and its
    Sortino 0.39 -> 0.31 — which is the reason to trust it. Coverage was unaffected: VWO's
    pre-inception history is chain-linked from EEM (`HISTORY_BACKFILL`) and VNQ (2004-09)
    predates the binding HYG (2007-04). Recorded in `strategy_specs/vaa.md`.
    """
    fidelity = 'faithful'
    source = ('Keller & Keuning, VAA, SSRN 3002624, Table 5 and n.11 (VAA-G4: T=1, B=1; '
              'n.11 names VEA/VWO/BND as the funds actually backtested)')

    def __init__(self):
        super().__init__()
        self.name = "VAA_G4"
        self.offensive = ['SPY', 'VEA', 'VWO', 'BND']
        self.defensive = ['SHY', 'IEF', 'LQD']
        self.T = 1
        self.B = 1

# DELETED 2026-07-29: EVERY leveraged VAA wrap (vaa_leveraged.py, G3 and G4 at 2x).
#
# Not on performance — on RULE 4 in common/letf_mapper.py, which is checkable by reading
# the class and needs no returns in front of it: *a wrap may change what is HELD, never
# what decides to DE-RISK.*
#
# VAA declares no canary on purpose. Read `sleeves()` above: it returns `'canary': []`, and
# the docstring explains why. VAA's protection IS the breadth count over its own offensive
# universe, `b = (off_scores <= 0).sum()`, divided by `B`. A leveraged wrap must restrict
# that universe to what executes at one uniform multiple, and `VAALeveraged` accordingly set
# `self.B = n`. So the restriction did not narrow the portfolio and leave the rule alone —
# it rebuilt the rule out of three US equity ETFs correlating 0.79-0.91 with each other.
# Counting how many of twelve asset classes have rolled over is a market-breadth signal;
# counting how many of SPY/QQQ/IWM have rolled over is asking one question three times.
#
# The measurement, over 2010-02..2026-06: VAA_G3_Leveraged_2X correlated **0.376** with
# VAA_G12 — the lowest wrap-to-parent figure in the registry, against 0.823 for BAA_G4.
# That is a statement about fidelity, not about returns, which is why it was allowed to
# decide. The behavioural consequence was consistent with it: VAA's wraps went defensive
# one to two months AFTER the exogenous-canary families in every crash measured
# (2011, 2015-16, 2018 Q4, COVID, 2022), and at 2x that lag cost -34% to -38% in 2022.
#
# There is no fix, which is the point. Enlarging the universe until the breadth count means
# something again requires assets with no admissible LETF (RULE 1/RULE 2), and any smaller
# universe reproduces the defect. VAA is a fine strategy that does not admit a leveraged
# variant. Not every model has to have one.
