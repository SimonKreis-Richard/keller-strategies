"""
DAA (Defensive Asset Allocation) – Keller & Keuning 2018, SSRN 3212862.
Momentum 13612W, separate canary, CF = b/B, Easy Trading.

The selection rule is the thing to get right here, and this repository got it wrong until
2026-07-29: DAA's Top-T is EQUAL-WEIGHT AND SIGN-AGNOSTIC. The paper says so three times.

    §8: "notice that we did not use absolute momentum (ie. eliminating bad assets in favor
    of cash), except for the number of bad canary assets ... So for DAA, we did not
    eliminate bad assets in the top T selection of risky assets, although we reduced the
    top T"

    Conclusions: "EW-Top T (T<=N, long only), without intrinsic or absolute momentum."

    n.17 records absolute momentum on the risky Top-T as an alternative they TRIED —
    "slightly better returns with slightly worse drawdowns" — and did not adopt.

That separation is the whole difference between DAA and VAA. VAA counts breadth over its
own offensive universe; DAA moves the count to a dedicated two-asset canary so the
offensive sleeve's own breadth never touches the allocation. Reinstating a `> 0` filter
would collapse the two designs back together.
"""

from common.momentum import calc_13612w
from .base import BaseStrategy
import numpy as np
import pandas as pd

class DAA_G12(BaseStrategy):
    fidelity = 'faithful'
    source = 'Keller & Keuning, DAA, SSRN 3212862, §2 and §7 (DAA-G12: T=6, B=2, C3 cash)'

    def __init__(self):
        super().__init__("DAA_G12")
        self.offensive = ['SPY','IWM','QQQ','VGK','EWJ','VWO','VNQ','GSG','GLD','TLT','HYG','LQD']
        self.canary = ['VWO','BND']
        self.defensive = ['SHY','IEF','LQD']
        self.T = 6
        self.B = 2

    def sleeves(self):
        # LQD is dual-role in G12 (offensive universe and defensive candidate); the
        # VWO/BND canary resolves it. Subclasses inherit this unchanged because they only
        # ever rebind self.offensive / self.defensive / self.canary.
        return {'offensive': set(self.offensive),
                'defensive': set(self.defensive),
                'canary': list(self.canary)}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        return self._generate(scores_13612w)

    def _generate(self, scores):
        alloc = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        def_scores = scores[self.defensive]
        # Reindex to keep the full index: a fully-NaN defensive row would otherwise be
        # dropped, making best_def_series[date] raise a KeyError / create a junk column.
        best_def_series = def_scores.dropna(how='all').idxmax(axis=1).reindex(def_scores.index)
        # Skip first 12 months
        for i in range(12, len(scores)):
            date = scores.index[i]
            row = scores.iloc[i]
            best_def = best_def_series.loc[date]
            canary_scores = row[self.canary]
            b = (canary_scores <= 0).sum() + canary_scores.isna().sum()
            if self.T == 1:
                # DAA n.8, verbatim: "We also have added the rule that with T=1, CF is
                # simply b/B, in line with the ET idea, with b the number of bad assets."
                # The floor formula collapses at T=1 — floor(b/B) is 0 for every b < B, so
                # a T=1, B=2 book would jump 0% -> 100% with no middle rung. The paper saw
                # that coming and legislated the direct rule. Unimplemented until
                # 2026-07-30; no registered entry reaches it (every DAA has T >= 4), but a
                # `faithful` label covers the code's rules, not just the code's reachable
                # rules. The single risky slot is then held at 1-CF beside best_def at CF,
                # which the shared weighting arithmetic below already produces.
                cf = min(float(b) / self.B, 1.0)
                n_cash_slots = self.T if cf >= 1.0 else 0
            else:
                n_cash_slots = int(np.floor(b * self.T / self.B))
                cf = n_cash_slots / self.T
            if n_cash_slots >= self.T:
                if pd.notna(best_def):
                    alloc.loc[date, best_def] = 1.0
                continue

            n_risky = self.T - n_cash_slots

            # The Top-T is SIGN-AGNOSTIC. DAA's protection is the canary and nothing else:
            # it has already decided, above, how many of the T slots go to cash. Filtering
            # the survivors to positive momentum on top of that would apply the absolute
            # momentum the paper explicitly declines to use (§8, Conclusions, n.17) and
            # would let offensive breadth move the allocation, which is VAA's mechanism,
            # not DAA's. `.dropna()` stays: a missing score is missing data, not bad
            # momentum, and must never be rankable.
            off = row[self.offensive].dropna()
            if len(off) == 0:
                if pd.notna(best_def):
                    alloc.loc[date, best_def] = 1.0
                continue

            top_assets = off.nlargest(min(n_risky, len(off))).index.tolist()
            w_risky = (1.0 - cf) / len(top_assets)
            for a in top_assets:
                alloc.loc[date, a] = w_risky
            if cf > 0 and pd.notna(best_def):
                alloc.loc[date, best_def] += cf
        return alloc

class DAA_G4(DAA_G12):
    fidelity = 'faithful'
    source = 'Keller & Keuning, DAA, SSRN 3212862, §3 (R4 = SPY, VEA, VWO, BND; T=4)'

    def __init__(self):
        super().__init__()
        self.name = "DAA_G4"
        self.offensive = ['SPY','VEA','VWO','BND']
        self.T = 4

class DAA_G6(DAA_G12):
    fidelity = 'faithful'
    source = 'Keller & Keuning, DAA, SSRN 3212862, §7 (DAA-G6 universe, T=6)'

    def __init__(self):
        super().__init__()
        self.name = "DAA_G6"
        self.offensive = ['SPY','VEA','VWO','LQD','TLT','HYG']
        self.T = 6

# DELETED 2026-07-28: DAA_U6 and DAA_U15 (custom US-only universes that additionally put
# BIL inside the OFFENSIVE momentum ranking, so a T-bill fund competed against equities on
# 13612 momentum — either a dead slot or a duplicate of the canary), and DAA1_G12.
#
# DAA1_G12 is the one worth spelling out. Its defensive sleeve was `SHV, IEF, UST`, and UST
# is a 2x leveraged 7-10y treasury ETF: its risk-off state DOUBLED duration risk. Defence is
# held at 1x, always — `common/letf_mapper.assert_unlevered_defensive` now enforces that for
# every registered strategy, and the engine fails any strategy that breaks it. DAA1 also had
# the highest coverage cost in the registry: UST's 2010-02 inception dragged the whole
# comparison window from 2008-07 to 2011-04.

# NOTE: DAA_Leveraged (the full G12 universe executed via LETFs) was removed. Half of
# Keller's G12 offensive universe — VGK, EWJ, VWO, VNQ, GSG, HYG, LQD — has no admissible
# leveraged product, so those weights fell through to 1x while SPY/QQQ/IWM/TLT levered.
# Effective leverage then swung with the monthly momentum draw and the backtest no longer
# described the portfolio held. Use the uniform-ratio sizes in strategies/daa_leveraged.py.
