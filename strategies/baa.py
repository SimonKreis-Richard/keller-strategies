"""
BAA (Bold Asset Allocation) – Keller 2022.
Uses SMA12 for offensive/defensive, 13612W for canaries and absolute filter for defensive assets.
"""

from common.momentum import calc_sma12_ratio, calc_13612w
from .base import BaseStrategy
import pandas as pd

class BAA_G12(BaseStrategy):
    fidelity = 'faithful'
    source = 'Keller, BAA, SSRN 4166845, Fig 3 (NO=12, ND=7, NP=4, TO=6, TD=3, B=1)'

    def __init__(self):
        super().__init__("BAA_G12")
        self.offensive = ['SPY','QQQ','IWM','VGK','EWJ','VWO','VNQ','DBC','GLD','TLT','HYG','LQD']
        self.canary = ['SPY','VWO','VEA','BND']
        self.defensive = ['TIP','DBC','BIL','IEF','TLT','LQD','BND']
        self.TO = 6
        self.TD = 3

    def sleeves(self):
        # TLT, DBC and LQD are dual-role: momentum assets in the offensive universe and
        # candidates in the defensive basket. The SPY/VWO/VEA/BND canary decides which.
        return {'offensive': set(self.offensive),
                'defensive': set(self.defensive),
                'canary': list(self.canary)}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        # Calculate SMA12 ratio for offensive and defensive
        sma_ratio = calc_sma12_ratio(prices)
        # fast scores for canaries only
        fast = scores_13612w
        alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # Skip first 12 months
        for i in range(12, len(prices)):
            date = prices.index[i]
            # Canary test
            canary_fast = fast.iloc[i][self.canary]
            if (canary_fast <= 0).any() or canary_fast.isna().any():
                # Defensive : select TD=3 best SMA, then absolute filter
                def_row = sma_ratio.iloc[i][self.defensive].sort_values(ascending=False)
                topTD = def_row.head(self.TD).index.tolist()
                final_def = []
                for a in topTD:
                    # absolute defensive momentum is based on SMA12. `>=`, not `>`: BAA §2
                    # replaces a slot whose momentum is "less than BIL", so exact equality
                    # KEEPS the asset. Measure-zero in floating point against a distinct
                    # ticker (BIL against itself hits it every month, with the same outcome
                    # either way), so this is paper-exactness rather than a number change —
                    # verified against the golden master when changed on 2026-07-30.
                    sma_a = sma_ratio.iloc[i][a]
                    sma_bil = sma_ratio.iloc[i]['BIL']
                    if pd.notna(sma_a) and sma_a >= sma_bil:
                        final_def.append(a)
                    else:
                        final_def.append('BIL')
                w = 1.0 / len(final_def)
                for a in final_def:
                    alloc.loc[date, a] += w
            else:
                # Offensive : TO=6 best SMAs
                off_row = sma_ratio.iloc[i][self.offensive].sort_values(ascending=False)
                topTO = off_row.head(self.TO).index.tolist()
                w = 1.0 / self.TO
                for a in topTO:
                    alloc.loc[date, a] += w
        return alloc

class BAA_G4(BAA_G12):
    fidelity = 'faithful'
    source = 'Keller, BAA, SSRN 4166845, Fig 6 (SelO = QQQ, VWO, VEA, BND; TO=1, TD=3)'

    def __init__(self):
        super().__init__()
        self.name = "BAA_G4"
        self.offensive = ['QQQ','VWO','VEA','BND']
        self.TO = 1

class BAA_SPY(BAA_G12):
    """Keller's own BAA-SPY (Fig 11). A CONTROL here, but a PUBLISHED one — see `HAA_Simple`.

    This docstring said the opposite until 2026-07-30: "Unlike HAA-Simple this one really is
    unpublished: Keller names no single-asset BAA variant." BAA §5 asks, in as many words,
    "would this (defensive/protective) part of our BAA model also works with just SPY as
    offensive asset (so SO=SPY and TO=NO=1)?" and answers with Fig 11, captioned BAA-SPY:
    SelO=SPY, SelD=TIP/DBC/BIL/IEF/TLT/LQD/BND, SelP=SPY/VWO/VEA/BND, NO=1, ND=7, NP=4,
    B=1, TO=1, TD=3 — every one of the nine parameters this class inherits from `BAA_G12`.
    The sentence was written by analogy to HAA_Simple instead of by reading BAA §5, which is
    the fourth label found wrong and the fourth found wrong in the same direction
    (under-claiming). `tests/test_paper_rules.py::TestFidelityAgainstSource` now pins every
    label so the fifth cannot arrive the same way.

    `faithful` and `control` are orthogonal, exactly as `HAA_Simple` already asserts: the
    role records why it is excluded from selection statistics (a one-asset universe measures
    the timing rule, not an investable strategy); the fidelity records whose design it is.
    """
    fidelity = 'faithful'
    source = 'Keller, BAA, SSRN 4166845, §5 and Fig 11 (BAA-SPY: NO=TO=1, SelO=SPY)'
    role = 'control'

    def __init__(self):
        super().__init__()
        self.name = "BAA_G1_SPY"
        self.offensive = ['SPY']
        self.TO = 1

# DELETED 2026-07-28: BAA_G4_T2 — the same universe as BAA_G4 with TO changed from 1 to 2,
# rho 0.894. A parameter twiddle on an unchanged universe is a multiple-testing dimension
# with no mechanism behind it. BAA_G12_T3 had already been cut on the same criterion
# (rho 0.951); leaving G4_T2 in place was an inconsistency.

# NOTE: BAA_Leveraged (the full G12 universe executed via LETFs) was removed. Most of the
# G12 offensive universe — VGK, EWJ, VWO, VNQ, GLD, HYG — has no admissible leveraged
# product at 3x, and TLT/DBC are dual-role (also defensive) so they are held 1x by design.
# Only SPY/QQQ/IWM ever levered, leaving an offensive sleeve that mixed 3x and 1x depending
# on the month. Use the uniform-ratio sizes in strategies/baa_leveraged.py.
