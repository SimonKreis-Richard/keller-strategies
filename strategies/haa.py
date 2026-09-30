"""
HAA (Hybrid Asset Allocation) – Official variants.
All use the 13612U momentum (unweighted) and the TIP canary.
"""

from common.momentum import calc_13612u
from .base import BaseStrategy
import pandas as pd

class HAABase(BaseStrategy):
    def __init__(self, name, offensive, TO):
        super().__init__(name, score_type='unweighted')
        self.offensive = offensive
        self.canary = ['TIP']
        self.defensive = ['BIL', 'IEF']
        self.TO = TO  # number of assets to select (NO/2)

    def sleeves(self):
        # IEF is dual-role: it sits in the offensive universe of the larger variants AND is
        # one of the two defensive candidates. The TIP canary resolves which role it plays.
        return {'offensive': set(self.offensive),
                'defensive': set(self.defensive),
                'canary': list(self.canary)}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        # main.py passes 13612U scores in the scores_13612w argument for HAA
        scores = scores_13612w
        alloc = pd.DataFrame(0.0, index=scores.index, columns=prices.columns)
        def_scores = scores[self.defensive]
        best_def_series = def_scores.dropna(how='all').idxmax(axis=1)

        # Skip first 12 months (lookback)
        for i in range(12, len(scores)):
            date = scores.index[i]
            row = scores.iloc[i]
            
            if date in best_def_series.index and not pd.isna(best_def_series[date]):
                best_def = best_def_series[date]
            else:
                best_def = self.defensive[0]
                
            tip_score = row['TIP']
            if pd.isna(tip_score) or tip_score <= 0:
                alloc.loc[date, best_def] = 1.0
                continue
            # TopX Selection
            off_scores = row[self.offensive].sort_values(ascending=False)
            topX = off_scores.head(self.TO).index.tolist()
            final = []
            for a in topX:
                if off_scores[a] > 0:
                    final.append(a)
                else:
                    final.append(best_def)
            w = 1.0 / len(final)
            for a in final:
                alloc.loc[date, a] += w
        return alloc

# Concrete variants. The paper presents HAA-8 and HAA-12; those two are registered as
# strategies. HAA_G4 and HAA_G16 were custom sizes and were DELETED on 2026-07-28 — G16
# additionally measured rho 0.935 against G12 while starting a full year later (SCZ,
# 2007-12), so it paid coverage and returned nothing.
class HAA_Simple(HAABase):
    """Keller & Keuning's own HAA-Simple. A CONTROL by role, faithful by construction.

    §6: "in fig 12 we will look at a very special HAA case with only one risky asset SPY
    (NO=TO=1) ... We will call this special version with only SPY the HAA-Simple", with
    SelO = SPY, SelD = BIL/IEF, SelP = TIP. That is exactly what is built below, so the
    label was wrong: this was `custom` until 2026-07-29, which claimed the author had not
    published a variant he named in a figure caption.

    `role = 'control'` stays, and the two attributes answer different questions. Fidelity
    asks whether anyone published this; role asks whether you would hold it. A single-asset
    degenerate case is excluded from the selection statistics because a portfolio nobody
    would choose is not a trial anyone ran — but it answers a question the family cannot
    answer about itself: how much of HAA's record is the timing rule and how much is the
    twelve-asset universe? Worth remembering that this control once beat HAA_G12 on both
    Sharpe and Sortino.
    """
    fidelity = 'faithful'
    source = 'Keller & Keuning, HAA, SSRN 4346906, §6 and Fig 12 (HAA-Simple)'
    role = 'control'

    def __init__(self):
        super().__init__("HAA_G1_Simple", ['SPY'], TO=1)

class HAA_Balanced(HAABase):
    fidelity = 'faithful'
    source = 'Keller & Keuning, HAA, SSRN 4346906, Fig 6 (HAA-8: NO=8, TO=4)'

    def __init__(self):
        offensive = ['SPY','IWM','VEA','VWO','VNQ','DBC','IEF','TLT']
        super().__init__("HAA_G8_Balanced", offensive, TO=4)

class HAA_12(HAABase):
    fidelity = 'faithful'
    source = 'Keller & Keuning, HAA, SSRN 4346906, Fig 10 (HAA-12: NO=12, TO=6)'

    def __init__(self):
        offensive = ['SPY','QQQ','IWM','VGK','EWJ','VWO','VNQ','DBC','GLD','IEF','TLT','LQD']
        super().__init__("HAA_G12", offensive, TO=6)
