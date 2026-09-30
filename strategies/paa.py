"""
PAA (Protective Asset Allocation) – Keller 2016.
Momentum = SMA12_ratio (MOM12). Breadth n = # positive among offensive (size 12).
Protection variants: PAA0, PAA1, PAA2.

Defensive = IEF, and that is the PAPER's deliberate choice, not a simplification made here.
Keller & Butler §2: "We will, however, use a more bond-like cash proxy by using the exchange
traded intermediate (7-10y) treasury bond fund IEF", with a footnote preferring it over
shorter treasuries for liquidity. (Keller's later BAA paper restates PAA-G12 with SelD =
{BIL, IEF}, ND=2 — a restatement, and the original governs; the 2026-07-30 audit flagged
the discrepancy between the two sources as its largest unverified point, and reading SSRN
2759734 settles it in favour of what this file already does.)
"""

from common.momentum import calc_sma12_ratio
from .base import BaseStrategy
import pandas as pd

class PAA(BaseStrategy):
    #: On the BASE class, so the leveraged wraps inherit a citation to the paper whose
    #: signal they run. `PAA2` narrows it to the specific model below.
    source = 'Keller & Butler, PAA, SSRN 2759734, §3 (protection factor a; N=12, T=6)'

    def __init__(self, name, variant='PAA2'):
        super().__init__(name)
        self.variant = variant
        self.offensive = ['SPY','QQQ','IWM','VGK','EWJ','EEM','IYR','GSG','GLD','TLT','LQD','HYG']
        self.defensive = ['IEF']  # can expand to SHY/IEF later
        self.T = 6

    def sleeves(self):
        # Like VAA, PAA has no canary: its protection is a breadth count over the offensive
        # universe fed through CF = (N-n)/(N - a*N/4). IEF is the sole defensive bucket and
        # is not in the offensive universe, so no dual role arises.
        return {'offensive': set(self.offensive),
                'defensive': set(self.defensive),
                'canary': []}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        mom12 = calc_sma12_ratio(prices)
        alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # Skip first 12 months
        for i in range(12, len(mom12)):
            date = mom12.index[i]
            row = mom12.iloc[i]
            off_mom = row[self.offensive]
            n_pos = (off_mom > 0).sum()
            # CF based on variant
            N = len(self.offensive)
            if self.variant == 'PAA0':
                cf = (N - n_pos) / N
            elif self.variant == 'PAA1':
                cf = max(0.0, min(1.0, (N - n_pos) / (0.75 * N)))
            else:  # PAA2
                cf = max(0.0, min(1.0, (N - n_pos) / (0.5 * N)))
            # Selection of risky assets — GOOD (positive-momentum) assets ONLY, and that
            # is the PAPER'S OWN RULE, not an implementation artefact. Keller & Butler §3,
            # recipe step 2, verbatim: "Determine the Top (Top<=N) good assets with the
            # highest momentum ... If n<Top, only the n good assets (with positive
            # momentum) will be included in this risky EW portfolio."
            #
            # A 2026-07-30 external audit asked for this filter's removal, citing BAA §2's
            # restatement ("like for PAA, we don't use absolute momentum for the Top6
            # selection") — the audit had not read the PAA paper (it said so) and Keller's
            # 2022 recollection contradicts his own 2016 recipe. The original governs the
            # entry whose `source` cites it. Note the conflict is inert either way: PAA2
            # holds risk only when n_pos >= 7, so the top six are all positive and the two
            # readings pick identical portfolios in every measured month. DAA is the
            # opposite case — ITS paper disclaims the filter three times, and there the
            # filter was a real defect, removed 2026-07-29. Same-looking line, opposite
            # verdicts, each decided by its own primary source.
            pos_assets = off_mom[off_mom > 0]
            if len(pos_assets) == 0:
                alloc.loc[date, self.defensive[0]] += 1.0
                continue
            top_assets = pos_assets.nlargest(min(self.T, len(pos_assets))).index.tolist()
            weight_risky = (1.0 - cf) / len(top_assets)
            for a in top_assets:
                alloc.loc[date, a] += weight_risky
            # self.defensive[0], not a literal: a subclass rebinding the defensive bucket
            # must not silently keep allocating to IEF.
            alloc.loc[date, self.defensive[0]] += cf
        return alloc

class PAA2(PAA):
    """Keller & Butler's PAA2 — the a=2, high-protection model, on the N=12 universe.

    `PAA2` is the PAPER's own name, not a version number. The paper defines a protection
    factor a and names the three resulting models PAA0 / PAA1 / PAA2 ("low, medium and high
    protection ... We will denote these models by PAA0, PAA1 and PAA2 respectively", §3), and
    it designates this one as its conclusion: "It is this PAA2 model which we consider our
    alternative for a 1-year term deposit." The registry key was `PAA_G12_V2` until
    2026-07-29, which read like a second version of the code and hid whose label it was.

    Bond fraction: CF = (N-n)/(N - 2N/4) = (12-n)/6.

    The a=0 and a=1 variants (PAA0/PAA1) are no longer registered: they measured
    rho 0.974-0.982 against this one over 2012-2024 and added no information. The
    `variant` parameter on PAA still implements all three formulas for ad-hoc study.
    """
    fidelity = 'faithful'
    source = 'Keller & Butler, PAA, SSRN 2759734, §3 (PAA2: a=2, N=12, T=6, IEF)'

    def __init__(self):
        super().__init__("PAA2_G12", variant='PAA2')

# DELETED 2026-07-29: EVERY leveraged PAA wrap (paa_leveraged.py, G3 and G4 at 2x).
#
# Same finding as VAA, same rule — RULE 4 in common/letf_mapper.py: *a wrap may change what
# is HELD, never what decides to DE-RISK.* PAA has no canary either (`sleeves()` returns
# `'canary': []`); its protection is `CF = (N - n_pos) / (0.5 * N)` with `N` the size of the
# offensive universe. A wrap restricting that universe to three or four tickers rewrites N,
# and therefore rewrites the protection factor Keller spent the paper calibrating.
#
# What it degenerated into is worth spelling out, because it is arithmetic rather than
# opinion. At N=3 the formula gives CF = (3 - n_pos)/1.5, clipped to [0, 1]:
#
#     n_pos = 3 -> 0%      n_pos = 2 -> 67%      n_pos <= 1 -> 100%
#
# So the "high protection" model went fully defensive the moment two of three near-identical
# US equity ETFs were negative — and given their correlations, that is nearly the same event
# as all three. Three rungs where the paper has thirteen, and a hair trigger on a signal
# with no independent information in it.
#
# Measured: PAA_G3_Leveraged_2X correlated 0.543 with PAA2_G12 (G4: 0.677), against 0.823
# for BAA_G4. Like VAA, it fired one to two months late in every crash measured.
#
# PAA keeps its strongest property in the unlevered G12 form and does not need a wrap to
# show it: it traded 6.1-6.7x notional per year against 11-13x for every other family, less
# than half the rotation and therefore less than half the cost and daily-reset drag.
