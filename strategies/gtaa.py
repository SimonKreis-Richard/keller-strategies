"""
GTAA (Global Tactical Asset Allocation) — after Faber (2006).
Uses a 10-month Simple Moving Average (SMA) for absolute momentum.

**Fidelity: faithful.** `GTAA_G5` is Faber's published five-asset timing model — five equal
sleeves (US equity, foreign equity, bonds, commodities, real estate), each held only while its
own price is above its 10-month SMA, with the failing sleeves' weight in cash. It is the only
variant registered here, and the only Faber entry in the registry.

The `GTAA_G13_Moderate` and `GTAA_G13_Aggressive` variants were deleted on 2026-07-28 — see
the note at the bottom of this file. See also `strategy_specs/gtaa.md` and `KNOWN_GAPS.md`.
"""

from common.momentum import calc_sma10_ratio
from .base import BaseStrategy
import pandas as pd


class _GTAASleeves:
    """Shared sleeve declaration for the three GTAA variants.

    GTAA is per-asset trend following: each slot is held only while its own price is above
    its SMA, and every failing slot's weight goes to the cash proxy. There is no canary —
    the absolute-momentum filter IS the de-risking rule, applied asset by asset.
    """

    def sleeves(self):
        return {'offensive': set(self.universe) - {self.cash_proxy},
                'defensive': {self.cash_proxy},
                'canary': []}


class GTAA_5(_GTAASleeves, BaseStrategy):
    """Faber's five-asset timing model, on ETF stand-ins for his index series.

    `proxy`, not `faithful`, since 2026-07-29. The algorithm and the five asset classes are
    Faber's exactly; the instruments are not, and cannot be — Faber (2006) times INDICES
    (S&P 500, MSCI EAFE, GSCI, NAREIT, 10-year US Treasury), not funds, and several had no
    tradable ETF at the time. DBC is a broad commodity fund tracking the DBIQ index, which
    is not the GSCI; IEF is a 7-10y Treasury fund, not the 10-year note.

    That is exactly what `proxy` means in `BaseStrategy`: the paper's algorithm and
    universe, different funds. Calling it `faithful` claimed a correspondence to published
    tickers that no published tickers exist for.
    """
    fidelity = 'proxy'
    source = ('Faber, A Quantitative Approach to Tactical Asset Allocation, SSRN 962461 '
              '(2006), five-asset timing model vs 10-month SMA; DBC stands in for the GSCI '
              'and IEF for the 10-year Treasury, neither of which Faber traded as a fund')

    def __init__(self):
        super().__init__("GTAA_G5")
        # 5 asset classes: US Stocks, Foreign Stocks, Bonds, Commodities, Real Estate
        self.universe = ['SPY', 'EFA', 'IEF', 'DBC', 'VNQ']
        self.cash_proxy = 'BIL'

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        sma_ratio = calc_sma10_ratio(prices)
        alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        for i in range(10, len(prices)):
            date = prices.index[i]
            row = sma_ratio.iloc[i]
            
            invested_weight = 1.0 / len(self.universe)
            cash_weight = 0.0
            
            for asset in self.universe:
                val = row[asset]
                if pd.notna(val) and val > 0:
                    alloc.loc[date, asset] += invested_weight
                else:
                    cash_weight += invested_weight
                    
            if cash_weight > 0:
                alloc.loc[date, self.cash_proxy] += cash_weight
                
        return alloc


# DELETED 2026-07-28: GTAA_G13_Moderate and GTAA_G13_Aggressive.
#
# Neither 2026-07-28 audit could tie the 13-asset universe or the top-6 overlay to any
# published Faber variant, so both were `cannot verify` — two unverifiable variants on the
# SAME universe (rho 0.897), which is one degree of freedom, not two. The Aggressive one was
# worse: it ranked with Keller's 13612U score and filtered with Faber's 10-month SMA, a hybrid
# neither author published.
#
# GTAA_G5 is retained and is the faithful one: it maps cleanly onto the published five-asset
# timing model, and it is now the only Faber entry in the registry.
