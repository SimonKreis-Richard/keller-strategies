"""
Single-module dual momentum on a leverage-admissible universe — CUSTOM.

Wrap pattern: relative momentum picks the best 12-month performer in the universe, absolute
momentum holds it only if its 12-month return beats the T-bill (BIL) return, otherwise BIL.
The signal runs on 1x assets, then the equity pick is mapped to its LETF while BIL is held
1x — defence is never levered.

The *mechanism* is Antonacci's, but **the universe is not**. Antonacci's modules pair assets
that are economically opposed (US vs ex-US equity, high yield vs credit, equity vs mortgage
REITs, gold vs long Treasuries). `[SPY, QQQ, IWM]` is three slices of the same market, chosen
here for one reason only: they are the assets that execute at a uniform 2x multiple (see
common/letf_mapper.py, which enforces it at construction). That is a leverage constraint
dictating an investment universe, which is why this is labelled `custom` and the composite in
strategies/gem.py is not.

  G3 [SPY, QQQ, IWM] -> 2x and 3x   (G2 [SPY, QQQ] was deleted 2026-07-28: rho 0.92)

The four-module composite could not be levered at all: its modules trade VEA/HYG/LQD/REM/VNQ,
none of which has an admissible leveraged product, so only SPY/GLD/TLT ever levered and the
effective leverage swung with the monthly module draws.
"""
from strategies.base import BaseStrategy
from common.letf_mapper import LETFMapper, assert_no_dual_role_mixes_the_sleeve
import pandas as pd


def _lev_str(leverage):
    s = str(leverage).upper()
    return s if s.endswith('X') else s + 'X'


class DMLeveraged(BaseStrategy):
    """Single-module dual momentum (relative + absolute vs T-bill), executed via LETFs."""

    #: A universe chosen by what has leveraged products, not by what diversifies.
    fidelity = 'custom'
    source = ('departs from Antonacci, SSRN 2042750, Table 10 — the modules whose assets '
              'have no admissible uniform-ratio LETF are dropped, which is a universe '
              'leverage forced and nobody published')

    def __init__(self, universe, leverage=3, cash='BIL'):
        n = len(universe)
        super().__init__(f"DM_G{n}_Leveraged_{_lev_str(leverage)}")
        self.universe = list(universe)
        self.cash = cash
        self.leverage = leverage
        # Reject any universe that cannot execute uniformly at this ratio (the cash
        # bucket is excluded — it is held 1x by design).
        LETFMapper(leverage=leverage).validate_universe(
            set(self.universe) - {cash}, self.name)
        # RULE 1's other half: a universe that also names the cash bucket would hold it at 1x
        # beside levered siblings. DM's vocabulary is `universe`/`cash` rather than
        # `offensive`/`defensive`, which is exactly why the guard takes the sets explicitly.
        assert_no_dual_role_mixes_the_sleeve(self.name, self.universe, {cash}, leverage)

    def sleeves(self):
        # The traded book holds the LETF images, not the 1x signal assets — coverage and
        # the ledger both need the real tickers. BIL is the sole defensive bucket and is
        # held at 1x; the absolute-momentum test against it is the de-risking rule, so
        # there is no separate canary.
        mapper = LETFMapper(leverage=self.leverage)
        risky = set(self.universe) - {self.cash}
        images = {mapper.get_letf(a) for a in risky}
        return {'offensive': risky | {i for i in images if i},
                'defensive': {self.cash},
                'canary': []}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        ret12 = prices.pct_change(12)
        alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for i in range(12, len(prices)):
            date = prices.index[i]
            row = ret12.iloc[i]
            scores = {a: row[a] for a in self.universe if a in row.index and pd.notna(row[a])}
            if not scores:
                continue
            best = max(scores.items(), key=lambda kv: kv[1])[0]
            cash_ret = row[self.cash] if self.cash in row.index else float('nan')
            # Absolute momentum: hold the winner only if it beats T-bills, else hold cash.
            if pd.notna(cash_ret) and scores[best] > cash_ret:
                alloc.loc[date, best] = 1.0
            else:
                alloc.loc[date, self.cash] = 1.0
        mapper = LETFMapper(leverage=self.leverage)
        # Leverage the equity sleeve only; the cash bucket (BIL) is held 1x.
        return mapper.translate(alloc, prices, set(self.universe))


# Factory functions
# DELETED 2026-07-28: the G2 size (SPY + QQQ). rho(SPY, QQQ) is 0.92 in monthly returns, so
# a momentum "selection" between the two is not a selection; the variant collapsed to levered
# US large cap, on or off, which is the one thing these families exist NOT to be. Its measured
# correlation with its own G3 twin was 0.93 — the extra ticker changed almost nothing either.
# G3 (widest all-equity) and G4 (adds GLD, rho 0.08 to SPY — the only genuine diversifier
# available at a uniform 2x) are retained. Both are CUSTOM universes: the signal math is the
# paper's, the asset lists were invented here to satisfy leverage homogeneity.

def DM_G3_Leveraged(leverage=2):
    """Widest all-equity universe. Defaults to 2x — see the note on defaults below."""
    return DMLeveraged(['SPY', 'QQQ', 'IWM'], leverage)


# ---------------------------------------------------------------------------------------- #
# DEFAULTS. Every `*_Leveraged(leverage=...)` factory defaults to **2x**. Until 2026-07-29
# BAA's and DM's defaulted to 3 while HAA's and DAA's defaulted to 2 — harmless while the
# registry passed the ratio explicitly, and a trap the moment `leverage=3` started implying
# `role='exploratory'`: a caller relying on the default would silently build an entry excluded
# from the selection statistics. The 3x entries have their own no-argument factories below.
#
# 3x UNIVERSES, added 2026-07-29. `role='exploratory'` is derived from `leverage` on
# LeveragedWrapMixin, so every entry below is measured and reported in full while being
# excluded from the selection statistics — nobody in this project intends to hold 3x.
#
# THE RATIO LADDER IS COMPLETE ONLY AT G3, and this asymmetry must not be papered over:
#
#     universe                     2x            3x         comparable?
#     [SPY, QQQ, IWM]        SSO QLD UWM   UPRO TQQQ TNA     YES - true like-for-like
#     [SPY, QQQ, IWM, GLD]   + UGL         no 3x gold        no
#     [.., TLT] / [.., EEM]  UBT/EET under the $100M floor   3x only
#
# So `*_G3_Leveraged_2X` vs `*_G3_Leveraged_3X` is the ONLY pairing in this repository where
# the leverage level is the single changed variable. Any conclusion about ratio drawn from a
# G4-vs-G5 comparison is confounded by the universe, because the universes differ.
#
# NAMING HAZARD, stated because the convention hides it: `G{n}` counts offensive assets, and
# at n=4 the admissible universe DIFFERS BY RATIO. `*_G4_Leveraged_2X` is [.., GLD] (gold has
# no 3x product); `BAA_G4_Leveraged_3X` is [.., EEM]. Same name shape, different asset lists.
# Read the factory, not the key.
# ---------------------------------------------------------------------------------------- #

def DM_G3_Leveraged_3X():
    """The 3x half of the only clean ratio pair. Twins DM_G3_Leveraged_2X exactly."""
    return DMLeveraged(['SPY', 'QQQ', 'IWM'], 3)


def DM_G5_Leveraged_3X():
    """Widest 3x universe: adds TLT and EEM. Dual momentum holds the single winner, so there
    is no selection count and no cash ladder — the absolute-momentum test against BIL is the
    whole de-risking rule.

    Because it holds ONE asset, this is the entry most likely to sit 100% in TMF at 3x for
    months at a stretch. That makes it the sharpest available measurement of what a levered
    long-duration position does to a portfolio, and the least advisable thing in the registry.
    """
    return DMLeveraged(['SPY', 'QQQ', 'IWM', 'TLT', 'EEM'], 3)


#: Compatibility aliases for the pre-2026-07-29 names.
GEMLeveraged = DMLeveraged
GEM_G3_Leveraged = DM_G3_Leveraged
