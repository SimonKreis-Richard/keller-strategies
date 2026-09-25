"""
DAA leveraged sized variants: DAA_G3 and DAA_G4.

Wrap pattern: subclasses the canonical DAA_G12 and runs its exact signal logic (13612W
momentum, separate VWO/BND canary, cash-slot ladder) on a restricted offensive universe,
then maps the whole offensive sleeve to LETFs while the defensive sleeve (SHY/IEF/LQD) is
held unleveraged at 1x. DAA qualifies for a wrap at all under RULE 4: its canary is two
exogenous tickers, and no restriction of the offensive universe can reach them.

  G3 [SPY, QQQ, IWM]       -> 2x and 3x
  G4 [SPY, QQQ, IWM, GLD]  -> 2x ONLY (no 3x gold product exists)

HOW T IS CHOSEN, AND HOW THE STORY HAD TO BE CORRECTED TWICE
------------------------------------------------------------
`T` does two jobs in DAA, and that is the whole story. It is the number of offensive slots
held, AND the denominator of the cash ladder `CF = floor(b*T/B) / T`. A wrap sets it from
the size of the restricted universe, so a universe chosen for LETF availability ends up
setting a parameter that governs PROTECTION. That is the trap.

Take `T = n // 2` alone and G3 gives T=1 against B=2, where the BARE floor formula loses
the middle rung:

    cash slots = floor(b*T/B)      b=0     b=1      b=2
      DAA_G12    (T=6, B=2)         0%     50%     100%
      DAA_G4_Lev (T=2, B=2)         0%     50%     100%
      T=1, bare floor               0%      0%     100%   <- the middle rung vanishes

With T=1 on the bare floor, the variant did nothing at all when exactly one canary was
down. VWO went negative in 2020-01 while BND stayed positive, so b held at 1 and it carried
2x equity through the COVID crash for -35.1%; the identical thing happened for three months
in 2011 for -32.0%. `DAA_G4_Leveraged_2X`, which keeps the rung, took -4.9% and -9.4%.

**On 2026-07-29 that was treated as grounds to DELETE the G3 size. Wrong fix, reversed the
same day** in favour of choosing T rather than filtering variants:

    T = max(n // 2, B)

**On 2026-07-30 the RATIONALE was corrected too.** The paragraph that stood here said the
Easy Trading rounding was Keller's only rule and T = max(n//2, B) the only available fix.
DAA note 8 says otherwise, and had said so since 2018: "with T=1, CF is simply b/B". The
paper anticipated exactly this case and legislated for it, and `strategies/daa.py` now
implements it. So a T=1 wrap would resolve all three canary states (0% / 50% / 100%) while
holding one asset, and the -35.1% COVID figure above describes the bare floor this
repository used to run, not the paper. T = max(n//2, B) STAYS — as a CONCENTRATION
convention: it keeps every wrap on the same arithmetic as the registered parents (T >= B)
and holds a less concentrated book than T=1 would. Whether `DAA_G3_Leveraged_2X` should
instead revert to T=1 is now an ordinary custom-universe decision, recorded as open in
strategy_specs/daa.md — and it must not be decided by reading the measured comparison
table, which is precisely the optimisation-on-the-backtest this repository refuses.

Note what none of this is saying. `floor(b*T/B)/T` is Keller's Easy Trading ROUNDING of the
real rule, CF = b/B; the two agree whenever T/B is a whole number, which is why G12 (T=6,
B=2) and G4 (T=2, B=2) were never affected. The defect was never in the paper — and the fix
was in the paper too, which is the half this file previously failed to say.

The former G2/G6/G8 universes and the full-universe DAA_Leveraged wrap were removed
earlier: they needed VGK/VNQ/EWJ/VWO/HYG/LQD/GSG products that are either nonexistent,
below the liquidity floor, or of the wrong multiple, so part of their offensive sleeve
silently executed at 1x. This universe is smaller than Keller's published DAA variants, so
it is a custom *size* of the faithful DAA algorithm — not Keller's G4 universe.
"""
from strategies.daa import DAA_G12
from strategies.base import LeveragedWrapMixin
from common.letf_mapper import (LETFMapper, assert_protection_survives_restriction,
                                assert_no_dual_role_mixes_the_sleeve)


def _lev_str(leverage):
    s = str(leverage).upper()
    return s if s.endswith('X') else s + 'X'


class DAALeveraged(LeveragedWrapMixin, DAA_G12):
    """Canonical DAA signal on a leverageable universe, executed via LETFs (defence 1x)."""

    def __init__(self, universe, leverage=3):
        super().__init__()
        n = len(universe)
        self.offensive = list(universe)
        # T = max(n//2, B). The n//2 term is the wrap's own convention for how many of the
        # restricted universe's assets to hold (L9, mirroring BAA/HAA). The floor at `self.B`
        # was introduced as RULE 5 (the cash-ladder denominator must have at least as many
        # rungs as the canary has states); since 2026-07-30 it is a CONCENTRATION convention
        # rather than a protection repair, because DAA n.8's T=1 rule is now implemented in
        # strategies/daa.py and a T=1 ladder no longer collapses. See the module docstring
        # for the full two-correction history. The guard below is kept as belt-and-braces —
        # it exempts T == 1 and still catches a future 1 < T < B ladder.
        self.T = max(1, n // 2, self.B)
        self.name = f"DAA_G{n}_Leveraged_{_lev_str(leverage)}"
        self.leverage = leverage
        # Reject any universe that cannot execute uniformly at this ratio (RULE 1)...
        LETFMapper(leverage=leverage).validate_universe(
            set(self.offensive) - set(self.defensive), self.name)
        # ...and any universe small enough that T < B collapses the cash ladder (RULE 5).
        assert_protection_survives_restriction(self)
        # RULE 1, the half validate_universe structurally cannot see (dual-role assets are
        # excluded from leverage BEFORE that check runs). DAA's defensive basket is SHY/IEF/LQD,
        # so this cannot fire on the registered universes — it is here so a future one cannot
        # introduce a mixed sleeve silently.
        assert_no_dual_role_mixes_the_sleeve(self.name, self.offensive, self.defensive,
                                            leverage)

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        signal_alloc = self._generate(scores_13612w)
        mapper = LETFMapper(leverage=self.leverage)
        leverage_assets = set(self.offensive) - set(self.defensive)
        return mapper.translate(signal_alloc, prices, leverage_assets)


# DELETED 2026-07-28: the G2 size (SPY + QQQ). rho(SPY, QQQ) is 0.911 in monthly returns, so
# a momentum "selection" between the two is not a selection; the variant collapsed to levered
# US large cap, on or off, which is the one thing these families exist NOT to be.
#
# G4 adds GLD, which correlates 0.08 with SPY and is the only genuine diversifier available
# at a uniform 2x — the other three (SPY/QQQ/IWM) correlate 0.79-0.91 with each other. It is
# a CUSTOM universe: the signal math is the paper's, the asset list was invented here to
# satisfy leverage homogeneity.

def DAA_G3_Leveraged(leverage=2):
    """The widest all-equity universe. RESTORED 2026-07-29 — see the module docstring.

    T = max(1, 3//2, B=2) = 2, so the cash ladder resolves all three canary counts
    (0% / 50% / 100%) exactly as DAA_G12's T=6 ladder does.
    """
    return DAALeveraged(['SPY', 'QQQ', 'IWM'], leverage)




def DAA_G4_Leveraged(leverage=2):
    """2x only — GLD has no 3x product, so a 3x G4 would be a 3x/2x hybrid."""
    return DAALeveraged(['SPY', 'QQQ', 'IWM', 'GLD'], leverage)


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

def DAA_G3_Leveraged_3X():
    """The 3x half of the only clean ratio pair. Twins DAA_G3_Leveraged_2X exactly.

    T = max(1, 3//2, B=2) = 2, so the cash ladder resolves all three canary counts here too.
    """
    return DAALeveraged(['SPY', 'QQQ', 'IWM'], 3)


def DAA_G5_Leveraged_3X():
    """Widest 3x universe: adds TLT and EEM. T = max(2, B=2) = 2.

    TLT is offensive-only in DAA (defensive is SHY/IEF/LQD), so it maps to TMF. See
    HAA_G5_Leveraged_3X for why a 3x TLT sleeve is a measurement rather than a recommendation.
    """
    return DAALeveraged(['SPY', 'QQQ', 'IWM', 'TLT', 'EEM'], 3)
