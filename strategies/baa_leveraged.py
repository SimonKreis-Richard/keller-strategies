"""
BAA leveraged sized variants: BAA_G2, BAA_G3, BAA_G4.

Wrap pattern: each variant subclasses the canonical BAA_G12 and runs its exact signal
logic (SMA12 selection, 4-asset 13612W canary SPY/VWO/VEA/BND, absolute defensive filter
vs BIL) on a restricted offensive universe, then maps the whole offensive sleeve to LETFs
while the defensive sleeve is held unleveraged at 1x.

Admissible universes are those where EVERY offensive asset executes at the SAME multiple
(see common/letf_mapper.py). `validate_universe` enforces this at construction time:

  G2 [SPY, QQQ]            -> 2x and 3x
  G3 [SPY, QQQ, IWM]       -> 2x and 3x
  G4 [SPY, QQQ, IWM, GLD]  -> 2x ONLY (no 3x gold product exists)

Note that BAA's defensive list contains TLT and DBC. Any offensive universe including a
dual-role asset would see it excluded from leverage and held 1x alongside leveraged
siblings — another mixed-ratio sleeve. The admissible universes above avoid this by
construction, and validate_universe checks the post-exclusion set.

The former G6/G8 universes and the full-universe BAA_Leveraged wrap were removed for
mixed-ratio offensive sleeves. Custom *sizes* of the faithful BAA algorithm.
"""
from strategies.baa import BAA_G12
from strategies.base import LeveragedWrapMixin
from common.letf_mapper import LETFMapper, assert_no_dual_role_mixes_the_sleeve


def _lev_str(leverage):
    s = str(leverage).upper()
    return s if s.endswith('X') else s + 'X'


class BAALeveraged(LeveragedWrapMixin, BAA_G12):
    """Canonical BAA signal on a leverageable universe, executed via LETFs (defence 1x)."""

    def __init__(self, universe, leverage=3):
        super().__init__()
        n = len(universe)
        self.offensive = list(universe)
        # TO = NO/2. THIS IS A CHOICE, and it is the last undocumented liberty in the
        # leveraged subsystem — recorded here on 2026-07-29 rather than left implicit.
        #
        # Keller publishes TWO ratios for BAA and they disagree: BAA-G12 takes TO=6 of NO=12
        # (one half), BAA-G4 takes TO=1 of NO=4 (one quarter). A wrap has to pick one. NO/2
        # is chosen, for a reason that is about this universe rather than about performance:
        #
        #   At n=3 the question does not arise — both ratios floor to TO=1.
        #   At n=4 they differ, and one quarter means holding exactly ONE of SPY/QQQ/IWM/GLD.
        #   Three of those four are the same bet (rho 0.79-0.91); GLD is the only asset in
        #   the whole 2x-admissible set that diversifies (rho 0.04-0.08). TO=1 would spend
        #   roughly half its months holding gold ALONE at 2x, and the other half holding a
        #   single levered equity sleeve. TO=2 lets the diversifier be held ALONGSIDE an
        #   equity leg, which is the only reason GLD was worth admitting to the universe.
        #
        # Measured, for information and NOT as the justification: over 2010-02..2026-06,
        # TO=1 gives CAGR 15.38% / MaxDD -24.48% / Sortino 1.18, TO=2 gives 14.53% /
        # -18.02% / 1.38, rho 0.879. TO=2 is better here. That is deliberately not the
        # argument: choosing a parameter by re-running the backtest is the degree of freedom
        # this repository spends most of its guards refusing, and it would be no more
        # defensible for a wrap than it was for the sixteen entries deleted on 2026-07-28.
        self.TO = max(1, n // 2)
        self.name = f"BAA_G{n}_Leveraged_{_lev_str(leverage)}"
        self.leverage = leverage
        # Reject any universe that cannot execute uniformly at this ratio.
        LETFMapper(leverage=leverage).validate_universe(
            set(self.offensive) - set(self.defensive), self.name)
        # RULE 1 again, and for BAA this one BITES. Until 2026-07-29 the comment here claimed
        # validate_universe "caught dual-role assets too". It does not, and cannot: it is handed
        # `offensive - defensive`, from which a dual-role asset has already been removed. BAA's
        # defensive basket holds TLT and DBC, so `BAALeveraged(['SPY','QQQ','IWM','TLT'], 3)`
        # validated cleanly while describing a sleeve that holds TLT at 1x beside TNA at 3x.
        # Only the choice of factories below prevented it, which is not enforcement.
        assert_no_dual_role_mixes_the_sleeve(self.name, self.offensive, self.defensive,
                                            leverage)

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        signal_alloc = super().generate_allocations(prices, scores_13612w, ret_12m, ret_3m)
        mapper = LETFMapper(leverage=self.leverage)
        leverage_assets = set(self.offensive) - set(self.defensive)
        return mapper.translate(signal_alloc, prices, leverage_assets)


# Factory functions for each admissible universe
# DELETED 2026-07-28: the G2 size (SPY + QQQ). rho(SPY, QQQ) is 0.92 in monthly returns, so
# a momentum "selection" between the two is not a selection; the variant collapsed to levered
# US large cap, on or off, which is the one thing these families exist NOT to be. Its measured
# correlation with its own G3 twin was 0.93 — the extra ticker changed almost nothing either.
# G3 (widest all-equity) and G4 (adds GLD, rho 0.08 to SPY — the only genuine diversifier
# available at a uniform 2x) are retained. Both are CUSTOM universes: the signal math is the
# paper's, the asset lists were invented here to satisfy leverage homogeneity.

def BAA_G3_Leveraged(leverage=2):
    """Widest all-equity universe. Defaults to 2x — see the note on defaults below."""
    return BAALeveraged(['SPY', 'QQQ', 'IWM'], leverage)



def BAA_G4_Leveraged(leverage=2):
    """2x only — GLD has no 3x product, so a 3x G4 would be a 3x/2x hybrid."""
    return BAALeveraged(['SPY', 'QQQ', 'IWM', 'GLD'], leverage)


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

def BAA_G3_Leveraged_3X():
    """The 3x half of the only clean ratio pair. Twins BAA_G3_Leveraged_2X exactly."""
    return BAALeveraged(['SPY', 'QQQ', 'IWM'], 3)


def BAA_G4_Leveraged_3X():
    """BAA's widest admissible 3x universe, and it is EEM rather than TLT.

    BAA is the one family excluded from the TLT and DBC universes, because both sit in its
    defensive basket (TIP/DBC/BIL/IEF/TLT/LQD/BND). A dual-role asset is excluded from leverage
    and held at 1x, so the sleeve would be part-3x and part-1x — refused by
    `assert_no_dual_role_mixes_the_sleeve`. Note `validate_universe` does NOT refuse it: it sees
    only the post-exclusion set. EEM is in no BAA basket, so it maps to EDC cleanly.

    Careful: this is a DIFFERENT universe from BAA_G4_Leveraged_2X, which is [.., GLD].
    """
    return BAALeveraged(['SPY', 'QQQ', 'IWM', 'EEM'], 3)
