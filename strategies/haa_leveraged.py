"""
HAA leveraged sized variants: HAA_G3, HAA_G4.

ADDED 2026-07-29, and the reason is worth stating before the code, because this is the one
family that was missing rather than one that had to be cut.

The four rules in `common/letf_mapper.py` are what decide whether a family may have a
leveraged variant at all. HAA passes all of them, and passes RULE 4 — *a wrap may change
what is HELD, never what decides to DE-RISK* — more cleanly than anything else registered:

  * **The canary is TIP, and it is exogenous.** It is not in the offensive universe, so no
    restriction of that universe can touch it. `TIP <= 0 -> 100% defensive` behaves exactly
    the same whether the offensive sleeve has twelve members or three. This is the property
    VAA and PAA lack, and lacking it is why their wraps were deleted the same day.

  * **The absolute-momentum filter has no denominator.** HAA replaces any Top-X pick whose
    own score is <= 0 with the best defensive asset — a PER-ASSET test, not a breadth count
    over N. Shrinking the universe changes which assets are eligible; it does not move a
    threshold. Compare PAA, whose `CF = (N - n_pos) / (0.5 * N)` is rewritten wholesale the
    moment N changes.

  * **TO = NO/2 is Keller's own rule here.** The other wraps set `T = n // 2` because
    something had to be chosen, and in DAA that invented parameter turned out to also govern
    the cash ladder (RULE 5, which is what removed DAA_G3). In HAA, NO/2 is what the paper
    does: HAA-12 takes TO=6 of NO=12, HAA-8 takes TO=4 of NO=8. So `n // 2` is not a
    liberty taken here, it is the published ratio, and it governs selection only — HAA has
    no cash ladder for it to break.

  * **The defensive sleeve is BIL/IEF, chosen between by momentum, and held at 1x** (RULE 3).
    IEF is dual-role in the larger HAA universes; it is not in these, so no dual-role asset
    is forced to 1x alongside levered siblings.

WHAT IS STILL CUSTOM. The asset lists. Nobody published a three- or four-asset HAA, and
nobody published a levered one. The signal math is Keller's, unchanged, on a universe this
repository invented to satisfy leverage homogeneity. `LeveragedWrapMixin` hard-codes
`fidelity = 'custom'` so this can never inherit HAA's `faithful` claim.

  G3 [SPY, QQQ, IWM]       -> 2x and 3x admissible
  G4 [SPY, QQQ, IWM, GLD]  -> 2x ONLY (no 3x gold product exists)

Only the 2x sizes are registered, on the same evidence as every other family: the 3x twins
measured rho 0.996-0.999 against their 2x siblings, so they add a rescaling and not an
observation — and no 3x product predates 2008-11, meaning no 3x drawdown here has ever seen
a bear market. The factories still accept `leverage=3` for ad-hoc study.

An honest caveat on G3, which applies to every G3 in the registry and is not specific to
HAA: SPY, QQQ and IWM correlate 0.911 / 0.886 / 0.789 with one another in monthly returns.
A Top-1 selection among three assets that close together is a thin selection. It clears the
bar that removed the G2 size (rho 0.911 vs the 0.92 that made SPY+QQQ "not a choice") but
not by much, and the honest reading is that G3 is a levered US-equity timing model rather
than a diversified one. G4's GLD, at rho 0.04-0.08 against all three, is the only asset
available at a uniform 2x that changes the portfolio's character rather than its beta.
"""
from strategies.haa import HAA_12
from strategies.base import LeveragedWrapMixin
from common.letf_mapper import (LETFMapper, assert_protection_survives_restriction,
                                assert_no_dual_role_mixes_the_sleeve)


def _lev_str(leverage):
    s = str(leverage).upper()
    return s if s.endswith('X') else s + 'X'


class HAALeveraged(LeveragedWrapMixin, HAA_12):
    """Canonical HAA signal on a leverageable universe, executed via LETFs (defence 1x).

    Subclasses HAA_12 rather than HAABase for the same reason the DAA and BAA wraps
    subclass their own G12: `LeveragedWrapMixin.source` derives the citation from
    `super().source`, and only the concrete published variants carry one. Hanging the wrap
    off the abstract base would give it an empty citation, which
    `tests/test_anchors.py` refuses — correctly.
    """

    def __init__(self, universe, leverage=2):
        super().__init__()
        n = len(universe)
        self.name = f'HAA_G{n}_Leveraged_{_lev_str(leverage)}'
        self.offensive = list(universe)
        # TO = NO/2 is HAA's published ratio, not a parameter invented for the wrap.
        self.TO = max(1, n // 2)
        self.leverage = leverage
        # RULE 1: every offensive asset must execute at the SAME multiple. The defensive
        # names are subtracted first because they are held at 1x by design and are never
        # mapped, so they must not be judged against the LETF map.
        LETFMapper(leverage=leverage).validate_universe(
            set(self.offensive) - set(self.defensive), self.name)
        # RULE 5: a no-op for HAA (it has no `B`, so no cash ladder to collapse). Called
        # anyway, so that a future HAA variant which grows one cannot skip the check by
        # inheriting a constructor that never looked.
        assert_protection_survives_restriction(self)
        # RULE 1, the half validate_universe structurally cannot see: a dual-role asset is
        # excluded from leverage BEFORE that check runs, so it validates a sleeve it is not
        # describing. HAA's defensive basket is BIL/IEF, so this cannot fire on the registered
        # universes — it is here so a future one cannot introduce a mixed sleeve silently.
        assert_no_dual_role_mixes_the_sleeve(self.name, self.offensive, self.defensive,
                                            leverage)

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        signal_alloc = super().generate_allocations(prices, scores_13612w, ret_12m, ret_3m)
        mapper = LETFMapper(leverage=self.leverage)
        leverage_assets = set(self.offensive) - set(self.defensive)
        return mapper.translate(signal_alloc, prices, leverage_assets)


def HAA_G3_Leveraged(leverage=2):
    """Widest all-equity universe executable at a uniform multiple. TO=1."""
    return HAALeveraged(['SPY', 'QQQ', 'IWM'], leverage)




def HAA_G4_Leveraged(leverage=2):
    """2x only — GLD has no 3x product, so a 3x G4 would be a 3x/2x hybrid. TO=2."""
    return HAALeveraged(['SPY', 'QQQ', 'IWM', 'GLD'], leverage)


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

def HAA_G3_Leveraged_3X():
    """The 3x half of the only clean ratio pair. Twins HAA_G3_Leveraged_2X exactly."""
    return HAALeveraged(['SPY', 'QQQ', 'IWM'], 3)


def HAA_G5_Leveraged_3X():
    """Widest universe admissible at ANY ratio: adds TLT and EEM, both 3x-only.

    TLT is purely offensive in HAA_G12 (its defensive basket is BIL/IEF), so it maps to TMF
    without mixing the sleeve. That is a real risk position and not a defensive one: TMF's
    measured maximum drawdown since 2009-04 is -91.6%, and rho(SPY, TLT) in monthly returns has
    been POSITIVE since 2020. It is here to be MEASURED, not endorsed. TO = 5//2 = 2.
    """
    return HAALeveraged(['SPY', 'QQQ', 'IWM', 'TLT', 'EEM'], 3)
