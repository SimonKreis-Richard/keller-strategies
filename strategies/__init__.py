# Package strategies – the registered models.
#
# This list mirrors main.ALL_STRATEGIES, which carries the admission rule: a paper-faithful
# variant, the same on proxy funds, a leverage-forced custom universe, a control, or a
# benchmark. Sixteen entries were deleted on 2026-07-28; the reason for each is recorded in
# the module where its class used to live.

from .haa import (
    HAA_Simple, HAA_Balanced, HAA_12
)
from .daa import (
    DAA_G12, DAA_G4, DAA_G6
)
from .vaa import (
    VAA_G12, VAA_G4
)
from .baa import (
    BAA_G12, BAA_G4, BAA_SPY
)
from .paa import PAA2
from .gem import DMComposite
from .gtaa import GTAA_5

# Leveraged variants — CUSTOM universes, admitted only by the five rules in
# common/letf_mapper.py. Two of those rules cut this list on 2026-07-29:
#
#   RULE 4 (the wrap may change what is HELD, never what decides to DE-RISK) removed the
#   VAA and PAA wraps outright. Both families protect by counting breadth over their own
#   offensive universe, so shrinking that universe to SPY/QQQ/IWM(/GLD) does not narrow
#   the portfolio, it replaces the risk sensor. See strategies/vaa.py and strategies/paa.py.
#
#   RULE 5 (a cash ladder must survive the restriction) removed DAA_G3: T=1 against B=2
#   rounded a lone dead canary down to zero de-risking.
#
# HAA joined for the same reason the others left — its TIP canary is exogenous and the
# restriction cannot reach it. G2 was already gone: SPY+QQQ correlate 0.911, so its
# "selection" selected nothing. G4 is 2x-only because gold has no 3x product.
from .haa_leveraged import (
    HAA_G3_Leveraged, HAA_G4_Leveraged,
    HAA_G3_Leveraged_3X, HAA_G5_Leveraged_3X
)
from .daa_leveraged import (
    DAA_G3_Leveraged, DAA_G4_Leveraged,
    DAA_G3_Leveraged_3X, DAA_G5_Leveraged_3X
)
from .baa_leveraged import (
    BAA_G3_Leveraged, BAA_G4_Leveraged,
    BAA_G3_Leveraged_3X, BAA_G4_Leveraged_3X
)
from .gem_leveraged import (
    DM_G3_Leveraged, DM_G3_Leveraged_3X, DM_G5_Leveraged_3X
)

__all__ = [
    'HAA_Simple', 'HAA_Balanced', 'HAA_12',
    'DAA_G12', 'DAA_G4', 'DAA_G6',
    'VAA_G12', 'VAA_G4',
    'BAA_G12', 'BAA_G4', 'BAA_SPY',
    'PAA2',
    'DMComposite',
    'GTAA_5',

    # Leveraged G3/G4 variants at 2x (uniform ratio only, custom universes)
    'HAA_G3_Leveraged', 'HAA_G4_Leveraged',
    'DAA_G3_Leveraged', 'DAA_G4_Leveraged',
    'BAA_G3_Leveraged', 'BAA_G4_Leveraged',
    'DM_G3_Leveraged',

    # 3x, role='exploratory' — measured and reported, excluded from the selection statistics.
    # The G3 pairs are the only place in this repo where the leverage LEVEL is the single
    # changed variable; the wider ones are 3x-only because no admissible 2x bond or EM
    # product exists.
    'HAA_G3_Leveraged_3X', 'HAA_G5_Leveraged_3X',
    'DAA_G3_Leveraged_3X', 'DAA_G5_Leveraged_3X',
    'BAA_G3_Leveraged_3X', 'BAA_G4_Leveraged_3X',
    'DM_G3_Leveraged_3X', 'DM_G5_Leveraged_3X',
]
