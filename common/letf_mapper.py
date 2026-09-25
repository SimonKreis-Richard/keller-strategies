"""
Signal-asset -> leveraged-ETF mapping, and the admissibility rules that govern it.

WHY THIS FILE CARRIES RULES AT ALL. The leveraged variants are the one part of this
repository with no author to defer to. Keller published HAA, DAA, VAA, BAA and PAA with
their universes and parameters fixed; nobody published a levered version of any of them.
Every choice here — which assets, which ratio, what the defensive sleeve does — was made
in this repo, which means the only thing standing between a variant and a curve-fit is a
rule written down BEFORE the backtest and applied without exception. These are those rules.
They are stated as structure, not as performance: each one can be checked by reading the
code, with no returns in front of you. That is the property that makes them honest.

Four rules decide whether a leveraged variant may exist. RULES 1-2 gate the (signal asset,
ratio) pairs below; RULES 4-5 gate the STRATEGY that wants to use them. A variant failing
any of them must not be constructible — see `validate_universe` and
`assert_protection_survives_restriction`.

RULE 1 — LEVERAGE HOMOGENEITY
    Every asset in a strategy's OFFENSIVE universe must be executable at the SAME
    multiple. A universe that lands partly on 3x products and partly on 2x products
    (or partly on unleveraged originals) has an effective leverage that drifts with
    the monthly signal draw: the backtest stops describing the portfolio actually
    held. Mixed-ratio offensive sleeves are forbidden.

    This is why GLD is absent from MAP_3X: no 3x gold ETF exists anywhere. UGL is a
    2x product. Listing it under MAP_3X silently produced 3x/2x hybrid sleeves.
    Same for GSG, whose only candidate (UCO) is 2x.

    The defensive sleeve and the canary assets are deliberately EXEMPT: defence is
    held unleveraged at 1x by design, and canaries are signal-only, never traded.

RULE 2 — SURVIVORSHIP / LIQUIDITY FLOOR
    LIQUIDITY_FLOOR_USD = $100M of fund AUM.

    The binding risk at retail size is not the bid/ask spread — a five-figure monthly
    rotation is immaterial against even a $30M fund. It is ISSUER CLOSURE. A levered
    ETF charging ~0.95% on $50M grosses ~$475k/yr, which does not cover the swap desk,
    the daily rebalance operations, compliance and market-maker support. Issuers
    routinely liquidate these. A closure mid-rotation forces realisation on the
    issuer's date, not yours, and leaves the strategy with no successor product.

    $100M is deliberately the PERMISSIVE end of a defensible range (a stricter desk
    would use $250M-$500M). It is set low on purpose: every pair rejected below still
    fails at the most generous threshold that can be justified.

RULE 3 — DEFENCE IS NEVER LEVERED
    Enforced by `assert_unlevered_defensive` below. A defensive sleeve exists to cut the
    risk being carried; one that is itself levered raises it and calls the result risk-off.

RULE 4 — A WRAP MAY CHANGE WHAT IS HELD, NEVER WHAT DECIDES TO DE-RISK
    A leveraged wrap restricts the offensive universe to whatever executes at one uniform
    multiple — in practice SPY/QQQ/IWM(/GLD). If the family's de-risking signal is a
    FUNCTION of that universe, restricting it silently rewrites the protection rule, and
    the result is not the paper's strategy under leverage. It is a weaker strategy wearing
    the name.

    PASSES — the signal comes from somewhere the restriction cannot reach:
      BAA  canary SPY/VWO/VEA/BND, B=1        (four exogenous tickers, untouched)
      DAA  canary VWO/BND, B=2                (two exogenous tickers, untouched)
      HAA  canary TIP                         (one exogenous ticker, untouched)
      DM   absolute momentum on the winner    (per-asset test, no denominator)

    FAILS STRUCTURALLY — the signal IS the universe:
      VAA  `self.B = len(offensive)`, breadth over the offensive sleeve itself
      PAA  `N  = len(offensive)`,     breadth over the offensive sleeve itself

    VAA and PAA declare no canary ON PURPOSE (`sleeves()` returns `'canary': []`); breadth
    over a wide, diverse universe is their whole protection mechanism. Count that breadth
    over three US equity ETFs at rho 0.79-0.91 instead of twelve asset classes and the
    denominator no longer measures anything. Their wraps were deleted on 2026-07-29.

    The measured signature of the failure, over 2010-02..2026-06: correlation between each
    wrap and its own unlevered parent — BAA_G4 0.823, DAA_G4 0.794, PAA_G4 0.677,
    VAA_G4 0.593, and VAA_G3 0.376. The exogenous-canary wraps track the strategy they
    claim to run; the endogenous ones drift away from it, worst where the universe is
    smallest. That ordering is a fidelity measurement, not a performance ranking, which is
    why it is allowed to decide admission.

RULE 5 — A CASH LADDER MUST SURVIVE THE RESTRICTION
    Stated CONSTRUCTIVELY:  T = max(n // 2, B)
    Backstopped by `assert_protection_survives_restriction` below.

    DAA and VAA quantise the cash fraction as `floor(b*T/B) / T`. That expression is
    Keller's EASY TRADING rounding of the actual rule, CF = b/B — a convenience so the
    result lands on tradeable fractions of T slots. It is harmless when T >= B and
    destructive when 1 < T < B, because the floor then swallows whole canary states.

    A wrap sets T from the size of the restricted universe, so it can walk into T < B
    without anyone choosing it. With `T = n//2` alone, G3 gives T=1 against B=2, and the
    BARE floor formula loses the middle rung:

        cash slots = floor(b*T/B)      b=0     b=1      b=2
          DAA_G12    (T=6, B=2)         0%     50%     100%
          DAA_G4_Lev (T=2, B=2)         0%     50%     100%
          T=1 vs B=2, bare floor        0%      0%     100%   <- the rung vanishes

    One dead canary then produces NO de-risking where the paper produces half. It is not a
    hypothetical: VWO went negative in 2020-01 and BND did not, so b stayed at 1 and a T=1
    G3 wrap held 2x long through the COVID crash for -35.1%; the same happened for three
    months in 2011 for -25.0%. Its G4 sibling, which keeps the rung, took -4.5% and +7.0%.

    CORRECTION 2026-07-30 — THE PAPER SAW T=1 COMING, AND THIS FILE MISATTRIBUTED THE GAP.
    Earlier revisions reasoned as if the floor formula were all the paper offers at T=1 and
    presented `T = max(n//2, B)` as the only available fix. DAA note 8 legislates the case
    directly: "with T=1, CF is simply b/B" — the ladder survives at T=1, holding the single
    risky slot at 1-CF beside the best defensive asset. That rule is now implemented in
    `strategies/daa.py`, and VAA's own (different!) T=1 rule — §4, all-in-cash when b>=1 —
    in `strategies/vaa.py`; each paper governs its own family. The T=1 figures above were
    measured on the pre-n.8 code and describe the bare floor, not the paper's full ruleset.
    The defect was never in the paper, AND the paper's own fix was sitting in a footnote
    this repository had not implemented — both halves are true and the second was missing.

    WHAT T = max(n//2, B) IS NOW: A CONCENTRATION CONVENTION, not a protection repair. With
    n.8 in place a T=1 DAA wrap would resolve every canary state and be admissible; the
    convention stands anyway (decided 2026-07-30 WITHOUT consulting the measured table
    below, which is exactly why it may cite it) because T >= B keeps every wrap on the same
    arithmetic as the registered parents instead of on a branch no registered entry
    exercises, and holding max(n//2, B) slots keeps the wrap's concentration nearer the
    parent's. History: on 2026-07-29 the T < B case was first handled by DELETING
    `DAA_G3_Leveraged_2X`; that discarded a usable configuration and was reversed the same
    day. The guard below is retained as belt-and-braces for a future family that grows a
    quantised ladder without implementing its paper's T=1 amendment.

    MEASURED, and reported here because the rule was argued from these windows. Same wrap,
    same data, only T differs (2008-03..2026-06):

                          T=1        T=2
        CAGR            22.89%     19.05%
        MaxDD          -46.58%    -35.70%
        Sortino           1.26       1.36
        UPI               1.49       1.46
        vol             33.45%     23.48%
        COVID 2020-02..03  -35.07%    -15.99%
        2011-07..09        -24.98%     -7.24%
        ADVERSE (equity)   -67.05%    -41.33%

    And the mechanism is separable, because the two ladders differ ONLY when b=1:
        b=0 months (111): identical ladder, T=2 holds 2 assets not 1 -> sd 9.04% vs 8.03%
        b=1 months  (74): the rung acts     -> sd 12.35% vs 6.12%, worst -27.2% vs -13.3%
        b=2 months  (35): both 100% cash    -> -1.97% vs -1.57%, i.e. the control holds
    The b=1 column is the restored rung doing exactly what a 50% cash allocation should. The
    CAGR given up comes from the b=0 column, where holding two assets instead of one is
    simply less concentrated in a bull market. Both effects are real; only the first is RULE 5.

REJECTED_MAPPINGS records every pair that was removed and why, so the audit trail
survives and the products are not silently reintroduced.
"""

import pandas as pd
import numpy as np

#: Every leveraged product this repository knows about — the mapped images AND the ones it
#: refuses to map. Kept as one explicit list because `assert_unlevered_defensive` below is
#: only as good as this set: a levered ticker missing from it passes the check silently.
#: `tests/test_anchors.py` asserts it covers every image in both maps.
LEVERAGED_TICKERS = frozenset({
    # 2x, mapped
    'SSO', 'QLD', 'UWM', 'UGL',
    # 3x, mapped
    'UPRO', 'TQQQ', 'TNA', 'TMF', 'EDC',
    # levered products that exist but are NOT admissible as maps (AUM floor, wrong
    # underlying, or simply never mapped) — still levered, so still banned from defence
    'UST', 'UBT', 'UCO', 'EET', 'EFO', 'URE', 'EURL', 'DRN',
})


def assert_no_dual_role_mixes_the_sleeve(name, offensive, defensive, leverage):
    """Raise when a DUAL-ROLE asset would be held at 1x beside levered siblings (RULE 1).

    `validate_universe` cannot catch this, and the gap is structural rather than an oversight:
    it is called on `set(offensive) - set(defensive)`, the post-exclusion set, so an asset that
    was excluded *because* it is dual-role has already been removed before the check runs. The
    universe then validates perfectly while the sleeve it describes is mixed.

    Concretely, and this is why the check exists. BAA's defensive basket is
    `TIP, DBC, BIL, IEF, TLT, LQD, BND`. Give `BAALeveraged` the universe
    `[SPY, QQQ, IWM, TLT]` at 3x and `translate` maps only `SPY/QQQ/IWM` (to UPRO/TQQQ/TNA),
    passing TLT through at 1x on its own ticker. In any month the signal picks TLT as an
    OFFENSIVE holding, the book is part-3x and part-1x — an effective leverage that drifts with
    the monthly draw, which is exactly what RULE 1 forbids. `validate_universe(['SPY','QQQ',
    'IWM'])` returns happily, because TLT is not in what it was handed.

    Until 2026-07-29 the only thing preventing this was the choice of universes in the factory
    list, and `baa_leveraged.py`'s docstring said so out loud: *"The admissible universes above
    avoid this by construction."* A rule enforced by which factories somebody happened to write
    is not enforced. Adding a 3x BAA on a TLT universe is a natural thing to try — the plan for
    the 3x expansion predicted `validate_universe` would refuse it, and it does not.

    A dual-role asset is fine when NOTHING in the sleeve is levered, and fine when the asset is
    purely defensive (never selected as a risk position). What is refused is the mixture.

    Takes the two sets EXPLICITLY rather than sniffing attributes off the strategy. Families
    name these differently — DAA/BAA/HAA use `offensive`/`defensive`, DM uses `universe`/`cash` —
    and `BaseStrategy.sleeves` exists precisely because the engine used to guess among five such
    names and silently returned an empty set for a strategy that used a sixth (audit finding M6).
    A guard that guessed would no-op on the family it failed to recognise, which is the worst
    possible failure mode for a check nobody watches.
    """
    lev = float(leverage or 1.0)
    if lev == 1.0:
        return
    offensive, defensive = set(offensive or ()), set(defensive or ())
    dual = offensive & defensive
    levered = offensive - defensive
    if dual and levered:
        raise ValueError(
            f'{name}: {sorted(dual)} is in BOTH the offensive universe and the '
            f'defensive basket, so it is excluded from leverage and held at 1x, while '
            f'{sorted(levered)} execute at {lev:g}x. That is a mixed-ratio offensive sleeve — '
            f'effective leverage drifts with the monthly signal draw and the backtest stops '
            f'describing the portfolio held (RULE 1). `validate_universe` cannot see this: it '
            f'is given the post-exclusion set, from which {sorted(dual)} has already been '
            f'removed. Choose a universe whose offensive assets are all strictly offensive '
            f'for this family.')


def holds_leveraged_product(strategy):
    """True when ANY sleeve of `strategy` can hold a leveraged product.

    Used by `main.run_backtest` to keep such an entry from setting the shared ranked window.
    Every LETF launched late — UWM (2007-01) is the earliest 2x, TNA (2008-11) the earliest 3x —
    so an entry that holds one always shortens the comparison, and always by years.

    Structural on purpose: it reads the declared tickers, so it cannot be defeated by a
    fidelity label. A 100%-UPRO benchmark has as good a claim to `faithful` as a 100%-SPY one,
    and the fidelity filter alone would therefore have let it cost every other row its 2008.

    The CANARY is excluded from the check, because a canary is read and never traded — a wrap
    reading VWO/BND is not holding anything levered. `assert_unlevered_defensive` is the
    stricter, separate rule that the DEFENSIVE sleeve in particular must stay at 1x.
    """
    try:
        s = strategy.sleeves()
    except Exception:
        return False
    held = set()
    for key in ('offensive', 'defensive'):
        held |= set(s.get(key) or ())
    # Universes are also declared directly on some strategies; a wrap rewrites `self.offensive`
    # and maps it at generate time, so read both rather than trusting one.
    held |= set(getattr(strategy, 'offensive', None) or ())
    held |= set(getattr(strategy, 'assets', None) or ())
    if bool(held & LEVERAGED_TICKERS):
        return True
    # A wrap declares its 1x signal universe and swaps in the LETF at generate time, so the
    # tickers above are the UNLEVERED names. `leverage` is what gives it away.
    return float(getattr(strategy, 'leverage', 1.0) or 1.0) != 1.0


def assert_unlevered_defensive(strategy):
    """Raise unless the strategy's DEFENSIVE sleeve is free of leveraged products.

    **Defence is held at 1x. Always.** The point of a defensive sleeve is to reduce the
    risk being carried; a sleeve that is itself levered increases it, and calls the result
    risk-off. `DAA1_G12` held `UST` — a 2x 7-10y treasury ETF — as one of three defensive
    candidates, so its risk-off state doubled duration risk. It was deleted on 2026-07-28
    and this check exists so nothing like it returns unnoticed.

    A leveraged BOND product is not banned outright: it is banned *in the defensive sleeve*.
    Corporate credit (LQD) sits in the OFFENSIVE universe of DAA and BAA, so a 2x credit
    product would be admissible there under the ordinary homogeneity and liquidity rules —
    it would be a risk position, declared as one.

    Canaries are exempt: they are signal-only and never traded.
    """
    sleeves = strategy.sleeves()
    bad = sorted(set(sleeves.get('defensive') or ()) & LEVERAGED_TICKERS)
    if bad:
        raise ValueError(
            f'{strategy.name}: {bad} is a leveraged product declared in the DEFENSIVE '
            f'sleeve. Defence is held at 1x by design — a levered defensive sleeve raises '
            f'the risk it exists to cut. If the exposure is meant to be a risk position, '
            f'declare it in the offensive universe (as Keller does for LQD) and let the '
            f'homogeneity and liquidity rules judge it there.')


def assert_protection_survives_restriction(strategy):
    """Raise unless the strategy's cash ladder still resolves every canary count (RULE 5).

    Applies to the families that quantise the cash fraction as `floor(b*T/B) / T` — DAA and
    VAA. When `T < B` the floor collapses whole canary states to zero de-risking, so the
    variant reports a protection rule it does not have. See the worked table in the module
    docstring.

    **This is now a BACKSTOP, not the primary mechanism.** RULE 5 is enforced constructively:
    wraps set `T = max(n//2, B)`, so `T < B` is unreachable by construction. This function was
    written as the primary check, and under it `DAA_G3_Leveraged_2X` was deleted — a variant
    that `T = max(n//2, B)` makes perfectly admissible, and which was restored. It is kept
    because a future family could grow a ladder without inheriting that expression, and the
    failure it catches is silent: a wrap that keeps Keller's rounding while losing the rule
    the rounding approximates still produces plausible numbers.

    A strategy with `B <= 1` has no ladder to lose: `b` is 0 or 1 and the rule is binary in
    the paper too (BAA). A strategy with no `B` at all — HAA, DM — never enters this branch.

    Called from every leveraged wrap's constructor, next to `validate_universe`, so an
    inadmissible variant cannot be built rather than merely being absent from the registry.
    """
    T, B = getattr(strategy, 'T', None), getattr(strategy, 'B', None)
    if T is None or B is None or B <= 1:
        return
    if T == 1:
        # Exempt since 2026-07-30: the papers legislate T=1 themselves and the code now
        # implements both versions — DAA n.8 (CF = b/B, graded) in strategies/daa.py and
        # VAA §4 (all-in-cash when b >= 1) in strategies/vaa.py. A T=1 ladder no longer
        # collapses; refusing it would block a configuration the paper resolves.
        return
    if T < B:
        raise ValueError(
            f'{strategy.name}: 1 < T={T} < B={B}, so the cash ladder floor(b*T/B)/T rounds '
            f'{B - T} of the {B + 1} canary states down to no de-risking at all. The '
            f'restricted universe chose T here, nobody did — and the paper\'s own rule is '
            f'CF = b/B, of which floor(b*T/B)/T is only the Easy Trading rounding. A wrap '
            f'that keeps the rounding while losing the rule it approximates is not the '
            f'strategy it claims to be. Use a universe large enough that T >= B (T == 1 is '
            f'exempt: the papers legislate it directly and the code implements them).')


class LETFMapper:
    """Maps signal assets to tradeable LETF equivalents."""

    #: Minimum fund AUM for a leveraged ETF to be eligible (see RULE 2 above).
    LIQUIDITY_FLOOR_USD = 100_000_000

    #: Approximate AUM of the retained products, mid-2026, for the audit trail.
    #: Refresh periodically: a product drifting under LIQUIDITY_FLOOR_USD must be
    #: removed from its map, which in turn invalidates any universe that needs it.
    RETAINED_AUM_USD = {
        'SSO':  5_100_000_000,   # ProShares Ultra S&P500 2x
        'QLD': 13_400_000_000,   # ProShares Ultra QQQ 2x
        'UWM':    220_000_000,   # ProShares Ultra Russell2000 2x
        'UGL':    660_000_000,   # ProShares Ultra Gold 2x
        'UPRO': 3_970_000_000,   # ProShares UltraPro S&P500 3x
        'TQQQ':36_200_000_000,   # ProShares UltraPro QQQ 3x
        'TNA':  1_640_000_000,   # Direxion Small Cap Bull 3x
        'TMF':  3_100_000_000,   # Direxion 20+ Yr Treasury Bull 3x
        'EDC':    190_000_000,   # Direxion MSCI Emerging Markets Bull 3x
    }

    #: (signal asset, ratio) pairs that were audited and REJECTED, with the reason.
    #: Do not reinstate any of these without re-running the audit.
    REJECTED_MAPPINGS = {
        ('GLD', '3x'): "no 3x gold ETF exists; UGL is 2x -> breaks leverage homogeneity",
        ('GSG', '2x'): "UCO tracks WTI crude only, not a broad commodity index -> wrong underlying",
        ('GSG', '3x'): "no 3x commodity ETF exists, and UCO is the wrong underlying anyway",
        ('TLT', '2x'): "UBT ~$65M AUM -> below the $100M floor (closure risk)",
        ('EEM', '2x'): "EET ~$40M AUM -> below the $100M floor (closure risk)",
        ('VGK', '2x'): "EFO ~$27M AUM -> below the $100M floor (closure risk)",
        ('VGK', '3x'): "EURL ~$50M AUM -> below the $100M floor (closure risk)",
        ('VNQ', '2x'): "URE ~$55M AUM -> below the $100M floor (closure risk)",
        ('VNQ', '3x'): "DRN ~$62M AUM -> below the $100M floor (closure risk)",
    }

    #: 3x map. Note the absence of GLD/GSG (no 3x product exists) and of VGK/VNQ
    #: (products exist but sit under the liquidity floor).
    MAP_3X = {
        'SPY': 'UPRO', 'QQQ': 'TQQQ', 'IWM': 'TNA',
        'TLT': 'TMF',  'EEM': 'EDC',
    }
    #: 2x map. TLT and EEM drop out here even though their 3x counterparts survive:
    #: Direxion's 3x treasury/EM funds captured the flows, ProShares' 2x ones did not.
    MAP_2X = {
        'SPY': 'SSO', 'QQQ': 'QLD', 'IWM': 'UWM',
        'GLD': 'UGL',
    }

    def __init__(self, leverage=3):
        """
        Parameters:
        - leverage: 2, 3, '2x', or '3x'
        """
        self.leverage = leverage
        # Handle string or int input
        if isinstance(leverage, int):
            self.lev_str = f"{leverage}x"
        else:
            self.lev_str = str(leverage).lower()

        self.map = self.MAP_3X if '3' in self.lev_str else self.MAP_2X
        self.ratio = '3x' if '3' in self.lev_str else '2x'

    def get_letf(self, signal_asset):
        """
        Get LETF equivalent for a signal asset.
        Returns: LETF ticker or None if no LETF available.
        """
        return self.map.get(signal_asset)

    def unmappable(self, assets):
        """Return the subset of `assets` with no admissible LETF at this ratio."""
        return [a for a in assets if a not in self.map]

    def validate_universe(self, assets, strategy_name=None):
        """
        Enforce RULE 1 on a strategy's OFFENSIVE universe.

        Every offensive asset must map to a product at THIS ratio, so the whole sleeve
        executes at one uniform multiple. Raises ValueError otherwise — a leveraged
        strategy with a mixed-ratio offensive sleeve must not be constructible at all,
        because the failure mode it produces (weights silently falling through to the
        unleveraged original) is invisible in the results.

        Call this from the constructor of every leveraged strategy.
        """
        missing = self.unmappable(assets)
        if not missing:
            return
        label = strategy_name or 'universe'
        reasons = []
        for a in missing:
            why = self.REJECTED_MAPPINGS.get((a, self.ratio), "no leveraged product mapped")
            reasons.append(f"  {a} @ {self.ratio}: {why}")
        raise ValueError(
            f"{label}: offensive universe is not executable at a uniform {self.ratio}. "
            f"Unmappable assets would silently fall through to 1x, producing a "
            f"signal-dependent effective leverage.\n" + "\n".join(reasons)
        )

    def translate(self, alloc_df, prices_df, leverage_assets=None):
        """
        Takes a DataFrame of signal allocations (dates x assets) and returns a new
        DataFrame expressed in tradeable tickers for the leveraged backtest.

        Rules:
        - Offensive asset eligible for leverage WITH a LETF mapping → redirect weight to
          the leveraged ETF.
        - Asset WITHOUT a LETF mapping, OR excluded from leverage (typically the defensive
          sleeve: IEF, SHY, BIL, BND, LQD, HYG, TIP) → held UNLEVERAGED (1x) on its own
          ticker. This matches the intended design: leverage the offence, keep the defence
          at normal exposure (so defensive months earn the real bond/cash return, not 0%).
        - If a mapped LETF ticker is missing from the price data, its weight is dropped
          to cash (with a warning).

        The 1x fall-through is reserved for the DEFENSIVE sleeve and canaries. It must
        never catch an offensive asset: `validate_universe` is what guarantees that, and
        it is called in each leveraged strategy's constructor.

        Parameters:
        - leverage_assets: optional iterable of tickers ELIGIBLE for leverage (typically
          the strategy's offensive sleeve). If provided, only these tickers are mapped to
          LETFs; every other asset — including a momentum asset that also serves
          defensively (e.g. TLT in BAA) — is held unleveraged at 1x. If None, every
          mapped ticker is leveraged (legacy behaviour).
        """
        lev_set = set(leverage_assets) if leverage_assets is not None else None
        final_alloc = pd.DataFrame(0.0, index=alloc_df.index, columns=prices_df.columns)
        missing_letfs = set()

        for col in alloc_df.columns:
            eligible = (lev_set is None) or (col in lev_set)
            if eligible and col in self.map:
                # Offensive sleeve → leveraged ETF
                letf_col = self.map[col]
                if letf_col in final_alloc.columns:
                    final_alloc[letf_col] += alloc_df[col]
                else:
                    missing_letfs.add(letf_col)
            else:
                # Defensive / canary / leverage-excluded → hold the original asset 1x
                if col in final_alloc.columns:
                    final_alloc[col] += alloc_df[col]
                # If the original asset is absent from prices, its weight becomes true cash.

        if missing_letfs:
            print(f"Warning: mapped LETF tickers missing from price data; their weight was dropped to cash: {sorted(missing_letfs)}")

        return final_alloc
