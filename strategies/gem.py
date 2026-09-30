"""
Antonacci's dual momentum. One registered entry: GEM, the single-module flagship.

DELETED 2026-09-23: `DM_G8_Composite` (class `DMComposite`), Antonacci's equally-weighted
four-module composite from *Risk Premia Harvesting Through Dual Momentum* (SSRN 2042750,
2012), Table 10. Recorded here, where the class lived, as every deletion in this registry is.

| paper (Table 10) | asset 1              | asset 2                | defensive       |
|---|---|---|---|
| Equities         | MSCI U.S. -> `SPY`   | MSCI EAFE+ -> `VEA`    | T-bills -> `BIL` |
| Credit Risk      | Hi Yield -> `HYG`    | Credit -> `LQD`        | T-bills -> `BIL` |
| REITs            | Equity REIT -> `VNQ` | Mortgage REIT -> `REM` | T-bills -> `BIL` |
| Stress           | Gold -> `GLD`        | LT Treasuries -> `TLT` | T-bills -> `BIL` |

Each module chose RELATIVELY first on 12-month return (p. 4, p. 6), then held the winner only
if it beat T-bills; 25% per module (§9). The rules were the paper's and were verified against
it on 2026-07-29, when the entry was relabelled from `custom` to `proxy`.

WHY IT WENT. Not a defect: history. The era moved to 2000-01 that day on mutual-fund and index
donors (`common/data_engine.HISTORY_BACKFILL`) and every other published entry reached it.
This one could not: REM (2007-05) is the only product in the registry with no admissible
donor. The candidates are the mortgage REITs still listed, and an equal-weight basket of them
returned -9% through the GFC against REM's -52% — the ones that failed in 2008 are precisely
the ones no longer listed. Kept, it would alone have held the shared window at 2008-07 for
every other row. The owner judged it had never distinguished itself, and deleted it.

Restoring it means restoring REM to `main.TICKERS` and accepting either a 2008-07 window
for the registry or an exclusion from window-setting argued in its own right.
"""

from .base import BaseStrategy
import pandas as pd

class GEMClassic(BaseStrategy):
    """Antonacci's Global Equities Momentum — the single-module flagship, added 2026-07-30.

    This is the strategy people usually MEAN by "dual momentum", and it is a different
    object from the deleted four-module composite (see above): one module, US equity
    against all-world ex-US equity, defending into AGGREGATE BONDS rather than T-bills.
    Source: *Dual Momentum Investing* (McGraw-Hill, 2014) and the published GEM decision
    tree at optimalmomentum.com, whose
    stated assets are the S&P 500, MSCI ACWI ex-US, and the Barclays US Aggregate.

    THE DECISION TREE, in the book's order — which is NOT the 2012 paper's order:

        1. ABSOLUTE first, gauged on the S&P 500 ALONE: is SPY's 12-month return above the
           12-month T-bill return? If not -> aggregate bonds, and the relative comparison
           is never consulted.
        2. RELATIVE second: hold the better 12-month performer of SPY and ACWI ex-US.

    The 2012 paper's equities module (the deleted composite) inverted this: relative first,
    then the absolute test on the WINNER. The two orderings differ in real months — e.g.
    ex-US wins relative while SPY beats bills and ex-US does not: GEM holds ex-US, the
    2012 module holds T-bills. Antonacci is explicit that GEM's absolute gauge is the S&P
    500 as "the barometer of the state of the market", so this class implements the book
    tree and `tests/test_paper_rules.py::TestGEMFlowchart` pins the ordering against the
    mutation that would quietly turn it back into the 2012 module.

    Tickers: SPY / VEU / BND, with BIL as the T-bill gauge. VEU is the Vanguard FTSE
    All-World ex-US — the same fund family n.11 of the VAA paper says Keller actually ran,
    chosen here over iShares ACWX for the same reasons (fees, and VEU's 2007-03 inception
    predates ACWX's 2008-03). BND is spliced from AGG in the data engine, which IS the
    book's index. `proxy`, not `faithful`: published rules, substitute instruments.

    NO LEVERAGED VARIANT EXISTS, and the derivation is short enough to state here:
    RULE 1 (leverage homogeneity) requires the whole offensive sleeve to execute at one
    multiple, and VEU has no admissible LETF at any ratio (EFO, 2x EAFE, ~$27M, is under
    the $100M floor — and it tracks EAFE, not ACWI ex-US, even so). A wrap restricted to
    US tickers with products ([SPY, QQQ, IWM]) is not GEM any more — it is single-module
    dual momentum on a US universe, which is EXACTLY `DM_G3_Leveraged_2X/3X`, already
    registered. The only remaining difference would be defending into BND instead of BIL:
    a parameter twiddle on an existing entry, the class of thing `BAA_G4_T2` was deleted
    for. GEM therefore joins VAA and PAA in having no wrap — theirs under RULE 4, this one
    under RULE 1 plus the duplication bar.
    """

    fidelity = 'proxy'
    source = ('Antonacci, Dual Momentum Investing (McGraw-Hill, 2014), the GEM decision '
              'tree (S&P 500 / ACWI ex-US / US Aggregate; absolute momentum gauged on the '
              'S&P 500 first); ETF stand-ins SPY / VEU / BND, T-bill gauge BIL')

    def __init__(self):
        super().__init__('GEM_G2_Classic')
        self.offensive = ['SPY', 'VEU']
        self.defensive = ['BND']
        # Signal-only: BIL is the absolute-momentum gauge and is never held — GEM defends
        # into aggregate bonds. Declared as canary so coverage counts it (BIL's constructed
        # history reaches the data start, so it never binds).
        self.canary = ['BIL']

    def sleeves(self):
        return {'offensive': set(self.offensive),
                'defensive': set(self.defensive),
                'canary': list(self.canary)}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        ret_12 = prices[['SPY', 'VEU', 'BND', 'BIL']].pct_change(12)

        for i in range(12, len(prices)):
            date = prices.index[i]
            row = ret_12.iloc[i]
            r_spy, r_veu, r_bil = row['SPY'], row['VEU'], row['BIL']

            # Missing inputs default to the defensive asset, the convention the deleted
            # composite used too. Unreachable inside the measured window: coverage trims the
            # start to the first month where all four series (VEU spliced from VGTSX before
            # its 2007-03 inception) have a full warm-up.
            if pd.isna(r_spy) or pd.isna(r_veu) or pd.isna(r_bil):
                if pd.notna(prices.iloc[i]['BND']):
                    alloc.loc[date, 'BND'] = 1.0
                continue

            # 1. Absolute momentum, gauged on SPY alone (the book's tree). `>` strictly:
            #    "does not show positive momentum with respect to Treasury bills" fails
            #    the test, so an exact tie defends. 2. Relative momentum on the survivors;
            #    an exact SPY/VEU tie goes to SPY — arbitrary, stated, measure-zero.
            if r_spy > r_bil:
                alloc.loc[date, 'SPY' if r_spy >= r_veu else 'VEU'] = 1.0
            else:
                alloc.loc[date, 'BND'] = 1.0

        return alloc


# NOTE: the four-module universe executed via LETFs was removed. Of the eight risk assets,
# only SPY, TLT and GLD have any leveraged product — VEA, HYG, LQD, VNQ and REM have none. The
# credit and REIT modules therefore always executed at 1x while the equity and stress modules
# levered, so effective portfolio leverage swung between roughly 1x and 2.75x purely on the
# monthly module draws. Use the uniform-ratio single-module sleeve in
# strategies/gem_leveraged.py.
