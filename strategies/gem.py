"""
Antonacci's four-module composite dual momentum, on ETF proxies.

**This file was mislabelled until 2026-07-29.** It called itself `GEM_G8_FourModule_Custom`
and stated that Antonacci "never proposed" its credit, REIT and gold/Treasury pairs. He did.
They are his, from the paper this repository already had on disk: *Risk Premia Harvesting
Through Dual Momentum* (SSRN 2042750, first version 2012-04-18), §§3-6 and Table 10.

The error was in the ancestor, not the algorithm. **Global Equities Momentum (GEM)** is a
*different* Antonacci strategy — one module, US equity versus ex-US equity, defending into
aggregate bonds. Measuring this file against GEM made a faithful implementation look custom.

| paper (Table 10) | module here | asset 1 | asset 2 | defensive |
|---|---|---|---|---|
| Equities    | equities | MSCI U.S. → `SPY`    | MSCI EAFE+ → `VEA`     | T-bills → `BIL` |
| Credit Risk | credit   | Hi Yield → `HYG`     | Credit → `LQD`         | T-bills → `BIL` |
| REITs       | reits    | Equity REIT → `VNQ`  | Mortgage REIT → `REM`  | T-bills → `BIL` |
| Stress      | stress   | Gold → `GLD`         | LT Treasuries → `TLT`  | T-bills → `BIL` |

Every rule below is the paper's, quoted where it matters:

* **Two-stage selection**, p. 4: "First, we choose between our module's non-Treasury bill
  assets using relative strength momentum. If our selected asset does not also show positive
  momentum with respect to Treasury bills ... we select Treasury bills as an alternative."
* **Twelve-month formation period**, p. 6: "Since twelve months is more common and has lower
  transaction costs, we will use that timeframe."
* **Equal weights**, §9: the composite is "an equally weighted composite of all four dual
  momentum modules", with a footnote citing DeMiguel, Garlappi & Uppal (2009) for why equal
  weighting rather than an optimiser.

The remaining gap is the one that earns `fidelity = 'proxy'` rather than `'faithful'`: the
paper runs on index series back to 1974, this runs on ETFs whose oldest constraint (VEA,
2007-07) puts the first measurable month in 2008-09. Same rules, same universe, different
instruments — and a sample missing 34 of the paper's 38 years.
"""

from .base import BaseStrategy
import pandas as pd

class DMComposite(BaseStrategy):
    #: The paper's algorithm on the paper's universe, using ETF substitutes for its index
    #: series. Not 'faithful' only because the instruments differ; no rule was invented here.
    fidelity = 'proxy'
    source = ('Antonacci, Risk Premia Harvesting Through Dual Momentum, SSRN 2042750 '
              '(2012), Table 10 (four-module composite); ETF stand-ins for the index '
              'series the paper trades')

    def __init__(self, name="DM_G8_Composite"):
        super().__init__(name)

        # Antonacci (2012), Table 10. Each module runs dual momentum on its own pair and
        # defends into T-bills independently of the other three.
        self.modules = {
            "equities": {
                "asset1": "SPY",  # MSCI US proxy
                "asset2": "VEA",  # MSCI EAFE+ proxy
                "cash": "BIL"     # 3-Month T-bills
            },
            "credit": {
                "asset1": "HYG",  # High Yield proxy
                "asset2": "LQD",  # Intermediate Credit proxy
                "cash": "BIL"
            },
            "reits": {
                "asset1": "VNQ",  # Equity REIT proxy
                "asset2": "REM",  # Mortgage REIT proxy
                "cash": "BIL"
            },
            "stress": {
                "asset1": "GLD",  # Gold
                "asset2": "TLT",  # Long Treasury Bond
                "cash": "BIL"
            }
        }

        # Collect all unique assets to ensure data covers them
        self.all_assets = set()
        for mod in self.modules.values():
            self.all_assets.update([mod["asset1"], mod["asset2"], mod["cash"]])
        self.assets = list(self.all_assets)

    def sleeves(self):
        # Each module holds asset1, asset2 or its own cash bucket. There is no canary: the
        # de-risking test is per-module absolute momentum against the T-bill return.
        cash = {m['cash'] for m in self.modules.values()}
        risky = set()
        for m in self.modules.values():
            risky.update((m['asset1'], m['asset2']))
        return {'offensive': risky - cash, 'defensive': cash, 'canary': []}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        alloc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # 12-month lookback for relative AND absolute momentum (paper, p. 6)
        lookback = 12

        # Compute 12-month returns
        ret_12 = prices[self.assets].pct_change(lookback)

        module_weight = 1.0 / len(self.modules)  # equally weighted composite: 25% per module

        # Start iterating from the point where we have lookback data
        for i in range(lookback, len(prices)):
            date = prices.index[i]
            current_returns = ret_12.iloc[i]

            # Process each module
            for mod_name, mod_config in self.modules.items():
                asset1 = mod_config["asset1"]
                asset2 = mod_config["asset2"]
                cash_asset = mod_config["cash"]

                ret1 = current_returns[asset1]
                ret2 = current_returns[asset2]
                cash_ret = current_returns[cash_asset]

                # Check for NaNs due to differing ETF inception dates
                # If an asset in a module lacks data, we'll try to default to cash or the other asset
                # Strictly following exact logic requires data for both.
                if pd.isna(ret1) or pd.isna(ret2) or pd.isna(cash_ret):
                    # Default allocating to cash if insufficient data
                    alloc.loc[date, cash_asset] += module_weight
                    continue

                # 1. Relative Momentum: select asset with higher 12-month return.
                # An exact tie goes to asset2 — an ARBITRARY convention, stated so nobody
                # reads meaning into it. The paper never addresses ties, and in floating
                # point over real prices the case is measure-zero.
                selected_asset = asset1 if ret1 > ret2 else asset2
                selected_ret = ret1 if ret1 > ret2 else ret2

                # 2. Absolute Momentum: compare selected asset return to T-bill return
                if selected_ret > cash_ret:
                    # Hold selected asset
                    alloc.loc[date, selected_asset] += module_weight
                else:
                    # Hold T-bills (cash)
                    alloc.loc[date, cash_asset] += module_weight

        return alloc


class GEMClassic(BaseStrategy):
    """Antonacci's Global Equities Momentum — the single-module flagship, added 2026-07-30.

    This is the strategy people usually MEAN by "dual momentum", and it is a different
    object from `DMComposite` above: one module, US equity against all-world ex-US equity,
    defending into AGGREGATE BONDS rather than T-bills. Source: *Dual Momentum Investing*
    (McGraw-Hill, 2014) and the published GEM decision tree at optimalmomentum.com, whose
    stated assets are the S&P 500, MSCI ACWI ex-US, and the Barclays US Aggregate.

    THE DECISION TREE, in the book's order — which is NOT the 2012 paper's order:

        1. ABSOLUTE first, gauged on the S&P 500 ALONE: is SPY's 12-month return above the
           12-month T-bill return? If not -> aggregate bonds, and the relative comparison
           is never consulted.
        2. RELATIVE second: hold the better 12-month performer of SPY and ACWI ex-US.

    The 2012 paper's equities module (`DMComposite` above) inverts this: relative first,
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
        # history reaches 2000, so it never binds; VEU 2007-03 does).
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

            # Missing inputs default to the defensive asset, mirroring DMComposite's
            # convention. Unreachable inside the measured window: coverage trims the start
            # to VEU's 2007-03 inception plus warm-up, where all four series exist.
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


#: Compatibility alias for the pre-2026-07-29 class name. New code should use DMComposite.
GEM = DMComposite

# NOTE: the four-module universe executed via LETFs was removed. Of the eight risk assets,
# only SPY, TLT and GLD have any leveraged product — VEA, HYG, LQD, VNQ and REM have none. The
# credit and REIT modules therefore always executed at 1x while the equity and stress modules
# levered, so effective portfolio leverage swung between roughly 1x and 2.75x purely on the
# monthly module draws. Use the uniform-ratio single-module sleeve in
# strategies/gem_leveraged.py.
