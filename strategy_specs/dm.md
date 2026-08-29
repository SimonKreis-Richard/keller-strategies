# DM — Antonacci's dual momentum (composite, and GEM)

| | |
|---|---|
| **Source** | Antonacci, G. (2012), *Risk Premia Harvesting Through Dual Momentum*, SSRN **2042750**. First version 2012-04-18; this version 2016-10-01. First place, 2012 NAAIM Wagner Awards. |
| **Local copy** | [`academic-papers/2012-gem-ssrn-2042750.pdf`](../academic-papers/2012-gem-ssrn-2042750.pdf) (37 pp.) |
| **Implementation** | [`strategies/gem.py`](../strategies/gem.py), [`strategies/gem_leveraged.py`](../strategies/gem_leveraged.py) |
| **Fidelity** | ✅ `DM_G8_Composite` is **proxy**: the paper's rules and universe on ETF substitutes. `DM_G3_Leveraged_2X` is **custom**. |

> Written from the PAPER, not from the code. Page numbers refer to the local PDF.

## Correction, 2026-07-29

This spec previously described the implementation as `GEM_G8_FourModule_Custom`, "❌ NOT a
reproduction", and asserted that it used "three asset pairs Antonacci never proposed". **That
was wrong.** The credit, REIT and gold/Treasury pairs are Antonacci's own, from §§3-6 and
Table 10 of the paper above — the paper this repository already had on disk and cited as the
strategy's "nominal ancestor".

The error was in the *ancestor*, not the algorithm. **Global Equities Momentum (GEM)** is a
different Antonacci strategy: one module, US versus ex-US equity, defending into aggregate
bonds. The file implemented the 2012 paper's **modular** construction and was then measured
against GEM, which made a faithful implementation look invented. Fixing it required renaming,
not rewriting: not one line of the allocation logic changed.

This is the one direction of labelling error this repository had not yet caught — *under*
claiming. It is worth as much attention as over-claiming, because a wrong `custom` label is
still a wrong label, and it was used to argue the strategy might be deletable.

## Rules, as published

**Dual momentum** (p. 4) is two filters applied in order, per module:

> "First, we choose between our module's non-Treasury bill assets using relative strength
> momentum. If our selected asset does not also show positive momentum with respect to
> Treasury bills (meaning it does not have positive absolute momentum), we select Treasury
> bills as an alternative proxy investment."

**Formation period** (p. 6): twelve months, chosen explicitly —

> "Since twelve months is more common and has lower transaction costs, we will use that
> timeframe."

**The four modules** (§§3-6, summarised in Table 10) and **equal weighting** (§9): the
headline portfolio is "an equally weighted composite of all four dual momentum modules", with
a footnote citing DeMiguel, Garlappi & Uppal (2009) — equal weights beat optimisers once
estimation error is counted.

| module | paper's asset 1 | paper's asset 2 | defensive |
|---|---|---|---|
| Equities | MSCI U.S. | MSCI EAFE+ | T-bills |
| Credit Risk | Hi Yield | Credit | T-bills |
| REITs | Equity REIT | Mortgage REIT | T-bills |
| Stress | Gold | LT Treasuries | T-bills |

The stress module's logic (§6): "Both gold and long-term Treasury bonds may react positively
to weakness in the economy... Gold represents a flight from uncertainty, while Treasuries
represent a flight toward quality."

## Registered variants

| key | universe | type |
|---|---|---|
| `DM_G8_Composite` | SPY/VEA, HYG/LQD, VNQ/REM, GLD/TLT, each defending into BIL, 25% per module | proxy |
| `GEM_G2_Classic` | SPY vs VEU, absolute gauge on SPY vs BIL **first**, defending into BND | proxy |
| `DM_G3_Leveraged_2X` | one module on SPY/QQQ/IWM, defending into BIL, offensive sleeve mapped to 2x LETFs | custom |

## GEM — the single-module flagship (added 2026-07-30)

**Source:** *Dual Momentum Investing* (McGraw-Hill, 2014) and the published GEM decision
tree (optimalmomentum.com). Stated assets: S&P 500, MSCI ACWI ex-US, Barclays US Aggregate.
There is no SSRN PDF of the book in `academic-papers/`, which is why the fidelity pin holds
the book title rather than a paper number.

**The decision tree, in the BOOK's order — which is NOT this paper's order:**

1. **Absolute momentum first, gauged on the S&P 500 alone:** 12-month SPY return vs the
   12-month T-bill return. If SPY fails → **aggregate bonds**, and the relative leg is
   never consulted.
2. **Relative momentum second:** hold the better 12-month performer of SPY and ACWI ex-US.

The 2012 paper's equities module (above) inverts this — relative first, absolute on the
*winner*, defending into T-bills. The orderings disagree observably (e.g. SPY fails the
gate while winning the relative leg: GEM holds BND where the 2012 module holds BIL), and
`tests/test_paper_rules.py::TestGEMFlowchart` pins the book's ordering against the mutation
that would quietly "unify" the two. Tickers: `SPY / VEU / BND`, gauge `BIL`. VEU (Vanguard
FTSE All-World ex-US, 2007-03) stands in for ACWI ex-US — same fund-family preference as
the VAA n.11 correction, and it predates iShares ACWX (2008-03). BND is spliced from AGG,
which is the book's own aggregate index.

**GEM has no leveraged variant, and that is a derivation, not an omission.** RULE 1: VEU
has no LETF above the $100M floor (EFO ~$27M, and it tracks EAFE, not ACWI ex-US). A wrap
restricted to US tickers with products (`SPY/QQQ/IWM`) is no longer GEM — it is
single-module dual momentum on a US universe, which is exactly `DM_G3_Leveraged_2X/3X`,
already registered; the residual difference (defending into BND rather than BIL) is a
parameter twiddle of the kind `BAA_G4_T2` was deleted for. GEM joins VAA and PAA in
having no wrap — theirs under RULE 4, this one under RULE 1 plus the duplication bar.

## Deviations

- **ETF proxies for index series.** The paper runs on MSCI/index data from January 1974; this
  runs on `SPY, VEA, HYG, LQD, VNQ, REM, GLD, TLT, BIL`. The binding inception is VEA
  (2007-07), so the first measurable month is **2008-09** — the sample misses 34 of the
  paper's 38 years, including every regime that motivated the stress module.
- **A module with any missing input allocates its 25% to BIL**, the direction that de-risks.
  The paper has no such case; its indices exist throughout.
- **`DM_G3_Leveraged_2X` is custom and the reason is worth naming.** Antonacci's modules pair
  assets that are economically *opposed*. `{SPY, QQQ, IWM}` is three slices of one market,
  chosen because they are what executes at a uniform 2x multiple. That is a leverage
  constraint dictating an investment universe, which is the opposite of the paper's logic.

## Why it is kept

Beyond fidelity: it is **the most mechanistically distinct entry in the registry** — its
highest correlation against any other variant is ρ = 0.705, where the next most distinct sits
at 0.741. Everything else here is a momentum ranking over overlapping universes; this is the
one row in the regime panel that can move differently. A suite whose participation ratio is
~3 effective independent bets cannot afford to lose its most decorrelated member.

## Cannot verify

- The paper's 1974-2011 sample. Nothing here reaches before 2008.
- Whether the ETF proxies would have reproduced the paper's index-level module selections
  over the overlapping years — in particular `REM` for the mortgage-REIT leg, whose index
  history and ETF history diverge sharply through 2008-2009.
