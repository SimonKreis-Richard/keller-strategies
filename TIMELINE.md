# The Genealogy of Tactical Asset Allocation: An Iterative Science

Tactical Asset Allocation (TAA) is not an arbitrary collection of rules or "best guesses." It is an iterative research programme: each model is published, tested and revised in the open. As financial markets have evolved, exposing the flaws of previous models, quantitative researchers have continuously refined their algorithms to adapt. 

Today, this suite is heavily dominated by the research of **Dr. Wouter Keller**. The reason is simple: his application of signal processing mathematics to financial markets fundamentally changed the landscape. While early pioneers built the foundation, we are currently in "Keller's Era." 

Here is the evolutionary timeline of how we got here.

---

### Era 1: The Awakening of "Market Timing" (2006 - 2012)

*Before 2000, passive buy-and-hold (the classic 60/40 portfolio) was the consensus. The dot-com crash (2000-2002) and the Global Financial Crisis (2007-2009) increased interest in mechanical downside protection.*

#### The Standard Model: Faber's Timing Model (GTAA)
* **Author:** Meb Faber
* **Key Innovation:** **A 10-month simple moving average filter, applied to each asset independently**.
* **The Thesis:** Hold an asset while its month-end price is above its 10-month simple moving average, and move that asset's sleeve to cash when it falls below. There is no ranking and no portfolio-level switch (see [`strategy_specs/gtaa.md`](strategy_specs/gtaa.md)).
* **The Pivot (Why it evolved):** The filter is binary and slow to react, and it can whipsaw in sideways or choppy markets.

---

### Era 2: The Hegemony of "Dual Momentum" (2012 - 2016)

*The post-2008 boom saw the rise of global ETFs. Investors wanted more than just downside protection; they wanted outperformance by dynamically rotating into the strongest regions of the world.*

#### The Standard Model: GEM (Global Equities Momentum)
* **Author:** Gary Antonacci
* **Key Innovation:** **Relative + Absolute Momentum**.
* **The Thesis:** Knowing *if* you should invest (Absolute momentum) is only half the battle; you must also know *where* (Relative momentum). GEM compares US Equities, World Equities, and Bonds, buying the single strongest asset over the last 12 months.
* **The Pivot (Why it evolved):** A 12-month lookback is a slow-turning cargo ship. The volatile sideways markets of 2015-2016 exposed a fatal flaw: the algorithm reacted too late, often buying at the local top and selling at the local bottom. More agility was desperately needed.

---

### Era 3: The Engineering Revolution (2016 - 2018)

*The arrival of data scientists and mathematicians like Dr. Wouter Keller shifted the paradigm from "financial observation" to "signal processing mathematics."*

#### The Standard Model: VAA (Vigilant Asset Allocation)
* **Author:** Dr. Wouter Keller
* **Key Innovation:** **Breadth & Momentum Velocity (13612W Score)**.
* **The Thesis:** 
  1. **Velocity:** Stop relying solely on a lagging 12-month lookback. VAA weights the most recent month heavily (a 12-4-2-1 weighting across 1, 3, 6, and 12 months), creating immediate, aggressive reactions to market regime changes.
  2. **Breadth (Crash Protection):** Instead of looking at a single index to gauge risk, VAA checks if the *entire* universe is healthy. If even a single engine in the offensive universe starts smoking (negative absolute momentum), the algorithm ejects completely to defensive bonds.
* **The Pivot (Why it evolved):** VAA is a "Formula 1" car. It delivers exceptionally high performance, but it is exhausting to drive. It generates high turnover and false positives, causing psychological friction and transaction costs for the investor over the long term.

---

### Era 4: The Age of Robustness & Canary Signals (2018 - 2022)

*The quest for stability. How do we keep VAA's crash protection without its nervous twitchiness?*

#### The Standard Model: DAA (Defensive Asset Allocation)
* **Authors:** Dr. Wouter Keller & J.W. Keuning
* **Key Innovation:** **The Canary Universe & Graduated Protection**.
* **The Thesis:**
  1. **Signal Decoupling:** Use *external* "canary in the coal mine" assets (like Emerging Markets `VWO` or US Bonds `BND`) to judge systemic danger, rather than relying on the assets you actually trade.
  2. **Graduated Response:** Market risk is rarely binary (0 or 1). DAA introduces a 50% step-down phase. This is designed to smooth the equity curve and to keep the portfolio from selling everything on a false alarm.
* **Status:** Implemented here as the `DAA_*` entries; see [`strategy_specs/daa.md`](strategy_specs/daa.md).

---

### Era 5: Hybrids & Leverage (2022 - Present)

*The return of inflation and the 2022 decline in both stocks and bonds put traditional 60/40 portfolios under pressure, and made cash a weaker defensive asset in real terms.*

#### The Standard Models: HAA (Hybrid Asset Allocation), BAA (Bold Asset Allocation) & Leveraged Wraps
* **Authors:** Dr. Wouter Keller (HAA, BAA); leveraged "return stacking" ideas from practitioners such as ReSolve.
* **Key Innovation:** **Simpler canaries, and leverage applied only to the offensive sleeve**.
* **The Thesis:**
  1. **HAA & BAA:** HAA uses a single canary, TIPS (`TIP`), with a defensive sleeve that chooses between BIL and IEF. BAA uses a four-asset canary (`SPY, VWO, VEA, BND`) that sends the whole portfolio defensive when any one of them turns negative, and ranks assets with a separate SMA12 measure.
  2. **Leveraged wraps:** Leveraged ETFs (2x, 3x) can hold the offensive sleeve of a published strategy while its own protection rule decides when to de-risk. In this repository only the families whose de-risking signal survives that restriction have a wrap (HAA, BAA, DAA and dual momentum; not VAA or PAA), and 3x wraps are registered for measurement only. See [`LEVERAGE.md`](LEVERAGE.md).
