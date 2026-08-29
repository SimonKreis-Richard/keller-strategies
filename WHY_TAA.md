# Why TAA? A Personal Manifesto

If you are reading this, you might be wondering why someone would spend the time to build and open-source a comprehensive Tactical Asset Allocation (TAA) engine. After all, isn't investing solved? Hasn't the debate been settled by low-cost index funds?

Here is how it all started for me, and why I believe the current mainstream investing narrative is fundamentally flawed.

### The Rise of the Indexing Narrative

I am a millennial. Like many of my generation, I started working, saving, and investing right as low-cost, self-managed brokerages were democratizing finance. The dominant, unshakeable narrative I was taught was simple: buy low-cost index ETFs, hold them forever, and ignore the noise. 

Investing in the stock market responsibly is deeply intimidating for most people. The volatility is anxiety-inducing, and riding the market's rollercoaster takes a psychological toll. In that context, the passive ETF narrative was incredibly comforting. 

I want to be fair: the mainstream indexing narrative is vastly more scientific and well-founded than the era of active mutual-fund stock-picking that preceded it. The statistical studies showing that passive ETFs consistently beat active portfolio managers over the long term are mathematically sound. 

But after a few years of investing, I started looking under the hood. And what I found were glaring flaws in the "buy and hold" dogma.

### The Cracks in the "Buy and Hold" Foundation

ETFs are a brilliant financial invention, but the *methodology* of blind indexing is increasingly problematic:

1. **The Illusion of Diversification:** Market-cap-weighted indices are supposed to offer broad diversification. But with the massive, disproportionate growth of the US tech sector, an S&P 500 or global ETF is now incredibly top-heavy. You aren't buying a diversified global economy; you are buying a concentrated bet on a handful of mega-cap tech stocks. 
2. **The Passive Bubble & Price Discovery:** As Michael Burry and others have pointed out, there is a dark side to the passive indexing craze. When trillions of dollars flow mechanically into index funds regardless of underlying company fundamentals, it artificially inflates valuations. We are witnessing the opacification of true price discovery. The market is going up simply because blind capital is programmed to buy it.
3. **The Dangerous US Bias:** Most "all-in-one" investment products are heavily biased toward the United States. While the US has experienced a miraculous macro cycle of dominance since 2008, history shows that market leadership rotates. We may be nearing the end of that US-centric macro cycle. Blindly weighting your portfolio by market cap—which essentially buys *past* performance—is a simplistic and dangerous approach to the future.
4. **The Threat of the "Lost Decade":** The US market is trading at historically extreme valuations. How would you feel buying and holding through a "lost decade," where your portfolio goes nowhere (or crashes) for 10 to 15 years, like it did from 2000 to 2013? 

The financial industry sells the narrative that passive buy-and-hold is the *only* mathematically sound approach. But in reality? It is intellectual laziness disguised as financial wisdom.

### The Alternative: Tactical Asset Allocation

Simple, systematic Tactical Asset Allocation (TAA) strategies have a long published record of improving **risk-adjusted** outcomes — lower volatility, shallower drawdowns, better Sharpe and Sortino — by mechanically stepping aside when trend and breadth break down. That is the claim the literature supports, and it is the claim this project is built on.

What TAA does **not** reliably do is beat a rising equity index on raw return. This document used to say it did; the engine in this repository says otherwise, and the measurement wins. See the evidence note at the end.

Yet, human complacency is powerful. The financial industry has spent billions convincing retail investors that they cannot beat the market, so nobody pays attention to the quantitative reality. 

TAA solves the most agonizing problem of passive investing: **watching the ship sink but staying onboard because the brochure told you to "buy and hold."** TAA engines monitor the macro environment and mechanically step aside to safety when the market breaks down. 

We have been force-fed the narrative that passive ETFs are the unbeatable endgame of investing. They are not. They are suboptimal. 

While the parallel between the shift from mutual funds to ETFs is interesting, I am not naive enough to suggest TAA will be the "next big thing." It requires discipline, and frankly, most people are too complacent to manage it on their own. TAA isn't a mass-market product to be sold or democratized; it is a specialized tool for those willing to take their financial destiny into their own hands.

I built this project because I refuse to be a passive passenger in an overvalued market. I feel much more comfortable navigating the potential stagnation of the next decade armed with these algorithms than I ever did with a blind "buy and hold" philosophy. 

---

## Evidence note — what this repository actually measures

*Added 2026-07-28, after the audit. This section is deliberately unflattering to the essay
above; the point of building a measurement engine is to let it disagree with you.*

Over **2015-01 → 2024-12**, fills at the next open, 0.10%/side one-way cost, Sharpe and Sortino
net of the realised BIL return:

| | CAGR | MaxDD | Sortino |
|---|---:|---:|---:|
| SPY (buy and hold) | **12.99%** | −23.31% | 1.12 |
| HAA_G12 | 8.72% | **−4.89%** | **1.91** |
| BAA_G12 | 4.97% | −10.47% | 0.86 |
| DAA_G12 | 3.45% | −18.38% | 0.38 |

Three things follow, and all three are worth sitting with:

1. **The risk-adjusted claim survives; the absolute-return claim does not.** HAA_G12 delivered
   a fifth of SPY's drawdown and a materially better Sortino — while giving up 4.3 pp of annual
   return. That is a trade, not a free lunch, and it was made over a decade in which US
   equities essentially only went up.
2. **Post-publication, every strategy tested underperformed SPY on return.** The backtest report
   prints this as its own column precisely so it cannot be quietly averaged into a headline. A
   rule's pre-publication record is its author's search space; only the years after publication
   are evidence about the future, and there are not many of them yet.
3. **The drawdown numbers are the luckiest calendar available.** HAA_G12's −4.89% is the *best
   of 20* equally defensible monthly rebalance schedules; the same signal on the 10th trading
   day drew down −26.6%. See [`KNOWN_GAPS.md`](KNOWN_GAPS.md) §4.

None of that makes the case for TAA in this essay wrong. A strategy that gives up return in a
bull market to avoid riding a lost decade is doing exactly what it says on the tin, and you
cannot test that promise in a sample with no lost decade in it — this one has none. But the
honest version of the argument is *"I am buying insurance and I know what the premium is"*, not
*"this beats the index."*


This repository is my quantitative life raft. I hope it helps you build yours.
