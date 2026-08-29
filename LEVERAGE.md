# Leverage Justification — Mathematical Foundation & Constraints

> ### 📍 Where to start (2026-07-29)
>
> **[§8](#8-which-strategies-admit-a-leveraged-variant-and-why)** — which families may have a
> leveraged variant at all, and the measured evidence. **[§9](#9-every-liberty-taken--the-complete-register)**
> — the complete register of every design choice made here, since these variants have no
> author to defer to. [`common/letf_mapper.py`](common/letf_mapper.py) is the
> machine-readable source of truth: it holds the five admissibility rules and enforces them
> at construction, so an inadmissible variant cannot be built rather than merely being
> absent from the registry.
>
> The registered set is **G3 and G4 at 2x** for HAA, BAA, DAA and DM (G3), plus **eight 3x
> entries carrying `role='exploratory'`** — measured and ranked, excluded from the selection
> statistics, hidden by default. §8's ratio-ladder table is the payoff: **four G3 pairs out of
> four show 3x raising CAGR while lowering Sortino AND UPI.**
> VAA and PAA have no leveraged variant and are not expected to gain one.
>
> ### ⚠️ Standing caveats
>
> - **The Kelly, volatility-decay and lifecycle arguments in §§1-7 are unaffected by any
>   registry change** — they are properties of leverage itself, not of a strategy.
> - **§1 and §6 were corrected on 2026-07-29 and the correction is substantive.** The
>   "Kelly ≈ 1.2-1.3x" figure that steps 1, 6 and 8 of §10's chain rested on was never
>   measured; it followed from two round numbers asserted in a table. Measured full Kelly on
>   SPY is **4.49x** (half 2.24x). Separately, §6's claim that *"the canary reduces effective
>   volatility"* is **false as stated** — the volatility of the months a canary strategy is
>   actually invested is **higher** than its blended volatility, for all three families.
>   §6's conclusion survives; its mechanism did not.
> - **No 3x backtest here has ever seen a real bear market.** SSO/QLD launched mid-2006, UGL
>   late 2008, all 3x products in 2009.
> - **Sections written before 2026-07-29 have been corrected in place, and the corrections
>   are marked.** Three claims in this document were found to be factually wrong, not merely
>   stale: that BAA "never goes to 100% cash", that the leveraged variants hold "100% cash"
>   defensively, and that they "standardise on VWO+BND" for the canary. Each correction is
>   flagged where it applies. If you read an earlier copy of this file, re-read §§8-9.

## Executive Summary

A 2x leveraged variant is NOT the mathematically optimal leverage level, and **nothing in
this document can tell you what that level is.** Measured on this repository's own data
(2008-06…2026-06), full Kelly on SPY is **4.49x** and half-Kelly **2.24x** — so 2x sits just
below half Kelly. But `f* = μ/σ²` is **linear in the equity risk premium**, and on a
conservative long-run premium of 5-6% half-Kelly falls to 1.0-1.2x, which makes 2x
approximately *full* Kelly. Both readings are the same formula with a different μ.

The defensible summary is therefore narrower than it used to be: **2x is the lowest leverage
available as a single exchange-traded product** (there is no 1.3x LETF, and the choice is 2x
or 3x), it is close to half Kelly under the realised premium of this window, and it is
aggressive under any conservative one. The canary raises the leverage a *strategy* can carry
relative to buy-and-hold — see §6 — but the mechanism is not the one earlier editions
claimed, and §6 has been corrected.

This document justifies every leverage decision in this project. **§9 is the one to read if you only read one**: it lists every liberty taken, why, what the rejected alternative was, and whether code enforces it or only a comment records it.

---

## 1. The Kelly Criterion — Optimal Leverage

> **⚠️ CORRECTED 2026-07-29.** Every earlier edition of this section claimed *"practical
> Kelly ≈ f\*/2 ≈ 1.2-1.3x for equities"*, and steps 1, 6 and 8 of §10's justification chain
> were built on it. **The number was never measured.** It followed from two round numbers
> asserted in the table below — μ ≈ 6%, σ ≈ 15% — which give f\* = 2.67x and half 1.33x. The
> arithmetic was fine; the inputs were invented.
>
> Measured on this repository's own data, full Kelly on SPY is **4.49x** and half-Kelly is
> **2.24x** — roughly 1.7× the figure the whole document rested on. The correction does not
> weaken the case for 2x. It **changes the argument**, and §1.4 explains why the replacement
> is stronger than what it replaces.

### Formula

The Kelly Criterion determines the fraction of capital that maximises the long-run
*geometric* growth rate:

```
f* = μ / σ²

where:
  f* = growth-optimal leverage
  μ  = expected EXCESS return over cash (annualised)
  σ² = variance of returns (annualised)
```

### 1.1 Measured, not assumed

Window **2008-06 … 2026-06, 217 months** — bound by HYG (first tradable 2007-04-11) plus 13
months of momentum warm-up, which is what DAA_G12 and BAA_G12 need before either can be
measured. Every row below therefore covers the same months, including the SPY row. Monthly
total returns on real month-end trading dates; `σ` = monthly stdev (ddof=1) × √12, which is
byte-identical to the `vol` column `common/metrics.py::calculate_metrics` prints in the
report; `μ` is net of **realised BIL** (mean 1.25%/yr over this window), not a flat
assumption.

| Portfolio | μ (excess) | σ | full Kelly f\* | half-Kelly |
|---|---|---|---|---|
| **SPY** (buy & hold) | **11.15%** | **15.76%** | **4.49x** | **2.24x** |
| SPY_Benchmark (as the report prices it) | 11.28% | 16.50% | 4.14x | 2.07x |
| HAA_G12 | 8.48% | 8.46% | 11.85x | 5.92x |
| BAA_G12 | 6.19% | 8.17% | 9.28x | 4.64x |
| DAA_G12 | 6.70% | 8.61% | 9.05x | 4.52x |
| Golden_Butterfly | 6.54% | 8.82% | 8.40x | 4.20x |

Two rows for SPY because the execution convention moves the answer: buy-and-hold month-end
prices give 4.49x, the registry's `SPY_Benchmark` — same asset, priced through the ledger at
next-open — gives 4.14x. **An 8% spread in f\* from a convention choice alone** is the first
warning that this quantity does not deserve two significant figures.

### 1.2 f\* is linear in the one parameter nobody knows

`f* = μ/σ²` is **linear in μ**. σ is estimable from a few years of data and reasonably
stable; μ is the equity risk premium, which is not knowable to a percentage point over any
horizon anyone has. So the entire uncertainty of the answer flows straight from the least
knowable input:

| assumed μ (excess) | f\* | half-Kelly |
|---|---|---|
| 4% | 1.61x | 0.80x |
| 5% | 2.01x | 1.01x |
| 6% | 2.41x | 1.21x |
| 7% | 2.82x | 1.41x |
| 8% | 3.22x | 1.61x |
| **11.15% (measured here)** | **4.49x** | **2.24x** |

**11.15% is not a forecast, it is a description of a bull market.** 2008-06 opens near the
bottom of the financial crisis and the window contains the longest expansion in US history.
If you instead assume a long-run equity risk premium in the 4-6% region — a common range in
the literature, and asserted here rather than derived — then f\* is 1.6-2.4x and half-Kelly is
0.8-1.2x. The old "1.2-1.3x" figure was wrong as a measurement but **it is roughly the right
answer for a 5% premium**: it was a defensible assumption presented as a derivation. The
defect was the presentation, and it mattered because everything downstream cited it as a
result.

### 1.3 Full vs half Kelly — the exact result

`g(f)` is the geometric growth rate at leverage `f`:

```
g(f) = f·μ − f²σ²/2          a downward parabola, zero at f = 0 and f = 2f*
g'(f) = μ − fσ² = 0    →     f* = μ/σ²
```

Two consequences, and the second is the one that matters:

**Half Kelly keeps three quarters of the growth.** Substituting `f = α·f*` and using
`f* = μ/σ²`, every scale factor of Kelly has a growth ratio that depends on nothing else:

```
g(α·f*) / g(f*) = 2α − α²          for every μ and σ

  α = 0.25  ->  43.8% of the growth
  α = 0.50  ->  75.0% of the growth      <- half Kelly
  α = 0.75  ->  93.8% of the growth
  α = 1.00  -> 100.0%
  α = 2.00  ->   0.0%
```

Verified on the measured SPY estimates: `g(4.49) = 25.00%`, `g(2.24) = 18.73%`, ratio
`0.750000`. Half Kelly gives up a quarter of the growth rate for **exactly half** the
volatility (`σ_levered = f·σ`, so halving `f` halves σ precisely) and roughly half the
drawdown.

**Half is a convention, not an optimum.** The `2α − α²` curve is smooth: growth *per unit of
leverage* is `(2 − α)`, which keeps rising as α falls, so there is no point on the curve that
is uniquely best. What makes α = ½ the standard choice is that the trade is legible — three
quarters of the growth for half the risk — and that it leaves room for the error in μ
described below. Any claim that half Kelly is *the* correct fraction is a claim about risk
tolerance dressed as mathematics.

**The curve is dangerously asymmetric.** Under-betting costs growth only. Over-betting costs
growth *and* adds risk, and past `2f*` the growth rate is **negative** while the leverage —
and the drawdown — keeps rising. Since the error in `f*` is dominated by the error in μ, and
μ is unknowable, **you bet below your estimate on purpose.** Half Kelly is not caution
bolted onto the formula; it is what the formula's own shape recommends once you admit μ is
an estimate.

*(The older text justified halving by fat tails, citing MacLean, Thorp & Ziemba. Fat tails
are a real reason to bet less, but they do not produce the factor of exactly one half —
that number is a convention. The 0.75 identity above is exact and needs no distributional
assumption beyond the ones already in `g(f)`.)*

### 1.4 Where this leaves 2x — a weaker claim, honestly stated

On the measured window half-Kelly is 2.07-2.24x, so **2x sits just below half Kelly**. On a
long-run 5-6% risk premium half-Kelly is 1.0-1.2x, and **2x is approximately full Kelly**.
Both readings use the same formula and the same σ; they differ only in μ.

So the honest position is not *"2x is the closest available instrument to the optimum"* — it
is:

> **2x is defensible under this window's realised premium and aggressive under any
> conservative long-run premium.** Which of those you believe is a judgement about the equity
> risk premium, not an output of the Kelly criterion.

That is a smaller claim than the one it replaces, and it is the one the arithmetic supports.
§7's drawdown mathematics and §2's decay term are independent arguments and are unaffected.

**Reference:** MacLean, L.C., Thorp, E.O. & Ziemba, W.T. (2011). "The Kelly Capital Growth
Investment Criterion." World Scientific.

> **Reproducing §1.1.** The measurement is not a committed script, which is the same weakness
> that let the old figure survive — so the convention is stated instead, in enough detail to
> redo in a dozen lines: take `PriceStore.month_end_dates()` (real last trading days, **never**
> a `resample('ME')` label — see the warning in `common/data_engine.py`), price with
> `monthly_adj_close()`, net against `build_rf_series(store, dates, 'BIL')`, then
> `σ = r.std(ddof=1)·√12` and `μ = (r − rf).mean()·12`. Feeding calendar month-end labels to
> `build_rf_series` silently turns every weekend month-end into an assumed flat 3%/yr and
> understates μ by ~0.2pp; that mistake was made and caught while producing this table.

---

## 2. The Laffer Curve of Leverage

Leverage has diminishing returns. Beyond a certain point, the cost of leverage (volatility decay, drawdown amplification) exceeds the benefit (higher returns).

### Volatility Decay Formula

Leveraged ETFs reset daily, causing geometric compounding effects:

```
R_LETF ≈ L × R_underlying - (L² - L) × σ² / 2

where:
  L = leverage multiplier (2 or 3)
  σ = annualized volatility of the underlying
```

The decay term `(L² - L) × σ² / 2` grows **quadratically** with leverage:

| Leverage (L) | L² - L | Decay Multiplier |
|-------------|--------|-----------------|
| 1.0x | 0 | 0 (no decay) |
| 1.3x | 0.39 | 0.39 × σ²/2 |
| 2.0x | 2.00 | 2.00 × σ²/2 |
| 3.0x | 6.00 | 6.00 × σ²/2 |

### Numerical Example (σ = 15%)

| Leverage | Expected Return | Decay Cost | Net Return | Drawdown Risk |
|----------|----------------|-----------|------------|---------------|
| 1.0x | 10.0% | 0.0% | 10.0% | -50% |
| 1.3x | 13.0% | -0.2% | 12.8% | -55% |
| 2.0x | 20.0% | -1.1% | 18.9% | -70% |
| 3.0x | 30.0% | -3.4% | 26.6% | -90% |

**At 3x, the decay costs 3.4% annually** — this is a permanent drag on returns that compounds over time.

### The Optimal Point

The optimal leverage maximizes the **growth rate** (not the return):

```
G(L) = L × μ - L² × σ² / 2

dG/dL = μ - L × σ² = 0

L_optimal = μ / σ² = Kelly
```

Beyond Kelly, each unit of additional leverage **reduces** the growth rate — and past `2f*`
the growth rate is negative.

> **⚠️ CORRECTED 2026-07-29.** This paragraph used to end: *"This is why 3x underperforms 2x
> on a risk-adjusted basis — it is past the optimum."* **That is false on this repository's
> own measurements, and it was the weakest argument in the document.** With SPY's measured
> μ_exc 11.15% and σ 15.76% (§1.1), the growth curve reads:
>
> | f | 1.0x | 1.3x | 1.9x | 2.0x | 2.24x | **3.0x** | 4.49x = f\* | 8.97x = 2f\* |
> |---|---|---|---|---|---|---|---|---|
> | g(f) | 9.90% | 12.39% | 16.69% | 17.32% | 18.73% | **22.26%** | 25.00% | 0.00% |
>
> **g(3.0) > g(2.0).** On these estimates 3x is not past the optimum — it is comfortably
> below it, and grows faster than 2x. The real arguments against 3x are the decay term above,
> the drawdown mathematics in §7, and the fact that f\* itself is unreliable (§1.2): at a 5%
> risk premium f\* is 2.01x and 3x *is* past the optimum. **The case against 3x cannot be made
> from the growth curve alone, and pretending otherwise invited exactly the rebuttal that the
> curve refutes it.** See §7 for the argument that does not depend on μ.

---

## 3. The Opportunity Cost of No Leverage

Choosing not to use leverage is itself a decision with a cost. Over long horizons, the difference between a leveraged and unleveraged portfolio is substantial.

### Compounding Example (100k$ initial, 20 years)

| Strategy | CAGR | Final Value | Difference |
|----------|------|-------------|------------|
| DAA unleveraged | ~12% | ~964k$ | Baseline |
| DAA 2x | ~17% | ~2,311k$ | **+1,347k$** |
| DAA 3x | ~22% | ~5,234k$ | +4,270k$ (but higher risk) |

*(Illustrative compounding arithmetic, not a backtest result — see the audit note at the
top of this document.)*

**The unleveraged investor leaves ~1.3M$ on the table** over 20 years on a 100k$ investment. This is not theoretical — it is the mathematical consequence of compounding at a lower rate.

### Why This Matters

The opportunity cost grows **exponentially** with time. Over 5 years, the difference is modest. Over 20-30 years, it is life-changing. For an investor with a long time horizon, the question is not "can I afford to use leverage?" but "can I afford NOT to?"

---

## 4. Lifecycle Investing — Leveraging When Young

Ayres & Nalebuff (2010), economists at Yale, propose a framework where investors should use leverage **early in their investment lifecycle** when their human capital (future earnings) is large relative to their financial capital.

### The Core Insight

A young investor with 50k$ in savings and 40 years of future earnings has most of their "wealth" locked in future labor income — which is effectively "out of the market." This creates an **implicit under-allocation to equities**.

```
Young investor:
  Financial capital:    50k$    (in the market)
  Human capital:      1,500k$   (future earnings, NOT in the market)
  Total wealth:       1,550k$
  Equity exposure:      3.2%    ← massively under-allocated
```

Using leverage on the financial capital partially corrects this imbalance:

```
With 2x leverage on 50k$:
  Effective equity exposure: 100k$ / 1,550k$ = 6.5%  ← still conservative
```

### Application to Our Strategy

A 2x DAA variant is well-suited for lifecycle investing because:

1. **The canary limits drawdowns** — the primary risk of leverage is managed
2. **Quarterly rebalancing** — low maintenance, fits a long-term approach
3. **Systematic** — removes emotional decision-making
4. **Scalable** — can reduce leverage as financial capital grows relative to human capital

**Reference:** Ayres, I. & Nalebuff, B. (2010). "Lifecycle Investing: A New, Safe, and Audacious Way to Improve the Performance of Your Retirement Portfolio." Basic Books.

---

## 5. Why 2x LETFs and Not 1.3x

### The Instrument Constraint

**This is the section that does the work, and it needs no Kelly figure at all.** Whatever the
growth-optimal leverage is — §1 measures 4.49x on this window and 1.6-2.4x on a conservative
risk premium — the set of things you can *buy* as a single exchange-traded product is fixed:

| Leverage Level | Available Instrument | Notes |
|---------------|---------------------|-------|
| 1.0x | Regular ETFs | No leverage |
| anything between 1.0x and 2.0x | **No LETF exists** | Not available as a product |
| 2.0x | SSO, QLD, UWM, UGL | Retained: all clear the $100M AUM floor |
| 3.0x | UPRO, TQQQ, TNA, TMF, EDC | Retained: all clear the $100M AUM floor |

**There is no LETF between 1x and 2x.** The choice is binary: 2x or 3x. Earlier editions
framed this as *"Kelly suggests 1.2-1.3x, and no such product exists"* — the framing is
withdrawn with the figure (see §1), but **the constraint it described is unaffected**, and it
is the constraint, not the figure, that forces the decision. 1.3x is reachable only on
margin, which is §5's next subsection and a different instrument with different properties.

Earlier revisions of this table listed UBT/EET (2x) and EURL/DRN (3x) as "available,
liquid" — they hold $27M–$65M and fail the floor — and listed **UGL under 3x, which is the
error that produced the mixed-ratio sleeves.** UGL is a 2x product; no 3x gold ETF exists.

The retained set is not symmetric, and the asymmetry drives universe design:

| | 2x | 3x |
|---|---|---|
| US equity (SPY/QQQ/IWM) | ✅ | ✅ |
| Gold (GLD) | ✅ UGL | ❌ nothing exists |
| Long treasury (TLT) | ❌ UBT too small | ✅ TMF |
| Emerging (EEM) | ❌ EET too small | ✅ EDC |

At 3x you can hold three US equity indices and nothing else that diversifies equity beta
in a bull-market regime. Adding gold **forces the drop to 2x**. Leverage ratio and
diversification trade off directly; no configuration delivers both.

### Why LETFs Instead of Margin

A margin loan can be set to any ratio, including the 1.0-2.0x range no LETF covers — and on a
conservative long-run risk premium that range is where half Kelly falls (§1.2). That is a real
advantage of margin and the repository measures it: `README.md`'s margin table reports HAA/DAA/BAA
at a **flat 1.3x and a signal-following 1.3x** against their unlevered baselines. The measured
verdict there is that 1.3x margin buys CAGR and sells everything else: it worsens max drawdown,
Sortino *and* UPI in all six levered rows — so being nearer a conservative Kelly estimate did
**not** translate into a better risk-adjusted result. Leveraged ETFs are nonetheless the preferred vehicle here, for reasons that are about
the instrument rather than the ratio:

1. **Regulatory simplicity** — LETFs are regular exchange-traded products. They do not involve borrowing, margin agreements, or broker-specific credit arrangements. They are treated as standard securities in virtually every jurisdiction.

2. **Registered account compatibility** — LETFs can be held in tax-advantaged accounts (401k, IRA, RRSP, TFSA, ISA, and equivalents worldwide) where margin is typically prohibited. This makes them accessible to investors across all account types.

3. **Lower regulatory risk** — Margin activity (borrowing to invest) can attract scrutiny from tax authorities in many jurisdictions, potentially contributing to professional trader reclassification. With a salary, low trade frequency, and moderate gains, this risk is generally low. However, LETFs eliminate this concern entirely — they are standard ETFs that attract no more attention than holding a regular index fund.

4. **No margin calls** — LETFs cannot trigger margin calls. The maximum loss is the capital invested. Margin can result in losses exceeding the initial investment.

5. **Simplicity** — No credit agreements, no interest payments, no margin maintenance requirements. Buy and hold like any other ETF.

**Nuance on margin risk:** Margin at moderate leverage (1.3x) does NOT automatically classify an investor as professional in most jurisdictions. Tax authorities typically evaluate the overall picture: employment income, trade frequency, gain ratio, and leverage are all considered together. Missing one criterion (leverage) while satisfying the others (salary, low frequency, moderate gains) is unlikely to trigger reclassification. However, LETFs are strictly safer — they eliminate the leverage criterion entirely rather than relying on a favorable holistic assessment.

**Bottom line:** LETFs are a turnkey leverage solution that works everywhere — all account types, all jurisdictions, all investor profiles — without the operational complexity or regulatory ambiguity of margin. Margin is a viable alternative for anyone who wants a ratio between 1x and 2x, which no LETF supplies, and who is comfortable with the slightly elevated regulatory profile — bearing in mind that the repository's own measurement of 1.3x margin worsened every risk-adjusted metric it improved the CAGR of.

### The structural difference: margin does not de-lever by itself

The two instruments are **not** interchangeable at equal nominal leverage, and the gap is
not a detail.

A leveraged-ETF portfolio **de-levers automatically in risk-off**. When the canary flips
and the strategy rotates from UPRO into IEF, exposure falls to 1x — the defensive sleeve
is an ordinary unlevered ETF. Leverage is applied by *what you hold*, so exiting the
offensive sleeve exits the leverage in the same trade.

Flat margin does the opposite. At `LEVERAGE_FACTOR = 1.3` the loan is drawn continuously,
so in a defensive month you have **borrowed money to buy treasuries** — paying the borrow
rate to hold an asset yielding roughly the borrow rate, while remaining levered into
whatever drawdown triggered the defensive signal in the first place. Moving from a
registered account (LETFs) to a margin account and keeping "1.3x" reads like an instrument
swap, but it changes the risk profile.

`MARGIN_FOLLOWS_SIGNAL = True` (the default) reproduces the LETF behaviour:

```
effective_leverage_t = 1 + (LEVERAGE_FACTOR - 1) x offensive_weight_t
interest_t           =     (LEVERAGE_FACTOR - 1) x offensive_weight_t x rate/12
```

Offensive contribution levered, defensive contribution at 1x, interest charged only on
what was actually drawn.

**Re-measured 2026-07-28 by the corrected engine** (fills at the next open, 0.10%/side one-way
per leg, cash in BIL, Sharpe/Sortino net of realised BIL), over **2008-06 → 2026-06** — not a
window anybody chose: the floor is derived and the run ends at the last complete month in the
data. **HYG (2007-04) is what sets this start, and it is why these three cannot be pushed back
further**: no US high-yield ETF exists before it.

**Re-measured again 2026-07-29**, twice: first after the history extension pulled three more
months of the crisis into the sample and deepened every drawdown here, then after METH-001
and METH-002 corrected DAA's selection rule and SMA12's window. HAA_G12 is unmoved by the
second pass — it uses neither — which is the check that the corrections did what they claim
and nothing else.

| Strategy | Mode | CAGR | MaxDD | Sortino | UPI | Avg lev | Turn/yr |
|---|---|---|---|---|---|---|---|
| HAA_G12 | unlevered 1.0x | 9.79% | **−11.19%** | **1.90** | **3.44** | 1.00x | 6.8 |
| HAA_G12 | flat 1.3x | 10.74% | −15.02% | 1.61 | 2.51 | 1.30x | 8.8 |
| HAA_G12 | **signal-following 1.3x** | **10.90%** | **−14.52%** | 1.64 | 2.64 | 1.22x | 8.2 |
| DAA_G12 | unlevered 1.0x | 7.85% | **−18.38%** | **1.39** | **1.10** | 1.00x | 12.2 |
| DAA_G12 | flat 1.3x | 8.21% | −25.36% | 1.13 | 0.75 | 1.30x | 15.8 |
| DAA_G12 | **signal-following 1.3x** | **8.66%** | **−22.02%** | 1.23 | 0.97 | 1.20x | 14.3 |
| BAA_G12 | unlevered 1.0x | 7.34% | **−10.70%** | **1.35** | **1.50** | 1.00x | 10.2 |
| BAA_G12 | flat 1.3x | 7.56% | −18.65% | 1.08 | 0.95 | 1.30x | 13.3 |
| BAA_G12 | **signal-following 1.3x** | **8.05%** | **−14.13%** | 1.28 | 1.37 | 1.14x | 11.5 |

The qualitative conclusion **survives, and got cleaner**: signal-following beats flat margin on
CAGR, max drawdown, Sortino and UPI simultaneously, in all three families, while borrowing less
on average. The one tie that used to sit in this table — HAA's drawdown — disappeared once the
crisis months entered the sample. With a real 2008 in it, de-levering into the defensive sleeve
stops being a rounding difference. That is not alpha: it is the removal of a cost that was
buying no exposure.

**What the corrected numbers add, and the old ones hid:** against the *unlevered* baseline,
1.3x margin raises CAGR by **+0.22 to +1.11 pp** (flat: +0.95 / +0.36 / +0.22;
signal-following: +1.11 / +0.81 / +0.71) and worsens max drawdown, Sortino **and** UPI in
**all six levered rows**. Leverage here buys return and sells risk-adjusted quality.

*(Corrected 2026-07-29: this paragraph read "0.8–1.1 pp", which is the range of the
signal-following rows only, and not quite that — BAA's is +0.71. The range above is read off
the table rather than recalled beside it.)* That is a legitimate
trade to make deliberately; it is not a free improvement, and the pre-audit tables — which
carried no risk-free rate, no per-leg cost, and no unlevered row — made it look like one.

**Four honest caveats:**

1. **A margin account can be called intramonth; an LETF cannot.** Conditioning leverage on
   a monthly signal does nothing between rebalance dates. The maximum loss on an LETF is
   the capital invested; margin can lose more, and can force liquidation at the worst
   moment. This is the one risk the alignment does **not** remove. *Amended 2026-07-30:* the
   ledger still does not model it — `run_ledger` draws a debit balance and capitalises interest
   on it but never compares equity to a maintenance requirement, so every levered figure above
   remains an upper bound. What changed is that the risk is now **priced separately**, in
   `common/margin_sizing.py`. See §5.4.
2. **Turnover rises with leverage** (see the last column): rebalancing a levered book trades
   more notional. This is now charged properly — cost is levied on the executed levered
   notional, per leg — where the pre-audit engine charged it on the 1x signal weights.
3. **Interest is day-counted on the debit balance actually drawn**, settled at each
   rebalance. It is not compounded daily.
4. **These three start 2008-06 and no earlier, because they hold high yield.** HYG is the
   oldest US high-yield ETF in existence (2007-04-11; JNK is 2007-12-04), so this table still
   opens after the October 2007 peak. The unlevered families that do NOT hold high yield —
   HAA, GTAA_G5, the G4 sizes — now reach 2007-03 and can be read through the whole bear.
   The levered variants reach 2008-03 at best (UWM, 2007-01), and 2010-01 for the G4 sizes
   (UGL, 2008-12). **A levered strategy's behaviour entering a crash remains the thing this
   document is least able to tell you about.**

### 5.4 How much margin leverage a model can actually carry — `common/margin_sizing.py`

Everything above measures what margin *did* at a leverage somebody chose. This asks the prior
question: **what leverage survives?** Four independent ceilings, minimum retained, and the output
that matters is which one bound.

**CAP 1 — margin-call survival.** At leverage `f` with maintenance margin `m`, liquidation
follows a decline `d` in the positions held:

```
d_max = (1/f − m) / (1 − m)          f = 1 / (m + d(1−m))
```

The debit balance is not constant through the decline — `ledger.py` capitalises interest — so
with `c = r_b · h/12` accrued over a drawdown of peak-to-trough duration `h`:

```
d_max = 1 − (f−1)(1+c) / (f(1−m))    f = (1+c) / ((1+c) − (1−d)(1−m))
```

`h` is a measurement, which is why `metrics.max_drawdown_months` was added. At `m=0.30`,
`d=0.60`, `r_b=6%`, `h=30`:

| variant | f |
|---|---|
| plain | **1.389** |
| + interest accrual | **1.322** |
| + crisis margin (`m=0.45`) | **1.282** |
| at `m=0.75`, a 3x LETF | **1.111** |

The target decline is `d = k · DD_adj`, and `k` — in multiples of drawdown, default 3 — is the
module's only preference parameter.

**CAP 2 — Kelly, as a gate.** `f* = (μ_e − s·ω_off)/σ²`. The borrow spread enters as a flat
subtraction because it is charged on `(f−1)`, not `f`, and only on the offensive fraction under
`MARGIN_FOLLOWS_SIGNAL=True`. The Sharpe is haircut, and the haircut is **derived**: the
expected maximum Sharpe across N trials under the null is `sd_SR · 1.878` at N=19, where `sd_SR`
is the cross-sectional spread of the trials' own Sharpes — a measurement, not a coefficient. The
more severe of that and Mertens' one-sigma lower bound is used.

**CAP 3 — carry. Requalified as a diagnostic, and it never binds.** It fires when
`r_b ≥ μ − σ²/2`; CAP 2's gate fires when `μ − r_b < σ²`. The first region is a strict subset of
the second, so CAP 3 cannot be the binding constraint. Kept as an input sanity-check and as the
fallback when CAP 2 is non-calculable.

**CAP 4 — borrowing capacity.** A parameter, and frequently the one that actually binds. It must
be the **taxable** account's equity: registered accounts do not permit margin.

**DD_adj — both corrections measured, not marked up.** Sample bias: the 95th percentile of the
maximum drawdown over a 240-month horizon, from a stationary block bootstrap of the model's own
returns. Intra-period: the daily-path drawdown of the **held** allocation, divided by the
month-end one. Both factors are reported separately, so the markup is visible rather than buried
in a coefficient. `DD_adj` is floored at the observed drawdown — a stress shallower than one that
already happened is not a stress.

**Where the drift argument in the request was inverted.** Maintaining target leverage *protects*:
when equity falls, restoring the target requires **selling**. At `m=0.25`, `f=2`, two −20% months
liquidate an unmanaged book at −36% cumulative and a monthly-reset book survives. So
`d_max = (1/f − m)/(1−m)` is already the fully-drifted answer measured from the anchor. What the
static calculation genuinely misses is interest accrual, `m` rising mid-crisis, the un-defended
**sub-monthly** excursion, and a one-sided policy that re-levers on gains but never de-levers on
losses. All four are tested in `tests/test_margin_sizing.py`.

**Measured over the registry, re-run 2026-07-31 and now REPRODUCIBLE.** Until that date this
table came from an ad-hoc script that was not kept, so the repository's answer to its own
central question could not be regenerated from the repository. `common/leverage_advice.py` is
the driver it lacked: the CLI report prints the table as its `SUSTAINABLE MARGIN LEVERAGE`
section, and the dashboard carries the same figure as a `Max margin` column.

*Broker parameters, all of them assumptions —* `m` 30% on an ordinary ETF position and that
base times the **fund's own multiple** for a leveraged one (60% at 2x, 90% at 3x), ×1.5 again
for the crisis variant that is the one reported; `r_b` 6% against a **measured** `r_f` of
1.246%; `k=3`; capacity unsupplied. Not the owner's broker — there is not one yet. The 2026-07-30
revision of this table used a flat 75% for any LETF book and an assumed 1.9% cash rate; the
per-ticker rule is stricter on 3x and the wider borrow spread lowers every Kelly figure, which
is why several `f_kelly` entries moved.

| Model | f | binding cap | m | DD month-end | DD daily | DD_adj | f_kelly |
|---|---|---|---|---|---|---|---|
| HAA_G12 | **1.271** | margin call | 0.45 | −8.4% | −13.4% | 20.1% | 4.13 |
| BAA_G12 | 1.198 | margin call | 0.45 | −10.7% | −13.0% | 21.4% | 2.42 |
| HAA_G8_Balanced | 1.196 | margin call | 0.45 | −9.9% | −14.4% | 23.3% | 2.48 |
| DAA_G12 | 1.173 | margin call | 0.45 | −18.4% | −18.6% | 23.6% | 1.83 |
| HAA_G1_Simple | 1.162 | margin call | 0.45 | −17.1% | −19.8% | 24.7% | 4.43 |
| BAA_G1_SPY | 1.139 | margin call | 0.45 | −12.5% | −14.7% | 25.6% | 2.38 |
| DAA_G6 | 1.123 | margin call | 0.45 | −19.9% | −20.3% | 25.9% | 1.15 |
| PAA2_G12 | 1.113 | margin call | 0.45 | −18.2% | −20.5% | 26.4% | 2.25 |
| DAA_G4 | 1.067 | **Kelly** | 0.45 | −21.0% | −21.3% | 28.8% | 1.07 |
| VAA_G4 | 1.058 | margin call | 0.45 | −21.7% | −21.8% | 29.6% | 1.12 |
| BAA_G4 | 1.014 | margin call | 0.45 | −15.3% | −19.7% | 32.4% | 1.64 |
| every 2x/3x wrap | **1.000** | margin call / Kelly | 0.90–0.99 | −18 to −54% | −27 to −70% | 45–95% | 0.66–2.62 |
| VAA_G12 | 1.000 | margin call | 0.45 | −28.0% | −29.4% | 36.4% | 1.02 |
| GEM_G2_Classic | 1.000 | **Kelly gate** | 0.45 | −22.9% | −33.7% | 56.5% | 0.23 |
| DM_G8_Composite | 1.000 | **Kelly gate** | 0.45 | −14.8% | −21.6% | 34.7% | −2.41 |
| Golden_Butterfly | 1.000 | **Kelly gate** | 0.45 | −16.9% | −20.1% | 30.3% | −0.58 |
| GTAA_G5 | 1.000 | **Kelly gate** | 0.45 | −13.3% | −13.8% | 23.2% | −4.32 |
| DM_G5_Leveraged_3X | 1.000 | **Kelly gate** | 0.99 | −81.8% | −84.4% | 100.0% | −0.08 |

**The safety factor is the whole grey zone, and it is one number.** Sweeping the module's only
preference parameter over its three named presets, same data and same everything else:

| k | entries above 1.00x | best |
|---|---|---|
| 5 — prudent | **0** | — |
| 3 — balanced | 11 | HAA_G12 at 1.27x |
| 2 — aggressive | 13 | HAA_G12 at 1.47x |

At `k=5` the registry supports no margin at all. That is the honest shape of the answer: the
recommendation is not a number this data pins down to two decimals, it is a band from 1.0x to
about 1.5x whose position is set by how many times the stressed drawdown you insist on
surviving. Nothing about a broker moves it nearly as much.

Three things fall out of that table, and none of them is a preference:

1. **At `k=3`, nothing in the registry supports meaningful margin.** The best entry is 1.27x, and
   it is the shallowest-drawdown model in the repository. `k=3` against a 25% stressed drawdown
   demands surviving a 75% decline, and at `m=0.45` that is worth about 1.2x. Anyone expecting
   1.3x to be comfortably available should read that row first.
2. **The intra-period correction is large, and it is real.** *Every* row's daily-path drawdown
   exceeds its month-end one — from +0.2% relative (`VAA_G4`, which was already de-risked into
   the worst of it) to **+59.7%** (`HAA_G12`, −8.4% on month ends against −13.4% on the daily
   path of the same held allocation). `HAA_G3_Leveraged_2X` posts the deepest absolute gap:
   −35.7% against **−51.7%**. That gap is invisible everywhere else in this repository, because
   `run_ledger` never looks inside a month.
3. **Thirteen entries fail the Kelly gate outright** at a 4.75pp borrow spread — including
   `Golden_Butterfly`, the 60/40, `GTAA_G5`, `GEM_G2_Classic`, the DM composite and the levered
   benchmarks. A low-volatility portfolio cannot pay 4.75pp on borrowed money; the gate is not
   being conservative, it is doing arithmetic. Which also means the *spread*, not the drawdown,
   is the variable most worth negotiating — and it is the one input here that a broker actually
   sets.

**The standing constraint.** The module sizes for indefinite sustainability and takes no
objective function, no target return, and no search over the backtest.
`test_no_optimisation_surface` fails if one is added, including if it is asked for later.

### The Decision Matrix

| Approach | Leverage | Regulatory Risk | Account Compatibility | Chosen? |
|----------|----------|----------------|----------------------|---------|
| No leverage | 1.0x | ✅ None | ✅ All accounts | ⚠️ The default; §3 prices its opportunity cost |
| Margin | any ratio, incl. 1.3x | ⚠️ Low risk with salary | ❌ Restricted in registered accounts | ⚠️ Viable; measured, and it worsened every risk metric |
| 2x LETF | 2.0x | ✅ None | ✅ All accounts | ✅ **SELECTED** |
| 3x LETF | 3.0x | ✅ None | ✅ All accounts | ⚠️ **Exploration only** — see below |

**2x is selected on the instrument constraint and the decay/drawdown arithmetic, not on a
Kelly figure.** It is the lowest ratio purchasable as a single product, its decay term is a
third of 3x's, and §7's drawdown mathematics do not depend on an estimate of μ.

> **⚠️ CORRECTED 2026-07-29.** The 3x row used to read *"❌ Too far from Kelly"* and the
> summary line claimed 2x was *"close enough to Kelly to be effective"*. Both inherited the
> withdrawn 1.2-1.3x figure, and the first is **backwards**: measured full Kelly on SPY is
> 4.49x, so 3x is *below* it, not beyond it (§2's corrected growth table shows g(3.0) >
> g(2.0)). 3x is declined here because its decay term is three times 2x's, because §7's
> drawdown arithmetic is unforgiving of it, and because **no 3x product has ever traded
> through a bear market** — not because it overshoots a growth optimum. 3x variants are
> registered for *measurement* under `role='exploratory'`; they are excluded from the
> selection statistics and are not recommendations.

---

## 6. How the Canary Shifts the Optimal Leverage

> **⚠️ CORRECTED 2026-07-29 — the conclusion survives, the mechanism did not.** Every earlier
> edition of this section asserted that *"the canary reduces effective volatility"* and
> derived the higher leverage from that. It was never measured. **Measured, it is false:** the
> volatility of the months a canary strategy is actually invested is *higher* than its blended
> volatility, in all three families. The old section also contained a regime table whose
> every input disagreed with the data, a volatility formula that is not how variance
> combines, and a "practical Kelly" figure that was a quarter of the full Kelly it was
> derived from rather than a half. All four are corrected below.

### 6.1 What the canary actually does to volatility

Measured over 2008-06…2026-06 (217 months), splitting each family's own months by whether its
signal was **fully offensive** in the month that earned the return:

| family | months fully offensive | σ, all months | σ, offensive months only | |
|---|---|---|---|---|
| HAA_G12 | 158 / 217 | 8.46% | **9.26%** | higher |
| DAA_G12 | 109 / 217 | 8.61% | **10.00%** | higher |
| BAA_G12 | 98 / 217 | 8.17% | **9.01%** | higher |

**The canary does not truncate the volatility of the invested state. It dilutes the
unconditional figure by spending a third to a half of all months in near-zero-volatility
assets.** Those are different things, and only the first would justify levering harder:

- If the canary genuinely made the *invested* months calmer, leverage applied to those months
  would face a lower σ, and the Kelly headroom would be real.
- What it does instead is add cash-like months to the average. Conditional on being invested,
  σ is **higher** than the blended number — necessarily so, since the blend includes the calm
  months.

And leverage lands **exactly on the offensive months**, because RULE 3 holds the defensive
sleeve at 1x (`assert_unlevered_defensive`). So the blended σ is the wrong input for a
levered wrap. **The design's own defence-at-1x rule is what makes the offensive σ the
relevant one.**

### 6.2 What the canary does instead — and why the conclusion still holds

The canary raises μ on the invested months at the same time as σ, and by more:

| family | blended f\* | **offensive-months f\*** | offensive μ | offensive σ |
|---|---|---|---|---|
| HAA_G12 | 11.85x | 11.46x | 9.82% | 9.26% |
| DAA_G12 | 9.05x | **10.82x** | 10.82% | 10.00% |
| BAA_G12 | 9.28x | **12.83x** | 10.42% | 9.01% |

For DAA and BAA the offensive-months Kelly is *higher* than the blended one — the months the
signal chooses to be invested are better months, not merely more volatile ones. Against SPY's
4.49x, every one of these is two to three times larger.

**So the old conclusion — a canary strategy can carry more leverage than buy-and-hold —
survives the correction.** What fails is the reason given for it. The canary earns its
headroom by *selecting better months*, not by making the invested months calmer.

### 6.3 Why none of these numbers should be acted on

An f\* of 9-13x means Kelly is telling you to lever a TAA strategy ten times. Nobody should,
and the reason is §1.2: **f\* is linear in μ, and these μ estimates are the most
over-fitted quantities in the document.** They are the realised excess returns of strategies
selected, in part, for having done well over this exact window, measured from near the bottom
of the financial crisis.

The correct reading is a negative one:

> **Kelly, applied to a canary strategy on this data, returns a number so large that it
> functions as a reductio.** It tells you the framework has no traction here — not that 10x is
> available. What it does establish, robustly, is an *ordering*: a canary strategy tolerates
> more leverage than the same assets held unconditionally. It cannot tell you how much more.

That ordering is all §§5 and 7 need. The choice of 2x rests on the instrument constraint
(§5), the daily-reset decay term (§2) and the drawdown mathematics (§7) — **not on a Kelly
figure**, which is the one thing this section can no longer be used to supply.

### 6.4 What the canary demonstrably does do

Drawdown, which is a different quantity from volatility and the one that actually destroys a
levered portfolio:

- The unlevered families' maximum drawdowns run −11% to −18% against SPY's **−46.3%** over
  the same window.
- §8's measurement of the levered wraps shows `BAA_G3/G4_Leveraged_2X` positive in both
  `bear_covid` and `bear_2022` — the only wraps that are.
- This is the real defence, and it is a *path* property. Levering a strategy that avoids
  −46% drawdowns is a categorically different act from levering one that does not.

**The canary and leverage work as a system** — that part was always right. The system works
by cutting the depth of the path, not by lowering the volatility of the months you hold risk.

---

## 7. Why Not 3x — The Destruction Risk

### Asymmetric Drawdown Mathematics

| Underlying Drawdown | 2x LETF Loss | 3x LETF Loss | Recovery Needed (3x) |
|--------------------|-------------|-------------|---------------------|
| -10% | -20% | -30% | +43% |
| -20% | -40% | -60% | +150% |
| -33% | -66% | **-99%** | **+9900%** |
| -40% | -80% | **-120%** (wiped out) | Impossible |

**At -33% underlying drawdown, the 3x LETF loses 99% of its value.** This is effectively permanent capital destruction.

### Historical Precedent

> **These are ILLUSTRATIVE, not measured.** The LETF columns are derived from the decay model
> above applied to the index drawdown; they are not backtest output, and for 2008 they could
> not be — UPRO launched 2009-06 and TQQQ 2010-02, so no 3x product existed during the GFC at
> all. That absence is the point of the row, and it is why the regime panel prints
> `n/a (inception …)` rather than a number. Anything the engine *can* measure is in the tables
> further up this document.

| Event | S&P 500 Drawdown | 2x LETF (modelled) | 3x LETF (modelled) |
|-------|-----------------|---------|---------|
| 2008 GFC | -50% | -77% | -95% |
| 2020 COVID | -34% | -57% | -76% |
| 2022 Bear | -25% | -43% | -59% |

**The 2008 GFC would have destroyed a 3x portfolio** (-95% requires +1900% to recover). The 2x portfolio suffered but recovered within ~2 years.

### Why Sortino Alone Is Insufficient

The Sortino ratio measures average downside risk, but does not capture:

1. **Maximum drawdown magnitude** — the worst single loss
2. **Recovery time** — how long to return to previous peak
3. **Tail risk** — probability of catastrophic loss
4. **Path dependency** — sequence of returns matters for leveraged products

A strategy with Sortino 1.70 (3x) but MaxDD -50% is **worse** than a strategy with Sortino 1.56 (2x) and MaxDD -22%, because:
- The -50% drawdown may force liquidation at the worst time
- Recovery from -50% requires +100% gain
- Psychological toll of -50% causes behavioral errors

**Metrics required for leverage evaluation:**
- Sortino (risk-adjusted return) — **necessary but not sufficient**
- MaxDD (worst case) — **critical for survival**
- Recovery time (months to break even) — **critical for patience**
- CAGR (absolute return) — **the actual wealth created**

---

## 8. Which Strategies Admit a Leveraged Variant, and Why

> **Rewritten 2026-07-29.** This section used to be titled *"Why DAA Is the Only Viable
> Strategy for Leverage"* and concluded **"If using leverage, use DAA. Period."** That
> conclusion is withdrawn, and so is the reasoning behind it, which contained a plain factual
> error: it claimed BAA "only goes to 50% defensive, never 100% cash". BAA's canary has
> `B = 1`, so a single dead canary sends it 100% defensive — it is the FASTEST and most
> complete exit in the whole canon, not a partial one. Measured, BAA is defensive in 54% of
> all months. The old table also ranked variants that no longer exist.

### The rule comes first, and it is structural

The leveraged variants are the one part of this repository with no author to defer to.
Keller fixed the universes and parameters of HAA/DAA/VAA/BAA/PAA; nobody published a levered
version of any of them. So the only thing separating a variant from a curve-fit is a rule
written down **before** the backtest and checkable **without** one. The five rules live in
[`common/letf_mapper.py`](common/letf_mapper.py). The one that decides which families are
eligible at all is:

> **RULE 4 — a wrap may change what is HELD, never what decides to DE-RISK.**

A wrap must restrict the offensive universe to what executes at one uniform multiple, in
practice SPY/QQQ/IWM(/GLD). If a family's de-risking signal is a *function of that universe*,
restricting it does not narrow the portfolio and leave the rule alone — it rebuilds the rule
out of three US equity ETFs that correlate 0.79–0.91 with each other.

| Family | De-risking signal | Reachable by the restriction? | Wrap |
|---|---|---|---|
| **HAA** | canary `TIP` | No — exogenous | ✅ G3, G4 |
| **BAA** | canary `SPY/VWO/VEA/BND`, `B=1` | No — exogenous | ✅ G3, G4 |
| **DAA** | canary `VWO/BND`, `B=2` | No — exogenous | ✅ G4 only (RULE 5) |
| **DM** | absolute momentum on the winner | No — per-asset, no denominator | ✅ G3 |
| **VAA** | breadth over its own offensive universe, `B = n` | **Yes** | ❌ deleted |
| **PAA** | breadth over its own offensive universe, `N = n` | **Yes** | ❌ deleted |

VAA and PAA declare `'canary': []` **on purpose** — breadth over a wide, diverse universe is
their entire protection mechanism. There is no fix: enlarging the universe until the breadth
count means something again requires assets with no admissible LETF (RULES 1–2), and any
smaller universe reproduces the defect. **Not every model has to have a leveraged version.**

The measured signature, over 2010-02…2026-06 — correlation between each wrap and its own
unlevered parent:

| Wrap | ρ vs parent | | Wrap | ρ vs parent |
|---|---|---|---|---|
| `BAA_G4_Leveraged_2X` | **0.823** | | `PAA_G4_Leveraged_2X` | 0.677 *(deleted)* |
| `DAA_G4_Leveraged_2X` | **0.794** | | `VAA_G4_Leveraged_2X` | 0.593 *(deleted)* |
| `HAA_G4_Leveraged_2X` | **0.784** | | `PAA_G3_Leveraged_2X` | 0.543 *(deleted)* |
| `HAA_G3_Leveraged_2X` | 0.662 | | `VAA_G3_Leveraged_2X` | **0.376** *(deleted)* |

The exogenous-canary wraps track the strategy they claim to run; the endogenous ones drift
away from it, worst where the universe is smallest. **That is a fidelity measurement, not a
performance ranking**, which is why it was allowed to decide admission. Deleting VAA and PAA
because they performed badly would have been selecting on the ranked table — the one thing
this repository forbids itself, since it prints the near-zero rank correlation between
disjoint sub-periods in every report.

**RULE 5** is about a parameter rather than a family, and the history is worth keeping because
the first version of the rule was wrong. In DAA, `T` does two jobs: it is the number of
offensive slots held *and* the denominator of the cash ladder `floor(b·T/B)/T` — Keller's *Easy
Trading* rounding of the real rule `CF = b/B`. A wrap sets `T` from the restricted universe
size, so a universe chosen for **LETF availability** ends up setting a parameter that governs
**protection**. Taking `T = n//2` alone, G3 gives `T = 1` against `B = 2`:

```
cash slots = floor(b*T/B)      b=0     b=1      b=2
  DAA_G12    (T=6, B=2)         0%     50%     100%
  DAA_G4_Lev (T=2, B=2)         0%     50%     100%
  T=1 vs B=2, bare floor        0%      0%     100%   <- the middle rung vanishes
```

One dead canary then produces no de-risking at all — **on the bare floor formula**. VWO went
negative in 2020-01 while BND stayed positive, so `b` held at 1 and a T=1 wrap carried 2x
equity through COVID for **−35.07%**, and for three months in 2011 for **−24.98%**.

> **⚠️ CORRECTED 2026-07-29 — and this one is a correction of a decision, not of a number.**
> `DAA_G3_Leveraged_2X` was **deleted** under RULE 5 read as a *filter* ("refuse any variant
> with `T < B`"), and **restored the same day** once the rule was re-read as *choosing* `T`:
>
> ```
> T = max(n // 2, B)
> ```
>
> `T` must have at least as many rungs as the canary has states. G3 gets `T = 2`, the rung
> survives, and the defect is unreachable at any universe size. **The variant was never
> defective — the parameter was**, and deleting a usable configuration to avoid a parameter
> choice that had a correct answer available threw away information for nothing.

> **⚠️ CORRECTED AGAIN 2026-07-30 — the rationale, this time.** The paragraphs above treated
> the Easy Trading rounding as all the paper offers at `T = 1`. DAA **note 8** legislates the
> case directly — *"with T=1, CF is simply b/B"* — so the paper's own ruleset keeps all three
> rungs at `T = 1`, and the −35.07% COVID figure describes the bare floor this repository
> used to run, not the paper. `strategies/daa.py` implements n.8 since 2026-07-30 (and
> `strategies/vaa.py` implements VAA's own, *different*, `T = 1` rule: §4 goes all-in-cash
> at `b ≥ 1` — each paper governs its own family). `T = max(n//2, B)` therefore stands as a
> **concentration convention**, not a protection repair; whether the G3 wrap should revert
> to `T = 1` is an open registry decision that must not be made from the measured table
> below. `assert_protection_survives_restriction` is kept as a backstop, exempting `T = 1`.

**Measured, same wrap, same data, only `T` differs (2008-03…2026-06):**

| | T=1 | T=2 (as built) |
|---|---|---|
| CAGR | 22.89% | 19.05% |
| Max drawdown | −46.58% | **−35.70%** |
| Sortino | 1.26 | **1.36** |
| UPI | 1.49 | 1.46 |
| Volatility | 33.45% | **23.48%** |
| COVID (2020-02…03) | −35.07% | **−15.99%** |
| 2011-07…09 | −24.98% | **−7.24%** |
| ADVERSE (equity cycle) | −67.05% | **−41.33%** |

The rung is demonstrably the mechanism, because the two ladders differ **only** at `b = 1`:

| months | what differs between the arms | σ, T=1 | σ, T=2 |
|---|---|---|---|
| `b = 0` (111) | holding count only (2 assets vs 1) | 9.04% | 8.03% |
| `b = 1` (74) | **the restored rung**, 0% vs 50% cash | 12.35% | **6.12%** |
| `b = 2` (35) | nothing — both hold 100% cash | −1.97% total | −1.57% total |

`b = 1` is a 50% cash allocation behaving exactly as it should: volatility halves, worst month
−27.2% → −13.3%. `b = 2` is the **control**, and its near-identity is what licenses reading the
other two rows. **The 3.8pp of CAGR given up comes from the `b = 0` row**, where holding two
assets instead of one is simply less concentrated in a bull market — not from the rung. Both
effects are real, and only the first is RULE 5.

**And note where the restored variant lands in the table below: seventh of nine on Sortino,
below the G4 sibling that replaced it.** It is registered because the parameter is now right,
not because the row is good — the same standard that kept `BAA_G3_Leveraged_2X` (last on
Sortino) and deleted the two PAA wraps that outranked it.

### The Evidence (2010-2026)

Regenerated over **2010-02 → 2026-06**. The start is not a choice: it is the common window the
engine derives for this line-up, bounded by UGL's 2008-12 inception plus warm-up, and the end
is the last complete month in the data. This table deliberately runs with
`RANKED_WINDOW_POLICY='all'` — every row here is a `custom` leveraged wrap, so the point is a
like-for-like comparison among them, not a comparison against published entries. Fills at the next
open, 0.10%/side one-way per leg, cash in BIL, Sharpe/Sortino net of the realised BIL return.
The former table here described G6 universes that no longer exist and whose offensive sleeves
silently mixed 3x, 2x and 1x exposure; it has been replaced, not annotated.

| Strategy | CAGR | MaxDD | Sortino | UPI | Vol | Turn/yr |
|---|---|---|---|---|---|---|
| `HAA_G12` (1x, for scale) | 10.13% | **−7.99%** | **2.24** | **4.22** | 7.6% | 6.9 |
| `HAA_G4_Leveraged_2X` | 20.72% | −22.80% | **1.59** | **2.66** | 20.3% | 7.9 |
| `SPY_Benchmark` (1x reference) | 14.48% | −23.31% | 1.39 | 2.29 | 15.0% | 0.1 |
| `BAA_G4_Leveraged_2X` | 14.53% | **−18.02%** | 1.38 | 1.91 | 17.0% | 11.8 |
| `HAA_G3_Leveraged_2X` | 20.19% | −35.74% | 1.32 | 1.70 | 25.9% | 9.0 |
| `DAA_G4_Leveraged_2X` | 15.51% | −34.67% | 1.29 | 1.14 | 20.3% | 13.4 |
| `DAA_G3_Leveraged_2X` | 17.27% | −35.70% | 1.27 | 1.27 | 22.0% | 11.5 |
| `DM_G3_Leveraged_2X` | **23.09%** | −47.79% | 1.27 | 1.31 | 32.7% | 4.6 |
| `BAA_G3_Leveraged_2X` | 11.47% | −33.79% | 0.92 | 0.82 | 20.1% | 12.1 |

*(`DAA_G3_Leveraged_2X` added 2026-07-29 on restoration. Every other row is byte-identical to
the pre-restoration table — the `T` floor is inactive for every universe whose `n//2` already
clears `B`, which is the check that it was added without disturbing anything.)*

**`HAA_G4_Leveraged_2X` is the first leveraged variant in this repository ever to beat plain
SPY on both Sortino and UPI, and the honest thing to do with that fact is to distrust it.**
It was added on 2026-07-29 — the same day the table shrank from ten rows to seven. A variant
introduced today, topping a table trimmed today, is exactly the shape a search-until-it-wins
result takes. Three things are worth checking before believing it:

1. **The admission rule was written before the measurement and does not mention returns.**
   HAA qualified because its TIP canary is exogenous and its absolute-momentum filter is
   per-asset — both readable in `strategies/haa.py` with no backtest open. The same rule
   deleted four variants, two of which (`PAA_G4`, `VAA_G4`) had *better* headline numbers
   than `BAA_G3`, which survived. A rule that keeps a loser and cuts a winner is at least
   not being applied to the leaderboard.
2. **Best-of-N.** Seven variants over one window: the expected maximum Sortino under the null
   is not 1.39. The report's selection-statistics block computes this properly; read it there
   rather than trusting this table's top row.
3. **The window is ~90% bull market.** It begins 2010-02, after the GFC bottom, and 2022 is
   its only genuine bear ([`KNOWN_GAPS.md`](KNOWN_GAPS.md) §1).

That last point is not a caveat, it is the finding. Compounding **only the objectively
adverse months** of the era — the equity-cycle panel's `ADVERSE` bucket — inverts the table:

| Wrap | Full window (Sortino) | `ADVERSE` total | `bear_covid` | `bear_2022` |
|---|---|---|---|---|
| `BAA_G4_Leveraged_2X` | 1.38 | **+10.3%** | **+5.9%** | **+4.2%** |
| `BAA_G3_Leveraged_2X` | 0.92 | **+23.2%** | **+5.9%** | **+4.2%** |
| `DAA_G4_Leveraged_2X` | 1.29 | −21.4% | −4.5% | −17.7% |
| `HAA_G4_Leveraged_2X` | **1.59** | −27.7% | −19.2% | −10.4% |
| `HAA_G3_Leveraged_2X` | 1.32 | −53.0% | −35.1% | −3.5% |
| `DM_G3_Leveraged_2X` | 1.27 | −61.3% | −35.1% | −25.9% |

**HAA wins the headline; BAA wins the crisis.** Both readings are correct, and the mechanism
that explains the split is the same one RULE 4 is about — but a dimension of it the rule does
not capture. RULE 4 guarantees a wrap runs *the paper's* protection rule. It says nothing
about whether that rule is **fast** enough for 2x, and the two exogenous canaries differ
sharply on exactly that:

| | BAA | HAA |
|---|---|---|
| Canary | 4 tickers, `B=1` → **any** one down ⇒ 100% defensive | **1** ticker (TIP) |
| Months defensive | 54% | 28% |
| First >50% defensive, COVID | **2020-01** | 2020-03 (G3) / never crossed 50% (G4) |
| First >50% defensive, 2022 | **2021-12** | 2022-01 |

BAA is early and often, and pays for it with forgone upside (14.5% CAGR against HAA's 20.7%).
HAA is late and rarely, and pays for it in crashes. Over a window that is 90% bull market, the
second bill is the one that mostly does not come due — which is precisely why the full-window
Sortino column should not be read as a recommendation.

A caveat that was **predicted before it was measured and confirmed anyway**: a single-asset
canary is the narrowest sensor in the canon, and at 2x the narrowness shows. In COVID,
`HAA_G12` also fired only in 2020-03 — but it lost just −2.3%, because twelve asset classes
gave it somewhere to rotate. Its three-asset wrap had nowhere to go and lost −35.1%. The
canary survived the restriction, as RULE 4 requires; the *portfolio* did not.

**Two claims from earlier editions of this section are withdrawn:**

- ~~"Not one leveraged variant beats plain SPY on risk-adjusted terms over this window."~~
  `HAA_G4_Leveraged_2X` does, on both Sortino and UPI. See the three caveats above before
  treating that as an endorsement.
- ~~"If using leverage, use DAA. Period."~~ DAA now ranks fourth of seven on Sortino, its G3
  size was deleted for a construction defect, and the mechanism the claim rested on (BAA
  "never goes to 100% cash") was simply false.

### What the exit mechanics actually look like

The earlier edition contrasted a "clean binary" DAA against a "gradual, dangerous" VAA. The
measurement does not support that framing either — the number of de-risking levels each wrap
*actually used* over 197 months:

| Wrap | Levels used | Flips/yr | Turnover/yr |
|---|---|---|---|
| `BAA_G3` / `BAA_G4` | 2 — `0%, 100%` | 4.1 / 4.3 | 11.4 / 11.8 |
| `HAA_G3` | 2 — `0%, 100%` | 2.2 | 8.6 |
| `HAA_G4` | 3 — `0%, 50%, 100%` | 1.6 | 7.9 |
| `DAA_G4` | 3 — `0%, 50%, 100%` | 2.1 | 13.4 |
| `DM_G3` | 2 — `0%, 100%` | 0.5 | 4.4 |

Nothing here is "continuous". A restricted universe quantises every ladder, because the number
of rungs is bounded by the number of selection slots — which is why RULE 5 exists. The real
distinction among the survivors is not binary-vs-gradual, it is **how broad the sensor is and
how early it trips**, and the table above is where to read it.

### The decomposition: how much is the timing rule, and how much is just leverage?

**Until 2026-07-29 this question could not be answered here at all**, because every leveraged
wrap was compared only against **1x** references. Four passive benchmarks were added to close
that (`strategies/passive.py`); they carry no signal, no universe choice and no fitted
parameter, so there is nothing in them that could have been tuned.

Measured 2010-08…2026-06 (191 months — the window UPRO's 2009-06 inception plus warm-up
allows), `RANKED_WINDOW_POLICY='all'` so every row spans the same months, ranked by Sortino:

| Entry | CAGR | MaxDD | Sortino | UPI | Vol | Turn/yr |
|---|---|---|---|---|---|---|
| `HAA_G4_Leveraged_2X` | 20.70% | −22.19% | **1.66** | **2.88** | 19.5% | 8.0 |
| `BAA_G4_Leveraged_2X` | 14.61% | **−16.98%** | 1.49 | 2.06 | 16.1% | 11.7 |
| **`Sixty_Forty_1X`** | 9.82% | −19.57% | **1.49** | 1.93 | 8.9% | **0.3** |
| `SPY_Benchmark` (1x) | 14.66% | −23.31% | 1.42 | 2.34 | 14.7% | 0.1 |
| `Golden_Butterfly` | 8.03% | −16.93% | 1.36 | 1.76 | 8.0% | 0.4 |
| `HAA_G3_Leveraged_2X` | 19.77% | −35.74% | 1.36 | 1.76 | 24.8% | 9.0 |
| `DAA_G3_Leveraged_2X` | 16.92% | −35.70% | 1.29 | 1.25 | 21.2% | 11.5 |
| **`SPY_2X_Benchmark`** | 24.00% | −45.88% | **1.29** | 1.75 | 29.8% | **0.1** |
| `DM_G3_Leveraged_2X` | 22.81% | −47.79% | 1.28 | 1.30 | 32.1% | 4.6 |
| **`SPY_3X_Benchmark`** | **31.11%** | −65.40% | 1.26 | 1.45 | 45.2% | 0.1 |
| **`RiskParity_3X`** | 18.52% | **−68.27%** | **1.08** | **0.66** | 27.3% | 0.6 |

Four things fall out, and three of them are unflattering to the wraps:

1. **The canary earns its keep where it earns it.** `HAA_G4_Leveraged_2X` beats
   `SPY_2X_Benchmark` by 0.37 of Sortino and halves the drawdown (−22.2% vs −45.9%). *That*
   is what a timing rule is worth on the same leverage — and it is the first number in this
   document that isolates it.
2. **Two wraps do not beat buying and holding SSO.** `DAA_G3_Leveraged_2X` ties it on Sortino
   (1.29) and `DM_G3_Leveraged_2X` is behind it (1.28), both with 40-100× the turnover. On this
   window their signals bought nothing a 2x buy-and-hold did not already have. This is exactly
   what the benchmark was added to make visible, and it would have stayed invisible without it.
3. **The classic 60/40 matches a leveraged TAA wrap on risk-adjusted terms** — `Sixty_Forty_1X`
   ties `BAA_G4_Leveraged_2X` at 1.49 Sortino, at a third of the volatility and **1/40th of the
   turnover**. It gives up 4.8pp of CAGR to do it. Whether that trade is worth taking is a
   decision about risk appetite; the point is that the boring reference is not embarrassed here.
4. **Adding levered treasuries to levered equity made it WORSE.** `RiskParity_3X` (UPRO 55 /
   TMF 45) is beaten by plain `SPY_3X_Benchmark` on Sortino (1.08 vs 1.26), on UPI (0.66 vs
   1.45) *and* on drawdown (−68.3% vs −65.4%) — while giving up 12.6pp of CAGR. **The 45% TMF
   sleeve, over this window, was pure cost.** That is a direct measurement of the position §6
   and §9 take on levered bonds, and it comes from Hedgefundie's own weights rather than from
   anything invented here.

**The caveat that governs all four rows, and it is not small.** This window begins 2010-08. It
contains no bear market for any 3x product, because none existed before 2008-11 — so
`SPY_3X_Benchmark`'s −65.4% is its drawdown *in a bull market*, and §7's arithmetic says the
2008 figure would have been about −95%. Every 3x number in this table should be read as an
upper bound on how well 3x behaves, not an estimate of it.

**`RiskParity_2X` is not in this table because it cannot be built.** No admissible 2x treasury
product exists (UBT holds ~$65M, under the RULE 2 floor), so the risk-parity reference jumps
1x → 3x with nothing between. That is a fact about the product market, and it is why the 2x/3x
comparison in this document is only ever clean on all-equity universes.

### The ratio ladder: what 3x actually does

**This is the only measurement in the repository where the leverage LEVEL is the single changed
variable.** Each `*_G3_Leveraged_3X` runs the identical signal on the identical universe over
the identical months as its 2x twin. Nothing else differs. Window **2011-04…2026-06, 183
months**, bound by TQQQ's 2010-02 inception plus warm-up, `RANKED_WINDOW_POLICY='all'`.

| pair | CAGR | MaxDD | Sortino | UPI | Vol | ρ(2x,3x) |
|---|---|---|---|---|---|---|
| `HAA_G3_Leveraged_2X` | 18.83% | −35.74% | **1.29** | **1.64** | 24.8% | |
| `HAA_G3_Leveraged_3X` | 23.91% | −54.33% | 1.21 | 1.33 | 38.0% | 0.9991 |
| `BAA_G3_Leveraged_2X` | 10.77% | −33.79% | **0.92** | **0.78** | 18.9% | |
| `BAA_G3_Leveraged_3X` | 13.22% | −50.88% | 0.88 | 0.61 | 28.3% | 0.9967 |
| `DAA_G3_Leveraged_2X` | 14.82% | −35.70% | **1.15** | **1.06** | 20.5% | |
| `DAA_G3_Leveraged_3X` | 19.39% | −47.79% | 1.11 | 0.96 | 31.1% | 0.9980 |
| `DM_G3_Leveraged_2X` | 20.60% | −47.79% | **1.17** | **1.14** | 31.6% | |
| `DM_G3_Leveraged_3X` | 24.56% | −69.40% | 1.11 | 0.90 | 48.4% | 0.9993 |

**Four pairs out of four, without exception: 3x raises CAGR by 2.5–5.1pp and lowers Sortino
*and* UPI, while deepening the maximum drawdown by 13–22pp.** There is no family here for which
the third turn of leverage is a risk-adjusted improvement. That unanimity is the finding.

**ρ ≈ 0.997–0.999 — and this is precisely why the correlation argument for *deleting* these
entries was wrong.** An earlier proposal (RULE 6) would have refused any 3x variant whose
universe was already covered at 2x, on the grounds that ρ ≈ 0.999 makes it redundant. The table
above is the refutation: ρ that high means the two series have near-identical **shape**, and
says nothing whatever about **magnitude**. `DM_G3` at 3x draws down −69.4% against its twin's
−47.8% at a correlation of 0.9993. The depth is the one number not derivable from the 2x
sibling, and it is the number worth having.

**RULE 3 shows up in the data, and it is the prettiest result here.** `BAA_G3_Leveraged_2X` and
`BAA_G3_Leveraged_3X` returned **exactly the same** +4.19% through `bear_2022` and **exactly the
same** +5.86% through `bear_covid`. Not similar — identical. BAA was fully defensive in every
month of both, and the defensive sleeve is held at 1x regardless of the wrap's ratio, so the
leverage level was simply not in play. *A canary that fires early makes the ratio irrelevant
exactly when the ratio would hurt most.* That is the mechanism §6 was reaching for, measured.

### Tier 2 — the 3x-only universes, and the one that should frighten you

These carry TLT (via TMF) and EEM (via EDC), neither of which has an admissible 2x product, so
they exist at 3x or not at all. No 2x twin exists to compare them against — any conclusion drawn
across them is confounded by the universe as well as the ratio.

| entry | CAGR | MaxDD | Sortino | UPI | Vol | worst month | bear_2022 |
|---|---|---|---|---|---|---|---|
| `HAA_G5_Leveraged_3X` | 23.98% | −36.52% | **1.51** | **2.24** | 29.0% | −22.9% | −15.9% |
| `DAA_G5_Leveraged_3X` | 16.29% | −47.78% | 1.05 | 0.65 | 28.9% | −26.9% | −22.7% |
| `BAA_G4_Leveraged_3X` | 15.15% | −46.49% | 1.04 | 0.88 | 25.7% | −22.8% | **+4.2%** |
| **`DM_G5_Leveraged_3X`** | **6.40%** | **−81.80%** | **0.53** | **0.09** | 45.7% | −35.6% | **−38.6%** |
| `SPY_3X_Benchmark` | 28.20% | −65.40% | 1.17 | 1.28 | 45.3% | −54.5% | −61.9% |
| `SPY_Benchmark` (1x) | 13.91% | −23.31% | 1.33 | 2.16 | 14.8% | −16.4% | −23.3% |

Three things to take from it:

1. **`DM_G5_Leveraged_3X` is the worst entry in the registry by every measure that matters** —
   UPI 0.09, drawdown −81.8%, and it *underperformed 1x SPY on CAGR* while carrying three times
   the volatility. Dual momentum holds a single winner, so with TMF in the universe it sat 100%
   in 3x long-duration treasuries through the rate-hiking cycle: −38.6% through 2022 alone. Its
   own docstring predicted this before it was run, and it is registered *because* it is the
   sharpest available demonstration of what levered bonds in an offensive sleeve can do.
2. **Plain 1x SPY (Sortino 1.33) beats every 3x entry here except one.** Not on return — on
   risk-adjusted return. The exception, `HAA_G5_Leveraged_3X` at 1.51, is a wider universe on a
   shorter window and should be treated as a hypothesis, not a result.
3. **Two wraps again fail to beat unprotected buy-and-hold at their own ratio.** `SPY_3X` posts
   Sortino 1.17 against `DAA_G3_3X` 1.11 and `DM_G3_3X` 1.11. The timing rule earned nothing
   there.

**Every number in both tables is a bull-market number and cannot be otherwise.** UPRO launched
2009-06, TQQQ 2010-02, TMF 2009-04, EDC 2008-12: **not one 3x drawdown in this repository has
been measured through a bear market.** §7's arithmetic puts a 3x portfolio through 2008 at
roughly −95%, and the deepest figure above is −81.8% *without* one. Every 3x entry carries
`role='exploratory'` — measured and reported in full, excluded from the selection statistics,
flagged `(expl)` in the ranked table, and hidden behind `SHOW_EXPLORATORY` by default. **They are
instruments of measurement, not candidates.**

`EDC` deserves its own line: **$190M AUM, the thinnest mapping in this repository** and barely
over a RULE 2 floor that is itself set at the permissive end of a defensible range. Two of the
four Tier-2 entries depend on it.

---

## 9. Every Liberty Taken — the complete register

**Why this section exists.** Keller published HAA, DAA, VAA, BAA and PAA with their
universes and parameters fixed; **nobody published a levered version of any of them.** Every
choice in this subsystem was made here. There is no author to defer to and no specification
to check against, so the only thing separating these variants from a curve-fit is that each
decision is written down, justified on structure rather than on returns, and — wherever
possible — enforced by code that refuses the alternative rather than by a comment asking
politely.

This is the complete list. If a design decision in the leveraged subsystem is not here, it
is undocumented and should be treated as suspect.

| # | Liberty | Enforced by |
|---|---|---|
| L1 | LETF execution rather than margin | design; both are available and compose |
| L2 | Offensive universe restricted to one uniform multiple | `validate_universe` (RULE 1) |
| L3 | $100M AUM floor on every product | `RETAINED_AUM_USD` / `REJECTED_MAPPINGS` (RULE 2) |
| L4 | Defence held at 1x, in real defensive assets | `assert_unlevered_defensive` (RULE 3) |
| L5 | Only families with an exogenous de-risking signal get a wrap | RULE 4 + `test_anchors.py` |
| L6 | Each wrap keeps its OWN family's canary, unchanged | RULE 4 |
| L7 | Cash ladders must survive the restriction: **`T = max(n//2, B)`** | the constructor computes it; `assert_protection_survives_restriction` backstops (RULE 5) |
| L8 | 3x registered but `role='exploratory'` — measured, never a candidate | derived from `leverage` on `BaseStrategy.role`; `test_no_three_x_entry_can_reach_the_selection_statistics` |
| L9 | `TO = NO/2` in the BAA wraps | documented choice, not enforced |
| L10 | A dual-role asset may not sit in a levered offensive universe | `assert_no_dual_role_mixes_the_sleeve` (RULE 1's second half) |

---

### L1 — LETF execution rather than margin

Both mechanisms exist in this engine and **compose multiplicatively by design**: a 2x LETF
strategy run at `LEVERAGE_FACTOR = 1.3` carries ~2.6x. That is coherent, not a bug.

The wraps use LETFs because of a structural difference §5 covers in full: **an LETF portfolio
de-levers by itself when the signal goes risk-off** — rotating UPRO → IEF drops exposure to 1x
automatically — whereas flat margin keeps the loan drawn and buys treasuries with borrowed
money while still levered into the drawdown. `MARGIN_FOLLOWS_SIGNAL` closes most of that gap
for the margin path, but not all of it: **a margin account can be called intramonth; an LETF
cannot.**

Signal/trade separation is the other half. Momentum is always computed on the **1x** series
(SPY, not UPRO) and only the resulting weight is redirected to the leveraged product. Ranking
on LETF prices would let the daily-reset path distort the signal that decides the position.

### L2 — Reduced universe (RULE 1)

| Keller G12 | Ours | Reason |
|-----------|------|--------|
| SPY, QQQ, IWM, VGK, EWJ, VWO, VNQ, GSG, GLD, TLT, HYG, LQD | 2x: SPY, QQQ, IWM, GLD | Only these four map to a retained 2x product |
| " | 3x: SPY, QQQ, IWM | Only these three map to a retained 3x product |

**Why reduced universe is superior to "skip" approach:**

Two alternatives were considered and rejected.

**"Skip" — rank the full G12 universe, then pass over assets without an LETF.** This
corrupts the signal: if VGK ranks #1 but has no product, the skip approach selects #7 — yet
VGK being #1 has already shifted the relative ranking of everything else. The decision is
partly based on non-tradable information.

**"Fall through" — rank G12, and hold unmapped winners at 1x.** This is what the removed
full-universe wraps actually did, and it is worse than "skip" because it is invisible.
Nothing in the output distinguishes a month running at 3x from a month running at 1.4x;
effective leverage becomes a random variable driven by the signal draw, and MaxDD/Sortino
computed over that path describe no portfolio anyone could hold to a fixed risk budget.

The reduced-universe approach calculates momentum **only on assets that execute at the
target multiple**. Every asset in the signal is tradable, at the intended leverage.

| Approach | Signal Purity | Effective Leverage | Chosen? |
|----------|--------------|--------------------|---------|
| G12 + skip | ⚠️ Corrupted (non-tradable assets influence ranking) | Uniform | ❌ |
| G12 + fall-through | ⚠️ Corrupted | ❌ Drifts silently with the signal | ❌ |
| Reduced universe | ✅ Pure (only tradable assets) | ✅ Uniform and stated | ✅ |

**This is a deliberate trade-off:** we lose the European, Japanese, emerging, REIT,
commodity and credit sleeves, but gain a portfolio whose leverage is a known constant.

**The cost, stated plainly.** What survives at a uniform 2x is SPY, QQQ, IWM and GLD — and
the first three correlate **0.911 / 0.886 / 0.789** with one another in monthly returns. A
G3 wrap is therefore a *levered US-equity timing model*, not a diversified one, and its
"selection" among three near-identical assets is thin: the G2 size (SPY + QQQ) was deleted at
ρ 0.911 on the grounds that a momentum choice between near-identical assets is not a choice,
and G3 clears that bar only barely. GLD, at ρ 0.04–0.08 against all three, is the only asset
in the admissible set that changes the portfolio's character rather than its beta — which is
the whole reason the G4 sizes exist and the argument behind L9 below.

### L3 — $100M AUM floor (RULE 2)

The binding risk at retail size is **issuer closure**, not spread. A levered ETF charging
~0.95% on $50M grosses ~$475k/yr, which does not cover a swap desk, daily rebalance
operations, compliance and market-maker support; issuers routinely liquidate these, and a
closure mid-rotation forces realisation on the issuer's date rather than yours, with no
successor product. $100M is deliberately the **permissive** end of a defensible range — a
stricter desk would use $250M–$500M — so that everything rejected below it fails at the most
generous threshold that can be justified. Six products were rejected on this rule alone
(UBT, EET, EFO, URE, EURL, DRN); each is recorded in `REJECTED_MAPPINGS` with its AUM.

### L4 — Defence is held at 1x, in REAL defensive assets

> **Correction, 2026-07-29.** Earlier editions of this section claimed the leveraged variants
> hold *"Defensive: 100% cash"* because *"LETFs on bonds amplify interest rate risk"*. That
> was never what the code did, and the stated reason confuses two things. The defensive
> sleeve is held **unleveraged, in its own real assets** — SHY/IEF/LQD for DAA and VAA,
> BIL/IEF for HAA, the seven-name basket for BAA. Holding cash instead would forfeit the
> bond return in exactly the months the strategy is defensive, which is a large part of what
> these strategies earn.

The actual rule is narrower and harder: **no leveraged product may appear in a defensive
sleeve, ever.** `DAA1_G12` declared `UST` — a 2x 7-10y treasury ETF — as one of three
defensive candidates, so its risk-off state *doubled* duration risk while the engine reported
it as risk-off. It was deleted, and `assert_unlevered_defensive` now refuses to price any
strategy that declares one. A leveraged *credit* product would still be admissible in the
**offensive** universe, where Keller puts LQD, because there it is declared as what it is.

### L5 / L6 — Only exogenous-signal families get a wrap, and each keeps its own canary

> **Correction, 2026-07-29.** An earlier edition claimed *"the leveraged implementation
> standardizes on VWO+BND for consistency"*. It does not, and it must not. Standardising the
> canary would erase the only thing distinguishing the families once their offensive
> universes have all been restricted to the same four tickers — and it is the exact failure
> RULE 4 exists to prevent.

Every wrap runs its **own** family's canary, untouched: HAA reads `TIP`, BAA reads
`SPY/VWO/VEA/BND`, DAA reads `VWO/BND`. That is checked by
`test_strategy_differences.py`, including the sharpest form of it — collapse VWO and BND to
nothing, and HAA's allocation must not move at all.

Which families are eligible is §8's subject and is not repeated here. The short version:
VAA and PAA protect by counting breadth over their *own* offensive universe, so a wrap
rewrites the protection rule instead of narrowing the portfolio; they have no leveraged
variant and should never get one.

### L7 — Cash ladders must survive the restriction (RULE 5)

Covered in §8. **`T = max(n//2, B)`**, computed in the constructor, because `floor(b·T/B)/T` is
only Keller's Easy Trading *rounding* of `CF = b/B` and stops approximating it once `T < B`.
`T` must have at least as many rungs as the canary has states.

The liberty here is `n//2` (that is L9's choice, shared with BAA and HAA). **The floor at `B` is
not a liberty — it is a repair**, and the constructive form is what distinguishes the two: the
wrap is free to choose how many slots to hold, and not free to choose a number that discards
canary states. Stated as a filter instead, the same requirement deleted
`DAA_G3_Leveraged_2X` on 2026-07-29; it was restored the same day.

**Does this reopen VAA or PAA?** No, and the answer is worth stating because the question is
natural. Under `T = max(n//2, B)`, VAA's `B = n` would give `T = n`, so its ladder *would*
resolve every canary count — L7 is satisfied. **They stay excluded under L5/RULE 4**, which is
a different objection: their protection is not a ladder at all, it is a *breadth count over
their own offensive universe* (`self.B = n` in VAA, `N = len(offensive)` in PAA). Restricting
that universe for LETF execution rewrites the protection rule itself, and no choice of `T`
touches that. **Different rule, unchanged verdict, permanently.**

### L8 — 3x is registered, and carries `role='exploratory'`

> **⚠️ REVERSED 2026-07-29.** This entry used to read *"2x only — 3x variants are constructible
> but none is registered"*, and gave two reasons. **One of them was a bad argument and the
> owner rejected it.**

The rejected reason was correlation: every 3x twin measures **ρ ≈ 0.997–0.999** against its 2x
sibling, so it looked derivable and redundant. It is not, and the old text carried its own
refutation two sentences later: *"ρ ≈ 0.999 means near-identical shape, not magnitude."*
**Magnitude is the whole point.** `DM_G3` draws down −47.8% at 2x and −69.4% at 3x with ρ =
0.9993. The depth is the one thing the 2x sibling cannot tell you, and suppressing the 3x entry
suppressed exactly that. See §8's ratio-ladder table.

The surviving reason is real and unchanged: **no 3x product predates 2008-11** (UPRO 2009-06,
TQQQ 2010-02, TMF 2009-04, EDC 2008-12), so no 3x drawdown here has been measured through a
bear market and none ever can be. That is grounds for keeping such a row out of the *candidate*
set — not for refusing to measure it.

So the liberty is now: **register 3x, and separate "measured" from "candidate" with the role.**

| | 2x | 3x |
|---|---|---|
| registered | ✅ | ✅ (8 entries) |
| ranked in the table | ✅ | ✅, flagged `(expl)` |
| counts as a TRIAL in the selection statistics | ✅ | ❌ |
| shown by default in the picker / CLI | ✅ | ❌ (`SHOW_EXPLORATORY`) |

**Enforced, not remembered.** `BaseStrategy.role` derives `'exploratory'` from
`leverage >= 3`, so a new 3x factory acquires it whether or not anyone thinks about it. That
derivation was first written on `LeveragedWrapMixin` and **silently missed two of the eight
entries**, because `DMLeveraged` subclasses `BaseStrategy` directly and uses no mixin: six got
the role, two did not, and the two that did not would have been counted as trials and ranked as
portfolios one might hold. It now lives on the base class.

Twelve new registry rows (eight 3x wraps plus four passive references) moved the trial count
from **18 to 19**. That ratio is why the role is non-negotiable rather than merely tidy.

### L9 — `TO = NO/2` in the BAA wraps

**This is the one liberty with no external anchor at all, and it is a choice.** Keller
publishes two ratios for BAA and they disagree: BAA-G12 takes `TO = 6` of `NO = 12` (one
half), BAA-G4 takes `TO = 1` of `NO = 4` (one quarter). A wrap has to pick one.

`NO/2` is used. At `n = 3` the question does not arise — both ratios floor to `TO = 1`. At
`n = 4` they differ, and the argument is about *this* universe rather than about performance:
one quarter means holding exactly **one** of SPY/QQQ/IWM/GLD, three of which are the same bet
(ρ 0.79–0.91). `TO = 1` would spend roughly half its months holding **gold alone at 2x** and
the other half a single levered equity sleeve. `TO = 2` lets the diversifier be held
*alongside* an equity leg — which is the only reason GLD was worth admitting.

Measured, for information and **not** as the justification — over 2010-02…2026-06:

| | CAGR | MaxDD | Sortino | UPI | ρ |
|---|---|---|---|---|---|
| `TO = 1` | 15.38% | −24.48% | 1.18 | 1.27 | 0.879 |
| `TO = 2` *(used)* | 14.53% | **−18.02%** | **1.38** | **1.91** | |

`TO = 2` is better here. **That is deliberately not the argument.** Choosing a parameter by
re-running the backtest is the degree of freedom this repository spends most of its guards
refusing, and it would be no more defensible for a wrap than it was for the sixteen entries
deleted on 2026-07-28 — several of which were nothing but a parameter twiddle on an unchanged
universe. The numbers are printed here so that a future reader can see the choice was not
made *against* them, not so that they can be read as having made it.

Unlike L2–L8, this one is documented rather than enforced: no assertion can distinguish a
justified `TO` from an unjustified one. It is the weakest link in the chain and is labelled
as such.

*(HAA's wraps use the same `n // 2` expression, but there it is not a liberty:
`TO = NO/2` is Keller's own published ratio — HAA-12 takes 6 of 12, HAA-8 takes 4 of 8. DM's
wrap has no selection count at all; dual momentum holds the single winner.)*

### L10 — a dual-role asset may not sit in a levered offensive universe

Added 2026-07-29, and it is a **repair rather than a choice** — which is why it is enforced in
code rather than argued here.

RULE 1 requires the whole offensive sleeve to execute at one multiple. `validate_universe`
checks that, but it is called on `set(offensive) - set(defensive)` — the *post-exclusion* set.
An asset that is dual-role has therefore already been removed before the check runs, so the
universe validates while the sleeve it describes is mixed:

```
BAALeveraged(['SPY', 'QQQ', 'IWM', 'TLT'], 3)
  validate_universe(['SPY','QQQ','IWM'])   -> PASSES  (TLT was removed first)
  translate(...)                           -> SPY/QQQ/IWM as UPRO/TQQQ/TNA, TLT at 1x
  a month where the signal picks TLT       -> part 3x, part 1x
```

TLT and DBC are both in BAA's defensive basket (`TIP, DBC, BIL, IEF, TLT, LQD, BND`), so this
was reachable for BAA and not for HAA or DAA, where TLT is offensive-only.

**What made this worth fixing rather than noting:** until 2026-07-29 the only thing preventing
it was the list of factories somebody had written, and `baa_leveraged.py`'s docstring said so
outright — *"The admissible universes above avoid this by construction, and validate_universe
checks the post-exclusion set."* The second clause was simply false, and the plan for the 3x
expansion predicted `validate_universe` would refuse a 3x BAA on TLT. It did not.
`assert_no_dual_role_mixes_the_sleeve` now does, and it takes its two sets **explicitly**
rather than sniffing attributes — DM names them `universe`/`cash`, and a guard that guessed
would have silently no-opped on the one family shaped differently from the others.

---

## 10. Summary — The Complete Justification Chain

> **⚠️ STEPS 1, 6, 7 AND 8 WERE REWRITTEN 2026-07-29.** The chain used to open *"Kelly
> Criterion says optimal leverage ≈ 1.2-1.3x"* and close the leverage argument with *"2x is
> the closest available instrument to Kelly"*. That figure was never measured (§1), and the
> canary mechanism in step 7 is false as it was stated (§6). **The chain no longer routes
> through a Kelly number at all** — it routes through the instrument constraint, which is a
> fact about the market rather than an estimate. The conclusion is unchanged; it now rests on
> premises that can be checked.

```
1. Kelly gives the growth-optimal leverage as f* = μ/σ², which is LINEAR in the equity
   risk premium — measured 4.49x on SPY here, 1.6-2.4x on a conservative 4-6% premium.
   The spread is too wide to select an instrument with. Kelly establishes an ORDERING,
   not a level.
       ↓
2. No LETF exists between 1x and 2x — a fact about the product market, not an estimate
       ↓
3. Margin reaches any ratio, but keeps the loan drawn through a drawdown; measured at
   1.3x it worsened max drawdown, Sortino and UPI in every case (README)
       ↓
4. LETFs come in 2x or 3x only
       ↓
5. 3x has 3× worse volatility decay, and no 3x product has ever seen a bear market
       ↓
6. 2x is therefore the LOWEST leverage purchasable as a single product — chosen as a
   floor on available leverage, not as an approximation to an optimum
       ↓
7. An exogenous canary lets a strategy carry more leverage than buy-and-hold — NOT by
   lowering the volatility of the invested months (measured: that volatility is HIGHER),
   but by selecting better months and by cutting drawdown DEPTH (§6)
       ↓
8. 2x + canary is defensible; how far it sits below the growth optimum is not knowable
   from this data, and §6.3 explains why the f* of 9-13x it implies is a reductio
       ↓
9. No leverage = significant opportunity cost over long horizons
       ↓
10. Lifecycle investing supports leverage when the time horizon is long
       ↓
11. Signal-trade separation with a reduced universe = clean signal
       ↓
12. Uniform-ratio universe = effective leverage is a known constant, not a signal artefact
       ↓
13. RULE 4 = only families whose de-risking signal survives the restriction get a wrap
    (HAA, BAA, DAA, DM — not VAA, not PAA)
       ↓
14. RULE 5 = and only where the cash ladder survives it too (so DAA at G4, not G3)
```

> **The chain ends at a set of admissible variants, not at a recommendation.** An earlier
> edition ended step 13 with *"`DAA_G4_Leveraged_2X` = optimal configuration given all
> constraints"* and called the result *"not a compromise… the mathematically justified
> optimum"*. That claim is withdrawn. Steps 1-12 are arguments about **instruments and
> mechanics**; nothing in them can identify an optimal *strategy*, and the measurement does
> not support the one that was named — `DAA_G4_Leveraged_2X` ranks fourth of seven on Sortino
> over 2010-02…2026-06. Which variant is best depends on what you are asking of it: §8's own
> table shows HAA leading the full window and BAA leading every adverse month. **The report
> prints the rank correlation of that table between disjoint sub-periods, and it is
> approximately zero.**

---

### The external precedent — corrected 2026-07-29, and it was wrong about almost everything

> **⚠️ THIS SECTION PREVIOUSLY MISDESCRIBED KEUNING'S DESIGN ON EVERY MATERIAL POINT**, and then
> used the misdescription as independent corroboration of this repository's own approach. It
> claimed he ran *"offensive UPRO / TMF / TNA — three assets, i.e. a G3"*, that he *"reduced the
> UNIVERSE to assets that all have leveraged products"*, and that he *"went to cash in the
> defensive state"*. **He did none of those things.** The post was read on 2026-07-29 and the
> corrections are below. The claimed convergence was the worst part: a source cited as arriving
> independently at this repo's central rule in fact does the opposite of it.

Wouter Keuning, *"Exploring Smart Leverage: DAA on Steroids"*, `indexswingtrader.blogspot.com`
(TrendXplorer), **December 2018**. What he actually published:

| | Keuning | This repository |
|---|---|---|
| ratio | **2x only. No 3x, no UPRO/TQQQ/TNA/TMF anywhere** | 2x registered, 3x `exploratory` |
| offensive universe | the **full R12**, with 5 of 12 swapped for 2x: SPY→**SSO**, QQQ→**QLD**, IWM→**UWM**, VNQ→**URE**, TLT→**UBT**. VGK, EWJ, VWO, GSG, GLD, HYG, LQD stay at **1x** | universe RESTRICTED to what executes at one uniform multiple |
| he calls it | a *"limited double leverage setup"* | RULE 1 calls it a mixed-ratio sleeve and refuses it |
| defensive sleeve | SHY at 1x **and `UST` — the 2x 7-10y treasury ETF** | RULE 3: defence is never levered |
| canary | VWO / BND, signal-only, unleveraged | same |
| signals | computed on the **1x** tickers, leverage applied at execution | same |
| T / B | T = 6 of 12, B = 2 | unchanged in the wraps |

**It is not implemented here, and it cannot be without breaking three rules.** This is the
honest outcome of the research rather than a reason to bend anything:

1. **RULE 1 (leverage homogeneity).** Seven of his twelve offensive assets execute at 1x and
   five at 2x, so effective leverage drifts with the monthly draw — between 1.0x and 2.0x
   depending on which assets momentum happens to select. That is not a side effect of his
   design; *"limited"* double leverage **is** the design. It is the exact configuration
   `validate_universe` was written to refuse, and the reason the G6/G8 wraps were deleted here.
2. **RULE 2 ($100M AUM floor).** `URE` (~$55M) and `UBT` (~$65M) are both already in this
   repository's `REJECTED_MAPPINGS` for failing that floor. His universe depends on both.
   **Keuning names this problem himself:** *"Liquidity, low trading volumes, and
   assets-under-management requirements limit the practical application of leveraged assets."*
3. **RULE 3 (defence is never levered).** `UST` is a 2x product sitting in his defensive
   sleeve. `DAA1_G12` was deleted from this registry on 2026-07-28 for precisely that, and
   `assert_unlevered_defensive` refuses to price it.

**RULE 5 is NOT among the objections, and an earlier claim that it was is withdrawn.** His
`T = 6` against `B = 2` gives a ladder that resolves every canary count — better than any wrap
here. That claim was made without reading his `T`; the retraction stands and the actual
objections are the three above.

**What he does corroborate, stated narrowly:** signal/trade separation (momentum computed on the
1x series, leverage applied only at execution) and an unleveraged signal-only canary. Both are
enforced here. He does **not** corroborate universe restriction — he is the counterexample to it.

And his own conclusion is worth more than any of the above: **"Results are therefore purely
hypothetical and no investor could have attained these results."** He also names the mechanism
this document calls decay — *"Daily re-leveraging combined with high volatility creates
compounding issues, often referred to as the 'constant leverage trap'"* — and argues smart
leverage works only in lower-volatility regimes, which is the same claim §6 makes and §6 now
qualifies with a measurement.

---

## References

1. Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Technical Journal, 35(4), 917-926.
2. MacLean, L.C., Thorp, E.O. & Ziemba, W.T. (2011). "The Kelly Capital Growth Investment Criterion." World Scientific.
3. Keller, W. (2018). "Defensive Asset Allocation." SSRN 3212862.
4. Keuning, W. (December 2018). "Exploring Smart Leverage: DAA on Steroids." TrendXplorer,
   `indexswingtrader.blogspot.com/2018/12/exploring-smart-leverage-daa-on-steroids.html`.
   *A 2x "limited double leverage" study on the full DAA R12 universe — NOT the 3x,
   reduced-universe design this document credited him with until 2026-07-29. See "The
   external precedent" above for what he actually specified and why it is not
   implemented here.*
5. Cheng, M. & Madhavan, A. (2009). "The Dynamics of Leveraged and Inverse Exchange-Traded Funds." Journal of Investment Management.
6. Ayres, I. & Nalebuff, B. (2010). "Lifecycle Investing: A New, Safe, and Audacious Way to Improve the Performance of Your Retirement Portfolio." Basic Books.
