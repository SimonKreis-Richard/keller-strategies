# KNOWN_GAPS.md — what this engine cannot establish

A backtest is only trustworthy to the extent that its limits are stated. This file is the
standing, permanent list of them: things the engine **cannot** determine, things it
deliberately does **not** model, and findings that were **deleted with their code rather than
fixed**. It is not a TODO list. Most entries here will never be closed, and pretending
otherwise is how a limitation turns into a claim.

Created 2026-07-28 during the remediation of an adversarial audit; extended 2026-07-29
after a second and a third. **This file is now the whole record.** The audit findings, the
work queue and the remediation report were deleted on 2026-07-29 — process scaffolding, not
project documentation, and every conclusion they carried that still matters was distilled
here. The rest lives in `git log`, which is where a record of *how* the work was done
belongs.

---

## 1. Sample and history

**No pre-2000 history, and very little before 2007.** The store downloads from 2000-01-01 and
almost every ticker is an ETF with a 2003-2010 inception. Since 2026-07-29 four of them are
extended backwards — see §2, *Constructed history* — which buys the 2008 crisis for the
families that do not hold high yield and buys nothing at all for the ones that do.
Consequences:

- Keller's published results run Dec-1970 → Dec-2022 on index data. **No result in this repo
  is a reproduction of them**, and no test asserts equality — only that the *shape* is not
  impossible. `tests/test_anchors.py::TestPublishedAnchor` says so in its docstring.
- The 1970s inflation regime, the 1987 crash, and the 1990s bull market are all outside the
  sample. A strategy whose protective logic was designed against those regimes has never been
  tested against them here.
- Only two genuine equity bear markets sit inside the full window (2007-2009, 2022), and only
  one of those predates most of the leveraged products.

**The effective sample is far shorter than it looks.** 36<!-- facts:registry.n_registered:int --> variants over ~18 years is not 400
years of evidence: the report's own participation ratio puts the suite at **2.65<!-- facts:selection.participation_ratio:num2 --> effective
independent bets**, with PC1 explaining 59.4%<!-- facts:selection.pc1_share:pct1 --> of the variance.

**Leveraged variants have no bear-market history at all.** No 3x product predates 2008-11
(UPRO 2009-06, TQQQ 2010-02); the 2x products used here mostly start 2006-2007. The regime
panel therefore prints `n/a (inception …)` for most of the GFC, and that gap is
permanent — it cannot be filled by simulation without inventing the thing being tested.

## 2. Data provenance

**The cache used to hold two adjustment vintages at once, and it corrupted a live signal
(fixed 2026-09-01).** `Adj Close` is a total-return series, so the day a fund goes ex-dividend
Yahoo rescales every bar before that date. The incremental refresh rewrote only a trailing
90-day window, so after any distribution the cached history was spliced: recent bars carrying
the new adjustment, older bars the old one. Nothing about the join looked wrong — every price
was plausible and the curve was smooth — but any return whose endpoints straddled it was
measured across a discontinuity no market ever traded, and always in the same direction, since
stale bars are missing adjustments and are therefore too HIGH. Momentum consequently read too
LOW.

Measured on TIP at the 2026-08-31 decision: `r6` and `r12` each understated by 0.73pp and the
13612U canary score by 0.36pp. At the 2026-07-31 decision the same defect **flipped the HAA
canary from alive to dead**, sending the live portfolio to cash a month early. Nine tickers
were affected on the working cache — every monthly distributor in the universe (AGG, BIL, BND,
HYG, IEF, LQD, SHY, TIP, TLT), including the cash ticker the risk-free series is built from.

**How far it reached, measured rather than asserted.** Only a window STRADDLING the seam is
wrong; a window entirely on one side of it is internally consistent. The seam sits about 90
days behind the run date, so a historical backtest is distorted only across its final ~12
months and long-run metrics barely move — but a LIVE monthly decision always sits on the seam,
which is why the live path is where this actually bit. Replaying every decision from 2005-01 to
2026-08 with one distribution cycle of staleness on the lagged legs: **10 of 260 baskets change
(3.8%)** — six canary flips and four single-asset substitutions. The error is one-directional:
stale bars are too high, momentum reads too low, so every flip is toward MORE cash. The defect
can make this repository hold cash it should not have held; it cannot make it hold risk it
should not have held.

**It is now checked on every run, offline.** `PriceStore._verify_adjustment_vintage` tests a
property of the vendor's data model rather than of this code: `f = adj_close / close` is the
cumulative dividend-adjustment factor, distributions only ever scale EARLIER bars down, and raw
closes are untouched — so **f can only rise with time**, and two vintages spliced together put a
downward step in it exactly at the seam. Measured across 37 tickers: **108 ms, no network**, worst
genuine step −2.11e-06, and the defect that flipped the canary shows up at −0.0073 — a
signal-to-noise ratio of about 3450. Because it needs no network it also works on the paths that
hid the bug: offline, `download=False`, and inside the refresh throttle.

Two exclusions were found by measurement, not by reasoning. A **constructed span** is donor data
on the donor's own price scale, so its junction is a legitimate level change: on the working
store those junctions step −0.126 (VWO), −0.232 (BIL) and −0.018 (BND), each landing exactly on
that span's `real_from`. `VEA` passes only because EFA's factor happens to sit close, so all four
declared spans are excluded rather than the three that fire. The newest `SETTLED_DAYS` bars are
also skipped, where a dividend can be announced and not yet applied.

The verdict is **reported** in the backtest header and **refused** on the live path — per
strategy, scoped to the tickers that entered that decision, including its sleeves and canary
rather than only what it holds. A store that verified nothing reports `not_applicable`, never
`ok`, and the live path refuses that too: sizing orders from data nothing has checked is exactly
the 2026-09-01 situation. The verdict and the detector's settings now travel in
`provenance()`, so a report written before this guard and one written after are no longer
indistinguishable.

**Why five rounds of QA missed it.** Every test in the suite reads frozen fixtures, and every
audit so far reviewed the code and reproduced the repo's numbers *from the repo's own cache*.
Nothing in the process ever compared the cached panel against the vendor it came from, so a
data layer quietly disagreeing with Yahoo was outside what any of it could see. The 2026-08-01
audit's third structural recommendation was precisely a vendor cross-check; it was deferred as
"no urgency". **A self-consistency test cannot see a data layer that is wrong in a
self-consistent way.** `tests/test_guards.py::TestTheAdjustmentVintageStaysConsistent` now pins
the invariant, and `PriceStore` re-downloads in full any ticker whose adjustment was restated.


**Yahoo serves the CURRENT vintage, not point-in-time.** `Adj Close` is back-adjusted for
every dividend and split announced since. A backtest therefore sees adjustment factors that
did not exist at the decision date. Two things were measured on this and **found immaterial**,
which is worth recording so nobody re-opens them:

- Delaying every dividend by two trading days moved HAA-G8 CAGR by **−0.0019 pp** and changed
  **zero** allocations.
- Re-running on a fresh Yahoo download versus the cache changed **0 of 120** allocations and
  **< 0.00001 pp** of CAGR.

So dividend look-ahead and Yahoo restatement are not material here. What remains unverifiable
is *survivorship*: a delisted product simply is not in Yahoo, so a universe assembled today is
a universe of survivors.

**No point-in-time macro data.** FRED serves revised UNRATE vintages, not what was published
at the time. This mattered for LAA and RAA, which are now deleted — restoring either means
restoring this caveat, and doing it properly needs ALFRED vintages, not FRED.

**Constructed history — four tickers, all flagged, all pre-2008.** Nothing in this repository
could be measured through the onset of the 2008 crisis, because the binding constraint was
never a strategy: it was a fund's inception date. Two narrow mechanisms address that, and both
are declared in `common/data_engine.py`, reported by `provenance()`, and carried in every
manifest:

| Ticker | Extended | With | Note |
|---|---|---|---|
| `VEA` | 2001-08 .. 2007-07 | `EFA` | both tracked MSCI EAFE over the spliced years |
| `VWO` | 2003-04 .. 2005-03 | `EEM` | both tracked MSCI EM over the spliced years |
| `BND` | 2003-09 .. 2007-04 | `AGG` | both track the Bloomberg US Aggregate |
| `BIL` | 2000-01 .. 2007-05 | `^IRX` | **synthetic**: 13-week T-bill discount rate, converted to a bond-equivalent return, accrued ACT/360, less BIL's own 0.1356% expense ratio |

Three fences, each asserted by `tests/test_guards.py::TestConstructedHistory`: a constructed
span always ends **before** its ticker's real first observation, so nothing invented can reach
the live order path; the splice is chain-linked on level, so the junction day carries the
donor's *return* and no jump between price scales; and the rate symbol is deleted from the
store once consumed, because `^IRX` is a yield and a yield must never be rankable or holdable.

**What is honest about this and what is not.** The three fund splices are weak constructions:
donor and recipient tracked the *same index* over the spliced years — VEA and VWO only moved
to FTSE in 2013, long after — so what is spliced is the same exposure through a different
wrapper. The synthetic bill is a stronger claim: it is a price series **no market produced**.
It is built from a published rate with the fund's own fee subtracted so it meets BIL without a
step, and the alternative was worse — SHY returned **+6.62% in 2008** against BIL's **+1.59%**,
so splicing SHY would have handed every risk-off month of the crisis a bond rally and
manufactured the very result being measured. But it remains constructed data in a repository
whose rule is *refuse rather than approximate*, and that is why it is fenced, flagged, and
written down here rather than mentioned in a commit message.

**The RISK-FREE RATE is partly constructed too, not just the tradable cash sleeve.**
`BIL`'s synthetic span feeds `build_rf_series`, so **87 of 305** monthly risk-free
observations are accrued from `^IRX` rather than realised, and one earlier month is a flat
3%/yr assumption. Every Sharpe, Sortino and UPI for a strategy whose window opens before
2007-05 is net of that series. The report header now names the three provenances separately;
until 2026-07-29 it called all of them "realised", which was the same class of error this
file exists to prevent — and it was introduced by the very commit that fenced everything
else. The fence is the reason: `_extend_history` deliberately makes a constructed span
indistinguishable from real history *for* `first_tradable_date`, because the coverage guard
needs exactly that, and `build_rf_series` used the same call as its honesty test.

## 3. Execution realism — what is modelled and what is not

Modelled: fill at the session after the decision (`next_open`), one-way cost charged per leg
on the notional actually traded, position drift between rebalances, an explicit cash account,
margin interest day-counted on the debit balance actually drawn.

**Not modelled, in rough order of how much they would cost:**

- **Slippage, spread and market impact.** Cost is a flat 0.1%/side. A real fill on a thin
  leveraged ETF at month-end is worse than that, and the model has no size dependence at all.
- **Taxes.** Every figure is pre-tax. A 7-15× annual turnover strategy in a taxable account
  is a materially different proposition from the same strategy in a registered one.
- **LETF tracking error and borrow.** Leveraged-ETF results use the products' real price
  history, so their financing and decay are genuinely included — but their *future* tracking
  error, expense changes, and the risk of issuer closure are not. The $100M AUM floor in
  `common/letf_mapper.py` is a mitigation, not a model.
- **Intramonth margin calls.** Margin leverage is applied at month-end and interest accrues to
  the next rebalance. A mid-month drawdown that would have triggered a real margin call is
  invisible: the backtest simply rides through it. **Any levered result should be read as an
  upper bound.**
- **`run_ledger` never tests a maintenance requirement — at all, not just intramonth.** Stated
  separately from the line above because it is a stronger claim than that one made. The ledger
  draws a debit balance (`debt`), capitalises interest on it, and de-levers with the signal, and
  there is no comparison of equity to `m × assets` anywhere in the repository. So a levered
  backtest that a broker would have closed out at the trough reports a CAGR instead. Since
  2026-07-30 the risk is priced **beside** the engine rather than inside it, by
  `common/margin_sizing.py`; nothing in `run_ledger` changed, and the levered rows in the main
  table are exactly as optimistic as they were. Since 2026-07-31 that pricing is no longer
  optional reading: `common/leverage_advice.py` drives it on every run, so the report carries a
  `SUSTAINABLE MARGIN LEVERAGE` section and the dashboard a `Max margin` column beside each row
  it applies to. The gap is unchanged — what changed is that you can no longer read the levered
  table without the number that says what the broker would have done.
- **The evidence behind that sizing was computed and shown to nobody until 2026-08-01.** An
  audit of this repository's own output found that `ResultRecord.report()` — which formats the
  Sharpe haircuts, the drawdown decomposition and the block sensitivity — had exactly one
  caller, and it was a test. `compare_table`, the only renderer with a consumer, prints the
  twelve columns needed to ACT and dropped every column needed to DOUBT. Three of the omissions
  defeated their own stated purpose: `deflated_sharpe_probability` is the module's answer to
  "is this Sharpe real given that N variants were searched" and reached no screen at all;
  `block_sensitivity` exists, in the module's own words, "so the one genuinely arbitrary choice
  is visible instead" of hidden, and was hidden; `dd_factor_sample` and `dd_factor_intraperiod`
  are the two factors the adjusted drawdown is deliberately composed of rather than being one
  opaque number, and only their product was shown. `margin_sizing.evidence_table` now prints
  all of them, under the sizing table in the report and inside its dashboard panel.
- **The intra-period gap is now measured, and it is large.**
  `margin_sizing.intraperiod_max_drawdown` values each month's *held* allocation on daily
  prices. Every registry entry's daily
  drawdown exceeds its month-end drawdown, by +0.2% relative (`VAA_G4`) to +59.7% (`HAA_G12`,
  −8.4% against −13.4%); `HAA_G3_Leveraged_2X` posts the deepest absolute gap, −35.7% on month
  ends and −51.7% on the daily path of the same holdings. The month-end figure
  in the main table is the one every other document in this repository quotes.
- **Integer shares in backtests.** The backtest allocates continuous weights. The live sizing
  path *does* compute whole shares across accounts, so live and backtest differ slightly by
  construction.
- **Order rejection, halts, and settlement.** Not modelled.

## 4. Statistical limits

**The leaderboard has no demonstrated predictive content.** Every report prints the Spearman
rank correlation of the ranking metric between disjoint sub-periods. It lands near zero
(measured: −0.30 / −0.14 / +0.17 across three sub-periods of the full registry) with **no**
strategy present in every sub-period top-5. The ranked table is a *description* of what each
regime rewarded. It is not a selection procedure, and this file exists partly to stop it being
read as one.

**Date selection: nothing is chosen any more, and here is exactly what that means.** The
`START_DATE` / `END_DATE` settings were **deleted** on 2026-07-28. They are inert if left in a
personal config, and the loader says so.

| Instrument | Window | Chosen by |
|---|---|---|
| Regime partitions (NBER / S&P 500 / FOMC / CPI) | each strategy's **full** measurable history, cut at frozen boundaries | **frozen in code** |
| Named stress episodes (9) | each strategy's **full** measurable history | **frozen in code** |
| Before/after publication | full history, split at the family's publication date | **frozen in code** |
| Era floor (2004-11) | first month a STRATEGY exists — DAA_G4 / VAA_G4, bound by GLD 2004-11-18 | **derived from data** |
| Ranked table, chart, CAGR/Sharpe/Sortino/UPI | era floor → last complete month, raised to the latest inception among the strategies compared | **derived from data** |

Three properties are asserted by `tests/test_eras.py` and will fail the build if they regress:
every month of the era belongs to **exactly one** segment of each partition; the segments
compound back to the era's own return; and every row of the ranked table spans **identical**
months, with the binding strategy named. `tests/test_guards.py::TestRegimePanelIgnoresTheRequestedWindow`
still guards the panels against inheriting the ranked window.

**What this does NOT fix.** Removing the date box removes *your* discretion, not every
degree of freedom:

- **The era floor is itself a limitation, and its nature changed.** It used to be set by a
  product (BIL, 2007-05); since 2026-07-29 it is set by a download parameter
  (`DATA_START_DATE`). Lowering that parameter buys nothing: before 2000 only SPY, QQQ and EWJ
  exist, and no strategy here can be built from those three.
- **The segment boundaries are ex-post.** The NBER dated the 2020 trough in July 2021. These
  panels describe behaviour in a regime; they say nothing about identifying one in real time,
  and no part of the engine feeds them back into a signal.
- **The inflation layer is hand-entered** from the published BLS CPI-U series. It is the only
  segmentation no test in this repository can re-derive from its own data. The equity-cycle
  boundaries *are* re-derived: `TestEquityCycleAgainstRealPrices` recovers the 2022 peak
  (2022-01-03) and trough (2022-10-12) from the frozen fixture with a mechanical drawdown
  rule and checks them against the dates in `common/eras.py`.
- **A ranked table is still not a selection procedure**, over any window. See the paragraph
  above about rank correlation near zero. Rank to describe; select on structure.

**CLOSED 2026-07-29 — for half the registry.** This file used to say, correctly at the time,
that *HAA cannot be measured through the GFC at all*. That is no longer true. The history
extension in `common/data_engine.py` pushed BIL, VEA, VWO and BND back past their fund
inceptions, and the binding constraint moved from a product to the market's own product
history. Where each entry now starts, measured:

| First measurable month | Bound by | Entries |
|---|---|---|
| 2004-10 | AGG 2003-09 | `DAA_G4`, `VAA_G4` |
| 2005-02 | TIP 2003-12 | `HAA_G1_Simple` |
| **2007-03** | **DBC 2006-02** | `HAA_G12`, `HAA_G8_Balanced`, `BAA_G4`, `BAA_G1_SPY`, `GTAA_G5` |
| 2008-03 | UWM 2007-01 | the four G3 leveraged sleeves |
| 2008-05 | **HYG 2007-04** | `DAA_G12`, `DAA_G6`, `VAA_G12`, `BAA_G12`, `PAA2_G12` |
| 2008-07 | REM 2007-05 | `DM_G8_Composite` |
| 2010-01 | UGL 2008-12 | the four G4 leveraged sleeves |

The S&P peaked **2007-10-09**. Everything at 2007-03 or earlier can therefore be read through
the entire bear market; everything below that line cannot, and never will be.

**The line is drawn by one ticker: high yield.** HYG (2007-04-11) is the oldest US high-yield
ETF that exists — JNK is 2007-12-04 — so *every strategy holding high yield is bounded at
2008-05 by a fact about the ETF industry, not about the strategy*. Same for mortgage REITs
(REM, 2007-05) and broad commodities (DBC, 2006-02, which is what sets the 2007-03 line).

**What it cost to learn this.** `HAA_G12`'s maximum drawdown was **−7.99%** measured from
2008-09. Measured from 2007-03, through the crisis it now actually contains, it is
**−11.24%** — and its levered 1.3x variant went from −10.63% to −14.52%. Nothing about the
strategy changed. The old number was the drawdown of a sample that began after the crash.

**Nothing reaches the dot-com bust.** EEM 2003-04, TIP 2003-12, GLD 2004-11, DBC 2006-02: the
multi-asset universes did not exist. The `bear_dotcom` segment and the `dotcom` episode were
both REMOVED on 2026-07-31, when the era floor moved to 2004-11 — the first month a strategy
rather than a passive benchmark can be measured. Until then they were populated by
`SPY_Benchmark` alone, and a leaderboard of one is a fact about coverage wearing the clothes
of a comparison. Restoring them means lowering the floor back below 2003 and accepting eight
single-row segments with it.

**The true number of trials is unknowable.** The repo has a handful of commits, so the search
that produced the current universes, parameters and cut list cannot be reconstructed. Every
multiple-testing correction in the report is therefore a lower bound on the real penalty.
And since 2026-07-29 **`role` is itself one of the levers that sets the counted number**:
whether an entry is a "trial" is decided by an attribute this project assigns, and the twelve
3x entries are excluded from the count (19, not 31) on the stated ground that nobody here
intends to hold 3x. That ground is the right criterion and it is credible today — but the
mechanism would equally permit a future exclusion on a weaker rationale, so it is named here
as a degree of freedom rather than left implicit (2026-07-30 audit, Q5).

**Timing luck is real, it is large, and month-end is the luckiest draw on drawdown.**
Month-end rebalancing is one schedule out of ~20 equally defensible monthly calendars. Run
`tools/timing_luck.py` to reproduce; measured 2013-01 → 2024-12 with `next_open` applied
uniformly to all 20 schedules:

| | month-end | spread across 20 schedules | month-end's percentile |
|---|---:|---|---:|
| HAA_G12 CAGR | 7.80% | 7.80% … 10.22% | **worst of 20** |
| HAA_G12 MaxDD | **−4.89%** | −26.62% … −4.89% | **best of 20** |
| HAA_G8 CAGR | 7.25% | 7.06% … 9.27% | 15th |
| HAA_G8 MaxDD | **−6.64%** | −30.94% … −6.64% | **best of 20** |

**The headline drawdown of the HAA family is the single most flattering calendar available.**
The same signal, rebalanced on the 10th trading day instead of the last, drew down −26.6%
(G12) and −30.9% (G8). Nothing about the strategy changed; only the day did. Any statement of
the form "this strategy's worst loss was about 5%" is a statement about the calendar.

This also settles the disagreement between the two audits, which reported month-end as
respectively the 5th and the 80th percentile on CAGR. Both were right about their own window:
month-end is 15th percentile on CAGR over 2013-2024 and 65th over 2015-2024. The CAGR ranking
is unstable; the drawdown ranking is not.

**Post-publication samples are short.** HAA has ~2 years of out-of-sample history, BAA ~3.
Nothing in this repo can distinguish a 2-year underperformance from a broken strategy.

**The ranking is now cross-validated, and the result is middling.** Since 2026-08-01
`common/robustness.py` computes the probability of backtest overfitting (Bailey, Borwein,
López de Prado & Zhu 2017) by combinatorially symmetric cross-validation: sixteen blocks of the
shared 2008-07..2026-06 window, all 12 870 equal splits, in-sample winner looked up
out-of-sample. **PBO = 34.9%<!-- facts:robustness.pbo_strategy:pct1 -->** — better than a coin flip, well short of a
procedure you would lean on. `HAA_G12` takes the in-sample crown in 64.3% of splits and
`DAA_G6` in 9.3%, so the top row is not stable, and the resampled leaderboard puts `HAA_G12`
in the top three of 87.4%<!-- facts:robustness.top.0.p_top_k:pct1 --> of bootstrap histories against 48.8% for the runner-up.

Every figure in this section is annotated with the fact it comes from and checked against
`tests/fixtures/run_facts.json` by `tests/test_paper_rules.py`, so it cannot drift out of the
prose again — which it has done twice. It read **35.9%** until 2026-08-02, when the section was
found netting Sharpes at rf = 0 beside a leaderboard netted at the realised rate (AUD-02), and
**42.2%** until 2026-09-01, when the cache was found holding two adjustment vintages. The
present figure is measured on repaired data over a window three months longer, so the move
combines both; it has not been decomposed, and saying so is cheaper than pretending otherwise.

Two limits, and the first is the one that matters:

* **PBO cannot see Keller's search, only ours.** It measures whether OUR ranking of these
  entries is stable. The choices made before the data reached this repository — a 13-month
  lookback, a top-N, a breadth threshold, all fitted on these same decades — are invisible to
  it. A low PBO would say "our leaderboard is reproducible", never "these strategies are not
  overfitted". Only a parameter sweep re-running the engine can address the second claim, and
  this repository does not do that yet.
* **No resampling repairs a biased sample.** A bootstrap draws from the distribution it is
  given. If these twenty years were generous to US assets, every one of the 2 000 resampled
  paths is generous too. Nothing in this repository forecasts, and nothing here should be read
  as an answer to "what if the next twenty years are unkind" — that needs different data or a
  stated assumption, not a simulation.

**The family view sharpens the claim.** Since 2026-08-01 both measurements also run POOLED
by family and by de-risking mechanism, and pooling improves the cross-validation
monotonically: strategy-level PBO **34.9%<!-- facts:robustness.pbo_strategy:pct1 -->**, family-level **32.2%<!-- facts:robustness.pbo_family:pct1 -->**,
mechanism-level **21.9%<!-- facts:robustness.pbo_mechanism:pct1 -->**. (Under the
pre-AUD-02 rf = 0 convention the family level looked like a step *backwards* — 38.0% against
35.9% — an artefact of the rate flattering the low-vol variants that dominate their families'
medians.) The resampled ranking confirms it: the HAA family is top-3 in **89.3%<!-- facts:robustness.family_rank.HAA.p_top_k:pct1 -->** of alternative
histories (first in 70.7%<!-- facts:robustness.family_rank.HAA.p_first:pct1 -->), and the two canary mechanisms dominate — dedicated-basket canaries
(DAA+BAA+their wraps) are top-3 in 72.0% of histories, breadth protection 25.1%, absolute
momentum 12.7%, per-asset trend 0.2%. "Hold a canary-protected family, and prefer HAA"
survives resampling; no finer claim does.

**The "Max margin" column used to depend on which entries you ran — fixed 2026-09-01.**
The multiple-testing haircut derived its trial count and Sharpe spread from *the run's*
population, so a 3-entry run recommended `HAA_G12` at 1.35x where the 36-entry run said
1.27x (measured 2026-08-01, AUD-06): the smaller run received the milder haircut, in the
flattering direction, even though the search that produced the pick was always the full
registry. The population now comes from `tests/fixtures/run_facts.json`, written by
`tools/emit_facts.py` from a full-registry run, so ticking fewer entries no longer buys a
gentler answer. When that artefact is missing the code falls back to the run's own
population and the assumption lines SAY SO — a silent fallback would restore the defect and
hide it, which is worse than the defect.

**The leverage frontier is month-end and constant-f, and says so.** The same resampled
histories walked at every leverage level under the ledger's monthly-reset policy, with each
entry's own maintenance requirement (the crisis-stressed one CAP 1 used). Measured: margin
calls at month-end granularity become material only around **2.0x** for the unlevered
registry (f@1% between 1.80x and 2.05x), while the sizing advice sits at 1.0–1.3x — the gap
IS the intraperiod factor plus the k×DD_adj safety margin, which the monthly paths cannot
see. At every recommended level, P(call) across all 2 000 histories is 0.0% — the two
methods, one closed-form from a drawdown quantile, one path-walking, bracket the same band
from opposite sides. On top of a 2x LETF the frontier collapses to **1.05x** (m=0.90),
which is the numerical form of the standing doctrine that margin and 3x wraps do not stack.

## 5. Fidelity gaps in surviving strategies

Since 2026-07-28 every registered entry carries a declared label, printed in the report's
`Type` column and asserted by `tests/test_anchors.py`: **faithful** (the author's universe and
parameters), **proxy** (same, on substitute funds), **custom** (a universe nobody published),
**control**, **benchmark**. Sixteen entries that could not earn one of those labels honestly
were deleted rather than annotated — see §7. What remains, and what is still imperfect:

**Since 2026-07-29 every entry must also carry a `source` citation** — paper, section or
figure — asserted for all 36 by `tests/test_anchors.py`. The reason is that **four** labels
have now been found wrong (`DM_G8_Composite`, `HAA_G1_Simple`, `VAA_G4`, and on 2026-07-30
`BAA_G1_SPY`), **all four in the under-claiming direction**, while the only test on them
checked that the value was one of three legal strings, which every wrong answer also is. A
citation does not make a label correct; it makes it checkable against the PDF in
`academic-papers/`. The fourth arrived AFTER the citation requirement, which proved the
point the requirement's own paragraph conceded: a citation is checkable by a human and by
nothing in CI. `tests/test_paper_rules.py::TestFidelityAgainstSource` now pins every
registry key's `(fidelity, SSRN number, section/figure)` in a table a reviewer must extend
to register anything — so the fifth mislabel has to get past a diff review naming the paper,
not just a string-membership check.

- **`BAA_G1_SPY` is `faithful`, and was mislabelled `custom` until 2026-07-30.** Its
  docstring asserted "Keller names no single-asset BAA variant"; BAA §5 presents exactly this
  configuration and Fig 11 captions it **BAA-SPY** (`SelO=SPY, NO=TO=1`, same defensive
  seven, same canary, TD=3). The docstring was written by analogy to `HAA_Simple` rather
  than by reading BAA §5. Four for four is no longer a coincidence, it is a bias worth
  naming: when this project guesses a label, it guesses AGAINST itself.

- **`VAA_G4` is `faithful`, and was mislabelled `proxy` until 2026-07-29.** The label rested
  on "the paper specifies `SPY, EFA, EEM, AGG`". The paper's own footnote 11 says otherwise:
  *"We actually used (proxies for) Vanguard ETFs **VEA, VWO, VNQ, and BND** instead of the
  mentioned (and more common) iShares ETFs EFA, EEM, IYR, and AGG, respectively, in nearly
  all our backtest"* — and the DAA paper restates VAA-G4 as `R4 = SPY, VEA, VWO, BND`. The
  code's tickers are Keller's.
- **`VAA_G12` now holds `VWO`/`VNQ`, and held `EEM`/`IYR` until 2026-07-29 — the same
  footnote, applied to one variant and not the other.** That was an inconsistency, not a
  decision, and it survived one round of audit behind a bad defence: *"Keller treats the two
  fund families as interchangeable, so either set is his."* Footnote 11 does not say the
  funds are equivalent. It says **which ones he ran**, and gives a reason that still checks
  out — VWO 0.06% vs EEM 0.72%, VNQ 0.13% vs IYR 0.42% (2026-07). The one part of the
  footnote that no longer holds is "similar AUM's": VNQ is now ~18x IYR's size.
  **The correction cost performance**, which is why it is worth trusting: over 2015-2024,
  CAGR 3.53% -> 3.03%, MaxDD -27.03% -> -28.03%, Sortino 0.39 -> 0.31. Coverage unchanged
  (VWO is chain-linked from EEM by `HISTORY_BACKFILL`; VNQ predates the binding HYG).
- **`GTAA_G5` is `proxy`, and was `faithful` until 2026-07-29.** Faber (2006) times
  *indices* — S&P 500, MSCI EAFE, GSCI, NAREIT, 10-year Treasury — not funds. `DBC` tracks
  the DBIQ index, not the GSCI; `IEF` is a 7-10y fund, not the 10-year note. `faithful`
  claimed correspondence to published tickers that do not exist.
- **`HAA_G1_Simple` is `faithful`, and was `custom` until 2026-07-29.** HAA §6 defines it:
  *"in fig 12 we will look at a very special HAA case with only one risky asset SPY
  (NO=TO=1) ... We will call this special version with only SPY the HAA-Simple"*, with
  `SelD = BIL/IEF` and `SelP = TIP`. Calling it custom claimed the author had not published
  a variant he named in a figure caption. It stays `role = 'control'`, which is a separate
  question: fidelity asks whether anyone published this, role asks whether you would hold it.
- **`DM_G8_Composite` is `proxy`, and it used to be labelled `custom` — wrongly.** It is
  Antonacci's equally-weighted four-module composite from SSRN 2042750 (§§3-6, Table 10): the
  credit, REIT and gold/Treasury pairs that the docs claimed he "never proposed" are his. The
  file had been measured against **Global Equities Momentum**, a different strategy of his,
  and mislabelled on that basis. Corrected 2026-07-29 by renaming, not rewriting. The
  remaining gap is instruments: index series in the paper, ETFs here, so the sample starts
  2008-09 and misses 34 of the paper's 38 years. **Under-claiming is a labelling failure too**
  — this one was nearly used as an argument to delete a paper-faithful strategy.
- **The leveraged G3/G4 universes were never defined by anyone.** The signal math is the
  paper's; the asset lists were invented here so the offensive sleeve executes at one uniform
  multiple. `LeveragedWrapMixin` hard-codes `fidelity = 'custom'` so a wrap can never inherit
  its parent's claim.
- **Not every family admits a leveraged variant, and which ones do is decided structurally.**
  Since 2026-07-29 the rule is: *a wrap may change what is HELD, never what decides to
  DE-RISK* ([`common/letf_mapper.py`](common/letf_mapper.py), RULE 4). VAA and PAA protect by
  counting breadth over their **own** offensive universe, so restricting that universe for
  uniform-ratio LETF execution rebuilds the protection rule out of three US equity ETFs at
  rho 0.79-0.91. Their four wraps were deleted. This is a **fidelity** judgement, not a
  performance one — measured as correlation to the unlevered parent (VAA_G3 0.376, PAA_G3
  0.543, against 0.784-0.823 for the exogenous-canary families) — and it deliberately kept
  `BAA_G3_Leveraged_2X`, which ranked *below* two of the deleted entries on Sortino. A rule
  that cuts a winner and keeps a loser is at least not being applied to the leaderboard.
  RULE 5 removed one more, `DAA_G3_Leveraged_2X`, and then **put it back the same day** — the
  clearest case in this file of a rule stated in the wrong form. `T=1` against `B=2` did round a
  lone dead canary down to zero de-risking, so it held 2x equity through COVID for -35.1%; but
  the fix belonged in the parameter, not the registry. RULE 5 is now constructive —
  `T = max(n//2, B)` — which gives G3 a `T` of 2, restores the middle rung, and makes the state
  unreachable at any universe size. **Deleting a variant to avoid choosing a parameter that had
  a correct answer available discarded information for nothing.** The restored entry ranks
  seventh of nine on Sortino, so the restoration is not a performance argument either.
  **Corrected again 2026-07-30: the rationale had misattributed the T=1 collapse to the
  paper.** DAA n.8 legislates the case directly — "with T=1, CF is simply b/B" — and
  `strategies/daa.py` now implements it (VAA's own, different, §4 T=1 rule likewise in
  `strategies/vaa.py`). The -35.1% figure describes the bare floor formula this repository
  used to run, not the paper's ruleset. `T = max(n//2, B)` stands as a CONCENTRATION
  convention; whether the G3 wrap should revert to T=1 is an open registry decision recorded
  in `strategy_specs/daa.md`, and must not be decided from the measured table there.
- **Keuning's "DAA on Steroids" is NOT implemented, and the reason is his design rather than
  ours.** It was researched on 2026-07-29 (TrendXplorer, December 2018) and it breaks three of
  this repository's leverage rules at once: seven of its twelve offensive assets execute at 1x
  while five execute at 2x (RULE 1 — he calls this a *"limited double leverage setup"*, so the
  mixed sleeve IS the design); it depends on `URE` (~$55M) and `UBT` (~$65M), both already in
  `REJECTED_MAPPINGS` under the $100M floor (RULE 2 — a limitation **he names himself**); and it
  holds `UST`, a 2x treasury ETF, in its DEFENSIVE sleeve (RULE 3 — `DAA1_G12` was deleted here
  for exactly that). **RULE 5 is not among the objections**: his `T = 6` against `B = 2` resolves
  every canary count, and an earlier claim in this project that his design failed RULE 5 was made
  without reading his `T` and is withdrawn. Until 2026-07-29 `LEVERAGE.md` described him as
  running 3x on a reduced UPRO/TMF/TNA universe and cited that as independent corroboration of
  this repo's universe-restriction rule. **He uses no 3x product at all, does not restrict the
  universe, and is the counterexample rather than the corroboration.** His own verdict on his
  backtest: *"Results are therefore purely hypothetical and no investor could have attained
  these results."*
- **RULE 4 does not promise the protection rule is FAST, only that it is the paper's.** HAA
  gained wraps under it on 2026-07-29 and `HAA_G4_Leveraged_2X` became the first leveraged
  entry here to beat SPY on Sortino and UPI — while losing -19.2% in `bear_covid` and -27.7%
  across every adverse month, where `BAA_G4_Leveraged_2X` gained +5.9% and +10.3%. A single
  exogenous canary (TIP) survives the restriction intact and is still the narrowest sensor in
  the canon. See [`LEVERAGE.md`](LEVERAGE.md) §8; do not read that Sortino as an endorsement.
- **`HAA_G1_Simple` and `BAA_G1_SPY` are `control`.** Single-asset degenerate cases, kept to
  show how much of a family's record is its timing rule versus its universe. They appear in
  the ranked table marked `(ctrl)` and are excluded from the selection statistics — a
  portfolio you would never hold is not a trial you ran. `HAA_G1_Simple` regularly ranks in
  the top three, which is the point of having it.

**Defence is never levered — now a hard rule, not a habit.** `DAA1_G12` declared `UST`, a 2x
7-10y treasury ETF, as one of three defensive candidates: its risk-off state doubled duration
risk while the engine reported it as risk-off. It was deleted, and
`common/letf_mapper.assert_unlevered_defensive` now refuses to price any strategy declaring a
leveraged product in its defensive sleeve. A leveraged *credit* product would still be
admissible in the OFFENSIVE universe — where Keller puts LQD — because there it would be
declared as what it is, a risk position.

## 6. Findings DELETED with their code, not fixed

The 2026-07-28 registry cut removed FAA, MAA, EAA, LAA, RAA and CAA. Eight audit findings went
with them. They are recorded here so nobody hunts for a fix that does not exist, and so that
**reintroducing any of these strategies means reintroducing its bug**:

| Finding | Strategy | What it was |
|---|---|---|
| EAA 85.7% uninvested | `EAA_G7` | months allocating only 0.1429 of the book, the rest silently earning 0% |
| MAA stub / concentration | `MAA_G7*` | shrinkage parameters did nothing — `MAA_G7` vs `MAA_G7_TV` measured ρ = **1.000** |
| CAA renormalisation | `CAA_G8_*` | renormalising after clipping broke the very constraint that was clipped |
| RAA NaN canary reads bullish | `RAA_G5` | `any(score <= 0)` is False for NaN, so crash protection was structurally off before BND's 2007-04 inception |
| RAA unrecognised sleeve | `RAA_G5` | `risky_assets` was a sixth attribute name the sleeve resolver never looked at |
| UNRATE point-in-time vintages | `LAA`, `RAA` | FRED serves revised values, never what was published at the decision date |
| FAA/MAA daily estimators | `FAA_G7`, `MAA_G7*` | volatility and correlation estimated from four monthly observations instead of daily data |
| EAA nominal vs excess returns | `EAA_G7` | the paper's formulation uses excess returns; the code used nominal |

The sleeve-vocabulary problem that produced two of these is fixed at the root:
`BaseStrategy.sleeves()` is now a mandatory explicit declaration, asserted for all 36 registry
keys by `tests/test_anchors.py::TestRegistry`.

### The second cut, 2026-07-28 (39 → 25; the registry has since grown to 36)

A separate reduction, on a stated **admission rule** rather than on measured redundancy: an
entry is registered only if it is (1) a universe and parameterisation its author published,
(2) the same on substitute funds, (3) a universe this repo was *forced* to invent so a
leveraged sleeve executes at one uniform multiple, (4) a single-asset control, or (5) a
passive benchmark. Everything else is a degree of freedom wearing a strategy's name.

| Deleted | Why |
|---|---|
| `DAA1_G12` | non-paper defensive sleeve containing **UST, a 2x treasury ETF** — a levered defence. Also the most expensive entry for coverage: it alone dragged the comparison window from 2008-07 to 2011-04 |
| `HAA_G4`, `HAA_G16` | custom sizes (the paper presents HAA-8 and HAA-12); G16 also ρ 0.935 vs G12 while starting a year later |
| `DAA_U6`, `DAA_U15`, `VAA_U6`, `VAA_U15` | custom US-only universes that ranked `BIL` inside the *offensive* momentum universe |
| `BAA_G4_T2` | `TO` twiddle on an unchanged universe, ρ 0.894 — the criterion that had already removed `BAA_G12_T3` |
| `GTAA_G13_Moderate`, `GTAA_G13_Aggressive` | no published Faber source for either, same universe (ρ 0.897); the Aggressive one hybridised Keller's 13612U with Faber's SMA |
| `DAA/VAA/BAA/PAA_G2_Leveraged_2X` | G2 is SPY + QQQ, **ρ 0.92** — a two-asset momentum choice between near-identical assets is not a choice; ρ 0.93 against their own G3 twins |

Two consequences worth stating. First, the cut **lengthened** the comparison window for
everything that survived: the full-registry ranked window moved from 2011-04 to 2010-02, bound
by UGL (2008-12) via the G4 leveraged sizes — and then to **2008-07** on 2026-07-29, when
`RANKED_WINDOW_POLICY` stopped letting `custom` entries set it at all. Those four G4 wraps were
costing all 25 then-registered rows nineteen months, so the headline drawdown column excluded the 2008 crisis
immediately after the history extension had worked to include it. They are still measured, over
their own 2010-02 window, in a separate block that says why it is not comparable. The window is
now bound by `DM_G8_Composite` (REM, 2007-05), a published entry — a constraint about the ETF
industry rather than about this repository's choices. Second, the effective sample did
*not* shrink — the participation ratio was ~3.0 effective bets at 39 variants because they all
trade the same momentum signal on overlapping universes. What the cut bought is a lower
multiple-testing bar and a registry in which every entry can say who published it.

## 7. Corrections to the audit itself

**C1's magnitude was overstated ~5.6×.** The audit reported same-bar execution as worth
+2.09 pp of CAGR (Codex independently +2.47 pp). Both compared `close(t) → close(t+1)` against
`open(t+1) → close(t+1)` — a *shorter holding period*, which drops one overnight leg from the
series without ever crediting it. Measured apples-to-apples against the correct open-to-open
alternative, it is **+0.37 pp** on HAA-G8, +0.87 pp on the 2x variant, and **−0.10 pp** on
VAA_G12. The defect is real; the number was not — and the honest version is the smaller one,
which is the direction an audit least likes to correct in. `tests/test_anchors.py` carries the
measurement.

**m3's conclusion was wrong, in the under-claiming direction.** The audit found that `GEM_G8`
is not Antonacci's *Global Equities Momentum* — correct — and concluded it was therefore a
custom strategy. It is not. It implements the **modular** construction from the paper the
finding itself cites, Antonacci (2012) SSRN 2042750: four modules defending into T-bills,
equally weighted (§9, Table 10). The remediation then wrote a spec asserting three of its
pairs were "asset pairs Antonacci never proposed"; they are his. Corrected 2026-07-29 —
renamed `DM_G8_Composite`, relabelled `proxy`, spec rewritten as `strategy_specs/dm.md`, no
allocation logic touched.

**Two Keller-compliance defects survived BOTH audits and were found by a third, 2026-07-29.**
Every number this repository published for DAA, BAA and PAA before that date was produced by
a rule its paper does not state. They are fixed; the record of what was wrong belongs here.

- **DAA filtered its Top-T to positive momentum.** The paper declines absolute momentum on
  the risky sleeve three times — §8 *"we did not eliminate bad assets in the top T selection
  of risky assets"*, the Conclusions *"EW-Top T ... without intrinsic or absolute
  momentum"*, and n.17, which records it as a variant they **tried and did not adopt**. So
  the code ran the paper's rejected alternative under the label `faithful`. The two rules
  disagreed in 4.1% of `DAA_G12`'s months, 15.7% of `DAA_G4`'s and 31.2% of `DAA_G6`'s,
  concentrated in 2008-11..2009-03 and 2020-03, with a maximum weight difference of a
  complete portfolio.
- **`SMA12` averaged twelve prices; Keller defines it over thirteen.** BAA n.5: the momentum
  *"equals the present price pt divided by the average of the **last 13 asset prices
  including the present** (also noted as SMA13), minus 1"*. One character, and it shifted
  BAA's offensive ranking, its defensive ranking, its absolute test against BIL, and PAA's
  breadth count. `BAA_G12`'s published −11.14% maximum drawdown was −10.70%.

**Why two audits missed them, which is the part worth keeping.** Neither defect was subtle
to *check* — both are one sentence of the source paper against one line of code. They
survived because **no test compared any strategy against an independently coded version of
its paper's rule**. Every strategy test asserted self-consistency: weights sum to one, a dead
canary de-risks, the top four are equal-weighted. A twelve-price average and a
positive-momentum filter satisfy all of those, because the result still looks like a
momentum score. The 2026-07-28 audit signed off *"the Keller core is faithful"* having
verified DAA's canary, `B`, `T` and cash-fraction rounding — and never the selection rule —
and *"SMA12 relative momentum ✓"* without checking the window length.

`tests/test_paper_rules.py` is the structural answer: one file, per family, comparing
allocations against arithmetic written out in the test. The mutation standard it must meet
is stated in its docstring — change `rolling(13)` to `rolling(9)`, drop DAA's cash slots,
swap 13612W for 13612U, invert a BAA canary comparison, and each must break an assertion
**there**, not merely move a number in the golden master. A golden master can only say that
a number changed; it cannot say which rule is right.

This is the only finding in either audit that made the repository claim **less** fidelity than
it had, and it is worth keeping visible for that reason: it survived because the
implementation was checked against the wrong paper by the right author, and because "custom"
reads as the humble, safe answer. It is not safe. A false `custom` label was one step from
being used as an argument to delete a paper-faithful strategy.

---

## Reading rule

If a number in this repository matters to a decision, check it against this file first. The
engine is now honest about *how* it computes; it cannot be honest about what it never saw.
