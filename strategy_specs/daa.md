# DAA — Defensive Asset Allocation

| | |
|---|---|
| **Source** | Keller, W.J. & Keuning, J.W. (2018), *Defensive Asset Allocation (DAA)*, SSRN **3212862** |
| **Local copy** | [`academic-papers/2018-daa-ssrn-3212862.pdf`](../academic-papers/2018-daa-ssrn-3212862.pdf) (29 pp.) |
| **Implementation** | [`strategies/daa.py`](../strategies/daa.py) |
| **Fidelity** | ✅ Faithful for G12. `U6`, `U15` and the leveraged sizes are custom universes. |

> Written from the PAPER, not from the code. Page numbers refer to the local PDF.

## Rules, as published

**Momentum — 13612W** (p. 4). Same 12-4-2-1 weighted score as VAA.

**The key change from VAA: a SEPARATE canary** (pp. 1-5). DAA's contribution is decoupling the
crash signal from the trading universe. Breadth is measured over just two assets — **VWO**
(emerging equity) and **BND** (US aggregate bond) — chosen as early-warning instruments, while
the offensive universe is ranked independently. VAA had to make the same universe do both jobs.

**Cash fraction** (pp. 5, 8-9). Identical machinery to VAA, applied to the canary count:

```
b            = number of canary assets with score ≤ 0   (b ∈ {0, 1, 2})
n_cash_slots = floor( b · T / B )
CF           = n_cash_slots / T
```

With `T = 6`, `B = 2`: one dead canary ⇒ CF = 1/2, two ⇒ CF = 1 (fully defensive). This is the
whole of DAA's protection — graded, not binary.

**Offensive selection — SIGN-AGNOSTIC, and this is the rule most often got wrong.** The top
`T − n_cash_slots` offensive assets by 13612W score, **whatever their sign**, equal-weighted
across `1 − CF`. The paper says so three times:

> §8: *"notice that we did not use absolute momentum (ie. eliminating bad assets in favor of
> cash), except for the number of bad canary assets ... So for DAA, we did not eliminate bad
> assets in the top T selection of risky assets, although we reduced the top T"*

> Conclusions: *"DAA is equivalent to VAA ... so EW-Top T (T<=N, long only), **without
> intrinsic or absolute momentum**."*

> n.17 records absolute momentum on the risky top T as a variant they **tried** — *"slightly
> better returns with slightly worse drawdowns"* — and did not adopt.

The canary alone decides how much goes to cash; the offensive sleeve's own breadth never
does. That separation is the entire difference between DAA and VAA, which counts breadth over
its offensive universe. This repository applied a `> 0` filter until 2026-07-29 — the paper's
rejected variant, under the label `faithful`. The two rules disagreed in 4.1% of `DAA_G12`'s
months, 15.7% of `DAA_G4`'s and 31.2% of `DAA_G6`'s. A missing score is still excluded: that
is missing data, not bad momentum, and must not be rankable.

**Defensive allocation.** The cash fraction goes entirely to the single best of `{SHY, IEF,
LQD}` by 13612W score.

**NaN policy.** A canary with no score counts as dead. This is the direction that de-risks; the
opposite convention (which RAA used, now deleted) structurally disables crash protection over
any period preceding a canary's inception.

## Registered variants

| key | offensive universe | T | B | defensive |
|---|---|---:|---:|---|
| `DAA_G12` | SPY, IWM, QQQ, VGK, EWJ, VWO, VNQ, GSG, GLD, TLT, HYG, LQD | 6 | 2 | SHY, IEF, LQD |
| `DAA_G4` | SPY, VEA, VWO, BND | 4 | 2 | SHY, IEF, LQD |
| `DAA_G6` | SPY, VEA, VWO, LQD, TLT, HYG | 6 | 2 | SHY, IEF, LQD |

Canary is `{VWO, BND}` throughout. LQD is dual-role in G12 (offensive universe *and* defensive
candidate); the canary resolves which role it plays each month.

`DAA_G3_Leveraged_2X` and `DAA_G4_Leveraged_2X` run this exact signal on a restricted universe
that is executable at one uniform multiple, then map the offensive sleeve to LETFs. The signal
math is Keller's; the universes are not — see [`../LEVERAGE.md`](../LEVERAGE.md).

### How `T` is chosen in the wraps: `T = max(n//2, B)`

`T` does **two** jobs in DAA, and that is the whole subtlety. It is the number of offensive
slots held, and it is the denominator of the cash ladder `floor(b·T/B) / T` — which is Keller's
*Easy Trading* **rounding** of the real rule `CF = b/B`. A wrap sets `T` from the restricted
universe size, so a universe chosen for *LETF availability* ends up setting a parameter that
governs *protection*.

Taking `T = n//2` alone, G3 gives `T = 1` against `B = 2`:

| | b = 0 | b = 1 | b = 2 |
|---|---|---|---|
| `DAA_G12` (T=6, B=2) | 0% | **50%** | 100% |
| `DAA_G4_Leveraged_2X` (T=2, B=2) | 0% | **50%** | 100% |
| `DAA_G3_Leveraged_2X` (T=2, B=2) — **as built** | 0% | **50%** | 100% |
| a T=1 G3 wrap — **what `n//2` alone would give** | 0% | **0%** | 100% |

With `T = 1` **on the bare floor formula** one dead canary produces no de-risking at all. VWO
went negative in 2020-01 while BND stayed positive, so `b` held at 1 and such a wrap carried
2x equity through COVID for **−35.1%**, and for three months in 2011 for **−25.0%**.

> **Second correction of the record (2026-07-30): the paper legislates `T = 1` directly, and
> this section previously failed to say so.** DAA **note 8**: *"We also have added the rule
> that with T=1, CF is simply b/B, in line with the ET idea, with b the number of bad
> assets."* The rounding is the paper's convenience for `T ≥ 2`; at `T = 1` the paper's own
> rule keeps all three rungs (0% / 50% / 100%) while holding a single asset at `1−CF`.
> `strategies/daa.py` implements n.8 since 2026-07-30 (no registered entry reaches it — all
> have `T ≥ 4` — but a `faithful` label covers the rules, not just the reachable rules). The
> −35.1% / −25.0% figures above were measured on the bare floor this repository used to run,
> **not** on the paper's ruleset. The prior paragraph's conclusion — "the floor at `B` is what
> prevents this" — was therefore the truth about this repository's code and a misattribution
> about the paper.

**What `T = max(n//2, B)` is now: a concentration convention.** With n.8 implemented, a `T=1`
wrap resolves every canary state and would be admissible; the convention stands because
`T ≥ B` keeps every wrap on the same arithmetic as the registered parents and holds a less
concentrated book. Whether `DAA_G3_Leveraged_2X` should instead revert to `T = 1` is an
**open registry decision** — and it must not be decided from the measured table below, which
exists to document the mechanism, not to pick the parameter. The rounding and the rule agree
whenever `T/B` is a whole number, which is why G12 and G4 were never affected.

> **First correction of the record.** On 2026-07-29 `DAA_G3_Leveraged_2X` was **deleted** under
> RULE 5 read as a filter ("refuse any variant with `T < B`"), and restored the same day once
> the rule was re-read as choosing `T`. The variant was never defective — the parameter was.
> `assert_protection_survives_restriction` is retained as a backstop (exempting `T = 1` since
> the n.8 implementation).

**Measured, same data, only `T` differs (2008-03…2026-06):**

| | T=1 | T=2 (as built) |
|---|---|---|
| CAGR | 22.89% | 19.05% |
| Max drawdown | −46.58% | **−35.70%** |
| Sortino | 1.26 | **1.36** |
| UPI | 1.49 | 1.46 |
| Volatility | 33.45% | **23.48%** |
| COVID (2020-02…03) | −35.07% | **−15.99%** |
| 2011-07…09 | −24.98% | **−7.24%** |
| ADVERSE, equity cycle | −67.05% | **−41.33%** |

**And the mechanism is separable**, because the two ladders differ *only* at `b = 1`:

| months | what differs | σ at T=1 | σ at T=2 |
|---|---|---|---|
| `b=0` (111) | holding count only | 9.04% | 8.03% |
| `b=1` (74) | **the restored rung** | 12.35% | **6.12%** |
| `b=2` (35) | nothing — both 100% cash | −1.97% total | −1.57% total |

The `b=1` row is the rung doing exactly what a 50% cash allocation should: volatility halves
and the worst month goes from −27.2% to −13.3%. The `b=2` row is the control, and it confirms
the split. **The 3.8pp of CAGR given up comes from the `b=0` row** — holding two assets instead
of one is simply less concentrated in a bull market. Both effects are real; only the first is
RULE 5, and it would be overclaiming to credit the rung with the whole difference.

## Deviations

- **`DAA1_G12` was DELETED on 2026-07-28, and the reason is now a hard rule.** Its defensive
  sleeve was `SHV, IEF, UST`, and UST is a **2x leveraged** 7-10y treasury ETF: its risk-off
  state doubled duration risk while the engine still reported it as risk-off. Defence is held
  at 1x, always — `common/letf_mapper.assert_unlevered_defensive` now refuses to price any
  strategy that declares a leveraged product in its defensive sleeve, and the engine fails it
  rather than measuring it. (A leveraged *credit* product would be admissible in the
  OFFENSIVE universe, where Keller puts LQD; the ban is on levering the defence.) DAA1 was
  also the most expensive entry in the registry for coverage: UST's 2010-02 inception dragged
  the whole comparison window from 2008-07 to 2011-04.
- **`DAA_U6` and `DAA_U15` were DELETED on 2026-07-28.** Custom universes, and both ranked
  `BIL` inside the OFFENSIVE momentum universe — a T-bill fund competing against equities on
  13612 momentum is either a dead slot or a duplicate of the canary the strategy already
  has.
- **Sample.** The paper measures from the 1970s on index data. No result here is a
  reproduction.

## Cannot verify

- The pre-2000 published record.
- Whether `BND` (2007-04 inception) behaves as the paper's aggregate-bond canary proxy did
  over the earlier decades.
