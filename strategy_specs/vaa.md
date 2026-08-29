# VAA — Vigilant Asset Allocation

| | |
|---|---|
| **Source** | Keller, W.J. & Keuning, J.W. (2017), *Breadth Momentum and Vigilant Asset Allocation (VAA)*, SSRN **3002624** |
| **Local copy** | [`academic-papers/2017-vaa-ssrn-3002624.pdf`](../academic-papers/2017-vaa-ssrn-3002624.pdf) (37 pp.) |
| **Implementation** | [`strategies/vaa.py`](../strategies/vaa.py) |
| **Fidelity** | ✅ Faithful for both, on the paper's own tickers — see *Deviations* for the footnote that settles it. |

> Written from the PAPER, not from the code. Page numbers refer to the local PDF.

## Rules, as published

**Momentum — 13612W** (p. 4). The 12-4-2-1 weighted score:

```
score = 12·r1 + 4·r3 + 2·r6 + 1·r12
```

**Breadth momentum — the paper's central contribution** (pp. 1-5). VAA has no external canary.
The offensive universe *is* the breadth indicator: `b` counts how many of the N offensive
assets score non-positive, and the cash fraction follows from `b` alone.

**Cash fraction** (p. 5):

```
CF = (1/T) · rounddown( b·T / B ),   capped at 1
```

with `T` the number of offensive slots and `B` the breadth parameter. The paper's own worked
example on p. 5 — *T = 3, b = 2, B = 4 gives CF = 1/3* — is what pins the `rounddown`. Both
2026-07-28 audits initially suspected that quantisation was an implementation artefact; it is
not, it is the paper's rule, and `strategies/vaa.py` implements it exactly via
`n_cash_slots = floor(b·T/B)`, `CF = n_cash_slots / T`.

**Risk-off allocation** (p. 5). The cash fraction goes entirely to the **single best**
defensive asset by 13612W score, not spread across the defensive sleeve. When
`n_cash_slots ≥ T`, the whole book sits there.

**Rebalancing** (p. 5, note 7). Monthly, including rebalancing open positions back to their
prescribed `w = 1/T` weights — so turnover is not zero even in a month with no change of
holdings. The ledger models this correctly: positions drift between rebalances and are traded
back, which is why VAA shows the highest realised turnover in the registry (~16×/yr).

**Top-T is SIGN-AGNOSTIC** (§5). The paper declines absolute momentum on the selection and
says why: *"in the case where b<N, the (rounded) cash fraction CF is always higher with VAA
than in the traditional Dual case. So no bad assets will show up in the remaining Top
assets."* The breadth count has already been spent by the cash ladder; filtering the
survivors to positive momentum would count it twice. **`strategies/vaa.py` applied exactly
that `> 0` filter until 2026-07-30** — the same defect corrected in DAA on 2026-07-29,
unswept in its two siblings. Measured before the change: **zero months** of the real panel
where the filter altered the Top-T (the §5 argument holds structurally at `T=2, B=4`), so
the correction is a provable no-op on the registry, guarded by the golden master.

**`T = 1` goes all-in-cash the moment `b ≥ 1`** (§4): *"When B=1 or T=1 the whole portfolio
is fully invested in cash (when b>=1) or fully invested in the top T risky asset(s) (when
b=0)."* The floor formula only reproduces this at `B = 1` (which is `VAA_G4`'s registered
configuration); at `T=1, B>1` it would stay fully risky, and since 2026-07-30 the code
implements the paper's sentence. Note this is **VAA's own rule and differs from DAA n.8**,
which grades the same case to `CF = b/B` — each paper governs its own family.

**NaN policy.** A missing score counts toward `b`, i.e. toward risk-off. Never the reverse.
A missing score is also never *rankable*: the Top-T draws from `dropna()`, not from a sign
filter.

## Registered variants

| key | offensive universe | T | B | defensive |
|---|---|---:|---:|---|
| `VAA_G12` | SPY, IWM, QQQ, VGK, EWJ, **VWO, VNQ**, GSG, GLD, TLT, LQD, HYG | 2 | 4 | SHY, IEF, LQD |
| `VAA_G4` | SPY, VEA, VWO, BND | 1 | 1 | SHY, IEF, LQD |

**VAA has no leveraged variant, and that is a finding rather than an omission.** The two
`VAA_G3/G4_Leveraged_2X` wraps were DELETED on 2026-07-29 under RULE 4 in
[`../common/letf_mapper.py`](../common/letf_mapper.py) — *a wrap may change what is HELD,
never what decides to DE-RISK.*

VAA declares no canary on purpose (`sleeves()` returns `'canary': []`, and the docstring
explains why): its crash signal **is** the breadth count `b` over its own offensive universe,
divided by `B`. A leveraged wrap must restrict that universe to what executes at one uniform
multiple, and `VAALeveraged` accordingly set `self.B = n`. So the restriction did not narrow
the portfolio and leave the rule alone — it rebuilt the rule out of three US equity ETFs
correlating 0.79-0.91 with each other. Counting how many of twelve asset classes have rolled
over is a market-breadth signal; counting how many of SPY/QQQ/IWM have is asking one question
three times.

Measured over 2010-02…2026-06: `VAA_G3_Leveraged_2X` correlated **0.376** with `VAA_G12` —
the lowest wrap-to-parent figure in the registry, against 0.823 for `BAA_G4`. That is a
fidelity measurement, not a performance ranking, which is why it was allowed to decide. There
is no fix: enlarging the universe until the breadth count means something again requires
assets with no admissible LETF, and any smaller universe reproduces the defect.

## Deviations

- **`VAA_G4`'s tickers ARE the paper's, and this spec said otherwise until 2026-07-29.**
  The body text names `SPY, EFA, EEM, AGG`, but footnote 11 says what was actually run:
  *"We actually used (proxies for) Vanguard ETFs **VEA, VWO, VNQ, and BND** instead of the
  mentioned (and more common) iShares ETFs EFA, EEM, IYR, and AGG, respectively, in nearly
  all our backtest since these ETFs has lower fees and similar AUM's"*. The DAA paper
  restates VAA-G4 directly as `R4 = SPY, VEA, VWO, BND`. So `VAA_G4` is **faithful**; it
  carried `proxy` on a premise its own source contradicts.
- **`VAA_G12` now holds `VWO` and `VNQ`, and held `EEM` and `IYR` until 2026-07-29.**
  The same footnote 11 governs both variants, and it was applied to `VAA_G4` while `VAA_G12`
  was left on the iShares names — an inconsistency, not a decision. The earlier defence
  ("Keller treats the two fund families as interchangeable, so either set is his") does not
  survive re-reading: the footnote does not say the funds are equivalent, it says **which
  ones he ran**, and gives his reason. His reason is verifiable and holds today:

  | exposure | Keller's fund | body-text fund | expense ratio | AUM |
  |---|---|---|---|---|
  | emerging markets | **VWO** | EEM | **0.06%** vs 0.72% | both large |
  | US real estate | **VNQ** | IYR | **0.13%** vs 0.42% | $54.4B vs $3.1B |

  (Expense ratios and AUM as of 2026-07. Note the one thing the footnote gets wrong for
  today's market: it says "similar AUM's", and VNQ is now roughly 18x the size of IYR.)

  **The switch made the strategy look WORSE, which is the useful part.** Over 2015-01…2024-12
  (the golden-master window) CAGR fell 3.53% → 3.03%, MaxDD deepened −27.03% → −28.03%, and
  Sortino fell 0.39 → 0.31. A ticker correction that cost performance is a correction, not a
  fit. Coverage is unaffected: `VWO`'s pre-inception history is chain-linked from `EEM`
  (`HISTORY_BACKFILL`, both tracked MSCI EM until 2013), `VNQ` (2004-09) starts well before
  the binding `HYG` (2007-04), and the measured window stays 2008-06…2026-06.
- **`VAA_U6` and `VAA_U15` were DELETED on 2026-07-28.** The algorithm was the paper's; the
  asset lists were not, and both ranked `BIL` inside the offensive momentum universe. They
  were, measured, the two most DISTINCT entries in the whole registry (ρ_max 0.655 and 0.676)
  and that was not enough to keep them: distinctness manufactured by an unpublished universe
  is a degree of freedom, not evidence.
- **No canary is declared to `sleeves()`, deliberately.** VAA's de-risking rule is a breadth
  *count*, whereas `BaseStrategy.defensive_mask` implements "any canary down ⇒ risk-off".
  Passing the offensive universe in as a canary would mark dual-role LQD defensive in almost
  every month, since some asset in a 12-asset universe is nearly always negative. With no
  canary, dual-role LQD counts as offensive — the conservative direction, per `BaseStrategy`.

## Cannot verify

- The paper's pre-2000 sample.
- Whether the Vanguard and iShares funds would have produced the same monthly selections as each other, in either variant. Keller asserts they are interchangeable; this repo has not measured it.
