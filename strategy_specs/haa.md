# HAA — Hybrid Asset Allocation

| | |
|---|---|
| **Source** | Keller, W.J. (2023), *Hybrid Asset Allocation (HAA)*, SSRN **4346906** |
| **Local copy** | [`academic-papers/2023-haa-ssrn-4346906.pdf`](../academic-papers/2023-haa-ssrn-4346906.pdf) (15 pp.) |
| **Implementation** | [`strategies/haa.py`](../strategies/haa.py) |
| **Fidelity** | ✅ Faithful. Verified line-by-line against the PDF by two independent audits, 2026-07-28. |

> **Written from the PAPER, not from the code.** A spec derived from the implementation is
> circular: it agrees with the code even when the code is wrong, which is how the pre-2026-07
> specs in this folder passed every review while the engine around them was broken. Page
> numbers refer to the local PDF. Silence under *Deviations* means "matches the paper".

## Rules, as published

**Momentum — 13612U** (p. 3). Unweighted average of the 1-, 3-, 6- and 12-month total returns:

```
score = ( r1 + r3 + r6 + r12 ) / 4
```

This is the *unweighted* variant. DAA/VAA/BAA use the 12-4-2-1 **weighted** form; feeding one
where the other is expected silently changes every ranking. `strategies/haa.py` declares
`score_type='unweighted'` and `main.py` routes `calc_13612u` accordingly.

**Canary — one external asset, TIP** (pp. 2, 5-6). HAA's defining departure from VAA and DAA
is that crash protection is driven by a *single asset outside the offensive universe* rather
than by a breadth count across it. TIP's 13612U score ≤ 0 ⇒ risk-off.

**Universe and TopX** (pp. 2, 6-8). `NO` offensive assets, of which the top `TO = NO/2` are held
equal-weight. Published variants:

| variant | NO | TO | ND | TD | NP |
|---|---:|---:|---:|---:|---:|
| HAA-8 ("Balanced") | 8 | 4 | 2 | 1 | 1 |
| HAA-12 | 12 | 6 | 2 | 1 | 1 |

**Defensive sleeve — BIL or IEF** (pp. 2, 6). `ND = 2` candidates, `TD = 1` chosen: whichever has
the higher 13612U score.

**Two DISTINCT risk-off mechanisms.** Conflating them is the most common misreading:

1. *Canary risk-off* — TIP ≤ 0 ⇒ **100%** into the single best defensive asset.
2. *Per-slot absolute filter* — with the canary alive, a selected offensive asset scoring ≤ 0
   has **its slot** (not the whole book) replaced by the best defensive asset. Several slots
   can be replaced independently in the same month.

**Transaction cost** (p. 7). The paper assumes a **one-way** cost of 0.1%, so a full A→B
rotation costs 0.2% of notional. `COST_PCT_PER_SIDE = 0.001` matches this. The pre-audit engine
charged `Σ|Δw|/2 × 0.1%` — half of one-way.

**Published results** (p. 7), Dec-1970 → Dec-2022, TC = 0.1%:

| variant | CAGR | MaxDD | Vol | Sharpe | UPI |
|---|---:|---:|---:|---:|---:|
| HAA-8 | 15.9% | **−9.7%** | 9.4% | **1.21** | **4.88** |
| HAA-12 | 15.9% | −10.7% | 9.6% | 1.19 | 4.50 |

Read that before preferring G12 on performance grounds: in Keller's own full sample **HAA-8
beats HAA-12 on both max drawdown and Sharpe at identical CAGR.**

## Registered variants

| key | offensive universe | TO | type |
|---|---|---:|---|
| `HAA_G8_Balanced` | SPY, IWM, VEA, VWO, VNQ, DBC, IEF, TLT | 4 | faithful |
| `HAA_G12` | + QQQ, VGK, EWJ, GLD, LQD | 6 | faithful |
| `HAA_G1_Simple` | SPY | 1 | faithful, **control** |
| `HAA_G3_Leveraged_2X` | SPY, QQQ, IWM → SSO, QLD, UWM | 1 | custom |
| `HAA_G4_Leveraged_2X` | + GLD → UGL | 2 | custom |

All use canary `TIP` and defensive `{BIL, IEF}`. IEF is dual-role (offensive universe *and*
defensive candidate) in the larger variants; `sleeves()` declares it in both and the canary
resolves which role it plays each month.

**The leveraged sizes were ADDED on 2026-07-29**, the same day VAA's and PAA's were deleted,
and for the same reason: RULE 4 in [`../common/letf_mapper.py`](../common/letf_mapper.py) —
*a wrap may change what is HELD, never what decides to DE-RISK.* HAA passes it most cleanly of
any family. Its canary is `TIP`, which is not in the offensive universe and so cannot be
touched by restricting it; its absolute-momentum filter is a per-asset test with no
denominator that moves when the universe shrinks; and `TO = NO/2` is Keller's own published
ratio, so even the selection count is not a liberty here. The universes are still invented —
`LeveragedWrapMixin` hard-codes `fidelity = 'custom'`.

Read [`../LEVERAGE.md`](../LEVERAGE.md) §8 before the numbers. `HAA_G4_Leveraged_2X` is the
first leveraged entry in this repository to beat SPY on Sortino and UPI, and it lost −19.2% in
`bear_covid` where `BAA_G4_Leveraged_2X` gained +5.9%. A single exogenous canary survives the
restriction intact and is still the narrowest sensor in the canon.

## Deviations

- **`HAA_G4` and `HAA_G16` were DELETED on 2026-07-28.** Both were custom sizes — the paper
  presents HAA-8 and HAA-12 — and G16 additionally measured ρ = 0.935 against G12 while
  starting a full year later (SCZ, 2007-12), so it paid coverage and returned nothing.
- **`HAA_G1_Simple` is retained as a CONTROL, not a strategy.** The single-asset degenerate
  case isolates how much of the family's record is the timing rule and how much is the
  twelve-asset universe. It is reported in the table, marked `(ctrl)`, and excluded from the
  selection statistics. Worth remembering that it has outranked `HAA_G12` on Sharpe and
  Sortino more than once.
- **Sample.** The paper measures Dec-1970 → Dec-2022 on index data; this engine has no
  pre-2000 history and uses ETFs. **No result here is a reproduction of the published one.**
  `tests/test_anchors.py::TestPublishedAnchor` asserts only that the shape is not impossible,
  and says so in its docstring.

## Cannot verify

- The 1970-2000 portion of the published record — no data.
- Whether Keller's index proxies and this repo's ETFs would have ranked identically over the
  overlapping years.
