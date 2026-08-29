# PAA — Protective Asset Allocation

| | |
|---|---|
| **Source** | Keller, W.J. & Butler, A. (2016), *Protective Asset Allocation (PAA): A Simple Momentum-Based Alternative for Term Deposits*, SSRN **2759734** |
| **Local copy** | [`academic-papers/2016-paa-ssrn-2759734.pdf`](../academic-papers/2016-paa-ssrn-2759734.pdf) (24 pp.) |
| **Implementation** | [`strategies/paa.py`](../strategies/paa.py) |
| **Fidelity** | ✅ Faithful. |

> Written from the PAPER, not from the code. Page numbers refer to the local PDF.

## Rules, as published

**Momentum — SMA12 ratio** (pp. 3-4). `price / SMA(12) − 1`, where the moving average runs over
the **last 13 prices including the present**. An asset is "good" when this is positive.

The paper states the same definition from the weighting side (§1): the filter is *"based on a
linearly decreasing weight filter over the previous 12 monthly returns (so the most recent
month return is weighted 12 times ... up to the oldest (12th) month one time)"*. Averaging
thirteen prices produces exactly those weights, `(12, 11, ..., 1)`. Averaging twelve produces
`(11, 10, ..., 1, 0)`: the oldest month falls out and the filter spans eleven returns, not
twelve. This repository averaged twelve until 2026-07-29, which moved PAA's breadth count `n`
and therefore its bond fraction.

**Bond fraction, via the protection factor** (p. 6, eq. 2). With `N` assets in the universe and
`n` of them good:

```
BF = (N − n) / (N − n1),      n1 = a · N / 4,      BF = 100% if n ≤ n1
```

`a ≥ 0` is the **protection factor**. For `N = 12` this gives:

| variant | a | n1 | denominator | cash fraction |
|---|---:|---:|---:|---|
| PAA0 | 0 | 0 | 12 | `(12 − n) / 12` |
| PAA1 | 1 | 3 | 9 | `(12 − n) / 9`, capped at 1 |
| PAA2 | 2 | 6 | 6 | `(12 − n) / 6`, capped at 1 |

The `/12`, `/9`, `/6` denominators in the code are **not** magic numbers — they are `N − a·N/4`
evaluated at `a = 0, 1, 2`. Both 2026-07-28 audits initially flagged them as suspicious; the
paper's equation resolves it.

**Risky selection** (p. 6, step 2). Top `Top ≤ N` **good** assets by momentum, equal-weighted
across `1 − BF`. If fewer than `Top` assets are good, only the good ones are held. `Top = 6`
here. Verbatim: *"If n<Top, only the n good assets (with positive momentum) will be included
in this risky EW portfolio."*

> **Source conflict, recorded 2026-07-30.** Keller's later BAA paper (§2) restates PAA as
> *"like for PAA, we don't use absolute momentum for the Top6 selection of the Offensive
> universe, only relative momentum"* — which contradicts the 2016 recipe's step 2 above. A
> 2026-07-30 external audit, working from the BAA restatement without the PAA original,
> asked for the positive-momentum filter's removal; reading SSRN 2759734 settles it the
> other way, and the filter stays. The conflict is inert in practice: PAA2 holds risk only
> when `n ≥ 7`, so the top six are all positive and both readings pick identical portfolios
> in every measured month.

**Defensive sleeve.** The bond fraction goes to a single bond asset — `IEF` in this
implementation.

**No canary.** Like VAA, PAA's de-risking is a breadth count over the offensive universe, not a
separate signal.

## Registered variants

| key | a | universe |
|---|---:|---|
| `PAA2_G12` | 2 | SPY, QQQ, IWM, VGK, EWJ, EEM, IYR, GSG, GLD, TLT, LQD, HYG |

**`PAA2` is the paper's own model name, not a version number.** §3: "We will distinguish between low (PAA0), medium (PAA1) and high protection strategies (PAA2), depending on the chosen protection rate (a=0,1,2, resp.)", and the paper designates this one as its conclusion: "It is this PAA2 model which we consider our alternative for a 1-year term deposit." The registry key was `PAA_G12_V2` until 2026-07-29, which read like a second version of the code and hid whose label it was.

Only the `a = 2` (most protective) variant is registered. `PAA0` and `PAA1` measured ρ =
0.974-0.982 against it over 2012-2024 and added no measurable information; the `PAA` base class
still accepts `variant='PAA0'|'PAA1'|'PAA2'` for ad-hoc study, so all three formulas remain
available and testable.

**PAA has no leveraged variant, and that is a finding rather than an omission.** The two
`PAA_G3/G4_Leveraged_2X` wraps were DELETED on 2026-07-29 under RULE 4 in
[`../common/letf_mapper.py`](../common/letf_mapper.py) — *a wrap may change what is HELD,
never what decides to DE-RISK.* PAA declares no canary on purpose: its protection is
`CF = (N - n_pos) / (0.5 · N)` with `N` the size of the offensive universe. A wrap must
restrict that universe to what executes at one uniform LETF multiple, which rewrites `N` and
therefore rewrites the protection factor the paper calibrates. At `N = 3` the formula
degenerates to three rungs — `0% / 67% / 100%` — and trips fully defensive as soon as two of
three US equity ETFs at ρ ≈ 0.9 are negative.

Measured: `PAA_G3_Leveraged_2X` correlated 0.543 with `PAA2_G12` (G4: 0.677), against
0.784-0.823 for the exogenous-canary families. That is a fidelity measurement, not a
performance ranking — the same rule kept `BAA_G3_Leveraged_2X`, which ranked *below* both
deleted PAA wraps on Sortino.

Worth keeping in view: PAA's strongest measured property needs no wrap to show. It traded
**6.1-6.7× notional per year against 11-13× for every other family** — less than half the
rotation, and therefore less than half the cost and the daily-reset drag.

## Deviations

- **Defensive sleeve fixed to IEF.** The paper discusses SHY and IEF as bond candidates; this
  implementation always uses IEF. That is a simplification, not a rule from the paper.
- **Only `a = 2` is registered** (see above).
- **`Top = 6` is fixed**, where the paper treats `Top` as a free parameter.

## Cannot verify

- The paper's pre-2000 sample.
- Whether the specific 12-asset universe here matches the paper's exactly; the composition is
  consistent with the text but the paper does not present it as a closed enumerated list.
