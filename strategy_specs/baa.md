# BAA — Bold Asset Allocation

| | |
|---|---|
| **Source** | Keller, W.J. (2022), *Relative and Absolute Momentum in Times of Rising/Low Yields: Bold Asset Allocation (BAA)*, SSRN **4166845** |
| **Local copy** | [`academic-papers/2022-baa-ssrn-4166845.pdf`](../academic-papers/2022-baa-ssrn-4166845.pdf) (14 pp.) |
| **Implementation** | [`strategies/baa.py`](../strategies/baa.py) |
| **Fidelity** | ✅ Faithful for G12, G4 and G1_SPY. |

> Written from the PAPER, not from the code. Page numbers refer to the local PDF.

## Rules, as published

**Two different momentum measures, used for two different jobs** (pp. 3-7). This is BAA's
defining feature and the most common source of misimplementation:

| purpose | measure |
|---|---|
| canary (crash signal) | **13612W**, the fast 12-4-2-1 weighted score |
| ranking offensive and defensive assets | **SMA12 ratio**, `price / SMA(12) − 1`, where SMA(12) is the average of the **last 13 prices including the present** (n.5) |

Using one measure for both jobs is a different strategy.

**SMA(12) counts LAGS, not prices** (n.5). Verbatim: *"The SMA(12) momentum (with lag 12
months) equals the present price pt divided by the average of the **last 13 asset prices
including the present** (also noted as SMA13), minus 1."* Thirteen prices, not twelve. This
repository averaged twelve until 2026-07-29, which dropped the oldest month out of the filter
entirely and shifted every offensive rank, every defensive rank, and the absolute test below.
Faber's SMA(10) genuinely does count ten prices — the two authors use different conventions,
and `tests/test_paper_rules.py` pins both so neither can be "made consistent" with the other.

**Canary — four assets, binary trigger** (pp. 1-4, 7). `SPY, VWO, VEA, BND`. If **any** of the
four has a 13612W score ≤ 0 (or no data), the portfolio is fully defensive. Unlike DAA there is
no graded cash fraction: BAA's protection is a switch, not a dial.

**Offensive mode.** Top `TO` offensive assets by SMA12 ratio, equal-weighted. `TO = 6` for G12,
`TO = 1` for G4.

**Defensive mode — top TD, then an absolute filter against BIL** (pp. 5, 7, 9-10). Take the top
`TD = 3` defensive candidates by SMA12 ratio; for each slot, hold that asset only if its SMA12
ratio exceeds **BIL's**, otherwise hold BIL. Each slot is filtered independently, so a
defensive month can end up part BIL and part bonds. Note the comparison is against BIL's own
SMA12 ratio — not against zero, and not against a nominal yield.

## Registered variants

| key | offensive universe | TO | paper location |
|---|---|---:|---|
| `BAA_G12` | SPY, QQQ, IWM, VGK, EWJ, VWO, VNQ, DBC, GLD, TLT, HYG, LQD | 6 | Fig 3 |
| `BAA_G4` | QQQ, VWO, VEA, BND | 1 | Fig 6 |
| `BAA_G1_SPY` | SPY | 1 | **§5 and Fig 11 — Keller's own BAA-SPY** |

Canary `{SPY, VWO, VEA, BND}`, defensive `{TIP, DBC, BIL, IEF, TLT, LQD, BND}`, `TD = 3`
throughout. TLT, DBC and LQD are **dual-role** — momentum assets in the offensive universe and
candidates in the defensive basket — and `sleeves()` declares them in both so the canary can
resolve which role they play each month.

## Deviations

- **`BAA_G4_T2` was DELETED on 2026-07-28.** It changed only `TO` (1 → 2) on an unchanged
  universe, ρ = 0.894 against `BAA_G4`. A parameter twiddle with no mechanism behind it is a
  multiple-testing dimension; `BAA_G12_T3` had already been cut on the same criterion, and
  leaving `G4_T2` was an inconsistency.
- **`BAA_G1_SPY` is retained as a CONTROL, not a strategy** — the degenerate single-asset case
  that shows how much of BAA's record is the canary machinery and how much is the universe.
  Marked `(ctrl)` in the report and excluded from the selection statistics.
- **`BAA_G1_SPY` carried `fidelity='custom'` until 2026-07-30**, on a docstring claim that
  "Keller names no single-asset BAA variant". BAA §5 publishes exactly this configuration and
  Fig 11 captions it **BAA-SPY**, with every parameter matching what the class inherits from
  `BAA_G12` (`SelO=SPY, NO=TO=1`, the seven-asset SelD, the G4 canary, TD=3). Fourth label
  found wrong, fourth in the under-claiming direction; the pinned table in
  `tests/test_paper_rules.py::TestFidelityAgainstSource` now guards all 35.
- **`BAA_G12_T3` was deleted** on 2026-07-28: it measured ρ = 0.951 against `BAA_G12` and
  carried no information the parent did not already show.

## Cannot verify

- The paper's pre-2000 sample.
- Whether the exact defensive basket used here matches the paper's in every variant — the
  seven-asset list is stable across the code and consistent with the text, but the paper
  presents it inline rather than as an enumerated table.
