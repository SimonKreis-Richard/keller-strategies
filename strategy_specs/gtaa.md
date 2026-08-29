# GTAA — Global Tactical Asset Allocation

| | |
|---|---|
| **Source** | Faber, M.T. (2006), *A Quantitative Approach to Tactical Asset Allocation*, SSRN **962461** |
| **Local copy** | [`academic-papers/2006-tta-ssrn-962461.pdf`](../academic-papers/2006-tta-ssrn-962461.pdf) (70 pp.) |
| **Implementation** | [`strategies/gtaa.py`](../strategies/gtaa.py) |
| **Fidelity** | ✅ Faithful. `GTAA_G5` is the only registered variant; the two `G13` variants were deleted 2026-07-28. |

> Written from the PAPER, not from the code. Page numbers refer to the local PDF.

## Rules, as published

**The timing model** (pp. 19-21). The paper's rule is deliberately minimal:

```
Buy   when  monthly price > 10-month simple moving average
Sell  when  monthly price < 10-month simple moving average
```

Evaluated **per asset, independently**, on month-end closes only. Each asset that fails its own
filter moves to cash; there is no ranking, no canary, and no portfolio-level switch. This is
absolute momentum applied asset by asset — the whole of it.

**The five-asset portfolio** (pp. 19-21). Five equal 20% sleeves: US equity, foreign equity,
bonds, commodities, real estate. Each sleeve is either invested or in cash according to its own
10-month filter, so the book is 0%, 20%, 40%, 60%, 80% or 100% invested.

**Monthly rebalancing.** Signals are evaluated once per month on the close.

## Registered variants

| key | universe | rule | fidelity |
|---|---|---|---|
| `GTAA_G5` | SPY, EFA, IEF, DBC, VNQ | 10-month SMA per asset, 1/5 each, failures to BIL | **proxy** |

Cash proxy `BIL`.

## Deviations, and the `cannot verify` finding

**`GTAA_G5` maps cleanly** onto the published five-asset timing model. `SPY / EFA / IEF / DBC /
VNQ` are the standard ETF proxies for the paper's five asset classes.

**It is `proxy`, not `faithful` — relabelled 2026-07-29.** Faber (2006) times *indices*, not
funds: S&P 500, MSCI EAFE, GSCI, NAREIT and the 10-year US Treasury. `DBC` tracks the DBIQ
index, which is not the GSCI, and `IEF` is a 7-10y fund, not the 10-year note. Several of
these had no tradable ETF when the paper was written, so there are no published tickers to be
faithful to. That is exactly the defined case for `proxy`: the paper's algorithm and universe,
different funds. The `10-month SMA` filter itself is Faber's exactly, and counts **ten
prices** — deliberately not Keller's SMA(12), which counts thirteen.

**The two `G13` variants were DELETED on 2026-07-28.** Neither could be tied to any specific
published Faber variant by either audit: his later work describes wider universes and top-N
overlays, but neither the exact 13-asset list nor the top-6 rule was ever matched to a
citation. Two unverifiable variants on the *same* universe (ρ = 0.897 between them) is one
degree of freedom, not two.

The Aggressive one was the weaker of the pair: it ranked with **13612U** — a Keller
construction — and filtered with Faber's **10-month SMA**, a hybrid neither author published.

`GTAA_G5` is retained and is now the only Faber entry in the registry.

## Cannot verify

- The paper's pre-2000 sample (it runs back to 1973 on index data).
- Whether the ETF proxies used for `GTAA_G5` would have reproduced the paper's index-level
  signals over the overlapping years.
