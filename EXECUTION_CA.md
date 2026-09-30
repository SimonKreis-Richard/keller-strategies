# HAA_G12_CA — HAA_G12 executed in Canadian-listed funds

`HAA_G12_CA` is **not a strategy**. It is an execution layer: the HAA_G12 decision is computed
exactly as the registry computes it, on the US tickers Keller published, and only then is each
chosen US ticker bought through a Canadian-listed fund. It changes **what is held**, never
**what is chosen**. It has no backtest of its own and is deliberately absent from the registry.

It exists for an investor who must hold products listed in Canada. **It checks listing and
domicile, not tax consequences**: every fund below is a Canadian-resident trust on a Canadian
exchange, and whether that satisfies any provision of the Income Tax Act is a question for a
tax advisor, not for this code ([`common/ca_mapping.py`](common/ca_mapping.py), RULE 3). The
funds' underlying holdings are US and foreign securities in every case.

**It never places an order.** It prints a plan to type into a broker, and journals it.

## Run it

```bash
cp ca_execution.example.json ca_execution.json   # then enter your own cash and holdings
venv/Scripts/python.exe -m tools.ca_orders
```

`ca_execution.json` is **gitignored**, like `user_config.json`: it holds real balances. The
template is fictional. Options: `--rate 1.3850` fixes USD/CAD by hand, `--no-journal` prints
without writing, `--config PATH` reads another file, `--demo` runs the offline demonstration
below.

What happens, and where it can refuse:

1. **The book** is read and validated. An unknown key, a US ticker, a negative balance or
   `leverage` above 1.0 stops the run before anything is downloaded.
2. **The signal** is HAA_G12's, from the engine's own price store, through the same guards
   as the live report ([`common/live_guards.py`](common/live_guards.py)): a price that is
   constructed rather than traded, a store whose adjustment history was never verified, or a
   dividend-adjustment seam on anything the strategy can hold is a refusal. A signal two or
   more months old is printed first, in capitals.
3. **Prices** are Yahoo's last closes for the Canadian funds (`.TO` on the TSX, `.NE` on Cboe
   Canada). A holding or a target with no quote, or a quote more than 7 days old, is a
   refusal. A price typed under `prices` in the config wins over the download.
4. **USD/CAD** is `--rate`, else the config's `usdcad`, else the Bank of Canada's daily rate,
   else Yahoo's `CAD=X`. **There is no default rate**: if all fail, the run stops
   ([`tools/fx_rate.py`](tools/fx_rate.py)).
5. **Orders** come from [`common/ca_execution.py`](common/ca_execution.py): sells first,
   crediting the currency of the class sold; then at most one conversion, in whichever
   direction is short; then buys in whole units against the cash actually there. The run
   asserts that the target weights reached the orders unaltered and that no US ticker did.
6. **The journal** goes to `logs/execution_ca/<signal>_<time>.json` and `.csv`. `logs/` is
   gitignored as a directory. Every CSV row names the signal date that authorised it.

**Leverage stays at 1x (EXEC-001).** `execution.leverage` above 1.0 is refused, because the
code cannot know whether a balance is equity or buying power. Below 1.0 is allowed: holding
cash back needs no information the code lacks.

**`prefer_usd_units`** spends USD already in the account on a fund's `.U` class, when that
USD covers the whole purchase. It never converts CAD to reach a `.U` class. The only sleeve
that forces a CAD→USD conversion is BIL, whose only image, UBIL.U, trades in USD — so every
move into HAA's T-bill sleeve from a CAD book pays one conversion.

## The mapping

Thirteen signal tickers, each mapped to one Canadian fund. TIP is the canary: it is read,
never traded, and deliberately has no image. Liquidity figures are the issuer's own ETF Facts,
and **their reporting dates differ by issuer** — hence the last column.

| Signal | CAD class | USD class | Exchange | MER | Avg daily volume | Avg spread | ETF Facts as of | Different asset? |
|---|---|---|---|---|---|---|---|---|
| SPY | ZSP | ZSP.U | TSX | 0.09% | 1,146,487 | 0.02% | 2025-11-30 | |
| QQQ | ZNQ | ZNQ.U | TSX | 0.39% | 128,126 | 0.03% | 2025-11-30 | |
| IWM | ZSML | ZSML.U | TSX | 0.22% | 31,578 | 0.22% | 2025-11-30 | **yes** |
| VGK | XEU | — | TSX | 0.28% | 55,006 | 0.11% | 2026-04-30 | **yes** (mild) |
| EWJ | ZJPN | — | TSX | 0.39% | 18,362 | 0.22% | 2025-11-30 | |
| VWO | XEC | XEC.U | TSX | 0.28% | 429,853 | 0.06% | 2026-04-30 | **yes** |
| VNQ | CGR | — | TSX | 0.72% | 26,664 | 0.16% | 2026-04-30 | **yes** (largest) |
| DBC | ZCOM | — | Cboe CA | 0.30% | *not published* | *not published* | 2025-08-31 | **yes** |
| GLD | CGL.C | — | TSX | 0.55% | 81,549 | 0.09% | 2025-08-31 | |
| TLT | ZTL | ZTL.U | Cboe CA | 0.22% | 38,721 | 0.22% | 2025-11-30 | |
| IEF | ZTM | ZTM.U | Cboe CA | 0.22% | 7,553 | 0.12% | 2025-11-30 | **yes** |
| LQD | ZIC | ZIC.U | TSX | 0.28% | 119,491 | 0.14% | 2025-11-30 | **yes** |
| BIL | — | UBIL.U | TSX | 0.13% | 127,592 (USD) | 0.02% | 2026-03-31 | |

**Seven of the thirteen buy a different asset from the one signalled:**

- **VNQ → CGR** — US REITs signalled, *global* real estate bought. The largest deviation.
- **DBC → ZCOM** — a different commodity index with a different roll rule. ZCOM launched in
  2025-10, publishes no volume or spread yet, and is an alternative mutual fund that may
  borrow and use leverage, which DBC does not. The weakest link in the table.
- **IEF → ZTM** and **LQD → ZIC** — shorter duration than the bond signalled. Paid on every
  rate move while held, and IEF is one of HAA's two defensive assets.
- **IWM → ZSML** — Russell 2000 signalled; S&P SmallCap 600, which screens for profitability,
  bought.
- **VWO → XEC** — FTSE Emerging excludes South Korea; MSCI EM IMI includes it.
- **VGK → XEU** — FTSE vs MSCI developed Europe; high overlap, the mildest of the seven.

**How far execution drifts from the signal has not yet been measured.** That is the
execution-gap report, not yet written. Until it exists, the table above is the whole account
of the difference.

Liquidity alarms the orders carry: the USD classes are thinner and wider than the CAD ones —
ZTL.U spreads 0.42%, ZIC.U 0.36%, XEC.U 0.34%, and ZSML.U traded on 161 of 251 days — and
ZCOM's missing figures are reported as *could not be checked*, never as clean.

Not used, and why, in `REJECTED_MAPPINGS`: XSU and XEH (CAD-hedged, which would remove the
USD exposure the backtest carries), CGL (hedged gold), CBIL (Canadian T-bills — a different
asset from BIL; it would avoid the conversion, at the cost of an eighth deviation, the only
one chosen rather than forced), and VFV (admissible, but with no USD class where ZSP has one).

## The demonstration

```bash
venv/Scripts/python.exe -m tools.ca_orders --demo
```

Two months on **fictional capital** — 100,000 CAD, nothing held — with the signals HAA_G12
actually gave, Yahoo's actual closes and the Bank of Canada's actual rates on those dates, all
recorded in [`examples/ca_demo.json`](examples/ca_demo.json). It runs offline, on a fresh
clone, and reproduces [`examples/ca_rotation_demo.csv`](examples/ca_rotation_demo.csv) exactly
(a test requires it).

- **2026-07-31**: canary alive. Six sleeves at 1/6 each — ZCOM, ZJPN, ZSML, ZSP, XEU, CGR —
  all bought in CAD, no conversion; 288 CAD of whole-unit rounding stays as cash.
- **2026-08-31**: TIP's momentum turned negative, the canary died, and HAA_G12 went 100% to
  BIL. All six are sold, the CAD proceeds are converted once, and UBIL.U is bought.

Not advice and not a track record: two months, chosen because they show a full rotation.

Building it found three defects in `common/ca_execution.py`, each now fixed and tested:

- a fund leaving the target was sold to one unit short: 461 × 38.49 / 38.49 floors to 460;
- `prefer_usd_units`, the default, converted a CAD-only book to USD to buy `.U` classes;
- whole-unit rounding was reported per sleeve as *not fully invested*, which buried any real
  shortfall.

## Files

| File | Role | Network |
|---|---|---|
| [`common/ca_mapping.py`](common/ca_mapping.py) | the table, its five rules, liquidity flags | no |
| [`common/ca_execution.py`](common/ca_execution.py) | weights → two-currency orders; the FX rate is an argument | no |
| [`common/live_guards.py`](common/live_guards.py) | the refusals shared with the live report | no |
| [`tools/fx_rate.py`](tools/fx_rate.py) | USD/CAD: manual, Bank of Canada, Yahoo — or an error | yes |
| [`tools/ca_orders.py`](tools/ca_orders.py) | the driver: book → signal → quotes → FX → orders → journal | yes |
| `ca_execution.example.json` | fictional template; copy to the gitignored `ca_execution.json` | — |
| `examples/` | the offline demonstration | — |

The tests (`tests/test_ca_mapping.py`, `test_ca_execution.py`, `test_ca_orders.py`,
`test_fx_rate.py`, `test_live_guards.py`) are offline. Every fetcher is injected, and the
refusal paths are tested with fetchers that raise if called.
