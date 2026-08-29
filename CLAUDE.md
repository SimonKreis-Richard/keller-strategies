# AGENTS.md — Keller Strategies

Guidance for AI agents working in this repository.

## What this is

A quantitative **Tactical Asset Allocation (TAA) backtesting engine** implementing
Dr. Wouter Keller's canon (HAA, DAA, VAA, BAA, PAA) plus a custom four-module dual-momentum
sleeve, Faber's GTAA, and leveraged-ETF variants. 36 registered variants, each labelled
`faithful` / `proxy` / `custom` / `control` / `benchmark` in the report.

> The engine was rebuilt on 2026-07-28 around an explicit execution ledger, after an
> adversarial audit found that execution, cost, cash, the rebalance date and the meaning of a
> missing price were all implied by a single vectorised expression. Two further audits on
> 2026-07-29 found two Keller-compliance defects that had survived it — see
> `tests/test_paper_rules.py`, which exists so that class of defect cannot survive again.
> **[`KNOWN_GAPS.md`](KNOWN_GAPS.md) is the standing list of what the engine still cannot
> establish, and now the only record of what the audits found — read it before quoting any
> number.**

- **Language:** Python 3.11+ (`requirements.txt`: pandas, numpy, yfinance, matplotlib, nicegui;
  `requirements.lock` pins the exact environment that produced the current numbers)
- **Data:** Yahoo Finance via `yfinance`, **daily** bars cached to `data/cache/daily_*.csv`
- **Vibecoded origin:** prefer validating behavior by running code over assuming from names.

## How to run

```bash
pip install -r requirements.txt
python main.py                 # backtest or live mode, per user_config.json / dashboard defaults
python main.py --live          # force live signal mode
python main.py --list          # list all available strategy keys
python main.py --strategy HAA_G12 DAA_G12   # run specific strategies
python app.py                  # NiceGUI desktop dashboard (Backtest + Live tabs)
```

All user-facing knobs are **documented with defaults** in the USER DASHBOARD block at
the top of `main.py` and **overridden per key by `user_config.json`** (gitignored —
holds the user's personal values: broker balances, strategy picks, dates, leverage;
template in `user_config.example.json`). Never put personal values back into `main.py`.
Precedence for strategy selection: CLI `--strategy` > `user_config.json` `STRATEGIES`
> the `STRATEGIES_TO_RUN` catalog.

## Architecture (3 layers)

```
DATA      common/data_engine.py     PriceStore — DAILY bars, real trading dates, complete
                                    months only, provenance hash (Yahoo is the only source)
SIGNAL    strategies/*.py           momentum/canary logic → weights on ORIGINAL ETFs
EXECUTION common/ledger.py          fill date, fill price, drift, per-leg cost, cash
                                    account, margin and day-counted interest
GUARDS    common/coverage.py        from when a strategy can honestly be measured
REPORTING main.py + common/         metrics (incl. UPI), regimes, selection stats, manifests
SIZING    common/margin_sizing.py   sustainable MARGIN leverage from a model's KPIs. Stands
                                    BESIDE the engine, not inside it: run_ledger never tests a
                                    maintenance requirement, and this does not change that
          common/leverage_advice.py the driver for it — broker assumptions, trial count, one
                                    recommendation per metrics row. Feeds the report's
                                    SUSTAINABLE MARGIN LEVERAGE section and the GUI column
DOUBT     common/robustness.py      how much of the leaderboard survives a different sample:
                                    PBO by CSCV (deterministic, no seed), the ranking rebuilt
                                    on joint stationary-bootstrap resamples, both POOLED by
                                    family and by de-risking mechanism, and the LEVERAGE
                                    FRONTIER — the same resampled histories walked at
                                    constant margin, cross-checking CAP 1 from an
                                    independent direction. Answers "how much of rank 1 is
                                    search", NOT "what happens next"
UI        app.py (NiceGUI)          thin GUI over main.load_data / run_backtest /
                                    compute_live_signals — no strategy logic
```

`load_data(config)` returns **four** values: `(prices, scores_w, scores_u, store)`. The store
is part of the contract because execution is priced at the session *after* the decision, which
a monthly panel cannot express. `run_backtest(..., store=store)` raises without it.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full model.

Compute/presentation split: `size_positions` and `compute_live_signals` return plain
data structures (orders, warnings, canary states); `run_live_mode` is the CLI printer
and the GUI's Live tab is the other consumer. Keep new engine features print-free.

Live sizing prices: the monthly signal decides WHAT to hold, but live-mode share
quantities are sized at the latest market quote (`get_live_prices`, raw Close), with
loud fallback to the month-end close when offline. Never size live orders on the
monthly cache alone — its adjusted closes can be days old and dividend-adjusted,
which oversizes orders and gets them rejected by the broker.

Key invariant: **the signal is always computed on original (1x) ETFs.** Leverage is
applied only at execution — never bake LETF tickers into a signal universe.

## How leverage works (important — two independent mechanisms)

1. **Leveraged ETFs (2x/3x):** realized through the *real* price history of products
   like UPRO/TQQQ/TMF (mapping in `common/letf_mapper.py`). Their internal financing
   cost and volatility decay are already in those prices — do not re-model them.
2. **Margin (`LEVERAGE_FACTOR`):** borrowed money (brokerage margin / credit line)
   applied to active strategies' returns. `MARGIN_BORROW_RATE` charges interest on the
   borrowed portion. The two compose: a 2x LETF strategy at `LEVERAGE_FACTOR=1.3` ≈ 2.6x
   effective exposure.

**`MARGIN_FOLLOWS_SIGNAL` (default True) — margin de-levers with the signal.** LETFs
de-lever for free: rotating from UPRO into IEF drops exposure to 1x. Flat margin does not
— the loan stays drawn, so the defensive sleeve is bought with borrowed money and the
portfolio rides the drawdown levered. With the flag on, borrowing is scaled by the
offensive weight:

```
effective_leverage_t = 1 + (LEVERAGE_FACTOR - 1) x offensive_weight_t
interest             = debt_actually_drawn x rate x days / 365   (settled each rebalance)
```

so the offensive contribution is levered, the defensive contribution is not, and interest
accrues only on what was actually drawn — day-counted, not a flat monthly twelfth.

`BaseStrategy.offensive_weight()` / `defensive_mask()` resolve the sleeve split from each
strategy's **`sleeves()` declaration**. They no longer infer anything: the attribute-sniffing
resolver they replaced looked for five different names for a strategy's cash bucket and
silently returned an empty set for a strategy that used a sixth. Dual-role assets (TLT/DBC/LQD
in BAA, IEF in HAA) are resolved by the canary and default to **offensive** when there is no
canary — never claim de-escalation the signal cannot prove.

One subtlety worth knowing: a *deliberate* cash-ticker holding follows the mode's rule (flat
margin stays drawn through it), while only the **uninvested residue** is always unlevered.
Forcing both to 1x made "flat" not flat and blurred the very A/B this flag exists to make. Set
the flag to False to reproduce flat-margin results.

Design rules baked into the code:
- **Offence is leveraged, defence is held at 1x.** `letf_mapper.translate()` maps the
  offensive sleeve to LETFs and passes unmapped (defensive) assets — IEF/SHY/BIL/BND/
  LQD/HYG/TIP — through at 1x on their own ticker, so defensive months earn the real
  bond/cash return rather than 0%.
- **Passive benchmarks (SPY_Benchmark, Golden_Butterfly) run at 1x** — they are a
  clean reference and are excluded from margin leverage.
- **The offensive sleeve must execute at ONE uniform multiple.** A universe that lands
  partly on 3x products and partly on 2x products (or on unleveraged originals) has an
  effective leverage that drifts with the monthly signal draw, so the backtest stops
  describing the portfolio held. `LETFMapper.validate_universe()` enforces this and is
  called from every leveraged strategy's constructor — an inadmissible universe raises
  at construction rather than failing silently at execution. The defensive sleeve and
  the canaries are deliberately exempt: defence is held at 1x by design.

## Leveraged variants — all use the wrap pattern

Every leveraged variant subclasses its canonical strategy, runs that exact signal, then
calls `LETFMapper.translate(...)`. No separate simplified signal logic exists.

- **Sized variants** in `strategies/*_leveraged.py` apply the canonical algorithm to a
  restricted universe that is fully executable at one ratio. These sizes were never
  defined by Keller — they are custom universes, but the signal math is the faithful one.

  | Universe | Assets | Admissible ratios | Registered |
  |---|---|---|---|
  | G2 | SPY, QQQ | 2x, 3x | 2x as `strategy` |
  | G3 | SPY, QQQ, IWM | 2x, 3x | 2x as `strategy`, 3x as `exploratory` |
  | G4 (2x) | SPY, QQQ, IWM, GLD | 2x only — no 3x gold product exists | 2x as `strategy` |
  | G4 (3x) / G5 | + EEM and/or TLT (both 3x-only tickers) | 3x only | 3x as `exploratory` |

  NAMING HAZARD: `G{n}` counts offensive assets, and at n=4 the admissible universe DIFFERS
  BY RATIO (`*_G4_Leveraged_2X` ends in GLD; `BAA_G4_Leveraged_3X` ends in EEM). Read the
  factory, not the key — the full ladder is documented in `strategies/haa_leveraged.py`.

  **2x entries are `role='strategy'`; 3x entries are `role='exploratory'`** — registered,
  measured and reported in full, excluded from every selection statistic and barred from
  setting the ranked window (`strategies/base.py`). The registry WAS all-2x from 2026-07-28
  to 2026-07-29; the 3x entries were then admitted under this role because ρ ≈ 0.996-0.999
  against the 2X siblings describes the SHAPE and says nothing about the DEPTH — 4 of 4
  measured pairs deepen drawdown 13-22 pp — and no 3x product predates 2008-11, so none has
  bear-market history. Do not promote a 3x entry to `role='strategy'` without new evidence.

- `DM_G3_Leveraged_2X` is a single-module dual-momentum sleeve (relative + absolute vs the
  BIL T-bill return) on `{SPY, QQQ, IWM}` — a universe chosen by what has 2x products, hence
  `custom`. It is a different object from `DM_G8_Composite`, which is Antonacci's published
  four-module portfolio.
- **Full-universe wraps were removed** (`DAA_Leveraged`, `VAA_Leveraged`, `BAA_Leveraged`,
  `PAA_Leveraged`, the four-module DM wrap) along with the G6/G8 sizes. Keller's G12 universes need
  VGK/EWJ/VWO/VNQ/GSG/HYG/LQD, none of which has an admissible leveraged product, so part
  of their offensive sleeve silently executed at 1x. Do not reintroduce them.

## Conventions

- The `strategy_specs/` files and `academic-papers/` (Keller's SSRN papers) are the
  source of truth for strategy logic. Check an implementation against its spec.
- **Execution is priced at the session AFTER the decision** (`EXECUTION_CONVENTION='next_open'`,
  `common/ledger.py`). A signal computed from the month-end close cannot be filled at that same
  close, which is what the pre-audit engine did. Measured worth of the difference: ≈ +0.37 pp of
  CAGR on HAA-G8, up to +0.87 pp levered. (An earlier +2.09 pp figure compared a full holding
  period against a truncated one and was ~5.6x too large; see `KNOWN_GAPS.md` §7.)
  `signal_close` still
  exists as a convention, purely to reproduce pre-audit numbers; never quote it as a result.
- **Every new strategy MUST implement `sleeves()`** returning
  `{'offensive': set, 'defensive': set, 'canary': list}`. `BaseStrategy.sleeves()` raises;
  there is deliberately no inferred default. A leveraged wrap must declare the **LETF images**
  it actually holds, not the 1x signal assets — mix in `LeveragedWrapMixin` and it is handled.
  This replaced a resolver that sniffed five attribute names and silently returned an empty
  set for a strategy that used a sixth.
- **Costs are one-way, per leg.** `COST_PCT_PER_SIDE = 0.001` means a full A→B rotation costs
  0.2% of notional. The old `TRANSACTION_COST_PCT` key still loads, with the same value now
  meaning twice what it used to.
- **The daily cache re-checks Yahoo at most every `CACHE_REFRESH_HOURS` (default 6).** It
  used to re-download a 90-day window of every ticker on EVERY run — 9.1s and a ~10MB CSV
  rewrite, to learn nothing, because the only guard compared the newest bar against the
  calendar date and that is false all day. A skipped refresh is printed in the report
  header, never silent. `--refresh` or `CACHE_REFRESH_HOURS=0` forces it.
- Don't refactor purely for style. Prefer targeted bug fixes and well-scoped features.

## Verifying changes

```bash
python -m unittest discover -s tests      # attendu : Ran 421 tests ... OK
#   Mesuré le 2026-08-19 : 421 tests, 1 erreur de CHARGEMENT --
#   tests/test_dashboard_render.py fait `from nicegui import ui` et nicegui n'est
#   pas installe dans l'environnement courant (Python 3.14). Ce n'est pas une
#   regression : `pip install -r requirements.txt` la fait disparaitre. Le chiffre
#   de 466 annonce precedemment n'a pas ete reproduit -- re-mesurer avant de le citer.
```

Deterministic and network-free. Three things it checks that are worth knowing about:

1. **Golden master** (`tests/test_golden_master.py`) pins 8 strategies × 6 metrics over a
   frozen daily fixture. If your change moves a number, this fails **on purpose**. Regenerate
   with `python -m tests.test_golden_master --record` and commit the JSON *in the same commit
   as the change*, with a new entry appended to its `history` array. Never regenerate it to
   make a test go green.
2. **Guards** (`tests/test_guards.py`) assert what the engine must REFUSE: an incomplete
   month, a window predating a strategy's own products, weights that do not sum to 1, a held
   ticker with no price, a data gap longer than 5 trading days, a canary reading bullish on
   NaN.
3. **Paper rules** (`tests/test_paper_rules.py`) compare each family's allocations against
   its published rule, re-derived on hand-built panels. This file exists because two
   Keller-compliance defects survived two audits and 131 self-consistency tests. The
   standard it must meet is stated in its docstring: change `rolling(13)` to `rolling(9)`,
   drop DAA's cash slots, swap 13612W for 13612U, invert a BAA canary comparison — each has
   to break an assertion THERE. A golden master can only say a number moved; it cannot say
   which rule is right.
4. **Anchors** (`tests/test_anchors.py`) compare against things the code did not produce:
   momentum scores derived from a closed form in the test file, HAA baskets worked out by
   hand, Keller's stated 0.1% one-way cost expressed as arithmetic, the HAA paper's published
   result shape as a sanity band, and the registry count.

That third category is the point. Before 2026-07-28 all 55 tests asserted `f(x) == f(x)`, and
every one of them passed against an engine with four critical defects. **A new test that only
compares the code to itself adds coverage but no assurance.**

After a change also run a real backtest and read the coverage line, the "what was actually
traded" table and the regime panel — not just the CAGR column. Numbers shift legitimately when
execution conventions, leverage or defensive handling change; the manifest beside each saved
report records which conditions produced it.

## Key documentation

- `KNOWN_GAPS.md` — **read this before quoting any number.** What the engine cannot establish,
  what it does not model, and the eight findings deleted with their code rather than fixed.
- `ARCHITECTURE.md` — module layout and the execution model
- `LEVERAGE.md` — leverage rationale (Kelly criterion, DAA vs VAA under leverage)
- `WHY_TAA.md` — the case for tactical asset allocation
- `TIMELINE.md` — evolution of TAA from Faber to Keller
