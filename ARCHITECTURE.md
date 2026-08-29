# Keller Strategies — Architecture

Last rewritten 2026-07-28, after the execution-ledger refactor. If a description here
disagrees with the code, the code is right and this file is a bug.

## Principles

- **Separation of signal and execution.** A strategy answers *what to hold*. It never decides
  *at what price*, *on what date*, *at what cost*, or *with what leverage*. Those four
  questions belong to `common/ledger.py` and are configured, not implied.
- **Nothing is inferred.** Every strategy declares its sleeves explicitly. Every execution
  choice is a named field with a documented default. The engine has no "best guess" path,
  because the defects the 2026-07-28 audit found were all guesses that looked like behaviour.
- **Refuse rather than approximate.** An allocation the engine cannot price raises. A month it
  cannot vouch for is not emitted. A missing price is not a 0% return.
- **Academic fidelity, declared per strategy.** Every registered entry carries a `fidelity`
  (`faithful` / `proxy` / `custom`) and a `role` (`strategy` / `control` / `benchmark`) as
  class attributes, printed in the report's `Type` column. The default is the WEAK claim —
  an entry that forgets to declare is reported as custom — and `LeveragedWrapMixin` hard-codes
  `custom` so a wrap can never inherit its parent's claim. Sixteen entries that could not earn
  an honest label were **deleted** on 2026-07-28 rather than annotated; the admission rule and
  the deletion list live above `main.ALL_STRATEGIES` and in
  [`KNOWN_GAPS.md`](KNOWN_GAPS.md) §5-6.
- **Defence is held at 1x. Always.** A defensive sleeve containing a leveraged product raises
  the risk it exists to cut. `run_backtest` calls `assert_unlevered_defensive` before pricing
  anything and FAILS the strategy otherwise.

## The three layers

```
DATA       common/data_engine.py   PriceStore — daily bars, real trading dates,
                                   complete months only, provenance hash
SIGNAL     strategies/*.py         momentum / canary logic -> target weights on the
                                   tickers actually traded (LETF images included)
EXECUTION  common/ledger.py        run_ledger — fill date, fill price, drift, per-leg
                                   cost, cash account, margin and interest
REPORTING  main.py + common/       metrics, coverage, regimes, selection statistics,
                                   run manifests
SIZING     common/margin_sizing.py what MARGIN leverage a record survives, driven by
           + leverage_advice.py    common/leverage_advice.py. Stands BESIDE the engine:
                                   run_ledger never tests a maintenance requirement
DOUBT      common/robustness.py    PBO by combinatorially symmetric cross-validation, the
                                   leaderboard rebuilt on joint bootstrap resamples — both
                                   pooled by family and by mechanism — and the leverage
                                   frontier walking those same histories at constant
                                   margin. Measures the SELECTION, not the future
UI         app.py (NiceGUI)        thin GUI over load_data / run_backtest /
                                   compute_live_signals — no strategy logic
```

## `common/data_engine.py` — `PriceStore`

Caches **daily** `open` / `close` / `adj_close` per ticker in `data/cache/daily_*.csv`.

```python
store = PriceStore(TICKERS, start='2000-01-01', cache_dir='data/cache')
store.adj_close()                 # daily, indexed by REAL trading dates
store.adj_open()                  # opens rebased: adj_open = open * adj_close / close
store.month_end_dates(start, end) # last trading day of each COMPLETE month
store.rebalance_dates('day_7')    # Nth trading day of each complete month
store.monthly_adj_close(dates)    # the signal panel, sampled at real dates
store.first_tradable_date('TQQQ') # -> 2010-02-11
store.provenance()                # sha256, row count, as_of, last complete month, fills
```

Four rules, each of which is a fix:

1. **A month is emitted only once the store holds a trading day in a LATER month.** The old
   `resample('ME').last()` had no such notion and published a five-day-old price under a
   month-end label, which corrupted a live order basket in production.
2. **Index labels are real trading dates** — `2023-12-29`, never the synthetic `2023-12-31`.
3. **Incremental refresh re-downloads and OVERWRITES a trailing 90-day window.** The old
   appender accepted only strictly-newer dates, so a row written mid-month survived forever.
4. **One missing-data policy, applied once and recorded.** Forward-fill at most
   `MAX_STALE_DAYS = 5` trading days *inside* a ticker's life; leading NaNs are never filled
   (that is what makes `first_tradable_date` meaningful) and a price is never carried past a
   ticker's last real observation. A longer interior gap raises `DataGapError`.

Alternative constructors — `from_adjusted`, `from_daily_fixture`, `from_monthly_csv`,
`from_frames` — build stores for tests and for the frozen fixtures. A store without opens
reports `has_intraday = False`, and an open-priced execution convention then **raises** rather
than silently falling back to the close.

## `common/coverage.py` — from when can this be measured?

```python
tradable_universe(strategy)   # everything it can HOLD, LETF images included
signal_universe(strategy)     # tradable + canaries
earliest_valid_start(strategy, store, warmup_months=13)
coverage_report(strategy, store, requested_start)   # -> trimmed?, binding ticker, message
```

`earliest_valid_start` works over the **signal** universe, not just the tradable one: a signal
computed from a canary that has no data is not the strategy, it is whatever the NaN policy
happens to do. A ticker missing from the store entirely raises `KeyError` — that is a
configuration error, not a short history, and trimming around it would silently change what
the strategy is.

## `common/ledger.py` — the execution model

Replaces the single expression that used to be the whole of execution:

```python
strat_ret = (alloc_shifted * monthly_returns).sum(axis=1) * LEVERAGE - friction
```

That line silently decided five separate things, each of which became an audit finding: the
execution price, the rebalance date, the meaning of a missing price, the meaning of cost, and
the meaning of cash.

```python
@dataclass
class ExecutionConfig:
    convention: str = 'next_open'      # | 'next_close' | 'signal_close'
    cost_bps_per_side: float = 10.0    # ONE-WAY, per leg
    cash_ticker: str = 'BIL'
    borrow_rate: float = 0.06
    leverage: float = 1.0
    leverage_follows_signal: bool = True
    charge_terminal_liquidation: bool = True
    strict_invariants: bool = True

result = run_ledger(target_weights, store, exec_cfg, sleeves, label=name)
```

Semantics, stated because the old code left all of them implicit:

- **Decision date D** is a real trading date; the signal saw closes up to and including D.
- **Execution date E** is the first session strictly after D. The fill is `adj_open[E]` under
  `next_open`. `signal_close` reproduces the pre-audit engine and exists for the golden master
  only — it is look-ahead by construction, worth ≈ +0.37 pp of CAGR on HAA-G8.
- Positions are valued throughout at the **execution price series**, so a period return is
  open-to-open or close-to-close, never a mix.
- Positions are held as **notional and they drift**. Trades are `target − drifted`, which
  makes the free drift-rebalance impossible by construction.
- **Cost is charged per leg on the notional actually traded**, including initial deployment
  and terminal liquidation (both reported separately). A full A→B rotation costs
  `2 × cost_bps_per_side`.
- **Cash is an account.** Uninvested weight earns `cash_ticker`. `None` means 0% *and* a
  recorded warning.
- **Leverage acts on the sleeve it is borrowed against.** With `leverage_follows_signal`, the
  defensive sleeve is held at 1x and interest is day-counted on the debit balance actually
  drawn. A deliberate cash-ticker *holding* follows the mode's rule; only the uninvested
  residue is always unlevered.

`validate_targets` runs before anything is priced and raises `WeightInvariantError` when a
rebalance's weights do not sum to 1 or when a held ticker has no price on its execution date.
`LedgerResult` carries `returns`, `gross_returns`, `turnover`, `n_trades`, `cost_paid`,
`effective_leverage`, `cash_weight`, `equity`, `exec_dates` and `warnings`.

## `common/metrics.py`

`calculate_metrics(returns, rf, start_label)` returns a **dict**, not the former 6-tuple —
adding a seventh positional value to something unpacked at every call site is a silent
corruption waiting to happen, and UPI made a seventh value necessary.

- The wealth curve **includes the initial 1.0**, so a first-month drawdown is visible. It used
  to report `[-50%, 0%, +10%]` as MaxDD = 0.00%.
- `rf` accepts a scalar annual rate or a realised monthly cash series. `build_rf_series`
  produces the latter from the cash ticker's own history, falling back to a documented
  constant before BIL's 2007-05 inception and *saying which months were assumed*.
- `ulcer_index` (RMS drawdown) and `ulcer_performance_index` = `(CAGR − rf) / UI` — Keller's
  own primary ranking metric.

## `common/eras.py`, `common/regimes.py`, `common/selection.py`, `common/manifest.py`

- **eras** — the file that replaced the date picker. `COMMON_ERA_START = '2004-11-01'` is
  derived, not chosen: the download starts 2000-01 and SPY is the only ticker that old, plus
  13 complete months of warm-up. It moved back from 2008-07 on 2026-07-29, when the history
  extension freed the strategies that do not hold high yield. Four **partitions** of that era ship frozen — NBER business cycle, S&P 500
  bull/bear, FOMC rate cycle, BLS CPI inflation regime — each one exhaustive and disjoint, so
  its segments compound back to the era's own return (`tests/test_eras.py` asserts exactly
  that, and `validate_partition` runs at import so a typo is an ImportError). `common_window`
  derives the one window the ranked table uses: the latest inception among the strategies
  being compared, floored at the era start, with the binding strategy named.
- **regimes** — eight frozen drawdown episodes plus family publication dates. Unlike the
  partitions these are *not* exhaustive — they are named stress windows, and they reach back
  before the era for the few assets that existed. A panel cell prints `n/a (inception
  YYYY-MM)` where the strategy's own products did not exist. Never a number, never 0%.
- Both panels are measured over each strategy's **full** measurable history, *not* over the
  ranked window — otherwise they would inherit the window and stop being an antidote to
  date-selection bias. That is why `run_backtest` runs the ledger **twice** per strategy: once
  over the common window for the headline table, once over `[coverage_earliest, last complete
  month]` for the panels and the publication split.
- **selection** — expected best-of-N under the null (Blom's order statistic, with a bisection
  normal quantile so scipy stays out), the participation ratio (effective independent bets),
  and Spearman rank stability between disjoint sub-periods. No scipy.
- **manifest** — every saved report gets a `*.manifest.json` sibling: commit, dirty-diff hash,
  data provenance, resolved config, dependency versions, and the window actually measured.
  Account balances are removed by an explicit redaction list.

## Strategies (`strategies/*.py`)

```python
class SomeStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("SomeStrategy", is_active=True)   # False for passive benchmarks
        self.offensive = [...]
        self.canary = [...]
        self.defensive = [...]

    def sleeves(self):
        """MANDATORY. No inference, no attribute sniffing."""
        return {'offensive': set(self.offensive),
                'defensive': set(self.defensive),
                'canary': list(self.canary)}

    def generate_allocations(self, prices, scores_13612w, ret_12m, ret_3m):
        """Faithful algorithm. Returns target weights per asset per date."""
```

`BaseStrategy.sleeves()` **raises** by default. It replaced a resolver that sniffed five
different attribute names for a strategy's cash bucket and silently returned an empty set when
a strategy used a sixth — which is exactly what happened to RAA. A vocabulary that must be
guessed will be guessed wrong again; declaring it is the only fix that does not recur.
`tests/test_anchors.py::TestRegistry` asserts the declaration for all 36 registry keys, plus
two more invariants: every entry declares a `fidelity` and a `role`, and **no entry declares a
leveraged product in its defensive sleeve** — `common/letf_mapper.assert_unlevered_defensive`,
which `run_backtest` also calls before pricing anything. Defence is held at 1x, always.

`defensive_mask(alloc, scores)` and `offensive_weight(alloc, scores)` resolve the per-month
split, with dual-role assets (TLT/DBC/LQD in BAA, IEF in HAA) disambiguated by the canary and
defaulting to **offensive** when there is no canary — the conservative direction.

Leveraged variants use a **wrap pattern**: subclass the canonical strategy, run its exact
signal on 1x assets, then `LETFMapper.translate(...)`. They mix in `LeveragedWrapMixin` so
their `sleeves()` declares the LETF images actually held. Admissible universes are those that
execute at one uniform multiple — G2 (SPY/QQQ) and G3 (SPY/QQQ/IWM), plus G4 (+GLD) at 2x
only. 2x wraps carry `role='strategy'`; since 2026-07-29 the registry also carries twelve 3x
entries under `role='exploratory'` — measured and reported in full, excluded from every
selection statistic (`strategies/base.py`).

## `main.py` — orchestration

1. `load_store(config)` → `PriceStore`; `build_signal_panel(store, config)` → monthly panel
   and momentum scores **over the full history**, ending at the last COMPLETE month. Slicing
   prices at a start date first and then dropping 13 warm-up rows is what made a report
   labelled 2015-2025 measure 2016-02 → 2024-12.
2. **Phase 1, nothing priced yet:** for each strategy, generate allocations → `coverage_report`
   (trim and say why, or fail under `COVERAGE_POLICY='strict'`). Then `eras.common_window`
   over the survivors fixes ONE decision calendar for the whole table, so its rows are
   comparable by construction rather than by hope.
3. **Phase 2:** for each strategy, slice to those decision dates → `run_ledger` →
   `calculate_metrics` with a realised risk-free series; then a second ledger pass over the
   strategy's own full history for the regime panels.
4. A strategy that fails an invariant is reported as a **FAILURE and contributes no metrics**.
   A month the engine cannot price is a month with no return, not a month with a 0% return.
5. `print_report` emits the ranked table plus five blocks that are not decoration: what was
   actually traded, the four regime partitions, the named stress episodes, the
   before/after-publication split, and the selection statistics with rank stability.
6. `save_outputs` writes the report, the chart and the manifest.
7. **Live mode** is presentation-free at the core: `compute_live_signals(...)` and
   `size_positions(...)` return plain data; `run_live_mode(...)` is the CLI printer and the
   GUI's Live tab is the other consumer. Share quantities are sized at the latest live quote
   (`get_live_prices`, raw Close) while the signal stays month-end.

## GUI (`app.py`, NiceGUI)

Thin dashboard over the engine — no strategy logic. Backtest tab calls `main.load_data` +
`main.run_backtest`; Live tab calls `main.compute_live_signals`. Reads and writes
`user_config.json`. Native window with `pywebview` (`python app.py`, or `run_dashboard.bat`),
otherwise the browser.

## Adding a strategy

1. New class in the appropriate family file, inheriting `BaseStrategy`.
2. Universes and parameters in `__init__`.
3. **`sleeves()` — mandatory.** Include LETF images if it is a wrap.
4. `generate_allocations` per the paper, with the citation in `strategy_specs/`.
5. Register it in `ALL_STRATEGIES` and update the count assertion in
   `tests/test_anchors.py::TestRegistry` — deliberately, with the reasoning in the commit.
6. Run `python -m unittest discover -s tests`, then a real backtest, and check that the
   coverage line and the regime panel say what you expect.
