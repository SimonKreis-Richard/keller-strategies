# Contributing to Keller Strategies

First off, thank you for considering contributing to Keller Strategies! It's people like you that make open-source such a great community to learn, inspire, and create.

## 🚀 How Can I Contribute?

### 1. Reporting Bugs
If you find a bug in the code, or a mathematical discrepancy between a strategy's implementation and its original academic paper, please open an Issue. Include:
* A clear description of the issue.
* The specific strategy name and parameters.
* References to the academic paper highlighting the discrepancy (if applicable).

### 2. Suggesting Enhancements / New Strategies
We are always looking to expand the suite of available Tactical Asset Allocation strategies. If you want to request or implement a new strategy:
1. Open an Issue outlining the strategy concepts and linking to its SSRN/academic paper.
2. Ensure the strategy relies on monthly rebalancing (as this is the core engine's structure) and relies on standard Yahoo Finance price data.

### 3. Pull Requests
If you want to contribute code directly:
1. **Fork the repo** and create your branch from `main`.
2. **Implement your strategy** by inheriting from `BaseStrategy` in `strategies/base.py`. Use
   the `common.momentum` tools for momentum/SMA calculation.
3. Add your strategy to `ALL_STRATEGIES` in `main.py`, and update the registry-count assertion
   in `tests/test_anchors.py::TestRegistry` deliberately, with the reasoning in your commit
   message. The count is asserted precisely so the registry cannot quietly regrow.
4. Ensure your code is documented in **English** and follows PEP-8.
5. Make sure it runs without errors in both Backtest and Live modes.
6. Run the full suite: `python -m unittest discover -s tests`.

### Five hard requirements for a new strategy

These are not style preferences. Each one exists because its absence caused a real defect that
survived multiple review passes.

1. **Declare `sleeves()`.** Return `{'offensive': set, 'defensive': set, 'canary': list}`.
   `BaseStrategy.sleeves()` raises; there is no inferred default and there will not be one. A
   leveraged wrap must declare the **LETF images it actually holds** — mix in
   `LeveragedWrapMixin` and that is handled. The resolver this replaced sniffed five attribute
   names and silently returned an empty set for a strategy that used a sixth.
2. **Ship a paper citation.** Add a `strategy_specs/<name>.md` that cites the paper, the page,
   and the verbatim rule, then notes every deliberate deviation (proxy tickers, universe size,
   parameter choices). A spec written *from the code* is circular and has no audit value — the
   whole point is to have something outside the implementation to check it against. If the
   strategy is custom, say so in its name and its docstring, as
   `DM_G3_Leveraged_2X` does. And check the other direction before you claim custom: the
   four-module composite carried a `_Custom` suffix for months because it was compared to
   the wrong Antonacci paper. A wrong label is a wrong label in both directions.
3. **The backtest must be coverage-guard clean.** Run it from a start date well before your
   universe's inception and confirm the engine *trims and names the binding ticker* rather
   than producing numbers. If `run_ledger` raises `WeightInvariantError`, your allocation has
   months that do not sum to 1 or that hold a ticker with no price — fix the strategy, never
   the invariant.
4. **Add at least one test with an external anchor.** A test that compares the code to itself
   adds coverage but no assurance: before 2026-07-28 all 55 tests did exactly that, and every
   one passed against an engine with four critical defects. Anchor to arithmetic done by hand
   in the test file, a published table, or a case whose answer is known independently. See
   `tests/test_anchors.py`.
5. **Test your SELECTION RULE against the paper, in `tests/test_paper_rules.py`, and cite the
   source in a `source` class attribute.** Requirement 4 is not enough on its own, and that
   is not a hypothesis: on 2026-07-29 a third audit found DAA applying an absolute-momentum
   filter its paper disclaims three times, and `SMA12` averaging twelve prices where Keller
   defines thirteen. Both had passed 131 tests and two audits, because every strategy test
   asserted self-consistency — weights sum to one, a dead canary de-risks — and a wrong rule
   satisfies all of those. Build a panel whose correct allocation you can derive on paper,
   and assert that. Never call the production function to decide what the production function
   should have produced. The `source` attribute (paper, section, figure) is asserted for every
   registered entry, because `fidelity` has now been wrong three times as a bare string.

If your change moves any number, `tests/test_golden_master.py` will fail. That is intended.
Regenerate with `python -m tests.test_golden_master --record` and commit the JSON **in the
same commit as the change**, with a new entry in its `history` array explaining what moved and
why. Never regenerate it just to get a green run.

## Code Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full picture. In brief:

* `main.py` — orchestration, parameters, reporting.
* `common/data_engine.py` — `PriceStore`: daily bars, real trading dates, complete months only.
* `common/ledger.py` — the execution model: fill date, fill price, cost, cash, leverage.
* `common/coverage.py` — from when a strategy can honestly be measured.
* `common/eras.py` — the frozen era and its four pre-registered regime partitions. **Editing a
  segment boundary is a change to the measuring instrument, not to a parameter.** Every
  boundary is a dated public fact (NBER, FOMC, BLS, S&P 500); if you move one, cite the source
  in the commit message, and never move one because a strategy looks bad on one side of it.
  `validate_partition` runs at import, so a partition with a gap or an overlap is an
  ImportError rather than a quietly wrong panel.
* `common/metrics.py`, `regimes.py`, `selection.py`, `manifest.py` — reporting and provenance.
* `strategies/` — one file per family. Each strategy implements `generate_allocations(self,
  prices, scores_13612w, ret_12m, ret_3m)` **and** `sleeves(self)`.
* `strategy_specs/` — the paper-side spec for each algorithm.
* `tests/` — guards, external anchors, and the golden master.

Read [`KNOWN_GAPS.md`](KNOWN_GAPS.md) before quoting any number from this repository.

Happy Backtesting!
