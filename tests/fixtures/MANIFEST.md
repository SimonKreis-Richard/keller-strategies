# Test fixtures — manifest

Every file here is **frozen input**. Tests that assert numbers are only meaningful if the
data underneath them cannot drift, so these files are tracked in git (`.gitignore` carries an
explicit `!tests/fixtures/*.csv` negation) and must never be regenerated casually. Changing a
fixture changes the meaning of every test that reads it: re-record the hash below in the same
commit and say in the commit message what moved and why.

| file | sha256 | shape | span | role |
|---|---|---:|---|---|
| `frozen_prices_2026-06-08.csv` | `299c2d3fcd38c5351b81214bd972c6aa95990e4ef2b7e1cb3292c64a0734ee94` | 318 × 59 monthly | 2000-01-31 → 2026-06-30 | partial-month regression (C4) |
| `frozen_daily_adj_close.csv` | `cc5d5fa13918e428d208dbfae9bbbc194cc6db6f77bae113100c10a4831f060f` | 3290 × 24 daily | 2012-01-03 → 2025-01-31 | golden master, ledger tests |
| `frozen_daily_adj_open.csv` | `cecf3fde7dd93cad7541193627673e52c011f4f2655ae62ae4aaf9845aace212` | 3290 × 24 daily | 2012-01-03 → 2025-01-31 | golden master, ledger tests |
| `golden_master.json` | (regenerated with the behaviour it describes) | 8 strategies | 2015-01-01 → 2025-01-01 | frozen engine metrics |

## `frozen_prices_2026-06-08.csv`

A byte-for-byte copy of `data/cache/prices_master_cache.csv` as it stood on 2026-07-28,
taken **before** the cache was purged. It is the monthly (month-end labelled) adjusted-close
panel the pre-refactor engine consumed.

**Its last row is deliberately corrupt and must stay that way.** The row is labelled
`2026-06-30` but holds prices from `2026-06-05`: the month had not finished when the cache was
written, and `resample('ME').last()` emitted the partial month under a synthetic month-end
label anyway. That is audit finding C4, and this file is its regression test
(`tests/test_guards.py::test_partial_month_is_refused`). Do not "fix" the row.

Column `IWN` is present in the file but no longer in the engine's ticker list — it was orphaned
when RAA was deleted. Extra columns are harmless; the store selects what it needs.

## `frozen_daily_adj_close.csv` / `frozen_daily_adj_open.csv`

Daily bars for the 24 tickers the golden-master strategies can touch, 2012-01 → 2025-01,
**both already on the adjusted scale** (`adj_open = open × adj_close / close`, computed once
at capture). Storing adjusted opens rather than a raw/adjusted pair means there is no factor
to keep in sync and no way for the fixture to drift when Yahoo restates a dividend.

They are daily, not monthly, because the production execution convention fills at the OPEN of
the session AFTER the decision. A monthly panel cannot express that at all — which is why the
pre-audit engine had no choice but to fill at the signal's own close.

Neither file contains a single NaN over its span. That is a property worth preserving: it
means a test failure here is a logic failure, never a data artefact.

## `golden_master.json`

Metrics for 8 strategies over 2015-01-01 → 2025-01-01 at `LEVERAGE_FACTOR=1.0`, recorded to 6
decimal places. Its purpose is to make every behaviour change visible: when a fix moves a
number, the JSON is updated **in the same commit as the fix**, and that diff is the audit
trail of which fix moved which number. The file's own `history` array records each such move.
