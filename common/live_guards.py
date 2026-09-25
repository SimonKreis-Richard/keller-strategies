"""
The refusals that stand between a monthly allocation and a real order.

WHY THIS IS A MODULE AND NOT A BLOCK INSIDE `compute_live_signals`
------------------------------------------------------------------
It was a block inside `compute_live_signals` until 2026-09-17, and that was fine while
exactly one code path produced orders. It stopped being fine when a second one appeared: the
Canadian execution layer (`common/ca_mapping.py` and its driver) computes the same
allocation row and sizes it into real trades, so a guard that lives in the caller is a guard
the second caller can forget.

The comment above the block being extracted already said what happens then, about the
backtest's guards reaching the live path a year late:

    These ran only in `run_backtest` until 2026-07-29. Being latently satisfied is not the
    same as being enforced.

Extracting them costs one indirection and removes the possibility entirely. This module
imports nothing from `main`, on purpose: importing `main` prints a report banner and drags
the whole registry in — `common/facts.py` already has to wrap that import in
`redirect_stdout` to survive it.

WHAT IS AND IS NOT HERE
-----------------------
`live_guards` holds every refusal that needs the ALLOCATION ROW to exist. There is one other
live guard, `letf_mapper.assert_unlevered_defensive`, and it necessarily runs EARLIER —
before `generate_allocations` is called at all, because a strategy whose defensive sleeve is
levered must not be asked for an allocation in the first place. It is therefore not folded
in here, and `assert_every_live_guard_ran` exists so a caller can state that it ran both
rather than hoping.

THE RETURN VALUE IS A LIST OF PROBLEMS, NOT A BOOLEAN
-----------------------------------------------------
Every entry is a line the user reads next to the orders that were refused. An empty list
means "all of these were checked and all of them passed"; it never means "nothing was
checked", because the store-verification branch below turns an unverified store into a
problem rather than into silence. That asymmetry is the whole design: an `ok` that means
"I verified nothing" is the failure mode this repository keeps closing.
"""

from common.ledger import validate_row


def live_guards(strat, alloc, prices, signal_date, store=None):
    """Every reason to refuse to size `alloc` into orders, as a list of message lines.

    Extracted verbatim from `main.compute_live_signals` on 2026-09-17. Behaviour is
    unchanged: same checks, same order, same wording, so the messages a user has learned to
    read do not move under them.

    * `strat`  — the strategy; `sleeves()` is read for the blast radius of a data refusal.
    * `alloc`  — the full allocation frame; only the last row is judged.
    * `prices` — the monthly panel; `prices.loc[signal_date]` must exist.
    * `store`  — the `PriceStore`. **`None` disables the two data checks**, which is correct
      for a caller that has no store (the GUI's fixture mode) and is the reason the
      unverified-store branch below is written against `store is not None`: a caller that
      HAS a store and has not verified it is refused, a caller that has none was never
      claiming otherwise.
    """
    problems = validate_row(alloc.iloc[-1], prices.loc[signal_date],
                            label=strat.name, when=signal_date)
    # Nothing invented may be sized into an order. Every constructed span ends before
    # its ticker's real inception by construction, so this is cheap and makes the claim
    # structural rather than incidental — today it holds by ~18 years of margin.
    if store is not None:
        row = alloc.iloc[-1]
        for t in row[row > 1e-9].index:
            real_from = store.constructed_before(t)
            if real_from is not None and signal_date < real_from:
                problems.append(f'  {t} is CONSTRUCTED before {real_from.date()}; the '
                                f'signal date {signal_date.date()} predates the fund')
    # A price that disagrees with its own adjustment history must not be sized into an
    # order. Found 2026-09-01: a spliced vintage understated momentum and flipped a live
    # canary, so the book went to cash a month early. The blast radius is deliberately
    # THIS strategy's decision, not the whole registry — and it covers the sleeves and
    # the canary, not merely what is held: a corrupted score on an unheld candidate is
    # what changes the selection, and a dead canary sends the book to cash without ever
    # appearing in the allocation row.
    if store is not None:
        v = getattr(store, 'verification', None)
        if v is None or v.get('status') in ('not_applicable', 'skipped'):
            problems.append('  the price store was never verified against its own '
                            'adjustment history; live orders must not be sized from '
                            'data nothing has checked')
        elif v.get('status') == 'disagrees':
            sl = strat.sleeves()
            universe = set(alloc.iloc[-1][alloc.iloc[-1] > 1e-9].index)
            universe |= set(sl.get('offensive', ())) | set(sl.get('defensive', ()))
            universe |= set(sl.get('canary', ()))
            hit = sorted({x['ticker'] for x in v['violations']} & universe)
            if hit:
                problems.append(
                    f'  {", ".join(hit)} carr{"ies" if len(hit) == 1 else "y"} more '
                    f'than one dividend-adjustment vintage, so momentum over windows '
                    f'crossing the seam is wrong (and wrong LOW). Re-run with '
                    f'--refresh, or delete data/cache to rebuild.')
    return problems


def refusal_message(problems):
    """The one sentence every live path prefixes its refusals with.

    Here rather than at each call site so the two order-producing paths cannot describe the
    same refusal differently — which is how a user ends up believing they are two different
    problems.
    """
    return 'refusing to size this allocation:\n' + '\n'.join(problems)
