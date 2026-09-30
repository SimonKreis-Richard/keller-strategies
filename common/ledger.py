"""
The execution ledger.

Before this module existed, the whole of execution was one vectorised expression:

    strat_ret = (alloc_shifted * monthly_returns).sum(axis=1) * LEVERAGE_FACTOR - friction

That single line silently decided five things, each of which turned out to be a separate
audit finding: the execution *price* (it used the very close that produced the signal), the
rebalance *date*, the meaning of a *missing price* (`sum(skipna=True)` turned it into a 0%
return), the meaning of *cost* (a function of weight change, not of trades), and the meaning
of *cash* (nothing — uninvested weight evaporated).

`run_ledger` makes all five explicit. Positions are carried as **notional**, they **drift**
between rebalances, trades are the difference between target and drifted notional, cost is
charged per leg on what was actually traded, uninvested weight sits in a real cash asset,
and leverage is applied to the sleeve it is actually borrowed against.

Accounting, stated once so nothing is left implicit:

* **Decision date D** — a real trading date. The signal saw closes up to and including D.
* **Execution date E** — under `next_open` (the default) the first session strictly after D,
  filled at `adj_open[E]`. `next_close` fills at that session's close. `signal_close` fills at
  D's own close; it reproduces the pre-audit engine and exists for the golden master, not for
  describing a portfolio anyone could have held.
* Positions are valued throughout at the **execution price series** — opens under `next_open`,
  closes otherwise — so a period return is open-to-open or close-to-close, never a mix.
* The return over interval (E_i → E_{i+1}) is indexed by the **later decision date** D_{i+1},
  matching the convention a monthly report reads with: row t is the return earned during
  period t from the allocation decided at t-1.
* Cost for the trade executed at E_i is deducted at E_i, i.e. inside the interval reported at
  D_{i+1} — again matching the old convention, so the two are comparable.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONVENTIONS = ('next_open', 'next_close', 'signal_close')


class WeightInvariantError(ValueError):
    """A target allocation the engine refuses to price.

    Raised for a rebalance whose weights do not sum to 1, or which holds a ticker that has
    no price on its execution date. Both used to pass silently: `DataFrame.sum(axis=1)`
    skips NaN, so a month 100% invested in a product that did not yet exist was reported as
    a +0.00% return, and a month whose weights summed to 0.0 was reported the same way.
    Warning and continuing is exactly the failure mode being removed here.
    """


@dataclass
class ExecutionConfig:
    """Everything the old one-liner decided implicitly, made explicit and configurable."""

    #: 'next_open'  — fill at the open of the session AFTER the decision. The default, and
    #:                what the owner actually does (after-hours GTC orders).
    #: 'next_close' — fill at that session's close.
    #: 'signal_close' — fill at the decision close. Pre-audit behaviour; look-ahead by
    #:                construction (worth +2.09 pp of CAGR on HAA-G8). Golden master only.
    convention: str = 'next_open'

    #: ONE-WAY cost, per leg, in basis points of traded notional. Keller's HAA paper states
    #: "we assume a one-way transaction costs of 0.1% (=TC)", so a full A->B rotation costs
    #: 2 x 10 bps. The pre-audit engine charged sum|dw|/2 x 0.1%, i.e. half of one-way.
    cost_bps_per_side: float = 10.0

    #: Uninvested weight earns this asset's return. None => 0% and a recorded warning.
    cash_ticker: str = 'BIL'

    #: Annual rate on the debit balance actually drawn.
    borrow_rate: float = 0.06

    #: Margin multiple (separate from leveraged-ETF products, whose financing is already in
    #: their price history).
    leverage: float = 1.0

    #: True  => borrow only against the OFFENSIVE sleeve, so the loan is repaid as the signal
    #:          de-risks. effective_leverage = 1 + (L-1) x offensive_weight.
    #: False => flat leverage, loan drawn in risk-off months too.
    leverage_follows_signal: bool = True

    #: Charge a liquidation of the final book. Reported separately either way.
    charge_terminal_liquidation: bool = True

    #: Refuse to price a rebalance whose weights do not sum to 1, or which holds a ticker
    #: with no price on its execution date. Turning this off is a research affordance, not
    #: a production one: every number produced with it off is unaudited.
    strict_invariants: bool = True

    def __post_init__(self):
        if self.convention not in CONVENTIONS:
            raise ValueError(f'convention must be one of {CONVENTIONS}, got {self.convention!r}')
        if self.cost_bps_per_side < 0:
            raise ValueError('cost_bps_per_side must be >= 0')


@dataclass
class LedgerResult:
    returns: pd.Series                 # net, indexed by decision date of the period END
    gross_returns: pd.Series           # before costs and interest
    turnover: pd.Series                # notional traded / equity, per rebalance
    n_trades: pd.Series                # positions touched per rebalance
    cost_paid: pd.Series               # currency cost per rebalance, on equity = 1.0 at start
    effective_leverage: pd.Series
    cash_weight: pd.Series
    equity: pd.Series                  # wealth curve, starts at 1.0 on the FIRST execution
    exec_dates: pd.Series              # decision date -> execution date
    first_date: pd.Timestamp = None
    last_date: pd.Timestamp = None
    initial_deployment_cost: float = 0.0
    terminal_liquidation_cost: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def annual_turnover(self):
        """Realised one-way turnover per year, as a fraction of equity."""
        if self.turnover.empty or self.first_date is None:
            return float('nan')
        years = max((self.last_date - self.first_date).days / 365.25, 1e-9)
        return float(self.turnover.sum() / years)


def _execution_price_frame(store, convention):
    return store.adj_open() if convention == 'next_open' else store.adj_close()


def validate_row(weights, prices_row, label='strategy', when=None, fill=None,
                 tolerance=1e-6):
    """The single-row form of the two invariants below. Returns a list of problems.

    Factored out so the LIVE path and the BACKTEST path cannot drift apart. Until
    2026-07-29 only the backtest validated anything: `compute_live_signals` went straight
    from `generate_allocations` to `size_positions`, which turns weights into whole-share
    orders against real balances. Every guarantee this repository makes about an allocation
    was enforced on the path that spends no money and skipped on the path that does.

    `prices_row` is a Series of prices for the date the weights would be executed at.
    """
    problems = []
    stamp = f'{pd.Timestamp(when).date()}: ' if when is not None else ''
    total = float(weights.sum())
    if abs(total - 1.0) > tolerance:
        problems.append(f'  {stamp}weights sum to {total:.6f}, not 1.0 '
                        f'(held: {sorted(weights[weights.abs() > 1e-9].index.tolist())})')
    held = weights[weights > 1e-9].index
    cols = set(prices_row.index)
    blind = sorted(t for t in held if t not in cols or pd.isna(prices_row.get(t)))
    if blind:
        at = f'(fill {pd.Timestamp(fill).date()}) ' if fill is not None else ''
        problems.append(f'  {stamp}{at}no price for {blind}, which carry '
                        f'{float(weights.reindex(blind).fillna(0.0).sum()):.1%} of the book')
    return problems


def validate_targets(target_weights, decisions, exec_dates, px, label='strategy',
                     tolerance=1e-6, max_report=8):
    """Refuse an allocation the engine cannot honestly price.

    Two invariants, both of which the pre-audit engine violated in production:

    * every rebalance's weights sum to 1.0 — `DAA_G12`, `VAA_G12` produced months summing to
      **0.0** (their best defensive asset was NaN, so nothing was allocated) and those months
      were reported as a flat 0% rather than as the missing data they were;
    * every ticker carrying weight has a price on its execution date — a 3x variant run from
      a 2000 start held 100% of the book in a product that did not exist for 32 months.
    """
    problems = []
    for d, e in zip(decisions, exec_dates):
        problems += validate_row(target_weights.loc[d], px.loc[e], label=label, when=d,
                                 fill=e, tolerance=tolerance)
        if len(problems) >= max_report:
            problems.append(f'  ... (stopped after {max_report} problems)')
            break
    if problems:
        raise WeightInvariantError(
            f'{label}: refusing to price this allocation.\n' + '\n'.join(problems) +
            '\n\nThis is not a warning. A month the engine cannot price is a month with no '
            'return, not a month with a 0% return. Fix the window (coverage guard), the '
            'universe, or the strategy — do not relax the invariant.')


def run_ledger(target_weights, store, exec_cfg, sleeves=None, label='strategy'):
    """Turn a table of target weights into a traded, costed, cash-aware return series.

    Parameters
    ----------
    target_weights : DataFrame
        Indexed by DECISION date (real trading dates), columns are the tickers actually
        held — already LETF-translated for leveraged wraps.
    store : PriceStore
    exec_cfg : ExecutionConfig
    sleeves : dict or None
        Optional ``{'defensive_mask': DataFrame}`` aligned to `target_weights`, flagging the
        weight held DEFENSIVELY. Used only when margin follows the signal: defensive weight
        is held at 1x and never borrowed against.
    """
    warnings = []
    if target_weights.empty:
        empty = pd.Series(dtype=float)
        return LedgerResult(empty, empty, empty, empty, empty, empty, empty, empty,
                            pd.Series(dtype='datetime64[ns]'), warnings=['no decision dates'])

    if exec_cfg.convention == 'next_open' and not store.has_intraday:
        raise ValueError(
            "convention='next_open' needs opens, and this store has none "
            '(has_intraday=False). Say what you mean: pass a store with daily bars, or '
            "convention='signal_close' if you are deliberately reproducing the pre-audit "
            'engine.')

    prices = _execution_price_frame(store, exec_cfg.convention)
    decisions = pd.DatetimeIndex(target_weights.index)

    # --- map every decision date to its execution date ------------------------------- #
    exec_dates, kept = [], []
    for d in decisions:
        e = d if exec_cfg.convention == 'signal_close' else store.next_trading_day(d)
        if e is None:
            warnings.append(f'{d.date()}: no session after the decision — dropped. The last '
                            f'decision in a store can never be executed.')
            continue
        kept.append(d)
        exec_dates.append(e)
    decisions = pd.DatetimeIndex(kept)
    if len(decisions) < 2:
        empty = pd.Series(dtype=float)
        warnings.append('fewer than two executable decisions — nothing to measure')
        return LedgerResult(empty, empty, empty, empty, empty, empty, empty, empty,
                            pd.Series(exec_dates, index=decisions), warnings=warnings)
    exec_dates = pd.DatetimeIndex(exec_dates)

    tickers = [c for c in target_weights.columns if c in prices.columns]
    dropped = [c for c in target_weights.columns if c not in prices.columns]
    if dropped:
        held = target_weights[dropped].abs().sum().sum()
        if held > 0:
            warnings.append(f'tickers absent from the price store carried weight and were '
                            f'routed to cash: {sorted(dropped)}')

    cash = exec_cfg.cash_ticker
    if cash is None:
        warnings.append('cash_ticker=None — uninvested weight is held as a cash BALANCE and '
                        'earns 0%. That is a modelling choice, not a market fact; say so '
                        'wherever the result is quoted.')
    elif cash not in prices.columns:
        warnings.append(f'cash_ticker {cash!r} is absent from the price store — uninvested '
                        f'weight is held as a cash BALANCE at 0%, which understates the '
                        f'defensive sleeve.')
        cash = None

    px = prices.reindex(exec_dates)[tickers + ([cash] if cash and cash not in tickers else [])]

    if exec_cfg.strict_invariants:
        validate_targets(target_weights.loc[decisions], decisions, exec_dates, px, label)

    # --- leverage multiplier per (date, ticker) --------------------------------------- #
    L = float(exec_cfg.leverage)
    def_mask = None
    if sleeves is not None:
        def_mask = sleeves.get('defensive_mask')
    if def_mask is not None:
        def_mask = def_mask.reindex(index=decisions, columns=tickers).fillna(False)
    elif L != 1.0 and exec_cfg.leverage_follows_signal:
        # A configured behaviour must not be silently replaced by its opposite. Without a
        # mask the loop below applies L to every ticker — flat leverage — while the config
        # says signal-following, and the A/B this flag exists for would silently return no
        # difference. A warning rather than a raise: strict_invariants governs refusals
        # about WEIGHTS, and a research caller deliberately passing no sleeves is a
        # legitimate use — the same treatment cash_ticker=None gets above.
        warnings.append(
            "leverage_follows_signal=True but no {'defensive_mask': DataFrame} was supplied "
            "in `sleeves` — FLAT leverage was applied to every position. Pass "
            "sleeves={'defensive_mask': strat.defensive_mask(targets, scores)} or set "
            "leverage_follows_signal=False and mean it.")

    # --- walk the ledger --------------------------------------------------------------- #
    n = len(decisions)
    pos = pd.Series(0.0, index=px.columns)     # gross notional per ticker (incl. borrowed)
    debt = 0.0                                 # margin principal actually drawn
    equity = 1.0                               # net asset value
    # Uninvested weight when there is no cash TICKER to hold it in. Without this the residue
    # was simply never recorded: `target` omitted it, so `pos.sum()` omitted it, and it left
    # the portfolio entirely. On a dead-flat market a book that is 80% invested then reports
    # equity 1.0 -> 0.80 -> 0.64 -> 0.512 — not a one-off 20% loss but 20% compounded at
    # EVERY rebalance, while the warning above said "earns 0%". Production never reached it
    # (strict_invariants refuses a row that does not sum to 1, and the residue is nonzero
    # only for a ticker absent from the store), but "unreachable" is a property of today's
    # call sites, and the docstring's claim has to be true where the claim is made.
    idle_cash = 0.0

    idx_returns, net_ret, gross_ret = [], [], []
    turnover, n_trades, cost_paid, eff_lev, cash_w = [], [], [], [], []
    equity_curve = [1.0]
    initial_cost = 0.0
    prev_px = prev_exec = None
    equity_at_rebalance = 1.0                  # equity BEFORE the last rebalance's cost
    equity_after_cost = 1.0                    # ... and after it

    for i in range(n):
        d, e = decisions[i], exec_dates[i]
        row_px = px.loc[e]

        # 1. drift the book from the previous execution to this one, then settle interest
        if i > 0:
            ratio = (row_px / prev_px).replace([np.inf, -np.inf], np.nan)
            held = pos.abs() > 1e-12
            lost = pos.index[held & ratio.isna()]
            if len(lost):
                warnings.append(f'{e.date()}: no price for {sorted(lost)} — position carried '
                                f'flat over the period. Do not trust this period.')
            pos = pos * ratio.fillna(1.0)

            days = max((e - prev_exec).days, 0)
            interest = debt * exec_cfg.borrow_rate * days / 365.0
            # Idle cash does not drift — that is what 0% means — but it is still capital.
            equity_gross = float(pos.sum()) + idle_cash - debt   # before interest, after drift
            equity = equity_gross - interest
            debt += interest

            # gross = the levered asset return alone: no cost, no financing.
            gross_ret.append(equity_gross / equity_after_cost - 1.0)
            # net charges the rebalance cost paid at the START of this interval, which is
            # why the divisor is the pre-cost equity.
            net_ret.append(equity / equity_at_rebalance - 1.0)
            idx_returns.append(d)
            equity_curve.append(equity)

        # 2. rebalance at E — except on the final decision, whose holding period lies
        #    outside the sample. Trading into a position we never measure would charge a
        #    cost that no return ever pays for. (The pre-audit engine dropped the last
        #    allocation the same way, via alloc.shift(1); this makes it deliberate.)
        if i == n - 1:
            break

        w = target_weights.loc[d].reindex(px.columns).fillna(0.0)
        residual = max(0.0, 1.0 - float(w[tickers].sum()))

        mult = pd.Series(L, index=px.columns, dtype=float)
        if L != 1.0 and exec_cfg.leverage_follows_signal and def_mask is not None:
            # Borrow only against the offensive sleeve: defence is held at 1x, so the loan
            # is repaid as the signal de-risks instead of riding the drawdown levered.
            flags = def_mask.loc[d].reindex(px.columns).fillna(False).astype(bool)
            mult[flags.values] = 1.0

        # A DELIBERATE cash-ticker holding is a position, and follows the mode's rule above:
        # flat margin stays drawn through it (that is what "flat" means, and what makes the
        # A/B against signal-following meaningful), while signal-following has already
        # de-levered it via the defensive mask. Only the UNINVESTED residue below is
        # unlevered — nobody borrows to hold idle cash, in either mode.
        target = w * mult * equity
        target_idle = 0.0
        if residual > 0:
            if cash is not None:
                target[cash] = target.get(cash, 0.0) + residual * equity
            else:
                target_idle = residual * equity

        # A ticker with no price on E cannot be traded at all.
        untradeable = row_px.isna() & ((target - pos).abs() > 1e-12)
        if untradeable.any():
            warnings.append(f'{e.date()}: cannot execute {sorted(row_px.index[untradeable])} '
                            f'— no price on the execution date. Weight left where it was.')
            target = target.where(~untradeable, pos)

        traded = (target - pos).abs()
        cost = float(traded.sum()) * exec_cfg.cost_bps_per_side / 10_000.0
        if i == 0:
            initial_cost = cost

        # An idle cash BALANCE costs nothing to move into or out of, so it is not in
        # `traded` — but it is capital, so it is in exposure and in the cash weight.
        turnover.append(float(traded.sum()) / equity if equity > 0 else np.nan)
        n_trades.append(int((traded > 1e-9).sum()))
        cost_paid.append(cost)
        eff_lev.append((float(target.sum()) + target_idle) / equity if equity > 0 else np.nan)
        held_cash = (float(target.get(cash, 0.0)) if cash else 0.0) + target_idle
        cash_w.append(held_cash / equity if equity > 0 else 0.0)

        equity_at_rebalance = equity
        equity_after_cost = equity - cost
        # Paying the cost shrinks the book proportionally; the weights are unchanged.
        shrink = equity_after_cost / equity if equity > 0 else 1.0
        pos = target * shrink
        idle_cash = target_idle * shrink
        debt = max(float(pos.sum()) + idle_cash - equity_after_cost, 0.0)
        equity = equity_after_cost
        prev_px, prev_exec = row_px, e

    # 3. terminal liquidation — charged so the reported result is one a real investor could
    #    have realised, and reported separately so it can be backed out.
    terminal_cost = float(pos.abs().sum()) * exec_cfg.cost_bps_per_side / 10_000.0
    if exec_cfg.charge_terminal_liquidation and net_ret:
        base = equity_curve[-1]
        equity_curve[-1] = base - terminal_cost
        net_ret[-1] = (1.0 + net_ret[-1]) * (equity_curve[-1] / base) - 1.0

    ret = pd.Series(net_ret, index=pd.DatetimeIndex(idx_returns), dtype=float)
    gross = pd.Series(gross_ret, index=pd.DatetimeIndex(idx_returns), dtype=float)
    rebal_idx = decisions[:len(turnover)]
    return LedgerResult(
        returns=ret,
        gross_returns=gross,
        turnover=pd.Series(turnover, index=rebal_idx, dtype=float),
        n_trades=pd.Series(n_trades, index=rebal_idx, dtype=int),
        cost_paid=pd.Series(cost_paid, index=rebal_idx, dtype=float),
        effective_leverage=pd.Series(eff_lev, index=rebal_idx, dtype=float),
        cash_weight=pd.Series(cash_w, index=rebal_idx, dtype=float),
        equity=pd.Series(equity_curve, index=pd.DatetimeIndex([decisions[0]] + idx_returns),
                         dtype=float),
        exec_dates=pd.Series(exec_dates, index=decisions),
        first_date=ret.index[0] if len(ret) else None,
        last_date=ret.index[-1] if len(ret) else None,
        initial_deployment_cost=initial_cost,
        terminal_liquidation_cost=terminal_cost,
        warnings=warnings,
    )
