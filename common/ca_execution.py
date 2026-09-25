"""
Target weights in, Canadian orders out — two currencies, whole shares, nothing invented.

WHAT THIS DOES AND DOES NOT DECIDE
----------------------------------
IN:  `{signal_ticker: weight}` produced by HAA_G12, plus the cash, the positions already
     held, the last price of each execution ticker, and an FX rate.
OUT: a list of orders in Canadian tickers, the conversions needed to pay for them, the cash
     left over in each currency, and a rotation row per trade for the accountant.

It decides nothing about WHAT to hold. The weights arrive already decided and are copied
through: `assert_weights_unchanged` is the test that says so, and it compares against the
allocation row rather than against anything computed here.

THE RATE IS AN ARGUMENT, NEVER A FETCH
--------------------------------------
Nothing in this module opens a socket. `FxRate` is constructed by the caller — from the Bank
of Canada, from a manual override, from a test — and carries its own source and date so the
journal can record WHICH rate produced an order. This is the same boundary
`tools/vendor_crosscheck.py` draws: network code stays out of the code that decides, so the
decision can be tested without one and cannot fail because a website is down.

There is deliberately no default rate. `FxRate` refuses a non-finite or non-positive value,
and `build_orders` refuses a `None`. A silently-defaulted exchange rate misprices every
order in the book by the same factor, in the same direction, with no symptom.

LEVERAGE STAYS AT 1x (EXEC-001, reaffirmed 2026-09-17)
------------------------------------------------------
`leverage > 1.0` is REFUSED, with the same reasoning `main.compute_live_signals` prints: the
code cannot tell whether a stated balance is equity or buying power, and margin is not
available in every account. The parameter exists, is validated, and says no — which is
different from not having it, because a config that asks for 1.3x gets an error instead of
silently receiving 1x orders. Values below 1.0 are allowed: deliberately holding cash back
is a choice the code CAN honour, since it needs no information it does not have.

THE COST OF A CONVERSION — A UNIT AMBIGUITY, RESOLVED
-----------------------------------------------------
The brief specified `conversion_cost_pct: 0.002` and annotated it "IBKR : ~0,002 % + min 2
USD". Those disagree by a factor of 100: as a fraction 0.002 is 20 bps, and 0.002% is 0.2 bp.
Rather than pick one, the cost is expressed here in BASIS POINTS with an explicit floor —
`fx_cost_bps` and `fx_cost_min_usd` — which is the idiom `ledger.ExecutionConfig` already
uses for trading cost and which cannot be read two ways. The defaults are IBKR's published
schedule as the annotation describes it (0.2 bp, USD 2 minimum); they are a default, not a
measurement, and they are the caller's to override.

The commission is charged in USD, which is what IBKR does, and is deducted from the USD side
of the conversion whichever way it runs.

SELLS SETTLE BEFORE BUYS ARE SIZED
-----------------------------------
Orders are produced in the sequence they can actually be paid for: liquidations first,
crediting cash in the currency of the class sold, then one conversion if a currency is short,
then purchases sized against what is really there. Sizing every leg against the opening
balance is how a plan arrives at the broker and is rejected for funds.
"""

import math
from dataclasses import dataclass

from common import ca_mapping as cam


class CAExecutionError(ValueError):
    """An input this module refuses to turn into orders.

    Every refusal below names what was wrong and what it would take to fix it. None of them
    fall back to a default: the whole point of the guard is that a wrong order is worse than
    no order, and a defaulted input produces a wrong order with no symptom.
    """


@dataclass(frozen=True)
class FxRate:
    """One USD/CAD rate, with the provenance that has to appear in the journal.

    `rate` is CAD per USD — the Bank of Canada's `FXUSDCAD` series convention, so 1 USD buys
    `rate` CAD. Named `usdcad` in the config for the same reason.
    """

    usdcad: float
    source: str
    as_of: str

    def __post_init__(self):
        r = self.usdcad
        if r is None or not isinstance(r, (int, float)) or isinstance(r, bool):
            raise CAExecutionError(
                f'the USD/CAD rate must be a number, got {r!r}. There is no default: an '
                f'assumed rate misprices every line in the book by the same factor, in the '
                f'same direction, and shows no symptom.')
        if not math.isfinite(r) or r <= 0:
            raise CAExecutionError(f'the USD/CAD rate must be finite and positive, got {r!r}.')
        if not self.source or not self.as_of:
            raise CAExecutionError(
                'an FX rate must carry its source and its date; the journal records which '
                'rate produced an order, and "unknown" is not a rate anybody can check.')


@dataclass(frozen=True)
class CAExecutionConfig:
    """Everything about HOW to trade, separated from WHAT to hold."""

    #: Multiple applied to the target weights. > 1.0 is refused; see EXEC-001 above.
    leverage: float = 1.0
    #: Route a purchase through the fund's USD unit class when one exists AND the USD cash
    #: already in the account covers it in full; otherwise buy the CAD class. Never converts
    #: CAD to reach a `.U` class. Worth reading `ca_mapping`'s USD columns before relying on
    #: it: ZTL.U spreads 0.42%, ZIC.U 0.36%, XEC.U 0.34%, and ZSML.U trades 161 days of 251.
    prefer_usd_units: bool = True
    #: FX commission in basis points of the converted notional. IBKR's published 0.2 bp.
    fx_cost_bps: float = 0.2
    #: Minimum FX commission, in USD. IBKR's published USD 2.00.
    fx_cost_min_usd: float = 2.0
    #: Skip any order below this fraction of NAV, as a percentage. 0 disables. HAA rotates
    #: monthly, so a dust leg pays a full spread for a rounding artefact.
    min_trade_pct: float = 0.0

    def __post_init__(self):
        lev = float(self.leverage)
        if lev > 1.0:
            raise CAExecutionError(
                f'leverage={lev:g} is refused (EXEC-001). Live sizing in this repository '
                f'stays at 1x of the stated balances and borrows nothing, because the code '
                f'cannot tell whether a balance is EQUITY or BUYING POWER at your broker, '
                f'and margin is not available in every account. Size the borrowed slice '
                f'yourself against common/margin_sizing.py and enter it as a larger balance '
                f'if that is what you mean.')
        if lev <= 0:
            raise CAExecutionError(f'leverage={lev:g} must be positive.')


def _entry_for(signal_ticker, mapping):
    entry = mapping.get(signal_ticker)
    if entry is None:
        raise CAExecutionError(
            f'{signal_ticker} has no Canadian execution image. The mapping covers '
            f'{sorted(mapping)}; a weight on anything else must not be silently dropped '
            f'(RULE 4).')
    return entry


def _classes(entry):
    """`{ticker: currency}` for the one or two unit classes of an entry."""
    out = {}
    if entry.exec_ticker_cad:
        out[entry.exec_ticker_cad] = 'CAD'
    if entry.exec_ticker_usd:
        out[entry.exec_ticker_usd] = 'USD'
    return out


def _price(ticker, prices):
    px = prices.get(ticker)
    if px is None or not isinstance(px, (int, float)) or isinstance(px, bool) \
            or not math.isfinite(px) or px <= 0:
        raise CAExecutionError(
            f'no usable price for {ticker} (got {px!r}). An order cannot be sized against a '
            f'missing price, and substituting a stale one is how a rebalance gets valued at '
            f'a number no market produced.')
    return float(px)


def _to_cad(amount, currency, fx):
    return amount * fx.usdcad if currency == 'USD' else amount


def _whole(qty):
    """An integral quantity as an `int`, so a journal never records 492 units as `492.0`."""
    return int(qty) if float(qty).is_integer() else qty


def _largest_sellable_usd(cash_usd, config):
    """The most USD that can be converted AWAY when the commission is also paid in USD.

    Solves ``n + max(n * r, m) <= cash_usd`` for the largest `n`, where `r` is the commission
    rate and `m` its floor. Two regimes, and the answer is whichever is both feasible and
    larger: the floor binds on a small conversion (`n = cash - m`) and the rate binds on a
    large one (`n = cash / (1 + r)`).

    Written out rather than approximated because the naive version — convert the whole
    balance, then charge the fee — overdraws it by exactly the fee, silently, on every
    rotation that liquidates a `.U` holding to buy a CAD product.
    """
    r = config.fx_cost_bps / 10_000.0
    m = config.fx_cost_min_usd
    best = 0.0
    floor_regime = cash_usd - m
    if floor_regime > 0 and floor_regime * r <= m:
        best = max(best, floor_regime)
    rate_regime = cash_usd / (1.0 + r)
    if rate_regime > 0 and rate_regime * r >= m:
        best = max(best, rate_regime)
    return best


def build_orders(weights, cash, positions, prices, fx, config=None, signal_date=None,
                 mapping=None):
    """Translate target weights into Canadian orders across two currencies.

    * `weights`   — `{signal_ticker: weight}`. Copied through untouched; this function never
      reorders, rescales or drops one. A ticker with no image RAISES rather than vanishing.
    * `cash`      — `{'CAD': float, 'USD': float}`; missing keys are 0.
    * `positions` — `[{'ticker': <execution ticker>, 'qty': float, 'currency': 'CAD'|'USD'}]`.
      A position in a `.U` class and one in the CAD class of the same fund are ONE sleeve and
      are netted; they are two names for the same exposure.
    * `prices`    — `{execution_ticker: price}`, each in its own class's trading currency.
    * `fx`        — an `FxRate`. Required; there is no default.
    * `signal_date` — stamped on every rotation row as the authority for the trade.

    Returns a dict with `nav_cad_before`, `nav_cad_after`, `orders`, `fx_orders`,
    `residual_cash`, `rotation_rows`, `flags` and `fx`.
    """
    mapping = cam.CA_MAPPING if mapping is None else mapping
    config = CAExecutionConfig() if config is None else config
    if not isinstance(fx, FxRate):
        raise CAExecutionError(
            'an FxRate is required. Passing None to mean "look it up later" is how a rate of '
            '1.0 ends up in a live order; construct one from the Bank of Canada, from your '
            'broker, or by hand, and it will be journalled with the orders.')

    cash_cad = float((cash or {}).get('CAD', 0.0) or 0.0)
    cash_usd = float((cash or {}).get('USD', 0.0) or 0.0)
    positions = list(positions or ())
    flags = []

    # ---- validate the inputs before valuing anything ---------------------------------- #
    held_tickers = [p['ticker'] for p in positions]
    cam.assert_no_us_tickers(held_tickers, mapping)

    live_weights = {k: float(v) for k, v in (weights or {}).items() if float(v) > 1e-9}
    for signal_ticker in live_weights:
        _entry_for(signal_ticker, mapping)
    total_weight = sum(live_weights.values())
    if live_weights and abs(total_weight - 1.0) > 1e-6:
        raise CAExecutionError(
            f'target weights sum to {total_weight:.6f}, not 1. A row that does not sum to one '
            f'is the defect `ledger.validate_row` exists to catch: summing with skipna made a '
            f'month invested in a nonexistent product report +0.00%.')

    # Which execution ticker each held position belongs to, and which sleeve.
    sleeve_of_ticker = {}
    for signal_ticker, entry in mapping.items():
        for ticker in _classes(entry):
            sleeve_of_ticker[ticker] = signal_ticker

    # ---- value the book --------------------------------------------------------------- #
    held_qty = {}
    for p in positions:
        held_qty[p['ticker']] = held_qty.get(p['ticker'], 0.0) + float(p['qty'])

    position_value_cad = {}
    for ticker, qty in held_qty.items():
        currency = _classes(mapping[sleeve_of_ticker[ticker]])[ticker]
        position_value_cad[ticker] = _to_cad(qty * _price(ticker, prices), currency, fx)

    nav_before = cash_cad + _to_cad(cash_usd, 'USD', fx) + sum(position_value_cad.values())
    if nav_before <= 0:
        raise CAExecutionError(
            f'net asset value is {nav_before:,.2f} CAD. There is nothing to allocate, and a '
            f'zero or negative book is a data problem rather than a portfolio.')

    # Current exposure per SLEEVE, netting the two unit classes of one fund.
    current_cad = {}
    for ticker, value in position_value_cad.items():
        sleeve = sleeve_of_ticker[ticker]
        current_cad[sleeve] = current_cad.get(sleeve, 0.0) + value

    # ---- targets ---------------------------------------------------------------------- #
    lev = float(config.leverage)
    target_cad = {s: nav_before * w * lev for s, w in live_weights.items()}
    min_trade_cad = nav_before * (config.min_trade_pct / 100.0)

    orders, rotation_rows, fx_orders = [], [], []

    def _emit(ticker, sleeve, side, qty, currency, price):
        value = qty * price
        entry = mapping[sleeve]
        use_usd = currency == 'USD'
        order_flags = cam.liquidity_flags(entry, shares=qty, use_usd_class=use_usd)
        flags.extend(order_flags)
        orders.append({'signal_ticker': sleeve, 'ticker': ticker, 'exchange': entry.exchange,
                       'side': side, 'qty': qty, 'currency': currency, 'price': price,
                       'value': value, 'value_cad': _to_cad(value, currency, fx),
                       'flags': order_flags})
        rotation_rows.append({
            'signal_date': str(signal_date) if signal_date is not None else None,
            'signal_ticker': sleeve, 'exec_ticker': ticker, 'exchange': entry.exchange,
            'side': side, 'qty': qty, 'currency': currency, 'price': price, 'value': value,
            'reason': f'HAA_G12 signal of {signal_date}' if signal_date is not None
                      else 'HAA_G12 signal (date not supplied)'})

    # ---- 1. SELLS, which credit the cash the buys are sized against -------------------- #
    # Every class of every sleeve, including sleeves that have left the target entirely.
    for ticker in sorted(held_qty):
        sleeve = sleeve_of_ticker[ticker]
        surplus_cad = current_cad.get(sleeve, 0.0) - target_cad.get(sleeve, 0.0)
        if surplus_cad <= max(min_trade_cad, 0.01):
            continue
        currency = _classes(mapping[sleeve])[ticker]
        price = _price(ticker, prices)
        # Sell no more of THIS class than it holds; a sleeve split across two classes
        # liquidates the one being iterated and the loop reaches the other on its own.
        want = surplus_cad / fx.usdcad if currency == 'USD' else surplus_cad
        if sleeve not in target_cad:
            # Leaving the target: every unit goes. Dividing the position's own value back by
            # its price is 460.99999... for 461 units of ZCOM at 38.49, and the floor of that
            # left one unit behind — found by the demonstration in tools/ca_orders.py.
            qty = held_qty[ticker]
        else:
            qty = min(held_qty[ticker], math.floor(want / price + 1e-9))
        qty = _whole(qty)
        if qty <= 0:
            continue
        _emit(ticker, sleeve, 'SELL', qty, currency, price)
        proceeds = qty * price
        if currency == 'USD':
            cash_usd += proceeds
        else:
            cash_cad += proceeds
        current_cad[sleeve] = current_cad.get(sleeve, 0.0) - _to_cad(proceeds, currency, fx)

    # ---- 2. Decide the class each purchase will use, then the currency shortfall ------- #
    # A USD class is used only when USD ALREADY IN THE ACCOUNT covers the whole purchase.
    # Until 2026-09-23 `prefer_usd_units` routed to the `.U` class whatever the cash, so a
    # CAD-only book with the default setting converted most of itself to USD every month to
    # buy the thinner, wider-spread class — the opposite of what the setting's own comment
    # promised. Only a sleeve with no CAD class (BIL -> UBIL.U) may cause a CAD->USD
    # conversion, and it claims the USD first because it has no alternative.
    wanted = []
    for sleeve, target in sorted(target_cad.items(), key=lambda kv: -kv[1]):
        shortfall_cad = target - current_cad.get(sleeve, 0.0)
        if shortfall_cad > max(min_trade_cad, 0.01):
            wanted.append((sleeve, shortfall_cad, mapping[sleeve]))
    usd_free = cash_usd - sum(s / fx.usdcad for _, s, e in wanted if e.exec_ticker_cad is None)
    buys = []
    for sleeve, shortfall_cad, entry in wanted:
        usd_ticker = entry.exec_ticker_usd
        cad_ticker = entry.exec_ticker_cad
        if cad_ticker is None:
            ticker, currency = usd_ticker, 'USD'      # BIL -> UBIL.U, USD-only by construction
        elif (usd_ticker is not None and config.prefer_usd_units
              and usd_free >= shortfall_cad / fx.usdcad - 1e-9):
            ticker, currency = usd_ticker, 'USD'
            usd_free -= shortfall_cad / fx.usdcad
        else:
            ticker, currency = cad_ticker, 'CAD'
        buys.append({'sleeve': sleeve, 'ticker': ticker, 'currency': currency,
                     'shortfall_cad': shortfall_cad})

    need_usd = sum(b['shortfall_cad'] for b in buys if b['currency'] == 'USD') / fx.usdcad
    need_cad = sum(b['shortfall_cad'] for b in buys if b['currency'] == 'CAD')

    # ---- 3. One conversion, in whichever direction is short --------------------------- #
    def _convert(direction, notional_usd):
        """`direction` is 'USD->CAD' or 'CAD->USD'. Commission is charged in USD, IBKR-style,
        and deducted from the USD side whichever way the trade runs."""
        nonlocal cash_cad, cash_usd
        cost_usd = max(notional_usd * config.fx_cost_bps / 10_000.0, config.fx_cost_min_usd)
        if direction == 'USD->CAD':
            cash_usd -= notional_usd + cost_usd
            cash_cad += notional_usd * fx.usdcad
        else:
            cash_cad -= notional_usd * fx.usdcad
            cash_usd += notional_usd - cost_usd
        fx_orders.append({'direction': direction, 'usd': notional_usd,
                          'cad': notional_usd * fx.usdcad, 'rate': fx.usdcad,
                          'cost_usd': cost_usd, 'cost_cad': cost_usd * fx.usdcad,
                          'source': fx.source, 'as_of': fx.as_of,
                          'description': f'FX: convert {notional_usd:,.2f} USD -> '
                                         f'{notional_usd * fx.usdcad:,.2f} CAD at '
                                         f'{fx.usdcad:.4f}' if direction == 'USD->CAD'
                                         else f'FX: convert {notional_usd * fx.usdcad:,.2f} '
                                              f'CAD -> {notional_usd:,.2f} USD at '
                                              f'{fx.usdcad:.4f}'})
        rotation_rows.append({
            'signal_date': str(signal_date) if signal_date is not None else None,
            'signal_ticker': None, 'exec_ticker': f'FX {direction}', 'exchange': None,
            'side': 'CONVERT', 'qty': notional_usd, 'currency': 'USD',
            'price': fx.usdcad, 'value': notional_usd * fx.usdcad,
            'reason': f'funding for the HAA_G12 signal of {signal_date}'})

    def _cost_of(notional_usd):
        return max(notional_usd * config.fx_cost_bps / 10_000.0, config.fx_cost_min_usd)

    if need_usd > cash_usd + 1e-9 and cash_cad > 0:
        # CAD -> USD. The CAD side pays for the notional; the commission comes out of the
        # USD received, so a conversion smaller than the commission delivers nothing and is
        # not proposed at all.
        notional = min(need_usd - cash_usd, cash_cad / fx.usdcad)
        if notional > _cost_of(notional):
            _convert('CAD->USD', notional)
    elif need_cad > cash_cad + 1e-9 and cash_usd > 0:
        # USD -> CAD. Both the notional AND the commission leave the USD balance, so the
        # conversion has to be sized to leave the fee behind. Selling the whole USD balance
        # and charging the fee on top overdraws it — found by
        # `test_cash_is_never_driven_negative`, which is the reason a conservation test
        # covers several shapes rather than the one the author had in mind.
        notional = min((need_cad - cash_cad) / fx.usdcad,
                       _largest_sellable_usd(cash_usd, config))
        if notional > 0:
            _convert('USD->CAD', notional)

    # ---- 4. BUYS, sized against the cash that is actually there ----------------------- #
    for buy in buys:
        entry = mapping[buy['sleeve']]
        price = _price(buy['ticker'], prices)
        lot = max(1, int(entry.min_lot))
        if buy['currency'] == 'USD':
            budget = min(buy['shortfall_cad'] / fx.usdcad, cash_usd)
        else:
            budget = min(buy['shortfall_cad'], cash_cad)
        if budget <= 0:
            continue
        qty = math.floor(budget / price / lot) * lot
        if qty <= 0:
            continue
        _emit(buy['ticker'], buy['sleeve'], 'BUY', qty, buy['currency'], price)
        spent = qty * price
        if buy['currency'] == 'USD':
            cash_usd -= spent
        else:
            cash_cad -= spent
        current_cad[buy['sleeve']] = current_cad.get(buy['sleeve'], 0.0) \
            + _to_cad(spent, buy['currency'], fx)

    # ---- 5. What is left, and what could not be checked ------------------------------- #
    nav_after = cash_cad + _to_cad(cash_usd, 'USD', fx) + sum(current_cad.values())

    # A gap smaller than one unit of what the sleeve buys is whole-unit rounding: it is
    # summed into one note. A gap of a unit or more means cash that could not reach the
    # sleeve, and that is a warning per sleeve. Before 2026-09-23 both were the same warning,
    # so every line of an ordinary rotation read "NOT fully invested" and a real shortfall
    # could not be told apart from the noise.
    unit_cad = {}
    for buy in buys:
        px = prices.get(buy['ticker'])
        if isinstance(px, (int, float)) and px > 0:
            lot = max(1, int(mapping[buy['sleeve']].min_lot))
            unit_cad[buy['sleeve']] = _to_cad(px * lot, buy['currency'], fx)
    rounding_left = 0.0
    for sleeve, target in sorted(target_cad.items()):
        gap = target - current_cad.get(sleeve, 0.0)
        if gap <= max(min_trade_cad, 0.01) + 1e-6:
            continue
        if gap < unit_cad.get(sleeve, math.inf):
            rounding_left += gap
            continue
        flags.append(('warn',
                      f'{sleeve}: {gap:,.2f} CAD of a {target:,.2f} CAD target could not be '
                      f'deployed — more than one unit, so not rounding: the cash was in the '
                      f'wrong currency or too small to convert. The book is NOT fully '
                      f'invested at these orders.'))
    if rounding_left > 0.01:
        flags.append(('info',
                      f'whole-unit rounding leaves {rounding_left:,.2f} CAD of the targets '
                      f'uninvested ({rounding_left / nav_before:.2%} of the book); it stays '
                      f'as cash.'))

    return {
        'signal_date': str(signal_date) if signal_date is not None else None,
        'nav_cad_before': nav_before,
        'nav_cad_after': nav_after,
        'leverage': lev,
        'orders': orders,
        'fx_orders': fx_orders,
        'rotation_rows': rotation_rows,
        'residual_cash': {'CAD': cash_cad, 'USD': cash_usd},
        'target_cad': target_cad,
        'flags': flags,
        'fx': {'usdcad': fx.usdcad, 'source': fx.source, 'as_of': fx.as_of},
    }


def assert_weights_unchanged(weights, result, tol=1e-9):
    """The signal reached execution intact.

    Checks that every weight carried a target and that no sleeve was invented. It is the
    machine-readable form of the one rule the whole layer exists under: HAA_G12_CA changes
    the instruments, never the decision.
    """
    live = {k: float(v) for k, v in (weights or {}).items() if float(v) > 1e-9}
    targets = result['target_cad']
    if set(live) != set(targets):
        raise CAExecutionError(
            f'the execution layer changed the universe: signal {sorted(live)} against '
            f'targets {sorted(targets)}. It may change WHAT IS HELD, never WHAT WAS CHOSEN.')
    nav, lev = result['nav_cad_before'], result['leverage']
    for sleeve, weight in live.items():
        expected = nav * weight * lev
        if abs(targets[sleeve] - expected) > max(tol, abs(expected) * 1e-12):
            raise CAExecutionError(
                f'{sleeve}: target {targets[sleeve]:,.6f} does not equal weight {weight} x '
                f'NAV {nav:,.6f} x leverage {lev:g}.')


def rotation_csv(result):
    """The rotation registry, as CSV text. One row per trade, each naming its signal.

    Written to `logs/execution_ca/` by the driver, which `.gitignore` excludes as a
    DIRECTORY: these rows carry real quantities and real dollar values.
    """
    import csv
    import io

    fields = ['signal_date', 'signal_ticker', 'exec_ticker', 'exchange', 'side', 'qty',
              'currency', 'price', 'value', 'reason']
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in result['rotation_rows']:
        writer.writerow({k: row.get(k) for k in fields})
    return buf.getvalue()
