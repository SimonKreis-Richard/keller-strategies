"""
Savings projection: starting balances, monthly contributions, tax by Canadian account type and
inflation, carried forward on a HAIRCUT version of a strategy's own record.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
Everything else in this repository describes the past; this is the one place that looks
forward, so it says exactly what it assumes. The projected return is NOT the backtest's CAGR.
It is the Sharpe that survives both haircuts `recommend_leverage` already applies —
`SR_used = min(SR - E[max SR], SR - z*SE)`, from `margin_sizing.haircut_sharpe` — times the
strategy's measured volatility, over a cash rate held at the level the era realised:

    mean excess (annual, arithmetic) = SR_used * sigma
    future monthly return            = rf_m + mean_excess/12 + (excess_t - mean(excess))

The last term is the strategy's own monthly excess, DEMEANED: the shape of the record — its
volatility, skew, autocorrelation, the length of its drawdowns — is kept, and only its level
is replaced by the haircut one. Paths are drawn from that series by the same stationary
bootstrap the drawdown quantile uses, so volatility drag turns the arithmetic mean into a
geometric one without anybody choosing a number for it.

Two limits no amount of care here removes (KNOWN_GAPS.md §4): a bootstrap draws from the sample
it is given, so if these twenty-six years were generous every path is generous in its shape;
and the haircut corrects for OUR search across the registry, not for Keller's parameter search
before the data reached us. This is an assumption derived from the haircut backtest, stated in
full beside every number it produces. It is not a forecast.

TAX
---
Every rate is the owner's input. No Canada Revenue Agency rule, limit or bracket is written into
this module: they change every year, and whether they apply to someone is a question for that
person's advisor (the same boundary as `common/ca_mapping.py` RULE 3). What the module fixes is
only the MECHANISM of each account type:

    CELI            nothing taxed, in or out
    REER, CELIAPP   contribution deducted at `marginal_rate_now`. With `reinvest_refund` the
                    refund is invested in the account with the contribution — the gross-up a
                    reduced source deduction makes possible — so the deposit is
                    effort / (1 - rate_now). REER withdrawals are taxed at
                    `marginal_rate_retirement`; a qualifying CELIAPP withdrawal is not.
    NON_ENREGISTRE  each year's gain taxed at rate_now * capital_gains_inclusion, losses
                    carried forward. EVERY gain is treated as realised in the year it is made:
                    these strategies turn over six to twelve times a year, so deferral is the
                    exception, and assuming it would flatter the taxable account.

`monthly_contribution` is always the saver's OUT-OF-POCKET effort, after tax. That is the only
definition under which a CELI and a REER can be compared at all.

CURRENCY
--------
Currency-neutral, by the owner's decision (2026-09-23). Amounts are entered in one declared
currency, `PROJECTION.currency`, and projected in it; the other currency is a DISPLAY, at
today's USD/CAD held constant (`tools/fx_rate.py` — manual, Bank of Canada or Yahoo, and no
default: with no rate the second currency is simply not shown). No exchange-rate path is
modelled.

Pure: no print, no network, no file. `tools/projection.py` and the dashboard print.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import margin_sizing as ms
from common import metrics as metrics_mod

#: The four mechanisms. The names are the Canadian ones because the rules are Canadian.
ACCOUNT_KINDS = ('CELI', 'REER', 'CELIAPP', 'NON_ENREGISTRE')
KIND_ALIASES = {'TFSA': 'CELI', 'RRSP': 'REER', 'FHSA': 'CELIAPP', 'TAXABLE': 'NON_ENREGISTRE',
                'NON-ENREGISTRE': 'NON_ENREGISTRE', 'NONENREGISTRE': 'NON_ENREGISTRE'}
DEDUCTIBLE = ('REER', 'CELIAPP')

#: Resampling defaults, the same as the drawdown quantile's (`MarginPolicy`): not knobs. A
#: projection whose spread could be tuned would be tuned.
N_PATHS = 2000
BLOCK_MONTHS = 12
SHARPE_Z = 1.0
CHECKPOINT_YEARS = (5, 10, 20)
QUANTILES = (0.10, 0.50, 0.90)
CURRENCIES = ('CAD', 'USD')


class ConfigError(ValueError):
    """The PROJECTION block cannot be read as written."""


class ProjectionError(ValueError):
    """A projection that would have to guess. Refused rather than approximated."""


@dataclass(frozen=True)
class AccountPlan:
    name: str
    kind: str
    start_balance: float
    monthly_contribution: float = 0.0     # out-of-pocket effort, after tax
    contribution_growth: float = 0.0      # per year, applied to the effort
    annual_cap: float = None              # on DEPOSITS, per projection year; None = none
    reinvest_refund: bool = True          # REER / CELIAPP only


@dataclass(frozen=True)
class TaxAssumptions:
    marginal_rate_now: float
    marginal_rate_retirement: float
    capital_gains_inclusion: float = 0.5


@dataclass(frozen=True)
class ReturnAssumption:
    """The projected return of one entry, and every number it was derived from."""

    name: str
    selected: bool                 # False for a benchmark: no multiple-testing term
    haircut: ms.SharpeHaircut
    vol: float                     # annualised, measured
    rf_annual: float               # the era's realised cash rate, held flat
    mean_excess_observed: float    # annual, arithmetic: SR_obs * vol
    mean_excess_used: float        # annual, arithmetic: SR_used * vol
    first: object
    last: object
    n_months: int
    population: tuple              # (n_trials, trial_sharpe_sd, where it came from)

    @property
    def arithmetic_annual(self):
        return self.rf_annual + self.mean_excess_used


@dataclass
class Schedule:
    """One account's contributions, which depend on nothing random."""

    deposits: np.ndarray           # into the account, end of each month
    effort: np.ndarray             # out of the saver's pocket, same months
    refunds_kept: np.ndarray       # refunds NOT reinvested, received as cash
    capped: dict = field(default_factory=dict)    # {projection year: effort above the cap}


@dataclass
class Projection:
    """What `project` returns. Values are AFTER-TAX liquidation values, in nominal dollars."""

    horizon_months: int
    inflation: float
    tax: TaxAssumptions
    accounts: tuple
    schedules: dict                # {account name: Schedule}
    invested: np.ndarray           # (H+1,) after-tax starting value + cumulative effort
    returns: dict                  # {entry name: ReturnAssumption}
    values: dict                   # {entry name: (n_paths, H+1)}, plus 'savings account'
    drawdown_p95: dict             # {entry name: P95 max drawdown of the return path}
    population_note: str
    n_paths: int
    block_months: int
    seed: int
    currency: str = 'CAD'          # what every amount above is in
    fx: object = None              # USD/CAD with `usdcad`, `source`, `as_of`; None = no rate
    fx_note: str = ''              # why there is no rate, when there is none

    def deflator(self):
        t = np.arange(self.horizon_months + 1) / 12.0
        return (1.0 + self.inflation) ** t

    @property
    def other_currency(self):
        return 'USD' if self.currency == 'CAD' else 'CAD'

    def to_other(self, amount):
        """`amount` in the other currency at today's rate, or None when there is no rate."""
        if self.fx is None:
            return None
        rate = float(self.fx.usdcad)                   # CAD per USD
        return amount / rate if self.currency == 'CAD' else amount * rate


# --------------------------------------------------------------------------------------- #
#  the return assumption
# --------------------------------------------------------------------------------------- #

def return_assumption(entry, n_trials, trial_sharpe_sd, population_source='',
                      sharpe_z=SHARPE_Z):
    """The haircut return of one `metrics_data` entry. Refuses when no haircut exists.

    A benchmark was never chosen from a list, so it takes only the estimation haircut. Every
    other role — strategy, control, exploratory — does take the selection haircut: a variant is
    a trial whether or not the selection statistics count it, and projecting it without the
    haircut would make the least-examined entries the most flattering to project.
    """
    returns = pd.Series(entry['returns']).dropna()
    if len(returns) < 24:
        raise ProjectionError(f"{entry['name']}: {len(returns)} months of record; a projection "
                              'needs at least 24 to measure what it carries forward')
    selected = entry.get('role', 'strategy') != 'benchmark'
    cut = ms.haircut_sharpe(entry['sharpe'], returns, n_trials, trial_sharpe_sd, sharpe_z,
                            selected=selected)
    if math.isnan(cut.used):
        raise ProjectionError(f"{entry['name']}: no haircut Sharpe could be computed "
                              f"({'; '.join(cut.notes) or 'no reason recorded'}). Projecting "
                              'the unhaircut record is exactly what this module exists to refuse.')
    if selected and math.isnan(cut.deflated):
        raise ProjectionError(f"{entry['name']}: {' '.join(cut.assumptions)} Refusing to "
                              'project a selected entry without its selection haircut.')
    vol = float(entry['vol'])
    return ReturnAssumption(
        name=entry['name'], selected=selected, haircut=cut, vol=vol,
        rf_annual=float(entry['rf_annual']),
        mean_excess_observed=cut.observed * vol, mean_excess_used=cut.used * vol,
        first=returns.index[0], last=returns.index[-1], n_months=len(returns),
        population=(n_trials, trial_sharpe_sd, population_source))


def recentre(returns, rf_monthly, target_excess_annual, future_rf_annual):
    """The record's monthly excess, demeaned, re-levelled to the haircut, over a flat rf.

    `rf_monthly` is the realised cash series of the same months — the one the Sharpe was
    netted against. The future cash month is the GEOMETRIC monthly equivalent of
    `future_rf_annual`, so a cash account compounds to exactly that rate.
    """
    r = pd.Series(returns).dropna()
    rf = metrics_mod._as_monthly_rf(rf_monthly, r.index)
    excess = r - rf
    rf_m = (1.0 + float(future_rf_annual)) ** (1.0 / 12.0) - 1.0
    return rf_m + float(target_excess_annual) / 12.0 + (excess - excess.mean())


def future_series(entry, assumption):
    return recentre(entry['returns'], entry['rf_series'], assumption.mean_excess_used,
                    assumption.rf_annual)


def bootstrap_paths(series_by_name, horizon_months, n_paths=N_PATHS,
                    block_months=BLOCK_MONTHS, seed=None):
    """{name: (n_paths, horizon)} drawn with ONE set of indices over the months all share.

    One set, as `robustness.rank_bootstrap` does, so the strategy and its comparator live
    through the same alternative history: a path where the strategy is spared a crash its
    benchmark suffers would compare two different worlds.
    """
    if seed is None:
        raise ProjectionError('seed is required: an unseeded projection is not reproducible')
    frame = pd.concat(series_by_name, axis=1, join='inner').dropna()
    if len(frame) < 24:
        raise ProjectionError(f'the entries share {len(frame)} months; at least 24 are needed')
    rng = np.random.default_rng(int(seed))
    idx = ms.stationary_bootstrap_indices(len(frame), int(horizon_months), block_months,
                                          int(n_paths), rng)
    values = frame.to_numpy(dtype=float)
    return {name: values[:, j][idx] for j, name in enumerate(frame.columns)}


def drawdown_quantile(paths, quantile=0.95):
    """Quantile of the maximum-drawdown SEVERITY of the return paths. Returns NEGATIVE."""
    wealth = np.cumprod(1.0 + paths, axis=1)
    wealth = np.concatenate([np.ones((wealth.shape[0], 1)), wealth], axis=1)
    worst = (wealth / np.maximum.accumulate(wealth, axis=1) - 1.0).min(axis=1)
    return -float(np.quantile(-worst, quantile))


# --------------------------------------------------------------------------------------- #
#  contributions and tax
# --------------------------------------------------------------------------------------- #

def schedule(account, tax, horizon_months):
    """Deposits and out-of-pocket effort, month by month. Deterministic."""
    h = int(horizon_months)
    deposits, effort, kept = np.zeros(h), np.zeros(h), np.zeros(h)
    capped = {}
    rate = float(tax.marginal_rate_now)
    deductible = account.kind in DEDUCTIBLE
    gross_up = deductible and account.reinvest_refund
    used = 0.0
    for t in range(h):
        year = t // 12
        if t % 12 == 0:
            used = 0.0
        want_effort = account.monthly_contribution * (1.0 + account.contribution_growth) ** year
        want = want_effort / (1.0 - rate) if gross_up else want_effort
        dep = want
        if account.annual_cap is not None:
            dep = min(want, max(account.annual_cap - used, 0.0))
        used += dep
        paid = dep * (1.0 - rate) if gross_up else dep
        if want - dep > 1e-9:
            capped[year + 1] = capped.get(year + 1, 0.0) + (want_effort - paid)
        deposits[t], effort[t] = dep, paid
        if deductible and not account.reinvest_refund:
            kept[t] = dep * rate
    return Schedule(deposits, effort, kept, capped)


def withdrawal_rate(account, tax):
    return float(tax.marginal_rate_retirement) if account.kind == 'REER' else 0.0


def simulate(paths, account, sched, tax, income_inclusion=None):
    """After-tax liquidation value of one account along each path, (n_paths, H+1).

    `paths` is (n_paths, H) monthly returns. Contributions land at the END of each month, the
    convention of the ordinary annuity formula the anchor test checks against. A taxable
    account is taxed at each projection year's end and at the horizon, on the year's gain net
    of the year's deposits, with losses carried forward. `income_inclusion` overrides the
    capital-gains inclusion — interest on a savings account is taxed in full.
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    n, h = paths.shape
    inclusion = (tax.capital_gains_inclusion if income_inclusion is None
                 else float(income_inclusion))
    taxable = account.kind == 'NON_ENREGISTRE'
    keep = 1.0 - withdrawal_rate(account, tax)
    bal = np.full(n, float(account.start_balance))
    out = np.empty((n, h + 1))
    out[:, 0] = bal * keep
    year_start, year_dep, carry = bal.copy(), 0.0, np.zeros(n)
    for t in range(h):
        bal = bal * (1.0 + paths[:, t]) + sched.deposits[t]
        year_dep += sched.deposits[t]
        if taxable and (t % 12 == 11 or t == h - 1):
            net = (bal - year_start - year_dep) - carry
            carry = np.maximum(-net, 0.0)
            bal = bal - np.maximum(net, 0.0) * tax.marginal_rate_now * inclusion
            year_start, year_dep = bal.copy(), 0.0
        out[:, t + 1] = bal * keep
    return out


# --------------------------------------------------------------------------------------- #
#  the whole projection
# --------------------------------------------------------------------------------------- #

def project(entries, accounts, tax, horizon_years, inflation, n_trials, trial_sharpe_sd,
            population_note='', n_paths=N_PATHS, block_months=BLOCK_MONTHS, seed=None,
            currency='CAD', fx=None, fx_note=''):
    """Project every entry in `entries` (metrics_data rows) over the same accounts and paths.

    A savings account at the era's cash rate is always added: it is the line a bank's own
    projection draws, and the only one here with no model in it.
    """
    if not accounts:
        raise ProjectionError('no account to project')
    horizon = int(round(float(horizon_years) * 12))
    if horizon < 12:
        raise ProjectionError('the horizon must be at least one year')
    assumptions = {e['name']: return_assumption(e, n_trials, trial_sharpe_sd,
                                                population_note) for e in entries}
    series = {e['name']: future_series(e, assumptions[e['name']]) for e in entries}
    paths = bootstrap_paths(series, horizon, n_paths, block_months, seed)

    schedules = {a.name: schedule(a, tax, horizon) for a in accounts}
    start = sum(a.start_balance * (1.0 - withdrawal_rate(a, tax)) for a in accounts)
    effort = sum(schedules[a.name].effort for a in accounts)
    invested = np.concatenate([[start], start + np.cumsum(effort)])

    values = {}
    for name, p in paths.items():
        values[name] = sum(simulate(p, a, schedules[a.name], tax) for a in accounts)
    rf = next(iter(assumptions.values())).rf_annual
    cash = np.full((1, horizon), (1.0 + rf) ** (1.0 / 12.0) - 1.0)
    values['savings account'] = sum(simulate(cash, a, schedules[a.name], tax,
                                             income_inclusion=1.0) for a in accounts)
    return Projection(
        horizon_months=horizon, inflation=float(inflation), tax=tax, accounts=tuple(accounts),
        schedules=schedules, invested=invested, returns=assumptions, values=values,
        drawdown_p95={n: drawdown_quantile(p) for n, p in paths.items()},
        population_note=population_note, n_paths=int(n_paths), block_months=block_months,
        seed=int(seed), currency=currency, fx=fx, fx_note=fx_note)


def checkpoints(horizon_months):
    years = [y for y in CHECKPOINT_YEARS if y * 12 < horizon_months]
    return [y * 12 for y in years] + [horizon_months]


def summary(projection, name, month):
    """P10/P50/P90 of the after-tax value at `month`, nominal and real, and P(below invested)."""
    v = projection.values[name][:, month]
    q = np.quantile(v, QUANTILES) if v.size > 1 else np.repeat(v, len(QUANTILES))
    real = q / projection.deflator()[month]
    return {'month': month, 'nominal': tuple(float(x) for x in q),
            'real': tuple(float(x) for x in real),
            'other': tuple(projection.to_other(float(x)) for x in q),
            'invested': float(projection.invested[month]),
            'p_below_invested': float(np.mean(v < projection.invested[month]))}


# --------------------------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------------------------- #

_TOP_KEYS = {'strategy', 'comparator', 'horizon_years', 'inflation', 'currency', 'usdcad',
             'tax', 'accounts'}
_TAX_KEYS = {'marginal_rate_now', 'marginal_rate_retirement', 'capital_gains_inclusion'}
_ACCOUNT_KEYS = {'kind', 'monthly_contribution', 'contribution_growth', 'annual_cap',
                 'reinvest_refund', 'start_balance'}


def _number(value, where, low=None, high=None, high_open=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        raise ConfigError(f'{where} must be a number, got {value!r}.')
    value = float(value)
    if low is not None and value < low:
        raise ConfigError(f'{where} must be at least {low:g}, got {value:g}.')
    if high is not None and (value >= high if high_open else value > high):
        raise ConfigError(f'{where} must be {"below" if high_open else "at most"} {high:g}, '
                          f'got {value:g}.')
    return value


def _unknown(block, allowed, where):
    extra = sorted(k for k in block if k not in allowed and not str(k).startswith('_'))
    if extra:
        raise ConfigError(f'{where}: unknown key(s) {", ".join(extra)}. Allowed: '
                          f'{", ".join(sorted(allowed))}.')


def kind_of(value, where):
    if value is None or value == '':
        raise ConfigError(f'{where}: no kind chosen. Choose one of {", ".join(ACCOUNT_KINDS)} '
                          '— it is the account\'s tax treatment, and nothing guesses it.')
    key = str(value).strip().upper().replace(' ', '_')
    key = KIND_ALIASES.get(key, key)
    if key not in ACCOUNT_KINDS:
        raise ConfigError(f'{where}: kind {value!r} is not one of {", ".join(ACCOUNT_KINDS)} '
                          f'(or {", ".join(sorted(KIND_ALIASES))}).')
    return key


def parse_config(user_cfg):
    """(settings, accounts, tax) from a user-config dict. Strict: a guess is a refusal.

    Starting balances come from `BROKER_ACCOUNTS`, matched by `account_name`, so the
    projection starts where the live book stands. Every broker account must be given a kind —
    a balance whose tax treatment is unknown cannot be projected after tax — and an account
    the broker list lacks must carry its own `start_balance`.
    """
    block = user_cfg.get('PROJECTION')
    if not isinstance(block, dict):
        raise ConfigError('no PROJECTION block in the user config. Copy the one in '
                          'user_config.example.json and enter your own values.')
    _unknown(block, _TOP_KEYS, 'PROJECTION')
    strategy = block.get('strategy')
    if not isinstance(strategy, str) or not strategy:
        raise ConfigError('PROJECTION.strategy must name a registry entry.')
    comparator = block.get('comparator')
    if comparator is not None and (not isinstance(comparator, str) or comparator == strategy):
        raise ConfigError('PROJECTION.comparator must be null or another entry\'s name.')
    horizon = _number(block.get('horizon_years'), 'PROJECTION.horizon_years', 1, 60)
    inflation = _number(block.get('inflation'), 'PROJECTION.inflation', 0.0, 0.20)
    currency = block.get('currency')
    if currency not in CURRENCIES:
        raise ConfigError(f'PROJECTION.currency must be one of {", ".join(CURRENCIES)}: the '
                          f'currency the balances and contributions are entered in, got '
                          f'{currency!r}. Nothing guesses it.')
    usdcad = block.get('usdcad')
    if usdcad is not None:
        usdcad = _number(usdcad, 'PROJECTION.usdcad', 0.0)
        if usdcad == 0:
            raise ConfigError('PROJECTION.usdcad must be positive, or null to look it up.')

    t = block.get('tax')
    if not isinstance(t, dict):
        raise ConfigError('PROJECTION.tax must be an object.')
    _unknown(t, _TAX_KEYS, 'PROJECTION.tax')
    tax = TaxAssumptions(
        marginal_rate_now=_number(t.get('marginal_rate_now'),
                                  'PROJECTION.tax.marginal_rate_now', 0.0, 1.0, True),
        marginal_rate_retirement=_number(t.get('marginal_rate_retirement'),
                                         'PROJECTION.tax.marginal_rate_retirement', 0.0, 1.0,
                                         True),
        capital_gains_inclusion=_number(t.get('capital_gains_inclusion', 0.5),
                                        'PROJECTION.tax.capital_gains_inclusion', 0.0, 1.0))

    broker = {}
    for acct in user_cfg.get('BROKER_ACCOUNTS') or []:
        broker[acct.get('account_name')] = acct.get('account_balance')
    specs = block.get('accounts')
    if not isinstance(specs, dict) or not specs:
        raise ConfigError('PROJECTION.accounts must map at least one account name to its kind.')
    missing = sorted(n for n in broker if n not in specs)
    if missing:
        raise ConfigError(f'BROKER_ACCOUNTS holds {", ".join(map(str, missing))}, which '
                          'PROJECTION.accounts does not give a kind. A balance whose tax '
                          'treatment is unknown cannot be projected after tax.')
    accounts = []
    for name, spec in specs.items():
        where = f'PROJECTION.accounts.{name}'
        if not isinstance(spec, dict):
            raise ConfigError(f'{where} must be an object.')
        _unknown(spec, _ACCOUNT_KEYS, where)
        kind = kind_of(spec.get('kind'), where)
        if 'start_balance' in spec and name in broker:
            raise ConfigError(f'{where}: start_balance is given, and BROKER_ACCOUNTS also '
                              'holds this account. Remove one: two balances are one guess.')
        if 'start_balance' in spec:
            start = _number(spec['start_balance'], f'{where}.start_balance', 0.0)
        elif name in broker:
            start = _number(broker[name], f'BROKER_ACCOUNTS[{name}].account_balance', 0.0)
        else:
            raise ConfigError(f'{where}: not in BROKER_ACCOUNTS and no start_balance given.')
        cap = spec.get('annual_cap')
        if 'reinvest_refund' in spec:
            if kind not in DEDUCTIBLE:
                raise ConfigError(f'{where}: reinvest_refund only applies to a deductible '
                                  f'account ({", ".join(DEDUCTIBLE)}), not {kind}.')
            if not isinstance(spec['reinvest_refund'], bool):
                raise ConfigError(f'{where}.reinvest_refund must be true or false.')
        accounts.append(AccountPlan(
            name=str(name), kind=kind, start_balance=start,
            monthly_contribution=_number(spec.get('monthly_contribution', 0.0),
                                         f'{where}.monthly_contribution', 0.0),
            contribution_growth=_number(spec.get('contribution_growth', 0.0),
                                        f'{where}.contribution_growth', -0.5, 0.5),
            annual_cap=None if cap is None else _number(cap, f'{where}.annual_cap', 0.0),
            reinvest_refund=spec.get('reinvest_refund', True)))
    settings = {'strategy': strategy, 'comparator': comparator, 'horizon_years': horizon,
                'inflation': inflation, 'currency': currency, 'usdcad': usdcad}
    return settings, accounts, tax


# --------------------------------------------------------------------------------------- #
#  the report
# --------------------------------------------------------------------------------------- #

def _money(x):
    return f'{x:>13,.0f}'


def assumption_lines(p):
    """Printed BEFORE any projected number: what the numbers are made of."""
    lines = ['SAVINGS PROJECTION — an assumption derived from the haircut backtest, '
             'not a forecast', '']
    for a in p.returns.values():
        c = a.haircut
        lines.append(f'  {a.name}: {a.n_months} months of record, '
                     f'{a.first:%Y-%m}..{a.last:%Y-%m}, sigma {a.vol:.2%}')
        if a.selected:
            lines.append(f'    Sharpe {c.observed:.2f} observed -> {c.deflated:.2f} after the '
                         f'selection haircut, {c.lower_bound:.2f} at -1 standard error -> '
                         f'{c.used:.2f} used (the lower)')
        else:
            lines.append(f'    Sharpe {c.observed:.2f} observed -> {c.used:.2f} used '
                         '(benchmark: estimation haircut only, nobody selected it)')
        lines.append(f'    mean excess {a.mean_excess_observed:.2%} observed -> '
                     f'{a.mean_excess_used:.2%} projected; arithmetic return '
                     f'{a.arithmetic_annual:.2%} before volatility drag')
    first = next(iter(p.returns.values()))
    lines += [
        f'  cash rate held at {first.rf_annual:.2%} (the era\'s realised rate); '
        f'inflation {p.inflation:.2%} (stated)',
        (f'  amounts in {p.currency}; {p.other_currency} shown at 1 USD = '
         f'{p.fx.usdcad:.4f} CAD ({p.fx.source}, {p.fx.as_of}), held constant — returns are '
         'treated as currency-neutral' if p.fx is not None else
         f'  amounts in {p.currency}; {p.other_currency} not shown: {p.fx_note or "no rate"}'),
        f'  selection population: {p.population_note}',
        f'  {p.n_paths} paths, stationary bootstrap, {p.block_months}-month blocks, seed '
        f'{p.seed}; leverage 1.0',
        f'  tax: marginal now {p.tax.marginal_rate_now:.1%}, at withdrawal '
        f'{p.tax.marginal_rate_retirement:.1%}, capital-gains inclusion '
        f'{p.tax.capital_gains_inclusion:.0%}; every taxable gain realised yearly',
        '  Foreign withholding tax and the Canadian execution gap are NOT modelled '
        '(KNOWN_GAPS.md).',
        '']
    width = max(12, *(len(a.name) for a in p.accounts))
    for a in p.accounts:
        s = p.schedules[a.name]
        line = (f'  {a.name:<{width}} {a.kind:<15} start {a.start_balance:>12,.0f} '
                f'{p.currency}   effort {a.monthly_contribution:,.0f}/month')
        if a.contribution_growth:
            line += f', +{a.contribution_growth:.1%}/yr'
        if a.kind in DEDUCTIBLE:
            line += (', refund reinvested (deposit = effort / (1 - rate))'
                     if a.reinvest_refund else ', refund kept as cash')
        lines.append(line)
        if s.capped:
            first_year = min(s.capped)
            lines.append(f'  {"":<{width}} annual cap {a.annual_cap:,.0f} binds from year '
                         f'{first_year}: {sum(s.capped.values()):,.0f} of effort over the '
                         'horizon is NOT invested')
        if s.refunds_kept.sum() > 0:
            lines.append(f'  {"":<{width}} refunds kept as cash, not in the values below: '
                         f'{s.refunds_kept.sum():,.0f}')
    return lines


def table_lines(p):
    """The projected after-tax values at each checkpoint, for every entry."""
    lines = []
    names = list(p.returns) + ['savings account']
    today = "median, today's $"
    other = f'median, {p.other_currency}' if p.fx is not None else ''
    for month in checkpoints(p.horizon_months):
        years = month / 12
        lines += ['', f'  after {years:g} years — invested {p.invested[month]:,.0f} '
                      f'{p.currency} (start after tax + effort)',
                  f'  {"":<22}{"P10":>13} {"median":>13} {"P90":>13}   '
                  f'{today:>18}  P(< invested)  {other:>14}']
        for name in names:
            s = summary(p, name, month)
            lo, mid, hi = s['nominal']
            conv = f'  {s["other"][1]:>14,.0f}' if p.fx is not None else ''
            if name == 'savings account':
                lines.append(f'  {name:<22}{"":>13} {_money(mid)} {"":>13}   '
                             f'{s["real"][1]:>18,.0f}  {"-":>12}{conv}')
            else:
                lines.append(f'  {name:<22}{_money(lo)} {_money(mid)} {_money(hi)}   '
                             f'{s["real"][1]:>18,.0f}  {s["p_below_invested"]:>12.1%}{conv}')
    lines += ['', '  95th-percentile worst drawdown over the horizon, on the return path: '
              + ', '.join(f'{n} {d:.1%}' for n, d in p.drawdown_p95.items()),
              '  Values are after-tax liquidation values, nominal unless marked. P10 means one '
              'path in ten ends lower.']
    return lines


def report_lines(p):
    return assumption_lines(p) + table_lines(p)


def to_dict(p):
    """A JSON-safe record of the run, for `backtest_results/`."""
    return {
        'currency': p.currency,
        'fx': (None if p.fx is None else
               {'usdcad': p.fx.usdcad, 'source': p.fx.source, 'as_of': p.fx.as_of}),
        'fx_note': p.fx_note,
        'horizon_months': p.horizon_months, 'inflation': p.inflation,
        'tax': vars(p.tax), 'n_paths': p.n_paths, 'block_months': p.block_months,
        'seed': p.seed, 'population': p.population_note,
        'accounts': [vars(a) for a in p.accounts],
        'returns': {n: {'selected': a.selected, 'sharpe_observed': a.haircut.observed,
                        'sharpe_deflated': a.haircut.deflated,
                        'sharpe_lower_bound': a.haircut.lower_bound,
                        'sharpe_used': a.haircut.used, 'vol': a.vol, 'rf_annual': a.rf_annual,
                        'mean_excess_observed': a.mean_excess_observed,
                        'mean_excess_used': a.mean_excess_used,
                        'window': [str(a.first.date()), str(a.last.date())],
                        'n_months': a.n_months}
                    for n, a in p.returns.items()},
        'checkpoints': {n: [summary(p, n, m) for m in checkpoints(p.horizon_months)]
                        for n in p.values},
        'drawdown_p95': p.drawdown_p95,
    }
