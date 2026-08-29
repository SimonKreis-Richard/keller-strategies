"""
Sustainable margin leverage for a TAA model, from its KPIs.

WHAT THIS ANSWERS, and what it deliberately does not. Given a model's measured record, how
much *borrowed* leverage can it carry indefinitely without being liquidated? Not optimally —
**sustainably**. There is no objective function here, no target return, and no search over the
backtest. `tests/test_margin_sizing.py::test_no_optimisation_surface` fails if one is added.

WHY IT HAD TO BE WRITTEN. `common/ledger.py` prices margin leverage — it draws a debit
balance (`debt`), capitalises interest on it (`debt += interest`), and de-levers with the
signal when `leverage_follows_signal` — but **it never once compares equity to a maintenance
requirement.** There is no such comparison anywhere in the repository. Every levered figure it
has ever produced therefore assumes the margin call does not happen. This module is where that
assumption gets priced, and the honest reading of its output is that some of those figures
describe a portfolio that would have been closed out by a broker.

THE OUTPUT THAT MATTERS IS NOT THE NUMBER. It is `binding_constraint`: which real-world limit
stops this model. A model capped at 1.4x by its own drawdown is a different object from one
capped at 1.4x by a credit line, and only the second can be relaxed by phoning a broker.

--------------------------------------------------------------------------------------------
THE FOUR CAPS
--------------------------------------------------------------------------------------------

CAP 1 — MARGIN-CALL SURVIVAL. The principal constraint, and the only one whose absence makes
the whole result meaningless (see "non-calculable" below).

At leverage `f`, assets `A = f*E` and debt `D = (f-1)*E`. Liquidation is `E' = m*A'`, so after
a decline `d` in the positions held:

    f(1-d) - (f-1) = m*f*(1-d)      =>      d_max = (1/f - m) / (1 - m)
    inverted:                               f     = 1 / (m + d(1-m))

That form assumes the debt is CONSTANT through the decline, which `ledger.py` itself
contradicts. With `c = r_b * h/12` accrued over a drawdown of duration `h` months:

    d_max = 1 - (f-1)(1+c) / (f(1-m))
    f     = (1+c) / ((1+c) - (1-d)(1-m))

and both reduce to the first pair at `c = 0`. `h` is not a parameter: it is the peak-to-trough
duration of the model's own worst drawdown, which is why `metrics.max_drawdown_months` had to
be added. Peak-to-TROUGH rather than peak-to-recovery, because the call fires at the trough;
interest keeps accruing afterwards but no longer threatens anything.

Arithmetic, at m=0.30, d=0.60, r_b=6%, h=30 months (c=0.15):

    plain formula                1.389
    with interest accrual        1.322
    with m_crise = 1.5m = 0.45   1.282
    with m = 0.75 (a 3x LETF)    1.111     <- see CAP 1 AND LEVERAGED PRODUCTS

The target decline is `d = k * DD_adj`, and `k` is the module's ONLY preference parameter.

CAP 1 AND LEVERAGED PRODUCTS. `m` is not one number when the book holds LETFs — brokers
commonly apply a multiple of the base maintenance rate to 3x products. At m=0.75 the survivable
leverage is 1.11x, so **margin on top of the registry's 3x entries is arithmetically
unavailable**, not merely imprudent. A scalar `m` is therefore REFUSED when
`letf_mapper.holds_leveraged_product` is true: pass a per-ticker mapping or get
"non-calculable". Silently applying a 25% equity rate to a UPRO/TMF book overstates the
answer by roughly 3x, which is failure mode F1.

CAP 2 — KELLY GATE. `f_kelly = SR/sigma` is `mu_e/sigma^2`, the same quantity as `LEVERAGE.md`
section 1. Two corrections to the naive form:

  * The borrow cost applies to `(f-1)`, not to `f`, and under `MARGIN_FOLLOWS_SIGNAL=True`
    only to the offensive fraction. So with `s = r_b - r_f` and `w_off` the mean offensive
    weight, `g(f) = f*mu_e - (f-1)*s*w_off - f^2 sigma^2/2` and

        f_kelly = (mu_e - s * w_off) / sigma^2

    This is an approximation: the effective leverage is a series, not a scalar (see
    `LedgerResult.effective_leverage`). Named as one, and listed in the output's assumptions.

  * SR is haircut for selection bias, and the haircut is DERIVED rather than chosen, because
    this repository knows its own trial count (19 selection trials). Under the null, the
    expected maximum Sharpe across N trials (Bailey & Lopez de Prado) is

        E[max SR] = sd_SR * ( (1-gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)) )
        N = 19, gamma = 0.5772  =>  E[max SR] = 1.878 * sd_SR

    where `sd_SR` is the cross-sectional standard deviation of the trials' own Sharpes — a
    measurement. The haircut is a SUBTRACTION of that term, not a multiplication by a chosen
    coefficient. In parallel, Mertens' higher-moment standard error gives a sampling-error
    lower bound `SR - z*SE`, using the model's measured skew and kurtosis so that negative
    skew widens the interval as it should. **The more severe of the two is used**, and both
    are reported.

    On a model with SR ~= 1.0 and sd_SR ~= 0.25 the derived haircut is ~0.47, i.e. ~47% —
    next to Harvey & Liu's "halve it" rule of thumb. That agreement is the check that says the
    implementation is not broken, not a coincidence to be proud of.

  * Used as a GATE, never a target. `f_kelly < 1` => no leverage is justified and the walk
    stops. Otherwise it enters the `min` and is usually irrelevant, exactly as expected for
    low-vol high-Sharpe models. It is the haircut that makes it capable of binding at all.

CAP 3 — CARRY. **A DIAGNOSTIC, NOT A CAP**, and this is a deliberate departure from the
request. It is mathematically redundant: it fires when `r_b >= mu - sigma^2/2`, while CAP 2
fires (`f_kelly < 1`) when `mu - r_b < sigma^2`, so

    { mu <= r_b + sigma^2/2 }  is a STRICT SUBSET of  { mu < r_b + sigma^2 }

and CAP 3 can never be the binding constraint. It also compares a MARGINAL cost (`r_b`) to an
AVERAGE return (`CAGR`); the correct marginal comparison is CAP 2's first-order condition. It
is retained for two reasons: it is the one line a human reads to sanity-check the inputs, and
it is the fallback when CAP 2 is non-calculable. It never enters the `min`.

CAP 4 — BORROWING CAPACITY. A parameter, and frequently the constraint that actually binds.
One warning it cannot enforce: registered accounts do not permit margin, so the capacity figure
must be the taxable account's equity, not the household's. Filling it in with a total makes
every other number in the output describe a portfolio that cannot be held (F12).

--------------------------------------------------------------------------------------------
DD_adj — THE TWO CORRECTIONS, MEASURED RATHER THAN ASSUMED
--------------------------------------------------------------------------------------------

(a) SAMPLE BIAS. An observed maximum drawdown is a sample minimum; the next one is worse. So
    the input is not the observed figure but a high quantile of the drawdown distribution over
    the INTENDED HOLDING HORIZON, obtained by stationary block bootstrap (Politis & Romano) of
    the model's own monthly returns. Horizon, quantile, expected block length and seed are all
    named parameters.

    **The multiplier is an output, not an input.** `dd_factor_sample` reports
    `DD_bootstrapped / DD_observed` per model, and `block_sensitivity` reports the same
    quantile at three block lengths, so the one genuinely arbitrary choice is visible instead
    of buried. That is the whole reason to prefer this over a coefficient: the coefficient
    still exists, but it is measured and printed.

    Below `min_history_months` the answer is non-calculable. There is deliberately no fallback
    heuristic: a made-up markup on a record that never saw 2008 is precisely F2.

(b) INTRA-PERIOD RISK. A monthly-rebalanced model does not react inside the month, so the
    decline in the positions actually held exceeds what a month-end series shows. This is
    MEASURED, not marked up: hold the month's allocation fixed, value it on daily prices, and
    take the ratio of the daily-path drawdown to the month-end one. `intraperiod_max_drawdown`
    does it; the caller passes the result in, because this module takes KPIs and not a
    PriceStore.

    Named residual: daily closes are not intraday, and a margin call is intraday.
    `intraday_pad` exists for it and DEFAULTS TO 1.0 — no padding. A declared gap beats an
    invented coefficient, and it is listed in `invalidating_assumptions` on every call.

    Composed as two visible factors rather than one daily bootstrap. A daily bootstrap would
    be more rigorous and would hide the decomposition, and the decomposition is the point.

    FLOOR: `DD_adj` is never allowed below the observed drawdown. A stressed figure shallower
    than one that already happened is not a stress. When the floor binds it is flagged
    (`dd_floor_binding`), because that means the bootstrap is understating.

--------------------------------------------------------------------------------------------
LEVERAGE DRIFT — WHERE THE REQUEST IS INVERTED
--------------------------------------------------------------------------------------------

Maintaining target leverage PROTECTS against liquidation. When equity falls, `f_eff` rises, and
restoring the target requires SELLING. Worked: m=0.25, f=2, two -20% moves. Unmanaged,
`f_eff` goes 2 -> 2.67 -> 4.57 and the call fires at -36% cumulative. Reset to target after the
first, and it survives. A continuously-maintained constant leverage is never liquidated at all.

So `d_max = (1/f - m)/(1-m)` is already the FULLY-DRIFTED answer, measured from the anchor at
which leverage was last set — the pessimistic case on that axis, not the optimistic one. What
the static calculation genuinely misses is four other things:

    (i)   interest capitalised through the decline        -> the `c` term above
    (ii)  `m` raised by the broker during the crisis      -> `crisis_margin_multiple`
    (iii) the un-defended sub-monthly excursion           -> DD_adj correction (b)
    (iv)  a ONE-SIDED policy that re-levers on gains but never de-levers on losses -> F5

(iii) is the real gap in this engine specifically, and it follows from `ledger.py:340`
(`target = w * mult * equity`): the reset is monthly and unconditional, while the margin call
is continuous. The distance between those two frequencies is the entire residual risk.

`simulate_margin_path` walks a path under a named policy so (iii) and (iv) are testable rather
than argued.

--------------------------------------------------------------------------------------------
NON-CALCULABLE, AND HOW IT PROPAGATES
--------------------------------------------------------------------------------------------

No input is defaulted silently. A missing or unreliable input makes its cap non-calculable, and:

  * CAP 1 non-calculable  => the WHOLE result is non-calculable. It is the principal
    constraint, and returning `min` over the survivors would answer a cheerful 3x from a
    credit-line parameter while the question of whether the model survives is unanswered.
  * CAP 2 / CAP 4 non-calculable => the result is returned with that axis marked unbounded and
    the fact recorded in `invalidating_assumptions`.

COMPARABILITY. One `MarginPolicy` instance is shared across models and echoed verbatim into
every result, so `test_parameter_block_is_identical_across_models` can assert the blocks match.
Per-model measurements differ; per-model *parameters* do not.
"""

from dataclasses import dataclass, field, replace
from statistics import NormalDist

import math
import numpy as np
import pandas as pd

from common import metrics as metrics_mod

#: Euler-Mascheroni. Appears in the expected-maximum-of-N-draws term of the Sharpe haircut.
EULER_MASCHERONI = 0.5772156649015329

_NORM = NormalDist()

#: Parameter names a public entry point may never accept. The constraint is the owner's and it
#: is structural rather than a promise: `f` must be sustainable indefinitely, never fitted to
#: the sample. `test_no_optimisation_surface` reads the signatures against this set.
FORBIDDEN_OBJECTIVE_PARAMS = frozenset({
    'target', 'target_return', 'target_cagr', 'target_leverage', 'objective', 'maximise',
    'maximize', 'optimise', 'optimize', 'fit', 'tune', 'search',
})


class NotCalculable(ValueError):
    """An input is missing or unreliable, so no number may be returned for this cap."""


#: Highest maintenance requirement the CAP 1 formulas will carry. `m = 1` is a pole — the closed
#: forms divide by `(1 - m)` — and it describes a position the broker lends nothing against, so
#: the meaningful answer there is "no margin", i.e. `f = 1.00x`, not an exception.
#:
#: This ceiling was applied only to the CRISIS variant until 2026-07-31, which left a live crash
#: one slider-move away: a 3x book at a 35% base rate resolves to `m = 1.05` and
#: `leverage_for_threshold` raised `ValueError` — uncaught, because `recommend_leverage` catches
#: only `NotCalculable`, so the whole run died rather than one row. Clamping is now applied to
#: both variants and NOTED on the result when it bites, because a clamped `m` is an answer about
#: an assumption rather than about the model.
MAINTENANCE_CEILING = 0.99


# --------------------------------------------------------------------------------------- #
#  the shared parameter block
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MarginPolicy:
    """Every numeric choice, named, with its justification. There are no other constants.

    Shared across models unchanged — that is what makes results alignable side by side. Echoed
    verbatim into each `MarginRecommendation`.
    """

    #: THE only preference parameter, in MULTIPLES OF DRAWDOWN. `k=3` means the position must
    #: survive three times the stressed drawdown before the broker liquidates. Deliberately
    #: never converted into an opaque constant: it appears in the output as a multiple and the
    #: reported threshold/observed-drawdown ratio is provably >= k (see `_INVARIANT` note).
    safety_factor_k: float = 3.0

    #: Broker maintenance margin. Either a scalar (a homogeneous unlevered equity book) or a
    #: {ticker: rate} mapping. REQUIRED — there is no plausible default, and the plausible-
    #: looking one (0.25, Reg-T) is wrong by ~3x for a book holding LETFs. `None` =>
    #: CAP 1 non-calculable => whole result non-calculable.
    maintenance_margin: object = None

    #: Multiple applied to `m` for the crisis variant. The real failure mode is `m` rising
    #: while the market falls, and 1.5x is the low end of what brokers have done in past
    #: dislocations. A parameter because it is a claim about broker behaviour, not arithmetic.
    crisis_margin_multiple: float = 1.5

    #: Report the crisis variant as THE recommendation rather than as a footnote. Default True:
    #: a module that exists to model the failure mode should headline the number that survives
    #: it. Both variants are always present in `caps` either way.
    use_crisis_margin: bool = True

    #: Charge interest accrual over the drawdown's peak-to-trough duration. Default True; the
    #: ledger capitalises interest, so ignoring it here would contradict the engine.
    use_interest_accrual: bool = True

    #: Annual rate on the debit balance. `ledger.ExecutionConfig.borrow_rate` uses 0.06; there
    #: is no default here on purpose, because the broker is not yet known.
    borrow_rate_annual: float = None

    #: Annual risk-free rate, for the borrow SPREAD `s = r_b - r_f`. Pass the model's own
    #: `metrics['rf_annual']` so the spread is measured against the same cash series the
    #: Sharpe was netted against.
    risk_free_annual: float = None

    #: Maximum leverage the broker or credit line permits. **Must be the equity of the
    #: account that can actually borrow** — registered accounts cannot, so a household total
    #: makes every other figure describe an unattainable portfolio.
    borrowing_capacity_leverage: float = None

    #: Intended holding horizon for the drawdown distribution. 240 months = 20 years. This is
    #: the horizon the leverage must survive, and it is NOT the backtest length: sizing to the
    #: drawdown of an 18-year sample is sizing to the sample.
    horizon_months: int = 240

    #: Quantile of the bootstrapped drawdown distribution. A preference, exposed.
    dd_quantile: float = 0.95

    #: Expected block length for the stationary bootstrap. 12 months because the signals are
    #: built on 12-month lookbacks, so the dependence the strategy trades has that scale.
    #: `block_sensitivity` reports the quantile at each of `block_months_sensitivity` so the
    #: choice is visible rather than buried.
    block_months: int = 12
    block_months_sensitivity: tuple = (6, 12, 24)

    #: Bootstrap paths. 2000 is enough for a 0.95 quantile to be stable to ~1pp.
    n_bootstrap: int = 2000

    #: REQUIRED. Determinism is a stated constraint, and an unseeded bootstrap breaks it.
    seed: int = None

    #: Shortest record from which a drawdown distribution may be extrapolated. 120 months
    #: because a 20-year horizon quantile bootstrapped from under a decade is an extrapolation
    #: dressed as a measurement. Below it: non-calculable, with no fallback heuristic.
    min_history_months: int = 120

    #: Number of SELECTION TRIALS the candidate came from — 19 in this repository. Drives the
    #: expected-maximum-Sharpe haircut. `None` => the multiple-testing haircut is skipped and
    #: the fact is recorded in the assumptions.
    n_trials: int = None

    #: Cross-sectional standard deviation of those trials' Sharpes. A measurement, not a guess.
    trial_sharpe_sd: float = None

    #: Standard errors for the sampling-error lower bound on Sharpe. 1.0 = one sigma.
    sharpe_z: float = 1.0

    #: Multiplier on the measured intra-period uplift, for the residual between daily closes
    #: and true intraday. DEFAULTS TO 1.0 — no padding. The gap is declared in every result's
    #: `invalidating_assumptions` instead of being papered over with a number nobody measured.
    intraday_pad: float = 1.0

    #: Which leverage-maintenance policy the recommendation assumes. Documented as a design
    #: decision, not an implementation detail — see the module docstring's DRIFT section.
    #: 'reset_monthly' is what `ledger.py:340` actually does.
    leverage_policy: str = 'reset_monthly'

    #: Presets for `safety_factor_k`, in multiples of drawdown.
    PRESETS = {'prudent': 5.0, 'balanced': 3.0, 'aggressive': 2.0}

    @classmethod
    def preset(cls, name, **kwargs):
        """A policy at one of the named risk presets. Everything else still has to be passed."""
        if name not in cls.PRESETS:
            raise ValueError(f'unknown preset {name!r}; choose from {sorted(cls.PRESETS)}')
        return cls(safety_factor_k=cls.PRESETS[name], **kwargs)

    def __post_init__(self):
        if self.safety_factor_k <= 0:
            raise ValueError('safety_factor_k must be > 0 — it is a multiple of drawdown')
        if not 0.0 < self.dd_quantile < 1.0:
            raise ValueError('dd_quantile must lie strictly inside (0, 1)')
        if self.crisis_margin_multiple < 1.0:
            raise ValueError('crisis_margin_multiple must be >= 1: the crisis case cannot be '
                             'a LOWER maintenance requirement than the normal one')
        if self.intraday_pad < 1.0:
            raise ValueError('intraday_pad must be >= 1: it pads a measured excursion, it '
                             'cannot shrink one')
        if self.leverage_policy not in POLICIES:
            raise ValueError(f'leverage_policy must be one of {sorted(POLICIES)}')

    def fingerprint(self):
        """The parameter block as a comparable tuple, for the cross-model identity assertion."""
        m = self.maintenance_margin
        m_key = tuple(sorted(m.items())) if isinstance(m, dict) else m
        return (('maintenance_margin', m_key),) + self.preference_fingerprint()

    def preference_fingerprint(self):
        """The block WITHOUT the maintenance mapping — what `compare_table` requires to match.

        `maintenance_margin` is excluded deliberately, and the distinction is the difference
        between a preference and a fact. `k`, the quantile, the horizon, the rates and the seed
        are CHOICES: two models sized under different ones cannot be read side by side. The
        maintenance requirement is a PROPERTY OF THE HOLDINGS — a UPRO/TMF book faces a
        different rate than an SPY/IEF book at the same broker on the same day. Requiring it to
        match would make the aligned table unbuildable for a registry that contains both, which
        is precisely the registry it exists to compare.
        """
        return (
            ('safety_factor_k', self.safety_factor_k),
            ('crisis_margin_multiple', self.crisis_margin_multiple),
            ('use_crisis_margin', self.use_crisis_margin),
            ('use_interest_accrual', self.use_interest_accrual),
            ('borrow_rate_annual', self.borrow_rate_annual),
            ('risk_free_annual', self.risk_free_annual),
            ('borrowing_capacity_leverage', self.borrowing_capacity_leverage),
            ('horizon_months', self.horizon_months), ('dd_quantile', self.dd_quantile),
            ('block_months', self.block_months),
            ('block_months_sensitivity', tuple(self.block_months_sensitivity)),
            ('n_bootstrap', self.n_bootstrap), ('seed', self.seed),
            ('min_history_months', self.min_history_months), ('n_trials', self.n_trials),
            ('trial_sharpe_sd', self.trial_sharpe_sd), ('sharpe_z', self.sharpe_z),
            ('intraday_pad', self.intraday_pad), ('leverage_policy', self.leverage_policy),
        )


# --------------------------------------------------------------------------------------- #
#  the model's measured record
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelKPIs:
    """What the module needs to know about one model. All of it measured, none of it chosen."""

    name: str
    monthly_returns: pd.Series          # net, as `LedgerResult.returns`
    max_dd: float                       # observed, NEGATIVE, as `metrics['max_dd']`
    max_dd_months: int                  # peak-to-trough duration, `metrics.max_drawdown_months`
    sharpe: float                       # annualised, net of the same rf as `rf_annual`
    vol: float                          # annualised
    cagr: float
    rf_annual: float
    #: Mean offensive weight, from `BaseStrategy.offensive_weight(alloc, scores).mean()`. The
    #: borrowed fraction under `MARGIN_FOLLOWS_SIGNAL=True`, so it scales the carry cost in
    #: CAP 2. Pass 1.0 for flat margin.
    offensive_weight_mean: float
    #: True when any sleeve can hold a leveraged product — `letf_mapper.holds_leveraged_product`.
    #: Forces a per-ticker `maintenance_margin`; a scalar is refused.
    holds_leveraged_product: bool
    #: Tickers the book can hold, used to look up per-ticker maintenance rates.
    held_tickers: tuple = ()
    #: Maximum drawdown of the DAILY path with each month's allocation HELD, from
    #: `intraperiod_max_drawdown`. Negative. `None` => the intra-period correction is
    #: non-calculable, and so is CAP 1.
    daily_max_dd: float = None

    @classmethod
    def from_metrics(cls, name, metrics, monthly_returns, offensive_weight_mean,
                     holds_leveraged_product, held_tickers=(), daily_max_dd=None):
        """Build from a `metrics.calculate_metrics` dict, so the KPIs are the reported ones.

        Reading them off the same dict the report prints is the point: a sizing module that
        recomputes its own Sharpe can disagree with the table beside it and nobody would know.
        """
        return cls(
            name=name,
            monthly_returns=monthly_returns,
            max_dd=float(metrics['max_dd']),
            max_dd_months=int(metrics.get('max_dd_months')
                              if metrics.get('max_dd_months') is not None
                              else metrics_mod.max_drawdown_months(monthly_returns)),
            sharpe=float(metrics['sharpe']),
            vol=float(metrics['vol']),
            cagr=float(metrics['cagr']),
            rf_annual=float(metrics['rf_annual']),
            offensive_weight_mean=float(offensive_weight_mean),
            holds_leveraged_product=bool(holds_leveraged_product),
            held_tickers=tuple(held_tickers),
            daily_max_dd=None if daily_max_dd is None else float(daily_max_dd),
        )


# --------------------------------------------------------------------------------------- #
#  results
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Cap:
    """One independent ceiling. `value is None` means NON-CALCULABLE, never "no limit"."""

    name: str
    value: float
    basis: str                 # how it was derived, or why it could not be
    binds: bool = True         # False for CAP 3, which is a diagnostic and never enters the min

    @property
    def calculable(self):
        return self.value is not None

    def __str__(self):
        v = 'non calculable' if self.value is None else f'{self.value:.3f}x'
        return f'{self.name}: {v} ({self.basis})'


@dataclass(frozen=True)
class MarginRecommendation:
    """The full answer. Read `binding_constraint` before `recommended_leverage`."""

    model: str
    recommended_leverage: float          # None => non-calculable
    binding_constraint: str
    leverage_justified: bool
    caps: dict

    # the drawdown chain, each factor separately so each can be argued with
    dd_observed: float
    dd_bootstrapped: float
    dd_factor_sample: float
    dd_factor_intraperiod: float
    dd_adj: float
    dd_floor_binding: bool
    dd_clamped_at_one: bool
    max_dd_months: int
    block_sensitivity: dict

    # the Sharpe chain, likewise
    sharpe_observed: float
    sharpe_deflated: float
    sharpe_lower_bound: float
    sharpe_used: float
    deflated_sharpe_probability: float
    return_skew: float
    return_kurtosis: float
    return_autocorr_1: float

    # what a broker would actually do
    implied_liquidation_threshold: float
    threshold_over_observed_maxdd: float
    maintenance_margin_used: float
    interest_accrued_fraction: float

    policy: MarginPolicy
    invalidating_assumptions: tuple = ()
    notes: tuple = ()

    def cap_value(self, name):
        cap = self.caps.get(name)
        return None if cap is None else cap.value

    def report(self):
        """Fixed-width block, one model. Aligns across models by construction."""
        f = 'non calculable' if self.recommended_leverage is None \
            else f'{self.recommended_leverage:.2f}x'
        lines = [
            f'{self.model}',
            f'  recommended leverage      {f}',
            f'  binding constraint        {self.binding_constraint}',
            f'  leverage justified        {"yes" if self.leverage_justified else "NO"}',
            '  caps',
        ]
        for cap in self.caps.values():
            if cap.binds:
                tag = ''
            elif cap.name == 'carry_diagnostic':
                tag = '   [diagnostic, never binds]'
            else:
                # A CAP 1 variant that is real and computed but is not the one `use_crisis_margin`
                # / `use_interest_accrual` selected. Calling it a "diagnostic" would misdescribe
                # it: it is the same cap under a different broker assumption.
                tag = '   [variant, not retained]'
            v = 'non calculable' if cap.value is None else f'{cap.value:8.3f}x'
            lines.append(f'    {cap.name:<32} {v}{tag}')
        lines += [
            f'  drawdown  observed {self.dd_observed:7.2%}  bootstrapped '
            f'{self.dd_bootstrapped:7.2%}  adjusted {self.dd_adj:7.2%}',
            f'    sample factor {self.dd_factor_sample:.3f}x   '
            f'intra-period factor {self.dd_factor_intraperiod:.3f}x   '
            f'duration {self.max_dd_months} months',
            f'  sharpe    observed {self.sharpe_observed:6.3f}  deflated '
            f'{self.sharpe_deflated:6.3f}  lower bound {self.sharpe_lower_bound:6.3f}  '
            f'used {self.sharpe_used:6.3f}',
            f'  liquidation at        {self.implied_liquidation_threshold:7.2%} decline '
            f'= {self.threshold_over_observed_maxdd:.2f}x the observed drawdown',
        ]
        if self.invalidating_assumptions:
            lines.append('  if these are false, the result is void:')
            lines += [f'    - {a}' for a in self.invalidating_assumptions]
        return '\n'.join(lines)


# --------------------------------------------------------------------------------------- #
#  cap 1 mechanics
# --------------------------------------------------------------------------------------- #

def liquidation_threshold(leverage, maintenance_margin, accrued=0.0):
    """Decline in the POSITIONS HELD that triggers the call, at `leverage`.

    `d_max = 1 - (f-1)(1+c) / (f(1-m))`, which is `(1/f - m)/(1-m)` when `c = 0`. Returns 1.0
    at `f = 1`: an unlevered book has no call, whatever `m` is.
    """
    f, m, c = float(leverage), float(maintenance_margin), float(accrued)
    if f <= 1.0:
        return 1.0
    if not 0.0 <= m < 1.0:
        raise ValueError(f'maintenance margin must lie in [0, 1), got {m}')
    return 1.0 - (f - 1.0) * (1.0 + c) / (f * (1.0 - m))


def leverage_for_threshold(decline, maintenance_margin, accrued=0.0):
    """The inverse: the largest `f` surviving a decline of `decline` before liquidation.

    `f = (1+c) / ((1+c) - (1-d)(1-m))`, i.e. `1/(m + d(1-m))` when `c = 0`. Clamped at 1.0 from
    below, because a decline no leverage survives is answered by holding no leverage — not by a
    number under one, and not by a division error (F13).
    """
    d, m, c = float(decline), float(maintenance_margin), float(accrued)
    if not 0.0 <= m < 1.0:
        raise ValueError(f'maintenance margin must lie in [0, 1), got {m}')
    d = min(max(d, 0.0), 1.0)
    denom = (1.0 + c) - (1.0 - d) * (1.0 - m)
    if denom <= 0.0:
        return math.inf
    return max(1.0, (1.0 + c) / denom)


def accrued_interest_fraction(borrow_rate_annual, months):
    """`c = r_b * h/12`, simple accrual over the drawdown's peak-to-trough duration.

    Simple rather than compounded, matching `ledger.py`'s `debt * borrow_rate * days / 365`.
    Using a different convention here than the engine charges would make the two disagree
    about the same loan.
    """
    if borrow_rate_annual is None or months is None:
        raise NotCalculable('interest accrual needs both a borrow rate and a drawdown duration')
    return float(borrow_rate_annual) * float(months) / 12.0


def resolve_maintenance_margin(policy, kpis):
    """The single `m` CAP 1 uses, or raise `NotCalculable`.

    A scalar is REFUSED when the book can hold a leveraged product (the owner's decision,
    2026-07-30). Brokers commonly apply a multiple of the base maintenance rate to 3x products,
    and at m=0.75 the survivable leverage is 1.11x — so applying an equity rate to a UPRO/TMF
    book does not shade the answer, it triples it.

    Given a mapping, the MAXIMUM over the tickers the book can hold is used. Conservative on
    purpose: the requirement that binds is the one on the position held when the decline
    arrives, and which position that is cannot be known in advance.
    """
    m = policy.maintenance_margin
    if m is None:
        raise NotCalculable('maintenance_margin is required; there is no defensible default '
                            '(0.25 is Reg-T for unlevered equity and wrong by ~3x for a book '
                            'holding LETFs)')
    if isinstance(m, dict):
        if not kpis.held_tickers:
            raise NotCalculable('a per-ticker maintenance_margin needs `held_tickers` to look '
                                'up; none were declared')
        missing = sorted(set(kpis.held_tickers) - set(m))
        if missing:
            raise NotCalculable(f'no maintenance rate given for {missing}; a book whose '
                                f'requirement is unknown for any holding has no calculable cap')
        return max(float(m[t]) for t in kpis.held_tickers)
    if kpis.holds_leveraged_product:
        raise NotCalculable(
            f'{kpis.name} can hold a leveraged product, so a SCALAR maintenance margin is '
            f'refused (RULE, 2026-07-30). Brokers commonly apply a multiple of the base rate '
            f'to 3x products: at m=0.75 the survivable leverage is 1.11x, against '
            f'{leverage_for_threshold(0.60, 0.30):.2f}x at m=0.30 on the same drawdown. Pass a '
            f'{{ticker: rate}} mapping covering every holding.')
    return float(m)


# --------------------------------------------------------------------------------------- #
#  drawdown distribution
# --------------------------------------------------------------------------------------- #

def stationary_bootstrap_indices(n_obs, horizon, expected_block, n_paths, rng):
    """Politis-Romano stationary bootstrap, as INDICES into a series of length `n_obs`.

    Stationary rather than fixed-block because a fixed block length makes the resampled series
    non-stationary in a way that biases extremes, and extremes are the entire output here.

    Indices rather than values because there are two consumers with different needs and only
    one of them is resampling a single series. `common/robustness.py` resamples a whole
    leaderboard and MUST apply one set of indices to every strategy at once, so that each path
    is a coherent alternative history instead of N unrelated ones. Returning the draw itself
    is what lets both callers share one definition of "a resampled month".
    """
    p = 1.0 / float(expected_block)
    restart = rng.random((n_paths, horizon)) < p
    fresh = rng.integers(0, n_obs, size=(n_paths, horizon))
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    cur = rng.integers(0, n_obs, size=n_paths)
    for t in range(horizon):
        idx[:, t] = cur
        cur = np.where(restart[:, t], fresh[:, t], (cur + 1) % n_obs)
    return idx


def _stationary_bootstrap_paths(returns, horizon, expected_block, n_paths, rng):
    """The same draw, applied to one series. Kept as the name the drawdown quantile uses."""
    r = np.asarray(returns, dtype=float)
    return r[stationary_bootstrap_indices(len(r), horizon, expected_block, n_paths, rng)]


def bootstrap_drawdown_quantile(returns, horizon_months, expected_block, n_paths, quantile,
                                seed):
    """Quantile of the maximum-drawdown distribution over `horizon_months`. Returns NEGATIVE.

    The wealth path carries a leading 1.0, matching `metrics.wealth_curve`. That is not
    cosmetic: without it a drawdown beginning in the first month is invisible, which was audit
    finding #1 in `common/metrics.py`, and a sizing module that reproduced the bug would report
    a shallower stress than the report it sits beside.
    """
    if seed is None:
        raise NotCalculable('seed is required — an unseeded bootstrap is not reproducible, '
                            'and determinism is a stated constraint')
    r = pd.Series(returns).dropna()
    if r.empty:
        raise NotCalculable('no returns to resample')
    rng = np.random.default_rng(int(seed))
    paths = _stationary_bootstrap_paths(r.to_numpy(dtype=float), int(horizon_months),
                                        expected_block, int(n_paths), rng)
    wealth = np.cumprod(1.0 + paths, axis=1)
    wealth = np.concatenate([np.ones((wealth.shape[0], 1)), wealth], axis=1)
    dd = wealth / np.maximum.accumulate(wealth, axis=1) - 1.0
    worst = dd.min(axis=1)                      # negative, one per path
    # Quantile of SEVERITY, then re-signed. Taking `np.quantile(worst, 1-q)` would be the same
    # number but reads as the wrong tail, and this is code somebody will have to check.
    return -float(np.quantile(-worst, quantile))


def intraperiod_max_drawdown(daily_prices, monthly_weights, cash_ticker=None):
    """Maximum drawdown of the DAILY path with each month's allocation HELD fixed.

    This is correction (b) as a measurement instead of a markup. `monthly_weights` is the
    target allocation frame indexed by decision date; between decision dates the weights are
    held and the positions drift with daily prices, which is what a monthly-rebalanced model
    actually does inside the month.

    The uplift over the month-end drawdown is real and unmodelled anywhere else in this
    repository: `ledger.run_ledger` drifts positions from one execution date to the next and
    never looks in between, so nothing it reports can see an excursion that recovered by the
    month end (F11).

    WRITTEN AGAINST THE ARRAYS, not the frames, since 2026-07-31. The obvious form — slice the
    daily frame with `px.loc[(px.index >= start) & (px.index <= end)]` once per month — builds a
    fresh boolean mask over the WHOLE history for each of ~230 months, and then does it again
    for each of 36 strategies. That is 50x the work of the same arithmetic on positional slices,
    and it is what made this measurement too slow to sit in a dashboard column. The index is
    sorted (first statement below), so `searchsorted` gives exactly the same window.
    `tests/test_margin_sizing.py::TestIntraperiodFastPath` pins it against the frame-wise
    reference on random data, because "faster and equal" is a claim that has to be checked
    rather than asserted — it was equal to 1.2e-15 over the whole registry when it landed.

    `np.nansum` where pandas would have used `Series.sum`: pandas skips NaN by default and
    numpy propagates it, so a ticker that goes missing mid-month would silently NaN the entire
    equity path. Same convention, stated rather than inherited.
    """
    px = daily_prices.sort_index()
    w = monthly_weights.sort_index()
    cols = [c for c in w.columns if c in px.columns]
    if not cols:
        raise NotCalculable('no held ticker has daily prices')
    px, w = px[cols], w[cols]
    values, index = px.to_numpy(dtype=float), px.index
    weights_all = w.to_numpy(dtype=float)

    pieces, stamps, level = [], [], 1.0
    dates = list(w.index)
    for i, start in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else index[-1]
        lo = int(index.searchsorted(start, 'left'))
        hi = int(index.searchsorted(end, 'right'))
        if hi - lo < 2:
            continue
        weights = weights_all[i]
        total = float(np.nansum(weights))
        if total <= 0:
            continue
        first = values[lo]
        # A ticker with no price on the first day of the window cannot be valued from it, so it
        # is dropped and the rest are re-weighted by their own sum below — the same treatment
        # the frame-wise form gave it.
        held = (np.abs(weights) > 1e-12) & ~np.isnan(first)
        if not held.any():
            continue
        w_held = weights[held]
        rel = values[lo:hi][:, held] / first[held]
        path = level * np.nansum(rel * w_held, axis=1) / float(w_held.sum())
        pieces.append(path[1:])
        stamps.append(index[lo + 1:hi])
        level = float(path[-1])
    if not pieces:
        raise NotCalculable('no valuable daily window')
    curve = pd.Series(np.concatenate([[1.0]] + pieces),
                      index=pd.DatetimeIndex([dates[0]]).append(stamps))
    return float(metrics_mod.drawdown_series(curve).min())


# --------------------------------------------------------------------------------------- #
#  sharpe haircuts
# --------------------------------------------------------------------------------------- #

def expected_max_sharpe(n_trials, trial_sharpe_sd):
    """`E[max SR]` across `n_trials` independent trials under the null (Bailey & Lopez de Prado).

    `sd_SR * ((1-g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))`. At N=19 — this repository's
    selection-trial count — the bracket is 1.878, so the haircut is `1.878 * sd_SR`. Both
    inputs are counted or measured, which is the whole reason this is preferred to a chosen
    coefficient.
    """
    if n_trials is None or trial_sharpe_sd is None:
        raise NotCalculable('the multiple-testing haircut needs both the trial count and the '
                            'cross-sectional sd of the trials\' Sharpes')
    n = int(n_trials)
    if n < 2:
        return 0.0
    sd = float(trial_sharpe_sd)
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n * math.e))
    return sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def sharpe_standard_error(sharpe_annual, returns, periods_per_year=12):
    """Mertens' higher-moment standard error of an annualised Sharpe.

    `Var(SR_p) = (1 + SR_p^2/2 - g3*SR_p + (g4-3)/4 * SR_p^2) / T`, annualised by
    `sqrt(periods_per_year)`. The higher-moment form rather than the iid one because it uses
    the model's MEASURED skew and kurtosis: negative skew widens the interval, which is the
    correct direction for a strategy that de-risks late.
    """
    r = pd.Series(returns).dropna()
    t = len(r)
    if t < 3:
        raise NotCalculable('too few observations for a Sharpe standard error')
    sr_p = float(sharpe_annual) / math.sqrt(periods_per_year)
    g3 = float(r.skew())
    g4 = float(r.kurt()) + 3.0            # pandas reports EXCESS kurtosis
    var = (1.0 + 0.5 * sr_p ** 2 - g3 * sr_p + 0.25 * (g4 - 3.0) * sr_p ** 2) / t
    return math.sqrt(max(var, 0.0)) * math.sqrt(periods_per_year)


def deflated_sharpe_probability(sharpe_annual, returns, n_trials, trial_sharpe_sd,
                                periods_per_year=12):
    """`P(true SR > 0)` after correcting for selection across `n_trials`. Reported, not gating.

    A probability is the honest read of the haircut: subtracting `E[max SR]` gives a number,
    but only this says how confident the subtraction leaves you.
    """
    e_max = expected_max_sharpe(n_trials, trial_sharpe_sd)
    se = sharpe_standard_error(sharpe_annual, returns, periods_per_year)
    if se <= 0:
        return float('nan')
    return float(_NORM.cdf((float(sharpe_annual) - e_max) / se))


# --------------------------------------------------------------------------------------- #
#  the caps
# --------------------------------------------------------------------------------------- #

def cap_margin_survival(dd_adj, maintenance_margin, safety_factor_k, accrued=0.0):
    """CAP 1. The largest `f` surviving `k * DD_adj` before the broker liquidates."""
    d = float(safety_factor_k) * abs(float(dd_adj))
    return leverage_for_threshold(d, maintenance_margin, accrued), min(d, 1.0)


def cap_kelly(sharpe_used, vol, borrow_spread, offensive_weight_mean):
    """CAP 2. `f* = (mu_e - s*w_off) / sigma^2`, with `mu_e = SR_used * sigma`.

    The spread enters as a flat subtraction because it is charged on `(f-1)`, not on `f`:
    `d/df[(f-1)s] = s`. Scaling it by `f` instead — the obvious reading of "mu net of borrowing
    cost" — would tilt the whole curve rather than shifting its optimum, and would make the
    answer wrong in a direction that looks conservative.
    """
    sigma = float(vol)
    if sigma <= 0:
        raise NotCalculable('zero volatility: Kelly is undefined')
    mu_e = float(sharpe_used) * sigma
    return (mu_e - float(borrow_spread) * float(offensive_weight_mean)) / (sigma ** 2)


def carry_diagnostic(cagr, borrow_rate_annual):
    """CAP 3, REQUALIFIED AS A DIAGNOSTIC. Returns `(carry_is_positive, margin_in_pp)`.

    Never enters the `min`, because it cannot bind: it fires when `r_b >= mu - sigma^2/2`,
    while CAP 2's gate fires when `mu - r_b < sigma^2`, and the first region is a strict subset
    of the second. Kept because it is the line a human reads to check the inputs, and because
    it is the fallback when CAP 2 is non-calculable.
    """
    if borrow_rate_annual is None:
        raise NotCalculable('carry needs a borrow rate')
    return bool(float(cagr) > float(borrow_rate_annual)), float(cagr) - float(borrow_rate_annual)


# --------------------------------------------------------------------------------------- #
#  the walk
# --------------------------------------------------------------------------------------- #

def recommend_leverage(kpis, policy):
    """Sustainable margin leverage for one model, with the constraint that determined it.

    Takes only measured KPIs and a shared parameter block. Deliberately accepts no target and
    no objective: see `FORBIDDEN_OBJECTIVE_PARAMS`.
    """
    notes, assumptions = [], [
        'The broker may raise the maintenance requirement AND recall the loan mid-decline. '
        f'crisis_margin_multiple={policy.crisis_margin_multiple:g} covers the first; nothing '
        'covers the second.',
        f'Daily closes are not intraday, and a margin call is intraday. intraday_pad='
        f'{policy.intraday_pad:g} declares the residual rather than guessing it.',
        'The two drawdown corrections COMPOSE MULTIPLICATIVELY, which assumes the intra-period '
        'uplift measured on the observed sample carries over to a deeper bootstrapped one. It '
        'cannot hold exactly at the extreme — a drawdown is bounded below by -100%, so the '
        'ratio must compress as the base deepens. The product therefore OVERSTATES DD_adj for '
        'the deepest rows, which is conservative here but is an approximation, not a result.',
        'Caps do not compose across models: the drawdown of a blend is not the maximum of its '
        'members\' drawdowns. Sizing several models jointly requires the blend\'s own record.',
        f'`{policy.leverage_policy}` is assumed for maintaining target leverage. `ledger.py` '
        'resets monthly and unconditionally; the margin call is continuous.',
    ]

    # --- DD_adj: two measured factors, a floor, and a clamp ---------------------------- #
    dd_obs = abs(float(kpis.max_dd))
    dd_boot = dd_factor_sample = dd_factor_intra = dd_adj = float('nan')
    dd_floor_binding = dd_clamped = False
    block_sens, dd_error = {}, None
    try:
        n_obs = len(pd.Series(kpis.monthly_returns).dropna())
        if n_obs < policy.min_history_months:
            raise NotCalculable(
                f'{n_obs} months of history, below min_history_months='
                f'{policy.min_history_months}. A 20-year drawdown quantile bootstrapped from '
                f'less than a decade is an extrapolation wearing a measurement\'s clothes, and '
                f'there is deliberately no fallback heuristic.')
        if kpis.daily_max_dd is None:
            raise NotCalculable(
                'daily_max_dd is required: the intra-period correction is MEASURED from the '
                'daily path of the held allocation (`intraperiod_max_drawdown`), and assuming '
                'a factor of 1.0 would be exactly the silent default this module refuses.')
        dd_boot = abs(bootstrap_drawdown_quantile(
            kpis.monthly_returns, policy.horizon_months, policy.block_months,
            policy.n_bootstrap, policy.dd_quantile, policy.seed))
        dd_factor_sample = dd_boot / dd_obs if dd_obs > 0 else float('nan')
        dd_factor_intra = (abs(float(kpis.daily_max_dd)) / dd_obs) if dd_obs > 0 else 1.0
        dd_factor_intra = max(dd_factor_intra, 1.0) * policy.intraday_pad

        dd_adj = dd_boot * dd_factor_intra
        if dd_adj < dd_obs:
            # A stress shallower than one that already happened is not a stress. When this
            # binds, the bootstrap is understating and that is worth knowing.
            dd_adj, dd_floor_binding = dd_obs, True
        if dd_adj > 1.0:
            dd_adj, dd_clamped = 1.0, True

        for b in policy.block_months_sensitivity:
            block_sens[int(b)] = abs(bootstrap_drawdown_quantile(
                kpis.monthly_returns, policy.horizon_months, b, policy.n_bootstrap,
                policy.dd_quantile, policy.seed))
    except NotCalculable as exc:
        dd_error = str(exc)

    # --- CAP 1 ------------------------------------------------------------------------- #
    caps = {}
    m_used = accrued = float('nan')
    if dd_error is not None:
        for label in ('margin_survival', 'margin_survival_crisis'):
            caps[label] = Cap(label, None, dd_error)
    else:
        try:
            m_resolved = resolve_maintenance_margin(policy, kpis)
            m_base = min(m_resolved, MAINTENANCE_CEILING)
            if m_resolved > MAINTENANCE_CEILING:
                notes.append(
                    f'Maintenance requirement resolved to {m_resolved:.2f} — at or above 1.00, '
                    f'a position nothing may be borrowed against. Clamped to '
                    f'{MAINTENANCE_CEILING:g} so CAP 1 answers "no margin" rather than dividing '
                    f'by zero; the honest reading of this row is 1.00x by construction.')
            m_crisis = min(m_base * policy.crisis_margin_multiple, MAINTENANCE_CEILING)
            accrued_base = (accrued_interest_fraction(policy.borrow_rate_annual,
                                                      kpis.max_dd_months)
                            if policy.use_interest_accrual else 0.0)
            for label, m, c in (
                ('margin_survival', m_base, 0.0),
                ('margin_survival_accrual', m_base, accrued_base),
                ('margin_survival_crisis', m_crisis, 0.0),
                ('margin_survival_crisis_accrual', m_crisis, accrued_base),
            ):
                f, _ = cap_margin_survival(dd_adj, m, policy.safety_factor_k, c)
                caps[label] = Cap(
                    label, f,
                    f'survives {policy.safety_factor_k:g} x DD_adj={dd_adj:.2%} at m={m:.2f}, '
                    f'accrued={c:.2%}',
                    binds=False)
            # Exactly one CAP 1 variant binds, and which one is a named policy choice.
            retained = ('margin_survival_crisis_accrual' if policy.use_crisis_margin
                        else 'margin_survival_accrual')
            if not policy.use_interest_accrual:
                retained = retained.replace('_accrual', '')
            caps = {k: (replace(v, binds=(k == retained)) if k.startswith('margin_survival')
                        else v) for k, v in caps.items()}
            m_used = m_crisis if policy.use_crisis_margin else m_base
            accrued = accrued_base
            notes.append(f'CAP 1 retained variant: {retained}')
        except NotCalculable as exc:
            for label in ('margin_survival', 'margin_survival_accrual',
                          'margin_survival_crisis', 'margin_survival_crisis_accrual'):
                caps[label] = Cap(label, None, str(exc))

    # --- Sharpe chain and CAP 2 -------------------------------------------------------- #
    r = pd.Series(kpis.monthly_returns).dropna()
    skew = float(r.skew()) if len(r) > 2 else float('nan')
    kurt = float(r.kurt()) + 3.0 if len(r) > 3 else float('nan')
    # `r.corr(r.shift(1))` rather than `r.autocorr(1)`: identical value, and it does not depend
    # on a convenience method surviving the next pandas major.
    ac1 = float(r.corr(r.shift(1))) if len(r) > 2 else float('nan')
    sr_obs = float(kpis.sharpe)
    sr_defl = sr_lower = sr_used = float('nan')
    dsr = float('nan')
    try:
        sr_defl = sr_obs - expected_max_sharpe(policy.n_trials, policy.trial_sharpe_sd)
        dsr = deflated_sharpe_probability(sr_obs, r, policy.n_trials, policy.trial_sharpe_sd)
    except NotCalculable as exc:
        assumptions.append(f'No multiple-testing haircut applied: {exc}. The reported Sharpe '
                           'is therefore the selected one, and selection makes it optimistic.')
    try:
        sr_lower = sr_obs - policy.sharpe_z * sharpe_standard_error(sr_obs, r)
    except NotCalculable as exc:
        notes.append(f'no Sharpe standard error: {exc}')
    candidates = [s for s in (sr_defl, sr_lower) if not math.isnan(s)]
    if candidates:
        sr_used = min(candidates)          # the more severe of the two haircuts

    if math.isnan(sr_used):
        caps['kelly'] = Cap('kelly', None, 'no haircut Sharpe could be computed, so no Kelly '
                                           'gate; used as a gate it must not be guessed')
    elif policy.borrow_rate_annual is None or policy.risk_free_annual is None:
        caps['kelly'] = Cap('kelly', None, 'Kelly needs the borrow spread s = r_b - r_f, and '
                                           'at least one of the two rates was not given')
    else:
        spread = float(policy.borrow_rate_annual) - float(policy.risk_free_annual)
        try:
            f_k = cap_kelly(sr_used, kpis.vol, spread, kpis.offensive_weight_mean)
            caps['kelly'] = Cap('kelly', f_k,
                                f'(mu_e - s*w_off)/sigma^2 at SR_used={sr_used:.3f}, '
                                f's={spread:.2%}, w_off={kpis.offensive_weight_mean:.2f}')
        except NotCalculable as exc:
            caps['kelly'] = Cap('kelly', None, str(exc))

    # --- CAP 3, diagnostic only -------------------------------------------------------- #
    try:
        positive, margin_pp = carry_diagnostic(kpis.cagr, policy.borrow_rate_annual)
        caps['carry_diagnostic'] = Cap(
            'carry_diagnostic', float('inf') if positive else 1.0,
            f'CAGR {kpis.cagr:.2%} {"exceeds" if positive else "does NOT exceed"} borrow '
            f'{policy.borrow_rate_annual:.2%} by {margin_pp:+.2%} — diagnostic; strictly '
            f'redundant with the Kelly gate and never binds',
            binds=False)
    except NotCalculable as exc:
        caps['carry_diagnostic'] = Cap('carry_diagnostic', None, str(exc), binds=False)

    # --- CAP 4 ------------------------------------------------------------------------- #
    if policy.borrowing_capacity_leverage is None:
        caps['borrowing_capacity'] = Cap(
            'borrowing_capacity', None,
            'no capacity given. It must be the leverage the TAXABLE account can reach: '
            'registered accounts do not permit margin, so a household total would describe an '
            'unattainable portfolio')
        assumptions.append('Borrowing capacity was not supplied, so that axis is unbounded '
                           'here. It is frequently the constraint that actually binds.')
    else:
        caps['borrowing_capacity'] = Cap(
            'borrowing_capacity', float(policy.borrowing_capacity_leverage),
            'broker / credit-line parameter (taxable account only)')

    # --- the minimum, and what it means ------------------------------------------------ #
    cap1 = next((c for c in caps.values()
                 if c.name.startswith('margin_survival') and c.binds), None)
    if cap1 is None or not cap1.calculable:
        why = (caps.get('margin_survival') or Cap('margin_survival', None, 'unavailable')).basis
        return MarginRecommendation(
            model=kpis.name, recommended_leverage=None,
            binding_constraint='non calculable (CAP 1 — margin survival)',
            leverage_justified=False, caps=caps,
            dd_observed=-dd_obs, dd_bootstrapped=-dd_boot, dd_factor_sample=dd_factor_sample,
            dd_factor_intraperiod=dd_factor_intra, dd_adj=dd_adj,
            dd_floor_binding=dd_floor_binding, dd_clamped_at_one=dd_clamped,
            max_dd_months=int(kpis.max_dd_months), block_sensitivity=block_sens,
            sharpe_observed=sr_obs, sharpe_deflated=sr_defl, sharpe_lower_bound=sr_lower,
            sharpe_used=sr_used, deflated_sharpe_probability=dsr,
            return_skew=skew, return_kurtosis=kurt, return_autocorr_1=ac1,
            implied_liquidation_threshold=float('nan'),
            threshold_over_observed_maxdd=float('nan'),
            maintenance_margin_used=m_used, interest_accrued_fraction=accrued,
            policy=policy,
            invalidating_assumptions=tuple(assumptions),
            notes=tuple(notes + [
                'CAP 1 is the principal constraint. With it non-calculable the whole result is '
                'non-calculable: returning the minimum of the surviving caps would answer a '
                'cheerful number from a credit-line parameter while leaving the question of '
                'survival unanswered. ' + why]))

    kelly = caps['kelly']
    if kelly.calculable and kelly.value < 1.0:
        # The gate, and it stops the walk. Not a cap of 0.83x — a refusal.
        return MarginRecommendation(
            model=kpis.name, recommended_leverage=1.0,
            binding_constraint='kelly_gate — no leverage justified',
            leverage_justified=False, caps=caps,
            dd_observed=-dd_obs, dd_bootstrapped=-dd_boot, dd_factor_sample=dd_factor_sample,
            dd_factor_intraperiod=dd_factor_intra, dd_adj=dd_adj,
            dd_floor_binding=dd_floor_binding, dd_clamped_at_one=dd_clamped,
            max_dd_months=int(kpis.max_dd_months), block_sensitivity=block_sens,
            sharpe_observed=sr_obs, sharpe_deflated=sr_defl, sharpe_lower_bound=sr_lower,
            sharpe_used=sr_used, deflated_sharpe_probability=dsr,
            return_skew=skew, return_kurtosis=kurt, return_autocorr_1=ac1,
            implied_liquidation_threshold=1.0, threshold_over_observed_maxdd=1.0 / dd_obs,
            maintenance_margin_used=m_used, interest_accrued_fraction=accrued,
            policy=policy, invalidating_assumptions=tuple(assumptions),
            notes=tuple(notes + [
                f'f_kelly = {kelly.value:.3f} < 1 after the haircut, so borrowed money is '
                f'expected to reduce compound growth. The walk stops here; the other caps are '
                f'reported but no leverage is justified at any of them.']))

    binding = min((c for c in caps.values() if c.binds and c.calculable),
                  key=lambda c: c.value)
    f_rec = float(binding.value)
    threshold = liquidation_threshold(f_rec, m_used, accrued)
    ratio = threshold / dd_obs if dd_obs > 0 else float('inf')

    # The invariant that says the pipeline is intact. f <= f_cap1 => d_max(f) >= k*DD_adj >=
    # k*DD_observed, so the ratio is provably >= k — UNLESS the target decline was clamped at
    # 100%, in which case the bound is 1/DD_observed. A value below both is a bug, by
    # construction rather than by convention. Asserted in `test_threshold_ratio_invariant`.
    if not dd_clamped and ratio + 1e-9 < policy.safety_factor_k:
        notes.append(f'INVARIANT VIOLATED: threshold/observed = {ratio:.3f} < k='
                     f'{policy.safety_factor_k:g}. This is a bug, not a result.')

    return MarginRecommendation(
        model=kpis.name, recommended_leverage=f_rec,
        binding_constraint=binding.name, leverage_justified=f_rec > 1.0, caps=caps,
        dd_observed=-dd_obs, dd_bootstrapped=-dd_boot, dd_factor_sample=dd_factor_sample,
        dd_factor_intraperiod=dd_factor_intra, dd_adj=dd_adj,
        dd_floor_binding=dd_floor_binding, dd_clamped_at_one=dd_clamped,
        max_dd_months=int(kpis.max_dd_months), block_sensitivity=block_sens,
        sharpe_observed=sr_obs, sharpe_deflated=sr_defl, sharpe_lower_bound=sr_lower,
        sharpe_used=sr_used, deflated_sharpe_probability=dsr,
        return_skew=skew, return_kurtosis=kurt, return_autocorr_1=ac1,
        implied_liquidation_threshold=threshold, threshold_over_observed_maxdd=ratio,
        maintenance_margin_used=m_used, interest_accrued_fraction=accrued,
        policy=policy, invalidating_assumptions=tuple(assumptions), notes=tuple(notes))


def compare_table(recommendations):
    """Recommendations aligned side by side, one row per model.

    Alignability is a stated requirement, not a convenience: the whole value of the module is
    reading `binding_constraint` DOWN the column and seeing that (say) every wrap is capped by
    its own drawdown while every unlevered family is capped by the credit line. That comparison
    is only legitimate if the parameter block is identical across rows, so this refuses to
    print rows whose policies differ rather than lining up numbers that are not comparable.
    """
    rows = [r for r in recommendations if r is not None]
    if not rows:
        return '(nothing to compare)'
    prints = {r.policy.preference_fingerprint() for r in rows}
    if len(prints) > 1:
        raise ValueError(
            'compare_table was given recommendations produced under DIFFERENT PREFERENCE '
            'parameters (k, quantile, horizon, rates, seed). Lining those up would invite '
            'exactly the comparison they do not support. The maintenance mapping is allowed to '
            'differ — that is a property of the holdings, not a choice.')

    head = (f'{"model":<26} {"f":>7} {"binding":<32} {"m":>5} {"cap1":>7} {"cap1_cri":>8} '
            f'{"kelly":>8} {"cap":>6} {"DD_obs":>8} {"DD_adj":>8} {"liq":>8} {"liq/DD":>7}')
    out = [head, '-' * len(head)]
    for r in sorted(rows, key=lambda x: (x.recommended_leverage is None,
                                         -(x.recommended_leverage or 0.0))):
        def cell(name, width=7):
            v = r.cap_value(name)
            if v is None:
                return f'{"n/c":>{width}}'
            return f'{"inf":>{width}}' if math.isinf(v) else f'{v:>{width}.3f}'
        f = 'n/c' if r.recommended_leverage is None else f'{r.recommended_leverage:.3f}'
        m = r.maintenance_margin_used
        m_cell = f'{"n/c":>5}' if m is None or math.isnan(m) else f'{m:>5.2f}'
        out.append(
            f'{r.model:<26} {f:>7} {r.binding_constraint[:32]:<32} {m_cell} '
            f'{cell("margin_survival")} {cell("margin_survival_crisis", 8)} '
            f'{cell("kelly", 8)} {cell("borrowing_capacity", 6)} '
            f'{r.dd_observed:>8.2%} {r.dd_adj:>8.2%} '
            f'{r.implied_liquidation_threshold:>8.2%} '
            f'{r.threshold_over_observed_maxdd:>7.2f}')
    out.append('')
    out.append(f'k = {rows[0].policy.safety_factor_k:g} x DD_adj    '
               f'seed = {rows[0].policy.seed}    '
               f'horizon = {rows[0].policy.horizon_months} months    '
               f'quantile = {rows[0].policy.dd_quantile:g}')
    out.append('n/c = non calculable, never a silent default. liq = decline in the positions '
               'held that triggers the call.')
    return '\n'.join(out)


def evidence_table(recommendations):
    """The inputs `compare_table` consumes and does not show. One row per model.

    Added 2026-08-01 after an audit of this repository's own output found that every number
    below was computed on every run and rendered nowhere. `report()` formats most of them and
    is called from exactly one place: a test. `compare_table` — the only renderer with a
    consumer — prints the twelve columns needed to ACT and drops the ones needed to DOUBT.

    Three of the omissions were self-defeating rather than merely thin:

    * `deflated_sharpe_probability` is the module's answer to "is this Sharpe real, given
      that N of them were searched" — the multiple-testing correction the whole selection
      apparatus exists to apply — and it reached no screen at all.
    * `block_sensitivity` exists, in this module's own words, "so the one genuinely arbitrary
      choice is visible instead" of hidden. It was hidden.
    * `dd_factor_sample` and `dd_factor_intraperiod` are the two factors the adjusted drawdown
      is deliberately COMPOSED of, rather than being one opaque daily bootstrap. Showing only
      the product turns a decomposition back into the opaque number it was built to replace.
    """
    rows = [r for r in recommendations if r is not None]
    if not rows:
        return '(nothing to compare)'

    blocks = sorted({b for r in rows for b in (r.block_sensitivity or {})})
    bhead = ''.join(f'{f"DD@{b}mo":>8}' for b in blocks)
    head = (f'{"model":<26} {"SR_obs":>7} {"SR_def":>7} {"SR_low":>7} {"SR_use":>7} '
            f'{"P(DSR)":>7} {"DD_obs":>8} {"DD_boot":>8} {"xsample":>8} {"xintra":>7} '
            f'{"DD_adj":>8} {"mo":>4}{bhead}')
    out = [head, '-' * len(head)]

    def num(v, width, spec):
        return (f'{"n/c":>{width}}' if v is None or (isinstance(v, float) and math.isnan(v))
                else f'{v:>{width}{spec}}')

    for r in sorted(rows, key=lambda x: (x.recommended_leverage is None,
                                         -(x.recommended_leverage or 0.0))):
        sens = r.block_sensitivity or {}
        cells = ''.join(num(-abs(sens[b]) if b in sens else None, 8, '.2%') for b in blocks)
        out.append(
            f'{r.model:<26} {num(r.sharpe_observed, 7, ".3f")} '
            f'{num(r.sharpe_deflated, 7, ".3f")} {num(r.sharpe_lower_bound, 7, ".3f")} '
            f'{num(r.sharpe_used, 7, ".3f")} '
            f'{num(r.deflated_sharpe_probability, 7, ".1%")} '
            f'{num(r.dd_observed, 8, ".2%")} {num(r.dd_bootstrapped, 8, ".2%")} '
            f'{num(r.dd_factor_sample, 8, ".3f")} {num(r.dd_factor_intraperiod, 7, ".3f")} '
            f'{num(r.dd_adj, 8, ".2%")} {r.max_dd_months:>4}{cells}')

    out += [
        '',
        'SR_obs  the Sharpe this record printed.   SR_def  minus E[max SR] across the trials '
        'actually run',
        '        (Bailey & Lopez de Prado) — what pure search over that many variants produces '
        'from no skill.',
        'SR_low  observed minus one Mertens higher-moment standard error — SAMPLING error, a '
        'different',
        '        objection from selection.  SR_use = min(SR_def, SR_low): whichever doubt bites '
        'harder.',
        'P(DSR)  probability the true Sharpe exceeds the selection threshold. BELOW ~95% the '
        'edge is not',
        '        established at this sample size, whatever the leaderboard says.',
        'DD_adj = DD_boot x xsample x xintra. `xsample` carries the deeper bootstrapped '
        'quantile onto the',
        '        observed path; `xintra` is how much worse the DAILY path was than the '
        'month-end series that',
        '        every other table in this repository is built from. A margin call is '
        'continuous; month-ends',
        '        are not.  DD@Nmo re-runs the quantile at other block lengths, so the one '
        'arbitrary choice',
        '        in the bootstrap is a column you can read rather than an assumption you have '
        'to trust.',
    ]
    return '\n'.join(out)


# --------------------------------------------------------------------------------------- #
#  path simulation — so the drift argument is tested rather than asserted
# --------------------------------------------------------------------------------------- #

#: How target leverage is maintained. Named because it is a DESIGN DECISION with a measurable
#: consequence, not an implementation detail.
#:
#:   reset_monthly  what `ledger.py:340` does: notional back to f*equity every period,
#:                  unconditionally. Protective — restoring the target after a loss SELLS.
#:   none           never rebalanced. `f_eff` ratchets up through a decline; this is the case
#:                  the closed-form `d_max` describes.
#:   bands          rebalance only when `f_eff` leaves [lo, hi]. Lower turnover, larger
#:                  residual.
#:   one_sided_up   re-lever on gains, never de-lever on losses. A plausible implementation of
#:                  "maintain target exposure" and a liquidation engine. It exists here so a
#:                  test can prove it (F5).
POLICIES = ('reset_monthly', 'none', 'bands', 'one_sided_up')


@dataclass
class PathOutcome:
    liquidated: bool
    step: int = None                  # index of the period in which the call fired
    trigger: str = None              # 'period_end' or 'intra_period'
    decline_at_call: float = None    # cumulative decline in the POSITIONS at that moment
    effective_leverage: list = field(default_factory=list)
    equity: list = field(default_factory=list)


def simulate_margin_path(returns, leverage, maintenance_margin, policy='reset_monthly',
                         borrow_rate_annual=0.0, periods_per_year=12, bands=(0.9, 1.1),
                         intra_period_lows=None, crisis_margin_from=None,
                         crisis_margin_multiple=1.5):
    """Walk a return path at `leverage` and report the first margin call, if any.

    The maintenance test happens INSIDE the period, before any rebalance. That ordering is the
    whole point: the call is continuous and the reset is monthly, so a position can be closed
    out at a level the month-end series never shows.

    `intra_period_lows[i]` is the worst fractional level reached inside period `i` relative to
    its start (0.85 = down 15% at the trough, recovering by the close). It is what makes the
    sub-monthly excursion visible, and nothing in `ledger.run_ledger` can see it.

    `crisis_margin_from` raises `m` by `crisis_margin_multiple` from that period onwards — the
    real failure mode, `m` rising while the market falls.
    """
    if policy not in POLICIES:
        raise ValueError(f'policy must be one of {sorted(POLICIES)}')
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    m0 = float(maintenance_margin)
    f_target = float(leverage)
    per_period_rate = float(borrow_rate_annual) / periods_per_year

    equity, assets = 1.0, f_target
    debt = assets - equity
    peak_assets = assets
    out = PathOutcome(liquidated=False)

    for i, ret in enumerate(r):
        m = m0 * (crisis_margin_multiple
                  if crisis_margin_from is not None and i >= crisis_margin_from else 1.0)

        # 1. intra-period excursion, tested BEFORE the period's close
        if intra_period_lows is not None and i < len(intra_period_lows):
            low = float(intra_period_lows[i])
            a_low = assets * low
            d_low = debt * (1.0 + per_period_rate)
            if a_low - d_low < m * a_low:
                out.liquidated, out.step, out.trigger = True, i, 'intra_period'
                out.decline_at_call = 1.0 - a_low / peak_assets
                return out

        # 2. the period's close, then interest
        assets *= (1.0 + ret)
        debt *= (1.0 + per_period_rate)
        equity = assets - debt
        peak_assets = max(peak_assets, assets)
        out.effective_leverage.append(assets / equity if equity > 0 else math.inf)
        out.equity.append(equity)

        if equity < m * assets:
            out.liquidated, out.step, out.trigger = True, i, 'period_end'
            out.decline_at_call = 1.0 - assets / peak_assets
            return out

        # 3. maintain target leverage, per policy
        f_eff = assets / equity
        if policy == 'reset_monthly':
            assets, debt = f_target * equity, f_target * equity - equity
        elif policy == 'bands':
            lo, hi = bands
            if not (f_target * lo <= f_eff <= f_target * hi):
                assets, debt = f_target * equity, f_target * equity - equity
        elif policy == 'one_sided_up':
            # Re-lever on gains, never de-lever on losses. The bug, made explicit.
            if f_eff < f_target:
                assets, debt = f_target * equity, f_target * equity - equity
        # 'none': carry the drifted book forward untouched

    return out
