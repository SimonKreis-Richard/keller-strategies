"""
The consumer `common/margin_sizing.py` did not have.

WHY THIS FILE EXISTS. `margin_sizing` answers "what margin leverage does this model's record
support?" with four caps, a bootstrapped drawdown and a derived Sharpe haircut — and until
2026-07-31 nothing called it. The registry table in LEVERAGE.md section 5.4 was produced by an
ad-hoc script that was not kept, which means the repository's own answer to its central question
was not reproducible from the repository. This module is the missing driver: it turns a
`run_backtest` metrics list into recommendations, and it is what the CLI report and the
dashboard's "Max margin" column both call.

It deliberately holds NO arithmetic. Every formula stays in `margin_sizing`; what lives here is
the assembly — which broker assumptions to use in the absence of a broker, where the trial count
comes from, and how a recommendation is rendered in one table cell.

--------------------------------------------------------------------------------------------
WHAT YOU NEED FROM A BROKER TO GET AN ANSWER: LESS THAN IT LOOKS
--------------------------------------------------------------------------------------------
The question this module was written to settle is whether a recommendation requires the owner's
actual borrowing terms — a credit line and a negotiated rate that are not known and will not be
for some time. Cap by cap:

  * CAP 4, borrowing capacity — NOT REQUIRED. Left unsupplied, the axis is reported unbounded
    and the fact is carried in `invalidating_assumptions` on every row. It is frequently the cap
    that actually binds in real life, so its absence is stated, never defaulted.
  * The BORROW RATE — required for CAP 2 and for the interest that capitalises during a
    drawdown, but it was already a parameter of this repository (`MARGIN_BORROW_RATE`, 6%) and
    it is an assumption you vary rather than a fact you look up. Its effect is legible: the
    spread `r_b - r_f` is what the Kelly gate divides by, and at a 4-5pp spread the gate is what
    stops most of the registry.
  * The MAINTENANCE MARGIN — required, and this is the only input with no repository default.
    `margin_sizing` refuses to invent one, correctly: 25% is Reg-T for unlevered equity and
    wrong by roughly 3x for a book holding LETFs. What this module supplies instead is a
    REGULATORY-CONVENTION table (below), exposed as one number the user can move.

So the honest summary is: **a first-order answer needs no broker data at all**, and the caps
that would need it say so on their own rows instead of quietly using a plausible number.

THE RULE OF THUMB, IN ONE LINE. Everything else is refinement of CAP 1, which is:

    f = 1 / (m + k * DD * (1 - m))

leverage `f` survivable at maintenance requirement `m` if you insist on surviving `k` times a
drawdown of `DD`. At m=0.45 and a 25% stressed drawdown with k=3, that is 1.2x. Anyone who
expected 1.3x to be comfortably available should read that line before the table.

--------------------------------------------------------------------------------------------
THE MAINTENANCE TABLE, AND WHY IT IS DERIVED RATHER THAN LISTED
--------------------------------------------------------------------------------------------
`MAINTENANCE_BASE` = 0.30 for an ordinary long ETF position. FINRA Rule 4210(c) sets the FLOOR
at 25% of market value; house requirements on broad-market ETFs are commonly higher, and 30% is
the assumption taken here. It is deliberately above the regulatory minimum and deliberately a
parameter: it is the number to replace first when a real account exists.

For a leveraged product the requirement is the base rate times the FUND'S OWN MULTIPLE — the
convention brokers publish for LETFs, and the reason a 3x book cannot be margined at all
(m = 0.90, and 1.35 once the crisis multiple is applied, i.e. above 1). That multiple is read
off `LETFMapper.MAP_2X` / `MAP_3X`, not from a hand-kept list here, so a product added to either
map inherits its maintenance rate the same day and cannot be forgotten.

Both numbers are ASSUMPTIONS about a broker nobody has chosen yet. They are named in the
policy block that every recommendation echoes, so a reader can see exactly what was assumed
without reading this file.

--------------------------------------------------------------------------------------------
WHAT THIS MODULE REFUSES TO DO
--------------------------------------------------------------------------------------------
Skip a row. An entry whose KPIs are incomplete comes back as a recommendation of `None` with the
reason attached, and the caller renders "n/c". An entry too short for the drawdown bootstrap
does the same. There is no fallback heuristic anywhere in the chain, because a made-up markup on
a record that never saw 2008 is exactly the failure mode `margin_sizing` was written against.
"""

from dataclasses import dataclass, replace

from common import margin_sizing as ms
from common.letf_mapper import LETFMapper
from common.selection import trial_sharpe_spread

#: Bootstrap seed. Fixed and stated: `margin_sizing` refuses an unseeded bootstrap, and a seed
#: that moved between runs would make the column change without the data changing.
SEED = 20260731

#: House maintenance requirement assumed for an ordinary long ETF position. See the module
#: docstring — a parameter, above the 25% regulatory floor, and the first thing to replace.
MAINTENANCE_BASE = 0.30

#: {leveraged ticker: the fund's own multiple}, derived from the two admissible maps rather than
#: listed. `tests/test_leverage_advice.py` asserts it covers every image in both.
PRODUCT_MULTIPLE = {img: mult
                    for mapping, mult in ((LETFMapper.MAP_2X, 2.0), (LETFMapper.MAP_3X, 3.0))
                    for img in mapping.values()}

#: How a binding constraint is named in a table cell. The long form stays on the recommendation;
#: this is what fits in a column, and it must not lose the distinction between "the broker would
#: have closed you out" and "borrowed money would not have helped".
CONSTRAINT_LABELS = {
    'margin_survival': 'margin call',
    'margin_survival_accrual': 'margin call',
    'margin_survival_crisis': 'margin call',
    'margin_survival_crisis_accrual': 'margin call',
    'kelly': 'Kelly',
    'borrowing_capacity': 'credit line',
}


def product_multiple(ticker):
    """The leverage multiple of a tradeable product: 1.0 for an ordinary fund."""
    return float(PRODUCT_MULTIPLE.get(ticker, 1.0))


def maintenance_map(held_tickers, base=MAINTENANCE_BASE):
    """{ticker: maintenance requirement} over exactly the tickers a book can hold.

    Covers every holding by construction, which is what `resolve_maintenance_margin` demands
    before it will produce a cap — a book whose requirement is unknown for any one holding has
    no calculable cap, and that refusal is the point rather than an obstacle.
    """
    return {t: float(base) * product_multiple(t) for t in held_tickers}


@dataclass(frozen=True)
class Advice:
    """Recommendations for one backtest run, plus what could not be advised on and why."""

    by_name: dict            # {strategy name: MarginRecommendation}
    policy: object           # the shared MarginPolicy, maintenance mapping excluded
    skipped: tuple           # ((name, reason), ...) — entries that never reached the module
    run_leverage: float      # LEVERAGE_FACTOR of the run these KPIs came from
    maintenance_base: float  # the assumed house rate the per-ticker mapping was built from

    @property
    def recommendations(self):
        """Every recommendation, in the order the entries were measured."""
        return list(self.by_name.values())

    def table(self):
        """`margin_sizing.compare_table` over the lot — aligned, one row per model."""
        return ms.compare_table(self.recommendations) if self.by_name else '(nothing to size)'

    def evidence(self):
        """The inputs behind that table — the Sharpe haircuts, the drawdown decomposition and
        the bootstrap's sensitivity to its own block length.

        Separate from `table()` rather than folded into it because they answer two different
        questions and only the first one is actionable. `table()` says what leverage survives;
        this says how much to believe it.
        """
        return (ms.evidence_table(self.recommendations) if self.by_name
                else '(nothing to size)')


def build_policy(metrics_data, k=3.0, borrow_rate_annual=0.06,
                 capacity_leverage=None, seed=SEED):
    """The shared parameter block, with the two MEASURED inputs read off the run itself.

    `n_trials` and `trial_sharpe_sd` come from `selection.trial_sharpe_spread`, so the Sharpe
    haircut is derived from this suite's own cross-sectional spread rather than from a chosen
    coefficient. `risk_free_annual` is the rate the ranked rows' Sharpes were already netted
    against — asking `margin_sizing` for a borrow SPREAD measured against a different cash
    series than the Sharpe would silently mix two rates.

    `maintenance_margin` is left None here and filled per model in `advise`: it is a property of
    the holdings, not a preference, and `compare_table` allows exactly that one field to differ.
    """
    n_trials, sd = trial_sharpe_spread(metrics_data)
    ranked = [d for d in metrics_data if d.get('in_ranked_window')]
    rf_row = ranked[0] if ranked else (metrics_data[0] if metrics_data else None)
    rf = float(rf_row['rf_annual']) if rf_row and rf_row.get('rf_annual') is not None else None
    return ms.MarginPolicy(
        safety_factor_k=float(k),
        maintenance_margin=None,
        borrow_rate_annual=float(borrow_rate_annual),
        risk_free_annual=rf,
        borrowing_capacity_leverage=capacity_leverage,
        seed=int(seed),
        n_trials=n_trials,
        trial_sharpe_sd=sd,
    )


#: KPI keys `run_backtest` must attach for an entry to be sizeable. Listed rather than probed one
#: by one so the reason given to the user names the missing field.
REQUIRED_KPIS = ('returns', 'max_dd', 'max_dd_months', 'sharpe', 'vol', 'cagr', 'rf_annual',
                 'offensive_weight_mean', 'held_tickers')


def advise(metrics_data, k=3.0, borrow_rate_annual=0.06, maintenance_base=MAINTENANCE_BASE,
           capacity_leverage=None, seed=SEED, run_leverage=1.0):
    """Size every entry in a `run_backtest` metrics list. Returns an `Advice`.

    `daily_max_dd` is the one KPI allowed to be absent: it makes CAP 1 non-calculable, which
    `margin_sizing` reports as a row rather than an omission. Everything in `REQUIRED_KPIS` is
    structural — without it the entry never reaches the module and is named in `skipped`.
    """
    policy = build_policy(metrics_data, k=k, borrow_rate_annual=borrow_rate_annual,
                          capacity_leverage=capacity_leverage, seed=seed)
    by_name, skipped = {}, []
    for d in metrics_data:
        missing = [key for key in REQUIRED_KPIS if d.get(key) is None]
        if missing:
            skipped.append((d.get('name', '?'),
                            f'no {", ".join(missing)} on the metrics entry'))
            continue
        held = tuple(d['held_tickers'])
        if not held:
            skipped.append((d['name'], 'declares no held tickers, so no maintenance '
                                       'requirement can be looked up'))
            continue
        kpis = ms.ModelKPIs(
            name=d['name'],
            monthly_returns=d['returns'],
            max_dd=float(d['max_dd']),
            max_dd_months=int(d['max_dd_months']),
            sharpe=float(d['sharpe']),
            vol=float(d['vol']),
            cagr=float(d['cagr']),
            rf_annual=float(d['rf_annual']),
            offensive_weight_mean=float(d['offensive_weight_mean']),
            holds_leveraged_product=bool(d.get('holds_leveraged_product')),
            held_tickers=held,
            daily_max_dd=(None if d.get('daily_max_dd') is None
                          else float(d['daily_max_dd'])),
        )
        by_name[d['name']] = ms.recommend_leverage(
            kpis, replace(policy, maintenance_margin=maintenance_map(held, maintenance_base)))
    return Advice(by_name=by_name, policy=policy, skipped=tuple(skipped),
                  run_leverage=float(run_leverage),
                  maintenance_base=float(maintenance_base))


def short_constraint(rec):
    """The binding constraint in a few characters, for a table column.

    'no leverage' is NOT the same answer as 'margin call' at 1.00x, and both print 1.00x: the
    first says borrowed money would have reduced compound growth, the second that the account
    would have been closed out. Collapsing them into "1.00x" would throw away the only part of
    the row that tells you what to do about it.
    """
    if rec is None:
        return 'n/c'
    name = rec.binding_constraint
    if name.startswith('kelly_gate'):
        return 'no leverage'
    if name.startswith('non calculable'):
        return 'n/c'
    return CONSTRAINT_LABELS.get(name, name)


def cell(rec):
    """(leverage text, constraint text) for one table row."""
    if rec is None or rec.recommended_leverage is None:
        return 'n/c', short_constraint(rec)
    return f'{rec.recommended_leverage:.2f}x', short_constraint(rec)


def headline(advice):
    """One line naming the best sizeable entry, or why there is none.

    Written as a sentence rather than a number because the answer this module usually gives is
    "essentially none, and here is the one that gets closest" — which is a finding, and reads as
    a bug when it appears as a column of 1.00x with no comment.
    """
    sizeable = [r for r in advice.recommendations
                if r.recommended_leverage is not None and r.recommended_leverage > 1.0]
    if not sizeable:
        return ('No entry in this run supports margin above 1.00x at '
                f'k={advice.policy.safety_factor_k:g} x drawdown. That is the finding, not a '
                f'missing calculation — see the derivation below for which cap stopped each.')
    best = max(sizeable, key=lambda r: r.recommended_leverage)
    return (f'{len(sizeable)} entr(ies) support margin above 1.00x; the highest is '
            f'{best.model} at {best.recommended_leverage:.2f}x, capped by '
            f'{short_constraint(best)}.')


def assumption_lines(advice):
    """The parameter block as text — what was assumed, in the order it matters."""
    p = advice.policy
    lines = [
        f'k = {p.safety_factor_k:g} x the stressed drawdown must be survivable before the '
        f'broker liquidates (the only preference parameter).',
        f'Maintenance requirement: {advice.maintenance_base:.0%} assumed on an ordinary ETF '
        f'position, x the fund\'s own multiple for a leveraged product '
        f'(2x -> {advice.maintenance_base * 2:.0%}, 3x -> {advice.maintenance_base * 3:.0%}), '
        f'x{p.crisis_margin_multiple:g} again for the crisis variant, which is the one '
        f'reported. FINRA Rule 4210(c) floors the base at 25%; house rates are higher. Not '
        f'your broker\'s numbers — nobody has one yet.',
        f'Borrow rate {p.borrow_rate_annual:.2%}/yr against a realised cash rate of '
        f'{p.risk_free_annual:.2%} — the spread is what the Kelly gate charges.',
        f'Drawdown stressed to the {p.dd_quantile:.0%} quantile over {p.horizon_months} months '
        f'(stationary block bootstrap, {p.n_bootstrap} paths, seed {p.seed}), then multiplied '
        f'by the MEASURED daily-path uplift of the same held allocation.',
    ]
    if p.n_trials:
        lines.append(f'Sharpe haircut derived from this suite: {p.n_trials} selection trials '
                     f'with a cross-sectional spread of {p.trial_sharpe_sd:.3f}. That is '
                     f'THIS RUN\'s population: selecting a subset of the registry shrinks '
                     f'the trial count and MILDENS the haircut, but the search that produced '
                     f'the pick was the full registry — a partial run flatters (AUD-06).')
    else:
        lines.append('No multiple-testing haircut: too few trials to measure a spread. The '
                     'Sharpe used is therefore the selected one, and selection flatters it.')
    if p.borrowing_capacity_leverage is None:
        lines.append('Borrowing capacity NOT supplied, so that axis is unbounded here. In a '
                     'real account it is frequently the cap that binds, and it must be the '
                     'TAXABLE account\'s equity — registered accounts do not permit margin.')
    if advice.run_leverage != 1.0:
        lines.append(f'This run is already levered at {advice.run_leverage:g}x, so every row '
                     f'below sizes margin ON TOP of that. Margin on margin is not a portfolio '
                     f'anyone holds — read the column at 1x.')
    return lines
