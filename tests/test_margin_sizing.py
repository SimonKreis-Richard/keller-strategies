"""Margin-sizing tests.

Two kinds. The first block checks the ARITHMETIC — the closed forms, their inverses, the
worked numbers in the module docstring, and the claim that CAP 3 can never bind. Those are
provable, so they are asserted exactly.

The second block is one test per FAILURE MODE named in the spec. Each is written to fail if the
mitigation is removed, and where a failure mode can only be *reduced* rather than eliminated
(F2) the test says so in its own docstring rather than pretending otherwise.

F1  scalar maintenance margin against a book holding LETFs
F2  short or benign sample understating the drawdown
F3  bootstrap blocks destroying regime persistence
F4  interest accrual ignored
F5  a one-sided leverage policy that re-levers on gains and never de-levers on losses
F6  the broker raising m mid-crisis
F7  the Kelly gate failing to stop a low-Sharpe model
F8  a missing input producing a number anyway
F9  comparability broken by per-model parameters
F10 non-deterministic bootstrap
F11 the intra-period uplift measured on the wrong series
F12 borrowing capacity binding
F13 k * DD_adj >= 100%
F14 somebody adding an optimisation surface
F15 the threshold/observed-drawdown ratio invariant
"""

import dataclasses
import inspect
import math
import unittest

import numpy as np
import pandas as pd

from common import margin_sizing as ms
from common import metrics as metrics_mod


def _returns(n=240, mu=0.0075, sigma=0.016, seed=7, crash_at=None, crash=-0.06):
    """A synthetic monthly record, optionally with a multi-month crash written into it."""
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sigma, n)
    if crash_at is not None:
        r[crash_at:crash_at + 4] = crash
    idx = pd.date_range('2000-01-31', periods=n, freq='ME')
    return pd.Series(r, index=idx)


def _kpis(returns=None, name='SYNTH', holds_letf=False, held=('SPY',), daily_uplift=1.10,
          sharpe=None, vol=None):
    """`ModelKPIs` whose KPIs are the ones `calculate_metrics` reports for `returns`."""
    returns = _returns() if returns is None else returns
    m = metrics_mod.calculate_metrics(returns, rf=0.02)
    return ms.ModelKPIs(
        name=name, monthly_returns=returns,
        max_dd=m['max_dd'], max_dd_months=m['max_dd_months'],
        sharpe=m['sharpe'] if sharpe is None else sharpe,
        vol=m['vol'] if vol is None else vol,
        cagr=m['cagr'], rf_annual=m['rf_annual'],
        offensive_weight_mean=0.75, holds_leveraged_product=holds_letf,
        held_tickers=held,
        daily_max_dd=m['max_dd'] * daily_uplift)


def _policy(**kw):
    """A complete, calculable policy. Tests override one field at a time.

    The borrow SPREAD is 1.5pp rather than the 4pp a 6%-over-2% pairing implies. Not to be
    generous — with a 4pp spread against a 5%-vol strategy the Kelly gate refuses leverage
    outright, which is a correct result and a useless fixture: every test would then assert
    the same refusal instead of exercising the cap it is about. That the gate does bite at a
    wide spread is asserted directly in `test_f7_the_kelly_gate_stops_a_low_sharpe_model`.
    """
    base = dict(safety_factor_k=3.0, maintenance_margin=0.30, borrow_rate_annual=0.05,
                risk_free_annual=0.035, borrowing_capacity_leverage=4.0, seed=11,
                n_bootstrap=400, n_trials=19, trial_sharpe_sd=0.15, min_history_months=120)
    base.update(kw)
    return ms.MarginPolicy(**base)


def _margin_caps(rec):
    """The CAP 1 variants only. `kelly` and `borrowing_capacity` also bind — they are caps."""
    return [c.name for c in rec.caps.values() if c.name.startswith('margin_survival') and c.binds]


# ------------------------------------------------------------------------------------- #
#  the arithmetic
# ------------------------------------------------------------------------------------- #

class TestClosedForms(unittest.TestCase):

    def test_threshold_matches_the_textbook_form_without_accrual(self):
        """`d_max = (1/f - m)/(1-m)` — the form the spec was written against."""
        for f in (1.25, 1.5, 2.0, 3.0):
            for m in (0.0, 0.25, 0.30, 0.50):
                self.assertAlmostEqual(ms.liquidation_threshold(f, m),
                                       (1.0 / f - m) / (1.0 - m), places=12,
                                       msg=f'f={f} m={m}')

    def test_threshold_and_inverse_round_trip(self):
        for f in (1.1, 1.5, 2.0, 2.5, 3.0):
            for m in (0.0, 0.25, 0.30, 0.45, 0.75):
                for c in (0.0, 0.05, 0.15):
                    d = ms.liquidation_threshold(f, m, c)
                    if d <= 0:
                        continue           # no leverage survives; the inverse clamps to 1.0
                    self.assertAlmostEqual(ms.leverage_for_threshold(d, m, c), f, places=9,
                                           msg=f'f={f} m={m} c={c}')

    def test_the_worked_numbers_in_the_docstring(self):
        """m=0.30, d=0.60, r_b=6%, h=30 months. Quoted in the module docstring and in LEVERAGE.md."""
        c = ms.accrued_interest_fraction(0.06, 30)
        self.assertAlmostEqual(c, 0.15, places=12)
        self.assertAlmostEqual(ms.leverage_for_threshold(0.60, 0.30), 1.389, places=3)
        self.assertAlmostEqual(ms.leverage_for_threshold(0.60, 0.30, c), 1.322, places=3)
        self.assertAlmostEqual(ms.leverage_for_threshold(0.60, 0.45), 1.282, places=3)
        self.assertAlmostEqual(ms.leverage_for_threshold(0.60, 0.75), 1.111, places=3)

    def test_an_unlevered_book_has_no_margin_call(self):
        for m in (0.0, 0.25, 0.90):
            self.assertEqual(ms.liquidation_threshold(1.0, m), 1.0)

    def test_maintaining_leverage_survives_where_drifting_does_not(self):
        """The inversion in the request, as arithmetic.

        Restoring target leverage after a LOSS means SELLING, so it protects. The closed-form
        `d_max` is therefore already the fully-drifted answer, not the optimistic one.
        """
        path = [-0.20, -0.20]
        drifted = ms.simulate_margin_path(path, 2.0, 0.25, policy='none')
        reset = ms.simulate_margin_path(path, 2.0, 0.25, policy='reset_monthly')
        self.assertTrue(drifted.liquidated)
        self.assertFalse(reset.liquidated)
        # ... and the drifted call fires at the cumulative decline the closed form predicts.
        self.assertLess(abs(drifted.decline_at_call
                            - ms.liquidation_threshold(2.0, 0.25)), 0.05)

    def test_expected_max_sharpe_bracket_at_nineteen_trials(self):
        """N=19 gives a bracket of 1.878. The repository's own selection-trial count."""
        self.assertAlmostEqual(ms.expected_max_sharpe(19, 1.0), 1.878, places=3)
        self.assertAlmostEqual(ms.expected_max_sharpe(19, 0.25), 1.878 * 0.25, places=3)
        # More trials, bigger haircut. Monotone, or the correction is not a correction.
        self.assertLess(ms.expected_max_sharpe(5, 0.25), ms.expected_max_sharpe(50, 0.25))

    def test_negative_skew_widens_the_sharpe_interval(self):
        """Mertens' form must react to the moments. An iid formula would not."""
        rng = np.random.default_rng(3)
        sym = pd.Series(rng.normal(0.008, 0.03, 240))
        left = sym.copy()
        left.iloc[:8] = -0.16                     # a left tail, same length
        se_sym = ms.sharpe_standard_error(0.9, sym)
        se_left = ms.sharpe_standard_error(0.9, left)
        self.assertLess(left.skew(), sym.skew())
        self.assertGreater(se_left, se_sym)

    def test_carry_can_never_bind_before_the_kelly_gate(self):
        """CAP 3 is a strict subset of CAP 2, which is why it was requalified as a diagnostic.

        carry fires  <=>  mu - sigma^2/2 <= r_b  <=>  mu - r_b <= sigma^2/2
        kelly  fires  <=>  mu - r_b < sigma^2
        and sigma^2/2 < sigma^2, so the first implies the second. Swept, not argued.
        """
        for sigma in (0.05, 0.08, 0.12, 0.20, 0.35):
            for mu in np.arange(-0.02, 0.30, 0.01):
                for rb in (0.02, 0.05, 0.06, 0.09):
                    cagr_geo = mu - sigma ** 2 / 2.0
                    carry_fires = cagr_geo <= rb
                    kelly_fires = ms.cap_kelly(mu / sigma, sigma, rb, 1.0) < 1.0
                    if carry_fires:
                        self.assertTrue(kelly_fires,
                                        f'carry fired but Kelly did not: mu={mu:.3f} '
                                        f'sigma={sigma} rb={rb}')

    def test_carry_is_reported_and_never_enters_the_minimum(self):
        rec = ms.recommend_leverage(_kpis(), _policy())
        self.assertIn('carry_diagnostic', rec.caps)
        self.assertFalse(rec.caps['carry_diagnostic'].binds)
        self.assertNotEqual(rec.binding_constraint, 'carry_diagnostic')

    def test_max_drawdown_months_is_peak_to_trough(self):
        r = pd.Series([0.05, -0.10, -0.10, -0.10, 0.20, 0.20],
                      index=pd.date_range('2020-01-31', periods=6, freq='ME'))
        # Peak after month 1, trough after month 4 -> three months down.
        self.assertEqual(metrics_mod.max_drawdown_months(r), 3)
        rising = pd.Series([0.01, 0.02, 0.03],
                           index=pd.date_range('2020-01-31', periods=3, freq='ME'))
        self.assertEqual(metrics_mod.max_drawdown_months(rising), 0)
        self.assertEqual(metrics_mod.max_drawdown_months(pd.Series(dtype=float)), 0)
        self.assertEqual(metrics_mod.calculate_metrics(r)['max_dd_months'], 3)


# ------------------------------------------------------------------------------------- #
#  the failure modes
# ------------------------------------------------------------------------------------- #

class TestFailureModes(unittest.TestCase):

    def test_f1_scalar_margin_is_refused_for_a_levered_book(self):
        """A 25-30% equity rate against a UPRO/TMF book overstates the answer by ~3x."""
        rec = ms.recommend_leverage(_kpis(holds_letf=True, held=('UPRO', 'TMF')), _policy())
        self.assertIsNone(rec.recommended_leverage)
        self.assertIn('CAP 1', rec.binding_constraint)
        self.assertIn('leveraged product', rec.caps['margin_survival'].basis)

        # A per-ticker mapping is accepted, and the levered rate is what binds.
        rec2 = ms.recommend_leverage(
            _kpis(holds_letf=True, held=('UPRO', 'TMF')),
            _policy(maintenance_margin={'UPRO': 0.75, 'TMF': 0.75}))
        self.assertIsNotNone(rec2.recommended_leverage)
        self.assertAlmostEqual(rec2.maintenance_margin_used, min(0.75 * 1.5, 0.99), places=9)
        self.assertLess(rec2.recommended_leverage, 1.2)

    def test_f1_a_missing_per_ticker_rate_is_not_calculable(self):
        rec = ms.recommend_leverage(
            _kpis(holds_letf=True, held=('UPRO', 'TMF')),
            _policy(maintenance_margin={'UPRO': 0.75}))
        self.assertIsNone(rec.recommended_leverage)
        self.assertIn('TMF', rec.caps['margin_survival'].basis)

    def test_f2_short_history_is_not_calculable(self):
        """The hard half of F2: below `min_history_months` there is no number and no fallback."""
        rec = ms.recommend_leverage(_kpis(_returns(n=60)), _policy())
        self.assertIsNone(rec.recommended_leverage)
        self.assertIn('min_history_months', rec.caps['margin_survival'].basis)

    def test_f2_a_benign_sample_is_still_stressed_above_its_own_drawdown(self):
        """The soft half, stated honestly: this REDUCES F2, it does not remove it.

        A record that never saw a crash will always be sized more generously than one that did.
        What the module guarantees is that the figure it sizes against is strictly worse than
        the benign sample's own drawdown, and that the markup is reported rather than buried.
        """
        benign = _kpis(_returns(n=180, seed=5), name='BENIGN')
        rec = ms.recommend_leverage(benign, _policy())
        self.assertGreater(rec.dd_adj, abs(benign.max_dd))
        self.assertGreater(rec.dd_factor_sample, 1.0)
        self.assertGreater(rec.dd_factor_intraperiod, 1.0)

    def test_f3_autocorrelation_is_reported_and_block_length_is_swept(self):
        """Blocks destroy long regime persistence, so the choice must be visible."""
        rng = np.random.default_rng(4)
        eps, r = rng.normal(0, 0.03, 240), []
        prev = 0.0
        for e in eps:
            prev = 0.6 * prev + e                    # AR(1), strongly persistent
            r.append(0.006 + prev)
        ar = pd.Series(r, index=pd.date_range('2000-01-31', periods=240, freq='ME'))
        rec = ms.recommend_leverage(_kpis(ar, name='AR1'), _policy())
        self.assertGreater(rec.return_autocorr_1, 0.3)
        self.assertEqual(sorted(rec.block_sensitivity), [6, 12, 24])
        # The claim is NOT that longer blocks always find deeper drawdowns — they preserve more
        # persistence but resample less variety, and which effect wins depends on the decay
        # rate. The claim is that the choice MOVES THE ANSWER by more than rounding, so leaving
        # it undeclared would hide a real degree of freedom.
        spread = max(rec.block_sensitivity.values()) - min(rec.block_sensitivity.values())
        self.assertGreater(spread, 0.01, 'block length changed the drawdown by under 1pp; '
                                         'the sensitivity sweep is not measuring anything')

    def test_f4_accrual_lowers_the_cap_and_the_accrual_variant_is_the_one_retained(self):
        kpis = _kpis()
        rec = ms.recommend_leverage(kpis, _policy())
        plain = rec.caps['margin_survival'].value
        accrued = rec.caps['margin_survival_accrual'].value
        self.assertGreater(kpis.max_dd_months, 0)
        self.assertLess(accrued, plain)
        self.assertGreater(rec.interest_accrued_fraction, 0.0)
        self.assertEqual(_margin_caps(rec), ['margin_survival_crisis_accrual'])

    def test_f4_accrual_can_be_switched_off_and_then_nothing_accrues(self):
        rec = ms.recommend_leverage(_kpis(), _policy(use_interest_accrual=False))
        self.assertEqual(rec.interest_accrued_fraction, 0.0)
        self.assertEqual(_margin_caps(rec), ['margin_survival_crisis'])
        self.assertEqual(rec.caps['margin_survival_crisis'].value,
                         rec.caps['margin_survival_crisis_accrual'].value)

    def test_f5_a_one_sided_policy_liquidates_where_the_two_sided_one_survives(self):
        """The owner's requested test: drift kills a path a static reading calls survivable.

        `one_sided_up` re-levers on gains and never de-levers on losses — a plausible reading of
        "maintain target exposure". Two -20% months at f=2, m=0.25: the two-sided reset sells
        into the first decline and survives the second; the one-sided policy does not.
        """
        path = [-0.20, -0.20]
        two_sided = ms.simulate_margin_path(path, 2.0, 0.25, policy='reset_monthly')
        one_sided = ms.simulate_margin_path(path, 2.0, 0.25, policy='one_sided_up')
        self.assertFalse(two_sided.liquidated)
        self.assertTrue(one_sided.liquidated)
        self.assertEqual(one_sided.step, 1)
        self.assertEqual(one_sided.trigger, 'period_end')

    def test_f5_an_intra_period_excursion_liquidates_a_path_month_ends_call_safe(self):
        """The gap between a monthly reset and a continuous margin call, as a path.

        The month closes -25%, which at f=2 and m=0.25 is inside the 33.3% the closed form
        allows. Inside the month the book was down 35% — past the call. That is the shape of
        March 2020, and `ledger.run_ledger` cannot see it: it drifts positions from one
        execution date to the next and never looks in between.
        """
        path = [0.0, -0.25]
        blind = ms.simulate_margin_path(path, 2.0, 0.25, policy='reset_monthly')
        seeing = ms.simulate_margin_path(path, 2.0, 0.25, policy='reset_monthly',
                                        intra_period_lows=[1.0, 0.65])
        self.assertFalse(blind.liquidated)
        self.assertTrue(seeing.liquidated)
        self.assertEqual(seeing.trigger, 'intra_period')
        self.assertEqual(seeing.step, 1)

    def test_f6_a_crisis_margin_increase_liquidates_a_survivable_decline(self):
        """m=0.25, f=2: d_max is 33.3% normally and 20.0% at m=0.375. A 25% decline splits them."""
        self.assertAlmostEqual(ms.liquidation_threshold(2.0, 0.25), 1.0 / 3.0, places=9)
        self.assertAlmostEqual(ms.liquidation_threshold(2.0, 0.375), 0.20, places=9)
        calm = ms.simulate_margin_path([-0.25], 2.0, 0.25, policy='none')
        crisis = ms.simulate_margin_path([-0.25], 2.0, 0.25, policy='none',
                                        crisis_margin_from=0, crisis_margin_multiple=1.5)
        self.assertFalse(calm.liquidated)
        self.assertTrue(crisis.liquidated)

    def test_f6_a_maintenance_requirement_at_or_above_one_answers_rather_than_raising(self):
        """A 3x book at a 35% base rate resolves to m=1.05. That used to kill the whole run.

        `recommend_leverage` catches `NotCalculable` and nothing else, so the `ValueError` from
        `leverage_for_threshold`'s domain check propagated out of the sizing layer and took the
        backtest with it — reachable from the dashboard by moving one slider. The answer at
        m >= 1 is 1.00x to two decimals ("the broker lends nothing against this"), said out
        loud in the notes rather than arrived at silently.
        """
        kpis = _kpis(held=('UPRO',), holds_letf=True)
        rec = ms.recommend_leverage(kpis, _policy(maintenance_margin={'UPRO': 1.05}))
        self.assertGreaterEqual(rec.recommended_leverage, 1.0)
        self.assertLess(rec.recommended_leverage, 1.02)
        self.assertTrue(any('Clamped' in n for n in rec.notes), rec.notes)
        self.assertEqual(rec.maintenance_margin_used, ms.MAINTENANCE_CEILING)

    def test_f6_both_margin_variants_are_always_reported(self):
        rec = ms.recommend_leverage(_kpis(), _policy())
        for label in ('margin_survival', 'margin_survival_accrual',
                      'margin_survival_crisis', 'margin_survival_crisis_accrual'):
            self.assertIn(label, rec.caps)
            self.assertIsNotNone(rec.caps[label].value)
        self.assertLess(rec.caps['margin_survival_crisis'].value,
                        rec.caps['margin_survival'].value)

    def test_f7_the_kelly_gate_stops_a_low_sharpe_model(self):
        """SR 0.10 at 12% vol gives f* = 0.83 before any haircut. No leverage is justified."""
        kpis = _kpis(sharpe=0.10, vol=0.12)
        rec = ms.recommend_leverage(kpis, _policy(risk_free_annual=0.05))   # spread 0
        self.assertFalse(rec.leverage_justified)
        self.assertEqual(rec.recommended_leverage, 1.0)
        self.assertIn('kelly_gate', rec.binding_constraint)
        self.assertLess(rec.caps['kelly'].value, 1.0)

    def test_f7_the_haircut_is_the_more_severe_of_the_two_and_it_bites(self):
        rec = ms.recommend_leverage(_kpis(), _policy())
        self.assertLess(rec.sharpe_deflated, rec.sharpe_observed)
        self.assertLess(rec.sharpe_lower_bound, rec.sharpe_observed)
        self.assertEqual(rec.sharpe_used, min(rec.sharpe_deflated, rec.sharpe_lower_bound))
        # The haircut is exactly the expected-maximum term, and it is a SUBTRACTION.
        self.assertAlmostEqual(rec.sharpe_observed - rec.sharpe_deflated,
                               ms.expected_max_sharpe(19, 0.15), places=12)

    def test_f7_no_trial_count_means_no_multiple_testing_haircut_and_says_so(self):
        rec = ms.recommend_leverage(_kpis(), _policy(n_trials=None, trial_sharpe_sd=None))
        self.assertTrue(math.isnan(rec.sharpe_deflated))
        self.assertEqual(rec.sharpe_used, rec.sharpe_lower_bound)
        self.assertTrue(any('multiple-testing' in a for a in rec.invalidating_assumptions))

    def test_f8_a_missing_input_never_produces_a_number(self):
        """The core of F8: CAP 1 unanswerable must not be papered over by a cheerful CAP 4."""
        rec = ms.recommend_leverage(_kpis(), _policy(maintenance_margin=None,
                                                    borrowing_capacity_leverage=3.0))
        self.assertIsNone(rec.recommended_leverage)
        self.assertIn('non calculable', rec.binding_constraint)
        self.assertEqual(rec.caps['borrowing_capacity'].value, 3.0)   # reported, not used

    def test_f8_each_input_in_turn(self):
        for field, marker in (('maintenance_margin', 'margin_survival'),
                              ('seed', 'margin_survival')):
            rec = ms.recommend_leverage(_kpis(), _policy(**{field: None}))
            self.assertIsNone(rec.recommended_leverage, f'{field} was defaulted silently')
            self.assertIsNone(rec.caps[marker].value)

    def test_f8_a_missing_daily_drawdown_is_not_calculable(self):
        """Assuming an intra-period factor of 1.0 is exactly the silent default being refused."""
        rec = ms.recommend_leverage(dataclasses.replace(_kpis(), daily_max_dd=None), _policy())
        self.assertIsNone(rec.recommended_leverage)
        self.assertIn('daily_max_dd', rec.caps['margin_survival'].basis)

    def test_f8_secondary_caps_degrade_without_voiding_the_result(self):
        rec = ms.recommend_leverage(_kpis(), _policy(borrowing_capacity_leverage=None))
        self.assertIsNotNone(rec.recommended_leverage)
        self.assertIsNone(rec.caps['borrowing_capacity'].value)
        self.assertTrue(any('capacity' in a for a in rec.invalidating_assumptions))

    def test_f9_the_parameter_block_is_identical_across_models(self):
        policy = _policy()
        a = ms.recommend_leverage(_kpis(_returns(seed=1), name='A'), policy)
        b = ms.recommend_leverage(_kpis(_returns(seed=2), name='B'), policy)
        self.assertEqual(a.policy.fingerprint(), b.policy.fingerprint())
        self.assertNotEqual(a.dd_bootstrapped, b.dd_bootstrapped)   # measurements do differ

    def test_f10_the_bootstrap_is_deterministic_and_the_seed_is_in_the_block(self):
        kpis = _kpis()
        first = ms.recommend_leverage(kpis, _policy(seed=99))
        again = ms.recommend_leverage(kpis, _policy(seed=99))
        other = ms.recommend_leverage(kpis, _policy(seed=100))
        self.assertEqual(first.dd_bootstrapped, again.dd_bootstrapped)
        self.assertEqual(first.recommended_leverage, again.recommended_leverage)
        self.assertNotEqual(first.dd_bootstrapped, other.dd_bootstrapped)
        self.assertIn(('seed', 99), first.policy.fingerprint())

    def test_f10_no_seed_is_not_calculable(self):
        with self.assertRaises(ms.NotCalculable):
            ms.bootstrap_drawdown_quantile(_returns(), 240, 12, 100, 0.95, None)

    def test_f11_the_intra_period_measure_sees_a_trough_that_recovered(self):
        """A month-end series showing 0% while the held book was down 30% inside the month."""
        days = pd.bdate_range('2020-01-31', '2020-03-31')
        px = pd.Series(100.0, index=days, name='X')
        dip = (days >= '2020-02-10') & (days <= '2020-02-20')
        px[dip] = 70.0
        prices = px.to_frame()
        weights = pd.DataFrame({'X': [1.0, 1.0]},
                               index=pd.DatetimeIndex(['2020-01-31', '2020-02-28']))

        monthly = pd.Series([0.0], index=pd.DatetimeIndex(['2020-02-28']))
        self.assertEqual(metrics_mod.calculate_metrics(monthly)['max_dd'], 0.0)

        daily_dd = ms.intraperiod_max_drawdown(prices, weights)
        self.assertLess(daily_dd, -0.29)

    def test_f11_the_uplift_can_never_shrink_the_drawdown(self):
        """Even given a daily figure shallower than the monthly one, the factor floors at 1."""
        kpis = _kpis(daily_uplift=0.5)
        rec = ms.recommend_leverage(kpis, _policy())
        self.assertGreaterEqual(rec.dd_factor_intraperiod, 1.0)

    def test_f12_borrowing_capacity_binds_and_is_named(self):
        rec = ms.recommend_leverage(_kpis(), _policy(borrowing_capacity_leverage=1.0))
        self.assertEqual(rec.recommended_leverage, 1.0)
        self.assertEqual(rec.binding_constraint, 'borrowing_capacity')
        self.assertFalse(rec.leverage_justified)

    def test_f13_an_unsurvivable_drawdown_clamps_to_one_without_exploding(self):
        """k * DD_adj >= 100% must give 1.0x and a flag, not a negative number or a ZeroDivision."""
        rec = ms.recommend_leverage(_kpis(_returns(crash_at=100, crash=-0.28)),
                                    _policy(safety_factor_k=5.0))
        self.assertTrue(rec.dd_clamped_at_one)
        self.assertEqual(rec.dd_adj, 1.0)
        self.assertEqual(rec.caps['margin_survival'].value, 1.0)
        self.assertEqual(rec.recommended_leverage, 1.0)
        self.assertEqual(ms.leverage_for_threshold(1.5, 0.30), 1.0)

    def test_f14_no_optimisation_surface(self):
        """The owner's standing constraint, enforced structurally.

        `f` must be sustainable indefinitely, never fitted to the sample — including if it is
        asked for later. No public entry point may accept an objective or a target.
        """
        public = [ms.recommend_leverage, ms.cap_margin_survival, ms.cap_kelly,
                  ms.carry_diagnostic, ms.leverage_for_threshold, ms.liquidation_threshold,
                  ms.bootstrap_drawdown_quantile, ms.simulate_margin_path]
        for fn in public:
            names = set(inspect.signature(fn).parameters)
            bad = names & ms.FORBIDDEN_OBJECTIVE_PARAMS
            self.assertEqual(bad, set(), f'{fn.__name__} grew an objective parameter: {bad}')
        policy_fields = set(inspect.signature(ms.MarginPolicy).parameters)
        self.assertEqual(policy_fields & ms.FORBIDDEN_OBJECTIVE_PARAMS, set())

    def test_f15_the_threshold_ratio_is_never_below_k(self):
        """Derived, not conventional. f <= f_cap1 => d_max(f) >= k*DD_adj >= k*DD_observed.

        A value below k means the chain from DD_adj to the reported threshold is broken, which
        is why this is asserted across seeds and safety factors rather than spot-checked.
        """
        for k in (2.0, 3.0, 5.0):
            for seed in (1, 2, 3):
                rec = ms.recommend_leverage(_kpis(_returns(seed=seed)), _policy(
                    safety_factor_k=k, borrowing_capacity_leverage=1.5))
                if rec.recommended_leverage is None or rec.dd_clamped_at_one:
                    continue
                self.assertGreaterEqual(rec.threshold_over_observed_maxdd + 1e-9, k,
                                        f'k={k} seed={seed}: {rec.threshold_over_observed_maxdd}')
                self.assertEqual(rec.notes, tuple(n for n in rec.notes
                                                  if 'INVARIANT VIOLATED' not in n))

    def test_f15_the_clamped_branch_has_its_own_bound(self):
        rec = ms.recommend_leverage(_kpis(_returns(crash_at=100, crash=-0.28)),
                                    _policy(safety_factor_k=5.0))
        self.assertTrue(rec.dd_clamped_at_one)
        self.assertAlmostEqual(rec.threshold_over_observed_maxdd,
                               1.0 / abs(rec.dd_observed), places=9)


def _frame_wise_intraperiod(daily_prices, monthly_weights):
    """The frame-wise form `intraperiod_max_drawdown` had until 2026-07-31.

    Kept HERE rather than in the module so the optimisation has something independent to be
    equal to. It is the readable statement of the intent — slice the daily frame between two
    decision dates, hold the weights, value the book — and the shipped version is the same
    arithmetic on positional slices, 50x faster because it stops rebuilding a boolean mask over
    the whole history once per month per strategy.
    """
    px = daily_prices.sort_index()
    w = monthly_weights.sort_index()
    cols = [c for c in w.columns if c in px.columns]
    px, w = px[cols], w[cols]
    equity, level, dates = [], 1.0, list(w.index)
    for i, start in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else px.index[-1]
        window = px.loc[(px.index >= start) & (px.index <= end)]
        if len(window) < 2:
            continue
        weights = w.loc[start]
        if float(weights.sum()) <= 0:
            continue
        held = weights.index[(weights.abs() > 1e-12) & window.iloc[0].notna()]
        if len(held) == 0:
            continue
        rel = window[held].div(window[held].iloc[0], axis=1)
        path = level * (rel * weights[held]).sum(axis=1) / float(weights[held].sum())
        equity.append(path.iloc[1:])
        level = float(path.iloc[-1])
    curve = pd.concat([pd.Series([1.0], index=[dates[0]])] + equity)
    return float(metrics_mod.drawdown_series(curve).min())


class TestIntraperiodFastPath(unittest.TestCase):
    """The positional rewrite must be the frame-wise answer, not merely a plausible one.

    Over the whole registry the two agreed to 1.2e-15 when the change landed; these cases carry
    the awkward parts a clean random walk would not exercise — a ticker that has no price on the
    first day of a window, a month with no allocation at all, a month with a single session.
    """

    def _prices(self, seed, n_tickers=4, days=900):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range('2015-01-01', periods=days)
        steps = rng.normal(0.0004, 0.011, size=(days, n_tickers))
        return pd.DataFrame(100.0 * np.cumprod(1.0 + steps, axis=0), index=idx,
                            columns=[f'T{i}' for i in range(n_tickers)])

    def _weights(self, px, seed):
        """Month-end decision dates drawn from the trading calendar itself, two names each."""
        rng = np.random.default_rng(seed + 1000)
        dates = pd.DatetimeIndex(px.index.to_series().groupby(px.index.to_period('M')).max())
        rows = []
        for _ in range(len(dates)):
            w = np.zeros(px.shape[1])
            w[rng.choice(px.shape[1], size=2, replace=False)] = 0.5
            rows.append(w)
        return pd.DataFrame(rows, index=dates, columns=px.columns)

    def test_it_equals_the_frame_wise_reference(self):
        for seed in (1, 2, 3):
            px = self._prices(seed)
            w = self._weights(px, seed)
            self.assertAlmostEqual(ms.intraperiod_max_drawdown(px, w),
                                   _frame_wise_intraperiod(px, w), places=12,
                                   msg=f'seed {seed}')

    def test_it_equals_the_reference_with_a_late_ticker_and_an_empty_month(self):
        px = self._prices(11)
        w = self._weights(px, 11)
        # T3 does not exist for the first year: dropped from any window whose first day has no
        # price for it, and the remaining holdings re-weighted among themselves.
        px.loc[px.index[:250], 'T3'] = np.nan
        # A month holding nothing at all — `total <= 0` — must be skipped by both forms.
        w.iloc[5] = 0.0
        self.assertAlmostEqual(ms.intraperiod_max_drawdown(px, w),
                               _frame_wise_intraperiod(px, w), places=12)

    def test_a_single_session_window_is_skipped_by_both(self):
        px = self._prices(5, days=40)
        w = pd.DataFrame({'T0': [1.0, 1.0], 'T1': [0.0, 0.0], 'T2': [0.0, 0.0],
                          'T3': [0.0, 0.0]},
                         index=pd.DatetimeIndex([px.index[0], px.index[-1]]))
        self.assertAlmostEqual(ms.intraperiod_max_drawdown(px, w),
                               _frame_wise_intraperiod(px, w), places=12)


class TestPolicyValidation(unittest.TestCase):

    def test_nonsense_parameters_are_refused_at_construction(self):
        for kw in ({'safety_factor_k': 0.0}, {'dd_quantile': 1.0}, {'dd_quantile': 0.0},
                   {'crisis_margin_multiple': 0.9}, {'intraday_pad': 0.9},
                   {'leverage_policy': 'whatever'}):
            with self.assertRaises(ValueError, msg=str(kw)):
                _policy(**kw)

    def test_presets_are_multiples_of_drawdown(self):
        self.assertEqual(ms.MarginPolicy.PRESETS,
                         {'prudent': 5.0, 'balanced': 3.0, 'aggressive': 2.0})
        p = ms.MarginPolicy.preset('prudent', maintenance_margin=0.3, seed=1)
        self.assertEqual(p.safety_factor_k, 5.0)
        with self.assertRaises(ValueError):
            ms.MarginPolicy.preset('reckless')

    def test_compare_table_refuses_rows_from_different_parameter_blocks(self):
        """Alignability is the requirement. Lining up incomparable rows would defeat it."""
        a = ms.recommend_leverage(_kpis(name='A'), _policy())
        b = ms.recommend_leverage(_kpis(name='B'), _policy(safety_factor_k=5.0))
        with self.assertRaises(ValueError):
            ms.compare_table([a, b])
        text = ms.compare_table([a, ms.recommend_leverage(_kpis(name='C'), _policy())])
        self.assertIn('A', text)
        self.assertIn('C', text)
        self.assertIn('non calculable', text)

    def test_compare_table_prints_non_calculable_rather_than_a_number(self):
        rec = ms.recommend_leverage(_kpis(holds_letf=True, held=('UPRO',)), _policy())
        self.assertIn('n/c', ms.compare_table([rec]))

    def test_the_report_names_the_binding_constraint_first(self):
        rec = ms.recommend_leverage(_kpis(), _policy())
        text = rec.report()
        self.assertIn('binding constraint', text)
        self.assertIn('if these are false', text)
        self.assertIn('diagnostic, never binds', text)

    def test_every_result_carries_the_assumptions_that_void_it(self):
        rec = ms.recommend_leverage(_kpis(), _policy())
        joined = ' '.join(rec.invalidating_assumptions)
        for expected in ('recall the loan', 'intraday', 'compose', 'reset_monthly'):
            self.assertIn(expected, joined)


if __name__ == '__main__':
    unittest.main()
