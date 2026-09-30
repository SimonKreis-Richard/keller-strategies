"""
Savings projection on the haircut backtest — what the accounts could become, stated as an
assumption and never as a forecast.

    venv/Scripts/python.exe -m tools.projection                 # reads user_config.json
    venv/Scripts/python.exe -m tools.projection --years 30      # override the horizon
    venv/Scripts/python.exe -m tools.projection --strategy BAA_G12
    venv/Scripts/python.exe -m tools.projection --demo          # the example's fictional accounts
    venv/Scripts/python.exe -m tools.projection --no-save       # print only
    venv/Scripts/python.exe -m tools.projection --rate 1.3850   # USD/CAD for the display

WHAT IT DOES
------------
1. Reads the PROJECTION block of `user_config.json` (strict; `common/projection.parse_config`).
   Starting balances come from `BROKER_ACCOUNTS`; every account needs a tax kind. `--demo`
   reads `user_config.example.json` instead, whose accounts are fictional.
2. Measures the strategy and its comparator through `tools/backtest_driver.run` — the
   backtest-only path, at leverage 1.0, whatever `LEVERAGE_FACTOR` says.
3. Haircuts each Sharpe against the FULL registry's selection population (`run_facts.json`,
   AUD-06), recentres the record on it and resamples it (`common/projection.project`).
   Without that population it refuses: a two-entry run cannot measure the search that
   produced the pick, and projecting an unhaircut Sharpe is what the module exists to stop.
4. Prints the assumptions first, then the after-tax values in `PROJECTION.currency` and, at
   today's USD/CAD held constant, in the other currency (`tools/fx_rate.py`: `--rate`, else
   `PROJECTION.usdcad`, else the Bank of Canada, else Yahoo; with none, the other currency
   is not shown rather than guessed). Saves a JSON record to `backtest_results/`
   (gitignored).

Nothing here places an order or reaches the live path.
"""
import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common import projection as proj                       # noqa: E402
from common.robustness import SEED                          # noqa: E402

EXAMPLE_PATH = os.path.join(ROOT, 'user_config.example.json')
RESULTS_DIR = os.path.join(ROOT, 'backtest_results')


def _load_example():
    with open(EXAMPLE_PATH, encoding='utf-8') as fh:
        return json.load(fh)


def _population():
    from common.leverage_advice import REGISTRY_FACTS, registry_trial_population
    n, sd = registry_trial_population()
    if n is None:
        raise proj.ProjectionError(
            f'no selection population in {os.path.relpath(REGISTRY_FACTS, ROOT)}. Regenerate '
            'it with `python -m tools.emit_facts`; a projection of a selected strategy without '
            'its multiple-testing haircut is refused.')
    return n, sd, (f'N={n} trials, trial Sharpe sd {sd:.3f}, full registry '
                   '(tests/fixtures/run_facts.json)')


def save(p, settings, directory=None, now=None):
    directory = directory or RESULTS_DIR
    os.makedirs(directory, exist_ok=True)
    stamp = (now or datetime.datetime.now()).strftime('%Y%m%d_%H%M%S')
    path = os.path.join(directory, f'projection_{stamp}.json')
    record = {'settings': settings, **proj.to_dict(p)}
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=2, default=str)
        fh.write('\n')
    return path


def _resolve_fx(manual):
    from tools.fx_rate import resolve_usdcad
    return resolve_usdcad(manual=manual)


def compute(settings, accounts, tax, run_fn=None, population_fn=None, fx_resolver=None):
    """The Projection for `settings`. Separate from `main` so the dashboard can call it."""
    names = [settings['strategy']] + ([settings['comparator']] if settings['comparator']
                                      else [])
    if run_fn is None:
        from tools.backtest_driver import run as run_fn
    n_trials, sd, note = (population_fn or _population)()
    try:
        metrics = run_fn(names, LEVERAGE_FACTOR=1.0)[0]
    except KeyError as exc:
        raise proj.ConfigError(f'{exc} is not a registry entry '
                               '(`python -m tools.backtest_driver` lists them).') from None
    by_name = {d['name']: d for d in metrics}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise proj.ProjectionError(f'the backtest produced no metrics for {", ".join(missing)}')
    fx, fx_note = None, ''
    try:
        fx = (fx_resolver or _resolve_fx)(settings.get('usdcad'))
    except ValueError as exc:            # CAExecutionError: every source failed, or implausible
        fx_note = str(exc).splitlines()[0]
    return proj.project([by_name[n] for n in names], accounts, tax,
                        settings['horizon_years'], settings['inflation'], n_trials, sd,
                        population_note=note, seed=SEED,
                        currency=settings.get('currency', 'CAD'), fx=fx, fx_note=fx_note)


def main(argv=None, run_fn=None, config_loader=None, population_fn=None, results_dir=None,
         fx_resolver=None):
    parser = argparse.ArgumentParser(description='Savings projection on the haircut backtest. '
                                                 'An assumption, not a forecast.')
    parser.add_argument('--strategy', default=None, help='override PROJECTION.strategy')
    parser.add_argument('--comparator', default=None, help='override PROJECTION.comparator')
    parser.add_argument('--years', type=float, default=None,
                        help='override PROJECTION.horizon_years')
    parser.add_argument('--demo', action='store_true',
                        help='use the fictional accounts of user_config.example.json')
    parser.add_argument('--rate', type=float, default=None,
                        help='USD/CAD (CAD per USD) for the second currency; skips the lookup')
    parser.add_argument('--no-save', action='store_true', help='print only; write nothing')
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

    try:
        if args.demo:
            cfg = _load_example()
        else:
            if config_loader is None:
                from common.user_config import load_user_config as config_loader
            cfg = config_loader()
        block = dict(cfg.get('PROJECTION') or {})
        for key, value in (('strategy', args.strategy), ('comparator', args.comparator),
                           ('horizon_years', args.years), ('usdcad', args.rate)):
            if value is not None:
                block[key] = value
        settings, accounts, tax = proj.parse_config({**cfg, 'PROJECTION': block}
                                                    if cfg.get('PROJECTION') else cfg)
        p = compute(settings, accounts, tax, run_fn, population_fn, fx_resolver)
    except (proj.ConfigError, proj.ProjectionError) as exc:
        print(f'projection: {exc}')
        return 1

    print('\n'.join(proj.report_lines(p)))
    if args.demo:
        print('\n  DEMONSTRATION: the accounts above are the fictional ones of '
              'user_config.example.json.')
    if not args.no_save:
        print(f'\n  saved: {save(p, settings, results_dir)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
