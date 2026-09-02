"""
Regenerate `tests/fixtures/run_facts.json` — the artefact the documentation is checked against.

Deliberately a separate, explicit command, exactly like `test_golden_master --record`. An
artefact that regenerates itself on every run pins nothing: the documentation test would
then compare prose against whatever the last run happened to produce, and a number could
drift in both places at once without anything going red.

    venv/Scripts/python.exe -m tools.emit_facts            # full registry, with robustness
    venv/Scripts/python.exe -m tools.emit_facts --quick    # skip the expensive block

The robustness block enumerates 12 870 CSCV splits and draws 2 000 bootstrap paths, so the
full run takes a few minutes. Run it whenever a number that documentation quotes has
legitimately moved, and commit the JSON in the SAME commit as the change that moved it.

Goes through `tools/backtest_driver.py`, so it can never reach the live path.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.facts import build_facts, write_facts          # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS_PATH = os.path.join(ROOT_DIR, 'tests', 'fixtures', 'run_facts.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Regenerate the documentation facts file.')
    parser.add_argument('--quick', action='store_true',
                        help='skip the robustness block (PBO and the rank bootstrap)')
    parser.add_argument('--out', default=FACTS_PATH)
    args = parser.parse_args(argv)

    from tools.backtest_driver import run, build_config
    import main as engine

    metrics, _results, _prices, store, cfg = run()
    facts = build_facts(metrics, config=cfg, store=store,
                        include_robustness=not args.quick,
                        rank_key=engine.RANK_BY)
    write_facts(args.out, facts)

    print('\nwrote {}'.format(args.out))
    print('  registry            : {}'.format(facts['registry']['n_registered']))
    print('  ranked / measured   : {} / {}'.format(facts['window']['n_ranked'],
                                                   facts['window']['n_measured']))
    print('  window              : {} .. {} (set by {})'.format(
        facts['window'].get('window_start'), facts['window'].get('window_end'),
        facts['window'].get('window_binding')))
    integrity = facts.get('integrity') or {}
    print('  data check          : {} ({} tickers)'.format(
        integrity.get('status'), integrity.get('n_tickers_checked')))
    rob = facts.get('robustness')
    if rob:
        print('  PBO strat/fam/mech  : {} / {} / {}'.format(
            rob.get('pbo_strategy'), rob.get('pbo_family'), rob.get('pbo_mechanism')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
