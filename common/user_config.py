"""Personal-settings loader.

All user-specific values (broker account balances, selected strategies, leverage,
dates, sizing knobs...) live in `user_config.json` at the project root. That file is
GITIGNORED so personal data never reaches the public repository. A committed template
with placeholder values is provided as `user_config.example.json`.

Resolution order for every setting:
    user_config.json  >  the sanitized defaults in main.py's USER DASHBOARD

- Missing file  -> defaults apply silently (fresh clone just works).
- Malformed file -> we FAIL LOUDLY. Live order sizing must never fall back to
  placeholder balances because of a silent JSON typo.
"""
import json
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, 'user_config.json')


#: Keys renamed on 2026-07-28. The old name still loads so an existing personal config keeps
#: working, but the MEANING of this one changed and silence would be dishonest — see below.
_DEPRECATED = {
    'TRANSACTION_COST_PCT': (
        'COST_PCT_PER_SIDE',
        'the value is now charged ONE-WAY PER LEG, so a full A->B rotation costs TWICE what '
        'the old key charged. Keller\'s HAA paper assumes 0.1% one-way; the old code charged '
        'sum|dw|/2 x 0.1%, i.e. half of it. Your results will be lower, and correctly so.'),
}

#: Keys REMOVED on 2026-07-28 because the setting itself was the defect. Left in a personal
#: config they are simply inert, and saying so is better than letting someone believe a date
#: they typed still controls anything.
_REMOVED = {
    'START_DATE': 'the evaluation window is no longer selectable',
    'END_DATE': 'the evaluation window is no longer selectable',
}

_REMOVED_WHY = (
    "A window you choose is a result you chose: 2015 flatters trend following, 2009 flatters "
    "buy-and-hold. The engine now runs a FIXED era (from 2001-03, the first month its assets "
    "allow) and reports every strategy across pre-registered market regimes dated by the "
    "NBER, the FOMC, the BLS and the S&P 500. See common/eras.py.")


def load_user_config(path=CONFIG_PATH):
    """Return the user-config dict, or {} when the file doesn't exist."""
    try:
        with open(path, encoding='utf-8') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"ERROR: {os.path.basename(path)} is not valid JSON ({e}).\n"
            "Fix or delete the file (defaults will then apply). Refusing to run with "
            "a broken personal config — live sizing could otherwise use wrong balances."
        )

    for old, (new, why) in _DEPRECATED.items():
        if old in cfg and new not in cfg:
            print(f"[DEPRECATED] {os.path.basename(path)}: '{old}' -> '{new}'. It still "
                  f"loads, but {why}")

    stale = [k for k in _REMOVED if k in cfg]
    if stale:
        print(f"[IGNORED] {os.path.basename(path)}: {', '.join(sorted(stale))} - "
              f"{_REMOVED[stale[0]]}. {_REMOVED_WHY}")
    return cfg


def save_user_config(cfg, path=CONFIG_PATH):
    """Persist the user-config dict (used by the GUI's 'save my settings' action)."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
