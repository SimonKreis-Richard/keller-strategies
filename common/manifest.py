"""
Run manifests: every saved artefact carries the conditions that produced it.

The 2026-07-28 audit found 92 saved reports in `backtest_results/` and could not establish,
for any of them, which commit produced it, what data it read, what execution convention it
assumed, or what window it actually measured. They had to be quarantined wholesale rather
than corrected, because there was no way to tell which ones were affected by what.

A manifest costs a few hundred bytes and makes that impossible to repeat. It records the
things that change the number: the code, the data, the resolved configuration, and the
measured — not requested — span.

Nothing personal goes in a manifest. `user_config.json` holds real account balances and is
gitignored; the redaction below is a hard filter, not a convention, so a manifest can never
become the route by which a balance reaches the repository.
"""

import datetime
import hashlib
import json
import os
import platform
import subprocess

import pandas as pd

from common import eras

#: Config keys that must never be written to disk. Balances and account names are the
#: owner's private data; the sizing knobs are harmless but travel with them.
_REDACT = {'BROKER_ACCOUNTS', 'ACCOUNT_BALANCE', 'ACCOUNTS'}


def _git(*args, cwd=None):
    try:
        out = subprocess.run(('git',) + args, cwd=cwd, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_state(repo_dir):
    """Commit, branch, and a hash of the uncommitted diff.

    The diff hash is what makes a manifest usable on a dirty worktree: two runs from the
    same commit but different working trees get different hashes, so 'same commit' can never
    silently mean 'same code'.
    """
    diff = _git('diff', 'HEAD', cwd=repo_dir) or ''
    return {
        'commit': _git('rev-parse', 'HEAD', cwd=repo_dir),
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD', cwd=repo_dir),
        'dirty': bool(diff),
        'diff_sha256': hashlib.sha256(diff.encode('utf-8')).hexdigest() if diff else None,
    }


def dependency_versions():
    mods = {}
    for name in ('pandas', 'numpy', 'yfinance', 'matplotlib', 'nicegui'):
        try:
            mods[name] = __import__(name).__version__
        except Exception:
            mods[name] = None
    mods['python'] = platform.python_version()
    return mods


def _safe_config(config):
    out = {}
    for k, v in config.items():
        if k in _REDACT:
            out[k] = '<redacted>'
        elif isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
            out[k] = list(v)
        else:
            out[k] = repr(v)[:200]
    return out


def build_manifest(config, store, metrics_data, outputs=None, repo_dir=None):
    """Everything needed to reproduce, or to refuse to trust, a saved artefact."""
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    measured = [d for d in metrics_data if d.get('first_return') is not None]
    return {
        'generated_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'git': git_state(repo_dir),
        'dependencies': dependency_versions(),
        'data': store.provenance() if store is not None else None,
        'config': _safe_config(config),
        'execution': {
            'convention': config.get('EXECUTION_CONVENTION'),
            'cost_pct_per_side': config.get('COST_PCT_PER_SIDE'),
            'cost_note': 'one-way, charged per leg on notional actually traded',
            'cash_ticker': config.get('CASH_TICKER'),
            'coverage_policy': config.get('COVERAGE_POLICY'),
            'leverage': config.get('LEVERAGE_FACTOR'),
            'margin_follows_signal': config.get('MARGIN_FOLLOWS_SIGNAL'),
            # Hardcoded False because it is a property of run_ledger, not a setting: the
            # ledger draws a debit balance and accrues interest on it, and never once
            # compares equity to a maintenance requirement. A saved artefact with
            # leverage > 1.0 is therefore an un-liquidated upper bound, and the manifest
            # has to say so or the number outlives the caveat. See KNOWN_GAPS.md §3 and
            # common/margin_sizing.py, which prices the constraint beside the engine.
            'maintenance_margin_monitored': False,
        },
        'measured': {
            # No 'requested_window': there is nothing to request. The era floor is frozen in
            # common/eras.py and the ranked window is derived from the strategies compared.
            'era_floor': str(config.get('START_DATE')),
            'ranked_window': ([str(pd.Timestamp(measured[0]['window_start']).date()),
                               str(pd.Timestamp(measured[0]['window_end']).date())]
                              if measured and measured[0].get('window_start') is not None
                              else None),
            'ranked_window_binding': (measured[0].get('window_binding')
                                      if measured else None),
            'segmentations': [s.key for s in eras.SEGMENTATIONS],
            'actual_window': ([str(min(d['first_return'] for d in measured).date()),
                               str(max(d['last_return'] for d in measured).date())]
                              if measured else None),
            'n_strategies': len(metrics_data),
            'per_strategy': [{
                'name': d['name'],
                'first_return': str(d['first_return'].date()) if d.get('first_return') is not None else None,
                'last_return': str(d['last_return'].date()) if d.get('last_return') is not None else None,
                'n_periods': d.get('n_periods'),
                'annual_turnover': round(float(d.get('annual_turnover', float('nan'))), 4),
                'total_cost_bps': round(float(d.get('total_cost_bps', 0.0)), 2),
                'coverage_trimmed': d.get('coverage_trimmed'),
                'binding_ticker': d.get('binding_ticker'),
                'rf_annual': round(float(d.get('rf_annual', 0.0)), 6),
                # The rate alone does not say whether it was traded or accrued from a
                # published yield. A Sharpe computed against a CONSTRUCTED rf is a
                # different claim from one computed against BIL, and the manifest is
                # where that distinction has to survive the report scrolling away.
                'rf_desc': d.get('rf_desc'),
            } for d in metrics_data],
        },
        'outputs': outputs or {},
    }


def file_sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def write_manifest(path, manifest):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
        fh.write('\n')
    return path
