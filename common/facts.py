"""
Every number the prose quotes, computed once and written down.

WHY THIS EXISTS
---------------
The registry count has drifted out of the documentation five separate times, once within a
day of the test that was meant to pin it, and the published PBO figures were hand-copied
into two files. Both are the same disease: **prose caching computation**. A cache needs
invalidation, and a hand-maintained regex is not one -- it pins the phrasings that went
stale last time, not the ones that will go stale next.

So the engine emits its facts, and the documentation test compares prose against the
artefact instead of against a list of remembered sentences. New phrasing is then covered for
free; a new NUMBER is covered as soon as it enters this file.

WHAT IS AND IS NOT IN HERE
--------------------------
Facts are things the engine COMPUTED on a run. Two consequences worth stating:

* The frozen copy at `tests/fixtures/run_facts.json` is the contract the tests read, and
  regenerating it is an explicit act (`tools/emit_facts.py`), never a side effect of a run.
  Same discipline as the golden master, and for the same reason: an artefact that updates
  itself pins nothing.
* `include_robustness` is off by default. PBO enumerates 12 870 splits and the rank
  bootstrap draws 2 000 paths; making every saved report pay that would turn a guard into a
  tax, and the numbers that move are the ones a regeneration is for.
"""
import datetime as _dt
import json

import pandas as pd


def _round(value, digits=6):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _registry_facts():
    """Counts straight off `main.ALL_STRATEGIES`, which is the only authority on them."""
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    by_fidelity, by_role = {}, {}
    for name, factory in main.ALL_STRATEGIES.items():
        try:
            strat = factory()
        except Exception:                                # noqa: BLE001
            continue
        fid = getattr(strat, 'fidelity', 'unknown')
        role = getattr(strat, 'role', 'strategy')
        by_fidelity[fid] = by_fidelity.get(fid, 0) + 1
        by_role[role] = by_role.get(role, 0) + 1
    return {
        'n_registered': len(main.ALL_STRATEGIES),
        'by_fidelity': dict(sorted(by_fidelity.items())),
        'by_role': dict(sorted(by_role.items())),
        'keys': sorted(main.ALL_STRATEGIES),
    }


def _window_facts(metrics_data, config):
    ranked = [d for d in metrics_data if d.get('in_ranked_window')]
    first = ranked[0] if ranked else (metrics_data[0] if metrics_data else None)
    out = {'era_floor': str(config.get('START_DATE')) if config else None,
           'n_ranked': len(ranked), 'n_measured': len(metrics_data)}
    if first is not None:
        for key in ('window_start', 'window_end', 'window_binding'):
            value = first.get(key)
            out[key] = str(value.date()) if hasattr(value, 'date') else (
                str(value) if value is not None else None)
        out['rf_annual'] = _round(first.get('rf_annual'))
    return out


def _metric_facts(metrics_data):
    fields = ('cagr', 'max_dd', 'sharpe', 'sortino', 'vol', 'upi')
    return {d['name']: {f: _round(d.get(f)) for f in fields}
            for d in sorted(metrics_data, key=lambda d: d['name'])
            if d.get('first_return') is not None}


def _selection_facts(metrics_data):
    from common.selection import participation_ratio, trial_sharpe_spread
    n_trials, spread = trial_sharpe_spread(metrics_data)
    out = {'n_trials': n_trials, 'trial_sharpe_sd': _round(spread)}
    series = {d['name']: d['returns'] for d in metrics_data
              if d.get('returns') is not None and not d['returns'].empty
              and d.get('role', 'strategy') == 'strategy' and d.get('is_active')}
    pr = participation_ratio(series) if len(series) >= 2 else None
    if pr:
        out['participation_ratio'] = _round(pr.get('participation_ratio'), 4)
        out['pc1_share'] = _round(pr.get('pc1_share'), 4) if 'pc1_share' in pr else None
    return out


def _integrity_facts(store):
    if store is None:
        return None
    verification = getattr(store, 'verification', None)
    return {
        'status': (verification or {}).get('status'),
        'n_tickers_checked': (verification or {}).get('n_tickers_checked'),
        'worst_step': _round((verification or {}).get('worst_step'), 10),
        'n_violations': len((verification or {}).get('violations', []) or []),
        'readjusted': list(getattr(store, 'readjusted', [])),
    }


def _robustness_facts(metrics_data, rank_key):
    """The expensive block. Only computed when explicitly asked for."""
    from common import robustness as rb
    from common.margin_sizing import NotCalculable
    try:
        frame, binding = rb.common_frame(metrics_data)
    except NotCalculable as exc:
        return {'status': 'not computed', 'reason': str(exc)}
    rf = rb.realised_rf(metrics_data)
    out = {'rf_annual': _round(rf), 'binding': binding,
           'first': f'{frame.index[0]:%Y-%m}', 'last': f'{frame.index[-1]:%Y-%m}',
           'n_months': int(frame.shape[0]), 'n_strategies': int(frame.shape[1])}
    try:
        res = rb.pbo(frame=frame, binding=binding, rf_annual=rf)
        out['pbo_strategy'] = _round(res.pbo, 5)
        out['n_splits'] = int(res.n_splits)
        out['winner_share'] = [[n, _round(s, 5)] for n, s in res.winner_share[:5]]
    except NotCalculable as exc:
        out['pbo_strategy'] = None
        out['pbo_reason'] = str(exc)
    for grouping, fn in (('family', rb.family_of), ('mechanism', rb.mechanism_of)):
        try:
            grouped = rb.group_pbo(frame=frame, binding=binding, groups=fn,
                                   grouping=grouping, rf_annual=rf)
            out[f'pbo_{grouping}'] = _round(grouped.pbo, 5)
        except NotCalculable:
            out[f'pbo_{grouping}'] = None
    try:
        ranked = rb.rank_bootstrap(frame=frame, binding=binding, rank_key=rank_key,
                                   rf_annual=rf)
        out['rank_key'] = rank_key
        out['top'] = [{'name': r['name'], 'p_top_k': _round(r['p_top_k'], 5),
                       'p_first': _round(r['p_first'], 5)} for r in ranked.rows[:5]]
    except NotCalculable:
        out['top'] = []
    # The pooled view. Documentation quotes these ("the HAA family is top-3 in N% of
    # alternative histories"), so they belong in the artefact rather than in a sentence
    # somebody has to remember to re-measure.
    for grouping, fn in (('family', rb.family_of), ('mechanism', rb.mechanism_of)):
        try:
            grouped = rb.group_rank_bootstrap(frame=frame, binding=binding, groups=fn,
                                              grouping=grouping, rank_key=rank_key,
                                              rf_annual=rf)
            out[f'{grouping}_rank'] = {
                row['group']: {'p_top_k': _round(row['p_top_k'], 5),
                               'p_first': _round(row['p_first'], 5),
                               'best_member': row.get('best_member')}
                for row in grouped.rows}
        except NotCalculable:
            out[f'{grouping}_rank'] = {}
    return out


def build_facts(metrics_data, config=None, store=None, include_robustness=False,
                rank_key='sortino'):
    """The run, reduced to the numbers documentation is allowed to quote."""
    metrics_data = list(metrics_data or [])
    facts = {
        'generated_at': _dt.datetime.now().isoformat(timespec='seconds'),
        'registry': _registry_facts(),
        'window': _window_facts(metrics_data, config),
        'selection': _selection_facts(metrics_data) if metrics_data else {},
        'integrity': _integrity_facts(store),
        'metrics': _metric_facts(metrics_data),
    }
    if include_robustness and metrics_data:
        facts['robustness'] = _robustness_facts(metrics_data, rank_key)
    return facts


def write_facts(path, facts):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(facts, fh, indent=2, sort_keys=True, default=str)
        fh.write('\n')
    return path


def load_facts(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)
