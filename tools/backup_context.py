"""
Back up the local-only agent bundle into its own git repository.

WHY THIS EXISTS
---------------
`AGENTS.md`, `CLAUDE.md`, `SETUP.md`, `memory/PROJECT.md` and the skills are gitignored by
decision: this repository publishes the software, not the working notes about how it is
developed. The consequence is that the project's decision record is **unversioned**, and it
is the only record left — the git history was squashed to a single commit on 2026-08-29, and
the agent harness's private memory was orphaned the same day when the project folder moved.
Two of the three context stores were lost in a single day; the third currently exists in one
copy, on one disk, with no history.

This copies that bundle into a separate repository so it has versions and can live somewhere
else. It is deliberately a SEPARATE repo rather than a branch: a branch of the public
repository would be public, which is the thing the gitignore rules exist to prevent.

USAGE
-----
    venv/Scripts/python.exe -m tools.backup_context
    venv/Scripts/python.exe -m tools.backup_context --dest "D:/backups/ks-context"

Defaults to a sibling directory `keller-strategies-context`. It initialises the repo on
first run and commits on every run; it never adds a remote and never pushes. Making this
leave the machine is a decision for the owner, not a side effect of running a script -- the
bundle describes their positions and their reasoning, and where that goes is theirs to
choose. The command to do it is printed at the end.
"""
import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Exactly what the .gitignore keeps out of the public repository, and nothing else.
BUNDLE = ('AGENTS.md', 'CLAUDE.md', 'SETUP.md', 'memory', '.claude/skills', '.agents/skills')

#: Never copied, even though it sits in an included directory: one machine's permission
#: grants, and the file the whole privacy convention exists for.
NEVER = {'settings.local.json', 'user_config.json'}

README = """# keller-strategies — context bundle

The working notes for [`keller-strategies`](https://github.com/SimonKreis-Richard/keller-strategies),
kept out of that repository on purpose: it publishes the software, this holds the reasoning.

`AGENTS.md` is the entry point; `memory/PROJECT.md` is the decision record — what was
decided, on what criterion, what was rejected and why, and what is still open.

Written by `tools/backup_context.py` in the main repository. Do not edit here: edit in the
project and re-run the backup, or the two copies will disagree and the older one will look
just as authoritative.

**Keep this repository private.** It describes the owner's positions, preconditions and
reasoning. Nothing in it is a credential, and none of it belongs in public either.
"""


def _git(args, cwd):
    return subprocess.run(('git',) + tuple(args), cwd=cwd, capture_output=True,
                          text=True, encoding='utf-8', errors='replace')


def _copy(src, dest):
    if os.path.isdir(src):
        for name in sorted(os.listdir(src)):
            if name in NEVER or name == '.git':
                continue
            _copy(os.path.join(src, name), os.path.join(dest, name))
    elif os.path.isfile(src):
        if os.path.basename(src) in NEVER:
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Back up the local-only agent bundle.')
    parser.add_argument('--dest', default=os.path.join(os.path.dirname(ROOT_DIR),
                                                       'keller-strategies-context'))
    args = parser.parse_args(argv)
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    if not os.path.isdir(os.path.join(dest, '.git')):
        _git(['init', '-q'], cwd=dest)
        print('initialised a repository at {}'.format(dest))

    copied, missing = [], []
    for rel in BUNDLE:
        src = os.path.join(ROOT_DIR, *rel.split('/'))
        if not os.path.exists(src):
            missing.append(rel)
            continue
        _copy(src, os.path.join(dest, *rel.split('/')))
        copied.append(rel)

    with open(os.path.join(dest, 'README.md'), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(README)

    # A copied `user_config.json` would defeat the whole privacy convention, so the check is
    # made against what actually landed rather than against what was meant to be skipped.
    for walk_root, _dirs, files in os.walk(dest):
        if '.git' in walk_root.split(os.sep):
            continue
        for name in files:
            if name in NEVER:
                raise SystemExit('REFUSING: {} reached the backup'.format(
                    os.path.join(walk_root, name)))

    _git(['add', '-A'], cwd=dest)
    status = _git(['status', '--porcelain'], cwd=dest).stdout.strip()
    if status:
        message = 'context bundle as of {}'.format(_dt.date.today().isoformat())
        result = _git(['commit', '-q', '-m', message], cwd=dest)
        if result.returncode != 0:
            print(result.stdout or result.stderr)
    print('\ncopied : {}'.format(', '.join(copied)))
    if missing:
        print('missing: {}  (nothing to back up for these)'.format(', '.join(missing)))
    print('changes: {}'.format('committed' if status else 'none since the last backup'))
    print('\nNot pushed anywhere. To give it an off-machine home, create a PRIVATE repo and:')
    print('    cd "{}"'.format(dest))
    print('    gh repo create keller-strategies-context --private --source=. --push')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
