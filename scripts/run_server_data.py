"""Print a declarative data plan locally; execute it only at its server root."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tomllib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path('configs/server-data.toml'))
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    plan = tomllib.loads(args.plan.read_text())
    root = Path(plan['server_root'])
    if args.execute and Path.cwd().resolve() != root.resolve():
        p.error(f'Execution requires working directory {root}; local use is plan-only.')
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    logs = root/plan['log_root']
    logs.mkdir(parents=True, exist_ok=True)
    # Validate all destinations before launching the first download.
    for source in plan['sources']:
        dest = root/source['output']
        if dest.exists():
            raise FileExistsError(f'{dest} already exists; inspect it before making a new plan')
    failures = []
    for source in plan['sources']:
        command = [sys.executable, '-u', *source['command']]
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log = logs/f'{source["name"]}-{stamp}.log'
        print(f'[ACQUIRE] {source["name"]}\n[LOG] {log}', flush=True)
        with log.open('x') as handle:
            result = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        event = dict(source=source['name'], command=command, started_utc=stamp,
                     returncode=result.returncode, log=str(log), output=source['output'])
        with (logs/'runs.jsonl').open('a') as handle:
            handle.write(json.dumps(event)+'\n')
        if result.returncode:
            if plan.get('continue_on_error', False):
                failures.append(source['name'])
                print(f'[FAILED] {source["name"]}; see {log}', flush=True)
                continue
            raise RuntimeError(f'Acquisition failed: {source["name"]}; see {log}')
        manifest = root/source['output']/'manifest.json'
        if not manifest.is_file():
            raise RuntimeError(f'Missing manifest: {manifest}')
        print(f'[DONE] {source["name"]}\n[MANIFEST] {manifest}', flush=True)
    if failures:
        raise RuntimeError(f'Sources needing investigation: {failures}')


if __name__ == '__main__':
    main()
