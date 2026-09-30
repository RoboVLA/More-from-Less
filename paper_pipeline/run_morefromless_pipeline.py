"""Reproduce saved offline curves; optionally check new method reference code."""
import argparse
import json
import subprocess
import sys
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('paper_pipeline/configs/example_pipeline_config.json'))
    p.add_argument('--stage', action='append')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--include-disabled', action='store_true', help='Include optional reference checks in dry-run listing.')
    p.add_argument('--strict-inputs', action='store_true', help='Check inputs during dry-run too.')
    args=p.parse_args()
    config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    wanted=set(args.stage or [])
    unknown=wanted-{s['name'] for s in config['stages']}
    if unknown:p.error('Unknown stage: '+', '.join(sorted(unknown)))
    for stage in config['stages']:
        if wanted and stage['name'] not in wanted:continue
        if not wanted and not stage.get('enabled_by_default',True) and not (args.dry_run and args.include_disabled):continue
        if not stage.get('available',False):
            if wanted or args.include_disabled:print(f"[{stage['name']}] UNAVAILABLE: {stage['description']}")
            if wanted and not args.dry_run:return 2
            continue
        missing_modules=[m for m in stage.get('required_modules',[]) if importlib.util.find_spec(m) is None]
        if missing_modules and not args.dry_run:
            print('Install optional dependencies first: '+', '.join(missing_modules),file=sys.stderr);return 2
        command=[sys.executable if v=='{python}' else v for v in stage['command']]
        missing=[v for v in stage.get('required_inputs',[]) if not (ROOT/v).is_file()]
        print(f"[{stage['name']}] {stage.get('scope','example')}: {stage['description']}")
        if missing and (not args.dry_run or args.strict_inputs):
            print('Missing inputs: '+', '.join(missing),file=sys.stderr);return 2
        print('  '+' '.join(command))
        if not args.dry_run:
            result=subprocess.run(command,cwd=ROOT,check=False)
            if result.returncode:return result.returncode
    return 0
if __name__=='__main__':raise SystemExit(main())
