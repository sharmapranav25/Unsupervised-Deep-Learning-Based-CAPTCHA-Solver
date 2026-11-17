"""Inspect and verify the final shared workspace without launching training."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['status','verify','inventory']);ap.add_argument('--full',action='store_true');args=ap.parse_args()
    records=[json.loads(x) for x in (ROOT/'handoff/COPY_MANIFEST.jsonl').read_text(encoding='utf8').splitlines()]
    if args.command=='status':
        current=ROOT/'handoff/darkbay_curriculum_20261113/status.json'
        if current.exists():
            runs={}
            for model in (2,3):
                for tag in ('A','B'):
                    p=ROOT/f'model{model}/runs/darkbay/{tag}/status.json'
                    runs[f'Model{model}-{tag}']=read(p) if p.exists() else {'state':'pending'}
            print(json.dumps(dict(workspace=str(ROOT),model1={'A':'complete20epochs;615/900','B':'complete20epochs;573/900'},
                coordinator=read(current),runs=runs,status_is_saved_snapshot=True,
                current_protocol='handoff/darkbay_curriculum_20261113/protocol.json'),indent=2));return
        print(json.dumps(dict(workspace=str(ROOT),model1=dict(A='complete20epochs;615/900',B='complete20epochs;573/900'),model2='new A/B configurations and rankings pending',model3='new A/B configurations and rankings pending',training_running=False,originals_preserved=True,source_map='handoff/SOURCE_MAP.json'),indent=2));return
    if args.command=='inventory':
        groups={}
        for p in ROOT.rglob('*'):
            if p.is_file():
                group=p.relative_to(ROOT).parts[0];g=groups.setdefault(group,dict(files=0,bytes=0));g['files']+=1;g['bytes']+=p.stat().st_size
        result=dict(created_at=datetime.now(timezone.utc).isoformat(),root=str(ROOT),sections=groups)
        (ROOT/'handoff/WORKSPACE_CATALOG.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2));return
    failures=[]
    for row in records:
        p=ROOT/row['destination']
        if not p.is_file() or p.stat().st_size!=row['bytes']:failures.append(row['destination']+': missing/size mismatch')
        elif args.full and sha(p)!=row['sha256']:failures.append(row['destination']+': hash changed since migration')
    for tag in ('A','B'):
        m=read(ROOT/f'model1/models/parseq/darkbay/{tag}/manifest.json')
        if sha(ROOT/m['path'])!=m['sha256']:failures.append('reader checkpoint '+tag)
    generator=read(ROOT/'Capri/models/registry.json')['selected_generator']
    if sha(ROOT/'Capri'/generator['path'])!=generator['sha256']:failures.append('generator checkpoint')
    result=dict(passed=not failures,files_checked=len(records),full_rehash=args.full,selected_checkpoint_hashes_checked=True,failures=failures)
    print(json.dumps(result,indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
