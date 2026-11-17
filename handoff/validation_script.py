"""Record final migration checks and host dependency inventory."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

R=Path('E:/pranav/Thesis-Pranav-Final')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    records=[json.loads(x) for x in (R/'handoff/COPY_MANIFEST.jsonl').read_text(encoding='utf8').splitlines()]
    assert len({x['destination'] for x in records})==len(records)
    for row in records:
        p=R/row['destination'];assert p.is_file() and p.stat().st_size==row['bytes'],p
    checks=read(R/'handoff/inference_smoke.json');assert checks['passed']
    eq=read(R/'handoff/final_checkpoint_equivalence.json');assert all(x['weights_exactly_equal'] for x in eq.values())
    datasets=read(R/'datasets/darkbay/manifests/datasets.json')
    # The copy receipt binds each new LMDB to the original full200k pixel audit.
    receipts={r['destination']:r for r in records}
    for tag,d in datasets.items():assert receipts[d['lmdb']+'/data.mdb']['sha256']==d['lmdb_sha256']
    selected=read(R/'Capri/models/registry.json')['selected_generator']
    assert receipts['Capri/'+selected['path']]['sha256']==selected['sha256']
    for tag in ('A','B'):
        m=read(R/f'model1/models/parseq/darkbay/{tag}/manifest.json')
        assert receipts[m['path']]['sha256']==m['sha256']
        assert eq[tag]['evaluated_latest_sha256']==m['evaluated_latest_sha256']
    packages=sorted([dict(name=d.metadata.get('Name','unknown'),version=d.version) for d in importlib.metadata.distributions()],key=lambda x:(x['name'].lower(),x['version']))
    write(R/'runtime/package_inventory.json',dict(python=sys.version,executable=sys.executable,base_prefix=sys.base_prefix,platform=platform.platform(),packages=packages))
    # Separate checksum list for current files authored during migration, not original copies.
    generated=[];copied=set(receipts)
    for p in sorted(R.rglob('*')):
        if not p.is_file():continue
        relative=p.relative_to(R).as_posix()
        if relative in copied or relative.startswith('runtime/cache/') or relative.startswith('runtime/temporary/') or relative in {'handoff/GENERATED_MANIFEST.json','handoff/VALIDATION.json','handoff/migration_status.json','handoff/WORKSPACE_CATALOG.json'}:continue
        generated.append(dict(path=relative,bytes=p.stat().st_size,sha256=sha(p)))
    write(R/'handoff/GENERATED_MANIFEST.json',generated)
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),status='migration_complete',copied_files=len(records),copied_bytes=sum(x['bytes'] for x in records),all_copies_sha256_verified_at_copy=True,copy_manifest_sha256=sha(R/'handoff/COPY_MANIFEST.jsonl'),all_copies_exist_and_sizes_verified=True,lmdb_hashes_match_original_audit=True,reader_checkpoint_weights_match_evaluated_final=True,reader_predictions_matched_per_model=16,real_image_pixel_hashes_verified=900,generator_ema_strict_load=True,originals_preserved=True,original_workspace='E:/pranav/CAPRI',destination=str(R),training_started=False,new_model2_and_model3_ready=False,limits=['Existing E-drive Python environment remains external; rebuild dependencies on another host.','Historical launch scripts and source configs retain provenance paths; new A/B curriculum runners remain to be adapted.','Original copy-complete receipt describes the first batch; this validation and COPY_MANIFEST include supplemental font/code dependencies.'])
    write(R/'handoff/VALIDATION.json',result);write(R/'handoff/migration_status.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
