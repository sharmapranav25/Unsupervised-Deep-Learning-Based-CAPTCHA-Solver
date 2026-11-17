"""Fresh source/fit/admission. Recipe failure is bounded and does not block export."""
import argparse,collections,copy,json,shutil,sys,time
from pathlib import Path
from . import common as W
from .common import ROOT,STUDY,SOURCE,FIT,DATA,read,write,sha,now

def references():
    import hashlib,numpy as np
    from PIL import Image
    refs=read(ROOT/'Capri/runs/controller_v3split_diverse4_darkbay_20261114/targets.json')['fit']
    evaluation=read(ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json')
    assert len(refs)==100 and len(evaluation)==900
    hashes=[]
    for r in refs:
        im=Image.open(r['source']).convert('RGB');h=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
        assert h==r['pixel_sha256'];hashes.append(h)
    assert len(set(hashes))==100 and not set(hashes)&{r['pixel_sha256'] for r in evaluation}
    write(STUDY/'targets.json',dict(fit=refs))
    write(STUDY/'reference_split.json',dict(verified_at=now(),references=100,evaluation=900,overlap=0,reference_sha256=sha(STUDY/'targets.json'),evaluation_manifest_sha256=sha(ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json')))
    (STUDY/'selected_generator').mkdir(exist_ok=True)
    selected=read(ROOT/'handoff/cryptbb_v3_20261114/selected_generator/completion.json')
    assert sha(Path(selected['checkpoint']))==selected['sha256']
    write(STUDY/'selected_generator/completion.json',selected)

def source():
    from ... import cryptbb_v3_source as S
    S.STUDY,S.SOURCE,S.FIT=STUDY,SOURCE,FIT
    S.prepare()

def fit():
    from ... import controller_v3_split as V
    V.V.SOURCES['darkbay']=SOURCE
    import os
    os.environ['CAPRI_RUN_DIR']=str(FIT);os.environ['CAPRI_TARGET_MANIFEST']=str(STUDY/'targets.json')
    if not (STUDY/'fast_features_verification.json').exists():
        diff=V.verify('darkbay');assert diff==0
        write(STUDY/'fast_features_verification.json',dict(passed=True,max_abs_diff=diff))
    V.run('darkbay',100,FIT,12)

def admit():
    # v3 already tries up to four candidate recipes on fresh seeds; one further
    # deterministic candidate pass is bounded, then a failed recipe is excluded.
    from concurrent.futures import ProcessPoolExecutor
    from ... import controller_v2 as V
    state=read(FIT/'status.json');assert state['state']=='complete' and state['inputs']==list(range(100))
    original=read(FIT/'entries.json');history=read(FIT/'controller_history.json');recipes=read(SOURCE/'refined_recipes.json')
    assert len(original)==400 and len(history)==100
    entries=copy.deepcopy(original);repairs=[]
    with ProcessPoolExecutor(max_workers=8,initializer=V._install) as pool:
        for entry in entries:
            W.stop()
            if not entry.get('unvalidated'):continue
            i=entry['input_index'];base=recipes[i]
            from ...controller_v3_split import OWNS
            def compatible(candidate):
                if entry['controller']=='mixed':return True
                changed={k for k in set(candidate['controls'])|set(entry['controls']) if candidate['controls'].get(k)!=entry['controls'].get(k)}
                return changed<=set(OWNS[entry['controller']])
            candidates=sorted([r for r in history[i]['records'] if r['engine']==entry['engine'] and r['usable']==V.FIT_SEEDS and compatible(r)],key=lambda r:r['scores'][entry['judge']])[:4]
            accepted=False
            for candidate in candidates:
                jobs=[(V.draw_recipe(base,1700003+i*1009+j*104729),candidate['controls'],entry['engine']) for j in range(4)]
                rows=list(pool.map(V.render_job,jobs))
                if all(V.usable(r,base['description']['length']) for r in rows):
                    entry.update(controls=candidate['controls'],unvalidated=False,repair_validation='Four additional deterministic fresh render seeds; original fit objective retained',validation_score=candidate['scores'][entry['judge']]);accepted=True;break
            repairs.append(dict(entry_id=entry['entry_id'],candidates_tried=len(candidates),repaired=accepted))
    accepted=[e for e in entries if e.get('unvalidated') is False]
    from .corpus import allocation
    counts=allocation(accepted)
    write(STUDY/'admitted_entries.json',entries)
    write(STUDY/'accepted_recipe_protocol.json',dict(created_at=now(),fit_sha256=sha(FIT/'entries.json'),admitted_sha256=sha(STUDY/'admitted_entries.json'),accepted_ids=[e['entry_id'] for e in accepted],rejected_ids=[e['entry_id'] for e in entries if e not in accepted],per_entry_counts=counts,repair_budget='At most four alternate candidates x four fresh seeds per failed recipe, then exclude',repairs=repairs,policy='User authorized proceeding after failed repairs; quotas redistributed within the same arm'))
    DATA.mkdir(exist_ok=True);shutil.copyfile(ROOT/'datasets/darkbay/A/charset.txt',DATA/'charset.txt')
    write(STUDY/'fit_completion.json',dict(completed_at=now(),entries=400,entries_sha256=sha(FIT/'entries.json'),accepted=len(accepted),rejected=400-len(accepted)))

def configs():
    import yaml
    for tag in 'AB':
        d=DATA/tag;r=read(d/'data_receipt.json')
        assert r['images']==r['unique_pixels']==100000
        assert sha(d/'train_lmdb/data.mdb')==r['lmdb_sha256']
        assert sha(d/'sample_provenance.jsonl')==r['provenance_sha256']
        cfg=yaml.safe_load((ROOT/f'model1/configs/cryptbb_v3/{tag}/prepared.yml').read_text())
        def relocate(v):
            if isinstance(v,dict):return {k:relocate(x) for k,x in v.items()}
            if isinstance(v,list):return [relocate(x) for x in v]
            if isinstance(v,str):return v.replace('cryptbb_v3',W.FAMILY)
            return v
        cfg=relocate(cfg);out=ROOT/'model1/configs'/W.FAMILY/tag/'prepared.yml';out.parent.mkdir(parents=True,exist_ok=True)
        assert not out.exists();out.write_text(yaml.safe_dump(cfg,sort_keys=False))
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in W.sources()]
    for tag in 'AB':
        p=ROOT/'model1/configs'/W.FAMILY/tag/'prepared.yml';manifest.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    dest=STUDY/'model1';dest.mkdir(exist_ok=True)
    write(dest/'agreement_protocol.json',dict(selection='Fixed final20 primary; real per-epoch diagnostics only',teacher='Opposite Model1 epoch20'))
    protocol=dict(sources=manifest,configs=[dict(dataset=t,path=(ROOT/'model1/configs'/W.FAMILY/t/'prepared.yml').relative_to(ROOT).as_posix(),sha256=sha(ROOT/'model1/configs'/W.FAMILY/t/'prepared.yml')) for t in 'AB'],selection_protocol_sha256=sha(dest/'agreement_protocol.json'),seed=48,epochs=20,batch_size=128,steps_per_epoch=782,steps_each=15640,draws_per_epoch=100096,initialization='independent scratch',execution='A/B concurrent; complete GPU batches serialized',real_labels_used=False)
    write(dest/'training_protocol.json',protocol);shutil.copyfile(dest/'training_protocol.json',dest/'protocol.json')
    write(STUDY/'datasets_ready.json',dict(created_at=now(),images_each=100000,datasets={t:read(DATA/t/'data_receipt.json') for t in 'AB'}))

def main():
    W.environment();W.verify_sources();W.stop()
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['references','source','fit','admit','configs']);a=p.parse_args();globals()[a.stage]()
if __name__=='__main__':main()
