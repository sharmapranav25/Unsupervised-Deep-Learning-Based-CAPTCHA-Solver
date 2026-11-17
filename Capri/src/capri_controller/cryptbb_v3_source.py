"""New CryptBB split -> selected CAPRI recipes -> exact-feature verification -> v3 fit.
Old controller sources are never edited; source binding is process-local.
"""
import argparse,copy,os,shutil,types
from pathlib import Path
from .decoder.common import read,write,sha,now
ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'handoff/cryptbb_v3_20261114'
SOURCE=ROOT/'Capri/runs/cryptbb_v3_source_20261114'
FIT=ROOT/'Capri/runs/cryptbb_controller_v3split_full_20261114'
SPLIT=ROOT/'evaluation_data/model4_reservation_20261114/cryptbb'

def recipe_only(preview=False,model_initializer=None,calibration_source=None,production_model=False):
    import numpy as np,torch
    from concurrent.futures import ProcessPoolExecutor
    from . import pipeline as P,runtime as U,controller_v2 as V
    from .controllers import Mixed
    from .artifacts import load_images
    from .controller_v3_split import FastMixed
    assert not preview and production_model and model_initializer is not None
    assert not SOURCE.exists(),'Existing source evidence preserved'
    SOURCE.mkdir(parents=True);P.ROOT=SOURCE
    refs=read(STUDY/'targets.json')['fit'];assert len(refs)==100 and len({r['pixel_sha256'] for r in refs})==100
    real=load_images(refs);write(SOURCE/'inputs.json',refs)
    model=Mixed();model_initializer(model);raw=[];cleaned=[];logits=[]
    hook=model.custom.hybrid.register_forward_hook(lambda mod,args,out:logits.append({k:v[0].detach().cpu().tolist() for k,v in out.items() if isinstance(v,torch.Tensor) and not k.startswith('ctc')}))
    try:
        for i,im in enumerate(real):
            prediction=model.custom.describe([im])[0];d,changes=P.clean(prediction)
            raw.append(dict(index=i,source=refs[i],prediction=prediction,logits=logits[-1]));cleaned.append(dict(index=i,description=d,changes=changes))
    finally:hook.remove()
    U.dump(SOURCE/'raw_hybrid_predictions.json',raw)
    counts={str(n):sum(r['prediction']['length']==n for r in raw) for n in range(2,9)}
    lengths,decision=P.choose_lengths(counts,preview=False)
    U.dump(SOURCE/'length_decision.json',dict(**decision,assigned_lengths=lengths,model_sha256=sha(model.custom.description_checkpoint),source='Fresh pixel-only predictions on the new reserved reference100; no label access'))
    recipes=[]
    for i,r in enumerate(cleaned):
        d=r['description'];old=d['length'];d['length']=lengths[i]
        r['changes'].append(dict(rule='model_majority_length',before=old,after=lengths[i],votes=decision['votes']))
        rr=P.recipe(d,1100003+i*7919,i);before=copy.deepcopy(rr['text_spec'])
        for k,lo,hi in [('rotation',0,45),('jitter',0,.2),('size_jitter',0,.3),('spacing',-.35,.4),('x_scale',.4,1.6),('curve',0,.3)]:rr['text_spec'][k]=float(np.clip(rr['text_spec'][k],lo,hi))
        r['renderer_before_clamping']=before;r['renderer_after_clamping']=rr['text_spec'];recipes.append(rr)
    U.dump(SOURCE/'refined_predictions.json',cleaned);U.dump(SOURCE/'refined_recipes.json',recipes)
    # Same native calibration procedure, now derived only from the new reference recipes.
    with ProcessPoolExecutor(max_workers=12,initializer=V._install) as pool:
        rows=list(pool.map(P.R.safe_render,[(r,P.R.expand(dict(length=r['description']['length']))) for r in recipes[:24]]))
        fast=FastMixed.__new__(FastMixed);fast.m=model;fast.pool=pool
        features=fast.extract([r.get('image',np.full((64,256,3),255,np.uint8)) for r in rows])
        np.savez(SOURCE/'calibration_features.npz',**{str(i)+'__'+k:v for i,part in enumerate(features) for k,v in part.items()})
    # v3 loads but does not use old entries. Explicit empty compatibility file; no legacy fit.
    write(SOURCE/'entries.json',[])
    write(SOURCE/'protocol.json',dict(created_at=now(),inputs=100,selected_generator=True,recipe_logic='pipeline.clean,choose_lengths,recipe and identical generic clamping',calibration='Fresh first24 synthetic recipes from these100 reference pixels; native pipeline neutral rendering',labels_used=False,old_reference_recipes_reused=False,compatibility_entries='Empty; v3 does not consume old entries'))
    write(SOURCE/'status.json',dict(state='complete',inputs=100,recipes=100,decision=decision,completed_at=now()))

def prepare():
    os.environ['CAPRI_RUN_DIR']=str(SOURCE);os.environ['CAPRI_TARGET_MANIFEST']=str(STUDY/'targets.json')
    from . import pipeline as P,controller_comparison as C
    C.NEW_RUN=STUDY/'selected_generator'
    original=P.main
    try:
        P.main=recipe_only
        C.worker(types.SimpleNamespace(out=str(SOURCE),targets=str(STUDY/'targets.json'),version='capri2',calibration=None),full=True)
    finally:P.main=original

def bound(stage):
    os.environ['CAPRI_RUN_DIR']=str(FIT);os.environ['CAPRI_TARGET_MANIFEST']=str(STUDY/'targets.json')
    from . import controller_v3_split as C
    C.V.SOURCES['cryptbb']=SOURCE
    if stage=='verify':
        diff=C.verify('cryptbb');assert diff==0.0
        write(STUDY/'fast_features_verification.json',dict(passed=True,max_abs_diff=diff,completed_at=now(),source=str(SOURCE),controller_sha256=sha(Path(C.__file__))))
    else:
        assert read(STUDY/'fast_features_verification.json')['passed']
        C.run('cryptbb',100,FIT,14)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','verify','fit']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    else:bound(a.stage)
