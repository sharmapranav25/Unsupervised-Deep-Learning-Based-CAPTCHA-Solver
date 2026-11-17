"""Requested CryptBB v3 feature-priority Model3 diagnostic on reserved251; CPU only, immutable checkpoint, labels after freeze."""
import argparse, copy, gc, hashlib, inspect, os, shutil, sys, textwrap, time, types
from pathlib import Path
from ..decoder.common import read,write as atomic_write,sha,now,StageLock
ROOT=Path(__file__).resolve().parents[4]

def write(path,value):
    for attempt in range(20):
        try:return atomic_write(path,value)
        except PermissionError:
            if attempt==19:raise
            time.sleep(0.1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['B'],required=True);ap.add_argument('--epoch',type=int,required=True);ap.add_argument('--attempt',type=int,default=1);ap.add_argument('--request-manifest',required=True);a=ap.parse_args();assert a.attempt>=1
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path[:0]=[str(ROOT/'model1/src/parseq_packages'),str(ROOT/'model1/src/OpenOCR')]
    from . import cryptbb_feature_b_v1 as M
    training_digest=M.check_sources()
    import numpy as np,torch
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from rapidfuzz.distance import Levenshtein
    torch.set_num_threads(2)
    run=M.OUT;checkpoint=run/f'checkpoints/epoch_{a.epoch:02d}.pth'
    history=read(run/'history.json');assert a.epoch in [r['epoch'] for r in history]
    request=read(Path(a.request_manifest))
    receipt=next(r for r in request['checkpoints'] if r['dataset']==a.dataset and r['epoch']==a.epoch)
    digest=sha(checkpoint);assert digest==receipt['sha256']
    committed=read(checkpoint.with_suffix('.json'));assert committed['sha256']==digest and committed['protocol_sha256']==training_digest
    assert sha(run/f'assessments/epoch_{a.epoch:02d}/feedback.json')==committed['assessment_sha256']
    assert a.attempt==1
    out=ROOT/'model3/runs'/M.FAMILY/'diagnostics/real_cryptbb'/f'epoch_{a.epoch:02d}'
    if (out/'summary.json').exists():
        summary=read(out/'summary.json');assert summary['checkpoint_sha256']==digest and sha(out/'predictions.json')==read(out/'prediction_freeze.json')['sha256']
        assert read(out/'verification.json')['passed'];print(summary);return
    if out.exists():raise RuntimeError('Preserve incomplete evaluation for inspection: '+str(out))
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'evaluator_source.py')
    with StageLock(out,'evaluation'):
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['epoch']==a.epoch and ck['global_step']==782*a.epoch and ck['protocol_sha256']==training_digest and ck['config']['Global']['seed']==48
        cfg=copy.deepcopy(ck['config']);dec=build_post_process(cfg['PostProcess'],cfg['Global']);cfg['Architecture']['Decoder']['out_channels']=dec.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(ck['state_dict'],strict=True);model.cpu().eval();del ck;gc.collect()
        original=model.decoder.forward_test.__func__;source=textwrap.dedent(inspect.getsource(original));assert source.count('_device = memory.get_device()')==1
        adapted=source.replace('_device = memory.get_device()','_device = memory.device');(out/'cpu_forward_test.py').write_text(adapted,encoding='utf8')
        namespace=dict(original.__globals__);exec(compile(adapted,str(out/'cpu_forward_test.py'),'exec'),namespace);model.decoder.forward_test=types.MethodType(namespace['forward_test'],model.decoder)
        manifest=ROOT/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json';rows=read(manifest);assert len(rows)==251 and len({r['opaque_id'] for r in rows})==251
        policy=read(ROOT/'handoff/cryptbb_v3_training_20261114/agreement_protocol.json');assert sha(manifest)==policy['manifest_sha256']
        sources={role:read(ROOT/f'evaluation_data/model4_reservation_20261114/cryptbb/{role}.json') for role in ['reference_pool','pseudo_label_pool']}
        excluded={r['pixel_sha256'] for role in sources for r in sources[role]};assert not excluded&{r['pixel_sha256'] for r in rows}
        pixels=[]
        for r in rows:
            im=Image.open(ROOT/r['image_path']).convert('RGB');assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256'];pixels.append(im)
        protocol=dict(created_at=now(),training_protocol_sha256=training_digest,checkpoint=str(checkpoint.relative_to(ROOT)),seed=48,model=3,dataset=a.dataset,epoch=a.epoch,checkpoint_sha256=digest,manifest_sha256=sha(manifest),images=251,device='cpu',precision='FP32',threads=2,batch_size=16,preprocessing='RGB PIL bicubic32x128 ToTensor Normalize0.5',decoding='Saved free AR and1 refinement; no length forcing or text normalization',purpose='User-requested diagnostic only; no training/curriculum/selection changes',labels='Opened only after persisted prediction freeze',cpu_adapter='Process-local memory.get_device() -> memory.device',decoder_source_sha256=sha(Path(inspect.getsourcefile(original))),evaluator_sha256=sha(Path(__file__)))
        write(out/'protocol.json',protocol);tr=T.Compose([T.Resize((32,128),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
        values=[];start=time.monotonic()
        with torch.inference_mode():
            for i in range(0,len(rows),16):
                decoded=dec(model(torch.stack([tr(im) for im in pixels[i:i+16]])))
                assert len(decoded)==len(rows[i:i+16])
                values.extend(dict(opaque_id=r['opaque_id'],prediction=p[0]) for r,p in zip(rows[i:i+16],decoded))
                write(out/'status.json',dict(state='predicting',processed=len(values),total=251,updated_at=now()))
                if i%80==0:print(f'Predicted {len(values)}/251',flush=True)
        assert len(values)==251 and sha(checkpoint)==digest
        write(out/'predictions.json',values);write(out/'prediction_freeze.json',dict(frozen_at=now(),sha256=sha(out/'predictions.json'),checkpoint_sha256=digest,images=251))
        label_path=ROOT/'evaluation_data/model4_reservation_20261114/scoring_only/cryptbb/evaluation_pool_labels.json'
        if not (out.parent/'evaluation_inputs.json').exists():
            write(out.parent/'evaluation_inputs.json',dict(labels_sha256=sha(label_path),manifest_sha256=sha(manifest),frozen_at=now(),predictions_already_frozen=True))
        assert sha(label_path)==read(out.parent/'evaluation_inputs.json')['labels_sha256']
        labels=read(label_path);assert set(labels)=={r['opaque_id'] for r in values}
        correct=sum(r['prediction']==labels[r['opaque_id']] for r in values);edits=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in values)
        summary=dict(model=3,dataset=a.dataset,epoch=a.epoch,correct=correct,total=251,accuracy_percent=100*correct/251,cer_percent=100*edits/sum(len(x) for x in labels.values()),checkpoint_sha256=digest,device='cpu',seconds=time.monotonic()-start,diagnostic_only=True,training_changed=False)
        write(out/'per_image_results.json',[dict(**r,label=labels[r['opaque_id']],correct=r['prediction']==labels[r['opaque_id']]) for r in values])
        assert sum(r['correct'] for r in read(out/'per_image_results.json'))==correct
        assert sha(out/'predictions.json')==read(out/'prediction_freeze.json')['sha256'] and sha(checkpoint)==digest
        assert M.check_sources()==training_digest
        write(out/'verification.json',dict(passed=True,predictions_frozen_before_labels=True,identities_verified=251,checkpoint_hash_verified=True,training_sources_unchanged=True,correct_recounted=correct))
        write(out/'summary.json',summary);write(out/'status.json',dict(state='complete',updated_at=now()));print(summary,flush=True)
if __name__=='__main__':main()
