"""Requested CryptBB Model1 diagnostic; CPU only, immutable checkpoint, labels after freeze."""
import argparse, copy, gc, hashlib, inspect, os, shutil, textwrap, time, types
from pathlib import Path
from ..decoder.common import read,write,sha,now,StageLock
ROOT=Path(__file__).resolve().parents[4]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['A','B'],required=True);ap.add_argument('--epoch',type=int,required=True);a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np,torch
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from rapidfuzz.distance import Levenshtein
    torch.set_num_threads(2)
    run=ROOT/f'model1/runs/cryptbb/{a.dataset}';checkpoint=run/f'epoch_{a.epoch}.pth'
    receipt=read(run/f'epoch_{a.epoch}_receipt.json');assert receipt['epoch']==a.epoch and receipt['global_step']==782*a.epoch
    digest=sha(checkpoint);assert digest==receipt['sha256']
    out=ROOT/f'model1/results/cryptbb/{a.dataset}/requested_cpu_epoch{a.epoch:02d}'
    if (out/'summary.json').exists():
        summary=read(out/'summary.json');assert summary['checkpoint_sha256']==digest and sha(out/'predictions.json')==read(out/'prediction_freeze.json')['sha256'];print(summary);return
    if out.exists():raise RuntimeError('Preserve incomplete evaluation for inspection: '+str(out))
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'evaluator_source.py')
    with StageLock(out,'evaluation'):
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['epoch']==a.epoch
        cfg=copy.deepcopy(ck['config']);dec=build_post_process(cfg['PostProcess'],cfg['Global']);cfg['Architecture']['Decoder']['out_channels']=dec.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(ck['state_dict'],strict=True);model.cpu().eval();del ck;gc.collect()
        original=model.decoder.forward_test.__func__;source=textwrap.dedent(inspect.getsource(original));assert source.count('_device = memory.get_device()')==1
        adapted=source.replace('_device = memory.get_device()','_device = memory.device');(out/'cpu_forward_test.py').write_text(adapted,encoding='utf8')
        namespace=dict(original.__globals__);exec(compile(adapted,str(out/'cpu_forward_test.py'),'exec'),namespace);model.decoder.forward_test=types.MethodType(namespace['forward_test'],model.decoder)
        manifest=ROOT/'evaluation_data/cryptbb/image_manifest.json';rows=read(manifest);assert len(rows)==551 and len({r['opaque_id'] for r in rows})==551
        policy=read(ROOT/'handoff/cryptbb_model1_preparation/agreement_protocol.json');assert sha(manifest)==policy['manifest_sha256']
        sources=read(ROOT/'handoff/cryptbb_model1_preparation/source_pixels.json')
        excluded={r['pixel_sha256'] for role in ['reference','selftrain'] for r in sources[role]};assert not excluded&{r['pixel_sha256'] for r in rows}
        pixels=[]
        for r in rows:
            im=Image.open(r['image_path']).convert('RGB');assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256'];pixels.append(im)
        protocol=dict(created_at=now(),model=1,dataset=a.dataset,epoch=a.epoch,checkpoint_sha256=digest,manifest_sha256=sha(manifest),images=551,device='cpu',precision='FP32',threads=2,batch_size=16,preprocessing='RGB PIL bicubic32x128 ToTensor Normalize0.5',decoding='Saved free AR and1 refinement; no length forcing or text normalization',purpose='User-requested diagnostic only; no training/curriculum/selection changes',labels='Opened only after persisted prediction freeze',cpu_adapter='Process-local memory.get_device() -> memory.device',decoder_source_sha256=sha(Path(inspect.getsourcefile(original))),evaluator_sha256=sha(Path(__file__)))
        write(out/'protocol.json',protocol);tr=T.Compose([T.Resize((32,128),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
        values=[];start=time.monotonic()
        with torch.inference_mode():
            for i in range(0,len(rows),16):
                decoded=dec(model(torch.stack([tr(im) for im in pixels[i:i+16]])))
                assert len(decoded)==len(rows[i:i+16])
                values.extend(dict(opaque_id=r['opaque_id'],prediction=p[0]) for r,p in zip(rows[i:i+16],decoded))
                write(out/'status.json',dict(state='predicting',processed=len(values),total=551,updated_at=now()))
                if i%80==0:print(f'Predicted {len(values)}/551',flush=True)
        assert len(values)==551 and sha(checkpoint)==digest
        write(out/'predictions.json',values);write(out/'prediction_freeze.json',dict(frozen_at=now(),sha256=sha(out/'predictions.json'),checkpoint_sha256=digest,images=551))
        labels=read(ROOT/'evaluation_data/cryptbb/scoring_labels/scoring_labels.json');assert set(labels)=={r['opaque_id'] for r in values}
        correct=sum(r['prediction']==labels[r['opaque_id']] for r in values);edits=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in values)
        summary=dict(model=1,dataset=a.dataset,epoch=a.epoch,correct=correct,total=551,accuracy_percent=100*correct/551,cer_percent=100*edits/sum(len(x) for x in labels.values()),checkpoint_sha256=digest,device='cpu',seconds=time.monotonic()-start,diagnostic_only=True,training_changed=False)
        write(out/'per_image_results.json',[dict(**r,label=labels[r['opaque_id']],correct=r['prediction']==labels[r['opaque_id']]) for r in values])
        assert sum(r['correct'] for r in read(out/'per_image_results.json'))==correct
        assert sha(out/'predictions.json')==read(out/'prediction_freeze.json')['sha256'] and sha(checkpoint)==digest
        training_protocol=read(ROOT/'handoff/cryptbb_model1_training_20261113/protocol.json')
        for r in training_protocol['sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
        write(out/'verification.json',dict(passed=True,predictions_frozen_before_labels=True,identities_verified=551,checkpoint_hash_verified=True,training_sources_unchanged=True,correct_recounted=correct))
        write(out/'summary.json',summary);write(out/'status.json',dict(state='complete',updated_at=now()));print(summary,flush=True)
if __name__=='__main__':main()
