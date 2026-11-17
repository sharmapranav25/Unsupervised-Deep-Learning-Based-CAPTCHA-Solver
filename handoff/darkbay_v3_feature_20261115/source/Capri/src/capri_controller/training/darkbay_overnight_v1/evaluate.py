"""Finite CPU-only diagnostics on the original900; freeze before scoring."""
import argparse,copy,gc,hashlib,inspect,os,sys,textwrap,time,types
from pathlib import Path
from . import common as W
from .common import ROOT,STUDY,FAMILY,read,write,sha,now
from ...decoder.common import StageLock

def evaluate(number,tag,receipt):
    import numpy as np,torch
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from rapidfuzz.distance import Levenshtein
    epoch=receipt['epoch'];run=W.rundir(number,tag);checkpoint=run/receipt['checkpoint']
    digest=sha(checkpoint);assert digest==receipt['sha256']
    protocol=STUDY/'model1/protocol.json' if number==1 else W.stage(number,tag)/'protocol.json'
    assert receipt['protocol_sha256']==sha(protocol)
    out=ROOT/f'model{number}/results'/FAMILY/tag/f'epoch_{epoch:02d}'
    if (out/'summary.json').exists():
        s=read(out/'summary.json');assert s['checkpoint_sha256']==digest and read(out/'verification.json')['passed']
        assert sha(out/'predictions.json')==read(out/'prediction_freeze.json')['sha256'];return s
    assert not out.exists(),'Preserve incomplete evaluator output for explicit recovery'
    out.mkdir(parents=True)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['epoch']==epoch and ck['global_step']==782*epoch and ck['protocol_sha256']==sha(protocol) and ck['config']['Global']['seed']==48
    cfg=copy.deepcopy(ck['config']);decoder=build_post_process(cfg['PostProcess'],cfg['Global']);cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
    model=build_model(cfg['Architecture']);model.load_state_dict(ck['state_dict'],strict=True);model.cpu().eval();del ck;gc.collect()
    original=model.decoder.forward_test.__func__;code=textwrap.dedent(inspect.getsource(original));assert code.count('_device = memory.get_device()')==1
    adapted=code.replace('_device = memory.get_device()','_device = memory.device');(out/'cpu_forward_test.py').write_text(adapted)
    namespace=dict(original.__globals__);exec(compile(adapted,str(out/'cpu_forward_test.py'),'exec'),namespace);model.decoder.forward_test=types.MethodType(namespace['forward_test'],model.decoder)
    manifest=ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json';rows=read(manifest)
    assert len(rows)==len({r['opaque_id'] for r in rows})==900
    assert sha(manifest)==read(STUDY/'reference_split.json')['evaluation_manifest_sha256']
    assert not {r['pixel_sha256'] for r in rows}&{r['pixel_sha256'] for r in read(STUDY/'targets.json')['fit']}
    images=[]
    for r in rows:
        im=Image.open(ROOT/r['image_path']).convert('RGB');assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256'];images.append(im)
    write(out/'protocol.json',dict(model=number,dataset=tag,epoch=epoch,checkpoint_sha256=digest,training_protocol_sha256=sha(protocol),manifest_sha256=sha(manifest),images=900,device='cpu',precision='FP32',batch=16,threads=2,labels_after_freeze=True,selection='Fixed final20 primary; interim diagnostics do not steer training',historical_test=True))
    transform=T.Compose([T.Resize((32,128),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
    values=[];start=time.monotonic()
    with torch.inference_mode():
        for i in range(0,900,16):
            W.stop();pred=decoder(model(torch.stack([transform(im) for im in images[i:i+16]])))
            values.extend(dict(opaque_id=r['opaque_id'],prediction=p[0]) for r,p in zip(rows[i:i+16],pred))
            write(out/'status.json',dict(state='predicting',processed=len(values),total=900,updated_at=now()))
            if i%160==0:print(number,tag,epoch,len(values),'/900',flush=True)
    assert len(values)==900 and sha(checkpoint)==digest
    write(out/'predictions.json',values);write(out/'prediction_freeze.json',dict(sha256=sha(out/'predictions.json'),checkpoint_sha256=digest,frozen_at=now()))
    labels=read(ROOT/'evaluation_data/darkbay/scoring_labels/scoring_labels.json')
    assert set(labels)=={r['opaque_id'] for r in values}
    correct=sum(r['prediction']==labels[r['opaque_id']] for r in values);edits=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in values)
    scored=[dict(**r,label=labels[r['opaque_id']],correct=r['prediction']==labels[r['opaque_id']]) for r in values];write(out/'per_image_results.json',scored)
    assert sum(r['correct'] for r in read(out/'per_image_results.json'))==correct
    assert sha(out/'predictions.json')==read(out/'prediction_freeze.json')['sha256'] and sha(checkpoint)==digest
    W.verify_sources();write(out/'verification.json',dict(passed=True,all900_pixels_verified=True,reference100_excluded=True,predictions_frozen_before_labels=True,checkpoint_rehashed=True,correct_recounted=correct))
    summary=dict(model=number,dataset=tag,epoch=epoch,correct=correct,total=900,accuracy_percent=100*correct/900,cer_percent=100*edits/sum(map(len,labels.values())),checkpoint_sha256=digest,seconds=time.monotonic()-start,diagnostic_only=True)
    write(out/'summary.json',summary);write(out/'status.json',dict(state='complete',updated_at=now()));return summary

def main():
    W.environment();os.environ['CUDA_VISIBLE_DEVICES']='';W.verify_sources()
    import torch
    torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all']);p.parse_args()
    with StageLock(STUDY,'evaluation'):
        capture=[]
        for number in (1,2,3):
            for tag in 'AB':
                paths=sorted(W.rundir(number,tag).glob('epoch_*_receipt.json')) if number==1 else sorted((W.rundir(number,tag)/'checkpoints').glob('epoch_*.json'))
                capture.extend((number,tag,read(path)) for path in paths)
        write(STUDY/'evaluation_snapshot.json',dict(at=now(),checkpoints=capture))
        results=[]
        for number,tag,r in sorted(capture,key=lambda x:(x[0],x[1],x[2]['epoch'])):
            results.append(evaluate(number,tag,r));write(STUDY/'results.json',dict(updated_at=now(),results=results,selection='Fixed final20',complete=len(results)==120))
        final=[r for r in results if r['epoch']==20]
        lines=['# Fresh DarkBay v3 reader results','','Historical real900, reference100 excluded. Fixed final20.','','| Model | Dataset | Correct | Accuracy |','|---|---|---:|---:|']
        lines.extend(f"| {r['model']} | {r['dataset']} | {r['correct']}/900 | {r['accuracy_percent']:.2f}% |" for r in final)
        lines+=['',f'Verified {len(results)}/120 checkpoint evaluations. Training and full-pool assessment time are separate in each Model3 completion receipt.']
        (STUDY/'RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
