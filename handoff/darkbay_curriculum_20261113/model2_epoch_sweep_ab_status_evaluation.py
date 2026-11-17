"""Requested epoch-matched A/B diagnostics, isolated from the frozen trainer."""
import argparse
import copy
from contextlib import nullcontext
import gc
import hashlib
import inspect
import os
from pathlib import Path
import sys
import textwrap
import time
import types

from .ab_curriculum import ROOT, STUDY, read, write, sha, now, gpu_slot


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--epoch',type=int,required=True)
    ap.add_argument('--model',type=int,choices=[2,3],default=2)
    ap.add_argument('--model-only',action='store_true')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=ap.parse_args()
    assert args.seed==48
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path[:0]=[str(ROOT/'model1/src/parseq_packages'),str(ROOT/'model1/src/OpenOCR')]
    import numpy as np
    import torch
    from PIL import Image
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from .ds2_data import transform
    torch.set_num_threads(2)
    slot = gpu_slot if args.device=='cuda' else nullcontext
    batch_size = 64 if args.device=='cuda' else 16
    if args.device=='cuda':
        assert torch.cuda.is_available()
    manifest=read(ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json')
    assert len(manifest)==900
    refs=read(ROOT/'Capri/configs/targets/darkbay_final.json')['fit']
    assert not {r['pixel_sha256'] for r in refs}&{r['pixel_sha256'] for r in manifest}
    pixels=[]
    for r in manifest:
        im=Image.open(ROOT/r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
        pixels.append(im)
    frozen=[]; tr=transform()
    for number in ((args.model,) if args.model_only else (args.model,1)):
        for tag in ('A','B'):
            name=f'epoch_{args.epoch:02d}.pth' if number in (2,3) else f'epoch_{args.epoch}.pth'
            path=ROOT/f'model{number}/runs/darkbay/{tag}/checkpoints'/name
            if number==1 and args.epoch==20 and not path.exists():
                path=ROOT/f'model1/models/parseq/darkbay/{tag}/epoch_20.pth'
            digest=sha(path)
            out=ROOT/f'model{number}/results/darkbay/{tag}/interim_{args.device}_epoch{args.epoch:02d}_seed{args.seed}'
            if (out/'prediction_freeze.json').exists():
                freeze=read(out/'prediction_freeze.json')
                assert freeze['checkpoint_sha256']==digest
                assert freeze['predictions_sha256']==sha(out/'predictions.json')
                frozen.append((number,tag,out)); continue
            assert not out.exists(), 'Preserve incomplete evaluation for inspection: '+str(out)
            out.mkdir(parents=True)
            ck=torch.load(path,map_location='cpu',weights_only=False)
            assert ck['epoch']==args.epoch and ck['config']['Global']['seed']==args.seed
            cfg=copy.deepcopy(ck['config'])
            charset=str(ROOT/f'datasets/darkbay/{tag}/charset.txt')
            cfg['PostProcess']['character_dict_path']=charset
            cfg['Global']['character_dict_path']=charset
            dec=build_post_process(cfg['PostProcess'],cfg['Global'])
            cfg['Architecture']['Decoder']['out_channels']=dec.get_character_num()
            model=build_model(cfg['Architecture'])
            model.load_state_dict(ck['state_dict'],strict=True); del ck
            with slot():
                model.to(args.device).eval()
            original=model.decoder.forward_test.__func__
            source=textwrap.dedent(inspect.getsource(original))
            assert source.count('_device = memory.get_device()')==1
            adapted=source.replace('_device = memory.get_device()','_device = memory.device')
            namespace=dict(original.__globals__)
            exec(compile(adapted,'<cpu-device-adapter>','exec'),namespace)
            model.decoder.forward_test=types.MethodType(namespace['forward_test'],model.decoder)
            write(out/'protocol.json',dict(model=number,dataset=tag,epoch=args.epoch,seed=args.seed,
                checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=digest,
                device=args.device,threads=2,batch_size=batch_size,precision='FP32',images=900,
                gpu_scheduling='Existing training GPU lock per inference batch; no training state changes' if args.device=='cuda' else None,
                decoding='Same saved free AR decoder plus one refinement; no forced length or case normalization',
                cpu_adapter='Process-local memory.get_device() -> memory.device; training sources unchanged',
                evaluator_sha256=sha(Path(__file__)),created_at=now(),
                selection='User-requested latest fully saved epoch captured at start; diagnostics only',
                labels='Opened only after every requested prediction file is frozen',
                training_changed=False,historical_repeatedly_inspected_test=True))
            rows=[]; start=time.monotonic()
            with torch.inference_mode():
                for i in range(0,900,batch_size):
                    x=torch.stack([tr(im) for im in pixels[i:i+batch_size]])
                    with slot():
                        decoded=dec(model(x.to(args.device)))
                    rows.extend(dict(opaque_id=r['opaque_id'],prediction=p[0])
                                for r,p in zip(manifest[i:i+batch_size],decoded))
                    if i%(batch_size*5)==0:
                        print(f'Model{number}-{tag} {len(rows)}/900',flush=True)
                        write(out/'status.json',dict(state='predicting',processed=len(rows),total=900))
            assert len(rows)==900 and sha(path)==digest
            write(out/'predictions.json',rows)
            write(out/'prediction_freeze.json',dict(checkpoint_sha256=digest,
                predictions_sha256=sha(out/'predictions.json'),images=900,created_at=now(),seconds=time.monotonic()-start))
            frozen.append((number,tag,out))
            del model,dec; gc.collect()
            if args.device=='cuda': torch.cuda.empty_cache()
    labels=read(ROOT/'evaluation_data/darkbay/scoring_labels/scoring_labels.json')
    summaries={}
    for number,tag,out in frozen:
        rows=read(out/'predictions.json')
        assert {r['opaque_id'] for r in rows}==set(labels)
        correct=sum(r['prediction']==labels[r['opaque_id']] for r in rows)
        result=dict(model=number,dataset=tag,epoch=args.epoch,seed=args.seed,
                    correct=correct,total=900,accuracy_percent=100*correct/900,
                    device=args.device,diagnostic_only=True)
        write(out/'summary.json',result)
        write(out/'status.json',dict(state='complete',processed=900,updated_at=now()))
        summaries[f'Model{number}-{tag}']=result
    if args.model_only:
        result=dict(created_at=now(),tested_epoch=args.epoch,seed=args.seed,results=summaries,diagnostic_only=True,training_changed=False)
        write(STUDY/f'interim_model{args.model}_epoch{args.epoch:02d}.json',result)
        print(result,flush=True)
        return
    comparisons={tag:{f'model{args.model}':summaries[f'Model{args.model}-{tag}'],
        'model1':summaries[f'Model1-{tag}'],
        'difference_correct':summaries[f'Model{args.model}-{tag}']['correct']-summaries[f'Model1-{tag}']['correct'],
        'difference_percentage_points':(summaries[f'Model{args.model}-{tag}']['correct']-summaries[f'Model1-{tag}']['correct'])/9}
        for tag in ('A','B')}
    result=dict(created_at=now(),tested_epoch=args.epoch,seed=args.seed,comparisons=comparisons,
                training_changed=False,checkpoint_selection_unchanged='fixed final20')
    filename = f'interim_comparison_epoch{args.epoch:02d}.json' if args.model==2 else f'interim_model{args.model}_comparison_epoch{args.epoch:02d}.json'
    write(STUDY/filename,result)
    print(result,flush=True)


if __name__=='__main__':main()
