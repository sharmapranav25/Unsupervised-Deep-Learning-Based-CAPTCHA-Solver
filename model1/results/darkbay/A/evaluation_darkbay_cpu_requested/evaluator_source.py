"""Requested final-reader diagnostic without allocating memory on the training GPU."""
import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from ..decoder.common import ROOT, read, write, sha, now
from . import capri2_parseq as F


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    run=Path(args.run).resolve();F.configure(run);F.verify(run)
    out=run/'evaluation_darkbay_cpu_requested'
    if out.exists():raise FileExistsError('Preserve existing evaluation: '+str(out))
    out.mkdir();shutil.copyfile(__file__,out/'evaluator_source.py')
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from rapidfuzz.distance import Levenshtein
    torch.set_num_threads(4)
    completion=read(run/'completion.json');assert completion['all_complete'] and len(completion['seeds'])==1
    record=completion['seeds'][0];checkpoint=Path(record['checkpoint'])
    assert record['epoch']==20 and sha(checkpoint)==record['sha256']
    rows=read(F.BASE/'evaluation_darkbay/image_manifest.json');assert len(rows)==900
    rows=[dict(r,image_path=str(F.map_old(r['image_path']))) for r in rows]
    pixels=[]
    for r in rows:
        im=Image.open(r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
        pixels.append(im)
    assert len({r['pixel_sha256'] for r in rows})==900
    assert not {r['pixel_sha256'] for r in rows}&{r['pixel_sha256'] for r in read(run/'targets.json')['fit']}
    write(out/'image_manifest.json',rows)
    write(out/'protocol.json',dict(created_at=now(),device='cpu',precision='FP32',cpu_threads=4,
        checkpoint=record,images=900,excluded_controller_references=100,
        preprocessing='RGB,bicubic32x128,ToTensor,Normalize0.5',
        decoding='Saved PARSeq autoregressive decoder, one refinement, no forced length or narrowed charset',
        label_access='Only after all final predictions are frozen',
        purpose='User-requested Model1-A final-checkpoint diagnostic during Model1-B training',
        model_selection='Fixed final epoch20; diagnostic does not alter ongoing training',
        comparison_limit='Historical CAPRI1 inference used GPU FP32; CPU numerical differences are possible',
        historical_real_test=True,evaluator_sha256=sha(Path(__file__))))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=copy.deepcopy(saved['config'])
    decoder=build_post_process(cfg['PostProcess'],cfg['Global'])
    cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
    model=build_model(cfg['Architecture']);model.load_state_dict(saved['state_dict'],strict=True)
    model.cpu().eval();del saved;gc.collect()
    transform=T.Compose([T.Resize((32,128),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
    values=[];started=time.time()
    with torch.inference_mode():
        for i in range(0,len(rows),16):
            chunk=rows[i:i+16];x=torch.stack([transform(im) for im in pixels[i:i+16]])
            decoded=decoder(model(x));assert len(decoded)==len(chunk)
            values.extend(dict(opaque_id=r['opaque_id'],prediction=text,
                confidence=float(conf) if np.isfinite(conf) else None) for r,(text,conf) in zip(chunk,decoded))
            write(out/'status.json',dict(stage='inference',completed=len(values),total=900,seconds=time.time()-started))
            if len(values)%160==0:print('Predicted',len(values),'/900',flush=True)
    assert len(values)==900
    write(out/'predictions.json',values)
    write(out/'prediction_freeze.json',dict(file='predictions.json',sha256=sha(out/'predictions.json'),count=900,frozen_at=now()))
    # Labels are first opened after the predictions above are persisted and hash-frozen.
    labels=read(F.BASE/'evaluation_darkbay/scoring_labels.json')
    predictions={r['opaque_id']:r['prediction'] for r in values};assert set(predictions)==set(labels)
    correct=sum(predictions[k]==v for k,v in labels.items())
    edits=sum(Levenshtein.distance(v,predictions[k]) for k,v in labels.items())
    baseline=next(r for r in read(F.BASE/'evaluation_darkbay/summary.json')['results'] if r['seed']==record['seed'])
    summary=dict(correct=correct,total=900,exact_match_percent=100*correct/900,
        character_error_rate_percent=100*edits/sum(map(len,labels.values())),checkpoint_epoch=20,
        checkpoint_sha256=record['sha256'],device='cpu',seconds=time.time()-started,
        previous_capri_reader=baseline,change_percentage_points=100*correct/900-baseline['exact_match_percent'],
        checkpoint_selection='Fixed final epoch20',historical_real_test=True)
    assert sha(out/'predictions.json')==read(out/'prediction_freeze.json')['sha256']
    assert sha(checkpoint)==record['sha256']
    write(out/'summary.json',summary)
    write(out/'per_image_results.json',[dict(**r,label=labels[r['opaque_id']],
        correct=r['prediction']==labels[r['opaque_id']]) for r in values])
    write(out/'status.json',dict(stage='complete',completed=900,total=900,completed_at=now()))
    (out/'RESULTS.md').write_text('# Model 1 A on labeled DarkBay\n\n'+json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
