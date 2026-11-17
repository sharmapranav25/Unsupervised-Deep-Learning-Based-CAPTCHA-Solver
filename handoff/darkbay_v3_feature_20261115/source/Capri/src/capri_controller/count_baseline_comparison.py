"""Compare the deployed original CAPRI with a frozen new-model count diagnostic."""
import argparse, os, json
from pathlib import Path
import torch
from .decoder.common import ROOT, read, write, sha, now
from .lightweight.legacy_model import Iter2Model
from .lightweight_count_evaluation import batch_input, statistics

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    reference=Path(args.reference).resolve();out=Path(args.out).resolve()
    if out.drive.upper()!='E:' or out.exists():raise ValueError('Fresh E-drive output required')
    out.mkdir(parents=True);(out/'temporary').mkdir()
    for key in ('TEMP','TMP','TORCH_HOME','HF_HOME','XDG_CACHE_HOME'):os.environ[key]=str(out/'temporary')
    torch.set_num_threads(4)
    def progress(state,**extra):
        value=dict(time=now(),state=state,**extra);write(out/'status.json',value);print(value,flush=True)
    try:
        registry=read(ROOT/'models/registry.json');baseline=registry['models']['hybrid'];cp=ROOT/baseline['path']
        assert sha(cp)==baseline['sha256']
        assert read(reference/'status.json')['state']=='complete'
        manifests=read(reference/'image_manifests.json')
        assert sum(map(len,manifests.values()))==1200
        write(out/'image_manifests.json',manifests)
        sources=[Path(__file__),ROOT/'capri.py',ROOT/'src/capri_controller/lightweight_count_evaluation.py',ROOT/'src/capri_controller/lightweight/legacy_model.py',ROOT/'src/capri_controller/lightweight/primitives.py']
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
        protocol=dict(created_at=now(),baseline_checkpoint=str(cp),baseline_sha256=baseline['sha256'],baseline_selection='Currently deployed original CAPRI, epoch15 EMA, fixed before scoring',reference=str(reference),reference_manifest_sha256=sha(reference/'image_manifests.json'),preprocessing='Identical to new-model diagnostic: RGB, height96, width cap480, bilinear, right edge padding, batch8, CPU FP32. This controls preprocessing; historical deployed inference used Lanczos and cap960.',label_policy='Read saved scoring counts only after baseline predictions are persisted and SHA256-frozen. No selection or tuning.',source_hashes=hashes)
        write(out/'protocol.json',protocol)
        for p in sources:
            dst=out/'provenance'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        model=Iter2Model().eval().requires_grad_(False)
        model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True)
        predictions={}
        for name,rows in manifests.items():
            records=[]
            for start in range(0,len(rows),8):
                batch=rows[start:start+8];image,cols,_=batch_input(batch);logits=model(image,cols)['length']
                assert torch.isfinite(logits).all()
                records.extend(dict(opaque_id=r['opaque_id'],pixel_sha256=r['pixel_sha256'],capri_count=int(logits[i].argmax())+2,capri_logits=logits[i].tolist()) for i,r in enumerate(batch))
                if (start+len(batch))%100<8:progress('predicting',scheme=name,images=len(records),total=len(rows))
            predictions[name]=records
        write(out/'predictions.json',predictions)
        digest=sha(out/'predictions.json');write(out/'prediction_freeze.json',dict(created_at=now(),sha256=digest,images=1200))
        truth=read(reference/'scoring_counts.json');new=read(reference/'predictions.json');new_summary=read(reference/'summary.json')
        assert sha(reference/'predictions.json')==new_summary['prediction_sha256']==read(reference/'prediction_freeze.json')['sha256']
        comparison={}
        for name in ('DarkBay','CryptBB','City','SimpleCaptcha'):
            old_scores=statistics(predictions[name],truth[name],'capri_count');new_scores=statistics(new[name],truth[name],'capri_count')
            assert json.loads(json.dumps(new_scores))==new_summary['schemes'][name]['capri']
            assert [(r['opaque_id'],r['pixel_sha256']) for r in predictions[name]]==[(r['opaque_id'],r['pixel_sha256']) for r in new[name]]
            comparison[name]=dict(previous_capri=old_scores,new_capri=new_scores,change_percentage_points=100*(new_scores['accuracy']-old_scores['accuracy']))
        assert sha(out/'predictions.json')==digest and sha(cp)==baseline['sha256']
        assert all(sha(ROOT/p)==h for p,h in hashes.items())
        write(out/'scoring_counts.json',truth)
        summary=dict(completed_at=now(),schemes=comparison,baseline_checkpoint=str(cp),baseline_sha256=baseline['sha256'],new_checkpoint=new_summary['checkpoint'],new_sha256=new_summary['checkpoint_sha256'],baseline_prediction_sha256=digest,new_prediction_sha256=new_summary['prediction_sha256'],same_images_labels_preprocessing=True,limitations='Historical mostly fixed-length target sets. City49unscored; SimpleCaptcha ground truth from saved publisher count metadata. This is count accuracy, not text recognition. No checkpoint selection or deployment changes.')
        write(out/'summary.json',summary)
        lines=['# Previous CAPRI versus new CAPRI: internal count','', 'Original deployed epoch15 EMA versus new completed epoch20 EMA. Both use the same 1,200 images, labels, CPU FP32 and new-model preprocessing. The old model was rerun; its historical results used different samples and Lanczos/cap960 preprocessing.','', '| Scheme | Previous CAPRI | New CAPRI | Change (percentage points) |','|---|---:|---:|---:|']
        for name,v in comparison.items():
            a=v['previous_capri'];b=v['new_capri'];lines.append(f'| {name} | {a["correct"]}/{a["n"]} ({a["accuracy"]:.1%}) | {b["correct"]}/{b["n"]} ({b["accuracy"]:.1%}) | {v["change_percentage_points"]:+.1f} |')
        lines+=['',summary['limitations'],'','Baseline predictions were frozen before saved scoring labels were read. Exact image IDs and pixel hashes match the new evaluation. Checkpoint and prediction hashes verified. Reproduction uses capri.py compare-capri-count --reference <reference run> --out <fresh E-drive path>.']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        progress('complete',schemes=comparison)
    except Exception as e:
        progress('failed',error=repr(e));raise
