"""Isolated user-requested CPU diagnostics; never changes training or selection."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path('E:/pranav/Thesis-Pranav-Final')
sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.training import cryptbb_feature_b_v1 as M
from capri_controller.decoder.common import read,write,sha,now,StageLock

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['evaluate']);a=p.parse_args()
    out=ROOT/'model3/runs'/M.FAMILY/'diagnostics/real_cryptbb';out.mkdir(parents=True,exist_ok=True)
    with StageLock(out,'coordinator'):
        digest=M.check_sources();checkpoints=[]
        for path in sorted((M.OUT/'checkpoints').glob('epoch_*.json')):
            r=read(path);assert r['protocol_sha256']==digest
            assert sha(M.OUT/r['checkpoint'])==r['sha256']
            checkpoints.append(dict(dataset='B',epoch=r['epoch'],sha256=r['sha256']))
        assert checkpoints
        manifest=out/f'request_through_epoch_{checkpoints[-1]["epoch"]:02d}.json'
        if manifest.exists():assert read(manifest)['checkpoints']==checkpoints
        else:write(manifest,dict(requested_at=now(),purpose='User requested all available checkpoints; diagnostic only',training_protocol_sha256=digest,checkpoints=checkpoints))
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'Capri/src');env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_VISIBLE_DEVICES']=''
        for r in checkpoints:
            subprocess.run([sys.executable,'-B','-u','-m','capri_controller.training.cryptbb_feature_b_diagnostic','--dataset','B','--epoch',str(r['epoch']),'--request-manifest',str(manifest)],env=env,cwd=ROOT,check=True)
        summaries=[]
        for r in checkpoints:
            d=out/f'epoch_{r["epoch"]:02d}';s=read(d/'summary.json');v=read(d/'verification.json');f=read(d/'prediction_freeze.json');rows=read(d/'per_image_results.json')
            assert v['passed'] and f['checkpoint_sha256']==r['sha256']==s['checkpoint_sha256']
            assert sha(d/'predictions.json')==f['sha256']
            predictions=read(d/'predictions.json');assert {x['opaque_id']:x['prediction'] for x in predictions}=={x['opaque_id']:x['prediction'] for x in rows}
            assert len(rows)==len({x['opaque_id'] for x in rows})==251
            assert sum(x['prediction']==x['label'] for x in rows)==s['correct']
            baseline=ROOT/'model2/results/cryptbb_v3_epoch20_static_B_20261115/B'/f'epoch_{r["epoch"]:02d}/summary.json'
            if baseline.exists():
                b=read(baseline);s['model2_same_epoch_correct']=b['correct'];s['model2_same_epoch_accuracy_percent']=b['accuracy_percent'];s['difference_pp']=s['accuracy_percent']-b['accuracy_percent']
            summaries.append(s)
        assert M.check_sources()==digest
        write(out/'all_checkpoints_summary.json',dict(measured_at=now(),snapshot_manifest=manifest.name,results=summaries,training_status=read(M.OUT/'status.json'),diagnostic_only=True))
        print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()