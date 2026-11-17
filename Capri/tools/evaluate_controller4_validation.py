"""Evaluate frozen SynthTIGER validation on saved PARSeq checkpoints, no optimization."""
import argparse, copy, gc, hashlib, io, json, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from capri_controller.training import ab_curriculum as core
ROOT=core.ROOT
DATA=ROOT/'datasets/darkbay/validation/controller4_synthtiger_1000'
OUT=core.STUDY/'controller4_validation_sweep'
FIG=ROOT/'thesis/figures/darkbay/controller4_validation'

def discover():
    jobs=[];missing=[]
    for n in (1,2,3):
        for tag in ('A','B'):
            limit=20 if n<3 else core.read(ROOT/f'model3/runs/darkbay/{tag}/status.json')['completed_epoch']
            for epoch in range(1,21):
                name=f'epoch_{epoch}.pth' if n==1 else f'epoch_{epoch:02d}.pth'
                p=ROOT/f'model{n}/runs/darkbay/{tag}/checkpoints'/name
                if n==1 and epoch==20 and not p.exists():p=ROOT/f'model1/models/parseq/darkbay/{tag}/epoch_20.pth'
                if epoch>limit or not p.exists():
                    missing.append(dict(model=n,dataset=tag,epoch=epoch,reason='not committed yet' if epoch>limit else 'checkpoint not retained'))
                else:jobs.append(dict(model=n,dataset=tag,epoch=epoch,checkpoint=p.relative_to(ROOT).as_posix(),checkpoint_sha256=core.sha(p)))
    return jobs,missing

def destination(job):
    return ROOT/f"model{job['model']}/results/darkbay/{job['dataset']}/controller4_validation/epoch_{job['epoch']:02d}"

def collect(protocol):
    results=[]
    for job in protocol['jobs']:
        out=destination(job)
        if not (out/'summary.json').exists():continue
        s=core.read(out/'summary.json');rows=core.read(out/'predictions.json')
        assert s['checkpoint_sha256']==job['checkpoint_sha256']
        assert s['manifest_sha256']==protocol['manifest_sha256'] and core.sha(out/'predictions.json')==s['predictions_sha256']
        assert len(rows)==1000 and {r['index'] for r in rows}==set(range(1,1001))
        results.append(s)
    best=[]
    for n in (1,2,3):
        for tag in ('A','B'):
            subset=[r for r in results if r['model']==n and r['dataset']==tag]
            if subset:
                b=max(subset,key=lambda r:(r['correct'],-r['epoch']))
                best.append(dict(model=n,dataset=tag,epoch=b['epoch'],correct=b['correct'],accuracy_percent=b['accuracy_percent'],evaluated_epochs=[r['epoch'] for r in subset],tied_epochs=[r['epoch'] for r in subset if r['correct']==b['correct']]))
    result=dict(updated_at=core.now(),total=1000,results=results,best_validation=best,missing=protocol['missing'],planned=len(protocol['jobs']),evaluated=len(results),metric='Case-sensitive whole-string exact match',training_changed=False)
    core.write(OUT/'results.json',result)
    lines=['# Controller4 synthetic validation accuracy','', 'All values are percentages on the same1000 held-out images. A/B are separate readers. Missing checkpoints are not estimated.','', '| Epoch | M1-A | M1-B | M2-A | M2-B | M3-A | M3-B |','|---|---|---|---|---|---|---|']
    lookup={(r['model'],r['dataset'],r['epoch']):r for r in results}
    for e in range(1,21):
        cells=[]
        for n in (1,2,3):
            for t in ('A','B'):
                r=lookup.get((n,t,e));cells.append(f"{r['accuracy_percent']:.1f}% ({r['correct']}/1000)" if r else '--')
        lines.append('| '+str(e)+' | '+' | '.join(cells)+' |')
    (OUT/'epoch_accuracy.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

def plots(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    for tag in ('A','B'):
        fig,ax=plt.subplots(figsize=(10.5,6))
        for n,color in [(1,'#159447'),(2,'#2463EB'),(3,'#DC3545')]:
            rows=sorted([r for r in result['results'] if r['model']==n and r['dataset']==tag],key=lambda r:r['epoch'])
            ax.plot([r['epoch'] for r in rows],[r['accuracy_percent'] for r in rows],marker='o',markersize=4,color=color,label=f'Model {n}',linewidth=2)
        ax.set(title=f'DarkBay - reader dataset {tag}\nSynthTIGER held-out validation: 1,000 images',xlabel='Epoch',ylabel='Exact-match accuracy (%)',xlim=(.7,20.3),ylim=(0,100))
        ax.set_xticks(range(1,21));ax.grid(alpha=.2);ax.legend();fig.tight_layout(rect=(0,.06,1,1))
        fig.text(.07,.025,'Raw measurements. Model 1 epoch 1 unavailable; uncommitted Model 3 epochs omitted.',fontsize=9)
        for ext in ('png','svg'):fig.savefig(FIG/f'dataset_{tag.lower()}.{ext}',dpi=170)
        plt.close(fig)
    core.write(FIG/'plotted_data.json',result)

def compare_real(result):
    # Read existing diagnostic summaries only, after the validation sweep. No real labels or new real inference.
    from scipy.stats import spearmanr
    comparisons=[]
    for n in (1,2,3):
        for tag in ('A','B'):
            paired=[]
            for r in result['results']:
                if (r['model'],r['dataset'])!=(n,tag):continue
                p=ROOT/f"model{n}/results/darkbay/{tag}/interim_cuda_epoch{r['epoch']:02d}_seed48/summary.json"
                if p.exists():
                    real=core.read(p);assert real['total']==900
                    paired.append(dict(epoch=r['epoch'],validation_correct=r['correct'],real_correct=real['correct'],real_summary=p.relative_to(ROOT).as_posix(),real_summary_sha256=core.sha(p)))
            stats={}
            for label,lo in [('all_available',1),('epochs6_onward',6)]:
                subset=[p for p in paired if p['epoch']>=lo]
                if len(subset)>=3:
                    rho=float(spearmanr([p['validation_correct'] for p in subset],[p['real_correct'] for p in subset]).statistic)
                    stats[label]=dict(epochs=len(subset),spearman_rho=rho if rho==rho else None)
            comparisons.append(dict(model=n,dataset=tag,pairs=paired,descriptive_correlations=stats))
    core.write(OUT/'real_comparison.json',dict(comparisons=comparisons,note='Retrospective association using previously inspected real diagnostics, not proof of generalization. Missing real diagnostics omitted; no real labels opened or checkpoints selected by real scores.'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (OUT/'sweep.lock').open('a+b') as lock:
        lock.seek(0,2)
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        core.bootstrap()
        import torch, numpy as np, lmdb
        from PIL import Image
        from openrec.modeling import build_model
        from openrec.postprocess import build_post_process
        from capri_controller.training.ds2_data import transform
        torch.set_num_threads(2)
        protocol_path=OUT/'protocol.json'
        if protocol_path.exists():
            protocol=core.read(protocol_path)
            assert protocol['evaluator_sha256']==core.sha(Path(__file__)),'Preserve frozen evaluator'
        else:
            jobs,missing=discover()
            protocol=dict(created_at=core.now(),jobs=jobs,missing=missing,manifest_sha256=core.sha(DATA/'manifest.jsonl'),lmdb_sha256=core.sha(DATA/'validation_lmdb/data.mdb'),evaluator_sha256=core.sha(Path(__file__)),decoder_source_sha256=core.sha(ROOT/'model1/src/OpenOCR/openrec/modeling/decoders/parseq_decoder.py'),images=1000,metric='Case-sensitive whole-string exact match',decoding='Saved free AR decoder, one refinement; no target-length constraint or label-conditioned retry',batch_size=64,precision='FP32',selection='Descriptive best validation epoch; ties choose earliest. Existing fixed final20 primary result unchanged.',scope='All available fully committed checkpoints captured before inference',gpu_scheduling='Existing per-batch training lock',real_labels_used=False)
            core.write(protocol_path,protocol)
            (OUT/'frozen_evaluator.source.py').write_bytes(Path(__file__).read_bytes())
        if args.report_only:
            result=collect(protocol);plots(result);compare_real(result);return
        assert core.sha(DATA/'manifest.jsonl')==protocol['manifest_sha256']
        assert core.sha(DATA/'validation_lmdb/data.mdb')==protocol['lmdb_sha256']
        manifest=[json.loads(s) for s in (DATA/'manifest.jsonl').read_text().splitlines()]
        assert len(manifest)==1000
        env=lmdb.open(str(DATA/'validation_lmdb'),readonly=True,lock=False,readahead=False)
        pixels=[];labels={};tr=transform()
        with env.begin() as tx:
            assert int(tx.get(b'num-samples'))==1000
            for r in manifest:
                idx=r['index'];im=Image.open(io.BytesIO(tx.get(f'image-{idx:09d}'.encode()))).convert('RGB')
                assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
                labels[idx]=tx.get(f'label-{idx:09d}'.encode()).decode();assert labels[idx]==r['text']
                pixels.append(tr(im))
        env.close();xall=torch.stack(pixels);del pixels
        completed=0
        for job in protocol['jobs']:
            out=destination(job);n=job['model'];tag=job['dataset'];epoch=job['epoch'];p=ROOT/job['checkpoint']
            if (out/'summary.json').exists():
                completed+=1;continue
            assert core.sha(p)==job['checkpoint_sha256']
            started=time.monotonic();print(f'START Model{n}-{tag} epoch{epoch}',flush=True)
            core.write(OUT/'status.json',dict(state='running',completed=completed,total=len(protocol['jobs']),current=job,updated_at=core.now(),pid=os.getpid()))
            ck=torch.load(p,map_location='cpu',weights_only=False)
            assert ck['epoch']==epoch and ck['config']['Global']['seed']==48
            cfg=copy.deepcopy(ck['config']);charset=str(ROOT/f'datasets/darkbay/{tag}/charset.txt')
            cfg['Global']['character_dict_path']=charset;cfg['PostProcess']['character_dict_path']=charset
            dec=build_post_process(cfg['PostProcess'],cfg['Global']);cfg['Architecture']['Decoder']['out_channels']=dec.get_character_num()
            model=build_model(cfg['Architecture']);model.load_state_dict(ck['state_dict'],strict=True);del ck
            assert model.decoder.refine_iters==1
            with core.gpu_slot():model.cuda().eval()
            rows=[]
            with torch.inference_mode():
                for start in range(0,1000,64):
                    with core.gpu_slot():decoded=dec(model(xall[start:start+64].cuda()))
                    rows.extend(dict(index=start+k+1,prediction=v[0]) for k,v in enumerate(decoded))
            assert len(rows)==1000
            core.write(out/'predictions.json',rows)
            core.write(out/'prediction_freeze.json',dict(checkpoint_sha256=job['checkpoint_sha256'],predictions_sha256=core.sha(out/'predictions.json'),manifest_sha256=protocol['manifest_sha256'],created_at=core.now()))
            correct=sum(r['prediction']==labels[r['index']] for r in rows)
            summary=dict(**job,correct=correct,total=1000,accuracy_percent=correct/10,manifest_sha256=protocol['manifest_sha256'],predictions_sha256=core.sha(out/'predictions.json'),seconds=time.monotonic()-started,created_at=core.now())
            core.write(out/'summary.json',summary)
            del model,dec;gc.collect();torch.cuda.empty_cache();completed+=1
            print(f'RESULT Model{n}-{tag} epoch{epoch}: {correct}/1000 ({correct/10:.1f}%) [{completed}/{len(protocol["jobs"])}] {summary["seconds"]:.1f}s',flush=True)
            collect(protocol)
        result=collect(protocol);assert result['evaluated']==result['planned']
        for job in protocol['jobs']:
            out=destination(job);rows=core.read(out/'predictions.json');s=core.read(out/'summary.json')
            assert sum(r['prediction']==labels[r['index']] for r in rows)==s['correct']
        plots(result);compare_real(result)
        core.write(OUT/'verification.json',dict(passed=True,evaluations=len(result['results']),images_per_evaluation=1000,all_prediction_ids_hashes_and_counts_verified=True,manifest_sha256=protocol['manifest_sha256'],created_at=core.now()))
        core.write(OUT/'status.json',dict(state='complete',completed=completed,total=completed,updated_at=core.now()))
        print('SWEEP COMPLETE',flush=True)
if __name__=='__main__':main()
