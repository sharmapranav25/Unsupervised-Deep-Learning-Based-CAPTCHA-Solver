"""Requested full synthetic-training accuracy; immutable checkpoint, shared GPU lock."""
import argparse,copy,gc,hashlib,io,json,os,shutil,time,traceback
from pathlib import Path
from ..decoder.common import read,write,sha,now,StageLock
from .cryptbb_v3_model1 import gpu_slot,verify_sources
ROOT=Path(__file__).resolve().parents[4]
STUDY=ROOT/'handoff/cryptbb_v3_training_20261114'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epoch',required=True,type=int);a=ap.parse_args()
    import lmdb,numpy as np,torch
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from rapidfuzz.distance import Levenshtein
    torch.set_num_threads(2)
    run=ROOT/'model1/runs/cryptbb_v3/B';data=ROOT/'datasets/cryptbb_v3/B'
    cp=run/f'epoch_{a.epoch}.pth';receipt=read(run/f'epoch_{a.epoch}_receipt.json')
    digest=sha(cp);assert digest==receipt['sha256'] and receipt['global_step']==782*a.epoch
    out=ROOT/f'model1/results/cryptbb_v3/B/training_full_gpu_epoch{a.epoch:02d}'
    assert not out.exists(),'Preserve existing evaluation output'
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'evaluator_source.py')
    with StageLock(out,'evaluation'):
      try:
        write(out/'status.json',dict(state='verifying_inputs',updated_at=now(),pid=os.getpid()))
        verify_sources();dr=read(data/'data_receipt.json')
        assert sha(data/'train_lmdb/data.mdb')==dr['lmdb_sha256']
        assert sha(data/'sample_provenance.jsonl')==dr['provenance_sha256']
        assert sha(ROOT/'datasets/cryptbb_v3/charset.txt')==dr['charset_sha256']
        rows=[json.loads(line) for line in (data/'sample_provenance.jsonl').read_text().splitlines()]
        assert len(rows)==100000 and [r['index'] for r in rows]==list(range(1,100001))
        assert len({r['pixel_sha256'] for r in rows})==100000
        ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['epoch']==a.epoch
        cfg=copy.deepcopy(ck['config']);dec=build_post_process(cfg['PostProcess'],cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels']=dec.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(ck['state_dict'],strict=True);model.eval();del ck;gc.collect()
        tr=T.Compose([T.Resize((32,128),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
        protocol=dict(created_at=now(),epoch=a.epoch,dataset='B',images=100000,checkpoint_sha256=digest,data_receipt=dr,device='cuda',precision='FP32',batch_size=64,threads=2,decoding='Saved free AR and one refinement; no teacher forcing or length forcing',preprocessing='RGB PIL bicubic32x128 ToTensor Normalize0.5',gpu_execution='Existing training gpu_slot per batch; no training-process mutation',evaluator_sha256=sha(Path(__file__)),diagnostic_only=True,training_changed=False)
        write(out/'protocol.json',protocol)
        with gpu_slot():
            free,total=torch.cuda.mem_get_info();assert free>2*1024**3,('Insufficient free GPU memory',free)
            model.cuda();torch.cuda.reset_peak_memory_stats()
        env=lmdb.open(str(data/'train_lmdb'),readonly=True,lock=False,readahead=False,max_readers=64)
        correct=edits=chars=processed=0;groups={k:dict(total=0,correct=0,edits=0,characters=0) for k in dr['controllers']};start=time.monotonic()
        with env.begin() as tx, (out/'predictions.jsonl').open('x',encoding='utf8') as f, torch.inference_mode():
            assert int(tx.get(b'num-samples'))==100000
            for offset in range(0,100000,64):
                if (out/'STOP').exists() or (STUDY/'STOP').exists():raise RuntimeError('Stop marker present; evaluation stops')
                batch=rows[offset:offset+64];images=[];labels=[]
                for r in batch:
                    raw=tx.get(f"image-{r['index']:09d}".encode());label=tx.get(f"label-{r['index']:09d}".encode()).decode()
                    assert label==r['text'];im=Image.open(io.BytesIO(raw)).convert('RGB')
                    assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
                    images.append(tr(im));labels.append(label)
                x=torch.stack(images)
                with gpu_slot():
                    x=x.cuda();pred=model(x);decoded=dec(pred.cpu());del pred,x
                assert len(decoded)==len(batch)
                for r,label,pred in zip(batch,labels,decoded):
                    prediction=pred[0];ok=prediction==label;e=Levenshtein.distance(label,prediction)
                    correct+=ok;edits+=e;chars+=len(label);processed+=1
                    g=groups[r['controller']];g['total']+=1;g['correct']+=ok;g['edits']+=e;g['characters']+=len(label)
                    f.write(json.dumps(dict(index=r['index'],pixel_sha256=r['pixel_sha256'],controller=r['controller'],label=label,prediction=prediction,correct=ok,edits=e))+'\n')
                f.flush()
                if processed%512==0 or processed==100000:
                    elapsed=time.monotonic()-start
                    status=dict(state='evaluating',updated_at=now(),pid=os.getpid(),processed=processed,total=100000,seconds=elapsed,images_per_second=processed/elapsed,estimated_seconds_remaining=(100000-processed)*elapsed/processed)
                    write(out/'status.json',status);print(status,flush=True)
        env.close();assert processed==100000 and all(g['total']==25000 for g in groups.values())
        verify_sources();assert sha(cp)==digest
        write(out/'prediction_freeze.json',dict(created_at=now(),sha256=sha(out/'predictions.jsonl'),checkpoint_sha256=digest,images=100000))
        recount=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
        assert len(recount)==100000 and len({r['index'] for r in recount})==100000
        assert sum(r['correct'] for r in recount)==correct and sum(r['edits'] for r in recount)==edits
        for g in groups.values():g.update(accuracy_percent=100*g['correct']/g['total'],character_accuracy_percent=100*(1-g['edits']/g['characters']))
        summary=dict(epoch=a.epoch,dataset='B',total=processed,correct=correct,accuracy_percent=100*correct/processed,cer_percent=100*edits/chars,character_accuracy_percent=100*(1-edits/chars),character_errors=edits,reference_characters=chars,controllers=groups,seconds=time.monotonic()-start,checkpoint_sha256=digest,device='cuda',precision='FP32',diagnostic_only=True,training_changed=False)
        write(out/'verification.json',dict(passed=True,all100000unique_images=True,pixel_and_synthetic_label_identities_verified=True,checkpoint_hash_verified=True,training_sources_unchanged=True,counts_recomputed=True))
        write(out/'summary.json',summary);write(out/'status.json',dict(state='complete',updated_at=now(),processed=100000,total=100000));print(summary,flush=True)
      except BaseException as exc:
        write(out/'failure.json',dict(at=now(),error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
