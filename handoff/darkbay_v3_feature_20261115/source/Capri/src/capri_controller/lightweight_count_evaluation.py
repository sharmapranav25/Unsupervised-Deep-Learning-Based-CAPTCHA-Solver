"""Requested count-only target diagnostic; predictions frozen before real-label scoring."""
import argparse,collections,csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .decoder.common import ROOT,read,write,sha,now
from .lightweight.model import LightweightCapri

HIST=ROOT/'runs/capri_decoder_500k_20261007/diagnostics'

def mapped(path):
    s=str(path).replace('\\','/')
    old='C:/Users/pranav/Documents/ChatGPT/thesis/'
    p=ROOT/s[len(old):] if s.lower().startswith(old.lower()) else Path(s)
    p=p.resolve()
    if p.drive.upper()!='E:':raise ValueError('Target data must resolve on E: '+str(p))
    if not p.is_file():raise FileNotFoundError(p)
    return p

def image_input(row):
    p=mapped(row['image_path']);rgb=np.asarray(Image.open(p).convert('RGB'))
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['pixel_sha256']
    h,w=rgb.shape[:2];width=max(8,min(480,round(w*96/h)))
    rgb=np.asarray(Image.fromarray(rgb).resize((width,96),Image.Resampling.BILINEAR))
    return rgb,width

def batch_input(rows):
    images=[image_input(r) for r in rows];widths=np.array([w for _,w in images]);cols=(widths+7)//8;wp=int((max(cols.max()*8,widths.max())+32+7)//8*8)
    x=np.stack([np.pad(im,((0,0),(0,wp-w),(0,0)),mode='edge').transpose(2,0,1) for im,w in images])
    return torch.from_numpy(x.copy()).float()/255,torch.from_numpy(cols.copy()).long(),torch.from_numpy(widths.copy()).long()

def counts_only(model,image,cols,widths):
    # Exact training forward for both count branches; DINO modifies only font heads.
    p=model.capri(image,cols)['length']
    x=F.interpolate(image[:,:,:96,:480],size=(64,320),mode='bilinear',align_corners=False,antialias=True) if image.shape[-1]>=480 else F.interpolate(F.pad(image,(0,480-image.shape[-1],0,0),mode='replicate'),size=(64,320),mode='bilinear',align_corners=False,antialias=True)
    x=F.pad(x,(0,64),value=1.);cw=(widths.float()*2/3).round().long()
    mask=(torch.arange(384)[None,:]<cw[:,None])[:,None,:].expand(-1,64,-1)
    return p,model.counter(x,mask)

def statistics(rows,truth,field):
    matched=[(r,truth[r['opaque_id']]) for r in rows if r['opaque_id'] in truth]
    n=len(matched);correct=sum(r[field]==t for r,t in matched)
    return dict(correct=correct,n=n,accuracy=correct/n if n else None,mean_absolute_error=sum(abs(r[field]-t) for r,t in matched)/n if n else None,predicted_count_distribution=dict(sorted(collections.Counter(r[field] for r in rows).items())),unscored=len(rows)-n)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    run=Path(a.run).resolve();out=Path(a.out).resolve()
    if out.drive.upper()!='E:' or out.exists():raise ValueError('Fresh E-drive output required')
    out.mkdir(parents=True);temp=out/'temporary';temp.mkdir()
    for key in ('TEMP','TMP','TORCH_HOME','HF_HOME','XDG_CACHE_HOME'):os.environ[key]=str(temp)
    os.environ['PYTHONDONTWRITEBYTECODE']='1';torch.set_num_threads(4)
    def progress(state,**kw):
        d=dict(time=now(),state=state,**kw);write(out/'status.json',d);print(json.dumps(d),flush=True)
    try:
        pointer=read(run/'resume.json');checkpoint=Path(pointer['checkpoint']);receipt=read(checkpoint.parent/'receipt.json')
        digest=sha(checkpoint);assert digest==receipt['sha256']
        training_protocol=read(run/'protocol.json');assert all(sha(ROOT/p)==h for p,h in training_protocol['sources'].items())
        sets=read(HIST/'real_three_schemes_100_epoch8_20261008_v2/image_manifests.json')
        sets['DarkBay']=read(HIST/'real_darkbay_epoch8_20261008/image_manifest.json')
        assert {k:len(v) for k,v in sets.items()}=={'CryptBB':100,'City':100,'SimpleCaptcha':100,'DarkBay':900}
        opaque={name:[dict(opaque_id=r['opaque_id'],image_path=str(mapped(r['image_path'])),pixel_sha256=r['pixel_sha256']) for r in rows] for name,rows in sets.items()}
        write(out/'image_manifests.json',opaque)
        write(out/'protocol.json',dict(created_at=now(),checkpoint=str(checkpoint),checkpoint_sha256=digest,checkpoint_pointer=pointer,weights='EMA',selection='Latest committed resume pointer captured once at evaluation start; not quality-selected',counts={k:len(v) for k,v in sets.items()},device='CPU float32,4threads; GPU training untouched',inference='Both unchanged count subgraphs, pixel-only; no DINO/font inference required for counts; full-forward parity checked',preprocessing='Same RGB96high,widthcap480,bilinear,edgepad and count preprocessing as training',label_policy='All predictions persisted and hash-frozen before this program reads scoring labels. No threshold tuning, checkpoint selection or curriculum changes.',source_sha256=sha(__file__),entry_sha256=sha(ROOT/'capri.py'),training_sources=training_protocol['sources']))
        progress('loading_checkpoint',checkpoint=str(checkpoint),step=pointer['step'],epoch=pointer['epoch'])
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);model=LightweightCapri();model.load_state_dict(state['ema'],strict=True);sample=state['sampler_state'];del state
        model.eval();model.requires_grad_(False)
        # The fast count path must agree exactly with the full checkpoint model.
        b=batch_input(opaque['CryptBB'][:2]);full=model(*b);p,q=counts_only(model,*b)
        assert torch.equal(full['length'],p) and torch.equal(full['independent_count'],q)
        write(out/'count_forward_parity.json',dict(passed=True,full_vs_count_only_bit_identical=True,device='CPU float32',sampler_state=sample))
        predictions={};started=time.time()
        for name,rows in opaque.items():
            records=[]
            for start in range(0,len(rows),8):
                batch=rows[start:start+8];p,q=counts_only(model,*batch_input(batch))
                assert torch.isfinite(p).all() and torch.isfinite(q).all()
                for i,r in enumerate(batch):records.append(dict(opaque_id=r['opaque_id'],pixel_sha256=r['pixel_sha256'],capri_count=int(p[i].argmax())+2,separate_count=int(q[i].argmax())+2,capri_logits=p[i].tolist(),separate_logits=q[i].tolist()))
                if (start+len(batch))%100<8:progress('predicting',scheme=name,images=len(records),total=len(rows))
            predictions[name]=records
        write(out/'predictions.json',predictions);freeze=dict(created_at=now(),sha256=sha(out/'predictions.json'),checkpoint_sha256=digest,images=1200,seconds=time.time()-started);write(out/'prediction_freeze.json',freeze)
        # Only scoring below can access answer-bearing source names/annotations.
        progress('scoring_frozen_predictions')
        config=read(ROOT/'configs/local.json')['decoder_scheme_inputs'];labelsources={};truth={}
        dark=ROOT/'runs/parseq_darkbay_100k_20261005/evaluation_darkbay/scoring_labels.json';labels=read(dark);labelsources[str(dark)]=sha(dark);truth['DarkBay']={r['opaque_id']:len(labels[r['opaque_id']]) for r in sets['DarkBay']}
        crypt=Path(config['cryptbb'])/'labels.csv';labelsources[str(crypt)]=sha(crypt)
        with crypt.open(encoding='utf-8-sig',newline='') as f:labels={r['filename']:r['label'] for r in csv.DictReader(f)}
        truth['CryptBB']={r['opaque_id']:len(labels[Path(r['source']).name]) for r in sets['CryptBB']}
        metadata=HIST/'real_three_schemes_100_epoch8_20261008_v2/simple_dataset_metadata.json';assert 'always exactly 5 characters' in read(metadata)['description'];labelsources[str(metadata)]=sha(metadata)
        truth['SimpleCaptcha']={r['opaque_id']:5 for r in sets['SimpleCaptcha']}
        city={}
        for split in ('benchmark','test','dev'):
            path=Path(config['city_scoring_splits'])/split/'city.jsonl';labelsources[str(path)]=sha(path)
            for line in path.read_text(encoding='utf8').splitlines():
                row=json.loads(line);h=row['sha256'];n=len(row['label'])
                if h in city:assert city[h]==n
                city[h]=n
        truth['City']={}
        for r in sets['City']:
            h=sha(mapped(r['source']))
            if h in city:truth['City'][r['opaque_id']]=city[h]
        scored={k:dict(images=len(rows),capri=statistics(rows,truth[k],'capri_count'),separate_transformer=statistics(rows,truth[k],'separate_count'),truth_count_distribution=dict(sorted(collections.Counter(truth[k].values()).items()))) for k,rows in predictions.items()}
        assert sha(out/'predictions.json')==freeze['sha256'];assert all(sha(ROOT/p)==h for p,h in training_protocol['sources'].items())
        summary=dict(completed_at=now(),checkpoint=str(checkpoint),checkpoint_sha256=digest,checkpoint_pointer=pointer,sampler_state=sample,weights='EMA',schemes=scored,label_sources=labelsources,prediction_sha256=freeze['sha256'],limitations='Historical target sets, fixed existing sample. CPU FP32. SimpleCaptcha count based on saved publisher metadata. City only exact-file-hash matched annotations; remaining images not assigned assumed labels. No model selection or training changes.')
        write(out/'summary.json',summary);write(out/'scoring_counts.json',truth)
        lines=['# Latest saved CAPRI: target character counts','',f'Checkpoint: {checkpoint}; EMA; epoch {sample["epoch"]}, step {sample["step"]}.','', '| Scheme | CAPRI internal count | Separate count transformer |','|---|---:|---:|']
        for name,s in scored.items():
            x=s['capri'];y=s['separate_transformer'];lines.append(f'| {name} | {x["correct"]}/{x["n"]} ({x["accuracy"]:.1%}) | {y["correct"]}/{y["n"]} ({y["accuracy"]:.1%}) |')
        lines+=['',summary['limitations'],'','Predictions were frozen before scoring. Training source hashes remain unchanged.']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
        progress('complete',schemes=scored)
    except Exception as e:progress('failed',error=repr(e));raise
