"""Paired CAPRI1/CAPRI2 examples with unchanged custom, DINO and mixed controllers."""
import argparse, hashlib, html, os, subprocess, sys, time, types
from pathlib import Path
from .decoder.common import ROOT,read,write,sha,now

FAMILIES=('DarkBay','CryptBB','City')
ARMS=('custom','dino','mixed')
NEW_RUN=ROOT/'runs/capri_lightweight_joint20_100k_20261028'

def worker(args):
    import numpy as np
    import torch
    from PIL import Image
    os.environ['CAPRI_RUN_DIR']=str(Path(args.out).resolve())
    os.environ['CAPRI_TARGET_MANIFEST']=str(Path(args.targets).resolve())
    from . import runtime as U
    from . import pipeline as P
    from .lightweight.legacy_model import Iter2Model
    from .lightweight.model import LightweightCapri
    from feature_bank.iter2.generate import TEXT_HEADS,CATEGORIES,STROKES
    from .decoder.appearance import CATEGORIES as NEW_CATEGORIES,STROKES as NEW_STROKES
    assert tuple(CATEGORIES)==tuple(NEW_CATEGORIES) and tuple(STROKES)==tuple(NEW_STROKES)
    torch.manual_seed(101073);np.random.seed(101073)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    selection=read(NEW_RUN/'completion.json') if args.version=='capri2' else U.registry()['models']['hybrid']
    cp=Path(selection['checkpoint']) if args.version=='capri2' else U.model_path('hybrid')
    assert sha(cp)==selection['sha256']
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network=LightweightCapri() if args.version=='capri2' else Iter2Model()
            state=torch.load(cp,map_location='cpu',weights_only=False)
            self.network.load_state_dict(state['ema'] if args.version=='capri2' else state,strict=True)
        def forward(self,image,cols,widths):
            if args.version=='capri1':return self.network(image,cols)
            # Trained DINO residual font heads; internal CAPRI count. Independent counter is unused.
            out=self.network.capri(image,cols);embedding,font=self.network.font_features(image,widths)
            out['font_cat']=out['font_cat']+font[:,:5];out['font_stroke']=out['font_stroke']+font[:,5:8]
            out['font_ordinal']=out['font_ordinal']+font[:,8:48].reshape(-1,4,10);out['font_italic']=out['font_italic']+font[:,48]
            return out
    @torch.inference_mode()
    def describe(self,images):
        descriptions=[]
        for im in images:
            h,w=im.shape[:2];rw=max(8,min(480,round(w*96/h)));cols=(rw+7)//8;wp=(cols*8+32+7)//8*8
            a=np.asarray(Image.fromarray(im).resize((rw,96),Image.Resampling.BILINEAR))
            a=np.pad(a,((0,0),(0,wp-rw),(0,0)),mode='edge')
            x=torch.from_numpy(a.transpose(2,0,1).copy()).to(self.device).float()[None]/255
            c=torch.tensor([cols],device=self.device);width=torch.tensor([rw],device=self.device)
            o=self.hybrid(x,c,width)
            if not getattr(self,'comparison_parity_checked',False):
                expected=self.hybrid.network(x,c,width) if args.version=='capri2' else self.hybrid.network(x,c)
                checked=('length','eff_presence','eff_ordinal','txt_presence','txt_ordinal','font_cat','font_stroke','font_ordinal','font_italic')
                assert all(torch.equal(o[k],expected[k]) for k in checked)
                write(P.ROOT/'adapter_verification.json',dict(passed=True,full_model_bit_identical=list(checked),version=args.version,checkpoint_sha256=selection['sha256'],internal_count=True,dino_font_branch=args.version=='capri2',separate_counter_used=False))
                self.comparison_parity_checked=True
            groups={}
            for block,prefix,names in [('effects','eff',self.effect_names),('text_features','txt',TEXT_HEADS)]:
                groups[block]={name:dict(present=bool(o[prefix+'_presence'][0,i]>0),level=int((o[prefix+'_ordinal'][0,i]>0).sum()),probability=float(o[prefix+'_presence'][0,i].sigmoid())) for i,name in enumerate(names)}
            font=dict(category=CATEGORIES[int(o['font_cat'].argmax())],stroke=STROKES[int(o['font_stroke'].argmax())],italic=bool(o['font_italic'][0]>0))
            font.update({name:int((o['font_ordinal'][0,i]>0).sum()) for i,name in enumerate(['weight','width','x_height','slant'])})
            content=['digits','upper','lower','mixed'][int(np.argmax([groups['text_features'][n]['probability'] for n in ['digits_only','uppercase_only','lowercase_only','mixed_case_alnum']]))]
            descriptions.append(dict(size=[w,h],length=int(o['length'].argmax())+2,font=font,content=content,**groups))
        return descriptions
    def initialize(model):
        model.custom.hybrid=Adapter().to(model.custom.device).eval().requires_grad_(False)
        model.custom.describe=types.MethodType(describe,model.custom)
        model.custom.description_checkpoint=cp
        write(P.ROOT/'model_manifest.json',dict(version=args.version,description_checkpoint=str(cp),description_sha256=selection['sha256'],controller_models={k:v for k,v in U.registry()['models'].items() if k!='hybrid'},preprocessing='Shared RGB96,bilinear,widthcap480,edgepad; FP32',count='CAPRI internal length',font='Trained DINO residual heads' if args.version=='capri2' else 'Original CAPRI font heads'))
    P.main(preview=True,model_initializer=initialize,calibration_source=args.calibration)

def mapped(path):
    s=str(path).replace('\\','/');old='C:/Users/pranav/Documents/ChatGPT/thesis/'
    p=ROOT/s[len(old):] if s.lower().startswith(old.lower()) else Path(s)
    p=p.resolve()
    if p.drive.upper()!='E:' or not p.is_file():raise ValueError(str(p))
    return p

def report(out,results):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    font_path=Path('E:/pranav/CAPRI/assets/report_fonts/arial.ttf')
    # Use the installed font read-only; the rendered files are all saved on E.
    font=ImageFont.truetype(str(font_path) if font_path.exists() else 'C:/Windows/Fonts/arial.ttf',20)
    small=ImageFont.truetype(str(font_path) if font_path.exists() else 'C:/Windows/Fonts/arial.ttf',16)
    page=['<!doctype html><meta charset="utf-8"><title>CAPRI 1 vs CAPRI 2: controller examples</title><style>body{font:17px system-ui;margin:28px;color:#182033;background:#f4f6f8}img{max-width:100%;height:auto}section{margin:30px 0}a{color:#145aad}.note{max-width:1100px}</style><h1>CAPRI 1 vs CAPRI 2</h1><p class="note">Original epoch-15 CAPRI versus new epoch-20 CAPRI with trained DINO font heads and internal count. Same reference pixels, custom/DINO/mixed controller weights, calibration, candidate-search policy and seed schedules. Six fixed inputs per scheme; all 100 reference images contribute to appearance comparison. Fresh synthetic text is generated independently of target text.</p><p class="note">The model-vote length policy is unchanged: at least 60/100 votes selects a common count; otherwise the bounded preview cycles equally through observed predicted counts. Models may therefore generate different lengths. Shared preprocessing is RGB96/bilinear/cap480. Visual previews do not establish reader accuracy. CAPRI 2 training lacked text-ordinal and font width/slant supervision; those outputs remain a limitation of recipe initialization. Controllers and renderer are unchanged.</p>']
    overview=Image.new('RGB',(1680,80+3*310),'#f4f6f8');od=ImageDraw.Draw(overview)
    titles=['Target','C1 Custom','C1 DINO','C1 Mixed','C2 Custom','C2 DINO','C2 Mixed']
    for j,title in enumerate(titles):od.text((j*240+10,20),title,font=font,fill='#182033')
    audit={}
    for si,name in enumerate(FAMILIES):
        refs=read(out/'inputs'/f'{name}.json')['fit'];target_hashes={r['pixel_sha256'] for r in refs}
        labels={};entries={};failures=[]
        for version in ('capri1','capri2'):
            run=out/name/version
            entries[version]=read(run/'entries.json') if (run/'entries.json').exists() else []
            labels[version]={a:read(run/a/'labels.json') if (run/a/'labels.json').exists() else [] for a in ARMS}
            if results.get(name,{}).get(version,{}).get('returncode')!=0:failures.append(version)
        sheet=Image.new('RGB',(1680,100+6*130),'#f4f6f8');d=ImageDraw.Draw(sheet)
        for j,title in enumerate(titles):d.text((j*240+10,18),title,font=font,fill='#182033')
        hashes=[];strict={};counts={}
        for i in range(6):
            cells=[Image.open(refs[i]['source']).convert('RGB')];captions=['Input '+str(i+1)]
            for version in ('capri1','capri2'):
                for arm in ARMS:
                    rows=labels[version][arm]
                    if i<len(rows):
                        r=rows[i];im=Image.open(out/name/version/r['image']).convert('RGB')
                        h=hashlib.sha256(np.asarray(im).tobytes()).hexdigest();assert h not in target_hashes;hashes.append(h)
                        entry=next(e for e in entries[version] if e['controller']==arm and e['input_index']==i)
                        assert r['complete'] and r['usable'] and len(r['text'])==entry['controls']['length']
                        captions.append(f"{len(r['text'])} chars; strict {'pass' if r['strict'] else 'fail'}")
                    else:
                        im=Image.new('RGB',(220,70),'#ffe7e7');ImageDraw.Draw(im).text((8,18),'No saved usable example',font=small,fill='#852b2b');captions.append('Failure retained')
                    cells.append(im)
            for j,(im,caption) in enumerate(zip(cells,captions)):
                tile=ImageOps.contain(im,(224,88),Image.Resampling.NEAREST)
                sheet.paste(tile,(j*240+(240-tile.width)//2,55+i*130));d.text((j*240+8,146+i*130),caption,font=small,fill='#455369')
                if i<2:
                    y=80+si*310+i*130;overview.paste(tile,(j*240+(240-tile.width)//2,y+25));od.text((j*240+8,y),name+' '+str(i+1) if j==0 else caption,font=small,fill='#455369')
        sheet.save(out/(name+'.png'))
        for version in ('capri1','capri2'):
            strict[version]={a:sum(bool(r['strict']) for r in labels[version][a]) for a in ARMS}
            run=out/name/version
            if (run/'length_decision.json').exists():counts[version]=read(run/'length_decision.json')
        calibration_equal=None
        if not failures:
            a=read(out/name/'capri1/controller_calibration.json');b=read(out/name/'capri2/controller_calibration.json');a.pop('source');b.pop('source')
            assert a==b;calibration_equal=True
            assert read(out/name/'capri1/inputs.json')==read(out/name/'capri2/inputs.json')==refs
            assert all(read(out/name/v/'adapter_verification.json')['passed'] for v in ('capri1','capri2'))
            assert len(hashes)==36
        audit[name]=dict(examples=len(hashes),unique_output_pixels=len(set(hashes)),strict_per_arm=strict,failures=failures,identical_controller_calibration=calibration_equal,length_decisions=counts)
        page+=['<section><h2>'+name+'</h2><p>CAPRI 1: '+html.escape(str(counts.get('capri1',{}).get('counts',{})))+'; CAPRI 2: '+html.escape(str(counts.get('capri2',{}).get('counts',{})))+'</p><img src="'+name+'.png"><p><a href="'+name+'.png">Open full-size comparison</a></p></section>']
    overview.save(out/'overview.png');(out/'report.html').write_text('\n'.join(page),encoding='utf-8')
    write(out/'audit.json',dict(created_at=now(),schemes=audit,total_examples=sum(x['examples'] for x in audit.values()),expected_examples=108,target_labels_used=False,registry_changed=False))
    return audit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--worker',action='store_true');ap.add_argument('--version',choices=['capri1','capri2']);ap.add_argument('--targets');ap.add_argument('--calibration')
    args=ap.parse_args();out=Path(args.out).resolve()
    if out.drive.upper()!='E:':raise ValueError('E-drive output required')
    if args.worker:return worker(args)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'inputs').mkdir();(out/'temporary').mkdir()
    for key in ('TEMP','TMP','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','MPLCONFIGDIR'):os.environ[key]=str(out/'temporary')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    import numpy as np
    from PIL import Image
    registry=read(ROOT/'models/registry.json');registry_sha=sha(ROOT/'models/registry.json')
    for entry in registry['models'].values():assert sha(ROOT/entry['path'])==entry['sha256']
    completion=read(NEW_RUN/'completion.json');assert completion['epochs']==20 and sha(Path(completion['checkpoint']))==completion['sha256']
    for name in FAMILIES:
        source=ROOT/('capri_length_cryptbb_100/predictions.json' if name=='CryptBB' else f'capri_length_100/{name}/predictions.json')
        rows=read(source);assert len(rows)==100 and len({r['pixel_sha256'] for r in rows})==100
        folder=out/'inputs'/name;folder.mkdir();refs=[]
        for i,r in enumerate(rows):
            original=mapped(r['source']);im=Image.open(original).convert('RGB');assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
            path=folder/f'{i:03d}_{r["pixel_sha256"][:12]}.png';im.save(path)
            refs.append(dict(source=str(path),pixel_sha256=r['pixel_sha256']))
        write(out/'inputs'/f'{name}.json',dict(fit=refs,source_manifest=str(source),source_manifest_sha256=sha(source)))
    sources=[ROOT/'capri.py',ROOT/'configs/local.json',ROOT/'models/registry.json',ROOT/'src/capri_controller/controller_comparison.py',ROOT/'src/capri_controller/pipeline.py',ROOT/'src/capri_controller/controllers.py',ROOT/'src/capri_controller/artifacts.py',ROOT/'src/capri_controller/runtime.py',*list((ROOT/'src/capri_controller/features').glob('*.py')),*list((ROOT/'src/capri_controller/rendering').glob('*.py')),*list((ROOT/'src/capri_controller/lightweight').glob('*.py'))]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for p in sources:
        dest=out/'provenance'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(out/'protocol.json',dict(created_at=now(),families=FAMILIES,reference_images_per_scheme=100,examples_per_controller=6,controllers=ARMS,capri1=registry['models']['hybrid'],capri2=completion,controller_registry_sha256=registry_sha,calibration='CAPRI1 first24 model-derived calibration recipes reused unchanged for CAPRI2 within each scheme',selection='First6 fixed fit inputs, no selection by visual outcome',preprocessing='Both RGB96,bilinear,widthcap480,FP32',seeds='Same existing pipeline seed schedules and length fallback seed101073',target_labels=False,separate_counter_used=False,source_hashes=hashes))
    results={};started=time.time()
    for name in FAMILIES:
        results[name]={}
        for version in ('capri1','capri2'):
            run=out/name/version;run.parent.mkdir(exist_ok=True)
            write(out/'status.json',dict(state='running',scheme=name,version=version,completed=results,seconds=time.time()-started))
            cmd=[sys.executable,'-B','-u',str(ROOT/'capri.py'),'compare-capri-controllers','--worker','--out',str(run),'--version',version,'--targets',str(out/'inputs'/f'{name}.json')]
            if version=='capri2':
                calibration=out/name/'capri1/calibration_features.npz'
                if not calibration.exists():results[name][version]=dict(returncode=-1,error='CAPRI1 shared calibration unavailable');continue
                cmd+=['--calibration',str(calibration)]
            with (out/f'{name}_{version}.log').open('w',encoding='utf-8') as log:
                proc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            results[name][version]=dict(returncode=proc.returncode,log=f'{name}_{version}.log')
            write(out/'results.json',results);print(name,version,'finished',proc.returncode,flush=True)
    audit=report(out,results)
    assert sha(ROOT/'models/registry.json')==registry_sha
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    write(out/'status.json',dict(state='complete' if all(not s['failures'] for s in audit.values()) else 'completed_with_errors',seconds=time.time()-started,examples=sum(s['examples'] for s in audit.values())))
    print('Comparison report:',out/'report.html',flush=True)
