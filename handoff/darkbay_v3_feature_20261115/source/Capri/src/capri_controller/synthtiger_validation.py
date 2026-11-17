"""Independent CPU-only SynthTIGER validation controller, never a training source."""
import argparse, hashlib, io, json, math, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'Capri/runs/darkbay_controller4_synthtiger_1000_20261113'
DATA=ROOT/'datasets/darkbay/validation/controller4_synthtiger_1000'
PKGS=ROOT/'runtime/controller4_packages'
SEED_FIT=3900000000
SEED_EXPORT=4010000000
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYGAME_HIDE_SUPPORT_PROMPT='1')
sys.path.insert(0,str(PKGS))
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.distance import cdist
from synthtiger import layers,components
from fontTools.ttLib import TTFont
cv2.setNumThreads(1)
try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)
except Exception:
    pass

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def phash(a): return hashlib.sha256(np.asarray(a,dtype=np.uint8).tobytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');os.replace(q,p)

class LatinTextLayer(layers.TextLayer):
    """Pillow>=10 getlength and basic Latin layout; no external RAQM needed."""
    def _get_direction(self,vertical):
        assert not vertical
        return None
    def _get_bbox(self,text,font,vertical):
        assert not vertical
        ascent,descent=font.getmetrics()
        return [0,-ascent,int(math.ceil(font.getlength(text)))+4,ascent+descent+4]

def feature(a):
    x=np.asarray(a,dtype=np.float64)/255.;g=x.mean(2)
    dark=g<.16;light=g>.85
    bg=x[(g>.20)&(g<.82)]
    if len(bg)<20: bg=x.reshape(-1,3)
    clean=x.copy();clean[dark|light]=np.median(bg,axis=0)
    low=cv2.GaussianBlur(clean.mean(2),(3,3),.7)
    dx=np.abs(np.diff(low,axis=1));dy=np.abs(np.diff(low,axis=0))
    return np.r_[np.quantile(g,[.05,.25,.5,.75,.95]),np.median(bg,axis=0),
       np.std(bg,axis=0),dark.mean(),light.mean(),dx.mean(),dy.mean(),
       (dx>.035).mean(),(dy>.035).mean(),np.mean(x.max(2)-x.min(2)),
       np.mean(np.abs(np.diff(g,axis=1))),np.mean(np.abs(np.diff(g,axis=0)))]

def glyph_feature(mask):
    from scipy.ndimage import binary_fill_holes
    yy,xx=np.where(mask)
    if len(xx)<8:return None
    m=mask[yy.min():yy.max()+1,xx.min():xx.max()+1].astype('uint8')
    h,w=m.shape;m=cv2.resize(m,(max(2,round(32*w/h)),32),interpolation=cv2.INTER_NEAREST)
    area=max(1,m.sum());filled=binary_fill_holes(m)
    contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hull=cv2.convexHull(np.concatenate(contours))
    hullmask=np.zeros_like(m);cv2.fillConvexPoly(hullmask,hull,1)
    return np.array([w/h,m.mean(),cv2.erode(m,np.ones((3,3),'uint8')).sum()/area,
        cv2.erode(m,np.ones((5,5),'uint8')).sum()/area,
        (filled.sum()-area)/max(1,filled.sum()),area/max(1,hullmask.sum()),
        np.std(m.mean(0)),np.std(m.mean(1))],dtype=float)

def shape_fonts(fonts,charset,pixels,H):
    dest=RUN/'font_shape_matching.json'
    if dest.exists():return read(dest)['selected_fonts']
    target=[]
    for a in pixels:
        g=a.mean(2)/255
        for binary in [g<.16,g>.85]:
            _,labels,boxes,_=cv2.connectedComponentsWithStats(binary.astype('uint8'))
            for i,b in enumerate(boxes[1:],1):
                if b[4]>=12 and 4<=b[2]<=40 and 12<=b[3]<H:
                    f=glyph_feature(labels[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]==i)
                    if f is not None:target.append(f)
    T=np.stack(target);std=np.maximum(T.std(0),.03);T=T/std
    results=[]
    for font in fonts:
        F=[]
        for c in charset:
            a=LatinTextLayer(c,str(ROOT/font['path']),32).output()
            f=glyph_feature(a[:,:,3]>96)
            if f is not None:F.append(f)
        if len(F)!=len(charset):continue
        F=np.stack(F)/std
        dist=cdist(F,T)
        # Symmetric distribution distance covers multiple unknown reference glyphs.
        score=float(2*dist.mean()-cdist(F,F).mean()-cdist(T,T).mean())
        results.append(dict(font=font,score=score))
    results.sort(key=lambda x:x['score'])
    selected=[r['font'] for r in results[:64]]
    assert len(selected)==64
    write(dest,dict(method='Energy distance on unlabeled connected-component shape/stroke statistics',
        target_components=len(T),candidate_fonts=len(fonts),selected_fonts=selected,ranking=results))
    return selected

def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    expected=read(ROOT/'Capri/controllers/darkbay/inputs.json')
    expected_hashes={r['pixel_sha256'] for r in expected}
    paths=sorted((ROOT/'Capri/controllers/darkbay/reference_images').glob('*.png'))
    pixels=[np.asarray(Image.open(p).convert('RGB')) for p in paths]
    assert len(paths)==100 and {phash(a) for a in pixels}==expected_hashes
    evaluation={r['pixel_sha256'] for r in read(ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json')}
    assert not evaluation&expected_hashes
    vote=read(ROOT/'Capri/controllers/darkbay/length_decision.json')
    counts=sorted(((int(k),v) for k,v in vote['counts'].items()),key=lambda kv:-kv[1])
    assert counts[0][1]>=50 and counts[0][1]>counts[1][1]
    charset=(ROOT/'datasets/darkbay/A/charset.txt').read_text().splitlines()
    assert charset==(ROOT/'datasets/darkbay/B/charset.txt').read_text().splitlines()
    assert all(len(c)==1 and c.isascii() and c.isalnum() for c in charset)
    W,H=Image.open(paths[0]).size
    assert all(a.shape==(H,W,3) for a in pixels)
    stats=[]
    for a in pixels:
        x=a.astype(float);g=x.mean(2)/255
        bg=x[(g>.20)&(g<.82)]
        if len(bg)<20:bg=x.reshape(-1,3)
        _,_,boxes,_=cv2.connectedComponentsWithStats(((g<.16)|(g>.85)).astype('uint8'))
        heights=[int(b[3]) for b in boxes[1:] if b[4]>=8 and 3<=b[2]<W*.3 and 7<=b[3]<H]
        stats.append(dict(background=np.median(bg,axis=0).tolist(),spread=float(np.median(np.std(bg,axis=0))),
            height=float(np.median(heights)) if heights else H*.6,
            white_probability=float((g>.85).sum()/max(1,((g<.16)|(g>.85)).sum()))))
    manifest=RUN/'font_manifest.json'
    if manifest.exists(): fonts=read(manifest)
    else:
        fonts=[];seen=set()
        for p in sorted((ROOT/'Capri').rglob('*')):
            if p.suffix.lower() not in ('.ttf','.otf'):continue
            digest=sha(p)
            if digest in seen:continue
            seen.add(digest)
            try:
                font=TTFont(str(p),lazy=True); cmap=font.getBestCmap();ok=all(ord(c) in cmap for c in charset);font.close()
                if not ok:continue
                f=ImageFont.truetype(str(p),32)
                if min(f.getlength(c) for c in charset)<2:continue
                fonts.append(dict(path=str(p.relative_to(ROOT)),sha256=digest))
            except Exception:continue
        assert len(fonts)>=10
        write(manifest,fonts)
    for f in fonts: assert sha(ROOT/f['path'])==f['sha256']
    font_check=RUN/'font_render_validation.json'
    if not font_check.exists():
        accepted=[];rejected=[]
        for f in fonts:
            try:
                for c in charset:
                    a=LatinTextLayer(c,str(ROOT/f['path']),int(H*.85)).output()
                    assert int((a[:,:,3]>16).sum())>=8, 'empty or tiny rendered glyph '+c
                accepted.append(f)
            except Exception as e:rejected.append(dict(font=f,reason=str(e)))
        assert len(accepted)>=10
        write(font_check,dict(accepted=accepted,rejected=rejected,characters_tested_per_font=len(charset)))
    fonts=shape_fonts(read(font_check)['accepted'],charset,pixels,H)
    context=dict(width=W,height=H,length=counts[0][0],vote_count=counts[0][1],charset=charset,
        stats=stats,fonts=fonts,reference_hashes=sorted(expected_hashes),
        length_decision_sha256=sha(ROOT/'Capri/controllers/darkbay/length_decision.json'),
        charset_sha256=sha(ROOT/'datasets/darkbay/A/charset.txt'))
    write(RUN/'reference_measurements.json',context)
    return context,np.stack([feature(a) for a in pixels])

def render(ctx,recipe,seed):
    rng=np.random.default_rng(seed);np.random.seed(seed)
    W,H,n=ctx['width'],ctx['height'],ctx['length']
    ref=ctx['stats'][int(rng.integers(len(ctx['stats'])))];base=np.array(ref['background'])
    # Sample fresh stripe geometry, never paste reference image pixels.
    bg=layers.RectLayer((W,H),tuple(base)+(255,))
    strips=[];x=0
    while x<W:
        width=int(rng.integers(recipe['stripe_min'],recipe['stripe_max']+1))
        col=np.clip(base+rng.normal(0,max(4,ref['spread'])*recipe['stripe_contrast']),35,220)
        patch=layers.RectLayer((min(width,W-x),H),tuple(col)+(255,));patch.topleft=(x,0)
        strips.append(patch);x+=width
    for i in range(int(recipe['horiz_count'])):
        col=np.clip(base+rng.normal(0,max(4,ref['spread'])*recipe['stripe_contrast']),35,220)
        patch=layers.RectLayer((W,int(rng.integers(1,5))),tuple(col)+(255,));patch.topleft=(0,int(rng.integers(0,H-4)));strips.append(patch)
    for patch in strips:bg=bg.paste(patch)
    text=''.join(rng.choice(ctx['charset'],n));glyphs=[];details=[]
    slot=(W-8)/n
    for i,char in enumerate(text):
        f=ctx['fonts'][int(rng.integers(len(ctx['fonts'])))];white=bool(rng.random()<ref['white_probability'])
        shade=int(rng.integers(238,256) if white else rng.integers(0,25))
        layer=LatinTextLayer(char,str(ROOT/f['path']),int(H*.85),color=(shade,shade,shade,255))
        arr=layer.output();yy,xx=np.where(arr[:,:,3]>0)
        assert len(xx)>0
        layer=layers.Layer(arr[yy.min():yy.max()+1,xx.min():xx.max()+1])
        height=float(np.clip(ref['height']*recipe['height_scale']*rng.uniform(.8,1.2),H*.32,H*.86))
        layer.size=(min(slot*.85,layer.width/layer.height*height*recipe['width_scale']),height)
        angle=float(rng.uniform(-recipe['rotation'],recipe['rotation']))
        components.Rotate(angle=(angle,angle),ccw=1).apply([layer],{'angle':angle,'ccw':True})
        arr=layer.output();yy,xx=np.where(arr[:,:,3]>1)
        layer=layers.Layer(arr[yy.min():yy.max()+1,xx.min():xx.max()+1])
        scale=min(1.,(slot-2)/layer.width,(H-4)/layer.height);layer.size=layer.size*scale
        centerx=4+(i+.5)*slot+rng.uniform(-1,1)
        centery=H/2+rng.uniform(-recipe['jitter'],recipe['jitter'])*H
        layer.center=(centerx,np.clip(centery,2+layer.height/2,H-2-layer.height/2))
        mask=layer.output([0,0,W,H])[:,:,3]
        yy,xx=np.where(mask>16)
        assert len(xx)>=8 and xx.min()>0 and yy.min()>0 and xx.max()<W-1 and yy.max()<H-1
        details.append(dict(character=char,font=f['path'],font_sha256=f['sha256'],rotation=angle,
            bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],visible_pixels=int(len(xx))))
        glyphs.append(layer)
    for layer in glyphs:bg=bg.paste(layer)
    image=np.clip(bg.output([0,0,W,H])[:,:,:3],0,255).astype('uint8')
    if recipe['blur']>0:image=cv2.GaussianBlur(image,(3,3),recipe['blur'])
    return image,text,details

def sheet(images,path,labels=None):
    W,H=200,50;cw,ch=220,76
    canvas=Image.new('RGB',(cw*4,ch*math.ceil(len(images)/4)),(240,240,240));draw=ImageDraw.Draw(canvas)
    for i,a in enumerate(images):
        x=(i%4)*cw+10;y=(i//4)*ch+3
        canvas.paste(Image.fromarray(a).resize((W,H)),(x,y))
        if labels:draw.text((x,y+52),labels[i],fill='black')
    canvas.save(path)

def fit():
    assert not (RUN/'calibration_shape_matched.json').exists(),'Calibration already exists; preserve frozen fitting evidence'
    ctx,target=prepare();std=np.maximum(target.std(0),.02);T=target/std
    rng=np.random.default_rng(47138);candidates=[];start=time.monotonic()
    for i in range(64):
        recipe=dict(stripe_min=int(rng.choice([2,3,4,6])),stripe_max=int(rng.choice([9,13,18,24])),
            stripe_contrast=float(rng.uniform(.5,1.4)),horiz_count=int(rng.integers(0,4)),
            height_scale=float(rng.uniform(.85,1.30)),width_scale=float(rng.uniform(.75,1.25)),
            rotation=float(rng.uniform(5,23)),jitter=float(rng.uniform(.04,.20)),blur=float(rng.choice([0.,.25,.4])))
        images=[render(ctx,recipe,SEED_FIT+j)[0] for j in range(24)]
        F=np.stack([feature(a) for a in images])/std
        score=float(2*cdist(F,T).mean()-cdist(F,F).mean()-cdist(T,T).mean())
        candidates.append(dict(id=i,recipe=recipe,pixel_energy_distance=score))
        write(RUN/'fit_progress.json',dict(candidates=i+1,total=64,seconds=time.monotonic()-start))
        if i%8==0:print('fit',i+1,'/64',score,flush=True)
    selected=sorted(candidates,key=lambda v:v['pixel_energy_distance'])[:8]
    result=dict(method='Seeded random search; standardized pixel-descriptor energy distance to100 unlabeled references',
        independent='SynthTIGER renderer and hand-engineered image statistics; no Custom/DINO embeddings or OCR',
        shared='Reference pixels, font assets, charset and previously frozen CAPRI count vote',
        candidates=candidates,selected=selected,calibration_seed=SEED_FIT,export_seed=SEED_EXPORT,
        protocol='No real scoring labels, no reader predictions/accuracy, all calibration renders excluded from export',
        calibration_images=64*24,seconds=time.monotonic()-start,source_sha256=sha(__file__))
    write(RUN/'calibration_shape_matched.json',result)
    ims=[];labels=[]
    for i in range(32):
        a,t,_=render(ctx,selected[i%8]['recipe'],SEED_FIT+10000+i);ims.append(a);labels.append(t)
    sheet(ims,RUN/'pilot_shape_matched.png',labels)
    print('Fit complete; pilot saved',flush=True)

def generate():
    ctx=read(RUN/'reference_measurements.json');fitdata=read(RUN/'calibration_shape_matched.json')
    assert fitdata['source_sha256']==sha(__file__),'Controller changed after fitting'
    assert not DATA.exists(),'Existing validation corpus must be preserved'
    for f in ctx['fonts']:assert sha(ROOT/f['path'])==f['sha256']
    trainhashes=set();traintexts=set();seed_ranges=[]
    for tag in ('A','B'):
        for line in (ROOT/f'model1/runs/darkbay/{tag}/sample_provenance.jsonl').read_text().splitlines():
            r=json.loads(line);trainhashes.add(r['pixel_sha256']);traintexts.add(r['text'])
        for p in (ROOT/f'datasets/darkbay/{tag}/corpus').glob('*.json'):
            r=read(p)
            if 'seed_base' in r:seed_ranges.append((r['seed_base'],r['seed_base']+100000))
    forbidden=trainhashes|set(ctx['reference_hashes'])|{r['pixel_sha256'] for r in read(ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json')}
    protocol=dict(controller='controller4_synthtiger_pixel_energy_v1',role='validation_only',count=1000,
        source_sha256=sha(__file__),calibration_sha256=sha(RUN/'calibration_shape_matched.json'),
        reference_measurements_sha256=sha(RUN/'reference_measurements.json'),seed_base=SEED_EXPORT,
        generation_framework='SynthTIGER1.2.1',selected_before_reader_evaluation=True,
        historical_limit='Post-hoc DarkBay validation experiment; existing fixed epoch20 primary result retained',
        target_labels_used=False,reader_used=False,length=ctx['length'],length_vote_count=ctx['vote_count'],
        shared_assets=fitdata['shared'],compatibility='Isolated numpy1.26.4; Latin TextLayer getlength/basic-layout adapter for Pillow11',
        limitation='Pixel similarity is not proof of real-transfer checkpoint ranking or human legibility')
    write(RUN/'generation_protocol.json',protocol)
    (DATA/'images').mkdir(parents=True)
    records=[];seen=set();texts=set();i=0;attempt=0
    while i<1000:
        seed=SEED_EXPORT+attempt;attempt+=1
        assert not any(lo<=seed<hi for lo,hi in seed_ranges)
        selected=fitdata['selected'][i%len(fitdata['selected'])]
        a,text,details=render(ctx,selected['recipe'],seed);digest=phash(a)
        if digest in forbidden or digest in seen or text in traintexts or text in texts:continue
        seen.add(digest);texts.add(text);rel=f'images/{i:06d}.png';Image.fromarray(a).save(DATA/rel)
        records.append(dict(index=i+1,image=rel,text=text,seed=seed,recipe_id=selected['id'],
            pixel_sha256=digest,file_sha256=sha(DATA/rel),glyphs=details));i+=1
        if i%100==0:write(RUN/'generation_status.json',dict(state='generating',images=i,total=1000));print('generated',i,flush=True)
    (DATA/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    (DATA/'labels.tsv').write_text(''.join(r['image']+'\t'+r['text']+'\n' for r in records),encoding='utf-8')
    (DATA/'charset.txt').write_text('\n'.join(ctx['charset'])+'\n',encoding='utf-8')
    write(DATA/'protocol.json',protocol)
    images=[];names=[]
    for r in records[:32]:images.append(np.asarray(Image.open(DATA/r['image'])));names.append(r['text'])
    sheet(images,RUN/'validation_contact_sheet.png',names)
    # Pack only under a validation-specific path. No training config points here.
    import lmdb
    env=lmdb.open(str(DATA/'validation_lmdb'),map_size=256*1024*1024)
    with env.begin(write=True) as tx:
        for r in records:
            tx.put(('image-%09d'%r['index']).encode(),(DATA/r['image']).read_bytes())
            tx.put(('label-%09d'%r['index']).encode(),r['text'].encode())
        tx.put(b'num-samples',b'1000')
    env.sync();env.close()
    audit=dict(images=1000,unique_pixels=len(seen),unique_labels=len(texts),rejections=attempt-1000,
        train_pixel_overlap=len(seen&trainhashes),reference_pixel_overlap=len(seen&set(ctx['reference_hashes'])),
        real_test_pixel_overlap=len(seen&forbidden),train_text_overlap=len(texts&traintexts),
        seed_range_disjoint=True,manifest_sha256=sha(DATA/'manifest.jsonl'),labels_sha256=sha(DATA/'labels.tsv'),
        lmdb_sha256=sha(DATA/'validation_lmdb/data.mdb'),protocol_sha256=sha(DATA/'protocol.json'))
    assert audit['train_pixel_overlap']==audit['reference_pixel_overlap']==audit['real_test_pixel_overlap']==audit['train_text_overlap']==0
    env=lmdb.open(str(DATA/'validation_lmdb'),readonly=True,lock=False)
    with env.begin() as tx:
        assert int(tx.get(b'num-samples'))==1000
        for r in records:
            a=np.asarray(Image.open(io.BytesIO(tx.get(('image-%09d'%r['index']).encode()))).convert('RGB'))
            assert phash(a)==r['pixel_sha256'] and a.shape==(ctx['height'],ctx['width'],3)
            assert tx.get(('label-%09d'%r['index']).encode()).decode()==r['text']
            assert len(r['text'])==ctx['length'] and set(r['text'])<=set(ctx['charset'])
            assert len(r['glyphs'])==ctx['length'] and all(g['visible_pixels']>=8 for g in r['glyphs'])
    env.close();audit['all1000_decoded_and_labels_verified']=True
    for idx in [0,499,999]:
        r=records[idx];a,t,d=render(ctx,next(s['recipe'] for s in fitdata['selected'] if s['id']==r['recipe_id']),r['seed'])
        assert phash(a)==r['pixel_sha256'] and t==r['text']
    audit['deterministic_replay_checks']=3
    write(RUN/'audit.json',audit);write(DATA/'audit.json',audit)
    write(RUN/'generation_status.json',dict(state='complete',images=1000,training_started=False))
    print(json.dumps(audit),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['fit','generate'],required=True);args=ap.parse_args()
    if args.stage=='fit':fit()
    else:generate()
if __name__=='__main__':main()
