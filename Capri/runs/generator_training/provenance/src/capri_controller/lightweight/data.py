"""Lossless manifest selection -> original CAPRI height96 RGB cache and masked labels."""
import collections,concurrent.futures,hashlib,io,json,math,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from ..decoder.common import read,write,sha,now
from ..decoder.appearance import font_targets,CATEGORIES,STROKES
from .. import architecture_data as A

H=96;MAX_W=480;MARGIN=32;BUDGET=128*352
CHARS='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
TEXT=['per_char_rotation','vertical_jitter','size_variation','text_on_curve','slant','overlap_tight_spacing','wide_spacing','condensed','expanded','font_mixture','multicolor_text','two_layer','digits_only','uppercase_only','lowercase_only','mixed_case_alnum']
EFFECTS=['bg_gradient','bg_image_texture','bg_blocks','bg_grid','bg_circles','bg_decoy_chars','border_frame','straight_lines','bands','curves','scribbles','dot_noise','blob_noise','bg_pattern','glyph_shadow','extrude_3d','glyph_outline','hollow_outline_glyphs','stroke_bold','stroke_thin','glyph_fill_texture','translucent_text','strike_through','noise_over_text','occlusion','glyph_cuts','wave_warp','morph_distortion','perspective','blur','local_warp','inverted_regions','pixel_noise','compression','pixelation','low_resolution','colour_quantization']
LABELS=['eff_present','eff_level','txt_present','txt_level','chars','length','font','families','single_family','ctc_valid']

def status(out,**kw):
    d=dict(time=now(),**kw);write(Path(out)/'status.json',d);print(json.dumps(d),flush=True)

def mode_known(values):
    v=[int(x) for x in values if int(x)>=0]
    if not v:return -1
    counts=collections.Counter(v);m=max(counts.values());best=[k for k,n in counts.items() if n==m]
    return best[0] if len(best)==1 else -1

def labels(d,fonts,fids):
    g=d['glyphs'];r=d['recipe'];t=r['text'];n=d['count'];s=''.join(x['char'] for x in g)
    assert n==len(g)==len(s) and 2<=n<=8 and all(c in CHARS for c in s)
    ep=[d['effects'][h]['present'] for h in EFFECTS];el=[d['effects'][h]['level'] for h in EFFECTS]
    tp=[int(any(abs(x['angle'])>1e-6 for x in g)),int(t.get('jitter',0)>0),int(t.get('size_jitter',0)>0),int(t.get('curve',0)!=0),int(t.get('shear',0)!=0),int(t.get('spacing',0)<0),int(t.get('spacing',0)>0),int(t.get('stretch',1)<1),int(t.get('stretch',1)>1),int(len({x['font_key'] for x in g})>1),int(len({tuple(x['colour']) for x in g})>1),int(d['rows']>1),int(s.isdigit()),int(s.isalpha() and s.isupper()),int(s.isalpha() and s.islower()),int(sum((any(c.isdigit() for c in s),any(c.islower() for c in s),any(c.isupper() for c in s)))>1)]
    # The decoder corpus has no legacy ordinal text-head labels. Do not invent them.
    tl=[-1]*len(TEXT)
    meta=font_targets(g,[fonts[x['font_key']] for x in g]);v=meta['values'];ok=meta['valid']
    def vals(k):return v[k][ok[k]].tolist()
    def median_level(k):
        a=vals(k);return int(np.clip(np.rint(np.median(a)*10),0,10)) if a else -1
    # Width/slant legacy ordinal scales are not equivalent to available font metadata.
    f=[mode_known(vals('category')),mode_known(vals('stroke')),median_level('weight'),-1,median_level('x_height'),-1,int(any(vals('italic'))) if vals('italic') else -1]
    fam=sorted({fids[fonts[x['font_key']]['split_group']] for x in g});family=np.full(8,-1,np.int32);family[:len(fam)]=fam
    chars=np.full(8,-1,np.int16);chars[:n]=[CHARS.index(c) for c in s]
    return dict(eff_present=np.array(ep,np.int8),eff_level=np.array(el,np.int8),txt_present=np.array(tp,np.int8),txt_level=np.array(tl,np.int8),chars=chars,length=np.int16(n),font=np.array(f,np.int16),families=family,single_family=np.int32(fam[0] if len(fam)==1 else -1),ctc_valid=np.bool_(d['rows']==1))

def _chunk(job):
    root,out,split,num,rows,fonts,fids,reference=job
    root=Path(root);dest=Path(out)/split/f'part_{num:04d}';dest.mkdir(parents=True)
    offsets=[];widths=[];lab=[];offset=0;handles={}
    with (dest/'images.u8').open('wb') as stream:
        for row in rows:
            archive=row['archive']
            if archive not in handles:handles[archive]=zipfile.ZipFile(root/archive)
            z=handles[archive];rgb=A.pixels(z.read(row['stem']+'.png'),row['pixel_sha256'])
            h,w=rgb.shape[:2];nw=max(8,min(MAX_W,round(w*H/h)))
            im=np.asarray(Image.fromarray(rgb).resize((nw,H),Image.Resampling.BILINEAR))
            stream.write(im.transpose(1,0,2).copy().tobytes());offsets.append(offset);widths.append(nw);offset+=nw
            if not reference:
                d=json.loads(z.read(row['stem']+'.json'));assert d['recipe']['split']==split
                assert all(fonts[x['font_key']]['split']==split for x in d['glyphs'])
                lab.append(labels(d,fonts,fids))
    for z in handles.values():z.close()
    arrays=dict(offsets=np.array(offsets,np.int64),widths=np.array(widths,np.int16))
    if not reference:arrays.update({k:np.stack([a[k] for a in lab]) for k in LABELS})
    np.savez(dest/'index.npz',**arrays)
    write(dest/'meta.json',dict(rows=rows,reference=reference,split=split,images=len(rows)))
    return dict(path=dest.relative_to(out).as_posix(),images=len(rows),sha256={p.name:sha(p) for p in dest.iterdir()})

def prepare(dataset,out):
    dataset=Path(dataset).resolve();out=Path(out).resolve()
    if out.drive.upper()!='E:' or out.exists():raise ValueError('Fresh E-drive output required')
    out.mkdir(parents=True);status(out,state='verifying_source')
    verified=A.verify(dataset);fonts={f['key']:f for f in read(dataset/'fonts.json')['fonts']}
    families=sorted({f['split_group'] for f in fonts.values()});fids={f:i for i,f in enumerate(families)}
    train=A.read_rows(dataset/'train.jsonl');dev=A.read_rows(dataset/'dev.jsonl')
    assert len(train)==100000 and len(dev)==2000
    assert not ({r['pixel_sha256'] for r in train}&{r['pixel_sha256'] for r in dev})
    jobs=[];sets={'train':train,'dev':dev}
    for split in ('train','dev'):
        refs=A.read_rows(dataset/f'references_{split}.jsonl');groups=collections.defaultdict(list)
        for r in refs:groups[r['font_key']].append(r)
        sets['references_'+split]=[r for k in sorted(groups) for r in sorted(groups[k],key=lambda x:x['specimen_id'])[:3]]
    for split,rows in sets.items():
        for i in range(0,len(rows),2000):jobs.append((str(dataset),str(out),split,i//2000,rows[i:i+2000],fonts,fids,split.startswith('references')))
    receipts=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for receipt in pool.map(_chunk,jobs):
            receipts.append(receipt);status(out,state='building_cache',parts=len(receipts),total_parts=len(jobs),images=sum(r['images'] for r in receipts))
    protocol=dict(created_at=now(),source=str(dataset),source_integrity_sha256=sha(dataset/'integrity.json'),verification=verified,counts={k:len(v) for k,v in sets.items()},families=families,effect_heads=EFFECTS,text_heads=TEXT,preprocessing='Original CAPRI RGB resize height96,widthcap480,bilinear; width-batched edge padding and pixel budget',label_limits='Same 100k scenes. Recipe-derived text presence. Missing legacy text ordinal/width/slant labels masked; multi-row CTC masked, unknown font attributes masked. No invented ordinal labels. Three deterministic reference pages per file.',parts=receipts)
    write(out/'cache.json',protocol);status(out,state='complete',counts=protocol['counts'])

class Cache:
    def __init__(self,root,split):
        self.root=Path(root);self.split=split;self.protocol=read(self.root/'cache.json');self.rows=[];self.parts=[];self.lookup=[]
        for r in self.protocol['parts']:
            if not r['path'].startswith(split+'/'):continue
            p=self.root/r['path'];z=np.load(p/'index.npz',allow_pickle=False);ix={k:z[k] for k in z.files};z.close()
            meta=read(p/'meta.json');mm=np.memmap(p/'images.u8',np.uint8,'r').reshape(-1,H,3);k=len(self.parts)
            self.parts.append((ix,mm));self.rows.extend(meta['rows']);self.lookup.extend((k,i) for i in range(len(ix['widths'])))
        self.widths=np.array([self.parts[p][0]['widths'][i] for p,i in self.lookup]);self.family_ids={s:i for i,s in enumerate(self.protocol['families'])}
        assert len(self.rows)==self.protocol['counts'][split]
    def __len__(self):return len(self.rows)
    def image(self,j):
        p,i=self.lookup[j];ix,mm=self.parts[p];o,w=int(ix['offsets'][i]),int(ix['widths'][i]);return np.asarray(mm[o:o+w]).transpose(1,0,2)
    def get(self,key,indices):return np.stack([self.parts[self.lookup[j][0]][0][key][self.lookup[j][1]] for j in indices])
    def collate(self,indices):
        widths=self.widths[indices];cols=(widths+7)//8;wp=int((max(cols.max()*8,widths.max())+MARGIN+7)//8*8)
        images=np.empty((len(indices),3,H,wp),np.uint8)
        for k,j in enumerate(indices):
            im=self.image(j);images[k]=np.pad(im,((0,0),(0,wp-im.shape[1]),(0,0)),mode='edge').transpose(2,0,1)
        return dict(image=images,cols=cols.astype(np.int64),widths=widths.astype(np.int64),indices=np.array(indices,np.int64),**({k:self.get(k,indices) for k in LABELS} if self.split in ('train','dev') else {}))
    def batches(self,epoch,batch=128,budget=BUDGET,shuffle=True):
        rng=np.random.default_rng(epoch);order=np.arange(len(self));
        if shuffle:rng.shuffle(order)
        result=[]
        for start in range(0,len(order),batch*32):
            part=sorted(order[start:start+batch*32],key=lambda i:self.widths[i]);cur=[]
            for j in part:
                w=int(self.widths[j])+MARGIN
                if cur and (len(cur)>=batch or (len(cur)+1)*w>budget):result.append(cur);cur=[]
                cur.append(int(j))
            if cur:result.append(cur)
        if shuffle:rng.shuffle(result)
        assert sorted(i for b in result for i in b)==list(range(len(self)))
        return result

def batches_prefetch(ds,batches):
    # Bounded queue: large memmaps are never copied into child processes.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending=collections.deque();it=iter(batches)
        for _ in range(8):
            x=next(it,None)
            if x is not None:pending.append(pool.submit(ds.collate,x))
        while pending:
            yield pending.popleft().result()
            x=next(it,None)
            if x is not None:pending.append(pool.submit(ds.collate,x))
