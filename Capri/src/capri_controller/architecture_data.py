
"""Prepare portable shared-CAPRI + independent-count data; no training."""
import collections, hashlib, io, json, shutil, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
SCHEMA='capri-shared-plus-count-data-v1'
SUFFIXES=('.png','.background.png','.masks.npz','.json')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def jhash(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def guard(ok,msg):
    if not ok:raise ValueError(msg)
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def jsonl(p,rows):
    with Path(p).open('w',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def read_rows(p):
    with Path(p).open(encoding='utf8') as f:return [json.loads(line) for line in f if line.strip()]
def local(root,rel):
    p=(Path(root)/rel).resolve();guard(p.is_relative_to(Path(root).resolve()),'Path escapes package');return p
def pixels(data,expected):
    a=np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
    guard(hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()==expected,'Pixel hash mismatch')
    return a

def scene(z,stem,pixel_hash,split,fonts):
    payload={ext:z.read(stem+ext) for ext in SUFFIXES}
    v=json.loads(payload['.json']);rgb=pixels(payload['.png'],pixel_hash);h,w=rgb.shape[:2];n=v['count']
    guard(v['pixel_sha256']==pixel_hash and 2<=n<=8 and len(v['glyphs'])==n,'Count/image identity mismatch')
    guard(v['recipe']['split']==split and v['recipe']['requested_count']==n,'Scene split/count mismatch')
    for g in v['glyphs']:
        f=fonts[g['font_key']];guard(f['split']==split and f['sha256']==g['font_sha256'],'Font identity/split mismatch')
        for field in ('box_core','box_visible','box_decorated'):
            x0,y0,x1,y1=g[field];guard(0<=x0<x1<=w and 0<=y0<y1<=h,'Invalid glyph box')
    bg=np.asarray(Image.open(io.BytesIO(payload['.background.png'])).convert('RGB'))
    guard(bg.shape==rgb.shape,'Background shape mismatch')
    with np.load(io.BytesIO(payload['.masks.npz']),allow_pickle=False) as m:
        for key in ('visible','geometry','decoration'):guard(m[key].shape==(n,h,w) and m[key].dtype==np.uint8,'Mask mismatch')
    guard(len(v['effects'])==37,'Effect inventory mismatch')
    for e in v['effects'].values():guard(e['present'] in (0,1) and np.isfinite(e['measured']),'Invalid effect label')
    return payload,v,rgb

class _Reader:
    def init(self,root,manifest):
        self.root=Path(root).resolve();self.rows=read_rows(self.root/manifest);self.handles=collections.OrderedDict()
    def __len__(self):return len(self.rows)
    def archive(self,rel):
        if rel not in self.handles:self.handles[rel]=zipfile.ZipFile(local(self.root,rel))
        self.handles.move_to_end(rel)
        if len(self.handles)>4:self.handles.popitem(last=False)[1].close()
        return self.handles[rel]
    def close(self):
        for z in self.handles.values():z.close()
        self.handles.clear()
    def __getstate__(self):
        s=self.__dict__.copy();s['handles']=collections.OrderedDict();return s

class PreparedScenes(_Reader):
    def __init__(self,root,split='train',mode='count'):
        guard(split in ('train','dev') and mode in ('count','scene'),'Unsupported split/mode')
        self.init(root,split+'.jsonl');self.mode=mode
    def __getitem__(self,i):
        r=self.rows[i];z=self.archive(r['archive']);rgb=pixels(z.read(r['stem']+'.png'),r['pixel_sha256']);target={'count':r['count']}
        if self.mode=='scene':
            target['annotation']=json.loads(z.read(r['stem']+'.json'))
            target['background_rgb']=np.asarray(Image.open(io.BytesIO(z.read(r['stem']+'.background.png'))).convert('RGB'))
            with np.load(io.BytesIO(z.read(r['stem']+'.masks.npz')),allow_pickle=False) as m:target['masks']={k:m[k].copy() for k in m.files}
        return dict(inputs={'rgb':rgb},targets=target,audit={'id':r['id'],'pixel_sha256':r['pixel_sha256']})

class PreparedReferences(_Reader):
    def __init__(self,root,split='train',purpose='optimization'):
        guard((split,purpose) in (('train','optimization'),('dev','selection_library')),'Held-out references cannot train')
        self.init(root,'references_'+split+'.jsonl')
    def __getitem__(self,i):
        r=self.rows[i];z=self.archive(r['archive'])
        return dict(inputs={'rgb':pixels(z.read(r['stem']+'.png'),r['pixel_sha256'])},
                    library_labels=json.loads(z.read(r['stem']+'.json')),reference_ambiguity={'family_groups':r.get('pixel_alias_family_groups',[r['family_group']]),'cross_split':r.get('cross_split_pixel_alias',False)},audit={'id':r['id'],'pixel_sha256':r['pixel_sha256']})

def verify(root):
    root=Path(root).resolve();integrity=read(root/'integrity.json');guard(integrity['schema']==SCHEMA,'Wrong schema')
    for rel,r in integrity['artifacts'].items():
        p=local(root,rel);guard(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Changed artifact: '+rel)
    for split in ('train','dev'):
        data=PreparedScenes(root,split,'count')
        for i in sorted({0,len(data)//2,len(data)-1}):
            x=data[i];guard(set(x['inputs'])=={'rgb'} and set(x['targets'])=={'count'},'Count input/target leakage')
        data.close();data=PreparedScenes(root,split,'scene');x=data[0]
        guard(x['targets']['annotation']['count']==x['targets']['count'],'Scene loader mismatch');data.close()
        data=PreparedReferences(root,split,'optimization' if split=='train' else 'selection_library')
        guard(set(data[0]['inputs'])=={'rgb'},'Reference input leakage');data.close()
    try:PreparedReferences(root,'dev','optimization')
    except ValueError:pass
    else:raise ValueError('DEV references admitted to optimization')
    return dict(artifacts_hashed=len(integrity['artifacts']),count_scene_reference_readers_checked=True,dev_training_refused=True)

def prepare(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    guard(source.drive.upper()=='E:' and output.drive.upper()=='E:','All preparation must use E:')
    guard(not output.exists(),'Destination exists; use a fresh run directory');output.mkdir(parents=True)
    start=time.monotonic()
    def progress(stage,**kw):
        value=dict(stage=stage,elapsed_seconds=round(time.monotonic()-start,2),**kw);write(output/'status.json',value);print(json.dumps(value),flush=True)
    try:
        progress('validate_source')
        protocol=read(source/'corpus_protocol.json');ph=jhash(protocol);done=read(source/'corpus/completion.json')
        guard(done['state']=='complete' and done['protocol_hash']==ph and done['accepted_unique']==500000,'Corpus incomplete/changed')
        fp=source/'fonts/font_manifest.json';guard(sha(fp)==protocol['font_manifest_sha256'],'Font manifest changed')
        fm=read(fp);fonts={f['key']:f for f in fm['fonts']}
        groups={s:{f['split_group'] for f in fonts.values() if f['split']==s} for s in ('train','dev','calibration','test')}
        for a in groups:
            for b in groups:
                if a!=b:guard(not groups[a]&groups[b],'Cross-split font/outline-alias group')
        spec=dict(schema=SCHEMA,created_at=datetime.now(timezone.utc).isoformat(),architecture='shared CNN+DINO features, dedicated background/appearance/localization/font heads, independent count transformer',
                  source_protocol_sha256=ph,source_font_manifest_sha256=sha(fp),source_workspace_relative='runs/capri_decoder_500k_20261007',
                  sizes={'train':20000,'dev':2000},count_classes=list(range(2,9)),training_started=False,
                  subset_policy='Exact frozen paired pilot indices from 20261008. DEV is previously used, not untouched.',
                  input_contract='RGB pixels only; validity derives from padding. Synthetic labels are separate targets.',
                  reserved={'calibration':10000,'test':20000,'policy':'No reserved archive/label access; selection must freeze before calibration, calibration before test.'},
                  background_semantics='Underlying background after under-layer effects/warps; excludes over-layer/final photometric effects.',
                  reference_policy='TRAIN only for optimization; DEV references only for selection gallery. Pixel-duplicate/alias exclusions required.',
                  exact_joint_balance=False)
        write(output/'protocol.json',spec);prov=output/'provenance';prov.mkdir()
        for original,name in [(source/'corpus_protocol.json','original_corpus_protocol.json'),(fp,'original_font_manifest.json'),(Path(__file__),'preparation_source.py')]:shutil.copy2(original,prov/name)
        write(output/'fonts.json',dict(source_sha256=sha(fp),fonts=[{k:v for k,v in f.items() if k not in ('path','provenance')} for f in fonts.values()],note='Training uses saved pixels; font binaries remain external. Original provenance retained separately.'))
        inventories=[];manifests={};audits={};scene_pixels={}
        for split,total in spec['sizes'].items():
            sp=source/f'diagnostics/timed_pilot_20261008/{split}_subset.json';selection=read(sp);indices=selection['indices'];selected=set(indices)
            guard(len(indices)==len(selected)==total==selection['images'] and indices==sorted(indices),'Invalid fixed selection')
            guard(all((i^1) in selected for i in indices),'Broken paired selection');shutil.copy2(sp,prov/f'{split}_selection.json')
            dest=output/split;dest.mkdir();records=[];seen=set();preview=[];cursor=0;dst=None
            dist={k:collections.Counter() for k in ('count','family','font_file','layout','pair_kind','multicolour','effect','effect_level')}
            try:
                for rp in sorted((source/'corpus'/split).glob('*.receipt.json')):
                    rec=read(rp);first=cursor;cursor+=len(rec['images'])
                    guard(rec['split']==split and rec['protocol_hash']==ph,'Receipt identity mismatch')
                    chosen=[(first+j,r) for j,r in enumerate(rec['images']) if first+j in selected]
                    if not chosen:continue
                    archive=rp.with_name(rp.name.replace('.receipt.json','.zip'));guard(sha(archive)==rec['sha256'],'Source archive hash mismatch')
                    inventories.append(dict(role='scene_'+split,path=archive.relative_to(source).as_posix(),sha256=rec['sha256'],bytes=archive.stat().st_size))
                    with zipfile.ZipFile(archive) as z:
                        for index,r in chosen:
                            stem=r['stem'];guard(int(stem.rsplit('_',1)[1])==index,'Subset source order changed')
                            payload,v,rgb=scene(z,stem,r['pixel_sha256'],split,fonts);guard(r['pixel_sha256'] not in seen,'Duplicate scene pixels');seen.add(r['pixel_sha256'])
                            if len(records)%500==0:
                                if dst is not None:dst.close()
                                dp=dest/f'shard_{len(records)//500:04d}.zip';dst=zipfile.ZipFile(dp,'x',compression=zipfile.ZIP_STORED)
                            for ext,data in payload.items():dst.writestr(stem+ext,data)
                            recipe=v['recipe'];families=sorted({fonts[g['font_key']]['split_group'] for g in v['glyphs']});keys=sorted({g['font_key'] for g in v['glyphs']})
                            records.append(dict(id=stem,archive=dp.relative_to(output).as_posix(),stem=stem,pixel_sha256=r['pixel_sha256'],count=v['count'],native_hw=list(rgb.shape[:2]),split=split,source_index=index,pair_group=recipe['pair_group'],pair_kind=recipe['pair_kind'],font_keys=keys,family_groups=families))
                            for key,value in [('count',str(v['count'])),('layout',recipe['text']['layout']),('pair_kind',str(recipe['pair_kind'])),('multicolour',str(recipe['requested_multicolour']))]:dist[key][value]+=1
                            dist['family'].update(families);dist['font_file'].update(keys)
                            for key,e in v['effects'].items():
                                if e['present']:dist['effect'][key]+=1;dist['effect_level'][f'{key}/{e["level"]}']+=1
                            if len(preview)<14:preview.append((rgb.copy(),v,stem))
                    if len(inventories)%25==0:progress('repack_'+split,images=len(records),target=total)
            finally:
                if dst is not None:dst.close()
            guard(cursor==protocol['split_sizes'][split] and len(records)==total,'Incomplete source/subset')
            jsonl(output/f'{split}.jsonl',records);manifests[split]=records;scene_pixels[split]=seen
            audits[split]=dict(images=len(records),unique_pixels=len(seen),all_selected_pixels_masks_boxes_backgrounds_checked=True,subset_source_sha256=sha(sp),family_groups=len(dist['family']),distributions={k:dict(sorted(v.items())) for k,v in dist.items()})
            sheet=Image.new('RGB',(1000,1190),'#eeeeee');draw=ImageDraw.Draw(sheet)
            for n,(rgb,v,stem) in enumerate(preview):
                img=Image.fromarray(rgb);img.thumbnail((480,130));x=n%2*500;y=n//2*170;sheet.paste(img,(x+8,y+25));draw.text((x+8,y+5),f'{stem} count={v["count"]} {v["recipe"]["text"]["layout"]}',fill='black')
            sheet.save(output/f'{split}_preview.png');progress('repacked_'+split,images=len(records))
        guard(not scene_pixels['train']&scene_pixels['dev'],'TRAIN/DEV scene leakage')
        refpixels={}
        for directory,split,role in [('font_references_train_v1','train','optimization_library'),('font_gallery_dev_v1','dev','selection_library_only')]:
            progress('copy_'+directory);sr=source/directory;rp=read(sr/'protocol.json');rd=read(sr/'completion.json')
            guard(rd['state']=='complete' and rd['protocol_hash']==jhash(rp),'Reference completion mismatch')
            guard(rp['font_manifest_sha256']==spec['source_font_manifest_sha256'],'Reference font manifest mismatch')
            dest=output/('references_'+split);dest.mkdir();records=[];dups=collections.Counter()
            for receipt_path in sorted(sr.glob('shard_*.receipt.json')):
                rec=read(receipt_path);archive=receipt_path.with_name(receipt_path.name.replace('.receipt.json','.zip'))
                guard(rec['first']==len(records) and rec['protocol_hash']==rd['protocol_hash'] and sha(archive)==rec['sha256'],'Reference identity/continuity mismatch')
                dp=dest/archive.name;shutil.copy2(archive,dp);guard(sha(dp)==rec['sha256'],'Reference copy corrupted')
                inventories.append(dict(role=role,path=archive.relative_to(source).as_posix(),sha256=rec['sha256'],bytes=archive.stat().st_size))
                with zipfile.ZipFile(dp) as z:
                    for r in rec['records']:
                        guard(r['accepted'],'Rejected reference');v=json.loads(z.read(r['stem']+'.json'));rgb=pixels(z.read(r['stem']+'.png'),r['pixel_sha256'])
                        guard(v['recipe']['split']==split and v['recipe']['specimen_id']==r['specimen_id'] and v['count']==len(v['glyphs']),'Reference label mismatch')
                        keys={g['font_key'] for g in v['glyphs']};guard(len(keys)==1,'Multi-font reference');font=fonts[next(iter(keys))]
                        guard(font['split']==split and all(g['font_sha256']==font['sha256'] for g in v['glyphs']),'Reference font leakage')
                        records.append(dict(id=r['stem'],stem=r['stem'],archive=dp.relative_to(output).as_posix(),pixel_sha256=r['pixel_sha256'],font_key=font['key'],family_group=font['split_group'],split=split,role=role,specimen_id=r['specimen_id'],native_hw=list(rgb.shape[:2]),axes=v['glyphs'][0]['axes']))
                        dups[r['pixel_sha256']]+=1
            guard(len(records)==rd.get('accepted_references',rd.get('accepted')),'Reference count mismatch')
            jsonl(output/f'references_{split}.jsonl',records);family_index=collections.defaultdict(list)
            for i,r in enumerate(records):family_index[r['family_group']].append(i)
            write(output/f'reference_families_{split}.json',dict(family_index))
            required={f for r in manifests[split] for f in r['family_groups']};guard(required<=set(family_index),'Missing scene-family reference support')
            audits['references_'+split]=dict(images=len(records),families=len(family_index),unique_pixels=len(dups),duplicate_images=sum(n-1 for n in dups.values()),all_scene_families_covered=True,all_reference_pixels_verified=True,role=role)
            refpixels[split]=set(dups)
            shutil.copy2(sr/'protocol.json',prov/f'references_{split}_protocol.json');shutil.copy2(sr/'completion.json',prov/f'references_{split}_completion.json')
        alias_report=annotate_reference_aliases(output)
        guard(not refpixels['train']&scene_pixels['dev'] and not refpixels['dev']&scene_pixels['train'],'Scene/reference split leakage')
        write(output/'source_archives.json',inventories)
        write(output/'audit.json',dict(schema=SCHEMA,splits=audits,font_groups_disjoint_all_four_splits=True,train_dev_scene_pixels_disjoint=True,reference_pixels_disjoint=not bool(alias_report['cross_split_groups']),reference_aliases=alias_report,reserved_archives_opened=0,
              limitations=['DEV already used in prior pilot; not untouched.','Duplicate references retained; exclude trivial positives and contradictory cross-family negatives.','Background excludes final/over-layer effects.','Mask absent intensity and unknown font attributes.','No predicted-region feature cache or trained count model.','No exact joint balancing or measured training speed/quality claimed.']))
        write(output/'reserved_splits.json',dict(source_workspace_relative=spec['source_workspace_relative'],font_manifest_sha256=spec['source_font_manifest_sha256'],splits=spec['reserved']))
        docs=f"""# CAPRI shared-feature + independent-count dataset

Prepared from frozen synthetic data; no generation, training or real labels.

- 20,000 TRAIN and 2,000 DEV scenes, exact paired subsets from the previous pilot.
- {audits['references_train']['images']:,} TRAIN font references; {audits['references_dev']['images']:,} DEV selection-gallery specimens.
- Archive and runtime manifest paths are package-relative: copy this folder independently for learning. Historical provenance retains original paths but is never executed.
- Font binaries and reserved10k calibration/20k final-test images remain in the original workspace, outside this initial package.

## Supervision and readers

`train.jsonl`/`dev.jsonl`: counts, image hashes, family groups and paired units. Scene shards contain unchanged RGB, glyph masks, underlying background RGB and original synthetic JSON annotations.

`capri_controller.architecture_data.PreparedScenes(root, split, mode='count')` reads only RGB plus separate count targets. Mode `scene` additionally returns native-resolution masks/background/annotations as targets. Inputs contain only RGB. Trainers must apply image-only preprocessing, derive padding validity from pixels and transform localization targets consistently. Count-transformer resolution is not chosen by this package. Call close() on readers when finished.

`PreparedReferences(root, 'train', 'optimization')` is for font training. DEV requires purpose='selection_library' and may never provide gradients. References expose pixels as inputs and annotations separately for association after predicted-region extraction. Never force the known specimen count or use ground-truth boxes as target model inputs. Family-balanced sampling can use `reference_families_*.json`. Pixel-hash duplicates/aliases must be excluded as trivial positives or contradictory negatives. Cross-split reference specimens can be pixel-identical for different font families; reference_aliases.json records this ambiguity. Preserve candidate-family ties; report ambiguous retrieval cases separately. Family groups and scene images remain split-disjoint.

Mask unknown font attributes and conditional intensity when effects are absent. Background target means underlying rendered background after under-layer effects/warps, excluding final/over-layer effects. It is not the exact final observed background. Text colour and layout remain in original annotations.

## Reproduce and verify

From the E-drive workspace with the Python executable in configs/local.json:

    python -B capri.py prepare-architecture-data --source runs/capri_decoder_500k_20261007 --out runs/<fresh-output>
    python -B capri.py verify-architecture-data --dataset runs/{output.name}

Existing destinations are refused. `integrity.json` hashes prepared artifacts except operational status/integrity itself. `provenance/preparation_source.py` preserves the preparation implementation. `audit.json` contains distributions/coverage/limitations. Calibration/test access remains subject to model-selection and calibration freezes. No training is started; the eight-hour training proposal is separate from preparation.
"""
        (output/'README.md').write_text(docs,encoding='utf8')
        artifacts={p.relative_to(output).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='status.json'}
        write(output/'integrity.json',dict(schema=SCHEMA,artifacts=artifacts));checked=verify(output)
        progress('complete',images=22000,reference_images=sum(audits['references_'+s]['images'] for s in ('train','dev')),artifact_bytes=sum(r['bytes'] for r in artifacts.values()),verification=checked)
        return read(output/'status.json')
    except Exception as exc:progress('failed',error=repr(exc));raise


def annotate_reference_aliases(root):
    root=Path(root);rows={s:read_rows(root/f'references_{s}.jsonl') for s in ('train','dev')}
    buckets=collections.defaultdict(list)
    for split in rows:
        for r in rows[split]:buckets[r['pixel_sha256']].append(r)
    aliases={};cross=[]
    for digest,items in buckets.items():
        families=sorted({r['family_group'] for r in items});splits=sorted({r['split'] for r in items})
        if len(items)>1:
            aliases[digest]=dict(family_groups=families,splits=splits,specimens=[{'split':r['split'],'id':r['id'],'font_key':r['font_key']} for r in items])
        if len(splits)>1:cross.append(digest)
        for r in items:
            r['pixel_alias_family_groups']=families;r['cross_split_pixel_alias']=len(splits)>1
    for split in rows:jsonl(root/f'references_{split}.jsonl',rows[split])
    result=dict(cross_split_groups=len(cross),cross_split_train_images=sum(r['cross_split_pixel_alias'] for r in rows['train']),
                cross_split_dev_images=sum(r['cross_split_pixel_alias'] for r in rows['dev']),
                policy='Preserve evidence and family identities. Exclude pixel duplicates as trivial positives and cross-family negatives; preserve tied retrieval candidates and report ambiguous cases separately. Do not claim pixel-disjoint reference libraries.')
    write(root/'reference_aliases.json',dict(summary=result,groups=aliases))
    return result


def finalize_existing(source, output):
    """Recover only a fully copied attempt stopped on reference pixel ambiguity.

    Rechecks copied scene payloads against the original archives and all reference
    archive hashes. Never changes archive bytes or source experiments.
    """
    source,output=Path(source).resolve(),Path(output).resolve()
    guard(source.drive.upper()=='E:' and output.drive.upper()=='E:','E-drive only')
    prior=read(output/'status.json')
    guard(prior.get('stage')=='failed' and prior.get('error')=="ValueError('Reference split leakage')",'Only known alias-audit stop may finalize')
    guard(not (output/'integrity.json').exists(),'Already finalized')
    write(output/'provenance/initial_attempt_status.json',prior)
    shutil.copy2(output/'provenance/preparation_source.py',output/'provenance/preparation_source_initial_attempt.py')
    shutil.copy2(Path(__file__),output/'provenance/preparation_source.py')
    start=time.monotonic()
    def progress(stage,**kw):
        v=dict(stage=stage,elapsed_seconds=round(time.monotonic()-start,2),**kw);write(output/'status.json',v);print(json.dumps(v),flush=True)
    try:
        spec=read(output/'protocol.json');original_protocol=read(source/'corpus_protocol.json')
        guard(jhash(original_protocol)==spec['source_protocol_sha256'],'Source protocol changed')
        guard(sha(source/'fonts/font_manifest.json')==spec['source_font_manifest_sha256'],'Source fonts changed')
        fonts={f['key']:f for f in read(output/'fonts.json')['fonts']};source_fonts={f['key']:f for f in read(source/'fonts/font_manifest.json')['fonts']}
        guard(all(f=={k:v for k,v in source_fonts[key].items() if k not in ('path','provenance')} for key,f in fonts.items()) and len(fonts)==len(source_fonts),'Prepared font metadata changed')
        groups={s:{f['split_group'] for f in fonts.values() if f['split']==s} for s in ('train','dev','calibration','test')}
        for a in groups:
            for b in groups:
                if a!=b:guard(not groups[a]&groups[b],'Font group split leakage')
        audits={};inventories=[];seen={}
        for split,total in spec['sizes'].items():
            progress('recheck_'+split)
            rows=read_rows(output/f'{split}.jsonl');selection=read(output/f'provenance/{split}_selection.json')
            guard(sha(output/f'provenance/{split}_selection.json')==sha(source/f'diagnostics/timed_pilot_20261008/{split}_subset.json'),'Selection changed')
            guard([r['source_index'] for r in rows]==selection['indices'] and len(rows)==total,'Wrong prepared subset')
            lookup={};cursor=0
            for rp in sorted((source/'corpus'/split).glob('*.receipt.json')):
                receipt=read(rp);guard(receipt['split']==split and receipt['protocol_hash']==spec['source_protocol_sha256'],'Source receipt mismatch')
                path=rp.with_name(rp.name.replace('.receipt.json','.zip'))
                for r in receipt['images']:lookup[r['stem']]=(path,r['pixel_sha256'],cursor);cursor+=1
            guard(cursor==original_protocol['split_sizes'][split],'Source split incomplete')
            handles=collections.OrderedDict();seen[split]=set();dist={k:collections.Counter() for k in ('count','family','font_file','layout','pair_kind','multicolour','effect','effect_level')}
            def archive(path):
                if path not in handles:handles[path]=zipfile.ZipFile(path)
                handles.move_to_end(path)
                if len(handles)>6:handles.popitem(last=False)[1].close()
                return handles[path]
            try:
                for i,r in enumerate(rows):
                    sp,expected,index=lookup[r['stem']];guard(expected==r['pixel_sha256'] and index==r['source_index'],'Source row mismatch')
                    z=archive(local(output,r['archive']));payload,v,rgb=scene(z,r['stem'],expected,split,fonts)
                    oz=archive(sp)
                    guard(all(data==oz.read(r['stem']+ext) for ext,data in payload.items()),'Prepared bytes differ from original source')
                    guard(r['count']==v['count'] and r['native_hw']==list(rgb.shape[:2]),'Prepared manifest count/shape mismatch')
                    fams=sorted({fonts[g['font_key']]['split_group'] for g in v['glyphs']});keys=sorted({g['font_key'] for g in v['glyphs']})
                    guard(r['family_groups']==fams and r['font_keys']==keys,'Prepared font metadata mismatch')
                    guard(expected not in seen[split],'Duplicate scene pixels');seen[split].add(expected)
                    recipe=v['recipe'];guard(recipe['pair_group']==r['pair_group'] and recipe['pair_kind']==r['pair_kind'],'Pair identity mismatch')
                    for key,value in [('count',str(v['count'])),('layout',recipe['text']['layout']),('pair_kind',str(recipe['pair_kind'])),('multicolour',str(recipe['requested_multicolour']))]:dist[key][value]+=1
                    dist['family'].update(fams);dist['font_file'].update(keys)
                    for key,e in v['effects'].items():
                        if e['present']:dist['effect'][key]+=1;dist['effect_level'][f'{key}/{e["level"]}']+=1
                    if (i+1)%5000==0:progress('recheck_'+split,images=i+1)
            finally:
                for z in handles.values():z.close()
            guard(dict(dist['count'])==selection['distributions']['count'],'Frozen count distribution changed')
            audits[split]=dict(images=len(rows),unique_pixels=len(seen[split]),family_groups=len(dist['family']),all_pixels_masks_boxes_backgrounds_checked=True,all_prepared_payloads_byte_identical_to_source=True,distributions={k:dict(sorted(v.items())) for k,v in dist.items()})
        guard(not seen['train']&seen['dev'],'Scene split leakage')
        refpixels={}
        for directory,split,role in [('font_references_train_v1','train','optimization_library'),('font_gallery_dev_v1','dev','selection_library_only')]:
            progress('recheck_reference_'+split);rows=read_rows(output/f'references_{split}.jsonl');cursor=0
            rp=read(source/directory/'protocol.json');completion=read(source/directory/'completion.json')
            guard(completion['protocol_hash']==jhash(rp) and rp['font_manifest_sha256']==spec['source_font_manifest_sha256'],'Reference source identity changed')
            for receipt_path in sorted((source/directory).glob('*.receipt.json')):
                receipt=read(receipt_path);filename=receipt_path.name.replace('.receipt.json','.zip');p=output/('references_'+split)/filename
                guard(receipt['first']==cursor and receipt['protocol_hash']==completion['protocol_hash'] and sha(p)==receipt['sha256'],'Copied reference archive corrupted')
                inventories.append(dict(role=role,path=(Path(directory)/filename).as_posix(),sha256=receipt['sha256'],bytes=p.stat().st_size))
                with zipfile.ZipFile(p) as z:
                    for expected in receipt['records']:
                        r=rows[cursor];cursor+=1
                        guard(r['stem']==expected['stem'] and r['pixel_sha256']==expected['pixel_sha256'] and r['specimen_id']==expected['specimen_id'],'Reference manifest identity mismatch')
                        guard(r['archive']==p.relative_to(output).as_posix(),'Reference archive locator mismatch')
                        label=json.loads(z.read(r['stem']+'.json'));f=fonts[r['font_key']]
                        guard(f['split']==r['split']==split and f['split_group']==r['family_group'] and label['recipe']['split']==split,'Reference split mismatch')
                        guard(all(g['font_key']==f['key'] and g['font_sha256']==f['sha256'] for g in label['glyphs']),'Reference font labels changed')
            guard(cursor==len(rows)==completion.get('accepted_references',completion.get('accepted')),'Missing reference records')
            counts=collections.Counter(r['pixel_sha256'] for r in rows);refpixels[split]=set(counts);family_index=collections.defaultdict(list)
            for i,r in enumerate(rows):family_index[r['family_group']].append(i)
            required=set(audits[split]['distributions']['family']);guard(required<=set(family_index),'Missing reference family support')
            write(output/f'reference_families_{split}.json',dict(family_index))
            audits['references_'+split]=dict(images=len(rows),families=len(family_index),unique_pixels=len(counts),duplicate_images=sum(n-1 for n in counts.values()),all_scene_families_covered=True,archive_bytes_match_original_receipts=True,pixel_decoding='All reference RGB decoded and verified during initial complete copy; finalization reverified original archive hashes.',role=role)
        guard(not refpixels['train']&seen['dev'] and not refpixels['dev']&seen['train'],'Reference/scene cross-split leakage')
        alias=annotate_reference_aliases(output)
        spec['reference_policy']='All original references retained. Cross-split pixel-identical specimens are explicit ambiguity groups, not claimed independent references. Preserve family ties and exclude contradictory negatives/trivial positives. TRAIN only for optimization; DEV gallery only for selection.'
        write(output/'protocol.json',spec)
        write(output/'audit.json',dict(schema=SCHEMA,splits=audits,font_groups_disjoint_all_four_splits=True,train_dev_scene_pixels_disjoint=True,reference_pixels_disjoint=False,reference_aliases=alias,reserved_archives_opened=0,limitations=['DEV previously used, not untouched.','57 DEV library specimens share pixels with TRAIN references; exact-family retrieval has ambiguous cases.','Highly degraded/overlapping/decoy scenes retained; structural validity does not imply visual count/readability certainty.','Unknown font attributes and absent conditional intensity require loss masks.','Background targets exclude final/over-layer effects.','No predicted-region cache or trained transformer is included.']))
        write(output/'source_archives.json',dict(references=inventories,scenes='Every prepared scene PNG,JSON,mask and background payload independently compared byte-for-byte to original source archives during finalization.'))
        write(output/'reserved_splits.json',dict(source_workspace_relative=spec['source_workspace_relative'],splits=spec['reserved'],font_manifest_sha256=spec['source_font_manifest_sha256']))
        readme=f"""# Shared CAPRI + independent count: prepared learning dataset

20,000 TRAIN and2,000 DEV scenes;36,492 TRAIN font references;3,060 DEV selection-gallery references. All runtime archive locators are relative. This folder can be copied for learning without old C-drive paths. Historical provenance files retain original paths but are never executed. Font binaries and reserved10k calibration/20k final-test images remain outside this package in the original corpus.

## Architecture and data

Shared CNN+DINO features support background/effects, localization/layout and font heads. A separate transformer learns count2–8. Scene shards preserve RGB, native-resolution glyph masks, underlying background and synthetic JSON labels. Inputs to readers contain only RGB; count, fonts, boxes, masks and synthetic text are separate targets, never inference inputs.

`PreparedScenes(root, split='train', mode='count')` in capri_controller.architecture_data reads RGB and a separate count target, skipping masks/font annotations. Mode='scene' loads all targets. Trainers must apply image-only preprocessing and consistently transform localization labels. No count-transformer resolution is selected here. Call close() on readers.

`PreparedReferences(root, 'train', 'optimization')` supplies training font support; DEV requires purpose='selection_library' and must never provide gradients. Family indexes enable balanced reference sampling. Regions must be predicted from pixels; ground-truth library boxes associate/audit predictions afterward, not force known count/crops.

## Font ambiguity discovered during preparation

57 DEV specimens are pixel-identical to173 TRAIN specimens across57 image-hash groups despite disjoint full-font family/outline groups. All are preserved; dropping all affected DEV specimens would eliminate the Bayon family gallery. `reference_aliases.json` lists exact groups; manifests and readers expose family aliases and cross-split flags. Exclude duplicate specimens as trivial positives and same-pixel/different-family contradictory negatives. Preserve tied retrieval candidates and report ambiguous cases separately. Do not claim fully pixel-disjoint reference libraries or perfectly identifiable exact font families. TRAIN/DEV CAPTCHA scene pixels are disjoint, and cross-split scene/reference exact overlap is absent.

Mask unknown font attributes and intensity targets for absent effects. Background targets represent underlying background after under-layer effects/warps, excluding final photometric/over-layer effects. Strong blur, overlap and decoy examples are retained, so structural validity is not a visual readability guarantee. DEV is reused historical synthetic development data, not a new test.

## Verify/reproduce

Use the configured E-drive Python with -B from the shared workspace:

    python -B capri.py prepare-architecture-data --source runs/capri_decoder_500k_20261007 --out runs/<fresh-output>
    python -B capri.py verify-architecture-data --dataset runs/{output.name}

The first preparation stopped on reference ambiguity after copying all data; initial status/source are preserved in provenance. `capri.py finalize-architecture-data --source runs/capri_decoder_500k_20261007 --dataset runs/{output.name}` recovered only that known failed stage, rechecked every scene payload against source and reference archive hashes, then recorded the ambiguity policy. Future fresh preparation records aliases directly. `integrity.json` hashes package artifacts; `audit.json` documents coverage and limits. No training or real-label scoring ran. Calibration/final-test archives were not opened. Training and model-selection decisions remain separate.
"""
        (output/'README.md').write_text(readme,encoding='utf8')
        artifacts={p.relative_to(output).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='status.json'}
        write(output/'integrity.json',dict(schema=SCHEMA,artifacts=artifacts));checked=verify(output)
        progress('complete',images=22000,reference_images=39552,reference_aliases=alias,artifact_bytes=sum(x['bytes'] for x in artifacts.values()),verification=checked)
        return read(output/'status.json')
    except Exception as exc:progress('failed',error=repr(exc));raise
