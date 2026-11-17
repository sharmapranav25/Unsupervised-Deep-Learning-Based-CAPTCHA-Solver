
"""Expand a verified 20k package to exactly 100k TRAIN images; no training."""
import collections, concurrent.futures, hashlib, json, os, shutil, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
from . import architecture_data as D

_FONTS=None

def select_indices(total, families, count=100000, seed=918):
    D.guard(total%2==0 and count%2==0 and 0<count<=total and families>0,'Invalid paired selection sizes')
    buckets=collections.defaultdict(list)
    for group in range(total//2):
        rank=hashlib.sha256(f'{seed}/train/{group}'.encode()).hexdigest()
        buckets[((group//7)%families,group%7)].append((rank,group*2,group*2+1))
    for values in buckets.values():values.sort()
    order=sorted(buckets,key=lambda k:hashlib.sha256(f'{seed}/{k}'.encode()).hexdigest())
    selected=[];cycle=0
    while len(selected)<count:
        added=0
        for key in order:
            if len(selected)==count:break
            if cycle<len(buckets[key]):selected.extend(buckets[key][cycle][1:]);added+=2
        D.guard(added>0,'Selection exhausted');cycle+=1
    return sorted(selected)

def _init(fonts):
    global _FONTS
    _FONTS=fonts

def _shard(task):
    source=Path(task['source']);out=Path(task['out'])
    D.guard(D.sha(source)==task['sha256'],'Source archive changed: '+str(source))
    records=[];dist={k:collections.Counter() for k in ('count','family','font_file','layout','pair_kind','multicolour','effect','effect_level')}
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(out,'x',compression=zipfile.ZIP_STORED) as target:
        for index,r in task['rows']:
            stem=r['stem'];D.guard(int(stem.rsplit('_',1)[1])==index,'Source order mismatch')
            payload,v,rgb=D.scene(original,stem,r['pixel_sha256'],'train',_FONTS)
            for ext,data in payload.items():target.writestr(stem+ext,data)
            recipe=v['recipe'];families=sorted({_FONTS[g['font_key']]['split_group'] for g in v['glyphs']});keys=sorted({g['font_key'] for g in v['glyphs']})
            records.append(dict(id=stem,archive='train_extra/'+out.name,stem=stem,pixel_sha256=r['pixel_sha256'],count=v['count'],native_hw=list(rgb.shape[:2]),split='train',source_index=index,pair_group=recipe['pair_group'],pair_kind=recipe['pair_kind'],font_keys=keys,family_groups=families))
            for key,val in [('count',str(v['count'])),('layout',recipe['text']['layout']),('pair_kind',str(recipe['pair_kind'])),('multicolour',str(recipe['requested_multicolour']))]:dist[key][val]+=1
            dist['family'].update(families);dist['font_file'].update(keys)
            for key,e in v['effects'].items():
                if e['present']:dist['effect'][key]+=1;dist['effect_level'][f'{key}/{e["level"]}']+=1
    return dict(rows=records,distributions={k:dict(v) for k,v in dist.items()},source=dict(path=task['source_relative'],sha256=task['sha256'],bytes=source.stat().st_size),output=dict(path='train_extra/'+out.name,sha256=D.sha(out),bytes=out.stat().st_size))

def prepare(source,base,output,workers=4):
    source,base,output=[Path(p).resolve() for p in (source,base,output)]
    D.guard(all(p.drive.upper()=='E:' for p in (source,base,output)),'All work must stay on E:')
    D.guard(1<=workers<=8,'Workers must be 1..8')
    D.guard(not output.exists(),'Destination exists; use a fresh directory')
    output.mkdir(parents=True);start=time.monotonic()
    def progress(stage,**kw):
        v=dict(stage=stage,elapsed_seconds=round(time.monotonic()-start,2),**kw);D.write(output/'status.json',v);print(json.dumps(v),flush=True)
    try:
        progress('verify_existing_20k_package')
        base_check=D.verify(base);base_integrity=D.read(base/'integrity.json');base_spec=D.read(base/'protocol.json');base_audit=D.read(base/'audit.json')
        D.guard(D.read(base/'status.json')['stage']=='complete' and base_spec['sizes']=={'train':20000,'dev':2000},'Expected completed20k/2k package')
        source_protocol=D.read(source/'corpus_protocol.json');ph=D.jhash(source_protocol)
        D.guard(ph==base_spec['source_protocol_sha256'],'Wrong source corpus')
        completion=D.read(source/'corpus/completion.json');D.guard(completion['state']=='complete' and completion['accepted_unique']==500000 and completion['protocol_hash']==ph,'Source incomplete')
        D.guard(D.sha(source/'fonts/font_manifest.json')==base_spec['source_font_manifest_sha256'],'Source font manifest changed')
        fonts={f['key']:f for f in D.read(base/'fonts.json')['fonts']}
        families=len({f['family_group'] for f in fonts.values() if f['split']=='train'})
        indices=select_indices(source_protocol['split_sizes']['train'],families)
        old=D.read_rows(base/'train.jsonl');old_indices={r['source_index'] for r in old};selected=set(indices)
        D.guard(old_indices<=selected and len(old_indices)==20000,'New selection must extend prior20k')
        D.guard(set(select_indices(source_protocol['split_sizes']['train'],families,20000))==old_indices,'Prior selection algorithm changed')
        extra=selected-old_indices;D.guard(len(extra)==80000,'Exactly80k new TRAIN scenes required')
        D.guard(all((i^1) in selected for i in selected),'Paired selection broken')
        spec=dict(base_spec);spec.update(created_at=datetime.now(timezone.utc).isoformat(),sizes={'train':100000,'dev':2000},
            authorization='User explicitly requested100,000 training images, matching prior CAPRI training size. Development and font references are additional.',
            subset_policy='Deterministic extension of seed918 original family/count-anchor round-robin paired selection from450k TRAIN; prior20k is a subset. Same previously used2k DEV, never optimization.',
            seed=918,workers=workers,base_integrity_sha256=D.sha(base/'integrity.json'),training_started=False,
            runtime_note='The earlier20k eight-hour budget is not an ETA for this100k dataset. All100k selected TRAIN examples must be eligible for training; no silent reduction.')
        D.write(output/'protocol.json',spec)
        D.write(output/'selection_100k.json',dict(images=100000,indices=indices,index_sha256=D.jhash(indices),seed=918,original_anchor_family_count=families,counts='2..8; actual label counts audited',old_20k_included=True,policy=spec['subset_policy']))
        progress('copy_verified_existing_data',training_images=20000,extra_training_images=80000)
        skip={'protocol.json','README.md','audit.json','train.jsonl','source_archives.json'}
        for rel,r in base_integrity['artifacts'].items():
            if rel in skip:continue
            dst=D.local(output,rel);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(D.local(base,rel),dst)
            D.guard(dst.stat().st_size==r['bytes'] and D.sha(dst)==r['sha256'],'Copied base artifact changed: '+rel)
        prov=output/'provenance/expansion_100k';prov.mkdir(parents=True)
        for original,name in [(base/'integrity.json','base_integrity.json'),(base/'audit.json','base_audit.json'),(base/'protocol.json','base_protocol.json'),(Path(__file__),'expansion_source.py'),(Path(D.__file__),'data_helpers_source.py')]:shutil.copy2(original,prov/name)
        D.write(prov/'base_verification.json',base_check)
        (output/'train_extra').mkdir();tasks=[];cursor=0
        for rp in sorted((source/'corpus/train').glob('*.receipt.json')):
            rec=D.read(rp);first=cursor;cursor+=len(rec['images'])
            D.guard(rec['protocol_hash']==ph and rec['split']=='train','Receipt identity changed')
            chosen=[(first+j,r) for j,r in enumerate(rec['images']) if first+j in extra]
            if not chosen:continue
            archive=rp.with_name(rp.name.replace('.receipt.json','.zip'));out=output/'train_extra'/archive.name
            tasks.append(dict(source=str(archive),out=str(out),sha256=rec['sha256'],source_relative=archive.relative_to(source).as_posix(),rows=chosen))
        D.guard(cursor==450000 and sum(len(t['rows']) for t in tasks)==80000,'Source TRAIN selection incomplete')
        stats={k:collections.Counter(v) for k,v in base_audit['splits']['train']['distributions'].items()}
        records=list(old);source_archives=[];extra_receipts=[];last_update=time.monotonic()
        # CPU processes only; children inherit E-drive temporary/cache destinations and -B.
        temp=output/'temporary';temp.mkdir();os.environ['TEMP']=str(temp);os.environ['TMP']=str(temp)
        os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
        progress('validate_and_copy_additional_train',images=20000,target=100000,workers=workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=_init,initargs=(fonts,)) as pool:
            for result in pool.map(_shard,tasks,chunksize=1):
                records.extend(result['rows']);source_archives.append(result['source']);extra_receipts.append(result['output'])
                for key,values in result['distributions'].items():stats[key].update(values)
                if time.monotonic()-last_update>=20:
                    progress('validate_and_copy_additional_train',images=len(records),target=100000,shards=len(extra_receipts));last_update=time.monotonic()
        records.sort(key=lambda r:r['source_index'])
        D.guard(len(records)==100000 and [r['source_index'] for r in records]==indices,'Wrong final training size/order')
        pixels={r['pixel_sha256'] for r in records};D.guard(len(pixels)==100000,'Duplicate TRAIN pixels')
        dev=D.read_rows(output/'dev.jsonl');D.guard(len(dev)==2000 and not pixels&{r['pixel_sha256'] for r in dev},'Train/DEV leakage')
        devrefs=D.read_rows(output/'references_dev.jsonl');D.guard(not pixels&{r['pixel_sha256'] for r in devrefs},'TRAIN scene/DEV library exact overlap')
        references=D.read(output/'reference_families_train.json');required={f for r in records for f in r['family_groups']}
        D.guard(required<=set(references),'Missing reference support for training families')
        D.guard(set(stats['count'])=={str(i) for i in range(2,9)} and sum(stats['count'].values())==100000,'Count classes/total mismatch')
        D.jsonl(output/'train.jsonl',records)
        audit=dict(base_audit);audit['splits']=dict(base_audit['splits'])
        audit['splits']['train']=dict(images=100000,unique_pixels=100000,family_groups=len(required),
            distributions={k:dict(sorted(v.items())) for k,v in stats.items()},all_pixels_masks_boxes_backgrounds_checked=True,
            validation='20k from hash-verified prior package with byte-to-source audits; additional80k fully pixel/count/box/mask/background/font checked while copying from hash-verified source archives.',
            pairs_preserved=True,old_20k_included=True,all_scene_families_have_reference_support=True)
        audit['training_size_authorization']='Exactly100k training CAPTCHA images, excluding2k DEV and auxiliary font specimens.'
        audit['reserved_archives_opened']=0;D.write(output/'audit.json',audit)
        D.write(output/'source_archives.json',dict(additional_80k_source_archives=source_archives,prepared_extra_archive_receipts=extra_receipts,base_package_integrity_sha256=spec['base_integrity_sha256']))
        # Existing first14 examples remain a fixed preview, explicitly a subset of this100k.
        readme=f"""# CAPRI:100,000 training images

This is the user's requested training set for shared CNN+DINO visual features, dedicated background/appearance/localization/font heads and an independent count transformer.

- **100,000 unique training CAPTCHA images** (not100k total including validation).
- **2,000 separate development images**, identical to the earlier package for comparison.
-36,492 auxiliary TRAIN font specimens and3,060 DEV selection-gallery specimens; these are additional to the100k CAPTCHA images.
- Original10k calibration/20k final-test archives remain reserved and were not opened.

Selection is a deterministic seed918 family/count-anchor paired sample from the450k source TRAIN split, extending the prior20k selection. It contains all the prior20k plus80k more; original source and the earlier package remain unchanged. Actual count/font/effect/layout coverage is in audit.json. No exact joint balance is claimed. DEV has been used previously and is not an untouched benchmark.

## Use

Runtime archive paths are package-relative, so the folder can be copied. Use capri_controller.architecture_data.PreparedScenes(root, 'train', 'count') for RGB plus separate count targets, or mode='scene' for native-resolution masks/background/annotations. Inputs contain only RGB; derive padding validity from pixels and transform localization targets consistently in the trainer. The count transformer's resolution is still an implementation choice.

PreparedReferences(root,'train','optimization') supplies auxiliary font support; DEV requires purpose='selection_library' and is excluded from gradient training. Original font binaries remain external; stored rendered references are sufficient for the prepared learning data. Historical provenance may contain old paths but is not executed. Call close() on readers.

## Known limits

57 DEV reference images are identical to173 TRAIN reference specimens across different named font families. Preserve reference_aliases.json and reader ambiguity flags; exclude duplicate positives/contradictory negatives, retain candidate-family ties and report ambiguous retrieval separately. Reference libraries are not completely pixel-disjoint; CAPTCHA scenes and full font/outline groups retain their split boundaries.

Mask unknown font attributes and conditional effect intensity for absent effects. Background targets exclude final/over-layer effects. Strong blur, overlap and decoys remain; structural validity does not guarantee human readability/count certainty. train_preview.png shows the same fixed first14 prior-subset examples, not a comprehensive review of all100k.

## Reproduce / verify

From E:/pranav/CAPRI using the configured Python with -B:

    python -B capri.py prepare-architecture-100k --source runs/capri_decoder_500k_20261007 --base runs/capri_shared_count_dataset_20261027 --out runs/<fresh-output> --workers4
    python -B capri.py verify-architecture-data --dataset runs/{output.name}

Use `--workers 4` (space between flag and value) when running the preparation command. Existing output directories are refused. integrity.json hashes prepared artifacts; selection_100k.json freezes sample identities; provenance/expansion_100k preserves implementation and verified base provenance.

Training has not started. The earlier eight-hour proposal used20k images and is not a measured100k runtime promise. A full training epoch must cover the entire100k set; a timed partial pass must be reported as partial, never as a full epoch or a secretly reduced training set.
"""
        readme=readme.replace('--workers4','--workers 4').replace('Use `--workers 4` (space between flag and value) when running the preparation command. ','')
        (output/'README.md').write_text(readme,encoding='utf8')
        progress('hash_and_verify_final_package',training_images=100000,development_images=2000)
        artifacts={p.relative_to(output).as_posix():dict(sha256=D.sha(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='status.json'}
        D.write(output/'integrity.json',dict(schema=D.SCHEMA,artifacts=artifacts));checks=D.verify(output)
        progress('complete',training_images=100000,development_images=2000,reference_images=39552,artifact_bytes=sum(v['bytes'] for v in artifacts.values()),verification=checks)
        return D.read(output/'status.json')
    except Exception as exc:progress('failed',error=repr(exc));raise
