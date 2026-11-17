"""CryptBB v3 accepted-only corpus: reject invalid recipes, retain 25k images per arm."""
import argparse,copy,collections,hashlib,io,json,os,time,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from .decoder.common import read,write,sha,now,StageLock
ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'handoff/cryptbb_v3_20261114'
FIT=ROOT/'Capri/runs/cryptbb_controller_v3split_full_20261114'
SOURCE=ROOT/'Capri/runs/cryptbb_v3_source_20261114'
ARMS=['custom','dino','mixed','synthtiger'];BASES={'A':50000000000,'B':60000000000}
FORBIDDEN=set()

def forbidden(tag):
    result=set(read(ROOT/'Capri/runs/cryptbb_model1_preparation_20261113/excluded_pixels.json'))
    for role in ['reference_pool','pseudo_label_pool','evaluation_pool']:
        result.update(r['pixel_sha256'] for r in read(ROOT/f'evaluation_data/model4_reservation_20261114/cryptbb/{role}.json'))
    for old in [ROOT/f'datasets/cryptbb/{t}/sample_provenance.jsonl' for t in 'AB']+([ROOT/'datasets/cryptbb_v3/A/sample_provenance.jsonl'] if tag=='B' else []):
        with old.open() as f:
            for line in f:result.add(json.loads(line)['pixel_sha256'])
    return result

def init(tag):
    global FORBIDDEN
    from .controller_v2 import _install
    _install();FORBIDDEN=forbidden(tag)

def generate_shard(task):
    from .controller_v2 import render_job,draw_recipe,usable,MIN_TEXT_CONTRAST
    from PIL import Image
    import numpy as np
    tag,entry,recipe,position,count,out,digest=task
    out=Path(out);name=entry['entry_id'];dest=out/(name+'.zip');receipt=dest.with_suffix('.json')
    if receipt.exists():
        r=read(receipt);assert r['entries_sha256']==digest and r['count']==count and r['seed_base']==BASES[tag] and sha(dest)==r['zip_sha256'];return r
    assert not dest.exists(),f'Uncommitted shard preserved: {dest}'
    assert not entry.get('unvalidated'),f'Unvalidated recipe {name}'
    expected_engine='synthtiger' if entry['controller']=='synthtiger' else 'capri';assert entry['engine']==expected_engine
    labels=[];rejected=[];seen=set();start=time.time();tmp=dest.with_suffix('.pending')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED) as z:
        for j in range(count):
            for attempt in range(100):
                seed=BASES[tag]+position*10000000+j*10000+attempt
                row=render_job((draw_recipe(recipe,seed),entry['controls'],entry['engine']))
                reason=None
                if not usable(row,recipe['description']['length']):reason=row.get('error','completeness/usability/contrast/length')
                if reason is None:
                    h=hashlib.sha256(row['image'].tobytes()).hexdigest()
                    if h in seen or h in FORBIDDEN:reason='duplicate or forbidden pixel hash'
                if reason is None:break
                rejected.append(dict(sample=j,attempt=attempt,seed=seed,reason=reason))
            else:raise RuntimeError(f'{name} sample{j}:100 retries exhausted')
            seen.add(h);assert row['text_contrast']>=MIN_TEXT_CONTRAST
            image=f'{j:05d}.png';buf=io.BytesIO();im=Image.fromarray(row['image']);im.save(buf,format='PNG');z.writestr(image,buf.getvalue())
            labels.append(dict(image=image,text=row['text'],seed=seed,entry_id=name,controller=entry['controller'],engine=entry['engine'],input_index=entry['input_index'],pixel_sha256=h,complete=bool(row['complete']),usable=bool(row['usable']),strict=bool(row['strict']),text_contrast=float(row['text_contrast']),attempts=attempt+1,size=list(im.size),text_spec=row['text_spec'],effects_spec=row['effects_spec']))
        enc=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()
        z.writestr('labels.jsonl',''.join(json.dumps(r,default=enc)+'\n' for r in labels));z.writestr('rejections.jsonl',''.join(json.dumps(r)+'\n' for r in rejected))
    os.replace(tmp,dest)
    r=dict(entry_id=name,controller=entry['controller'],engine=entry['engine'],count=count,entries_sha256=digest,seed_base=BASES[tag],zip_sha256=sha(dest),rejections=len(rejected),seconds=time.time()-start)
    write(receipt,r);return r

def allocation(entries):
    """Equal arm mass; distribute each 25k quota across its admitted recipes by stable ID."""
    counts={}
    for arm in ARMS:
        group=sorted((e for e in entries if e['controller']==arm),key=lambda e:e['entry_id'])
        assert group, f'No admitted recipes for {arm}'
        q,r=divmod(25000,len(group))
        counts.update({e['entry_id']:q+(i<r) for i,e in enumerate(group)})
    assert sum(counts.values())==100000
    return counts

def validate_entries():
    done=read(STUDY/'fit_completion.json');assert sha(FIT/'entries.json')==done['entries_sha256']
    all_entries=read(FIT/'entries.json');assert len(all_entries)==400 and len({e['entry_id'] for e in all_entries})==400
    assert collections.Counter(e['controller'] for e in all_entries)==dict.fromkeys(ARMS,100)
    assert all({e['input_index'] for e in all_entries if e['controller']==arm}==set(range(100)) for arm in ARMS)
    entries=[e for e in all_entries if e.get('unvalidated') is False]
    rejected=[e['entry_id'] for e in all_entries if e.get('unvalidated') is not False]
    assert all(e['validation_score'] is not None for e in entries)
    counts=allocation(entries)
    frozen=read(STUDY/'accepted_recipe_protocol.json')
    assert frozen['fit_sha256']==sha(FIT/'entries.json')
    assert frozen['accepted_ids']==[e['entry_id'] for e in entries] and frozen['rejected_ids']==rejected
    assert frozen['per_entry_counts']==counts
    return entries

def generate(tag,workers=12):
    entries=validate_entries();counts=allocation(entries);recipes=read(SOURCE/'refined_recipes.json');dest=ROOT/f'datasets/cryptbb_v3/{tag}';out=dest/'corpus';out.mkdir(parents=True,exist_ok=True)
    digest=sha(FIT/'entries.json')
    with StageLock(dest,'generation'):
        protocol=dict(created_at=now(),version='controller_v3_split_accepted_recipes',tag=tag,images=100000,entries=len(entries),per_entry_counts=counts,per_arm=25000,rejected_entries=read(STUDY/'accepted_recipe_protocol.json')['rejected_ids'],seed_base=BASES[tag],seed_formula='base + entry_position*10000000 + sample*10000 + retry (0..99)',entries_sha256=digest,recipes_sha256=sha(SOURCE/'refined_recipes.json'),render='controller_v2.render_job including engine switch and contrast40',real_labels_used=False)
        if (dest/'protocol.json').exists():
            old=read(dest/'protocol.json');assert all(old[k]==v for k,v in protocol.items() if k!='created_at')
        else:write(dest/'protocol.json',protocol)
        tasks=[(tag,e,recipes[e['input_index']],i,counts[e['entry_id']],str(out),digest) for i,e in enumerate(entries)]
        results=[];start=time.time()
        with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(tag,)) as pool:
            futures=[pool.submit(generate_shard,t) for t in tasks]
            for f in as_completed(futures):
                results.append(f.result());write(dest/'corpus_status.json',dict(completed_shards=len(results),completed_images=sum(r['count'] for r in results),total=100000,seconds=time.time()-start));print(tag,len(results),f'/{len(entries)} shards',flush=True)
        write(dest/'corpus_summary.json',sorted(results,key=lambda r:r['entry_id']))

def pack(tag):
    import lmdb,numpy as np
    from PIL import Image
    dest=ROOT/f'datasets/cryptbb_v3/{tag}';db=dest/'train_lmdb';receipt=dest/'data_receipt.json'
    if receipt.exists():assert sha(db/'data.mdb')==read(receipt)['lmdb_sha256'];return
    assert not db.exists(),'Incomplete LMDB preserved for inspection'
    charset_path=ROOT/'datasets/cryptbb_v3/charset.txt';charset=set(charset_path.read_text().splitlines());length=read(SOURCE/'length_decision.json')['default_length'];exclude=forbidden(tag)
    entries={e['entry_id']:e for e in validate_entries()};digest=sha(FIT/'entries.json')
    summaries=read(dest/'corpus_summary.json');assert len(summaries)==len(entries);assert {s['entry_id'] for s in summaries}==set(entries);counts_by_entry=allocation(list(entries.values()))
    seen=set();seeds=set();counts=collections.Counter();n=0;db.mkdir();env=lmdb.open(str(db),map_size=8*1024**3)
    with (dest/'sample_provenance.jsonl').open('w',encoding='utf8') as index:
        try:
            for s in summaries:
                p=dest/'corpus'/(s['entry_id']+'.zip');assert sha(p)==s['zip_sha256'] and s['entries_sha256']==digest and s['seed_base']==BASES[tag]
                with zipfile.ZipFile(p) as z,env.begin(write=True) as tx:
                    rows=[json.loads(x) for x in z.read('labels.jsonl').decode().splitlines()];assert len(rows)==counts_by_entry[s['entry_id']] and s['count']==len(rows)
                    assert len([x for x in z.namelist() if x.endswith('.png')])==counts_by_entry[s['entry_id']]
                    for r in rows:
                        b=z.read(r['image']);im=Image.open(io.BytesIO(b)).convert('RGB');h=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                        assert h==r['pixel_sha256'] and h not in seen and h not in exclude and r['seed'] not in seeds
                        assert r['controller']==entries[s['entry_id']]['controller'] and r['engine']==entries[s['entry_id']]['engine']
                        assert r['complete'] and r['usable'] and r['text_contrast']>=40 and len(r['text'])==length and set(r['text'])<=charset
                        assert r['text']==r['text_spec']['text'] and r['size']==list(im.size)
                        n+=1;seen.add(h);seeds.add(r['seed']);counts[r['controller']]+=1
                        tx.put(f'image-{n:09d}'.encode(),b);tx.put(f'label-{n:09d}'.encode(),r['text'].encode());tx.put(f'wh-{n:09d}'.encode(),f'{im.width}_{im.height}'.encode())
                        index.write(json.dumps(dict(index=n,shard=str(p.relative_to(dest)),image=r['image'],text=r['text'],pixel_sha256=h,controller=r['controller'],engine=r['engine'],entry_id=r['entry_id'],input_index=r['input_index'],seed=r['seed']))+'\n')
            assert n==100000 and counts==dict.fromkeys(ARMS,25000)
            with env.begin(write=True) as tx:tx.put(b'num-samples',str(n).encode())
            env.sync()
        finally:env.close()
    with lmdb.open(str(db),readonly=True,lock=False) as check:
        with check.begin() as tx:
            assert int(tx.get(b'num-samples'))==100000
            for i in [1,50000,100000]:assert tx.get(f'image-{i:09d}'.encode()) and tx.get(f'label-{i:09d}'.encode())
    write(receipt,dict(images=n,unique_pixels=len(seen),unique_seeds=len(seeds),controllers=dict(counts),all_images_decoded=True,old_AB_and_real_pixel_overlap=0,new_A_overlap=0 if tag=='B' else None,lmdb_sha256=sha(db/'data.mdb'),provenance_sha256=sha(dest/'sample_provenance.jsonl'),charset_sha256=sha(charset_path)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generate','pack']);ap.add_argument('--dataset',choices=['A','B'],required=True);ap.add_argument('--workers',type=int,default=12);a=ap.parse_args()
    if a.stage=='generate':generate(a.dataset,a.workers)
    else:pack(a.dataset)
if __name__=='__main__':main()


