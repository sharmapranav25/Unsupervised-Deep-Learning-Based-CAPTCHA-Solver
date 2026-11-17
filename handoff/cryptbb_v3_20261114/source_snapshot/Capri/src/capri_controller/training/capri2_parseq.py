"""CAPRI2 DarkBay: full controllers, audited100k corpus, matched three-seed PARSeq."""
import argparse,collections,copy,gc,hashlib,io,json,os,shutil,subprocess,sys,time,traceback,types,zipfile
from pathlib import Path
from ..decoder.common import ROOT,read,write,sha,now,StageLock
BASE=ROOT/'runs/parseq_darkbay_100k_20261005'
OLDGEN=ROOT/'capri_darkbay_workflow'
CAPRI2=ROOT/'runs/capri_lightweight_joint20_100k_20261028'
OPENOCR=Path('E:/pranav/Final sem/Models/level 1/upstream/OpenOCR')
SEEDS=(48,123,456)

def configure(out):
    if out.drive.upper()!='E:' or not out.is_relative_to(ROOT/'runs') or out in (BASE,CAPRI2):raise ValueError('Fresh experiment below E-drive runs required')
    cache=out/'temporary';cache.mkdir(parents=True,exist_ok=True)
    for k in ('TEMP','TMP','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','MPLCONFIGDIR'):os.environ[k]=str(cache)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path[:0]=[str(ROOT/'.runtime/parseq_packages'),str(OPENOCR)]

def status(out,stage,state='running',**extra):
    record=dict(updated_at=now(),stage=stage,state=state,pid=os.getpid(),**extra)
    write(out/'pipeline_status.json',record);write(out/'status.json',record);print(json.dumps(record),flush=True)

def map_old(path):
    s=str(path).replace('\\','/');old='C:/Users/pranav/Documents/ChatGPT/thesis/'
    p=ROOT/s[len(old):] if s.lower().startswith(old.lower()) else Path(s)
    if p.resolve().drive.upper()!='E:' or not p.is_file():raise FileNotFoundError(p)
    return p.resolve()

def normalized_config(cfg):
    if isinstance(cfg,dict):return {k:('<path>' if k in ('character_dict_path','output_dir','save_res_path','data_dir_list') else normalized_config(v)) for k,v in cfg.items()}
    if isinstance(cfg,list):return [normalized_config(v) for v in cfg]
    return cfg

def setup(out):
    if out.exists():raise FileExistsError('Use --resume for an existing prepared experiment')
    out.mkdir(parents=True);configure(out)
    import numpy as np,yaml
    from PIL import Image
    assert shutil.disk_usage(out).free>40*1024**3
    registry=read(ROOT/'models/registry.json');old_models=read(OLDGEN/'model_manifest.json')
    for k,v in registry['models'].items():assert sha(ROOT/v['path'])==v['sha256']==old_models[k]['sha256']
    completion=read(CAPRI2/'completion.json');assert completion['epochs']==20 and sha(Path(completion['checkpoint']))==completion['sha256']
    base_completion=read(BASE/'completion.json');assert base_completion['all_complete']
    for row in base_completion['seeds']:assert sha(BASE/f'seed_{row["seed"]}/latest.pth')==row['sha256']
    refs=read(OLDGEN/'inputs.json');assert len(refs)==100 and len({r['pixel_sha256'] for r in refs})==100
    (out/'inputs').mkdir();opaque=[]
    for i,r in enumerate(refs):
        im=Image.open(map_old(r['source'])).convert('RGB');assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
        p=out/'inputs'/f'{i:03d}_{r["pixel_sha256"][:12]}.png';im.save(p);opaque.append(dict(source=str(p),pixel_sha256=r['pixel_sha256']))
    write(out/'targets.json',dict(fit=opaque));shutil.copyfile(BASE/'charset.txt',out/'charset.txt')
    configurations={}
    for seed in SEEDS:
        original=yaml.safe_load((BASE/f'seed_{seed}.yml').read_text());cfg=copy.deepcopy(original)
        old_prefix='C:\\Users\\pranav\\Documents\\ChatGPT\\thesis\\runs\\parseq_darkbay_100k_20261005'
        def relocate(x):
            if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
            if isinstance(x,list):return [relocate(v) for v in x]
            if isinstance(x,str):return x.replace(old_prefix,str(out)).replace(str(BASE),str(out))
            return x
        cfg=relocate(cfg);assert normalized_config(original)==normalized_config(cfg)
        assert 'C:' not in json.dumps(cfg) and cfg['Global']['epoch_num']==20 and cfg['Eval'] is None
        (out/f'seed_{seed}.yml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8');configurations[str(seed)]=sha(out/f'seed_{seed}.yml')
    files=[ROOT/'capri.py',ROOT/'configs/local.json',ROOT/'models/registry.json',ROOT/'src/capri_controller/pipeline.py',ROOT/'src/capri_controller/controller_comparison.py',ROOT/'src/capri_controller/corpus.py',ROOT/'src/capri_controller/controllers.py',ROOT/'src/capri_controller/artifacts.py',ROOT/'src/capri_controller/runtime.py']
    for folder in ('training','features','rendering','networks','lightweight'):files+=list((ROOT/'src/capri_controller'/folder).glob('*.py'))
    frozen=[]
    for p in files:
        dst=out/'provenance'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);frozen.append(dict(path=str(p),sha256=sha(p)))
    for p in OPENOCR.rglob('*.py'):
        if '.git' in p.parts:continue
        dst=out/'provenance/OpenOCR'/p.relative_to(OPENOCR);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);frozen.append(dict(path=str(p),sha256=sha(p)))
    protocol=dict(created_at=now(),experiment='CAPRI2 DarkBay generator -> matched PARSeq baseline',baseline=str(BASE),generator_checkpoint=completion,reference_input_sha256=sha(out/'targets.json'),baseline_input_sha256=sha(OLDGEN/'inputs.json'),baseline_recipes_sha256=sha(OLDGEN/'refined_recipes.json'),registry_sha256=sha(ROOT/'models/registry.json'),images=100000,seeds=SEEDS,epochs=20,batch_size=128,expected_training_steps_per_seed=15640,initialization='from_scratch, same as CAPRI1 baseline',reader='SVTRv2LNConvTwo33 + PARSeq decoder',optimizer='AdamW lr0.00035,weight_decay0.05;OneCycle,warmup2epochs',augmentation='Exact saved CAPRI1 reader config and loader; no added curriculum or augmentation',checkpoint_selection='Fixed final epoch20; no real-target-based selection',generation=dict(controllers=['custom','dino','mixed'],entries=300,allocation=[33334,33333,33333],count='Internal CAPRI, unique leading count >=50/100; stop if none',font='Trained DINO residual font heads',separate_counter_used=False,calibration='Reconstruct CAPRI1 original first24 model-derived calibration renders; freeze features for CAPRI2 controllers',seed_base=2000000000,preprocessing='CAPRI2 RGB96,bilinear,widthcap480; historical CAPRI1 used Lanczos,widthcap960'),evaluation='Same900 historical real DarkBay pixels, excluding100 references; freeze all three final predictions before reading scoring labels',source_files=frozen,config_sha256=configurations,limits=['Historical real test has been examined before','CAPRI2 text-ordinal and font width/slant supervision absent; preserve actual outputs','Generated-data quality and reader gains remain unproven','Baseline source/config matched; old run did not preserve full external source hashes, current OpenOCR snapshotted'])
    write(out/'protocol.json',protocol)
    write(out/'configuration_verification.json',dict(passed=True,all_three_reader_configs_identical_except_paths=True,baseline_checkpoints_verified=True,active_registry_unchanged=True))
    (out/'README.md').write_text('''# CAPRI2 DarkBay PARSeq experiment

Authorized full run:100 references ->300 custom/DINO/mixed recipes ->100,000 unique synthetic images ->three PARSeq seeds48,123,456 for20epochs each ->same900-image real evaluation.

Read pipeline_status.json for the orchestrator stage, generator/status.json for controller progress, generator/corpus_status.json for corpus generation, and seed_<seed>/train.log for actual reader progress. Preparation and training are separate stages. Final completion requires experiment_completion.json. All artifacts are under this E-drive directory.

The CAPRI1 reader configurations are copied with output/data/charset paths relocated only. Initialization is from scratch. AdamW lr0.00035,weight_decay0.05,batch128,OneCycle with2-epoch warmup,mixed precision. Fixed final epoch20 checkpoints. Baseline reports782 batches/epoch and15,640 total steps per seed; the new loader must match before training. No curriculum or new augmentation. All100k generated images are retained; strict readability is diagnostic, not a filter.

CAPRI2 uses its final epoch20 EMA, internal count and trained DINO residual font heads. Full100-input fitting uses all three existing controllers with original CAPRI1 calibration features reconstructed from its first24 recipes. Count threshold50%, no manually supplied family length. Frozen checkpoints, source snapshots and provenance are in protocol.json and provenance/. Nothing is deployed by this experiment.

Preflight on existing baseline synthetic data tests the unchanged reader without optimizer updates, then a second preflight tests the actual CAPRI2 corpus. Dataset audits decode and hash all PNGs, verify100k unique images, controller balance and synthetic text labels, and exclude exact overlap with100 fit and900 test pixels. Evaluation reads real labels only after all final predictions are hash-frozen. Read-only baseline remains runs/parseq_darkbay_100k_20261005.

Start/resume from E:/pranav/CAPRI using the configured E-drive Python: capri.py train-capri2-parseq --out <this directory> --resume. A live pipeline lock prevents duplicate runs. Incomplete controller or LMDB stages stop for inspection; they are not silently overwritten. Corpus shards resume only if their hashes and recipe manifests match. Reader recovery uses the existing baseline checkpoint mechanism; an interrupted epoch may be replayed and bit-exact resume is not promised. Create STOP to stop after the current stage; it does not interrupt a reader epoch or kill another job.

Initial budget: roughly7-9hours total based on the old three-reader duration (~5h43m), plus full controller fitting, corpus generation and audits. Actual ETA depends on measured throughput and other GPU use. Weak/faint generator outputs remain a known limitation; success does not imply better reader accuracy.
''',encoding='utf8')
    status(out,'prepared',training_started=False)

def verify(out):
    p=read(out/'protocol.json')
    assert all(sha(Path(r['path']))==r['sha256'] for r in p['source_files'])
    assert all(sha(out/f'seed_{s}.yml')==h for s,h in p['config_sha256'].items())
    assert sha(ROOT/'models/registry.json')==p['registry_sha256']
    assert sha(Path(p['generator_checkpoint']['checkpoint']))==p['generator_checkpoint']['sha256']

def preflight(out):
    import torch,numpy as np
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer
    actual=(out/'data_receipt.json').exists();tag='new' if actual else 'baseline';receipt=out/f'preflight_{tag}.json'
    if receipt.exists():assert read(receipt)['passed'];return
    source=out if actual else BASE;assert sha(source/'train_lmdb/data.mdb')==read(source/'data_receipt.json')['lmdb_sha256']
    assert torch.cuda.is_available(),'CUDA required; refusing silent CPU training'
    cfg=Config(str(out/'seed_48.yml'));cfg.cfg['Global']['output_dir']=str(out/f'preflight_{tag}_no_updates');cfg.cfg['Train']['dataset']['data_dir_list']=[str(source/'train_lmdb')]
    torch.set_num_threads(8);trainer=Trainer(cfg,mode='train',task='rec');assert trainer.valid_dataloader is None
    assert len(trainer.train_dataloader.dataset)==100000 and len(trainer.train_dataloader)==782
    for i,m in enumerate(trainer.model.modules()):
        if isinstance(getattr(m,'rng',None),np.random.Generator):m.rng=np.random.default_rng(48+i)
    batch=[v.to(trainer.device) for v in next(iter(trainer.train_dataloader))];start=time.time()
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=trainer.model(batch[0],data=batch[1:]);loss=trainer.loss_class(pred,batch)['loss']
    assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize()
    assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
    write(receipt,dict(passed=True,created_at=now(),data=tag,dataset_images=100000,batches_per_epoch=782,batch_shape=list(batch[0].shape),loss=float(loss),seconds=time.time()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,optimizer_steps=0,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0)))
    print('Reader preflight passed:',tag,flush=True)

def calibration(out):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from ..controllers import Mixed
    from ..rendering import renderer as R
    dest=out/'baseline_calibration.npz'
    if dest.exists():assert sha(dest)==read(out/'baseline_calibration.json')['sha256'];return
    recipes=read(OLDGEN/'refined_recipes.json');assert len(recipes)==100
    model=Mixed()
    with ProcessPoolExecutor(max_workers=8,initializer=R.install) as pool:rows=list(pool.map(R.safe_render,[(r,R.expand(dict(length=r['description']['length']))) for r in recipes[:24]]))
    features=model.extract([r.get('image',np.full((64,256,3),255,np.uint8)) for r in rows])
    np.savez(dest,**{str(i)+'__'+k:v for i,part in enumerate(features) for k,v in part.items()})
    write(out/'baseline_calibration.json',dict(created_at=now(),sha256=sha(dest),recipe_sha256=sha(OLDGEN/'refined_recipes.json'),source='Original CAPRI1 first24 recipes, same renderer and controller features',render_errors=[r.get('error') for r in rows if 'error' in r]))

def controllers(out):
    dest=out/'generator'
    if dest.exists():
        assert read(dest/'status.json').get('state')=='complete','Partial controller stage requires inspection; no overwrite'
        assert len(read(dest/'entries.json'))==300;return
    from ..controller_comparison import worker
    worker(types.SimpleNamespace(out=str(dest),targets=str(out/'targets.json'),version='capri2',calibration=str(out/'baseline_calibration.npz')),full=True)
    assert len(read(dest/'entries.json'))==300 and read(dest/'length_decision.json')['threshold']==.5
    assert not read(dest/'protocol.json')['preview_only']

def corpus(out):
    os.environ['CAPRI_RUN_DIR']=str(out/'generator');os.environ['CAPRI_CORPUS_SEED_BASE']='2000000000'
    from .. import corpus as C
    assert C.ROOT==out/'generator';sys.argv=['generate','--workers','8'];C.main()
    audited=read(out/'generator/corpus_audit.json');assert audited['images']==100000 and audited['target_pixel_overlap']==0
    summaries=sorted(read(out/'generator/corpus_summary.json'),key=lambda r:r['entry_id'])
    write(out/'generator/dataset_manifest.json',dict(count=100000,controllers=audited['controllers'],entries_sha256=sha(out/'generator/entries.json'),shards=[dict(**s,path='corpus/'+s['entry_id']+'.zip') for s in summaries]))

def pack(out):
    import lmdb,numpy as np
    from PIL import Image
    source=out/'generator';receipt=out/'data_receipt.json';db=out/'train_lmdb'
    if receipt.exists():assert sha(db/'data.mdb')==read(receipt)['lmdb_sha256'];return
    if db.exists():raise RuntimeError('Incomplete LMDB preserved; inspect before retry')
    manifest=read(source/'dataset_manifest.json');assert manifest['count']==100000
    expected_length=read(source/'length_decision.json')['default_length']
    excluded={r['pixel_sha256'] for r in read(out/'targets.json')['fit']}
    excluded.update(r['pixel_sha256'] for r in read(BASE/'evaluation_darkbay/image_manifest.json'));assert len(excluded)==1000
    db.mkdir();env=lmdb.open(str(db),map_size=8*1024**3);seen=set();counts=collections.Counter();lengths=collections.Counter();n=0;strict=0
    charset=set((out/'charset.txt').read_text().splitlines())
    with (out/'sample_provenance.jsonl').open('w',encoding='utf8') as index:
        try:
            for shard in manifest['shards']:
                path=source/shard['path'];assert sha(path)==shard['zip_sha256']
                with zipfile.ZipFile(path) as z,env.begin(write=True) as txn:
                    rows=[json.loads(x) for x in z.read('labels.jsonl').decode().splitlines()];assert len(rows)==shard['count']
                    for row in rows:
                        data=z.read(row['image']);im=Image.open(io.BytesIO(data)).convert('RGB');digest=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
                        assert digest==row['pixel_sha256'] and digest not in seen and digest not in excluded
                        assert row['complete'] and row['usable'] and row['text']==row['text_spec']['text'] and set(row['text'])<=charset
                        assert len(row['text'])==expected_length
                        seen.add(digest);n+=1;counts[row['controller']]+=1;lengths[len(row['text'])]+=1;strict+=row['strict']
                        txn.put(f'image-{n:09d}'.encode(),data);txn.put(f'label-{n:09d}'.encode(),row['text'].encode());txn.put(f'wh-{n:09d}'.encode(),f'{im.width}_{im.height}'.encode())
                        index.write(json.dumps(dict(index=n,shard=shard['path'],image=row['image'],text=row['text'],pixel_sha256=digest,controller=row['controller'],entry_id=row['entry_id']))+'\n')
                if n%10000<334:print('Packed',n,'/100000',flush=True)
            assert n==100000 and dict(counts)==manifest['controllers'] and sorted(counts.values())==[33333,33333,33334]
            with env.begin(write=True) as txn:txn.put(b'num-samples',str(n).encode())
            env.sync()
        finally:env.close()
    write(receipt,dict(images=n,unique_pixels=len(seen),controllers=dict(counts),lengths=dict(lengths),strict_readability=strict,filtering='None; all100k retained',source_manifest_sha256=sha(source/'dataset_manifest.json'),lmdb_sha256=sha(db/'data.mdb'),provenance_sha256=sha(out/'sample_provenance.jsonl'),charset_sha256=sha(out/'charset.txt'),real_fit_and_test_pixel_overlap=0))

def train_seed(out,seed,resume):
    assert read(out/'preflight_new.json')['passed']
    from . import train_parseq as T
    T.RUN=out
    T.one(seed,None,True,resume=resume)

def evaluate(out):
    import numpy as np,torch
    from PIL import Image
    from torchvision import transforms as TF
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    dest=out/'evaluation_darkbay'
    if dest.exists():
        assert (dest/'summary.json').exists(),'Incomplete evaluation preserved'
        frozen=read(dest/'prediction_freeze.json');assert all(sha(dest/v['file'])==v['sha256'] for v in frozen.values());return
    dest.mkdir();rows=read(BASE/'evaluation_darkbay/image_manifest.json');assert len(rows)==900
    records=[dict(r,image_path=str(map_old(r['image_path']))) for r in rows]
    for r in records:assert hashlib.sha256(np.asarray(Image.open(r['image_path']).convert('RGB')).tobytes()).hexdigest()==r['pixel_sha256']
    assert not {r['pixel_sha256'] for r in records}&{r['pixel_sha256'] for r in read(out/'targets.json')['fit']}
    write(dest/'image_manifest.json',records);completion=read(out/'completion.json');assert completion['all_complete']
    tf=TF.Compose([TF.Resize((32,128),interpolation=TF.InterpolationMode.BICUBIC),TF.ToTensor(),TF.Normalize(.5,.5)])
    torch.set_num_threads(8);assert torch.cuda.is_available();predictions={};frozen={}
    for row in completion['seeds']:
        assert row['epoch']==20 and sha(Path(row['checkpoint']))==row['sha256'];seed=row['seed']
        ck=torch.load(row['checkpoint'],map_location='cpu',weights_only=False);cfg=copy.deepcopy(ck['config'])
        decoder=build_post_process(cfg['PostProcess'],cfg['Global']);cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(ck['state_dict'],strict=True);model.cuda().eval();del ck
        values=[]
        with torch.inference_mode():
            for i in range(0,len(records),64):
                chunk=records[i:i+64];x=torch.stack([tf(Image.open(r['image_path']).convert('RGB')) for r in chunk]).cuda()
                decoded=decoder(model(x));assert len(decoded)==len(chunk)
                values.extend(dict(opaque_id=r['opaque_id'],prediction=t,confidence=float(c) if np.isfinite(c) else None) for r,(t,c) in zip(chunk,decoded))
        path=dest/f'seed_{seed}_predictions.json';write(path,values);frozen[str(seed)]=dict(file=path.name,sha256=sha(path),count=900);predictions[seed]={v['opaque_id']:v['prediction'] for v in values}
        del model;gc.collect();torch.cuda.empty_cache()
    write(dest/'prediction_freeze.json',frozen)
    labels=read(BASE/'evaluation_darkbay/scoring_labels.json');assert set(labels)=={r['opaque_id'] for r in records}
    from rapidfuzz.distance import Levenshtein
    results=[]
    for seed,pred in predictions.items():
        assert sha(dest/frozen[str(seed)]['file'])==frozen[str(seed)]['sha256']
        correct=sum(pred[k]==v for k,v in labels.items());distance=sum(Levenshtein.distance(v,pred[k]) for k,v in labels.items())
        results.append(dict(seed=seed,correct=correct,total=900,exact_match_percent=100*correct/900,character_error_rate_percent=100*distance/sum(map(len,labels.values()))))
    old=read(BASE/'evaluation_darkbay/summary.json');old_by_seed={r['seed']:r for r in old['results']}
    summary=dict(results=results,mean_exact_match_percent=float(np.mean([r['exact_match_percent'] for r in results])),baseline=old,comparison=[dict(seed=r['seed'],capri1_correct=old_by_seed[r['seed']]['correct'],capri2_correct=r['correct'],change_percentage_points=r['exact_match_percent']-old_by_seed[r['seed']]['exact_match_percent']) for r in results],selection='All fixed final epoch20 checkpoints; predictions frozen before scoring',historical_real_test=True)
    write(dest/'summary.json',summary);write(dest/'scoring_labels.json',labels)
    write(dest/'protocol.json',dict(checkpoints=completion['seeds'],same900_image_hashes_as_baseline=True,training_initialization='from_scratch',preprocessing='RGB,bicubic32x128,ToTensor,Normalize0.5',decoding='PARSeq autoregressive with one refinement; no forced length or narrowed charset',labels_read_after_all_prediction_freezes=True))
    (dest/'RESULTS.md').write_text('# CAPRI2 versus CAPRI1 DarkBay readers\n\n'+json.dumps(summary,indent=2)+'\n',encoding='utf8')

def child(out,stage,seed=None,resume=False):
    verify(out)
    if (out/'STOP').exists():raise RuntimeError('User STOP marker found before next stage')
    status(out,stage,current_seed=seed,training_started=stage=='train-seed')
    cmd=[sys.executable,'-B','-u',str(ROOT/'capri.py'),'train-capri2-parseq','--out',str(out),'--stage',stage]
    if seed is not None:cmd+=['--seed',str(seed)]
    if resume:cmd+=['--resume']
    log=out/(stage+('_'+str(seed) if seed is not None else '')+'.console.log')
    with log.open('a',encoding='utf8') as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(out/'active_child.json',dict(pid=proc.pid,stage=stage,seed=seed,command=cmd,log=str(log),started_at=now()))
        code=proc.wait()
    if code:raise RuntimeError(f'{stage} failed with exit{code}; inspect {log}')

def run(out,resume):
    if not resume:setup(out)
    else:configure(out);verify(out)
    with StageLock(out,'pipeline'):
        try:
            child(out,'preflight');child(out,'calibration');child(out,'controllers');child(out,'corpus');child(out,'pack');child(out,'preflight')
            receipts=[]
            for seed in SEEDS:
                done=out/f'seed_{seed}/completion.json'
                if done.exists():
                    r=read(done);assert r['epoch']==20 and sha(Path(r['checkpoint']))==r['sha256']
                else:
                    child(out,'train-seed',seed,resume=resume and (out/f'seed_{seed}/latest.pth').exists());r=read(done)
                receipts.append(r)
            write(out/'completion.json',dict(seeds=receipts,epochs=20,all_complete=True))
            child(out,'evaluate');verify(out)
            write(out/'experiment_completion.json',dict(completed_at=now(),reader_training_complete=True,real_evaluation_complete=True,summary=str(out/'evaluation_darkbay/summary.json')))
            status(out,'complete',state='complete',training_started=True,seeds=list(SEEDS),epochs=20)
        except Exception as exc:
            status(out,'failed',state='failed',error=repr(exc));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--seed',type=int,choices=SEEDS)
    ap.add_argument('--stage',choices=['run','setup','preflight','calibration','controllers','corpus','pack','train-seed','evaluate'],default='run');args=ap.parse_args();out=Path(args.out).resolve()
    if args.stage=='run':return run(out,args.resume)
    if args.stage=='setup':return setup(out)
    configure(out);verify(out)
    if args.stage=='train-seed':
        if args.seed is None:raise ValueError('Seed required')
        return train_seed(out,args.seed,args.resume)
    return globals()[args.stage](out)
