"""Resumable CryptBB data and Model1 setup preparation. Never trains a reader."""
import argparse, collections, copy, hashlib, io, json, os, shutil, subprocess, sys, types, zipfile
from pathlib import Path
from .decoder.common import read, write, sha, now, StageLock
ROOT=Path(__file__).resolve().parents[3]
CAPRI=ROOT/'Capri'
OUT=ROOT/'handoff/cryptbb_model1_preparation'
RUN=CAPRI/'runs/cryptbb_model1_preparation_20261113'
HIST=Path('E:/pranav/CAPRI/runs/capri1_vs_capri2_controller_examples_20261029/CryptBB')
SEEDS={'A':2000000000,'B':6000000000}

def status(stage,state='running',**kw):
    value=dict(updated_at=now(),stage=stage,state=state,pid=os.getpid(),reader_training_started=False,**kw)
    write(OUT/'status.json',value)
    with (OUT/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(value)+'\n')
    print(json.dumps(value),flush=True)

def safe_write(path,value):
    if path.exists():assert read(path)==value,'Existing artifact differs: '+str(path)
    else:write(path,value)

def stop():
    if (OUT/'STOP').exists():raise RuntimeError('User STOP marker; preparation stopped before next stage')

def verify_dependencies():
    p=read(OUT/'protocol.json')
    for x in p['frozen_files']:assert sha(ROOT/x['path'])==x['sha256'],x['path']
    assert sha(CAPRI/'models/registry.json')==p['registry_sha256']
    return p

def setup():
    from PIL import Image
    import numpy as np
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    if (OUT/'setup_receipt.json').exists():verify_dependencies();return
    study=ROOT/'handoff/darkbay_curriculum_20261113'
    assert read(study/'status.json')['state']=='complete_darkbay_waiting_for_cryptbb'
    assert len(read(study/'family_report_validation_selected.json')['results'])==6
    for n in [2,3]:
        for tag in 'AB':
            r=read(ROOT/f'model{n}/runs/darkbay/{tag}/completion.json')
            assert r['epoch']==20 and r['global_step']==15640
            assert sha(ROOT/f'model{n}/runs/darkbay/{tag}'/r['checkpoint'])==r['sha256']
    records=read(OUT/'source_pixels.json');audit=read(OUT/'source_audit.json')
    assert audit['counts']==dict(original=600,reference=100,selftrain=100,test=551)
    assert audit['historical_reference_pixel_hashes_match'] and not audit['decode_errors']
    for key in ['reference__test','reference__selftrain','selftrain__test']:assert audit['overlap'][key]==0
    for role,rows in records.items():
        for r in rows:
            with Image.open(r['source']) as im:
                im=im.convert('RGB');assert list(im.size)==r['size'] and hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256']
    refs=[];lookup={r['pixel_sha256']:r for r in records['reference']}
    for i,r in enumerate(read(HIST/'capri2/inputs.json')):
        source=lookup[r['pixel_sha256']];p=RUN/'inputs'/f'{i:03d}_{r["pixel_sha256"][:12]}.png';p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():Image.open(source['source']).convert('RGB').save(p)
        assert hashlib.sha256(Image.open(p).convert('RGB').tobytes()).hexdigest()==r['pixel_sha256']
        refs.append(dict(source=str(p),pixel_sha256=r['pixel_sha256']))
    safe_write(RUN/'targets.json',dict(fit=refs))
    # Historical filenames explicitly documented as scoring labels in DATASETS.md.
    label_doc=Path('E:/pranav/cryptbb_sourcefree/DATASETS.md')
    assert 'filenames ARE the labels' in label_doc.read_text(encoding='utf8')
    images=[];labels={};mapping=[]
    for i,r in enumerate(records['test']):
        opaque=f'cryptbb_{i:04d}';p=ROOT/'evaluation_data/cryptbb/images'/(opaque+'.png');p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():Image.open(r['source']).convert('RGB').save(p)
        assert hashlib.sha256(Image.open(p).convert('RGB').tobytes()).hexdigest()==r['pixel_sha256']
        images.append(dict(opaque_id=opaque,image_path=str(p),pixel_sha256=r['pixel_sha256'],size=r['size']))
        labels[opaque]=Path(r['source']).stem
        mapping.append(dict(opaque_id=opaque,source=r['source'],source_file_sha256=r['file_sha256']))
    safe_write(ROOT/'evaluation_data/cryptbb/image_manifest.json',images)
    safe_write(ROOT/'evaluation_data/cryptbb/scoring_labels/scoring_labels.json',labels)
    safe_write(ROOT/'evaluation_data/cryptbb/scoring_labels/provenance.json',dict(label_source=str(label_doc),label_source_sha256=sha(label_doc),mapping=mapping,warning='Historical filename labels, not independently reannotated; scoring only. Historical selftrain100 excluded.'))
    exclusions=sorted({r['pixel_sha256'] for rows in records.values() for r in rows})
    safe_write(RUN/'excluded_pixels.json',exclusions)
    # Broad DarkBay charset is retained, never derived from real CryptBB labels.
    p=ROOT/'datasets/cryptbb/charset.txt';p.parent.mkdir(parents=True,exist_ok=True)
    source=ROOT/'datasets/darkbay/A/charset.txt'
    if not p.exists():shutil.copyfile(source,p)
    assert sha(p)==sha(source)
    registry=read(CAPRI/'models/registry.json');selected=registry['selected_generator']
    cp=CAPRI/selected['path'];assert sha(cp)==selected['sha256'] and selected['internal_count'] and not selected['separate_counter_used']
    safe_write(RUN/'selected_generator/completion.json',dict(epochs=selected['epoch'],checkpoint=str(cp),sha256=selected['sha256'],weights=selected['weights']))
    files=[Path(__file__),CAPRI/'src/capri_controller/controller_comparison.py',CAPRI/'src/capri_controller/pipeline.py',CAPRI/'src/capri_controller/corpus.py',CAPRI/'src/capri_controller/training/capri2_parseq.py']
    for folder in ['features','rendering','lightweight','networks']:files+=list((CAPRI/'src/capri_controller'/folder).glob('*.py'))
    files+=[CAPRI/'src/capri_controller/controllers.py',CAPRI/'src/capri_controller/runtime.py',CAPRI/'configs/local.json']
    frozen=[]
    for f in files:
        relative=f.relative_to(ROOT);dest=RUN/'provenance'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(f,dest)
        assert sha(dest)==sha(f);frozen.append(dict(path=relative.as_posix(),sha256=sha(f)))
    protocol=dict(created_at=now(),scope='CryptBB datasets and Model1 setup only',reader_training_authorized=False,reader_training_started=False,selected_generator=selected,registry_sha256=sha(CAPRI/'models/registry.json'),reference_manifest_sha256=sha(RUN/'targets.json'),source_audit_sha256=sha(OUT/'source_audit.json'),images_each=100000,render_seed_bases=SEEDS,controllers=['custom','dino','mixed'],allocation=dict(custom=33334,dino=33333,mixed=33333),count_rule='Fresh pixel-only selected Capri internal count; unique leading count >=50/100, ties rejected',calibration='Reconstruct historical CryptBB capri1 first24 recipes with current verified controller features, matching DarkBay calibration procedure',real_test_images=551,excluded_historical_selftrain_images=100,real_labels_in_generation=False,reader=dict(epochs=20,batch_size=128,steps_per_epoch=782,draws_per_epoch=100096,seed=48,initialization='from scratch',checkpoint_selection='Frozen synthetic validation only, earliest epoch on ties; CryptBB validation set must be frozen before future training'),frozen_files=frozen)
    safe_write(OUT/'protocol.json',protocol)
    write(OUT/'setup_receipt.json',dict(completed_at=now(),references=100,test=551,all_source_pixels_reverified=True,reference_test_overlap=0,labels_isolated=True,training_started=False,targets_sha256=sha(RUN/'targets.json')))
    status('inputs_staged','complete')

def fit():
    verify_dependencies();os.environ['CAPRI_RUN_DIR']=str(RUN/'generator');os.environ['CAPRI_TARGET_MANIFEST']=str(RUN/'targets.json')
    dest=RUN/'generator'
    if dest.exists():
        assert (dest/'status.json').exists() and read(dest/'status.json').get('state')=='complete','Partial controller fit preserved; inspect before retry'
        assert len(read(dest/'entries.json'))==300;return
    from .training import capri2_parseq as F
    F.OLDGEN=HIST/'capri1'
    F.calibration(RUN)
    from . import controller_comparison as C
    C.NEW_RUN=RUN/'selected_generator'  # Registry-backed compatibility receipt, not historical weights.
    C.worker(types.SimpleNamespace(out=str(dest),targets=str(RUN/'targets.json'),version='capri2',calibration=str(RUN/'baseline_calibration.npz')),full=True)
    decision=read(dest/'length_decision.json');assert decision['votes']>=50 and decision['mode']=='majority'
    assert decision['model_sha256']==read(OUT/'protocol.json')['selected_generator']['sha256']
    assert len(read(dest/'raw_hybrid_predictions.json'))==100 and len(read(dest/'entries.json'))==300
    write(OUT/'count_vote_receipt.json',dict(completed_at=now(),fresh_inference=True,decision=decision,raw_predictions_sha256=sha(dest/'raw_hybrid_predictions.json')))

def stage_corpus(tag):
    verify_dependencies();source=RUN/'generator';dest=ROOT/f'datasets/cryptbb/{tag}'
    assert read(source/'status.json')['state']=='complete';dest.mkdir(parents=True,exist_ok=True)
    for name in ['entries.json','inputs.json','length_decision.json','protocol.json','model_manifest.json']:
        p=dest/name
        if not p.exists():shutil.copyfile(source/name,p)
        assert sha(p)==sha(source/name)
    safe_write(dest/'controller_reuse.json',dict(source=str(source),entries_sha256=sha(source/'entries.json'),render_seed_base=SEEDS[tag],fitting_repeated=False))
    return dest

def pack(tag):
    import lmdb
    from PIL import Image
    dest=ROOT/f'datasets/cryptbb/{tag}';receipt=dest/'data_receipt.json';db=dest/'train_lmdb'
    if receipt.exists():assert sha(db/'data.mdb')==read(receipt)['lmdb_sha256'];return
    assert read(dest/'corpus_audit.json')['images']==100000
    if db.exists():raise RuntimeError('Incomplete LMDB preserved; inspect before resuming '+tag)
    charset=set((ROOT/'datasets/cryptbb/charset.txt').read_text().splitlines());excluded=set(read(RUN/'excluded_pixels.json'))
    expected_length=read(dest/'length_decision.json')['default_length'];seen=set();counts=collections.Counter();n=0
    db.mkdir();env=lmdb.open(str(db),map_size=8*1024**3)
    with (dest/'sample_provenance.jsonl').open('w',encoding='utf8') as index:
        try:
            for shard in sorted(read(dest/'corpus_summary.json'),key=lambda x:x['entry_id']):
                path=dest/'corpus'/(shard['entry_id']+'.zip');assert sha(path)==shard['zip_sha256'] and shard['seed_base']==SEEDS[tag]
                with zipfile.ZipFile(path) as z,env.begin(write=True) as txn:
                    rows=[json.loads(x) for x in z.read('labels.jsonl').decode().splitlines()];assert len(rows)==shard['count']
                    for row in rows:
                        data=z.read(row['image']);im=Image.open(io.BytesIO(data)).convert('RGB');digest=hashlib.sha256(im.tobytes()).hexdigest()
                        assert digest==row['pixel_sha256'] and digest not in seen and digest not in excluded
                        assert row['complete'] and row['usable'] and row['text']==row['text_spec']['text'] and set(row['text'])<=charset and len(row['text'])==expected_length
                        n+=1;seen.add(digest);counts[row['controller']]+=1
                        txn.put(f'image-{n:09d}'.encode(),data);txn.put(f'label-{n:09d}'.encode(),row['text'].encode());txn.put(f'wh-{n:09d}'.encode(),f'{im.width}_{im.height}'.encode())
                        index.write(json.dumps(dict(index=n,shard=str(path.relative_to(dest)),image=row['image'],text=row['text'],pixel_sha256=digest,controller=row['controller'],seed=row['seed']))+'\n')
            assert n==100000 and dict(counts)==dict(custom=33334,dino=33333,mixed=33333)
            with env.begin(write=True) as txn:txn.put(b'num-samples',str(n).encode())
            env.sync()
        finally:env.close()
    with lmdb.open(str(db),readonly=True,lock=False) as check:
        with check.begin() as txn:assert int(txn.get(b'num-samples'))==100000
    write(receipt,dict(images=n,controllers=dict(counts),lmdb_sha256=sha(db/'data.mdb'),provenance_sha256=sha(dest/'sample_provenance.jsonl'),all_images_decoded=True,unique_pixels=len(seen),all_original_reference_and_historical_eval_pixel_overlap=0,charset_sha256=sha(ROOT/'datasets/cryptbb/charset.txt')))

def configs():
    import yaml
    from .training.capri2_parseq import normalized_config
    hashes=[];sets=[]
    for tag in 'AB':
        d=ROOT/f'datasets/cryptbb/{tag}';receipt=read(d/'data_receipt.json');assert sha(d/'train_lmdb/data.mdb')==receipt['lmdb_sha256']
        assert sha(d/'sample_provenance.jsonl')==receipt['provenance_sha256']
        pixels={json.loads(x)['pixel_sha256'] for x in (d/'sample_provenance.jsonl').read_text().splitlines()};assert len(pixels)==100000;sets.append(pixels)
        original=yaml.safe_load((ROOT/f'model1/configs/darkbay/{tag}/relocated.yml').read_text());cfg=copy.deepcopy(original)
        def relocate(x):
            if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
            if isinstance(x,list):return [relocate(v) for v in x]
            if isinstance(x,str):return x.replace(f'datasets\\darkbay\\{tag}\\charset.txt','datasets\\cryptbb\\charset.txt').replace(f'datasets\\darkbay\\{tag}',f'datasets\\cryptbb\\{tag}').replace(f'model1\\runs\\reproduction_{tag}',f'model1\\runs\\cryptbb\\{tag}')
            return x
        cfg=relocate(cfg);assert normalized_config(original)==normalized_config(cfg)
        assert cfg['Global']['pretrained_model'] is None and cfg['Global']['checkpoints'] is None and cfg['Global']['epoch_num']==20 and cfg['Eval'] is None
        assert 'darkbay' not in json.dumps(cfg).lower() and 'C:' not in json.dumps(cfg)
        p=ROOT/f'model1/configs/cryptbb/{tag}/prepared.yml';p.parent.mkdir(parents=True,exist_ok=True)
        text=yaml.safe_dump(cfg,sort_keys=False)
        if p.exists():assert p.read_text()==text
        else:p.write_text(text,encoding='utf8')
        hashes.append(dict(dataset=tag,path=str(p.relative_to(ROOT)),sha256=sha(p),source_sha256=sha(ROOT/f'model1/configs/darkbay/{tag}/relocated.yml')))
    assert not sets[0]&sets[1]
    write(OUT/'datasets_ready.json',dict(completed_at=now(),images_each=100000,total_unique_pixels=200000,cross_dataset_pixel_overlap=0,configs=hashes,training_started=False,synthetic_validation='Required before training; not yet prepared for CryptBB'))

def preflight(tag):
    import numpy as np,torch
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer
    ready=read(OUT/'datasets_ready.json');config=next(x for x in ready['configs'] if x['dataset']==tag)
    assert sha(ROOT/config['path'])==config['sha256'];dest=ROOT/f'model1/runs/cryptbb/{tag}/preflight'
    receipt=dest/'receipt.json'
    if receipt.exists():assert read(receipt)['passed'];return
    cfg=Config(str(ROOT/config['path']));cfg.cfg['Global']['output_dir']=str(dest)
    torch.set_num_threads(8);trainer=Trainer(cfg,mode='train',task='rec')
    assert trainer.valid_dataloader is None and len(trainer.train_dataloader.dataset)==100000 and len(trainer.train_dataloader)==782
    trainer.model.eval();before=[p.detach().clone() for p in trainer.model.parameters()]
    batch=next(iter(trainer.train_dataloader));x=batch[0].to(trainer.device)
    assert list(x.shape)==[128,3,32,128]
    with torch.inference_mode():y=trainer.model(x)
    assert torch.isfinite(y).all() and all(torch.equal(a,b) for a,b in zip(before,trainer.model.parameters()))
    write(receipt,dict(passed=True,completed_at=now(),dataset_images=100000,batches_per_epoch=782,batch_shape=list(x.shape),parameters=sum(p.numel() for p in trainer.model.parameters()),optimizer_steps=0,backward_calls=0,model_parameters_unchanged=True,initialization='from scratch',config_sha256=config['sha256']))

def child(stage,tag=None):
    stop();verify_dependencies();cmd=[sys.executable,'-B','-u',str(CAPRI/'capri.py'),'prepare-cryptbb-model1','--stage',stage]
    if tag:cmd+=['--dataset',tag]
    label=stage+('_'+tag if tag else '');status(label)
    with (OUT/(label+'.log')).open('a',encoding='utf8') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(OUT/'active_child.json',dict(pid=p.pid,stage=label,command=cmd,started_at=now()))
        code=p.wait()
    write(OUT/'active_child.json',dict(stage=label,exit_code=code,finished_at=now()))
    if code:raise RuntimeError(label+' failed, see '+str(OUT/(label+'.log')))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with StageLock(OUT,'preparation'):
        try:
            setup();child('fit')
            for tag in 'AB':child('generate',tag);child('pack',tag)
            child('configs')
            for tag in 'AB':child('preflight',tag)
            write(OUT/'completion.json',dict(completed_at=now(),datasets=read(OUT/'datasets_ready.json'),reader_training_started=False,limitations=['Historical real labels sourced from documented filename convention, not independently reannotated','CryptBB synthetic validation set must be prepared and frozen before future training']))
            status('ready','complete',reader_training_authorized=False)
        except Exception as exc:status('failed','failed',error=repr(exc));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['run','setup','fit','generate','pack','configs','preflight'],default='run');ap.add_argument('--dataset',choices=['A','B']);a=ap.parse_args()
    assert ROOT.drive.upper()=='E:'
    if a.stage in ['generate','pack','preflight'] and not a.dataset:ap.error('--dataset required')
    if a.stage=='run':return run()
    if a.stage=='setup':return setup()
    verify_dependencies();stop()
    if a.stage=='generate':
        dest=stage_corpus(a.dataset);os.environ['CAPRI_RUN_DIR']=str(dest);os.environ['CAPRI_CORPUS_SEED_BASE']=str(SEEDS[a.dataset])
        from . import corpus
        assert corpus.ROOT==dest;sys.argv=['generate','--workers','4'];return corpus.main()
    if a.stage in ['pack','preflight']:return globals()[a.stage](a.dataset)
    return globals()[a.stage]()
if __name__=='__main__':main()
