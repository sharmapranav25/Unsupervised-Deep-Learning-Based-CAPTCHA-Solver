"""Isolated seed-123 DarkBay Model1-A replication; immutable upstream loop."""
import argparse, contextlib, copy, gc, hashlib, inspect, json, os, random
import shutil, subprocess, sys, time, traceback
from pathlib import Path
from capri_controller.decoder.common import StageLock, read, write, sha, now
from capri_controller.training import cryptbb_model1 as B
from capri_controller.training.darkbay_overnight_v1 import common as W

ROOT=W.ROOT
NAME='darkbay_v3_model1_A_seed123_20261116'
OLD=ROOT/'handoff/darkbay_v3_feature_20261115'
STUDY=ROOT/'handoff'/NAME
OUT=ROOT/'model1/runs'/NAME/'A'
RESULTS=ROOT/'model1/results'/NAME/'A'
CONFIG=ROOT/'model1/configs'/NAME/'A/prepared.yml'
ENTRY=ROOT/'model1/src'/NAME/'capri.py'
SEED=123

def stop():
    assert not (STUDY/'STOP').exists(), 'User STOP; no automatic removal'

def environment():
    W.environment()
    os.environ['PYTHONHASHSEED']=str(SEED)
    B.STUDY=STUDY

def verify(data=False):
    p=read(STUDY/'protocol.json')
    for r in p['sources']+p['inputs']:
        assert sha(ROOT/r['path'])==r['sha256'],r['path']
    if data:
        d=W.DATA/'A';r=read(d/'data_receipt.json')
        assert r['images']==r['unique_pixels']==100000
        for path,key in [(d/'train_lmdb/data.mdb','lmdb_sha256'),(d/'sample_provenance.jsonl','provenance_sha256'),(W.DATA/'charset.txt','charset_sha256')]:
            assert sha(path)==r[key],str(path)
    return p

def identities():
    import psutil
    rows=[]
    for lock in [OLD/'coordinator.lock.json',OLD/'fast_gpu_worker.lock.json',OLD/'model3/B/worker_B.lock.json']:
        if lock.exists():
            r=read(lock)
            try:live=abs(psutil.Process(r['pid']).create_time()-r['process_created'])<.02
            except psutil.NoSuchProcess:live=False
            rows.append(dict(path=str(lock),identity=r,live=live))
    return rows

def gpu_owners():
    import psutil
    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    owners=[]
    # WDDM returns graphics applications and stale PIDs in this query. Match
    # actual live Python compute identities; old-study locks are checked too.
    for line in s.splitlines():
        try:
            p=psutil.Process(int(line.split(',')[0]))
            if 'python' in p.name().lower():
                owners.append(dict(pid=p.pid,process_created=p.create_time(),command=p.cmdline()))
        except (psutil.NoSuchProcess,ValueError):pass
        except psutil.AccessDenied:
            if 'python' in line.lower():owners.append(dict(unresolved=line))
    return owners

@contextlib.contextmanager
def gpu_owner():
    import msvcrt
    stop()
    # Same interprocess byte lock used by the original study, without removing
    # or bypassing its STOP for any old worker. This NEW study has its own STOP.
    with (OLD/'gpu.lock').open('a+b') as f:
        f.seek(0)
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:
            assert not any(r['live'] for r in identities()), 'Old study still active; queue'
            assert not gpu_owners(), 'Another CUDA compute owner exists; queue'
            with StageLock(STUDY,'gpu_owner'):yield
        finally:
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def prepare():
    import yaml
    assert not (STUDY/'protocol.json').exists(), 'Frozen protocol already exists'
    STUDY.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'model1/configs/darkbay_v3_feature_20261115/A/prepared.yml').read_text())
    original=copy.deepcopy(cfg)
    cfg['Global'].update(seed=SEED,output_dir=str(OUT),save_res_path=str(OUT/'predictions_unused.txt'))
    cfg['Train']['sampler']['seed']=SEED
    CONFIG.parent.mkdir(parents=True,exist_ok=True);CONFIG.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    check=copy.deepcopy(cfg);check['Global'].update({k:original['Global'][k] for k in ['seed','output_dir','save_res_path']});del check['Train']['sampler']['seed']
    assert check==original
    files=list((ROOT/'model1/src/OpenOCR').rglob('*.py'))
    files+=list((ROOT/'model1/src/parseq_packages').rglob('*.py'))
    files += [Path(__file__),ENTRY,Path(B.__file__),Path(W.__file__),ROOT/'Capri/src/capri_controller/decoder/common.py',ROOT/'Capri/src/capri_controller/training/ab_curriculum.py']
    sources=[]
    for p in sorted(set(files)):
        rel=p.relative_to(ROOT);dest=STUDY/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        sources.append(dict(path=rel.as_posix(),sha256=sha(p)))
    inputs=[CONFIG,ROOT/'model1/configs/darkbay_v3_feature_20261115/A/prepared.yml',OLD/'model1/protocol.json',OLD/'reference_split.json',OLD/'targets.json',W.DATA/'A/data_receipt.json',W.DATA/'charset.txt',ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json']
    seed_audit=dict(global_seed=123,python_hash_seed=123,trainer='Unchanged Trainer.set_random_seed(123): Python, NumPy, torch CPU, all CUDA',dataset='Original traversal seeded by Trainer; dataset.seed=123+(epoch-1)',sampler_constructor='Pass explicit seed123 instead of implicit None entropy',sampler_iteration='Original RatioSampler algorithm, only random.seed(self.epoch) replaced by random.seed(self.seed+self.epoch), initial counter0',module_generators='Original module enumeration preserved; hardcoded48+i becomes123+i before any training forward',inactive_entropy='Other decoders/CMER transforms not configured. PARSeq default_rng() reset deterministically before use.',workers=0,notes='Seed changes include previously hardcoded/entropy-derived streams by explicit user request; padding, coverage, order algorithm and upstream training operations unchanged.')
    write(STUDY/'seed_audit.json',seed_audit)
    protocol=dict(created_at=now(),authorization='User explicitly requested fresh Model1-A seed123 replication, fast setting, existing exact A dataset, all20 real900 diagnostics and matched seed48 comparisons.',name=NAME,scope='Model1-A ONLY; no B, Model2, Model3; old study stays paused',seed=123,epochs=20,batch_size=128,steps_per_epoch=782,total_steps=15640,draws_per_epoch=100096,unique_images=100000,dataset=str(W.DATA/'A'),dataset_reused=True,dataset_receipt=read(W.DATA/'A/data_receipt.json'),initialization='scratch; no old checkpoint loaded into production',selection='Fixed epoch20 primary; all earlier scores retrospective diagnostics only',seed_audit=seed_audit,training_loop='Unchanged upstream Trainer.train; atomic full-state archive hook; no prefetch or worker-count change',execution='One GPU owner retains allocator cache; gated on exact24-update baseline/cached parity and full-state recovery',evaluation='After training, one GPU evaluator, all20 checkpoints fixed900; predictions frozen before labels; separate process and costs; compare original seed48 same epoch',sources=sources,inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in inputs],old_STOP_preserved=True)
    write(STUDY/'protocol.json',protocol);verify(data=True)
    write(STUDY/'prepared.json',dict(at=now(),protocol_sha256=sha(STUDY/'protocol.json'),data_hash_verified=True,readiness_passed=False,production_started=False))

def install_seed_adapter():
    import numpy as np
    from tools.data import ratio_sampler as RS, ratio_dataset_tvresize as DS
    if getattr(RS.RatioSampler,'replication_seed_adapter',False):return
    original_ds=DS.RatioDataSetTVResize
    class SeededDataset(original_ds):
        def __init__(self,config,mode,logger,seed=None,epoch=1,task='rec'):
            super().__init__(config,mode,logger,seed=seed,epoch=epoch,task=task)
            self.seed=int(config['Global']['seed'])+epoch-1
    DS.RatioDataSetTVResize=SeededDataset
    source=inspect.getsource(RS.RatioSampler.__iter__)
    assert source.count('random.seed(self.epoch)')==1
    import textwrap
    namespace=dict(RS.__dict__)
    exec(compile(textwrap.dedent(source).replace('random.seed(self.epoch)','random.seed(self.seed + self.epoch)'),str(Path(__file__)),'exec'),namespace)
    class SeededSampler(RS.RatioSampler):
        replication_seed_adapter=True
        __iter__=namespace['__iter__']
    RS.RatioSampler=SeededSampler

def make_trainer(out,seed=SEED):
    import numpy as np,torch
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer
    install_seed_adapter()
    cfg=Config(str(CONFIG));cfg.cfg['Global'].update(seed=seed,output_dir=str(out));cfg.cfg['Train']['sampler']['seed']=seed
    assert cfg.cfg['Global']['checkpoints'] is None and cfg.cfg['Global']['pretrained_model'] is None
    torch.set_num_threads(4);assert torch.cuda.is_available() and torch.cuda.device_count()==1
    t=Trainer(cfg,mode='train',task='rec')
    assert t.valid_dataloader is None and len(t.train_dataloader)==782 and len(t.train_dataloader.dataset)==100000 and not t.train_dataloader.dataset.need_reset
    generators=[]
    for i,m in enumerate(t.model.modules()):
        if isinstance(getattr(m,'rng',None),np.random.Generator):
            m.rng=np.random.default_rng(seed+i);generators.append(dict(module_index=i,seed=seed+i))
    assert generators
    t.replication_generators=generators
    return t

def close_trainer(t):
    for v in t.train_dataloader.dataset.lmdb_sets.values():v['env'].close()

def state(t):
    from capri_controller.training.darkbay_overnight_v1.engine import to_cpu
    return to_cpu(dict(model=t.model.state_dict(),optimizer=t.optimizer.state_dict(),scheduler=t.lr_scheduler.state_dict(),scaler=t.scaler.state_dict(),rng=B.rng_state(t.model),loader=B.loader_state(t)))

def exact(a,b):
    from capri_controller.training.darkbay_overnight_v1.engine import exact as compare
    return compare(a,b)

def batch_sha(batch):
    return hashlib.sha256(b''.join(x.numpy().tobytes() for x in batch)).hexdigest()

def readiness():
    import torch,numpy as np
    from capri_controller.training import ab_curriculum as C
    verify(data=True);stop();folder=STUDY/'readiness';folder.mkdir(exist_ok=True)
    assert not (folder/'receipt.json').exists()
    with gpu_owner():
        trials=[];initial=None
        for fast in [False,True]:
            t=make_trainer(folder/('cached' if fast else 'baseline'))
            fingerprint=C.model_hash(t.model)
            if initial is None:initial=fingerprint
            else:assert fingerprint==initial
            ls=B.loader_state(t);rs=B.rng_state(t.model);plans=[]
            for epoch in range(1,21):
                draws=list(iter(t.train_dataloader.batch_sampler));ids=[x[2] for b in draws for x in b]
                assert len(draws)==782 and all(len(b)==128 for b in draws) and len(ids)==100096 and len(set(ids))==100000
                plans.append(hashlib.sha256(np.asarray(draws,dtype=np.int64).tobytes()).hexdigest())
            B.restore_loader(t,ls);B.restore_rng(t.model,rs)
            assert t.train_dataloader.batch_sampler.seed==123 and t.train_dataloader.batch_sampler.epoch==0
            losses=[];batches=[];start=time.monotonic();iterator=iter(t.train_dataloader)
            for i in range(24):
                stop();batch=next(iterator);assert list(batch[0].shape)==[128,3,32,128]
                batches.append(batch_sha(batch));losses.append(B.update_step(t,batch))
                if not fast:torch.cuda.synchronize();torch.cuda.empty_cache();time.sleep(.005)
            torch.cuda.synchronize();seconds=time.monotonic()-start
            del iterator
            record=dict(fast=fast,seconds=seconds,losses=losses,batch_hashes=batches,epoch_draw_hashes=plans)
            if not fast:
                B.save_atomic(folder/'baseline24.pth',state(t))
            else:
                expected=torch.load(folder/'baseline24.pth',map_location='cpu',weights_only=False);exact(expected,state(t));del expected
                assert losses==trials[0]['losses'] and batches==trials[0]['batch_hashes'] and plans==trials[0]['epoch_draw_hashes']
                B.save_atomic(folder/'recovery_probe.pth',B.capture(t,1,782,{}))
                # Save at an epoch boundary-shaped state, then verify the next
                # iterator/steps in a newly constructed trainer after restore.
                iterator=iter(t.train_dataloader);next_hashes=[];next_losses=[]
                for i in range(3):
                    batch=next(iterator);next_hashes.append(batch_sha(batch));next_losses.append(B.update_step(t,batch))
                del iterator
                B.save_atomic(folder/'recovery_expected.pth',state(t))
                generators=t.replication_generators
            trials.append(record);close_trainer(t);del t;gc.collect();torch.cuda.empty_cache()
        t=make_trainer(folder/'restored')
        saved=torch.load(folder/'recovery_probe.pth',map_location='cpu',weights_only=False)
        assert all(k in saved for k in ['optimizer','scheduler','scaler','rng','loader','state_dict'])
        B.restore(t,saved);del saved
        iterator=iter(t.train_dataloader)
        for i in range(3):
            batch=next(iterator);assert batch_sha(batch)==next_hashes[i];assert B.update_step(t,batch)==next_losses[i]
        del iterator
        expected=torch.load(folder/'recovery_expected.pth',map_location='cpu',weights_only=False);exact(expected,state(t));del expected
        close_trainer(t);del t;gc.collect();torch.cuda.empty_cache()
        t=make_trainer(folder/'seed48_control',seed=48)
        assert C.model_hash(t.model)!=initial
        assert C.model_hash(t.model)==read(OLD/'model1/readiness/A/receipt.json')['initial_tensor_sha256']
        control=list(iter(t.train_dataloader.batch_sampler))
        assert hashlib.sha256(np.asarray(control,dtype=np.int64).tobytes()).hexdigest()!=trials[0]['epoch_draw_hashes'][0]
        close_trainer(t);del t;gc.collect();torch.cuda.empty_cache()
    write(folder/'receipt.json',dict(passed=True,at=now(),protocol_sha256=sha(STUDY/'protocol.json'),initial_tensor_sha256=initial,trials=trials,full_state_recovery_exact=True,recovery_next_batches=3,all20_epoch_budgets_verified=True,seed48_initialization_and_draws_different=True,module_rng_seeds=generators,production_weights_used=False,recovery_probe_is_disposable=True))

class StopLoader:
    def __init__(self,inner):self.inner=inner
    def __len__(self):return len(self.inner)
    def __getattr__(self,name):return getattr(self.inner,name)
    def __iter__(self):
        for batch in self.inner:
            stop();yield batch

def worker(resume=False):
    import torch,psutil
    from tools.engine import trainer as engine
    from capri_controller.training import ab_curriculum as C
    verify(data=True);gate=read(STUDY/'readiness/receipt.json');assert gate['passed'] and gate['protocol_sha256']==sha(STUDY/'protocol.json')
    with gpu_owner(),StageLock(OUT,'training'):
        assert resume or not (OUT/'latest.pth').exists()
        t=make_trainer(OUT);initial=C.model_hash(t.model);assert initial==gate['initial_tensor_sha256']
        if resume:
            r=read(OUT/'latest_receipt.json');assert sha(OUT/'latest.pth')==r['sha256'];B.restore(t,torch.load(OUT/'latest.pth',map_location='cpu',weights_only=False))
        t.train_dataloader=StopLoader(t.train_dataloader)
        write(OUT/'launch.json',dict(at=now(),pid=os.getpid(),process_created=psutil.Process().create_time(),resume=resume,scratch=not resume,initial_tensor_sha256=initial,seed=SEED,protocol_sha256=sha(STUDY/'protocol.json')))
        def save_hook(model,cfg,optimizer,lr_scheduler,epoch,global_step,metrics,is_best=False,logger=None,prefix=None):
            assert not is_best and global_step==epoch*782
            if prefix is not None:return
            stop();verify();archive=OUT/f'epoch_{epoch}.pth';assert not archive.exists()
            B.save_atomic(archive,B.capture(t,epoch,global_step,metrics));tmp=OUT/'latest.copying';shutil.copyfile(archive,tmp);os.replace(tmp,OUT/'latest.pth')
            r=dict(epoch=epoch,global_step=global_step,checkpoint=archive.name,sha256=sha(archive),protocol_sha256=sha(STUDY/'protocol.json'),committed_at=now())
            write(OUT/f'epoch_{epoch}_receipt.json',r);write(OUT/'latest_receipt.json',r)
        engine.save_ckpt=save_hook
        count=[t.status.get('global_step',0)]
        def after_step(opt,args,kwargs):
            count[0]+=1
            if count[0]==1 or count[0]%25==0:
                write(OUT/'progress.json',dict(at=now(),pid=os.getpid(),process_created=psutil.Process().create_time(),optimizer_steps=count[0],epoch=(count[0]-1)//782+1,step_in_epoch=(count[0]-1)%782+1))
        t.optimizer.register_step_post_hook(after_step);start=time.monotonic();t.train()
        r=read(OUT/'latest_receipt.json');assert r['epoch']==20 and r['global_step']==15640
        write(OUT/'training_completion.json',dict(**r,training_seconds=time.monotonic()-start,selection='fixed epoch20',real_labels_used=False))

def evaluate():
    import numpy as np,torch
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    verify();RESULTS.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=read(ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json');assert len(rows)==len({r['opaque_id'] for r in rows})==900
    assert not {r['pixel_sha256'] for r in rows}&{r['pixel_sha256'] for r in read(OLD/'targets.json')['fit']}
    transform=T.Compose([T.Resize((32,128),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
    images=[]
    for r in rows:
        im=Image.open(ROOT/r['image_path']).convert('RGB');assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256'];images.append(transform(im))
    summaries=[];start_all=time.monotonic()
    with gpu_owner(),StageLock(STUDY,'evaluation'):
        for epoch in range(1,21):
            stop();r=read(OUT/f'epoch_{epoch}_receipt.json');path=OUT/r['checkpoint'];assert sha(path)==r['sha256'];dest=RESULTS/f'epoch_{epoch:02d}'
            if (dest/'summary.json').exists():
                s=read(dest/'summary.json');assert s['checkpoint_sha256']==r['sha256'] and sha(dest/'predictions.json')==read(dest/'prediction_freeze.json')['sha256'];summaries.append(s);continue
            assert not dest.exists(),'Preserve partial evaluation for diagnosed recovery'
            dest.mkdir();ck=torch.load(path,map_location='cpu',weights_only=False)
            assert ck['epoch']==epoch and ck['global_step']==epoch*782 and ck['protocol_sha256']==sha(STUDY/'protocol.json') and ck['config']['Global']['seed']==123
            cfg=copy.deepcopy(ck['config']);decoder=build_post_process(cfg['PostProcess'],cfg['Global']);cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
            model=build_model(cfg['Architecture']);model.load_state_dict(ck['state_dict'],strict=True);model.cuda().eval();del ck;gc.collect()
            values=[];start=time.monotonic()
            # Audit gate enforces label isolation until predictions are durable.
            EVAL_LABEL_GATE[0]=False
            with torch.inference_mode():
                for i in range(0,900,4):
                    stop();pred=decoder(model(torch.stack(images[i:i+4]).cuda()))
                    values.extend(dict(opaque_id=x['opaque_id'],prediction=p[0]) for x,p in zip(rows[i:i+4],pred))
                    write(dest/'status.json',dict(state='predicting',processed=len(values),at=now()))
            assert len(values)==900 and sha(path)==r['sha256']
            write(dest/'predictions.json',values);write(dest/'prediction_freeze.json',dict(sha256=sha(dest/'predictions.json'),checkpoint_sha256=r['sha256'],frozen_at=now()))
            EVAL_LABEL_GATE[0]=True
            labels=read(ROOT/'evaluation_data/darkbay/scoring_labels/scoring_labels.json');assert set(labels)=={v['opaque_id'] for v in values}
            correct=sum(v['prediction']==labels[v['opaque_id']] for v in values);del labels
            base=ROOT/f'model1/results/darkbay_v3_feature_20261115/A/epoch_{epoch:02d}'
            baseline=read(base/'summary.json');assert read(base/'verification.json')['passed'] and sha(base/'predictions.json')==read(base/'prediction_freeze.json')['sha256']
            oldr=read(ROOT/f'model1/runs/darkbay_v3_feature_20261115/A/epoch_{epoch}_receipt.json');assert oldr['sha256']==baseline['checkpoint_sha256']
            s=dict(epoch=epoch,seed=123,correct=correct,total=900,accuracy_percent=correct/9,seed48_correct=baseline['correct'],seed48_accuracy_percent=baseline['correct']/9,difference_percentage_points=(correct-baseline['correct'])/9,checkpoint_sha256=r['sha256'],evaluation_seconds=time.monotonic()-start,primary=epoch==20)
            assert sha(dest/'predictions.json')==read(dest/'prediction_freeze.json')['sha256']
            write(dest/'verification.json',dict(passed=True,predictions_frozen_before_labels=True,all900_pixel_hashes_verified=True,reference100_excluded=True,checkpoint_hash_verified=True,precision='IEEE FP32; TF32 disabled',batch=4))
            write(dest/'summary.json',s);summaries.append(s);write(STUDY/'results.json',dict(complete=False,epochs=summaries))
            del model;gc.collect();torch.cuda.empty_cache()
    assert len(summaries)==20
    write(STUDY/'results.json',dict(complete=True,primary_epoch=20,epochs=summaries,evaluation_seconds=time.monotonic()-start_all,training_seconds=read(OUT/'training_completion.json')['training_seconds']))
    lines=['# Seed123 Model1-A replication','','Fixed epoch20 primary; earlier peaks retrospective. Same historical real900.','','| Epoch | Seed123 correct/900 | Accuracy | Seed48 correct/900 | Difference (pp) |','|---:|---:|---:|---:|---:|']
    lines += [f"| {r['epoch']} | {r['correct']} | {r['accuracy_percent']:.4f}% | {r['seed48_correct']} | {r['difference_percentage_points']:+.4f} |" for r in summaries]
    (STUDY/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

EVAL_LABEL_GATE=[False]
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0]).replace('\\','/').lower()
        if 'scoring_labels' in p and not EVAL_LABEL_GATE[0]:raise RuntimeError('Real labels forbidden before prediction freeze')
        if CURRENT_STAGE[0] in ('readiness','worker') and ('/evaluation_data/' in p and '/split_manifests/' not in p or '/results/' in p):raise RuntimeError('Real evaluation pixels/labels/results forbidden in training/readiness')
CURRENT_STAGE=['']

def child(stage,extra=()):
    import psutil
    stop();cmd=[sys.executable,'-B','-u',str(ENTRY),stage,*extra]
    with (STUDY/f'{stage}.log').open('a',encoding='utf8') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    identity=dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),command=cmd,at=now())
    write(STUDY/f'{stage}_launch.json',identity)
    while p.poll() is None:
        write(STUDY/'status.json',dict(state=stage,at=now(),child=identity,readiness_passed=(STUDY/'readiness/receipt.json').exists()))
        time.sleep(5)
    write(STUDY/f'{stage}_exit.json',dict(**identity,exit_code=p.returncode,exited_at=now()))
    assert p.returncode==0, f'{stage} failed; inspect logs, no blind retry'

def coordinate():
    import psutil
    with StageLock(STUDY,'coordinator'):
        try:
            while any(r['live'] for r in identities()) or gpu_owners():
                stop();write(STUDY/'status.json',dict(state='queued',at=now(),old_study=identities(),gpu_owners=gpu_owners(),readiness_passed=False));time.sleep(15)
            if not (STUDY/'readiness/receipt.json').exists():child('readiness')
            if not (OUT/'training_completion.json').exists():child('worker')
            if not (STUDY/'results.json').exists() or not read(STUDY/'results.json')['complete']:child('evaluate')
            write(STUDY/'completion.json',dict(at=now(),training=read(OUT/'training_completion.json'),evaluations=20,primary_epoch=20,protocol_sha256=sha(STUDY/'protocol.json')))
            write(STUDY/'status.json',dict(state='complete',at=now()))
        except BaseException:
            write(STUDY/'failure.json',dict(at=now(),error=traceback.format_exc(),STOP_present=(STUDY/'STOP').exists(),action='Diagnose; do not blindly retry or resume historical work'));raise

def main():
    environment();p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','readiness','worker','evaluate','coordinate','launch','inventory']);p.add_argument('--resume',action='store_true');a=p.parse_args();CURRENT_STAGE[0]=a.stage
    sys.addaudithook(audit)
    if a.stage=='prepare':return prepare()
    if a.stage=='inventory':
        from capri_controller.inventory import refresh
        return refresh()
    if a.stage=='launch':
        import psutil
        verify();stop();lock=STUDY/'coordinator.lock.json'
        if lock.exists():
            r=read(lock)
            try:assert abs(psutil.Process(r['pid']).create_time()-r['process_created'])>=.02,'Coordinator already active'
            except psutil.NoSuchProcess:pass
        cmd=[sys.executable,'-B','-u',str(ENTRY),'coordinate']
        with (STUDY/'coordinator.log').open('a',encoding='utf8') as log:q=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        r=dict(pid=q.pid,process_created=psutil.Process(q.pid).create_time(),command=cmd,at=now());write(STUDY/'coordinator_launch.json',r);print(json.dumps(r));return
    if a.stage=='worker':return worker(a.resume)
    return globals()[a.stage]()

if __name__=='__main__':main()
