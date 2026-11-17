"""Model3-A: frozen teacher buckets, reference priorities and mastery feedback."""
import argparse
import contextlib
import copy
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from .. import ab_curriculum as C
from . import common as W
from ..cryptbb_feature_v1_sampler import FeatureSampler, POLICY, self_test
from . import metadata as META
from ...decoder.common import StageLock

ROOT=W.ROOT
TAG="A"
TEACHER=None
FAMILY='cryptbb_v3_feature_priority_v1_20261115'
STUDY=ROOT/'handoff'/FAMILY
OUT=ROOT/'model3/runs'/FAMILY/'A'
LAUNCHER=ROOT/'model3/src/feature_priority_v1/capri.py'
BASE=ROOT/'handoff/cryptbb_v3_epoch20_static_A_20261115'
read,sha,now=C.read,C.sha,C.now
_write=C.write


def write(path,value):
    for attempt in range(50):
        try:return _write(path,value)
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)


class Stopped(RuntimeError):pass
def stop_check():
    if (STUDY/'STOP').exists():raise Stopped('STOP requested; no automatic restart')


def no_labels(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower().replace('\\','/')
        if any(s in p for s in ('/evaluation_data/','/model1/results/','/model2/results/','scoring_only/','evaluation_pool_labels','per_image_results.json')):
            raise RuntimeError('Real evaluation pixels and labels forbidden in Model3 training')


def config(out):
    import yaml
    cfg=yaml.safe_load((ROOT/'model1/configs/cryptbb_v3/A/prepared.yml').read_text())
    cfg['Global'].update(output_dir=str(out),save_res_path=str(out/'unused.txt'))
    assert cfg['Global']['pretrained_model'] is None and cfg['Global']['checkpoints'] is None
    assert cfg['Train']['loader']['min_ratio']==cfg['Train']['loader']['max_ratio']==4
    return cfg


def dataset():
    import lmdb,numpy as np
    from .ds2_data import ReaderDataset
    env=lmdb.open(str(ROOT/'datasets/cryptbb_v3/A/train_lmdb'),readonly=True,lock=False,readahead=False)
    with env.begin() as tx:assert int(tx.get(b'num-samples'))==100000
    return ReaderDataset(env,np.arange(1,100001),ROOT/'datasets/cryptbb_v3/charset.txt')


def load_features():
    import numpy as np
    meta=read(STUDY/'features/manifest.json')
    path=STUDY/'features/features.npz';assert sha(path)==meta['features_sha256']
    with np.load(path,allow_pickle=False) as z:
        return FeatureSampler(z['buckets'],z['groups'],meta),z['truth'].copy()


def check_sources():
    p=read(STUDY/'protocol.json')
    for r in p['sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert sha(STUDY/'features/manifest.json')==p['feature_manifest_sha256']
    return sha(STUDY/'protocol.json')


@contextlib.contextmanager
def gpu_owner():
    import msvcrt
    path=ROOT/'runtime/model3_feature_gpu.lock'
    with path.open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


class OrderedBatches:
    def __init__(self,ds,indices,batch_size):
        self.ds=ds;self.indices=indices;self.batch_size=batch_size
    def __iter__(self):
        assert self.ds.aug is None
        # One lookahead batch; CPU preprocessing overlaps the current GPU step.
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='model3_prefetch') as pool:
            chunks=[self.indices[i:i+self.batch_size] for i in range(0,len(self.indices),self.batch_size)]
            pending=pool.submit(C.batch_at,self.ds,chunks[0])
            for i in range(len(chunks)):
                batch=pending.result()
                if i+1<len(chunks):pending=pool.submit(C.batch_at,self.ds,chunks[i+1])
                yield batch


def predict(model,decoder,ds,indices,batch_size=256,progress=None):
    import torch
    predictions=[];rng=C.rng_state(model);was_training=model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in OrderedBatches(ds,indices,batch_size):
                stop_check()
                with W.gpu_slot():
                    memory=model.encoder(batch[0].cuda())
                    output=model.decoder.forward_test(memory)
                    assert torch.isfinite(output).all()
                    predictions.extend(v[0] for v in decoder(output))
                    del memory,output
                if progress and (len(predictions)%6400==0 or len(predictions)==len(indices)):
                    progress(len(predictions))
                del batch
    finally:
        C.restore_rng(model,rng);model.train(was_training)
    assert len(predictions)==len(indices)
    return predictions


def assess(model,decoder,ds,truth,sm,epoch):
    import numpy as np
    dest=OUT/f'assessments/epoch_{epoch:02d}';dest.mkdir(parents=True,exist_ok=False)
    before=C.model_hash(model);start=time.monotonic()
    def progress(n):
        write(OUT/'status.json',dict(state='assessing',epoch=epoch,completed_epoch=epoch-1,
              trained_epoch=epoch,global_step=epoch*782,processed=n,total=100000,seconds=time.monotonic()-start,updated_at=now()))
    progress(0)
    predictions=predict(model,decoder,ds,np.arange(100000),progress=progress)
    assert C.model_hash(model)==before,'Assessment modified weights/buffers'
    write(dest/'predictions.json',[dict(index=i+1,prediction=p) for i,p in enumerate(predictions)])
    correct=np.asarray(predictions)==truth
    np.save(dest/'correct.npy',correct)
    feedback=sm.update(correct,epoch) if epoch>=2 else dict(completed_epoch=epoch,overall_accuracy=float(correct.mean()),adaptation=False)
    feedback.update(seconds=time.monotonic()-start,predictions_sha256=sha(dest/'predictions.json'),
                    correct_sha256=sha(dest/'correct.npy'),model_tensor_sha256=before)
    write(dest/'feedback.json',feedback)
    return feedback


def to_cpu(v):
    import torch
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:to_cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [to_cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(to_cpu(x) for x in v)
    return copy.deepcopy(v)


def exact(a,b):
    import numpy as np,torch
    if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def restore(path,model,opt,sched,scaler,sm,digest):
    import torch
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['protocol_sha256']==digest
    model.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer'])
    sched.load_state_dict(ck['lr_scheduler']);scaler.load_state_dict(ck['scaler'])
    sm.load_state_dict(ck['adaptive']);C.restore_rng(model,ck['rng'])
    assert sm.epoch==ck['epoch']+1
    return ck


def verify_idle():
    import psutil
    ancestors={p.pid for p in psutil.Process().parents()};conflicts=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=p.info['cmdline'] or [];joined=' '.join(cmd)
            if p.pid==os.getpid() or p.pid in ancestors:continue
            if 'python' not in (p.info['name'] or '').lower():continue
            if ('capri_controller.training.' in joined or 'capri.py' in joined or 'recover_b_once.py' in joined) and any(s in cmd for s in ('worker','coordinate','rank','check')):
                conflicts.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    assert not conflicts,conflicts
    write(STUDY/'idle_verified.json',dict(checked_at=now(),conflicts=conflicts))


def check():
    import numpy as np,torch
    digest=check_sources();verify_idle();stop_check();C.bootstrap();C.STUDY=STUDY
    try:sys.audit('open',str(ROOT/'evaluation_data/forbidden_probe'),'r',0)
    except RuntimeError:pass
    else:raise AssertionError('Real-data guard failed')
    baseline_hashes={r['path'].replace('\\','/'):r['sha256'] for r in read(BASE/'protocol.json')['sources']}
    for path in (Path(C.__file__),Path(C.__file__).with_name('ds2_data.py')):
        assert sha(path)==baseline_hashes[path.relative_to(ROOT).as_posix()]
    # Own the GPU for the entire pilot; keep cache between complete steps.
    with StageLock(STUDY,'readiness'),gpu_owner():
        C.gpu_slot=W.gpu_slot
        out=STUDY/'checks';out.mkdir(exist_ok=True);cfg=config(out);ds=dataset();sm,truth=load_features()
        with ds.env.begin() as tx:
            for i in range(100000):assert tx.get(f'label-{i+1:09d}'.encode()).decode()==truth[i]
        model,decoder=C.make_model(cfg);initial=C.model_hash(model)
        assert initial==read(BASE/'readiness.json')['datasets'][TAG]['initial_tensor_sha256']
        opt,sched,scaler=C.optimizer_for(model,cfg)
        d,_=sm.draw();batch=C.batch_at(ds,d[:128]);loss,_=C.step(model,opt,sched,scaler,batch)
        before=C.model_hash(model);rng=to_cpu(C.rng_state(model))
        a=predict(model,decoder,ds,np.arange(512),256)
        b=predict(model,decoder,ds,np.arange(512),64)
        again=predict(model,decoder,ds,np.arange(512),256)
        assert a==again and before==C.model_hash(model);exact(rng,C.rng_state(model))
        early_batch_differences=sum(x!=y for x,y in zip(a,b))
        ck=out/'resume_probe.pth';C.save_checkpoint(ck,model,opt,sched,scaler,sm,cfg,1,[],digest)
        d1,r1=sm.draw();batch=C.batch_at(ds,d1[:128]);loss1,norm1=C.step(model,opt,sched,scaler,batch)
        expected=to_cpu(dict(model=model.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),scaler=scaler.state_dict(),rng=C.rng_state(model),sampler=sm.state_dict()))
        del model,decoder,opt,sched,scaler;gc.collect();torch.cuda.empty_cache()
        model,decoder=C.make_model(cfg);opt,sched,scaler=C.optimizer_for(model,cfg);sm,_=load_features()
        restore(ck,model,opt,sched,scaler,sm,digest)
        d2,r2=sm.draw();assert np.array_equal(d1,d2) and r1==r2
        fast_batch=next(iter(OrderedBatches(ds,d2[:128],128)))
        exact(batch,fast_batch);loss2,norm2=C.step(model,opt,sched,scaler,fast_batch)
        assert loss1==loss2 and norm1==norm2
        exact(expected,dict(model=model.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),scaler=scaler.state_dict(),rng=C.rng_state(model),sampler=sm.state_dict()))
        # Exercise production-sized adaptive state and checkpoint serialization.
        correct=np.asarray([(i%4)!=0 for i in range(100000)],bool)
        sm.update(correct,2);saved=sm.state_dict();d3,r3=sm.draw();replay,_=load_features();replay.load_state_dict(saved)
        q,qr=replay.draw();assert np.array_equal(d3,q) and r3==qr
        assert np.array_equal(sm.buckets,sm.original_buckets)
        # Verify scoring batch-size parity on a mature reader, not only a nearly
        # untrained pilot whose outputs might all be empty.
        teacher=TEACHER
        assert sha(teacher)==read(STUDY/'features/manifest.json')['ranking']['teacher_sha256']
        mature=torch.load(teacher,map_location='cpu',weights_only=False)
        model.load_state_dict(mature['state_dict'],strict=True);del mature
        probe=np.linspace(0,99999,512,dtype=np.int64)
        mature256=predict(model,decoder,ds,probe,256)
        mature64=predict(model,decoder,ds,probe,64)
        assert mature256==mature64
        assert len(set(mature256))>100,'Probe should have diverse learned outputs'
        write(STUDY/'readiness.json',dict(passed=True,protocol_sha256=digest,checked_at=now(),initial_tensor_sha256=initial,
            all100k_synthetic_labels_match=True,policy_tests=self_test(),warmup_draws_exact=True,
            fixed256_inference_repeats_exactly=True,early_256_vs64_string_differences=early_batch_differences,
            mature_inference256_matches64_on512_images=True,
            mature_probe_unique_strings=len(set(mature256)),real_data_guard_tested=True,baseline_step_and_preprocessing_source_hashes_match=True,
            inference_preserves_weights_and_all_rng=True,
            full128_resume_weights_optimizer_scheduler_scaler_rng_sampler_exact=True,production_size_adaptive_replay=True,
            losses=[loss,loss1,loss2],peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,pilot_weights_disposable=True))
        ds.env.close()
    print('Model3-A readiness passed',flush=True)


def worker(resume=False):
    import numpy as np,torch
    digest=check_sources();ready=read(STUDY/'readiness.json')
    assert ready['passed'] and ready['protocol_sha256']==digest
    stop_check();C.bootstrap();C.STUDY=STUDY;C.gpu_slot=W.gpu_slot
    assert resume or not OUT.exists(),'No automatic restart'
    OUT.mkdir(parents=True,exist_ok=True)
    with StageLock(STUDY,'worker_A'),gpu_owner():
        cfg=config(OUT);sm,truth=load_features();ds=dataset();model,decoder=C.make_model(cfg)
        opt,sched,scaler=C.optimizer_for(model,cfg);history=[];first=1
        if resume:
            receipts=sorted((OUT/'checkpoints').glob('epoch_*.json'));assert receipts
            rec=read(receipts[-1]);path=OUT/rec['checkpoint'];assert sha(path)==rec['sha256']
            ck=restore(path,model,opt,sched,scaler,sm,digest);history=ck['history'];first=ck['epoch']+1;del ck
            # Only explicit resume archives uncommitted next-epoch products.
            recovery=OUT/'recovery'/str(time.time_ns())
            for path in [OUT/f'assessments/epoch_{first:02d}',OUT/f'sampling_epoch_{first:02d}.json',OUT/f'draws_epoch_{first:02d}.npy',OUT/f'checkpoints/epoch_{first:02d}.pth']:
                if path.exists():recovery.mkdir(parents=True,exist_ok=True);shutil.move(str(path),str(recovery/path.name))
        else:
            assert C.model_hash(model)==ready['initial_tensor_sha256']
            write(OUT/'initialization.json',dict(pretrained=False,seed=48,model_tensor_sha256=ready['initial_tensor_sha256'],protocol_sha256=digest))
        try:
            for epoch in range(first,21):
                assert digest==check_sources();stop_check();start=time.monotonic();losses=[]
                draws,receipt=sm.draw()
                if epoch<=2:
                    baseline=np.load(BASE/f'draws/{TAG}_epochs01_20.npy',mmap_mode='r')
                    assert np.array_equal(sm.ids[draws],baseline[epoch-1])
                np.save(OUT/f'draws_epoch_{epoch:02d}.npy',sm.ids[draws]);write(OUT/f'sampling_epoch_{epoch:02d}.json',receipt)
                for j,batch in enumerate(OrderedBatches(ds,draws,128),1):
                    stop_check();loss,norm=C.step(model,opt,sched,scaler,batch);losses.append(loss)
                    if j==1 or j%25==0:
                        status=dict(state='training',epoch=epoch,completed_epoch=epoch-1,batch=j,global_step=(epoch-1)*782+j,
                            loss=loss,gradient_norm=norm,epoch_seconds=time.monotonic()-start,mass=receipt['mass'],updated_at=now())
                        write(OUT/'status.json',status);print(json.dumps(status),flush=True)
                assert len(losses)==782
                training_seconds=time.monotonic()-start
                feedback=assess(model,decoder,ds,truth,sm,epoch)
                history.append(dict(epoch=epoch,mean_loss=float(np.mean(losses)),training_seconds=training_seconds,
                    assessment_seconds=feedback['seconds'],synthetic_accuracy=feedback['overall_accuracy'],sampling=receipt,
                    active_next=feedback.get('active_families',[]),next_mass=feedback.get('next_mass',[.5,.5,0.])))
                path=OUT/f'checkpoints/epoch_{epoch:02d}.pth';assert not path.exists()
                C.save_checkpoint(path,model,opt,sched,scaler,sm,cfg,epoch,history,digest)
                write(path.with_suffix('.json'),dict(epoch=epoch,global_step=epoch*782,checkpoint=str(path.relative_to(OUT)),
                    sha256=sha(path),protocol_sha256=digest,assessment_sha256=sha(OUT/f'assessments/epoch_{epoch:02d}/feedback.json'),committed_at=now()))
                write(OUT/'history.json',history)
                write(OUT/'status.json',dict(state='epoch_complete',completed_epoch=epoch,global_step=epoch*782,updated_at=now()))
            completion=read(OUT/'checkpoints/epoch_20.json')
            completion.update(completed_at=now(),selection='Fixed final epoch20',training_seconds=sum(r['training_seconds'] for r in history),
                              assessment_seconds=sum(r['assessment_seconds'] for r in history),real_labels_read=False)
            write(OUT/'completion.json',completion)
        except BaseException:
            write(OUT/'failure.json',dict(updated_at=now(),completed_epoch=len(history),error=traceback.format_exc(),automatic_restart=False))
            raise
        finally:ds.env.close()


