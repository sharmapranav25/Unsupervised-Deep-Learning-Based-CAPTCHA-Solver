"""Frozen epoch20-teacher static B CryptBB experiment; separate from historical adapters.

Entry: python -B -m capri_controller.training.cryptbb_epoch20_static_b STAGE
Stages setup/check/worker/coordinate/evaluate/report; STOP is checked each batch.
"""
import argparse
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
from ..decoder.common import read, sha, now, StageLock
from . import ab_curriculum as C
from . import cryptbb_v3_model2_selected_v1 as OLD

ROOT = Path(__file__).resolve().parents[4]
FAMILY = 'cryptbb_v3_epoch20_static_B_20261115'
STUDY = ROOT / 'handoff' / FAMILY
MODULE = 'capri_controller.training.cryptbb_epoch20_static_b'
original_draws = C.draws_for
config, dataset = OLD.config, OLD.dataset

def rank_dir(tag):
    assert tag == 'B'
    return ROOT/'model2/rankings'/FAMILY/tag
_atomic_write = C.write
def write(path,value):
    for attempt in range(50):
        try: return _atomic_write(path,value)
        except PermissionError:
            if attempt==49: raise
            time.sleep(.1)



class Stopped(RuntimeError):
    pass


def stop_check():
    if (STUDY/'STOP').exists():
        raise Stopped('User STOP; preserve partial work and resume only with explicit authorization')


def run_dir(number, tag):
    assert number == 2 and tag == 'B'
    return ROOT/'model2/runs'/FAMILY/tag


def result_dir(tag, epoch):
    return ROOT/'model2/results'/FAMILY/tag/f'epoch_{epoch:02d}'


def check_sources():
    protocol = read(STUDY/'protocol.json')
    for row in protocol['sources']:
        assert sha(ROOT/row['path']) == row['sha256'], row['path']
    return sha(STUDY/'protocol.json')


def draws_for(sampler, number, epoch, count=C.DRAWS):
    return original_draws(sampler, number, epoch, count)


def bind():
    C.write=write; C.STUDY=STUDY; C.config=config; C.dataset=dataset; C.rank_dir=rank_dir
    C.run_dir=run_dir; C.check_sources=check_sources; C.draws_for=draws_for
    # Use C's identical complete-step lock in the NEW study directory.


def no_labels(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0]).lower().replace('\\','/')
        if any(x in path for x in ('scoring_only/', 'evaluation_pool_labels', 'per_image_results.json')):
            raise RuntimeError('Real scoring labels forbidden in training/readiness/selection')


def setup():
    STUDY.mkdir(parents=True,exist_ok=True)
    assert not (STUDY/'protocol.json').exists(), 'Frozen study exists'
    assert not run_dir(2,'B').exists() and not rank_dir('B').exists()
    prior=ROOT/'handoff/cryptbb_v3_cumulative_static_v1_20261115'
    assert read(prior/'report_verification.json')['complete']
    for row in read(prior/'protocol.json')['sources']:
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
    assert read(ROOT/'handoff/cryptbb_v3_epoch20_static_A_20261115/report_verification.json')['complete']
    OLD.M1.check_data('B')
    ck=ROOT/'model1/runs/cryptbb_v3/A/epoch_20.pth'
    receipt=read(ck.parent/'epoch_20_receipt.json')
    assert receipt['epoch']==20 and receipt['global_step']==15640
    assert sha(ck)==receipt['sha256']=='a599a42d9a33e991cc7ae1f22ab92851bd3495e7d9f9fce517fdb1feb81cc8c3'
    paths={ROOT/r['path'] for r in read(prior/'protocol.json')['sources']}
    paths.update(Path(__file__).parent.glob('cryptbb_epoch20_static_b*.py'))
    sources=[]
    for path in sorted(paths):
        rel=path.relative_to(ROOT);dst=STUDY/'source'/rel
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
        sources.append(dict(path=rel.as_posix(),sha256=sha(path)))
    write(STUDY/'protocol.json',dict(created_at=now(),family=FAMILY,sources=sources,
        authorized_datasets=['B'],seed=48,epochs=20,batch=128,steps_per_epoch=782,
        draws_per_epoch=100096,total_steps=15640,pool_size=100000,
        teacher=dict(reader='Model1-A',epoch=20,checkpoint=str(ck.relative_to(ROOT)),sha256=receipt['sha256']),
        data_receipt=read(ROOT/'datasets/cryptbb_v3/B/data_receipt.json'),
        schedule={'1':[1,0,0],'2':[.5,.5,0],'3-20':[.15,.25,.60]},
        ranking='Original summed synthetic character NLL, free AR+one refinement; ordinal thirds with image-ID ties',
        initialization='Independent scratch seed48; teacher is ranking only',
        optimizer='Unchanged AdamW .00035/.05; OneCycle2/20; BF16; clip20',
        preprocessing='Unchanged RGB PIL bicubic32x128 ToTensor Normalize(.5,.5)',
        selection='Independent original v1 exact-string rank1 across all20; original choose_k/fallback5; earliest tie',
        real_labels_used=False,automatic_restarts=0,
        execution='One B GPU run; CPU evaluations; bounded5second JSON retry; retained CUDA cache and ordered single CPU-batch prefetch from epoch1, exact24step equivalence gated',
        comparison='DarkBay schedule and epoch20 cross-teacher. Prior CryptBB static used A12; no causal claim from one run.'))


def rank():
    import numpy as np,torch
    digest=check_sources();p=read(STUDY/'protocol.json');out=rank_dir('B')
    assert not out.exists(),'Preserve existing ranking, inspect before retry'
    out.mkdir(parents=True)
    with StageLock(STUDY,'ranking'),StageLock(STUDY,'training_worker'):
        ck=ROOT/p['teacher']['checkpoint'];assert sha(ck)==p['teacher']['sha256']
        model,decoder=C.make_model(config('B',out))
        state=torch.load(ck,map_location='cpu',weights_only=False)
        model.load_state_dict(state['state_dict'],strict=True);del state
        ds=dataset('B');original_batch=C.batch_at
        def stopped_batch(*args,**kwargs):
            stop_check();return original_batch(*args,**kwargs)
        C.batch_at=stopped_batch
        try:raw,seconds=C.score_pool(model,decoder,ds,out)
        finally:ds.env.close();C.batch_at=original_batch
        ids=np.arange(1,100001);buckets=np.empty(100000,dtype=np.int64)
        for b,group in enumerate(np.array_split(np.lexsort((ids,raw)),3)):buckets[group]=b
        assert np.isfinite(raw).all() and (raw>=0).all()
        np.savez_compressed(out/'ranking.npz',ids=ids,nll=raw,buckets=buckets)
        write(out/'completion.json',dict(dataset='B',teacher='A',teacher_epoch=20,
            images=100000,seconds=seconds,protocol_sha256=digest,teacher_sha256=sha(ck),
            sha256=sha(out/'ranking.npz'),predictions_sha256=sha(out/'predictions.json'),
            bucket_counts=np.bincount(buckets).tolist(),completed_at=now()))


def prepare_draws():
    import numpy as np
    digest=check_sources();out=rank_dir('B');done=read(out/'completion.json')
    assert done['protocol_sha256']==digest and done['images']==100000
    assert done['sha256']==sha(out/'ranking.npz') and done['predictions_sha256']==sha(out/'predictions.json')
    assert done['teacher_sha256']==read(STUDY/'protocol.json')['teacher']['sha256']
    assert [r['index'] for r in read(out/'predictions.json')]==list(range(1,100001))
    with np.load(out/'ranking.npz') as z:
        ids,raw,buckets=z['ids'],z['nll'],z['buckets']
        assert np.array_equal(ids,np.arange(1,100001)) and np.isfinite(raw).all() and (raw>=0).all()
        expected=np.empty(100000,dtype=np.int64)
        for b,g in enumerate(np.array_split(np.lexsort((ids,raw)),3)):expected[g]=b
        assert np.array_equal(expected,buckets) and np.bincount(buckets).tolist()==[33334,33333,33333]
    sampler=C.sampler_for('B');sequences=[];records=[]
    for epoch in range(1,21):
        before=copy.deepcopy(sampler);draws,receipt=draws_for(sampler,2,epoch)
        replay,again=original_draws(before,2,epoch)
        assert np.array_equal(draws,replay) and receipt==again
        assert len(draws)==100096 and len(draws)%128==0
        if epoch<=2:assert (sampler.buckets[draws]<epoch).all()
        else:assert receipt['draws']==[15014,25024,60058]
        sequences.append(sampler.ids[draws]);records.append(receipt)
    dest=STUDY/'draws';dest.mkdir(exist_ok=True)
    np.save(dest/'B_epochs01_20.npy',np.stack(sequences))
    write(STUDY/'draw_manifest.json',dict(passed=True,datasets={'B':dict(epochs=records,
        sha256=sha(dest/'B_epochs01_20.npy'))},ranking_sha256=done['sha256']))
    old=ROOT/'model2/rankings/cryptbb_v3_rank1_v1/B/ranking.npz'
    with np.load(old) as z:old_buckets=z['buckets']
    transition=np.zeros((3,3),dtype=int)
    for a,b in zip(old_buckets,buckets):transition[a,b]+=1
    write(STUDY/'ranking_comparison.json',dict(old_teacher_epoch=12,new_teacher_epoch=20,
        old_ranking_sha256=sha(old),new_ranking_sha256=done['sha256'],
        changed_bucket_count=int((old_buckets!=buckets).sum()),old_to_new_bucket_counts=transition.tolist(),
        diagnostic_only=True,real_labels_read=False))


def exact(a,b):
    import numpy as np
    import torch
    if torch.is_tensor(a): assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): exact(x,y)
    else: assert a==b


def check():
    import torch
    digest=check_sources(); stop_check()
    OLD.STUDY=STUDY
    OLD.extra_preprocessing_checks()
    out=STUDY/'checks'; out.mkdir(exist_ok=True)
    checks={}; initial=[]
    for tag in 'B':
        cfg=config(tag,out); ds=dataset(tag); sm=C.sampler_for(tag)
        model,decoder=C.make_model(cfg); opt,sched,scaler=C.optimizer_for(model,cfg)
        initial.append(C.model_hash(model)); draws_for(sm,2,1)
        batch=C.batch_at(ds,range(128)); loss,norm=C.step(model,opt,sched,scaler,batch)
        saved=out/f'{tag}_resume_probe.pth'
        C.save_checkpoint(saved,model,opt,sched,scaler,sm,cfg,1,[],digest)
        loss1,norm1=C.step(model,opt,sched,scaler,batch)
        expected=copy.deepcopy(dict(model={k:v.cpu() for k,v in model.state_dict().items()},
            optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),rng=C.rng_state(model)))
        # Store expected tensors on CPU so the second model does not exceed VRAM.
        for state in expected['optimizer']['state'].values():
            for k,v in state.items():
                if torch.is_tensor(v): state[k]=v.cpu()
        del model,opt,sched,scaler; gc.collect(); torch.cuda.empty_cache()
        model,decoder=C.make_model(cfg); assert C.model_hash(model)==initial[-1]
        opt,sched,scaler=C.optimizer_for(model,cfg); sm2=C.sampler_for(tag)
        ck=C.load_checkpoint(saved,model,opt,sched,scaler,sm2,digest); del ck
        exact(sm.state_dict(),sm2.state_dict())
        loss2,norm2=C.step(model,opt,sched,scaler,batch)
        exact(expected,dict(model=model.state_dict(),optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),scaler=scaler.state_dict(),rng=C.rng_state(model)))
        assert loss1==loss2 and norm1==norm2
        checks[tag]=dict(full_batch=128,loss=loss,gradient=norm,replay_loss=loss2,
            exact_all_weights_optimizer_scheduler_scaler_python_numpy_torch_cuda_module_rng=True,
            sampler_exact=True,initial_tensor_sha256=initial[-1],peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        ds.env.close(); del model,opt,sched,scaler,expected,batch; gc.collect(); torch.cuda.empty_cache()
    assert initial[0]=='143edd906a13f432b8722ab68dba98cdab4f5eab20fbaae3ac1e06aef7401e43'
    from .cryptbb_epoch20_static_b_checks import lifecycle_checks
    lifecycle=lifecycle_checks()
    write(STUDY/'readiness.json',dict(passed=True,created_at=now(),protocol_sha256=digest,
        datasets=checks,preprocessing=read(STUDY/'preprocessing_verification.json'),
        lifecycle=lifecycle,pilot_weights_disposable=True,real_labels_read=False))
    print('Readiness passed: exact full-batch checkpoint replay on B',flush=True)


def worker(tag,resume=False):
    import numpy as np
    import torch
    digest=check_sources(); ready=read(STUDY/'readiness.json')
    assert ready['passed'] and ready['protocol_sha256']==digest
    stop_check(); out=run_dir(2,tag)
    assert resume or not out.exists(), 'Existing output requires inspected explicit resume'
    out.mkdir(parents=True,exist_ok=True)
    with StageLock(STUDY,'worker_'+tag), StageLock(STUDY,'training_worker'):
        cfg=config(tag,out); sm=C.sampler_for(tag); ds=dataset(tag)
        model,decoder=C.make_model(cfg); opt,sched,scaler=C.optimizer_for(model,cfg)
        history=[]; first=1
        if resume:
            records=sorted((out/'checkpoints').glob('epoch_*.json'))
            assert records, 'No committed checkpoint to recover'
            receipt=read(records[-1]); path=out/receipt['checkpoint']
            assert sha(path)==receipt['sha256'] and receipt['protocol_sha256']==digest
            ck=C.load_checkpoint(path,model,opt,sched,scaler,sm,digest)
            history=ck['history']; first=ck['epoch']+1; del ck
        else:
            fingerprint=C.model_hash(model)
            assert fingerprint==ready['datasets'][tag]['initial_tensor_sha256']
            write(out/'initialization.json',dict(pretrained=False,seed=48,model_tensor_sha256=fingerprint,protocol_sha256=digest))
        expected=np.load(STUDY/f'draws/{tag}_epochs01_20.npy')
        assert sha(STUDY/f'draws/{tag}_epochs01_20.npy')==read(STUDY/'draw_manifest.json')['datasets'][tag]['sha256']
        try:
            for epoch in range(first,21):
                stop_check(); start=time.monotonic(); losses=[]
                draws,receipt=draws_for(sm,2,epoch)
                assert np.array_equal(sm.ids[draws],expected[epoch-1])
                write(out/f'sampling_epoch_{epoch:02d}.json',receipt)
                for batch_index,offset in enumerate(range(0,len(draws),128),1):
                    stop_check()
                    loss,norm=C.step(model,opt,sched,scaler,C.batch_at(ds,draws[offset:offset+128]))
                    losses.append(loss)
                    if batch_index==1 or batch_index%25==0:
                        status=dict(state='training',dataset=tag,epoch=epoch,completed_epoch=epoch-1,
                            batch=batch_index,global_step=(epoch-1)*782+batch_index,loss=loss,gradient_norm=norm,
                            epoch_seconds=time.monotonic()-start,updated_at=now())
                        write(out/'status.json',status); print(json.dumps(status),flush=True)
                history.append(dict(epoch=epoch,mean_loss=float(np.mean(losses)),training_wall_seconds=time.monotonic()-start))
                archive=out/f'checkpoints/epoch_{epoch:02d}.pth'
                assert not archive.exists(), 'Immutable checkpoint archive exists; inspect before recovery'
                C.save_checkpoint(archive,model,opt,sched,scaler,sm,cfg,epoch,history,digest)
                write(archive.with_suffix('.json'),dict(epoch=epoch,global_step=epoch*782,
                    checkpoint=str(archive.relative_to(out)),sha256=sha(archive),protocol_sha256=digest,committed_at=now()))
                write(out/'history.json',history)
                write(out/'status.json',dict(state='epoch_complete',completed_epoch=epoch,global_step=epoch*782,updated_at=now()))
            last=out/'checkpoints/epoch_20.json'; done=read(last)
            done.update(dataset=tag,training_wall_seconds=sum(r['training_wall_seconds'] for r in history),
                selection='Pending independent original v1 selection',completed_at=now())
            write(out/'completion.json',done)
        except Stopped:
            write(out/'stop_receipt.json',dict(stopped_at=now(),completed_epoch=len(history),
                note='Partial epoch discarded; immutable committed checkpoints preserved'))
            raise
        finally: ds.env.close()


def child(stage,tag=None,extra=()):
    import psutil
    args=[sys.executable,'-B','-u','-X','faulthandler','-m',MODULE,stage]
    if tag: args+=['--dataset',tag]
    args+=list(extra)
    stem=stage+('_'+tag if tag else '')
    with (STUDY/(stem+'.log')).open('a',encoding='utf8') as stream:
        p=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    write(STUDY/(stem+'_launch.json'),dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),args=args,created_at=now()))
    return p


def valid_completion(tag):
    done=read(run_dir(2,tag)/'completion.json')
    assert done['epoch']==20 and done['global_step']==15640
    assert done['protocol_sha256']==check_sources()
    assert sha(run_dir(2,tag)/done['checkpoint'])==done['sha256']
    assert [r['epoch'] for r in read(run_dir(2,tag)/'history.json')]==list(range(1,21))
    return done


def supervise(tag,p,on_tick=lambda:None):
    first=None; verified=False
    while p.poll() is None:
        stop_check(); on_tick();observe('training')
        status=run_dir(2,tag)/'status.json'
        if status.exists():
            row=read(status)
            if row.get('global_step',0)>0 and first is None: first=row
            if first and row.get('global_step',0)>first['global_step'] and not verified:
                import psutil
                identity=read(STUDY/f'worker_{tag}.lock.json')
                process=psutil.Process(identity['pid'])
                assert abs(process.create_time()-identity['process_created'])<.1
                assert MODULE in process.cmdline() and 'worker' in process.cmdline()
                write(STUDY/f'launch_verified_{tag}.json',dict(verified_at=now(),worker=identity,
                    wrapper=read(STUDY/f'worker_{tag}_launch.json'),first=first,progress=row,
                    protocol_sha256=check_sources(),scratch=read(run_dir(2,tag)/'initialization.json')))
                if tag=='B': shutil.copyfile(STUDY/'launch_verified_B.json',STUDY/'launch_verified.json')
                verified=True
        time.sleep(5)
    stop_check()
    assert p.returncode==0, f'{tag} worker failed: {p.returncode}'
    return valid_completion(tag)


def coordinate():
    with StageLock(STUDY,'coordinator'):
        check_sources();stop_check();verify_idle()
        assert not run_dir(2,'B').exists(), 'No automatic restarts'
        awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        assert awake
        write(STUDY/'awake.json',dict(requested_at=now(),system_required=True,previous_state=awake))
        evaluator=None;active=None
        try:
            for stage in ['rank','draws','check','fastcheck']:
                write(STUDY/'status.json',dict(state=stage,updated_at=now()))
                active=child(stage)
                while active.poll() is None:
                    stop_check();observe(stage);time.sleep(5)
                assert active.returncode==0,(stage,active.returncode)
                active=None
            assert read(STUDY/'readiness.json')['passed']
            evaluator=child('evaluate',extra=['--watch'])
            active=child('worker','B')
            write(STUDY/'status.json',dict(state='training',dataset='B',updated_at=now()))
            def tick():
                if evaluator.poll() is not None and evaluator.returncode!=0:
                    write(STUDY/'evaluation_failure.json',dict(exit_code=evaluator.returncode,updated_at=now(),action='Inspect; training unchanged'))
            supervise('B',active,tick);active=None
            write(STUDY/'training_complete.json',dict(completed_at=now(),B=valid_completion('B')))
            while evaluator.poll() is None:
                stop_check();observe('evaluating');time.sleep(5)
            assert evaluator.returncode==0,'Evaluation failed; preserve checkpoints'
            assert read(STUDY/'report_verification.json')['complete']
            write(STUDY/'status.json',dict(state='complete',updated_at=now(),report='MORNING_REPORT.md'));observe('complete')
        except BaseException as exc:
            state='stopped' if isinstance(exc,Stopped) else 'failed'
            write(STUDY/'status.json',dict(state=state,error=traceback.format_exc(),updated_at=now()))
            if not (STUDY/'STOP').exists():(STUDY/'STOP').write_text('Coordinator failure; inspect before any restart.\n')
            for p in (active,evaluator):
                if p is not None and p.poll() is None:
                    try:p.wait(timeout=60)
                    except subprocess.TimeoutExpired:pass
            observe(state);raise
        finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def verify_idle():
    import psutil
    conflicts=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=p.info['cmdline'] or [];joined=' '.join(cmd)
            if p.pid!=os.getpid() and ('capri_controller.training.' in joined or 'recover_b_once.py' in joined):
                if any(x in cmd for x in ['worker','rank','check','coordinate']):
                    # A venv launcher can be this coordinator's parent.
                    if p.pid not in [q.pid for q in psutil.Process().parents()]:conflicts.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not conflicts,conflicts
    write(STUDY/'idle_verified.json',dict(checked_at=now(),conflicts=conflicts))


def observe(stage):
    import psutil
    processes={}
    for name in ['coordinator','ranking','worker_B','evaluation_report']:
        path=STUDY/(name+'.lock.json')
        if path.exists():
            rec=read(path)
            try:
                p=psutil.Process(rec['pid']);rec['identity_matches']=abs(p.create_time()-rec['process_created'])<.1
                rec['cmdline']=p.cmdline()
            except psutil.NoSuchProcess:rec['identity_matches']=False
            processes[name]=rec
    status=run_dir(2,'B')/'status.json';progress=rank_dir('B')/'progress.json'
    value=dict(observed_at=now(),stage=stage,processes=processes,STOP_present=(STUDY/'STOP').exists(),
        training=read(status) if status.exists() else None,ranking=read(progress) if progress.exists() else None)
    write(STUDY/'live_monitor.json',value)
    write(Path('E:/pranav/CAPRI/runs/cryptbb_epoch20_static_B_20261115/live_monitor.json'),value)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['setup','rank','draws','check','fastcheck','worker','coordinate','evaluate','report'])
    ap.add_argument('--dataset',choices=['B']); ap.add_argument('--resume',action='store_true'); ap.add_argument('--watch',action='store_true')
    a=ap.parse_args(); bind()
    if a.stage in ['setup','rank','draws','check','fastcheck','worker','coordinate']: sys.addaudithook(no_labels)
    if a.stage in ['rank','check','fastcheck','worker']: C.bootstrap()
    if a.stage=='setup': setup()
    elif a.stage=='rank': rank()
    elif a.stage=='draws': prepare_draws()
    elif a.stage=='check': check()
    elif a.stage=='fastcheck':
        from .cryptbb_epoch20_static_b_fast import verify
        verify()
    elif a.stage=='worker':
        from .cryptbb_epoch20_static_b_fast import install_fast
        v=read(STUDY/'fast_verification.json')
        assert v['passed'] and v['protocol_sha256']==check_sources()
        pf=install_fast()
        try: worker(a.dataset,a.resume)
        finally: pf.close()
    elif a.stage=='coordinate': coordinate()
    else:
        from . import cryptbb_epoch20_static_b_report as R
        if a.stage=='evaluate': R.evaluate(a.watch)
        else:
            with StageLock(STUDY,'evaluation_report'): R.report()


if __name__=='__main__':
    sys.modules[MODULE]=sys.modules[__name__]
    main()
