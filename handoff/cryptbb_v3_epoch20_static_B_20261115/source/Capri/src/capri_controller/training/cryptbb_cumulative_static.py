"""Frozen cumulative static CryptBB experiment; separate from historical adapters.

Entry: python -B -m capri_controller.training.cryptbb_cumulative_static STAGE
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
FAMILY = 'cryptbb_v3_cumulative_static_v1_20261115'
STUDY = ROOT / 'handoff' / FAMILY
MODULE = 'capri_controller.training.cryptbb_cumulative_static'
EXPECTED_RANK = {'A':'9bd75fa67421da57a51976e7229bfc91efa25f8756e36d4836a1f7c80952b752', 'B':'313fe5177989eae33d0dd028d1c770d976e7973b1fc3f2d7ca38166fbd7ec53c'}
original_draws = C.draws_for
config, dataset, rank_dir = OLD.config, OLD.dataset, OLD.rank_dir
write = C.write


class Stopped(RuntimeError):
    pass


def stop_check():
    if (STUDY/'STOP').exists():
        raise Stopped('User STOP; preserve partial work and resume only with explicit authorization')


def run_dir(number, tag):
    assert number == 2 and tag in 'AB'
    return ROOT/'model2/runs'/FAMILY/tag


def result_dir(tag, epoch):
    return ROOT/'model2/results'/FAMILY/tag/f'epoch_{epoch:02d}'


def check_sources():
    protocol = read(STUDY/'protocol.json')
    for row in protocol['sources']:
        assert sha(ROOT/row['path']) == row['sha256'], row['path']
    return sha(STUDY/'protocol.json')


def draws_for(sampler, number, epoch, count=C.DRAWS):
    import numpy as np
    assert number == 2 and sampler.epoch == epoch
    if epoch <= 2:
        return original_draws(sampler, number, epoch, count)
    rng = np.random.default_rng(C.SEED + epoch*100003)
    draws = rng.choice(len(sampler.ids), count, replace=True)
    # The RNG uses exact uniform choice over every image, not stratified thirds.
    sampler.mass = np.bincount(sampler.buckets, minlength=3)/len(sampler.ids)
    receipt = dict(epoch=epoch, mass=sampler.mass.tolist(),
        draws=np.bincount(sampler.buckets[draws], minlength=3).tolist(), total=count,
        dropped=0, sample_ids_sha256=hashlib.sha256(sampler.ids[draws].tobytes()).hexdigest(),
        probability_each=1/len(sampler.ids), method='uniform_full_pool_with_replacement')
    sampler.epoch += 1
    return draws, receipt


def bind():
    C.STUDY=STUDY; C.config=config; C.dataset=dataset; C.rank_dir=rank_dir
    C.run_dir=run_dir; C.check_sources=check_sources; C.draws_for=draws_for
    # Use C's identical complete-step lock in the NEW study directory.


def no_labels(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = str(args[0]).lower().replace('\\','/')
        if any(x in path for x in ('scoring_only/', 'evaluation_pool_labels', 'per_image_results.json')):
            raise RuntimeError('Real scoring labels forbidden in training/readiness/selection')


def setup():
    import numpy as np
    STUDY.mkdir(parents=True, exist_ok=True)
    assert not (STUDY/'protocol.json').exists(), 'Frozen study already exists'
    assert all(not run_dir(2,t).exists() for t in 'AB')
    assert (OLD.STUDY/'STOP').exists(), 'Preserve historical STOP'
    OLD.check_sources()
    OLD.selection()
    imports = {}
    for tag in 'AB':
        OLD.M1.check_data(tag)
        ck=OLD.teacher(tag); row=OLD.teacher_row(tag)
        assert sha(ck)==row['checkpoint_sha256']
        assert sha(ck)==read(ck.parent/f"epoch_{row['selected_epoch']}_receipt.json")['sha256']
        rd=rank_dir(tag); done=read(rd/'completion.json')
        assert done['protocol_sha256']==sha(OLD.STUDY/'protocol.json')
        assert done['teacher_sha256']==sha(ck) and done['images']==100000
        assert done['teacher']==('B' if tag=='A' else 'A')
        assert sha(rd/'ranking.npz')==done['sha256']==EXPECTED_RANK[tag]
        assert sha(rd/'predictions.json')==done['predictions_sha256']
        assert [r['index'] for r in read(rd/'predictions.json')]==list(range(1,100001))
        with np.load(rd/'ranking.npz') as z:
            ids, raw, buckets = z['ids'], z['nll'], z['buckets']
            assert np.array_equal(ids,np.arange(1,100001)) and raw.shape==(100000,)
            assert np.isfinite(raw).all() and (raw>=0).all()
            expected=np.empty(100000,dtype=np.int64)
            for b,g in enumerate(np.array_split(np.lexsort((ids,raw)),3)): expected[g]=b
            assert np.array_equal(buckets,expected)
            assert np.bincount(buckets).tolist()==[33334,33333,33333]
        imports[tag]=dict(ranking=str(rd.relative_to(ROOT)),ranking_sha256=done['sha256'],
            old_completion=done,old_completion_sha256=sha(rd/'completion.json'),
            data_receipt=read(ROOT/f'datasets/cryptbb_v3/{tag}/data_receipt.json'),
            teacher=row,old_protocol_sha256=done['protocol_sha256'])
    paths={ROOT/r['path'] for r in read(OLD.STUDY/'protocol.json')['sources']}
    paths.update(Path(__file__).parent.glob('cryptbb_cumulative*.py'))
    paths.add(ROOT/'analysis/checkpoint_selection/ckpt_select/estimator.py')
    sources=[]
    for path in sorted(paths):
        rel=path.relative_to(ROOT); dst=STUDY/'source'/rel
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dst)
        sources.append(dict(path=rel.as_posix(),sha256=sha(path)))
    write(STUDY/'ranking_import.json',dict(created_at=now(),datasets=imports,
          note='Read-only import from old protocol; original completion receipts unchanged'))
    write(STUDY/'protocol.json',dict(created_at=now(),family=FAMILY,sources=sources,
        seed=48,epochs=20,batch=128,steps_per_epoch=782,draws_per_epoch=100096,
        total_steps=15640,total_presentations=2001920,pool_size=100000,
        schedule={'1':'old easy stream','2':'old 50048 easy + 50048 medium shuffled stream',
                  '3-20':'numpy default_rng(48+epoch*100003).choice(100000,100096,replace=True)'},
        initialization='Independent scratch seed48; teachers are ranking only',
        config='Unchanged prepared.yml except output paths; AdamW .00035/.05; OneCycle2/20; BF16 clip20',
        preprocessing='Unchanged RGB PIL bicubic32x128 ToTensor Normalize(.5,.5)',
        execution='One GPU training worker; A then B; CPU checkpoint evaluation',
        sampling_seed='48+epoch*100003',ranking_import_sha256=sha(STUDY/'ranking_import.json'),
        selection='Independent original v1 exact-string masked rank1; k0..5 choose_k/fallback5; highest factor earliest tie',
        validation='Predeclared candidate; no synthetic validation tuning or real-label schedule search',
        real_labels_used=False,extra_training_authorized=False))
    draw_manifest={}
    for tag in 'AB':
        sampler=C.sampler_for(tag); sequences=[]; records=[]
        for epoch in range(1,21):
            before=copy.deepcopy(sampler); draws,receipt=draws_for(sampler,2,epoch)
            replay,replay_receipt=draws_for(copy.deepcopy(before),2,epoch)
            assert np.array_equal(draws,replay) and receipt==replay_receipt
            assert len(draws)==100096 and len(draws)%128==0
            if epoch<=2:
                old,old_r=original_draws(before,2,epoch)
                assert np.array_equal(draws,old) and receipt==old_r
                assert (sampler.buckets[draws] < epoch).all()
            else:
                independent=np.random.default_rng(48+epoch*100003).choice(100000,100096,replace=True)
                assert np.array_equal(draws,independent)
            ids=sampler.ids[draws]; sequences.append(ids); records.append(receipt)
        out=STUDY/'draws'; out.mkdir(exist_ok=True)
        full=np.stack(sequences); np.save(out/f'{tag}_epochs01_20.npy',full)
        # Future controls: freeze random expansion and exact exposure permutation, no training.
        permutation_seed=480015; shuffled=np.random.default_rng(permutation_seed).permutation(full.ravel())
        assert np.array_equal(np.bincount(full.ravel()),np.bincount(shuffled))
        np.save(out/f'{tag}_exact_exposure_shuffled.npy',shuffled)
        thirds=np.random.default_rng(480016).permutation(np.arange(1,100001))
        np.save(out/f'{tag}_random_thirds.npy',thirds)
        draw_manifest[tag]=dict(epochs=records,path=f'draws/{tag}_epochs01_20.npy',
            sha256=sha(out/f'{tag}_epochs01_20.npy'),exact_exposure_sha256=sha(out/f'{tag}_exact_exposure_shuffled.npy'),
            permutation_seed=permutation_seed,random_thirds_seed=480016,
            random_thirds_sha256=sha(out/f'{tag}_random_thirds.npy'),total=full.size)
    write(STUDY/'draw_manifest.json',dict(passed=True,datasets=draw_manifest,
        controls='Uniform, random expansion, difficulty expansion, exact exposure shuffled; only difficulty A/B trained. Same data, initialization, preprocessing, optimizer and budget required. Historical Model1/static are not causal controls.'))
    print('Setup and all 40 epoch draw streams verified',flush=True)


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
    for tag in 'AB':
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
    assert initial[0]==initial[1]
    from .cryptbb_cumulative_checks import lifecycle_checks
    lifecycle=lifecycle_checks()
    write(STUDY/'readiness.json',dict(passed=True,created_at=now(),protocol_sha256=digest,
        datasets=checks,preprocessing=read(STUDY/'preprocessing_verification.json'),
        lifecycle=lifecycle,pilot_weights_disposable=True,real_labels_read=False))
    print('Readiness passed: exact full-batch checkpoint replay on A and B',flush=True)


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
        stop_check(); on_tick()
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
                if tag=='A': shutil.copyfile(STUDY/'launch_verified_A.json',STUDY/'launch_verified.json')
                verified=True
        time.sleep(5)
    stop_check()
    assert p.returncode==0, f'{tag} worker failed: {p.returncode}'
    return valid_completion(tag)


def coordinate():
    with StageLock(STUDY,'coordinator'):
        check_sources(); stop_check()
        assert read(STUDY/'readiness.json')['passed']
        assert all(not run_dir(2,t).exists() for t in 'AB'), 'No automatic restarts'
        awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        assert awake, 'Unable to request host stay awake'
        write(STUDY/'awake.json',dict(requested_at=now(),system_required=True,previous_state=awake))
        evaluator=None; active=None
        try:
            evaluator=child('evaluate',extra=['--watch'])
            for tag in 'AB':
                stop_check()
                if tag=='B': valid_completion('A')
                active=child('worker',tag)
                write(STUDY/'status.json',dict(state='training',dataset=tag,updated_at=now()))
                # Evaluation failure is surfaced but never changes training decisions.
                def tick():
                    if evaluator.poll() is not None and evaluator.returncode!=0:
                        write(STUDY/'evaluation_failure.json',dict(exit_code=evaluator.returncode,updated_at=now(),action='Inspect; training unchanged'))
                supervise(tag,active,tick); active=None
            write(STUDY/'training_pair_complete.json',dict(completed_at=now(),A=valid_completion('A'),B=valid_completion('B')))
            while evaluator.poll() is None:
                stop_check(); time.sleep(5)
            assert evaluator.returncode==0, 'Evaluation/report failed; training checkpoints preserved'
            assert read(STUDY/'report_verification.json')['complete']
            write(STUDY/'status.json',dict(state='complete',updated_at=now(),report='MORNING_REPORT.md'))
        except BaseException as exc:
            state='stopped' if isinstance(exc,Stopped) else 'failed'
            write(STUDY/'status.json',dict(state=state,error=traceback.format_exc(),updated_at=now()))
            if not (STUDY/'STOP').exists(): (STUDY/'STOP').write_text('Coordinator failure; inspect before any restart.\n')
            # Workers check STOP before every optimizer update; wait for their clean exit.
            for p in (active,evaluator):
                if p is not None and p.poll() is None:
                    try: p.wait(timeout=60)
                    except subprocess.TimeoutExpired: pass
            raise
        finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['setup','check','worker','coordinate','evaluate','report'])
    ap.add_argument('--dataset',choices=['A','B']); ap.add_argument('--resume',action='store_true'); ap.add_argument('--watch',action='store_true')
    a=ap.parse_args(); bind()
    if a.stage in ['setup','check','worker','coordinate']: sys.addaudithook(no_labels)
    if a.stage in ['check','worker']: C.bootstrap()
    if a.stage=='setup': setup()
    elif a.stage=='check': check()
    elif a.stage=='worker': worker(a.dataset,a.resume)
    elif a.stage=='coordinate': coordinate()
    else:
        from . import cryptbb_cumulative_report as R
        if a.stage=='evaluate': R.evaluate(a.watch)
        else:
            with StageLock(STUDY,'evaluation_report'): R.report()


if __name__=='__main__': main()
