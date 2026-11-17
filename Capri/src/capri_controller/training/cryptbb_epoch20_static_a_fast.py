"""Verified execution-only optimization of the frozen epoch20-teacher A run."""
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
from . import cryptbb_epoch20_static_a as M

M.bind()
C=M.C
ROOT=M.ROOT
CHANGE=M.STUDY/'fast_execution_v1'
MODULE='capri_controller.training.cryptbb_epoch20_static_a_fast'
ORIGINAL_BATCH=C.batch_at
ORIGINAL_SLOT=C.gpu_slot
ORIGINAL_DRAWS=M.draws_for
ORIGINAL_DATASET=M.dataset
ORIGINAL_CHECK=M.check_sources


def check_execution():
    digest=ORIGINAL_CHECK()
    p=M.read(CHANGE/'execution_protocol.json')
    assert M.sha(Path(__file__))==p['source_sha256']
    assert p['training_protocol_sha256']==digest
    return digest


@contextlib.contextmanager
def cached_gpu_slot():
    """Same exclusive byte lock; keep allocator cache between synchronous steps."""
    import msvcrt
    with (M.STUDY/'gpu.lock').open('a+b') as f:
        f.seek(0,2)
        if f.tell()==0:f.write(b'0');f.flush()
        while True:
            try:
                f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(.05)
        try:yield
        finally:
            # C.step already synchronizes by converting loss and norm to Python floats.
            # This path is only used by the single-owner training worker.
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


class Prefetch:
    """One ordered lookahead batch, using deterministic CPU preprocessing only."""
    def __init__(self):
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cryptbb_batch')
        self.future=None;self.rows=None;self.position=0;self.closed=False

    def plan(self,draws):
        import numpy as np
        if self.future is not None:self.future.result();self.future=None
        self.rows=np.asarray(draws,dtype=np.int64).reshape(-1,128)
        self.position=0

    def batch(self,ds,indices):
        import numpy as np
        assert not self.closed and ds.aug is None
        actual=np.asarray(indices,dtype=np.int64)
        assert self.rows is not None and np.array_equal(actual,self.rows[self.position]),'Sample order changed'
        if self.future is None:self.future=self.executor.submit(ORIGINAL_BATCH,ds,actual.copy())
        result=self.future.result();self.future=None;self.position+=1
        if self.position<len(self.rows):
            self.future=self.executor.submit(ORIGINAL_BATCH,ds,self.rows[self.position].copy())
        return result

    def close(self):
        if not self.closed:
            self.executor.shutdown(wait=True,cancel_futures=True)
            self.future=None;self.closed=True


class ClosingEnv:
    def __init__(self,env,prefetch):self.env=env;self.prefetch=prefetch
    def __getattr__(self,name):return getattr(self.env,name)
    def close(self):
        self.prefetch.close()
        self.env.close()


def install_fast():
    pf=Prefetch()
    def draws(*args,**kwargs):
        values,receipt=ORIGINAL_DRAWS(*args,**kwargs)
        pf.plan(values);return values,receipt
    def dataset(tag):
        ds=ORIGINAL_DATASET(tag)
        assert ds.aug is None
        ds.env=ClosingEnv(ds.env,pf)
        return ds
    C.gpu_slot=cached_gpu_slot;C.batch_at=pf.batch
    M.draws_for=draws;M.dataset=dataset;M.check_sources=check_execution
    return pf


def to_cpu(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [to_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(to_cpu(v) for v in value)
    return copy.deepcopy(value)


def batch_hash(batch):
    h=hashlib.sha256()
    for tensor in batch:
        h.update(str((tensor.dtype,tuple(tensor.shape))).encode())
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def verify():
    import numpy as np,torch
    digest=check_execution()
    assert (M.STUDY/'STOP').read_text()=='User-authorized pause after epoch4 for verified faster execution.\n'
    M.verify_idle()
    paused=M.read(CHANGE/'pause_checkpoint.json');checkpoint=M.run_dir(2,'A')/paused['checkpoint']
    assert paused['epoch']==4 and M.sha(checkpoint)==paused['sha256']
    assert not (M.run_dir(2,'A')/'checkpoints/epoch_05.json').exists()
    N=24;trials=[];expected=None
    with M.StageLock(M.STUDY,'training_worker'),M.StageLock(CHANGE,'verification'):
        for fast in [False,True]:
            C.gpu_slot=cached_gpu_slot if fast else ORIGINAL_SLOT
            cfg=M.config('A',CHANGE/'disposable_probe')
            model,decoder=C.make_model(cfg);opt,sched,scaler=C.optimizer_for(model,cfg)
            sampler=C.sampler_for('A')
            ck=C.load_checkpoint(checkpoint,model,opt,sched,scaler,sampler,digest)
            assert ck['epoch']==4 and ck['global_step']==3128;del ck
            ds=ORIGINAL_DATASET('A');assert ds.aug is None
            draws,receipt=ORIGINAL_DRAWS(sampler,2,5)
            frozen=np.load(M.STUDY/'draws/A_epochs01_20.npy')[4]
            assert np.array_equal(sampler.ids[draws],frozen)
            pf=Prefetch() if fast else None
            if pf:pf.plan(draws[:N*128])
            values=[];hashes=[];durations=[]
            torch.cuda.synchronize();start=time.perf_counter()
            try:
                for i in range(N):
                    tick=time.perf_counter();indices=draws[i*128:(i+1)*128]
                    batch=pf.batch(ds,indices) if pf else ORIGINAL_BATCH(ds,indices)
                    hashes.append(batch_hash(batch))
                    values.append(C.step(model,opt,sched,scaler,batch))
                    durations.append(time.perf_counter()-tick)
                    if (i+1)%8==0:print(json.dumps(dict(fast=fast,batches=i+1,seconds=time.perf_counter()-start)),flush=True)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                state=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                    scaler=scaler.state_dict(),rng=C.rng_state(model),sampler=sampler.state_dict())
                if not fast:expected=to_cpu(state)
                else:
                    M.exact(expected,state)
                    assert values==trials[0]['loss_gradient'] and hashes==trials[0]['batch_sha256']
                trials.append(dict(fast=fast,batches=N,seconds=seconds,
                    median_step_seconds=float(np.median(durations[2:])),loss_gradient=values,batch_sha256=hashes))
            finally:
                if pf:pf.close()
                ds.env.close()
            del model,decoder,opt,sched,scaler,sampler,state,batch
            gc.collect();torch.cuda.empty_cache()
        ratio=trials[0]['seconds']/trials[1]['seconds']
        result=dict(passed=True,checked_at=M.now(),checkpoint_sha256=paused['sha256'],
            execution_source_sha256=M.sha(Path(__file__)),training_protocol_sha256=digest,
            batches=N,batch_size=128,exact_inputs_losses_gradients_weights_optimizer_scheduler_scaler_rng_sampler=True,
            trials=trials,speedup=ratio,use_fast=ratio>1.03,production_weights_untouched=True)
        M.write(CHANGE/'verification.json',result)
        print(json.dumps(dict(passed=True,speedup=ratio,use_fast=result['use_fast'])),flush=True)


def child(stage):
    import psutil
    args=[sys.executable,'-B','-u','-X','faulthandler','-m',MODULE,stage]
    with (CHANGE/(stage+'.log')).open('a',encoding='utf8') as f:
        p=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    M.write(CHANGE/(stage+'_launch.json'),dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),args=args,created_at=M.now()))
    return p


def launch():
    check_execution();M.verify_idle()
    v=M.read(CHANGE/'verification.json')
    assert v['passed'] and v['execution_source_sha256']==M.sha(Path(__file__))
    assert not (CHANGE/'coordinate_launch.json').exists(),'Resume already launched; inspect'
    assert not (M.STUDY/'evaluation_report.lock.json').exists()
    stop=M.STUDY/'STOP'
    assert stop.read_text()=='User-authorized pause after epoch4 for verified faster execution.\n'
    for name in ['status.json','evaluation_failure.json','live_monitor.json']:
        path=M.STUDY/name
        if path.exists():shutil.copyfile(path,CHANGE/('before_resume_'+name))
    stop.rename(CHANGE/'STOP_consumed_by_user_authorized_resume')
    M.write(CHANGE/'resume_authorization.json',dict(authorized_by='User: pause at next completion, set up fast version, then resume',
        created_at=M.now(),checkpoint=M.read(CHANGE/'pause_checkpoint.json'),use_fast=v['use_fast']))
    child('coordinate')


def coordinate():
    import psutil
    check_execution();M.stop_check()
    with M.StageLock(M.STUDY,'coordinator'):
        awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);assert awake
        worker=None;evaluator=None
        try:
            evaluator=child('evaluate');worker=child('worker')
            M.write(M.STUDY/'status.json',dict(state='training',dataset='A',execution='fast_execution_v1',updated_at=M.now()))
            first=None;verified=False
            while worker.poll() is None:
                M.stop_check();M.observe('training')
                path=M.run_dir(2,'A')/'status.json'
                if path.exists():
                    row=M.read(path)
                    if row.get('global_step',0)>3128 and row.get('updated_at','')>M.read(CHANGE/'worker_launch.json')['created_at']:
                        if first is None:first=row
                        if not verified and row['global_step']>first['global_step']:
                            identity=M.read(M.STUDY/'worker_A.lock.json');p=psutil.Process(identity['pid'])
                            assert abs(p.create_time()-identity['process_created'])<.1 and MODULE in p.cmdline()
                            M.write(CHANGE/'resume_verified.json',dict(verified_at=M.now(),worker=identity,first=first,progress=row,
                                checkpoint=M.read(CHANGE/'pause_checkpoint.json'),verification_sha256=M.sha(CHANGE/'verification.json')))
                            verified=True
                if evaluator.poll() is not None and evaluator.returncode!=0:
                    M.write(M.STUDY/'evaluation_failure.json',dict(exit_code=evaluator.returncode,updated_at=M.now(),action='Inspect; training unchanged'))
                time.sleep(5)
            assert worker.returncode==0,worker.returncode
            done=M.valid_completion('A');M.write(M.STUDY/'training_complete.json',dict(completed_at=M.now(),A=done))
            while evaluator.poll() is None:M.stop_check();M.observe('evaluating');time.sleep(5)
            assert evaluator.returncode==0
            assert M.read(M.STUDY/'report_verification.json')['complete']
            M.write(M.STUDY/'status.json',dict(state='complete',execution='fast_execution_v1',updated_at=M.now()));M.observe('complete')
        except BaseException:
            M.write(M.STUDY/'status.json',dict(state='failed',error=traceback.format_exc(),updated_at=M.now()))
            if not (M.STUDY/'STOP').exists():(M.STUDY/'STOP').write_text('Fast execution stopped; inspect before any recovery.\n')
            for p in (worker,evaluator):
                if p is not None and p.poll() is None:
                    try:p.wait(timeout=60)
                    except subprocess.TimeoutExpired:pass
            M.observe('failed');raise
        finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def evaluate():
    from . import cryptbb_epoch20_static_a_report as R
    base=R.report
    def report():
        done=base()
        result=M.read(M.STUDY/'results.json')
        result['execution_change']=dict(after_epoch=4,manifest='fast_execution_v1/execution_protocol.json',
            verification='fast_execution_v1/verification.json',user_authorized=True,
            measured_speedup=M.read(CHANGE/'verification.json')['speedup'],
            numerical_protocol_unchanged=True)
        M.write(M.STUDY/'results.json',result)
        with (M.STUDY/'MORNING_REPORT.md').open('a',encoding='utf8') as f:
            f.write('\nExecution change after epoch4: user-authorized checkpoint pause/resume. Batch128, sample order, weights/optimizer/scheduler/scaler/RNG preserved.24 actual subsequent updates reproduced bit-for-bit before resume; see fast_execution_v1. Disposable verification updates are not training progress.\n')
        v=M.read(M.STUDY/'report_verification.json')
        v.update(report_sha256=M.sha(M.STUDY/'MORNING_REPORT.md'),results_sha256=M.sha(M.STUDY/'results.json'))
        M.write(M.STUDY/'report_verification.json',v)
        return done
    R.report=report
    R.evaluate(True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['verify','launch','coordinate','worker','evaluate'])
    args=ap.parse_args()
    if args.stage!='evaluate':sys.addaudithook(M.no_labels)
    if args.stage in ['verify','worker']:C.bootstrap()
    if args.stage=='verify':verify()
    elif args.stage=='launch':launch()
    elif args.stage=='coordinate':coordinate()
    elif args.stage=='evaluate':evaluate()
    else:
        check_execution();v=M.read(CHANGE/'verification.json');assert v['passed']
        pf=install_fast() if v['use_fast'] else None
        try:M.worker('A',resume=True)
        finally:
            if pf:pf.close()


if __name__=='__main__':main()
