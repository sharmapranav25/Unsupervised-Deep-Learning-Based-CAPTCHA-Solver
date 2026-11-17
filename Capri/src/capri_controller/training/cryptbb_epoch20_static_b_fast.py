"""Execution-only optimization for the separate epoch20-teacher B run."""
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
from . import cryptbb_epoch20_static_b as M

M.bind()
C=M.C
ROOT=M.ROOT
CHANGE=M.STUDY/'fast_execution_v1'
MODULE='capri_controller.training.cryptbb_epoch20_static_b_fast'
ORIGINAL_BATCH=C.batch_at
ORIGINAL_SLOT=C.gpu_slot
ORIGINAL_DRAWS=M.draws_for
ORIGINAL_DATASET=M.dataset
ORIGINAL_CHECK=M.check_sources


def check_execution():
    return ORIGINAL_CHECK()


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
    digest=check_execution();M.stop_check()
    assert not M.run_dir(2,'B').exists()
    out=M.STUDY/'fast_checks';out.mkdir(exist_ok=True)
    N=24;trials=[];expected=None
    with M.StageLock(M.STUDY,'training_worker'),M.StageLock(M.STUDY,'fast_verification'):
        for fast in [False,True]:
            C.gpu_slot=cached_gpu_slot if fast else ORIGINAL_SLOT
            cfg=M.config('B',out)
            model,decoder=C.make_model(cfg);opt,sched,scaler=C.optimizer_for(model,cfg)
            assert C.model_hash(model)==M.read(M.STUDY/'readiness.json')['datasets']['B']['initial_tensor_sha256']
            sampler=C.sampler_for('B');ds=ORIGINAL_DATASET('B');assert ds.aug is None
            draws,receipt=ORIGINAL_DRAWS(sampler,2,1)
            frozen=np.load(M.STUDY/'draws/B_epochs01_20.npy')[0]
            assert np.array_equal(sampler.ids[draws],frozen)
            pf=Prefetch() if fast else None
            if pf:pf.plan(draws[:N*128])
            values=[];hashes=[]
            torch.cuda.synchronize();start=time.perf_counter()
            try:
                for i in range(N):
                    M.stop_check();indices=draws[i*128:(i+1)*128]
                    batch=pf.batch(ds,indices) if pf else ORIGINAL_BATCH(ds,indices)
                    hashes.append(batch_hash(batch));values.append(C.step(model,opt,sched,scaler,batch))
                    if (i+1)%8==0:print(json.dumps(dict(fast=fast,batches=i+1,seconds=time.perf_counter()-start)),flush=True)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                state=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                    scaler=scaler.state_dict(),rng=C.rng_state(model),sampler=sampler.state_dict())
                if not fast:expected=to_cpu(state)
                else:
                    M.exact(expected,state)
                    assert values==trials[0]['loss_gradient'] and hashes==trials[0]['batch_sha256']
                trials.append(dict(fast=fast,batches=N,seconds=seconds,loss_gradient=values,batch_sha256=hashes))
            finally:
                if pf:pf.close()
                ds.env.close()
            del model,decoder,opt,sched,scaler,sampler,state,batch
            gc.collect();torch.cuda.empty_cache()
        M.write(M.STUDY/'fast_verification.json',dict(passed=True,checked_at=M.now(),protocol_sha256=digest,
            batches=N,batch_size=128,exact_inputs_losses_gradients_weights_optimizer_scheduler_scaler_rng_sampler=True,
            trials=trials,speedup=trials[0]['seconds']/trials[1]['seconds'],pilot_weights_disposable=True))
        print('B fast execution exact24step verification passed',flush=True)
