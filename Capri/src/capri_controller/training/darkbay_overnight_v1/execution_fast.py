"""User-authorized serial GPU ownership with retained allocator cache."""
import contextlib,gc,os,sys,time
from pathlib import Path
from . import common as W
from . import execution_fp32 as FP
from ...decoder.common import StageLock
D=W.STUDY/'recovery/20261116_fast_serial'

def verify():
    FP.verify_execution();m=W.read(D/'manifest.json')
    assert W.sha(Path(__file__))==m['overlay_sha256']
    assert W.sha(W.STUDY/'source_manifest.json')==m['frozen_manifest_sha256']
    return m

def idle():
    import psutil
    for n in (2,3):
        for tag in 'AB':
            p=W.stage(n,tag)/f'worker_{tag}.lock.json'
            if not p.exists():continue
            r=W.read(p)
            try:
                q=psutil.Process(r['pid'])
                assert abs(q.create_time()-r['process_created'])>=.02,('Live worker',r)
            except psutil.NoSuchProcess:pass

def fast_worker():
    import torch
    from . import engine as E,readers
    verify();idle();gate=W.read(D/'parity.json');assert gate['passed']
    assert gate['overlay_sha256']==W.sha(Path(__file__))
    tag=sys.argv[sys.argv.index('--dataset')+1];n=int(sys.argv[sys.argv.index('--model')+1])
    original_slot=W.gpu_slot;original_write=E.write;original_check=E.check_sources
    def check_sources():
        verify();return original_check()
    E.check_sources=check_sources
    # Acquire the existing GPU byte lock once for the worker's complete lifetime.
    # The unchanged C.step still performs all optimizer operations and BF16 math.
    with StageLock(W.STUDY,'fast_gpu_worker'),original_slot():
        W.gpu_slot=contextlib.nullcontext;W.C.gpu_slot=contextlib.nullcontext
        record=dict(at=W.now(),pid=os.getpid(),model=n,dataset=tag,overlay_sha256=W.sha(Path(__file__)),manifest_sha256=W.sha(D/'manifest.json'),execution='serial_cached_gpu',prefetch='unchanged OrderedBatches',inference='unchanged IEEE FP32 overlay',protocol_sha256=W.sha(W.stage(n,tag)/'protocol.json'))
        W.write(W.stage(n,tag)/'execution_fast_active.json',record)
        def write(path,value):
            original_write(path,value)
            if path.parent.name=='checkpoints' and path.suffix=='.json':
                original_write(W.rundir(n,tag)/'execution_receipts'/path.name,dict(**record,checkpoint_receipt_sha256=W.sha(path),epoch=value['epoch'],committed_at=W.now()))
        E.write=write
        FP.main()

def sequential(stage,model=None):
    from . import coordinator as K
    assert stage=='worker' and model in (2,3)
    idle()
    for tag in 'AB':
        run=W.rundir(model,tag);completion=run/'completion.json'
        if completion.exists():
            r=W.read(completion);assert r['epoch']==20 and W.sha(run/r['checkpoint'])==r['sha256'];continue
        args=['worker','--dataset',tag,'--model',str(model)]
        if list((run/'checkpoints').glob('epoch_*.json')):args+=['--resume']
        K.one('readers',args,f'worker_M{model}_{tag}',[run/'status.json',run/'history.json'])

def coordinator():
    from . import coordinator as K
    original_pair=K.paired;original_launch=K.launch_child
    def pair(stage,model=None):
        if stage=='worker':return sequential(stage,model)
        return original_pair(stage,model)
    def launch(module,args,name):
        verify()
        if module in ('readers','coordinator'):return original_launch('execution_fast',[module,*args],name)
        return original_launch(module,args,name)
    K.paired=pair;K.launch_child=launch
    sys.argv.pop(1);K.main()

def probe():
    import numpy as np,torch
    from . import readers,engine as E
    C=W.C;verify();idle();C.bootstrap();results=[];original_slot=W.gpu_slot
    for tag in 'AB':
        E=readers.bind(2,tag);ds=E.dataset();digest=E.check_sources()
        rec=W.read(E.OUT/'checkpoints/epoch_05.json');path=E.OUT/rec['checkpoint']
        assert W.sha(path)==rec['sha256'] and rec['global_step']==3910
        trials=[];expected=None;expected_draws=None
        for fast in (False,True):
            cfg=E.config(D);model,decoder=C.make_model(cfg);opt,sched,scaler=C.optimizer_for(model,cfg);sm,_=E.load_features()
            ck=E.restore(path,model,opt,sched,scaler,sm,digest)
            assert ck['epoch']==5 and ck['global_step']==3910 and sm.epoch==6
            E.exact(ck['state_dict'],model.state_dict());E.exact(ck['optimizer'],opt.state_dict());E.exact(ck['lr_scheduler'],sched.state_dict());E.exact(ck['scaler'],scaler.state_dict());E.exact(ck['rng'],C.rng_state(model));E.exact(ck['adaptive'],sm.state_dict());del ck
            draws,receipt=sm.draw()
            if expected_draws is None:expected_draws=draws.copy()
            else:assert np.array_equal(expected_draws,draws)
            C.gpu_slot=contextlib.nullcontext if fast else original_slot
            values=[];hashes=[]
            with (original_slot() if fast else contextlib.nullcontext()):
                torch.cuda.synchronize();start=time.perf_counter()
                for batch in E.OrderedBatches(ds,draws[:24*128],128):
                    import hashlib
                    hashes.append(hashlib.sha256(b''.join(v.numpy().tobytes() for v in batch)).hexdigest())
                    values.append(C.step(model,opt,sched,scaler,batch))
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
            state=E.to_cpu(dict(model=model.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),scaler=scaler.state_dict(),rng=C.rng_state(model),sampler=sm.state_dict()))
            if not fast:expected=state
            else:
                E.exact(expected,state);assert trials[0]['values']==values and trials[0]['batch_hashes']==hashes
            trials.append(dict(fast=fast,seconds=elapsed,batches=24,values=values,batch_hashes=hashes))
            del model,decoder,opt,sched,scaler,sm,state;gc.collect();torch.cuda.empty_cache()
            print({'tag':tag,'fast':fast,'seconds':elapsed},flush=True)
        ds.env.close();results.append(dict(dataset=tag,checkpoint_sha256=rec['sha256'],epoch=5,full_state_restore_exact=True,all24_updates_exact=True,trials=trials,speedup=trials[0]['seconds']/trials[1]['seconds']))
    C.gpu_slot=original_slot
    W.write(D/'parity.json',dict(passed=True,at=W.now(),overlay_sha256=W.sha(Path(__file__)),pilot_weights_disposable=True,real_labels_used=False,results=results))

def main():
    W.environment();verify();mode=sys.argv[1]
    if mode=='probe':
        with StageLock(W.STUDY,'fast_probe'):probe()
    elif mode=='coordinator':
        assert W.read(D/'parity.json')['passed'];coordinator()
    elif mode=='readers':
        if sys.argv[2]=='worker':fast_worker()
        else:FP.main()
    else:raise ValueError(mode)
if __name__=='__main__':main()
