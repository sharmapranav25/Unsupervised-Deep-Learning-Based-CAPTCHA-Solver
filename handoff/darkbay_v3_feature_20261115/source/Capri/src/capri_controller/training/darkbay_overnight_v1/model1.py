"""CryptBB v3 Model1 A/B: original upstream loop, concurrent processes, GPU batch lock."""
import argparse,contextlib,copy,os,shutil,time
from pathlib import Path
from ...decoder.common import read,write,sha,now,StageLock
from .. import cryptbb_model1 as BASE
from .common import ROOT,FAMILY,DATA
STUDY=ROOT/'handoff'/FAMILY/'model1'

def verify_sources():
    p=read(STUDY/'training_protocol.json')
    for r in p['sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert sha(STUDY/'agreement_protocol.json')==p['selection_protocol_sha256']
    return p

@contextlib.contextmanager
def gpu_slot():
    import msvcrt,torch
    from . import common as W
    W.stop()
    with (W.STUDY/'gpu.lock').open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        while True:
            try:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(.025)
        try:yield
        finally:
            torch.cuda.synchronize();torch.cuda.empty_cache();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    time.sleep(.005)

class LockedLoader:
    """Prepare CPU data concurrently; hold lock for the consumer's entire GPU step."""
    def __init__(self,inner):self.inner=inner
    def __len__(self):return len(self.inner)
    def __getattr__(self,name):return getattr(self.inner,name)
    def __iter__(self):
        for batch in self.inner:
            from . import common as W
            W.stop()
            with gpu_slot():yield batch

def row_for(tag):
    p=ROOT/f'model1/configs/{FAMILY}/{tag}/prepared.yml'
    return dict(dataset=tag,path=p.relative_to(ROOT).as_posix(),sha256=sha(p)),read(STUDY/'agreement_protocol.json')

def bind():
    from tools.engine import trainer as engine
    BASE.STUDY=STUDY;BASE.readiness=row_for
    # BASE capture references protocol.json; this is an identical immutable copy.
    assert sha(STUDY/'protocol.json')==sha(STUDY/'training_protocol.json')
    from . import common as W
    BASE.write=W.write
    original=engine.build_dataloader
    def build(*args,**kwargs):
        loader=original(*args,**kwargs)
        mode=args[1] if len(args)>1 else kwargs.get('mode')
        return LockedLoader(loader) if mode=='Train' and loader is not None else loader
    engine.build_dataloader=build

def check_data(tag):
    row,_=row_for(tag);p=verify_sources();assert row in p['configs']
    d=DATA/tag;r=read(d/'data_receipt.json')
    assert sha(d/'train_lmdb/data.mdb')==r['lmdb_sha256'] and sha(d/'sample_provenance.jsonl')==r['provenance_sha256']
    assert sha(DATA/'charset.txt')==r['charset_sha256']

def probe(tag):
    import torch,numpy as np
    check_data(tag);bind();out=STUDY/'readiness'/tag;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'receipt.json').exists()
    with gpu_slot():t=BASE.make_trainer(tag,out)
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    write(out/'initialized.json',dict(pid=os.getpid(),at=now()))
    from .. import ab_curriculum as C
    initial_tensor_sha256=C.model_hash(t.model)
    other=STUDY/'readiness'/('B' if tag=='A' else 'A')/'initialized.json';deadline=time.time()+180
    while not other.exists():
        assert time.time()<deadline,'Other parallel probe did not initialize';time.sleep(.5)
    batches=iter(t.train_dataloader)
    try:
        first=next(batches);assert list(first[0].shape)==[128,3,32,128]
        loss1=BASE.update_step(t,first);BASE.save_atomic(out/'recovery_probe.pth',BASE.capture(t,1,782,{}))
        second=next(batches);loss2=BASE.update_step(t,second)
        expected={k:v.detach().cpu().clone() for k,v in t.model.state_dict().items()}
        sched=copy.deepcopy(t.lr_scheduler.state_dict());scaler=copy.deepcopy(t.scaler.state_dict())
        saved=torch.load(out/'recovery_probe.pth',map_location='cpu',weights_only=False);BASE.restore(t,saved)
        replay=BASE.update_step(t,second);diff=max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in t.model.state_dict().items())
        assert diff==0 and loss2==replay and t.lr_scheduler.state_dict()==sched and t.scaler.state_dict()==scaler
    finally:batches.close()
    ls=BASE.loader_state(t);rs=BASE.rng_state(t.model);draws=list(iter(t.train_dataloader.batch_sampler));BASE.restore_loader(t,ls);BASE.restore_rng(t.model,rs);assert draws==list(iter(t.train_dataloader.batch_sampler))
    ids=[x[2] for b in draws for x in b];assert len(ids)==100096 and len(set(ids))==100000
    write(out/'receipt.json',dict(passed=True,dataset=tag,initial_tensor_sha256=initial_tensor_sha256,completed_at=now(),protocol_sha256=sha(STUDY/'training_protocol.json'),losses=[loss1,loss2,replay],recovery_max_abs_difference=diff,sampler_replay=True,batch_shape=list(first[0].shape),other_probe_initialized=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,production_weights_used=False))

def worker(tag,resume=False):
    import torch
    from tools.engine import trainer as engine
    check_data(tag);bind()
    for t in 'AB':
        r=read(STUDY/'readiness'/t/'receipt.json');assert r['passed'] and r['protocol_sha256']==sha(STUDY/'training_protocol.json')
    out=ROOT/f'model1/runs/{FAMILY}/{tag}';out.mkdir(parents=True,exist_ok=True)
    with StageLock(out,'training'):
        assert resume or not (out/'latest.pth').exists()
        with gpu_slot():
            t=BASE.make_trainer(tag,out)
            if resume:
                r=read(out/'latest_receipt.json');assert sha(out/'latest.pth')==r['sha256'];BASE.restore(t,torch.load(out/'latest.pth',map_location='cpu',weights_only=False))
        torch.set_num_threads(4)
        write(out/'launch.json',dict(started_at=now(),pid=os.getpid(),dataset=tag,epochs=20,parallel=True,gpu_batches_serialized=True,resume=resume,protocol_sha256=sha(STUDY/'training_protocol.json')))
        def save_hook(model,cfg,optimizer,lr_scheduler,epoch,global_step,metrics,is_best=False,logger=None,prefix=None):
            assert not is_best and global_step==epoch*782
            if prefix is not None:return
            archive=out/f'epoch_{epoch}.pth';assert not archive.exists()
            BASE.save_atomic(archive,BASE.capture(t,epoch,global_step,metrics));tmp=out/'latest.copying';shutil.copyfile(archive,tmp);os.replace(tmp,out/'latest.pth')
            r=dict(epoch=epoch,global_step=global_step,checkpoint=archive.name,sha256=sha(archive),completed_at=now(),protocol_sha256=sha(STUDY/'training_protocol.json'))
            write(out/f'epoch_{epoch}_receipt.json',r);write(out/'latest_receipt.json',r)
            if (STUDY/'STOP').exists():raise RuntimeError('Requested stop after committed epoch')
        engine.save_ckpt=save_hook
        count=[t.status.get('global_step',0)]
        def after_step(opt,args,kwargs):
            count[0]+=1
            if count[0]==1 or count[0]%50==0:write(out/'progress.json',dict(updated_at=now(),pid=os.getpid(),optimizer_steps=count[0],epoch=(count[0]-1)//782+1,step_in_epoch=(count[0]-1)%782+1))
        t.optimizer.register_step_post_hook(after_step);start=time.time();t.train()
        r=read(out/'latest_receipt.json');assert r['epoch']==20 and r['global_step']==15640
        write(out/'training_completion.json',dict(completed_at=now(),seconds=time.time()-start,**{k:v for k,v in r.items() if k!='completed_at'},selection_pending=True))


from .common import write
