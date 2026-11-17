"""CryptBB v2 Model1 A/B: original upstream loop, concurrent processes, GPU batch lock."""
import argparse,contextlib,copy,os,shutil,time
from pathlib import Path
from ..decoder.common import read,write,sha,now,StageLock
from . import cryptbb_model1 as BASE
ROOT=Path(__file__).resolve().parents[4];STUDY=ROOT/'handoff/cryptbb_v2_20261114'

def verify_sources():
    p=read(STUDY/'training_protocol.json')
    for r in p['sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert sha(STUDY/'agreement_protocol.json')==p['selection_protocol_sha256']
    return p

@contextlib.contextmanager
def gpu_slot():
    import msvcrt,torch
    with (STUDY/'gpu.lock').open('a+b') as f:
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
            with gpu_slot():yield batch

def row_for(tag):
    p=ROOT/f'model1/configs/cryptbb_v2/{tag}/prepared.yml'
    return dict(dataset=tag,path=str(p.relative_to(ROOT)),sha256=sha(p)),read(STUDY/'agreement_protocol.json')

def bind():
    from tools.engine import trainer as engine
    BASE.STUDY=STUDY;BASE.readiness=row_for
    # BASE capture references protocol.json; this is an identical immutable copy.
    assert sha(STUDY/'protocol.json')==sha(STUDY/'training_protocol.json')
    original=engine.build_dataloader
    def build(*args,**kwargs):
        loader=original(*args,**kwargs)
        mode=args[1] if len(args)>1 else kwargs.get('mode')
        return LockedLoader(loader) if mode=='Train' and loader is not None else loader
    engine.build_dataloader=build

def prepare():
    import yaml
    from ..cryptbb_v2_corpus import validate_entries
    validate_entries();rows=[];sets=[]
    for tag in 'AB':
        d=ROOT/f'datasets/cryptbb_v2/{tag}';r=read(d/'data_receipt.json')
        assert r['images']==r['unique_pixels']==100000 and sha(d/'train_lmdb/data.mdb')==r['lmdb_sha256']
        assert sha(d/'sample_provenance.jsonl')==r['provenance_sha256']
        import json
        sets.append({json.loads(x)['pixel_sha256'] for x in (d/'sample_provenance.jsonl').read_text().splitlines()})
        original=ROOT/f'model1/configs/cryptbb/{tag}/prepared_fixed32x128.yml'
        cfg=yaml.safe_load(original.read_text())
        def relocate(x):
            if isinstance(x,dict):return {k:relocate(v) for k,v in x.items()}
            if isinstance(x,list):return [relocate(v) for v in x]
            if isinstance(x,str):return x.replace('\\cryptbb\\','\\cryptbb_v2\\').replace('/cryptbb/','/cryptbb_v2/')
            return x
        cfg=relocate(cfg);assert cfg['Train']['loader']['min_ratio']==cfg['Train']['loader']['max_ratio']==4
        p=ROOT/f'model1/configs/cryptbb_v2/{tag}/prepared.yml';p.parent.mkdir(parents=True,exist_ok=True)
        assert not p.exists(),'Preserve existing v2 config'
        p.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8');rows.append(row_for(tag)[0])
    assert not sets[0]&sets[1]
    files=[Path(__file__),Path(BASE.__file__),ROOT/'Capri/src/capri_controller/cryptbb_v2_corpus.py',ROOT/'Capri/src/capri_controller/cryptbb_v2_workflow.py',ROOT/'Capri/src/capri_controller/decoder/common.py']
    for directory in ['model1/src/OpenOCR/tools','model1/src/OpenOCR/openrec']:
        files+=list((ROOT/directory).rglob('*.py'))
    sources=[]
    for p in files:
        rel=p.relative_to(ROOT);dest=STUDY/'training_sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources.append(dict(path=str(rel),sha256=sha(p)))
    protocol=dict(created_at=now(),authorization=read(STUDY/'authorization.json'),configs=rows,sources=sources,selection_protocol_sha256=sha(STUDY/'agreement_protocol.json'),epochs=20,steps_per_epoch=782,steps_each=15640,draws_per_epoch=100096,seed=48,batch_size=128,initialization='independent scratch',execution='Two concurrent processes; original upstream training loop; lock around full GPU batch via dataloader wrapper; CPU loading concurrent',preprocessing='Original RGB32x128 min_ratio=max_ratio4, no augmentation changes',memory='Optimizer/model states remain resident; release unused allocator blocks between complete batches',real_labels_used_for_training=False,new_AB_pixel_overlap=0,training_config_changes='Only corpus/charset/output paths changed from corrected CryptBB baseline')
    write(STUDY/'training_protocol.json',protocol);shutil.copyfile(STUDY/'training_protocol.json',STUDY/'protocol.json')
    write(STUDY/'datasets_ready.json',dict(images_each=100000,total_unique=200000,cross_dataset_overlap=0,configs=rows))

def check_data(tag):
    row,_=row_for(tag);p=verify_sources();assert row in p['configs']
    d=ROOT/f'datasets/cryptbb_v2/{tag}';r=read(d/'data_receipt.json')
    assert sha(d/'train_lmdb/data.mdb')==r['lmdb_sha256'] and sha(d/'sample_provenance.jsonl')==r['provenance_sha256']
    assert sha(ROOT/'datasets/cryptbb_v2/charset.txt')==r['charset_sha256']

def probe(tag):
    import torch,numpy as np
    check_data(tag);bind();out=STUDY/'readiness'/tag;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'receipt.json').exists()
    with gpu_slot():t=BASE.make_trainer(tag,out)
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    write(out/'initialized.json',dict(pid=os.getpid(),at=now()))
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
    write(out/'receipt.json',dict(passed=True,dataset=tag,completed_at=now(),protocol_sha256=sha(STUDY/'training_protocol.json'),losses=[loss1,loss2,replay],recovery_max_abs_difference=diff,sampler_replay=True,batch_shape=list(first[0].shape),other_probe_initialized=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,production_weights_used=False))

def worker(tag,resume=False):
    import torch
    from tools.engine import trainer as engine
    check_data(tag);bind()
    for t in 'AB':
        r=read(STUDY/'readiness'/t/'receipt.json');assert r['passed'] and r['protocol_sha256']==sha(STUDY/'training_protocol.json')
    out=ROOT/f'model1/runs/cryptbb_v2/{tag}';out.mkdir(parents=True,exist_ok=True)
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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','probe','worker']);ap.add_argument('--dataset',choices=['A','B']);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='probe':probe(a.dataset)
    else:worker(a.dataset,a.resume)
if __name__=='__main__':main()
