"""DarkBay wrappers for unchanged Model1 and static/feature-priority numerical steps."""
import argparse,contextlib,copy,gc,json,os,shutil,sys,time
from pathlib import Path
from . import common as W
from .common import ROOT,STUDY,DATA,FAMILY,read,write,sha,now,C

def rank(tag):
    import numpy as np,torch
    W.verify_sources();C.bootstrap();C.STUDY=STUDY;C.gpu_slot=W.gpu_slot
    dest=STUDY/'rankings'/tag
    if (dest/'completion.json').exists():
        r=read(dest/'completion.json');assert sha(dest/'ranking.npz')==r['sha256'];return
    assert not dest.exists();dest.mkdir(parents=True)
    other='B' if tag=='A' else 'A';path=W.rundir(1,other)/'epoch_20.pth'
    receipt=read(W.rundir(1,other)/'epoch_20_receipt.json');assert sha(path)==receipt['sha256']
    with W.gpu_slot():
        model,decoder=C.make_model(W.config(tag,dest));ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['epoch']==20 and ck['global_step']==15640;model.load_state_dict(ck['state_dict'],strict=True);del ck
    ds=W.dataset(tag)
    try:raw,seconds=C.score_pool(model,decoder,ds,dest)
    finally:ds.env.close()
    ids=np.arange(1,100001);buckets=np.empty(100000,dtype=np.int64)
    for b,ix in enumerate(np.array_split(np.lexsort((ids,raw)),3)):buckets[ix]=b
    np.savez_compressed(dest/'ranking.npz',ids=ids,nll=raw,buckets=buckets)
    write(dest/'completion.json',dict(dataset=tag,teacher=other,teacher_epoch=20,images=100000,teacher_sha256=sha(path),sha256=sha(dest/'ranking.npz'),predictions_sha256=sha(dest/'predictions.json'),bucket_counts=np.bincount(buckets).tolist(),seconds=seconds,protocol_sha256=sha(STUDY/'model1/protocol.json')))

class StaticSampler:
    def __init__(self,buckets):
        import numpy as np
        self.ids=np.arange(1,100001);self.buckets=buckets.copy();self.original_buckets=buckets.copy();self.epoch=1;self.mass=np.array([1.,0.,0.])
    def draw(self):return C.draws_for(self,2,self.epoch)
    def state_dict(self):return dict(epoch=self.epoch,mass=self.mass.copy(),bucket_sha256=__import__('hashlib').sha256(self.buckets.tobytes()).hexdigest())
    def load_state_dict(self,state):
        assert state['bucket_sha256']==self.state_dict()['bucket_sha256'];self.epoch=state['epoch'];self.mass=state['mass'].copy()

def bind(model_number,tag):
    from . import engine as E
    import numpy as np
    E.STUDY=W.stage(model_number,tag);E.OUT=W.rundir(model_number,tag);E.BASE=STUDY/'model1'
    E.ROOT=ROOT;E.config=lambda out:W.config(tag,out);E.dataset=lambda:W.dataset(tag)
    E.gpu_owner=contextlib.nullcontext;E.stop_check=W.stop;E.verify_idle=lambda:None
    E.TAG=tag;E.TEACHER=ROOT/'model1/runs'/FAMILY/('B' if tag=='A' else 'A')/'epoch_20.pth'
    if model_number==2:
        def load():
            with np.load(STUDY/'rankings'/tag/'ranking.npz') as z:s=StaticSampler(z['buckets'])
            return s,None
        E.load_features=load
        def assess(model,decoder,ds,truth,sm,epoch):
            result=dict(seconds=0,overall_accuracy=None,adaptation=False)
            write(E.OUT/f'assessments/epoch_{epoch:02d}/feedback.json',result);return result
        E.assess=assess
    return E

def prepare():
    import numpy as np
    from ..cryptbb_feature_v1_sampler import FeatureSampler,self_test,POLICY
    from . import metadata
    W.verify_sources()
    checks={t:read(STUDY/f'model1/readiness/{t}/receipt.json') for t in 'AB'}
    assert all(r['passed'] for r in checks.values())
    write(STUDY/'model1/readiness.json',dict(passed=True,datasets=checks))
    for tag in 'AB':
        for number in (2,3):
            dest=W.stage(number,tag);dest.mkdir(parents=True,exist_ok=True)
            if (dest/'protocol.json').exists():continue
            meta=metadata.build(ROOT,dest/'features',read,sha,write,tag)
            with np.load(dest/'features/features.npz') as z:
                sm=FeatureSampler(z['buckets'],z['groups'],meta);static=StaticSampler(z['buckets'])
            expected=[]
            for epoch in range(1,21):
                d,r=static.draw();expected.append(static.ids[d])
                if epoch<=2:
                    a,ar=sm.draw();assert np.array_equal(sm.ids[a],static.ids[d]) and ar==r
            drawpath=STUDY/f'model1/draws/{tag}_epochs01_20.npy';drawpath.parent.mkdir(exist_ok=True)
            if drawpath.exists():assert np.array_equal(np.load(drawpath),expected)
            else:np.save(drawpath,np.stack(expected))
            sources=read(STUDY/'source_manifest.json')
            cfg=ROOT/'model1/configs'/FAMILY/tag/'prepared.yml';sources=sources+[dict(path=cfg.relative_to(ROOT).as_posix(),sha256=sha(cfg))]
            protocol=dict(created_at=now(),model=number,dataset=tag,seed=48,epochs=20,batch=128,steps_per_epoch=782,total_steps=15640,draws_per_epoch=100096,teacher=meta['ranking'],sources=sources,feature_manifest_sha256=sha(dest/'features/manifest.json'),policy=POLICY if number==3 else {'1':[1,0,0],'2':[.5,.5,0],'3-20':[.15,.25,.6]},selection='Fixed final20',execution='A/B parallel processes, GPU lock per full128 training or256 inference batch',real_labels_used=False)
            write(dest/'policy_tests.json',self_test());write(dest/'protocol.json',protocol)

def check(tag):
    from . import model1 as M
    M.probe(tag)

def curriculum_check(tag):
    E=bind(3,tag);E.check()
    # Static sampler and optimizer are unchanged; verify exact draw streams and
    # static checkpoint next-update replay independently, not by copying readiness.
    import numpy as np,torch
    E=bind(2,tag);digest=E.check_sources();C.bootstrap();C.STUDY=E.STUDY;C.gpu_slot=W.gpu_slot
    cfg=W.config(tag,E.STUDY/'checks');ds=W.dataset(tag);sm,_=E.load_features()
    with W.gpu_slot():model,decoder=C.make_model(cfg);opt,sched,scaler=C.optimizer_for(model,cfg)
    initial=C.model_hash(model);assert initial==read(STUDY/f'model1/readiness/{tag}/receipt.json')['initial_tensor_sha256']
    d,_=sm.draw();C.step(model,opt,sched,scaler,C.batch_at(ds,d[:128]))
    path=E.STUDY/'checks/replay.pth';C.save_checkpoint(path,model,opt,sched,scaler,sm,cfg,1,[],digest)
    d1,r1=sm.draw();batch=C.batch_at(ds,d1[:128]);loss1,norm1=C.step(model,opt,sched,scaler,batch)
    expected=E.to_cpu(dict(model=model.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),scaler=scaler.state_dict(),rng=C.rng_state(model),sampler=sm.state_dict()))
    del model,decoder,opt,sched,scaler;gc.collect();torch.cuda.empty_cache()
    with W.gpu_slot():model,decoder=C.make_model(cfg);opt,sched,scaler=C.optimizer_for(model,cfg)
    sm,_=E.load_features();E.restore(path,model,opt,sched,scaler,sm,digest);d2,r2=sm.draw();assert np.array_equal(d1,d2) and r1==r2
    loss2,norm2=C.step(model,opt,sched,scaler,C.batch_at(ds,d2[:128]));assert loss1==loss2 and norm1==norm2
    E.exact(expected,dict(model=model.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),scaler=scaler.state_dict(),rng=C.rng_state(model),sampler=sm.state_dict()))
    write(E.STUDY/'readiness.json',dict(passed=True,protocol_sha256=digest,initial_tensor_sha256=initial,full128_resume_exact=True,scratch=True,checked_at=now()))
    ds.env.close()

def main():
    W.environment();W.verify_sources();W.stop();C.write=write
    from .engine import no_labels
    sys.addaudithook(no_labels)
    original_make=C.make_model
    def guarded_make(cfg):
        with W.gpu_slot():return original_make(cfg)
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['model1-check','model1-worker','rank','prepare','curriculum-check','worker']);p.add_argument('--dataset',choices=['A','B']);p.add_argument('--model',type=int,choices=[2,3]);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.stage.startswith('model1'):
        C.bootstrap()
        from . import model1 as M
        if a.stage=='model1-check':M.probe(a.dataset)
        else:M.worker(a.dataset,a.resume)
    elif a.stage=='rank':rank(a.dataset)
    elif a.stage=='prepare':prepare()
    elif a.stage=='curriculum-check':curriculum_check(a.dataset)
    else:
        E=bind(a.model,a.dataset)
        C.make_model=guarded_make
        E.worker(a.resume)
if __name__=='__main__':main()
