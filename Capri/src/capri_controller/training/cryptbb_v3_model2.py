"""CryptBB v3 static Model2: original curriculum engine, new data/teacher paths."""
import argparse,copy,gc,io,json,os,shutil,sys,subprocess
from pathlib import Path
from ..decoder.common import read,write,sha,now,StageLock
from . import ab_curriculum as C
from . import cryptbb_v3_model1 as M1
ROOT=Path(__file__).resolve().parents[4];STUDY=M1.STUDY/'model2'

def config(tag,out):
    import yaml
    cfg=yaml.safe_load((ROOT/f'model1/configs/cryptbb_v3/{tag}/prepared.yml').read_text())
    cfg['Global'].update(output_dir=str(out),save_res_path=str(out/'unused.txt'))
    assert cfg['Train']['loader']['min_ratio']==cfg['Train']['loader']['max_ratio']==4
    assert cfg['Global']['pretrained_model'] is None and cfg['Global']['checkpoints'] is None
    return cfg

def dataset(tag):
    import lmdb,numpy as np
    from .ds2_data import ReaderDataset
    env=lmdb.open(str(ROOT/f'datasets/cryptbb_v3/{tag}/train_lmdb'),readonly=True,lock=False,readahead=False)
    with env.begin() as tx:assert int(tx.get(b'num-samples'))==100000
    return ReaderDataset(env,np.arange(1,100001),ROOT/'datasets/cryptbb_v3/charset.txt')

def rank_dir(tag):return ROOT/f'model2/rankings/cryptbb_v3/{tag}'
def run_dir(number,tag):
    assert number==2
    return ROOT/f'model2/runs/cryptbb_v3/{tag}'
def teacher(tag):return ROOT/f'model1/runs/cryptbb_v3/{"B" if tag=="A" else "A"}/epoch_20.pth'

def check_sources():
    p=read(STUDY/'protocol.json')
    for r in p['sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert sha(M1.STUDY/'agreement_protocol.json')==p['selection_protocol_sha256']
    return sha(STUDY/'protocol.json')

def bind():
    C.STUDY=STUDY;C.config=config;C.dataset=dataset;C.rank_dir=rank_dir;C.run_dir=run_dir;C.check_sources=check_sources;C.gpu_slot=M1.gpu_slot

def setup():
    STUDY.mkdir(exist_ok=True);inputs={}
    for tag in 'AB':
        M1.check_data(tag)
        ck=teacher(tag);r=read(ck.parent/'epoch_20_receipt.json');done=read(ck.parent/'training_completion.json')
        assert done['epoch']==20 and r['epoch']==20 and sha(ck)==r['sha256']
        inputs[tag]={'data_sha256':read(ROOT/f'datasets/cryptbb_v3/{tag}/data_receipt.json')['lmdb_sha256'],'teacher':str(ck.relative_to(ROOT)),'teacher_sha256':r['sha256']}
    files=[Path(__file__),Path(C.__file__),Path(M1.__file__),Path(__file__).with_name('ds2_data.py'),Path(__file__).with_name('ds3_data.py')]+list((ROOT/'model1/src/OpenOCR').rglob('*.py'))
    sources=[]
    for p in files:
        rel=p.relative_to(ROOT);dest=STUDY/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources.append({'path':rel.as_posix(),'sha256':sha(p)})
    protocol=dict(created_at=now(),family='CryptBB v3',model=2,inputs=inputs,sources=sources,seed=48,epochs=20,pool_size=100000,batch=128,steps_per_epoch=782,draws_per_epoch=100096,total_steps=15640,initialization='Independent scratch; never initialize from Model1',preprocessing='Exact Model1 RGB PIL bicubic32x128, ToTensor, Normalize(.5,.5); no new augmentation',optimizer='AdamW lr0.00035 wd0.05; OneCycle warmup2/20; BF16; clip20',teachers={'A':'Model1-B epoch20','B':'Model1-A epoch20'},ranking='Original ab_curriculum.score_pool; FP32 free AR +1 refinement, synthetic-only summed character NLL, ordinal thirds with image-ID tie break; fixed for all20epochs',static_schedule={'1':[1,0,0],'2':[.5,.5,0],'3-20':[.15,.25,.60]},sampling='With replacement, fixed100096 draws/epoch',parallel='A/B concurrent processes; complete GPU batch lock; unchanged batch128',selection_protocol_sha256=sha(M1.STUDY/'agreement_protocol.json'),reporting_selection='Prospective same-epoch A/B agreement on251 unlabeled evaluation pixels, epochs2-20, earliest tie; labels not used',real_labels_used=False,model3_authorized=False)
    assert not (STUDY/'protocol.json').exists();write(STUDY/'protocol.json',protocol)

def rank(tag):
    import numpy as np,torch
    digest=check_sources();p=read(STUDY/'protocol.json');out=rank_dir(tag);assert not out.exists();out.mkdir(parents=True)
    ck=teacher(tag);assert sha(ck)==p['inputs'][tag]['teacher_sha256']
    with M1.gpu_slot():
        model,decoder=C.make_model(config(tag,out));state=torch.load(ck,map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict'],strict=True);del state
    ds=dataset(tag)
    try:raw,seconds=C.score_pool(model,decoder,ds,out)
    finally:ds.env.close()
    ids=np.arange(1,100001);buckets=np.empty(100000,dtype=np.int64)
    for b,group in enumerate(np.array_split(np.lexsort((ids,raw)),3)):buckets[group]=b
    np.savez_compressed(out/'ranking.npz',ids=ids,nll=raw,buckets=buckets)
    write(out/'completion.json',dict(dataset=tag,teacher='B' if tag=='A' else 'A',images=100000,seconds=seconds,protocol_sha256=digest,teacher_sha256=sha(ck),sha256=sha(out/'ranking.npz'),predictions_sha256=sha(out/'predictions.json'),bucket_counts=np.bincount(buckets).tolist()))

def readiness():
    import numpy as np,torch
    from PIL import Image
    from torchvision import transforms as T
    from tools.data.ratio_dataset_tvresize import RatioDataSetTVResize
    digest=check_sources();out=STUDY/'checks';out.mkdir(exist_ok=True);ds=dataset('A')
    baseline=RatioDataSetTVResize.__new__(RatioDataSetTVResize);baseline.base_shape=[[64,64],[96,48],[112,40],[128,32]];baseline.base_h=32;baseline.padding_rand=False;baseline.padding_doub=False;baseline.interpolation=T.InterpolationMode.BICUBIC;baseline.transforms=T.Compose([T.ToTensor(),T.Normalize(.5,.5)]);baseline.error=0
    for i in [0,49999,99999]:
        with ds.env.begin() as tx:im=Image.open(io.BytesIO(tx.get(f'image-{i+1:09d}'.encode()))).convert('RGB')
        assert torch.equal(baseline.resize_norm_img({'image':im},4,padding=False)['image'],ds[i][0])
    sampler=C.sampler_for('A');counts=[]
    for epoch in range(1,21):
        draws,r=C.draws_for(sampler,2,epoch);assert len(draws)==100096 and np.bincount(sampler.buckets[draws],minlength=3).tolist()==r['draws'];counts.append(r['draws'])
    cfg=config('A',out)
    with M1.gpu_slot():model,decoder=C.make_model(cfg);opt,sched,scaler=C.optimizer_for(model,cfg)
    small=C.batch_at(ds,range(8));sampler=C.sampler_for('A');C.draws_for(sampler,2,1)
    C.step(model,opt,sched,scaler,small)
    rng=C.rng_state(model);before=C.model_hash(model);model.eval();chars=(ROOT/'datasets/cryptbb_v3/charset.txt').read_text().splitlines()
    with M1.gpu_slot():a,pa=C.score_batch(model,decoder,small,chars);b,pb=C.score_batch(model,decoder,small,chars)
    assert np.array_equal(a,b) and pa==pb and before==C.model_hash(model);C.restore_rng(model,rng)
    saved=out/'resume_probe.pth';C.save_checkpoint(saved,model,opt,sched,scaler,sampler,cfg,1,[],digest)
    loss1,_=C.step(model,opt,sched,scaler,small);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};expected_lr=copy.deepcopy(sched.state_dict());expected_rng=C.rng_state(model)
    del model,opt,sched,scaler;gc.collect();torch.cuda.empty_cache()
    with M1.gpu_slot():model,decoder=C.make_model(cfg);opt,sched,scaler=C.optimizer_for(model,cfg)
    sm2=C.sampler_for('A');ck=C.load_checkpoint(saved,model,opt,sched,scaler,sm2,digest);del ck
    loss2,_=C.step(model,opt,sched,scaler,small);error=max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in model.state_dict().items());assert error<=1e-6 and abs(loss1-loss2)<=1e-6 and expected_lr==sched.state_dict()
    actual=C.rng_state(model);assert torch.equal(expected_rng['torch'],actual['torch']) and all(torch.equal(x,y) for x,y in zip(expected_rng['cuda'],actual['cuda'])) and expected_rng['modules']==actual['modules']
    assert np.array_equal(sampler.buckets,sm2.buckets) and sampler.epoch==sm2.epoch
    ds.env.close();del model,opt,sched,scaler,expected;gc.collect();torch.cuda.empty_cache()
    # A/B full-batch pilots are separate simultaneous processes, all weights disposable.
    children=[]
    for tag in 'AB':
        log=(STUDY/f'pilot_{tag}.log').open('a',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.cryptbb_v3_model2','pilot','--dataset',tag],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);children.append((tag,p,log))
    try:
        codes={t:p.wait() for t,p,l in children};assert all(c==0 for c in codes.values()),codes
    finally:
        for t,p,l in children:l.close()
    pilots={t:read(STUDY/f'pair_pilot/{t}/completion.json') for t in 'AB'};assert all(r['passed'] for r in pilots.values())
    write(STUDY/'readiness.json',dict(passed=True,created_at=now(),protocol_sha256=digest,preprocessing_exact_matches=3,static_schedule_counts=counts,deterministic_scoring=True,checkpoint_next_step_max_tensor_error=error,resume_losses=[loss1,loss2],rng_scheduler_sampler_replay=True,paired_full_batch_pilots=pilots,real_labels_used=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['setup','rank','readiness','pilot','worker']);ap.add_argument('--dataset',choices=['A','B']);ap.add_argument('--resume',action='store_true');a=ap.parse_args();bind();C.bootstrap()
    if a.stage=='setup':setup()
    elif a.stage=='rank':rank(a.dataset)
    elif a.stage=='readiness':readiness()
    else:
        with StageLock(STUDY,('pilot_' if a.stage=='pilot' else 'worker_')+a.dataset):
            C.train(2,a.dataset,resume=a.resume,pilot=a.stage=='pilot')
            if a.stage=='worker':
                p=run_dir(2,a.dataset)/'completion.json';r=read(p);r.update(selection='Pending frozen A/B agreement reporting rule',reporting_selection_pending=True);write(p,r)

