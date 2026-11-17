"""CryptBB v3 static Model2: original curriculum engine, new data/teacher paths."""
import argparse,copy,gc,io,json,os,shutil,sys,subprocess,time,traceback
from pathlib import Path
from ..decoder.common import read,write,sha,now,StageLock
from . import ab_curriculum as C
from . import cryptbb_v3_model1 as M1
ROOT=Path(__file__).resolve().parents[4];STUDY=ROOT/'handoff/cryptbb_v3_model2_rank1_v1_20261115'
SELECTION_SHA='82546ce9b99494f7268553802014b5070be4004a2764fbd4bdfd6443f9687051'

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

def rank_dir(tag):return ROOT/f'model2/rankings/cryptbb_v3_rank1_v1/{tag}'
def run_dir(number,tag):
    assert number==2
    return ROOT/f'model2/runs/cryptbb_v3_rank1_v1/{tag}'
def selection():
    assert sha(STUDY/'teacher_selection.json')==SELECTION_SHA
    return read(STUDY/'teacher_selection.json')

def teacher_row(tag):return selection()['selected']['B' if tag=='A' else 'A']
def teacher(tag):return ROOT/teacher_row(tag)['checkpoint']

def check_sources():
    p=read(STUDY/'protocol.json')
    for r in p['sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    assert sha(STUDY/'teacher_selection.json')==p['selection_protocol_sha256']==SELECTION_SHA
    return sha(STUDY/'protocol.json')

def bind():
    C.STUDY=STUDY;C.config=config;C.dataset=dataset;C.rank_dir=rank_dir;C.run_dir=run_dir;C.check_sources=check_sources;C.gpu_slot=M1.gpu_slot

def setup():
    STUDY.mkdir(exist_ok=True);inputs={};assert not (STUDY/'protocol.json').exists();selection()
    for tag in 'AB':
        M1.check_data(tag)
        ck=teacher(tag);row=teacher_row(tag);r=read(ck.parent/f"epoch_{row['selected_epoch']}_receipt.json");done=read(ck.parent/'training_completion.json')
        assert done['epoch']==20 and r['epoch']==row['selected_epoch'] and sha(ck)==r['sha256']==row['checkpoint_sha256']
        inputs[tag]={'data_sha256':read(ROOT/f'datasets/cryptbb_v3/{tag}/data_receipt.json')['lmdb_sha256'],'teacher':str(ck.relative_to(ROOT)),'teacher_sha256':r['sha256'],'teacher_epoch':row['selected_epoch']}
    files=[Path(__file__),Path(C.__file__),Path(M1.__file__),Path(__file__).with_name('ds2_data.py'),Path(__file__).with_name('ds3_data.py')]+list((ROOT/'model1/src/OpenOCR').rglob('*.py'))
    files += [ROOT/f'model1/configs/cryptbb_v3/{t}/prepared.yml' for t in 'AB']
    files += [ROOT/'datasets/cryptbb_v3/charset.txt',Path(__file__).parents[1]/'decoder/common.py']
    sources=[]
    for p in files:
        rel=p.relative_to(ROOT);dest=STUDY/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources.append({'path':rel.as_posix(),'sha256':sha(p)})
    protocol=dict(created_at=now(),family='CryptBB v3',model=2,inputs=inputs,sources=sources,seed=48,epochs=20,pool_size=100000,batch=128,steps_per_epoch=782,draws_per_epoch=100096,total_steps=15640,initialization='Independent scratch; never initialize from Model1',preprocessing='Exact Model1 RGB PIL bicubic32x128, ToTensor, Normalize(.5,.5); no new augmentation',optimizer='AdamW lr0.00035 wd0.05; OneCycle warmup2/20; BF16; clip20',teachers={t:dict(reader='B' if t=='A' else 'A',epoch=teacher_row(t)['selected_epoch'],checkpoint=teacher_row(t)['checkpoint'],sha256=teacher_row(t)['checkpoint_sha256']) for t in 'AB'},ranking='Original ab_curriculum.score_pool; FP32 free AR +1 refinement, synthetic-only summed character NLL, ordinal thirds with image-ID tie break; fixed for all20epochs',static_schedule={'1':[1,0,0],'2':[.5,.5,0],'3-20':[.15,.25,.60]},sampling='With replacement, fixed100096 draws/epoch',parallel='A/B concurrent processes; complete GPU batch lock; unchanged batch128',selection_protocol_sha256=SELECTION_SHA,reporting_selection='Independent original v1 masked rank-one exact-string agreement, epochs1-20 on251 unlabeled pixels, k0-5 original choose_k fallback5, highest factor earliest exact tie; freeze before scoring',protocol_revision='Selected cross-teachers v1; supersedes epoch20 teachers only for this separate run',real_labels_used=False,model3_authorized=False)
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
    digest=check_sources();verify_rankings();out=STUDY/'checks';out.mkdir(exist_ok=True);ds=dataset('A')
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
        log=(STUDY/f'pilot_{tag}.log').open('a',encoding='utf-8');p=subprocess.Popen([sys.executable,'-B','-u','-m',__package__+'.cryptbb_v3_model2_selected_v1','pilot','--dataset',tag],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);children.append((tag,p,log))
    try:
        codes={t:p.wait() for t,p,l in children};assert all(c==0 for c in codes.values()),codes
    finally:
        for t,p,l in children:l.close()
    pilots={t:read(STUDY/f'pair_pilot/{t}/completion.json') for t in 'AB'};assert all(r['passed'] for r in pilots.values())
    write(STUDY/'readiness.json',dict(passed=True,created_at=now(),protocol_sha256=digest,preprocessing_exact_matches=3,static_schedule_counts=counts,deterministic_scoring=True,checkpoint_next_step_max_tensor_error=error,resume_losses=[loss1,loss2],rng_scheduler_sampler_replay=True,paired_full_batch_pilots=pilots,real_labels_used=False))


def verify_rankings(persist=True):
    import numpy as np
    digest=check_sources();records={}
    for tag in 'AB':
        p=rank_dir(tag);r=read(p/'completion.json');row=teacher_row(tag)
        assert r['protocol_sha256']==digest and r['teacher_sha256']==row['checkpoint_sha256']==sha(teacher(tag))
        assert r['teacher']==('B' if tag=='A' else 'A') and r['images']==100000
        assert sha(p/'ranking.npz')==r['sha256'] and sha(p/'predictions.json')==r['predictions_sha256']
        predictions=read(p/'predictions.json');assert [x['index'] for x in predictions]==list(range(1,100001))
        with np.load(p/'ranking.npz') as z:
            ids=z['ids'];raw=z['nll'];buckets=z['buckets']
            assert np.array_equal(ids,np.arange(1,100001)) and np.isfinite(raw).all() and (raw>=0).all()
            expected=np.empty(100000,dtype=np.int64)
            for b,group in enumerate(np.array_split(np.lexsort((ids,raw)),3)):expected[group]=b
            assert np.array_equal(buckets,expected) and np.bincount(buckets).tolist()==[33334,33333,33333]
        sampler=C.sampler_for(tag);counts=[]
        for epoch in range(1,21):
            draws,receipt=C.draws_for(sampler,2,epoch)
            expected_counts=[100096,0,0] if epoch==1 else [50048,50048,0] if epoch==2 else [15014,25024,60058]
            assert receipt['draws']==expected_counts and len(draws)==100096
            assert np.bincount(sampler.buckets[draws],minlength=3).tolist()==expected_counts
            counts.append(expected_counts)
        records[tag]=dict(ranking_sha256=r['sha256'],teacher_epoch=row['selected_epoch'],sampler_counts=counts)
    if persist:write(STUDY/'ranking_verification.json',dict(passed=True,protocol_sha256=digest,datasets=records,created_at=now()))

def extra_preprocessing_checks():
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from tools.data.ratio_dataset_tvresize import RatioDataSetTVResize
    baseline=RatioDataSetTVResize.__new__(RatioDataSetTVResize)
    baseline.base_shape=[[64,64],[96,48],[112,40],[128,32]];baseline.base_h=32
    baseline.padding_rand=False;baseline.padding_doub=False;baseline.interpolation=T.InterpolationMode.BICUBIC
    baseline.transforms=T.Compose([T.ToTensor(),T.Normalize(.5,.5)]);baseline.error=0
    for tag in 'AB':
        ds=dataset(tag)
        try:
            for i in [0,49999,99999]:
                with ds.env.begin() as tx:im=Image.open(io.BytesIO(tx.get(f'image-{i+1:09d}'.encode()))).convert('RGB')
                assert torch.equal(baseline.resize_norm_img({'image':im},4,padding=False)['image'],ds[i][0])
        finally:ds.env.close()
    write(STUDY/'preprocessing_verification.json',dict(passed=True,exact_matches=6,datasets=['A','B']))

def child(stage,tag=None):
    args=[sys.executable,'-B','-u','-X','faulthandler','-m',__package__+'.cryptbb_v3_model2_selected_v1',stage]
    if tag:args+=['--dataset',tag]
    with (STUDY/(stage+('_'+tag if tag else '')+'.log')).open('a',encoding='utf8') as log:
        p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    import psutil
    write(STUDY/(stage+('_'+tag if tag else '')+'_launch.json'),dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),args=args,created_at=now()))
    return p

def coordinate():
    import psutil
    with StageLock(STUDY,'coordinator'):
        try:
            assert not (STUDY/'STOP').exists()
            assert all(not run_dir(2,t).exists() and not rank_dir(t).exists() for t in 'AB'),'Inspect existing artifacts; no implicit restart'
            for stage,tag in [('setup',None),('rank','A'),('rank','B'),('readiness',None)]:
                write(STUDY/'status.json',dict(state=stage,dataset=tag,pid=os.getpid(),updated_at=now()))
                p=child(stage,tag);assert p.wait()==0,(stage,tag)
            verify_rankings();r=read(STUDY/'readiness.json');assert r['passed'] and r['protocol_sha256']==check_sources()
            children={t:child('worker',t) for t in 'AB'}
            write(STUDY/'status.json',dict(state='training',pid=os.getpid(),updated_at=now()))
            first={};verified=False
            while any(p.poll() is None for p in children.values()):
                for t,p in children.items():assert p.poll() in (None,0),('worker failed',t,p.returncode)
                statuses={t:read(run_dir(2,t)/'status.json') for t in 'AB' if (run_dir(2,t)/'status.json').exists()}
                for t,r in statuses.items():
                    if r.get('global_step',0)>0 and t not in first:first[t]=r
                if not verified and len(first)==2 and all(statuses[t].get('global_step',0)>first[t]['global_step'] for t in 'AB'):
                    identities={t:read(STUDY/('worker_'+t+'.lock.json')) for t in 'AB'}
                    for t,identity in identities.items():
                        process=psutil.Process(identity['pid']);assert abs(process.create_time()-identity['process_created'])<1
                        assert 'worker' in process.cmdline() and t in process.cmdline()
                        init=read(run_dir(2,t)/'initialization.json');assert init['pretrained'] is False and init['seed']==48
                    assert read(run_dir(2,'A')/'initialization.json')['model_tensor_sha256']==read(run_dir(2,'B')/'initialization.json')['model_tensor_sha256']
                    write(STUDY/'launch_verified.json',dict(verified_at=now(),protocol_sha256=check_sources(),workers=identities,first=first,progress=statuses,scratch_initialization_verified=True))
                    verified=True
                time.sleep(5)
            for t,p in children.items():
                assert p.wait()==0
                r=read(run_dir(2,t)/'completion.json');assert r['epoch']==20 and sha(run_dir(2,t)/r['checkpoint'])==r['sha256']
            write(STUDY/'status.json',dict(state='complete_model2_pair',updated_at=now(),model3_started=False))
        except Exception:
            write(STUDY/'status.json',dict(state='failed',error=traceback.format_exc(),updated_at=now()))
            if 'children' in locals() and any(p.poll() is None for p in children.values()):
                (STUDY/'STOP').write_text('Coordinator failure; stop at committed epoch boundary. Inspect before resume.\n')
            raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['setup','rank','readiness','pilot','worker','coordinate']);ap.add_argument('--dataset',choices=['A','B']);ap.add_argument('--resume',action='store_true');a=ap.parse_args();bind();C.bootstrap()
    if a.stage=='coordinate':coordinate()
    elif a.stage=='setup':setup()
    elif a.stage=='rank':rank(a.dataset)
    elif a.stage=='readiness':extra_preprocessing_checks();readiness()
    else:
        if a.stage=='worker':
            verify_rankings(persist=False);r=read(STUDY/'readiness.json');assert r['passed'] and r['protocol_sha256']==check_sources()
        with StageLock(STUDY,('pilot_' if a.stage=='pilot' else 'worker_')+a.dataset):
            C.train(2,a.dataset,resume=a.resume,pilot=a.stage=='pilot')
            if a.stage=='worker':
                p=run_dir(2,a.dataset)/'completion.json';r=read(p);r.update(selection='Pending independent original v1 masked rank-one reporting rule',reporting_selection_pending=True);write(p,r)

