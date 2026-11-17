"""Restore the original CAPRI joint recipe on the prepared100k TRAIN split.

Frozen original decoder modules are reused. DINO alone is frozen; no standalone
count-transformer or stopped head-only weights are used. New runs only.
"""
import copy,math,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from . import architecture_data as D
from .architecture_training_data import Scenes,References
from .architecture_training import setup,report,loader,main_eval,update_ema
from .decoder import training,pilot_losses
from .decoder.timed_pilot import objective
from .decoder.joint_phase import JointPreparation
from .decoder.reference_sampling import ReferenceSampler
from .decoder.data import ShapeBatchPlan,exact_shape_collate
from .decoder.optimization import finite_update
from .decoder.checkpoints import save_epoch,load_resume,require_finite
from .decoder.common import ROOT,read,write,sha,now,StageLock

SEED=918
ORIGINAL=ROOT/'runs/capri_decoder_500k_20261007'
INITIAL=ORIGINAL/'training_recovery/baseline_localization/epoch_20000/checkpoint.pt'
INITIAL_SHA='6bee2dbcee7a965f24998855f343f545f2dfd075b2d3d0233409fa5cfb49d7e5'
EPOCHS=8

def sources():
    paths=[Path(__file__)]+[ROOT/'src/capri_controller'/name for name in ('architecture_training.py','architecture_training_data.py','architecture_data.py','count_transformer.py','cli.py')]+[ROOT/'configs/local.json']
    paths+=list((ROOT/'src/capri_controller/decoder').glob('*.py'))
    paths+=list(Path(read(ROOT/'configs/local.json')['decoder_dinov2_source']).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def construct():
    # Reproduce original warmup -> joint transition, including raw initialization.
    training.initialize();model=training.construct()
    saved=torch.load(INITIAL,map_location='cpu',weights_only=False);model.load_state_dict(saved['raw'],strict=True);del saved
    model.requires_grad_(True);model.set_dino_trainable(0);model.train()
    assert all(p.requires_grad==(not name.startswith('dino.')) for name,p in model.named_parameters())
    return model

def components(dataset):
    train=Scenes(dataset);dev=Scenes(dataset,'dev');refs=References(dataset);devrefs=References(dataset,'dev')
    assert len(train)==100000 and len(dev)==2000
    families={f:i for f,i in train.family_indices.items() if f in {r['family_group'] for r in refs.rows}}
    sampler=ReferenceSampler(refs,families,SEED);prepare=JointPreparation(sampler,4,2,'family')
    loss_fn=objective(families,training.LOCAL_WEIGHTS)
    original=read(ORIGINAL/'diagnostics/timed_pilot_20261008/protocol.json')['objective']
    for key in ('active_tasks','group_weights','task_scales','reference_policy','temperature','auxiliary_weight','auxiliary_task_weights'):
        if loss_fn.contract[key]!=original[key]:raise ValueError('Original objective differs: '+key)
    pilot_losses.install()
    return train,dev,refs,devrefs,prepare,loss_fn

def optimizer(model,total_steps):
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5,weight_decay=.01,foreach=True)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/100)*(.2+.8*.5*(1+math.cos(math.pi*min(s,total_steps)/total_steps))))
    return opt,sch,torch.amp.GradScaler('cuda',init_scale=16.)

def readiness(dataset,out):
    dataset=Path(dataset).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve previous readiness artifacts')
    setup(out)
    try:
        report(out,state='verifying_dataset_and_initialization')
        verification=D.verify(dataset)
        if sha(INITIAL)!=INITIAL_SHA:raise ValueError('Original warmup checkpoint changed')
        train,dev,refs,devrefs,prepare,loss_fn=components(dataset);model=construct()
        plan=ShapeBatchPlan(train,8,epoch=1,seed=SEED)
        assert sorted(i for b in plan.batches for i in b)==list(range(100000))
        torch.cuda.reset_peak_memory_stats()
        opt,sch,scaler=optimizer(model,len(plan)*EPOCHS)
        for group in opt.param_groups:group['lr']=5e-5
        maximum=max(plan.batches,key=lambda ids:len(ids)*train.rows[ids[0]]['shape'][0]*train.rows[ids[0]]['shape'][1])
        batch=exact_shape_collate([train[i] for i in maximum]);prepared=prepare(batch,dict(epoch=1,optimizer_update=1,consumed_batches=0))
        def forward():
            with torch.autocast('cuda',dtype=torch.float16):return loss_fn(model,prepared)
        report(out,state='checking_full_model_gradients')
        value,_,_=finite_update(model,opt,scaler,forward)
        groups={}
        for prefix in ('cnn.','dino_projection.','laterals.','refine.','memory.','decoder.','box_heads.','count.','font_','effect_','background_under.'):
            selected=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            groups[prefix]=bool(selected) and any(float(g.float().abs().sum())>0 for g in selected)
        assert all(groups.values()),groups
        assert all(p.grad is None and not p.requires_grad for n,p in model.named_parameters() if n.startswith('dino.'))
        maximum_peak=torch.cuda.max_memory_allocated()/2**30
        if maximum_peak>11.5:raise RuntimeError('Full joint model leaves insufficient GPU reserve')
        # Probe the original full objective on TRAIN only, then discard all updates.
        before=float(value);tiny=[]
        for _ in range(24):
            value,_,_=finite_update(model,opt,scaler,forward);tiny.append(float(value))
        after=sum(tiny[-4:])/4
        if not after<before:raise ValueError('Joint TRAIN tiny-learning gate failed')
        report(out,state='profiling_full_joint_training',learning_before=before,learning_after=after)
        sizes=[];seconds=[]
        for index,batch in enumerate(loader(train,plan,'main')):
            start=time.perf_counter();prepared=prepare(batch,dict(epoch=1,optimizer_update=index+1,consumed_batches=index))
            value,_,_=finite_update(model,opt,scaler,forward);torch.cuda.synchronize()
            if index>=4:sizes.append(len(batch['targets']));seconds.append(time.perf_counter()-start)
            if index==35:break
        # Full-model checkpoint round trip, including trainability and optimizer.
        ema=copy.deepcopy(model.state_dict());cp=save_epoch(out/'checkpoint_smoke',1,model,ema,opt,sch,scaler,dict(epoch=1,consumed_batches=0),'readiness',{})
        frozen_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        restored=load_resume(cp,model,opt,sch,scaler,'readiness')
        assert all(torch.equal(frozen_state[k],v.cpu()) for k,v in model.state_dict().items())
        del frozen_state,restored,ema
        # Exercise complete alias-aware evaluation on a small DEV-only diagnostic.
        dev.rows=dev.rows[:8];refs.rows=refs.rows[:8];devrefs.rows=devrefs.rows[:8]
        smoke=main_eval(model,dev,refs,devrefs,out/'evaluation_smoke.json');assert smoke['images']==8
        result=dict(passed=True,created_at=now(),dataset=str(dataset),integrity_sha256=sha(dataset/'integrity.json'),initialization=str(INITIAL),initialization_sha256=INITIAL_SHA,initialization_weights='raw',verification=verification,original_objective_equal=True,trainable_groups_receive_gradients=groups,dino_frozen_without_gradients=True,parameter_counts=model.parameter_counts(),max_area_peak_allocated_GiB=maximum_peak,tiny_train_loss=dict(before=before,after=after),images_per_second=sum(sizes)/sum(seconds),all_100k_indices_exactly_once=True,checkpoint_round_trip=True,evaluation_smoke_images=8,training_epochs=EPOCHS,sources=sources())
        write(out/'readiness.json',result);report(out,state='ready',result=result)
        for ds in (train,dev,refs,devrefs):ds.close()
        return result
    except Exception as exc:report(out,state='failed',error=repr(exc));raise

def run(dataset,out,ready,resume=False):
    dataset=Path(dataset).resolve();out=Path(out).resolve();ready=Path(ready).resolve()
    if out.exists() and not resume:raise FileExistsError('Preserve previous run; explicit resume required')
    setup(out)
    with StageLock(out,'training'):
      try:
        r=read(ready/'readiness.json')
        if not r['passed'] or r['sources']!=sources():raise ValueError('Passed readiness must match current code/config')
        if sha(dataset/'integrity.json')!=r['integrity_sha256'] or sha(INITIAL)!=INITIAL_SHA:raise ValueError('Data/initialization identity changed')
        # Explicitly refuse concurrent continuation of the stopped experiment.
        old=ROOT/'runs/capri_shared_count_training_100k_20261027/status.json'
        if old.exists() and read(old)['state'] not in ('stopped','failed','complete'):raise RuntimeError('Previous experiment is not stopped')
        if not resume:
            protocol=dict(created_at=now(),authorization='User: train it the old way',method='Original joint CAPRI recipe; all parameters except DINO trainable',initialization=dict(checkpoint=str(INITIAL),sha256=INITIAL_SHA,weights='raw',reason='Same localization-warmup checkpoint as original joint pilot; no weights from stopped run'),dataset=str(dataset),integrity_sha256=r['integrity_sha256'],epochs=EPOCHS,batch_size=8,workers=4,seed=SEED,optimizer=dict(name='AdamW',lr=5e-5,weight_decay=.01,foreach=True,clip=1.,schedule='100-update warmup; cosine to20% over8 full100k epochs'),checkpoints='Raw/EMA/optimizer/scheduler/scaler/RNG/sampler every500updates and each epoch; early step25 recovery',selection='Fixed final epoch8 EMA; no automatic deployment; synthetic DEV diagnostics only',differences_from_original='100k TRAIN instead of20k; no old8h wall cap; portable E dataset; alias-aware evaluation; same original joint objective, trainability, initialization, references and optimizer recipe.',separate_count_transformer='Stopped; neither retrained nor used as count authority',sources=sources())
            write(out/'protocol.json',protocol)
            for rel in protocol['sources']:
                dest=out/'provenance'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        protocol=read(out/'protocol.json');ph=D.jhash(protocol)
        if protocol['sources']!=sources():raise ValueError('Run source snapshot changed')
        train,dev,refs,devrefs,prepare,loss_fn=components(dataset)
        write(out/'objective.json',loss_fn.contract);write(out/'reference_policy.json',prepare.contract)
        plans=[ShapeBatchPlan(train,8,epoch=e,seed=SEED).batches for e in range(1,EPOCHS+1)]
        assert all(sorted(i for b in p for i in b)==list(range(100000)) for p in plans)
        model=construct();opt,sch,scaler=optimizer(model,sum(map(len,plans)));ema=copy.deepcopy(model.state_dict())
        pointer=out/'resume.json';step=0;first_epoch=1;cursor=0
        if resume and pointer.exists():
            state=load_resume(read(pointer)['checkpoint'],model,opt,sch,scaler,ph)
            ema={k:v.cuda() for k,v in state['ema'].items()};sample=state['sampler_state'];step=sample['global_step'];first_epoch=sample['epoch'];cursor=sample['consumed_batches'];del state
        estimate=800000/r['images_per_second']
        if not (out/'eta.json').exists():write(out/'eta.json',dict(created_at=now(),expected_seconds=estimate*1.25,range_seconds=[estimate*1.1,estimate*1.5],basis='Full joint GPU throughput, plus10-50% validation/loader/checkpoint allowance',started_unix=time.time()))
        if not (out/'initial_dev.json').exists():
            report(out,state='evaluating_initial_joint_model')
            main_eval(model,dev,refs,devrefs,out/'initial_dev.json')
        if (out/'STOP').exists():report(out,state='stopped',stage='joint',epoch=first_epoch,images=0);return
        def checkpoint(epoch,index,complete):
            dest=out/('epochs' if complete else 'recoveries');number=epoch if complete else step
            path=dest/f'epoch_{number:04d}'/'checkpoint.pt'
            if path.exists():
                if sha(path)!=read(path.parent/'receipt.json')['sha256']:raise ValueError('Existing checkpoint changed')
            else:
                sample=dict(epoch=epoch,consumed_batches=index,global_step=step,snapshot=sha(dataset/'train.jsonl'))
                path=save_epoch(dest,number,model,ema,opt,sch,scaler,sample,ph,dict(epoch_complete=complete))
            write(pointer,dict(checkpoint=str(path),epoch=epoch,global_step=step,epoch_complete=complete));return path
        for epoch in range(first_epoch,EPOCHS+1):
            plan=plans[epoch-1];begin=cursor if epoch==first_epoch else 0;images=sum(len(ids) for ids in plan[:begin]);fresh=0;started=time.time();model.train()
            for index,batch in enumerate(loader(train,plan[begin:],'main'),start=begin):
                if (out/'STOP').exists():
                    if step:checkpoint(epoch,index,False)
                    report(out,state='stopped',stage='joint',epoch=epoch,images=images,step=step);return
                prepared=prepare(batch,dict(epoch=epoch,optimizer_update=step+1,consumed_batches=index))
                def forward():
                    with torch.autocast('cuda',dtype=torch.float16):return loss_fn(model,prepared)
                value,norm,retries=finite_update(model,opt,scaler,forward);sch.step();step+=1;update_ema(model,ema,step)
                n=len(batch['targets']);images+=n;fresh+=n
                if step%25==1:
                    speed=fresh/max(.01,time.time()-started);remaining=((EPOCHS-epoch)*100000+100000-images)/speed*1.25
                    report(out,state='training',stage='joint',epoch=epoch,epochs=EPOCHS,images=images,epoch_images=100000,step=step,loss=float(value),gradient_norm=float(norm),amp_retries=retries,images_per_second=speed,estimated_remaining_seconds=remaining)
                if step==25 or step%500==0:checkpoint(epoch,index+1,False)
            assert images==100000
            cp=checkpoint(epoch,len(plan),True);metric=out/f'dev_epoch_{epoch:02d}.json'
            report(out,state='evaluating',stage='joint',epoch=epoch,checkpoint=str(cp))
            if not metric.exists():
                raw={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.load_state_dict(ema)
                main_eval(model,dev,refs,devrefs,metric);model.load_state_dict(raw);del raw
            write(out/f'epoch_{epoch:02d}_complete.json',dict(checkpoint=str(cp),metrics=str(metric),images=images,epoch_complete=True))
            metrics=read(metric);initial=read(out/'initial_dev.json')
            # Preserve the original fixed synthetic regression guard.
            if metrics['count_comparator']['accuracy']<initial['count_comparator']['accuracy']-.10 or (metrics['localization']['f1'] or 0)<(initial['localization']['f1'] or 0)-.10:
                report(out,state='stopped_quality_guard',stage='joint',completed_epochs=epoch,reason='More than10percentage-point synthetic count/localization regression');return
        final=out/'epochs'/f'epoch_{EPOCHS:04d}'/'checkpoint.pt'
        write(out/'completion.json',dict(completed_at=now(),epochs=EPOCHS,images_per_epoch=100000,checkpoint=str(final),sha256=sha(final),weights='EMA',activated=False))
        report(out,state='complete',stage='joint',completed_epochs=EPOCHS,activated=False)
        for ds in (train,dev,refs,devrefs):ds.close()
      except Exception as exc:report(out,state='failed',error=repr(exc));raise
