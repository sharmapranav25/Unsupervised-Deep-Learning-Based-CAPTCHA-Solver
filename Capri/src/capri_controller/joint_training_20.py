"""Twenty full100k epochs: jointly update CAPRI, all DINO blocks, and separate count."""
import copy,math,shutil,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import architecture_data as D
from .architecture_training_data import Scenes,References,Counts,count_pixels
from .architecture_training import setup,report,main_eval,count_eval,update_ema
from .count_transformer import CountTransformer
from .decoder import training,pilot_losses
from .decoder.model import CapriDecoder
from .decoder.timed_pilot import objective
from .decoder.joint_phase import JointPreparation
from .decoder.reference_sampling import ReferenceSampler
from .decoder.data import ShapeBatchPlan,exact_shape_collate
from .decoder.optimization import finite_update
from .decoder.checkpoints import save_epoch,load_resume,require_finite
from .decoder.common import ROOT,read,write,sha,now,StageLock

EPOCHS=20
SEED=918
SOURCE=ROOT/'runs/capri_decoder_500k_20261007'
INITIAL=SOURCE/'training_recovery/baseline_localization/epoch_20000/checkpoint.pt'
INITIAL_SHA='6bee2dbcee7a965f24998855f343f545f2dfd075b2d3d0233409fa5cfb49d7e5'

class AllTrainableDecoder(CapriDecoder):
    def set_dino_trainable(self,last_blocks):
        if last_blocks=='all':
            self.dino_last_blocks=len(self.dino.blocks);self.dino.requires_grad_(True);self.dino.eval()
        else:super().set_dino_trainable(last_blocks)

class JointCapri(nn.Module):
    def __init__(self):
        super().__init__();c=read(ROOT/'configs/local.json')
        self.capri=AllTrainableDecoder(c['decoder_dinov2_source'],c['decoder_dinov2_checkpoint'],backbone=c['decoder_backbone'])
        state=torch.load(INITIAL,map_location='cpu',weights_only=False)
        self.capri.load_state_dict(state['raw'],strict=True);del state
        self.capri.requires_grad_(True);self.capri.set_dino_trainable('all')
        self.counter=CountTransformer()
    def resume_configuration(self):
        return dict(architecture='joint-capri-and-separate-count-v1',capri=self.capri.resume_configuration(),counter=self.counter.resume_configuration(),all_components_trainable=True)

class JointScenes(Scenes):
    def __getitem__(self,i):
        item=super().__getitem__(i);r=self.rows[i]
        rgb=D.pixels(self.archive(r['archive']).read(r['stem']+'.png'),r['pixel_sha256'])
        x,v=count_pixels(rgb);item['count_inputs']=dict(pixels=x,valid_pixels=v)
        return item

def collate(items):
    result=exact_shape_collate(items)
    result['count_inputs']={k:torch.stack([x['count_inputs'][k] for x in items]) for k in ('pixels','valid_pixels')}
    return result

def loader(ds,batches):
    return DataLoader(ds,batch_sampler=batches,collate_fn=collate,num_workers=4,pin_memory=True,prefetch_factor=2,generator=torch.Generator().manual_seed(SEED))

def source_hashes():
    paths=[Path(__file__),ROOT/'configs/local.json']+[ROOT/'src/capri_controller'/n for n in ('architecture_training.py','architecture_training_data.py','architecture_data.py','count_transformer.py','cli.py')]
    paths+=list((ROOT/'src/capri_controller/decoder').glob('*.py'))
    paths+=list(Path(read(ROOT/'configs/local.json')['decoder_dinov2_source']).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def components(dataset):
    train=JointScenes(dataset);dev=Scenes(dataset,'dev');refs=References(dataset);devrefs=References(dataset,'dev');countdev=Counts(dataset,'dev')
    assert len(train)==100000 and len(dev)==len(countdev)==2000
    fam={f:i for f,i in train.family_indices.items() if f in {r['family_group'] for r in refs.rows}}
    prepare=JointPreparation(ReferenceSampler(refs,fam,SEED),4,2,'family')
    loss=objective(fam,training.LOCAL_WEIGHTS);pilot_losses.install()
    return train,dev,refs,devrefs,countdev,prepare,loss

def optimizer(model,steps):
    groups=[dict(params=list(model.capri.dino.parameters()),lr=5e-6,name='pretrained_dino'),dict(params=[p for n,p in model.capri.named_parameters() if not n.startswith('dino.')],lr=5e-5,name='capri'),dict(params=list(model.counter.parameters()),lr=3e-4,name='independent_count')]
    opt=torch.optim.AdamW(groups,weight_decay=.01,foreach=True)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/300)*(.02+.98*.5*(1+math.cos(math.pi*min(s,steps)/steps))))
    return opt,sch,torch.amp.GradScaler('cuda',init_scale=16.)

def combined_loss(model,prepared,main_objective,parts=None):
    joint=main_objective(model.capri,prepared)
    inputs={k:v.cuda(non_blocking=True) for k,v in prepared['count_inputs'].items()}
    logits=model.counter(**inputs)
    targets=torch.tensor([int(t['count'])-2 for t in prepared['targets']],device='cuda')
    separate=F.cross_entropy(logits.float(),targets)
    if parts is not None:parts.update(capri=float(joint.detach()),independent_count=float(separate.detach()))
    return joint+separate

def gradients(model):
    result={}
    for prefix in ('capri.cnn.','capri.dino.','capri.dino_projection.','capri.laterals.','capri.refine.','capri.memory.','capri.decoder.','capri.box_heads.','capri.count.','capri.font_','capri.effect_','capri.background_under.','counter.'):
        gs=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        result[prefix]=bool(gs) and any(float(g.float().abs().sum())>0 for g in gs)
    return result

def readiness(dataset,out,batch_size=8):
    out=Path(out).resolve();dataset=Path(dataset).resolve()
    if out.exists():raise FileExistsError('Preserve prior check results')
    setup(out)
    try:
        report(out,state='verifying_data_and_initialization')
        verification=D.verify(dataset)
        assert sha(INITIAL)==INITIAL_SHA
        train,dev,refs,devrefs,countdev,prepare,loss_fn=components(dataset)
        # Exact same raw RGB drives both branches, with pixel-only preprocessing.
        c=Counts(dataset);a=train[0];b=c[0]
        assert torch.equal(a['count_inputs']['pixels'],b[0]) and torch.equal(a['count_inputs']['valid_pixels'],b[1]);c.close()
        model=JointCapri().cuda();assert all(p.requires_grad for p in model.parameters())
        plan=ShapeBatchPlan(train,batch_size,epoch=1,seed=SEED)
        assert sorted(i for batch in plan.batches for i in batch)==list(range(100000))
        opt,sch,scaler=optimizer(model,len(plan)*EPOCHS)
        for group,lr in zip(opt.param_groups,(5e-6,5e-5,3e-4)):group['lr']=lr
        torch.cuda.reset_peak_memory_stats()
        maximum=max(plan.batches,key=lambda b:len(b)*train.rows[b[0]]['shape'][0]*train.rows[b[0]]['shape'][1])
        batch=collate([train[i] for i in maximum]);prepared=prepare(batch,dict(epoch=1,optimizer_update=1,consumed_batches=0));parts={}
        def forward():
            with torch.autocast('cuda',dtype=torch.float16):return combined_loss(model,prepared,loss_fn,parts)
        report(out,state='checking_gradients_in_all_components',batch_size=batch_size)
        value,_,_=finite_update(model,opt,scaler,forward);flow=gradients(model)
        assert all(flow.values()),flow
        maximum_peak=torch.cuda.max_memory_allocated()/2**30
        before=dict(parts);tiny=[]
        for _ in range(24):
            finite_update(model,opt,scaler,forward);tiny.append(dict(parts))
        after={k:sum(v[k] for v in tiny[-4:])/4 for k in before}
        assert after['capri']<before['capri'] and after['independent_count']<before['independent_count'],(before,after)
        report(out,state='profiling_joint_updates',loss_before=before,loss_after=after)
        sizes=[];times=[]
        for index,batch in enumerate(loader(train,plan)):
            start=time.perf_counter();prepared=prepare(batch,dict(epoch=1,optimizer_update=index+1,consumed_batches=index))
            finite_update(model,opt,scaler,forward);torch.cuda.synchronize()
            if index>=4:sizes.append(len(batch['targets']));times.append(time.perf_counter()-start)
            if index==35:break
        ema=copy.deepcopy(model.state_dict());cp=save_epoch(out/'checkpoint_smoke',1,model,ema,opt,sch,scaler,dict(epoch=1,consumed_batches=0),'joint-readiness',{})
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        restored=load_resume(cp,model,opt,sch,scaler,'joint-readiness')
        assert all(torch.equal(expected[k],v.cpu()) for k,v in model.state_dict().items())
        del expected,restored,ema
        dev.rows=dev.rows[:8];refs.rows=refs.rows[:8];devrefs.rows=devrefs.rows[:8];countdev.rows=countdev.rows[:16]
        main_eval(model.capri,dev,refs,devrefs,out/'main_eval_smoke.json');count_eval(model.counter,countdev,out/'count_eval_smoke.json')
        result=dict(passed=True,created_at=now(),dataset=str(dataset),integrity_sha256=sha(dataset/'integrity.json'),initialization=str(INITIAL),initialization_sha256=INITIAL_SHA,verification=verification,all_parameters_trainable=True,gradient_flow=flow,dino_blocks_trainable=len(model.capri.dino.blocks),loss_before=before,loss_after=after,images_per_second=sum(sizes)/sum(times),max_area_peak_allocated_GiB=maximum_peak,batch_size=batch_size,epochs=EPOCHS,train_images=100000,dev_images=2000,parameters=sum(p.numel() for p in model.parameters()),checkpoint_round_trip=True,both_evaluations_passed=True,all_100k_indices_once=True,sources=source_hashes(),architecture=model.resume_configuration())
        write(out/'readiness.json',result);report(out,state='ready',result=result)
        for ds in (train,dev,refs,devrefs,countdev):ds.close()
    except Exception as exc:report(out,state='failed',error=repr(exc));raise

def evaluate_pair(model,dev,refs,devrefs,countdev,out,prefix):
    out=Path(out);main=main_eval(model.capri,dev,refs,devrefs,out/(prefix+'_main.json'))
    counter=count_eval(model.counter,countdev,out/(prefix+'_count.json'))
    saved=torch.load(out/(prefix+'_count.pt'),map_location='cpu',weights_only=False)
    estimates={countdev.rows[int(i)]['id']:int(p.argmax())+2 for i,p in zip(saved['indices'],saved['logits'])}
    rows=D.read_rows(out/(prefix+'_main.predictions.jsonl'))
    for r in rows:r.update(independent_count=estimates[r['id']],count_region_disagreement=estimates[r['id']]!=r['detected_region_count'])
    D.jsonl(out/(prefix+'_consistency.jsonl'),rows)
    summary=dict(main=main,count=counter,independent_count_region_disagreements=sum(r['count_region_disagreement'] for r in rows),images=len(rows))
    write(out/(prefix+'.json'),summary);return summary

def run(dataset,out,ready,resume=False):
    dataset=Path(dataset).resolve();out=Path(out).resolve();ready=Path(ready).resolve()
    if out.exists() and not resume:raise FileExistsError('Preserve existing run; explicit resume required')
    setup(out)
    with StageLock(out,'training'):
      try:
        r=read(ready/'readiness.json')
        assert r['passed'] and r['sources']==source_hashes(),'Readiness code/config changed'
        assert sha(dataset/'integrity.json')==r['integrity_sha256'] and sha(INITIAL)==INITIAL_SHA
        old=ROOT/'runs/capri_shared_count_training_100k_20261027/status.json'
        assert not old.exists() or read(old)['state'] in ('stopped','failed','complete'),'Earlier run still active'
        if not resume:
            protocol=dict(created_at=now(),authorization='User: train the new CAPRI all components together on100k images for20epochs, with font intelligence and the separate count transformer.',epochs=EPOCHS,images_per_epoch=100000,dev_images=2000,batch_size=r['batch_size'],seed=SEED,initialization=dict(capri=str(INITIAL),sha256=INITIAL_SHA,weights='raw localization-warmup; new joint optimizer',counter='random seed918; no stopped-run weights'),trainability='All CNN/DINO/fusion/localization/font/background/count parameters trainable; DINO stochastic layers kept in eval mode but gradients enabled through all12blocks',objective='Original joint CAPRI task-group objective plus independent count CE at weight1 on every identical image batch; TRAIN reference support every4updates,2specimens',optimizer=dict(name='AdamW',capri_lr=5e-5,dino_lr=5e-6,counter_lr=3e-4,weight_decay=.01,schedule='300update warmup, cosine to2percent over20full epochs',clip=1.),font_safeguards='Existing supervised-reference loss excludes duplicate glyph positives and cross-family identical-specimen negatives. DEV never optimization. Alias-aware evaluation retains tied family candidates.',selection='Fixed epoch20 EMA; preserve every epoch and report synthetic DEV, no real labels or automatic deployment',dataset=str(dataset),integrity_sha256=r['integrity_sha256'],sources=source_hashes(),readiness=str(ready),checkpoint_policy='Full combined raw/EMA/optimizer/scheduler/scaler/RNG/sampler at step25, every500updates and everyepoch',scope='Single joint20epoch training run; no staged head-only/count-only phases; distinct count architecture remains pixel-only and does not force localization boxes.')
            write(out/'protocol.json',protocol)
            for rel in protocol['sources']:
                dest=out/'provenance'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        protocol=read(out/'protocol.json');ph=D.jhash(protocol)
        assert protocol['sources']==source_hashes()
        train,dev,refs,devrefs,countdev,prepare,loss_fn=components(dataset)
        write(out/'main_objective.json',loss_fn.contract);write(out/'reference_policy.json',prepare.contract)
        plans=[ShapeBatchPlan(train,r['batch_size'],epoch=e,seed=SEED).batches for e in range(1,EPOCHS+1)]
        assert all(sorted(i for b in p for i in b)==list(range(100000)) for p in plans)
        model=JointCapri().cuda();opt,sch,scaler=optimizer(model,sum(map(len,plans)));ema=copy.deepcopy(model.state_dict())
        step=0;first_epoch=1;cursor=0;pointer=out/'resume.json'
        if resume and pointer.exists():
            state=load_resume(read(pointer)['checkpoint'],model,opt,sch,scaler,ph);ema={k:v.cuda() for k,v in state['ema'].items()};sample=state['sampler_state'];step=sample['global_step'];first_epoch=sample['epoch'];cursor=sample['consumed_batches'];del state
        if not (out/'eta.json').exists():
            seconds=EPOCHS*100000/r['images_per_second'];write(out/'eta.json',dict(created_at=now(),started_unix=time.time(),expected_seconds=seconds*1.25,range_seconds=[seconds*1.1,seconds*1.5],basis='Measured full joint GPU speed plus10-50percent loader/evaluation/checkpoint overhead'))
        if not (out/'initial_dev.json').exists():
            report(out,state='evaluating_initial_models',stage='joint_all')
            evaluate_pair(model,dev,refs,devrefs,countdev,out,'initial_dev')
        def checkpoint(epoch,index,complete):
            dest=out/('epochs' if complete else 'recoveries');number=epoch if complete else step;path=dest/f'epoch_{number:04d}'/'checkpoint.pt'
            if path.exists():assert sha(path)==read(path.parent/'receipt.json')['sha256']
            else:path=save_epoch(dest,number,model,ema,opt,sch,scaler,dict(epoch=epoch,consumed_batches=index,global_step=step,snapshot=sha(dataset/'train.jsonl')),ph,dict(epoch_complete=complete))
            write(pointer,dict(checkpoint=str(path),epoch=epoch,global_step=step,epoch_complete=complete));return path
        for epoch in range(first_epoch,EPOCHS+1):
            plan=plans[epoch-1];begin=cursor if epoch==first_epoch else 0;images=sum(len(b) for b in plan[:begin]);fresh=0;start=time.time();model.train()
            for index,batch in enumerate(loader(train,plan[begin:]),start=begin):
                if (out/'STOP').exists():
                    if step:checkpoint(epoch,index,False)
                    report(out,state='stopped',stage='joint_all',epoch=epoch,images=images,step=step);return
                prepared=prepare(batch,dict(epoch=epoch,optimizer_update=step+1,consumed_batches=index));parts={}
                def forward():
                    with torch.autocast('cuda',dtype=torch.float16):return combined_loss(model,prepared,loss_fn,parts)
                value,norm,retries=finite_update(model,opt,scaler,forward);sch.step();step+=1;update_ema(model,ema,step);n=len(batch['targets']);images+=n;fresh+=n
                if step%25==1:
                    speed=fresh/max(.01,time.time()-start);remaining=((EPOCHS-epoch)*100000+100000-images)/speed*1.25
                    report(out,state='training',stage='joint_all',epoch=epoch,epochs=EPOCHS,images=images,epoch_images=100000,step=step,loss=float(value),loss_parts=parts,gradient_norm=float(norm),amp_retries=retries,images_per_second=speed,estimated_remaining_seconds=remaining)
                if step==25 or step%500==0:checkpoint(epoch,index+1,False)
            assert images==100000
            cp=checkpoint(epoch,len(plan),True);report(out,state='evaluating',stage='joint_all',epoch=epoch,checkpoint=str(cp))
            if not (out/f'dev_epoch_{epoch:02d}.json').exists():
                raw={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.load_state_dict(ema)
                evaluate_pair(model,dev,refs,devrefs,countdev,out,f'dev_epoch_{epoch:02d}');model.load_state_dict(raw);del raw
            write(out/f'epoch_{epoch:02d}_complete.json',dict(checkpoint=str(cp),images=images,epoch_complete=True))
        cp=out/'epochs'/f'epoch_{EPOCHS:04d}'/'checkpoint.pt'
        write(out/'completion.json',dict(completed_at=now(),epochs=EPOCHS,images_per_epoch=100000,checkpoint=str(cp),sha256=sha(cp),weights='EMA for both models',activated=False))
        report(out,state='complete',stage='joint_all',epochs=EPOCHS,activated=False)
        for ds in (train,dev,refs,devrefs,countdev):ds.close()
      except Exception as exc:report(out,state='failed',error=repr(exc));raise
