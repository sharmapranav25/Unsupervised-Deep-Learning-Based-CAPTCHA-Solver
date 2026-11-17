"""One joint optimization loop using the original CAPRI recipe with explicit additions."""
import copy,json,math,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import data as D,legacy_losses as L
from .model import LightweightCapri
from ..decoder.common import ROOT,read,write,sha,now,StageLock
from ..decoder.checkpoints import save_epoch,load_resume,require_finite

EPOCHS=20;SEED=0
BASE=dict(lr=.0005,min_lr=.00001,warmup_steps=300,weight_decay=.05,ema=.999,grad_clip=1.,batch=128)

def setup(out):
    out=Path(out).resolve()
    if out.drive.upper()!='E:':raise ValueError('E-drive only')
    out.mkdir(parents=True,exist_ok=True)
    for key in ('TEMP','TMP','TORCH_HOME','HF_HOME','XDG_CACHE_HOME'):
        p=out/'temporary';p.mkdir(exist_ok=True);os.environ[key]=str(p)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')

def report(out,**kw):
    v=dict(time=now(),pid=os.getpid(),**kw);write(Path(out)/'status.json',v)
    with (Path(out)/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(v)+'\n')
    print(json.dumps(v),flush=True)

def sources():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'src/capri_controller/count_transformer.py',ROOT/'src/capri_controller/cli.py',ROOT/'configs/local.json']
    paths+=list((ROOT/'src/capri_controller/decoder').glob('*.py'))
    paths+=list(Path(read(ROOT/'configs/local.json')['decoder_dinov2_source']).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def tensor_batch(nb,indices=None):
    b={k:torch.from_numpy(np.ascontiguousarray(v if indices is None else v[indices])).cuda(non_blocking=True) for k,v in nb.items()}
    b['image']=b['image'].float()/255;return b

def schedule(step,total):
    if step<300:return (step+1)/300
    return .02+.98*.5*(1+math.cos(math.pi*min(1.,(step-300)/max(1,total-300))))

def optimizer(model,total):
    groups={}
    for n,p in model.named_parameters():
        branch='dino' if n.startswith('dino.') else 'new' if n.startswith(('font_','counter.')) else 'capri'
        decay=p.ndim>=2 and not any(s in n for s in ('bias','emb','pos'))
        groups.setdefault((branch,decay),[]).append(p)
    opt=torch.optim.AdamW([dict(params=p,lr=5e-6 if branch=='dino' else BASE['lr'],weight_decay=BASE['weight_decay'] if decay else 0.,name=branch) for (branch,decay),p in groups.items()],betas=(.9,.95),foreach=True)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:schedule(s,total));return opt,sch

@torch.no_grad()
def ema_update(model,ema,step):
    d=min(BASE['ema'],step/(step+9))
    for k,v in model.state_dict().items():
        if v.is_floating_point():ema[k].mul_(d).add_(v.detach(),alpha=1-d)
        else:ema[k].copy_(v)

class References:
    def __init__(self,ds):
        assert ds.split=='references_train';self.ds=ds;self.by_family={};self.alias=set()
        for i,r in enumerate(ds.rows):
            assert r['split']=='train'
            family=ds.family_ids[r['family_group']];self.by_family.setdefault(family,[]).append(i)
            ids=[ds.family_ids[f] for f in r['pixel_alias_family_groups'] if f in ds.family_ids]
            self.alias.update((a,b) for a in ids for b in ids if a!=b)
    def choose(self,nb,step):
        rng=np.random.default_rng(SEED+step*31337);families=sorted({int(f) for f in nb['single_family'] if int(f)>=0})
        rng.shuffle(families);indices=[]
        for f in families[:8]:indices.append(int(rng.choice(self.by_family[f])))
        return indices

def contrastive(z,families,hashes,aliases):
    if len(z)<2:return z.sum()*0
    same=families[:,None]==families[None,:]
    allowed=~torch.eye(len(z),dtype=torch.bool,device=z.device)
    duplicate=torch.tensor([[a==b for b in hashes] for a in hashes],device=z.device);allowed &= ~duplicate
    for a,b in aliases:allowed &= ~((families[:,None]==a)&(families[None,:]==b))
    positive=same&allowed;eligible=positive.any(-1)&((~same)&allowed).any(-1)
    if not eligible.any():return z.sum()*0
    logits=z[eligible].float()@z.float().T/.1
    lp=logits-torch.logsumexp(logits.masked_fill(~allowed[eligible],-torch.inf),-1,keepdim=True)
    return -(lp.masked_fill(~positive[eligible],0).sum(-1)/positive[eligible].sum(-1)).mean()

def core_losses(o,b,eff_w,txt_w):
    # Same seven fixed loss groups; mask fields absent in the prepared corpus.
    per={}
    per['eff_presence']=L.presence_loss(o['eff_presence'],b['eff_present'])
    per['eff_ordinal']=L.ordinal_loss(o['eff_ordinal'],b['eff_level'].long(),b['eff_level']>0,eff_w)
    per['txt_presence']=L.presence_loss(o['txt_presence'],b['txt_present'])
    per['txt_ordinal']=L.ordinal_loss(o['txt_ordinal'],b['txt_level'].long(),b['txt_level']>0,txt_w)
    groups={k:(v[0]*(v[1]>0)).sum()/(v[1]>0).sum().clamp_min(1) for k,v in per.items()}
    valid=b['ctc_valid'].bool()
    groups['ctc']=L.ctc_loss(o['ctc'][valid],b['chars'][valid],b['length'][valid],b['cols'][valid]) if valid.any() else o['ctc'].float().sum()*0
    groups['length']=F.cross_entropy(o['length'].float(),b['length'].long()-2)
    f=b['font'].long();fp=[]
    for k,col in [('font_cat',0),('font_stroke',1)]:
        valid=f[:,col]>=0;fp.append(F.cross_entropy(o[k][valid].float(),f[valid,col]) if valid.any() else o[k].float().sum()*0)
    valid=f[:,6]>=0;fp.append(F.binary_cross_entropy_with_logits(o['font_italic'][valid].float(),f[valid,6].float()) if valid.any() else o['font_italic'].float().sum()*0)
    fl,fn=L.ordinal_loss(o['font_ordinal'],f[:,2:6].clamp_min(0),f[:,2:6]>=0);fp.append((fl*(fn>0)).sum()/(fn>0).sum().clamp_min(1))
    groups['font']=sum(fp)/4
    groups['separate_count']=F.cross_entropy(o['independent_count'].float(),b['length'].long()-2)
    return groups

def weights(ds):
    return (L.level_weights(ds.get('eff_level',range(len(ds)))).cuda(),L.level_weights(ds.get('txt_level',range(len(ds)))).cuda())

def step_update(model,opt,nb,micro,ref_sampler,step,eff_w,txt_w,gradients=False):
    opt.zero_grad(set_to_none=True);n=len(nb['image']);parts={};refids=ref_sampler.choose(nb,step)
    # Reference support is attached to the microbatch containing its chosen family.
    for start in range(0,n,micro):
        end=min(n,start+micro);b=tensor_batch(nb,slice(start,end))
        families_here={int(f) for f in nb['single_family'][start:end] if f>=0}
        chosen=[i for i in refids if ref_sampler.ds.family_ids[ref_sampler.ds.rows[i]['family_group']] in families_here]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            o=model(b['image'],b['cols'],b['widths']);groups=core_losses(o,b,eff_w,txt_w)
            use=b['single_family']>=0;z=o['font_embedding'][use];f=b['single_family'][use].long()
            hashes=[ref_sampler.scene_hashes[int(i)] for i in nb['indices'][start:end][nb['single_family'][start:end]>=0]]
            if chosen:
                rb=tensor_batch(ref_sampler.ds.collate(chosen));rz,_=model.font_features(rb['image'],rb['widths'])
                z=torch.cat((z,rz));f=torch.cat((f,torch.tensor([ref_sampler.ds.family_ids[ref_sampler.ds.rows[i]['family_group']] for i in chosen],device='cuda')))
                hashes += [ref_sampler.ds.rows[i]['pixel_sha256'] for i in chosen]
            groups['font_reference']=contrastive(z,f,hashes,ref_sampler.alias)
            loss=sum(groups.values())*((end-start)/n)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite joint loss')
        loss.backward()
        for k,v in groups.items():parts[k]=parts.get(k,0.)+float(v.detach())*(end-start)/n
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    flow={}
    if gradients:
        for prefix in ('capri.cnn.','capri.eff_','capri.blocks.','capri.font_','font_embedding.','font_outputs.','counter.','dino.patch_embed.'):
            flow[prefix]=any(p.grad is not None and bool(p.grad.abs().sum()>0) for name,p in model.named_parameters() if name.startswith(prefix))
        for i in range(12):flow[f'dino.block{i}']=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.dino.blocks[i].parameters())
    opt.step();return parts,float(norm),flow

@torch.no_grad()
def evaluate(model,ds,trainref,devref,out):
    model.eval();n=correct=counter=fontcat=fontn=0;effect=0.;records=[]
    for nb in D.batches_prefetch(ds,ds.batches(0,batch=16,budget=16*512,shuffle=False)):
        b=tensor_batch(nb)
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(b['image'],b['cols'],b['widths'])
        require_finite(o);p=o['length'].argmax(-1)+2;q=o['independent_count'].argmax(-1)+2
        correct+=int((p==b['length']).sum());counter+=int((q==b['length']).sum());n+=len(p)
        valid=b['font'][:,0]>=0;fontn+=int(valid.sum());fontcat+=int((o['font_cat'][valid].argmax(-1)==b['font'][valid,0]).sum())
        effect+=float(F.binary_cross_entropy_with_logits(o['eff_presence'].float(),b['eff_present'].float(),reduction='sum'))
        for j,i in enumerate(nb['indices']):records.append(dict(id=ds.rows[int(i)]['id'],count=int(p[j]),independent_count=int(q[j]),font_embedding=o['font_embedding'][j].cpu(),families=nb['families'][j].tolist()))
    # Font library uses pixels only; DEV references are for evaluation only.
    vectors=[];families=[];aliases=[]
    for ref in (trainref,devref):
        for nb in D.batches_prefetch(ref,ref.batches(0,batch=32,budget=32*512,shuffle=False)):
            b=tensor_batch(nb)
            with torch.autocast('cuda',dtype=torch.bfloat16):z,_=model.font_features(b['image'],b['widths'])
            vectors.append(z.float().cpu())
            for i in nb['indices']:
                r=ref.rows[int(i)];families.append(ref.family_ids[r['family_group']]);aliases.append([ref.family_ids[f] for f in r['pixel_alias_family_groups']])
    bank=torch.cat(vectors).cuda();f=torch.tensor(families,device='cuda');hit=single=unique=tied=0
    for start in range(0,len(records),64):
        block=records[start:start+64];z=torch.stack([r['font_embedding'] for r in block]).cuda();sim=z@bank.T
        for j,r in enumerate(block):
            best=sim[j].max();ids=torch.where(sim[j]>=best-1e-6)[0].tolist();candidates=sorted({a for i in ids for a in aliases[i]})
            truth=[a for a in r['families'] if a>=0];r['font_candidates']=candidates
            if len(truth)==1:
                single+=1;unique+=int(len(candidates)==1);hit+=int(len(candidates)==1 and candidates[0]==truth[0]);tied+=int(len(candidates)>1)
            del r['font_embedding']
    result=dict(images=n,count_correct=correct,count_accuracy=correct/n,separate_count_correct=counter,separate_count_accuracy=counter/n,font_category_correct=fontcat,font_category_support=fontn,font_category_accuracy=fontcat/max(1,fontn),font_retrieval_single_family_support=single,font_retrieval_unambiguous_correct=hit,font_retrieval_accuracy=hit/max(1,single),font_tied_predictions=tied,effect_bce=effect/(n*37),limitation='Reused synthetic DEV. Whole-image font retrieval; multi-font scenes excluded from exact single-family metric. Known pixel aliases retained. No real/calibration/test.')
    write(out,result)
    with Path(out).with_suffix('.predictions.jsonl').open('w',encoding='utf8') as stream:
        for r in records:stream.write(json.dumps(r)+'\n')
    model.train();return result

def datasets(root):
    train=D.Cache(root,'train');dev=D.Cache(root,'dev');refs=D.Cache(root,'references_train');devrefs=D.Cache(root,'references_dev')
    sampler=References(refs);sampler.scene_hashes=[r['pixel_sha256'] for r in train.rows]
    return train,dev,refs,devrefs,sampler

def readiness(dataset,out):
    out=Path(out).resolve();dataset=Path(dataset).resolve()
    if out.exists():raise FileExistsError(out)
    setup(out)
    try:
        report(out,state='verifying_cache')
        cache=read(dataset/'cache.json')
        for part in cache['parts']:
            for name,digest in part['sha256'].items():assert sha(dataset/part['path']/name)==digest
        train,dev,refs,devrefs,sampler=datasets(dataset);ew,tw=weights(train)
        plan=train.batches(1);assert len(train)==100000
        report(out,state='profiling_original_core_with_new_branches',logical_batches=len(plan))
        model=LightweightCapri().cuda();model.train();opt,sch=optimizer(model,len(plan)*EPOCHS)
        # Choose an actual wide batch; microbatches accumulate into the original128/pixel-budget batch.
        maximum=max(plan,key=lambda ids:(int(max(train.widths[ids])),len(ids)));nb=train.collate(maximum)
        trials=[];chosen=None
        for micro in (32,16,8):
            try:
                opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();t=time.time()
                parts,norm,flow=step_update(model,opt,nb,micro,sampler,1,ew,tw,True);torch.cuda.synchronize()
                assert all(flow.values()),flow
                peak=torch.cuda.max_memory_allocated()/2**30
                trials.append(dict(microbatch=micro,passed=True,peak_GiB=peak,seconds=time.time()-t));chosen=micro;break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();trials.append(dict(microbatch=micro,passed=False,error='CUDA out of memory'))
        if chosen is None:raise RuntimeError('No safe microbatch')
        # Fixed TRAIN batch: both count heads and joint objective must learn.
        for gr in opt.param_groups:gr['lr']=5e-6 if gr['name']=='dino' else BASE['lr']
        tiny=train.collate(plan[0]);history=[]
        for i in range(18):
            parts,_,_=step_update(model,opt,tiny,chosen,sampler,i+2,ew,tw);history.append(parts)
        before=history[0];after={k:sum(x[k] for x in history[-3:])/3 for k in before}
        assert sum(after.values())<sum(before.values()) and after['separate_count']<before['separate_count'],(before,after)
        report(out,state='profiling_throughput',microbatch=chosen,tiny_before=before,tiny_after=after)
        times=[];sizes=[]
        for i,nb in enumerate(D.batches_prefetch(train,plan[:16])):
            t=time.time();step_update(model,opt,nb,chosen,sampler,i+50,ew,tw);torch.cuda.synchronize()
            if i>=2:times.append(time.time()-t);sizes.append(len(nb['image']))
        ema=copy.deepcopy(model.state_dict());cp=save_epoch(out/'checkpoints',1,model,ema,opt,sch,None,dict(epoch=1,consumed_batches=0,step=0),'readiness',{})
        restored=load_resume(cp,model,opt,sch,None,'readiness');assert all(torch.equal(v.cpu(),restored['raw'][k]) for k,v in model.state_dict().items());del restored,ema
        # Bound evaluator smoke to complete small cache subsets via explicit diagnostic clones.
        def subset(ds,n):
            obj=copy.copy(ds);obj.rows=ds.rows[:n];obj.lookup=ds.lookup[:n];obj.widths=ds.widths[:n];return obj
        evaluate(model,subset(dev,32),subset(refs,16),subset(devrefs,16),out/'eval_smoke.json')
        result=dict(passed=True,time=now(),dataset=str(dataset),cache_sha256=sha(dataset/'cache.json'),sources=sources(),epochs=20,train_images=100000,dev_images=2000,logical_batch=128,pixel_budget=D.BUDGET,microbatch=chosen,steps_epoch1=len(plan),parameter_count=sum(p.numel() for p in model.parameters()),core_parameters=sum(p.numel() for p in model.capri.parameters()),architecture=model.resume_configuration(),gradient_flow=flow,memory_trials=trials,tiny_before=before,tiny_after=after,images_per_second=sum(sizes)/sum(times),checkpoint_restored=True,evaluation_passed=True)
        write(out/'readiness.json',result);report(out,state='ready',images_per_second=result['images_per_second'],microbatch=chosen,parameters=result['parameter_count'])
    except Exception as exc:report(out,state='failed',error=repr(exc));raise

def run(dataset,out,ready,resume=False):
    dataset=Path(dataset).resolve();out=Path(out).resolve();ready=Path(ready).resolve()
    if out.exists() and not resume:raise FileExistsError(out)
    setup(out)
    with StageLock(out,'training'):
      try:
        r=read(ready/'readiness.json');assert r['passed'] and r['sources']==sources();assert sha(dataset/'cache.json')==r['cache_sha256']
        train,dev,refs,devrefs,sampler=datasets(dataset);ew,tw=weights(train)
        plans=[train.batches(e) for e in range(1,21)];model=LightweightCapri().cuda();model.train();opt,sch=optimizer(model,sum(map(len,plans)));ema=copy.deepcopy(model.state_dict())
        if not resume:
            protocol=dict(created_at=now(),authorization='Original lightweight CAPRI + fine-tuned DINO font branch + separate count transformer;100k,20epochs jointly.',epochs=20,images_per_epoch=100000,seed=0,baseline_recipe=BASE,adam_betas=[.9,.95],precision='bfloat16,TF32 as original',dino_lr=5e-6,logical_batch=128,pixel_budget=D.BUDGET,microbatch=r['microbatch'],initialization='Original small CAPRI and separate counter random seed0; DINO official pretrained; no weights from deleted run',architecture=model.resume_configuration(),dataset=str(dataset),cache_sha256=r['cache_sha256'],sources=sources(),loss='Original seven equal-weight groups, missing labels masked; independent count CE and font reference contrastive each weight1',references='Up to8 TRAIN reference specimens per logical batch from represented single-family scenes; distinct pixel hashes; known alias-family negatives excluded',selection='Fixed final epoch20 EMA; synthetic DEV reporting only; no early quality stop or automatic deployment',differences='Added trainable DINO font residual heads/128D retrieval and separate count transformer; microbatch accumulation for memory. Same selected100k new scenes, with available-label masks. No large decoder/localization/mask branch.')
            write(out/'protocol.json',protocol)
            for rel in protocol['sources']:
                p=out/'provenance'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
        protocol=read(out/'protocol.json');assert protocol['sources']==sources();ph=hashlib_protocol(protocol)
        step=0;first=1;cursor=0
        if resume:
            pointer=read(out/'resume.json');state=load_resume(pointer['checkpoint'],model,opt,sch,None,ph);ema={k:v.cuda() for k,v in state['ema'].items()};sample=state['sampler_state'];step=sample['step'];first=sample['epoch'];cursor=sample['consumed_batches'];del state
        def checkpoint(epoch,index,complete=False):
            dest=out/('epochs' if complete else 'recoveries');number=epoch if complete else step;p=dest/f'epoch_{number:04d}'/'checkpoint.pt'
            if not p.exists():p=save_epoch(dest,number,model,ema,opt,sch,None,dict(epoch=epoch,consumed_batches=index,step=step),ph,dict(epoch_complete=complete))
            else:assert sha(p)==read(p.parent/'receipt.json')['sha256']
            write(out/'resume.json',dict(checkpoint=str(p),epoch=epoch,step=step,epoch_complete=complete));return p
        if not (out/'initial_dev.json').exists():
            report(out,state='evaluating_initial_models');evaluate(model,dev,refs,devrefs,out/'initial_dev.json')
        write(out/'initial_eta.json',dict(created_at=now(),profile_images_per_second=r['images_per_second'],estimated_hours=2000000/r['images_per_second']/3600*1.2))
        for epoch in range(first,21):
            plan=plans[epoch-1];begin=cursor if epoch==first else 0;images=sum(map(len,plan[:begin]));fresh=0;t=time.time();model.train()
            for index,nb in enumerate(D.batches_prefetch(train,plan[begin:]),start=begin):
                if (out/'STOP').exists():
                    if step:checkpoint(epoch,index)
                    report(out,state='stopped',epoch=epoch,step=step,images=images);return
                parts,norm,_=step_update(model,opt,nb,r['microbatch'],sampler,step+1,ew,tw);sch.step();step+=1;ema_update(model,ema,step);images+=len(nb['image']);fresh+=len(nb['image'])
                if step%10==1:
                    speed=fresh/max(.01,time.time()-t);report(out,state='training',epoch=epoch,epochs=20,images=images,epoch_images=100000,step=step,loss_parts=parts,loss=sum(parts.values()),gradient_norm=norm,images_per_second=speed,estimated_remaining_seconds=((20-epoch)*100000+100000-images)/speed*1.2)
                if step==10 or step%250==0:checkpoint(epoch,index+1)
            assert images==100000;p=checkpoint(epoch,len(plan),True);report(out,state='evaluating',epoch=epoch,checkpoint=str(p))
            if not (out/f'dev_epoch_{epoch:02d}.json').exists():
                raw={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.load_state_dict(ema);evaluate(model,dev,refs,devrefs,out/f'dev_epoch_{epoch:02d}.json');model.load_state_dict(raw);del raw
            write(out/f'epoch_{epoch:02d}_complete.json',dict(epoch=epoch,images=images,checkpoint=str(p)))
        write(out/'completion.json',dict(completed_at=now(),epochs=20,images_per_epoch=100000,checkpoint=str(p),sha256=sha(p),weights='EMA',activated=False));report(out,state='complete',epochs=20)
      except Exception as exc:report(out,state='failed',error=repr(exc));raise

def hashlib_protocol(protocol):
    import hashlib
    return hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
