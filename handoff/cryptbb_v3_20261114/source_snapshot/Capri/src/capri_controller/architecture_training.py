"""100k training: independent count, then frozen-core font/appearance heads."""
import collections,copy,json,math,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import architecture_data as D
from .architecture_training_data import Scenes,References,Counts
from .count_transformer import CountTransformer
from .decoder import training,pilot_losses
from .decoder.common import ROOT,read,write,sha,now,StageLock
from .decoder.data import ShapeBatchPlan,exact_shape_collate
from .decoder.reference_data import reference_collate
from .decoder.reference_sampling import ReferenceSampler
from .decoder.font_training import reference_vectors
from .decoder.font_extraction import visual_regions
from .decoder.timed_pilot_eval import association,gallery_rows
from .decoder.appearance import appearance_losses
from .decoder.joint_phase import device_copy
from .decoder.optimization import finite_update
from .decoder.checkpoints import save_epoch,load_resume,require_finite
from .decoder.metrics import count_statistics,localization_statistics

SEED=918
POLICY=dict(instance_threshold=.5,visibility_threshold=.25,library_iou_threshold=.5)

def setup(out):
    if out.resolve().drive.upper()!='E:':raise ValueError('All outputs must stay on E:')
    for name in ('temporary','torch_cache','provenance'):(out/name).mkdir(parents=True,exist_ok=True)
    for k,v in dict(TEMP=out/'temporary',TMP=out/'temporary',TORCH_HOME=out/'torch_cache',HF_HOME=out/'torch_cache',XDG_CACHE_HOME=out/'torch_cache').items():os.environ[k]=str(v)
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['OMP_NUM_THREADS']='4'
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this run')

def report(out,**kw):
    v=dict(updated_at=now(),pid=os.getpid(),**kw);write(out/'status.json',v)
    with (out/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(v)+'\n')
    print(json.dumps(v),flush=True)

def main_model(checkpoint):
    model=training.construct();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['ema'],strict=True);del saved
    model.requires_grad_(False)
    for name,p in model.named_parameters():
        if name.startswith(training.SECONDARY):p.requires_grad_(True)
    model.set_dino_trainable(0);model.train();return model

def alias_edges(refs,family_ids):
    edges=set()
    for row in refs.rows:
        ids=[family_ids[f] for f in row['pixel_alias_family_groups'] if f in family_ids]
        edges.update((a,b) for a in ids for b in ids if a!=b)
    return sorted(edges)

def font_loss(vectors,families,keys,chars,edges):
    # Different glyph content is required for positives. Never treat identical
    # specimens as easy positives or push known specimen-alias families apart.
    n=len(vectors);device=vectors.device
    same=families[:,None]==families[None,:]
    duplicate=torch.tensor([[i!=j and a is not None and a==b for j,b in enumerate(keys)] for i,a in enumerate(keys)],device=device,dtype=torch.bool)
    same_specimen=torch.tensor([[a is not None and b is not None and a[0]==b[0] for b in keys] for a in keys],device=device,dtype=torch.bool)
    ambiguous=same_specimen&~same
    for a,b in edges:ambiguous|=(families[:,None]==a)&(families[None,:]==b)
    content=torch.tensor([[a!=b for b in chars] for a in chars],device=device,dtype=torch.bool)
    allowed=~(torch.eye(n,device=device,dtype=torch.bool)|duplicate|ambiguous)
    positive=same&allowed&content;negative=~same&allowed
    eligible=positive.any(-1)&negative.any(-1)
    audit=dict(vectors=n,eligible=int(eligible.sum()),duplicate_pairs=int(duplicate.sum())//2,ambiguous_pairs=int(ambiguous.sum())//2)
    if not eligible.any():return vectors.sum()*0,audit
    z=F.normalize(vectors.float(),dim=-1);logits=z[eligible]@z.T/.1
    lp=logits-torch.logsumexp(logits.masked_fill(~allowed[eligible],-torch.inf),dim=-1,keepdim=True)
    return -(lp.masked_fill(~positive[eligible],0).sum(-1)/positive[eligible].sum(-1)).mean(),audit

def main_loss(model,batch,refs,family_ids,edges):
    device=next(model.parameters()).device
    data=device_copy(batch,device);pred=model(**data['inputs'])
    with torch.no_grad():assign=pilot_losses.match(pred,data['targets'])
    vectors=[torch.cat([pred['font_embeddings'][i,pi] for i,(pi,_) in enumerate(assign)])]
    families=[torch.cat([t['font_family'][ti] for t,(_,ti) in zip(data['targets'],assign)])]
    chars=[batch['audit'][i]['characters'][int(t)] for i,(_,ti) in enumerate(assign) for t in ti]
    keys=[None]*len(chars)
    for ref in refs:
        ref=device_copy(ref,device)
        if any(v['split']!='train' for v in ref['library_labels']):raise ValueError('DEV references cannot optimize')
        rp=model(**ref['inputs']);v,f,k,_=reference_vectors(rp,ref,family_ids,POLICY)
        charmap={a['pixel_sha256']:v['characters'] for a,v in zip(ref['audit'],ref['library_labels'])}
        vectors.append(v);families.append(f);keys.extend(k);chars.extend(charmap[h][i] for h,i in k)
    font,audit=font_loss(torch.cat(vectors),torch.cat(families),keys,chars,edges)
    parts,_=appearance_losses(pred,data['targets'],assign)
    loss=font+sum(v for k,v in parts.items() if k.startswith('font_'))/6
    loss=loss+2*parts['background_under']+parts['colour_rgb']
    presence=torch.stack([t['effect_presence'] for t in data['targets']]).float()
    intensity=torch.stack([t['effect_intensity'] for t in data['targets']]).float()
    loss=loss+F.binary_cross_entropy_with_logits(pred['effect_presence_logits'].float(),presence)
    loss=loss+.5*(F.smooth_l1_loss(pred['intensity_predictions'].float(),intensity,reduction='none')*presence).sum()/presence.sum().clamp_min(1)
    loss=loss+.25*F.binary_cross_entropy_with_logits(pred['composition_logits'].float(),torch.stack([t['composition'] for t in data['targets']]).float())
    loss=loss+.25*F.cross_entropy(pred['row_logits'].float(),torch.stack([t['rows'] for t in data['targets']])-1)
    return loss,audit

def count_plan(length,batch,epoch,shuffle=True):
    ids=torch.randperm(length,generator=torch.Generator().manual_seed(SEED+epoch)).tolist() if shuffle else list(range(length))
    return [ids[i:i+batch] for i in range(0,length,batch)]

def loader(ds,plan,stage,workers=4):
    kw=dict(dataset=ds,batch_sampler=plan,num_workers=workers,pin_memory=True,generator=torch.Generator().manual_seed(SEED),collate_fn=exact_shape_collate if stage=='main' else None)
    if workers:kw['prefetch_factor']=2
    return DataLoader(**kw)

def optimize(model,steps,stage):
    lr=3e-4 if stage=='count' else 1e-4
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01,foreach=True)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*min(s,steps)/max(1,steps)))))
    return opt,sch,torch.amp.GradScaler('cuda',init_scale=16.)

@torch.no_grad()
def count_eval(model,ds,out):
    model.eval();logits=[];truth=[];indices=[]
    for x,v,y,i in loader(ds,count_plan(len(ds),64,0,False),'count'):
        with torch.autocast('cuda',dtype=torch.float16):p=model(x.cuda(),v.cuda())
        require_finite(p);logits.append(p.float().cpu());truth.append(y);indices.append(i)
    p=torch.cat(logits);y=torch.cat(truth);i=torch.cat(indices)
    assert len(y)==len(ds) and len(i.unique())==len(ds)
    stats=dict(images=len(y),correct=int((p.argmax(-1)==y).sum()),accuracy=float((p.argmax(-1)==y).float().mean()),nll=float(F.cross_entropy(p,y)),per_count={str(k+2):dict(total=int((y==k).sum()),correct=int(((p.argmax(-1)==y)&(y==k)).sum())) for k in range(7)})
    torch.save(dict(logits=p,targets=y+2,indices=i),out.with_suffix('.pt'));write(out,stats);model.train();return stats

@torch.no_grad()
def main_eval(model,dev,trainrefs,devrefs,destination):
    model.eval();rows=gallery_rows(trainrefs,devrefs);buckets=collections.defaultdict(list)
    for ds,i in rows:buckets[tuple(ds.rows[i]['shape'])].append((ds,i))
    vectors=[];labels=[];alias_sets=[]
    for items in buckets.values():
        for start in range(0,len(items),8):
            batch=reference_collate([ds[i] for ds,i in items[start:start+8]])
            with torch.autocast('cuda',dtype=torch.float16):pred=model(**device_copy(batch['inputs'],'cuda'))
            for region,label in zip(visual_regions(pred,.5,.25),batch['library_labels']):
                for p,t in association(region['boxes'],label['boxes_visible']):
                    vectors.append(region['embeddings'][p].float());labels.append(label['family_group']);alias_sets.append(label['alias_families'])
    families=sorted({ds.rows[i]['family_group'] for ds,i in rows});fid={f:i for i,f in enumerate(families)}
    bank=torch.stack(vectors) if vectors else torch.empty(0,128,device='cuda')
    ids=torch.tensor([fid[f] for f in labels],device='cuda');inverse={v:k for k,v in dev.family_indices.items()}
    total=matched=unique=correct=tied=tiehit=0;bg=effect=0.;counts=[];logits=[];presence=[];boxes=[];truth=[];predictions=[]
    plan=ShapeBatchPlan(dev,8,epoch=0,shuffle=False)
    for batch in loader(dev,plan,'main'):
        with torch.autocast('cuda',dtype=torch.float16):pred=model(**device_copy(batch['inputs'],'cuda'))
        require_finite({k:pred[k] for k in ('font_embeddings','background_under','count_logits')})
        regions=visual_regions(pred,.5,.25)
        for j,(region,target) in enumerate(zip(regions,batch['targets'])):
            sets=[]
            if len(bank) and len(region['embeddings']):
                similarity=region['embeddings'].float()@bank.T
                scores=torch.full((len(similarity),len(families)),-torch.inf,device='cuda')
                scores.scatter_reduce_(1,ids[None].expand(len(similarity),-1),similarity,reduce='amax',include_self=True)
                # Explicitly retain near-equal candidates and known pixel alias
                # families for the maximizing reference, never alphabetical wins.
                for q in range(len(scores)):
                    best=scores[q].max();candidates={families[k] for k in torch.where(scores[q]>=best-1e-6)[0].tolist()}
                    for k in torch.where(similarity[q]>=best-1e-6)[0].tolist():candidates.update(alias_sets[k])
                    sets.append(sorted(candidates))
            else:sets=[[] for _ in region['query_ids']]
            total+=len(target['font_family'])
            for p,t in association(region['boxes'],target['boxes_visible']):
                matched+=1;f=inverse[int(target['font_family'][t])];s=sets[p]
                if len(s)==1:unique+=1;correct+=int(f in s)
                if len(s)>1:tied+=1;tiehit+=int(f in s)
            predictions.append(dict(id=batch['audit'][j]['stem'],query_ids=region['query_ids'].tolist(),font_candidates=sets,decoder_count=int(pred['count_logits'][j].argmax())+2,detected_region_count=len(region['query_ids'])))
            v=target['valid_mask'].cuda();y=target['background_under'].cuda()/v.clamp_min(1e-8)[None]
            bg+=float(((pred['background_under'][j].float()-y).abs()*v).sum()/(3*v.sum()))
            effect+=float(F.binary_cross_entropy_with_logits(pred['effect_presence_logits'][j].float(),target['effect_presence'].cuda()))
        counts.extend(int(t['count']) for t in batch['targets']);truth.extend(t['boxes_visible'] for t in batch['targets'])
        logits.append(pred['count_logits'].float().cpu());presence.append(pred['presence_logits'].float().cpu());boxes.append(pred['boxes'].float().cpu())
    assert len(counts)==len(dev)
    metrics=dict(images=len(dev),background_mae=bg/len(dev),effect_bce=effect/len(dev),count_comparator=count_statistics(torch.cat(logits),counts,torch.cat(presence),.5),localization=localization_statistics(torch.cat(boxes),torch.cat(presence),truth,.5),font=dict(total_glyphs=total,matched_glyphs=matched,unique_candidate_predictions=unique,unambiguous_correct=correct,end_to_end_unambiguous_top1=correct/max(1,total),tied_predictions=tied,tie_contains_truth=tiehit,candidate_families=len(families),represented_families=len(set(labels)),gallery_specimens=len(rows),prototype_count=len(labels)),limitations='Reused synthetic DEV; exact-specimen alias candidates retained, not proof of family equivalence. No real/test labels or calibration.')
    D.jsonl(destination.with_suffix('.predictions.jsonl'),predictions);write(destination,metrics);model.train();return metrics

def readiness(dataset,out,checkpoint):
    out=Path(out).resolve();dataset=Path(dataset).resolve()
    if out.exists():raise FileExistsError('Preserve existing readiness run')
    setup(out);report(out,state='readiness',stage='verify_data')
    verification=D.verify(dataset);train=Scenes(dataset);dev=Scenes(dataset,'dev');counts=Counts(dataset);refs=References(dataset)
    assert len(train)==len(counts)==100000 and len(dev)==2000
    sampler=ReferenceSampler(refs,train.family_indices,SEED);edges=alias_edges(refs,train.family_indices)
    cp=Path(checkpoint).resolve();receipt=read(cp.parent/'receipt.json');assert sha(cp)==receipt['sha256']
    checks={};speeds={}
    # Duplicate/alias exclusions remain differentiable and do not create NaNs.
    z=torch.randn(5,128,device='cuda',requires_grad=True);ids=torch.tensor([0,0,1,1,2],device='cuda')
    l,a=font_loss(z,ids,[('same',0),('same',0),('same',1),('other',0),None],list('ababc'),[(0,1),(1,0)])
    l.backward();require_finite(z.grad);checks['alias_loss_finite_and_excludes_pairs']=a['duplicate_pairs']>0 and a['ambiguous_pairs']>0
    for stage in ('count','main'):
        report(out,state='readiness',stage=stage)
        model=CountTransformer().cuda() if stage=='count' else main_model(cp)
        ds=counts if stage=='count' else train
        plan=count_plan(len(ds),32,1) if stage=='count' else ShapeBatchPlan(ds,8,epoch=1)
        opt,sch,scaler=optimize(model,100,stage);timings=[];sizes=[];losses=[];last=None
        torch.cuda.reset_peak_memory_stats()
        for step,batch in enumerate(loader(ds,plan,stage)):
            started=time.perf_counter()
            if stage=='count':
                x,v,y,i=batch;x=x.cuda();v=v.cuda();y=y.cuda()
                def forward():
                    with torch.autocast('cuda',dtype=torch.float16):return F.cross_entropy(model(x,v).float(),y)
                n=len(y)
            else:
                selection=sampler.select(batch,1,2,'family');rb=sampler.materialize(selection) if step%4==0 else []
                def forward():
                    with torch.autocast('cuda',dtype=torch.float16):return main_loss(model,batch,rb,train.family_indices,edges)[0]
                n=len(batch['targets'])
            value,norm,retries=finite_update(model,opt,scaler,forward);torch.cuda.synchronize()
            timings.append(time.perf_counter()-started);sizes.append(n);losses.append(float(value));last=batch
            if step==31:break
        # Fixed tiny batch optimization gate, independent of DEV and reset before production.
        before=float(forward().detach());tiny=[]
        for step in range(32):
            value,_,_=finite_update(model,opt,scaler,forward);tiny.append(float(value))
        after=sum(tiny[-8:])/8
        if not after<before:raise ValueError(stage+' tiny training did not reduce loss')
        checks[stage+'_tiny_loss_decreases']=dict(before=before,after=after)
        if stage=='main':
            frozen=[name for name,p in model.named_parameters() if not p.requires_grad]
            checks['frozen_core_has_no_gradients']=all(p.grad is None for name,p in model.named_parameters() if name in frozen)
        else:
            model.eval()
            with torch.no_grad():
                a=model(x,v);changed=x.clone();changed.masked_fill_(~v[:,None].expand_as(x),.123);b=model(changed,v)
            torch.testing.assert_close(a,b);checks['count_padding_canonicalized']=True
        speeds[stage]=dict(images_per_second=sum(sizes[4:])/sum(timings[4:]),mean_update_seconds=sum(timings[4:])/len(timings[4:]),peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        write(out/'partial_readiness.json',dict(checks=checks,speeds=speeds))
        del model,opt,sch,scaler;torch.cuda.empty_cache()
    result=dict(passed=True,created_at=now(),dataset=str(dataset),integrity_sha256=sha(dataset/'integrity.json'),checkpoint=str(cp),checkpoint_sha256=sha(cp),verification=verification,checks=checks,speeds=speeds,protocol=dict(count_epochs=8,main_epochs=4,count_batch=32,main_batch=8,all_training_images_per_epoch=100000),limitation='32-update speed sample excludes loader startup, validation and checkpoint I/O; production ETA adds overhead and reports measured progress.')
    write(out/'readiness.json',result);report(out,state='readiness_complete',result=result)
    train.close();dev.close();counts.close();refs.close();return result

def update_ema(model,ema,step):
    with torch.no_grad():
        decay=min(.999,1-1/(step+1));trainable={n for n,p in model.named_parameters() if p.requires_grad}
        for n,p in model.state_dict().items():
            if n in trainable:ema[n].lerp_(p,1-decay)

def run(dataset,out,ready,resume=False):
    out=Path(out).resolve();dataset=Path(dataset).resolve();ready=Path(ready).resolve()
    if out.exists() and not resume:raise FileExistsError('Existing training run preserved; use --resume')
    setup(out)
    with StageLock(out,'training'):
      try:
        r=read(ready/'readiness.json')
        if not r['passed'] or sha(dataset/'integrity.json')!=r['integrity_sha256'] or sha(r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('Readiness identity changed')
        source_paths=[Path(__file__),Path(__file__).with_name('count_transformer.py'),Path(__file__).with_name('architecture_training_data.py'),Path(__file__).with_name('architecture_data.py'),Path(__file__).with_name('cli.py'),ROOT/'configs/local.json']+list((ROOT/'src/capri_controller/decoder').glob('*.py'))
        vendor=Path(read(ROOT/'configs/local.json')['decoder_dinov2_source'])
        source_paths+=list(vendor.rglob('*.py'))
        sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
        if not resume:
            protocol=dict(created_at=now(),authorization='User requested start training on100k and a completion ETA.',dataset=str(dataset),integrity_sha256=r['integrity_sha256'],initialization=dict(main=r['checkpoint'],sha256=r['checkpoint_sha256'],weights='EMA epoch8; new head optimizer',count='random seed918'),seed=SEED,count_epochs=8,main_epochs=4,count_batch=32,main_batch=8,workers=4,sources=sources,scope='Frozen CNN,DINO,fusion,localization; train font and appearance heads. Independent count transformer. No specialists/distillation or active-model replacement.',font_policy='Different-character positives; duplicate-reference identities excluded; known TRAIN specimen-alias family pairs excluded from negatives without merging; DEV gallery never optimization; ties retained during evaluation.',selection='Count lowest DEV NLL; main fixed final fourth epoch. No real/test labels; no deployment/calibration implied.',precision='FP16 with finite-gradient retries, TF32 disabled',full_epoch=100000,readiness=r)
            write(out/'protocol.json',protocol)
            for p in source_paths:
                dst=out/'provenance'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        protocol=read(out/'protocol.json');ph=D.jhash(protocol)
        if protocol['sources']!=sources:raise ValueError('Training code changed; preserve run and create reviewed continuation')
        # Full data hashes are checked once in readiness; every decoded image is also hash checked.
        scenes=Scenes(dataset);dev=Scenes(dataset,'dev');refs=References(dataset);devrefs=References(dataset,'dev');counts=Counts(dataset);countdev=Counts(dataset,'dev')
        assert len(scenes)==len(counts)==100000 and len(dev)==len(countdev)==2000
        sampler=ReferenceSampler(refs,scenes.family_indices,SEED);edges=alias_edges(refs,scenes.family_indices)
        start=time.time();estimated={s:100000/r['speeds'][s]['images_per_second'] for s in ('count','main')}
        write(out/'eta.json',dict(created_at=now(),estimated_training_seconds=8*estimated['count']+4*estimated['main'],expected_total_seconds=(8*estimated['count']+4*estimated['main'])*1.25,range_seconds=[(8*estimated['count']+4*estimated['main'])*1.1,(8*estimated['count']+4*estimated['main'])*1.6],basis='Readiness GPU sample, plus10-60% loader/evaluation/checkpoint overhead. Refine from production.',started_unix=start))
        for stage in ('count','main'):
            if (out/(stage+'_complete.json')).exists():continue
            report(out,state='initializing',stage=stage)
            epochs=protocol[stage+'_epochs'];bs=protocol[stage+'_batch'];ds=counts if stage=='count' else scenes
            model=CountTransformer().cuda() if stage=='count' else main_model(Path(r['checkpoint']))
            plans=[count_plan(len(ds),bs,e) if stage=='count' else ShapeBatchPlan(ds,bs,epoch=e,seed=SEED).batches for e in range(1,epochs+1)]
            assert all(len([i for b in plan for i in b])==len(set(i for b in plan for i in b))==100000 for plan in plans)
            opt,sch,scaler=optimize(model,sum(map(len,plans)),stage);ema=copy.deepcopy(model.state_dict());step=0;first_epoch=1;cursor=0
            if stage=='main' and not (out/'initial_main_dev.json').exists():
                report(out,state='evaluating_initial_main',stage=stage)
                main_eval(model,dev,refs,devrefs,out/'initial_main_dev.json')
            pointer=out/(stage+'_resume.json')
            if resume and pointer.exists():
                point=read(pointer);saved=load_resume(point['checkpoint'],model,opt,sch,scaler,ph)
                # load_resume returns saved state and restores optimizer/RNG/execution.
                ema={k:v.cuda() for k,v in saved['ema'].items()};state=saved['sampler_state'];step=state['global_step'];first_epoch=state['epoch'];cursor=state['consumed_batches']
            def checkpoint(epoch,consumed,complete):
                state=dict(stage=stage,epoch=epoch,consumed_batches=consumed,global_step=step,snapshot=sha(dataset/'train.jsonl'))
                dest=out/stage/('epochs' if complete else 'recoveries')
                existing=dest/f'epoch_{epoch if complete else step:04d}'/'checkpoint.pt'
                if existing.exists():
                    receipt=read(existing.parent/'receipt.json')
                    if sha(existing)!=receipt['sha256']:raise ValueError('Existing resume checkpoint changed')
                    p=existing
                else:p=save_epoch(dest,epoch if complete else step,model,ema,opt,sch,scaler,state,ph,dict(epoch_complete=complete))
                write(pointer,dict(checkpoint=str(p),epoch=epoch,global_step=step));return p
            for epoch in range(first_epoch,epochs+1):
                plan=plans[epoch-1];begin=cursor if epoch==first_epoch else 0;images=sum(len(b) for b in plan[:begin]);epoch_start=time.time();new_images=0
                model.train()
                for index,batch in enumerate(loader(ds,plan[begin:],stage),start=begin):
                    if (out/'STOP').exists():
                        if step:checkpoint(epoch,index,False)
                        report(out,state='stopped',stage=stage,epoch=epoch,images=images);return
                    if stage=='count':
                        x,v,y,_=batch;x=x.cuda(non_blocking=True);v=v.cuda(non_blocking=True);y=y.cuda(non_blocking=True);n=len(y)
                        def forward():
                            with torch.autocast('cuda',dtype=torch.float16):return F.cross_entropy(model(x,v).float(),y)
                    else:
                        rb=sampler.materialize(sampler.select(batch,epoch,2,'family')) if step%4==0 else [];n=len(batch['targets'])
                        def forward():
                            with torch.autocast('cuda',dtype=torch.float16):return main_loss(model,batch,rb,scenes.family_indices,edges)[0]
                    value,norm,retries=finite_update(model,opt,scaler,forward);sch.step();step+=1;update_ema(model,ema,step);images+=n;new_images+=n
                    if step%25==1:
                        speed=new_images/max(.01,time.time()-epoch_start)
                        remaining=((epochs-epoch)*100000+100000-images)/speed
                        if stage=='count':remaining+=4*estimated['main']
                        report(out,state='training',stage=stage,epoch=epoch,epochs=epochs,step=step,images=images,epoch_images=100000,loss=float(value),gradient_norm=float(norm),images_per_second=speed,estimated_remaining_seconds=remaining*1.25,amp_retries=retries)
                    if step%1000==0:checkpoint(epoch,index+1,False)
                assert images==100000
                cp=checkpoint(epoch,len(plan),True);report(out,state='evaluating',stage=stage,epoch=epoch,checkpoint=str(cp))
                raw={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.load_state_dict(ema)
                metric_path=out/stage/f'dev_epoch_{epoch:02d}.json'
                metrics=count_eval(model,countdev,metric_path) if stage=='count' else main_eval(model,dev,refs,devrefs,metric_path)
                model.load_state_dict(raw);del raw
                if stage=='main':
                    count_choice=read(out/'count_complete.json')['selected_epoch']
                    count_saved=torch.load(out/'count'/f'dev_epoch_{count_choice:02d}.pt',map_location='cpu',weights_only=False)
                    estimates={countdev.rows[int(i)]['id']:int(p.argmax())+2 for i,p in zip(count_saved['indices'],count_saved['logits'])}
                    predictions=D.read_rows(metric_path.with_suffix('.predictions.jsonl'))
                    for row in predictions:
                        row['independent_count']=estimates[row['id']]
                        row['count_region_disagreement']=row['independent_count']!=row['detected_region_count']
                    D.jsonl(metric_path.with_suffix('.consistency.jsonl'),predictions)
                    metrics['independent_count_region_disagreements']=sum(v['count_region_disagreement'] for v in predictions)
                    write(metric_path,metrics)
                write(out/stage/f'epoch_{epoch:02d}_complete.json',dict(checkpoint=str(cp),metrics=str(metric_path),images=images,elapsed_seconds=time.time()-epoch_start))
            if stage=='count':
                scores=[(read(out/stage/f'dev_epoch_{e:02d}.json')['nll'],e) for e in range(1,epochs+1)];chosen=min(scores)[1]
            else:chosen=epochs
            selected=out/stage/'epochs'/f'epoch_{chosen:04d}'/'checkpoint.pt'
            write(out/(stage+'_complete.json'),dict(completed_at=now(),epochs=epochs,images_per_epoch=100000,selected_epoch=chosen,checkpoint=str(selected),sha256=sha(selected),weights='EMA',activated=False))
            del model,opt,sch,scaler,ema;torch.cuda.empty_cache()
        report(out,state='complete',elapsed_hours=(time.time()-start)/3600,training_complete=True,activated=False,limitations='Synthetic development evaluation only; count routing/calibration and renderer integration remain separate.')
        for ds in (scenes,dev,refs,devrefs,counts,countdev):ds.close()
      except Exception as exc:
        report(out,state='failed',error=repr(exc));raise
