"""Authorized CryptBB Model1 pair; upstream training with recoverable epoch saves."""
import argparse, copy, hashlib, importlib.util, os, random, shutil, subprocess, sys, time
from pathlib import Path
from ..decoder.common import read, write, sha, now, StageLock
ROOT=Path(__file__).resolve().parents[4]
STUDY=ROOT/'handoff/cryptbb_model1_training_20261113'
PREP=ROOT/'handoff/cryptbb_model1_preparation'

def readiness(tag):
    spec=importlib.util.spec_from_file_location('cryptbb_ready',ROOT/'model1/src/train_cryptbb.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.readiness(tag)

def environment():
    for key in ['TEMP','TMP','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        assert Path(os.environ[key]).drive.upper()=='E:',key

def rng_state(model):
    import numpy as np, torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),modules={n:copy.deepcopy(m.rng.bit_generator.state) for n,m in model.named_modules() if isinstance(getattr(m,'rng',None),np.random.Generator)})

def restore_rng(model,state):
    import numpy as np, torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    for n,m in model.named_modules():
        if n in state['modules']:m.rng.bit_generator.state=copy.deepcopy(state['modules'][n])

def loader_state(trainer):
    loader=trainer.train_dataloader
    return dict(order=loader.dataset.data_idx_order_list.copy(),sampler=copy.deepcopy({k:v for k,v in vars(loader.batch_sampler).items() if k!='data_source'}))

def restore_loader(trainer,state):
    trainer.train_dataloader.dataset.data_idx_order_list=state['order'].copy()
    for k,v in state['sampler'].items():setattr(trainer.train_dataloader.batch_sampler,k,copy.deepcopy(v))

def make_trainer(tag,out):
    import numpy as np, torch
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer
    row,_=readiness(tag);cfg=Config(str(ROOT/row['path']))
    cfg.cfg['Global']['output_dir']=str(out)
    assert cfg.cfg['Global']['pretrained_model'] is None and cfg.cfg['Global']['checkpoints'] is None
    assert cfg.cfg['Global']['epoch_num']==20 and cfg.cfg['Eval'] is None
    torch.set_num_threads(8);assert torch.cuda.is_available()
    trainer=Trainer(cfg,mode='train',task='rec')
    assert trainer.valid_dataloader is None and len(trainer.train_dataloader)==782 and len(trainer.train_dataloader.dataset)==100000
    for i,m in enumerate(trainer.model.modules()):
        if isinstance(getattr(m,'rng',None),np.random.Generator):m.rng=np.random.default_rng(48+i)
    return trainer

def capture(trainer,epoch,step,metrics):
    return dict(epoch=epoch,global_step=step,state_dict=trainer.model.state_dict(),optimizer=trainer.optimizer.state_dict(),scheduler=trainer.lr_scheduler.state_dict(),scaler=trainer.scaler.state_dict(),rng=rng_state(trainer.model),loader=loader_state(trainer),config=trainer.cfg,metrics=metrics,protocol_sha256=sha(STUDY/'protocol.json'))

def restore(trainer,state):
    assert state['protocol_sha256']==sha(STUDY/'protocol.json')
    trainer.model.load_state_dict(state['state_dict'],strict=True);trainer.optimizer.load_state_dict(state['optimizer']);trainer.lr_scheduler.load_state_dict(state['scheduler']);trainer.scaler.load_state_dict(state['scaler'])
    restore_loader(trainer,state['loader']);restore_rng(trainer.model,state['rng'])
    trainer.status=dict(epoch=state['epoch']+1,global_step=state['global_step'],metrics=state['metrics'])

def save_atomic(path,state):
    import torch
    path=Path(path);tmp=path.with_suffix('.partial');torch.save(state,tmp);os.replace(tmp,path)

def verify_sources():
    protocol=read(STUDY/'protocol.json')
    for row in protocol['sources']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    assert sha(PREP/'agreement_protocol.json')==protocol['selection_protocol_sha256']
    return protocol

def prepare():
    environment();STUDY.mkdir(parents=True,exist_ok=True)
    if (STUDY/'protocol.json').exists():verify_sources();return
    rows=[readiness(tag)[0] for tag in 'AB']
    files=[Path(__file__),ROOT/'Capri/capri.py',ROOT/'model1/src/train_cryptbb.py']
    for directory in ['model1/src/OpenOCR/tools','model1/src/OpenOCR/openrec']:
        files+=list((ROOT/directory).rglob('*.py'))
    sources=[]
    for p in files:
        rel=p.relative_to(ROOT);dest=STUDY/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources.append(dict(path=rel.as_posix(),sha256=sha(p)))
    protocol=dict(created_at=now(),authorization='User explicitly requested starting CryptBB Model One now on 13 November 2026',scope='Model1-A then Model1-B only; stop after both readers; no Model2/3 launch',datasets=rows,epochs=20,steps_per_epoch=782,draws_per_epoch=100096,steps_per_run=15640,batch_size=128,seed=48,initialization='scratch independently for A and B',preprocessing='RGB PIL bicubic32x128 normalization0.5; min_ratio=max_ratio=4 repair; no added augmentation',selection_protocol_sha256=sha(PREP/'agreement_protocol.json'),selection='max_same_epoch_ab_exact_agreement_earliest_tie',candidate_epochs=list(range(2,21)),real_labels_used_for_training=False,synthetic_validation_required_for_model1=False,execution='sequential on one GPU',training_loop='Unmodified upstream Trainer.train; process-local atomic epoch-save hook adds scaler, RNG and sampler recovery',sources=sources,limits=['Shared unlabeled real selection/scoring pool is transductive','Single run per dataset','No automatic evaluation or later model launch; report selection pending after training'])
    write(STUDY/'protocol.json',protocol)

def update_step(trainer,batch):
    import torch
    trainer.model.train();batch=[x.to(trainer.device) for x in batch]
    with torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16):
        preds=trainer.model(batch[0],data=batch[1:]);loss=trainer.loss_class(preds,batch)['loss']
    assert torch.isfinite(loss)
    trainer.scaler.scale(loss).backward();trainer.scaler.unscale_(trainer.optimizer)
    assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),trainer.grad_clip_val)
    trainer.scaler.step(trainer.optimizer);trainer.scaler.update();trainer.optimizer.zero_grad(set_to_none=True);trainer.lr_scheduler.step()
    return float(loss.detach())

def smoke():
    import numpy as np, torch
    prepare();out=STUDY/'readiness';out.mkdir(exist_ok=True)
    t=make_trainer('A',out);torch.cuda.reset_peak_memory_stats();start=time.time()
    batches=iter(t.train_dataloader);batch=next(batches);assert list(batch[0].shape)==[128,3,32,128]
    before={k:v.detach().cpu().clone() for k,v in t.model.state_dict().items()}
    loss1=update_step(t,batch)
    assert any(not torch.equal(before[k],v.detach().cpu()) for k,v in t.model.state_dict().items());del before
    state=capture(t,1,782,{})
    save_atomic(out/'recovery_probe.pth',state)
    second=next(batches);loss2=update_step(t,second)
    expected={k:v.detach().cpu().clone() for k,v in t.model.state_dict().items()}
    expected_scheduler=copy.deepcopy(t.lr_scheduler.state_dict());expected_scaler=copy.deepcopy(t.scaler.state_dict())
    saved=torch.load(out/'recovery_probe.pth',map_location='cpu',weights_only=False);restore(t,saved)
    replay_loss=update_step(t,second)
    max_diff=max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in t.model.state_dict().items())
    assert max_diff==0 and loss2==replay_loss and t.lr_scheduler.state_dict()==expected_scheduler and t.scaler.state_dict()==expected_scaler
    # Epoch-boundary order recovery must preserve full coverage and padded budget.
    ls=loader_state(t);rs=rng_state(t.model)
    draws=list(iter(t.train_dataloader.batch_sampler));restore_loader(t,ls);restore_rng(t.model,rs)
    replay=list(iter(t.train_dataloader.batch_sampler));assert draws==replay
    ids=[x[2] for b in draws for x in b];assert len(ids)==100096 and len(set(ids))==100000 and len(draws)==782
    write(out/'receipt.json',dict(passed=True,completed_at=now(),protocol_sha256=sha(STUDY/'protocol.json'),losses=[loss1,loss2,replay_loss],recovery_model_max_abs_difference=max_diff,optimizer_scheduler_scaler_rng_replayed=True,sampler_draws_replayed=True,unique_images=100000,draws=100096,steps=782,batch_shape=list(batch[0].shape),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,seconds=time.time()-start,production_weights_used=False))
    print('Readiness passed',flush=True)

def worker(tag,resume=False):
    import torch
    from tools.engine import trainer as engine
    environment();verify_sources();check=read(STUDY/'readiness/receipt.json');assert check['passed'] and check['protocol_sha256']==sha(STUDY/'protocol.json')
    out=ROOT/f'model1/runs/cryptbb/{tag}';latest=out/'latest.pth'
    with StageLock(PREP,'reader_training'),StageLock(out,'training'):
        assert resume or not latest.exists(),'Existing weights require explicit resume'
        t=make_trainer(tag,out)
        if resume:
            receipt=read(out/'latest_receipt.json');assert sha(latest)==receipt['sha256']
            restore(t,torch.load(latest,map_location='cpu',weights_only=False))
        write(out/'launch.json',dict(started_at=now(),pid=os.getpid(),dataset=tag,resume=resume,protocol_sha256=sha(STUDY/'protocol.json'),epochs=20,budget_steps=15640,selection='A/B same-epoch agreement after both runs'))
        def save_hook(model,cfg,optimizer,lr_scheduler,epoch,global_step,metrics,is_best=False,logger=None,prefix=None):
            assert not is_best and global_step==epoch*782
            if prefix is not None:return  # The first call atomically creates both archive and latest.
            state=capture(t,epoch,global_step,metrics);archive=out/f'epoch_{epoch}.pth'
            assert not archive.exists(),'Immutable epoch already exists'
            save_atomic(archive,state)
            temporary=out/'latest.copying';shutil.copyfile(archive,temporary);os.replace(temporary,latest)
            digest=sha(archive);receipt=dict(epoch=epoch,global_step=global_step,checkpoint=archive.name,sha256=digest,completed_at=now(),protocol_sha256=state['protocol_sha256'])
            write(out/f'epoch_{epoch}_receipt.json',receipt);write(out/'latest_receipt.json',receipt)
            write(out/'status.json',dict(state='training' if epoch<20 else 'trained',**receipt))
            if (STUDY/'STOP').exists():raise RuntimeError('Requested STOP after committed epoch')
        engine.save_ckpt=save_hook
        # Read-only post-step receipt confirms real optimizer updates, without extra forwards.
        count=[t.status.get('global_step',0)]
        def after_step(opt,args,kwargs):
            count[0]+=1
            if count[0]==1 or count[0]%50==0:
                write(out/'progress.json',dict(updated_at=now(),pid=os.getpid(),optimizer_steps=count[0],epoch=(count[0]-1)//782+1,step_in_epoch=(count[0]-1)%782+1))
        t.optimizer.register_step_post_hook(after_step)
        start=time.time();t.train()
        saved=read(out/'latest_receipt.json');assert saved['epoch']==20 and saved['global_step']==15640
        write(out/'training_completion.json',dict(completed_at=now(),seconds=time.time()-start,**{k:v for k,v in saved.items() if k!='completed_at'},selection_pending=True))

def coordinate(resume=False):
    environment();verify_sources()
    with StageLock(STUDY,'coordinator'):
        try:
            for tag in 'AB':
                if (STUDY/'STOP').exists():raise RuntimeError('STOP before next reader')
                out=ROOT/f'model1/runs/cryptbb/{tag}'
                if (out/'training_completion.json').exists():
                    r=read(out/'training_completion.json');assert r['epoch']==20 and sha(out/r['checkpoint'])==r['sha256'];continue
                cmd=[sys.executable,'-B','-u',str(ROOT/'Capri/capri.py'),'train-cryptbb-model1','worker','--dataset',tag]
                if (out/'latest.pth').exists():
                    assert resume,'Existing checkpoint; explicit coordinator --resume required';cmd+=['--resume']
                write(STUDY/'status.json',dict(state='training',dataset=tag,updated_at=now(),pid=os.getpid()))
                with (STUDY/f'train_{tag}.log').open('a',encoding='utf8') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    write(STUDY/'active_child.json',dict(pid=child.pid,dataset=tag,command=cmd,started_at=now()));code=child.wait()
                assert code==0,f'Reader {tag} failed with exit {code}; inspect log'
            write(STUDY/'completion.json',dict(completed_at=now(),readers=[read(ROOT/f'model1/runs/cryptbb/{tag}/training_completion.json') for tag in 'AB'],selection_pending=True,later_models_started=False))
            write(STUDY/'status.json',dict(state='training_complete_selection_pending',updated_at=now()))
        except Exception as exc:
            write(STUDY/'status.json',dict(state='failed',error=repr(exc),updated_at=now()));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','smoke','coordinate','worker']);ap.add_argument('--dataset',choices=['A','B']);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if a.stage=='worker':assert a.dataset;return worker(a.dataset,a.resume)
    if a.stage=='coordinate':return coordinate(a.resume)
    return globals()[a.stage]()
