"""Future explicit Model1 launch; default is a read-only readiness check."""
import argparse,sys,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.decoder.common import read,write,sha,now,StageLock

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['A','B'],required=True)
    ap.add_argument('--start-training',action='store_true',help='Use only after new user authorization; default never trains')
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    out=ROOT/f'model1/runs/cryptbb/{args.dataset}';prep=ROOT/'handoff/cryptbb_model1_preparation'
    complete=prep/'completion.json';validation=prep/'validation_protocol.json'
    missing=[str(p.relative_to(ROOT)) for p in [complete,validation] if not p.exists()]
    if not args.start_training:
        print({'dataset':args.dataset,'training_started':False,'missing_readiness':missing,'note':'Training requires explicit new user authorization, completed data/preflight and frozen synthetic validation.'});return
    if missing:raise RuntimeError('Not ready: '+str(missing))
    for k in ['TEMP','TMP','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        if k not in os.environ or Path(os.environ[k]).drive.upper()!='E:':raise RuntimeError('Source environment.ps1 first; all caches must be on E:')
    ready=read(prep/'datasets_ready.json');r=next(r for r in ready['configs'] if r['dataset']==args.dataset)
    config=ROOT/r['path'];assert sha(config)==r['sha256']
    assert read(out/'preflight/receipt.json')['passed']
    d=ROOT/f'datasets/cryptbb/{args.dataset}';receipt=read(d/'data_receipt.json');assert sha(d/'train_lmdb/data.mdb')==receipt['lmdb_sha256']
    v=read(validation);assert v['training_overlap']==0 and v['selection']=='highest_synthetic_exact_match_earliest_tie' and not v['real_labels_used']
    manifest=ROOT/v['manifest'];assert sha(manifest)==v['manifest_sha256']
    latest=out/'latest.pth'
    if latest.exists() and not args.resume:raise RuntimeError('Existing reader checkpoint; explicit --resume required')
    if args.resume and not latest.exists():raise RuntimeError('No checkpoint to resume')
    import numpy as np,torch
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer
    assert torch.cuda.is_available()
    cfg=Config(str(config));assert cfg.cfg['Global']['pretrained_model'] is None and cfg.cfg['Global']['epoch_num']==20 and cfg.cfg['Eval'] is None
    cfg.cfg['Global']['checkpoints']=str(latest) if args.resume else None
    with StageLock(out,'training'):
        torch.set_num_threads(8);trainer=Trainer(cfg,mode='train',task='rec')
        assert trainer.valid_dataloader is None and len(trainer.train_dataloader.dataset)==100000 and len(trainer.train_dataloader)==782
        for i,m in enumerate(trainer.model.modules()):
            if isinstance(getattr(m,'rng',None),np.random.Generator):m.rng=np.random.default_rng(48+i)
        write(out/'launch.json',dict(started_at=now(),dataset=args.dataset,seed=48,epochs=20,initialization='from scratch' if not args.resume else 'resume',config_sha256=r['sha256'],validation_protocol_sha256=sha(validation),selection='synthetic validation only after training; real labels never select',budget_steps=15640))
        started=time.time();trainer.train()
        ck=torch.load(latest,map_location='cpu',weights_only=False);assert ck['epoch']==20 and ck['global_step']==15640
        write(out/'training_completion.json',dict(completed_at=now(),epoch=20,global_step=15640,seconds=time.time()-started,checkpoint=str(latest),sha256=sha(latest),selection_pending=True))
if __name__=='__main__':main()
