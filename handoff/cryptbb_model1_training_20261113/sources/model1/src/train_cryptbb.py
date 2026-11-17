"""Model1 CryptBB launch helper; default verifies readiness, never trains.

Only --start-training after a new explicit user instruction launches a reader.
"""
import argparse, sys, os, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.decoder.common import read,write,sha,now,StageLock


def readiness(tag):
    prep=ROOT/'handoff/cryptbb_model1_preparation'
    required=[prep/'completion.json',prep/'datasets_ready_v2.json',prep/'agreement_protocol.json']
    missing=[str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:raise RuntimeError('Incomplete preparation: '+str(missing))
    complete=read(required[0]);assert complete['all_preflights_passed'] and not complete['reader_training_started']
    ready=read(required[1]);assert ready['images_each']==100000 and ready['cross_dataset_pixel_overlap']==0
    assert complete['datasets']==ready
    row=next(r for r in ready['configs'] if r['dataset']==tag)
    assert sha(ROOT/row['path'])==row['sha256']
    preflight=read(ROOT/row['preflight'])
    assert preflight['passed'] and preflight['batch_shape']==[128,3,32,128]
    assert preflight['optimizer_steps']==0 and preflight['config_sha256']==row['sha256']
    pool=ROOT/f'datasets/cryptbb/{tag}';receipt=read(pool/'data_receipt.json')
    for path,key in [(pool/'train_lmdb/data.mdb','lmdb_sha256'),(pool/'sample_provenance.jsonl','provenance_sha256'),(ROOT/'datasets/cryptbb/charset.txt','charset_sha256')]:
        assert sha(path)==receipt[key],str(path)
    policy=read(required[2]);assert sha(required[2])==complete['selection_protocol_sha256']
    assert policy['selection']=='max_same_epoch_ab_exact_agreement_earliest_tie'
    assert policy['candidate_epochs']==list(range(2,21)) and not policy['real_labels_used'] and not policy['real_images_used_for_training']
    manifest=ROOT/policy['manifest'];assert sha(manifest)==policy['manifest_sha256'] and len(read(manifest))==551
    return row,policy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['A','B'],required=True)
    ap.add_argument('--start-training',action='store_true',help='Only after explicit new user authorization; default only checks readiness')
    ap.add_argument('--resume',action='store_true',help='Reserved; blocked until upstream continuation/RNG is validated')
    args=ap.parse_args()
    if not args.start_training:
        try:
            row,policy=readiness(args.dataset)
            print(dict(dataset=args.dataset,setup_ready=True,training_started=False,config=row['path'],selection=policy['selection'],note='New user authorization required before launch; resume not validated'))
        except (AssertionError,RuntimeError,FileNotFoundError,KeyError) as exc:
            print(dict(dataset=args.dataset,setup_ready=False,training_started=False,error=repr(exc)))
        return
    if args.resume:raise RuntimeError('Resume has not been validated for upstream decoder RNG/optimizer continuation; preserve existing checkpoints and implement verified recovery first')
    row,policy=readiness(args.dataset)
    for key in ['TEMP','TMP','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        if key not in os.environ or Path(os.environ[key]).drive.upper()!='E:':raise RuntimeError('Source environment.ps1; all caches must be on E:')
    out=ROOT/f'model1/runs/cryptbb/{args.dataset}'
    if (out/'latest.pth').exists():raise RuntimeError('Existing reader checkpoint; do not overwrite or restart')
    import numpy as np,torch
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer
    assert torch.cuda.is_available()
    cfg=Config(str(ROOT/row['path']))
    assert cfg.cfg['Global']['pretrained_model'] is None and cfg.cfg['Global']['checkpoints'] is None and cfg.cfg['Global']['epoch_num']==20 and cfg.cfg['Eval'] is None
    assert cfg.cfg['Train']['loader']['min_ratio']==cfg.cfg['Train']['loader']['max_ratio']==4
    # Whole-run lock prevents two unarbitrated full readers sharing this16GB GPU.
    with StageLock(ROOT/'handoff/cryptbb_model1_preparation','reader_training'), StageLock(out,'training'):
        torch.set_num_threads(8);trainer=Trainer(cfg,mode='train',task='rec')
        assert trainer.valid_dataloader is None and len(trainer.train_dataloader.dataset)==100000 and len(trainer.train_dataloader)==782
        for i,module in enumerate(trainer.model.modules()):
            if isinstance(getattr(module,'rng',None),np.random.Generator):module.rng=np.random.default_rng(48+i)
        write(out/'launch.json',dict(started_at=now(),dataset=args.dataset,seed=48,epochs=20,initialization='from scratch',config_sha256=row['sha256'],selection_protocol_sha256=sha(ROOT/'handoff/cryptbb_model1_preparation/agreement_protocol.json'),selection=policy['selection'],budget_steps=15640))
        started=time.time();trainer.train()
        ck=torch.load(out/'latest.pth',map_location='cpu',weights_only=False)
        assert ck['epoch']==20 and ck['global_step']==15640
        write(out/'training_completion.json',dict(completed_at=now(),epoch=20,global_step=15640,seconds=time.time()-started,checkpoint=str(out/'latest.pth'),sha256=sha(out/'latest.pth'),selection_pending=True))

if __name__=='__main__':main()
