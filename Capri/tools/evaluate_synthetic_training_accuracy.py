"""Read-only checkpoint inference on own synthetic training pool."""
import argparse
import gc
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from capri_controller.training import ab_curriculum as core

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epoch',type=int,required=True)
    args=ap.parse_args()
    core.bootstrap()
    import torch
    results=[]
    for number in (1,2):
        for tag in ('A','B'):
            name=f'epoch_{args.epoch}.pth' if number==1 else f'epoch_{args.epoch:02d}.pth'
            checkpoint=core.ROOT/f'model{number}/runs/darkbay/{tag}/checkpoints'/name
            if number==1 and args.epoch==20:
                checkpoint=core.ROOT/f'model1/models/parseq/darkbay/{tag}/epoch_20.pth'
            digest=core.sha(checkpoint)
            out=core.ROOT/f'model{number}/results/darkbay/{tag}/synthetic_training_epoch{args.epoch:02d}'
            if (out/'summary.json').exists():
                summary=core.read(out/'summary.json')
                assert summary['checkpoint_sha256']==digest
                assert summary['predictions_sha256']==core.sha(out/'predictions.json')
                results.append(summary)
                continue
            if (out/'protocol.json').exists():
                assert core.read(out/'protocol.json')['checkpoint_sha256']==digest
            core.write(out/'protocol.json',dict(model=number,dataset=tag,epoch=args.epoch,
                checkpoint=str(checkpoint),checkpoint_sha256=digest,created_at=core.now(),
                scope='Own 100000-image synthetic training pool; not held-out validation',
                decoding='Same score_pool as Model3 adaptive rescoring, including fixed architecture-horizon retry',
                evaluator_sha256=core.sha(Path(__file__)),training_changed=False))
            print(f'START Model{number}-{tag} epoch{args.epoch}',flush=True)
            seconds=None
            if not (out/'predictions.json').exists():
                with core.gpu_slot():
                    model,decoder=core.make_model(core.config(tag,out))
                    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    assert ck['epoch']==args.epoch
                    model.load_state_dict(ck['state_dict'],strict=True)
                    del ck
                ds=core.dataset(tag)
                _,seconds=core.score_pool(model,decoder,ds,out)
                ds.env.close()
                del model,decoder,ds
                gc.collect()
                torch.cuda.empty_cache()
            predictions=core.read(out/'predictions.json')
            assert len(predictions)==core.N
            assert {p['index'] for p in predictions}==set(range(1,core.N+1))
            import lmdb
            env=lmdb.open(str(core.ROOT/f'datasets/darkbay/{tag}/train_lmdb'),readonly=True,lock=False,readahead=False)
            with env.begin() as tx:
                correct=sum(p['prediction']==tx.get(('label-%09d'%p['index']).encode()).decode() for p in predictions)
            env.close()
            summary=dict(model=number,dataset=tag,epoch=args.epoch,correct=correct,total=core.N,
                accuracy_percent=100*correct/core.N,checkpoint_sha256=digest,
                predictions_sha256=core.sha(out/'predictions.json'),seconds=seconds,
                scope='Synthetic training-pool exact match',created_at=core.now())
            core.write(out/'summary.json',summary)
            results.append(summary)
            print('RESULT '+str(summary),flush=True)
    core.write(core.STUDY/f'synthetic_training_comparison_epoch{args.epoch:02d}.json',results)
if __name__=='__main__':
    main()
