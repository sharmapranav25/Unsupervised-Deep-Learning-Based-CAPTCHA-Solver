"""User-requested Model1 A/B all-epoch diagnostics; never optimizes readers."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from capri_controller.training import ab_status_evaluation as evaluator
from capri_controller.training.ab_curriculum import ROOT, STUDY, read, write, now

def collect():
    rows=[]
    for epoch in range(1,21):
        row={'epoch':epoch}
        for tag in ('A','B'):
            path=ROOT/f'model1/results/darkbay/{tag}/interim_cuda_epoch{epoch:02d}_seed48/summary.json'
            row[tag]=read(path) if path.exists() else None
        rows.append(row)
    for tag in ('A','B'):
        write(ROOT/f'model1/results/darkbay/{tag}/epoch_accuracy.json',dict(
            updated_at=now(),dataset=tag,images=900,seed=48,selection='fixed final epoch20',
            diagnostic_only=True,epochs=[r[tag] for r in rows if r[tag] is not None]))
    write(STUDY/'model1_epoch_accuracy.json',dict(updated_at=now(),rows=rows,
        diagnostic_only=True,selection='fixed final epoch20',averaged=False))
    lines=['# Model1 DarkBay accuracy by epoch','',
        'Exact case-sensitive accuracy on the same 900 historical real images. Seed48; A and B reported separately. Predictions frozen before scoring. Diagnostics do not change fixed final epoch20 selection.','',
        '| Epoch | Model1-A | Model1-B |','|---|---|---|']
    for r in rows:
        values=[f"{r[t]['correct']}/900 ({r[t]['accuracy_percent']:.2f}%)" if r[t] else ('Unavailable: checkpoint not retained' if r['epoch']==1 else 'Pending') for t in ('A','B')]
        lines.append(f"| {r['epoch']} | {values[0]} | {values[1]} |")
    (STUDY/'model1_epoch_accuracy.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return sum(r['A'] is not None and r['B'] is not None for r in rows)

if __name__=='__main__':
    for epoch in range(2,21):
        for tag in ('A','B'):
            ck=ROOT/f'model1/runs/darkbay/{tag}/checkpoints/epoch_{epoch}.pth' if epoch<20 else ROOT/f'model1/models/parseq/darkbay/{tag}/epoch_20.pth'
            assert ck.is_file()
    collect()
    try:
        for epoch in range(2,21):
            write(STUDY/'model1_epoch_accuracy_status.json',dict(state='evaluating',epoch=epoch,updated_at=now()))
            sys.argv=['evaluate_ab_checkpoint','--seed','48','--epoch',str(epoch),'--model','1','--model-only','--device','cuda']
            evaluator.main()
            completed=collect()
            print(f'ACCURACY_TABLE {completed}/19 available epochs saved',flush=True)
        write(STUDY/'model1_epoch_accuracy_status.json',dict(state='complete',epochs=19,unavailable_epochs=[1],reason='Original epoch1 checkpoint not retained; original save_epoch_step=[1,1] uses epoch > 1',updated_at=now()))
    except Exception as exc:
        write(STUDY/'model1_epoch_accuracy_status.json',dict(state='failed',error=repr(exc),updated_at=now()))
        raise
