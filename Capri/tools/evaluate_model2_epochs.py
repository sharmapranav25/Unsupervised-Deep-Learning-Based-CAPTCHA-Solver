"""User-requested Model2 A/B all-epoch diagnostics; never optimizes readers."""
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
            path=ROOT/f'model2/results/darkbay/{tag}/interim_cuda_epoch{epoch:02d}_seed48/summary.json'
            row[tag]=read(path) if path.exists() else None
        rows.append(row)
    for tag in ('A','B'):
        write(ROOT/f'model2/results/darkbay/{tag}/epoch_accuracy.json',dict(
            updated_at=now(),dataset=tag,images=900,seed=48,selection='fixed final epoch20',
            diagnostic_only=True,epochs=[r[tag] for r in rows if r[tag] is not None]))
    write(STUDY/'model2_epoch_accuracy.json',dict(updated_at=now(),rows=rows,
        diagnostic_only=True,selection='fixed final epoch20',averaged=False))
    lines=['# Model2 DarkBay accuracy by epoch','',
        'Exact case-sensitive accuracy on the same 900 historical real images. Seed48; A and B reported separately. Predictions frozen before scoring. Diagnostics do not change fixed final epoch20 selection.','',
        '| Epoch | Model2-A | Model2-B |','|---|---|---|']
    for r in rows:
        values=[f"{r[t]['correct']}/900 ({r[t]['accuracy_percent']:.2f}%)" if r[t] else 'Pending' for t in ('A','B')]
        lines.append(f"| {r['epoch']} | {values[0]} | {values[1]} |")
    (STUDY/'model2_epoch_accuracy.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return sum(r['A'] is not None and r['B'] is not None for r in rows)

if __name__=='__main__':
    for epoch in range(1,21):
        for tag in ('A','B'):
            assert (ROOT/f'model2/runs/darkbay/{tag}/checkpoints/epoch_{epoch:02d}.pth').is_file()
    collect()
    try:
        for epoch in range(1,21):
            write(STUDY/'model2_epoch_accuracy_status.json',dict(state='evaluating',epoch=epoch,updated_at=now()))
            sys.argv=['evaluate_ab_checkpoint','--seed','48','--epoch',str(epoch),'--model','2','--model-only','--device','cuda']
            evaluator.main()
            completed=collect()
            print(f'ACCURACY_TABLE {completed}/20 epochs saved',flush=True)
        write(STUDY/'model2_epoch_accuracy_status.json',dict(state='complete',epochs=20,updated_at=now()))
    except Exception as exc:
        write(STUDY/'model2_epoch_accuracy_status.json',dict(state='failed',error=repr(exc),updated_at=now()))
        raise
