"""Freeze the saved checkpoint pool, measure each once, and tabulate diagnostics."""
import csv,io,json,os,subprocess,sys,traceback
from pathlib import Path
from ..decoder.common import read,write,sha,now,StageLock
from . import cryptbb_v3_model2_selected_v1 as M
ROOT=M.ROOT;STUDY=ROOT/'handoff/cryptbb_v3_model2_measurements_20261115'

def report():
    rows=[]
    for r in read(STUDY/'requested_checkpoints.json')['checkpoints']:
        out=ROOT/f"model2/results/cryptbb_v3_rank1_v1/{r['dataset']}/requested_cpu_epoch{r['epoch']:02d}"
        if not (out/'summary.json').exists():continue
        summary=read(out/'summary.json');freeze=read(out/'prediction_freeze.json');verification=read(out/'verification.json')
        assert verification['passed'] and summary['checkpoint_sha256']==r['sha256']==freeze['checkpoint_sha256']
        assert sha(out/'predictions.json')==freeze['sha256']
        baseline=ROOT/f"model1/results/cryptbb_v3/{r['dataset']}/requested_cpu_epoch{r['epoch']:02d}"
        base=read(baseline/'summary.json');bf=read(baseline/'prediction_freeze.json')
        assert base['model']==1 and base['epoch']==r['epoch'] and base['total']==summary['total']==251
        assert sha(baseline/'predictions.json')==bf['sha256'] and base['checkpoint_sha256']==bf['checkpoint_sha256']
        rows.append(dict(dataset=r['dataset'],epoch=r['epoch'],model2_correct=summary['correct'],total=251,
            model2_accuracy_percent=summary['accuracy_percent'],model2_cer_percent=summary['cer_percent'],
            model1_correct=base['correct'],model1_accuracy_percent=base['accuracy_percent'],
            difference_correct=summary['correct']-base['correct'],difference_percentage_points=summary['accuracy_percent']-base['accuracy_percent'],
            checkpoint_sha256=r['sha256'],predictions_sha256=freeze['sha256'],result=str(out.relative_to(ROOT))))
    write(STUDY/'measurements.json',dict(updated_at=now(),images=251,case_sensitive_full_string=True,diagnostic_only=True,rows=rows))
    if rows:
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (STUDY/'measurements.csv').write_text(stream.getvalue(),encoding='utf8')
    return rows

def main():
    with StageLock(STUDY,'evaluation_batch'):
        try:
            request=read(STUDY/'requested_checkpoints.json');digest=M.check_sources();assert digest==request['training_protocol_sha256']
            for r in request['checkpoints']:
                assert sha(ROOT/r['checkpoint'])==r['sha256']
                write(STUDY/'status.json',dict(state='evaluating',dataset=r['dataset'],epoch=r['epoch'],updated_at=now()))
                args=[sys.executable,'-B','-u','-m','capri_controller.training.cryptbb_v3_model2_checkpoint_evaluation','--dataset',r['dataset'],'--epoch',str(r['epoch'])]
                with (STUDY/f"{r['dataset']}_epoch{r['epoch']:02d}.log").open('a',encoding='utf8') as log:
                    p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    assert p.wait()==0,(r['dataset'],r['epoch'])
                report()
            rows=report();assert len(rows)==len(request['checkpoints']) and M.check_sources()==digest
            write(STUDY/'completion.json',dict(completed_at=now(),checkpoints=len(rows),training_sources_unchanged=True,summary_sha256=sha(STUDY/'measurements.json')))
            write(STUDY/'status.json',dict(state='complete',checkpoints=len(rows),updated_at=now()))
        except Exception:
            write(STUDY/'status.json',dict(state='failed',error=traceback.format_exc(),updated_at=now()));raise

if __name__=='__main__':main()
