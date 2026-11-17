"""Single-owner CPU evaluation, isolated original-v1 selection, and progress/final report."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from . import cryptbb_cumulative_static as M


def run_module(module,args,log):
    with log.open('a',encoding='utf8') as stream:
        p=subprocess.Popen([sys.executable,'-B','-u','-m','capri_controller.training.'+module,*args],
            cwd=M.ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    import psutil
    M.write(M.STUDY/'evaluation_child.json',dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),module=module,args=args,created_at=M.now()))
    while p.poll() is None:
        try: M.stop_check()
        except M.Stopped:
            # CPU evaluator has no training state; allow current immutable evaluation to finish.
            p.wait(timeout=60)
            raise
        time.sleep(2)
    assert p.returncode==0,(module,args,p.returncode)


def evaluate(watch=False):
    with M.StageLock(M.STUDY,'evaluation_report'):
        try:
            M.check_sources()
            while True:
                M.stop_check()
                committed=[]
                for tag in 'AB':
                    for path in sorted((M.run_dir(2,tag)/'checkpoints').glob('epoch_*.json')):
                        r=M.read(path); assert r['protocol_sha256']==M.check_sources()
                        committed.append(dict(dataset=tag,epoch=r['epoch'],sha256=r['sha256']))
                M.write(M.STUDY/'evaluation_request.json',dict(created_at=M.now(),checkpoints=committed))
                for row in committed:
                    M.stop_check(); tag,epoch=row['dataset'],row['epoch']; out=M.result_dir(tag,epoch)
                    if (out/'summary.json').exists():
                        verify_result(out,row['sha256'])
                        continue
                    # Existing incomplete output is deliberately retained for inspection.
                    run_module('cryptbb_cumulative_evaluation',['--dataset',tag,'--epoch',str(epoch),
                        '--request-manifest',str(M.STUDY/'evaluation_request.json')],M.STUDY/f'evaluation_{tag}_{epoch:02d}.log')
                    verify_result(out,row['sha256'])
                    report()
                complete=report()
                if complete or not watch: return
                time.sleep(20)
        except BaseException:
            M.write(M.STUDY/'evaluation_failure.json',dict(error=traceback.format_exc(),updated_at=M.now()))
            raise


def verify_result(out,expected=None):
    summary=M.read(out/'summary.json'); freeze=M.read(out/'prediction_freeze.json')
    assert M.read(out/'verification.json')['passed']
    assert M.sha(out/'predictions.json')==freeze['sha256']
    assert freeze['checkpoint_sha256']==summary['checkpoint_sha256']
    if expected: assert summary['checkpoint_sha256']==expected
    rows=M.read(out/'per_image_results.json')
    assert len(rows)==251 and len({r['opaque_id'] for r in rows})==251
    correct=sum(r['prediction']==r['label'] for r in rows)
    assert correct==summary['correct'] and summary['total']==251
    pred=M.read(out/'predictions.json')
    assert {r['opaque_id']:r['prediction'] for r in pred}=={r['opaque_id']:r['prediction'] for r in rows}
    return summary


def historical(number,tag,epoch):
    family='cryptbb_v3' if number==1 else 'cryptbb_v3_rank1_v1'
    parent=M.ROOT/f'model{number}/results'/family/tag
    candidates=list(parent.glob(f'requested_cpu_epoch{epoch:02d}/summary.json'))
    candidates+=list(parent.glob(f'requested_cpu_epoch{epoch:02d}_attempt*/summary.json'))
    for p in reversed(candidates):
        if (p.parent/'verification.json').exists():
            return verify_result(p.parent)
    return None


def select(tag):
    path=M.STUDY/f'selection_{tag}.json'
    if not path.exists():
        run_module('cryptbb_cumulative_select',[tag,str(path)],M.STUDY/f'selection_{tag}.log')
    r=M.read(path)
    assert r['candidate_epochs']==list(range(1,21)) and r['labels_read'] is False
    for p in r['prediction_sources']:
        assert M.sha(Path(p['path']))==p['sha256']
        assert M.sha(M.run_dir(2,tag)/f"checkpoints/epoch_{p['epoch']:02d}.pth")==p['checkpoint_sha256']
    return r


def report():
    rows=[]; readers={}; all_complete=True
    for tag in 'AB':
        run=M.run_dir(2,tag); history=M.read(run/'history.json') if (run/'history.json').exists() else []
        status=M.read(run/'status.json') if (run/'status.json').exists() else dict(state='queued',completed_epoch=0,global_step=0)
        measured=[]
        for h in history:
            epoch=h['epoch']; out=M.result_dir(tag,epoch)
            if not (out/'summary.json').exists(): continue
            current=verify_result(out,M.read(run/f'checkpoints/epoch_{epoch:02d}.json')['sha256'])
            baseline=historical(1,tag,epoch); old=historical(2,tag,epoch) if tag=='A' or epoch<=8 else None
            row=dict(dataset=tag,epoch=epoch,mean_training_loss=h['mean_loss'],training_wall_seconds=h['training_wall_seconds'],
                correct=current['correct'],total=251,accuracy_percent=100*current['correct']/251,
                model1_correct=baseline['correct'] if baseline else None,previous_static_correct=old['correct'] if old else None,
                model1_difference_pp=100*(current['correct']-baseline['correct'])/251 if baseline else None,
                previous_static_difference_pp=100*(current['correct']-old['correct'])/251 if old else None)
            rows.append(row); measured.append(row)
        selection=None
        if len(measured)==20:
            # Selector is a fresh process with an audit hook denying all score artifacts.
            selected=select(tag); epoch=selected['selected_epoch']
            selected_score=next(r for r in measured if r['epoch']==epoch)
            selection=dict(epoch=epoch,top3=selected['top3'],resolved_k=selected['resolved_k'],
                fallback_used=selected['fallback_used'],mask_diagnostics=selected['mask_diagnostics'],
                checkpoint_sha256=M.sha(run/f'checkpoints/epoch_{epoch:02d}.pth'),correct=selected_score['correct'],
                accuracy_percent=selected_score['accuracy_percent'],selection_sha256=M.sha(M.STUDY/f'selection_{tag}.json'))
        complete=(run/'completion.json').exists() and len(measured)==20 and selection is not None
        if complete: M.valid_completion(tag)
        all_complete &= complete
        best=max((r['correct'] for r in measured),default=None)
        readers[tag]=dict(status=status,completed_epochs=len(history),evaluated_epochs=len(measured),
            backlog=[r['epoch'] for r in history if r['epoch'] not in [m['epoch'] for m in measured]],
            training_wall_seconds=sum(r['training_wall_seconds'] for r in history),
            committed_updates=len(history)*782,selection=selection,
            all20_mean_accuracy_percent=sum(r['accuracy_percent'] for r in measured)/20 if len(measured)==20 else None,
            retrospective_best=dict(correct=best,epochs=[r['epoch'] for r in measured if r['correct']==best],diagnostic_only=True),
            model1_all20_mean_percent=sum(historical(1,tag,e)['correct'] for e in range(1,21))/20/251*100,
            previous_static_shared_mean_percent=sum(r['previous_static_correct'] for r in measured if r['previous_static_correct'] is not None)/max(1,sum(r['previous_static_correct'] is not None for r in measured))/251*100,
            cumulative_shared_mean_percent=sum(r['correct'] for r in measured if r['previous_static_correct'] is not None)/max(1,sum(r['previous_static_correct'] is not None for r in measured))/251*100,
            shared_epochs=[r['epoch'] for r in measured if r['previous_static_correct'] is not None])
    failures={p.name:M.read(p) for p in M.STUDY.glob('*failure*.json')}
    result=dict(updated_at=M.now(),complete=all_complete,readers=readers,rows=rows,failures=failures,
        automatic_restarts=0,limits=['Single seed48 pair; no schedule tuning on clean synthetic validation.',
        'Uniform/random expansion/exact exposure shuffled controls have manifests only, not training results.',
        'Historical Model1 and previous static comparisons do not establish curriculum causality.',
        'Previous static B stopped with eight saved epochs; compare shared epochs only.',
        'Model2 loss is epoch mean; Model1 logged loss may be rolling median.'])
    M.write(M.STUDY/'results.json',result)
    if rows:
        with (M.STUDY/'results.csv').open('w',newline='',encoding='utf8') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# CryptBB cumulative static morning report','',f"Updated {result['updated_at']}. Status: {'COMPLETE' if all_complete else 'IN PROGRESS; final selections pending until all20 predictions exist'}.",'']
    for tag,r in readers.items():
        lines += [f"## Reader {tag}",'',f"Saved {r['completed_epochs']}/20 epochs; {r['committed_updates']}/15640 committed updates. Live status: {json.dumps(r['status'])}.",
            f"Training wall time: {r['training_wall_seconds']/3600:.2f} hours. Evaluated {r['evaluated_epochs']}; backlog {r['backlog']}.",'']
        if r['selection']:
            s=r['selection']; lines += [f"Original v1 selected epoch {s['epoch']}: {s['correct']}/251 ({s['accuracy_percent']:.2f}%). Top three {s['top3']}; k={s['resolved_k']}; fallback={s['fallback_used']}. Checkpoint SHA256 `{s['checkpoint_sha256']}`.",
                f"Mask checks: `{json.dumps(s['mask_diagnostics'])}`.",
                f"All20 mean {r['all20_mean_accuracy_percent']:.4f}%; Model1 all20 mean {r['model1_all20_mean_percent']:.4f}%. Retrospective best (diagnostic only): {r['retrospective_best']}.",'']
        if r['shared_epochs']:
            lines += [f"Shared previous-static epochs {r['shared_epochs']}: cumulative mean {r['cumulative_shared_mean_percent']:.4f}%, previous static mean {r['previous_static_shared_mean_percent']:.4f}%.",'']
    lines += ['| Reader | Epoch | Mean loss | Correct/251 | Accuracy | Model1 correct | Delta pp | Previous static correct | Delta pp |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x): return '—' if x is None else f'{x:.2f}'
    for r in rows: lines.append(f"| {r['dataset']} | {r['epoch']} | {r['mean_training_loss']:.6f} | {r['correct']}/251 | {r['accuracy_percent']:.2f}% | {r['model1_correct']} | {fmt(r['model1_difference_pp'])} | {r['previous_static_correct'] if r['previous_static_correct'] is not None else '—'} | {fmt(r['previous_static_difference_pp'])} |")
    lines += ['',f'Failures: {json.dumps(failures)}. Automatic restarts: 0.','']+result['limits']
    (M.STUDY/'MORNING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    M.write(M.STUDY/'report_verification.json',dict(complete=all_complete,updated_at=M.now(),evaluated=len(rows),
        report_sha256=M.sha(M.STUDY/'MORNING_REPORT.md'),results_sha256=M.sha(M.STUDY/'results.json')))
    return all_complete
