"""Finish all20 A diagnostics, freeze prediction-only selection, then assemble scored report."""
import json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('E:/pranav/Thesis-Pranav-Final');sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.decoder.common import read,write,sha,now
from capri_controller.training import cryptbb_v3_model2_selected_v1 as M
run=M.run_dir(2,'A');history=read(run/'history.json');assert [h['epoch'] for h in history]==list(range(1,21))
complete=read(run/'completion.json');assert complete['epoch']==20 and sha(run/complete['checkpoint'])==complete['sha256']
digest=M.check_sources();stamp=now().replace(':','').replace('.','_')
out=ROOT/'handoff/cryptbb_v3_model2_measurements_20261115'/('A_all20_'+stamp);out.mkdir()
rows=[]
for epoch in range(1,21):
    ck=run/f'checkpoints/epoch_{epoch:02d}.pth';rows.append(dict(dataset='A',epoch=epoch,checkpoint=str(ck.relative_to(ROOT)),sha256=sha(ck)))
request=out/'request.json';write(request,dict(frozen_at=now(),training_protocol_sha256=digest,checkpoints=rows))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1')
env['PYTHONPATH']=';'.join(str(ROOT/p) for p in ('Capri/src','model1/src/parseq_packages','model1/src/OpenOCR'))
env['TEMP']=env['TMP']=str(ROOT/'runtime/temporary')
for r in rows:
    dest=ROOT/f"model2/results/cryptbb_v3_rank1_v1/A/requested_cpu_epoch{r['epoch']:02d}"
    if not (dest/'summary.json').exists():
        print(f"Measuring A epoch{r['epoch']}",flush=True)
        with (out/f"epoch{r['epoch']:02d}.log").open('a',encoding='utf8') as log:
            subprocess.run([sys.executable,'-B','-u','-m','capri_controller.training.cryptbb_v3_model2_checkpoint_evaluation_v2','--dataset','A','--epoch',str(r['epoch']),'--request-manifest',str(request)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,check=True)
    freeze=read(dest/'prediction_freeze.json');assert freeze['checkpoint_sha256']==r['sha256'] and sha(dest/'predictions.json')==freeze['sha256']
    assert read(dest/'verification.json')['passed']
    print(f"A epoch{r['epoch']} predictions verified",flush=True)
selection_path=out/'reporting_selection_A.json'
selector=Path(__file__).with_name('final_selector_a.py');shutil.copyfile(selector,out/'selector_source.py')
print('Running unchanged original v1 selector in isolated label-blocked process',flush=True)
subprocess.run([sys.executable,'-B',str(selector),str(selection_path)],cwd=ROOT,env=env,check=True)
selection=read(selection_path);assert not selection['provisional'] and selection['candidate_epochs']==list(range(1,21)) and not selection['labels_read']
write(out/'selection_freeze.json',dict(frozen_at=now(),selection_sha256=sha(selection_path),selector_sha256=sha(selector),training_protocol_sha256=digest))
# Only now combine existing scoring summaries and compare with Model1.
table=[]
for r in rows:
    epoch=r['epoch'];dest=ROOT/f'model2/results/cryptbb_v3_rank1_v1/A/requested_cpu_epoch{epoch:02d}'
    result=read(dest/'summary.json');assert result['checkpoint_sha256']==r['sha256'] and result['total']==251
    baselines={t:read(ROOT/f'model1/results/cryptbb_v3/{t}/requested_cpu_epoch{epoch:02d}/summary.json') for t in 'AB'}
    table.append(dict(epoch=epoch,correct=result['correct'],total=251,accuracy_percent=result['accuracy_percent'],cer_percent=result['cer_percent'],mean_training_loss=history[epoch-1]['mean_loss'],model1_A_correct=baselines['A']['correct'],model1_B_correct=baselines['B']['correct'],selected=epoch==selection['selected_epoch']))
assert M.check_sources()==digest
best=max(r['correct'] for r in table);best_epochs=[r['epoch'] for r in table if r['correct']==best]
summary=dict(completed_at=now(),selection=selection['selected_epoch'],top3=selection['top3'],resolved_k=selection['resolved_k'],fallback_used=selection['fallback_used'],selection_sha256=sha(selection_path),best_observed_epochs=best_epochs,best_observed_correct=best,rows=table,training_changed=False,real_scores_used_for_selection=False)
write(out/'all_A_epochs.json',summary)
lines=['# CryptBB v3 Model2-A: all20 checkpoints','',f"Frozen original v1 rank-one choice: epoch{selection['selected_epoch']}; top3 {selection['top3']}; k{selection['resolved_k']}; fallback={selection['fallback_used']}.",'','Epoch | Model2-A correct/251 | Accuracy | Model1-A correct/251 | Model1-B correct/251 | Mean training loss','---|---:|---:|---:|---:|---:']
for r in table:lines.append(f"{r['epoch']}{' (selected)' if r['selected'] else ''} | {r['correct']} | {r['accuracy_percent']:.2f}% | {r['model1_A_correct']} | {r['model1_B_correct']} | {r['mean_training_loss']:.6f}")
lines+=['',f'Observed best (diagnostic only): epochs{best_epochs}, {best}/251. Selection used frozen predictions only; scoring summaries did not enter the selector.','B training unchanged.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
note=f"## {now()}: Model2-A all20 checkpoints measured and reporting selection frozen\n\nAll20 immutable A archives verified and evaluated on fixed251 CryptBB images with CPU FP32/two threads; existing results reused and missing15-19 measured. Original v1 estimator used in separate process blocking labels and scoring summaries. Final reporting choice epoch{selection['selected_epoch']}, top3 {selection['top3']}, resolved k{selection['resolved_k']}, fallback={selection['fallback_used']}. Prediction provenance and actual checkpoint hashes verified. Evidence: {out.relative_to(ROOT).as_posix()}/REPORT.md, reporting_selection_A.json and selection_freeze.json. Observed best diagnostic epochs{best_epochs}: {best}/251. No training settings, frozen source, initialization, curriculum or B execution changed.\n\n"
state=ROOT/'handoff/CURRENT_STATE.md';state.write_text(note+state.read_text(encoding='utf-8-sig'),encoding='utf8')
with (ROOT/'handoff/CHANGELOG.md').open('a',encoding='utf8') as f:f.write('\n'+note)
shutil.copyfile(__file__,out/'measurement_source.py')
write(out/'verification.json',dict(passed=True,epochs=20,images_per_epoch=251,training_sources_unchanged=True,selection_prediction_only=True,selection_sha256=sha(selection_path)))
print(json.dumps(dict(path=str(out),**summary),indent=2),flush=True)
