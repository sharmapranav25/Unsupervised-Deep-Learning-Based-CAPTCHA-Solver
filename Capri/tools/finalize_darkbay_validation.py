"""Finalize adopted DarkBay validation-only checkpoint selection after20epochs."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from capri_controller.training import ab_curriculum as core
ROOT=core.ROOT;STUDY=core.STUDY
SWEEP=STUDY/'controller4_validation_sweep_final'
OUT=STUDY/'controller4_validation_selected_real_final'
def main():
    verified=core.read(SWEEP/'verification.json');assert verified['passed'] and verified['evaluations']==118
    sweep=core.read(SWEEP/'results.json');protocol=core.read(SWEEP/'protocol.json')
    jobs={(j['model'],j['dataset'],j['epoch']):j for j in protocol['jobs']}
    historical=core.read(STUDY/'family_report.json')
    for n in (2,3):
        for tag in ('A','B'):
            done=core.read(ROOT/f'model{n}/runs/darkbay/{tag}/completion.json')
            assert done['epoch']==20 and done['global_step']==15640 and done['sha256']==jobs[n,tag,20]['checkpoint_sha256']
            d=ROOT/f'model{n}/results/darkbay/{tag}/evaluation_epoch20'
            fr=core.read(d/'prediction_freeze.json');assert fr['checkpoint_sha256']==done['sha256'] and fr['predictions_sha256']==core.sha(d/'predictions.json')
    choices=[]
    for n in (1,2,3):
        for tag in ('A','B'):
            rows=[r for r in sweep['results'] if r['model']==n and r['dataset']==tag]
            assert sorted(r['epoch'] for r in rows)==list(range(2 if n==1 else 1,21))
            b=max(rows,key=lambda r:(r['correct'],-r['epoch']))
            choices.append(dict(model=n,dataset=tag,epoch=b['epoch'],validation_correct=b['correct'],validation_accuracy_percent=b['accuracy_percent'],checkpoint=b['checkpoint'],checkpoint_sha256=b['checkpoint_sha256']))
    freeze=OUT/'selection_freeze.json'
    selection=dict(rule='Highest frozen synthetic validation exact match; earliest epoch on ties',choices=choices,validation_results_sha256=core.sha(SWEEP/'results.json'),policy_sha256=core.sha(STUDY/'validation_selection_policy.json'),real_scores_used_for_selection=False,eligible='Model1 retained2-20; Model2/3 all1-20',created_at=core.now())
    if freeze.exists():assert core.read(freeze)['choices']==choices
    else:core.write(freeze,selection)
    frozen=[]
    for c in choices:
        n,t,e=c['model'],c['dataset'],c['epoch']
        out=ROOT/f'model{n}/results/darkbay/{t}/interim_cuda_epoch{e:02d}_seed48'
        if not (out/'prediction_freeze.json').exists() and e==20:out=ROOT/f'model{n}/results/darkbay/{t}/evaluation_epoch20'
        assert (out/'prediction_freeze.json').exists(),'Missing selected real predictions: '+str(out)
        fr=core.read(out/'prediction_freeze.json');assert fr['checkpoint_sha256']==c['checkpoint_sha256'] and fr['predictions_sha256']==core.sha(out/'predictions.json')
        rows=core.read(out/'predictions.json');assert len(rows)==900 and len({r['opaque_id'] for r in rows})==900
        frozen.append((c,out,rows,fr))
    labels=core.read(ROOT/'evaluation_data/darkbay/scoring_labels/scoring_labels.json')
    results=[]
    for c,out,rows,fr in frozen:
        assert {r['opaque_id'] for r in rows}==set(labels)
        count=sum(r['prediction']==labels[r['opaque_id']] for r in rows);assert count==core.read(out/'summary.json')['correct']
        results.append(dict(**c,real_correct=count,real_total=900,real_accuracy_percent=count/9,predictions= (out/'predictions.json').relative_to(ROOT).as_posix(),predictions_sha256=fr['predictions_sha256'],training_outputs=f"model{c['model']}/runs/darkbay/{c['dataset']}",validation_outputs=f"model{c['model']}/results/darkbay/{c['dataset']}/controller4_validation",training_completed_epochs=20))
    report=dict(state='DarkBay complete under adopted validation selection',created_at=core.now(),results=results,selection_freeze_sha256=core.sha(freeze),verification='All118 validation evaluations verified; four curriculum completion checkpoint hashes match; six selected real predictions/identities/counts verified',historical_fixed20_report='handoff/darkbay_curriculum_20261113/family_report.json',validation_sweep='handoff/darkbay_curriculum_20261113/controller4_validation_sweep_final',plots='thesis/figures/darkbay/controller4_validation_final',length_threshold=historical['length_threshold'],limitations=['Validation-selection policy adopted retrospectively after real diagnostics had been viewed','Real set is repeatedly inspected historical data','Model1 epoch1 checkpoint unavailable','One run per dataset; no repeated-seed evidence','Synthetic validation is separate from training, but visual similarity does not prove real-domain equivalence'],cryptbb_scope='Preparation and setup only; no reader training')
    core.write(OUT/'results.json',report);core.write(STUDY/'family_report_validation_selected.json',report)
    lines=['# DarkBay final results: synthetic-validation-selected checkpoints','','All readers completed20training epochs. Each epoch choice uses only the frozen1000-image synthetic validation accuracy; earliest epoch breaks ties. Real labels score the frozen choice. The adopted rule is retrospective for this existing study.','','| Reader | Selected epoch | Validation correct/1000 | Real correct/900 | Real accuracy |','|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"| Model{r['model']}-{r['dataset']} | {r['epoch']} | {r['validation_correct']} | {r['real_correct']} | {r['real_accuracy_percent']:.2f}% |")
    lines+=['','## Threshold','', 'The40% length-accuracy claim is not established. DarkBay used the unique leading count with89/100 votes for6; the rule is vote agreement >=50/100 with ties rejected. CryptBB must recompute its own vote from reference pixels.','','## Saved outputs','', '- Training checkpoints, metrics, logs and loss plots: model{1,2,3}/runs/darkbay/{A,B}; Model1 final20 archive also under model1/models/parseq/darkbay/{A,B}.','- Frozen ranking: model2/rankings/darkbay/{A,B}; Model3 references the same opposite-Model1 rankings and saves updates under its run adaptive/ folder.','- All validation predictions: model{1,2,3}/results/darkbay/{A,B}/controller4_validation/epoch_NN.','- Complete validation table/protocol/verification: handoff/darkbay_curriculum_20261113/controller4_validation_sweep_final.','- Final selection freeze and verified selected real results: handoff/darkbay_curriculum_20261113/controller4_validation_selected_real_final.','- Charts: thesis/figures/darkbay/controller4_validation_final.','- Real prediction paths and hashes for each selected reader are listed in results.json.','- Historical fixed20 real results: model{2,3}/results/darkbay/{A,B}/evaluation_epoch20 and handoff/darkbay_curriculum_20261113/family_report.json. Earlier validation-through17 reports remain preserved.','','## Limits and issues','', 'No training failure was observed. Model1 epoch1 is unavailable. This is one trajectory per dataset, and the real test has been repeatedly inspected. Synthetic accuracy is low and validation/real trajectories do not align consistently; selections nevertheless remain fixed by the adopted synthetic-only rule. Source, checkpoint, prediction and image identities were verified.','', 'The user-authorized next stage is CryptBB dataset and Model1 setup preparation only. CryptBB training, City and other STR models remain unauthorized at this stage.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
