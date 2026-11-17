"""Freeze synthetic-validation epoch choices, then score their real predictions."""
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from capri_controller.training import ab_curriculum as core
ROOT=core.ROOT
SOURCE=core.STUDY/'controller4_validation_sweep'
OUT=core.STUDY/'controller4_validation_selected_real'

def main():
    data=core.read(SOURCE/'results.json')
    choices=[]
    for n in (1,2,3):
        for tag in ('A','B'):
            candidates=[r for r in data['results'] if r['model']==n and r['dataset']==tag]
            best=max(candidates,key=lambda r:(r['correct'],-r['epoch']))
            choices.append(dict(model=n,dataset=tag,epoch=best['epoch'],validation_correct=best['correct'],validation_total=1000,validation_accuracy_percent=best['accuracy_percent'],checkpoint=best['checkpoint'],checkpoint_sha256=best['checkpoint_sha256'],eligible_epochs=[r['epoch'] for r in candidates]))
    freeze=OUT/'selection_freeze.json'
    if freeze.exists():
        saved=core.read(freeze);assert saved['choices']==choices and saved['validation_results_sha256']==core.sha(SOURCE/'results.json')
    else:
        core.write(freeze,dict(created_at=core.now(),choices=choices,validation_results_sha256=core.sha(SOURCE/'results.json'),validation_manifest_sha256=core.read(SOURCE/'protocol.json')['manifest_sha256'],rule='Highest synthetic validation exact-match accuracy per reader; earliest epoch on ties',real_accuracy_used_for_choice=False,scope='Retrospective experiment; Model3 choices limited to epochs1-17',primary_study_selection='Fixed epoch20 remains unchanged',evaluator_sha256=core.sha(Path(__file__))))
    print('FROZEN '+str([(r['model'],r['dataset'],r['epoch']) for r in choices]),flush=True)
    for choice in choices:
        n,tag,e=choice['model'],choice['dataset'],choice['epoch']
        out=ROOT/f'model{n}/results/darkbay/{tag}/interim_cuda_epoch{e:02d}_seed48'
        if not (out/'summary.json').exists():
            # Existing diagnostic evaluates both A/B at this epoch; the companion is not a selected result.
            subprocess.run([sys.executable,'-B',str(ROOT/'Capri/tools/evaluate_ab_checkpoint.py'),'--seed','48','--epoch',str(e),'--model',str(n),'--model-only','--device','cuda'],cwd=str(ROOT),check=True)
    # All selected predictions must be frozen before opening scoring labels.
    frozen=[]
    for choice in choices:
        n,tag,e=choice['model'],choice['dataset'],choice['epoch']
        out=ROOT/f'model{n}/results/darkbay/{tag}/interim_cuda_epoch{e:02d}_seed48'
        receipt=core.read(out/'prediction_freeze.json')
        assert receipt['checkpoint_sha256']==choice['checkpoint_sha256']
        assert receipt['predictions_sha256']==core.sha(out/'predictions.json')
        rows=core.read(out/'predictions.json');assert len(rows)==900 and len({r['opaque_id'] for r in rows})==900
        frozen.append((choice,out,rows,receipt))
    labels=core.read(ROOT/'evaluation_data/darkbay/scoring_labels/scoring_labels.json')
    results=[]
    for choice,out,rows,receipt in frozen:
        assert {r['opaque_id'] for r in rows}==set(labels)
        correct=sum(r['prediction']==labels[r['opaque_id']] for r in rows)
        summary=core.read(out/'summary.json');assert summary['correct']==correct and summary['total']==900
        result=dict(**choice,real_correct=correct,real_total=900,real_accuracy_percent=100*correct/900,real_predictions= (out/'predictions.json').relative_to(ROOT).as_posix(),real_predictions_sha256=receipt['predictions_sha256'],real_summary_sha256=core.sha(out/'summary.json'))
        results.append(result)
    report=dict(created_at=core.now(),selection_freeze_sha256=core.sha(freeze),results=results,verification='All six checkpoint identities, prediction hashes,900-image identity sets and exact-match counts verified',limitation='Historical repeatedly inspected real test; retrospective validation design; Model3 epoch selection provisional through17',training_changed=False)
    core.write(OUT/'results.json',report)
    lines=['# Real DarkBay accuracy at synthetic-validation-selected epochs','','Selection rule: highest exact-match accuracy on the frozen1000 synthetic images per reader; earliest epoch on ties. Selected epochs were frozen before this evaluation.','', '| Reader | Selected epoch | Synthetic validation | Real DarkBay | Correct /900 |','|---|---|---|---|---|']
    for r in results:lines.append(f"| Model{r['model']}-{r['dataset']} | {r['epoch']} | {r['validation_accuracy_percent']:.1f}% | {r['real_accuracy_percent']:.2f}% | {r['real_correct']}/900 |")
    lines.extend(['','Model1 eligible epochs2-20 (epoch1 unavailable); Model2 epochs1-20; Model3 epochs1-17 captured at the validation sweep. Model3 selections may change when its remaining epochs are evaluated.','', 'Five existing frozen real evaluations were reused. Missing Model3-B epoch14 was evaluated using the existing paired diagnostic; its companion Model3-A epoch14 result is auxiliary and does not replace the selected Model3-A epoch16. All selected prediction hashes, checkpoint identities and exact900-image sets were verified, then counts were recomputed against scoring labels.','', 'The real test was previously inspected and this synthetic validation design is retrospective. This is a separate selection experiment; original fixed epoch20 primary results and ongoing training are unchanged.'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
