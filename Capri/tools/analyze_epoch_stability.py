"""Descriptive epoch variation on fixed common windows; no selection or training."""
from pathlib import Path
import json, hashlib, statistics
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'handoff/darkbay_curriculum_20261113/epoch_stability_20261113'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def measure(values):
    changes=[b-a for a,b in zip(values,values[1:])]
    return dict(epochs=len(values),transitions=len(changes),mean_accuracy_percent=statistics.mean(values),accuracy_population_sd_pp=statistics.pstdev(values),mean_absolute_change_pp=statistics.mean(abs(x) for x in changes),change_population_sd_pp=statistics.pstdev(changes),rms_change_pp=(statistics.mean(x*x for x in changes))**.5,maximum_absolute_change_pp=max(abs(x) for x in changes),signed_changes_pp=changes)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    evidence=[];series={}
    for n in (1,2,3):
        for tag in ('A','B'):
            key=f'Model{n}-{tag}';points={}
            for e in range(2,20):
                out=ROOT/f'model{n}/results/darkbay/{tag}/interim_cuda_epoch{e:02d}_seed48'
                s=read(out/'summary.json');fr=read(out/'prediction_freeze.json');pred=read(out/'predictions.json')
                assert len(pred)==900 and len({p['opaque_id'] for p in pred})==900
                assert sha(out/'predictions.json')==fr['predictions_sha256']
                assert (s['model'],s['dataset'],s['epoch'],s['total'])==(n,tag,e,900)
                assert abs(s['accuracy_percent']-s['correct']/9)<1e-9
                points[e]=s['correct']/9
                evidence.append(dict(path=(out/'summary.json').relative_to(ROOT).as_posix(),sha256=sha(out/'summary.json'),predictions_sha256=fr['predictions_sha256'],checkpoint_sha256=fr['checkpoint_sha256']))
            series[key]=points
    results=[]
    for lo in (2,6):
        for key,points in series.items():
            epochs=list(range(lo,20));values=[points[e] for e in epochs]
            results.append(dict(scope='real_darkbay',reader=key,window=f'{lo}-19',epoch_ids=epochs,accuracy_percent=values,**measure(values)))
    validation=ROOT/'handoff/darkbay_curriculum_20261113/controller4_validation_sweep/results.json'
    v=read(validation)
    for lo in (2,6):
        for n in (1,2,3):
            for tag in ('A','B'):
                epochs=list(range(lo,18));lookup={r['epoch']:r['accuracy_percent'] for r in v['results'] if r['model']==n and r['dataset']==tag}
                values=[lookup[e] for e in epochs]
                results.append(dict(scope='synthetic_validation',reader=f'Model{n}-{tag}',window=f'{lo}-17',epoch_ids=epochs,accuracy_percent=values,**measure(values)))
    report=dict(results=results,evidence=evidence,validation_source_sha256=sha(validation),analysis_source_sha256=sha(Path(__file__)),definitions={'accuracy_population_sd_pp':'Population SD of observed epoch accuracies, ddof0; includes trends','mean_absolute_change_pp':'Mean absolute difference between consecutive epoch accuracies; lower means smaller jumps','change_population_sd_pp':'Population SD of signed consecutive differences, ddof0'},protocol='Same contiguous epoch window for all six readers within each comparison. Real2-19 and6-19; synthetic2-17 and6-17. No interpolation, smoothing, A/B averaging, checkpoint selection or training changes.',limitation='Descriptive variation along one training trajectory per dataset, not independent-run variance or statistical proof of curriculum effects. Real results are repeatedly inspected historical diagnostics.')
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Epoch-to-epoch accuracy variation','',report['protocol'],'',report['limitation'],'']
    for scope,windows in [('real_darkbay',['2-19','6-19']),('synthetic_validation',['2-17','6-17'])]:
        for window in windows:
            lines.extend([f'## {scope}: epochs{window}','','All deviations are percentage points. Lower means less observed variation.','','| Reader | Accuracy SD | Mean absolute epoch change | SD of signed changes | Largest change |','|---|---:|---:|---:|---:|'])
            for r in results:
                if r['scope']==scope and r['window']==window:
                    lines.append(f"| {r['reader']} | {r['accuracy_population_sd_pp']:.2f} | {r['mean_absolute_change_pp']:.2f} | {r['change_population_sd_pp']:.2f} | {r['maximum_absolute_change_pp']:.2f} |")
            lines.append('')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__':main()
