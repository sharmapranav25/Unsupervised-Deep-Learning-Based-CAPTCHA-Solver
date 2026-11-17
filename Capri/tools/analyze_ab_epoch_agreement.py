"""Retrospective label-free A/B epoch selection, frozen before label scoring."""
from pathlib import Path
import argparse,collections,hashlib,json,datetime,math
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'handoff/darkbay_curriculum_20261113'
OUT=STUDY/'ab_epoch_agreement_20261113'
FIG=ROOT/'thesis/figures/darkbay/ab_epoch_agreement_20261113'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
def prediction_dir(n,t,e):
    p=ROOT/f'model{n}/results/darkbay/{t}/interim_cuda_epoch{e:02d}_seed48'
    if not (p/'prediction_freeze.json').exists() and e==20:p=ROOT/f'model{n}/results/darkbay/{t}/evaluation_epoch20'
    assert (p/'prediction_freeze.json').exists(),str(p)
    return p

def freeze():
    if (OUT/'selection_freeze.json').exists():raise FileExistsError('Selection already frozen; preserve it')
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_at=now(),question='Can same-epoch A/B full-string agreement select a useful real-transfer checkpoint without labels?',eligible_epochs={'1':list(range(2,21)),'2':list(range(1,21)),'3':list(range(1,21))},primary='Maximize raw exact-string A/B agreement over all900 images, earliest epoch breaks ties',secondary_predeclared='Maximize exact-string agreement where the shared output contains a non-whitespace character; denominator remains all900 images; earliest tie',no_confidence_or_length_filter=True,same_epoch_pair=True,selection_reads_real_labels=False,source=Path(__file__).relative_to(ROOT).as_posix(),source_sha256=sha(Path(__file__)),existing_study_selection_unchanged=True,limitations=['Retrospective hypothesis after real accuracy curves had been inspected','Same unlabeled900 real images used for agreement then scoring: transductive diagnostic, not an untouched test','Model1 epoch1 unavailable; Model2/3 epoch1 included and empty-output agreement explicitly reported','A/B share architecture and generator recipes, so errors may be correlated','Same epoch selected jointly for A/B, unlike independent synthetic-validation choices'])
    write(OUT/'protocol.json',protocol)
    validation=read(STUDY/'controller4_validation_sweep_final/results.json')['results']
    hashes={(r['model'],r['dataset'],r['epoch']):r['checkpoint_sha256'] for r in validation}
    evidence=[];rows=[];all_ids=None
    for n in [1,2,3]:
        for e in protocol['eligible_epochs'][str(n)]:
            pair={}
            for t in 'AB':
                p=prediction_dir(n,t,e);fr=read(p/'prediction_freeze.json');data=read(p/'predictions.json')
                assert sha(p/'predictions.json')==fr['predictions_sha256'] and fr['checkpoint_sha256']==hashes[(n,t,e)]
                assert len(data)==900 and len({r['opaque_id'] for r in data})==900
                pred={r['opaque_id']:r['prediction'] for r in data};assert all(isinstance(v,str) for v in pred.values())
                if all_ids is None:all_ids=set(pred)
                assert set(pred)==all_ids;pair[t]=pred
                c=collections.Counter(pred.values())
                evidence.append(dict(model=n,dataset=t,epoch=e,path=(p/'predictions.json').relative_to(ROOT).as_posix(),predictions_sha256=fr['predictions_sha256'],checkpoint_sha256=fr['checkpoint_sha256'],unique_strings=len(c),largest_same_output_count=max(c.values()),empty_outputs=sum(not v.strip() for v in pred.values())))
            a,b=pair['A'],pair['B'];agree=[k for k in sorted(a) if a[k]==b[k]];nonempty=[k for k in agree if a[k].strip()]
            rows.append(dict(model=n,epoch=e,total=900,raw_agree=len(agree),raw_agreement_percent=len(agree)/9,nonempty_agree=len(nonempty),nonempty_agreement_percent=len(nonempty)/9,shared_empty=len(agree)-len(nonempty)))
    choices=[]
    for n in [1,2,3]:
        eligible=[r for r in rows if r['model']==n]
        for name,field in [('raw','raw_agree'),('nonempty','nonempty_agree')]:
            best=max(eligible,key=lambda r:(r[field],-r['epoch']))
            choices.append(dict(model=n,rule=name,epoch=best['epoch'],agreement_count=best[field],agreement_percent=best[field]/9,shared_empty=best['shared_empty']))
    write(OUT/'prediction_inventory.json',evidence);write(OUT/'agreement_curves.json',rows)
    freeze=dict(created_at=now(),protocol_sha256=sha(OUT/'protocol.json'),inventory_sha256=sha(OUT/'prediction_inventory.json'),curves_sha256=sha(OUT/'agreement_curves.json'),choices=choices,real_labels_read=False)
    write(OUT/'selection_freeze.json',freeze);print(json.dumps(freeze,indent=2))

def score():
    import numpy as np
    f=read(OUT/'selection_freeze.json');assert not f['real_labels_read']
    assert sha(OUT/'protocol.json')==f['protocol_sha256'] and sha(OUT/'prediction_inventory.json')==f['inventory_sha256'] and sha(OUT/'agreement_curves.json')==f['curves_sha256']
    evidence=read(OUT/'prediction_inventory.json');predictions={}
    for r in evidence:
        p=ROOT/r['path'];assert sha(p)==r['predictions_sha256'];predictions[(r['model'],r['dataset'],r['epoch'])]={x['opaque_id']:x['prediction'] for x in read(p)}
    # Labels are first accessed here, after persisted selection choices are verified.
    labelpath=ROOT/'evaluation_data/darkbay/scoring_labels/scoring_labels.json';labels=read(labelpath);assert len(labels)==900
    rows=read(OUT/'agreement_curves.json');scored=[]
    for r in rows:
        n,e=r['model'],r['epoch'];a=predictions[(n,'A',e)];b=predictions[(n,'B',e)];assert set(a)==set(b)==set(labels)
        agree={k for k in labels if a[k]==b[k]};correct_shared=sum(a[k]==labels[k] for k in agree)
        ac=sum(a[k]==labels[k] for k in labels);bc=sum(b[k]==labels[k] for k in labels)
        assert len(agree)==r['raw_agree'] and sum(bool(a[k].strip()) for k in agree)==r['nonempty_agree']
        scored.append(dict(**r,A_correct=ac,B_correct=bc,A_accuracy_percent=ac/9,B_accuracy_percent=bc/9,mean_accuracy_percent=(ac+bc)/18,shared_correct=correct_shared,shared_wrong=len(agree)-correct_shared,agreement_precision_percent=100*correct_shared/len(agree) if agree else None))
    baseline=read(STUDY/'controller4_validation_selected_real_final/results.json')['results'];results=[]
    def rank(x):return [sum(y<a for y in x)+(sum(y==a for y in x)+1)/2 for a in x]
    associations=[]
    for n in [1,2,3]:
        base=[r for r in baseline if r['model']==n];base_mean=sum(r['real_correct'] for r in base)/18
        for choice in [x for x in f['choices'] if x['model']==n]:
            r=next(x for x in scored if x['model']==n and x['epoch']==choice['epoch'])
            results.append(dict(**choice,A_correct=r['A_correct'],B_correct=r['B_correct'],A_accuracy_percent=r['A_accuracy_percent'],B_accuracy_percent=r['B_accuracy_percent'],mean_accuracy_percent=r['mean_accuracy_percent'],shared_correct=r['shared_correct'],shared_wrong=r['shared_wrong'],synthetic_selected_mean_percent=base_mean,change_vs_synthetic_selection_pp=r['mean_accuracy_percent']-base_mean))
        for lo in [2,6]:
            rr=[x for x in scored if x['model']==n and x['epoch']>=lo]
            associations.append(dict(model=n,window=f'{lo}-20',spearman_agreement_vs_mean_real=float(np.corrcoef(rank([x['nonempty_agree'] for x in rr]),rank([x['mean_accuracy_percent'] for x in rr]))[0,1])))
    write(OUT/'scored_curves.json',scored)
    result=dict(created_at=now(),selection_freeze_sha256=sha(OUT/'selection_freeze.json'),scoring_labels_sha256=sha(labelpath),selected_results=results,associations=associations,readers_verified=len(evidence),images_per_reader=900,existing_selection_changed=False)
    write(OUT/'results.json',result)
    FIG.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
    for n,ax in zip([1,2,3],axes):
        rr=[r for r in scored if r['model']==n];epochs=[r['epoch'] for r in rr]
        ax.plot(epochs,[r['nonempty_agreement_percent'] for r in rr],label='A/B nonempty agreement',color='#7647b6',marker='.',lw=1.5)
        ax.plot(epochs,[r['mean_accuracy_percent'] for r in rr],label='Mean real accuracy (scoring only)',color='#117a65',marker='.',lw=1.5)
        selected=next(x for x in results if x['model']==n and x['rule']=='nonempty');ax.axvline(selected['epoch'],color='#7647b6',ls='--',alpha=.6)
        ax.set_title(f'Model {n}: selected epoch {selected["epoch"]}');ax.set_xlabel('Epoch');ax.set_ylim(0,100);ax.set_xticks([2,6,10,14,18,20]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Percent of 900 real images');handles,leg=axes[0].get_legend_handles_labels();fig.legend(handles,leg,loc='lower center',ncol=2,bbox_to_anchor=(.5,.02));fig.suptitle('Retrospective label-free A/B agreement selection');fig.tight_layout(rect=[0,.1,1,.93]);fig.savefig(FIG/'agreement_vs_real.png',dpi=160);fig.savefig(FIG/'agreement_vs_real.svg');plt.close(fig)
    lines=['# A/B agreement checkpoint-selection diagnostic','','Selections were frozen using predictions only, then real labels scored them. Exact whole-string agreement, same-epoch A/B pairs, earliest tie. Nonempty agreement is a predeclared secondary rule; its denominator is all900 images. No confidence or family-length threshold.','','| Model | Rule | Epoch | Agreement | A real | B real | Mean real | Change vs synthetic selection |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f'| {r["model"]} | {r["rule"]} | {r["epoch"]} | {r["agreement_percent"]:.2f}% | {r["A_accuracy_percent"]:.2f}% | {r["B_accuracy_percent"]:.2f}% | {r["mean_accuracy_percent"]:.2f}% | {r["change_vs_synthetic_selection_pp"]:+.2f}pp |')
    lines+=['','## Association after initial learning','','| Model | Epoch window | Spearman agreement vs mean real accuracy |','|---|---|---:|']
    for r in associations:lines.append(f'| {r["model"]} | {r["window"]} | {r["spearman_agreement_vs_mean_real"]:.3f} |')
    lines+=['','## Limits','','Retrospective hypothesis developed after real curves had been viewed. The same900 unlabeled real images determine agreement then receive label scoring: transductive diagnostic, not an untouched external test. A/B share architecture and recipes; agreement can contain correlated errors. No rule was retuned after scoring. Model1 epoch1 is unavailable. Same-epoch joint selection constrains both runs; the existing synthetic-validation policy selects runs independently. Original study selections remain unchanged.','',f'Checked {len(evidence)} prediction files, hashes, checkpoint identities against the verified sweep, and900 unique matched image IDs each.','', 'Saved: protocol.json, prediction_inventory.json, agreement_curves.json, selection_freeze.json, scored_curves.json, results.json. Plot: thesis/figures/darkbay/ab_epoch_agreement_20261113/ agreement_vs_real.png and .svg.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','score']);a=ap.parse_args();globals()[a.stage]()
