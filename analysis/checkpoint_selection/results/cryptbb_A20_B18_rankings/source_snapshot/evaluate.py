"""The ONLY module allowed to open real scoring labels. Requires a verified freeze."""
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,kendalltau,binomtest
from .loader import read,write,sha
from .estimator import CONFIG

def paired_test(selected,best,seed=4817):
    delta=selected.astype(int)-best.astype(int)
    win=int(np.sum(delta==1));loss=int(np.sum(delta==-1))
    p=binomtest(win,win+loss,.5).pvalue if win+loss else 1.
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,CONFIG['bootstrap_samples'],500):
        ix=rng.integers(0,len(delta),(min(500,CONFIG['bootstrap_samples']-start),len(delta)))
        samples.extend(delta[ix].mean(axis=1))
    lo,hi=np.quantile(samples,[.025,.975])
    return dict(selected_only_correct=win,best_only_correct=loss,mcnemar_p=float(p),difference=float(delta.mean()),bootstrap_ci_low=float(lo),bootstrap_ci_high=float(hi),bootstrap_samples=CONFIG['bootstrap_samples'])

def calibration(table,labels,kernel):
    rows=[]
    for i,key in enumerate(table.ids):
        strings=table.predictions[:,i];truth=labels[key]
        values=[(None,strings,truth)] if kernel=='exact' else [(j,[s[j] if j<len(s) else None for s in strings],truth[j] if j<len(truth) else None) for j in range(max(map(len,strings)))]
        for position,v,target in values:
            counts=Counter(v);freq=max(counts.values());modes=[s for s,c in counts.items() if c==freq]
            rows.append(dict(opaque_id=key,position=position,vote_share=freq/len(strings),tie_count=len(modes),correct_credit=(1/len(modes) if target in modes else 0),unique_mode=len(modes)==1))
    df=pd.DataFrame(rows);df['bin']=pd.cut(df.vote_share,[0,.2,.4,.6,.8,1.000001],include_lowest=True,right=True).astype(str)
    aggregate=df.groupby(['bin','unique_mode'],observed=True).agg(n=('correct_credit','size'),mean_vote_share=('vote_share','mean'),p_mode_correct=('correct_credit','mean')).reset_index()
    aggregate['kernel']=kernel
    return rows,aggregate

def evaluate(root,out,tables):
    out=Path(out);freeze=read(out/'selection_freeze.json')
    for name,digest in freeze['files'].items():assert sha(out/name)==digest, 'Frozen selection artifact changed'
    for name,digest in freeze['code_hashes'].items():assert sha(Path(__file__).parent/name)==digest,'Frozen code changed'
    assert freeze['labels_read'] is False
    # Only after all frozen decisions are checked do real labels enter memory.
    labelpath=Path(root)/'evaluation_data/model4_reservation_20261114/scoring_only/cryptbb/evaluation_pool_labels.json'
    labels=read(labelpath)
    for table in tables.values():assert set(table.ids)==set(labels)
    score_rows=read(out/'scores.json');metric_rows=[];cache={};truth_curves=[];attractors=[];calibrations=[];checks=[]
    for run,table in tables.items():
        truth=np.array([labels[i] for i in table.ids],object);correct=table.predictions==truth[None,:];accuracy=correct.mean(axis=1)
        best=int(np.argmax(accuracy));best_epochs=table.epochs[accuracy==accuracy.max()].tolist()
        for e,a in zip(table.epochs,accuracy):truth_curves.append(dict(run=run,epoch=int(e),correct=int(round(a*len(truth))),total=len(truth),accuracy=float(a)))
        for row in [r for r in score_rows if r['run']==run]:
            scores=np.array(row['scores']);t=int(np.argmax(scores));order=np.lexsort((table.epochs,-scores))
            assert int(table.epochs[t])==row['selected_epoch']
            key=(run,t,best)
            if key not in cache:cache[key]=paired_test(correct[t],correct[best])
            constant=np.ptp(scores)==0
            rec={k:v for k,v in row.items() if k!='scores'}
            rec.update(true_selected_accuracy=float(accuracy[t]),selected_correct=int(correct[t].sum()),best_accuracy=float(accuracy[best]),best_epoch=int(table.epochs[best]),regret=float(accuracy[best]-accuracy[t]),spearman=float(spearmanr(scores,accuracy).statistic) if not constant else None,kendall=float(kendalltau(scores,accuracy).statistic) if not constant else None,top3_contains_best=bool(any(int(table.epochs[x]) in best_epochs for x in order[:3])),mae=float(np.mean(np.abs(scores-accuracy))) if row['method'].startswith('rank1') else None,**cache[key])
            metric_rows.append(rec)
            if run=='Model1-B' and row['method'].startswith('rank1'):
                ix=int(np.flatnonzero(table.epochs==13)[0]);checks.append(dict(kernel=row['kernel'],k=row['k'],method=row['method'],estimate=float(scores[ix]),observed_accuracy=float(accuracy[ix]),absolute_error=float(abs(scores[ix]-accuracy[ix])),within_predeclared_0_03=bool(abs(scores[ix]-accuracy[ix])<=.03)))
        for i,key in enumerate(table.ids):
            counts=Counter(table.predictions[:,i]);top=max(counts.values())
            for s,count in counts.items():
                if count==top and count/len(table.epochs)>=.5 and s!=labels[key]:
                    attractors.append(dict(run=run,opaque_id=key,wrong_string=s,true_label=labels[key],count=count,epochs=len(table.epochs),vote_share=count/len(table.epochs)))
        for kernel in ['exact','char']:
            details,agg=calibration(table,labels,kernel);agg.insert(0,'run',run);calibrations.append(agg)
            write(out/f'{run}_{kernel}_mode_calibration_items.json',details)
        if run=='Model1-A':
            i17=int(np.flatnonzero(table.epochs==17)[0]);i19=int(np.flatnonzero(table.epochs==19)[0])
            check=dict(comparison='Model1-A epoch19 minus epoch17',epoch17_correct=int(correct[i17].sum()),epoch19_correct=int(correct[i19].sum()),**paired_test(correct[i19],correct[i17]))
            write(out/'A17_vs_A19.json',check)
    pd.DataFrame(metric_rows).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(truth_curves).to_csv(out/'true_accuracy.csv',index=False)
    pd.concat(calibrations).to_csv(out/'mode_calibration.csv',index=False)
    write(out/'attractors.json',attractors);write(out/'B_epoch13_absolute_accuracy_check.json',checks)
    write(out/'evaluation_receipt.json',dict(selection_freeze_sha256=sha(out/'selection_freeze.json'),labels_sha256=sha(labelpath),rows=len(metric_rows),bootstrap_seed=4817,bootstrap_samples=10000,accuracy_used_for_selection=False,uncertainty='Conditional paired comparisons to an oracle chosen on these same images; not selection-adjusted or multiplicity-corrected.'))
    return pd.DataFrame(metric_rows),pd.DataFrame(truth_curves)
