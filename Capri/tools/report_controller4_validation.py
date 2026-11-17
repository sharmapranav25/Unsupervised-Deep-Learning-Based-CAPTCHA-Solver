"""Render the completed frozen validation sweep and descriptive real comparison."""
from pathlib import Path
import json, math, os
ROOT=Path(__file__).resolve().parents[2]
os.environ['MPLCONFIGDIR']=str(ROOT/'runtime/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
STUDY=ROOT/'handoff/darkbay_curriculum_20261113/controller4_validation_sweep'
FIG=ROOT/'thesis/figures/darkbay/controller4_validation'
def main():
    result=json.loads((STUDY/'results.json').read_text())
    verification=json.loads((STUDY/'verification.json').read_text());assert verification['passed']
    comparison=json.loads((STUDY/'real_comparison.json').read_text())
    colors={1:'#159447',2:'#2463EB',3:'#DC3545'}
    upper=max(20,5*math.ceil((max(r['accuracy_percent'] for r in result['results'])+1)/5))
    for tag in ('A','B'):
        fig,ax=plt.subplots(figsize=(11,6.5));fig.subplots_adjust(left=.09,right=.97,bottom=.17,top=.78)
        fig.text(.09,.94,f'DarkBay - Dataset {tag}',fontsize=21,fontweight='bold')
        fig.text(.09,.89,'Accuracy on 1,000 held-out SynthTIGER images',fontsize=13,color='#526075')
        for n in (1,2,3):
            rows=sorted([r for r in result['results'] if r['model']==n and r['dataset']==tag],key=lambda r:r['epoch'])
            ax.plot([r['epoch'] for r in rows],[r['accuracy_percent'] for r in rows],color=colors[n],marker='o',markersize=4.5,linewidth=2,label=f'Model {n}')
        ax.set(xlim=(.7,20.3),ylim=(0,upper),xlabel='Epoch',ylabel='Exact-match validation accuracy (%)')
        ax.set_xticks(range(1,21));ax.set_yticks(range(0,upper+1,2));ax.grid(axis='y',alpha=.2)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=3)
        for side in ('top','right'):ax.spines[side].set_visible(False)
        fig.text(.09,.07,f'Y-axis: 0-{upper}%. Raw measured accuracy; no smoothing or A/B averaging.',fontsize=10,color='#526075')
        fig.text(.09,.035,'Model 1 epoch 1 unavailable. Model 3 includes only checkpoints committed at sweep start.',fontsize=9.5,color='#526075')
        for ext in ('png','svg'):fig.savefig(FIG/f'dataset_{tag.lower()}_detail.{ext}',dpi=180)
        plt.close(fig)
    lines=['# Controller4 validation sweep results','',f"Verified {result['evaluated']} checkpoint evaluations, each on the same1000 images.",'','| Reader | Highest validation accuracy | Epoch(s) | Latest evaluated | Latest accuracy | Late-epoch real/validation Spearman rho |','|---|---|---|---|---|---|']
    summary=[]
    for b in result['best_validation']:
        n,t=b['model'],b['dataset'];r=max([v for v in result['results'] if v['model']==n and v['dataset']==t],key=lambda v:v['epoch'])
        c=next(c for c in comparison['comparisons'] if c['model']==n and c['dataset']==t)
        rho=c['descriptive_correlations'].get('epochs6_onward',{}).get('spearman_rho')
        valbestreal=next((p for p in c['pairs'] if p['epoch']==b['epoch']),None)
        summary.append(dict(model=n,dataset=t,best=b,latest=r,late_epoch_spearman=rho,real_at_earliest_validation_best=valbestreal))
        lines.append(f"| M{n}-{t} | {b['accuracy_percent']:.1f}% ({b['correct']}/1000) | {', '.join(map(str,b['tied_epochs']))} | {r['epoch']} | {r['accuracy_percent']:.1f}% | {rho:.3f} |")
    lines.extend(['','The primary metric is case-sensitive whole-string exact match with standard saved free autoregressive decoding plus refinement. Every prediction identity, hash and count was verified. Model1 epoch1 is unavailable, not imputed. Model3 uncommitted epochs were excluded.','', 'Correlations are descriptive, over epochs6 onward with already-saved real diagnostics. Missing real diagnostics are omitted. These serial epoch measurements are not independent trials. The real test was previously inspected and this validation set was designed retrospectively. These associations do not establish prospective checkpoint-selection reliability.','', 'Full per-epoch table: epoch_accuracy.md. Frozen protocol and inputs: protocol.json. Predictions and summaries: modelN/results/darkbay/{A,B}/controller4_validation/epoch_NN. Charts: thesis/figures/darkbay/controller4_validation (overview0-100% and detail0-20% PNG/SVG).','', 'No training, curriculum, generator settings or primary fixed-epoch20 selection changed. One early empty decoder output caused an unused confidence-mean warning; exact-match scoring uses strings only and all1000 predictions were retained.'])
    (STUDY/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (STUDY/'compact_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
