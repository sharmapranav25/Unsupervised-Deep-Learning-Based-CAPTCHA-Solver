"""Run version-2 regime analysis; freeze choices before optional label evaluation."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .loader import load,write,sha,DEFAULT_ROOT
from .run import clean
from .regimes import CONFIG_V2,k_cap,item_agreements,fit_mask,select_mask,degeneracy,partner_report,joint_coupling,bootstrap_factors,neighbour_diagnostics,combine,decomposition
from .baselines import mode_scores

def plot_diagnostics(out,name,kernel,variant,epochs,A,fits,k,boot,near,profiles):
    prefix=f'{name}_{kernel}_{variant}';f=fits[k]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,data,title in [(axes[0],A,'Agreement'),(axes[1],A-f['prediction'],'Signed residual')]:
        im=ax.imshow(data,origin='lower',extent=[.5,len(A)+.5,.5,len(A)+.5],cmap='viridis' if title=='Agreement' else 'coolwarm');fig.colorbar(im,ax=ax);ax.set(title=title,xlabel='Epoch',ylabel='Epoch')
    fig.suptitle(prefix);fig.tight_layout();fig.savefig(out/f'{prefix}_matrices.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for kk in fits:
        rr=[r for r in profiles if r['k']==kk];ax.plot([r['distance'] for r in rr],[r['mean_absolute_residual'] for r in rr],label=f'k={kk}')
    ax.set(xlabel='Epoch distance',ylabel='Mean absolute residual',title=prefix);ax.legend();fig.tight_layout();fig.savefig(out/f'{prefix}_residual_distance.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    axes[0].plot(epochs,f['scores'],'o-',label='Distant factor (not accuracy)');axes[0].fill_between(epochs,boot['low'],boot['high'],alpha=.2,label='Conditional image-bootstrap 95% CI');axes[0].legend()
    axes[1].plot(epochs,[np.nan if v is None else v for v in near['mean_kappa']],'o-',label='Neighbour kappa');axes[1].legend();axes[1].set_xlabel('Epoch');fig.suptitle(prefix);fig.tight_layout();fig.savefig(out/f'{prefix}_uncertainty_coupling.png',dpi=140);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--kernel',choices=['all','exact','char','edit'],default='all');ap.add_argument('--evaluate',action='store_true');args=ap.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    from .simulate import validate,generate
    gate=validate();new_gate=[]
    for case in [1,2,3]:
        tab,truth,_=generate(case);A=item_agreements(tab,'exact').mean(2);k,passed,fits,_,_=select_mask(A,tab.epochs);j=joint_coupling(A,tab.epochs,fits[0]['scores'])
        new_gate.append(dict(case=case,cap=k_cap(len(tab.epochs)),chosen_k=k,criterion_passed=passed,capped_pick=int(tab.epochs[np.argmax(fits[k]['scores'])]),joint_pick=j['selected_epoch'],true_best=int(tab.epochs[np.argmax(truth)]),joint_converged=j['converged']))
    write(out/'simulation_gate.json',dict(original_required_tests=gate,v2_outcomes=new_gate,note='Simulation outcomes reported without tuning. Original case2 used k5, disallowed by v2.'))
    source=Path(__file__).parent;code={p.name:sha(p) for p in source.glob('*.py')};snap=out/'source_snapshot';snap.mkdir()
    for p in source.glob('*.py'):(snap/p.name).write_bytes(p.read_bytes())
    write(out/'pre_real_protocol.json',dict(config=CONFIG_V2,code_hashes=code,created_at=datetime.now(timezone.utc).isoformat(),burnin='Flags from all-partners k0 factor <0.01, or >50% empty/identical strings across images; fixed for bootstrap. Flag all degenerate epochs, report whether early. Excluded epochs inferred afterward, never steer active scores.',bootstrap='2000 paired image resamples, refit scores; original k/exclusions frozen. Conditional percentile CIs, not selection-adjusted or accuracy CIs.',mode_decomposition='Regional voter-pool modes, not additive decomposition of nonlinear global mode.',joint_fit='Four deterministic starts; log tau bounded 0.05..100*n for numerical stability; boundary solutions reported.',posthoc='Version2 requested after earlier results/labels were inspected. Retrospective evaluation, no thresholds tuned on the 251 labels.'))
    tables={name:load(name,args.root,20) for name in ['Model1-A','Model1-B']};write(out/'prediction_sources.json',{name:list(t.sources) for name,t in tables.items()})
    kernels=['exact','char','edit'] if args.kernel=='all' else [args.kernel];summaries=[];details={};partners=[];epoch_rows=[];decomp_rows=[];profiles_all=[]
    for name,table in tables.items():
        print('Processing',name,flush=True)
        for kernel in kernels:
            items=item_agreements(table,kernel);A=items.mean(2);np.save(out/f'{name}_{kernel}_agreement.npy',A)
            initial=fit_mask(A,table.epochs,0);flags=degeneracy(table,initial['scores']);excluded=[i for i,r in enumerate(flags) if r['flagged']]
            for variant,exclude in [('all_partners',[]),('without_degenerate_partners',excluded)]:
                print(' ',kernel,variant,'bootstrap 2000',flush=True)
                k,passed,fits,criteria,profiles=select_mask(A,table.epochs,exclude);f=fits[k]
                for kk,ff in fits.items():partners.extend(dict(run=name,kernel=kernel,variant=variant,**r) for r in partner_report(table.epochs,ff,kk))
                profiles_all.extend(dict(run=name,kernel=kernel,variant=variant,**r) for r in profiles)
                boot=bootstrap_factors(items,table.epochs,k,f);np.save(out/f'{name}_{kernel}_{variant}_bootstrap.npy',boot['samples'])
                near=neighbour_diagnostics(A,table.epochs,k,f);selection=combine(table.epochs,k,f,boot,near)
                joint=joint_coupling(A,table.epochs,fits[0]['scores'],exclude);decomp=decomposition(table,A,table.epochs,k,f)
                decomp_rows.extend(dict(run=name,input_kernel=kernel,variant=variant,**r) for r in decomp['rows'])
                for i,e in enumerate(table.epochs):epoch_rows.append(dict(run=name,kernel=kernel,variant=variant,epoch=int(e),k=k,distant_factor=float(f['scores'][i]),ci_low=float(boot['low'][i]),ci_high=float(boot['high'][i]),se=float(boot['se'][i]),distant_partners=int(f['partners'][i].sum()),mean_neighbour_kappa=near['mean_kappa'][i],plateau=float(selection['plateau_scores'][i]),candidate=int(e) in selection['candidates'],joint_factor=float(joint['scores'][i])))
                row=dict(run=name,kernel=kernel,variant=variant,k_max=k_cap(len(A)),chosen_k=k,mask_criterion_passed=passed,distant_pick=selection['distant_pick'],combined_pick=selection['combined_pick'],joint_pick=joint['selected_epoch'],candidates=selection['candidates'],excluded_epochs=table.epochs[exclude].tolist(),joint_kappa0=joint['kappa0'],joint_tau=joint['tau'],joint_tau_at_bound=joint['tau_at_numerical_bound'])
                summaries.append(row)
                details[f'{name}_{kernel}_{variant}']=dict(summary=row,flags=flags,criteria=criteria,all_k_picks=[dict(k=kk,selected_epoch=int(table.epochs[np.argmax(ff['scores'])])) for kk,ff in fits.items()],scores=f['scores'],bootstrap=dict(low=boot['low'],high=boot['high'],se=boot['se'],max_iterations=boot['max_iterations']),neighbour=near,selection=selection,joint=joint,decomposition=decomp,original_modes={ker:int(table.epochs[np.argmax(mode_scores(table,ker))]) for ker in ['exact','char']})
                plot_diagnostics(out,name,kernel,variant,table.epochs,A,fits,k,boot,near,profiles)
                print('  picks:',selection['distant_pick'],selection['combined_pick'],joint['selected_epoch'],'k=',k,'passed=',passed,flush=True)
    write(out/'regime_details.json',clean(details));write(out/'rankings.json',clean(summaries))
    for name,rows in [('summary_rankings',summaries),('partner_diagnostics',partners),('epoch_diagnostics',epoch_rows),('decomposition',decomp_rows),('residual_by_distance',profiles_all)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    write(out/'selection_freeze.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),labels_read=False,code_hashes=code,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    if args.evaluate:
        from .evaluate import evaluate_regimes
        evaluate_regimes(args.root,out,tables)
    print('Completed:',out.resolve(),flush=True)
if __name__=='__main__':main()
