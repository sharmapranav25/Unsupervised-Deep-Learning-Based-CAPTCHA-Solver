"""CLI separates prediction-only selection/freeze from opt-in label evaluation."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from .loader import load,write,read,sha,DEFAULT_ROOT
from .agreement import matrix
from .baselines import mode_scores,pairwise_scores
from .estimator import rank1,rank2,choose_k,residual_profile,CONFIG

def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',choices=['Model1-A','Model1-B','all'],default='Model1-A')
    ap.add_argument('--kernel',choices=['exact','char','edit','all'],default='exact')
    ap.add_argument('--k',choices=['auto','0','1','2','3','4','5'],default='auto')
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--a-through',type=int,default=20)
    ap.add_argument('--b-through',type=int,default=16)
    ap.add_argument('--evaluate',action='store_true',help='After freezing selection, score using real labels')
    args=ap.parse_args();out=args.out or Path('results')/f'{args.run}_{args.kernel}_k{args.k}'
    out.mkdir(parents=True,exist_ok=False)
    # Mandatory simulation gate before loading any real prediction tables.
    from .simulate import validate
    gate=validate();write(out/'simulation_gate.json',dict(passed=True,records=gate,config=CONFIG))
    code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'):(snapshot/p.name).write_bytes(p.read_bytes())
    write(out/'pre_real_protocol.json',dict(created_at=datetime.now(timezone.utc).isoformat(),config=CONFIG,code_hashes=code,simulation_gate_sha256=sha(out/'simulation_gate.json'),mode_ties=['any','fractional'],mode_length_rule='Per-image maximum observed predicted length; absent positions vote None, not filled from labels.',rank1_interpretation='Internal agreement factors used only for epoch ranking, not absolute accuracy or synthetic-to-real gap estimation.',epoch_windows={'Model1-A':args.a_through,'Model1-B':args.b_through},cross_run='Historical Model1-A versus v3 Model1-B; not matched-generator independent seeds.',calibration_bins=[0,.2,.4,.6,.8,1],b13_near_tolerance=.03,optional_collision_term='Not fitted; c and accuracy are additionally non-identifiable without anchors.',automatic_k='Smallest passing k; if none pass, report k5 fallback explicitly, not a successful band detection.',rank2_flag='In-sample additional structure, not proof of wrongness; components not identified as truth/trap.',selection_tie='earliest epoch',bootstrap_samples=10000))
    tables={name:load(name,args.root,args.a_through if name=='Model1-A' else args.b_through) for name in ['Model1-A','Model1-B']}
    names=list(tables) if args.run=='all' else [args.run]
    kernels=['exact','char','edit'] if args.kernel=='all' else [args.kernel]
    write(out/'prediction_sources.json',{name:list(t.sources) for name,t in tables.items()})
    scores=[];diagnostics=[];curves=[];fits_saved={}
    def add(table,method,kernel,k,v,**extra):
        assert np.isfinite(v).all()
        scores.append(dict(run=table.run,method=method,kernel=kernel,k=k,selected_epoch=int(table.epochs[np.argmax(v)]),scores=v.tolist(),**extra))
    for name in names:
        table=tables[name];other=tables['Model1-B' if name=='Model1-A' else 'Model1-A'];print('Selecting',name,flush=True)
        for kernel in kernels:
            A=matrix(table,kernel);np.save(out/f'{name}_{kernel}_agreement.npy',A)
            fits={k:rank1(A,table.epochs,k) for k in CONFIG['k_values']}
            auto,criteria,passed=choose_k(A,table.epochs,fits)
            use=auto if args.k=='auto' else int(args.k)
            add(table,'pairwise_all',kernel,'all',pairwise_scores(A,table.epochs))
            cross=matrix(table,kernel,other);np.save(out/f'{name}_{kernel}_cross_agreement.npy',cross)
            add(table,'cross_pairwise',kernel,'cross',cross.mean(axis=1))
            for k,f in fits.items():
                assert f['converged'],'rank1 did not converge'
                add(table,'pairwise_masked',kernel,k,pairwise_scores(A,table.epochs,k))
                add(table,'rank1',kernel,k,f['scores'])
                g=rank2(A,table.epochs,k,f)
                for r in residual_profile(A,table.epochs,f):curves.append(dict(run=name,kernel=kernel,k=k,**r))
                diagnostics.append(dict(run=name,kernel=kernel,k=k,rank1_mse=f['mse'],rank2_mse=g['mse'],relative_mse_improvement=g['relative_mse_improvement'],absolute_rmse_improvement=g['absolute_rmse_improvement'],trap_component_flag=g['trap_component_flag'],rank2_converged=g['converged'],selected_epoch=int(table.epochs[np.argmax(f['scores'])]),triplet_counts=f['triplet_counts']))
                fits_saved[f'{name}_{kernel}_{k}']=dict(epochs=table.epochs,rank1=f,rank2=g)
            add(table,'rank1_auto' if args.k=='auto' else 'rank1_requested',kernel,'auto' if args.k=='auto' else int(args.k),fits[use]['scores'],resolved_k=use,k_criterion_passed=passed)
            write(out/f'{name}_{kernel}_k_selection.json',dict(chosen_k=auto,criterion_passed=passed,criteria=criteria))
            if kernel in ('exact','char'):
                for fractional in [False,True]:
                    suffix='fractional' if fractional else 'anytie'
                    add(table,'mode_self_'+suffix,kernel,'all',mode_scores(table,kernel,fractional=fractional))
                    add(table,'mode_loo_'+suffix,kernel,0,mode_scores(table,kernel,exclude='loo',fractional=fractional))
                    for k in CONFIG['k_values']:
                        add(table,'mode_neighbour_'+suffix,kernel,k,mode_scores(table,kernel,exclude='neighbour',k=k,fractional=fractional))
                    add(table,'cross_mode_'+suffix,kernel,'cross',mode_scores(table,kernel,fractional=fractional,other=other))
            print(' ',kernel,'auto k',auto,'rank1 epoch',int(table.epochs[np.argmax(fits[auto]['scores'])]),flush=True)
    write(out/'scores.json',scores);write(out/'factorizations.json',clean(fits_saved));write(out/'rank_diagnostics.json',diagnostics)
    pd.DataFrame(curves).to_csv(out/'residual_by_distance.csv',index=False)
    write(out/'selection_freeze.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),labels_read=False,code_hashes=code,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},selected=[{k:v for k,v in r.items() if k!='scores'} for r in scores]))
    selected_tables={name:tables[name] for name in names}
    from .plots import plot_all
    if args.evaluate:
        from .evaluate import evaluate
        summary,truth=evaluate(args.root,out,selected_tables)
        plot_all(out,selected_tables,truth)
    else:
        from .plots import plot_rankings
        from .ranking_only import ranking_only
        result=ranking_only(out);write(out/'ranking_only.json',result)
        pd.DataFrame(result['rankings']).to_csv(out/'summary_rankings.csv',index=False)
        plot_rankings(out,selected_tables)
    print('Complete:',out.resolve(),flush=True)
if __name__=='__main__':main()
