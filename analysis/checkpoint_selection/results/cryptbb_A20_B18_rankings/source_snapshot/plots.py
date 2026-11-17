"""Scientific plots; optional truth is provided only after selection freeze."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .loader import read

def plot_all(out,tables,truth=None):
    out=Path(out);fits=read(out/'factorizations.json');scores=read(out/'scores.json');profiles=pd.read_csv(out/'residual_by_distance.csv')
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    for name,t in tables.items():
        kernels=sorted(set(r['kernel'] for r in scores if r['run']==name))
        for kernel in kernels:
            choice=read(out/f'{name}_{kernel}_k_selection.json');k=choice['chosen_k'];f=fits[f'{name}_{kernel}_{k}'];A=np.load(out/f'{name}_{kernel}_agreement.npy')
            estimate=np.array(f['rank1']['scores']);res=A-np.outer(estimate,estimate)
            fig,ax=plt.subplots(1,2,figsize=(11,4.6))
            extent=[t.epochs[0]-.5,t.epochs[-1]+.5,t.epochs[-1]+.5,t.epochs[0]-.5]
            im=ax[0].imshow(A,vmin=0,vmax=1,cmap='viridis',extent=extent);fig.colorbar(im,ax=ax[0],fraction=.045);ax[0].set_title('Pairwise agreement')
            lim=max(.01,np.max(np.abs(res)));im=ax[1].imshow(res,vmin=-lim,vmax=lim,cmap='RdBu_r',extent=extent);fig.colorbar(im,ax=ax[1],fraction=.045);ax[1].set_title(f'Signed rank-1 residual (k={k})')
            for a in ax:a.set_xlabel('Epoch');a.set_ylabel('Epoch')
            fig.suptitle(f'{name} | {kernel} | diagonal retained for display only');fig.tight_layout();fig.savefig(out/f'{name}_{kernel}_heatmaps.png');plt.close(fig)
            fig,ax=plt.subplots(figsize=(9,4.5))
            for kk in range(6):
                ff=fits[f'{name}_{kernel}_{kk}'];ax.plot(t.epochs,ff['rank1']['scores'],alpha=.2,lw=1,color='#4256a4')
            ax.plot(t.epochs,estimate,'o-',color='#4256a4',label=f'Rank-1 factor, auto k={k}')
            if truth is not None:
                rr=truth[truth.run==name];ax.plot(rr.epoch,rr.accuracy,'s-',color='#c56822',label='Actual exact accuracy (evaluation only)')
            ax.set(xlabel='Epoch',ylabel='Fraction',ylim=(0,1),title=f'{name} | {kernel}: factors versus accuracy');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'{name}_{kernel}_accuracy.png');plt.close(fig)
            fig,ax=plt.subplots(figsize=(8,4.5))
            for kk in range(6):
                rr=profiles[(profiles.run==name)&(profiles.kernel==kernel)&(profiles.k==kk)]
                ax.plot(rr.distance,rr.mean_absolute_residual,'o-',lw=1,label=f'k={kk}')
            ax.set(xlabel='Epoch distance',ylabel='Mean absolute residual',title=f'{name} | {kernel}: residual by distance');ax.legend(ncol=3);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'{name}_{kernel}_residual_distance.png');plt.close(fig)
            fig,ax=plt.subplots(figsize=(8,4))
            x=np.array(f['rank2']['loadings']);ax.plot(t.epochs,x[:,0],'o-',label='Component 1');ax.plot(t.epochs,x[:,1],'s-',label='Component 2')
            ax.set(xlabel='Epoch',ylabel='Loading',title=f'{name} | {kernel}: rank 2, k={k} (components unidentified)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'{name}_{kernel}_rank2_loadings.png');plt.close(fig)


def plot_rankings(out,tables):
    """Display epoch preference ranks only, no absolute-accuracy interpretation."""
    from scipy.stats import rankdata
    out=Path(out);scores=read(out/'scores.json')
    for name,t in tables.items():
        fig,ax=plt.subplots(figsize=(9,4.5))
        for kernel in ['exact','char','edit']:
            rows=[r for r in scores if r['run']==name and r['method']=='rank1_auto' and r['kernel']==kernel]
            if rows:
                r=rows[0];ax.plot(t.epochs,rankdata(-np.array(r['scores']),method='min'),'o-',label=f"{kernel}, k={r['resolved_k']}")
        ax.set(xlabel='Checkpoint epoch',ylabel='Preference rank (1 = best)',title=f'{name}: label-free checkpoint ranking')
        ax.invert_yaxis();ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'{name}_rankings.png');plt.close(fig)
