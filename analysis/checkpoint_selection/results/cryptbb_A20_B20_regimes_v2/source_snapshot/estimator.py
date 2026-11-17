"""Masked factorization. All constants fixed before real-data selection."""
import numpy as np
from scipy.optimize import least_squares
CONFIG=dict(seed=4817,k_values=list(range(6)),residual_absolute_tolerance=0.005,residual_relative_tolerance=0.25,rank2_relative_mse_improvement=0.35,rank2_absolute_rmse_improvement=0.003,rank2_starts=4,bootstrap_samples=10000)

def mask_for(epochs,k):return np.abs(epochs[:,None]-epochs[None,:])>k

def triplet_init(A,mask):
    n=len(A);a=np.zeros(n);counts=[]
    for t in range(n):
        v=[]
        for u in range(n):
            for w in range(u+1,n):
                if mask[t,u] and mask[t,w] and mask[u,w] and A[u,w]>1e-12:
                    v.append(np.sqrt(max(0,A[t,u]*A[t,w]/A[u,w])))
        counts.append(len(v))
        a[t]=np.median(v) if v else np.sqrt(np.mean(A[t,mask[t]]))
    return np.clip(a,1e-6,1),counts

def rank1(A,epochs,k=0):
    mask=mask_for(epochs,k);assert mask.sum(axis=1).min()>0
    a,counts=triplet_init(A,mask);initial=a.copy();last=np.inf
    for iteration in range(20000):
        old=a.copy()
        # Gauss-Seidel coordinate descent: each update minimizes its constrained coordinate.
        for t in range(len(a)):
            v=a[mask[t]];den=v@v
            a[t]=np.clip(A[t,mask[t]]@v/den,0,1) if den>1e-20 else 0.
        loss=np.mean((A-np.outer(a,a))[mask]**2)
        assert loss<=last+1e-10,'Coordinate descent increased loss'
        if np.max(np.abs(old-a))<1e-10:break
        last=loss
    return dict(scores=a,initial=initial,triplet_counts=counts,mask=mask,mse=float(loss),converged=iteration<19999,iterations=iteration+1,prediction=np.outer(a,a))

def rank2(A,epochs,k,r1=None):
    r1=rank1(A,epochs,k) if r1 is None else r1
    mask=r1['mask'];ii,jj=np.where(np.triu(mask,1));n=len(A)
    rng=np.random.default_rng(CONFIG['seed']+k);best=None
    def fun(v):
        x=v.reshape(n,2);return (x[ii]*x[jj]).sum(axis=1)-A[ii,jj]
    def jac(v):
        x=v.reshape(n,2);J=np.zeros((len(ii),2*n));r=np.arange(len(ii))
        for c in range(2):J[r,2*ii+c]=x[jj,c];J[r,2*jj+c]=x[ii,c]
        return J
    for start in range(CONFIG['rank2_starts']):
        if start==0:x=np.column_stack([r1['scores'],rng.uniform(.01,.1,n)])
        else:x=rng.uniform(.02,.8,(n,2))
        fit=least_squares(fun,x.ravel(),jac=jac,bounds=(0,1),max_nfev=1000,ftol=1e-10,xtol=1e-10,gtol=1e-9)
        mse=float(np.mean(fit.fun**2))
        if best is None or mse<best[0]:best=(mse,fit)
    mse,fit=best;x=fit.x.reshape(n,2)
    x=x[:,np.argsort(-np.sum(x*x,axis=0))] # arbitrary energy ordering, NOT truth/trap identification
    relative=(r1['mse']-mse)/max(r1['mse'],1e-15);absolute=np.sqrt(r1['mse'])-np.sqrt(mse)
    flag=relative>CONFIG['rank2_relative_mse_improvement'] and absolute>CONFIG['rank2_absolute_rmse_improvement']
    return dict(loadings=x,mse=mse,relative_mse_improvement=float(relative),absolute_rmse_improvement=float(absolute),trap_component_flag=bool(flag),converged=bool(fit.success),prediction=x@x.T)

def residual_profile(A,epochs,fit):
    d=np.abs(epochs[:,None]-epochs[None,:]);res=np.abs(A-fit['prediction'])
    return [dict(distance=int(v),mean_absolute_residual=float(res[d==v].mean()),pairs=int((d==v).sum()//2)) for v in sorted(set(d.ravel())-{0})]
def choose_k(A,epochs,fits):
    records=[]
    for k,fit in fits.items():
        d=np.abs(epochs[:,None]-epochs[None,:]);res=np.abs(A-fit['prediction'])
        near=res[d==k+1];far=res[d>=max(k+2,int(np.ceil((epochs.max()-epochs.min()+1)/2)))]
        nm=float(near.mean());fm=float(far.mean());tol=max(CONFIG['residual_absolute_tolerance'],CONFIG['residual_relative_tolerance']*fm)
        records.append(dict(k=k,next_distance_residual=nm,far_residual=fm,tolerance=tol,passes=bool(nm<=fm+tol)))
    passed=[r['k'] for r in records if r['passes']]
    return (min(passed) if passed else max(fits)),records,bool(passed)
