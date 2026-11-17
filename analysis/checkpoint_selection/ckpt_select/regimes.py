"""Prediction-only capped-mask and temporal-coupling experiment, version 2."""
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
from .agreement import similarity
from .baselines import mode_credit
from .estimator import triplet_init

CONFIG_V2 = dict(seed=4817, bootstrap_samples=2000, bootstrap_batch=100,
    residual_absolute_tolerance=.005, residual_relative_tolerance=.25,
    low_partner_score=.05, low_score_fraction_warning=.5,
    degenerate_score=.01, degenerate_output_share=.5,
    denominator_floor=1e-6, bootstrap_confidence=.95,
    candidate_rule='95% percentile CI overlaps the distant-score winner CI',
    plateau_rule='mean distant score over original epochs t-k through t+k, including self',
    fallback_rule='largest permitted k; explicitly mark criterion failure',
    coupling_tau_min=.05, coupling_tau_max_multiplier=100,
    interpretation='Uncalibrated agreement-factor ranking scores, not accuracy probabilities')

def k_cap(n):
    if n < 4: raise ValueError('At least four epochs required')
    return (n-2)//4

def item_agreements(table, kernel):
    p=table.predictions;n,m=p.shape
    x=np.empty((n,n,m),float)
    for i in range(n):
        for j in range(i,n):
            v=(p[i]==p[j]).astype(float) if kernel=='exact' else np.array([similarity(a,b,kernel) for a,b in zip(p[i],p[j])])
            x[i,j]=x[j,i]=v
    return x

def fit_mask(A, epochs, k, excluded=()):
    n=len(A);active=np.ones(n,bool);active[list(excluded)]=False
    d=np.abs(epochs[:,None]-epochs[None,:]);mask=(d>k)&active[:,None]&active[None,:]
    if active.sum()<3 or np.any(mask.sum(1)[active]==0):
        raise ValueError('Insufficient nondegenerate distant partners')
    ix=np.flatnonzero(active); sub=A[np.ix_(ix,ix)]; sm=mask[np.ix_(ix,ix)]
    z,counts=triplet_init(sub,sm); a=np.zeros(n);a[ix]=z
    for it in range(20000):
        old=a.copy()
        for t in ix:
            use=mask[t];v=a[use];den=v@v
            a[t]=np.clip(A[t,use]@v/den,0,1) if den>1e-20 else 0.
        if np.max(np.abs(a-old))<1e-10:break
    if it==19999:raise RuntimeError('Masked fit failed to converge')
    partners=(d>k)&active[None,:]
    # Excluded epochs never influence active fits; infer their scores from active partners afterward.
    supported=partners.sum(1)>0
    for t in np.flatnonzero(~active):
        use=partners[t];v=a[use];den=v@v
        a[t]=np.clip(A[t,use]@v/den,0,1) if den>1e-20 else 0.
    return dict(scores=a,mask=mask,partners=partners,active=active,supported=supported,
        prediction=np.outer(a,a),iterations=it+1,mse=float(np.mean((A-np.outer(a,a))[mask]**2)))

def select_mask(A,epochs,excluded=()):
    fits={};criteria=[];profiles=[]
    d=np.abs(epochs[:,None]-epochs[None,:]);active=np.ones(len(A),bool);active[list(excluded)]=False
    allowed=active[:,None]&active[None,:]
    for k in range(k_cap(len(A))+1):
        try:f=fit_mask(A,epochs,k,excluded)
        except ValueError as e:
            criteria.append(dict(k=k,passes=False,available=False,reason=str(e)));continue
        fits[k]=f;res=np.abs(A-f['prediction'])
        near=res[(d==k+1)&allowed];far=res[(d>=max(k+2,int(np.ceil((epochs.max()-epochs.min()+1)/2))))&allowed]
        nm=float(near.mean()) if near.size else None;fm=float(far.mean()) if far.size else None
        tol=max(.005,.25*fm) if fm is not None else None
        passed=nm is not None and fm is not None and nm<=fm+tol
        criteria.append(dict(k=k,available=True,next_distance_residual=nm,far_residual=fm,tolerance=tol,passes=bool(passed)))
        for dist in range(1,int(d.max())+1):
            values=res[(d==dist)&allowed]
            profiles.append(dict(k=k,distance=dist,mean_absolute_residual=float(values.mean()) if values.size else None,pairs=int(values.size//2)))
    if not fits:raise ValueError('No identifiable masked fit')
    passing=[x['k'] for x in criteria if x['passes']];chosen=min(passing) if passing else max(fits)
    return chosen,bool(passing),fits,criteria,profiles

def degeneracy(table,a):
    rows=[]
    for t,e in enumerate(table.epochs):
        values=table.predictions[t]; empty=float(np.mean(values==''));dominant=max(Counter(values).values())/len(values)
        reasons=[]
        if empty>.5:reasons.append('more_than_half_empty')
        if dominant>.5:reasons.append('more_than_half_identical_outputs')
        if a[t]<.01:reasons.append('factor_below_0.01')
        rows.append(dict(epoch=int(e),empty_share=empty,dominant_output_share=dominant,initial_factor=float(a[t]),flagged=bool(reasons),reasons=reasons,first_half=bool(t<len(a)//2)))
    return rows

def partner_report(epochs,fit,k):
    rows=[]
    for t,e in enumerate(epochs):
        use=fit['partners'][t];v=fit['scores'][use];low=float(np.mean(v<.05)) if len(v) else None
        rows.append(dict(epoch=int(e),k=k,score=float(fit['scores'][t]),unmasked_partners=int(use.sum()),
            mean_partner_factor=float(v.mean()) if len(v) else None,low_partner_fraction=low,
            mostly_low_partner_warning=bool(low is not None and low>.5),supported=bool(fit['supported'][t])))
    return rows

def joint_coupling(A,epochs,initial,excluded=()):
    n=len(A);active=np.ones(n,bool);active[list(excluded)]=False
    mask=(~np.eye(n,dtype=bool))&active[:,None]&active[None,:]
    ii,jj=np.where(np.triu(mask,1));dist=np.abs(epochs[ii]-epochs[jj]);ix=np.flatnonzero(active)
    # Optimize active factors only, then infer excluded ones without letting them steer the fit.
    pos={t:p for p,t in enumerate(ix)};i=np.array([pos[t] for t in ii]);j=np.array([pos[t] for t in jj]);q=len(ix)
    def fun(x):
        a=x[:q];c=x[q]*np.exp(-dist/np.exp(x[q+1]))
        return a[i]*a[j]+c*(1-a[i])*(1-a[j])-A[ii,jj]
    def jac(x):
        a=x[:q];tau=np.exp(x[q+1]);decay=np.exp(-dist/tau);c=x[q]*decay;J=np.zeros((len(i),q+2));r=np.arange(len(i))
        J[r,i]=a[j]-c*(1-a[j]);J[r,j]=a[i]-c*(1-a[i])
        J[:,q]=decay*(1-a[i])*(1-a[j]);J[:,q+1]=c*(dist/tau)*(1-a[i])*(1-a[j]);return J
    starts=[]
    for c,tau in [(0.05,1.),(.4,3.),(.9,10.),(.5,float(n))]:
        x=np.r_[np.clip(initial[ix],1e-5,1-1e-5),c,np.log(tau)]
        f=least_squares(fun,x,jac=jac,bounds=(np.r_[np.zeros(q+1),np.log(.05)],np.r_[np.ones(q+1),np.log(100*n)]),max_nfev=2000,ftol=1e-11,xtol=1e-11,gtol=1e-9)
        starts.append(f)
    successful=[f for f in starts if f.success]
    if not successful:raise RuntimeError('No converged joint-coupling fit')
    f=min(successful,key=lambda f:np.mean(f.fun**2));a=np.zeros(n);a[ix]=f.x[:q];c=float(f.x[q]);tau=float(np.exp(f.x[q+1]))
    for t in np.flatnonzero(~active):
        cc=c*np.exp(-np.abs(epochs[t]-epochs[ix])/tau);base=cc*(1-a[ix]);slope=a[ix]-base
        a[t]=np.clip(np.dot(slope,A[t,ix]-base)/max(np.dot(slope,slope),1e-20),0,1)
    distance=np.abs(epochs[:,None]-epochs[None,:]);pred=np.outer(a,a)+c*np.exp(-distance/tau)*np.outer(1-a,1-a)
    return dict(scores=a,kappa0=c,tau=tau,mse=float(np.mean(f.fun**2)),prediction=pred,converged=True,
        starts=[dict(mse=float(np.mean(z.fun**2)),converged=bool(z.success),selected_epoch=int(epochs[ix[np.argmax(z.x[:q])]])) for z in starts],
        tau_at_numerical_bound=bool(tau<.0501 or tau>99.9*n),selected_epoch=int(epochs[np.argmax(a)]))

def bootstrap_factors(items,epochs,k,fit,samples=2000,seed=4817):
    """Paired image bootstrap; fixed k/exclusions, full refit in vectorized batches."""
    n,_,m=items.shape;rng=np.random.default_rng(seed);all_scores=[];all_iters=[]
    active=fit['active'];ix=np.flatnonzero(active);mask=fit['mask'];flat=items.reshape(n*n,m).T
    for start in range(0,samples,100):
        count=min(100,samples-start);w=rng.multinomial(m,np.full(m,1/m),size=count)/m
        As=(w@flat).reshape(count,n,n);a=np.broadcast_to(fit['scores'],(count,n)).copy()
        for it in range(20000):
            old=a.copy()
            for t in ix:
                use=mask[t];v=a[:,use];den=np.sum(v*v,axis=1);num=np.sum(As[:,t,use]*v,axis=1)
                a[:,t]=np.clip(np.divide(num,den,out=np.zeros_like(num),where=den>1e-20),0,1)
            if np.max(np.abs(old-a))<1e-8:break
        if it==19999:raise RuntimeError('Bootstrap refit did not converge')
        for t in np.flatnonzero(~active):
            use=fit['partners'][t];v=a[:,use];den=np.sum(v*v,axis=1);num=np.sum(As[:,t,use]*v,axis=1)
            a[:,t]=np.clip(np.divide(num,den,out=np.zeros_like(num),where=den>1e-20),0,1)
        all_scores.append(a);all_iters.append(it+1)
    x=np.concatenate(all_scores);return dict(samples=x,low=np.quantile(x,.025,axis=0),high=np.quantile(x,.975,axis=0),se=x.std(0,ddof=1),max_iterations=max(all_iters))

def decay_fit(dist,kappa):
    dist=np.asarray(dist,float);v=np.asarray(kappa,float);ok=np.isfinite(v)&np.isfinite(dist)
    dist=dist[ok];v=v[ok]
    if len(v)<2 or len(np.unique(dist))<2:return dict(available=False,reason='Need at least two distances to identify decay',pairs=len(v))
    best=None
    for tau in [1.,3.,10.,100.]:
        f=least_squares(lambda x:x[0]*np.exp(-dist/np.exp(x[1]))-v,[.2,np.log(tau)],bounds=([0,np.log(.05)],[1,np.log(2000)]),max_nfev=1000)
        if f.success and (best is None or np.sum(f.fun**2)<np.sum(best.fun**2)):best=f
    if best is None:return dict(available=False,reason='Fit failed',pairs=len(v))
    return dict(available=True,kappa0=float(best.x[0]),tau=float(np.exp(best.x[1])),pairs=len(v),mse=float(np.mean(best.fun**2)),tau_at_numerical_bound=bool(np.exp(best.x[1])>1998 or np.exp(best.x[1])<.0501))

def neighbour_diagnostics(A,epochs,k,fit):
    a=fit['scores'];n=len(a);pairs=[];rows=[];active=fit['active']
    for i in range(n):
        for j in range(i+1,n):
            d=int(epochs[j]-epochs[i])
            if d>k:continue
            den=(1-a[i])*(1-a[j]);res=float(A[i,j]-a[i]*a[j]);kap=res/den if den>1e-6 else None
            pairs.append(dict(left=int(epochs[i]),right=int(epochs[j]),distance=d,residual=res,kappa=kap,denominator=float(den),eligible=bool(active[i] and active[j])))
    for t,e in enumerate(epochs):
        for d in range(1,k+1):
            pp=[p for p in pairs if p['distance']==d and (p['left']==e or p['right']==e) and active[int(p['right']-1 if p['left']==e else p['left']-1)]]
            valid=[p['kappa'] for p in pp if p['kappa'] is not None]
            rows.append(dict(epoch=int(e),distance=d,partners=len(pp),residual=float(np.mean([p['residual'] for p in pp])) if pp else None,kappa=float(np.mean(valid)) if valid else None,undefined_kappa_pairs=len(pp)-len(valid)))
    fitted={}
    for label,test in [('all',lambda p:True),('first_half',lambda p:p['right']<=epochs[n//2-1]),('second_half',lambda p:p['left']>=epochs[n//2])]:
        pp=[p for p in pairs if p['eligible'] and p['kappa'] is not None and test(p)]
        fitted[label]=decay_fit([p['distance'] for p in pp],[p['kappa'] for p in pp])
    mean=[]
    for e in epochs:
        values=[p['kappa'] for p in rows if p['epoch']==e and p['kappa'] is not None]
        mean.append(float(np.mean(values)) if values else None)
    return dict(pairs=pairs,by_epoch_distance=rows,mean_kappa=mean,decay=fitted,
        flip_rates=[dict(epoch=int(epochs[t]),next_epoch=int(epochs[t+1]),disagreement=float(1-A[t,t+1])) for t in range(n-1)])

def combine(epochs,k,fit,boot,neighbours):
    a=fit['scores'];best=int(np.argmax(a));supported=fit['supported']
    candidates=np.flatnonzero((boot['high']>=boot['low'][best])&(boot['low']<=boot['high'][best])&supported)
    plateau=np.array([np.mean(a[np.abs(epochs-e)<=k]) for e in epochs])
    kap=[np.inf if v is None else v for v in neighbours['mean_kappa']]
    pick=sorted(candidates,key=lambda t:(-plateau[t],kap[t],epochs[t]))[0]
    return dict(distant_pick=int(epochs[best]),combined_pick=int(epochs[pick]),candidates=epochs[candidates].tolist(),plateau_scores=plateau,ci_rule=CONFIG_V2['candidate_rule'],tie_rule='lower mean neighbour kappa, then earlier epoch; undefined kappa sorts last')

def decomposition(table,A,epochs,k,fit):
    d=np.abs(epochs[:,None]-epochs[None,:]);n=len(epochs);active=fit['active'];rows=[]
    widths=[max(map(len,table.predictions[:,i])) for i in range(len(table.ids))]
    for t,e in enumerate(epochs):
        pools={'neighbour':(d[t]>=1)&(d[t]<=k)&active,'distant':(d[t]>k)&active}
        total=sum(int(v.sum()) for v in pools.values())
        for region,use in pools.items():
            count=int(use.sum());mean=float(A[t,use].mean()) if count else None
            rows.append(dict(epoch=int(e),region=region,method='pairwise',kernel='input',partners=count,score=mean,weighted_contribution=mean*count/total if count else None))
            for kernel in ['exact','char']:
                for fractional in [False,True]:
                    credits=[]
                    if count:
                        for i in range(len(table.ids)):
                            s=table.predictions[t,i];v=table.predictions[use,i]
                            if kernel=='exact':credits.append(mode_credit(s,v,fractional))
                            else:
                                credits.extend(mode_credit(s[j] if j<len(s) else None,[x[j] if j<len(x) else None for x in v],fractional) for j in range(widths[i]))
                    rows.append(dict(epoch=int(e),region=region,method='mode_fractional' if fractional else 'mode_anytie',kernel=kernel,partners=count,score=float(np.mean(credits)) if credits else None,weighted_contribution=None))
    picks=[]
    for method,kernel in sorted(set((r['method'],r['kernel']) for r in rows)):
        for region in ['neighbour','distant']:
            rr=[r for r in rows if r['method']==method and r['kernel']==kernel and r['region']==region and r['score'] is not None]
            picks.append(dict(method=method,kernel=kernel,region=region,selected_epoch=max(rr,key=lambda r:r['score'])['epoch'] if rr else None))
    return dict(rows=rows,picks=picks,mode_caveat='Regional modes are separately recomputed voter-pool diagnostics; mode is nonlinear and these are NOT additive components of the original global mode score. Pairwise weighted contributions do add to the eligible off-diagonal mean.')
