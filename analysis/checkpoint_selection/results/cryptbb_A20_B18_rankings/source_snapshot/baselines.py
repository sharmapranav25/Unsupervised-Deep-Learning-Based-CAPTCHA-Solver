"""Mode and pairwise rules accepting prediction-only tables."""
from collections import Counter
import numpy as np
from .loader import PredictionTable

def mode_credit(value,values,fractional=False):
    c=Counter(values)
    if not c:return float('nan')
    top=max(c.values());m=[v for v,n in c.items() if n==top]
    return (1/len(m) if fractional else 1.) if value in m else 0.
def mode_scores(table,kernel='exact',exclude='none',k=0,fractional=False,other=None):
    assert type(table) is PredictionTable
    source=table if other is None else other
    assert table.ids==source.ids and kernel in ('exact','char')
    # Length comes from predictions only. Missing position is a literal absence vote.
    widths=[max(len(s) for s in list(table.predictions[:,i])+list(source.predictions[:,i])) for i in range(len(table.ids))]
    scores=[]
    for t,epoch in enumerate(table.epochs):
        use=np.ones(len(source.epochs),bool)
        if other is None and exclude=='loo':use=source.epochs!=epoch
        if other is None and exclude=='neighbour':use=np.abs(source.epochs-epoch)>k
        if not use.any():raise ValueError('No remaining voters')
        total=0.;count=0
        for i in range(len(table.ids)):
            s=table.predictions[t,i];v=source.predictions[use,i]
            if kernel=='exact':total+=mode_credit(s,v,fractional);count+=1
            else:
                for j in range(widths[i]):
                    total+=mode_credit(s[j] if j<len(s) else None,[x[j] if j<len(x) else None for x in v],fractional);count+=1
        scores.append(total/count if count else 1.)
    return np.array(scores)
def pairwise_scores(A,epochs,k=None):
    if k is None:return A.mean(axis=1) # Includes own diagonal, as requested.
    mask=np.abs(epochs[:,None]-epochs[None,:])>k
    assert mask.sum(axis=1).min()>0
    return (A*mask).sum(axis=1)/mask.sum(axis=1) # normalize unequal voter counts
