"""String similarities; no label input."""
import numpy as np
from .loader import PredictionTable

def levenshtein(a,b):
    if len(a)<len(b):a,b=b,a
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]
def similarity(x,y,kernel):
    if kernel=='exact':return float(x==y)
    if not x and not y:return 1.
    if kernel=='char' and len(x)==len(y):return sum(a==b for a,b in zip(x,y))/len(x)
    assert kernel in ('char','edit')
    return 1-levenshtein(x,y)/max(len(x),len(y))
def matrix(table,kernel,other=None):
    assert type(table) is PredictionTable
    other=table if other is None else other
    assert type(other) is PredictionTable and table.ids==other.ids
    n=len(table.epochs);q=len(other.epochs);out=np.empty((n,q))
    for t in range(n):
        for u in range(q):
            if other is table and u<t:out[t,u]=out[u,t];continue
            x,y=table.predictions[t],other.predictions[u]
            out[t,u]=np.mean(x==y) if kernel=='exact' else np.mean([similarity(a,b,kernel) for a,b in zip(x,y)])
    return out
