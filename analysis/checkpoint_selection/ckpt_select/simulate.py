"""Deterministic synthetic prediction tables with known correctness probabilities."""
import numpy as np
from .loader import PredictionTable
from .agreement import matrix
from .estimator import rank1,rank2,CONFIG

def generate(case,m=2000,seed=4817):
    rng=np.random.default_rng(seed);n=20;epochs=np.arange(1,n+1)
    truth=np.array([f'T{i:05d}' for i in range(m)],object)
    if case==1:
        a=np.linspace(.3,.78,n);a[12]=.93
        correct=rng.random((n,m))<a[:,None]
        pred=np.array([[truth[i] if correct[t,i] else f'W{t:02d}:{i:05d}' for i in range(m)] for t in range(n)],object)
    elif case==2:
        a=np.full(n,.45);a[5]=.72;a[-6:]=.65
        correct=rng.random((n,m))<a[:,None]
        pred=np.array([[truth[i] if correct[t,i] else f'W{t:02d}:{i:05d}' for i in range(m)] for t in range(n)],object)
        # A Markov wrong-string identity with exponentially decaying reuse in the late block.
        # Correctness stays independent; only errors are coupled. No coupling outside that block.
        identity=np.arange(m)*0
        for t in range(n-6,n):
            reset=rng.random(m)>.98
            identity[reset]=t
            for i in np.flatnonzero(~correct[t]):pred[t,i]=f'C{identity[i]:02d}:{i:05d}'
    elif case==3:
        # pi of items permit a common trap; late checkpoints fall into it increasingly.
        pi=.45;a=np.linspace(.4,.62,n);a[7]=.75
        trap_items=rng.random(m)<pi
        propensity=np.linspace(.02,.95,n)
        fall=trap_items[None,:] & (rng.random((n,m))<propensity[:,None])
        correct=(rng.random((n,m))<a[:,None]) & ~fall
        pred=np.array([[f'Z{i:05d}' if fall[t,i] else truth[i] if correct[t,i] else f'W{t:02d}:{i:05d}' for i in range(m)] for t in range(n)],object)
        a=a*(1-pi*propensity)
    else:raise ValueError(case)
    return PredictionTable(f'simulation{case}',epochs,tuple(str(i) for i in range(m)),pred),a,truth

def validate():
    records=[]
    table,a,truth=generate(1);A=matrix(table,'exact');f=rank1(A,table.epochs,0)
    mae=float(np.mean(np.abs(f['scores']-a)))
    assert mae<.02,(mae,f['scores'],a)
    assert np.argmax(f['scores'])==np.argmax(a)
    records.append(dict(case=1,mae=mae,selected_epoch=int(table.epochs[np.argmax(f['scores'])]),passed=True))
    table,a,truth=generate(2);A=matrix(table,'exact');raw=rank1(A,table.epochs,0);masked=rank1(A,table.epochs,5)
    assert np.argmax(raw['scores'])>=14,raw['scores']
    assert np.argmax(masked['scores'])==np.argmax(a),(masked['scores'],a)
    records.append(dict(case=2,unmasked_epoch=int(table.epochs[np.argmax(raw['scores'])]),masked_epoch=int(table.epochs[np.argmax(masked['scores'])]),true_best_epoch=int(table.epochs[np.argmax(a)]),passed=True))
    table,a,truth=generate(3);A=matrix(table,'exact');f=rank1(A,table.epochs,3);g=rank2(A,table.epochs,3,f)
    assert g['trap_component_flag'],g
    records.append(dict(case=3,relative_mse_improvement=g['relative_mse_improvement'],absolute_rmse_improvement=g['absolute_rmse_improvement'],passed=True))
    return records
if __name__=='__main__':
    import json
    print(json.dumps(validate(),indent=2))
