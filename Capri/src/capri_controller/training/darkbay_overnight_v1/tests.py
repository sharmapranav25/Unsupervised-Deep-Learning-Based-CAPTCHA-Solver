"""CPU-only prelaunch checks: policy, warmup, budgets and orchestration boundaries."""
import ast,contextlib,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import common as W,coordinator as Q
from .readers import StaticSampler
from ..cryptbb_feature_v1_sampler import FeatureSampler,self_test

def main():
    W.environment();W.verify_sources()
    checks=self_test()
    buckets=np.repeat(np.arange(3),[33334,33333,33333]);groups=np.zeros((100000,4),np.int32)
    groups[:25000,1:]=-1;meta=dict(families=[dict(name=n,importance=p,reference_count=int(p*100)) for n,p in [('width',1.),('rotation',.9),('grain',.3),('font',.1)]])
    s=StaticSampler(buckets);f=FeatureSampler(buckets,groups,meta)
    for epoch in range(1,21):
        before=s.state_dict();d,r=s.draw();again=StaticSampler(buckets);again.load_state_dict(before);dd,rr=again.draw()
        assert np.array_equal(d,dd) and r==rr and len(d)==100096
        expected=[100096,0,0] if epoch==1 else [50048,50048,0] if epoch==2 else [15014,25024,60058]
        assert r['draws']==expected and r['total']//128==782
        fd,fr=f.draw()
        if epoch<=2:assert np.array_equal(d,fd) and r==fr
        if epoch>=2:
            correct=np.arange(100000)%4<3
            out=f.update(correct,epoch)
            assert all(not r['learned'] for r in out['families'])
            f.update(np.ones(100000,bool),epoch);assert not f.last_assessment['active_families']
            f.update(np.zeros(100000,bool),epoch);assert all(r['forgetting'] for r in f.last_assessment['families'])
    checks.update(full20epoch_static_budget=True,warmup_exact=True,adaptive_full20epoch_progression=True)
    # Both A/B launches must be issued before supervising either, without GPU work.
    calls=[]
    with patch.object(Q,'launch_child',side_effect=lambda m,a,n:(calls.append(('launch',n)) or {'fake':n})),patch.object(Q,'supervise',side_effect=lambda j,p:calls.append(('supervise',len(j)))):
        Q.paired('model1-check')
    assert len(calls)==3 and calls[0][0]==calls[1][0]=='launch' and calls[2]==('supervise',2)
    checks['A_B_started_before_waiting']=True
    # Source-level boundaries verify that generation completion cannot skip audits.
    text=Path(Q.__file__).read_text();tree=ast.parse(text)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assert text.index("one('generation',['configs']")<text.index("paired('model1-worker')")<text.index("paired('worker',2)")<text.index("paired('worker',3)")
    assert 'await_previous()' in ast.unparse(run)
    checks['data_audits_precede_training_and_stage_order']=True
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    W.write(W.STUDY/'cpu_readiness.json',dict(passed=True,checked_at=W.now(),checks=checks,gpu_readiness='Deferred until fresh datasets and prior CryptBB worker completion; mandatory gates in coordinator'))
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
