"""Independent scenario checks for the declared Model3 policy."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path('E:/pranav/Thesis-Pranav-Final')
sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.training.cryptbb_feature_v1_sampler import FeatureSampler
from capri_controller.training.cryptbb_feature_b_v1 import write,STUDY,sha

buckets=np.repeat(np.arange(3),300)
groups=np.full((900,4),-1,np.int32)
groups[:300,0]=0;groups[300:600,1]=0;groups[600:,2]=0;groups[:,3]=0
manifest=dict(families=[dict(name=n,importance=p,reference_count=int(p*100))
                       for n,p in [('a',1.),('b',.9),('c',.13),('d',.05)]])
s=FeatureSampler(buckets,groups,manifest);s.draw();s.draw()
ok=np.zeros(900,bool);ok[:240]=True;ok[300:450]=True;ok[600:750]=True
r=s.update(ok,2)
assert r['active_families']==['b','c','d']
assert r['families'][0]['learned'] and r['families'][0]['attention']==.25
expected=np.array([.06,.475,.09])
assert np.allclose(r['bucket_demand'],expected)
assert np.allclose(s.mass,.2/3+.8*expected/expected.sum())
assert s.mass[1]>s.mass[2]>s.mass[0]
ok[210:240]=False
r=s.update(ok,2)
assert r['active_families']==['a','b','c'] and r['families'][0]['forgetting']
assert r['families'][3]['attention']==0
# Full20-epoch lifecycle; learned means >75%, and counts always remain full batches.
records=[]
for epoch in range(3,21):
    before=s.state_dict();draw,rec=s.draw();rep=FeatureSampler(buckets,groups,manifest);rep.load_state_dict(before)
    draw2,rec2=rep.draw();assert np.array_equal(draw,draw2) and rec==rec2
    assert len(draw)==100096 and len(draw)%128==0
    assert np.array_equal(s.buckets,buckets)
    s.update(np.full(900,epoch%2==0,bool),epoch)
    assert np.all(s.conditional>0) and min(s.mass)>=.2/3-1e-12
    records.append(rec)
# Splitting a family into equally supported groups must not multiply its priority.
m=dict(families=[dict(name='same',importance=1.,reference_count=100)])
a=FeatureSampler(buckets,np.zeros((900,1),np.int32),m)
b=FeatureSampler(buckets,(np.arange(900)%2).reshape(-1,1),m)
for x in (a,b):x.draw();x.draw();x.update(np.arange(900)%4<2,2)
assert np.allclose(a.mass,b.mass)
assert np.allclose(a.last_assessment['bucket_demand'],b.last_assessment['bucket_demand'])
write(STUDY/'scenario_verification.json',dict(passed=True,progression=True,forgetting_reactivation=True,
    explicit_demand_formula=True,all_three_bucket_weights_respond=True,learned_maintenance=True,
    lifecycle20=True,all_epoch_draw_replays=True,bin_count_does_not_multiply_importance=True,
    script_sha256=sha(Path(__file__)),epochs=records))
print('Independent policy scenarios passed')
