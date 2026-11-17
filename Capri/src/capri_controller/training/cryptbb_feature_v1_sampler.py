"""Reference-priority curriculum; immutable teacher buckets, synthetic-only feedback."""
import copy
import hashlib
import numpy as np

POLICY = dict(active_families=3, learned_strictly_above=.75,
              learned_attention=.25, exploration=.20, min_cell_support=100,
              seed=48, draws=100096, batch=128)


class FeatureSampler:
    def __init__(self, buckets, groups, manifest):
        self.ids = np.arange(1, len(buckets)+1, dtype=np.int64)
        self.buckets = np.asarray(buckets, dtype=np.int64).copy()
        self.buckets.setflags(write=False)
        self.original_buckets = self.buckets.copy()
        self.groups = np.asarray(groups, dtype=np.int32)
        self.manifest = copy.deepcopy(manifest)
        self.epoch = 1
        self.mass = np.array([1.,0.,0.])
        self.conditional = np.zeros(len(buckets), dtype=np.float64)
        self.last_assessment = None
        self.learned = np.zeros(len(manifest['families']), dtype=bool)

    def update(self, correct, completed_epoch):
        assert completed_epoch >= 2 and self.epoch == completed_epoch+1
        correct = np.asarray(correct, dtype=bool)
        assert correct.shape == self.buckets.shape
        assert np.array_equal(self.buckets, self.original_buckets)
        families = self.manifest['families']
        accuracy = []
        for f in range(len(families)):
            mask = self.groups[:,f] >= 0
            accuracy.append(float(correct[mask].mean()) if mask.any() else None)
        learned = np.array([a is not None and a > POLICY['learned_strictly_above'] for a in accuracy])
        order = sorted(range(len(families)), key=lambda f:(-families[f]['importance'],families[f]['name']))
        active = [f for f in order if families[f]['importance'] > 0 and accuracy[f] is not None and not learned[f]][:POLICY['active_families']]
        active_set = set(active)
        attention = np.array([1. if f in active_set else POLICY['learned_attention'] if learned[f] else 0.
                              for f in range(len(families))])
        # Each (family, setting subgroup, bucket) gets a rate, never a raw error count.
        # Split each family's importance equally over its supported setting groups.
        # This avoids multiplying importance just because a feature has more bins.
        demand = np.zeros(3, dtype=np.float64)
        local = np.zeros(len(correct), dtype=np.float64)
        cells = []
        for f, family in enumerate(families):
            present = self.groups[:,f] >= 0
            labels = np.unique(self.groups[present,f])
            if not len(labels): continue
            for g in labels:
                group_mask = self.groups[:,f] == g
                for b in range(3):
                    ix = np.flatnonzero(group_mask & (self.buckets == b))
                    if not len(ix): continue
                    acc = float(correct[ix].mean())
                    support = min(1.,len(ix)/POLICY['min_cell_support'])
                    weight = family['importance'] * attention[f] * (1.-acc) * support / len(labels)
                    demand[b] += weight
                    local[ix] += weight / len(ix)
                    cells.append(dict(family=family['name'],group=int(g),bucket=b,images=len(ix),
                                      correct=int(correct[ix].sum()),accuracy=acc,support_factor=support,demand=weight))
        explore = POLICY['exploration']
        self.mass = np.full(3,1/3) if demand.sum()==0 else explore/3 + (1-explore)*demand/demand.sum()
        for b in range(3):
            ix = np.flatnonzero(self.buckets==b)
            s = local[ix].sum()
            self.conditional[ix] = 1/len(ix) if s==0 else explore/len(ix)+(1-explore)*local[ix]/s
            assert np.isclose(self.conditional[ix].sum(),1,rtol=0,atol=1e-12)
        assert np.isfinite(self.mass).all() and np.all(self.mass>=explore/3-1e-12)
        assert np.isclose(self.mass.sum(),1)
        self.last_assessment = dict(completed_epoch=completed_epoch,overall_accuracy=float(correct.mean()),
            active_families=[families[f]['name'] for f in active],bucket_demand=demand.tolist(),next_mass=self.mass.tolist(),
            families=[dict(name=row['name'],importance=row['importance'],reference_count=row['reference_count'],
                           accuracy=accuracy[f],learned=bool(learned[f]),attention=float(attention[f]),
                           forgetting=bool(self.learned[f] and not learned[f])) for f,row in enumerate(families)],cells=cells)
        self.learned = learned
        return copy.deepcopy(self.last_assessment)

    def draw(self):
        assert np.array_equal(self.buckets,self.original_buckets), 'Teacher membership changed'
        epoch = self.epoch
        if epoch<=2:self.mass = np.array([1.,0.,0.] if epoch==1 else [.5,.5,0.])
        else:assert self.last_assessment['completed_epoch']==epoch-1
        count=POLICY['draws']; counts=np.rint(count*self.mass).astype(np.int64)
        counts[np.argmax(self.mass)] += count-counts.sum()
        rng=np.random.default_rng(POLICY['seed']+epoch*100003)
        parts=[]
        for b,n in enumerate(counts):
            if not n:continue
            ix=np.flatnonzero(self.buckets==b)
            if epoch<=2:parts.append(rng.choice(ix,n,replace=True))
            else:parts.append(rng.choice(ix,n,replace=True,p=self.conditional[ix]))
        draws=np.concatenate(parts);rng.shuffle(draws)
        assert len(draws)==count and count%POLICY['batch']==0
        assert np.bincount(self.buckets[draws],minlength=3).tolist()==counts.tolist()
        receipt=dict(epoch=epoch,mass=self.mass.tolist(),draws=counts.tolist(),total=count,dropped=0,
                     sample_ids_sha256=hashlib.sha256(self.ids[draws].tobytes()).hexdigest())
        self.epoch+=1
        return draws,receipt

    def state_dict(self):
        return copy.deepcopy(dict(epoch=self.epoch,mass=self.mass,conditional=self.conditional,
            last_assessment=self.last_assessment,learned=self.learned,
            bucket_sha256=hashlib.sha256(self.buckets.tobytes()).hexdigest(),
            group_sha256=hashlib.sha256(self.groups.tobytes()).hexdigest()))

    def load_state_dict(self,state):
        assert state['bucket_sha256']==hashlib.sha256(self.buckets.tobytes()).hexdigest()
        assert state['group_sha256']==hashlib.sha256(self.groups.tobytes()).hexdigest()
        for k in ('epoch','mass','conditional','last_assessment','learned'):
            setattr(self,k,copy.deepcopy(state[k]))


def self_test():
    buckets=np.repeat(np.arange(3),300)
    groups=np.tile(np.arange(3),(900,1)).astype(np.int32)
    manifest=dict(families=[dict(name=n,importance=p,reference_count=int(p*100))
                            for n,p in [('spacing',1.),('width',.99),('circles',.13)]])
    s=FeatureSampler(buckets,groups,manifest)
    for epoch in (1,2):
        d,r=s.draw();assert r['draws']==([100096,0,0] if epoch==1 else [50048,50048,0])
    ok=np.tile(np.r_[np.ones(225,bool),np.zeros(75,bool)],3)
    r=s.update(ok,2);assert not any(v['learned'] for v in r['families'])
    ok[[225,525,825]]=True
    r=s.update(ok,2);assert all(v['learned'] for v in r['families']) and not r['active_families']
    ok[300:600]=False
    r=s.update(ok,2);assert s.mass[1]>s.mass[0] and all(v['forgetting'] for v in r['families'])
    state=s.state_dict();d,r=s.draw();t=FeatureSampler(buckets,groups,manifest);t.load_state_dict(state)
    d2,r2=t.draw();assert np.array_equal(d,d2) and r==r2
    assert np.array_equal(s.buckets,buckets) and np.all(s.conditional>0)
    # Equal error rates give reference frequency its intended influence.
    sums={n:sum(c['demand'] for c in s.last_assessment['cells'] if c['family']==n) for n in ('spacing','circles')}
    assert np.isclose(sums['spacing']/sums['circles'],1/.13)
    # The same groups with all-correct feedback retain coverage and terminate frontier.
    s.update(np.ones(900,bool),3);assert np.allclose(s.mass,1/3)
    assert not s.last_assessment['active_families']
    return dict(passed=True,threshold_strict=True,forgetting=True,reference_importance=True,
                immutable_buckets=True,exact_replay=True,full_batches=True,coverage=True)
