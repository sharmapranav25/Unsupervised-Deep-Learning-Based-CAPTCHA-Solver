from pathlib import Path
import ast,contextlib,types,json,copy
import numpy as np
from capri_controller.training import cryptbb_v3_model1 as M1,cryptbb_v3_model2 as M2
from capri_controller.training import ab_curriculum as C
ROOT=Path.cwd();S=ROOT/'handoff/cryptbb_v3_training_20261114';S.mkdir(exist_ok=False)
rows=[];buckets=np.repeat(np.arange(3),[33334,33333,33333]);ids=np.arange(1,100001)
for epoch in range(1,21):
 sm=types.SimpleNamespace(epoch=epoch,mass=np.array([1.,0.,0.]),buckets=buckets,ids=ids)
 sm2=copy.deepcopy(sm);a,r=C.draws_for(sm,2,epoch);b,r2=C.draws_for(sm2,2,epoch)
 assert np.array_equal(a,b) and r==r2 and len(a)==100096 and len(a)//128==782 and a.min()>=0 and a.max()<100000
 expected=[100096,0,0] if epoch==1 else [50048,50048,0] if epoch==2 else [15014,25024,60058]
 assert np.bincount(buckets[a],minlength=3).tolist()==expected
 rows.append(r)
events=[]
@contextlib.contextmanager
def lock():
 events.append('acquire')
 try:yield
 finally:events.append('release')
class Loader:
 dataset='sentinel';batch_sampler='sampler'
 def __len__(self):return 2
 def __iter__(self):
  for i in range(2):events.append('load');yield i
M1.gpu_slot=lock
loader=M1.LockedLoader(Loader());it=iter(loader)
assert len(loader)==2 and loader.dataset=='sentinel' and loader.batch_sampler=='sampler'
assert next(it)==0;events.append('step');assert events==['load','acquire','step']
assert next(it)==1;events.append('step');it.close();assert events==['load','acquire','step','release','load','acquire','step','release']
assert str(M2.teacher('A')).replace('\\','/').endswith('model1/runs/cryptbb_v3/B/epoch_20.pth')
assert str(M2.teacher('B')).replace('\\','/').endswith('model1/runs/cryptbb_v3/A/epoch_20.pth')
try:M2.run_dir(3,'A');raise RuntimeError('Model3 path accepted')
except AssertionError:pass
result={'passed':True,'static_schedule_twenty_epochs':rows,'full_batch_lock_lifecycle':events,'cross_teacher_paths_verified':True,'model3_rejected':True,'gpu_work_performed':False,'actual_data_gpu_preflight':'Pending completed v3 datasets and later Model1 teachers'}
(S/'prelaunch_checks.json').write_text(json.dumps(result,indent=2)+'\n');print('Passed: twenty-epoch static budgets/replay, cross-teacher paths, full-step lock lifecycle, Model3 exclusion.')
