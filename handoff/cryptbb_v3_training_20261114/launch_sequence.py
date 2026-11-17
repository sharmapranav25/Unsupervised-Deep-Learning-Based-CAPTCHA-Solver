from pathlib import Path
import json,hashlib,shutil,sys,subprocess
import psutil
from datetime import datetime,timezone
R=Path.cwd();S=R/'handoff/cryptbb_v3_training_20261114'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
assert read(S/'prelaunch_checks.json')['passed'];assert not (S/'launch.json').exists()
write(S/'authorization.json',dict(at=datetime.now(timezone.utc).isoformat(),user_request='After CryptBB v3 generation, train Model1-A/B in parallel, then Model2-A/B in parallel',priority='CryptBB Model1 -> Model2 before pending Model4 teacher queue',epochs_each=20,batch_size=128,seed=48,teacher_rankings='Model1-B epoch20 ranks A; Model1-A epoch20 ranks B',model3_authorized=False,production_gate='Both validated100k corpora + exact-data paired preflight',notes='No relaxed controller admission; no restarts of stopped v2 or historical Model1-B'))
p=read(R/'handoff/cryptbb_v2_20261114/agreement_protocol.json');manifest=R/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json'
p.update(version=3,experiment='cryptbb_controller_v3_split',manifest=manifest.relative_to(R).as_posix(),manifest_sha256=sha(manifest),images=251,selection_unit='One shared epoch within each Model1 or Model2 A/B pair',pool_usage='Same251 evaluation pixels used unlabeled for agreement and subsequently scored; transductive historical data, not untouched test',training_authorized=True,teacher_policy='Opposite Model1 epoch20, independent of reporting selection',inherited_policy_sha256=sha(R/'handoff/cryptbb_v2_20261114/agreement_protocol.json'))
assert len(read(manifest))==251;write(S/'agreement_protocol.json',p)
marker=R/'model4/runs/darkbay_AB_teachers_20261114/STOP';write(S/'teacher_queue_hold.json',dict(marker=marker.relative_to(R).as_posix(),sha256=sha(marker),release='After successful Model1 and Model2 completion, resume previously authorized teacher queue only if marker is unchanged'))
files=[R/'Capri/src/capri_controller/training'/f'{n}.py' for n in ['cryptbb_v3_model1','cryptbb_v3_model2','cryptbb_v3_sequence','cryptbb_model1','ab_curriculum','ds2_data','ds3_data']]
files += [R/'Capri/src/capri_controller'/f'{n}.py' for n in ['cryptbb_v3_corpus','cryptbb_v3_source','cryptbb_v3_workflow','decoder/common']]
files += list((R/'model1/src/OpenOCR').rglob('*.py'))
files += [R/f'model1/configs/cryptbb/{t}/prepared_fixed32x128.yml' for t in 'AB']
files += [S/'authorization.json',S/'agreement_protocol.json',S/'teacher_queue_hold.json',R/'model4/src/model4_teachers/gpu_queue.py',R/'model4/launch_teachers.ps1',manifest]
records=[]
for f in sorted(set(files)):
 rel=f.relative_to(R);dest=S/'launch_sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);h=sha(f);assert sha(dest)==h;records.append(dict(path=rel.as_posix(),sha256=h))
write(S/'launch_sources.json',records)
with (S/'coordinator.log').open('a',encoding='utf-8') as log:
 cmd=[sys.executable,'-B','-u','-m','capri_controller.training.cryptbb_v3_sequence'];q=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
r=dict(pid=q.pid,process_created=psutil.Process(q.pid).create_time(),started_at=datetime.now(timezone.utc).isoformat(),command=cmd);write(S/'launch.json',r);print(json.dumps(dict(launched=r,frozen_sources=len(records)),indent=2))
