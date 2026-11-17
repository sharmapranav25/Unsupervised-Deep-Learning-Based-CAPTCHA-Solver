from pathlib import Path
import shutil,subprocess,sys,psutil,ast
from capri_controller.decoder.common import read,write,sha,now
R=Path.cwd();D=R/'handoff/cryptbb_v3_20261114';S=R/'handoff/cryptbb_v3_training_20261114';rec=D/'recovery_20261114'
assert read(rec/'preflight.json')['passed']
# Freeze new admitted-recipe policy and adapters without touching original data pipeline guards.
files=[R/'Capri/src/capri_controller'/f'{n}.py' for n in ['cryptbb_v3_accepted_corpus','cryptbb_v3_recovery']]+[D/'accepted_recipe_protocol.json']
records=[]
for f in files:
 rel=f.relative_to(R);dest=rec/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);records.append(dict(path=rel.as_posix(),sha256=sha(f)))
write(rec/'sources.json',records)
# Freeze the reviewed v3-only changes before either model has initialized.
old=read(S/'launch_sources.json');allowed={'Capri/src/capri_controller/training/cryptbb_v3_model1.py','Capri/src/capri_controller/training/cryptbb_v3_sequence.py'}
assert {r['path'] for r in old if sha(R/r['path'])!=r['sha256']}==allowed
records=[]
for f in sorted(set([R/r['path'] for r in old]+files)):
 rel=f.relative_to(R);dest=S/'launch_sources_recovery'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);records.append(dict(path=rel.as_posix(),sha256=sha(f)))
write(S/'launch_sources.json',records)
write(rec/'recovery_protocol.json',dict(at=now(),reason='Reject17 failed recipes; retain383 validated recipes and equal25000 images per arm. Original fit and failed admission preserved.',training='Model1 A/B parallel, then cross ranks and Model2 A/B parallel; actual-data preflight mandatory',model4='Remain explicitly stopped; no automatic restart',changed_training_sources=sorted(allowed),original_code_and_recipe_hashes_verified=True))
for module,folder,label in [('capri_controller.cryptbb_v3_recovery',rec,'generation'),('capri_controller.training.cryptbb_v3_sequence',S,'coordinator_recovery')]:
 with (folder/(label+'.log')).open('a',encoding='utf8') as log:
  cmd=[sys.executable,'-B','-u','-m',module];p=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  row=dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),at=now(),command=cmd);write(folder/(label+'_launch.json'),row);print(row,flush=True)
