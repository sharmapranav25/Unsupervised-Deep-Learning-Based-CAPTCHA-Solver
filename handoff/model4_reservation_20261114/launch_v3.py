from pathlib import Path
import json,hashlib,shutil,subprocess,sys,os,ast
import psutil
from datetime import datetime,timezone
ROOT=Path.cwd();S=ROOT/'handoff/cryptbb_v3_20261114';S.mkdir(exist_ok=False)
SPLIT=ROOT/'evaluation_data/model4_reservation_20261114'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
assert sha(ROOT/'Capri/src/capri_controller/controller_v3_split.py')=='87993051b97a49704b03d0fe912b967a20913e8ab1f780ed622144deb1a92097'
assert read(SPLIT/'verification.json')['passed']
refs=read(SPLIT/'cryptbb/reference_pool.json');pseudo=read(SPLIT/'cryptbb/pseudo_label_pool.json');test=read(SPLIT/'cryptbb/evaluation_pool.json')
assert [len(x) for x in [refs,pseudo,test]]==[100,300,251]
sets=[{r['pixel_sha256'] for r in x} for x in [refs,pseudo,test]];assert sum(map(len,sets))==len(set.union(*sets))
write(S/'targets.json',{'fit':[{'source':str(ROOT/r['image_path']),'pixel_sha256':r['pixel_sha256']} for r in refs]})
selection=read(ROOT/'Capri/runs/cryptbb_model1_preparation_20261113/selected_generator/completion.json');assert sha(Path(selection['checkpoint']))==selection['sha256'];write(S/'selected_generator/completion.json',selection)
write(S/'authorization.json',{'user_request':'Start CryptBB dataset generation after reserving 100 reference and 300 Model4 images for each family','images_each':100000,'references':100,'arms':['custom','dino','mixed','synthtiger'],'training_authorized':False,'old_v2_stays_stopped':True})
write(S/'run_protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),'controller':'v3 split','reference_manifest_sha256':sha(SPLIT/'cryptbb/reference_pool.json'),'new_reference_set':True,'reference_origin':'100 historical selftrain images from labeled651, with opaque names and labels isolated; no label access by generator','pseudo_label_pool':300,'evaluation_pool':251,'images_each':100000,'entries':400,'images_per_entry':250,'images_per_arm':25000,'render_seed_bases':{'A':50000000000,'B':60000000000},'calibration':'Native calibration on first24 freshly inferred reference recipes; replaces historical reference-derived calibration','fresh_count_vote':'Unique model-derived majority >=50/100; no manual family length','entry_gate':'All400 entries must validate; otherwise stop and report IDs','selection':'No readers launched, no checkpoint selection in this data-only workflow','old_evidence_preserved':True})
files=set((ROOT/'Capri/src/capri_controller').rglob('*.py'))
files.update(ROOT/x for x in ['Capri/models/registry.json','Capri/configs/local.json','datasets/cryptbb/charset.txt','Capri/runs/cryptbb_model1_preparation_20261113/excluded_pixels.json'])
files.update(SPLIT/'cryptbb'/f'{role}.json' for role in ['reference_pool','pseudo_label_pool','evaluation_pool'])
files.update([SPLIT/'protocol.json',SPLIT/'verification.json',S/'targets.json',S/'selected_generator/completion.json',S/'authorization.json',S/'run_protocol.json'])
frozen=[]
for p in sorted(files):
 rel=p.relative_to(ROOT);dest=S/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);digest=sha(p);assert sha(dest)==digest;frozen.append({'path':rel.as_posix(),'sha256':digest})
write(S/'workflow_sources.json',frozen)
cmd=[sys.executable,'-B','-u','-m','capri_controller.cryptbb_v3_workflow']
with (S/'workflow.log').open('a',encoding='utf-8') as log:p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
receipt={'pid':p.pid,'process_created':psutil.Process(p.pid).create_time(),'started_at':datetime.now(timezone.utc).isoformat(),'command':cmd};write(S/'workflow_launch.json',receipt)
print(json.dumps({'launched':receipt,'frozen_source_files':len(frozen),'splits':read(SPLIT/'protocol.json')['counts']},indent=2))
