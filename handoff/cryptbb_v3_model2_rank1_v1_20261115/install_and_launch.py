"""Install the reviewed selected-teacher adapter and start its gated coordinator."""
import ast,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import psutil
ROOT=Path('E:/pranav/Thesis-Pranav-Final');OUT=Path(__file__).parent
STUDY=ROOT/'handoff/cryptbb_v3_model2_rank1_v1_20261115'
DEST=ROOT/'Capri/src/capri_controller/training/cryptbb_v3_model2_selected_v1.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
assert not DEST.exists() and not STUDY.exists(),'Preserve existing deployment; inspect before retry'
for path in ('model2/runs/cryptbb_v3','model2/runs/cryptbb_v3_rank1_v1','model2/rankings/cryptbb_v3','model2/rankings/cryptbb_v3_rank1_v1'):
    assert not (ROOT/path).exists(),path
processes=[]
for p in psutil.process_iter(['pid','name','cmdline','create_time']):
    if 'python' not in (p.info['name'] or '').lower():continue
    cmd=' '.join(p.info['cmdline'] or [])
    if 'install_and_launch.py' in cmd:continue
    processes.append(p.info)
    assert not any(x in cmd for x in ('cryptbb_v3','ab_curriculum','training_full_gpu','run_ab_study')),cmd
old=ast.parse((DEST.with_name('cryptbb_v3_model2.py')).read_text())
new=ast.parse((OUT/DEST.name).read_text())
def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
a,b=functions(old),functions(new)
assert all(a[n]==b[n] for n in ('config','dataset','rank')),'Core configuration or ranking changed'
STUDY.mkdir()
shutil.copyfile(OUT/DEST.name,DEST)
for name in ('teacher_selection.json','verify_selection.py','prepare_adapter.py','install_and_launch.py'):
    shutil.copyfile(OUT/name,STUDY/name)
write(STUDY/'deployment.json',dict(created_at=now(),source=str(DEST.relative_to(ROOT)),sha256=sha(DEST),selection_sha256=sha(STUDY/'teacher_selection.json'),original_adapter_sha256=sha(DEST.with_name('cryptbb_v3_model2.py')),processes_before=processes,core_config_dataset_rank_ast_unchanged=True))
note=f'''\n\n## {now()}: CryptBB v3 Model2 selected cross-teachers authorized\nOriginal v1 masked rank-one reproduced on epochs1-20/251 unlabeled predictions independently: current Model1-A12 and B13. Both use fallback k5 after all automatic criteria failed; scores are not accuracy estimates. Actual selected checkpoint bytes rehashed and matched frozen predictions/epoch receipts. New teacher-selection manifest and separate adapter: handoff/cryptbb_v3_model2_rank1_v1_20261115; Capri/src/capri_controller/training/cryptbb_v3_model2_selected_v1.py. B13 ranks A, A12 ranks B. Original adapter, protocols and snapshots preserved. Fresh rankings/runs use model2/{{rankings,runs}}/cryptbb_v3_rank1_v1. Core configuration, dataset and ranking functions match original AST; scratch initialization, seed48, batch128, 20epochs, 100096 draws/epoch, fixed curriculum, preprocessing and optimizer retained. Gated coordinator runs setup, both rankings, sampler/preprocessing/resume/pair pilots, then workers. See status.json and launch_verified.json for actual progress; deployment alone is not launch validation. No Model3/4 or cancelled evaluation is launched. Explicit maintained entry: python -B -m capri_controller.training.cryptbb_v3_model2_selected_v1 coordinate.\n'''
for p in (ROOT/'handoff/CHANGELOG.md',ROOT/'handoff/CURRENT_STATE.md'):
    with p.open('a',encoding='utf8') as f:f.write(note)
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'Capri/src');env['PYTHONDONTWRITEBYTECODE']='1'
args=[sys.executable,'-B','-u','-X','faulthandler','-m','capri_controller.training.cryptbb_v3_model2_selected_v1','coordinate']
with (STUDY/'coordinator.log').open('a',encoding='utf8') as log:
    p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
write(STUDY/'coordinator_launch.json',dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),created_at=now(),args=args))
print(json.dumps(dict(pid=p.pid,study=str(STUDY),deployment_sha256=sha(STUDY/'deployment.json')),indent=2))
