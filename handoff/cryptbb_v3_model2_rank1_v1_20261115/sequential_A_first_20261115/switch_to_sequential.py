"""Install a separate observer and replace only the paired coordinator."""
import ast,json,os,shutil,subprocess,sys
from pathlib import Path
import psutil
ROOT=Path('E:/pranav/Thesis-Pranav-Final');OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.decoder.common import read,write,sha,now
from capri_controller.training import cryptbb_v3_model2_selected_v1 as M
STUDY=M.STUDY;REC=STUDY/'sequential_A_first_20261115'
assert not REC.exists(),'Inspect existing switch before retry'
assert not (STUDY/'STOP').exists()
digest=M.check_sources();control={t:read(STUDY/f'worker_{t}.lock.json') for t in 'AB'}
control['coordinator']=read(STUDY/'coordinator.lock.json')
processes={}
for tag,row in control.items():
    p=psutil.Process(row['pid']);assert abs(p.create_time()-row['process_created'])<1
    cmd=p.cmdline();assert 'capri_controller.training.cryptbb_v3_model2_selected_v1' in cmd
    assert ('coordinate' in cmd) if tag=='coordinator' else ('worker' in cmd and tag in cmd)
    processes[tag]=p
src=OUT/'cryptbb_v3_model2_sequential.py';ast.parse(src.read_text())
dest=ROOT/'Capri/src/capri_controller/training'/src.name;assert not dest.exists()
REC.mkdir();shutil.copyfile(src,dest);shutil.copyfile(Path(__file__),REC/Path(__file__).name)
write(REC/'control.json',dict(control,created_at=now(),protocol_sha256=digest,source_sha256=sha(dest),
    intent='User requested one at a time; retain A; pause B at first committed epoch, then resume B after A completes',
    training_configuration_unchanged=True))
write(REC/'previous_status.json',read(STUDY/'status.json'))
# Coordinator owns no optimizer; neither training worker is touched here.
processes['coordinator'].terminate();processes['coordinator'].wait(timeout=30)
write(REC/'old_coordinator_replaced.json',dict(at=now(),identity=control['coordinator'],A_untouched=True,B_untouched=True))
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'Capri/src');env['PYTHONDONTWRITEBYTECODE']='1'
args=[sys.executable,'-B','-u','-m','capri_controller.training.cryptbb_v3_model2_sequential']
with (REC/'observer.log').open('a',encoding='utf8') as log:
    p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
write(REC/'observer_launch.json',dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),created_at=now(),args=args))
note=f'''## {now()}: User changed Model2 execution to one model at a time

Model2-A is first; B will pause at its first committed epoch and resume from that checkpoint after A completes20epochs. Separate observer cryptbb_v3_model2_sequential replaces only the exact paired coordinator identity. Existing worker code, frozen protocol, rankings and training settings remain unchanged; this execution-order override is recorded in handoff/cryptbb_v3_model2_rank1_v1_20261115/sequential_A_first_20261115. Inspect B_paused_receipt.json and A_only_verified.json before claiming A has exclusive training use. Until then B is finishing its current epoch to preserve optimizer/scheduler/RNG/sampler state. No Model3/4 launch.

'''
path=ROOT/'handoff/CURRENT_STATE.md';path.write_text(note+path.read_text(encoding='utf-8-sig'),encoding='utf8')
with (ROOT/'handoff/CHANGELOG.md').open('a',encoding='utf8') as f:f.write('\n'+note)
print(json.dumps(dict(observer_pid=p.pid,record=str(REC),state='B_pause_pending_checkpoint'),indent=2))
