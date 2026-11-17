import ast,os,shutil,subprocess,sys
from pathlib import Path
import psutil
ROOT=Path('E:/pranav/Thesis-Pranav-Final');OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.decoder.common import read,write,sha,now
from capri_controller.training import cryptbb_v3_model2_selected_v1 as M
STUDY=ROOT/'handoff/cryptbb_v3_model2_measurements_20261115';assert not STUDY.exists()
digest=M.check_sources();rows=[]
for tag in 'AB':
    run=M.run_dir(2,tag);history=read(run/'history.json')
    for record in history:
        epoch=record['epoch'];ck=run/f'checkpoints/epoch_{epoch:02d}.pth';assert ck.exists()
        rows.append(dict(dataset=tag,epoch=epoch,checkpoint=str(ck.relative_to(ROOT)),sha256=sha(ck)))
assert rows
for name in ('cryptbb_v3_model2_checkpoint_evaluation.py','cryptbb_v3_model2_measurements.py'):
    src=OUT/name;ast.parse(src.read_text());dest=ROOT/'Capri/src/capri_controller/training'/name;assert not dest.exists()
    shutil.copyfile(src,dest)
STUDY.mkdir()
write(STUDY/'requested_checkpoints.json',dict(frozen_at=now(),scope='All completed checkpoints at request capture; no mutable latest files',training_protocol_sha256=digest,checkpoints=rows,device='cpu',threads=2,training_or_selection_changed=False))
for p in OUT.glob('*.py'):shutil.copyfile(p,STUDY/p.name)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
env['PYTHONPATH']=';'.join(str(ROOT/p) for p in ('Capri/src','model1/src/parseq_packages','model1/src/OpenOCR'))
env.update(TEMP=str(ROOT/'runtime/temporary'),TMP=str(ROOT/'runtime/temporary'),TORCH_HOME=str(ROOT/'runtime/cache/torch'),HF_HOME=str(ROOT/'runtime/cache/huggingface'),NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1')
args=[sys.executable,'-B','-u','-m','capri_controller.training.cryptbb_v3_model2_measurements']
with (STUDY/'batch.log').open('a',encoding='utf8') as log:
    p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
write(STUDY/'launch.json',dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),created_at=now(),args=args,checkpoint_count=len(rows)))
note=f'''\n\n## {now()}: User-requested Model2 checkpoint measurements\nFrozen current saved checkpoint pool ({len(rows)} archives) for CPU-only2thread diagnostics on251 CryptBB evaluation pixels. Separate adapters cryptbb_v3_model2_checkpoint_evaluation.py and cryptbb_v3_model2_measurements.py reuse established preprocessing/free AR/refinement and process-local CPU device fix. Checkpoint/manifest/pixel identities verified; predictions saved and hash-frozen before labels; exact whole-string accuracy and CER plus saved same-epoch Model1 comparison. Outputs: model2/results/cryptbb_v3_rank1_v1 and handoff/cryptbb_v3_model2_measurements_20261115. Diagnostics do not select checkpoints or modify training/curriculum. A retains GPU; B stays paused until A completion. Consult batch completion/status before claiming all measurements complete.\n'''
with (ROOT/'handoff/CHANGELOG.md').open('a',encoding='utf8') as f:f.write(note)
print(dict(pid=p.pid,checkpoints=len(rows),study=str(STUDY)))
