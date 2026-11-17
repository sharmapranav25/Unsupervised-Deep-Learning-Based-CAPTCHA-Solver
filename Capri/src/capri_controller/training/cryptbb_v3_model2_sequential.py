"""Preserve existing workers/checkpoints; change only A/B execution order."""
import contextlib,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from ..decoder.common import read,write,sha,now,StageLock
from . import cryptbb_v3_model2_selected_v1 as M
ROOT=M.ROOT;STUDY=M.STUDY;REC=STUDY/'sequential_A_first_20261115'

def exact(row):
    p=psutil.Process(row['pid']);assert abs(p.create_time()-row['process_created'])<1
    return p

def alive(row):
    try:return exact(row).is_running()
    except psutil.NoSuchProcess:return False

def status(state,**extra):write(STUDY/'status.json',dict(state=state,execution='sequential_A_then_B',updated_at=now(),**extra))

@contextlib.contextmanager
def quiet_gpu_boundary():
    import msvcrt
    with (M.M1.STUDY/'gpu.lock').open('r+b') as f:
        while True:
            try:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(.025)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def run():
    control=read(REC/'control.json');a_dir=M.run_dir(2,'A');b_dir=M.run_dir(2,'B')
    with StageLock(STUDY,'coordinator'):
        try:
            digest=M.check_sources()
            status('waiting_for_B_committed_epoch_then_A_only',A_running=True,B_pause_requested=True)
            while not (b_dir/'history.json').exists():
                assert not (STUDY/'STOP').exists(),'Stop marker detected'
                exact(control['A']);exact(control['B']);time.sleep(.1)
            # history.json is written only after latest and the epoch archive commit.
            with quiet_gpu_boundary():
                b=exact(control['B']);b.suspend()
            try:
                import torch
                history=read(b_dir/'history.json');epoch=history[-1]['epoch']
                archive=b_dir/f'checkpoints/epoch_{epoch:02d}.pth'
                archive_sha=sha(archive);assert sha(b_dir/'latest.pth')==archive_sha
                ck=torch.load(archive,map_location='cpu',weights_only=False)
                assert ck['epoch']==epoch and ck['global_step']==epoch*782 and ck['protocol_sha256']==digest
                assert all(k in ck for k in ('state_dict','optimizer','lr_scheduler','scaler','rng','adaptive','config','history'))
                assert ck['config']['Global']['pretrained_model'] is None and ck['config']['Global']['checkpoints'] is None
                assert ck['adaptive']['epoch']==epoch+1
                del ck
                before=read(a_dir/'status.json');b.terminate();b.wait(timeout=30)
            except Exception:
                if alive(control['B']):b.resume()
                raise
            pause=dict(paused_at=now(),process=control['B'],epoch=epoch,global_step=epoch*782,
                checkpoint=str(archive.relative_to(b_dir)),sha256=archive_sha,protocol_sha256=digest,
                reason='User requested one model at a time; A first, B resumes after A completion',
                optimizer_scheduler_rng_sampler_preserved=True,A_interrupted=False)
            write(REC/'B_paused_receipt.json',pause);write(b_dir/'pause.json',pause)
            status('A_only_B_paused',B_saved_epoch=epoch,A_running=True,B_queued_after_A=True)
            while True:
                assert not (STUDY/'STOP').exists(),'Stop marker detected'
                exact(control['A']);progress=read(a_dir/'status.json')
                if progress.get('global_step',0)>before.get('global_step',0):break
                time.sleep(2)
            write(REC/'A_only_verified.json',dict(verified_at=now(),A_identity=control['A'],B_identity_absent=not alive(control['B']),A_before=before,A_after=progress,B_pause=pause))
            while not (a_dir/'completion.json').exists():
                assert not (STUDY/'STOP').exists(),'Stop marker detected'
                exact(control['A']);time.sleep(10)
            done=read(a_dir/'completion.json');assert done['epoch']==20 and done['global_step']==15640 and done['protocol_sha256']==digest
            assert sha(a_dir/done['checkpoint'])==done['sha256']
            while alive(control['A']):time.sleep(1)
            write(REC/'A_completion_verified.json',dict(verified_at=now(),completion=done))
            assert not alive(control['B']) and not (STUDY/'STOP').exists()
            assert sha(b_dir/'latest.pth')==archive_sha and M.check_sources()==digest
            args=[sys.executable,'-B','-u','-X','faulthandler','-m','capri_controller.training.cryptbb_v3_model2_selected_v1','worker','--dataset','B','--resume']
            with (REC/'B_resume.log').open('a',encoding='utf8') as log:
                child=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            write(REC/'B_resume_launch.json',dict(pid=child.pid,process_created=psutil.Process(child.pid).create_time(),args=args,created_at=now(),resume_from=pause))
            status('A_complete_B_resuming',A_completed_epoch=20,B_resume_epoch=epoch)
            verified=False
            while child.poll() is None:
                progress=read(b_dir/'status.json')
                if not verified and progress.get('global_step',0)>epoch*782:
                    identity=read(STUDY/'worker_B.lock.json');exact(identity)
                    write(REC/'B_resume_verified.json',dict(verified_at=now(),identity=identity,progress=progress))
                    status('A_complete_B_only',A_completed_epoch=20,B_running=True);verified=True
                time.sleep(5)
            assert child.returncode==0,('B worker failed',child.returncode)
            done=read(b_dir/'completion.json');assert done['epoch']==20 and done['protocol_sha256']==digest
            assert sha(b_dir/done['checkpoint'])==done['sha256']
            write(REC/'B_completion_verified.json',dict(verified_at=now(),completion=done))
            status('complete_model2_pair',model3_started=False)
        except Exception:
            status('attention_required',error=traceback.format_exc(),no_automatic_retries=True)
            raise

if __name__=='__main__':run()
