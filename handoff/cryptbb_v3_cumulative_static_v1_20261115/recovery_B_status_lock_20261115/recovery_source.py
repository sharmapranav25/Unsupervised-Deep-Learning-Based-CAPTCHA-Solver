"""One user-approved B recovery; frozen training code unchanged, bounded I/O retry only."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
ROOT=Path('E:/pranav/Thesis-Pranav-Final')
sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.training import cryptbb_cumulative_static as M
M.bind()
REC=M.STUDY/'recovery_B_status_lock_20261115'
ORIGINAL_WRITE=M.write


def retry_write(path,value):
    for attempt in range(50):
        try: return ORIGINAL_WRITE(path,value)
        except PermissionError:
            if attempt==49: raise
            time.sleep(.1)


M.write=retry_write;M.C.write=retry_write


def identity(pid):
    p=psutil.Process(pid)
    return dict(pid=pid,process_created=p.create_time(),cmdline=p.cmdline())


def verify_idle():
    for name in ['coordinator','training_worker','worker_B','evaluation_report']:
        path=M.STUDY/(name+'.lock.json')
        if path.exists():
            r=M.read(path)
            if psutil.pid_exists(r['pid']):
                assert abs(psutil.Process(r['pid']).create_time()-r['process_created'])>.1,(name,r)
    matches=[]
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            cmd=p.info['cmdline'] or []
            if M.MODULE in cmd and any(x in cmd for x in ['coordinate','worker','evaluate']): matches.append(p.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    assert not matches,matches


def launch_child(stage):
    with (REC/(stage+'.log')).open('a',encoding='utf8') as f:
        p=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),stage],cwd=ROOT,
            stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    M.write(REC/(stage+'_launch.json'),identity(p.pid));return p


def annotate_report(base):
    complete=base()
    result=M.read(M.STUDY/'results.json')
    result['user_authorized_restarts']=1
    result['recovery']=M.read(REC/'preflight.json')
    M.write(M.STUDY/'results.json',result)
    report=M.STUDY/'MORNING_REPORT.md'
    with report.open('a',encoding='utf8') as f:
        f.write('\nRecovery: B stopped at05:49 Pacific after a Windows status-file rename denial. Epoch3 was fully committed; partial epoch4 was discarded. User authorized one resume from epoch3, preserving all training state. A remained complete. Frozen sources unchanged; recovery wrapper retries PermissionError on JSON writes for up to5seconds. Failure evidence: recovery_B_status_lock_20261115. User-authorized restarts:1.\n')
    verification=M.read(M.STUDY/'report_verification.json')
    verification.update(report_sha256=M.sha(report),results_sha256=M.sha(M.STUDY/'results.json'),user_authorized_restarts=1)
    M.write(M.STUDY/'report_verification.json',verification)
    return complete


def preflight_launch():
    verify_idle(); assert not REC.exists(),'Recovery already attempted; no blind retry'
    assert (M.STUDY/'STOP').read_text()=='Coordinator failure; inspect before any restart.\n'
    digest=M.check_sources();M.OLD.check_sources();M.valid_completion('A')
    run=M.run_dir(2,'B');r=M.read(run/'checkpoints/epoch_03.json');ckpath=run/r['checkpoint']
    assert r['epoch']==3 and r['global_step']==2346 and r['protocol_sha256']==digest and M.sha(ckpath)==r['sha256']
    import torch
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    assert ck['epoch']==3 and ck['global_step']==2346 and ck['protocol_sha256']==digest
    assert ck['adaptive']['epoch']==4 and [h['epoch'] for h in ck['history']]==[1,2,3]
    assert all(k in ck for k in ['state_dict','optimizer','lr_scheduler','scaler','rng','adaptive'])
    assert all(k in ck['rng'] for k in ['python','numpy','torch','cuda','modules'])
    del ck
    # Retry changes no payload or destination; verify bounded transient behavior.
    global ORIGINAL_WRITE
    original=ORIGINAL_WRITE;calls=[]
    def transient(path,value):
        calls.append((path,value))
        if len(calls)<3:raise PermissionError('simulated sharing violation')
        return 'passed'
    ORIGINAL_WRITE=transient
    assert retry_write('probe',{'x':1})=='passed' and len(calls)==3
    ORIGINAL_WRITE=original
    REC.mkdir();shutil.copyfile(__file__,REC/'recovery_source.py')
    for name in ['status.json','evaluation_failure.json','coordinator.stderr.log','worker_B.log']:
        p=M.STUDY/name
        if p.exists():shutil.copyfile(p,REC/name)
    pending=run/'status.json.pending'
    if pending.exists():shutil.copyfile(pending,REC/'B_status_pending.json')
    M.write(REC/'preflight.json',dict(verified_at=M.now(),authorization='User: Resume B and evaluations',
        cause='Windows PermissionError replacing B status.json at first batch of epoch4',
        resume_epoch=3,checkpoint_sha256=r['sha256'],protocol_sha256=digest,
        full_state_verified=True,frozen_sources_unchanged=True,retry_test_passed=True,
        recovery_source_sha256=M.sha(Path(__file__)),automatic_retries=0,user_authorized_recovery=1))
    # Preserve the coordinator-created marker after explicit user authorization.
    (M.STUDY/'STOP').rename(REC/'STOP_before_authorized_resume')
    if (M.STUDY/'evaluation_failure.json').exists():
        (M.STUDY/'evaluation_failure.json').rename(REC/'evaluation_failure_original.json')
    M.write(M.STUDY/'status.json',dict(state='recovery_starting',dataset='B',updated_at=M.now()))
    launch_child('coordinate')
    print(json.dumps(M.read(REC/'preflight.json'),indent=2),flush=True)


def coordinate():
    worker=evaluator=None
    with M.StageLock(M.STUDY,'coordinator'):
        assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            M.stop_check();M.valid_completion('A')
            evaluator=launch_child('evaluate');worker=launch_child('worker')
            M.write(M.STUDY/'status.json',dict(state='training',dataset='B',recovery=1,updated_at=M.now()))
            previous=2346;verified=False
            while worker.poll() is None:
                M.stop_check()
                row=M.read(M.run_dir(2,'B')/'status.json')
                if row.get('global_step',0)>previous:
                    if not verified:
                        M.write(REC/'resume_progress_first.json',row);previous=row['global_step'];verified=True
                    elif not (REC/'resume_verified.json').exists():
                        ident=M.read(M.STUDY/'worker_B.lock.json');p=psutil.Process(ident['pid'])
                        assert abs(p.create_time()-ident['process_created'])<.1
                        M.write(REC/'resume_verified.json',dict(verified_at=M.now(),worker=ident,
                            first=M.read(REC/'resume_progress_first.json'),progress=row,
                            protocol_sha256=M.check_sources(),restored_checkpoint_epoch=3))
                if evaluator.poll() is not None and evaluator.returncode!=0:
                    M.write(REC/'evaluation_failed.json',dict(exit_code=evaluator.returncode,updated_at=M.now()))
                time.sleep(5)
            assert worker.returncode==0,('B recovery failed',worker.returncode)
            M.valid_completion('B')
            M.write(M.STUDY/'training_pair_complete.json',dict(completed_at=M.now(),A=M.valid_completion('A'),B=M.valid_completion('B'),user_authorized_restarts=1))
            while evaluator.poll() is None:M.stop_check();time.sleep(5)
            assert evaluator.returncode==0
            assert M.read(M.STUDY/'report_verification.json')['complete']
            M.write(M.STUDY/'status.json',dict(state='complete',updated_at=M.now(),user_authorized_restarts=1))
        except BaseException:
            M.write(M.STUDY/'status.json',dict(state='failed',error=traceback.format_exc(),updated_at=M.now(),recovery_attempt=1))
            if not (M.STUDY/'STOP').exists():(M.STUDY/'STOP').write_text('Recovery failed; diagnose without another automatic restart.\n')
            raise
        finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='launch':preflight_launch()
    elif stage=='worker':
        sys.addaudithook(M.no_labels);M.C.bootstrap();M.worker('B',resume=True)
    elif stage=='coordinate':coordinate()
    elif stage=='evaluate':
        from capri_controller.training import cryptbb_cumulative_report as R
        base=R.report;R.report=lambda:annotate_report(base);R.evaluate(watch=True)
    else:raise ValueError(stage)
