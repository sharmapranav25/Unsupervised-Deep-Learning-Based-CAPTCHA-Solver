"""One bounded restart after a native CUDA failure; preserve original training protocol."""
import argparse,faulthandler,os,subprocess,sys,time
from pathlib import Path
import psutil
from ..decoder.common import read,write,sha,now,StageLock
from . import cryptbb_v3_sequence as Q
ROOT=Q.ROOT;STUDY=Q.STUDY;REC=STUDY/'native_recovery_20261114'
def verify():
    Q.verify()
    for r in read(REC/'sources.json'):assert sha(ROOT/r['path'])==r['sha256'],r['path']
def worker(tag):
    faulthandler.enable();verify()
    from . import cryptbb_v3_model1 as M
    write(REC/f'{tag}_worker.json',dict(pid=os.getpid(),at=now()))
    M.worker(tag,resume=(ROOT/f'model1/runs/cryptbb_v3/{tag}/latest.pth').exists())
def pair():
    group=[]
    def launch(tag):
        verify();log=(REC/f'worker_{tag}.log').open('a',encoding='utf8')
        cmd=[sys.executable,'-B','-X','faulthandler','-u','-m','capri_controller.training.cryptbb_v3_resume','--worker',tag]
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        group.append((tag,p,log));write(REC/f'{tag}_launch.json',dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),at=now(),command=cmd))
        return p
    try:
        Q.state('model1_recovery','starting_A_before_B',training_started=True)
        a=launch('A');deadline=time.time()+300
        while True:
            assert a.poll() is None, f'A failed before paired startup: {a.returncode}'
            verify();path=ROOT/'model1/runs/cryptbb_v3/A/progress.json'
            if path.exists() and (REC/'A_worker.json').exists():
                r=read(path)
                if r['pid']==read(REC/'A_worker.json')['pid'] and r['optimizer_steps']>=50:break
            assert time.time()<deadline,'A did not reach50 updates in5minutes'
            time.sleep(3)
        launch('B');Q.state('model1_recovery','running_parallel',training_started=True,A='epoch1 scratch recovery',B='resumed epoch1 checkpoint')
        failed=False
        while any(p.poll() is None for t,p,l in group):
            user_stop=(STUDY/'STOP').exists()
            errors={t:p.poll() for t,p,l in group if p.poll() not in (None,0)}
            if errors or user_stop:
                failed=True
                if not user_stop:(STUDY/'STOP').write_text('Automatic stop after native recovery child failure; preserve committed checkpoints.\n')
                Q.state('model1_recovery','stopping_after_failure_or_request',exit_codes={t:p.poll() for t,p,l in group})
            time.sleep(5)
        assert not failed and all(p.returncode==0 for t,p,l in group), [(t,p.returncode) for t,p,l in group]
    finally:
        # If startup fails, signal any still-running peer to stop at its committed epoch.
        if any(p.poll() is None for t,p,l in group) and not (STUDY/'STOP').exists():
            (STUDY/'STOP').write_text('Automatic stop after native recovery coordinator failure; preserve committed checkpoints.\n')
        for t,p,l in group:l.close()
def run():
    with StageLock(STUDY,'coordinator'):
        try:
            verify();pair();verify()
            m1={t:read(ROOT/f'model1/runs/cryptbb_v3/{t}/training_completion.json') for t in 'AB'};assert all(r['epoch']==20 for r in m1.values())
            for stage,paired in [('setup',False),('rank',True),('readiness',False),('worker',True)]:
                verify();Q.children('capri_controller.training.cryptbb_v3_model2',stage,paired)
            m2={t:read(ROOT/f'model2/runs/cryptbb_v3/{t}/completion.json') for t in 'AB'};assert all(r['epoch']==20 for r in m2.values())
            write(STUDY/'completion.json',dict(completed_at=now(),model1=m1,model2=m2,reporting_selection_pending=True,model3_started=False))
            Q.state('complete','model1_and_model2_complete_selection_pending');Q.release_teacher_queue()
        except Exception as exc:
            Q.state('native_recovery','failed',error=repr(exc),automatic_retry_exhausted=True);raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['A','B']);a=ap.parse_args()
    if a.worker:worker(a.worker)
    else:run()
