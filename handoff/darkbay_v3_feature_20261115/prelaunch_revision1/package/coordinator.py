"""Durable fresh-data -> parallel readers coordinator; heartbeat diagnoses stalls."""
import argparse,ctypes,json,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from . import common as W
from .common import ROOT,STUDY,SOURCE,FIT,DATA,FAMILY,read,write,sha,now
from ...decoder.common import StageLock

def launch_child(module,args,name):
    W.stop();W.verify_sources();cmd=[sys.executable,'-B','-u','-m',__package__+'.'+module,*args]
    path=STUDY/'logs'/f'{name}.log';path.parent.mkdir(exist_ok=True)
    with path.open('a',encoding='utf8') as log:p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    ident=dict(name=name,pid=p.pid,process_created=psutil.Process(p.pid).create_time(),args=cmd,started_at=now(),log=str(path.relative_to(ROOT)))
    write(STUDY/'launches'/f'{name}_{time.time_ns()}.json',ident)
    return dict(process=p,identity=ident,log=path,start=time.monotonic(),last_progress=time.monotonic(),signature=None)

def supervise(jobs,progress_paths):
    while True:
        W.stop();active=[];failed=[]
        for job,paths in zip(jobs,progress_paths):
            proc=job['process'];code=proc.poll();stamp=[]
            for path in paths:
                if path.exists():
                    st=path.stat();stamp.append((str(path),st.st_size,st.st_mtime_ns))
            # Logs are progress evidence for preparatory/CPU stages too.
            st=job['log'].stat();stamp.append(('log',st.st_size,st.st_mtime_ns))
            signature=tuple(stamp)
            if signature!=job['signature']:job['last_progress']=time.monotonic();job['signature']=signature
            stagnant=time.monotonic()-job['last_progress']
            row=dict(**job['identity'],exit_code=code,no_progress_seconds=round(stagnant,1),elapsed_seconds=round(time.monotonic()-job['start'],1))
            if code is None:
                active.append(row)
                if stagnant>600:write(STUDY/'attention.json',dict(reason='No observable progress for ten minutes',worker=row,observed_at=now(),action='Heartbeat must diagnose exact process; do not blindly kill or restart'))
            elif code:failed.append(row)
        write(STUDY/'live_monitor.json',dict(observed_at=now(),coordinator_pid=os.getpid(),active=active,failed=failed,state='running' if active else 'failed' if failed else 'stage_complete',STOP_present=False))
        if failed:
            # Keep healthy peer running; heartbeat can recover failed peer separately.
            write(STUDY/'failure.json',dict(at=now(),workers=failed,healthy_workers=active,action='Inspect failure; preserve artifacts; bounded full-state recovery authorized'))
            raise RuntimeError('Child failed: '+','.join(x['name'] for x in failed))
        if not active:return
        time.sleep(5)

def one(module,args,name,paths=()):
    job=launch_child(module,args,name);supervise([job],[list(paths)])

def paired(stage,model=None):
    jobs=[];paths=[]
    for tag in 'AB':
        if stage=='model1-worker' and (W.rundir(1,tag)/'training_completion.json').exists():
            r=read(W.rundir(1,tag)/'training_completion.json');assert r['epoch']==20 and sha(W.rundir(1,tag)/r['checkpoint'])==r['sha256'];continue
        if stage=='worker' and (W.rundir(model,tag)/'completion.json').exists():
            r=read(W.rundir(model,tag)/'completion.json');assert r['epoch']==20 and sha(W.rundir(model,tag)/r['checkpoint'])==r['sha256'];continue
        args=[stage,'--dataset',tag]
        if model:args+=['--model',str(model)]
        if stage=='model1-worker' and (W.rundir(1,tag)/'latest_receipt.json').exists():args+=['--resume']
        if stage=='worker' and list((W.rundir(model,tag)/'checkpoints').glob('epoch_*.json')):args+=['--resume']
        jobs.append(launch_child('readers',args,f'{stage}_M{model or 1}_{tag}'))
        paths.append([W.rundir(model or 1,tag)/'status.json',W.rundir(model or 1,tag)/'progress.json',STUDY/'rankings'/tag/'progress.json'])
    if jobs:supervise(jobs,paths)

def await_previous():
    prior=ROOT/'handoff/cryptbb_v3_feature_priority_B_v1_20261115'
    while True:
        W.stop();p=read(prior/'live_monitor.json');done=(ROOT/'model3/runs/cryptbb_v3_feature_priority_B_v1_20261115/B/completion.json')
        if done.exists():
            r=read(done);assert r['epoch']==20 and r['global_step']==15640
            assert sha(done.parent/r['checkpoint'])==r['sha256']
            lock=prior/'worker_B.lock.json'
            if not lock.exists():break
        if p.get('worker_alive') is False and not done.exists():raise RuntimeError('Prior CryptBB B stopped without completion; diagnose before claiming GPU is available')
        write(STUDY/'live_monitor.json',dict(observed_at=now(),state='waiting_for_cryptbb_B_completion',prior=p,coordinator_pid=os.getpid(),STOP_present=False))
        time.sleep(15)
    write(STUDY/'prior_run_completed.json',dict(verified_at=now(),completion=r))

def run():
    W.environment();W.verify_sources()
    with StageLock(STUDY,'coordinator'):
        awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);assert awake
        try:
            if not (STUDY/'reference_split.json').exists():one('generation',['references'],'verify_references')
            await_previous()
            if not (SOURCE/'status.json').exists():one('generation',['source'],'fresh_reference_recipes',[SOURCE/'status.json'])
            assert read(SOURCE/'status.json')['state']=='complete'
            if not (FIT/'status.json').exists():one('generation',['fit'],'v3_split_fit',[FIT/'status.json'])
            assert read(FIT/'status.json')['state']=='complete'
            if not (STUDY/'accepted_recipe_protocol.json').exists():one('generation',['admit'],'bounded_recipe_admission')
            # Generation CPU workers are bounded; B excludes A's final pixels.
            for tag in 'AB':
                if not (DATA/tag/'corpus_summary.json').exists():one('corpus',['generate','--dataset',tag,'--workers','12'],'generate_'+tag,[DATA/tag/'corpus_status.json'])
                if not (DATA/tag/'data_receipt.json').exists():one('corpus',['pack','--dataset',tag],'pack_'+tag,[DATA/tag/'sample_provenance.jsonl'])
            if not (STUDY/'datasets_ready.json').exists():one('generation',['configs'],'verify_data_and_configs')
            ready=[STUDY/f'model1/readiness/{tag}/receipt.json' for tag in 'AB']
            if not all(p.exists() and read(p)['passed'] for p in ready):paired('model1-check')
            paired('model1-worker')
            paired('rank')
            one('readers',['prepare'],'freeze_curricula')
            for tag in 'AB':
                if not all((W.stage(n,tag)/'readiness.json').exists() and read(W.stage(n,tag)/'readiness.json')['passed'] for n in (2,3)):
                    one('readers',['curriculum-check','--dataset',tag],'curriculum_checks_'+tag)
            paired('worker',2);paired('worker',3)
            one('evaluate',['all'],'all_checkpoint_evaluations')
            write(STUDY/'completion.json',dict(completed_at=now(),models=[1,2,3],datasets=['A','B'],epochs_each=20,report=str(STUDY/'results.json')))
            write(STUDY/'live_monitor.json',dict(state='complete',observed_at=now(),coordinator_pid=os.getpid()))
        except BaseException:
            write(STUDY/'coordinator_failure.json',dict(at=now(),error=traceback.format_exc(),STOP_present=(STUDY/'STOP').exists(),recovery='User authorized bounded diagnosis/recovery; no source changes or blind restart'))
            raise
        finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    W.environment()
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['launch','run','inventory']);a=p.parse_args()
    if a.stage=='run':return run()
    if a.stage=='inventory':
        from ...inventory import refresh
        return refresh()
    W.verify_sources();W.stop();lock=STUDY/'coordinator.lock.json'
    if lock.exists():
        r=read(lock)
        if psutil.pid_exists(r['pid']) and abs(psutil.Process(r['pid']).create_time()-r['process_created'])<1:raise RuntimeError('Coordinator already active')
    job=launch_child('coordinator',['run'],'coordinator')
    write(STUDY/'coordinator_launch.json',job['identity']);print(json.dumps(job['identity']),flush=True)
if __name__=='__main__':main()
