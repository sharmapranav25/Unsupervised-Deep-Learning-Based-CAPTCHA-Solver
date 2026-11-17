"""Wait for validated CryptBB v3 data; Model1 pair -> cross rankings -> Model2 pair."""
import os,subprocess,sys,time
from pathlib import Path
import psutil
from ..decoder.common import read,write,sha,now,StageLock
ROOT=Path(__file__).resolve().parents[4];STUDY=ROOT/'handoff/cryptbb_v3_training_20261114';DATA=ROOT/'handoff/cryptbb_v3_20261114'

def state(stage,status,**kw):write(STUDY/'status.json',dict(stage=stage,state=status,updated_at=now(),**kw))
def verify():
    assert not (STUDY/'STOP').exists(),'STOP requested'
    for r in read(STUDY/'launch_sources.json'):assert sha(ROOT/r['path'])==r['sha256'],r['path']

def children(module,stage,pair=False):
    verify();group=[];label=module.rsplit('.',1)[-1]+'_'+stage
    state(label,'running_parallel' if pair else 'running')
    try:
        for tag in ('A','B') if pair else (None,):
            args=[stage]+(['--dataset',tag] if tag else [])
            cmd=[sys.executable,'-B','-u','-m',module]+args
            log=(STUDY/(label+('_'+tag if tag else '')+'.log')).open('a',encoding='utf-8')
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);group.append((tag,p,log))
        write(STUDY/'active_children.json',dict(stage=label,children=[dict(dataset=t,pid=p.pid,process_created=psutil.Process(p.pid).create_time()) for t,p,l in group],started_at=now()))
        failed=False
        while any(p.poll() is None for t,p,l in group):
            if (STUDY/'STOP').exists() or any(p.poll() not in (None,0) for t,p,l in group):
                failed=True
                (STUDY/'STOP').write_text('Stop after requested stop or child failure; preserve committed checkpoints.\n')
                (STUDY/'model2').mkdir(exist_ok=True);(STUDY/'model2/STOP').write_text('Sibling failure or user stop; stop at next committed epoch.\n')
                state(label,'stopping_after_failure_or_request',exit_codes={str(t):p.poll() for t,p,l in group})
            time.sleep(5)
        assert not failed and all(p.returncode==0 for t,p,l in group),[(t,p.returncode) for t,p,l in group]
    finally:
        for t,p,l in group:l.close()
    verify()

def release_teacher_queue():
    hold=read(STUDY/'teacher_queue_hold.json');marker=ROOT/hold['marker']
    if not marker.exists() or sha(marker)!=hold['sha256']:
        write(STUDY/'teacher_queue_release.json',dict(released=False,reason='Hold marker changed or removed externally; preserve current user intent'));return
    folder=marker.parent;lock=folder/'queue.lock.json';alive=False
    if lock.exists():
        r=read(lock)
        try:alive=abs(psutil.Process(r['pid']).create_time()-r['create_time'])<1
        except psutil.NoSuchProcess:pass
    marker.rename(folder/'STOP.completed_cryptbb_priority')
    if alive:
        write(STUDY/'teacher_queue_release.json',dict(released=True,existing_queue_alive=True,at=now()));return
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(str(ROOT/p) for p in ['model4/src','model4/shims','model4/site'])+os.pathsep+env.get('PYTHONPATH','');env['PYTHONUNBUFFERED']='1'
    with (folder/'queue_after_cryptbb.log').open('a',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-B','-u','-m','model4_teachers.gpu_queue'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    write(STUDY/'teacher_queue_release.json',dict(released=True,pid=p.pid,process_created=psutil.Process(p.pid).create_time(),at=now()))

def run():
    with StageLock(STUDY,'coordinator'):
        try:
            verify();state('datasets','waiting_for_validated_data')
            while not (DATA/'completion.json').exists():
                verify();s=read(DATA/'workflow_status.json')
                status='waiting_for_dataset_repair' if s.get('state')=='failed' else 'waiting_for_validated_data'
                state('datasets',status,generation=s,training_started=False)
                if s.get('state')=='stopped_by_user':raise RuntimeError('Generation stopped by user; do not bypass stop')
                time.sleep(30)
            done=read(DATA/'completion.json');assert all(r['images']==100000 and r['unique_pixels']==100000 for r in done['datasets'].values())
            from ..cryptbb_v3_corpus import validate_entries
            validate_entries()
            # Teacher queue is held; no third GPU workload competes with this pair.
            hold=read(STUDY/'teacher_queue_hold.json');assert sha(ROOT/hold['marker'])==hold['sha256']
            children('capri_controller.training.cryptbb_v3_model1','prepare')
            children('capri_controller.training.cryptbb_v3_model1','probe',True)
            probes=[read(STUDY/f'readiness/{t}/receipt.json') for t in 'AB'];assert all(r['passed'] for r in probes)
            write(STUDY/'parallel_readiness.json',dict(passed=True,readers=probes,completed_at=now()))
            children('capri_controller.training.cryptbb_v3_model1','worker',True)
            m1={t:read(ROOT/f'model1/runs/cryptbb_v3/{t}/training_completion.json') for t in 'AB'};assert all(r['epoch']==20 for r in m1.values())
            children('capri_controller.training.cryptbb_v3_model2','setup')
            children('capri_controller.training.cryptbb_v3_model2','rank',True)
            children('capri_controller.training.cryptbb_v3_model2','readiness')
            children('capri_controller.training.cryptbb_v3_model2','worker',True)
            m2={t:read(ROOT/f'model2/runs/cryptbb_v3/{t}/completion.json') for t in 'AB'};assert all(r['epoch']==20 for r in m2.values())
            write(STUDY/'completion.json',dict(completed_at=now(),model1=m1,model2=m2,reporting_selection_pending=True,model3_started=False))
            state('complete','model1_and_model2_complete_selection_pending')
            release_teacher_queue()
        except Exception as exc:state('stopped','stopped_by_user' if (STUDY/'STOP').exists() else 'failed',error=repr(exc));raise
if __name__=='__main__':run()
