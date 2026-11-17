"""Authorized CryptBB v2 workflow: existing full fit -> new A/B corpora -> parallel readers."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from .decoder.common import read,write,sha,now,StageLock
ROOT=Path(__file__).resolve().parents[3];STUDY=ROOT/'handoff/cryptbb_v2_20261114'

def verify():
    for r in read(STUDY/'workflow_sources.json'):assert sha(ROOT/r['path'])==r['sha256'],r['path']
    if (STUDY/'STOP').exists():raise RuntimeError('Workflow STOP requested')

def child(module,args,label):
    verify();cmd=[sys.executable,'-B','-u','-m',module]+args
    write(STUDY/'workflow_status.json',dict(stage=label,state='running',updated_at=now()))
    with (STUDY/(label+'.log')).open('a',encoding='utf8') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        write(STUDY/'active_child.json',dict(stage=label,pid=p.pid,command=cmd,started_at=now()));code=p.wait()
    assert code==0,f'{label} failed ({code}); inspect its log'

def pair(stage):
    verify();children=[]
    try:
        for tag in 'AB':
            cmd=[sys.executable,'-B','-u','-m','capri_controller.training.cryptbb_v2_model1',stage,'--dataset',tag]
            log=(STUDY/f'{stage}_{tag}.log').open('a',encoding='utf8')
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);children.append((tag,p,log))
        write(STUDY/'parallel_children.json',dict(stage=stage,started_at=now(),children=[dict(dataset=t,pid=p.pid) for t,p,l in children]))
        write(STUDY/'workflow_status.json',dict(stage=stage,state='running_parallel',updated_at=now()))
        while any(p.poll() is None for t,p,l in children):
            failures=[(t,p.returncode) for t,p,l in children if p.poll() not in (None,0)]
            if failures:
                (STUDY/'STOP').write_text('Automatic stop after paired process failure: '+repr(failures))
                raise RuntimeError('Paired process failed: '+repr(failures)+'; other reader stops at next committed epoch')
            time.sleep(5)
        assert all(p.returncode==0 for t,p,l in children)
    finally:
        for t,p,l in children:l.close()

def run():
    with StageLock(STUDY,'workflow'):
        try:
            verify();write(STUDY/'workflow_status.json',dict(stage='controller_fit',state='waiting_for_completion',updated_at=now()))
            while not (STUDY/'fit_completion.json').exists():
                verify();s=read(STUDY/'fit_status.json');assert s['state']!='failed',s
                import psutil
                r=read(STUDY/'fit_launch.json');p=psutil.Process(r['pid']);assert abs(p.create_time()-r['process_created'])<1
                time.sleep(15)
            from .cryptbb_v2_corpus import validate_entries
            validate_entries()
            base=ROOT/'datasets/cryptbb_v2';base.mkdir(exist_ok=True)
            charset=ROOT/'datasets/cryptbb/charset.txt'
            if (base/'charset.txt').exists():assert sha(base/'charset.txt')==sha(charset)
            else:shutil.copyfile(charset,base/'charset.txt')
            for tag in 'AB':
                child('capri_controller.cryptbb_v2_corpus',['generate','--dataset',tag,'--workers','12'],'generate_'+tag)
                child('capri_controller.cryptbb_v2_corpus',['pack','--dataset',tag],'pack_'+tag)
            child('capri_controller.training.cryptbb_v2_model1',['prepare'],'prepare_readers')
            pair('probe')
            assert all(read(STUDY/'readiness'/t/'receipt.json')['passed'] for t in 'AB')
            write(STUDY/'parallel_readiness.json',dict(passed=True,completed_at=now(),scope='Actual v2 A/B datasets, both resident, full128 GPU steps serialized, recovery replay exact',readers=[read(STUDY/'readiness'/t/'receipt.json') for t in 'AB']))
            pair('worker')
            done=[read(ROOT/f'model1/runs/cryptbb_v2/{t}/training_completion.json') for t in 'AB']
            assert all(r['epoch']==20 for r in done)
            write(STUDY/'completion.json',dict(completed_at=now(),readers=done,selection_pending=True,later_models_started=False))
            write(STUDY/'workflow_status.json',dict(stage='complete',state='training_complete_selection_pending',updated_at=now()))
        except Exception as exc:
            write(STUDY/'workflow_status.json',dict(state='failed',error=repr(exc),updated_at=now()));raise

if __name__=='__main__':run()
