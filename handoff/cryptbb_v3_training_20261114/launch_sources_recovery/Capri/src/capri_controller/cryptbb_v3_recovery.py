"""Resume completed fit with accepted recipes; preserve rejected evidence and full image audit."""
import os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
from .decoder.common import read,write,sha,now,StageLock
from .cryptbb_v3_accepted_corpus import ROOT,STUDY,validate_entries
RECOVERY=STUDY/'recovery_20261114'
def verify():
    assert not (STUDY/'STOP').exists(), 'User STOP requested'
    for name in ['workflow_sources.json','recovery_20261114/sources.json']:
        for r in read(STUDY/name):assert sha(ROOT/r['path'])==r['sha256'],r['path']
def child(stage,tag):
    verify();label=stage+'_'+tag
    cmd=[sys.executable,'-B','-u','-m','capri_controller.cryptbb_v3_accepted_corpus',stage,'--dataset',tag]+(['--workers','12'] if stage=='generate' else [])
    write(STUDY/'workflow_status.json',dict(stage=label,state='running',updated_at=now(),accepted_recipes=383,rejected_recipes=17,training_started=False))
    with (RECOVERY/(label+'.log')).open('a',encoding='utf8') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        receipt=dict(stage=label,pid=p.pid,process_created=psutil.Process(p.pid).create_time(),command=cmd,started_at=now())
        write(STUDY/'active_child.json',receipt);write(RECOVERY/(label+'_launch.json'),receipt)
        while p.poll() is None:
            if (STUDY/'STOP').exists():
                parent=psutil.Process(p.pid)
                assert abs(parent.create_time()-receipt['process_created'])<1
                group=parent.children(recursive=True)+[parent]
                for q in group:
                    try:q.terminate()
                    except psutil.NoSuchProcess:pass
                psutil.wait_procs(group,timeout=5)
                raise RuntimeError('User STOP requested')
            time.sleep(2)
        assert p.returncode==0,f'{label} failed: see {RECOVERY/(label+".log")}'
    verify()
def run():
    with StageLock(STUDY,'workflow'):
        try:
            verify();entries=validate_entries();assert len(entries)==383
            base=ROOT/'datasets/cryptbb_v3';base.mkdir(exist_ok=True)
            charset=ROOT/'datasets/cryptbb/charset.txt'
            if not (base/'charset.txt').exists():shutil.copyfile(charset,base/'charset.txt')
            assert sha(base/'charset.txt')==sha(charset)
            for tag in 'AB':
                child('generate',tag);child('pack',tag)
            datasets={t:read(base/t/'data_receipt.json') for t in 'AB'}
            assert all(r['images']==r['unique_pixels']==100000 for r in datasets.values())
            write(STUDY/'completion.json',dict(completed_at=now(),datasets=datasets,training_started=False,training_authorized=True,reader_preflight_pending=True,admission_protocol_sha256=sha(STUDY/'accepted_recipe_protocol.json')))
            write(STUDY/'workflow_status.json',dict(stage='complete',state='datasets_ready_training_not_started',updated_at=now(),training_started=False))
        except Exception as exc:
            write(STUDY/'workflow_status.json',dict(state='stopped_by_user' if (STUDY/'STOP').exists() else 'failed',error=repr(exc),updated_at=now(),training_started=False));raise
if __name__=='__main__':run()
