"""Authorized v3 data-only pipeline. No reader imports, probes or training launch."""
import collections,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
from .decoder.common import read,write,sha,now,StageLock
from .cryptbb_v3_source import ROOT,STUDY,SOURCE,FIT,SPLIT

def verify():
    if (STUDY/'STOP').exists():raise RuntimeError('User STOP requested')
    for r in read(STUDY/'workflow_sources.json'):assert sha(ROOT/r['path'])==r['sha256'],r['path']

def child(module,args,stage):
    verify();cmd=[sys.executable,'-B','-u','-m',module]+args
    write(STUDY/'workflow_status.json',dict(stage=stage,state='running',updated_at=now(),training_started=False))
    with (STUDY/(stage+'.log')).open('a',encoding='utf-8') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        receipt=dict(stage=stage,pid=p.pid,process_created=psutil.Process(p.pid).create_time(),command=cmd,started_at=now())
        write(STUDY/'active_child.json',receipt);write(STUDY/(stage+'_launch.json'),receipt)
        while p.poll() is None:
            if (STUDY/'STOP').exists():
                q=psutil.Process(p.pid);tree=q.children(recursive=True)+[q]
                for x in tree:
                    try:x.terminate()
                    except psutil.NoSuchProcess:pass
                psutil.wait_procs(tree,timeout=5)
                raise RuntimeError('User STOP requested; active child tree terminated')
            time.sleep(2)
        assert p.returncode==0,f'{stage} failed ({p.returncode}); inspect its log'
    verify()

def run():
    with StageLock(STUDY,'workflow'):
        try:
            verify()
            child('capri_controller.cryptbb_v3_source',['prepare'],'prepare_reference_recipes')
            source_files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in sorted(SOURCE.glob('*')) if p.is_file()]
            write(STUDY/'prepared_source_hashes.json',source_files)
            child('capri_controller.cryptbb_v3_source',['verify'],'verify_fast_features')
            child('capri_controller.cryptbb_v3_source',['fit'],'controller_fit')
            for r in source_files:assert sha(ROOT/r['path'])==r['sha256']
            state=read(FIT/'status.json');entries=read(FIT/'entries.json');history=read(FIT/'controller_history.json')
            assert state['state']=='complete' and state['inputs']==list(range(100)) and len(history)==100 and len(entries)==400
            assert collections.Counter(e['controller'] for e in entries)==dict.fromkeys(['custom','dino','mixed','synthtiger'],100)
            write(STUDY/'fit_completion.json',dict(completed_at=now(),entries=400,entries_sha256=sha(FIT/'entries.json'),source_hashes=source_files,unvalidated=[e['entry_id'] for e in entries if e.get('unvalidated')]))
            from .cryptbb_v3_corpus import validate_entries
            validate_entries() # Invalid entries stop export and are reported by ID.
            base=ROOT/'datasets/cryptbb_v3';base.mkdir(exist_ok=True)
            assert not (base/'charset.txt').exists();shutil.copyfile(ROOT/'datasets/cryptbb/charset.txt',base/'charset.txt')
            for tag in 'AB':
                child('capri_controller.cryptbb_v3_corpus',['generate','--dataset',tag,'--workers','12'],'generate_'+tag)
                child('capri_controller.cryptbb_v3_corpus',['pack','--dataset',tag],'pack_'+tag)
            done={t:read(base/t/'data_receipt.json') for t in 'AB'}
            assert all(r['images']==100000 and r['unique_pixels']==100000 for r in done.values())
            write(STUDY/'completion.json',dict(completed_at=now(),datasets=done,training_started=False,training_authorized=False,reader_preflight_pending=True))
            write(STUDY/'workflow_status.json',dict(stage='complete',state='datasets_ready_training_not_started',updated_at=now(),training_started=False))
        except Exception as exc:
            write(STUDY/'workflow_status.json',dict(state='stopped_by_user' if (STUDY/'STOP').exists() else 'failed',error=repr(exc),updated_at=now(),training_started=False));raise
if __name__=='__main__':run()
