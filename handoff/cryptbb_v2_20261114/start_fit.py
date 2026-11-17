from pathlib import Path
import sys,os,json,time,subprocess,psutil,hashlib
ROOT=Path('E:/pranav/Thesis-Pranav-Final');STUDY=ROOT/'handoff/cryptbb_v2_20261114'
sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.decoder.common import write,now,StageLock,sha
OUT=ROOT/'Capri/runs/cryptbb_controller_v2_free_full_20261114'
with StageLock(STUDY,'fit_coordinator'):
    try:
        pending=[]
        for p in psutil.process_iter(['cmdline']):
            c=p.info['cmdline'] or []
            if 'capri_controller.controller_v2' in c and '--inputs' in c and c[c.index('--inputs')+1]=='4':pending.append(p)
        write(STUDY/'fit_status.json',dict(state='waiting_for_existing_previews',pids=[p.pid for p in pending],updated_at=now()))
        for p in pending:
            while p.is_running():
                if (STUDY/'STOP').exists():raise RuntimeError('User STOP before full fit')
                time.sleep(5)
        expected=json.loads((STUDY/'fit_sources.json').read_text())
        for r in expected:assert sha(ROOT/r['path'])==r['sha256'],r['path']
        cmd=[sys.executable,'-B','-u','-m','capri_controller.controller_v2','--dataset','cryptbb','--inputs','100','--workers','14','--mode','free','--out',str(OUT)]
        with (STUDY/'controller_fit.log').open('a',encoding='utf8') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            write(STUDY/'fit_status.json',dict(state='fitting',pid=child.pid,process_created=psutil.Process(child.pid).create_time(),command=cmd,updated_at=now()))
            code=child.wait()
        assert code==0,f'Controller fit failed: {code}'
        for r in expected:assert sha(ROOT/r['path'])==r['sha256']
        entries=json.loads((OUT/'entries.json').read_text());assert len(entries)==400
        assert json.loads((OUT/'status.json').read_text())['processed']==100
        write(STUDY/'fit_completion.json',dict(completed_at=now(),entries=400,entries_sha256=sha(OUT/'entries.json'),output=str(OUT),unvalidated=sum(bool(e.get('unvalidated')) for e in entries)))
        write(STUDY/'fit_status.json',dict(state='complete',updated_at=now()))
    except Exception as exc:
        write(STUDY/'fit_status.json',dict(state='failed',error=repr(exc),updated_at=now()));raise
