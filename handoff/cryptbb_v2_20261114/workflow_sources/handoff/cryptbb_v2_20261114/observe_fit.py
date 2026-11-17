"""Observe an already-running full controller fit; never launches another fit."""
from pathlib import Path
import time,psutil
from capri_controller.decoder.common import read,write,sha,now,StageLock
ROOT=Path('E:/pranav/Thesis-Pranav-Final');S=ROOT/'handoff/cryptbb_v2_20261114';OUT=ROOT/'Capri/runs/cryptbb_controller_v2_free_full_20261114'
with StageLock(S,'fit_coordinator'):
    try:
        r=read(S/'fit_status.json');p=psutil.Process(r['pid']);assert abs(p.create_time()-r['process_created'])<1
        while p.is_running():time.sleep(10)
        status=read(OUT/'status.json');assert status['state']=='complete' and status['inputs']==list(range(100))
        entries=read(OUT/'entries.json');assert len(entries)==400 and len(read(OUT/'controller_history.json'))==100
        for r in read(S/'fit_sources.json'):assert sha(ROOT/r['path'])==r['sha256']
        write(S/'fit_completion.json',dict(completed_at=now(),entries=400,entries_sha256=sha(OUT/'entries.json'),output=str(OUT),unvalidated=sum(bool(e.get('unvalidated')) for e in entries),verified_final_status_schema=True))
        write(S/'fit_status.json',dict(state='complete',updated_at=now()))
    except Exception as exc:
        write(S/'fit_status.json',dict(state='failed',error=repr(exc),updated_at=now()));raise
