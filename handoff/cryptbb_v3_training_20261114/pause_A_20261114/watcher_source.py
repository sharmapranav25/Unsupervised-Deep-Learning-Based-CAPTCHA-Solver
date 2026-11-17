"""Observe the existing pair; release A after its verified epoch1 save, keep B running."""
import time
from pathlib import Path
import psutil
from ..decoder.common import read,write,sha,now,StageLock
ROOT=Path(__file__).resolve().parents[4];STUDY=ROOT/'handoff/cryptbb_v3_training_20261114';REC=STUDY/'pause_A_20261114'
def exact(row):
    p=psutil.Process(row['pid']);assert abs(p.create_time()-row['process_created'])<1
    return p

def state(value,**kw):write(STUDY/'status.json',dict(stage='model1_B_priority',state=value,updated_at=now(),**kw))
def run():
    control=read(STUDY/'pause_A_after_epoch1.json')
    with StageLock(STUDY,'coordinator'):
        try:
            state('waiting_for_A_epoch1_then_B_only',A_pause_requested=True,B_training=True)
            a_dir=ROOT/'model1/runs/cryptbb_v3/A';b_dir=ROOT/'model1/runs/cryptbb_v3/B'
            while not (a_dir/'epoch_1_receipt.json').exists():
                assert not (STUDY/'STOP').exists(),'Stop marker detected; preserve user intent'
                exact(control['A']);exact(control['B']);time.sleep(.1)
            a=exact(control['A']);a.suspend()
            try:
                receipt=read(a_dir/'epoch_1_receipt.json')
                assert receipt['epoch']==1 and receipt['global_step']==782
                assert receipt['protocol_sha256']==sha(STUDY/'training_protocol.json')
                assert sha(a_dir/receipt['checkpoint'])==receipt['sha256']
                assert sha(a_dir/'latest.pth')==receipt['sha256']
                assert read(a_dir/'latest_receipt.json')['epoch']==1
            except Exception:
                a.resume();raise
            progress=read(a_dir/'progress.json');a.terminate();a.wait(timeout=15)
            write(REC/'A_paused_receipt.json',dict(paused_at=now(),process=control['A'],verified_checkpoint=receipt,last_progress=progress,checkpoint_preserved=True,pause_is_user_requested=True,B_interrupted=False,method='Suspend on committed epoch1 receipt, verify archive/latest hashes, terminate only exact A PID to release GPU memory'))
            write(a_dir/'pause.json',dict(paused_at=now(),epoch=1,checkpoint=receipt['checkpoint'],sha256=receipt['sha256'],resume_requires_user_instruction=True))
            state('A_paused_epoch1_B_running',A_saved_epoch=1,B_training=True,Model2_waiting_for_A=True)
            while not (b_dir/'training_completion.json').exists():
                assert not (STUDY/'STOP').exists(),'Stop marker detected; preserve user intent'
                exact(control['B']);time.sleep(10)
            done=read(b_dir/'training_completion.json');assert done['epoch']==20
            assert sha(b_dir/done['checkpoint'])==done['sha256']
            write(REC/'B_completion_verified.json',dict(verified_at=now(),completion=done,A_remains_paused=True,Model2_started=False))
            state('B_complete_A_paused_awaiting_user',A_saved_epoch=1,B_saved_epoch=20,Model2_started=False)
        except Exception as exc:
            state('attention_required',error=repr(exc),A_pause_requested=True);raise
if __name__=='__main__':run()
