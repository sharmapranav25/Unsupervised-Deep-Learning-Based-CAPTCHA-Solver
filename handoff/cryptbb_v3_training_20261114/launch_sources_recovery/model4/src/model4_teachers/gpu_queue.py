"""Sequential GPU queue for the 10 teacher fine-tunes.

Waits until the CryptBB v3 data workflow is no longer running (complete / failed / stopped) and the GPU is
otherwise idle, then runs probe -> train for each model in a fresh process. A failed model is recorded and the
queue moves on. Create <RUN_DIR>/STOP to stop before the next model. Never overwrites finished models.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from .adapters import ALL
from .data import ROOT

RUN_DIR = ROOT / 'model4/runs/darkbay_AB_teachers_20261114'
FINAL_DIR = ROOT / 'Capri/fine-tuned-STRs'
V3_STATUS = ROOT / 'handoff/cryptbb_v3_20261114/workflow_status.json'
GPU_IDLE_MIB = 3000


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(p, x):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def gpu_used_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def v3_running():
    try:
        s = json.loads(V3_STATUS.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return False, None
    return s.get('state') == 'running', s


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    lock = RUN_DIR / 'queue.lock.json'
    if lock.exists():
        other = json.loads(lock.read_text(encoding='utf-8'))
        try:
            import psutil
            alive = psutil.pid_exists(other['pid']) and abs(psutil.Process(other['pid']).create_time() - other['create_time']) < 1
        except Exception:
            alive = True
        if alive:
            raise SystemExit(f'queue already running: {other}')
    import psutil
    me = psutil.Process()
    dump(lock, {'pid': me.pid, 'create_time': me.create_time(), 'started_at': now()})
    status = {'started_at': now(), 'models': {m: 'pending' for m in ALL}, 'state': 'waiting_for_gpu'}
    stat_p = RUN_DIR / 'queue_status.json'

    def save(**k):
        status.update(k, updated_at=now())
        dump(stat_p, status)

    save()
    while True:  # gate: v3 not running and GPU idle
        if (RUN_DIR / 'STOP').exists():
            save(state='stopped_by_marker_before_gate')
            lock.unlink(missing_ok=True)
            return
        running, s = v3_running()
        used = gpu_used_mib()
        if not running and used < GPU_IDLE_MIB:
            break
        save(state='waiting_for_gpu', v3=s, gpu_used_mib=used)
        time.sleep(60)
    save(state='running', gate_passed_at=now(), v3=v3_running()[1], gpu_used_mib=gpu_used_mib())

    for name in ALL:
        if (RUN_DIR / 'STOP').exists():
            save(state='stopped_by_marker')
            return
        if (FINAL_DIR / name / 'model.pt').exists():
            status['models'][name] = 'done (existing)'
            save()
            continue
        for mode in ('probe', 'train'):
            if mode == 'probe' and (RUN_DIR / name / 'probe.json').exists():
                continue
            status['models'][name] = f'{mode}_running'
            save()
            with open(RUN_DIR / name / f'{mode}.stdout.txt', 'a', encoding='utf-8') as f:
                rc = subprocess.run([sys.executable, '-B', '-u', '-m', 'model4_teachers.run', '--model', name,
                                     '--mode', mode], stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
            if rc != 0:
                status['models'][name] = f'{mode}_failed (rc={rc}); see {name}/{mode}.stdout.txt'
                save()
                break
        else:
            status['models'][name] = 'done'
            save()
    save(state='finished')
    lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
