from pathlib import Path
import argparse,json,subprocess,sys,os
ROOT=Path('E:/pranav/Thesis-Pranav-Final');OUT=ROOT/'handoff/cryptbb_v2_20261114/existing_data_parallel_smoke'
from capri_controller.decoder.common import read,write,sha,now
ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['A','B']);a=ap.parse_args()
if a.dataset:
    from capri_controller.training import cryptbb_v2_model1 as M
    original_ready=M.BASE.readiness
    M.STUDY=OUT
    M.row_for=original_ready
    M.check_data=lambda tag:original_ready(tag)
    M.probe(a.dataset)
else:
    OUT.mkdir(exist_ok=False)
    protocol=dict(scope='Disposable GPU/recovery compatibility probe on existing A/B corpora; not production readiness or training',created_at=now(),new_module_sha256=sha(ROOT/'Capri/src/capri_controller/training/cryptbb_v2_model1.py'))
    write(OUT/'training_protocol.json',protocol);write(OUT/'protocol.json',protocol)
    children=[]
    for tag in 'AB':
        log=(OUT/f'{tag}.log').open('w')
        p=subprocess.Popen([sys.executable,'-B','-u',__file__,'--dataset',tag],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);children.append((tag,p,log))
    for tag,p,log in children:
        code=p.wait();log.close();assert code==0,f'{tag} probe failed: {code}; inspect {OUT}'
    results=[read(OUT/'readiness'/t/'receipt.json') for t in 'AB']
    assert all(r['passed'] for r in results)
    write(OUT/'summary.json',dict(passed=True,scope=protocol['scope'],results=results));print(json.dumps(results))
