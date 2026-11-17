"""Adapt the established diagnostic without modifying frozen training sources."""
import ast
from pathlib import Path
ROOT=Path('E:/pranav/Thesis-Pranav-Final');OUT=Path(__file__).parent
s=(ROOT/'Capri/src/capri_controller/training/cryptbb_v3_checkpoint_evaluation.py').read_text()
def replace(a,b):
    global s
    assert a in s,a
    s=s.replace(a,b)
replace('CryptBB v3 Model1 diagnostic','CryptBB v3 selected-teacher Model2 diagnostic')
replace('import argparse, copy, gc, hashlib, inspect, os, shutil, textwrap, time, types','import argparse, copy, gc, hashlib, inspect, os, shutil, sys, textwrap, time, types')
replace("os.environ['CUDA_VISIBLE_DEVICES']=''", "os.environ['CUDA_VISIBLE_DEVICES']=''\n    sys.path[:0]=[str(ROOT/'model1/src/parseq_packages'),str(ROOT/'model1/src/OpenOCR')]\n    from . import cryptbb_v3_model2_selected_v1 as M\n    training_digest=M.check_sources()")
replace("run=ROOT/f'model1/runs/cryptbb_v3/{a.dataset}';checkpoint=run/f'epoch_{a.epoch}.pth'\n    receipt=read(run/f'epoch_{a.epoch}_receipt.json');assert receipt['epoch']==a.epoch and receipt['global_step']==782*a.epoch\n    digest=sha(checkpoint);assert digest==receipt['sha256']", "run=M.run_dir(2,a.dataset);checkpoint=run/f'checkpoints/epoch_{a.epoch:02d}.pth'\n    history=read(run/'history.json');assert a.epoch in [r['epoch'] for r in history]\n    request=read(ROOT/'handoff/cryptbb_v3_model2_measurements_20261115/requested_checkpoints.json')\n    receipt=next(r for r in request['checkpoints'] if r['dataset']==a.dataset and r['epoch']==a.epoch)\n    digest=sha(checkpoint);assert digest==receipt['sha256']")
replace("f'model1/results/cryptbb_v3/{a.dataset}/requested_cpu_epoch", "f'model2/results/cryptbb_v3_rank1_v1/{a.dataset}/requested_cpu_epoch")
replace("assert ck['epoch']==a.epoch", "assert ck['epoch']==a.epoch and ck['global_step']==782*a.epoch and ck['protocol_sha256']==training_digest and ck['config']['Global']['seed']==48")
replace('model=1,dataset=a.dataset', 'model=2,dataset=a.dataset')
replace("protocol=dict(created_at=now(),model=2", "protocol=dict(created_at=now(),training_protocol_sha256=training_digest,checkpoint=str(checkpoint.relative_to(ROOT)),seed=48,model=2")
replace("training_protocol=read(ROOT/'handoff/cryptbb_v3_training_20261114/training_protocol.json')\n        for r in training_protocol['sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']", "assert M.check_sources()==training_digest")
ast.parse(s)
(OUT/'cryptbb_v3_model2_checkpoint_evaluation.py').write_text(s,encoding='utf8')
print('Staged isolated CPU evaluator')
