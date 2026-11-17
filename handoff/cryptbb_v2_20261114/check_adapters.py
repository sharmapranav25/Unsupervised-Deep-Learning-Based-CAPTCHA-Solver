from pathlib import Path
import ast,copy,json,sys,contextlib
ROOT=Path('E:/pranav/Thesis-Pranav-Final');OUT=ROOT/'handoff/cryptbb_v2_20261114/adapter_checks';OUT.mkdir(exist_ok=True)
from capri_controller.decoder.common import write,sha
for p in [ROOT/'Capri/src/capri_controller/cryptbb_v2_corpus.py',ROOT/'Capri/src/capri_controller/cryptbb_v2_workflow.py',ROOT/'Capri/src/capri_controller/training/cryptbb_v2_model1.py']:ast.parse(p.read_text(encoding='utf-8-sig'))
from capri_controller import cryptbb_v2_corpus as C
from capri_controller.controller_v2 import SOURCES
from capri_controller.decoder.common import read
from capri_controller.training import cryptbb_v2_model1 as M
# Loader wrapper must preserve CPU loading outside lock, full consumer step inside,
# and release on exceptions/close. No CUDA or production training needed.
events=[]
@contextlib.contextmanager
def fake_slot():
    events.append('acquire')
    try:yield
    finally:events.append('release')
M.gpu_slot=fake_slot
class Fake:
    dataset='dataset';batch_sampler='sampler'
    def __len__(self):return 2
    def __iter__(self):
        for i in range(2):events.append('load');yield i
loader=M.LockedLoader(Fake());assert len(loader)==2 and loader.dataset=='dataset' and loader.batch_sampler=='sampler'
g=iter(loader);assert next(g)==0;events.append('step');assert next(g)==1;events.append('step');g.close()
assert events==['load','acquire','step','release','load','acquire','step','release'],events
# Corpus worker test covers each engine on existing validated four-input previews.
preview=ROOT/'Capri/runs/controller_v2c_Afree_diverse4_cryptbb_20261114'
entries=read(preview/'entries.json');recipes=read(SOURCES['cryptbb']/'refined_recipes.json');C.init('A');dest=OUT/'shards';dest.mkdir(exist_ok=True)
result=[]
for arm in C.ARMS:
    e=next(e for e in entries if e['controller']==arm and not e.get('unvalidated'))
    result.append(C.generate_shard(('A',e,recipes[e['input_index']],1000+C.ARMS.index(arm),3,str(dest),sha(preview/'entries.json'))))
import zipfile,io,hashlib
from PIL import Image
import numpy as np
for r in result:
    with zipfile.ZipFile(dest/(r['entry_id']+'.zip')) as z:
        for row in [json.loads(x) for x in z.read('labels.jsonl').decode().splitlines()]:
            a=np.array(Image.open(io.BytesIO(z.read(row['image']))).convert('RGB'));assert hashlib.sha256(a.tobytes()).hexdigest()==row['pixel_sha256'];assert row['text_contrast']>=40
write(OUT/'receipt.json',dict(passed=True,syntax_modules=3,loader_lock_events=events,all_four_engines_exported=True,decoded_samples=12,shards=result,scope='Adapter checks using preview recipes; does not replace actual full-corpus or paired GPU preflight'))
print('Adapter checks passed: 12 decoded samples across four arms; full-step loader-lock lifecycle verified')
