"""Frozen identity and bounded operations for the fresh DarkBay v3 study."""
import contextlib,json,os,sys,time
from pathlib import Path
from .. import ab_curriculum as C
ROOT=C.ROOT
FAMILY='darkbay_v3_feature_20261115'
STUDY=ROOT/'handoff'/FAMILY
SOURCE=ROOT/'Capri/runs'/f'{FAMILY}_source'
FIT=ROOT/'Capri/runs'/f'{FAMILY}_fit'
DATA=ROOT/'datasets'/FAMILY
read,sha,now=C.read,C.sha,C.now
_write=C.write
def write(path,value):
    for i in range(50):
        try:return _write(path,value)
        except PermissionError:
            if i==49:raise
            time.sleep(.1)
def stop():
    if (STUDY/'STOP').exists():raise RuntimeError('User STOP; no automatic removal')
def environment():
    for k,v in dict(TEMP='temporary',TMP='temporary',TORCH_HOME='cache/torch',HF_HOME='cache/huggingface',XDG_CACHE_HOME='cache',MPLCONFIGDIR='cache/matplotlib').items():
        p=ROOT/'runtime'/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CAPRI_CONFIG=str(ROOT/'Capri/configs/local.json'),CAPRI_PROJECT=str(ROOT/'Capri/dependencies/sourcefree_synth'),CAPRI_RENDERER=str(ROOT/'Capri/renderer'))
    sys.path[:0]=[str(ROOT/'model1/src/parseq_packages'),str(ROOT/'model1/src/OpenOCR')]
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'Capri/src'),str(ROOT/'model1/src/parseq_packages'),str(ROOT/'model1/src/OpenOCR')])
def sources():
    base=ROOT/'Capri/src/capri_controller'
    generator=list(base.glob('*.py'))
    for folder in ('rendering','features','networks','lightweight','decoder'):
        generator.extend((base/folder).rglob('*.py'))
    return sorted(set(C.code_files()+generator+list(Path(__file__).parent.glob('*.py'))+[ROOT/'Capri/src/capri_controller/training/cryptbb_feature_v1.py',ROOT/'Capri/src/capri_controller/training/cryptbb_feature_v1_metadata.py',ROOT/'Capri/src/capri_controller/training/cryptbb_feature_v1_sampler.py',ROOT/'Capri/src/capri_controller/training/cryptbb_model1.py']))
def verify_sources():
    for r in read(STUDY/'source_manifest.json'):assert sha(ROOT/r['path'])==r['sha256'],r['path']
def rundir(model,tag):return ROOT/f'model{model}/runs'/FAMILY/tag
def stage(model,tag):return STUDY/f'model{model}'/tag
def config(tag,out):
    import yaml
    cfg=yaml.safe_load((ROOT/'model1/configs'/FAMILY/tag/'prepared.yml').read_text())
    cfg['Global'].update(output_dir=str(out),save_res_path=str(out/'unused.txt'))
    assert cfg['Global']['pretrained_model'] is None and cfg['Global']['checkpoints'] is None
    return cfg
def dataset(tag):
    import lmdb,numpy as np
    from ..ds2_data import ReaderDataset
    env=lmdb.open(str(DATA/tag/'train_lmdb'),readonly=True,lock=False,readahead=False)
    with env.begin() as tx:assert int(tx.get(b'num-samples'))==100000
    return ReaderDataset(env,np.arange(1,100001),DATA/'charset.txt')
@contextlib.contextmanager
def gpu_slot():
    import msvcrt,torch
    stop();path=STUDY/'gpu.lock'
    with path.open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        start=time.monotonic()
        while True:
            stop()
            try:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:
                if time.monotonic()-start>300:raise TimeoutError('GPU lock has not advanced in five minutes')
                time.sleep(.025)
        try:yield
        finally:
            torch.cuda.synchronize();torch.cuda.empty_cache();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    time.sleep(.005)
