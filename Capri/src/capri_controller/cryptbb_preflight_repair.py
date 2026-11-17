"""Finish existing CryptBB preparation after ratio-loader preflight failure.

Preserves frozen generator sources, corpora and original configs/failed logs.
No train(), backward(), optimizer.step(), or real-label reads.
"""
import copy, gc, hashlib, io, json
from pathlib import Path
from .decoder.common import read, write, sha, now, StageLock
from .cryptbb_preparation import ROOT, OUT, verify_dependencies, stop

def write_once(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Existing repair artifact differs: '+str(path))
    else:
        write(path, value)

def configs():
    import yaml
    verify_dependencies()
    old = read(OUT/'datasets_ready.json')
    assert old['total_unique_pixels']==200000 and old['cross_dataset_pixel_overlap']==0
    rows=[]; pixel_sets=[]
    for row in old['configs']:
        tag=row['dataset']; pool=ROOT/f'datasets/cryptbb/{tag}'
        receipt=read(pool/'data_receipt.json')
        assert receipt['images']==receipt['unique_pixels']==100000
        assert receipt['all_images_decoded'] and receipt['all_original_reference_and_historical_eval_pixel_overlap']==0
        for path,key in [(pool/'train_lmdb/data.mdb','lmdb_sha256'),(pool/'sample_provenance.jsonl','provenance_sha256'),(ROOT/'datasets/cryptbb/charset.txt','charset_sha256')]:
            assert sha(path)==receipt[key],str(path)
        pixels={json.loads(line)['pixel_sha256'] for line in (pool/'sample_provenance.jsonl').read_text().splitlines()}
        assert len(pixels)==100000;pixel_sets.append(pixels)
        original=ROOT/row['path'];assert sha(original)==row['sha256']
        cfg=yaml.safe_load(original.read_text());fixed=copy.deepcopy(cfg)
        assert cfg['Train']['loader']['max_ratio']==4 and 'min_ratio' not in cfg['Train']['loader']
        fixed['Train']['loader']['min_ratio']=4
        check=copy.deepcopy(fixed);del check['Train']['loader']['min_ratio'];assert check==cfg
        dest=original.with_name('prepared_fixed32x128.yml');text=yaml.safe_dump(fixed,sort_keys=False)
        if dest.exists():assert dest.read_text()==text
        else:dest.write_text(text,encoding='utf8')
        rows.append(dict(dataset=tag,path=str(dest.relative_to(ROOT)),sha256=sha(dest),source_config=str(original.relative_to(ROOT)),source_sha256=sha(original),preflight=f'model1/runs/cryptbb/{tag}/preflight_fixed32x128/receipt.json'))
    assert not pixel_sets[0]&pixel_sets[1]
    ready=dict(version=2,images_each=100000,total_unique_pixels=200000,cross_dataset_pixel_overlap=0,configs=rows,training_started=False,change='Only add loader min_ratio=4 alongside max_ratio=4 to guarantee original intended32x128 input. Pools unchanged.',supersedes='datasets_ready.json for config paths; original retained')
    write_once(OUT/'datasets_ready_v2.json',ready)
    return ready

def preflight(row):
    import numpy as np, torch
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms import functional as F
    from tools.engine.config import Config
    from tools.engine.trainer import Trainer
    tag=row['dataset'];dest=ROOT/row['preflight'];config=ROOT/row['path']
    assert sha(config)==row['sha256']
    if dest.exists():
        saved=read(dest);assert saved['passed'] and saved['config_sha256']==row['sha256'];return
    cfg=Config(str(config));cfg.cfg['Global']['output_dir']=str(dest.parent)
    assert cfg.cfg['Eval'] is None and cfg.cfg['Global']['pretrained_model'] is None and cfg.cfg['Global']['checkpoints'] is None
    torch.set_num_threads(8);trainer=Trainer(cfg,mode='train',task='rec')
    assert trainer.valid_dataloader is None
    ds=trainer.train_dataloader.dataset
    assert len(ds)==100000 and len(trainer.train_dataloader)==782
    assert np.all(ds.wh_ratio==4) and ds.padding is False
    assert len(trainer.optimizer.state)==0
    trainer.model.eval()
    def state_hash():
        h=hashlib.sha256()
        for name,value in trainer.model.state_dict().items():
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=state_hash()
    batch=next(iter(trainer.train_dataloader));x=batch[0].to(trainer.device)
    assert list(x.shape)==[128,3,32,128],list(x.shape)
    assert torch.isfinite(x).all()
    # Verify actual loader values equal the curriculum's intended fixed transform.
    for index in range(8):
        lmdb_index,file_index=map(int,ds.data_idx_order_list[index])
        txn=ds.lmdb_sets[lmdb_index]['txn'];data=txn.get(f'image-{file_index:09d}'.encode())
        im=Image.open(io.BytesIO(data)).convert('RGB')
        expected=F.normalize(F.to_tensor(F.resize(im,[32,128],interpolation=T.InterpolationMode.BICUBIC)),[.5],[.5])
        actual=ds[[128,32,index,4]][0]
        assert torch.equal(actual,expected),'Preprocessing parity failed'
    with torch.inference_mode():y=trainer.model(x)
    assert torch.isfinite(y).all() and before==state_hash()
    assert len(trainer.optimizer.state)==0
    write(dest,dict(passed=True,completed_at=now(),dataset_images=100000,batches_per_epoch=782,batch_shape=list(x.shape),parameters=sum(p.numel() for p in trainer.model.parameters()),optimizer_steps=0,backward_calls=0,model_parameters_and_buffers_unchanged=True,initialization='from scratch',config_sha256=row['sha256'],fixed_transform_parity_samples=8,all_dataset_ratios=4,output_shape=list(y.shape)))
    del trainer,ds,x,y,batch;gc.collect();torch.cuda.empty_cache()

def selection_protocol():
    # Pixel manifest only. This function never opens real scoring labels.
    from PIL import Image
    import numpy as np
    manifest=ROOT/'evaluation_data/cryptbb/image_manifest.json';rows=read(manifest)
    assert len(rows)==551 and len({r['opaque_id'] for r in rows})==551
    for row in rows:
        pixels=np.asarray(Image.open(row['image_path']).convert('RGB'))
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['pixel_sha256']
    protocol=dict(version=1,family='CryptBB',selection='max_same_epoch_ab_exact_agreement_earliest_tie',manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest),images=551,candidate_epochs=list(range(2,21)),selection_unit='Shared epoch per Model1/2/3 A/B pair',normalization='Exact case-sensitive decoded strings; no correction or length filter; empty strings remain in raw agreement',decoder='Existing greedy PARSeq free AR plus1 refinement; no forced target length',real_labels_used=False,real_images_used_for_training=False,pool_usage='Same551 pixels used unlabeled for selection and then real scoring: transductive, not independent untouched test',supersedes='Historical synthetic-validation requirement in preparation protocol and launcher',source_policy='handoff/darkbay_curriculum_20261113/agreement_selection_policy.json',teacher_policy='Opposite Model1 epoch20 for initial rankings, separate from reporting checkpoint selection',training_authorized=False)
    write_once(OUT/'agreement_protocol.json',protocol)
    return protocol

def main():
    assert ROOT.drive.upper()=='E:'
    import os
    for key in ['TEMP','TMP','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        assert Path(os.environ[key]).drive.upper()=='E:',key
    with StageLock(OUT,'preparation'):
        stop();verify_dependencies()
        if (OUT/'completion.json').exists():
            assert read(OUT/'completion.json')['repair_source_sha256']==sha(Path(__file__));print('Already complete');return
        write(OUT/'status.json',dict(updated_at=now(),stage='preflight_repair',state='running',pid=os.getpid(),reader_training_started=False))
        try:
            ready=configs()
            for row in ready['configs']:
                stop();preflight(row)
            policy=selection_protocol()
            write(OUT/'completion.json',dict(completed_at=now(),datasets=ready,reader_training_started=False,reader_training_authorized=False,repair_source_sha256=sha(Path(__file__)),original_failure='preflight_A expected32x128 but aspect ratio3 selected40x112; original failure/configs retained',selection_protocol_sha256=sha(OUT/'agreement_protocol.json'),limitations=['Historical scoring labels follow documented filenames, not independent annotation','Agreement selection is transductive and does not guarantee best real checkpoint','Model1 resume fidelity and CryptBB Model2/3 adapters remain future work; no training launched'],all_preflights_passed=True))
            write(OUT/'status.json',dict(updated_at=now(),stage='ready',state='complete',reader_training_started=False,reader_training_authorized=False))
            print('CryptBB dataset/setup preparation complete; no training started',flush=True)
        except Exception as exc:
            write(OUT/'status.json',dict(updated_at=now(),stage='preflight_repair_failed',state='failed',error=repr(exc),reader_training_started=False));raise

if __name__=='__main__':main()
