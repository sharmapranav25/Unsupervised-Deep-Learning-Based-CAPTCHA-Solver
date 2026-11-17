"""Load from the final workspace and match frozen predictions; no real labels."""
import copy
import gc
import hashlib
import inspect
import json
import os
import sys
import textwrap
import types
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
for k,sub in {'TEMP':'temporary','TMP':'temporary','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface','XDG_CACHE_HOME':'cache','MPLCONFIGDIR':'cache/matplotlib'}.items():
    p=ROOT/'runtime'/sub;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
sys.path[:0]=[str(ROOT/'model1/src/parseq_packages'),str(ROOT/'model1/src/OpenOCR'),str(ROOT/'Capri/src')]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    import lmdb
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    import openrec
    openocr_source=Path(inspect.getfile(build_model)).resolve()
    assert openocr_source.is_relative_to(ROOT)
    torch.set_num_threads(4)
    manifest=read(ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json')
    assert len(manifest)==900
    refs=read(ROOT/'Capri/configs/targets/darkbay_final.json')['fit']
    assert not {r['pixel_sha256'] for r in refs}&{r['pixel_sha256'] for r in manifest}
    for r in manifest:
        im=Image.open(ROOT/r['image_path']).convert('RGB')
        assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
    tr=T.Compose([T.Resize((32,128),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(.5,.5)])
    results=[]
    for tag,evaluation in [('A','evaluation_darkbay_cpu_requested_v2'),('B','evaluation_darkbay_cpu_epoch20_requested')]:
        data=ROOT/f'datasets/darkbay/{tag}/train_lmdb'
        env=lmdb.open(str(data),readonly=True,lock=False,readahead=False)
        with env.begin() as tx:
            n=int(tx.get(b'num-samples'));assert n==100000
            for index in (1,50000,100000):
                assert tx.get(f'image-{index:09d}'.encode()) and tx.get(f'label-{index:09d}'.encode())
        env.close()
        checkpoint=ROOT/f'model1/models/parseq/darkbay/{tag}/epoch_20.pth'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['epoch']==20
        cfg=copy.deepcopy(saved['config']);charset=str(ROOT/f'datasets/darkbay/{tag}/charset.txt')
        cfg['PostProcess']['character_dict_path']=charset
        cfg['Global']['character_dict_path']=charset
        decoder=build_post_process(cfg['PostProcess'],cfg['Global'])
        cfg['Architecture']['Decoder']['out_channels']=decoder.get_character_num()
        model=build_model(cfg['Architecture']);model.load_state_dict(saved['state_dict'],strict=True);model.cpu().eval();del saved
        original=model.decoder.forward_test.__func__
        source=textwrap.dedent(inspect.getsource(original));assert source.count('_device = memory.get_device()')==1
        namespace=dict(original.__globals__);exec(compile(source.replace('_device = memory.get_device()','_device = memory.device'),'<cpu-device-adapter>','exec'),namespace)
        model.decoder.forward_test=types.MethodType(namespace['forward_test'],model.decoder)
        rows=manifest[:16]
        x=torch.stack([tr(Image.open(ROOT/r['image_path']).convert('RGB')) for r in rows])
        with torch.inference_mode():decoded=decoder(model(x))
        frozen={r['opaque_id']:r['prediction'] for r in read(ROOT/f'model1/results/darkbay/{tag}/{evaluation}/predictions.json')}
        assert all(text==frozen[row['opaque_id']] for row,(text,_) in zip(rows,decoded))
        results.append(dict(dataset=tag,lmdb_images=n,checkpoint_epoch=20,frozen_predictions_matched=16))
        del model,decoder,x;gc.collect()
    from capri_controller.lightweight.model import LightweightCapri
    model=LightweightCapri();state=torch.load(ROOT/'Capri/models/capri/checkpoint.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['ema'],strict=True)
    result=dict(passed=True,verified_at=datetime.now(timezone.utc).isoformat(),device='cpu',reader_results=results,real_image_pixel_hashes_verified=900,controller_reference_overlap=0,real_scoring_labels_read=False,capri_ema_strict_load=True,openocr_loaded_from=str(openocr_source),all_project_inputs_from_new_workspace=True,external_python_environment=True,training_launched=False)
    (ROOT/'handoff/inference_smoke.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
