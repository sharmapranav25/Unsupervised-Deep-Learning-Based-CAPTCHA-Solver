"""Frozen official GLM-OCR baseline on the existing 551-image CryptBB historical test split."""
import hashlib
import json
import os
from pathlib import Path
import time

WORK=Path(__file__).resolve().parents[3]
OUT=WORK/'runs/glm_ocr_cryptbb_20261113'
MODEL=Path('E:/pranav/CAPRI/models/external/GLM-OCR')
SOURCE=WORK.parent/'evaluation_data/cryptbb'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    import transformers
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration
    torch.set_num_threads(4)
    torch.manual_seed(48)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.30)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise FileExistsError('Completed evaluation exists')
    records=read(SOURCE/'image_manifest.json');assert len(records)==551
    receipt=read(MODEL/'download_receipt.json')
    for row in receipt['files']:assert sha(MODEL/row['file'])==row['sha256']
    protocol=dict(model='zai-org/GLM-OCR',revision=receipt['revision'],download_receipt_sha256=sha(MODEL/'download_receipt.json'),
        images=551,image_manifest_sha256=sha(SOURCE/'image_manifest.json'),prompt='Text Recognition:',
        preprocessing='Official AutoProcessor, original RGB pixels, no custom crop or enlargement',
        generation=dict(do_sample=False,max_new_tokens=64,num_beams=1),dtype='bfloat16',attention='sdpa',batch_size=1,
        scoring='Case-sensitive whole-string match after trimming only leading/trailing whitespace; raw text retained',
        label_access='Predictions frozen before labels loaded for scoring',
        packages=dict(torch=torch.__version__,transformers=transformers.__version__),
        limitation='Pretrained external multimodal baseline explicitly requested by user; not part of the no-VLM CAPRI controller or DS2 training. Historical real evaluation set; pretrained corpus overlap unknown.',
        code_sha256=sha(Path(__file__)),adapted_from_sha256=sha(Path(__file__).with_name('glm_ocr.py')),reader_training=False)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:dump(OUT/'protocol.json',protocol)
    processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True)
    model=GlmOcrForConditionalGeneration.from_pretrained(str(MODEL),torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    rows=[];journal=OUT/'predictions.jsonl'
    if journal.exists():
        rows=[json.loads(line) for line in journal.read_text().splitlines()]
        assert [r['opaque_id'] for r in rows]==[r['opaque_id'] for r in records[:len(rows)]]
    started=time.time()
    with journal.open('a',encoding='utf8',buffering=1) as f,torch.inference_mode():
        for index in range(len(rows),len(records)):
            r=records[index];im=Image.open(r['image_path']).convert('RGB')
            assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
            messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':'Text Recognition:'}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
            tokens=model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=64)
            generated=tokens[0,inputs['input_ids'].shape[-1]:]
            raw=processor.decode(generated,skip_special_tokens=True)
            row=dict(opaque_id=r['opaque_id'],raw_prediction=raw,prediction=raw.strip(),generated_tokens=len(generated),hit_token_limit=len(generated)>=64)
            f.write(json.dumps(row)+'\n');rows.append(row)
            if (index+1)%25==0 or index==0:
                state=dict(pid=os.getpid(),stage='inference',completed=index+1,total=551,elapsed_seconds=time.time()-started,peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
                dump(OUT/'status.json',state);print(state,flush=True)
    dump(OUT/'predictions.json',rows)
    digest=sha(OUT/'predictions.json');dump(OUT/'prediction_freeze.json',dict(sha256=digest,count=len(rows)))
    # Only after all outputs are frozen do we load the target answers.
    from rapidfuzz.distance import Levenshtein
    labels=read(SOURCE/'scoring_labels/scoring_labels.json')
    assert set(labels)=={r['opaque_id'] for r in rows}
    correct=sum(r['prediction']==labels[r['opaque_id']] for r in rows)
    edits=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in rows)
    summary=dict(model='GLM-OCR',revision=receipt['revision'],correct=correct,total=551,accuracy=100*correct/551,
        CER=100*edits/sum(map(len,labels.values())),raw_exact_correct=sum(r['raw_prediction']==labels[r['opaque_id']] for r in rows),
        hit_token_limit=sum(r['hit_token_limit'] for r in rows),evaluation_family='CryptBB',historical_selftrain_images_excluded=100,
        preprocessing='Official processor; no target-specific tuning',prediction_sha256=digest)
    assert sha(OUT/'predictions.json')==digest
    dump(OUT/'summary.json',summary);dump(OUT/'status.json',dict(stage='complete',**summary));print(summary,flush=True)


if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'evaluation.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        except Exception as exc:
            dump(OUT/'failure.json',dict(pid=os.getpid(),error=repr(exc)));raise
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
