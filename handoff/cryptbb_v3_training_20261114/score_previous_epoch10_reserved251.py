from pathlib import Path
import hashlib,shutil
import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein
from capri_controller.decoder.common import read,write,sha,now
R=Path.cwd();old=R/'model1/results/cryptbb/A/requested_cpu_epoch10';out=R/'model1/results/cryptbb/A/reserved251_cpu_epoch10';new=R/'model1/results/cryptbb_v3/B/requested_cpu_epoch02'
manifest=R/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json';old_manifest=R/'evaluation_data/cryptbb/image_manifest.json'
p=read(old/'protocol.json');freeze=read(old/'prediction_freeze.json');cp=R/'model1/runs/cryptbb/A/epoch_10.pth';receipt=read(cp.with_name('epoch_10_receipt.json'))
assert sha(cp)==receipt['sha256']==p['checkpoint_sha256']==freeze['checkpoint_sha256']
assert sha(old_manifest)==p['manifest_sha256'];assert sha(old/'predictions.json')==freeze['sha256'];assert read(old/'verification.json')['passed']
preds={r['opaque_id']:r['prediction'] for r in read(old/'predictions.json')};old_images=read(old_manifest);assert len(preds)==551 and set(preds)=={r['opaque_id'] for r in old_images}
by_hash={r['pixel_sha256']:preds[r['opaque_id']] for r in old_images};rows=read(manifest);assert len(rows)==251
np_=read(new/'protocol.json');assert sha(manifest)==np_['manifest_sha256'];assert p['preprocessing']==np_['preprocessing'] and p['decoding']==np_['decoding']
values=[]
for r in rows:
 im=Image.open(R/r['image_path']).convert('RGB');assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
 values.append(dict(opaque_id=r['opaque_id'],prediction=by_hash[r['pixel_sha256']]))
assert not out.exists();out.mkdir(parents=True);shutil.copyfile(__file__,out/'scorer_source.py')
write(out/'protocol.json',dict(**{**p,'images':251,'manifest_sha256':sha(manifest),'created_at':now()},source_predictions_sha256=freeze['sha256'],method='Subset frozen551 predictions by exact RGB hash; no new inference'))
write(out/'predictions.json',values);write(out/'prediction_freeze.json',dict(at=now(),sha256=sha(out/'predictions.json'),images=251,checkpoint_sha256=sha(cp)))
labels=read(R/'evaluation_data/model4_reservation_20261114/scoring_only/cryptbb/evaluation_pool_labels.json');assert set(labels)=={r['opaque_id'] for r in values}
correct=sum(r['prediction']==labels[r['opaque_id']] for r in values);edits=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in values);chars=sum(map(len,labels.values()))
s=dict(model=1,dataset='A',epoch=10,correct=correct,total=251,accuracy_percent=100*correct/251,cer_percent=100*edits/chars,edit_based_character_accuracy_percent=100*(1-edits/chars),character_errors=edits,reference_characters=chars,checkpoint_sha256=sha(cp),device='cpu',no_new_inference=True,diagnostic_only=True,training_changed=False)
write(out/'summary.json',s);write(out/'verification.json',dict(passed=True,source_prediction_hash_verified=True,pixel_identities_verified=251,predictions_frozen_before_labels=True,same_preprocessing_and_decoding_as_current_B=True))
print(s)
