from pathlib import Path
import hashlib,shutil
import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein
from capri_controller.decoder.common import read,write,sha,now
R=Path.cwd();old=R/'Capri/runs/glm_ocr_cryptbb_20261113';out=R/'Capri/runs/glm_ocr_cryptbb_reserved251_20261114'
original_manifest=R/'evaluation_data/cryptbb/image_manifest.json';manifest=R/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json'
protocol=read(old/'protocol.json');assert sha(original_manifest)==protocol['image_manifest_sha256']
predictions=read(old/'predictions.json');freeze=read(old/'prediction_freeze.json');assert sha(old/'predictions.json')==freeze['sha256'] and freeze['count']==551
old_images=read(original_manifest);assert len(old_images)==551
prediction_by_id={r['opaque_id']:r for r in predictions};assert set(prediction_by_id)=={r['opaque_id'] for r in old_images}
by_hash={r['pixel_sha256']:prediction_by_id[r['opaque_id']] for r in old_images};assert len(by_hash)==551
rows=read(manifest);assert len(rows)==251
assert sha(manifest)==read(R/'handoff/cryptbb_v3_training_20261114/agreement_protocol.json')['manifest_sha256']
values=[]
for r in rows:
 im=Image.open(R/r['image_path']).convert('RGB');assert hashlib.sha256(np.asarray(im).tobytes()).hexdigest()==r['pixel_sha256']
 p=by_hash[r['pixel_sha256']]
 values.append(dict(opaque_id=r['opaque_id'],pixel_sha256=r['pixel_sha256'],source_opaque_id=p['opaque_id'],prediction=p['prediction'],raw_prediction=p['raw_prediction']))
assert len({r['opaque_id'] for r in values})==251
assert not out.exists(),'Preserve earlier result';out.mkdir(parents=True)
shutil.copyfile(__file__,out/'scorer_source.py')
write(out/'protocol.json',dict(at=now(),model='GLM-OCR',revision=protocol['revision'],images=251,manifest_sha256=sha(manifest),source_manifest_sha256=sha(original_manifest),source_prediction_sha256=freeze['sha256'],method='Subset previously frozen551 predictions by exact RGB pixel hash; no new inference',scoring='Case-sensitive exact match after original whitespace trim; also report untrimmed raw exact',training_changed=False,preprocessing=protocol['preprocessing']))
write(out/'predictions.json',values);write(out/'prediction_freeze.json',dict(at=now(),count=251,sha256=sha(out/'predictions.json'),source_sha256=freeze['sha256']))
# Open reserved evaluation labels only after subset predictions are persisted and hash-frozen.
labels=read(R/'evaluation_data/model4_reservation_20261114/scoring_only/cryptbb/evaluation_pool_labels.json');assert set(labels)=={r['opaque_id'] for r in values}
correct=sum(r['prediction']==labels[r['opaque_id']] for r in values);raw_correct=sum(r['raw_prediction']==labels[r['opaque_id']] for r in values)
edits=sum(Levenshtein.distance(labels[r['opaque_id']],r['prediction']) for r in values);characters=sum(len(t) for t in labels.values())
summary=dict(model='GLM-OCR',correct=correct,total=251,accuracy_percent=100*correct/251,raw_exact_correct=raw_correct,character_errors=edits,reference_characters=characters,cer_percent=100*edits/characters,edit_based_character_accuracy_percent=100*(1-edits/characters),prediction_sha256=sha(out/'predictions.json'),no_new_inference=True,training_changed=False)
write(out/'summary.json',summary);write(out/'per_image_results.json',[dict(**r,label=labels[r['opaque_id']],correct=r['prediction']==labels[r['opaque_id']]) for r in values])
assert sha(out/'predictions.json')==read(out/'prediction_freeze.json')['sha256'];assert sha(old/'predictions.json')==freeze['sha256']
write(out/'verification.json',dict(passed=True,pixel_identities_verified=251,source_predictions_hash_verified=True,subset_frozen_before_labels=True,original_predictions_unchanged=True,model_not_loaded=True))
print(summary)
