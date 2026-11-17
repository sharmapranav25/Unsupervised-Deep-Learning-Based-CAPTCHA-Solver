"""Preserve a failed prelaunch check before updating its verification contract."""
import json,hashlib,shutil
from pathlib import Path
R=Path('E:/pranav/Thesis-Pranav-Final')
S=R/'handoff/cryptbb_v3_feature_priority_v1_20261115'
assert not (R/'model3/runs/cryptbb_v3_feature_priority_v1_20261115/A').exists()
assert not (S/'readiness.json').exists()
archive=S/'prelaunch_revision2';archive.mkdir()
for name in ('protocol.json','check.log','inference_batch_diagnosis.json'):
    shutil.copyfile(S/name,archive/name)
shutil.copytree(S/'source',archive/'source')
p=json.loads((S/'protocol.json').read_text())
for row in p['sources']:
    source=R/row['path'];row['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copyfile(source,S/'source'/row['path'])
p['prelaunch_revision']=3
p['assessment_batch_convention']='Frozen ascending-ID batches256 plus final160. Exact same-batch replay; no universal invariance to batch size claimed. Early probe differed2strings/512 but0correct for both; mature probe512strings exact,348correct for both.'
(S/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
print('Preserved prelaunch revision2; refreshed source hashes for revision3')
