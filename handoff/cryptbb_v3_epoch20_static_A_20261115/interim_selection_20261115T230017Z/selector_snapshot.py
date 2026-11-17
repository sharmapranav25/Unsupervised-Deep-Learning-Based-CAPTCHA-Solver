"""Prediction-only final application of the unchanged original v1 selector."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
TAG=sys.argv[1]
assert TAG in ('A','B')
FAMILY='cryptbb_v3_epoch20_static_A_20261115'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower().replace('\\','/')
        if any(x in p for x in ('scoring_only/','labels.json','summary.json','per_image_results.json','results.json','all_measurements')):
            raise RuntimeError('Scoring data forbidden during selection')
sys.addaudithook(audit)
source=ROOT/'analysis/checkpoint_selection/ckpt_select/estimator.py'
assert sha(source)=='dee14c99919ffb4d51ba47b2104f766cea584dc113a995ab857a037539421150'
sys.path.insert(0,str(source.parent.parent))
from ckpt_select.estimator import rank1,choose_k,CONFIG
run=ROOT/'model2/runs'/FAMILY/TAG
epochs=np.arange(1,17)
assert epochs.tolist()==list(range(1,17))
manifest=ROOT/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json'
ids=sorted(r['opaque_id'] for r in read(manifest));assert len(ids)==251
predictions=[];sources=[]
for epoch in epochs.tolist():
    out=ROOT/'model2/results'/FAMILY/TAG/f'epoch_{epoch:02d}'
    p=out/'predictions.json';fr=read(out/'prediction_freeze.json');protocol=read(out/'protocol.json')
    assert protocol['model']==2 and protocol['dataset']==TAG and protocol['epoch']==epoch
    assert protocol['manifest_sha256']==sha(manifest)
    ck=run/f'checkpoints/epoch_{epoch:02d}.pth'
    assert sha(ck)==fr['checkpoint_sha256']==protocol['checkpoint_sha256']
    assert sha(p)==fr['sha256']
    rows=read(p);assert all(set(r)=={'opaque_id','prediction'} for r in rows)
    values={r['opaque_id']:r['prediction'] for r in rows}
    assert len(rows)==len(values)==251 and sorted(values)==ids
    predictions.append([values[i] for i in ids])
    sources.append(dict(epoch=epoch,path=str(p),sha256=sha(p),checkpoint_sha256=fr['checkpoint_sha256']))
pred=np.array(predictions,object);agreement=(pred[:,None,:]==pred[None,:,:]).mean(2)
fits={k:rank1(agreement,epochs,k) for k in range(6)}
assert all(f['converged'] for f in fits.values())
k,criteria,passed=choose_k(agreement,epochs,fits)
scores=fits[k]['scores'];order=sorted(range(len(epochs)),key=lambda i:(-float(scores[i]),int(epochs[i])))
result=dict(created_at=datetime.now(timezone.utc).isoformat(),model=2,dataset=TAG,provisional=True,
    selector='masked_rank1_v1_exact_auto',estimator_sha256=sha(source),estimator_config=CONFIG,
    candidate_epochs=epochs.tolist(),selected_epoch=int(epochs[order[0]]),top3=[int(epochs[i]) for i in order[:3]],
    scores=scores.tolist(),resolved_k=k,fallback_used=not passed,mask_diagnostics=criteria,
    labels_read=False,training_changed=False,prediction_sources=sources,
    note='Provisional selection across available epochs1-16. Factors are not accuracy estimates.')
dest=Path(sys.argv[2]);assert not dest.exists()
dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps(dict(path=str(dest),**{k:v for k,v in result.items() if k not in ('prediction_sources','estimator_config','mask_diagnostics')}),indent=2))
