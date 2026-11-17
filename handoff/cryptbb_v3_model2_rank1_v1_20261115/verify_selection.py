"""Reproduce the adopted prediction-only selector; never open scoring labels."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path('E:/pranav/Thesis-Pranav-Final')
CAPRI=Path('E:/pranav/CAPRI')
OUT=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower().replace('\\','/')
        if any(x in p for x in ('scoring_only/','labels.json','summary.json','evaluation.json','evaluation.csv','scored_curves.json')):
            raise RuntimeError('Scoring data forbidden during selection')
sys.addaudithook(audit)
source=ROOT/'analysis/checkpoint_selection/ckpt_select/estimator.py'
assert sha(source)=='dee14c99919ffb4d51ba47b2104f766cea584dc113a995ab857a037539421150'
sys.path.insert(0,str(source.parent.parent))
from ckpt_select.estimator import rank1,choose_k,CONFIG
assert read(CAPRI/'configs/checkpoint_analysis_defaults.json')['primary_selector']['id']=='masked_rank1_v1_exact_auto'
baseline={r['run']:r['baseline'] for r in read(CAPRI/'runs/rank1_improvement_tests_20261115/rankings.json')}
manifest=ROOT/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json'
ids=sorted(r['opaque_id'] for r in read(manifest));assert len(ids)==251
results={}
for tag,folder in [('A','current_v3_a_rank1_20261115'),('B','v1_masked_rank1_all_20261115')]:
    folder=CAPRI/'runs'/folder
    freeze=read(folder/'selection_freeze.json')
    for name in ('prediction_sources.json','rankings.json','protocol.json'):
        assert sha(folder/name)==freeze['files'][name]
    sources=read(folder/'prediction_sources.json')
    if tag=='B':sources=sources['CryptBB251/Model1-B']
    assert [s['epoch'] for s in sources]==list(range(1,21))
    predictions=[]
    for s in sources:
        p=Path(s['path']);assert p.is_relative_to(ROOT/f'model1/results/cryptbb_v3/{tag}')
        assert sha(p)==s['sha256']
        rows=read(p);assert all(set(r)=={'opaque_id','prediction'} for r in rows)
        values={r['opaque_id']:r['prediction'] for r in rows}
        assert len(rows)==len(values)==251 and sorted(values)==ids
        protocol=read(p.parent/'protocol.json');fr=read(p.parent/'prediction_freeze.json')
        assert protocol['epoch']==s['epoch'] and protocol['dataset']==tag and protocol['model']==1
        assert protocol['manifest_sha256']==sha(manifest)
        assert protocol['checkpoint_sha256']==fr['checkpoint_sha256']==s['checkpoint_sha256']
        assert fr['sha256']==s['sha256']
        receipt=read(ROOT/f"model1/runs/cryptbb_v3/{tag}/epoch_{s['epoch']}_receipt.json")
        assert receipt['sha256']==s['checkpoint_sha256']
        predictions.append([values[i] for i in ids])
    pred=np.array(predictions,object);agreement=(pred[:,None,:]==pred[None,:,:]).mean(2);epochs=np.arange(1,21)
    fits={k:rank1(agreement,epochs,k) for k in range(6)}
    assert all(f['converged'] for f in fits.values())
    k,criteria,passed=choose_k(agreement,epochs,fits);scores=fits[k]['scores']
    order=sorted(range(20),key=lambda i:(-float(scores[i]),i+1))
    old=baseline['CryptBB251/Model1-A-current-v3' if tag=='A' else 'CryptBB251/Model1-B']
    np.testing.assert_allclose(scores,old['scores'],rtol=0,atol=1e-10)
    epoch=order[0]+1;assert epoch==old['selected_epoch']
    ck=ROOT/f'model1/runs/cryptbb_v3/{tag}/epoch_{epoch}.pth'
    actual=sha(ck);assert actual==sources[epoch-1]['checkpoint_sha256']
    results[tag]=dict(selected_epoch=epoch,checkpoint=ck.relative_to(ROOT).as_posix(),checkpoint_sha256=actual,
        top3=[i+1 for i in order[:3]],scores=scores.tolist(),resolved_k=k,mask_criterion_passed=passed,
        fallback_used=not passed,mask_diagnostics=criteria,prediction_sources=sources,
        evidence_freeze=str(folder/'selection_freeze.json'),evidence_freeze_sha256=sha(folder/'selection_freeze.json'))
result=dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),selector='masked_rank1_v1_exact_auto',
    estimator_sha256=sha(source),estimator_config=CONFIG,candidate_epochs=list(range(1,21)),images=251,
    image_manifest_sha256=sha(manifest),rule='Exact case-sensitive whole strings; original rank1 at k0-5; original choose_k; highest factor, earliest exact tie',
    warning='All automatic mask checks failed for both runs; original fallback k5 retained. Factors are not accuracy estimates.',
    labels_read=False,selected=results,cross_teachers={'A':'B','B':'A'},initialization='Independent scratch; selected checkpoints rank only',
    verifier_sha256=sha(Path(__file__)))
dest=OUT/'teacher_selection.json';assert not dest.exists()
dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({t:{k:v for k,v in r.items() if k in ('selected_epoch','top3','checkpoint_sha256','resolved_k','fallback_used')} for t,r in results.items()},indent=2))
