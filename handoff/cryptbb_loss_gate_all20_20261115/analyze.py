"""Retrospective full20 estimation with loss<0.1 winner eligibility; label-isolated selection."""
import sys,json,hashlib,re
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('E:/pranav/Thesis-Pranav-Final')
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower().replace('\\','/')
        if any(x in p for x in ('scoring_only/','labels.json','summary.json','per_image_results.json','results.json','all_measurements')):
            raise RuntimeError('Scoring data forbidden during selection: '+p)
def select():
    sys.addaudithook(audit)
    import numpy as np
    source=ROOT/'analysis/checkpoint_selection/ckpt_select/estimator.py'
    assert sha(source)=='dee14c99919ffb4d51ba47b2104f766cea584dc113a995ab857a037539421150'
    sys.path.insert(0,str(source.parent.parent))
    from ckpt_select.estimator import rank1,choose_k,mask_for,CONFIG
    manifest=ROOT/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json'
    ids=sorted(r['opaque_id'] for r in read(manifest));assert len(ids)==251
    assert read(ROOT/'handoff/cryptbb_v3_epoch20_static_B_20261115/report_verification.json')['complete']
    runs=[]
    for number,tag in [(1,'A'),(1,'B'),(2,'A'),(2,'B')]:
        family='cryptbb_v3' if number==1 else f'cryptbb_v3_epoch20_static_{tag}_20261115'
        run=ROOT/f'model{number}/runs'/family/tag
        losses={}
        if number==1:
            loss_path=run/'train.log'
            for line in loss_path.read_text(encoding='utf8').splitlines():
                m=re.search(r'epoch: \[(\d+)/20\], global_step: (\d+), lr: [^,]+, loss: ([0-9.]+)',line)
                if m and int(m[2])==int(m[1])*782:losses[int(m[1])]=float(m[3])
            definition='End-of-epoch logged rolling median; not full-epoch mean'
        else:
            loss_path=run/'history.json'
            losses={r['epoch']:r['mean_loss'] for r in read(loss_path)}
            definition='Full-epoch mean of sampled training-batch losses'
        assert sorted(losses)==list(range(1,21))
        eligible=sorted(e for e,v in losses.items() if v<0.1)
        assert eligible, 'No eligible checkpoint; do not silently invent a fallback'
        predictions=[];sources=[];missing=[]
        for e in range(1,21):
            parent=ROOT/f'model{number}/results'/family/tag
            candidates=[parent/f'epoch_{e:02d}'] if number==2 else [parent/f'requested_cpu_epoch{e:02d}',*sorted(parent.glob(f'requested_cpu_epoch{e:02d}_attempt*'))]
            available=[p for p in candidates if (p/'prediction_freeze.json').exists() and (p/'verification.json').exists()]
            if not available:missing.append(e);continue
            p=available[-1];freeze=read(p/'prediction_freeze.json');protocol=read(p/'protocol.json')
            assert read(p/'verification.json')['passed']
            assert protocol['model']==number and protocol['dataset']==tag and protocol['epoch']==e
            assert protocol['manifest_sha256']==sha(manifest)
            ck=run/(f'epoch_{e}.pth' if number==1 else f'checkpoints/epoch_{e:02d}.pth')
            assert sha(ck)==freeze['checkpoint_sha256']==protocol['checkpoint_sha256']
            pred_path=p/'predictions.json';assert sha(pred_path)==freeze['sha256']
            rows=read(pred_path);values={r['opaque_id']:r['prediction'] for r in rows}
            assert len(rows)==len(values)==251 and sorted(values)==ids
            assert all(set(r)=={'opaque_id','prediction'} for r in rows)
            predictions.append([values[i] for i in ids]);sources.append(dict(epoch=e,path=str(p),prediction_sha256=sha(pred_path),checkpoint_sha256=freeze['checkpoint_sha256']))
        epochs=np.array([s['epoch'] for s in sources]);pred=np.array(predictions,object)
        assert not missing and epochs.tolist()==list(range(1,21))
        agreement=(pred[:,None,:]==pred[None,:,:]).mean(2)
        masks={k:mask_for(epochs,k) for k in CONFIG['k_values']}
        supported=[k for k,mask in masks.items() if mask.sum(axis=1).min()>0]
        invalid=[dict(k=k,isolated_epochs=epochs[mask.sum(axis=1)==0].tolist()) for k,mask in masks.items() if k not in supported]
        assert supported==list(range(6)) and not invalid
        fits={k:rank1(agreement,epochs,k) for k in CONFIG['k_values']}
        assert fits and all(f['converged'] for f in fits.values())
        k,criteria,passed=choose_k(agreement,epochs,fits)
        scores=fits[k]['scores'];all_order=sorted(range(len(epochs)),key=lambda i:(-float(scores[i]),int(epochs[i])))
        order=[i for i in all_order if int(epochs[i]) in eligible]
        assert order and float(scores[order[0]])==max(float(scores[e-1]) for e in eligible)
        record=dict(model=number,dataset=tag,family=family,threshold=0.1,comparison='strictly below',loss_definition=definition,
            loss_source=str(loss_path),loss_source_sha256=sha(loss_path),recorded_losses=losses,eligible_epochs=eligible,
            candidate_epochs=epochs.tolist(),pending_evaluation_epochs=missing,provisional=max(losses)<20 or bool(missing),
            original_unmodified_selector_applicable=not invalid,unsupported_masks=invalid,supported_masks=supported,
            adaptation='No estimator adaptation: original full20 fits and masks0-5; loss gate applied only to winner eligibility',
            unrestricted_selected_epoch=int(epochs[all_order[0]]),all20_ranking=[int(epochs[i]) for i in all_order],
            eligible_ranking=[int(epochs[i]) for i in order],
            selected_epoch=int(epochs[order[0]]),top3=[int(epochs[i]) for i in order[:3]],scores=scores.tolist(),resolved_k=k,
            fallback_used=not passed,mask_diagnostics=criteria,triplet_counts=fits[k]['triplet_counts'],
            labels_read=False,training_changed=False,prediction_sources=sources)
        runs.append(record)
        print(json.dumps({key:record[key] for key in ('model','dataset','candidate_epochs','selected_epoch','top3','resolved_k','fallback_used','provisional')}),flush=True)
    write(OUT/'selection.json',dict(created_at=datetime.now(timezone.utc).isoformat(),retrospective=True,
        threshold_user_requested=0.1,estimator_sha256=sha(source),estimator_config=CONFIG,runs=runs,labels_read=False,
        note='Original full20 estimator unchanged. Retrospective winner-eligibility gate only; official production selections preserved.'))
def score():
    selection=read(OUT/'selection.json');rows=[]
    for r in selection['runs']:
        scores=[]
        for e in r['candidate_epochs']:
            src=next(s for s in r['prediction_sources'] if s['epoch']==e);p=Path(src['path'])
            assert sha(p/'predictions.json')==src['prediction_sha256']
            per=read(p/'per_image_results.json');summary=read(p/'summary.json')
            assert len(per)==251
            correct=sum(x['prediction']==x['label'] for x in per)
            assert correct==summary['correct'] and summary['checkpoint_sha256']==src['checkpoint_sha256']
            scores.append(dict(epoch=e,recorded_loss=r['recorded_losses'][str(e)],eligible=e in r['eligible_epochs'],
                estimated_score=r['scores'][e-1],correct=correct,total=251,accuracy_percent=100*correct/251))
        rows.append(dict(model=r['model'],dataset=r['dataset'],candidate_epochs=r['candidate_epochs'],
            eligible_epochs=r['eligible_epochs'],provisional=r['provisional'],
            selected=next(s for s in scores if s['epoch']==r['selected_epoch']),
            unrestricted_selected=next(s for s in scores if s['epoch']==r['unrestricted_selected_epoch']),
            final_epoch=scores[-1],top3=[next(s for s in scores if s['epoch']==e) for e in r['top3']],
            all_epochs=scores,all20_mean_accuracy_percent=sum(s['accuracy_percent'] for s in scores)/20,
            resolved_k=r['resolved_k'],fallback_used=r['fallback_used']))
    write(OUT/'scored_results.json',dict(selection_sha256=sha(OUT/'selection.json'),selection_frozen_before_scoring=True,rows=rows))
    print(json.dumps([{k:v for k,v in r.items() if k!='all_epochs'} for r in rows],indent=2))
if __name__=='__main__':
    if sys.argv[1]=='select':select()
    elif sys.argv[1]=='score':score()
    else:raise ValueError(sys.argv[1])
