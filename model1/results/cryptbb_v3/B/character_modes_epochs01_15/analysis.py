import json,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
R=Path.cwd();B=R/'model1/results/cryptbb_v3/B';W=B/'prediction_modes_epochs01_15';C=B/'character_modes_epochs01_15'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def vote(values):
 c=Counter(values);f=max(c.values());m=sorted(k for k,v in c.items() if v==f)
 return dict(mode=m[0] if len(m)==1 else None,modes=m,mode_frequency=f,tied=len(m)>1,vote_counts=dict(sorted(c.items())))
manifest=R/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json'
ids=sorted(r['opaque_id'] for r in read(manifest));assert len(set(ids))==len(ids)==251
pred={};sources=[]
for e in range(1,16):
 p=B/f'requested_cpu_epoch{e:02d}'/'predictions.json';f=read(p.parent/'prediction_freeze.json');assert sha(p)==f['sha256']
 rows=read(p);d={r['opaque_id']:r['prediction'] for r in rows};assert len(rows)==251 and sorted(d)==ids and all(len(s)==5 for s in d.values())
 pred[e]=d;sources.append(dict(epoch=e,path=str(p.relative_to(R)),sha256=sha(p),checkpoint_sha256=f['checkpoint_sha256']))
protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),epochs=list(pred),sources=sources,manifest_sha256=sha(manifest),method='Same exact case-sensitive whole-string and independent character-position votes as epochs1-13. Self-votes included; all tied modes preserved; coverage accepts any tied candidate; no tiebreak.',observed_prediction_length=5,labels_used=False,training_or_selection_changed=False)
write(W/'protocol.json',protocol);write(C/'protocol.json',protocol)
whole=[];chars=[]
for i in ids:
 strings=[pred[e][i] for e in pred]
 whole.append(dict(opaque_id=i,**vote(strings),predictions_by_epoch=strings))
 positions=[dict(position=j+1,**vote(s[j] for s in strings)) for j in range(5)]
 consensus=''.join(p['mode'] for p in positions) if not any(p['tied'] for p in positions) else None
 chars.append(dict(opaque_id=i,positions=positions,unique_consensus=consensus,has_tied_positions=consensus is None,consensus_seen_in_any_epoch=consensus in strings if consensus is not None else None,predictions_by_epoch=strings))
write(W/'modes.json',whole);write(C/'character_modes.json',chars)
wsummary=dict(images=251,epochs_per_image=15,unique_modes=sum(not r['tied'] for r in whole),tied_modes=sum(r['tied'] for r in whole),strict_majority=sum(r['mode_frequency']>=8 for r in whole),mode_frequency_distribution=dict(sorted(Counter(r['mode_frequency'] for r in whole).items())),labels_used=False)
unique=sum(not p['tied'] for r in chars for p in r['positions'])
csummary=dict(images=251,epochs=15,observed_length_all_predictions=5,positions=1255,unique_character_modes=unique,tied_character_modes=1255-unique,positions_with_strict_majority=sum(p['mode_frequency']>=8 for r in chars for p in r['positions']),images_with_unique_consensus=sum(not r['has_tied_positions'] for r in chars),images_with_tied_positions=sum(r['has_tied_positions'] for r in chars),unique_consensus_not_seen_in_any_epoch=sum(r['consensus_seen_in_any_epoch'] is False for r in chars),labels_used=False)
wc=[];cc=[]
for e in pred:
 u=sum(not r['tied'] and pred[e][r['opaque_id']] in r['modes'] for r in whole)
 t=sum(r['tied'] and pred[e][r['opaque_id']] in r['modes'] for r in whole)
 wc.append(dict(epoch=e,unique_mode_matches=u,tied_mode_candidate_matches=t,total_mode_matches=u+t,mode_agreement_percent=100*(u+t)/251,strict_majority_matches=sum(r['mode_frequency']>=8 and pred[e][r['opaque_id']] in r['modes'] for r in whole)))
 hits=uhits=all5=0
 for r in chars:
  s=pred[e][r['opaque_id']];h=[s[j] in p['modes'] for j,p in enumerate(r['positions'])]
  hits+=sum(h);uhits+=sum(v and not p['tied'] for v,p in zip(h,r['positions']));all5+=all(h)
 cc.append(dict(epoch=e,character_mode_matches=hits,total_positions=1255,character_agreement_percent=100*hits/1255,unique_character_mode_matches=uhits,unique_character_positions=unique,tied_character_candidate_matches=hits-uhits,all5_character_modes_matched=all5,whole_string_mode_matches=u+t))
 # Independent direct frequency check from predictions.
 recount=0
 for i in ids:
  for j in range(5):
   counts=Counter(pred[k][i][j] for k in pred);recount+=counts[pred[e][i][j]]==max(counts.values())
 assert recount==hits
 assert u+t==sum(Counter(pred[k][i] for k in pred)[pred[e][i]]==max(Counter(pred[k][i] for k in pred).values()) for i in ids)
write(W/'epoch_mode_coverage.json',wc);write(C/'epoch_character_coverage.json',cc)
write(W/'summary.json',wsummary);write(C/'summary.json',csummary)
for folder in [W,C]:write(folder/'verification.json',dict(passed=True,identities=251,predictions=3765,epochs=15,source_hashes_verified=True,independent_recounts_passed=True,labels_read=False,files={p.name:sha(p) for p in folder.glob('*.json')}))
print(json.dumps(dict(whole_summary=wsummary,character_summary=csummary,ranked_whole=sorted(wc,key=lambda r:-r['total_mode_matches']),ranked_character=sorted(cc,key=lambda r:-r['character_mode_matches'])),indent=2))
