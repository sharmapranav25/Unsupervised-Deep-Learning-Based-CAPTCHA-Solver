import json, hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
R=Path.cwd()
O=R/'model1/results/cryptbb/A/character_modes_epochs01_20'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj): (O/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
manifest=R/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json'
ids=sorted(x['opaque_id'] for x in read(manifest));assert len(ids)==len(set(ids))==251
pred={};sources=[]
for e in range(1,21):
 folder=R/f'model1/results/cryptbb/A/reserved251_cpu_epoch{e:02d}'
 p=folder/'predictions.json';f=read(folder/'prediction_freeze.json')
 assert sha(p)==f['sha256'] and f['images']==251
 rows=read(p);assert len(rows)==251
 pred[e]={r['opaque_id']:r['prediction'] for r in rows}
 assert sorted(pred[e])==ids and all(len(s)==5 for s in pred[e].values())
 sources.append(dict(epoch=e,path=str(p.relative_to(R)),sha256=sha(p),checkpoint_sha256=f['checkpoint_sha256']))
write('protocol.json',dict(created_at=datetime.now(timezone.utc).isoformat(),epochs=list(pred),sources=sources,manifest_sha256=sha(manifest),method='Same as current run: independent case-sensitive character modes at each observed position; any tied mode counts as agreement; each epoch contributes its own vote. Whole-string modes also retained. No tie-breaking.',observed_length=5,labels_used=False,selection_changed=False))
def vote(values):
 c=Counter(values);n=max(c.values());return dict(modes=sorted(k for k,v in c.items() if v==n),mode_frequency=n,vote_counts=dict(sorted(c.items())))
chars=[];whole=[]
for i in ids:
 strings=[pred[e][i] for e in pred]
 positions=[dict(position=j+1,**vote(s[j] for s in strings)) for j in range(5)]
 chars.append(dict(opaque_id=i,positions=positions,predictions_by_epoch=strings))
 whole.append(dict(opaque_id=i,**vote(strings)))
unique=sum(len(p['modes'])==1 for row in chars for p in row['positions'])
coverage=[]
for e in pred:
 matches=unique_matches=all5=whole_matches=0
 for cr,wr in zip(chars,whole):
  s=pred[e][cr['opaque_id']]
  hits=[s[j] in p['modes'] for j,p in enumerate(cr['positions'])]
  matches+=sum(hits);all5+=all(hits)
  unique_matches+=sum(h and len(p['modes'])==1 for h,p in zip(hits,cr['positions']))
  whole_matches+=s in wr['modes']
 coverage.append(dict(epoch=e,character_mode_matches=matches,total_positions=1255,character_agreement_percent=100*matches/1255,unique_character_mode_matches=unique_matches,unique_character_positions=unique,all5_character_modes_matched=all5,whole_string_mode_matches=whole_matches))
# Independently recompute agreement directly from the original predictions.
for row in coverage:
 e=row['epoch'];total=0
 for i in ids:
  for j in range(5):
   c=Counter(pred[k][i][j] for k in pred)
   total+=c[pred[e][i][j]]==max(c.values())
 assert total==row['character_mode_matches']
best=max(r['character_mode_matches'] for r in coverage)
bestwhole=max(r['whole_string_mode_matches'] for r in coverage)
summary=dict(images=251,epochs=20,positions=1255,unique_character_modes=unique,tied_character_modes=1255-unique,character_agreement_top_epochs=[r['epoch'] for r in coverage if r['character_mode_matches']==best],whole_string_agreement_top_epochs=[r['epoch'] for r in coverage if r['whole_string_mode_matches']==bestwhole],labels_used=False,selection_changed=False)
write('character_modes.json',chars);write('whole_string_modes.json',whole);write('epoch_character_coverage.json',coverage);write('summary.json',summary)
write('agreement_freeze.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),files={n:sha(O/n) for n in ['protocol.json','character_modes.json','whole_string_modes.json','epoch_character_coverage.json','summary.json']},independent_recount_passed=True))
# Only after agreement is frozen, join previously saved diagnostic accuracies.
accuracy=R/'model1/results/cryptbb/A/reserved251_all_epochs.json'
a={r['epoch']:r for r in read(accuracy)['rows']}
comparison=[dict(**row,correct=a[row['epoch']]['correct'],accuracy_percent=a[row['epoch']]['accuracy_percent']) for row in coverage]
write('comparison_with_saved_accuracy.json',dict(agreement_freeze_sha256=sha(O/'agreement_freeze.json'),accuracy_source_sha256=sha(accuracy),retrospective=True,rows=comparison))
print(json.dumps(dict(summary=summary,rows=comparison),indent=2))
