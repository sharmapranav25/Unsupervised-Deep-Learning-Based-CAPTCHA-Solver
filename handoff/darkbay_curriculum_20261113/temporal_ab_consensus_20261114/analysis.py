import json, hashlib
from pathlib import Path
from collections import Counter
from fractions import Fraction
from datetime import datetime, timezone
R=Path.cwd();S=R/'handoff/darkbay_curriculum_20261113'
O=S/'temporal_ab_consensus_20261114'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):(O/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),models=[1,2,3],epochs={'1':list(range(2,21)),'2':list(range(1,21)),'3':list(range(1,21))},pool='Historical matched900 DarkBay images, not the later600-image reserved evaluation pool',temporal_rule='For A and B separately, independent case-sensitive per-position modes across every eligible epoch; self-votes included; retain all tied modes; accept any tied mode. Average A and B agreement fractions to choose one shared epoch.',length_rule='Per image use maximum observed prediction length across eligible A/B predictions. Missing positions are explicit null votes, never silently dropped or filled from labels; no edit alignment or target-length constraint.',ab_rule='Exact full-string A/B agreement at the same epoch divided by900',combined_rule='0.5 * temporal character agreement fraction + 0.5 * exact A/B agreement fraction',tie_rule='Earliest epoch, exact rational comparison',label_access_during_selection=False,weights_tuned=False,training_or_primary_selection_changed=False,limitations=['Retrospective experiment after historical accuracy inspection','A/B and epochs have correlated errors','Different numbers of available epochs: Model1 epoch1 missing','Shared epoch selection for both readers; accuracy reported separately'])
write('protocol.json',protocol)
inv=read(S/'ab_epoch_agreement_20261113/prediction_inventory.json');pred={};ids=None
for r in inv:
 n,t,e=r['model'],r['dataset'],r['epoch']
 if e not in protocol['epochs'][str(n)]:continue
 p=R/r['path'];assert sha(p)==r['predictions_sha256']
 freeze=read(p.parent/'prediction_freeze.json')
 assert freeze['checkpoint_sha256']==r['checkpoint_sha256'] and freeze['predictions_sha256']==sha(p)
 rows=read(p);d={x['opaque_id']:x['prediction'] for x in rows};assert len(rows)==len(d)==900
 if ids is None:ids=sorted(d)
 assert sorted(d)==ids
 pred[n,t,e]=d
write('source_inventory.json',inv)
curves=[];choices=[];votes=[];stats=[]
for n in protocol['models']:
 epochs=protocol['epochs'][str(n)]
 widths={i:max(len(pred[n,t,e][i]) for t in 'AB' for e in epochs) for i in ids}
 denominator=sum(widths.values());modes={};tied=Counter();missing=Counter()
 for t in 'AB':
  for i in ids:
   positions=[]
   for j in range(widths[i]):
    v=[pred[n,t,e][i][j] if j<len(pred[n,t,e][i]) else None for e in epochs]
    c=Counter(v);top=max(c.values());candidates=[k for k,vv in c.items() if vv==top]
    modes[t,i,j]=set(candidates);tied[t]+=len(candidates)>1;missing[t]+=c.get(None,0)
    positions.append(dict(position=j+1,modes=candidates,frequency=top,votes=[dict(character=k,count=v) for k,v in c.items()]))
   votes.append(dict(model=n,dataset=t,opaque_id=i,positions=positions))
 eligible=[]
 for e in epochs:
  counts={}
  for t in 'AB':
   total=0
   for i in ids:
    s=pred[n,t,e][i]
    for j in range(widths[i]):total+=(s[j] if j<len(s) else None) in modes[t,i,j]
   counts[t]=total
  ab=sum(pred[n,'A',e][i]==pred[n,'B',e][i] for i in ids)
  temporal=Fraction(counts['A']+counts['B'],2*denominator);agreement=Fraction(ab,900);combined=(temporal+agreement)/2
  row=dict(model=n,epoch=e,A_character_matches=counts['A'],B_character_matches=counts['B'],positions_per_reader=denominator,A_temporal_percent=100*counts['A']/denominator,B_temporal_percent=100*counts['B']/denominator,temporal_percent=float(100*temporal),ab_agree=ab,ab_percent=float(100*agreement),combined_percent=float(100*combined))
  curves.append(row);eligible.append((row,temporal,agreement,combined))
 for rule,index in [('temporal',1),('ab',2),('combined',3)]:
  chosen=max(eligible,key=lambda x:(x[index],-x[0]['epoch']))
  ties=[x[0]['epoch'] for x in eligible if x[index]==chosen[index]]
  choices.append(dict(model=n,rule=rule,epoch=chosen[0]['epoch'],all_maximizing_epochs=ties,score_percent=float(100*chosen[index])))
 stats.append(dict(model=n,positions_per_reader=denominator,tied_positions=dict(tied),missing_position_votes=dict(missing)))
write('character_votes.json',votes);write('curves.json',curves);write('vote_statistics.json',stats)
write('selection_freeze.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),choices=choices,files={name:sha(O/name) for name in ['protocol.json','source_inventory.json','character_votes.json','curves.json','vote_statistics.json','analysis.py']},labels_read=False))
# Verify chosen scores by independently recounting direct vote frequencies.
for choice in choices:
 n,e=choice['model'],choice['epoch'];epochs=protocol['epochs'][str(n)]
 row=next(r for r in curves if r['model']==n and r['epoch']==e)
 for t in 'AB':
  hits=0
  for i in ids:
   width=max(len(pred[n,tt,ee][i]) for tt in 'AB' for ee in epochs)
   for j in range(width):
    c=Counter(pred[n,t,ee][i][j] if j<len(pred[n,t,ee][i]) else None for ee in epochs)
    s=pred[n,t,e][i];key=s[j] if j<len(s) else None
    hits+=c[key]==max(c.values())
  assert hits==row[t+'_character_matches']
# Accuracy is joined only after the selection freeze; no raw labels are needed.
scored_source=S/'ab_epoch_agreement_20261113/scored_curves.json'
scored={(r['model'],r['epoch']):r for r in read(scored_source)}
for row in curves:assert row['ab_agree']==scored[row['model'],row['epoch']]['raw_agree']
results=[];best_observed=[]
for c in choices:
 r=scored[c['model'],c['epoch']]
 results.append(dict(**c,A_correct=r['A_correct'],B_correct=r['B_correct'],A_accuracy_percent=r['A_accuracy_percent'],B_accuracy_percent=r['B_accuracy_percent']))
for n in protocol['models']:
 for t in 'AB':
  rr=[scored[n,e] for e in protocol['epochs'][str(n)]];best=max(r[t+'_correct'] for r in rr)
  best_observed.append(dict(model=n,dataset=t,correct=best,accuracy_percent=best/9,epochs=[r['epoch'] for r in rr if r[t+'_correct']==best]))
write('results.json',dict(selection_freeze_sha256=sha(O/'selection_freeze.json'),scored_source_sha256=sha(scored_source),results=results,best_observed_per_reader=best_observed,independent_recount_passed=True,existing_ab_counts_reproduced=True))
print(json.dumps(dict(results=results,best_observed_per_reader=best_observed,statistics=stats),indent=2))
