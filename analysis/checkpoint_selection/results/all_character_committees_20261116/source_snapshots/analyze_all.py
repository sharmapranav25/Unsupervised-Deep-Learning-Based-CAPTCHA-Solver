"""Apply frozen11/20 character committee independently to every available study run."""
import collections,datetime,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path('E:/pranav/Thesis-Pranav-Final')
OUT=Path(__file__).parent
ORIGINAL=Path('E:/pranav/CAPRI/runs/darkbay_m3a_character_committee_20261116/analyze.py')
EXPECTED='0e0ad07390cbc82c0fdda94154aa1d0ccabcf8a02f0d27889d49dfdbbe6af9fe'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
assert sha(ORIGINAL)==EXPECTED
spec=importlib.util.spec_from_file_location('frozen_committee',ORIGINAL);C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
MANIFESTS={'CryptBB':ROOT/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json','DarkBay':ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json'}
LABELS={'CryptBB':ROOT/'evaluation_data/model4_reservation_20261114/scoring_only/cryptbb/evaluation_pool_labels.json','DarkBay':ROOT/'evaluation_data/darkbay/scoring_labels/scoring_labels.json'}
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0]).lower().replace('\\','/')
        if any(k in path for k in ('scoring_only','scoring_labels','labels.json','summary.json','per_image_results','scored_results','history.json')):
            raise RuntimeError('Scoring data forbidden during selection: '+path)

def inventory():
    ids={d:sorted(x['opaque_id'] for x in read(p)) for d,p in MANIFESTS.items()}
    groups=collections.defaultdict(lambda:collections.defaultdict(list));excluded=[];archive_counts=[]
    roots=[]
    for model in (1,2,3):
        base=ROOT/f'model{model}/results'
        for family in sorted(base.iterdir()):
            if not family.is_dir():continue
            if family.name=='historical':
                archive_counts.append(dict(path=str(family),reason='Older archived experiments outside the canonical A/B study series',frozen_prediction_files=len(list(family.rglob('prediction_freeze.json')))))
                continue
            if not family.name.startswith(('cryptbb','darkbay')):continue
            roots.append((model,family.name,family))
    for family in ('cryptbb_v3_feature_priority_v1_20261115','cryptbb_v3_feature_priority_B_v1_20261115'):
        roots.append((3,family,ROOT/'model3/runs'/family/'diagnostics/real_cryptbb'))
    for model,family,base in roots:
        for freeze_path in sorted(base.rglob('prediction_freeze.json')):
            if any(p in ('controller4_validation','synthetic_training_epoch12') for p in freeze_path.parts):continue
            dest=freeze_path.parent;pp=dest/'predictions.json';protocol_path=dest/'protocol.json'
            if not pp.exists() or not protocol_path.exists():continue
            protocol=read(protocol_path);freeze=read(freeze_path)
            epoch=protocol.get('epoch');tag=protocol.get('dataset')
            if not isinstance(epoch,int) or not 1<=epoch<=20 or tag not in ('A','B'):continue
            domain='CryptBB' if family.startswith('cryptbb') else 'DarkBay'
            count=protocol.get('images',freeze.get('images'))
            if count!=len(ids[domain]):continue
            digest=sha(pp);expected=freeze.get('sha256',freeze.get('predictions_sha256'))
            assert digest==expected,str(pp)
            rows=read(pp)
            assert all(set(r)=={'opaque_id','prediction'} for r in rows),str(pp)
            mapping={r['opaque_id']:r['prediction'] for r in rows}
            assert len(rows)==len(mapping)==len(ids[domain]) and sorted(mapping)==ids[domain],str(pp)
            assert all(isinstance(s,str) for s in mapping.values())
            assert freeze['checkpoint_sha256']==protocol['checkpoint_sha256']
            if 'manifest_sha256' in protocol:assert protocol['manifest_sha256']==sha(MANIFESTS[domain]),str(pp)
            run_id=f'{domain}_M{model}_{family}_{tag}'
            rec=dict(run_id=run_id,domain=domain,model=model,family=family,dataset=tag,epoch=epoch,path=str(pp),prediction_sha256=digest,checkpoint_sha256=freeze['checkpoint_sha256'],protocol_sha256=sha(protocol_path),freeze_sha256=sha(freeze_path),manifest_sha256=sha(MANIFESTS[domain]),manifest_hash_in_original_protocol='manifest_sha256' in protocol)
            groups[run_id][epoch].append(rec)
    chosen=[]
    for run_id,epochs in sorted(groups.items()):
        selected=[];duplicates=[]
        for epoch,candidates in sorted(epochs.items()):
            assert len({x['checkpoint_sha256'] for x in candidates})==1,(run_id,epoch,'Different checkpoints under same run')
            def priority(c):
                name=Path(c['path']).parent.name
                return (0 if name==f'epoch_{epoch:02d}' else 1 if name.startswith('reserved251_cpu') else 2 if name.startswith('requested_cpu') else 3 if name.startswith('interim_cuda') else 4,name)
            candidates.sort(key=priority);pick=candidates[0];selected.append(pick)
            if len(candidates)>1:duplicates.append(dict(epoch=epoch,selected=pick['path'],alternatives=[x['path'] for x in candidates[1:]],identical_prediction_hashes=len({c['prediction_sha256'] for c in candidates})==1))
        info={k:selected[0][k] for k in ('run_id','domain','model','family','dataset')}
        info.update(available_epochs=sorted(epochs),missing_epochs=[e for e in range(1,21) if e not in epochs],sources=selected,duplicate_resolution=duplicates)
        if info['missing_epochs']:excluded.append(info)
        else:chosen.append(info)
    return chosen,excluded,archive_counts

def select():
    sys.addaudithook(guard);C.tests()
    assert not (OUT/'selection_freeze.json').exists(),'Preserve frozen analysis'
    protocol=dict(at=now(),original_method_sha256=EXPECTED,original_protocol_sha256=sha(ORIGINAL.parent/'protocol.json'),implementation_sha256=sha(Path(__file__)),threshold=11,committee_size=20,epochs=list(range(1,21)),normalization='ASCII letter case ignored; digits exact',skipping='No11/20 character majority; missing/nonalphanumeric cannot be keys',candidate_scoring='Same fixed key denominator; missing candidate character mismatches; self-vote included',positions='Left-aligned; widths from maximum prediction length per image',tie='Earliest',loss_filter=None,domain_sizes={'CryptBB':251,'DarkBay':900},scope='Canonical A/B study series under model1/2/3 results, plus CryptBB feature-priority diagnostics; complete20 prediction sets only; old historical archive separately inventoried',duplicate_policy='Prefer canonical epoch_XX, reserved251_cpu, requested_cpu, interim_cuda, then other; lexical order within preference; checkpoint hashes must agree',official_selections_changed=False,real_scores_read_after_all_choices_frozen=True)
    write(OUT/'protocol.json',protocol)
    groups,incomplete,archives=inventory();write(OUT/'inventory.json',dict(complete=groups,incomplete=incomplete,archives=archives))
    results=[]
    for group in groups:
        mapping={r['epoch']:{x['opaque_id']:x['prediction'] for x in read(Path(r['path']))} for r in group['sources']}
        ids=sorted(mapping[1]);counts={e:0 for e in range(1,21)};keys=[];width_total=0;n=0;images=0;support=collections.Counter()
        for opaque in ids:
            strings=[mapping[e][opaque] for e in range(1,21)];width=max(map(len,strings));width_total+=width;keyrows=[]
            for position in range(width):
                k=C.key([s[position] if position<len(s) else None for s in strings])
                if k is None:continue
                char,votes=k;n+=1;support[votes]+=1;keyrows.append(dict(position=position,character=char,votes=votes))
                for e,s in zip(range(1,21),strings):counts[e]+=int(position<len(s) and C.norm(s[position])==char)
            images+=bool(keyrows);keys.append(dict(opaque_id=opaque,keys=keyrows,predicted_width=width))
        if not n:
            raise RuntimeError('No usable keys: '+group['run_id'])
        epoch,ties=C.choose(counts)
        for e in range(1,21):
            recounted=sum(k['position']<len(mapping[e][r['opaque_id']]) and C.norm(mapping[e][r['opaque_id']][k['position']])==k['character'] for r in keys for k in r['keys'])
            assert recounted==counts[e]
        keypath=OUT/'keys'/f"{group['run_id']}.json";write(keypath,keys)
        result=dict(**group,selected_epoch=epoch,tied_epochs=ties,keyed_positions=n,predicted_positions=width_total,key_coverage_percent=100*n/width_total,images_with_keys=images,images=len(ids),support_histogram=dict(sorted(support.items())),keys_path=str(keypath),keys_sha256=sha(keypath),scores=[dict(epoch=e,matches=counts[e],agreement_percent=100*counts[e]/n) for e in range(1,21)])
        if group['family']=='darkbay_v3_feature_20261115' and group['model']==3 and group['dataset']=='A':
            original=read(ORIGINAL.parent/'selection.json');assert epoch==original['selected_epoch'] and ties==original['tied_epochs'] and n==original['coverage']['keyed_positions'] and [r['matches'] for r in result['scores']]==[r['matches'] for r in original['scores']]
        results.append(result);print('SELECTED',group['run_id'],epoch,'ties',ties,'keys',n,'/',width_total,flush=True)
    write(OUT/'selection.json',dict(at=now(),runs=results,protocol_sha256=sha(OUT/'protocol.json'),inventory_sha256=sha(OUT/'inventory.json'),fixtures_passed=True,independent_recount_passed=True,original_m3a_reproduced=True,labels_access_denied=True))
    write(OUT/'selection_freeze.json',dict(at=now(),selection_sha256=sha(OUT/'selection.json'),protocol_sha256=sha(OUT/'protocol.json'),inventory_sha256=sha(OUT/'inventory.json')))
    print('FROZEN',len(results),'complete runs;',len(incomplete),'incomplete',flush=True)

def score():
    freeze=read(OUT/'selection_freeze.json')
    for name in ('selection','protocol','inventory'):assert sha(OUT/(name+'.json'))==freeze[name+'_sha256']
    assert read(OUT/'protocol.json')['implementation_sha256']==sha(Path(__file__))
    selection=read(OUT/'selection.json');labels={d:read(p) for d,p in LABELS.items()};results=[]
    for run in selection['runs']:
        truth=labels[run['domain']];rows=[]
        assert sha(Path(run['keys_path']))==run['keys_sha256']
        for src in run['sources']:
            p=Path(src['path']);assert sha(p)==src['prediction_sha256'];values=read(p)
            assert set(truth)=={r['opaque_id'] for r in values}
            correct=sum(r['prediction']==truth[r['opaque_id']] for r in values)
            agreement=run['scores'][src['epoch']-1]
            rows.append(dict(epoch=src['epoch'],correct=correct,total=len(truth),accuracy_percent=100*correct/len(truth),matches=agreement['matches'],agreement_percent=agreement['agreement_percent']))
        selected=rows[run['selected_epoch']-1];final=rows[-1];best=max(r['correct'] for r in rows)
        result={k:run[k] for k in ('run_id','domain','model','family','dataset','selected_epoch','tied_epochs','keyed_positions','predicted_positions','key_coverage_percent','images','images_with_keys')}
        result.update(selected_correct=selected['correct'],selected_accuracy_percent=selected['accuracy_percent'],final_correct=final['correct'],final_accuracy_percent=final['accuracy_percent'],difference_pp=selected['accuracy_percent']-final['accuracy_percent'],difference_correct=selected['correct']-final['correct'],best_epochs=[r['epoch'] for r in rows if r['correct']==best],best_correct=best,best_accuracy_percent=100*best/len(truth),all20_mean_accuracy_percent=sum(r['accuracy_percent'] for r in rows)/20,rows=rows)
        results.append(result);print('SCORED',run['run_id'],'epoch',selected['epoch'],'correct',selected['correct'],'/',len(truth),'final',final['correct'],'delta',round(result['difference_pp'],4),flush=True)
    counts=dict(improved=sum(r['difference_correct']>0 for r in results),unchanged=sum(r['difference_correct']==0 for r in results),worse=sum(r['difference_correct']<0 for r in results))
    write(OUT/'scored_results.json',dict(at=now(),runs=results,outcomes=counts,selection_sha256=freeze['selection_sha256'],case_sensitive_exact_accuracy=True,all_choices_frozen_before_any_labels=True))
    lines=['# Frozen20-epoch character committee across CryptBB and DarkBay','','Same11/20 rule as the initial DarkBay Model3-A test. Each run is its own committee. All20 epochs eligible, no loss gate; case-insensitive ASCII letters, exact digits; skip disputed/missing-majority positions; earliest tie. Real evaluation remains case-sensitive exact-string. All choices frozen together before labels were read. Repeatedly inspected historical tests: retrospective diagnostic, no official selection or training changes.','',f"Compared with final20: {counts['improved']} improved, {counts['unchanged']} unchanged, {counts['worse']} worse. Runs are correlated; this tally is descriptive, not an independent-trial significance test.",'','| Domain / family | Model / arm | Selected epoch | Selected accuracy | Final20 accuracy | Difference pp | Key coverage |','|---|---|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['domain']} / {r['family']} | M{r['model']}-{r['dataset']} | {r['selected_epoch']} | {r['selected_correct']}/{r['images']} ({r['selected_accuracy_percent']:.2f}%) | {r['final_correct']}/{r['images']} ({r['final_accuracy_percent']:.2f}%) | {r['difference_pp']:+.2f} | {r['key_coverage_percent']:.2f}% |")
    lines+=['','## Missing20-epoch prediction sets','']
    for r in read(OUT/'inventory.json')['incomplete']:lines.append(f"- {r['run_id']}: missing epochs {r['missing_epochs']}; no19-epoch/partial committee substituted.")
    lines+=['','## Limits','','Letter-case mistakes do not distinguish epochs in selection, but do count against real exact accuracy. Shared errors, correlated checkpoints, self-votes, positional shifts and skipped tails can all bias this selector. No target-derived lengths or scoring-label thresholds. Earlier archived experiments outside the canonical A/B series are inventoried separately; no extra training or inference was started. Full per-epoch scores, ties, input hashes and coverage are in the JSON artifacts.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    assert sha(OUT/'selection.json')==freeze['selection_sha256']
    write(OUT/'verification.json',dict(passed=True,at=now(),runs=len(results),outcomes=counts,selection_sha256=freeze['selection_sha256'],scored_results_sha256=sha(OUT/'scored_results.json'),report_sha256=sha(OUT/'REPORT.md'),selection_unchanged_after_scoring=True,all_prediction_hashes_rechecked=True))
if __name__=='__main__':
    if sys.argv[1]=='select':select()
    elif sys.argv[1]=='score':score()
    else:raise ValueError('select or score')
