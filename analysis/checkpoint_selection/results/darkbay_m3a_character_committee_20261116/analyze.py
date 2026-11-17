"""Twenty-checkpoint character majority selection; labels only in separate score stage."""
import collections,datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path('E:/pranav/Thesis-Pranav-Final')
FAMILY='darkbay_v3_feature_20261115'
OUT=Path(__file__).parent
PRED=ROOT/'model3/results'/FAMILY/'A'
RUN=ROOT/'model3/runs'/FAMILY/'A'
STUDY=ROOT/'handoff'/FAMILY
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def norm(c):return c.lower() if c is not None and (('A'<=c<='Z') or ('a'<=c<='z')) else c
def character(c):return c is not None and (('a'<=c<='z') or ('0'<=c<='9'))
def key(votes):
    counts=collections.Counter(norm(c) for c in votes)
    candidates=[(c,n) for c,n in counts.items() if character(c) and n>=11]
    assert len(candidates)<=1
    return candidates[0] if candidates else None
def choose(scores):
    best=max(scores.values());ties=sorted(e for e,s in scores.items() if s==best)
    return ties[0],ties
def tests():
    assert key(['A']*6+['a']*5+['b']*9)==('a',11)
    assert key(['a']*10+['b']*10) is None
    assert key(['a']*10+['b']*5+['c']*5) is None
    assert key(['1']*11+['I']*9)==('1',11)
    assert key([None]*11+['a']*9) is None
    assert key(['a']*11+[None]*9)==('a',11)
    assert key(['!']*20) is None
    assert choose({3:5,2:5,1:0})==(2,[2,3])
    # Unanimous retained positions necessarily give identical scores.
    assert len({sum(norm(c)==k for c,k in zip(['A','2'],['a','2'])) for _ in range(20)})==1

def select():
    def no_scores(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(args[0]).replace('\\','/').lower()
            if any(x in p for x in ('scoring_labels','per_image_results','summary.json','results.json','history.json')):
                raise RuntimeError('Scoring information forbidden during selection: '+p)
    sys.addaudithook(no_scores)
    assert not (OUT/'selection.json').exists(),'Preserve existing selection'
    tests()
    protocol=dict(version=1,at=now(),family=FAMILY,model=3,dataset='A',images=900,epochs=list(range(1,21)),committee='All20 checkpoints of this single run, including the candidate itself',threshold_votes=11,normalization='ASCII letters case-insensitive; digits exact; no letter-digit equivalence',positions='Left-aligned zero-based positions; width=max predicted length per image; no label-derived lengths or edit alignment',missing='Missing/non-alphanumeric characters cannot form keys; count among20 voters; missing candidate is a mismatch at a keyed position',disputed='Skip positions without an alphanumeric character supported by at least11/20',score='Total matching keyed positions divided by total keyed positions; same denominator for every epoch; no penalty at skipped positions',tie='Earliest epoch',empty_key='No selection; error',loss_filter=None,primary_accuracy='Case-sensitive exact whole-string accuracy, read only after selection freeze',interpretation='Retrospective diagnostic on repeatedly inspected historical test; consensus is not correctness; fixed-final20 official choice unchanged',script_sha256=sha(Path(__file__)))
    write(OUT/'protocol.json',protocol)
    manifest_path=ROOT/'evaluation_data/darkbay/split_manifests/image_manifest.json'
    manifest=read(manifest_path);ids=sorted(r['opaque_id'] for r in manifest)
    assert len(ids)==len(set(ids))==900
    assert sha(manifest_path)==read(STUDY/'reference_split.json')['evaluation_manifest_sha256']
    assert not {r['pixel_sha256'] for r in manifest}&{r['pixel_sha256'] for r in read(STUDY/'targets.json')['fit']}
    predictions={};provenance=[]
    for epoch in range(1,21):
        dest=PRED/f'epoch_{epoch:02d}';path=dest/'predictions.json';freeze=read(dest/'prediction_freeze.json')
        receipt=read(RUN/f'checkpoints/epoch_{epoch:02d}.json');ep=read(dest/'protocol.json')
        assert sha(path)==freeze['sha256']
        assert freeze['checkpoint_sha256']==receipt['sha256']==ep['checkpoint_sha256']
        assert receipt['epoch']==epoch and receipt['global_step']==782*epoch
        assert receipt['protocol_sha256']==sha(STUDY/'model3/A/protocol.json')==ep['training_protocol_sha256']
        assert ep['manifest_sha256']==sha(manifest_path) and ep['labels_after_freeze']
        rows=read(path);mapping={r['opaque_id']:r['prediction'] for r in rows}
        assert len(rows)==900 and sorted(mapping)==ids and all(isinstance(x,str) for x in mapping.values())
        predictions[epoch]=mapping
        provenance.append(dict(epoch=epoch,path=str(path),prediction_sha256=sha(path),checkpoint_sha256=receipt['sha256']))
    keys=[];width_total=0;skipped=collections.Counter();vote_histogram=collections.Counter()
    counts={e:0 for e in predictions};per_image=[]
    for opaque in ids:
        strings=[predictions[e][opaque] for e in range(1,21)]
        width=max(map(len,strings));width_total+=width;image_keys=[]
        for position in range(width):
            votes=[s[position] if position<len(s) else None for s in strings]
            result=key(votes)
            if result is None:
                c=collections.Counter(norm(x) for x in votes)
                skipped['majority_missing_or_non_alphanumeric' if any(n>=11 and not character(k) for k,n in c.items()) else 'no_strict_character_majority']+=1
                continue
            char,support=result;vote_histogram[support]+=1
            image_keys.append(dict(position=position,character=char,votes=support))
            for epoch,s in zip(range(1,21),strings):
                if position<len(s) and norm(s[position])==char:counts[epoch]+=1
        keys.append(dict(opaque_id=opaque,keys=image_keys,predicted_width=width))
        per_image.append(len(image_keys))
    n=sum(per_image)
    assert n>0,'No usable character keys; cannot select an epoch'
    selected,ties=choose(counts)
    # Independently recount all keyed comparisons against the saved key representation.
    for epoch in range(1,21):
        independent=sum(k['position']<len(predictions[epoch][row['opaque_id']]) and norm(predictions[epoch][row['opaque_id']][k['position']])==k['character'] for row in keys for k in row['keys'])
        assert independent==counts[epoch]
    write(OUT/'keys.json',keys)
    decision=dict(at=now(),selected_epoch=selected,tied_epochs=ties,protocol_sha256=sha(OUT/'protocol.json'),keys_sha256=sha(OUT/'keys.json'),input_predictions=provenance,coverage=dict(keyed_positions=n,predicted_positions=width_total,percent=100*n/width_total,images_with_keys=sum(x>0 for x in per_image),images_without_keys=sum(x==0 for x in per_image),skipped=dict(skipped),support_histogram=dict(sorted(vote_histogram.items()))),scores=[dict(epoch=e,matches=counts[e],keyed_positions=n,agreement_percent=100*counts[e]/n) for e in range(1,21)],synthetic_fixture_tests_passed=True,independent_score_recount=True,labels_access_denied=True)
    write(OUT/'selection.json',decision)
    write(OUT/'selection_freeze.json',dict(at=now(),selection_sha256=sha(OUT/'selection.json'),protocol_sha256=sha(OUT/'protocol.json'),keys_sha256=sha(OUT/'keys.json')))
    print(json.dumps(dict(selected_epoch=selected,tied_epochs=ties,coverage=decision['coverage'],top_scores=sorted(decision['scores'],key=lambda r:(-r['matches'],r['epoch']))[:5]),indent=2))

def score():
    freeze=read(OUT/'selection_freeze.json');s=read(OUT/'selection.json')
    for name in ('selection','protocol','keys'):assert sha(OUT/(name+'.json'))==freeze[name+'_sha256']
    assert read(OUT/'protocol.json')['script_sha256']==sha(Path(__file__))
    predictions={}
    for r in s['input_predictions']:
        path=Path(r['path']);assert sha(path)==r['prediction_sha256']
        predictions[r['epoch']]=read(path)
    selected_path=RUN/f"checkpoints/epoch_{s['selected_epoch']:02d}.pth"
    assert sha(selected_path)==s['input_predictions'][s['selected_epoch']-1]['checkpoint_sha256']
    labels=read(ROOT/'evaluation_data/darkbay/scoring_labels/scoring_labels.json')
    rows=[]
    for agreement in s['scores']:
        epoch=agreement['epoch'];pred=predictions[epoch]
        assert set(labels)=={r['opaque_id'] for r in pred}
        correct=sum(r['prediction']==labels[r['opaque_id']] for r in pred)
        rows.append(dict(agreement,correct=correct,total=900,accuracy_percent=100*correct/900))
    selected=rows[s['selected_epoch']-1];last=rows[-1]
    best=max(r['correct'] for r in rows);best_epochs=[r['epoch'] for r in rows if r['correct']==best]
    result=dict(at=now(),selected=selected,final_epoch=last,retrospective_best_epochs=best_epochs,retrospective_best_correct=best,retrospective_best_accuracy_percent=100*best/900,all20_mean_accuracy_percent=sum(r['accuracy_percent'] for r in rows)/20,selection_sha256=freeze['selection_sha256'],labels_read_after_selection_freeze=True,selected_checkpoint_rehashed=True,rows=rows)
    write(OUT/'scored_results.json',result)
    lines=['# DarkBay Model3-A twenty-epoch character committee','',f"Strict11/20 majority, case-insensitive ASCII letters, exact digits, disputed/missing-majority positions skipped. All20 epochs eligible; self-votes included; earliest tie. The900 historical DarkBay images are used, not CryptBB251. No loss gate. Choice frozen before a separate scoring process. Official fixed-final20 selection unchanged.",'',f"Selected epoch {selected['epoch']}: {selected['correct']}/900 ({selected['accuracy_percent']:.2f}%). Final20: {last['correct']}/900 ({last['accuracy_percent']:.2f}%). Retrospective best: epochs {best_epochs}, {best}/900 ({100*best/900:.2f}%).",'',f"Keys: {s['coverage']['keyed_positions']}/{s['coverage']['predicted_positions']} predicted positions ({s['coverage']['percent']:.2f}%), covering {s['coverage']['images_with_keys']}/900 images. Agreement score is not accuracy. Skipped positions receive no penalty; systematic shared mistakes and correlated epochs remain limitations. Lengths use predictions only; positional alignment does not correct insertion/deletion shifts. Case-insensitive selection is evaluated with the original case-sensitive exact-string metric.",'','| Epoch | Key matches | Agreement % | Correct/900 | Real accuracy % |','|---:|---:|---:|---:|---:|']
    lines.extend(f"| {r['epoch']} | {r['matches']} | {r['agreement_percent']:.4f} | {r['correct']} | {r['accuracy_percent']:.2f} |" for r in rows)
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    assert sha(OUT/'selection.json')==freeze['selection_sha256']
    write(OUT/'verification.json',dict(passed=True,at=now(),epochs=20,images=900,selection_unchanged_after_scoring=True,scored_results_sha256=sha(OUT/'scored_results.json'),report_sha256=sha(OUT/'REPORT.md'),selection_sha256=freeze['selection_sha256']))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':
    if sys.argv[1]=='select':select()
    elif sys.argv[1]=='score':score()
    else:raise ValueError('Use select or score')
