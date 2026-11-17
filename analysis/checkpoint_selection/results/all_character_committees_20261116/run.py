"""Complete inventory adapters; unchanged frozen committee algorithm."""
import importlib.util,sys
from pathlib import Path
BASE=Path('E:/pranav/CAPRI/runs/all_character_committees_20261116/analyze_all.py')
spec=importlib.util.spec_from_file_location('committee_batch',BASE);G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
G.OUT=Path(__file__).parent
original_inventory=G.inventory
original_write=G.write

def valid_rows(path):
    rows=G.read(path);ids=sorted(r['opaque_id'] for r in G.read(G.MANIFESTS['DarkBay']))
    assert len(rows)==900 and sorted(r['opaque_id'] for r in rows)==ids
    assert all({'opaque_id','prediction'}<=set(r)<= {'opaque_id','prediction','confidence','joint_confidence'} for r in rows)

def inventory():
    complete,incomplete,archives=original_inventory()
    for tag in 'AB':
        rid=f'DarkBay_M3_darkbay_{tag}'
        item=next(r for r in incomplete if r['run_id']==rid)
        assert item['missing_epochs']==[20]
        dest=G.ROOT/f'model3/results/darkbay/{tag}/evaluation_epoch20';p=dest/'predictions.json';f=G.read(dest/'prediction_freeze.json')
        receipt_path=G.ROOT/f'model3/runs/darkbay/{tag}/completion.json';r=G.read(receipt_path)
        protocol=G.ROOT/'handoff/darkbay_curriculum_20261113/protocol.json'
        assert G.sha(p)==f['predictions_sha256'] and f['images']==900
        assert f['checkpoint_sha256']==r['sha256'] and r['epoch']==20 and r['global_step']==15640
        assert G.sha(protocol)==r['protocol_sha256']==f['protocol_sha256']
        valid_rows(p)
        item['sources'].append(dict(run_id=rid,domain='DarkBay',model=3,family='darkbay',dataset=tag,epoch=20,path=str(p),prediction_sha256=G.sha(p),checkpoint_sha256=f['checkpoint_sha256'],protocol_sha256=G.sha(protocol),freeze_sha256=G.sha(dest/'prediction_freeze.json'),manifest_sha256=G.sha(G.MANIFESTS['DarkBay']),manifest_hash_in_original_protocol=False,completion_receipt_sha256=G.sha(receipt_path)))
        item['available_epochs']=list(range(1,21));item['missing_epochs']=[]
        incomplete.remove(item);complete.append(item)
    for model,family in ((2,'Darkbay_ds2_noFont-intel'),(3,'Darkbay_model3_adaptive_20261006')):
        base=G.ROOT/f'model{model}/results/historical'/family
        for seed in (48,123,456):
            rid=f'DarkBay_M{model}_historical_{family}_seed{seed}';sources={}
            for dest in sorted((base/'interim_evaluation').glob(f'seed_{seed}_epoch_*')):
                if not (dest/'prediction_freeze.json').exists():continue
                p=dest/'predictions.json';protocol=G.read(dest/'protocol.json');f=G.read(dest/'prediction_freeze.json')
                assert G.sha(p)==f['sha256'] and protocol['seed']==seed and protocol['images']==900
                valid_rows(p);e=protocol['epoch']
                sources[e]=dict(run_id=rid,domain='DarkBay',model=model,family='historical_'+family,dataset=f'seed{seed}',epoch=e,path=str(p),prediction_sha256=G.sha(p),checkpoint_sha256=protocol['sha256'],protocol_sha256=G.sha(dest/'protocol.json'),freeze_sha256=G.sha(dest/'prediction_freeze.json'),manifest_sha256=G.sha(G.MANIFESTS['DarkBay']),manifest_hash_in_original_protocol=False,confidence_ignored=True)
            final=base/'evaluation_darkbay'
            if (final/'protocol.json').exists():
                protocol=G.read(final/'protocol.json');f=G.read(final/'prediction_freeze.json')
                r=next(x for x in protocol['checkpoints'] if x['seed']==seed);e=r['epoch']
                if e not in sources:
                    p=final/f'seed_{seed}_predictions.json';assert G.sha(p)==f[f'seed_{seed}'];valid_rows(p)
                    sources[e]=dict(run_id=rid,domain='DarkBay',model=model,family='historical_'+family,dataset=f'seed{seed}',epoch=e,path=str(p),prediction_sha256=G.sha(p),checkpoint_sha256=r['sha256'],protocol_sha256=G.sha(final/'protocol.json'),freeze_sha256=G.sha(final/'prediction_freeze.json'),manifest_sha256=G.sha(G.MANIFESTS['DarkBay']),manifest_hash_in_original_protocol=False,confidence_ignored=True)
            item=dict(run_id=rid,domain='DarkBay',model=model,family='historical_'+family,dataset=f'seed{seed}',available_epochs=sorted(sources),missing_epochs=[e for e in range(1,21) if e not in sources],sources=[sources[e] for e in sorted(sources)],duplicate_resolution=[])
            (incomplete if item['missing_epochs'] else complete).append(item)
    for r in archives:r['reason']='Archived seed runs inspected where prediction protocols exist; complete20 sets included separately. Model1 archive has only final/isolated checkpoints.'
    return sorted(complete,key=lambda r:r['run_id']),sorted(incomplete,key=lambda r:r['run_id']),archives

def write(path,value):
    if path==G.OUT/'protocol.json':
        value['scope']='All available canonical CryptBB/DarkBay study runs plus archived DarkBay seed runs; exact20 prediction sets only; no partial committee substitution'
        value['inventory_adapter_sha256']=G.sha(Path(__file__))
        value['legacy_adapters']='Legacy M3 final20 uses prediction freeze linked to completion and frozen training protocol. Archived seed runs use per-epoch freeze+protocol, ignore confidence, and validate all900 opaque IDs.'
    original_write(path,value)
G.inventory=inventory;G.write=write
if __name__=='__main__':
    if sys.argv[1]=='select':G.select()
    elif sys.argv[1]=='score':
        assert G.read(G.OUT/'protocol.json')['inventory_adapter_sha256']==G.sha(Path(__file__))
        G.score()
    else:raise ValueError('select or score')
