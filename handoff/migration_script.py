"""Copy-first, hash-verified thesis migration. Originals are never changed/deleted."""
import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

SRC = Path('E:/pranav/CAPRI')
DST = Path('E:/pranav/Thesis-Pranav-Final')
SF = Path('E:/pranav/Final sem/sourcefree_synth')
RENDER = Path('E:/pranav/Final sem/captcha-effects')
OCR = Path('E:/pranav/Final sem/Models/level 1/upstream/OpenOCR')
A = SRC/'runs/parseq_darkbay_capri2_100k_20261112'
B = SRC/'runs/parseq_darkbay_dataset_b_100k_20261112'
GEN = SRC/'runs/capri_lightweight_joint20_100k_20261028'
STUDY = SRC/'runs/darkbay_ab_study_20261112'
SKIP = {'.git', '__pycache__', '.pytest_cache', '.venv', 'node_modules', 'temporary', '.cache'}
plan = {}

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def add(s, d):
    assert s.is_file(), s
    assert not s.is_symlink(), s
    d=Path(d)
    assert not d.is_absolute() and '..' not in d.parts
    key=d.as_posix()
    assert key not in plan or plan[key]['source']==str(s), key
    plan[key]={'source':str(s), 'destination':key, 'bytes':s.stat().st_size}
def tree(s, d, excluded=(), only=None):
    assert s.is_dir(), s
    for base, dirs, files in os.walk(s, followlinks=False):
        dirs[:]=sorted(x for x in dirs if x not in SKIP and x not in excluded and not (Path(base)/x).is_symlink())
        for name in sorted(files):
            p=Path(base)/name
            if name in excluded or p.suffix in {'.pyc', '.pyo'} or (only and not only(p)): continue
            add(p,Path(d)/p.relative_to(s))
def tops(s,d,exclude=()):
    for p in sorted(s.iterdir()):
        if p.is_file() and p.name not in exclude: add(p,Path(d)/p.name)

def make_plan():
    tree(SRC/'src','Capri/src')
    tree(SRC/'tools','Capri/tools')
    tree(SRC/'configs','Capri/configs',excluded=('local.json',))
    add(SRC/'configs/local.json','handoff/original_configuration/local.json')
    add(SRC/'models/registry.json','handoff/original_configuration/registry.json')
    add(SRC/'capri.py','Capri/capri.py')
    tree(SRC/'docs','handoff/historical_docs')
    for p in ('README.md','AGENTS.md','CAPRI_WORKFLOW.md'):
        add(SRC/p,'handoff/original_configuration/'+p)
    tree(SRC/'models/active','Capri/models/controller_dependencies')
    tops(GEN,'Capri/runs/generator_training')
    tree(GEN/'provenance','Capri/runs/generator_training/provenance')
    tree(GEN/'epochs','Capri/runs/generator_training/epochs',excluded=('epoch_0020',))
    add(GEN/'epochs/epoch_0020/checkpoint.pt','Capri/models/capri/checkpoint.pt')
    add(GEN/'epochs/epoch_0020/receipt.json','Capri/models/capri/original_receipt.json')
    tree(SRC/'runs/capri_shared_count_dataset_100k_20261027','Capri/training_data/scenes_and_font_references')
    tree(SRC/'runs/capri_lightweight_cache_100k_20261028_v2','Capri/training_data/prepared_cache')
    tree(RENDER,'Capri/renderer',excluded=('calibration.log',))
    for folder in ('sourcefree_synth','adapters','configs','style_pipeline','resources','vendor'):
        tree(SF/folder,'Capri/dependencies/sourcefree_synth/'+folder)
    for p in ('README.md','ARCHITECTURE.md','METHODS.md','requirements.txt','requirements-models.txt'):
        add(SF/p,'Capri/dependencies/sourcefree_synth/'+p)
    tree(OCR,'model1/src/OpenOCR')
    tree(SRC/'.runtime/parseq_packages','model1/src/parseq_packages')
    add(SRC/'runs/capri_decoder_500k_20261007/pretrained/dinov2/dinov2_vits14_reg4_pretrain.pth','Capri/models/pretrained/dinov2_vits14_reg4_pretrain.pth')
    for tag, run in [('A',A),('B',B)]:
        data=f'datasets/darkbay/{tag}'
        tree(run/'train_lmdb',data+'/train_lmdb',excluded=('lock.mdb',))
        tree(run/'generator/corpus',data+'/corpus')
        for p in ('charset.txt','data_receipt.json','sample_provenance.jsonl'):
            add(run/p,data+'/'+p)
        for p in ('corpus_audit.json','corpus_summary.json','dataset_manifest.json','allocation.json'):
            add(run/'generator'/p,data+'/manifests/'+p)
        record=f'model1/runs/darkbay/{tag}'
        tops(run,record)
        tree(run/'provenance',record+'/provenance')
        tree(run/'seed_48',record+'/training',only=lambda p:p.suffix!='.pth')
        for p in sorted((run/'seed_48').glob('epoch_*.pth')):
            if p.name=='epoch_20.pth': add(p,f'model1/models/parseq/darkbay/{tag}/epoch_20.pth')
            else: add(p,record+'/checkpoints/'+p.name)
        for evaluation in sorted(run.glob('evaluation*')):
            if evaluation.is_dir(): tree(evaluation,f'model1/results/darkbay/{tag}/'+evaluation.name)
        add(run/'seed_48.yml',f'model1/configs/darkbay/{tag}/original.yml')
    tree(A/'generator','Capri/controllers/darkbay',excluded=('corpus',))
    # B uses identical controller recipes; preserve its generation receipts without duplicating the controllers.
    tops(B/'generator','datasets/darkbay/B/generation_provenance')
    tree(A/'inputs','Capri/controllers/darkbay/reference_images')
    ev=SRC/'runs/parseq_darkbay_100k_20261005/evaluation_darkbay'
    tree(ev/'images','evaluation_data/darkbay/images')
    add(ev/'scoring_labels.json','evaluation_data/darkbay/scoring_labels/scoring_labels.json')
    for p in ('image_manifest.json','protocol.json','scoring_label_audit.json','source_provenance.json'):
        add(ev/p,'evaluation_data/darkbay/split_manifests/original_'+p)
    tree(STUDY,'handoff/study_records')
    for tag, folder in [('model1','parseq_darkbay_100k_20261005'),('model2','Darkbay_ds2_noFont-intel'),('model3','Darkbay_model3_adaptive_20261006')]:
        source=SRC/'runs'/folder
        dest=tag+'/results/historical/'+folder
        tops(source,dest,exclude=('samples.json','curriculum_samples.json','entries.json'))
        # Historical scores and protocol evidence; older weights/data stay at recorded original paths.
        for sub in ('evaluation_darkbay','interim_evaluation','all_epoch_diagnostics','source_snapshot'):
            if (source/sub).is_dir(): tree(source/sub,dest+'/'+sub,excluded=('images',))
    for folder in ('capri_previous_vs_new_count_20261029_v2','capri_lightweight_internal_count_epoch20_20261029','capri1_vs_capri2_controller_examples_20261029'):
        tree(SRC/'runs'/folder,'Capri/evaluation/'+folder)
    add(Path(__file__),'handoff/migration_script.py')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--copy',action='store_true');args=ap.parse_args()
    assert DST.drive.upper()=='E:' and SRC.drive.upper()=='E:'
    assert read(A/'completion.json')['all_complete'] and read(B/'completion.json')['all_complete']
    assert (STUDY/'STOP').exists()
    make_plan()
    totals={}
    for row in plan.values():
        section=row['destination'].split('/')[0]
        totals[section]=totals.get(section,0)+row['bytes']
    summary=dict(created_at=now(),source=str(SRC),destination=str(DST),files=len(plan),bytes=sum(totals.values()),bytes_by_section=totals,originals_preserved=True)
    write(SRC/'maintenance/thesis_migration_plan_20261112.json',dict(**summary,files_to_copy=list(plan.values())))
    print(json.dumps(summary),flush=True)
    if not args.copy:return
    assert shutil.disk_usage(DST).free>summary['bytes']+10*1024**3
    folders=['Capri/configs','Capri/models','Capri/runs','datasets/darkbay/manifests','evaluation_data/darkbay/scoring_labels','evaluation_data/darkbay/split_manifests','thesis/chapters','thesis/figures','thesis/tables','thesis/bibliography','handoff','runtime/cache','runtime/temporary']
    for model in ('model1','model2','model3'):
        folders += [model+'/'+x for x in ('src','configs','runs','results')]
        for tag in ('A','B'):folders += [model+'/models/parseq/darkbay/'+tag]
    folders += ['model2/rankings/darkbay/A','model2/rankings/darkbay/B']
    for folder in folders:(DST/folder).mkdir(parents=True,exist_ok=True)
    write(DST/'handoff/MIGRATION_PLAN.json',summary)
    receipt=DST/'handoff/COPY_MANIFEST.jsonl'
    completed={}
    if receipt.exists():
        for line in receipt.read_text(encoding='utf8').splitlines():
            r=json.loads(line);completed[r['destination']]=r
    started=time.time();done_bytes=0;last=time.time()
    with receipt.open('a',encoding='utf8') as log:
        for index,(relative,row) in enumerate(plan.items(),1):
            s=Path(row['source']);d=DST/relative
            assert d.is_relative_to(DST)
            if relative in completed:
                old=completed[relative]
                assert d.is_file() and d.stat().st_size==row['bytes'] and sha(d)==old['sha256'], d
                assert s.stat().st_size==row['bytes'] and sha(s)==old['sha256'], s
            else:
                digest=sha(s)
                d.parent.mkdir(parents=True,exist_ok=True)
                if d.exists():
                    assert d.is_file() and sha(d)==digest, 'Refuse overwrite: '+str(d)
                else:shutil.copy2(s,d)
                assert sha(d)==digest, d
                log.write(json.dumps(dict(**row,sha256=digest,verified_at=now()))+'\n');log.flush()
            done_bytes+=row['bytes']
            if time.time()-last>15 or index==len(plan):
                progress=dict(updated_at=now(),stage='copy_and_hash_verify',files_done=index,files_total=len(plan),bytes_done=done_bytes,bytes_total=summary['bytes'],seconds=time.time()-started,current=relative)
                write(DST/'handoff/migration_status.json',progress);print(json.dumps(progress),flush=True);last=time.time()
    write(DST/'handoff/migration_copy_complete.json',dict(**summary,completed_at=now(),all_files_sha256_verified=True,receipt_sha256=sha(receipt)))
    print('COPY VERIFIED',flush=True)

if __name__=='__main__':main()
