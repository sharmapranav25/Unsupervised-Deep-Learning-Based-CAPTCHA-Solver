"""Two frozen corpora, one standard PARSeq reader per corpus, cross-curriculum handoff."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from ..decoder.common import ROOT, StageLock, read, write, sha, now
from . import capri2_parseq as F

REPRODUCIBILITY_SEED = 48  # One run only; no repetition sweep.
B_RENDER_SEED = 6000000000


def event(out, stage, **fields):
    record = dict(updated_at=now(), stage=stage, pid=os.getpid(), **fields)
    write(out/'status.json', record)
    with (out/'events.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)


def check_stop(out):
    if (out/'STOP').exists():
        raise RuntimeError('Study STOP marker found; no next stage started')


def call(out, label, command, target=None, env=None):
    check_stop(out)
    if target is not None:
        F.verify(target)
        F.status(target, label, study=str(out), repetitions=1,
                 training_started=label.startswith('train_'))
    event(out, label, state='running')
    log=out/(label+'.console.log')
    with log.open('a', encoding='utf8') as stream:
        proc=subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        write(out/'active_child.json', dict(pid=proc.pid, command=command,
              started_at=now(), stage=label, log=str(log)))
        code=proc.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}; see {log}')


def stage(out, target, name, label):
    cmd=[sys.executable, '-B', '-u', str(ROOT/'capri.py'), 'train-capri2-parseq',
         '--out', str(target), '--stage', name]
    if name=='train-seed':
        cmd+=['--seed', str(REPRODUCIBILITY_SEED)]
        if (target/f'seed_{REPRODUCIBILITY_SEED}/latest.pth').exists():cmd+=['--resume']
    call(out, label, cmd, target=target)


def amend(target, dataset, study):
    path=target/'protocol.json';p=read(path)
    if p.get('ab_study')==str(study):
        assert p['seeds']==[REPRODUCIBILITY_SEED]
        return
    assert not any(target.glob('seed_*/latest.pth')), 'Revision must precede training'
    shutil.copyfile(path,target/'protocol.before_single_run_ab.json')
    p.update(ab_study=str(study),dataset=dataset,seeds=[REPRODUCIBILITY_SEED],
             reader_runs=1,epochs=20,extra_repetitions_cancelled=[123,456],
             evaluation='One final epoch20 reader on the same900 images; freeze predictions before scoring',
             user_revision='Two datasets A/B, cross-dataset teachers, one Model1 run per dataset')
    p['generation']['seed_base']=2000000000 if dataset=='A' else B_RENDER_SEED
    write(path,p)
    guide=target/'README.md'
    old=guide.read_text(encoding='utf8')
    guide.write_text('# Current plan: Dataset '+dataset+', one reader run\n\n'
        'User cancelled repeated training runs. Only the single seed_48 configuration is executed; '
        '48 is a recorded reproducibility setting, not a multi-seed experiment. Both A and B must '
        'pass audits before either reader trains. Use the A/B coordinator, not the historical '
        'three-run launcher. Study: '+str(study)+'\n\n'
        'The earlier setup guide below is historical and superseded where it specifies three runs.\n\n'+old,
        encoding='utf8')


def prepare_b(out, a, b):
    if not b.exists():
        call(out,'setup_B',[sys.executable,'-B','-u',str(ROOT/'capri.py'),
             'train-capri2-parseq','--out',str(b),'--stage','setup'])
    amend(b,'B',out)
    source=a/'generator';dest=b/'generator'
    assert read(source/'status.json').get('state')=='complete'
    names=['entries.json','inputs.json','length_decision.json','model_manifest.json',
           'protocol.json','adapter_verification.json','controller_calibration.json',
           'calibration_features.npz','refined_recipes.json','refined_predictions.json',
           'raw_hybrid_predictions.json','controller_history.json']
    if not dest.exists():dest.mkdir()
    copied=[]
    for name in names:
        original=source/name;target=dest/name
        if target.exists():assert sha(target)==sha(original), name
        else:shutil.copyfile(original,target)
        copied.append(dict(source=str(original),destination=str(target),sha256=sha(target)))
    assert len(read(dest/'entries.json'))==300
    assert {r['pixel_sha256'] for r in read(dest/'inputs.json')}=={
        r['pixel_sha256'] for r in read(b/'targets.json')['fit']}
    write(dest/'controller_reuse.json',dict(source=str(source),files=copied,
          fitting_repeated=False,render_seed_base=B_RENDER_SEED))
    write(dest/'status.json',dict(stage='controllers',state='complete',processed=100,total=100,
          source=str(source),reused=True))


def wait_for_original_stop(out,a):
    import psutil
    event(out,'waiting_for_A_generation_and_original_launcher_stop',state='running')
    while (a/'pipeline.lock.json').exists():
        check_stop(out)
        lock=read(a/'pipeline.lock.json')
        try:
            proc=psutil.Process(lock['pid'])
            if abs(proc.create_time()-lock['process_created'])>=1:break
        except psutil.NoSuchProcess:break
        time.sleep(15)
    assert read(a/'generator/corpus_audit.json')['images']==100000
    marker=a/'STOP'
    if marker.exists():
        text=marker.read_text(encoding='utf-8-sig')
        assert 'Coordinator: '+out.name in text, 'Unrelated user STOP must not be removed'
        marker.rename(a/'STOP.original_three_run_plan_cancelled.txt')


def manifest(target):
    source=target/'generator';audit=read(source/'corpus_audit.json')
    assert audit['images']==100000 and audit['target_pixel_overlap']==0
    rows=sorted(read(source/'corpus_summary.json'),key=lambda r:r['entry_id'])
    write(source/'dataset_manifest.json',dict(count=100000,controllers=audit['controllers'],
          entries_sha256=sha(source/'entries.json'),
          shards=[dict(**r,path='corpus/'+r['entry_id']+'.zip') for r in rows]))


def audit_pair(out,a,b):
    # Each individual pack stage decodes every PNG and checks its recorded pixel hash.
    def pixels(target):
        receipt=read(target/'data_receipt.json')
        assert sha(target/'sample_provenance.jsonl')==receipt['provenance_sha256']
        with (target/'sample_provenance.jsonl').open(encoding='utf8') as stream:
            hashes={json.loads(line)['pixel_sha256'] for line in stream}
        assert len(hashes)==100000
        return hashes,receipt
    ah,ar=pixels(a);bh,br=pixels(b)
    assert not ah&bh,'A/B pixel overlap; stop before training'
    assert sha(a/'generator/entries.json')==sha(b/'generator/entries.json')
    assert ar['controllers']==br['controllers']==dict(custom=33334,dino=33333,mixed=33333)
    for target,base in ((a,2000000000),(b,B_RENDER_SEED)):
        assert all(r.get('seed_base',2000000000)==base for r in read(target/'generator/corpus_summary.json'))
    assert 2000000000+300*10000000 <= B_RENDER_SEED
    write(out/'datasets_ready.json',dict(created_at=now(),A=str(a),B=str(b),images_each=100000,
        total_unique_pixels=200000,cross_dataset_pixel_overlap=0,identical_controller_recipes=True,
        disjoint_render_seed_intervals=True,A_lmdb_sha256=ar['lmdb_sha256'],B_lmdb_sha256=br['lmdb_sha256'],
        target_pixel_overlap_each=0,controllers_each=ar['controllers'],
        note='Independent rendered pixels, shared target-conditioned recipes; not independent CAPTCHA families'))


def train_one(out,target,label):
    assert read(out/'datasets_ready.json')['total_unique_pixels']==200000
    stage(out,target,'preflight','preflight_'+label)
    done=target/f'seed_{REPRODUCIBILITY_SEED}/completion.json'
    if not done.exists():stage(out,target,'train-seed','train_Model1_'+label)
    record=read(done)
    assert record['epoch']==20 and record['global_step']==15640 and sha(Path(record['checkpoint']))==record['sha256']
    write(target/'completion.json',dict(seeds=[record],epochs=20,all_complete=True,
          requested_reader_runs=1,study=str(out),dataset=label))
    return record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--a',required=True);ap.add_argument('--b',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();a=Path(args.a).resolve();b=Path(args.b).resolve();out=Path(args.out).resolve()
    assert len({a,b,out})==3
    for p in (a,b,out):assert p.drive.upper()=='E:' and p.is_relative_to(ROOT/'runs')
    if out.exists() and not args.resume:raise FileExistsError('Explicit --resume required')
    F.configure(out)
    protocol=out/'protocol.json'
    if not protocol.exists():
        write(protocol,dict(created_at=now(),A=str(a),B=str(b),images_each=100000,reader_runs_each=1,
            epochs=20,batch_size=128,reproducibility_seed=REPRODUCIBILITY_SEED,
            directions={'A_to_B':'Model1-A ranks B for Model2-B and initial Model3-B ranking',
                        'B_to_A':'Model1-B ranks A for Model2-A and initial Model3-A ranking'},
            adaptive='Model3 subsequently rescores its own training pool as it learns',
            curriculum_training_launched=False,curriculum_validation_protocol='Pending before Model2/3 launch',
            model1_selection='Fixed epoch20, no validation or real-test selection',
            baseline_comparison='Dataset A single run matched to historical CAPRI1 seed48; report B separately, never pick the better A/B result',
            source_sha256=sha(Path(__file__))))
        shutil.copyfile(__file__,out/'ab_study.source.py')
        (out/'README.md').write_text('# Dataset A/B study\n\nTwo100k corpora from identical frozen controllers, '
            'independent rendering seeds, one20epoch PARSeq run per corpus. All data must audit before either '
            'reader trains. No repeated training runs. The folder seed_48 records reproducibility only.\n\n'
            'A: '+str(a)+'\n\nB: '+str(b)+'\n\n'
            'Model1-A will rank B and Model1-B will rank A. Model2 uses a frozen ranking; Model3 starts '
            'from that ranking and adapts. Curriculum validation/selection rules must be fixed before '
            'those later experiments; this coordinator launches only the two Model1 readers.\n\n'
            'Read status.json and active_child.json, then dataset corpus_status.json or seed_48/train.log. '
            'datasets_ready.json is the200k-pixel audit. completion.json requires both readers and evaluation. '
            'Original artifacts and pre-revision protocol copies remain preserved. No generator deployment.\n\n'
            'Resume with tools/run_ab_study.py and the same --a --b --out arguments plus --resume. '
            'Do not resume the historical three-run launcher. A study STOP file stops between stages.\n',encoding='utf8')
    p=read(protocol);assert p['A']==str(a) and p['B']==str(b) and p['source_sha256']==sha(Path(__file__))
    with StageLock(out,'pipeline'):
        try:
            prepare_b(out,a,b)
            if not (b/'generator/corpus_audit.json').exists():
                env=os.environ.copy();env['CAPRI_CORPUS_SEED_BASE']=str(B_RENDER_SEED)
                call(out,'generate_B',[sys.executable,'-B','-u',str(ROOT/'capri.py'),
                     'generate','--run',str(b/'generator'),'--workers','4'],target=b,env=env)
            wait_for_original_stop(out,a)
            with StageLock(a,'pipeline'),StageLock(b,'pipeline'):
                amend(a,'A',out)
                for label,target in (('A',a),('B',b)):
                    manifest(target);stage(out,target,'pack','pack_'+label)
                audit_pair(out,a,b)
                records={label:train_one(out,target,label) for label,target in (('A',a),('B',b))}
                for label,target in (('A',a),('B',b)):
                    stage(out,target,'evaluate','evaluate_'+label)
                    write(target/'experiment_completion.json',dict(completed_at=now(),reader_training_complete=True,
                        real_evaluation_complete=True,requested_reader_runs=1,dataset=label,study=str(out)))
                    F.status(target,'complete',state='complete',reader_runs=1,training_started=True,study=str(out))
                write(out/'completion.json',dict(completed_at=now(),readers=records,reader_runs_total=2,
                    datasets_ready=True,curriculum_training_launched=False,
                    evaluation={label:read(target/'evaluation_darkbay/summary.json') for label,target in (('A',a),('B',b))}))
                event(out,'complete',state='complete',reader_runs_total=2)
        except Exception as exc:
            event(out,'failed',state='failed',error=repr(exc));raise


if __name__=='__main__':main()
