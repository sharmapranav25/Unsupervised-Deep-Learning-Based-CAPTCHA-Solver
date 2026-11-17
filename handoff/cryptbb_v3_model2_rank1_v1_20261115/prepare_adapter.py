"""Stage a new version; do not edit the canonical or historical adapters here."""
from pathlib import Path
import ast,hashlib
ROOT=Path('E:/pranav/Thesis-Pranav-Final')
OUT=Path(__file__).parent
old=ROOT/'Capri/src/capri_controller/training/cryptbb_v3_model2.py'
s=old.read_text(encoding='utf8')
def replace(a,b):
    global s
    assert a in s,a
    s=s.replace(a,b)
replace('import argparse,copy,gc,io,json,os,shutil,sys,subprocess','import argparse,copy,gc,io,json,os,shutil,sys,subprocess,time,traceback')
replace("STUDY=M1.STUDY/'model2'", "STUDY=ROOT/'handoff/cryptbb_v3_model2_rank1_v1_20261115'\nSELECTION_SHA='"+hashlib.sha256((OUT/'teacher_selection.json').read_bytes()).hexdigest()+"'")
replace('model2/rankings/cryptbb_v3/', 'model2/rankings/cryptbb_v3_rank1_v1/')
replace('model2/runs/cryptbb_v3/', 'model2/runs/cryptbb_v3_rank1_v1/')
replace('def teacher(tag):return ROOT/f\'model1/runs/cryptbb_v3/{"B" if tag=="A" else "A"}/epoch_20.pth\'', '''def selection():
    assert sha(STUDY/'teacher_selection.json')==SELECTION_SHA
    return read(STUDY/'teacher_selection.json')

def teacher_row(tag):return selection()['selected']['B' if tag=='A' else 'A']
def teacher(tag):return ROOT/teacher_row(tag)['checkpoint']''')
replace("assert sha(M1.STUDY/'agreement_protocol.json')==p['selection_protocol_sha256']", "assert sha(STUDY/'teacher_selection.json')==p['selection_protocol_sha256']==SELECTION_SHA")
replace("STUDY.mkdir(exist_ok=True);inputs={}", "STUDY.mkdir(exist_ok=True);inputs={};assert not (STUDY/'protocol.json').exists();selection()")
replace("ck=teacher(tag);r=read(ck.parent/'epoch_20_receipt.json');done=read(ck.parent/'training_completion.json')\n        assert done['epoch']==20 and r['epoch']==20 and sha(ck)==r['sha256']", "ck=teacher(tag);row=teacher_row(tag);r=read(ck.parent/f\"epoch_{row['selected_epoch']}_receipt.json\");done=read(ck.parent/'training_completion.json')\n        assert done['epoch']==20 and r['epoch']==row['selected_epoch'] and sha(ck)==r['sha256']==row['checkpoint_sha256']")
replace("'teacher_sha256':r['sha256']}", "'teacher_sha256':r['sha256'],'teacher_epoch':row['selected_epoch']}")
replace("sources=[]\n    for p in files:", "files += [ROOT/f'model1/configs/cryptbb_v3/{t}/prepared.yml' for t in 'AB']\n    files += [ROOT/'datasets/cryptbb_v3/charset.txt',Path(__file__).parents[1]/'decoder/common.py']\n    sources=[]\n    for p in files:")
replace("teachers={'A':'Model1-B epoch20','B':'Model1-A epoch20'}", "teachers={t:dict(reader='B' if t=='A' else 'A',epoch=teacher_row(t)['selected_epoch'],checkpoint=teacher_row(t)['checkpoint'],sha256=teacher_row(t)['checkpoint_sha256']) for t in 'AB'}")
replace("selection_protocol_sha256=sha(M1.STUDY/'agreement_protocol.json'),reporting_selection='Prospective same-epoch A/B agreement on251 unlabeled evaluation pixels, epochs2-20, earliest tie; labels not used'", "selection_protocol_sha256=SELECTION_SHA,reporting_selection='Independent original v1 masked rank-one exact-string agreement, epochs1-20 on251 unlabeled pixels, k0-5 original choose_k fallback5, highest factor earliest exact tie; freeze before scoring',protocol_revision='Selected cross-teachers v1; supersedes epoch20 teachers only for this separate run'")
replace("digest=check_sources();out=STUDY/'checks';out.mkdir(exist_ok=True);ds=dataset('A')", "digest=check_sources();verify_rankings();out=STUDY/'checks';out.mkdir(exist_ok=True);ds=dataset('A')")
replace("__package__+'.cryptbb_v3_model2'", "__package__+'.cryptbb_v3_model2_selected_v1'")
replace("r.update(selection='Pending frozen A/B agreement reporting rule',reporting_selection_pending=True)", "r.update(selection='Pending independent original v1 masked rank-one reporting rule',reporting_selection_pending=True)")
# Validate both dataset preprocessing and sampler budgets before original replay/pair pilots.
insert='''
def verify_rankings(persist=True):
    import numpy as np
    digest=check_sources();records={}
    for tag in 'AB':
        p=rank_dir(tag);r=read(p/'completion.json');row=teacher_row(tag)
        assert r['protocol_sha256']==digest and r['teacher_sha256']==row['checkpoint_sha256']==sha(teacher(tag))
        assert r['teacher']==('B' if tag=='A' else 'A') and r['images']==100000
        assert sha(p/'ranking.npz')==r['sha256'] and sha(p/'predictions.json')==r['predictions_sha256']
        predictions=read(p/'predictions.json');assert [x['index'] for x in predictions]==list(range(1,100001))
        with np.load(p/'ranking.npz') as z:
            ids=z['ids'];raw=z['nll'];buckets=z['buckets']
            assert np.array_equal(ids,np.arange(1,100001)) and np.isfinite(raw).all() and (raw>=0).all()
            expected=np.empty(100000,dtype=np.int64)
            for b,group in enumerate(np.array_split(np.lexsort((ids,raw)),3)):expected[group]=b
            assert np.array_equal(buckets,expected) and np.bincount(buckets).tolist()==[33334,33333,33333]
        sampler=C.sampler_for(tag);counts=[]
        for epoch in range(1,21):
            draws,receipt=C.draws_for(sampler,2,epoch)
            expected_counts=[100096,0,0] if epoch==1 else [50048,50048,0] if epoch==2 else [15014,25024,60058]
            assert receipt['draws']==expected_counts and len(draws)==100096
            assert np.bincount(sampler.buckets[draws],minlength=3).tolist()==expected_counts
            counts.append(expected_counts)
        records[tag]=dict(ranking_sha256=r['sha256'],teacher_epoch=row['selected_epoch'],sampler_counts=counts)
    if persist:write(STUDY/'ranking_verification.json',dict(passed=True,protocol_sha256=digest,datasets=records,created_at=now()))

def extra_preprocessing_checks():
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from tools.data.ratio_dataset_tvresize import RatioDataSetTVResize
    baseline=RatioDataSetTVResize.__new__(RatioDataSetTVResize)
    baseline.base_shape=[[64,64],[96,48],[112,40],[128,32]];baseline.base_h=32
    baseline.padding_rand=False;baseline.padding_doub=False;baseline.interpolation=T.InterpolationMode.BICUBIC
    baseline.transforms=T.Compose([T.ToTensor(),T.Normalize(.5,.5)]);baseline.error=0
    for tag in 'AB':
        ds=dataset(tag)
        try:
            for i in [0,49999,99999]:
                with ds.env.begin() as tx:im=Image.open(io.BytesIO(tx.get(f'image-{i+1:09d}'.encode()))).convert('RGB')
                assert torch.equal(baseline.resize_norm_img({'image':im},4,padding=False)['image'],ds[i][0])
        finally:ds.env.close()
    write(STUDY/'preprocessing_verification.json',dict(passed=True,exact_matches=6,datasets=['A','B']))

def child(stage,tag=None):
    args=[sys.executable,'-B','-u','-X','faulthandler','-m',__package__+'.cryptbb_v3_model2_selected_v1',stage]
    if tag:args+=['--dataset',tag]
    with (STUDY/(stage+('_'+tag if tag else '')+'.log')).open('a',encoding='utf8') as log:
        p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    import psutil
    write(STUDY/(stage+('_'+tag if tag else '')+'_launch.json'),dict(pid=p.pid,process_created=psutil.Process(p.pid).create_time(),args=args,created_at=now()))
    return p

def coordinate():
    import psutil
    with StageLock(STUDY,'coordinator'):
        try:
            assert not (STUDY/'STOP').exists()
            assert all(not run_dir(2,t).exists() and not rank_dir(t).exists() for t in 'AB'),'Inspect existing artifacts; no implicit restart'
            for stage,tag in [('setup',None),('rank','A'),('rank','B'),('readiness',None)]:
                write(STUDY/'status.json',dict(state=stage,dataset=tag,pid=os.getpid(),updated_at=now()))
                p=child(stage,tag);assert p.wait()==0,(stage,tag)
            verify_rankings();r=read(STUDY/'readiness.json');assert r['passed'] and r['protocol_sha256']==check_sources()
            children={t:child('worker',t) for t in 'AB'}
            write(STUDY/'status.json',dict(state='training',pid=os.getpid(),updated_at=now()))
            first={};verified=False
            while any(p.poll() is None for p in children.values()):
                for t,p in children.items():assert p.poll() in (None,0),('worker failed',t,p.returncode)
                statuses={t:read(run_dir(2,t)/'status.json') for t in 'AB' if (run_dir(2,t)/'status.json').exists()}
                for t,r in statuses.items():
                    if r.get('global_step',0)>0 and t not in first:first[t]=r
                if not verified and len(first)==2 and all(statuses[t].get('global_step',0)>first[t]['global_step'] for t in 'AB'):
                    identities={t:read(STUDY/('worker_'+t+'.lock.json')) for t in 'AB'}
                    for t,identity in identities.items():
                        process=psutil.Process(identity['pid']);assert abs(process.create_time()-identity['process_created'])<1
                        assert 'worker' in process.cmdline() and t in process.cmdline()
                        init=read(run_dir(2,t)/'initialization.json');assert init['pretrained'] is False and init['seed']==48
                    assert read(run_dir(2,'A')/'initialization.json')['model_tensor_sha256']==read(run_dir(2,'B')/'initialization.json')['model_tensor_sha256']
                    write(STUDY/'launch_verified.json',dict(verified_at=now(),protocol_sha256=check_sources(),workers=identities,first=first,progress=statuses,scratch_initialization_verified=True))
                    verified=True
                time.sleep(5)
            for t,p in children.items():
                assert p.wait()==0
                r=read(run_dir(2,t)/'completion.json');assert r['epoch']==20 and sha(run_dir(2,t)/r['checkpoint'])==r['sha256']
            write(STUDY/'status.json',dict(state='complete_model2_pair',updated_at=now(),model3_started=False))
        except Exception:
            write(STUDY/'status.json',dict(state='failed',error=traceback.format_exc(),updated_at=now()))
            if 'children' in locals() and any(p.poll() is None for p in children.values()):
                (STUDY/'STOP').write_text('Coordinator failure; stop at committed epoch boundary. Inspect before resume.\\n')
            raise

'''
replace("if __name__=='__main__':",insert+"\nif __name__=='__main__':")
replace("choices=['setup','rank','readiness','pilot','worker']", "choices=['setup','rank','readiness','pilot','worker','coordinate']")
replace("if a.stage=='setup':setup()", "if a.stage=='coordinate':coordinate()\n    elif a.stage=='setup':setup()")
replace("elif a.stage=='readiness':readiness()", "elif a.stage=='readiness':extra_preprocessing_checks();readiness()")
replace("else:\n        with StageLock", "else:\n        if a.stage=='worker':\n            verify_rankings(persist=False);r=read(STUDY/'readiness.json');assert r['passed'] and r['protocol_sha256']==check_sources()\n        with StageLock")
ast.parse(s)
(OUT/'cryptbb_v3_model2_selected_v1.py').write_text(s,encoding='utf8')
print('Staged new adapter; original preserved. '+str(len(s.splitlines()))+' lines')
