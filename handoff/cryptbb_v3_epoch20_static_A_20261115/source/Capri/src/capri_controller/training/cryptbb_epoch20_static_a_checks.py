"""Fast lifecycle integration checks; temporary mock training never touches production."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from . import cryptbb_epoch20_static_a as M


def lifecycle_checks():
    checks={}
    with TemporaryDirectory(prefix='cumulative_checks_',dir=M.ROOT/'runtime/temporary') as temp:
        root=Path(temp); study=root/'study'; study.mkdir(); digest='mock-protocol'
        model=object(); ds=SimpleNamespace(env=SimpleNamespace(close=lambda:None))
        sampler=SimpleNamespace(ids=np.arange(1,100001))
        expected=np.ones((20,100096),dtype=np.int64)
        (study/'draws').mkdir()
        for tag in 'A': np.save(study/f'draws/{tag}_epochs01_20.npy',expected)
        M.write(study/'draw_manifest.json',dict(datasets={t:dict(sha256=M.sha(study/f'draws/{t}_epochs01_20.npy')) for t in 'A'}))
        M.write(study/'readiness.json',dict(passed=True,protocol_sha256=digest,datasets={t:dict(initial_tensor_sha256='scratch') for t in 'A'}))
        def save(path,*args): path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'immutable')
        def draws(*args): return np.zeros(100096,dtype=np.int64),dict(total=100096)
        def run_dir(n,t): return root/'runs'/t
        with patch.object(M,'STUDY',study),patch.object(M,'run_dir',run_dir),patch.object(M,'check_sources',return_value=digest),patch.object(M,'config',return_value={}),patch.object(M,'dataset',return_value=ds),patch.object(M,'draws_for',side_effect=draws),patch.object(M.C,'sampler_for',return_value=sampler),patch.object(M.C,'make_model',return_value=(model,None)),patch.object(M.C,'optimizer_for',return_value=(None,None,None)),patch.object(M.C,'model_hash',return_value='scratch'),patch.object(M.C,'batch_at',return_value=None),patch.object(M.C,'save_checkpoint',side_effect=save):
            with patch.object(M.C,'step',return_value=(1.,2.)) as step,patch('builtins.print'):
                M.worker('A'); assert step.call_count==15640
            M.valid_completion('A'); assert len(list((run_dir(2,'A')/'checkpoints').glob('*.pth')))==20
            checks['20_epoch_transition_and_completion']=True
            saved_hash=M.sha(run_dir(2,'A')/'checkpoints/epoch_01.pth')
            try: M.worker('A')
            except AssertionError: pass
            else: raise AssertionError('Overwrite permitted')
            assert M.sha(run_dir(2,'A')/'checkpoints/epoch_01.pth')==saved_hash
            checks['immutable_archive_and_existing_run_guard']=True
            # Exercise STOP during either reader using independent output locations.
            for tag in 'A':
                stopped_dir=lambda n,t: root/('stop_'+tag)/t
                calls=[]
                def step_stop(*args):
                    calls.append(1)
                    if len(calls)==2: (study/'STOP').write_text('test')
                    return 1.,2.
                with patch.object(M,'run_dir',stopped_dir),patch.object(M.C,'step',side_effect=step_stop),patch('builtins.print'):
                    try: M.worker(tag)
                    except M.Stopped: pass
                    else: raise AssertionError('STOP ignored')
                assert len(calls)==2 and not (stopped_dir(2,tag)/'completion.json').exists()
                (study/'STOP').unlink()
            checks['STOP_A_and_B_before_next_optimizer_step']=True
            failed=SimpleNamespace(poll=lambda:1,returncode=1)
            try: M.supervise('B',failed)
            except AssertionError: pass
            else: raise AssertionError('Worker failure not propagated')
            checks['worker_failure_propagated']=True
            (study/'STOP').write_text('test')
            try: M.supervise('B',failed)
            except M.Stopped: pass
            else: raise AssertionError('STOP must precede next launch/completion')
            (study/'STOP').unlink()
            checks['STOP_precedes_B_transition']=True
    return checks
