"""Independent adapter equivalence and worker lifecycle checks; no optimization."""
import ast
import contextlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path('E:/pranav/Thesis-Pranav-Final')
sys.path.insert(0,str(ROOT/'Capri/src'))
from capri_controller.training import cryptbb_feature_b_v1 as M
SRC=Path(M.__file__).parent
receipt=M.read(M.STUDY/'installation.json')
a=(SRC/'cryptbb_feature_v1.py').read_text()
for old,new in receipt['replacements']: a=a.replace(old,new)
anchor="    paths.update(Path(__file__).parent.glob('cryptbb_feature_v1*.py'))"
a=a.replace(anchor,anchor+"\n    paths.update(Path(__file__).parent.glob('cryptbb_feature_b_v1*.py'))\n    paths.add(STUDY/'verify_policy.py')\n    paths.add(STUDY/'verify_adapter.py')")
a=a.replace("Path(__file__).with_name('ds2_data.py')):","Path(__file__).with_name('ds2_data.py'),ROOT/'model1/configs/cryptbb_v3/B/prepared.yml'):")
assert ast.dump(ast.parse(a))==ast.dump(ast.parse(Path(M.__file__).read_text()))
for row in M.read(ROOT/'handoff/cryptbb_v3_feature_priority_v1_20261115/protocol.json')['sources']:
    assert M.sha(ROOT/row['path'])==row['sha256']
checks=dict(adapter_AST_only_declared_identity_and_provenance_changes=True,A_frozen_sources_unchanged=True)

with TemporaryDirectory(prefix='model3_b_lifecycle_',dir=ROOT/'runtime/temporary') as temp:
    root=Path(temp); study=root/'study'; study.mkdir(); base=root/'base';(base/'draws').mkdir(parents=True)
    np.save(base/'draws/B_epochs01_20.npy',np.ones((2,100096),dtype=np.int64))
    M.write(study/'readiness.json',dict(passed=True,protocol_sha256='mock',initial_tensor_sha256='scratch'))
    class Sampler:
        ids=np.ones(100000,dtype=np.int64)
        def draw(self):return np.zeros(100096,dtype=np.int64),dict(mass=[1,0,0])
    ds=SimpleNamespace(env=SimpleNamespace(close=lambda:None))
    def save(path,*args):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'mock checkpoint')
    def assess(*args):
        epoch=args[-1];dest=M.OUT/f'assessments/epoch_{epoch:02d}/feedback.json'
        M.write(dest,dict(seconds=0,overall_accuracy=.5))
        return M.read(dest)
    with contextlib.ExitStack() as stack:
        for obj,name,value in [(M,'STUDY',study),(M,'BASE',base),(M,'OUT',root/'run'),
            (M,'check_sources',lambda:'mock'),(M,'config',lambda out:{}),(M,'dataset',lambda:ds),
            (M,'load_features',lambda:(Sampler(),None)),(M,'gpu_owner',contextlib.nullcontext),
            (M,'OrderedBatches',lambda *args:range(782)),(M,'assess',assess),
            (M.C,'bootstrap',lambda:None),(M.C,'make_model',lambda cfg:(None,None)),
            (M.C,'optimizer_for',lambda *args:(None,None,None)),(M.C,'model_hash',lambda model:'scratch'),
            (M.C,'save_checkpoint',save)]:stack.enter_context(patch.object(obj,name,value))
        with patch.object(M.C,'step',return_value=(1.,2.)) as step,patch('builtins.print'):
            M.worker();assert step.call_count==15640
        done=M.read(M.OUT/'completion.json');assert done['epoch']==20 and done['global_step']==15640
        assert len(list((M.OUT/'checkpoints').glob('*.pth')))==20
        checks['20_epochs_15640_updates_20_committed_archives']=True
        try:M.worker()
        except AssertionError:pass
        else:raise AssertionError('Existing output accepted')
        checks['existing_run_refused']=True
        calls=[]
        def stop_step(*args):
            calls.append(1)
            if len(calls)==2:(study/'STOP').write_text('Disposable STOP test')
            return 1.,2.
        with patch.object(M,'OUT',root/'stopped'),patch.object(M.C,'step',side_effect=stop_step),patch('builtins.print'):
            try:M.worker()
            except M.Stopped:pass
            else:raise AssertionError('STOP ignored')
            assert len(calls)==2 and not (M.OUT/'completion.json').exists()
            assert M.read(M.OUT/'failure.json')['automatic_restart'] is False
        (study/'STOP').unlink()
        checks['STOP_before_next_update_and_no_restart']=True
        failed=SimpleNamespace(pid=123,poll=lambda:1,returncode=1)
        fakeproc=SimpleNamespace(create_time=lambda:1.)
        with patch.object(M,'OUT',root/'failed'),patch.object(M.subprocess,'Popen',return_value=failed),patch('psutil.Process',return_value=fakeproc),patch.object(M,'StageLock',lambda *args:contextlib.nullcontext()),patch.object(M.ctypes.windll.kernel32,'SetThreadExecutionState',return_value=1):
            try:M.coordinate()
            except AssertionError:pass
            else:raise AssertionError('Failure accepted')
            assert (study/'STOP').exists() and M.read(study/'status.json')['state']=='failed_or_stopped'
        checks['coordinator_failure_stops_without_restart']=True
M.write(M.STUDY/'adapter_verification.json',dict(passed=True,checks=checks,script_sha256=M.sha(Path(__file__))))
print(json.dumps(checks,indent=2))
