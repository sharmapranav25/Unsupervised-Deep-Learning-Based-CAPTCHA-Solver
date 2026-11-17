"""Study-scoped IEEE FP32 assessment execution; frozen engine stays intact."""
import contextlib
import sys
from pathlib import Path
from . import common as W


@contextlib.contextmanager
def ieee_inference():
    import torch
    before=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        yield
    finally:
        torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32=before


def verify_execution():
    W.verify_sources();W.stop()
    manifest=W.read(W.STUDY/'execution_fp32_manifest.json')
    assert W.sha(Path(__file__))==manifest['overlay_sha256']
    assert W.sha(W.STUDY/'source_manifest.json')==manifest['frozen_manifest_sha256']
    return manifest


def install():
    from . import engine as E
    original=E.predict
    def predict(*args,**kwargs):
        verify_execution()
        with ieee_inference():
            return original(*args,**kwargs)
    E.predict=predict


def main():
    W.environment();manifest=verify_execution()
    mode=sys.argv.pop(1)
    if mode=='readers':
        from . import readers
        install()
        stage=sys.argv[1]
        if stage in ('curriculum-check','worker'):
            tag=sys.argv[sys.argv.index('--dataset')+1]
            numbers=(2,3) if stage=='curriculum-check' else (int(sys.argv[sys.argv.index('--model')+1]),)
            for number in numbers:
                protocol=W.read(W.stage(number,tag)/'protocol.json')
                assert protocol['execution_precision']['manifest_sha256']==W.sha(W.STUDY/'execution_fp32_manifest.json')
                W.write(W.stage(number,tag)/'execution_fp32_active.json',dict(at=W.now(),stage=stage,manifest_sha256=W.sha(W.STUDY/'execution_fp32_manifest.json'),overlay_sha256=manifest['overlay_sha256'],inference_cudnn_tf32=False,inference_matmul_tf32=False,training_flags_restored=True))
        readers.main()
    elif mode=='coordinator':
        from . import coordinator as K
        original=K.launch_child
        def launch_child(module,args,name):
            verify_execution()
            if module in ('readers','coordinator'):
                return original('execution_fp32',[module,*args],name)
            return original(module,args,name)
        K.launch_child=launch_child
        K.main()
    else:
        raise ValueError(mode)


if __name__=='__main__':main()
