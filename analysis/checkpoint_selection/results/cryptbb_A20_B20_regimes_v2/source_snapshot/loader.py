"""Loads only frozen predictions, never labels or accuracy summaries."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np
DEFAULT_ROOT=Path('E:/pranav/Thesis-Pranav-Final')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
@dataclass(frozen=True)
class PredictionTable:
    run: str
    epochs: np.ndarray
    ids: tuple
    predictions: np.ndarray
    sources: tuple=()
    def __post_init__(self):
        assert self.predictions.shape==(len(self.epochs),len(self.ids))
        assert len(set(self.ids))==len(self.ids)
        assert np.all(np.diff(self.epochs)>0)
        assert all(isinstance(s,str) for s in self.predictions.flat)
        self.predictions.setflags(write=False)

def load(run,root=DEFAULT_ROOT,through_epoch=None):
    assert run in ('Model1-A','Model1-B')
    root=Path(root)
    manifest=root/'evaluation_data/model4_reservation_20261114/cryptbb/evaluation_pool.json'
    ids=tuple(sorted(x['opaque_id'] for x in read(manifest)))
    assert len(ids)==251
    stop=(20 if run=='Model1-A' else 16) if through_epoch is None else through_epoch
    assert isinstance(stop,int) and 12<=stop<=20
    epochs=np.arange(1,stop+1)
    base=root/('model1/results/cryptbb/A' if run=='Model1-A' else 'model1/results/cryptbb_v3/B')
    prefix='reserved251_cpu_epoch' if run=='Model1-A' else 'requested_cpu_epoch'
    values=[];sources=[]
    for e in epochs:
        p=base/f'{prefix}{e:02d}'/'predictions.json'
        rows=read(p);f=read(p.parent/'prediction_freeze.json')
        assert sha(p)==f['sha256'] and f['images']==251
        assert all(set(row)=={'opaque_id','prediction'} for row in rows), 'Labels/extra fields forbidden'
        d={row['opaque_id']:row['prediction'] for row in rows}
        assert len(rows)==251 and tuple(sorted(d))==ids
        values.append([d[i] for i in ids])
        sources.append(dict(epoch=int(e),path=str(p),sha256=sha(p),checkpoint_sha256=f['checkpoint_sha256']))
    return PredictionTable(run,epochs,ids,np.array(values,dtype=object),tuple(sources))
