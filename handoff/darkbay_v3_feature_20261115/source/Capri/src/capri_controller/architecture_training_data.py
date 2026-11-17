"""Prepared package adapters; only pixels enter either model."""
import collections,json
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from . import architecture_data as D
from .decoder.data import CorpusDataset,pixel_inputs,resize_geometry,box_targets

class Scenes(CorpusDataset):
    def __init__(self,root,split='train'):
        self.run=Path(root);self.split=split;self.handles=collections.OrderedDict()
        self.fonts={f['key']:f for f in D.read(self.run/'fonts.json')['fonts']}
        self.family_indices={f:i for i,f in enumerate(sorted({v['split_group'] for v in self.fonts.values()}))}
        self.rows=[]
        for r in D.read_rows(self.run/(split+'.jsonl')):
            h,w=r['native_hw'];self.rows.append(dict(r,archive=str(D.local(self.run,r['archive'])),shape=resize_geometry(w,h)[2:]))
        self.snapshot_hash=D.sha(self.run/(split+'.jsonl'))
    def __getitem__(self,i):
        item=super().__getitem__(i);r=self.rows[i]
        label=json.loads(self.archive(r['archive']).read(r['stem']+'.json'))
        item['audit']['characters']=[g['char'] for g in label['glyphs']]
        return item

class References(D.PreparedReferences):
    def __init__(self,root,split='train'):
        super().__init__(root,split,'optimization' if split=='train' else 'selection_library')
        self.fonts={f['key']:f for f in D.read(self.root/'fonts.json')['fonts']}
        self.split=split;self.snapshot_hash=D.sha(self.root/('references_'+split+'.jsonl'))
        for r in self.rows:
            h,w=r['native_hw'];r['shape']=resize_geometry(w,h)[2:];r['record']=dict(r)
    def __getitem__(self,i):
        item=super().__getitem__(i);r=self.rows[i];v=item['library_labels']
        assert v['recipe']['split']==self.split
        inputs,geometry=pixel_inputs(item['inputs']['rgb'])
        labels={k:r[k] for k in ('specimen_id','font_key','family_group','split')}
        font=self.fonts[r['font_key']]
        labels.update(font_sha256=font['sha256'],style=font['style'],axes=r['axes'])
        labels.update(boxes_visible=box_targets(v['glyphs'],'box_visible',geometry),characters=[g['char'] for g in v['glyphs']],alias_families=r['pixel_alias_family_groups'])
        return dict(inputs=inputs,library_labels=labels,audit=dict(stem=r['stem'],pixel_sha256=r['pixel_sha256'],geometry=geometry))

def count_pixels(rgb):
    # Aspect-preserving fit, fixed white canvas, no label-dependent crop/scale.
    h,w=rgb.shape[:2];scale=min(64/h,384/w);nh=max(1,round(h*scale));nw=max(1,round(w*scale))
    x=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
    x=F.interpolate(x[None],size=(nh,nw),mode='bilinear',align_corners=False,antialias=True)[0]
    valid=torch.zeros(64,384,dtype=torch.bool);valid[:nh,:nw]=True
    return F.pad(x,(0,384-nw,0,64-nh),value=1.),valid

class Counts(D.PreparedScenes):
    def __init__(self,root,split='train'):super().__init__(root,split,'count')
    def __getitem__(self,i):
        item=super().__getitem__(i);x,v=count_pixels(item['inputs']['rgb'])
        return x,v,torch.tensor(item['targets']['count']-2),i
