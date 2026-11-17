
"""Behavior checks for portable dataset readers and leakage/integrity guards."""
import hashlib, io, json, shutil, zipfile
from pathlib import Path
import numpy as np
from . import architecture_data as D

def run(package, output):
    root=Path(package).resolve();out=Path(output).resolve()
    D.guard(out.drive.upper()=='E:' and not out.exists(),'Fresh E-drive diagnostics directory required')
    out.mkdir(parents=True);checks={}
    data=D.PreparedScenes(root,'train','count');row=data.rows[0];z=data.archive(row['archive']);reads=[]
    class Watch:
        def read(self,name):reads.append(name);return z.read(name)
    data.archive=lambda rel:Watch()
    sample=data[0]
    D.guard(reads==[row['stem']+'.png'],'Count loader accessed non-image supervision')
    D.guard(set(sample['inputs'])=={'rgb'} and set(sample['targets'])=={'count'},'Count contract leaked labels')
    checks['count_reads_no_mask_font_or_annotation_files']=True
    fonts={f['key']:f for f in D.read(root/'fonts.json')['fonts']}
    original={ext:z.read(row['stem']+ext) for ext in D.SUFFIXES}
    class Payload:
        def __init__(self,values):self.values=values
        def read(self,name):return self.values[name[len(row['stem']):]]
    changed=dict(original);label=json.loads(changed['.json']);label['count']=9;changed['.json']=json.dumps(label).encode()
    try:D.scene(Payload(changed),row['stem'],row['pixel_sha256'],'train',fonts)
    except ValueError:checks['invalid_count_refused']=True
    else:raise AssertionError('Invalid count accepted')
    try:D.scene(Payload(original),row['stem'],row['pixel_sha256'],'dev',fonts)
    except ValueError:checks['wrong_scene_split_refused']=True
    else:raise AssertionError('Wrong scene split accepted')
    try:D.pixels(original['.png'],'0'*64)
    except ValueError:checks['bad_pixel_hash_refused']=True
    else:raise AssertionError('Bad pixel hash accepted')
    try:D.PreparedReferences(root,'dev','optimization')
    except ValueError:checks['dev_references_cannot_optimize']=True
    else:raise AssertionError('DEV reference training accepted')
    try:D.local(root,'../outside.zip')
    except ValueError:checks['path_escape_refused']=True
    else:raise AssertionError('Path escape accepted')
    moved=out/'relocated_sample';moved.mkdir();target=moved/row['archive'];target.parent.mkdir()
    shutil.copy2(root/row['archive'],target);D.jsonl(moved/'train.jsonl',[row])
    other=D.PreparedScenes(moved,'train','count');new=other[0]
    D.guard(np.array_equal(sample['inputs']['rgb'],new['inputs']['rgb']) and sample['targets']==new['targets'],'Relocated read differs')
    checks['relative_manifest_relocation_preserves_pixels_and_count']=True;other.close();data.close()
    D.write(out/'checks.json',dict(dataset=str(root),checks=checks,all_passed=True))
    shutil.copy2(Path(__file__),out/'check_source.py')
    return checks
