"""Frozen pixel-reference importance and per-image synthetic feature groups."""
import json
import hashlib
import zipfile
from collections import defaultdict
import numpy as np

GEOMETRY = {
    'spacing': ('spacing', ['wide_spacing','overlap_tight_spacing']),
    'width': ('x_scale', ['condensed','expanded']),
    'rotation': ('rotation', ['per_char_rotation']),
    'vertical_jitter': ('jitter', ['vertical_jitter']),
    'size_variation': ('size_jitter', ['size_variation']),
    'slant': ('slant', ['slant']),
    'text_curve': ('curve', ['text_on_curve']),
}
EFFECTS = {'grid':'bg_grid','border':'border_frame','blocks':'bg_blocks','grain':'pixel_noise',
           'circles':'bg_circles','curves':'curves','occluders':'occlusion','dots':'dot_noise',
           'morph':'morph_distortion','texture':'bg_image_texture','blobs':'blob_noise',
           'noise_over':'noise_over_text','jpeg':'compression','pattern':'bg_pattern','scribbles':'scribbles'}


def build(root, out, read, sha, write):
    data=root/'datasets/cryptbb_v3/A'
    raw_path=root/'Capri/runs/cryptbb_v3_source_20261114/raw_hybrid_predictions.json'
    raw=read(raw_path);assert len(raw)==100 and {r['index'] for r in raw}==set(range(100))
    ranking=root/'model2/rankings/cryptbb_v3_epoch20_static_A_20261115/A'
    done=read(ranking/'completion.json')
    assert done['teacher']=='B' and done['teacher_epoch']==20
    assert sha(ranking/'ranking.npz')==done['sha256']
    with np.load(ranking/'ranking.npz',allow_pickle=False) as z:
        ids=z['ids'].copy();buckets=z['buckets'].copy();nll=z['nll'].copy()
    assert ids.tolist()==list(range(1,100001)) and np.isfinite(nll).all()
    expected=np.empty(100000,dtype=np.int64)
    for b,ix in enumerate(np.array_split(np.lexsort((ids,nll)),3)):expected[ix]=b
    assert np.array_equal(expected,buckets)
    receipt=read(data/'data_receipt.json')
    assert sha(data/'sample_provenance.jsonl')==receipt['provenance_sha256']
    assert sha(data/'train_lmdb/data.mdb')==receipt['lmdb_sha256']
    prov=defaultdict(dict)
    for line in (data/'sample_provenance.jsonl').read_text().splitlines():
        p=json.loads(line);prov[p['shard']][p['image']]=p
    numeric={name:np.zeros(100000,float) for name in GEOMETRY}
    effects={name:np.zeros(100000,float) for name in EFFECTS}
    font_ids=[None]*100000;charsets=[None]*100000;truth=[None]*100000;seen=set();font_refs=defaultdict(set);hashes={}
    for shard,items in prov.items():
        with zipfile.ZipFile(data/shard) as z:blob=z.read('labels.jsonl')
        hashes[shard]=hashlib.sha256(blob).hexdigest()
        rows=[json.loads(line) for line in blob.splitlines()];assert len(rows)==len(items)
        for r in rows:
            p=items[r['image']];i=p['index']-1
            assert i not in seen;seen.add(i)
            assert all(r[k]==p[k] for k in ('pixel_sha256','text','seed','entry_id','input_index','controller','engine'))
            t=r['text_spec'];e={v['name']:v['level'] for v in r['effects_spec']['effects']}
            for name,(key,_) in GEOMETRY.items():numeric[name][i]=t.get(key,0)
            for name in EFFECTS:effects[name][i]=e.get(name,0)
            font=tuple(t['fonts']);font_ids[i]=font;font_refs[font].add(r['input_index']);truth[i]=r['text'];charsets[i]=t['charset']
    assert seen==set(range(100000))
    families=[];columns=[]
    def add(name,count,values,present,source):
        values=np.asarray(values);present=np.asarray(present,bool)
        if not present.any() or count==0:return
        unique=np.unique(values[present]);cuts=[]
        if len(unique)>8:
            cuts=np.unique(np.quantile(values[present],[.2,.4,.6,.8])).tolist()
            bins=np.searchsorted(cuts,values,side='right')
        else:bins=np.searchsorted(unique,values)
        column=np.where(present,bins,-1).astype(np.int32)
        labels={int(g):dict(min=float(values[column==g].min()),max=float(values[column==g].max()),
                           images=int((column==g).sum())) for g in np.unique(column[present])}
        families.append(dict(name=name,reference_count=count,importance=count/100,source=source,
                             cuts=cuts,groups=labels,images=int(present.sum())))
        columns.append(column)
    for name,(field,keys) in GEOMETRY.items():
        count=sum(any(r['prediction']['text_features'][k]['present'] for k in keys) for r in raw)
        v=numeric[name]
        present=np.ones(100000,bool) if name in ('spacing','width') else v!=0
        add(name,count,v,present,dict(kind='raw_pixel_reference_prediction',keys=keys,renderer_field=field))
    for name,key in EFFECTS.items():
        count=sum(r['prediction']['effects'][key]['present'] for r in raw)
        add('effect:'+name,count,effects[name],effects[name]>0,
            dict(kind='raw_pixel_reference_prediction',key=key,renderer_effect=name))
    for charset,key in [('mixed','mixed_case_alnum'),('upper','uppercase_only'),('lower','lowercase_only'),('digits','digits_only')]:
        count=sum(r['prediction']['text_features'][key]['present'] for r in raw)
        mask=np.asarray(charsets)==charset
        add('content:'+charset,count,np.zeros(100000),mask,
            dict(kind='raw_pixel_reference_prediction',key=key,renderer_charset=charset))
    # Font names come from fitted controller recipes, not an assertion that the
    # original site's exact font has been recovered. Count distinct references.
    for font,refs in sorted(font_refs.items()):
        mask=np.array([v==font for v in font_ids])
        add('font:'+','.join(font),len(refs),np.zeros(100000),mask,
            dict(kind='fitted_font_reference_coverage',font_set=list(font),reference_ids=sorted(refs)))
    order=sorted(range(len(families)),key=lambda f:(-families[f]['importance'],families[f]['name']))
    families=[families[f] for f in order];matrix=np.stack([columns[f] for f in order],axis=1)
    for i,f in enumerate(families):f['rank']=i+1
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'features.npz',ids=ids,buckets=buckets,groups=matrix,truth=np.asarray(truth))
    manifest=dict(images=100000,references=100,families=families,ranking=done,
        features_sha256=sha(out/'features.npz'),reference_predictions_sha256=sha(raw_path),
        provenance_sha256=receipt['provenance_sha256'],lmdb_sha256=receipt['lmdb_sha256'],metadata_hashes=hashes,
        grouping='Exact settings when <=8 values, otherwise metadata-only quintile bins; no correctness used.',
        importance='Raw reference prevalence for geometry/effects. Exact synthetic font sets use unique source-reference coverage of fitted recipes, not inferred real font identity.',
        ties='Descending reference prevalence then lexical family name; never student accuracy or real labels.',
        exclusions='Zero-reference features have no targeted priority but all their images retain uniform coverage. No real strings or labels read.')
    write(out/'manifest.json',manifest)
    return manifest
