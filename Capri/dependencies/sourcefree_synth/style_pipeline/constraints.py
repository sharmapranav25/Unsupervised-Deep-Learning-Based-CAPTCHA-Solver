"""Optional measured-color constraint; does not fit or replace classifiers."""
import copy
import numpy as np

def preserve_achromatic(reference, candidate, saturation_limit=.03):
    """Project colors to equal RGB channels only for near-achromatic references.

    This experimental constraint preserves scalar intensity and never modifies
    feature predictions, text geometry, or the input dictionaries.
    """
    out=copy.deepcopy(candidate)
    applied=float(reference['saturation_mean'])<=saturation_limit
    if applied:
        def gray(c):
            value=int(round(float(np.mean(c))))
            return [value,value,value]
        for key in ('background_palette','foreground_palette'):
            out[key]=[gray(c) for c in out[key]]
        for key in ('gradient_left','gradient_right'):
            if key in out:out[key]=gray(out[key])
        for block in out['blocks']:block['rgb_mean']=gray(block['rgb_mean'])
        out['saturation_mean']=0.
        out['achromatic_noise']=True
    return out,dict(applied=applied,source_saturation=float(reference['saturation_mean']),limit=saturation_limit)
