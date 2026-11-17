"""Conservative proposal grouping for count diagnostics, not OCR or truth."""
import numpy as np

def count_evidence(style):
    proposals=style['glyph_proposals']
    if not proposals:return dict(count=None,confidence='insufficient',reason='no regions')
    tallest=max(p['height'] for p in proposals)
    # Remove short fragments; consider a coherent band of tall regions.
    candidates=[p for p in proposals if p['height']>=max(.45,.72*tallest) and .15<p['center'][1]<.85]
    accepted=[]
    for p in sorted(candidates,key=lambda p:-p['height']):
        if all(abs(p['center'][0]-q['center'][0])>.45*min(p['width'],q['width']) for q in accepted):accepted.append(p)
    accepted.sort(key=lambda p:p['center'][0]);count=len(accepted)
    if count<2:return dict(count=None,confidence='insufficient',reason='fewer than two tall regions')
    xs=np.array([p['center'][0] for p in accepted]);ys=np.array([p['center'][1] for p in accepted])
    spacing=np.diff(xs);cv=float(spacing.std()/max(spacing.mean(),1e-6))
    reliable=2<=count<=8 and cv<.35 and ys.std()<.12 and xs[-1]-xs[0]>.35
    return dict(count=count,confidence='candidate' if reliable else 'low',spacing_cv=cv,
                vertical_std=float(ys.std()),centers=xs.tolist(),reason='unverified tall-region spatial consensus')
