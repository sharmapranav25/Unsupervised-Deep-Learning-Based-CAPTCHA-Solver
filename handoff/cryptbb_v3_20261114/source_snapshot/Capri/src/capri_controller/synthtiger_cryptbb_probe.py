"""Bounded CPU probe of the existing independent SynthTIGER design on CryptBB."""
import copy, json, time
from pathlib import Path
from . import synthtiger_validation as S
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from scipy.spatial.distance import cdist
ROOT=S.ROOT
GEN=ROOT/'Capri/runs/cryptbb_model1_preparation_20261113/generator'
OUT=ROOT/'Capri/runs/cryptbb_synthtiger_probe_20261114'

def main():
    assert not OUT.exists(), 'Preserve existing evidence; use a fresh run'
    OUT.mkdir(parents=True); S.RUN=OUT
    start=time.monotonic()
    inputs=S.read(GEN/'inputs.json')
    pixels=[np.asarray(Image.open(r['source']).convert('RGB')) for r in inputs]
    assert len(pixels)==100
    for r,a in zip(inputs,pixels): assert S.phash(a)==r['pixel_sha256']
    H,W=pixels[0].shape[:2]; assert all(a.shape==(H,W,3) for a in pixels)
    vote=S.read(GEN/'length_decision.json');assert vote['mode']=='majority'
    charset=(ROOT/'datasets/cryptbb/charset.txt').read_text().splitlines()
    old=ROOT/'Capri/runs/darkbay_controller4_synthtiger_1000_20261113'
    assert charset==(ROOT/'datasets/darkbay/A/charset.txt').read_text().splitlines()
    fonts=S.read(old/'font_render_validation.json')['accepted']
    for f in fonts:assert S.sha(ROOT/f['path'])==f['sha256']
    S.write(OUT/'protocol.json',dict(role='diagnostic_preview_only',references=100,fit_candidates=64,fit_seeds_per_candidate=24,fresh_test_images=100,fit_seed=S.SEED_FIT,test_seed=4120000000,reference_pixels_only=True,target_labels_used=False,reader_used=False,length=vote['default_length'],count_votes=vote['votes'],font_pool='314 previously render-verified assets, re-ranked using CryptBB pixels',source_sha256=S.sha(__file__),renderer_source_sha256=S.sha(S.__file__),limitations=['Existing native SynthTIGER recipe family: striped backgrounds and grayscale glyphs; not a CAPRI recipe translator','Similarity diagnostic, not OCR or transfer accuracy','No production corpus, reader, committee rejection gate, or active training changes']))
    S.write(OUT/'status.json',dict(stage='font_matching',fonts=len(fonts)))
    fonts=S.shape_fonts(fonts,charset,pixels,H)
    stats=[]
    for a in pixels:
        x=a.astype(float);g=x.mean(2)/255
        bg=x[(g>.20)&(g<.82)]
        if len(bg)<20:bg=x.reshape(-1,3)
        _,_,boxes,_=S.cv2.connectedComponentsWithStats(((g<.16)|(g>.85)).astype('uint8'))
        heights=[int(b[3]) for b in boxes[1:] if b[4]>=8 and 3<=b[2]<W*.3 and 7<=b[3]<H]
        stats.append(dict(background=np.median(bg,axis=0).tolist(),spread=float(np.median(np.std(bg,axis=0))),height=float(np.median(heights)) if heights else H*.6,white_probability=float((g>.85).sum()/max(1,((g<.16)|(g>.85)).sum()))))
    ctx=dict(width=W,height=H,length=vote['default_length'],charset=charset,stats=stats,fonts=fonts)
    S.write(OUT/'reference_measurements.json',ctx)
    target=np.stack([S.feature(a) for a in pixels]);std=np.maximum(target.std(0),.02);T=target/std;tt=cdist(T,T).mean()
    def distance(images):
        F=np.stack([S.feature(a) for a in images])/std
        return float(2*cdist(F,T).mean()-cdist(F,F).mean()-tt)
    rng=np.random.default_rng(47138);candidates=[]
    for i in range(64):
        recipe=dict(stripe_min=int(rng.choice([2,3,4,6])),stripe_max=int(rng.choice([9,13,18,24])),stripe_contrast=float(rng.uniform(.5,1.4)),horiz_count=int(rng.integers(0,4)),height_scale=float(rng.uniform(.85,1.30)),width_scale=float(rng.uniform(.75,1.25)),rotation=float(rng.uniform(5,23)),jitter=float(rng.uniform(.04,.20)),blur=float(rng.choice([0.,.25,.4])))
        ims=[S.render(ctx,recipe,S.SEED_FIT+j)[0] for j in range(24)]
        candidates.append(dict(id=i,recipe=recipe,pixel_energy_distance=distance(ims)))
        S.write(OUT/'status.json',dict(stage='fit',completed=i+1,total=64,seconds=time.monotonic()-start))
        if i%8==0:print('fit',i+1,'/64',flush=True)
    selected=sorted(candidates,key=lambda c:c['pixel_energy_distance'])[:8]
    S.write(OUT/'fit.json',dict(candidates=candidates,selected=selected))
    fresh=[];records=[];seen=set();refhashes={r['pixel_sha256'] for r in inputs}
    (OUT/'images').mkdir()
    for i in range(100):
        seed=4120000000+i;chosen=selected[i%8];a,t,details=S.render(ctx,chosen['recipe'],seed);digest=S.phash(a)
        assert digest not in seen|refhashes;seen.add(digest)
        assert len(t)==ctx['length'] and set(t)<=set(charset) and len(details)==ctx['length']
        Image.fromarray(a).save(OUT/f'images/{i:03d}.png');fresh.append(a)
        records.append(dict(index=i,seed=seed,recipe_id=chosen['id'],text=t,pixel_sha256=digest,glyphs=details))
    for i in [0,49,99]:
        r=records[i];a,t,_=S.render(ctx,next(c['recipe'] for c in selected if c['id']==r['recipe_id']),r['seed'])
        assert S.phash(a)==r['pixel_sha256'] and t==r['text']
    S.write(OUT/'manifest.json',records)
    pilot=S.read(GEN/'pilot/labels.json')
    scores={arm:distance([np.array(Image.open(GEN/pilot[3*i+j]['image']).convert('RGB')) for i in range(100)]) for j,arm in enumerate(['custom','dino','mixed'])}
    scores['synthtiger']=distance(fresh)
    # Existing saved starting-candidate scores, only a descriptive committee audit.
    histories=S.read(GEN/'controller_history.json');audit=[]
    for h in histories:
        neutral=h['fit'][0];assert neutral['phase']=='initial'
        ranks={a:sum(r['scores'][a]<neutral['scores'][a] for r in h['fit'])+1 for a in ['custom','dino','mixed']}
        audit.append(dict(input_index=h['input_index'],neutral_scores=neutral['scores'],neutral_usable_of_two=neutral['usable'],rank_among_candidates=ranks,candidates=len(h['fit'])))
    S.write(OUT/'starting_recipe_committee_audit.json',dict(note='Ranks are relative to candidates for the same reference, not calibrated pass/fail judgments. Mixed is not independent. No recipes rejected.',references=audit))
    S.write(OUT/'summary.json',dict(images=100,unique_pixels=len(seen),complete_glyph_records=100,deterministic_replays=3,pixel_energy_distance_lower_better=scores,metric_caveat='SynthTIGER optimized this descriptor on fitting seeds; fresh seeds test stability, not an independent quality metric or real OCR accuracy. Other arms use original saved pilots.',initial_recipes_fully_usable_on_both_trial_renders=sum(x['neutral_usable_of_two']==2 for x in audit),total_initial_recipes=100,committee_gate_deployed=False,seconds=time.monotonic()-start))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    sheet=Image.new('RGB',(1110,850),'#f4f6f8');d=ImageDraw.Draw(sheet)
    d.text((20,12),'CryptBB: existing controllers and independent SynthTIGER probe',font=font,fill='#182033')
    d.text((20,48),'First three samples in saved order. SynthTIGER is distribution-fitted; text differs.',font=small,fill='#485166')
    for i in range(3):
        imgs=[pixels[i]]+[np.array(Image.open(GEN/pilot[3*i+j]['image']).convert('RGB')) for j in range(3)]+[fresh[i]]
        for k,a in enumerate(imgs):
            if i==0:d.text((15,130+k*135),['Reference','Custom','DINO','Mixed','SynthTIGER'][k],font=font,fill='#182033')
            tile=ImageOps.contain(Image.fromarray(a),(286,100),Image.Resampling.NEAREST);sheet.paste(tile,(195+300*i,100+k*135))
    d.text((20,807),'Native SynthTIGER design tested unchanged: stripes + grayscale glyphs. Diagnostic only.',font=small,fill='#485166')
    sheet.save(OUT/'comparison.png')
    S.sheet(fresh[:24],OUT/'synthtiger_samples.png',[r['text'] for r in records[:24]])
    S.write(OUT/'status.json',dict(stage='complete',seconds=time.monotonic()-start));print(json.dumps(S.read(OUT/'summary.json')),flush=True)
if __name__=='__main__':main()
