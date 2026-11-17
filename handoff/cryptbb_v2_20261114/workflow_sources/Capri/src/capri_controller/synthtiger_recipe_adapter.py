"""Render saved CAPRI starting recipes with SynthTIGER glyphs and shared layout/effects.

This is a recipe-consumption adapter, not the independent stripe-based controller.
No fitting or controller overrides are applied in this diagnostic.
"""
import copy, math, json, time
from pathlib import Path
from . import synthtiger_validation as S
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from .rendering import renderer as R
ROOT=S.ROOT
GEN=ROOT/'Capri/runs/cryptbb_model1_preparation_20261113/generator'
OUT=ROOT/'Capri/runs/cryptbb_synthtiger_starting_recipes_20261114'
CALLS=0

class RecipeTextLayer(S.LatinTextLayer):
    """Preserve variable-font axes and safely pad glyph bearings on modern Pillow."""
    def __init__(self,meta,char,size,weight,width_axis,stroke):
        self.meta=meta;self.weight=weight;self.width_axis=width_axis;self.stroke=stroke
        super().__init__(char,meta['path'],size,color=(255,255,255,255),bold=stroke)
    def _read_font(self,path,size):
        font=ImageFont.truetype(path,size)
        if self.meta.get('axes') and (self.weight is not None or self.width_axis is not None):
            vals=[]
            for a in self.meta['axes']:
                v=self.weight if a['tag']=='wght' else self.width_axis if a['tag']=='wdth' else None
                vals.append(a['default'] if v is None else float(np.clip(v,a['min'],a['max'])))
            font.set_variation_by_axes(vals)
        return font
    def _get_image(self,text,font,color,bold,vertical):
        assert not vertical
        stroke=int(bold);l,t,r,b=font.getbbox(text,stroke_width=stroke)
        pad=max(4,stroke+2)
        im=Image.new('RGBA',(max(1,r-l+2*pad),max(1,b-t+2*pad)))
        ImageDraw.Draw(im).text((pad-l,pad-t),text,font=font,fill=color,stroke_width=stroke,stroke_fill=color)
        return np.asarray(im,dtype=np.float32),[0,0,im.width,im.height]

def glyph_mask(meta,ch,size_px,ss=4,weight=None,width_axis=None,stroke=0,x_scale=1.,slant=0.,angle=0.):
    global CALLS
    CALLS+=1
    layer=RecipeTextLayer(meta,ch,max(1,int(size_px*ss)),weight,width_axis,int(stroke*ss))
    # Native SynthTIGER layer geometry and raster output; no CAPRI glyph fallback.
    layer.size=(layer.width*max(.3,x_scale),layer.height)
    q=layer.quad; center=layer.center
    q[:,0] += float(slant)*(q[:,1]-center[1]);layer.quad=q
    if angle:S.components.Rotate(angle=(angle,angle),ccw=1).apply([layer],{'angle':float(angle),'ccw':True})
    alpha=np.clip(layer.output()[:,:,3],0,255).astype('uint8')
    im=Image.fromarray(alpha).resize((max(1,round(alpha.shape[1]/ss)),max(1,round(alpha.shape[0]/ss))),Image.Resampling.BOX)
    m=np.asarray(im,dtype=np.float32)/255.;y,x=np.nonzero(m>.05)
    return m[y.min():y.max()+1,x.min():x.max()+1] if len(x) else None

def draw(recipe,tiger):
    before=R.ORIGINAL
    np.random.seed(int(recipe['seed'])%2**32)
    try:
        if tiger:R.ORIGINAL=glyph_mask
        return R.render((copy.deepcopy(recipe),R.expand(dict(length=recipe['description']['length']))))
    finally:R.ORIGINAL=before

def compact(row):
    return {k:v for k,v in row.items() if k not in ('image','mask')}

def main():
    assert not OUT.exists(),'Preserve previous evidence'
    OUT.mkdir(parents=True);(OUT/'capri').mkdir();(OUT/'synthtiger').mkdir()
    R.install();start=time.monotonic()
    recipes=S.read(GEN/'refined_recipes.json');pilots=S.read(GEN/'pilot/labels.json');refs=S.read(GEN/'inputs.json')
    original_hash=S.sha(GEN/'refined_recipes.json')
    coverage=dict(synthtiger=['chosen font file','variable-font weight','bold stroke','glyph rasterization','horizontal scale','slant/shear','rotation'],shared_capri=['layout and spacing','size jitter','vertical jitter','curve','palette selection','background','all requested under/glyph/over/post effects'],omitted_fields=[],controller_adjustments=False)
    S.write(OUT/'protocol.json',dict(role='diagnostic_recipe_adapter',count=100,recipe_source=str(GEN/'refined_recipes.json'),recipe_source_sha256=original_hash,source_sha256=S.sha(__file__),text_and_seed='Copied from each first saved controller pilot, synthetic strings only; matched between CAPRI and SynthTIGER',coverage=coverage,target_labels_used=False,reader_used=False,training_data_changed=False))
    records=[];ims=[]
    for i,base in enumerate(recipes):
        r=copy.deepcopy(base);r['seed']=pilots[3*i]['seed'];r['text_spec']['text']=pilots[3*i]['text']
        unmodified=copy.deepcopy(r)
        a=draw(r,False);b=draw(r,True)
        assert r==unmodified
        assert a['text_spec']==b['text_spec']==r['text_spec']
        assert a['effects_spec']==b['effects_spec']==r['effects_spec']
        assert a['text']==b['text']==r['text_spec']['text']
        assert len(b['glyphs'])==r['description']['length']
        assert [x['font'] for x in a['glyphs']]==[x['font'] for x in b['glyphs']]
        assert a['image'].shape==b['image'].shape==(r['size'][1],r['size'][0],3)
        Image.fromarray(a['image']).save(OUT/f'capri/{i:03d}.png');Image.fromarray(b['image']).save(OUT/f'synthtiger/{i:03d}.png')
        records.append(dict(input_index=i,render_recipe=r,capri=compact(a),synthtiger=compact(b)))
        if i<6:ims.append((a['image'],b['image']))
        S.write(OUT/'status.json',dict(stage='rendering',completed=i+1,total=100))
        if (i+1)%20==0:print('rendered',i+1,flush=True)
    for i in [0,49,99]:
        row=draw(records[i]['render_recipe'],True)
        assert row['pixel_sha256']==records[i]['synthtiger']['pixel_sha256']
    assert S.sha(GEN/'refined_recipes.json')==original_hash
    S.write(OUT/'manifest.json',records)
    summary=dict(recipes=100,paired_images=200,synthtiger_glyph_calls=CALLS,recipe_specs_preserved=100,fonts_preserved=100,texts_and_seeds_matched=100,deterministic_replays=3,capri_complete=sum(r['capri']['complete'] for r in records),synthtiger_complete=sum(r['synthtiger']['complete'] for r in records),capri_usable=sum(r['capri']['usable'] for r in records),synthtiger_usable=sum(r['synthtiger']['usable'] for r in records),synthtiger_unique_pixels=len({r['synthtiger']['pixel_sha256'] for r in records}),coverage=coverage,seconds=time.monotonic()-start)
    S.write(OUT/'summary.json',summary)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    sheet=Image.new('RGB',(1110,555),'#f4f6f8');d=ImageDraw.Draw(sheet)
    d.text((18,14),'Same Capri starting recipes: original rendering vs SynthTIGER adapter',font=font,fill='#182033')
    d.text((18,52),'First three recipes. Same fonts, synthetic text, seeds and recipe settings; no refinement.',font=small,fill='#485166')
    for i in range(3):
        ref=np.array(Image.open(refs[i]['source']).convert('RGB'));assert S.phash(ref)==refs[i]['pixel_sha256']
        for k,a in enumerate([ref,*ims[i]]):
            if i==0:d.text((16,128+k*135),['Reference','Capri render','SynthTIGER'][k],font=font,fill='#182033')
            tile=ImageOps.contain(Image.fromarray(a),(286,100),Image.Resampling.NEAREST);sheet.paste(tile,(195+300*i,100+k*135))
    d.text((18,515),'SynthTIGER renders glyphs; CAPRI layout, colours and effects remain shared. No stripe template.',font=small,fill='#485166')
    sheet.save(OUT/'comparison.png')
    S.write(OUT/'status.json',dict(stage='complete',seconds=time.monotonic()-start));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
