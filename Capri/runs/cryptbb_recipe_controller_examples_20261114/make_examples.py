from pathlib import Path
import json, copy, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from capri_controller.rendering import renderer as R
root=Path('E:/pranav/Thesis-Pranav-Final')
g=root/'Capri/runs/cryptbb_model1_preparation_20261113/generator'
out=root/'Capri/runs/cryptbb_recipe_controller_examples_20261114'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
recipes=read(g/'refined_recipes.json'); entries=read(g/'entries.json'); labels=read(g/'pilot/labels.json'); refs=read(g/'inputs.json')
R.install()
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
heading=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',28)
sheet=Image.new('RGB',(1110,835),'#f4f6f8'); d=ImageDraw.Draw(sheet)
d.text((22,15),'CryptBB: starting recipe and controller results',font=heading,fill='#182033')
d.text((22,55),'First three reference images, in saved order. Synthetic text is matched within each example.',font=small,fill='#485166')
stages=['Real reference','Capri recipe*','Custom','DINO','Mixed']
for k,label in enumerate(stages):
    y=135+k*130
    d.text((18,y+32),label,font=font,fill='#182033')
records=[]
for i in range(3):
    es=[e for e in entries if e['input_index']==i]
    ll=labels[3*i:3*i+3]
    assert len({x['seed'] for x in ll})==1 and len({x['text'] for x in ll})==1
    recipe=copy.deepcopy(recipes[i]);recipe['seed']=ll[0]['seed'];recipe['text_spec']['text']=ll[0]['text']
    before=R.render((recipe,R.expand(dict(length=recipe['description']['length']))))
    Image.fromarray(before['image']).save(out/f'example_{i:03d}_recipe.png')
    replays=[]
    for entry,saved in zip(es,ll):
        rerender=R.render((recipe,entry['controls']))
        actual=np.array(Image.open(g/saved['image']).convert('RGB'))
        assert np.array_equal(rerender['image'],actual), entry['entry_id']+' replay mismatch'
        replays.append(dict(entry=entry['entry_id'],pixel_exact_replay=True,controls=entry['controls'],final_text_spec=saved['text_spec']))
    imgs=[Image.open(refs[i]['source']).convert('RGB'),Image.fromarray(before['image'])]+[Image.open(g/x['image']).convert('RGB') for x in ll]
    x=195+i*300
    d.text((x,100),f'Example {i+1} / reference {i:03d}',font=small,fill='#182033')
    card=Image.new('RGB',(620,790),'#f4f6f8'); cd=ImageDraw.Draw(card)
    cd.text((20,12),f'CryptBB example {i+1}',font=heading,fill='#182033')
    for k,im in enumerate(imgs):
        tile=ImageOps.contain(im,(286,100),Image.Resampling.NEAREST)
        sheet.paste(tile,(x+(286-tile.width)//2,135+k*130+(100-tile.height)//2))
        cd.text((20,65+k*135),stages[k],font=font,fill='#182033')
        large=ImageOps.contain(im,(580,102),Image.Resampling.NEAREST)
        card.paste(large,(20+(580-large.width)//2,94+k*135))
    cd.text((20,755),'* Re-rendered from saved recipe, before controller changes.',font=small,fill='#485166')
    card.save(out/f'example_{i+1}.png')
    records.append(dict(input_index=i,reference=refs[i],seed=recipe['seed'],synthetic_text=recipe['text_spec']['text'],base_recipe=recipes[i],baseline_usable=before['usable'],controllers=replays))
d.text((22,795),'* Starting recipe re-rendered with neutral controls; controller images are the original saved pilots.',font=small,fill='#485166')
sheet.save(out/'comparison.png')
(out/'evidence.json').write_text(json.dumps(dict(selection='First three references, no visual selection',method='Neutral recipe replay uses same synthetic text and seed as saved pilots; all nine controller replays verified pixel-exact.',examples=records),indent=2),encoding='utf8')
with (root/'handoff/CHANGELOG.md').open('a',encoding='utf8') as f:
    f.write('\n\n## 2026-11-14 CryptBB recipe/controller visual examples\nCreated Capri/runs/cryptbb_recipe_controller_examples_20261114 with the first three references, neutral-control recipe replays, original controller pilots, and evidence.json. All nine controller replays matched saved pixels exactly. CPU rendering only; no training or generator source/config changes. Starting recipe panels are explicitly marked re-renders. B cancellation remains in force; A continues independently.\n')
print(json.dumps(dict(output=str(out/'comparison.png'),controller_replays_verified=9,baseline_usable=[r['baseline_usable'] for r in records])))
