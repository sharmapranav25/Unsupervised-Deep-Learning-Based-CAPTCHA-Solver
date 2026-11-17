from pathlib import Path
import json,hashlib,time
from PIL import Image,ImageDraw,ImageFont,ImageOps
from capri_controller.controller_v2 import _install,draw_recipe,render_job
ROOT=Path.cwd();FIT=ROOT/'Capri/runs/cryptbb_controller_v3split_full_20261114';SRC=ROOT/'Capri/runs/cryptbb_v3_source_20261114'
OUT=ROOT/'Capri/runs/cryptbb_v3_live_preview_20261114';OUT.mkdir(exist_ok=False)
for attempt in range(5):
 try:
  raw=(FIT/'entries.json').read_bytes();entries=json.loads(raw);break
 except json.JSONDecodeError:time.sleep(.1)
(OUT/'entries_snapshot.json').write_bytes(raw)
recipes=json.loads((SRC/'refined_recipes.json').read_text());refs=json.loads((SRC/'inputs.json').read_text())
indices=[0,4,8,12,16];arms=['custom','dino','synthtiger','mixed'];lookup={(e['input_index'],e['controller']):e for e in entries}
assert all((i,a) in lookup for i in indices for a in arms)
fontpath='C:/Windows/Fonts/arial.ttf';title=ImageFont.truetype(fontpath,27);font=ImageFont.truetype(fontpath,20);small=ImageFont.truetype(fontpath,16)
w=1700;top=120;rowh=190;cw=330
sheet=Image.new('RGB',(w,top+len(indices)*rowh+40),'#f2f4f7');d=ImageDraw.Draw(sheet)
d.text((20,14),'CryptBB v3 — fresh samples from completed recipes',font=title,fill='#182033')
d.text((20,51),'First-17 snapshot: reference indices 0, 4, 8, 12, 16. Two new strings per controller; no visual selection.',font=small,fill='#475569')
headers=['Real reference','Custom: thickness / colour','DINO: background / layout','SynthTIGER: font','Mixed: combined + refine']
for c,h in enumerate(headers):d.text((20+c*cw,88),h,font=font,fill='#182033')
_install();records=[]
for row,i in enumerate(indices):
 y=top+row*rowh
 for c in range(5):d.rounded_rectangle((14+c*cw,y,14+(c+1)*cw-8,y+rowh-9),radius=8,fill='white')
 ref=Image.open(refs[i]['source']).convert('RGB');tile=ImageOps.contain(ref,(310,100),Image.Resampling.NEAREST);sheet.paste(tile,(20+(310-tile.width)//2,y+30));d.text((24,y+145),f'Reference {i:03d}',font=small,fill='#475569')
 for col,arm in enumerate(arms,1):
  e=lookup[(i,arm)];flags=[]
  for j in range(2):
   seed=80000000000+i*100000+col*1000+j;r=render_job((draw_recipe(recipes[i],seed),e['controls'],e['engine']))
   image=r.get('image');ok=bool(r.get('complete') and r.get('usable'));flags.append(ok)
   record={'reference_index':i,'controller':arm,'engine':e['engine'],'seed':seed,'complete':bool(r.get('complete')),'usable':bool(r.get('usable')),'unvalidated_entry':bool(e.get('unvalidated')),'text':r.get('text'),'contrast':r.get('text_contrast'),'error':r.get('error')}
   if image is not None:
    im=Image.fromarray(image);name=f'{i:03d}_{arm}_{j}.png';im.save(OUT/name);record.update(image=name,pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest());tile=ImageOps.contain(im,(310,64),Image.Resampling.NEAREST);sheet.paste(tile,(20+col*cw+(310-tile.width)//2,y+10+j*69))
   else:d.text((24+col*cw,y+20+j*69),'Render failed — retained',font=small,fill='#b91c1c')
   records.append(record)
  msg='2/2 usable' if all(flags) else f'{sum(flags)}/2 usable — failures retained'
  if e.get('unvalidated'):msg+='; recipe unvalidated'
  d.text((24+col*cw,y+151),msg,font=small,fill='#475569' if all(flags) and not e.get('unvalidated') else '#b91c1c')
sheet.save(OUT/'comparison.png')
receipt={'entry_snapshot_sha256':hashlib.sha256(raw).hexdigest(),'indices':indices,'render_count':len(records),'usable':sum(r['usable'] and r['complete'] for r in records),'recipe_unvalidated':sorted({f"{r['controller']}_{r['reference_index']:03d}" for r in records if r['unvalidated_entry']}),'new_seeds_not_production_pool':True,'training_unchanged':True,'records':records}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='records'},indent=2));print(str(OUT/'comparison.png'))
