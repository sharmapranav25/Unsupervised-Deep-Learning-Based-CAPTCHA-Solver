"""Render the first164 completed v3 recipes into an evidence gallery; no fit edits."""
from pathlib import Path
import json,hashlib,time,html,math
from concurrent.futures import ProcessPoolExecutor
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Capri/runs/cryptbb_v3_gallery164_20261114'
FIT=ROOT/'Capri/runs/cryptbb_controller_v3split_full_20261114'
SOURCE=ROOT/'Capri/runs/cryptbb_v3_source_20261114'
ARMS=['custom','dino','synthtiger','mixed']

def init():
 from capri_controller.controller_v2 import _install
 _install()

def render(task):
 from capri_controller.controller_v2 import render_job,draw_recipe
 e,recipe,seed=task
 row=render_job((draw_recipe(recipe,seed),e['controls'],e['engine']))
 r={'entry_id':e['entry_id'],'input_index':e['input_index'],'controller':e['controller'],'engine':e['engine'],'seed':seed,'recipe_unvalidated':bool(e.get('unvalidated')),'render_complete':bool(row.get('complete')),'render_usable':bool(row.get('usable')),'text':row.get('text'),'contrast':row.get('text_contrast'),'error':row.get('error')}
 if row.get('image') is not None:
  im=Image.fromarray(row['image']);name='samples/'+e['entry_id']+'.png';im.save(OUT/name);r.update(image=name,pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest())
 return r

def main():
 assert not OUT.exists();(OUT/'samples').mkdir(parents=True)
 for attempt in range(5):
  try:raw=(FIT/'entries.json').read_bytes();allentries=json.loads(raw);break
  except json.JSONDecodeError:time.sleep(.1)
 (OUT/'full_entries_snapshot.json').write_bytes(raw)
 entries=[e for e in allentries if e['input_index']<41];assert len(entries)==164
 assert all(len([e for e in entries if e['input_index']==i])==4 for i in range(41))
 (OUT/'selected_entries.json').write_text(json.dumps(entries,indent=2))
 recipes=json.loads((SOURCE/'refined_recipes.json').read_text());refs=json.loads((SOURCE/'inputs.json').read_text())
 with ProcessPoolExecutor(max_workers=2,initializer=init) as pool:
  rows=list(pool.map(render,[(e,recipes[e['input_index']],81000000000+e['input_index']*10000+ARMS.index(e['controller'])) for e in entries],chunksize=2))
 lookup={(r['input_index'],r['controller']):r for r in rows}
 (OUT/'references').mkdir()
 for i in range(41):
  with Image.open(refs[i]['source']) as im:
   im=im.convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==refs[i]['pixel_sha256'];im.save(OUT/'references'/f'{i:03d}.png')
 # Six readable contact sheets, retaining every failed render/recipe.
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
 pages=[]
 for pg in range(math.ceil(41/8)):
  indices=list(range(pg*8,min(41,(pg+1)*8)));cw=280;rh=130
  sheet=Image.new('RGB',(1420,110+len(indices)*rh),'#f2f4f7');d=ImageDraw.Draw(sheet)
  d.text((15,12),f'CryptBB v3: all 164 recipes — page {pg+1}/6',font=font(24),fill='#182033')
  d.text((15,43),'One fresh render per recipe. Red = recipe validation or this render failed; no retries or visual selection.',font=font(16),fill='#475569')
  for c,h in enumerate(['Real reference','Custom','DINO','SynthTIGER','Mixed']):d.text((15+c*cw,76),h,font=font(21),fill='#182033')
  for rowidx,i in enumerate(indices):
   y=110+rowidx*rh
   for c in range(5):d.rounded_rectangle((10+c*cw,y,5+(c+1)*cw,y+rh-7),radius=6,fill='white')
   tile=ImageOps.contain(Image.open(OUT/'references'/f'{i:03d}.png').convert('RGB'),(260,80),Image.Resampling.NEAREST);sheet.paste(tile,(17+(260-tile.width)//2,y+4));d.text((18,y+90),f'Reference {i:03d}',font=font(16),fill='#475569')
   for c,arm in enumerate(ARMS,1):
    r=lookup[i,arm]
    if 'image' in r:
     tile=ImageOps.contain(Image.open(OUT/r['image']).convert('RGB'),(260,80),Image.Resampling.NEAREST);sheet.paste(tile,(17+c*cw+(260-tile.width)//2,y+4))
    else:d.text((18+c*cw,y+30),'Render failed',font=font(18),fill='#b91c1c')
    bad=r['recipe_unvalidated'] or not(r['render_complete'] and r['render_usable']);caption=r['entry_id']
    d.text((18+c*cw,y+88),caption,font=font(16),fill='#b91c1c' if bad else '#475569')
    if bad:d.text((18+c*cw,y+106),('Recipe unvalidated; ' if r['recipe_unvalidated'] else '')+('render failed' if not(r['render_complete'] and r['render_usable']) else 'render passed'),font=font(13),fill='#b91c1c')
  name=f'page_{pg+1:02d}.png';sheet.save(OUT/name);pages.append(name)
 invalid=sum(r['recipe_unvalidated'] for r in rows);passed=sum(r['render_complete'] and r['render_usable'] for r in rows)
 receipt={'recipes':164,'reference_indices':list(range(41)),'samples_per_recipe':1,'rendered_images':sum('image' in r for r in rows),'passed_render_checks':passed,'unvalidated_recipes':invalid,'snapshot_sha256':hashlib.sha256(raw).hexdigest(),'source_recipes_sha256':hashlib.sha256((SOURCE/'refined_recipes.json').read_bytes()).hexdigest(),'parallel_preview_workers':2,'production_changed':False,'pages':pages,'records':rows}
 (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 parts=['<!doctype html><html><head><meta charset="utf-8"><title>CryptBB v3 — 164 recipes</title><style>body{font:16px system-ui;background:#f2f4f7;color:#182033;margin:24px}header{position:sticky;top:0;background:#f2f4f7ee;padding:12px 0;z-index:2}h1{margin:0 0 8px}p{margin:6px 0}input,select{padding:8px;margin:8px 8px 8px 0;font:inherit}.group{display:grid;grid-template-columns:repeat(5,minmax(210px,1fr));gap:10px;margin:12px 0;overflow-x:auto}.card{background:white;border-radius:8px;padding:12px}.card.bad{border:2px solid #b91c1c}.card img{display:block;width:100%;height:90px;object-fit:contain;image-rendering:auto}small{display:block;color:#475569;margin-top:7px}.bad small{color:#b91c1c}a{color:#1260ac}.hidden{display:none}</style></head><body><header><h1>CryptBB v3: 164 completed recipes</h1>',f'<p>41 references × 4 controllers. One fresh sample per recipe; {passed}/164 renders passed checks. {invalid} recipes were unvalidated.</p>','<p>These are preview renders, not the production A/B corpus. Failed cases are retained. Click an image for full size.</p><input id="search" placeholder="Reference index, e.g. 016"><label><input type="checkbox" id="failed">Show rows with failures only</label><p>Contact sheets: '+ ' · '.join(f'<a href="{p}">Page {i+1}</a>' for i,p in enumerate(pages))+'</p></header>']
 for i in range(41):
  groupbad=any(lookup[i,a]['recipe_unvalidated'] or not(lookup[i,a]['render_complete'] and lookup[i,a]['render_usable']) for a in ARMS)
  parts.append(f'<section class="group" data-index="{i:03d}" data-failed="{int(groupbad)}"><div class="card"><b>Reference {i:03d}</b><a href="references/{i:03d}.png"><img src="references/{i:03d}.png"></a><small>Real reference pixels</small></div>')
  for arm in ARMS:
   r=lookup[i,arm];bad=r['recipe_unvalidated'] or not(r['render_complete'] and r['render_usable']);parts.append('<div class="card'+(' bad' if bad else '')+'"><b>'+html.escape(r['entry_id'])+'</b>')
   if 'image' in r:parts.append(f'<a href="{r["image"]}"><img src="{r["image"]}"></a>')
   else:parts.append('<p>Render failed</p>')
   parts.append('<small>'+('Recipe unvalidated' if r['recipe_unvalidated'] else 'Recipe validated')+' · '+('render passed' if r['render_complete'] and r['render_usable'] else 'render failed')+'</small></div>')
  parts.append('</section>')
 parts.append('<script>function filter(){const q=document.querySelector("#search").value.trim(),bad=document.querySelector("#failed").checked;document.querySelectorAll(".group").forEach(x=>x.classList.toggle("hidden",!(x.dataset.index.includes(q)&&(!bad||x.dataset.failed==="1"))))}document.querySelector("#search").oninput=filter;document.querySelector("#failed").onchange=filter;</script></body></html>')
 (OUT/'gallery.html').write_text('\n'.join(parts),encoding='utf-8')
 print(json.dumps({k:v for k,v in receipt.items() if k not in ['records','reference_indices']},indent=2),flush=True)
if __name__=='__main__':main()
