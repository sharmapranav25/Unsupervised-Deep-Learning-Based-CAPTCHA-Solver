from pathlib import Path
import json,hashlib,io,zipfile
from PIL import Image,ImageDraw,ImageFont,ImageOps
root=Path.cwd();old=root/'datasets/cryptbb/A';new=root/'Capri/runs/cryptbb_v3_live_preview_20261114';out=root/'Capri/runs/cryptbb_old100k_vs_v3_20261114';out.mkdir(exist_ok=False)
oldrefs=json.loads((root/'Capri/runs/cryptbb_model1_preparation_20261113/generator/inputs.json').read_text());newrefs=json.loads((root/'Capri/runs/cryptbb_v3_source_20261114/inputs.json').read_text());newrows=json.loads((new/'receipt.json').read_text())['records'];summary={r['entry_id']:r for r in json.loads((old/'corpus_summary.json').read_text())}
indices=[0,8,16];arms=['custom','dino','synthtiger','mixed'];records=[];F=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
sheet=Image.new('RGB',(1700,1400),'#f2f4f7');d=ImageDraw.Draw(sheet)
d.text((20,14),'Earlier CryptBB 100k corpus vs current v3 preview',font=F(27),fill='#182033')
d.text((20,54),'Old: actual saved dataset-A images. New: fresh v3 renders. Reference sets differ, so this is not a controlled comparison.',font=F(17),fill='#475569')
for c,h in enumerate(['Reference used','Custom','DINO','SynthTIGER','Mixed']):d.text((20+330*c,90),h,font=F(21),fill='#182033')
for group,i in enumerate(indices):
 for version in ['old','new']:
  y=125+group*420+(0 if version=='old' else 198)
  for c in range(5):d.rounded_rectangle((14+c*330,y,14+(c+1)*330-8,y+188),radius=8,fill='white' if version=='old' else '#eef6ff')
  rp=Path((oldrefs if version=='old' else newrefs)[i]['source']);ref=Image.open(rp).convert('RGB');tile=ImageOps.contain(ref,(310,100),Image.Resampling.NEAREST);sheet.paste(tile,(20+(310-tile.width)//2,y+12));d.text((24,y+130),f'{"OLD 100k" if version=="old" else "NEW v3"} — reference {i:03d}',font=F(19),fill='#182033')
  for col,arm in enumerate(arms,1):
   rr=[]
   if version=='old':
    if arm=='synthtiger':d.text((24+col*330,y+64),'Not in old training corpus',font=F(18),fill='#64748b');continue
    shard=old/'corpus'/f'{arm}_{i:03d}.zip';digest=hashlib.sha256(shard.read_bytes()).hexdigest();assert digest==summary[f'{arm}_{i:03d}']['zip_sha256']
    with zipfile.ZipFile(shard) as z:
     labels=[json.loads(line) for line in z.read('labels.jsonl').decode().splitlines()]
     for j,r in enumerate(labels[:2]):
      im=Image.open(io.BytesIO(z.read(r['image']))).convert('RGB');h=hashlib.sha256(im.tobytes()).hexdigest();assert h==r['pixel_sha256'];dest=out/f'old_{i:03d}_{arm}_{j}.png';im.save(dest);rr.append((im,True,False));records.append({'version':'old','input_index':i,'controller':arm,'source_shard':str(shard),'source_image':r['image'],'pixel_sha256':h,'image':dest.name})
   else:
    for r in [r for r in newrows if r['reference_index']==i and r['controller']==arm]:
     im=Image.open(new/r['image']).convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==r['pixel_sha256'];rr.append((im,r['complete'] and r['usable'],r['unvalidated_entry']));records.append(dict(version='new',**r))
   for j,(im,ok,bad) in enumerate(rr):
    tile=ImageOps.contain(im,(310,64),Image.Resampling.NEAREST);sheet.paste(tile,(20+col*330+(310-tile.width)//2,y+10+j*70))
   caption='Actual saved training samples' if version=='old' else f'{sum(r[1] for r in rr)}/2 pass render checks'
   color='#b91c1c' if any(not r[1] or r[2] for r in rr) else '#475569';d.text((24+col*330,y+151),caption,font=F(16),fill=color)
   if any(r[2] for r in rr):d.text((24+col*330,y+169),'Recipe failed validation',font=F(14),fill='#b91c1c')
sheet.save(out/'comparison.png');(out/'receipt.json').write_text(json.dumps({'indices':indices,'old_samples':'First two saved samples from each indexed shard; verified zip and RGB hashes','new_samples':'Same fresh preview samples already shown','different_reference_sets':True,'not_accuracy_evidence':True,'records':records},indent=2)+'\n');print(str(out/'comparison.png'))
