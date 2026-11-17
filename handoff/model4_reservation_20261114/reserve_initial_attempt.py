"""Freeze label-blind Model4 reserves; no pseudo-label inference or training."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evaluation_data/model4_reservation_20261114'
SPLITS=Path('E:/pranav/Final sem/CaptchaThesis/system5_clean_baseline/splits')
CITY=Path('E:/pranav/City (live)-20260503T045450Z-3-001/City (live)')
SALT='model4-reservation-20261114-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def pix(p):
 with Image.open(p) as im:
  im=im.convert('RGB');return hashlib.sha256(im.tobytes()).hexdigest(),im.copy()
def rel(p):return p.relative_to(ROOT).as_posix()
def rank(r):return hashlib.sha256((SALT+':'+r['pixel_sha256']).encode()).hexdigest()
if OUT.exists():raise RuntimeError('Refuse to overwrite an existing reservation')
inputs=[]; pools={}; labels={}; provenance={}; oldsplit={}
for family,manifest in [('darkbay','evaluation_data/darkbay/split_manifests/image_manifest.json'),('cryptbb','evaluation_data/cryptbb/image_manifest.json')]:
 mp=ROOT/manifest;lp=ROOT/f'evaluation_data/{family}/scoring_labels/scoring_labels.json'
 inputs += [{'path':str(p),'sha256':sha(p)} for p in [mp,lp]]
 rows=read(mp);lab=read(lp);pools[family]=[];labels[family]={};provenance[family]={}
 for r in rows:
  p=Path(r['image_path']);p=p if p.is_absolute() else ROOT/p
  h,im=pix(p);assert h==r['pixel_sha256'];oid=family+'_'+h
  pools[family].append(dict(opaque_id=oid,pixel_sha256=h,source=p))
  labels[family][oid]=lab[r['opaque_id']];provenance[family][oid]={'source':str(p),'previous_id':r['opaque_id']}
annotations={}
for split in ['dev','test']:
 p=SPLITS/split/'city.jsonl';inputs.append({'path':str(p),'sha256':sha(p)})
 for line in p.read_text(encoding='utf-8-sig').splitlines():
  r=json.loads(line);assert r['sha256'] not in annotations;annotations[r['sha256']]=dict(r,old_split=split)
pools['city']=[];labels['city']={};provenance['city']={};matched=set()
for p in sorted(CITY.glob('*.png')):
 fh=sha(p)
 if fh not in annotations:continue
 assert fh not in matched;matched.add(fh);r=annotations[fh];h,im=pix(p);oid='city_'+h
 pools['city'].append(dict(opaque_id=oid,pixel_sha256=h,source=p));labels['city'][oid]=r['label'];oldsplit[oid]=r['old_split']
 provenance['city'][oid]={'source':str(p),'source_file_sha256':fh,'previous_split':r['old_split'],'previous_image_id':r['image_id']}
assert len(matched)==999
summary={};all_hashes={};outputs=[]
for family,rows in pools.items():
 assert len({r['pixel_sha256'] for r in rows})==len(rows)
 chosen=sorted(rows,key=rank)[:300] if family!='city' else [r for r in rows if oldsplit[r['opaque_id']]=='dev']
 selected={r['opaque_id'] for r in chosen};assert len(selected)==300
 partition={'pseudo_label_pool':[],'evaluation_pool':[]};score={'pseudo_label_pool':{},'evaluation_pool':{}}
 for r in rows:
  kind='pseudo_label_pool' if r['opaque_id'] in selected else 'evaluation_pool'
  p=OUT/family/'images'/kind/(r['opaque_id']+'.png');p.parent.mkdir(parents=True,exist_ok=True)
  h,im=pix(r['source']);im.save(p);h2,_=pix(p);assert h==h2==r['pixel_sha256']
  partition[kind].append(dict(opaque_id=r['opaque_id'],image_path=rel(p),pixel_sha256=h,size=list(im.size)))
  score[kind][r['opaque_id']]=labels[family][r['opaque_id']]
  provenance[family][r['opaque_id']]['source_file_sha256']=sha(r['source'])
 for kind in partition:
  partition[kind].sort(key=lambda r:r['opaque_id']);dump(OUT/family/(kind+'.json'),partition[kind]);dump(OUT/'scoring_only'/family/(kind+'_labels.json'),score[kind])
 ps={r['pixel_sha256'] for r in partition['pseudo_label_pool']};ev={r['pixel_sha256'] for r in partition['evaluation_pool']};assert not ps&ev
 dump(OUT/'scoring_only'/family/'source_provenance.json',provenance[family])
 summary[family]={'model4_reserved':len(ps),'evaluation':len(ev),'historical_excluded':100 if family!='city' else 0,'eligible_images':len(rows),'pixel_overlap':0,'selection':'Existing dev split; no new label-based selection' if family=='city' else 'SHA256 ordering of fixed salt and pixel hash; no labels or predictions used'}
 all_hashes[family]=ps|ev
# Explicitly retain the historical excluded image identities, without repurposing them.
excluded={}
for family,folder in [('darkbay',Path('E:/pranav/DarkBay_Labeled-20260503T045615Z-3-001/DarkBay_Labeled')),('cryptbb',Path('E:/pranav/cryptbb_sourcefree/data/cryptbb_split/selftrain'))]:
 rr=[]
 for p in folder.rglob('*'):
  if p.suffix.lower() not in ['.png','.jpg','.jpeg']:continue
  h,_=pix(p)
  if h not in all_hashes[family]:rr.append({'pixel_sha256':h,'source':str(p)})
 assert len(rr)==100 and len({r['pixel_sha256'] for r in rr})==100
 excluded[family]=rr
dump(OUT/'scoring_only/historical_exclusions.json',excluded)
protocol={'created_at':datetime.now(timezone.utc).isoformat(),'user_request':'Reserve 300 images from each dataset for later pseudo-labeling in Model4','status':'reserved_only_no_pseudo_labels_or_training','selection_salt':SALT,'counts':summary,'input_manifests':inputs,'script_sha256':sha(Path(__file__)),'rules':['All model-facing paths are opaque and contain no ground-truth labels.','scoring_only is forbidden as a Model4 training input; pseudo labels must come from the future frozen teacher procedure.','Reserved images must not enter future evaluation pools or new controller fitting; exclude by pixel hash.','Use the same remaining evaluation pool for future cross-model comparisons; historical full-pool metrics remain historical.','Teacher, confidence/filtering policy, training budget and launch are not authorized or decided by this reservation.','Historical exposure remains; these are not newly untouched test sets.','Old manifests, datasets and stopped jobs are unchanged.']}
dump(OUT/'protocol.json',protocol)
for p in sorted(OUT.rglob('*.json')):outputs.append({'path':rel(p),'sha256':sha(p)})
dump(OUT/'verification.json',{'passed':True,'all_images_decoded_and_pixel_verified':True,'opaque_paths':True,'pseudo_evaluation_pixel_overlap':0,'counts':summary,'manifest_hashes':outputs})
print(json.dumps(summary,indent=2),flush=True)
