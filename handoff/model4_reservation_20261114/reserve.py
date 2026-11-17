"""Reserve 100 pixel-only references + 300 Model4 images per family."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evaluation_data/model4_reservation_20261114'
SPLITS=Path('E:/pranav/Final sem/CaptchaThesis/system5_clean_baseline/splits');CITY=Path('E:/pranav/City (live)-20260503T045450Z-3-001/City (live)')
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
if OUT.exists():raise RuntimeError('Refuse to overwrite reservation')
inputs=[];pools={};labels={};provenance={};oldsplit={};references={}
for family,manifest,extra in [('darkbay','evaluation_data/darkbay/split_manifests/image_manifest.json','E:/pranav/DarkBay_Labeled-20260503T045615Z-3-001/DarkBay_Labeled'),('cryptbb','evaluation_data/cryptbb/image_manifest.json','E:/pranav/cryptbb_sourcefree/data/cryptbb_split/selftrain')]:
 mp=ROOT/manifest;lp=ROOT/f'evaluation_data/{family}/scoring_labels/scoring_labels.json';inputs += [{'path':str(p),'sha256':sha(p)} for p in [mp,lp]]
 rows=read(mp);lab=read(lp);pools[family]=[];labels[family]={};provenance[family]={};references[family]=set();seen=set()
 for r in rows:
  p=Path(r['image_path']);p=p if p.is_absolute() else ROOT/p;h,im=pix(p);assert h==r['pixel_sha256'];oid=family+'_'+h;seen.add(h)
  pools[family].append(dict(opaque_id=oid,pixel_sha256=h,source=p));labels[family][oid]=lab[r['opaque_id']];provenance[family][oid]={'source':str(p),'previous_id':r['opaque_id']}
 for p in sorted(Path(extra).rglob('*')):
  if p.suffix.lower() not in ['.png','.jpg','.jpeg']:continue
  h,im=pix(p)
  if h in seen:continue
  seen.add(h);oid=family+'_'+h;references[family].add(oid);pools[family].append(dict(opaque_id=oid,pixel_sha256=h,source=p));labels[family][oid]=p.stem
  provenance[family][oid]={'source':str(p),'previous_role':'controller_reference' if family=='darkbay' else 'historical_selftrain','label_origin':'Historical filename annotation; isolated from model-facing manifest'}
 assert len(references[family])==100
annotations={}
for split in ['dev','test']:
 p=SPLITS/split/'city.jsonl';inputs.append({'path':str(p),'sha256':sha(p)})
 for line in p.read_text(encoding='utf-8-sig').splitlines():
  r=json.loads(line);assert r['sha256'] not in annotations;annotations[r['sha256']]=dict(r,old_split=split)
pools['city']=[];labels['city']={};provenance['city']={};matched=set();duplicate_city_files=0
for p in sorted(CITY.glob('*.png')):
 fh=sha(p)
 if fh not in annotations:continue
 if fh in matched:duplicate_city_files+=1;continue
 matched.add(fh);r=annotations[fh];h,im=pix(p);oid='city_'+h
 pools['city'].append(dict(opaque_id=oid,pixel_sha256=h,source=p));labels['city'][oid]=r['label'];oldsplit[oid]=r['old_split']
 provenance['city'][oid]={'source':str(p),'source_file_sha256':fh,'previous_split':r['old_split'],'previous_image_id':r['image_id']}
assert len(matched)==999
references['city']={r['opaque_id'] for r in sorted(pools['city'],key=rank) if oldsplit[r['opaque_id']]=='test'}
references['city']={r['opaque_id'] for r in sorted([r for r in pools['city'] if r['opaque_id'] in references['city']],key=rank)[:100]}
summary={};outputs=[]
for family,rows in pools.items():
 assert len({r['pixel_sha256'] for r in rows})==len(rows)
 available=[r for r in rows if r['opaque_id'] not in references[family]]
 chosen=sorted(available,key=rank)[:300] if family!='city' else [r for r in available if oldsplit[r['opaque_id']]=='dev']
 selected={r['opaque_id'] for r in chosen};assert len(selected)==300
 partition={k:[] for k in ['reference_pool','pseudo_label_pool','evaluation_pool']};score={k:{} for k in partition}
 for r in rows:
  oid=r['opaque_id'];kind='reference_pool' if oid in references[family] else 'pseudo_label_pool' if oid in selected else 'evaluation_pool'
  p=OUT/family/'images'/kind/(oid+'.png');p.parent.mkdir(parents=True,exist_ok=True)
  h,im=pix(r['source']);im.save(p);h2,_=pix(p);assert h==h2==r['pixel_sha256']
  partition[kind].append(dict(opaque_id=oid,image_path=rel(p),pixel_sha256=h,size=list(im.size)));score[kind][oid]=labels[family][oid];provenance[family][oid]['source_file_sha256']=sha(r['source'])
 for kind in partition:
  partition[kind].sort(key=lambda r:r['opaque_id']);dump(OUT/family/(kind+'.json'),partition[kind]);dump(OUT/'scoring_only'/family/(kind+'_labels.json'),score[kind])
 sets=[{r['pixel_sha256'] for r in rr} for rr in partition.values()];assert sum(map(len,sets))==len(set.union(*sets))
 dump(OUT/'scoring_only'/family/'source_provenance.json',provenance[family])
 summary[family]={'reference_images':len(partition['reference_pool']),'model4_reserved':len(partition['pseudo_label_pool']),'evaluation':len(partition['evaluation_pool']),'total_unique_images':len(rows),'cross_split_pixel_overlap':0}
 assert summary[family]['reference_images']==100 and summary[family]['evaluation']=={'darkbay':600,'cryptbb':251,'city':599}[family]
protocol={'created_at':datetime.now(timezone.utc).isoformat(),'user_request':'Reserve 300 images each for Model4, and 100 each for references','status':'splits_frozen_no_pseudo_labels_or_training','selection_salt':SALT,'counts':summary,'duplicate_city_source_files_ignored':duplicate_city_files,'input_manifests':inputs,'script_sha256':sha(Path(__file__)),'selection':{'darkbay':'Reuse historical 100 controller references; hash-order remaining 900 for 300 Model4.','cryptbb':'Use historical selftrain100 as reference pixels; hash-order historical test551 for 300 Model4. This replaces the old distinct unlabeled100 reference set for the next run.','city':'Reuse dev300 as Model4; hash-order test699 for 100 reference and 599 evaluation.'},'rules':['No labels or predictions used in new hash-order selection. Existing City dev split is historical, not newly randomized.','Model-facing paths are opaque and contain no ground-truth labels.','scoring_only is forbidden as a generation or Model4 training input; pseudo labels must come from a future frozen teacher procedure.','Only reference_pool is authorized for new controller fitting. Exclude pseudo_label_pool and evaluation_pool by pixel hash.','Use the same remaining evaluation pool for future cross-model comparisons; historical full-pool metrics stay historical.','Teacher, filtering policy and Model4 training budget remain undecided.','Historical exposure remains; these are not newly untouched tests.','Old manifests, datasets and stopped jobs remain unchanged.']}
dump(OUT/'protocol.json',protocol)
for p in sorted(OUT.rglob('*.json')):outputs.append({'path':rel(p),'sha256':sha(p)})
dump(OUT/'verification.json',{'passed':True,'all_images_decoded_and_pixel_verified':True,'opaque_paths':True,'cross_split_pixel_overlap':0,'counts':summary,'manifest_hashes':outputs})
print(json.dumps(summary,indent=2),flush=True)
