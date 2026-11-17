"""Show frozen epoch rankings only; never read labels or accuracy estimates."""
import argparse
from pathlib import Path
import numpy as np
from .loader import read,sha,write

def ranking_only(out):
 out=Path(out);freeze=read(out/'selection_freeze.json')
 assert not freeze['labels_read']
 for name,digest in freeze['files'].items():assert sha(out/name)==digest
 records=[]
 for r in read(out/'scores.json'):
  # Epoch grids are frozen and consecutive from1 for both requested runs.
  order=np.argsort(-np.asarray(r['scores']),kind='stable')
  records.append(dict(run=r['run'],method=r['method'],kernel=r['kernel'],k=r['k'],selected_epoch=r['selected_epoch'],top3_epochs=(order[:3]+1).tolist(),resolved_k=r.get('resolved_k'),mask_criterion_passed=r.get('k_criterion_passed')))
 return dict(purpose='Predict epoch ranking only; no absolute accuracy or domain-gap estimation',selection_freeze_sha256=sha(out/'selection_freeze.json'),rankings=records)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path('results/cryptbb_A20_B16_verified'));a=ap.parse_args()
 result=ranking_only(a.results);write(a.results/'ranking_only.json',result)
 import json
 print(json.dumps([r for r in result['rankings'] if r['method']=='rank1_auto' and r['kernel']=='exact'],indent=2))
