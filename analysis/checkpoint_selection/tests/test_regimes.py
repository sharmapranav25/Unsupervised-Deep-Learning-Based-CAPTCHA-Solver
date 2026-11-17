import ast
from pathlib import Path
import numpy as np
import pytest
from ckpt_select.regimes import k_cap,fit_mask,select_mask,item_agreements,bootstrap_factors,joint_coupling,neighbour_diagnostics,combine,degeneracy,decomposition
from ckpt_select.loader import PredictionTable
from ckpt_select.simulate import generate

def test_caps_and_support():
 assert k_cap(20)==4 and k_cap(15)==3
 a=np.linspace(.2,.9,20);A=np.outer(a,a);np.fill_diagonal(A,1);e=np.arange(1,21)
 k,passed,fits,criteria,_=select_mask(A,e)
 assert set(fits)==set(range(5)) and k==0 and passed
 assert fits[4]['partners'].sum(1).min()==11
 assert np.max(np.abs(fits[4]['scores']-a))<1e-7

def test_excluded_epochs_cannot_steer_fit():
 e=np.arange(1,11);a=np.linspace(.2,.8,10);A=np.outer(a,a);np.fill_diagonal(A,1)
 b=fit_mask(A,e,2,[0,1]);alter=A.copy();alter[:2,:]=.99;alter[:,:2]=.99;c=fit_mask(alter,e,2,[0,1])
 assert np.allclose(b['scores'][2:],c['scores'][2:])
 assert not c['partners'][:,:2].any()

def test_joint_recovers_known_decay_model():
 e=np.arange(1,21);a=.35+.35*np.sin(np.linspace(0,np.pi,20));a[8]=.85;d=np.abs(e[:,None]-e[None,:]);A=np.outer(a,a)+.6*np.exp(-d/2.5)*np.outer(1-a,1-a);np.fill_diagonal(A,1)
 f=joint_coupling(A,e,fit_mask(A,e,0)['scores'])
 assert np.max(np.abs(f['scores']-a))<1e-5
 assert abs(f['kappa0']-.6)<1e-4 and abs(f['tau']-2.5)<1e-4
 assert f['selected_epoch']==9

def test_vectorized_bootstrap_matches_individual_refits():
 table,_,_=generate(1,m=100);items=item_agreements(table,'exact');A=items.mean(2);f=fit_mask(A,table.epochs,2,[0]);b=bootstrap_factors(items,table.epochs,2,f,samples=8,seed=19)
 w=np.random.default_rng(19).multinomial(100,np.full(100,.01),size=8)/100
 for i in range(8):
  expected=fit_mask(np.einsum('m,ijm->ij',w[i],items),table.epochs,2,[0])['scores']
  assert np.max(np.abs(expected-b['samples'][i]))<1e-5
 assert np.array_equal(b['samples'],bootstrap_factors(items,table.epochs,2,f,samples=8,seed=19)['samples'])

def test_neighbour_kappa_and_flip():
 e=np.arange(1,9);a=np.linspace(.3,.6,8);d=np.abs(e[:,None]-e[None,:]);A=np.outer(a,a)+.4*np.exp(-d/2)*np.outer(1-a,1-a);np.fill_diagonal(A,1)
 f=dict(scores=a,active=np.ones(8,bool));z=neighbour_diagnostics(A,e,3,f)
 assert np.allclose([p['kappa'] for p in z['pairs']],[.4*np.exp(-p['distance']/2) for p in z['pairs']])
 assert abs(z['decay']['all']['kappa0']-.4)<1e-5
 assert abs(z['flip_rates'][0]['disagreement']-(1-A[0,1]))<1e-12

def test_candidates_plateau_and_tie_break():
 e=np.arange(1,5);f=dict(scores=np.array([.2,.8,.8,.2]),supported=np.ones(4,bool));b=dict(low=np.array([.1,.7,.7,.1]),high=np.array([.3,.9,.9,.3]));near=dict(mean_kappa=[0,.4,.2,0])
 s=combine(e,1,f,b,near);assert s['candidates']==[2,3] and s['combined_pick']==3

def test_burnin_and_empty_neighbour_pool():
 p=np.array([['']*4,['x']*4,['a','b','c','d'],['e','f','g','h']],object);t=PredictionTable('test',np.arange(1,5),tuple('abcd'),p)
 flags=degeneracy(t,np.array([.1,.1,.1,.001]));assert [x['flagged'] for x in flags]==[True,True,False,True]
 A=item_agreements(t,'exact').mean(2);f=fit_mask(A,t.epochs,0);d=decomposition(t,A,t.epochs,0,f)
 assert all(r['score'] is None for r in d['rows'] if r['region']=='neighbour')

def test_v2_selection_has_no_eval_dependency():
 source=(Path(__file__).parents[1]/'ckpt_select/regimes.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
 assert 'scoring_only' not in source
 for node in ast.walk(tree):
  if isinstance(node,ast.ImportFrom):assert 'evaluate' not in (node.module or '')

def test_v2_rejects_tampered_freeze_before_labels(tmp_path):
 import json
 from ckpt_select.evaluate import evaluate_regimes
 (tmp_path/'selection_freeze.json').write_text(json.dumps(dict(labels_read=False,files={'scores.json':'bad'},code_hashes={})))
 (tmp_path/'scores.json').write_text('[]')
 with pytest.raises(AssertionError,match='Frozen selection artifact changed'):evaluate_regimes(tmp_path/'no_labels',tmp_path,{})
