import ast
from pathlib import Path
import numpy as np
from ckpt_select.simulate import generate
from ckpt_select.agreement import matrix,similarity
from ckpt_select.estimator import rank1,rank2
from ckpt_select.baselines import mode_scores,mode_credit
from ckpt_select.loader import PredictionTable

def test_independent():
 t,a,_=generate(1);f=rank1(matrix(t,'exact'),t.epochs)
 assert np.mean(np.abs(f['scores']-a))<.02
 assert np.argmax(f['scores'])==np.argmax(a)
def test_temporal_cluster():
 t,a,_=generate(2);A=matrix(t,'exact')
 assert np.argmax(rank1(A,t.epochs,0)['scores'])>=14
 assert np.argmax(rank1(A,t.epochs,5)['scores'])==np.argmax(a)
def test_trap():
 t,_,_=generate(3);A=matrix(t,'exact');f=rank1(A,t.epochs,3)
 assert rank2(A,t.epochs,3,f)['trap_component_flag']
def test_kernel_edges():
 assert similarity('','', 'edit')==1
 assert similarity('a','', 'char')==0
 assert similarity('ab','a','char')==.5
 assert similarity('ab','ba','char')==0
 assert similarity('Ab','ab','exact')==0
def test_ties_and_exclusion():
 assert mode_credit('a',['a','b'],True)==.5
 assert mode_credit('a',['a','b'],False)==1
 t=PredictionTable('test',np.arange(1,4),('x',),np.array([['a'],['a'],['b']],object))
 assert np.allclose(mode_scores(t,exclude='loo',fractional=True),[.5,.5,0])
 assert np.allclose(mode_scores(t,exclude='neighbour',k=0,fractional=True),[.5,.5,0])
def test_selection_modules_cannot_import_evaluation():
 root=Path(__file__).parents[1]/'ckpt_select'
 for name in ['loader','agreement','baselines','estimator']:
  source=(root/(name+'.py')).read_text();tree=ast.parse(source)
  for node in ast.walk(tree):
   if isinstance(node,ast.ImportFrom):assert 'evaluate' not in (node.module or '')
   if isinstance(node,ast.Import):assert all('evaluate' not in x.name for x in node.names)
  assert 'scoring_only' not in source and 'scoring_labels' not in source


def test_all_modules_parse():
 root=Path(__file__).parents[1]/"ckpt_select"
 for p in root.glob("*.py"):compile(p.read_text(),str(p),"exec")


def test_mcnemar_known_discordances():
 from ckpt_select.evaluate import paired_test
 selected=np.array([True]*3+[False]*8+[False]*240)
 best=np.array([False]*3+[True]*8+[False]*240)
 r=paired_test(selected,best)
 assert r['selected_only_correct']==3 and r['best_only_correct']==8
 assert r['mcnemar_p']==0.2265625
 assert r['bootstrap_samples']==10000
 assert r['bootstrap_ci_low']<0<r['bootstrap_ci_high']

def test_label_evaluation_rejects_tampered_freeze(tmp_path):
 import pytest,json
 from ckpt_select.evaluate import evaluate
 (tmp_path/'scores.json').write_text('[]')
 (tmp_path/'selection_freeze.json').write_text(json.dumps(dict(files={'scores.json':'invalid'},code_hashes={},labels_read=False)))
 with pytest.raises(AssertionError,match='Frozen selection artifact changed'):
  evaluate(tmp_path/'nonexistent_label_root',tmp_path,{})
