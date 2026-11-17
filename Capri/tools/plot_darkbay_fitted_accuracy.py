"""Cubic least-squares visual summaries of recorded real accuracy."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'handoff/darkbay_curriculum_20261113'
OUT=ROOT/'thesis/figures/darkbay/real_accuracy_epochs01_11_cubic'
OUT.mkdir(parents=True,exist_ok=True)
colors={1:'#159447',2:'#2463EB',3:'#DC3545'}
markers={1:'o',2:'s',3:'^'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
rows={};sources=[]
for model in (1,2,3):
 p=STUDY/f'model{model}_epoch_accuracy.json'
 rows[model]=json.loads(p.read_text(encoding='utf8'))['rows'][:11]
 sources.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
results={};equations=['Cubic least-squares fits to real DarkBay accuracy through epoch11.',
 'e = epoch; y = fitted accuracy in percent. Fit raw values, not moving averages.',
 'Same degree3 for all runs. Dataset A and B separate; no extrapolation.', '']
for tag in ('A','B'):
 fig,ax=plt.subplots(figsize=(13.2,6.5))
 fig.subplots_adjust(left=.075,right=.69,bottom=.17,top=.76)
 fig.text(.075,.94,f'DarkBay — Dataset {tag}',fontsize=20,fontweight='bold',color='#172033')
 fig.text(.075,.89,'Real accuracy · cubic least-squares fits through epoch 11',fontsize=13,color='#526075')
 fig.text(.735,.76,'Fitted equations',fontsize=13,fontweight='bold',color='#172033')
 fig.text(.735,.718,'e = epoch; y = accuracy (%)',fontsize=10,color='#526075')
 results[tag]={}
 for model in (1,2,3):
  points=[(r['epoch'],100*r[tag]['correct']/900) for r in rows[model] if r[tag] is not None]
  x,y=np.array(points,dtype=float).T
  coeff=np.polyfit(x,y,3)
  fitted=np.polyval(coeff,x)
  rmse=float(np.sqrt(np.mean((fitted-y)**2)))
  grid=np.linspace(x.min(),x.max(),400)
  ax.scatter(x,y,color=colors[model],marker=markers[model],s=32,alpha=.38,zorder=3)
  ax.plot(grid,np.polyval(coeff,grid),color=colors[model],linewidth=2.4,label=f'Model {model}')
  a,b,c,d=coeff
  equation=f'y = {a:.6f}*e^3 {b:+.6f}*e^2 {c:+.6f}*e {d:+.6f}'
  ybase=.64-(model-1)*.185
  fig.text(.735,ybase,f'Model {model}',color=colors[model],fontsize=12,fontweight='bold')
  fig.text(.735,ybase-.047,rf'$y = {a:.4f}e^3 {b:+.4f}e^2$',fontsize=11,color='#172033')
  fig.text(.755,ybase-.091,rf'${c:+.4f}e {d:+.4f}$',fontsize=11,color='#172033')
  fig.text(.735,ybase-.127,f'Fit RMSE: {rmse:.2f} percentage points',fontsize=9,color='#526075')
  results[tag][f'Model {model}']={'coefficients_descending':coeff.tolist(),'equation':equation,
      'epoch_domain':[int(x.min()),int(x.max())],'rmse_percentage_points':rmse,
      'measured_points':[{'epoch':int(e),'accuracy_percent':float(v)} for e,v in points]}
  equations += [f'Dataset {tag}, Model {model}: {equation}',f'  Domain: {int(x.min())} <= e <= {int(x.max())}; RMSE {rmse:.6f} percentage points.']
 ax.set(xlim=(.75,11.25),ylim=(-2,100),xlabel='Epoch',ylabel='Real accuracy (%)')
 ax.set_xticks(range(1,12));ax.set_yticks(range(0,101,10));ax.grid(axis='y',color='#E1E6EE',linewidth=.8);ax.set_axisbelow(True)
 for side in ('top','right'):ax.spines[side].set_visible(False)
 for side in ('left','bottom'):ax.spines[side].set_color('#BBC4D0')
 ax.tick_params(colors='#42516A')
 ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=3,frameon=False,borderaxespad=0)
 fig.text(.075,.072,'Lines: fitted equations. Faint points: measured exact-match accuracy on the same 900 real images.',fontsize=10,color='#526075')
 fig.text(.075,.035,'Descriptive fits only. Model 1 starts at epoch 2 because its epoch-1 checkpoint is unavailable.',fontsize=9.5,color='#526075')
 for ext in ('png','svg'):fig.savefig(OUT/f'dataset_{tag.lower()}_cubic_fit_epochs01_11.{ext}',dpi=180,facecolor='white')
 plt.close(fig)
(OUT/'equations.txt').write_text('\n'.join(equations)+'\n',encoding='utf8')
(OUT/'fit_data.json').write_text(json.dumps({'method':'Unweighted ordinary least squares, cubic polynomial, identical degree across all6 runs','sources':sources,'results':results,
 'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf8')
print('\n'.join(equations))
