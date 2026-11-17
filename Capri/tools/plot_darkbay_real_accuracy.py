"""Plot recorded real DarkBay accuracy; no inference or training."""
from pathlib import Path
import json
import hashlib
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'handoff/darkbay_curriculum_20261113'
OUT=ROOT/'thesis/figures/darkbay/real_accuracy_epochs01_11'
OUT.mkdir(parents=True,exist_ok=True)
colors={1:'#159447',2:'#2463EB',3:'#DC3545'}
markers={1:'o',2:'s',3:'^'}
rows={}
sources=[]
for model in (1,2,3):
    path=STUDY/f'model{model}_epoch_accuracy.json'
    rows[model]=json.loads(path.read_text(encoding='utf8'))['rows'][:11]
    assert [r['epoch'] for r in rows[model]]==list(range(1,12))
    sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,
    'svg.fonttype':'none','savefig.facecolor':'white'})
plotted={}
for tag in ('A','B'):
    fig,ax=plt.subplots(figsize=(10,6.2))
    fig.subplots_adjust(left=.10,right=.97,bottom=.17,top=.76)
    fig.text(.10,.94,f'DarkBay — Dataset {tag}',fontsize=20,fontweight='bold',color='#172033')
    fig.text(.10,.891,'Real CAPTCHA accuracy through epoch 11',fontsize=13,color='#526075')
    plotted[tag]={}
    for model in (1,2,3):
        values=[]; points=[]
        for r in rows[model]:
            v=r[tag]
            if v is None:
                assert model==1 and r['epoch']==1
                values.append(math.nan); points.append({'epoch':1,'accuracy_percent':None,'correct':None})
                continue
            assert v['total']==900 and v['seed']==48
            value=100*v['correct']/900
            assert abs(value-v['accuracy_percent'])<1e-9
            values.append(value);points.append({'epoch':r['epoch'],'accuracy_percent':value,'correct':v['correct']})
        ax.plot(range(1,12),values,color=colors[model],marker=markers[model],markersize=5.8,
            linewidth=2.2,label=f'Model {model}',markeredgecolor='white',markeredgewidth=.6)
        plotted[tag][f'Model {model}']=points
    ax.set(xlim=(.75,11.25),ylim=(-2,100),xlabel='Epoch',ylabel='Real accuracy (%)')
    ax.set_xticks(range(1,12));ax.set_yticks(range(0,101,10))
    ax.grid(axis='y',color='#E1E6EE',linewidth=.8)
    ax.set_axisbelow(True)
    for side in ('top','right'): ax.spines[side].set_visible(False)
    for side in ('left','bottom'): ax.spines[side].set_color('#BBC4D0')
    ax.tick_params(colors='#42516A')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=3,frameon=False,borderaxespad=0,
        handlelength=2.6,columnspacing=2.2)
    fig.text(.10,.072,'Exact, case-sensitive matches on the same 900 real DarkBay images.',fontsize=10,color='#526075')
    fig.text(.10,.035,'Model 1: epoch 1 unavailable (checkpoint not retained). No averages or smoothing.',fontsize=9.5,color='#526075')
    for ext in ('png','svg'):
        fig.savefig(OUT/f'dataset_{tag.lower()}_real_accuracy_epochs01_11.{ext}',dpi=180)
    plt.close(fig)
(OUT/'plotted_data.json').write_text(json.dumps({'images':900,'epochs':11,'datasets':plotted},indent=2)+'\n',encoding='utf8')
(OUT/'provenance.json').write_text(json.dumps({'sources':sources,'plot_script':str(Path(__file__).relative_to(ROOT)),
    'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'colors':colors,'missing':'Model1 epoch1 unavailable, represented as a gap',
    'selection':'No training or checkpoint-selection changes'},indent=2)+'\n',encoding='utf8')
print(OUT)
