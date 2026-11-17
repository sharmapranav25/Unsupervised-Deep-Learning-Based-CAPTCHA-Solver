"""End-to-end evaluation and exact count-checkpoint resume checks."""
import copy,torch
from pathlib import Path
from .architecture_training import setup,main_model,main_eval,count_eval,optimize,update_ema
from .architecture_training_data import Scenes,References,Counts
from .count_transformer import CountTransformer
from .decoder.checkpoints import save_epoch,load_resume
from .decoder.common import write

def run(dataset,out,checkpoint):
    out=Path(out);setup(out)
    counts=Counts(dataset,'dev');counts.rows=counts.rows[:16]
    optimization=Counts(dataset,'train');optimization.rows=optimization.rows[:1]
    model=CountTransformer().cuda();opt,sch,scaler=optimize(model,10,'count')
    x,v,y,_=optimization[0];x=x[None].cuda();v=v[None].cuda();y=y[None].cuda()
    def step():
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x,v),y);loss.backward();opt.step();sch.step();return float(loss)
    step();ema=copy.deepcopy(model.state_dict())
    cp=save_epoch(out/'resume',1,model,ema,opt,sch,None,dict(epoch=1,consumed_batches=1),'check',{})
    before=step();expected=copy.deepcopy(model.state_dict())
    load_resume(cp,model,opt,sch,None,'check');after=step()
    assert before==after
    assert all(torch.equal(expected[k],v) for k,v in model.state_dict().items())
    count_eval(model,counts,out/'count_eval_smoke.json');del model,opt,sch,scaler,ema,expected;torch.cuda.empty_cache()
    model=main_model(checkpoint);dev=Scenes(dataset,'dev');dev.rows=dev.rows[:8]
    refs=References(dataset);refs.rows=refs.rows[:8];devrefs=References(dataset,'dev');devrefs.rows=devrefs.rows[:8]
    result=main_eval(model,dev,refs,devrefs,out/'main_eval_smoke.json')
    assert result['images']==8
    write(out/'checks.json',dict(passed=True,exact_count_resume=True,count_evaluation_images=16,main_evaluation_images=8,meaning='Smoke checks only, no quality estimate; production uses all100k TRAIN and2000 DEV.'))
    for ds in (optimization,counts,dev,refs,devrefs):ds.close()
