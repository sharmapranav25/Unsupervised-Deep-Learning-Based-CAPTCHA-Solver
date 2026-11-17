import sys,contextlib
from pathlib import Path
sys.path.insert(0,'E:/pranav/Thesis-Pranav-Final/Capri/src')
from capri_controller.training import cryptbb_feature_v1 as M
import numpy as np,torch
sys.addaudithook(M.no_labels)
M.C.bootstrap();M.C.gpu_slot=contextlib.nullcontext
with M.gpu_owner():
    ds=M.dataset();sm,truth=M.load_features();cfg=M.config(M.STUDY/'checks')
    model,decoder=M.C.make_model(cfg);opt,sched,scaler=M.C.optimizer_for(model,cfg)
    draws,_=sm.draw();M.C.step(model,opt,sched,scaler,M.C.batch_at(ds,draws[:128]))
    results={}
    for name in ('early','mature'):
        if name=='mature':
            ck=torch.load(M.ROOT/'model1/runs/cryptbb_v3/B/epoch_20.pth',map_location='cpu',weights_only=False)
            model.load_state_dict(ck['state_dict'],strict=True);del ck
        indices=np.arange(512) if name=='early' else np.linspace(0,99999,512,dtype=np.int64)
        before=M.C.model_hash(model);rng=M.to_cpu(M.C.rng_state(model))
        a=M.predict(model,decoder,ds,indices,256);b=M.predict(model,decoder,ds,indices,64);again=M.predict(model,decoder,ds,indices,256)
        M.exact(rng,M.C.rng_state(model))
        results[name]=dict(weights_unchanged=before==M.C.model_hash(model),fixed_batch_repeated_equal=a==again,
            different_strings=sum(x!=y for x,y in zip(a,b)),correct256=int((np.asarray(a)==truth[indices]).sum()),
            correct64=int((np.asarray(b)==truth[indices]).sum()),examples=[dict(index=int(i+1),batch256=x,batch64=y) for i,x,y in zip(indices,a,b) if x!=y][:10])
    ds.env.close()
M.write(M.STUDY/'inference_batch_diagnosis.json',results)
print(results)
