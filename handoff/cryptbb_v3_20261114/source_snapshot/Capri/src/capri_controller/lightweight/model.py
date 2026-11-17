"""Original 3.66M CAPRI core, full DINO fine-tuning for fonts, separate count transformer."""
import importlib,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .legacy_model import Iter2Model
from ..count_transformer import CountTransformer
from ..decoder.common import ROOT,read,sha

class LightweightCapri(nn.Module):
    def __init__(self):
        super().__init__();cfg=read(ROOT/'configs/local.json');source=Path(cfg['decoder_dinov2_source'])
        if str(source) not in sys.path:sys.path.insert(0,str(source))
        backbones=importlib.import_module('dinov2.hub.backbones')
        assert Path(backbones.__file__).resolve().is_relative_to(source.resolve())
        self.capri=Iter2Model();assert sum(p.numel() for p in self.capri.parameters())==3662050
        self.dino=backbones.dinov2_vits14_reg(pretrained=False)
        self.dino.load_state_dict(torch.load(cfg['decoder_dinov2_checkpoint'],map_location='cpu',weights_only=True))
        self.pretrained_sha256=sha(cfg['decoder_dinov2_checkpoint'])
        self.font_embedding=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,128))
        self.font_outputs=nn.Linear(768,49) # category5,stroke3,4x10 ordinal,italic1
        self.counter=CountTransformer()
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225])[None,:,None,None])
        self.requires_grad_(True)
    def train(self,mode=True):
        super().train(mode);self.dino.eval();return self
    def font_features(self,image,widths):
        # Fixed pixel-only canvas avoids changing DINO positions with batch partners.
        x=image[:,:,:96,:480];x=F.pad(x,(0,490-x.shape[-1],0,2),value=1.)
        valid=(torch.arange(490,device=x.device)[None,:]<widths[:,None])[:,None,None,:].expand(-1,1,98,-1)
        valid=valid&(torch.arange(98,device=x.device)[None,None,:,None]<96)
        x=torch.where(valid,x,torch.ones_like(x))
        f=self.dino.forward_features((x-self.rgb_mean)/self.rgb_std)
        coverage=F.avg_pool2d(valid.float(),14).flatten(1)
        pooled=(f['x_norm_patchtokens']*coverage[:,:,None]).sum(1)/coverage.sum(1,keepdim=True)
        features=torch.cat((f['x_norm_clstoken'],pooled),-1)
        return F.normalize(self.font_embedding(features).float(),dim=-1),self.font_outputs(features)
    def forward(self,image,cols,widths):
        out=self.capri(image,cols);embedding,font=self.font_features(image,widths)
        out['font_cat']=out['font_cat']+font[:,:5];out['font_stroke']=out['font_stroke']+font[:,5:8]
        out['font_ordinal']=out['font_ordinal']+font[:,8:48].reshape(-1,4,10);out['font_italic']=out['font_italic']+font[:,48]
        # Same cached RGB, independent count-only architecture; labels never enter it.
        x=F.interpolate(image[:,:,:96,:480],size=(64,320),mode='bilinear',align_corners=False,antialias=True) if image.shape[-1]>=480 else F.interpolate(F.pad(image,(0,480-image.shape[-1],0,0),mode='replicate'),size=(64,320),mode='bilinear',align_corners=False,antialias=True)
        x=F.pad(x,(0,64),value=1.)
        count_width=(widths.float()*2/3).round().long()
        mask=(torch.arange(384,device=x.device)[None,:]<count_width[:,None])[:,None,:].expand(-1,64,-1)
        out['independent_count']=self.counter(x,mask);out['font_embedding']=embedding
        return out
    def resume_configuration(self):
        return dict(architecture='original-iter2-plus-finetuned-dino-font-plus-separate-count-v1',core_params=3662050,dino='dinov2_vits14_reg',dino_trainable_blocks=12,pretrained_sha256=self.pretrained_sha256,font_canvas=[98,490],count=self.counter.resume_configuration())
