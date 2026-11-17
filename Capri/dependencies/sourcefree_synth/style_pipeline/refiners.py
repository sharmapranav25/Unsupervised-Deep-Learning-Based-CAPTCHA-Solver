"""Small, from-scratch PyTorch refiners; no pretrained encoder or VLM.

The first experiment protects synthetic text exactly and only learns background
and interference shifts. This deliberately cannot repair a wrong glyph font.
"""
import torch
from torch import nn
from torch.nn import functional as F


def condition(x, family, count=5):
    onehot=F.one_hot(family,count).to(x.dtype)[:,:,None,None]
    return torch.cat((x,onehot.expand(-1,-1,x.shape[2],x.shape[3])),1)


class CNNRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(9,32,3,padding=1),nn.SiLU(),
            nn.Conv2d(32,32,3,padding=1),nn.SiLU(),
            nn.Conv2d(32,32,3,padding=2,dilation=2),nn.SiLU(),nn.Conv2d(32,3,3,padding=1))
        nn.init.zeros_(self.body[-1].weight);nn.init.zeros_(self.body[-1].bias)

    def forward(self,x,mask,family):
        residual=.30*torch.tanh(self.body(condition(torch.cat((x,mask),1),family)))
        return torch.where(mask.bool(),x,(x+residual).clamp(0,1))


class TransformerRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Conv2d(9,48,8,stride=8)
        self.position=nn.Parameter(torch.zeros(1,8*24,48))
        layer=nn.TransformerEncoderLayer(48,4,128,dropout=0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.decode=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,3,3,padding=1))
        nn.init.zeros_(self.decode[-1].weight);nn.init.zeros_(self.decode[-1].bias)

    def forward(self,x,mask,family):
        z=self.embed(condition(torch.cat((x,mask),1),family));b,c,h,w=z.shape
        z=self.encoder(z.flatten(2).transpose(1,2)+self.position[:,:h*w])
        z=z.transpose(1,2).reshape(b,c,h,w)
        residual=.30*torch.tanh(self.decode(F.interpolate(z,size=x.shape[2:],mode='bilinear',align_corners=False)))
        return torch.where(mask.bool(),x,(x+residual).clamp(0,1))


class PatchCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(8,32,4,2,1),nn.LeakyReLU(.2),
            nn.Conv2d(32,64,4,2,1),nn.LeakyReLU(.2),nn.Conv2d(64,64,3,1,1),
            nn.LeakyReLU(.2),nn.Conv2d(64,1,3,1,1))

    def forward(self,x,family):return self.body(condition(x,family))


def fixed_features(x):
    """Differentiable low-level features, independent of Phase 1 CV proposals."""
    mean=x.mean((2,3));second=(x*x).mean((2,3));parts=[mean,second]
    for size in (1,2,4):
        y=F.avg_pool2d(x,size) if size>1 else x
        parts.extend(((y[:,:,:,1:]-y[:,:,:,:-1]).abs().mean((2,3)),
                      (y[:,:,1:,:]-y[:,:,:-1,:]).abs().mean((2,3))))
    low=F.adaptive_avg_pool2d(x,(3,10))
    parts.extend((low.mean(3).flatten(1),low.mean(2).flatten(1)))
    return torch.cat(parts,1)


def moment_loss(fake,real,family,scale):
    a=fixed_features(fake);b=fixed_features(real);loss=a.new_tensor(0.)
    for k in range(5):
        aa=a[family==k]/scale;bb=b[family==k]/scale
        # Small balanced batches: diagonal second moments are more stable than
        # a full covariance estimate. This is inspired by, not Deep CORAL.
        loss=loss+(aa.mean(0)-bb.mean(0)).square().mean()
        loss=loss+.1*(aa.std(0,unbiased=False)-bb.std(0,unbiased=False)).square().mean()
    return loss/5
