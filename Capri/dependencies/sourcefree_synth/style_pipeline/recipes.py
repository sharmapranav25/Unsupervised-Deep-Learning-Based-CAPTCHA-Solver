"""Bounded general renderer controls and small recipe-prediction networks."""
import copy
import numpy as np
import torch
from torch import nn

PARAMETERS=['background_r','background_g','background_b','text_height','tilt','spacing','line_count','line_width','dot_count','curve_count','curve_amplitude']

def apply_recipe(style,values):
    p=np.clip(np.asarray(values,float),-1,1)
    if p.shape!=(11,) or not np.isfinite(p).all():raise ValueError('Invalid recipe')
    if not p.any():return copy.deepcopy(style)
    s=copy.deepcopy(style);offset=p[:3]*45
    def color(c):return np.clip(np.array(c)+offset,0,255).astype(int).tolist()
    s['background_palette']=[color(c) for c in s['background_palette']]
    for key in ('gradient_left','gradient_right'):
        if key in s:s[key]=color(s[key])
    s['dark_background']=np.mean(s['background_palette'][0])<128
    s['text_height']=float(np.clip(s['text_height']*np.exp(p[3]*.4),.25,.75))
    s['text_tilt_proxy']=float(np.clip((s['text_tilt_proxy'] or 0)*np.exp(p[4]),0,25))
    for key,index,multiplier in [('spacing_gain',5,.2),('line_count_gain',6,1.6),('line_width_gain',7,1.4),
        ('dot_count_gain',8,1.6),('curve_count_gain',9,1.2),('curve_amplitude_gain',10,1.2)]:s[key]=float(np.exp(p[index]*multiplier))
    return s

class RecipeCNN(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(3,24,5,2,2),nn.SiLU(),nn.Conv2d(24,48,3,2,1),nn.SiLU(),
            nn.Conv2d(48,64,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d((2,6)),nn.Flatten(),nn.Linear(768,64),nn.SiLU(),nn.Linear(64,11),nn.Tanh())
    def forward(self,x):return self.net(x)

class RecipeTransformer(nn.Module):
    def __init__(self):
        super().__init__();self.patch=nn.Conv2d(3,48,8,8);self.position=nn.Parameter(torch.zeros(1,192,48))
        layer=nn.TransformerEncoderLayer(48,4,128,0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Sequential(nn.Linear(48,11),nn.Tanh())
    def forward(self,x):return self.head(self.encoder(self.patch(x).flatten(2).transpose(1,2)+self.position).mean(1))
