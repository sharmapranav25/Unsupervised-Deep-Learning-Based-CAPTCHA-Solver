"""Independent compact pixel-only count transformer, classes 2 through 8."""
import torch
from torch import nn
from torch.nn import functional as F
class CountTransformer(nn.Module):
    def __init__(self):
        super().__init__();self.patch=nn.Conv2d(3,128,8,stride=8)
        self.position=nn.Parameter(torch.randn(1,384,128)*.02)
        layer=nn.TransformerEncoderLayer(128,4,384,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(128);self.head=nn.Linear(128,7)
    def forward(self,pixels,valid_pixels):
        if pixels.shape[1:]!=(3,64,384) or valid_pixels.shape!=(len(pixels),64,384):raise ValueError('Count pixel geometry mismatch')
        pixels=torch.where(valid_pixels[:,None],pixels,torch.ones_like(pixels))
        tokens=self.patch((pixels-.5)/.5).flatten(2).transpose(1,2)+self.position
        coverage=F.avg_pool2d(valid_pixels[:,None].float(),8).flatten(1)
        z=self.norm(self.encoder(tokens,src_key_padding_mask=coverage==0))
        return self.head((z*coverage[:,:,None]).sum(1)/coverage.sum(1,keepdim=True).clamp_min(1))
    def resume_configuration(self):return dict(architecture='capri-independent-count-v1',canvas=[64,384],patch=8,width=128,layers=4,heads=4,classes=list(range(2,9)))
