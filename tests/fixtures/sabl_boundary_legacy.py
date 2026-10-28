"""Verbatim original helper/class fragments, prior 79ba593 scientific source. No model or optimizer."""
import torch
from torch import nn
from torch.nn import functional as F

def need(ok, message):
    if not ok: raise ValueError(message)

def finite(x):
    need(bool(torch.isfinite(x).all()), 'nonfinite SABL tensor')

class BoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU(),nn.Conv2d(256,256,3,padding=1),nn.ReLU())
        self.att_x=nn.Conv2d(256,1,1);self.att_y=nn.Conv2d(256,1,1)
        self.post_x=nn.Conv1d(256,256,3,padding=1);self.post_y=nn.Conv1d(256,256,3,padding=1)
        self.up_x=nn.ConvTranspose1d(256,256,2,stride=2);self.up_y=nn.ConvTranspose1d(256,256,2,stride=2)
        self.cls=nn.Linear(256,1);self.offset=nn.Linear(256,1)
        for module in self.modules():
            if isinstance(module,(nn.Conv1d,nn.Conv2d,nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(module.weight,mode='fan_in',nonlinearity='relu' if module not in (self.up_x,self.up_y) else 'leaky_relu')
                nn.init.zeros_(module.bias)
        for module,std in ((self.att_x,.01),(self.att_y,.01),(self.cls,.01),(self.offset,.001)):
            nn.init.normal_(module.weight,std=std);nn.init.zeros_(module.bias)

    def forward(self,roi,arm):
        finite(roi)
        if arm=='A':roi=roi.mean((-2,-1),keepdim=True).expand_as(roi)
        x=self.pre(roi)
        ax=self.att_x(x).sigmoid();ay=self.att_y(x).sigmoid()
        dx=ax.sum(2,keepdim=True);dy=ay.sum(3,keepdim=True)
        need(bool((dx>0).all() and (dy>0).all()),'zero attention denominator')
        fx=(x*(ax/dx)).sum(2);fy=(x*(ay/dy)).sum(3)
        fx=F.relu(self.up_x(F.relu(self.post_x(fx)))).transpose(1,2)
        fy=F.relu(self.up_y(F.relu(self.post_y(fy)))).transpose(1,2)
        features=torch.stack((fx[:,:7],fx[:,7:].flip(1),fy[:,:7],fy[:,7:].flip(1)),1)
        return self.cls(features).squeeze(-1),self.offset(features).squeeze(-1)
