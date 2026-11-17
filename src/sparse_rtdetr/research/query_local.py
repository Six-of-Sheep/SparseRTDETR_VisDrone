"""R25 v1: ordinary-query conditioned 2D local continuous box residual.
No old research install. Native backbone/encoder/decoder/criterion remain intact.
"""
import copy, math
import torch
from torch import nn
from torchvision.ops import roi_align

VERSION='r25-query-local-v1'

def roi_boxes(b,hw):
    # Same canvas units as native normalized cxcywh; clip expanded corners before sampling.
    h,w=hw
    center=b.detach()[...,:2];half=b.detach()[...,2:]*.75
    xy=torch.cat([(center-half).clamp(0,1),(center+half).clamp(0,1)],-1)
    xy=xy*xy.new_tensor([w,h,w,h])
    ids=torch.arange(b.shape[0],device=b.device,dtype=xy.dtype)[:,None,None].expand(-1,b.shape[1],1)
    return torch.cat([ids,xy],-1).reshape(-1,5)

class LocalHead(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in ('A1','A2'):raise ValueError('R25 arm')
        self.mode=mode
        self.wq=nn.Linear(256,64);self.ln=nn.LayerNorm(64)
        self.wk=nn.Linear(64,64,bias=False);self.wp=nn.Linear(2,64,bias=False)
        self.wv=nn.Linear(64,64);self.hidden=nn.Linear(132,64);self.out=nn.Linear(64,4)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        a=torch.linspace(-1,1,7);y,x=torch.meshgrid(a,a,indexing='ij')
        self.register_buffer('relative_xy',torch.stack([x,y],-1).reshape(49,2),persistent=False)
    def forward(self,q,b,f4,hw):
        rois=roi_boxes(b.float(),hw)
        # Aligned RoIAlign geometry is fixed; torchvision handles deterministic CUDA fallback.
        tokens=roi_align(f4,rois,(7,7),spatial_scale=.25,
                         sampling_ratio=2,aligned=True).flatten(2).transpose(1,2)
        B,Q=q.shape[:2];tokens=tokens.reshape(B,Q,49,64)
        z=self.ln(self.wq(q))
        condition=q.mean(1,keepdim=True).expand_as(q) if self.mode=='A1' else q
        zatt=self.ln(self.wq(condition))
        k=self.wk(tokens)+self.wp(self.relative_xy.to(tokens.dtype))
        a=torch.softmax((zatt[:,:,None,:]*k).sum(-1)/8.,dim=-1)
        u=(a[:,:,:,None]*self.wv(tokens)).sum(2)
        delta=self.out(torch.relu(self.hidden(torch.cat([z,u,b.detach()],-1))))
        from src.zoo.rtdetr.utils import inverse_sigmoid
        refined=torch.sigmoid(inverse_sigmoid(b)+delta)
        return refined,delta,rois,a

class QueryLocalModel(nn.Module):
    def forward(self,x,targets=None):
        captured={}
        def f4_hook(m,args,out):captured['f4']=out
        def q_hook(m,args):captured['q']=args[0][:,-300:]
        hf=self.backbone.res_layers[0].register_forward_hook(f4_hook)
        hq=self.decoder.dec_score_head[-1].register_forward_pre_hook(q_hook)
        try:
            features=self.backbone(x);features=self.encoder(features)
            out=self.decoder(features,targets)
            q=captured['q'];f4=captured['f4'].view_as(captured['f4']);b=out['pred_boxes']
            if q.shape[1]!=300 or b.shape[1]!=300 or f4.shape[1]!=64:raise ValueError('R25 ordinary/F4 shape')
            refined,delta,rois,att=self.local_head(q,b,f4,x.shape[-2:])
            # Monitoring is value-only and does not enter criterion or inference decisions.
            self.local_observation={'coarse':b.detach(),'refined':refined.detach(),'delta':delta.detach(),
                 'rois':rois.detach(),'attention':att.detach(),'F4_shape':list(f4.shape),'query_shape':list(q.shape)}
            if getattr(self,'capture_local_gradients',False):
                if q.requires_grad:q.retain_grad()
                if f4.requires_grad:f4.retain_grad()
                self.local_gradient_tensors=(q,f4)
            out['pred_boxes']=refined
            return out
        finally:hf.remove();hq.remove()

def install(model,arm):
    if arm=='A0':return model
    if hasattr(model,'local_head'):raise ValueError('R25 double install')
    # Independent initialization; no effect on restored data/model/DN random streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(25001);head=LocalHead(arm)
    model.__class__=QueryLocalModel;model.add_module('local_head',head)
    model.local_observation={}
    return model
