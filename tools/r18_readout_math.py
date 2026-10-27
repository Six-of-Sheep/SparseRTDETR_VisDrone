"""FP32 batched implementation of the frozen PR25 arithmetic; no detector."""
import torch
from torch import nn


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def xyxy(b):
    return torch.cat((b[..., :2]-b[..., 2:]/2, b[..., :2]+b[..., 2:]/2), -1)


def grid(b):
    b = b.detach()
    require(torch.isfinite(b).all() and (b[..., 2:] > 0).all(), 'invalid box')
    a = 3**(1/15)
    w = b.new_tensor([-4]+[1-a**i for i in range(15,0,-1)]+[0]+[a**i-1 for i in range(1,16)]+[4])
    d = .5+w/4
    c, s = b[..., :2], b[..., 2:]
    return torch.stack((c[...,0,None]-s[...,0,None]*d,c[...,1,None]-s[...,1,None]*d,
                        c[...,0,None]+s[...,0,None]*d,c[...,1,None]+s[...,1,None]*d),-2)


def transport(p, x, y):
    x,y=x.detach(),y.detach()
    require(torch.isfinite(p).all() and (p>=0).all() and ((p.sum(-1)-1).abs()<2e-6).all(),'invalid probability')
    ys,order=y.sort(-1)
    require((ys.diff(dim=-1)>0).all(),'invalid grid')
    xc=x.clamp(ys[...,:1],ys[...,-1:])
    hi=torch.searchsorted(ys.contiguous(),xc.contiguous()).clamp(1,32)
    lo=hi-1
    yl=ys.gather(-1,lo); yh=ys.gather(-1,hi)
    alpha=(xc-yl)/(yh-yl)
    out=torch.zeros_like(p).scatter_add(-1,order.gather(-1,lo),p*(1-alpha))
    out=out.scatter_add(-1,order.gather(-1,hi),p*alpha)
    outside=((x<ys[...,:1])|(x>ys[...,-1:])).to(p.dtype)
    return out,(p*outside).sum(-1), (out*y).sum(-1)-(p*x).sum(-1)


def targets(gt,g):
    x=xyxy(gt.detach()).unsqueeze(-1)
    ys,order=g.detach().sort(-1)
    xc=x.clamp(ys[...,:1],ys[...,-1:])
    hi=torch.searchsorted(ys.contiguous(),xc.contiguous()).clamp(1,32);lo=hi-1
    a=(xc-ys.gather(-1,lo))/(ys.gather(-1,hi)-ys.gather(-1,lo))
    out=torch.zeros_like(g).scatter_add(-1,order.gather(-1,lo),1-a)
    return out.scatter_add(-1,order.gather(-1,hi),a).detach()


def decode(p,g):
    e=(p*g.detach()).sum(-1)
    b=torch.cat(((e[...,:2]+e[...,2:])/2,e[...,2:]-e[...,:2]),-1)
    require(torch.isfinite(b).all() and (b[...,2:]>0).all(),'invalid decoded box')
    return b


def overlaps(b,t):
    a,c=xyxy(b),xyxy(t)
    inter=(torch.minimum(a[...,2:],c[...,2:])-torch.maximum(a[...,:2],c[...,:2])).clamp(min=0).prod(-1)
    union=b[...,2:].prod(-1)+t[...,2:].prod(-1)-inter
    enclosing=(torch.maximum(a[...,2:],c[...,2:])-torch.minimum(a[...,:2],c[...,:2])).prod(-1)
    iou=inter/union
    return iou,iou-(enclosing-union)/enclosing


class Readout(nn.Module):
    def __init__(self,arm):
        super().__init__();require(arm in ('C0','C1','M'),'arm')
        self.arm=arm
        self.heads=nn.ModuleList([nn.Linear(256,132) for _ in range(3)])
        for p in self.parameters(): nn.init.zeros_(p)

    def forward(self,q,r):
        require(q.dtype==torch.float32 and r.dtype==torch.float32,'FP32 required')
        q=q.detach();r=r.detach();g=grid(r); states=[];p=None;zsum=None
        for i,head in enumerate(self.heads):
            z=head(q[...,i,:,:]).reshape(*r.shape[:-1],4,33)
            require(torch.isfinite(z).all(),'nonfinite logits')
            ng=grid(r if i==0 or self.arm=='C0' else states[-1]['box'])
            outside=torch.zeros_like(z[...,0]);bias=outside
            if i==0 or self.arm=='C1': lp=z.log_softmax(-1)
            elif self.arm=='C0': lp=(zsum+z).log_softmax(-1)
            else:
                prior,outside,bias=transport(p,g,ng)
                lp=((prior+1e-8).log()+z).log_softmax(-1)
            zsum=z if zsum is None else zsum+z
            p=lp.exp(); b=decode(p,ng)
            states.append(dict(box=b,p=p,logp=lp,g=ng,z=z,outside=outside,bias=bias,oldg=g))
            g=ng
        return states


def loss(states,gt,mask):
    require(mask.any(),'empty matching batch')
    total=0
    for s in states:
        b=s['box'][mask];t=gt[mask].detach()
        label=targets(t,s['g'][mask])
        ce=-(label*s['logp'][mask]).sum(-1).mean()
        total=total+ce+5*(b-t).abs().mean()+2*(1-overlaps(b,t)[1]).mean()
    return total/3


def local_uniform(states):
    # Same old/new grids, same residual. No head invocation or cascading.
    result=[]
    for i in (1,2):
        s=states[i];old=states[i-1]['p']
        u=torch.ones_like(old)/33
        prior,_,_=transport(u,s['oldg'],s['g'])
        result.append(decode(((prior+1e-8).log()+s['z']).softmax(-1),s['g']))
    return result
