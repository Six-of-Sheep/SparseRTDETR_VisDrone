"""PR26 SABL-style conventional control. Native detector is never replaced."""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from ..protocol_r18 import model as common
from ..protocol_r18 import engine as training
from ..protocol_r18.geometry import validate_geometry
from ..baseline.training_v2b import _optimizer
from .sabl_axis_norm import normalized_sigmoid_axis

VERSION = 'r18-sabl-control-logsigmoid-v2'
PARAMETERS = {'N':20094584, 'S':21932156, 'A':21932156}
N_SHA = 'be5e9e1fb7a514eceaf2a1343dc29d2f512115bdd85b7fcc26dc11d3a66c9a83'


def need(ok, message):
    if not ok: raise ValueError(message)


def finite(x):
    need(bool(torch.isfinite(x).all()), 'nonfinite SABL tensor')


def corners(box):
    return torch.cat((box[...,:2]-box[...,2:]/2, box[...,:2]+box[...,2:]/2),-1)


def buckets(reference):
    r=reference.detach().float();finite(r)
    need(r.shape[-1]==4 and bool((r[...,2:]>0).all()), 'invalid reference box')
    c=r[...,:2];size=r[...,2:]*1.7;lo=c-size/2;hi=c+size/2;delta=size/14
    k=torch.arange(7,device=r.device,dtype=r.dtype)+.5
    x=torch.stack((lo[...,0,None]+delta[...,0,None]*k,
                   hi[...,0,None]-delta[...,0,None]*k,
                   lo[...,1,None]+delta[...,1,None]*k,
                   hi[...,1,None]-delta[...,1,None]*k),-2)
    scale=delta[...,[0,0,1,1]]
    finite(x);finite(scale);need(bool((scale>0).all()),"unrepresentable bucket width")
    return x,scale


def encode(reference,gt):
    x,scale=buckets(reference);g=corners(gt.detach().float());finite(g)
    need(bool((gt[...,2:]>0).all()), 'invalid GT box')
    t=(x-g[...,None])/scale[...,None];finite(t)
    order=torch.argsort(t.abs(),dim=-1,stable=True)
    label=torch.zeros_like(t).scatter_(-1,order[...,:1],1.)
    # Exactly the pinned coder's <1 neighboring-negative exclusion.
    cw=(~((t.abs()<1)&(label==0))).to(t.dtype)
    ow=torch.zeros_like(t).scatter_(-1,order[...,:1],1.)
    second=order[...,1:2];ow.scatter_(-1,second,(t.abs().gather(-1,second)<1).to(t.dtype))
    return t,label,cw,ow


def decode(reference,logits,offsets):
    finite(logits);finite(offsets)
    x,scale=buckets(reference);idx=logits.argmax(-1,keepdim=True)
    e=(x.gather(-1,idx)-scale[...,None]*offsets.gather(-1,idx)).squeeze(-1)
    finite(e)
    # Edge order here is L,R,T,B, unlike xyxy.
    wh=torch.stack((e[...,1]-e[...,0],e[...,3]-e[...,2]),-1)
    valid=(wh>0).all(-1)
    center=torch.stack(((e[...,0]+e[...,1])/2,(e[...,2]+e[...,3])/2),-1)
    # Invalid boxes have zero area for the existing postprocessor; never swap/fallback.
    box=torch.cat((center,torch.where(valid[...,None],wh,torch.zeros_like(wh))),-1)
    return box,valid


def boundary_loss(pred,targets,indices):
    z=pred['logits'];d=pred['offsets'];r=pred['reference'];selected=[];gt=[]
    for b,(src,dst) in enumerate(indices):
        selected.extend((b,int(q)) for q in src)
        gt.extend(targets[b]['boxes'][j] for j in dst)
    m=len(selected)
    if not m:return {'loss_bucket':z.sum()*0, 'loss_offset':d.sum()*0}, {'matches':0,'bucket_denominator':1,'offset_denominator':1}
    b,q=zip(*selected);b=list(b);q=list(q);t,y,cw,ow=encode(r[b,q],torch.stack(gt))
    cls=(F.binary_cross_entropy_with_logits(z[b,q],y,reduction='none')*cw).sum()/(4*m)
    reg=(F.smooth_l1_loss(d[b,q],t,beta=.1,reduction='none')*ow).sum()/m
    return {'loss_bucket':cls,'loss_offset':reg}, {'matches':m,'bucket_denominator':4*m,'offset_denominator':m}


def roi_features(field,reference,input_hw):
    r=reference.detach().float();finite(r)
    need(bool((r[...,2:]>0).all()),'invalid RoI reference')
    b,q=r.shape[:2];height,width=input_hw
    pixels=corners(r)*r.new_tensor([width,height,width,height])
    ids=torch.arange(b,device=r.device,dtype=r.dtype)[:,None,None].expand(b,q,1)
    rois=torch.cat((ids,pixels),-1).reshape(-1,5)
    return roi_align(field.float(),rois,(7,7),spatial_scale=1/8,sampling_ratio=2,aligned=True)


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
        # Same real-valued normalized sigmoid; stable evaluation, same axes.
        wx=normalized_sigmoid_axis(self.att_x(x),2)
        wy=normalized_sigmoid_axis(self.att_y(x),3)
        fx=(x*wx).sum(2);fy=(x*wy).sum(3)
        fx=F.relu(self.up_x(F.relu(self.post_x(fx)))).transpose(1,2)
        fy=F.relu(self.up_y(F.relu(self.post_y(fy)))).transpose(1,2)
        features=torch.stack((fx[:,:7],fx[:,7:].flip(1),fy[:,:7],fy[:,7:].flip(1)),1)
        return self.cls(features).squeeze(-1),self.offset(features).squeeze(-1)


class Model(nn.Module):
    def __init__(self,native,arm):
        super().__init__();need(arm in PARAMETERS,'unknown arm');self.native=native;self.arm=arm
        if arm!='N':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0);self.boundary=BoundaryHead()
        need(sum(p.numel() for p in self.parameters())==PARAMETERS[arm],'parameter inventory')
        self.last_dtype=None;self.last_boundary_dtype=None

    @property
    def decoder(self):return self.native.decoder
    @property
    def encoder(self):return self.native.encoder
    @property
    def backbone(self):return self.native.backbone

    def forward(self,images,targets=None):
        if self.arm=='N':
            out=self.native(images,targets);self.last_dtype=str(out['pred_logits'].dtype);return out
        seen=[]
        h=self.native.encoder.register_forward_hook(lambda module,args,output:seen.append(output[0]))
        try:out=self.native(images,targets)
        finally:h.remove()
        need(len(seen)==1 and seen[0].shape[1]==256,'F8 capture identity')
        self.last_dtype=str(out['pred_logits'].dtype)
        r=out['pred_boxes'].detach().float();need(r.shape[1:]==(300,4),'ordinary Q300')
        finite(r);need(bool((r[...,2:]>0).all()),'invalid native coarse box')
        b,q=r.shape[:2]
        with torch.autocast(images.device.type,enabled=False):
            roi=roi_features(seen[0],r,images.shape[-2:])
            z,d=self.boundary(roi,self.arm);self.last_boundary_dtype=str(z.dtype);z=z.reshape(b,q,4,7);d=d.reshape(b,q,4,7)
        if self.training:
            return {**out,'sabl':{'logits':z,'offsets':d,'reference':r}}
        refined,valid=decode(r,z,d)
        return {**out,'coarse_pred_boxes':out['pred_boxes'],'pred_boxes':refined,'refined_valid':valid}


class Criterion(nn.Module):
    def __init__(self,base,arm):
        super().__init__();self.base=base;self.arm=arm;self.last_diagnostics=[];self.last_matches=[]
    @property
    def matcher(self):return self.base.matcher
    def recipe(self):return {'native':self.base.recipe(),'arm':self.arm,'version':VERSION,'new_losses':[] if self.arm=='N' else ['bucket_BCE/4M','offset_SmoothL1_beta0.1/M']}
    def forward(self,out,targets):
        native={k:v for k,v in out.items() if k!='sabl'}
        losses=self.base(native,targets);self.last_diagnostics=list(self.base.last_diagnostics)
        if self.arm=='N':return losses
        final=[x for x in self.last_diagnostics if x['head']=='final'];need(len(final)==1,'native final matching diagnostic')
        pairs=[]
        for target,queries,ids in zip(targets,final[0]['matched_queries'],final[0]['matched_GT_ids']):
            mapping={key:i for i,key in enumerate(target['gt_ids'])};need(len(mapping)==len(target['gt_ids']),'duplicate GT ID')
            pairs.append((torch.tensor(queries,device=out['pred_boxes'].device,dtype=torch.long),torch.tensor([mapping[k] for k in ids],device=out['pred_boxes'].device,dtype=torch.long)))
        self.last_matches=pairs
        extra,diag=boundary_loss(out['sabl'],targets,pairs);self.last_diagnostics.append({'head':'SABL_ordinary_only',**diag})
        return {**losses,**extra}


def build(arm,config,checkpoint=None):
    native,base,_,_,_=common.native_objects(config)
    common.vendor._configure_model_sampling(native,config)
    from ..protocol_r18.criterion import adapt
    if checkpoint:
        from ..protocol_r18.checkpoint import file_sha
        need(file_sha(checkpoint)==N_SHA,'certified N SHA')
        p=torch.load(checkpoint,map_location='cpu',weights_only=True)
        need(p['format']=='sparse_rtdetr.sampling_span_fullstate_v1' and p['binding']['arm']=='r18-sampling-span-v1/B0','N identity')
        need(p['engine']['completed_updates']==36480 and p['engine']['completed_epochs']==120 and p['ema']['updates']==36480,'N e120 EMA')
        native.load_state_dict(p['ema']['module'],strict=True)
    return Model(native,arm),Criterion(adapt(base),arm)


@dataclass(frozen=True)
class TrainConfig(common.Config):
    epochs:int=24
    learning_rate:float=1e-5
    warmup_steps:int=200
    def binding(self):
        return {**super().binding(),'arm':VERSION,'scheduler':'warmup200_cosine7296_min1e-6','endpoint':'local_e24_EMA','native_DN':'unchanged','SABL_DN':'none'}


class Engine(training.Engine):
    def rate(self,update):
        need(1<=update<=7296,'24 epoch clock')
        return 1e-5*update/200 if update<=200 else 1e-6+.5*(1e-5-1e-6)*(1+math.cos(math.pi*(update-200)/(7296-200)))
    def health(self,model,optimizer=None,ema=None,*,gradients=False):
        need(sum(p.numel() for p in model.parameters())==PARAMETERS[model.arm],'model parameter change')
        # Native geometry uses the wrapper's unchanged decoder/encoder properties;
        # the certified health routine batches device checks without per-tensor sync.
        return training.assert_healthy(model,optimizer,ema,gradients=gradients)
    def train_window(self,*a,**k):
        need(self.completed_updates<7296,'24 epoch complete')
        return super().train_window(*a,**k)


def make_engine(arm,synthetic,checkpoint=None,device='cpu'):
    cfg=common.Config(input_hw=(128,128),physical_batch_size=1,amp_dtype='float32') if synthetic else common.Config()
    model,criterion=build(arm,cfg,checkpoint);model.to(device)
    train=TrainConfig(input_hw=cfg.input_hw,physical_batch_size=cfg.physical_batch_size,amp_dtype=cfg.amp_dtype)
    opt=_optimizer(model,train);ema=common.CommonEMA(model)
    return Engine(model,criterion,opt,ema,train)
