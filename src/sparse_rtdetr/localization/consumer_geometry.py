"""consumer_geometry_contract_v1: checks only, all original arithmetic retained."""
import contextlib,contextvars,hashlib,importlib,json
from pathlib import Path
import torch
from .operators import NumericalIntegrityError

RULES={'name':'consumer_geometry_contract_v1_terminal_dn_aux_v1','valid_decoded_and_reference':'ordinary/DN+ and intermediate DN-: finite w/h>0 before/after FP32; only authenticated terminal valid DN- decoded output finite auxiliary state; all references/initial/support keep original valid',
 'padding':'original width/support exemption only; retain all computation and all-tensor finite/mass checks',
 'support':'actual initial FP64 atoms finite; existing 64*eps64*max(1,abs(x)) neighboring gap rule unchanged',
 'overlap':'actual FP32 corner intervals and areas positive finite; actual pairwise union and GIoU enclosure positive finite; output/cost/backward finite',
 'overlap_policy':'positive-area representation is a model semantic policy, not a universal IoU mathematical necessity',
 'ignore':'original FP64 full predicted area positive finite; actual intersections/IoA finite; threshold0.5 and matched retention unchanged',
 'attention':'all actual inputs/position/offset/logits/weights/locations/sampling/features and required gradients finite; no distinct-point demand',
 'loss':'all actual classification/FGL/KL/confidence/weightedterms/aggregates and gradients finite; unchanged empty branches and denominators',
 'arithmetic':'FP32 softmax; FP64 mass-normalized centered decode; all other existing dtypes unchanged',
 'changes':'accept/reject rules only; no new clamp/epsilon/drop/mask/regularizer/threshold scan'}
_AUDIT=contextvars.ContextVar('consumer_geometry_audit',default=None)
_LOCATION=contextvars.ContextVar('consumer_geometry_location',default={})

def identity():
    return {**RULES,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

class Audit:
    def __init__(self):self.records=[];self.counts={};self.gradient_counts={};self.microbatch=None
    def note(self,name,**fields):
        self.counts[name]=self.counts.get(name,0)+1
        self.records.append({'consumer':name,'microbatch':self.microbatch,**_LOCATION.get(),**fields})
    def summary(self):return {'contract':identity(),'counts':self.counts,'gradient_counts':self.gradient_counts,'records':self.records}

@contextlib.contextmanager
def auditing(audit):
    token=_AUDIT.set(audit)
    try:yield audit
    finally:_AUDIT.reset(token)

def fail(name,reason,**fields):
    raise NumericalIntegrityError('consumer geometry: '+name+': '+reason,consumer=name,geometry_contract=RULES['name'],
                                  location={**_LOCATION.get(),'microbatch':getattr(_AUDIT.get(),'microbatch',None)},**fields)

def tensor(name,value,gradient=True):
    if not isinstance(value,torch.Tensor):raise TypeError('geometry check expects actual tensor: '+name)
    if not bool(torch.isfinite(value).all()):fail(name,'nonfinite tensor',shape=list(value.shape),dtype=str(value.dtype),bad_indices=torch.nonzero(~torch.isfinite(value))[:16].detach().cpu().tolist())
    a=_AUDIT.get()
    if a is not None:
        detached=value.detach();a.note(name,shape=list(value.shape),dtype=str(value.dtype),elements=value.numel(),finite=True,
                                      minimum=float(detached.min()) if value.numel() and value.is_floating_point() else None,
                                      maximum=float(detached.max()) if value.numel() and value.is_floating_point() else None)
    if gradient and value.requires_grad:
        # Observe actual reverse pass; returning None leaves the gradient unchanged.
        location=dict(_LOCATION.get());micro=getattr(a,'microbatch',None)
        def observe(grad):
            if not bool(torch.isfinite(grad).all()):
                raise NumericalIntegrityError('consumer geometry: '+name+': nonfinite backward',consumer=name,location={**location,'microbatch':micro})
            if a is not None:a.gradient_counts[name]=a.gradient_counts.get(name,0)+1
            return None
        value.register_hook(observe)
    return value

def tree(name,value):
    if isinstance(value,torch.Tensor):tensor(name,value)
    elif isinstance(value,dict):
        for key,v in value.items():tree(name+'/'+str(key),v)
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):tree(name+'/'+str(i),v)

def positive_box(name,box,valid=None):
    tensor(name,box)
    checked=box if valid is None else box[valid]
    bad=(box[...,2:]<=0).any(-1)
    if valid is not None:bad=bad & valid
    if bool(bad.any()):
        indices=torch.nonzero(bad)[:16]
        fail(name,'valid box has nonpositive width/height',bad_indices=indices.detach().cpu().tolist(),bad_box_values=box[bad][:16].detach().cpu().tolist())
    a=_AUDIT.get()
    if a is not None:a.note(name+'/positive_semantics',valid_boxes=int(checked.numel()//4),padding_geometry_exempt=valid is not None)

def corners(name,boxes):
    tensor(name,boxes)
    wh=boxes[...,2:]-boxes[...,:2];tensor(name+'/intervals',wh,False)
    if bool((wh<=0).any()):fail(name,'actual consumer corner interval is not positive',dtype=str(boxes.dtype),bad_indices=torch.nonzero((wh<=0).any(-1))[:16].detach().cpu().tolist())
    area=wh.prod(-1);tensor(name+'/area',area,False)
    if bool((area<=0).any()):fail(name,'actual consumer area is not positive',dtype=str(boxes.dtype),bad_indices=torch.nonzero(area<=0)[:16].detach().cpu().tolist())
    return area

def overlap(name,operation,a,b,giou=False):
    # Admission uses detached duplicates; the original operation receives original tensors.
    with torch.no_grad():
        aa=corners(name+'/a',a);bb=corners(name+'/b',b)
        if a.numel() and b.numel():
            low=torch.maximum(a[:,None,:2],b[None,:,:2]);high=torch.minimum(a[:,None,2:],b[None,:,2:])
            intersection=(high-low).clamp_min(0).prod(-1) # unchanged original intersection expression, check only
            union=aa[:,None]+bb[None,:]-intersection
            tensor(name+'/intersection',intersection,False);tensor(name+'/union',union,False)
            if bool((union<=0).any()):fail(name,'nonpositive actual pairwise union')
            if giou:
                enclosure=(torch.maximum(a[:,None,2:],b[None,:,2:])-torch.minimum(a[:,None,:2],b[None,:,:2])).prod(-1)
                tensor(name+'/enclosure',enclosure,False)
                if bool((enclosure<=0).any()):fail(name,'nonpositive actual pairwise enclosing area')
    result=operation(a,b);tree(name+'/output',result);return result

@contextlib.contextmanager
def matching(matcher):
    module=importlib.import_module(type(matcher).__module__)
    original_iou=module.generalized_box_iou;original_solve=module.linear_sum_assignment
    def iou(a,b):return overlap('matcher/giou',original_iou,a,b,True)
    def solve(cost,*args,**kwargs):
        # Actual cost passed to SciPy; no replacement, no nan_to_num.
        if isinstance(cost,torch.Tensor):tensor('matcher/cost',cost,False)
        else:
            import numpy as np
            if not np.isfinite(cost).all():fail('matcher/cost','nonfinite actual assignment cost')
        return original_solve(cost,*args,**kwargs)
    module.generalized_box_iou=iou;module.linear_sum_assignment=solve
    try:yield
    finally:module.generalized_box_iou=original_iou;module.linear_sum_assignment=original_solve

@contextlib.contextmanager
def layer(layer,index,reference,valid):
    """Transient hooks preserve registered module/cache layout and saved callable identity."""
    from ..baseline import training_v2b_deterministic_sampling as sampling
    token=_LOCATION.set({'decoder_layer':index});handles=[];cross=layer.cross_attn
    core=cross.ms_deformable_attn_core;bilinear=sampling.bilinear_grid_sample
    def sample(value,grid):
        tensor('attention/bilinear_value',value);tensor('attention/bilinear_grid',grid)
        out=bilinear(value,grid);tensor('attention/bilinear_output',out);return out
    def core_checked(value,shapes,locations,weights,points,*args,**kwargs):
        tensor('attention/core_value',value);tensor('attention/locations',locations);tensor('attention/weights',weights)
        if any(int(h)<=0 or int(w)<=0 for h,w in shapes) or any(int(p)<=0 for p in points):fail('attention/layout','nonpositive feature dimension or point count')
        if bool((weights<0).any()):fail('attention/weights','negative softmax weight')
        mass=weights.sum(-1);tensor('attention/weight_mass',mass,False)
        if bool((mass<=0).any()):fail('attention/weights','nonpositive softmax mass')
        out=core(value,shapes,locations,weights,points,*args,**kwargs);tensor('attention/core_output',out);return out
    def hook(name):
        def observe(module,args,kwargs,output):
            tree(name+'/input',args);tree(name+'/kwargs',kwargs);tree(name,output)
        return observe
    try:
        positive_box('attention/reference',reference,valid)
        for name,module in (('self_attention',layer.self_attn),('attention/value_projection',cross.value_proj),
                            ('attention/offset',cross.sampling_offsets),('attention/weight_logits',cross.attention_weights),('attention/output_projection',cross.output_proj)):
            handles.append(module.register_forward_hook(hook(name),with_kwargs=True))
        cross.ms_deformable_attn_core=core_checked;sampling.bilinear_grid_sample=sample
        yield
    finally:
        cross.ms_deformable_attn_core=core;sampling.bilinear_grid_sample=bilinear
        for handle in handles:handle.remove()
        _LOCATION.reset(token)
