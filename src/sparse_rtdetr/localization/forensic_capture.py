"""Read actual runtime locals before the ORIGINAL positivity call; no arithmetic edit."""
import ast,contextlib,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from . import distribution
from .runio import require,reference,publish,event,fsync_directory

def call_line(path,function,callee,label=None):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'));found=[]
    for f in ast.walk(tree):
        if isinstance(f,ast.FunctionDef) and f.name==function:
            for n in ast.walk(f):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==callee:
                    if label is None or (n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value==label):found.append(n.lineno)
    require(len(found)==1,'ambiguous original capture boundary');return found[0]

def cpu(value):return value.detach().cpu().clone() if isinstance(value,torch.Tensor) else value

class Capture:
    def __init__(self,folder,identity,targets,logical_metadata,microbatch=0):
        self.folder=Path(folder);self.identity=identity;self.targets=targets;self.metadata=logical_metadata;self.microbatch=microbatch
        self.heads={};self.pending=None;self.support=None;self.records=[];self.handles=[];self.seen_model_layers=[]
        self.center_code=distribution.centered_box_from_probabilities.__code__;self.support_code=distribution.support_torch.__code__
        self.positive_line=call_line(distribution.__file__,'centered_box_from_probabilities','positive_box','decode/box64')

    def head_hook(self,index):
        def hook(module,args,output):
            require(index not in self.heads,'regression head called twice in target forward')
            self.heads[index]={'input_actual':cpu(args[0]),'raw_output_actual':cpu(output)}
            return None
        return hook

    def layer_begin(self,frame):
        v=frame.f_locals;i=int(v['index']);require(i==len(self.seen_model_layers) and i in (0,1,2),'target layer order/scope differs')
        require(i in self.heads,'actual residual head output was not captured')
        self.pending={'layer':i,'fixed_initial':cpu(v['initial']),'attention_reference':cpu(v['ref']),'query_valid':cpu(v['valid']),
                      'cumulative_logits_actual':cpu(v['corners']),'previous_cumulative_logits_actual':cpu(v['previous_logits']),
                      'previous_feature_actual':cpu(v['previous_feature']),**self.heads[i]}
        self.seen_model_layers.append(i);self.support=None

    def trace(self,frame,kind,arg):
        if kind=='call':
            return self.trace if frame.f_code in (self.model_code,self.center_code,self.support_code) else None
        if frame.f_code is self.model_code and kind=='line' and frame.f_lineno==self.decode_line:self.layer_begin(frame)
        elif frame.f_code is self.support_code and kind=='return' and self.pending is not None:self.support=cpu(arg)
        elif frame.f_code is self.center_code and kind=='line' and frame.f_lineno==self.positive_line:self.save_before_check(frame.f_locals)
        return self.trace

    def roles(self,query_count,valid):
        max_gt=max(len(t['labels']) for t in self.targets);dn=query_count-300
        require(dn>=0 and (dn==0 or dn%(2*max_gt)==0),'actual DN layout not recognized')
        groups=dn//(2*max_gt) if dn else 0;rows=[]
        for b,t in enumerate(self.targets):
            require(t['image_id']==self.metadata['ids'][self.microbatch*8+b],'actual target/image identity differs')
            require(len(t['gt_ids'])==len(t['labels'])==len(t['boxes']),'actual GT IDs differ')
            for q in range(query_count):
                r={'sample_in_microbatch':b,'logical_sample':self.microbatch*8+b,'query':q,'image_id':t['image_id'],'valid':bool(valid[b,q]),'GT_id':None,'GT_slot':None}
                if q<dn:
                    slot=q%(2*max_gt);g=slot%max_gt;positive=slot<max_gt
                    r.update(DN_group=q//(2*max_gt),GT_slot=g,role='DN_positive' if positive else 'DN_negative')
                    if g<len(t['gt_ids']):r.update(GT_id=t['gt_ids'][g],GT_label=int(t['labels'][g]),GT_box=cpu(t['boxes'][g]).tolist(),GT_association='denoising-construction slot; negative is not a positive regression match')
                    else:r['role']='padding'
                else:r.update(role='ordinary',normal_query=q-dn,GT_association='not matched: no loss/matcher executed')
                rows.append(r)
        return {'DN_queries':dn,'max_GT':max_gt,'DN_groups':groups,'roles':rows,'targets':[{'image_id':t['image_id'],'GT_ids':t['gt_ids'],'labels':cpu(t['labels']),'boxes':cpu(t['boxes'])} for t in self.targets]}

    def save_before_check(self,v):
        require(self.pending is not None and self.support is not None,'actual initial/support context missing before check')
        i=self.pending['layer'];require(i==len(self.records),'capture repeated/partial layer')
        payload={**self.pending,'schema':'f0-actual-pre-positivity-forensic-v1','execution':self.identity,'microbatch':self.microbatch,'logical_input':self.metadata,
                 'probabilities_actual_FP32':cpu(v['probabilities']),'mass_actual_FP64':cpu(v['mass']),'W_atoms_actual_FP64':cpu(v['atoms']),
                 'physical_support_actual_W_order_FP64':self.support,'relative_means_actual_FP64':cpu(v['relative']),'signed_displacements_actual_FP64':cpu(v['delta']),
                 'box64_actual':cpu(v['box64']),'centered_reference_actual_FP64':cpu(v['r']),
                 'side_order':['left','top','right','bottom'],'support_direction_in_W_order':['decreasing','decreasing','increasing','increasing'],
                 'residual_logits_FP32_from_actual_head_cast':self.pending['raw_output_actual'].float(),
                 'before_original_positive_box':True,'arithmetic_overridden':False,'normal_protection_bypassed':False}
        p=payload['probabilities_actual_FP32'].numpy().astype(np.float64);s=self.support.numpy();mass=p.sum(-1)
        payload['physical_edge_means_derived_CPU_FP64']=torch.from_numpy((p*s).sum(-1)/mass)
        valid=payload['query_valid'];payload.update(self.roles(payload['box64_actual'].shape[1],valid))
        box=payload['box64_actual'];payload['nonpositive_valid_queries']=torch.nonzero((box[...,2:]<=0).any(-1)&valid).tolist()
        path=self.folder/f'layer-{i:02d}-actual.pth';partial=path.with_suffix('.pth.partial')
        require(not path.exists() and not partial.exists(),'preserve existing forensic capture')
        with partial.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.link(partial,path);fsync_directory(path.parent);partial.unlink();fsync_directory(path.parent)
        ref=reference(path);summary={'schema':payload['schema'],'layer':i,'microbatch':self.microbatch,'capture':ref,'bad_indices':payload['nonpositive_valid_queries'],
                                   'before_original_positive_box':True,'execution':self.identity}
        publish(self.folder/f'layer-{i:02d}-summary.json',summary);self.records.append(summary)
        event(self.folder,'forensic_layer_saved',parent_checkpoint_sha256=self.identity['forensic_authorization']['parent_checkpoint_sha256'],
              completed_updates=13693,candidate_update=13694,epoch=46,window=13,microbatch=0,layer=i,capture_sha256=ref['sha256'])

    @contextlib.contextmanager
    def activate(self,decoder):
        require(sys.gettrace() is None,'existing Python trace would conflict with bounded capture')
        self.model_code=decoder.forward.__func__.__code__;self.decode_line=call_line(self.model_code.co_filename,'forward','decode')
        require(len(decoder.dec_bbox_head)==3,'forensic decoder scope differs')
        self.folder.mkdir(exist_ok=False)
        for i,h in enumerate(decoder.dec_bbox_head):self.handles.append(h.register_forward_hook(self.head_hook(i)))
        try:
            sys.settrace(self.trace);yield self
        finally:
            sys.settrace(None)
            for h in self.handles:h.remove()
            self.handles=[];self.pending=None;self.support=None
