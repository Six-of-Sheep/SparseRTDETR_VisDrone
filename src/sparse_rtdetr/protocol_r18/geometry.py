"""Admission checks only; no coordinate arithmetic replacement."""
import contextlib,importlib
import torch
from ..localization import consumer_geometry as geo
from ..r0.engine import tensor_digest

def geometry_identity(model):
    tensors=[(n,x) for n,x in model.state_dict().items() if n.rsplit('.',1)[-1] in {'anchors','valid_mask','num_points_scale'}]
    tensors.extend((f'pos_embed{i}',getattr(model.encoder,f'pos_embed{i}')) for i in model.encoder.use_encoder_idx)
    return {'canvas':list(model.decoder.eval_spatial_size),'cache_sha256':tensor_digest(tensors),
            'position':'native_width_first_meshgrid','arithmetic':'native_sigmoid_inverse_sigmoid_unchanged'}

def validate_geometry(model):
    d=model.decoder;e=model.encoder
    native=importlib.import_module('src.zoo.rtdetr.rtdetrv2_decoder')
    if type(d) is not native.RTDETRTransformerv2 or type(d.decoder) is not native.TransformerDecoder:
        raise ValueError('native decoder class differs')
    if any(hasattr(d,n) for n in ('pre_bbox_head','r0_geometry_parameters','reg_scale','box_history')):
        raise ValueError('R0/distribution adaptation found')
    if len(d.dec_bbox_head)!=3 or len(d.dec_score_head)!=3 or d.num_queries!=300 or d.num_classes!=10 or d.num_denoising!=100:
        raise ValueError('native head/denoising recipe differs')
    if any(h.layers[-1].out_features!=4 for h in d.dec_bbox_head) or not d.aux_loss:
        raise ValueError('native scalar regression/aux loss differs')
    if list(e.eval_spatial_size)!=list(d.eval_spatial_size):raise ValueError('canvas cache mismatch')
    if 'build_2d_sincos_position_embedding' in e.__dict__:raise ValueError('native position callable replaced')
    h,w=d.eval_spatial_size
    for i in e.use_encoder_idx:
        expected=type(e).build_2d_sincos_position_embedding(w//e.feat_strides[i],h//e.feat_strides[i],e.hidden_dim)
        actual=getattr(e,f'pos_embed{i}')
        if actual.dtype!=expected.dtype or not torch.equal(actual.cpu(),expected):raise ValueError('native position cache bytes differ')
    anchors,mask=d._generate_anchors()
    if not torch.equal(d.anchors.cpu(),anchors) or not torch.equal(d.valid_mask.cpu(),mask):raise ValueError('native anchor/cache bytes differ')

@contextlib.contextmanager
def forward_guards(model):
    handles=[]
    def reference(module,args,kwargs):
        ref=args[1] if len(args)>1 else kwargs['reference_points']
        geo.positive_box('native/attention_reference',ref)
    def output(module,args,result):
        if isinstance(result,torch.Tensor):geo.tensor('native/layer_output',result)
    for layer in model.decoder.decoder.layers:
        handles.append(layer.cross_attn.register_forward_pre_hook(reference,with_kwargs=True))
        handles.append(layer.register_forward_hook(output))
    try:yield
    finally:
        for h in handles:h.remove()
