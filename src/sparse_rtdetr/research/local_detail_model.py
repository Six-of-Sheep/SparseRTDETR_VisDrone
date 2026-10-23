"""Explicit H2 N/U/D wiring. No vendor edits, hidden sampling or new losses."""
import functools
import importlib

import torch
from torch import nn
from torch.nn import functional as F

from ..protocol_r18.model import build as common_build
from ..protocol_r18.geometry import validate_geometry as native_geometry
from ..baseline.training_v2b_deterministic_sampling import (
    bilinear_grid_sample, deformable_attention_core_func_v2)

VERSION = 'r18-difference-center-residual-v1'
ARMS = ('N', 'U', 'D')
PARAMETERS = dict(N=20094584, U=20079160, D=20078646)


def shared_sample_core(value, shapes, locations, weights):
    """Native gather order; return the SAME unweighted fine samples, not a resample."""
    bs, _, heads, channels = value.shape
    queries = locations.shape[1]
    levels = value.permute(0, 2, 3, 1).flatten(0, 1).split([h*w for h,w in shapes], dim=-1)
    grids = (2*locations-1).permute(0, 2, 1, 3, 4).flatten(0, 1).split([4,4,4], dim=-2)
    samples = [bilinear_grid_sample(v.reshape(bs*heads, channels, h, w), g)
               for v, g, (h,w) in zip(levels, grids, shapes)]
    a = weights.permute(0, 2, 1, 3).reshape(bs*heads, 1, queries, 12)
    output = (torch.cat(samples, dim=-1)*a).sum(-1).reshape(bs, heads*channels, queries).permute(0,2,1)
    fine = samples[0].reshape(bs, heads, channels, queries, 4).permute(0,3,4,1,2).flatten(-2)
    return output, fine


class DetailAttention(nn.Module):
    def __init__(self, native):
        super().__init__()
        for name in ('embed_dim','num_heads','num_levels','offset_scale','num_points_list',
                     'total_points','method','head_dim'):
            setattr(self, name, getattr(native, name))
        self.register_buffer('num_points_scale', native.num_points_scale.clone())
        self.attention_weights = native.attention_weights
        self.value_proj = native.value_proj
        self.output_proj = native.output_proj
        self.sampling_offsets = nn.Linear(256, 128)
        keep = torch.arange(192).reshape(8,12,2)[:,4:,:].flatten()
        with torch.no_grad():
            self.sampling_offsets.weight.copy_(native.sampling_offsets.weight[keep])
            self.sampling_offsets.bias.copy_(native.sampling_offsets.bias[keep])
        self.detail_readout = nn.Linear(512, 2, bias=False)
        nn.init.zeros_(self.detail_readout.weight)
        self._pending_detail = None

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        if self._pending_detail is not None:
            raise RuntimeError('unconsumed H2 samples')
        if reference_points.shape[2:] != (1,4) or reference_points.requires_grad:
            raise ValueError('H2 requires the native detached final-layer box reference')
        bs, nq = query.shape[:2]
        projected = self.value_proj(value)
        if value_mask is not None:
            projected = projected*value_mask.to(projected.dtype).unsqueeze(-1)
        projected = projected.reshape(bs, value.shape[1], 8, 32)
        offsets = self.sampling_offsets(query).reshape(bs,nq,8,8,2)
        ref = reference_points[:,:,None,:,:]
        coarse = ref[...,:2] + offsets*self.num_points_scale[4:].to(query.dtype).unsqueeze(-1)*ref[...,2:]*self.offset_scale
        directions = reference_points.new_tensor([[.25,0],[-.25,0],[0,.25],[0,-.25]])
        fine = (reference_points[...,:2] + reference_points[...,2:]*directions).unsqueeze(2).expand(-1,-1,8,-1,-1)
        locations = torch.cat((fine,coarse),dim=-2)
        weights = F.softmax(self.attention_weights(query).reshape(bs,nq,8,12),dim=-1)
        output, samples = shared_sample_core(projected,value_spatial_shapes,locations,weights)
        dx=(samples[:,:,0]-samples[:,:,1])*.5
        dy=(samples[:,:,2]-samples[:,:,3])*.5
        self._pending_detail = self.detail_readout(torch.cat((dx,dy),dim=-1))
        return self.output_proj(output)


class DetailDecoder(nn.Module):
    def __init__(self, native, arm):
        super().__init__()
        self.layers = native.layers
        self.hidden_dim, self.num_layers, self.eval_idx = native.hidden_dim, native.num_layers, native.eval_idx
        self.arm = arm

    def forward(self, target, ref_points_unact, memory, memory_spatial_shapes,
                bbox_head, score_head, query_pos_head, attn_mask=None, memory_mask=None):
        native = importlib.import_module('src.zoo.rtdetr.rtdetrv2_decoder')
        boxes, logits = [], []
        ref_detach = F.sigmoid(ref_points_unact)
        output = target
        attention = self.layers[2].cross_attn
        try:
            for i, layer in enumerate(self.layers):
                output = layer(output,ref_detach.unsqueeze(2),memory,memory_spatial_shapes,
                               attn_mask,memory_mask,query_pos_head(ref_detach))
                delta = bbox_head[i](output)
                if i == 2:
                    detail = attention._pending_detail
                    attention._pending_detail = None
                    if detail is None: raise RuntimeError('H2 sample reuse absent')
                    center = detail + delta[...,:2] if self.arm == 'U' else detail
                    delta = torch.cat((center,delta[...,-2:]),dim=-1)
                inter = F.sigmoid(delta + native.inverse_sigmoid(ref_detach))
                if self.training:
                    logits.append(score_head[i](output))
                    # Preserve native undetached previous-layer reference for returned aux/final boxes.
                    boxes.append(inter if i == 0 else F.sigmoid(delta+native.inverse_sigmoid(ref_points)))
                elif i == self.eval_idx:
                    logits.append(score_head[i](output)); boxes.append(inter); break
                ref_points, ref_detach = inter, inter.detach()
        finally:
            attention._pending_detail = None
        return torch.stack(boxes), torch.stack(logits)


def validate_model(model, arm):
    if arm not in ARMS: raise ValueError('unregistered H2 arm')
    if getattr(model,'h2_arm',None) != arm: raise ValueError('unbound H2 model arm')
    d = model.decoder
    native = importlib.import_module('src.zoo.rtdetr.rtdetrv2_decoder')
    if type(d) is not native.RTDETRTransformerv2:
        raise ValueError('outer native R18 decoder differs')
    if any('forward' in m.__dict__ for m in (model,d,d.decoder)):
        raise ValueError('unbound H2 decoder callable')
    expected_decoder = native.TransformerDecoder if arm == 'N' else DetailDecoder
    if type(d.decoder) is not expected_decoder or (arm != 'N' and d.decoder.arm != arm):
        raise ValueError('H2 decoder identity differs')
    if (d.decoder.eval_idx,d.decoder.num_layers,d.decoder.hidden_dim,len(d.decoder.layers)) != (2,3,256,3):
        raise ValueError('H2 decoder layer/evaluation configuration differs')
    if (d.num_queries,d.num_classes,d.num_denoising,d.num_layers,d.nhead,d.num_levels) != (300,10,100,3,8,3) or not d.aux_loss:
        raise ValueError('H2 query/aux/DN recipe differs')
    if (d.label_noise_ratio,d.box_noise_scale,d.learn_query_content,d.query_select_method,d.cross_attn_method) != (.5,1.,False,'default','default'):
        raise ValueError('H2 native input/DN configuration differs')
    for i, layer in enumerate(d.decoder.layers):
        m = layer.cross_attn
        custom = arm != 'N' and i == 2
        if type(layer) is not native.TransformerDecoderLayer or type(m) is not (DetailAttention if custom else native.MSDeformableAttention):
            raise ValueError('H2 attention inventory differs')
        if any(getattr(layer,'dropout'+str(j)).p != 0 for j in range(1,5)):
            raise ValueError('H2 dropout configuration differs')
        if (m.embed_dim,m.num_heads,m.num_levels,m.offset_scale,m.method,m.num_points_list) != (256,8,3,.5,'default',[4,4,4]):
            raise ValueError('H2 sampling configuration differs')
        if not torch.equal(m.num_points_scale.cpu(),torch.full((12,),.25)):
            raise ValueError('H2 point scale differs')
        if custom:
            if m.sampling_offsets.out_features != 128 or m.detail_readout.weight.shape != (2,512) or m.detail_readout.bias is not None or m._pending_detail is not None:
                raise ValueError('H2 detail projection/cache differs')
        else:
            core=m.ms_deformable_attn_core
            if not isinstance(core,functools.partial) or core.func is not deformable_attention_core_func_v2 or core.args or core.keywords != {'method':'default'}:
                raise ValueError('H2 native core differs')
        if 'forward' in m.__dict__ or 'forward' in layer.__dict__:
            raise ValueError('unbound H2 callable')
    if [h.layers[-1].out_features for h in d.dec_bbox_head] != [4,4,2 if arm=='D' else 4]:
        raise ValueError('H2 center output rows differ')
    # Check native caches without relaxing the legacy class guard.
    h,w=d.eval_spatial_size
    if list(model.encoder.eval_spatial_size) != [h,w]: raise ValueError('H2 canvas differs')
    for i in model.encoder.use_encoder_idx:
        expected=type(model.encoder).build_2d_sincos_position_embedding(w//model.encoder.feat_strides[i],h//model.encoder.feat_strides[i],256)
        if not torch.equal(getattr(model.encoder,f'pos_embed{i}').cpu(),expected): raise ValueError('H2 position cache differs')
    anchors, mask=d._generate_anchors()
    if not torch.equal(d.anchors.cpu(),anchors) or not torch.equal(d.valid_mask.cpu(),mask): raise ValueError('H2 anchor cache differs')
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[arm]:
        raise ValueError('H2 parameter count differs')
    if arm == 'N': native_geometry(model)
    return dict(version=VERSION,arm=arm,parameters=PARAMETERS[arm],points=[4,4,4],radius_fraction=.25,
                sampling='deterministic_gather_shared_fine_samples',reference='native_detached_sampling_undetached_aux')


def build(config, arm, *, pretrained=None, synthetic=False):
    if arm not in ARMS: raise ValueError('unregistered H2 arm')
    model, criterion, info = common_build(config,pretrained_path=pretrained,synthetic=synthetic)
    if arm != 'N':
        with torch.random.fork_rng(devices=[]):
            inner=model.decoder.decoder
            inner.layers[2].cross_attn=DetailAttention(inner.layers[2].cross_attn)
            model.decoder.decoder=DetailDecoder(inner,arm)
            if arm == 'D':
                head=model.decoder.dec_bbox_head[2].layers[-1]
                wh=nn.Linear(256,2)
                with torch.no_grad():
                    wh.weight.copy_(head.weight[2:]); wh.bias.copy_(head.bias[2:])
                model.decoder.dec_bbox_head[2].layers[-1]=wh
    model.h2_arm=arm
    info['h2']=validate_model(model,arm)
    info['trainable_parameters']=PARAMETERS[arm]
    return model,criterion,info
