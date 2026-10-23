"""CPU design leaf for H2: signed finest-level detail -> final center residual.

No detector adapter, data, checkpoint, training loop, evaluator or GPU launcher.
The pair-difference is an existing PDC operation, not a claimed new operator.
"""
from __future__ import annotations

import torch
from torch import nn
from ..baseline.training_v2b_deterministic_sampling import bilinear_grid_sample


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def locations(reference):
    """Four fixed cardinal locations at +/- quarter of the detached box extent.

    Input B,Q,4 normalized cxcywh. No feature-cell floor, clipping or GT input.
    Sampling uses the production attention-detach convention. Return B,Q,4,2.
    """
    _require(reference.ndim==3 and reference.shape[-1]==4 and reference.is_floating_point(), 'reference shape/type')
    _require(torch.isfinite(reference).all() and (reference[...,2:]>0).all(), 'finite positive extent')
    ref=reference.detach()
    direction=ref.new_tensor([[.25,0],[-.25,0],[0,.25],[0,-.25]])
    return ref[...,None,:2]+ref[...,None,2:]*direction


def local_samples(projected_finest, reference):
    """Same FP32/FP64 bilinear/zeros/align_corners=False gather as native core.

    Synthetic projected value map is B,C,H,W; return B,Q,4,C. In a future
    integrated core these values MUST be reused by the native attention sum,
    without a second sampler. This leaf is not an integrated detector.
    """
    _require(projected_finest.ndim==4 and projected_finest.shape[0]==reference.shape[0], 'feature batch/shape')
    _require(projected_finest.device==reference.device and projected_finest.dtype==reference.dtype, 'shared dtype/device')
    _require(torch.isfinite(projected_finest).all(), 'nonfinite feature')
    sample=bilinear_grid_sample(projected_finest,locations(reference)*2-1)
    return sample.permute(0,2,3,1)


def odd_even(sample):
    """Existing paired pixel differences and means, ordered x then y."""
    _require(sample.ndim==4 and sample.shape[-2]==4 and torch.isfinite(sample).all(), 'four finite samples')
    odd=torch.cat(((sample[:,:,0]-sample[:,:,1])/2,
                   (sample[:,:,2]-sample[:,:,3])/2),-1)
    even=torch.cat(((sample[:,:,0]+sample[:,:,1])/2,
                    (sample[:,:,2]+sample[:,:,3])/2),-1)
    return odd,even


def inverse_sigmoid(value, eps=1e-5):
    """Exact expression of pinned native utils.inverse_sigmoid, not a new loss."""
    value=value.clip(min=0.,max=1.)
    return torch.log(value.clip(min=eps)/(1-value).clip(min=eps))


def compose_box(reference, detail_delta, context_delta, *, additive_control=False):
    """D: center=detail only; U: center=detail+native contextual center.

    Width/height always use contextual delta. 'reference' is the native output
    reference: in production its aux-gradient role must remain distinct from
    detached sampling references. No promise of a globally context-free model.
    """
    _require(reference.shape==context_delta.shape and detail_delta.shape==reference.shape[:-1]+(2,), 'box shape')
    _require(all(torch.isfinite(x).all() for x in (reference,detail_delta,context_delta)), 'finite box inputs')
    center=detail_delta+context_delta[...,:2] if additive_control else detail_delta
    delta=torch.cat((center,context_delta[...,2:]),-1)
    return (inverse_sigmoid(reference)+delta).sigmoid()


class LocalDetailCenter(nn.Module):
    """The only added trainable leaf: bias-free 2C -> 2 linear readout.

    Zero init follows the native zero residual initialization. No feature
    normalization/nonlinearity is inserted before the signed difference.
    """
    def __init__(self,channels=256):
        super().__init__()
        _require(type(channels) is int and channels>0, 'channels')
        self.readout=nn.Linear(2*channels,2,bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self,projected_finest,reference,context_delta,*,additive_control=False):
        sampled=local_samples(projected_finest,reference)
        odd,even=odd_even(sampled)
        detail=self.readout(odd)
        return dict(box=compose_box(reference,detail,context_delta,additive_control=additive_control),
                    detail_delta=detail,odd=odd,even=even,sampled=sampled,
                    locations=locations(reference))
