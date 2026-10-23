"""Reference-extent adapter: a one-cell floor on finest-level sampling extent.

No global/vendor patch; no change to regression references, losses or queries.
The legacy production validator deliberately rejects this class. The explicit
span_model adapter admits it only under the new sampling-span task binding;
legacy task identities are not widened.
"""
from __future__ import annotations

import copy
import functools
from pathlib import Path

import torch

from sparse_rtdetr.baseline import training_v2b as vendor
from sparse_rtdetr.baseline.training_v2b_deterministic_sampling import (
    deformable_attention_core_func_v2,
)

ROOT = Path(__file__).resolve().parents[3]
with vendor._vendor_path(vendor._vendor_root(ROOT)):
    from src.zoo.rtdetr.rtdetrv2_decoder import MSDeformableAttention


def sampling_references(reference, shapes, *, enabled=True):
    """Expand B,Q,1,4 refs in native *point order*, not in level order.

    The native 4D-reference branch broadcasts against B,Q,heads,12,2 offsets.
    Passing B,Q,3,4 here would be incorrect. Four consecutive points per level
    are used; only the first four receive the one-cell extent floor.
    Checks synchronize CPU tensors in this prototype, not a latency claim.
    """
    if (reference.ndim != 4 or reference.shape[-2:] != (1, 4)
            or not reference.is_floating_point()
            or not torch.isfinite(reference).all()
            or not (reference[..., 2:] > 0).all()):
        raise ValueError('expected finite B,Q,1,4 references with positive extent')
    if (len(shapes) != 3 or any(len(s) != 2 or any(type(v) is not int or v <= 0 for v in s)
                                for s in shapes)):
        raise ValueError('expected three positive integer feature H,W pairs')
    if any(shapes[i][0] < shapes[i+1][0] or shapes[i][1] < shapes[i+1][1] for i in range(2)):
        raise ValueError('feature order must be finest to coarsest')
    if not enabled:
        return reference
    # No in-place edits on original decoder references. No center clipping.
    spans = reference[..., 2:].expand(-1, -1, 12, -1)
    floor = reference.new_zeros(12, 2)
    height, width = shapes[0]
    floor[:4, 0] = 1.0 / width
    floor[:4, 1] = 1.0 / height
    centers = reference[..., :2].expand(-1, -1, 12, -1)
    return torch.cat((centers, torch.maximum(spans, floor)), dim=-1)


class FinestSpanAttention(MSDeformableAttention):
    """Same parameters/state keys and native forward; reference adapter only."""

    def __init__(self, embed_dim=256, num_heads=8, *, enabled=True):
        super().__init__(embed_dim, num_heads, 3, [4, 4, 4], method='default', offset_scale=0.5)
        self.span_enabled = enabled  # configuration, explicitly not a checkpoint tensor

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        refs = sampling_references(reference_points, value_spatial_shapes, enabled=self.span_enabled)
        return super().forward(query, refs, value, value_spatial_shapes, value_mask)


def bind_common_core(attention):
    """Owned instance only; all three arms use the common deterministic core."""
    attention.ms_deformable_attn_core = functools.partial(
        deformable_attention_core_func_v2, method='default')
    return attention


def from_baseline(baseline, arm):
    """Construct a CPU leaf, sharing every tensor except native control buffers.

    B1 uses native radial-bias initialization for [8,2,2]; copying B0's [4,4,4]
    radial bias would not implement that conventional control. RNG fork prevents
    replacement construction from changing the shared training RNG stream.
    """
    if (type(baseline) is not MSDeformableAttention
            or baseline.num_points_list != [4, 4, 4] or baseline.num_levels != 3
            or baseline.offset_scale != 0.5 or baseline.method != 'default'
            or next(baseline.parameters()).device.type != 'cpu'):
        raise ValueError('prototype requires the exact CPU native common leaf')
    if arm not in ('B0', 'B1', 'P'):
        raise ValueError('unknown arm')
    if arm == 'B0':
        return bind_common_core(copy.deepcopy(baseline))
    with torch.random.fork_rng(devices=[]):
        if arm == 'P':
            result = FinestSpanAttention(baseline.embed_dim, baseline.num_heads)
        else:
            result = MSDeformableAttention(baseline.embed_dim, baseline.num_heads, 3,
                                           [8, 2, 2], method='default', offset_scale=0.5)
    state = copy.deepcopy(baseline.state_dict())
    if arm == 'B1':
        native_control = result.state_dict()
        for name in ('sampling_offsets.bias', 'num_points_scale'):
            state[name] = native_control[name]
    result.load_state_dict(state, strict=True)
    result.train(baseline.training)
    return bind_common_core(result)


def adapt_cpu_model(baseline, arm):
    """CPU construction seam, used by the explicit production adapter; legacy guard unchanged."""
    vendor.validate_model_sampling(baseline, sampling_backend='deterministic_gather')
    result = copy.deepcopy(baseline)
    layers = result.decoder.decoder.layers
    if len(layers) != 3:
        raise ValueError('expected three common decoder layers')
    for layer in layers:
        layer.cross_attn = from_baseline(layer.cross_attn, arm)
    return result
