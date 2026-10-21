"""Explicit sampling-span model admission; legacy common guards stay strict."""
import functools
import torch
from .model import Config, build as common_build
from .geometry import validate_geometry
from ..research.sampling_span import (adapt_cpu_model, FinestSpanAttention,
    MSDeformableAttention, deformable_attention_core_func_v2)

ARMS = ('B0', 'B1', 'P')
VERSION = 'r18-sampling-span-v1'


def validate_sampling(model, arm):
    if arm not in ARMS:
        raise ValueError('unregistered sampling-span arm')
    validate_geometry(model)
    names = [n for n, m in model.named_modules() if isinstance(m, MSDeformableAttention)]
    expected = [f'decoder.decoder.layers.{i}.cross_attn' for i in range(3)]
    if names != expected:
        raise ValueError('sampling-span attention inventory differs')
    rows = []
    for name in names:
        m = model.get_submodule(name)
        points = [8, 2, 2] if arm == 'B1' else [4, 4, 4]
        cls = FinestSpanAttention if arm == 'P' else MSDeformableAttention
        f = m.ms_deformable_attn_core
        scale = torch.tensor([1/n for n in points for _ in range(n)], dtype=torch.float32)
        if (type(m) is not cls or m.num_points_list != points or m.num_levels != 3
                or m.num_heads != 8 or m.embed_dim != 256 or m.offset_scale != .5
                or m.method != 'default' or (arm == 'P' and m.span_enabled is not True)
                or not torch.equal(m.num_points_scale.cpu(), scale)
                or not isinstance(f, functools.partial)
                or f.func is not deformable_attention_core_func_v2
                or f.args or f.keywords != {'method': 'default'}
                or 'forward' in m.__dict__):
            raise ValueError('sampling-span class/core/config binding differs: '+name)
        rows.append(dict(name=name, type=cls.__module__+'.'+cls.__name__, points=points,
                         extent_floor_cells=1 if arm == 'P' else 0))
    return dict(version=VERSION, arm=arm, backend='deterministic_gather', modules=rows)


def build(config, arm, *, pretrained=None, synthetic=False):
    model, criterion, info = common_build(config, pretrained_path=pretrained, synthetic=synthetic)
    model = adapt_cpu_model(model, arm)
    info['sampling'] = validate_sampling(model, arm)
    info['sampling_span_version'] = VERSION
    return model, criterion, info
