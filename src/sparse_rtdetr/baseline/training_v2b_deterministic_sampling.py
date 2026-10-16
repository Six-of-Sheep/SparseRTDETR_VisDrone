"""Unconnected, deterministic candidate for RT-DETRv2 bilinear sampling.

The only supported geometry is bilinear / zeros / align_corners=False. Four
explicit ``gather`` operations preserve that mathematical function and both
input derivatives; this is not discrete attention and does not detach grids.
Native CUDA grid-sampler backward uses atomic accumulation. PyTorch 2.4.1
documents a deterministic CUDA backward for ``gather`` when global deterministic
algorithms are enabled. CUDA use therefore requires that flag, without warn-only
mode. This module never enables flags, initializes CUDA, or changes model wiring.

Only FP32/FP64 computation is currently supported. CPU autocast's FP32 rule and
CUDA autocast's promote rule are explicit; mixed BF16/FP32 inputs are cast *before*
the four gathers so gradient accumulation occurs in the native operator dtype.
All-low-precision CUDA calls remain unsupported pending separate validation.
Inputs are expected to be finite, as required by the surrounding training gate.
Floating-point association may differ from native sampling. Mathematical and
CPU numerical agreement do not establish GPU repeatability or exact BN replay.
"""
from __future__ import annotations

import torch


class DeterministicSamplingError(ValueError):
    """The candidate's explicitly validated operator scope was exceeded."""


_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _sampling_dtype(value_dtype: torch.dtype, grid_dtype: torch.dtype, *,
                    device_type: str, autocast_enabled: bool,
                    autocast_dtype: torch.dtype | None) -> torch.dtype:
    """Pure dtype policy; inspecting it never calls a CUDA API."""
    if value_dtype not in _FLOAT_DTYPES or grid_dtype not in _FLOAT_DTYPES:
        raise DeterministicSamplingError("sampling requires real floating tensors")
    if device_type not in {"cpu", "cuda"}:
        raise DeterministicSamplingError("only explicit CPU/CUDA sampling is supported")
    # FP64 is not autocast-eligible. Mixing it with another dtype is not a
    # permitted way to broaden native grid_sample's same-dtype contract.
    if torch.float64 in {value_dtype, grid_dtype}:
        if value_dtype != grid_dtype:
            raise DeterministicSamplingError("FP64 sampling inputs must have the same dtype")
        return torch.float64
    if not autocast_enabled:
        if value_dtype != grid_dtype:
            raise DeterministicSamplingError("sampling inputs must have the same dtype outside autocast")
        return value_dtype
    if device_type == "cpu":
        return torch.float32  # Native CPU grid_sampler autocasts to FP32.
    if autocast_dtype not in {torch.float16, torch.bfloat16}:
        raise DeterministicSamplingError("unsupported CUDA autocast lower-precision dtype")
    if any(dtype not in {torch.float32, autocast_dtype} for dtype in (value_dtype, grid_dtype)):
        raise DeterministicSamplingError("input dtype is outside native CUDA autocast promotion scope")
    return torch.float32 if torch.float32 in {value_dtype, grid_dtype} else autocast_dtype


def bilinear_grid_sample(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Sample NCHW values at NHW2 grids without changing backend or AMP state.

    The result is NCHW with the grid's output spatial dimensions. Native's
    ``((g + 1) * size - 1) / 2`` coordinate expression is retained instead of
    algebraically reassociating it. Invalid neighbors contribute zero, while
    in-range neighbors retain gradients for both ``value`` and ``grid``.
    """
    if not isinstance(value, torch.Tensor) or not isinstance(grid, torch.Tensor):
        raise DeterministicSamplingError("value and grid must be tensors")
    if value.ndim != 4 or grid.ndim != 4 or grid.shape[-1] != 2:
        raise DeterministicSamplingError("expected NCHW value and NHW2 grid")
    if value.shape[0] != grid.shape[0] or value.shape[2] <= 0 or value.shape[3] <= 0:
        raise DeterministicSamplingError("sampling batch or input spatial shape differs")
    if value.device != grid.device:
        raise DeterministicSamplingError("sampling tensors must share the same device")
    device_type = value.device.type
    if device_type not in {"cpu", "cuda"}:
        raise DeterministicSamplingError("only explicit CPU/CUDA sampling is supported")
    autocast_enabled = torch.is_autocast_enabled(device_type)
    dtype = _sampling_dtype(value.dtype, grid.dtype, device_type=device_type,
                            autocast_enabled=autocast_enabled,
                            autocast_dtype=torch.get_autocast_dtype(device_type) if autocast_enabled else None)
    if dtype not in {torch.float32, torch.float64}:
        raise DeterministicSamplingError("candidate supports only FP32/FP64 operator computation")
    if device_type == "cuda" and (not torch.are_deterministic_algorithms_enabled()
                                  or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise DeterministicSamplingError("CUDA candidate requires deterministic algorithms with warn_only=False")
    n, channels, height, width = value.shape
    out_height, out_width = grid.shape[1:3]
    with torch.autocast(device_type=device_type, enabled=False):
        values = value.to(dtype=dtype).reshape(n, channels, height * width)
        coordinates = grid.to(dtype=dtype)
        # For any spatial size S >= 1, bilinear zeros support (including
        # one-sided boundary derivatives) is contained in normalized
        # [-1 - 1/S, 1 + 1/S], hence in [-2, 2]. A finite point with either
        # coordinate outside [-4, 4] is strictly outside that support. Replace
        # the whole pair *before* scaling: finite extreme coordinates can
        # otherwise overflow and produce NaN gradients through masked 0 * NaN.
        # The wider guard preserves native rounding at all support boundaries.
        # Nonfinite coordinates remain outside the supported input contract;
        # this finite guard must not silently sanitize them.
        far_outside = torch.isfinite(coordinates).all(dim=-1) & (coordinates.abs() > 4).any(dim=-1)
        active = ~far_outside
        coordinates = torch.where(active.unsqueeze(-1), coordinates, 0)
        x = ((coordinates[..., 0] + 1) * width - 1) / 2
        y = ((coordinates[..., 1] + 1) * height - 1) / 2
        x0, y0 = torch.floor(x), torch.floor(y)
        x1, y1 = x0 + 1, y0 + 1
        neighbors = (
            (x0, y0, (x1 - x) * (y1 - y)),
            (x1, y0, (x - x0) * (y1 - y)),
            (x0, y1, (x1 - x) * (y - y0)),
            (x1, y1, (x - x0) * (y - y0)),
        )
        output = None
        for ix, iy, weight in neighbors:
            valid = active & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            # Mask before integer conversion; far-out coordinates cannot create
            # invalid gather indices. The mask also preserves zeros padding.
            safe_x = torch.where(valid, ix, 0).to(torch.int64)
            safe_y = torch.where(valid, iy, 0).to(torch.int64)
            indices = (safe_y * width + safe_x).reshape(n, 1, -1).expand(n, channels, -1)
            sampled = torch.gather(values, 2, indices).reshape(n, channels, out_height, out_width)
            summand = torch.where(valid.unsqueeze(1), sampled * weight.unsqueeze(1), 0)
            output = summand if output is None else output + summand
        return output


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: list[int],
    method="default",
) -> torch.Tensor:
    """Unconnected RT-DETRv2 default attention core with candidate sampling.

    Keep the vendored v2 permute/flatten/split/concat/multiply/sum expressions,
    including their order and enclosing autocast behavior. Only its bilinear
    sampling call is replaced. Value, locations and attention weights all keep
    their gradients. Discrete sampling is deliberately outside this scope.
    """
    if method != "default":
        raise DeterministicSamplingError("candidate attention core requires method='default'")
    bs, _, n_head, c = value.shape
    _, Len_q, _, _, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_locations_list[level]
        sampling_value_l = bilinear_grid_sample(value_l, sampling_grid_l)
        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)
    return output.permute(0, 2, 1)
