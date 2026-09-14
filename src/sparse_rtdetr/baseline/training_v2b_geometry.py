"""Observed RT-DETR evaluation geometry, including ordinary tensor caches.

The snapshot is device independent so CPU construction and placed models share
the same geometry identity. Raw and EMA snapshots remain separate: vendor EMA
updates floating persistent anchors, whose saved rounding must be preserved.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn


class ModelGeometryError(ValueError):
    """Live model attributes or cached tensors disagree with the input geometry."""


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ModelGeometryError(f"{name} must be an integer >= {minimum}")
    return value


def _spatial_size(value: Any, name: str) -> list[int]:
    if type(value) is int:
        value = [value, value]
    if type(value) not in (tuple, list) or len(value) != 2:
        raise ModelGeometryError(f"{name} must contain height and width")
    return [_integer(dimension, name) for dimension in value]


def _strides(value: Any, name: str) -> list[int]:
    if type(value) not in (tuple, list) or not value:
        raise ModelGeometryError(f"{name} must be a nonempty stride sequence")
    strides = [_integer(stride, name) for stride in value]
    if len(set(strides)) != len(strides):
        raise ModelGeometryError(f"{name} contains duplicate strides")
    return strides


def _tensor(value: Any, name: str, shape: tuple[int, ...],
            dtype: torch.dtype) -> tuple[dict, torch.Tensor]:
    if not isinstance(value, torch.Tensor):
        raise ModelGeometryError(f"{name} must be a tensor")
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise ModelGeometryError(f"{name} cache shape/dtype mismatch")
    host = value.detach().cpu().contiguous()
    raw = host.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(host.shape), "dtype": str(host.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }, host


def snapshot_model_geometry(model: nn.Module, *,
                            expected_input_size: int | tuple[int, int] | list[int]) -> dict:
    """Validate and hash actual encoder/decoder attributes and cache values.

    This does not rebuild caches, mutate a model, consume RNG, or compare EMA
    floating anchors with the raw model. Checkpoint restore owns saved EMA bytes.
    """
    expected = _spatial_size(expected_input_size, "expected_input_size")
    if not isinstance(model, nn.Module):
        raise ModelGeometryError("model must be a torch module")
    encoder, decoder = getattr(model, "encoder", None), getattr(model, "decoder", None)
    if not isinstance(encoder, nn.Module) or not isinstance(decoder, nn.Module):
        raise ModelGeometryError("RT-DETR encoder and decoder modules are required")
    encoder_size = _spatial_size(getattr(encoder, "eval_spatial_size", None),
                                 "encoder.eval_spatial_size")
    decoder_size = _spatial_size(getattr(decoder, "eval_spatial_size", None),
                                 "decoder.eval_spatial_size")
    if encoder_size != expected or decoder_size != expected:
        raise ModelGeometryError("encoder/decoder eval_spatial_size mismatch")
    encoder_strides = _strides(getattr(encoder, "feat_strides", None), "encoder.feat_strides")
    decoder_strides = _strides(getattr(decoder, "feat_strides", None), "decoder.feat_strides")
    if encoder_strides != decoder_strides:
        raise ModelGeometryError("encoder/decoder feature strides differ")
    if any(dimension % stride for dimension in expected for stride in encoder_strides):
        raise ModelGeometryError("input dimensions must be divisible by every feature stride")
    encoder_hidden = _integer(getattr(encoder, "hidden_dim", None), "encoder.hidden_dim")
    decoder_hidden = _integer(getattr(decoder, "hidden_dim", None), "decoder.hidden_dim")
    if encoder_hidden != decoder_hidden:
        raise ModelGeometryError("encoder/decoder hidden dimensions differ")
    indices = getattr(encoder, "use_encoder_idx", None)
    if type(indices) not in (tuple, list) or not indices:
        raise ModelGeometryError("encoder.use_encoder_idx must be nonempty")
    indices = [_integer(index, "encoder.use_encoder_idx", 0) for index in indices]
    if len(indices) != len(set(indices)) or any(index >= len(encoder_strides) for index in indices):
        raise ModelGeometryError("encoder.use_encoder_idx is out of range or duplicated")
    levels = _integer(getattr(decoder, "num_levels", None), "decoder.num_levels")
    if levels != len(decoder_strides):
        raise ModelGeometryError("decoder level count differs from feature strides")

    positions = {}
    for index in indices:
        name = f"pos_embed{index}"
        stride = encoder_strides[index]
        shape = (1, (expected[0] // stride) * (expected[1] // stride), encoder_hidden)
        if name in encoder._parameters or name in encoder._buffers:
            raise ModelGeometryError(f"encoder.{name} must remain the vendor ordinary tensor cache")
        reference, values = _tensor(getattr(encoder, name, None), "encoder." + name,
                                    shape, torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise ModelGeometryError(f"encoder.{name} contains nonfinite values")
        positions[name] = reference
    count = sum((expected[0] // stride) * (expected[1] // stride)
                for stride in decoder_strides)
    for name in ("anchors", "valid_mask"):
        if name not in decoder._buffers or name in decoder._non_persistent_buffers_set:
            raise ModelGeometryError(f"decoder.{name} must remain a persistent buffer")
    anchors, anchor_values = _tensor(getattr(decoder, "anchors", None), "decoder.anchors",
                                     (1, count, 4), torch.float32)
    valid_mask, mask_values = _tensor(getattr(decoder, "valid_mask", None), "decoder.valid_mask",
                                      (1, count, 1), torch.bool)
    mask = mask_values.expand_as(anchor_values)
    if not bool(torch.isfinite(anchor_values[mask]).all()):
        raise ModelGeometryError("valid decoder anchors must be finite")
    if not bool(torch.isposinf(anchor_values[~mask]).all()):
        raise ModelGeometryError("invalid decoder anchors require the vendor +inf sentinel")
    return {
        "schema_version": 1,
        "input_size": expected,
        "encoder": {
            "eval_spatial_size": encoder_size, "feat_strides": encoder_strides,
            "hidden_dim": encoder_hidden, "use_encoder_idx": indices,
            "position_caches": positions,
        },
        "decoder": {
            "eval_spatial_size": decoder_size, "feat_strides": decoder_strides,
            "hidden_dim": decoder_hidden, "num_levels": levels,
            "anchors": anchors, "valid_mask": valid_mask,
        },
    }
