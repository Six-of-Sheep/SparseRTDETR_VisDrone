"""Sub-stride detail path for the common-protocol R18 (r10 exploration prototype).

Mechanism (H-beta of the r10 screening; see
``delivery/research/r18-explore-20261031-001/EXPLORATION_R1.md``): on the frozen
carrier every ground-truth instance's best encoder candidate lives at level 0
(stride 8), the pool coverage at IoU 0.75 for raw-area <= 1024 objects is 0.421,
and the native model discards the stride-4 backbone stage entirely.  This module
restores that detail as a parameter-light, inference-retained path:

  * ``BackboneCapture`` wraps ``model.backbone`` (PResNet), transplanting its
    children so every state-dict key keeps its native name, and records the
    stride-4 stage output during the forward in a layout-invisible capture
    slot that is consumed and cleared by the same forward (the frozen
    checkpoint format digests plain tensor attributes, so the capture must not
    survive module state);
  * the stride-4 tensor is folded to the stride-8 grid with
    ``F.pixel_unshuffle(x, 2)`` (64 ch x 2 x 2 -> 256 ch) and injected into the
    decoder's finest level after its native 1x1 projection:
    ``proj_feats[0] + gate * conv1x1(source)`` with ``gate`` initialized to
    zero, so at step 0 the installed model is functionally identical to the
    native one;
  * mode ``detail`` uses the pixel-unshuffled stride-4 source; mode ``control``
    uses the projected level-0 feature itself (identical module, identical
    parameter count and grid, no additional information), so a trained
    comparison isolates "sub-stride detail" from "extra capacity".

Installation happens *after* the frozen state is strict-loaded and before the
optimizer/EMA are constructed (new state-dict keys only exist on installed
arms).  The decoder's ``_get_encoder_input`` is replaced by a
``functools.partial`` bound to the decoder instance; that is the only patch
form whose self-reference survives ``copy.deepcopy`` (the EMA is a deep copy
of the live model), and ``assert_installed`` verifies the identity links on
both the live model and the EMA after engine construction.

Scope notes: the native decoder class is intentionally NOT subclassed
(``protocol_r18.geometry.validate_geometry`` requires the exact native type);
no encoder attribute is set; the wrapper adds no state-dict keys of its own.
CPU checks: ``test_detail_path.py`` and ``detail_smoke.py``.
"""
from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("detail", "control")
DETAIL_PARAMETERS = 256 * 256 + 256  # 1x1 conv weight + per-channel gate


def require(condition, message):
    if not condition:
        raise ValueError(message)


class BackboneCapture(nn.Module):
    """PResNet wrapper: transplanted children (native key names) + stride-4 capture.

    The forward mirrors ``PResNet.forward`` verbatim (conv1 -> max_pool2d -> the
    four ``res_layers`` stages) and additionally records the stage-0 output
    (the stride-4 feature) into ``_detail_capture``.  The recorded tensor is
    value-only (``detach``): the injection conv/gate still train, the plain
    dict attribute stays invisible to the checkpoint model-layout digest and
    deepcopy-safe, and no buffer/parameter is added.  The capture is consumed
    and cleared by the decoder's encoder-input step of the same forward.
    """

    def __init__(self, inner):
        super().__init__()
        for name, value in list(vars(inner).items()):
            if name.startswith("_") or name == "training" or isinstance(value, nn.Module):
                continue
            object.__setattr__(self, name, value)
        self.conv1 = inner.conv1
        self.res_layers = inner.res_layers
        self.return_idx = list(inner.return_idx)
        self.out_channels = list(inner.out_channels)
        self.out_strides = list(inner.out_strides)
        self._detail_capture = {}

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx == 0:
                self._detail_capture["s4"] = x.detach()
            if idx in self.return_idx:
                outs.append(x)
        return outs


class DetailInjection(nn.Module):
    """gate * conv1x1(source); gate zero-initialized (identity at step 0)."""

    def __init__(self, channels=256):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)
        self.gate = nn.Parameter(torch.zeros(channels))


def _encoder_input(decoder, mode, feats):
    """Native ``_get_encoder_input`` extended with the level-0 injection.

    Projection/flatten arithmetic mirrors
    ``src.zoo.rtdetr.rtdetrv2_decoder.RTDETRTransformerv2._get_encoder_input``
    verbatim; the injection line is the only addition.
    """
    proj_feats = [decoder.input_proj[i](feat) for i, feat in enumerate(feats)]
    if decoder.num_levels > len(proj_feats):
        len_srcs = len(proj_feats)
        for i in range(len_srcs, decoder.num_levels):
            if i == len_srcs:
                proj_feats.append(decoder.input_proj[i](feats[-1]))
            else:
                proj_feats.append(decoder.input_proj[i](proj_feats[-1]))
    if mode == "detail":
        capture = decoder.reference_backbone._detail_capture
        source = capture.get("s4")
        require(source is not None,
                "stride-4 capture is empty; the backbone forward must run first")
        require(source.dim() == 4
                and source.shape[-2] == proj_feats[0].shape[-2] * 2
                and source.shape[-1] == proj_feats[0].shape[-1] * 2,
                "stride-4 grid is not exactly twice the finest-level grid")
        capture["s4"] = None
        source = F.pixel_unshuffle(source, 2)
    else:
        source = proj_feats[0]
    injection = decoder.detail_inject
    proj_feats[0] = proj_feats[0] + injection.gate.view(1, -1, 1, 1) * injection.conv(source)
    feat_flatten = []
    spatial_shapes = []
    for feat in proj_feats:
        _, _, h, w = feat.shape
        feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
        spatial_shapes.append([h, w])
    feat_flatten = torch.concat(feat_flatten, 1)
    return feat_flatten, spatial_shapes


def install_detail_path(model, mode):
    """Install the capture wrapper and the injection path; returns the parameter delta."""
    require(mode in MODES, "unknown detail mode: " + str(mode))
    require(type(model.backbone).__name__ == "PResNet", "unexpected backbone type")
    require(not isinstance(model.backbone, BackboneCapture), "already installed")
    decoder = model.decoder
    require(not hasattr(decoder, "detail_inject"), "detail path already installed")
    before = sum(p.numel() for p in model.parameters())
    wrapper = BackboneCapture(model.backbone)
    model.backbone = wrapper
    decoder.detail_inject = DetailInjection()
    object.__setattr__(decoder, "reference_backbone", wrapper)
    decoder._get_encoder_input = functools.partial(_encoder_input, decoder, mode)
    after = sum(p.numel() for p in model.parameters())
    require(after - before == DETAIL_PARAMETERS, "unexpected detail parameter delta")
    return after - before


def assert_installed(model, mode):
    """Verify the identity links survived construction and ``copy.deepcopy``."""
    decoder = model.decoder
    require(hasattr(decoder, "detail_inject"), "detail path missing")
    bound = decoder._get_encoder_input
    require(isinstance(bound, functools.partial), "encoder-input binding is not a partial")
    require(bound.args[0] is decoder, "encoder-input binding does not reference this decoder copy")
    require(bound.args[1] == mode, "detail mode binding differs")
    require(getattr(decoder, "reference_backbone", None) is model.backbone,
            "decoder does not reference its own backbone copy")
    require(isinstance(model.backbone, BackboneCapture), "backbone wrapper missing")
    return True
