"""Sub-stride detail path for the common-protocol R18 (r10 execution-round prototype).

Mechanism (H-beta of the r10 screening; see the single exploration record
``delivery/research/r18-explore-20261031-001/EXPLORATION_R1.md``): on the frozen
carrier every ground-truth instance's best encoder candidate lives at level 0
(stride 8), the pool coverage at IoU 0.75 for raw-area <= 1024 objects is 0.421,
and the native model discards the stride-4 backbone stage entirely.  This module
exposes that stage as a parameter-light, inference-retained side path.

Exact insertion point: inside the decoder's ``_get_encoder_input``, on the
level-0 branch, AFTER the native 1x1 ``input_proj[0]`` projection and BEFORE the
per-level flatten/concat into the encoder token sequence:

    proj_feats[0] = proj_feats[0] + gate * conv1x1(pixel_unshuffle(stride4))

It is NOT inside the backbone (the backbone is only tapped by a key-preserving
wrapper whose forward mirrors ``PResNet.forward``), NOT a new pyramid level, NOT
a new head/anchor, and it does not touch the query construction.

Relation to strided convolutions: ``pixel_unshuffle(x, 2)`` followed by a 1x1
linear map (64ch -> 4x64ch -> 256ch here) is algebraically identical to a single
2x2 stride-2 convolution applied to the stride-4 tensor, i.e. the operator is
the space-to-depth + linear primitive family already used by the cited
precedents (SPD-Conv style downsampling).  This round claims NO originality and
performed no new literature sweep; the declared differences to the nearest
precedents are only: the tap point (the stride-4 stage output the native model
otherwise discards), the insertion interface (encoder input construction, after
the native projection, detection protocol unchanged), the zero-initialized gate
(bitwise identity at step 0), and the capacity-matched control arm.

Gradient definition (fixed mechanism of this round, NOT a save-state
workaround): the branch input is stop-gradient in BOTH modes.  The main path
(``proj_feats[0]`` as consumed by the encoder) trains normally; the branch
trains only ``gate`` and ``conv``.  Mode ``detail`` uses the stride-4 capture
(detached at capture so it never holds a graph); mode ``control`` uses
``proj_feats[0].detach()`` (the same tensor, a stop-gradient copy, so no
sub-stride information enters).  A forward-time guard enforces
``not source.requires_grad`` and ``LAST_SOURCE_INFO`` records the evidence for
the checks.  The wrapper stores its transient capture in a plain dict slot that
is consumed and cleared by the same forward (the frozen checkpoint format
digests plain tensor attributes, so the capture must not survive module state).

Installation happens *after* the frozen parent state is strict-loaded and before
the optimizer/EMA are constructed.  The 1x1 conv is initialized from an explicit
seeded generator: installation consumes NO global RNG draws, so the three arms
share identical data/augmentation/DN random streams.  The decoder's
``_get_encoder_input`` is replaced by a ``functools.partial`` bound to the
decoder instance; that is the only patch form whose self-reference survives
``copy.deepcopy`` (the EMA is a deep copy of the live model), and
``assert_installed`` verifies the identity links on both the live model and the
EMA after engine construction.

Scope notes: the native decoder class is intentionally NOT subclassed
(``validate_geometry`` requires the exact native type); no encoder attribute is
set; the wrapper adds no state-dict keys of its own.  CPU checks:
``test_detail_path.py`` and ``detail_smoke.py``.
"""
from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("detail", "control")
DETAIL_PARAMETERS = 256 * 256 + 256  # 1x1 conv weight + per-channel gate
LAST_SOURCE_INFO = {}  # evidence hook: mode / detach flags / shape of the branch input


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
    """gate * conv1x1(source); gate zero-initialized (identity at step 0).

    The 1x1 conv is xavier-uniform initialized from an explicit seeded
    generator (equivalent bound sqrt(6/(fan_in+fan_out)), gain 1), so installing
    the module consumes no global RNG draws and both installed arms receive
    identical new-module initial values.
    """

    def __init__(self, channels=256, seed=0):
        super().__init__()
        # The nn.Conv2d constructor itself draws from the global RNG; bracket it
        # with a save/restore so installation is fully RNG-neutral (the actual
        # weights come from the explicit seeded generator below).
        rng_state = torch.random.get_rng_state()
        try:
            self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        finally:
            torch.random.set_rng_state(rng_state)
        bound = math.sqrt(6.0 / (channels + channels))
        with torch.no_grad():
            self.conv.weight.uniform_(-bound, bound, generator=torch.Generator().manual_seed(seed))
        self.gate = nn.Parameter(torch.zeros(channels))


def _encoder_input(decoder, mode, feats):
    """Native ``_get_encoder_input`` extended with the level-0 injection.

    Projection/flatten arithmetic mirrors
    ``src.zoo.rtdetr.rtdetrv2_decoder.RTDETRTransformerv2._get_encoder_input``
    verbatim; the injection line and the stop-gradient definition are the only
    additions.
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
        pre = bool(source.requires_grad)          # False by construction: detached at capture
        source = F.pixel_unshuffle(source, 2).detach()
    else:
        pre = bool(proj_feats[0].requires_grad)   # True on a live training forward
        source = proj_feats[0].detach()
    LAST_SOURCE_INFO.clear()
    LAST_SOURCE_INFO.update(mode=mode, source_requires_grad_before_branch_detach=pre,
                            source_requires_grad=bool(source.requires_grad),
                            shape=list(source.shape))
    require(not source.requires_grad, "branch source must be stop-gradient")
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


def install_detail_path(model, mode, seed=0):
    """Install the capture wrapper and the injection path; returns the parameter delta."""
    require(mode in MODES, "unknown detail mode: " + str(mode))
    require(type(model.backbone).__name__ == "PResNet", "unexpected backbone type")
    require(not isinstance(model.backbone, BackboneCapture), "already installed")
    decoder = model.decoder
    require(not hasattr(decoder, "detail_inject"), "detail path already installed")
    before = sum(p.numel() for p in model.parameters())
    wrapper = BackboneCapture(model.backbone)
    model.backbone = wrapper
    decoder.detail_inject = DetailInjection(seed=seed)
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
