"""S15: native RT-DETRv2-R18 continued from the R25 e060 state on a 1152x2016 train/eval canvas.

Zero new modules or parameters.  The only scientific change against the R25 A0 arm is the
canvas, which is part of the behaviour identity (``Config.binding()['input_hw']``).  Size
dependent caches (decoder anchors/valid_mask buffers and encoder position caches) are never
copied from the 768x1344 parent: the parent copies are first proven to be the exact native
768x1344 regeneration, then the S15 model keeps its own native 1152x2016 regeneration.

Optional ``activation_checkpoint`` is a memory implementation only: it recomputes selected
backbone/encoder blocks during backward and restores BatchNorm running statistics around the
recomputation so that every BN updates exactly once per forward, as without checkpointing.
It is off by default.  CPU off/on equality (prep-cpu-003) and the TASK_S15_EXECUTE_001 CUDA
complete-state acceptance (768x1344 A0 off/on) must both pass before formal use.
"""
from __future__ import annotations

import contextlib
import copy
import functools
import os
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from ..protocol_r18 import checkpoint as native
from ..protocol_r18.data import BoundDataset
from ..protocol_r18.engine import Engine, learning_rate, tensor_digest
from ..protocol_r18.geometry import geometry_identity, validate_geometry
from ..protocol_r18.model import CommonEMA, Config, build
from ..baseline.training_v2b import _optimizer
from ..localization.data import RawAugmentation

VERSION = "S15-scale-baseline-v1"
ARM = "S15"
PARENT_SHA = "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
PARENT_UPDATE = 18240
PARENT_HW = (768, 1344)
S15_HW = (1152, 2016)
SIZE_CACHE_BUFFERS = ("anchors", "valid_mask")
CHECKPOINT_SEGMENTS = ("backbone.res_layers", "encoder.fpn_blocks", "encoder.pan_blocks")


def production_config():
    return Config(seed=0, input_hw=S15_HW)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_production(cfg):
    """Only the canvas may differ from the frozen common recipe."""
    require(type(cfg) is Config and cfg == production_config(), "S15 production recipe differs")
    parent = Config(seed=0).binding()
    mine = cfg.binding()
    require({k for k in parent if parent[k] != mine[k]} == {"input_hw"}, "S15 binding differs beyond canvas")


def binding(cfg, *, activation_checkpoint=False):
    return {"version": VERSION, "arm": ARM, "config": cfg.binding(), "parent_sha256": PARENT_SHA,
            "parent_input_hw": list(PARENT_HW), "size_caches": "native regeneration at config.input_hw",
            "activation_checkpoint": bool(activation_checkpoint),
            "checkpoint_segments": list(CHECKPOINT_SEGMENTS) if activation_checkpoint else []}


def _is_size_cache(key):
    return key.rsplit(".", 1)[-1] in SIZE_CACHE_BUFFERS


def _parent_level_shapes(model, hw):
    return [[hw[0] // s, hw[1] // s] for s in model.encoder.feat_strides]


def _layout_without_size_caches(layout):
    state = {k: v for k, v in layout["state"].items() if not _is_size_cache(k)}
    caches = {k: v for k, v in layout["caches"].items()
              if not (k.endswith(":persistent_cache:anchors") or k.endswith(":persistent_cache:valid_mask")
                      or k.split(":")[-1].startswith("pos_embed"))}
    return {"modules": layout["modules"], "state": state, "parameters": layout["parameters"], "caches": caches}


def _bridge_state(parent_state, model, label):
    """Prove the parent size caches are native 768x1344, then substitute native S15 caches."""
    d = model.decoder
    anchors, mask = d._generate_anchors(spatial_shapes=_parent_level_shapes(model, PARENT_HW))
    own = model.state_dict()
    keys = sorted(k for k in parent_state if _is_size_cache(k))
    require(keys == ["decoder.anchors", "decoder.valid_mask"], f"{label}: unexpected size caches {keys}")
    require(native.exact_equal(parent_state["decoder.anchors"], anchors)
            and native.exact_equal(parent_state["decoder.valid_mask"], mask),
            f"{label}: parent size cache is not the native 768x1344 regeneration")
    bridged = dict(parent_state)
    for k in keys:
        bridged[k] = own[k].detach().clone()
    require(set(bridged) == set(own), f"{label}: state keys differ")
    for k, v in bridged.items():
        require(v.shape == own[k].shape and v.dtype == own[k].dtype, f"{label}: shape/dtype differs {k}")
    return bridged, {"parent_anchor_tokens": int(parent_state["decoder.anchors"].shape[1]),
                     "s15_anchor_tokens": int(own["decoder.anchors"].shape[1]),
                     "parent_cache_native_768x1344": True}


# ---------------------------------------------------------------- activation checkpointing
# Block recomputation accounting (not model state).  ``timed`` adds device synchronisation
# around each recomputation; acceptance may enable it, formal training only counts calls.
RECOMPUTE = {"calls": 0, "seconds": 0.0, "timed": False}


def _sync():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


class _BNStatGuard:
    """Restore BN running buffers after recomputation: exactly one update per forward."""

    def __init__(self, module):
        self.bns = [m for m in module.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]

    def __enter__(self):
        RECOMPUTE["calls"] += 1
        if RECOMPUTE["timed"]:
            _sync()
            self.started = time.perf_counter()
        self.saved = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
                      if m.track_running_stats else None for m in self.bns]

    def __exit__(self, *exc):
        with torch.no_grad():
            for m, s in zip(self.bns, self.saved):
                if s is not None:
                    m.running_mean.copy_(s[0]); m.running_var.copy_(s[1]); m.num_batches_tracked.copy_(s[2])
        if RECOMPUTE["timed"]:
            _sync()
            RECOMPUTE["seconds"] += time.perf_counter() - self.started
        return False


def _wrap(module):
    original = module.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        if not (module.training and torch.is_grad_enabled()):
            return original(*args, **kwargs)
        return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True,
                          context_fn=lambda: (contextlib.nullcontext(), _BNStatGuard(module)), **kwargs)

    module.forward = forward
    module._s15_checkpointed = True


def install_activation_checkpoint(model):
    wrapped = []
    for path in CHECKPOINT_SEGMENTS:
        owner = model
        for part in path.split("."):
            owner = getattr(owner, part)
        for i, block in enumerate(owner):
            _wrap(block)
            wrapped.append(f"{path}.{i}")
    return wrapped


def remove_activation_checkpoint(model):
    for m in model.modules():
        if getattr(m, "_s15_checkpointed", False):
            del m.forward
            del m._s15_checkpointed


# ---------------------------------------------------------------- construction from the parent
def create(parent, device="cpu", config=None, *, fixture=False, activation_checkpoint=False):
    """Bridge the R25 e060 complete state into an S15 engine; no parameter is added or reset."""
    cfg = config or production_config()
    if fixture:
        require(cfg.input_hw == S15_HW and cfg != production_config(), "fixture must be S15 canvas")
    else:
        validate_production(cfg)
    s = torch.load(parent, map_location="cpu", weights_only=True)
    require(native.file_sha(parent) == PARENT_SHA and s["engine"]["completed_updates"] == PARENT_UPDATE,
            "S15 parent identity")
    require(s["binding"]["config"] == Config(seed=0).binding(), "S15 parent recipe is not the common 768x1344")
    model, criterion, info = build(cfg, synthetic=True)
    require(_layout_without_size_caches(native.old._model_layout(model)) ==
            _layout_without_size_caches(s["model_layout"]), "S15 native model layout differs beyond size caches")
    require(criterion.recipe() == s["behavior"]["criterion"], "S15 criterion behaviour differs")
    validate_geometry(model)
    state, cache_check = _bridge_state(s["model"], model, "online")
    model.load_state_dict(state, strict=True)
    opt = _optimizer(model, cfg)
    saved = {}
    for g in s["optimizer_layout"]["groups"]:
        for name, i in zip(g["names"], g["ids"]):
            require(name not in saved, "duplicate parent optimizer name")
            saved[name] = s["optimizer"]["state"].get(i, {})
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    require(set(named) == set(saved), "S15 optimizer named coverage")
    for n, p in named.items():
        opt.state[p] = copy.deepcopy(saved[n])
    for g, oldg in zip(opt.param_groups, s["optimizer"]["param_groups"]):
        require(g["param_names"] == oldg["param_names"], "S15 optimizer recipe/name order")
        for k, v in oldg.items():
            if k != "params":
                g[k] = copy.deepcopy(v)
    ema = CommonEMA(model)
    ema_state = dict(s["ema"])
    ema_state["module"], ema_cache_check = _bridge_state(s["ema"]["module"], ema.module, "EMA")
    ema.load_state_dict(ema_state)
    for n, m in model.named_modules():
        m.training = s["model_training"][n]
    model.to(device); criterion.to(device); ema.module.to(device)
    for param, st in opt.state.items():
        for k, v in st.items():
            if isinstance(v, torch.Tensor) and k != "step":
                st[k] = v.to(param.device)
    e = Engine(model, criterion, opt, ema, cfg)
    e.completed_updates = s["engine"]["completed_updates"]
    e.parameter_steps = {n: s["engine"]["parameter_optimizer_steps"][n] for n in e.parameter_steps}
    e.r25_arm = ARM
    e.r25_parent = PARENT_SHA
    e.s15_activation_checkpoint = bool(activation_checkpoint)
    wrapped = install_activation_checkpoint(model) if activation_checkpoint else []
    validate_geometry(model); validate_geometry(ema.module)
    checks = {
        "online_exact_except_size_caches": all(torch.equal(model.state_dict()[n].cpu(), v)
                                               for n, v in s["model"].items() if not _is_size_cache(n)),
        "EMA_exact_except_size_caches": all(torch.equal(ema.module.state_dict()[n].cpu(), v)
                                            for n, v in s["ema"]["module"].items() if not _is_size_cache(n)),
        "optimizer_exact": all(native.exact_equal(_cpu(opt.state[named[n]]), _cpu(v)) for n, v in saved.items()),
        "parameter_steps_exact": e.parameter_steps == s["engine"]["parameter_optimizer_steps"],
        "EMA_updates": ema.updates, "completed_updates": e.completed_updates,
        "lr_next": learning_rate(e.completed_updates + 1),
        "new_parameters": 0, "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_sha256": tensor_digest(model.named_parameters()),
        "online_size_cache": cache_check, "EMA_size_cache": ema_cache_check,
        "geometry": geometry_identity(model), "EMA_geometry": geometry_identity(ema.module),
        "activation_checkpoint_segments": wrapped,
    }
    require(all(checks[k] for k in ("online_exact_except_size_caches", "EMA_exact_except_size_caches",
                                    "optimizer_exact", "parameter_steps_exact")), ("S15 bridge mismatch", checks))
    require(ema.updates == e.completed_updates == PARENT_UPDATE, "S15 clocks differ")
    if device != "cpu":
        native.restore_rng(s["rng"], e.device)
    else:
        rng = dict(s["rng"]); rng["cuda"] = None; native.restore_rng(rng, e.device)
    return e, checks


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return copy.deepcopy(value)


def capture(e):
    require(not e.phase and not e.failed and all(p.grad is None for p in e.model.parameters()),
            "not a clean S15 update boundary")
    return _cpu({"version": VERSION, "arm": e.r25_arm, "parent_sha256": e.r25_parent,
                 "config": e.config.binding(), "activation_checkpoint": e.s15_activation_checkpoint,
                 "model": e.model.state_dict(), "model_modes": {n: m.training for n, m in e.model.named_modules()},
                 "optimizer": e.optimizer.state_dict(),
                 "optimizer_names": [g["param_names"] for g in e.optimizer.param_groups],
                 "ema": e.ema.state_dict(), "engine": e.state_dict(), "rng": native.capture_rng(e.device)})


def save(e, path):
    p = Path(path)
    payload = capture(e)
    with p.open("xb") as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": native.file_sha(p), "update": e.completed_updates}


def restore(e, path, sha):
    """Restore into an engine built for the same canvas; any other canvas/version is rejected."""
    require(native.file_sha(path) == sha, "S15 saved SHA")
    s = torch.load(path, map_location="cpu", weights_only=True)
    require((s["version"], s["arm"], s["parent_sha256"], s["config"]) ==
            (VERSION, e.r25_arm, e.r25_parent, e.config.binding()), "S15 cross-version/arm/canvas/recipe state")
    require(s["optimizer_names"] == [g["param_names"] for g in e.optimizer.param_groups], "S15 optimizer names")
    e.model.load_state_dict(s["model"], strict=True)
    e.optimizer.load_state_dict(s["optimizer"])
    e.ema.load_state_dict(s["ema"])
    for n, m in e.model.named_modules():
        m.training = s["model_modes"][n]
    e.load_state_dict(s["engine"])
    native.restore_rng(s["rng"], e.device)
    validate_geometry(e.model); validate_geometry(e.ema.module)
    mine = capture(e)
    mine["activation_checkpoint"] = s["activation_checkpoint"]  # implementation flag, not state
    require(native.exact_equal(mine, s), "S15 complete restored state not exact")
    return {"full_state_exact": True, "update": e.completed_updates}


# ---------------------------------------------------------------- data
def train_dataset(train_core_binding, hw=S15_HW):
    r = train_core_binding["roots"]
    return BoundDataset(role="train_core", manifest_path=train_core_binding["manifest"], images_path=r["images"],
                        annotations_path=r["annotations"], seed=0, root_binding=r, input_hw=hw)


def development(binding, hw=S15_HW):
    from ..r0.evaluation_data import Development548
    data = Development548(binding)
    data.transform = RawAugmentation(hw)
    return data
