"""R27-002 L0 control: the same H input as R27, no local branch (TASK_R27_L0_CONTROL_002 section 3).

The detector receives the R27/S15 H tensor I_H (1152x2016, augmented, [0,1]).  The model resamples it with the R27
``to_low`` expression (FP32, autocast disabled, on the model device) and then runs only the native L network:
PResNet18 backbone, hybrid encoder, Q300 three-layer decoder and the native losses.  Nothing of R27 is installed:
no local shallow backbone, projection, local read, selector or route keys; no new parameter, buffer, loss key,
query or target rule.  The state dict keeps the native key set, so the e060 parent loads strictly.  Targets are the
normalised boxes of the H pipeline and are not re-filtered for L; anchors / position caches use the L canvas.

L0 is its own production identity.  It is not the R27 N arm (native H / S15), not R27-P with the branch switched
off, and not the historical A0 input rendered directly from the original image to L.
"""
from __future__ import annotations

import copy
import hashlib
import json

import torch

from . import r27_local_memory as R
from .r27_local_memory import H_HW, L_HW, PARENT_SHA, PARENT_UPDATE, require, to_low

VERSION = "R27-002-L0-control-v1"
ARM = "L0"
CHECKPOINT_SEGMENTS = ("backbone.res_layers", "encoder.fpn_blocks", "encoder.pan_blocks")

IDENTITY = {
    "version": VERSION, "arm": ARM, "task": "TASK_R27_L0_CONTROL_002",
    "H": list(H_HW), "L": list(L_HW), "H_to_L": R.IDENTITY["H_to_L"],
    "forward": "native RTDETR.forward(r27_local_memory.to_low(I_H), targets): native L backbone, hybrid encoder, "
               "Q300 three-layer decoder, native losses",
    "local_backbone": None, "local_projection": None, "local_read": None, "selector": None, "route_keys": None,
    "new_learnable_parameters": 0,
    "state_dict": "native key set, equal to the e060 parent model key set; no added module, parameter or buffer",
    "targets": "normalised boxes produced by the R27/S15 H pipeline (same sampling, augmentation and filtering, "
               "including the declared boundary filtering); not re-filtered for L",
    "anchor_position_cache_canvas": list(L_HW),
    "activation_checkpoint_segments": list(CHECKPOINT_SEGMENTS), "local_recompute": None,
    "parent_sha256": PARENT_SHA, "parent_update": PARENT_UPDATE,
    "is_not": {"R27_N": "the R27 N arm is the native H / S15 engine (global network on the H canvas)",
               "R27_P_branch_off": "not R27-P with the local branch disabled; nothing of R27 is installed",
               "A0_direct_L": "not the historical A0 input rendered directly from the original image to L; the "
                              "input is the augmented H tensor resampled by to_low inside the model"},
}
IDENTITY_SHA256 = hashlib.sha256(json.dumps(IDENTITY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

_TYPES = {}


def _forward(self, x, targets=None):
    """I_H -> to_low -> the unchanged native forward."""
    return self._l0_native.forward(self, to_low(x), targets)


def is_wrapped(model):
    """True only for the exact L0 class: a later class swap on top (for example an R27 install) is not L0."""
    cls = type(model)
    return (cls.__dict__.get("_l0_native") is not None and cls.__dict__.get("forward") is _forward
            and "forward" not in vars(model) and not hasattr(model, "r27_ctx"))


def _layout(model):
    return {"state": list(model.state_dict()), "parameters": [n for n, _ in model.named_parameters()],
            "buffers": [n for n, _ in model.named_buffers()], "modules": [n for n, _ in model.named_modules()]}


def wrap(model):
    """Swap the forward of a native RT-DETRv2-R18 built at L, in place; nothing is added to the module tree."""
    require(getattr(type(model), "_l0_native", None) is None, "L0 already wrapped")
    require(not any(hasattr(m, "r27_ctx") for m in model.modules()) and not any(R.is_new(k) for k in model.state_dict()),
            "L0 refuses a model with R27 installed")
    native = type(model)
    require(native.__name__ == "RTDETR", f"L0 expects the native RTDETR class, got {native.__name__}")
    ident = R._native_identity(model)
    before = _layout(model)
    if native not in _TYPES:  # same class swap as r27_local_memory._swap, with L0's own marker
        _TYPES[native] = type("L0RTDETR", (native,), {"forward": _forward, "_l0_native": native, "l0_arm": ARM})
    model.__class__ = _TYPES[native]
    require(_layout(model) == before and is_wrapped(model), "L0 wrap changed the module layout")
    return {"arm": ARM, "native": ident, "new_parameters": 0, "state_keys": len(before["state"]),
            "parameter_tensors": len(before["parameters"]), "buffers": len(before["buffers"])}


def _settings(group):
    return {k: v for k, v in group.items() if k != "params"}


# ---------------------------------------------------------------- engine construction from the e060 parent
def create(parent, device, *, fixture=False, activation_checkpoint=True):
    """The R27 ``create`` path without ``install`` and without the local recompute: global L network from the
    native e060 complete state, online and EMA wrapped, engine at the S15 H recipe (or the CPU H fixture)."""
    from . import scale_baseline as S
    from ..protocol_r18 import checkpoint as native
    from ..protocol_r18.engine import Engine, learning_rate, tensor_digest
    from ..protocol_r18.geometry import validate_geometry
    from ..protocol_r18.model import CommonEMA, Config, build
    from ..baseline.training_v2b import _optimizer
    require(tuple(S.CHECKPOINT_SEGMENTS) == CHECKPOINT_SEGMENTS, "L0 activation checkpoint segments differ from S15")
    gcfg = Config(seed=0)
    ecfg = Config(seed=0, input_hw=H_HW, physical_batch_size=1, amp_dtype="float32") if fixture else S.production_config(0)
    if not fixture:
        S.validate_production(ecfg)
    require(native.file_sha(parent) == PARENT_SHA, "L0 parent identity")
    s = torch.load(parent, map_location="cpu", weights_only=True)
    require(s["engine"]["completed_updates"] == PARENT_UPDATE, "L0 parent clock")
    require(s["binding"]["config"] == gcfg.binding() and s["engine"].get("config", s["binding"]["config"]) == gcfg.binding(),
            "L0 parent recipe is not the common 768x1344 seed0 recipe")
    model, criterion, _ = build(gcfg, synthetic=True)
    require(native.old._model_layout(model) == s["model_layout"], "L0 global model layout differs from the parent")
    require(criterion.recipe() == s["behavior"]["criterion"], "L0 criterion behaviour differs")
    validate_geometry(model)
    model.load_state_dict(s["model"], strict=True)
    ema = CommonEMA(model)
    ema.load_state_dict(s["ema"])
    for n, m in model.named_modules():
        m.training = s["model_training"][n]
    info = wrap(model)
    wrap(ema.module)
    ema.module.requires_grad_(False)
    saved = {}
    for g in s["optimizer_layout"]["groups"]:
        for name, i in zip(g["names"], g["ids"]):
            require(name not in saved, "duplicate parent optimizer name")
            saved[name] = s["optimizer"]["state"].get(i, {})
    opt = _optimizer(model, ecfg)
    old_groups = s["optimizer"]["param_groups"]
    require([g["name"] for g in opt.param_groups] == [g["name"] for g in old_groups], "L0 optimizer group order")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    require(sorted(names) == sorted(saved) == sorted(n for g in opt.param_groups for n in g["param_names"]),
            "L0 optimizer coverage differs from the parent")
    for g, og in zip(opt.param_groups, old_groups):
        for k, v in og.items():
            if k not in ("params", "param_names"):
                g[k] = copy.deepcopy(v)
        for n, p in zip(g["param_names"], g["params"]):
            opt.state[p] = copy.deepcopy(saved[n])
    model.to(device)
    criterion.to(device)
    ema.module.to(device)
    for p, st in opt.state.items():
        for k, v in st.items():
            if isinstance(v, torch.Tensor) and k != "step":
                st[k] = v.to(p.device)
    e = Engine(model, criterion, opt, ema, ecfg)
    e.completed_updates = s["engine"]["completed_updates"]
    old_steps = s["engine"]["parameter_optimizer_steps"]
    require(set(old_steps) == set(e.parameter_steps), "L0 parameter step coverage")
    e.parameter_steps = {n: old_steps[n] for n in e.parameter_steps}
    e.l0_arm = ARM
    # scale_baseline.capture/restore copy these two into the saved "arm" / "parent_sha256" fields and refuse a state
    # of another arm.  "L0" is none of S15 (R27 N), R27-C, R27-P or A0; the parent SHA is the verified file fact.
    e.r25_arm = ARM
    e.r25_parent = PARENT_SHA
    wrapped = S.install_activation_checkpoint(model) if activation_checkpoint else []
    e.s15_activation_checkpoint = bool(activation_checkpoint)
    validate_geometry(model)
    validate_geometry(ema.module)
    named = dict(model.named_parameters())
    sd, esd = model.state_dict(), ema.module.state_dict()
    marked = sorted(n for n, m in model.named_modules() if getattr(m, "_s15_checkpointed", False))
    checks = {
        "online_exact": all(torch.equal(sd[n].cpu(), v) for n, v in s["model"].items()),
        "EMA_exact": all(torch.equal(esd[n].cpu(), v) for n, v in s["ema"]["module"].items()),
        "optimizer_exact": len(opt.state) == len(saved) and all(
            native.exact_equal(S._cpu(opt.state[named[n]]), S._cpu(v)) for n, v in saved.items()),
        "optimizer_groups_exact": len(opt.param_groups) == len(old_groups) and all(
            native.exact_equal(_settings(g), _settings(og)) for g, og in zip(opt.param_groups, old_groups)),
        "parameter_steps_exact": e.parameter_steps == old_steps,
        "model_keys_equal_parent": set(sd) == set(s["model"]) and len(sd) == len(s["model"]),
        "EMA_keys_equal_parent": set(esd) == set(s["ema"]["module"]) and len(esd) == len(s["ema"]["module"]),
        "wrapped": {"online": is_wrapped(model), "EMA": is_wrapped(ema.module)},
        "activation_checkpoint_exact": marked == sorted(wrapped) and {w.rsplit(".", 1)[0] for w in wrapped} == (
            set(CHECKPOINT_SEGMENTS) if activation_checkpoint else set()),
        "new_parameters": sum(p.numel() for n, p in named.items() if n not in s["model"]),
        "new_parameter_names": [n for n in named if n not in s["model"]],
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_parameters": len(saved),
        "old_parameters_sha256": tensor_digest(model.named_parameters()),
        "activation_checkpoint_segments": wrapped, "local_recompute_segments": [],
        "engine_config": ecfg.binding(), "global_model_config": gcfg.binding(),
        "identity_sha256": IDENTITY_SHA256, "parent_sha256": PARENT_SHA,
        "EMA_updates": ema.updates, "completed_updates": e.completed_updates,
        "lr_next": learning_rate(e.completed_updates + 1), "r25_arm": e.r25_arm, **info,
    }
    require(all(checks[k] for k in ("online_exact", "EMA_exact", "optimizer_exact", "optimizer_groups_exact",
                                    "parameter_steps_exact", "model_keys_equal_parent", "EMA_keys_equal_parent",
                                    "activation_checkpoint_exact")) and all(checks["wrapped"].values())
            and checks["new_parameters"] == 0 and not checks["new_parameter_names"], ("L0 bridge mismatch", checks))
    require(ema.updates == e.completed_updates == PARENT_UPDATE, "L0 clocks differ")
    require(not hasattr(e, "r27_arm") and not hasattr(model, "r27_ctx"), "L0 engine carries an R27 arm")
    rng = dict(s["rng"])
    if device == "cpu":
        rng["cuda"] = None
    native.restore_rng(rng, e.device)
    return e, checks


def build_eval_model(state, device):
    """FP32 evaluation model from a saved module state (EMA or online); native key set only; no activation checkpoint."""
    from ..protocol_r18.engine import tensor_digest
    from ..protocol_r18.geometry import validate_geometry
    from ..protocol_r18.model import Config, build
    model, _, _ = build(Config(seed=0), synthetic=True)
    wrap(model)
    own = model.state_dict()
    extra, missing = sorted(set(state) - set(own)), sorted(set(own) - set(state))
    require(not extra and not missing, ("L0 evaluation state is not the native key set", extra[:4], missing[:4]))
    model.load_state_dict(state, strict=True)
    validate_geometry(model)
    model.eval().requires_grad_(False).to(device)
    require(tensor_digest(model.state_dict().items()) == tensor_digest(state.items()), "L0 evaluation strict load differs")
    return model
