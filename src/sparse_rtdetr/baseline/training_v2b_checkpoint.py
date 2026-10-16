"""Window-boundary checkpoints for the explicitly bound v2b engineering engine.

Only checkpoints returned by this module, with a trusted external SHA256, are
accepted. Deserialization always uses weights_only=True. No legacy training
checkpoint migration, automatic resume, or data access is provided.

The caller binding must declare initial_weights, data, code and config. Config
must include the declared per-group learning rates and all scheduler/warmup,
sampler and augmentation policies; state alone does not identify those policies.
For a mid-epoch resume, bind and supply explicit sampler/augmentation generators
or a deterministic cursor policy; saving RNG does not capture DataLoader workers.
The engine's public validator binds all execution policies, including the
mandatory BN backward-layout policy; legacy engine configurations are rejected.
Schema 2 binds an explicit prepared runtime and an acknowledged loader cursor.
Schema 1 remains readable only for CPU artifacts without a loader. CUDA requires
a previously admitted, seeded runtime; this module never implicitly initializes
CUDA. CPU inspection, save, and restore never call CUDA APIs.
The bound sampling inventory is the initialization snapshot: raw/EMA Python
callables must match it before capture or restoration, since state_dict does not
serialize them. Generic non-RT-DETR fixtures without that declaration remain valid.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 2
LEGACY_CPU_SCHEMA_VERSION = 1
FORMAT = "sparse_rtdetr.training_v2b_checkpoint"
_REQUIRED_BINDINGS = {"initial_weights", "data", "code", "config"}
_PAYLOAD_KEYS = {
    "format", "schema_version", "torch_version", "binding", "binding_sha256",
    "model", "model_layout", "model_training", "optimizer", "optimizer_layout",
    "ema", "ema_layout", "engine", "engine_type", "scheduler", "warmup", "rng",
    "runtime", "sampler",
}
_LEGACY_PAYLOAD_KEYS = _PAYLOAD_KEYS - {"runtime", "sampler"}


class CheckpointError(RuntimeError):
    """A checkpoint is unsafe, incompatible, incomplete, or cannot be published."""


def _fail(message: str) -> None:
    raise CheckpointError(message)


def _type_name(value: Any) -> str:
    cls = type(value)
    return cls.__module__ + "." + cls.__qualname__


def _json_value(value: Any, path: str = "binding") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _json_value(item, f"{path}/{index}")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{path}: JSON object keys must be strings")
            _json_value(item, f"{path}/{key}")
        return
    _fail(f"{path}: unsupported or non-finite JSON value")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _binding(value: Any) -> tuple[dict, str]:
    if type(value) is not dict or not _REQUIRED_BINDINGS <= value.keys():
        _fail("binding requires initial_weights, data, code, config")
    _json_value(value)
    if type(value["config"]) is not dict or not value["config"]:
        _fail("binding.config must be a nonempty JSON object")
    for name in ("initial_weights", "data", "code"):
        if type(value[name]) not in (str, dict) or not value[name]:
            _fail(f"binding.{name} must be a nonempty string or JSON object")
    if "binding_sha256" in value:
        body = {key: item for key, item in value.items() if key != "binding_sha256"}
        if value["binding_sha256"] != hashlib.sha256(_canonical(body)).hexdigest():
            _fail("binding's declared digest does not match its contents")
    raw = _canonical(value)
    identity = value.get("binding_sha256", hashlib.sha256(raw).hexdigest())
    return json.loads(raw), identity


def _safe(value: Any, path: str = "state") -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{path}: non-finite scalar")
        return value
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.device.type == "meta":
            _fail(f"{path}: unsupported tensor layout/device")
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if type(key) not in (str, int):
                _fail(f"{path}: unsupported mapping key")
            result[key] = _safe(item, f"{path}/{key}")
        return result
    if type(value) in (list, tuple):
        items = [_safe(item, f"{path}/{index}") for index, item in enumerate(value)]
        return tuple(items) if type(value) is tuple else items
    _fail(f"{path}: unsupported type {_type_name(value)}")


def _tensor_spec(value: torch.Tensor, *, digest: bool = False) -> dict:
    result = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if digest:
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        result["sha256"] = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    return result


def _model_layout(model: torch.nn.Module) -> dict:
    state = model.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        _fail("model state_dict must contain tensors only")
    caches = {}
    modules = {}
    for module_name, module in model.named_modules():
        modules[module_name] = _type_name(module)
        for name, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                caches[f"{module_name}:{name}"] = _tensor_spec(value, digest=True)
        for name in module._non_persistent_buffers_set:
            value = module._buffers[name]
            caches[f"{module_name}:buffer:{name}"] = (
                None if value is None else _tensor_spec(value, digest=True)
            )
        # These inspected vendor buffers are deterministic geometry constants,
        # unlike BatchNorm statistics. bind them despite persistent registration.
        for name in {"anchors", "valid_mask", "num_points_scale"} & module._buffers.keys():
            value = module._buffers[name]
            if value is not None:
                caches[f"{module_name}:persistent_cache:{name}"] = _tensor_spec(value, digest=True)
    return {
        "modules": modules,
        "state": {key: _tensor_spec(value) for key, value in state.items()},
        "parameters": [
            {"name": name, **_tensor_spec(value), "requires_grad": value.requires_grad}
            for name, value in model.named_parameters()
        ],
        "caches": caches,
    }


def _check_model_state(state: Any, layout: dict, label: str) -> None:
    if type(state) is not dict or set(state) != set(layout["state"]):
        _fail(f"{label}: model state keys differ")
    parameter_names = {entry["name"] for entry in layout["parameters"]}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            _fail(f"{label}/{name}: expected tensor")
        if _tensor_spec(value) != layout["state"][name]:
            _fail(f"{label}/{name}: tensor shape/dtype differs")
        leaf = name.rsplit(".", 1)[-1]
        if value.is_floating_point() or value.is_complex():
            if leaf == "anchors" and name not in parameter_names:
                # +inf is the inspected vendor's invalid-anchor sentinel.
                if bool(torch.isnan(value).any()) or bool(torch.isneginf(value).any()):
                    _fail(f"{label}/{name}: invalid anchor sentinel")
            elif not bool(torch.isfinite(value).all()):
                _fail(f"{label}/{name}: non-finite parameter or mutable buffer")
        if leaf == "num_batches_tracked" and bool((value < 0).any()):
            _fail(f"{label}/{name}: negative BatchNorm counter")
        if leaf == "running_var" and bool((value < 0).any()):
            _fail(f"{label}/{name}: negative BatchNorm running variance")
        if leaf in {"anchors", "valid_mask", "num_points_scale"}:
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            key = f"{parent}:persistent_cache:{leaf}"
            if key in layout["caches"] and _tensor_spec(value, digest=True) != layout["caches"][key]:
                _fail(f"{label}/{name}: persistent geometry cache identity differs")


def _upgrade_legacy_geometry_layout(layout: dict | None, state: dict | None) -> None:
    if layout is None or state is None:
        return
    parameters = {entry["name"] for entry in layout["parameters"]}
    for name, value in state.items():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"anchors", "valid_mask", "num_points_scale"} and name not in parameters:
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            layout["caches"][f"{parent}:persistent_cache:{leaf}"] = _tensor_spec(value, digest=True)


def _check_engine_owners(engine: Any, model: Any, optimizer: Any, ema: Any, scheduler: Any, warmup: Any) -> None:
    for name, expected in (
        ("model", model), ("optimizer", optimizer), ("ema", ema),
        ("scheduler", scheduler), ("warmup", warmup),
    ):
        if not hasattr(engine, name) or getattr(engine, name) is not expected:
            _fail(f"engine {name} must be the supplied checkpoint component")
    if scheduler is not None and getattr(scheduler, "optimizer", optimizer) is not optimizer:
        _fail("scheduler must reference the supplied optimizer")
    if warmup is not None and getattr(warmup, "lr_scheduler", scheduler) is not scheduler:
        _fail("warmup must reference the supplied scheduler")


def _engine_state(engine: Any) -> dict:
    state = _safe(engine.state_dict(), "engine")
    _check_engine(state)
    return state


def _check_engine(state: Any) -> None:
    from .training_v2b_engine import validate_engine_state_dict
    try:
        validate_engine_state_dict(state)
    except Exception as exc:
        raise CheckpointError(f"engine state is invalid: {exc}") from exc


def _optimizer_layout(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    names = {id(value): name for name, value in model.named_parameters()}
    seen = set()
    groups = []
    serialized = optimizer.state_dict()
    for group, saved_group in zip(optimizer.param_groups, serialized["param_groups"]):
        group_names = []
        for value in group["params"]:
            identity = id(value)
            if identity not in names or identity in seen:
                _fail("optimizer parameter is absent from model or duplicated")
            seen.add(identity)
            group_names.append(names[identity])
        options = {
            name: value for name, value in group.items()
            if name not in ("params", "lr")
        }
        groups.append({
            "names": group_names, "ids": list(saved_group["params"]),
            "options": _safe(options, "optimizer group options"),
        })
    return {
        "type": _type_name(optimizer),
        "defaults": _safe(optimizer.defaults, "optimizer defaults"),
        "groups": groups,
    }


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _check_bound_model_sampling(model: Any, ema: Any, bound: dict) -> None:
    """Validate non-serialized sampling behavior before any state/RNG mutation.

    The real v2b binding copies initialization['sampling'] into config. Compare
    that immutable inventory with both live modules using strict types, exact
    callable identity and the factory's inspected partial/method/module policy.
    No model is constructed, tensor is computed or CUDA API is called here.
    """
    config = bound["config"]
    models = (("raw", model), ("EMA", getattr(ema, "module", None)))
    if not any(key in config for key in ("sampling", "sampling_backend")):
        # Missing declarations are compatible with old generic toy artifacts,
        # but removing every sampling field must not bypass the real-model gate.
        for _, candidate in models:
            if isinstance(candidate, torch.nn.Module) and any(
                    base.__module__.endswith(".rtdetrv2_decoder")
                    and base.__name__ in {"MSDeformableAttention", "RTDETRTransformerv2"}
                    for module in candidate.modules() for base in type(module).__mro__):
                _fail("RT-DETRv2 sampling requires its bound initialization inventory")
        return
    backend = config.get("sampling_backend")
    declared = config.get("sampling")
    if (type(backend) is not str or type(declared) is not dict
            or type(declared.get("backend")) is not str or declared["backend"] != backend):
        _fail("bound sampling backend and initialization inventory are incomplete or inconsistent")
    from .training_v2b import validate_model_sampling
    for family, candidate in models:
        if not isinstance(candidate, torch.nn.Module):
            _fail(f"bound sampling requires a live {family} model")
        try:
            actual = validate_model_sampling(candidate, sampling_backend=backend)
        except Exception as exc:
            raise CheckpointError(f"{family} sampling callable/inventory validation failed: {exc}") from exc
        if not _equal(actual, declared):
            _fail(f"{family} sampling differs from the bound initialization inventory")


def _check_optimizer(state: Any, layout: dict, model: torch.nn.Module, updates: int) -> None:
    if type(layout) is not dict or set(layout) != {"type", "defaults", "groups"} or type(layout["groups"]) is not list:
        _fail("optimizer layout schema differs")
    layout_ids, layout_names = [], []
    for group in layout["groups"]:
        if type(group) is not dict or set(group) != {"names", "ids", "options"}:
            _fail("optimizer group layout schema differs")
        if type(group["names"]) is not list or type(group["ids"]) is not list or len(group["names"]) != len(group["ids"]):
            _fail("optimizer parameter name/ID counts differ")
        if any(type(name) is not str for name in group["names"]) or any(type(index) is not int for index in group["ids"]):
            _fail("optimizer parameter name/ID types differ")
        layout_names.extend(group["names"])
        layout_ids.extend(group["ids"])
    if len(set(layout_ids)) != len(layout_ids) or len(set(layout_names)) != len(layout_names):
        _fail("optimizer layout repeats a parameter")
    if type(state) is not dict or set(state) != {"state", "param_groups"}:
        _fail("optimizer state schema differs")
    if type(state["state"]) is not dict or type(state["param_groups"]) is not list:
        _fail("optimizer state containers differ")
    if len(state["param_groups"]) != len(layout["groups"]):
        _fail("optimizer parameter group count differs")
    parameters = dict(model.named_parameters()) if isinstance(model, torch.nn.Module) else dict(model)
    by_id = {}
    amsgrad_by_id = {}
    standard_adam = layout["type"] in {"torch.optim.adam.Adam", "torch.optim.adamw.AdamW"}
    for saved, expected in zip(state["param_groups"], layout["groups"]):
        if type(saved) is not dict or saved.get("params") != expected["ids"]:
            _fail("optimizer parameter group ID/order differs")
        opts = {key: value for key, value in saved.items() if key not in ("params", "lr")}
        if not _equal(opts, expected["options"]):
            _fail("optimizer declared parameter group options differ")
        lr = saved.get("lr")
        if isinstance(lr, torch.Tensor):
            valid_lr = lr.numel() == 1 and bool(torch.isfinite(lr).all()) and lr.item() >= 0
        else:
            valid_lr = type(lr) in (int, float) and math.isfinite(lr) and lr >= 0
        if not valid_lr:
            _fail("optimizer saved learning rate is invalid")
        by_id.update(zip(expected["ids"], (parameters[name] for name in expected["names"])))
        amsgrad_by_id.update({param_id: saved.get("amsgrad", False) for param_id in expected["ids"]})
    if any(type(key) is not int or key not in by_id for key in state["state"]):
        _fail("optimizer state references unknown parameters")
    for param_id, values in state["state"].items():
        if type(values) is not dict:
            _fail("optimizer parameter state must be a mapping")
        parameter = by_id[param_id]
        # Missing/empty state is legitimate for an unused or uninitialized
        # parameter. A nonempty Adam state, however, must be complete before
        # optimizer.load_state_dict is permitted to mutate a reconstruction.
        if standard_adam and values:
            required = {"step", "exp_avg", "exp_avg_sq"}
            if amsgrad_by_id[param_id]:
                required.add("max_exp_avg_sq")
            if set(values) != required:
                _fail("Adam/AdamW nonempty parameter state has missing or unexpected fields")
            for name in required - {"step"}:
                value = values[name]
                if not isinstance(value, torch.Tensor):
                    _fail("Adam/AdamW moment must be a tensor")
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    _fail("Adam/AdamW moment tensor shape/dtype differs from parameter")
                if not bool(torch.isfinite(value).all()):
                    _fail("Adam/AdamW moment tensor is non-finite")
                if name in {"exp_avg_sq", "max_exp_avg_sq"}:
                    real_values = torch.view_as_real(value) if value.is_complex() else value
                    if bool((real_values < 0).any()):
                        _fail("Adam/AdamW squared moment cannot be negative")
        for name, value in values.items():
            if name == "step":
                if isinstance(value, torch.Tensor):
                    if value.ndim != 0:
                        _fail("optimizer step must be a scalar tensor")
                    if standard_adam and value.dtype not in {torch.float32, torch.float64}:
                        _fail("Adam/AdamW step tensor must use a floating scalar dtype")
                    count = value.item()
                else:
                    count = value
                if type(count) not in (int, float) or not math.isfinite(count) or count != int(count) or not 0 <= count <= updates:
                    _fail("optimizer step exceeds engine updates or is invalid")
            elif isinstance(value, torch.Tensor):
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    _fail("optimizer moment tensor shape/dtype differs from parameter")


def _structure(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": _tensor_spec(value)}
    if isinstance(value, dict):
        return {key: _structure(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return {"sequence": type(value).__name__, "items": [_structure(item) for item in value]}
    return type(value).__name__


_COMPONENT_CONFIG = {
    "T_max", "eta_min", "milestones", "gamma", "step_size", "total_iters",
    "start_factor", "end_factor", "base_lrs", "warmup_steps", "warmup_duration",
    "warmup_iters", "duration", "warmup_end_values",
}


def _component(value: Any, label: str) -> dict | None:
    if value is None:
        return None
    state = _safe(value.state_dict(), label)
    if type(state) is not dict:
        _fail(f"{label}: state_dict must return a mapping")
    return {
        "type": _type_name(value),
        "state": state,
        "structure": _structure(state),
        "config": {key: item for key, item in state.items() if key in _COMPONENT_CONFIG},
    }


def _check_component(saved: Any, current: dict | None, label: str) -> None:
    if saved is None or current is None:
        if saved is not current:
            _fail(f"{label}: presence differs")
        return
    if type(saved) is not dict or set(saved) != {"type", "state", "structure", "config"}:
        _fail(f"{label}: component schema differs")
    if (
        saved["type"] != current["type"]
        or saved["structure"] != current["structure"]
        or saved["structure"] != _structure(saved["state"])
        or not _equal(saved["config"], current["config"])
        or not _equal(saved["config"], {key: value for key, value in saved["state"].items() if key in _COMPONENT_CONFIG})
    ):
        _fail(f"{label}: type, state structure, or declared configuration differs")


def _restore_component(value: Any, saved: dict) -> None:
    state = _safe(saved["state"], "component restore")
    if saved["type"] == "torch.optim.lr_scheduler.MultiStepLR":
        # The safe wire format stores integer-key mappings, never arbitrary
        # pickled classes. Native MultiStepLR also exposes a closed-form path
        # that requires Counter.elements(), so reconstruct its known type.
        state["milestones"] = Counter(state["milestones"])
    value.load_state_dict(state)


def _check_warmup_clock(component: dict | None, updates: int) -> None:
    # This is the inspected vendor contract: constructor step() leaves
    # last_step=0, and the engine calls step() after every optimizer update,
    # including updates after the warmup has finished. Other classes must not
    # silently inherit this counter convention.
    if component is not None and component["type"] == "src.optim.warmup.LinearWarmup":
        count = component["state"].get("last_step")
        if type(count) is not int or count != updates:
            _fail("vendor LinearWarmup last_step differs from engine updates")


def _check_scheduler_clock(component: dict | None, engine: dict, warmup: dict | None) -> None:
    if component is None or component["type"] != "torch.optim.lr_scheduler.MultiStepLR":
        return
    state = component["state"]
    milestones = state.get("milestones")
    if (type(milestones) is not dict or any(
            type(epoch) is not int or epoch < 0 or type(count) is not int or count <= 0
            for epoch, count in milestones.items())):
        _fail("MultiStepLR milestones must encode nonnegative integer epochs and positive counts")
    last_epoch, steps = state.get("last_epoch"), state.get("_step_count")
    completed_epochs = engine["epoch"] - int(engine["epoch_active"])
    if (
        type(last_epoch) is not int or not 0 <= last_epoch <= completed_epochs
        or type(steps) is not int or steps != last_epoch + 1
    ):
        _fail("MultiStepLR epoch/update clock is inconsistent with engine")
    if warmup is None and last_epoch != completed_epochs:
        _fail("MultiStepLR must step at every completed epoch without warmup")
    if warmup is not None and warmup["type"] == "src.optim.warmup.LinearWarmup":
        duration = warmup["state"].get("warmup_duration")
        if type(duration) is not int or duration <= 0:
            _fail("vendor LinearWarmup duration is invalid")
        if engine["optimizer_updates"] < duration and last_epoch != 0:
            _fail("MultiStepLR cannot advance before vendor warmup finishes")


def _ema(ema: Any, updates: int) -> tuple[dict | None, dict | None]:
    if ema is None:
        return None, None
    if not isinstance(getattr(ema, "module", None), torch.nn.Module):
        _fail("EMA must expose module and vendor-style state_dict")
    state = _safe(ema.state_dict(), "ema")
    layout = {
        "type": _type_name(ema),
        "model": _model_layout(ema.module),
        "config": {
            key: _safe(getattr(ema, key), f"ema/{key}")
            for key in ("decay", "warmups") if hasattr(ema, key)
        },
    }
    _check_ema(state, layout, updates)
    return state, layout


def _ema_layout_compatible(saved: Any, current: Any) -> bool:
    """Vendor EMA averages floating persistent caches as well as learned state.

    Such bytes can round even while raw geometry stays constant. Their saved
    hashes still authenticate the saved tensor values; compatibility with the
    freshly constructed EMA requires the same keys/shapes/dtypes, rather than
    incorrectly demanding the pre-update hash. Other cache identities stay exact.
    """
    if saved is None or current is None:
        return saved is current
    left, right = deepcopy(saved), deepcopy(current)
    for layout in (left, right):
        for name, spec in layout["model"]["caches"].items():
            if (":persistent_cache:" in name
                    and name.rsplit(":", 1)[-1] in {"anchors", "num_points_scale"}
                    and spec["dtype"].startswith(("torch.float", "torch.bfloat"))):
                spec["sha256"] = "vendor_ema_floating_cache_state_is_saved_and_restored"
    return left == right


def _check_ema(state: Any, layout: Any, updates: int) -> None:
    if state is None or layout is None:
        if state is not None or layout is not None:
            _fail("EMA state/layout presence differs")
        return
    if type(state) is not dict or set(state) != {"module", "updates"}:
        _fail("EMA state must contain module and updates")
    if type(state["updates"]) is not int or state["updates"] != updates:
        _fail("EMA update count differs from engine updates")
    _check_model_state(state["module"], layout["model"], "ema")


def _generators(value: Any) -> dict:
    if value is None:
        return {}
    if type(value) is not dict or any(
        type(key) is not str or not key or not isinstance(item, torch.Generator)
        for key, item in value.items()
    ):
        _fail("extra_rng_generators must map nonempty names to torch.Generator")
    return value


def _check_bound_runtime_identity(identity: dict, bound: dict) -> None:
    declared = bound["config"].get("device", "cpu")
    if declared != identity["device"]:
        _fail("runtime device differs from binding.config.device")
    if (identity["device"] != "cpu"
            and bound["config"].get("cuda_gpu_uuid") != identity["cuda_devices"][0]["uuid"]):
        _fail("CUDA GPU UUID differs from binding.config.cuda_gpu_uuid")
    from .training_v2b_device import validate_bound_runtime_policy
    try:
        validate_bound_runtime_policy(identity, bound["config"])
    except Exception as exc:
        raise CheckpointError(f"bound sampling/runtime backend policy differs: {exc}") from exc


def _check_generator_placement(generators: dict, identity: dict) -> None:
    allowed = {"cpu"} | {"cuda:" + str(item["index"]) for item in identity["cuda_devices"]}
    for name, generator in generators.items():
        if str(generator.device) not in allowed:
            _fail(f"generator {name}: device is outside the bound runtime")


def _runtime_identity(runtime: Any, bound: dict, bound_sha: str) -> dict:
    from .training_v2b_device import cpu_runtime_identity, validate_prepared_runtime
    identity = cpu_runtime_identity() if runtime is None else validate_prepared_runtime(runtime)
    _check_bound_runtime_identity(identity, bound)
    if runtime is not None and "seed" in bound["config"] and runtime.seed != bound["config"]["seed"]:
        _fail("prepared runtime seed differs from binding.config.seed")
    if identity["device"] != "cpu":
        if runtime.admitted_binding_sha256 != bound_sha:
            _fail("CUDA runtime admission belongs to a different run binding")
        if (bound["config"].get("device") != identity["device"]
                or bound["config"].get("cuda_gpu_uuid") != identity["cuda_devices"][0]["uuid"]):
            _fail("CUDA requested device / GPU UUID must be declared in binding.config")
    return identity


def _check_placement(model: Any, optimizer: Any, ema: Any, engine: Any, runtime: Any) -> None:
    from .training_v2b_device import validate_component_placement
    validate_component_placement(
        model=model, optimizer=optimizer, ema=ema,
        criterion=engine.criterion, runtime=runtime,
    )
    expected = "cpu" if runtime is None else str(runtime.device)
    if str(engine.device) != expected:
        _fail("engine device differs from the prepared runtime")


def _sampler_binding(bound: dict) -> str | None:
    data = bound["data"]
    return data.get("loader_binding_sha256") if type(data) is dict else None


def _check_sampler_saved(saved: Any, bound: dict, engine_state: dict) -> None:
    expected_sha = _sampler_binding(bound)
    if saved is None:
        if expected_sha is not None:
            _fail("bound loader cursor is missing from checkpoint")
        return
    if type(saved) is not dict or set(saved) != {"type", "state"} or type(saved["type"]) is not str or not saved["type"]:
        _fail("sampler checkpoint schema differs")
    from .training_v2b_data import validate_loader_state
    validate_loader_state(saved["state"], engine_state)
    if (type(expected_sha) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
            or saved["state"]["binding_sha256"] != expected_sha):
        _fail("sampler identity differs from binding.data.loader_binding_sha256")


def _sampler_state(sampler: Any, bound: dict, engine_state: dict) -> dict | None:
    if sampler is None:
        _check_sampler_saved(None, bound, engine_state)
        return None
    for name in ("state_dict", "validate_state_dict", "validate_engine_state", "load_state_dict"):
        if not callable(getattr(sampler, name, None)):
            _fail(f"sampler lacks {name}")
    state = _safe(sampler.state_dict(), "sampler")
    sampler.validate_state_dict(state)
    sampler.validate_engine_state(state, engine_state)
    saved = {"type": _type_name(sampler), "state": state}
    _check_sampler_saved(saved, bound, engine_state)
    return saved


def _check_sampler_live(saved: Any, sampler: Any, bound: dict, engine_state: dict) -> None:
    _check_sampler_saved(saved, bound, engine_state)
    if saved is None:
        if sampler is not None:
            _fail("checkpoint has no sampler; loader resume is unsupported")
        return
    if sampler is None or saved["type"] != _type_name(sampler):
        _fail("sampler type or presence differs")
    for name in ("state_dict", "validate_state_dict", "validate_engine_state", "load_state_dict"):
        if not callable(getattr(sampler, name, None)):
            _fail(f"sampler lacks {name}")
    sampler.validate_state_dict(saved["state"])
    sampler.validate_engine_state(saved["state"], engine_state)
    restore_validator = getattr(sampler, "validate_restore_state_dict", None)
    if restore_validator is not None:
        if not callable(restore_validator):
            _fail("sampler restore validator is not callable")
        restore_validator(saved["state"])


def _capture_rng(model: torch.nn.Module, ema: Any, generators: dict, runtime: Any = None) -> dict:
    from .training_v2b_device import capture_cuda_rng
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(numpy[0]),
            "keys": torch.tensor(numpy[1].astype(np.int64), dtype=torch.int64),
            "position": int(numpy[2]), "has_gauss": int(numpy[3]),
            "cached_gaussian": float(numpy[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda": capture_cuda_rng(runtime),
        "generators": {
            name: {"device": str(generator.device), "state": generator.get_state().cpu().clone()}
            for name, generator in generators.items()
        },
    }


def _numpy_state(value: Any) -> tuple:
    if type(value) is not dict or set(value) != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}:
        _fail("NumPy RNG state schema differs")
    keys = value["keys"]
    if (
        value["algorithm"] != "MT19937" or not isinstance(keys, torch.Tensor)
        or keys.dtype != torch.int64 or tuple(keys.shape) != (624,)
        or bool((keys < 0).any()) or bool((keys > 2**32 - 1).any())
        or type(value["position"]) is not int or not 0 <= value["position"] <= 624
        or type(value["has_gauss"]) is not int or value["has_gauss"] not in (0, 1)
        or type(value["cached_gaussian"]) is not float or not math.isfinite(value["cached_gaussian"])
    ):
        _fail("NumPy RNG state is invalid")
    return (
        value["algorithm"], keys.numpy().astype(np.uint32),
        value["position"], value["has_gauss"], value["cached_gaussian"],
    )


def _rng_tensor(value: Any, label: str) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu":
        _fail(f"{label}: invalid RNG byte state")


def _check_rng_structure(value: Any, runtime_identity: dict) -> None:
    from .training_v2b_device import validate_cuda_rng_states
    if type(value) is not dict or set(value) != {"python", "numpy", "torch_cpu", "cuda", "generators"}:
        _fail("RNG state schema differs")
    random.Random().setstate(value["python"])
    np.random.RandomState().set_state(_numpy_state(value["numpy"]))
    _rng_tensor(value["torch_cpu"], "torch CPU")
    torch.Generator(device="cpu").set_state(value["torch_cpu"])
    validate_cuda_rng_states(value["cuda"], runtime_identity)
    if type(value["generators"]) is not dict:
        _fail("explicit RNG generator state container differs")
    allowed = {"cpu"} | {"cuda:" + str(item["index"]) for item in runtime_identity["cuda_devices"]}
    cuda_sizes = {str(item["index"]): item["rng_state_bytes"] for item in runtime_identity["cuda_devices"]}
    for name, saved in value["generators"].items():
        if type(name) is not str or not name:
            _fail("explicit RNG generator name is invalid")
        if type(saved) is not dict or set(saved) != {"device", "state"} or saved["device"] not in allowed:
            _fail(f"generator {name}: device is outside the bound runtime")
        _rng_tensor(saved["state"], f"generator {name}")
        if saved["device"] == "cpu":
            torch.Generator(device="cpu").set_state(saved["state"])
        elif saved["state"].numel() != cuda_sizes[saved["device"].split(":")[1]]:
            _fail(f"generator {name}: CUDA RNG byte count differs")


def _check_rng(value: Any, model: torch.nn.Module, ema: Any, generators: dict,
               runtime: Any = None, runtime_identity: dict | None = None) -> None:
    from .training_v2b_device import (
        cpu_runtime_identity, preflight_cuda_rng_restore, _scratch_cuda_generator,
    )
    identity = cpu_runtime_identity() if runtime_identity is None else runtime_identity
    _check_rng_structure(value, identity)
    if set(value["generators"]) != set(generators):
        _fail("explicit RNG generator names differ")
    for name, generator in generators.items():
        saved = value["generators"][name]
        if saved["device"] != str(generator.device):
            _fail(f"generator {name}: device differs")
        if generator.device.type == "cpu":
            torch.Generator(device="cpu").set_state(saved["state"])
        else:
            _scratch_cuda_generator(generator.device.index).set_state(saved["state"])
    preflight_cuda_rng_restore(value["cuda"], runtime)


def _restore_rng(value: dict, generators: dict, runtime: Any = None) -> None:
    from .training_v2b_device import restore_cuda_rng
    random.setstate(value["python"])
    np.random.set_state(_numpy_state(value["numpy"]))
    torch.set_rng_state(value["torch_cpu"])
    restore_cuda_rng(value["cuda"], runtime)
    for name, saved in value["generators"].items():
        generators[name].set_state(saved["state"])


def _verified_payload(path: str | os.PathLike, expected_sha256: str, expected_binding: dict) -> tuple:
    if type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        _fail("a trusted lowercase external SHA256 is required")
    source = Path(path).absolute()
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha256:
        _fail("checkpoint SHA256 mismatch before deserialization")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(payload) is not dict:
        _fail("checkpoint schema keys differ")
    version = payload.get("schema_version")
    if payload.get("format") != FORMAT or type(version) is not int or version not in (LEGACY_CPU_SCHEMA_VERSION, SCHEMA_VERSION):
        _fail("checkpoint format/schema version differs")
    expected_keys = _PAYLOAD_KEYS if version == SCHEMA_VERSION else _LEGACY_PAYLOAD_KEYS
    if set(payload) != expected_keys:
        _fail("checkpoint schema keys differ")
    from .training_v2b_device import cpu_runtime_identity, validate_runtime_identity
    if version == LEGACY_CPU_SCHEMA_VERSION:
        rng = payload["rng"]
        if (type(rng) is not dict or rng.get("cuda") != {}
                or type(rng.get("generators")) is not dict
                or any(type(value) is not dict or value.get("device") != "cpu"
                       for value in rng["generators"].values())):
            _fail("legacy CUDA checkpoint lacks UUID/topology binding and cannot be migrated")
        payload = {**payload, "runtime": cpu_runtime_identity(), "sampler": None}
        # Reviewable schema migration: derive static persistent-cache identities
        # from authenticated legacy bytes. Restore still compares these values
        # with the fresh model's geometry before loading any saved state.
        _upgrade_legacy_geometry_layout(payload["model_layout"], payload["model"])
        if payload["ema"] is not None:
            _upgrade_legacy_geometry_layout(payload["ema_layout"]["model"], payload["ema"]["module"])
    validate_runtime_identity(payload["runtime"])
    if payload["torch_version"] != str(torch.__version__):
        _fail("checkpoint PyTorch version differs")
    bound, bound_sha = _binding(expected_binding)
    saved_bound, saved_sha = _binding(payload["binding"])
    if saved_bound != bound or saved_sha != bound_sha or payload["binding_sha256"] != bound_sha:
        _fail("checkpoint binding differs")
    _check_bound_runtime_identity(payload["runtime"], bound)
    return source, raw, payload, bound_sha


def inspect_checkpoint(path: str | os.PathLike, *, expected_sha256: str, expected_binding: dict) -> dict:
    """Inspect stored identity/counters using CPU weights-only loading.

    This validates the saved artifact without constructing a model. It does not
    certify compatibility with live objects, successful restoration, CUDA
    execution, DataLoader state, or scientific validity.
    """
    try:
        source, raw, payload, bound_sha = _verified_payload(path, expected_sha256, expected_binding)
        _check_engine(payload["engine"])
        _check_rng_structure(payload["rng"], payload["runtime"])
        _check_sampler_saved(payload["sampler"], payload["binding"], payload["engine"])
        model_layout = payload["model_layout"]
        _check_model_state(payload["model"], model_layout, "model")
        parameters = {}
        for entry in model_layout["parameters"]:
            if type(entry) is not dict or set(entry) != {"name", "shape", "dtype", "requires_grad"}:
                _fail("stored parameter layout schema differs")
            name = entry["name"]
            if type(name) is not str or name in parameters or name not in payload["model"]:
                _fail("stored parameter layout names differ")
            tensor = payload["model"][name]
            if _tensor_spec(tensor) != {"shape": entry["shape"], "dtype": entry["dtype"]} or type(entry["requires_grad"]) is not bool:
                _fail("stored parameter layout shape/dtype differs")
            parameters[name] = tensor
        _check_optimizer(payload["optimizer"], payload["optimizer_layout"], parameters, payload["engine"]["optimizer_updates"])
        _check_ema(payload["ema"], payload["ema_layout"], payload["engine"]["optimizer_updates"])
        _check_component(payload["scheduler"], payload["scheduler"], "scheduler")
        _check_component(payload["warmup"], payload["warmup"], "warmup")
        _check_warmup_clock(payload["warmup"], payload["engine"]["optimizer_updates"])
        _check_scheduler_clock(payload["scheduler"], payload["engine"], payload["warmup"])
        engine_state = _safe(payload["engine"], "engine")
        return {
            "path": str(source), "sha256": expected_sha256, "size_bytes": len(raw),
            "format": FORMAT, "schema_version": payload["schema_version"],
            "runtime": payload["runtime"], "sampler": payload["sampler"],
            "binding_sha256": bound_sha, "binding": payload["binding"],
            "engine": engine_state,
            "epoch": engine_state["epoch"], "epoch_active": engine_state["epoch_active"],
            "optimizer_updates": engine_state["optimizer_updates"],
            "microsteps": engine_state["microsteps"],
            "samples_seen": engine_state["microsteps"] * engine_state["config"]["physical_batch_size"],
            "inspection_scope": "stored_checkpoint_identity_and_counters",
        }
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"checkpoint inspection failed: {type(exc).__name__}: {exc}") from exc


def save_checkpoint(
    path: str | os.PathLike, *, model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, ema: Any, engine: Any,
    scheduler: Any = None, warmup: Any = None, binding: dict,
    extra_rng_generators: dict | None = None,
    runtime: Any = None, sampler: Any = None,
) -> dict:
    """Atomically publish a new checkpoint; never replace an existing path."""
    temporary = None
    try:
        destination = Path(path).absolute()
        if os.path.lexists(destination):
            _fail("checkpoint destination already exists; overwrite is forbidden")
        if not destination.parent.is_dir():
            _fail("checkpoint parent directory must already exist")
        bound, bound_sha = _binding(binding)
        _check_bound_model_sampling(model, ema, bound)
        _check_engine_owners(engine, model, optimizer, ema, scheduler, warmup)
        engine_state = _engine_state(engine)
        runtime_identity = _runtime_identity(runtime, bound, bound_sha)
        _check_placement(model, optimizer, ema, engine, runtime)
        sampler_state = _sampler_state(sampler, bound, engine_state)
        generators = _generators(extra_rng_generators)
        _check_generator_placement(generators, runtime_identity)
        layout = _model_layout(model)
        model_state = _safe(model.state_dict(), "model")
        _check_model_state(model_state, layout, "model")
        optimizer_layout = _optimizer_layout(model, optimizer)
        optimizer_state = _safe(optimizer.state_dict(), "optimizer")
        _check_optimizer(optimizer_state, optimizer_layout, model, engine_state["optimizer_updates"])
        ema_state, ema_layout = _ema(ema, engine_state["optimizer_updates"])
        scheduler_state = _component(scheduler, "scheduler")
        warmup_state = _component(warmup, "warmup")
        _check_warmup_clock(warmup_state, engine_state["optimizer_updates"])
        _check_scheduler_clock(scheduler_state, engine_state, warmup_state)
        payload = {
            "format": FORMAT, "schema_version": SCHEMA_VERSION,
            "torch_version": str(torch.__version__),
            "binding": bound, "binding_sha256": bound_sha,
            "model": model_state, "model_layout": layout,
            "model_training": {name: module.training for name, module in model.named_modules()},
            "optimizer": optimizer_state, "optimizer_layout": optimizer_layout,
            "ema": ema_state, "ema_layout": ema_layout,
            "engine": engine_state, "engine_type": _type_name(engine),
            "scheduler": scheduler_state,
            "warmup": warmup_state,
            "rng": _capture_rng(model, ema, generators, runtime),
            "runtime": runtime_identity, "sampler": sampler_state,
        }
        _check_rng(payload["rng"], model, ema, generators, runtime, runtime_identity)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        raw = Path(temporary).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        # Same-directory hard-link publication is atomic and fails if destination
        # appears concurrently. os.replace would violate the no-overwrite rule.
        os.link(temporary, destination)
        os.unlink(temporary)
        temporary = None
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {
            "path": str(destination), "sha256": sha, "size_bytes": len(raw),
            "schema_version": SCHEMA_VERSION, "binding_sha256": bound_sha,
            "epoch": engine_state["epoch"],
            "optimizer_updates": engine_state["optimizer_updates"],
        }
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"checkpoint save failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def restore_checkpoint(
    path: str | os.PathLike, *, model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, ema: Any, engine: Any,
    scheduler: Any = None, warmup: Any = None, expected_binding: dict,
    expected_sha256: str, extra_rng_generators: dict | None = None,
    runtime: Any = None, sampler: Any = None,
) -> dict:
    """Verify all identities/state layouts before strict load; restore RNG last.

    A trusted SHA256 is mandatory; it is checked on the exact bytes passed to
    weights_only=True. If a custom object's load_state_dict unexpectedly fails,
    discard the reconstruction; this API does not roll back arbitrary objects.
    """
    try:
        source, raw, payload, bound_sha = _verified_payload(path, expected_sha256, expected_binding)
        actual_sha = expected_sha256
        _check_bound_model_sampling(model, ema, payload["binding"])
        runtime_identity = _runtime_identity(runtime, payload["binding"], bound_sha)
        if payload["runtime"] != runtime_identity:
            _fail("runtime device, UUID/topology, backend policy, or CUDA build differs; remapping is unsupported")
        _check_placement(model, optimizer, ema, engine, runtime)
        _check_engine_owners(engine, model, optimizer, ema, scheduler, warmup)
        current_engine = _engine_state(engine)
        _check_engine(payload["engine"])
        if payload["engine_type"] != _type_name(engine) or payload["engine"]["config"] != current_engine["config"]:
            _fail("engine type/config differs")
        engine.validate_state_dict(payload["engine"])
        layout = _model_layout(model)
        if payload["model_layout"] != layout:
            _fail("model keys, parameter order, cache identity, shape or dtype differs")
        _check_model_state(payload["model"], layout, "model")
        modes = payload["model_training"]
        if type(modes) is not dict or set(modes) != set(layout["modules"]) or any(type(value) is not bool for value in modes.values()):
            _fail("model training-mode schema differs")
        optimizer_layout = _optimizer_layout(model, optimizer)
        if not _equal(payload["optimizer_layout"], optimizer_layout):
            _fail("optimizer parameter group identity/order or declared options differ")
        _check_optimizer(payload["optimizer"], optimizer_layout, model, payload["engine"]["optimizer_updates"])
        _, ema_layout = _ema(ema, current_engine["optimizer_updates"])
        if not _ema_layout_compatible(payload["ema_layout"], ema_layout):
            _fail("EMA type, model/cache identity, or configuration differs")
        _check_ema(payload["ema"], payload["ema_layout"], payload["engine"]["optimizer_updates"])
        _check_component(payload["scheduler"], _component(scheduler, "scheduler"), "scheduler")
        _check_component(payload["warmup"], _component(warmup, "warmup"), "warmup")
        _check_warmup_clock(payload["warmup"], payload["engine"]["optimizer_updates"])
        _check_scheduler_clock(payload["scheduler"], payload["engine"], payload["warmup"])
        generators = _generators(extra_rng_generators)
        _check_rng(payload["rng"], model, ema, generators, runtime, runtime_identity)
        _check_sampler_live(payload["sampler"], sampler, payload["binding"], payload["engine"])
        # All externally controlled identities, tensor layouts, counters, and RNG
        # containers have been checked before mutating any supplied state object.
        try:
            model.load_state_dict(payload["model"], strict=True)
            if ema is not None:
                ema.load_state_dict(payload["ema"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            _check_placement(model, optimizer, ema, engine, runtime)
            if scheduler is not None:
                _restore_component(scheduler, payload["scheduler"])
            if warmup is not None:
                _restore_component(warmup, payload["warmup"])
            engine.load_state_dict(payload["engine"])
            for name, module in model.named_modules():
                module.training = modes[name]
            if sampler is not None:
                sampler.load_state_dict(payload["sampler"]["state"])
            _restore_rng(payload["rng"], generators, runtime)
        except Exception as exc:
            engine.failed = True
            raise CheckpointError(
                "checkpoint load unexpectedly failed; reconstruction invalidated, discard all supplied objects"
            ) from exc
        return {
            "path": str(source), "sha256": actual_sha, "size_bytes": len(raw),
            "schema_version": payload["schema_version"], "binding_sha256": bound_sha,
            "runtime": payload["runtime"], "sampler_restored": sampler is not None,
            "epoch": payload["engine"]["epoch"],
            "epoch_active": payload["engine"]["epoch_active"],
            "optimizer_updates": payload["engine"]["optimizer_updates"],
            "microsteps": payload["engine"]["microsteps"],
        }
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"checkpoint restore failed: {type(exc).__name__}: {exc}") from exc
