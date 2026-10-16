"""Explicit CPU / single-CUDA-device placement and RNG contract for v2b.

Calling a CUDA preparation function is a separate execution step: it first
requires a fresh native hardware admission, then initializes CUDA. CPU callers
never query CUDA availability, state, topology, or generators. CUDA is restricted
to one UUID-selected visible device; device remapping is deliberately unsupported.
This module has CPU tests of CUDA control flow, not GPU verification.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
import random
import re
from typing import Any

import numpy as np
import torch
from torch import nn


class RuntimeDeviceError(RuntimeError):
    """Runtime identity, preparation, placement, or RNG state is incompatible."""


_SEAL = object()
_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_IDENTITY_KEYS = {
    "schema_version", "device", "torch_version", "cuda_build_version",
    "cuda_visible_devices", "cuda_devices", "remapping_policy", "backend_policy",
}


def cuda_backend_policy_for_sampling(sampling_backend: str) -> dict:
    """Return the declared CUDA policy without querying or initializing CUDA.

    Native sampling remains an explicit diagnostic/CPU-compatible choice.
    The gather candidate requires strict deterministic algorithms; the cuBLAS
    workspace is a separate requirement for this model's matrix operations.
    """
    if type(sampling_backend) is not str or sampling_backend not in {"native", "deterministic_gather"}:
        raise RuntimeDeviceError("unknown sampling backend")
    return {
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "cudnn_allow_tf32": False, "matmul_allow_tf32": False,
        "deterministic_algorithms": sampling_backend == "deterministic_gather",
        "deterministic_warn_only": False,
        "cublas_workspace_config": ":4096:8",
    }


def _bound_backend_policy(config: dict) -> dict | None:
    if type(config) is not dict:
        raise RuntimeDeviceError("bound runtime configuration must be an object")
    declared = {"sampling_backend", "cuda_backend_policy"} & set(config)
    # Generic historical checkpoint/device callers did not declare sampling.
    # New V2BConfig bindings always contain both fields and cannot use this path.
    if not declared:
        return None
    if declared != {"sampling_backend", "cuda_backend_policy"}:
        raise RuntimeDeviceError("sampling backend and CUDA policy must be declared together")
    expected = cuda_backend_policy_for_sampling(config["sampling_backend"])
    actual = config["cuda_backend_policy"]
    if type(actual) is not dict or set(actual) != set(expected):
        raise RuntimeDeviceError("declared CUDA backend policy schema differs")
    if any(type(actual[key]) is not type(value) or actual[key] != value
           for key, value in expected.items()):
        raise RuntimeDeviceError("declared CUDA backend policy differs from sampling backend")
    return expected


def validate_bound_runtime_policy(identity: dict, config: dict) -> None:
    """Validate requested versus recorded policy using CPU-only operations.

    CPU identities do not claim CUDA flags. Their future CUDA declaration must
    still agree with the selected sampler. CUDA identities must match every
    declared flag and workspace value, including strict versus warn-only mode.
    """
    checked = validate_runtime_identity(identity)
    expected = _bound_backend_policy(config)
    if expected is not None and checked["device"] == "cuda:0" and checked["backend_policy"] != expected:
        raise RuntimeDeviceError("runtime CUDA backend policy differs from bound sampling backend")


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False,
    ).encode()).hexdigest()


def _seed_cpu(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise RuntimeDeviceError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed also enqueues CUDA seeding and is intentionally avoided.
    torch.random.default_generator.manual_seed(seed)


def cpu_runtime_identity() -> dict:
    """Describe the CPU checkpoint path without calling any CUDA API."""
    return {
        "schema_version": 1, "device": "cpu",
        "torch_version": str(torch.__version__),
        "cuda_build_version": None,
        "cuda_visible_devices": None, "cuda_devices": [],
        "remapping_policy": "strict_no_device_or_uuid_remap",
        "backend_policy": {},
    }


def validate_runtime_identity(value: Any) -> dict:
    """Validate stored runtime metadata using CPU-only operations."""
    if type(value) is not dict or set(value) != _IDENTITY_KEYS:
        raise RuntimeDeviceError("runtime identity schema differs")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise RuntimeDeviceError("runtime identity version differs")
    if value["torch_version"] != str(torch.__version__):
        raise RuntimeDeviceError("runtime PyTorch version differs")
    if value["remapping_policy"] != "strict_no_device_or_uuid_remap":
        raise RuntimeDeviceError("runtime device remapping is unsupported")
    if value["device"] == "cpu":
        if value != cpu_runtime_identity():
            raise RuntimeDeviceError("CPU runtime identity contains CUDA metadata")
        return deepcopy(value)
    if value["device"] != "cuda:0":
        raise RuntimeDeviceError("only explicit cuda:0 with one UUID-selected device is supported")
    devices = value["cuda_devices"]
    if type(devices) is not list or len(devices) != 1:
        raise RuntimeDeviceError("single-device CUDA topology required")
    entry = devices[0]
    keys = {"index", "uuid", "name", "total_memory", "capability", "rng_state_bytes"}
    if type(entry) is not dict or set(entry) != keys or type(entry["index"]) is not int or entry["index"] != 0:
        raise RuntimeDeviceError("CUDA topology entry schema differs")
    if type(entry["uuid"]) is not str or _UUID.fullmatch(entry["uuid"]) is None:
        raise RuntimeDeviceError("full CUDA GPU UUID is required")
    if value["cuda_visible_devices"] != entry["uuid"]:
        raise RuntimeDeviceError("CUDA visibility must exactly select the bound UUID")
    if type(entry["name"]) is not str or not entry["name"]:
        raise RuntimeDeviceError("CUDA device name is missing")
    if any(type(entry[key]) is not int or entry[key] <= 0 for key in ("total_memory", "rng_state_bytes")):
        raise RuntimeDeviceError("CUDA topology capacity / RNG shape is invalid")
    capability = entry["capability"]
    if type(capability) is not list or len(capability) != 2 or any(type(x) is not int or x < 0 for x in capability):
        raise RuntimeDeviceError("CUDA capability is invalid")
    if type(value["cuda_build_version"]) is not str or not value["cuda_build_version"]:
        raise RuntimeDeviceError("CUDA build version is missing")
    policy = value["backend_policy"]
    if type(policy) is not dict or set(policy) != {
        "cudnn_benchmark", "cudnn_deterministic", "cudnn_allow_tf32",
        "matmul_allow_tf32", "deterministic_algorithms", "deterministic_warn_only",
        "cublas_workspace_config",
    }:
        raise RuntimeDeviceError("CUDA backend policy schema differs")
    for key in set(policy) - {"cublas_workspace_config"}:
        if type(policy[key]) is not bool:
            raise RuntimeDeviceError("CUDA backend flags must be boolean")
    if (policy["cudnn_benchmark"] is not False or policy["cudnn_deterministic"] is not True
            or policy["cudnn_allow_tf32"] is not False or policy["matmul_allow_tf32"] is not False):
        raise RuntimeDeviceError("CUDA numerical backend policy differs")
    if policy["cublas_workspace_config"] not in (None, ":16:8", ":4096:8"):
        raise RuntimeDeviceError("unrecognized cuBLAS deterministic workspace policy")
    return deepcopy(value)


def _backend_policy() -> dict:
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


@dataclass(frozen=True)
class PreparedRuntime:
    device: torch.device
    seed: int
    admitted_binding_sha256: str | None
    _identity: dict = field(repr=False)
    _identity_sha256: str = field(repr=False)
    _seal: object = field(repr=False)

    @property
    def identity(self) -> dict:
        return deepcopy(self._identity)

    def binding_config(self) -> dict:
        return self.identity


def validate_prepared_runtime(runtime: PreparedRuntime) -> dict:
    if type(runtime) is not PreparedRuntime or runtime._seal is not _SEAL:
        raise RuntimeDeviceError("runtime must come from prepare_runtime")
    identity = validate_runtime_identity(runtime._identity)
    if _digest(identity) != runtime._identity_sha256 or str(runtime.device) != identity["device"]:
        raise RuntimeDeviceError("prepared runtime identity was altered")
    if runtime.device.type == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != identity["cuda_visible_devices"]:
            raise RuntimeDeviceError("CUDA visibility changed after preparation")
        if _backend_policy() != identity["backend_policy"]:
            raise RuntimeDeviceError("CUDA numerical backend policy changed after preparation")
    return identity


def prepare_runtime(*, device: str | torch.device = "cpu", seed: int,
                    binding: dict | None = None, gpu_probe: Any = None,
                    expected_gpu_uuid: str | None = None) -> PreparedRuntime:
    """Seed CPU sources, and only after native admission initialize one GPU.

    The production CUDA branch requires a same-process native hardware token.
    CPU tests explicitly replace the native admission and CUDA APIs. The exact
    CUDA RNG is restored on resume. Explicit sampler bindings select numerical
    flags before CUDA initialization; flags alone do not prove full-model replay.
    """
    chosen = torch.device(device)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise RuntimeDeviceError("seed must be an integer in [0, 2**32)")
    if type(binding) is dict and type(binding.get("config")) is dict:
        declared_seed = binding["config"].get("seed", seed)
        if type(declared_seed) is not int or declared_seed != seed:
            raise RuntimeDeviceError("runtime seed differs from the run binding")
        requested_policy = _bound_backend_policy(binding["config"])
    else:
        requested_policy = None
    if str(chosen) == "cpu":
        if gpu_probe is not None or expected_gpu_uuid is not None:
            raise RuntimeDeviceError("CPU preparation does not accept a GPU admission")
        _seed_cpu(seed)
        identity = cpu_runtime_identity()
        return PreparedRuntime(chosen, seed, None, identity, _digest(identity), _SEAL)
    if str(chosen) != "cuda:0":
        raise RuntimeDeviceError("CUDA requires explicit cuda:0 and one UUID-selected visible device")
    if type(expected_gpu_uuid) is not str or _UUID.fullmatch(expected_gpu_uuid) is None:
        raise RuntimeDeviceError("CUDA requires the full native GPU UUID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu_uuid:
        raise RuntimeDeviceError("CUDA_VISIBLE_DEVICES must exactly equal the admitted GPU UUID")
    if type(binding) is not dict:
        raise RuntimeDeviceError("CUDA preparation requires the run binding")
    if (requested_policy is not None and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != requested_policy["cublas_workspace_config"]):
        raise RuntimeDeviceError("set the bound cuBLAS workspace in the process startup environment")
    from .training_v2b_hardware import require_native_hardware_admission
    require_native_hardware_admission(
        gpu_probe, binding=binding, expected_gpu_uuid=expected_gpu_uuid,
    )
    # No CUDA API above this point. Initialization is deliberately explicit.
    api = torch.cuda
    if api.is_initialized():
        raise RuntimeDeviceError("prepare CUDA before any previous CUDA initialization")
    # CUDA admission has passed, but no CUDA initialization or computation has
    # occurred. Never rely on ambient deterministic/warn-only flags for a new
    # sampler binding, and never set the workspace after a cuBLAS context exists.
    if requested_policy is not None:
        torch.use_deterministic_algorithms(
            requested_policy["deterministic_algorithms"],
            warn_only=requested_policy["deterministic_warn_only"],
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    _seed_cpu(seed)
    api.init()
    if api.device_count() != 1:
        raise RuntimeDeviceError("admitted single-GPU visibility does not match CUDA runtime")
    api.set_device(0)
    properties = api.get_device_properties(0)
    native_uuid = getattr(properties, "uuid", None)
    if native_uuid is not None and str(native_uuid).lower() != expected_gpu_uuid.lower():
        raise RuntimeDeviceError("CUDA properties UUID differs from native admission")
    # Older PyTorch versions omit UUID in properties. Exact full-UUID
    # CUDA_VISIBLE_DEVICES plus count=1 fixes the ordinal mapping in that case.
    api.default_generators[0].manual_seed(seed)
    rng_state = api.get_rng_state(0)
    if rng_state.device.type != "cpu" or rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
        raise RuntimeDeviceError("CUDA generator did not return portable CPU byte state")
    identity = {
        "schema_version": 1, "device": "cuda:0",
        "torch_version": str(torch.__version__),
        "cuda_build_version": str(torch.version.cuda),
        "cuda_visible_devices": expected_gpu_uuid,
        "cuda_devices": [{
            "index": 0, "uuid": expected_gpu_uuid, "name": str(properties.name),
            "total_memory": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
            "rng_state_bytes": rng_state.numel(),
        }],
        "remapping_policy": "strict_no_device_or_uuid_remap",
        "backend_policy": _backend_policy(),
    }
    validate_runtime_identity(identity)
    validate_bound_runtime_policy(identity, binding.get("config", {}))
    bound_sha = binding.get("binding_sha256", _digest(binding))
    return PreparedRuntime(chosen, seed, bound_sha, identity, _digest(identity), _SEAL)


def validate_cuda_rng_states(states: Any, identity: dict) -> None:
    identity = validate_runtime_identity(identity)
    expected = {str(entry["index"]): entry for entry in identity["cuda_devices"]}
    if type(states) is not dict or set(states) != set(expected):
        raise RuntimeDeviceError("CUDA RNG topology differs from runtime identity")
    for index, value in states.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.uint8 or value.ndim != 1
                or value.numel() != expected[index]["rng_state_bytes"]):
            raise RuntimeDeviceError("CUDA RNG byte state shape/dtype/device differs")


def capture_cuda_rng(runtime: PreparedRuntime | None) -> dict[str, torch.Tensor]:
    if runtime is None:
        return {}
    identity = validate_prepared_runtime(runtime)
    if runtime.device.type == "cpu":
        return {}
    if not torch.cuda.is_initialized() or torch.cuda.current_device() != 0:
        raise RuntimeDeviceError("CUDA runtime is no longer prepared on the bound device")
    result = {"0": torch.cuda.get_rng_state(0).detach().cpu().clone()}
    validate_cuda_rng_states(result, identity)
    return result


def _scratch_cuda_generator(index: int) -> torch.Generator:
    return torch.Generator(device=torch.device("cuda", index))


def preflight_cuda_rng_restore(states: dict, runtime: PreparedRuntime | None) -> None:
    identity = cpu_runtime_identity() if runtime is None else validate_prepared_runtime(runtime)
    validate_cuda_rng_states(states, identity)
    if identity["device"] == "cpu":
        return
    if not torch.cuda.is_initialized() or torch.cuda.current_device() != 0:
        raise RuntimeDeviceError("CUDA restore requires the previously prepared bound device")
    # Validate generator bytes in an independent generator before mutating live
    # model/optimizer state. The temporary generator never launches kernels.
    for index, value in states.items():
        _scratch_cuda_generator(int(index)).set_state(value)


def restore_cuda_rng(states: dict, runtime: PreparedRuntime | None) -> None:
    identity = cpu_runtime_identity() if runtime is None else validate_prepared_runtime(runtime)
    validate_cuda_rng_states(states, identity)
    if states and (not torch.cuda.is_initialized() or torch.cuda.current_device() != 0):
        raise RuntimeDeviceError("CUDA runtime changed before final RNG restore")
    for index, value in states.items():
        torch.cuda.set_rng_state(value, int(index))


def _plain_tensors(value: Any, prefix: str):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif type(value) is dict:
        for name, child in value.items():
            yield from _plain_tensors(child, prefix + "/" + str(name))
    elif type(value) in (tuple, list):
        for index, child in enumerate(value):
            yield from _plain_tensors(child, prefix + "/" + str(index))


def _module_tensors(module: nn.Module):
    for name, value in module.named_parameters():
        yield "parameter:" + name, value
    for name, value in module.named_buffers():
        yield "buffer:" + name, value
    for module_name, child in module.named_modules():
        for name, value in vars(child).items():
            if name not in {"_parameters", "_buffers", "_modules"}:
                yield from _plain_tensors(value, "cache:" + module_name + ":" + name)


def place_module(module: nn.Module, runtime: PreparedRuntime) -> nn.Module:
    """Place registered state and unregistered tensor caches before optimizer creation."""
    validate_prepared_runtime(runtime)
    if not isinstance(module, nn.Module):
        raise RuntimeDeviceError("component must be a torch module")
    old_state = {id(value): name for name, value in
                 list(module.named_parameters()) + list(module.named_buffers())}
    module.to(device=runtime.device)
    current = dict(list(module.named_parameters()) + list(module.named_buffers()))
    moved = {identity: current[name] for identity, name in old_state.items()}

    def move(value):
        if isinstance(value, torch.Tensor):
            if id(value) not in moved:
                moved[id(value)] = value.to(device=runtime.device)
            return moved[id(value)]
        if type(value) is dict:
            return {key: move(child) for key, child in value.items()}
        if type(value) is list:
            return [move(child) for child in value]
        if type(value) is tuple:
            return tuple(move(child) for child in value)
        return value

    for child in module.modules():
        for name, value in tuple(vars(child).items()):
            if name not in {"_parameters", "_buffers", "_modules"} and any(_plain_tensors(value, name)):
                setattr(child, name, move(value))
    validate_module_placement(module, runtime.device)
    return module


def validate_module_placement(module: nn.Module, device: str | torch.device) -> None:
    chosen = torch.device(device)
    if not isinstance(module, nn.Module):
        raise RuntimeDeviceError("component must be a torch module")
    for name, value in _module_tensors(module):
        if value.device != chosen:
            raise RuntimeDeviceError(f"{name}: tensor placement differs from {chosen}")


def validate_optimizer_placement(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    chosen = torch.device(device)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.device != chosen:
                raise RuntimeDeviceError("optimizer parameter placement differs")
            for name, value in optimizer.state.get(parameter, {}).items():
                if isinstance(value, torch.Tensor):
                    # Adam/W keeps scalar steps on CPU unless capturable/fused.
                    step_on_cpu = name == "step" and not (group.get("capturable", False) or group.get("fused", False))
                    expected = torch.device("cpu") if step_on_cpu else parameter.device
                    if value.device != expected:
                        raise RuntimeDeviceError(f"optimizer {name}: state placement differs")


def validate_component_placement(*, model: nn.Module, optimizer: torch.optim.Optimizer,
                                 criterion: Any = None, ema: Any = None,
                                 postprocessor: Any = None,
                                 runtime: PreparedRuntime | None = None) -> None:
    identity = cpu_runtime_identity() if runtime is None else validate_prepared_runtime(runtime)
    device = torch.device(identity["device"])
    validate_module_placement(model, device)
    for component in (criterion, postprocessor):
        if isinstance(component, nn.Module):
            validate_module_placement(component, device)
    if ema is not None:
        validate_module_placement(ema.module, device)
    validate_optimizer_placement(optimizer, device)
