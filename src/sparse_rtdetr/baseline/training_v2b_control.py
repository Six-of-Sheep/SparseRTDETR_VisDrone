"""Frozen 640 control worker and fail-closed smoke comparisons.

Importing this module performs no GPU observation, clock change, data read or
execution. A worker consumes an externally authenticated contract. The parent
controller owns authorization, the single-GPU sequence and final guardian
clearance. Worker PASS alone is never hardware clearance or scientific proof.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict
import hashlib
import io
import math
import os
from pathlib import Path
import random
import re
import stat
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .training_v2b import V2BConfig, build_v2b_components, logical_batch_indices
from .training_v2b_data import (
    TrainCoreDataConfig, build_train_core_loader, validate_worker_environment,
)
from .training_v2b_device import capture_cuda_rng, prepare_runtime
from .training_v2b_evidence import (
    canonical_json_bytes, canonical_sha256, file_reference, initial_parameter_reference,
    strict_json_loads, validate_run_binding, write_exclusive_json,
)
from .training_v2b_runtime import V2BTrainingSession, build_train_core_run_binding


class ControlWorkerError(RuntimeError):
    """Stop this invocation; do not retry, change batching or loosen tolerances."""


STAGES = ("smoke", "smoke_replay", "control30")
REPLAY_TOLERANCES = {
    "logical_loss_relative": 1e-4,
    "raw_parameter_relative_L2": 1e-5,
    "raw_parameter_max_absolute": 1e-5,
}
TRAIN_ANNOTATION_SHA256 = "da59e3c5dcd301c3bc699e14d95cad7de84f36ccefc2614411c4049fae2c96cb"
TRAIN_MANIFEST_SHA256 = "df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d"
PRETRAINED_SHA256 = "911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720"
_PRETRAINED_NAME = "ResNet18_vd_pretrained_from_paddle.pth"
_KEYS = {
    "schema_version", "kind", "campaign_id", "stage", "arm", "run_id",
    "repo_root", "output_dir", "config", "num_workers", "prefetch_factor",
    "torch_version", "code_files", "train_core", "pretrained",
    "development_binding", "epoch_order_sha256", "policy_bundle",
    "expected_gpu_uuid", "paired_reference", "replay_source",
}
_FORBIDDEN = {
    "confirmatory", "test", "visdrone2019-det-test-dev",
    "visdrone2019-det-test-challenge",
}


def _fail(message: str) -> None:
    raise ControlWorkerError(message)


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail(name + " must be a lowercase SHA-256")
    return value


def _path(value: Any, name: str) -> Path:
    if type(value) is not str or not value or not Path(value).is_absolute():
        _fail(name + " must be an explicit absolute path")
    path = Path(value)
    for candidate in (path, path.resolve()):
        if any(part.casefold() in _FORBIDDEN or
               part.casefold().startswith("confirmatory_") for part in candidate.parts):
            _fail(name + " crosses a forbidden data role")
    if path != path.resolve():
        _fail(name + " must be canonical and may not redirect through symlinks")
    return path


def _reference(reference: Any, name: str, *, verify: bool = False,
               expected_name: str | None = None) -> dict[str, Any]:
    if type(reference) is not dict:
        _fail(name + " must be a file reference")
    required = {"path", "sha256", "size_bytes"}
    if not required.issubset(reference):
        _fail(name + " reference is incomplete")
    path = _path(reference["path"], name)
    if expected_name is not None and path.name != expected_name:
        _fail(name + " filename differs from its authority")
    _sha(reference["sha256"], name)
    if type(reference["size_bytes"]) is not int or reference["size_bytes"] < 0:
        _fail(name + " has an invalid byte size")
    if reference.get("sha256_scope", "complete_file_bytes") != "complete_file_bytes":
        _fail(name + " must hash the complete file bytes")
    if verify:
        actual = file_reference(path)
        if any(actual[key] != reference[key] for key in required):
            _fail(name + " complete-file reference mismatch")
    return copy.deepcopy(reference)


def _read_bound_bytes(reference: dict, name: str) -> bytes:
    checked = _reference(reference, name)
    fd = os.open(checked["path"], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            _fail(name + " must be a regular file")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        _fail(name + " changed during its verified read")
    if len(raw) != checked["size_bytes"] or hashlib.sha256(raw).hexdigest() != checked["sha256"]:
        _fail(name + " complete-file reference mismatch")
    return raw


def _read_json_reference(reference: dict, name: str) -> Any:
    # Parse precisely the bytes just verified, never a second unverified read.
    return strict_json_loads(_read_bound_bytes(reference, name))


def _same(first: Any, second: Any, name: str) -> None:
    if canonical_sha256(first) != canonical_sha256(second):
        _fail(name + " mismatch")


def frozen_epoch_orders() -> dict[str, str]:
    return {
        str(epoch): canonical_sha256([
            list(batch) for batch in logical_batch_indices(
                4869, seed=0, epoch=epoch, logical_batch_size=16,
            )
        ])
        for epoch in range(1, 31)
    }


def validate_worker_contract(contract: Any, *, verify_files: bool = True) -> dict[str, Any]:
    """Reject scope/config/reference drift before model construction or admission."""
    if type(contract) is not dict or set(contract) != _KEYS:
        _fail("worker contract schema keys mismatch")
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1:
        _fail("worker contract schema version differs")
    if contract["kind"] != "v2b_640_control_worker" or contract["stage"] not in STAGES:
        _fail("unknown worker kind/stage")
    if contract["arm"] not in ("A", "B"):
        _fail("worker arm must be A or B")
    for name in ("campaign_id", "run_id"):
        if (type(contract[name]) is not str or
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", contract[name]) is None):
            _fail(name + " must be a bounded safe identifier")
    root = _path(contract["repo_root"], "repo_root")
    output = _path(contract["output_dir"], "output_dir")
    if root == output or root.is_relative_to(output):
        _fail("output may not contain the source checkout")
    config = V2BConfig() if contract["arm"] == "A" else V2BConfig(
        physical_batch_size=8, accumulation_steps=2,
    )
    _same(contract["config"], asdict(config), "frozen model configuration")
    if type(contract["num_workers"]) is not int or contract["num_workers"] not in (2, 4):
        _fail("exactly the CPU-verified worker count 2 or 4 must be frozen")
    if type(contract["prefetch_factor"]) is not int or contract["prefetch_factor"] != 2:
        _fail("worker prefetch factor must remain two")
    if type(contract["torch_version"]) is not str or contract["torch_version"] != str(torch.__version__):
        _fail("PyTorch version differs from the order/construction authority")
    if (type(contract["expected_gpu_uuid"]) is not str or
            re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                         contract["expected_gpu_uuid"]) is None):
        _fail("the full authorized GPU UUID is required")
    code = contract["code_files"]
    if type(code) is not dict or not code:
        _fail("frozen code file references are required")
    for name, reference in code.items():
        if (type(name) is not str or Path(name).is_absolute() or
                str(Path(name)) != name or ".." in Path(name).parts):
            _fail("code files must use exact checkout-relative names")
        checked = _reference(reference, "code " + name)
        if Path(checked["path"]) != root / name:
            _fail("code reference belongs to another checkout")
    data = contract["train_core"]
    if type(data) is not dict or set(data) != {
        "annotation", "manifest", "image_root", "expected_samples", "expected_batches_per_epoch",
    }:
        _fail("train_core authority schema differs")
    for name, filename, sha, size in (
        ("annotation", "train_core_coco.json", TRAIN_ANNOTATION_SHA256, 51148042),
        ("manifest", "train_core_manifest.json", TRAIN_MANIFEST_SHA256, 2955809),
    ):
        checked = _reference(data[name], "train_core " + name, expected_name=filename)
        if checked["sha256"] != sha or checked["size_bytes"] != size:
            _fail("train_core " + name + " is not the frozen input authority")
    _path(data["image_root"], "train_core image_root")
    if (type(data["expected_samples"]) is not int or data["expected_samples"] != 4869 or
            type(data["expected_batches_per_epoch"]) is not int or
            data["expected_batches_per_epoch"] != 304):
        _fail("the complete train_core/drop-five logical epoch must remain fixed")
    pretrained = _reference(contract["pretrained"], "pretrained", expected_name=_PRETRAINED_NAME)
    if pretrained["sha256"] != PRETRAINED_SHA256 or pretrained["size_bytes"] != 44878642:
        _fail("pretrained authority differs")
    orders = contract["epoch_order_sha256"]
    if type(orders) is not dict or set(orders) != {str(epoch) for epoch in range(1, 31)}:
        _fail("all thirty epoch order hashes must be frozen before launch")
    for name, value in orders.items():
        _sha(value, "epoch " + name + " order")
    _same(orders, frozen_epoch_orders(), "predefined logical sample order")
    if type(contract["policy_bundle"]) is not dict or not contract["policy_bundle"]:
        _fail("a frozen native hardware policy bundle is required")
    if type(contract["development_binding"]) is not dict:
        _fail("fixed development evaluator/data binding is required")
    # Validate role paths before opening any contract-supplied file.
    development = contract["development_binding"]
    for name in ("annotation", "manifest"):
        _reference(development.get(name), "development " + name,
                   expected_name="development_coco.json" if name == "annotation"
                   else "development_manifest.json")
    _path(development.get("image_root"), "development image_root")
    paired, replay = contract["paired_reference"], contract["replay_source"]
    if contract["arm"] == "A":
        if paired is not None:
            _fail("A must not consume a paired B reference")
    elif paired is None:
        _fail("B must bind the completed corresponding A run before launch")
    if paired is not None:
        _reference(paired, "paired worker result", expected_name="worker-result.json")
    if (contract["stage"] == "smoke_replay") != (replay is not None):
        _fail("only a fresh smoke_replay worker may restore the smoke checkpoint")
    if replay is not None:
        _reference(replay, "replay source", expected_name="worker-result.json")
    checked = strict_json_loads(canonical_json_bytes(contract))
    if verify_files:
        if Path(__file__).resolve() != root / "src/sparse_rtdetr/baseline/training_v2b_control.py":
            _fail("control module was imported from another checkout")
        for name, reference in code.items():
            _reference(reference, "code " + name, verify=True)
        for name in ("annotation", "manifest"):
            _reference(data[name], "train_core " + name, verify=True)
        _reference(pretrained, "pretrained", verify=True)
        from .training_v2b_development import validate_development_binding
        validate_development_binding(development, verify_files=True)
    return checked


def _environment(contract: dict) -> dict:
    expected = {
        "CUDA_VISIBLE_DEVICES": contract["expected_gpu_uuid"],
        "MKL_THREADING_LAYER": "GNU", "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    observed = {name: os.environ.get(name) for name in expected}
    _same(observed, expected, "process startup environment")
    if not sys.dont_write_bytecode:
        _fail("the worker must run with bytecode generation disabled")
    return observed


def _tensor_identity(value: torch.Tensor) -> dict:
    plain = value.detach().cpu().contiguous()
    return {
        "dtype": str(plain.dtype), "shape": list(plain.shape),
        "sha256": hashlib.sha256(plain.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def _state_identity(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": _tensor_identity(value)}
    if isinstance(value, np.ndarray):
        return {"ndarray": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, Mapping):
        return {"mapping_type": type(value).__name__, "items": [
            [type(key).__name__, key, _state_identity(child)]
            for key, child in sorted(value.items(), key=lambda row: (type(row[0]).__name__, str(row[0])))
        ]}
    if isinstance(value, (tuple, list)):
        return {"sequence_type": type(value).__name__,
                "items": [_state_identity(child) for child in value]}
    if type(value) in (str, int, bool, float, type(None)):
        return value
    _fail("unsupported live state value: " + type(value).__name__)


def _rng_identity(runtime: Any) -> dict:
    return {
        "python": canonical_sha256(_state_identity(random.getstate())),
        "numpy": canonical_sha256(_state_identity(np.random.get_state())),
        "torch_cpu": _tensor_identity(torch.random.default_generator.get_state()),
        "cuda": {name: _tensor_identity(value) for name, value in capture_cuda_rng(runtime).items()},
    }


def _bn_state(model: nn.Module) -> dict:
    result = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if not module.track_running_stats:
            _fail("the frozen model requires tracked BatchNorm statistics")
        row = {"training": module.training, "num_batches_tracked": int(module.num_batches_tracked.item())}
        for key in ("running_mean", "running_var"):
            value = getattr(module, key).detach().cpu()
            if not bool(torch.isfinite(value).all()):
                _fail("nonfinite BatchNorm " + name + "." + key)
            row[key] = {
                **_tensor_identity(value),
                "min": float(value.min().item()), "max": float(value.max().item()),
                "L2": float(torch.linalg.vector_norm(value.double()).item()),
            }
        result[name] = row
    if not result:
        _fail("no actual BatchNorm modules were observed")
    return result


def _clocks(components: Any, loader: Any) -> dict:
    engine, optimizer = components.engine, components.optimizer
    state = engine.state_dict()
    cursor = loader.state_dict()
    updates = state["optimizer_updates"]
    if (cursor["optimizer_updates"] != updates or
            components.ema.updates != updates or components.warmup.last_step != updates):
        _fail("optimizer/loader/EMA/warmup update clock mismatch")
    optimizer_steps, named_steps = [], {}
    parameter_names = {id(parameter): name for name, parameter in components.model.named_parameters()}
    for parameter, values in optimizer.state.items():
        if not values:
            continue
        if "step" not in values or id(parameter) not in parameter_names:
            _fail("AdamW parameter state has no recognized parameter/step clock")
        value = values["step"]
        step = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        # A conditionally unused parameter legitimately has fewer Adam updates.
        # Preserve every actual per-parameter clock instead of inventing updates.
        if not math.isfinite(step) or step != int(step) or not 0 <= step <= updates:
            _fail("AdamW parameter update clock is outside successful logical updates")
        optimizer_steps.append(int(step))
        named_steps[parameter_names[id(parameter)]] = int(step)
    if updates and not optimizer_steps:
        _fail("the optimizer has no parameter update state")
    completed_epochs = state["epoch"] - int(state["epoch_active"])
    duration = components.warmup.warmup_duration
    first_scheduler_epoch = math.ceil(duration / loader.batches_per_epoch)
    expected_scheduler = max(0, completed_epochs - first_scheduler_epoch + 1)
    if (components.scheduler.last_epoch != expected_scheduler or
            components.scheduler._step_count != expected_scheduler + 1):
        _fail("the nominal epoch scheduler clock differs")
    return {
        "engine": state, "loader": cursor, "ema_updates": components.ema.updates,
        "warmup": _state_identity(components.warmup.state_dict()),
        "scheduler": _state_identity(components.scheduler.state_dict()),
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        "adam_parameter_count_with_state": len(optimizer_steps),
        "adam_step_values": sorted(set(optimizer_steps)),
        "adam_steps_by_parameter": named_steps,
    }


def _finite_auxiliary_state(components: Any) -> dict:
    tensors = []
    for values in components.optimizer.state.values():
        tensors.extend(value.detach() for value in values.values()
                       if isinstance(value, torch.Tensor) and value.is_floating_point())
    tensors.extend(parameter.detach() for parameter in components.ema.module.parameters())
    # Anchors/position caches can contain legal infinity. BN buffers and every
    # parameter/moment must instead be finite, and checkpoints validate caches.
    for module in components.ema.module.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            tensors.extend([module.running_mean, module.running_var])
    by_device = {}
    for value in tensors:
        by_device.setdefault(value.device, []).append(torch.isfinite(value).all())
    if any(not bool(torch.stack(values).all()) for values in by_device.values()):
        _fail("nonfinite EMA parameter, BatchNorm buffer or AdamW state")
    return {"finite": True, "tensors_checked": len(tensors)}


def capture_control_state(components: Any, loader: Any, *, full: bool = False) -> dict:
    state = {
        "clocks": _clocks(components, loader),
        "rng": _rng_identity(components.runtime),
        "raw_bn": _bn_state(components.model), "ema_bn": _bn_state(components.ema.module),
    }
    if full:
        state.update(
            raw_state_sha256=canonical_sha256(_state_identity(components.model.state_dict())),
            raw_parameters=initial_parameter_reference({
                name: value.detach().cpu() for name, value in components.model.named_parameters()
            }),
            ema_state_sha256=canonical_sha256(_state_identity(components.ema.state_dict())),
            optimizer_state_sha256=canonical_sha256(_state_identity(components.optimizer.state_dict())),
            model_training={name: module.training for name, module in components.model.named_modules()},
            ema_training={name: module.training for name, module in components.ema.module.named_modules()},
        )
    return state


class _ForwardObservation:
    def __init__(self, components: Any):
        self.components = components
        self.handles = []
        self.model_calls, self.criterion_calls = [], []
        self.bn_calls = {name: 0 for name, module in components.model.named_modules()
                         if isinstance(module, nn.modules.batchnorm._BatchNorm)}

    def _pre(self, destination):
        def observe(module, args):
            kind = self.components.runtime.device.type
            row = {"autocast_enabled": bool(torch.is_autocast_enabled(kind)),
                   "autocast_dtype": str(torch.get_autocast_dtype(kind))}
            if args and isinstance(args[0], torch.Tensor):
                row["physical_batch_size"] = int(args[0].shape[0])
            destination.append(row)
        return observe

    def __enter__(self):
        self.handles.append(self.components.model.register_forward_pre_hook(self._pre(self.model_calls)))
        self.handles.append(self.components.criterion.register_forward_pre_hook(self._pre(self.criterion_calls)))
        for name, module in self.components.model.named_modules():
            if name in self.bn_calls:
                def observe(module, args, output, name=name):
                    if not module.training:
                        _fail("a frozen-policy live BN was observed in evaluation mode")
                    self.bn_calls[name] += 1
                self.handles.append(module.register_forward_hook(observe))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def evidence(self) -> dict:
        count = self.components.config.accumulation_steps
        if len(self.model_calls) != count or len(self.criterion_calls) != count:
            _fail("model/criterion physical forward count mismatch")
        for row in self.model_calls + self.criterion_calls:
            if row["autocast_enabled"] is not True or row["autocast_dtype"] != "torch.bfloat16":
                _fail("actual model/criterion AMP context differs from BF16")
        if any(row.get("physical_batch_size") != self.components.config.physical_batch_size
               for row in self.model_calls):
            _fail("actual model physical batch size differs")
        if any(value != count for value in self.bn_calls.values()):
            _fail("actual BN forward count differs from physical microbatches")
        return {"model": self.model_calls, "criterion": self.criterion_calls,
                "bn_forward_calls": self.bn_calls, "grad_scaler_present": False}


def verify_input_receipt(receipt: Any, *, expected_indices: list[int] | tuple[int, ...],
                         epoch: int, logical_batch_index: int,
                         paired_receipt: Any = None) -> None:
    if type(receipt) is not dict or set(receipt) != {
        "epoch", "logical_batch_index", "binding_sha256", "samples",
        "augmented_sha256", "receipt_sha256",
    }:
        _fail("logical input receipt schema mismatch")
    if receipt["epoch"] != epoch or receipt["logical_batch_index"] != logical_batch_index:
        _fail("logical input receipt position differs")
    for name in ("binding_sha256", "augmented_sha256", "receipt_sha256"):
        _sha(receipt[name], name)
    body = {name: value for name, value in receipt.items() if name != "receipt_sha256"}
    if canonical_sha256(body) != receipt["receipt_sha256"]:
        _fail("logical input receipt hash mismatch")
    samples = receipt["samples"]
    if type(samples) is not list or len(samples) != len(expected_indices):
        _fail("logical input sample cardinality mismatch")
    for offset, (sample, index) in enumerate(zip(samples, expected_indices)):
        if (type(sample) is not dict or type(sample.get("index")) is not int or sample["index"] != index
                or sample.get("epoch") != epoch or
                sample.get("order_position") != logical_batch_index * len(expected_indices) + offset
                or sample.get("binding_sha256") != receipt["binding_sha256"]):
            _fail("actual input differs from the predefined epoch order")
    if paired_receipt is not None:
        _same(receipt, paired_receipt, "paired actual input receipt")


def verify_window_ledger(window: dict, targets: list, *, config: V2BConfig,
                         epoch: int, logical_batch_index: int,
                         batches_per_epoch: int) -> None:
    """Check actual engine evidence against target counts and native DN geometry."""
    counts = [len(target["labels"]) for target in targets]
    total, denominator = sum(counts), max(sum(counts), 1)
    physical, accumulation = config.physical_batch_size, config.accumulation_steps
    micro_counts = [sum(counts[start:start + physical]) for start in range(0, len(counts), physical)]
    expected_groups, expected_queries = [], []
    for start in range(0, len(counts), physical):
        maximum = max(counts[start:start + physical])
        group = max(config.num_denoising // maximum, 1) if maximum and config.num_denoising else None
        expected_groups.append(group)
        expected_queries.append(2 * maximum * group if group is not None else None)
    updates = (epoch - 1) * batches_per_epoch + logical_batch_index + 1
    expected = {
        "epoch": epoch, "optimizer_updates": updates, "microsteps": updates * accumulation,
        "logical_batch_size": physical * accumulation, "target_total": total,
        "denominator": denominator, "microbatch_target_counts": micro_counts,
        "coefficients": [max(count, 1) / denominator for count in micro_counts],
        "dn_num_groups": expected_groups, "dn_query_counts": expected_queries, "phase": 0,
    }
    if len(counts) != physical * accumulation:
        _fail("ledger does not describe one complete logical batch")
    for name, value in expected.items():
        _same(window.get(name), value, "actual logical-window " + name)
    for name in ("loss", "gradient_norm_before_clip"):
        value = window.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            _fail("nonfinite logical-window " + name)
    losses = window.get("microbatch_losses")
    if type(losses) is not list or len(losses) != accumulation or any(
            type(row) is not dict or not row for row in losses):
        _fail("logical-window microbatch losses are incomplete")
    if window["gradient_norm_before_clip"] < 0:
        _fail("negative gradient norm")


def _loss_families(window: dict) -> dict:
    result = {"main": {}, "auxiliary": {}, "denoising": {}, "encoder": {}}
    for coefficient, losses in zip(window["coefficients"], window["microbatch_losses"]):
        for name, value in losses.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                _fail("nonfinite loss family")
            family = ("denoising" if "_dn_" in name else "auxiliary" if "_aux_" in name
                      else "encoder" if "_enc_" in name else "main")
            result[family][name] = result[family].get(name, 0.0) + coefficient * value
    return result


def _publish_parameters(path: Path, model: nn.Module) -> dict:
    values = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    payload = {"schema_version": 1, "parameters": values}
    stream = io.BytesIO()
    torch.save(payload, stream)
    raw = stream.getvalue()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as destination:
        destination.write(raw)
        destination.flush()
        os.fsync(destination.fileno())
    reference = file_reference(path)
    if reference["sha256"] != hashlib.sha256(raw).hexdigest():
        _fail("parameter snapshot publication changed bytes")
    return reference


def _load_parameters(reference: dict) -> dict[str, torch.Tensor]:
    raw = _read_bound_bytes(reference, "raw parameter snapshot before deserialization")
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(value) is not dict or value.get("schema_version") != 1 or set(value) != {"schema_version", "parameters"}:
        _fail("parameter snapshot schema differs")
    if type(value["parameters"]) is not dict or not value["parameters"]:
        _fail("parameter snapshot has no actual parameters")
    return value["parameters"]


def compare_parameter_maps(expected: Mapping[str, torch.Tensor],
                           actual: Mapping[str, torch.Tensor]) -> dict:
    if set(expected) != set(actual) or not expected:
        _fail("replay raw parameter names differ")
    error_squared = reference_squared = max_absolute = 0.0
    elements = 0
    for name in sorted(expected):
        left, right = expected[name], actual[name]
        if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or
                left.dtype != right.dtype or left.shape != right.shape):
            _fail("replay raw parameter dtype/shape differs: " + name)
        left, right = left.detach().cpu().double(), right.detach().cpu().double()
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            _fail("nonfinite replay raw parameter: " + name)
        delta = left - right
        error_squared += float(torch.sum(delta.square()).item())
        reference_squared += float(torch.sum(left.square()).item())
        if delta.numel():
            max_absolute = max(max_absolute, float(delta.abs().max().item()))
        elements += left.numel()
    relative = math.sqrt(error_squared) / max(math.sqrt(reference_squared), 1e-12)
    if (relative > REPLAY_TOLERANCES["raw_parameter_relative_L2"] or
            max_absolute > REPLAY_TOLERANCES["raw_parameter_max_absolute"]):
        _fail("same-arm raw parameter replay exceeds frozen numerical tolerances")
    return {"relative_L2": relative, "max_absolute": max_absolute, "elements": elements,
            "thresholds": copy.deepcopy(REPLAY_TOLERANCES)}


def compare_replay_records(expected: dict, actual: dict) -> dict:
    """Exact discrete/state checks and only the predeclared floating thresholds."""
    _sha(expected.get("run_binding_sha256"), "original window run binding")
    _sha(actual.get("run_binding_sha256"), "replayed window run binding")
    _same(expected["run_binding_sha256"], actual["run_binding_sha256"], "same-arm replay run binding")
    _same(expected["input"], actual["input"], "same-arm replay input")
    for name in ("clocks", "rng", "raw_bn", "ema_bn"):
        _same(expected["state"][name], actual["state"][name], "same-arm replay " + name)
    for name in (
        "epoch", "optimizer_updates", "microsteps", "logical_batch_size", "target_total",
        "denominator", "microbatch_target_counts", "coefficients", "dn_num_groups",
        "dn_query_counts", "phase",
    ):
        _same(expected["window"][name], actual["window"][name], "same-arm replay " + name)
    _same(expected["forward_observation"], actual["forward_observation"], "same-arm replay physical forwards")
    left, right = expected["window"]["loss"], actual["window"]["loss"]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (left, right)):
        _fail("nonfinite replay logical loss")
    relative = abs(right - left) / max(abs(left), 1e-12)
    if relative > REPLAY_TOLERANCES["logical_loss_relative"]:
        _fail("same-arm logical loss exceeds frozen replay tolerance")
    parameters = compare_parameter_maps(
        _load_parameters(expected["raw_parameters_reference"]),
        _load_parameters(actual["raw_parameters_reference"]),
    )
    return {"status": "PASS", "logical_loss_relative": relative, "raw_parameters": parameters,
            "exact": ["input", "clocks", "CPU_CUDA_RNG", "BN", "DN_metadata",
                      "denominator", "physical_forwards"]}


def _ema_exact_tensor_paths(state: dict, layout: dict) -> tuple[set[tuple], set[tuple]]:
    """Identify exact BN/geometry leaves from authenticated real module layout.

    Only RTDETRTransformerv2's registered anchors may carry +inf. Its whole
    tensor, including sentinel locations and finite entries, must be byte-exact.
    An arbitrary tensor whose name ends in anchors receives no exception.
    """
    try:
        model_layout = layout["model"]
        module_types, specifications = model_layout["modules"], model_layout["state"]
        parameter_names = {row["name"] for row in model_layout["parameters"]}
        caches = model_layout["caches"]
        values = state["module"]
    except (KeyError, TypeError) as exc:
        raise ControlWorkerError("EMA authenticated module layout is incomplete") from exc
    if set(values) != set(specifications):
        _fail("EMA state keys differ from the authenticated module layout")
    exact, allowed_positive_infinity = set(), set()
    vendor = "src.zoo.rtdetr.rtdetrv2_decoder."
    cache_modules = {
        "anchors": vendor + "RTDETRTransformerv2",
        "valid_mask": vendor + "RTDETRTransformerv2",
        "num_points_scale": vendor + "MSDeformableAttention",
    }
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            _fail("EMA state contains a non-tensor module leaf")
        spec = {"shape": list(value.shape), "dtype": str(value.dtype)}
        _same(specifications[name], spec, "EMA tensor layout " + name)
        parent, _, leaf = name.rpartition(".")
        path = ("module", name)
        if name not in parameter_names and leaf in {"running_mean", "running_var", "num_batches_tracked"}:
            exact.add(path)
        if leaf in cache_modules and name not in parameter_names:
            cache_name = parent + ":persistent_cache:" + leaf
            cache = caches.get(cache_name)
            if (module_types.get(parent) != cache_modules[leaf] or type(cache) is not dict or
                    set(cache) != {"shape", "dtype", "sha256"}):
                _fail("EMA geometry cache lacks the inspected vendor module identity: " + name)
            _same(cache, _tensor_identity(value), "authenticated EMA geometry bytes " + name)
            exact.add(path)
            if leaf == "anchors":
                allowed_positive_infinity.add(path)
    return exact, allowed_positive_infinity


def _compare_state_tree(expected: Any, actual: Any, *, family: str,
                        exact_tensor_paths: set[tuple] | None = None,
                        positive_infinity_paths: set[tuple] | None = None,
                        optimizer_moments_only: bool = False) -> dict:
    """Compare structure exactly and one independent floating tensor family."""
    exact_tensor_paths = exact_tensor_paths or set()
    positive_infinity_paths = positive_infinity_paths or set()
    rows, exact_tensors, sentinel_tensors = [], [], []
    difference_squared = reference_squared = max_absolute = 0.0

    def walk(left, right, path):
        nonlocal difference_squared, reference_squared, max_absolute
        label = "/".join(str(part) for part in path) or "<root>"
        if type(left) is not type(right):
            _fail(f"{family}/{label}: state type differs")
        if isinstance(left, torch.Tensor):
            if (left.shape != right.shape or left.dtype != right.dtype or
                    left.device.type != "cpu" or right.device.type != "cpu" or
                    left.layout != torch.strided or right.layout != torch.strided):
                _fail(f"{family}/{label}: tensor shape/dtype/device/layout differs")
            if left.is_complex() or right.is_complex():
                _fail(f"{family}/{label}: complex training state is unsupported")
            floating = left.is_floating_point()
            if floating:
                if path in positive_infinity_paths:
                    for value in (left, right):
                        if bool(torch.isnan(value).any()) or bool(torch.isneginf(value).any()):
                            _fail(f"{family}/{label}: invalid anchor sentinel")
                elif not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                    _fail(f"{family}/{label}: nonfinite floating state")
            is_moment = (
                len(path) == 3 and path[0] == "state" and type(path[1]) is int
                and path[2] in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
            )
            exact = (not floating or path in exact_tensor_paths or
                     (optimizer_moments_only and not is_moment))
            if path in positive_infinity_paths and not exact:
                _fail("an anchor sentinel exception must always use whole-tensor exact comparison")
            if exact:
                _same(_tensor_identity(left), _tensor_identity(right), f"{family}/{label} exact tensor")
                exact_tensors.append(label)
                if path in positive_infinity_paths and bool(torch.isposinf(left).any()):
                    sentinel_tensors.append(label)
                return
            a, b = left.double(), right.double()
            delta = a - b
            error = float(torch.sum(delta.square()).item())
            norm = float(torch.sum(a.square()).item())
            maximum = float(delta.abs().max().item()) if delta.numel() else 0.0
            if not all(math.isfinite(number) for number in (error, norm, maximum)):
                _fail(f"{family}/{label}: nonfinite comparison norm")
            difference_squared += error
            reference_squared += norm
            max_absolute = max(max_absolute, maximum)
            rows.append({
                "path": label, "elements": left.numel(),
                "reference_L2": math.sqrt(norm), "difference_L2": math.sqrt(error),
                "relative_L2": math.sqrt(error) / max(math.sqrt(norm), 1e-12),
                "max_absolute": maximum,
            })
            return
        if isinstance(left, dict):
            # Python considers True and 1 equal as dict keys; key types are
            # nevertheless part of the saved optimizer/schema identity.
            key_signature = lambda value: sorted(
                [(type(key).__name__, repr(key)) for key in value],
            )
            if key_signature(left) != key_signature(right):
                _fail(f"{family}/{label}: state keys differ")
            for key in left:
                if type(key) not in (str, int):
                    _fail(f"{family}/{label}: unsupported state key")
                walk(left[key], right[key], path + (key,))
            return
        if isinstance(left, (list, tuple)):
            if len(left) != len(right):
                _fail(f"{family}/{label}: state sequence length differs")
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, path + (index,))
            return
        if type(left) not in (str, int, float, bool, type(None)):
            _fail(f"{family}/{label}: unsupported non-tensor state")
        if type(left) is float and (not math.isfinite(left) or not math.isfinite(right)):
            _fail(f"{family}/{label}: nonfinite scalar state")
        # Scalar floats are clocks or hyperparameters, never Adam moments.
        _same(left, right, f"{family}/{label} exact scalar")

    walk(expected, actual, ())
    if not rows:
        _fail(f"{family}: no floating continuation tensors were authenticated")
    reference_norm = math.sqrt(reference_squared)
    difference_norm = math.sqrt(difference_squared)
    relative = difference_norm / max(reference_norm, 1e-12)
    if (not all(math.isfinite(value) for value in (reference_norm, difference_norm, relative, max_absolute))
            or relative > REPLAY_TOLERANCES["raw_parameter_relative_L2"]
            or max_absolute > REPLAY_TOLERANCES["raw_parameter_max_absolute"]):
        _fail(f"{family}: continuation exceeds frozen relative_L2/max_absolute 1e-5 tolerances "
              f"(reference_L2={reference_norm}, difference_L2={difference_norm}, "
              f"relative_L2={relative}, max_absolute={max_absolute})")
    return {
        "status": "PASS", "family": family, "floating_tensor_count": len(rows),
        "reference_L2": reference_norm, "difference_L2": difference_norm,
        "relative_L2": relative, "max_absolute": max_absolute,
        "thresholds": {"relative_L2": REPLAY_TOLERANCES["raw_parameter_relative_L2"],
                       "max_absolute": REPLAY_TOLERANCES["raw_parameter_max_absolute"]},
        "relative_denominator": "max(reference_family_L2,1e-12)",
        "worst_absolute_tensor": max(rows, key=lambda row: row["max_absolute"]),
        "worst_relative_tensor_observation": max(rows, key=lambda row: row["relative_L2"]),
        "tensor_observations": rows, "exact_tensors": exact_tensors,
        "positive_infinity_sentinel_exception": {
            "policy": "only_authenticated_RTDETRv2_decoder_anchors_whole_tensor_byte_exact",
            "tensors_with_sentinel": sentinel_tensors,
        },
    }


def _continuation_payload(reference: dict, binding: dict) -> dict:
    from .training_v2b_checkpoint import inspect_checkpoint
    checked = _reference(reference, "window-four checkpoint",
                         expected_name="checkpoint-window-04.pt")
    inspection = inspect_checkpoint(
        checked["path"], expected_sha256=checked["sha256"], expected_binding=binding,
    )
    if (inspection["schema_version"] != 2 or inspection["optimizer_updates"] != 4 or
            inspection["epoch"] != 1 or inspection["epoch_active"] is not True):
        _fail("continuation evidence is not the active fourth-window checkpoint")
    for name in ("size_bytes", "binding_sha256", "epoch", "optimizer_updates", "schema_version"):
        if name in checked and checked[name] != inspection[name]:
            _fail("checkpoint reference/inspection metadata differs: " + name)
    # Inspect the authenticated schema first, then deserialize precisely another
    # verified read of those same immutable bytes. No CUDA API is used.
    payload = torch.load(
        io.BytesIO(_read_bound_bytes(checked, "window-four checkpoint")),
        map_location="cpu", weights_only=True,
    )
    for name in ("ema", "ema_layout", "optimizer", "optimizer_layout"):
        if type(payload.get(name)) is not dict:
            _fail("checkpoint lacks complete continuation state: " + name)
    return payload


def compare_continuation_checkpoints(expected_reference: dict, actual_reference: dict, *,
                                    binding: dict) -> dict:
    """CPU-only, hash-authenticated EMA/moment acceptance after same-arm replay."""
    expected = _continuation_payload(expected_reference, binding)
    actual = _continuation_payload(actual_reference, binding)
    # Every layout (including geometry bytes), mode, RNG, sampler and non-loss
    # clock is fixed before accepting approximate floating continuation state.
    for name in ("ema_layout", "model_layout", "optimizer_layout", "engine", "scheduler", "warmup",
                 "rng", "runtime", "sampler", "model_training"):
        _same(_state_identity(expected[name]), _state_identity(actual[name]),
              "continuation checkpoint exact " + name)
    exact, sentinel = _ema_exact_tensor_paths(expected["ema"], expected["ema_layout"])
    actual_exact, actual_sentinel = _ema_exact_tensor_paths(actual["ema"], actual["ema_layout"])
    if exact != actual_exact or sentinel != actual_sentinel:
        _fail("EMA continuation exact/sentinel tensor roles differ")
    ema = _compare_state_tree(
        expected["ema"], actual["ema"], family="EMA",
        exact_tensor_paths=exact, positive_infinity_paths=sentinel,
    )
    moments = _compare_state_tree(
        expected["optimizer"], actual["optimizer"], family="AdamW_moments",
        optimizer_moments_only=True,
    )
    return {
        "status": "PASS", "expected_checkpoint": copy.deepcopy(expected_reference),
        "actual_checkpoint": copy.deepcopy(actual_reference),
        "EMA": ema, "AdamW_moments": moments,
        "families_are_aggregated_separately": True,
        "checkpoint_restore_boundary_still_requires_complete_exact_state": True,
    }


def _successful_source(reference: dict, *, stage: str, arm: str, campaign_id: str) -> dict:
    value = _read_json_reference(reference, "completed worker result")
    if (type(value) is not dict or value.get("status") != "PASS" or
            value.get("stage") != stage or value.get("arm") != arm or
            value.get("campaign_id") != campaign_id):
        _fail("paired/replay source is not the required successful worker")
    return value


def _receipt_index(result: dict) -> dict[tuple[int, int], dict]:
    references = result.get("receipts")
    if type(references) is not list or not references:
        _fail("completed worker lacks durable logical window receipts")
    index = {}
    for reference in references:
        _reference(reference, "window receipt")
        key = (reference.get("epoch"), reference.get("logical_batch_index"))
        if any(type(item) is not int for item in key) or key in index:
            _fail("duplicate or invalid durable logical receipt position")
        index[key] = reference
    return index


def _verify_common_initial_state(expected: dict, actual: dict) -> None:
    for name in (
        "rng", "raw_bn", "ema_bn", "raw_state_sha256", "raw_parameters",
        "ema_state_sha256", "optimizer_state_sha256", "model_training", "ema_training",
    ):
        _same(expected[name], actual[name], "actual shared CUDA initialization " + name)


def _common_identity(contract: dict, binding: dict, loader: Any) -> dict:
    config = copy.deepcopy(binding["config"])
    config.pop("physical_batch_size")
    config.pop("accumulation_steps")
    return {
        "campaign_id": contract["campaign_id"], "config": config,
        "code": binding["code"], "initial_weights": binding["initial_weights"],
        "train_core": binding["data"], "torch_version": str(torch.__version__),
        "worker_environment": validate_worker_environment(loader.config),
        "epoch_order_sha256": contract["epoch_order_sha256"],
        "development_binding_sha256": contract["development_binding"]["binding_sha256"],
        "policy_bundle_sha256": canonical_sha256(contract["policy_bundle"]),
    }


def _measure_window(session: Any, batch: Any, monitor: Any, *, expected_indices,
                    paired_record: dict | None = None,
                    replay_record: dict | None = None) -> dict:
    receipt = batch.evidence()
    verify_input_receipt(
        receipt, expected_indices=expected_indices, epoch=batch.epoch,
        logical_batch_index=batch.logical_batch_index,
        paired_receipt=None if paired_record is None else paired_record["input"],
    )
    if replay_record is not None:
        _same(receipt, replay_record["input"], "fresh replay input before model update")
    c = session.components
    before_bn = _bn_state(c.model)
    before = monitor.check(stage="before_window")
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with _ForwardObservation(c) as observations:
        record = session.train_batch(batch, hardware_probe=monitor.admission())
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    after = monitor.check(stage="after_window")
    memory = {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    verify_window_ledger(
        record["window"], batch.targets, config=c.config, epoch=batch.epoch,
        logical_batch_index=batch.logical_batch_index, batches_per_epoch=session.loader.batches_per_epoch,
    )
    state = capture_control_state(c, session.loader)
    count = c.config.accumulation_steps
    for name, row in state["raw_bn"].items():
        if row["num_batches_tracked"] - before_bn[name]["num_batches_tracked"] != count:
            _fail("BN tracked forward clock differs from physical microbatches")
    health = _finite_auxiliary_state(c)
    return {
        "schema_version": 1, "run_binding_sha256": session.binding["binding_sha256"],
        **record, "state": state,
        "bn_before": before_bn, "forward_observation": observations.evidence(),
        "loss_families_weighted": _loss_families(record["window"]),
        "auxiliary_numerical_health": health, "memory": memory,
        "synchronized_train_window_seconds": elapsed,
        "hardware_before": before, "hardware_after": after,
    }


def _write_checkpoint(session: Any, output: Path, name: str, monitor: Any) -> dict:
    monitor.check(stage="before_checkpoint")
    checkpoint = session.save(output / (name + ".pt"))
    torch.cuda.synchronize(0)
    monitor.check(stage="after_checkpoint")
    state = capture_control_state(session.components, session.loader, full=True)
    state_reference = write_exclusive_json(output / (name + "-state.json"), state)
    return {"checkpoint": checkpoint, "state_reference": state_reference}


def _run_active_windows(session: Any, monitor: Any, output: Path, *, limit: int,
                        paired_index: dict, replay_index: dict, save_midpoint: bool,
                        snapshots: bool, checkpoints: dict, receipts: list,
                        replay_checks: list) -> None:
    loader, c = session.loader, session.components
    epoch = c.engine.epoch
    plan = list(logical_batch_indices(
        len(loader.dataset), seed=c.config.seed, epoch=epoch,
        logical_batch_size=c.config.logical_batch_size,
    ))
    receipt_dir = output / "receipts" / f"epoch-{epoch:03d}"
    receipt_dir.mkdir(parents=True, exist_ok=False)
    produced = 0
    with contextlib.closing(iter(loader)) as iterator:
        while produced < limit:
            load_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                _fail("loader ended before the exact declared number of windows")
            load_seconds = time.perf_counter() - load_started
            key = (batch.epoch, batch.logical_batch_index)
            paired = None
            if paired_index:
                if key not in paired_index:
                    _fail("B has no authenticated A receipt for its next logical window")
                paired = _read_json_reference(paired_index[key], "paired A window")
            replay = None
            if replay_index:
                if key not in replay_index:
                    _fail("fresh replay lacks the original next logical window")
                replay = _read_json_reference(replay_index[key], "original replay window")
            record = _measure_window(
                session, batch, monitor, expected_indices=plan[batch.logical_batch_index],
                paired_record=paired, replay_record=replay,
            )
            record["logical_input_wait_seconds"] = load_seconds
            window_id = c.engine.optimizer_updates
            if snapshots and window_id in (3, 4):
                record["raw_parameters_reference"] = _publish_parameters(
                    output / f"raw-parameters-window-{window_id:02d}.pt", c.model,
                )
            if replay is not None:
                replay_checks.append(compare_replay_records(replay, record))
            reference = write_exclusive_json(
                receipt_dir / f"window-{batch.logical_batch_index + 1:04d}.json", record,
            )
            receipts.append({**reference, "epoch": batch.epoch,
                             "logical_batch_index": batch.logical_batch_index})
            produced += 1
            if save_midpoint and window_id == 2:
                checkpoints["window_2"] = _write_checkpoint(
                    session, output, "checkpoint-window-02", monitor,
                )
    if produced != limit or loader.has_active_iterator:
        _fail("logical loop failed to close at its declared boundary")


def run_worker(contract: dict, *, contract_reference: dict) -> dict:
    """Run one admitted stage. No retry, subprocess replay or implicit continuation."""
    checked = validate_worker_contract(contract, verify_files=False)
    _same(_read_json_reference(contract_reference, "worker contract"), checked, "contract bytes")
    output = Path(checked["output_dir"])
    output.mkdir(parents=False, exist_ok=False)
    monitor = None
    receipts, replay_checks, evaluations = [], [], []
    checkpoints = {}
    report = {
        "schema_version": 1, "status": "RUNNING", "stage": checked["stage"],
        "arm": checked["arm"], "run_id": checked["run_id"],
        "campaign_id": checked["campaign_id"],
        "contract_reference": copy.deepcopy(contract_reference),
        "worker_pid": os.getpid(), "receipts": receipts, "checkpoints": checkpoints,
        "evaluations": evaluations, "replay_checks": replay_checks,
        "guardian_clearance_required_after_worker_exit": True,
        "scientific_certified": False, "automatic_retry_or_batch_fallback": False,
    }
    try:
        # Import all integrations before CPU initialization so source inventory
        # is identical in original/replay/A/B workers.
        from .training_v2b_admission import MonitoredHardwareSession
        from .training_v2b_development import evaluate_development, preview_development
        environment = _environment(checked)
        checked = validate_worker_contract(checked, verify_files=True)
        report["startup_environment"] = environment
        write_exclusive_json(output / "worker-start.json", report)
        config = V2BConfig(**checked["config"])
        data = checked["train_core"]
        loader = build_train_core_loader(
            TrainCoreDataConfig(
                seed=config.seed, input_size=config.input_size,
                logical_batch_size=config.logical_batch_size,
                num_workers=checked["num_workers"], prefetch_factor=checked["prefetch_factor"],
            ),
            annotation_file=data["annotation"]["path"], annotation_sha256=data["annotation"]["sha256"],
            manifest_file=data["manifest"]["path"], manifest_sha256=data["manifest"]["sha256"],
            image_root=data["image_root"], repo_root=checked["repo_root"],
        )
        if len(loader.dataset) != 4869 or loader.batches_per_epoch != 304:
            _fail("actual loader does not describe the frozen complete train_core epoch")
        arguments = {
            "repo_root": checked["repo_root"],
            "pretrained_path": checked["pretrained"]["path"],
            "pretrained_sha256": checked["pretrained"]["sha256"],
        }
        cpu = build_v2b_components(config, **arguments)
        binding = build_train_core_run_binding(
            cpu, loader, run_id=checked["run_id"], repo_root=checked["repo_root"],
            additional_code_paths={name: reference["path"] for name, reference in checked["code_files"].items()},
            requested_device="cuda:0", cuda_gpu_uuid=checked["expected_gpu_uuid"],
        )
        if not set(binding["code"]).issubset(checked["code_files"]):
            _fail("actual imported source set is outside the frozen code inventory")
        report["binding_reference"] = write_exclusive_json(output / "run-binding.json", binding)
        report["initialization"] = cpu.initialization
        report["common_identity"] = _common_identity(checked, binding, loader)
        paired, original = None, None
        paired_index, replay_index = {}, {}
        expected_source_stage = "control30" if checked["stage"] == "control30" else "smoke"
        if checked["paired_reference"] is not None:
            paired = _successful_source(
                checked["paired_reference"], stage=expected_source_stage,
                arm="A", campaign_id=checked["campaign_id"],
            )
            _same(paired["common_identity"], report["common_identity"], "A/B frozen common identity")
            paired_index = _receipt_index(paired)
            expected_peer_keys = (
                {(epoch, index) for epoch in range(1, 31) for index in range(304)}
                if expected_source_stage == "control30" else {(1, index) for index in range(4)}
            )
            if set(paired_index) != expected_peer_keys:
                _fail("A source is not a complete counterpart for the declared B stage")
        if checked["replay_source"] is not None:
            original = _successful_source(
                checked["replay_source"], stage="smoke", arm=checked["arm"],
                campaign_id=checked["campaign_id"],
            )
            if original["run_id"] != checked["run_id"] or original["worker_pid"] == os.getpid():
                _fail("checkpoint replay requires a fresh process and the same run identity")
            _same(original["common_identity"], report["common_identity"], "same-arm replay common identity")
            source_binding = _read_json_reference(original["binding_reference"], "source smoke binding")
            _same(source_binding, binding, "same-arm checkpoint run binding")
            replay_index = _receipt_index(original)
            if set(replay_index) != {(1, index) for index in range(4)}:
                _fail("same-arm replay source is not a complete four-window smoke")
        monitor_dir = output / "native-hardware"
        # The native session exclusively creates its own evidence directory.
        monitor = MonitoredHardwareSession(
            binding, checked["policy_bundle"], monitor_dir,
            "train_core_30epoch" if checked["stage"] == "control30" else "paired_smoke",
            43200 if checked["stage"] == "control30" else 600,
        )
        monitor.start()
        runtime = prepare_runtime(
            device="cuda:0", seed=config.seed, binding=binding,
            gpu_probe=monitor.admission(), expected_gpu_uuid=checked["expected_gpu_uuid"],
        )
        components = build_v2b_components(config, runtime=runtime, **arguments)
        _same(cpu.initialization, components.initialization, "CPU/CUDA actual initialized model identity")
        del cpu
        session = V2BTrainingSession(components, loader, binding)
        report["runtime"] = runtime.identity
        initial_state = capture_control_state(components, loader, full=True)
        report["initial_state_reference"] = write_exclusive_json(output / "initial-state.json", initial_state)
        for source in (paired, original):
            if source is not None:
                _same(source["runtime"], report["runtime"], "actual paired/replay CUDA runtime identity")
                _verify_common_initial_state(
                    _read_json_reference(source["initial_state_reference"], "source initial state"),
                    initial_state,
                )

        def evaluation_gate():
            torch.cuda.synchronize(0)
            return monitor.check(stage="development_batch")

        component_map = {
            "model": components.model, "ema": components.ema,
            "postprocessor": components.postprocessor, "engine": components.engine,
            "runtime": components.runtime, "optimizer": components.optimizer,
            "scheduler": components.scheduler, "warmup": components.warmup,
        }
        if checked["stage"] in ("smoke", "smoke_replay"):
            if original is None:
                session.begin_epoch(1)
                if loader.state_dict()["order_sha256"] != checked["epoch_order_sha256"]["1"]:
                    _fail("actual epoch-one sampler differs from the frozen plan")
            else:
                midpoint = original["checkpoints"]["window_2"]
                _reference(midpoint["checkpoint"], "smoke checkpoint", verify=True,
                           expected_name="checkpoint-window-02.pt")
                monitor.check(stage="before_restore")
                session.restore(midpoint["checkpoint"]["path"], midpoint["checkpoint"]["sha256"])
                torch.cuda.synchronize(0)
                monitor.check(stage="after_restore")
                restored_state = capture_control_state(components, loader, full=True)
                _same(_read_json_reference(midpoint["state_reference"], "checkpoint boundary state"),
                      restored_state, "full checkpoint restored state")
                if components.engine.optimizer_updates != 2 or not components.engine.epoch_active:
                    _fail("replay checkpoint is not the active second logical-window boundary")
                report["restore_boundary_reference"] = write_exclusive_json(
                    output / "restored-boundary-state.json", restored_state,
                )
                report["restored_from"] = copy.deepcopy(midpoint)
            _run_active_windows(
                session, monitor, output, limit=4 if original is None else 2,
                paired_index=paired_index, replay_index=replay_index,
                save_midpoint=original is None, snapshots=True,
                checkpoints=checkpoints, receipts=receipts, replay_checks=replay_checks,
            )
            if components.engine.optimizer_updates != 4:
                _fail("smoke did not stop after exactly four logical windows")
            checkpoints["window_4"] = _write_checkpoint(
                session, output, "checkpoint-window-04", monitor,
            )
            before_preview = capture_control_state(components, loader, full=True)
            preview_dir = output / "development-preview"
            preview_dir.mkdir()
            evaluations.append(preview_development(
                component_map, data_binding=checked["development_binding"],
                output_dir=preview_dir, hardware_gate=evaluation_gate,
            ))
            _same(before_preview, capture_control_state(components, loader, full=True),
                  "development preview preserved complete training state")
        else:
            for epoch in range(1, 31):
                session.begin_epoch(epoch)
                if loader.state_dict()["order_sha256"] != checked["epoch_order_sha256"][str(epoch)]:
                    _fail("actual epoch sampler differs from its frozen complete plan")
                _run_active_windows(
                    session, monitor, output, limit=loader.batches_per_epoch,
                    paired_index=paired_index, replay_index={}, save_midpoint=False,
                    snapshots=False, checkpoints=checkpoints, receipts=receipts,
                    replay_checks=replay_checks,
                )
                epoch_record = session.finish_epoch()
                checkpoints[f"epoch_{epoch}"] = _write_checkpoint(
                    session, output, f"checkpoint-epoch-{epoch:03d}", monitor,
                )
                write_exclusive_json(output / f"epoch-{epoch:03d}.json", epoch_record)
                if epoch in (10, 20, 30):
                    evaluation_dir = output / f"development-epoch-{epoch:03d}"
                    evaluation_dir.mkdir()
                    before_evaluation = capture_control_state(components, loader, full=True)
                    evaluations.append(evaluate_development(
                        component_map, data_binding=checked["development_binding"],
                        output_dir=evaluation_dir, logged_epoch=epoch,
                        hardware_gate=evaluation_gate,
                    ))
                    _same(before_evaluation, capture_control_state(components, loader, full=True),
                          "development evaluation preserved complete training state")
            if (components.engine.optimizer_updates != 30 * 304 or components.engine.epoch_active
                    or len(receipts) != 30 * 304 or len(evaluations) != 3):
                _fail("thirty-epoch completion ledger is incomplete")
        report["final_state_reference"] = write_exclusive_json(
            output / "final-state.json", capture_control_state(components, loader, full=True),
        )
        report["receipt_count"] = len(receipts)
        report["logical_samples_committed"] = components.engine.optimizer_updates * 16
        report["logical_samples_executed_this_invocation"] = len(receipts) * 16
        report["replay_tolerances"] = copy.deepcopy(REPLAY_TOLERANCES)
        report["monitor_reference"] = monitor.finish()
        report["status"] = "PASS"
        write_exclusive_json(output / "worker-result.json", report)
        return report
    except BaseException as exc:
        report["status"] = "STOP_NO_RETRY"
        chain, cause = [], exc
        while cause is not None:
            chain.append({"type": type(cause).__name__, "message": str(cause)})
            cause = cause.__cause__
        report["failure"] = chain
        # Persist failure before the guardian is asked to terminate this owner.
        try:
            write_exclusive_json(output / "worker-result.json", report)
        finally:
            if monitor is not None:
                monitor.abort("worker failure: " + type(exc).__name__ + ": " + str(exc))
        raise


def _preview_evidence(result: dict) -> dict:
    evaluations = result.get("evaluations")
    if type(evaluations) is not list or len(evaluations) != 1:
        _fail("smoke must include exactly one bounded development preview")
    preview = evaluations[0]
    if (type(preview) is not dict or preview.get("role") != "development" or
            preview.get("scope") != "development_preview_only" or
            preview.get("logged_epoch") is not None or
            preview.get("full_development_evaluation") is not False or
            preview.get("scientific_comparison_eligible") is not False or
            preview.get("data", {}).get("evaluated_image_count") != 4 or
            len(preview.get("data", {}).get("image_receipts", [])) != 4 or
            preview.get("weights", {}).get("kind") != "ema" or
            preview.get("weights", {}).get("ema_updates") != 4):
        _fail("development smoke preview scope, coverage or EMA clock differs")
    if preview.get("data_binding_sha256") != result["common_identity"]["development_binding_sha256"]:
        _fail("preview data binding differs from the frozen worker")
    _same(preview["runtime"], result["runtime"], "preview actual runtime")
    published = _read_json_reference(preview["result_artifact"], "development preview result")
    _same(published, {key: value for key, value in preview.items() if key != "result_artifact"},
          "development preview published evidence")
    for name, reference in (
        ("preview predictions", preview["prediction_artifact"]),
        ("preview COCO tensor", preview["coco_secondary"]["tensor_artifact"]),
        ("preview primary metrics", preview["primary_legacy_formal_gt"]["result_artifact"]),
    ):
        _reference(reference, name, verify=True)
    for phase in ("before_batch", "after_batch"):
        rows = [row for row in preview["native_gate_callbacks"] if row.get("phase") == phase]
        if len(rows) != 1 or rows[0].get("batch") != 0:
            _fail("preview did not record its one native-gated batch")
    return preview


def _compare_previews(expected: dict, actual: dict) -> dict:
    for name in ("data", "data_binding_sha256", "policy"):
        _same(expected[name], actual[name], "development preview " + name)
    _same(expected["coco_secondary"]["axes"], actual["coco_secondary"]["axes"], "COCO preview axes")
    _same(expected["primary_legacy_formal_gt"]["algorithm_id"],
          actual["primary_legacy_formal_gt"]["algorithm_id"], "primary preview evaluator identity")
    return {
        "evaluated_images": 4, "actual_data_and_metric_axes_exact": True,
        "ema_state_exact": expected["weights"]["state_sha256"] == actual["weights"]["state_sha256"],
        "prediction_bytes_exact": expected["prediction_artifact"]["sha256"] == actual["prediction_artifact"]["sha256"],
        "primary_metrics_exact": expected["primary_legacy_formal_gt"]["metrics"] == actual["primary_legacy_formal_gt"]["metrics"],
        "secondary_metrics_exact": expected["coco_secondary"]["metrics"] == actual["coco_secondary"]["metrics"],
        "numerical_equalities_are_observations_not_new_replay_tolerances": True,
        "scientific_comparison_eligible": False,
    }


def _result_from_dir(directory: str | Path) -> tuple[dict, dict]:
    path = _path(str(Path(directory).absolute()), "smoke output") / "worker-result.json"
    reference = file_reference(path)
    result = _read_json_reference(reference, "smoke worker result")
    if type(result) is not dict or result.get("status") != "PASS":
        _fail("smoke worker did not complete successfully")
    return result, reference


def compare_smoke_outputs(a_smoke_dir: str | Path, a_replay_dir: str | Path,
                          b_smoke_dir: str | Path, b_replay_dir: str | Path) -> dict:
    """CPU comparison only. Controller separately proves guardian final clearance."""
    try:
        loaded = [_result_from_dir(path) for path in
                  (a_smoke_dir, a_replay_dir, b_smoke_dir, b_replay_dir)]
        a, ar, b, br = [row[0] for row in loaded]
        comparisons = []
        for base, replay, arm in ((a, ar, "A"), (b, br, "B")):
            if (base["stage"] != "smoke" or replay["stage"] != "smoke_replay" or
                    base["arm"] != arm or replay["arm"] != arm or
                    base["run_id"] != replay["run_id"] or base["worker_pid"] == replay["worker_pid"]):
                _fail("smoke/replay stage, arm, process or run identity differs")
            _same(base["common_identity"], replay["common_identity"], "same-arm smoke identities")
            _same(base["runtime"], replay["runtime"], "same-arm actual CUDA runtime")
            _verify_common_initial_state(
                _read_json_reference(base["initial_state_reference"], "original initialization"),
                _read_json_reference(replay["initial_state_reference"], "replay initialization"),
            )
            _same(_read_json_reference(base["checkpoints"]["window_2"]["state_reference"], "midpoint"),
                  _read_json_reference(replay["restore_boundary_reference"], "restored midpoint"),
                  "restored complete checkpoint boundary")
            base_index, replay_index = _receipt_index(base), _receipt_index(replay)
            if set(base_index) != {(1, index) for index in range(4)} or set(replay_index) != {(1, 2), (1, 3)}:
                _fail("smoke/replay did not execute exactly four/two declared windows")
            windows = []
            for key in sorted(replay_index):
                windows.append(compare_replay_records(
                    _read_json_reference(base_index[key], "original window"),
                    _read_json_reference(replay_index[key], "replayed window"),
                ))
            base_final = _read_json_reference(base["final_state_reference"], "original final state")
            replay_final = _read_json_reference(replay["final_state_reference"], "replay final state")
            for name in ("clocks", "rng", "raw_bn", "ema_bn"):
                _same(base_final[name], replay_final[name], "same-arm final " + name)
            checkpoint_binding = _read_json_reference(base["binding_reference"], "smoke checkpoint binding")
            _same(checkpoint_binding, _read_json_reference(replay["binding_reference"], "replay checkpoint binding"),
                  "continuation checkpoint run binding")
            continuation = compare_continuation_checkpoints(
                base["checkpoints"]["window_4"]["checkpoint"],
                replay["checkpoints"]["window_4"]["checkpoint"],
                binding=checkpoint_binding,
            )
            preview_comparison = _compare_previews(_preview_evidence(base), _preview_evidence(replay))
            comparisons.append({
                "arm": arm, "status": "PASS", "windows": windows,
                "development_preview": preview_comparison,
                "continuation_checkpoint_numerics": continuation,
                "ema_final_state_exact": base_final["ema_state_sha256"] == replay_final["ema_state_sha256"],
                "optimizer_final_state_exact": base_final["optimizer_state_sha256"] == replay_final["optimizer_state_sha256"],
            })
        _same(a["common_identity"], b["common_identity"], "A/B frozen common identity")
        _same(a["runtime"], b["runtime"], "A/B actual CUDA runtime")
        _verify_common_initial_state(
            _read_json_reference(a["initial_state_reference"], "A initialization"),
            _read_json_reference(b["initial_state_reference"], "B initialization"),
        )
        ai, bi = _receipt_index(a), _receipt_index(b)
        observations = []
        for key in sorted(ai):
            left = _read_json_reference(ai[key], "A smoke receipt")
            right = _read_json_reference(bi[key], "B smoke receipt")
            _same(left["input"], right["input"], "A/B actual logical input")
            for name in ("denominator", "target_total", "logical_batch_size", "optimizer_updates", "epoch"):
                _same(left["window"][name], right["window"][name], "A/B logical " + name)
            observations.append({
                "epoch": key[0], "logical_batch_index": key[1],
                "A_loss": left["window"]["loss"], "B_loss": right["window"]["loss"],
                "A_memory": left["memory"], "B_memory": right["memory"],
                "A_DN_groups": left["window"]["dn_num_groups"],
                "B_DN_groups": right["window"]["dn_num_groups"],
                "A_microsteps": left["window"]["microsteps"], "B_microsteps": right["window"]["microsteps"],
                "BN_DN_cross_arm_equality_required": False,
            })
        return {
            "schema_version": 1, "status": "PASS", "scope": "paired_640_engineering_smoke",
            "pass_scope": "exact_checkpoint_restore_input_RNG_BN_DN_clocks_and_declared_loss_raw_EMA_AdamW_tolerances",
            "continuation_numerics_observed_without_an_acceptance_threshold": [
                "development_predictions", "development_metrics",
            ],
            "workers": [row[1] for row in loaded], "same_arm_replay": comparisons,
            "between_arm_observations": observations,
            "between_arm_preview": _compare_previews(_preview_evidence(a), _preview_evidence(b)),
            "hardware_clearance": "controller_must_verify_each_native_guardian_final_after_worker_exit",
            "scientific_certified": False, "epoch_10_is_only_an_early_checkpoint": True,
        }
    except Exception as exc:
        return {"schema_version": 1, "status": "STOP_NO_RETRY", "failure": str(exc),
                "failure_type": type(exc).__name__, "scientific_certified": False}
