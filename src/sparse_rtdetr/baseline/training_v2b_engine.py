"""Logical-window accumulation for the v2b engineering baseline.

The objective is sum_i max(GT_i, 1) / max(sum_i GT_i, 1) * loss_i,
where loss_i sums the vendor criterion's already-weighted scalar losses.
DN is constructed by the model independently for each physical microbatch.
Live BatchNorm and native microbatch DN deliberately do not promise equivalence
to a single larger physical batch. This module does not load data or launch runs.

CPU BatchNorm backward receives contiguous gradients to avoid the singleton-
batch strided-gradient defect in PyTorch 2.4.1. CUDA keeps its native path.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


BN_BACKWARD_LAYOUT = "cpu_contiguous_cuda_native"


class AccumulationEngineError(RuntimeError):
    """An invalid contract, failed window, or unsafe engine state."""


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AccumulationEngineError(f"{field} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise AccumulationEngineError(f"{field} must be finite and positive")
    return float(value)


def _input_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if type(value) is int:
        dimensions = (value, value)
    elif type(value) is tuple and len(value) == 2:
        dimensions = value
    else:
        raise AccumulationEngineError("expected_input_size must be an integer, (H,W), or None")
    return (
        _integer(dimensions[0], "expected_input_size height", minimum=1),
        _integer(dimensions[1], "expected_input_size width", minimum=1),
    )


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move(child, device) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(child, device) for child in value)
    if isinstance(value, list):
        return [_move(child, device) for child in value]
    return value


def _contiguous_batchnorm_output_gradient(module, inputs, output) -> None:
    """Normalize only CPU backward layout; preserve forward values and dtype."""
    if isinstance(output, torch.Tensor) and output.device.type == "cpu" and output.requires_grad:
        output.register_hook(lambda gradient: gradient.contiguous())


@contextlib.contextmanager
def _batchnorm_backward_layout(model: nn.Module, device: torch.device):
    """Scope hooks to one window and restore existing hooks even after failure.

    A singleton-N gradient from flatten/transpose/cat can be channels-last with
    a degenerate batch stride. PyTorch 2.4.1 CPU BatchNorm backward then violates
    even dL/dbeta = sum(grad_output). Contiguous gradients restore the analytic
    derivative for both live and frozen BN, including BF16. No buffers, affine
    parameters, update clocks, or CUDA behavior are changed.
    """
    handles = []
    try:
        if device.type == "cpu":
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    handles.append(module.register_forward_hook(_contiguous_batchnorm_output_gradient))
        yield
    finally:
        for handle in handles:
            handle.remove()


def validate_engine_state_dict(state: Any) -> dict[str, Any]:
    """Validate a detached engine state without constructing or mutating a model."""
    state_keys = {
        "schema_version", "config", "epoch", "epoch_active",
        "epoch_start_optimizer_updates", "optimizer_updates", "microsteps",
        "phase", "failed",
    }
    config_keys = {
        "physical_batch_size", "accumulation_steps", "amp_dtype",
        "clip_max_norm", "bn_statistics", "expected_input_size", "bn_backward_layout",
    }
    if type(state) is not dict or set(state) != state_keys:
        raise AccumulationEngineError("engine state schema keys mismatch")
    if _integer(state["schema_version"], "schema_version", minimum=1) != 1:
        raise AccumulationEngineError("unsupported engine state schema version")
    config = state["config"]
    if type(config) is not dict or set(config) != config_keys:
        raise AccumulationEngineError("engine state config keys mismatch")
    _integer(config["physical_batch_size"], "physical_batch_size", minimum=1)
    accumulation_steps = _integer(config["accumulation_steps"], "accumulation_steps", minimum=1)
    clip_max_norm = _positive_number(config["clip_max_norm"], "clip_max_norm")
    if type(config["amp_dtype"]) is not str or config["amp_dtype"] not in ("float32", "bfloat16"):
        raise AccumulationEngineError("engine state amp_dtype is invalid")
    if type(config["bn_statistics"]) is not str or config["bn_statistics"] not in ("train", "frozen"):
        raise AccumulationEngineError("engine state bn_statistics is invalid")
    if type(config["bn_backward_layout"]) is not str or config["bn_backward_layout"] != BN_BACKWARD_LAYOUT:
        raise AccumulationEngineError("engine state bn_backward_layout is invalid")
    size = config["expected_input_size"]
    if size is not None:
        if type(size) is not list or len(size) != 2:
            raise AccumulationEngineError("state expected_input_size must be a two-integer JSON list")
        _input_size(tuple(size))

    epoch = _integer(state["epoch"], "epoch")
    updates = _integer(state["optimizer_updates"], "optimizer_updates")
    microsteps = _integer(state["microsteps"], "microsteps")
    start = _integer(state["epoch_start_optimizer_updates"], "epoch_start_optimizer_updates")
    if type(state["epoch_active"]) is not bool:
        raise AccumulationEngineError("epoch_active must be boolean")
    if _integer(state["phase"], "phase") != 0 or state["failed"] is not False:
        raise AccumulationEngineError("only successful window-boundary states can be restored")
    if microsteps != updates * accumulation_steps or start > updates:
        raise AccumulationEngineError("engine update/microstep counters are inconsistent")
    active = state["epoch_active"]
    if epoch == 0:
        if active or updates or microsteps or start:
            raise AccumulationEngineError("epoch zero must be the fresh engine state")
    elif (not active and updates <= start) or updates < epoch - int(active):
        raise AccumulationEngineError("engine epoch counters are inconsistent")
    return {
        **state,
        "config": {
            **config,
            "clip_max_norm": clip_max_norm,
            "expected_input_size": list(size) if size is not None else None,
        },
    }


class AccumulationEngine:
    """Train complete logical windows with one optimizer/EMA/warmup update.

    The caller constructs and places the model and criterion before supplying
    the optimizer. Call begin_epoch before train_window and finish_epoch after
    the last complete window. Incomplete logical windows must be dropped by the
    caller's common logical-batch sampler; this engine never silently drops one.

    A failed window permanently invalidates this engine instance. Model buffers
    updated during its forwards, including BN statistics, are not rolled back.
    Checkpoint restoration should use a fresh engine after such a failure.
    """

    STATE_SCHEMA_VERSION = 1
    _STATE_KEYS = {
        "schema_version", "config", "epoch", "epoch_active",
        "epoch_start_optimizer_updates", "optimizer_updates", "microsteps",
        "phase", "failed",
    }

    def __init__(
        self,
        model: nn.Module,
        criterion: Any,
        optimizer: torch.optim.Optimizer,
        *,
        physical_batch_size: int,
        accumulation_steps: int,
        device: str | torch.device = "cpu",
        amp_dtype: str | torch.dtype = "bfloat16",
        clip_max_norm: float = 0.1,
        ema: Any = None,
        warmup: Any = None,
        scheduler: Any = None,
        bn_statistics: str = "train",
        expected_input_size: int | tuple[int, int] | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise AccumulationEngineError("model must be a real torch.nn.Module")
        if not callable(criterion):
            raise AccumulationEngineError("criterion must be callable")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise AccumulationEngineError("optimizer must be a real torch optimizer")
        self.physical_batch_size = _integer(
            physical_batch_size, "physical_batch_size", minimum=1
        )
        self.accumulation_steps = _integer(
            accumulation_steps, "accumulation_steps", minimum=1
        )
        if amp_dtype in ("float32", torch.float32):
            self.amp_dtype = "float32"
        elif amp_dtype in ("bfloat16", torch.bfloat16):
            self.amp_dtype = "bfloat16"
        else:
            raise AccumulationEngineError("amp_dtype must be float32 or bfloat16")
        if bn_statistics not in ("train", "frozen"):
            raise AccumulationEngineError("bn_statistics must be train or frozen")
        self.bn_statistics = bn_statistics
        self.expected_input_size = _input_size(expected_input_size)
        self.clip_max_norm = _positive_number(clip_max_norm, "clip_max_norm")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise AccumulationEngineError("only cpu and cuda device types are supported")

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.ema = ema
        self.warmup = warmup
        self.scheduler = scheduler
        for component, name, methods in (
            (ema, "ema", ("update",)),
            (warmup, "warmup", ("step", "finished")),
            (scheduler, "scheduler", ("step",)),
        ):
            if component is not None and any(
                not callable(getattr(component, method, None)) for method in methods
            ):
                raise AccumulationEngineError(f"{name} lacks its required methods")

        model_parameters = list(model.parameters())
        if not model_parameters:
            raise AccumulationEngineError("model has no parameters")
        if self.device.index is None and self.device.type == "cuda":
            # Inspect placement without probing or initializing a CUDA device.
            first_device = model_parameters[0].device
            if first_device.type == "cuda":
                self.device = first_device
        if any(parameter.device != self.device for parameter in model_parameters):
            raise AccumulationEngineError("model parameters are not on the requested device")

        self._optimized_parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        identities = [id(parameter) for parameter in self._optimized_parameters]
        model_ids = {id(parameter) for parameter in model_parameters}
        trainable_ids = {
            id(parameter) for parameter in model_parameters if parameter.requires_grad
        }
        optimized_trainable_ids = {
            id(parameter)
            for parameter in self._optimized_parameters
            if parameter.requires_grad
        }
        if (
            len(identities) != len(set(identities))
            or not set(identities).issubset(model_ids)
            or not trainable_ids
            or optimized_trainable_ids != trainable_ids
        ):
            raise AccumulationEngineError(
                "optimizer must cover each trainable model parameter exactly once"
            )

        self.optimizer_updates = 0
        self.microsteps = 0
        self.epoch = 0
        self.epoch_active = False
        self.epoch_start_optimizer_updates = 0
        self.phase = 0
        self.failed = False

    @property
    def logical_batch_size(self) -> int:
        return self.physical_batch_size * self.accumulation_steps

    def _config(self) -> dict[str, Any]:
        return {
            "physical_batch_size": self.physical_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "amp_dtype": self.amp_dtype,
            "clip_max_norm": self.clip_max_norm,
            "bn_statistics": self.bn_statistics,
            "bn_backward_layout": BN_BACKWARD_LAYOUT,
            "expected_input_size": (
                list(self.expected_input_size) if self.expected_input_size is not None else None
            ),
        }

    def _ready(self) -> None:
        if self.failed:
            raise AccumulationEngineError("engine is failed and cannot continue")
        if self.phase != 0:
            raise AccumulationEngineError("engine is inside a logical window")

    def _training_mode(self) -> None:
        self.model.train()
        if isinstance(self.criterion, nn.Module):
            self.criterion.train()
        if self.bn_statistics == "frozen":
            for module in self.model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    if not module.track_running_stats:
                        raise AccumulationEngineError(
                            "cannot freeze statistics for BN without running statistics"
                        )
                    module.eval()
                    # Do not change affine parameters or requires_grad.

    def begin_epoch(self, epoch: int) -> None:
        self._ready()
        _integer(epoch, "epoch", minimum=1)
        if self.epoch_active or epoch != self.epoch + 1:
            raise AccumulationEngineError("epochs must begin consecutively after finish_epoch")
        self._training_mode()
        self.epoch = epoch
        self.epoch_active = True
        self.epoch_start_optimizer_updates = self.optimizer_updates

    def _validate_window(
        self, images: Any, targets: Any
    ) -> tuple[list[Mapping[str, Any]], list[int]]:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise AccumulationEngineError("images must be a BCHW tensor")
        if images.shape[0] != self.logical_batch_size or any(
            dimension <= 0 for dimension in images.shape[1:]
        ):
            raise AccumulationEngineError("incomplete or invalid logical image batch")
        if self.expected_input_size is not None and tuple(images.shape[-2:]) != self.expected_input_size:
            raise AccumulationEngineError("image spatial size does not match expected_input_size")
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise AccumulationEngineError("images must be finite floating point tensors")
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise AccumulationEngineError("targets must be a sequence of mappings")
        if len(targets) != self.logical_batch_size:
            raise AccumulationEngineError("target count does not match the logical batch")

        decoder = getattr(self.model, "decoder", None)
        num_classes = (
            _integer(decoder.num_classes, "model.decoder.num_classes", minimum=1)
            if decoder is not None and hasattr(decoder, "num_classes") else None
        )
        checked: list[Mapping[str, Any]] = []
        counts: list[int] = []
        for target in targets:
            if not isinstance(target, Mapping):
                raise AccumulationEngineError("each target must be a mapping")
            labels = target.get("labels")
            boxes = target.get("boxes")
            if (
                not isinstance(labels, torch.Tensor)
                or labels.ndim != 1
                or labels.dtype not in (torch.int32, torch.int64)
            ):
                raise AccumulationEngineError("labels must be a one-dimensional integer tensor")
            if (
                not isinstance(boxes, torch.Tensor)
                or boxes.ndim != 2
                or boxes.shape != (len(labels), 4)
                or not boxes.is_floating_point()
                or not bool(torch.isfinite(boxes).all())
            ):
                raise AccumulationEngineError("boxes must be finite floating point [GT,4]")
            if bool(((boxes < 0) | (boxes > 1)).any()) or bool((boxes[:, 2:] <= 0).any()):
                raise AccumulationEngineError("boxes must be normalized cxcywh with positive width/height")
            if bool((labels < 0).any()) or (
                num_classes is not None and bool((labels >= num_classes).any())
            ):
                raise AccumulationEngineError("labels contain out-of-range class indices")
            checked.append(target)
            counts.append(len(labels))
        return checked, counts

    def _loss_sum(self, losses: Any) -> tuple[torch.Tensor, dict[str, float]]:
        if not isinstance(losses, Mapping) or not losses:
            raise AccumulationEngineError("criterion must return a nonempty loss dictionary")
        values: list[torch.Tensor] = []
        observations: dict[str, float] = {}
        for key, loss in losses.items():
            if not isinstance(key, str) or not key:
                raise AccumulationEngineError("loss names must be nonempty strings")
            if (
                not isinstance(loss, torch.Tensor)
                or loss.ndim != 0
                or not loss.is_floating_point()
                or loss.device != self.device
            ):
                raise AccumulationEngineError("every loss must be a floating scalar on the model device")
            value = float(loss.detach().item())
            if not math.isfinite(value):
                raise AccumulationEngineError(f"nonfinite loss: {key}")
            observations[key] = value
            values.append(loss.float() if loss.dtype in (torch.float16, torch.bfloat16) else loss)
        total = values[0]
        for loss in values[1:]:
            total = total + loss
        if not total.requires_grad or not bool(torch.isfinite(total.detach())):
            raise AccumulationEngineError("summed loss must be finite and differentiable")
        return total, observations

    def _finite_gradients(self) -> None:
        gradients = [
            parameter.grad
            for parameter in self._optimized_parameters
            if parameter.grad is not None
        ]
        if not gradients:
            raise AccumulationEngineError("backward produced no model gradients")
        values = [
            gradient.coalesce().values() if gradient.is_sparse else gradient
            for gradient in gradients
        ]
        if not bool(torch.stack([torch.isfinite(value).all() for value in values]).all()):
            raise AccumulationEngineError("nonfinite model gradients")

    def train_window(self, images: torch.Tensor, targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        self._ready()
        if not self.epoch_active:
            raise AccumulationEngineError("begin_epoch is required before train_window")
        try:
            checked_targets, counts = self._validate_window(images, targets)
            self._training_mode()
            target_total = sum(counts)
            denominator = max(target_total, 1)
            micro_target_counts = [
                sum(counts[start:start + self.physical_batch_size])
                for start in range(0, self.logical_batch_size, self.physical_batch_size)
            ]
            coefficients = [max(count, 1) / denominator for count in micro_target_counts]
            micro_losses: list[dict[str, float]] = []
            dn_groups: list[int | None] = []
            dn_query_counts: list[int | None] = []
            objective_value = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            with _batchnorm_backward_layout(self.model, self.device):
                for index, start in enumerate(
                    range(0, self.logical_batch_size, self.physical_batch_size)
                ):
                    stop = start + self.physical_batch_size
                    micro_images = images[start:stop].to(device=self.device)
                    micro_targets = _move(checked_targets[start:stop], self.device)
                    context = (
                        torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                        if self.amp_dtype == "bfloat16" else contextlib.nullcontext()
                    )
                    with context:
                        outputs = self.model(micro_images, targets=micro_targets)
                        losses = self.criterion(outputs, micro_targets)
                    # Summation, coefficient application, and backward are outside AMP.
                    total, observed = self._loss_sum(losses)
                    weighted = total * coefficients[index]
                    if not bool(torch.isfinite(weighted.detach())):
                        raise AccumulationEngineError("nonfinite weighted microbatch loss")
                    objective_value += float(weighted.detach().item())
                    if not math.isfinite(objective_value):
                        raise AccumulationEngineError("nonfinite logical window loss")
                    weighted.backward()
                    self.phase += 1
                    self.microsteps += 1
                    self._finite_gradients()
                    micro_losses.append(observed)
                    metadata = outputs.get("dn_meta") if isinstance(outputs, Mapping) else None
                    group = metadata.get("dn_num_group") if isinstance(metadata, Mapping) else None
                    splits = metadata.get("dn_num_split") if isinstance(metadata, Mapping) else None
                    dn_groups.append(group if type(group) is int else None)
                    dn_query_counts.append(
                        splits[0]
                        if isinstance(splits, (list, tuple)) and splits and type(splits[0]) is int
                        else None
                    )

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self._optimized_parameters,
                self.clip_max_norm,
                error_if_nonfinite=True,
            )
            self._finite_gradients()
            result = self.optimizer.step()
            if result is False or (isinstance(result, Mapping) and result.get("skipped") is True):
                raise AccumulationEngineError("optimizer skipped its logical update")
            self.optimizer_updates += 1
            if not bool(torch.stack([
                torch.isfinite(parameter.detach()).all()
                for parameter in self._optimized_parameters
            ]).all()):
                raise AccumulationEngineError("nonfinite model parameters after optimizer step")
            if self.ema is not None:
                self.ema.update(self.model)
            if self.warmup is not None:
                self.warmup.step()
            self.phase = 0
            return {
                "epoch": self.epoch,
                "optimizer_updates": self.optimizer_updates,
                "microsteps": self.microsteps,
                "logical_batch_size": self.logical_batch_size,
                "target_total": target_total,
                "denominator": denominator,
                "microbatch_target_counts": micro_target_counts,
                "coefficients": coefficients,
                "microbatch_losses": micro_losses,
                "dn_num_groups": dn_groups,
                "dn_query_counts": dn_query_counts,
                "loss": objective_value,
                "gradient_norm_before_clip": float(gradient_norm.detach().item()),
                "phase": self.phase,
            }
        except Exception as exc:
            self.failed = True
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            if isinstance(exc, AccumulationEngineError):
                raise
            raise AccumulationEngineError(f"logical window failed at phase {self.phase}") from exc

    def finish_epoch(self, *, expected_optimizer_steps: int | None = None) -> dict[str, Any]:
        self._ready()
        if not self.epoch_active:
            raise AccumulationEngineError("no active epoch to finish")
        steps = self.optimizer_updates - self.epoch_start_optimizer_updates
        if expected_optimizer_steps is not None:
            _integer(expected_optimizer_steps, "expected_optimizer_steps", minimum=1)
        if steps <= 0 or (
            expected_optimizer_steps is not None and steps != expected_optimizer_steps
        ):
            self.failed = True
            raise AccumulationEngineError("epoch logical optimizer-step count mismatch")
        try:
            warmup_finished = self.warmup is None or bool(self.warmup.finished())
            scheduler_stepped = self.scheduler is not None and warmup_finished
            if scheduler_stepped:
                self.scheduler.step()
        except Exception as exc:
            self.failed = True
            raise AccumulationEngineError("epoch scheduler failed") from exc
        self.epoch_active = False
        return {
            "epoch": self.epoch,
            "epoch_optimizer_steps": steps,
            "optimizer_updates": self.optimizer_updates,
            "microsteps": self.microsteps,
            "scheduler_stepped": scheduler_stepped,
        }

    def validate_state_dict(self, state: Any) -> dict[str, Any]:
        """Check a candidate state and this engine's configuration without mutation."""
        validated = validate_engine_state_dict(state)
        if validated["config"] != self._config():
            raise AccumulationEngineError("engine state configuration mismatch")
        return validated

    def state_dict(self) -> dict[str, Any]:
        self._ready()
        return self.validate_state_dict({
            "schema_version": self.STATE_SCHEMA_VERSION,
            "config": self._config(),
            "epoch": self.epoch,
            "epoch_active": self.epoch_active,
            "epoch_start_optimizer_updates": self.epoch_start_optimizer_updates,
            "optimizer_updates": self.optimizer_updates,
            "microsteps": self.microsteps,
            "phase": self.phase,
            "failed": self.failed,
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._ready()
        validated = self.validate_state_dict(state)
        # Validate everything before mutation. Other components own their state.
        for field in (
            "epoch", "epoch_active", "epoch_start_optimizer_updates",
            "optimizer_updates", "microsteps", "phase", "failed",
        ):
            setattr(self, field, validated[field])
