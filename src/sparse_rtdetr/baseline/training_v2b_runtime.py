"""Explicit train_core/model/checkpoint wiring; no loop, evaluator or launcher.

Real-data engineering preflight only previews the loader. Calling train_batch
is an execution step for a separately authorized harness; CUDA additionally
requires a fresh native admission on every logical window. A failed update
requires a fresh session and an authenticated checkpoint, never silent retry.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from .training_v2b import V2BComponents, validate_model_geometry, validate_model_sampling
from .training_v2b_checkpoint import restore_checkpoint, save_checkpoint
from .training_v2b_data import TrainCoreLogicalLoader, LogicalBatch
from .training_v2b_device import (
    validate_bound_runtime_policy, validate_component_placement, validate_prepared_runtime,
)
from .training_v2b_evidence import (
    build_run_binding, initial_parameter_reference, validate_run_binding,
)
from .training_v2b_hardware import require_native_hardware_admission


class V2BSessionError(RuntimeError):
    """Live configuration, data cursor, model state or native admission differs."""


def _config(components: V2BComponents, loader: TrainCoreLogicalLoader, *,
            device: str, cuda_gpu_uuid: str | None) -> dict[str, Any]:
    if not isinstance(components, V2BComponents) or not isinstance(loader, TrainCoreLogicalLoader):
        raise V2BSessionError("real v2b components and a train_core loader are required")
    config = components.config.binding_config(
        device=device, scope="train_core_runtime_engineering", cuda_gpu_uuid=cuda_gpu_uuid,
    )
    for name in ("seed", "input_size", "logical_batch_size", "augmentation_stop_internal_epoch"):
        if getattr(loader.config, name) != config[name]:
            raise V2BSessionError("model/data configuration mismatch: " + name)
    # The factory's unconstructed vendor loader recipes are not the live
    # loader. Bind the actual data recipe separately; worker topology is a
    # runtime observation so a deterministic 0->2 worker restore is legal.
    resolved = copy.deepcopy(components.resolved_config)
    for key in ("train_dataloader", "val_dataloader", "evaluator"):
        resolved.pop(key, None)
    resolved["device"] = device
    return {**config, "resolved_model_optimizer_config": resolved,
             "geometry": copy.deepcopy(components.geometry),
             "sampling": copy.deepcopy(components.initialization["sampling"]),
            "pretrained": copy.deepcopy(components.initialization["pretrained"]),
            "loader_runtime_settings": "observed_separately_not_resume_semantics"}


def _check_construction_sources(components: V2BComponents, binding: Mapping[str, Any]) -> None:
    for family in ("package_sources", "vendor_sources"):
        for row in components.initialization[family]:
            if binding["code"].get(row["relative_path"], {}).get("sha256") != row["sha256"]:
                raise V2BSessionError("construction source differs from run binding: " + row["relative_path"])


def build_train_core_run_binding(components: V2BComponents, loader: TrainCoreLogicalLoader, *,
                                run_id: str, repo_root: str | Path,
                                additional_code_paths: Mapping[str, str | Path] | None = None,
                                requested_device: str = "cpu",
                                cuda_gpu_uuid: str | None = None) -> dict[str, Any]:
    """Bind initialized CPU identity before a future CUDA admission is requested.

    A CPU-built model may describe a requested CUDA run without initializing
    CUDA. After admission, construct with the prepared runtime and verify the
    same initial identity before creating a session. This breaks no hash cycle.
    """
    root = Path(repo_root).resolve()
    paths = {
        row["relative_path"]: root / row["relative_path"]
        for family in ("vendor_sources", "package_sources")
        for row in components.initialization[family]
    }
    for path in (root / "src/sparse_rtdetr/baseline").glob("training_v2b*.py"):
        paths[str(path.relative_to(root))] = path
    for path in (root / "vendor/rtdetrv2_pytorch/configs").rglob("*.yml"):
        paths[str(path.relative_to(root))] = path
    for row in loader.binding["source"]["sources"]:
        paths[row["path"]] = root / row["path"]
    for name, path in (additional_code_paths or {}).items():
        actual = Path(path).resolve()
        if not actual.is_relative_to(root) or str(actual.relative_to(root)) != name:
            raise V2BSessionError("additional source must use its exact checkout-relative name")
        paths[name] = actual
    binding = build_run_binding(
        run_id=run_id, code_paths=paths,
        config=_config(components, loader, device=requested_device, cuda_gpu_uuid=cuda_gpu_uuid),
        initial_parameters=components.initialization["parameters"],
        initial_state=components.initialization["model_state"],
        train_core_input=loader.binding,
    )
    _check_construction_sources(components, binding)
    return binding


class V2BTrainingSession:
    """One live engine, one acknowledged data cursor, and one immutable binding."""

    def __init__(self, components: V2BComponents, loader: TrainCoreLogicalLoader,
                 binding: Mapping[str, Any]):
        self.components, self.loader = components, loader
        self.binding = validate_run_binding(binding)
        self.failed = False
        self._preflight(verify_files=True)
        if components.engine.optimizer_updates == 0:
            actual = initial_parameter_reference({
                name: tensor.detach().cpu() for name, tensor in components.model.state_dict().items()
            })
            if actual != components.initialization["model_state"]:
                raise V2BSessionError("live initial model differs from the recorded initialization")
        if validate_model_geometry(components.model, components.config) != components.geometry:
            raise V2BSessionError("live model geometry differs from initialization")

    def _preflight(self, *, verify_files: bool) -> None:
        if self.failed:
            raise V2BSessionError("session failed; rebuild and restore a verified checkpoint")
        c = self.components
        checked = validate_run_binding(self.binding, verify_files=verify_files)
        identity = validate_prepared_runtime(c.runtime)
        validate_bound_runtime_policy(identity, checked["config"])
        uuid = identity["cuda_devices"][0]["uuid"] if identity["device"] == "cuda:0" else None
        expected = _config(c, self.loader, device=identity["device"], cuda_gpu_uuid=uuid)
        if checked["config"] != expected:
            raise V2BSessionError("live model/runtime configuration differs from binding")
        # Native diagnostic observers temporarily wrap the original core only
        # during train_batch. Candidate workers do not install that observer,
        # so their actual raw/EMA callables are checked before every forward.
        # Both backends are checked at construction/save/restore/epoch bounds.
        if c.config.sampling_backend == "deterministic_gather" or verify_files:
            for name, model in (("raw", c.model), ("EMA", c.ema.module)):
                actual_sampling = validate_model_sampling(model, sampling_backend=c.config.sampling_backend)
                if actual_sampling != c.initialization["sampling"]:
                    raise V2BSessionError(name + " sampling identity differs from initialization")
        if (checked["data"].get("kind") != "train_core_runtime"
                or checked["data"].get("loader") != self.loader.binding):
            raise V2BSessionError("live loader differs from bound train_core input")
        initial = checked["initial_weights"]
        body = {key: value for key, value in initial.items()
                if key not in ("parameter_reference", "identity_scope")}
        if (initial.get("parameter_reference") != c.initialization["parameters"]
                or body != c.initialization["model_state"]):
            raise V2BSessionError("initial model identity differs from run binding")
        _check_construction_sources(c, checked)
        if identity["device"] == "cuda:0" and c.runtime.admitted_binding_sha256 != checked["binding_sha256"]:
            raise V2BSessionError("CUDA runtime was admitted for a different run binding")
        self.loader.validate_engine_state(self.loader.state_dict(), c.engine.state_dict())
        validate_component_placement(model=c.model, optimizer=c.optimizer, criterion=c.criterion,
                                     postprocessor=c.postprocessor, ema=c.ema, runtime=c.runtime)

    def begin_epoch(self, epoch: int) -> None:
        self._preflight(verify_files=True)
        state = self.loader.state_dict()
        if (self.loader.has_active_iterator or state["epoch_active"]
                or type(epoch) is not int or epoch != state["epoch"] + 1):
            raise V2BSessionError("close and complete the prior epoch before beginning the next")
        try:
            self.components.engine.begin_epoch(epoch)
            self.loader.begin_epoch(epoch, self.components.engine.state_dict())
        except Exception:
            self.failed = True
            raise

    def train_batch(self, batch: LogicalBatch, *, hardware_probe: Any = None) -> dict[str, Any]:
        self._preflight(verify_files=False)
        # Pure validation comes before any model/BN change. A rejected stale
        # or mutated input does not poison an otherwise untouched session.
        self.loader.validate_pending_batch(batch)
        c = self.components
        if c.runtime.device.type == "cuda":
            try:
                require_native_hardware_admission(
                    hardware_probe, binding=self.binding,
                    expected_gpu_uuid=self.binding["config"]["cuda_gpu_uuid"],
                )
            except Exception:
                self.failed = True
                raise
        elif hardware_probe is not None:
            raise V2BSessionError("a CPU session does not consume a GPU admission")
        receipt = batch.evidence()
        try:
            record = c.engine.train_window(batch.images, batch.targets)
            self.loader.commit_batch(batch, c.engine.state_dict())
        except Exception:
            self.failed = True
            raise
        return {"input": receipt, "window": record}

    def finish_epoch(self) -> dict[str, Any]:
        self._preflight(verify_files=True)
        state = self.loader.state_dict()
        if (self.loader.has_active_iterator or not state["epoch_active"]
                or state["completed_batches"] != self.loader.batches_per_epoch):
            raise V2BSessionError("close the iterator and commit every full logical batch before finishing")
        try:
            record = self.components.engine.finish_epoch(
                expected_optimizer_steps=self.loader.batches_per_epoch,
            )
            self.loader.finish_epoch(self.components.engine.state_dict())
        except Exception:
            self.failed = True
            raise
        return {"epoch": record, "loader": self.loader.state_dict()}

    def save(self, path: str | Path) -> dict[str, Any]:
        self._preflight(verify_files=True)
        c = self.components
        return save_checkpoint(
            path, model=c.model, optimizer=c.optimizer, ema=c.ema, engine=c.engine,
            scheduler=c.scheduler, warmup=c.warmup, binding=self.binding,
            runtime=c.runtime, sampler=self.loader,
        )

    def restore(self, path: str | Path, expected_sha256: str) -> dict[str, Any]:
        self._preflight(verify_files=True)
        c = self.components
        if c.engine.epoch != 0 or self.loader.has_active_iterator:
            raise V2BSessionError("restore requires a fresh session with no active loader iterator")
        try:
            result = restore_checkpoint(
                path, model=c.model, optimizer=c.optimizer, ema=c.ema, engine=c.engine,
                scheduler=c.scheduler, warmup=c.warmup, expected_binding=self.binding,
                expected_sha256=expected_sha256, runtime=c.runtime, sampler=self.loader,
            )
        except Exception:
            self.failed = self.failed or c.engine.failed
            raise
        try:
            self._preflight(verify_files=True)
        except Exception:
            self.failed = True
            raise
        return result
