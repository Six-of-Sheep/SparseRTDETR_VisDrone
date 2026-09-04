"""CPU tests for the T5D inner entry boundary."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_entry as entry


ROOT = Path(__file__).resolve().parents[1]
_COUNTER = 0


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(role: str) -> dict[str, str]:
    protocol = entry._adapter.COMPONENT_PROTOCOL
    implementation = f"t5d-test-{role}-v1"
    return {"role": role, "protocol": protocol, "implementation": implementation, "source_sha256": _sha(_canonical({"role": role, "protocol": protocol, "implementation": implementation}))}


class _Model:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.identity = _identity("model")

    def parameter_descriptors(self) -> list[dict[str, object]]:
        self.events.append("model.parameter_descriptors")
        return [
            {"name": "backbone.conv.weight", "trainable": True, "numel": 10, "dtype": "float32"},
            {"name": "backbone.norm.weight", "trainable": True, "numel": 20, "dtype": "float32"},
            {"name": "decoder.weight", "trainable": True, "numel": 30, "dtype": "float32"},
            {"name": "frozen.bias", "trainable": False, "numel": 4, "dtype": "float32"},
        ]

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        self.events.append("model.forward")
        return {"batch_id": batch["batch_id"], "output_sha256": _sha(b"t5d-forward"), "finite": True}

    def backward(self, scaled_loss: float) -> float:
        self.events.append("model.backward")
        return 0.75

    def state_bytes(self) -> bytes:
        self.events.append("model.state_bytes")
        return b"t5d-raw-model"


class _Criterion:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.identity = _identity("criterion")

    def compute(self, forward_output: dict[str, object], batch: dict[str, object]) -> float:
        self.events.append("criterion.compute")
        return 0.25


class _Optimizer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.identity = _identity("optimizer")

    def zero_grad(self) -> None:
        self.events.append("optimizer.zero_grad")

    def step(self) -> None:
        self.events.append("optimizer.step")

    def parameter_groups(self) -> list[dict[str, object]]:
        self.events.append("optimizer.parameter_groups")
        return [
            {"name": "backbone_non_norm", "parameter_names": ["backbone.conv.weight"], "parameter_scope": "backbone", "include_substrings": [], "exclude_substrings": ["norm", "bn"], "learning_rate": 0.00001, "weight_decay": 0.0001, "numel": 10},
            {"name": "norm_or_bn", "parameter_names": ["backbone.norm.weight"], "parameter_scope": "remaining_trainable", "include_substrings": ["norm", "bn"], "exclude_substrings": [], "learning_rate": 0.0001, "weight_decay": 0.0, "numel": 20},
            {"name": "default", "parameter_names": ["decoder.weight"], "parameter_scope": "remaining_trainable", "include_substrings": [], "exclude_substrings": [], "learning_rate": 0.0001, "weight_decay": 0.0001, "numel": 30},
        ]

    def state_bytes(self) -> bytes:
        self.events.append("optimizer.state_bytes")
        return b"t5d-optimizer"


class _Scheduler:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.identity = _identity("lr_scheduler")

    def observe(self) -> dict[str, object]:
        self.events.append("lr_scheduler.observe")
        return {"epoch": 1, "step_unit": "epoch", "decay_events": 0}

    def state_bytes(self) -> bytes:
        self.events.append("lr_scheduler.state_bytes")
        return b"t5d-scheduler"


class _Scaler:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.identity = _identity("amp_scaler")

    def scale(self, loss: float) -> float:
        self.events.append("amp_scaler.scale")
        return loss * 65536.0

    def step(self) -> dict[str, object]:
        self.events.append("amp_scaler.step")
        return {"skipped": False, "overflow": False}

    def update(self) -> dict[str, object]:
        self.events.append("amp_scaler.update")
        return {"scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000, "growth_tracker": 1}

    def state_bytes(self) -> bytes:
        self.events.append("amp_scaler.state_bytes")
        return b"t5d-scaler"


class _EMA:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.identity = _identity("ema")

    def update(self, model: object) -> dict[str, object]:
        self.events.append("ema.update")
        return {"updates": 1, "decay": 0.9999, "warmups": 2000}

    def state_bytes(self) -> bytes:
        self.events.append("ema.state_bytes")
        return b"t5d-ema"


class _Evaluator:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.identity = _identity("primary_evaluator")

    def evaluate(self, development_batch: dict[str, object], weights: str, context: dict[str, object]) -> dict[str, object]:
        self.events.append("primary_evaluator.evaluate")
        return {
            "evaluator_id": "visdrone_official_primary_evaluator_v1",
            "protocol_id": "visdrone_official_style_v1",
            "role": "development",
            "weights": weights,
            "run_id": context["run_id"],
            "nonce": context["nonce"],
            "batch_id": development_batch["batch_id"],
            "epoch": context["epoch"],
            "global_optimizer_step": context["global_optimizer_step"],
            "ema_updates": context["ema_updates"],
            "training_contract_sha256": context["training_contract_sha256"],
            "runtime_plan_sha256": context["runtime_plan_sha256"],
            "ema_identity_sha256": context["ema_identity_sha256"],
            "state_predecessor_sha256": context["state_predecessor_sha256"],
            "AP": 0.5,
            "AP50": 0.7,
            "AR500": 0.6,
        }


class _BatchSource:
    def __init__(self, role: str, events: list[str]) -> None:
        self.role = role
        self.events = events
        self.identity = _identity(role)

    def next_batch(self) -> dict[str, object]:
        self.events.append(f"{self.role}.next_batch")
        size = 16 if self.role == "train_batch_source" else 32
        logical_role = "train_core" if self.role == "train_batch_source" else "development"
        batch_id = "t5d-train-batch" if size == 16 else "t5d-development-batch"
        return {"role": logical_role, "batch_id": batch_id, "batch_size": size, "payload_sha256": _sha(f"{logical_role}:{batch_id}".encode("ascii"))}


class _Serializer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.identity = _identity("checkpoint_serializer")

    def serialize(self, states: dict[str, bytes]) -> dict[str, bytes]:
        self.events.append("checkpoint_serializer.serialize")
        return copy.deepcopy(states)


def _components() -> tuple[dict[str, object], list[str]]:
    events: list[str] = []
    return {
        "model": _Model(events),
        "criterion": _Criterion(events),
        "optimizer": _Optimizer(events),
        "lr_scheduler": _Scheduler(events),
        "amp_scaler": _Scaler(events),
        "ema": _EMA(events),
        "primary_evaluator": _Evaluator(events),
        "train_batch_source": _BatchSource("train_batch_source", events),
        "development_batch_source": _BatchSource("development_batch_source", events),
        "checkpoint_serializer": _Serializer(events),
    }, events


def _descriptor(evidence_root: Path | None = None) -> dict[str, object]:
    global _COUNTER
    _COUNTER += 1
    if evidence_root is None:
        evidence_root = ROOT.parent / f".t5d-entry-test-{_COUNTER}"
    return entry.build_synthetic_entry_descriptor(ROOT, run_id=f"t5d-entry-{_COUNTER}", nonce=f"{_COUNTER:032x}", evidence_root=evidence_root, python_executable=Path(sys.executable).resolve())


def test_entry_import_surface_and_parent_torch_sentinel() -> None:
    assert set(entry.__all__) == {"TrainingEntryError", "build_synthetic_entry_descriptor", "validate_entry_descriptor", "run_synthetic_entry", "validate_entry_result"}
    source = (ROOT / entry.ENTRY_MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "import torch" not in source
    tree = ast.parse(source)
    imported = {alias.name.split(".", 1)[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update((node.module or "").split(".", 1)[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module != "__future__")
    assert not imported.intersection({"torch", "subprocess", "multiprocessing", "socket", "requests", "urllib", "vendor"})
    sentinel = object()
    had_torch = "torch" in sys.modules
    previous = sys.modules.get("torch")
    sys.modules["torch"] = sentinel
    try:
        assert sys.modules["torch"] is sentinel
        assert entry.ENTRY_MODE == "synthetic"
    finally:
        if had_torch:
            sys.modules["torch"] = previous
        else:
            sys.modules.pop("torch", None)


def test_entry_descriptor_binds_authorities_and_source_closure() -> None:
    descriptor = _descriptor()
    assert descriptor["mode"] == "synthetic"
    assert descriptor["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert descriptor["argv"][2] == "sparse_rtdetr.baseline.training_entry"
    paths = {row["relative_path"] for row in descriptor["source_bindings"]["repository_modules"]}
    assert paths == {path for _, path in entry.SOURCE_MODULES}
    assert descriptor["future_targets"] == entry.FUTURE_TARGETS
    assert descriptor["authorities"]["prepared"]["mode"] == "synthetic"
    assert descriptor["authorities"]["prepared"]["execution_policy"]["production_training_authorized"] is False


def test_entry_process_config_mode_is_bound_and_revalidated() -> None:
    descriptor = _descriptor()
    mode = (ROOT / entry.PROCESS_CONFIG_RELATIVE_PATH).stat().st_mode & 0o777
    assert descriptor["source_bindings"]["process_config"]["mode"] == mode
    mutated = copy.deepcopy(descriptor)
    mutated_mode = 0o664 if mode == 0o644 else 0o644
    mutated["source_bindings"]["process_config"]["mode"] = mutated_mode
    body = {key: value for key, value in mutated.items() if key != "aggregate_sha256"}
    mutated["aggregate_sha256"] = entry._digest(body)
    with pytest.raises(entry.TrainingEntryError, match="process config identity"):
        entry.validate_entry_descriptor(mutated)


@pytest.mark.parametrize("mutation", [
    lambda value: {**value, "mode": "real"},
    lambda value: {**value, "argv": [*value["argv"], "--injected"]},
    lambda value: {**value, "environment": {**value["environment"], "EXTRA": "1"}},
    lambda value: {**value, "nonce": "0" * 32},
    lambda value: {**value, "unexpected": False},
])
def test_entry_descriptor_mutations_fail_closed(mutation) -> None:
    descriptor = _descriptor()
    mutated = mutation(copy.deepcopy(descriptor))
    if mutated.get("nonce") == descriptor["nonce"]:
        mutated["nonce"] = "f" * 32
        mutated["aggregate_sha256"] = entry._digest({key: value for key, value in mutated.items() if key != "aggregate_sha256"})
    with pytest.raises(entry.TrainingEntryError):
        entry.validate_entry_descriptor(mutated)


def test_entry_synthetic_transaction_is_detached_and_exactly_once(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path / "entry-evidence")
    returned = entry.validate_entry_descriptor(descriptor)
    returned["future_targets"]["training_evidence_root"] = "changed"
    assert descriptor["future_targets"]["training_evidence_root"] == entry.TRAINING_EVIDENCE_ROOT_RELATIVE_PATH
    components, events = _components()
    result = entry.run_synthetic_entry(descriptor, components, tmp_path / "entry-evidence")
    assert result["mode"] == "synthetic"
    assert result["result"]["evidence"]["classification"] == "SYNTHETIC_TERMINAL_COMPLETE"
    assert events.count("model.forward") == 1
    assert events.count("primary_evaluator.evaluate") == 1
    with pytest.raises(entry.TrainingEntryError):
        entry.run_synthetic_entry(descriptor, _components()[0], tmp_path / "entry-evidence")


def test_entry_rejects_production_target_without_creating_it(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path / "entry-evidence")
    components, _ = _components()
    production = ROOT / entry.TRAINING_EVIDENCE_ROOT_RELATIVE_PATH
    with pytest.raises(entry.TrainingEntryError):
        entry.run_synthetic_entry(descriptor, components, production)
    assert not production.exists()
    assert not (tmp_path / entry.PROCESS_EVIDENCE_ROOT_RELATIVE_PATH).exists()


def test_entry_receipt_blocks_module_reload_and_root_replay(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path / "entry-replay")
    components, _ = _components()
    entry.run_synthetic_entry(descriptor, components, descriptor["evidence_root"])
    importlib.reload(entry)
    with pytest.raises(entry.TrainingEntryError):
        entry.run_synthetic_entry(descriptor, _components()[0], descriptor["evidence_root"])
    fresh_components, events = _components()
    with pytest.raises(entry.TrainingEntryError):
        entry.run_synthetic_entry(descriptor, fresh_components, tmp_path / "different-entry-root")
    assert events == []


def test_entry_public_file_writer_handles_partial_writes(monkeypatch, tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path / "entry-partial-write")
    components, _ = _components()
    original_write = entry.os.write
    calls = 0

    def partial_write(fd, payload):
        nonlocal calls
        calls += 1
        return original_write(fd, payload[:1])

    monkeypatch.setattr(entry.os, "write", partial_write)
    result = entry.run_synthetic_entry(descriptor, components, descriptor["evidence_root"])
    assert result["mode"] == "synthetic"
    assert calls > 1
