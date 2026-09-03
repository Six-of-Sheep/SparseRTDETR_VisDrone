from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import subprocess
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_adapter as adapter
from sparse_rtdetr.baseline import training_evidence


ROOT = Path(__file__).resolve().parents[1]
_PREPARED_COUNTER = 0


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _component_identity(role: str) -> dict[str, str]:
    protocol = adapter.COMPONENT_PROTOCOL
    implementation = f"t5c-test-{role}-v1"
    return {
        "role": role,
        "protocol": protocol,
        "implementation": implementation,
        "source_sha256": _sha(_canonical({"role": role, "protocol": protocol, "implementation": implementation})),
    }


def _batch(role: str, batch_id: str, size: int) -> dict[str, object]:
    return {
        "role": role,
        "batch_id": batch_id,
        "batch_size": size,
        "payload_sha256": _sha(f"{role}:{batch_id}".encode("ascii")),
    }


class _Model:
    def __init__(self, events: list[str], mutations: dict[str, object]) -> None:
        self.events = events
        self.mutations = mutations
        self.identity = _component_identity("model")

    def _value(self, name: str, value: object) -> object:
        mutation = self.mutations.get(f"model.{name}")
        return mutation(value) if callable(mutation) else value

    def parameter_descriptors(self) -> list[dict[str, object]]:
        self.events.append("model.parameter_descriptors")
        return self._value(
            "parameter_descriptors",
            [
                {"name": "backbone.conv.weight", "trainable": True, "numel": 10, "dtype": "float32"},
                {"name": "backbone.norm.weight", "trainable": True, "numel": 20, "dtype": "float32"},
                {"name": "decoder.weight", "trainable": True, "numel": 30, "dtype": "float32"},
                {"name": "frozen.bias", "trainable": False, "numel": 4, "dtype": "float32"},
            ],
        )

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        self.events.append("model.forward")
        value = {
            "batch_id": batch["batch_id"],
            "output_sha256": _sha(b"synthetic-forward-output"),
            "finite": True,
        }
        return self._value("forward", value)

    def backward(self, scaled_loss: float) -> float:
        self.events.append("model.backward")
        return self._value("backward", 0.75)

    def state_bytes(self) -> bytes:
        self.events.append("model.state_bytes")
        return self._value("state_bytes", b"raw-model-state-v1")


class _Criterion:
    def __init__(self, events: list[str], mutations: dict[str, object]) -> None:
        self.events = events
        self.mutations = mutations
        self.identity = _component_identity("criterion")

    def compute(self, forward_output: dict[str, object], batch: dict[str, object]) -> float:
        self.events.append("criterion.compute")
        mutation = self.mutations.get("criterion.compute")
        return mutation(0.25) if callable(mutation) else 0.25


class _Optimizer:
    def __init__(self, events: list[str], mutations: dict[str, object]) -> None:
        self.events = events
        self.mutations = mutations
        self.identity = _component_identity("optimizer")

    def zero_grad(self) -> None:
        self.events.append("optimizer.zero_grad")

    def step(self) -> None:
        self.events.append("optimizer.step")
        mutation = self.mutations.get("optimizer.step")
        if callable(mutation):
            mutation(None)

    def parameter_groups(self) -> list[dict[str, object]]:
        self.events.append("optimizer.parameter_groups")
        value = [
            {
                "name": "backbone_non_norm",
                "parameter_names": ["backbone.conv.weight"],
                "parameter_scope": "backbone",
                "include_substrings": [],
                "exclude_substrings": ["norm", "bn"],
                "learning_rate": 0.00001,
                "weight_decay": 0.0001,
                "numel": 10,
            },
            {
                "name": "norm_or_bn",
                "parameter_names": ["backbone.norm.weight"],
                "parameter_scope": "remaining_trainable",
                "include_substrings": ["norm", "bn"],
                "exclude_substrings": [],
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
                "numel": 20,
            },
            {
                "name": "default",
                "parameter_names": ["decoder.weight"],
                "parameter_scope": "remaining_trainable",
                "include_substrings": [],
                "exclude_substrings": [],
                "learning_rate": 0.0001,
                "weight_decay": 0.0001,
                "numel": 30,
            },
        ]
        mutation = self.mutations.get("optimizer.parameter_groups")
        return mutation(value) if callable(mutation) else value

    def state_bytes(self) -> bytes:
        self.events.append("optimizer.state_bytes")
        return b"optimizer-state-v1"


class _Scheduler:
    def __init__(self, events: list[str], mutations: dict[str, object]) -> None:
        self.events = events
        self.mutations = mutations
        self.identity = _component_identity("lr_scheduler")

    def observe(self) -> dict[str, object]:
        self.events.append("lr_scheduler.observe")
        value = {"epoch": 1, "step_unit": "epoch", "decay_events": 0}
        mutation = self.mutations.get("lr_scheduler.observe")
        return mutation(value) if callable(mutation) else value

    def state_bytes(self) -> bytes:
        self.events.append("lr_scheduler.state_bytes")
        return b"scheduler-state-v1"


class _Scaler:
    def __init__(self, events: list[str], mutations: dict[str, object]) -> None:
        self.events = events
        self.mutations = mutations
        self.identity = _component_identity("amp_scaler")

    def scale(self, loss: float) -> float:
        self.events.append("amp_scaler.scale")
        mutation = self.mutations.get("amp_scaler.scale")
        return mutation(loss * 65536.0) if callable(mutation) else loss * 65536.0

    def step(self) -> dict[str, object]:
        self.events.append("amp_scaler.step")
        value = {"skipped": False, "overflow": False}
        mutation = self.mutations.get("amp_scaler.step")
        return mutation(value) if callable(mutation) else value

    def update(self) -> dict[str, object]:
        self.events.append("amp_scaler.update")
        value = {
            "scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "growth_tracker": 1,
        }
        mutation = self.mutations.get("amp_scaler.update")
        return mutation(value) if callable(mutation) else value

    def state_bytes(self) -> bytes:
        self.events.append("amp_scaler.state_bytes")
        return b"amp-scaler-state-v1"


class _EMA:
    def __init__(self, events: list[str], mutations: dict[str, object]) -> None:
        self.events = events
        self.mutations = mutations
        self.identity = _component_identity("ema")

    def update(self, model: object) -> dict[str, object]:
        self.events.append("ema.update")
        mutation = self.mutations.get("ema.update")
        value = {"updates": 1, "decay": 0.9999, "warmups": 2000}
        return mutation(value) if callable(mutation) else value

    def state_bytes(self) -> bytes:
        self.events.append("ema.state_bytes")
        return b"ema-state-v1"


class _Evaluator:
    def __init__(self, events: list[str], mutations: dict[str, object]) -> None:
        self.events = events
        self.mutations = mutations
        self.identity = _component_identity("primary_evaluator")

    def evaluate(self, development_batch: dict[str, object], weights: str, context: dict[str, object]) -> dict[str, object]:
        self.events.append("primary_evaluator.evaluate")
        value = {
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
        mutation = self.mutations.get("primary_evaluator.evaluate")
        return mutation(value) if callable(mutation) else value


class _BatchSource:
    def __init__(self, role: str, events: list[str], mutations: dict[str, object]) -> None:
        self.role = role
        self.events = events
        self.mutations = mutations
        self.identity = _component_identity(role)

    def next_batch(self) -> dict[str, object]:
        self.events.append(f"{self.role}.next_batch")
        if self.role == "train_batch_source":
            value = _batch("train_core", "train-batch-001", 16)
        else:
            value = _batch("development", "development-batch-001", 32)
        mutation = self.mutations.get(f"{self.role}.next_batch")
        return mutation(value) if callable(mutation) else value


class _Serializer:
    def __init__(self, events: list[str], mutations: dict[str, object]) -> None:
        self.events = events
        self.mutations = mutations
        self.identity = _component_identity("checkpoint_serializer")

    def serialize(self, states: dict[str, bytes]) -> dict[str, bytes]:
        self.events.append("checkpoint_serializer.serialize")
        mutation = self.mutations.get("checkpoint_serializer.serialize")
        value = copy.deepcopy(states)
        return mutation(value) if callable(mutation) else value


def _components(mutations: dict[str, object] | None = None) -> tuple[dict[str, object], list[str]]:
    mutations = {} if mutations is None else mutations
    events: list[str] = []
    return {
        "model": _Model(events, mutations),
        "criterion": _Criterion(events, mutations),
        "optimizer": _Optimizer(events, mutations),
        "lr_scheduler": _Scheduler(events, mutations),
        "amp_scaler": _Scaler(events, mutations),
        "ema": _EMA(events, mutations),
        "primary_evaluator": _Evaluator(events, mutations),
        "train_batch_source": _BatchSource("train_batch_source", events, mutations),
        "development_batch_source": _BatchSource("development_batch_source", events, mutations),
        "checkpoint_serializer": _Serializer(events, mutations),
    }, events


def _assert_adapter_import_isolated_in_fresh_process() -> None:
    code = """
import json
import sys

before = "torch" in sys.modules
from sparse_rtdetr.baseline import training_adapter
after = "torch" in sys.modules
print(json.dumps({"before": before, "after": after}, sort_keys=True))
if before or after:
    raise SystemExit(3)
"""
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads(result.stdout) == {"before": False, "after": False}
    assert result.stderr == ""


def _prepared(suffix: str = "run") -> dict[str, object]:
    global _PREPARED_COUNTER
    _PREPARED_COUNTER += 1
    nonce = f"{_PREPARED_COUNTER:032x}"
    return adapter.prepare_training_adapter(
        ROOT,
        run_id=f"t5c-{suffix}-{_PREPARED_COUNTER}",
        nonce=nonce,
    )


@pytest.fixture
def evidence_root(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic-evidence"
    try:
        yield root
    finally:
        if root.is_symlink():
            root.unlink()
        elif root.exists():
            shutil.rmtree(root)


def _expect_failure(prepared: dict[str, object], components: dict[str, object], root: Path) -> None:
    with pytest.raises(adapter.PreparedTrainerError):
        adapter.run_synthetic_prepared_batch(prepared, components, root)


def test_import_isolation_and_public_api_surface() -> None:
    assert set(adapter.__all__) == {
        "PreparedTrainerError",
        "prepare_training_adapter",
        "validate_prepared_training_adapter",
        "run_synthetic_prepared_batch",
    }
    _assert_adapter_import_isolated_in_fresh_process()
    source = (ROOT / "src/sparse_rtdetr/baseline/training_adapter.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    tree = ast.parse(source)
    imported = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    }
    imported.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imported == {
        "copy",
        "hashlib",
        "inspect",
        "json",
        "math",
        "os",
        "pathlib",
        "typing",
        "sparse_rtdetr.baseline.primary_evaluator",
        "sparse_rtdetr.baseline.training_contract",
        "sparse_rtdetr.baseline.training_evidence",
        "sparse_rtdetr.baseline.training_runtime",
    }


def test_import_isolation_probe_is_independent_of_parent_torch_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setitem(sys.modules, "torch", sentinel)
    _assert_adapter_import_isolated_in_fresh_process()
    assert sys.modules["torch"] is sentinel


def test_positive_prepared_binding_is_detached_and_authorities_are_immutable() -> None:
    prepared = _prepared("positive")
    validated = adapter.validate_prepared_training_adapter(prepared)
    assert validated == prepared
    assert validated is not prepared
    assert validated["authorities"] is not prepared["authorities"]
    validated["authorities"]["identity"]["vendor_runtime"]["file_count"] = 0
    assert prepared["authorities"]["identity"]["vendor_runtime"]["file_count"] == 124


@pytest.mark.parametrize(
    "path",
    [
        ("schema_version",),
        ("adapter_id",),
        ("mode",),
        ("authorities",),
        ("component_roles",),
        ("state_machine",),
        ("data_roles",),
        ("optimizer_observation",),
        ("amp_observation",),
        ("ema_observation",),
        ("evaluator_integration",),
        ("selection_policy",),
        ("evidence_checkpoint_handoff",),
        ("execution_policy",),
        ("aggregate_sha256",),
    ],
)
def test_prepared_descriptor_rejects_missing_top_level_keys(path: tuple[str, ...]) -> None:
    prepared = _prepared("schema-missing")
    current: dict[str, object] = prepared
    if len(path) == 1:
        del current[path[0]]
    else:
        raise AssertionError(path)
    with pytest.raises(adapter.PreparedTrainerError):
        adapter.validate_prepared_training_adapter(prepared)


@pytest.mark.parametrize(
    "path",
    [
        ("component_roles", "roles"),
        ("state_machine", "states"),
        ("data_roles", "allowed"),
        ("optimizer_observation", "groups"),
        ("amp_observation", "init_scale"),
        ("ema_observation", "decay"),
        ("evaluator_integration", "evaluator_id"),
        ("selection_policy", "tie_breakers"),
        ("evidence_checkpoint_handoff", "required_state_order"),
        ("execution_policy", "one_batch_only"),
    ],
)
def test_prepared_descriptor_rejects_nested_extra_and_retyped_values(path: tuple[str, str]) -> None:
    prepared = _prepared("schema-nested")
    nested = prepared[path[0]]
    assert isinstance(nested, dict)
    nested[path[1]] = copy.deepcopy(nested[path[1]])
    if isinstance(nested[path[1]], list):
        nested[path[1]].append("drift")
    elif isinstance(nested[path[1]], bool):
        nested[path[1]] = 1
    else:
        nested[path[1]] = f"{nested[path[1]]}-drift"
    with pytest.raises(adapter.PreparedTrainerError):
        adapter.validate_prepared_training_adapter(prepared)


def test_prepared_descriptor_rejects_extra_object_roles() -> None:
    prepared = _prepared("schema-extra")
    prepared["authorities"]["identity"]["extra"] = True
    with pytest.raises(adapter.PreparedTrainerError):
        adapter.validate_prepared_training_adapter(prepared)


def test_authority_mutations_are_rejected_before_prepared_descriptor_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _prepared("authority-base")
    mutations = {
        "training": ("training_contract_binding", "canonical_sha256"),
        "runtime": ("runtime_plan_binding", "canonical_sha256"),
        "evidence": ("evidence_contract", "schema_version"),
        "primary": ("primary_evaluator_binding", "config_canonical_sha256"),
    }
    for name, (payload_key, field) in mutations.items():
        payload = copy.deepcopy(base["authorities"]["payloads"][payload_key])
        if name == "evidence":
            payload[field] = 2
        else:
            payload[field] = "0" * 64

        if name == "training":
            monkeypatch.setattr(adapter._training_contract, "training_contract_binding", lambda root, payload=payload: copy.deepcopy(payload))
        elif name == "runtime":
            monkeypatch.setattr(adapter._training_runtime, "training_runtime_plan_binding", lambda root, payload=payload: copy.deepcopy(payload))
        elif name == "evidence":
            monkeypatch.setattr(adapter._training_evidence, "load_training_evidence_contract", lambda root, payload=payload: copy.deepcopy(payload))
        else:
            monkeypatch.setattr(adapter._primary_evaluator, "primary_evaluator_contract_binding", lambda root, payload=payload: copy.deepcopy(payload))
        with pytest.raises(adapter.PreparedTrainerError):
            adapter.prepare_training_adapter(ROOT, run_id=f"t5c-authority-{name}", nonce=f"{name:0<32}")
        monkeypatch.undo()


def test_component_roles_call_exactly_once_and_state_predecessors_are_live(evidence_root: Path) -> None:
    prepared = _prepared("positive-transaction")
    components, events = _components()
    result = adapter.run_synthetic_prepared_batch(prepared, components, evidence_root)
    assert result["state_sequence"] == [
        {"sequence": 0, "state": "CREATED", "predecessor_sha256": None},
        {"sequence": 1, "state": "AUTHORITIES_BOUND", "predecessor_sha256": adapter._digest({"sequence": 0, "state": "CREATED", "predecessor_sha256": None})},
    ] + [
        {
            "sequence": index,
            "state": state,
            "predecessor_sha256": adapter._digest(result["state_sequence"][index - 1]),
        }
        for index, state in enumerate(adapter.STATE_SEQUENCE[2:], 2)
    ]
    assert result["evidence"] == {
        "status": "SYNTHETIC_TERMINAL_COMPLETE",
        "classification": "SYNTHETIC_TERMINAL_COMPLETE",
        "epoch_count": 1,
        "checkpoint_count": 1,
        "terminalized_once": True,
    }
    for role, counts in result["call_counts"].items():
        assert all(value == 1 for value in counts.values()), role
    assert events.index("optimizer.step") < events.index("ema.update")
    assert events.index("ema.update") < events.index("primary_evaluator.evaluate")
    assert (evidence_root / "checkpoints" / "checkpoint-last-0001.ckpt").is_file()


def test_second_run_retry_resume_overwrite_and_second_batch_are_rejected(evidence_root: Path) -> None:
    prepared = _prepared("reuse")
    components, _ = _components()
    result = adapter.run_synthetic_prepared_batch(prepared, components, evidence_root)
    assert result["mode"] == "synthetic"
    with pytest.raises(adapter.PreparedTrainerError):
        adapter.run_synthetic_prepared_batch(prepared, _components()[0], evidence_root.parent / "second")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "role": "test"},
        lambda value: {**value, "batch_id": "/absolute/data"},
        lambda value: {**value, "batch_id": "confirmatory-batch"},
    ],
)
def test_data_role_and_path_allowlist_fail_closed(mutation: object, evidence_root: Path) -> None:
    prepared = _prepared("data-role")
    components, _ = _components({"train_batch_source.next_batch": mutation})
    _expect_failure(prepared, components, evidence_root)


def test_production_evidence_root_and_existing_root_are_rejected(evidence_root: Path) -> None:
    prepared = _prepared("root-policy")
    components, _ = _components()
    production = ROOT / training_evidence.EVIDENCE_ROOT_RELATIVE_PATH
    with pytest.raises(adapter.PreparedTrainerError):
        adapter.run_synthetic_prepared_batch(prepared, components, production)
    existing = evidence_root
    existing.mkdir()
    with pytest.raises(adapter.PreparedTrainerError):
        adapter.run_synthetic_prepared_batch(_prepared("root-existing"), _components()[0], existing)


@pytest.mark.parametrize(
    "role",
    ["model", "criterion", "optimizer", "lr_scheduler", "amp_scaler", "ema", "primary_evaluator", "train_batch_source", "development_batch_source", "checkpoint_serializer"],
)
def test_component_role_inventory_rejects_missing_and_extra_roles(role: str) -> None:
    components, _ = _components()
    del components[role]
    with pytest.raises(adapter.PreparedTrainerError):
        adapter._validate_ports(components)
    components, _ = _components()
    components["extra"] = object()
    with pytest.raises(adapter.PreparedTrainerError):
        adapter._validate_ports(components)


def test_component_protocol_rejects_extra_callable_bad_signature_and_async() -> None:
    components, _ = _components()
    components["model"].extra = lambda: None
    with pytest.raises(adapter.PreparedTrainerError):
        adapter._validate_ports(components)

    components, _ = _components()
    components["model"].forward = lambda batch, extra: None
    with pytest.raises(adapter.PreparedTrainerError):
        adapter._validate_ports(components)

    async def async_forward(batch: dict[str, object]) -> dict[str, object]:
        return {}

    components, _ = _components()
    components["model"].forward = async_forward
    with pytest.raises(adapter.PreparedTrainerError):
        adapter._validate_ports(components)


@pytest.mark.parametrize(
    "key, value",
    [
        ("model.forward", lambda value: {**value, "finite": False}),
        ("criterion.compute", lambda value: math.nan),
        ("amp_scaler.step", lambda value: {"skipped": True, "overflow": False}),
        ("amp_scaler.update", lambda value: {**value, "growth_tracker": 2}),
        ("lr_scheduler.observe", lambda value: {**value, "decay_events": 1}),
        ("ema.update", lambda value: {**value, "updates": 2}),
        ("primary_evaluator.evaluate", lambda value: {**value, "weights": "raw"}),
        ("checkpoint_serializer.serialize", lambda value: {**value, "unexpected": b"x"}),
    ],
)
def test_each_late_transition_mutation_fails_closed(key: str, value: object, evidence_root: Path) -> None:
    prepared = _prepared(f"mutation-{key.replace('.', '-')}")
    components, events = _components({key: value})
    _expect_failure(prepared, components, evidence_root)
    with pytest.raises(adapter.PreparedTrainerError):
        adapter.run_synthetic_prepared_batch(prepared, _components()[0], evidence_root.parent / "retry")


def test_failed_transition_stops_downstream_calls(evidence_root: Path) -> None:
    prepared = _prepared("early-stop")
    components, events = _components({"train_batch_source.next_batch": lambda value: {**value, "role": "test"}})
    _expect_failure(prepared, components, evidence_root)
    assert "model.forward" not in events
    assert "criterion.compute" not in events
    assert "primary_evaluator.evaluate" not in events


@pytest.mark.parametrize(
    "mutation",
    [
        lambda groups: [groups[0], groups[1], {**groups[2], "parameter_names": ["missing.weight"]}],
        lambda groups: [groups[0], {**groups[1], "parameter_names": ["backbone.norm.weight", "backbone.norm.weight"]}, groups[2]],
        lambda groups: [{**groups[0], "learning_rate": True}, groups[1], groups[2]],
        lambda groups: [groups[0], groups[1], {**groups[2], "numel": True}],
        lambda groups: [groups[1], groups[0], groups[2]],
        lambda groups: [groups[0], groups[1], {**groups[2], "parameter_names": ["frozen.bias"]}],
    ],
)
def test_optimizer_group_coverage_exclusivity_and_numeric_mutations_fail(mutation: object, evidence_root: Path) -> None:
    prepared = _prepared("optimizer-mutation")
    components, _ = _components({"optimizer.parameter_groups": mutation})
    _expect_failure(prepared, components, evidence_root)


def test_optimizer_group_observation_has_exact_counts_numel_and_lr(evidence_root: Path) -> None:
    prepared = _prepared("optimizer-positive")
    result = adapter.run_synthetic_prepared_batch(prepared, _components()[0], evidence_root)
    observation = result["optimizer_observation"]
    assert observation["group_count"] == 3
    assert observation["trainable_parameter_count"] == 3
    assert observation["trainable_numel"] == 60
    assert [group["numel"] for group in observation["groups"]] == [10, 20, 30]
    assert [group["learning_rate"] for group in observation["groups"]] == [0.00001, 0.0001, 0.0001]


@pytest.mark.parametrize(
    "key, mutation",
    [
        ("amp_scaler.scale", lambda value: value + 1.0),
        ("amp_scaler.step", lambda value: {"skipped": False, "overflow": True}),
        ("amp_scaler.update", lambda value: {**value, "scale": 1.0}),
        ("amp_scaler.update", lambda value: {**value, "growth_interval": True}),
        ("model.backward", lambda value: math.inf),
    ],
)
def test_amp_nonfinite_skip_overflow_growth_and_scale_mutations_fail(key: str, mutation: object, evidence_root: Path) -> None:
    prepared = _prepared(f"amp-{key.replace('.', '-')}")
    components, _ = _components({key: mutation})
    _expect_failure(prepared, components, evidence_root)


def test_ema_requires_policy_order_count_and_distinct_checkpoint_states(evidence_root: Path) -> None:
    prepared = _prepared("ema-positive")
    components, events = _components()
    result = adapter.run_synthetic_prepared_batch(prepared, components, evidence_root)
    assert result["ema_observation"] == {"updates": 1, "decay": 0.9999, "warmups": 2000}
    assert events.index("optimizer.step") < events.index("ema.update")
    assert result["checkpoint"]["state_count"] == 12
    assert result["checkpoint"]["state_summary"]["raw_model"]["sha256"] != result["checkpoint"]["state_summary"]["ema"]["sha256"]


def test_selection_uses_unrounded_ap_then_ap50_ar500_and_earlier_epoch() -> None:
    selected = adapter._select_development_candidate([
        {"epoch": 1, "AP": 0.5000000000001, "AP50": 0.1, "AR500": 0.1, "weights": "ema"},
        {"epoch": 2, "AP": 0.5, "AP50": 0.99, "AR500": 0.99, "weights": "ema"},
    ])
    assert selected["epoch"] == 1
    selected = adapter._select_development_candidate([
        {"epoch": 1, "AP": 0.5, "AP50": 0.7, "AR500": 0.6, "weights": "ema"},
        {"epoch": 2, "AP": 0.5, "AP50": 0.7, "AR500": 0.6, "weights": "ema"},
    ])
    assert selected["epoch"] == 1
    with pytest.raises(adapter.PreparedTrainerError):
        adapter._select_development_candidate([
            {"epoch": 1, "AP": 0.5, "AP50": 0.7, "AR500": 0.6, "weights": "raw"},
        ])


def test_evaluator_is_development_only_and_binds_run_ema_and_predecessor(evidence_root: Path) -> None:
    prepared = _prepared("evaluator-positive")
    result = adapter.run_synthetic_prepared_batch(prepared, _components()[0], evidence_root)
    observed = result["evaluator_observation"]
    assert observed == {
        "evaluator_id": "visdrone_official_primary_evaluator_v1",
        "protocol_id": "visdrone_official_style_v1",
        "role": "development",
        "weights": "ema",
        "result_sha256": result["input_output_identity"]["evaluator_result_sha256"],
    }
    assert result["selection_observation"]["pass"] is True


def test_checkpoint_and_evidence_handoff_is_atomic_and_authority_bound(evidence_root: Path) -> None:
    prepared = _prepared("handoff")
    result = adapter.run_synthetic_prepared_batch(prepared, _components()[0], evidence_root)
    validated = training_evidence.validate_training_evidence(str(evidence_root))
    assert validated["status"] == "SYNTHETIC_TERMINAL_COMPLETE"
    assert validated["epoch_count"] == 1
    assert validated["checkpoint_count"] == 1
    assert validated["checkpoints"]["checkpoints/checkpoint-last-0001.ckpt"]["required_state_order"] == list(training_evidence.CHECKPOINT_REQUIRED_STATES)
    checkpoint = training_evidence.validate_training_checkpoint(
        str(evidence_root / "checkpoints" / "checkpoint-last-0001.ckpt")
    )
    assert checkpoint["training_contract_sha256"] == prepared["authorities"]["payloads"]["training_contract_binding"]["canonical_sha256"]
    assert result["evidence"]["classification"] == training_evidence.classify_training_evidence(str(evidence_root))


def test_checkpoint_serializer_rejects_raw_ema_alias_and_missing_state(evidence_root: Path) -> None:
    prepared = _prepared("checkpoint-alias")
    components, _ = _components({"checkpoint_serializer.serialize": lambda states: {**states, "ema": states["raw_model"]}})
    _expect_failure(prepared, components, evidence_root)

    prepared = _prepared("checkpoint-missing")
    components, _ = _components({"checkpoint_serializer.serialize": lambda states: {name: value for name, value in states.items() if name != "rng_states"}})
    _expect_failure(prepared, components, evidence_root)


def test_result_and_component_identity_outputs_are_detached(evidence_root: Path) -> None:
    prepared = _prepared("detached-result")
    components, _ = _components()
    result = adapter.run_synthetic_prepared_batch(prepared, components, evidence_root)
    result["authority_identity"]["training_contract"]["id"] = "mutated"
    assert prepared["authorities"]["identity"]["training_contract"]["id"] == "rtdetrv2_r18_visdrone_baseline_training_v1"
    result["component_identities"]["model"]["implementation"] = "mutated"
    assert components["model"].identity["implementation"] == "t5c-test-model-v1"


def test_no_production_artifact_or_external_execution_is_used(evidence_root: Path) -> None:
    production = ROOT / training_evidence.EVIDENCE_ROOT_RELATIVE_PATH
    assert not production.exists()
    prepared = _prepared("no-production")
    adapter.run_synthetic_prepared_batch(prepared, _components()[0], evidence_root)
    assert not production.exists()
    assert not any(path.name.startswith(".t5c") for path in ROOT.iterdir())


def test_clean_archive_missing_conversion_artifacts_fails_closed(tmp_path: Path) -> None:
    clean_root = tmp_path / "clean-root"
    clean_root.mkdir()
    with pytest.raises(adapter.PreparedTrainerError):
        adapter.prepare_training_adapter(clean_root, run_id="t5c-clean-archive", nonce="clean-archive")
