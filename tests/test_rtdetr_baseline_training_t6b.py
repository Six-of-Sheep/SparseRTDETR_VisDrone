"""CPU/fake coverage for the T6B production execution boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_t6_authorization as t6a
from sparse_rtdetr.baseline import training_t6_engine as engine
from sparse_rtdetr.baseline import training_t6_entry as entry
from sparse_rtdetr.baseline import training_t6_outer_launcher as outer
from sparse_rtdetr.baseline import training_t6_process_launcher as process


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_fake_launch_git_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthetic descriptors independent from the host checkout identity."""

    monkeypatch.setattr(entry, "_validate_launch_git_identity", lambda descriptor: None)


def _environment() -> dict[str, object]:
    return {
        "python": {"path": "/usr/bin/python", "realpath": "/usr/bin/python", "version": "3.10.16", "sha256": "7ed96f9b2f4d3da2c7c7233d9cb968163cfad01b02b2f2e6300cf036d769cfeb"},
        "tmux": {"path": "/usr/bin/tmux", "realpath": "/usr/bin/tmux", "version": "3.3a", "sha256": "b" * 64},
        "gpu": {"index": 0, "name": "NVIDIA GeForce RTX 4090 D", "uuid": "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8", "driver_version": "555.42", "cuda_version": "12.4", "power_limit_watts": 425.0},
        "filesystem": {"device": 1, "mount": "/tmp"},
        "variables": {"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")},
    }


def _t6a_environment() -> dict[str, object]:
    return {key: value for key, value in _environment().items() if key != "variables"}


def _t6a_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    contract_binding = t6a.training_launch_contract_binding(ROOT)
    contract = contract_binding["contract"]
    nonce = hashlib.sha256(b"t6b-cpu-fake-authorization").hexdigest()
    repository = {
        "repo_root": str(ROOT),
        "branch": "codex/p3-rtdetrv2-baseline-training-t6b-runtime-binding-r3-repair",
        "head": "1" * 40,
        "tree": "2" * 40,
        "parent": "3" * 40,
        "upstream": "origin/codex/p3-rtdetrv2-baseline-training-t6b-runtime-binding-r3-repair",
        "upstream_sha": "1" * 40,
    }
    data_roles = {
        "train_core": {"role": "train_core", "root": "/tmp/t6b_train_core_v1", "manifest_sha256": "5" * 64},
        "development": {"role": "development", "root": "/tmp/t6b_development_v1", "manifest_sha256": "6" * 64},
        "confirmatory": {"role": "confirmatory", "access": "sealed_and_forbidden", "identity_read": "forbidden"},
        "test": {"role": "test", "access": "forbidden", "identity_read": "forbidden"},
    }
    observed = {
        "repository": repository,
        "targets": copy.deepcopy(contract["targets"]),
        "target_absence": {"training_evidence_root": True, "process_evidence_root": True, "outer_evidence_root": True, "tmux_session": True, "receipt": True, "lock": True},
        "contract_identity": contract_binding["contract_identity"],
        "source_bindings": copy.deepcopy(contract["source_bindings"]),
        "environment_identity": _t6a_environment(),
        "data_roles": data_roles,
        "training_policy": copy.deepcopy(contract["training_policy"]),
        "state_machine": copy.deepcopy(contract["state_machine"]),
        "training_run_id": contract["run_identity"]["training_run_id"],
        "tmux_session_name": contract["run_identity"]["tmux_session_name"],
        "nonce": nonce,
    }
    authorization = {
        "schema_version": 1,
        "authorization_id": "rtdetrv2_r18_visdrone_training_owner_authorization_v1",
        "authorized": True,
        "contract_id": t6a.LAUNCH_CONTRACT_ID,
        "training_run_id": contract["run_identity"]["training_run_id"],
        "nonce": nonce,
        "tmux_session_name": contract["run_identity"]["tmux_session_name"],
        "targets": copy.deepcopy(contract["targets"]),
        "target_absence": copy.deepcopy(observed["target_absence"]),
        "repository": copy.deepcopy(repository),
        "contract_identity": copy.deepcopy(contract_binding["contract_identity"]),
        "source_bindings": copy.deepcopy(contract["source_bindings"]),
        "environment_identity": copy.deepcopy(_t6a_environment()),
        "data_roles": copy.deepcopy(data_roles),
        "training_policy": copy.deepcopy(contract["training_policy"]),
        "state_machine": copy.deepcopy(contract["state_machine"]),
        "owner": {"owner_id": "project_owner", "uid": 1000, "gid": 1000, "detached_external": True, "self_issued": False},
    }
    canonical = t6a.canonical_owner_authorization_bytes(authorization)
    raw = canonical + b"\n"
    file_identity = {"regular": True, "symlink": False, "mode": 384, "uid": 1000, "gid": 1000, "nlink": 1}
    binding = t6a.owner_authorization_binding(
        authorization,
        expected_raw_size_bytes=len(raw),
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
        contract_binding=contract_binding,
        observed_binding=observed,
        authorization_file_identity=file_identity,
    )
    return contract_binding, binding, observed


def _descriptor(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object], Path]:
    contract_binding, binding, _ = _t6a_fixture(tmp_path)
    auth_path = tmp_path / "detached-owner-authorization.json"
    auth_receipt = entry._authorization_receipt_path(auth_path)
    receipt = {
        "schema_version": 1,
        "kind": "T6B_DETACHED_AUTHORIZATION_CONSUMED",
        "authorization_id": binding["authorization_id"],
        "training_run_id": binding["authorization"]["training_run_id"],
        "tmux_session_name": binding["authorization"]["tmux_session_name"],
        "nonce": binding["authorization"]["nonce"],
        "authorization_path": str(auth_path),
        "authorization_binding_sha256": entry._digest(binding),
        "raw_size_bytes": binding["raw_size_bytes"],
        "raw_sha256": binding["raw_sha256"],
        "canonical_size_bytes": binding["canonical_size_bytes"],
        "canonical_sha256": binding["canonical_sha256"],
        "consumed": True,
    }
    entry._write_new(auth_receipt, entry._canonical(receipt), "test authorization receipt")
    repository = copy.deepcopy(binding["authorization"]["repository"])
    targets = {"training_evidence_root": str(tmp_path / "training-evidence"), "process_evidence_root": str(tmp_path / "process-evidence"), "outer_evidence_root": str(tmp_path / "outer-evidence")}
    descriptor = entry.build_production_entry_descriptor(
        ROOT,
        authorization_binding=binding,
        authorization_path=auth_path,
        authorization_receipt_path=auth_receipt,
        t6a_contract_binding=contract_binding,
        repository=repository,
        environment=_environment(),
        data_roles={"train_core": {"role": "train_core", "root": "/tmp/t6b_train_core_runtime_v1", "manifest_sha256": "b" * 64}, "development": {"role": "development", "root": "/tmp/t6b_development_runtime_v1", "manifest_sha256": "c" * 64}, "test": {"role": "test", "access": "forbidden", "identity_read": "forbidden"}, "confirmatory": {"role": "confirmatory", "access": "sealed_and_forbidden", "identity_read": "forbidden"}},
        cwd=ROOT,
        argv=["/usr/bin/python", "-m", "sparse_rtdetr.baseline.training_t6_entry"],
        targets=targets,
    )
    process_descriptor = process.build_production_process_descriptor(descriptor)
    outer_descriptor = outer.build_production_outer_descriptor(process_descriptor)
    return descriptor, process_descriptor, outer_descriptor, auth_receipt


class _Optimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class _Scaler:
    def scale(self, value: object) -> object:
        return value

    def step(self, optimizer: _Optimizer) -> bool:
        optimizer.step()
        return True

    def update(self) -> None:
        return None


class _EMA:
    def __init__(self) -> None:
        self.updates = 0

    def update(self, model: object) -> None:
        self.updates += 1


class _Writer:
    def __init__(self) -> None:
        self.saved: list[tuple[str, int]] = []

    def save_atomic(self, role: str, epoch: int, state: dict[str, object]) -> dict[str, object]:
        self.saved.append((role, epoch))
        return {"role": role, "epoch": epoch, "state_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()}

    def is_loadable(self, reference: dict[str, object]) -> bool:
        return type(reference.get("state_sha256")) is str


class _Evaluator:
    role = "development_only"

    def evaluate(self, weights: object, batches: object, epoch: int) -> dict[str, object]:
        return {"primary_score": float(epoch), "role": "development_only"}


def _engine_ports() -> dict[str, object]:
    return {
        "production_authorized": True,
        "authorize_production": lambda: True,
        "model_factory": lambda: (lambda batch: batch),
        "criterion_factory": lambda: (lambda output, batch: 1.0),
        "optimizer_factory": _Optimizer,
        "scheduler_factory": lambda: type("Scheduler", (), {"step": lambda self: None})(),
        "scaler_factory": _Scaler,
        "ema_factory": _EMA,
        "train_batches": lambda: [{"batch": "train"}],
        "development_batches": lambda: [{"batch": "development"}],
        "checkpoint_writer": _Writer,
        "evaluator": _Evaluator,
        "backward": lambda loss: None,
        "amp_counters": lambda: {"overflow_events": 0, "skipped_optimizer_steps": 0},
    }


def test_engine_accepts_vendor_batch_and_loss_dict_and_applies_warmup() -> None:
    policy = entry.load_t6b_config(ROOT)["training_policy"]
    warmup_calls: list[int] = []
    scheduler_calls: list[int] = []

    class Warmup:
        def step(self) -> None:
            warmup_calls.append(1)

        def finished(self) -> bool:
            return len(warmup_calls) >= 2000

    class Scheduler:
        def step(self) -> None:
            scheduler_calls.append(1)

    class Criterion:
        weight_dict = {"loss_cls": 1.0, "loss_bbox": 1.0}

        def __call__(self, outputs: object, targets: object, **kwargs: object) -> dict[str, float]:
            assert outputs == "vendor-output"
            assert type(targets) is list
            assert kwargs["epoch"] >= 1
            return {"loss_cls": 1.0, "loss_bbox": 2.0}

    ports = _engine_ports()
    ports.update(
        {
            "model_factory": lambda: (lambda samples, targets=None: "vendor-output"),
            "criterion_factory": Criterion,
            "scheduler_factory": Scheduler,
            "warmup_scheduler_factory": Warmup,
            "train_batches": lambda: [("vendor-samples", [{"labels": [0]}])],
        }
    )
    result = engine.run_training_engine(policy, ports, mode="cpu_fake")
    assert result["terminal"] == "TERMINAL_COMPLETE"
    assert len(warmup_calls) == 120
    assert scheduler_calls == []


def test_runtime_data_role_binding_uses_exact_manifest_identity(tmp_path: Path) -> None:
    from sparse_rtdetr.baseline import config as baseline_config

    artifact_root = ROOT / "artifacts" / "data" / "visdrone_protocol_v2_conversion_r3"
    roles = {
        "train_core": {"role": "train_core", "root": str(tmp_path / "train_core"), "manifest_sha256": hashlib.sha256((artifact_root / "train_core_manifest.json").read_bytes()).hexdigest()},
        "development": {"role": "development", "root": str(tmp_path / "development"), "manifest_sha256": hashlib.sha256((artifact_root / "development_manifest.json").read_bytes()).hexdigest()},
        "test": {"role": "test", "access": "forbidden", "identity_read": "forbidden"},
        "confirmatory": {"role": "confirmatory", "access": "sealed_and_forbidden", "identity_read": "forbidden"},
    }
    Path(roles["train_core"]["root"]).mkdir()
    Path(roles["development"]["root"]).mkdir()
    checked = engine._validate_runtime_data_roles(str(ROOT), roles, baseline_config)
    assert checked["train_core"]["runtime"]["runtime"]["role"] == "train_core"
    mutated = copy.deepcopy(roles)
    mutated["development"]["manifest_sha256"] = "0" * 64
    with pytest.raises(engine.TrainingEngineError, match="manifest identity"):
        engine._validate_runtime_data_roles(str(ROOT), mutated, baseline_config)


def test_default_resolver_receives_descriptor_bindings_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor, _, _, _ = _descriptor(tmp_path)
    observed: list[dict[str, object]] = []

    def resolver(**kwargs: object) -> dict[str, object]:
        observed.append(kwargs)
        return _engine_ports()

    monkeypatch.setattr(engine, "resolve_production_ports", resolver)
    result = entry.run_production_entry(descriptor, engine_ports=None, process_pid=4242)
    assert result["status"] == "TERMINAL_COMPLETE"
    assert observed == [
        {
            "repo_root": descriptor["repository"]["repo_root"],
            "data_roles": descriptor["data_roles"],
            "output_root": descriptor["evidence_root"],
        }
    ]


def _rewrite_canonical(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(entry._canonical(value))


def _refresh_entry_evidence(root: Path) -> None:
    result_path = root / "entry_result.json"
    result = json.loads(result_path.read_text())
    body = {key: value for key, value in result.items() if key != "aggregate_result_sha256"}
    result["aggregate_result_sha256"] = entry._digest(body)
    _rewrite_canonical(result_path, result)
    rows = entry._entry_inventory(root)
    inventory = {
        "schema_version": 1,
        "excluded": ["entry_artifact_inventory.json", "entry_completion.json"],
        "files": rows,
        "canonical_inventory_sha256": entry._digest(rows),
    }
    _rewrite_canonical(root / "entry_artifact_inventory.json", inventory)
    completion = {
        "schema_version": 1,
        "status": result["status"],
        "classification": result["classification"],
        "result_sha256": entry._sha(entry._canonical(result)),
        "inventory_sha256": entry._sha(entry._canonical(inventory)),
        "evidence_receipt_sha256": entry._sha(entry._read_stable_file(Path(result["evidence_receipt_path"]), "claim")),
        "failure_class": result["failure_class"],
    }
    _rewrite_canonical(root / "entry_completion.json", completion)


def _refresh_process_evidence(root: Path) -> None:
    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    body = {key: value for key, value in result.items() if key != "aggregate_result_sha256"}
    result["aggregate_result_sha256"] = process._digest(body)
    _rewrite_canonical(result_path, result)
    rows = process._inventory(root)
    inventory = {
        "schema_version": 1,
        "excluded": ["artifact_inventory.json", "completion.json"],
        "files": rows,
        "canonical_inventory_sha256": process._digest(rows),
    }
    _rewrite_canonical(root / "artifact_inventory.json", inventory)
    stream_path = root.parent / f".{root.name}{process.PROCESS_STREAM_RECEIPT_SUFFIX}"
    claim_path = Path(result["process_receipt_path"])
    claim_raw = entry._read_stable_file(claim_path, "process claim")
    stream_raw = entry._read_stable_file(stream_path, "process stream receipt")
    completion = {
        "schema_version": 1,
        "status": result["status"],
        "classification": result["classification"],
        "exit_code": result["return_code"],
        "descriptor_sha256": result["descriptor_sha256"],
        "result_sha256": process._sha(process._canonical(result)),
        "inventory_sha256": process._sha(process._canonical(inventory)),
        "stdout_size_bytes": result["stdout_size_bytes"],
        "stdout_sha256": result["stdout_sha256"],
        "stderr_size_bytes": result["stderr_size_bytes"],
        "stderr_sha256": result["stderr_sha256"],
        "entry_result_sha256": result["entry_result_sha256"],
        "state_transition_sha256": result["state_transition_sha256"],
        "receipt_sha256": process._sha(claim_raw),
        "stream_receipt_sha256": process._sha(stream_raw),
    }
    _rewrite_canonical(root / "completion.json", completion)


def _refresh_outer_evidence(root: Path) -> None:
    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    body = {key: value for key, value in result.items() if key != "aggregate_result_sha256"}
    result["aggregate_result_sha256"] = outer._digest(body)
    _rewrite_canonical(result_path, result)
    snapshot_path = root / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot.update(
        {
            "tmux_return_code": result["return_code"],
            "tmux_pid": result["pid"],
            "stdout_size_bytes": result["stdout_size_bytes"],
            "stdout_sha256": result["stdout_sha256"],
            "stderr_size_bytes": result["stderr_size_bytes"],
            "stderr_sha256": result["stderr_sha256"],
        }
    )
    _rewrite_canonical(snapshot_path, snapshot)
    rows = outer._inventory(root)
    inventory = {
        "schema_version": 1,
        "excluded": ["artifact_inventory.json", "completion.json"],
        "files": rows,
        "canonical_inventory_sha256": outer._digest(rows),
    }
    _rewrite_canonical(root / "artifact_inventory.json", inventory)
    claim_raw = entry._read_stable_file(Path(result["outer_receipt_path"]), "outer claim")
    stream_path = root.parent / f".{root.name}{outer.OUTER_STREAM_RECEIPT_SUFFIX}"
    stream_raw = entry._read_stable_file(stream_path, "outer stream receipt")
    completion = {
        "schema_version": 1,
        "status": result["status"],
        "classification": result["classification"],
        "exit_code": result["return_code"],
        "descriptor_sha256": result["descriptor_sha256"],
        "result_sha256": outer._sha(outer._canonical(result)),
        "inventory_sha256": outer._sha(outer._canonical(inventory)),
        "snapshot_sha256": outer._sha(outer._canonical(snapshot)),
        "stdout_size_bytes": result["stdout_size_bytes"],
        "stdout_sha256": result["stdout_sha256"],
        "stderr_size_bytes": result["stderr_size_bytes"],
        "stderr_sha256": result["stderr_sha256"],
        "receipt_sha256": outer._sha(claim_raw),
        "stream_receipt_sha256": outer._sha(stream_raw),
    }
    _rewrite_canonical(root / "completion.json", completion)


def test_config_and_import_boundaries_are_closed() -> None:
    config = entry.load_t6b_config(ROOT)
    assert config["readiness"]["static_config_authorizes_production"] is False
    assert config["invocation_policy"]["shell"] is False
    assert config["run_identity"]["run_id_source"] == "detached_owner_authorization"
    assert set(engine.__all__) == {"TrainingEngineError", "validate_training_policy", "resolve_production_ports", "run_training_engine"}


def test_engine_rejects_unauthorized_production_ports() -> None:
    policy = entry.load_t6b_config(ROOT)["training_policy"]
    with pytest.raises(engine.TrainingEngineError, match="requires entry-authorized"):
        engine.run_training_engine(policy, {"model_factory": lambda: None}, mode="production")


def test_engine_rejects_forged_production_flag_without_entry_capability() -> None:
    policy = entry.load_t6b_config(ROOT)["training_policy"]
    ports = _engine_ports()
    ports["production_authorized"] = True
    with pytest.raises(engine.TrainingEngineError, match="requires entry-authorized"):
        engine.run_training_engine(policy, ports, mode="production")


def test_descriptor_binds_repository_to_detached_authorization_and_live_source(tmp_path: Path) -> None:
    descriptor, _, _, _ = _descriptor(tmp_path)
    repository_mutation = copy.deepcopy(descriptor)
    repository_mutation["repository"]["head"] = "e" * 40
    repository_body = {key: value for key, value in repository_mutation.items() if key != "aggregate_sha256"}
    repository_mutation["aggregate_sha256"] = entry._digest(repository_body)
    with pytest.raises(entry.TrainingEntryError, match="detached authorization"):
        entry.validate_entry_descriptor(repository_mutation)

    source_mutation = copy.deepcopy(descriptor)
    source_mutation["source_bindings"]["t6b_modules"][0]["sha256"] = "d" * 64
    source_body = {key: value for key, value in source_mutation.items() if key != "aggregate_sha256"}
    source_mutation["aggregate_sha256"] = entry._digest(source_body)
    with pytest.raises(entry.TrainingEntryError, match="source identity"):
        entry.validate_entry_descriptor(source_mutation)


def test_entry_classifier_rejects_coordinated_result_repack(tmp_path: Path) -> None:
    descriptor, _, _, _ = _descriptor(tmp_path)
    result = entry.run_production_entry(descriptor, engine_ports=_engine_ports(), process_pid=4242)
    root = Path(descriptor["evidence_root"])
    assert result["status"] == "TERMINAL_COMPLETE"
    assert entry.classify_entry(root) == "TERMINAL_COMPLETE"
    result_path = root / "entry_result.json"
    persisted = json.loads(result_path.read_text())
    persisted["repository"]["head"] = "f" * 40
    _rewrite_canonical(result_path, persisted)
    _refresh_entry_evidence(root)
    assert entry.classify_entry(root) == "UNKNOWN"


def test_process_classifier_rejects_coordinated_binary_repack(tmp_path: Path) -> None:
    _, process_descriptor, _, _ = _descriptor(tmp_path)

    def failed(argv: list[str], cwd: str, environment: dict[str, str], child_descriptor: dict[str, object]) -> dict[str, object]:
        return {"return_code": 9, "stdout": b"original\x00", "stderr": b"err\xff", "pid": 9, "entry_result": None}

    process.run_process_once(process_descriptor, failed)
    root = Path(process_descriptor["process_evidence_root"])
    persisted = json.loads((root / "result.json").read_text())
    persisted["stdout_sha256"] = "0" * 64
    _rewrite_canonical(root / "result.json", persisted)
    _refresh_process_evidence(root)
    assert process.classify_process(root) == "UNKNOWN"


def test_outer_classifier_rejects_coordinated_binary_repack(tmp_path: Path) -> None:
    _, _, outer_descriptor, _ = _descriptor(tmp_path)

    def tmux(argv: list[str], cwd: str, environment: dict[str, str], descriptor: dict[str, object]) -> dict[str, object]:
        return {"return_code": 0, "stdout": b"tmux\x00accepted\xff", "stderr": b"outer\x80", "pid": 8181}

    outer.run_outer_once(outer_descriptor, tmux)
    root = Path(outer_descriptor["outer_evidence_root"])
    persisted = json.loads((root / "result.json").read_text())
    persisted["stdout_sha256"] = "1" * 64
    _rewrite_canonical(root / "result.json", persisted)
    _refresh_outer_evidence(root)
    assert outer.classify_outer(root) == "UNKNOWN"


@pytest.mark.parametrize(
    ("failure_kind", "failure_class", "return_code"),
    (
        ("timeout", "RUNNER_TIMEOUT", 124),
        ("signal", "RUNNER_SIGNAL", 130),
        ("exception", "RUNNER_EXCEPTION", 2),
        ("schema", "RUNNER_SCHEMA_FAILURE", 5),
        ("mutation", "RUNNER_MUTATED_INVOCATION", 6),
    ),
)
def test_process_runner_failures_are_terminal(
    tmp_path: Path,
    failure_kind: str,
    failure_class: str,
    return_code: int,
) -> None:
    case_root = tmp_path / failure_kind
    case_root.mkdir()
    _, process_descriptor, _, _ = _descriptor(case_root)
    calls: list[int] = []

    def runner(argv: list[str], cwd: str, environment: dict[str, str], child_descriptor: dict[str, object]) -> dict[str, object]:
        calls.append(1)
        if failure_kind == "timeout":
            raise TimeoutError("fake timeout")
        if failure_kind == "signal":
            raise KeyboardInterrupt()
        if failure_kind == "exception":
            raise ValueError("fake runner exception")
        if failure_kind == "schema":
            return {}
        argv.append("mutated")
        return {"return_code": 0, "stdout": b"", "stderr": b"", "pid": 1, "entry_result": None}

    result = process.run_process_once(process_descriptor, runner)
    assert result["status"] == "PERMANENT_FAIL"
    assert result["failure_class"] == failure_class
    assert result["return_code"] == return_code
    assert process.classify_process(process_descriptor["process_evidence_root"]) == "PERMANENT_FAIL"
    assert calls == [1]


def test_outer_target_absence_rejects_existing_object_types(tmp_path: Path) -> None:
    for kind in ("regular", "directory", "symlink", "hardlink", "fifo"):
        case_root = tmp_path / kind
        case_root.mkdir()
        _, _, outer_descriptor, _ = _descriptor(case_root)
        target = Path(outer_descriptor["targets"]["outer_evidence_root"])
        if kind == "regular":
            target.write_bytes(b"occupied")
        elif kind == "directory":
            target.mkdir()
        elif kind == "symlink":
            other = case_root / "other"
            other.mkdir()
            target.symlink_to(other, target_is_directory=True)
        elif kind == "hardlink":
            other = case_root / "occupied"
            other.write_bytes(b"occupied")
            os.link(other, target)
        else:
            os.mkfifo(target)
        with pytest.raises(outer.TrainingOuterError, match="target already exists"):
            outer.run_outer_once(outer_descriptor, lambda *args: {})




def test_full_fake_process_chain_is_exactly_once_and_terminal(tmp_path: Path) -> None:
    descriptor, process_descriptor, _, _ = _descriptor(tmp_path)
    calls: list[int] = []

    def child(argv: list[str], cwd: str, environment: dict[str, str], child_descriptor: dict[str, object]) -> dict[str, object]:
        calls.append(1)
        result = entry.run_production_entry(child_descriptor, engine_ports=_engine_ports(), process_pid=4242)
        return {"return_code": 0, "stdout": b"child\x00stdout\xff", "stderr": b"child\x80stderr", "pid": 4242, "entry_result": result}

    result = process.run_process_once(process_descriptor, child)
    assert result["status"] == "TERMINAL_COMPLETE"
    assert result["failure_class"] == "NONE"
    assert result["child_invocation_count"] == 1
    assert result["production_training_authorized"] is True
    assert calls == [1]
    assert process.classify_process(process_descriptor["process_evidence_root"]) == "TERMINAL_COMPLETE"
    assert (Path(process_descriptor["process_evidence_root"]) / "stdout.bin").read_bytes() == b"child\x00stdout\xff"
    with pytest.raises(process.TrainingProcessError, match="already exists"):
        process.run_process_once(process_descriptor, child)
    assert calls == [1]
    assert descriptor["mode"] == "production"


def test_process_failures_publish_permanent_failure_without_entry_result(tmp_path: Path) -> None:
    _, process_descriptor, _, _ = _descriptor(tmp_path)
    calls: list[int] = []

    def failed(argv: list[str], cwd: str, environment: dict[str, str], child_descriptor: dict[str, object]) -> dict[str, object]:
        calls.append(1)
        return {"return_code": 17, "stdout": b"bad\x00", "stderr": b"error\xff", "pid": 99, "entry_result": None}

    result = process.run_process_once(process_descriptor, failed)
    assert result["status"] == "PERMANENT_FAIL"
    assert result["failure_class"] == "RUNNER_NONZERO_EXIT"
    assert result["production_training_authorized"] is False
    assert result["return_code"] == 17
    assert process.classify_process(process_descriptor["process_evidence_root"]) == "PERMANENT_FAIL"
    assert calls == [1]


def test_outer_tmux_acceptance_is_not_training_certification(tmp_path: Path) -> None:
    _, process_descriptor, outer_descriptor, _ = _descriptor(tmp_path)
    outer_calls: list[int] = []

    def child(argv: list[str], cwd: str, environment: dict[str, str], child_descriptor: dict[str, object]) -> dict[str, object]:
        return {"return_code": 1, "stdout": b"child", "stderr": b"failure", "pid": 0, "entry_result": None}

    def tmux(argv: list[str], cwd: str, environment: dict[str, str], descriptor: dict[str, object]) -> dict[str, object]:
        outer_calls.append(1)
        process.run_process_once(process_descriptor, child)
        return {"return_code": 0, "stdout": b"tmux\x00accepted\xff", "stderr": b"", "pid": 8181}

    result = outer.run_outer_once(outer_descriptor, tmux)
    assert result["status"] == "TMUX_ACCEPTED"
    assert result["training_certified"] is False
    assert result["production_training_authorized"] is False
    assert outer.classify_outer(outer_descriptor["outer_evidence_root"]) == "LAUNCH_ACCEPTED"
    assert process.classify_process(process_descriptor["process_evidence_root"]) == "PERMANENT_FAIL"
    assert outer_calls == [1]
    with pytest.raises(outer.TrainingOuterError, match="already exists"):
        outer.run_outer_once(outer_descriptor, tmux)
    assert outer_calls == [1]


def test_source_and_binary_mutations_are_not_accepted(tmp_path: Path) -> None:
    _, process_descriptor, _, _ = _descriptor(tmp_path)

    def failed(argv: list[str], cwd: str, environment: dict[str, str], child_descriptor: dict[str, object]) -> dict[str, object]:
        return {"return_code": 9, "stdout": b"out", "stderr": b"err", "pid": 9, "entry_result": None}

    process.run_process_once(process_descriptor, failed)
    root = Path(process_descriptor["process_evidence_root"])
    (root / "stdout.bin").write_bytes(b"different")
    assert process.classify_process(root) == "UNKNOWN"
    mutated = copy.deepcopy(process_descriptor)
    mutated["entry_descriptor"]["source_bindings"]["t6b_modules"][0]["sha256"] = "d" * 64
    entry_body = {key: value for key, value in mutated["entry_descriptor"].items() if key != "aggregate_sha256"}
    mutated["entry_descriptor"]["aggregate_sha256"] = entry._digest(entry_body)
    body = {key: value for key, value in mutated.items() if key != "aggregate_sha256"}
    mutated["aggregate_sha256"] = process._digest(body)
    with pytest.raises(process.TrainingProcessError, match="source identity"):
        process.validate_process_descriptor(mutated)
