from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_contract
from sparse_rtdetr.baseline import training_evidence as evidence
from sparse_rtdetr.baseline import training_runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/sparse_rtdetr/baseline/training_evidence.py"
CONFIG = ROOT / evidence.EVIDENCE_CONTRACT_CONFIG_RELATIVE_PATH


@lru_cache(maxsize=1)
def _authority_bindings() -> tuple[dict, dict]:
    return (
        training_contract.training_contract_binding(str(ROOT)),
        training_runtime.training_runtime_plan_binding(str(ROOT)),
    )


def _bindings() -> tuple[dict, dict]:
    training, runtime = _authority_bindings()
    return copy.deepcopy(training), copy.deepcopy(runtime)


def _identity(name: str, digit: str = "0") -> dict[str, str]:
    return {"identity_id": name, "sha256": digit * 64}


def _states() -> dict[str, bytes]:
    return {name: (f"synthetic-{index}-{name}").encode("ascii") for index, name in enumerate(evidence.CHECKPOINT_REQUIRED_STATES)}


def _epoch(epoch: int, step: int | None = None, *, ema: int | None = None) -> dict:
    return {
        "epoch": epoch,
        "global_optimizer_step": epoch if step is None else step,
        "loss": 1.0 / epoch,
        "gradient_norm": 0.5,
        "amp_scale": 1.0,
        "amp_skipped_steps": 0,
        "amp_overflow_events": 0,
        "nonfinite_loss_count": 0,
        "nonfinite_gradient_count": 0,
        "ema_updates": epoch if ema is None else ema,
        "evaluator_result_sha256": f"{epoch:x}" * 64,
    }


def _writer(tmp_path: Path, *, mode: str = "synthetic") -> evidence.TrainingEvidenceWriter:
    tmp_path.mkdir(parents=True, exist_ok=True)
    training, runtime = _bindings()
    return evidence.TrainingEvidenceWriter(
        str(tmp_path / "evidence"),
        run_id="synthetic-run",
        nonce="synthetic-nonce",
        mode=mode,
        training_contract_binding=training,
        runtime_plan_binding=runtime,
        source_identity=_identity("source"),
        data_identity=_identity("development-data", "1"),
        environment_identity=_identity("cpu-environment", "2"),
    )


def _success(tmp_path: Path, *, epochs: int = 1) -> Path:
    writer = _writer(tmp_path)
    with writer:
        for number in range(1, epochs + 1):
            writer.write_epoch_record(_epoch(number))
            writer.write_checkpoint(_states(), role="last", loadability_probe=lambda payload: True)
        writer.complete()
    return tmp_path / "evidence"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict, *, pretty: bool = False) -> None:
    if pretty:
        path.write_bytes(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2).encode("utf-8"))
    else:
        path.write_bytes(_canonical(value))


def _mutate_json(path: Path, mutation) -> None:
    value = _read_json(path)
    mutation(value)
    _write_json(path, value)


def _checkpoint_metadata() -> dict:
    training, runtime = _bindings()
    return {
        "evidence_contract_id": evidence.EVIDENCE_CONTRACT_ID,
        "training_contract_id": evidence.TRAINING_CONTRACT_ID,
        "runtime_plan_id": evidence.RUNTIME_PLAN_ID,
        "mode": "synthetic",
        "run_id": "checkpoint-run",
        "nonce": "checkpoint-nonce",
        "epoch": 1,
        "global_optimizer_step": 1,
        "role": "last",
        "predecessor_evidence_sha256": "a" * 64,
        "source_identity": _identity("source"),
        "data_identity": _identity("development-data", "1"),
        "environment_identity": _identity("cpu-environment", "2"),
        "training_contract_sha256": training["canonical_sha256"],
        "runtime_plan_sha256": runtime["canonical_sha256"],
        "evaluator_id": "visdrone_official_primary_evaluator_v1",
    }


def test_contract_raw_and_canonical_identity_are_frozen():
    raw = CONFIG.read_bytes()
    contract = evidence.load_training_evidence_contract(str(ROOT))
    canonical = evidence.canonical_training_evidence_contract_bytes(contract)
    assert len(raw) == evidence.EVIDENCE_CONTRACT_RAW_SIZE_BYTES == 7139
    assert hashlib.sha256(raw).hexdigest() == evidence.EVIDENCE_CONTRACT_RAW_SHA256 == "4d6bad4afbede236f1169796aa640f9d82bdcf91d06b799d39c183b20c230c9e"
    assert len(canonical) == evidence.EVIDENCE_CONTRACT_CANONICAL_SIZE_BYTES == 6069
    assert hashlib.sha256(canonical).hexdigest() == evidence.EVIDENCE_CONTRACT_CANONICAL_SHA256 == "3184707c6cfa115477d007ac0be5eb142a054f78e6309a079ad1e9fab0db9bd6"
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\x00" not in raw and b"\r" not in raw
    assert not canonical.endswith(b"\n")


def test_contract_schema_and_owner_registry_are_closed():
    contract = evidence.load_training_evidence_contract(str(ROOT))
    assert set(contract) == set(evidence._FROZEN_CONTRACT)
    evidence._validate_owner_registry()
    leaves = set(evidence._FROZEN_LEAVES)
    owners = [set(value) for value in evidence._SCALAR_OWNER_REGISTRY.values()]
    assert sum(map(len, owners)) == len(leaves)
    assert set.union(*owners) == leaves
    assert all(not left & right for index, left in enumerate(owners) for right in owners[index + 1 :])


@pytest.mark.parametrize(
    "raw",
    [
        b"\xef\xbb\xbf{}\n",
        b'{"x":"bad\x00"}\n',
        b'{"x":1}\r\n',
        b'{"x":1,"x":2}\n',
        b"{}",
        b"{}\n\n",
        b"[1]\n",
    ],
)
def test_portable_parser_rejects_duplicate_bom_nul_cr_lf_and_nonobject(raw: bytes):
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence._parse_json(raw, "test", trailing_lf=True)


@pytest.mark.parametrize(
    "pointer",
    [
        ("training_contract", "id"),
        ("runtime_plan", "id"),
        ("execution_policy", "formal_run_count"),
        ("roots", "evidence_root"),
        ("files", "prepared"),
        ("terminal_policy", "classifier_states"),
        ("checkpoint_policy", "required_state_order"),
        ("readiness", "training_ready"),
    ],
)
def test_contract_scalar_and_container_mutations_fail_closed(pointer):
    contract = evidence.load_training_evidence_contract(str(ROOT))
    candidate = copy.deepcopy(contract)
    target = candidate
    for token in pointer[:-1]:
        target = target[token]
    original = target[pointer[-1]]
    target[pointer[-1]] = [] if isinstance(original, list) else ("changed" if isinstance(original, str) else 99)
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_evidence_contract(candidate)


def test_contract_authority_binding_and_detachment():
    training, runtime = _bindings()
    contract = evidence.load_training_evidence_contract(str(ROOT))
    validated = evidence.validate_training_evidence_contract(contract, training, runtime)
    validated["runtime_plan"]["id"] = "changed"
    assert evidence.load_training_evidence_contract(str(ROOT))["runtime_plan"]["id"] == evidence.RUNTIME_PLAN_ID
    bad_runtime = copy.deepcopy(runtime)
    bad_runtime["vendor_runtime_binding"]["total_size_bytes"] += 1
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_evidence_contract(contract, training, bad_runtime)
    bad_training = copy.deepcopy(training)
    bad_training["canonical_sha256"] = "f" * 64
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_evidence_contract(contract, bad_training, runtime)


def test_contract_binding_is_detached_and_does_not_create_files_or_change_environment(tmp_path: Path):
    before_environment = dict(os.environ)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    result = evidence.training_evidence_contract_binding(str(ROOT))
    result["contract"]["schema_version"] = 99
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert dict(os.environ) == before_environment
    assert before == after == []
    assert result["canonical_sha256"] == evidence.EVIDENCE_CONTRACT_CANONICAL_SHA256


def test_synthetic_writer_orders_records_checkpoints_and_terminalizes_once(tmp_path: Path):
    root = _success(tmp_path, epochs=2)
    result = evidence.validate_training_evidence(str(root))
    assert result["status"] == "SYNTHETIC_TERMINAL_COMPLETE"
    assert result["mode"] == "synthetic"
    assert result["epoch_count"] == 2
    assert result["checkpoint_count"] == 2
    assert evidence.classify_training_evidence(str(root)) == "SYNTHETIC_TERMINAL_COMPLETE"
    assert not list(root.glob(".*.tmp"))
    assert sorted(path.name for path in (root / "checkpoints").iterdir()) == [
        "checkpoint-last-0001.ckpt",
        "checkpoint-last-0002.ckpt",
    ]


def test_writer_rejects_duplicate_and_out_of_order_epochs(tmp_path: Path):
    writer = _writer(tmp_path)
    with writer:
        writer.write_epoch_record(_epoch(1))
        with pytest.raises(evidence.TrainingEvidenceError):
            writer.write_epoch_record(_epoch(1, 2))
        with pytest.raises(evidence.TrainingEvidenceError):
            writer.write_epoch_record(_epoch(3, 3))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss", float("nan")),
        ("gradient_norm", float("inf")),
        ("amp_scale", 0.0),
        ("amp_skipped_steps", 1),
        ("amp_overflow_events", 1),
        ("nonfinite_loss_count", 1),
        ("nonfinite_gradient_count", 1),
    ],
)
def test_writer_rejects_nonfinite_and_forbidden_epoch_observations(tmp_path: Path, field, value):
    writer = _writer(tmp_path)
    with writer:
        record = _epoch(1)
        record[field] = value
        with pytest.raises(evidence.TrainingEvidenceError):
            writer.write_epoch_record(record)


def test_writer_rejects_evaluator_checkpoint_and_identity_input_drift(tmp_path: Path):
    writer = _writer(tmp_path)
    with writer:
        record = _epoch(1)
        record["evaluator_id"] = "other"
        with pytest.raises(evidence.TrainingEvidenceError):
            writer.write_epoch_record(record)
        writer.write_epoch_record(_epoch(1))
        with pytest.raises(evidence.TrainingEvidenceError):
            writer.write_checkpoint(_states(), role="last", loadability_probe=lambda _: True, epoch=2)


def test_classifier_absent_in_progress_and_unknown_are_conservative(tmp_path: Path):
    absent = tmp_path / "absent"
    assert evidence.classify_training_evidence(str(absent)) == "ABSENT"
    writer = _writer(tmp_path)
    root = tmp_path / "evidence"
    assert evidence.classify_training_evidence(str(root)) == "IN_PROGRESS"
    writer.close()
    assert evidence.classify_training_evidence(str(root)) == "IN_PROGRESS"
    (root / "unexpected.json").write_bytes(b"{}")
    assert evidence.classify_training_evidence(str(root)) == "UNKNOWN"


def test_permanent_failure_is_immutable_and_terminal_failed(tmp_path: Path):
    writer = _writer(tmp_path)
    root = tmp_path / "evidence"
    with writer:
        result = writer.fail(RuntimeError("synthetic failure"))
        assert result["status"] == "TERMINAL_FAILED"
        with pytest.raises(evidence.TrainingEvidenceError):
            writer.fail(RuntimeError("retry"))
        with pytest.raises(evidence.TrainingEvidenceError):
            writer.complete()
    assert evidence.classify_training_evidence(str(root)) == "TERMINAL_FAILED"
    assert _read_json(root / "failure.json")["failure_type"] == "PERMANENT_FAIL"


def test_real_terminal_success_requires_120_epochs(tmp_path: Path):
    writer = _writer(tmp_path, mode="real")
    with writer:
        writer.write_epoch_record(_epoch(1))
        writer.write_checkpoint(_states(), role="last", loadability_probe=lambda _: True)
        with pytest.raises(evidence.TrainingEvidenceError):
            writer.complete()
    assert evidence.classify_training_evidence(str(tmp_path / "evidence")) == "IN_PROGRESS"


def test_validator_rejects_repacked_records_and_terminal_files(tmp_path: Path):
    root = _success(tmp_path)
    prepared = _read_json(root / "prepared.json")
    _write_json(root / "prepared.json", prepared, pretty=True)
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_evidence(str(root))

    root = _success(tmp_path / "second")
    completion = _read_json(root / "completion.json")
    _write_json(root / "completion.json", completion, pretty=True)
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_evidence(str(root))


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo", "missing"])
def test_validator_rejects_unstable_prepared_objects(tmp_path: Path, kind: str):
    root = _success(tmp_path)
    path = root / "prepared.json"
    external = tmp_path / "external"
    if kind == "symlink":
        path.unlink()
        path.symlink_to(external)
    elif kind == "hardlink":
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.hardlink_to(external)
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        path.unlink()
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_evidence(str(root))


def test_validator_rejects_socket_extra_file_and_root_symlink(tmp_path: Path):
    root = _success(tmp_path)
    extra = root / "unexpected.json"
    extra.write_bytes(b"{}")
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_evidence(str(root))

    root = _success(tmp_path / "alias")
    alias = tmp_path / "root-alias"
    alias.symlink_to(root, target_is_directory=True)
    assert evidence.classify_training_evidence(str(alias)) == "UNKNOWN"


def test_readonly_validation_does_not_change_evidence_identity(tmp_path: Path):
    root = _success(tmp_path)
    before = {
        path.relative_to(root).as_posix(): (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    evidence.validate_training_evidence(str(root))
    evidence.classify_training_evidence(str(root))
    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_atomic_checkpoint_publishes_exactly_once_and_is_loadable(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(mode=0o700)
    result = evidence.write_atomic_checkpoint(
        str(checkpoint_root),
        _states(),
        _checkpoint_metadata(),
        loadability_probe=lambda payload: {"schema_version": 1, "loadable": True},
    )
    path = checkpoint_root / "checkpoint-last-0001.ckpt"
    assert result["relative_path"] == "checkpoints/checkpoint-last-0001.ckpt"
    assert result["checkpoint_size_bytes"] == path.stat().st_size
    assert result["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    validated = evidence.validate_training_checkpoint(str(path))
    assert validated["required_state_order"] == list(evidence.CHECKPOINT_REQUIRED_STATES)
    assert not list(checkpoint_root.glob(".*.tmp"))
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.write_atomic_checkpoint(str(checkpoint_root), _states(), _checkpoint_metadata(), loadability_probe=lambda _: True)


def test_atomic_checkpoint_rejects_missing_parent_bad_states_and_bad_loader(tmp_path: Path):
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.write_atomic_checkpoint(str(tmp_path / "missing"), _states(), _checkpoint_metadata(), loadability_probe=lambda _: True)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(mode=0o700)
    bad_states = _states()
    del bad_states[evidence.CHECKPOINT_REQUIRED_STATES[0]]
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.write_atomic_checkpoint(str(checkpoint_root), bad_states, _checkpoint_metadata(), loadability_probe=lambda _: True)
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.write_atomic_checkpoint(str(checkpoint_root), _states(), _checkpoint_metadata(), loadability_probe=lambda _: False)
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.write_atomic_checkpoint(str(checkpoint_root), _states(), _checkpoint_metadata(), loadability_probe=lambda _: {"schema_version": 2, "loadable": True})


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "renamed", "bad_hex", "bad_sha"])
def test_checkpoint_state_schema_mutations_fail_closed(tmp_path: Path, mutation: str):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(mode=0o700)
    result = evidence.write_atomic_checkpoint(str(checkpoint_root), _states(), _checkpoint_metadata(), loadability_probe=lambda _: True)
    candidate = evidence.validate_training_checkpoint(str(checkpoint_root / "checkpoint-last-0001.ckpt"))
    states = candidate["states"]
    if mutation == "missing":
        states.pop()
    elif mutation == "extra":
        states.append(copy.deepcopy(states[-1]))
    elif mutation == "reordered":
        states[0], states[1] = states[1], states[0]
    elif mutation == "renamed":
        states[0]["name"] = "other"
    elif mutation == "bad_hex":
        states[0]["payload_hex"] = "not-hex"
    else:
        states[0]["sha256"] = "f" * 64
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_checkpoint(candidate)


def test_checkpoint_file_mutation_and_wrong_objects_are_rejected(tmp_path: Path):
    root = _success(tmp_path)
    checkpoint = root / "checkpoints/checkpoint-last-0001.ckpt"
    value = _read_json(checkpoint)
    value["states"][0]["payload_hex"] = "00"
    _write_json(checkpoint, value)
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_evidence(str(root))

    root = _success(tmp_path / "objects")
    checkpoint = root / "checkpoints/checkpoint-last-0001.ckpt"
    external = tmp_path / "objects-external"
    external.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoint.hardlink_to(external)
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.validate_training_evidence(str(root))


def test_checkpoint_loader_and_input_are_detached(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(mode=0o700)
    original = _states()
    observed = []

    def probe(payload: bytes):
        observed.append(bytes(payload))
        return True

    evidence.write_atomic_checkpoint(str(checkpoint_root), original, _checkpoint_metadata(), loadability_probe=probe)
    original[evidence.CHECKPOINT_REQUIRED_STATES[0]] = b"changed"
    assert observed and observed[0]
    assert evidence.validate_training_checkpoint(str(checkpoint_root / "checkpoint-last-0001.ckpt"))["states"][0]["name"] == "raw_model"


def test_writer_rejects_symlinked_checkpoint_root_and_noncanonical_paths(tmp_path: Path):
    real = tmp_path / "real-checkpoints"
    real.mkdir(mode=0o700)
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.write_atomic_checkpoint(str(alias), _states(), _checkpoint_metadata(), loadability_probe=lambda _: True)
    with pytest.raises(evidence.TrainingEvidenceError):
        evidence.TrainingEvidenceWriter(
            str(tmp_path / "parent" / ".." / "evidence"),
            run_id="run",
            nonce="nonce",
            mode="synthetic",
            training_contract_binding=_bindings()[0],
            runtime_plan_binding=_bindings()[1],
            source_identity=_identity("source"),
            data_identity=_identity("data", "1"),
            environment_identity=_identity("environment", "2"),
        )

def test_static_isolation_and_public_surface():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imported.add(node.module or "")
    forbidden = ("torch", "numpy", "PIL", "cv2", "subprocess", "socket", "urllib", "requests", "git")
    assert not any(name == item or name.startswith(item + ".") for name in imported for item in forbidden)
    assert "sparse_rtdetr.baseline.training_contract" in imported
    assert "sparse_rtdetr.baseline.training_runtime" in imported
    assert set(evidence.__all__) == {
        "load_training_evidence_contract",
        "validate_training_evidence_contract",
        "canonical_training_evidence_contract_bytes",
        "training_evidence_contract_binding",
        "TrainingEvidenceWriter",
        "write_atomic_checkpoint",
        "validate_training_evidence",
        "validate_training_checkpoint",
        "classify_training_evidence",
        "TrainingEvidenceError",
    }
    source = SOURCE.read_text(encoding="utf-8")
    assert ("/" + "media/") not in source and "CUDA_VISIBLE_DEVICES" not in source


def test_production_t5b_target_is_not_created():
    assert not (ROOT / evidence.EVIDENCE_ROOT_RELATIVE_PATH).exists()
    assert not (ROOT / evidence.CHECKPOINT_ROOT_RELATIVE_PATH).exists()
