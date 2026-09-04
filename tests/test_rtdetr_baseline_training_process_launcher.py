"""CPU mutation coverage for the synthetic T5D outer process boundary."""

from __future__ import annotations

import ast
import copy
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_entry as entry
from sparse_rtdetr.baseline import training_process_launcher as launcher
from test_rtdetr_baseline_training_entry import _components


ROOT = Path(__file__).resolve().parents[1]
_COUNTER = 1000


def _descriptor(tmp_path: Path, suffix: str = "run") -> dict[str, object]:
    global _COUNTER
    _COUNTER += 1
    entry_root = tmp_path / f"entry-{suffix}-{_COUNTER}"
    launch_root = tmp_path / f"launch-{suffix}-{_COUNTER}"
    inner = entry.build_synthetic_entry_descriptor(
        ROOT,
        run_id=f"t5d-launch-{_COUNTER}",
        nonce=f"{_COUNTER:032x}",
        evidence_root=entry_root,
        python_executable=Path(sys.executable).resolve(),
    )
    return launcher.build_synthetic_launch_descriptor(inner, launch_root)


def _success_runner(calls: list[object]):
    components, _ = _components()

    def runner(argv, cwd, environment, entry_descriptor):
        calls.append((list(argv), cwd, copy.deepcopy(environment), copy.deepcopy(entry_descriptor)))
        result = entry.run_synthetic_entry(entry_descriptor, components, Path(entry_descriptor["evidence_root"]))
        return {"return_code": 0, "stdout": b"child\x00stdout\xff", "stderr": b"child\x80stderr", "entry_result": result}

    return runner


def _publish_in_progress(descriptor: dict[str, object]) -> None:
    parent, _, _ = launcher._claim_launch_receipt(descriptor)
    try:
        launcher._create_root(parent, Path(descriptor["evidence_root"]).name)
    finally:
        parent.close()
    lease = launcher._open_root_lease(descriptor["evidence_root"])
    try:
        invocation = {"schema_version": 1, "status": "ACCEPTED", "descriptor_sha256": descriptor["aggregate_sha256"], "descriptor": copy.deepcopy(descriptor)}
        launcher._publish_at(lease, "invocation.json", launcher._canonical(invocation))
    finally:
        lease.close()


def test_launcher_descriptor_has_bound_root_and_detaches(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path)
    assert descriptor["mode"] == "synthetic"
    assert descriptor["runner_contract"]["max_calls"] == 1
    assert descriptor["state_machine"]["trace"][-1]["state"] == "PREPARED"
    assert descriptor["evidence_root"].startswith(str(tmp_path))
    detached = launcher.validate_launch_descriptor(descriptor)
    detached["entry_descriptor"]["mode"] = "changed"
    assert descriptor["entry_descriptor"]["mode"] == "synthetic"


def test_launcher_process_config_mode_is_cross_file_bound(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "config-mode")
    mode = descriptor["config_identity"]["mode"]
    assert mode == descriptor["entry_descriptor"]["source_bindings"]["process_config"]["mode"]
    mutated = copy.deepcopy(descriptor)
    mutated["config_identity"]["mode"] = 0o664 if mode == 0o644 else 0o644
    body = {key: value for key, value in mutated.items() if key != "aggregate_sha256"}
    mutated["aggregate_sha256"] = launcher._digest(body)
    with pytest.raises(launcher.TrainingProcessLauncherError, match="process config identity"):
        launcher.validate_launch_descriptor(mutated)


def test_fake_runner_launch_is_durable_and_exactly_once(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path)
    calls: list[object] = []
    result = launcher.run_synthetic_launch(descriptor, _success_runner(calls), descriptor["evidence_root"])
    assert result["status"] == "SYNTHETIC_TERMINAL_COMPLETE"
    assert result["classification"] == "SYNTHETIC_TERMINAL_COMPLETE"
    assert result["return_code"] == 0
    assert result["stdout_size_bytes"] == len(b"child\x00stdout\xff")
    assert result["stderr_size_bytes"] == len(b"child\x80stderr")
    assert len(calls) == 1
    assert launcher.classify_launch(descriptor["evidence_root"]) == "SYNTHETIC_TERMINAL_COMPLETE"
    assert sorted(path.name for path in Path(descriptor["evidence_root"]).iterdir()) == ["artifact_inventory.json", "completion.json", "invocation.json", "result.json", "stderr.bin", "stdout.bin"]
    with pytest.raises(launcher.TrainingProcessLauncherError):
        launcher.run_synthetic_launch(descriptor, _success_runner([]), descriptor["evidence_root"])
    assert len(calls) == 1


def test_same_descriptor_replay_after_module_reload_and_different_root_is_refused(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path)
    calls: list[object] = []
    launcher.run_synthetic_launch(descriptor, _success_runner(calls), descriptor["evidence_root"])
    importlib.reload(launcher)
    with pytest.raises(launcher.TrainingProcessLauncherError):
        launcher.run_synthetic_launch(descriptor, _success_runner(calls), descriptor["evidence_root"])
    assert len(calls) == 1
    fresh = _descriptor(tmp_path, "different")
    with pytest.raises(launcher.TrainingProcessLauncherError):
        launcher.run_synthetic_launch(descriptor, _success_runner(calls), fresh["evidence_root"])
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["nonzero", "exception", "timeout", "returned_generator", "malformed", "negative", "boolean", "too_large"])
def test_runner_failure_classes_are_terminal_and_fail_closed(tmp_path: Path, kind: str) -> None:
    descriptor = _descriptor(tmp_path, kind)
    calls: list[int] = []

    def runner(argv, cwd, environment, entry_descriptor):
        calls.append(1)
        if kind == "nonzero":
            return {"return_code": 7, "stdout": b"child-out", "stderr": b"child-error", "entry_result": None}
        if kind == "exception":
            raise RuntimeError("fake runner failure")
        if kind == "timeout":
            raise TimeoutError("fake timeout")
        if kind == "returned_generator":
            return (item for item in ())
        if kind == "malformed":
            return {"return_code": 0, "stdout": b"", "stderr": b""}
        value = -1 if kind == "negative" else True if kind == "boolean" else 256
        return {"return_code": value, "stdout": b"", "stderr": b"", "entry_result": None}

    result = launcher.run_synthetic_launch(descriptor, runner, descriptor["evidence_root"])
    assert len(calls) == 1
    assert result["status"] == "TERMINAL_FAILED"
    assert launcher.classify_launch(descriptor["evidence_root"]) == "TERMINAL_FAILED"
    expected = {
        "nonzero": "RUNNER_NONZERO_RETURN_CODE",
        "exception": "RUNNER_EXCEPTION",
        "timeout": "RUNNER_TIMEOUT",
        "returned_generator": "RUNNER_ASYNC_OR_GENERATOR_RESULT",
        "malformed": "RUNNER_SCHEMA_OR_ENTRY_FAILURE",
        "negative": "RUNNER_SCHEMA_OR_ENTRY_FAILURE",
        "boolean": "RUNNER_SCHEMA_OR_ENTRY_FAILURE",
        "too_large": "RUNNER_SCHEMA_OR_ENTRY_FAILURE",
    }
    assert result["failure_class"] == expected[kind]
    if kind == "nonzero":
        assert result["return_code"] == 7
    else:
        assert result["return_code"] == launcher.FAILURE_RETURN_CODES[expected[kind]]


def test_runner_signature_and_input_mutation_are_rejected(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "signature")

    def wrong(argv):
        return {}

    with pytest.raises(launcher.TrainingProcessLauncherError):
        launcher.run_synthetic_launch(descriptor, wrong, descriptor["evidence_root"])

    descriptor = _descriptor(tmp_path, "mutated")

    def mutate(argv, cwd, environment, entry_descriptor):
        argv.append("--mutated")
        return {"return_code": 1, "stdout": b"", "stderr": b"", "entry_result": None}

    result = launcher.run_synthetic_launch(descriptor, mutate, descriptor["evidence_root"])
    assert result["status"] == "TERMINAL_FAILED"
    assert result["failure_class"] == "RUNNER_MUTATED_INVOCATION"

    descriptor = _descriptor(tmp_path, "non-json-mutation")

    def mutate_with_object(argv, cwd, environment, entry_descriptor):
        environment["injected"] = object()
        return {"return_code": 1, "stdout": b"", "stderr": b"", "entry_result": None}

    result = launcher.run_synthetic_launch(descriptor, mutate_with_object, descriptor["evidence_root"])
    assert result["failure_class"] == "RUNNER_MUTATED_INVOCATION"
    assert result["return_code"] == launcher.FAILURE_RETURN_CODES["RUNNER_MUTATED_INVOCATION"]

    descriptor = _descriptor(tmp_path, "generator")

    def generator(argv, cwd, environment, entry_descriptor):
        yield None

    with pytest.raises(launcher.TrainingProcessLauncherError):
        launcher.run_synthetic_launch(descriptor, generator, descriptor["evidence_root"])


@pytest.mark.parametrize("field", ["argv", "environment", "mode", "state_machine", "runner_contract", "evidence_root", "receipt_path"])
def test_launch_descriptor_mutations_fail_closed(tmp_path: Path, field: str) -> None:
    descriptor = _descriptor(tmp_path, field)
    mutated = copy.deepcopy(descriptor)
    if field == "argv":
        mutated["argv"] = [*mutated["argv"], "--extra"]
    elif field == "environment":
        mutated["environment"]["EXTRA"] = "1"
    elif field == "mode":
        mutated["mode"] = "real"
    elif field == "state_machine":
        mutated["state_machine"]["trace"][1]["state"] = "ACCEPTED"
    elif field == "runner_contract":
        mutated["runner_contract"]["max_calls"] = 2
    elif field == "evidence_root":
        mutated["evidence_root"] = str(tmp_path / "other-root")
    else:
        mutated["receipt_path"] = str(tmp_path / "wrong-receipt")
    with pytest.raises(launcher.TrainingProcessLauncherError):
        launcher.validate_launch_descriptor(mutated)


def test_classifier_truth_table_and_real_success_refusal(tmp_path: Path) -> None:
    assert launcher.classify_launch(tmp_path / "absent") == "ABSENT"
    descriptor = _descriptor(tmp_path, "progress")
    _publish_in_progress(descriptor)
    assert launcher.classify_launch(descriptor["evidence_root"]) == "IN_PROGRESS"

    completed = _descriptor(tmp_path, "corrupt")
    launcher.run_synthetic_launch(completed, _success_runner([]), completed["evidence_root"])
    result_path = Path(completed["evidence_root"]) / "result.json"
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    raw["status"] = "TERMINAL_COMPLETE"
    result_path.write_bytes(launcher._canonical(raw))
    assert launcher.classify_launch(completed["evidence_root"]) == "TERMINAL_FAILED"

    real_refusal = tmp_path / "real-refusal"
    real_refusal.mkdir(mode=0o700)
    (real_refusal / "result.json").write_bytes(launcher._canonical({"status": "TERMINAL_COMPLETE"}))
    assert launcher.classify_launch(real_refusal) == "TERMINAL_FAILED"


@pytest.mark.parametrize("metadata", ["mode", "hardlink", "directory", "fifo"])
def test_classifier_rejects_anchor_object_and_metadata_drift(tmp_path: Path, metadata: str) -> None:
    descriptor = _descriptor(tmp_path, metadata)
    launcher.run_synthetic_launch(descriptor, _success_runner([]), descriptor["evidence_root"])
    root = Path(descriptor["evidence_root"])
    stdout = root / "stdout.bin"
    if metadata == "mode":
        stdout.chmod(0o640)
    elif metadata == "hardlink":
        backup = root / "stdout-copy.bin"
        stdout.rename(backup)
        stdout.hardlink_to(backup)
    elif metadata == "directory":
        stdout.unlink()
        stdout.mkdir(mode=0o700)
    else:
        stdout.unlink()
        os.mkfifo(stdout, 0o600)
    assert launcher.classify_launch(root) == "UNKNOWN"


def test_coordinated_json_repack_without_binary_change_is_rejected(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "coordinated-repack")
    launcher.run_synthetic_launch(descriptor, _success_runner([]), descriptor["evidence_root"])
    root = Path(descriptor["evidence_root"])
    result_path = root / "result.json"
    inventory_path = root / "artifact_inventory.json"
    completion_path = root / "completion.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    forged_stdout = b"forged stdout"
    result["stdout_size_bytes"] = len(forged_stdout)
    result["stdout_sha256"] = launcher._sha(forged_stdout)
    result_body = {key: value for key, value in result.items() if key != "aggregate_result_sha256"}
    result["aggregate_result_sha256"] = launcher._digest(result_body)
    for row in inventory["files"]:
        if row["relative_path"] == "stdout.bin":
            row["size_bytes"] = len(forged_stdout)
            row["sha256"] = launcher._sha(forged_stdout)
    inventory["canonical_inventory_sha256"] = launcher._digest(inventory["files"])
    result_raw = launcher._canonical(result)
    inventory_raw = launcher._canonical(inventory)
    completion["result_sha256"] = launcher._sha(result_raw)
    completion["inventory_sha256"] = launcher._sha(inventory_raw)
    completion["stdout_size_bytes"] = len(forged_stdout)
    completion["stdout_sha256"] = launcher._sha(forged_stdout)
    result_path.write_bytes(result_raw)
    inventory_path.write_bytes(inventory_raw)
    completion_path.write_bytes(launcher._canonical(completion))
    assert launcher.classify_launch(root) == "UNKNOWN"


def test_published_root_final_name_replacement_is_rejected(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "root-replacement")
    launcher.run_synthetic_launch(descriptor, _success_runner([]), descriptor["evidence_root"])
    root = Path(descriptor["evidence_root"])
    moved = root.with_name(root.name + "-moved")
    root.rename(moved)
    root.symlink_to(moved, target_is_directory=True)
    assert launcher.classify_launch(root) == "UNKNOWN"


def test_binary_anchor_mutation_and_inventory_repack_are_rejected(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "anchors")
    launcher.run_synthetic_launch(descriptor, _success_runner([]), descriptor["evidence_root"])
    root = Path(descriptor["evidence_root"])
    (root / "stdout.bin").write_bytes(b"different")
    assert launcher.classify_launch(root) == "UNKNOWN"


def test_descriptor_expected_binding_rejects_independently_repacked_result(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "expected-result")
    calls: list[object] = []
    launcher.run_synthetic_launch(descriptor, _success_runner(calls), descriptor["evidence_root"])
    published = launcher._validate_published_root(descriptor["evidence_root"], expected_descriptor=descriptor)
    forged = copy.deepcopy(published["result"])
    forged["run_id"] = "forged-run"
    forged_body = {key: value for key, value in forged.items() if key != "aggregate_result_sha256"}
    forged["aggregate_result_sha256"] = launcher._digest(forged_body)
    with pytest.raises(launcher.TrainingProcessLauncherError):
        launcher.validate_launch_result(forged, expected_descriptor=descriptor)


def test_bound_root_mismatch_is_rejected_before_runner_call(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "root-mismatch")
    calls: list[object] = []
    with pytest.raises(launcher.TrainingProcessLauncherError):
        launcher.run_synthetic_launch(descriptor, _success_runner(calls), tmp_path / "different-root")
    assert calls == []


def test_receipt_without_root_is_not_absent(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "orphan-receipt")
    launcher._claim_launch_receipt(descriptor)[0].close()
    assert not Path(descriptor["evidence_root"]).exists()
    assert launcher.classify_launch(descriptor["evidence_root"]) == "UNKNOWN"


def test_classifier_retries_interrupted_directory_enumeration(monkeypatch, tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path, "eintr-listdir")
    launcher.run_synthetic_launch(descriptor, _success_runner([]), descriptor["evidence_root"])
    original = launcher.os.listdir
    calls = 0

    def flaky(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError()
        return original(fd)

    monkeypatch.setattr(launcher.os, "listdir", flaky)
    assert launcher.classify_launch(descriptor["evidence_root"]) == "SYNTHETIC_TERMINAL_COMPLETE"
    assert calls >= 2


def test_launcher_import_surface_and_parent_torch_sentinel() -> None:
    assert set(launcher.__all__) == {"TrainingProcessLauncherError", "build_synthetic_launch_descriptor", "validate_launch_descriptor", "run_synthetic_launch", "validate_launch_result", "classify_launch"}
    source = (ROOT / launcher.LAUNCHER_MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    tree = ast.parse(source)
    imported = {alias.name.split(".", 1)[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update((node.module or "").split(".", 1)[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module != "__future__")
    assert not imported.intersection({"torch", "subprocess", "multiprocessing", "socket", "requests", "urllib", "vendor"})
    sentinel = object()
    sys.modules["torch"] = sentinel
    try:
        assert sys.modules["torch"] is sentinel
        assert launcher.LAUNCHER_MODE == "synthetic"
    finally:
        sys.modules.pop("torch", None)


def test_future_production_targets_remain_absent() -> None:
    for relative in entry.FUTURE_TARGETS.values():
        assert not (ROOT / relative).exists()
