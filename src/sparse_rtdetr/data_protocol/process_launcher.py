"""Process-evidence launcher for a future authorized converter run.

The launcher owns only its evidence directory. The child converter owns the
production output directory, which the launcher never creates or resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .cli import _split_arg
from .schema import ProtocolContractError, canonical_json_bytes


class ProcessLauncherContractError(ProtocolContractError):
    """Raised when launcher ownership or identity contracts fail."""


class EntryValidationError(ValueError):
    """Raised for a child output that cannot be accepted as completed."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


PROCESS_SCHEMA_VERSION = 2
CHILD_MODULE = "sparse_rtdetr.data_protocol.cli"
PROCESS_INVENTORY_EXCLUDED = frozenset({"process_inventory.json", "process_completion.json"})
ENTRY_INVENTORY_EXCLUDED = frozenset({"artifact_inventory.json", "completion.json"})
SIGNAL_GRACE_SECONDS = 5.0
_SIGNALS = tuple(
    value for value in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT)
    if value is not None
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path: Path, *, relative_path: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        return {"present": False, "relative_path": relative_path}
    return {
        "present": True,
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ProcessLauncherContractError(f"process evidence already exists: {path.name}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _safe_atomic_write(path: Path, payload: bytes) -> bool:
    try:
        _atomic_write(path, payload)
    except BaseException:
        return False
    return True


def _inventory(evidence_dir: Path, excluded: frozenset[str]) -> dict[str, object]:
    rows = []
    for path in sorted(evidence_dir.iterdir(), key=lambda item: item.name):
        if path.name in excluded or path.name.startswith("."):
            continue
        if path.is_symlink() or not path.is_file():
            raise ProcessLauncherContractError(f"invalid process evidence entry: {path.name}")
        rows.append({
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "excluded_from_inventory": sorted(excluded),
        "artifacts": rows,
        "canonical_inventory_sha256": _sha256_bytes(canonical_json_bytes(rows)),
    }


def _module_source(source_root: Path) -> Path:
    module_path = source_root / "sparse_rtdetr" / "data_protocol" / "cli.py"
    if module_path.is_symlink() or not module_path.is_file():
        raise ProcessLauncherContractError("child CLI source is missing or not regular")
    return module_path


def contract_check(source_root: Path | None = None) -> dict[str, object]:
    """Run a data-free launcher/child-source contract check."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProcessLauncherContractError("contract-check requires CUDA_VISIBLE_DEVICES=''")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ProcessLauncherContractError("contract-check requires PYTHONNOUSERSITE=1")
    module = __import__(CHILD_MODULE, fromlist=["contract_check"])
    module_path = Path(module.__file__).resolve()
    if source_root is not None:
        source_root = source_root.resolve()
        expected = _module_source(source_root).resolve()
        if module_path != expected:
            raise ProcessLauncherContractError("imported child module is outside source-root")
    child_result = module.contract_check()
    if child_result.get("status") != "PASS":
        raise ProcessLauncherContractError("child contract-check did not pass")
    return {
        "status": "PASS",
        "module": CHILD_MODULE,
        "module_file": str(module_path),
        "source_root": str(source_root) if source_root is not None else None,
        "cuda_visible_devices": "",
        "python_no_user_site": "1",
        "real_data_accessed": False,
        "output_directory_created": False,
        "process_evidence_created": False,
        "torch_imported": False,
        "model_constructed": False,
        "dataset_or_dataloader_constructed": False,
        "network_requested": False,
        "dataset_test_accessed_by_this_process": False,
    }


def _validate_run_args(
    data_root: Path,
    output_dir: Path,
    protocol_config: Path,
    splits: tuple[str, ...],
    process_evidence_dir: Path,
    child_python: Path,
    source_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProcessLauncherContractError("run requires CUDA_VISIBLE_DEVICES=''")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ProcessLauncherContractError("run requires PYTHONNOUSERSITE=1")
    if set(splits) != {"train", "val"} or len(splits) != 2:
        raise ProcessLauncherContractError("run requires exactly train,val")
    if not all(path.is_absolute() for path in (
        data_root, output_dir, protocol_config, process_evidence_dir, child_python, source_root
    )):
        raise ProcessLauncherContractError("launcher paths must be absolute")
    if any(part == ".." for part in output_dir.parts + process_evidence_dir.parts):
        raise ProcessLauncherContractError("launcher paths may not contain ..")
    if output_dir.exists() or output_dir.is_symlink():
        raise ProcessLauncherContractError("child output directory already exists")
    if process_evidence_dir.exists() or process_evidence_dir.is_symlink():
        raise ProcessLauncherContractError("process evidence directory already exists")
    if not child_python.is_file() or not os.access(child_python, os.X_OK):
        raise ProcessLauncherContractError("child Python must be an executable regular file")
    if not protocol_config.is_file() or protocol_config.is_symlink():
        raise ProcessLauncherContractError("protocol config must be a regular file")
    _module_source(source_root)
    return output_dir, process_evidence_dir, child_python, protocol_config, source_root


def _child_argv(
    child_python: Path,
    data_root: Path,
    output_dir: Path,
    protocol_config: Path,
    splits: tuple[str, ...],
) -> list[str]:
    return [
        str(child_python), "-m", CHILD_MODULE, "run",
        "--data-root", str(data_root),
        "--output-dir", str(output_dir),
        "--protocol-config", str(protocol_config),
        "--split", ",".join(splits),
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIG{signum}"


def _terminate_child(child: subprocess.Popen[bytes] | None) -> None:
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=SIGNAL_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            child.wait(timeout=SIGNAL_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _run_child(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    console,
) -> tuple[int, list[dict[str, object]], bool, dict[int, object]]:
    state: dict[str, object] = {
        "child": None,
        "signal_events": [],
        "grace_deadline": None,
        "kill_sent": False,
    }

    def handle_signal(signum: int, _frame) -> None:
        event: dict[str, object] = {
            "signal_number": signum,
            "signal_name": _signal_name(signum),
            "received_at_utc": _utc_now(),
            "forwarded_to_child_process_group": False,
        }
        child = state["child"]
        if isinstance(child, subprocess.Popen) and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
                event["forwarded_to_child_process_group"] = True
                state["grace_deadline"] = time.monotonic() + SIGNAL_GRACE_SECONDS
            except OSError as error:
                event["forward_error"] = type(error).__name__
        cast_events = state["signal_events"]
        assert isinstance(cast_events, list)
        cast_events.append(event)

    child = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    state["child"] = child
    previous_handlers = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    for signum in _SIGNALS:
        signal.signal(signum, handle_signal)
    selector = selectors.DefaultSelector()
    assert child.stdout is not None
    fd = child.stdout.fileno()
    os.set_blocking(fd, False)
    selector.register(fd, selectors.EVENT_READ)
    end_of_stream = False
    try:
        while not end_of_stream or child.poll() is None:
            deadline = state["grace_deadline"]
            timeout = 0.25
            if isinstance(deadline, float):
                timeout = max(0.0, min(timeout, deadline - time.monotonic()))
            for _key, _mask in selector.select(timeout):
                while True:
                    try:
                        chunk = os.read(fd, 1024 * 1024)
                    except BlockingIOError:
                        break
                    if not chunk:
                        end_of_stream = True
                        selector.unregister(fd)
                        break
                    console.write(chunk)
                    console.flush()
            if isinstance(deadline, float) and time.monotonic() >= deadline and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                    state["kill_sent"] = True
                except OSError:
                    pass
                state["grace_deadline"] = None
        returncode = child.wait()
    except BaseException:
        _terminate_child(child)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        raise
    finally:
        selector.close()
        child.stdout.close()
    console.flush()
    os.fsync(console.fileno())
    events = state["signal_events"]
    assert isinstance(events, list)
    return returncode, events, bool(state["kill_sent"]), previous_handlers


def _json_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} is not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _input_protocol_config_ref(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        return {"present": False, "relative_path": "protocol_config.json"}
    try:
        payload = path.read_bytes()
    except OSError:
        return {"present": False, "relative_path": "protocol_config.json"}
    return {
        "present": True,
        "relative_path": "protocol_config.json",
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _validate_input_protocol_config(path: Path) -> None:
    try:
        config = _json_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ProcessLauncherContractError(
            f"input protocol config is invalid: {type(error).__name__}"
        ) from error
    split = config.get("split")
    required = {
        "schema_version": 1,
        "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V2",
        "real_conversion_outputs_generated": False,
        "production_split_manifest_generated": False,
        "confirmatory_metrics_accessed": False,
        "test_access_allowed": False,
    }
    if any(config.get(key) != value for key, value in required.items()) or not isinstance(split, dict):
        raise ProcessLauncherContractError("input protocol config is not the frozen V2 pre-run schema")
    split_required = {
        "test": "disabled",
        "seed": 20260808,
        "salt": "P3-confirmatory-v1",
        "target_images": 647,
        "selection_policy": "feasibility_first_nearest_hash_prefix_v2",
        "planner": "plan_confirmatory_split_v2",
        "selection_allowed": False,
        "metrics_access_allowed": False,
        "single_final_access_only": True,
    }
    if any(split.get(key) != value for key, value in split_required.items()):
        raise ProcessLauncherContractError("input protocol V2 split identity mismatch")


def _sha256_field(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_entry(
    output_dir: Path,
    input_protocol_config_before: dict[str, object],
    input_protocol_config_after: dict[str, object],
) -> dict[str, object]:
    completion_path = output_dir / "completion.json"
    inventory_path = output_dir / "artifact_inventory.json"
    result: dict[str, object] = {
        "entry_completion": _file_ref(completion_path, relative_path="completion.json"),
        "entry_artifact_inventory": _file_ref(inventory_path, relative_path="artifact_inventory.json"),
        "entry_partial_inventory": _file_ref(output_dir / "partial_inventory.json", relative_path="partial_inventory.json"),
        "entry_completion_status": None,
        "entry_validation_message": None,
        "entry_success_accepted": False,
    }
    if not completion_path.is_file() or completion_path.is_symlink():
        result["entry_status"] = "FAILED_ENTRY_COMPLETION_MISSING"
        result["entry_validation_message"] = "completion.json is missing"
        return result
    try:
        completion = _json_object(completion_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        result["entry_status"] = "FAILED_ENTRY_COMPLETION_INVALID"
        result["entry_validation_message"] = f"completion.json invalid: {type(error).__name__}"
        return result
    result["entry_completion_status"] = completion.get("status")
    if completion.get("status") != "COMPLETED":
        result["entry_status"] = "FAILED_ENTRY_COMPLETION_INVALID"
        result["entry_validation_message"] = "completion status is not COMPLETED"
        return result
    required_completion = {
        "schema_version": 1,
        "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V2",
        "selection_policy": "feasibility_first_nearest_hash_prefix_v2",
        "selection_allowed": False,
        "metrics_access_allowed": False,
        "single_final_access_only": True,
        "project_test_split_historically_observed": True,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
        "completion_self_hash_included": False,
        "production_conversion_executed": True,
    }
    if any(completion.get(key) != expected for key, expected in required_completion.items()):
        result["entry_status"] = "FAILED_ENTRY_COMPLETION_INVALID"
        result["entry_validation_message"] = "completion contract fields are invalid"
        return result
    if not all(_sha256_field(completion.get(key)) for key in (
        "config_sha256", "split_plan_sha256", "artifact_inventory_sha256", "input_protocol_config_sha256"
    )):
        result["entry_status"] = "FAILED_ENTRY_COMPLETION_INVALID"
        result["entry_validation_message"] = "completion binding fields are invalid"
        return result
    if (
        input_protocol_config_before.get("present") is not True
        or input_protocol_config_after.get("present") is not True
        or input_protocol_config_before != input_protocol_config_after
        or completion.get("input_protocol_config_sha256") != input_protocol_config_before.get("sha256")
    ):
        result["entry_status"] = "FAILED_ENTRY_INPUT_PROTOCOL_CONFIG_BINDING"
        result["entry_validation_message"] = "input protocol config changed or completion binding mismatched"
        return result
    if not inventory_path.is_file() or inventory_path.is_symlink():
        result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_MISSING"
        result["entry_validation_message"] = "artifact_inventory.json is missing"
        return result
    actual_inventory_sha = _sha256_file(inventory_path)
    if actual_inventory_sha != completion["artifact_inventory_sha256"]:
        result["entry_status"] = "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH"
        result["entry_validation_message"] = "completion inventory SHA does not match"
        return result
    try:
        inventory = _json_object(inventory_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
        result["entry_validation_message"] = f"artifact inventory invalid: {type(error).__name__}"
        return result
    if inventory.get("schema_version") != 1 or inventory.get("excluded_from_inventory") != sorted(ENTRY_INVENTORY_EXCLUDED):
        result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
        result["entry_validation_message"] = "inventory schema or exclusions are invalid"
        return result
    rows = inventory.get("artifacts")
    if not isinstance(rows, list) or rows != sorted(rows, key=lambda item: item.get("relative_path", "") if isinstance(item, dict) else ""):
        result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
        result["entry_validation_message"] = "inventory artifact rows are invalid"
        return result
    if inventory.get("canonical_inventory_sha256") != _sha256_bytes(canonical_json_bytes(rows)):
        result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
        result["entry_validation_message"] = "inventory canonical SHA is invalid"
        return result
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
            result["entry_validation_message"] = "inventory row is not an object"
            return result
        name = row.get("relative_path")
        if not isinstance(name, str) or not name or name in ENTRY_INVENTORY_EXCLUDED or "/" in name or "\\" in name or name in {".", ".."}:
            result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
            result["entry_validation_message"] = "inventory path is invalid"
            return result
        if name in names or not isinstance(row.get("size_bytes"), int) or row["size_bytes"] < 0 or not _sha256_field(row.get("sha256")):
            result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
            result["entry_validation_message"] = "inventory row identity is invalid"
            return result
        names.add(name)
        artifact = output_dir / name
        if artifact.is_symlink() or not artifact.is_file():
            result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
            result["entry_validation_message"] = f"inventory artifact is not regular: {name}"
            return result
        if artifact.stat().st_size != row["size_bytes"] or _sha256_file(artifact) != row["sha256"]:
            result["entry_status"] = "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH"
            result["entry_validation_message"] = f"inventory binding mismatch: {name}"
            return result
    try:
        actual_names = set()
        for artifact in output_dir.iterdir():
            if artifact.name in ENTRY_INVENTORY_EXCLUDED:
                continue
            if artifact.is_symlink() or not artifact.is_file():
                raise ValueError(f"unlisted non-regular artifact: {artifact.name}")
            actual_names.add(artifact.name)
    except OSError as error:
        result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
        result["entry_validation_message"] = f"cannot enumerate entry output: {type(error).__name__}"
        return result
    if actual_names != names:
        result["entry_status"] = "FAILED_ENTRY_ARTIFACT_INVENTORY_INVALID"
        result["entry_validation_message"] = "inventory does not cover every output artifact"
        return result
    config_path = output_dir / "config.json"
    split_plan_path = output_dir / "split_plan.json"
    if not _sha256_field(completion.get("config_sha256")) or not _sha256_field(completion.get("split_plan_sha256")):
        result["entry_status"] = "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH"
        result["entry_validation_message"] = "config or split plan SHA is missing"
        return result
    if (
        "config.json" not in names
        or "split_plan.json" not in names
        or config_path.is_symlink()
        or not config_path.is_file()
        or split_plan_path.is_symlink()
        or not split_plan_path.is_file()
    ):
        result["entry_status"] = "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH"
        result["entry_validation_message"] = "config or split plan is not inventory-bound"
        return result
    if _sha256_file(config_path) != completion["config_sha256"] or _sha256_file(split_plan_path) != completion["split_plan_sha256"]:
        result["entry_status"] = "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH"
        result["entry_validation_message"] = "completion config or split plan SHA does not match"
        return result
    try:
        config = _json_object(config_path)
        split_plan = _json_object(split_plan_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        result["entry_status"] = "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH"
        result["entry_validation_message"] = f"config or split plan is invalid: {type(error).__name__}"
        return result
    split_config = config.get("split")
    if (
        config.get("schema_version") != 1
        or config.get("protocol_id") != "P3-VISDRONE-DATA-PROTOCOL-V2"
        or config.get("test_access_allowed") is not False
        or config.get("confirmatory_metrics_accessed") is not False
        or config.get("real_conversion_outputs_generated") is not True
        or config.get("production_split_manifest_generated") is not True
        or not isinstance(split_config, dict)
        or split_config.get("test") != "disabled"
        or split_config.get("seed") != 20260808
        or split_config.get("salt") != "P3-confirmatory-v1"
        or split_config.get("target_confirmatory_images") != 647
        or split_config.get("selection_policy") != "feasibility_first_nearest_hash_prefix_v2"
        or split_config.get("planner") != "plan_confirmatory_split_v2"
        or split_config.get("selection_allowed") is not False
        or split_config.get("metrics_access_allowed") is not False
        or split_config.get("single_final_access_only") is not True
    ):
        result["entry_status"] = "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH"
        result["entry_validation_message"] = "V2 output config contract is invalid"
        return result
    integer_fields = (
        "schema_version", "target_image_count", "selected_image_count", "seed",
        "evaluated_prefix_count", "feasible_prefix_count", "selected_prefix_length",
    )
    if (
        split_plan.get("schema_version") != 1
        or split_plan.get("selection_policy") != split_config.get("selection_policy")
        or split_plan.get("seed") != split_config.get("seed")
        or split_plan.get("salt") != split_config.get("salt")
        or split_plan.get("target_image_count") != split_config.get("target_confirmatory_images")
        or any(type(split_plan.get(key)) is not int for key in integer_fields)
        or split_plan.get("evaluated_prefix_count", 0) <= 0
        or split_plan.get("feasible_prefix_count", 0) <= 0
        or split_plan.get("feasible_prefix_count", 0) > split_plan.get("evaluated_prefix_count", 0)
        or split_plan.get("selected_prefix_length", 0) > split_plan.get("evaluated_prefix_count", 0)
        or not _sha256_field(split_plan.get("selected_group_list_sha256"))
        or split_plan.get("selection_allowed") is not False
        or split_plan.get("metrics_access_allowed") is not False
        or split_plan.get("single_final_access_only") is not True
        or not isinstance(split_plan.get("checks"), list)
        or not split_plan["checks"]
        or not all(
            isinstance(item, dict)
            and type(item.get("passed")) is bool
            and item.get("passed") is True
            for item in split_plan["checks"]
        )
    ):
        result["entry_status"] = "FAILED_ENTRY_INVENTORY_BINDING_MISMATCH"
        result["entry_validation_message"] = "V2 split plan contract is invalid"
        return result
    result["entry_status"] = "COMPLETED"
    result["entry_success_accepted"] = True
    return result


def _termination_signal(returncode: int | None) -> dict[str, object] | None:
    if returncode is None or returncode >= 0:
        return None
    number = -returncode
    return {"signal_number": number, "signal_name": _signal_name(number)}


def _write_internal_error(evidence_dir: Path, error: BaseException) -> dict[str, object]:
    payload = canonical_json_bytes({
        "schema_version": PROCESS_SCHEMA_VERSION,
        "status": "FAILED_LAUNCHER_INTERNAL",
        "exception_type": type(error).__name__,
        "message": str(error),
        "original_child_exception_preserved": True,
    })
    _safe_atomic_write(evidence_dir / "process_error.json", payload)
    return _file_ref(evidence_dir / "process_error.json", relative_path="process_error.json")


def _finalize(
    evidence_dir: Path,
    output_dir: Path,
    *,
    argv: list[str],
    child_python: Path,
    module_path: Path,
    source_root: Path,
    start_utc: str,
    start_time: float,
    returncode: int | None,
    signal_events: list[dict[str, object]],
    kill_sent: bool,
    internal_error: BaseException | None,
    input_protocol_config_before: dict[str, object],
    input_protocol_config_after: dict[str, object],
) -> int:
    if internal_error is not None:
        process_error = _write_internal_error(evidence_dir, internal_error)
    else:
        process_error = {"present": False, "relative_path": "process_error.json"}
    _safe_atomic_write(evidence_dir / "process_timing.json", canonical_json_bytes({
        "schema_version": PROCESS_SCHEMA_VERSION,
        "started_at_utc": start_utc,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.time() - start_time,
    }))
    exit_payload: bytes | None = None
    if isinstance(returncode, int):
        exit_payload = f"{returncode}\n".encode("ascii")
        _safe_atomic_write(evidence_dir / "process_exit_code.txt", exit_payload)
    exit_ref = _file_ref(evidence_dir / "process_exit_code.txt", relative_path="process_exit_code.txt")
    if returncode == 0 and internal_error is None:
        entry = _validate_entry(output_dir, input_protocol_config_before, input_protocol_config_after)
        status = str(entry["entry_status"])
    elif returncode == 2:
        entry = _validate_entry(
            output_dir, input_protocol_config_before, input_protocol_config_after
        ) if (output_dir / "completion.json").exists() else {
            "entry_completion": _file_ref(output_dir / "completion.json", relative_path="completion.json"),
            "entry_artifact_inventory": _file_ref(output_dir / "artifact_inventory.json", relative_path="artifact_inventory.json"),
            "entry_partial_inventory": _file_ref(output_dir / "partial_inventory.json", relative_path="partial_inventory.json"),
            "entry_completion_status": None,
            "entry_validation_message": None,
            "entry_success_accepted": False,
            "entry_status": "FAILED_CHILD_CONTRACT",
        }
        status = "FAILED_CHILD_CONTRACT"
    elif internal_error is not None:
        entry = {
            "entry_completion": _file_ref(output_dir / "completion.json", relative_path="completion.json"),
            "entry_artifact_inventory": _file_ref(output_dir / "artifact_inventory.json", relative_path="artifact_inventory.json"),
            "entry_partial_inventory": _file_ref(output_dir / "partial_inventory.json", relative_path="partial_inventory.json"),
            "entry_completion_status": None,
            "entry_validation_message": None,
            "entry_success_accepted": False,
            "entry_status": "FAILED_LAUNCHER_INTERNAL",
        }
        status = "FAILED_LAUNCHER_INTERNAL"
    elif returncode is not None:
        entry = {
            "entry_completion": _file_ref(output_dir / "completion.json", relative_path="completion.json"),
            "entry_artifact_inventory": _file_ref(output_dir / "artifact_inventory.json", relative_path="artifact_inventory.json"),
            "entry_partial_inventory": _file_ref(output_dir / "partial_inventory.json", relative_path="partial_inventory.json"),
            "entry_completion_status": None,
            "entry_validation_message": None,
            "entry_success_accepted": False,
            "entry_status": "FAILED_CHILD_GENERIC",
        }
        status = "FAILED_CHILD_GENERIC"
    else:
        entry = {
            "entry_completion": {"present": False, "relative_path": "completion.json"},
            "entry_artifact_inventory": {"present": False, "relative_path": "artifact_inventory.json"},
            "entry_partial_inventory": {"present": False, "relative_path": "partial_inventory.json"},
            "entry_completion_status": None,
            "entry_validation_message": None,
            "entry_success_accepted": False,
            "entry_status": "FAILED_LAUNCHER_INTERNAL",
        }
        status = "FAILED_LAUNCHER_INTERNAL"
    if status != "COMPLETED":
        _safe_atomic_write(evidence_dir / "partial_inventory.json", canonical_json_bytes(_inventory(
            evidence_dir, frozenset({"process_inventory.json", "process_completion.json", "partial_inventory.json"})
        )))
    process_inventory = _inventory(evidence_dir, PROCESS_INVENTORY_EXCLUDED)
    process_inventory_bytes = canonical_json_bytes(process_inventory)
    _safe_atomic_write(evidence_dir / "process_inventory.json", process_inventory_bytes)
    launcher_exit_code = 0 if status == "COMPLETED" else 2 if status == "FAILED_CHILD_CONTRACT" else 1
    process_completion = {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "status": status,
        "child_returncode": returncode,
        "termination_signal": _termination_signal(returncode),
        "launcher_exit_code": launcher_exit_code,
        "signal_events": signal_events,
        "child_kill_escalated_to_sigkill": kill_sent,
        "process_exit_code_persisted": exit_ref,
        "process_exit_code_actual_bytes_sha256": exit_ref.get("sha256"),
        "process_exit_code_matches_child": isinstance(returncode, int) and exit_ref.get("present") is True and exit_payload == (evidence_dir / "process_exit_code.txt").read_bytes(),
        "child_argv": argv,
        "child_python": str(child_python),
        "module": CHILD_MODULE,
        "module_file": str(module_path),
        "pythonpath": str(source_root),
        "cuda_visible_devices": "",
        "python_no_user_site": "1",
        "console": _file_ref(evidence_dir / "process_console.log", relative_path="process_console.log"),
        "entry_completion": entry["entry_completion"],
        "entry_completion_status": entry["entry_completion_status"],
        "entry_completion_required": True,
        "entry_success_accepted": entry["entry_success_accepted"],
        "entry_validation_message": entry["entry_validation_message"],
        "entry_artifact_inventory": entry["entry_artifact_inventory"],
        "entry_partial_inventory": entry["entry_partial_inventory"],
        "process_inventory": {
            "relative_path": "process_inventory.json",
            "size_bytes": len(process_inventory_bytes),
            "sha256": _sha256_bytes(process_inventory_bytes),
        },
        "process_error": process_error,
        "input_protocol_config_before": input_protocol_config_before,
        "input_protocol_config_after": input_protocol_config_after,
        "input_protocol_config_unchanged": input_protocol_config_before == input_protocol_config_after,
        "dataset_test_accessed_by_this_process": False,
        "confirmatory_metrics_accessed": False,
        "completion_self_hash_included": False,
        "child_failure_preserved_in_console": returncode is not None and returncode != 0,
    }
    _safe_atomic_write(evidence_dir / "process_completion.json", canonical_json_bytes(process_completion))
    return launcher_exit_code


def run(
    data_root: Path,
    output_dir: Path,
    protocol_config: Path,
    splits: tuple[str, ...],
    process_evidence_dir: Path,
    child_python: Path,
    source_root: Path,
) -> int:
    """Run one child and persist process evidence, including child failures."""

    output_dir, evidence_dir, child_python, protocol_config, source_root = _validate_run_args(
        data_root, output_dir, protocol_config, splits, process_evidence_dir, child_python, source_root
    )
    _validate_input_protocol_config(protocol_config)
    input_protocol_config_before = _input_protocol_config_ref(protocol_config)
    evidence_dir.mkdir(parents=True)
    module_path = _module_source(source_root).resolve()
    argv = _child_argv(child_python, data_root, output_dir, protocol_config, splits)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(source_root),
        "P3_VISDRONE_PRODUCTION_CONVERSION_AUTHORIZED": "1",
    }
    invocation = {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "child_argv": argv,
        "child_python": str(child_python),
        "module": CHILD_MODULE,
        "module_file": str(module_path),
        "source_root": str(source_root),
        "pythonpath": str(source_root),
        "cwd": str(source_root.parent.parent),
        "cuda_visible_devices": "",
        "python_no_user_site": "1",
        "process_evidence_portable": False,
        "absolute_runtime_paths_recorded": True,
        "portable_entry_artifacts_exclude_data_root": True,
        "output_dir": str(output_dir),
        "process_evidence_dir": str(evidence_dir),
        "input_protocol_config_before": input_protocol_config_before,
        "dataset_test_accessed_by_this_process": False,
        "confirmatory_metrics_accessed": False,
    }
    start_time = time.time()
    start_utc = _utc_now()
    returncode: int | None = None
    signal_events: list[dict[str, object]] = []
    kill_sent = False
    internal_error: BaseException | None = None
    previous_handlers: dict[int, object] = {}
    input_protocol_config_after = {"present": False, "relative_path": "protocol_config.json"}
    try:
        _atomic_write(evidence_dir / "process_invocation.json", canonical_json_bytes(invocation))
        console_path = evidence_dir / "process_console.log"
        with console_path.open("wb") as console:
            returncode, signal_events, kill_sent, previous_handlers = _run_child(
                argv, cwd=source_root.parent.parent, env=env, console=console
            )
    except BaseException as error:
        internal_error = error
    input_protocol_config_after = _input_protocol_config_ref(protocol_config)
    try:
        return _finalize(
            evidence_dir,
            output_dir,
            argv=argv,
            child_python=child_python,
            module_path=module_path,
            source_root=source_root,
            start_utc=start_utc,
            start_time=start_time,
            returncode=returncode,
            signal_events=signal_events,
            kill_sent=kill_sent,
            internal_error=internal_error,
            input_protocol_config_before=input_protocol_config_before,
            input_protocol_config_after=input_protocol_config_after,
        )
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visdrone-process-launcher")
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("contract-check")
    check.add_argument("--source-root", type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--data-root", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--protocol-config", required=True, type=Path)
    run_parser.add_argument("--split", required=True, type=_split_arg)
    run_parser.add_argument("--process-evidence-dir", required=True, type=Path)
    run_parser.add_argument("--child-python", required=True, type=Path)
    run_parser.add_argument("--source-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "contract-check":
            print(canonical_json_bytes(contract_check(args.source_root)).decode("utf-8"))
            return 0
        return run(
            args.data_root, args.output_dir, args.protocol_config, args.split,
            args.process_evidence_dir, args.child_python, args.source_root,
        )
    except ProcessLauncherContractError as error:
        print(f"visdrone-process-launcher: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"visdrone-process-launcher: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
