"""Evidence for v2b CPU engineering; importing this module never probes a GPU.

GPU observation is an explicit future-facing operation. It only reads process
output/sysfs and never changes hardware settings. An unavailable clock-cap
readback remains unavailable; a current or maximum clock is not cap evidence.
Persisted hashes establish integrity, not cryptographic proof of a collector.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


class EvidenceError(ValueError):
    """Evidence is incomplete, inconsistent, stale, or not native."""


_SCHEMA = 1
_NATIVE_TOKEN = object()
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DATA_ROLES = {"train_core", "development"}


def _strict(value: Any) -> None:
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise EvidenceError("JSON keys must be strings")
        for child in value.values():
            _strict(child)
    elif type(value) is list:
        for child in value:
            _strict(child)
    elif type(value) is float:
        if not math.isfinite(value):
            raise EvidenceError("non-finite JSON number")
    elif type(value) not in {str, int, bool, type(None)}:
        raise EvidenceError("only strict builtin JSON values are accepted")


def canonical_json_bytes(value: Any) -> bytes:
    _strict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise EvidenceError("hash input must be bytes")
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def strict_json_loads(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise EvidenceError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise EvidenceError("non-finite JSON constant: " + value)

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                            parse_constant=invalid_constant)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise EvidenceError("invalid strict JSON") from exc
    _strict(result)
    return result


def _copy(value: Any) -> Any:
    return strict_json_loads(canonical_json_bytes(value))


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise EvidenceError(name + " must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, name: str) -> None:
    if not value or type(value) not in {str, dict, list}:
        raise EvidenceError(name + " must be nonempty")


def _count(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EvidenceError(name + " must be an integer >= " + str(minimum))
    return value


def _read_regular(path: str | os.PathLike[str]) -> tuple[Path, bytes]:
    path = Path(path).absolute()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise EvidenceError("reference is not a regular file")
            raw = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise EvidenceError("cannot read regular reference: " + str(path)) from exc
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise EvidenceError("reference changed while being read")
    return path, raw


def file_reference(path: str | os.PathLike[str]) -> dict[str, Any]:
    path, raw = _read_regular(path)
    return {"path": str(path), "size_bytes": len(raw), "sha256": sha256_bytes(raw),
            "sha256_scope": "complete_file_bytes"}


def _verify_reference(reference: Mapping[str, Any]) -> bytes:
    if type(reference) is not dict:
        raise EvidenceError("file reference must be a dict")
    for key in ("path", "size_bytes", "sha256"):
        if key not in reference:
            raise EvidenceError("file reference missing " + key)
    _sha(reference["sha256"], "file sha256")
    _count(reference["size_bytes"], "file size")
    if reference.get("sha256_scope", "complete_file_bytes") != "complete_file_bytes":
        raise EvidenceError("file hash scope is not complete_file_bytes")
    _, raw = _read_regular(reference["path"])
    if len(raw) != reference["size_bytes"] or sha256_bytes(raw) != reference["sha256"]:
        raise EvidenceError("file reference size/hash mismatch")
    return raw


def _verify_checkpoint_reference(reference: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    if type(reference) is not dict or reference.get("binding_sha256") != binding["binding_sha256"]:
        raise EvidenceError("checkpoint run binding mismatch")
    _verify_reference(reference)
    from .training_v2b_checkpoint import CheckpointError, inspect_checkpoint

    try:
        inspected = inspect_checkpoint(reference["path"], expected_sha256=reference["sha256"],
                                       expected_binding=dict(binding))
    except CheckpointError as exc:
        raise EvidenceError("checkpoint payload inspection failed: " + str(exc)) from exc
    for key in ("size_bytes", "binding_sha256", "epoch", "epoch_active",
                "optimizer_updates", "microsteps", "samples_seen"):
        if key in reference and reference[key] != inspected[key]:
            raise EvidenceError("checkpoint reference metadata mismatch: " + key)
    _verify_bound_engine_configuration(binding["config"], inspected["engine"]["config"])
    return inspected


def _verify_bound_engine_configuration(config: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    """Resolved declarations and the persisted engine must describe one run."""
    for key in ("physical_batch_size", "accumulation_steps"):
        declared = _count(config.get(key), "bound engine " + key, 1)
        if declared != actual[key]:
            raise EvidenceError("bound engine configuration mismatch: " + key)
    for key in ("amp_dtype", "bn_statistics", "bn_backward_layout"):
        if config.get(key) != actual[key]:
            raise EvidenceError("bound engine configuration mismatch: " + key)
    clipping = _finite(config.get("clip_max_norm"), "bound engine clip_max_norm", 0.0)
    if clipping <= 0 or clipping != actual["clip_max_norm"]:
        raise EvidenceError("bound engine configuration mismatch: clip_max_norm")
    logical = _count(config.get("logical_batch_size"), "bound engine logical_batch_size", 1)
    if logical != actual["physical_batch_size"] * actual["accumulation_steps"]:
        raise EvidenceError("bound engine configuration mismatch: logical_batch_size")
    size = config.get("input_size")
    if type(size) is int:
        expected_size = [_count(size, "bound input_size", 1)] * 2
    elif type(size) is list and len(size) == 2:
        expected_size = [_count(value, "bound input_size", 1) for value in size]
    else:
        raise EvidenceError("bound engine input_size must be an integer or [H,W]")
    if expected_size != actual["expected_input_size"]:
        raise EvidenceError("bound engine configuration mismatch: input_size")
    if "expected_input_size" in config and config["expected_input_size"] != expected_size:
        raise EvidenceError("bound engine expected_input_size contradicts input_size")
    if config.get("device") != "cpu" or config.get("scope") != "cpu_synthetic_engineering":
        raise EvidenceError("bound engine device/scope must be CPU synthetic engineering")


def write_exclusive_json(path: str | os.PathLike[str], document: Any) -> dict[str, Any]:
    """Publish only to an absent path; never create parent directories or overwrite."""
    raw = canonical_json_bytes(document) + b"\n"
    target = Path(path).absolute()
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise EvidenceError("exclusive JSON publication failed: " + str(target)) from exc
    reference = file_reference(target)
    if reference["sha256"] != sha256_bytes(raw):
        raise EvidenceError("published JSON readback mismatch")
    return reference


def initial_parameter_reference(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Hash actual CPU tensor bytes, including buffers, without CUDA calls.

    Identity is not a numerical-health certificate. Legal positional/anchor
    buffers may contain infinity; their exact bytes still belong in identity.
    """
    import torch

    if not isinstance(parameters, Mapping) or not parameters:
        raise EvidenceError("initial parameters must be a nonempty tensor mapping")
    inventory = []
    if any(type(name) is not str for name in parameters):
        raise EvidenceError("parameter names must be strings")
    for name in sorted(parameters):
        tensor = parameters[name]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise EvidenceError("initial tensors must already be on CPU")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise EvidenceError("only dense, unquantized CPU tensors are supported")
        flat = tensor.detach().reshape(-1).contiguous()
        raw = flat.view(torch.uint8).numpy().tobytes()
        inventory.append({"name": name, "dtype": str(tensor.dtype),
                          "shape": list(tensor.shape), "size_bytes": len(raw),
                          "sha256": sha256_bytes(raw)})
    return {"sha256": canonical_sha256(inventory),
            "sha256_scope": "canonical_tensor_inventory_v1", "inventory": inventory}


def _validate_initial(reference: Any) -> None:
    if type(reference) is not dict or not reference.get("inventory"):
        raise EvidenceError("initial weights need a tensor inventory")
    if reference.get("sha256_scope") != "canonical_tensor_inventory_v1":
        raise EvidenceError("initial weights hash scope mismatch")
    if _sha(reference.get("sha256"), "initial weights hash") != canonical_sha256(reference["inventory"]):
        raise EvidenceError("initial weights inventory hash mismatch")
    names = []
    for row in reference["inventory"]:
        if type(row) is not dict or set(row) != {"name", "dtype", "shape", "size_bytes", "sha256"}:
            raise EvidenceError("invalid initial tensor inventory")
        if type(row["name"]) is not str or not row["name"] or type(row["dtype"]) is not str or not row["dtype"]:
            raise EvidenceError("tensor name/dtype must be nonempty strings")
        if type(row["shape"]) is not list:
            raise EvidenceError("tensor shape must be a list")
        for size in row["shape"]:
            _count(size, "tensor dimension")
        _count(row["size_bytes"], "tensor byte size")
        _sha(row["sha256"], "tensor byte hash")
        names.append(row["name"])
    if names != sorted(set(names)):
        raise EvidenceError("tensor inventory must be sorted and unique")
    if "parameter_reference" in reference:
        if reference.get("identity_scope") != "model_state_dict":
            raise EvidenceError("full state identity scope is missing")
        parameters = reference["parameter_reference"]
        _validate_initial(parameters)
        state_rows = {row["name"]: row for row in reference["inventory"]}
        if any(state_rows.get(row["name"]) != row for row in parameters["inventory"]):
            raise EvidenceError("parameter identity differs from the full initial state")


def build_run_binding(*, run_id: str, code_paths: Mapping[str, str | os.PathLike[str]],
                      config: Mapping[str, Any], initial_parameters: Mapping[str, Any],
                      initial_state: Mapping[str, Any] | None = None,
                      input_manifests: Mapping[str, str | os.PathLike[str]] | None = None,
                      synthetic_input: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Bind real source bytes, resolved config, CPU state identity, and input identity."""
    if type(run_id) is not str or not run_id:
        raise EvidenceError("run_id must be a nonempty string")
    if type(code_paths) is not dict or not code_paths or any(type(k) is not str for k in code_paths):
        raise EvidenceError("code_paths must be a nonempty named mapping")
    if type(config) is not dict or not config:
        raise EvidenceError("resolved config must be a nonempty dict")
    parameters = _copy(initial_parameters)
    _validate_initial(parameters)
    initial = parameters
    if initial_state is not None:
        initial = _copy(initial_state)
        initial["identity_scope"] = "model_state_dict"
        initial["parameter_reference"] = parameters
        _validate_initial(initial)
    if bool(input_manifests) == bool(synthetic_input):
        raise EvidenceError("choose exactly one of real manifests and explicit synthetic input")
    if input_manifests:
        if not set(input_manifests) <= _DATA_ROLES:
            raise EvidenceError("only train_core/development manifest roles are allowed")
        data = {"kind": "manifest_binding_only",
                "manifests": {role: file_reference(path) for role, path in input_manifests.items()},
                "dataset_executed": False}
    else:
        synthetic = _copy(synthetic_input)
        _nonempty(synthetic.get("fixture"), "synthetic fixture description")
        _sha(synthetic.get("tensor_sha256"), "synthetic input tensor hash")
        data = {"kind": "synthetic", "identity": synthetic, "real_dataset_accessed": False}
    binding = {"schema_version": _SCHEMA, "run_id": run_id,
               "code": {name: file_reference(path) for name, path in code_paths.items()},
               "config": _copy(config), "initial_weights": initial, "data": data}
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def validate_run_binding(binding: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any]:
    if type(binding) is not dict:
        raise EvidenceError("run binding must be a dict")
    required = {"schema_version", "run_id", "code", "config", "initial_weights", "data", "binding_sha256"}
    if set(binding) != required or binding["schema_version"] != _SCHEMA:
        raise EvidenceError("run binding schema mismatch")
    if type(binding["run_id"]) is not str or not binding["run_id"]:
        raise EvidenceError("run_id must be a nonempty string")
    for key in ("code", "config", "initial_weights", "data"):
        if type(binding[key]) is not dict or not binding[key]:
            raise EvidenceError(key + " identity must be a nonempty dict")
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if _sha(binding["binding_sha256"], "binding hash") != canonical_sha256(body):
        raise EvidenceError("run binding digest mismatch")
    _validate_initial(binding["initial_weights"])
    if verify_files:
        for reference in binding["code"].values():
            _verify_reference(reference)
    data = binding["data"]
    if data.get("kind") == "synthetic":
        _nonempty(data.get("identity", {}).get("fixture"), "synthetic fixture")
        _sha(data.get("identity", {}).get("tensor_sha256"), "synthetic tensor hash")
        if data.get("real_dataset_accessed") is not False:
            raise EvidenceError("synthetic input cannot claim real dataset access")
    elif data.get("kind") == "manifest_binding_only":
        manifests = data.get("manifests")
        if not manifests or not set(manifests) <= _DATA_ROLES:
            raise EvidenceError("invalid manifest role")
        if data.get("dataset_executed") is not False:
            raise EvidenceError("manifest identity does not prove dataset execution")
        if verify_files:
            for reference in manifests.values():
                _verify_reference(reference)
    else:
        raise EvidenceError("unknown input identity kind")
    return _copy(binding)


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise EvidenceError("native boot ID unavailable") from exc
    if not value:
        raise EvidenceError("empty native boot ID")
    return value


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _rapl_observation() -> dict[str, Any]:
    """Read package RAPL values; absence is unknown, never an invented limit."""
    packages = []
    root = Path("/sys/class/powercap")
    for package in sorted(root.glob("intel-rapl:*")):
        if package.name.count(":") != 1:
            continue
        row: dict[str, Any] = {"sysfs_path": str(package), "constraints": [], "errors": []}
        for field, filename in (("name", "name"), ("enabled", "enabled")):
            try:
                row[field] = (package / filename).read_text().strip()
            except OSError:
                row[field] = None
                row["errors"].append(filename)
        for named in sorted(package.glob("constraint_*_name")):
            prefix = named.name[:-5]
            try:
                name = named.read_text().strip()
                power_path = package / (prefix + "_power_limit_uw")
                item = {"name": name, "power_limit_uw": int(power_path.read_text()),
                        "power_limit_source": str(power_path)}
                window = package / (prefix + "_time_window_us")
                item["time_window_us"] = int(window.read_text()) if window.exists() else None
                row["constraints"].append(item)
            except (OSError, ValueError):
                row["errors"].append(str(named))
        packages.append(row)
    return {"source": "native_sysfs", "packages": packages,
            "available": bool(packages) and all(not p["errors"] for p in packages)}


class NativeObservation:
    """In-process collector result; supplied JSON is not a native capability."""
    __slots__ = ("_raw",)

    def __init__(self, token: object, payload: dict[str, Any]) -> None:
        if token is not _NATIVE_TOKEN:
            raise EvidenceError("native observation must come from the collector")
        object.__setattr__(self, "_raw", canonical_json_bytes(payload))

    def __setattr__(self, name: str, value: Any) -> None:
        raise EvidenceError("native observations are immutable")

    def as_dict(self) -> dict[str, Any]:
        return strict_json_loads(self._raw)

    @property
    def sha256(self) -> str:
        return sha256_bytes(self._raw)


def _start_observation(binding: Mapping[str, Any], kind: str) -> dict[str, Any]:
    checked = validate_run_binding(binding)
    return {"schema_version": _SCHEMA, "kind": kind, "collector": "training_v2b_evidence_v1",
            "collector_source": file_reference(__file__),
            "run_id": checked["run_id"], "run_binding_sha256": checked["binding_sha256"],
            "boot_id": _boot_id(), "host": platform.node(), "pid": os.getpid(),
            "utc_started": _utc_now(), "monotonic_started_ns": time.monotonic_ns()}


def _finish_observation(payload: dict[str, Any]) -> NativeObservation:
    payload["utc_finished"] = _utc_now()
    payload["monotonic_finished_ns"] = time.monotonic_ns()
    if payload["boot_id"] != _boot_id():
        raise EvidenceError("boot changed during native collection")
    return NativeObservation(_NATIVE_TOKEN, payload)


def collect_native_cpu_observation(binding: Mapping[str, Any]) -> NativeObservation:
    """Observe this CPU process without importing torch or querying a CUDA device."""
    payload = _start_observation(binding, "native_cpu_environment")
    try:
        installed_torch = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        installed_torch = None
    loaded_torch = sys.modules.get("torch")
    payload.update({
        "device_type": "cpu", "hardware_probe_executed": False, "gpu_ready": False,
        "cpu_environment_collected": True, "gpu_observed": False,
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "host_platform": {"system": platform.system(), "release": platform.release(),
                          "machine": platform.machine(), "cpu_count": os.cpu_count()},
        "torch": {"distribution_version": installed_torch,
                  "module_loaded": loaded_torch is not None,
                  "build_cuda": getattr(getattr(loaded_torch, "version", None), "cuda", None),
                  "cuda_runtime_or_device_queried": False},
        "environment": {name: os.environ.get(name) for name in
                        ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "rapl": _rapl_observation(),
    })
    return _finish_observation(payload)


def _run_gpu_query() -> bytes:
    """The sole GPU process-query boundary, called only by the explicit collector."""
    try:
        result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True,
                                check=True, timeout=15, env=os.environ.copy())
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceError("native GPU process query failed") from exc
    if type(result.stdout) is not bytes:
        raise EvidenceError("native GPU query did not return bytes")
    return result.stdout


def _kernel_driver_version() -> str | None:
    try:
        return Path("/sys/module/nvidia/version").read_text().strip()
    except OSError:
        return None


def collect_native_gpu_observation(binding: Mapping[str, Any], *, gpu_index: int = 0) -> NativeObservation:
    """Future authorized use only. No supplied-observation or runner injection API."""
    _count(gpu_index, "gpu_index")
    payload = _start_observation(binding, "native_gpu_process_and_sysfs")
    raw = _run_gpu_query()
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise EvidenceError("invalid native GPU XML") from exc
    gpus = root.findall("gpu")
    if gpu_index >= len(gpus):
        raise EvidenceError("requested native GPU index is unavailable")
    gpu = gpus[gpu_index]
    uuid = gpu.findtext("uuid")
    if not uuid:
        raise EvidenceError("native GPU UUID missing")
    processes = [{"pid": item.findtext("pid"), "name": item.findtext("process_name"),
                  "type": item.findtext("type"), "memory": item.findtext("used_memory")}
                 for item in gpu.findall("./processes/process_info")]
    kernel_driver = _kernel_driver_version()
    payload.update({
        "device_type": "cuda", "hardware_probe_executed": True, "gpu_ready": False,
        "gpu_index": gpu_index, "gpu_uuid": uuid, "gpu_name": gpu.findtext("product_name"),
        "driver_version": root.findtext("driver_version"), "kernel_driver_version": kernel_driver,
        "cuda_version_reported_by_driver": root.findtext("cuda_version"),
        "current_graphics_clock": gpu.findtext("./clocks/graphics_clock"),
        "device_max_graphics_clock": gpu.findtext("./max_clocks/graphics_clock"),
        "application_graphics_clock": gpu.findtext("./applications_clocks/graphics_clock"),
        "graphics_clock_cap_proof": {"verified": False, "upper_mhz": None,
                                    "reason": "no verified native locked-clock upper-bound readback"},
        "power_limit": (gpu.findtext("./gpu_power_readings/power_limit")
                        or gpu.findtext("./power_readings/power_limit")),
        "enforced_power_limit": (gpu.findtext("./gpu_power_readings/enforced_power_limit")
                                 or gpu.findtext("./power_readings/enforced_power_limit")),
        "memory_total": gpu.findtext("./fb_memory_usage/total"),
        "memory_used": gpu.findtext("./fb_memory_usage/used"),
        "temperature": gpu.findtext("./temperature/gpu_temp"),
        "processes": processes, "rapl": _rapl_observation(),
        "hardware_admission_pass": False,
        "unverified_requirements": ["graphics_clock_cap", "run_interval_health"],
        "raw_process_capture": {"argv": ["nvidia-smi", "-q", "-x"],
                                "stdout_utf8": raw.decode("utf-8"),
                                "sha256": sha256_bytes(raw), "sha256_scope": "complete_stdout_bytes"},
    })
    return _finish_observation(payload)


def validate_native_observation(observation: NativeObservation, binding: Mapping[str, Any], *,
                                max_age_seconds: float = 60.0, expected_boot_id: str | None = None,
                                expected_gpu_uuid: str | None = None,
                                require_gpu_admission: bool = False) -> dict[str, Any]:
    if type(observation) is not NativeObservation:
        raise EvidenceError("supplied observations cannot masquerade as native collection")
    checked = validate_run_binding(binding)
    payload = observation.as_dict()
    if payload.get("run_id") != checked["run_id"] or payload.get("run_binding_sha256") != checked["binding_sha256"]:
        raise EvidenceError("native observation run identity mismatch")
    _verify_reference(payload["collector_source"])
    if payload.get("host") != platform.node() or payload.get("pid") != os.getpid():
        raise EvidenceError("native observation host/process identity mismatch")
    actual_boot = _boot_id()
    if payload.get("boot_id") != actual_boot or (expected_boot_id is not None and payload["boot_id"] != expected_boot_id):
        raise EvidenceError("native observation boot identity mismatch")
    if expected_gpu_uuid is not None and payload.get("gpu_uuid") != expected_gpu_uuid:
        raise EvidenceError("native observation GPU UUID mismatch")
    if type(max_age_seconds) not in {int, float} or not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise EvidenceError("invalid observation age limit")
    started = _count(payload.get("monotonic_started_ns"), "observation start")
    finished = _count(payload.get("monotonic_finished_ns"), "observation finish")
    now = time.monotonic_ns()
    if started > finished or finished > now or now - started > max_age_seconds * 1e9:
        raise EvidenceError("native observation is stale or temporally inconsistent")
    try:
        utc_start = dt.datetime.fromisoformat(payload["utc_started"].replace("Z", "+00:00"))
        utc_end = dt.datetime.fromisoformat(payload["utc_finished"].replace("Z", "+00:00"))
        age = (dt.datetime.now(dt.timezone.utc) - utc_start).total_seconds()
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError("invalid observation UTC time") from exc
    if utc_start.tzinfo is None or utc_end.tzinfo is None or utc_end < utc_start or age < 0 or age > max_age_seconds:
        raise EvidenceError("native observation UTC freshness mismatch")
    if require_gpu_admission and payload.get("hardware_admission_pass") is not True:
        raise EvidenceError("native GPU admission is not closed")
    return payload


def _counters(value: Any) -> dict[str, Any]:
    keys = {"epoch", "optimizer_updates", "microsteps", "samples_seen"}
    if type(value) is not dict or set(value) != keys:
        raise EvidenceError("counters must contain exactly epoch/update/microstep/sample counts")
    for key in keys:
        _count(value[key], key, 0 if key == "epoch" else 1)
    if not value["samples_seen"] >= value["microsteps"] >= value["optimizer_updates"]:
        raise EvidenceError("sample/microstep/update counters are inconsistent")
    return _copy(value)


def _checkpoint_counts(inspected: Mapping[str, Any], value: Any) -> dict[str, Any]:
    counts = _counters(value)
    actual = {key: inspected[key] for key in counts}
    if counts != actual:
        raise EvidenceError("report counters differ from the actual checkpoint payload")
    engine = inspected["engine"]
    if engine["phase"] != 0 or engine["failed"] is not False:
        raise EvidenceError("checkpoint must be at a successful optimizer boundary")
    return counts


def _checkpoint_progress(inspected: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _copy(inspected["engine"][key]) for key in
            ("config", "epoch_active", "epoch_start_optimizer_updates", "phase")}


def _finite(value: Any, name: str, minimum: float | None = None) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise EvidenceError(name + " must be finite")
    if minimum is not None and value < minimum:
        raise EvidenceError(name + " is below its allowed minimum")
    return float(value)


def _verify_window_ledger(references: Any, binding: Mapping[str, Any],
                          inspected: Mapping[str, Any], counts: Mapping[str, Any]) -> None:
    """Verify a full fresh-run ledger, not a partial/resumed-run excerpt."""
    if type(references) is not list or not references:
        raise EvidenceError("completed report needs actual window evidence")
    engine = inspected["engine"]
    physical = engine["config"]["physical_batch_size"]
    accumulation = engine["config"]["accumulation_steps"]
    logical = physical * accumulation
    updates = 0
    last_epoch = 0
    epoch_start_update = 0
    for reference in references:
        document = strict_json_loads(_verify_reference(reference))
        if type(document) is not dict or document.get("run_binding_sha256") != binding["binding_sha256"]:
            raise EvidenceError("window evidence run binding mismatch")
        rows = document.get("windows")
        if type(rows) is not list or not rows:
            raise EvidenceError("window evidence must contain nonempty window records")
        for row in rows:
            if type(row) is not dict:
                raise EvidenceError("window record must be a dict")
            updates += 1
            if updates > counts["optimizer_updates"]:
                raise EvidenceError("too many window records for checkpoint counters")
            if (_count(row.get("optimizer_updates"), "window optimizer_updates", 1) != updates
                    or _count(row.get("microsteps"), "window microsteps", 1) != updates * accumulation):
                raise EvidenceError("window update/microstep counters are not contiguous")
            epoch = _count(row.get("epoch"), "window epoch", 1)
            if epoch not in {last_epoch, last_epoch + 1} or (updates == 1 and epoch != 1):
                raise EvidenceError("window epochs must start at one and be contiguous")
            if epoch != last_epoch:
                epoch_start_update = updates - 1
            last_epoch = epoch
            if _count(row.get("phase"), "window phase") != 0:
                raise EvidenceError("window is not a complete optimizer boundary")
            if _count(row.get("logical_batch_size"), "logical batch", 1) != logical:
                raise EvidenceError("window logical batch differs from checkpoint engine config")
            for key, expected in (("physical_batch_size", physical),
                                  ("accumulation_steps", accumulation),
                                  ("samples_seen", updates * logical)):
                if key in row and _count(row[key], key, 1) != expected:
                    raise EvidenceError("window batch/sample accounting mismatch: " + key)
            target_counts = row.get("microbatch_target_counts")
            if type(target_counts) is not list or len(target_counts) != accumulation:
                raise EvidenceError("window microbatch target count length mismatch")
            for count in target_counts:
                _count(count, "microbatch targets")
            total = _count(row.get("target_total"), "window targets")
            denominator = _count(row.get("denominator"), "window denominator", 1)
            if total != sum(target_counts) or denominator != max(total, 1):
                raise EvidenceError("window target normalization mismatch")
            coefficients = row.get("coefficients")
            losses = row.get("microbatch_losses")
            if (type(coefficients) is not list or len(coefficients) != accumulation
                    or type(losses) is not list or len(losses) != accumulation):
                raise EvidenceError("window coefficient/loss count mismatch")
            recomputed_loss = 0.0
            for target_count, coefficient, loss in zip(target_counts, coefficients, losses):
                coefficient = _finite(coefficient, "window coefficient", 0.0)
                if not math.isclose(coefficient, max(target_count, 1) / denominator,
                                    rel_tol=1e-12, abs_tol=1e-12):
                    raise EvidenceError("window accumulation coefficient mismatch")
                if type(loss) is not dict or not loss or any(type(key) is not str or not key for key in loss):
                    raise EvidenceError("window needs named microbatch losses")
                recomputed_loss += coefficient * sum(_finite(v, "microbatch loss") for v in loss.values())
            loss = _finite(row.get("loss"), "window loss")
            if not math.isclose(loss, recomputed_loss, rel_tol=1e-5, abs_tol=1e-6):
                raise EvidenceError("window loss differs from its weighted microbatch losses")
            # An exactly zero gradient is valid; it is not evidence of failure.
            _finite(row.get("gradient_norm_before_clip"), "window gradient norm", 0.0)
            for key in ("dn_num_groups", "dn_query_counts"):
                values = row.get(key)
                if type(values) is not list or len(values) != accumulation:
                    raise EvidenceError("window native DN metadata count mismatch")
                for value in values:
                    if value is not None:
                        _count(value, key)
        expected_chunk_counts = {"epoch": last_epoch, "optimizer_updates": updates,
                                 "microsteps": updates * accumulation, "samples_seen": updates * logical}
        if _counters(document.get("counters")) != expected_chunk_counts:
            raise EvidenceError("window evidence chunk counters mismatch")
    if (updates != counts["optimizer_updates"] or last_epoch != counts["epoch"]
            or epoch_start_update != engine["epoch_start_optimizer_updates"]):
        raise EvidenceError("window ledger does not cover the full checkpoint history")


def _metric_identity(binding: Mapping[str, Any], checkpoint_reference: Mapping[str, Any],
                     counts: Mapping[str, Any]) -> dict[str, Any]:
    return {"run_id": binding["run_id"], "run_binding_sha256": binding["binding_sha256"],
            "checkpoint_sha256": checkpoint_reference["sha256"],
            "input_identity_sha256": canonical_sha256(binding["data"]), "counters": _copy(counts)}


def _coco_precision_selection(source: Mapping[str, Any], selection: Any, units: Any) -> dict[str, Any]:
    """Extract COCO AP from a bound T x R x K x A x M accumulator."""
    import numpy as np

    parameters = source.get("parameters")
    required = {"iouType", "useCats", "iouThrs", "recThrs", "catIds",
                "areaRng", "areaRngLbl", "maxDets", "area_coordinate_space"}
    if type(parameters) is not dict or set(parameters) != required:
        raise EvidenceError("COCO source needs explicit evaluator parameters for every precision axis")
    if parameters["iouType"] != "bbox" or type(parameters["useCats"]) is not int or parameters["useCats"] != 1:
        raise EvidenceError("only category-aware COCO bbox precision is supported")
    for name in ("iouThrs", "recThrs"):
        values = parameters[name]
        if type(values) is not list or not values:
            raise EvidenceError(name + " must be a nonempty axis")
        for value in values:
            if not 0 <= _finite(value, name) <= 1:
                raise EvidenceError(name + " must lie in [0,1]")
        if values != sorted(set(values)):
            raise EvidenceError(name + " must be sorted and unique")
    for name in ("catIds", "maxDets"):
        values = parameters[name]
        if type(values) is not list or not values:
            raise EvidenceError(name + " must be a nonempty axis")
        for value in values:
            _count(value, name, 1 if name == "maxDets" else 0)
        if values != sorted(set(values)):
            raise EvidenceError(name + " must be sorted and unique")
    labels, ranges = parameters["areaRngLbl"], parameters["areaRng"]
    if (type(labels) is not list or not labels or type(ranges) is not list
            or len(labels) != len(ranges) or any(type(label) is not str or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise EvidenceError("area axis labels/ranges are invalid")
    for bounds in ranges:
        if (type(bounds) is not list or len(bounds) != 2
                or not 0 <= _finite(bounds[0], "area lower") < _finite(bounds[1], "area upper")):
            raise EvidenceError("area axis range must be finite and ordered")
    if parameters["area_coordinate_space"] not in {"original_image_pixels", "resized_image_pixels"}:
        raise EvidenceError("area coordinate space is not bound")
    expected_shape = (len(parameters["iouThrs"]), len(parameters["recThrs"]),
                      len(parameters["catIds"]), len(labels), len(parameters["maxDets"]))
    try:
        precision = np.asarray(source.get("precision"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("precision must be a rectangular numeric tensor") from exc
    if precision.shape != expected_shape:
        raise EvidenceError("precision shape does not match bound evaluator axes")
    if not np.isfinite(precision).all() or not ((precision == -1) | ((precision >= 0) & (precision <= 1))).all():
        raise EvidenceError("precision contains non-finite or invalid sentinel/value entries")
    if type(selection) is not dict or set(selection) != {"max_dets_index", "area_index", "iou_indices"}:
        raise EvidenceError("metric needs an explicit precision-axis selection")
    max_index = _count(selection["max_dets_index"], "maxDets index")
    area_index = _count(selection["area_index"], "area index")
    indices = selection["iou_indices"]
    if (max_index >= expected_shape[4] or area_index >= expected_shape[3]
            or type(indices) is not list or not indices):
        raise EvidenceError("precision-axis selection is empty or out of range")
    for index in indices:
        if _count(index, "IoU index") >= expected_shape[0]:
            raise EvidenceError("IoU index out of range")
    if indices != sorted(set(indices)):
        raise EvidenceError("IoU indices must be sorted and unique")
    selected = precision[indices, :, :, area_index, max_index]
    valid = selected[selected > -1]
    if not valid.size:
        raise EvidenceError("selected COCO precision is unavailable (-1)")
    factor = {"fraction": 1.0, "percent": 100.0}.get(units)
    if factor is None:
        raise EvidenceError("metric units must be explicit fraction or percent")
    label = labels[area_index]
    return {"name": "AP" if label == "all" else "AP_" + label,
            "max_dets": parameters["maxDets"][max_index],
            "area": {"label": label, "range": _copy(ranges[area_index]),
                     "coordinate_space": parameters["area_coordinate_space"]},
            "iou_definition": {"thresholds": [parameters["iouThrs"][index] for index in indices]},
            "precision_selection": _copy(selection), "units": units,
            "evaluator_parameters_sha256": canonical_sha256(parameters),
            "value": float(valid.mean() * factor)}


def _validate_coco_source(source: Any, identity: Mapping[str, Any]) -> None:
    if (type(source) is not dict or source.get("schema_version") != _SCHEMA
            or source.get("kind") != "supplied_cpu_coco_precision"
            or source.get("scientific_certified") is not False):
        raise EvidenceError("metric requires a bound supplied CPU COCO precision source")
    for key, value in identity.items():
        if source.get(key) != value:
            raise EvidenceError("precision source identity mismatch: " + key)
    evaluator = source.get("evaluator")
    if (type(evaluator) is not dict or evaluator.get("family") != "coco"
            or evaluator.get("role") != "diagnostic"):
        raise EvidenceError("COCO diagnostic and VisDrone primary roles must remain distinct")
    _nonempty(evaluator.get("id"), "evaluator id")
    if source.get("weights") not in {"raw", "ema"}:
        raise EvidenceError("precision source weights identity missing")
    _verify_reference(source.get("prediction_reference"))


def build_coco_ap_record(precision: Any, *, parameters: Mapping[str, Any],
                         prediction_reference: Mapping[str, Any], source_path: str | os.PathLike[str],
                         binding: Mapping[str, Any], checkpoint_reference: Mapping[str, Any],
                         counters: Mapping[str, Any], weights: str, evaluator_id: str,
                         max_dets_index: int, area_index: int, iou_indices: Sequence[int],
                         units: str = "percent") -> dict[str, Any]:
    """Bind supplied CPU precision/params/predictions and extract a diagnostic AP.

    This helper does not run an evaluator or prove how predictions were made.
    Its source is explicitly supplied CPU evidence, never scientific certification.
    """
    import numpy as np
    import torch

    checked = validate_run_binding(binding)
    inspected = _verify_checkpoint_reference(checkpoint_reference, checked)
    counts = _checkpoint_counts(inspected, counters)
    identity = _metric_identity(checked, checkpoint_reference, counts)
    if isinstance(precision, torch.Tensor):
        if precision.device.type != "cpu":
            raise EvidenceError("precision tensor must already be on CPU")
        precision = precision.detach().to(dtype=torch.float64).numpy()
    try:
        values = np.asarray(precision, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("invalid supplied CPU precision tensor") from exc
    source = {**identity, "schema_version": _SCHEMA, "kind": "supplied_cpu_coco_precision",
              "scientific_certified": False,
              "evaluator": {"id": evaluator_id, "family": "coco", "role": "diagnostic"},
              "weights": weights, "prediction_reference": _copy(prediction_reference),
              "parameters": _copy(parameters), "precision": values.tolist()}
    _validate_coco_source(source, identity)
    selection = {"max_dets_index": max_dets_index, "area_index": area_index,
                 "iou_indices": list(iou_indices)}
    extracted = _coco_precision_selection(source, selection, units)
    source_reference = write_exclusive_json(source_path, source)
    return {**identity, **extracted, "evaluator": _copy(source["evaluator"]),
            "weights": weights, "prediction_reference": _copy(prediction_reference),
            "raw_prediction_sha256": prediction_reference["sha256"],
            "raw_prediction_sha256_scope": "complete_file_bytes",
            "source_reference": source_reference, "source_kind": source["kind"]}


def validate_metric_record(record: Mapping[str, Any], *, binding: Mapping[str, Any],
                           checkpoint_reference: Mapping[str, Any],
                           counters: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read bound sources and recompute AP; bare scalar claims are unsupported."""
    checked = validate_run_binding(binding)
    inspected = _verify_checkpoint_reference(checkpoint_reference, checked)
    counts = _checkpoint_counts(inspected, counters)
    if type(record) is not dict:
        raise EvidenceError("metric must be a dict")
    identity = _metric_identity(checked, checkpoint_reference, counts)
    for key, value in identity.items():
        if record.get(key) != value:
            raise EvidenceError("metric identity mismatch: " + key)
    source = strict_json_loads(_verify_reference(record.get("source_reference")))
    _validate_coco_source(source, identity)
    if record.get("source_kind") != source["kind"] or "stats_index" in record:
        raise EvidenceError("metric must be extracted from its precision source, not stats indices")
    for key in ("evaluator", "weights", "prediction_reference"):
        if record.get(key) != source[key]:
            raise EvidenceError("metric source metadata mismatch: " + key)
    if (record.get("raw_prediction_sha256") != source["prediction_reference"]["sha256"]
            or record.get("raw_prediction_sha256_scope") != "complete_file_bytes"):
        raise EvidenceError("raw prediction file hash/scope mismatch")
    extracted = _coco_precision_selection(source, record.get("precision_selection"), record.get("units"))
    for key, value in extracted.items():
        if key == "value":
            reported = _finite(record.get(key), "metric value", 0.0)
            if not math.isclose(reported, value, rel_tol=1e-12, abs_tol=1e-12):
                raise EvidenceError("metric value differs from selected precision")
        elif record.get(key) != value:
            raise EvidenceError("metric label differs from bound precision selection: " + key)
    return _copy(record)


def build_completed_report(binding: Mapping[str, Any], observation: NativeObservation, *,
                           checkpoint_reference: Mapping[str, Any], counters: Mapping[str, Any],
                           window_references: Sequence[Mapping[str, Any]] = (),
                           metrics: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Complete only a synthetic CPU engineering report, never scientific certification."""
    checked = validate_run_binding(binding)
    observed = validate_native_observation(observation, checked)
    inspected = _verify_checkpoint_reference(checkpoint_reference, checked)
    report = {"schema_version": _SCHEMA, "status": "completed", "scope": "cpu_synthetic",
              "scientific_certified": False, "gpu_ready": False, "formal_training_executed": False,
              "run_binding": checked, "checkpoint_reference": _copy(checkpoint_reference),
              "counters": _checkpoint_counts(inspected, counters),
              "checkpoint_progress": _checkpoint_progress(inspected),
              "window_references": _copy(list(window_references)), "metrics": _copy(list(metrics)),
              "observation": {"payload": observed, "sha256": observation.sha256,
                              "sha256_scope": "canonical_observation_payload"}}
    report["report_sha256"] = canonical_sha256(report)
    return validate_completed_report(report, binding=checked, observation=observation)


def validate_completed_report(report: Mapping[str, Any], *, binding: Mapping[str, Any],
                              observation: NativeObservation) -> dict[str, Any]:
    checked = validate_run_binding(binding)
    observed = validate_native_observation(observation, checked)
    if type(report) is not dict:
        raise EvidenceError("report must be a dict")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != canonical_sha256(body):
        raise EvidenceError("completed report digest mismatch")
    if report.get("schema_version") != _SCHEMA or report.get("status") != "completed" or report.get("scope") != "cpu_synthetic":
        raise EvidenceError("only completed synthetic CPU engineering reports are supported")
    for flag in ("scientific_certified", "gpu_ready", "formal_training_executed"):
        if report.get(flag) is not False:
            raise EvidenceError("CPU report cannot claim scientific certification or GPU readiness")
    if checked["data"]["kind"] != "synthetic" or observed.get("device_type") != "cpu":
        raise EvidenceError("CPU report must bind synthetic inputs and a CPU observation")
    if observed.get("hardware_probe_executed") is not False or observed.get("gpu_observed") is not False:
        raise EvidenceError("CPU report cannot claim a GPU probe")
    if report.get("run_binding") != checked:
        raise EvidenceError("completed report run binding mismatch")
    expected_observation = {"payload": observed, "sha256": observation.sha256,
                            "sha256_scope": "canonical_observation_payload"}
    if report.get("observation") != expected_observation:
        raise EvidenceError("report native observation identity mismatch")
    inspected = _verify_checkpoint_reference(report.get("checkpoint_reference"), checked)
    counts = _checkpoint_counts(inspected, report.get("counters"))
    if report.get("checkpoint_progress") != _checkpoint_progress(inspected):
        raise EvidenceError("report progress differs from actual checkpoint engine state")
    _verify_window_ledger(report.get("window_references"), checked, inspected, counts)
    if type(report.get("metrics")) is not list:
        raise EvidenceError("report metrics must be a list")
    for metric in report["metrics"]:
        validate_metric_record(metric, binding=checked,
                               checkpoint_reference=report["checkpoint_reference"], counters=counts)
    return _copy(report)
