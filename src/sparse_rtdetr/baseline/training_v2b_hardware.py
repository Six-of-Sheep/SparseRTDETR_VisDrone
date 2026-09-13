"""Passive, source-bound hardware gates for future v2b GPU execution.

This module never imports torch, initializes CUDA, changes clocks/power limits,
or starts a workload. A native probe is evidence, not execution authorization.
The current NVSMI backend exposes no locked graphics-clock upper-bound getter;
current, maximum, application and supported clocks cannot fill that proof gap.
Every native probe therefore fails that gate until a separately reviewed native
readback backend exists. JSON supplied by callers cannot grant GPU admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from .training_v2b_evidence import (
    EvidenceError, canonical_json_bytes, canonical_sha256, file_reference,
    sha256_bytes, strict_json_loads, validate_run_binding,
)

_SCHEMA = 1
_TOKEN = object()
_BOOT_PATH = Path("/proc/sys/kernel/random/boot_id")
_RAPL_ROOT = Path("/sys/class/powercap")
_HWMON_ROOT = Path("/sys/class/hwmon")
_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_MAX_CAPTURE_BYTES = 8 * 1024 * 1024


class HardwareGateError(ValueError):
    """Native evidence is missing, stale, inconsistent or blocks GPU use."""


@dataclass(frozen=True)
class HardwarePolicy:
    expected_gpu_uuid: str
    expected_gpu_name: str = "NVIDIA GeForce RTX 4090 D"
    graphics_clock_upper_mhz: int = 1500
    cpu_pl1_upper_uw: int = 125_000_000
    cpu_pl2_upper_uw: int = 150_000_000
    gpu_stop_temperature_c: int = 80
    cpu_stop_temperature_c: int = 90
    preflight_min_free_memory_mib: int = 20_000
    preflight_max_used_memory_mib: int = 1024
    preflight_max_utilization_percent: int = 5
    interval_min_free_memory_mib: int = 64
    maximum_probe_age_seconds: int = 30
    allowed_graphics_executable_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.expected_gpu_uuid) is not str or _GPU_UUID.fullmatch(self.expected_gpu_uuid) is None:
            raise HardwareGateError("expected_gpu_uuid must be a complete native GPU UUID")
        if type(self.expected_gpu_name) is not str or not self.expected_gpu_name:
            raise HardwareGateError("expected_gpu_name must be nonempty")
        limits = {
            "graphics_clock_upper_mhz": 1500, "cpu_pl1_upper_uw": 125_000_000,
            "cpu_pl2_upper_uw": 150_000_000, "gpu_stop_temperature_c": 80,
            "cpu_stop_temperature_c": 90, "preflight_max_used_memory_mib": 1024,
            "preflight_max_utilization_percent": 5, "maximum_probe_age_seconds": 60,
        }
        for name, maximum in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise HardwareGateError(name + " exceeds the reviewed engineering boundary")
        for name in ("preflight_min_free_memory_mib", "interval_min_free_memory_mib"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise HardwareGateError(name + " must be a positive integer")
        if (type(self.allowed_graphics_executable_sha256) is not tuple
                or len(set(self.allowed_graphics_executable_sha256)) != len(self.allowed_graphics_executable_sha256)
                or any(type(item) is not str or re.fullmatch("[0-9a-f]{64}", item) is None
                       for item in self.allowed_graphics_executable_sha256)):
            raise HardwareGateError("graphics allowances require unique complete executable SHA256s")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["allowed_graphics_executable_sha256"] = list(self.allowed_graphics_executable_sha256)
        return result


class NativeHardwareProbe:
    """Same-process immutable capability; serialization is reporting only."""
    __slots__ = ("_raw",)

    def __init__(self, token: object, payload: dict[str, Any]) -> None:
        if token is not _TOKEN:
            raise HardwareGateError("hardware evidence must come from the native collector")
        object.__setattr__(self, "_raw", canonical_json_bytes(payload))

    def __setattr__(self, name: str, value: Any) -> None:
        raise HardwareGateError("native hardware probes are immutable")

    def __reduce__(self) -> Any:
        raise HardwareGateError("native hardware capabilities cannot be serialized for admission")

    def as_dict(self) -> dict[str, Any]:
        return strict_json_loads(self._raw)

    @property
    def sha256(self) -> str:
        return sha256_bytes(self._raw)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _boot_id() -> str:
    value = _BOOT_PATH.read_text(encoding="ascii").strip()
    if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is None:
        raise HardwareGateError("native boot ID is unavailable or invalid")
    return value


def _text_capture(raw: bytes) -> dict[str, Any]:
    result = {"sha256": sha256_bytes(raw), "size_bytes": len(raw),
              "sha256_scope": "complete_capture_bytes"}
    try:
        result["utf8"] = raw.decode("utf-8")
    except UnicodeError:
        import base64
        result.update(utf8=None, base64=base64.b64encode(raw).decode("ascii"))
    return result


def _native_file(path: Path) -> dict[str, Any]:
    started = time.monotonic_ns()
    result: dict[str, Any] = {"path": str(path), "started_ns": started}
    try:
        raw = path.read_bytes()
        result.update(_text_capture(raw))
        result.update(resolved_path=str(path.resolve()), error=None)
    except OSError as exc:
        result.update(utf8=None, error=type(exc).__name__ + ": " + str(exc))
    result["finished_ns"] = time.monotonic_ns()
    return result


def _run_native_command(argv: list[str], *, trace_loader: bool = False) -> dict[str, Any]:
    """Fixed read-only argv only; no shell, runner injection or setter fallback."""
    result: dict[str, Any] = {"requested_argv": list(argv), "started_ns": time.monotonic_ns(),
                              "returncode": None, "error": None, "truncated": False, "loader_trace": trace_loader}
    process = None
    try:
        located = shutil.which(argv[0])
        if located is None:
            raise FileNotFoundError(argv[0] + " was not found")
        binary = Path(located).resolve(strict=True)
        before = file_reference(binary)
        actual_argv = [str(binary), *argv[1:]]
        environment = os.environ.copy()
        environment.update(LC_ALL="C", LANG="C", SYSTEMD_PAGER="", PAGER="")
        if trace_loader:
            environment["LD_DEBUG"] = "libs"
        process = subprocess.Popen(actual_argv, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
        result.update(argv=actual_argv, subprocess_pid=process.pid, executable=before)
        try:
            stdout, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            result["error"] = "native read-only command timed out"
        result["returncode"] = process.returncode
        if len(stdout) > _MAX_CAPTURE_BYTES or len(stderr) > _MAX_CAPTURE_BYTES:
            result.update(truncated=True, error="native capture exceeded the bounded evidence size")
            stdout, stderr = stdout[:_MAX_CAPTURE_BYTES], stderr[:_MAX_CAPTURE_BYTES]
        result.update(stdout=_text_capture(stdout), stderr=_text_capture(stderr))
        if file_reference(binary) != before:
            result["error"] = "native query executable changed during collection"
    except (OSError, EvidenceError, subprocess.SubprocessError) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    result["finished_ns"] = time.monotonic_ns()
    return result


def _command_text(capture: Mapping[str, Any]) -> str:
    if (capture.get("error") is not None or capture.get("returncode") != 0
            or capture.get("truncated") is not False):
        raise HardwareGateError("native command failed or was incomplete")
    value = capture.get("stdout", {}).get("utf8")
    if type(value) is not str:
        raise HardwareGateError("native command output is not UTF8")
    stderr = capture.get("stderr", {}).get("utf8")
    if capture.get("loader_trace") is True:
        if type(stderr) is not str or any(line.strip() and re.match(r"^\s*[0-9]+:", line) is None
                                          for line in stderr.splitlines()):
            raise HardwareGateError("native loader trace contains an unresolved diagnostic")
    elif stderr != "":
        raise HardwareGateError("native command emitted an unresolved diagnostic")
    return value


def _value(text: str | None, unit: str = "") -> float | None:
    if type(text) is not str:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)" + (r"\s*" + re.escape(unit) if unit else ""), text.strip())
    return float(match.group(1)) if match else None


def parse_gpu_xml(raw: str, *, expected_gpu_uuid: str) -> dict[str, Any]:
    """Pure parser. Its returned dict is never a production admission token."""
    try:
        root = ET.fromstring(raw)
    except (ET.ParseError, TypeError) as exc:
        raise HardwareGateError("native NVSMI XML is malformed") from exc
    if root.tag != "nvidia_smi_log":
        raise HardwareGateError("native NVSMI XML root is invalid")
    gpus = root.findall("gpu")
    candidates = [gpu for gpu in gpus if gpu.findtext("uuid") == expected_gpu_uuid]
    if len(candidates) != 1:
        raise HardwareGateError("expected GPU UUID is absent or ambiguous")
    gpu = candidates[0]
    processes = []
    container = gpu.find("processes")
    if container is not None:
        for row in container.findall("process_info"):
            pid = row.findtext("pid")
            processes.append({"pid": int(pid) if pid and pid.isdecimal() else None,
                              "type": row.findtext("type"), "reported_name": row.findtext("process_name"),
                              "memory_mib": _value(row.findtext("used_memory"), "MiB")})
    thermal_reasons = {}
    reasons = gpu.find("clocks_event_reasons")
    if reasons is None:
        reasons = gpu.find("clocks_throttle_reasons")
    if reasons is not None:
        for row in reasons:
            if any(key in row.tag for key in ("thermal", "hw_slowdown", "power_brake")):
                thermal_reasons[row.tag] = row.text
    ecc = {}
    for row in gpu.findall("./ecc_errors/volatile/*"):
        if "uncorrect" in row.tag or "correctable" in row.tag:
            ecc[row.tag] = _value(row.text)
    power = gpu.find("gpu_power_readings")
    if power is None:
        power = gpu.find("power_readings")
    return {
        "uuid": gpu.findtext("uuid"), "name": gpu.findtext("product_name"),
        "pci_bus_id": gpu.findtext("./pci/pci_bus_id") or gpu.get("id"),
        "physical_index_in_nvml_listing": gpus.index(gpu),
        "driver_userspace_version": root.findtext("driver_version"),
        "driver_cuda_compatibility_version": root.findtext("cuda_version"),
        "current_graphics_clock_mhz": _value(gpu.findtext("./clocks/graphics_clock"), "MHz"),
        "device_max_graphics_clock_mhz": _value(gpu.findtext("./max_clocks/graphics_clock"), "MHz"),
        "application_graphics_clock_mhz": _value(gpu.findtext("./applications_clocks/graphics_clock"), "MHz"),
        "graphics_clock_cap_proof": {
            "verified": False, "locked_upper_mhz": None,
            "backend": "nvidia-smi-q-x",
            "reason": "this native backend exposes no locked graphics upper-bound readback",
        },
        "memory_total_mib": _value(gpu.findtext("./fb_memory_usage/total"), "MiB"),
        "memory_used_mib": _value(gpu.findtext("./fb_memory_usage/used"), "MiB"),
        "memory_free_mib": _value(gpu.findtext("./fb_memory_usage/free"), "MiB"),
        "temperature_c": _value(gpu.findtext("./temperature/gpu_temp"), "C"),
        "utilization_percent": _value(gpu.findtext("./utilization/gpu_util"), "%"),
        "power_limit_w": _value(power.findtext("requested_power_limit") or power.findtext("power_limit"), "W") if power is not None else None,
        "enforced_power_limit_w": _value(power.findtext("current_power_limit") or power.findtext("enforced_power_limit"), "W") if power is not None else None,
        "compute_mode": gpu.findtext("compute_mode"),
        "recovery_action": gpu.findtext("gpu_recovery_action"),
        "thermal_or_power_brake_reasons": thermal_reasons,
        "volatile_ecc_counts": ecc, "processes": processes,
        "process_list_readable": (container is not None and (not (container.text or "").strip()
                                                        or (container.text or "").strip() == "None")),
    }


_HEALTH_RULES = [
    ("GPU_XID", r"\bNVRM:.*\bXid\b|\bXid\s*\("),
    ("GPU_LOST", r"GPU has fallen off|fallen off the bus|RmInitAdapter failed"),
    ("DRIVER_MISMATCH", r"API mismatch|driver/library version mismatch"),
    ("BERT_UNRESOLVED", r"\bBERT:.*(?:Hardware Error|error records|Total records found:\s*[1-9])"),
    ("HARDWARE_ERROR", r"\[Hardware Error\]|\bMachine check\b|\bmachine-check\b"),
    ("PCIE_ERROR", r"\bAER:.*(?:error|fatal|uncorrect)|PCIe Bus Error:"),
    ("EDAC_ERROR", r"\bEDAC\b.*(?:\b[CU]E\b|\b[0-9]+ errors?\b|uncorrect)"),
    ("KERNEL_LOCKUP", r"(?:(?:watchdog|BUG:).*(?:soft|hard).*lockup)|rcu:.*stall|Kernel panic|Oops:|general protection fault"),
    ("THERMAL_ERROR", r"temperature above threshold|critical temperature|CPU.*throttl|thermal.*(?:shutdown|overheat)"),
    ("HOST_OOM", r"Out of memory: Killed process|oom-kill|invoked oom-killer"),
]
_HEALTH_RULES = [(code, re.compile(pattern, re.IGNORECASE)) for code, pattern in _HEALTH_RULES]


def parse_kernel_journal(raw: str, *, boot_id: str, expected_start_cursor: str | None = None) -> dict[str, Any]:
    """Require current-boot JSON and an exact retained cursor for every delta."""
    records, final_cursor = [], None
    for line in raw.splitlines():
        if line.startswith("-- cursor: "):
            if final_cursor is not None:
                raise HardwareGateError("ambiguous kernel journal cursor")
            final_cursor = line[len("-- cursor: "):].strip()
        elif line.strip():
            try:
                row = strict_json_loads(line.encode("utf-8"))
            except EvidenceError as exc:
                raise HardwareGateError("invalid kernel journal JSON") from exc
            if type(row) is not dict:
                raise HardwareGateError("invalid kernel journal record")
            if row.get("_BOOT_ID") != boot_id.replace("-", "") or row.get("_TRANSPORT") != "kernel":
                raise HardwareGateError("kernel journal boot/transport mismatch")
            if (type(row.get("MESSAGE")) is not str or type(row.get("__CURSOR")) is not str
                    or not row["__CURSOR"] or type(row.get("__MONOTONIC_TIMESTAMP")) is not str
                    or not row["__MONOTONIC_TIMESTAMP"].isdecimal()):
                raise HardwareGateError("kernel journal lacks interpretable message/cursor/time")
            records.append(row)
    if not records or final_cursor != records[-1]["__CURSOR"]:
        raise HardwareGateError("kernel journal is empty or lacks its final cursor")
    times = [int(row["__MONOTONIC_TIMESTAMP"]) for row in records]
    if times != sorted(times):
        raise HardwareGateError("kernel journal timestamps are not monotonic")
    if len({row["__CURSOR"] for row in records}) != len(records):
        raise HardwareGateError("kernel journal repeats a cursor")
    if expected_start_cursor is not None:
        if records[0]["__CURSOR"] != expected_start_cursor:
            raise HardwareGateError("prior kernel journal cursor was lost or rotated")
        examined = records[1:]
        coverage = "verified_cursor_delta"
    else:
        if times[0] > 120_000_000 or not any(row["MESSAGE"].startswith("Linux version ") for row in records[:32]):
            raise HardwareGateError("full-boot kernel history is unavailable")
        examined, coverage = records, "full_current_boot"
    findings = []
    for row in examined:
        codes = [code for code, pattern in _HEALTH_RULES if pattern.search(row["MESSAGE"])]
        if codes:
            findings.append({"codes": codes, "message": row["MESSAGE"], "cursor": row["__CURSOR"],
                             "monotonic_us": int(row["__MONOTONIC_TIMESTAMP"]),
                             "classification": ("firmware_boot_record_unresolved" if "BERT_UNRESOLVED" in codes
                                                else "kernel_health_event")})
    return {"readable": True, "coverage": coverage, "start_cursor": expected_start_cursor,
            "last_cursor": final_cursor, "records_examined": len(examined),
            "last_record_monotonic_us": times[-1], "findings": findings}


def _process_identity(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid}
    try:
        stat_path = Path("/proc") / str(pid) / "stat"
        before = stat_path.read_text(encoding="utf-8")
        executable = (Path("/proc") / str(pid) / "exe").resolve(strict=True)
        binary = file_reference(executable)
        after = stat_path.read_text(encoding="utf-8")
        # comm can contain spaces and parentheses; field22 is relative to the final ')'.
        start_before = int(before.rsplit(")", 1)[1].split()[19])
        start_after = int(after.rsplit(")", 1)[1].split()[19])
        if start_before != start_after:
            raise HardwareGateError("process was replaced while observed")
        result.update(start_ticks=start_before, executable=binary, readable=True)
    except (OSError, ValueError, IndexError, EvidenceError) as exc:
        result.update(readable=False, error=type(exc).__name__ + ": " + str(exc))
    return result


def _collect_rapl() -> dict[str, Any]:
    domains = []
    for package in sorted(_RAPL_ROOT.glob("intel-rapl:*")):
        if re.fullmatch(r"intel-rapl:[0-9]+", package.name) is None:
            continue
        files = {name: _native_file(package / name) for name in ("name", "enabled")}
        constraints = []
        for name_path in sorted(package.glob("constraint_*_name")):
            stem = name_path.name[:-len("name")]
            row = {name: _native_file(package / (stem + name))
                   for name in ("name", "power_limit_uw", "time_window_us")}
            constraints.append(row)
        domains.append({"path": str(package), "files": files, "constraints": constraints})
    return {"domains": domains, "available": bool(domains)}


def _file_text(record: Mapping[str, Any]) -> str | None:
    value = record.get("utf8")
    return value.strip() if record.get("error") is None and type(value) is str else None


def _collect_cpu_temperatures() -> dict[str, Any]:
    sensors = []
    for directory in sorted(_HWMON_ROOT.glob("hwmon*")):
        name = _native_file(directory / "name")
        if _file_text(name) not in {"coretemp", "k10temp"}:
            continue
        for path in sorted(directory.glob("temp*_input")):
            prefix = path.name[:-len("input")]
            files = {field: _native_file(directory / (prefix + field))
                     for field in ("input", "label", "max", "crit")}
            sensors.append({"hwmon": str(directory), "driver": name, "files": files})
    return {"sensors": sensors, "available": bool(sensors)}


def _capability_inventory(commands: dict[str, Any]) -> dict[str, Any]:
    commands["gpu_query_help"] = _run_native_command(["nvidia-smi", "--help-query-gpu"])
    commands["nvidia_smi_version"] = _run_native_command(["nvidia-smi", "--version"])
    result: dict[str, Any] = {"locked_upper_getter_implemented": False, "nvml_library": None,
                              "matching_exports": [], "error": None}
    try:
        _command_text(commands["gpu_query_help"])
        version_output = _command_text(commands["nvidia_smi_version"])
        result["version_output"] = version_output
        query = commands["gpu_xml"]
        _command_text(query)
        if query.get("loader_trace") is not True:
            raise HardwareGateError("native NVML loader trace was not captured")
        # This is the library actually loaded by this NVSMI query, not an ldconfig
        # candidate. LD_DEBUG=libs affects only the read-only query subprocess.
        stderr = query["stderr"]["utf8"]
        candidates = set(re.findall(r"calling init:\s*(/[^\s]*libnvidia-ml\.so[^\s]*)", stderr))
        if len(candidates) != 1:
            raise HardwareGateError("actual NVML userspace library identity is ambiguous")
        library = Path(candidates.pop()).resolve(strict=True)
        result["nvml_library"] = file_reference(library)
        result["library_identity_source"] = "gpu_xml_subprocess_native_dynamic_loader_trace"
        version = re.fullmatch(r"libnvidia-ml\.so\.([0-9]+(?:\.[0-9]+)+)", library.name)
        result["library_filename_version"] = version.group(1) if version else None
        commands["nvml_exports"] = _run_native_command(["nm", "-D", str(library)])
        symbols = _command_text(commands["nvml_exports"])
        result["matching_exports"] = [line.rsplit(None, 1)[-1] for line in symbols.splitlines()
                                      if any(key in line for key in ("LockedClock", "MinMaxClockOfPState", "ClockOffsets"))]
        result["has_named_locked_gpu_getter"] = "nvmlDeviceGetGpuLockedClocks" in result["matching_exports"]
        if file_reference(library) != result["nvml_library"]:
            raise HardwareGateError("NVML library changed during observation")
    except (HardwareGateError, OSError, EvidenceError) as exc:
        result["error"] = str(exc)
    return result


def _assess(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Pure gate calculation, always reapplied to the native capability."""
    findings: list[dict[str, str]] = []
    def fail(code: str, detail: str) -> None:
        findings.append({"code": code, "detail": detail})
    policy, gpu = payload["policy"], payload.get("gpu")
    if not gpu:
        fail("GPU_OBSERVATION_UNAVAILABLE", payload.get("gpu_error", "native query failed"))
    else:
        if gpu["uuid"] != policy["expected_gpu_uuid"] or gpu["name"] != policy["expected_gpu_name"]:
            fail("GPU_IDENTITY_MISMATCH", "native GPU name/UUID differs from the bound policy")
        kernel = _file_text(payload["native_files"]["kernel_driver_version"])
        if not kernel or kernel != gpu["driver_userspace_version"]:
            fail("DRIVER_VERSION_MISMATCH", "kernel and userspace NVIDIA driver versions are unavailable or different")
        # No caller can supply an invented clock proof: this backend has no getter.
        fail("GRAPHICS_CLOCK_CAP_UNVERIFIED", "NVSMI current/max/application clocks do not read back the locked upper bound")
        if gpu["current_graphics_clock_mhz"] is None:
            fail("GRAPHICS_CLOCK_UNREADABLE", "current graphics clock is missing")
        elif gpu["current_graphics_clock_mhz"] > policy["graphics_clock_upper_mhz"]:
            fail("GRAPHICS_CLOCK_ABOVE_LIMIT", "observed graphics clock exceeds the approved ceiling")
        required_numbers = ("memory_total_mib", "memory_used_mib", "memory_free_mib",
                            "temperature_c", "utilization_percent", "power_limit_w", "enforced_power_limit_w")
        if any(gpu.get(key) is None for key in required_numbers):
            fail("GPU_SENSOR_UNREADABLE", "required memory/temperature/utilization/power readback is missing")
        else:
            if (gpu["memory_total_mib"] <= 0 or gpu["memory_used_mib"] > gpu["memory_total_mib"]
                    or gpu["memory_free_mib"] > gpu["memory_total_mib"]
                    or gpu["memory_used_mib"] + gpu["memory_free_mib"] > gpu["memory_total_mib"] + 2
                    or gpu["utilization_percent"] > 100):
                fail("GPU_SENSOR_INCONSISTENT", "native GPU sensor values are inconsistent")
            if gpu["temperature_c"] >= policy["gpu_stop_temperature_c"]:
                fail("GPU_OVER_TEMPERATURE", "GPU temperature reached the bound stop threshold")
            if payload["phase"] == "preflight":
                if gpu["memory_free_mib"] < policy["preflight_min_free_memory_mib"]:
                    fail("GPU_MEMORY_BELOW_PREFLIGHT_RESERVE", "GPU free memory is below the bound preflight reserve")
                if gpu["memory_used_mib"] > policy["preflight_max_used_memory_mib"]:
                    fail("GPU_PREFLIGHT_MEMORY_OCCUPIED", "GPU memory is already occupied above the idle budget")
                if gpu["utilization_percent"] > policy["preflight_max_utilization_percent"]:
                    fail("GPU_PREFLIGHT_BUSY", "GPU utilization exceeds the idle budget")
            elif gpu["memory_free_mib"] < policy["interval_min_free_memory_mib"]:
                fail("GPU_MEMORY_BELOW_INTERVAL_RESERVE", "GPU free memory is below the bound interval reserve")
        if gpu.get("recovery_action") not in {None, "None", "N/A"}:
            fail("GPU_RECOVERY_REQUIRED", "driver reports a nonempty GPU recovery action")
        thermal = gpu["thermal_or_power_brake_reasons"]
        if not thermal or any(value not in {"Active", "Not Active"} for value in thermal.values()):
            fail("GPU_THROTTLE_STATUS_UNREADABLE", "thermal/power-brake status is unavailable")
        elif any(value == "Active" for value in thermal.values()):
            fail("GPU_THERMAL_OR_POWER_BRAKE", "hardware slowdown, thermal slowdown or external power brake is active")
        if any(value is not None and value > 0 for value in gpu["volatile_ecc_counts"].values()):
            fail("GPU_ECC_ERROR", "native volatile ECC counter is nonzero")
        if not gpu["process_list_readable"]:
            fail("GPU_PROCESSES_UNREADABLE", "GPU process list is missing")
        for process in gpu["processes"]:
            identity = process.get("identity", {})
            owner = identity.get("pid") == payload["process"]["pid"] and identity == payload["process"]
            if payload["phase"] == "interval" and owner:
                continue
            allowed_graphics = (process["type"] == "G" and identity.get("readable") is True
                                and identity["executable"]["sha256"] in policy["allowed_graphics_executable_sha256"])
            if not allowed_graphics:
                code = "GPU_GRAPHICS_OCCUPANT_UNAPPROVED" if process["type"] == "G" else "GPU_COMPETING_PROCESS"
                fail(code, "GPU PID " + str(process["pid"]) + " type " + str(process["type"]) + " is not allowed")
    if not payload["process"].get("readable"):
        fail("COLLECTOR_IDENTITY_UNREADABLE", "collector executable/PID start identity is not readable")
    inventory = payload["capability_inventory"]
    if inventory.get("nvml_library") is None:
        fail("NVML_LIBRARY_IDENTITY_UNREADABLE", "actual userspace NVML shared-library identity is unavailable")
    if gpu and inventory.get("library_filename_version") != gpu["driver_userspace_version"]:
        fail("NVML_LIBRARY_VERSION_MISMATCH", "actual loaded NVML filename version differs from its reported driver version")
    if inventory.get("error") is not None:
        fail("CLOCK_BACKEND_INVENTORY_INCOMPLETE", inventory["error"])
    rapl = payload["rapl"]
    if not rapl["available"]:
        fail("RAPL_UNAVAILABLE", "CPU package power constraints are not readable")
    for domain in rapl["domains"]:
        if _file_text(domain["files"]["enabled"]) != "1":
            fail("RAPL_DISABLED_OR_UNREADABLE", "CPU package RAPL is not enabled")
        values = {}
        for row in domain["constraints"]:
            name = _file_text(row["name"])
            if name in {"long_term", "short_term"}:
                if name in values:
                    fail("RAPL_AMBIGUOUS", "duplicate CPU package constraint names")
                values[name] = _value(_file_text(row["power_limit_uw"]))
                if _value(_file_text(row["time_window_us"])) is None:
                    fail("RAPL_WINDOW_UNREADABLE", "CPU package constraint window is unreadable")
        for name, limit in (("long_term", policy["cpu_pl1_upper_uw"]), ("short_term", policy["cpu_pl2_upper_uw"])):
            if values.get(name) is None or not 0 < values[name] <= limit:
                fail("RAPL_LIMIT_MISMATCH", "CPU package " + name + " limit is missing, zero or above the bound ceiling")
    temperatures = payload["cpu_temperatures"]
    if not temperatures["available"]:
        fail("CPU_TEMPERATURE_UNREADABLE", "CPU core/package temperature sensors are unavailable")
    for sensor in temperatures["sensors"]:
        value = _value(_file_text(sensor["files"]["input"]))
        if value is None:
            fail("CPU_TEMPERATURE_UNREADABLE", "a CPU temperature sensor is unreadable")
        elif value >= policy["cpu_stop_temperature_c"] * 1000:
            fail("CPU_OVER_TEMPERATURE", "CPU temperature reached the bound stop threshold")
    health = payload["kernel_health"]
    if not health.get("readable"):
        fail("KERNEL_HEALTH_UNAVAILABLE", health.get("error", "kernel history unavailable"))
    else:
        if health["last_record_monotonic_us"] * 1000 > payload["monotonic_finished_ns"]:
            fail("KERNEL_HEALTH_TIME_INVALID", "kernel record is newer than this observation")
        for row in health["findings"]:
            fail("KERNEL_" + row["codes"][0], row["message"])
        for row in health.get("prior_findings", []):
            fail("PRIOR_KERNEL_" + row["codes"][0], row["message"])
    if payload.get("prior_gate_blocked"):
        fail("PRIOR_PROBE_BLOCKED", "an interval cannot erase a blocked preflight")
    if payload.get("boot_id_after") != payload["boot_id"]:
        fail("BOOT_CHANGED_DURING_PROBE", "host rebooted while collecting hardware evidence")
    if payload["monotonic_finished_ns"] - payload["monotonic_started_ns"] > policy["maximum_probe_age_seconds"] * 1e9:
        fail("PROBE_CAPTURE_TOO_SLOW", "native collection exceeded its freshness budget")
    return findings


def _check_capability(probe: NativeHardwareProbe, binding: Mapping[str, Any], *,
                      max_age_seconds: float) -> dict[str, Any]:
    if type(probe) is not NativeHardwareProbe:
        raise HardwareGateError("supplied observation JSON cannot grant native hardware admission")
    if type(max_age_seconds) not in {int, float} or not 0 < max_age_seconds <= 60:
        raise HardwareGateError("native probe age limit must be in (0,60] seconds")
    try:
        bound = validate_run_binding(binding)
        payload = probe.as_dict()
        if payload["run_binding_sha256"] != bound["binding_sha256"] or payload["run_id"] != bound["run_id"]:
            raise HardwareGateError("native hardware probe run binding differs")
        references = list(payload["collector_sources"].values())
        references.extend(row["executable"] for row in payload["commands"].values() if "executable" in row)
        if payload["capability_inventory"].get("nvml_library") is not None:
            references.append(payload["capability_inventory"]["nvml_library"])
        for reference in references:
            if file_reference(reference["path"]) != reference:
                raise HardwareGateError("native hardware collector source changed")
        if payload["boot_id"] != _boot_id() or payload["boot_id_after"] != payload["boot_id"]:
            raise HardwareGateError("native hardware probe belongs to a different boot")
        if payload["host"] != platform.node() or payload["process"] != _process_identity(os.getpid()):
            raise HardwareGateError("native hardware probe process/executable identity changed")
        now = time.monotonic_ns()
        budget = min(max_age_seconds, payload["policy"]["maximum_probe_age_seconds"]) * 1e9
        if not payload["monotonic_started_ns"] <= payload["monotonic_finished_ns"] <= now:
            raise HardwareGateError("native hardware probe timing is inconsistent")
        if now - payload["monotonic_started_ns"] > budget:
            raise HardwareGateError("native hardware probe is stale")
        started = dt.datetime.fromisoformat(payload["utc_started"].replace("Z", "+00:00"))
        finished = dt.datetime.fromisoformat(payload["utc_finished"].replace("Z", "+00:00"))
        age = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        if started > finished or age < 0 or age > budget / 1e9:
            raise HardwareGateError("native hardware probe UTC freshness is inconsistent")
        return payload
    except EvidenceError as exc:
        raise HardwareGateError("native hardware probe binding is invalid: " + str(exc)) from exc


def collect_native_hardware_probe(binding: Mapping[str, Any], *, policy: HardwarePolicy,
                                  previous: NativeHardwareProbe | None = None) -> NativeHardwareProbe:
    """Collect native preflight or a same-process, same-boot cursor continuation.

    Calling this function performs passive native queries only. A previous JSON
    report is not accepted. The first probe checks full current-boot kernel logs;
    a continuation rechecks its exact retained cursor and carries prior findings.
    """
    if type(policy) is not HardwarePolicy:
        raise HardwareGateError("native hardware probe requires a validated HardwarePolicy")
    bound = validate_run_binding(binding)
    prior = None
    if previous is not None:
        prior = _check_capability(previous, binding, max_age_seconds=policy.maximum_probe_age_seconds)
        if prior["policy"] != policy.as_dict():
            raise HardwareGateError("hardware policy changed across a monitoring interval")
    from . import training_v2b_evidence
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA, "collector": "training_v2b_hardware_native_v1",
        "collector_sources": {"hardware": file_reference(__file__),
                              "evidence": file_reference(training_v2b_evidence.__file__)},
        "run_id": bound["run_id"], "run_binding_sha256": bound["binding_sha256"],
        "policy": policy.as_dict(), "policy_sha256": canonical_sha256(policy.as_dict()),
        "phase": "interval" if prior else "preflight",
        "previous_probe_sha256": previous.sha256 if previous is not None else None,
        "prior_gate_blocked": bool(prior and prior["gate"]["status"] != "PASS"),
        "host": platform.node(), "boot_id": _boot_id(), "process": _process_identity(os.getpid()),
        "utc_started": _utc(), "monotonic_started_ns": time.monotonic_ns(),
        "host_kernel_release": platform.release(), "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "environment": {key: os.environ.get(key) for key in
                        ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "execution": {"passive_hardware_queries": True, "cuda_api_called_by_probe": False,
                      "cuda_context_created_by_probe": False, "gpu_kernel_executed_by_probe": False,
                      "hardware_setting_changed": False, "training_started": False},
        "gpu_error": None, "gpu": None,
    }
    try:
        payload["torch_distribution_version"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        payload["torch_distribution_version"] = None
    files = {
        "kernel_driver_version": _native_file(Path("/sys/module/nvidia/version")),
        "kernel_driver_description": _native_file(Path("/proc/driver/nvidia/version")),
        "boot_id": _native_file(_BOOT_PATH),
    }
    commands = {"gpu_xml": _run_native_command(["nvidia-smi", "-q", "-x"], trace_loader=True)}
    try:
        payload["gpu"] = parse_gpu_xml(_command_text(commands["gpu_xml"]),
                                       expected_gpu_uuid=policy.expected_gpu_uuid)
        for process in payload["gpu"]["processes"]:
            process["identity"] = (_process_identity(process["pid"]) if process["pid"] is not None
                                   else {"readable": False, "error": "invalid GPU PID"})
    except HardwareGateError as exc:
        payload["gpu_error"] = str(exc)
    payload["capability_inventory"] = _capability_inventory(commands)
    payload.update(native_files=files, rapl=_collect_rapl(), cpu_temperatures=_collect_cpu_temperatures())
    cursor = prior["kernel_health"].get("last_cursor") if prior else None
    if prior and not cursor:
        payload["kernel_health"] = {"readable": False, "error": "prior probe lacks an interpretable kernel cursor"}
    else:
        journal_argv = ["journalctl", "--no-pager", "-k", "--boot=" + payload["boot_id"].replace("-", ""),
                        "--output=json", "--show-cursor"]
        if cursor:
            journal_argv.append("--cursor=" + cursor)
        commands["kernel_journal"] = _run_native_command(journal_argv)
        try:
            payload["kernel_health"] = parse_kernel_journal(_command_text(commands["kernel_journal"]),
                                                            boot_id=payload["boot_id"],
                                                            expected_start_cursor=cursor)
            if prior:
                payload["kernel_health"]["prior_findings"] = (
                    prior["kernel_health"].get("prior_findings", []) + prior["kernel_health"].get("findings", []))
        except HardwareGateError as exc:
            payload["kernel_health"] = {"readable": False, "error": str(exc)}
    payload.update(commands=commands, boot_id_after=_boot_id(), utc_finished=_utc(),
                   monotonic_finished_ns=time.monotonic_ns())
    stops = _assess(payload)
    payload["gate"] = {"status": "BLOCKED" if stops else "PASS", "stop_reasons": stops,
                       "gpu_compute_authorized": False, "production_authorized": False,
                       "scientific_certified": False,
                       "compute_exclusivity_observed": (payload["gpu"] is not None
                           and payload["gpu"]["process_list_readable"]
                           and not any(row["type"] != "G" for row in payload["gpu"]["processes"])),
                       "future_interval_monitoring_required": True}
    return NativeHardwareProbe(_TOKEN, payload)


def require_native_hardware_admission(probe: NativeHardwareProbe, *, binding: Mapping[str, Any],
                                      expected_gpu_uuid: str, max_age_seconds: float = 30.0) -> dict[str, Any]:
    """Require fresh native admission before CUDA initialization, never a dict."""
    from .training_v2b_admission import (
        MonitoredHardwareAdmission, require_monitored_hardware_admission,
    )
    if type(probe) is MonitoredHardwareAdmission:
        return require_monitored_hardware_admission(
            probe, binding=binding, expected_gpu_uuid=expected_gpu_uuid,
            max_age_seconds=max_age_seconds,
        )
    payload = _check_capability(probe, binding, max_age_seconds=max_age_seconds)
    if payload["policy"]["expected_gpu_uuid"] != expected_gpu_uuid or not payload.get("gpu") or payload["gpu"]["uuid"] != expected_gpu_uuid:
        raise HardwareGateError("native hardware admission GPU UUID differs")
    if payload["policy_sha256"] != canonical_sha256(payload["policy"]):
        raise HardwareGateError("native hardware policy digest differs")
    stops = _assess(payload)
    if stops or payload["gate"]["status"] != "PASS":
        raise HardwareGateError("native GPU admission BLOCKED: " + ", ".join(dict.fromkeys(row["code"] for row in stops)))
    return payload
