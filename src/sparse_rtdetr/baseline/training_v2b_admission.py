"""Scoped native admission with a separate, fail-closed GPU watchdog.

Nothing observes hardware or changes its settings at import time. ``start`` is
an explicitly authorized operation: it checks native idle evidence, verifies
the frozen clock-setting evidence, and starts an independent process. Native
setter modes make exactly one 1500/1500 attempt; the externally acknowledged
administrator mode never sets or resets clocks. This is monitored admission,
never a claim that a locked-clock getter was available.
The watchdog outlives the CUDA worker and publishes success only after its
pidfd signals exit and native observations confirm GPU context release.
"""
from __future__ import annotations

import ctypes
import datetime as dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import select
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

from . import training_v2b_evidence as ev
from . import training_v2b_hardware as hw

_TOKEN = object()
_ANCHOR_MANIFEST_SHA = "3a0f6fd00b84daf535205831be48ac2fb43820bd512f85df0cf35f56e7bd98b0"
_AUTHORIZATION_SHA = "e6b45aa30fe81f4df1138c5eba3033be8feaf2ea9ee96e55602fffd889059b85"
_REPAIR_AUTHORIZATION_SHA = "c8b46cc3d91e941c32047806e41e518476b00ce8e1deb62cb58deb42e9869e75"
_RESOLUTION_896_AUTHORIZATION_SHA = "ece816c4bfc93b62c403a00986a738f853b86e91a05953531ab93c0998538819"
_RESOLUTION_EVIDENCE_AUTHORIZATION_SHA = "0ea553eef36857f9686fa8bad3ad2b708af9f22a83f8f194b4405157a6a3b055"
_EXTERNAL_ADMIN_ATTESTATION_SHA = "a286392394db454370c77ed738c1a5b0155ea63b89fca3cd941870cbfb0a8300"
_EXTERNAL_CLOCK_RECEIPT_SHA = "aed11ff98f0b8e5478cc757e6f5929e22405972e535c97f1c586d9ec5c51847c"
_EXTERNAL_ADMIN_MODE = "external_admin_acknowledged"
_EXTERNAL_CLOCK_PROVENANCE = "user_supplied_original_terminal+native_command_journal"
_EXTERNAL_TERMINAL_TRAILER = "Connection to 192.168.0.198 closed."
_EXTERNAL_REFERENCE_FIELDS = ("user_attestation", "terminal_transcript", "native_command_journal",
                              "native_journal_query", "prior_journal_tool_capture")
_SCOPES = {"paired_smoke": 600, "train_core_30epoch": 43_200}
_PROC = Path("/proc")
_EDAC = Path("/sys/devices/system/edac/mc")
_FRAME_LIMIT = 2 * 1024 * 1024
_DRIVER_EVENT = re.compile(
    r"\b(?:NVRM|nvidia(?:[-_][a-z0-9]+)?|GPU)\b.*(?:\breset(?:ting|ted)?\b|"
    r"\bunload(?:ing|ed)?\b|\breload(?:ing|ed)?\b|\bunbind(?:ing)?\b|"
    r"\bremov(?:ing|ed)\b|\bload(?:ing|ed)\b.*(?:driver|kernel module))|"
    r"\b(?:loading|loaded|unloading|unloaded|reloading|reloaded)\b.*\b(?:nvidia|NVRM)\b",
    re.IGNORECASE,
)


class MonitoredHardwareError(hw.HardwareGateError):
    """A native observation, authorized scope or monitoring deadline failed."""


class _ExternalSudoJournalError(MonitoredHardwareError):
    def __init__(self, reason: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.external_sudo_rejection = evidence


class _OwnerIdentityError(MonitoredHardwareError):
    def __init__(self, message: str, observation: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.owner_identity_observation = _json_copy(observation)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _json_copy(value: Any) -> Any:
    return ev.strict_json_loads(ev.canonical_json_bytes(value))


def _publish_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish complete fsynced bytes atomically, without replacing any path."""
    temporary = path.parent / ("." + path.name + "." + os.urandom(12).hex() + ".tmp")
    try:
        ev.write_exclusive_json(temporary, value)
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return ev.file_reference(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_reference(ref: Mapping[str, Any]) -> Any:
    return ev.strict_json_loads(ev._verify_reference(ref))


def _proc_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("pid", "ppid", "start_ticks", "uid_fields", "gid_fields", "cmdline_sha256")
    exe_keys = ("path", "sha256", "size_bytes", "owner_uid", "owner_gid", "mode_octal", "group_or_other_writable")
    return {**{k: row[k] for k in keys}, "cgroup": row["cgroup"]["utf8"],
            "executable": {k: row["executable"][k] for k in exe_keys}}


def build_policy_bundle(reference_dir: str | os.PathLike[str], *, setter_mode: str = "direct",
                        authorization_reference: Mapping[str, Any],
                        external_clock_receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read the reviewed evidence anchor; this performs no hardware observation.

    A policy is an explicit declaration, not a caller-supplied observation. Its
    exception authority is the reviewed manifest, not arbitrary replacement JSON.
    """
    if setter_mode not in {"direct", "sudo_n", _EXTERNAL_ADMIN_MODE}:
        raise MonitoredHardwareError("setter mode must be explicitly direct, sudo_n or external_admin_acknowledged")
    if (setter_mode == _EXTERNAL_ADMIN_MODE) != (external_clock_receipt is not None):
        raise MonitoredHardwareError("external clock receipt is required only for external administrator mode")
    # Keep old declarations byte-for-byte reproducible. The repair authority
    # adds only a bounded synthetic diagnostic, never a hardware exception.
    authorizations = {
        _AUTHORIZATION_SHA: _SCOPES,
        _REPAIR_AUTHORIZATION_SHA: {**_SCOPES, "synthetic_operator_diagnostic": 600},
        # Native admission binds 896 dimensions and external clock evidence; no new hardware scope.
        _RESOLUTION_896_AUTHORIZATION_SHA: _SCOPES,
        # Only the new evidence authority adds bounded development evaluation.
        _RESOLUTION_EVIDENCE_AUTHORIZATION_SHA: {**_SCOPES, "development_cross_eval": 1800},
    }
    if type(authorization_reference) is not dict or type(authorization_reference.get("sha256")) is not str:
        raise MonitoredHardwareError("explicit execution authorization reference differs")
    scope_limits = authorizations.get(authorization_reference["sha256"])
    if scope_limits is None or ev.file_reference(authorization_reference["path"]) != authorization_reference:
        raise MonitoredHardwareError("explicit execution authorization reference differs")
    if (authorization_reference["sha256"] == _RESOLUTION_896_AUTHORIZATION_SHA
            and setter_mode != _EXTERNAL_ADMIN_MODE):
        raise MonitoredHardwareError("896 authorization requires external_admin_acknowledged; native clock setters are not authorized")
    if (authorization_reference["sha256"] == _RESOLUTION_EVIDENCE_AUTHORIZATION_SHA
            and setter_mode != _EXTERNAL_ADMIN_MODE):
        raise MonitoredHardwareError("resolution evidence authorization requires external_admin_acknowledged; native clock setters are not authorized")
    root = Path(reference_dir).absolute()
    manifest_ref = ev.file_reference(root / "manifest.json")
    if manifest_ref["sha256"] != _ANCHOR_MANIFEST_SHA:
        raise MonitoredHardwareError("unreviewed exception authority manifest")
    manifest = _read_reference(manifest_ref)
    names = ("current-hardware.json", "process-identities.json", "xorg-logind-session.json",
             "independent-review-addendum.json", "hardware-log-policy-review.json",
             "clock-evidence-protocol.json", "known-boot-bert-records.json", "xorg-identity-policy.json")
    refs = {name: manifest["files"][name] for name in names}
    for name, ref in refs.items():
        if Path(ref["path"]).absolute() != root / name:
            raise MonitoredHardwareError("exception reference escaped reviewed directory")
    native, identities, logind = (_read_reference(refs[name]) for name in names[:3])
    for name in names[3:]:
        ev._verify_reference(refs[name])
    occupants = identities["processes"]
    if len(occupants) != 1 or occupants[0]["native"]["type"] != "G":
        raise MonitoredHardwareError("reviewed anchor must contain exactly one Xorg")
    row = occupants[0]
    xorg, parent = _proc_projection(row["metadata"]), _proc_projection(row["parent"])
    if (not row["metadata"].get("stable_during_observation")
            or not row["parent"].get("stable_during_observation")
            or xorg["ppid"] != parent["pid"]):
        raise MonitoredHardwareError("reviewed Xorg ancestry is incomplete")
    expected_logind = {**logind["expected_identity_fields"], "Active": "yes", "State": "active"}
    if any(logind["properties"].get(k) != v for k, v in expected_logind.items()):
        raise MonitoredHardwareError("reviewed logind identity differs")
    journal_rows = _journal_records(hw._command_text(native["commands"]["kernel_journal"]))
    by_cursor = {row["__CURSOR"]: row for row in journal_rows}
    bert_raw = {row["cursor"]: ev.canonical_sha256(by_cursor[row["cursor"]])
                for row in native["kernel_health"]["findings"]}
    anchor = by_cursor[native["kernel_health"]["last_cursor"]]
    result = {
        "schema_version": 1, "mode": "native_monitored_scope_v1", "authority": manifest_ref,
        "authority_files": refs, "authorization_reference": dict(authorization_reference),
        "authorized_scope_limits_seconds": dict(scope_limits), "setter_mode": setter_mode,
        "host": native["host"], "boot_id": native["boot_id"], "gpu_uuid": native["gpu"]["uuid"],
        "pci_bus_id": native["gpu"]["pci_bus_id"], "hardware_policy": native["policy"],
        "expected_driver_userspace_version": native["gpu"]["driver_userspace_version"],
        "expected_kernel_driver_version": hw._file_text(native["native_files"]["kernel_driver_version"]),
        "expected_nvml_library": native["capability_inventory"]["nvml_library"],
        "expected_nvidia_smi_executable": native["commands"]["gpu_xml"]["executable"],
        "clock_min_mhz": 1500, "clock_max_mhz": 1500,
        "minimum_preload_clock_samples": 3, "xorg_memory_max_mib": 16,
        "clock_period_seconds": .2, "clock_max_gap_seconds": 1.,
        "health_period_seconds": 1., "health_max_gap_seconds": 2.,
        "guardian_heartbeat_period_seconds": .1, "guardian_heartbeat_max_gap_seconds": 1.,
        "startup_deadline_seconds": 30.,
        "owner_exit_timeout_seconds": 10., "post_exit_observation_seconds": 2.,
        "minimum_loaded_clock_samples_smoke": 3,
        "loaded_utilization_min_percent": 10,
        "xorg": xorg, "xorg_parent": parent, "logind": expected_logind,
        "known_bert_records": native["kernel_health"]["findings"],
        "known_bert_raw_record_sha256": bert_raw,
        "kernel_anchor_cursor": native["kernel_health"]["last_cursor"],
        "kernel_anchor_record_sha256": ev.canonical_sha256(anchor),
        "edac_unavailable_accepted": True,
        "clock_readback_verified": False, "settings_after_exit": "keep_1500_1500",
    }
    if setter_mode == _EXTERNAL_ADMIN_MODE:
        receipt = _validate_external_clock_receipt(external_clock_receipt, result)
        result["external_clock_receipt"] = dict(external_clock_receipt)
        result["external_journal_executable"] = dict(_read_reference(receipt["native_journal_query"])["executable"])
    result["bundle_sha256"] = ev.canonical_sha256(result)
    return result


def _validate_policy(bundle: Mapping[str, Any], scope: str, deadline: int) -> dict[str, Any]:
    if type(bundle) is not dict:
        raise MonitoredHardwareError("policy bundle must be an exact reviewed declaration")
    expected = build_policy_bundle(Path(bundle["authority"]["path"]).parent,
                                   setter_mode=bundle["setter_mode"],
                                   authorization_reference=bundle["authorization_reference"],
                                   external_clock_receipt=bundle.get("external_clock_receipt"))
    if bundle != expected:
        raise MonitoredHardwareError("policy bundle differs from its reviewed authority or scope")
    scope_limits = expected["authorized_scope_limits_seconds"]
    if (type(scope) is not str or scope not in scope_limits
            or type(deadline) is not int or not 1 <= deadline <= scope_limits[scope]):
        raise MonitoredHardwareError("workload deadline or scope is not authorized")
    expected.update(authorized_scope=scope, workload_deadline_seconds=deadline,
                    minimum_loaded_clock_samples=3)
    expected["policy_sha256"] = ev.canonical_sha256(expected)
    return expected


def _validate_resolution_evidence_binding(binding: Mapping[str, Any], policy: Mapping[str, Any],
                                          scope: str) -> None:
    """Keep the new authority inside its explicit data, geometry and seed scopes.

    This check receives a metadata-validated binding before any bound source or
    dataset file is opened. Scientific qualification and replication freezes
    remain the responsibility of the separately bound workload contract.
    """
    config, data = binding["config"], binding["data"]
    size = config.get("input_size")
    if type(size) is not int or size not in {640, 896}:
        raise MonitoredHardwareError("resolution evidence authorization requires input_size 640 or 896")
    if scope == "development_cross_eval":
        if (config.get("scope") != "development_cross_eval"
                or config.get("authorization_reference") != policy["authorization_reference"]):
            raise MonitoredHardwareError("development evaluation scope and exact authorization must be bound")
        if (data.get("kind") != "manifest_binding_only"
                or set(data.get("manifests", {})) != {"development"}):
            raise MonitoredHardwareError("development evaluation requires only its development manifest")
        path = Path(data["manifests"]["development"]["path"])
        forbidden = {"train_core", "confirmatory", "test", "visdrone2019-det-test-dev",
                     "visdrone2019-det-test-challenge"}
        candidates = (path, path.resolve())
        if any(candidate.name != "development_manifest.json"
               or any(part.casefold() in forbidden or part.casefold().startswith("confirmatory_")
                      for part in candidate.parts) for candidate in candidates):
            raise MonitoredHardwareError("development manifest path crosses its permitted data role")
    else:
        if (scope not in {"paired_smoke", "train_core_30epoch"}
                or config.get("scope") != "train_core_runtime_engineering"
                or data.get("kind") != "train_core_runtime"):
            raise MonitoredHardwareError("seed replication requires the actual train_core runtime binding")
        dimensions = {"physical_batch_size": 8, "accumulation_steps": 2}
        if (type(config.get("seed")) is not int or config["seed"] not in {1, 2}
                or any(type(config.get(key)) is not int or config[key] != value
                       for key, value in dimensions.items())):
            raise MonitoredHardwareError("seed replication requires seed 1 or 2 and physical batch 8 accumulation 2")


def _strict_process(pid: int) -> dict[str, Any]:
    root = _PROC / str(pid)
    before = (root / "stat").read_text()
    exe = (root / "exe").resolve(strict=True)
    reference = ev.file_reference(exe)
    info = exe.stat()
    fields = {}
    for line in (root / "status").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    cmdline = (root / "cmdline").read_bytes()
    cgroup = (root / "cgroup").read_text()
    after = (root / "stat").read_text()
    first, last = before.rsplit(")", 1)[1].split(), after.rsplit(")", 1)[1].split()
    if (first[19] != last[19] or first[1] != last[1]
            or (root / "exe").resolve(strict=True) != exe or ev.file_reference(exe) != reference):
        raise MonitoredHardwareError("process identity changed during observation")
    result = {"pid": pid, "ppid": int(first[1]), "start_ticks": int(first[19]),
              "uid_fields": [int(v) for v in fields["Uid"].split()],
              "gid_fields": [int(v) for v in fields["Gid"].split()],
              "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(), "cgroup": cgroup,
              "executable": {**{k: reference[k] for k in ("path", "sha256", "size_bytes")},
                             "owner_uid": info.st_uid, "owner_gid": info.st_gid,
                             "mode_octal": oct(stat.S_IMODE(info.st_mode)),
                             "group_or_other_writable": bool(info.st_mode & 0o022)}}
    return result


def _command(argv: list[str], *, timeout: float, trace_loader: bool = False,
             allow_failed_receipt: bool = False) -> dict[str, Any]:
    """Only internal fixed argv call sites; no shell, injection, retry or fallback."""
    start = time.monotonic_ns()
    utc_started = _utc()
    executable = Path(shutil.which(argv[0]) or argv[0]).resolve(strict=True)
    ref = ev.file_reference(executable)
    env = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}
    if trace_loader:
        env["LD_DEBUG"] = "libs"
    proc = subprocess.Popen([str(executable), *argv[1:]], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    error = None
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()  # This exact query/setter subprocess only; never a GPU occupant.
        out, err = proc.communicate()
        error = "native command deadline exceeded"
    if len(out) > 8 * 1024 * 1024 or len(err) > 8 * 1024 * 1024:
        raise MonitoredHardwareError("native capture exceeds evidence limit")
    record = {"argv": [str(executable), *argv[1:]], "executable": ref,
              "returncode": proc.returncode, "error": error, "truncated": False,
              "loader_trace": trace_loader, "stdout": hw._text_capture(out), "stderr": hw._text_capture(err),
              "started_ns": start, "finished_ns": time.monotonic_ns(),
              "utc_started": utc_started, "utc_finished": _utc()}
    if ev.file_reference(executable) != ref:
        raise MonitoredHardwareError("native executable changed during query")
    if (error or proc.returncode != 0) and not allow_failed_receipt:
        raise MonitoredHardwareError("native command failed: " + json.dumps(record, sort_keys=True))
    return record


def _xorg_identity(policy: Mapping[str, Any]) -> dict[str, Any]:
    xorg = _strict_process(policy["xorg"]["pid"])
    parent = _strict_process(policy["xorg_parent"]["pid"])
    if xorg != policy["xorg"] or parent != policy["xorg_parent"]:
        raise MonitoredHardwareError("Xorg or direct-parent identity changed")
    keys = list(policy["logind"])
    argv = ["loginctl", "show-session", policy["logind"]["Id"], "--no-pager"]
    for key in keys:
        argv += ["-p", key]
    capture = _command(argv, timeout=.5)
    properties = {}
    for line in hw._command_text(capture).splitlines():
        if "=" not in line:
            raise MonitoredHardwareError("malformed logind response")
        key, value = line.split("=", 1)
        if key in properties:
            raise MonitoredHardwareError("duplicate logind field")
        properties[key] = value
    if properties != policy["logind"]:
        raise MonitoredHardwareError("Xorg logind session changed")
    return {"xorg": xorg, "parent": parent, "logind": properties, "logind_command": capture,
            "grandparent_executable": "unavailable_not_required_by_explicit_policy"}


def _edac_observation() -> dict[str, Any]:
    rows = []
    for path in sorted(_EDAC.glob("mc[0-9]*")):
        if not re.fullmatch(r"mc[0-9]+", path.name):
            continue
        item = {"controller": path.name}
        for name in ("ce_count", "ue_count"):
            text = (path / name).read_text().strip()
            if not text.isdecimal():
                raise MonitoredHardwareError("EDAC counter is unreadable")
            item[name] = int(text)
        rows.append(item)
    return {"controller_counters_available": bool(rows), "controllers": rows,
            "counter_delta_available": False,  # This function returns a single observation.
            "unavailable_means": "unknown_not_zero" if not rows else None}


def _check_edac(current: Mapping[str, Any], previous: Mapping[str, Any] | None) -> None:
    if previous is not None and current != previous:
        raise MonitoredHardwareError("EDAC counters or observation coverage changed")
    if any(row[name] != 0 for row in current["controllers"] for name in ("ce_count", "ue_count")):
        raise MonitoredHardwareError("nonzero EDAC error counter")


def _assess_monitored(payload: Mapping[str, Any], policy: Mapping[str, Any], *,
                      owner: Mapping[str, Any] | None, initial: bool) -> None:
    if payload["host"] != policy["host"] or payload["boot_id"] != policy["boot_id"]:
        raise MonitoredHardwareError("reviewed host/boot exception expired")
    if not payload.get("gpu") or payload["gpu"]["pci_bus_id"] != policy["pci_bus_id"]:
        raise MonitoredHardwareError("GPU physical identity changed")
    if (payload["gpu"]["driver_userspace_version"] != policy["expected_driver_userspace_version"]
            or hw._file_text(payload["native_files"]["kernel_driver_version"]) != policy["expected_kernel_driver_version"]
            or payload["capability_inventory"]["nvml_library"] != policy["expected_nvml_library"]):
        raise MonitoredHardwareError("reviewed driver or NVML identity changed")
    if initial and payload["commands"]["gpu_xml"]["executable"] != policy["expected_nvidia_smi_executable"]:
        raise MonitoredHardwareError("reviewed nvidia-smi executable identity changed")
    health = payload["kernel_health"]
    known = policy["known_bert_records"]
    actual = health.get("findings", [])
    if initial:
        if actual != known or health.get("coverage") != "full_current_boot":
            raise MonitoredHardwareError("full-boot findings differ from exact BERT exception")
    elif actual or health.get("prior_findings"):
        raise MonitoredHardwareError("new kernel hardware/health event")
    processes = payload["gpu"]["processes"]
    expected_xorg = policy["xorg"]
    graphics = [p for p in processes if p["type"] == "G"]
    if len(graphics) != 1:
        raise MonitoredHardwareError("expected exactly the approved Xorg occupant")
    xorg = graphics[0]
    memory = xorg.get("memory_mib")
    if type(memory) not in (int, float) or not math.isfinite(memory) or not 0 <= memory <= policy["xorg_memory_max_mib"]:
        raise MonitoredHardwareError("Xorg memory is unreadable or exceeds its 16 MiB allowance")
    if (xorg["pid"] != expected_xorg["pid"] or not xorg["identity"].get("readable")
            or xorg["identity"]["start_ticks"] != expected_xorg["start_ticks"]
            or xorg["identity"]["executable"]["sha256"] != expected_xorg["executable"]["sha256"]):
        raise MonitoredHardwareError("native GPU Xorg identity differs")
    for process in processes:
        if process is xorg:
            continue
        if owner is None or process["type"] != "C" or process.get("identity") != owner:
            raise MonitoredHardwareError("unapproved GPU compute/graphics occupant")
    assessed = _json_copy(payload)
    assessed["policy"]["allowed_graphics_executable_sha256"] = [expected_xorg["executable"]["sha256"]]
    assessed["kernel_health"]["findings"] = []  # Only after exact full-row/empty-delta checks.
    assessed["kernel_health"]["prior_findings"] = []
    assessed["prior_gate_blocked"] = False  # Original strict result remains unchanged in persisted evidence.
    if owner is not None:
        assessed["process"] = dict(owner)
        assessed["phase"] = "interval"
    reasons = [r for r in hw._assess(assessed) if r["code"] != "GRAPHICS_CLOCK_CAP_UNVERIFIED"]
    if reasons:
        raise MonitoredHardwareError("native monitored preconditions failed: " + json.dumps(reasons))


def _journal_records(text: str) -> list[dict[str, Any]]:
    return [ev.strict_json_loads(line.encode()) for line in text.splitlines() if line.startswith("{")]


def _reject_driver_events(rows: list[dict[str, Any]]) -> None:
    if any(_DRIVER_EVENT.search(row["MESSAGE"]) for row in rows):
        raise MonitoredHardwareError("new driver reset/load/unload/reload event")


def _check_anchor(native: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    rows = _journal_records(hw._command_text(native["commands"]["kernel_journal"]))
    hits = [i for i, row in enumerate(rows) if row.get("__CURSOR") == policy["kernel_anchor_cursor"]]
    if len(hits) != 1:
        raise MonitoredHardwareError("reviewed journal anchor is not retained")
    if ev.canonical_sha256(rows[hits[0]]) != policy["kernel_anchor_record_sha256"]:
        raise MonitoredHardwareError("reviewed journal anchor record changed")
    for cursor, digest in policy["known_bert_raw_record_sha256"].items():
        records = [row for row in rows if row["__CURSOR"] == cursor]
        if len(records) != 1 or ev.canonical_sha256(records[0]) != digest:
            raise MonitoredHardwareError("reviewed complete BERT record changed")
    _reject_driver_events(rows[hits[0] + 1:])


def _setter(policy: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    if policy["setter_mode"] not in {"direct", "sudo_n"}:
        raise MonitoredHardwareError("external administrator mode cannot execute a native setter")
    binary = Path(shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi").resolve(strict=True)
    reference = ev.file_reference(binary)
    if reference != baseline["commands"]["gpu_xml"]["executable"]:
        raise MonitoredHardwareError("setter differs from observed native GPU executable")
    argv = [str(binary), "--id=" + policy["gpu_uuid"], "--lock-gpu-clocks=1500,1500"]
    direct = policy["setter_mode"] == "direct"
    if not direct:
        argv = ["sudo", "-n", "--", *argv]
    capture = _command(argv, timeout=3., trace_loader=direct, allow_failed_receipt=True)
    receipt = {"schema_version": 1, "setter_mode": policy["setter_mode"], "command": capture,
               "utc_started": capture.get("utc_started"), "utc_finished": capture.get("utc_finished"),
               "nvidia_smi_executable": reference, "gpu_uuid": policy["gpu_uuid"],
               "boot_id": hw._boot_id(), "clock_min_mhz": 1500, "clock_max_mhz": 1500,
               "native_return_success": capture["returncode"] == 0 and capture["error"] is None,
               "validation_errors": [], "locked_upper_readback_verified": False,
               "nvml_query_library": baseline["capability_inventory"]["nvml_library"],
               "setter_loaded_nvml_library": None, "no_retry_or_fallback": True}
    try:
        stdout = hw._command_text(capture)
        receipt["acknowledgement"] = _setter_acknowledgement(stdout, policy)
    except hw.HardwareGateError as exc:
        receipt["validation_errors"].append("setter acknowledgement/diagnostic rejected: " + str(exc))
    if direct:
        stderr = capture["stderr"].get("utf8") or ""
        candidates = set(re.findall(r"calling init:\s*(/[^\s]*libnvidia-ml\.so[^\s]*)", stderr))
        if len(candidates) != 1:
            receipt["validation_errors"].append("setter loaded-NVML identity unavailable")
        else:
            receipt["setter_loaded_nvml_library"] = ev.file_reference(Path(candidates.pop()).resolve(strict=True))
            if receipt["setter_loaded_nvml_library"] != receipt["nvml_query_library"]:
                receipt["validation_errors"].append("setter NVML differs from native query library")
    else:
        receipt["setter_library_scope"] = "sudo_secure_loader_not_traced; executable_and_pre_post_query_library_bound"
    if receipt["boot_id"] != policy["boot_id"]:
        receipt["validation_errors"].append("boot changed across setter")
    if not receipt["native_return_success"]:
        receipt["validation_errors"].append("native setter did not return success")
    return receipt


def _setter_acknowledgement(text: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 2 or lines[1] != "All done.":
        raise MonitoredHardwareError("setter acknowledgement is incomplete, ambiguous or has diagnostics")
    match = re.fullmatch(
        r'GPU clocks set to "\((?:gpuClkMin ([0-9]+), gpuClkMax ([0-9]+)|\s*([0-9]+),\s*([0-9]+))\)" '
        r'for GPU ([a-zA-Z0-9:.\-]+)', lines[0])
    if match is None:
        raise MonitoredHardwareError("unrecognized setter stdout or diagnostic")
    low, high = (int(match[1]), int(match[2])) if match[1] is not None else (int(match[3]), int(match[4]))
    target = match[5]
    if ((low, high) != (1500, 1500)
            or target.lower() not in {policy["gpu_uuid"].lower(), policy["pci_bus_id"].lower()}):
        raise MonitoredHardwareError("setter acknowledgement target/range differs")
    return {"target": target, "min_mhz": low, "max_mhz": high,
            "meaning": "native_setter_acknowledgement_not_locked_clock_readback"}


def _strict_journal_jsonl(raw: bytes) -> list[dict[str, Any]]:
    """The external journal has no cursor trailer or discarded diagnostic lines."""
    try:
        rows = [ev.strict_json_loads(line) for line in raw.splitlines()]
    except ev.EvidenceError as exc:
        raise MonitoredHardwareError("external command journal is not complete JSONL") from exc
    if not rows or any(type(row) is not dict for row in rows):
        raise MonitoredHardwareError("external command journal is empty or malformed")
    return rows


def _validate_external_clock_receipt(reference: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Verify reviewed bytes, not a caller-declared success or a locked-cap getter.

    These historical records do not grant admission by themselves. Every start
    still collects native idle/identity evidence and waits for preload samples;
    the independent guardian requires actual loaded samples and unchanged gaps.
    """
    ev._validate_reference_shape(reference)
    if (reference.get("sha256") != _EXTERNAL_CLOCK_RECEIPT_SHA
            or ev.file_reference(reference["path"]) != reference):
        raise MonitoredHardwareError("external clock receipt differs from reviewed complete bytes")
    receipt = _read_reference(reference)
    fixed = {
        "schema_version": 1, "kind": "reviewed_external_admin_clock_receipt",
        "provenance": _EXTERNAL_CLOCK_PROVENANCE, "locally_executed_setter": False,
        "native_exitcode": None, "native_exit_code": None, "native_return_success": None,
        "native_journal_query_returncode_is_setter_exitcode": False,
        "reported_acknowledgement_success": True, "acknowledgement_verified": True,
        "success_ack_observed": True, "locked_upper_readback_verified": False,
        "stdout_source": "user_supplied_terminal", "time_source": "native_sudo_journal_record",
        "separate_stderr_capture_available": False, "setter_loaded_nvml_library": None,
        "transport_line": _EXTERNAL_TERMINAL_TRAILER, "clock_min_mhz": 1500, "clock_max_mhz": 1500,
    }
    required = set(fixed) | set(_EXTERNAL_REFERENCE_FIELDS) | {
        "host", "boot_id", "gpu_uuid", "pci_bus_id", "command_argv", "command_journal_record_sha256",
        "native_sudo_pid", "native_command_realtime_timestamp_us"}
    if type(receipt) is not dict or set(receipt) != required:
        raise MonitoredHardwareError("external clock receipt schema differs from reviewed scope")
    for name, expected in fixed.items():
        if type(receipt[name]) is not type(expected) or receipt[name] != expected:
            raise MonitoredHardwareError("external clock receipt provenance/semantics differs: " + name)
    for name in ("host", "boot_id", "gpu_uuid", "pci_bus_id"):
        if receipt[name] != policy[name]:
            raise MonitoredHardwareError("external clock receipt identity differs: " + name)
    raw = {}
    for name in _EXTERNAL_REFERENCE_FIELDS:
        ref = receipt[name]
        raw[name] = ev._verify_reference(ref)
        if ev.file_reference(ref["path"]) != ref:
            raise MonitoredHardwareError("external evidence reference is not canonical: " + name)
    if receipt["user_attestation"]["sha256"] != _EXTERNAL_ADMIN_ATTESTATION_SHA:
        raise MonitoredHardwareError("external administrator authorization is not the reviewed user statement")
    try:
        terminal = raw["terminal_transcript"].decode("utf-8")
    except UnicodeError as exc:
        raise MonitoredHardwareError("external administrator terminal is not UTF8") from exc
    terminal_lines = terminal.splitlines()
    if len(terminal_lines) != 3 or terminal_lines[-1] != _EXTERNAL_TERMINAL_TRAILER:
        raise MonitoredHardwareError("external terminal must contain exact success lines and reviewed SSH trailer")
    _setter_acknowledgement("\n".join(terminal_lines[:-1]) + "\n", policy)
    argv = [policy["expected_nvidia_smi_executable"]["path"], "-i", policy["gpu_uuid"],
            "--lock-gpu-clocks=1500,1500"]
    if receipt["command_argv"] != argv:
        raise MonitoredHardwareError("external administrator native command target/range differs")
    pid = receipt["native_sudo_pid"]
    if type(pid) is not int or pid <= 0:
        raise MonitoredHardwareError("external sudo PID is unavailable")
    rows = _strict_journal_jsonl(raw["native_command_journal"])
    if len(rows) != 3:
        raise MonitoredHardwareError("external sudo command/open/close journal chain is incomplete")
    command, opened, closed = rows
    if ev.canonical_sha256(command) != receipt["command_journal_record_sha256"]:
        raise MonitoredHardwareError("external sudo command record differs from selected complete record")
    identity = {"_HOSTNAME": policy["host"], "_BOOT_ID": policy["boot_id"].replace("-", ""),
                "_PID": str(pid), "_UID": "1000", "_AUDIT_LOGINUID": "1000",
                "_EXE": "/usr/bin/sudo", "_COMM": "sudo", "SYSLOG_IDENTIFIER": "sudo",
                "_CMDLINE": "sudo " + " ".join(argv)}
    for row in rows:
        if any(row.get(k) != v for k, v in identity.items()):
            raise MonitoredHardwareError("external sudo journal native process/boot/command identity differs")
    if [row.get("_GID") for row in rows] != ["1000", "0", "0"]:
        raise MonitoredHardwareError("external sudo journal reviewed setuid/group transition differs")
    for key in ("_AUDIT_SESSION", "_SYSTEMD_SESSION", "_SYSTEMD_INVOCATION_ID", "_MACHINE_ID"):
        if type(command.get(key)) is not str or not command[key] or any(row.get(key) != command[key] for row in rows):
            raise MonitoredHardwareError("external sudo journal native session identity differs")
    if command["_AUDIT_SESSION"] != command["_SYSTEMD_SESSION"]:
        raise MonitoredHardwareError("external sudo audit and logind sessions differ")
    pattern = r"[ \t]*lyy : TTY=[^;\r\n]+ ; PWD=[^;\r\n]+ ; USER=root ; COMMAND=" + re.escape(" ".join(argv))
    if re.fullmatch(pattern, command.get("MESSAGE", "")) is None:
        raise MonitoredHardwareError("external sudo journal command did not target root with the exact setter")
    if (opened.get("MESSAGE") != "pam_unix(sudo:session): session opened for user root(uid=0) by lyy(uid=1000)"
            or closed.get("MESSAGE") != "pam_unix(sudo:session): session closed for user root"):
        raise MonitoredHardwareError("external sudo journal privileged session acknowledgement is incomplete")
    for key in ("__REALTIME_TIMESTAMP", "__MONOTONIC_TIMESTAMP"):
        values = [row.get(key) for row in rows]
        if any(type(v) is not str or not v.isdecimal() or int(v) <= 0 for v in values):
            raise MonitoredHardwareError("external sudo journal timestamps are unavailable")
        if not int(values[0]) < int(values[1]) < int(values[2]):
            raise MonitoredHardwareError("external sudo journal command/session ordering differs")
    if receipt["native_command_realtime_timestamp_us"] != command["__REALTIME_TIMESTAMP"]:
        raise MonitoredHardwareError("external command timestamp differs from its native source")
    query = ev.strict_json_loads(raw["native_journal_query"])
    query_executable = query.get("executable", {})
    if (query.get("argv") != [query_executable.get("path"), "-b", "_PID=" + str(pid), "--output=json", "--no-pager"]
            or Path(query_executable.get("path", "")).name != "journalctl"
            or hw._command_text(query).encode("utf-8") != raw["native_command_journal"]
            or query.get("stdout") != hw._text_capture(raw["native_command_journal"])
            or query.get("stderr") != hw._text_capture(b"")):
        raise MonitoredHardwareError("native journal query capture differs from the complete JSONL")
    ev._verify_reference(query_executable)
    prior = ev.strict_json_loads(raw["prior_journal_tool_capture"])
    if prior.get("status") != "fulfilled" or prior.get("value", {}).get("exit_code") != 0:
        raise MonitoredHardwareError("prior journal tool capture was incomplete")
    prior_output = prior["value"].get("output")
    if type(prior_output) is not str:
        raise MonitoredHardwareError("prior journal tool output is unavailable")
    prior_lines = prior_output.splitlines()
    if (not prior_lines or prior_lines[-1] != "-- cursor: " + closed["__CURSOR"]
            or _strict_journal_jsonl("\n".join(prior_lines[:-1]).encode("utf-8")) != rows):
        raise MonitoredHardwareError("prior and current native journal records differ")
    return receipt


def _external_clock_evidence(policy: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """A per-start review receipt: deliberately no call to _setter or _command."""
    reviewed = _validate_external_clock_receipt(policy["external_clock_receipt"], policy)
    if (baseline["boot_id"] != reviewed["boot_id"] or baseline["host"] != reviewed["host"]
            or baseline["gpu"]["uuid"] != reviewed["gpu_uuid"]
            or baseline["gpu"]["pci_bus_id"] != reviewed["pci_bus_id"]
            or baseline["commands"]["gpu_xml"]["executable"] != policy["expected_nvidia_smi_executable"]):
        raise MonitoredHardwareError("live hardware differs from reviewed external clock-setting identity")
    journal = _strict_journal_jsonl(ev._verify_reference(reviewed["native_command_journal"]))
    return {"schema_version": 1, "setter_mode": _EXTERNAL_ADMIN_MODE,
            "external_clock_receipt": dict(policy["external_clock_receipt"]),
            "locally_executed_setter": False, "command": None,
            "native_exitcode": None, "native_exit_code": None, "native_return_success": None,
            "reported_acknowledgement_success": True, "acknowledgement_verified": True,
            "success_ack_observed": True, "provenance": _EXTERNAL_CLOCK_PROVENANCE,
            "stdout_source": "user_supplied_terminal", "validation_errors": [],
            "reviewed_at_utc": _utc(), "time_source": "native_sudo_journal_record",
            "native_command_realtime_timestamp_us": reviewed["native_command_realtime_timestamp_us"],
            "native_sudo_journal_anchor": {"cursor": journal[0]["__CURSOR"],
                                            "record_sha256": reviewed["command_journal_record_sha256"]},
            "gpu_uuid": reviewed["gpu_uuid"], "pci_bus_id": reviewed["pci_bus_id"], "boot_id": reviewed["boot_id"],
            "clock_min_mhz": 1500, "clock_max_mhz": 1500, "locked_upper_readback_verified": False,
            "nvidia_smi_executable": baseline["commands"]["gpu_xml"]["executable"],
            "nvml_query_library": baseline["capability_inventory"]["nvml_library"],
            "setter_loaded_nvml_library": None, "no_retry_or_fallback": True,
            "settings_after_exit": "keep_1500_1500"}


def _external_clock_integrity_references(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipt = _validate_external_clock_receipt(policy["external_clock_receipt"], policy)
    query = _read_reference(receipt["native_journal_query"])
    return [dict(policy["external_clock_receipt"]),
            *(dict(receipt[name]) for name in _EXTERNAL_REFERENCE_FIELDS), dict(query["executable"])]


def _known_sudo_clock_mutation(row: Mapping[str, Any]) -> bool:
    """Recognize direct nvidia-smi mutations in trusted native sudo records.

    This does not claim visibility of privileged writes without sudo or opaque
    shell wrappers, and never treats arbitrary journal text as a command.
    """
    if row.get("_EXE") != "/usr/bin/sudo" or row.get("_COMM") != "sudo":
        return False
    message = row.get("MESSAGE")
    if type(message) is not str:
        raise MonitoredHardwareError("sudo journal message is unavailable")
    match = re.search(r"(?:^|;[ \t]*)COMMAND=([^\r\n]+)$", message)
    if match is None:
        return False  # PAM/session records are retained, but are not commands.
    try:
        argv = shlex.split(match[1], posix=True)
    except ValueError as exc:
        raise MonitoredHardwareError("native sudo command cannot be parsed") from exc
    if not argv or Path(argv[0]).name != "nvidia-smi":
        return False
    mutations = {"-lgc", "--lock-gpu-clocks", "-rgc", "--reset-gpu-clocks",
                 "-ac", "--applications-clocks", "-rac", "--reset-applications-clocks",
                 "-r", "--gpu-reset", "-lmc", "--lock-memory-clocks", "-rmc", "--reset-memory-clocks",
                 "-lmcd", "--lock-memory-clocks-deferred", "-rmcd", "--reset-memory-clocks-deferred"}
    return any(arg.split("=", 1)[0] in mutations for arg in argv[1:])


def _external_sudo_journal_sample(policy: Mapping[str, Any], anchor: Mapping[str, Any], *,
                                  evidence_dir: Path | None = None) -> dict[str, Any]:
    """Retain the anchor and raw incremental sudo log with unchanged deadlines."""
    capture = None
    try:
        executable = policy["external_journal_executable"]
        capture = _command([executable["path"], "--no-pager", "--boot=" + policy["boot_id"].replace("-", ""),
                            "--output=json", "--show-cursor", "--cursor=" + anchor["cursor"],
                            "_COMM=sudo", "_EXE=/usr/bin/sudo"], timeout=.7)
        if capture.get("executable") != executable:
            raise MonitoredHardwareError("external sudo journal collector executable identity differs")
        lines = hw._command_text(capture).splitlines()
        if len(lines) < 2 or not lines[-1].startswith("-- cursor: "):
            raise MonitoredHardwareError("external sudo journal cursor/coverage is unavailable")
        rows = _strict_journal_jsonl("\n".join(lines[:-1]).encode("utf-8"))
        if (rows[0].get("__CURSOR") != anchor["cursor"]
                or ev.canonical_sha256(rows[0]) != anchor["record_sha256"]):
            raise MonitoredHardwareError("external sudo journal retained anchor differs")
        if lines[-1] != "-- cursor: " + rows[-1].get("__CURSOR", ""):
            raise MonitoredHardwareError("external sudo journal terminal cursor differs")
        cursors, monotonic = [], []
        for row in rows:
            if (row.get("_BOOT_ID") != policy["boot_id"].replace("-", "")
                    or row.get("_HOSTNAME") != policy["host"]
                    or row.get("_EXE") != "/usr/bin/sudo" or row.get("_COMM") != "sudo"):
                raise MonitoredHardwareError("external sudo journal native source/boot differs")
            cursor, timestamp = row.get("__CURSOR"), row.get("__MONOTONIC_TIMESTAMP")
            if (type(cursor) is not str or not cursor or type(timestamp) is not str
                    or not timestamp.isdecimal() or int(timestamp) <= 0):
                raise MonitoredHardwareError("external sudo journal event identity/time is unavailable")
            cursors.append(cursor)
            monotonic.append(int(timestamp))
        if len(set(cursors)) != len(cursors) or any(a > b for a, b in zip(monotonic, monotonic[1:])):
            raise MonitoredHardwareError("external sudo journal event order/multiplicity differs")
        # Only the exact retained starting event is exempt. A later repeated
        # 1500/1500 setter also invalidates this fixed external setting event.
        if any(_known_sudo_clock_mutation(row) for row in rows[1:]):
            raise MonitoredHardwareError("external clock receipt invalidated by later sudo clock/reset command")
        return {"query": capture, "anchor": dict(anchor), "last_cursor": rows[-1]["__CURSOR"],
                "last_record_sha256": ev.canonical_sha256(rows[-1]), "records": len(rows),
                "known_later_clock_mutations": 0, "coverage": "retained_anchor_through_current_native_sudo",
                "unobserved_privileged_write_paths": ["without_sudo", "opaque_shell_wrappers"],
                "locked_upper_readback_verified": False}
    except (hw.HardwareGateError, ev.EvidenceError) as exc:
        rejection = {"error": type(exc).__name__ + ": " + str(exc), "native_query": capture,
                     "anchor": dict(anchor), "recorded_utc": _utc(), "no_retry_or_fallback": True}
        if evidence_dir is not None:
            # Startup only: there is no admitted CUDA worker yet.
            ev.write_exclusive_json(evidence_dir / "external-clock-journal-rejection.json", rejection)
        # During runtime the guardian first stops its exact worker, then writes
        # this capture. Failure evidence I/O cannot delay the stop notification.
        raise _ExternalSudoJournalError(str(exc), rejection) from exc


def _clock_sample(spec: Mapping[str, Any]) -> dict[str, Any]:
    policy = spec["policy"]
    capture = _command([spec["nvidia_smi_path"], "--id=" + policy["gpu_uuid"],
                        "--query-gpu=uuid,clocks.current.graphics,clocks.current.sm,utilization.gpu,memory.used,memory.free",
                        "--format=csv,noheader,nounits"], timeout=.7)
    if capture.get("executable") != policy["expected_nvidia_smi_executable"]:
        raise MonitoredHardwareError("clock sampler executable differs from policy")
    fields = [x.strip() for x in hw._command_text(capture).strip().split(",")]
    if len(fields) != 6 or fields[0] != policy["gpu_uuid"]:
        raise MonitoredHardwareError("clock sample identity/schema differs")
    values = [hw._value(v) for v in fields[1:]]
    if any(v is None or not math.isfinite(v) for v in values):
        raise MonitoredHardwareError("clock sample contains unavailable telemetry")
    clock, sm_clock, utilization, used, free = values
    if clock > 1500 or sm_clock > 1500 or utilization > 100:
        raise MonitoredHardwareError("graphics/SM clock exceeds 1500 MHz or utilization is invalid")
    if free < policy["hardware_policy"]["interval_min_free_memory_mib"]:
        raise MonitoredHardwareError("GPU interval memory reserve exhausted")
    return {"kind": "clock", "started_ns": capture["started_ns"], "finished_ns": capture["finished_ns"],
            "clock_mhz": clock, "graphics_clock_mhz": clock, "sm_clock_mhz": sm_clock,
            "utilization_percent": utilization,
            "memory_used_mib": used, "memory_free_mib": free, "command": capture}



_OWNER_EXIT_IDENTITY_WAIT_SECONDS = .2
_SAMPLER_SHUTDOWN_SECONDS = 1.


def _disappearing_owner_identity(value: Any, owner: Mapping[str, Any]) -> bool:
    # _process_identity also encodes proven replacement/integrity failures as
    # unreadable. Only a missing process/proc entry may await the original pidfd.
    return (type(value) is dict and set(value) == {"pid", "readable", "error"}
            and type(value["pid"]) is int and value["pid"] == owner["pid"]
            and value["readable"] is False and type(value["error"]) is str
            and value["error"].startswith(("FileNotFoundError: ", "ProcessLookupError: "))
            and any(repr("/proc/" + str(owner["pid"]) + "/" + leaf) in value["error"]
                    for leaf in ("stat", "exe")))


def _wait_owner_pidfd(pidfd: int, timeout_seconds: float) -> tuple:
    """One event wait on the already bound fd; never reopen or retry by PID."""
    return select.select([pidfd], [], [], timeout_seconds)


def _observe_owner_identity(spec: Mapping[str, Any], query: Mapping[str, Any],
                            processes: list[dict[str, Any]], *, health_started_ns: int,
                            owner_alive: bool) -> dict[str, Any]:
    """Resolve only disappearing /proc identity, inside the health sampler budget."""
    observation = {
        "schema_version": 1, "kind": "owner_identity_observation",
        "run_id": spec.get("run_id"), "run_binding_sha256": spec.get("binding_sha256"),
        "gpu_uuid": spec["policy"]["gpu_uuid"], "owner": _json_copy(spec["owner"]),
        "health_started_ns": health_started_ns, "owner_alive_input": owner_alive,
        "original_pidfd": spec["_owner_pidfd"], "gpu_query": _json_copy(query),
        "gpu_processes": _json_copy(processes), "observed_owner": None,
        "identity_read_started_ns": None, "identity_read_finished_ns": None,
        "pidfd_checks": [], "wait": None, "decision": "OBSERVING",
        "owner_alive_after": None, "completed_ns": None,
    }

    def publish() -> None:
        sink = spec.get("_owner_identity_observation_sink")
        if sink is not None:
            sink(_json_copy(observation))  # No lock is retained by the caller.

    def reject(message: str) -> None:
        observation.update(decision="REJECTED", failure=message, completed_ns=time.monotonic_ns())
        publish()
        raise _OwnerIdentityError(message, observation)

    def pidfd_check(stage: str) -> bool:
        row = {"stage": stage, "started_ns": time.monotonic_ns(),
               "finished_ns": None, "exited": None, "error": None}
        observation["pidfd_checks"].append(row)
        try:
            result = _pidfd_dead(spec["_owner_pidfd"])
        except BaseException as exc:
            row.update(finished_ns=time.monotonic_ns(), error=type(exc).__name__ + ": " + str(exc))
            reject("owner pidfd observation failed")
        row.update(finished_ns=time.monotonic_ns(), exited=result if type(result) is bool else None)
        if type(result) is not bool:
            reject("owner pidfd observation is malformed")
        return result

    publish()
    before = pidfd_check("before_identity_read")
    if type(owner_alive) is not bool:
        reject("owner liveness input is malformed")
    if not owner_alive:
        if not before:
            reject("original owner pidfd exit evidence regressed")
        observation.update(decision="EXIT_CONFIRMED", owner_alive_after=False,
                           completed_ns=time.monotonic_ns())
        publish()
        return observation

    observation["identity_read_started_ns"] = time.monotonic_ns()
    try:
        observed = hw._process_identity(spec["owner"]["pid"])
    except BaseException as exc:
        observation["identity_read_finished_ns"] = time.monotonic_ns()
        observation["identity_read_error"] = type(exc).__name__ + ": " + str(exc)
        reject("worker identity reader raised an exception")
    observation["identity_read_finished_ns"] = time.monotonic_ns()
    try:
        observation["observed_owner"] = _json_copy(observed)
    except (ev.EvidenceError, TypeError, ValueError):
        observation["observed_owner"] = {"non_json_python_type": type(observed).__module__ + "." + type(observed).__qualname__}
        reject("worker identity observation is not strict JSON")
    if (type(observed) is not dict or type(observed.get("readable")) is not bool
            or type(observed.get("pid")) is not int or observed["pid"] != spec["owner"]["pid"]):
        reject("worker identity observation is malformed")
    if observed["readable"]:
        # This rejection must precede every exit exemption, even an already-dead pidfd.
        if ev.canonical_json_bytes(observed) != ev.canonical_json_bytes(spec["owner"]):
            reject("worker process/executable identity changed")
    elif not _disappearing_owner_identity(observed, spec["owner"]):
        reject("unreadable worker identity is not an approved disappearing proc entry")
    after = pidfd_check("after_identity_read")
    if before and not after:
        reject("original owner pidfd exit evidence regressed")
    if not observed["readable"] and not after:
        began = time.monotonic_ns()
        wait = {"started_ns": began, "deadline_ns": began + int(_OWNER_EXIT_IDENTITY_WAIT_SECONDS * 1e9),
                "maximum_seconds": _OWNER_EXIT_IDENTITY_WAIT_SECONDS, "requested_timeout_seconds": None,
                "call_started_ns": None, "finished_ns": None, "elapsed_ns": None,
                "readable_fds": None, "writable_fds": None, "exceptional_fds": None,
                "error": None, "exit_confirmed": False}
        observation.update(decision="WAITING_FOR_ORIGINAL_PIDFD", wait=wait)
        publish()
        remaining = (wait["deadline_ns"] - time.monotonic_ns()) / 1e9
        if not 0 < remaining <= _OWNER_EXIT_IDENTITY_WAIT_SECONDS:
            reject("owner pidfd wait budget expired before the event wait")
        wait.update(requested_timeout_seconds=remaining, call_started_ns=time.monotonic_ns())
        try:
            ready = _wait_owner_pidfd(spec["_owner_pidfd"], remaining)
        except BaseException as exc:
            wait.update(finished_ns=time.monotonic_ns(), error=type(exc).__name__ + ": " + str(exc))
            wait["elapsed_ns"] = wait["finished_ns"] - began
            reject("owner pidfd event wait failed")
        wait.update(finished_ns=time.monotonic_ns())
        wait["elapsed_ns"] = wait["finished_ns"] - began
        if (type(ready) is not tuple or len(ready) != 3
                or any(type(items) is not list for items in ready)
                or any(type(fd) is not int for items in ready for fd in items)):
            reject("owner pidfd event wait returned a malformed result")
        wait.update(readable_fds=ready[0], writable_fds=ready[1], exceptional_fds=ready[2])
        if wait["finished_ns"] > wait["deadline_ns"]:
            reject("owner pidfd event wait exceeded its 0.2 second deadline")
        if ready != ([spec["_owner_pidfd"]], [], []):
            reject("owner pidfd event wait timed out or returned an unbound descriptor")
        after = pidfd_check("after_event_wait")
        if not after:
            reject("owner pidfd readiness did not confirm original process exit")
        wait["exit_confirmed"] = True
    observation.update(decision="EXIT_CONFIRMED" if after else "LIVE_OWNER_EXACT",
                       owner_alive_after=not after, completed_ns=time.monotonic_ns())
    publish()
    return observation


def _post_exit_processes(processes: list[dict[str, Any]], owner: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Retain all strangers; only the original exiting compute PID may linger."""
    retained, pending = [], False
    for row in processes:
        if row["pid"] == owner["pid"] and row["type"] == "C":
            identity = row.get("identity")
            if type(identity) is not dict or type(identity.get("readable")) is not bool:
                raise MonitoredHardwareError("exiting GPU process identity is malformed")
            if identity["readable"]:
                if ev.canonical_json_bytes(identity) != ev.canonical_json_bytes(owner):
                    raise MonitoredHardwareError("exited worker PID was reused or its executable identity changed")
            elif not _disappearing_owner_identity(identity, owner):
                raise MonitoredHardwareError("exiting GPU identity is not an approved disappearing proc entry")
            pending = True
        else:
            retained.append(row)
    return retained, pending


def _health_sample(spec: Mapping[str, Any], cursor: str, edac: Mapping[str, Any], *, owner_alive: bool,
                   external_sudo_anchor: Mapping[str, Any] | None = None) -> dict[str, Any]:
    started = time.monotonic_ns()
    policy = spec["policy"]
    for ref in spec["integrity_references"]:
        if ev.file_reference(ref["path"]) != ref:
            raise MonitoredHardwareError("bound collector/driver/library/external-evidence identity changed")
    external_sudo = None
    if policy.get("setter_mode") == _EXTERNAL_ADMIN_MODE:
        if external_sudo_anchor is None:
            raise MonitoredHardwareError("external sudo journal runtime anchor is unavailable")
        external_sudo = _external_sudo_journal_sample(policy, external_sudo_anchor)
    payload = _json_copy(spec["baseline"])
    payload["monotonic_started_ns"] = started
    payload["host"] = platform.node()
    payload["boot_id"] = hw._boot_id()
    query = _command([spec["nvidia_smi_path"], "-q", "-x"], timeout=.7)
    if query.get("executable") != policy["expected_nvidia_smi_executable"]:
        raise MonitoredHardwareError("health sampler executable differs from policy")
    payload["gpu"] = hw.parse_gpu_xml(hw._command_text(query), expected_gpu_uuid=policy["gpu_uuid"])
    for row in payload["gpu"]["processes"]:
        row["identity"] = hw._process_identity(row["pid"])
    payload["process"] = spec["owner"]
    owner_observation = _observe_owner_identity(
        spec, query, payload["gpu"]["processes"], health_started_ns=started, owner_alive=owner_alive)
    owner_alive = owner_observation["owner_alive_after"]
    payload["phase"] = "interval"
    payload["native_files"]["kernel_driver_version"] = hw._native_file(Path("/sys/module/nvidia/version"))
    graphics_identity = _xorg_identity(policy)
    payload.update(rapl=hw._collect_rapl(), cpu_temperatures=hw._collect_cpu_temperatures())
    journal = _command(["journalctl", "--no-pager", "-k", "--boot=" + policy["boot_id"].replace("-", ""),
                        "--output=json", "--show-cursor", "--cursor=" + cursor], timeout=.7)
    payload["kernel_health"] = hw.parse_kernel_journal(hw._command_text(journal), boot_id=policy["boot_id"],
                                                       expected_start_cursor=cursor)
    _reject_driver_events(_journal_records(hw._command_text(journal))[1:])
    current_edac = _edac_observation()
    _check_edac(current_edac, edac)
    payload.update(boot_id_after=hw._boot_id(), monotonic_finished_ns=time.monotonic_ns())
    owner_alive = owner_alive and not _pidfd_dead(spec["_owner_pidfd"])
    # After the pidfd signals exit, GPU release may briefly lag process exit.
    # That is never successful final evidence; a reused readable PID is foreign.
    pending = False
    if not owner_alive:
        retained, pending = _post_exit_processes(payload["gpu"]["processes"], spec["owner"])
        assessed = _json_copy(payload)
        assessed["gpu"]["processes"] = retained
    else:
        assessed = payload
    _assess_monitored(assessed, policy, owner=spec["owner"] if owner_alive else None, initial=False)
    result = {"kind": "health", "started_ns": started, "finished_ns": time.monotonic_ns(),
            "gpu": payload["gpu"], "kernel_health": payload["kernel_health"],
            "rapl": payload["rapl"], "cpu_temperatures": payload["cpu_temperatures"],
            "graphics_identity": graphics_identity, "edac": current_edac,
            "gpu_query": query, "journal_query": journal, "owner_gpu_release_pending": pending,
            "owner_pidfd_exited": not owner_alive, "owner_identity_observation": owner_observation}
    if external_sudo is not None:
        result["external_sudo_journal"] = external_sudo
    return result


def _send(sock: socket.socket, message: Mapping[str, Any]) -> None:
    raw = ev.canonical_json_bytes(message) + b"\n"
    if len(raw) > _FRAME_LIMIT:
        raise MonitoredHardwareError("monitor frame exceeds limit")
    sock.sendall(raw)


def _receive(sock: socket.socket, buffer: bytes) -> tuple[list[dict[str, Any]], bytes]:
    chunk = sock.recv(65536)
    if not chunk:
        raise EOFError("worker monitor control channel closed")
    buffer += chunk
    if len(buffer) > _FRAME_LIMIT:
        raise MonitoredHardwareError("monitor input exceeds frame limit")
    rows = []
    while b"\n" in buffer:
        raw, buffer = buffer.split(b"\n", 1)
        row = ev.strict_json_loads(raw)
        if type(row) is not dict:
            raise MonitoredHardwareError("monitor request must be an object")
        rows.append(row)
    return rows, buffer


def _deadline_error(now_ns: int, last: Mapping[str, int], policy: Mapping[str, Any], started_ns: int) -> str | None:
    if now_ns - started_ns > policy["workload_deadline_seconds"] * 1e9:
        return "authorized workload deadline exceeded"
    for kind in ("clock", "health"):
        if now_ns - last[kind] > policy[kind + "_max_gap_seconds"] * 1e9:
            return kind + " observation gap exceeded"
    return None


def _pidfd_dead(pidfd: int) -> bool:
    return bool(select.select([pidfd], [], [], 0)[0])


def _pidfd_backend() -> str:
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        return "python_pidfd"
    if sys.platform == "linux" and platform.machine() in {"x86_64", "aarch64"}:
        return "linux_syscalls_434_424"
    raise MonitoredHardwareError("this platform has no reviewed pidfd backend")


def _pidfd_syscall(number: int, *arguments: Any) -> int:
    # Some conda CPython builds used older headers and omit os.pidfd_open even
    # on a new Linux kernel. Select this ABI before execution; never fall back
    # from a failed operation to signalling a numeric PID.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(result)


def _open_pidfd(pid: int) -> int:
    if _pidfd_backend() == "python_pidfd":
        return os.pidfd_open(pid)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def _signal_pidfd(pidfd: int, sig: int) -> None:
    if _pidfd_backend() == "python_pidfd":
        signal.pidfd_send_signal(pidfd, sig)
    else:
        _pidfd_syscall(424, ctypes.c_int(pidfd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def _stop_owner(pidfd: int) -> None:
    """The fd refers to the exact owned worker, preventing PID reuse kills."""
    if not _pidfd_dead(pidfd):
        try:
            _signal_pidfd(pidfd, signal.SIGTERM)
        except ProcessLookupError:
            return
        if not select.select([pidfd], [], [], .5)[0]:
            try:
                _signal_pidfd(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                return


def _telemetry(state: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    clock, health = state.get("clock"), state.get("health")
    return {"run_id": spec["run_id"], "run_binding_sha256": spec["binding_sha256"],
            "policy_sha256": spec["policy"]["policy_sha256"], "owner": spec["owner"],
            "gpu_uuid": spec["policy"]["gpu_uuid"], "authorized_scope": spec["policy"]["authorized_scope"],
            "clock": {k: v for k, v in (clock or {}).items() if k != "command"},
            "gpu": health["gpu"] if health else None,
            "kernel_health": health["kernel_health"] if health else None,
            "health_started_ns": health["started_ns"] if health else None,
            "health_finished_ns": health["finished_ns"] if health else None,
            "loaded_clock_samples": state["loaded"], "locked_upper_readback_verified": False,
            "admission_mode": "native_monitored_scope_v1", "status": "MONITORED_SCOPE_READY"}


def _watchdog_main(spec: dict[str, Any], control: socket.socket, pidfd: int) -> None:
    """Independent process. Query and evidence threads never own its deadline loop."""
    spec = {**spec, "_owner_pidfd": pidfd}  # Process-local capability, never persisted.
    policy, root = spec["policy"], Path(spec["evidence_dir"])
    control.settimeout(.1)
    guardian_identity = hw._process_identity(os.getpid())
    if not guardian_identity.get("readable"):
        raise MonitoredHardwareError("guardian native identity is unavailable")
    heartbeat_path = root / "guardian-heartbeat.jsonl"
    heartbeat_fd = os.open(heartbeat_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
                           | getattr(os, "O_NOFOLLOW", 0), 0o600)
    state: dict[str, Any] = {"clock": None, "health": None, "clock_count": 0, "error": None, "loaded": 0,
                             "owner_alive": True, "own_compute_present": False, "sample_errors": []}
    lock, stop = threading.Lock(), threading.Event()

    def observe_owner(value: dict[str, Any]) -> None:
        # Detached memory only; reads, waits and I/O remain outside this lock.
        with lock:
            state["owner_identity_observation"] = value

    spec = {**spec, "_owner_identity_observation_sink": observe_owner}
    writes: queue.Queue[Any] = queue.Queue(maxsize=256)
    log_path = root / "monitor-samples.jsonl.gz"
    started = time.monotonic_ns()
    last = {"clock": started, "health": started}
    ready = False
    finishing_ns = dead_ns = None
    post_exit_health = 0
    last_health_seen = None
    writer_error: list[str] = []
    writer_finished = threading.Event()
    last_written = [started]
    chain = {"records": 0, "last_sha256": None}

    def writer() -> None:
        try:
            with open(log_path, "xb", buffering=0) as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                    while True:
                        row = writes.get()
                        if row is None:
                            break
                        entry = {"sequence": chain["records"], "previous_sha256": chain["last_sha256"], "record": row}
                        entry["sha256"] = ev.canonical_sha256(entry)
                        stream.write(ev.canonical_json_bytes(entry) + b"\n")
                        stream.flush()
                        os.fsync(raw.fileno())
                        last_written[0] = time.monotonic_ns()
                        chain.update(records=chain["records"] + 1, last_sha256=entry["sha256"])
                os.fsync(raw.fileno())
        except BaseException as exc:
            writer_error.append(type(exc).__name__ + ": " + str(exc))
        finally:
            writer_finished.set()

    def sample(kind: str) -> None:
        cursor, edac = spec["baseline"]["kernel_health"]["last_cursor"], spec["edac"]
        sudo_anchor = spec.get("external_sudo_anchor")
        period = policy[kind + "_period_seconds"]
        while not stop.is_set():
            began = time.monotonic()
            try:
                if kind == "clock":
                    row = _clock_sample(spec)
                else:
                    kwargs = {"owner_alive": state["owner_alive"]}
                    if policy.get("setter_mode") == _EXTERNAL_ADMIN_MODE:
                        kwargs["external_sudo_anchor"] = sudo_anchor
                    row = _health_sample(spec, cursor, edac, **kwargs)
                if kind == "health":
                    cursor = row["kernel_health"]["last_cursor"]
                    if policy.get("setter_mode") == _EXTERNAL_ADMIN_MODE:
                        sudo = row["external_sudo_journal"]
                        sudo_anchor = {"cursor": sudo["last_cursor"], "record_sha256": sudo["last_record_sha256"]}
                with lock:
                    if stop.is_set():
                        return
                    writes.put_nowait(row)
                    state[kind] = row
                    if kind == "clock":
                        state["clock_count"] += 1
                    last[kind] = row["started_ns"]  # Include native query latency in freshness.
                    if kind == "health":
                        state["own_compute_present"] = any(p["pid"] == spec["owner"]["pid"] and p["type"] == "C"
                                                            for p in row["gpu"]["processes"])
                    if (kind == "clock" and state["owner_alive"] and state["own_compute_present"]
                            and row["utilization_percent"] >= policy["loaded_utilization_min_percent"]):
                        state["loaded"] += 1
            except BaseException as exc:
                with lock:
                    message = type(exc).__name__ + ": " + str(exc)
                    state["error"] = state["error"] or message
                    state["sample_errors"].append(message)
                    if isinstance(exc, _ExternalSudoJournalError):
                        state.setdefault("external_sudo_rejection", exc.external_sudo_rejection)
                    if isinstance(exc, _OwnerIdentityError):
                        state.setdefault("owner_identity_rejection", exc.owner_identity_observation)
                return
            stop.wait(max(0., period - (time.monotonic() - began)))

    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()
    sampler_threads = [threading.Thread(target=sample, args=(kind,), daemon=True,
                                       name="native-" + kind + "-sampler") for kind in ("clock", "health")]
    for thread in sampler_threads:
        thread.start()
    buffer = b""
    failure = None
    final_telemetry = None
    channel_open = True
    last_heartbeat_ns = 0
    try:
        while True:
            now = time.monotonic_ns()
            dead = _pidfd_dead(pidfd)
            if dead and dead_ns is None:
                dead_ns = now
                state["owner_alive"] = False
                if finishing_ns is None:
                    raise MonitoredHardwareError("worker exited without finish declaration")
            with lock:
                frozen_last = dict(last)
                frozen_clock_count = state["clock_count"]
                sample_error = state["error"]
                snapshot = _telemetry(state, spec)
                clock_ready = frozen_clock_count >= policy["minimum_preload_clock_samples"]
                health_ready = state["health"] is not None
                health = state["health"]
            # Stamp after freezing the sampler state: a concurrent new sample
            # must not appear newer than this heartbeat's monotonic timestamp.
            now = time.monotonic_ns()
            error = sample_error or _deadline_error(now, frozen_last, policy, started)
            if error:
                raise MonitoredHardwareError(error)
            if writer_error or (now - last_written[0] > 2e9):
                raise MonitoredHardwareError("monitor evidence writer failed or stalled")
            announce_ready = clock_ready and health_ready and not ready
            if announce_ready:
                ready = True
            if announce_ready or now - last_heartbeat_ns >= policy["guardian_heartbeat_period_seconds"] * 1e9:
                # Only this deadline loop emits liveness. Sampling/writer threads
                # cannot keep the guardian apparently alive while it is stopped.
                heartbeat = {"schema_version": 1, "run_id": spec["run_id"],
                             "run_binding_sha256": spec["binding_sha256"], "policy_sha256": policy["policy_sha256"],
                             "gpu_uuid": policy["gpu_uuid"], "owner": spec["owner"],
                             "guardian_identity": guardian_identity, "monotonic_ns": now,
                             "last_clock_started_ns": frozen_last["clock"], "last_health_started_ns": frozen_last["health"],
                             "clock_samples": frozen_clock_count, "health_observed": health_ready,
                             "phase": "owner_exited" if dead else "finishing" if finishing_ns else "running" if ready else "preload"}
                raw_heartbeat = ev.canonical_json_bytes(heartbeat) + b"\n"
                if os.write(heartbeat_fd, raw_heartbeat) != len(raw_heartbeat):
                    raise MonitoredHardwareError("guardian heartbeat append was incomplete")
                last_heartbeat_ns = now
            if announce_ready:
                _send(control, {"op": "ready", "nonce": spec["nonce"], "telemetry": snapshot,
                                "guardian_identity": guardian_identity})
            if finishing_ns and not dead and now - finishing_ns > policy["owner_exit_timeout_seconds"] * 1e9:
                raise MonitoredHardwareError("finished worker did not exit within the frozen deadline")
            if dead:
                if now - dead_ns > policy["owner_exit_timeout_seconds"] * 1e9:
                    raise MonitoredHardwareError("CUDA context was not released after worker exit")
                if health and health["started_ns"] >= dead_ns and health["started_ns"] != last_health_seen:
                    last_health_seen = health["started_ns"]
                    if health["owner_gpu_release_pending"]:
                        post_exit_health = 0
                    else:
                        post_exit_health += 1
                if post_exit_health >= 2 and now - dead_ns >= policy["post_exit_observation_seconds"] * 1e9:
                    final_telemetry = snapshot
                    if state["loaded"] < policy["minimum_loaded_clock_samples"]:
                        raise MonitoredHardwareError("insufficient loaded clock observations for " + policy["authorized_scope"])
                    break
            if not dead and channel_open and select.select([control], [], [], .025)[0]:
                try:
                    messages, buffer = _receive(control, buffer)
                except EOFError:
                    if finishing_ns is None and not _pidfd_dead(pidfd):
                        raise MonitoredHardwareError("live worker lost monitor control channel")
                    channel_open = False
                    continue
                for message in messages:
                    if message.get("nonce") != spec["nonce"]:
                        raise MonitoredHardwareError("monitor control identity mismatch")
                    operation = message.get("op")
                    if operation == "abort":
                        raise MonitoredHardwareError("worker abort: " + str(message.get("reason")))
                    if operation == "finish":
                        if finishing_ns is not None:
                            raise MonitoredHardwareError("duplicate finish declaration")
                        finishing_ns = time.monotonic_ns()
                    elif operation != "check" or finishing_ns is not None:
                        raise MonitoredHardwareError("invalid monitored workload state transition")
                    _send(control, {"op": operation, "nonce": spec["nonce"], "telemetry": snapshot})
            else:
                time.sleep(.025)
    except BaseException as exc:
        failure = type(exc).__name__ + ": " + str(exc)
        # Close publication before the stop signal can wake a pending pidfd wait.
        # Error/identity observations still reach state during bounded closure.
        with lock:
            stop.set()
        _stop_owner(pidfd)  # Stop owned computation before any file I/O or thread join.
    finally:
        # Closing admission is atomic with nonblocking sample publication. A
        # producer can never enqueue behind the writer's terminal sentinel.
        with lock:
            stop.set()
        shutdown_deadline = time.monotonic() + _SAMPLER_SHUTDOWN_SECONDS
        for thread in sampler_threads:
            thread.join(max(0., shutdown_deadline - time.monotonic()))
        alive_samplers = [thread.name for thread in sampler_threads if thread.is_alive()]
        with lock:
            late_errors = list(state["sample_errors"])
            owner_observation = state.get("owner_identity_observation")
            owner_rejection = state.get("owner_identity_rejection")
            sudo_rejection = state.get("external_sudo_rejection")
        if alive_samplers or late_errors:
            failure = failure or ("sampler did not stop before evidence closure: " + ", ".join(alive_samplers)
                                  if alive_samplers else "sampler failed during closure: " + late_errors[0])
            _stop_owner(pidfd)
        try:
            writes.put_nowait(None)
        except queue.Full:
            failure = failure or "monitor evidence queue could not close"
        if not writer_finished.wait(2.):
            failure = failure or "monitor evidence writer did not close"
        if writer_error:
            failure = failure or "monitor evidence writer failed at close: " + "; ".join(writer_error)
        observation_refs = {}
        for name, value in (("owner-identity-observation.json", owner_observation),
                            ("owner-identity-rejection.json", owner_rejection),
                            ("external-clock-journal-rejection.json", sudo_rejection)):
            if value is None:
                continue
            try:
                observation_refs[name] = ev.write_exclusive_json(root / name, value)
            except BaseException as exc:
                failure = (failure + "; " if failure else "") + "rejection/observation evidence persistence failed: " + type(exc).__name__ + ": " + str(exc)
                _stop_owner(pidfd)
        result = {"schema_version": 1, "status": "FAIL" if failure else "PASS",
                  "run_id": spec["run_id"], "run_binding_sha256": spec["binding_sha256"],
                  "policy_sha256": policy["policy_sha256"], "authorized_scope": policy["authorized_scope"],
                  "gpu_uuid": policy["gpu_uuid"],
                  "owner": spec["owner"], "monitor_pid": os.getpid(), "finished_utc": _utc(),
                  "guardian_identity": guardian_identity, "pidfd_backend": _pidfd_backend(),
                  "failure": failure, "worker_exited": _pidfd_dead(pidfd),
                  "post_exit_health_observations": post_exit_health,
                  "sampled_clock_compliance": failure is None,
                  "locked_upper_readback_verified": False, "settings_after_exit": "keep_1500_1500",
                  "loaded_clock_samples": state["loaded"], "final_telemetry": final_telemetry,
                  "setter_receipt": spec["setter_reference"], "startup_evidence": spec["startup_reference"],
                  "strict_native_gate": "BLOCKED", "no_retry_or_fallback": True,
                  "owner_identity_evidence": observation_refs,
                  "sampler_shutdown": {"maximum_seconds": _SAMPLER_SHUTDOWN_SECONDS,
                                       "unfinished_threads": alive_samplers, "sample_errors": late_errors}}
        if writer_finished.is_set() and not writer_error:
            result["samples"] = ev.file_reference(log_path)
            result["sample_hash_chain"] = dict(chain)
        try:
            os.fsync(heartbeat_fd)
            os.close(heartbeat_fd)
            result["guardian_heartbeat"] = ev.file_reference(heartbeat_path)
            _publish_exclusive(root / "monitor-final.json", result)
        finally:
            control.close()
            os.close(pidfd)


class MonitoredHardwareAdmission:
    """An in-process handle to a live native watchdog, not a JSON observation."""
    __slots__ = ("_session",)

    def __init__(self, token: object, session: "MonitoredHardwareSession") -> None:
        if token is not _TOKEN:
            raise MonitoredHardwareError("admission requires a live native monitored session")
        object.__setattr__(self, "_session", session)

    def __setattr__(self, name: str, value: Any) -> None:
        raise MonitoredHardwareError("monitored admissions are immutable")

    def __reduce__(self) -> Any:
        raise MonitoredHardwareError("monitored admission cannot be serialized")


class MonitoredHardwareSession:
    def __init__(self, binding: Mapping[str, Any], policy_bundle: Mapping[str, Any],
                 evidence_dir: str | os.PathLike[str], authorized_scope: str,
                 workload_deadline_seconds: int = 600) -> None:
        evidence_authority = (type(policy_bundle) is dict
                              and type(policy_bundle.get("authorization_reference")) is dict
                              and policy_bundle["authorization_reference"].get("sha256")
                              == _RESOLUTION_EVIDENCE_AUTHORIZATION_SHA)
        if evidence_authority:
            self.policy = _validate_policy(policy_bundle, authorized_scope, workload_deadline_seconds)
            metadata = ev.validate_run_binding(binding, verify_files=False)
            _validate_resolution_evidence_binding(metadata, self.policy, authorized_scope)
            self.binding = ev.validate_run_binding(metadata)
        else:
            self.binding = ev.validate_run_binding(binding)
            self.policy = _validate_policy(policy_bundle, authorized_scope, workload_deadline_seconds)
        if self.policy["authorization_reference"]["sha256"] == _RESOLUTION_896_AUTHORIZATION_SHA:
            dimensions = {"input_size": 896, "physical_batch_size": 8, "accumulation_steps": 2}
            if any(type(self.binding["config"].get(key)) is not int
                   or self.binding["config"][key] != value for key, value in dimensions.items()):
                raise MonitoredHardwareError("896 authorization requires exact bound input_size=896, physical_batch_size=8, accumulation_steps=2")
        if authorized_scope == "synthetic_operator_diagnostic" and self.binding["data"]["kind"] != "synthetic":
            raise MonitoredHardwareError("synthetic operator diagnostic requires a synthetic input binding")
        if self.binding["config"].get("cuda_gpu_uuid") != self.policy["gpu_uuid"]:
            raise MonitoredHardwareError("workload GPU identity differs from policy")
        self.root = Path(evidence_dir).absolute()
        self._state = "new"
        self._channel: socket.socket | None = None
        self._child: subprocess.Popen | None = None
        self._buffer = b""
        self._nonce = os.urandom(32).hex()
        self._owner: dict[str, Any] | None = None
        self._admission: MonitoredHardwareAdmission | None = None
        self._last: dict[str, Any] | None = None
        self._guardian_identity: dict[str, Any] | None = None
        self._source_refs = [ev.file_reference(p) for p in (__file__, hw.__file__, ev.__file__)]

    def start(self) -> "MonitoredHardwareSession":
        if self._state != "new":
            raise MonitoredHardwareError("session start cannot be retried")
        self._state = "starting"
        self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        pidfd = None
        child = None
        try:
            ev.write_exclusive_json(self.root / "policy.json", self.policy)
            pidfd = _open_pidfd(os.getpid())
            self._owner = hw._process_identity(os.getpid())
            if not self._owner.get("readable"):
                raise MonitoredHardwareError("worker native identity is unreadable")
            native = hw.collect_native_hardware_probe(
                self.binding, policy=hw.HardwarePolicy(**{**self.policy["hardware_policy"],
                                                         "allowed_graphics_executable_sha256": ()})).as_dict()
            ev.write_exclusive_json(self.root / "strict-preflight.json", native)
            _assess_monitored(native, self.policy, owner=None, initial=True)
            _check_anchor(native, self.policy)
            identity = _xorg_identity(self.policy)
            edac = _edac_observation()
            _check_edac(edac, None)
            # External mode only reviews the prior administrator event. The
            # native modes retain exactly one explicitly selected setter call.
            external = self.policy["setter_mode"] == _EXTERNAL_ADMIN_MODE
            receipt = _external_clock_evidence(self.policy, native) if external else _setter(self.policy, native)
            setter_ref = ev.write_exclusive_json(self.root / "setter-receipt.json", receipt)
            if receipt["validation_errors"]:
                raise MonitoredHardwareError("native setter failed: " + json.dumps(receipt["validation_errors"]))
            external_sudo = (_external_sudo_journal_sample(self.policy, receipt["native_sudo_journal_anchor"],
                                                         evidence_dir=self.root) if external else None)
            post = hw.collect_native_hardware_probe(
                self.binding, policy=hw.HardwarePolicy(**{**self.policy["hardware_policy"],
                                                         "allowed_graphics_executable_sha256": ()})).as_dict()
            _assess_monitored(post, self.policy, owner=None, initial=True)
            _check_anchor(post, self.policy)
            if post["capability_inventory"] != native["capability_inventory"]:
                raise MonitoredHardwareError("driver/NVML identity changed across clock evidence verification")
            native_phase = "native_after_external_clock_review" if external else "native_post_setter"
            startup_ref = ev.write_exclusive_json(self.root / "startup.json", {
                native_phase: post, "xorg_identity": identity, "edac": edac,
                "setter_receipt": setter_ref, "policy_sha256": self.policy["policy_sha256"],
                "external_sudo_journal": external_sudo,
                "run_binding_sha256": self.binding["binding_sha256"], "strict_native_gate": "BLOCKED"})
            base = {k: v for k, v in post.items() if k not in ("commands", "collector_sources")}
            refs = self._source_refs + [post["capability_inventory"]["nvml_library"],
                                      post["commands"]["gpu_xml"]["executable"]]
            if external:
                refs += _external_clock_integrity_references(self.policy) + [setter_ref]
            spec = {"policy": self.policy, "owner": self._owner, "baseline": base,
                    "pidfd_backend": _pidfd_backend(),
                    "nvidia_smi_path": post["commands"]["gpu_xml"]["executable"]["path"],
                    "edac": edac, "integrity_references": refs, "run_id": self.binding["run_id"],
                    "binding_sha256": self.binding["binding_sha256"], "nonce": self._nonce,
                    "evidence_dir": str(self.root), "setter_reference": setter_ref,
                    "startup_reference": startup_ref}
            if external_sudo is not None:
                spec["external_sudo_anchor"] = {"cursor": external_sudo["last_cursor"],
                                                "record_sha256": external_sudo["last_record_sha256"]}
            spec_ref = ev.write_exclusive_json(self.root / "monitor-spec.json", spec)
            parent, child = socket.socketpair()
            self._channel = parent
            parent.settimeout(1.)
            env = os.environ.copy()
            package_root = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            self._child = subprocess.Popen(
                [sys.executable, "-c", "from sparse_rtdetr.baseline.training_v2b_admission import _entry; _entry()",
                 str(child.fileno()), str(pidfd), spec_ref["path"], spec_ref["sha256"]],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True, pass_fds=(child.fileno(), pidfd), start_new_session=True, env=env)
            child.close()
            child = None
            os.close(pidfd)
            pidfd = None
            # Cold interpreter/import time is not an admitted sampling gap.
            # The child enforces the unchanged 1 s / 2 s deadlines itself.
            self._wait_response("ready", timeout=self.policy["startup_deadline_seconds"])
            self._admission = MonitoredHardwareAdmission(_TOKEN, self)
            self._state = "running"
            _publish_exclusive(self.root / "admission-start.json", {
                "status": "MONITORED_SCOPE_READY", "monitor_pid": self._child.pid, "owner": self._owner,
                "guardian_identity": self._guardian_identity, "gpu_uuid": self.policy["gpu_uuid"],
                "heartbeat_path": str(self.root / "guardian-heartbeat.jsonl"),
                "authorized_scope": self.policy["authorized_scope"],
                "workload_deadline_seconds": self.policy["workload_deadline_seconds"],
                "heartbeat_period_seconds": self.policy["guardian_heartbeat_period_seconds"],
                "heartbeat_max_gap_seconds": self.policy["guardian_heartbeat_max_gap_seconds"],
                "clock_max_gap_seconds": self.policy["clock_max_gap_seconds"],
                "health_max_gap_seconds": self.policy["health_max_gap_seconds"],
                "collector_sources": self._source_refs, "monitor_spec_reference": spec_ref,
                "run_id": self.binding["run_id"], "run_binding_sha256": self.binding["binding_sha256"],
                "policy_sha256": self.policy["policy_sha256"], "setter_receipt": setter_ref,
                "clock_readback_verified": False})
            return self
        except BaseException as exc:
            self._state = "failed"
            if self._channel is not None:
                self._channel.close()
            ev.write_exclusive_json(self.root / "startup-failure.json", {"error": type(exc).__name__ + ": " + str(exc),
                                                                       "no_retry_or_fallback": True})
            raise
        finally:
            if pidfd is not None:
                os.close(pidfd)
            if child is not None:
                child.close()

    def _wait_response(self, operation: str, *, timeout: float = 1.) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._child is None or self._child.poll() is not None:
                raise MonitoredHardwareError("independent hardware watchdog exited")
            if select.select([self._channel], [], [], min(.1, end - time.monotonic()))[0]:
                rows, self._buffer = _receive(self._channel, self._buffer)
                if not rows:
                    continue
                if len(rows) != 1 or rows[0].get("op") != operation or rows[0].get("nonce") != self._nonce:
                    raise MonitoredHardwareError("monitor response identity/order differs")
                telemetry = rows[0]["telemetry"]
                if telemetry["run_binding_sha256"] != self.binding["binding_sha256"]:
                    raise MonitoredHardwareError("monitor binding differs")
                self._last = telemetry
                if operation == "ready":
                    guardian = rows[0].get("guardian_identity")
                    if guardian != hw._process_identity(self._child.pid):
                        raise MonitoredHardwareError("guardian native process identity differs")
                    self._guardian_identity = guardian
                return telemetry
        raise MonitoredHardwareError("watchdog response deadline exceeded")

    def admission(self) -> MonitoredHardwareAdmission:
        if self._state != "running" or self._admission is None:
            raise MonitoredHardwareError("session has no live admission")
        return self._admission

    def check(self, stage: str = "window") -> dict[str, Any]:
        # Full executable hashing runs in the independent health sampler, not
        # three times per logical window. PID/start identity remains checked here.
        own_pid = os.getpid()
        own_start = int((_PROC / str(own_pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])
        if (self._state != "running" or self._owner is None or own_pid != self._owner["pid"]
                or own_start != self._owner["start_ticks"]):
            raise MonitoredHardwareError("monitored worker identity/state changed")
        try:
            _send(self._channel, {"op": "check", "nonce": self._nonce, "stage": stage})
            return self._wait_response("check")
        except BaseException:
            self._state = "failed"
            if self._channel:
                self._channel.close()  # Guardian independently detects loss and stops this exact worker.
            raise

    def abort(self, reason: str) -> None:
        self._state = "failed"
        if self._channel is not None:
            try:
                _send(self._channel, {"op": "abort", "nonce": self._nonce, "reason": str(reason)})
            finally:
                self._channel.close()

    def finish(self) -> dict[str, Any]:
        if self._state != "running":
            raise MonitoredHardwareError("only an active workload can finish")
        _send(self._channel, {"op": "finish", "nonce": self._nonce})
        self._wait_response("finish")
        self._state = "finishing"
        return {"monitor_final_report": str(self.root / "monitor-final.json"),
                "monitor_pid": self._child.pid, "run_id": self.binding["run_id"],
                "guardian_identity": self._guardian_identity, "gpu_uuid": self.policy["gpu_uuid"],
                "run_binding_sha256": self.binding["binding_sha256"], "policy_sha256": self.policy["policy_sha256"],
                "owner": self._owner, "worker_must_exit": True}

    def __enter__(self) -> "MonitoredHardwareSession":
        return self.start()

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        if kind is not None:
            self.abort(str(error))
        elif self._state == "running":
            self.finish()


def require_monitored_hardware_admission(probe: MonitoredHardwareAdmission, *, binding: Mapping[str, Any],
                                         expected_gpu_uuid: str, max_age_seconds: float = 30.) -> dict[str, Any]:
    if type(probe) is not MonitoredHardwareAdmission:
        raise MonitoredHardwareError("supplied JSON cannot grant monitored admission")
    if type(max_age_seconds) not in {int, float} or not math.isfinite(max_age_seconds) or not 0 < max_age_seconds <= 60:
        raise MonitoredHardwareError("invalid native admission age")
    checked = ev.validate_run_binding(binding, verify_files=False)
    session = probe._session
    if (checked != session.binding or expected_gpu_uuid != session.policy["gpu_uuid"]
            or session._admission is not probe):
        raise MonitoredHardwareError("live monitored capability is bound to another run/device")
    for ref in session._source_refs:
        if ev.file_reference(ref["path"]) != ref:
            raise MonitoredHardwareError("admission implementation changed")
    telemetry = session.check(stage="native_admission")
    now = time.monotonic_ns()
    for field, maximum in (("clock", min(1., max_age_seconds)), ("health", min(2., max_age_seconds))):
        observed = telemetry["clock"]["started_ns"] if field == "clock" else telemetry["health_started_ns"]
        if observed > now or now - observed > maximum * 1e9:
            raise MonitoredHardwareError("monitored native admission is stale")
    return telemetry


def wait_for_monitored_finish(reference: Mapping[str, Any], *, timeout_seconds: float = 15.) -> dict[str, Any]:
    """Controller only, after worker exit. Reading this report grants no capability."""
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
        raise MonitoredHardwareError("invalid final report wait bound")
    path = Path(reference["monitor_final_report"])
    end = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= end:
            raise MonitoredHardwareError("monitor final report unavailable")
        time.sleep(.05)
    path, raw = ev._read_regular(path)
    report_reference = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                        "size_bytes": len(raw), "sha256_scope": "complete_file_bytes"}
    report = ev.strict_json_loads(raw)
    for key in ("run_id", "run_binding_sha256", "policy_sha256", "monitor_pid", "owner", "guardian_identity", "gpu_uuid"):
        if report.get(key) != reference.get(key):
            raise MonitoredHardwareError("monitor final evidence binding differs")
    if report.get("status") != "PASS" or report.get("worker_exited") is not True:
        raise MonitoredHardwareError("monitor workload failed: " + str(report.get("failure")))
    for name in ("samples", "setter_receipt", "startup_evidence", "guardian_heartbeat"):
        ref = report[name]
        if ev.file_reference(ref["path"]) != ref:
            raise MonitoredHardwareError("monitor final evidence changed")
    if ev.file_reference(path) != report_reference:
        raise MonitoredHardwareError("monitor final report changed while verified")
    return {**report, "final_report_reference": report_reference}


def _entry() -> None:
    fd, pidfd = int(sys.argv[1]), int(sys.argv[2])
    path, digest = Path(sys.argv[3]), sys.argv[4]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        _stop_owner(pidfd)
        raise MonitoredHardwareError("watchdog specification changed")
    spec = ev.strict_json_loads(raw)
    _watchdog_main(spec, socket.socket(fileno=fd), pidfd)
