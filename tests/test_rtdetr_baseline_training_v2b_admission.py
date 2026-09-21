"""CPU-only native gate contracts, including a genuinely separate watchdog.

GPU commands/setters are forbidden unless replaced with in-memory fixtures.
The watchdog tests may signal only their explicitly spawned CPU dummy worker.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import select
import socket
import subprocess
import sys
import threading
import types
import time

import pytest

from sparse_rtdetr.baseline import training_v2b_admission as ad
from sparse_rtdetr.baseline import training_v2b_evidence as ev
from sparse_rtdetr.baseline import training_v2b_hardware as hw
from test_rtdetr_baseline_training_v2b_hardware import bound, native_fixture, capture, journal, UUID, BOOT


@pytest.fixture(autouse=True)
def no_native_gpu_or_setter(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU test attempted an unmocked native GPU command")
    monkeypatch.setattr(ad, "_command", forbidden)
    monkeypatch.setattr(hw, "_run_native_command", forbidden)


def proc(pid=4661, ppid=4659):
    return {"pid": pid, "ppid": ppid, "start_ticks": 5989,
            "uid_fields": [1000] * 4, "gid_fields": [1000] * 4,
            "cmdline_sha256": "a" * 64, "cgroup": "0::/user.slice/user-1000.slice/session-3.scope\n",
            "executable": {"path": "/usr/lib/xorg/Xorg", "sha256": "b" * 64,
                           "size_bytes": 10, "owner_uid": 0, "owner_gid": 0,
                           "mode_octal": "0o755", "group_or_other_writable": False}}


def gpu_identity(p):
    return {"pid": p["pid"], "start_ticks": p["start_ticks"], "readable": True,
            "executable": {k: p["executable"][k] for k in ("path", "sha256", "size_bytes")}}


@pytest.fixture
def observation(native_fixture):
    native = hw.collect_native_hardware_probe(native_fixture["bound"], policy=native_fixture["policy"]).as_dict()
    xorg = proc()
    native["gpu"]["processes"] = [{"pid": xorg["pid"], "type": "G", "memory_mib": 4.,
                                    "identity": gpu_identity(xorg)}]
    native["commands"]["gpu_xml"]["executable"] = ev.file_reference(native_fixture["library"])
    policy = {"host": native["host"], "boot_id": BOOT, "gpu_uuid": UUID,
              "pci_bus_id": native["gpu"]["pci_bus_id"], "xorg": xorg,
              "xorg_parent": proc(4659, 4389), "hardware_policy": native_fixture["policy"].as_dict(),
              "xorg_memory_max_mib": 16,
              "expected_driver_userspace_version": native["gpu"]["driver_userspace_version"],
              "expected_kernel_driver_version": hw._file_text(native["native_files"]["kernel_driver_version"]),
              "expected_nvml_library": copy.deepcopy(native["capability_inventory"]["nvml_library"]),
              "expected_nvidia_smi_executable": copy.deepcopy(native["commands"]["gpu_xml"]["executable"]),
              "known_bert_records": [], "logind": {"Id": "3", "User": "1000", "Active": "yes"}}
    return native, policy


def test_monitored_policy_does_not_relabel_strict_evidence(observation):
    native, policy = observation
    before = copy.deepcopy(native)
    ad._assess_monitored(native, policy, owner=None, initial=True)
    assert native == before
    assert native["gate"]["status"] == "BLOCKED"
    assert "GRAPHICS_CLOCK_CAP_UNVERIFIED" in {r["code"] for r in hw._assess(native)}


@pytest.mark.parametrize("change", ["extra_g", "compute", "mixed", "pid", "start", "hash", "boot", "pci", "unreadable"])
def test_exact_occupant_and_host_identity_is_required(observation, change):
    native, policy = observation
    gpu = native["gpu"]
    if change in {"extra_g", "compute", "mixed"}:
        row = copy.deepcopy(gpu["processes"][0])
        row["type"] = {"extra_g": "G", "compute": "C", "mixed": "C+G"}[change]
        gpu["processes"].append(row)
    elif change == "pid": gpu["processes"][0]["pid"] += 1
    elif change == "start": gpu["processes"][0]["identity"]["start_ticks"] += 1
    elif change == "hash": gpu["processes"][0]["identity"]["executable"]["sha256"] = "c" * 64
    elif change == "boot": native["boot_id"] = "a-different-boot"
    elif change == "pci": gpu["pci_bus_id"] = "00000000:02:00.0"
    elif change == "unreadable": gpu["processes"][0]["identity"]["readable"] = False
    with pytest.raises(ad.MonitoredHardwareError):
        ad._assess_monitored(native, policy, owner=None, initial=True)


@pytest.mark.parametrize("change", ["clock", "temperature", "busy", "used", "free", "driver"])
def test_xorg_exception_does_not_waive_other_gates(observation, change):
    native, policy = observation
    field, value = {"clock": ("current_graphics_clock_mhz", 1501), "temperature": ("temperature_c", 80),
                    "busy": ("utilization_percent", 6), "used": ("memory_used_mib", 2048),
                    "free": ("memory_free_mib", 19000), "driver": ("driver_userspace_version", "different")}[change]
    native["gpu"][field] = value
    with pytest.raises(ad.MonitoredHardwareError):
        ad._assess_monitored(native, policy, owner=None, initial=True)


def test_bert_exception_is_full_record_and_never_allows_delta(observation):
    native, policy = observation
    raw = journal(["Linux version fixture", "BERT: [Hardware Error]: Skipped 1 error records",
                   "BERT: Total records found: 1"])
    native["kernel_health"] = hw.parse_kernel_journal(raw, boot_id=BOOT)
    policy["known_bert_records"] = copy.deepcopy(native["kernel_health"]["findings"])
    ad._assess_monitored(native, policy, owner=None, initial=True)
    native["kernel_health"]["findings"][0]["cursor"] += "new"
    with pytest.raises(ad.MonitoredHardwareError, match="exact BERT"):
        ad._assess_monitored(native, policy, owner=None, initial=True)
    native["kernel_health"]["findings"] = policy["known_bert_records"]
    with pytest.raises(ad.MonitoredHardwareError, match="new kernel"):
        ad._assess_monitored(native, policy, owner=native["process"], initial=False)


def test_bert_multiplicity_and_generic_machine_check_are_not_waived(observation):
    native, policy = observation
    raw = journal(["Linux version fixture", "BERT: Total records found: 1"])
    native["kernel_health"] = hw.parse_kernel_journal(raw, boot_id=BOOT)
    policy["known_bert_records"] = copy.deepcopy(native["kernel_health"]["findings"])
    native["kernel_health"]["findings"].append(copy.deepcopy(policy["known_bert_records"][0]))
    with pytest.raises(ad.MonitoredHardwareError):
        ad._assess_monitored(native, policy, owner=None, initial=True)
    native["kernel_health"] = hw.parse_kernel_journal(journal(["Linux version fixture", "mce: [Hardware Error]"]), boot_id=BOOT)
    with pytest.raises(ad.MonitoredHardwareError):
        ad._assess_monitored(native, policy, owner=None, initial=True)


def test_only_exact_owned_compute_is_allowed_during_interval(observation):
    native, policy = observation
    native["gpu"]["processes"].append({"pid": native["process"]["pid"], "type": "C", "identity": native["process"]})
    ad._assess_monitored(native, policy, owner=native["process"], initial=False)
    with pytest.raises(ad.MonitoredHardwareError):
        ad._assess_monitored(native, policy, owner=None, initial=True)


@pytest.mark.parametrize("readable", [True, False])
def test_post_exit_context_remains_pending_even_with_same_identity(readable):
    owner = gpu_identity(proc(51234))
    identity = copy.deepcopy(owner) if readable else _missing_owner_proc(owner)
    rows = [{"pid": owner["pid"], "type": "C", "identity": identity}]
    retained, pending = ad._post_exit_processes(rows, owner)
    assert retained == [] and pending is True
    assert len(rows) == 1


@pytest.mark.parametrize("change", ["start_ticks", "executable", "malformed"])
def test_post_exit_context_rejects_pid_reuse_or_malformed_identity(change):
    owner = gpu_identity(proc(51234))
    identity = copy.deepcopy(owner)
    if change == "start_ticks": identity["start_ticks"] += 1
    elif change == "executable": identity["executable"]["sha256"] = "c" * 64
    else: identity.pop("readable")
    with pytest.raises(ad.MonitoredHardwareError, match="reused|malformed"):
        ad._post_exit_processes([{"pid": owner["pid"], "type": "C", "identity": identity}], owner)


def test_post_exit_filter_retains_foreign_compute_and_mixed_graphics():
    owner = gpu_identity(proc(51234))
    rows = [{"pid": owner["pid"], "type": "C+G", "identity": owner},
            {"pid": owner["pid"] + 1, "type": "C", "identity": {"readable": False}},
            {"pid": 4661, "type": "G", "identity": gpu_identity(proc())}]
    retained, pending = ad._post_exit_processes(rows, owner)
    assert retained == rows and pending is False


@pytest.mark.parametrize("owner_exited", [True, False])
def test_health_rechecks_pidfd_when_worker_exits_during_identity_query(observation, monkeypatch, owner_exited):
    native, policy = observation
    owner = native["process"]
    gpu = copy.deepcopy(native["gpu"])
    gpu["processes"].append({"pid": owner["pid"], "type": "C", "memory_mib": 32})
    binary = policy["expected_nvidia_smi_executable"]["path"]
    def command(argv, **kwargs):
        assert argv[0] in {binary, "journalctl"}
        text = journal(["normal kernel event"], start=1) if argv[0] == "journalctl" else "CPU XML fixture"
        return {**capture(argv, text), "executable": policy["expected_nvidia_smi_executable"]}
    def identity(pid):
        if pid == owner["pid"]:
            return _missing_owner_proc(owner)
        assert pid == policy["xorg"]["pid"]
        return gpu_identity(policy["xorg"])
    def dead(pidfd):
        assert pidfd == 424242
        return owner_exited
    edac = {"controller_counters_available": False, "controllers": [],
            "counter_delta_available": False, "unavailable_means": "unknown_not_zero"}
    monkeypatch.setattr(ad, "_command", command)
    monkeypatch.setattr(hw, "parse_gpu_xml", lambda *args, **kwargs: copy.deepcopy(gpu))
    monkeypatch.setattr(hw, "_process_identity", identity)
    monkeypatch.setattr(ad, "_pidfd_dead", dead)
    monkeypatch.setattr(ad, "_wait_owner_pidfd", lambda fd, timeout: ([], [], []))
    monkeypatch.setattr(ad, "_xorg_identity", lambda p: {"cpu_fixture": True})
    monkeypatch.setattr(ad, "_edac_observation", lambda: copy.deepcopy(edac))
    spec = {"policy": policy, "baseline": native, "owner": owner, "nvidia_smi_path": binary,
            "integrity_references": [], "_owner_pidfd": 424242}
    if owner_exited:
        row = ad._health_sample(spec, "fixture:1", edac, owner_alive=True)
        assert row["owner_pidfd_exited"] is True
        assert row["owner_gpu_release_pending"] is True
    else:
        with pytest.raises(ad.MonitoredHardwareError, match="owner pidfd event wait"):
            ad._health_sample(spec, "fixture:1", edac, owner_alive=True)



def _missing_owner_proc(owner, error_type="FileNotFoundError"):
    return {"pid": owner["pid"], "readable": False,
            "error": error_type + ": [Errno 2] missing entry: " + repr("/proc/" + str(owner["pid"]) + "/exe")}


@pytest.fixture
def owner_identity_case(observation, monkeypatch):
    native, policy = observation
    owner = copy.deepcopy(native["process"])
    spec = {"policy": policy, "owner": owner, "_owner_pidfd": 424242,
            "run_id": "cpu-owner-identity", "binding_sha256": "2" * 64}
    query = capture(["CPU-fixture-no-GPU-query"], "CPU native query fixture")
    processes = [{"pid": owner["pid"], "type": "C", "identity": copy.deepcopy(owner)}]
    monkeypatch.setattr(hw, "_process_identity", lambda pid: copy.deepcopy(owner))
    monkeypatch.setattr(ad, "_pidfd_dead", lambda fd: False)
    return spec, query, processes


@pytest.mark.parametrize("dead", [False, True])
def test_owner_identity_exact_readable_never_waits(owner_identity_case, monkeypatch, dead):
    spec, query, processes = owner_identity_case
    monkeypatch.setattr(ad, "_pidfd_dead", lambda fd: dead)
    monkeypatch.setattr(ad, "_wait_owner_pidfd", lambda *a: pytest.fail("exact identity attempted wait"))
    result = ad._observe_owner_identity(spec, query, processes, health_started_ns=123, owner_alive=True)
    assert result["owner_alive_after"] is (not dead)
    assert result["wait"] is None and result["health_started_ns"] == 123
    assert result["observed_owner"] == spec["owner"]
    assert result["gpu_query"] == query and result["gpu_processes"] == processes


@pytest.mark.parametrize("dead", [False, True])
@pytest.mark.parametrize("change", ["pid", "start", "start_type", "executable", "path"])
def test_owner_identity_readable_mismatch_never_inherits_exit_exception(owner_identity_case, monkeypatch, dead, change):
    spec, query, processes = owner_identity_case
    changed = copy.deepcopy(spec["owner"])
    if change == "pid": changed["pid"] += 1
    elif change == "start": changed["start_ticks"] += 1
    elif change == "start_type": changed["start_ticks"] = float(changed["start_ticks"])
    elif change == "executable": changed["executable"]["sha256"] = "f" * 64
    else: changed["executable"]["path"] += "-different"
    monkeypatch.setattr(hw, "_process_identity", lambda pid: changed)
    monkeypatch.setattr(ad, "_pidfd_dead", lambda fd: dead)
    monkeypatch.setattr(ad, "_wait_owner_pidfd", lambda *a: pytest.fail("readable mismatch attempted wait"))
    with pytest.raises(ad._OwnerIdentityError) as failure:
        ad._observe_owner_identity(spec, query, processes, health_started_ns=123, owner_alive=True)
    record = failure.value.owner_identity_observation
    assert record["decision"] == "REJECTED" and record["observed_owner"] == changed
    assert record["wait"] is None and record["gpu_query"] == query


@pytest.mark.parametrize("dead", [False, True])
@pytest.mark.parametrize("change", ["none", "missing_fields", "wrong_pid", "readable_int", "extra", "non_json",
    "unknown_error", "replaced", "integrity", "permission", "different_missing_path"])
def test_owner_identity_only_strict_disappearing_proc_can_wait(owner_identity_case, monkeypatch, dead, change):
    spec, query, processes = owner_identity_case
    observed = _missing_owner_proc(spec["owner"])
    if change == "none": observed = None
    elif change == "missing_fields": observed.pop("error")
    elif change == "wrong_pid": observed["pid"] += 1
    elif change == "readable_int": observed["readable"] = 0
    elif change == "extra": observed["unexpected"] = True
    elif change == "non_json": observed = object()
    else:
        observed["error"] = {"unknown_error": "unexpected",
            "replaced": "HardwareGateError: process was replaced while observed",
            "integrity": "EvidenceError: executable hash changed", "permission": "PermissionError: access denied",
            "different_missing_path": "FileNotFoundError: missing '/tmp/python'"}[change]
    monkeypatch.setattr(hw, "_process_identity", lambda pid: observed)
    monkeypatch.setattr(ad, "_pidfd_dead", lambda fd: dead)
    monkeypatch.setattr(ad, "_wait_owner_pidfd", lambda *a: pytest.fail("invalid observation attempted wait"))
    with pytest.raises(ad._OwnerIdentityError) as failure:
        ad._observe_owner_identity(spec, query, processes, health_started_ns=123, owner_alive=True)
    record = failure.value.owner_identity_observation
    assert record["decision"] == "REJECTED" and record["wait"] is None
    assert ev.canonical_json_bytes(record)


@pytest.mark.parametrize("error_type", ["FileNotFoundError", "ProcessLookupError"])
def test_owner_identity_wait_is_once_on_original_pidfd_and_has_full_evidence(owner_identity_case, monkeypatch, error_type):
    spec, query, processes = owner_identity_case
    observed = _missing_owner_proc(spec["owner"], error_type)
    monkeypatch.setattr(hw, "_process_identity", lambda pid: observed)
    answers = iter([False, False, True])
    monkeypatch.setattr(ad, "_pidfd_dead", lambda fd: next(answers))
    calls, published = [], []
    spec["_owner_identity_observation_sink"] = published.append
    def wait(fd, timeout):
        calls.append((fd, timeout))
        assert fd == spec["_owner_pidfd"] and 0 < timeout <= .2
        return [fd], [], []
    monkeypatch.setattr(ad, "_wait_owner_pidfd", wait)
    result = ad._observe_owner_identity(spec, query, processes, health_started_ns=123, owner_alive=True)
    assert len(calls) == 1 and result["decision"] == "EXIT_CONFIRMED"
    assert result["owner_alive_after"] is False and result["health_started_ns"] == 123
    assert result["observed_owner"] == observed and result["wait"]["exit_confirmed"] is True
    assert result["wait"]["elapsed_ns"] == result["wait"]["finished_ns"] - result["wait"]["started_ns"]
    assert 0 <= result["wait"]["elapsed_ns"] <= 200_000_000
    assert [p["decision"] for p in published] == ["OBSERVING", "WAITING_FOR_ORIGINAL_PIDFD", "EXIT_CONFIRMED"]
    assert published[1]["wait"]["finished_ns"] is None
    assert result["gpu_query"] == query and result["gpu_processes"] == processes


@pytest.mark.parametrize("mode", ["timeout", "oserror", "interrupted", "shape", "wrong_fd",
                                 "duplicate_fd", "writable", "unconfirmed", "over_budget"])
def test_owner_identity_wait_fails_closed_without_retry(owner_identity_case, monkeypatch, mode):
    spec, query, processes = owner_identity_case
    monkeypatch.setattr(hw, "_process_identity", lambda pid: _missing_owner_proc(spec["owner"]))
    calls = []
    def wait(fd, timeout):
        calls.append((fd, timeout))
        if mode == "oserror": raise OSError("CPU wait failure")
        if mode == "interrupted": raise InterruptedError("CPU interrupted wait")
        if mode == "shape": return [fd]
        if mode == "wrong_fd": return [fd + 1], [], []
        if mode == "duplicate_fd": return [fd, fd], [], []
        if mode == "writable": return [], [fd], []
        if mode == "timeout": return [], [], []
        if mode == "over_budget": time.sleep(.21)
        return [fd], [], []
    monkeypatch.setattr(ad, "_wait_owner_pidfd", wait)
    with pytest.raises(ad._OwnerIdentityError) as failure:
        ad._observe_owner_identity(spec, query, processes, health_started_ns=123, owner_alive=True)
    record = failure.value.owner_identity_observation
    assert len(calls) == 1 and calls[0][0] == spec["_owner_pidfd"] and calls[0][1] <= .2
    assert record["decision"] == "REJECTED" and record["owner_alive_after"] is None
    assert record["wait"]["exit_confirmed"] is False and record["wait"]["finished_ns"] is not None
    assert record["health_started_ns"] == 123 and ev.canonical_json_bytes(record)


@pytest.mark.parametrize("which", ["identity", "pidfd"])
def test_owner_identity_reader_exceptions_preserve_raw_context(owner_identity_case, monkeypatch, which):
    spec, query, processes = owner_identity_case
    def fail(*args): raise OSError("CPU observer failure")
    monkeypatch.setattr(hw if which == "identity" else ad,
                        "_process_identity" if which == "identity" else "_pidfd_dead", fail)
    with pytest.raises(ad._OwnerIdentityError) as failure:
        ad._observe_owner_identity(spec, query, processes, health_started_ns=123, owner_alive=True)
    record = failure.value.owner_identity_observation
    assert record["gpu_query"] == query and record["gpu_processes"] == processes
    assert record["decision"] == "REJECTED" and record["completed_ns"] is not None


@pytest.mark.parametrize("error", ["HardwareGateError: process was replaced while observed",
                                  "EvidenceError: changed executable", "PermissionError: denied"])
def test_post_exit_unreadable_replacement_or_unknown_error_is_never_a_residual(error):
    owner = gpu_identity(proc(51234))
    identity = {"pid": owner["pid"], "readable": False, "error": error}
    with pytest.raises(ad.MonitoredHardwareError, match="disappearing proc"):
        ad._post_exit_processes([{"pid": owner["pid"], "type": "C", "identity": identity}], owner)


@pytest.mark.parametrize("exit_during_wait", [True, False])
def test_owner_identity_wait_uses_real_owned_cpu_pidfd(owner_identity_case, monkeypatch, exit_during_wait):
    if sys.platform != "linux": pytest.skip("Linux pidfd is required")
    import threading
    spec, query, _ = owner_identity_case
    ctx = mp.get_context("fork")
    read_fd, write_fd = os.pipe()
    worker = ctx.Process(target=_cpu_worker, args=(read_fd,))
    worker.start()
    pidfd = ad._open_pidfd(worker.pid)
    spec = {**spec, "owner": {"pid": worker.pid, "readable": True, "start_ticks": 1}, "_owner_pidfd": pidfd}
    monkeypatch.setattr(hw, "_process_identity", lambda pid: _missing_owner_proc(spec["owner"]))
    monkeypatch.setattr(ad, "_pidfd_dead", lambda fd: bool(select.select([fd], [], [], 0)[0]))
    timer = threading.Timer(.025, lambda: os.write(write_fd, b"x")) if exit_during_wait else None
    if timer: timer.start()
    try:
        if exit_during_wait:
            result = ad._observe_owner_identity(spec, query, [], health_started_ns=time.monotonic_ns(), owner_alive=True)
            assert result["decision"] == "EXIT_CONFIRMED" and result["wait"]["readable_fds"] == [pidfd]
            assert result["wait"]["exit_confirmed"] is True
            worker.join(1.)
            assert worker.exitcode == 0
        else:
            with pytest.raises(ad._OwnerIdentityError):
                ad._observe_owner_identity(spec, query, [], health_started_ns=time.monotonic_ns(), owner_alive=True)
            assert worker.is_alive()
    finally:
        if timer: timer.join(1.)
        os.write(write_fd, b"x")
        worker.join(1.)
        if worker.is_alive(): worker.terminate(); worker.join(1.)
        os.close(pidfd); os.close(read_fd); os.close(write_fd)


@pytest.mark.parametrize("memory", [None, float("nan"), 16.01, -1])
def test_graphics_allowance_has_its_own_readable_memory_budget(observation, memory):
    native, policy = observation
    native["gpu"]["processes"][0]["memory_mib"] = memory
    with pytest.raises(ad.MonitoredHardwareError, match="16 MiB"):
        ad._assess_monitored(native, policy, owner=None, initial=True)


def test_same_boot_does_not_allow_driver_library_or_binary_drift(observation):
    native, policy = observation
    for container, key in ((native["capability_inventory"]["nvml_library"], "sha256"),
                           (native["commands"]["gpu_xml"]["executable"], "sha256")):
        old = container[key]
        container[key] = "d" * 64
        with pytest.raises(ad.MonitoredHardwareError, match="identity changed"):
            ad._assess_monitored(native, policy, owner=None, initial=True)
        container[key] = old


@pytest.mark.parametrize("message", [
    "NVRM: loading NVIDIA UNIX x86_64 Kernel Module 580.178.04",
    "nvidia: module unloaded", "nvidia-modeset: Loading NVIDIA Kernel Mode Setting Driver",
    "GPU reset completed", "NVRM: resetting GPU", "Reloading nvidia kernel module",
    "nvidia 0000:01:00.0: Removing from iommu group",
])
def test_new_driver_reset_or_reload_is_a_stop_even_without_xid(message):
    with pytest.raises(ad.MonitoredHardwareError, match="reset/load"):
        ad._reject_driver_events([{"MESSAGE": message}])


def test_full_anchor_and_bert_records_are_compared_not_just_cursor(observation):
    native, policy = observation
    raw = journal(["Linux version fixture", "BERT: Total records found: 1", "normal anchor"])
    rows = ad._journal_records(raw)
    policy.update(kernel_anchor_cursor=rows[-1]["__CURSOR"],
                  kernel_anchor_record_sha256=ev.canonical_sha256(rows[-1]),
                  known_bert_raw_record_sha256={rows[1]["__CURSOR"]: ev.canonical_sha256(rows[1])})
    native["commands"]["kernel_journal"] = capture(["journalctl"], raw)
    ad._check_anchor(native, policy)
    for index, match in ((2, "anchor record changed"), (1, "BERT record changed")):
        modified = copy.deepcopy(rows)
        modified[index]["PRIORITY"] = "changed-but-not-in-parser-summary"
        changed = "\n".join(json.dumps(r) for r in modified) + "\n-- cursor: " + rows[-1]["__CURSOR"] + "\n"
        native["commands"]["kernel_journal"] = capture(["journalctl"], changed)
        with pytest.raises(ad.MonitoredHardwareError, match=match):
            ad._check_anchor(native, policy)


@pytest.fixture
def policy_bundle(tmp_path, monkeypatch, observation):
    native, policy = observation
    root = tmp_path / "reviewed-evidence"
    root.mkdir()
    def metadata(p):
        return {**copy.deepcopy(p), "cgroup": {"utf8": p["cgroup"]}, "stable_during_observation": True}
    logind = {"expected_identity_fields": {"Id": "3", "User": "1000"},
              "properties": {"Id": "3", "User": "1000", "Active": "yes", "State": "active"}}
    docs = {"current-hardware.json": native,
            "process-identities.json": {"processes": [{"native": {"type": "G"},
                "metadata": metadata(policy["xorg"]), "parent": metadata(policy["xorg_parent"])}]},
            "xorg-logind-session.json": logind}
    for name in ("independent-review-addendum.json", "hardware-log-policy-review.json", "clock-evidence-protocol.json",
                 "known-boot-bert-records.json", "xorg-identity-policy.json"):
        docs[name] = {"CPU_fixture": True}
    refs = {name: ev.write_exclusive_json(root / name, doc) for name, doc in docs.items()}
    manifest = ev.write_exclusive_json(root / "manifest.json", {"files": refs})
    monkeypatch.setattr(ad, "_ANCHOR_MANIFEST_SHAS", frozenset({manifest["sha256"]}))
    auth = tmp_path / "user-authorization.txt"
    auth.write_text("CPU fixture: no hardware authorization\n")
    auth_ref = ev.file_reference(auth)
    monkeypatch.setattr(ad, "_AUTHORIZATION_SHA", auth_ref["sha256"])
    return ad.build_policy_bundle(root, authorization_reference=auth_ref)


def test_policy_bundle_is_common_but_scoped_admission_hash_is_distinct(policy_bundle):
    first = ad._validate_policy(policy_bundle, "paired_smoke", 600)
    second = ad._validate_policy(policy_bundle, "train_core_30epoch", 43200)
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["policy_sha256"] != second["policy_sha256"]
    assert first["minimum_preload_clock_samples"] == 3
    assert first["minimum_loaded_clock_samples"] == second["minimum_loaded_clock_samples"] == 3
    assert first["expected_nvml_library"]["sha256"]


def test_phase_aware_policy_preserves_training_loaded_requirement(policy_bundle):
    training = ad._validate_policy(policy_bundle, "paired_smoke", 600, monitor_phase="training")
    quiescence = ad._validate_policy(policy_bundle, "paired_smoke", 600, monitor_phase="quiescence")
    restore = ad._validate_policy(policy_bundle, "paired_smoke", 600, monitor_phase="restore")
    assert training["monitor_phase"] == "training"
    assert training["loaded_samples_required"] is True
    assert training["minimum_loaded_clock_samples"] == 3
    for phase, selected in (("quiescence", quiescence), ("restore", restore)):
        assert selected["monitor_phase"] == phase
        assert selected["loaded_samples_required"] is False
        assert selected["minimum_loaded_clock_samples"] == 0
    assert len({training["policy_sha256"], quiescence["policy_sha256"], restore["policy_sha256"]}) == 3


def test_phase_aware_policy_rejects_non_training_scope(policy_bundle):
    with pytest.raises(ad.MonitoredHardwareError, match="paired_smoke"):
        ad._validate_policy(policy_bundle, "train_core_30epoch", 43200, monitor_phase="restore")


@pytest.mark.parametrize("field", ["clock_max_mhz", "xorg_memory_max_mib", "known_bert_records", "expected_nvml_library"])
def test_caller_cannot_edit_reviewed_policy(policy_bundle, field):
    policy_bundle[field] = "caller-supplied replacement"
    with pytest.raises(ad.MonitoredHardwareError, match="differs"):
        ad._validate_policy(policy_bundle, "paired_smoke", 600)


@pytest.mark.parametrize("scope,deadline", [("paired_smoke", 601), ("train_core_30epoch", 43201),
                                           ("production", 1), ("paired_smoke", True)])
def test_policy_scope_and_deadline_are_bounded(policy_bundle, scope, deadline):
    with pytest.raises(ad.MonitoredHardwareError, match="not authorized"):
        ad._validate_policy(policy_bundle, scope, deadline)


@pytest.fixture
def repair_policy_bundle(tmp_path, monkeypatch, policy_bundle):
    auth = tmp_path / "repair-authorization.txt"
    auth.write_text("CPU fixture: synthetic diagnostic repair authority; no GPU access\n")
    auth_ref = ev.file_reference(auth)
    monkeypatch.setattr(ad, "_REPAIR_AUTHORIZATION_SHA", auth_ref["sha256"])
    return ad.build_policy_bundle(Path(policy_bundle["authority"]["path"]).parent,
                                  authorization_reference=auth_ref)


def test_original_authority_keeps_its_exact_scope_declaration(policy_bundle, repair_policy_bundle):
    rebuilt = ad.build_policy_bundle(Path(policy_bundle["authority"]["path"]).parent,
                                     authorization_reference=policy_bundle["authorization_reference"])
    assert rebuilt == policy_bundle
    assert ev.canonical_json_bytes(rebuilt) == ev.canonical_json_bytes(policy_bundle)
    assert rebuilt["authorized_scope_limits_seconds"] == {"paired_smoke": 600, "train_core_30epoch": 43200}
    with pytest.raises(ad.MonitoredHardwareError, match="not authorized"):
        ad._validate_policy(rebuilt, "synthetic_operator_diagnostic", 600)


def test_repair_authority_adds_only_synthetic_scope_without_hardware_changes(policy_bundle, repair_policy_bundle):
    assert repair_policy_bundle["authorized_scope_limits_seconds"] == {
        "paired_smoke": 600, "train_core_30epoch": 43200, "synthetic_operator_diagnostic": 600}
    changed = {"authorization_reference", "authorized_scope_limits_seconds", "bundle_sha256"}
    assert {k: v for k, v in policy_bundle.items() if k not in changed} == {
        k: v for k, v in repair_policy_bundle.items() if k not in changed}
    for scope, deadline in repair_policy_bundle["authorized_scope_limits_seconds"].items():
        selected = ad._validate_policy(repair_policy_bundle, scope, deadline)
        assert selected["authorized_scope"] == scope
        assert selected["workload_deadline_seconds"] == deadline
        assert selected["minimum_loaded_clock_samples"] == 3

@pytest.fixture
def resolution_896_policy_bundle(tmp_path, monkeypatch, policy_bundle, external_receipt):
    auth = tmp_path / "resolution-896-authorization.txt"
    auth.write_text("CPU fixture: exact 896 capacity pilot and control; no GPU access\n")
    auth_ref = ev.file_reference(auth)
    monkeypatch.setattr(ad, "_RESOLUTION_896_AUTHORIZATION_SHA", auth_ref["sha256"])
    return ad.build_policy_bundle(Path(policy_bundle["authority"]["path"]).parent,
                                  authorization_reference=auth_ref,
                                  setter_mode=ad._EXTERNAL_ADMIN_MODE,
                                  external_clock_receipt=external_receipt["reference"])


def test_resolution_896_authority_preserves_scopes_and_hardware_rules(
        external_receipt, resolution_896_policy_bundle):
    original_external = external_receipt["build"](external_receipt["reference"])
    assert resolution_896_policy_bundle["authorized_scope_limits_seconds"] == {
        "paired_smoke": 600, "train_core_30epoch": 43200}
    assert resolution_896_policy_bundle["setter_mode"] == ad._EXTERNAL_ADMIN_MODE
    assert resolution_896_policy_bundle["external_clock_receipt"] == external_receipt["reference"]
    changed = {"authorization_reference", "bundle_sha256"}
    assert {k: v for k, v in original_external.items() if k not in changed} == {
        k: v for k, v in resolution_896_policy_bundle.items() if k not in changed}
    for scope, deadline in (("paired_smoke", 600), ("train_core_30epoch", 43200)):
        selected = ad._validate_policy(resolution_896_policy_bundle, scope, deadline)
        assert selected["authorized_scope"] == scope
        assert selected["workload_deadline_seconds"] == deadline


@pytest.mark.parametrize("mode", ["direct", "sudo_n"])
def test_resolution_896_authority_rejects_native_setter_policy_before_native_access(
        resolution_896_policy_bundle, monkeypatch, mode):
    monkeypatch.setattr(hw, "collect_native_hardware_probe",
                        lambda *a, **kw: pytest.fail("policy construction attempted native hardware access"))
    monkeypatch.setattr(ad, "_setter",
                        lambda *a, **kw: pytest.fail("policy construction attempted a clock setter"))
    with pytest.raises(ad.MonitoredHardwareError, match="896 authorization requires external_admin_acknowledged"):
        ad.build_policy_bundle(Path(resolution_896_policy_bundle["authority"]["path"]).parent,
            authorization_reference=resolution_896_policy_bundle["authorization_reference"], setter_mode=mode)


@pytest.mark.parametrize("mode", ["direct", "sudo_n"])
def test_resolution_896_session_rejects_rehashed_native_setter_policy_before_native_access(
        resolution_896_policy_bundle, bound, tmp_path, monkeypatch, mode):
    bundle = copy.deepcopy(resolution_896_policy_bundle)
    bundle["setter_mode"] = mode
    bundle.pop("external_clock_receipt")
    bundle.pop("external_journal_executable")
    bundle["bundle_sha256"] = ev.canonical_sha256({k: v for k, v in bundle.items() if k != "bundle_sha256"})
    bound = copy.deepcopy(bound)
    bound["config"].update(input_size=896, physical_batch_size=8, accumulation_steps=2, cuda_gpu_uuid=UUID)
    bound["binding_sha256"] = ev.canonical_sha256({k: v for k, v in bound.items() if k != "binding_sha256"})
    monkeypatch.setattr(hw, "collect_native_hardware_probe",
                        lambda *a, **kw: pytest.fail("session construction attempted native hardware access"))
    monkeypatch.setattr(ad, "_setter",
                        lambda *a, **kw: pytest.fail("session construction attempted a clock setter"))
    output = tmp_path / "rejected-native-setter"
    with pytest.raises(ad.MonitoredHardwareError, match="896 authorization requires external_admin_acknowledged"):
        ad.MonitoredHardwareSession(bound, bundle, output, "paired_smoke", 600)
    assert not output.exists()


@pytest.mark.parametrize("authority", ["original", "repair"])
@pytest.mark.parametrize("mode", ["direct", "sudo_n", ad._EXTERNAL_ADMIN_MODE])
def test_prior_authorities_keep_their_existing_setter_modes(
        policy_bundle, repair_policy_bundle, external_receipt, authority, mode):
    original = policy_bundle if authority == "original" else repair_policy_bundle
    receipt = external_receipt["reference"] if mode == ad._EXTERNAL_ADMIN_MODE else None
    rebuilt = ad.build_policy_bundle(Path(original["authority"]["path"]).parent,
        authorization_reference=original["authorization_reference"], setter_mode=mode,
        external_clock_receipt=receipt)
    assert rebuilt["setter_mode"] == mode
    assert rebuilt["authorized_scope_limits_seconds"] == original["authorized_scope_limits_seconds"]
    assert ad._validate_policy(rebuilt, "paired_smoke", 600)["setter_mode"] == mode
    if mode == "direct":
        assert ev.canonical_json_bytes(rebuilt) == ev.canonical_json_bytes(original)


@pytest.mark.parametrize("scope,deadline", [
    ("synthetic_operator_diagnostic", 600), ("paired_smoke", 601),
    ("train_core_30epoch", 43201), ("paired_smoke", True), ("production", 1),
])
def test_resolution_896_authority_cannot_expand_scope_or_deadline(
        resolution_896_policy_bundle, scope, deadline):
    with pytest.raises(ad.MonitoredHardwareError, match="not authorized"):
        ad._validate_policy(resolution_896_policy_bundle, scope, deadline)


@pytest.mark.parametrize("rehash", [False, True])
def test_resolution_896_authorization_requires_exact_complete_bytes(
        resolution_896_policy_bundle, rehash):
    ref = copy.deepcopy(resolution_896_policy_bundle["authorization_reference"])
    Path(ref["path"]).write_bytes(Path(ref["path"]).read_bytes() + b"changed\n")
    if rehash:
        ref = ev.file_reference(ref["path"])
    with pytest.raises(ad.MonitoredHardwareError, match="authorization reference differs"):
        ad.build_policy_bundle(Path(resolution_896_policy_bundle["authority"]["path"]).parent,
                               authorization_reference=ref, setter_mode=ad._EXTERNAL_ADMIN_MODE,
                               external_clock_receipt=resolution_896_policy_bundle["external_clock_receipt"])


def test_resolution_896_caller_rehash_cannot_add_synthetic_scope(resolution_896_policy_bundle):
    bundle = copy.deepcopy(resolution_896_policy_bundle)
    bundle["authorized_scope_limits_seconds"]["synthetic_operator_diagnostic"] = 600
    bundle["bundle_sha256"] = ev.canonical_sha256({k: v for k, v in bundle.items() if k != "bundle_sha256"})
    with pytest.raises(ad.MonitoredHardwareError, match="differs"):
        ad._validate_policy(bundle, "synthetic_operator_diagnostic", 600)


@pytest.mark.parametrize("scope,deadline", [("paired_smoke", 600), ("train_core_30epoch", 43200)])
def test_resolution_896_authority_accepts_exact_bound_dimensions_without_native_access(
        resolution_896_policy_bundle, bound, tmp_path, scope, deadline):
    bound = copy.deepcopy(bound)
    bound["config"].update(input_size=896, physical_batch_size=8, accumulation_steps=2, cuda_gpu_uuid=UUID)
    bound["binding_sha256"] = ev.canonical_sha256({k: v for k, v in bound.items() if k != "binding_sha256"})
    output = tmp_path / "never-started"
    session = ad.MonitoredHardwareSession(bound, resolution_896_policy_bundle, output, scope, deadline)
    assert session._state == "new"
    assert not output.exists()


@pytest.mark.parametrize("key,value", [
    ("input_size", 640), ("input_size", 960), ("input_size", 896.0), ("input_size", True),
    ("physical_batch_size", 16), ("physical_batch_size", 8.0), ("physical_batch_size", True),
    ("accumulation_steps", 1), ("accumulation_steps", 2.0), ("accumulation_steps", True),
    ("input_size", None), ("physical_batch_size", None), ("accumulation_steps", None),
])
def test_resolution_896_authority_rejects_other_bound_dimensions_before_native_access(
        resolution_896_policy_bundle, bound, tmp_path, key, value):
    bound = copy.deepcopy(bound)
    bound["config"].update(input_size=896, physical_batch_size=8, accumulation_steps=2, cuda_gpu_uuid=UUID)
    if value is None:
        del bound["config"][key]
    else:
        bound["config"][key] = value
    bound["binding_sha256"] = ev.canonical_sha256({k: v for k, v in bound.items() if k != "binding_sha256"})
    output = tmp_path / "never-started"
    with pytest.raises(ad.MonitoredHardwareError, match="896 authorization requires exact bound"):
        ad.MonitoredHardwareSession(bound, resolution_896_policy_bundle, output, "paired_smoke", 600)
    assert not output.exists()


@pytest.mark.parametrize("authority", ["original", "repair"])
def test_prior_authorities_do_not_inherit_resolution_896_binding_restrictions(
        policy_bundle, repair_policy_bundle, bound, tmp_path, authority):
    bound = copy.deepcopy(bound)
    bound["config"].update(input_size=640, physical_batch_size=16, accumulation_steps=1, cuda_gpu_uuid=UUID)
    bound["binding_sha256"] = ev.canonical_sha256({k: v for k, v in bound.items() if k != "binding_sha256"})
    bundle = policy_bundle if authority == "original" else repair_policy_bundle
    output = tmp_path / "legacy-never-started"
    session = ad.MonitoredHardwareSession(bound, bundle, output, "paired_smoke", 600)
    assert session._state == "new"
    assert not output.exists()



@pytest.mark.parametrize("deadline", [0, -1, 601, 43200, True, False, 600.0, "600", None])
def test_synthetic_diagnostic_has_strict_six_hundred_second_limit(repair_policy_bundle, deadline):
    with pytest.raises(ad.MonitoredHardwareError, match="not authorized"):
        ad._validate_policy(repair_policy_bundle, "synthetic_operator_diagnostic", deadline)


@pytest.mark.parametrize("authority", ["original", "repair"])
def test_caller_rehash_cannot_expand_authorized_scope(policy_bundle, repair_policy_bundle, authority):
    bundle = copy.deepcopy(policy_bundle if authority == "original" else repair_policy_bundle)
    bundle["authorized_scope_limits_seconds"]["synthetic_operator_diagnostic"] = 601
    bundle["bundle_sha256"] = ev.canonical_sha256({k: v for k, v in bundle.items() if k != "bundle_sha256"})
    with pytest.raises(ad.MonitoredHardwareError, match="differs"):
        ad._validate_policy(bundle, "synthetic_operator_diagnostic", 601)


@pytest.mark.parametrize("change", ["changed_file", "unknown_authority"])
def test_repair_authorization_requires_reviewed_complete_file(repair_policy_bundle, change):
    ref = copy.deepcopy(repair_policy_bundle["authorization_reference"])
    Path(ref["path"]).write_bytes(Path(ref["path"]).read_bytes() + b"changed\n")
    if change == "unknown_authority":
        ref = ev.file_reference(ref["path"])
    with pytest.raises(ad.MonitoredHardwareError, match="authorization reference differs"):
        ad.build_policy_bundle(Path(repair_policy_bundle["authority"]["path"]).parent,
                               authorization_reference=ref)


@pytest.mark.parametrize("input_kind", ["synthetic", "manifest_binding_only"])
def test_diagnostic_scope_requires_synthetic_binding_before_native_access(
        repair_policy_bundle, bound, tmp_path, monkeypatch, input_kind):
    def forbidden(*args, **kwargs):
        pytest.fail("diagnostic constructor attempted native hardware access")
    monkeypatch.setattr(hw, "collect_native_hardware_probe", forbidden)
    bound = copy.deepcopy(bound)
    bound["config"]["cuda_gpu_uuid"] = UUID
    if input_kind == "manifest_binding_only":
        manifest = tmp_path / "cpu-only-manifest.json"
        manifest.write_text('{"fixture": "no real data"}\n')
        bound["data"] = {"kind": input_kind, "manifests": {"train_core": ev.file_reference(manifest)},
                         "dataset_executed": False}
    bound["binding_sha256"] = ev.canonical_sha256({k: v for k, v in bound.items() if k != "binding_sha256"})
    ev.validate_run_binding(bound)
    root = tmp_path / "diagnostic-never-started"
    if input_kind == "synthetic":
        session = ad.MonitoredHardwareSession(bound, repair_policy_bundle, root, "synthetic_operator_diagnostic")
        assert session._state == "new"
        assert session.policy["workload_deadline_seconds"] == 600
    else:
        with pytest.raises(ad.MonitoredHardwareError, match="requires a synthetic input binding"):
            ad.MonitoredHardwareSession(bound, repair_policy_bundle, root, "synthetic_operator_diagnostic")
    assert not root.exists()


def test_policy_authority_file_mutation_is_rejected(policy_bundle):
    Path(policy_bundle["authority_files"]["xorg-identity-policy.json"]["path"]).write_text("{}\n")
    with pytest.raises(ev.EvidenceError, match="hash mismatch"):
        ad._validate_policy(policy_bundle, "paired_smoke", 600)


@pytest.fixture
def external_receipt(tmp_path, monkeypatch, policy_bundle):
    """Synthetic user/journal evidence; never a hardware authorization."""
    root = tmp_path / "external-clock"
    root.mkdir()
    terminal = root / "terminal.txt"
    terminal.write_text('GPU clocks set to "(gpuClkMin 1500, gpuClkMax 1500)" for GPU '
                        + policy_bundle["pci_bus_id"] + '\nAll done.\n' + ad._EXTERNAL_TERMINAL_TRAILER + '\n')
    attestation = root / "attestation.txt"
    attestation.write_text("CPU fixture only; no GPU or setter permitted\n")
    monkeypatch.setattr(ad, "_EXTERNAL_ADMIN_ATTESTATION_SHA", ev.file_reference(attestation)["sha256"])
    binary = root / "journalctl"
    binary.write_bytes(b"CPU fake journal query binary")
    argv = [policy_bundle["expected_nvidia_smi_executable"]["path"], "-i", UUID, "--lock-gpu-clocks=1500,1500"]
    common = {"_HOSTNAME": policy_bundle["host"], "_BOOT_ID": BOOT.replace("-", ""),
              "_PID": "2911481", "_UID": "1000", "_AUDIT_LOGINUID": "1000", "_EXE": "/usr/bin/sudo",
              "_COMM": "sudo", "SYSLOG_IDENTIFIER": "sudo", "_CMDLINE": "sudo " + " ".join(argv),
              "_AUDIT_SESSION": "1366", "_SYSTEMD_SESSION": "1366",
              "_SYSTEMD_INVOCATION_ID": "fixture-invocation", "_MACHINE_ID": "fixture-machine"}
    messages = ["     lyy : TTY=pts/1 ; PWD=/ ; USER=root ; COMMAND=" + " ".join(argv),
                "pam_unix(sudo:session): session opened for user root(uid=0) by lyy(uid=1000)",
                "pam_unix(sudo:session): session closed for user root"]
    rows = [{**common, "MESSAGE": message, "_GID": "1000" if i == 0 else "0", "__CURSOR": "sudo:" + str(i),
             "__MONOTONIC_TIMESTAMP": str(10_000 + i), "__REALTIME_TIMESTAMP": str(1_000_000 + i)}
            for i, message in enumerate(messages)]
    receipt = {"schema_version": 1, "kind": "reviewed_external_admin_clock_receipt",
               "provenance": ad._EXTERNAL_CLOCK_PROVENANCE, "locally_executed_setter": False,
               "native_exitcode": None, "native_exit_code": None, "native_return_success": None,
               "native_journal_query_returncode_is_setter_exitcode": False,
               "reported_acknowledgement_success": True, "acknowledgement_verified": True,
               "success_ack_observed": True, "locked_upper_readback_verified": False,
               "stdout_source": "user_supplied_terminal", "time_source": "native_sudo_journal_record",
               "separate_stderr_capture_available": False, "setter_loaded_nvml_library": None,
               "transport_line": ad._EXTERNAL_TERMINAL_TRAILER, "clock_min_mhz": 1500, "clock_max_mhz": 1500,
               **{key: policy_bundle[key] for key in ("host", "boot_id", "gpu_uuid", "pci_bus_id")},
               "user_attestation": ev.file_reference(attestation), "terminal_transcript": ev.file_reference(terminal),
               "command_argv": argv, "native_sudo_pid": 2911481,
               "native_command_realtime_timestamp_us": rows[0]["__REALTIME_TIMESTAMP"]}
    def save_rows():
        raw = b"\n".join(ev.canonical_json_bytes(row) for row in rows) + b"\n"
        (root / "journal.jsonl").write_bytes(raw)
        receipt["native_command_journal"] = ev.file_reference(root / "journal.jsonl")
        receipt["command_journal_record_sha256"] = ev.canonical_sha256(rows[0])
        query_argv = [str(binary), "-b", "_PID=2911481", "--output=json", "--no-pager"]
        query = {**capture(query_argv, raw.decode()), "argv": query_argv, "executable": ev.file_reference(binary)}
        (root / "query.json").write_bytes(ev.canonical_json_bytes(query))
        receipt["native_journal_query"] = ev.file_reference(root / "query.json")
        prior = {"status": "fulfilled", "value": {"exit_code": 0,
            "output": raw.decode() + "-- cursor: " + rows[-1]["__CURSOR"] + "\n"}}
        (root / "prior.json").write_bytes(ev.canonical_json_bytes(prior))
        receipt["prior_journal_tool_capture"] = ev.file_reference(root / "prior.json")
    def publish():
        path = root / "reviewed-receipt.json"
        path.write_bytes(ev.canonical_json_bytes(receipt))
        reference = ev.file_reference(path)
        monkeypatch.setattr(ad, "_EXTERNAL_CLOCK_RECEIPT_SHA", reference["sha256"])
        return reference
    def build(reference=None):
        return ad.build_policy_bundle(Path(policy_bundle["authority"]["path"]).parent,
            authorization_reference=policy_bundle["authorization_reference"], setter_mode=ad._EXTERNAL_ADMIN_MODE,
            external_clock_receipt=reference or publish())
    save_rows()
    return {"receipt": receipt, "rows": rows, "reference": publish(), "root": root,
            "publish": publish, "save_rows": save_rows, "build": build, "policy": policy_bundle}


def test_external_clock_evidence_is_reported_success_with_unknown_native_exit(external_receipt, observation):
    fixture = external_receipt
    bundle = fixture["build"](fixture["reference"])
    policy = ad._validate_policy(bundle, "paired_smoke", 600)
    result = ad._external_clock_evidence(policy, observation[0])
    assert result["command"] is None and result["locally_executed_setter"] is False
    assert result["native_exitcode"] is None and result["native_exit_code"] is None
    assert result["native_return_success"] is None
    assert result["reported_acknowledgement_success"] is True
    assert result["provenance"] == ad._EXTERNAL_CLOCK_PROVENANCE
    assert result["locked_upper_readback_verified"] is False
    assert policy["clock_max_mhz"] == 1500
    assert policy["minimum_preload_clock_samples"] == policy["minimum_loaded_clock_samples"] == 3
    assert policy["clock_period_seconds"] == .2
    assert policy["clock_max_gap_seconds"] == 1 and policy["health_max_gap_seconds"] == 2
    assert "clock" in ad._deadline_error(1_000_000_001, {"clock": 0, "health": 0}, policy, 0)
    with pytest.raises(ad.MonitoredHardwareError, match="cannot execute"):
        ad._setter(policy, observation[0])


@pytest.mark.parametrize("field,value", [("gpu_uuid", "wrong"), ("pci_bus_id", "00000000:02:00.0"),
    ("boot_id", "wrong"), ("clock_max_mhz", 1501), ("clock_min_mhz", 1000),
    ("native_exitcode", 0), ("native_return_success", True), ("reported_acknowledgement_success", False),
    ("locally_executed_setter", True), ("locked_upper_readback_verified", True),
    ("native_journal_query_returncode_is_setter_exitcode", True)])
def test_external_receipt_rejects_identity_range_or_fabricated_native_success(external_receipt, field, value):
    external_receipt["receipt"][field] = value
    with pytest.raises(ad.MonitoredHardwareError):
        external_receipt["build"]()


@pytest.mark.parametrize("change", ["missing_terminal", "incomplete_ack", "diagnostic", "wrong_close", "wrong_gpu"])
def test_external_original_stdout_is_required_and_only_exact_transport_trailer_is_removed(external_receipt, change):
    receipt = external_receipt["receipt"]
    if change == "missing_terminal":
        del receipt["terminal_transcript"]
    else:
        path = Path(receipt["terminal_transcript"]["path"])
        raw = path.read_text()
        if change == "incomplete_ack": raw = raw.replace("All done.\n", "")
        elif change == "diagnostic": raw += "Warning: incomplete setting\n"
        elif change == "wrong_close": raw = raw.replace("192.168.0.198", "192.168.0.199")
        else: raw = raw.replace("00000000:01:00.0", "00000000:02:00.0")
        path.write_text(raw)
        receipt["terminal_transcript"] = ev.file_reference(path)
    with pytest.raises(ad.MonitoredHardwareError):
        external_receipt["build"]()


@pytest.mark.parametrize("change", ["uid", "exe", "boot", "pid", "command", "open", "close", "order", "query_exit"])
def test_external_native_sudo_journal_requires_real_user_and_privileged_session_chain(external_receipt, change):
    rows = external_receipt["rows"]
    if change in {"uid", "exe", "boot", "pid"}:
        field, value = {"uid": ("_UID", "0"), "exe": ("_EXE", "/tmp/sudo"),
                        "boot": ("_BOOT_ID", "wrong"), "pid": ("_PID", "2911482")}[change]
        rows[0][field] = value
    elif change == "command": rows[0]["MESSAGE"] = rows[0]["MESSAGE"].replace("1500,1500", "1500,1501")
    elif change == "open": rows[1]["MESSAGE"] = "session opened without native root identity"
    elif change == "close": rows[2]["MESSAGE"] = "session closed without user identity"
    elif change == "order": rows[2]["__MONOTONIC_TIMESTAMP"] = rows[0]["__MONOTONIC_TIMESTAMP"]
    external_receipt["save_rows"]()
    if change == "query_exit":
        receipt = external_receipt["receipt"]
        path = Path(receipt["native_journal_query"]["path"])
        query = json.loads(path.read_bytes())
        query["returncode"] = 1
        path.write_bytes(ev.canonical_json_bytes(query))
        receipt["native_journal_query"] = ev.file_reference(path)
    with pytest.raises((ad.MonitoredHardwareError, hw.HardwareGateError)):
        external_receipt["build"]()


def test_external_receipt_bytes_cannot_be_replaced_after_policy_freeze(external_receipt):
    bundle = external_receipt["build"]()
    path = Path(bundle["external_clock_receipt"]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ad.MonitoredHardwareError, match="reviewed complete bytes"):
        ad._validate_policy(bundle, "paired_smoke", 600)


@pytest.mark.parametrize("field", ["receipt", *ad._EXTERNAL_REFERENCE_FIELDS])
def test_each_external_evidence_reference_is_rechecked_before_runtime_native_query(external_receipt, field):
    bundle = external_receipt["build"]()
    policy = ad._validate_policy(bundle, "paired_smoke", 600)
    refs = ad._external_clock_integrity_references(policy)
    selected = bundle["external_clock_receipt"] if field == "receipt" else external_receipt["receipt"][field]
    path = Path(selected["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    # The autouse forbidden _command proves the drift is caught first.
    with pytest.raises(ad.MonitoredHardwareError, match="external-evidence identity changed"):
        ad._health_sample({"policy": policy, "integrity_references": refs}, "fixture", {}, owner_alive=True)


@pytest.mark.parametrize("mode,has_receipt", [(ad._EXTERNAL_ADMIN_MODE, False), ("direct", True), ("sudo_n", True)])
def test_external_clock_evidence_cannot_silently_change_native_setter_modes(policy_bundle, external_receipt, mode, has_receipt):
    with pytest.raises(ad.MonitoredHardwareError, match="required only"):
        ad.build_policy_bundle(Path(policy_bundle["authority"]["path"]).parent, setter_mode=mode,
            authorization_reference=policy_bundle["authorization_reference"],
            external_clock_receipt=external_receipt["reference"] if has_receipt else None)


def test_external_session_start_never_calls_native_setter_and_binds_all_runtime_refs(
        external_receipt, bound, observation, tmp_path, monkeypatch):
    bundle = external_receipt["build"]()
    bound = copy.deepcopy(bound)
    bound["config"]["cuda_gpu_uuid"] = UUID
    bound["binding_sha256"] = ev.canonical_sha256({k: v for k, v in bound.items() if k != "binding_sha256"})
    session = ad.MonitoredHardwareSession(bound, bundle, tmp_path / "external-run", "paired_smoke")
    native, _ = observation
    class FakeProbe:
        def as_dict(self): return copy.deepcopy(native)
    monkeypatch.setattr(hw, "collect_native_hardware_probe", lambda *a, **kw: FakeProbe())
    monkeypatch.setattr(ad, "_xorg_identity", lambda p: {"fixture": True})
    monkeypatch.setattr(ad, "_edac_observation", lambda: {"controllers": []})
    monkeypatch.setattr(ad, "_setter", lambda *a, **kw: pytest.fail("external start attempted a native setter"))
    rows = external_receipt["rows"]
    calls = []
    def query(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == bundle["external_journal_executable"]["path"] and kwargs["timeout"] == .7
        return {**capture(argv, "\n".join(json.dumps(row) for row in rows) + "\n-- cursor: " + rows[-1]["__CURSOR"] + "\n"),
                "executable": bundle["external_journal_executable"]}
    monkeypatch.setattr(ad, "_command", query)
    monkeypatch.setattr(ad, "_open_pidfd", lambda pid: os.open(external_receipt["receipt"]["user_attestation"]["path"], os.O_RDONLY))
    class FakeGuardian:
        pid = 77777
    monkeypatch.setattr(ad.subprocess, "Popen", lambda *a, **kw: FakeGuardian())
    monkeypatch.setattr(session, "_wait_response", lambda *a, **kw: {})
    try:
        session.start()
        assert len(calls) == 1
        saved = json.loads((session.root / "setter-receipt.json").read_bytes())
        assert saved["native_return_success"] is None and saved["locally_executed_setter"] is False
        startup = json.loads((session.root / "startup.json").read_bytes())
        assert "native_after_external_clock_review" in startup and "native_post_setter" not in startup
        spec = json.loads((session.root / "monitor-spec.json").read_bytes())
        assert spec["external_sudo_anchor"]["cursor"] == rows[-1]["__CURSOR"]
        for ref in ad._external_clock_integrity_references(session.policy):
            assert ref in spec["integrity_references"]
        assert startup["strict_native_gate"] == "BLOCKED"
    finally:
        if session._channel is not None: session._channel.close()


@pytest.mark.parametrize("option", ["-lgc=1500,1500", "--lock-gpu-clocks=1500,1500", "-rgc", "--reset-gpu-clocks",
    "-ac=810,1500", "--applications-clocks=810,1500", "-rac", "--reset-applications-clocks", "-r", "--gpu-reset"])
def test_later_trusted_sudo_clock_or_reset_command_invalidates_external_event(option):
    row = {"_EXE": "/usr/bin/sudo", "_COMM": "sudo",
           "MESSAGE": "lyy : USER=root ; COMMAND=/usr/bin/nvidia-smi -i " + UUID + " " + option}
    assert ad._known_sudo_clock_mutation(row) is True


@pytest.mark.parametrize("command", ["/usr/bin/nvidia-smi -q", "/usr/bin/sudo -n -l -- /usr/bin/nvidia-smi -rgc",
    "/usr/bin/cat /tmp/nvidia-smi--lock-gpu-clocks.txt", "/usr/bin/echo nvidia-smi --reset-gpu-clocks"])
def test_sudo_readonly_permission_queries_and_arbitrary_text_are_not_clock_mutations(command):
    row = {"_EXE": "/usr/bin/sudo", "_COMM": "sudo", "MESSAGE": "lyy : USER=root ; COMMAND=" + command}
    assert ad._known_sudo_clock_mutation(row) is False
    row.update(_CMDLINE="sudo /usr/bin/nvidia-smi -rgc", MESSAGE="pam_unix(sudo:session): session closed for user root")
    assert ad._known_sudo_clock_mutation(row) is False
    row.update(_EXE="/usr/bin/echo", MESSAGE="lyy : USER=root ; COMMAND=/usr/bin/nvidia-smi -rgc")
    assert ad._known_sudo_clock_mutation(row) is False


@pytest.mark.parametrize("change", [None, "anchor", "boot", "cursor", "mutation", "collector"])
def test_external_sudo_cursor_continuity_and_rejection_preserve_native_capture(external_receipt, tmp_path, monkeypatch, change):
    policy = external_receipt["build"]()
    rows = copy.deepcopy(external_receipt["rows"])
    anchor = {"cursor": rows[0]["__CURSOR"], "record_sha256": ev.canonical_sha256(rows[0])}
    if change == "anchor": rows[0]["PRIORITY"] = "changed"
    elif change == "boot": rows[-1]["_BOOT_ID"] = "wrong"
    elif change == "mutation":
        rows.append({**rows[-1], "__CURSOR": "sudo:3", "__MONOTONIC_TIMESTAMP": "10003",
                     "MESSAGE": "lyy : USER=root ; COMMAND=/usr/bin/nvidia-smi --reset-gpu-clocks"})
    trailer = "wrong" if change == "cursor" else rows[-1]["__CURSOR"]
    text = "\n".join(json.dumps(row) for row in rows) + "\n-- cursor: " + trailer + "\n"
    calls = []
    def query(argv, **kwargs):
        calls.append((argv, kwargs))
        return {**capture(argv, text), "executable": {**policy["external_journal_executable"],
                **({"sha256": "f" * 64} if change == "collector" else {})}}
    monkeypatch.setattr(ad, "_command", query)
    if change:
        with pytest.raises(ad.MonitoredHardwareError):
            ad._external_sudo_journal_sample(policy, anchor, evidence_dir=tmp_path)
        saved = json.loads((tmp_path / "external-clock-journal-rejection.json").read_bytes())
        assert saved["native_query"]["stdout"]["utf8"] == text
    else:
        result = ad._external_sudo_journal_sample(policy, anchor, evidence_dir=tmp_path)
        assert result["records"] == 3 and result["last_cursor"] == rows[-1]["__CURSOR"]
        assert result["known_later_clock_mutations"] == 0
        assert result["locked_upper_readback_verified"] is False
    assert calls[0][1]["timeout"] == .7
    assert calls[0][0][0] == policy["external_journal_executable"]["path"]
    assert calls[0][0][-2:] == ["_COMM=sudo", "_EXE=/usr/bin/sudo"]


def test_start_failure_closes_pidfd_and_preserves_setter_attempt(policy_bundle, bound, observation, tmp_path, monkeypatch):
    bound = copy.deepcopy(bound)
    bound["config"]["cuda_gpu_uuid"] = UUID
    bound["binding_sha256"] = ev.canonical_sha256({k: v for k, v in bound.items() if k != "binding_sha256"})
    session = ad.MonitoredHardwareSession(bound, policy_bundle, tmp_path / "run", "paired_smoke")
    native, _ = observation
    class FakeProbe:
        def as_dict(self): return copy.deepcopy(native)
    monkeypatch.setattr(hw, "collect_native_hardware_probe", lambda *a, **kw: FakeProbe())
    monkeypatch.setattr(ad, "_xorg_identity", lambda p: {"fixture": True})
    monkeypatch.setattr(ad, "_edac_observation", lambda: {"controllers": []})
    attempts = []
    def setter(*args):
        attempts.append(1)
        return {"validation_errors": ["permission denied"], "native_return_success": False,
                "command": {"returncode": 4}, "no_retry_or_fallback": True}
    monkeypatch.setattr(ad, "_setter", setter)
    opened = []
    def open_fd(pid):
        fd = os.open(tmp_path / "user-authorization.txt", os.O_RDONLY)
        opened.append(fd)
        return fd
    monkeypatch.setattr(ad, "_open_pidfd", open_fd)
    with pytest.raises(ad.MonitoredHardwareError, match="permission denied"):
        session.start()
    assert len(attempts) == 1
    with pytest.raises(OSError): os.fstat(opened[0])
    assert json.loads((session.root / "setter-receipt.json").read_bytes())["command"]["returncode"] == 4
    with pytest.raises(ad.MonitoredHardwareError, match="cannot be retried"): session.start()


def test_edac_unavailable_is_not_zero_and_coverage_changes_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "_EDAC", tmp_path)
    missing = ad._edac_observation()
    assert missing["controller_counters_available"] is False
    assert missing["unavailable_means"] == "unknown_not_zero"
    ad._check_edac(missing, missing)
    controller = tmp_path / "mc0"
    controller.mkdir()
    (controller / "ce_count").write_text("0\n")
    (controller / "ue_count").write_text("0\n")
    zeros = ad._edac_observation()
    ad._check_edac(zeros, None)
    with pytest.raises(ad.MonitoredHardwareError, match="coverage"):
        ad._check_edac(zeros, missing)
    (controller / "ce_count").write_text("1\n")
    with pytest.raises(ad.MonitoredHardwareError, match="nonzero"):
        ad._check_edac(ad._edac_observation(), None)


@pytest.mark.parametrize("clock,sm,util,free", [(1501, 1500, 50, 1000), (1500, 1501, 50, 1000),
    ("N/A", 1500, 50, 1000), (1500, "N/A", 50, 1000), (1500, 1500, 101, 1000), (1500, 1500, 50, 63)])
def test_clock_sample_fails_closed(monkeypatch, clock, sm, util, free):
    monkeypatch.setattr(ad, "_command", lambda argv, **kw: capture(argv, f"{UUID}, {clock}, {sm}, {util}, 2000, {free}\n"))
    with pytest.raises(ad.MonitoredHardwareError):
        ad._clock_sample({"nvidia_smi_path": "/fixture/nvidia-smi", "policy": {
            "gpu_uuid": UUID, "expected_nvidia_smi_executable": None,
            "hardware_policy": {"interval_min_free_memory_mib": 64}}})


def test_clock_sample_keeps_capture_and_is_not_locked_readback(monkeypatch):
    calls = []
    def fake(argv, **kw):
        calls.append((argv, kw))
        return capture(argv, f"{UUID}, 1500, 1500, 80, 2000, 22000\n")
    monkeypatch.setattr(ad, "_command", fake)
    row = ad._clock_sample({"nvidia_smi_path": "/fixture/nvidia-smi", "policy": {
        "gpu_uuid": UUID, "expected_nvidia_smi_executable": None,
        "hardware_policy": {"interval_min_free_memory_mib": 64}}})
    assert row["clock_mhz"] == 1500
    assert row["graphics_clock_mhz"] == row["sm_clock_mhz"] == 1500
    assert row["command"]["stdout"]["sha256"]
    assert calls[0][1]["timeout"] < 1
    assert "clocks.current.graphics,clocks.current.sm" in calls[0][0][2]


def test_admission_cannot_be_constructed_from_json_or_pickled():
    with pytest.raises(ad.MonitoredHardwareError): ad.MonitoredHardwareAdmission(object(), {})
    token = ad.MonitoredHardwareAdmission(ad._TOKEN, object())
    with pytest.raises(ad.MonitoredHardwareError): pickle.dumps(token)
    with pytest.raises(ad.MonitoredHardwareError): token._session = object()
    with pytest.raises(ad.MonitoredHardwareError):
        ad.require_monitored_hardware_admission({}, binding={}, expected_gpu_uuid=UUID)


def test_dispatch_does_not_change_legacy_json_rejection():
    with pytest.raises(hw.HardwareGateError, match="supplied observation JSON"):
        hw.require_native_hardware_admission({}, binding={}, expected_gpu_uuid=UUID)


def test_dispatch_routes_only_exact_new_type(monkeypatch):
    token = ad.MonitoredHardwareAdmission(ad._TOKEN, object())
    sentinel = {"fixture": True}
    monkeypatch.setattr(ad, "require_monitored_hardware_admission", lambda probe, **kw: sentinel)
    assert hw.require_native_hardware_admission(token, binding={}, expected_gpu_uuid=UUID) is sentinel


def test_dispatch_accepts_same_source_admission_loaded_under_orchestration_alias(bound, observation, monkeypatch):
    """The historical runtime and current orchestration must share one ABI."""
    alias = "_t8b_admission_alias_fixture"
    package = types.ModuleType(alias)
    package.__path__ = [str(Path(ad.__file__).parent)]
    sys.modules[alias] = package
    loaded = {}
    try:
        for name in ("training_v2b_evidence", "training_v2b_hardware", "training_v2b_admission"):
            qualified = alias + "." + name
            spec = importlib.util.spec_from_file_location(qualified, Path(ad.__file__).parent / (name + ".py"))
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        alias_ad = loaded["training_v2b_admission"]
        alias_hw = loaded["training_v2b_hardware"]
        alias_ev = loaded["training_v2b_evidence"]
        session = object.__new__(alias_ad.MonitoredHardwareSession)
        session.binding = copy.deepcopy(bound)
        session.policy = {"gpu_uuid": UUID}
        session._state = "running"
        session._owner = alias_hw._process_identity(os.getpid())
        session._nonce = "a" * 64
        session._source_refs = [alias_ev.file_reference(path) for path in
                                (alias_ad.__file__, alias_hw.__file__, alias_ev.__file__)]
        session._channel = socket.socket()
        session._child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        session._guardian_identity = alias_hw._process_identity(session._child.pid)
        telemetry = {"clock": {"started_ns": time.monotonic_ns()},
                     "health_started_ns": time.monotonic_ns()}
        session._wait_response = lambda operation: telemetry
        monkeypatch.setattr(alias_ad, "_send", lambda *args, **kwargs: None)
        probe = alias_ad.MonitoredHardwareAdmission(alias_ad._TOKEN, session)
        session._admission = probe
        assert hw.require_native_hardware_admission(
            probe, binding=bound, expected_gpu_uuid=UUID) == telemetry
    finally:
        session_channel = locals().get("session", None)
        if session_channel is not None and getattr(session_channel, "_channel", None) is not None:
            session_channel._channel.close()
        if session_channel is not None and getattr(session_channel, "_child", None) is not None:
            session_channel._child.terminate()
            session_channel._child.wait(timeout=5)
        for name in ("training_v2b_admission", "training_v2b_hardware", "training_v2b_evidence"):
            sys.modules.pop(alias + "." + name, None)
        sys.modules.pop(alias, None)


def test_dispatch_rejects_lookalike_admission_from_unreviewed_module():
    class MonitoredHardwareAdmission:
        __slots__ = ("_session",)
        def __init__(self):
            self._session = object()
    with pytest.raises(hw.HardwareGateError, match="supplied .*JSON"):
        hw.require_native_hardware_admission(
            MonitoredHardwareAdmission(), binding={}, expected_gpu_uuid=UUID)


@pytest.mark.parametrize("mode", ["direct", "sudo_n"])
def test_setter_is_one_frozen_attempt_and_failure_receipt_is_preserved(tmp_path, monkeypatch, mode):
    binary, library = tmp_path / "nvidia-smi", tmp_path / "libnvidia-ml.so.580.178.04"
    binary.write_bytes(b"fake setter")
    library.write_bytes(b"fake library")
    monkeypatch.setattr(ad.shutil, "which", lambda name: str(binary))
    monkeypatch.setattr(hw, "_boot_id", lambda: BOOT)
    calls = []
    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        row = capture(argv, "permission denied\n", stderr="")
        row["returncode"] = 4
        return row
    monkeypatch.setattr(ad, "_command", fake)
    result = ad._setter({"setter_mode": mode, "gpu_uuid": UUID, "boot_id": BOOT}, {
        "commands": {"gpu_xml": {"executable": ev.file_reference(binary)}},
        "capability_inventory": {"nvml_library": ev.file_reference(library)}})
    assert len(calls) == 1
    assert calls[0][0][-2:] == ["--id=" + UUID, "--lock-gpu-clocks=1500,1500"]
    assert (calls[0][0][:3] == ["sudo", "-n", "--"]) == (mode == "sudo_n")
    assert result["native_return_success"] is False
    assert result["command"]["returncode"] == 4
    assert result["validation_errors"]
    assert result["locked_upper_readback_verified"] is False


@pytest.mark.parametrize("text", [
    'GPU clocks set to "(1500, 1500)" for GPU 00000000:01:00.0\nAll done.\n',
    'GPU clocks set to "(1500, 1500)" for GPU ' + UUID + '\nAll done.\n',
    'GPU clocks set to "(gpuClkMin 1500, gpuClkMax 1500)" for GPU 00000000:01:00.0\nAll done.\n',
    'GPU clocks set to "(gpuClkMin 1500, gpuClkMax 1500)" for GPU ' + UUID + '\nAll done.\n',
])
def test_exact_setter_acknowledgement_is_not_readback(text):
    result = ad._setter_acknowledgement(text, {"gpu_uuid": UUID, "pci_bus_id": "00000000:01:00.0"})
    assert result["min_mhz"] == result["max_mhz"] == 1500
    assert "not_locked_clock_readback" in result["meaning"]


@pytest.mark.parametrize("text", [
    'GPU clocks set to "(1500, 1501)" for GPU 00000000:01:00.0\nAll done.\n',
    'GPU clocks set to "(1500, 1500)" for GPU 00000000:02:00.0\nAll done.\n',
    'GPU clocks set to "(gpuClkMin 1500, gpuClkMax 1501)" for GPU 00000000:01:00.0\nAll done.\n',
    'GPU clocks set to "(1500, 1500)" for GPU 00000000:01:00.0\n',
    'GPU clocks set to "(1500, 1500)" for GPU 00000000:01:00.0\nAll done.\nAll done.\n',
    'GPU clocks set to "(1500, 1500)" for GPU 00000000:01:00.0\nAll done.\nConnection to 192.168.0.198 closed.\n',
    'GPU clocks set to "(gpuClkMin 1500, 1500)" for GPU 00000000:01:00.0\nAll done.\n',
    "All done.\n", "permission denied\n", "", "Warning: unsafe clock request\n",
])
def test_setter_acknowledgement_rejects_wrong_target_range_or_diagnostics(text):
    with pytest.raises(ad.MonitoredHardwareError):
        ad._setter_acknowledgement(text, {"gpu_uuid": UUID, "pci_bus_id": "00000000:01:00.0"})


def test_atomic_publication_never_exposes_partial_json_or_overwrites(tmp_path, monkeypatch):
    target = tmp_path / "complete.json"
    original = os.link
    original_reference = ad.ev.file_reference
    observed = []
    observed_nlinks = []
    def link(source, destination, **kwargs):
        assert not target.exists()
        assert json.loads(Path(source).read_bytes()) == {"complete": True}
        observed.append(True)
        return original(source, destination, **kwargs)
    def reference(path):
        absolute = Path(path).absolute()
        if absolute == target.absolute():
            observed_nlinks.append(absolute.lstat().st_nlink)
        return original_reference(path)
    monkeypatch.setattr(ad.os, "link", link)
    monkeypatch.setattr(ad.ev, "file_reference", reference)
    reference = ad._publish_exclusive(target, {"complete": True})
    assert observed and original_reference(target) == reference
    assert observed_nlinks == [1]
    monkeypatch.setattr(ad.os, "link", original)
    with pytest.raises(FileExistsError): ad._publish_exclusive(target, {"complete": False})
    assert json.loads(target.read_bytes()) == {"complete": True}
    assert not list(tmp_path.glob(".*.tmp"))


def test_deadlines_use_query_start_and_bound_scope():
    p = {"clock_max_gap_seconds": 1., "health_max_gap_seconds": 2., "workload_deadline_seconds": 10}
    assert ad._deadline_error(1_000_000_000, {"clock": 0, "health": 0}, p, 0) is None
    assert "clock" in ad._deadline_error(1_000_000_001, {"clock": 0, "health": 0}, p, 0)
    assert "health" in ad._deadline_error(2_000_000_001, {"clock": 2_000_000_000, "health": 0}, p, 0)
    assert "workload" in ad._deadline_error(11_000_000_000, {"clock": 11_000_000_000, "health": 11_000_000_000}, p, 0)


def _cpu_worker(fd):
    os.read(fd, 1)


@pytest.mark.parametrize("stall", [
    "clock", "health", "sudo_mutation", "owner_identity", "owner_late_identity",
    "owner_wait_health_gap", "no_finish", "footer", "slow_footer", "release_gap", "unloaded_formal", None,
])
def test_independent_watchdog_stops_only_owned_cpu_worker_or_waits_for_exit(tmp_path, monkeypatch, stall):
    if sys.platform != "linux":
        pytest.skip("Linux pidfd is required for this process-level CPU test")
    ctx = mp.get_context("fork")
    worker_read, worker_write = os.pipe()
    bystander_read, bystander_write = os.pipe()
    worker = ctx.Process(target=_cpu_worker, args=(worker_read,))
    bystander = ctx.Process(target=_cpu_worker, args=(bystander_read,))
    worker.start(); bystander.start()
    owner = hw._process_identity(worker.pid)
    assert owner["readable"] is True
    assert worker.pid != os.getpid() and worker.pid != bystander.pid
    release_observations = []
    def clock(spec):
        if stall == "clock": time.sleep(5.)
        now = time.monotonic_ns()
        return {"kind": "clock", "started_ns": now, "finished_ns": now,
                "utilization_percent": 0 if stall == "unloaded_formal" else 50,
                "clock_mhz": 1500, "memory_used_mib": 100, "memory_free_mib": 24000}
    def health(spec, cursor, edac, *, owner_alive):
        started = time.monotonic_ns()
        if stall == "health": time.sleep(5.)
        if stall == "sudo_mutation":
            raise ad._ExternalSudoJournalError("CPU fixture later sudo clock command", {"native_query": "CPU fixture"})
        if stall in {"owner_identity", "owner_late_identity"}:
            if stall == "owner_late_identity": time.sleep(2.15)
            observation = {
                "kind": "owner_identity_observation", "decision": "REJECTED",
                "health_started_ns": started, "owner": owner,
                "observed_owner": {**owner, "start_ticks": owner["start_ticks"] + 1},
                "gpu_query": {"native_query": "CPU fixture only"},
                "gpu_processes": [{"pid": worker.pid, "type": "C", "identity": owner}],
                "original_pidfd": spec["_owner_pidfd"], "wait": None,
            }
            spec["_owner_identity_observation_sink"](copy.deepcopy(observation))
            raise ad._OwnerIdentityError("CPU fixture readable owner identity changed", observation)
        owner_observation = None
        if stall == "owner_wait_health_gap":
            # Keep the original health start. The main 2 s deadline must stop
            # the owned child while this health thread waits on its real pidfd.
            time.sleep(1.88)
            owner_observation = ad._observe_owner_identity(
                spec, {"native_query": "CPU fixture only"},
                [{"pid": worker.pid, "type": "C", "identity": _missing_owner_proc(owner)}],
                health_started_ns=started, owner_alive=owner_alive)
        pending = False
        if not owner_alive:
            release_observations.append(True)
            pending = stall == "release_gap" and len(release_observations) == 2
        row = {"kind": "health", "started_ns": started, "finished_ns": time.monotonic_ns(),
               "gpu": {"processes": [{"pid": worker.pid, "type": "C"}] if owner_alive or pending else []},
               "kernel_health": {"last_cursor": "cpu-fixture", "findings": []},
               "owner_gpu_release_pending": pending, "owner_pidfd_exited": not owner_alive}
        if owner_observation is not None:
            row["owner_identity_observation"] = owner_observation
        return row
    monkeypatch.setattr(ad, "_clock_sample", clock)
    monkeypatch.setattr(ad, "_health_sample", health)
    if stall == "owner_wait_health_gap":
        native_identity = hw._process_identity
        monkeypatch.setattr(hw, "_process_identity",
            lambda pid: _missing_owner_proc(owner) if pid == owner["pid"] else native_identity(pid))
    if stall in {"sudo_mutation", "owner_identity", "owner_late_identity", "owner_wait_health_gap"}:
        stopped = {"value": False}
        original_stop, original_write = ad._stop_owner, ad.ev.write_exclusive_json
        def stop_owned(fd):
            original_stop(fd)
            stopped["value"] = True
        def persist(path, value):
            if Path(path).name in {"external-clock-journal-rejection.json",
                                    "owner-identity-observation.json", "owner-identity-rejection.json"}:
                assert stopped["value"], "runtime rejection I/O preceded owned-worker stop"
            return original_write(path, value)
        monkeypatch.setattr(ad, "_stop_owner", stop_owned)
        monkeypatch.setattr(ad.ev, "write_exclusive_json", persist)
    if stall == "footer":
        original_gzip = ad.gzip.GzipFile
        class BadFooter(original_gzip):
            def __exit__(self, *args):
                super().__exit__(*args)
                raise OSError("CPU fixture footer/fsync failure")
        monkeypatch.setattr(ad.gzip, "GzipFile", BadFooter)
    elif stall == "slow_footer":
        original_gzip = ad.gzip.GzipFile
        class SlowFooter(original_gzip):
            def __exit__(self, *args):
                result = super().__exit__(*args)
                time.sleep(2.2)
                return result
        monkeypatch.setattr(ad.gzip, "GzipFile", SlowFooter)
    policy = {"clock_period_seconds": .04 if stall == "owner_wait_health_gap" else .2,
              "health_period_seconds": 1., "clock_max_gap_seconds": 1.,
              "health_max_gap_seconds": 2., "workload_deadline_seconds": 10,
              "owner_exit_timeout_seconds": 5., "post_exit_observation_seconds": 2.,
              "minimum_loaded_clock_samples": 3, "loaded_utilization_min_percent": 10,
              "minimum_preload_clock_samples": 3,
              "guardian_heartbeat_period_seconds": .04 if stall == "owner_wait_health_gap" else .1,
              "guardian_heartbeat_max_gap_seconds": 1.,
              "policy_sha256": "1" * 64, "gpu_uuid": UUID,
              "authorized_scope": "train_core_30epoch" if stall == "unloaded_formal" else "paired_smoke"}
    marker = tmp_path / "marker.json"
    marker.write_text("{}\n")
    spec = {"policy": policy, "evidence_dir": str(tmp_path), "owner": owner,
            "baseline": {"kernel_health": {"last_cursor": "cpu-fixture"}}, "edac": {},
            "run_id": "cpu-independent-watchdog", "binding_sha256": "2" * 64, "nonce": "private-fixture",
            "setter_reference": ev.file_reference(marker), "startup_reference": ev.file_reference(marker)}
    left, right = socket.socketpair()
    pidfd = ad._open_pidfd(worker.pid)
    guardian = ctx.Process(target=ad._watchdog_main, args=(spec, right, pidfd))
    guardian.start()
    right.close(); os.close(pidfd)
    try:
        if stall not in {"clock", "health", "sudo_mutation", "owner_identity",
                          "owner_late_identity", "owner_wait_health_gap"}:
            assert select.select([left], [], [], 3.)[0]
            rows, rest = ad._receive(left, b"")
            assert rows[0]["op"] == "ready"
            heartbeats = [json.loads(line) for line in (tmp_path / "guardian-heartbeat.jsonl").read_bytes().splitlines()]
            assert heartbeats[-1]["clock_samples"] >= 3
            assert heartbeats[-1]["phase"] == "running"
            assert heartbeats[-1]["guardian_identity"]["pid"] == guardian.pid
            time.sleep(.8)
            if stall != "no_finish":
                ad._send(left, {"op": "finish", "nonce": "private-fixture"})
                assert select.select([left], [], [], 2.)[0]
                rows, rest = ad._receive(left, rest)
                assert rows[0]["op"] == "finish"
                assert not (tmp_path / "monitor-final.json").exists()
            os.write(worker_write, b"x")
        guardian.join(7.)
        assert not guardian.is_alive()
        worker.join(2.)
        assert not worker.is_alive()
        assert bystander.is_alive()
        result = json.loads((tmp_path / "monitor-final.json").read_bytes())
        assert result["status"] == ("PASS" if stall in {None, "release_gap", "slow_footer"} else "FAIL")
        heartbeat_rows = [json.loads(line) for line in (tmp_path / "guardian-heartbeat.jsonl").read_bytes().splitlines()]
        for row in heartbeat_rows:
            assert row["monotonic_ns"] >= max(row["last_clock_started_ns"], row["last_health_started_ns"])
        assert result["sampler_shutdown"]["maximum_seconds"] == 1.
        if stall in {"clock", "health"}:
            assert stall + " observation gap exceeded" in result["failure"]
            assert worker.exitcode < 0
            assert result["sampler_shutdown"]["unfinished_threads"] == ["native-" + stall + "-sampler"]
        else:
            assert result["sampler_shutdown"]["unfinished_threads"] == []
        if stall == "sudo_mutation":
            assert "later sudo clock command" in result["failure"]
            assert worker.exitcode < 0
            assert json.loads((tmp_path / "external-clock-journal-rejection.json").read_bytes()) == {"native_query": "CPU fixture"}
        elif stall in {"owner_identity", "owner_late_identity"}:
            expected = "health observation gap exceeded" if stall == "owner_late_identity" else "readable owner identity changed"
            assert expected in result["failure"]
            assert worker.exitcode < 0
            rejection = json.loads((tmp_path / "owner-identity-rejection.json").read_bytes())
            assert rejection["gpu_query"] == {"native_query": "CPU fixture only"}
            assert rejection["observed_owner"]["start_ticks"] != owner["start_ticks"]
            assert json.loads((tmp_path / "owner-identity-observation.json").read_bytes()) == rejection
            assert any("readable owner identity changed" in error for error in result["sampler_shutdown"]["sample_errors"])
        elif stall == "owner_wait_health_gap":
            assert "health observation gap exceeded" in result["failure"]
            assert worker.exitcode < 0
            observation = json.loads((tmp_path / "owner-identity-observation.json").read_bytes())
            wait = observation["wait"]
            assert observation["decision"] == "EXIT_CONFIRMED" and wait["exit_confirmed"] is True
            assert 0 < wait["elapsed_ns"] <= 200_000_000
            records = [json.loads(line)["record"] for line in gzip.decompress(Path(result["samples"]["path"]).read_bytes()).splitlines()]
            assert not [r for r in records if r["kind"] == "health"], "late health publication crossed admission closure"
            assert any(wait["started_ns"] <= r["started_ns"] <= wait["finished_ns"] for r in records)
            assert any(wait["started_ns"] <= r["monotonic_ns"] <= wait["finished_ns"] for r in heartbeat_rows)
            assert len({r["last_health_started_ns"] for r in heartbeat_rows}) == 1
            assert all(not r["health_observed"] for r in heartbeat_rows)
            assert not (tmp_path / "owner-identity-rejection.json").exists()
        elif stall == "no_finish":
            assert "worker exited without finish declaration" in result["failure"]
            assert worker.exitcode == 0
            assert result["post_exit_health_observations"] == 0
        elif stall == "footer":
            assert "writer failed at close" in result["failure"]
            assert result["sampled_clock_compliance"] is False
            assert worker.exitcode == 0
        elif stall == "unloaded_formal":
            assert "insufficient loaded clock observations for train_core_30epoch" in result["failure"]
            assert result["loaded_clock_samples"] == 0
            assert result["sampled_clock_compliance"] is False
            assert worker.exitcode == 0
        elif stall in {None, "release_gap", "slow_footer"}:
            assert result["worker_exited"] is True
            assert result["post_exit_health_observations"] >= 2
            assert result["locked_upper_readback_verified"] is False
            assert result["loaded_clock_samples"] >= 3
            records = [json.loads(line) for line in gzip.decompress(Path(result["samples"]["path"]).read_bytes()).splitlines()]
            if stall == "release_gap":
                release_rows = [r["record"] for r in records
                                if r["record"]["kind"] == "health" and r["record"]["owner_pidfd_exited"]]
                assert [r["owner_gpu_release_pending"] for r in release_rows] == [False, True, False, False]
                assert result["post_exit_health_observations"] == 2
            previous = None
            for i, record in enumerate(records):
                assert record["sequence"] == i
                assert record["previous_sha256"] == previous
                body = {k: v for k, v in record.items() if k != "sha256"}
                assert ev.canonical_sha256(body) == record["sha256"]
                previous = record["sha256"]
            assert result["sample_hash_chain"] == {"records": len(records), "last_sha256": previous}
            assert ev.file_reference(tmp_path / "guardian-heartbeat.jsonl") == result["guardian_heartbeat"]
        for name, ref in result["owner_identity_evidence"].items():
            assert ev.file_reference(tmp_path / name) == ref
    finally:
        for fd in (worker_write, bystander_write):
            try: os.write(fd, b"x")
            except OSError: pass
        for process in (worker, bystander, guardian):
            process.join(.5)
            if process.is_alive(): process.terminate(); process.join(2.)
        left.close()
        for fd in (worker_read, worker_write, bystander_read, bystander_write):
            os.close(fd)


def _complete_monitor_payload(tmp_path, target):
    marker = tmp_path / "terminal-marker.json"
    marker.write_text("{}\n")
    marker_ref = ev.file_reference(marker)
    descriptor = {
        "monitor_final_report": str(target), "run_id": "fixture",
        "run_binding_sha256": "2" * 64, "policy_sha256": "1" * 64,
        "monitor_pid": 123, "owner": {"pid": 12},
        "guardian_identity": {"pid": 123}, "gpu_uuid": UUID,
    }
    payload = {
        **descriptor, "status": "PASS", "worker_exited": True,
        "samples": marker_ref, "setter_receipt": marker_ref,
        "startup_evidence": marker_ref, "guardian_heartbeat": marker_ref,
    }
    return descriptor, payload


def test_finish_reader_waits_for_atomic_hardlink_cleanup(tmp_path):
    target = tmp_path / "monitor-final.json"
    temporary = tmp_path / ".monitor-final.staged.tmp"
    descriptor, payload = _complete_monitor_payload(tmp_path, target)
    ev.write_exclusive_json(temporary, payload)
    os.link(temporary, target)
    assert target.lstat().st_nlink == 2
    cleanup = threading.Thread(target=lambda: (time.sleep(.05), temporary.unlink()))
    cleanup.start()
    result = ad.wait_for_monitored_finish(descriptor, timeout_seconds=.5)
    cleanup.join()
    assert result["status"] == "PASS"
    assert result["final_report_reference"] == ev.file_reference(target)
    assert target.lstat().st_nlink == 1


def test_finish_reader_fails_closed_if_staging_link_never_closes(tmp_path):
    target = tmp_path / "monitor-final.json"
    temporary = tmp_path / ".monitor-final.staged.tmp"
    descriptor, payload = _complete_monitor_payload(tmp_path, target)
    ev.write_exclusive_json(temporary, payload)
    os.link(temporary, target)
    with pytest.raises(ad.MonitoredHardwareError, match="single-link terminal publication"):
        ad.wait_for_monitored_finish(descriptor, timeout_seconds=.05)
    assert target.lstat().st_nlink == 2


def test_finish_report_failure_is_not_admission(tmp_path):
    path = tmp_path / "monitor-final.json"
    ref = {"monitor_final_report": str(path), "run_id": "fixture", "run_binding_sha256": "2" * 64,
           "policy_sha256": "1" * 64, "monitor_pid": 123, "owner": {"pid": 12}}
    path.write_text(json.dumps({**ref, "status": "FAIL", "worker_exited": True, "failure": "clock gap"}))
    with pytest.raises(ad.MonitoredHardwareError, match="clock gap"):
        ad.wait_for_monitored_finish(ref)


def test_current_boot_authority_and_native_receipt_v2_anchors_are_frozen():
    assert "2b53b925d0392e1e84b6e08b38440fde998cf2e0335e06cf3ebf9fd622984709" in ad._ANCHOR_MANIFEST_SHAS
    assert ad._EXTERNAL_CLOCK_RECEIPT_V2_SHA == "be59e1cc6bf143f0351b8f72d8f4ec57681aad9ca500afd4fcbd0e994c9ce3f5"
