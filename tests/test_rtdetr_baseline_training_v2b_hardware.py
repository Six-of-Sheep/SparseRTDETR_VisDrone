"""CPU fixtures for passive GPU gate parsers and native-capability boundaries."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import pytest
import torch

from sparse_rtdetr.baseline import training_v2b_hardware as hw
from sparse_rtdetr.baseline import training_v2b_evidence as evidence

UUID = "GPU-11111111-2222-3333-4444-555555555555"
BOOT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def xml(*, process="", power="modern"):
    powers = ("<requested_power_limit>425.00 W</requested_power_limit>"
              "<current_power_limit>425.00 W</current_power_limit>" if power == "modern"
              else "<power_limit>425.00 W</power_limit><enforced_power_limit>425.00 W</enforced_power_limit>")
    return f"""<nvidia_smi_log><driver_version>580.178.04</driver_version>
    <cuda_version>13.0</cuda_version><gpu id="00000000:01:00.0">
    <product_name>NVIDIA GeForce RTX 4090 D</product_name><uuid>{UUID}</uuid>
    <clocks><graphics_clock>1500 MHz</graphics_clock></clocks>
    <max_clocks><graphics_clock>3105 MHz</graphics_clock></max_clocks>
    <applications_clocks><graphics_clock>N/A</graphics_clock></applications_clocks>
    <gpu_power_readings>{powers}</gpu_power_readings>
    <fb_memory_usage><total>24564 MiB</total><used>15 MiB</used><free>24067 MiB</free></fb_memory_usage>
    <temperature><gpu_temp>51 C</gpu_temp></temperature>
    <utilization><gpu_util>0 %</gpu_util></utilization><gpu_recovery_action>None</gpu_recovery_action>
    <clocks_event_reasons>
      <clocks_event_reason_hw_thermal_slowdown>Not Active</clocks_event_reason_hw_thermal_slowdown>
      <clocks_event_reason_hw_power_brake_slowdown>Not Active</clocks_event_reason_hw_power_brake_slowdown>
      <clocks_event_reason_hw_slowdown>Not Active</clocks_event_reason_hw_slowdown>
    </clocks_event_reasons>
    <ecc_errors><volatile><dram_uncorrectable>N/A</dram_uncorrectable></volatile></ecc_errors>
    <processes>{process}</processes></gpu></nvidia_smi_log>"""


def journal(messages, *, start=0, boot=BOOT, cursor_prefix="fixture"):
    rows = [{"_BOOT_ID": boot.replace("-", ""), "_TRANSPORT": "kernel", "MESSAGE": text,
             "__CURSOR": f"{cursor_prefix}:{index}", "__MONOTONIC_TIMESTAMP": str(1_000_000 + index)}
            for index, text in enumerate(messages, start)]
    return "\n".join(json.dumps(row) for row in rows) + f"\n-- cursor: {rows[-1]['__CURSOR']}\n"


def capture(argv, output, *, stderr="", trace_loader=False):
    return {"requested_argv": list(argv), "returncode": 0, "error": None, "truncated": False,
            "loader_trace": trace_loader, "stdout": hw._text_capture(output.encode()),
            "stderr": hw._text_capture(stderr.encode()), "started_ns": time.monotonic_ns(),
            "finished_ns": time.monotonic_ns()}


@pytest.fixture(autouse=True)
def forbid_unmocked_gpu_queries(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("all native GPU subprocesses must be mocked in CPU tests")
    monkeypatch.setattr(hw.subprocess, "Popen", forbidden)
    for name in ("init", "is_available", "is_initialized", "device_count", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def bound(tmp_path):
    source = tmp_path / "bound_source.py"
    source.write_text("FIXTURE = True\n", encoding="utf-8")
    state = evidence.initial_parameter_reference({"fixture": torch.tensor([0.])})
    return evidence.build_run_binding(
        run_id="native-gate-cpu-fixture", code_paths={"fixture": source},
        config={"scope": "cpu_hardware_parser_fixture", "device": "cpu"},
        initial_parameters=state, synthetic_input={"fixture": "CPU zero tensor", "tensor_sha256": state["sha256"]})


@pytest.fixture
def native_fixture(tmp_path, monkeypatch, bound):
    boot = tmp_path / "boot_id"
    boot.write_text(BOOT + "\n", encoding="ascii")
    monkeypatch.setattr(hw, "_BOOT_PATH", boot)
    rapl = tmp_path / "powercap"
    package = rapl / "intel-rapl:0"
    package.mkdir(parents=True)
    for name, value in {
        "name": "package-0", "enabled": "1",
        "constraint_0_name": "long_term", "constraint_0_power_limit_uw": "125000000",
        "constraint_0_time_window_us": "55967744",
        "constraint_1_name": "short_term", "constraint_1_power_limit_uw": "150000000",
        "constraint_1_time_window_us": "2440",
    }.items():
        (package / name).write_text(value + "\n", encoding="ascii")
    monkeypatch.setattr(hw, "_RAPL_ROOT", rapl)
    hwmon = tmp_path / "hwmon"
    sensor = hwmon / "hwmon0"
    sensor.mkdir(parents=True)
    for name, value in {"name": "coretemp", "temp1_input": "50000", "temp1_label": "Package id 0",
                        "temp1_max": "100000", "temp1_crit": "100000"}.items():
        (sensor / name).write_text(value + "\n", encoding="ascii")
    monkeypatch.setattr(hw, "_HWMON_ROOT", hwmon)
    library = tmp_path / "libnvidia-ml.so.580.178.04"
    library.write_bytes(b"CPU fake library bytes, never executable\n")
    source_reader = hw._native_file
    def read(path):
        if path == Path("/sys/module/nvidia/version"):
            return {**hw._text_capture(b"580.178.04\n"), "path": str(path), "error": None}
        if path == Path("/proc/driver/nvidia/version"):
            return {**hw._text_capture(b"NVRM version: 580.178.04\n"), "path": str(path), "error": None}
        return source_reader(path)
    monkeypatch.setattr(hw, "_native_file", read)
    config = {"xml": xml(), "journal": journal(["Linux version 6.8 fixture", "normal kernel event"]),
              "queries": [], "library": library, "package": package, "sensor": sensor}
    def command(argv, *, trace_loader=False):
        config["queries"].append(list(argv))
        if argv == ["nvidia-smi", "-q", "-x"]:
            return capture(argv, config["xml"], stderr=f"   99:\tcalling init: {library}\n",
                           trace_loader=trace_loader)
        if argv == ["nvidia-smi", "--help-query-gpu"]:
            return capture(argv, "clocks.current.graphics\nclocks.max.graphics\n")
        if argv == ["nvidia-smi", "--version"]:
            return capture(argv, "NVIDIA-SMI version : 580.178.04\nNVML version : 580.178\n")
        if argv[:2] == ["nm", "-D"]:
            return capture(argv, "000000 T nvmlDeviceSetGpuLockedClocks\n000001 T nvmlDeviceResetGpuLockedClocks\n")
        if argv[0] == "journalctl":
            return capture(argv, config["journal"])
        raise AssertionError("unexpected native command: " + repr(argv))
    monkeypatch.setattr(hw, "_run_native_command", command)
    config["policy"] = hw.HardwarePolicy(expected_gpu_uuid=UUID)
    config["bound"] = bound
    return config


def collect(fixture, **kwargs):
    return hw.collect_native_hardware_probe(fixture["bound"], policy=fixture["policy"], **kwargs)


def reasons(probe):
    return {row["code"] for row in probe.as_dict()["gate"]["stop_reasons"]}


@pytest.mark.parametrize("power", ["modern", "legacy"])
def test_gpu_parser_preserves_separate_clock_meanings_and_power_formats(power):
    value = hw.parse_gpu_xml(xml(power=power), expected_gpu_uuid=UUID)
    assert value["current_graphics_clock_mhz"] == 1500
    assert value["device_max_graphics_clock_mhz"] == 3105
    assert value["application_graphics_clock_mhz"] is None
    assert value["graphics_clock_cap_proof"]["verified"] is False
    assert value["graphics_clock_cap_proof"]["locked_upper_mhz"] is None
    assert value["power_limit_w"] == value["enforced_power_limit_w"] == 425


@pytest.mark.parametrize("raw", ["", "<bad>", "<other/>", xml().replace(UUID, "GPU-other")])
def test_gpu_parser_rejects_malformed_or_different_device(raw):
    with pytest.raises(hw.HardwareGateError):
        hw.parse_gpu_xml(raw, expected_gpu_uuid=UUID)


def test_gpu_parser_rejects_ambiguous_duplicate_uuid():
    raw = xml()
    gpu = raw[raw.index("<gpu id"):raw.index("</gpu>") + len("</gpu>")]
    with pytest.raises(hw.HardwareGateError, match="ambiguous"):
        hw.parse_gpu_xml(raw.replace("</nvidia_smi_log>", gpu + "</nvidia_smi_log>"), expected_gpu_uuid=UUID)


@pytest.mark.parametrize("text", ["None", "", "\n   "])
def test_empty_gpu_process_list_supported(text):
    assert hw.parse_gpu_xml(xml(process=text), expected_gpu_uuid=UUID)["process_list_readable"] is True


def test_unsupported_gpu_process_list_is_unknown_not_exclusive():
    assert hw.parse_gpu_xml(xml(process="N/A"), expected_gpu_uuid=UUID)["process_list_readable"] is False


@pytest.mark.parametrize("message,code", [
    ("NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus", "GPU_XID"),
    ("NVRM: API mismatch: client version 570 kernel 580", "DRIVER_MISMATCH"),
    ("mce: [Hardware Error]: CPU 0: Machine Check", "HARDWARE_ERROR"),
    ("BERT: [Hardware Error]: Skipped 1 error records", "BERT_UNRESOLVED"),
    ("BERT: Total records found: 1", "BERT_UNRESOLVED"),
    ("pcieport: AER: Corrected error received", "PCIE_ERROR"),
    ("EDAC MC0: 1 CE memory read error", "EDAC_ERROR"),
    ("watchdog: BUG: soft lockup - CPU#0 stuck for 23s", "KERNEL_LOCKUP"),
    ("CPU0: Core temperature above threshold, cpu clock throttled", "THERMAL_ERROR"),
    ("Out of memory: Killed process 5 (python)", "HOST_OOM"),
])
def test_hardware_error_categories_are_explicit(message, code):
    result = hw.parse_kernel_journal(journal(["Linux version 6.8 fixture", message]), boot_id=BOOT)
    assert code in result["findings"][0]["codes"]
    if code == "BERT_UNRESOLVED":
        assert result["findings"][0]["classification"] == "firmware_boot_record_unresolved"


def test_kernel_initialization_messages_are_not_failures():
    raw = journal(["Linux version 6.8 fixture", "ACPI: BERT 0x1234 000030 (v01 INTEL)",
                   "ACPI: Reserving BERT table memory at [mem 0x1-0x2]",
                   "NMI watchdog: Enabled. Permanently consumes one hw-PMU counter.",
                   "pcieport 0000:00:01.0: AER: enabled with IRQ 122",
                   "EDAC MC: Ver: 3.0.0", "CPU0: Thermal monitoring enabled (TM1)",
                   "NVRM: loading NVIDIA UNIX x86_64 Kernel Module 580.178.04"])
    result = hw.parse_kernel_journal(raw, boot_id=BOOT)
    assert result["findings"] == []
    assert result["coverage"] == "full_current_boot"


@pytest.mark.parametrize("raw,kwargs", [
    ("", {}),
    (journal(["Linux version 6.8 fixture"]).replace(BOOT.replace("-", ""), "0" * 32), {}),
    (journal(["normal event only"]), {}),
    (journal(["Linux version 6.8 fixture"]).replace("-- cursor: fixture:0", "-- cursor: wrong"), {}),
    (journal(["normal event"], start=1), {"expected_start_cursor": "fixture:0"}),
    (journal(["Linux version 6.8 fixture"]).replace('"MESSAGE":', '"MESSAGE": "duplicate", "MESSAGE":'), {}),
])
def test_kernel_incomplete_wrongboot_rotated_or_ambiguous_logs_rejected(raw, kwargs):
    with pytest.raises(hw.HardwareGateError):
        hw.parse_kernel_journal(raw, boot_id=BOOT, **kwargs)


def test_kernel_cursor_delta_includes_anchor_but_excludes_it_from_new_findings():
    raw = journal(["NVRM: Xid 79 old anchor", "new normal event"], start=8)
    result = hw.parse_kernel_journal(raw, boot_id=BOOT, expected_start_cursor="fixture:8")
    assert result["coverage"] == "verified_cursor_delta"
    assert result["records_examined"] == 1
    assert result["findings"] == []
    assert result["last_cursor"] == "fixture:9"


@pytest.mark.parametrize("kwargs", [
    {"graphics_clock_upper_mhz": 1501}, {"cpu_pl1_upper_uw": 125000001},
    {"cpu_pl2_upper_uw": 150000001}, {"gpu_stop_temperature_c": 81},
    {"cpu_stop_temperature_c": 91}, {"maximum_probe_age_seconds": 61},
    {"graphics_clock_upper_mhz": True}, {"expected_gpu_uuid": "0"},
    {"allowed_graphics_executable_sha256": ["a" * 64]},
])
def test_policy_cannot_silently_relax_reviewed_limits(kwargs):
    with pytest.raises(hw.HardwareGateError):
        hw.HardwarePolicy(**({"expected_gpu_uuid": UUID} | kwargs))


def test_collector_binds_native_capture_and_never_substitutes_current_clock(native_fixture):
    probe = collect(native_fixture)
    value = probe.as_dict()
    assert reasons(probe) == {"GRAPHICS_CLOCK_CAP_UNVERIFIED"}
    assert value["gate"]["status"] == "BLOCKED"
    assert value["gate"]["gpu_compute_authorized"] is False
    assert value["gate"]["compute_exclusivity_observed"] is True
    assert value["boot_id"] == value["boot_id_after"] == BOOT
    assert value["process"]["pid"] == os.getpid()
    assert value["execution"]["cuda_api_called_by_probe"] is False
    assert value["capability_inventory"]["nvml_library"]["sha256"] == evidence.sha256_bytes(
        native_fixture["library"].read_bytes())
    assert value["capability_inventory"]["has_named_locked_gpu_getter"] is False
    assert value["capability_inventory"]["library_identity_source"] == "gpu_xml_subprocess_native_dynamic_loader_trace"
    assert value["commands"]["gpu_xml"]["loader_trace"] is True
    assert all(argv[0] in {"nvidia-smi", "nm", "journalctl"} for argv in native_fixture["queries"])
    with pytest.raises(hw.HardwareGateError, match="GRAPHICS_CLOCK_CAP_UNVERIFIED"):
        hw.require_native_hardware_admission(probe, binding=native_fixture["bound"], expected_gpu_uuid=UUID)


def test_supplied_or_serialized_json_cannot_grant_admission(native_fixture):
    probe = collect(native_fixture)
    for bad in (probe.as_dict(), json.loads(json.dumps(probe.as_dict())), object()):
        with pytest.raises(hw.HardwareGateError, match="supplied"):
            hw.require_native_hardware_admission(bad, binding=native_fixture["bound"], expected_gpu_uuid=UUID)
    with pytest.raises(hw.HardwareGateError, match="collector"):
        hw.NativeHardwareProbe(object(), probe.as_dict())
    with pytest.raises(hw.HardwareGateError, match="immutable"):
        probe._raw = b"{}"
    with pytest.raises(hw.HardwareGateError, match="serialized"):
        pickle.dumps(probe)


def test_even_internal_relabel_cannot_forge_unsupported_clock_proof(native_fixture):
    original = collect(native_fixture).as_dict()
    original["gate"]["status"] = "PASS"
    original["gate"]["stop_reasons"] = []
    original["gpu"]["graphics_clock_cap_proof"].update(verified=True, locked_upper_mhz=1500)
    # An internal-token adversarial fixture tests the mandatory gate recomputation.
    changed = hw.NativeHardwareProbe(hw._TOKEN, original)
    with pytest.raises(hw.HardwareGateError, match="GRAPHICS_CLOCK_CAP_UNVERIFIED"):
        hw.require_native_hardware_admission(changed, binding=native_fixture["bound"], expected_gpu_uuid=UUID)


@pytest.mark.parametrize("old,new,reason", [
    ("<graphics_clock>1500 MHz", "<graphics_clock>1501 MHz", "GRAPHICS_CLOCK_ABOVE_LIMIT"),
    ("<gpu_temp>51 C", "<gpu_temp>80 C", "GPU_OVER_TEMPERATURE"),
    ("<free>24067 MiB", "<free>19999 MiB", "GPU_MEMORY_BELOW_PREFLIGHT_RESERVE"),
    ("<used>15 MiB", "<used>1025 MiB", "GPU_PREFLIGHT_MEMORY_OCCUPIED"),
    ("<gpu_util>0 %", "<gpu_util>6 %", "GPU_PREFLIGHT_BUSY"),
    ("<gpu_temp>51 C", "<gpu_temp>N/A", "GPU_SENSOR_UNREADABLE"),
    ("<gpu_recovery_action>None", "<gpu_recovery_action>Reset", "GPU_RECOVERY_REQUIRED"),
    ("<clocks_event_reason_hw_slowdown>Not Active", "<clocks_event_reason_hw_slowdown>Active", "GPU_THERMAL_OR_POWER_BRAKE"),
    ("<dram_uncorrectable>N/A", "<dram_uncorrectable>1", "GPU_ECC_ERROR"),
    ("<processes>", "<processes>N/A", "GPU_PROCESSES_UNREADABLE"),
])
def test_hardware_stop_conditions(native_fixture, old, new, reason):
    native_fixture["xml"] = native_fixture["xml"].replace(old, new)
    assert reason in reasons(collect(native_fixture))


@pytest.mark.parametrize("file,value,reason", [
    ("constraint_0_power_limit_uw", "126000000", "RAPL_LIMIT_MISMATCH"),
    ("constraint_1_power_limit_uw", "0", "RAPL_LIMIT_MISMATCH"),
    ("constraint_1_time_window_us", "unknown", "RAPL_WINDOW_UNREADABLE"),
    ("enabled", "0", "RAPL_DISABLED_OR_UNREADABLE"),
])
def test_cpu_power_limits_are_real_readbacks(native_fixture, file, value, reason):
    (native_fixture["package"] / file).write_text(value)
    assert reason in reasons(collect(native_fixture))


def test_cpu_over_temperature_blocks(native_fixture):
    (native_fixture["sensor"] / "temp1_input").write_text("90000\n")
    assert "CPU_OVER_TEMPERATURE" in reasons(collect(native_fixture))


@pytest.mark.parametrize("kind,reason", [("C", "GPU_COMPETING_PROCESS"), ("G", "GPU_GRAPHICS_OCCUPANT_UNAPPROVED"),
                                        ("C+G", "GPU_COMPETING_PROCESS"), ("unknown", "GPU_COMPETING_PROCESS")])
def test_gpu_process_exclusivity_is_not_self_declared(native_fixture, kind, reason):
    row = f"<process_info><pid>{os.getpid()}</pid><type>{kind}</type><process_name>python</process_name><used_memory>4 MiB</used_memory></process_info>"
    native_fixture["xml"] = xml(process=row)
    value = collect(native_fixture)
    assert reason in reasons(value)
    assert value.as_dict()["gate"]["compute_exclusivity_observed"] == (kind == "G")


def test_display_allowance_is_bound_to_actual_executable_hash(native_fixture):
    row = f"<process_info><pid>{os.getpid()}</pid><type>G</type><process_name>display</process_name><used_memory>4 MiB</used_memory></process_info>"
    native_fixture["xml"] = xml(process=row)
    identity = hw._process_identity(os.getpid())
    native_fixture["policy"] = hw.HardwarePolicy(
        expected_gpu_uuid=UUID, allowed_graphics_executable_sha256=(identity["executable"]["sha256"],))
    assert reasons(collect(native_fixture)) == {"GRAPHICS_CLOCK_CAP_UNVERIFIED"}


def test_native_health_findings_are_persisted_and_cannot_be_erased_by_delta(native_fixture):
    native_fixture["journal"] = journal(["Linux version 6.8 fixture", "BERT: [Hardware Error]: Skipped 1 error records"])
    one = collect(native_fixture)
    assert "KERNEL_BERT_UNRESOLVED" in reasons(one)
    native_fixture["journal"] = journal(["BERT: [Hardware Error]: Skipped 1 error records", "ordinary new event"], start=1)
    two = collect(native_fixture, previous=one)
    assert "PRIOR_KERNEL_BERT_UNRESOLVED" in reasons(two)
    assert "PRIOR_PROBE_BLOCKED" in reasons(two)
    assert two.as_dict()["kernel_health"]["findings"] == []
    assert two.as_dict()["previous_probe_sha256"] == one.sha256
    assert any("--cursor=fixture:1" in argv for argv in native_fixture["queries"])


def test_new_cursor_error_is_a_new_event_not_historical(native_fixture):
    one = collect(native_fixture)
    native_fixture["journal"] = journal(["normal kernel event", "NVRM: Xid 79 new"], start=1)
    two = collect(native_fixture, previous=one)
    assert "KERNEL_GPU_XID" in reasons(two)
    assert two.as_dict()["kernel_health"]["records_examined"] == 1


def test_rotated_cursor_blocks_interval(native_fixture):
    one = collect(native_fixture)
    native_fixture["journal"] = journal(["new event"], start=5)
    two = collect(native_fixture, previous=one)
    assert "KERNEL_HEALTH_UNAVAILABLE" in reasons(two)


def test_prior_json_and_policy_changes_rejected(native_fixture):
    one = collect(native_fixture)
    with pytest.raises(hw.HardwareGateError, match="supplied"):
        collect(native_fixture, previous=one.as_dict())
    native_fixture["policy"] = hw.HardwarePolicy(expected_gpu_uuid=UUID, graphics_clock_upper_mhz=1400)
    with pytest.raises(hw.HardwareGateError, match="policy changed"):
        collect(native_fixture, previous=one)


@pytest.mark.parametrize("change", ["boot", "pid", "stale", "source", "library", "gpu"])
def test_admission_checks_live_identity_freshness_and_hashes(native_fixture, monkeypatch, change):
    one = collect(native_fixture)
    kwargs = dict(binding=native_fixture["bound"], expected_gpu_uuid=UUID)
    if change == "boot":
        monkeypatch.setattr(hw, "_boot_id", lambda: "11111111-1111-1111-1111-111111111111")
    elif change == "pid":
        real = hw._process_identity(os.getpid())
        real["start_ticks"] += 1
        monkeypatch.setattr(hw, "_process_identity", lambda pid: real)
    elif change == "stale":
        monkeypatch.setattr(hw.time, "monotonic_ns", lambda: one.as_dict()["monotonic_started_ns"] + 31_000_000_000)
    elif change == "source":
        Path(native_fixture["bound"]["code"]["fixture"]["path"]).write_text("CHANGED = True\n")
    elif change == "library":
        native_fixture["library"].write_bytes(b"changed fixture library")
    elif change == "gpu":
        kwargs["expected_gpu_uuid"] = "GPU-22222222-2222-3333-4444-555555555555"
    with pytest.raises(hw.HardwareGateError) as exc:
        hw.require_native_hardware_admission(one, **kwargs)
    assert "GRAPHICS_CLOCK_CAP_UNVERIFIED" not in str(exc.value)


@pytest.mark.parametrize("mutate", [
    lambda row: row.update(returncode=9),
    lambda row: row.update(error="timeout"),
    lambda row: row.update(truncated=True),
    lambda row: row.update(stderr=hw._text_capture(b"NVML error\n")),
])
def test_failed_or_partial_native_queries_fail_closed(native_fixture, monkeypatch, mutate):
    real = hw._run_native_command
    def command(argv, **kwargs):
        value = real(argv, **kwargs)
        if argv == ["nvidia-smi", "-q", "-x"]:
            mutate(value)
        return value
    monkeypatch.setattr(hw, "_run_native_command", command)
    assert "GPU_OBSERVATION_UNAVAILABLE" in reasons(collect(native_fixture))


def test_probe_failure_cannot_inherit_current_clock_as_cap(native_fixture, monkeypatch):
    real = hw._run_native_command
    def command(argv, **kwargs):
        value = real(argv, **kwargs)
        if argv[0] == "journalctl":
            value["stderr"] = hw._text_capture(b"Insufficient permissions to read system journal\n")
        return value
    monkeypatch.setattr(hw, "_run_native_command", command)
    found = reasons(collect(native_fixture))
    assert {"KERNEL_HEALTH_UNAVAILABLE", "GRAPHICS_CLOCK_CAP_UNVERIFIED"} <= found


def test_no_torch_or_cuda_dependency_is_imported_by_hardware_module():
    import ast
    tree = ast.parse(Path(hw.__file__).read_text())
    names = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name == "torch" or name.startswith(("torch.", "pynvml", "cupy")) for name in names)
