"""New seed-2/896 invocation with an explicitly historical seed-2/640 control.

Only evidence admission and orchestration differ from the original campaign.
The model, data, update loop, loss, checkpoint, evaluator and hardware monitor
are imported unchanged. Old result identities are never rewritten.
"""
from __future__ import annotations
import copy
import math
import os
from pathlib import Path

from . import training_v2b_control as control
from . import training_v2b_replication_worker as replication
from .training_v2b import V2BConfig, build_v2b_components
from .training_v2b_data import TrainCoreDataConfig, build_train_core_loader
from .training_v2b_device import prepare_runtime
from .training_v2b_evidence import canonical_sha256, write_exclusive_json
from .training_v2b_runtime import V2BTrainingSession, build_train_core_run_binding
from .training_v2b_campaign import validate_frozen_source
from .training_v2b_seed2_completion import (
    CELL_ID, STAGES, HISTORICAL_CAMPAIGN_ID, Seed2CompletionError, admit_historical_seed2_control,
    validate_seed2_completion_freeze, validate_seed2_completion_worker_contract,
)
from .training_v2b_replication_worker import (
    _read_reference, _reference, _same, _schema, _native_identity,
    _MONITOR_IDENTITY_KEYS, _MONITOR_LEAVES,
    _execute_training_stage, _final_ledger, _integrations, _common_identity,
)

RESULT_KIND = "v2b_seed2_historical_control_completion_worker_result"
COMPLETION_KIND = "v2b_seed2_historical_control_completion_stage_completion"
GATE_KIND = "v2b_seed2_historical_control_completion_stage_gate"
COMPARISON_KIND = "v2b_seed2_historical_control_completion_smoke_comparison"
COMPLETION_KEYS = replication.COMPLETION_KEYS | {
    "log_capture_reference", "exit_reference", "historical_control_reference",
}
GATE_KEYS = replication.GATE_KEYS | {
    "historical", "supplemental_authorization_reference",
}


def _fail(message):
    raise Seed2CompletionError(message)


def _new_report(contract, contract_reference):
    report = replication._new_report(contract, contract_reference)
    report["kind"] = RESULT_KIND
    freeze = _read_reference(contract["freeze_reference"], "new freeze")
    report.update(
        supplemental_authorization_reference=copy.deepcopy(contract["supplemental_authorization_reference"]),
        historical_control_reference=copy.deepcopy(freeze["historical"]["control_result_reference"]),
        historical_matched_campaign_id=HISTORICAL_CAMPAIGN_ID,
        historical_control_is_new_campaign_product=False,
    )
    return report


def _successful_cell_source(reference, *, cell_id, stage, contract, freeze):
    result = _read_reference(reference, "new completed seed2/896 worker")
    if (type(result) is not dict or result.get("kind") != RESULT_KIND
            or result.get("status") != "PASS" or cell_id != CELL_ID
            or result.get("cell_id") != CELL_ID or result.get("stage") != stage
            or result.get("campaign_id") != freeze["campaign_id"]):
        _fail("source is not the required independently identified new completion stage")
    for key in ("freeze_reference", "qualification_reference", "supplemental_authorization_reference"):
        _same(result.get(key), contract[key], "new completed " + key)
    expected = freeze["cells"][CELL_ID]["invocations"][stage]
    for key in ("run_id", "invocation_id"):
        _same(result.get(key), expected[key], "completed " + key)
    if type(result.get("worker_pid")) is not int or result["worker_pid"] <= 0:
        _fail("completed worker PID is absent")
    for key, value in (("seed", 2), ("input_size", 896)):
        if type(result.get(key)) is not int or result[key] != value:
            _fail("completed " + key + " differs")
    _same(result.get("capacity_plan"), freeze["cells"][CELL_ID]["capacity_plan"], "historical capacity plan")
    if (result.get("guardian_clearance_required_after_worker_exit") is not True
            or result.get("scientific_certified") is not False
            or result.get("automatic_retry_or_batch_fallback") is not False
            or result.get("historical_control_is_new_campaign_product") is not False):
        _fail("source scope, historical provenance or guardian declarations differ")
    _same(result.get("historical_control_reference"), freeze["historical"]["control_result_reference"],
          "immutable historical control")
    source_contract = _read_reference(result.get("contract_reference"), "new source contract")
    validate_seed2_completion_worker_contract(source_contract, freeze, verify_files=False)
    _same(source_contract["freeze_reference"], contract["freeze_reference"], "new source freeze")
    if source_contract["stage"] != stage:
        _fail("source stage differs from its own contract")
    if Path(reference["path"]) != Path(expected["output_dir"]) / "worker-result.json":
        _fail("new result is outside its frozen invocation directory")
    final = _read_reference(result["final_state_reference"], "completed final state")
    updates = _final_ledger(source_contract, result, final)
    _same(result["receipt_count"], len(result["receipts"]), "completed receipt count")
    _same(result["logical_samples_committed"], updates * 16, "committed logical sample draws")
    _same(result["logical_samples_executed_this_invocation"], len(result["receipts"]) * 16,
          "executed logical sample draws")
    if stage == "control30":
        if (result.get("formal_started_from_pretrained_update_zero") is not True
                or result.get("formal_checkpoint_resume_used") is not False
                or result.get("primary_endpoint_epoch") != 30
                or result.get("health_only_epochs") != [10, 20]):
            _fail("formal completion changed update-zero start or fixed endpoints")
        if len(result["checkpoints"]) != 30 or len(result["evaluations"]) != 3:
            _fail("formal completion lacks thirty checkpoints and three fixed evaluations")
    return result, source_contract


def _verify_log_closure(completion, result, source_contract, freeze):
    execution = Path(freeze["output_root"]) / "execution"
    label = CELL_ID + "-" + source_contract["stage"]
    expected = {
        "log_capture_reference": execution / (label + ".log-capture.json"),
        "exit_reference": execution / (label + ".exit.json"),
    }
    for key, path in expected.items():
        if Path(_reference(completion[key], key)["path"]) != path:
            _fail("log/exit evidence escaped the owned execution directory")
    capture = _read_reference(completion["log_capture_reference"], "settled pipe log capture")
    if (capture.get("status") != "PASS" or capture.get("capture_status") != "PASS"
            or capture.get("primary_failure") is not None or capture.get("failure") is not None
            or type(capture.get("worker_returncode")) is not int or capture["worker_returncode"] != 0):
        _fail("complete pipe EOF and successful worker exit were not established")
    exit_record = _read_reference(completion["exit_reference"], "owned worker exit")
    if (type(exit_record.get("returncode")) is not int or exit_record["returncode"] != 0
            or exit_record.get("pid") != result["worker_pid"]):
        _fail("worker exit receipt differs")
    _same(exit_record.get("contract"), result["contract_reference"], "exit contract")
    _same(exit_record.get("log_capture_reference"), completion["log_capture_reference"], "exit settled logs")
    for stream in ("stdout", "stderr"):
        ref = _reference(capture.get("references", {}).get(stream), "settled " + stream, verify=True)
        _same(ref, exit_record.get(stream), "exit " + stream)
        if Path(ref["path"]) != execution / (label + "." + stream + ".log"):
            _fail("captured stream is not the owned log")
        state = capture.get("streams", {}).get(stream, {})
        if state.get("eof_seen") is not True or state.get("file_closed") is not True:
            _fail("log reference was published without true EOF and close")
        _same(state.get("bytes_written"), ref["size_bytes"], "captured stream byte count")
    _same(completion["historical_control_reference"], freeze["historical"]["control_result_reference"],
          "completion historical provenance")


def _verify_completion(reference, result_reference, result, source_contract, contract, freeze):
    completion = _read_reference(reference, "replication completion")
    _schema(completion, COMPLETION_KEYS, COMPLETION_KIND, "replication completion")
    if completion["status"] != "PASS":
        _fail("prerequisite completion is not PASS")
    for key in ("campaign_id", "cell_id", "stage", "freeze_reference"):
        _same(completion[key], result[key], "completion " + key)
    _same(completion["result_reference"], result_reference, "completed worker bytes")
    _same(completion["contract_reference"], result["contract_reference"], "completed contract bytes")
    if (type(completion["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(completion["elapsed_seconds"]) or completion["elapsed_seconds"] < 0):
        _fail("completion elapsed time is invalid")
    launch = _read_reference(completion["launch_reference"], "worker launch")
    if (type(launch) is not dict or type(launch.get("pid")) is not int
            or launch["pid"] != result["worker_pid"]):
        _fail("completion launch PID differs from its worker result")
    _same(launch.get("contract"), result["contract_reference"], "launch contract")
    monitor = completion["monitor"]
    if (type(monitor) is not dict or monitor.get("status") != "PASS"
            or monitor.get("worker_exited") is not True):
        _fail("native monitor has not closed after owner exit")
    monitor_reference = result.get("monitor_reference")
    if (type(monitor_reference) is not dict or monitor_reference.get("worker_must_exit") is not True
            or any(key not in monitor_reference or key not in monitor for key in _MONITOR_IDENTITY_KEYS)):
        _fail("source result lacks complete native finish identities")
    for key in _MONITOR_IDENTITY_KEYS:
        _same(monitor[key], monitor_reference[key], "native completion " + key)
    for name in ("run_binding_sha256", "policy_sha256"):
        control._sha(monitor[name], "native " + name)
    if (type(monitor["monitor_pid"]) is not int or monitor["monitor_pid"] <= 0
            or monitor["monitor_pid"] == result["worker_pid"]):
        _fail("native guardian PID is invalid")
    owner = _native_identity(monitor["owner"], "native owner")
    guardian = _native_identity(monitor["guardian_identity"], "native guardian")
    if (monitor["run_id"] != result["run_id"]
            or monitor["gpu_uuid"] != contract["expected_gpu_uuid"]
            or owner["pid"] != result["worker_pid"] or guardian["pid"] != monitor["monitor_pid"]):
        _fail("native monitor owner/run/GPU identity differs")
    _same(launch.get("owner"), owner, "launch/native owner identity")
    binding = _read_reference(result.get("binding_reference"), "completed run binding")
    _same(binding.get("binding_sha256"),
          canonical_sha256({key: value for key, value in binding.items() if key != "binding_sha256"}),
          "completed run binding digest")
    _same(binding["binding_sha256"], monitor["run_binding_sha256"], "native actual run binding")
    _same(binding.get("run_id"), result["run_id"], "native bound run id")
    _same(binding.get("code"), source_contract["code_files"], "native frozen source inventory")
    for name in ("seed", "input_size", "physical_batch_size", "accumulation_steps"):
        _same(binding.get("config", {}).get(name), source_contract["config"][name], "native bound " + name)
    scope, limit = (("train_core_30epoch", 43200) if source_contract["stage"] == "control30"
                    else ("paired_smoke", 600))
    expected_policy = copy.deepcopy(source_contract["policy_bundle"])
    expected_policy.update(authorized_scope=scope, workload_deadline_seconds=limit,
                           minimum_loaded_clock_samples=3)
    _same(monitor["policy_sha256"], canonical_sha256(expected_policy), "native frozen scoped policy")
    native_root = Path(source_contract["output_dir"]) / "native-hardware"
    final_reference = _reference(monitor.get("final_report_reference"), "native final report")
    if (final_reference["path"] != monitor_reference.get("monitor_final_report")
            or Path(final_reference["path"]) != native_root / "monitor-final.json"):
        _fail("native final report path differs from its owned invocation")
    final = _read_reference(final_reference, "native final report")
    _same(final, {key: value for key, value in monitor.items() if key != "final_report_reference"},
          "native closed report bytes")
    if final.get("sampled_clock_compliance") is not True:
        _fail("native final report lacks sampled 1500 MHz compliance")
    for key in _MONITOR_LEAVES:
        ref = _reference(final.get(key), "native " + key, verify=True)
        if Path(ref["path"]).parent != native_root:
            _fail("native leaf escaped its owned evidence directory")
    audit_ref = completion["capacity_memory_audit_reference"]
    if source_contract["stage"] == "capacity":
        audit = _read_reference(audit_ref, "capacity native memory audit")
        if (type(audit) is not dict or audit.get("status") != "PASS"
                or audit.get("kind") != "v2b_seed_resolution_replication_capacity_memory_audit"
                or audit.get("required_margin_mib") != 3072):
            _fail("capacity completion lacks the fixed 3 GiB memory acceptance")
        _same(audit.get("result_reference"), result_reference, "capacity audit result")
        _same(audit.get("monitor_final_report_reference"), final_reference, "capacity audit native report")
        for key in ("minimum_native_free_mib", "minimum_conservative_free_mib"):
            value = audit.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 3072:
                _fail("capacity audit memory margin is below 3 GiB")
    elif audit_ref is not None:
        _fail("only capacity completion may supply a capacity memory audit")
    _verify_log_closure(completion, result, source_contract, freeze)
    return completion


def _completed_stage(reference, freeze, freeze_reference, stage):
    completion = _read_reference(reference, "new stage completion")
    if completion.get("stage") != stage:
        _fail("new completion stage differs")
    contract = _read_reference(completion["contract_reference"], "new stage contract")
    validate_seed2_completion_worker_contract(contract, freeze, verify_files=False)
    _same(contract["freeze_reference"], freeze_reference, "completed stage freeze")
    result_reference = completion["result_reference"]
    result, checked = _successful_cell_source(
        result_reference, cell_id=CELL_ID, stage=stage, contract=contract, freeze=freeze)
    _verify_completion(reference, result_reference, result, checked, contract, freeze)
    return {
        "result": result, "result_reference": result_reference,
        "contract": checked, "contract_reference": completion["contract_reference"],
        "completion": completion, "completion_reference": reference,
    }


def validate_seed2_completion_stage_gate(gate, contract, freeze, *, verify_files=True):
    _schema(gate, GATE_KEYS, GATE_KIND, "new historical-control stage gate")
    if gate["status"] != "PASS":
        _fail("new stage gate is not PASS")
    for key in ("campaign_id", "cell_id", "stage", "freeze_reference",
                "supplemental_authorization_reference"):
        _same(gate[key], contract[key], "new stage gate " + key)
    _same(gate["historical"], freeze["historical"], "unaltered historical evidence bindings")
    stage = contract["stage"]
    expected = () if stage == "smoke" else ("smoke",) if stage == "smoke_replay" else ("smoke", "smoke_replay")
    if type(gate["prerequisites"]) is not list or len(gate["prerequisites"]) != len(expected):
        _fail("new stage prerequisites are missing or duplicated")
    indexed = {}
    for row, predecessor in zip(gate["prerequisites"], expected):
        if type(row) is not dict or set(row) != {
                "cell_id", "stage", "worker_result_reference", "completion_reference"}:
            _fail("new prerequisite reference schema differs")
        if row["cell_id"] != CELL_ID or row["stage"] != predecessor:
            _fail("new stage prerequisite sequence differs")
        for key in ("worker_result_reference", "completion_reference"):
            _reference(row[key], "new prerequisite " + key)
        if verify_files:
            context = _completed_stage(row["completion_reference"], freeze, contract["freeze_reference"], predecessor)
            _same(context["result_reference"], row["worker_result_reference"], "prerequisite result bytes")
        indexed[predecessor] = row
    if stage == "smoke_replay":
        _same(contract["replay_source"], indexed["smoke"]["worker_result_reference"], "only new smoke replay")
    expected_comparisons = {CELL_ID} if stage == "control30" else set()
    if type(gate["smoke_comparisons"]) is not dict or set(gate["smoke_comparisons"]) != expected_comparisons:
        _fail("new formal stage requires exactly its fresh smoke comparison")
    if stage == "control30":
        reference = _reference(gate["smoke_comparisons"][CELL_ID], "new smoke comparison")
        if verify_files:
            comparison = _read_reference(reference, "new smoke comparison")
            if (comparison.get("kind") != COMPARISON_KIND or comparison.get("status") != "PASS"
                    or comparison.get("restore_boundary_exact") is not True
                    or comparison.get("continuation_checkpoint_numerics", {}).get("status") != "PASS"):
                _fail("new smoke restore/continuation failed")
            for key in ("campaign_id", "cell_id", "freeze_reference"):
                _same(comparison.get(key), contract[key], "fresh smoke comparison " + key)
            for field, predecessor in (("original_reference", "smoke"), ("replay_reference", "smoke_replay")):
                _same(comparison.get(field), indexed[predecessor]["worker_result_reference"], "new smoke " + field)
            _same(comparison.get("replay_tolerances"), control.REPLAY_TOLERANCES, "unchanged replay tolerances")
            windows = comparison.get("windows", [])
            if len(windows) != 2 or any(row.get("status") != "PASS" for row in windows):
                _fail("fresh smoke comparison does not accept both continuation windows")
    return copy.deepcopy(gate)


def verify_seed2_completion_worker_authorities(contract, freeze):
    validate_seed2_completion_worker_contract(contract, freeze, verify_files=True)
    expected_path = Path(contract["repo_root"]) / "src/sparse_rtdetr/baseline/training_v2b_seed2_completion_worker.py"
    if Path(__file__).resolve() != expected_path:
        _fail("new worker imported from another checkout")
    validate_frozen_source(freeze["source"])
    historical = admit_historical_seed2_control(freeze)
    validate_seed2_completion_stage_gate(
        _read_reference(contract["stage_gate_reference"], "new stage gate"),
        contract, freeze, verify_files=True)
    return {"freeze": freeze, "historical": historical}


def validate_completed_seed2_896(completion_reference, *, expected_stage="control30", verify_files=True):
    """Authenticate an actual new stage, its native exit, logs and historical control."""
    if expected_stage not in STAGES:
        _fail("completion scope is outside the authorized three stages")
    completion = _read_reference(completion_reference, "new completed seed2/896")
    freeze_reference = completion["freeze_reference"]
    freeze = validate_seed2_completion_freeze(
        _read_reference(freeze_reference, "new completion freeze"), verify_files=verify_files)
    context = _completed_stage(completion_reference, freeze, freeze_reference, expected_stage)
    contract = context["contract"]
    validate_seed2_completion_stage_gate(
        _read_reference(contract["stage_gate_reference"], "completed stage gate"),
        contract, freeze, verify_files=verify_files)
    historical = admit_historical_seed2_control(freeze) if verify_files else None
    return {**context, "freeze": freeze, "freeze_reference": freeze_reference, "historical": historical}


def _invocation_sources(contract, freeze, report, binding, initialization, historical):
    from .training_v2b_resolution import compare_matched_initialization
    historical_stage = "control" if contract["stage"] == "control30" else "smoke"
    source = historical[historical_stage]
    matched = source["result"]
    _same(source["result_reference"], contract["matched_reference"], "authenticated historical matched source")
    report["matched_reference"] = copy.deepcopy(contract["matched_reference"])
    report["matched_initialization"] = compare_matched_initialization(matched, initialization)
    report["historical_matched_campaign_id"] = matched["campaign_id"]
    matched_index = control._receipt_index(matched)
    expected = ({(e, i) for e in range(1, 31) for i in range(304)}
                if contract["stage"] == "control30" else {(1, i) for i in range(4)})
    if set(matched_index) != expected:
        _fail("historical matched source lacks complete prescribed input receipts")
    original = own_smoke = None
    replay_index = {}
    if contract["replay_source"] is not None:
        original, _ = _successful_cell_source(
            contract["replay_source"], cell_id=CELL_ID, stage="smoke", contract=contract, freeze=freeze)
        if original["run_id"] != contract["run_id"] or original["worker_pid"] == os.getpid():
            _fail("new smoke replay requires a different process and unchanged checkpoint run identity")
        _same(original["common_identity"], report["common_identity"], "new same-cell replay common identity")
        _same(original["initialization"], initialization, "new same-cell replay CPU initialization")
        _same(_read_reference(original["binding_reference"], "new smoke binding"),
              binding, "new same-cell checkpoint run binding")
        replay_index = control._receipt_index(original)
        if set(replay_index) != {(1, i) for i in range(4)}:
            _fail("new smoke does not contain exactly four windows")
    if contract["stage"] == "control30":
        gate = _read_reference(contract["stage_gate_reference"], "new formal stage gate")
        row = next(row for row in gate["prerequisites"] if row["stage"] == "smoke")
        own_smoke, _ = _successful_cell_source(
            row["worker_result_reference"], cell_id=CELL_ID, stage="smoke", contract=contract, freeze=freeze)
        _same(own_smoke["common_identity"], report["common_identity"], "new formal/smoke common identity")
        _same(own_smoke["initialization"], initialization, "new formal update-zero CPU initialization")
        report["fresh_initialization_smoke_reference"] = copy.deepcopy(row["worker_result_reference"])
    return matched, original, own_smoke, matched_index, replay_index


def run_seed2_completion_worker(contract, *, contract_reference):
    """Execute one separately authorized invocation; this never launches another worker."""
    freeze = _read_reference(contract["freeze_reference"], "new worker freeze")
    checked = validate_seed2_completion_worker_contract(contract, freeze, verify_files=False)
    _same(_read_reference(contract_reference, "new worker contract"), checked, "invoked contract bytes")
    output = Path(checked["output_dir"])
    output.mkdir(parents=False, exist_ok=False)
    report, monitor = _new_report(checked, contract_reference), None
    try:
        MonitoredHardwareSession, evaluate_development, preview_development = _integrations()
        report["startup_environment"] = control._environment(checked)
        authorities = verify_seed2_completion_worker_authorities(checked, freeze)
        write_exclusive_json(output / "worker-start.json", report)
        config = V2BConfig(**checked["config"])
        data = checked["train_core"]
        loader = build_train_core_loader(
            TrainCoreDataConfig(
                seed=config.seed, input_size=config.input_size, logical_batch_size=16,
                num_workers=checked["num_workers"], prefetch_factor=checked["prefetch_factor"]),
            annotation_file=data["annotation"]["path"], annotation_sha256=data["annotation"]["sha256"],
            manifest_file=data["manifest"]["path"], manifest_sha256=data["manifest"]["sha256"],
            image_root=data["image_root"], repo_root=checked["repo_root"])
        if len(loader.dataset) != 4869 or loader.batches_per_epoch != 304:
            _fail("actual train_core loader differs from the frozen full epoch")
        arguments = {
            "repo_root": checked["repo_root"], "pretrained_path": checked["pretrained"]["path"],
            "pretrained_sha256": checked["pretrained"]["sha256"],
        }
        cpu = build_v2b_components(config, **arguments)
        binding = build_train_core_run_binding(
            cpu, loader, run_id=checked["run_id"], repo_root=checked["repo_root"],
            additional_code_paths={name: ref["path"] for name, ref in checked["code_files"].items()},
            requested_device="cuda:0", cuda_gpu_uuid=checked["expected_gpu_uuid"])
        _same(binding["code"], checked["code_files"], "actual complete imported source inventory")
        report["binding_reference"] = write_exclusive_json(output / "run-binding.json", binding)
        report["initialization"] = cpu.initialization
        report["common_identity"] = _common_identity(checked, binding, loader)
        matched, original, own_smoke, matched_index, replay_index = _invocation_sources(
            checked, authorities["freeze"], report, binding, cpu.initialization, authorities["historical"])
        monitor = MonitoredHardwareSession(
            binding, checked["policy_bundle"], output / "native-hardware",
            "train_core_30epoch" if checked["stage"] == "control30" else "paired_smoke",
            43200 if checked["stage"] == "control30" else 600)
        monitor.start()
        runtime = prepare_runtime(
            device="cuda:0", seed=config.seed, binding=binding,
            gpu_probe=monitor.admission(), expected_gpu_uuid=checked["expected_gpu_uuid"])
        components = build_v2b_components(config, runtime=runtime, **arguments)
        _same(cpu.initialization, components.initialization, "actual CPU/CUDA initialized model")
        del cpu
        session = V2BTrainingSession(components, loader, binding)
        report["runtime"] = runtime.identity
        initial = control.capture_control_state(components, loader, full=True)
        report["initial_state_reference"] = write_exclusive_json(output / "initial-state.json", initial)
        if matched is not None:
            from .training_v2b_resolution import compare_matched_initial_state
            report["matched_cuda_initialization"] = compare_matched_initial_state(
                matched, initial, runtime.identity)
        for source in (original, own_smoke):
            if source is not None:
                _same(source["runtime"], report["runtime"], "same-cell actual CUDA runtime")
                control._verify_common_initial_state(
                    _read_reference(source["initial_state_reference"], "same-cell initial state"), initial)
        _execute_training_stage(
            checked, session, monitor, output, report, original=original,
            matched_index=matched_index, replay_index=replay_index,
            evaluate_development=evaluate_development, preview_development=preview_development)
        final = control.capture_control_state(components, loader, full=True)
        updates = _final_ledger(checked, report, final)
        report["final_state_reference"] = write_exclusive_json(output / "final-state.json", final)
        report["receipt_count"] = len(report["receipts"])
        report["logical_samples_committed"] = updates * 16
        report["logical_samples_executed_this_invocation"] = len(report["receipts"]) * 16
        report["replay_tolerances"] = copy.deepcopy(control.REPLAY_TOLERANCES)
        if checked["stage"] == "control30":
            report.update(primary_endpoint_epoch=30, health_only_epochs=[10, 20],
                          formal_started_from_pretrained_update_zero=True,
                          formal_checkpoint_resume_used=False)
        report["monitor_reference"] = monitor.finish()
        report["status"] = "PASS"
        write_exclusive_json(output / "worker-result.json", report)
        return report
    except BaseException as exc:
        report["status"] = "STOP_NO_RETRY"
        chain, cause, seen = [], exc, set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            row = {"type": type(cause).__name__, "message": str(cause)}
            observation = getattr(cause, "replay_observation_reference", None)
            if observation is not None:
                row["observation_reference"] = copy.deepcopy(observation)
            chain.append(row)
            cause = cause.__cause__
        report["failure"] = chain
        try:
            write_exclusive_json(output / "worker-result.json", report)
        finally:
            if monitor is not None:
                monitor.abort("seed2 completion worker failure: " + type(exc).__name__ + ": " + str(exc))
        raise


def compare_seed2_completion_smoke(original_reference, replay_reference, *, freeze_reference):
    """Authenticate one same-cell replay on CPU; the controller closes each guardian."""
    try:
        base = _read_reference(original_reference, "original replication smoke")
        contract = _read_reference(base["contract_reference"], "original smoke contract")
        _same(contract["freeze_reference"], freeze_reference, "compared freeze")
        freeze = validate_seed2_completion_freeze(_read_reference(freeze_reference, "new completion freeze"), verify_files=False)
        validate_seed2_completion_worker_contract(contract, freeze, verify_files=False)
        base, _ = _successful_cell_source(
            original_reference, cell_id=contract["cell_id"], stage="smoke", contract=contract, freeze=freeze)
        replay, replay_contract = _successful_cell_source(
            replay_reference, cell_id=contract["cell_id"], stage="smoke_replay",
            contract=contract, freeze=freeze)
        _same(replay_contract["replay_source"], original_reference, "restored original smoke source")
        if (base["run_id"] != replay["run_id"] or base["worker_pid"] == replay["worker_pid"]
                or base["invocation_id"] == replay["invocation_id"]):
            _fail("same-cell replay must have a fresh invocation/PID and unchanged checkpoint run identity")
        for name in ("common_identity", "runtime", "initialization"):
            _same(base[name], replay[name], "same-cell smoke " + name)
        for result in (base, replay):
            _same(result["replay_tolerances"], control.REPLAY_TOLERANCES, "frozen replay tolerances")
        control._verify_common_initial_state(
            _read_reference(base["initial_state_reference"], "original initialization"),
            _read_reference(replay["initial_state_reference"], "replay initialization"))
        midpoint = base["checkpoints"]["window_2"]
        control._reference(midpoint["checkpoint"], "original midpoint checkpoint", verify=True,
                           expected_name="checkpoint-window-02.pt")
        _same(replay["restored_from"], midpoint, "restored checkpoint reference")
        _same(_read_reference(midpoint["state_reference"], "original checkpoint boundary"),
              _read_reference(replay["restore_boundary_reference"], "restored checkpoint boundary"),
              "restored complete checkpoint boundary")
        base_index, replay_index = control._receipt_index(base), control._receipt_index(replay)
        if set(base_index) != {(1, i) for i in range(4)} or set(replay_index) != {(1, 2), (1, 3)}:
            _fail("smoke/replay did not execute exactly four/two declared windows")
        windows = []
        for key in sorted(replay_index):
            comparison = control.compare_replay_records(
                control._read_json_reference(base_index[key], "original logical window"),
                control._read_json_reference(replay_index[key], "replayed logical window"))
            windows.append({**comparison, "epoch": key[0], "logical_batch_index": key[1]})
        base_final = _read_reference(base["final_state_reference"], "original final state")
        replay_final = _read_reference(replay["final_state_reference"], "replay final state")
        _final_ledger(contract, base, base_final)
        _final_ledger(replay_contract, replay, replay_final)
        for name in ("clocks", "rng", "raw_bn", "ema_bn"):
            _same(base_final[name], replay_final[name], "same-cell final " + name)
        binding = _read_reference(base["binding_reference"], "original smoke binding")
        _same(binding, _read_reference(replay["binding_reference"], "replay binding"),
              "smoke continuation checkpoint binding")
        continuation = control.compare_continuation_checkpoints(
            base["checkpoints"]["window_4"]["checkpoint"],
            replay["checkpoints"]["window_4"]["checkpoint"], binding=binding)
        preview = control._compare_previews(control._preview_evidence(base), control._preview_evidence(replay))
        return {
            "schema_version": 1, "kind": COMPARISON_KIND, "status": "PASS",
            "campaign_id": contract["campaign_id"], "cell_id": contract["cell_id"],
            "freeze_reference": copy.deepcopy(freeze_reference),
            "original_reference": copy.deepcopy(original_reference),
            "replay_reference": copy.deepcopy(replay_reference),
            "restore_boundary_exact": True, "windows": windows,
            "continuation_checkpoint_numerics": continuation,
            "replay_tolerances": copy.deepcopy(control.REPLAY_TOLERANCES),
            "development_preview": preview,
            "ema_final_state_exact": base_final["ema_state_sha256"] == replay_final["ema_state_sha256"],
            "optimizer_final_state_exact": base_final["optimizer_state_sha256"] == replay_final["optimizer_state_sha256"],
            "development_numerical_equalities_are_observations_not_acceptance_thresholds": True,
            "hardware_clearance": "controller_authenticates_native_final_after_each_owner_exit",
            "scientific_certified": False, "primary_endpoint_epoch": 30, "health_only_epochs": [10, 20],
        }
    except Exception as exc:
        return {
            "schema_version": 1, "kind": COMPARISON_KIND, "status": "STOP_NO_RETRY",
            "freeze_reference": copy.deepcopy(freeze_reference),
            "original_reference": copy.deepcopy(original_reference),
            "replay_reference": copy.deepcopy(replay_reference),
            "failure_type": type(exc).__name__, "failure": str(exc), "scientific_certified": False,
        }
