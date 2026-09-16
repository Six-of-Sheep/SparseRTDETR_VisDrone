"""Sequential seed-2 missing-cell completion, with immutable historical control."""
from __future__ import annotations
import copy
import sys
import time
from pathlib import Path
from .training_v2b_campaign import (
    GuardianSupervisor, _canonical, _worker_environment, _write_json, file_reference,
    validate_frozen_source,
)
from .training_v2b_replication_campaign import _require, _json, _same
from .training_v2b_seed2_completion import (
    CELL_ID, STAGES, admit_historical_seed2_control,
    validate_seed2_completion_freeze, seed2_completion_worker_contract,
    validate_seed2_completion_worker_contract,
)
from .training_v2b_seed2_process_logs import PipeLogCapture
from .training_v2b_seed2_completion_worker import (
    GATE_KIND, COMPLETION_KIND, _completed_stage, _successful_cell_source,
    _verify_completion, validate_seed2_completion_stage_gate,
    compare_seed2_completion_smoke,
)

LOG_EOF_TIMEOUT_SECONDS = 10.0


def make_seed2_completion_stage_gate(freeze, freeze_reference, stage, completed, comparisons):
    expected = () if stage == "smoke" else ("smoke",) if stage == "smoke_replay" else ("smoke", "smoke_replay")
    prerequisites = []
    for predecessor in expected:
        entry = completed[predecessor]
        context = _completed_stage(entry["completion_reference"], freeze, freeze_reference, predecessor)
        _same(context["result_reference"], entry["worker_result_reference"], "new prerequisite result")
        prerequisites.append(copy.deepcopy(entry))
    gate = {
        "schema_version": 1, "kind": GATE_KIND, "status": "PASS",
        "campaign_id": freeze["campaign_id"], "freeze_reference": copy.deepcopy(freeze_reference),
        "cell_id": CELL_ID, "stage": stage, "prerequisites": prerequisites,
        "smoke_comparisons": copy.deepcopy(comparisons) if stage == "control30" else {},
        "historical": copy.deepcopy(freeze["historical"]),
        "supplemental_authorization_reference": copy.deepcopy(freeze["supplemental_authorization_reference"]),
    }
    path = Path(freeze["output_root"]) / "stage-gates" / (CELL_ID + "-" + stage + ".json")
    return _write_json(path, gate)


def _verify_stage_completion(freeze, freeze_reference, contract, contract_reference,
                             result_reference, result, completion, candidate_reference, owner):
    verified_result, source_contract = _successful_cell_source(
        result_reference, cell_id=CELL_ID, stage=contract["stage"], contract=contract, freeze=freeze)
    _same(verified_result, result, "actual completed new worker")
    _same(source_contract, contract, "actual launched new contract")
    _same(result["contract_reference"], contract_reference, "launched contract bytes")
    _same(result["freeze_reference"], freeze_reference, "completed new freeze")
    _same(completion["monitor"]["owner"], owner, "actual native supervisor owner")
    verified = _verify_completion(
        candidate_reference, result_reference, verified_result, source_contract, contract, freeze)
    _same(verified, completion, "new completion candidate")
    return verified


def _run_stage(freeze, freeze_reference, contract, contract_reference):
    """Own one worker, retain the first failure, and publish only verified completion."""
    from .training_v2b_admission import _validate_policy, wait_for_monitored_finish
    root, execution = Path(freeze["repo_root"]), Path(freeze["output_root"])/"execution"
    cell, stage = contract["cell_id"], contract["stage"]
    label = cell+"-"+stage
    scope, limit = ("train_core_30epoch", 43200) if stage == "control30" else ("paired_smoke", 600)
    policy = _validate_policy(freeze["common"]["policy_bundle"], scope, limit)
    argv = [sys.executable, "-B", str(root/"tools/run_training_v2b_seed2_completion_worker.py"),
            "--contract", contract_reference["path"], "--contract-sha256", contract_reference["sha256"]]
    start = time.monotonic()
    capture = PipeLogCapture(execution/(label+".stdout.log"), execution/(label+".stderr.log"))
    process = capture.spawn(argv, cwd=root,
        env=_worker_environment({"expected_gpu_uuid": freeze["common"]["expected_gpu_uuid"]}),
        stdin=__import__("subprocess").DEVNULL, start_new_session=True)
    supervisor = None
    completion = None
    result_reference = None
    first_error = first_traceback = None
    cleanup_records, evidence_errors = [], []
    guardian_cleanup = None
    capture_report = log_capture_reference = exit_reference = None

    def describe(error):
        return {"type": type(error).__name__, "message": str(error)}

    def remember(error):
        nonlocal first_error, first_traceback
        if first_error is None:
            first_error, first_traceback = error, error.__traceback__

    def fallback_evidence(payload):
        # This controller stderr is already owned by the outer campaign.
        # A failed disk write must never replace the original runtime error.
        try:
            sys.stderr.write(_canonical(payload).decode("utf-8")+"\n")
            sys.stderr.flush()
        except BaseException:
            pass

    def evidence_error(operation, error):
        remember(error)
        row = {"operation": operation, "error": describe(error)}
        evidence_errors.append(row)
        fallback_evidence({
            "status": "STOP_NO_RETRY", "cell_id": cell, "stage": stage,
            "contract_reference": contract_reference,
            "first_failure": describe(first_error), "evidence_error": row,
        })

    def cleanup_step(operation, action):
        """Every completed cleanup attempt gets its own immutable receipt."""
        row = {"schema_version": 1, "kind": "v2b_seed2_completion_controller_cleanup_step",
               "cell_id": cell, "stage": stage, "contract_reference": contract_reference,
               "operation": operation}
        result = None
        try:
            result = action()
            row.update(status="PASS", result=result)
        except BaseException as error:
            remember(error)
            row.update(status="ERROR", error=describe(error))
        row["first_failure"] = None if first_error is None else describe(first_error)
        record = {"operation": operation, "status": row["status"], "reference": None}
        if "error" in row:
            record["error"] = row["error"]
        cleanup_records.append(record)
        path = execution/(label+".cleanup-"+str(len(cleanup_records)).zfill(2)+"-"+operation+".json")
        try:
            record["reference"] = _write_json(path, row)
        except BaseException as error:
            evidence_error("publish_cleanup_"+operation, error)
        return result

    def settle_logs():
        nonlocal capture_report, log_capture_reference, exit_reference
        if capture_report is None:
            capture_report = capture.finish(
                timeout_seconds=LOG_EOF_TIMEOUT_SECONDS,
                primary_failure=None if first_error is None else describe(first_error))
        if log_capture_reference is None:
            log_capture_reference = _write_json(execution/(label+".log-capture.json"), capture_report)
        if exit_reference is None:
            refs = capture_report.get("references") or {}
            exit_reference = _write_json(execution/(label+".exit.json"), {
                "returncode": process.poll(), "pid": process.pid, "contract": contract_reference,
                "stdout": refs.get("stdout"), "stderr": refs.get("stderr"),
                "log_capture_reference": log_capture_reference,
            })
        return {"capture_status": capture_report["capture_status"],
                "log_capture_reference": log_capture_reference, "exit_reference": exit_reference}

    def publish_failure():
        # Preserve the public failure schema; additional per-step details
        # live in separate cleanup records and the private summary.
        failure = {
            "status": "STOP_NO_RETRY", "cell_id": cell, "stage": stage,
            "contract_reference": contract_reference,
            "only_owned_campaign_processes_signalled": True,
            "guardian_cleanup": guardian_cleanup, "failure": describe(first_error),
        }
        try:
            _write_json(execution/(label+".controller-failure.json"), failure)
        except BaseException as error:
            evidence_error("publish_controller_failure", error)
        summary = {
            "schema_version": 1, "kind": "v2b_seed2_completion_controller_cleanup_summary",
            "status": "STOP_NO_RETRY", "cell_id": cell, "stage": stage,
            "contract_reference": contract_reference, "first_failure": describe(first_error),
            "only_owned_campaign_processes_signalled": True,
            "cleanup_records": cleanup_records, "evidence_errors": evidence_errors,
        }
        try:
            _write_json(execution/(label+".cleanup-summary.json"), summary)
        except BaseException as error:
            evidence_error("publish_cleanup_summary", error)

    try:
        supervisor = GuardianSupervisor(process.pid, Path(contract["output_dir"])/"native-hardware",
            run_id=contract["run_id"], gpu_uuid=freeze["common"]["expected_gpu_uuid"],
            policy_sha256=policy["policy_sha256"])
        launch_reference = _write_json(execution/(label+".launch.json"),
            {"argv": argv, "pid": process.pid, "owner": supervisor.owner,
             "contract": contract_reference, "cell_id": cell, "stage": stage,
             "monotonic_started": start})
        while process.poll() is None:
            _require(time.monotonic() <= start+limit+300, "worker total deadline exceeded")
            _require(not capture.status()["errors"], "owned worker log capture failed")
            supervisor.check()
            time.sleep(.05)
        result_path = Path(contract["output_dir"])/"worker-result.json"
        _require(process.returncode == 0 and result_path.is_file(),
                 label+" worker failed; no automatic retry")
        result_reference = file_reference(result_path)
        result = _json(result_reference)
        _require(result.get("status") == "PASS" and result.get("worker_pid") == process.pid,
                 "worker did not pass with the owned PID")
        for key, value in (("contract_reference", contract_reference),
                           ("freeze_reference", freeze_reference), ("run_id", contract["run_id"]),
                           ("invocation_id", contract["invocation_id"]), ("cell_id", cell), ("stage", stage)):
            _same(result.get(key), value, "worker result "+key)
        monitor = wait_for_monitored_finish(result["monitor_reference"])
        _require(monitor.get("status") == "PASS", "guardian did not close successfully")
        settle_logs()
        _require(capture_report["status"] == "PASS" and capture_report["capture_status"] == "PASS",
                 "owned worker logs did not reach verified EOF")
        validate_frozen_source(freeze["source"])
        completion = {
            "schema_version": 1, "kind": COMPLETION_KIND, "status": "PASS",
            "campaign_id": freeze["campaign_id"], "cell_id": cell, "stage": stage,
            "freeze_reference": freeze_reference, "contract_reference": contract_reference,
            "result_reference": result_reference, "launch_reference": launch_reference,
            "monitor": monitor, "elapsed_seconds": time.monotonic()-start,
            "capacity_memory_audit_reference": None,
            "log_capture_reference": log_capture_reference, "exit_reference": exit_reference,
            "historical_control_reference": copy.deepcopy(freeze["historical"]["control_result_reference"]),
        }
        candidate_reference = _write_json(execution/(label+".completion-candidate.json"), completion)
        _verify_stage_completion(
            freeze, freeze_reference, contract, contract_reference, result_reference, result,
            completion, candidate_reference, supervisor.owner)
    except BaseException as error:
        remember(error)

    if first_error is not None:
        if supervisor is not None:
            cleanup_step("stop_owned_worker", supervisor.stop_owned_worker)
        else:
            # An unreaped direct child cannot have had its PID reused.
            cleanup_step("stop_unreaped_child",
                         lambda: process.kill() if process.poll() is None else None)
        cleanup_step("wait_owned_worker", lambda: process.wait(timeout=5.))
        if supervisor is not None:
            # The existing pidfd helper refuses to signal a guardian until
            # owner exit is proven, including after a failed wait.
            guardian_cleanup = cleanup_step("close_failed_guardian", supervisor.close_failed_guardian)

    if supervisor is not None:
        cleanup_step("close_supervisor_handles", supervisor.close)
    if exit_reference is None:
        cleanup_step("settle_owned_logs", settle_logs)

    if first_error is not None:
        publish_failure()
        raise first_error.with_traceback(first_traceback)

    try:
        completion_reference = _write_json(execution/(label+".completion.json"), completion)
    except BaseException as error:
        remember(error)
        publish_failure()
        raise first_error.with_traceback(first_traceback)
    return {"cell_id": cell, "stage": stage, "worker_result_reference": result_reference,
            "completion_reference": completion_reference}


def run_seed2_completion_campaign(freeze, *, freeze_reference):
    """Run new smoke, new-process replay, then a single fresh-update-zero 896 arm."""
    _same(_json(freeze_reference), freeze, "invoked new freeze")
    validate_seed2_completion_freeze(freeze, verify_files=True)
    historical = admit_historical_seed2_control(freeze)
    _require(historical["control"]["result"]["campaign_id"] != freeze["campaign_id"],
             "historical control was relabelled as a new campaign product")
    root = Path(freeze["output_root"])
    execution = root / "execution"
    execution.mkdir(mode=0o700, exist_ok=False)
    contracts = execution / "contracts"
    contracts.mkdir(mode=0o700)
    (root / "stage-gates").mkdir(mode=0o700)
    completed, comparisons = {}, {}
    report = {
        "schema_version": 1, "kind": "v2b_seed2_historical_control_completion_campaign_result",
        "status": "RUNNING", "campaign_id": freeze["campaign_id"],
        "freeze_reference": freeze_reference, "completed": [], "smoke_comparisons": comparisons,
        "historical_control_reference": freeze["historical"]["control_result_reference"],
        "historical_control_completion_reference": freeze["historical"]["control_completion_reference"],
        "historical_control_is_new_campaign_product": False,
        "old_failed_campaign_reference": freeze["historical"]["failed_campaign_reference"],
        "old_campaign_status": "STOP_NO_RETRY",
        "formal_started": False, "automatic_retry": False,
        "primary_endpoint_epoch": 30, "health_only_epochs": [10, 20],
        "training_extension_authorized": False,
    }
    started = time.monotonic()

    def execute(stage):
        validate_frozen_source(freeze["source"])
        gate_reference = make_seed2_completion_stage_gate(
            freeze, freeze_reference, stage, completed, comparisons)
        replay = completed["smoke"]["worker_result_reference"] if stage == "smoke_replay" else None
        contract = seed2_completion_worker_contract(
            freeze, freeze_reference, stage, stage_gate_reference=gate_reference, replay_source=replay)
        validate_seed2_completion_worker_contract(contract, freeze, verify_files=True)
        validate_seed2_completion_stage_gate(_json(gate_reference), contract, freeze, verify_files=True)
        reference = _write_json(contracts / (CELL_ID + "-" + stage + ".json"), contract)
        entry = _run_stage(freeze, freeze_reference, contract, reference)
        completed[stage] = entry
        report["completed"].append(entry)

    try:
        execute("smoke")
        execute("smoke_replay")
        comparison = compare_seed2_completion_smoke(
            completed["smoke"]["worker_result_reference"],
            completed["smoke_replay"]["worker_result_reference"],
            freeze_reference=freeze_reference)
        _require(comparison.get("status") == "PASS", "new same-cell checkpoint replay failed")
        comparisons[CELL_ID] = _write_json(execution / (CELL_ID + ".smoke-comparison.json"), comparison)
        formal_reference = _write_json(root / "formal-start.json", {
            "status": "HISTORICAL_CAPACITY_AND_CONTROL_NEW_SMOKE_CLOSED",
            "freeze_reference": freeze_reference, "smoke_comparisons": comparisons,
            "completed": list(report["completed"]),
            "historical_control_reference": freeze["historical"]["control_result_reference"],
            "historical_control_is_new_campaign_product": False,
            "formal_order": [CELL_ID], "formal_start": "pretrained_update_zero",
            "code_change_after_this_point_requires_stop": True,
        })
        report.update(formal_started=True, formal_start_reference=formal_reference)
        execute("control30")
        validate_frozen_source(freeze["source"])
        report.update(status="PASS", elapsed_seconds=time.monotonic() - started)
        _write_json(root / "campaign-result.json", report)
        return report
    except BaseException as exc:
        report.update(status="STOP_NO_RETRY", elapsed_seconds=time.monotonic() - started,
                      failure={"type": type(exc).__name__, "message": str(exc)})
        _write_json(root / "campaign-result.json", report)
        raise
