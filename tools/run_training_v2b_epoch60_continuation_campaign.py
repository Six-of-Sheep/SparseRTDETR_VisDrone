"""Create and execute the frozen four-cell epoch30-to-60 continuation campaign.

This controller is a single fail-closed campaign transaction.  It runs each
worker contract at most once and never resumes or retries a failed stage.
GPU execution still requires a separate owner launch authorization outside
this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def _reference(path, expected_sha256):
    source = Path(path).resolve()
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("authority complete-file SHA-256 mismatch: " + str(source))
    return {"path": str(source), "sha256": digest, "size_bytes": len(raw),
            "sha256_scope": "complete_file_bytes"}


def _json(reference):
    raw = Path(reference["path"]).read_bytes()
    if len(raw) != reference["size_bytes"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("JSON authority changed: " + reference["path"])
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if type(value) is not dict:
        raise ValueError("JSON authority must contain one object")
    return value


def _write_exclusive(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw), "sha256_scope": "complete_file_bytes"}


def _create_campaign_directories(output):
    output.mkdir(mode=0o700)
    contracts_dir = output / "contracts"
    contracts_dir.mkdir(mode=0o700)
    execution_dir = output / "execution"
    execution_dir.mkdir(mode=0o700)
    return contracts_dir


def _worker_environment(startup_environment):
    environment = os.environ.copy()
    environment.update(startup_environment)
    return environment


def _invoke(worker, contract_reference, contract, *, cwd, startup_environment, execution):
    """Own, supervise and close one worker exactly once.

    A worker can only declare that it is about to exit.  The native guardian
    publishes its authoritative final report after that exit.  Consequently
    the controller must bind the child PID, supervise the guardian while the
    child is alive, settle stdout/stderr, wait for the post-exit monitor report,
    and only then publish a stage completion.
    """
    from sparse_rtdetr.baseline.training_v2b_admission import (
        _validate_policy, wait_for_monitored_finish,
    )
    from sparse_rtdetr.baseline.training_v2b_campaign import GuardianSupervisor
    from sparse_rtdetr.baseline import training_v2b_epoch60_continuation as continuation

    environment = _worker_environment(startup_environment)
    command = [sys.executable, "-B", str(worker), "--contract", contract_reference["path"],
               "--contract-sha256", contract_reference["sha256"]]
    label = contract["cell_id"] + "-" + contract["stage"]
    stdout_path = execution / (label + ".stdout.log")
    stderr_path = execution / (label + ".stderr.log")
    scope = "train_core_30epoch" if contract["stage"] == "formal60" else "paired_smoke"
    limit = 43200 if scope == "train_core_30epoch" else 600
    policy = _validate_policy(contract["policy_bundle"], scope, limit)
    started = time.monotonic()
    process = None
    supervisor = None
    launch_reference = None
    result_reference = None
    monitor = None
    first_error = first_traceback = None
    guardian_cleanup = None

    with open(stdout_path, "xb", buffering=0) as stdout, \
            open(stderr_path, "xb", buffering=0) as stderr:
        try:
            process = subprocess.Popen(
                command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True,
            )
            supervisor = GuardianSupervisor(
                process.pid, Path(contract["output_dir"]) / "native-hardware",
                run_id=contract["source_run_id"],
                gpu_uuid=contract["expected_gpu_uuid"],
                policy_sha256=policy["policy_sha256"],
            )
            launch_reference = _write_exclusive(execution / (label + ".launch.json"), {
                "argv": command, "pid": process.pid, "owner": supervisor.owner,
                "contract": contract_reference, "cell_id": contract["cell_id"],
                "stage": contract["stage"], "monotonic_started": started,
            })
            deadline = started + limit + 300
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise RuntimeError(label + " exceeded its fixed wall limit; no retry")
                supervisor.check()
                time.sleep(.05)
            result_path = Path(contract["output_dir"]) / "worker-result.json"
            if process.returncode != 0 or not result_path.is_file():
                raise RuntimeError(
                    f"{label} worker failed (exit {process.returncode}); no automatic retry")
            result_reference = continuation.file_reference(result_path)
            result = continuation.read_json_reference(
                result_reference, label + " worker result", verify=True)
            continuation.validate_continuation_result(result, contract, verify_files=True)
            monitor = wait_for_monitored_finish(result["monitor_reference"])
            if monitor.get("status") != "PASS":
                raise RuntimeError(label + " guardian did not close successfully")
        except BaseException as exc:
            first_error, first_traceback = exc, exc.__traceback__
            if supervisor is not None and process is not None and process.poll() is None:
                try:
                    supervisor.stop_owned_worker()
                except BaseException:
                    process.kill()
            elif process is not None and process.poll() is None:
                process.kill()
            if process is not None:
                try:
                    process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.)
            if supervisor is not None:
                try:
                    guardian_cleanup = supervisor.close_failed_guardian()
                except BaseException as cleanup_error:
                    guardian_cleanup = {"status": "ERROR", "type": type(cleanup_error).__name__,
                                        "message": str(cleanup_error)}
        finally:
            if supervisor is not None:
                supervisor.close()

    stdout_reference = continuation.file_reference(stdout_path)
    stderr_reference = continuation.file_reference(stderr_path)
    exit_reference = _write_exclusive(execution / (label + ".exit.json"), {
        "returncode": None if process is None else process.poll(),
        "pid": None if process is None else process.pid,
        "contract": contract_reference,
        "stdout": stdout_reference, "stderr": stderr_reference,
    })
    if first_error is not None:
        _write_exclusive(execution / (label + ".controller-failure.json"), {
            "status": "STOP_NO_RETRY", "cell_id": contract["cell_id"],
            "stage": contract["stage"], "contract_reference": contract_reference,
            "failure": {"type": type(first_error).__name__, "message": str(first_error)},
            "only_owned_campaign_processes_signalled": True,
            "guardian_cleanup": guardian_cleanup, "automatic_retry": False,
        })
        raise first_error.with_traceback(first_traceback)

    completion = {
        "schema_version": 1, "kind": continuation.COMPLETION_KIND, "status": "PASS",
        "campaign_id": contract["campaign_id"], "execution_id": contract["execution_id"],
        "cell_id": contract["cell_id"], "stage": contract["stage"],
        "contract_reference": contract_reference, "result_reference": result_reference,
        "monitor_final_reference": monitor["final_report_reference"],
        "launch_reference": launch_reference, "exit_reference": exit_reference,
        "elapsed_seconds": time.monotonic() - started, "automatic_retry": False,
    }
    try:
        continuation.validate_continuation_completion(completion, contract, verify_files=True)
    except BaseException as exc:
        _write_exclusive(execution / (label + ".controller-failure.json"), {
            "status": "STOP_NO_RETRY", "cell_id": contract["cell_id"],
            "stage": contract["stage"], "contract_reference": contract_reference,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "only_owned_campaign_processes_signalled": True,
            "guardian_cleanup": {"owner_exited": True, "monitor_final_observed": True},
            "automatic_retry": False,
        })
        raise
    completion_reference = _write_exclusive(
        execution / (label + ".completion.json"), completion)
    return {
        "argv": command, "returncode": 0, "stdout_reference": stdout_reference,
        "stderr_reference": stderr_reference, "result_reference": result_reference,
        "completion_reference": completion_reference,
        "monitor_final_reference": monitor["final_report_reference"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--t8a-report", required=True)
    parser.add_argument("--t8a-report-sha256", required=True)
    parser.add_argument("--policy-640", required=True)
    parser.add_argument("--policy-640-sha256", required=True)
    parser.add_argument("--policy-896", required=True)
    parser.add_argument("--policy-896-sha256", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.dont_write_bytecode = True
    from sparse_rtdetr.baseline import training_v2b_epoch60_continuation as continuation

    output = Path(args.output_root).resolve()
    report = {"schema_version": 1, "kind": "v2b_epoch60_continuation_campaign_result",
              "status": "RUNNING", "campaign_id": args.campaign_id,
              "formal_retry_count": 0, "gpu_worker_invocations": [], "results": {},
              "completions": {},
              "training_core_modified": False, "historical_artifacts_modified": False,
              "started_unix": time.time()}
    try:
        if output.exists() or not output.parent.is_dir():
            raise RuntimeError("campaign output must be a new child of an existing directory")
        t8a_ref = _reference(args.t8a_report, args.t8a_report_sha256)
        policy_refs = {
            640: _reference(args.policy_640, args.policy_640_sha256),
            896: _reference(args.policy_896, args.policy_896_sha256),
        }
        policies = {size: _json(reference) for size, reference in policy_refs.items()}
        freeze = continuation.make_epoch60_continuation_freeze(
            t8a_report_reference=t8a_ref, repo_root=repo_root,
            campaign_id=args.campaign_id, output_root=output)
        contracts_dir = _create_campaign_directories(output)
        freeze_reference = _write_exclusive(contracts_dir / "continuation-freeze.json", freeze)
        report["freeze_reference"] = freeze_reference
        report["policy_references"] = policy_refs
        _write_exclusive(output / "campaign-start.json", report)
        worker = repo_root / "tools/run_training_v2b_epoch60_continuation_worker.py"

        def run(cell_id, stage, *, smoke=None, replay=None, matched=None):
            contract = continuation.continuation_worker_contract(
                freeze, freeze_reference, cell_id=cell_id, stage=stage,
                policy_bundle=policies[int(cell_id[4:])], smoke_reference=smoke,
                replay_reference=replay, matched_reference=matched)
            reference = _write_exclusive(contracts_dir / f"{cell_id}-{stage}.json", contract)
            attempt = {
                "cell_id": cell_id, "stage": stage, "contract_reference": reference,
                "status": "STARTED", "automatic_retry": False,
            }
            report["gpu_worker_invocations"].append(attempt)
            try:
                invocation = _invoke(
                    worker, reference, contract, cwd=repo_root,
                    startup_environment=contract["startup_environment"],
                    execution=output / "execution",
                )
            except BaseException as exc:
                attempt.update(status="STOP_NO_RETRY", returncode=None,
                               failure={"type": type(exc).__name__, "message": str(exc)})
                raise
            attempt.update({
                "status": "PASS", "returncode": invocation["returncode"],
                "stdout_reference": invocation["stdout_reference"],
                "stderr_reference": invocation["stderr_reference"],
                "monitor_final_reference": invocation["monitor_final_reference"],
            })
            if invocation["returncode"] != 0:
                raise RuntimeError(f"{cell_id} {stage} worker failed without retry")
            result_reference = invocation["result_reference"]
            result = continuation.read_json_reference(result_reference, f"{cell_id} {stage} result")
            continuation.validate_continuation_result(result, contract, verify_files=True)
            key = f"{cell_id}:{stage}"
            report["results"][key] = result_reference
            report["completions"][key] = invocation["completion_reference"]
            return result_reference

        for seed in (1, 2):
            cell640, cell896 = f"s{seed}_r640", f"s{seed}_r896"
            smoke640 = run(cell640, "smoke")
            smoke896 = run(cell896, "smoke", matched=smoke640)
            replay640 = run(cell640, "smoke_replay", smoke=smoke640)
            replay896 = run(cell896, "smoke_replay", smoke=smoke896, matched=replay640)
            formal640 = run(cell640, "formal60", smoke=smoke640, replay=replay640)
            run(cell896, "formal60", smoke=smoke896, replay=replay896, matched=formal640)
        if (len(report["gpu_worker_invocations"]) != 12
                or len(report["results"]) != 12 or len(report["completions"]) != 12):
            raise RuntimeError("continuation campaign did not complete the exact twelve-stage plan")
        report["status"] = "PASS"
        report["completed_unix"] = time.time()
        _write_exclusive(output / "campaign-result.json", report)
        print(json.dumps({"status": "PASS", "campaign_id": args.campaign_id,
                          "output_root": str(output), "worker_invocation_count": 12,
                          "retry_count": 0}, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        report["status"] = "STOP_NO_RETRY"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["completed_unix"] = time.time()
        if output.is_dir():
            try:
                _write_exclusive(output / "campaign-result.json", report)
            except FileExistsError:
                pass
        print(json.dumps({"status": "STOP_NO_RETRY", "failure_type": type(exc).__name__,
                          "failure": str(exc), "worker_invocation_count":
                          len(report["gpu_worker_invocations"]), "retry_count": 0},
                         sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
