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


def _invoke(worker, contract_reference, *, cwd, startup_environment):
    environment = os.environ.copy()
    environment.update(startup_environment)
    command = [sys.executable, "-B", str(worker), "--contract", contract_reference["path"],
               "--contract-sha256", contract_reference["sha256"]]
    completed = subprocess.run(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               check=False)
    return {"argv": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


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
            invocation = _invoke(
                worker, reference, cwd=repo_root,
                startup_environment=contract["startup_environment"],
            )
            report["gpu_worker_invocations"].append({
                "cell_id": cell_id, "stage": stage, "contract_reference": reference,
                "returncode": invocation["returncode"],
                "stdout_sha256": hashlib.sha256(invocation["stdout"].encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(invocation["stderr"].encode()).hexdigest(),
            })
            if invocation["returncode"] != 0:
                raise RuntimeError(f"{cell_id} {stage} worker failed without retry: "
                                   + invocation["stdout"][-2000:] + invocation["stderr"][-2000:])
            result_reference = continuation.file_reference(
                Path(contract["output_dir"]) / "worker-result.json")
            result = continuation.read_json_reference(result_reference, f"{cell_id} {stage} result")
            continuation.validate_continuation_result(result, contract, verify_files=True)
            report["results"][f"{cell_id}:{stage}"] = result_reference
            return result_reference

        for seed in (1, 2):
            cell640, cell896 = f"s{seed}_r640", f"s{seed}_r896"
            smoke640 = run(cell640, "smoke")
            smoke896 = run(cell896, "smoke", matched=smoke640)
            replay640 = run(cell640, "smoke_replay", smoke=smoke640)
            replay896 = run(cell896, "smoke_replay", smoke=smoke896, matched=replay640)
            formal640 = run(cell640, "formal60", smoke=smoke640, replay=replay640)
            run(cell896, "formal60", smoke=smoke896, replay=replay896, matched=formal640)
        if len(report["gpu_worker_invocations"]) != 12 or len(report["results"]) != 12:
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
