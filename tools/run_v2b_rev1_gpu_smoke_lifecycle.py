#!/usr/bin/env python3
"""Process-lifecycle orchestrator for the REV1 bridge-016 smoke.

The parent process never imports torch and never initializes CUDA.  It owns the
training child lifecycle, waits/reaps it, verifies its disappearance, then
starts the owner=None quiescence child and the independent restore child.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def pid_absent(pid: int) -> bool:
    proc = Path("/proc") / str(pid)
    if proc.exists():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def run_child(label: str, command: list[str], *, cwd: Path, env: dict[str, str],
              evidence_root: Path) -> dict[str, Any]:
    evidence_root.mkdir(parents=True, exist_ok=False)
    stdout_path = evidence_root / "stdout.log"
    stderr_path = evidence_root / "stderr.log"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, cwd=str(cwd), env=env, stdout=stdout, stderr=stderr,
            text=True, close_fds=True,
        )
        pid = int(process.pid)
        write_json(evidence_root / "started.json", {
            "label": label, "pid": pid, "started_at": started,
            "argv": command, "cwd": str(cwd),
        })
        returncode = process.wait()
    ended = time.time()
    absent = pid_absent(pid)
    result = {
        "label": label, "pid": pid, "returncode": int(returncode),
        "started_at": started, "ended_at": ended,
        "reaped": True, "pid_absent": absent,
        "stdout": str(stdout_path), "stderr": str(stderr_path),
        "argv": command, "cwd": str(cwd),
    }
    write_json(evidence_root / "exit.json", result)
    if not absent:
        raise RuntimeError(f"{label} PID remains after wait/reap: {pid}")
    if returncode != 0:
        raise RuntimeError(f"{label} exited with rc={returncode}; stderr={stderr_path}")
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def parse_stdout_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"empty child stdout: {path}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"child stdout is not one JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"child stdout object required: {path}")
    return value


def _wait_for_monitored_finish(reference: dict[str, Any]) -> dict[str, Any]:
    """Close a guardian only after its owner child has been reaped.

    The lifecycle controller imports only the stdlib plus the admission
    evidence verifier; it never imports torch or initializes CUDA.
    """
    source_root = Path(__file__).parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from sparse_rtdetr.baseline.training_v2b_admission import wait_for_monitored_finish
    return wait_for_monitored_finish(reference)


def finalize_child_report(path: Path, report: dict[str, Any], label: str,
                          transaction_id: str) -> dict[str, Any]:
    """Publish monitor-final evidence in the parent after child reaping."""
    pending = {
        "PASS_PENDING_MONITOR": "PASS",
        "TRAINING_CHILD_PENDING_MONITOR": "TRAINING_CHILD_PASS",
    }
    status = report.get("status")
    if status not in pending:
        raise RuntimeError(f"{label} did not publish a deferred logical result: {status}")
    reference = report.get("monitor_finish")
    if not isinstance(reference, dict):
        raise RuntimeError(f"{label} missing deferred monitor reference")
    final = _wait_for_monitored_finish(reference)
    finalized = dict(report)
    finalized["monitor_final"] = final
    finalized["status"] = pending[status]
    finalized["monitor_final_pending"] = False
    write_json(path, finalized)
    require_monitor_final(finalized, label, transaction_id)
    return finalized


def require_monitor_final(report: dict[str, Any], label: str, transaction_id: str) -> dict[str, Any]:
    """A logical child PASS is insufficient until its independent monitor closes PASS."""
    final = report.get("monitor_final")
    if not isinstance(final, dict):
        raise RuntimeError(f"{label} missing monitor-final evidence")
    if final.get("status") != "PASS" or final.get("worker_exited") is not True:
        raise RuntimeError(f"{label} monitor-final failed: {final.get('failure')}")
    if final.get("transaction_id") != transaction_id:
        raise RuntimeError(f"{label} monitor transaction identity mismatch")
    if final.get("monitor_phase") not in {"training", "quiescence", "restore"}:
        raise RuntimeError(f"{label} monitor phase missing or invalid")
    return final


def args_for_child(args: argparse.Namespace, root: Path) -> list[str]:
    runner = Path(args.repo) / "tools" / "run_v2b_rev1_gpu_smoke.py"
    command = [
        str(args.python_executable), str(runner),
        "--root", str(root), "--repo", args.repo,
        "--derived", args.derived, "--derived-sha", args.derived_sha,
        "--manifest", args.manifest,
        "--policy-authority", args.policy_authority,
        "--policy-authority-id", args.policy_authority_id,
        "--policy-authority-identity-sha", args.policy_authority_identity_sha,
        "--auth", args.auth, "--auth-id", args.auth_id, "--auth-sha", args.auth_sha,
        "--exec-source-sha", args.exec_source_sha,
        "--exec-contract-sha", args.exec_contract_sha,
        "--execution-contract", args.execution_contract,
        "--bridge-binding-sha", args.bridge_binding_sha,
        "--bridge-manifest-sha", args.bridge_manifest_sha,
        "--training-contract-sha", args.training_contract_sha,
        "--gpu-uuid", args.gpu_uuid,
        "--expected-execution-contract-id", args.expected_execution_contract_id,
        "--invocation", args.structured_invocation,
        "--plan-sha", args.plan_sha,
    ]
    if args.execution_source:
        command += ["--execution-source", args.execution_source]
    if args.revision_verifier:
        command += ["--revision-verifier", args.revision_verifier]
    return command


def child_environment(args: argparse.Namespace, *, cpu: bool) -> dict[str, str]:
    env = dict(os.environ)
    contract = read_json(Path(args.execution_contract))
    startup = contract.get("startup_environment")
    if not isinstance(startup, dict):
        raise RuntimeError("execution contract startup environment is missing")
    static = startup.get("static")
    mode = startup.get("cpu_rehearsal" if cpu else "gpu")
    if not isinstance(static, dict) or not isinstance(mode, dict):
        raise RuntimeError("execution contract startup environment is invalid")
    env.update({str(key): str(value) for key, value in static.items()})
    env.update({str(key): str(value) for key, value in mode.items()})
    expected_cuda = "" if cpu else str(args.gpu_uuid)
    if env.get("CUDA_VISIBLE_DEVICES") != expected_cuda:
        raise RuntimeError("execution contract CUDA binding does not match lifecycle mode")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(Path(args.repo) / "src")
    return env


def write_ready_marker(args: argparse.Namespace, root: Path, *, mode: str) -> None:
    write_json(root / "handshake" / "ready.json", {
        "schema_version": 1,
        "event": "READY",
        "pid": os.getpid(),
        "cwd": str(Path(args.repo)),
        "python_executable": str(args.python_executable),
        "mode": mode,
        "cuda_initialized": False,
        "training_started": False,
        "execution_source_sha256": args.exec_source_sha,
        "execution_contract_sha256": args.exec_contract_sha,
        "authorization_id": args.auth_id,
        "authorization_sha256": args.auth_sha,
        "bridge_checkpoint_sha256": args.derived_sha,
        "next_boundary": {"epoch": 54, "logical_batch_index": 0},
    })


def run_cpu_rehearsal(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise RuntimeError("lifecycle launch root is not a directory")
    training_root = root / "training-child"
    restore_root = root / "restore-child"
    base = args_for_child(args, training_root)
    training = run_child(
        "TRAINING_CHILD_CPU_REHEARSAL",
        base + ["--cpu-rehearsal"],
        cwd=Path(args.repo), env=child_environment(args, cpu=True),
        evidence_root=training_root,
    )
    training_report = read_json(training_root / "cpu-rehearsal-report.json")
    if training_report.get("status") != "READY" or training_report.get("cuda_initialized") is not False:
        raise RuntimeError("CPU training-child rehearsal did not remain CUDA-free")
    write_ready_marker(args, root, mode="cpu_rehearsal")
    restore = run_child(
        "RESTORE_CHILD_CPU_REHEARSAL",
        args_for_child(args, restore_root) + ["--cpu-rehearsal"],
        cwd=Path(args.repo), env=child_environment(args, cpu=True),
        evidence_root=restore_root,
    )
    restore_report = read_json(restore_root / "cpu-rehearsal-report.json")
    if restore_report.get("status") != "READY" or restore_report.get("cuda_initialized") is not False:
        raise RuntimeError("CPU restore-child rehearsal did not remain CUDA-free")
    report = {
        "status": "REV1_RESTORE_LIFECYCLE_CPU_REHEARSAL_PASS",
        "cuda_initialized": False, "training_started": False,
        "training_child": training, "training_report": training_report,
        "restore_child": restore, "restore_report": restore_report,
        "restore_started_after_training_reaped": bool(training["reaped"] and training["pid_absent"]),
        "formal_launch_permitted": False,
    }
    write_json(root / "lifecycle-rehearsal-report.json", report)
    return report


def run_gpu_smoke(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise RuntimeError("lifecycle launch root is not a directory")
    preflight_root = root / "preflight-child"
    training_root = root / "training-child"
    quiescence_root = root / "quiescence-child"
    restore_root = root / "restore-child"
    env = child_environment(args, cpu=False)
    preflight = run_child(
        "PREFLIGHT_CHILD",
        args_for_child(args, preflight_root) + ["--quiescence-probe"],
        cwd=Path(args.repo), env=env, evidence_root=preflight_root,
    )
    preflight_report = read_json(preflight_root / "quiescence-report.json")
    preflight_report = finalize_child_report(
        preflight_root / "quiescence-report.json", preflight_report, "preflight", args.auth_id,
    )
    if preflight_report.get("status") != "PASS" or preflight_report.get("cuda_initialized") is not False:
        raise RuntimeError("preflight child failed before READY")
    write_ready_marker(args, root, mode="gpu_smoke")
    training = run_child(
        "TRAINING_CHILD",
        args_for_child(args, training_root) + ["--training-child"],
        cwd=Path(args.repo), env=env, evidence_root=training_root,
    )
    training_report = read_json(training_root / "training-report.json")
    training_report = finalize_child_report(
        training_root / "training-report.json", training_report, "training", args.auth_id,
    )
    if training_report.get("status") != "TRAINING_CHILD_PASS":
        raise RuntimeError("training child did not publish TRAINING_CHILD_PASS")
    checkpoint = Path(str(training_report["smoke_checkpoint_path"]))
    checkpoint_sha = str(training_report["smoke_checkpoint_sha256"])
    quiescence = run_child(
        "QUIESCENCE_CHILD",
        args_for_child(args, quiescence_root) + ["--quiescence-probe"],
        cwd=Path(args.repo), env=env, evidence_root=quiescence_root,
    )
    quiescence_report = read_json(quiescence_root / "quiescence-report.json")
    quiescence_report = finalize_child_report(
        quiescence_root / "quiescence-report.json", quiescence_report, "quiescence", args.auth_id,
    )
    if quiescence_report.get("status") != "PASS" or quiescence_report.get("cuda_initialized") is not False:
        raise RuntimeError("quiescence child failed owner=None admission")
    restore = run_child(
        "RESTORE_CHILD",
        args_for_child(args, restore_root) + [
            "--roundtrip", "--checkpoint", str(checkpoint), "--checkpoint-sha", checkpoint_sha,
        ],
        cwd=Path(args.repo), env=env, evidence_root=restore_root,
    )
    restore_report = parse_stdout_json(restore_root / "stdout.log")
    restore_report = finalize_child_report(
        restore_root / "restore-report.json", restore_report, "restore", args.auth_id,
    )
    if restore_report.get("status") != "PASS" or restore_report.get("deterministic") is not True:
        raise RuntimeError("restore child did not pass deterministic preview")
    report = {
        "status": "REV1_GPU_PREFORMAL_SMOKE_PASS",
        "preflight_child": preflight, "preflight_report": preflight_report,
        "training_child": training, "training_report": training_report,
        "quiescence_child": quiescence, "quiescence_report": quiescence_report,
        "restore_child": restore, "restore_report": restore_report,
        "smoke_checkpoint_sha256": checkpoint_sha,
        "formal_authorization_consumed": False,
        "formal_launch_permitted": False,
        "training_reaped_before_restore": bool(training["reaped"] and training["pid_absent"]),
    }
    write_json(root / "smoke-report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--derived", required=True)
    parser.add_argument("--derived-sha", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policy-authority", required=True)
    parser.add_argument("--policy-authority-id", required=True)
    parser.add_argument("--policy-authority-identity-sha", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--auth-id", required=True)
    parser.add_argument("--auth-sha", required=True)
    parser.add_argument("--exec-source-sha", required=True)
    parser.add_argument("--exec-contract-sha", required=True)
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--bridge-binding-sha", required=True)
    parser.add_argument("--bridge-manifest-sha", required=True)
    parser.add_argument("--training-contract-sha", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--expected-execution-contract-id", default="v2b-execution-contract-025")
    parser.add_argument("--execution-source")
    parser.add_argument("--revision-verifier")
    parser.add_argument("--structured-invocation", "--invocation", required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--cpu-rehearsal", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.cpu_rehearsal:
            result = run_cpu_rehearsal(args)
        else:
            result = run_gpu_smoke(args)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except BaseException as exc:
        root = Path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "lifecycle-report.json", {
            "status": "REV1_RESTORE_LIFECYCLE_FAIL",
            "error_type": type(exc).__name__, "error": str(exc),
            "formal_authorization_consumed": False,
            "formal_launch_permitted": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
