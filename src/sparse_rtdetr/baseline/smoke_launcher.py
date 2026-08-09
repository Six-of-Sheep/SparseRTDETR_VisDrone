"""Authorized Baseline Smoke V1 CLI and process-evidence launcher."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .smoke import (
    SMOKE_AUTH_ENV,
    SMOKE_NONCE_ENV,
    SMOKE_PROCESS_RELATIVE,
    SMOKE_OUTPUT_RELATIVE,
    SmokeContractError,
    contract_check,
    load_smoke_config,
    run_authorized_smoke,
    validate_real_smoke_environment,
)
from .smoke_evidence import (
    SmokeEvidenceError,
    atomic_write,
    canonical_json_bytes,
    file_ref,
    inventory,
    sha256_bytes,
    sha256_file,
    validate_entry_output,
)


class SmokeLauncherError(SmokeContractError):
    """Raised when process evidence ownership or launch identity fails."""


PROCESS_INVENTORY_EXCLUDED = frozenset({"process_inventory.json", "process_completion.json", "process_partial_inventory.json"})
SIGNAL_GRACE_SECONDS = 2.0
_SIGNALS = tuple(value for value in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT) if value is not None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIG{signum}"


def _validate_launch_paths(output_dir: Path, process_evidence_dir: Path, child_python: Path) -> None:
    if not output_dir.is_absolute() or not process_evidence_dir.is_absolute() or not child_python.is_absolute():
        raise SmokeLauncherError("smoke launcher paths must be absolute")
    if any(part == ".." for part in output_dir.parts + process_evidence_dir.parts):
        raise SmokeLauncherError("smoke launcher paths may not contain ..")
    if output_dir.exists() or output_dir.is_symlink():
        raise SmokeLauncherError("child output directory already exists")
    if process_evidence_dir.exists() or process_evidence_dir.is_symlink():
        raise SmokeLauncherError("process evidence directory already exists")
    if child_python.is_symlink() or not child_python.is_file() or not os.access(child_python, os.X_OK):
        raise SmokeLauncherError("child Python must be an executable regular file")


def _validate_frozen_runtime_paths(repo_root: Path, output_dir: Path, process_evidence_dir: Path) -> None:
    expected_output = repo_root.resolve() / SMOKE_OUTPUT_RELATIVE
    expected_process = repo_root.resolve() / SMOKE_PROCESS_RELATIVE
    if output_dir.resolve() != expected_output or process_evidence_dir.resolve() != expected_process:
        raise SmokeLauncherError("smoke output/evidence paths drift from the frozen contract")


def _atomic_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    atomic_write(path, payload)
    return sha256_bytes(payload)


def _send_group_signal(child: subprocess.Popen[bytes], signum: int) -> bool:
    try:
        os.killpg(child.pid, signum)
        return True
    except OSError:
        return False


def _run_child(child_argv: list[str], env: dict[str, str], console) -> tuple[int, list[dict[str, Any]], bool, dict[str, int | None]]:
    state: dict[str, Any] = {"child": None, "deadline": None, "kill_sent": False, "events": []}

    def handle_signal(signum: int, _frame) -> None:
        child = state["child"]
        event: dict[str, Any] = {
            "signal_number": signum,
            "signal_name": _signal_name(signum),
            "received_at_utc": _utc_now(),
            "forwarded_to_child_process_group": False,
        }
        if isinstance(child, subprocess.Popen) and child.poll() is None:
            event["forwarded_to_child_process_group"] = _send_group_signal(child, signum)
            state["deadline"] = time.monotonic() + SIGNAL_GRACE_SECONDS
        state["events"].append(event)

    previous = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    child: subprocess.Popen[bytes] | None = None
    session_id: int | None = None
    process_group_id: int | None = None
    try:
        for signum in _SIGNALS:
            signal.signal(signum, handle_signal)
        child = subprocess.Popen(
            child_argv,
            env=env,
            cwd=env.get("P3_SMOKE_REPO_ROOT") or None,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state["child"] = child
        session_id = os.getsid(child.pid)
        process_group_id = os.getpgid(child.pid)
        for event in state["events"]:
            if not event["forwarded_to_child_process_group"] and child.poll() is None:
                event["forwarded_to_child_process_group"] = _send_group_signal(child, event["signal_number"])
                state["deadline"] = time.monotonic() + SIGNAL_GRACE_SECONDS
        while child.poll() is None:
            deadline = state["deadline"]
            if isinstance(deadline, float) and time.monotonic() >= deadline:
                if _send_group_signal(child, signal.SIGKILL):
                    state["kill_sent"] = True
                state["deadline"] = None
            time.sleep(0.05)
        returncode = child.wait()
    except BaseException:
        if child is not None and child.poll() is None:
            _send_group_signal(child, signal.SIGTERM)
            try:
                child.wait(timeout=SIGNAL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _send_group_signal(child, signal.SIGKILL)
                child.wait()
        raise
    finally:
        for signum, previous_handler in previous.items():
            signal.signal(signum, previous_handler)
        console.flush()
        os.fsync(console.fileno())
    return returncode, list(state["events"]), bool(state["kill_sent"]), {"sid": session_id, "pgid": process_group_id}


def _entry_summary(output_dir: Path) -> dict[str, Any]:
    try:
        return validate_entry_output(output_dir)
    except (OSError, SmokeEvidenceError, ValueError) as exc:
        return {"entry_success_accepted": False, "entry_status": "FAILED_ENTRY_CONTRACT", "entry_validation_error": f"{type(exc).__name__}: {exc}"}


def _finalize_process(
    process_dir: Path,
    output_dir: Path,
    *,
    invocation: dict[str, Any],
    returncode: int | None,
    signal_events: list[dict[str, Any]],
    kill_sent: bool,
    process_started_at: str,
    process_elapsed_seconds: float,
    child_identity: dict[str, int | None],
    internal_error: BaseException | None = None,
) -> int:
    exit_ref: dict[str, Any] = {"present": False, "relative_path": "process_exit_code.txt"}
    if isinstance(returncode, int):
        atomic_write(process_dir / "process_exit_code.txt", f"{returncode}\n".encode("ascii"))
        exit_ref = file_ref(process_dir, "process_exit_code.txt")
    _atomic_json(process_dir / "process_timing.json", {
        "schema_version": 1,
        "started_at_utc": process_started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": process_elapsed_seconds,
    })
    entry = _entry_summary(output_dir) if output_dir.is_dir() else {
        "entry_success_accepted": False,
        "entry_status": "FAILED_ENTRY_MISSING",
        "entry_validation_error": "child output directory is missing",
    }
    success = returncode == 0 and entry.get("entry_success_accepted") is True and internal_error is None
    if not success:
        _atomic_json(process_dir / "process_partial_inventory.json", inventory(
            process_dir,
            frozenset({"process_partial_inventory.json", "process_inventory.json", "process_completion.json"}),
        ))
    process_inventory = inventory(process_dir, PROCESS_INVENTORY_EXCLUDED)
    process_inventory_bytes = canonical_json_bytes(process_inventory)
    atomic_write(process_dir / "process_inventory.json", process_inventory_bytes)
    status = "COMPLETED" if success else "FAILED_ENTRY_CONTRACT" if returncode == 0 else "FAILED_CHILD"
    completion = {
        "schema_version": 1,
        "status": status,
        "child_returncode": returncode,
        "launcher_exit_code": 0 if success else 2,
        "signal_events": signal_events,
        "child_kill_escalated_to_sigkill": kill_sent,
        "child_sid": child_identity.get("sid"),
        "child_pgid": child_identity.get("pgid"),
        "nonce": invocation["nonce"],
        "child_argv": invocation["child_argv"],
        "process_exit_code": exit_ref,
        "process_inventory": {"relative_path": "process_inventory.json", "size_bytes": len(process_inventory_bytes), "sha256": sha256_bytes(process_inventory_bytes)},
        "entry": entry,
        "internal_error": None if internal_error is None else {"type": type(internal_error).__name__, "message": str(internal_error)},
        "non_training": True,
        "non_selection": True,
        "non_reportable": True,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
    }
    _atomic_json(process_dir / "process_completion.json", completion)
    return 0 if success else 2


def launch_smoke(
    child_argv: list[str],
    output_dir: Path,
    process_evidence_dir: Path,
    *,
    child_python: Path,
    repo_root: Path,
    env: dict[str, str] | None = None,
    nonce: str | None = None,
) -> int:
    """Launch a child without creating or reserving its output directory."""

    _validate_launch_paths(output_dir, process_evidence_dir, child_python)
    if os.environ.get(SMOKE_AUTH_ENV) != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SmokeLauncherError("authorized smoke environment is not exact")
    process_evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    process_evidence_dir.mkdir()
    actual_nonce = nonce or secrets.token_hex(16)
    if not actual_nonce or any(character not in "0123456789abcdef" for character in actual_nonce):
        raise SmokeLauncherError("smoke nonce is invalid")
    child_env = dict(os.environ if env is None else env)
    child_env.update({
        "P3_SMOKE_REPO_ROOT": str(repo_root),
        "P3_SMOKE_OUTPUT_DIR": str(output_dir),
        SMOKE_NONCE_ENV: actual_nonce,
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    invocation = {
        "schema_version": 1,
        "mode": "smoke",
        "nonce": actual_nonce,
        "child_argv": child_argv,
        "child_python": str(child_python),
        "repo_root": str(repo_root),
        "output_dir_not_created_by_launcher": True,
        "process_evidence_owner": "launcher",
        "cuda_visible_devices": "0",
        "python_no_user_site": "1",
    }
    _atomic_json(process_evidence_dir / "process_invocation.json", invocation)
    console_path = process_evidence_dir / "process_console.log"
    start = time.monotonic()
    started_at = _utc_now()
    child_identity: dict[str, int | None] = {"sid": None, "pgid": None}
    signal_events: list[dict[str, Any]] = []
    kill_sent = False
    returncode: int | None = None
    internal_error: BaseException | None = None
    try:
        with console_path.open("xb") as console:
            returncode, signal_events, kill_sent, child_identity = _run_child(child_argv, child_env, console)
    except BaseException as error:
        internal_error = error
    return _finalize_process(
        process_evidence_dir,
        output_dir,
        invocation=invocation,
        returncode=returncode,
        signal_events=signal_events,
        kill_sent=kill_sent,
        process_started_at=started_at,
        process_elapsed_seconds=time.monotonic() - start,
        child_identity=child_identity,
        internal_error=internal_error,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtdetr-baseline-smoke")
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("contract-check")
    check.add_argument("--repo-root", type=Path, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--repo-root", type=Path, required=True)
    smoke.add_argument("--data-root", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--process-evidence-dir", type=Path, required=True)
    smoke.add_argument("--child-python", type=Path, default=Path(sys.executable))
    sub.add_parser("_child").add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "contract-check":
            print(canonical_json_bytes(contract_check(args.repo_root)).decode("utf-8"))
            return 0
        if args.mode == "_child":
            run_authorized_smoke(args.repo_root, Path(os.environ["P3_SMOKE_DATA_ROOT"]), Path(os.environ["P3_SMOKE_OUTPUT_DIR"]))
            return 0
        validate_real_smoke_environment()
        repo_root = args.repo_root.resolve()
        load_smoke_config(repo_root)
        if args.data_root is None or any(part.casefold() == "test" for part in args.data_root.parts):
            raise SmokeLauncherError("real smoke data root is invalid")
        _validate_frozen_runtime_paths(repo_root, args.output_dir, args.process_evidence_dir)
        child_argv = [
            str(args.child_python), "-m", "sparse_rtdetr.baseline.smoke_launcher", "_child",
            "--repo-root", str(repo_root),
        ]
        env = dict(os.environ)
        env["P3_SMOKE_DATA_ROOT"] = str(args.data_root)
        env["P3_SMOKE_OUTPUT_DIR"] = str(args.output_dir)
        return launch_smoke(
            child_argv,
            args.output_dir,
            args.process_evidence_dir,
            child_python=args.child_python,
            repo_root=repo_root,
            env=env,
        )
    except (SmokeContractError, SmokeEvidenceError, SmokeLauncherError) as error:
        print(f"rtdetr-baseline-smoke: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"rtdetr-baseline-smoke: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
