"""Execution-only tmux controller with durable STARTED/READY handshake.

This module deliberately has no torch, CUDA, dataset, or model imports.  It is
the narrow orchestration boundary used by the REV1 pre-formal smoke.  Runtime
evidence is intentionally separate from the scientific source identity.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
MODULE_NAME = "sparse_rtdetr.baseline.training_v2b_smoke_controller"
LAYOUT_VERSION = 1


class LaunchDirectoryError(RuntimeError):
    """The controller/runner launch-directory ownership contract was violated."""


def _layout(root: Path) -> dict[str, Path]:
    return {
        "controller": root / "controller",
        "handshake": root / "handshake",
        "runner": root / "runner",
    }


def _create_runner_workspace(root: Path) -> Path:
    """Create the runner-owned child exactly once; never create the launch root."""
    if not root.is_dir():
        raise LaunchDirectoryError("LAUNCH_ROOT_MISSING")
    runner = _layout(root)["runner"]
    try:
        runner.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise LaunchDirectoryError("RUNNER_WORKSPACE_ALREADY_EXISTS") from exc
    return runner


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    tmp.write_bytes(canonical_bytes(dict(value)))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _marker(root: Path, name: str) -> Path:
    return root / name


def _environment_allowlist(names: Sequence[str]) -> dict[str, str]:
    return {name: os.environ[name] for name in names if name in os.environ}


def build_plan(
    *,
    launch_id: str,
    evidence_root: Path,
    repo_root: Path,
    python_executable: Path,
    runner_argv: Sequence[str],
    tmux_executable: Path,
    session_name: str,
    environment: Mapping[str, str] | None = None,
    ready_timeout: float = 10.0,
    exit_timeout: float = 10.0,
) -> dict[str, Any]:
    """Build a fully explicit launch plan; no shell command is assembled."""
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_schema_version": LAYOUT_VERSION,
        "launch_id": launch_id,
        "evidence_root": str(evidence_root),
        "layout": {name: str(path) for name, path in _layout(evidence_root).items()},
        "cwd": str(repo_root),
        "python_executable": str(python_executable),
        "runner_argv": list(runner_argv),
        "argv": [str(python_executable), *list(runner_argv)],
        "environment": dict(sorted((environment or {}).items())),
        "tmux_executable": str(tmux_executable),
        "session_name": session_name,
        "tmux_argv": [str(tmux_executable), "new-session", "-d", "-s", session_name, "-c", str(repo_root)],
        "ready_timeout_seconds": float(ready_timeout),
        "exit_timeout_seconds": float(exit_timeout),
    }


def _quote_env(env: Mapping[str, str]) -> str:
    parts: list[str] = []
    for key, value in sorted(env.items()):
        parts.append(shlex.quote(key + "=" + value))
    return "env " + " ".join(parts) if parts else ""


def _wrapper_text(plan: Mapping[str, Any], wrapper: Path, stdout_path: Path, stderr_path: Path) -> str:
    """Return a POSIX wrapper which records child launch and exit separately."""
    root = Path(str(plan["evidence_root"]))
    layout = _layout(root)
    started = {
        "schema_version": SCHEMA_VERSION,
        "event": "STARTED",
        "launch_id": plan["launch_id"],
        "cwd": plan["cwd"],
        "python_executable": plan["python_executable"],
        "argv": plan["argv"],
        "session_name": plan["session_name"],
        "layout": {name: str(path) for name, path in layout.items()},
    }
    started_literal = shlex.quote(canonical_bytes(started).decode("utf-8"))
    runtime_environment = {
        **dict(plan["environment"]),
        "REV1_LAUNCH_ROOT": str(root),
        "REV1_CONTROLLER_DIR": str(layout["controller"]),
        "REV1_HANDSHAKE_DIR": str(layout["handshake"]),
        "REV1_RUNNER_DIR": str(layout["runner"]),
    }
    env_prefix = _quote_env(runtime_environment)
    command = " ".join(shlex.quote(str(x)) for x in plan["argv"])
    if env_prefix:
        command = env_prefix + " " + command
    stdout_q = shlex.quote(str(stdout_path))
    stderr_q = shlex.quote(str(stderr_path))
    runner_started = layout["handshake"] / "runner-started.json"
    exit_path = layout["handshake"] / "exit.json"
    started_path = layout["handshake"] / "started.json"
    return "\n".join(
        [
            "#!/bin/sh",
            "set +e",
            "started_path=" + shlex.quote(str(started_path)),
            "started_tmp=\"${started_path}.tmp-$$\"",
            "printf '%s\\n' " + started_literal + " > \"$started_tmp\" && mv \"$started_tmp\" \"$started_path\"",
            command + " >" + stdout_q + " 2>" + stderr_q + " &",
            "child_pid=$!",
            "printf '%s\\n' '{\"schema_version\":1,\"event\":\"RUNNER_STARTED\",\"launch_id\":\"" + str(plan["launch_id"]) + "\",\"child_pid\":'\"$child_pid\"'}' > " + shlex.quote(str(runner_started)) + ".tmp-$$",
            "mv " + shlex.quote(str(runner_started)) + ".tmp-$$ " + shlex.quote(str(runner_started)),
            "wait \"$child_pid\"",
            "rc=$?",
            "printf '%s\\n' '{\"schema_version\":1,\"event\":\"EXIT\",\"launch_id\":\"" + str(plan["launch_id"]) + "\",\"child_pid\":'\"$child_pid\"',\"exit_code\":'\"$rc\"'}' > " + shlex.quote(str(exit_path)) + ".tmp-$$",
            "mv " + shlex.quote(str(exit_path)) + ".tmp-$$ " + shlex.quote(str(exit_path)),
            "exit \"$rc\"",
            "",
        ]
    )


def _write_evidence(root: Path, value: Mapping[str, Any]) -> None:
    _atomic_json(_layout(root)["controller"] / "launch-evidence.json", value)


def _status_from_files(root: Path, ready: dict[str, Any] | None, exit_data: dict[str, Any] | None) -> str:
    if ready is None and exit_data is not None:
        return "RUNNER_EXITED_BEFORE_READY"
    if ready is not None and exit_data is not None:
        return "COMPLETED" if int(exit_data.get("exit_code", 1)) == 0 else "RUNNER_NONZERO_EXIT"
    return "READY_NOT_OBSERVED" if ready is None else "EXIT_TIMEOUT_AFTER_READY"


def launch(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Launch one wrapper and observe durable markers, never tmux alone."""
    root = Path(str(plan["evidence_root"]))
    if root.exists():
        raise FileExistsError(f"evidence root already exists: {root}")
    root.mkdir(parents=True)
    layout = _layout(root)
    layout["controller"].mkdir(parents=False, exist_ok=False)
    layout["handshake"].mkdir(parents=False, exist_ok=False)
    # The runner owns layout["runner"] and must create it itself exactly once.
    stdout_path = layout["controller"] / "stdout.log"
    stderr_path = layout["controller"] / "stderr.log"
    wrapper = layout["controller"] / "runner-wrapper.sh"
    request = {
        "schema_version": SCHEMA_VERSION,
        "launch_id": plan["launch_id"],
        "phase": "REQUESTED",
        "plan": dict(plan),
        "wrapper": str(wrapper),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "layout": {name: str(path) for name, path in layout.items()},
    }
    _write_evidence(root, request)
    cwd = Path(str(plan["cwd"]))
    python_executable = Path(str(plan["python_executable"]))
    if not cwd.is_dir():
        request.update({"status": "INVALID_CWD", "error": f"cwd is not a directory: {cwd}", "phase": "FINAL"})
        _write_evidence(root, request)
        return request
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        request.update({"status": "INVALID_PYTHON_EXECUTABLE", "error": f"python executable is not executable: {python_executable}", "phase": "FINAL"})
        _write_evidence(root, request)
        return request
    tmux_executable = str(plan["tmux_executable"])
    if os.path.sep not in tmux_executable and shutil.which(tmux_executable) is None:
        request.update({"status": "TMUX_CREATE_FAILED", "error": f"tmux executable not found: {tmux_executable}", "phase": "FINAL"})
        _write_evidence(root, request)
        return request
    wrapper.write_text(_wrapper_text(plan, wrapper, stdout_path, stderr_path), encoding="utf-8", newline="\n")
    wrapper.chmod(0o700)
    tmux_argv = [str(x) for x in plan["tmux_argv"]] + [str(wrapper)]
    request["tmux_command"] = tmux_argv
    _write_evidence(root, request)
    try:
        created = subprocess.run(tmux_argv, cwd=str(plan["cwd"]), capture_output=True, text=True, check=False)
    except OSError as exc:
        request.update({"status": "TMUX_CREATE_FAILED", "tmux_error": repr(exc), "phase": "FINAL"})
        _write_evidence(root, request)
        return request
    request["tmux_returncode"] = int(created.returncode)
    request["tmux_stdout"] = created.stdout
    request["tmux_stderr"] = created.stderr
    if created.returncode != 0:
        request.update({"status": "TMUX_CREATE_FAILED", "phase": "FINAL"})
        _write_evidence(root, request)
        return request
    request["tmux_create_observed"] = True
    _write_evidence(root, request)

    ready_deadline = time.monotonic() + float(plan["ready_timeout_seconds"])
    ready = _read_json(_marker(layout["handshake"], "ready.json"))
    exit_data = _read_json(_marker(layout["handshake"], "exit.json"))
    while ready is None and exit_data is None and time.monotonic() < ready_deadline:
        time.sleep(0.05)
        ready = _read_json(_marker(layout["handshake"], "ready.json"))
        exit_data = _read_json(_marker(layout["handshake"], "exit.json"))
    request["started_observed"] = _read_json(_marker(layout["handshake"], "started.json")) is not None
    request["runner_started_observed"] = _read_json(_marker(layout["handshake"], "runner-started.json")) is not None
    if ready is None:
        request.update({
            "status": _status_from_files(root, ready, exit_data),
            "ready_observed": False,
            "exit": exit_data,
            "stdout": stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else "",
            "stderr": stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else "",
            "phase": "FINAL",
        })
        _write_evidence(root, request)
        return request
    request["ready_observed"] = ready
    exit_deadline = time.monotonic() + float(plan["exit_timeout_seconds"])
    while exit_data is None and time.monotonic() < exit_deadline:
        time.sleep(0.05)
        exit_data = _read_json(_marker(layout["handshake"], "exit.json"))
    request.update({
        "status": _status_from_files(root, ready, exit_data),
        "exit": exit_data,
        "started_observed": _read_json(_marker(layout["handshake"], "started.json")) is not None,
        "runner_started_observed": _read_json(_marker(layout["handshake"], "runner-started.json")) is not None,
        "phase": "FINAL",
    })
    request["stdout"] = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    request["stderr"] = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    _write_evidence(root, request)
    return request


def _noop_runner(args: argparse.Namespace) -> int:
    root = Path(args.evidence_root)
    runner = _create_runner_workspace(root)
    handshake = _layout(root)["handshake"]
    time.sleep(float(args.ready_delay))
    _atomic_json(runner / "runner-evidence.json", {
        "schema_version": SCHEMA_VERSION, "event": "RUNNER_WORKSPACE_CREATED",
        "launch_id": args.launch_id, "root": str(root), "runner": str(runner),
        "cwd": os.getcwd(), "python_executable": sys.executable,
    })
    _atomic_json(
        handshake / "ready.json",
        {
            "schema_version": SCHEMA_VERSION,
            "event": "READY",
            "launch_id": args.launch_id,
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
            "argv": sys.argv,
            "cuda_initialized": False,
        },
    )
    time.sleep(float(args.duration))
    return 0


def _probe(args: argparse.Namespace) -> int:
    launch_id = args.launch_id or ("cpu-probe-" + uuid.uuid4().hex[:12])
    env = _environment_allowlist(args.environment_name)
    runner_argv = ["-m", MODULE_NAME, "--noop-runner", "--evidence-root", str(args.evidence_root), "--launch-id", launch_id, "--ready-delay", str(args.ready_delay), "--duration", str(args.duration)]
    plan = build_plan(
        launch_id=launch_id,
        evidence_root=Path(args.evidence_root),
        repo_root=Path(args.repo_root),
        python_executable=Path(args.python_executable),
        runner_argv=runner_argv,
        tmux_executable=Path(args.tmux_executable),
        session_name=args.session_name,
        environment=env,
        ready_timeout=args.ready_timeout,
        exit_timeout=args.exit_timeout,
    )
    result = launch(plan)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("status") == "COMPLETED" else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--noop-runner", action="store_true")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--tmux-executable", default="/usr/bin/tmux")
    parser.add_argument("--session-name", default="rev1-cpu-probe")
    parser.add_argument("--launch-id", default=None)
    parser.add_argument("--ready-delay", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=0.15)
    parser.add_argument("--ready-timeout", type=float, default=5.0)
    parser.add_argument("--exit-timeout", type=float, default=5.0)
    parser.add_argument("--environment-name", action="append", default=["PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "MKL_THREADING_LAYER", "OMP_NUM_THREADS", "MKL_NUM_THREADS"])
    args = parser.parse_args(argv)
    if args.noop_runner:
        return _noop_runner(args)
    if args.probe:
        return _probe(args)
    parser.error("select --probe or --noop-runner")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
