"""Durable, single-call outer launcher for Baseline Smoke V2."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .smoke import (
    SMOKE_V2_CONFIG_RELATIVE,
    SMOKE_V2_ID,
    load_smoke_config,
)
from .smoke_evidence import atomic_write, argv_sha256, canonical_json_bytes, sha256_bytes, sha256_file


class OuterLaunchError(Exception):
    """Raised when the outer evidence contract cannot be established."""


OUTER_COMPLETION = "outer_completion.json"
OUTER_INVENTORY = "outer_inventory.json"
OUTER_PARTIAL_INVENTORY = "outer_partial_inventory.json"
OUTER_EXCLUDED = frozenset({OUTER_COMPLETION, OUTER_INVENTORY, OUTER_PARTIAL_INVENTORY})
_ASCII_RE = re.compile(r"^[\x00-\x7f]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    atomic_write(path, payload)
    return sha256_bytes(payload)


def _regular_file(path: Path, executable: bool = False) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise OuterLaunchError(f"not a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise OuterLaunchError(f"not executable: {path}")


def _ascii(value: str, field: str) -> None:
    if type(value) is not str or _ASCII_RE.fullmatch(value) is None:
        raise OuterLaunchError(f"{field} must be ASCII")


def _bind_config(repo_root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _regular_file(config_path)
    config = load_smoke_config(repo_root, config_path)
    resolved = config_path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise OuterLaunchError("config is outside repo root") from exc
    if config.get("smoke_id") != SMOKE_V2_ID or relative != SMOKE_V2_CONFIG_RELATIVE:
        raise OuterLaunchError("outer launcher requires explicit R2 config")
    return config, {
        "smoke_id": SMOKE_V2_ID,
        "config_relative_path": relative,
        "config_path": str(resolved),
        "config_size_bytes": resolved.stat().st_size,
        "config_file_sha256": sha256_file(resolved),
        "config_canonical_sha256": sha256_bytes(canonical_json_bytes(config)),
    }


def _validate_outer_parent(outer_root: Path) -> None:
    parent = outer_root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise OuterLaunchError("outer evidence parent is not a regular directory")
    if stat.S_IMODE(parent.stat().st_mode) != 0o775 or parent.stat().st_uid != os.getuid():
        raise OuterLaunchError("outer evidence parent owner/mode mismatch")
    if outer_root.exists() or outer_root.is_symlink():
        raise OuterLaunchError("outer evidence root already exists")


def _validate_runtime_paths(repo_root: Path, config: dict[str, Any], output: Path, process: Path, outer: Path) -> None:
    runtime = config["runtime"]
    if output.resolve() != repo_root / runtime["output_relative_path"]:
        raise OuterLaunchError("R2 output path drift")
    if process.resolve() != repo_root / runtime["process_evidence_relative_path"]:
        raise OuterLaunchError("R2 process evidence path drift")
    if outer.resolve() != repo_root / runtime["outer_launch_evidence_relative_path"]:
        raise OuterLaunchError("R2 outer evidence path drift")
    if output.exists() or output.is_symlink() or process.exists() or process.is_symlink():
        raise OuterLaunchError("R2 child evidence path already exists")


def _validate_data_root(data_root: Path) -> Path:
    if not data_root.is_absolute() or data_root.is_symlink():
        raise OuterLaunchError("data root must be absolute and non-symlink")
    canonical = data_root.resolve(strict=True)
    if not canonical.is_dir() or canonical.is_symlink() or any(part.casefold() == "test" for part in canonical.parts):
        raise OuterLaunchError("data root is invalid")
    return canonical


def _inventory(root: Path, excluded: frozenset[str]) -> dict[str, Any]:
    rows = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name in excluded:
            continue
        if path.name.startswith(".") or path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise OuterLaunchError(f"invalid evidence file: {path.name}")
        rows.append({"relative_path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema_version": 1,
        "excluded_from_inventory": sorted(excluded),
        "artifacts": rows,
        "canonical_inventory_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def _file_ref(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        return {"present": False, "relative_path": name}
    return {"present": True, "relative_path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _inner_argv(repo: Path, data: Path, output: Path, process: Path, config: Path, child: Path) -> list[str]:
    return [
        str(child), "-u", "-m", "sparse_rtdetr.baseline.smoke_launcher", "smoke",
        "--repo-root", str(repo), "--data-root", str(data), "--output-dir", str(output),
        "--process-evidence-dir", str(process), "--config", str(config), "--child-python", str(child),
    ]


def _wrapper(
    pane: Path,
    plan: Path,
    plan_sha: str,
    config: Path,
    config_sha: str,
    inner: list[str],
    environment: dict[str, str],
) -> str:
    env_lines = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(environment.items()))
    inner_cmd = shlex.join(inner)
    return f"""#!/bin/sh
set -u
PANE={shlex.quote(str(pane))}
PLAN={shlex.quote(str(plan))}
CONFIG={shlex.quote(str(config))}
PLAN_SHA={shlex.quote(plan_sha)}
CONFIG_SHA={shlex.quote(config_sha)}
CONSOLE="$PANE/pane_console.log"
START="$PANE/pane_start.json"
RECEIPT="$PANE/pane_receipt.json"
EXIT="$PANE/pane_exit_code.txt"
TIMING="$PANE/pane_timing.json"
ERROR="$PANE/pane_error.json"
INVENTORY="$PANE/pane_inventory.json"
COMPLETION="$PANE/pane_completion.json"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || printf '%s' unknown)
INNER_STARTED=false
INNER_RC=125
ERROR_MESSAGE=''
SIGNAL_NAME=null
FINALIZED=false
{env_lines}
write_inventory() {{
  tmp="$PANE/.pane_inventory.json.tmp.$$"
  rows="$PANE/.pane_rows.json.tmp.$$"
  printf '%s' '[' > "$rows"
  first=true
  for name in pane_console.log pane_error.json pane_exit_code.txt pane_plan.json pane_receipt.json pane_start.json pane_timing.json pane_wrapper.sh; do
    path="$PANE/$name"
    if test -f "$path"; then
      size=$(wc -c < "$path" | tr -d ' ')
      sha=$(sha256sum "$path" 2>/dev/null | awk '{{print $1}}')
      if $first; then first=false; else printf '%s' ',' >> "$rows"; fi
      printf '{{"relative_path":"%s","size_bytes":%s,"sha256":"%s"}}' "$name" "$size" "$sha" >> "$rows"
    fi
  done
  printf '%s' ']' >> "$rows"
  row_sha=$(sha256sum "$rows" 2>/dev/null | awk '{{print $1}}')
  printf '{{"schema_version":1,"excluded_from_inventory":["pane_completion.json","pane_inventory.json"],"artifacts":%s,"canonical_inventory_sha256":"%s"}}\n' "$(cat "$rows")" "$row_sha" > "$tmp"
  rm -f "$rows"
  mv "$tmp" "$INVENTORY"
}}
finalize() {{
  if $FINALIZED; then return; fi
  FINALIZED=true
  FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || printf '%s' unknown)
  printf '%s\n' "$INNER_RC" > "$EXIT"
  printf '{{"schema_version":1,"started_at_utc":"%s","finished_at_utc":"%s","inner_started":%s,"inner_returncode":%s,"signal_name":%s}}\n' "$STARTED_AT" "$FINISHED_AT" "$INNER_STARTED" "$INNER_RC" "$SIGNAL_NAME" > "$TIMING"
  if test -n "$ERROR_MESSAGE"; then
    printf '{{"schema_version":1,"status":"FAILED","message":"%s","original_error_preserved":true}}\n' "$ERROR_MESSAGE" > "$ERROR"
  fi
  write_inventory
  isize=$(wc -c < "$INVENTORY" | tr -d ' ')
  isha=$(sha256sum "$INVENTORY" 2>/dev/null | awk '{{print $1}}')
  printf '{{"schema_version":1,"status":"PANE_COMPLETED","inner_started":%s,"inner_returncode":%s,"inner_argv_sha256":"%s","plan_sha256":"%s","pane_inventory":{{"relative_path":"pane_inventory.json","size_bytes":%s,"sha256":"%s"}},"signal_name":%s}}\n' "$INNER_STARTED" "$INNER_RC" {shlex.quote(argv_sha256(inner))} "$PLAN_SHA" "$isize" "$isha" "$SIGNAL_NAME" > "$COMPLETION"
}}
trap finalize EXIT
trap 'SIGNAL_NAME="HUP"; INNER_RC=129; exit 129' HUP
trap 'SIGNAL_NAME="INT"; INNER_RC=130; exit 130' INT
trap 'SIGNAL_NAME="TERM"; INNER_RC=143; exit 143' TERM
: > "$CONSOLE"
printf '{{"schema_version":1,"state":"STARTED","started_at_utc":"%s","pane_pid":%s,"plan_relative_path":"pane_plan.json"}}\n' "$STARTED_AT" "$$" > "$START"
printf '{{"schema_version":1,"state":"RECEIVED","started_at_utc":"%s","plan_sha256":"%s","config_file_sha256":"%s","inner_argv_sha256":"%s"}}\n' "$STARTED_AT" "$PLAN_SHA" "$CONFIG_SHA" {shlex.quote(argv_sha256(inner))} > "$RECEIPT"
printf '%s\n' 'P3_SMOKE_PANE_SHELL_STARTED' >> "$CONSOLE"
if test ! -f "$PLAN" || test "$(sha256sum "$PLAN" 2>/dev/null | awk '{{print $1}}')" != "$PLAN_SHA"; then
  ERROR_MESSAGE='pane plan binding failed'
elif test ! -f "$CONFIG" || test "$(sha256sum "$CONFIG" 2>/dev/null | awk '{{print $1}}')" != "$CONFIG_SHA"; then
  ERROR_MESSAGE='config binding failed before inner launcher'
fi
if test -z "$ERROR_MESSAGE"; then
  INNER_STARTED=true
  set +e
  {inner_cmd} >> "$CONSOLE" 2>&1
  INNER_RC=$?
  set -u
fi
exit "$INNER_RC"
"""


def _finish(
    outer: Path,
    launcher: Path,
    invocation: dict[str, Any],
    status: str,
    returncode: int | None,
    started_monotonic: float,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if error is not None:
        _write_json(launcher / "outer_error.json", {
            "schema_version": 1,
            "status": "FAILED",
            "exception_type": type(error).__name__,
            "message": str(error),
            "original_error_preserved": True,
        })
    _write_json(launcher / "outer_timing.json", {
        "schema_version": 1,
        "started_at_utc": invocation["created_at_utc"],
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
    })
    _write_json(launcher / OUTER_PARTIAL_INVENTORY, _inventory(launcher, OUTER_EXCLUDED))
    inventory_sha = _write_json(launcher / OUTER_INVENTORY, _inventory(launcher, OUTER_EXCLUDED))
    completion = {
        "schema_version": 1,
        "status": status,
        "nonportable": True,
        "tmux_called": returncode is not None,
        "tmux_returncode": returncode,
        "signal_events": [],
        "invocation": _file_ref(launcher, "outer_invocation.json"),
        "config_binding": _file_ref(launcher, "outer_config_binding.json"),
        "preflight": _file_ref(launcher, "preflight_audit.json"),
        "tmux_invocation": _file_ref(launcher, "tmux_invocation.json"),
        "tmux_stdout": _file_ref(launcher, "tmux_stdout.log"),
        "tmux_stderr": _file_ref(launcher, "tmux_stderr.log"),
        "outer_inventory": {"relative_path": OUTER_INVENTORY, "sha256": inventory_sha},
        "outer_partial_inventory": _file_ref(launcher, OUTER_PARTIAL_INVENTORY),
        "outer_timing": _file_ref(launcher, "outer_timing.json"),
        "error": _file_ref(launcher, "outer_error.json"),
        "failure": None if error is None else {"type": type(error).__name__, "message": str(error)},
        "finished_at_utc": _utc_now(),
    }
    completion_sha = _write_json(launcher / OUTER_COMPLETION, completion)
    return {"status": status, "completion_sha256": completion_sha, "outer_evidence_dir": str(outer)}


def run_outer_launch(
    *,
    repo_root: Path,
    data_root: Path,
    config_path: Path,
    output_dir: Path,
    process_evidence_dir: Path,
    outer_evidence_dir: Path,
    child_python: Path,
    tmux_executable: Path,
    tmux_session: str,
) -> dict[str, Any]:
    """Prepare evidence and issue exactly one tmux client call."""

    repo_root = Path(repo_root).resolve()
    outer = Path(outer_evidence_dir)
    started_monotonic = time.monotonic()
    _validate_outer_parent(outer)
    _ascii(tmux_session, "tmux_session")
    outer.mkdir()
    launcher = outer / "launcher"
    pane = outer / "pane"
    launcher.mkdir()
    pane.mkdir()
    invocation = {
        "schema_version": 1,
        "mode": "outer_launch",
        "status": "OUTER_PREPARED",
        "nonportable": True,
        "created_at_utc": _utc_now(),
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "process_evidence_dir": str(process_evidence_dir),
        "outer_evidence_dir": str(outer),
        "child_python": str(child_python),
        "tmux_executable": str(tmux_executable),
        "tmux_session": tmux_session,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED": "1",
            "PYTHONPATH": str(repo_root / "src"),
        },
    }
    _write_json(launcher / "outer_invocation.json", invocation)
    try:
        config_arg = Path(config_path)
        if not config_arg.is_absolute():
            config_arg = repo_root / config_arg
        config, binding = _bind_config(repo_root, config_arg)
        data = _validate_data_root(Path(data_root))
        output = Path(output_dir)
        process = Path(process_evidence_dir)
        _validate_runtime_paths(repo_root, config, output, process, outer)
        _regular_file(Path(child_python), executable=True)
        _regular_file(Path(tmux_executable), executable=True)
        for value, field in (
            (str(repo_root), "repo_root"), (str(data), "data_root"), (str(output), "output_dir"),
            (str(process), "process_evidence_dir"), (str(outer), "outer_evidence_dir"),
            (str(child_python), "child_python"), (str(tmux_executable), "tmux_executable"),
        ):
            _ascii(value, field)
        _write_json(launcher / "outer_config_binding.json", binding)
        environment = dict(invocation["environment"])
        inner = _inner_argv(repo_root, data, output, process, Path(binding["config_path"]), Path(child_python))
        inner_sha = argv_sha256(inner)
        nonce = sha256_bytes(canonical_json_bytes({"created_at_utc": invocation["created_at_utc"], "inner_argv_sha256": inner_sha}))[:32]
        plan = {
            "schema_version": 1,
            "state": "PREPARED",
            "single_use": True,
            "nonce": nonce,
            "smoke_id": SMOKE_V2_ID,
            "repo_root": str(repo_root),
            "data_root": str(data),
            "output_dir": str(output),
            "process_evidence_dir": str(process),
            "outer_evidence_dir": str(outer),
            "config_binding": binding,
            "child_python": str(child_python),
            "inner_argv": inner,
            "inner_argv_sha256": inner_sha,
            "tmux_session": tmux_session,
            "environment": environment,
            "created_at_utc": _utc_now(),
            "nonportable": True,
        }
        plan_sha = _write_json(pane / "pane_plan.json", plan)
        wrapper_path = pane / "pane_wrapper.sh"
        wrapper_bytes = _wrapper(pane, pane / "pane_plan.json", plan_sha, Path(binding["config_path"]), binding["config_file_sha256"], inner, environment).encode("ascii")
        atomic_write(wrapper_path, wrapper_bytes)
        wrapper_path.chmod(0o755)
        tmux_argv = [str(tmux_executable), "new-session", "-d", "-s", tmux_session, "-c", str(repo_root), str(wrapper_path)]
        tmux_identity = {
            "path": str(tmux_executable),
            "size_bytes": tmux_executable.stat().st_size,
            "sha256": sha256_file(tmux_executable),
            "mode": stat.S_IMODE(tmux_executable.stat().st_mode),
        }
        _write_json(launcher / "preflight_audit.json", {
            "schema_version": 1,
            "status": "PASS",
            "config_binding": binding,
            "data_root": str(data),
            "output_process_not_created_by_outer": True,
            "tmux_executable_identity": tmux_identity,
            "wrapper_path": str(wrapper_path),
            "wrapper_sha256": sha256_bytes(wrapper_bytes),
            "pane_plan_sha256": plan_sha,
            "inner_argv_sha256": inner_sha,
            "nonce": nonce,
            "created_at_utc": plan["created_at_utc"],
        })
        _write_json(launcher / "tmux_invocation.json", {
            "schema_version": 1,
            "argv": tmux_argv,
            "argv_sha256": argv_sha256(tmux_argv),
            "wrapper_sha256": sha256_bytes(wrapper_bytes),
            "pane_plan_sha256": plan_sha,
            "called_at_utc": _utc_now(),
            "tmux_executable_identity": tmux_identity,
        })
        result = subprocess.run(tmux_argv, cwd=repo_root, env={**os.environ, **environment}, capture_output=True, check=False)
        stdout = result.stdout if isinstance(result.stdout, bytes) else str(result.stdout or "").encode("utf-8")
        stderr = result.stderr if isinstance(result.stderr, bytes) else str(result.stderr or "").encode("utf-8")
        atomic_write(launcher / "tmux_stdout.log", stdout)
        atomic_write(launcher / "tmux_stderr.log", stderr)
        return _finish(outer, launcher, invocation, "TMUX_ACCEPTED" if result.returncode == 0 else "TMUX_REJECTED", result.returncode, started_monotonic)
    except BaseException as error:
        status = "PREFLIGHT_FAILED" if not (launcher / "tmux_invocation.json").exists() else "INTERRUPTED_WITH_EVIDENCE"
        return _finish(outer, launcher, invocation, status, None, started_monotonic, error)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def classify_outer_evidence(outer_evidence_dir: Path, output_dir: Path, process_evidence_dir: Path) -> str:
    """Classify existing evidence without modifying it."""

    root = Path(outer_evidence_dir)
    launcher = root / "launcher"
    pane = root / "pane"
    if not root.is_dir() or root.is_symlink() or not launcher.is_dir() or not pane.is_dir():
        return "UNKNOWN"
    completion = _read_json(launcher / OUTER_COMPLETION)
    if completion is None:
        return "OUTER_PREPARED" if (launcher / "outer_invocation.json").is_file() else "UNKNOWN"
    status = completion.get("status")
    if status == "PREFLIGHT_FAILED":
        return "PREFLIGHT_FAILED"
    if status == "TMUX_REJECTED":
        return "TMUX_REJECTED"
    if status not in {"TMUX_ACCEPTED", "INTERRUPTED_WITH_EVIDENCE"}:
        return "UNKNOWN"
    pane_started = (pane / "pane_start.json").is_file()
    pane_exit = (pane / "pane_exit_code.txt").is_file()
    process = Path(process_evidence_dir).is_dir()
    output = Path(output_dir).is_dir()
    if output and not process or process and not pane_started:
        return "UNKNOWN"
    if not pane_started:
        return "TMUX_ACCEPTED_PANE_NOT_STARTED"
    if not pane_exit:
        return "PANE_STARTED_INNER_NOT_STARTED"
    if not process:
        return "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"
    process_completion = _read_json(Path(process_evidence_dir) / "process_completion.json")
    entry_completion = _read_json(Path(output_dir) / "completion.json") if output else None
    if process_completion is None:
        if entry_completion is not None:
            return "ENTRY_PRESENT"
        return "INNER_PROCESS_EVIDENCE_PRESENT"
    if entry_completion is None:
        return "TERMINAL_FAILED" if process_completion.get("status") != "COMPLETED" else "INNER_PROCESS_EVIDENCE_PRESENT"
    if process_completion.get("status") == "COMPLETED" and entry_completion.get("status") == "COMPLETED":
        return "TERMINAL_COMPLETE"
    return "TERMINAL_FAILED"


def contract_check(repo_root: Path, config_path: Path) -> dict[str, Any]:
    config_arg = Path(config_path)
    if not config_arg.is_absolute():
        config_arg = Path(repo_root).resolve() / config_arg
    config, binding = _bind_config(Path(repo_root).resolve(), config_arg)
    return {
        "status": "PASS",
        "mode": "outer-contract-check",
        "smoke_id": config["smoke_id"],
        "config_binding": binding,
        "torch_imported": False,
        "real_data_accessed": False,
        "dataset_or_dataloader_constructed": False,
        "model_constructed": False,
        "output_directory_created": False,
        "process_evidence_created": False,
        "outer_evidence_created": False,
        "tmux_called": False,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtdetr-baseline-smoke-outer")
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("contract-check")
    check.add_argument("--repo-root", type=Path, required=True)
    check.add_argument("--config", type=Path, required=True)
    classify = sub.add_parser("classify")
    classify.add_argument("--outer-evidence-dir", type=Path, required=True)
    classify.add_argument("--output-dir", type=Path, required=True)
    classify.add_argument("--process-evidence-dir", type=Path, required=True)
    launch = sub.add_parser("launch")
    for name in ("repo-root", "data-root", "config", "output-dir", "process-evidence-dir", "outer-evidence-dir", "child-python", "tmux-executable", "tmux-session"):
        launch.add_argument("--" + name, required=True, type=str if name == "tmux-session" else Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "contract-check":
            print(canonical_json_bytes(contract_check(args.repo_root, args.config)).decode("utf-8"))
            return 0
        if args.mode == "classify":
            print(classify_outer_evidence(args.outer_evidence_dir, args.output_dir, args.process_evidence_dir))
            return 0
        result = run_outer_launch(
            repo_root=args.repo_root,
            data_root=args.data_root,
            config_path=args.config,
            output_dir=args.output_dir,
            process_evidence_dir=args.process_evidence_dir,
            outer_evidence_dir=args.outer_evidence_dir,
            child_python=args.child_python,
            tmux_executable=args.tmux_executable,
            tmux_session=args.tmux_session,
        )
        print(canonical_json_bytes(result).decode("utf-8"))
        return 0
    except Exception as error:
        print(f"rtdetr-baseline-smoke-outer: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
