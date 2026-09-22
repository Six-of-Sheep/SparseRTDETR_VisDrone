#!/usr/bin/env python3
"""Clean-environment entrypoint for the REV1 smoke controller.

This file intentionally imports only the Python standard library. It reads a
sealed structured invocation, validates the controller environment, and then
execs the frozen controller interpreter. Consequently PYTHONPATH is present
when the controller interpreter starts importing sparse_rtdetr; it is not
patched after import and no ambient interactive-shell environment is trusted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REQUIRED_ENV = (
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "PATH",
    "HOME",
)
PLACEHOLDER_MARKERS = ("TBD", "AUTO", "latest", "current", "infer-at-launch")


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def fail(reason: str) -> int:
    print(
        json.dumps(
            {"status": "CONTROLLER_STARTUP_ENVIRONMENT_FAIL", "error": reason},
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def walk_strings(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_strings(key)
            yield from walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, str):
        yield value


def load_invocation(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical(value):
        raise ValueError("INVOCATION_NON_CANONICAL")
    if not isinstance(value, dict) or value.get("schema_version") not in {2, 3}:
        raise ValueError("INVOCATION_SCHEMA_MISMATCH")
    if any(
        marker in item for item in walk_strings(value) for marker in PLACEHOLDER_MARKERS
    ):
        raise ValueError("INVOCATION_PLACEHOLDER")
    env = value.get("controller_env")
    argv = value.get("controller_argv")
    controller_python = value.get("controller_python")
    launcher = value.get("controller_launcher")
    repo = Path(str(value.get("repository_root", ""))).resolve(strict=True)
    cwd = Path(str(value.get("working_directory", ""))).resolve(strict=True)
    if cwd != repo or not cwd.is_dir():
        raise ValueError("CONTROLLER_WORKING_DIRECTORY_MISMATCH")
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise ValueError("CONTROLLER_ENVIRONMENT_INVALID")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(item, str) for item in argv
    ):
        raise ValueError("CONTROLLER_ARGV_INVALID")
    if not isinstance(controller_python, str) or argv[0] != controller_python:
        raise ValueError("CONTROLLER_INTERPRETER_BINDING_MISMATCH")
    if not isinstance(launcher, str) or len(argv) < 2 or argv[1] != launcher:
        raise ValueError("CONTROLLER_LAUNCHER_BINDING_MISMATCH")
    if not Path(controller_python).is_file() or not os.access(
        controller_python, os.X_OK
    ):
        raise ValueError("CONTROLLER_INTERPRETER_UNAVAILABLE")
    if not Path(launcher).is_file():
        raise ValueError("CONTROLLER_LAUNCHER_UNAVAILABLE")
    expected_src = str((repo / "src").resolve(strict=True))
    for name in REQUIRED_ENV:
        if name not in env or env[name] == "":
            raise ValueError(f"CONTROLLER_ENVIRONMENT_MISSING:{name}")
    if env["PYTHONPATH"] != expected_src:
        raise ValueError("CONTROLLER_PYTHONPATH_MISMATCH")
    if env["PYTHONHASHSEED"] != "0":
        raise ValueError("CONTROLLER_PYTHONHASHSEED_MISMATCH")
    if env["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("CONTROLLER_CUBLAS_WORKSPACE_CONFIG_MISMATCH")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--prelaunch-only", action="store_true")
    args = parser.parse_args()
    try:
        invocation = load_invocation(args.invocation.resolve(strict=True))
        env = {
            str(key): str(value)
            for key, value in invocation["controller_env"].items()
        }
        if args.prelaunch_only:
            env["REV1_PRELAUNCH_ONLY"] = "1"
        os.chdir(invocation["working_directory"])
        os.execve(invocation["controller_python"], invocation["controller_argv"], env)
    except Exception as exc:
        return fail(f"{type(exc).__name__}:{exc}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
