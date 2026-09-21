#!/usr/bin/env python3
"""REV1 smoke controller entry point with explicit runtime locators.

No physical locator is baked into this execution identity; all locators are
provided by the launch contract and are verified by the child before READY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_rtdetr.baseline.training_v2b_smoke_controller import build_plan, launch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--derived", required=True)
    parser.add_argument("--derived-sha", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--policy-size", type=int)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--auth-id", required=True)
    parser.add_argument("--auth-sha", required=True)
    parser.add_argument("--exec-source-sha", required=True)
    parser.add_argument("--exec-contract-sha", required=True)
    parser.add_argument("--bridge-binding-sha", required=True)
    parser.add_argument("--bridge-manifest-sha", required=True)
    parser.add_argument("--training-contract-sha", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--cpu-rehearsal", action="store_true")
    args = parser.parse_args()
    runner_argv = [
        args.runner,
        "--root", args.root,
        "--repo", args.repo,
        "--derived", args.derived,
        "--derived-sha", args.derived_sha,
        "--manifest", args.manifest,
        "--policy", args.policy,
        "--policy-sha", args.policy_sha,
        "--auth", args.auth,
        "--auth-id", args.auth_id,
        "--auth-sha", args.auth_sha,
        "--exec-source-sha", args.exec_source_sha,
        "--exec-contract-sha", args.exec_contract_sha,
        "--bridge-binding-sha", args.bridge_binding_sha,
        "--bridge-manifest-sha", args.bridge_manifest_sha,
        "--training-contract-sha", args.training_contract_sha,
        "--gpu-uuid", args.gpu_uuid,
    ]
    if args.policy_size is not None:
        runner_argv += ["--policy-size", str(args.policy_size)]
    runner_argv += ["--cpu-rehearsal" if args.cpu_rehearsal else ""]
    runner_argv = [item for item in runner_argv if item]
    environment = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path(args.repo) / "src"),
        "MKL_THREADING_LAYER": "GNU",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "CUDA_VISIBLE_DEVICES": "" if args.cpu_rehearsal else "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8",
    }
    plan = build_plan(
        launch_id="rev1-gpu-smoke-bridge006-cpu-rehearsal" if args.cpu_rehearsal else "rev1-gpu-smoke-bridge006",
        evidence_root=Path(args.root),
        repo_root=Path(args.repo),
        python_executable=Path(args.python_executable),
        runner_argv=runner_argv,
        tmux_executable=Path("tmux"),
        session_name=args.session,
        environment=environment,
        ready_timeout=300.0,
        exit_timeout=300.0,
    )
    result = launch(plan)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("status") == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
