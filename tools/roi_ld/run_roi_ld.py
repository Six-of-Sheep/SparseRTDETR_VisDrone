"""Registered r24 driver: preflight the frozen binding, launch the eight phases, summarise the run.

The engine itself owns all admission (source/manifest/plan digests, boot identity, GPU and disk
safety, duplicate-output refusal, one-shot token) inside ``slice_ft_execution``.  This driver adds
only the outer self-checks the round plan promises: that the release tree on disk matches the
manifest the plan names, that the driver bytes match the manifest, that the GPU has no compute
application, and that no attempt output exists yet.  It never retries and never repairs.

usage:
  run_roi_ld.py preflight --binding <plan-003 binding json>
  run_roi_ld.py launch    --binding <plan-003 binding json>
  run_roi_ld.py summary   --release <release dir> --contract <production contract>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DRIVER_REL = "tools/roi_ld/run_roi_ld.py"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding_check(binding_path):
    """Verify the binding document against the release tree it names."""
    b = load(binding_path)
    runner, science = b["runner"], b["science_plan"]
    release = Path(runner["release"])
    source = Path(runner["release_source"])
    manifest = Path(runner["release_manifest"])
    if not source.is_dir():
        raise SystemExit("release source missing: " + str(source))
    if sha(manifest) != runner["release_manifest_sha256"]:
        raise SystemExit("manifest bytes differ from the binding")
    if sha(science["path"]) != science["sha256"]:
        raise SystemExit("frozen science plan bytes differ from the binding")
    rows = {row["path"]: row for row in load(manifest)["source_files"]}
    if sha(source / DRIVER_REL) != rows[DRIVER_REL]["sha256"]:
        raise SystemExit("driver bytes differ from the manifest")
    return b, runner, source, manifest, rows


def preflight(binding_path):
    b, runner, source, manifest, rows = binding_check(binding_path)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True)
    busy = [line for line in apps.stdout.splitlines() if line.strip()]
    if busy:
        raise SystemExit("GPU busy: %s" % busy)
    for key in ("job_root", "runtime_root"):
        if Path(runner[key]).exists():
            raise SystemExit("%s already exists: %s" % (key, runner[key]))
    value = {"status": "PASS", "scope": "PREFLIGHT", "release": str(runner["release"]),
             "manifest_sha256": runner["release_manifest_sha256"], "commit": runner["commit"],
             "science_plan_sha256": b["science_plan"]["sha256"], "driver_sha256": rows[DRIVER_REL]["sha256"],
             "gpu_compute_apps": 0, "job_root": runner["job_root"], "CUDA_initialized": False,
             "training_started": False, "token_consumed": False}
    print(json.dumps(value, indent=1))
    return 0


def summarise(release, contract, job_root=None):
    sys.path.insert(0, str(Path(release) / "src"))
    from sparse_rtdetr.protocol_r18 import slice_ft_execution as ex
    c = ex.load(contract)
    root = Path(job_root or c["job_root"])
    value = {"status": "SUMMARY", "contract": {"path": c["contract_path"], "sha256": c["contract_sha256"],
                                              "scope": c["scope"], "task_id": c["task_id"]},
             "job_root": str(root), "chain_exit": None, "run_result": None, "evaluation": None}
    if (root / "CHAIN_EXIT.json").is_file():
        value["chain_exit"] = load(root / "CHAIN_EXIT.json")
    if (root / "RESULT.json").is_file():
        value["run_result"] = load(root / "RESULT.json")
    if (root / "evaluation" / "RESULT.json").is_file():
        result = load(root / "evaluation" / "RESULT.json")
        comparison = load(root / "evaluation" / "COMPARISON.json")
        value["evaluation"] = {"quality": result["quality"], "gates": result["gates"],
                               "cost_passed": result["cost_passed"],
                               "score_budget_calls": result["score_budget_calls"],
                               "scored_endpoint": result["scored_endpoint"],
                               "verdict": comparison["verdict"],
                               "differences_pp": comparison["differences_pp"]["A2_minus_A0"]["600"],
                               "measured_cost_ratios": result["measured_cost_ratios"]}
    print(json.dumps(value, ensure_ascii=False, indent=1))
    return 0


def launch(binding_path):
    b, runner, source, manifest, rows = binding_check(binding_path)
    preflight(binding_path)
    sys.path.insert(0, str(source / "src"))
    from sparse_rtdetr.protocol_r18 import slice_ft_execution as ex
    c = ex.load(runner["contract"])
    code = ex.launch(c, runner["token"])
    summarise(runner["release"], runner["contract"], runner["job_root"])
    print("driver_exit", code)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "launch", "summary"))
    parser.add_argument("--binding")
    parser.add_argument("--release")
    parser.add_argument("--contract")
    parser.add_argument("--job-root")
    args = parser.parse_args()
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    if args.action == "preflight":
        return preflight(args.binding)
    if args.action == "launch":
        return launch(args.binding)
    return summarise(args.release, args.contract, args.job_root)


if __name__ == "__main__":
    raise SystemExit(main())
