"""TASK_S15_SEED12_REPLICATION_001: immutable preparation and detached, journaled S15 seed1/seed2 replication.

  freeze        --release release-NNN                     frozen git-archive release + manifest
  a0-per-class  --release release-NNN --name A0_PER_CLASS-NNN.json
                                                          single CPU COCOeval of each pinned A0 top600 (2 total)
  contract      --release release-NNN --attempt formal-NNN --a0-per-class NAME [--prior formal-MMM]
  cpu-gate      --contract C --kind train|eval --seed S --out DIR   CUDA-hidden CPU gate from the frozen release
  preflight / launch --gates G1 G2 G3 G4 / status --contract C

Both seeds are planned, bound and frozen in one contract before any new endpoint exists.  The
detached supervisor runs: accept(seed1), accept(seed2), then per seed in fixed order 1, 2:
train-S15 -> input-alignment -> evaluation; finally the fixed three-seed summary.  A seed's result
never changes the other seed's task.  Seed0 (TASK_S15_EXECUTE_001, commit 61646b3) is referenced only.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
EXPERIMENTS = W / "experiments/localization-distribution"
ROUND = EXPERIMENTS / "s15-seed12-replication-20261106-001"
SEED0_ROUND = EXPERIMENTS / "s15-scale-baseline-20261105-001"
PYTHON = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
GPU = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
TASK_NAME = "TASK_S15_SEED12_REPLICATION_001.md"
TASK_SHA = "a3dd6b8f6eb703a678805bfe8b8140c25e4c428c3ecda9f279d61b7af79c72d6"
DATA_REF = EXPERIMENTS / "r18-tiny-positives-20261102-001/production-contract-002.json"
DATA_REF_SHA = "b0fe33198d8abc7f8b89a1cc9824de9abdba8721580932aa07be565ce40d896b"
R25_RELEASE_MANIFEST = EXPERIMENTS / "r25-query-local-regression-20261104-001/release-002/MANIFEST.json"
R25_RELEASE_MANIFEST_SHA = "31e6fb65ba0ef883b03afdeae459def5a25c528ef3264007d1501b21dc0ddb4c"
HIST = EXPERIMENTS / "r18-sahi-seed12-20261016-001"
RESUME = HIST / "protocol-resume-20261017-001"
PARENT_BYTES = 323_459_201
SEEDS = {
    1: {"formal": HIST / "seed1/formal", "infer": HIST / "seed1/infer-A", "evaluation": RESUME / "seed1/evaluation",
        "parent_sha": "6f8fd316a4e508b9cc8a76be2a18f7c5aaa0beb1b5bd809fd27cb83a1de05234",
        "e120_sha": "834cd4de9d4018ac9093f48c9ab459b88a19cb039bea1f780aa2d3ace11f9731",
        "metrics_sha": {600: "9f54db25849086de4b6701fe483d4cbf08acc4bd3e04cbad5f0d12ca1dd1e1f4",
                        300: "7c0a0d0d04a7aefc3e46dcec0a486131e5ba170bddaf63474c599e0310052b1b"}},
    2: {"formal": RESUME / "seed2/formal", "infer": RESUME / "seed2/infer-A", "evaluation": RESUME / "seed2/evaluation",
        "parent_sha": "c36e3c1fd4e5ad05cbbf8367d3e2b2c465858e0d69a233839c833ef29a200dd6",
        "e120_sha": "0a356a41c9b62e9ef5a934749ddb8bd12b088ca7954b96ba5ffe9b2de6ca4058",
        "metrics_sha": {600: "68f4833c74b437e0297c6cf0bc108907ed4f2577ef7ba6912ee477317e5c0804",
                        300: "adb472850ca87d708a86c1612cd41b902b19332a0e7740399d58805b7fc264d6"}},
}
SEED_ORDER = [1, 2]
SEED0 = {  # TASK_S15_EXECUTE_001 results and reusable engineering evidence (pinned, never rerun)
    "A0_768_equivalence": (SEED0_ROUND / "formal-002/accept/A0_768_EQUIVALENCE.json",
                           "c000130d8d2b690a78d7b031556296b37565ed3ce50b9b1f56d9d685dadd87fe"),
    "accept_closed": (SEED0_ROUND / "formal-002/accept/CLOSED.json",
                      "ffc0373440169726090c00a5236d05ef02dc684dae1d06e214550b3c8b866ff7"),
    "formal_closed": (SEED0_ROUND / "formal-002/CLOSED.json",
                      "65050257025304dff6135758fd9eed107b7d235398a49df03a1b39403e250c7b"),
    "contract": (SEED0_ROUND / "formal-002-CONTRACT.json",
                 "14e502a69a55496b53be8a89d078fba20322185731ebbe8f46d519b8fea852a1"),
    "comparison": (SEED0_ROUND / "formal-002/evaluation/COMPARISON.json",
                   "78b486390b60869533ece6f1160a744c741ec2eac1644a61c07f6c2ed94a6759"),
    "release_manifest": (SEED0_ROUND / "release-002/MANIFEST.json",
                         "6fff158dc77aa9e52374531f456d1e1be0851c5a93db00b2fff0cd98b87eb7bf"),
}
ALLOCATOR = "expandable_segments:True"
LOADER_CHECK_UPDATES = (18241, 35569)
WORKLOAD = {
    "arm": "S15", "seeds": SEED_ORDER, "seed_order": "sequential 1 then 2; fixed before any new endpoint",
    "epochs": [61, 120], "initial_update": 18240, "updates_per_epoch": 304,
    "formal_updates_per_seed": 18240, "stop_update": 36480, "save_epochs": [60, 75, 90, 105, 120],
    "train_batch": 8, "accumulation": 2, "precision": "bfloat16", "clip_norm": 0.1, "input_hw": [1152, 2016],
    "activation_checkpoint": True,
    "checkpoint_segments": ["backbone.res_layers", "encoder.fpn_blocks", "encoder.pan_blocks"],
    "allocator_conf": ALLOCATOR,
    "acceptance": {"S15_updates_per_seed": 3, "S15_training_forwards_per_seed": 6, "EMA_inference_forwards": 0,
                   "reused": "seed0 768 plain/checkpoint complete CUDA equivalence; not redone"},
    "evaluation": {
        "epoch": 120, "weights": ["ema"], "primary_weights": "ema",
        "online": "kept in the complete e120 checkpoint; not inferred or scored (pre-declared readout scope)",
        "view": "A", "images": 548, "batch": 4, "precision": "float32", "input_hw": [1152, 2016],
        "warmup_batches_per_endpoint": 4, "budgets": [600, 300], "score_calls_per_seed": 2,
        "per_class_APL_budget": 600, "S15_per_class_APL_extra_COCOeval_calls_per_seed": 1,
        "A0_per_class_APL_extra_COCOeval_calls_total": 2,
    },
}
DECISION_RULE = {
    "comparison": "per seed: S15 EMA e120 minus the same seed's historical native A0 e120 EMA (certified, pinned)",
    "primary": {"top600_APS": 0.50, "top600_AP75_small": 0.30},
    "protections": {"top600_AP": -0.10, "top300_AP": -0.20, "top300_APS": -0.20, "top600_APM": -0.50,
                    "top600_APL": -0.50},
    "per_seed_pass": "seed_passes_original_rule", "per_seed_fail": "closed_fixed_config",
    "seed0": "closed_fixed_config kept unchanged",
    "aggregate": "seed0/1/2 differences, mean, sample SD (n-1), min, max reported only; never overrides a seed's FAIL",
    "branches": {
        "1": "seed1 and seed2 both pass both primary cells: small-object primary gain repeated in three seeds; "
             "every protection failure listed",
        "2": "a new seed fails a primary cell: record size/direction and the seed; no general replication claim",
        "3": "APL protection failure repeated: size trade-off; inconsistent APL across seeds: cost stability unclear",
    },
    "mechanism_gate": None, "originality_claim": None,
}
RESOURCE = {
    "reference": {"train_process_wall_hours": "about 2 x 13.53 (seed0 measured; not a cap)",
                  "additional_bytes_estimate": 5_000_000_000},
    "artificial_resource_caps": None,
    "no_progress_seconds": 1800,
    "reserve_disk_bytes": 4 * 1024**3,
    "remaining_save_estimate_bytes": 3 * 1024**3,
}
MODEL_FILES = (
    "src/sparse_rtdetr/protocol_r18/model.py",
    "src/sparse_rtdetr/protocol_r18/engine.py",
    "src/sparse_rtdetr/protocol_r18/geometry.py",
    "src/sparse_rtdetr/protocol_r18/data.py",
    "src/sparse_rtdetr/protocol_r18/checkpoint.py",
    "src/sparse_rtdetr/localization/data.py",
    "src/sparse_rtdetr/baseline/training_v2b.py",
    "src/sparse_rtdetr/baseline/training_v2b_data.py",
    "src/sparse_rtdetr/baseline/training_v2b_engine.py",
    "src/sparse_rtdetr/research/query_local_runtime.py",
    "src/sparse_rtdetr/r0/evaluation.py",
    "src/sparse_rtdetr/r0/evaluation_data.py",
    "tools/query_local/formal_data.py",
    "infer_seed.py",
)
S15_FILES = ("src/sparse_rtdetr/research/scale_baseline.py",)
RUN_FILES = ("tools/scale_baseline/s15_formal_run.py", "tools/scale_baseline/s15_formal_train.py",
             "tools/scale_baseline/s15_formal_eval.py")
RELEASE_VERSION = "S15-SEED12-001-release-v1"
CONTRACT_VERSION = "S15-SEED12-001-contract-v1"


def require(condition, error):
    if not condition:
        raise ValueError(error)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def journal(path, value):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def ref(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def checked_ref(reference):
    actual = ref(reference["path"])
    require(actual == reference, f"material reference differs: {actual['path']}")
    return Path(actual["path"])


def pinned(path, sha, size=None):
    result = ref(path)
    require(result["sha256"] == sha and (size is None or result["bytes"] == size), f"pinned file mismatch: {path}")
    return result


def suffix(value, prefix):
    require(re.fullmatch(prefix + r"-[0-9]{3}", value), f"invalid {prefix} identity")
    return value


def source_paths(source):
    paths = [x for x in sorted(source.rglob("*")) if x.is_file() or x.is_symlink()]
    require(not any(p.is_symlink() for p in paths), "release contains a link")
    return paths


# ---------------------------------------------------------------- release
def freeze(release_name):
    release = ROUND / suffix(release_name, "release")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip(),
            "worktree must be clean to freeze")
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    r25_manifest = pinned(R25_RELEASE_MANIFEST, R25_RELEASE_MANIFEST_SHA)
    seed0_manifest = pinned(*SEED0["release_manifest"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    release.mkdir()
    source = release / "source"
    source.mkdir()
    archive = subprocess.check_output(["git", "archive", commit], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for entry in tar.getmembers():
            p = source / entry.name
            require(p.resolve().is_relative_to(source) and (entry.isdir() or entry.isfile()),
                    f"unsafe archive member: {entry.name}")
        tar.extractall(source)
    rows = []
    for path in source_paths(source):
        relative = path.relative_to(source).as_posix()
        actual = ref(path)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == actual["sha256"], f"archive blob differs: {relative}")
        rows.append({"path": relative, "sha256": actual["sha256"], "bytes": actual["bytes"]})
    current = {row["path"]: row["sha256"] for row in rows}
    base = {row["path"]: row["sha256"] for row in read(r25_manifest["path"])["source_files"]}
    same = {p: base.get(p) == current.get(p) and p in current for p in MODEL_FILES}
    require(all(same.values()), f"native model/engine/data/evaluation computation differs from R25 A0 release: {same}")
    seed0 = {row["path"]: row["sha256"] for row in read(seed0_manifest["path"])["source_files"]}
    changed = sorted(p for p in set(seed0) | set(current) if seed0.get(p) != current.get(p))
    require(set(changed) <= set(S15_FILES + RUN_FILES), f"changes beyond S15 seed binding/run files: {changed}")
    path = release / "MANIFEST.json"
    write_new(path, {"version": RELEASE_VERSION, "task": task, "commit": commit,
                     "R25_A0_release_manifest": r25_manifest,
                     "native_files_identical_to_R25_A0_release": sorted(MODEL_FILES),
                     "seed0_S15_release_manifest": seed0_manifest, "changed_vs_seed0_S15_release": changed,
                     "source_files": rows})
    print(json.dumps(ref(path)), flush=True)


def verify_source(source, manifest):
    indexed = {row["path"]: row for row in manifest["source_files"]}
    found = {p.relative_to(source).as_posix(): p for p in source_paths(source)}
    require(len(indexed) == len(manifest["source_files"]) and indexed.keys() == found.keys(),
            "source inventory differs from release manifest")
    for name, path in found.items():
        row = indexed[name]
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"], f"source byte identity differs: {name}")


def release_manifest(release_name):
    release = ROUND / suffix(release_name, "release")
    manifest_ref = ref(release / "MANIFEST.json")
    manifest = read(manifest_ref["path"])
    require(manifest["version"] == RELEASE_VERSION and manifest["task"] == pinned(ROUND / TASK_NAME, TASK_SHA),
            "wrong release for the S15 seed12 replication")
    verify_source(release / "source", manifest)
    return release, manifest_ref, manifest


# ---------------------------------------------------------------- historical native A0 lineage
def a0_reference_refs(seed):
    """Verify that the seed's historical A0 e120 EMA descends from its e060 parent in one formal chain,
    with the native recipe/clock, and that the pinned metrics score that endpoint's own A predictions."""
    spec = SEEDS[seed]
    formal, infer, evaluation = spec["formal"], spec["infer"], spec["evaluation"]
    e060, e120 = formal / "checkpoint-epoch-060.pth", formal / "checkpoint-epoch-120.pth"
    result = read(formal / "RESULT.json")
    require((result["status"], result["seed"], result["epoch"], result["updates"], result["actual_updates"],
             result["checkpoint"]["sha256"], result["checkpoint"]["path"], result["used_acceptance_state"],
             result["evaluation_before_endpoint"], result["holdout_accessed"]) ==
            ("R18_COMMON_E120_EMA_LOCKED", seed, 120, 36480, 36480, spec["e120_sha"], str(e120), False, False, False),
            f"seed{seed} historical formal result")
    closed = read(formal / "CLOSED.json")
    require(closed["status"] == "PASS" and closed["seed"] == seed and closed["result"]["sha256"] == file_sha(formal / "RESULT.json"),
            f"seed{seed} historical formal closure")
    lock = read(formal / "endpoint-lock.json")
    require(lock["status"] == "R18_COMMON_E120_EMA_LOCKED" and lock["checkpoint"]["sha256"] == spec["e120_sha"],
            f"seed{seed} historical EMA endpoint lock")
    side = read(e060.with_name(e060.name + ".complete.json"))
    require((side["sha256"], side["size_bytes"], side["completed_updates"], side["path"]) ==
            (spec["parent_sha"], PARENT_BYTES, 18240, str(e060)), f"seed{seed} e060 sidecar")
    chain = []
    for epoch in range(60, 121):
        row = read(formal / f"epoch-{epoch:03d}-complete.json")
        ck = row["checkpoint"]
        require(row["seed"] == seed and row["epoch"] == epoch and row["updates"] == epoch * 304
                and ck["completed_updates"] == epoch * 304 and ck["path"] == str(formal / f"checkpoint-epoch-{epoch:03d}.pth")
                and ck["binding_sha256"] == side["binding_sha256"] and ck["scope"] == side["scope"],
                f"seed{seed} historical epoch chain at {epoch}")
        chain.append(ck["sha256"])
    require(chain[0] == spec["parent_sha"] and chain[-1] == spec["e120_sha"], f"seed{seed} chain endpoints")
    binding = read(formal / "binding.json")
    require(binding["config"]["seed"] == seed and binding["config"]["input_hw"] == [768, 1344]
            and binding["scope"] == side["scope"], f"seed{seed} historical training binding")
    audits = formal / "audits"
    missing = [u for u in range(18241, 36481) if not (audits / f"update-{u:05d}-direct.json").is_file()]
    require(not missing, f"seed{seed} historical audits missing: {missing[:5]}")
    inference = read(infer / "RESULT.json")
    require((inference["status"], inference["seed"], inference["checkpoint_sha256"], inference["images"],
             inference["model_batch_forwards"], inference["FP32"], inference["batch4"], inference["detector_updates"]) ==
            ("PASS", seed, spec["e120_sha"], 548, 137, True, True, 0), f"seed{seed} historical A inference")
    predictions = {str(b): inference["predictions"][str(b)] for b in (600, 300)}
    for b in predictions:
        require(ref(predictions[b]["path"]) == predictions[b], f"seed{seed} historical A prediction bytes")
    scored = read(evaluation / "RESULT.json")
    require(scored["status"] == "PASS" and scored["seed"] == seed and scored["checkpoint_sha256"] == spec["e120_sha"]
            and scored["predictions"]["A"] == predictions, f"seed{seed} historical evaluation binding")
    metrics = {str(b): pinned(evaluation / f"A-top{b}-metrics.json", spec["metrics_sha"][b]) for b in (600, 300)}
    return {"seed": seed, "A0_e120_sha256": spec["e120_sha"], "A0_e120_file": ref(e120),
            "A0_metrics": metrics, "A0_predictions": predictions,
            "A0_inference_result": ref(infer / "RESULT.json"), "A0_evaluation_result": ref(evaluation / "RESULT.json"),
            "A0_formal_result": ref(formal / "RESULT.json"), "A0_endpoint_lock": ref(formal / "endpoint-lock.json"),
            "lineage": {"formal_dir": str(formal), "e060_to_e120_checkpoint_sha256": chain,
                        "binding_sha256": side["binding_sha256"], "scope": side["scope"],
                        "audits_present_18241_36480": 36480 - 18240, "endpoint": "e120 EMA (endpoint lock)"}}


def history(seed):
    formal = SEEDS[seed]["formal"]
    return {"formal_binding": ref(formal / "binding.json"), "audits_dir": str(formal / "audits"),
            "loader_check_audits": {str(u): ref(formal / f"audits/update-{u:05d}-direct.json")
                                    for u in LOADER_CHECK_UPDATES},
            "input_plans_dir": str(formal)}


def seed0_evidence():
    return {k: pinned(path, sha) for k, (path, sha) in SEED0.items()}


def identity_for(seed, manifest, task, parent, data, evaluation_policy, reference):
    sources = {row["path"]: row["sha256"] for row in manifest["source_files"]}
    return {"task": task, "seed": seed, "parent": parent, "data": data, "evaluation_policy": evaluation_policy,
            "native_computation": {name: sources[name] for name in MODEL_FILES},
            "s15_canvas_seed_and_memory_implementation": {name: sources[name] for name in S15_FILES},
            "workload": WORKLOAD, "decision_rule": DECISION_RULE, "A0_reference": reference}


# ---------------------------------------------------------------- contract
def contract_name(attempt):
    return ROUND / (suffix(attempt, "formal") + "-CONTRACT.json")


def contract(release_name, attempt, per_class_name, prior_name):
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    release, manifest_ref, manifest = release_manifest(release_name)
    data_ref = pinned(DATA_REF, DATA_REF_SHA)
    old = read(data_ref["path"])
    per_class = ref(ROUND / per_class_name)
    per_class_body = read(per_class["path"])
    require(per_class_body["status"] == "PASS" and per_class_body["COCOeval_calls"] == 2
            and per_class_body["data_receipt"] == data_ref, "A0 per-class extraction receipt")
    seeds = {}
    for seed in SEED_ORDER:
        parent = pinned(SEEDS[seed]["formal"] / "checkpoint-epoch-060.pth", SEEDS[seed]["parent_sha"], PARENT_BYTES)
        reference = {**a0_reference_refs(seed), "A0_per_class": per_class}
        require(per_class_body["seeds"][str(seed)]["metrics"] == reference["A0_metrics"]["600"]
                and per_class_body["seeds"][str(seed)]["prediction"] == reference["A0_predictions"]["600"],
                f"seed{seed} A0 per-class binding")
        scientific = identity_for(seed, manifest, task, parent, old["data"], old["evaluation_policy"], reference)
        root = ROUND / attempt / f"seed{seed}"
        seeds[str(seed)] = {"seed": seed, "parent": parent, "reference": reference, "history": history(seed),
                            "scientific_binding": scientific, "formal_identity": canonical_sha(scientific),
                            "outputs": {"root": str(root), "accept": str(root / "accept"), "train": str(root / "train-S15"),
                                        "evaluation": str(root / "evaluation")}}
    path = contract_name(attempt)
    prior = None
    if prior_name:
        require(prior_name != attempt, "cannot continue the same attempt")
        old_contract = load_contract(contract_name(prior_name))
        require(all(old_contract["seeds"][s]["formal_identity"] == seeds[s]["formal_identity"] for s in seeds),
                "prior science identity differs")
        prior_path = Path(old_contract["attempt_root"])
        require((prior_path / "STOP.json").is_file() and not (prior_path / "CLOSED.json").exists(),
                "continuation requires a stopped, nonclosed prior attempt")
        require(old_contract["source_manifest"]["sha256"] != manifest_ref["sha256"],
                "continuation requires the specific repaired release")
        prior = {"contract": ref(contract_name(prior_name)), "stop": ref(prior_path / "STOP.json")}
    value = {
        "version": CONTRACT_VERSION, "attempt": attempt, "contract_path": str(path),
        "attempt_root": str(ROUND / attempt), "prior": prior, "task": task, "seed_order": SEED_ORDER, "seeds": seeds,
        "workload": WORKLOAD, "decision_rule": DECISION_RULE, "resource_policy": RESOURCE,
        "reused_evidence": seed0_evidence(), "source_root": str(release / "source"), "source_manifest": manifest_ref,
        "source_commit": manifest["commit"], "data_receipt": data_ref, "data": old["data"],
        "development_ids": old["development_ids"], "evaluation_policy": old["evaluation_policy"],
        "environment": old["environment"], "python": PYTHON, "gpu_uuid": GPU,
    }
    value["contract_sha256"] = canonical_sha(value)
    write_new(path, value)
    print(json.dumps({"file": ref(path), "contract_sha256": value["contract_sha256"],
                      "formal_identities": {s: seeds[s]["formal_identity"] for s in seeds}}), flush=True)


def load_contract(path, *, check_source=True):
    path = Path(path).absolute()
    contract = read(path)
    binding = dict(contract)
    supplied = binding.pop("contract_sha256", None)
    require(supplied == canonical_sha(binding), "canonical contract digest differs")
    require(contract["version"] == CONTRACT_VERSION and contract["workload"] == WORKLOAD
            and contract["decision_rule"] == DECISION_RULE and contract["seed_order"] == SEED_ORDER
            and sorted(contract["seeds"]) == [str(s) for s in SEED_ORDER], "S15 seed12 workload or decision rule differs")
    for key, section in contract["seeds"].items():
        seed = int(key)
        require(section["seed"] == seed and section["formal_identity"] == canonical_sha(section["scientific_binding"])
                and section["scientific_binding"]["seed"] == seed, f"seed{seed} formal identity differs")
        require(section["parent"]["path"] == str(SEEDS[seed]["formal"] / "checkpoint-epoch-060.pth")
                and section["parent"]["sha256"] == SEEDS[seed]["parent_sha"], f"seed{seed} parent path/digest differs")
        require(section["outputs"]["root"] == str(ROUND / contract["attempt"] / f"seed{seed}"), "seed output root differs")
    require(contract["source_root"] == str(Path(contract["source_manifest"]["path"]).parent / "source"),
            "source manifest/root binding differs")
    require(contract["contract_path"] == str(path), "contract relocated")
    require(contract["attempt_root"] == str(ROUND / suffix(contract["attempt"], "formal")), "attempt path differs")
    require(contract["resource_policy"] == RESOURCE and path == contract_name(contract["attempt"]),
            "resource policy or actual contract path differs")
    if check_source:
        manifest = read(checked_ref(contract["source_manifest"]))
        require(manifest["version"] == RELEASE_VERSION and manifest["commit"] == contract["source_commit"]
                and manifest["task"] == contract["task"], "contract release identity differs")
    return contract


def seed_view(contract, seed):
    """Flat view of one seed: common contract fields plus that seed's parent/reference/identity/outputs."""
    require(str(seed) in contract["seeds"], f"seed {seed} not in contract")
    view = {k: v for k, v in contract.items() if k != "seeds"}
    view.update(contract["seeds"][str(seed)])
    return view


def validate_materials(c):
    started = time.monotonic()
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    require(c["task"] == task, "task changed")
    data_receipt = pinned(DATA_REF, DATA_REF_SHA)
    old = read(data_receipt["path"])
    require(c["data_receipt"] == data_receipt and c["data"] == old["data"]
            and c["development_ids"] == old["development_ids"] and c["evaluation_policy"] == old["evaluation_policy"]
            and c["environment"] == old["environment"] and c["python"] == PYTHON and c["gpu_uuid"] == GPU,
            "admitted materials or environment changed")
    checked_ref(c["evaluation_policy"])
    require(c["reused_evidence"] == seed0_evidence(), "seed0 evidence changed")
    manifest = read(checked_ref(c["source_manifest"]))
    require(manifest["version"] == RELEASE_VERSION and manifest["commit"] == c["source_commit"]
            and manifest["task"] == task, "release binding differs")
    verify_source(Path(c["source_root"]), manifest)
    for seed in SEED_ORDER:
        s = c["seeds"][str(seed)]
        parent = pinned(SEEDS[seed]["formal"] / "checkpoint-epoch-060.pth", SEEDS[seed]["parent_sha"], PARENT_BYTES)
        require(s["parent"] == parent, f"seed{seed} parent changed")
        checked_ref(s["reference"]["A0_per_class"])
        require({**a0_reference_refs(seed), "A0_per_class": s["reference"]["A0_per_class"]} == s["reference"],
                f"seed{seed} A0 reference materials changed")
        require(s["history"] == history(seed), f"seed{seed} historical audit references changed")
        require(s["scientific_binding"] == identity_for(seed, manifest, task, parent, c["data"], c["evaluation_policy"],
                                                        s["reference"]), f"seed{seed} science projection differs")
    source = Path(c["source_root"])
    sys.path.insert(0, str(source / "src"))
    from sparse_rtdetr.localization.material_paths import MANIFESTS, metadata_path
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    counts, data_bytes = {}, 0
    for role, (digest, count, split) in MANIFESTS.items():
        entry = c["data"][role]
        path = metadata_path(role, entry["manifest"], digest)
        rows = read(pinned(path, digest)["path"])["records"]
        ids = [row["stable_image_id"] for row in rows]
        require(len(ids) == len(set(ids)) == count, f"{role} member count or uniqueness mismatch")
        if role == "development":
            require(ids == c["development_ids"], "development order differs")
        roots = entry["roots"]
        require(roots["role"] == role, "data root role differs")
        for kind in ("images", "annotations"):
            directory = Path(roots[kind])
            require(directory.is_dir() and directory.resolve() == directory and directory.name == kind
                    and directory.parent.name == role and not any(p.is_symlink() for p in (directory, *directory.parents)),
                    "data root differs from staged role directory")
        for row in rows:
            require(row["split"] == split and row["stable_image_id"] == stable_image_id(split, row["relative_path"]),
                    "stable data identity mismatch")
            for kind, key, field in (("images", "relative_path", "image"),
                                     ("annotations", "annotation_relative_path", "annotation")):
                relative = Path(row[key])
                require(len(relative.parts) == 3 and relative.parts[:2] == (split, kind)
                        and relative.name not in {".", ".."}, "data path is not a bound role member")
                bound = pinned(Path(roots[kind]) / relative.name, row[field + "_sha256"], row[field + "_size_bytes"])
                data_bytes += bound["bytes"]
        counts[role] = count
    return {"source_files": len(manifest["source_files"]), "verified_role_counts": counts,
            "verified_data_bytes": data_bytes, "seconds": time.monotonic() - started}


def execution_environment(device, tmp):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES=device, CUBLAS_WORKSPACE_CONFIG=":4096:8", MKL_THREADING_LAYER="GNU",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF=ALLOCATOR)
    if tmp:
        env.update(TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp))
    return env


def hidden_environment(c):
    code = """
import importlib.metadata as md, json, sys, torch, torchvision
print(json.dumps({'torch': torch.__version__.split('+')[0], 'torchvision': torchvision.__version__.split('+')[0],
 'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'python': sys.executable,
 'pycocotools': md.version('pycocotools'), 'cuda_available': torch.cuda.is_available()}))
"""
    observed = json.loads(subprocess.check_output([c["python"], "-B", "-c", code], cwd=c["source_root"],
                                                  env=execution_environment("", ""), text=True))
    require(all(observed[key] == value for key, value in c["environment"].items()) and observed["python"] == c["python"]
            and not observed["cuda_available"], f"runtime environment mismatch: {observed}")
    return observed


def command_output(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=15).strip()


def protect_disk(tmp=None, estimate=True):
    free = {"round": shutil.disk_usage(ROUND).free, "temporary": shutil.disk_usage(tmp or "/tmp").free,
            "shared_memory": shutil.disk_usage("/dev/shm").free}
    reserve = RESOURCE["reserve_disk_bytes"]
    required = reserve + (RESOURCE["remaining_save_estimate_bytes"] if estimate else 0)
    require(free["round"] > required, f"insufficient storage for safe next-stage write: {free}")
    require(free["temporary"] > reserve and free["shared_memory"] > 2 * 1024**3,
            f"insufficient temporary or DataLoader shared memory: {free}")
    return free


def resources():
    devices = command_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free,driver_version",
                              "--format=csv,noheader,nounits"])
    target = [line for line in devices.splitlines() if line.split(",")[0].strip() == GPU]
    require(len(target) == 1, "bound GPU not present")
    # Admission only: S15 needs essentially the whole device.  No allocator fraction or VRAM cap is set.
    require(int(target[0].split(",")[3].strip()) > 23000, f"bound GPU is not free: {target[0]}")
    apps = command_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                           "--format=csv,noheader,nounits"])
    require(not any(GPU in line for line in apps.splitlines()), f"bound GPU already in use: {apps}")
    return {"gpu": target[0], "compute_apps": apps, "free_bytes": protect_disk(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def preflight(path):
    c = load_contract(path)
    return {"status": "PASS", "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
            "materials": validate_materials(c), "environment": hidden_environment(c)}


def _cpu_job(argv, log_path, source_root):
    env = execution_environment("", "")
    env["S15_CPU_THREADS"] = "8"
    started = time.monotonic()
    with Path(log_path).open("x") as log:
        code = subprocess.call(["nice", "-n", "19", *argv], cwd=source_root, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT)
    return code, time.monotonic() - started


def a0_per_class(release_name, name):
    release, manifest_ref, _ = release_manifest(release_name)
    out = ROUND / name
    require(re.fullmatch(r"A0_PER_CLASS-[0-9]{3}\.json", name) and not out.exists(), "new A0 per-class output name")
    source = release / "source"
    argv = [PYTHON, "-B", str(source / "tools/scale_baseline/s15_formal_eval.py"), "a0-per-class", "--out", str(out)]
    code, wall = _cpu_job(argv, ROUND / (out.stem + ".log"), source)
    print(json.dumps({"returncode": code, "wall_seconds": wall, "release_manifest": manifest_ref,
                      "out": str(out)}), flush=True)
    require(code == 0, "A0 per-class extraction failed")


def cpu_gate(path, kind, seed, out):
    """Run one CUDA-hidden CPU gate for one seed from the frozen release (lowest priority)."""
    c = load_contract(path)
    require(seed in SEED_ORDER, "seed not in contract")
    out = Path(out).absolute()
    require(out.parent == ROUND and not out.exists(), "CPU gate output must be a new directory in the round")
    program, action = {"train": ("s15_formal_train.py", "cpu"), "eval": ("s15_formal_eval.py", "cpu-check")}[kind]
    argv = [c["python"], "-B", str(Path(c["source_root"]) / "tools/scale_baseline" / program),
            action, "--contract", c["contract_path"], "--seed", str(seed), "--out", str(out)]
    code, wall = _cpu_job(argv, ROUND / (out.name + ".log"), c["source_root"])
    print(json.dumps({"kind": kind, "seed": seed, "returncode": code, "wall_seconds": wall, "out": str(out)}), flush=True)
    require(code == 0, f"CPU gate {kind} seed{seed} failed; see {out}")


def check_gate(path, c):
    reference = ref(path)
    gate = read(path)
    require(gate["status"] == "PASS" and gate["contract_sha256"] == c["contract_sha256"]
            and gate["seed"] in SEED_ORDER and (gate.get("action") == "cpu" or gate.get("mode") == "eval"),
            "CPU gate identity or status mismatch")
    return ("train" if gate.get("action") == "cpu" else "eval", gate["seed"]), reference


def process_alive(pid):
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def latest_active():
    for directory in sorted(ROUND.glob("formal-[0-9][0-9][0-9]")):
        if not directory.is_dir() or (directory / "STOP.json").is_file() or (directory / "CLOSED.json").is_file():
            continue
        launch = directory / "LAUNCH.json"
        state = ("active" if process_alive(read(launch)["pid"]) else "uncertain terminal") if launch.is_file() \
            else "missing launch receipt"
        raise RuntimeError(f"existing unclosed formal attempt ({state}): {directory}")


def lock_round():
    handle = (ROUND / ".s15-seed12.lock").open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("S15 seed12 run lock is held")
    return handle


def launch(path, gate_paths):
    c = load_contract(path)
    attempt = Path(c["attempt_root"])
    with lock_round() as handle:
        latest_active()
        if not c["prior"]:
            require(not list(ROUND.glob("formal-[0-9][0-9][0-9]")),
                    "existing formal attempt; continuation requires its STOP and a new release")
        require(not attempt.exists(), "duplicate attempt")
        gates = dict(check_gate(p, c) for p in gate_paths)
        require(set(gates) == {(k, s) for k in ("train", "eval") for s in SEED_ORDER}, f"CPU gates incomplete: {sorted(gates)}")
        verified = preflight(path)
        current = resources()
        attempt.mkdir()
        tmp = Path(tempfile.mkdtemp(prefix="s15s12-", dir="/tmp"))
        request = attempt / "REQUEST.json"
        write_new(request, {"contract": ref(path), "gates": {f"{k}-seed{s}": v for (k, s), v in sorted(gates.items())},
                            "preflight": verified, "resources": current,
                            "temporary_directory": str(tmp), "prior": c["prior"], "epoch_seconds": time.time()})
        argv = [c["python"], "-B", str(Path(c["source_root"]) / "tools/scale_baseline/s15_formal_run.py"),
                "_supervise", "--contract", c["contract_path"], "--lock-fd", str(handle.fileno()),
                "--request-sha", file_sha(request)]
        try:
            with (attempt / "supervisor.log").open("x") as log:
                child = subprocess.Popen(argv, cwd=c["source_root"], start_new_session=True, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, env=execution_environment(GPU, tmp),
                                         pass_fds=(handle.fileno(),))
        except BaseException as exc:
            write_new(attempt / "STOP.json", {"status": "STOP", "stage": "supervisor-spawn", "error": repr(exc),
                                              "contract_sha256": c["contract_sha256"]})
            raise
        receipt = {"status": "STARTED", "pid": child.pid, "argv": argv, "cwd": c["source_root"],
                   "contract_sha256": c["contract_sha256"], "request": ref(request), "epoch_seconds": time.time()}
        write_new(attempt / "LAUNCH.json", receipt)
        print(json.dumps(receipt), flush=True)


# ---------------------------------------------------------------- stages
def close_stage(stage, c, action, seed):
    path = stage / "CLOSED.json"
    row = read(path)
    require(row["status"] == "PASS" and row["contract_sha256"] == c["contract_sha256"] and row["seed"] == seed
            and (row.get("action") == action or (action == "evaluate" and row.get("mode") == "eval")),
            f"stage closure differs: {path}")
    return row


def endpoint_receipt(directory, view, epoch):
    path = directory / f"checkpoint-epoch-{epoch:03d}.pth"
    side = read(path.with_suffix(".pth.json"))
    require((side["status"], side["scope"], side["arm"], side["seed"], side["formal_identity"], side["contract_sha256"],
             side["global_update"]) == ("COMPLETE", "formal", "S15", view["seed"], view["formal_identity"],
                                        view["contract_sha256"], epoch * 304)
            and side["file"]["path"] == str(path), "checkpoint binding differs")
    checked_ref(side["file"])
    return path


def carry_from(c):
    """Per seed: accepted stages of stopped prior attempts with the same scientific identities."""
    empty = {seed: (None, None, None) for seed in SEED_ORDER}
    if not c["prior"]:
        return empty
    prior_path = checked_ref(c["prior"]["contract"])
    previous = load_contract(prior_path)
    stop = read(checked_ref(c["prior"]["stop"]))
    require(stop["contract_sha256"] == previous["contract_sha256"], "prior STOP binding differs")
    inherited = carry_from(previous)
    for seed in SEED_ORDER:
        view = seed_view(previous, seed)
        require(view["formal_identity"] == c["seeds"][str(seed)]["formal_identity"], "prior seed identity differs")
        accept, completed, resume = inherited[seed]
        out = view["outputs"]
        if (Path(out["accept"]) / "CLOSED.json").exists():
            close_stage(Path(out["accept"]), previous, "accept", seed)
            accept = ref(Path(out["accept"]) / "CLOSED.json")
        directory = Path(out["train"])
        if (directory / "CLOSED.json").exists():
            closed = close_stage(directory, previous, "train", seed)
            require(closed["global_update"] == WORKLOAD["stop_update"], "prior completed S15 identity differs")
            endpoint_receipt(directory, view, 120)
            completed, resume = {"path": str(directory), "contract": str(prior_path), "seed": seed}, None
            require(not list(Path(out["evaluation"]).glob("*-metrics.json")),
                    "prior evaluation already scored; continuation must account those score calls")
        elif directory.exists() and any(directory.glob("checkpoint-epoch-*.pth.json")):
            eligible = [ep for ep in WORKLOAD["save_epochs"] if (directory / f"checkpoint-epoch-{ep:03d}.pth.json").exists()]
            resume = {"checkpoint": str(endpoint_receipt(directory, view, max(eligible))), "contract": str(prior_path)}
        elif directory.exists() and (directory / "UPDATES.jsonl").exists():
            require((directory / "UPDATES.jsonl").stat().st_size == 0, "S15 had updates without a recoverable formal state")
        inherited[seed] = (accept, completed, resume)
    return inherited


def input_alignment(view, train_root):
    """S15 order/draws equal the seed's own historical native audit row by row; canvas-dependent
    GT/target/pixel identities are counted, not required equal."""
    sys.path[:0] = [str(Path(view["source_root"]) / "src")]
    from sparse_rtdetr.localization.runio import digest
    from sparse_rtdetr.baseline.training_v2b_data import _sample_seed
    seed = view["seed"]
    audits = Path(view["history"]["audits_dir"])
    differ_fields = ("gt_ids_sha256", "augmentation_trace_sha256", "augmented_samples_sha256", "input_tensor_sha256",
                     "target_sha256")
    start = WORKLOAD["initial_update"]
    counts = {k: 0 for k in differ_fields}
    digest_all, gt, n = hashlib.sha256(), 0, 0
    started = time.monotonic()
    with (Path(train_root["path"]) / "UPDATES.jsonl").open() as mine:
        for line in mine:
            row = json.loads(line)["input"]
            n += 1
            update = start + n
            hist = read(audits / f"update-{update:05d}-direct.json")
            require(hist["seed"] == seed and hist["update"] == update and row["update"] == update, "audit cursor mismatch")
            h, result = hist["result"]["input"], hist["result"]
            epoch, window = result["epoch"], result["window"]
            seeds = [_sample_seed(seed, epoch, window * 16 + i, image_id) for i, image_id in enumerate(h["ids"])]
            expected = {"update": update, "epoch": epoch, "window": window, "image_ids_sha256": digest(h["ids"]),
                        "augmentation_seeds_sha256": digest(seeds), "augmentation_strong": epoch < 118}
            require(all(row[k] == v for k, v in expected.items()), f"S15 order/draw differs from native seed{seed} at {update}")
            other = {"gt_ids_sha256": digest(h["GT_ids"]), "augmentation_trace_sha256": h["metadata_sha256"],
                     "augmented_samples_sha256": digest(h["augmented_sha256"]),
                     "input_tensor_sha256": h["input_tensor_sha256"], "target_sha256": h["target_sha256"]}
            for k in differ_fields:
                counts[k] += row[k] != other[k]
            gt += json.loads(line)["gt_count"]
            digest_all.update(bytes.fromhex(canonical_sha(row)))
    require(n == WORKLOAD["formal_updates_per_seed"], "S15 input audit length differs")
    return {"seed": seed, "updates": n,
            "order_and_draws_equal_native_seed_audit": ["update", "epoch", "window", "image_ids_sha256",
                                                        "augmentation_seeds_sha256", "augmentation_strong"],
            "augmentation_seeds": "recomputed from the historical row's image ids and positions with the native "
                                  "v2b sample-seed rule; the 768 loader check matched actual historical traces",
            "rows_with_canvas_dependent_difference": counts, "S15_positive_GT_total_after_sanitize": gt,
            "input_sha256": digest_all.hexdigest(), "full_audit": ref(Path(train_root["path"]) / "UPDATES.jsonl"),
            "native_audit_dir": str(audits), "seconds": time.monotonic() - started}


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_child(child):
    if group_alive(child.pid):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    if group_alive(child.pid):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=10)


def progress_time(path, log):
    files = [log, path / "UPDATES.jsonl", path / "progress.jsonl"]
    files.extend(path.glob("*/progress.jsonl"))
    return max(p.stat().st_mtime_ns for p in files if p.exists())


def run_stage(c, attempt, tmp, seed, stage, program, action, extra, ledger):
    protect_disk(tmp)
    view = seed_view(c, seed)
    directory = Path(view["outputs"][stage])
    label = f"seed{seed}-{stage}"
    log = attempt / f"{label}.log"
    argv = [c["python"], "-B", str(ROOT / "tools/scale_baseline" / program), action,
            "--contract", c["contract_path"], "--seed", str(seed), "--out", str(directory), *extra]
    began = time.monotonic()
    row = {"stage": label, "seed": seed, "started_epoch_seconds": time.time(), "argv": argv}
    ledger.append(row)
    journal(attempt / "STAGES.jsonl", {"event": "START", **row})
    child = None
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=execution_environment(GPU, tmp), start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        stamp = progress_time(directory, log)
        last = time.monotonic()
        while child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                protect_disk(tmp, estimate=False)
                updated = progress_time(directory, log)
                if updated != stamp:
                    stamp, last = updated, time.monotonic()
                require(time.monotonic() - last < RESOURCE["no_progress_seconds"], f"no progress in {label}")
        require(child.returncode == 0, f"{label} subprocess exited {child.returncode}; see {log}")
        deadline = time.monotonic() + 5
        while group_alive(child.pid) and time.monotonic() < deadline:
            time.sleep(.1)
        require(not group_alive(child.pid), f"{label} left a live worker process")
        result = close_stage(directory, c, action, seed)
        row.update({"status": "PASS", "result": ref(directory / "CLOSED.json")})
        return result
    except BaseException:
        row["status"] = "STOP"
        if (directory / "STOP.json").is_file():
            row["result"] = ref(directory / "STOP.json")
        if child is not None:
            stop_child(child)
        raise
    finally:
        row["seconds"] = time.monotonic() - began
        row["returncode"] = child.returncode if child is not None else None
        journal(attempt / "STAGES.jsonl", {"event": "FINISH", **row})


def byte_usage(path):
    return int(command_output(["du", "-sb", str(path)]).split()[0]) if Path(path).exists() else 0


def measurements(tmp):
    paths = {"experiment": ROUND, "temporary": tmp, "torch_cache": Path.home() / ".cache/torch",
             "torch_extensions": Path.home() / ".cache/torch_extensions", "triton_cache": Path.home() / ".triton"}
    result = {}
    for kind, path in paths.items():
        try:
            result[kind] = byte_usage(path)
        except (ValueError, OSError, subprocess.SubprocessError):
            result[kind] = None
    return result


# ---------------------------------------------------------------- fixed three-seed summary
CELLS = ("top600_APS", "top600_AP75_small", "top600_AP", "top300_AP", "top300_APS", "top600_APM", "top600_APL")


def summary(c, comparisons):
    """Seed0/1/2 cell differences with mean, sample SD (n-1), min, max; per-seed verdicts unchanged."""
    seed0 = read(checked_ref(c["reused_evidence"]["comparison"]))
    cells0 = seed0["EMA_gates"]["cells"]
    per_seed = {"0": {"decision": seed0["scientific_decision"], "cells": cells0,
                      "per_class_APL_top600": seed0["per_class_APL_top600"]["EMA"]}}
    for seed, comparison in comparisons.items():
        per_seed[str(seed)] = {"decision": comparison["scientific_decision"], "cells": comparison["EMA_gates"]["cells"],
                               "per_class_APL_top600": comparison["per_class_APL_top600"]["EMA"]}
    aggregate = {}
    for cell in CELLS:
        values = [per_seed[s]["cells"][cell]["difference_pp"] for s in ("0", "1", "2")]
        mean = sum(values) / 3
        aggregate[cell] = {"values_pp_seed0_1_2": values, "mean_pp": mean,
                           "sample_sd_pp_n_minus_1": math.sqrt(sum((v - mean) ** 2 for v in values) / 2),
                           "min_pp": min(values), "max_pp": max(values),
                           "minimum_pp": cells0[cell]["minimum_pp"],
                           "seeds_failing": [s for s in ("0", "1", "2") if not per_seed[s]["cells"][cell]["passed"]]}
    new = [str(s) for s in SEED_ORDER]
    primary_new = all(per_seed[s]["cells"][k]["passed"] for s in new for k in ("top600_APS", "top600_AP75_small"))
    apl_fail = aggregate["top600_APL"]["seeds_failing"]
    branches = []
    if primary_new:
        branches.append("1")
    else:
        branches.append("2")
    if len(apl_fail) == 3:
        branches.append("3-repeated")
    elif apl_fail:
        branches.append("3-inconsistent")
    return {"status": "PASS", "per_seed": per_seed, "aggregate_descriptive_only": aggregate,
            "primary_both_new_seeds_passed": primary_new,
            "protection_failures": {s: [k for k, v in per_seed[s]["cells"].items() if not v["passed"]] for s in per_seed},
            "APL_failing_seeds": apl_fail, "preregistered_branches": branches,
            "notes": ["sample SD over 3 seeds is not a standard error or confidence interval",
                      "the mean never overrides a seed FAIL",
                      "same 548-image development set, not independent data validation",
                      "seed0 A0 reference is R25 A0 (native seed0 e060->e120 rerun); seed1/2 use their historical A0"]}


def _supervise(path, lock_fd, request_sha):
    c = load_contract(path)
    attempt = Path(c["attempt_root"])
    require(ROOT == Path(c["source_root"]), "supervisor did not run from frozen source")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == ALLOCATOR,
            "wrong visible GPU or allocator")
    os.fstat(lock_fd)
    req_path = attempt / "REQUEST.json"
    require(file_sha(req_path) == request_sha, "launch request bytes differ")
    req = read(req_path)
    checked_ref(req["contract"])
    for gate in req["gates"].values():
        checked_ref(gate)
    tmp = Path(req["temporary_directory"])
    require(os.environ.get("TMPDIR") == str(tmp), "short temporary path is not active")
    receipt = attempt / "LAUNCH.json"
    deadline = time.monotonic() + 10
    while not receipt.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    require(read(receipt)["pid"] == os.getpid(), "attempt launcher PID differs")
    started = time.monotonic()
    ledger, stage = [], "initialize"
    before = measurements(tmp)

    def interrupt(signum, _):
        raise RuntimeError(f"supervisor signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    result = {"contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
              "formal_identities": {s: c["seeds"][s]["formal_identity"] for s in c["seeds"]}}
    comparisons, decisions = {}, {}
    try:
        inherited = carry_from(c)
        write_new(attempt / "INHERITED.json", {"seeds": {str(s): {"accept": a, "train_root": t, "resume": r}
                                                         for s, (a, t, r) in inherited.items()}, "prior": c["prior"]})
        for seed in SEED_ORDER:
            Path(seed_view(c, seed)["outputs"]["root"]).mkdir(exist_ok=True)
        for seed in SEED_ORDER:  # both seeds' acceptance before any formal update
            if inherited[seed][0] is None:
                stage = f"seed{seed}-accept"
                row = run_stage(c, attempt, tmp, seed, "accept", "s15_formal_train.py", "accept", [], ledger)
                acc = WORKLOAD["acceptance"]
                require(row["totals"]["optimizer_updates"] == acc["S15_updates_per_seed"]
                        and row["totals"]["full_model_forwards"] == acc["S15_training_forwards_per_seed"]
                        and row["S15"]["replay_complete_state_and_record_exact"] is True
                        and row["formal_state_usable"] is False, "acceptance work count or status differs")
        for seed in SEED_ORDER:
            view = seed_view(c, seed)
            _, completed, resume = inherited[seed]
            if completed is None:
                stage = f"seed{seed}-train-S15"
                args = ["--resume", resume["checkpoint"], "--prior-contract", resume["contract"]] if resume else []
                row = run_stage(c, attempt, tmp, seed, "train", "s15_formal_train.py", "train", args, ledger)
                require(row["arm"] == "S15" and row["effective_updates"] == WORKLOAD["formal_updates_per_seed"]
                        and row["global_update"] == WORKLOAD["stop_update"], "formal training terminal count differs")
                completed = {"path": view["outputs"]["train"], "contract": str(path), "seed": seed}
            stage = f"seed{seed}-input-alignment"
            root = Path(view["outputs"]["root"])
            write_new(root / "TRAIN_ROOT.json", completed)
            write_new(root / "INPUT_ALIGNMENT.json", input_alignment(view, completed))
            stage = f"seed{seed}-evaluation"
            row = run_stage(c, attempt, tmp, seed, "evaluation", "s15_formal_eval.py", "evaluate",
                            ["--train-root", str(root / "TRAIN_ROOT.json")], ledger)
            comparison = read(checked_ref(row["result"]))
            require(comparison["score_calls"] == WORKLOAD["evaluation"]["score_calls_per_seed"]
                    and comparison["seed"] == seed, "formal evaluation score count differs")
            comparisons[seed] = comparison
            decisions[str(seed)] = comparison["scientific_decision"]
        stage = "summary"
        write_new(attempt / "SEED_SUMMARY.json", summary(c, comparisons))
        result.update(status="PASS", scientific_decisions=decisions, summary=ref(attempt / "SEED_SUMMARY.json"))
    except BaseException as exc:
        result.update(status="STOP", stage=stage, error=repr(exc), scientific_decisions_so_far=decisions)
        raise
    finally:
        storage = measurements(tmp)
        result.update(wall_seconds=time.monotonic() - started, stages=ledger, storage_before_bytes=before,
                      storage_after_bytes=storage, prior=c["prior"],
                      storage_delta_bytes={k: storage[k] - before[k] if storage[k] is not None and before[k] is not None
                                           else None for k in storage})
        write_new(attempt / ("CLOSED.json" if result["status"] == "PASS" else "STOP.json"), result)


def supervise(path, lock_fd, request_sha):
    try:
        _supervise(path, lock_fd, request_sha)
    except BaseException as exc:
        c = load_contract(path, check_source=False)
        attempt = Path(c["attempt_root"])
        if not (attempt / "STOP.json").exists() and not (attempt / "CLOSED.json").exists():
            write_new(attempt / "STOP.json", {"status": "STOP", "stage": "supervisor-initialization",
                                              "contract_sha256": c["contract_sha256"], "error": repr(exc)})
        raise


def status(path):
    c = load_contract(path)
    attempt = Path(c["attempt_root"])
    for name in ("CLOSED.json", "STOP.json"):
        if (attempt / name).is_file():
            return read(attempt / name)
    if not attempt.exists():
        return {"status": "NOT_STARTED"}
    if not (attempt / "LAUNCH.json").is_file():
        return {"status": "STARTUP_UNCERTAIN"}
    launch_receipt = read(attempt / "LAUNCH.json")
    return {"status": "RUNNING" if process_alive(launch_receipt["pid"]) else "TERMINAL_UNKNOWN", "launch": launch_receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("freeze")
    prep.add_argument("--release", required=True)
    per = actions.add_parser("a0-per-class")
    per.add_argument("--release", required=True)
    per.add_argument("--name", required=True)
    con = actions.add_parser("contract")
    con.add_argument("--release", required=True)
    con.add_argument("--attempt", required=True)
    con.add_argument("--a0-per-class", required=True)
    con.add_argument("--prior")
    gate = actions.add_parser("cpu-gate")
    gate.add_argument("--contract", type=Path, required=True)
    gate.add_argument("--kind", choices=("train", "eval"), required=True)
    gate.add_argument("--seed", type=int, required=True)
    gate.add_argument("--out", type=Path, required=True)
    for name in ("preflight", "launch", "status", "_supervise"):
        call = actions.add_parser(name)
        call.add_argument("--contract", type=Path, required=True)
        if name == "launch":
            call.add_argument("--gates", type=Path, nargs=4, required=True)
        elif name == "_supervise":
            call.add_argument("--lock-fd", type=int, required=True)
            call.add_argument("--request-sha", required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.release)
    elif args.action == "a0-per-class":
        a0_per_class(args.release, args.name)
    elif args.action == "contract":
        contract(args.release, args.attempt, args.a0_per_class, args.prior)
    elif args.action == "cpu-gate":
        cpu_gate(args.contract, args.kind, args.seed, args.out)
    elif args.action == "preflight":
        print(json.dumps(preflight(args.contract)), flush=True)
    elif args.action == "launch":
        launch(args.contract.absolute(), args.gates)
    elif args.action == "status":
        print(json.dumps(status(args.contract)), flush=True)
    else:
        supervise(args.contract, args.lock_fd, args.request_sha)


if __name__ == "__main__":
    main()
