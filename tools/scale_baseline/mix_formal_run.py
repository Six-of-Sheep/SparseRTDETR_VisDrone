"""TASK_MIX_SCALE_001: immutable preparation and detached, journaled single-arm MIX run (seed0).

  table         --release release-NNN        write the scale table from the frozen release's rule (once)
  freeze        --release release-NNN        frozen git-archive release + manifest
  contract      --release release-NNN --attempt formal-NNN [--prior formal-MMM]
  cpu-gate      --contract C --kind train|eval --out DIR      CUDA-hidden gate from the frozen release
  preflight / launch --gates G1 G2 / status --contract C

Supervisor order: accept -> train-MIX -> input-alignment -> infer MIX@H -> infer MIX@L -> score (CPU).
Native A0@L, S15@H and S15@L are pinned references; they are never re-inferred, re-scored or retrained.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
EXPERIMENTS = W / "experiments/localization-distribution"
ROUND = EXPERIMENTS / "mix-scale-baseline-20261108-001"
R25 = EXPERIMENTS / "r25-query-local-regression-20261104-001"
S15R = EXPERIMENTS / "s15-scale-baseline-20261105-001"
S12R = EXPERIMENTS / "s15-seed12-replication-20261106-001"
CROSS = EXPERIMENTS / "s15-cross-scale-diagnostic-20261107-001"
PYTHON = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
GPU = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
TASK = ("TASK_MIX_SCALE_001.md", "7b9acffb48a55c6967dbff0e2a4bc80c194db02f338c056aed9d873fdf40ac77")
REVIEW = ("REVIEW_AND_DIRECTION_001.md", "700d5357553e9d9e52dd4255266f950ed2134948465a0a5cbc04a7d7b2bbb65f")
DATA_REF = EXPERIMENTS / "r18-tiny-positives-20261102-001/production-contract-002.json"
DATA_REF_SHA = "b0fe33198d8abc7f8b89a1cc9824de9abdba8721580932aa07be565ce40d896b"
PARENT = (EXPERIMENTS / "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal"
          "/checkpoint-epoch-060.pth", "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a", 323460289)
TABLE_NAME = "MIX_LH_V1_TABLE.tsv"
R25_RELEASE_MANIFEST = (R25 / "release-002/MANIFEST.json", "31e6fb65ba0ef883b03afdeae459def5a25c528ef3264007d1501b21dc0ddb4c")
SEED12_RELEASE_MANIFEST = (S12R / "release-001/MANIFEST.json",
                           "87a4c6d029fd0f19ed20dcf9cfe94e5853098a551a16a5d7ca1687fde56f0864")
# Historical same-canvas ledgers: R25 A0 (768) and S15 (1152), each the full audit bound by its e120 sidecar.
LEDGERS = {"A0_updates": (R25 / "formal-001/train-A0/checkpoint-epoch-120.pth.json",
                          "78c3e896a11de260f2a09e4a860457fbd868eeec81ce1f1632efdfc0ddd8d966"),
           "S15_updates": (S15R / "formal-002/train-S15/checkpoint-epoch-120.pth.json",
                           "656f7be2357f20dce50a718f1093a639c1a32a7faad69d03184e36e43f32871f")}
REFERENCE_FILES = {
    "S15_comparison": (S15R / "formal-002/evaluation/COMPARISON.json",
                       "78b486390b60869533ece6f1160a744c741ec2eac1644a61c07f6c2ed94a6759"),
    "four_cell": (CROSS / "diag-003/summary/FOUR_CELL.json",
                  "375a3ee6cb908aa706c6b57c52d786d8910be85dbdeab1398d60e61a129e8e13"),
    "S15_contract": (S15R / "formal-002-CONTRACT.json", "14e502a69a55496b53be8a89d078fba20322185731ebbe8f46d519b8fea852a1"),
}
METRIC_FILES = {
    "A0@L": {"600": (R25 / "formal-001/evaluation/A0-EMA-A-top600-metrics.json",
                     "2b4556491834cbac2099caaa22979092614c9c5e649dde5aec85765d6358652a"),
             "300": (R25 / "formal-001/evaluation/A0-EMA-A-top300-metrics.json",
                     "89ebae18c3bde3ceb32927792a0e01950b5badaeb7dd36ba9d9f3dd0fde5a217")},
    "S15@H": {"600": (S15R / "formal-002/evaluation/S15-EMA-A-top600-metrics.json",
                      "ab9c410a6e36d00b7a148e106e33cc17d3ebcc5fdb3e5a418f7bfbc4fd6e48ff"),
              "300": (S15R / "formal-002/evaluation/S15-EMA-A-top300-metrics.json",
                      "f523eff31d6e2e41fb2a63cb08e616d69bff70aa68166149a3fa58d7edc7b8c7")},
    "S15@L": {"600": (CROSS / "diag-001/score/M_HL-EMA-top600-metrics.json",
                      "e545ac6c30a5cb1ac159e3ced629650b59fa62b066ed880e9713f9156ddafb86"),
              "300": (CROSS / "diag-001/score/M_HL-EMA-top300-metrics.json",
                      "0277fd34ab31281ec12f59b187b36a16d483fd531d27368690196300a0d6135f")},
}
REFUSAL_FILES = {
    "other_seed_parent": (EXPERIMENTS / "r18-sahi-seed12-20261016-001/seed1/formal/checkpoint-epoch-060.pth",
                          "6f8fd316a4e508b9cc8a76be2a18f7c5aaa0beb1b5bd809fd27cb83a1de05234"),
    "S15_e120": (S15R / "formal-002/train-S15/checkpoint-epoch-120.pth",
                 "656f7be2357f20dce50a718f1093a639c1a32a7faad69d03184e36e43f32871f"),
}
REUSED_EVIDENCE = {
    "S15_A0_768_plain_checkpoint_CUDA_equivalence": (S15R / "formal-002/accept/A0_768_EQUIVALENCE.json",
                                                      "c000130d8d2b690a78d7b031556296b37565ed3ce50b9b1f56d9d685dadd87fe"),
    "S15_accept_closed": (S15R / "formal-002/accept/CLOSED.json",
                          "ffc0373440169726090c00a5236d05ef02dc684dae1d06e214550b3c8b866ff7"),
}
ALLOCATOR = "expandable_segments:True"
EVALUATION = {
    "epoch": 120, "weights": ["ema"], "online": "kept in the complete e120 checkpoint; not inferred or scored",
    "primary_endpoint": "MIX@H 1152x2016", "secondary_endpoint": "MIX@L 768x1344 (readout only)",
    "canvases": {"H": [1152, 2016], "L": [768, 1344]}, "view": "A", "images": 548, "batch": 4,
    "precision": "float32", "warmup_batches_per_endpoint": 4, "steady_batches_per_endpoint": 137,
    "budgets": [600, 300], "score_calls": 4, "per_class_APL_budget": 600,
    "per_class_APL_extra_COCOeval_calls_max_per_endpoint": 1,
}
WORKLOAD = {
    "arm": "MIX", "seed": 0, "epochs": [61, 120], "initial_update": 18240, "updates_per_epoch": 304,
    "formal_updates": 18240, "stop_update": 36480, "save_epochs": [60, 75, 90, 105, 120],
    "train_batch": 8, "accumulation": 2, "precision": "bfloat16", "clip_norm": 0.1,
    "canvases": {"L": [768, 1344], "H": [1152, 2016]},
    "schedule": "e061-e117: per epoch 152 L / 152 H by (SHA256('MIX_LH_V1|seed=0|epoch=e|window=w'), w) ascending, "
                "first 152 L; e118-e120 all H; window time order unchanged; one canvas per optimizer update",
    "canvas_totals": {"L": 8664, "H": 9576},
    "engine_state_recipe": "S15 production recipe (1152x2016 persistent caches); L updates use the native 768 recipe",
    "activation_checkpoint": True,
    "checkpoint_segments": ["backbone.res_layers", "encoder.fpn_blocks", "encoder.pan_blocks"],
    "allocator_conf": ALLOCATOR,
    "acceptance": {"sequence": "acceptance-only L,H,L on data windows 0,1,2; save u1; fresh engine replays u2,u3",
                   "updates": 5, "training_forwards": 10},
    "evaluation": EVALUATION,
}
DECISION_RULE = {
    "primary_comparison": "MIX@H e120 EMA minus native A0@L e120 EMA (seed0, R25 formal-001)",
    "seven_cells": {"top600_APS": 0.50, "top600_AP75_small": 0.30, "top600_AP": -0.10, "top300_AP": -0.20,
                    "top300_APS": -0.20, "top600_APM": -0.50, "top600_APL": -0.50},
    "retention": {"cells": ["top600_APS", "top600_AP75_small"], "fraction": 0.80,
                  "rule": "MIX@H - A0@L >= 0.80 x (S15@H - A0@L), certified unrounded metrics"},
    "pass": "single_seed_preliminary_scale_tradeoff_reduction", "fail": "closed_fixed_config",
    "secondary": "MIX@L never changes the decision", "originality_claim": None,
}
RESOURCE = {"reference_train_wall_hours": "10-14 (not a cap)", "artificial_resource_caps": None,
            "no_progress_seconds": 1800, "reserve_disk_bytes": 4 * 1024**3, "remaining_save_estimate_bytes": 2 * 1024**3}
MODEL_FILES = (
    "src/sparse_rtdetr/protocol_r18/model.py", "src/sparse_rtdetr/protocol_r18/engine.py",
    "src/sparse_rtdetr/protocol_r18/geometry.py", "src/sparse_rtdetr/protocol_r18/data.py",
    "src/sparse_rtdetr/protocol_r18/checkpoint.py", "src/sparse_rtdetr/localization/data.py",
    "src/sparse_rtdetr/baseline/training_v2b.py", "src/sparse_rtdetr/baseline/training_v2b_data.py",
    "src/sparse_rtdetr/baseline/training_v2b_engine.py", "src/sparse_rtdetr/research/query_local_runtime.py",
    "src/sparse_rtdetr/r0/evaluation.py", "src/sparse_rtdetr/r0/evaluation_data.py",
    "tools/query_local/formal_data.py", "infer_seed.py",
)
S15_FILES = ("src/sparse_rtdetr/research/scale_baseline.py", "tools/scale_baseline/s15_formal_eval.py")
MIX_FILES = ("src/sparse_rtdetr/research/mix_scale.py", "tools/scale_baseline/mix_formal_run.py",
             "tools/scale_baseline/mix_formal_train.py", "tools/scale_baseline/mix_formal_eval.py")
ALLOWED_CHANGES_VS_SEED12 = set(MIX_FILES) | {"tools/scale_baseline/s15_cross_scale.py"}
RELEASE_VERSION = "MIX-SCALE-001-release-v1"
CONTRACT_VERSION = "MIX-SCALE-001-contract-v1"


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


def task_refs():
    return {"task": pinned(ROUND / TASK[0], TASK[1]), "review": pinned(ROUND / REVIEW[0], REVIEW[1])}


# ---------------------------------------------------------------- release
def freeze(release_name):
    release = ROUND / suffix(release_name, "release")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip(), "worktree must be clean")
    tasks = task_refs()
    r25_manifest, seed12_manifest = pinned(*R25_RELEASE_MANIFEST), pinned(*SEED12_RELEASE_MANIFEST)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    release.mkdir()
    source = release / "source"
    source.mkdir()
    archive = subprocess.check_output(["git", "archive", commit], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for entry in tar.getmembers():
            p = source / entry.name
            require(p.resolve().is_relative_to(source) and (entry.isdir() or entry.isfile()), f"unsafe member {entry.name}")
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
    same = {p: p in current and base.get(p) == current.get(p) for p in MODEL_FILES}
    require(all(same.values()), f"native computation differs from the R25 A0 release: {same}")
    seed12 = {row["path"]: row["sha256"] for row in read(seed12_manifest["path"])["source_files"]}
    require(all(p in current and seed12.get(p) == current[p] for p in S15_FILES), "S15 bridge/evaluator files differ")
    changed = sorted(p for p in set(seed12) | set(current) if seed12.get(p) != current.get(p))
    require(set(changed) <= ALLOWED_CHANGES_VS_SEED12, f"changes beyond MIX files: {changed}")
    require(set(MIX_FILES) <= set(current), "MIX files missing from release")
    path = release / "MANIFEST.json"
    write_new(path, {"version": RELEASE_VERSION, **tasks, "commit": commit, "R25_A0_release_manifest": r25_manifest,
                     "native_files_identical_to_R25_A0_release": sorted(MODEL_FILES),
                     "S15_seed12_release_manifest": seed12_manifest,
                     "S15_files_identical_to_seed12_release": sorted(S15_FILES),
                     "changed_vs_S15_seed12_release": changed, "source_files": rows})
    print(json.dumps(ref(path)), flush=True)


def verify_source(source, manifest):
    indexed = {row["path"]: row for row in manifest["source_files"]}
    found = {p.relative_to(source).as_posix(): p for p in source_paths(source)}
    require(len(indexed) == len(manifest["source_files"]) and indexed.keys() == found.keys(),
            "source inventory differs from release manifest")
    for name, path in found.items():
        row = indexed[name]
        require(path.stat().st_size == row["bytes"] and file_sha(path) == row["sha256"], f"source bytes differ: {name}")


def release_manifest(release_name):
    release = ROUND / suffix(release_name, "release")
    manifest_ref = ref(release / "MANIFEST.json")
    manifest = read(manifest_ref["path"])
    require(manifest["version"] == RELEASE_VERSION and manifest["task"] == task_refs()["task"], "wrong MIX release")
    verify_source(release / "source", manifest)
    return release, manifest_ref, manifest


def _mix_module(release):
    sys.path.insert(0, str(release / "source" / "src"))
    from sparse_rtdetr.research import mix_scale
    require(Path(mix_scale.__file__).resolve().is_relative_to((release / "source").resolve()),
            "MIX module not imported from the frozen release")
    return mix_scale


def table(release_name):
    release, manifest_ref, _ = release_manifest(release_name)
    M = _mix_module(release)
    path = ROUND / TABLE_NAME
    require(not path.exists(), "MIX scale table already written")
    raw = M.table_bytes(M.generate_rows())
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    s = M.formal_schedule(path, file_sha(path))
    print(json.dumps({"table": ref(path), "identity": s.identity, "release_manifest": manifest_ref}), flush=True)


# ---------------------------------------------------------------- references
def ledgers():
    result = {}
    for name, (sidecar, e120_sha) in LEDGERS.items():
        side = read(sidecar)
        require(side["status"] == "COMPLETE" and side["global_update"] == 36480 and side["file"]["sha256"] == e120_sha
                and side["scope"] == "formal", f"{name}: e120 sidecar identity")
        audit = ref(side["audit"]["path"])
        require(audit == side["audit"], f"{name}: ledger is not the full e120-bound audit")
        result[name] = audit
    return result


def references():
    return {**{k: pinned(*v) for k, v in REFERENCE_FILES.items()},
            **{name: {"metrics": {b: pinned(*v) for b, v in budgets.items()}} for name, budgets in METRIC_FILES.items()}}


# ---------------------------------------------------------------- contract
def contract_name(attempt):
    return ROUND / (suffix(attempt, "formal") + "-CONTRACT.json")


def contract(release_name, attempt, prior_name):
    tasks = task_refs()
    release, manifest_ref, manifest = release_manifest(release_name)
    M = _mix_module(release)
    table_ref = ref(ROUND / TABLE_NAME)
    schedule = M.formal_schedule(table_ref["path"], table_ref["sha256"])
    data_ref = pinned(DATA_REF, DATA_REF_SHA)
    old = read(data_ref["path"])
    parent = pinned(*PARENT)
    sources = {row["path"]: row["sha256"] for row in manifest["source_files"]}
    refs = references()
    scientific = {"task": tasks["task"], "seed": 0, "parent": parent, "data": old["data"],
                  "evaluation_policy": old["evaluation_policy"],
                  "native_computation": {n: sources[n] for n in MODEL_FILES},
                  "s15_bridge_and_mix_implementation": {n: sources[n] for n in
                                                        ("src/sparse_rtdetr/research/scale_baseline.py",
                                                         "src/sparse_rtdetr/research/mix_scale.py")},
                  "schedule": schedule.identity, "workload": WORKLOAD, "decision_rule": DECISION_RULE,
                  "references": refs}
    root = ROUND / attempt
    path = contract_name(attempt)
    prior = None
    if prior_name:
        require(prior_name != attempt, "cannot continue the same attempt")
        old_contract = load_contract(contract_name(prior_name))
        require(old_contract["formal_identity"] == canonical_sha(scientific), "prior science identity differs")
        prior_path = Path(old_contract["attempt_root"])
        require((prior_path / "STOP.json").is_file() and not (prior_path / "CLOSED.json").exists(),
                "continuation requires a stopped, nonclosed prior attempt")
        require(old_contract["source_manifest"]["sha256"] != manifest_ref["sha256"], "continuation requires a new release")
        prior = {"contract": ref(contract_name(prior_name)), "stop": ref(prior_path / "STOP.json")}
    value = {
        "version": CONTRACT_VERSION, "attempt": attempt, "contract_path": str(path), "attempt_root": str(root),
        "prior": prior, **tasks, "seed": 0, "parent": parent,
        "schedule": {"table": table_ref, "table_sha256": table_ref["sha256"], "identity": schedule.identity},
        "history": ledgers(), "references": refs,
        "refusal_materials": {k: pinned(*v) for k, v in REFUSAL_FILES.items()},
        "reused_evidence": {k: pinned(*v) for k, v in REUSED_EVIDENCE.items()},
        "scientific_binding": scientific, "formal_identity": canonical_sha(scientific),
        "outputs": {"root": str(root), "accept": str(root / "accept"), "train": str(root / "train-MIX"),
                    "infer_H": str(root / "evaluation/MIX-H"), "infer_L": str(root / "evaluation/MIX-L"),
                    "score": str(root / "evaluation/score")},
        "workload": WORKLOAD, "decision_rule": DECISION_RULE, "resource_policy": RESOURCE,
        "source_root": str(release / "source"), "source_manifest": manifest_ref, "source_commit": manifest["commit"],
        "data_receipt": data_ref, "data": old["data"], "development_ids": old["development_ids"],
        "evaluation_policy": old["evaluation_policy"], "environment": old["environment"], "python": PYTHON,
        "gpu_uuid": GPU,
    }
    value["contract_sha256"] = canonical_sha(value)
    write_new(path, value)
    print(json.dumps({"file": ref(path), "contract_sha256": value["contract_sha256"],
                      "formal_identity": value["formal_identity"]}), flush=True)


def load_contract(path, *, check_source=True):
    path = Path(path).absolute()
    c = read(path)
    binding = dict(c)
    supplied = binding.pop("contract_sha256", None)
    require(supplied == canonical_sha(binding), "canonical contract digest differs")
    require(c["version"] == CONTRACT_VERSION and c["workload"] == WORKLOAD and c["decision_rule"] == DECISION_RULE
            and c["resource_policy"] == RESOURCE, "MIX workload, decision rule or resources differ")
    require(c["formal_identity"] == canonical_sha(c["scientific_binding"])
            and c["scientific_binding"]["schedule"] == c["schedule"]["identity"]
            and c["schedule"]["table"]["sha256"] == c["schedule"]["table_sha256"] == c["schedule"]["identity"]["table_sha256"]
            and c["schedule"]["identity"]["mode"] == "formal", "MIX formal identity or table binding differs")
    require(c["parent"]["path"] == str(PARENT[0]) and c["parent"]["sha256"] == PARENT[1], "MIX parent differs")
    require(c["source_root"] == str(Path(c["source_manifest"]["path"]).parent / "source"), "source binding differs")
    require(c["contract_path"] == str(path) and path == contract_name(c["attempt"])
            and c["attempt_root"] == str(ROUND / suffix(c["attempt"], "formal")), "contract relocated")
    if check_source:
        manifest = read(checked_ref(c["source_manifest"]))
        require(manifest["version"] == RELEASE_VERSION and manifest["commit"] == c["source_commit"]
                and manifest["task"] == c["task"], "contract release identity differs")
    return c


def validate_materials(c):
    started = time.monotonic()
    require({"task": c["task"], "review": c["review"]} == task_refs(), "task changed")
    data_receipt = pinned(DATA_REF, DATA_REF_SHA)
    old = read(data_receipt["path"])
    require(c["data_receipt"] == data_receipt and c["data"] == old["data"] and c["development_ids"] == old["development_ids"]
            and c["evaluation_policy"] == old["evaluation_policy"] and c["environment"] == old["environment"]
            and c["python"] == PYTHON and c["gpu_uuid"] == GPU, "admitted materials or environment changed")
    checked_ref(c["evaluation_policy"])
    require(c["parent"] == pinned(*PARENT), "parent changed")
    require(c["history"] == ledgers() and c["references"] == references(), "ledgers or references changed")
    require(c["refusal_materials"] == {k: pinned(*v) for k, v in REFUSAL_FILES.items()}
            and c["reused_evidence"] == {k: pinned(*v) for k, v in REUSED_EVIDENCE.items()}, "pinned evidence changed")
    checked_ref(c["schedule"]["table"])
    manifest = read(checked_ref(c["source_manifest"]))
    require(manifest["version"] == RELEASE_VERSION and manifest["commit"] == c["source_commit"], "release binding differs")
    verify_source(Path(c["source_root"]), manifest)
    M = _mix_module(Path(c["source_root"]).parent)
    require(M.formal_schedule(c["schedule"]["table"]["path"], c["schedule"]["table_sha256"]).identity
            == c["schedule"]["identity"], "table identity differs")
    from sparse_rtdetr.localization.material_paths import MANIFESTS, metadata_path
    from sparse_rtdetr.data_protocol.schema import stable_image_id
    counts, data_bytes = {}, 0
    for role, (digest, count, split) in MANIFESTS.items():
        entry = c["data"][role]
        rows = read(pinned(metadata_path(role, entry["manifest"], digest), digest)["path"])["records"]
        ids = [row["stable_image_id"] for row in rows]
        require(len(ids) == len(set(ids)) == count, f"{role} member count or uniqueness mismatch")
        if role == "development":
            require(ids == c["development_ids"], "development order differs")
        roots = entry["roots"]
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
                require(len(relative.parts) == 3 and relative.parts[:2] == (split, kind), "data path not a role member")
                data_bytes += pinned(Path(roots[kind]) / relative.name, row[field + "_sha256"], row[field + "_size_bytes"])["bytes"]
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
    require(all(observed[k] == v for k, v in c["environment"].items()) and observed["python"] == c["python"]
            and not observed["cuda_available"], f"runtime environment mismatch: {observed}")
    return observed


def command_output(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=15).strip()


def protect_disk(tmp=None, estimate=True):
    free = {"round": shutil.disk_usage(ROUND).free, "temporary": shutil.disk_usage(tmp or "/tmp").free,
            "shared_memory": shutil.disk_usage("/dev/shm").free}
    reserve = RESOURCE["reserve_disk_bytes"]
    required = reserve + (RESOURCE["remaining_save_estimate_bytes"] if estimate else 0)
    require(free["round"] > required, f"insufficient storage: {free}")
    require(free["temporary"] > reserve and free["shared_memory"] > 2 * 1024**3, f"insufficient tmp/shm: {free}")
    return free


def resources():
    devices = command_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free,driver_version",
                              "--format=csv,noheader,nounits"])
    target = [line for line in devices.splitlines() if line.split(",")[0].strip() == GPU]
    require(len(target) == 1 and int(target[0].split(",")[3].strip()) > 23000, f"bound GPU not free: {devices}")
    apps = command_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                           "--format=csv,noheader,nounits"])
    require(not any(GPU in line for line in apps.splitlines()), f"bound GPU already in use: {apps}")
    return {"gpu": target[0], "compute_apps": apps, "free_bytes": protect_disk(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def preflight(path):
    c = load_contract(path)
    return {"status": "PASS", "contract_sha256": c["contract_sha256"], "source_commit": c["source_commit"],
            "materials": validate_materials(c), "environment": hidden_environment(c)}


def cpu_gate(path, kind, out):
    c = load_contract(path)
    out = Path(out).absolute()
    require(out.parent == ROUND and not out.exists(), "CPU gate output must be a new directory in the round")
    program, action = {"train": ("mix_formal_train.py", "cpu"), "eval": ("mix_formal_eval.py", "cpu-check")}[kind]
    argv = [c["python"], "-B", str(Path(c["source_root"]) / "tools/scale_baseline" / program), action,
            "--contract", c["contract_path"], "--out", str(out)]
    env = execution_environment("", "")
    env["MIX_CPU_THREADS"] = "8"
    started = time.monotonic()
    with (ROUND / (out.name + ".log")).open("x") as log:
        code = subprocess.call(["nice", "-n", "19", *argv], cwd=c["source_root"], env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({"kind": kind, "returncode": code, "wall_seconds": time.monotonic() - started, "out": str(out)}),
          flush=True)
    require(code == 0, f"CPU gate {kind} failed; see {out}")


def check_gate(path, c):
    reference = ref(path)
    gate = read(path)
    require(gate["status"] == "PASS" and gate["contract_sha256"] == c["contract_sha256"]
            and gate["action"] in ("cpu", "cpu-check"), "CPU gate identity or status mismatch")
    return ("train" if gate["action"] == "cpu" else "eval"), reference


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
        launch_file = directory / "LAUNCH.json"
        state = ("active" if process_alive(read(launch_file)["pid"]) else "uncertain terminal") \
            if launch_file.is_file() else "missing launch receipt"
        raise RuntimeError(f"existing unclosed formal attempt ({state}): {directory}")


def lock_round():
    handle = (ROUND / ".mix.lock").open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("MIX run lock is held")
    return handle


def launch(path, gate_paths):
    c = load_contract(path)
    attempt = Path(c["attempt_root"])
    with lock_round() as handle:
        latest_active()
        if not c["prior"]:
            require(not list(ROUND.glob("formal-[0-9][0-9][0-9]")), "existing formal attempt; continuation needs a prior")
        require(not attempt.exists(), "duplicate attempt")
        gates = dict(check_gate(p, c) for p in gate_paths)
        require(set(gates) == {"train", "eval"}, f"CPU gates incomplete: {sorted(gates)}")
        verified = preflight(path)
        current = resources()
        attempt.mkdir()
        tmp = Path(tempfile.mkdtemp(prefix="mix-", dir="/tmp"))
        request = attempt / "REQUEST.json"
        write_new(request, {"contract": ref(path), "gates": gates, "preflight": verified, "resources": current,
                            "temporary_directory": str(tmp), "prior": c["prior"], "epoch_seconds": time.time()})
        argv = [c["python"], "-B", str(Path(c["source_root"]) / "tools/scale_baseline/mix_formal_run.py"),
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
def close_stage(directory, c, action):
    row = read(Path(directory) / "CLOSED.json")
    require(row["status"] == "PASS" and row["contract_sha256"] == c["contract_sha256"] and row["action"] == action,
            f"stage closure differs: {directory}")
    return row


def endpoint_receipt(directory, c, epoch):
    path = Path(directory) / f"checkpoint-epoch-{epoch:03d}.pth"
    side = read(path.with_suffix(".pth.json"))
    require((side["status"], side["scope"], side["arm"], side["formal_identity"], side["contract_sha256"],
             side["global_update"], side["schedule"]) ==
            ("COMPLETE", "formal", "MIX", c["formal_identity"], c["contract_sha256"], epoch * 304, c["schedule"]["identity"])
            and side["file"]["path"] == str(path), "checkpoint binding differs")
    checked_ref(side["file"])
    return path


def carry_from(c):
    """Accepted stages of a stopped prior attempt with the same scientific identity: (accept, completed, resume)."""
    if not c["prior"]:
        return None, None, None
    prior_path = checked_ref(c["prior"]["contract"])
    previous = load_contract(prior_path)
    stop = read(checked_ref(c["prior"]["stop"]))
    require(stop["contract_sha256"] == previous["contract_sha256"] and previous["formal_identity"] == c["formal_identity"],
            "prior STOP/identity binding differs")
    accept, completed, resume = carry_from(previous)
    out = previous["outputs"]
    if (Path(out["accept"]) / "CLOSED.json").exists():
        close_stage(out["accept"], previous, "accept")
        accept = ref(Path(out["accept"]) / "CLOSED.json")
    directory = Path(out["train"])
    if (directory / "CLOSED.json").exists():
        closed = close_stage(directory, previous, "train")
        require(closed["global_update"] == WORKLOAD["stop_update"], "prior completed MIX identity differs")
        endpoint_receipt(directory, previous, 120)
        completed, resume = {"path": str(directory), "contract": str(prior_path)}, None
        require(not list(Path(out["score"]).glob("*-metrics.json")), "prior evaluation already scored")
    elif directory.exists() and any(directory.glob("checkpoint-epoch-*.pth.json")):
        eligible = [ep for ep in WORKLOAD["save_epochs"] if (directory / f"checkpoint-epoch-{ep:03d}.pth.json").exists()]
        resume = {"checkpoint": str(endpoint_receipt(directory, previous, max(eligible))), "contract": str(prior_path)}
    elif directory.exists() and (directory / "UPDATES.jsonl").exists():
        require((directory / "UPDATES.jsonl").stat().st_size == 0, "MIX had updates without a recoverable formal state")
    return accept, completed, resume


def input_alignment(c, train_root):
    """Every formal row: canvas equals the frozen table; all native receipt fields equal the historical
    same-canvas ledger row (R25 A0 for L, S15 for H); per-epoch quotas and totals close."""
    sys.path.insert(0, str(Path(c["source_root"]) / "src"))
    M = _mix_module(Path(c["source_root"]).parent)
    started = time.monotonic()
    s = M.formal_schedule(c["schedule"]["table"]["path"], c["schedule"]["table_sha256"])
    hist = {}
    for name in ("A0_updates", "S15_updates"):
        rows = {}
        with checked_ref(c["history"][name]).open() as stream:
            for line in stream:
                row = json.loads(line)
                rows[row["input"]["update"]] = row
        hist[name] = rows
    per_epoch, gt, n, h_gt_equal = {}, {"L": 0, "H": 0}, 0, 0
    digest = hashlib.sha256()
    with (Path(train_root["path"]) / "UPDATES.jsonl").open() as mine:
        for line in mine:
            row = json.loads(line)
            n += 1
            update = WORKLOAD["initial_update"] + n
            inp = row["input"]
            canvas = s.canvas(update)
            require(inp["update"] == update and row["canvas"] == inp["canvas"] == inp["worker_canvas"] == canvas
                    and inp["expected_hw"] == inp["actual_hw"] == list(M.CANVAS[canvas]) and inp["schedule_mode"] == "formal",
                    f"MIX row {update}: canvas differs from the table")
            ref_row = hist["A0_updates" if canvas == "L" else "S15_updates"][update]
            require(M.native_fields(inp) == ref_row["input"], f"MIX row {update} input differs from the {canvas} ledger")
            if canvas == "H":
                require(row["gt_count"] == ref_row["gt_count"], f"MIX row {update} GT count differs from S15")
                h_gt_equal += 1
            per_epoch.setdefault(inp["epoch"], Counter())[canvas] += 1
            gt[canvas] += row["gt_count"]
            digest.update(bytes.fromhex(canonical_sha(inp)))
    require(n == WORKLOAD["formal_updates"], "MIX input audit length differs")
    for ep, counts in per_epoch.items():
        want = {"L": 152, "H": 152} if ep <= 117 else {"H": 304}
        require(dict(counts) == want, f"epoch {ep} quota {dict(counts)}")
    totals = {k: sum(v.get(k, 0) for v in per_epoch.values()) for k in ("L", "H")}
    require(totals == WORKLOAD["canvas_totals"], f"MIX totals {totals}")
    return {"updates": n, "canvas_totals": totals, "per_epoch_quota_closed": True,
            "rows_equal_historical_same_canvas_ledger": n, "fields": list(M.NATIVE_RECEIPT_FIELDS),
            "H_rows_gt_count_equal_S15": h_gt_equal, "positive_GT_after_sanitize_by_canvas": gt,
            "input_sha256": digest.hexdigest(), "full_audit": ref(Path(train_root["path"]) / "UPDATES.jsonl"),
            "ledgers": c["history"], "seconds": time.monotonic() - started}


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
    files = [log, path / "UPDATES.jsonl", path / "progress.jsonl", path / "events.jsonl"]
    return max(p.stat().st_mtime_ns for p in files if p.exists())


def run_stage(c, attempt, tmp, label, program, action, directory, extra, ledger, device):
    protect_disk(tmp)
    directory = Path(directory)
    log = attempt / f"{label}.log"
    argv = [c["python"], "-B", str(ROOT / "tools/scale_baseline" / program), action,
            "--contract", c["contract_path"], "--out", str(directory), *extra]
    began = time.monotonic()
    row = {"stage": label, "started_epoch_seconds": time.time(), "argv": argv}
    ledger.append(row)
    journal(attempt / "STAGES.jsonl", {"event": "START", **row})
    child = None
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=execution_environment(device, tmp), start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        stamp, last = progress_time(directory, log), time.monotonic()
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
        result = close_stage(directory, c, action)
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


def measurements(tmp):
    paths = {"experiment": ROUND, "temporary": tmp, "torch_cache": Path.home() / ".cache/torch",
             "torch_extensions": Path.home() / ".cache/torch_extensions", "triton_cache": Path.home() / ".triton"}
    result = {}
    for kind, path in paths.items():
        try:
            result[kind] = int(command_output(["du", "-sb", str(path)]).split()[0]) if Path(path).exists() else 0
        except (ValueError, OSError, subprocess.SubprocessError):
            result[kind] = None
    return result


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
              "formal_identity": c["formal_identity"]}
    out = c["outputs"]
    try:
        accept, completed, resume = carry_from(c)
        write_new(attempt / "INHERITED.json", {"accept": accept, "train_root": completed, "resume": resume,
                                               "prior": c["prior"]})
        Path(out["root"]).mkdir(exist_ok=True)
        if accept is None:
            stage = "accept"
            row = run_stage(c, attempt, tmp, "accept", "mix_formal_train.py", "accept", out["accept"], [], ledger, GPU)
            acc = WORKLOAD["acceptance"]
            require(row["totals"]["optimizer_updates"] == acc["updates"]
                    and row["totals"]["full_model_forwards"] == acc["training_forwards"]
                    and row["replay_results_and_complete_state_exact"] is True and row["formal_state_usable"] is False,
                    "acceptance work count or status differs")
        if completed is None:
            stage = "train-MIX"
            args = ["--resume", resume["checkpoint"], "--prior-contract", resume["contract"]] if resume else []
            row = run_stage(c, attempt, tmp, "train-MIX", "mix_formal_train.py", "train", out["train"], args, ledger, GPU)
            require(row["arm"] == "MIX" and row["effective_updates"] == WORKLOAD["formal_updates"]
                    and row["global_update"] == WORKLOAD["stop_update"]
                    and row["canvas_totals"] == WORKLOAD["canvas_totals"], "formal training terminal count differs")
            completed = {"path": out["train"], "contract": str(path)}
        stage = "input-alignment"
        root = Path(out["root"])
        write_new(root / "TRAIN_ROOT.json", completed)
        write_new(root / "INPUT_ALIGNMENT.json", input_alignment(c, completed))
        Path(out["infer_H"]).parent.mkdir(exist_ok=True)
        for canvas in ("H", "L"):
            stage = f"infer-MIX@{canvas}"
            run_stage(c, attempt, tmp, stage, "mix_formal_eval.py", "infer", out[f"infer_{canvas}"],
                      ["--canvas", canvas, "--train-root", str(root / "TRAIN_ROOT.json")], ledger, GPU)
        stage = "score"
        row = run_stage(c, attempt, tmp, "score", "mix_formal_eval.py", "score", out["score"],
                        ["--infer-H", out["infer_H"], "--infer-L", out["infer_L"]], ledger, "")
        comparison = read(checked_ref(row["result"]))
        require(comparison["score_calls"] == EVALUATION["score_calls"] and comparison["extra_COCOeval_calls"] == 2,
                "fixed score count differs")
        result.update(status="PASS", scientific_decision=comparison["scientific_decision"],
                      branch=comparison["gates"]["branch"], comparison=row["result"])
    except BaseException as exc:
        result.update(status="STOP", stage=stage, error=repr(exc))
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
    for name in ("table", "freeze"):
        actions.add_parser(name).add_argument("--release", required=True)
    con = actions.add_parser("contract")
    con.add_argument("--release", required=True)
    con.add_argument("--attempt", required=True)
    con.add_argument("--prior")
    gate = actions.add_parser("cpu-gate")
    gate.add_argument("--contract", type=Path, required=True)
    gate.add_argument("--kind", choices=("train", "eval"), required=True)
    gate.add_argument("--out", type=Path, required=True)
    for name in ("preflight", "launch", "status", "_supervise"):
        call = actions.add_parser(name)
        call.add_argument("--contract", type=Path, required=True)
        if name == "launch":
            call.add_argument("--gates", type=Path, nargs=2, required=True)
        elif name == "_supervise":
            call.add_argument("--lock-fd", type=int, required=True)
            call.add_argument("--request-sha", required=True)
    args = parser.parse_args()
    if args.action == "table":
        table(args.release)
    elif args.action == "freeze":
        freeze(args.release)
    elif args.action == "contract":
        contract(args.release, args.attempt, args.prior)
    elif args.action == "cpu-gate":
        cpu_gate(args.contract, args.kind, args.out)
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
