"""TASK_S15_CROSS_SCALE_001: fixed seed0 2x2 cross-scale diagnostic.  RECORD_ONLY, zero training.

Rows = e060->e120 training configuration of the checkpoint (A0: native 768x1344, S15: 1152x2016).
Columns = evaluation canvas of this readout (L=768x1344, H=1152x2016).

  M_LL  A0 @ L   certified R25 formal-001 cache (reused; only the GPU consistency gate re-forwards 8 images)
  M_LH  A0 @ H   new full 548-image EMA endpoint
  M_HL  S15 @ L  new full 548-image EMA endpoint
  M_HH  S15 @ H  certified S15 formal-002 cache (reused; only the GPU consistency gate re-forwards 8 images)

The checkpoint's training configuration and this readout's canvas are declared separately.  Each
endpoint's learned parameters, EMA and BN statistics are loaded byte-for-byte; only the canvas
dependent caches are the native regeneration at the evaluation canvas (decoder anchors/valid_mask
buffers, encoder sin-cos position cache, eval_spatial_size).  The source caches are first proven to be
the native regeneration of the checkpoint's own training canvas.  Nothing is written to any checkpoint.
The legacy training/evaluation loaders and their legality checks are not modified.

  freeze    --release release-NNN
  contract  --release release-NNN --attempt diag-NNN
  cpu-check --contract C --out DIR            CUDA hidden; identities, caches, coordinates, GT-free path, 4 CPU forwards
  gpu-gate  --contract C --out DIR            M_LL and M_HH first 8 images (2 batches each, 4 forwards), no scoring
  infer     --contract C --cell LH|HL --out DIR   4 warm-up + 137 batches, FP32 batch4, fresh process per cell
  score     --contract C --out DIR            4 fixed score_budget calls (LH/HL x top600/top300)
  large     --contract C --score DIR --out DIR   8 top600 lists, one CPU COCOeval each, bounded large description
  summary   --contract C --score DIR --large DIR --out DIR
  run       --contract C --cpu-check DIR      detached sequential supervisor: gpu-gate, infer LH, infer HL, score, large, summary
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "query_local"), str(ROOT / "tools" / "scale_baseline"),
                str(ROOT)]

from s15_formal_run import (MODEL_FILES, R25_RELEASE_MANIFEST, R25_RELEASE_MANIFEST_SHA, canonical_sha, file_sha,
                            journal, pinned, read, ref, suffix, verify_source, write_new)

W = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
EXP = W / "experiments/localization-distribution"
ROUND = EXP / "s15-cross-scale-diagnostic-20261107-001"
R25 = EXP / "r25-query-local-regression-20261104-001"
S15R = EXP / "s15-scale-baseline-20261105-001"
S12R = EXP / "s15-seed12-replication-20261106-001"
SAHI = EXP / "r18-sahi-seed12-20261016-001"
PYTHON = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
GPU = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
TASK_NAME = "TASK_S15_CROSS_SCALE_001.md"
TASK_SHA = "96bec0f01df271a658f7b461a7f91b53f56bae250143c56ec95b83901d33a500"
SEED12_RELEASE_MANIFEST = S12R / "release-001/MANIFEST.json"
SEED12_RELEASE_MANIFEST_SHA = "87a4c6d029fd0f19ed20dcf9cfe94e5853098a551a16a5d7ca1687fde56f0864"
THIS_FILE = "tools/scale_baseline/s15_cross_scale.py"
RELEASE_VERSION = "S15-CROSS-SCALE-001-release-v1"
CONTRACT_VERSION = "S15-CROSS-SCALE-001-contract-v1"

L, H = (768, 1344), (1152, 2016)
CANVAS = {"L": L, "H": H}
CELLS = {"LL": ("A0", "L", "L"), "LH": ("A0", "L", "H"), "HL": ("S15", "H", "L"), "HH": ("S15", "H", "H")}
NEW_CELLS = ("LH", "HL")
DIAGONAL = ("LL", "HH")
END = 36480
BATCH = 4
WARMUPS = 4
GATE_BATCHES = 2
SEED0_PARENT = "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"

ENDPOINTS = {
    "A0": {"file": R25 / "formal-001/train-A0/checkpoint-epoch-120.pth",
           "sha256": "78c3e896a11de260f2a09e4a860457fbd868eeec81ce1f1632efdfc0ddd8d966",
           "ema_weights_sha256": "a322373726cc3c666c8ba0ea79a5dadcc80c149d87f8c4094e9c9c807652d676",
           "contract": R25 / "formal-001-CONTRACT.json",
           "contract_sha256": "178eeb821a2049e12c0ba0c21ebd2f4c628715f8c1358218aa60c9b240e32ece",
           "formal_identity": "e9b6ff34e9ec1e42a75fd6cb46b4e2faad938512dcd4327049faee3e149bbe8a",
           "schema": "r25-complete-state-v1", "train_canvas": "L"},
    "S15": {"file": S15R / "formal-002/train-S15/checkpoint-epoch-120.pth",
            "sha256": "656f7be2357f20dce50a718f1093a639c1a32a7faad69d03184e36e43f32871f",
            "ema_weights_sha256": "072616ff442dec05693f8f4f98974213f59a35b72edfbf4a1fb87bb7fd49ac68",
            "contract": S15R / "formal-002-CONTRACT.json",
            "contract_sha256": "22a1ac219788353ec2e4befb31d427002e65612c43b3baf5f7cce75e81c23884",
            "formal_identity": "6e4780e3db859231aa0a0385aabd522879f8fcf85d66c7af476e8eba2e6cda63",
            "schema": "s15-complete-state-v1", "train_canvas": "H"},
}
# Certified diagonal caches (final ranked predictions and metrics only; no raw query cache exists).
DIAGONAL_CACHE = {
    "LL": {"result": R25 / "formal-001/evaluation/A0-EMA/RESULT.json",
           "predictions": {"600": (R25 / "formal-001/evaluation/A0-EMA/top600-predictions.json",
                                   "95eb697cd4bd30a1bebdebd052b88d4cd71d22bd3ae8fd96a018bcef8c91c0e0"),
                           "300": (R25 / "formal-001/evaluation/A0-EMA/top300-predictions.json",
                                   "69b3aa352e7c497a250bc69fcef1e50331fcfb6ff95c1ba4b028f659eb680df3")},
           "metrics": {"600": (R25 / "formal-001/evaluation/A0-EMA-A-top600-metrics.json",
                               "2b4556491834cbac2099caaa22979092614c9c5e649dde5aec85765d6358652a"),
                       "300": (R25 / "formal-001/evaluation/A0-EMA-A-top300-metrics.json",
                               "89ebae18c3bde3ceb32927792a0e01950b5badaeb7dd36ba9d9f3dd0fde5a217")}},
    "HH": {"result": S15R / "formal-002/evaluation/S15-EMA/RESULT.json",
           "predictions": {"600": (S15R / "formal-002/evaluation/S15-EMA/top600-predictions.json",
                                   "f8e89847c4ebcc67f8ed08b8f794099f7459b9051b096517656afcb6b6af0ac5"),
                           "300": (S15R / "formal-002/evaluation/S15-EMA/top300-predictions.json",
                                   "6bfe753fa811afb8dd96abe6493f2b232b954c60949795d6a09d56d3319a8092")},
           "metrics": {"600": (S15R / "formal-002/evaluation/S15-EMA-A-top600-metrics.json",
                               "ab9c410a6e36d00b7a148e106e33cc17d3ebcc5fdb3e5a418f7bfbc4fd6e48ff"),
                       "300": (S15R / "formal-002/evaluation/S15-EMA-A-top300-metrics.json",
                               "f523eff31d6e2e41fb2a63cb08e616d69bff70aa68166149a3fa58d7edc7b8c7")}},
}
# Existing top600 lists of seed1/seed2 (A0@L, S15@H) and the per-class APL they were reported with.
SEED12_LISTS = {
    (1, "A0@L"): (SAHI / "seed1/infer-A/top600-predictions.json",
                  "722f3157570679afd62a0613f85f7324b7fb68fe871480c881fca81f02903ea6",
                  SAHI / "protocol-resume-20261017-001/seed1/evaluation/A-top600-metrics.json",
                  "9f54db25849086de4b6701fe483d4cbf08acc4bd3e04cbad5f0d12ca1dd1e1f4"),
    (1, "S15@H"): (S12R / "formal-001/seed1/evaluation/S15-EMA/top600-predictions.json",
                   "87f15dbdce49f28b3444ea1ae2310af4dfe4832f71067d012abe3a92eaa0a0dd",
                   S12R / "formal-001/seed1/evaluation/S15-seed1-EMA-A-top600-metrics.json",
                   "401d0679821c6aaf643af06fa3a2f5772b3db093621973d3b718b623275373fc"),
    (2, "A0@L"): (SAHI / "protocol-resume-20261017-001/seed2/infer-A/top600-predictions.json",
                  "38f86ae49363264915314bc01e9f4f47d44b79d787c2bbd75259810bae5c18f6",
                  SAHI / "protocol-resume-20261017-001/seed2/evaluation/A-top600-metrics.json",
                  "68f4833c74b437e0297c6cf0bc108907ed4f2577ef7ba6912ee477317e5c0804"),
    (2, "S15@H"): (S12R / "formal-001/seed2/evaluation/S15-EMA/top600-predictions.json",
                   "2baf949ae8a25d6aabc093a0cb0c1bb128977bca5baa5400c7ea89c2d8f30ca9",
                   S12R / "formal-001/seed2/evaluation/S15-seed2-EMA-A-top600-metrics.json",
                   "70dd828deb9a207dd06a5d3fea9ba3f8bbc8b0e9aefd8ec6ff3d4f48f9874849"),
}
PER_CLASS_COMPARISONS = {
    0: (S15R / "formal-002/evaluation/COMPARISON.json", "78b486390b60869533ece6f1160a744c741ec2eac1644a61c07f6c2ed94a6759"),
    1: (S12R / "formal-001/seed1/evaluation/COMPARISON.json", "0e7b94fa422159bae3a908c9459c28463145db585d25be5250627149bd771dfb"),
    2: (S12R / "formal-001/seed2/evaluation/COMPARISON.json", "eafc37364535833acdfd5d36b8fbacc17425d2fe73859bbe6a4c839010d00149"),
}
WORKLOAD = {
    "seed": 0, "endpoint": "e120 EMA only", "cells": {k: {"checkpoint_training": v[0], "training_canvas": list(CANVAS[v[1]]),
                                                         "evaluation_canvas": list(CANVAS[v[2]])} for k, v in CELLS.items()},
    "new_full_cells": list(NEW_CELLS), "reused_diagonal_cells": list(DIAGONAL),
    "images": 548, "batch": BATCH, "precision": "float32", "raw_queries": 300, "budgets": [600, 300],
    "warmup_batches_per_new_cell": WARMUPS, "measured_batches_per_new_cell": 137,
    "gpu_gate": {"cells": list(DIAGONAL), "images": 8, "batches_per_cell": GATE_BATCHES, "warmup": 0, "scoring": False,
                 "compared_level": "final ranked top600/top300 predictions (no raw query cache exists)"},
    "planned_gpu_full_forwards": 2 * (WARMUPS + 137) + 2 * GATE_BATCHES,
    "cpu_forwards": {"cells": list(CELLS), "images": 4, "forwards": 4},
    "score_budget_calls": 4, "large_COCOeval_calls_max": 8, "updates": 0,
    "forbidden": ["optimizer", "backward", "training update", "BN re-estimation", "test-time adaptation",
                  "online weights", "prediction fusion", "endpoint selection", "rescoring diagonal cells"],
    "decision": "RECORD_ONLY",
}
METRICS = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
           "AP75_small")
LARGE_NAMES = METRICS[:12]
OLD_GATES = ((600, "APS", 0.50), (600, "AP75_small", 0.30), (600, "AP", -0.10), (300, "AP", -0.20),
             (300, "APS", -0.20), (600, "APM", -0.50), (600, "APL", -0.50))


def require(condition, error):
    if not condition:
        raise ValueError(error)


def pin(path, sha):
    return pinned(Path(path), sha)


# ================================================================ release / contract (no torch)
def freeze(release_name):
    ROUND.mkdir(exist_ok=True)
    release = ROUND / suffix(release_name, "release")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip(), "worktree must be clean")
    task = pin(ROUND / TASK_NAME, TASK_SHA)
    r25_manifest = pin(R25_RELEASE_MANIFEST, R25_RELEASE_MANIFEST_SHA)
    seed12_manifest = pin(SEED12_RELEASE_MANIFEST, SEED12_RELEASE_MANIFEST_SHA)
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
    for path in sorted(x for x in source.rglob("*") if x.is_file()):
        relative = path.relative_to(source).as_posix()
        actual = ref(path)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == actual["sha256"], f"archive blob differs: {relative}")
        rows.append({"path": relative, "sha256": actual["sha256"], "bytes": actual["bytes"]})
    current = {row["path"]: row["sha256"] for row in rows}
    base = {row["path"]: row["sha256"] for row in read(r25_manifest["path"])["source_files"]}
    require(all(p in current and base.get(p) == current[p] for p in MODEL_FILES),
            "native computation differs from the R25 A0 release")
    seed12 = {row["path"]: row["sha256"] for row in read(seed12_manifest["path"])["source_files"]}
    changed = sorted(p for p in set(seed12) | set(current) if seed12.get(p) != current.get(p))
    require(changed == [THIS_FILE], f"changes beyond the cross-scale entry: {changed}")
    path = release / "MANIFEST.json"
    write_new(path, {"version": RELEASE_VERSION, "task": task, "commit": commit, "R25_A0_release_manifest": r25_manifest,
                     "native_files_identical_to_R25_A0_release": sorted(MODEL_FILES),
                     "S15_seed12_release_manifest": seed12_manifest, "changed_vs_S15_seed12_release": changed,
                     "source_files": rows})
    print(json.dumps(ref(path)), flush=True)


def contract(release_name, attempt):
    release = ROUND / suffix(release_name, "release")
    manifest_ref = ref(release / "MANIFEST.json")
    manifest = read(manifest_ref["path"])
    task = pin(ROUND / TASK_NAME, TASK_SHA)
    require(manifest["version"] == RELEASE_VERSION and manifest["task"] == task, "wrong release")
    verify_source(release / "source", manifest)
    s15c = read(ENDPOINTS["S15"]["contract"])
    r25c = read(ENDPOINTS["A0"]["contract"])
    require(s15c["contract_sha256"] == ENDPOINTS["S15"]["contract_sha256"]
            and r25c["contract_sha256"] == ENDPOINTS["A0"]["contract_sha256"], "endpoint contract canonical digests")
    require(s15c["data"] == r25c["data"] and s15c["development_ids"] == r25c["development_ids"]
            and s15c["evaluation_policy"] == r25c["evaluation_policy"] and s15c["environment"] == r25c["environment"],
            "A0 and S15 certified evaluation bindings differ")
    endpoints = {}
    for arm, spec in ENDPOINTS.items():
        endpoints[arm] = {"file": pin(spec["file"], spec["sha256"]), "sidecar": ref(spec["file"].with_suffix(".pth.json")),
                          "contract": ref(spec["contract"]), "contract_sha256": spec["contract_sha256"],
                          "formal_identity": spec["formal_identity"], "schema": spec["schema"],
                          "ema_weights_sha256": spec["ema_weights_sha256"], "train_canvas": spec["train_canvas"]}
    diagonal = {}
    for cell, spec in DIAGONAL_CACHE.items():
        result = ref(spec["result"])
        body = read(result["path"])
        preds = {b: pin(*spec["predictions"][b]) for b in ("600", "300")}
        arm = CELLS[cell][0]
        require(body["predictions"] == preds and body["checkpoint"] == endpoints[arm]["file"]
                and body["weights_sha256"] == ENDPOINTS[arm]["ema_weights_sha256"] and body["kind"] == "EMA"
                and body["images"] == 548 and body["batch_size"] == BATCH, f"{cell} certified cache binding")
        diagonal[cell] = {"result": result, "predictions": preds,
                          "metrics": {b: pin(*spec["metrics"][b]) for b in ("600", "300")}}
    seed12 = {f"seed{s}:{label}": {"predictions": pin(p, ps), "metrics": pin(m, ms)}
              for (s, label), (p, ps, m, ms) in SEED12_LISTS.items()}
    comparisons = {str(s): pin(p, sha) for s, (p, sha) in PER_CLASS_COMPARISONS.items()}
    root = ROUND / attempt
    value = {"version": CONTRACT_VERSION, "attempt": attempt, "contract_path": str(ROUND / f"{attempt}-CONTRACT.json"),
             "attempt_root": str(root), "task": task, "workload": WORKLOAD,
             "source_root": str(release / "source"), "source_manifest": manifest_ref, "source_commit": manifest["commit"],
             "endpoints": endpoints, "diagonal_cache": diagonal, "seed12_lists": seed12, "per_class_comparisons": comparisons,
             "data": s15c["data"], "development_ids": s15c["development_ids"],
             "evaluation_policy": s15c["evaluation_policy"], "environment": s15c["environment"],
             "S15_activation_checkpoint_flag": s15c["workload"]["activation_checkpoint"],
             "python": PYTHON, "gpu_uuid": GPU}
    value["contract_sha256"] = canonical_sha(value)
    path = Path(value["contract_path"])
    write_new(path, value)
    print(json.dumps({"file": ref(path), "contract_sha256": value["contract_sha256"]}), flush=True)


def load_contract(path):
    path = Path(path).absolute()
    c = read(path)
    body = dict(c)
    require(body.pop("contract_sha256") == canonical_sha(body), "contract canonical digest")
    require(c["version"] == CONTRACT_VERSION and c["workload"] == WORKLOAD and c["contract_path"] == str(path),
            "contract version/workload/path")
    require(c["task"] == pin(ROUND / TASK_NAME, TASK_SHA), "task changed")
    manifest = read(pin(c["source_manifest"]["path"], c["source_manifest"]["sha256"])["path"])
    require(manifest["commit"] == c["source_commit"]
            and c["source_root"] == str(Path(c["source_manifest"]["path"]).parent / "source"), "release binding")
    for arm, spec in ENDPOINTS.items():
        e = c["endpoints"][arm]
        require(e["file"]["path"] == str(spec["file"]) and e["file"]["sha256"] == spec["sha256"]
                and e["ema_weights_sha256"] == spec["ema_weights_sha256"] and e["contract_sha256"] == spec["contract_sha256"]
                and e["formal_identity"] == spec["formal_identity"], f"{arm} endpoint pin differs from code")
    return c


# ================================================================ torch side
def _torch():
    import torch
    return torch


def endpoint_payload(c, arm, path=None):
    """Verified e120 complete state of one pinned endpoint; any other file, seed, arm or clock is refused."""
    torch = _torch()
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    spec = c["endpoints"][arm]
    path = Path(path or spec["file"]["path"]).absolute()
    require(ref(path) == spec["file"], f"{arm} endpoint file identity (wrong endpoint, seed or altered file): {path}")
    side = read(path.with_suffix(".pth.json"))
    require(ref(path.with_suffix(".pth.json")) == spec["sidecar"], f"{arm} sidecar identity")
    binding = {k: v for k, v in side.items() if k not in ("status", "file")}
    require((side["status"], side["schema"], side["scope"], side["arm"], side["global_update"], side["contract_sha256"],
             side["formal_identity"], side["parent_sha256"], side["cursor"]["seed"], side["cursor"]["end"]) ==
            ("COMPLETE", spec["schema"], "formal", arm, END, spec["contract_sha256"], spec["formal_identity"],
             SEED0_PARENT, 0, True) and side["file"] == spec["file"], f"{arm} e120 sidecar identity")
    audit = side["audit"]
    h = hashlib.sha256()
    with Path(audit["path"]).open("rb") as f:
        h.update(f.read(audit["bytes"]))
    require(h.hexdigest() == audit["sha256"], f"{arm} formal audit prefix")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_payload(payload, arm, hashlib.sha256(ck.canonical(binding)).hexdigest(), c)
    return payload


def validate_payload(payload, arm, binding_sha, c):
    from sparse_rtdetr.protocol_r18.model import Config
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research.query_local import VERSION as R25_VERSION
    require(arm in ENDPOINTS, "unknown checkpoint arm")
    require(payload.get("formal_binding_sha256") == binding_sha, f"{arm} payload/sidecar binding digest")
    if arm == "A0":
        require(payload["version"] == R25_VERSION and payload["config"] == Config(seed=0).binding(),
                "A0 payload is not the seed0 native 768x1344 R25 A0 state")
    else:
        require(payload["version"] == S.VERSION and payload["config"] == S.production_config(0).binding()
                and payload["activation_checkpoint"] == c["S15_activation_checkpoint_flag"],
                "S15 payload is not the seed0 1152x2016 S15 state")
    require(payload["arm"] == arm and payload["parent_sha256"] == SEED0_PARENT, f"{arm} payload arm/seed parent")
    require(payload["engine"]["completed_updates"] == payload["ema"]["updates"] == END, f"{arm} payload is not e120")


def ema_state(payload, c, arm, kind="EMA"):
    torch = _torch()
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    require(kind == "EMA", "only the e120 EMA endpoint is admissible (online is not part of this diagnostic)")
    state = payload["ema"]["module"]
    require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()), "EMA state not FP32")
    require(tensor_digest(state.items()) == c["endpoints"][arm]["ema_weights_sha256"],
            f"{arm} EMA weight digest differs from the certified endpoint")
    return state


def level_shapes(model, hw):
    return [[hw[0] // s, hw[1] // s] for s in model.encoder.feat_strides]


def build_cell(c, cell, state):
    """Native model at the evaluation canvas, loaded with the endpoint's EMA state except canvas caches."""
    torch = _torch()
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.protocol_r18.geometry import geometry_identity, validate_geometry
    from sparse_rtdetr.protocol_r18.model import Config, build
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research.query_local import install
    arm, train, evaluation = CELLS[cell]
    require(c["endpoints"][arm]["train_canvas"] == train, "cell training canvas binding")
    model, _, _ = build(Config(seed=0, input_hw=CANVAS[evaluation]), synthetic=True)
    if arm == "A0":
        install(model, "A0")  # identical to the certified R25 path (no-op for A0)
    validate_geometry(model)
    own = model.state_dict()
    keys = sorted(k for k in state if S._is_size_cache(k))
    require(keys == ["decoder.anchors", "decoder.valid_mask"], f"unexpected size caches {keys}")
    anchors, mask = model.decoder._generate_anchors(spatial_shapes=level_shapes(model, CANVAS[train]))
    require(ck.exact_equal(state["decoder.anchors"], anchors) and ck.exact_equal(state["decoder.valid_mask"], mask),
            f"{cell}: endpoint size cache is not the native regeneration of its training canvas {train}")
    loaded = dict(state)
    for k in keys:
        loaded[k] = own[k].detach().clone()
    require(set(loaded) == set(own), "state keys differ")
    model.load_state_dict(loaded, strict=True)
    validate_geometry(model)
    model.eval().requires_grad_(False)
    after = model.state_dict()
    non_size = [(k, v) for k, v in state.items() if k not in keys]
    require(all(torch.equal(after[k], v) for k, v in non_size), f"{cell}: learned/EMA/BN tensors not byte-identical")
    bn = [n for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    source_sha = tensor_digest(state.items())
    loaded_sha = tensor_digest(after.items())
    if train == evaluation:
        require(loaded_sha == source_sha, f"{cell}: native-canvas cell must load the certified bytes unchanged")
    require(not any(getattr(m, "_s15_checkpointed", False) for m in model.modules()), "no activation checkpoint at eval")
    record = {"cell": cell, "checkpoint_training": arm, "training_canvas": list(CANVAS[train]),
              "evaluation_canvas": list(CANVAS[evaluation]), "source_EMA_sha256": source_sha,
              "loaded_model_sha256": loaded_sha,
              "non_size_tensor_sha256": tensor_digest(non_size), "non_size_tensors": len(non_size),
              "non_size_tensors_byte_identical": True, "batchnorm_modules": len(bn),
              "substituted_state_keys": keys if train != evaluation else [],
              "source_size_cache_is_native_regeneration_of_training_canvas": True,
              "source_anchor_tokens": int(state["decoder.anchors"].shape[1]),
              "evaluation_anchor_tokens": int(after["decoder.anchors"].shape[1]),
              "geometry": geometry_identity(model), "validate_geometry_at_evaluation_canvas": True}
    return model, record


@contextlib.contextmanager
def no_ground_truth(*datas):
    """Arm one guard that refuses every annotation read or parse on the prediction path."""
    import sparse_rtdetr.localization.data as LD
    roots = {Path(d.spec["roots"]["annotations"]).resolve() for d in datas}
    original = LD.read_bound
    counts = {"image_reads": 0, "annotation_reads": 0}

    def guarded(path, *args, **kwargs):
        p = Path(path).resolve()
        if any(p == a or a in p.parents for a in roots):
            counts["annotation_reads"] += 1
            raise RuntimeError("ground truth read on the prediction path")
        counts["image_reads"] += 1
        return original(path, *args, **kwargs)

    def raw(*_args, **_kwargs):
        counts["annotation_reads"] += 1
        raise RuntimeError("ground truth parse on the prediction path")

    LD.read_bound = guarded
    for d in datas:
        d.raw = raw
    try:
        yield counts
    finally:
        LD.read_bound = original
        for d in datas:
            del d.raw


def development(c, canvas):
    from sparse_rtdetr.research import scale_baseline as S
    data = S.development(c["data"]["development"], CANVAS[canvas])
    require([r["stable_image_id"] for r in data.records] == c["development_ids"], "development members or order changed")
    return data


def forward_batch(model, data, rows, device, raw_digest=None):
    """One FP32 full-model forward at the data canvas; identical arithmetic to the certified evaluators."""
    torch = _torch()
    from infer_seed import load_original, pixels
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.r0.evaluation import rank_candidates
    device = torch.device(device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    start = time.perf_counter()
    images = []
    for row in rows:
        with load_original(data, row) as image:
            images.append(pixels(data, image))
    x = torch.stack(images).to(device)
    sync()
    input_done = time.perf_counter()
    with torch.autocast(device.type, enabled=False), forward_guards(model):
        result = model(x)
    sync()
    forward_done = time.perf_counter()
    require(set(result) == {"pred_boxes", "pred_logits"} and result["pred_boxes"].shape == (len(rows), 300, 4)
            and result["pred_logits"].shape == (len(rows), 300, 10)
            and result["pred_boxes"].dtype == result["pred_logits"].dtype == torch.float32, "endpoint output schema")
    boxes, logits = result["pred_boxes"].cpu(), result["pred_logits"].cpu()
    output_done = time.perf_counter()
    batch_digest = hashlib.sha256(logits.contiguous().numpy().tobytes() + boxes.contiguous().numpy().tobytes()).hexdigest()
    if raw_digest is not None:
        raw_digest.update(batch_digest.encode())
    six = [rank_candidates(lg, bx, row, 600) for lg, bx, row in zip(logits, boxes, rows)]
    predictions = {600: [p for g in six for p in g], 300: [p for g in six for p in g[:300]]}
    rank_done = time.perf_counter()
    return predictions, batch_digest, {"data_and_h2d": input_done - start, "forward": forward_done - input_done,
                                       "cpu_materialization": output_done - forward_done,
                                       "ranking": rank_done - output_done, "total": rank_done - start}


def _rss():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


# ---------------------------------------------------------------- CPU check
def cpu_check(c, out):
    torch = _torch()
    from torchvision.transforms.v2 import functional as TF
    from infer_seed import BUDGETS, _group, _prefix, load_original
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.protocol_r18.model import Config, build
    from sparse_rtdetr.r0.evaluation import POLICY, rank_candidates
    from sparse_rtdetr.localization.consumer_geometry import corners
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU check hides CUDA")
    started, cpu0 = time.perf_counter(), time.process_time()
    result = {"status": "RUNNING"}
    # policy and data
    policy = c["evaluation_policy"]
    require(ref(policy["path"]) == policy and read(policy["path"]) == POLICY, "certified evaluation policy differs")
    data = {k: development(c, k) for k in CANVAS}
    row = data["L"].records[0]
    pixel_checks = {}
    with load_original(data["L"], row) as image:
        for k, d in data.items():
            from infer_seed import pixels
            x = pixels(d, image)
            direct = TF.pil_to_tensor(TF.resize(image, list(CANVAS[k]))).float() / 255.
            require(x.dtype == torch.float32 and tuple(x.shape) == (3, *CANVAS[k]) and torch.equal(x, direct),
                    f"{k} pixels differ from one direct resize of the original")
            pixel_checks[k] = {"shape": list(x.shape), "equal_direct_resize": True}
    # L transform equals the certified Development548 view A item pixels (the R25 path)
    certified, _ = data["L"].item(row, 0)
    with load_original(data["L"], row) as image:
        from infer_seed import pixels
        require(torch.equal(pixels(data["L"], image), certified), "L pixels differ from certified view A")
    pixel_checks["L_equals_certified_view_A_item"] = True
    result["pixels"] = {"image": row["stable_image_id"], **pixel_checks}

    # ---- endpoint identities and refusals
    payloads = {arm: endpoint_payload(c, arm) for arm in ENDPOINTS}
    states = {arm: ema_state(payloads[arm], c, arm) for arm in ENDPOINTS}
    refused = []

    def expect_refusal(label, fn):
        try:
            fn()
        except (ValueError, RuntimeError, KeyError):
            refused.append(label)
        else:
            raise AssertionError("accepted " + label)

    s15_dir = Path(c["endpoints"]["S15"]["file"]["path"]).parent
    expect_refusal("S15 slot given the e105 checkpoint (wrong endpoint)",
                   lambda: endpoint_payload(c, "S15", s15_dir / "checkpoint-epoch-105.pth"))
    expect_refusal("S15 slot given seed1 S15 e120 (wrong seed)",
                   lambda: endpoint_payload(c, "S15", S12R / "formal-001/seed1/train-S15/checkpoint-epoch-120.pth"))
    expect_refusal("S15 slot given the A0 e120 file (wrong arm)",
                   lambda: endpoint_payload(c, "S15", c["endpoints"]["A0"]["file"]["path"]))
    expect_refusal("A0 slot given the R25 A2 e120 file (wrong arm)",
                   lambda: endpoint_payload(c, "A0", R25 / "formal-001/train-A2/checkpoint-epoch-120.pth"))
    side_sha = {}
    for arm in ENDPOINTS:
        side = read(Path(c["endpoints"][arm]["file"]["path"]).with_suffix(".pth.json"))
        side_sha[arm] = hashlib.sha256(ck.canonical({k: v for k, v in side.items() if k not in ("status", "file")})).hexdigest()
    s15 = payloads["S15"]
    a0 = payloads["A0"]
    expect_refusal("S15 payload with clock 36479",
                   lambda: validate_payload({**s15, "engine": {**s15["engine"], "completed_updates": 36479}}, "S15", side_sha["S15"], c))
    from sparse_rtdetr.research import scale_baseline as S
    expect_refusal("S15 payload with seed1 recipe",
                   lambda: validate_payload({**s15, "config": S.production_config(1).binding()}, "S15", side_sha["S15"], c))
    expect_refusal("S15 payload with seed1 parent",
                   lambda: validate_payload({**s15, "parent_sha256": "6f8fd316a4e508b9cc8a76be2a18f7c5aaa0beb1b5bd809fd27cb83a1de05234"},
                                            "S15", side_sha["S15"], c))
    expect_refusal("A0 payload declared as trained at 1152x2016",
                   lambda: validate_payload({**a0, "config": Config(seed=0, input_hw=H).binding()}, "A0", side_sha["A0"], c))
    expect_refusal("A0 payload presented as S15", lambda: validate_payload(a0, "S15", side_sha["A0"], c))
    expect_refusal("payload/sidecar binding digest swapped", lambda: validate_payload(s15, "S15", side_sha["A0"], c))
    expect_refusal("online weights requested", lambda: ema_state(s15, c, "S15", kind="online"))
    key = next(k for k, v in s15["ema"]["module"].items() if v.is_floating_point() and "anchors" not in k)
    tampered = {**s15, "ema": {**s15["ema"], "module": {**s15["ema"]["module"], key: s15["ema"]["module"][key] + 0.001}}}
    expect_refusal("EMA tensor altered", lambda: ema_state(tampered, c, "S15"))
    expect_refusal("A0 EMA presented under the S15 digest", lambda: ema_state(a0, c, "S15"))
    ema_vs_online = {}
    for arm in ENDPOINTS:
        p = payloads[arm]
        differs = [k for k in p["model"] if not torch.equal(p["model"][k], p["ema"]["module"][k])]
        require(differs and states[arm] is p["ema"]["module"], f"{arm} EMA selection is not distinct from online")
        ema_vs_online[arm] = {"tensors_differing_online_vs_EMA": len(differs), "selected": "ema.module"}
    result["identity"] = {"endpoints": {arm: c["endpoints"][arm] for arm in ENDPOINTS}, "refused": refused,
                          "EMA_selection": ema_vs_online}

    # ---- canvas-dependent caches: discovered from two fresh native builds, then rebuilt per cell
    models = {k: build(Config(seed=0, input_hw=CANVAS[k]), synthetic=True)[0] for k in CANVAS}
    sl, sh = models["L"].state_dict(), models["H"].state_dict()
    require(set(sl) == set(sh), "state keys depend on canvas")
    state_diff = sorted(k for k in sl if sl[k].shape != sh[k].shape or not torch.equal(sl[k], sh[k]))
    attr_diff = []
    for (name, ml), (_, mh) in zip(models["L"].named_modules(), models["H"].named_modules()):
        for attr, v in vars(ml).items():
            if attr.startswith("_") or attr == "training":
                continue
            w = vars(mh).get(attr)
            if isinstance(v, torch.Tensor):
                if v.shape != w.shape or not torch.equal(v, w):
                    attr_diff.append(f"{name}.{attr}")
            elif isinstance(v, (int, float, str, bool, list, tuple, type(None))) and v != w:
                attr_diff.append(f"{name}.{attr}")
    require(state_diff == ["decoder.anchors", "decoder.valid_mask"], f"unexpected canvas dependent state {state_diff}")
    del models
    gc.collect()
    cells = {}
    built = {}
    for cell, (arm, train, evaluation) in CELLS.items():
        model, record = build_cell(c, cell, states[arm])
        record["canvas_attributes_from_native_build"] = attr_diff
        cells[cell] = record
        built[cell] = model
    require(cells["LL"]["loaded_model_sha256"] == c["endpoints"]["A0"]["ema_weights_sha256"]
            and cells["HH"]["loaded_model_sha256"] == c["endpoints"]["S15"]["ema_weights_sha256"],
            "diagonal cells do not load the certified weights")
    require(cells["LH"]["non_size_tensor_sha256"] == cells["LL"]["non_size_tensor_sha256"]
            and cells["HL"]["non_size_tensor_sha256"] == cells["HH"]["non_size_tensor_sha256"],
            "cross cells change learned/EMA/BN tensors")
    result["canvas_caches"] = {"state_keys_depending_on_canvas": state_diff,
                               "non_state_attributes_depending_on_canvas": attr_diff,
                               "rebuilt_natively_at_evaluation_canvas": state_diff + attr_diff,
                               "cells": cells}

    # ---- coordinate mapping: normalized cxcywh -> original pixels depends only on the original size
    g = torch.Generator().manual_seed(7)
    logits = torch.randn((300, 10), generator=g)
    boxes = torch.cat((torch.rand((300, 2), generator=g) * 0.8 + 0.1, torch.rand((300, 2), generator=g) * 0.1 + 0.01), 1)
    preds = rank_candidates(logits, boxes, row, 600)
    xy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), -1)
    xy = xy * torch.tensor([row["width"], row["height"]] * 2, dtype=torch.float32)
    require(all(p["bbox_xyxy"] == [float(v) for v in xy[p["query_id"]].tolist()] for p in preds),
            "original-pixel mapping differs")
    result["coordinates"] = {"mapping": "normalized cxcywh -> corners * (original width, height); canvas independent",
                             "rank_candidates_exact": True, "original_hw": [row["height"], row["width"]]}

    # ---- GT-free prediction path: 4 actual CPU forwards (one per cell, first 4 images), zero updates
    rows = data["L"].records[:BATCH]
    ids = [r["stable_image_id"] for r in rows]
    forwards = {}
    with no_ground_truth(data["L"], data["H"]) as guard:
        expect_refusal("annotation parse inside the GT guard", lambda: data["H"].raw(row))
        import sparse_rtdetr.localization.data as LD
        ann = Path(c["data"]["development"]["roots"]["annotations"]) / Path(row["annotation_relative_path"]).name
        expect_refusal("annotation read inside the GT guard",
                       lambda: LD.read_bound(ann, row["annotation_sha256"], row["annotation_size_bytes"]))
        before = dict(guard)
        for cell, model in built.items():
            canvas = CELLS[cell][2]
            sha = tensor_digest(model.state_dict().items())
            rng = ck.capture_rng(torch.device("cpu"))
            with torch.inference_mode():
                p, digest, timing = forward_batch(model, data[canvas], rows, "cpu")
            require(tensor_digest(model.state_dict().items()) == sha and ck.exact_equal(ck.capture_rng(torch.device("cpu")), rng)
                    and all(q.grad is None for q in model.parameters()), f"{cell} CPU forward changed state or RNG")
            grouped = {b: _group(p[b], ids, b, whole=True, exact=True) for b in BUDGETS}
            _prefix(grouped[600], grouped[300], ids, whole=True)
            corners("cross/original_pixel_corners", torch.tensor([q["bbox_xyxy"] for q in p[600]], dtype=torch.float32))
            forwards[cell] = {"images": ids, "raw_output_sha256": digest, "timing": timing, "per_image": len(p[600]) // BATCH}
        after = dict(guard)
    require(before["annotation_reads"] == after["annotation_reads"] == 2
            and after["image_reads"] - before["image_reads"] == len(CELLS) * BATCH, f"GT guard accounting {before} {after}")
    result["GT_free_prediction_path"] = {"guard_refusals_proven": 2, "annotation_reads_during_forwards": 0,
                                         "image_reads_during_forwards": after["image_reads"] - before["image_reads"]}
    result["cpu_forwards"] = forwards
    del built
    gc.collect()
    # ---- checkpoint files untouched
    for arm in ENDPOINTS:
        require(ref(c["endpoints"][arm]["file"]["path"]) == c["endpoints"][arm]["file"], f"{arm} checkpoint changed")
    require(not torch.cuda.is_initialized(), "CPU check initialized CUDA")
    result.update(status="PASS", refused_count=len(refused), checkpoint_files_unchanged=True,
                  counts={"cpu_forwards": len(forwards), "updates": 0, "score_calls": 0},
                  cost={"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu0,
                        "peak_rss_bytes": _rss()})
    write_new(out / "CPU_CHECK.json", result)
    return result


# ---------------------------------------------------------------- GPU
def setup_cuda():
    torch = _torch()
    from r25_run import setup
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU, "bound GPU not selected")
    setup("cuda:0")
    return torch.device("cuda:0")


def load_cell(c, cell, device):
    arm = CELLS[cell][0]
    payload = endpoint_payload(c, arm)
    state = ema_state(payload, c, arm)
    model, record = build_cell(c, cell, state)
    del payload, state
    gc.collect()
    model.to(device)
    return model, record


def gpu_gate(c, out):
    torch = _torch()
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    started = time.perf_counter()
    device = setup_cuda()
    counts = {"attempted_forwards": 0, "completed_forwards": 0}
    cells = {}
    for cell in DIAGONAL:
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        model, record = load_cell(c, cell, device)
        data = development(c, CELLS[cell][2])
        sha = tensor_digest(model.state_dict().items())
        rng = ck.capture_rng(device)
        rows = data.records[:GATE_BATCHES * BATCH]
        new = {600: [], 300: []}
        digests, timing = [], []
        with no_ground_truth(data) as guard, torch.inference_mode():
            for b in range(GATE_BATCHES):
                counts["attempted_forwards"] += 1
                p, d, t = forward_batch(model, data, rows[b * BATCH:(b + 1) * BATCH], device)
                counts["completed_forwards"] += 1
                new[600] += p[600]
                new[300] += p[300]
                digests.append(d)
                timing.append(t)
        require(guard["annotation_reads"] == 0, "GT read during gate")
        require(tensor_digest(model.state_dict().items()) == sha and ck.exact_equal(ck.capture_rng(device), rng)
                and all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters()),
                f"{cell} gate modified model state or RNG")
        ids = {r["stable_image_id"] for r in rows}
        comparison = {}
        for budget in (600, 300):
            ref_ = c["diagonal_cache"][cell]["predictions"][str(budget)]
            old = read(pin(ref_["path"], ref_["sha256"])["path"])
            old = [p for p in old if p["image_id"] in ids]
            require(len(old) == len(new[budget]) == len(rows) * budget, f"{cell} gate prediction count")
            mismatched = [i for i, (a, b) in enumerate(zip(old, new[budget])) if a != b]
            comparison[str(budget)] = {"records": len(old), "fields_compared": ["image_id", "query_id", "category_id",
                                                                              "score", "bbox_xyxy"],
                                       "mismatched_records": len(mismatched), "exact": not mismatched}
            del old
        require(all(v["exact"] for v in comparison.values()), f"{cell} gate differs from certified cache: {comparison}")
        cells[cell] = {"load": record, "images": [r["stable_image_id"] for r in rows], "raw_output_batch_sha256": digests,
                       "timing": timing, "certified_comparison": comparison, "model_and_RNG_unchanged": True,
                       "annotation_reads": 0, "image_reads": guard["image_reads"],
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "seconds": time.perf_counter() - t0,
                       "compared_level": "final ranked predictions only; the certified caches contain no raw query tensors"}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    require(counts["completed_forwards"] == 2 * GATE_BATCHES, "gate forward count")
    result = {"status": "PASS", "cells": cells, **counts, "updates": 0, "score_calls": 0,
              "contract_sha256": c["contract_sha256"], "elapsed_seconds": time.perf_counter() - started}
    write_new(out / "GPU_GATE.json", result)
    return result


def infer(c, cell, out):
    torch = _torch()
    from infer_seed import BUDGETS
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    require(cell in NEW_CELLS, "only the two new cells are inferred in full")
    counts = {"attempted_forwards": 0, "completed_forwards": 0}
    started = time.perf_counter()
    try:
        device = setup_cuda()
        torch.cuda.reset_peak_memory_stats(device)
        model, record = load_cell(c, cell, device)
        data = development(c, CELLS[cell][2])
        sha = tensor_digest(model.state_dict().items())
        rng = ck.capture_rng(device)
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - started
        raw = hashlib.sha256()
        warm, measured, batch_digests = [], [], []
        predictions = {b: [] for b in BUDGETS}
        with no_ground_truth(data) as guard, torch.inference_mode():
            for b in range(WARMUPS):
                counts["attempted_forwards"] += 1
                _, _, t = forward_batch(model, data, data.records[b * BATCH:(b + 1) * BATCH], device)
                counts["completed_forwards"] += 1
                warm.append(t)
            for offset in range(0, len(data.records), BATCH):
                counts["attempted_forwards"] += 1
                p, d, t = forward_batch(model, data, data.records[offset:offset + BATCH], device, raw)
                counts["completed_forwards"] += 1
                measured.append(t)
                batch_digests.append(d)
                for b in BUDGETS:
                    predictions[b].extend(p[b])
        require(guard["annotation_reads"] == 0, "GT read on the prediction path")
        require(len(warm) == WARMUPS and len(measured) == 137 and counts["completed_forwards"] == WARMUPS + 137,
                "batch count")
        require(tensor_digest(model.state_dict().items()) == sha and ck.exact_equal(ck.capture_rng(device), rng)
                and all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters()),
                f"{cell} inference modified model state or RNG")
        refs = {}
        for b in BUDGETS:
            path = out / f"top{b}-predictions.json"
            write_new(path, predictions[b])
            refs[str(b)] = ref(path)
        arm, train, evaluation = CELLS[cell]
        result = {"status": "PASS", "cell": cell, "checkpoint_training": arm, "training_canvas": list(CANVAS[train]),
                  "evaluation_canvas": list(CANVAS[evaluation]), "seed": 0, "kind": "EMA", "epoch": 120, "update": END,
                  "device": str(device), "FP32": True, "batch_size": BATCH, "contract_sha256": c["contract_sha256"],
                  "checkpoint": c["endpoints"][arm]["file"], "load": record, "weights_sha256": sha,
                  "predictions": refs, "images": len(data.records), "steady_forwards": len(measured),
                  "warmup_forwards": len(warm), "raw_output_sha256": raw.hexdigest(),
                  "raw_output_batch_sha256": batch_digests, "load_seconds": load_seconds,
                  "warmup_seconds": sum(t["total"] for t in warm),
                  "steady_timing": {k: sum(t[k] for t in measured) for k in measured[0]},
                  "cold_first_forward_seconds": warm[0]["forward"], "elapsed_seconds": time.perf_counter() - started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "annotation_reads": 0, "image_reads": guard["image_reads"], "model_and_RNG_unchanged": True,
                  "updates": 0, **counts}
        write_new(out / "RESULT.json", result)
        return result
    except BaseException as exc:
        peak = {}
        with contextlib.suppress(Exception):
            peak = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        write_new(out / "STOP.json", {"status": "STOP_CUDA_OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "STOP",
                                      "cell": cell, **counts, **peak, "elapsed_seconds": time.perf_counter() - started,
                                      "error": str(exc)[:4000], "traceback": traceback.format_exc()})
        raise


# ---------------------------------------------------------------- scoring (CPU)
def score(c, out, infer_dirs):
    from infer_seed import BUDGETS, _group, _prefix, bridge_values
    from sparse_rtdetr.r0.evaluation import score_budget
    started = time.perf_counter()
    data = development(c, "L")
    rows, ids = data.records, c["development_ids"]
    parsed = [data.raw(row) for row in rows]
    gt_ll = {b: read(pin(c["diagonal_cache"]["LL"]["metrics"][b]["path"], c["diagonal_cache"]["LL"]["metrics"][b]["sha256"])["path"])
             ["spatial_filter_counts"] for b in ("600", "300")}
    calls, metrics_refs, values = [], {}, {}
    for cell in NEW_CELLS:
        result_ref = ref(Path(infer_dirs[cell]) / "RESULT.json")
        result = read(result_ref["path"])
        require((result["status"], result["cell"], result["contract_sha256"], result["images"], result["steady_forwards"],
                 result["warmup_forwards"]) == ("PASS", cell, c["contract_sha256"], 548, 137, WARMUPS), f"{cell} inference")
        preds, grouped = {}, {}
        for b in BUDGETS:
            r = result["predictions"][str(b)]
            require(ref(r["path"]) == r, f"{cell} prediction file")
            preds[b] = read(r["path"])
            grouped[b] = _group(preds[b], ids, b, whole=True, exact=True)
        _prefix(grouped[600], grouped[300], ids, whole=True)
        del grouped
        metrics_refs[cell], values[cell] = {}, {}
        for b in BUDGETS:
            t = time.perf_counter()
            m = score_budget(rows, parsed, preds[b], c["source_root"])
            m["S15_cross_scale_binding"] = {"contract_sha256": c["contract_sha256"], "cell": cell, "budget": b,
                                            "inference_result": result_ref, "predictions": result["predictions"][str(b)],
                                            "weights_sha256": result["weights_sha256"],
                                            "evaluation_canvas": result["evaluation_canvas"],
                                            "checkpoint_training": result["checkpoint_training"]}
            require({k: m["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")} ==
                    {k: gt_ll[str(b)][k] for k in ("GT_before", "GT_after")}, f"{cell} GT set differs from M_LL")
            path = out / f"M_{cell}-EMA-top{b}-metrics.json"
            write_new(path, m)
            metrics_refs[cell][str(b)] = ref(path)
            values[cell][str(b)] = bridge_values(m)
            calls.append({"cell": cell, "budget": b, "seconds": time.perf_counter() - t,
                          "spatial_filter_counts": m["spatial_filter_counts"]})
        del preds
        gc.collect()
    require(len(calls) == 4, "fixed score count")
    result = {"status": "PASS", "score_budget_calls": len(calls), "calls": calls, "metrics": metrics_refs,
              "contract_sha256": c["contract_sha256"], "wall_seconds": time.perf_counter() - started,
              "peak_rss_bytes": _rss()}
    write_new(out / "SCORE.json", result)
    return result


# ---------------------------------------------------------------- bounded large description (CPU)
LIST_ORDER = ("seed0:A0@L", "seed0:S15@H", "seed0:A0@H", "seed0:S15@L",
              "seed1:A0@L", "seed1:S15@H", "seed2:A0@L", "seed2:S15@H")
CELL_OF = {"seed0:A0@L": "LL", "seed0:S15@H": "HH", "seed0:A0@H": "LH", "seed0:S15@L": "HL"}
PAIRS = [("seed0:A0@L", "seed0:S15@H", "seed0 diagonal S15@H vs A0@L (M_HH vs M_LL)"),
         ("seed1:A0@L", "seed1:S15@H", "seed1 diagonal"), ("seed2:A0@L", "seed2:S15@H", "seed2 diagonal"),
         ("seed0:A0@L", "seed0:S15@L", "same L evaluation: M_HL vs M_LL"),
         ("seed0:A0@H", "seed0:S15@H", "same H evaluation: M_HH vs M_LH"),
         ("seed0:A0@L", "seed0:A0@H", "fixed A0: M_LH vs M_LL"),
         ("seed0:S15@L", "seed0:S15@H", "fixed S15: M_HH vs M_HL")]
THRESHOLDS = (0.50, 0.75)


def list_sources(c, score_dir):
    sources = {}
    for cell, name in (("LL", "seed0:A0@L"), ("HH", "seed0:S15@H")):
        d = c["diagonal_cache"][cell]
        sources[name] = (d["predictions"]["600"], d["metrics"]["600"])
    sc = read(Path(score_dir) / "SCORE.json")
    for cell, name in (("LH", "seed0:A0@H"), ("HL", "seed0:S15@L")):
        m = sc["metrics"][cell]["600"]
        pred = read(m["path"])["S15_cross_scale_binding"]["predictions"]
        sources[name] = (pred, m)
    for name, v in c["seed12_lists"].items():
        sources[name] = (v["predictions"], v["metrics"])
    return sources


def _quantiles(values):
    import numpy as np
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p10": float(np.quantile(a, .1)),
            "p90": float(np.quantile(a, .9)), "positive": int((a > 0).sum()), "zero": int((a == 0).sum()),
            "negative": int((a < 0).sum())}


def describe_list(rows, parsed, predictions, metrics):
    """One full COCOeval identical to the certified bridge; large-branch facts at IoU 0.50/0.75."""
    import numpy as np
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    dataset, det, counts = bridge_inputs(rows, parsed, predictions)
    require(counts == metrics["spatial_filter_counts"], "spatial filter counts differ from the scored file")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = dataset; coco.createIndex()
        dt = coco.loadRes(det)
        e = COCOeval(coco, dt, "bbox")
        e.params.imgIds = sorted(r["id"] for r in dataset["images"]); e.params.catIds = list(range(1, 11))
        e.params.maxDets = [1, 10, 100]
        e.evaluate(); e.accumulate(); e.summarize()
    values = {k: float(v) for k, v in zip(LARGE_NAMES, e.stats)}
    small = e.eval["precision"][5, :, :, 1, 2]
    values["AP75_small"] = float(small[small > -1].mean())
    certified = metrics["COCO_bridge"]["metrics"]
    require(all(values[k] == certified[k] for k in METRICS), "recomputed aggregates differ from the scored file")
    p = e.params
    A, I = len(p.areaRng), len(p.imgIds)
    ia = p.areaRngLbl.index("large")
    lo, hi = p.areaRng[ia]
    tix = {t: int(np.argmin(np.abs(p.iouThrs - t))) for t in THRESHOLDS}
    require(all(abs(p.iouThrs[i] - t) < 1e-9 for t, i in tix.items()), "IoU threshold index")
    per_class_apl = []
    for k, cat in enumerate(p.catIds):
        pr = e.eval["precision"][:, :, k, ia, -1]
        v = pr[pr > -1]
        per_class_apl.append(float(v.mean()) if len(v) else None)
    require(abs(float(np.mean([v for v in per_class_apl if v is not None])) - values["APL"]) < 1e-14, "APL macro")
    # score rank of every detection inside its image-category list (COCOeval's own mergesort order)
    rank = {}
    for (img, cat), dts in e._dts.items():
        order = np.argsort([-d["score"] for d in dts], kind="mergesort")
        for r, i in enumerate(order):
            rank[dts[i]["id"]] = r + 1
    dets_by_img = {}
    for d in dt.dataset["annotations"]:
        dets_by_img.setdefault(d["image_id"], []).append(d)
    det_by_id = {d["id"]: d for d in dt.dataset["annotations"]}
    gt_by_id = {g["id"]: g for g in dataset["annotations"]}
    gts_by_img = {}
    for g in dataset["annotations"]:
        gts_by_img.setdefault(g["image_id"], []).append(g)

    def iou(boxes, box):
        if not boxes:
            return np.zeros(0)
        return np.asarray(mask_utils.iou([list(b) for b in boxes], [list(box)], [0])).reshape(-1)

    gt_rows = {}      # valid large GT id -> facts
    fp_rows = []      # valid FP facts
    per_class = {cat: {str(t): {"GT": 0, "TP": 0, "FP": 0, "FN": 0} for t in THRESHOLDS} for cat in p.catIds}
    for k, cat in enumerate(p.catIds):
        for i, img in enumerate(p.imgIds):
            ev = e.evalImgs[k * A * I + ia * I + i]
            if ev is None:
                continue
            gt_ids, gt_ig = list(ev["gtIds"]), np.asarray(ev["gtIgnore"]).astype(bool)
            dt_ids = list(ev["dtIds"])
            same = [d for d in dets_by_img.get(img, []) if d["category_id"] == cat]
            anyc = dets_by_img.get(img, [])
            for t in THRESHOLDS:
                ti = tix[t]
                dm, di = ev["dtMatches"][ti], ev["dtIgnore"][ti]
                gm = ev["gtMatches"][ti]
                valid_g = [j for j in range(len(gt_ids)) if not gt_ig[j]]
                tp = sum(1 for j in valid_g if gm[j] > 0)
                fp_idx = [n for n in range(len(dt_ids)) if dm[n] == 0 and not di[n]]
                c_ = per_class[cat][str(t)]
                c_["GT"] += len(valid_g); c_["TP"] += tp; c_["FN"] += len(valid_g) - tp; c_["FP"] += len(fp_idx)
                require(sum(1 for n in range(len(dt_ids)) if dm[n] > 0 and not di[n]) == tp, "TP accounting")
                for j in valid_g:
                    gid = gt_ids[j]
                    g = gt_by_id[gid]
                    row = gt_rows.setdefault(gid, {"gt": gid, "image": img, "category": cat, "area": g["area"]})
                    if "same_iou" not in row:
                        si = iou([d["bbox"] for d in same], g["bbox"])
                        ai = iou([d["bbox"] for d in anyc], g["bbox"])
                        row["same_max_iou"] = float(si.max()) if si.size else 0.0
                        row["any_max_iou"] = float(ai.max()) if ai.size else 0.0
                        row["same_iou"] = si
                        row["_same_ids"] = [d["id"] for d in same]
                    si, sids = row["same_iou"], row["_same_ids"]
                    hit = [n for n in range(len(sids)) if si[n] >= t]
                    ranks = sorted(rank[sids[n]] for n in hit)
                    md = int(gm[j])
                    facts = {"exists_same_class": bool(hit), "exists_same_class_in_maxDets": bool(ranks and ranks[0] <= 100),
                             "best_rank_same_class_at_threshold": ranks[0] if ranks else None,
                             "exists_any_class": bool(row["any_max_iou"] >= t), "matched": md > 0}
                    if md > 0:
                        d = det_by_id[md]
                        facts.update(matched_iou=float(iou([d["bbox"]], g["bbox"])[0]), matched_score=float(d["score"]),
                                     matched_rank=rank[md])
                    elif ranks and ranks[0] <= 100:
                        # candidates within maxDets exist: record what COCOeval actually matched them to
                        taken = []
                        for n in hit:
                            r_ = rank[sids[n]]
                            if r_ > 100:
                                continue
                            pos = dt_ids.index(sids[n])
                            m = int(dm[pos])
                            taken.append("matched_valid_large_GT" if m and not di[pos] else
                                         "matched_ignored_GT" if m else "unmatched")
                        facts["in_maxDets_candidate_outcomes"] = sorted(set(taken))
                    row[str(t)] = facts
                gl = [g["bbox"] for g in gts_by_img.get(img, []) if g["category_id"] == cat and not g["iscrowd"]]
                gv = [gt_by_id[gt_ids[j]]["bbox"] for j in range(len(gt_ids)) if not gt_ig[j]]
                go = [g["bbox"] for g in gts_by_img.get(img, []) if g["category_id"] != cat and not g["iscrowd"]]
                for n in fp_idx:
                    d = det_by_id[dt_ids[n]]
                    sv, sa, so = (float(iou(group, d["bbox"]).max()) if group else 0.0 for group in (gv, gl, go))
                    fp_rows.append({"t": t, "category": cat, "rank": rank[d["id"]], "score": d["score"],
                                    "max_iou_valid_large_same_class_GT": sv, "max_iou_same_class_GT_any_area": sa,
                                    "max_iou_other_class_GT": so})
    for row in gt_rows.values():
        row.pop("same_iou", None); row.pop("_same_ids", None)
    return {"aggregates": values, "per_class_APL": per_class_apl, "counts": per_class, "gt": gt_rows, "fp": fp_rows,
            "spatial_filter_counts": counts, "COCOeval_calls": 1}


def summarize_list(desc):
    out = {"aggregates": desc["aggregates"], "per_class_APL": desc["per_class_APL"],
           "spatial_filter_counts": desc["spatial_filter_counts"], "per_class": {}}
    for cat in range(1, 11):
        entry = {"APL": desc["per_class_APL"][cat - 1]}
        rows = [r for r in desc["gt"].values() if r["category"] == cat]
        for t in THRESHOLDS:
            k = str(t)
            n = len(rows)
            fn = [r for r in rows if not r[k]["matched"]]
            fps = [f for f in desc["fp"] if f["category"] == cat and f["t"] == t]
            entry[k] = {**desc["counts"][cat][k],
                        "coverage": {"valid_large_GT": n,
                                     "same_class_candidate_at_threshold": sum(r[k]["exists_same_class"] for r in rows),
                                     "same_class_candidate_at_threshold_within_maxDets100":
                                         sum(r[k]["exists_same_class_in_maxDets"] for r in rows),
                                     "any_class_candidate_at_threshold": sum(r[k]["exists_any_class"] for r in rows),
                                     "same_class_max_IoU": _quantiles([r["same_max_iou"] for r in rows]),
                                     "any_class_max_IoU": _quantiles([r["any_max_iou"] for r in rows])},
                        "FN_evidence": {"FN": len(fn),
                                        "no_same_class_candidate_at_threshold_in_list": sum(not r[k]["exists_same_class"] for r in fn),
                                        "of_which_any_class_candidate_at_threshold":
                                            sum((not r[k]["exists_same_class"]) and r[k]["exists_any_class"] for r in fn),
                                        "same_class_candidate_only_beyond_maxDets100":
                                            sum(r[k]["exists_same_class"] and not r[k]["exists_same_class_in_maxDets"] for r in fn),
                                        "same_class_candidate_within_maxDets100": sum(r[k]["exists_same_class_in_maxDets"] for r in fn),
                                        "within_maxDets_candidate_outcomes": _count(
                                            "+".join(r[k].get("in_maxDets_candidate_outcomes", [])) for r in fn
                                            if r[k]["exists_same_class_in_maxDets"])},
                        "valid_FP_flags_non_exclusive": {
                            "FP": len(fps),
                            "IoU_at_threshold_with_valid_large_same_class_GT_matched_by_another_detection":
                                sum(f["max_iou_valid_large_same_class_GT"] >= t for f in fps),
                            "IoU_at_threshold_with_same_class_GT_any_area": sum(f["max_iou_same_class_GT_any_area"] >= t for f in fps),
                            "same_class_GT_max_IoU_0.1_to_threshold":
                                sum(0.1 <= f["max_iou_same_class_GT_any_area"] < t for f in fps),
                            "same_class_GT_max_IoU_below_0.1": sum(f["max_iou_same_class_GT_any_area"] < 0.1 for f in fps),
                            "IoU_at_threshold_with_other_class_GT": sum(f["max_iou_other_class_GT"] >= t for f in fps),
                            "rank_within_image_class": _quantiles([f["rank"] for f in fps])}}
        out["per_class"][str(cat)] = entry
    return out


def _count(items):
    result = {}
    for x in items:
        result[x] = result.get(x, 0) + 1
    return result


def compare_lists(a, b):
    out = {}
    for cat in [str(i) for i in range(1, 11)] + ["all"]:
        out[cat] = {}
        ga = {g: r for g, r in a["gt"].items() if cat == "all" or r["category"] == int(cat)}
        gb = {g: r for g, r in b["gt"].items() if cat == "all" or r["category"] == int(cat)}
        require(set(ga) == set(gb), "valid large GT sets differ between lists")
        for t in THRESHOLDS:
            k = str(t)
            ha = {g for g, r in ga.items() if r[k]["matched"]}
            hb = {g for g, r in gb.items() if r[k]["matched"]}
            common = sorted(ha & hb)
            out[cat][k] = {"GT": len(ga), "hit_A": len(ha), "hit_B": len(hb), "new_in_B": len(hb - ha),
                           "lost_in_B": len(ha - hb), "common": len(common),
                           "common_hit_change_B_minus_A_conditional_subset": {
                               "matched_IoU": _quantiles([gb[g][k]["matched_iou"] - ga[g][k]["matched_iou"] for g in common]),
                               "matched_score": _quantiles([gb[g][k]["matched_score"] - ga[g][k]["matched_score"] for g in common]),
                               "matched_rank": _quantiles([gb[g][k]["matched_rank"] - ga[g][k]["matched_rank"] for g in common])},
                           "coverage_same_class_at_threshold_A_B": [sum(r[k]["exists_same_class"] for r in ga.values()),
                                                                    sum(r[k]["exists_same_class"] for r in gb.values())],
                           "lost_GT_in_B_with_same_class_candidate_at_threshold_in_B":
                               sum(gb[g][k]["exists_same_class"] for g in ha - hb),
                           "lost_GT_in_B_with_same_class_candidate_within_maxDets_in_B":
                               sum(gb[g][k]["exists_same_class_in_maxDets"] for g in ha - hb)}
    return out


def large(c, score_dir, out):
    started, cpu0 = time.perf_counter(), time.process_time()
    data = development(c, "L")
    rows = data.records
    parsed = [data.raw(row) for row in rows]
    sources = list_sources(c, score_dir)
    comparisons = {s: read(pin(v["path"], v["sha256"])["path"]) for s, v in c["per_class_comparisons"].items()}
    lists, timings, raw_gt = {}, {}, {}
    for name in LIST_ORDER:
        pred_ref, metric_ref = sources[name]
        t = time.perf_counter()
        preds = read(pin(pred_ref["path"], pred_ref["sha256"])["path"])
        metrics = read(pin(metric_ref["path"], metric_ref["sha256"])["path"])
        desc = describe_list(rows, parsed, preds, metrics)
        del preds
        seed, label = name.split(":")
        s = seed[-1]
        if label in ("A0@L", "S15@H"):
            arm = "A0" if label == "A0@L" else "S15"
            cached = comparisons[s]["per_class_APL_top600"]["EMA"][arm]
            require([r["APL"] for r in cached] == desc["per_class_APL"], f"{name} per-class APL differs from the certified table")
            require([r["GT"] for r in cached] == [desc["counts"][k]["0.5"]["GT"] for k in range(1, 11)],
                    f"{name} large GT denominators differ from the certified table")
        lists[name] = {"predictions": pred_ref, "metrics": metric_ref, "summary": summarize_list(desc),
                       "per_class_APL_matches_certified_table": label in ("A0@L", "S15@H"),
                       "aggregates_equal_scored_file": True}
        raw_gt[name] = desc
        timings[name] = time.perf_counter() - t
        gc.collect()
    pairs = {f"{a} -> {b}": {"description": text, "A": a, "B": b, "comparison": compare_lists(raw_gt[a], raw_gt[b])}
             for a, b, text in PAIRS}
    gt_detail = {name: {str(g): {k: v for k, v in r.items()} for g, r in d["gt"].items()} for name, d in raw_gt.items()}
    result = {"status": "PASS", "lists": lists, "pairs": pairs, "COCOeval_calls": len(LIST_ORDER), "score_budget_calls": 0,
              "thresholds": list(THRESHOLDS), "maxDets": 100, "area": "COCO large [96^2, 1e10] on original raw area",
              "scope_note": "list-internal geometric coverage of submitted, spatially filtered top600 candidates; "
                            "not oracle TP, not query-pool recall, not an AP upper bound",
              "timings_seconds": timings, "wall_seconds": time.perf_counter() - started,
              "cpu_seconds": time.process_time() - cpu0, "peak_rss_bytes": _rss()}
    write_new(out / "LARGE_DESCRIPTION.json", result)
    write_new(out / "LARGE_GT_DETAIL.json", gt_detail)
    return result


# ---------------------------------------------------------------- four-cell summary
def summary(c, score_dir, large_dir, out):
    values, official, refs = {}, {}, {}
    sc = read(Path(score_dir) / "SCORE.json")
    for cell in CELLS:
        values[cell], official[cell], refs[cell] = {}, {}, {}
        for b in ("600", "300"):
            r = c["diagonal_cache"][cell]["metrics"][b] if cell in DIAGONAL else sc["metrics"][cell][b]
            m = read(pin(r["path"], r["sha256"])["path"])
            values[cell][b] = {k: m["COCO_bridge"]["metrics"][k] for k in METRICS}
            official[cell][b] = m["official"]["AP"]
            refs[cell][b] = r
    diff = {}
    for b in ("600", "300"):
        v = values
        diff[b] = {}
        for k in METRICS:
            d = {"train_at_L_eval  M_HL-M_LL": v["HL"][b][k] - v["LL"][b][k],
                 "train_at_H_eval  M_HH-M_LH": v["HH"][b][k] - v["LH"][b][k],
                 "eval_scale_A0    M_LH-M_LL": v["LH"][b][k] - v["LL"][b][k],
                 "eval_scale_S15   M_HH-M_HL": v["HH"][b][k] - v["HL"][b][k]}
            inter = d["eval_scale_S15   M_HH-M_HL"] - d["eval_scale_A0    M_LH-M_LL"]
            alt = d["train_at_H_eval  M_HH-M_LH"] - d["train_at_L_eval  M_HL-M_LL"]
            require(abs(inter - alt) < 1e-12, "interaction identity")
            diag = v["HH"][b][k] - v["LL"][b][k]
            require(abs(diag - (d["train_at_L_eval  M_HL-M_LL"] + d["eval_scale_S15   M_HH-M_HL"])) < 1e-12
                    and abs(diag - (d["eval_scale_A0    M_LH-M_LL"] + d["train_at_H_eval  M_HH-M_LH"])) < 1e-12,
                    "diagonal path identity")
            diff[b][k] = {**{name: 100. * x for name, x in d.items()}, "interaction I": 100. * inter,
                          "diagonal M_HH-M_LL": 100. * diag}
    gates = {}
    for pair, (lhs, rhs) in {"M_HL-M_LL": ("HL", "LL"), "M_HH-M_LH": ("HH", "LH"), "M_HH-M_LL (certified)": ("HH", "LL")}.items():
        gates[pair] = {f"top{b}_{m}": {"difference_pp": 100. * (values[lhs][str(b)][m] - values[rhs][str(b)][m]),
                                       "old_threshold_pp": t,
                                       "distance_to_old_threshold_pp": 100. * (values[lhs][str(b)][m] - values[rhs][str(b)][m]) - t}
                       for b, m, t in OLD_GATES}
    ld = read(Path(large_dir) / "LARGE_DESCRIPTION.json")
    per_class = {cell: ld["lists"][name]["summary"]["per_class_APL"] for name, cell in CELL_OF.items()}
    gt_large = [ld["lists"]["seed0:A0@L"]["summary"]["per_class"][str(k)]["0.5"]["GT"] for k in range(1, 11)]
    pc_diff = {}
    for k in range(10):
        v = {cell: per_class[cell][k] for cell in CELLS}
        if any(x is None for x in v.values()):
            pc_diff[str(k + 1)] = None
            continue
        pc_diff[str(k + 1)] = {"GT_large": gt_large[k], **{cell: 100. * v[cell] for cell in CELLS},
                               "M_HL-M_LL": 100. * (v["HL"] - v["LL"]), "M_HH-M_LH": 100. * (v["HH"] - v["LH"]),
                               "M_LH-M_LL": 100. * (v["LH"] - v["LL"]), "M_HH-M_HL": 100. * (v["HH"] - v["HL"]),
                               "I": 100. * ((v["HH"] - v["HL"]) - (v["LH"] - v["LL"]))}
    result = {"status": "PASS", "decision": "RECORD_ONLY", "units": "values 0..1; differences pp",
              "values": values, "official_VisDrone_style_AP_percent": official, "metrics": refs,
              "differences_pp": diff, "old_gate_distances_record_only": gates,
              "per_class_APL_top600_pp": pc_diff, "contract_sha256": c["contract_sha256"]}
    write_new(out / "FOUR_CELL.json", result)
    return result


# ---------------------------------------------------------------- supervisor
def environment(device):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES=device, CUBLAS_WORKSPACE_CONFIG=":4096:8", MKL_THREADING_LAYER="GNU",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONUNBUFFERED="1")
    return env


def run(contract_path, cpu_dir):
    c = load_contract(contract_path)
    require(Path(c["source_root"]).resolve() == ROOT.resolve(), "must run from the frozen release")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    require(GPU not in apps, f"bound GPU busy: {apps}")
    root = Path(c["attempt_root"])
    root.mkdir(exist_ok=False)
    cpu = read(Path(cpu_dir) / "CPU_CHECK.json")
    require(cpu["status"] == "PASS", "CPU check did not pass")
    stages = root / "STAGES.jsonl"
    script = str(Path(c["source_root"]) / THIS_FILE)
    plan = [("gpu-gate", GPU, ["gpu-gate", "--out", str(root / "gpu-gate")]),
            ("infer-LH", GPU, ["infer", "--cell", "LH", "--out", str(root / "infer-LH")]),
            ("infer-HL", GPU, ["infer", "--cell", "HL", "--out", str(root / "infer-HL")]),
            ("score", "", ["score", "--infer-root", str(root), "--out", str(root / "score")]),
            ("large", "", ["large", "--score", str(root / "score"), "--out", str(root / "large")]),
            ("summary", "", ["summary", "--score", str(root / "score"), "--large", str(root / "large"),
                             "--out", str(root / "summary")])]
    journal(stages, {"event": "start", "time": time.time(), "contract_sha256": c["contract_sha256"],
                     "cpu_check": ref(Path(cpu_dir) / "CPU_CHECK.json"), "pid": os.getpid()})
    total = time.monotonic()
    for name, device, argv in plan:
        free = os.statvfs(root)
        require(free.f_bavail * free.f_frsize > 50 * 1024**3, "insufficient disk")
        t = time.monotonic()
        with (root / f"{name}.log").open("x") as log:
            code = subprocess.call([c["python"], "-B", script, *argv, "--contract", c["contract_path"]],
                                   cwd=c["source_root"], env=environment(device), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        journal(stages, {"stage": name, "exit": code, "seconds": time.monotonic() - t, "time": time.time()})
        if code != 0:
            write_new(root / "STOP.json", {"status": "STOP", "stage": name, "exit": code,
                                           "wall_seconds": time.monotonic() - total})
            return
    write_new(root / "CLOSED.json", {"status": "PASS", "wall_seconds": time.monotonic() - total,
                                     "contract_sha256": c["contract_sha256"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "contract", "cpu-check", "gpu-gate", "infer", "score", "large",
                                           "summary", "run"))
    parser.add_argument("--release")
    parser.add_argument("--attempt")
    parser.add_argument("--contract")
    parser.add_argument("--cell")
    parser.add_argument("--out")
    parser.add_argument("--infer-root")
    parser.add_argument("--score")
    parser.add_argument("--large")
    parser.add_argument("--cpu-check")
    args = parser.parse_args()
    if args.action == "freeze":
        return freeze(args.release)
    if args.action == "contract":
        return contract(args.release, args.attempt)
    if args.action == "run":
        return run(args.contract, args.cpu_check)
    c = load_contract(args.contract)
    require(Path(c["source_root"]).resolve() == ROOT.resolve(), "must run from the frozen release")
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        if args.action == "cpu-check":
            _torch().set_num_threads(8)
            result = cpu_check(c, out)
        elif args.action == "gpu-gate":
            result = gpu_gate(c, out)
        elif args.action == "infer":
            result = infer(c, args.cell, out)
        elif args.action == "score":
            root = Path(args.infer_root)
            result = score(c, out, {cell: root / f"infer-{cell}" for cell in NEW_CELLS})
        elif args.action == "large":
            result = large(c, args.score, out)
        else:
            result = summary(c, args.score, args.large, out)
        require(result["status"] == "PASS", "incomplete")
        write_new(out / "CLOSED.json", {"status": "PASS", "action": args.action, "contract_sha256": c["contract_sha256"],
                                        "wall_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time(),
                                        "peak_rss_bytes": _rss()})
    except BaseException as exc:
        if not (out / "STOP.json").exists():
            write_new(out / "STOP.json", {"status": "STOP", "action": args.action, "error": str(exc)[:4000],
                                          "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
