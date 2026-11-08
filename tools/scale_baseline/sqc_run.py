"""TASK_SHARED_QUERY_CLASSIFICATION_001 (SQC-001): stages of the frozen-detector shared-query classification readout.

  freeze    --release release-NNN
  contract  --release release-NNN --attempt sqc-NNN        writes the fixed head split first (manifest ids only)
  unit      --contract C --out DIR                        pytest of the affected units, CUDA hidden
  cpu-check --contract C --out DIR                        first 8 head-training images (hash order), real CPU path
  gpu-check --contract C --out DIR                        same 8 images on CUDA; 2 head updates per arm + resume replay
  extract   --contract C --role train --out DIR           frozen H full / L partial forwards of train_core 4869 (GT-free)
  labels    --contract C --extract DIR --out DIR          offline native match on (s_H, b_H) + VFL targets (CPU)
  fit       --contract C --arm B1|B2 --extract DIR --labels DIR --out DIR   2184 head updates, fixed final state
  dev       --contract C --fit-B1 DIR --fit-B2 DIR --out DIR   548 development forwards, B0/B1/B2 lists
  timing    --contract C --fit-B1 DIR --fit-B2 DIR --dev DIR --out DIR   independent B0/B1/B2 deployment passes
  score     --contract C --dev DIR --out DIR              4 score_budget calls + 2 per-class COCOeval; fixed gates
  run       --contract C --unit DIR --cpu-check DIR       detached sequential supervisor for gpu-check..score
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import math
import os
import shutil
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
ROUND = EXP / "shared-query-classification-20261109-001"
S15R = EXP / "s15-scale-baseline-20261105-001"
R25 = EXP / "r25-query-local-regression-20261104-001"
XR = EXP / "s15-cross-scale-diagnostic-20261107-001"
MIXR = EXP / "mix-scale-baseline-20261108-001"
PYTHON = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
GPU = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
TASK_NAME = "TASK_SHARED_QUERY_CLASSIFICATION_001.md"
TASK_SHA = "dfc0ddfbb9e304675869abd8c8aa3edb03f115d9f427e213d2cbd0b52b932aa0"
MIX_RELEASE_MANIFEST = MIXR / "release-001/MANIFEST.json"
MIX_RELEASE_MANIFEST_SHA = "5c47b17fa90fd7321108673cc48f1f3ffe12c87ad3a7c47ba12e2beb0fe91a81"
THIS_FILES = ("tools/scale_baseline/sqc_run.py", "src/sparse_rtdetr/research/shared_query_cls.py",
              "tests/test_shared_query_cls.py")
RELEASE_VERSION = "SQC-001-release-v1"
CONTRACT_VERSION = "SQC-001-contract-v1"
END = 36480
BATCH = 4
WARMUPS = 4
CHECK_IMAGES = 8

S15_ENDPOINT = {"file": S15R / "formal-002/train-S15/checkpoint-epoch-120.pth",
                "sha256": "656f7be2357f20dce50a718f1093a639c1a32a7faad69d03184e36e43f32871f",
                "ema_weights_sha256": "072616ff442dec05693f8f4f98974213f59a35b72edfbf4a1fb87bb7fd49ac68",
                "contract": S15R / "formal-002-CONTRACT.json",
                "contract_file_sha256": "14e502a69a55496b53be8a89d078fba20322185731ebbe8f46d519b8fea852a1",
                "contract_sha256": "22a1ac219788353ec2e4befb31d427002e65612c43b3baf5f7cce75e81c23884",
                "formal_identity": "6e4780e3db859231aa0a0385aabd522879f8fcf85d66c7af476e8eba2e6cda63",
                "schema": "s15-complete-state-v1", "train_canvas": "H"}
REFERENCES = {
    "S15_EMA_result": (S15R / "formal-002/evaluation/S15-EMA/RESULT.json", None),
    "S15_EMA_top600": (S15R / "formal-002/evaluation/S15-EMA/top600-predictions.json",
                       "f8e89847c4ebcc67f8ed08b8f794099f7459b9051b096517656afcb6b6af0ac5"),
    "S15_EMA_top300": (S15R / "formal-002/evaluation/S15-EMA/top300-predictions.json",
                       "6bfe753fa811afb8dd96abe6493f2b232b954c60949795d6a09d56d3319a8092"),
    "S15@H_600": (S15R / "formal-002/evaluation/S15-EMA-A-top600-metrics.json",
                  "ab9c410a6e36d00b7a148e106e33cc17d3ebcc5fdb3e5a418f7bfbc4fd6e48ff"),
    "S15@H_300": (S15R / "formal-002/evaluation/S15-EMA-A-top300-metrics.json",
                  "f523eff31d6e2e41fb2a63cb08e616d69bff70aa68166149a3fa58d7edc7b8c7"),
    "A0@L_600": (R25 / "formal-001/evaluation/A0-EMA-A-top600-metrics.json",
                 "2b4556491834cbac2099caaa22979092614c9c5e649dde5aec85765d6358652a"),
    "A0@L_300": (R25 / "formal-001/evaluation/A0-EMA-A-top300-metrics.json",
                 "89ebae18c3bde3ceb32927792a0e01950b5badaeb7dd36ba9d9f3dd0fde5a217"),
    "S15_comparison": (S15R / "formal-002/evaluation/COMPARISON.json",
                       "78b486390b60869533ece6f1160a744c741ec2eac1644a61c07f6c2ed94a6759"),
}
METRICS = ("AP", "APS", "AP75_small", "APM", "APL", "AP50", "AP75")
G_CELLS = ((600, "APS", 0.50), (600, "AP75_small", 0.30), (600, "AP", -0.10), (300, "AP", -0.20),
           (300, "APS", -0.20), (600, "APM", -0.50), (600, "APL", -0.50))
RETENTION = ((600, "APS"), (600, "AP75_small"))
RETENTION_FRACTION = 0.80
I_CELLS = ((600, "APL", 0.50), (600, "APS", -0.20), (600, "AP75_small", -0.20), (600, "AP", -0.10),
           (600, "APM", -0.50), (300, "AP", -0.20), (300, "APS", -0.20))
DECISIONS = {"G_and_I": "single_seed_cross_scale_readout_signal",
             "G_only": "net_benefit_without_cross_scale_evidence", "not_G": "closed_fixed_config"}
CLASS_NAMES = {1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van", 6: "truck", 7: "tricycle",
               8: "awning-tricycle", 9: "bus", 10: "motor"}
WORKLOAD = {
    "detector": "seed0 S15 formal-002 e120 EMA; eval, FP32, frozen (0 updates)",
    "canvases": {"H": [1152, 2016], "L": [768, 1344]}, "last_layer": 2, "queries": 300, "classes": 10,
    "extraction_batch": BATCH, "extraction_precision": "float32",
    "H_full_batches": {"train_core": 1218, "development": 137},
    "L_partial_batches": {"train_core": 1218, "development": 137},
    "head": {"hidden": 128, "input": 516, "parameters": 67466, "init": "CPU generator seed0 Xavier-uniform fc1, zero rest",
             "epochs": 8, "step_images": 16, "steps_per_epoch": 273, "updates_per_arm": 2184, "arms": ["B1", "B2"],
             "optimizer": "AdamW lr1e-3 betas(0.9,0.999) eps1e-8 wd1e-4; no warmup/scheduler; clip 1.0; FP32",
             "endpoint": "final state after epoch 8 (no selection)"},
    "checks": {"cpu_images": CHECK_IMAGES, "gpu_images": CHECK_IMAGES, "gpu_head_updates": 6},
    "evaluation": {"images": 548, "budgets": [600, 300], "score_budget_calls": 4, "per_class_COCOeval": 2,
                   "B0": "certified S15@H metrics reused; native lists must equal the certified lists"},
    "timing": {"endpoints": ["B0", "B1", "B2"], "warmup_batches": WARMUPS, "measured_batches": 137, "batch": BATCH,
               "B0": "H full inference + list", "B1": "H full inference + head + list",
               "B2": "H full inference + L backbone/encoder/projection + extra last layer + head + list"},
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _torch():
    import torch
    return torch


# ================================================================ release / contract
def freeze(release_name):
    ROUND.mkdir(exist_ok=True)
    release = ROUND / suffix(release_name, "release")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip(), "worktree must be clean")
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    r25 = pinned(R25_RELEASE_MANIFEST, R25_RELEASE_MANIFEST_SHA)
    mix = pinned(MIX_RELEASE_MANIFEST, MIX_RELEASE_MANIFEST_SHA)
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
    base = {row["path"]: row["sha256"] for row in read(r25["path"])["source_files"]}
    require(all(p in current and base.get(p) == current[p] for p in MODEL_FILES),
            "native computation differs from the R25 A0 release")
    old = {row["path"]: row["sha256"] for row in read(mix["path"])["source_files"]}
    changed = sorted(p for p in set(old) | set(current) if old.get(p) != current.get(p))
    allowed = set(THIS_FILES) | {"tools/scale_baseline/collect_mix.py", "tools/scale_baseline/supervision_by_canvas.py"}
    require(set(changed) <= allowed and set(THIS_FILES) <= set(changed), f"changes beyond SQC entry: {changed}")
    path = release / "MANIFEST.json"
    write_new(path, {"version": RELEASE_VERSION, "task": task, "commit": commit, "R25_A0_release_manifest": r25,
                     "native_files_identical_to_R25_A0_release": sorted(MODEL_FILES),
                     "MIX_release_manifest": mix, "changed_vs_MIX_release": changed, "source_files": rows})
    print(json.dumps(ref(path)), flush=True)


def byte_budget():
    n, d = 4869, 548
    feat = 300 * 256 * 4
    native = 300 * 14 * 4
    budget = {"train_features_bytes": n * 2 * feat, "train_native_bytes": n * native,
              "dev_features_bytes": d * 2 * feat, "dev_native_bytes": d * native,
              "labels_bytes_estimate": n * 300 * (8 + 4 + 10 + 1) + 20_000_000,
              "predictions_bytes_estimate": 6 * 60_000_000, "head_states_bytes_estimate": 2 * 10 * 1_200_000}
    budget["total_estimate_bytes"] = sum(budget.values())
    budget["reserve_bytes"] = 20 * 1024 ** 3
    return budget


def contract(release_name, attempt):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import shared_query_cls as Q
    release = ROUND / suffix(release_name, "release")
    manifest_ref = ref(release / "MANIFEST.json")
    manifest = read(manifest_ref["path"])
    task = pinned(ROUND / TASK_NAME, TASK_SHA)
    require(manifest["version"] == RELEASE_VERSION and manifest["task"] == task, "wrong release")
    verify_source(release / "source", manifest)
    s15c_ref = pinned(S15_ENDPOINT["contract"], S15_ENDPOINT["contract_file_sha256"])
    s15c = read(s15c_ref["path"])
    require(s15c["contract_sha256"] == S15_ENDPOINT["contract_sha256"], "S15 contract canonical digest")
    endpoint = {"file": pinned(S15_ENDPOINT["file"], S15_ENDPOINT["sha256"]),
                "sidecar": ref(S15_ENDPOINT["file"].with_suffix(".pth.json")), "contract": s15c_ref,
                **{k: S15_ENDPOINT[k] for k in ("contract_sha256", "formal_identity", "schema", "ema_weights_sha256",
                                                "train_canvas")}}
    references = {k: (pinned(p, sha) if sha else ref(p)) for k, (p, sha) in REFERENCES.items()}
    result = read(references["S15_EMA_result"]["path"])
    require(result["predictions"] == {"600": references["S15_EMA_top600"], "300": references["S15_EMA_top300"]}
            and result["weights_sha256"] == S15_ENDPOINT["ema_weights_sha256"] and result["kind"] == "EMA"
            and result["checkpoint"] == endpoint["file"] and result["images"] == 548 and result["batch_size"] == BATCH,
            "certified S15 EMA readout binding")
    # fixed head split from manifest identities only; written before any new feature or label exists
    data = S.train_dataset(s15c["data"]["train_core"], hw=Q.H_HW, seed=0)
    ids = [r["stable_image_id"] for r in data.records]
    split = Q.split_ids(ids)
    split_path = ROUND / f"{attempt}-SPLIT.json"
    write_new(split_path, {"rule": "sha256('sqc-v1|'+stable_image_id) digest bytes ascending, ties by id; first 512 "
                                   "head validation, rest head training", "train_core_manifest_sha256": data.manifest_sha256,
                           "images": len(ids), "validation": split["validation"], "train": split["train"],
                           "validation_sha256": canonical_sha(split["validation"]),
                           "train_sha256": canonical_sha(split["train"]),
                           "order_rule": "per epoch e in 1..8: sha256('sqc-order-v1|'+str(e)+'|'+id) ascending",
                           "check_images": split["train"][:CHECK_IMAGES]})
    budget = byte_budget()
    free = shutil.disk_usage(ROUND).free
    require(free > budget["total_estimate_bytes"] + budget["reserve_bytes"], f"disk free {free} below budget")
    root = ROUND / attempt
    value = {"version": CONTRACT_VERSION, "attempt": attempt, "contract_path": str(ROUND / f"{attempt}-CONTRACT.json"),
             "attempt_root": str(root), "task": task, "workload": WORKLOAD, "library_version": Q.VERSION,
             "source_root": str(release / "source"), "source_manifest": manifest_ref, "source_commit": manifest["commit"],
             "endpoint": endpoint, "S15_activation_checkpoint_flag": s15c["workload"]["activation_checkpoint"],
             "references": references, "split": ref(split_path),
             "data": s15c["data"], "development_ids": s15c["development_ids"],
             "evaluation_policy": s15c["evaluation_policy"], "environment": s15c["environment"],
             "decision_rule": {"G": [list(x) for x in G_CELLS], "retention": {"cells": [list(x) for x in RETENTION],
                                                                             "fraction": RETENTION_FRACTION},
                               "I_B2_minus_B1": [list(x) for x in I_CELLS], "labels": DECISIONS,
                               "net_reference": "native A0@L e120 EMA", "cross_scale_reference": "B1"},
             "byte_budget": budget, "disk_free_at_contract": free, "python": PYTHON, "gpu_uuid": GPU}
    value["contract_sha256"] = canonical_sha(value)
    path = Path(value["contract_path"])
    write_new(path, value)
    print(json.dumps({"file": ref(path), "contract_sha256": value["contract_sha256"], "split": value["split"]}), flush=True)


def load_contract(path):
    path = Path(path).absolute()
    c = read(path)
    body = dict(c)
    require(body.pop("contract_sha256") == canonical_sha(body), "contract canonical digest")
    require(c["version"] == CONTRACT_VERSION and c["workload"] == WORKLOAD and c["contract_path"] == str(path),
            "contract version/workload/path")
    require(c["task"] == pinned(ROUND / TASK_NAME, TASK_SHA), "task changed")
    manifest = read(pinned(c["source_manifest"]["path"], c["source_manifest"]["sha256"])["path"])
    require(manifest["commit"] == c["source_commit"] and Path(c["source_root"]) == ROOT, "stage must run from the release")
    require(c["endpoint"]["file"]["sha256"] == S15_ENDPOINT["sha256"]
            and c["endpoint"]["ema_weights_sha256"] == S15_ENDPOINT["ema_weights_sha256"], "endpoint pin")
    require(ref(c["split"]["path"]) == c["split"], "split file changed")
    return c


def split_of(c):
    return read(c["split"]["path"])


# ================================================================ models and data
def _xc(c):
    return {"endpoints": {"S15": c["endpoint"]}, "S15_activation_checkpoint_flag": c["S15_activation_checkpoint_flag"]}


def load_models(c, device):
    """S15 e120 EMA at H (native canvas) and the same learned tensors at L (native L caches); frozen eval."""
    torch = _torch()
    import s15_cross_scale as X
    xc = _xc(c)
    payload = X.endpoint_payload(xc, "S15")
    state = X.ema_state(payload, xc, "S15")
    mh, rh = X.build_cell(xc, "HH", state)
    ml, rl = X.build_cell(xc, "HL", state)
    require(rh["loaded_model_sha256"] == c["endpoint"]["ema_weights_sha256"], "H model is not the certified EMA")
    require(rh["non_size_tensor_sha256"] == rl["non_size_tensor_sha256"], "H/L instances hold different learned tensors")
    del payload, state
    gc.collect()
    for m in (mh, ml):
        m.eval().requires_grad_(False)
        m.to(device)
    return mh, ml, {"H": rh, "L": rl}


def criterion_recipe():
    from sparse_rtdetr.protocol_r18.model import Config, build
    from sparse_rtdetr.research import shared_query_cls as Q
    _, criterion, info = build(Config(seed=0, input_hw=Q.H_HW), synthetic=True)
    rec = criterion.recipe()
    require(rec["alpha"] == 0.75 and rec["gamma"] == 2.0 and rec["weights"]["loss_vfl"] == 1.0
            and info["config"]["weights"]["loss_vfl"] == 1.0, f"native VFL recipe differs: {rec}")
    return criterion, {"alpha": criterion.alpha, "gamma": criterion.gamma,
                       "loss_weight": criterion.weight_dict["loss_vfl"], "native_recipe": rec}


def train_data(c):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import shared_query_cls as Q
    data = S.train_dataset(c["data"]["train_core"], hw=Q.H_HW, seed=0)
    split = split_of(c)
    require(sorted(split["train"] + split["validation"]) == sorted(r["stable_image_id"] for r in data.records)
            and split["train_core_manifest_sha256"] == data.manifest_sha256, "split membership differs from train_core")
    return data


def dev_data(c):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import shared_query_cls as Q
    data = S.development(c["data"]["development"], Q.H_HW)
    require([r["stable_image_id"] for r in data.records] == c["development_ids"], "development members or order")
    return data


def roots_of(data):
    if hasattr(data, "images_path"):
        return Path(data.images_path), Path(data.annotations_path)
    return Path(data.spec["roots"]["images"]), Path(data.spec["roots"]["annotations"])


@contextlib.contextmanager
def gt_guard(data):
    """Refuse every annotation read while forwards run; count image reads."""
    import sparse_rtdetr.localization.data as LD
    images, annotations = (p.resolve() for p in roots_of(data))
    original = LD.read_bound
    counts = {"image_reads": 0, "annotation_reads": 0}

    def guarded(path, *args, **kwargs):
        p = Path(path).resolve()
        if p == annotations or annotations in p.parents:
            counts["annotation_reads"] += 1
            raise RuntimeError("ground truth read on the forward path")
        counts["image_reads"] += 1
        return original(path, *args, **kwargs)

    LD.read_bound = guarded
    had_raw = hasattr(data, "raw")
    if had_raw:
        def raw(*_a, **_k):
            counts["annotation_reads"] += 1
            raise RuntimeError("ground truth parse on the forward path")
        data.raw = raw
    try:
        yield counts
    finally:
        LD.read_bound = original
        if had_raw:
            del data.raw


def load_pixels(data, row, resizers):
    """Original RGB bytes (bound SHA/size) -> one direct resize per canvas, as the certified evaluator."""
    from PIL import Image
    from torchvision.transforms.v2 import functional as TF
    import sparse_rtdetr.localization.data as LD
    images, _ = roots_of(data)
    raw = LD.read_bound(images / Path(row["relative_path"]).name, row["image_sha256"], row["image_size_bytes"])
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert("RGB")
        image.load()
    require((image.width, image.height) == (row["width"], row["height"]), "decoded size differs from manifest")
    out = {k: TF.pil_to_tensor(r(image)).float() / 255. for k, r in resizers.items()}
    image.close()
    return out


def resizers():
    from sparse_rtdetr.localization.data import RawAugmentation
    from sparse_rtdetr.research import shared_query_cls as Q
    return {"H": RawAugmentation(Q.H_HW).resize, "L": RawAugmentation(Q.L_HW).resize}


def readout_batch(mh, ml, xh, xl, device):
    """One H full forward with the captured last-layer entry and one L partial readout of the same images."""
    torch = _torch()
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.research import shared_query_cls as Q
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device.type, enabled=False), forward_guards(mh):
        out, entry, z = Q.h_forward(mh, xh.to(device))
        sync()
        t1 = time.perf_counter()
        zl, memory, shapes = Q.cross_readout(mh, ml, entry, xl.to(device))
        sync()
        t2 = time.perf_counter()
    s, b = out["pred_logits"], out["pred_boxes"]
    n = xh.shape[0]
    require(s.shape == (n, 300, 10) and b.shape == (n, 300, 4) and z.shape == zl.shape == (n, 300, 256)
            and s.dtype == b.dtype == z.dtype == zl.dtype == torch.float32, "readout schema")
    require(bool(torch.isfinite(z).all() and torch.isfinite(zl).all() and torch.isfinite(s).all()), "nonfinite readout")
    require([list(x) for x in entry.spatial_shapes] == Q.level_shapes(Q.H_HW)
            and [list(x) for x in shapes] == Q.level_shapes(Q.L_HW), "H/L spatial shapes")
    return {"s": s, "b": b, "z": z, "zl": zl, "entry": entry, "timing": {"H_full": t1 - t0, "L_partial_and_layer": t2 - t1}}


def frozen_digest(*models):
    from sparse_rtdetr.research import shared_query_cls as Q
    return [Q.digest_tensors(m.state_dict().items()) for m in models]


def assert_frozen(models, digests, device, rng):
    torch = _torch()
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    require(frozen_digest(*models) == digests, "detector state changed")
    require(all(not m.training for mm in models for m in mm.modules()), "detector left eval mode")
    require(all(p.grad is None and not p.requires_grad for mm in models for p in mm.parameters()), "detector gradient")
    if rng is not None:
        require(ck.exact_equal(ck.capture_rng(device), rng), "RNG consumed by frozen forwards")


def labels_for(criterion, data, row, s, b):
    """Offline (after forwards): raw annotation -> H training-geometry target -> native match on (s_H, b_H)."""
    import sparse_rtdetr.localization.data as LD
    from sparse_rtdetr.localization.data import RawAugmentation, parse_raw
    from sparse_rtdetr.research import shared_query_cls as Q
    _, annotations = roots_of(data)
    raw = LD.read_bound(annotations / Path(row["annotation_relative_path"]).name, row["annotation_sha256"],
                        row["annotation_size_bytes"])
    parsed = parse_raw(raw, row["stable_image_id"], expected_rows=row["raw_annotation_rows"])
    target = Q.h_training_target(RawAugmentation(Q.H_HW), parsed, (row["height"], row["width"]))
    lab = Q.query_labels(criterion, s, b, target)
    lab["target_sha256"] = Q.target_digest(target)
    lab["gt_count_after_sanitize"] = len(target["labels"])
    lab["raw_valid_gt"] = len(parsed["gt"])
    return lab, target, parsed


def setup(device_name):
    torch = _torch()
    from r25_run import setup as native_setup
    if device_name == "cuda:0":
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU, "bound GPU not selected")
    else:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU stage must hide CUDA")
    native_setup(device_name)
    return torch.device(device_name)


def _rss():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _peak(device):
    torch = _torch()
    if device.type != "cuda":
        return {"peak_rss_bytes": _rss()}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "peak_rss_bytes": _rss()}


def stage(fn):
    """Uniform receipt: RESULT.json on success, STOP.json with traceback on any failure."""
    def run(c, out, *args, **kwargs):
        torch = _torch()
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        started, cpu0 = time.perf_counter(), time.process_time()
        try:
            result = fn(c, out, *args, **kwargs)
            result.update(status="PASS", contract_sha256=c["contract_sha256"],
                          wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu0)
            write_new(out / "RESULT.json", result)
            return result
        except BaseException as exc:
            peak = {}
            with contextlib.suppress(Exception):
                if torch.cuda.is_initialized():
                    peak = {"peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            write_new(out / "STOP.json", {"status": "STOP_CUDA_OOM" if isinstance(exc, torch.cuda.OutOfMemoryError)
                                          else "STOP", "stage": fn.__name__, **peak,
                                          "wall_seconds": time.perf_counter() - started,
                                          "error": str(exc)[:4000], "traceback": traceback.format_exc()})
            raise
    run.__name__ = fn.__name__
    return run


def receipt(path, c):
    p = Path(path) / "RESULT.json"
    r = read(p)
    require(r["status"] == "PASS" and r["contract_sha256"] == c["contract_sha256"], f"receipt {p}")
    return r, ref(p)


# ================================================================ unit tests (CPU)
@stage
def unit(c, out):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([PYTHON, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_shared_query_cls.py"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    (out / "pytest.log").write_text(proc.stdout + proc.stderr)
    require(proc.returncode == 0, f"unit tests failed:\n{proc.stdout[-3000:]}")
    tail = [l for l in proc.stdout.splitlines() if "passed" in l]
    return {"pytest": tail[-1] if tail else proc.stdout[-200:], "log": ref(out / "pytest.log"),
            "file": ref(ROOT / "tests/test_shared_query_cls.py")}


# ================================================================ CPU check (real path, first 8 head-training images)
@stage
def cpu_check(c, out):
    torch = _torch()
    from torchvision.transforms.v2 import functional as TF
    from PIL import Image
    from sparse_rtdetr.localization.data import RawAugmentation
    from sparse_rtdetr.r0.evaluation import rank_candidates
    from sparse_rtdetr.research import shared_query_cls as Q
    import s15_cross_scale as X
    device = setup("cpu")
    split = split_of(c)
    data = train_data(c)
    rows = [data.by_id[i] for i in split["check_images"]]
    require([r["stable_image_id"] for r in rows] == split["train"][:CHECK_IMAGES], "check images are the first 8")
    # identity refusals of the certified loader
    refused = []
    xc = _xc(c)
    for label, path in (("e105 in the e120 slot", S15R / "formal-002/train-S15/checkpoint-epoch-105.pth"),
                        ("A0 e120 in the S15 slot", R25 / "formal-001/train-A0/checkpoint-epoch-120.pth")):
        try:
            X.endpoint_payload(xc, "S15", path)
        except (ValueError, RuntimeError, KeyError):
            refused.append(label)
    require(len(refused) == 2, "certified loader accepted a wrong endpoint")
    mh, ml, load = load_models(c, device)
    digests = frozen_digest(mh, ml)
    criterion, recipe = criterion_recipe()
    rs = resizers()
    pixel_checks, forwards, label_rows = [], [], []
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    rng = ck.capture_rng(device)
    native = {}
    with gt_guard(data) as guard:
        try:
            import sparse_rtdetr.localization.data as LD
            LD.read_bound(roots_of(data)[1] / Path(rows[0]["annotation_relative_path"]).name, "0" * 64, 0)
        except RuntimeError:
            pass
        require(guard["annotation_reads"] == 1, "GT guard is not armed")
        for k in range(0, CHECK_IMAGES, BATCH):
            batch = rows[k:k + BATCH]
            px = [load_pixels(data, r, rs) for r in batch]
            for r, p in zip(batch, px):
                import sparse_rtdetr.localization.data as LD
                images, _ = roots_of(data)
                with Image.open(io.BytesIO(LD.read_bound(images / Path(r["relative_path"]).name, r["image_sha256"],
                                                         r["image_size_bytes"]))) as im:
                    im = im.convert("RGB")
                    for key, hw in (("H", Q.H_HW), ("L", Q.L_HW)):
                        direct = TF.pil_to_tensor(TF.resize(im, list(hw))).float() / 255.
                        require(torch.equal(direct, p[key]) and tuple(p[key].shape) == (3, *hw), "pixel mapping")
                pixel_checks.append({"image": r["stable_image_id"], "original_hw": [r["height"], r["width"]],
                                     "H": list(p["H"].shape), "L": list(p["L"].shape), "equal_direct_resize": True})
            xh = torch.stack([p["H"] for p in px])
            xl = torch.stack([p["L"] for p in px])
            res = readout_batch(mh, ml, xh, xl, device)
            with torch.inference_mode():
                z_replay = Q.last_layer(mh, res["entry"])
                s_head = Q.native_logits(mh, res["z"])
            require(torch.equal(z_replay, res["z"]) and torch.equal(s_head, res["s"]), "same-H replay differs")
            require(res["entry"].target.shape == (BATCH, 300, 256) and res["entry"].reference.shape == (BATCH, 300, 1, 4),
                    "Q300 entry shape")
            for i, r in enumerate(batch):
                native[r["stable_image_id"]] = (res["s"][i].clone(), res["b"][i].clone(), res["z"][i].clone(),
                                                res["zl"][i].clone())
            forwards.append({"images": [r["stable_image_id"] for r in batch], "timing": res["timing"],
                             "query_identity": res["entry"].identity(),
                             "z_H_equals_replay": True, "s_H_equals_native_score_head": True,
                             "L_spatial_shapes": Q.level_shapes(Q.L_HW), "H_spatial_shapes": Q.level_shapes(Q.H_HW)})
        guard_counts = dict(guard)
    require(guard_counts["annotation_reads"] == 1 and guard_counts["image_reads"] == 2 * CHECK_IMAGES,
            f"forward phase read GT or wrong image count: {guard_counts}")
    assert_frozen((mh, ml), digests, device, rng)
    # offline labels (after all forwards)
    init = Q.initial_state()
    heads = {arm: Q.new_head(init) for arm in Q.ARMS}
    require(Q.head_digest(heads["B1"]) == Q.head_digest(heads["B2"]), "arm initial bytes")
    for r in rows:
        s, b, z, zl = native[r["stable_image_id"]]
        lab, target, parsed = labels_for(criterion, data, r, s, b)
        import sparse_rtdetr.localization.data as LD
        images, _ = roots_of(data)
        with Image.open(io.BytesIO(LD.read_bound(images / Path(r["relative_path"]).name, r["image_sha256"],
                                                 r["image_size_bytes"]))) as im:
            im = im.convert("RGB")
            real = Q.h_training_target(RawAugmentation(Q.H_HW), parsed, (r["height"], r["width"]), image=im)
        require(Q.target_digest(real) == lab["target_sha256"], "blank-canvas label path differs from the image path")
        orig = {g["id"]: g["box"] for g in parsed["gt"]}
        boxes = torch.tensor([orig[g] for g in target["gt_ids"]], dtype=torch.float64).reshape(-1, 4)
        w, h = r["width"], r["height"]
        expect = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 1] + boxes[:, 3]) / 2 / h,
                              (boxes[:, 2] - boxes[:, 0]) / w, (boxes[:, 3] - boxes[:, 1]) / h), 1)
        clipped = (boxes[:, 0] < 0) | (boxes[:, 1] < 0) | (boxes[:, 2] > w) | (boxes[:, 3] > h)
        err = (target["boxes"].double() - expect).abs()[~clipped]
        require(err.numel() == 0 or float(err.max()) < 2e-6, f"normalized GT differs from original/size: {float(err.max())}")
        with torch.no_grad():
            for arm in Q.ARMS:
                new = Q.apply_head(heads[arm], s[None], z[None], Q.source_features(arm, z, zl)[None], b[None])
                require(torch.equal(new[0], s), "zero head is not the identity")
            preds = rank_candidates(s, b, r, 600)
        require(len(preds) == 600, "ranking")
        background = int((lab["classes"] == 10).sum())
        label_rows.append({"image": r["stable_image_id"], "raw_valid_gt": lab["raw_valid_gt"],
                           "gt_after_sanitize": lab["gt_count_after_sanitize"],
                           "matched_queries": len(lab["matched_queries"]), "background_queries": background,
                           "queries": 300, "classes_supervised": 10, "loss_elements": int(lab["keep"].numel()),
                           "masked_negative_elements": int((~lab["keep"]).sum()),
                           "max_norm_box_error_vs_original": float(err.max()) if err.numel() else 0.0,
                           "clipped_gt": int(clipped.sum()), "target_sha256": lab["target_sha256"]})
        require(background + len(lab["matched_queries"]) == 300 and len(lab["matched_queries"]) == min(300, lab["num_gt"]),
                "query/background accounting")
    return {"images": [r["stable_image_id"] for r in rows], "refused": refused, "load": load, "criterion": recipe,
            "pixels": pixel_checks, "forwards": forwards, "labels": label_rows, "GT_guard": guard_counts,
            "detector_frozen": True, "zero_head_identity": True, "initial_head_sha256": Q.head_digest(heads["B1"]),
            "counts": {"H_full_forwards_cpu": CHECK_IMAGES // BATCH, "L_partial_cpu": CHECK_IMAGES // BATCH,
                       "replays": CHECK_IMAGES // BATCH, "head_updates": 0}, **_peak(device)}


# ================================================================ GPU check (same 8 images; 6 head updates)
def _labels_batch(labels, ids, device):
    torch = _torch()
    return {"classes": torch.stack([labels[i]["classes"] for i in ids]).to(device),
            "quality": torch.stack([labels[i]["quality"] for i in ids]).to(device),
            "keep": torch.stack([labels[i]["keep"] for i in ids]).to(device),
            "num_gt": sum(labels[i]["num_gt"] for i in ids)}


@stage
def gpu_check(c, out):
    torch = _torch()
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.research import shared_query_cls as Q
    device = setup("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    split = split_of(c)
    data = train_data(c)
    rows = [data.by_id[i] for i in split["check_images"]]
    mh, ml, load = load_models(c, device)
    digests = frozen_digest(mh, ml)
    criterion, recipe = criterion_recipe()  # builds reseed python/numpy; snapshot after
    rng = ck.capture_rng(device)
    rs = resizers()
    counts = {"H_full_forwards": 0, "L_partial_readouts": 0, "same_H_replays": 0, "head_updates": 0}
    native, batches = {}, []
    with gt_guard(data) as guard:
        for k in range(0, CHECK_IMAGES, BATCH):
            batch = rows[k:k + BATCH]
            px = [load_pixels(data, r, rs) for r in batch]
            xh, xl = torch.stack([p["H"] for p in px]), torch.stack([p["L"] for p in px])
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), forward_guards(mh):
                plain = mh(xh.to(device))  # certified evaluator arithmetic, no capture hooks
            counts["H_full_forwards"] += 1
            res = readout_batch(mh, ml, xh, xl, device)
            counts["H_full_forwards"] += 1
            counts["L_partial_readouts"] += 1
            with torch.inference_mode(), torch.autocast("cuda", enabled=False), forward_guards(mh):
                replay = Q.last_layer(mh, res["entry"])
                counts["same_H_replays"] += 1
                x = res["entry"].reference.squeeze(2).clip(0., 1.)
                inv = torch.log(x.clip(min=1e-5) / (1 - x).clip(min=1e-5))
                rebuilt = torch.sigmoid(mh.decoder.dec_bbox_head[Q.LAST_LAYER](res["z"]) + inv)
                again = mh(xh.to(device))  # after the L calls: H path must be unpolluted
                counts["H_full_forwards"] += 1
                head_logits = Q.native_logits(mh, res["z"])
            checks = {"hooked_equals_plain_logits": torch.equal(plain["pred_logits"], res["s"]),
                      "hooked_equals_plain_boxes": torch.equal(plain["pred_boxes"], res["b"]),
                      "replay_equals_native_z": torch.equal(replay, res["z"]),
                      "score_head_equals_native_logits": torch.equal(head_logits, res["s"]),
                      "box_rebuilt_equals_native": torch.equal(rebuilt, res["b"]),
                      "H_after_L_logits_equal": torch.equal(again["pred_logits"], res["s"]),
                      "H_after_L_boxes_equal": torch.equal(again["pred_boxes"], res["b"]),
                      "z_L_differs_from_z_H": not torch.equal(res["zl"], res["z"])}
            require(all(checks.values()), f"GPU readout check failed: {checks}")
            for i, r in enumerate(batch):
                native[r["stable_image_id"]] = {k2: res[k2][i].detach().clone() for k2 in ("s", "b", "z", "zl")}
            batches.append({"images": [r["stable_image_id"] for r in batch], "checks": checks, "timing": res["timing"],
                            "query_identity": res["entry"].identity(),
                            "z_L_minus_z_H_max_abs": float((res["zl"] - res["z"]).abs().max())})
        guard_counts = dict(guard)
    require(guard_counts["annotation_reads"] == 0, "GT read during GPU forwards")
    assert_frozen((mh, ml), digests, device, rng)
    labels = {}
    for r in rows:
        n = native[r["stable_image_id"]]
        lab, _, _ = labels_for(criterion, data, r, n["s"].cpu(), n["b"].cpu())
        labels[r["stable_image_id"]] = lab
    ids = [r["stable_image_id"] for r in rows]
    steps = [ids[:BATCH], ids[BATCH:]]

    def feature_batch(arm, sel):
        st = lambda key: torch.stack([native[i][key] for i in sel])
        return {"s_H": st("s"), "b_H": st("b"), "z_H": st("z"), "z_source": Q.source_features(arm, st("z"), st("zl")),
                **_labels_batch(labels, sel, device)}

    init = Q.initial_state()
    arms = {}
    for arm in Q.ARMS:
        head = Q.new_head(init, device)
        opt = Q.new_optimizer(head)
        initial = Q.head_digest(head)
        r1 = Q.head_step(head, opt, feature_batch(arm, steps[0]), recipe)
        counts["head_updates"] += 1
        saved_path = out / f"{arm}-after-step1.pth"
        state = Q.head_state(head, opt, {"arm": arm, "global_step": 1, "next_step_images": steps[1]},
                             ck.capture_rng(device))
        with saved_path.open("xb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        r2 = Q.head_step(head, opt, feature_batch(arm, steps[1]), recipe)
        counts["head_updates"] += 1
        direct = {"head": Q.head_digest(head), "optimizer": Q.optimizer_digest(opt), "result": r2,
                  "rng": ck.capture_rng(device)}
        loaded = torch.load(saved_path, map_location="cpu", weights_only=True)
        require(loaded["cursor"]["global_step"] == 1 and loaded["cursor"]["arm"] == arm, "saved cursor")
        head2 = Q.new_head(loaded["head"], device)
        opt2 = Q.new_optimizer(head2)
        opt2.load_state_dict(loaded["optimizer"])
        ck.restore_rng(loaded["rng"], device)
        r2b = Q.head_step(head2, opt2, feature_batch(arm, loaded["cursor"]["next_step_images"]), recipe)
        counts["head_updates"] += 1
        replay = {"head": Q.head_digest(head2), "optimizer": Q.optimizer_digest(opt2), "result": r2b,
                  "rng": ck.capture_rng(device)}
        exact = {"head": replay["head"] == direct["head"], "optimizer": replay["optimizer"] == direct["optimizer"],
                 "result": replay["result"] == direct["result"], "rng": ck.exact_equal(replay["rng"], direct["rng"]),
                 "cursor": loaded["cursor"]["next_step_images"] == steps[1]}
        require(all(exact.values()), f"{arm} resume replay differs: {exact}")
        arms[arm] = {"initial_head_sha256": initial, "step1": r1, "step2": r2, "replayed_step2": r2b,
                     "final_head_sha256": direct["head"], "final_optimizer_sha256": direct["optimizer"],
                     "resume_exact": exact, "saved_state": ref(saved_path)}
        del head, opt, head2, opt2
    require(arms["B1"]["initial_head_sha256"] == arms["B2"]["initial_head_sha256"], "arm initial bytes differ")
    require(counts["head_updates"] == 6, "head update count")
    assert_frozen((mh, ml), digests, device, None)
    return {"images": ids, "load": load, "criterion": recipe, "batches": batches, "GT_guard": guard_counts,
            "labels": {i: {k2: labels[i][k2] for k2 in ("num_gt", "matched_queries", "target_sha256")} for i in ids},
            "arms": arms, "counts": counts, "detector_updates": 0, "detector_frozen": True, **_peak(device)}


# ================================================================ formal train_core extraction
def _save(path, value):
    torch = _torch()
    with Path(path).open("xb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    return ref(path)


@stage
def extract(c, out, role):
    torch = _torch()
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import shared_query_cls as Q
    require(role == "train", "only the train_core cache is extracted here (development in the dev stage)")
    device = setup("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    data = train_data(c)
    rows = data.records
    n = len(rows)
    require(n == Q.TRAIN_CORE_IMAGES, "train_core size")
    budget = c["byte_budget"]
    require(shutil.disk_usage(out).free > budget["total_estimate_bytes"] + budget["reserve_bytes"], "disk below budget")
    mh, ml, load = load_models(c, device)
    digests = frozen_digest(mh, ml)
    rng = ck.capture_rng(device)
    rs = resizers()
    z_h = torch.empty((n, 300, 256), dtype=torch.float32)
    z_l = torch.empty((n, 300, 256), dtype=torch.float32)
    s_h = torch.empty((n, 300, 10), dtype=torch.float32)
    b_h = torch.empty((n, 300, 4), dtype=torch.float32)
    identity = []
    timing = {"data": 0.0, "H_full": 0.0, "L_partial_and_layer": 0.0, "d2h": 0.0}
    counts = {"H_full_forwards": 0, "L_partial_readouts": 0, "same_H_replays": 0}
    with gt_guard(data) as guard:
        for k in range(0, n, BATCH):
            batch = rows[k:k + BATCH]
            t0 = time.perf_counter()
            px = [load_pixels(data, r, rs) for r in batch]
            xh, xl = torch.stack([p["H"] for p in px]), torch.stack([p["L"] for p in px])
            timing["data"] += time.perf_counter() - t0
            res = readout_batch(mh, ml, xh, xl, device)
            counts["H_full_forwards"] += 1
            counts["L_partial_readouts"] += 1
            for key, v in res["timing"].items():
                timing[key] += v
            if k == 0:
                with torch.inference_mode():
                    require(torch.equal(Q.last_layer(mh, res["entry"]), res["z"]), "same-H replay (first batch)")
                counts["same_H_replays"] += 1
            t1 = time.perf_counter()
            sl = slice(k, k + len(batch))
            z_h[sl], z_l[sl], s_h[sl], b_h[sl] = res["z"].cpu(), res["zl"].cpu(), res["s"].cpu(), res["b"].cpu()
            timing["d2h"] += time.perf_counter() - t1
            qid = res["entry"].identity()
            for i, r in enumerate(batch):
                identity.append({"image": r["stable_image_id"], "query_entry_sha256": qid[i],
                                 "native_sha256": Q.digest_tensors([("s", s_h[k + i]), ("b", b_h[k + i])])})
            del res
            if counts["H_full_forwards"] % 100 == 0:
                print(json.dumps({"event": "extract", "batches": counts["H_full_forwards"]}), flush=True)
        guard_counts = dict(guard)
    require(guard_counts["annotation_reads"] == 0 and guard_counts["image_reads"] == n, f"GT guard {guard_counts}")
    require(counts["H_full_forwards"] == counts["L_partial_readouts"] == math.ceil(n / BATCH) == 1218, "batch count")
    assert_frozen((mh, ml), digests, device, rng)
    ids = [r["stable_image_id"] for r in rows]
    t2 = time.perf_counter()
    files = {"FEATURES.pt": _save(out / "FEATURES.pt", {"ids": ids, "z_H": z_h, "z_L": z_l}),
             "NATIVE.pt": _save(out / "NATIVE.pt", {"ids": ids, "s_H": s_h, "b_H": b_h})}
    write_new(out / "IDENTITY.json", identity)
    files["IDENTITY.json"] = ref(out / "IDENTITY.json")
    return {"role": "train", "images": n, "files": files, "load": load, "timing": timing, "counts": counts,
            "save_seconds": time.perf_counter() - t2, "GT_guard": guard_counts, "detector_updates": 0,
            "tensor_shapes": {"z_H": list(z_h.shape), "z_L": list(z_l.shape), "s_H": list(s_h.shape), "b_H": list(b_h.shape)},
            "digests": {"z_H": Q.digest_tensors([("z_H", z_h)]), "z_L": Q.digest_tensors([("z_L", z_l)]),
                        "s_H": Q.digest_tensors([("s_H", s_h)]), "b_H": Q.digest_tensors([("b_H", b_h)])},
            **_peak(device)}


# ================================================================ offline labels (CPU)
@stage
def labels(c, out, extract_dir):
    torch = _torch()
    from sparse_rtdetr.research import shared_query_cls as Q
    setup("cpu")
    ex, ex_ref = receipt(extract_dir, c)
    native = torch.load(pinned(ex["files"]["NATIVE.pt"]["path"], ex["files"]["NATIVE.pt"]["sha256"])["path"],
                        map_location="cpu", weights_only=True)
    data = train_data(c)
    ids = native["ids"]
    require(ids == [r["stable_image_id"] for r in data.records], "native cache order")
    criterion, recipe = criterion_recipe()
    n = len(ids)
    classes = torch.empty((n, 300), dtype=torch.int64)
    quality = torch.empty((n, 300), dtype=torch.float32)
    keep = torch.empty((n, 300, 10), dtype=torch.bool)
    bucket = torch.empty((n, 300), dtype=torch.int8)
    num_gt = torch.empty(n, dtype=torch.int64)
    rows_out = []
    for k, r in enumerate(data.records):
        lab, _, _ = labels_for(criterion, data, r, native["s_H"][k], native["b_H"][k])
        classes[k], quality[k], keep[k], bucket[k] = lab["classes"], lab["quality"], lab["keep"], lab["bucket"]
        num_gt[k] = lab["num_gt"]
        rows_out.append({"image": r["stable_image_id"], "num_gt": lab["num_gt"], "raw_valid_gt": lab["raw_valid_gt"],
                         "matched_queries": lab["matched_queries"], "matched_gt_ids": lab["matched_gt_ids"],
                         "target_sha256": lab["target_sha256"]})
    files = {"LABELS.pt": _save(out / "LABELS.pt", {"ids": ids, "classes": classes, "quality": quality, "keep": keep,
                                                     "bucket": bucket, "num_gt": num_gt})}
    write_new(out / "LABEL_ROWS.json", rows_out)
    files["LABEL_ROWS.json"] = ref(out / "LABEL_ROWS.json")
    matched = int((classes < 10).sum())
    return {"extract": ex_ref, "files": files, "criterion": recipe, "images": n, "GT_after_sanitize": int(num_gt.sum()),
            "raw_valid_GT": sum(x["raw_valid_gt"] for x in rows_out), "matched_queries": matched,
            "background_queries": n * 300 - matched, "masked_negative_elements": int((~keep).sum()),
            "matched_by_size": {name: int((bucket == i).sum()) for i, name in enumerate(Q.SIZE_BUCKETS)},
            "images_without_GT": int((num_gt == 0).sum()),
            "matching": "native HungarianMatcher on native (s_H, b_H), once per image, CPU FP32",
            "digests": {"classes": Q.digest_tensors([("c", classes)]), "quality": Q.digest_tensors([("q", quality)]),
                        "keep": Q.digest_tensors([("k", keep)])}, "peak_rss_bytes": _rss()}


# ================================================================ head fitting (one arm per process)
def evaluate_head(head, idx_list, tensors, recipe, device):
    torch = _torch()
    from sparse_rtdetr.research import shared_query_cls as Q
    zh, zs, sh, bh, cl, qu, ke, bu, ng = tensors
    totals = {"terms": 0.0, "GT": 0, "chunks": 0, "chunk_loss_sum": 0.0}
    deco = {}
    with torch.no_grad():
        for k in range(0, len(idx_list), Q.STEP_IMAGES):
            idx = torch.tensor(idx_list[k:k + Q.STEP_IMAGES], device=device)
            logits = Q.apply_head(head, sh[idx], zh[idx], zs[idx], bh[idx])
            g = int(ng[idx].sum())
            loss = Q.vfl_loss(logits, cl[idx], qu[idx], ke[idx], g, recipe["alpha"], recipe["gamma"], recipe["loss_weight"])
            totals["terms"] += float(Q.vfl_terms(logits, cl[idx], qu[idx], ke[idx], recipe["alpha"], recipe["gamma"]).sum())
            totals["GT"] += g
            totals["chunks"] += 1
            totals["chunk_loss_sum"] += float(loss)
            for key, v in Q.loss_decomposition(logits, cl[idx], qu[idx], ke[idx], bu[idx], recipe["alpha"],
                                               recipe["gamma"]).items():
                deco[key] = deco.get(key, 0) + v
    return {"mean_step_style_loss": totals["chunk_loss_sum"] / totals["chunks"],
            "global_loss": totals["terms"] / max(totals["GT"], 1) * recipe["loss_weight"],
            "GT": totals["GT"], "chunks": totals["chunks"], "decomposition_unnormalised": deco,
            "per_matched_query": {b: deco[b] / deco[b + "_queries"] if deco[b + "_queries"] else None
                                  for b in Q.SIZE_BUCKETS},
            "background_per_query": deco["background"] / deco["background_queries"]}


@stage
def fit(c, out, arm, extract_dir, labels_dir):
    torch = _torch()
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import shared_query_cls as Q
    require(arm in Q.ARMS, "arm")
    device = setup("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    ex, ex_ref = receipt(extract_dir, c)
    lb, lb_ref = receipt(labels_dir, c)
    require(lb["extract"] == ex_ref, "labels of another extraction")

    def load(name, r):
        return torch.load(pinned(r["files"][name]["path"], r["files"][name]["sha256"])["path"], map_location="cpu",
                          weights_only=True, mmap=True)
    feats, native, labs = load("FEATURES.pt", ex), load("NATIVE.pt", ex), load("LABELS.pt", lb)
    ids = native["ids"]
    require(feats["ids"] == ids == labs["ids"], "cache id order")
    pos = {i: k for k, i in enumerate(ids)}
    zh, zl = feats["z_H"].to(device), feats["z_L"].to(device)
    sh, bh = native["s_H"].to(device), native["b_H"].to(device)
    cl, qu, ke, bu, ng = (labs[k].to(device) for k in ("classes", "quality", "keep", "bucket", "num_gt"))
    zs = Q.source_features(arm, zh, zl)
    tensors = (zh, zs, sh, bh, cl, qu, ke, bu, ng)
    load_seconds = time.perf_counter() - t0
    _, recipe = criterion_recipe()
    split = split_of(c)
    steps = Q.schedule(split["train"])
    require(len(steps) == 2184, "schedule length")
    val_idx = [pos[i] for i in split["validation"]]
    head = Q.new_head(Q.initial_state(), device)
    opt = Q.new_optimizer(head)
    initial_sha = Q.head_digest(head)
    rng0 = ck.capture_rng(device)
    v0 = evaluate_head(head, val_idx, tensors, recipe, device)
    log = out / "STEPS.jsonl"
    epoch_states, losses = {}, []
    t1 = time.perf_counter()
    for epoch, k, gstep, sel in steps:
        idx = torch.tensor([pos[i] for i in sel], device=device)
        batch = {"s_H": sh[idx], "b_H": bh[idx], "z_H": zh[idx], "z_source": zs[idx], "classes": cl[idx],
                 "quality": qu[idx], "keep": ke[idx], "num_gt": int(ng[idx].sum())}
        r = Q.head_step(head, opt, batch, recipe)
        losses.append(r["loss"])
        journal(log, {"arm": arm, "epoch": epoch, "step_in_epoch": k, "global_step": gstep, "images": len(sel), **r})
        if gstep == len(steps) or steps[gstep][0] != epoch:
            epoch_states[str(epoch)] = _save(out / f"HEAD-e{epoch}.pth", Q.head_state(
                head, opt, {"arm": arm, "epoch": epoch, "global_step": gstep}, ck.capture_rng(device)))
    train_seconds = time.perf_counter() - t1
    require(len(losses) == 2184 and all(math.isfinite(x) for x in losses), "update count / finite losses")
    require(ck.exact_equal(ck.capture_rng(device), rng0), "head fitting consumed RNG (no stochastic op expected)")
    vf = evaluate_head(head, val_idx, tensors, recipe, device)
    per_epoch = {str(e): sum(losses[(e - 1) * 273:e * 273]) / 273 for e in range(1, 9)}
    return {"arm": arm, "extract": ex_ref, "labels": lb_ref, "criterion": recipe, "optimizer": Q.OPTIMIZER,
            "clip_norm": Q.CLIP_NORM, "updates": len(losses), "detector_updates": 0,
            "initial_head_sha256": initial_sha, "final_head_sha256": Q.head_digest(head),
            "final_optimizer_sha256": Q.optimizer_digest(opt), "final": epoch_states["8"], "epoch_states": epoch_states,
            "endpoint": "epoch 8 final state (fixed; no selection)", "steps_log": ref(log),
            "train_loss_mean_per_epoch": per_epoch, "first_step_loss": losses[0], "last_step_loss": losses[-1],
            "validation_initial": v0, "validation_final": vf, "load_seconds": load_seconds,
            "train_seconds": train_seconds, "head_parameters": sum(p.numel() for p in head.parameters()),
            **_peak(device)}


# ================================================================ development endpoints (548; one shared forward)
def load_heads(c, fit_dirs, device):
    torch = _torch()
    from sparse_rtdetr.research import shared_query_cls as Q
    heads, refs = {}, {}
    for arm in Q.ARMS:
        r, rr = receipt(fit_dirs[arm], c)
        require(r["arm"] == arm and r["updates"] == 2184, f"{arm} fit receipt")
        state = torch.load(pinned(r["final"]["path"], r["final"]["sha256"])["path"], map_location="cpu", weights_only=True)
        require(state["cursor"] == {"arm": arm, "epoch": 8, "global_step": 2184}, f"{arm} final cursor")
        head = Q.new_head(state["head"], device).eval().requires_grad_(False)
        require(Q.head_digest(head) == r["final_head_sha256"], f"{arm} final head bytes")
        heads[arm], refs[arm] = head, {"fit": rr, "final": r["final"], "head_sha256": r["final_head_sha256"],
                                       "initial_head_sha256": r["initial_head_sha256"]}
    require(refs["B1"]["initial_head_sha256"] == refs["B2"]["initial_head_sha256"], "arms started from different bytes")
    return heads, refs


def ranked(endpoint_logits, b, rows):
    from sparse_rtdetr.r0.evaluation import rank_candidates
    six = [rank_candidates(lg, bx, row, 600) for lg, bx, row in zip(endpoint_logits, b, rows)]
    return {600: [p for g in six for p in g], 300: [p for g in six for p in g[:300]]}, six


def list_digest(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@stage
def dev(c, out, fit_dirs):
    torch = _torch()
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import shared_query_cls as Q
    device = setup("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    heads, head_refs = load_heads(c, fit_dirs, device)
    zero = Q.new_head(Q.initial_state(), device).eval().requires_grad_(False)
    data = dev_data(c)
    rows = data.records
    mh, ml, load = load_models(c, device)
    digests = frozen_digest(mh, ml)
    rng = ck.capture_rng(device)
    rs = resizers()
    n = len(rows)
    feats = {k: torch.empty((n, 300, d), dtype=torch.float32) for k, d in (("z_H", 256), ("z_L", 256), ("s_H", 10),
                                                                           ("b_H", 4), ("s_B1", 10), ("s_B2", 10))}
    lists = {e: {600: [], 300: []} for e in ("B0", "B1", "B2")}
    per_image = {e: [] for e in lists}
    counts = {"H_full_forwards": 0, "L_partial_readouts": 0, "same_H_replays": 0}
    box_checks = 0
    with gt_guard(data) as guard:
        for k in range(0, n, BATCH):
            batch = rows[k:k + BATCH]
            px = [load_pixels(data, r, rs) for r in batch]
            xh, xl = torch.stack([p["H"] for p in px]), torch.stack([p["L"] for p in px])
            res = readout_batch(mh, ml, xh, xl, device)
            counts["H_full_forwards"] += 1
            counts["L_partial_readouts"] += 1
            with torch.inference_mode():
                if k == 0:
                    require(torch.equal(Q.last_layer(mh, res["entry"]), res["z"]), "same-H replay (dev first batch)")
                    counts["same_H_replays"] += 1
                for arm in Q.ARMS:  # zero-step heads are the identity on every development batch
                    z0 = Q.apply_head(zero, res["s"], res["z"], Q.source_features(arm, res["z"], res["zl"]), res["b"])
                    require(torch.equal(z0, res["s"]), "zero head is not the identity")
                s1 = Q.apply_head(heads["B1"], res["s"], res["z"], Q.source_features("B1", res["z"], res["zl"]), res["b"])
                s2 = Q.apply_head(heads["B2"], res["s"], res["z"], Q.source_features("B2", res["z"], res["zl"]), res["b"])
            b = res["b"].cpu()
            logits = {"B0": res["s"].cpu(), "B1": s1.cpu(), "B2": s2.cpu()}
            sl = slice(k, k + len(batch))
            feats["z_H"][sl], feats["z_L"][sl] = res["z"].cpu(), res["zl"].cpu()
            feats["s_H"][sl], feats["b_H"][sl], feats["s_B1"][sl], feats["s_B2"][sl] = (logits["B0"], b, logits["B1"],
                                                                                         logits["B2"])
            for e, lg in logits.items():
                pr, six = ranked(lg, b, batch)
                for budget in (600, 300):
                    lists[e][budget].extend(pr[budget])
                per_image[e].extend(list_digest(g) for g in six)
                for img, row in enumerate(batch):  # every listed box is the native H box of its query
                    bb = b[img]
                    xy = torch.cat((bb[:, :2] - bb[:, 2:] / 2, bb[:, :2] + bb[:, 2:] / 2), -1) * \
                        torch.tensor([row["width"], row["height"]] * 2, dtype=torch.float32)
                    for p in six[img]:
                        require(p["bbox_xyxy"] == [float(v) for v in xy[p["query_id"]].tolist()], "box differs from b_H")
                        box_checks += 1
            del res
        guard_counts = dict(guard)
    require(guard_counts["annotation_reads"] == 0 and guard_counts["image_reads"] == n, f"GT guard {guard_counts}")
    require(counts["H_full_forwards"] == 137, "dev batch count")
    assert_frozen((mh, ml), digests, device, rng)
    b0 = {}
    for budget in (600, 300):  # B0 must reproduce the certified S15@H EMA lists exactly
        r = c["references"][f"S15_EMA_top{budget}"]
        cert = read(pinned(r["path"], r["sha256"])["path"])
        mism = sum(1 for a, x in zip(cert, lists["B0"][budget]) if a != x) + abs(len(cert) - len(lists["B0"][budget]))
        b0[str(budget)] = {"records": len(cert), "mismatched_records": mism, "exact": mism == 0}
        del cert
    require(all(v["exact"] for v in b0.values()), f"B0 differs from the certified S15@H lists: {b0}")
    files = {}
    for e in ("B1", "B2"):
        for budget in (600, 300):
            p = out / f"{e}-top{budget}-predictions.json"
            write_new(p, lists[e][budget])
            files[f"{e}_{budget}"] = ref(p)
    files["DEV_FEATURES.pt"] = _save(out / "DEV_FEATURES.pt", {"ids": [r["stable_image_id"] for r in rows], **feats})
    write_new(out / "PER_IMAGE_LIST_SHA256.json", per_image)
    files["PER_IMAGE_LIST_SHA256.json"] = ref(out / "PER_IMAGE_LIST_SHA256.json")
    changed = {}
    for e in ("B1", "B2"):
        d = feats[f"s_{e}"] - feats["s_H"]
        top = {}
        for budget in (600, 300):
            base = {(p["image_id"], p["query_id"], p["category_id"]) for p in lists["B0"][budget]}
            new = {(p["image_id"], p["query_id"], p["category_id"]) for p in lists[e][budget]}
            top[str(budget)] = {"entries_not_in_B0": len(new - base), "B0_entries_dropped": len(base - new)}
        changed[e] = {"mean_abs_delta": float(d.abs().mean()), "max_abs_delta": float(d.abs().max()),
                      "mean_delta": float(d.mean()), "list_membership_vs_B0": top}
    return {"heads": head_refs, "load": load, "files": files, "B0_vs_certified": b0, "counts": counts,
            "box_checks_records": box_checks, "boxes": "every listed box equals b_H of its query (checked per record)",
            "logit_changes": changed, "GT_guard": guard_counts, "detector_updates": 0, **_peak(device)}


# ================================================================ timing (independent per-endpoint deployment passes)
@stage
def timing(c, out, fit_dirs, dev_dir):
    torch = _torch()
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.research import shared_query_cls as Q
    device = setup("cuda:0")
    dv, dv_ref = receipt(dev_dir, c)
    f = dv["files"]["PER_IMAGE_LIST_SHA256.json"]
    expected = read(pinned(f["path"], f["sha256"])["path"])
    heads, head_refs = load_heads(c, fit_dirs, device)
    data = dev_data(c)
    rows = data.records
    mh, ml, load = load_models(c, "cpu")
    rs_all = resizers()
    batches = [rows[k:k + BATCH] for k in range(0, len(rows), BATCH)]
    sync = lambda: torch.cuda.synchronize(device)
    passes = {}
    for endpoint in ("B0", "B1", "B2"):
        mh.to(device)
        ml.to(device if endpoint == "B2" else "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        rs = rs_all if endpoint == "B2" else {"H": rs_all["H"]}
        record = {"warmup": [], "measured": []}
        digests = []
        with gt_guard(data) as guard:
            for phase, bl in (("warmup", batches[:WARMUPS]), ("measured", batches)):
                for batch in bl:
                    sync()
                    t = {"start": time.perf_counter()}
                    px = [load_pixels(data, r, rs) for r in batch]
                    xh = torch.stack([p["H"] for p in px]).to(device)
                    xl = torch.stack([p["L"] for p in px]).to(device) if endpoint == "B2" else None
                    sync()
                    t["input"] = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", enabled=False), forward_guards(mh):
                        if endpoint == "B0":
                            o = mh(xh)
                            s, b = o["pred_logits"], o["pred_boxes"]
                            sync()
                            t["H"] = t["L"] = t["head"] = time.perf_counter()
                        else:
                            o, entry, z = Q.h_forward(mh, xh)
                            s, b = o["pred_logits"], o["pred_boxes"]
                            sync()
                            t["H"] = time.perf_counter()
                            zl = Q.cross_readout(mh, ml, entry, xl)[0] if endpoint == "B2" else z
                            sync()
                            t["L"] = time.perf_counter()
                            s = Q.apply_head(heads[endpoint], s, z, zl, b)
                            sync()
                            t["head"] = time.perf_counter()
                    s, b = s.cpu(), b.cpu()
                    t["d2h"] = time.perf_counter()
                    _, six = ranked(s, b, batch)
                    t["end"] = time.perf_counter()
                    record[phase].append({"input_decode_resize_h2d": t["input"] - t["start"], "H_full": t["H"] - t["input"],
                                          "L_partial_and_layer": t["L"] - t["H"], "head": t["head"] - t["L"],
                                          "d2h": t["d2h"] - t["head"], "list_build": t["end"] - t["d2h"],
                                          "total": t["end"] - t["start"]})
                    if phase == "measured":
                        digests.extend(list_digest(g) for g in six)
            reads = dict(guard)
        require(reads["annotation_reads"] == 0, "GT read in timing")
        steady = {k: sum(r[k] for r in record["measured"]) for k in record["measured"][0]}
        passes[endpoint] = {"steady_seconds": steady, "steady_per_image_ms": 1000 * steady["total"] / len(rows),
                            "warmup_seconds": sum(r["total"] for r in record["warmup"]),
                            "warmup_batches": len(record["warmup"]), "measured_batches": len(record["measured"]),
                            "lists_equal_dev_stage": digests == expected[endpoint],
                            "mismatched_images": sum(1 for a, x in zip(digests, expected[endpoint]) if a != x),
                            **_peak(device)}
    detector = sum(p.numel() for p in mh.parameters())
    nbytes = lambda m: sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))
    return {"dev": dv_ref, "heads": head_refs, "passes": passes, "protocol": WORKLOAD["timing"], "order": "B0, B1, B2",
            "parameters": {"detector": detector, "head": Q.HEAD_PARAMETERS, "B0": detector,
                           "B1": detector + Q.HEAD_PARAMETERS, "B2": detector + Q.HEAD_PARAMETERS,
                           "B2_note": "same detector weights; this implementation holds a second canvas instance",
                           "H_instance_bytes": nbytes(mh), "L_instance_bytes": nbytes(ml)},
            "ratios_vs_B0": {e: passes[e]["steady_seconds"]["total"] / passes["B0"]["steady_seconds"]["total"]
                             for e in passes},
            "forwards": {"H_full": 3 * (WARMUPS + 137), "L_partial": WARMUPS + 137, "head_calls": 2 * (WARMUPS + 137)}}


# ================================================================ scoring and fixed gates (CPU)
def references(c):
    from infer_seed import bridge_values
    r = c["references"]
    values, gt = {}, {}
    for name in ("A0@L", "S15@H"):
        values[name], gt[name] = {}, {}
        for b in ("600", "300"):
            m = read(pinned(r[f"{name}_{b}"]["path"], r[f"{name}_{b}"]["sha256"])["path"])
            values[name][b] = bridge_values(m)
            gt[name][b] = {k: m["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
    require(gt["A0@L"] == gt["S15@H"], "reference GT sets differ")
    comp = read(pinned(r["S15_comparison"]["path"], r["S15_comparison"]["sha256"])["path"])
    require(comp["scores"]["A0"]["EMA"] == values["A0@L"] and comp["scores"]["S15"]["EMA"] == values["S15@H"],
            "S15 comparison binding")
    per_class = {"A0@L": {row["category_id"]: row for row in comp["per_class_APL_top600"]["EMA"]["A0"]},
                 "S15@H": {row["category_id"]: row for row in comp["per_class_APL_top600"]["EMA"]["S15"]}}
    gains = {f"top{b}_{m}": 100. * (values["S15@H"][str(b)][m] - values["A0@L"][str(b)][m]) for b, m in RETENTION}
    return values, gt["A0@L"], per_class, gains


def diff(a, b):
    return {x: {m: 100. * (a[x][m] - b[x][m]) for m in METRICS} for x in ("600", "300")}


def gate_g(vx, a0, gains):
    d = diff(vx, a0)
    cells = {f"top{b}_{m}": {"difference_pp": d[str(b)][m], "minimum_pp": lim, "passed": d[str(b)][m] >= lim}
             for b, m, lim in G_CELLS}
    ret = {}
    for b, m in RETENTION:
        key = f"top{b}_{m}"
        need = RETENTION_FRACTION * gains[key]
        ret[key] = {"minus_A0_pp": d[str(b)][m], "S15_minus_A0_pp": gains[key], "required_pp": need,
                    "retained_fraction": d[str(b)][m] / gains[key], "passed": d[str(b)][m] >= need}
    ok = all(v["passed"] for v in cells.values()) and all(v["passed"] for v in ret.values())
    return {"cells": cells, "retention": ret, "passed": ok,
            "failed": [k for k, v in cells.items() if not v["passed"]] + [k + "_retention" for k, v in ret.items()
                                                                          if not v["passed"]]}


def gate_i(v2, v1):
    d = diff(v2, v1)
    cells = {f"top{b}_{m}": {"difference_pp": d[str(b)][m], "minimum_pp": lim, "passed": d[str(b)][m] >= lim}
             for b, m, lim in I_CELLS}
    return {"cells": cells, "passed": all(v["passed"] for v in cells.values()),
            "failed": [k for k, v in cells.items() if not v["passed"]]}


@stage
def score(c, out, dev_dir):
    from infer_seed import _group, _prefix, bridge_values
    from s15_formal_eval import per_class_large
    from sparse_rtdetr.r0.evaluation import POLICY, bridge_inputs, score_budget
    setup("cpu")
    policy = c["evaluation_policy"]
    require(ref(policy["path"]) == policy and read(policy["path"]) == POLICY, "certified evaluation policy differs")
    dv, dv_ref = receipt(dev_dir, c)
    ref_values, ref_gt, ref_pc, gains = references(c)
    data = dev_data(c)
    rows, ids = data.records, c["development_ids"]
    parsed = [data.raw(row) for row in rows]
    preds = {}
    pre = {}
    t0 = time.perf_counter()
    for e in ("B1", "B2"):  # precheck before any COCOeval: schema, order, prefix, GT set
        preds[e] = {}
        for b in (600, 300):
            r = dv["files"][f"{e}_{b}"]
            preds[e][b] = read(pinned(r["path"], r["sha256"])["path"])
        grouped = {b: _group(preds[e][b], ids, b, whole=True, exact=True) for b in (600, 300)}
        _prefix(grouped[600], grouped[300], ids, whole=True)
        counts = {}
        for b in (600, 300):
            _, _, cnt = bridge_inputs(rows, parsed, preds[e][b])
            counts[str(b)] = cnt
            require({k: cnt[k] for k in ("GT_before", "GT_after")} == ref_gt[str(b)], f"{e} top{b} GT set differs")
        pre[e] = {"grouped_exact_and_prefix": True, "spatial_filter_counts": counts}
    precheck_seconds = time.perf_counter() - t0
    values, metrics, calls, per_class = {}, {}, [], {}
    for e in ("B1", "B2"):
        values[e], metrics[e] = {}, {}
        for b in (600, 300):
            t = time.perf_counter()
            m = score_budget(rows, parsed, preds[e][b], c["source_root"])
            m["SQC_binding"] = {"contract_sha256": c["contract_sha256"], "endpoint": e, "budget": b,
                                "predictions": dv["files"][f"{e}_{b}"], "dev": dv_ref,
                                "head_sha256": dv["heads"][e]["head_sha256"]}
            require({k: m["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")} == ref_gt[str(b)], "GT set")
            p = out / f"{e}-top{b}-metrics.json"
            write_new(p, m)
            metrics[e][str(b)] = ref(p)
            values[e][str(b)] = bridge_values(m)
            calls.append({"endpoint": e, "budget": b, "seconds": time.perf_counter() - t,
                          "spatial_filter_counts": m["spatial_filter_counts"]})
            if b == 600:
                t = time.perf_counter()
                rows_pc = per_class_large(rows, parsed, preds[e][b], m["COCO_bridge"])
                require(all(x["GT"] == ref_pc["A0@L"][x["category_id"]]["GT"] for x in rows_pc), "per-class GT")
                per_class[e] = {"rows": rows_pc, "seconds": time.perf_counter() - t}
    require(len(calls) == 4 and len(per_class) == 2, "fixed score/COCOeval count")
    allv = {**values, "B0=S15@H": ref_values["S15@H"], "A0@L": ref_values["A0@L"]}
    g = {e: gate_g(values[e], ref_values["A0@L"], gains) for e in ("B1", "B2")}
    i = gate_i(values["B2"], values["B1"])
    decision = DECISIONS["G_and_I"] if g["B2"]["passed"] and i["passed"] else \
        DECISIONS["G_only"] if g["B2"]["passed"] else DECISIONS["not_G"]
    classes = []
    for row in per_class["B2"]["rows"]:
        k = row["category_id"]
        apl = {"B2": row["APL"], "B1": next(x["APL"] for x in per_class["B1"]["rows"] if x["category_id"] == k),
               "B0": ref_pc["S15@H"][k]["APL"], "A0@L": ref_pc["A0@L"][k]["APL"]}
        pp = lambda a, b: None if apl[a] is None or apl[b] is None else 100. * (apl[a] - apl[b])
        classes.append({"category_id": k, "name": CLASS_NAMES[k], "GT_large": row["GT"], "APL": apl,
                        "B2-B1_pp": pp("B2", "B1"), "B2-B0_pp": pp("B2", "B0"), "B1-B0_pp": pp("B1", "B0"),
                        "B2-A0_pp": pp("B2", "A0@L"), "B1-A0_pp": pp("B1", "A0@L"),
                        "macro_contribution_B2-B1_pp": None if pp("B2", "B1") is None else pp("B2", "B1") / 10.})
    differences = {"B2-A0@L": diff(values["B2"], ref_values["A0@L"]), "B1-A0@L": diff(values["B1"], ref_values["A0@L"]),
                   "B2-B1": diff(values["B2"], values["B1"]), "B2-B0": diff(values["B2"], ref_values["S15@H"]),
                   "B1-B0": diff(values["B1"], ref_values["S15@H"]),
                   "B0-A0@L (reference)": diff(ref_values["S15@H"], ref_values["A0@L"])}
    return {"dev": dv_ref, "precheck": pre, "precheck_seconds": precheck_seconds, "scores": allv, "metrics": metrics,
            "differences_pp": differences, "G": g, "I": i, "scientific_decision": decision,
            "B1_passes_G": g["B1"]["passed"], "S15_gain_pp": gains, "per_class_APL_top600": classes,
            "per_class_COCOeval_seconds": {e: v["seconds"] for e, v in per_class.items()}, "score_calls": calls,
            "score_budget_calls": len(calls), "extra_COCOeval_calls": len(per_class), "GT_counts": ref_gt,
            "score_units": "0..1 COCO bridge", "delta_units": "pp", "peak_rss_bytes": _rss()}


# ================================================================ detached supervisor
def _env(gpu):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES=GPU if gpu else "", CUBLAS_WORKSPACE_CONFIG=":4096:8", MKL_THREADING_LAYER="GNU",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONUNBUFFERED="1")
    return env


def chain(c):
    a = Path(c["attempt_root"])
    con = ["--contract", c["contract_path"]]
    return [
        ("gpu-check", True, ["gpu-check", *con, "--out", str(a / "gpu-check")]),
        ("extract-train", True, ["extract", *con, "--role", "train", "--out", str(a / "extract-train")]),
        ("labels", False, ["labels", *con, "--extract", str(a / "extract-train"), "--out", str(a / "labels")]),
        ("fit-B1", True, ["fit", *con, "--arm", "B1", "--extract", str(a / "extract-train"), "--labels",
                          str(a / "labels"), "--out", str(a / "fit-B1")]),
        ("fit-B2", True, ["fit", *con, "--arm", "B2", "--extract", str(a / "extract-train"), "--labels",
                          str(a / "labels"), "--out", str(a / "fit-B2")]),
        ("dev", True, ["dev", *con, "--fit-B1", str(a / "fit-B1"), "--fit-B2", str(a / "fit-B2"), "--out", str(a / "dev")]),
        ("score", False, ["score", *con, "--dev", str(a / "dev"), "--out", str(a / "score")]),
        ("timing", True, ["timing", *con, "--fit-B1", str(a / "fit-B1"), "--fit-B2", str(a / "fit-B2"), "--dev",
                          str(a / "dev"), "--out", str(a / "timing")]),
    ]


def run(c, gates, start_at=None):
    a = Path(c["attempt_root"])
    a.mkdir(exist_ok=True)
    for g in gates:
        receipt(g, c)
    log = a / "STAGES.jsonl"
    steps = chain(c)
    names = [s[0] for s in steps]
    begin = names.index(start_at) if start_at else 0
    for name, _, _ in steps[:begin]:
        receipt(a / name, c)
    journal(log, {"event": "supervisor_start", "pid": os.getpid(), "time": time.time(), "start_at": names[begin],
                  "gates": [str(g) for g in gates]})
    for name, gpu, args in steps[begin:]:
        t = time.time()
        journal(log, {"event": "stage_start", "stage": name, "time": t})
        with (a / f"{name}.log").open("x") as f:
            proc = subprocess.run([PYTHON, "-B", str(ROOT / "tools/scale_baseline/sqc_run.py"), *args], cwd=ROOT,
                                  env=_env(gpu), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        journal(log, {"event": "stage_end", "stage": name, "returncode": proc.returncode, "seconds": time.time() - t})
        if proc.returncode != 0:
            write_new(a / "STOP.json", {"status": "STOP", "stage": name, "returncode": proc.returncode, "time": time.time()})
            return 1
    write_new(a / "CLOSED.json", {"status": "CLOSED_PASS", "stages": names[begin:], "time": time.time()})
    journal(log, {"event": "supervisor_end", "time": time.time()})
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command")
    p.add_argument("--release")
    p.add_argument("--attempt")
    p.add_argument("--contract")
    p.add_argument("--out")
    p.add_argument("--role")
    p.add_argument("--arm")
    p.add_argument("--extract")
    p.add_argument("--labels")
    p.add_argument("--fit-B1")
    p.add_argument("--fit-B2")
    p.add_argument("--dev")
    p.add_argument("--gates", nargs="*", default=[])
    p.add_argument("--start-at")
    a = p.parse_args()
    if a.command == "freeze":
        return freeze(a.release)
    if a.command == "contract":
        return contract(a.release, a.attempt)
    c = load_contract(a.contract)
    fits = {"B1": a.fit_B1, "B2": a.fit_B2}
    commands = {"unit": lambda: unit(c, a.out), "cpu-check": lambda: cpu_check(c, a.out),
                "gpu-check": lambda: gpu_check(c, a.out), "extract": lambda: extract(c, a.out, a.role),
                "labels": lambda: labels(c, a.out, a.extract),
                "fit": lambda: fit(c, a.out, a.arm, a.extract, a.labels), "dev": lambda: dev(c, a.out, fits),
                "timing": lambda: timing(c, a.out, fits, a.dev), "score": lambda: score(c, a.out, a.dev),
                "run": lambda: sys.exit(run(c, a.gates, a.start_at))}
    require(a.command in commands, f"unknown command {a.command}")
    commands[a.command]()


if __name__ == "__main__":
    main()
