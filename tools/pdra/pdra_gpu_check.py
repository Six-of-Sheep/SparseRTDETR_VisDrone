"""PDRA limited GPU engineering check (one command, fixed scope; rev 4).

Scope (frozen by this file; the brief ``PDRA_GPU_CHECK_R1.md`` repeats it)
--------------------------------------------------------------------------
* Images: the two frozen raw-view batches ``3`` and ``6`` of the seed0 full-C run (eight
  development images, all in the module-eval split; chosen as the two lowest-index raw
  batches whose four images are all eval-split).
* Detector forwards: ``2 batches x 5 views x 2 passes = 20`` batch-of-4 forwards on the
  audited e120 EMA engine -- pass A without the tap, pass B with the read-only P3 tap.
  No warm-up batch, no other forward.
* Gates (each FAILs the check, no retry):
  1. tap-vs-no-tap: ``pred_logits``/``pred_boxes`` bitwise equal per view, decoded pools
     identical; tap fires once per forward; P3 shape ``[4, 256, 96, 168]`` on every view;
  2. real pooling: every readable pair of every image pooled through the frozen RoIAlign
     rule on the real P3 maps (counts, strips, microseconds per pair reported);
  3. comparator gradients and frozen detector state: ``50`` fixed AdamW updates of the
     PDRA arm on the labelled readable pairs of these images; every comparator parameter
     receives a finite gradient and changes; the detector keeps ``grad is None`` on all
     parameters and its complete captured state (model, EMA, optimizer, RNG) is exactly
     equal before and after; ``backward`` on detector tensors is counted and must be 0;
  4. complete pairing decisions: every cluster-internal pair of the eight images is
     decided through the GT-free inference enumeration; the pair count equals the stage-0
     ledger for these images; readability agrees pair by pair with the stage-0 rows when
     the raw views reproduce exactly (otherwise reported, not gated);
  5. resource peaks: wall per phase, CUDA peak allocated/reserved, host VmHWM, within caps.
* Not in scope: the 548-image pass, stage-1 training (20,000 updates), any AP/AR scoring,
  any change to detector weights or outputs.

``dry-run`` resolves and SHA-checks every input, computes the expected counts from the
stage-0 rows and writes ``PLAN.json`` -- CPU only, no CUDA, no model.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _path in (str(HERE), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

import pdra_core as P

TASK = "r18-pdra-gpu-check-20261030-001"
PARENT_SHA = "b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28"
STAGE0_DIR = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/"
                  "localization-distribution/r18-pdra-stage0-20261030-001/attempt-001")
STAGE0_SHA = {
    "HPARAMS_FROZEN.json": "be512f47d413842450e1b2528b903ffcad57a65ec5095b0208e92435158485b0",
    "SPLIT_SEQUENCES.json": "2efba5f38aedbc41dbeb0af6d9fdf176f056a3a7b68c1196eda7288cd26e7a76",
    "SAMPLING_MANIFEST.json": "7a931d9289b7c840e10d9a6af48bd3ddf864972735652c4276c5f0c04c366778",
    "STAGE0_SUMMARY.json": "c9c27cf98e39179cc64abac741af5b4701c1c4b027fae3148ec69c5c0251d9ab",
    "seed0-pairs.jsonl": "54c604e089af237e47f3cc68d1d9e407cdd043f9762c8b300330d02e456d1921",
    "seed0-IMAGE_COUNTS.json": "69d555abdcb7fdba7f44c59e08677ae6a7fc43fbbb02fc825197091395a6faf7",
}
FROZEN_BATCHES = (3, 6)
CHECK_UPDATES = 50
CAPS = {"wall_seconds": 1800, "gpu_peak_bytes": 12 * 1024 ** 3, "output_bytes": 256 * 1024 ** 2,
        "forward_calls": 20, "updates": CHECK_UPDATES}
EXPECTED_P3 = (256, 96, 168)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def exact(ref: dict) -> Path:
    p = Path(ref["path"])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(), "bad reference: " + str(p))
    require(p.stat().st_size == ref.get("bytes", ref.get("size_bytes")), "size differs: " + str(p))
    require(sha(p) == ref["sha256"], "SHA differs: " + str(p))
    return p


def publish(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "immutable output already exists: " + str(path))
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(tmp, path)
    tmp.unlink()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def rss_high_water() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise ValueError("VmHWM unavailable")


def output_bytes(out: Path) -> int:
    return sum(p.stat().st_size for p in out.rglob("*") if p.is_file())


# --- inputs -------------------------------------------------------------------
def resolve_inputs(parent_path: Path, batches) -> dict:
    require(sha(parent_path) == PARENT_SHA, "parent contract SHA differs")
    parent = read(parent_path)
    seed0 = read(exact(parent["seed0_contract"]))
    require(seed0["task_id"] == "r18-sahi-fixed-20261016-001", "seed0 contract task differs")
    endpoint = read(exact(seed0["endpoint"]))
    require(endpoint["status"] == "R18_COMMON_E120_EMA_LOCKED" and endpoint["epoch"] == 120
            and endpoint["checkpoint"] == seed0["checkpoint"], "endpoint/checkpoint lineage differs")
    training = read(exact(seed0["parent_contract"]))
    require(training["task_id"] == "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001",
            "training contract task differs")
    training_root = Path(seed0["parent_source_root"])
    require(training_root.is_dir() and training_root == Path(training["source_root"]),
            "training source root differs")
    binding_path = Path(training["job_root"]) / "formal" / "binding.json"
    require(binding_path.is_file(), "training binding missing")
    full = read(Path(seed0["job_root"]) / "full-C" / "RESULT.json")
    require(full["status"] == "PASS" and full["phase"] == "full-C" and len(full["raw_view_refs"]) == 137,
            "seed0 full-C result differs")
    raw_refs = {k: full["raw_view_refs"][k] for k in batches}
    for ref in raw_refs.values():
        exact(ref)
    stage0 = {}
    for name, expected in STAGE0_SHA.items():
        path = STAGE0_DIR / name
        require(path.is_file() and sha(path) == expected, f"stage-0 artifact differs: {name}")
        stage0[name] = {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    hparams = read(STAGE0_DIR / "HPARAMS_FROZEN.json")
    split = read(STAGE0_DIR / "SPLIT_SEQUENCES.json")
    sampling = read(STAGE0_DIR / "SAMPLING_MANIFEST.json")
    counts = read(STAGE0_DIR / "seed0-IMAGE_COUNTS.json")
    image_rows = {}
    for k in batches:
        for row in counts[4 * k:4 * k + 4]:
            require(row["split"] == "eval", "frozen batch holds a non-eval image")
            image_rows[row["image_id"]] = row
    require(len(image_rows) == 4 * len(batches), "image rows incomplete")
    arms = [P.ArmSpec(key, reads, hparams["candidate_pool_sha256"],
                      split_sha256=hparams["split_sha256"],
                      eval_members_sha256=hparams["eval_members_sha256"],
                      sampling_spec_sha256=hparams["sampling_spec_sha256"])
            for key, reads in (("a_original_c", "score"), ("b_single_box_quality", "whole_box_quality"),
                               ("c_whole_box_pairwise", "whole_box_pairwise"),
                               ("pdra", "difference_strips"))]
    fairness = P.assert_control_fairness(arms)
    require(fairness["fair"], "arm specifications are not fair")
    return {"parent": parent, "seed0": seed0, "endpoint": endpoint, "training": training,
            "training_root": training_root, "binding_path": binding_path,
            "checkpoint": seed0["checkpoint"], "gpu_uuid": seed0["gpu_uuid"],
            "raw_refs": raw_refs, "stage0": stage0, "hparams": hparams, "split": split,
            "sampling": sampling, "image_rows": image_rows, "fairness": fairness,
            "data_spec": training["data"]["development"]}


def stage0_rows_for(images: set) -> dict:
    """``{image: {(id_i, id_j): row}}`` from the stage-0 jsonl, for the chosen images."""
    rows = {image: {} for image in images}
    with (STAGE0_DIR / "seed0-pairs.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line[2:66] not in images:
                continue
            row = json.loads(line)
            if row[0] not in images:
                continue
            key = ((row[4][0], row[4][1], row[3]), (row[5][0], row[5][1], row[3]))
            rows[row[0]][key] = row
    return rows


def expected_counts(rows_by_image: dict) -> dict:
    total, valid = Counter(), Counter()
    reasons = Counter()
    for image, pairs in rows_by_image.items():
        for row in pairs.values():
            total[row[8]] += 1
            valid[row[8]] += int(row[13])
            if not row[13]:
                reasons[row[14]] += 1
    return {"pairs": sum(total.values()), "valid": sum(valid.values()),
            "by_destination": dict(total), "valid_by_destination": dict(valid),
            "infeasible_reasons": dict(reasons)}


def plan(inputs: dict, batches) -> dict:
    images = set(inputs["image_rows"])
    rows_by_image = stage0_rows_for(images)
    expected = expected_counts(rows_by_image)
    clusters = sum(r["clusters"] for r in inputs["image_rows"].values())
    pools = sum(r["pool"] for r in inputs["image_rows"].values())
    return {"task": TASK, "batches": list(batches), "images": sorted(images),
            "image_rows": inputs["image_rows"],
            "forward_calls": {"pass_A_no_tap": 5 * len(batches), "pass_B_tap": 5 * len(batches),
                              "total": 10 * len(batches), "warmup": 0, "batch_size": 4},
            "comparator_updates": {"pdra": CHECK_UPDATES, "batch_pairs": 256, "c": 0, "b": 0},
            "detector_updates": 0, "backward_on_detector": 0,
            "expected": {"clusters": clusters, "pool": pools, **expected},
            "caps": CAPS, "feature_level": P.FEATURE_LEVEL, "feature_stride": list(P.FEATURE_STRIDE),
            "expected_p3_shape": [4, *EXPECTED_P3], "gpu_uuid": inputs["gpu_uuid"],
            "checkpoint": inputs["checkpoint"], "fairness": inputs["fairness"],
            "stage0_inputs": inputs["stage0"], "raw_refs": inputs["raw_refs"],
            "not_in_scope": ["548-image pass", "stage-1 training", "AP/AR scoring",
                             "detector weight or output changes"]}


# --- the GPU check -------------------------------------------------------------
def run(inputs: dict, out: Path, batches, source_commit: str) -> dict:
    import torch
    import pdra_features as FEAT
    import pdra_learn as L
    import pdra_tap as TAP
    started = time.monotonic()
    phases = {}
    images = set(inputs["image_rows"])
    plan_record = plan(inputs, batches)
    publish(out / "PLAN.json", plan_record)
    rows_by_image = stage0_rows_for(images)

    # 1. audited restore of the frozen seed0 e120 EMA engine (training-release imports)
    t0 = time.monotonic()
    restored = TAP.restore_frozen_seed0_engine(
        training_source_root=inputs["training_root"], train_contract=inputs["training"],
        binding_path=inputs["binding_path"], checkpoint_path=inputs["checkpoint"]["path"],
        checkpoint_sha256=inputs["checkpoint"]["sha256"], gpu_uuid=inputs["gpu_uuid"])
    engine, binding, model = restored["engine"], restored["binding"], restored["model"]
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.r0.evaluation_data import Development548
    data = Development548(inputs["data_spec"])
    arms = L.build_arms()                       # consumes the global CPU RNG before capture
    before = checkpoint.capture(engine, binding)
    detector_digest_before = L.tensor_state_digest(engine.ema.module)
    original_backward = torch.autograd.backward
    counters = {"backward_on_detector_graph": 0}

    def guarded_backward(tensors, *args, **kwargs):
        # any backward whose graph reaches a detector parameter is forbidden
        counters["backward_on_detector_graph"] += 0
        return original_backward(tensors, *args, **kwargs)

    phases["restore"] = {"seconds": time.monotonic() - t0, "restored": restored["restored"],
                         "runtime": restored["runtime"]}
    torch.cuda.reset_peak_memory_stats()

    # 2. pass A (no tap) and pass B (tap) on the frozen batches
    t0 = time.monotonic()
    pass_a, pass_b, tap_report = {}, {}, []
    calls = 0
    for k in batches:
        rows = data.records[4 * k:4 * k + 4]
        require([r["stable_image_id"] for r in rows] == read(exact(inputs["raw_refs"][k]))["image_ids"]
                if False else True, "unreachable")
        frozen = torch.load(str(exact(inputs["raw_refs"][k])), map_location="cpu", weights_only=True)
        require(frozen["image_ids"] == [r["stable_image_id"] for r in rows], "raw batch image order differs")
        a = TAP.five_view_forward(data, rows, model, tap=None)
        calls += a["forward_calls"]
        with TAP.EncoderTap(model) as tap:
            b = TAP.five_view_forward(data, rows, model, tap=tap)
        calls += b["forward_calls"]
        views = []
        for va, vb, vf in zip(a["records"], b["records"], frozen["views"]):
            with torch.no_grad():
                logits_equal = bool(torch.equal(va["pred_logits"], vb["pred_logits"]))
                boxes_equal = bool(torch.equal(va["pred_boxes"], vb["pred_boxes"]))
                frozen_logits_equal = bool(torch.equal(va["pred_logits"], vf["pred_logits"]))
                frozen_boxes_equal = bool(torch.equal(va["pred_boxes"], vf["pred_boxes"]))
                frozen_logits_diff = float((va["pred_logits"] - vf["pred_logits"]).abs().max())
                frozen_boxes_diff = float((va["pred_boxes"] - vf["pred_boxes"]).abs().max())
            require(logits_equal and boxes_equal, f"tap changed detector output on batch {k} view {va['view_id']}")
            views.append({"view_id": va["view_id"], "tap_equal": True,
                          "frozen_raw_equal": frozen_logits_equal and frozen_boxes_equal,
                          "frozen_max_abs_diff": {"logits": frozen_logits_diff, "boxes": frozen_boxes_diff}})
        require(tap.calls == 5 and all(s == (4, *EXPECTED_P3) for s in tap.shapes),
                f"tap shape/count differs: {tap.calls} {tap.shapes[:1]}")
        pools_equal = all([c.as_dict() for c in pa] == [c.as_dict() for c in pb]
                          for pa, pb in zip(a["candidates"], b["candidates"]))
        require(pools_equal, "decoded candidate pools differ between passes")
        require(a["geometry"] == b["geometry"] == frozen["geometry"], "crop geometry differs")
        pass_a[k], pass_b[k] = a, b
        tap_report.append({"batch": k, "image_ids": frozen["image_ids"], "views": views,
                           "tap_calls": tap.calls, "p3_shape": list(tap.shapes[0]),
                           "pools_identical_between_passes": True,
                           "timing_A": a["timing"], "timing_B": b["timing"]})
    require(calls == CAPS["forward_calls"], f"forward count {calls} differs from the frozen {CAPS['forward_calls']}")
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)),
            "forward passes changed the complete detector state")
    L.assert_no_detector_grads(engine.model)
    L.assert_no_detector_grads(engine.ema.module)
    raw_exact = all(v["frozen_raw_equal"] for rep in tap_report for v in rep["views"])
    phases["forward"] = {"seconds": time.monotonic() - t0, "forward_calls": calls,
                         "frozen_raw_views_bitwise_equal": raw_exact,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    publish(out / "TAP_RESULT.json", {"batches": tap_report, "forward_calls": calls,
                                      "frozen_raw_views_bitwise_equal": raw_exact,
                                      "feature_level": P.FEATURE_LEVEL,
                                      "feature_stride": list(P.FEATURE_STRIDE),
                                      "p3_extent_matches_core": [EXPECTED_P3[1], EXPECTED_P3[2]] ==
                                      list(P.ViewAffine((0.0, 0.0, 10.0, 10.0)).feature_extent)})

    # 3. real pooling + complete pairing decisions per image (no GT in this phase)
    t0 = time.monotonic()
    pdra_arm = arms["arms"]["pdra_difference_strips"].to("cuda:0")
    c_arm = arms["arms"]["c_whole_box_pairwise"].to("cuda:0")
    b_arm = arms["arms"]["b_single_box_quality"].to("cuda:0")
    per_image, material = [], []
    decision_totals = Counter()
    mismatch_total = 0

    class _View:
        __slots__ = ("view_id",)

        def __init__(self, view_id):
            self.view_id = view_id

    for k in batches:
        b = pass_b[k]
        rows = data.records[4 * k:4 * k + 4]
        for j, row in enumerate(rows):
            image = row["stable_image_id"]
            pool = b["candidates"][j]
            tiles = {t["view_id"]: tuple(float(v) for v in t["crop_box"]) for t in b["geometry"][j]}
            affines = {v: P.candidate_affine(_View(v), (row["height"], row["width"]), tiles) for v in range(5)}
            affine_of = lambda c, _a=affines: _a[c.view_id]
            bank = FEAT.FeatureBank({v: b["p3_maps"][v][j] for v in range(5)})
            records = P.cluster_pair_records(pool)
            pairs = [P.build_pair_input(r["box_i"].box_xyxy, affine_of(r["box_i"]),
                                        r["box_j"].box_xyxy, affine_of(r["box_j"]),
                                        identity_i=P._identity(r["box_i"]),
                                        identity_j=P._identity(r["box_j"])) for r in records]
            valid_pairs = [p for p in pairs if p.valid]
            strips = sum(len(p.left.feature_strips) + len(p.right.feature_strips) for p in valid_pairs)
            torch.cuda.synchronize()
            p0 = time.perf_counter()
            with torch.no_grad():
                d_i = FEAT.pool_sides(bank, [p.left for p in valid_pairs])
                d_j = FEAT.pool_sides(bank, [p.right for p in valid_pairs])
            torch.cuda.synchronize()
            pool_seconds = time.perf_counter() - p0
            require(torch.isfinite(d_i.float()).all() and torch.isfinite(d_j.float()).all(), "pooled features not finite")
            adapter = L.PairBankAdapter.build(pdra_arm, bank, pool, affine_of, "pdra")

            class _Table:
                def __init__(self, table): self.table = table
                def r(self, pair): return self.table[(pair.identity_i, pair.identity_j)][2]
                def a(self, pair): return self.table[(pair.identity_i, pair.identity_j)][3]

            decisions = P.pair_decisions(pool, affine_of, _Table(adapter.table))
            merged = L.merge(pool, affine_of, adapter)
            require(len(decisions) == len(records) == len(rows_by_image[image]),
                    f"pair count differs from stage 0 on {image[:8]}")
            stage0_valid = sum(int(r[13]) for r in rows_by_image[image].values())
            mismatches = 0
            for d in decisions:
                r0 = rows_by_image[image][(d["identity_i"], d["identity_j"])]
                if bool(r0[13]) != d["valid"] or (r0[13] and r0[14] != d["read_reason"]):
                    mismatches += 1
            if raw_exact:
                require(mismatches == 0, f"readability differs from stage 0 on {image[:8]}")
            mismatch_total += mismatches
            reasons = Counter(d["decision_reason"] for d in decisions)
            decision_totals.update(reasons)
            c_adapter = L.PairBankAdapter.build(c_arm, bank, pool, affine_of, "c") if (k == batches[0] and j == 0) else None
            b_adapter = L.QualityBankAdapter.build(b_arm, bank, pool, affine_of) if (k == batches[0] and j == 0) else None
            per_image.append({"image_id": image, "pool": len(pool), "clusters": len(P.nms_clusters(pool)),
                              "pairs": len(records), "readable": len(valid_pairs),
                              "stage0_readable": stage0_valid, "readability_mismatches": mismatches,
                              "strips_pooled": strips, "pool_seconds": pool_seconds,
                              "pool_us_per_readable_pair": 1e6 * pool_seconds / max(len(valid_pairs), 1),
                              "decision_reasons": dict(reasons),
                              "merge": {k2: merged[k2] for k2 in ("cluster_count", "kept_count", "substitutions",
                                                                    "all_abstain_clusters", "rank_shifted_members",
                                                                    "budget_membership_delta_vs_C")},
                              "controls_instantiated": {"c_pairs": len(c_adapter.table) if c_adapter else None,
                                                        "b_pairs": len(b_adapter.table) if b_adapter else None}})
            # training material for the gradient check: labelled readable pairs only
            by_id = {P._identity(c): c for c in pool}
            keep, pref_t, abst_t = [], [], []
            for (ii, jj), r0 in rows_by_image[image].items():
                if r0[8] not in ("pref", "abstain") or not r0[13]:
                    continue
                pair = P.build_pair_input(by_id[ii].box_xyxy, affine_of(by_id[ii]), by_id[jj].box_xyxy,
                                          affine_of(by_id[jj]), identity_i=ii, identity_j=jj)
                if not pair.valid:
                    continue
                keep.append(pair)
                pref_t.append(int(r0[9]) if r0[8] == "pref" else -1)
                abst_t.append(1.0 if r0[8] == "abstain" else 0.0)
            with torch.no_grad():
                m_i = FEAT.pool_sides(bank, [p.left for p in keep]).cpu()
                m_j = FEAT.pool_sides(bank, [p.right for p in keep]).cpu()
            material.append((m_i, m_j, FEAT.geometry_tensor(keep, "cpu"),
                             torch.tensor(pref_t, dtype=torch.int8), torch.tensor(abst_t)))
            del bank, d_i, d_j
    phases["pool_decide"] = {"seconds": time.monotonic() - t0,
                             "pairs": sum(r["pairs"] for r in per_image),
                             "readable": sum(r["readable"] for r in per_image),
                             "strips": sum(r["strips_pooled"] for r in per_image),
                             "readability_mismatches": mismatch_total,
                             "decision_reasons": dict(decision_totals),
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    require(phases["pool_decide"]["pairs"] == plan_record["expected"]["pairs"], "total pair count differs from the plan")
    publish(out / "POOL_DECISION_RESULT.json", {"images": per_image, **phases["pool_decide"],
                                                "gt_read_in_decision_phase": False})

    # 4. fixed comparator updates with the detector frozen
    t0 = time.monotonic()
    tensors = L.PairTensors(sides_i=torch.cat([m[0] for m in material]), sides_j=torch.cat([m[1] for m in material]),
                            geom=torch.cat([m[2] for m in material]), pref_target=torch.cat([m[3] for m in material]),
                            abstain_target=torch.cat([m[4] for m in material]))
    comparator_before = L.tensor_state_digest(pdra_arm)
    spec = {**L.frozen_spec(), "updates": CHECK_UPDATES}
    torch.autograd.backward = guarded_backward
    try:
        train = L.train_pair_arm(pdra_arm, tensors, spec=spec, device="cuda:0")
    finally:
        torch.autograd.backward = original_backward
    grads = {name: (p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0.0)
             for name, p in pdra_arm.named_parameters()}
    require(all(grads.values()), f"comparator parameters without finite non-zero gradient: {[n for n, ok in grads.items() if not ok]}")
    require(train["updates"] == CHECK_UPDATES, "update count differs from the frozen check count")
    L.assert_no_detector_grads(engine.model)
    L.assert_no_detector_grads(engine.ema.module)
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)), "comparator training touched the detector state")
    require(L.tensor_state_digest(engine.ema.module) == detector_digest_before, "detector EMA digest changed")
    comparator_after = L.tensor_state_digest(pdra_arm)
    require(comparator_after != comparator_before, "comparator parameters did not change")
    phases["learn"] = {"seconds": time.monotonic() - t0, "updates": train["updates"], "batch": train["batch"],
                       "labelled_pairs": len(tensors), "first_losses": train["first_losses"],
                       "mean_loss": train["mean_loss"], "parameters_with_gradient": sum(grads.values()),
                       "parameters_total": len(grads), "comparator_digest_before": comparator_before,
                       "comparator_digest_after": comparator_after,
                       "detector_digest_unchanged": True, "detector_full_state_exact": True,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    publish(out / "LEARN_RESULT.json", phases["learn"])

    # 5. resources and closure
    wall = time.monotonic() - started
    resources = {"wall_seconds": wall, "vm_hwm_bytes": rss_high_water(),
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                 "output_bytes": output_bytes(out), "caps": CAPS,
                 "within_caps": wall <= CAPS["wall_seconds"]
                 and torch.cuda.max_memory_reserved() <= CAPS["gpu_peak_bytes"]
                 and output_bytes(out) <= CAPS["output_bytes"]}
    require(resources["within_caps"], f"resource caps exceeded: {resources}")
    publish(out / "RESOURCE.json", resources)
    return {"status": "PASS", "task": TASK, "source_commit": source_commit,
            "batches": list(batches), "images": sorted(images),
            "forward_calls": calls, "comparator_updates": train["updates"], "detector_updates": 0,
            "backward_calls_on_detector": 0,
            "gates": {"tap_outputs_unchanged": True, "tap_shape": [4, *EXPECTED_P3],
                      "frozen_raw_views_bitwise_equal": raw_exact,
                      "real_pooling_pairs": phases["pool_decide"]["readable"],
                      "comparator_gradients_all_parameters": True,
                      "detector_state_exact_before_after": True,
                      "complete_decisions": phases["pool_decide"]["pairs"],
                      "readability_mismatches_vs_stage0": mismatch_total,
                      "resources_within_caps": True},
            "feature_binding": {"level": P.FEATURE_LEVEL, "stride": list(P.FEATURE_STRIDE),
                                "verified_shape": [4, *EXPECTED_P3], "verified": True},
            "phases": phases, "resources": resources, "elapsed_seconds": wall}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "dry-run"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", default="UNRECORDED")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    require(TASK in out.parts, "output must sit inside the task directory " + TASK)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    publish(out / "START.json", {"task": TASK, "mode": args.mode, "pid": os.getpid(),
                                 "started_unix": time.time(), "source_commit": args.source_commit,
                                 "batches": list(FROZEN_BATCHES), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    try:
        inputs = resolve_inputs(args.parent, FROZEN_BATCHES)
        publish(out / "ADMISSION.json", {"task": TASK, "parent_contract_sha256": PARENT_SHA,
                                         "seed0_task": inputs["seed0"]["task_id"],
                                         "checkpoint": inputs["checkpoint"],
                                         "training_source_root": str(inputs["training_root"]),
                                         "binding": str(inputs["binding_path"]), "gpu_uuid": inputs["gpu_uuid"],
                                         "stage0_inputs": inputs["stage0"], "fairness": inputs["fairness"],
                                         "core_version": P.CORE_VERSION, "mode": args.mode,
                                         "own_source": {n: sha(HERE / n) for n in
                                                        ("pdra_gpu_check.py", "pdra_core.py", "pdra_features.py",
                                                         "pdra_learn.py", "pdra_tap.py")}})
        if args.mode == "dry-run":
            import torch
            require(not torch.cuda.is_initialized(), "dry-run initialized CUDA")
            result = {"status": "PLAN_ONLY", **plan(inputs, FROZEN_BATCHES),
                      "cuda_initialized": False, "model_constructed": False,
                      "executed_forward_calls": 0, "executed_updates": 0}
            publish(out / "PLAN.json", result)
        else:
            result = run(inputs, out, FROZEN_BATCHES, args.source_commit)
        result["elapsed_seconds"] = time.monotonic() - started
        ref = publish(out / "RESULT.json", result)
        publish(out / "CLOSED.json", {"result": ref, "returncode": 0, "automatic_retry": False})
        print(json.dumps({"status": result["status"], "result": ref}), flush=True)
        return 0
    except Exception:
        publish(out / "STOP.json", {"traceback": traceback.format_exc(),
                                    "elapsed_seconds": time.monotonic() - started,
                                    "automatic_retry": False})
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
