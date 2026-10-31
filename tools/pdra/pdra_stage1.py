"""PDRA stage 1 (feature pass + fixed training + eval diagnostics) and stage 2 (full
inference with representative replacement) -- rev 8 runners.

Modes
-----
``plan``        CPU only.  Resolves and SHA-checks every input, re-verifies the frozen
                sampling manifest against the stage-0 rows, instantiates the four arm
                specifications with the real identities, emits the frozen learning spec
                (capacity-matched widths and per-sample compute), the frozen final
                criteria (``pdra_score.CRITERIA``) and the stage-1 / stage-2 / scoring
                budgets (feature-extraction forwards, descriptor cache, full eval
                inference, scoring calls, update counts, measured frozen forward time).
``run-stage1``  GPU.  Requires a PASSing GPU engineering check result.  One five-view
                pass over the 548 development images (137 x 5 = 685 batch forwards, zero
                extra), fp16 descriptor caches for the frozen train sample (222,755
                pairs) and the eval readable pref/abstain pairs, fixed-count training of
                the three freshly initialised arms (train split only -- eval rows are
                never used for updates, tuning, checkpoint selection or early stopping),
                and the eval diagnostics by the frozen buckets.  Diagnostics are reported,
                never gated: the detection evaluation always proceeds (``pdra_score.py``).
``run-stage2``  GPU (rev 8: the entry verifies the four stage-1 artifact pins, restores
                the audited engine and only then moves the frozen arm weights to CUDA).
                Same five-view pass; for every image the complete GT-free inference
                enumeration decides all cluster-internal pairs for PDRA, (c) and (b); the
                representative replacement with re-ranking produces top600/top300
                predictions per arm in the frozen prediction schema, next to original C.
                Scoring is the separate CPU step ``pdra_score.py score``.

rev 7 (2026-10-31): the three arms train with capacity-matched widths
(``pdra_learn.LEARNING_SPEC['hidden_by_arm']``); runtime constraints are enforced while
each stage runs (wall clock and output bytes; CUDA allocated/reserved peaks are recorded,
never gated -- no artificial memory threshold and no allocator quota; a real OOM is an
exception -> STOP.json, no batch or precision fallback, no retry).  Nothing here reads
ground truth at inference; stage 1 opens annotations only for the (b) member targets after
each image's feature pass, and reads training/eval labels solely from the frozen stage-0
rows.

rev 8 (2026-10-31): the stage-2 entry restores the audited seed0 engine *before* any CUDA
touch -- ``deterministic_device_setup`` refuses to run once CUDA is initialized, so the
frozen arm weights are moved to CUDA only after the restore completed (the rev-7 order
stopped exactly there), and the detector baseline capture keeps its stage-1 invariant
(fresh arm construction first, capture second, inference third).  The entry also SHA-pins
the closed stage-1 attempt on the command line: the RESULT file, its CLOSED receipt, the
LEARNING_FROZEN spec file and the recorded stage-1 source commit are re-verified before
the restore, and the arm files stay bound through the pinned RESULT's SHA record.  A pin,
criteria or learning-spec mismatch is a STOP, never a retrain.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _path in (str(HERE), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pdra_core as P
import pdra_gpu_check as G
import pdra_score as SCORE

STAGE1_TASK_PREFIX = "r18-pdra-stage1-"
STAGE2_TASK_PREFIX = "r18-pdra-stage2-"
# rev 4: the computed fp16 need is 9.25 GB (slot-padded PDRA sides 6.94 GB + whole-box
# pair cache 1.39 GB + <=0.91 GB member boxes); the rev-3 "<= 8 GB" figure was an
# estimate and is replaced by this computed requirement with a 10 GiB server-only cap.
CACHE_CAP_BYTES = 10 * 1024 ** 3
# rev 7 runtime caps (2026-10-31 preference): wall clock and output bytes are enforced
# while the stage runs; CUDA peaks are recorded, never gated; no allocator quota.
STAGE1_CAPS = {"wall_seconds": 3600, "output_bytes": 11 * 1024 ** 3}
STAGE2_CAPS = {"wall_seconds": 5400, "output_bytes": 2 * 1024 ** 3}
GT_USE_STAGE1 = {
    "stage1_train_labels": ("frozen stage-0 rows (seed0-pairs.jsonl), train split only; the sampling "
                            "manifest was re-verified against these rows in plan"),
    "stage1_eval_labels": ("frozen stage-0 rows, eval split; used for the eval diagnostics only, "
                           "after training, never for updates, tuning, checkpoint selection or "
                           "early stopping"),
    "annotations_opened": ("only for the (b)-arm member matched-GT IoU targets, through the identical "
                           "_gt_items chain, after each image's five-view pass"),
    "pooling_and_pairs_gt_free": True,
}
GT_USE_STAGE2 = {
    "annotations_read_anywhere": False,
    "labels_read_anywhere": False,
    "decision_enumeration": ("the complete cluster-internal pair set of every image "
                             "(cluster_pair_records, identical to the training pairing domain); "
                             "no GT filter, no label-based narrowing"),
    "note": "the stage-2 path has no ground-truth argument anywhere; scoring is a separate CPU step",
}
require, publish, read, exact, sha = G.require, G.publish, G.read, G.exact, G.sha


# --- shared input selection -----------------------------------------------------
def manifest_pairs(ids_path) -> dict:
    """``{image: {(id_i, id_j): tag}}`` from the frozen train sampling file."""
    out = defaultdict(dict)
    for line in Path(ids_path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        tag, identifier = line.split("\t", 1)
        image, a, b = identifier.split("|")
        va, qa, ca = (int(v) for v in a.split(","))
        vb, qb, cb = (int(v) for v in b.split(","))
        out[image][((va, qa, ca), (vb, qb, cb))] = tag
    return out


def select_rows(train_pairs: dict) -> dict:
    """Stream the stage-0 rows once: train sampled pairs and eval readable pref/abstain."""
    rows = {"train": defaultdict(dict), "eval": defaultdict(dict)}
    with (G.STAGE0_DIR / "seed0-pairs.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = ((row[4][0], row[4][1], row[3]), (row[5][0], row[5][1], row[3]))
            if row[2] == "train":
                if key in train_pairs.get(row[0], {}):
                    rows["train"][row[0]][key] = row
            elif row[8] in ("pref", "abstain") and row[13]:
                rows["eval"][row[0]][key] = row
    return rows


def image_split() -> tuple[set, set]:
    """The frozen module train/eval image split from the stage-0 image counts (338/210)."""
    counts = read(G.STAGE0_DIR / "seed0-IMAGE_COUNTS.json")
    require(isinstance(counts, list) and len(counts) == 548, "image-counts file differs")
    train = {row["image_id"] for row in counts if row["split"] == "train"}
    evaluation = {row["image_id"] for row in counts if row["split"] == "eval"}
    require(len(train) == 338 and len(evaluation) == 210,
            "split sizes differ from the frozen 338 train / 210 eval")
    return train, evaluation


# --- budgets --------------------------------------------------------------------
def budgets(inputs: dict, freeze: dict) -> dict:
    """Every budget the full run consumes, split into its four cost blocks (rev 7)."""
    summary = read(G.STAGE0_DIR / "STAGE0_SUMMARY.json")
    ledgers = summary["ledgers"]
    full = read(Path(inputs["seed0"]["job_root"]) / "full-C" / "RESULT.json")
    forward_seconds = float(full["timing"]["e2e_total_seconds"])
    train_pairs = inputs["sampling"]["n_pair_total"]
    eval_pref = ledgers["eval"]["feasible_counts"]["pref"]
    eval_abstain = ledgers["eval"]["feasible_counts"]["abstain"]
    stage1_pairs = train_pairs + eval_pref + eval_abstain
    readable_all = sum(ledgers["all"]["feasible_counts"].values())
    all_pairs = summary["scale"]["all"]["pairs"]
    pool_all = summary["scale"]["all"]["pool"]
    spec = L_LEARNING_SPEC()
    capacity = {key: {"parameters": row["parameters"], "hidden": row["hidden"],
                      "weight_macs_per_sample": row["weight_macs_per_sample"],
                      "sample_unit": row["sample_unit"]}
                for key, row in freeze["arms"].items()}
    feature_extraction = {
        "stage1_batch_forwards": 137 * 5, "stage2_batch_forwards": 137 * 5,
        "batch_size": 4, "views": 5, "images_per_stage": 548,
        "extra_forwards_beyond_the_single_five_view_pass": 0,
        "note": ("each stage makes exactly one five-view pass over the same 548 images; the P3 tap is a "
                 "read-only hook, stage-1 diagnostics and stage-2 adapters add no forward, and the "
                 "scoring step makes no model call at all"),
    }
    descriptor_cache = {
        "pdra_sides_bytes": FEAT_cache_budget(stage1_pairs),
        "c_whole_pairs_bytes": 2 * FEAT_whole_dim() * 2 * stage1_pairs,
        "b_boxes_upper_bytes": 2 * FEAT_whole_dim() * 2 * train_pairs,
        "total_required_upper_bytes": (FEAT_cache_budget(stage1_pairs)
                                       + 2 * FEAT_whole_dim() * 2 * stage1_pairs
                                       + 2 * FEAT_whole_dim() * 2 * train_pairs),
        "cap_bytes": CACHE_CAP_BYTES, "server_only": True,
        "note": "(b) is an upper bound (unique-member de-duplication shrinks it at run time)",
    }
    training = {
        "updates": {"pdra_difference_strips": spec["updates"],
                    "c_whole_box_pairwise": spec["updates"],
                    "b_single_box_quality": spec["updates"],
                    "batch_pairs": spec["batch_pairs"]},
        "capacity": capacity,
        "training_macs_total": freeze["capacity_control"]["training_macs_total"],
        "optimizer": spec["optimizer"], "grad_clip": spec["grad_clip"],
        "fresh_initialization": ("arms constructed by pdra_learn.build_arms() under the frozen seed; "
                                 "no comparator or optimizer state is loaded from disk or from the GPU "
                                 "engineering check"),
        "train_split_only": ("updates consume only the train-split sample; eval rows are never used for "
                             "updates, tuning, checkpoint selection or early stopping"),
    }
    eval_diagnostics = {
        "split": "eval", "pairs": eval_pref + eval_abstain,
        "buckets": "delta-IoU x area buckets, clear pairs (>= 0.25), cross-GT abstention",
        "diagnostic_only": ("these diagnostics never gate the detection evaluation; the frozen final "
                            "criteria live in pdra_score.CRITERIA and are evaluated in the separate "
                            "scoring step"),
    }
    stage1 = {
        "images": 548, "feature_extraction": feature_extraction,
        "descriptor_cache": descriptor_cache, "training": training,
        "eval_diagnostics": eval_diagnostics,
        "wall_cap_seconds": STAGE1_CAPS["wall_seconds"],
        "output_cap_bytes": STAGE1_CAPS["output_bytes"],
        "returned_bytes_cap": 1024 ** 2,
        "time_basis": {"frozen_full_C_e2e_seconds_548_images": forward_seconds,
                       "pooling_estimate_seconds": "seconds (measured per pair by the GPU check)",
                       "training_estimate": "3 arms x 20,000 updates of <=0.70M-parameter MLPs; minutes"},
    }
    inference_macs = {
        "pdra_difference_strips": readable_all * capacity["pdra_difference_strips"]["weight_macs_per_sample"],
        "c_whole_box_pairwise": readable_all * capacity["c_whole_box_pairwise"]["weight_macs_per_sample"],
        "b_single_box_quality": pool_all * capacity["b_single_box_quality"]["weight_macs_per_sample"],
        "definition": ("weight MACs of the readable-pair adapters (pair arms: both sides encoded per "
                       "readable pair) and of the per-member quality head (per pool member)"),
    }
    stage2 = {
        "images": 548, "feature_extraction": feature_extraction,
        "pairs_decided": all_pairs, "pairs_pooled_readable": readable_all,
        "pooled_on_the_fly_no_cache": True,
        "comparator_batches_per_image_max": int(-(-51210 // 4096)),
        "decision_layer": "pure-Python Copeland over all cluster pairs (measured per pair by the check)",
        "arms": ["a_original_c", "b_single_box_quality", "c_whole_box_pairwise", "pdra_difference_strips"],
        "inference_macs": inference_macs,
        "outputs": "top600/top300 predictions per arm (frozen schema) + replacement stats",
        "wall_cap_seconds": STAGE2_CAPS["wall_seconds"],
        "output_cap_bytes": STAGE2_CAPS["output_bytes"],
        "time_basis": {"frozen_full_C_e2e_seconds_548_images": forward_seconds,
                       "decision_layer_estimate": "measured by the GPU check per pair"},
    }
    scoring = {
        "mode": "pdra_score.py score (CPU, frozen score_budget)",
        "calls": "4 arms x 2 budgets x 2 splits (eval 210 primary, train 338 diagnostic)",
        "model_calls": 0,
        "wall_cap_seconds": SCORE.WALL_CAP_SECONDS,
        "output_cap_bytes": SCORE.OUTPUT_CAP_BYTES,
        "criteria": "pdra_score.CRITERIA (frozen 2026-10-31, before any learning or score result)",
    }
    return {"stage1": stage1, "stage2": stage2, "scoring": scoring,
            "chain_document": "PDRA_FULL_RUN_R1.md (single command; stop on first failure, no retry)",
            "estimates_marked": "all seconds are estimates until the stage runs measure them"}


def L_LEARNING_SPEC() -> dict:
    import pdra_learn as L
    return L.frozen_spec()


def FEAT_cache_budget(n_pairs: int) -> int:
    import pdra_features as FEAT
    return FEAT.cache_budget_bytes(n_pairs)


def FEAT_whole_dim() -> int:
    import pdra_features as FEAT
    return FEAT.whole_dim()


def plan(inputs: dict, out: Path) -> dict:
    import pdra_learn as L
    verification = L.verify_sampling(G.STAGE0_DIR / "train-sampling-ids.txt",
                                     G.STAGE0_DIR / "seed0-pairs.jsonl", inputs["sampling"])
    require(verification["status"] == "PASS", f"frozen sampling manifest failed verification: {verification}")
    freeze = L.freeze_payload()
    publish(out / "LEARNING_FROZEN.json", freeze)
    publish(out / "SAMPLING_VERIFICATION.json", verification)
    publish(out / "CRITERIA_FROZEN.json", {"criteria": SCORE.CRITERIA,
                                           "criteria_sha256": SCORE.criteria_sha256()})
    result = {"status": "PLAN_ONLY", "core_version": P.CORE_VERSION,
              "feature_level": P.FEATURE_LEVEL, "feature_binding_verified": P.FEATURE_BINDING_VERIFIED,
              "fairness": inputs["fairness"], "learning_spec_sha256": freeze["learning_spec_sha256"],
              "arms": freeze["arms"], "capacity_control": freeze["capacity_control"],
              "criteria_sha256": SCORE.criteria_sha256(),
              "sampling_verification": verification["status"],
              "budgets": budgets(inputs, freeze),
              "runtime_caps": {"stage1": STAGE1_CAPS, "stage2": STAGE2_CAPS,
                               "score_wall_seconds": SCORE.WALL_CAP_SECONDS},
              "stage0_inputs": inputs["stage0"],
              "cuda_initialized": False, "model_constructed": False,
              "executed_forward_calls": 0, "executed_updates": 0}
    return result


# --- stage 1 ---------------------------------------------------------------------
def _gt_items(data, row, ev):
    """GT items per category for training targets (same helper chain as stage 0)."""
    dataset, _, _ = ev.bridge_inputs([row], [data.raw(row)], [])
    by_cat = defaultdict(list)
    for g in dataset["annotations"]:
        if g["iscrowd"] == 0:
            x, y, w, h = g["bbox"]
            by_cat[g["category_id"]].append(P.GTItem(g["raw_annotation_id"], g["category_id"],
                                                     (float(x), float(y), float(x + w), float(y + h)),
                                                     float(g["area"])))
    return by_cat


def _affines(row, geometry):
    class _View:
        __slots__ = ("view_id",)

        def __init__(self, view_id):
            self.view_id = view_id
    tiles = {t["view_id"]: tuple(float(v) for v in t["crop_box"]) for t in geometry}
    affines = {v: P.candidate_affine(_View(v), (row["height"], row["width"]), tiles) for v in range(5)}
    return lambda c, _a=affines: _a[c.view_id]


def run_stage1(inputs: dict, out: Path, gpu_check_result: dict, source_commit: str, *,
               budget_started: float | None = None, accounting: G.Accounting | None = None) -> dict:
    import torch
    import pdra_features as FEAT
    import pdra_learn as L
    import pdra_tap as TAP
    require(gpu_check_result.get("status") == "PASS" and gpu_check_result["feature_binding"]["verified"],
            "a PASSing GPU engineering check with a verified P3 binding is required")
    started = time.monotonic() if budget_started is None else budget_started
    accounting = G.Accounting("r18-pdra-stage1", "run-stage1") if accounting is None else accounting

    def guard(label: str) -> dict:
        return G.enforce_runtime(started=started, label=label, caps=STAGE1_CAPS, out=out,
                                 torch_mod=torch, accounting=accounting)

    accounting.phase("plan")
    plan_record = plan(inputs, out)
    publish(out / "PLAN.json", plan_record)
    guard("plan")
    accounting.complete("plan")
    train_pairs = manifest_pairs(G.STAGE0_DIR / "train-sampling-ids.txt")
    rows = select_rows(train_pairs)
    train_images, eval_images = image_split()

    accounting.phase("restore")
    restored = TAP.restore_frozen_seed0_engine(
        training_source_root=inputs["training_root"], train_contract=inputs["training"],
        binding_path=inputs["binding_path"], checkpoint_path=inputs["checkpoint"]["path"],
        checkpoint_sha256=inputs["checkpoint"]["sha256"], gpu_uuid=inputs["gpu_uuid"])
    engine, binding, model = restored["engine"], restored["binding"], restored["model"]
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.r0.evaluation_data import Development548
    from sparse_rtdetr.r0 import evaluation as ev
    data = Development548(inputs["data_spec"])
    arms = L.build_arms()                       # fresh construction; consumes the CPU RNG before capture
    arm_init = {key: {"parameters": arms["report"][key]["parameters"],
                      "hidden": arms["report"][key]["hidden"],
                      "weight_macs_per_sample": arms["report"][key]["weight_macs_per_sample"],
                      "init_digest": L.tensor_state_digest(arm)}
                for key, arm in arms["arms"].items()}
    before = checkpoint.capture(engine, binding)
    restore_peak_allocated = int(torch.cuda.max_memory_allocated())
    restore_peak_reserved = int(torch.cuda.max_memory_reserved())
    guard("restore")
    torch.cuda.reset_peak_memory_stats()
    accounting.complete("restore")

    accounting.phase("features")
    caches = {"train": defaultdict(list), "eval": defaultdict(list)}
    box_desc, box_target, box_seen = [], [], set()
    calls = 0
    for k in range(137):
        batch_rows = data.records[4 * k:4 * k + 4]
        with TAP.EncoderTap(model) as tap:
            b = TAP.five_view_forward(data, batch_rows, model, tap=tap)
        calls += b["forward_calls"]
        accounting.forwards(b["forward_calls"])
        for j, row in enumerate(batch_rows):
            image = row["stable_image_id"]
            pool = b["candidates"][j]
            by_id = {P._identity(c): c for c in pool}
            affine_of = _affines(row, b["geometry"][j])
            bank = FEAT.FeatureBank({v: b["p3_maps"][v][j] for v in range(5)})
            for split in ("train", "eval"):
                wanted = rows[split].get(image, {})
                if not wanted:
                    continue
                pairs, meta = [], []
                for (ii, jj), r0 in wanted.items():
                    pair = P.build_pair_input(by_id[ii].box_xyxy, affine_of(by_id[ii]),
                                              by_id[jj].box_xyxy, affine_of(by_id[jj]),
                                              identity_i=ii, identity_j=jj)
                    require(pair.valid, "frozen sampled/eval pair is no longer readable")
                    pairs.append(pair)
                    meta.append(r0)
                with torch.no_grad():
                    s_i = FEAT.pool_sides(bank, [p.left for p in pairs]).cpu()
                    s_j = FEAT.pool_sides(bank, [p.right for p in pairs]).cpu()
                    w_i = FEAT.pool_whole(bank, [FEAT.whole_box_read(p.box_i, affine_of(by_id[p.identity_i])) for p in pairs]).cpu()
                    w_j = FEAT.pool_whole(bank, [FEAT.whole_box_read(p.box_j, affine_of(by_id[p.identity_j])) for p in pairs]).cpu()
                geom = FEAT.geometry_tensor(pairs, "cpu")
                pref_t = torch.tensor([int(r0[9]) if r0[8] == "pref" else -1 for r0 in meta], dtype=torch.int8)
                abst_t = torch.tensor([1.0 if r0[8] == "abstain" else 0.0 for r0 in meta])
                c = caches[split]
                c["s_i"].append(s_i); c["s_j"].append(s_j); c["w_i"].append(w_i); c["w_j"].append(w_j)
                c["geom"].append(geom); c["pref"].append(pref_t); c["abstain"].append(abst_t)
                c["meta"].extend([[image, r0[8], r0[9], r0[10], r0[12], r0[6]] for r0 in meta])
                if split == "train":
                    gts = _gt_items(data, row, ev)
                    members = {ident for key in wanted for ident in key}
                    fresh = [ident for ident in sorted(members) if (image, ident) not in box_seen]
                    reads = [FEAT.whole_box_read(by_id[ident].box_xyxy, affine_of(by_id[ident])) for ident in fresh]
                    keep = [(ident, r) for ident, r in zip(fresh, reads) if r.valid]
                    if keep:
                        with torch.no_grad():
                            desc = FEAT.pool_whole(bank, [r for _, r in keep]).cpu()
                        box_desc.append(desc)
                        for ident, _ in keep:
                            cand = by_id[ident]
                            hits = P.match_gts_iou(cand.box_xyxy, cand.category_id, gts.get(cand.category_id, []))
                            box_target.append(hits[0][0] if hits else 0.0)
                            box_seen.add((image, ident))
            del bank
        guard(f"features/batch-{k}")
    require(calls == 137 * 5, "stage-1 forward count differs")
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)), "feature pass changed the detector")
    accounting.complete("features")

    accounting.phase("cache")
    tensors = {}
    for split in ("train", "eval"):
        c = caches[split]
        tensors[split] = {k2: torch.cat(v) for k2, v in c.items() if k2 != "meta"}
        tensors[split]["meta"] = c["meta"]
    train_meta_images = {row[0] for row in tensors["train"]["meta"]}
    eval_meta_images = {row[0] for row in tensors["eval"]["meta"]}
    require(train_meta_images <= train_images, "training sample holds a non-train image")
    require(eval_meta_images <= eval_images, "eval diagnostics sample holds a non-eval image")
    cache_dir = out / "cache"
    cache_refs = {}
    for split in ("train", "eval"):
        payload = {k2: v for k2, v in tensors[split].items()}
        cache_refs[split] = FEAT.save_cache(cache_dir / f"{split}-descriptors.pt", payload)
    cache_refs["b_boxes"] = FEAT.save_cache(cache_dir / "train-boxes.pt",
                                            {"desc": torch.cat(box_desc), "target": torch.tensor(box_target)})
    total_cache = sum(ref["bytes"] for ref in cache_refs.values())
    require(total_cache <= CACHE_CAP_BYTES, "descriptor cache exceeds the 10 GiB cap")
    publish(out / "CACHE.json", {"refs": cache_refs, "bytes": total_cache,
                                 "train_pairs": int(tensors["train"]["s_i"].shape[0]),
                                 "eval_pairs": int(tensors["eval"]["s_i"].shape[0]),
                                 "b_boxes": len(box_target)})
    require(int(tensors["train"]["s_i"].shape[0]) == inputs["sampling"]["n_pair_total"], "train cache size differs from the manifest")
    guard("cache")
    accounting.complete("cache")

    # fixed training -- train split only, freshly initialised arms, frozen update count
    accounting.phase("train")
    train_reports = {}
    tr = tensors["train"]
    pdra_tensors = L.PairTensors(tr["s_i"], tr["s_j"], tr["geom"], tr["pref"], tr["abstain"])
    c_tensors = L.PairTensors(tr["w_i"], tr["w_j"], tr["geom"], tr["pref"], tr["abstain"])

    def progress_for(key):
        def callback(record, _key=key):
            accounting.update(_key)
            update = int(record["update"])
            if update == 1 or update % 2000 == 0:
                guard(f"train/{_key}/update-{update}")
        return callback

    train_reports["pdra_difference_strips"] = L.train_pair_arm(
        arms["arms"]["pdra_difference_strips"], pdra_tensors,
        progress=progress_for("pdra_difference_strips"))
    train_reports["c_whole_box_pairwise"] = L.train_pair_arm(
        arms["arms"]["c_whole_box_pairwise"], c_tensors,
        progress=progress_for("c_whole_box_pairwise"))
    train_reports["b_single_box_quality"] = L.train_quality_arm(
        arms["arms"]["b_single_box_quality"], torch.cat(box_desc), torch.tensor(box_target),
        progress=progress_for("b_single_box_quality"))
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)), "training touched the detector")
    L.assert_no_detector_grads(engine.model)
    L.assert_no_detector_grads(engine.ema.module)
    arm_refs = {}
    for key, arm in arms["arms"].items():
        path = out / "arms" / f"{key}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(arm.state_dict(), path)
        arm_refs[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
    guard("train")
    accounting.complete("train")

    # eval diagnostics (labels come from the stage-0 rows, never from the decision path;
    # diagnostics never gate the detection evaluation)
    accounting.phase("diagnostics")
    diagnostics = evaluate_arms(arms["arms"], tensors["eval"], L, FEAT)
    publish(out / "DIAGNOSTICS.json", diagnostics)
    guard("diagnostics")
    accounting.complete("diagnostics")
    wall = time.monotonic() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    runtime = {"wall_seconds": wall,
               "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
               "restore_peak_allocated_bytes": restore_peak_allocated,
               "restore_peak_reserved_bytes": restore_peak_reserved,
               "vm_hwm_bytes": G.rss_high_water(),
               "memory_peaks_recorded_not_gated": True,
               "memory_gate": ("none: no artificial CUDA threshold and no allocator quota; a real OOM "
                               "is an exception -> STOP.json"),
               "output_bytes": G.output_bytes(out), "caps": STAGE1_CAPS,
               "constraints_enforced_during_run": ["wall_seconds", "output_bytes"],
               "budget_observations": accounting.snapshot()["budget_observations"],
               "within_caps": (wall <= STAGE1_CAPS["wall_seconds"]
                               and G.output_bytes(out) <= STAGE1_CAPS["output_bytes"])}
    require(runtime["within_caps"], f"stage-1 runtime caps exceeded: {runtime}")
    return {"status": "PASS", "task": out.parts[-2] if len(out.parts) > 1 else str(out),
            "source_commit": source_commit, "forward_calls": calls, "detector_updates": 0,
            "arm_capacity": arms["report"], "capacity_control": plan_record["capacity_control"],
            "criteria_sha256": plan_record["criteria_sha256"],
            "arms_fresh": {"freshly_initialized": True, "state_loaded_from_disk": False,
                           "optimizer_state_loaded": False, "arms": arm_init},
            "split_use": {"train_images_in_updates": len(train_meta_images),
                          "eval_images_in_diagnostics": len(eval_meta_images),
                          "eval_images_in_updates": 0,
                          "eval_used_for": "diagnostics only, after training",
                          "used_for_tuning": False, "used_for_checkpoint_selection": False,
                          "early_stopping": False, "fixed_updates": L.LEARNING_SPEC["updates"]},
            "training": train_reports, "arms": arm_refs, "cache": cache_refs,
            "diagnostics": diagnostics, "diagnostics_after_training": True,
            "engineering_checks": {"train_split_only": True, "forward_count_exact": True,
                                   "detector_unchanged": True,
                                   "cache_within_cap": total_cache <= CACHE_CAP_BYTES},
            "gt_use": GT_USE_STAGE1,
            "runtime": runtime, "accounting": accounting.snapshot(),
            "elapsed_seconds": wall,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved}


def evaluate_arms(arms: dict, ev_t: dict, L, FEAT) -> dict:
    """Per-arm eval diagnostics on the eval readable pref/abstain pairs, frozen buckets.

    Diagnostic only (rev 7): the values are reported with the run; they gate nothing.  The
    frozen final criteria are evaluated in the separate CPU scoring step on the actual
    detection outputs.
    """
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n = int(ev_t["s_i"].shape[0])
    meta = ev_t["meta"]
    winners = {}
    with torch.no_grad():
        for key in ("pdra_difference_strips", "c_whole_box_pairwise"):
            arm = arms[key].to(device).eval()
            side_i = ev_t["s_i"] if key.startswith("pdra") else ev_t["w_i"]
            side_j = ev_t["s_j"] if key.startswith("pdra") else ev_t["w_j"]
            out = []
            for start in range(0, n, 4096):
                r, a = arm.forward_batch(side_i[start:start + 4096].to(device),
                                         side_j[start:start + 4096].to(device),
                                         ev_t["geom"][start:start + 4096].to(device))
                for rv, av in zip(r.tolist(), a.tolist()):
                    if av >= P.DELTA_ABSTAIN:
                        out.append(0)
                    elif rv >= 0.5 + P.DELTA_PREF:
                        out.append(1)
                    elif rv <= 0.5 - P.DELTA_PREF:
                        out.append(-1)
                    else:
                        out.append(0)
            winners[key] = out
        quality = arms["b_single_box_quality"].to(device).eval()
        q_i = torch.cat([quality.value(ev_t["w_i"][s:s + 4096].to(device)).cpu() for s in range(0, n, 4096)])
        q_j = torch.cat([quality.value(ev_t["w_j"][s:s + 4096].to(device)).cpu() for s in range(0, n, 4096)])
        scale = P.DELTA_PREF / (2.0 * L.LEARNING_SPEC["tau_q"])
        r_b = (0.5 + (q_i - q_j) * scale).clamp(0.0, 1.0).tolist()
        winners["b_single_box_quality"] = [1 if r >= 0.5 + P.DELTA_PREF else (-1 if r <= 0.5 - P.DELTA_PREF else 0) for r in r_b]
    winners["a_original_c"] = [1] * n       # frozen order: the earlier member has the higher score
    report = {"pairs": n, "arms": {}}
    for key, votes in winners.items():
        buckets = defaultdict(lambda: {"pairs": 0, "voted": 0, "correct": 0})
        abstain_pairs = abstain_votes = 0
        for vote, m in zip(votes, meta):
            dest, label, delta, area = m[1], m[2], m[3], m[4]
            if dest == "abstain":
                abstain_pairs += 1
                abstain_votes += int(vote != 0)
                continue
            expected = 1 if label == 1 else -1
            keys = ["all", "diou:" + P.delta_bucket_of(abs(delta)), "area:" + P.area_bucket_of(float(area))]
            if abs(delta) >= P.TAU_CLEAR:
                keys.append("clear")
            for b in keys:
                buckets[b]["pairs"] += 1
                buckets[b]["voted"] += int(vote != 0)
                buckets[b]["correct"] += int(vote == expected)
        table = {}
        for b, v in buckets.items():
            table[b] = {**v, "accuracy_on_voted": (v["correct"] / v["voted"]) if v["voted"] else None,
                        "coverage": v["voted"] / v["pairs"] if v["pairs"] else None,
                        "accuracy_counting_abstain_as_wrong": v["correct"] / v["pairs"] if v["pairs"] else None}
        report["arms"][key] = {"buckets": table, "cross_gt_pairs": abstain_pairs,
                               "cross_gt_misarbitration_rate": (abstain_votes / abstain_pairs) if abstain_pairs else None,
                               "large_object_area_gt_9216": table.get("area:area_gt_9216")}

    def acc(key):
        return report["arms"][key]["buckets"].get("clear", {}).get("accuracy_counting_abstain_as_wrong")

    def mis(key):
        return report["arms"][key]["cross_gt_misarbitration_rate"]
    clear_acc = {key: acc(key) for key in report["arms"]}
    cross_gt = {key: mis(key) for key in report["arms"]}
    diagnostic_flag = (clear_acc["pdra_difference_strips"] is not None
                       and clear_acc["b_single_box_quality"] is not None
                       and clear_acc["c_whole_box_pairwise"] is not None
                       and clear_acc["pdra_difference_strips"]
                       >= max(clear_acc["b_single_box_quality"], clear_acc["c_whole_box_pairwise"]) + 0.05)
    safety_flag = (cross_gt["pdra_difference_strips"] is not None
                   and cross_gt["c_whole_box_pairwise"] is not None
                   and cross_gt["pdra_difference_strips"] <= cross_gt["c_whole_box_pairwise"] + 0.02)
    report["diagnostics_only"] = {
        "statement": ("clear-pair accuracy, coverage and the cross-GT misarbitration rate are "
                      "diagnostics: they never gate the detection evaluation.  The frozen final "
                      "criteria (pdra_score.CRITERIA) are evaluated on the actual detection outputs "
                      "in the separate CPU scoring step."),
        "clear_pairs_acc_pdra_ge_max_b_c_plus_5pp": diagnostic_flag,
        "cross_gt_misarbitration_le_c_plus_2pp": safety_flag,
        "metric_definition": "accuracy counts abstentions as wrong (coverage reported separately)",
        "automatic_next_stage": False,
    }
    return report


def verify_stage1_binding(stage1_result_path, stage1_result_sha256, stage1_closed_sha256,
                          stage1_learning_frozen_sha256, stage1_source_commit) -> dict:
    """SHA-pin the closed stage-1 attempt consumed by a stage-2 resume (rev 8; CPU only).

    The four values are passed on the command line and re-verified here before the audited
    engine restore: the RESULT file, its CLOSED receipt, the LEARNING_FROZEN spec file and
    the stage-1 source commit.  The arm files stay bound through the pinned RESULT's SHA
    record, checked again in the arms phase.  A pin mismatch, a criteria difference or a
    learning-spec drift raises (-> STOP.json); it never triggers a retrain.
    """
    import pdra_learn as L
    result_path = Path(stage1_result_path)
    require(result_path.is_file() and not result_path.is_symlink(),
            f"stage-1 RESULT missing: {result_path}")
    require(isinstance(stage1_source_commit, str) and len(stage1_source_commit) == 40
            and all(char in "0123456789abcdef" for char in stage1_source_commit),
            "the stage-1 commit pin must be a full 40-hex commit")
    require(sha(result_path) == stage1_result_sha256, "stage-1 RESULT differs from the pinned SHA")
    result = read(result_path)
    require(result.get("status") == "PASS", "stage-1 result is not PASS")
    require(result.get("source_commit") == stage1_source_commit,
            "stage-1 source commit differs from the pin")
    require(result.get("criteria_sha256") == SCORE.criteria_sha256(),
            "stage-1 criteria differ from the frozen scoring criteria")
    closed_path = result_path.parent / "CLOSED.json"
    require(closed_path.is_file() and not closed_path.is_symlink(),
            f"stage-1 CLOSED missing: {closed_path}")
    require(sha(closed_path) == stage1_closed_sha256, "stage-1 CLOSED differs from the pinned SHA")
    closed = read(closed_path)
    closed_result = closed.get("result") if isinstance(closed.get("result"), dict) else {}
    require(closed.get("returncode") == 0 and closed.get("automatic_retry") is False,
            "stage-1 CLOSED is not a clean close")
    require(closed_result.get("sha256") == stage1_result_sha256
            and Path(str(closed_result.get("path", ""))) == result_path,
            "stage-1 CLOSED does not reference the pinned RESULT")
    frozen_path = result_path.parent / "LEARNING_FROZEN.json"
    require(frozen_path.is_file() and not frozen_path.is_symlink(),
            f"stage-1 learning spec missing: {frozen_path}")
    require(sha(frozen_path) == stage1_learning_frozen_sha256,
            "stage-1 learning spec differs from the pinned SHA")
    frozen = read(frozen_path)
    require(frozen.get("learning_spec_sha256") == L.learning_spec_sha(),
            "stage-1 learning spec is incompatible with the current learning code")
    record = {"stage1_result_path": str(result_path),
              "stage1_result_sha256": stage1_result_sha256,
              "stage1_closed_sha256": stage1_closed_sha256,
              "stage1_learning_frozen_sha256": stage1_learning_frozen_sha256,
              "stage1_source_commit": stage1_source_commit,
              "criteria_sha256": result["criteria_sha256"],
              "learning_spec_sha256": frozen["learning_spec_sha256"],
              "verified": True}
    return {"result": result, "closed": closed, "learning_frozen": frozen, "record": record}


# --- stage 2 ----------------------------------------------------------------------
def prediction(image_id, candidate) -> dict:
    return {"image_id": image_id, "query_id": candidate.query_id, "category_id": candidate.category_id,
            "score": candidate.score, "bbox_xyxy": list(candidate.box_xyxy),
            "view_id": candidate.view_id, "view_candidate_rank": candidate.view_candidate_rank}


def run_stage2(inputs: dict, out: Path, gpu_check_result: dict, arm_dir: Path, source_commit: str, *,
               stage1_result_path: Path | None = None, stage1_result_sha256: str | None = None,
               stage1_closed_sha256: str | None = None,
               stage1_learning_frozen_sha256: str | None = None,
               stage1_source_commit: str | None = None,
               budget_started: float | None = None,
               accounting: G.Accounting | None = None) -> dict:
    import torch
    import pdra_features as FEAT
    import pdra_learn as L
    import pdra_tap as TAP
    from slice_core import merge_candidates
    require(gpu_check_result.get("status") == "PASS" and gpu_check_result["feature_binding"]["verified"],
            "a PASSing GPU engineering check with a verified P3 binding is required")
    started = time.monotonic() if budget_started is None else budget_started
    accounting = G.Accounting("r18-pdra-stage2", "run-stage2") if accounting is None else accounting
    require(stage1_result_path is not None and stage1_result_sha256 and stage1_closed_sha256
            and stage1_learning_frozen_sha256 and stage1_source_commit,
            "stage 2 requires --stage1-result together with the four SHA/commit pins")

    def guard(label: str) -> dict:
        return G.enforce_runtime(started=started, label=label, caps=STAGE2_CAPS, out=out,
                                 torch_mod=torch, accounting=accounting)

    accounting.phase("stage1_binding")
    binding_record = verify_stage1_binding(stage1_result_path, stage1_result_sha256,
                                           stage1_closed_sha256, stage1_learning_frozen_sha256,
                                           stage1_source_commit)
    stage1_result = binding_record["result"]
    accounting.complete("stage1_binding")

    # rev 8: the audited restore runs first -- deterministic_device_setup refuses to run once
    # CUDA is initialized -- and the arm weights are moved to CUDA only after it completed.
    accounting.phase("restore")
    restored = TAP.restore_frozen_seed0_engine(
        training_source_root=inputs["training_root"], train_contract=inputs["training"],
        binding_path=inputs["binding_path"], checkpoint_path=inputs["checkpoint"]["path"],
        checkpoint_sha256=inputs["checkpoint"]["sha256"], gpu_uuid=inputs["gpu_uuid"])
    engine, binding, model = restored["engine"], restored["binding"], restored["model"]
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.r0.evaluation_data import Development548
    data = Development548(inputs["data_spec"])
    restore_peak_reserved = int(torch.cuda.max_memory_reserved())
    guard("restore")
    torch.cuda.reset_peak_memory_stats()
    accounting.complete("restore")

    accounting.phase("arms")
    arms = L.build_arms()["arms"]
    provenance = {}
    for key, arm in arms.items():
        path = arm_dir / f"{key}.pt"
        require(path.is_file() and not path.is_symlink(), f"arm file missing: {path}")
        entry = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        expect = stage1_result["arms"][key]
        require(Path(expect["path"]) == path and expect["sha256"] == entry["sha256"]
                and expect["bytes"] == entry["bytes"],
                f"arm {key} differs from the stage-1 record")
        entry["stage1_record_sha256_match"] = True
        provenance[key] = entry
        arm.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
        arm.to("cuda:0").eval()
    # capture the detector baseline after the fresh arm construction (which consumes the CPU
    # RNG) and before inference -- the same invariant as stage 1
    before = checkpoint.capture(engine, binding)
    accounting.complete("arms")

    accounting.phase("inference")
    preds = {arm: {600: [], 300: []} for arm in ("a_original_c", "b_single_box_quality",
                                                 "c_whole_box_pairwise", "pdra_difference_strips")}
    stats, calls, decision_seconds, pairs_total = [], 0, 0.0, 0
    decision_totals = {key: {"pairs": 0, "decided": 0, "abstained": 0, "infeasible": 0}
                       for key in ("pdra_difference_strips", "c_whole_box_pairwise", "b_single_box_quality")}
    for k in range(137):
        batch_rows = data.records[4 * k:4 * k + 4]
        with TAP.EncoderTap(model) as tap:
            b = TAP.five_view_forward(data, batch_rows, model, tap=tap)
        calls += b["forward_calls"]
        accounting.forwards(b["forward_calls"])
        for j, row in enumerate(batch_rows):
            image = row["stable_image_id"]
            pool = b["candidates"][j]
            affine_of = _affines(row, b["geometry"][j])
            bank = FEAT.FeatureBank({v: b["p3_maps"][v][j] for v in range(5)})
            records = P.cluster_pair_records(pool)
            pairs_total += len(records)
            original = merge_candidates(pool)
            for budget in (600, 300):
                preds["a_original_c"][budget].extend(prediction(image, c) for c in original[f"top{budget}"])
            d0 = time.perf_counter()
            adapters = {"pdra_difference_strips": L.PairBankAdapter.build(arms["pdra_difference_strips"], bank, pool, affine_of, "pdra"),
                        "c_whole_box_pairwise": L.PairBankAdapter.build(arms["c_whole_box_pairwise"], bank, pool, affine_of, "c"),
                        "b_single_box_quality": L.QualityBankAdapter.build(arms["b_single_box_quality"], bank, pool, affine_of)}
            for key, adapter in adapters.items():
                require(len(adapter.table) == len(records),
                        f"{key} adapter does not cover the complete cluster-internal pair set")
            row_stats = {"image_id": image, "pool": len(pool), "clusters": original["kept_count"],
                         "pairs": len(records)}
            for key, adapter in adapters.items():
                merged = L.merge(pool, affine_of, adapter)
                for name in decision_totals[key]:
                    decision_totals[key][name] += int(merged["decision_stats"][name])
                for budget in (600, 300):
                    preds[key][budget].extend(prediction(image, c) for c in merged[f"top{budget}"])
                row_stats[key] = {k2: merged[k2] for k2 in ("substitutions", "all_abstain_clusters",
                                                             "rank_shifted_members", "budget_membership_delta_vs_C",
                                                             "decision_stats")}
            decision_seconds += time.perf_counter() - d0
            stats.append(row_stats)
            del bank
        guard(f"inference/batch-{k}")
    require(calls == 137 * 5, "stage-2 forward count differs")
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)), "inference changed the detector")
    summary = read(G.STAGE0_DIR / "STAGE0_SUMMARY.json")
    pairs_expected = int(summary["scale"]["all"]["pairs"])
    require(pairs_total == pairs_expected,
            f"decided pair count {pairs_total} differs from the stage-0 total {pairs_expected}")
    accounting.complete("inference")

    accounting.phase("publish")
    refs = {}
    for key, budgets_ in preds.items():
        refs[key] = {}
        for budget, items in budgets_.items():
            refs[key][str(budget)] = publish(out / f"{key}-top{budget}-predictions.json", items)
    publish(out / "REPLACEMENT_STATS.json", {"images": stats, "decision_seconds": decision_seconds,
                                             "decision_stats_totals": decision_totals,
                                             "pairs_decided": pairs_total, "gt_use": GT_USE_STAGE2})
    guard("publish")
    accounting.complete("publish")
    wall = time.monotonic() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    runtime = {"wall_seconds": wall, "peak_allocated_bytes": peak_allocated,
               "peak_reserved_bytes": peak_reserved,
               "restore_peak_reserved_bytes": restore_peak_reserved,
               "vm_hwm_bytes": G.rss_high_water(),
               "memory_peaks_recorded_not_gated": True,
               "memory_gate": ("none: no artificial CUDA threshold and no allocator quota; a real OOM "
                               "is an exception -> STOP.json"),
               "output_bytes": G.output_bytes(out), "caps": STAGE2_CAPS,
               "constraints_enforced_during_run": ["wall_seconds", "output_bytes"],
               "budget_observations": accounting.snapshot()["budget_observations"],
               "within_caps": (wall <= STAGE2_CAPS["wall_seconds"]
                               and G.output_bytes(out) <= STAGE2_CAPS["output_bytes"])}
    require(runtime["within_caps"], f"stage-2 runtime caps exceeded: {runtime}")
    return {"status": "PASS", "source_commit": source_commit, "forward_calls": calls,
            "detector_updates": 0, "predictions": refs,
            "pairs_decided": pairs_total, "pairs_expected": pairs_expected,
            "complete_pair_domain": pairs_total == pairs_expected,
            "decision_stats_totals": decision_totals,
            "decision_seconds": decision_seconds,
            "arms_provenance": provenance,
            "stage1_binding": binding_record["record"],
            "gt_use": GT_USE_STAGE2,
            "scoring": ("separate CPU step: pdra_score.py score (frozen criteria on the eval split; "
                        "train split diagnostic)"),
            "runtime": runtime, "accounting": accounting.snapshot(),
            "elapsed_seconds": wall}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "run-stage1", "run-stage2"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu-check", type=Path, default=None, help="RESULT.json of the PASSing GPU check")
    parser.add_argument("--arms", type=Path, default=None, help="stage-1 arms directory (stage 2)")
    parser.add_argument("--stage1-result", type=Path, default=None,
                        help="stage-1 RESULT.json; arm files are verified against its SHA record (stage 2)")
    parser.add_argument("--stage1-result-sha256", default=None,
                        help="pinned SHA-256 of the stage-1 RESULT.json (stage 2)")
    parser.add_argument("--stage1-closed-sha256", default=None,
                        help="pinned SHA-256 of the stage-1 CLOSED.json (stage 2)")
    parser.add_argument("--stage1-learning-frozen-sha256", default=None,
                        help="pinned SHA-256 of the stage-1 LEARNING_FROZEN.json (stage 2)")
    parser.add_argument("--stage1-source-commit", default=None,
                        help="stage-1 source commit, full 40-hex (stage 2)")
    parser.add_argument("--source-commit", default="UNRECORDED")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if args.mode != "plan":
        prefix = STAGE1_TASK_PREFIX if args.mode == "run-stage1" else STAGE2_TASK_PREFIX
        require(any(part.startswith(prefix) for part in out.parts), "output must sit inside the authorized task directory")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    task = "r18-pdra-stage1" if args.mode != "run-stage2" else "r18-pdra-stage2"
    accounting = G.Accounting(task, args.mode)
    start_record = {"mode": args.mode, "pid": os.getpid(), "started_unix": time.time(),
                    "source_commit": args.source_commit,
                    "caps": STAGE1_CAPS if args.mode != "run-stage2" else STAGE2_CAPS}
    if args.mode == "run-stage2":
        start_record["stage1_binding_pins"] = {
            "result_sha256": args.stage1_result_sha256,
            "closed_sha256": args.stage1_closed_sha256,
            "learning_frozen_sha256": args.stage1_learning_frozen_sha256,
            "source_commit": args.stage1_source_commit}
    publish(out / "START.json", start_record)
    try:
        inputs = G.resolve_inputs(args.parent, G.FROZEN_BATCHES)
        if args.mode == "plan":
            import torch
            result = plan(inputs, out)
            require(not torch.cuda.is_initialized(), "plan initialized CUDA")
            G.enforce_runtime(started=started, label="plan", caps=STAGE1_CAPS, out=out,
                              accounting=accounting)
            result["accounting"] = accounting.snapshot()
        else:
            require(args.gpu_check is not None, "--gpu-check is required")
            gpu_check_result = read(args.gpu_check)
            if args.mode == "run-stage1":
                result = run_stage1(inputs, out, gpu_check_result, args.source_commit,
                                    budget_started=started, accounting=accounting)
            else:
                require(args.arms is not None, "--arms is required for stage 2")
                require(args.stage1_result is not None and args.stage1_result_sha256
                        and args.stage1_closed_sha256 and args.stage1_learning_frozen_sha256
                        and args.stage1_source_commit,
                        "stage 2 requires --stage1-result together with the four SHA/commit pins")
                result = run_stage2(inputs, out, gpu_check_result, args.arms.resolve(), args.source_commit,
                                    stage1_result_path=args.stage1_result.resolve(),
                                    stage1_result_sha256=args.stage1_result_sha256,
                                    stage1_closed_sha256=args.stage1_closed_sha256,
                                    stage1_learning_frozen_sha256=args.stage1_learning_frozen_sha256,
                                    stage1_source_commit=args.stage1_source_commit,
                                    budget_started=started, accounting=accounting)
        result["elapsed_seconds"] = time.monotonic() - started
        ref = publish(out / "RESULT.json", result)
        publish(out / "CLOSED.json", {"result": ref, "returncode": 0, "automatic_retry": False})
        print(json.dumps({"status": result["status"], "result": ref}), flush=True)
        return 0
    except Exception as exc:
        stop = {"status": "FAIL", "mode": args.mode, "out": str(out),
                "source_commit": args.source_commit,
                "exception": {"type": type(exc).__name__, "message": str(exc),
                              "observation": getattr(exc, "observation", None)},
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
                "accounting": accounting.snapshot(), "automatic_retry": False}
        publish(out / "STOP.json", stop)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
