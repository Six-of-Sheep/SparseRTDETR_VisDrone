"""PDRA stage 1 (feature pass + fixed training + eval diagnostics) and stage 2 (full
inference with representative replacement) -- rev 4 runners.

Modes
-----
``plan``        CPU only.  Resolves and SHA-checks every input, re-verifies the frozen
                sampling manifest against the stage-0 rows, instantiates the four arm
                specifications with the real identities, emits the frozen learning spec and
                the stage-1 / stage-2 budgets (forwards, pooled pairs, cache bytes, update
                counts, measured frozen forward time as the basis of the GPU estimate).
``run-stage1``  GPU.  Requires a PASSing GPU engineering check result.  One five-view pass
                over the 548 development images (137 x 5 = 685 batch forwards, zero extra),
                fp16 descriptor caches for the frozen train sample (222,755 pairs) and the
                eval readable pref/abstain pairs, fixed-count training of the three arms,
                eval diagnostics by the frozen buckets, frozen stage-1 gates reported.
``run-stage2``  GPU.  Same forward pass; for every image the complete GT-free inference
                enumeration decides all cluster-internal pairs for PDRA, (c) and (b); the
                representative replacement with re-ranking produces top600/top300
                predictions per arm in the frozen prediction schema, next to original C.
                Scoring is a separate CPU step on the frozen evaluator.

Nothing here reads ground truth at inference; GT is opened only to build training targets
(stage 1) and to read the stage-0 label rows for eval diagnostics.
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

STAGE1_TASK_PREFIX = "r18-pdra-stage1-"
STAGE2_TASK_PREFIX = "r18-pdra-stage2-"
# rev 4: the computed fp16 need is 9.25 GB (slot-padded PDRA sides 6.94 GB + whole-box
# pair cache 1.39 GB + <=0.91 GB member boxes); the rev-3 "<= 8 GB" figure was an
# estimate and is replaced by this computed requirement with a 10 GiB server-only cap.
CACHE_CAP_BYTES = 10 * 1024 ** 3
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


# --- budgets --------------------------------------------------------------------
def budgets(inputs: dict) -> dict:
    import pdra_features as FEAT
    import pdra_learn as L
    summary = read(G.STAGE0_DIR / "STAGE0_SUMMARY.json")
    ledgers = summary["ledgers"]
    full = read(Path(inputs["seed0"]["job_root"]) / "full-C" / "RESULT.json")
    forward_seconds = float(full["timing"]["e2e_total_seconds"])
    train_pairs = inputs["sampling"]["n_pair_total"]
    eval_pref = ledgers["eval"]["feasible_counts"]["pref"]
    eval_abstain = ledgers["eval"]["feasible_counts"]["abstain"]
    stage1_pairs = train_pairs + eval_pref + eval_abstain
    readable_all = sum(ledgers["all"]["feasible_counts"].values())
    spec = L.frozen_spec()
    stage1 = {
        "images": 548, "batch_forwards": 137 * 5, "extra_forwards": 0,
        "pooled_pairs": {"train_sampled": train_pairs, "eval_readable_pref": eval_pref,
                         "eval_readable_abstain": eval_abstain, "total": stage1_pairs},
        "cache_bytes_fp16": {"pdra_sides": FEAT.cache_budget_bytes(stage1_pairs),
                             "c_whole_pairs": 2 * FEAT.whole_dim() * 2 * stage1_pairs,
                             "b_boxes_upper": 2 * FEAT.whole_dim() * 2 * train_pairs,
                             "total_required_upper": (FEAT.cache_budget_bytes(stage1_pairs)
                                                      + 2 * FEAT.whole_dim() * 2 * stage1_pairs
                                                      + 2 * FEAT.whole_dim() * 2 * train_pairs),
                             "cap": CACHE_CAP_BYTES, "server_only": True},
        "updates": {"pdra": spec["updates"], "c": spec["updates"], "b": spec["updates"],
                    "batch": spec["batch_pairs"]},
        "eval_decisions": eval_pref + eval_abstain,
        "time_basis": {"frozen_full_C_e2e_seconds_548_images": forward_seconds,
                       "forward_estimate_seconds": forward_seconds,
                       "pooling_estimate_seconds": "seconds (measured by the GPU check per pair)",
                       "training_estimate": "3 arms x 20,000 updates of <=0.69M-parameter MLPs; minutes",
                       "wall_cap_seconds": 3600, "gpu_hour_cap": 1.0},
        "returned_bytes_cap": 1024 ** 2,
    }
    stage2 = {
        "images": 548, "batch_forwards": 137 * 5, "extra_forwards": 0,
        "pairs_decided": summary["scale"]["all"]["pairs"], "pairs_pooled_readable": readable_all,
        "pooled_on_the_fly_no_cache": True,
        "comparator_batches_per_image_max": int(-(-51210 // 4096)),
        "decision_layer": "pure-Python Copeland over all cluster pairs (measured by the check per pair)",
        "arms": ["a_original_c", "b_single_box_quality", "c_whole_box_pairwise", "pdra_difference_strips"],
        "outputs": "top600/top300 predictions per arm (frozen schema), replacement stats; ~113 MB per 600-file",
        "time_basis": {"frozen_full_C_e2e_seconds_548_images": forward_seconds,
                       "wall_cap_seconds": 5400},
        "scoring": "separate CPU step on the frozen evaluator (score_budget), 2 budgets x 3 new arms",
    }
    return {"stage1": stage1, "stage2": stage2,
            "estimates_marked": "all seconds are estimates until the GPU check and stage runs measure them"}


def plan(inputs: dict, out: Path) -> dict:
    import pdra_learn as L
    verification = L.verify_sampling(G.STAGE0_DIR / "train-sampling-ids.txt",
                                     G.STAGE0_DIR / "seed0-pairs.jsonl", inputs["sampling"])
    require(verification["status"] == "PASS", f"frozen sampling manifest failed verification: {verification}")
    freeze = L.freeze_payload()
    publish(out / "LEARNING_FROZEN.json", freeze)
    publish(out / "SAMPLING_VERIFICATION.json", verification)
    result = {"status": "PLAN_ONLY", "core_version": P.CORE_VERSION,
              "feature_level": P.FEATURE_LEVEL, "feature_binding_verified": P.FEATURE_BINDING_VERIFIED,
              "fairness": inputs["fairness"], "learning_spec_sha256": freeze["learning_spec_sha256"],
              "arms": freeze["arms"], "sampling_verification": verification["status"],
              "budgets": budgets(inputs), "stage0_inputs": inputs["stage0"],
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


def run_stage1(inputs: dict, out: Path, gpu_check_result: dict, source_commit: str) -> dict:
    import torch
    import pdra_features as FEAT
    import pdra_learn as L
    import pdra_tap as TAP
    require(gpu_check_result.get("status") == "PASS" and gpu_check_result["feature_binding"]["verified"],
            "a PASSing GPU engineering check with a verified P3 binding is required")
    started = time.monotonic()
    plan_record = plan(inputs, out)
    publish(out / "PLAN.json", plan_record)
    train_pairs = manifest_pairs(G.STAGE0_DIR / "train-sampling-ids.txt")
    rows = select_rows(train_pairs)
    restored = TAP.restore_frozen_seed0_engine(
        training_source_root=inputs["training_root"], train_contract=inputs["training"],
        binding_path=inputs["binding_path"], checkpoint_path=inputs["checkpoint"]["path"],
        checkpoint_sha256=inputs["checkpoint"]["sha256"], gpu_uuid=inputs["gpu_uuid"])
    engine, binding, model = restored["engine"], restored["binding"], restored["model"]
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.r0.evaluation_data import Development548
    from sparse_rtdetr.r0 import evaluation as ev
    data = Development548(inputs["data_spec"])
    arms = L.build_arms()
    before = checkpoint.capture(engine, binding)
    caches = {"train": defaultdict(list), "eval": defaultdict(list)}
    box_desc, box_target, box_seen = [], [], set()
    calls = 0
    torch.cuda.reset_peak_memory_stats()
    for k in range(137):
        batch_rows = data.records[4 * k:4 * k + 4]
        with TAP.EncoderTap(model) as tap:
            b = TAP.five_view_forward(data, batch_rows, model, tap=tap)
        calls += b["forward_calls"]
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
    require(calls == 137 * 5, "stage-1 forward count differs")
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)), "feature pass changed the detector")
    tensors = {}
    for split in ("train", "eval"):
        c = caches[split]
        tensors[split] = {k2: torch.cat(v) for k2, v in c.items() if k2 != "meta"}
        tensors[split]["meta"] = c["meta"]
    cache_dir = out / "cache"
    cache_refs = {}
    for split in ("train", "eval"):
        payload = {k2: v for k2, v in tensors[split].items()}
        cache_refs[split] = FEAT.save_cache(cache_dir / f"{split}-descriptors.pt", payload)
    cache_refs["b_boxes"] = FEAT.save_cache(cache_dir / "train-boxes.pt",
                                            {"desc": torch.cat(box_desc), "target": torch.tensor(box_target)})
    total_cache = sum(ref["bytes"] for ref in cache_refs.values())
    require(total_cache <= CACHE_CAP_BYTES, "descriptor cache exceeds the 8 GB cap")
    publish(out / "CACHE.json", {"refs": cache_refs, "bytes": total_cache,
                                 "train_pairs": int(tensors["train"]["s_i"].shape[0]),
                                 "eval_pairs": int(tensors["eval"]["s_i"].shape[0]),
                                 "b_boxes": len(box_target)})
    require(int(tensors["train"]["s_i"].shape[0]) == inputs["sampling"]["n_pair_total"], "train cache size differs from the manifest")
    # fixed training
    train_reports = {}
    tr = tensors["train"]
    pdra_tensors = L.PairTensors(tr["s_i"], tr["s_j"], tr["geom"], tr["pref"], tr["abstain"])
    c_tensors = L.PairTensors(tr["w_i"], tr["w_j"], tr["geom"], tr["pref"], tr["abstain"])
    train_reports["pdra_difference_strips"] = L.train_pair_arm(arms["arms"]["pdra_difference_strips"], pdra_tensors)
    train_reports["c_whole_box_pairwise"] = L.train_pair_arm(arms["arms"]["c_whole_box_pairwise"], c_tensors)
    train_reports["b_single_box_quality"] = L.train_quality_arm(arms["arms"]["b_single_box_quality"],
                                                                torch.cat(box_desc), torch.tensor(box_target))
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)), "training touched the detector")
    arm_refs = {}
    for key, arm in arms["arms"].items():
        path = out / "arms" / f"{key}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(arm.state_dict(), path)
        arm_refs[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
    # eval diagnostics (labels come from the stage-0 rows, never from the decision path)
    diagnostics = evaluate_arms(arms["arms"], tensors["eval"], L, FEAT)
    publish(out / "DIAGNOSTICS.json", diagnostics)
    wall = time.monotonic() - started
    return {"status": "PASS", "task": out.parts[-2] if len(out.parts) > 1 else str(out),
            "source_commit": source_commit, "forward_calls": calls, "detector_updates": 0,
            "training": train_reports, "arms": arm_refs, "cache": cache_refs,
            "gates": diagnostics["gates"], "elapsed_seconds": wall,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def evaluate_arms(arms: dict, ev_t: dict, L, FEAT) -> dict:
    """Per-arm eval diagnostics on the eval readable pref/abstain pairs, frozen buckets."""
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
                               "cross_gt_misarbitration_rate": (abstain_votes / abstain_pairs) if abstain_pairs else None}
    def acc(key):
        return report["arms"][key]["buckets"].get("clear", {}).get("accuracy_counting_abstain_as_wrong")
    def mis(key):
        return report["arms"][key]["cross_gt_misarbitration_rate"]
    main_gate = (acc("pdra_difference_strips") is not None and acc("b_single_box_quality") is not None
                 and acc("c_whole_box_pairwise") is not None
                 and acc("pdra_difference_strips") >= max(acc("b_single_box_quality"), acc("c_whole_box_pairwise")) + 0.05)
    safety_gate = (mis("pdra_difference_strips") is not None and mis("c_whole_box_pairwise") is not None
                   and mis("pdra_difference_strips") <= mis("c_whole_box_pairwise") + 0.02)
    report["gates"] = {"main_clear_pairs_acc_pdra_ge_max_b_c_plus_5pp": main_gate,
                       "safety_cross_gt_misarbitration_le_c_plus_2pp": safety_gate,
                       "metric_definition": "accuracy counts abstentions as wrong (coverage reported separately)",
                       "automatic_next_stage": False}
    return report


# --- stage 2 ----------------------------------------------------------------------
def prediction(image_id, candidate) -> dict:
    return {"image_id": image_id, "query_id": candidate.query_id, "category_id": candidate.category_id,
            "score": candidate.score, "bbox_xyxy": list(candidate.box_xyxy),
            "view_id": candidate.view_id, "view_candidate_rank": candidate.view_candidate_rank}


def run_stage2(inputs: dict, out: Path, gpu_check_result: dict, arm_dir: Path, source_commit: str) -> dict:
    import torch
    import pdra_features as FEAT
    import pdra_learn as L
    import pdra_tap as TAP
    from slice_core import merge_candidates
    require(gpu_check_result.get("status") == "PASS" and gpu_check_result["feature_binding"]["verified"],
            "a PASSing GPU engineering check with a verified P3 binding is required")
    started = time.monotonic()
    arms = L.build_arms()["arms"]
    for key, arm in arms.items():
        arm.load_state_dict(torch.load(str(arm_dir / f"{key}.pt"), map_location="cpu", weights_only=True))
        arm.to("cuda:0").eval()
    restored = TAP.restore_frozen_seed0_engine(
        training_source_root=inputs["training_root"], train_contract=inputs["training"],
        binding_path=inputs["binding_path"], checkpoint_path=inputs["checkpoint"]["path"],
        checkpoint_sha256=inputs["checkpoint"]["sha256"], gpu_uuid=inputs["gpu_uuid"])
    engine, binding, model = restored["engine"], restored["binding"], restored["model"]
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.r0.evaluation_data import Development548
    data = Development548(inputs["data_spec"])
    before = checkpoint.capture(engine, binding)
    preds = {arm: {600: [], 300: []} for arm in ("a_original_c", "b_single_box_quality",
                                                 "c_whole_box_pairwise", "pdra_difference_strips")}
    stats, calls, decision_seconds = [], 0, 0.0
    for k in range(137):
        batch_rows = data.records[4 * k:4 * k + 4]
        with TAP.EncoderTap(model) as tap:
            b = TAP.five_view_forward(data, batch_rows, model, tap=tap)
        calls += b["forward_calls"]
        for j, row in enumerate(batch_rows):
            image = row["stable_image_id"]
            pool = b["candidates"][j]
            affine_of = _affines(row, b["geometry"][j])
            bank = FEAT.FeatureBank({v: b["p3_maps"][v][j] for v in range(5)})
            original = merge_candidates(pool)
            for budget in (600, 300):
                preds["a_original_c"][budget].extend(prediction(image, c) for c in original[f"top{budget}"])
            d0 = time.perf_counter()
            adapters = {"pdra_difference_strips": L.PairBankAdapter.build(arms["pdra_difference_strips"], bank, pool, affine_of, "pdra"),
                        "c_whole_box_pairwise": L.PairBankAdapter.build(arms["c_whole_box_pairwise"], bank, pool, affine_of, "c"),
                        "b_single_box_quality": L.QualityBankAdapter.build(arms["b_single_box_quality"], bank, pool, affine_of)}
            row_stats = {"image_id": image, "pool": len(pool), "clusters": original["kept_count"]}
            for key, adapter in adapters.items():
                merged = L.merge(pool, affine_of, adapter)
                for budget in (600, 300):
                    preds[key][budget].extend(prediction(image, c) for c in merged[f"top{budget}"])
                row_stats[key] = {k2: merged[k2] for k2 in ("substitutions", "all_abstain_clusters",
                                                             "rank_shifted_members", "budget_membership_delta_vs_C")}
            decision_seconds += time.perf_counter() - d0
            stats.append(row_stats)
            del bank
    require(calls == 137 * 5, "stage-2 forward count differs")
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)), "inference changed the detector")
    refs = {}
    for key, budgets_ in preds.items():
        refs[key] = {}
        for budget, items in budgets_.items():
            refs[key][str(budget)] = publish(out / f"{key}-top{budget}-predictions.json", items)
    publish(out / "REPLACEMENT_STATS.json", {"images": stats, "decision_seconds": decision_seconds})
    return {"status": "PASS", "source_commit": source_commit, "forward_calls": calls, "detector_updates": 0,
            "predictions": refs, "decision_seconds": decision_seconds,
            "scoring": "not run here; score each arm/budget with the frozen evaluator in a separate CPU step",
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "run-stage1", "run-stage2"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu-check", type=Path, default=None, help="RESULT.json of the PASSing GPU check")
    parser.add_argument("--arms", type=Path, default=None, help="stage-1 arms directory (stage 2)")
    parser.add_argument("--source-commit", default="UNRECORDED")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if args.mode != "plan":
        prefix = STAGE1_TASK_PREFIX if args.mode == "run-stage1" else STAGE2_TASK_PREFIX
        require(any(part.startswith(prefix) for part in out.parts), "output must sit inside the authorized task directory")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    publish(out / "START.json", {"mode": args.mode, "pid": os.getpid(), "started_unix": time.time(),
                                 "source_commit": args.source_commit})
    try:
        inputs = G.resolve_inputs(args.parent, G.FROZEN_BATCHES)
        if args.mode == "plan":
            import torch
            result = plan(inputs, out)
            require(not torch.cuda.is_initialized(), "plan initialized CUDA")
        else:
            require(args.gpu_check is not None, "--gpu-check is required")
            gpu_check_result = read(args.gpu_check)
            if args.mode == "run-stage1":
                result = run_stage1(inputs, out, gpu_check_result, args.source_commit)
            else:
                require(args.arms is not None, "--arms is required for stage 2")
                result = run_stage2(inputs, out, gpu_check_result, args.arms.resolve(), args.source_commit)
        result["elapsed_seconds"] = time.monotonic() - started
        ref = publish(out / "RESULT.json", result)
        publish(out / "CLOSED.json", {"result": ref, "returncode": 0, "automatic_retry": False})
        print(json.dumps({"status": result["status"], "result": ref}), flush=True)
        return 0
    except Exception:
        publish(out / "STOP.json", {"traceback": traceback.format_exc(),
                                    "elapsed_seconds": time.monotonic() - started, "automatic_retry": False})
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
