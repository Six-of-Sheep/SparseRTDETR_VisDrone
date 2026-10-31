"""PDRA frozen scoring step (CPU) and chain cost wrap-up (rev 1).

The criteria in ``CRITERIA`` were frozen on 2026-10-31 BEFORE any stage-1 learning result
or detection score existed, together with the capacity-matched learning spec.  The scoring
step evaluates the actual detection outputs of the four arms on the module-untrained eval
sequences (210 images, primary) and on the module-train sequences (diagnostic) with the
frozen CPU evaluator ``score_budget`` of the certified seed0 source release, and publishes
the pre-registered cell table and verdict.

Nothing here calls the detector: the two five-view passes (stage 1, stage 2) and the
training updates are the entire model compute; scoring reads prediction JSONs only.

Modes
-----
``plan``   CPU.  Resolves the frozen inputs and the image split, emits the exact score
           calls and the caps.  No evaluator import, no prediction needed.
``score``  CPU.  Scores every arm x budget on the eval and train splits, applies the
           frozen criteria and publishes SCORE.json plus per-call metric files.
``wrap``   CPU.  Reads the stage-1 / stage-2 RESULTs and SCORE.json, totals the wall
           clock, peaks, server bytes and the return whitelist, publishes
           COST_SUMMARY.json inside the score directory.

Scoring imports the evaluator from the certified seed0 training release (guarded by
``pdra_tap._require_training_source``) so the evaluator config and authority manifest
identities match the frozen certified chain.
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

import pdra_gpu_check as G

TASK_PREFIX = "r18-pdra-score-"
WALL_CAP_SECONDS = 1800
OUTPUT_CAP_BYTES = 256 * 1024 ** 2
CAPS = {"wall_seconds": WALL_CAP_SECONDS, "output_bytes": OUTPUT_CAP_BYTES}
ARMS = ("a_original_c", "b_single_box_quality", "c_whole_box_pairwise", "pdra_difference_strips")
PDRA_ARM = "pdra_difference_strips"
CONTROLS = ("a_original_c", "b_single_box_quality", "c_whole_box_pairwise")
PRIMARY_SPLIT, DIAGNOSTIC_SPLIT = "eval", "train"
PRIMARY_IMAGES, DIAGNOSTIC_IMAGES = 210, 338
BUDGETS = (600, 300)
METRIC_KEYS = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AP75_small",
               "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")

CRITERIA = {
    "spec_version": "pdra-criteria-v1-20261031",
    "frozen_on": "2026-10-31",
    "frozen_before": "any stage-1 learning result or detection score",
    "primary_question": ("on the module-untrained eval sequences, does PDRA simultaneously "
                         "outperform (a) original C, (b) the capacity-matched single-box quality "
                         "control and (c) the capacity-matched whole-box pairwise control?"),
    "primary_split": PRIMARY_SPLIT,
    "primary_images": PRIMARY_IMAGES,
    "diagnostic_split": DIAGNOSTIC_SPLIT,
    "diagnostic_images": DIAGNOSTIC_IMAGES,
    "budgets": list(BUDGETS),
    "primary_budget": 600,
    "metric_source": "COCO_bridge.metrics of the frozen score_budget evaluator (original 0..1 units)",
    "delta_unit": "pp = (PDRA - control) in 0..1, multiplied by 100 exactly once",
    "gates_primary_600_pp": {"AP": 0.20, "APS": 0.30, "AP75_small": 0.0, "APL": -0.20},
    "gates_secondary_300_pp": {"AP": -0.20, "APS": -0.20},
    "applies_to": list(CONTROLS),
    "large_object_protection": ("APL >= -0.20pp versus every control at the primary budget; the "
                                "stage-1 area-bucket diagnostics and the per-control APL cells are "
                                "reported alongside"),
    "diagnostics_only": ("clear-pair accuracy, coverage, abstention behaviour and the delta-IoU "
                         "buckets are diagnostics; they never gate the detection evaluation and "
                         "are not part of this verdict"),
    "evidence_rule": ("a cell is insufficient_evidence when the frozen evaluator returns -1 for "
                      "the metric on the eval split (no GT in that area/class aggregation); "
                      "insufficient cells cannot pass; any insufficient cell makes the verdict "
                      "EVIDENCE_INSUFFICIENT (reported as such, not as a scientific failure)"),
    "verdict_rule": ("PASS iff every (control, budget, metric) cell holds on the eval split; "
                     "FAIL otherwise; a FAIL closes only this frozen configuration"),
    "conclusion_semantics": {
        "pdra_not_above_c": "the mechanism adds nothing over the original selection on this configuration",
        "pdra_ties_b": ("no incremental value of relative judgement over the capacity-matched "
                        "independent quality head on this configuration"),
        "pdra_ties_c": ("no incremental value of reading the difference region over reading whole "
                        "boxes with the same architecture on this configuration"),
        "scope": ("a single result closes only the frozen configuration it measured; it does not "
                  "invalidate the method class, and no threshold is modified after seeing results"),
    },
    "scope_statement": ("eval split only; one frozen configuration; criteria frozen before any "
                        "learning or detection result existed"),
}

require, publish, read, exact = G.require, G.publish, G.read, G.exact


def criteria_sha256(criteria: dict | None = None) -> str:
    payload = criteria if criteria is not None else CRITERIA
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def split_images() -> tuple[list, list]:
    """The frozen module train/eval image split (338 / 210) from the stage-0 counts."""
    counts = read(G.STAGE0_DIR / "seed0-IMAGE_COUNTS.json")
    require(isinstance(counts, list) and len(counts) == 548, "image-counts file differs")
    train = [row["image_id"] for row in counts if row["split"] == "train"]
    evaluation = [row["image_id"] for row in counts if row["split"] == "eval"]
    require(len(train) == DIAGNOSTIC_IMAGES and len(evaluation) == PRIMARY_IMAGES,
            "split sizes differ from the frozen 338 train / 210 eval")
    return train, evaluation


def verdict(eval_metrics: dict) -> dict:
    """Apply the frozen cell table to the eval-split metrics of every arm.

    ``eval_metrics`` is ``{arm: {budget: {metric: value_in_0..1_or_-1}}}``.
    """
    cells = []
    for budget, table in ((600, CRITERIA["gates_primary_600_pp"]),
                          (300, CRITERIA["gates_secondary_300_pp"])):
        for control in CONTROLS:
            for metric, threshold in table.items():
                pdra_value = eval_metrics[PDRA_ARM][budget].get(metric)
                control_value = eval_metrics[control][budget].get(metric)
                cell = {"budget": budget, "control": control, "metric": metric,
                        "pdra": pdra_value, "control_value": control_value,
                        "delta_pp": None, "threshold_pp": threshold, "status": None}
                if (pdra_value is None or control_value is None
                        or pdra_value < 0 or control_value < 0):
                    cell["status"] = "insufficient_evidence"
                    cells.append(cell)
                    continue
                delta = (pdra_value - control_value) * 100.0
                cell["delta_pp"] = delta
                cell["status"] = "pass" if delta >= threshold else "fail"
                cells.append(cell)
    failed = [f"{c['control']}@{c['budget']}:{c['metric']}" for c in cells if c["status"] == "fail"]
    insufficient = [f"{c['control']}@{c['budget']}:{c['metric']}"
                    for c in cells if c["status"] == "insufficient_evidence"]
    if insufficient:
        status = "EVIDENCE_INSUFFICIENT"
    elif failed:
        status = "FAIL"
    else:
        status = "PASS"
    return {"status": status, "cells": cells, "failed_cells": failed,
            "insufficient_cells": insufficient,
            "scope": CRITERIA["scope_statement"]}


def _scoring_modules(inputs):
    """Certified evaluator imports: the seed0 training release must win the sys.path race."""
    import pdra_tap as TAP
    TAP._require_training_source(inputs["training_root"])
    import torch
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "scoring must run with CUDA hidden and must not initialize CUDA")
    from sparse_rtdetr.r0.evaluation import score_budget
    from sparse_rtdetr.r0.evaluation_data import Development548
    return score_budget, Development548


def plan(inputs: dict, out: Path) -> dict:
    import pdra_learn as L
    train, evaluation = split_images()
    return {"status": "PLAN_ONLY",
            "criteria": CRITERIA, "criteria_sha256": criteria_sha256(),
            "learning_spec_sha256": L.learning_spec_sha(),
            "split_images": {"eval": len(evaluation), "train": len(train)},
            "score_calls": [{"split": split, "arm": arm, "budget": budget}
                            for split in (PRIMARY_SPLIT, DIAGNOSTIC_SPLIT)
                            for arm in ARMS for budget in BUDGETS],
            "evaluator": ("frozen score_budget (official + COCO bridge) on the certified seed0 "
                          "training release"),
            "prediction_files_expected": [f"{arm}-top{budget}-predictions.json"
                                          for arm in ARMS for budget in BUDGETS],
            "model_calls": 0, "caps": CAPS, "cuda_initialized": False}


def score(inputs: dict, stage2_dir: Path, out: Path, source_commit: str, *,
          budget_started: float | None = None) -> dict:
    started = time.monotonic() if budget_started is None else budget_started

    def guard(label: str) -> dict:
        return G.enforce_runtime(started=started, label=label, caps=CAPS, out=out)

    plan_record = plan(inputs, out)
    publish(out / "PLAN.json", plan_record)
    guard("plan")
    score_budget, Development548 = _scoring_modules(inputs)
    train_ids, eval_ids = split_images()
    data = Development548(inputs["data_spec"])
    rows_by_id = {row["stable_image_id"]: row for row in data.records}
    require(set(rows_by_id) == set(train_ids) | set(eval_ids),
            "development membership differs from the frozen split")
    predictions = {}
    for arm in ARMS:
        for budget in BUDGETS:
            path = stage2_dir / f"{arm}-top{budget}-predictions.json"
            require(path.is_file() and not path.is_symlink(), f"missing predictions: {path}")
            items = read(path)
            predictions[(arm, budget)] = {"path": str(path), "bytes": path.stat().st_size,
                                          "sha256": G.sha(path), "items": items}
    metrics = {}
    coverage = {}
    for split_name, ids in ((PRIMARY_SPLIT, eval_ids), (DIAGNOSTIC_SPLIT, train_ids)):
        id_set = set(ids)
        rows = [row for row in data.records if row["stable_image_id"] in id_set]
        require(len(rows) == len(ids), "split rows differ from the frozen split")
        parsed = [data.raw(row) for row in rows]
        metrics[split_name] = {}
        coverage[split_name] = {}
        for arm in ARMS:
            metrics[split_name][arm] = {}
            coverage[split_name][arm] = {}
            for budget in BUDGETS:
                bundle = predictions[(arm, budget)]
                selected = [p for p in bundle["items"] if p["image_id"] in id_set]
                for p in selected:
                    require(set(p) == {"image_id", "query_id", "category_id", "score",
                                       "bbox_xyxy", "view_id", "view_candidate_rank"},
                            "prediction schema differs")
                counts = Counter(p["image_id"] for p in selected)
                require(set(counts) <= id_set, "a prediction names an image outside the split")
                require(all(value <= budget for value in counts.values()),
                        f"an image exceeds the top{budget} budget")
                result = score_budget(rows, parsed, selected, str(inputs["training_root"]))
                bridge = result["COCO_bridge"]["metrics"]
                metrics[split_name][arm][budget] = {key: float(bridge[key]) for key in METRIC_KEYS}
                coverage[split_name][arm][budget] = {
                    "images_with_predictions": len(counts), "images_in_split": len(ids),
                    "detections": len(selected)}
                publish(out / f"metrics-{split_name}-{arm}-top{budget}.json",
                        {"arm": arm, "split": split_name, "budget": budget,
                         "COCO_bridge": result["COCO_bridge"],
                         "official": result["official"],
                         "spatial_filter_counts": result.get("spatial_filter_counts")})
                guard(f"score/{split_name}/{arm}/{budget}")
    verdict_record = verdict(metrics[PRIMARY_SPLIT])
    return {"status": verdict_record["status"],
            "task": out.parts[-2] if len(out.parts) > 1 else str(out),
            "source_commit": source_commit,
            "criteria": CRITERIA, "criteria_sha256": criteria_sha256(),
            "split_images": {"eval": len(eval_ids), "train": len(train_ids)},
            "metrics": metrics, "coverage": coverage, "verdict": verdict_record,
            "diagnostics_note": ("train-split metrics and all non-cell metrics are diagnostic; the "
                                 "delta-IoU buckets, abstention behaviour and area-bucket tables "
                                 "live in the stage-1 DIAGNOSTICS.json and the stage-2 "
                                 "REPLACEMENT_STATS.json"),
            "prediction_files": {f"{arm}-top{budget}": {k: v for k, v in predictions[(arm, budget)].items()
                                                        if k != "items"}
                                 for arm in ARMS for budget in BUDGETS},
            "model_calls": 0, "holdout_accessed": False,
            "elapsed_seconds": time.monotonic() - started}


def dir_bytes(path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def wrap(stage1_dir: Path, stage2_dir: Path, score_dir: Path) -> dict:
    """Cost wrap-up of the chain: wall clock, peaks, server bytes and return whitelist."""
    s1 = read(stage1_dir / "RESULT.json")
    s2 = read(stage2_dir / "RESULT.json")
    sc = read(score_dir / "SCORE.json")
    whitelist = [
        ("stage1", stage1_dir / "RESULT.json"), ("stage1", stage1_dir / "PLAN.json"),
        ("stage1", stage1_dir / "CACHE.json"), ("stage1", stage1_dir / "DIAGNOSTICS.json"),
        ("stage1", stage1_dir / "LEARNING_FROZEN.json"),
        ("stage1", stage1_dir / "SAMPLING_VERIFICATION.json"),
        ("stage1", stage1_dir / "CRITERIA_FROZEN.json"), ("stage1", stage1_dir / "CLOSED.json"),
        ("stage2", stage2_dir / "RESULT.json"), ("stage2", stage2_dir / "CLOSED.json"),
        ("score", score_dir / "SCORE.json"), ("score", score_dir / "PLAN.json"),
        ("score", score_dir / "CLOSED.json"),
    ]
    files = [{"stage": stage, "path": str(path), "bytes": path.stat().st_size}
             for stage, path in whitelist if path.is_file()]
    total = sum(item["bytes"] for item in files)
    require(total + 4096 <= 1024 ** 2,
            f"return whitelist {total} bytes leaves no headroom under 1 MiB")
    steps = [
        {"step": "stage1", "wall_seconds": s1["runtime"]["wall_seconds"],
         "peak_allocated_bytes": s1["runtime"]["peak_allocated_bytes"],
         "peak_reserved_bytes": s1["runtime"]["peak_reserved_bytes"],
         "output_bytes": dir_bytes(stage1_dir), "caps": s1["runtime"]["caps"]},
        {"step": "stage2", "wall_seconds": s2["runtime"]["wall_seconds"],
         "peak_allocated_bytes": s2["runtime"]["peak_allocated_bytes"],
         "peak_reserved_bytes": s2["runtime"]["peak_reserved_bytes"],
         "output_bytes": dir_bytes(stage2_dir), "caps": s2["runtime"]["caps"]},
        {"step": "score", "wall_seconds": sc["elapsed_seconds"],
         "output_bytes": dir_bytes(score_dir), "caps": CAPS},
    ]
    return {"status": "PASS",
            "steps": steps,
            "total_wall_seconds": sum(step["wall_seconds"] for step in steps),
            "peak_reserved_bytes_max": max(step.get("peak_reserved_bytes") or 0 for step in steps),
            "server_bytes_added": dir_bytes(stage1_dir) + dir_bytes(stage2_dir) + dir_bytes(score_dir),
            "returned_whitelist": {"files": files, "bytes": total, "cap_bytes": 1024 ** 2,
                                   "note": "COST_SUMMARY.json itself adds a few KiB"},
            "predictions_returned": False, "cache_returned": False,
            "verdict_status": sc["status"], "automatic_retry": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "score", "wrap"))
    parser.add_argument("--parent", type=Path, default=None)
    parser.add_argument("--stage1", type=Path, default=None)
    parser.add_argument("--stage2", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", default="UNRECORDED")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    require(any(part.startswith(TASK_PREFIX) for part in out.parts),
            "output must sit inside the authorized score task directory " + TASK_PREFIX)
    if args.mode == "wrap":
        require(args.stage1 is not None and args.stage2 is not None,
                "--stage1 and --stage2 are required for wrap")
        require((out / "SCORE.json").is_file(), "wrap needs the score step's SCORE.json")
        try:
            payload = wrap(args.stage1.resolve(), args.stage2.resolve(), out)
            ref = publish(out / "COST_SUMMARY.json", payload)
            print(json.dumps({"status": "PASS", "cost_summary": ref}), flush=True)
            return 0
        except Exception:
            traceback.print_exc()
            return 2
    require(args.parent is not None, "--parent is required")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    publish(out / "START.json", {"mode": args.mode, "pid": os.getpid(), "started_unix": time.time(),
                                 "source_commit": args.source_commit, "caps": CAPS})
    try:
        inputs = G.resolve_inputs(args.parent, ())
        if args.mode == "plan":
            import torch
            result = plan(inputs, out)
            require(not torch.cuda.is_initialized(), "plan initialized CUDA")
            G.enforce_runtime(started=started, label="plan", caps=CAPS, out=out)
        else:
            require(args.stage2 is not None, "--stage2 is required")
            result = score(inputs, args.stage2.resolve(), out, args.source_commit,
                           budget_started=started)
        result["elapsed_seconds"] = time.monotonic() - started
        primary = out / ("SCORE.json" if args.mode == "score" else "RESULT.json")
        ref = publish(primary, result)
        publish(out / "CLOSED.json", {"result": ref, "returncode": 0, "automatic_retry": False})
        print(json.dumps({"status": result["status"], "result": ref}), flush=True)
        return 0
    except Exception as exc:
        stop = {"status": "FAIL", "mode": args.mode, "out": str(out),
                "source_commit": args.source_commit,
                "exception": {"type": type(exc).__name__, "message": str(exc),
                              "observation": getattr(exc, "observation", None)},
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started, "automatic_retry": False}
        publish(out / "STOP.json", stop)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
