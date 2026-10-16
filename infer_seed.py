"""Seed-specific frozen SAHI readout. Caller owns restore, resources and lifecycle.

No training dispatch lives here. A is generated independently for each seed,
then every later whole-image branch must reproduce that seed's raw cache.
The frozen slice_core module is reused without scientific changes.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import os
from pathlib import Path
import time
import traceback

from slice_core import (Candidate, EXPECTED_SHAPE, map_crop_xyxy, merge_candidates,
                        select_view_candidates, tile_windows)

BUDGETS = (600, 300)
METRICS = ("AP", "APS", "AP75_small", "APM", "APL", "AP50", "AP75")
MAIN_LIMITS_PP = {"AP": .30, "APS": .50, "AP75_small": -.20,
                  "APM": -.50, "APL": -.50}
SECONDARY_LIMITS_PP = {"AP": -.20, "APS": -.20}
WORKLOADS = {"infer-A": (548, 0, 1), "accept-A": (64, 2, 1),
             "accept-B": (64, 2, 1), "accept-C": (64, 2, 5),
             "full-C": (548, 2, 5)}


def _reference(ref):
    from sparse_rtdetr.localization.runio import reference, require
    actual = reference(ref["path"])
    expected_size = ref.get("bytes", ref.get("size_bytes"))
    require(actual["sha256"] == ref["sha256"] and actual["bytes"] == expected_size,
            "bound file missing, damaged or size changed: " + ref["path"])
    return actual


def _read_ref(ref):
    from sparse_rtdetr.localization.runio import read
    _reference(ref)
    return read(ref["path"])


def _full_ids(c):
    from sparse_rtdetr.localization.runio import require
    ids = c.get("development_ids", c.get("full_image_ids"))
    require(isinstance(ids, list) and len(ids) == len(set(ids)) == 548,
            "development membership is not the frozen548")
    return ids


def _scope(c):
    from sparse_rtdetr.localization.runio import require
    require(type(c.get("seed")) is int and c["seed"] in (1, 2),
            "only the frozen seed1/seed2 repeats are authorized")
    require(c.get("scope") == f"formal_R18_seed{c['seed']}_e120",
            "explicit seed-specific model scope missing")
    cp = c["checkpoint"]
    require(cp.get("completed_updates") == 36480,
            "inference checkpoint is not the e120 terminal state")
    if "scope" in cp:
        require(cp["scope"] == c["scope"], "checkpoint scope differs from seed identity")
    return cp


def _dataset(c):
    from sparse_rtdetr.r0.evaluation_data import Development548
    from sparse_rtdetr.localization.runio import require
    data = Development548(c["data"]["development"])
    require([r["stable_image_id"] for r in data.records] == _full_ids(c),
            "frozen development order changed")
    return data


def load_original(data, row):
    """Only original RGB image bytes; annotations are not opened here."""
    from PIL import Image
    from sparse_rtdetr.localization.data import read_bound
    raw = read_bound(Path(data.spec["roots"]["images"]) / Path(row["relative_path"]).name,
                     row["image_sha256"], row["image_size_bytes"])
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert("RGB")
        image.load()
    if (image.width, image.height) != (row["width"], row["height"]):
        raise ValueError("original dimensions differ from bound image metadata")
    return image


def pixels(data, image):
    from torchvision.transforms.v2 import functional as TF
    return TF.pil_to_tensor(data.transform.resize(image)).float() / 255.


def make_view_pixels(data, rows, sliced):
    views = [[] for _ in range(5 if sliced else 1)]
    geometry = []
    for row in rows:
        image = load_original(data, row)
        try:
            views[0].append(pixels(data, image))
            tiles = tile_windows(image.height, image.width)
            geometry.append([t.as_dict() for t in tiles])
            if sliced:
                for tile in tiles:
                    crop = image.crop(tile.crop_box)
                    try:
                        views[tile.view_id].append(pixels(data, crop))
                    finally:
                        crop.close()
        finally:
            image.close()
    return views, geometry


def prediction(image_id, candidate, *, whole=False):
    result = {"image_id": image_id, "query_id": candidate.query_id,
              "category_id": candidate.category_id, "score": candidate.score,
              "bbox_xyxy": list(candidate.box_xyxy)}
    if not whole:
        result.update(view_id=candidate.view_id,
                      view_candidate_rank=candidate.view_candidate_rank)
    return result


def save_tensor(path, value):
    import torch
    from sparse_rtdetr.localization.runio import fsync_directory, reference
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    ref = reference(partial)
    os.link(partial, path)
    partial.unlink()
    fsync_directory(path.parent)
    ref["path"] = str(path)
    return ref


def _identity(c):
    return {"task_id": c["task_id"], "seed": c["seed"], "scope": c["scope"],
            "source_sha256": c["source_sha256"],
            "contract_sha256": c["contract_sha256"],
            "checkpoint_sha256": c["checkpoint"]["sha256"]}


def _closed_result(c, phase):
    from sparse_rtdetr.localization.runio import read, reference, require
    if phase == 'infer-A' and c.get('parent_reuse'):
        from resume_support import original_A_contract
        c = original_A_contract(c)
    out = Path(c["job_root"]) / phase
    closed = read(out / "CLOSED.json")
    require(closed["status"] == "PASS" and closed["seed"] == c["seed"]
            and closed["mode"] == phase
            and closed["result"] == reference(out / "RESULT.json"),
            "phase closure/result bytes differ: " + phase)
    result = read(out / "RESULT.json")
    for key, value in _identity(c).items():
        require(result[key] == value, "phase identity differs: " + phase + "/" + key)
    return result


def _load_own_raw(c, data):
    import torch
    from sparse_rtdetr.localization.runio import read, require
    from sparse_rtdetr.localization.consumer_geometry import positive_box
    result = _closed_result(c, "infer-A")
    require(result["status"] == "PASS" and result["phase"] == "infer-A",
            "seed-specific original A phase is not closed PASS")
    from resume_support import original_A_contract
    cache_identity = _identity(original_A_contract(c))
    for key, value in cache_identity.items():
        require(result[key] == value, "original A identity differs: " + key)
    ref = result["raw"]
    _reference(ref)
    raw = torch.load(ref["path"], map_location="cpu", weights_only=True)
    require(raw["schema"] == "r18-sahi-repeat-own-A-raw-v1", "own raw schema differs")
    for key, value in cache_identity.items():
        require(raw[key] == value, "own raw identity differs: " + key)
    require(raw["image_ids"] == [r["stable_image_id"] for r in data.records]
            and raw["epoch"] == 120 and raw["EMA"] is True,
            "own raw image/EMA identity differs")
    require(raw["pred_logits"].shape == (548, 300, 10)
            and raw["pred_boxes"].shape == (548, 300, 4)
            and raw["pred_logits"].dtype == raw["pred_boxes"].dtype == torch.float32
            and torch.equal(raw["query_ids"], torch.arange(300))
            and torch.isfinite(raw["pred_logits"]).all(), "own raw shape/dtype differs")
    positive_box("sahi-repeat/own-A-raw", raw["pred_boxes"])
    return raw, result


def cpu_checks(c):
    """Actual new image/crop path against the certified original CPU transform.

    No checkpoint or model is needed before training. GT is used by the legacy
    transform comparison only; the new view function itself never opens GT.
    """
    import torch
    from sparse_rtdetr.localization.runio import require
    require(not torch.cuda.is_initialized(), "CPU inference interface initialized CUDA")
    require(c["seed"] in (1, 2) and type(c["seed"]) is int, "CPU seed scope differs")
    data = _dataset(c)
    reference_ref = c["seed0_math_cache"]
    _reference(reference_ref)
    original = torch.load(reference_ref["path"], map_location="cpu", weights_only=True)
    require(original["schema"] == "native_r18_common_raw_queries_v1"
            and original["image_ids"] == _full_ids(c)
            and original["checkpoint_sha256"] ==
                "13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757",
            "seed0 cached math fixture is not the certified original endpoint")
    from sparse_rtdetr.r0.evaluation import rank_candidates
    ids = c["fixed64"]
    require(len(ids) == len(set(ids)) == 64, "CPU interface needs original fixed64")
    rows = []
    for image_id in ids:
        row = data.by_id[image_id]
        views, geometry = make_view_pixels(data, [row], True)
        expected, _ = data.item(row, data.records.index(row))
        require(torch.equal(views[0][0], expected),
                "new GT-free whole pixels differ from original development transform")
        i = _full_ids(c).index(image_id)
        boxes, logits = original["pred_boxes"][i], original["pred_logits"][i]
        xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2), -1)
        xyxy *= torch.tensor([row["width"], row["height"]] * 2, dtype=torch.float32)
        selected = select_view_candidates(xyxy.tolist(), logits.sigmoid().numpy(), view_id=0)
        require([prediction(image_id, p, whole=True) for p in selected]
                == rank_candidates(logits, boxes, row, 600),
                "cached mathematical fixture decode/rank differs from original helper")
        require(len(views) == 5 and len(geometry) == 1 and len(geometry[0]) == 4,
                "new whole/four-tile wiring differs")
        for tensors in views:
            require(len(tensors) == 1 and tensors[0].shape == (3, 768, 1344)
                    and tensors[0].dtype == torch.float32 and torch.isfinite(tensors[0]).all(),
                    "new crop input is nonfinite or wrong shape/precision")
        rows.append({"image_id": image_id, "geometry": geometry[0],
                     "whole_pixels_sha256": hashlib.sha256(views[0][0].numpy().tobytes()).hexdigest()})
    import unittest
    import test_slice_core
    import test_seed_readout
    tests = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module)
                               for module in (test_slice_core, test_seed_readout)])
    test_result = unittest.TextTestRunner(verbosity=1).run(tests)
    require(test_result.wasSuccessful(), "core/readout CPU interface tests failed")
    require(not torch.cuda.is_initialized(), "CPU inference interface initialized CUDA")
    return {"status": "PASS", "seed": c["seed"], "images": rows, "actual_fixed64": True,
            "original_pixel_path_exact": True, "new_views_GT_free": True,
            "GT_used_only_in_legacy_transform_comparison": True,
            "seed0_cached_formula_fixture_only": reference_ref,
            "fixture_is_not_new_seed_output_evidence": True,
            "own_seed_output_check_pending_GPU": True,
            "core_and_readout_tests": test_result.testsRun,
            "model_constructions": 0, "forward_calls": 0, "CUDA_initialized": False}


def gpu(c, out, phase, engine, binding, *, emit=None, admit=None):
    """Run one fixed phase on the caller's already fully restored e120 engine.

    emit(event_name, **fields) and admit() let the independent supervisor retain
    its authenticated protocol and real resource channel. After immutable output
    and state checks, SAHI phases emit their workload closure. The independent
    worker subsequently authenticates RESULT and closes the process phase.
    """
    import torch
    from sparse_rtdetr.localization import worker as base
    from sparse_rtdetr.localization.runio import event, publish, require
    from sparse_rtdetr.localization.consumer_geometry import corners, positive_box
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.r0.evaluation import rank_candidates
    _scope(c)
    require(phase in WORKLOADS, "unknown fixed inference phase")
    require(engine.completed_updates == engine.ema.updates == 36480,
            "inference engine is not the locked e120 terminal state")
    out = Path(out)
    emit = emit or (lambda kind, **fields: event(out, kind, **fields))
    admit = admit or (lambda: base.live_admission(out, c))
    data = _dataset(c)
    count, warmups, factor = WORKLOADS[phase]
    ids = _full_ids(c) if count == 548 else c["fixed64"]
    require(len(ids) == len(set(ids)) == count, "phase image workload differs")
    rows = [data.by_id[i] for i in ids]
    index = {row["stable_image_id"]: i for i, row in enumerate(data.records)}
    original = None if phase == "infer-A" else _load_own_raw(c, data)[0]
    sliced = factor == 5
    merged = phase not in ("infer-A", "accept-A")
    model = engine.ema.module
    model.eval()
    before = checkpoint.capture(engine, binding)
    original_backward = torch.autograd.backward
    original_step, original_ema = engine.optimizer.step, engine.ema.update
    counters = {"backward": 0, "optimizer": 0, "EMA": 0}

    def forbidden(name):
        def fail(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError("forbidden detector mutation: " + name)
        return fail

    torch.autograd.backward = forbidden("backward")
    engine.optimizer.step, engine.ema.update = forbidden("optimizer"), forbidden("EMA")
    calls, whole_exact = 0, 0
    timings, raw_rows, view_refs = [], [], []
    preds = {600: [], 300: []}

    def batch(batch_rows):
        nonlocal calls
        torch.cuda.synchronize()
        start = time.perf_counter()
        views, geometry = make_view_pixels(data, batch_rows, sliced)
        preprocessed = time.perf_counter()
        candidates = [[] for _ in batch_rows]
        raw_whole, records = None, []
        forward_seconds, decode_seconds = 0., 0.
        for view_id, image_tensors in enumerate(views):
            x = torch.stack(image_tensors).to("cuda:0")
            torch.cuda.synchronize()
            f0 = time.perf_counter()
            with torch.autocast("cuda", enabled=False), forward_guards(model):
                output = model(x)
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - f0
            calls += 1
            d0 = time.perf_counter()
            require(set(output) == {"pred_logits", "pred_boxes"}, "model output namespace differs")
            positive_box("sahi-repeat/output", output["pred_boxes"])
            require(torch.isfinite(output["pred_logits"]).all(), "nonfinite class logits")
            logits = output["pred_logits"].detach().cpu()
            boxes = output["pred_boxes"].detach().cpu()
            require(logits.dtype == boxes.dtype == torch.float32
                    and logits.shape == (4, 300, 10) and boxes.shape == (4, 300, 4),
                    "inference shape/precision differs")
            if view_id == 0:
                raw_whole = {"pred_logits": logits, "pred_boxes": boxes}
            for j, row in enumerate(batch_rows):
                tile = tile_windows(row["height"], row["width"])[view_id - 1] if view_id else None
                width, height = (tile.width, tile.height) if tile else (row["width"], row["height"])
                xyxy = torch.cat((boxes[j, :, :2] - boxes[j, :, 2:] / 2,
                                  boxes[j, :, :2] + boxes[j, :, 2:] / 2), -1)
                xyxy *= torch.tensor([width, height] * 2, dtype=torch.float32)
                corners("sahi-repeat/view-pixel-corners", xyxy)
                mapped = map_crop_xyxy(xyxy.tolist(), tile) if tile else xyxy.tolist()
                selected = select_view_candidates(mapped, logits[j].sigmoid().numpy(), view_id=view_id)
                candidates[j].extend(selected)
            records.append({"view_id": view_id, "pred_logits": logits, "pred_boxes": boxes})
            decode_seconds += time.perf_counter() - d0
            del x, output
        merge_start = time.perf_counter()
        final = []
        for j, row in enumerate(batch_rows):
            if merged:
                merged_candidates = merge_candidates(candidates[j])
                six, three = merged_candidates["top600"], merged_candidates["top300"]
            else:
                six, three = candidates[j][:600], candidates[j][:300]
            final.append({600: [prediction(row["stable_image_id"], v, whole=not merged) for v in six],
                          300: [prediction(row["stable_image_id"], v, whole=not merged) for v in three]})
        torch.cuda.synchronize()
        end = time.perf_counter()
        timing = {"e2e_seconds": end - start, "preprocess_seconds": preprocessed - start,
                  "forward_seconds": forward_seconds, "decode_rank_seconds": decode_seconds,
                  "merge_seconds": end - merge_start}
        timing["stack_H2D_sync_and_other_seconds"] = timing["e2e_seconds"] - sum(
            timing[k] for k in ("preprocess_seconds", "forward_seconds", "decode_rank_seconds", "merge_seconds"))
        return final, records, raw_whole, geometry, timing

    started = time.monotonic()
    try:
        with torch.inference_mode():
            for k in range(warmups):
                admit()
                batch(rows[k * 4:(k + 1) * 4])
            if warmups:
                emit("sahi_warmup_complete", warmup_original_batches=warmups,
                     forward_calls=calls, seed=c["seed"])
            torch.cuda.reset_peak_memory_stats()
            for k in range(count // 4):
                batch_rows = rows[k * 4:(k + 1) * 4]
                admit()
                final, records, actual, geometry, timing = batch(batch_rows)
                timings.append(timing)
                if original is not None:
                    indices = [index[row["stable_image_id"]] for row in batch_rows]
                    for key in ("pred_boxes", "pred_logits"):
                        require(torch.equal(actual[key], original[key][indices]),
                                "seed-specific whole output changed: " + key)
                    whole_exact += 4
                for j, row in enumerate(batch_rows):
                    legacy = rank_candidates(actual["pred_logits"][j], actual["pred_boxes"][j], row, 600)
                    if not merged:
                        require(final[j][600] == legacy and final[j][300] == legacy[:300],
                                "whole candidate decode/rank differs from original helper")
                    for budget in BUDGETS:
                        preds[budget].extend(final[j][budget])
                if phase == "infer-A":
                    raw_rows.append(actual)
                    for j, row in enumerate(batch_rows):
                        emit("evaluation_image_saved", evaluation_image_number=k * 4 + j + 1,
                             image_id=row["stable_image_id"], checkpoint_sha256=c["checkpoint"]["sha256"],
                             seed=c["seed"])
                else:
                    emit("sahi_batch_complete", batch_number=k + 1, original_images=(k + 1) * 4,
                         forward_calls=calls, image_ids=[r["stable_image_id"] for r in batch_rows],
                         whole_output_exact=True, e2e_batch_seconds=timing["e2e_seconds"], seed=c["seed"])
                if phase == "full-C":
                    view_refs.append(save_tensor(out / "raw-views" / f"batch-{k:03d}.pth",
                        {"schema": "sahi-repeat-raw-views-v1", **_identity(c),
                         "image_ids": [r["stable_image_id"] for r in batch_rows],
                         "geometry": geometry, "views": records}))
    except BaseException:
        publish(out / "GPU_FAILURE_STATE.json", {**_identity(c), "phase": phase,
                "forward_calls": calls, "images_verified": whole_exact, "counters": counters,
                "traceback": traceback.format_exc()})
        raise
    finally:
        torch.autograd.backward = original_backward
        engine.optimizer.step, engine.ema.update = original_step, original_ema
    require(calls == factor * (count // 4 + warmups)
            and counters == {"backward": 0, "optimizer": 0, "EMA": 0},
            "fixed forward/mutation budget differs")
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)),
            "inference changed complete model/BN/optimizer/EMA/RNG state")
    require(all(p.grad is None for m in (engine.model, engine.ema.module) for p in m.parameters()),
            "inference created detector gradient")
    refs = {}
    from sparse_rtdetr.localization.runio import reference
    for budget in BUDGETS:
        path = out / f"top{budget}-predictions.json"
        publish(path, preds[budget])
        refs[str(budget)] = reference(path)
    result = {"schema": "r18-sahi-repeat-inference-v1", **_identity(c), "status": "PASS",
              "phase": phase, "images": count, "model_batch_forwards": calls,
              "warmup_original_batches": warmups, "whole_output_exact": whole_exact,
              "whole_reference": "self independent original A cache" if original is not None else
                                  "new seed-specific A; original rank helper exact on all548",
              "state_RNG_exact": True, "detector_updates": 0, "backward_calls": 0,
              "FP32": True, "batch4": True, "predictions": refs, "raw_view_refs": view_refs,
              "timing": {"e2e_total_seconds": sum(t["e2e_seconds"] for t in timings),
                         "e2e_per_original_image_seconds": sum(t["e2e_seconds"] for t in timings) / count,
                         "batches": timings},
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "elapsed_seconds": time.monotonic() - started, "holdout_accessed": False}
    if phase == "infer-A":
        result["raw"] = save_tensor(out / f"raw-queries-seed{c['seed']}-e120.pth",
            {"schema": "r18-sahi-repeat-own-A-raw-v1", **_identity(c), "epoch": 120, "EMA": True,
             "image_ids": ids, "query_ids": torch.arange(300),
             "pred_logits": torch.cat([r["pred_logits"] for r in raw_rows]),
             "pred_boxes": torch.cat([r["pred_boxes"] for r in raw_rows])})
    publish(out / "RESULT.json", result)
    if phase != "infer-A":
        emit("sahi_complete", seed=c["seed"], result=reference(out / "RESULT.json"),
             detector_updates=0, backward_calls=0, state_unchanged=True,
             forward_calls=calls)
    return result


def _candidate(p, rank, *, whole):
    if whole:
        if p.get("view_id", 0) != 0:
            raise ValueError("whole candidate has nonzero view")
        view = 0
    else:
        if "view_id" not in p:
            raise ValueError("merged candidate has no view identity")
        view = p["view_id"]
    return Candidate(tuple(p["bbox_xyxy"]), p["score"], view, p["query_id"], p["category_id"],
                     p.get("view_candidate_rank", rank), EXPECTED_SHAPE)


def _group(predictions, ids, budget, *, whole, exact):
    from sparse_rtdetr.localization.runio import require
    grouped = {i: [] for i in ids}
    for p in predictions:
        require(isinstance(p, dict) and p.get("image_id") in grouped, "prediction image is outside frozen548")
        grouped[p["image_id"]].append(p)
    for records in grouped.values():
        require(len(records) == budget if exact else len(records) <= budget, "prediction budget differs")
        candidates = [_candidate(p, rank, whole=whole) for rank, p in enumerate(records, 1)]
        require(len({p.identity for p in candidates}) == len(candidates), "duplicate candidate identity")
        require(candidates == sorted(candidates, key=lambda p: p.sort_key), "candidate total order differs")
    return grouped


def _prefix(six, three, ids, *, whole):
    from sparse_rtdetr.localization.runio import require
    def key(p):
        candidate = _candidate(p, 1, whole=whole)
        return candidate.identity, candidate.score, candidate.box_xyxy
    for image in ids:
        require([key(p) for p in six[image][:300]] == [key(p) for p in three[image]],
                "top300 is not the fixed ordered prefix")


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric metric required")
    return float(value)


def bridge_values(record):
    from sparse_rtdetr.localization.runio import require
    bridge = record["COCO_bridge"]
    require(bridge["units"] == "0..1 AP, -1 means no GT in an area/class aggregation",
            "metric units differ; pp conversion must happen once")
    values = {name: _finite(bridge["metrics"][name]) for name in METRICS}
    require(all(0 <= v <= 1 for v in values.values()), "metric is unavailable or outside0..1")
    return values


def compare(values):
    deltas = {str(b): {first + "-" + second: {name:
        (values[first][str(b)][name] - values[second][str(b)][name]) * 100.
        for name in METRICS} for first, second in (("B", "A"), ("C", "A"), ("C", "B"))}
        for b in BUDGETS}
    gates = {}
    for budget, limits in ((600, MAIN_LIMITS_PP), (300, SECONDARY_LIMITS_PP)):
        for name, limit in limits.items():
            delta = deltas[str(budget)]["C-A"][name]
            gates[f"top{budget}_{name}"] = {"difference_pp": delta,
                "minimum_pp": limit, "passed": delta >= limit}
    return deltas, gates


def cost(c):
    from sparse_rtdetr.localization.runio import read, reference, require
    seconds, refs = {}, {}
    for arm in ("A", "B", "C"):
        path = Path(c["job_root"]) / ("accept-" + arm) / "RESULT.json"
        result = _closed_result(c, "accept-" + arm)
        require(result["status"] == "PASS" and result["phase"] == "accept-" + arm,
                "cost phase did not pass engineering checks")
        for key, value in _identity(c).items():
            require(result[key] == value, "cost phase identity differs: " + key)
        require(result["images"] == 64 and result["FP32"] is True and result["batch4"] is True,
                "cost phases are not matched64/FP32/batch4")
        seconds[arm] = _finite(result["timing"]["e2e_total_seconds"])
        require(seconds[arm] > 0, "positive measured end-to-end time required")
        refs[arm] = reference(path)
    ratio = seconds["C"] / seconds["A"]
    return {"scope": "matched fixed64, same seed e120, FP32/batch4, separate A/B/C processes",
            "seconds": seconds, "C_over_A": ratio, "B_over_A": seconds["B"] / seconds["A"],
            "C_over_B": seconds["C"] / seconds["B"], "maximum_C_over_A": 6.,
            "passed": ratio <= 6., "acceptance_results": refs,
            "whole_branch_really_executed": True, "no_cost_based_retry": True}


def evaluate(c, out, *, emit=None):
    """Score each own A/B/C budget once. Scientific FAIL is a completed result."""
    import torch
    from sparse_rtdetr.localization.runio import event, read, publish, reference, require
    from sparse_rtdetr.r0.evaluation import score_budget, POLICY
    _scope(c)
    require(not torch.cuda.is_initialized(), "CPU evaluation initialized CUDA")
    if "evaluation_protocol" in c:
        require(read(c["evaluation_protocol"]) == POLICY, "frozen evaluation policy differs")
        from sparse_rtdetr.localization.runio import sha
        require(sha(c["evaluation_protocol"]) == c["evaluation_protocol_sha256"],
                "evaluation protocol binding differs")
    out = Path(out)
    emit = emit or (lambda kind, **fields: event(out, kind, **fields))
    data = _dataset(c)
    ids, rows = _full_ids(c), data.records
    own_raw, a_result = _load_own_raw(c, data)
    full_result = _closed_result(c, "full-C")
    require(full_result["status"] == "PASS" and full_result["whole_output_exact"] == 548,
            "full-C engineering evidence incomplete")
    for key, value in _identity(c).items():
        require(full_result[key] == value, "full-C identity differs: " + key)
    predictions, refs = {"A": {}, "B": {}, "C": {}}, {"A": {}, "B": {}, "C": {}}
    grouped = {"A": {}, "C": {}}
    for arm, result in (("A", a_result), ("C", full_result)):
        for budget in BUDGETS:
            key = str(budget)
            refs[arm][key] = result["predictions"][key]
            records = _read_ref(refs[arm][key])
            predictions[arm][key] = records
            grouped[arm][key] = _group(records, ids, budget, whole=arm == "A", exact=arm == "A")
        _prefix(grouped[arm]["600"], grouped[arm]["300"], ids, whole=arm == "A")
    # A's independent serialized candidates must agree with its own raw logits/boxes.
    from sparse_rtdetr.r0.evaluation import rank_candidates
    for i, row in enumerate(rows):
        require(rank_candidates(own_raw["pred_logits"][i], own_raw["pred_boxes"][i], row, 600)
                == grouped["A"]["600"][row["stable_image_id"]], "own A raw/predictions differ")
    counts = []
    for budget in BUDGETS:
        predictions["B"][str(budget)] = []
    for image in ids:
        candidates = [_candidate(p, rank, whole=True)
                      for rank, p in enumerate(grouped["A"]["600"][image], 1)]
        merged = merge_candidates(candidates)
        for budget in BUDGETS:
            predictions["B"][str(budget)].extend(prediction(image, p) for p in merged[f"top{budget}"])
        counts.append({"image_id": image, "600": len(merged["top600"]), "300": len(merged["top300"])})
    for budget in BUDGETS:
        key = str(budget)
        path = out / f"B-top{budget}-predictions.json"
        publish(path, predictions["B"][key])
        refs["B"][key] = reference(path)
    publish(out / "B-POSTPROCESS.json", {**_identity(c), "status": "PASS",
        "input": refs["A"]["600"], "GT_read_for_postprocessing": False,
        "NMS_IoU": .5, "no_padding_or_refill": True, "predictions": refs["B"], "counts": counts})
    # GT is opened only after all three arms' merged output is locked.
    parsed = [data.raw(row) for row in rows]
    values, metrics, metric_refs = {}, {}, {}
    started, ordinal = time.monotonic(), 0
    repo_root = Path(c["source_root"])
    for arm in ("A", "B", "C"):
        values[arm], metrics[arm], metric_refs[arm] = {}, {}, {}
        for budget in BUDGETS:
            key = str(budget)
            record = score_budget(rows, parsed, predictions[arm][key], repo_root)
            record["sahi_binding"] = {**_identity(c), "arm": arm, "budget": budget,
                "prediction": refs[arm][key], "own_seed_A": True,
                "original_raw_ignore_reused": True, "bootstrap_calls": 0}
            path = out / f"{arm}-top{budget}-metrics.json"
            publish(path, record)
            metrics[arm][key] = record
            values[arm][key] = bridge_values(record)
            metric_refs[arm][key] = reference(path)
            ordinal += 1
            emit("evaluation_budget_complete", evaluation_number=ordinal, arm=arm, budget=budget,
                 name=f"{arm}-top{budget}", images=548, checkpoint_sha256=c["checkpoint"]["sha256"],
                 seed=c["seed"])
    deltas, gates = compare(values)
    costs = cost(c)
    comparison = {"schema": "r18-sahi-repeat-ABC-comparison-v1", **_identity(c), "images": 548,
        "baseline": f"common-protocol-adapted RT-DETRv2-R18 seed{c['seed']} e120 EMA",
        "COCO_bridge_original_score_units": "0..1", "difference_units": "percentage points",
        "scores": values, "differences_pp": deltas, "gates_C_minus_A": gates,
        "benefit_and_protection_passed": all(g["passed"] for g in gates.values()),
        "cost": costs, "screen_passed": all(g["passed"] for g in gates.values()) and costs["passed"],
        "official_separate": {a: {str(b): metrics[a][str(b)]["official"] for b in BUDGETS}
                              for a in ("A", "B", "C")},
        "new_score_budget_calls": 6, "bootstrap_calls": 0,
        "scientific_failure_does_not_abort_other_frozen_seed": True,
        "query_identity_crossseed_correspondence": False, "holdout_accessed": False}
    publish(out / "COMPARISON.json", comparison)
    table = io.StringIO(newline="")
    writer = csv.writer(table)
    writer.writerow(["seed", "budget", "metric", "A_score", "B_score", "C_score", "B_minus_A_pp", "C_minus_A_pp", "C_minus_B_pp"])
    for budget in BUDGETS:
        for metric in METRICS:
            key = str(budget)
            writer.writerow([c["seed"], budget, metric, *(values[a][key][metric] for a in ("A", "B", "C")),
                             *(deltas[key][pair][metric] for pair in ("B-A", "C-A", "C-B"))])
    path = out / "TABLE.csv"
    with path.open("x", encoding="utf8", newline="") as stream:
        stream.write(table.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    require(not torch.cuda.is_initialized(), "CPU evaluation initialized CUDA")
    result = {"schema": "r18-sahi-repeat-evaluation-v1", **_identity(c), "status": "PASS",
        "images": 548, "budgets": [600, 300], "predictions": refs, "metrics": metric_refs,
        "comparison": reference(out / "COMPARISON.json"), "table": reference(path),
        "new_score_budget_calls": 6, "forward_calls": 0, "detector_updates": 0,
        "bootstrap_calls": 0, "CUDA_initialized": False, "holdout_accessed": False,
        "benefit_and_protection_passed": comparison["benefit_and_protection_passed"],
        "cost_passed": costs["passed"], "screen_passed": comparison["screen_passed"],
        "elapsed_seconds": time.monotonic() - started}
    publish(out / "RESULT.json", result)
    return result
