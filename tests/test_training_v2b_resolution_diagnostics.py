"""Synthetic CPU fixtures: exact AP resampling, conservative ledgers and routing."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from sparse_rtdetr.baseline import training_v2b_resolution_diagnostics as diagnostic


def annotation(i, image, category, box, *, crowd=0):
    return {"id": i, "image_id": image, "category_id": category,
            "bbox": list(map(float, box)), "area": float(box[2] * box[3]),
            "iscrowd": crowd, "stable_annotation_id": f"gt-{i}"}


def detection(image, category, box, score):
    return {"image_id": image, "category_id": category,
            "bbox": list(map(float, box)), "score": float(score)}


def image(i, width=128, height=128):
    return {"id": i, "width": width, "height": height,
            "split": "val", "stable_image_id": f"image-{i}"}


def coco_fixture():
    gt = {"images": [image(11), image(12), image(13)],
          "categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}],
          "annotations": [
              annotation(1, 11, 1, [1, 1, 10, 10]),
              annotation(2, 11, 2, [32, 32, 10, 10]),
              annotation(3, 11, 2, [40, 50, 40, 40]),
              annotation(4, 12, 1, [1, 1, 10, 10]),
              annotation(5, 12, 1, [45, 45, 20, 20], crowd=1),
              annotation(6, 12, 2, [0, 0, 110, 110]),
              annotation(7, 13, 1, [1, 1, 10, 10]),
          ]}
    pred = [
        detection(11, 1, [75, 1, 10, 10], .8),
        detection(11, 1, [1, 1, 10, 10], .8),
        detection(11, 1, [1, 1, 10, 10], .8),
        detection(11, 2, [32, 32, 10, 10], .8),
        detection(11, 2, [40, 50, 40, 40], .7),
        detection(12, 1, [1, 1, 10, 10], .8),
        detection(12, 1, [48, 48, 3, 3], .8),
        detection(12, 1, [51, 51, 3, 3], .8),
        detection(12, 2, [0, 0, 109, 109], .7),
        # Image 13 has formal GT but no predictions.
    ]
    manifest = [{"coco_image_id": i, "split": "val", "sequence_key": f"seq-{i}",
                 "image_sha256": ("a" if i in (11, 13) else "b") * 64}
                for i in (11, 12, 13)]
    return gt, pred, manifest


def faster_reference(gt, predictions, maximum=100):
    from faster_coco_eval import COCO, COCOeval_faster
    coco_gt = COCO(copy.deepcopy(gt), print_function=lambda *a: None)
    rows = copy.deepcopy(predictions)
    for i, row in enumerate(rows, 1):
        row.update(id=i, area=row["bbox"][2] * row["bbox"][3], iscrowd=0)
    coco_dt = COCO({"images": copy.deepcopy(gt["images"]),
                    "categories": copy.deepcopy(gt["categories"]),
                    "annotations": rows}, print_function=lambda *a: None)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox", print_function=lambda *a: None)
    evaluator.params.maxDets = [1, 10, maximum]
    evaluator.evaluate()
    evaluator.accumulate()
    p, r = evaluator.eval["precision"], evaluator.eval["recall"]

    def mean(x):
        x = x[x > -1.]
        return float(x.mean() * 100.) if x.size else None

    result = {"AP": mean(p[:, :, :, 0, -1]), "AP50": mean(p[0, :, :, 0, -1]),
              "AP75": mean(p[5, :, :, 0, -1]), f"AR{maximum}": mean(r[:, :, 0, -1])}
    for area, index in (("small", 1), ("medium", 2), ("large", 3)):
        result[f"AP_{area}"] = mean(p[:, :, :, index, -1])
        result[f"AR_{area}"] = mean(r[:, :, index, -1])
    return result


@pytest.mark.parametrize("ids", [
    (11, 12, 13), (11, 11, 12), (12, 13, 11, 11), (12, 12), (13,), (11, 13, 13),
])
@pytest.mark.parametrize("maximum", [100, 300])
def test_cached_coco_ap_equals_literal_duplicate_dataset_and_faster_backend(ids, maximum):
    gt, predictions, _ = coco_fixture()
    original = copy.deepcopy((gt, predictions))
    cache = diagnostic.prepare_coco_cache(gt, predictions, max_dets=maximum)
    copied_gt, copied_pred = diagnostic.literal_resample_coco(gt, predictions, ids)
    reference = faster_reference(copied_gt, copied_pred, maximum)
    observed = cache.score(ids)
    for key, value in reference.items():
        if value is None:
            assert observed[key] is None
        else:
            assert observed[key] == pytest.approx(value, abs=1e-12, rel=0)
    assert (gt, predictions) == original
    assert len({x["id"] for x in copied_gt["images"]}) == len(ids)
    assert len({x["id"] for x in copied_gt["annotations"]}) == len(copied_gt["annotations"])



@pytest.mark.parametrize("prime_reference", [False, True])
def test_reference_cache_survives_actual_development_vendor_import_order(prime_reference):
    # A fresh interpreter is required: both lazy-first-use orders matter, and
    # the real vendor alias installation must not leak from this test's process.
    root = Path(__file__).resolve().parents[1]
    script = r"""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
fixtures = runpy.run_path(str(root / "tests/test_training_v2b_resolution_diagnostics.py"))
diagnostic = fixtures["diagnostic"]
gt, predictions, _ = fixtures["coco_fixture"]()
if sys.argv[2] == "True":
    # Include a previously imported genuine native extension, as well as an
    # already-built cache, before the vendor replaces the public package.
    importlib.import_module("pycocotools.coco")
    native = sys.modules["pycocotools._mask"]
    metadata_fields = ("__name__", "__package__", "__loader__", "__spec__")
    native_metadata = tuple(getattr(native, name) for name in metadata_fields)
    earlier = diagnostic.prepare_coco_cache(gt, predictions).score((11, 11, 12))
    assert tuple(getattr(native, name) for name in metadata_fields) == native_metadata

from sparse_rtdetr.baseline import training_v2b_development as development
development._vendor_tools(root)
import faster_coco_eval
aliases = {name: module for name, module in tuple(sys.modules.items())
           if name == "pycocotools" or name.startswith("pycocotools.")}
metadata_fields = ("__name__", "__package__", "__loader__", "__spec__")
alias_metadata = {name: tuple(getattr(module, key, None) for key in metadata_fields)
                  for name, module in aliases.items()}
assert aliases["pycocotools"] is faster_coco_eval
assert aliases["pycocotools.cocoeval"] is faster_coco_eval.cocoeval
distribution = importlib.metadata.distribution("pycocotools")
for maximum in (100, 300):
    cache = diagnostic.prepare_coco_cache(gt, predictions, max_dets=maximum)
    for ids in ((11, 12, 13), (11, 11, 12), (12, 12), (13,)):
        copied_gt, copied_predictions = diagnostic.literal_resample_coco(gt, predictions, ids)
        expected = fixtures["faster_reference"](copied_gt, copied_predictions, maximum)
        actual = cache.score(ids)
        for key, value in expected.items():
            if value is None:
                assert actual[key] is None, (key, actual, expected)
            else:
                assert abs(actual[key] - value) <= 1e-12, (key, actual, expected)
    assert cache.matched_ids(11, iou=.5, area="small") == {1, 2}
    identity = cache.describe()["matching_backend_identity"]
    assert identity["distribution"] == "pycocotools"
    assert identity["version"] == distribution.version
    for name, reference in identity["files"].items():
        actual_path = Path(distribution.locate_file("pycocotools/" + name)).resolve()
        assert reference["path"] == str(actual_path)
        assert reference["sha256"] == hashlib.sha256(actual_path.read_bytes()).hexdigest()
    current_aliases = {name: module for name, module in tuple(sys.modules.items())
                       if name == "pycocotools" or name.startswith("pycocotools.")}
    assert current_aliases == aliases
    assert all(tuple(getattr(module, key, None) for key in metadata_fields) == alias_metadata[name]
               for name, module in current_aliases.items())
if sys.argv[2] == "True":
    assert diagnostic.prepare_coco_cache(gt, predictions).score((11, 11, 12)) == earlier
import torch
assert not torch.cuda.is_initialized()
print(json.dumps({"alias_isolation": "PASS", "reference_primed": sys.argv[2],
                  "distribution_version": distribution.version, "cuda_initialized": False}))
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(root), str(prime_reference)],
        cwd=root, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"alias_isolation": "PASS"' in result.stdout


def test_dataset_ap_is_not_mean_image_ap():
    gt, predictions, _ = coco_fixture()
    cache = diagnostic.prepare_coco_cache(gt, predictions)
    full = cache.score(cache.image_ids)["AP"]
    naive = np.mean([cache.score((i,))["AP"] for i in cache.image_ids])
    assert abs(full - naive) > 1.
    assert cache.score((12, 11, 11)) == cache.score((11, 11, 12))


def test_duplicate_content_connects_different_sequences_into_one_cluster():
    gt, _, manifest = coco_fixture()
    assert diagnostic.build_development_clusters(manifest, (11, 12, 13)) == ((11, 13), (12,))
    manifest[1]["sequence_key"] = manifest[0]["sequence_key"]
    assert diagnostic.build_development_clusters(manifest, (11, 12, 13)) == ((11, 12, 13),)


def fixture_policy(images=3, clusters=2, **changes):
    return replace(diagnostic.FROZEN_POLICY, fixture_only=True, expected_images=images,
                   expected_clusters=clusters, cluster_replicates=7, image_replicates=5,
                   **changes)


def test_four_cells_share_cluster_and_image_draws_and_ci_is_percentile_of_paired_deltas():
    gt, predictions, manifest = coco_fixture()
    cache = diagnostic.prepare_coco_cache(gt, predictions)
    calls = []

    def primary(ids):
        calls.append(ids)
        return {"AP": float(len(set(ids)))}

    report = diagnostic.paired_bootstrap_2x2(
        {cell: cache for cell in diagnostic.CELLS}, manifest,
        primary_scorers={cell: primary for cell in diagnostic.CELLS}, policy=fixture_policy())
    for i in range(0, len(calls), 4):
        assert calls[i:i + 4] == [calls[i]] * 4
    for section in report["resampling"].values():
        values = np.asarray(section["distribution"])
        assert not np.any(values)
        for contrast in section["contrasts"].values():
            for metric in contrast.values():
                assert metric["point_pp"] == 0.
                assert metric["ci95_pp"] == [0., 0.]
    assert report["clusters"] == [[11, 13], [12]]
    assert report["seed_gate"]["eligible_for_fixed_paired_seed1_and_seed2"] is False
    repeated = diagnostic.paired_bootstrap_2x2(
        {cell: cache for cell in diagnostic.CELLS}, manifest, policy=fixture_policy())
    assert repeated["resampling"]["cluster"]["draws_sha256"] == report["resampling"]["cluster"]["draws_sha256"]


def test_nonzero_contrasts_and_quantiles_are_recomputed_on_shared_images():
    gt, predictions, manifest = coco_fixture()
    before = diagnostic.prepare_coco_cache(gt, predictions)
    after = diagnostic.prepare_coco_cache(
        gt, predictions + [detection(13, 1, [1, 1, 10, 10], .9)])
    caches = dict(zip(diagnostic.CELLS, (before, before, before, after)))
    report = diagnostic.paired_bootstrap_2x2(caches, manifest, policy=fixture_policy())
    section = report["resampling"]["cluster"]
    distribution = np.asarray(section["distribution"])
    quantiles = np.quantile(distribution, [.025, .975], axis=0, method="linear")
    for c, name in enumerate(report["contrasts"]):
        for m, metric in enumerate(report["metrics"]):
            assert section["contrasts"][name][metric]["ci95_pp"] == quantiles[:, c, m].tolist()
    delta = after.score(after.image_ids)["AP_small"] - before.score(before.image_ids)["AP_small"]
    assert section["contrasts"]["diagonal_total"]["coco_AP_small"]["point_pp"] == delta
    assert section["contrasts"]["interaction"]["coco_AP_small"]["point_pp"] == delta


def test_production_policy_drift_and_missing_cells_or_different_gt_fail():
    gt, predictions, manifest = coco_fixture()
    cache = diagnostic.prepare_coco_cache(gt, predictions)
    with pytest.raises(diagnostic.DiagnosticError, match="policy drift"):
        diagnostic.paired_bootstrap_2x2(
            {cell: cache for cell in diagnostic.CELLS}, manifest,
            policy=replace(diagnostic.FROZEN_POLICY, cluster_replicates=2))
    with pytest.raises(diagnostic.DiagnosticError, match="four"):
        diagnostic.paired_bootstrap_2x2({diagnostic.CELLS[0]: cache}, manifest,
                                       policy=fixture_policy())
    changed = copy.deepcopy(gt)
    changed["annotations"][0]["bbox"][0] += 1
    other = diagnostic.prepare_coco_cache(changed, predictions)
    with pytest.raises(diagnostic.DiagnosticError, match="GT"):
        diagnostic.paired_bootstrap_2x2(
            dict(zip(diagnostic.CELLS, (cache, cache, cache, other))), manifest,
            policy=fixture_policy())


def error_fixture():
    gt = {"images": [image(i) for i in range(1, 11)],
          "categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}],
          "annotations": [annotation(i, i, 1, [10, 10, 10, 10]) for i in range(1, 10)]}
    gt["annotations"].append(annotation(11, 3, 1, [10, 10, 10, 10]))
    p = [
        detection(1, 1, [10, 10, 10, 10], .9),
        detection(2, 1, [70, 70, 10, 10], .9),
        detection(2, 1, [90, 90, 10, 10], .8),
        detection(2, 1, [10, 10, 10, 10], .7),
        detection(3, 1, [10, 10, 10, 10], .9),
        detection(4, 2, [10, 10, 10, 10], .9),
        detection(5, 1, [15, 10, 10, 10], .9),
        detection(6, 2, [15, 10, 10, 10], .9),
        detection(7, 1, [70, 70, 10, 10], .9),
        detection(8, 1, [10, 10, 10, 10], .9),
        detection(8, 1, [10, 10, 10, 10], .8),
        detection(10, 1, [70, 70, 10, 10], .9),
    ]
    lineage = []
    for row in gt["annotations"]:
        x, y, w, h = row["bbox"]
        lineage.append({"annotation_id": row["stable_annotation_id"],
                        "image_id": f"image-{row['image_id']}", "split": "val",
                        "raw_fields": [str(x), str(y), str(w), str(h), "1", "1", "1", "2"],
                        "status": "keep", "bbox_xyxy": [x, y, x + w, y + h],
                        "area": row["area"], "coco_category_id": row["category_id"]})
    return gt, p, lineage


def test_small_gt_and_prediction_ledgers_are_exclusive_and_conserve_counts():
    gt, p, lineage = error_fixture()
    result = diagnostic.decompose_errors(gt, p, lineage, max_dets=2)
    at50 = result["by_iou"]["0.5"]
    assert sum(at50["gt_counts"].values()) == len(gt["annotations"])
    assert sum(at50["prediction_counts"].values()) == len(p)
    assert all(at50["gt_counts"][k] for k in diagnostic.GT_REASONS)
    assert at50["prediction_counts"]["duplicate"] == 1
    assert at50["prediction_counts"]["maxdets_excluded"] == 1
    assert at50["small_unmatched"] == len(gt["annotations"]) - at50["gt_counts"]["true_positive"]
    assert all(x["tags"]["truncation"] == 1 and x["tags"]["occlusion"] == 2
               for x in at50["gt_rows"])
    assert result["lineage_binding"]["spatial_ignore_coverage_verified"] is False


def test_lineage_joins_are_exact_and_missing_labels_stay_unavailable():
    gt, predictions, lineage = error_fixture()
    bad = copy.deepcopy(lineage)
    bad[0]["bbox_xyxy"][0] += 1
    with pytest.raises(diagnostic.DiagnosticError, match="join mismatch"):
        diagnostic.decompose_errors(gt, predictions, bad)
    result = diagnostic.decompose_errors(gt, predictions)
    assert result["lineage_binding"]["available"] is False
    assert all(x["tags"]["occlusion"] is None for x in result["by_iou"]["0.5"]["gt_rows"])


def test_non_development_and_nonfinite_input_fail():
    gt, predictions, _ = coco_fixture()
    gt["images"][0]["split"] = "confirmatory"
    with pytest.raises(diagnostic.DiagnosticError, match="development"):
        diagnostic.prepare_coco_cache(gt, predictions)
    gt["images"][0]["split"] = "val"
    predictions[0]["score"] = float("nan")
    with pytest.raises(diagnostic.DiagnosticError, match="finite"):
        diagnostic.prepare_coco_cache(gt, predictions)


def routing_fixture():
    gt = {"images": [image(1, 640, 640)],
          "categories": [{"id": 1, "name": "one"}],
          "annotations": [annotation(1, 1, 1, [10, 10, 10, 10]),
                          annotation(2, 1, 1, [200, 200, 10, 10]),
                          annotation(3, 1, 1, [400, 400, 10, 10])]}
    coarse = []
    fine = [detection(1, 1, x["bbox"], .9) for x in gt["annotations"]]
    raw = {1: {"pred_logits": np.full((1, 10), -1.3862943611198906),
               "pred_boxes": np.asarray([[15 / 640, 15 / 640, 10 / 640, 10 / 640]])}}
    return gt, coarse, fine, raw


def test_equal_routing_budget_oracle_and_coarse_selection_does_not_see_fine_or_gt():
    gt, coarse, fine, raw = routing_fixture()
    policy = fixture_policy(images=1, clusters=1, routing_tiles=1)
    first = diagnostic.routing_oracle_diagnostic(
        gt, coarse, fine, raw_coarse_by_image=raw, policy=policy)
    none = diagnostic.routing_oracle_diagnostic(
        gt, coarse, [], raw_coarse_by_image=raw, policy=policy)
    key = "coarse_query_uncertainty_small"
    assert first["per_image"][0]["selected_tiles"][key] == [0]
    assert none["per_image"][0]["selected_tiles"][key] == [0]
    assert all(len(v) == 1 for v in first["per_image"][0]["selected_tiles"].values())
    assert first["totals"]["gt_recoverable_oracle_non_deployable"] >= first["totals"][key]
    assert first["cost"]["pixel_proxy_per_image"] == 640**2 + (896 // 4 + 32)**2
    assert first["cost"]["actual_sparse_flops"] is None
    assert first["cost"]["actual_sparse_latency"] is None
    assert first["cost"]["diagnostic_gpu_forward_count"] == 0
    assert none["capture_fractions"][key] is None


def test_raw_query_coverage_and_shape_are_required_for_routing():
    gt, coarse, fine, raw = routing_fixture()
    policy = fixture_policy(images=1, clusters=1)
    with pytest.raises(diagnostic.DiagnosticError, match="coverage"):
        diagnostic.routing_oracle_diagnostic(gt, coarse, fine, raw_coarse_by_image={}, policy=policy)
    raw[1]["pred_boxes"] = np.zeros((1, 4))
    with pytest.raises(diagnostic.DiagnosticError, match="geometry"):
        diagnostic.routing_oracle_diagnostic(gt, coarse, fine, raw_coarse_by_image=raw, policy=policy)


@pytest.mark.parametrize("seed", range(5))
def test_mixed_random_geometry_classes_crowds_and_score_ties_against_faster(seed):
    rng = np.random.default_rng(seed)
    gt = {"images": [image(i, 160, 160) for i in (1, 2, 3)],
          "categories": [{"id": i, "name": str(i)} for i in (1, 2, 3)],
          "annotations": []}
    pred = []
    aid = 0
    for iid in (1, 2, 3):
        for _ in range(8):
            aid += 1
            box = [*rng.integers(0, 90, size=2).tolist(),
                   *rng.integers(5, 60, size=2).tolist()]
            cat = int(rng.integers(1, 4))
            gt["annotations"].append(annotation(aid, iid, cat, box,
                                                 crowd=int(rng.random() < .15)))
            pred.append(detection(iid, cat, box, float(rng.choice([.5, .75]))))
            box2 = [box[0] + int(rng.integers(-5, 10)), box[1], box[2], box[3]]
            pred.append(detection(iid, int(rng.integers(1, 4)), box2,
                                  float(rng.choice([.25, .5, .75]))))
    cache = diagnostic.prepare_coco_cache(gt, pred)
    ids = (3, 1, 1, 2, 2)
    copied_gt, copied_pred = diagnostic.literal_resample_coco(gt, pred, ids)
    expected = faster_reference(copied_gt, copied_pred)
    actual = cache.score(ids)
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None
        else:
            assert actual[key] == pytest.approx(value, abs=1e-12, rel=0)


def test_coco_area_boundary_membership_is_preserved():
    gt = {"images": [image(1, 256, 256)], "categories": [{"id": 1, "name": "one"}],
          "annotations": [annotation(1, 1, 1, [1, 1, 32, 32]),
                          annotation(2, 1, 1, [100, 100, 96, 96])]}
    pred = [detection(1, 1, row["bbox"], .5) for row in gt["annotations"]]
    cache = diagnostic.prepare_coco_cache(gt, pred)
    expected = faster_reference(gt, pred)
    for key, value in expected.items():
        assert cache.score((1,))[key] == pytest.approx(value, abs=1e-12, rel=0)
    assert expected["AP_small"] > 99
    assert expected["AP_medium"] > 99
    assert expected["AP_large"] > 99


def test_ranking_diagnostic_counts_fp_tp_inversions_using_stable_ties():
    gt, pred, _ = coco_fixture()
    report = diagnostic.decompose_errors(gt, pred, include_rows=False)
    row = report["by_iou"]["0.5"]["ranking_by_category"]["1"]
    assert row["true_positives"] == 2
    assert row["false_positives"] == 2
    # Stable score order: FP, TP, FP duplicate (image11), TP(image12).
    assert row["fp_ranked_before_tp_pairs"] == 3
    assert row["fp_before_tp_pair_fraction"] == .75


def test_seed_gate_refuses_unbound_error_report_and_does_not_use_zero_ci():
    gt, pred, manifest = coco_fixture()
    cache = diagnostic.prepare_coco_cache(gt, pred)
    errors = diagnostic.decompose_errors(gt, pred, include_rows=False)
    report = diagnostic.paired_bootstrap_2x2(
        {cell: cache for cell in diagnostic.CELLS}, manifest,
        primary_scorers={cell: (lambda ids: {"AP": 50.}) for cell in diagnostic.CELLS},
        policy=fixture_policy(), error_reports={cell: errors for cell in diagnostic.CELLS})
    assert report["seed_gate"]["available"] is True
    assert report["seed_gate"]["checks"]["cluster_small_ci_lower_positive"] is False
    assert report["seed_gate"]["eligible_for_fixed_paired_seed1_and_seed2"] is False
    broken = copy.deepcopy(errors)
    broken["prediction_sha256"] = "0" * 64
    with pytest.raises(diagnostic.DiagnosticError, match="identity"):
        diagnostic.paired_bootstrap_2x2(
            {cell: cache for cell in diagnostic.CELLS}, manifest,
            primary_scorers={cell: (lambda ids: {"AP": 50.}) for cell in diagnostic.CELLS},
            policy=fixture_policy(),
            error_reports=dict(zip(diagnostic.CELLS, (errors, errors, errors, broken))))


def raw_coverage_fixture():
    gt = {"images": [image(1)], "categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}],
          "annotations": [annotation(1, 1, 1, [10, 10, 10, 10]),
                          annotation(2, 1, 1, [40, 40, 10, 10]),
                          annotation(3, 1, 1, [105, 105, 10, 10])]}
    logits = np.full((3, 10), -10., dtype=np.float32)
    logits[0, 0], logits[0, 1], logits[1, 0], logits[2, 1] = 2., 3., -5., 4.
    boxes = np.asarray([[15 / 128, 15 / 128, 10 / 128, 10 / 128],
                        [45 / 128, 45 / 128, 10 / 128, 10 / 128],
                        [75 / 128, 75 / 128, 10 / 128, 10 / 128]], dtype=np.float32)
    scores = (1. / (1. + np.exp(-logits.astype(np.float64)))).astype(np.float32)
    top = np.asarray([21, 1], dtype=np.int64)
    raw = {1: {"pred_logits": logits, "pred_boxes": boxes,
               "pred_sigmoid_scores": scores, "topk_indices": top}}
    xywh = diagnostic._raw_boxes_original_xywh(boxes, 128, 128)
    predictions = [detection(1, int(k) % 10 + 1, xywh[int(k) // 10].tolist(),
                             float(scores.reshape(-1)[k])) for k in top]
    return gt, predictions, raw


def test_raw_query_coverage_separates_geometry_absence_flat_truncation_and_class_retention():
    gt, predictions, raw = raw_coverage_fixture()
    report = diagnostic.raw_query_coverage(gt, predictions, raw,
                                           policy=fixture_policy(images=1, clusters=1))
    counts = report["saved_candidate_absence_partition_at_0_1"]
    assert counts == {"saved_geometric_candidate_exists": 1,
                      "geometric_queries_lost_to_flat_topk": 1, "raw_geometry_absent": 1}
    rows = {r["annotation_id"]: r for r in report["per_gt"]}
    assert rows[1]["best_iou_query_id"] == 0
    assert rows[1]["highest_raw_iou"] == 1.
    assert rows[1]["correct_pair_score_rank_canonical"] == 3
    assert rows[1]["correct_pair_actual_flat_topk_retained"] is False
    assert rows[1]["best_query_top1_correct"] is False
    assert rows[2]["coverage"]["0.5"]["raw_geometry_covered"] is True
    assert rows[2]["coverage"]["0.5"]["saved_any_class_covered"] is False
    assert rows[3]["coverage"]["0.5"]["raw_geometry_covered"] is False
    assert report["by_iou"]["0.5"]["raw_geometry_covered"] == 2
    assert report["by_iou"]["0.5"]["saved_correct_class_covered"] == 0
    assert report["image_checks"][0]["max_score_abs_difference"] == 0.


def test_raw_score_tie_rank_never_overrides_actual_gpu_topk_membership():
    gt = {"images": [image(1)],
          "categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}],
          "annotations": [annotation(1, 1, 1, [10, 10, 10, 10])]}
    logits = np.full((1, 10), -10., dtype=np.float32)
    logits[0, :2] = 0.
    boxes = np.asarray([[15 / 128, 15 / 128, 10 / 128, 10 / 128]], dtype=np.float32)
    scores = (1. / (1. + np.exp(-logits.astype(np.float64)))).astype(np.float32)
    raw = {1: {"pred_logits": logits, "pred_boxes": boxes,
               "pred_sigmoid_scores": scores, "topk_indices": np.asarray([1], dtype=np.int64)}}
    predictions = [detection(1, 2, [10, 10, 10, 10], .5)]
    report = diagnostic.raw_query_coverage(gt, predictions, raw,
                                           policy=fixture_policy(images=1, clusters=1))
    row = report["per_gt"][0]
    assert row["correct_pair_score_rank_canonical"] == 1
    assert row["correct_pair_score_rank_tie_interval"] == [1, 2]
    assert row["correct_pair_actual_flat_topk_retained"] is False


def test_raw_coverage_requires_runtime_scores_actual_topk_and_exact_geometry():
    gt, predictions, raw = raw_coverage_fixture()
    policy = fixture_policy(images=1, clusters=1)
    missing = copy.deepcopy(raw)
    del missing[1]["pred_sigmoid_scores"]
    with pytest.raises(diagnostic.DiagnosticError, match="runtime sigmoid"):
        diagnostic.raw_query_coverage(gt, predictions, missing, policy=policy)
    bad = copy.deepcopy(raw)
    bad[1]["topk_indices"][1] = 10  # A low-score pair cannot be the saved top2.
    with pytest.raises(diagnostic.DiagnosticError, match="score top-k"):
        diagnostic.raw_query_coverage(gt, predictions, bad, policy=policy)
    changed = copy.deepcopy(predictions)
    changed[0]["bbox"][0] += 1.
    with pytest.raises(diagnostic.DiagnosticError, match="reproduce"):
        diagnostic.raw_query_coverage(gt, changed, raw, policy=policy)


def test_frozen_method_binds_the_raw_query_observation_contract():
    method = diagnostic.frozen_method()
    assert method["raw_query_observation"]["iou_thresholds"] == [.1, .5, .75]
    assert method["raw_query_observation"]["probabilities"] == "saved_runtime_float32_sigmoid"
    assert "topk_indices" in method["raw_query_observation"]["required_arrays"]
