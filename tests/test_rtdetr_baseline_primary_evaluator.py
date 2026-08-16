from __future__ import annotations

import copy
import hashlib
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sparse_rtdetr.baseline.primary_evaluator import (
    AUTHORITY_COMMIT,
    AUTHORITY_FILE_SHA256,
    AUTHORITY_INVENTORY_SHA256,
    AUTHORITY_MANIFEST_CANONICAL_SHA256,
    AUTHORITY_MANIFEST_RAW_SHA256,
    IOU_THRESHOLDS,
    MAX_DETS,
    METRIC_NAMES,
    PrimaryEvaluatorContractError,
    evaluate_primary_v1,
    load_primary_evaluator_authority_manifest,
    load_primary_evaluator_contract,
    primary_evaluator_contract_binding,
    validate_primary_evaluator_contract,
    validate_primary_evaluator_result,
)
from sparse_rtdetr.data_protocol.evaluation import (
    COCODiagnosticInput,
    Detection,
    PrimaryEvaluatorInput,
    PrimaryEvaluatorInputV2,
    PrimaryGroundTruth,
    PrimaryImageV2,
    assert_primary_input,
    assert_secondary_input,
)


ROOT = Path(__file__).resolve().parents[1]


def _image(image_id: str = "image-001", width: int = 32, height: int = 32) -> PrimaryImageV2:
    return PrimaryImageV2(image_id, width, height)


def _det(image_id: str = "image-001", category: int = 1, box=(2.0, 2.0, 12.0, 12.0), score: float = 1.0) -> Detection:
    return Detection(image_id, category, tuple(float(item) for item in box), float(score))


def _gt(image_id: str = "image-001", category: int = 1, box=(2.0, 2.0, 12.0, 12.0), *, ignored: bool = False, ignore_region: bool = False) -> PrimaryGroundTruth:
    box = tuple(float(item) for item in box)
    area = (box[2] - box[0]) * (box[3] - box[1])
    return PrimaryGroundTruth(
        f"ann-{image_id}-{category}-{ignored}-{ignore_region}",
        image_id,
        category,
        box,
        float(area),
        ignore_region,
        ignored,
    )


def _input(images, detections=(), ground_truth=()):
    return PrimaryEvaluatorInputV2(tuple(images), tuple(detections), tuple(ground_truth))


@pytest.fixture(scope="module")
def binding():
    return primary_evaluator_contract_binding(ROOT)


def _reference_round(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _reference_xywh(box):
    return box[0], box[1], box[2] - box[0], box[3] - box[1]


def _reference_rounded_box(box):
    x, y, width, height = _reference_xywh(box)
    return max(1, _reference_round(x)), max(1, _reference_round(y)), max(1, _reference_round(width)), max(1, _reference_round(height))


def _reference_integral(image, regions):
    pixels = [[0.0 for _ in range(image.width)] for _ in range(image.height)]
    for region in regions:
        x, y, width, height = _reference_rounded_box(region.bbox_xyxy)
        x = max(1, min(image.width, x))
        y = max(1, min(image.height, y))
        x_end = min(image.width, x + width)
        y_end = min(image.height, y + height)
        for row in range(y - 1, y_end):
            for col in range(x - 1, x_end):
                pixels[row][col] = 1.0
    integral = [[0.0 for _ in range(image.width)] for _ in range(image.height)]
    for row in range(image.height):
        for col in range(image.width):
            integral[row][col] = pixels[row][col]
            if row:
                integral[row][col] += integral[row - 1][col]
            if col:
                integral[row][col] += integral[row][col - 1]
            if row and col:
                integral[row][col] -= integral[row - 1][col - 1]
    return integral


def _reference_fraction(integral, image, box):
    x, y, width, height = _reference_rounded_box(box)
    x = max(1, min(image.width, x))
    y = max(1, min(image.height, y))
    x_end = max(1, min(image.width, x + width))
    y_end = max(1, min(image.height, y + height))
    tl = integral[y - 1][x - 1]
    tr = integral[y - 1][x_end - 1]
    bl = integral[y_end - 1][x - 1]
    br = integral[y_end - 1][x_end - 1]
    return (tl + br - tr - bl) / float(width * height)


def _reference_overlap(det, gt, ignored):
    dx, dy, dw, dh = det
    gx, gy, gw, gh = gt
    width = min(dx + dw, gx + gw) - max(dx, gx)
    height = min(dy + dh, gy + gh) - max(dy, gy)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return intersection / (dw * dh if ignored else dw * dh + gw * gh - intersection)


def _reference_match(detections, ground_truth, threshold):
    ordered_gt = sorted(enumerate(ground_truth), key=lambda item: (item[1][1], item[0]))
    gt_boxes = [item[1][0] for item in ordered_gt]
    gt_ignored = [item[1][1] for item in ordered_gt]
    gt_state = [-1 if item else 0 for item in gt_ignored]
    det_state = [0] * len(detections)
    for det_index, (_, det_box) in enumerate(detections):
        best_overlap = threshold
        best_gt = -1
        best_match = 0
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_state[gt_index] == 1:
                continue
            if best_match != 0 and gt_state[gt_index] == -1:
                break
            overlap = _reference_overlap(det_box, gt_box, gt_state[gt_index] == -1)
            if overlap < best_overlap:
                continue
            best_overlap = overlap
            best_gt = gt_index
            best_match = -1 if gt_state[gt_index] == -1 else 1
        if best_match == -1:
            det_state[det_index] = -1
        elif best_match == 1:
            gt_state[best_gt] = 1
            det_state[det_index] = 1
    original = [0] * len(ground_truth)
    for index, (original_index, _) in enumerate(ordered_gt):
        original[original_index] = gt_state[index]
    return original, det_state


def _reference_vocap(recall, precision):
    if not recall:
        return 0.0
    mrec = [0.0, *recall, 1.0]
    mpre = [0.0, *precision, 0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changed = [index for index in range(1, len(mrec)) if mrec[index] != mrec[index - 1]]
    return sum((mrec[index] - mrec[index - 1]) * mpre[index] for index in changed)


def _reference_eval(value: PrimaryEvaluatorInputV2):
    by_image = {image.image_id: image for image in value.images}
    regions = {image.image_id: [] for image in value.images}
    for item in value.ground_truth:
        if item.category_id == 0 or item.ignore_region:
            regions[item.image_id].append(item)
    integrals = {image_id: _reference_integral(by_image[image_id], rows) for image_id, rows in regions.items()}
    gt = {image.image_id: [] for image in value.images}
    det = {image.image_id: [] for image in value.images}
    for item in value.ground_truth:
        if item.ignore_region or item.category_id in (0, 11):
            continue
        if _reference_fraction(integrals[item.image_id], by_image[item.image_id], item.bbox_xyxy) < 0.5:
            gt[item.image_id].append(item)
    for item in value.detections:
        if _reference_fraction(integrals[item.image_id], by_image[item.image_id], item.bbox_xyxy) < 0.5:
            det[item.image_id].append(item)
    occurrences = []
    for image in value.images:
        occurrences.extend(sorted({item.category_id for item in gt[image.image_id]}))
    ap = [[0.0 for _ in IOU_THRESHOLDS] for _ in range(10)]
    ar = [[[0.0 for _ in MAX_DETS] for _ in IOU_THRESHOLDS] for _ in range(10)]
    for class_index, category in enumerate(range(1, 11)):
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            for max_index, max_dets in enumerate(MAX_DETS):
                gt_matches = []
                records = []
                for image in value.images:
                    image_gt = [item for item in gt[image.image_id] if item.category_id == category]
                    image_det = [item for item in det[image.image_id]][:max_dets]
                    image_det = [item for item in image_det if item.category_id == category]
                    gt_states, det_states = _reference_match(
                        [(_item.score, _reference_xywh(_item.bbox_xyxy)) for _item in image_det],
                        [(_reference_xywh(_item.bbox_xyxy), _item.ignored) for _item in image_gt],
                        threshold,
                    )
                    gt_matches.extend(gt_states)
                    records.extend((_item.score, state) for _item, state in zip(image_det, det_states))
                records = [record for _, record in sorted(enumerate(records), key=lambda item: (-item[1][0], item[0]))]
                tp = 0
                fps = 0
                recall = []
                precision = []
                for _, state in records:
                    tp += state == 1
                    fps += state == 0
                    recall.append(tp / max(1, len(gt_matches)))
                    precision.append(tp / max(1, tp + fps))
                if gt_matches:
                    ar[class_index][threshold_index][max_index] = max(recall or [0.0]) * 100.0
                if max_index == 3:
                    ap[class_index][threshold_index] = _reference_vocap(recall, precision) * 100.0
    rows = [category - 1 for category in occurrences]
    if rows:
        metrics = (
            sum(ap[row][col] for row in rows for col in range(10)) / (len(rows) * 10),
            sum(ap[row][0] for row in rows) / len(rows),
            sum(ap[row][5] for row in rows) / len(rows),
            *[sum(ar[row][col][max_index] for row in rows for col in range(10)) / (len(rows) * 10) for max_index in range(4)],
        )
    else:
        metrics = (0.0,) * 7
    return metrics, ap, ar


def test_authority_manifest_and_contract_identities(binding):
    manifest = load_primary_evaluator_authority_manifest(ROOT)
    assert manifest["commit_oid"] == AUTHORITY_COMMIT
    assert manifest["tree_oid"] == "038b9e68c6e9a93a64662a4d7a39be2cd2c0654e"
    assert manifest["inventory"]["canonical_inventory_sha256"] == AUTHORITY_INVENTORY_SHA256
    assert binding["authority_manifest_raw_sha256"] == AUTHORITY_MANIFEST_RAW_SHA256
    assert binding["authority_manifest_canonical_sha256"] == AUTHORITY_MANIFEST_CANONICAL_SHA256
    assert binding["config"]["policy"]["independent_audit_pass"] is False
    assert binding["config"]["policy"]["training_gate_open"] is False


def test_authority_source_trace_is_exact(binding):
    rows = {row["relative_path"]: row["sha256"] for row in binding["authority_manifest"]["inventory"]["rows"]}
    assert {key: rows[key] for key in AUTHORITY_FILE_SHA256} == AUTHORITY_FILE_SHA256


def test_source_exact_reference_matches_perfect_case(binding):
    value = _input([_image()], [_det()], [_gt()])
    result = evaluate_primary_v1(value, binding)
    metrics, ap, ar = _reference_eval(value)
    assert result.metrics == pytest.approx(metrics, rel=0, abs=1e-12)
    for actual_row, expected_row in zip(result.ap_by_class_iou, ap):
        assert actual_row == pytest.approx(expected_row, rel=0, abs=1e-12)
    for actual_class, expected_class in zip(result.ar_by_class_iou_max_dets, ar):
        for actual_row, expected_row in zip(actual_class, expected_class):
            assert actual_row == pytest.approx(expected_row, rel=0, abs=1e-12)
    assert result.AP == 100.0
    assert result.AP50 == 100.0
    assert result.AP75 == 100.0
    assert result.AR1 == 100.0
    assert result.AR500 == 100.0
    validate_primary_evaluator_result(result, value, binding)


def test_false_positive_false_negative_and_empty_are_finite(binding):
    image = _image()
    false_positive = _input([image], [_det(box=(20.0, 20.0, 25.0, 25.0))], [_gt()])
    false_negative = _input([image], [], [_gt()])
    empty = _input([image], [], [])
    assert evaluate_primary_v1(false_positive, binding).AP == 0.0
    assert evaluate_primary_v1(false_negative, binding).AR500 == 0.0
    result = evaluate_primary_v1(empty, binding)
    assert result.metrics == (0.0,) * 7
    assert all(math.isfinite(value) for value in result.metrics)


def test_max_dets_prefix_is_before_category_filter(binding):
    value = _input(
        [_image()],
        [_det(category=2, score=0.99), _det(category=1, score=0.50)],
        [_gt(category=1)],
    )
    result = evaluate_primary_v1(value, binding)
    assert result.AR1 == 0.0
    assert result.AR500 == 100.0
    assert result.AP == 100.0


def test_equal_score_order_is_preserved(binding):
    first_fp = _det(box=(20.0, 20.0, 25.0, 25.0), score=0.5)
    second_tp = _det(score=0.5)
    value = _input([_image()], [first_fp, second_tp], [_gt()])
    result = evaluate_primary_v1(value, binding)
    assert result.AP50 == pytest.approx(50.0, rel=0, abs=1e-12)


def test_ignored_gt_uses_ignored_overlap_and_can_absorb_multiple(binding):
    value = _input(
        [_image()],
        [_det(score=0.9), _det(box=(3.0, 3.0, 11.0, 11.0), score=0.8)],
        [_gt(ignored=True)],
    )
    result = evaluate_primary_v1(value, binding)
    assert result.AP == 0.0
    assert result.match_counts_by_class_iou_max_dets[0][0][3] == (1, 0, 0, 2)


def test_spatial_ignore_removes_detection_at_or_above_fraction(binding):
    region = _gt(category=0, box=(1.0, 1.0, 17.0, 17.0), ignore_region=True, ignored=True)
    value = _input([_image(width=20, height=20)], [_det(box=(1.0, 1.0, 17.0, 17.0))], [region, _gt()])
    result = evaluate_primary_v1(value, binding)
    assert result.evaluated_detection_count == 0
    assert result.AP == 0.0


def test_category_zero_and_category_eleven_are_not_scored(binding):
    value = _input([_image()], [_det()], [_gt(category=0, ignore_region=True, ignored=True), _gt(category=11, ignored=True)])
    result = evaluate_primary_v1(value, binding)
    assert result.eval_class_occurrences == ()
    assert result.metrics == (0.0,) * 7


def test_result_is_detached_and_repacking_is_rejected(binding):
    value = _input([_image()], [_det()], [_gt()])
    result = evaluate_primary_v1(value, binding)
    with pytest.raises(PrimaryEvaluatorContractError):
        validate_primary_evaluator_result(replace(result, AP=0.0), value, binding)
    with pytest.raises(PrimaryEvaluatorContractError):
        validate_primary_evaluator_result(replace(result, canonical_result_sha256="0" * 64), value, binding)
    assert isinstance(result.ap_by_class_iou, tuple)
    assert isinstance(result.ar_by_class_iou_max_dets, tuple)


def test_contract_mutations_fail_closed(binding):
    bad = copy.deepcopy(binding["config"])
    bad["algorithm"]["max_dets"][0] = True
    with pytest.raises(PrimaryEvaluatorContractError):
        validate_primary_evaluator_contract(bad, binding["authority_manifest"])
    bad = copy.deepcopy(binding["config"])
    bad["authority"]["manifest_relative_path"] = "other.json"
    with pytest.raises(PrimaryEvaluatorContractError):
        validate_primary_evaluator_contract(bad, binding["authority_manifest"])
    bad_binding = dict(binding)
    bad_binding["config_raw_sha256"] = "0" * 64
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([]), bad_binding)


def test_input_validation_rejects_unsorted_or_invalid_types(binding):
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([_image()], [_det(score=0.1), _det(score=0.2)], []), binding)
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([_image()], [Detection("image-001", True, (2.0, 2.0, 12.0, 12.0), 1.0)], []), binding)
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([_image()], [], [PrimaryGroundTruth("a", "image-001", None, (2.0, 2.0, 12.0, 12.0), 100.0, False, False)]), binding)


def test_secondary_schema_cannot_enter_primary_and_primary_cannot_enter_secondary():
    primary = PrimaryEvaluatorInput((), (), ())
    secondary = COCODiagnosticInput((), (), ())
    with pytest.raises(Exception):
        assert_primary_input(secondary)
    with pytest.raises(Exception):
        assert_secondary_input(primary)


def test_primary_output_has_no_ap_small_and_no_torch_import():
    from sparse_rtdetr.baseline import primary_evaluator

    source = Path(primary_evaluator.__file__).read_text(encoding="utf-8")
    assert "torch" not in source
    assert "AP_small" not in METRIC_NAMES
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    isolated = subprocess.run(
        [sys.executable, "-c", "from sparse_rtdetr.baseline import primary_evaluator; import sys; raise SystemExit(0 if 'torch' not in sys.modules else 3)"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr


def test_contract_config_is_closed_and_exact(binding):
    config = load_primary_evaluator_contract(ROOT)
    assert config["algorithm"]["iou_thresholds"] == list(IOU_THRESHOLDS)
    assert config["algorithm"]["max_dets"] == list(MAX_DETS)
    assert config["output_contract"]["metric_names"] == list(METRIC_NAMES)
    assert config["output_contract"]["ap_small_emitted"] is False
    assert config["policy"]["confirmatory_access_allowed"] is False
    assert config["policy"]["test_access_allowed"] is False
