from __future__ import annotations

import copy
import errno
import hashlib
import math
import os
import random
import socket
import stat
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
from sparse_rtdetr.baseline import primary_evaluator as evaluator_module
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
    counts = [[[[0 for _ in range(4)] for _ in MAX_DETS] for _ in IOU_THRESHOLDS] for _ in range(10)]
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
                counts[class_index][threshold_index][max_index] = (
                    len(gt_matches),
                    sum(state == 1 for _, state in records),
                    sum(state == 0 for _, state in records),
                    sum(state == -1 for _, state in records),
                )
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
    return metrics, ap, ar, counts


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
    metrics, ap, ar, counts = _reference_eval(value)
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
    assert result.match_counts_by_class_iou_max_dets == tuple(tuple(tuple(tuple(row) for row in iou) for iou in category) for category in counts)
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


def test_authoritative_binding_requires_all_frozen_identity_fields(binding):
    value = _input([])
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(value, binding["config"])
    for key in ("config_raw_size_bytes", "config_raw_sha256", "authority_manifest_raw_size_bytes"):
        bad = dict(binding)
        bad.pop(key)
        with pytest.raises(PrimaryEvaluatorContractError):
            evaluate_primary_v1(value, bad)
    bad = dict(binding)
    bad["extra"] = False
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(value, bad)
    bad = dict(binding)
    bad["config_raw_size_bytes"] = True
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(value, bad)
    bad = dict(binding)
    bad["config_raw_sha256"] = "A" * 64
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(value, bad)


def test_recursive_builtin_contract_and_manifest_closure(binding):
    class TextSubclass(str):
        pass

    bad = copy.deepcopy(binding)
    bad["config"]["algorithm"]["rounding"] = TextSubclass("matlab_half_away_from_zero")
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([]), bad)
    bad = copy.deepcopy(binding)
    bad["authority_manifest"]["inventory"]["rows"][0]["git_blob_oid"] = TextSubclass(
        bad["authority_manifest"]["inventory"]["rows"][0]["git_blob_oid"]
    )
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([]), bad)


def test_input_identity_and_category_relations_fail_closed(binding):
    duplicate = replace(_gt(), annotation_id="duplicate")
    duplicate_again = replace(_gt(category=2), annotation_id="duplicate")
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([_image()], [], [duplicate, duplicate_again]), binding)
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([_image()], [], [_gt(ignore_region=True)]), binding)
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([_image()], [], [_gt(category=0, ignore_region=False)]), binding)
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluate_primary_v1(_input([_image()], [], [_gt(category=11, ignore_region=True)]), binding)


def test_result_schema_rejects_json_equivalent_repacks(binding):
    class TupleSubclass(tuple):
        pass

    class FloatSubclass(float):
        pass

    class IntSubclass(int):
        pass

    value = _input([_image()], [_det()], [_gt()])
    result = evaluate_primary_v1(value, binding)
    mutations = [
        replace(result, metrics=TupleSubclass(result.metrics)),
        replace(result, AP=FloatSubclass(result.AP)),
        replace(result, evaluated_image_count=IntSubclass(result.evaluated_image_count)),
        replace(result, ap_by_class_iou=result.ap_by_class_iou[:-1]),
        replace(result, ar_by_class_iou_max_dets=TupleSubclass(result.ar_by_class_iou_max_dets)),
        replace(result, match_counts_by_class_iou_max_dets=((),) + result.match_counts_by_class_iou_max_dets[1:]),
    ]
    for mutation in mutations:
        with pytest.raises(PrimaryEvaluatorContractError):
            validate_primary_evaluator_result(mutation, value, binding)


def _read_fixture(root: Path, relative: str = "payload") -> bytes:
    with evaluator_module._RepositoryBoundary(root) as boundary:
        raw = boundary.read_bytes(relative, "fixture")
        boundary.assert_stable()
        return raw


def test_fd_repository_boundary_rejects_symlink_and_hardlink_objects(tmp_path):
    real_root = tmp_path / "repo"
    real_root.mkdir()
    (real_root / "nested").mkdir()
    (real_root / "nested" / "payload").write_bytes(b"payload")
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(PrimaryEvaluatorContractError):
        evaluator_module._RepositoryBoundary(root_link)

    (real_root / "nested-link").symlink_to(real_root / "nested", target_is_directory=True)
    with pytest.raises(PrimaryEvaluatorContractError):
        _read_fixture(real_root, "nested-link/payload")
    (real_root / "final-link").symlink_to(real_root / "nested" / "payload")
    with pytest.raises(PrimaryEvaluatorContractError):
        _read_fixture(real_root, "final-link")
    (real_root / "hardlink").hardlink_to(real_root / "nested" / "payload")
    with pytest.raises(PrimaryEvaluatorContractError):
        _read_fixture(real_root, "hardlink")


@pytest.mark.parametrize("kind", ("directory", "fifo", "socket"))
def test_fd_repository_boundary_rejects_non_regular_final_objects(tmp_path, kind, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "object"
    sock = None
    try:
        if kind == "directory":
            path.mkdir()
        elif kind == "fifo":
            os.mkfifo(path)
        else:
            # The managed sandbox forbids AF_UNIX bind; preserve the production
            # lstat decision with a syscall-faithful stat-result substitution.
            original_stat = evaluator_module.os.stat

            def socket_stat(name, *args, **kwargs):
                observed = original_stat(name, *args, **kwargs)
                if name == "object" and kwargs.get("follow_symlinks") is False and kwargs.get("dir_fd") is not None:
                    fields = list(observed)
                    fields[0] = stat.S_IFSOCK | 0o700
                    return os.stat_result(fields)
                return observed

            monkeypatch.setattr(evaluator_module.os, "stat", socket_stat)
        with pytest.raises(PrimaryEvaluatorContractError):
            _read_fixture(root, "object")
    finally:
        if sock is not None:
            sock.close()


def test_fd_repository_boundary_handles_partial_reads_eintr_and_premature_eof(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "payload").write_bytes(b"0123456789")
    original_read = evaluator_module.os.read
    calls = {"count": 0}

    def partial_read(fd, size):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError(errno.EINTR, "interrupted")
        return original_read(fd, min(size, 2))

    monkeypatch.setattr(evaluator_module.os, "read", partial_read)
    assert _read_fixture(root) == b"0123456789"
    assert calls["count"] > 2

    def premature_eof(fd, size):
        return b""

    monkeypatch.setattr(evaluator_module.os, "read", premature_eof)
    with pytest.raises(PrimaryEvaluatorContractError):
        _read_fixture(root)


def test_fd_repository_boundary_does_not_leak_descriptors(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "payload").write_bytes(b"payload")
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(25):
        assert _read_fixture(root) == b"payload"
    after = len(os.listdir("/proc/self/fd"))
    assert after == before


def _matrix_case(seed: int) -> PrimaryEvaluatorInputV2:
    rng = random.Random(seed)
    images = tuple(_image(f"case-{seed}-image-{index}", 18 + index * 3, 20 + index * 2) for index in range(1 + seed % 3))
    detections = []
    ground_truth = []
    annotation_index = 0
    for image_index, image in enumerate(images):
        if seed % 4 == 0 or image_index == 0:
            region_box = (0.5, 0.5, float(image.width // 2 + 2), float(image.height // 2 + 2))
            ground_truth.append(PrimaryGroundTruth(
                f"case-{seed}-ann-{annotation_index}", image.image_id, 0, region_box,
                (region_box[2] - region_box[0]) * (region_box[3] - region_box[1]), True, True,
            ))
            annotation_index += 1
        if seed % 5 == 0:
            other_box = (2.5, 2.5, 7.5, 8.5)
            ground_truth.append(PrimaryGroundTruth(
                f"case-{seed}-ann-{annotation_index}", image.image_id, 11, other_box,
                (other_box[2] - other_box[0]) * (other_box[3] - other_box[1]), False, bool(seed % 2),
            ))
            annotation_index += 1
        if seed % 9 != 0:
            for offset in range(1 + (seed + image_index) % 4):
                category = 1 + (seed + image_index + offset) % 10
                x1 = 0.5 + ((seed + offset) % 4) * 0.5
                y1 = 0.5 + ((seed + image_index + offset) % 3) * 0.5
                x2 = x1 + 4.5 + (offset % 2)
                y2 = y1 + 4.0 + ((seed + offset) % 2)
                ground_truth.append(PrimaryGroundTruth(
                    f"case-{seed}-ann-{annotation_index}", image.image_id, category,
                    (x1, y1, x2, y2), (x2 - x1) * (y2 - y1), False, bool((seed + offset) % 6 == 0),
                ))
                annotation_index += 1
        image_detections = []
        for offset in range(2 + (seed + image_index) % 5):
            category = 1 + (seed + image_index + offset * 2) % 10
            box = (0.5 + (offset % 3) * 0.5, 0.5 + ((seed + offset) % 3) * 0.5, 6.0 + offset, 6.5 + offset)
            score = (0.9, 0.7, 0.7, 0.4, 0.2, 0.1)[offset]
            image_detections.append(Detection(image.image_id, category, box, score))
        detections.extend(image_detections)
    return _input(images, detections, ground_truth)


def test_independent_reference_matrix_has_100_valid_cases(binding):
    for seed in range(100):
        value = _matrix_case(seed)
        result = evaluate_primary_v1(value, binding)
        metrics, ap, ar, counts = _reference_eval(value)
        assert result.metrics == pytest.approx(metrics, rel=0, abs=1e-12)
        for actual_row, expected_row in zip(result.ap_by_class_iou, ap):
            assert actual_row == pytest.approx(expected_row, rel=0, abs=1e-12)
        for actual_category, expected_category in zip(result.ar_by_class_iou_max_dets, ar):
            for actual_row, expected_row in zip(actual_category, expected_category):
                assert actual_row == pytest.approx(expected_row, rel=0, abs=1e-12)
        expected_counts = tuple(tuple(tuple(tuple(row) for row in iou) for iou in category) for category in counts)
        assert result.match_counts_by_class_iou_max_dets == expected_counts
        validate_primary_evaluator_result(result, value, binding)
