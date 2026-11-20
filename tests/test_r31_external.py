"""R31-001 (TASK_R31_EXTERNAL_UAVDT_001): the pure helpers of the external UAVDT-M evaluation.

No model, no data file: the toolkit ignore rule against an independent per-row judgement, the vehicle candidate rule,
the registered label, the magnification bins, the dHash arithmetic, the frame selection of the runner and (when
pycocotools is installed) the COCOeval plumbing: image-subset accumulation, GT boxes as detections, the per-class
top-100 submission.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import r31_core as C  # noqa: E402
import r31_external as E  # noqa: E402


# ---------------------------------------------------------------- toolkit files
def test_parse_tables_accept_crlf_and_an_empty_ignore_file():
    gt = C.parse_gt("1,7,10,20,30,40,1,2,1\r\n2,7,11,21,30,40,1,2,3\r\n")
    assert gt.dtype == np.int64 and gt.tolist() == [[1, 7, 10, 20, 30, 40, 1, 2, 1], [2, 7, 11, 21, 30, 40, 1, 2, 3]]
    assert C.parse_ignore("").shape == (0, 5)
    assert C.parse_ignore("3,-1,5,6,70,80,1,-1,-1\r\n").tolist() == [[3, 5, 6, 70, 80]]
    det = C.parse_detections("4,-1,1.5,2.5,10,20,0.75,1,-1\r\n")
    assert det.tolist() == [[4.0, 1.5, 2.5, 10.0, 20.0, 0.75]]
    with pytest.raises(ValueError):
        C.parse_gt("1,7,10,20,0,40,1,2,1\n")              # a zero side
    with pytest.raises(ValueError):
        C.parse_gt("1,7,10,20,30,40,1,2,4\n")             # a category the toolkit does not define


def test_attribute_members_and_rows():
    assert C.attribute_sequence("M_attr/test/M0701 _attr.txt") == ("test", "M0701")
    assert C.attribute_sequence("M_attr/train/M0101_attr.txt") == ("train", "M0101")
    assert C.attribute_sequence("M_attr/test/") is None and C.attribute_sequence("M_attr/readme.txt") is None
    assert C.parse_attributes("1,0,0,1,0,0,0,1,0,0\r\n") == ["daylight", "low-alt", "side-view"]
    attrs = {s: ["daylight", "low-alt"] for s in C.TEST_SEQUENCES}
    attrs["M0701"] = ["fog", "high-alt", "bird-view"]
    groups = C.sequence_groups(attrs)
    assert groups["fog"] == ["M0701"] == groups["high-alt"] and len(groups["daylight"]) == 19 and groups["night"] == []
    attrs["M0203"] = ["daylight"]
    with pytest.raises(ValueError):
        C.sequence_groups(attrs)


def test_census_counts_use_the_selection_definitions():
    gt = {"A": np.array([[1, 1, 0, 0, 16, 16, 1, 1, 1],          # area 256: not lt16, small
                         [1, 2, 100, 100, 32, 32, 1, 1, 2],     # area 1024: not small in the census
                         [2, 3, 10, 10, 15, 17, 1, 1, 1],       # area 255: lt16
                         [2, 4, 0, 0, 96, 96, 1, 1, 3]])}       # area 9216: ge96
    ign = {"A": np.array([[2, 5, 5, 30, 30]])}                  # frame 2: covers (5, 5)-(35, 35)
    c = C.census(gt, ign)
    assert (c["gt"], c["lt16"], c["16to32"], c["small"], c["32to96"], c["ge96"]) == (4, 1, 1, 2, 1, 1)
    assert c["area_eq_1024"] == 1 and c["area_eq_9216"] == 1 and c["cat"] == {"car": 2, "truck": 1, "bus": 1}
    assert c["gt_center_in_ignore"] == 1 and c["gt_center_in_ignore_small"] == 1 and c["gt_strictly_inside_ignore"] == 1
    assert c["per_sequence"]["A"] == {"gt": 4, "small": 2, "lt16": 1, "ignore_boxes": 1, "frames_with_gt": 2, "first_frame": 1,
                                      "last_frame": 2, "max_gt_per_frame": 2, "frames_with_ignore": 1,
                                      "cat": {"car": 2, "truck": 1, "bus": 1}}


# ---------------------------------------------------------------- the ignore rule
def test_ignore_rule_is_strict_on_every_edge_and_needs_one_rectangle():
    rects = np.array([[10, 10, 20, 20], [30, 10, 20, 20]], dtype=np.float64)     # (10,10)-(30,30) and (30,10)-(50,30)
    boxes = np.array([[11, 11, 29, 29],          # strictly inside the first
                      [10, 11, 29, 29],          # touches the left edge
                      [11, 10, 29, 29],          # touches the top edge
                      [11, 11, 30, 29],          # touches the right edge
                      [11, 11, 29, 30],          # touches the bottom edge
                      [20, 15, 40, 25],          # inside the union, inside neither rectangle
                      [31, 11, 49, 29],          # strictly inside the second
                      [60, 60, 70, 70]], dtype=np.float64)
    assert C.inside_ignore(boxes, rects).tolist() == [True, False, False, False, False, False, True, False]
    assert C.inside_ignore(boxes, np.zeros((0, 4))).tolist() == [False] * 8
    assert C.inside_ignore(np.zeros((0, 4)), rects).shape == (0,)


def test_scoring_path_rule_equals_the_independent_judgement_on_a_detection_table():
    rng = np.random.default_rng(31)
    ign = np.concatenate([rng.integers(1, 6, (40, 1)), rng.integers(0, 60, (40, 2)), rng.integers(5, 60, (40, 2))], 1).astype(np.float64)
    det = np.concatenate([rng.integers(1, 8, (3000, 1)), rng.integers(0, 110, (3000, 2)), rng.integers(1, 25, (3000, 2)),
                          rng.random((3000, 1))], 1).astype(np.float64)          # integer edges: many exact ties with a rectangle edge
    mine = C.deleted_by_frame(det, ign)
    independent = np.array(C.toolkit_deleted(det.tolist(), ign.tolist()), dtype=bool)
    assert mine.dtype == bool and 0 < mine.sum() < len(det) and (mine == independent).all()
    assert not C.deleted_by_frame(det[det[:, 0] > 5], ign).any()                 # frames without a rectangle


# ---------------------------------------------------------------- candidates
def test_vehicle_score_order_and_submission():
    rng = np.random.default_rng(5)
    scores = rng.random((C.QUERIES, C.CLASSES), dtype=np.float32)
    vehicle = C.class_scores(scores, C.VEHICLE_COLUMNS)
    assert vehicle.dtype == np.float32
    assert all(vehicle[q] == max(scores[q, 3], scores[q, 4], scores[q, 5], scores[q, 8]) for q in range(C.QUERIES))
    assert (C.class_scores(scores, C.THREE_CLASS_COLUMNS[1]) == np.maximum(scores[:, 3], scores[:, 4])).all()
    assert (C.class_scores(scores, C.THREE_CLASS_COLUMNS[2]) == scores[:, 5]).all()
    assert (C.class_scores(scores, C.THREE_CLASS_COLUMNS[3]) == scores[:, 8]).all()
    tied = np.array([0.5, 0.9, 0.5, 0.9, 0.1], dtype=np.float32)
    assert C.candidate_order(tied).tolist() == [1, 3, 0, 2, 4]                   # ties by query index
    inside = np.array([False, True, False, False, False])
    assert C.submission(inside, tied).tolist() == [3, 0, 2, 4]                   # deleted, not refilled
    assert C.submission(inside, tied, limit=2).tolist() == [3, 0]
    assert C.submission(np.zeros(5, dtype=bool), tied).tolist() == [1, 3, 0, 2, 4]


def test_pixel_boxes_and_the_gt_round_trip():
    xywh = np.array([[0, 0, 1024, 540], [100, 200, 17, 23], [1000, 500, 24, 41], [3, 5, 1, 1]])
    emitted = C.normalised_cxcywh(xywh, 1024, 540)
    assert emitted.dtype == np.float32
    px = C.pixel_boxes(emitted, 1024, 540)
    truth = np.stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], 1)
    assert px.dtype == np.float32 and np.abs(px.astype(np.float64) - truth).max() < 1e-3
    with pytest.raises(ValueError):
        C.pixel_boxes(emitted.astype(np.float64), 1024, 540)


def test_pixel_boxes_is_the_arithmetic_of_the_frozen_ranking():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    boxes = rng.random((C.QUERIES, 4), dtype=np.float32) * np.float32(0.9) + np.float32(0.01)
    t = torch.from_numpy(boxes)
    xy = torch.cat((t[:, :2] - t[:, 2:] / 2, t[:, :2] + t[:, 2:] / 2), -1)
    for width, height in ((1024, 540), (1360, 765), (2000, 1500)):
        want = xy * torch.tensor([width, height] * 2, dtype=torch.float32)
        assert np.array_equal(C.pixel_boxes(boxes, width, height), want.numpy())


def test_records_equal_reports_the_first_difference():
    a = [{"image_id": "x", "query_id": 1, "score": 0.5}, {"image_id": "x", "query_id": 2, "score": 0.25}]
    assert C.records_equal(a, [dict(r) for r in a]) is None
    b = [dict(r) for r in a]
    b[1]["score"] = 0.2500001
    assert C.records_equal(a, b)["index"] == 1
    assert C.records_equal(a, b[:1]) == {"index": None, "lengths": [2, 1]}


# ---------------------------------------------------------------- label and paired differences
def _d(ap, aps, ap75s):
    return {"AP": ap, "APS": aps, "AP75_small": ap75s}


def test_label_cases():
    assert C.label([_d(0.0, 1.0, 1.0)] * 3) == "external_gain_repeated_3of3"                       # the bounds are inclusive
    assert C.label([_d(0.5, 2.0, 1.5), _d(0.2, 1.2, 1.0), _d(0.0, 1.0, 3.0)]) == "external_gain_repeated_3of3"
    assert C.label([_d(0.5, 2.0, 1.5), _d(0.2, 1.2, 0.9), _d(0.1, 1.0, 3.0)]) == "external_gain_reduced"   # AP75-small short once
    assert C.label([_d(-0.1, 2.0, 1.5), _d(0.2, 1.2, 1.9), _d(0.1, 1.0, 3.0)]) == "external_gain_reduced"  # AP below 0 once
    assert C.label([_d(0.5, 0.1, -1.0)] * 3) == "external_gain_reduced"
    assert C.label([_d(0.5, -0.1, 1.0), _d(0.0, -2.0, 0.0), _d(1.0, -0.01, 5.0)]) == "external_gain_reversed"
    assert C.label([_d(0.5, 0.1, 1.0), _d(0.0, -2.0, 0.0), _d(1.0, 0.3, 5.0)]) == "external_gain_mixed"
    assert C.label([_d(0.5, 0.0, 1.0), _d(0.0, 2.0, 2.0), _d(1.0, 0.3, 5.0)]) == "external_gain_mixed"     # a zero is not positive
    assert C.label([_d(0.5, 0.0, 1.0), _d(0.0, -2.0, 2.0), _d(1.0, -0.3, 5.0)]) == "external_gain_mixed"   # a zero is not negative
    with pytest.raises(ValueError):
        C.label([_d(0.5, 1.0, 1.0)] * 2)
    with pytest.raises(ValueError):
        C.label([_d(0.5, None, 1.0)] * 3)
    cells = C.rule_cells(_d(-0.2, 1.0, 0.99))
    assert [cells[m]["passed"] for m in ("APS", "AP75_small", "AP")] == [True, False, False]


def test_paired_differences_mean_and_sample_std():
    fg = {0: {"AP": 0.30, "APS": 0.20, "AP75_small": 0.10}, 1: {"AP": 0.31, "APS": 0.22, "AP75_small": None},
          2: {"AP": 0.29, "APS": 0.21, "AP75_small": 0.12}}
    l0 = {0: {"AP": 0.29, "APS": 0.18, "AP75_small": 0.10}, 1: {"AP": 0.31, "APS": 0.19, "AP75_small": 0.11},
          2: {"AP": 0.30, "APS": 0.20, "AP75_small": 0.10}}
    p = C.paired(fg, l0, metrics=C.MAIN)
    assert [round(p["per_seed"][s]["APS"], 9) for s in (0, 1, 2)] == [2.0, 3.0, 1.0]
    assert abs(p["APS"]["mean"] - 2.0) < 1e-9 and abs(p["APS"]["sample_std"] - 1.0) < 1e-9          # n - 1
    assert abs(p["AP"]["mean"] - 0.0) < 1e-9 and abs(p["AP"]["sample_std"] - 1.0) < 1e-9
    assert p["per_seed"][1]["AP75_small"] is None and p["AP75_small"] == {"mean": None, "sample_std": None}
    assert C.mean_std([1.0]) == {"mean": 1.0, "sample_std": None}
    with pytest.raises(ValueError):
        C.paired(fg, {0: l0[0]})


# ---------------------------------------------------------------- magnification tables
def test_bins_and_geometry_reading():
    edges = [0.5, 0.999999, 1.0, 1.199999, 1.2, 1.399999, 1.4, 3.0]
    assert [C.bin_index(g) for g in edges] == [0, 0, 1, 1, 2, 2, 3, 3] == C.bin_indices(edges).tolist()
    g = np.array([0.9, 1.1, 1.1, 1.3, 1.5, 2.0])
    area = np.array([100.0, 255.0, 256.0, 1024.0, 1025.0, 9216.0])
    r = C.geometry_reading(g, area)
    assert r["small"]["n"] == 4 and r["small"]["bin_counts"] == [1, 2, 1, 0] and r["small"]["frac_g_lt_1"] == 0.25
    assert r["small"]["frac_g_lt_1p2"] == 0.75 and r["small_area<16^2"]["n"] == 2 and r["small_16^2<=area<=32^2"]["n"] == 2
    assert r["medium"]["n"] == 3 and r["large"]["n"] == 1                                    # COCO's closed end points
    assert abs(r["small"]["g_geomean"] - math.exp(np.log([0.9, 1.1, 1.1, 1.3]).mean())) < 1e-12
    assert C.geometry_reading(np.zeros(0), np.zeros(0))["small"] == {"n": 0}


def test_bin_table_rows_and_recall():
    g = np.array([0.9, 0.95, 1.1, 1.3, 1.5, 1.6])
    area = np.array([100.0, 300.0, 100.0, 300.0, 1024.0, 255.0])
    iou = {"L0": np.array([0.6, 0.4, 0.8, 0.5, 0.74, 0.0]), "FG": np.array([0.8, 0.5, 0.75, 0.2, 0.76, 0.9])}
    rows = C.bin_table(g, area, iou, pairs=(("FG", "L0"),))
    assert len(rows) == 5 * 3 and [r["n"] for r in rows if r["area_group"] == "small"] == [2, 1, 1, 2, 6]
    first = rows[0]
    assert first["bin"] == "g<1.0" and first["recall@0.5"] == {"L0": 0.5, "FG": 1.0} and first["difference_pp@0.5"] == {"FG-L0": 50.0}
    assert first["recall@0.75"] == {"L0": 0.0, "FG": 0.5}
    total = [r for r in rows if r["bin"] == "all" and r["area_group"] == "small"][0]
    assert total["recall@0.5"] == {"L0": 4 / 6, "FG": 5 / 6}
    empty = [r for r in rows if r["bin"] == "1.0<=g<1.2" and r["area_group"] == "16^2<=area<=32^2"][0]
    assert empty["n"] == 0 and empty["recall@0.5"] == {"L0": None, "FG": None} and empty["difference_pp@0.5"] == {"FG-L0": None}
    with pytest.raises(ValueError):
        C.bin_table(g, area + 1000.0, iou)


def test_best_iou():
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 120]], dtype=np.float64)
    cand = np.array([[0, 0, 10, 5], [5, 5, 15, 15], [200, 200, 210, 210]], dtype=np.float64)
    assert C.best_iou(gt, cand).tolist() == [0.5, 0.0]
    assert C.best_iou(gt, np.zeros((0, 4))).tolist() == [0.0, 0.0] and C.best_iou(np.zeros((0, 4)), cand).shape == (0,)
    degenerate = np.array([[3, 3, 3, 8]], dtype=np.float64)                      # a zero-width candidate scores IoU 0
    assert C.best_iou(gt, degenerate).tolist() == [0.0, 0.0]


def test_canvas_sizes_and_quantiles():
    r = C.canvas_sizes([20.0, 10.0], [18.0, 9.0], 1024, 540, (768, 1344))
    assert abs(r["width"]["q50"] - 15.0 * 1344 / 1024) < 1e-9 and abs(r["height"]["mean"] - 13.5 * 768 / 540) < 1e-9
    assert r["share_area_le_32^2"] == 1.0 and r["share_area_lt_16^2"] == 0.5
    assert C.quantiles([])["q50"] is None and C.quantiles([])["n"] == 0


# ---------------------------------------------------------------- dHash
def test_dhash_and_hamming():
    ramp = np.tile(np.arange(9), (8, 1))
    assert C.dhash(ramp) == 2 ** 64 - 1 and C.dhash(ramp[:, ::-1]) == 0 and C.dhash(np.zeros((8, 9))) == 0
    one = np.zeros((8, 9))
    one[0, 1] = 1                                               # only the first comparison of the first row is "brighter"
    assert C.dhash(one) == 1 << 63
    with pytest.raises(ValueError):
        C.dhash(np.zeros((9, 8)))
    a = np.array([0, 2 ** 64 - 1, 0b1011], dtype=np.uint64)
    b = np.array([0, 0b0001, 2 ** 63], dtype=np.uint64)
    assert C.hamming(a, b).tolist() == [[0, 1, 1], [64, 63, 63], [3, 2, 4]]
    assert C.every_tenth(25) == [1, 11, 21] and C.every_tenth(10) == [1] and C.every_tenth(11) == [1, 11]


# ---------------------------------------------------------------- runner constants and frame selection
def test_runner_units_and_pins():
    assert E.UNITS == E.MAIN_UNITS + E.REFERENCE_UNITS and sorted(E.PIN) == sorted(E.UNITS)
    assert [E.parse(u) for u in ("L0-s0", "FG-s2", "S15-s1")] == [("L0", 0), ("FG", 2), ("S15", 1)]
    with pytest.raises(ValueError):
        E.parse("W-s0")
    assert all(len(p["weights"]) == 64 and len(p["endpoint"][2]) == 64 and p["endpoint"][0].endswith("checkpoint-epoch-120.pth")
               and p["predictions"][0].endswith("top600-predictions.json") for p in E.PIN.values())
    assert len({p["weights"] for p in E.PIN.values()}) == 9 and len({p["endpoint"][2] for p in E.PIN.values()}) == 9
    assert sum(C.SEQUENCE_FRAMES) == C.FRAMES == 16592 and len(C.TEST_SEQUENCES) == 20 and C.FRAMES % E.BATCH == 0
    assert set(C.SEQUENCES_GE_1000_SMALL) < set(C.TEST_SEQUENCES) and len(C.SEQUENCES_GE_1000_SMALL) == 16
    assert float.fromhex(E.MAPPING["fwhm_hex"]) == E.MAPPING["fwhm"] == 19.03125
    assert E.is_dev("/x/round/dev-round-1") and not E.is_dev("/x/round")
    assert set(E.STAGES) == {"data", "gtcheck", "wire", "infer", "geometry", "score", "dhash", "tables"}
    assert E.IMAGE_MEMBER.fullmatch("UAV-benchmark-M/M0203/img000001.jpg").groups() == ("M0203", "000001")
    assert E.IMAGE_MEMBER.fullmatch("UAV-benchmark-M/M0203/../img000001.jpg") is None
    assert E.IMAGE_MEMBER.fullmatch("UAV-benchmark-M/M0203/img000001.jpg.exe") is None


def test_frame_selection_and_annotation_rows():
    full = E.selection({"dev_limit": None})
    assert len(full) == C.FRAMES and full[0] == 0 and full[-1] == C.FRAMES - 1
    dev = E.selection({"dev_limit": 8})
    offsets = np.concatenate([[0], np.cumsum(C.SEQUENCE_FRAMES)])
    assert len(dev) == 160 and dev[:9].tolist() == list(range(8)) + [1007] and (dev.reshape(20, 8)[:, 0] == offsets[:-1]).all()
    gt_offsets = np.array([0, 2, 2, 5, 6])                                       # four frames with 2, 0, 3, 1 rows
    idx, pos = E.selected_rows(gt_offsets, np.array([0, 2, 3]))
    assert idx.tolist() == [0, 1, 2, 3, 4, 5] and pos.tolist() == [0, 0, 1, 1, 1, 2]
    idx, pos = E.selected_rows(gt_offsets, np.array([1]))
    assert idx.shape == (0,) and pos.shape == (0,)
    assert E.xyxy_of([[1, 2, 3, 4]]).tolist() == [[1.0, 2.0, 4.0, 6.0]]


def test_pack_sorts_by_frame_and_handles_rows_outside_the_sequence():
    tables = {s: np.zeros((0, 5), dtype=np.int64) for s in C.TEST_SEQUENCES}
    tables["M0203"] = np.array([[2, 10, 10, 5, 5], [1, 20, 20, 5, 5], [2, 30, 30, 5, 5]])
    tables["M0205"] = np.array([[1, 40, 40, 5, 5], [647, 50, 50, 5, 5]])         # M0205 has 646 frames
    frame, rows, offsets, dropped = E._pack(tables, "ignore", True)
    assert frame.tolist() == [0, 1, 1, 1007] and rows[:, 1].tolist() == [20, 10, 30, 40] and dropped == {"M0205": 1}
    assert offsets.shape == (C.FRAMES + 1,) and offsets[:3].tolist() == [0, 1, 3] and offsets[1007] == 3 and offsets[-1] == 4
    with pytest.raises(ValueError):
        E._pack(tables, "GT", False)


# ---------------------------------------------------------------- COCOeval plumbing
def _random_scene(rng, n_images, categories=(1,)):
    gt_image, gt_xywh, gt_cat, det = [], [], [], {"image": [], "category": [], "xyxy": [], "score": []}
    for i in range(n_images):
        for _ in range(int(rng.integers(0, 9))):
            w, h = rng.integers(4, 120, 2)
            x, y = rng.integers(0, 900), rng.integers(0, 400)
            gt_image.append(i)
            gt_xywh.append([x, y, w, h])
            gt_cat.append(int(rng.choice(categories)))
            for _ in range(int(rng.integers(0, 3))):                         # detections near the GT
                jitter = rng.normal(0, 3, 4)
                det["image"].append(i)
                det["category"].append(int(rng.choice(categories)))
                det["xyxy"].append([x + jitter[0], y + jitter[1], x + w + jitter[2], y + h + jitter[3]])
                det["score"].append(float(rng.random()))
        for _ in range(int(rng.integers(0, 6))):                             # background detections
            x, y = rng.random() * 900, rng.random() * 400
            det["image"].append(i)
            det["category"].append(int(rng.choice(categories)))
            det["xyxy"].append([x, y, x + 30, y + 30])
            det["score"].append(float(rng.random()))
    order = np.lexsort((-np.array(det["score"]), np.array(det["image"])))   # by image, then by score
    det = {k: np.array(v)[order] for k, v in det.items()}
    return np.array(gt_image), np.array(gt_xywh).reshape(-1, 4), np.array(gt_cat), det


def test_accumulate_on_an_image_subset_equals_a_direct_evaluation():
    pytest.importorskip("pycocotools")
    rng = np.random.default_rng(2031)
    gi, gx, gc, det = _random_scene(rng, 14)
    names = {1: "vehicle"}
    e = C.cocoeval(C.coco_dataset(14, gi, gx, gc, names), C.detection_records(det["image"], det["category"], det["xyxy"], det["score"]))
    labels = [r[0] for r in C.AREA_RANGES]
    full = C.named_metrics(C.read_cells(*C.accumulate(e), labels))
    assert full["AP"] is not None and 0 < full["AP"] < 1
    keep = np.array([1, 2, 5, 6, 7, 11, 13])
    sub_precision, sub_recall = C.accumulate(e, keep)
    remap = {int(k): n for n, k in enumerate(keep)}
    gm, dm = np.isin(gi, keep), np.isin(det["image"], keep)
    direct = C.cocoeval(C.coco_dataset(len(keep), np.array([remap[int(i)] for i in gi[gm]]), gx[gm], gc[gm], names),
                        C.detection_records(np.array([remap[int(i)] for i in det["image"][dm]]), det["category"][dm],
                                            det["xyxy"][dm], det["score"][dm]))
    want_precision, want_recall = C.accumulate(direct)
    assert np.array_equal(sub_precision, want_precision) and np.array_equal(sub_recall, want_recall)
    again_precision, again_recall = C.accumulate(e)                              # the full reading is not disturbed by a subset reading
    assert C.named_metrics(C.read_cells(again_precision, again_recall, labels)) == full
    with pytest.raises(ValueError):
        C.accumulate(e, [3, 2])


def test_gt_boxes_as_detections_score_one():
    pytest.importorskip("pycocotools")
    rng = np.random.default_rng(7)
    gi, gx, gc, _ = _random_scene(rng, 10)
    px = C.pixel_boxes(C.normalised_cxcywh(gx, *C.IMAGE_WH), *C.IMAGE_WH)
    e = C.cocoeval(C.coco_dataset(10, gi, gx, gc, {1: "vehicle"}), C.detection_records(gi, gc, px, np.ones(len(gi))),
                   area_ranges=C.SIDE_AREA_RANGES, max_dets=(1, 10, 300))
    cells = C.read_cells(*C.accumulate(e), [r[0] for r in C.SIDE_AREA_RANGES])
    assert cells["all"]["AP"] > 1 - 1e-9 and cells["all"]["AR"] == 1.0


def test_three_class_top100_submission_equals_cocoeval_truncation():
    pytest.importorskip("pycocotools")
    rng = np.random.default_rng(11)
    names = dict(C.UAVDT_CLASSES)
    gi = np.zeros(12, dtype=np.int64)
    gx = np.concatenate([rng.integers(0, 800, (12, 2)), rng.integers(10, 60, (12, 2))], 1)
    gc = rng.integers(1, 4, 12)
    boxes = np.concatenate([rng.random((C.QUERIES, 2)) * 800, np.zeros((C.QUERIES, 2))], 1)
    boxes[:, 2:] = boxes[:, :2] + 10 + rng.random((C.QUERIES, 2)) * 50
    boxes[:12] = np.stack([gx[:, 0], gx[:, 1], gx[:, 0] + gx[:, 2], gx[:, 1] + gx[:, 3]], 1) + rng.normal(0, 1, (12, 4))
    scores = rng.random((C.QUERIES, C.CLASSES), dtype=np.float32)
    inside = rng.random(C.QUERIES) < 0.1

    def records(limit):
        image, category, xyxy, score = [], [], [], []
        for k, columns in C.THREE_CLASS_COLUMNS.items():
            s = C.class_scores(scores, columns)
            kept = C.submission(inside, s, limit=limit)
            image.append(np.zeros(len(kept), dtype=np.int64))
            category.append(np.full(len(kept), k))
            xyxy.append(boxes[kept])
            score.append(s[kept])
        return C.detection_records(np.concatenate(image), np.concatenate(category), np.concatenate(xyxy), np.concatenate(score))

    dataset = C.coco_dataset(1, gi, gx, gc, names)
    everything, first_hundred = records(None), records(100)
    assert len(everything) == 3 * int((~inside).sum()) > len(first_hundred) == 300
    a = C.accumulate(C.cocoeval(dataset, everything, area_ranges=C.SIDE_AREA_RANGES))
    b = C.accumulate(C.cocoeval(C.coco_dataset(1, gi, gx, gc, names), first_hundred, area_ranges=C.SIDE_AREA_RANGES))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    per_class = [C.read_cells(a[0], a[1], ["all", "small"], category=k)["all"]["AP"] for k in range(3)]
    assert len(per_class) == 3
