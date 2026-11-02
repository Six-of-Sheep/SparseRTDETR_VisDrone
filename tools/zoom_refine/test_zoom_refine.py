"""r20 zoom-refine unit checks (CPU only, no model, no CUDA, no data access).

usage: python -I tools/zoom_refine/test_zoom_refine.py        (from the worktree root)
"""
import math
import sys

import torch

sys.path.insert(0, ".")
from tools.zoom_refine.zoom_refine import (  # noqa: E402
    BAND_AREA, CANVAS_HW, WINDOW_RAW_HW, Window, box_area, box_iou, canvas_rect,
    control_pixels, invariants, plan_windows, substitute, window_pixels,
)

TESTS = []


def case(fn):
    TESTS.append(fn)
    return fn


def det(query_id, category_id, box, score):
    return {"image_id": "img", "query_id": query_id, "category_id": category_id,
            "score": score, "bbox_xyxy": list(box)}


@case
def test_planner_is_deterministic_and_greedy():
    dets = [det(1, 1, (100, 100, 116, 116), 0.9),
            det(2, 1, (104, 104, 118, 118), 0.8),
            det(3, 2, (1900, 1000, 1916, 1016), 0.7),
            det(4, 1, (500, 500, 600, 600), 0.95)]      # out of band (area 10,000)
    first, stats_a = plan_windows(1920, 1080, dets, windows=2)
    second, stats_b = plan_windows(1920, 1080, dets, windows=2)
    assert first == second, "planner is not deterministic"
    assert stats_a == stats_b
    assert stats_a["band_points"] == 3, stats_a
    assert len(first) == 2, "two separated clusters must need two windows"
    assert stats_a["covered"] == 3 and stats_a["coverage"] == 1.0, stats_a
    assert all(0 <= w.x0 and w.x1 <= 1920 and 0 <= w.y0 and w.y1 <= 1080 for w in first)
    assert all(w.height == WINDOW_RAW_HW[0] and w.width == WINDOW_RAW_HW[1] for w in first)
    assert first[0].crop_box == (0, 0, 960, 540), first[0]
    assert first[1].crop_box == (960, 540, 1920, 1080), first[1]


@case
def test_one_window_can_cover_two_nearby_clusters():
    dets = [det(1, 1, (100, 100, 116, 116), 0.9), det(3, 2, (900, 500, 916, 516), 0.7)]
    windows, stats = plan_windows(1920, 1080, dets, windows=2)
    assert len(windows) == 1 and stats["coverage"] == 1.0, stats


@case
def test_planner_clamps_and_handles_empty():
    tiny, stats = plan_windows(400, 300, [det(1, 1, (10, 10, 20, 20), 0.5)], windows=2)
    assert len(tiny) == 1 and tiny[0].crop_box == (0, 0, 400, 300), tiny
    assert stats["band_points"] == 1 and stats["coverage"] == 1.0
    none, stats_none = plan_windows(100, 100, [], windows=2)
    assert none == [] and stats_none["coverage"] == 0.0


@case
def test_canvas_rect_is_the_inverse_scale():
    # a 960x540 raw window is exactly half the canvas on each axis: magnification 2x
    win = Window(1, 0, 0, 960, 540, 1920, 1080)
    x0, y0, x1, y1 = canvas_rect(win)
    assert (x0, y0) == (0.0, 0.0)
    assert abs(x1 - CANVAS_HW[1] / 2) < 1e-9 and abs(y1 - CANVAS_HW[0] / 2) < 1e-9, (x1, y1)
    middle = Window(1, 480, 270, 480 + 960, 270 + 540, 1920, 1080)
    x0, y0, x1, y1 = canvas_rect(middle)
    assert abs((x1 - x0) - CANVAS_HW[1] / 2) < 1e-9 and abs((y1 - y0) - CANVAS_HW[0] / 2) < 1e-9
    assert abs(x0 - CANVAS_HW[1] / 4) < 1e-9 and abs(y0 - CANVAS_HW[0] / 4) < 1e-9, (x0, y0)


@case
def test_window_and_control_renderings_have_the_canvas_shape():
    from PIL import Image
    image = Image.fromarray(torch.randint(0, 255, (1080, 1920, 3), dtype=torch.uint8).numpy())
    win = Window(1, 480, 270, 480 + 960, 270 + 540, 1920, 1080)
    method = window_pixels(image, win)
    assert tuple(method.shape) == (3, *CANVAS_HW), method.shape
    base = torch.rand(3, *CANVAS_HW)
    control = control_pixels(base, win)
    assert tuple(control.shape) == (3, *CANVAS_HW)
    assert torch.isfinite(control).all()
    flat = control.flatten()
    assert flat.std() > 0, "control rendering is degenerate"


@case
def test_substitution_changes_boxes_only():
    whole = [det(1, 1, (100.0, 100.0, 116.0, 116.0), 0.9),
             det(2, 2, (900.0, 500.0, 916.0, 516.0), 0.8),
             det(3, 1, (500.0, 500.0, 540.0, 540.0), 0.7)]
    windows = [{"image_id": "img", "query_id": 11, "category_id": 1, "score": 0.95, "window_id": 1,
                "bbox_xyxy": [101.0, 101.0, 117.0, 117.0]},
               {"image_id": "img", "query_id": 12, "category_id": 2, "score": 0.99, "window_id": 2,
                "bbox_xyxy": [899.0, 499.0, 917.0, 517.0]}]
    repaired, stats = substitute(whole, windows)
    report = invariants(whole, repaired)
    assert report["ok"], report
    assert report["changed_boxes"] == 2, report
    assert stats["band_considered"] == 2 and stats["repaired"] == 2, stats
    assert repaired[0]["bbox_xyxy"] == [101.0, 101.0, 117.0, 117.0]
    assert repaired[2]["bbox_xyxy"] == whole[2]["bbox_xyxy"], "out-of-band box must not change"
    assert [p["score"] for p in repaired] == [p["score"] for p in whole]
    assert [p["query_id"] for p in repaired] == [p["query_id"] for p in whole]


@case
def test_substitution_respects_iou_floor_and_class():
    whole = [det(1, 1, (100.0, 100.0, 116.0, 116.0), 0.9)]
    far = [{"category_id": 1, "query_id": 5, "score": 0.9, "window_id": 1,
            "bbox_xyxy": [300.0, 300.0, 316.0, 316.0]}]
    other_class = [{"category_id": 2, "query_id": 6, "score": 0.9, "window_id": 1,
                    "bbox_xyxy": [100.0, 100.0, 116.0, 116.0]}]
    for candidates in (far, other_class):
        repaired, stats = substitute(whole, candidates)
        assert stats["repaired"] == 0, stats
        assert repaired[0]["bbox_xyxy"] == whole[0]["bbox_xyxy"]


@case
def test_substitution_counts_are_stable_under_repetition():
    whole = [det(i, 1, (100.0 + i, 100.0, 116.0 + i, 116.0), 0.5 + i / 100.0) for i in range(5)]
    windows = [{"category_id": 1, "query_id": 7 + i, "score": 0.8, "window_id": 1,
                "bbox_xyxy": [100.5 + i, 100.0, 116.5 + i, 116.0]} for i in range(5)]
    one, stats_one = substitute(whole, windows)
    two, stats_two = substitute(whole, windows)
    assert stats_one == stats_two and one == two


@case
def test_band_boundary_follows_coco_small():
    assert box_area((0.0, 0.0, 32.0, 32.0)) == BAND_AREA
    inside = [det(1, 1, (0.0, 0.0, 31.9, 32.0), 0.9)]
    outside = [det(1, 1, (0.0, 0.0, 32.0, 32.0), 0.9)]
    assert plan_windows(100, 100, inside, windows=1)[1]["band_points"] == 1
    assert plan_windows(100, 100, outside, windows=1)[1]["band_points"] == 0


@case
def test_iou_helpers_match_reference():
    assert abs(box_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-12
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(math.isclose(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 5 / 15, rel_tol=1e-12)) in (0, 1)


def main():
    failures = []
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:                     # noqa: BLE001 - report, never hide
            failures.append((fn.__name__, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"TESTS {len(TESTS) - len(failures)}/{len(TESTS)} PASS")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
