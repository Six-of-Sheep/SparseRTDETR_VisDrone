"""CPU unit checks for the r23 baseline mechanism: mirrored view, geometric pairing, mean fusion.

Run from the frozen release:  <python> -I tools/flipview/test_flipview.py
Exact-boundary cases use a power-of-two canvas so the constructed IoU values are exact in float32.
"""
import inspect
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.flipview.flipview import (NUM_QUERIES, PAIRING_MODES, PARTNER_IOU, SMALL_AREA,  # noqa: E402
                                     cxcywh_to_xyxy, fuse_view_boxes, iou_matrix, mirror_boxes,
                                     pixel_boxes, xyxy_to_cxcywh)

W, H = 1024.0, 512.0                       # powers of two, so every constructed corner is exact
CHECKS = 23
FAR = [[900.0, 200.0, 960.0, 260.0]]       # 3600 px and far away: never small, never a good partner


def boxes_from_pixels(rows):
    """Pixel xyxy rows -> normalized cxcywh rows, exactly, for exact pixel coordinates."""
    pixels = torch.tensor(rows, dtype=torch.float32)
    scale = torch.tensor([W, H] * 2, dtype=torch.float32)
    return xyxy_to_cxcywh(pixels / scale)


def boxes_to_pixels(boxes):
    return pixel_boxes(boxes, W, H)


def two_view(original_rows, mirrored_rows):
    """The module requires one row per query in both views; pad the shorter side with far boxes."""
    original = boxes_from_pixels(original_rows)
    mirrored = boxes_from_pixels(mirrored_rows)
    if len(original_rows) < len(mirrored_rows):
        pad = boxes_from_pixels(FAR * (len(mirrored_rows) - len(original_rows)))
        original = torch.cat((original, pad), dim=0)
    if len(mirrored_rows) < len(original_rows):
        pad = boxes_from_pixels(FAR * (len(original_rows) - len(mirrored_rows)))
        mirrored = torch.cat((mirrored, pad), dim=0)
    return original, mirrored


def check(name, condition):
    if not condition:
        raise SystemExit("FAILED: %s" % name)
    print("PASS %s" % name)
    return 1


def main():
    passed = 0

    # 1. the mirror is an involution and touches only cx
    sample = boxes_from_pixels([[0, 0, 3, 3], [10, 20, 40, 60], [100, 100, 100.5, 100.5]])
    twice = mirror_boxes(mirror_boxes(sample))
    passed += check("mirror_twice_is_identity", bool(torch.equal(twice, sample)))
    once = mirror_boxes(sample)
    passed += check("mirror_moves_cx_only",
                    bool(torch.equal(once[:, 1:], sample[:, 1:]))
                    and bool(torch.equal(once[:, 0], 1.0 - sample[:, 0])))

    # 2. exact area boundary: 1024 px is outside the small band, 992.25 px is inside
    original, mirrored = two_view([[0, 0, 31.5, 31.5]], [[0, 0, 31.5, 31.5]])
    _f_in, mask_in, counters_in = fuse_view_boxes(original, mirrored, W, H)
    original, mirrored = two_view([[0, 0, 32.0, 32.0]], [[0, 0, 32.0, 32.0]])
    _f_out, mask_out, counters_out = fuse_view_boxes(original, mirrored, W, H)
    passed += check("area_boundary_992_is_treated",
                    bool(mask_in[0]) and counters_in["treated_queries"] == 1)
    passed += check("area_boundary_1024_is_not_treated", not bool(mask_out[0]))
    passed += check("area_reader_is_the_original_box",
                    counters_in["small_band_queries"] == 1 and counters_out["small_band_queries"] == 0)

    # 3. exact threshold boundary: three-by-three boxes offset by one pixel give IoU exactly 0.5
    original, at_threshold = two_view([[0, 0, 3, 3]], [[1, 0, 4, 3]])
    passed += check("constructed_iou_is_exactly_half",
                    float(iou_matrix(boxes_to_pixels(original), boxes_to_pixels(at_threshold))[0, 0])
                    == 0.5)
    _f_half, mask_half, _c_half = fuse_view_boxes(original, at_threshold, W, H)
    _o_below, below = two_view([[0, 0, 3, 3]], [[1.25, 0, 4.25, 3]])   # IoU 5.25/12.75 = 0.4118
    _f_below, mask_below, _c_below = fuse_view_boxes(_o_below, below, W, H)
    passed += check("threshold_is_inclusive_at_half", bool(mask_half[0]))
    passed += check("below_threshold_is_untouched", not bool(mask_below[0]))
    single_original, single_mirrored = boxes_from_pixels([[0, 0, 3, 3]]), \
        boxes_from_pixels([[1, 0, 4, 3]])
    try:
        fuse_view_boxes(single_original, single_mirrored, W, H)
        passed += check("single_mirrored_query_is_refused", False)
    except ValueError:
        passed += check("single_mirrored_query_is_refused", True)

    # 4. the fusion is the arithmetic mean in pixel space
    a, b = two_view([[0, 0, 4, 4]], [[1, 0, 5, 4]])                    # IoU 0.6
    fused, mask, _c = fuse_view_boxes(a, b, W, H)
    expected = boxes_from_pixels([[0.5, 0.0, 4.5, 4.0]])
    passed += check("fusion_is_the_arithmetic_mean",
                    bool(mask[0]) and bool(torch.allclose(fused, expected, atol=1e-6)))

    # 5. untreated rows are bitwise identical to the input, treated rows differ
    many, partner = two_view([[0, 0, 4, 4], [0, 0, 60, 60], [5, 5, 8, 8]],
                            [[1, 0, 5, 4], [0, 0, 61, 61], [200, 200, 201, 201]])
    fused, mask, counters_many = fuse_view_boxes(many, partner, W, H)
    passed += check("untreated_rows_bitwise_identical", bool(torch.equal(fused[~mask], many[~mask])))
    passed += check("large_original_is_never_treated",
                    not bool(mask[1]) and counters_many["treated_queries"] == 1)
    passed += check("treated_row_actually_moved", not bool(torch.equal(fused[0], many[0])))

    # 6. both pairing modes share the treated set exactly and differ only in the partner box
    fused_best, mask_best, counters_best = fuse_view_boxes(many, partner, W, H, mode="best")
    fused_second, mask_second, counters_second = fuse_view_boxes(many, partner, W, H, mode="second")
    moved_best = (fused_best != many).any(dim=1)
    moved_second = (fused_second != many).any(dim=1)
    passed += check("pairing_modes_share_the_treated_set",
                    bool(torch.equal(mask_best, mask_second))
                    and bool(torch.equal(moved_best, moved_second))
                    and counters_best["treated_queries"] == counters_second["treated_queries"])
    passed += check("partner_iou_is_recorded_for_both",
                    len(counters_best["partner_iou_of_treated"]) == counters_best["treated_queries"]
                    and len(counters_second["second_iou_of_treated"]) == counters_second["treated_queries"])
    best_values = torch.tensor(counters_best["partner_iou_of_treated"])
    second_values = torch.tensor(counters_second["second_iou_of_treated"])
    passed += check("second_partner_never_overlaps_more_than_the_best",
                    bool((second_values <= best_values + 1e-6).all()))

    # 7. the second-highest partner is the one used by the sensitivity control
    three, mirrored = two_view([[0, 0, 10, 10]],
                              [[0, 0, 10, 10], [1, 0, 11, 10], [100, 100, 110, 110]])
    best_box = fuse_view_boxes(three, mirrored, W, H, mode="best")[0]
    second_box = fuse_view_boxes(three, mirrored, W, H, mode="second")[0]
    passed += check("best_mode_uses_the_top_partner",
                    bool(torch.allclose(best_box, boxes_from_pixels([[0, 0, 10, 10]]), atol=1e-6)))
    passed += check("second_mode_uses_the_second_partner",
                    bool(torch.allclose(second_box, boxes_from_pixels([[0.5, 0, 10.5, 10]]), atol=1e-6)))

    # 8. determinism, mode coverage and the structural ground-truth freedom
    again = fuse_view_boxes(many, partner, W, H)[0]
    passed += check("deterministic_repeat", bool(torch.equal(again, fused)))
    passed += check("mode_vocabulary", set(PAIRING_MODES) == {"best", "second"})
    parameters = set(inspect.signature(fuse_view_boxes).parameters)
    passed += check("no_ground_truth_parameter",
                    not any(word in name for name in parameters
                            for word in ("gt", "target", "label", "truth")))
    constants = {"NUM_QUERIES": NUM_QUERIES, "SMALL_AREA": SMALL_AREA, "PARTNER_IOU": PARTNER_IOU}
    passed += check("frozen_constants", constants == {"NUM_QUERIES": 300, "SMALL_AREA": 1024.0,
                                                      "PARTNER_IOU": 0.5})
    passed += check("cxcywh_round_trip",
                    bool(torch.allclose(xyxy_to_cxcywh(cxcywh_to_xyxy(sample)), sample, atol=1e-6)))

    print("UNIT_OK %d/%d" % (passed, CHECKS))


if __name__ == "__main__":
    main()
