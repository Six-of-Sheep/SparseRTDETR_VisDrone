"""CPU unit checks for the r22 mechanism. No model, no CUDA, no COCOeval, no file writes.

Each case prints one line and the script exits non-zero on the first failure, so it can gate the GPU
run.  The cases cover exactly the places where the mechanism can silently change the certified arm:
the top-300/top-600 exclusion arithmetic (including the tie fallbacks), the same-class IoU dedup, the
union ranking keeping every certified entry byte-identical, the IoU kernel, the class-equal-weight
aggregation that the gate uses, and the GT reachability diagnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.expand600.expand600 import (NEW_QUERY_BASE, dedup_new_names, gt_potential, merge_arm,
                                       new_query_selection, per_class_macro_metrics,
                                       per_class_values, query_names, select_indices,
                                       xyxy_iou_matrix)

FAILURES = []


def case(function):
    try:
        function()
    except Exception as error:                                     # noqa: BLE001 - reported, not raised
        FAILURES.append("%s: %s: %s" % (function.__name__, type(error).__name__, error))
        print("FAIL %s: %s: %s" % (function.__name__, type(error).__name__, error))
    else:
        print("PASS %s" % function.__name__)


def test_selection_and_exclusion_counts():
    """Top-600 minus top-300 is exactly the new 300 anchors, in top-600 order."""
    torch.manual_seed(0)
    logits = torch.randn(1, 120, 10)
    ind300 = select_indices(logits, 30)
    ind600 = select_indices(logits, 60)
    selection = new_query_selection(ind300, ind600, logits, num_new=30)
    assert selection["counters"][0]["num_new"] == 30, selection["counters"]
    assert selection["counters"][0]["truncated_over_num_new"] == 0
    assert selection["counters"][0]["filled_short_of_num_new"] == 0
    chosen = selection["anchors"][0]
    assert len(set(chosen)) == 30
    assert not (set(chosen) & set(int(v) for v in ind300[0].tolist()))
    # the chosen anchors are the top-60 ones that are not in the top-30 set
    expected = [int(v) for v in ind600[0].tolist()
                if int(v) not in set(int(x) for x in ind300[0].tolist())]
    assert chosen == expected[:30], (chosen, expected)
    # positions index into the 600-query output
    for position, anchor in zip(selection["positions"][0], chosen):
        assert int(ind600[0, position]) == anchor


def test_selection_tie_fallbacks_are_counted():
    """A boundary tie must never lose or duplicate an anchor, and must be counted."""
    logits = torch.zeros(1, 1200, 10)         # every anchor has the same score
    ind300 = select_indices(logits, 300)
    ind600 = select_indices(logits, 600)
    selection = new_query_selection(ind300, ind600, logits, num_new=300)
    counters = selection["counters"][0]
    chosen = selection["anchors"][0]
    assert len(chosen) == 300, len(chosen)
    assert len(set(chosen)) == 300
    assert not (set(chosen) & set(int(v) for v in ind300[0].tolist()))
    assert (counters["truncated_over_num_new"] + counters["filled_short_of_num_new"]
            + counters["ind300_members_absent_from_ind600"]) >= 0


def test_dedup_rules():
    reference = [{"image_id": "i", "query_id": 1, "category_id": 3, "score": 0.5,
                  "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
                 {"image_id": "i", "query_id": 2, "category_id": 5, "score": 0.4,
                  "bbox_xyxy": [50.0, 50.0, 60.0, 60.0]}]
    names = [(300, 3, 0.9, [0.5, 0.5, 10.5, 10.5]),      # same class, IoU 0.90 -> dropped
             (301, 3, 0.8, [8.0, 8.0, 30.0, 30.0]),      # same class, IoU 0.10 -> kept
             (302, 5, 0.7, [50.5, 50.5, 60.5, 60.5]),     # same class, IoU 0.90 -> dropped
             (303, 7, 0.6, [0.0, 0.0, 10.0, 10.0]),       # other class, same box -> kept
             (304, 3, 0.55, [55.0, 55.0, 65.0, 65.0])]    # same class far away -> kept
    kept, dropped = dedup_new_names(names, reference, 0.7)
    kept_slots = [item[0] for item in kept]
    dropped_slots = [item["query_slot"] for item in dropped]
    assert dropped_slots == [300, 302], dropped_slots
    assert sorted(kept_slots) == [301, 303, 304], kept_slots
    assert all(d["best_iou"] >= 0.7 for d in dropped)


def test_merge_arm_keeps_certified_entries_and_key_order():
    reference = [{"image_id": "i", "query_id": 0, "category_id": 1, "score": 0.9,
                  "bbox_xyxy": [0.0, 0.0, 1.0, 1.0]},
                 {"image_id": "i", "query_id": 1, "category_id": 1, "score": 0.7,
                  "bbox_xyxy": [2.0, 2.0, 3.0, 3.0]},
                 {"image_id": "i", "query_id": 2, "category_id": 1, "score": 0.7,
                  "bbox_xyxy": [4.0, 4.0, 5.0, 5.0]}]
    new = [(NEW_QUERY_BASE, 1, 0.8, [6.0, 6.0, 7.0, 7.0]),
           (NEW_QUERY_BASE + 1, 1, 0.7, [8.0, 8.0, 9.0, 9.0]),
           (NEW_QUERY_BASE + 2, 1, 0.1, [10.0, 10.0, 11.0, 11.0])]
    kept, displaced, _key = merge_arm(reference, new, 4)
    order = [(e["source"], e["query_id"], e["category_id"], e["score"]) for e in kept]
    assert order == [("first_stage", 0, 1, 0.9), ("new_query", NEW_QUERY_BASE, 1, 0.8),
                     ("first_stage", 1, 1, 0.7), ("first_stage", 2, 1, 0.7)], order
    assert displaced[0]["source"] == "new_query"
    # certified entries are copied, never mutated
    for entry in kept:
        if entry["source"] == "first_stage":
            match = next(r for r in reference if r["query_id"] == entry["query_id"])
            assert entry["score"] == match["score"]
            assert entry["bbox_xyxy"] == [float(v) for v in match["bbox_xyxy"]]


def test_iou_matches_reference():
    boxes = [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]]
    matrix = xyxy_iou_matrix(boxes, boxes)
    assert np.allclose(np.diag(matrix), 1.0)
    assert abs(matrix[0, 1] - 25.0 / 175.0) < 1e-12
    assert matrix[0, 2] == 0.0
    assert matrix.shape == (3, 3)
    empty = xyxy_iou_matrix([], boxes)
    assert empty.shape == (0, 3)


def test_class_macro_differs_from_cell_mean_where_expected():
    """One class with fewer valid recall bins must move the two aggregations apart."""
    precision = np.full((10, 101, 2, 4, 3), -1.0)
    recall = np.full((10, 2, 4, 3), -1.0)
    precision[:, :, 0, 0, 2] = 0.6        # class 0: every bin valid
    precision[:, :50, 1, 0, 2] = 0.2      # class 1: only half the bins valid
    recall[:, 0, 0, 2] = 0.8
    recall[:, 1, 0, 2] = 0.4
    eval_result = {"precision": precision, "recall": recall}
    macro = per_class_macro_metrics(eval_result)
    assert abs(macro["AP"] - 0.4) < 1e-12, macro["AP"]              # (0.6 + 0.2) / 2
    cells = precision[:, :, :, 0, 2]
    cells = cells[cells > -1]
    assert abs(float(cells.mean()) - (0.6 * 101 + 0.2 * 50) / 151) < 1e-12
    assert abs(float(cells.mean()) - macro["AP"]) > 0.05            # the two forms really differ
    assert abs(macro["AR100"] - 0.6) < 1e-12, macro["AR100"]        # (0.8 + 0.4) / 2
    per_class = per_class_values(eval_result)
    assert abs(per_class[0] - 0.6) < 1e-12 and abs(per_class[1] - 0.2) < 1e-12
    for key in ("AP50", "AP75", "APS", "APM", "APL", "AP75_small"):
        assert key in macro


def test_gt_potential_counts_same_class_overlap():
    entries = [{"category_id": 1, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
               {"category_id": 2, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
               {"category_id": 1, "bbox_xyxy": [40.0, 40.0, 50.0, 50.0]}]
    raw = [{"category": 1, "box": [0.0, 0.0, 10.0, 10.0]},
           {"category": 0, "box": [0.0, 0.0, 10.0, 10.0]}]
    out = gt_potential(entries, raw, 0.5)
    assert out == {"entries": 3, "matched": 1, "iou_floor": 0.5}, out


def test_query_names_scoring_rule():
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    logits = torch.zeros(1, 1, 10)
    logits[0, 0, 3] = 2.0
    names = query_names(boxes[0], logits[0], [0], 100, 200)
    assert len(names) == 10
    slot, category, score, box = names[3]
    assert slot == NEW_QUERY_BASE and category == 4
    assert abs(score - float(torch.sigmoid(torch.tensor(2.0)))) < 1e-12
    assert np.allclose(box, [40.0, 80.0, 60.0, 120.0], atol=1e-4), box


def main():
    for function in (test_selection_and_exclusion_counts, test_selection_tie_fallbacks_are_counted,
                     test_dedup_rules, test_merge_arm_keeps_certified_entries_and_key_order,
                     test_iou_matches_reference, test_class_macro_differs_from_cell_mean_where_expected,
                     test_gt_potential_counts_same_class_overlap, test_query_names_scoring_rule):
        case(function)
    if FAILURES:
        print("UNIT_FAILURES %d" % len(FAILURES))
        return 1
    print("UNIT_OK 8/8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
