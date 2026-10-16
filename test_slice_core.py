"""Synthetic CPU-only contract checks. Run: python -B test_slice_core.py."""

import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

from slice_core import (
    Candidate, Tile, box_iou64, classwise_nms, deserialize_candidates,
    map_crop_xyxy, merge_candidates, select_view_candidates,
    serialize_candidates, tile_geometry, tile_windows,
)


def cand(query=0, category=1, view=0, score=0.9, box=(0, 0, 2, 2), rank=1,
         shape=(300, 10)):
    return Candidate(tuple(box), score, view, query, category, rank, shape)


class SliceContractTests(unittest.TestCase):
    def test_exact_integer_ceil_and_order(self):
        tiles = tile_windows(1080, 1920)
        self.assertEqual([t.view_id for t in tiles], [1, 2, 3, 4])
        self.assertEqual([t.crop_box for t in tiles], [
            (0, 0, 1067, 600), (853, 0, 1920, 600),
            (0, 480, 1067, 1080), (853, 480, 1920, 1080)])
        self.assertEqual(tile_windows(9, 9)[0].crop_box, (0, 0, 5, 5))

    def test_integer_edges_cover_entire_image_no_padding(self):
        for height, width in ((2, 3), (3, 2), (11, 17), (17, 11), (768, 1344)):
            tiles = tile_windows(height, width)
            self.assertEqual(tiles[0].crop_box[:2], (0, 0))
            self.assertEqual(tiles[-1].crop_box[2:], (width, height))
            self.assertTrue(all(0 <= t.x0 < t.x1 <= width and
                                0 <= t.y0 < t.y1 <= height for t in tiles))
            for y in range(height):
                self.assertTrue(any(t.y0 <= y < t.y1 for t in tiles))
            for x in range(width):
                self.assertTrue(any(t.x0 <= x < t.x1 for t in tiles))

    def test_tiny_duplicate_geometry_keeps_four_view_identities(self):
        tiles = tile_windows(1, 1)
        self.assertEqual(len(tiles), 4)
        self.assertEqual({t.crop_box for t in tiles}, {(0, 0, 1, 1)})
        self.assertEqual([t.view_id for t in tiles], [1, 2, 3, 4])

    def test_overlap_reports_actual_not_assumed_fraction(self):
        exact = tile_geometry(9, 9)
        self.assertEqual(exact["overlap_height_px"], 1)
        self.assertEqual(exact["overlap_height_fraction_of_tile"], 0.2)
        rounded = tile_geometry(1080, 1920)
        self.assertEqual(rounded["overlap_width_px"], 214)
        self.assertNotEqual(rounded["overlap_width_fraction_of_tile"], 0.2)

    def test_dimension_and_tile_rejection(self):
        for args in ((0, 1), (-1, 2), (1.0, 2), (True, 2)):
            with self.assertRaises(ValueError):
                tile_windows(*args)
        with self.assertRaises(ValueError):
            Tile(1, 0, 0, 10, 10, 9, 10)

    def test_crop_mapping_adds_origin_without_clipping(self):
        tile = tile_windows(9, 9)[-1]
        self.assertEqual(map_crop_xyxy([(-2, -1, 6, 7)], tile), [(2, 3, 10, 11)])
        self.assertEqual(map_crop_xyxy([], tile), [])

    def test_mapping_validation(self):
        tile = tile_windows(9, 9)[0]
        for box in ((0, 0, 0, 1), (0, 0, 1, math.nan), (0, 0, math.inf, 1)):
            with self.assertRaises(ValueError):
                map_crop_xyxy([box], tile)

    def test_candidate_topk_category_is_one_based_and_ties_stable(self):
        selected = select_view_candidates(
            [(0, 0, 1, 1), (1, 1, 2, 2)], [[0.5, 0.5], [0.5, 0.6]],
            view_id=3, limit=3, expected_shape=(2, 2))
        self.assertEqual([c.identity for c in selected], [(3, 1, 2), (3, 0, 1), (3, 0, 2)])
        self.assertEqual([c.view_candidate_rank for c in selected], [1, 2, 3])
        self.assertTrue(all(c.input_shape == (2, 2) for c in selected))

    def test_original_q300xc10_budget600_and_zero_scores_retained(self):
        boxes = [(q * 2, 0, q * 2 + 1, 1) for q in range(300)]
        selected = select_view_candidates(boxes, [[0.0] * 10 for _ in range(300)], view_id=0)
        self.assertEqual(len(selected), 600)
        self.assertEqual(selected[0].identity, (0, 0, 1))
        self.assertEqual(selected[-1].identity, (0, 59, 10))

    def test_all_input_values_validated_even_outside_budget(self):
        boxes = [(0, 0, 1, 1), (0, 0, 2, 2)]
        for scores in ([[0.9, 0.8], [0.0, math.nan]],
                       [[0.9, 0.8], [0.0, -0.1]]):
            with self.assertRaises(ValueError):
                select_view_candidates(boxes, scores, view_id=0, limit=1, expected_shape=(2, 2))
        with self.assertRaises(ValueError):
            select_view_candidates(boxes, [[0.9], [0.1]], view_id=0, expected_shape=(2, 2))
        with self.assertRaises(ValueError):
            select_view_candidates([(0, 0, 1, 1)], [[0.9, 0.8]], view_id=0)

    def test_iou_half_exactly_is_not_suppressed(self):
        first = cand(query=0, box=(0, 0, 3, 2))
        second = cand(query=1, box=(1, 0, 4, 2), score=0.8)
        self.assertEqual(box_iou64(first.box_xyxy, second.box_xyxy), 0.5)
        self.assertEqual(len(classwise_nms([first, second])), 2)
        above = cand(query=2, box=(0.9, 0, 3.9, 2), score=0.7)
        self.assertGreater(box_iou64(first.box_xyxy, above.box_xyxy), 0.5)
        self.assertEqual(classwise_nms([first, above]), [first])

    def test_classwise_only_not_cross_class(self):
        first, second = cand(category=1), cand(category=2)
        self.assertEqual(len(classwise_nms([first, second])), 2)

    def test_global_nms_ties_order_view_query_category(self):
        items = [cand(view=4, query=0), cand(view=0, query=2),
                 cand(view=0, query=1), cand(view=1, query=0)]
        self.assertEqual(classwise_nms(items), [items[2]])
        nonoverlap = [cand(query=1, category=2, box=(10, 0, 11, 1)),
                      cand(query=1, category=1, box=(0, 0, 1, 1)),
                      cand(view=1, query=0, box=(20, 0, 21, 1))]
        self.assertEqual([x.identity for x in classwise_nms(nonoverlap)],
                         [(0, 1, 1), (0, 1, 2), (1, 0, 1)])

    def test_same_query_id_in_different_views_is_not_correspondence(self):
        a = cand(view=0, query=7, box=(0, 0, 1, 1))
        b = cand(view=1, query=7, box=(10, 10, 11, 11))
        self.assertEqual(len(classwise_nms([a, b])), 2)
        # Different query IDs are nevertheless duplicates if class/geometry overlap.
        c = cand(view=2, query=19, box=(0, 0, 1, 1), score=0.8)
        self.assertEqual(classwise_nms([a, b, c]), [a, b])

    def test_duplicate_identity_and_bad_metadata_refused(self):
        with self.assertRaises(ValueError):
            merge_candidates([cand(), cand(box=(20, 20, 21, 21))])
        for kwargs in ({"category": 0}, {"category": 11}, {"view": 5},
                       {"query": 300}, {"score": math.inf}, {"box": (0, 0, 0, 1)}):
            with self.assertRaises(ValueError):
                cand(**kwargs)
        with self.assertRaises(ValueError):
            merge_candidates([cand(), cand(view=1, shape=(301, 10))])

    def test_final_budgets_global_not_per_class_no_padding(self):
        items = [cand(query=q % 300, view=q // 300, score=0.5,
                      box=(q * 2, 0, q * 2 + 1, 1)) for q in range(700)]
        result = merge_candidates(reversed(items))
        self.assertEqual(result["candidate_count"], 700)
        self.assertEqual(result["kept_count"], 700)
        self.assertEqual(result["top600"], items[:600])
        self.assertEqual(result["top300"], items[:300])
        self.assertEqual(merge_candidates(items[:2])["top600"], items[:2])

    def test_empty_merge_and_serialization(self):
        result = merge_candidates([])
        self.assertEqual(result["top600"], [])
        self.assertEqual(result["top300"], [])
        self.assertEqual(result["suppressed_count"], 0)
        self.assertEqual(deserialize_candidates([]), [])

    def test_json_round_trip_preserves_metadata_scores_and_boxes(self):
        items = [cand(view=1, query=17, category=10, score=0.3125,
                      box=(-10.5, 2, 30, 40.125), rank=18)]
        records = json.loads(json.dumps(serialize_candidates(items), allow_nan=False))
        self.assertEqual(records[0]["input_shape"], [300, 10])
        self.assertEqual(deserialize_candidates(records), items)

    def test_deterministic_permutation_and_no_input_mutation(self):
        items = [cand(query=q, box=(q * 2, 0, q * 2 + 1, 1)) for q in range(10)]
        expected = serialize_candidates(merge_candidates(items)["top600"])
        rng = random.Random(0)
        shuffled = list(items)
        rng.shuffle(shuffled)
        before = serialize_candidates(shuffled)
        self.assertEqual(serialize_candidates(merge_candidates(shuffled)["top600"]), expected)
        self.assertEqual(serialize_candidates(shuffled), before)

    def test_import_has_no_cuda_torch_or_rng_side_effect(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import random,sys; b=random.getstate(); import slice_core; "
             "assert random.getstate()==b; assert 'torch' not in sys.modules; print('PASS')"],
            cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "PASS")

    def test_vectorized_nms_matches_scalar_fixed_case(self):
        rng = random.Random(20261016)
        items = []
        for index in range(120):
            x, y = rng.uniform(-20, 20), rng.uniform(-20, 20)
            width, height = rng.uniform(1, 12), rng.uniform(1, 12)
            items.append(cand(query=index, category=index % 3 + 1,
                              score=(index % 7) / 7,
                              box=(x, y, x + width, y + height)))
        scalar = []
        for item in sorted(items, key=lambda value: value.sort_key):
            if not any(previous.category_id == item.category_id and
                       box_iou64(previous.box_xyxy, item.box_xyxy) > 0.5
                       for previous in scalar):
                scalar.append(item)
        self.assertEqual(classwise_nms(items), scalar)


if __name__ == "__main__":
    unittest.main(verbosity=2)
