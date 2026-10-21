"""Scalar geometry and provenance checks; no detector/torch imports."""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / "tools/paper_sampling_diagram.py"
SPEC = importlib.util.spec_from_file_location("paper_sampling_diagram", PATH)
diagram = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagram)


class GeometryTests(unittest.TestCase):
    def test_feature_coordinates_align_corners_false(self):
        self.assertEqual(diagram.normalized_to_feature((0, 0), (6, 8)), (-.5, -.5))
        self.assertEqual(diagram.normalized_to_feature((1, 1), (6, 8)), (7.5, 5.5))
        self.assertEqual(diagram.normalized_to_feature((.5, .5), (6, 8)), (3.5, 2.5))

    def test_finest_only_floor_and_unchanged_reference(self):
        reference = (.02, .03)
        self.assertEqual(diagram.sampling_extent("P", reference, (6, 8), 0), (.125, 1/6))
        for arm, level in (("B0", 0), ("B1", 0), ("P", 1), ("P", 2)):
            self.assertEqual(diagram.sampling_extent(arm, reference, (6, 8), level), reference)
        self.assertEqual(reference, (.02, .03))
        self.assertEqual(diagram.sampling_extent("P", (.2, .3), (6, 8), 0), (.2, .3))

    def test_actual_offset_scaling_and_no_clipping(self):
        point = diagram.sampling_points((.9, .9), (.8, .8), [(2, -2)], 1)[0]
        self.assertAlmostEqual(point[0], 1.7)
        self.assertAlmostEqual(point[1], .1)

    def test_native_initial_bias_point_count_relation(self):
        for n in (2, 4, 8):
            offsets = diagram.native_radial_offsets(n, head=0)
            self.assertEqual(offsets, tuple((float(k), 0.0) for k in range(1, n+1)))
            points = diagram.sampling_points((.5, .5), (.2, .2), offsets, n)
            self.assertAlmostEqual(points[-1][0], .6)
        self.assertAlmostEqual(diagram.native_radial_offsets(4)[-1][0], 4)
        self.assertAlmostEqual(diagram.native_radial_offsets(4)[-1][1], 4)

    def test_bilinear_cell_center_and_weight_sum(self):
        rows = diagram.bilinear_neighbors((3.5/8, 2.5/6), (6, 8))
        self.assertEqual((rows[0]["u"], rows[0]["v"]), (3, 2))
        self.assertEqual([r["effective_weight"] for r in rows], [1, 0, 0, 0])
        rows = diagram.bilinear_neighbors((3.75/8, 2.75/6), (6, 8))
        self.assertEqual([r["raw_weight"] for r in rows], [.5625, .1875, .1875, .0625])
        self.assertEqual(sum(r["effective_weight"] for r in rows), 1)

    def test_zero_padding_keeps_coordinate_and_removes_invalid_weight(self):
        rows = diagram.bilinear_neighbors((0, .5), (6, 8))
        self.assertEqual({r["u"] for r in rows}, {-1, 0})
        self.assertAlmostEqual(sum(r["effective_weight"] for r in rows), .5)
        self.assertEqual(sum(r["raw_weight"] for r in rows), 1)
        rows = diagram.bilinear_neighbors((-1, -1), (6, 8))
        self.assertTrue(all(r["effective_weight"] == 0 for r in rows))

    def test_zero_offsets_disprove_span_lower_bound(self):
        extent = diagram.sampling_extent("P", (.02, .03), (6, 8), 0)
        self.assertEqual((extent[0]*8, extent[1]*6), (1, 1))
        points = diagram.sampling_points((.5, .5), extent, [(0, 0)]*4, 4)
        self.assertEqual(diagram.actual_span(points), (0, 0))

    def test_same_stencil_different_weights(self):
        example = diagram.example()
        p = example["bilinear_example_P_k2"]["neighbors"]
        b = example["same_stencil_baseline_B0_k2"]["neighbors"]
        self.assertEqual([(r["u"], r["v"]) for r in p], [(r["u"], r["v"]) for r in b])
        self.assertNotEqual([r["effective_weight"] for r in p], [r["effective_weight"] for r in b])

    def test_explicit_constructed_status_and_no_quality_values(self):
        example = diagram.example()
        self.assertIn("NOT MODEL OBSERVATIONS", example["status"])
        self.assertEqual(set(example["arms"]), {"B0", "B1", "P"})
        self.assertEqual(example["zero_offset_counterexample"]["actual_span_cells_uv"], (0, 0))
        self.assertEqual(example["arms"]["B1"]["levels"][0]["uniform_initial_attention_mass"], 2/3)
        self.assertNotIn("AP", example)

    def test_rejects_invalid_inputs(self):
        for point in ((math.nan, 0), (0, math.inf), (0,), (True, 0)):
            with self.assertRaises(ValueError):
                diagram.normalized_to_feature(point, (6, 8))
        for shape in ((0, 8), (6.0, 8), (True, 8)):
            with self.assertRaises(ValueError):
                diagram.normalized_to_feature((0, 0), shape)
        with self.assertRaises(ValueError):
            diagram.sampling_extent("P", (0, .1), (6, 8), 0)
        with self.assertRaises(ValueError):
            diagram.sampling_points((0, 0), (.1, .1), [(0, 0)], 4)

    def test_import_does_not_load_model_or_plotting_dependencies(self):
        code = ("import importlib.util,sys; "
                f"s=importlib.util.spec_from_file_location('diagram',{str(PATH)!r}); "
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.example();"
                "assert 'torch' not in sys.modules;assert 'matplotlib' not in sys.modules")
        subprocess.run([sys.executable, "-B", "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
