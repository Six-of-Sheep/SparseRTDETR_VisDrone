"""SYNTHETIC_TEST_ONLY: small schemas; never detector/AP/GPU tests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools/paper_sampling_mechanism.py"
SPEC = importlib.util.spec_from_file_location("paper_sampling_mechanism", TOOL)
DISPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISPLAY)


def raw_fixture():
    return {"data_kind": "SYNTHETIC_TEST_ONLY", "observations": [
        {"arm": "P", "layer": layer, "group": group, "uses_GT": False,
         "reference_instances": 300 if group == "ordinary" else 20,
         "reference_below_cell_any": 0 if group == "ordinary" else 10,
         "reference_floor_trigger_fraction_any": 0 if group == "ordinary" else .5,
         "sampled_queries_per_image": 8, "DN_includes_native_padding": group == "DN",
         "actual_span_cells_x": {"count": 16, "mean": .25 + layer, "min": 0,
                                 "p50": .2 + layer, "p90": .3 + layer, "max": .5 + layer}}
        for layer in (0, 1) for group in ("ordinary", "DN")
    ]}


class MechanismDisplayTests(unittest.TestCase):
    def test_raw_quantiles_are_kept_per_record(self):
        counts, rows, _ = DISPLAY.normalize(raw_fixture(), scope="test-fixture")
        self.assertEqual(len(counts), 4)
        q = [r for r in rows if r["statistic"] == "p90"]
        self.assertEqual(len(q), 4)
        self.assertEqual(len({r["record"] for r in q}), 4)
        self.assertFalse(any("pooled" in r["statistic"] for r in rows))

    def test_summary_record_means_not_averaged(self):
        payload = {"accept-P": {"DN": {
            "records": 2, "reference_instances": 30, "below_cell_any": 15,
            "reference_floor_trigger_fraction_any": .5, "floor_applied_instances": 15,
            "sampled_actual_span_cells_x_mean_by_record": [1, 10]}}}
        counts, rows, _ = DISPLAY.normalize(payload, scope="gpu-acceptance-only")
        means = [r for r in rows if r["statistic"] == "record_mean"]
        self.assertEqual([r["value"] for r in means], [1, 10])
        self.assertEqual([r["count"] for r in means], ["UNKNOWN", "UNKNOWN"])
        self.assertEqual(counts[0]["DN_includes_native_padding"], "UNKNOWN")

    def test_existing_weighted_summary_is_copied(self):
        payload = {"accept-P": {"ordinary": {
            "actual_span_cells_x": {"count": 32, "weighted_mean": .375}}}}
        counts, rows, _ = DISPLAY.normalize(payload, scope="gpu-acceptance-only")
        exact = [r for r in rows if r["metric"] == "actual_span_cells_x"]
        self.assertEqual(exact[0]["value"], .375)
        self.assertEqual(exact[0]["count"], 32)
        self.assertEqual(counts[0]["reference_instances"], "UNKNOWN")

    def test_synthetic_cannot_enter_real_scope(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            DISPLAY.normalize(raw_fixture(), scope="selected-training-windows")

    def test_acceptance_cannot_enter_training_scope(self):
        with self.assertRaisesRegex(ValueError, "relabelled"):
            DISPLAY.normalize({"accept-B0": {"ordinary": {"records": 0}}}, scope="selected-training-windows")

    def test_bad_reference_count_fraction_rejected(self):
        payload = raw_fixture()
        payload["observations"][0]["reference_floor_trigger_fraction_any"] = .9
        with self.assertRaisesRegex(ValueError, "conflicts"):
            DISPLAY.normalize(payload, scope="test-fixture")

    def test_gt_wrong_group_or_large_sample_rejected(self):
        for key, value in (("uses_GT", True), ("group", "pooled"), ("sampled_queries_per_image", 9)):
            with self.subTest(key=key):
                payload = raw_fixture()
                payload["observations"][0][key] = value
                with self.assertRaises(ValueError):
                    DISPLAY.normalize(payload, scope="test-fixture")

    def test_missing_measurements_are_unknown(self):
        _, rows, _ = DISPLAY.normalize(raw_fixture(), scope="test-fixture")
        missing = [r for r in rows if r["metric"] == "bilinear_effective_weight_abs_change"]
        self.assertTrue(missing)
        self.assertTrue(all(r["value"] == "UNKNOWN" for r in missing))

    def test_sha_rejects_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.json"
            p.write_text(json.dumps(raw_fixture()), encoding="utf-8")
            out = Path(tmp) / "rejected"
            result = subprocess.run([sys.executable, "-B", str(TOOL), "--input", str(p),
                                     "--sha256", "0" * 64, "--scope", "test-fixture", "--output", str(out)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("SHA256 mismatch", result.stderr)
            self.assertFalse(out.exists())

    def test_cli_writes_labelled_tables_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.json"
            p.write_text(json.dumps(raw_fixture()), encoding="utf-8")
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            out = Path(tmp) / "display"
            result = subprocess.run([sys.executable, "-B", str(TOOL), "--input", str(p),
                                     "--sha256", digest, "--scope", "test-fixture", "--output", str(out)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = (out / "MECHANISM_DISPLAY.md").read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC_TEST_ONLY", report)
            self.assertIn("UNKNOWN", report)
            self.assertTrue((out / "measurements.csv").is_file())
            provenance = json.loads((out / "DISPLAY_PROVENANCE.json").read_text())
            self.assertFalse(provenance["scientific_quality_evaluated"])
            self.assertEqual(provenance["sources"][0]["sha256"], digest)
            self.assertNotIn("torch", sys.modules)

    def test_effect_identity_and_pair_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "main.json"
            p.write_text(json.dumps(raw_fixture()), encoding="utf-8")
            e = Path(tmp) / "effects.json"
            e.write_text(json.dumps({"B0": {"ordinary": {}}}), encoding="utf-8")
            args = argparse.Namespace(input=p, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                      effects=e, effects_sha256=hashlib.sha256(e.read_bytes()).hexdigest(),
                                      output=Path(tmp) / "out", scope="test-fixture", plots=False)
            with self.assertRaisesRegex(ValueError, "identity"):
                DISPLAY.run(args)
            self.assertFalse(args.output.exists())
            args.effects_sha256 = None
            with self.assertRaisesRegex(ValueError, "required together"):
                DISPLAY.run(args)

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nonfinite.json"
            p.write_text('{"x":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                DISPLAY.checked_json(p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
