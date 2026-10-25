"""Actual offline CLI acceptance: constructed receipts, never scientific scores."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from h2_paper_fixture import fixture, rewrite, report


class ResultCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h2-paper-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, edit=None, mutate=None, closed=True, synthetic=True, figures=False):
        folder = self.root / "receipts"
        idx, cl = fixture(folder, edit)
        if mutate:
            mutate(folder)
        output = self.root / "tables"
        cmd = [sys.executable, "-B", str(report.Path(report.__file__)), "--index", str(idx), "--index-sha256", report.sha(idx.read_bytes()), "--output", str(output)]
        if closed:
            cmd += ["--closed", str(cl), "--closed-sha256", report.sha(cl.read_bytes())]
        if synthetic:
            cmd += ["--synthetic"]
        if not figures:
            cmd += ["--no-figures"]
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.last_command = cmd; self.last_output = output
        return done, output

    def accepted(self, result):
        done, output = result
        self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
        return json.loads((output / "NORMALIZED.json").read_text(encoding="utf-8"))

    def rejected(self, result, reason):
        done, output = result
        self.assertEqual(done.returncode, 2, done.stderr + done.stdout)
        self.assertIn(reason, done.stderr)
        self.assertFalse(output.exists())

    def test_complete_cli_and_units(self):
        r = self.accepted(self.run_cli())
        self.assertEqual((len(r["absolute"]), len(r["deltas"]), len(r["gates"]), len(r["cost_trials"])), (120, 120, 22, 18))
        self.assertEqual(r["quality"], "NOT_SCIENTIFIC")
        self.assertEqual(r["reported_quality"], "FAIL")
        self.assertTrue(r["evidence_complete"])
        for ev in ("COCO_bridge", "primary"):
            a = next(x for x in r["absolute"] if x["arm"] == "N" and x["budget"] == 600 and x["evaluator"] == ev and x["metric"] == "AP")
            self.assertAlmostEqual(a["value_percent"], 30)
            d = next(x for x in r["deltas"] if x["pair"] == "D-U" and x["budget"] == 600 and x["evaluator"] == ev and x["metric"] == "AP")
            self.assertAlmostEqual(d["delta_pp"], .3)
        self.assertIn("SYNTHETIC TEST DATA", (self.last_output / "TABLES.md").read_text(encoding="utf-8"))
        self.assertIn("SYNTHETIC TEST DATA", (self.last_output / "MAIN_TABLE.tex").read_text(encoding="utf-8"))
        self.assertIn(r"\textbf{SYNTHETIC TEST DATA", (self.last_output / "MAIN_TABLE.tex").read_text(encoding="utf-8"))
        for path in self.last_output.glob("*.csv"):
            self.assertIn("data_kind", path.read_text(encoding="utf-8-sig").splitlines()[0])
            self.assertIn("SYNTHETIC TEST DATA", path.read_text(encoding="utf-8-sig").splitlines()[1])

    def test_real_plot_cli_watermarks_and_sha(self):
        r = self.accepted(self.run_cli(figures=True))
        output = self.last_output
        for name in ("paired_deltas", "cost", "centers"):
            self.assertGreater((output / (name + ".png")).stat().st_size, 1000)
            svg = (output / (name + ".svg")).read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC ONLY", svg)
            self.assertIn("SYNTHETIC TEST DATA", svg)
        index = json.loads((output / "OUTPUT_INDEX.json").read_text())
        for row in index["files"]:
            self.assertEqual(row["sha256"], report.sha((output / row["name"]).read_bytes()))

    def test_rounding_cannot_turn_fail_into_pass(self):
        def edit(o):
            o["scores"]["D"]["600"]["COCO_bridge"]["metrics"]["AP75_small"] = .30399999999999
        r = self.accepted(self.run_cli(edit))
        g = next(x for x in r["gates"] if x["pair"] == "D-U" and x["budget"] == 600 and x["metric"] == "AP75_small")
        self.assertEqual(report.formatted(g["delta_pp"]), "0.300000")
        self.assertEqual(g["recorded_status"], "FAIL")

    def test_undefined_coco_is_unknown(self):
        def edit(o):
            o["scores"]["D"]["600"]["COCO_bridge"]["metrics"]["AP75_small"] = -1
        r = self.accepted(self.run_cli(edit))
        d = next(x for x in r["deltas"] if x["pair"] == "D-U" and x["budget"] == 600 and x["metric"] == "AP75_small")
        self.assertIsNone(d["delta_pp"])
        self.assertFalse(r["evidence_complete"])
        self.assertNotIn("-100.000000", (self.last_output / "TABLES.md").read_text(encoding="utf-8"))

    def test_missing_metric_and_gate_are_unknown(self):
        def edit(o):
            del o["scores"]["D"]["600"]["COCO_bridge"]["metrics"]["APS"]
        r = self.accepted(self.run_cli(edit))
        self.assertEqual(len(r["gates"]), 22)
        self.assertFalse(r["evidence_complete"])
        self.assertTrue(any(x["status"] == "UNKNOWN" for x in r["gates"]))

    def test_missing_closed_is_not_terminal_pass(self):
        r = self.accepted(self.run_cli(closed=False))
        self.assertFalse(r["evidence_complete"])
        self.assertIn("root CLOSED: not supplied separately", r["missing"])

    def test_missing_score_file_is_unknown(self):
        r = self.accepted(self.run_cli(mutate=lambda f: (f / "D-600.json").unlink()))
        self.assertFalse(r["evidence_complete"])
        self.assertTrue(all(x["value_percent"] is None for x in r["absolute"] if x["arm"] == "D" and x["budget"] == 600))

    def test_same_size_corruption_rejected_before_write(self):
        def bad(f):
            p = f / "D-600.json"; raw = p.read_bytes(); p.write_bytes(raw.replace(b"0.304", b"0.404", 1))
        self.rejected(self.run_cli(mutate=bad), "SHA mismatch")

    def test_swapped_arms_rejected(self):
        def bad(f):
            a, b = f / "D-600.json", f / "U-600.json"
            aa, bb = a.read_bytes(), b.read_bytes(); a.write_bytes(bb); b.write_bytes(aa)
        self.rejected(self.run_cli(mutate=bad), "mismatch")

    def test_wrong_score_reference_role_rejected(self):
        def bad(f):
            rewrite(f, "score-RESULT.json", lambda v: v["results"]["D"].update({"600": v["results"]["U"]["600"]}))
        self.rejected(self.run_cli(mutate=bad), "reference path/role mismatch")

    def test_fixture_requires_explicit_mode(self):
        self.rejected(self.run_cli(synthetic=False), "synthetic/production mode mismatch")

    def test_wrong_runtime_commit_rejected(self):
        def bad(f):
            rewrite(f, "contract.json", lambda v: v.update(commit="0" * 40))
        self.rejected(self.run_cli(mutate=bad), "commit mismatch")

    def test_wrong_seed_rejected(self):
        self.rejected(self.run_cli(mutate=lambda f: rewrite(f, "contract.json", lambda v: v.update(seed=1))), "task/scope/seed mismatch")

    def test_embedded_score_disagreement_rejected(self):
        def bad(f):
            rewrite(f, "compare-RESULT.json", lambda v: v["absolute"]["D"]["600"]["official"].update(AP=90))
        self.rejected(self.run_cli(mutate=bad), "embedded score differs")

    def test_duplicate_gate_rejected(self):
        self.rejected(self.run_cli(mutate=lambda f: rewrite(f, "compare-RESULT.json", lambda v: v["quality_gates"].append(v["quality_gates"][0]))), "duplicate quality gate")

    def test_changed_threshold_rejected(self):
        self.rejected(self.run_cli(mutate=lambda f: rewrite(f, "compare-RESULT.json", lambda v: v["quality_gates"][0].update(minimum=-999))), "changed gate threshold")

    def test_missing_gate_is_unknown(self):
        r = self.accepted(self.run_cli(mutate=lambda f: rewrite(f, "compare-RESULT.json", lambda v: v["quality_gates"].pop())))
        self.assertFalse(r["evidence_complete"])
        self.assertEqual(sum(x["recorded_status"] == "UNKNOWN" for x in r["gates"]), 1)

    def test_center_empty_sample(self):
        def edit(o):
            o["centers"]["D"].update(matched_small=0, improved=0, unchanged=0, worsened=0,
                coarse_mean=None, coarse_median=None, final_mean=None, final_median=None)
        r = self.accepted(self.run_cli(edit))
        self.assertEqual(r["centers"][0]["status"], "NOT_RECORDED_HISTORICAL_N")
        self.assertEqual(r["centers"][2]["status"], "EMPTY_SAMPLE")

    def test_center_direction_and_different_denominators_preserved(self):
        r = self.accepted(self.run_cli())
        u, d = r["centers"][1:]
        self.assertNotEqual(u["matched_small"], d["matched_small"])
        self.assertLess(u["mean_change"], 0)
        self.assertGreater(u["final_median"] - u["coarse_median"], 0)

    def test_quality_fail_cost_pass_are_separate(self):
        def edit(o):
            for rr in o["cost"]["measurements"].values():
                for row in rr:
                    row.update(peak_allocated_bytes=4096, peak_reserved_bytes=8192)
        def update_flags(f):
            rewrite(f, "compare-RESULT.json", lambda v: v.update(cost_passed=True))
            rewrite(f, "RESULT.json", lambda v: v.update(cost_passed=True))
        r = self.accepted(self.run_cli(edit=edit, mutate=update_flags))
        self.assertEqual((r["reported_quality"], r["reported_cost"]), ("FAIL", "PASS"))
        self.assertEqual(r["cost"], "NOT_SCIENTIFIC")

    def test_contradictory_cost_decision_rejected(self):
        self.rejected(self.run_cli(mutate=lambda f: rewrite(f, "compare-RESULT.json", lambda v: v.update(cost_passed=True))), "cost decision contradicts")

    def test_contradictory_defined_flag_rejected(self):
        def mutate(f):
            rewrite(f, "compare-RESULT.json", lambda v: v["quality_gates"][0].update(defined=False, passed=False))
        self.rejected(self.run_cli(mutate=mutate), "definition disagrees")

    def test_bad_center_denominator_rejected(self):
        self.rejected(self.run_cli(edit=lambda o: o["centers"]["D"].update(matched_small=20)), "center count denominator")

    def test_wrong_units_rejected(self):
        self.rejected(self.run_cli(edit=lambda o: o["scores"]["D"]["600"]["COCO_bridge"].update(units="percent")), "units/protocol mismatch")

    def test_wrong_closed_index_rejected(self):
        def mutate(f):
            p = f.parent / (f.name + "-ROOT_CLOSED.json")
            v = json.loads(p.read_text()); v["export_index"]["sha256"] = "0" * 64
            p.write_text(json.dumps(v))
        self.rejected(self.run_cli(mutate=mutate), "CLOSED export index mismatch")

    def test_near_boundary_gate_copy_must_be_exact(self):
        def edit(o):
            o["scores"]["D"]["600"]["COCO_bridge"]["metrics"]["AP75_small"] = .30399999999999
        def mutate(f):
            def change(v):
                g = next(g for g in v["quality_gates"] if g["pair"] == "D-U" and g["budget"] == 600 and g["metric"] == "AP75_small")
                g.update(delta_pp=.3, passed=True)
            rewrite(f, "compare-RESULT.json", change)
        self.rejected(self.run_cli(edit=edit, mutate=mutate), "exact duplicate mismatch")

    def test_exponent_overflow_rejected_before_write(self):
        def bad(f):
            p = f / "SHA_INDEX.json"; text = p.read_text(); p.write_text(text.replace('"status": "PASS"', '"extra": 1e999, "status": "PASS"', 1))
        self.rejected(self.run_cli(mutate=bad), "non-finite JSON float")

    def test_stop_is_refused(self):
        self.rejected(self.run_cli(mutate=lambda f: (f / "FINALIZATION_STOP.json").write_text("{}")), "STOP file present")

    def test_existing_output_not_overwritten(self):
        self.accepted(self.run_cli())
        before = (self.last_output / "OUTPUT_INDEX.json").read_bytes()
        p = subprocess.run(self.last_command, capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(p.returncode, 2)
        self.assertEqual(before, (self.last_output / "OUTPUT_INDEX.json").read_bytes())

    def test_duplicate_index_name_refused(self):
        def bad(f):
            p = f / "SHA_INDEX.json"; v = json.loads(p.read_text()); v["files"].append(v["files"][0]); p.write_text(json.dumps(v))
        self.rejected(self.run_cli(mutate=bad), "duplicate export name")

    def test_nan_refused_even_with_valid_sha(self):
        def bad(f):
            rewrite(f, "D-600.json", lambda v: v["official"].update(AP=float("nan")))
        self.rejected(self.run_cli(mutate=bad), "non-finite JSON constant")


if __name__ == "__main__":
    unittest.main()
