import sys
from pathlib import Path
import unittest
import subprocess
import tempfile
import hashlib
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from c_stage_core import STAGES, BUCKETS, availability, class_cap, ious


class StageTests(unittest.TestCase):
    def test_each_disjoint_bucket(self):
        for lost in range(7):
            boxes = {s: [[0, 0, 10, 10]] if i < lost else [] for i, s in enumerate(STAGES)}
            r = availability([0, 0, 10, 10], boxes)
            self.assertEqual(r['first_unavailable']['0.75'], BUCKETS[lost])

    def test_threshold_and_empty(self):
        b = {s: [[0, 0, 7.5, 10]] for s in STAGES}
        self.assertEqual(availability([0, 0, 10, 10], b)['first_unavailable']['0.75'], 'available')
        b = {s: [] for s in STAGES}
        self.assertEqual(availability([0, 0, 10, 10], b)['first_unavailable']['0.5'], 'no_raw_geometry')

    def test_non_nested_rejected(self):
        b = {s: [[0, 0, 10, 10]] for s in STAGES}; b['hard_nms'] = []
        with self.assertRaises(ValueError): availability([0, 0, 10, 10], b)

    def test_geometry_rejected(self):
        for b in ([0, 0, 0, 1], [0, 0, float('nan'), 1], [1, 0, 0, 1]):
            with self.assertRaises(ValueError): ious([0, 0, 1, 1], [b])

    def test_continuous_outside_canvas(self):
        self.assertAlmostEqual(float(ious([0, 0, 10, 10], [[-10, 0, 10, 10]])[0]), .5)

    def test_class_cap_preserves_order(self):
        values = [SimpleNamespace(category_id=1, i=0), SimpleNamespace(category_id=2, i=1),
                  SimpleNamespace(category_id=1, i=2)]
        self.assertEqual([v.i for v in class_cap(values, 1)], [0, 1])

    def test_real_cli_closure_duplicate_and_failure(self):
        cli = Path(__file__).resolve().parents[1] / 'tools/c_stage_audit.py'
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'success'
            cmd = [sys.executable, '-B', str(cli), 'fixture', '--output', str(output)]
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertTrue((output / 'CLOSED.json').is_file())
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
            failed = Path(root) / 'failed'
            result = subprocess.run([sys.executable,'-B',str(cli),'fixture','--output',str(failed),'--inject-failure'],capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertTrue((failed / 'STOP.json').is_file())
            self.assertFalse((failed / 'RESULT.json').exists())


if __name__ == '__main__': unittest.main()
