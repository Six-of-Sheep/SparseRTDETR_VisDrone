import hashlib
import random
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from slice_core import Candidate, classwise_nms, NMS_IOU, box_iou64  # noqa: E402
from c_stage_core import ious  # noqa: E402
from nms_attribution import (classwise_nms_with_suppressor, sort_key, fate, argmax_ties,  # noqa: E402
                             category, shared_counts, attribute_gt)


def cand(box, score, view, query, cat, rank=1):
    return Candidate(tuple(float(v) for v in box), float(score), view, query, cat, rank)


def random_pool(seed, n=400):
    rng = random.Random(seed)
    pool, used = [], set()
    while len(pool) < n:
        view, query, cat = rng.randrange(5), rng.randrange(300), rng.randrange(1, 4)
        if (view, query, cat) in used:
            continue
        used.add((view, query, cat))
        cx, cy = rng.uniform(0, 120), rng.uniform(0, 120)
        w, h = rng.uniform(4, 30), rng.uniform(4, 30)
        score = rng.choice([.9, .5, .25]) if rng.random() < .3 else rng.random()
        pool.append(cand((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), score, view, query, cat))
    return pool


class MirrorTests(unittest.TestCase):
    def test_mirror_equals_frozen_and_suppressor_invariants(self):
        for seed in range(6):
            pool = random_pool(seed)
            kept, suppressed, stats = classwise_nms_with_suppressor(pool, NMS_IOU)
            self.assertEqual([c.identity for c in kept], [c.identity for c in classwise_nms(pool)])
            self.assertEqual(len(kept) + len(suppressed), len(pool))
            self.assertGreater(len(suppressed), 0)
            by_id = {c.identity: c for c in pool}
            kept_ids = {c.identity for c in kept}
            for removed, entry in suppressed.items():
                pivot, c = by_id[entry['suppressor']], by_id[removed]
                self.assertIn(entry['suppressor'], kept_ids)
                self.assertNotIn(removed, kept_ids)
                self.assertEqual(pivot.category_id, c.category_id)
                self.assertGreater(entry['iou'], NMS_IOU)
                self.assertAlmostEqual(entry['iou'], box_iou64(pivot.box_xyxy, c.box_xyxy), places=12)
                self.assertLess(sort_key(pivot), sort_key(c))
                for q in kept:
                    if q.category_id == c.category_id and sort_key(q) < sort_key(pivot):
                        self.assertLessEqual(box_iou64(q.box_xyxy, c.box_xyxy), NMS_IOU)
            self.assertEqual(stats['nms_iou'], NMS_IOU)
            self.assertGreater(stats['comparisons'], 0)

    def test_exact_half_retained_and_boundary_counted(self):
        a, b = cand((0, 0, 10, 10), .9, 0, 0, 1), cand((0, 0, 10, 5), .8, 0, 1, 1)
        kept, suppressed, stats = classwise_nms_with_suppressor([a, b], NMS_IOU)
        self.assertEqual([c.identity for c in kept], [a.identity, b.identity])
        self.assertEqual(suppressed, {})
        self.assertEqual(stats['boundary_within_eps'], 1)

    def test_equal_score_tie_breaks_by_view_query_category(self):
        a, b = cand((0, 0, 10, 10), .7, 1, 5, 2), cand((1, 0, 11, 10), .7, 0, 9, 2)
        kept, suppressed, _ = classwise_nms_with_suppressor([a, b], NMS_IOU)
        self.assertEqual([c.identity for c in kept], [b.identity])
        self.assertEqual(suppressed[a.identity]['suppressor'], b.identity)
        self.assertEqual([c.identity for c in classwise_nms([a, b])], [b.identity])

    def test_first_pivot_wins_no_chain(self):
        a = cand((0, 0, 10, 10), .9, 0, 0, 1)
        b = cand((0, 0, 10, 8), .8, 0, 1, 1)
        c = cand((0, 4, 10, 12), .7, 0, 2, 1)
        kept, suppressed, _ = classwise_nms_with_suppressor([a, b, c], NMS_IOU)
        self.assertEqual([x.identity for x in kept], [a.identity, c.identity])
        self.assertEqual(suppressed[b.identity]['suppressor'], a.identity)

    def test_duplicate_identity_rejected(self):
        a, b = cand((0, 0, 10, 10), .7, 1, 5, 2), cand((5, 5, 10, 10), .6, 1, 5, 2)
        with self.assertRaises(ValueError):
            classwise_nms_with_suppressor([a, b], NMS_IOU)


class AttributionTests(unittest.TestCase):
    def test_fate_order(self):
        m = {'top600': {1, 2, 3}, 'filtered600': {1, 2}, 'cap600': {1},
             'top300': set(), 'filtered300': set(), 'cap300': set()}
        self.assertEqual(fate(1, m, 600), 'in_final')
        self.assertEqual(fate(2, m, 600), 'class100_removed')
        self.assertEqual(fate(3, m, 600), 'ignore_removed')
        self.assertEqual(fate(4, m, 600), 'budget_removed')
        self.assertEqual(fate(1, m, 300), 'budget_removed')

    def test_argmax_ties_and_category(self):
        self.assertEqual(argmax_ties([.2, .7, .7]), ([1, 2], .7))
        self.assertEqual(argmax_ties([]), ([], None))
        self.assertEqual(category(True, .6), 'K1_mid')
        self.assertEqual(category(True, .4), 'K1_low')
        self.assertEqual(category(False, .6), 'K2')
        self.assertEqual(category(False, .4), 'K3')
        self.assertEqual(category(True, 0.), 'K3')
        self.assertEqual(category(False, None), 'K3')

    def test_shared_counts(self):
        counts = shared_counts({'g1': [(0, 1, 1), (1, 2, 1)], 'g2': [(0, 1, 1)]})
        self.assertEqual(counts[(0, 1, 1)], 2)
        self.assertEqual(counts[(1, 2, 1)], 1)

    @staticmethod
    def membership(*final):
        s = set(final)
        return {k: set(s) for k in ('top600', 'filtered600', 'cap600', 'top300', 'filtered300', 'cap300')}

    @staticmethod
    def entry(p, iou):
        return {'suppressor': p.identity, 'iou': iou, 'step': 0,
                'suppressor_box': p.box_xyxy, 'suppressor_score': p.score}

    def test_attribute_categories_and_fates(self):
        g1 = {'id': 'g1', 'box': [0., 0., 10., 10.]}
        g2 = {'id': 'g2', 'box': [12., 0., 22., 10.]}
        tau = .75
        c1 = cand((0, 0, 10, 9), .5, 1, 1, 1)
        c2 = cand((0, 0, 10, 8), .4, 2, 7, 1)
        p_mid = cand((0, 0, 10, 6), .6, 1, 3, 1)
        p_k2 = cand((12, 0, 22, 8), .7, 0, 2, 1)
        p_k3 = cand((30, 30, 40, 40), .8, 3, 0, 1)
        p_low = cand((0, 0, 10, 4), .9, 4, 1, 1)
        suppressed = {c1.identity: self.entry(p_mid, .667), c2.identity: self.entry(p_k2, .55)}
        membership = self.membership(p_mid.identity)
        rec = attribute_gt(g1, [c1, c2], suppressed, [g1, g2], tau, membership, ious)
        self.assertEqual(rec['lost_count'], 2)
        self.assertEqual(rec['categories'], ['K1_mid', 'K2'])
        self.assertTrue(rec['mixed'])
        self.assertEqual(rec['distinct_suppressors'], 2)
        self.assertEqual(rec['same_view_count'], 1)
        r1, r2 = rec['candidates']
        self.assertEqual(r1['category'], 'K1_mid')
        self.assertTrue(r1['same_view'])
        self.assertAlmostEqual(r1['suppressor_gt_iou'], .6)
        self.assertEqual(r1['suppressor_argmax_gt'], ['g1'])
        self.assertEqual(r1['suppressor_fate'], {'600': 'in_final', '300': 'in_final'})
        self.assertEqual(r2['category'], 'K2')
        self.assertEqual(r2['suppressor_argmax_gt'], ['g2'])
        self.assertEqual(r2['suppressor_fate']['600'], 'budget_removed')
        self.assertAlmostEqual(r2['candidate_gt_iou'], .8)
        for pivot, kind in ((p_k3, 'K3'), (p_low, 'K1_low')):
            rec = attribute_gt(g1, [c1], {c1.identity: self.entry(pivot, .6)}, [g1, g2], tau,
                               membership, ious)
            self.assertEqual(rec['candidates'][0]['category'], kind)
        g3 = {'id': 'g3', 'box': [0., 0., 10., 10.]}
        rec = attribute_gt(g1, [c1], {c1.identity: self.entry(p_mid, .6)}, [g1, g2, g3], tau,
                           membership, ious)
        self.assertTrue(rec['candidates'][0]['suppressor_argmax_tie'])
        self.assertEqual(rec['candidates'][0]['suppressor_argmax_gt'], ['g1', 'g3'])

    def test_attribute_preconditions(self):
        g1 = {'id': 'g1', 'box': [0., 0., 10., 10.]}
        g2 = {'id': 'g2', 'box': [12., 0., 22., 10.]}
        c1 = cand((0, 0, 10, 9), .5, 1, 1, 1)
        p_mid = cand((0, 0, 10, 6), .6, 1, 3, 1)
        p_hi = cand((0, 0, 10, 9.5), .95, 0, 0, 1)
        membership = self.membership(p_mid.identity)
        with self.assertRaises(ValueError):
            attribute_gt(g1, [c1], {}, [g1, g2], .75, membership, ious)
        with self.assertRaises(ValueError):
            attribute_gt(g1, [], {c1.identity: self.entry(p_mid, .6)}, [g1, g2], .75, membership, ious)
        with self.assertRaises(ValueError):
            attribute_gt(g1, [c1], {c1.identity: self.entry(p_hi, .9)}, [g1, g2], .75, membership, ious)
        with self.assertRaises(ValueError):
            attribute_gt(g1, [c1], {c1.identity: self.entry(p_mid, .6)}, [g2], .75, membership, ious)


class CliFixtureTests(unittest.TestCase):
    def test_fixture_closure_duplicate_and_failure(self):
        cli = ROOT / 'tools/nms_suppressor_audit.py'
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'success'
            cmd = [sys.executable, '-B', str(cli), 'fixture', '--output', str(output)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((output / 'CLOSED.json').is_file())
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in output.iterdir()})
            failed = Path(root) / 'failed'
            run = subprocess.run([sys.executable, '-B', str(cli), 'fixture', '--output', str(failed),
                                  '--inject-failure'], capture_output=True)
            self.assertEqual(run.returncode, 2)
            self.assertTrue((failed / 'STOP.json').is_file())
            self.assertFalse((failed / 'RESULT.json').exists())
            bad = subprocess.run([sys.executable, '-B', str(cli), 'run',
                                  '--output', str(Path(root) / 'x')], capture_output=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
