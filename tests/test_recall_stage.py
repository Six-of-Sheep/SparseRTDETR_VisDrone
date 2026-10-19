import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from recall_stage_core import iou_matrix, pair_match, maximum_match, transition
sys.path.insert(0, str(ROOT))
from slice_core import Candidate, classwise_nms
from recall_stage_core import cap
from recall_stage_run import analyze_set


class CoreTests(unittest.TestCase):
    def test_exact_threshold_and_later_GT_tie(self):
        self.assertEqual(pair_match([[.5, .5]], [False]*2, [False]*2, .5).tolist(), [-1, 0])
        self.assertEqual(pair_match([[.749999]], [False], [False], .75).tolist(), [-1])

    def test_ignored_priority_and_crowd_reuse(self):
        self.assertEqual(pair_match([[.7, 1], [.8, 1]], [False, True], [False, True], .5).tolist(), [0, 1])
        self.assertEqual(pair_match([[1], [1]], [True], [True], .5).tolist(), [1])

    def test_crowd_denominator(self):
        self.assertAlmostEqual(iou_matrix([[0,0,1,1]], [[0,0,10,10]])[0,0], .01)
        self.assertEqual(iou_matrix([[0,0,1,1]], [[0,0,10,10]], [True])[0,0], 1.)

    def test_greedy_reassignment_under_deletion(self):
        v = np.array([[.8,.9], [.1,.8], [.8,.1]])
        self.assertEqual(pair_match(v, [False]*2, [False]*2, .75).tolist(), [2,0])
        self.assertEqual(pair_match(v[1:], [False]*2, [False]*2, .75).tolist(), [1,0])
        # Same available geometry; greedy can miss a match that maximum matching recovers.
        v = np.array([[.9,.8], [.8,.1]])
        self.assertEqual(sum(pair_match(v,[False]*2,[False]*2,.75)>=0), 1)
        self.assertEqual(maximum_match(v>=.75), 2)

    def test_geometry_and_matching_are_not_same(self):
        v = iou_matrix([[0,0,10,10]], [[0,0,10,10], [0,0,10,10]])
        self.assertEqual(int(np.count_nonzero(v.max(axis=0)>=.75)), 2)
        self.assertEqual(int(np.count_nonzero(pair_match(v,[False]*2,[False]*2,.75)>=0)), 1)

    def test_invalid_and_empty(self):
        self.assertEqual(pair_match(np.empty((0,2)),[False]*2,[False]*2,.5).tolist(), [-1,-1])
        self.assertEqual(maximum_match(np.empty((0,2),bool)), 0)
        for box in ([0,0,0,1], [0,0,float('nan'),1], [0,0,-1,1]):
            with self.assertRaises(ValueError): iou_matrix([box],[[0,0,1,1]])
        with self.assertRaises(ValueError): pair_match([[1.1]],[False],[False],.5)
        self.assertEqual(transition([True,False], [False,True]),dict(retained=0,lost=1,gained=1,neither=0))

    def test_actual_CLI_close_failure_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'ok'
            cmd = [sys.executable,'-B',str(ROOT/'tools/recall_stage_job.py'),'fixture','--output',str(out)]
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode, 0)
            self.assertTrue((out/'CLOSED.json').is_file())
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode, 0)
            cmd[-1] = str(Path(d)/'fail')
            self.assertEqual(subprocess.run(cmd+['--inject-failure'],capture_output=True).returncode,2)
            self.assertFalse((Path(d)/'fail/CLOSED.json').exists())
            self.assertFalse(json.loads((Path(d)/'fail/STOP.json').read_text())['automatic_retry'])

    def test_frozen_COCO_matching_parity(self):
        # Required dependency on the server; no model or full-set scoring.
        from pycocotools.cocoeval import COCOeval
        rng = np.random.RandomState(48)
        for crowd_bits in itertools.product((False,True), repeat=2):
            for ignore_bits in itertools.product((False,True), repeat=2):
                v = rng.uniform(0,1,(7,2)); v[0] = [.75,.75]
                for t in (.5,.75):
                    e = COCOeval()
                    e.params.iouThrs = np.array([t]); e.params.useCats = True
                    gt = [dict(id=i+1,ignore=int(ignore_bits[i]),iscrowd=int(crowd_bits[i]),area=10) for i in range(2)]
                    dt = [dict(id=i+1,score=1-i/10,area=10) for i in range(7)]
                    e._gts[(1,1)], e._dts[(1,1)], e.ious[(1,1)] = gt, dt, v
                    result = e.evaluateImg(1,1,[0,100],100)
                    want = {g:int(d)-1 for g,d in zip(result['gtIds'],result['gtMatches'][0])}
                    actual = pair_match(v, ignore_bits, crowd_bits,t)
                    self.assertEqual(actual.tolist(), [want[1],want[2]])

    def test_pipeline_classes_area_and_cardinality(self):
        gt=[dict(bbox=[0,0,10,10],category_id=1,iscrowd=0,area=100),
            dict(bbox=[0,0,10,10],category_id=2,iscrowd=0,area=100),
            dict(bbox=[0,0,100,100],category_id=1,iscrowd=0,area=10000)]
        ps=[Candidate((0,0,10,10),.9,0,0,1,1),Candidate((0,0,10,10),.8,0,1,1,2)]
        s=analyze_set(ps,gt,'small',.75,upper_bound=True)
        self.assertEqual(s['geometry'].tolist(),[True,True,False])
        self.assertEqual(s['correct_geometry'].tolist(),[True,False,False])
        self.assertEqual(s['correct_index'].tolist(),[0,-1,0])
        self.assertEqual(s['match'].tolist(),[True,False,False])
        self.assertEqual(s['eligible'].tolist(),[True,True,False])
        self.assertEqual(len(classwise_nms(ps)),1)
        self.assertEqual(analyze_set([],gt,'large',.5)['match'].tolist(),[False]*3)
        self.assertEqual(analyze_set(ps,[],'large',.5)['eligible'].tolist(),[])

    def test_factorial_replays_order_budget_and_classcap(self):
        ps=[Candidate((i*20.,0.,i*20.+10.,10.),1-i/1000,0,i,1,i+1) for i in range(150)]
        self.assertEqual(len(cap(ps)),100)
        self.assertEqual(len(cap(ps,False)),150)
        # NMS always returns score/identity order; no independent sorting deletion.
        self.assertEqual(classwise_nms(list(reversed(ps))),ps)
        region_filter=lambda cs:[p for p in cs if p.query_id>=5]
        self.assertEqual(len(region_filter(ps[:6])),1)
        self.assertEqual(len(region_filter(ps)[:6]),6)


if __name__ == '__main__': unittest.main()
