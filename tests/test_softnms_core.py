"""Independent analytic examples; no detector, network, GT cache or GPU."""
import dataclasses
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from slice_core import Candidate, box_iou64
from softnms_core import gaussian_soft_nms, compare_scores, LIMITS, METRICS
from softnms_job import Journal, digest, read, PLAN


def c(q,score,box=(0.,0.,10.,10.),view=0,cat=1):
    return Candidate(box,score,view,q,cat,1)


def scalar_oracle(items):
    # Literal scalar algorithm independent of vectorized IoU and argmax.
    pending=list(items);result=[]
    while pending:
        pending.sort(key=lambda x:x.sort_key)
        best=pending.pop(0);result.append(best)
        pending=[dataclasses.replace(v,score=v.score*math.exp(-box_iou64(v.box_xyxy,best.box_xyxy)**2/.5))
                 if v.category_id==best.category_id else v for v in pending]
    return sorted(result,key=lambda x:x.sort_key)


class FixedSoftNMS(unittest.TestCase):
    def test_dynamic_reordering(self):
        rows=[c(0,.9),c(1,.8),c(2,.7,(20,20,30,30))]
        actual=gaussian_soft_nms(rows)['all']
        self.assertEqual([v.query_id for v in actual],[0,2,1])
        self.assertAlmostEqual(actual[-1].score,.8*math.exp(-2),places=16)

    def test_cumulative_decay(self):
        rows=[c(0,.9),c(1,.8),c(2,.7)]
        actual=gaussian_soft_nms(rows)['all']
        self.assertAlmostEqual(actual[2].score,.7*math.exp(-2)*math.exp(-2),places=16)

    def test_each_iteration_tie_identity(self):
        rows=[c(8,.4),c(1,.4),c(0,.4,view=1)]
        actual=gaussian_soft_nms(rows)['all']
        self.assertEqual([v.identity for v in actual],[(0,1,1),(0,8,1),(1,0,1)])

    def test_class_isolation(self):
        actual=gaussian_soft_nms([c(0,.9),c(1,.8,cat=2)])['all']
        self.assertEqual([v.score for v in actual],[.9,.8])

    def test_zero_retained(self):
        rows=[c(2,0.),c(0,0.),c(1,0.)]
        r=gaussian_soft_nms(rows)
        self.assertEqual([v.query_id for v in r['all']],[0,1,2])
        self.assertEqual(r['stats']['input_zero_scores'],3)
        self.assertEqual(r['stats']['underflow_to_zero'],0)

    def test_actual_subnormal_underflow(self):
        r=gaussian_soft_nms([c(0,1.),c(1,float.fromhex('0x0.0000000000001p-1022'))])
        self.assertEqual(len(r['all']),2)
        self.assertEqual(r['all'][1].score,0.)
        self.assertEqual(r['stats']['underflow_to_zero'],1)

    def test_cumulative_underflow_no_cutoff(self):
        rows=[c(q,1.,view=v) for v in range(2) for q in range(200)]
        r=gaussian_soft_nms(rows)
        self.assertEqual(len(r['all']),400)
        self.assertGreater(r['stats']['underflow_to_zero'],0)
        self.assertGreater(r['stats']['subnormal_scores'],0)

    def test_prefix_more_than_600(self):
        rows=[c(q,.5,(q*20+v*7000,0,q*20+v*7000+10,10),view=v) for v in range(3) for q in range(300)]
        r=gaussian_soft_nms(reversed(rows))
        self.assertEqual(len(r['top600']),600)
        self.assertEqual(r['top300'],r['top600'][:300])
        self.assertEqual(r['stats']['candidate_count'],900)

    def test_geometry_is_not_changed(self):
        rows=[c(1,.8,(-5,-5,15,15)),c(0,.9)]
        lookup={v.identity:v.box_xyxy for v in rows}
        self.assertTrue(all(v.box_xyxy==lookup[v.identity] for v in gaussian_soft_nms(rows)['all']))

    def test_no_hard_threshold_or_pixel_plus_one(self):
        # IoU 1/3: Gaussian must decay even below 0.5.
        rows=[c(0,.9),c(1,.8,(5,0,15,10))]
        actual=gaussian_soft_nms(rows)['all'][1].score
        self.assertAlmostEqual(actual,.8*math.exp(-2/9),places=15)

    def test_scalar_oracle(self):
        rng=random.Random(1701)
        rows=[c(q,rng.random(),(q%5,q%7,q%5+10,q%7+8),cat=q%3+1) for q in range(40)]
        a=gaussian_soft_nms(rows)['all'];b=scalar_oracle(rows)
        self.assertEqual([v.identity for v in a],[v.identity for v in b])
        for x,y in zip(a,b):self.assertAlmostEqual(x.score,y.score,places=14)

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError):gaussian_soft_nms([c(0,.9),c(0,.8)])

    def test_bad_score_and_box_rejected(self):
        for score in [float('nan'),float('inf'),-.1,1.01]:
            with self.assertRaises(ValueError):c(0,score)
        for box in [(0,0,0,10),(2,0,1,1),(0,0,float('inf'),1)]:
            with self.assertRaises(ValueError):c(0,.9,box)

    def test_empty(self):
        self.assertEqual(gaussian_soft_nms([])['all'],[])

    def test_plan_matches_frozen_gates(self):
        p=read(PLAN)
        self.assertEqual(p['gates_pp'],LIMITS)
        self.assertEqual(p['sigma'],.5)
        self.assertEqual(p['seeds'],[0,1,2])

    def test_single_pp_conversion_and_fail_closed(self):
        scores={a:{b:{m:.25 for m in METRICS} for b in ('600','300')} for a in ('A','B','C','D','E')}
        r=compare_scores(scores);self.assertFalse(r['quality_passed'])
        scores['D']['600']['AP']=.251
        self.assertAlmostEqual(compare_scores(scores)['differences_pp']['600']['D-C']['AP'],.1)
        scores['D']['600']['APS']=float('nan')
        with self.assertRaises(ValueError):compare_scores(scores)

    def test_journal_missing_reordered_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            j=Journal(directory,'fixture')
            with self.assertRaises(ValueError):j.emit('parents_closed')
            j.emit('started')
            with self.assertRaises(ValueError):j.emit('started')
            with self.assertRaises(ValueError):Journal.validate(j.path,'fixture',complete=True)
            with self.assertRaises(ValueError):Journal.validate(j.path,'wrong')


if __name__=='__main__':unittest.main()
